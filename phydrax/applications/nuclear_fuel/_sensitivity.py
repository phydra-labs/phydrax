"""Forward implicit tangents on a frozen grid of accepted steps."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import assert_never, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._tree_math import tree_zeros_like
from ..._validation import canonical_identifier
from ...linalg import (
    FunctionLinearOperator,
    prepare_linearization,
    PreparedLinearization,
    RHSLayout,
    solve as solve_linear,
)
from ...observation import CoordinateLayout
from ...typing import as_array, Dim, Float64, parse, Scope, Size
from ._acceptance import fuel_step_evidence, FuelAcceptancePolicy, FuelStepEvidence
from ._bundles import FuelModelBundle
from ._problem import PreparedFuelRod
from ._state import FuelRodState


class FuelParameterDim(Dim):
    """Source-law or initial-history parameter coordinates."""


@dataclass(frozen=True, slots=True)
class FuelFixedGrid:
    """Frozen accepted-step times; replay does not differentiate adaptivity or retries."""

    times: tuple[float, ...]
    prepared_id: str
    grid_id: str

    def __init__(self, times: Sequence[float], prepared_id: str, /) -> None:
        grid_times = np.asarray(times, dtype=np.float64)
        if (
            grid_times.ndim != 1
            or grid_times.size < 2
            or np.any(~np.isfinite(grid_times))
            or np.any(np.diff(grid_times) <= 0)
        ):
            raise ValueError(
                "Frozen grid requires at least two finite increasing accepted times."
            )
        identifier = canonical_identifier(prepared_id, "prepared_id")
        canonical_times = tuple(float(time) for time in grid_times)
        object.__setattr__(self, "times", canonical_times)
        object.__setattr__(self, "prepared_id", identifier)
        object.__setattr__(
            self,
            "grid_id",
            canonical_fingerprint(
                {
                    "kind": "fuel-frozen-step-grid",
                    "times": canonical_times,
                    "prepared_id": identifier,
                    "adaptive_grid_derivative": False,
                }
            ),
        )

    def validate_states(
        self, states: tuple[FuelRodState, ...], /
    ) -> tuple[FuelRodState, ...]:
        if len(states) != len(self.times):
            raise ValueError("Replay states must cover the entire frozen step grid.")
        checked: list[FuelRodState] = []
        for state, time in zip(states, self.times, strict=True):
            if state.prepared_id != self.prepared_id:
                raise ValueError("Replay state belongs to another prepared rod.")
            checked_time = eqx.error_if(
                state.time,
                state.time != time,
                "Replay state time differs from the frozen accepted-step grid.",
            )
            checked.append(eqx.tree_at(lambda record: record.time, state, checked_time))
        return tuple(checked)


class FuelParameterization(StrictModule):
    """Dynamic physical coordinates and their source-bound bundle update.

    The update replaces existing dynamic law leaves by immutable tree
    replacement; geometry, source IDs, and static method identity must not
    change with the values. Coordinate labels and units are given explicitly,
    never inferred from tree ordering.
    """

    layout: CoordinateLayout
    units: tuple[str, ...] = eqx.field(static=True)
    update: Callable[[FuelModelBundle, Array], FuelModelBundle] = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: CoordinateLayout,
        units: tuple[str, ...],
        update: Callable[[FuelModelBundle, Array], FuelModelBundle],
        /,
        *,
        parameterization_id: str,
    ) -> None:
        if len(units) != layout.size or any(not unit for unit in units):
            raise ValueError(
                "Every physical parameter needs a nonempty SI or source unit."
            )
        if not callable(update):
            raise TypeError("Bundle parameter update must be callable.")
        self.layout = layout
        self.units = units
        self.update = update
        self.parameterization_id = canonical_identifier(
            parameterization_id, "parameterization_id"
        )

    def bundle(self, base: FuelModelBundle, values: ArrayLike, /) -> FuelModelBundle:
        scope = Scope()
        parse(self.layout.size, Size[FuelParameterDim], "parameter_count", scope=scope)
        parameters = as_array(
            values, Float64[FuelParameterDim], "physical_parameters", scope=scope
        )
        result = self.update(base, parameters)
        if not isinstance(result, FuelModelBundle):
            raise TypeError("Parameterization update must return a FuelModelBundle.")
        if jax.tree_util.tree_structure(result) != jax.tree_util.tree_structure(base):
            raise ValueError(
                "Parameter updates must not change the bundle tree structure or static source identity."
            )
        return result


class FuelReplayResult(StrictModule):
    """Fixed-grid replay states with each step's evidence and branch-change flags."""

    states: tuple[FuelRodState, ...]
    nonlinear_success: Array
    branch_changed: Array
    attempts: tuple[FuelStepEvidence, ...]
    successful: Array
    grid_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def stack_trees[T: StrictModule](trees: tuple[T, ...], /) -> T:
    """Stack homogeneous PyTrees along a new leading axis; static fields are kept."""
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees)


def unstack_tree[T: StrictModule](tree: T, count: int, /) -> tuple[T, ...]:
    """Split a stacked PyTree into ``count`` records along its leading axis.

    ``count`` is the frozen grid length or the capped coordinate count, so the
    unrolled output is bounded and charged in the tangent working-set estimate.
    """
    return tuple(jax.tree.map(lambda leaf: leaf[index], tree) for index in range(count))


def array_tree_bytes(tree: StrictModule, /) -> int:
    """Bytes of the array leaves of ``tree``; static fields are not counted."""
    return sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree_util.tree_leaves(tree)
        if eqx.is_array(leaf)
    )


def replay_fuel_grid(
    prepared: PreparedFuelRod,
    grid: FuelFixedGrid,
    /,
    *,
    parameters: FuelModelBundle | None = None,
    initial_state: FuelRodState | None = None,
    fill_pressure_scale: ArrayLike = 1.0,
    acceptance_policy: FuelAcceptancePolicy | None = None,
) -> FuelReplayResult:
    """Solve and check every fixed-grid transition, without adaptivity or retries.

    Returned candidate states are usable only when ``successful`` is true; every
    physical, domain, and balance failure is reported in ``attempts``.
    """
    if prepared.prepared_id != grid.prepared_id:
        raise ValueError("Frozen replay grid belongs to another prepared rod.")
    bundle = prepared.admission.bundle if parameters is None else parameters
    state = (
        prepared.initial_state(parameters=bundle, fill_pressure_scale=fill_pressure_scale)
        if initial_state is None
        else initial_state
    )
    if state.prepared_id != prepared.prepared_id:
        raise ValueError("Replay initial state belongs to another prepared rod.")
    checked_time = eqx.error_if(
        state.time,
        state.time != grid.times[0],
        "Replay initial time must match the grid start.",
    )
    state = eqx.tree_at(lambda record: record.time, state, checked_time)
    times = jnp.asarray(grid.times, dtype=state.time.dtype)
    intervals = (
        times[:-1],
        jnp.diff(times),
        jnp.arange(len(grid.times) - 1, dtype=jnp.int32),
    )

    def advance(
        previous: FuelRodState, interval: tuple[Array, Array, Array]
    ) -> tuple[FuelRodState, tuple[FuelRodState, Array, Array, FuelStepEvidence]]:
        time, dt, step_index = interval
        solved = prepared.solve_candidate(previous, time, dt, parameters=bundle)
        candidate = prepared.candidate_evaluation(solved).candidate
        evidence = fuel_step_evidence(
            prepared,
            solved,
            previous,
            time,
            dt,
            step_index,
            parameters=bundle,
            policy=acceptance_policy,
        )
        branch_changed = jnp.any(
            prepared.branch_signature(candidate, parameters=bundle)
            != prepared.branch_signature(previous, parameters=bundle)
        )
        nonlinear_success = (
            solved.thermomechanical_result.successful & solved.coupling_result.successful
        )
        return candidate, (candidate, nonlinear_success, branch_changed, evidence)

    step_count = len(grid.times) - 1
    _, (candidates, nonlinear_success, branch_changed, evidence) = jax.lax.scan(
        advance, state, intervals
    )
    return FuelReplayResult(
        states=(state,) + unstack_tree(candidates, step_count),
        nonlinear_success=nonlinear_success,
        branch_changed=branch_changed,
        attempts=unstack_tree(evidence, step_count),
        successful=jnp.all(nonlinear_success) & jnp.all(evidence.accepted),
        grid_id=grid.grid_id,
        prepared_id=prepared.prepared_id,
    )


class FuelSensitivityResult(StrictModule):
    """Full history-state tangents with linear-solve and branch-change evidence."""

    tangents: tuple[tuple[FuelRodState, ...], ...]
    linear_success: Array
    linear_status: Array
    branch_changed: Array
    branchwise_valid: Array
    successful: Array
    parameter_layout: CoordinateLayout
    parameter_values: Array
    base_states: tuple[FuelRodState, ...]
    units: tuple[str, ...] = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    working_set_bytes: int = eqx.field(static=True)


# Upper bound on forward tangent coordinates: every coordinate carries a full
# state-history tangent and one right-hand side of every step solve.
MAXIMUM_FORWARD_PARAMETER_COUNT = 64


def _tangent_working_set_bytes(
    initial_state: FuelRodState,
    joint_unknown: Array,
    parameters: Array,
    /,
    *,
    coordinate_count: int,
    time_count: int,
) -> int:
    """Bytes retained by forward tangents over a frozen grid of ``time_count`` times."""
    state_bytes = array_tree_bytes(initial_state)
    unknown_itemsize = joint_unknown.dtype.itemsize
    factorization_bytes = joint_unknown.size**2 * unknown_itemsize
    # Per grid time: supplied primal history, stacked scan input, scanned
    # tangent output, and unstacked public tangents.
    history_state_count = (2 * coordinate_count + 2) * time_count
    # Batched step workspace and the fabrication/history/coordinate
    # zero-tangent buffers.
    workspace_state_count = 3 * (coordinate_count + 1) + 2
    # Right-hand-side and solution columns plus one zero unknown.
    solve_bytes = (
        2 * joint_unknown.size * coordinate_count + joint_unknown.size
    ) * unknown_itemsize
    # Direction basis plus two coordinate vectors.
    coordinate_bytes = (
        coordinate_count**2 + 2 * coordinate_count
    ) * parameters.dtype.itemsize
    return (
        factorization_bytes
        + state_bytes * (history_state_count + workspace_state_count)
        + solve_bytes
        + coordinate_bytes
    )


class _StepPushforward(StrictModule):
    """Tangent of one frozen step's joint residual and full candidate state.

    Inputs are tangents of the candidate joint unknown, the accepted history
    state, and the parameter coordinates.
    """

    linearization: PreparedLinearization
    zero_unknown: Array
    zero_history: FuelRodState
    zero_parameters: Array

    def __call__(
        self,
        unknown_tangent: Array,
        history_tangent: FuelRodState,
        parameter_tangent: Array,
        /,
    ) -> tuple[Array, FuelRodState]:
        # Equinox marks symbolic zero directions with None; the linearization
        # requires explicit floating leaves and keeps static None leaves.
        tangent = eqx.combine(
            (unknown_tangent, history_tangent, parameter_tangent),
            (self.zero_unknown, self.zero_history, self.zero_parameters),
        )
        output = self.linearization.jvp(tangent)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError(
                "Joint step pushforward must return residual and candidate-state tangents."
            )
        residual_tangent, state_tangent = output
        if not isinstance(residual_tangent, Array) or not isinstance(
            state_tangent, FuelRodState
        ):
            raise TypeError(
                "Joint step pushforward must return an Array residual tangent and a FuelRodState tangent."
            )
        return residual_tangent, state_tangent


class FuelSensitivityPlan(StrictModule):
    """Forward tangents of a fixed-grid history with respect to bundle coordinates.

    Construction refuses a parameter count above ``parameter_capacity`` and an
    estimated tangent working set above ``maximum_working_set_bytes``.
    """

    prepared: PreparedFuelRod
    grid: FuelFixedGrid = eqx.field(static=True)
    parameterization: FuelParameterization
    parameter_values: Array
    parameter_capacity: int = eqx.field(static=True)
    maximum_working_set_bytes: int = eqx.field(static=True)
    working_set_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedFuelRod,
        grid: FuelFixedGrid,
        parameterization: FuelParameterization,
        values: ArrayLike,
        /,
        *,
        parameter_capacity: int,
        maximum_working_set_bytes: int,
    ) -> None:
        coordinate_count = parameterization.layout.size
        parameters = as_array(values, Float64[FuelParameterDim], "parameter_values")
        if grid.prepared_id != prepared.prepared_id or parameters.shape != (
            coordinate_count,
        ):
            raise ValueError(
                "Sensitivity grid and parameter values must match the prepared rod and coordinate layout."
            )
        if (
            not 1
            <= coordinate_count
            <= parameter_capacity
            <= MAXIMUM_FORWARD_PARAMETER_COUNT
        ):
            raise ValueError(
                "Forward sensitivity parameter capacity must satisfy "
                f"1 <= coordinate count <= capacity <= {MAXIMUM_FORWARD_PARAMETER_COUNT}."
            )
        if maximum_working_set_bytes <= 0:
            raise ValueError("Sensitivity working-set bound must be positive.")
        initial_state = prepared.initial_state(
            parameters=parameterization.bundle(prepared.admission.bundle, parameters)
        )
        working_set_bytes = _tangent_working_set_bytes(
            initial_state,
            prepared.pack_joint(initial_state),
            parameters,
            coordinate_count=coordinate_count,
            time_count=len(grid.times),
        )
        if working_set_bytes > maximum_working_set_bytes:
            raise ValueError(
                f"Fuel tangent working set {working_set_bytes} exceeds the bound of {maximum_working_set_bytes} bytes."
            )
        self.prepared = prepared
        self.grid = grid
        self.parameterization = parameterization
        self.parameter_values = parameters
        self.parameter_capacity = parameter_capacity
        self.maximum_working_set_bytes = maximum_working_set_bytes
        self.working_set_bytes = working_set_bytes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fuel-forward-implicit-tangents",
                "prepared": prepared.prepared_id,
                "grid": grid.grid_id,
                "coordinates": parameterization.layout.layout_id,
                "parameterization": parameterization.parameterization_id,
                "capacity": parameter_capacity,
                "working_set_bound": maximum_working_set_bytes,
            }
        )

    def evaluate(
        self,
        states: tuple[FuelRodState, ...],
        /,
        *,
        initial_tangents: tuple[FuelRodState, ...] | None = None,
    ) -> FuelSensitivityResult:
        """Propagate parameter tangents along the accepted history on the grid.

        ``initial_tangents`` gives one full initial-state tangent per coordinate.
        Without it the grid must start at time zero and the fabrication
        derivative of the initial state is used.
        """
        states = self.grid.validate_states(states)
        coordinate_count = self.parameterization.layout.size
        bundle = self.parameterization.bundle(
            self.prepared.admission.bundle, self.parameter_values
        )
        direction_basis = jnp.eye(coordinate_count, dtype=self.parameter_values.dtype)
        zero_initial = tree_zeros_like(eqx.filter(states[0], eqx.is_inexact_array))
        initial_history_tangents = self._initial_tangents(
            initial_tangents, direction_basis, zero_initial
        )
        times = jnp.asarray(self.grid.times, dtype=states[0].time.dtype)
        candidate_states = stack_trees(states[1:])

        def advance(
            carry: tuple[FuelRodState, FuelRodState], interval: tuple[FuelRodState, Array]
        ) -> tuple[
            tuple[FuelRodState, FuelRodState], tuple[FuelRodState, Array, Array, Array]
        ]:
            previous, previous_tangents = carry
            candidate, dt = interval
            candidate_tangents, linear_success, linear_status = self._step_tangents(
                previous, previous_tangents, candidate, dt, direction_basis
            )
            branch_changed = jnp.any(
                self.prepared.branch_signature(previous, parameters=bundle)
                != self.prepared.branch_signature(candidate, parameters=bundle)
            )
            return (candidate, candidate_tangents), (
                candidate_tangents,
                linear_success,
                linear_status,
                branch_changed,
            )

        _, (propagated, linear_success, linear_status, branch_changed) = jax.lax.scan(
            advance,
            (states[0], initial_history_tangents),
            (candidate_states, jnp.diff(times)),
        )
        history = (initial_history_tangents,) + unstack_tree(propagated, len(states) - 1)
        return FuelSensitivityResult(
            tangents=tuple(unstack_tree(step, coordinate_count) for step in history),
            linear_success=linear_success,
            linear_status=linear_status,
            branch_changed=branch_changed,
            branchwise_valid=linear_success & ~branch_changed,
            successful=jnp.all(linear_success),
            parameter_layout=self.parameterization.layout,
            parameter_values=self.parameter_values,
            base_states=states,
            units=self.parameterization.units,
            grid_id=self.grid.grid_id,
            plan_id=self.plan_id,
            working_set_bytes=self.working_set_bytes,
        )

    def _initial_tangents(
        self,
        initial_tangents: tuple[FuelRodState, ...] | None,
        direction_basis: Array,
        zero_initial: FuelRodState,
        /,
    ) -> FuelRodState:
        """Stacked initial-history tangents, one per coordinate direction."""
        if initial_tangents is None:
            if self.grid.times[0] != 0.0:
                raise ValueError(
                    "Restart sensitivities require explicit initial-history tangents; fabrication derivatives apply only at time zero."
                )
            return self._fabrication_tangents(direction_basis, zero_initial)
        if len(initial_tangents) != self.parameterization.layout.size:
            raise ValueError(
                "Initial history tangents require one full state per coordinate."
            )
        return stack_trees(
            tuple(eqx.combine(tangent, zero_initial) for tangent in initial_tangents)
        )

    def _fabrication_tangents(
        self, direction_basis: Array, zero_initial: FuelRodState, /
    ) -> FuelRodState:
        """Derivative of the fabricated initial state along every direction."""

        def fabricate(parameters: Array) -> FuelRodState:
            return self.prepared.initial_state(
                parameters=self.parameterization.bundle(
                    self.prepared.admission.bundle, parameters
                )
            )

        def differentiate(direction: Array) -> FuelRodState:
            tangent = eqx.filter_jvp(fabricate, (self.parameter_values,), (direction,))[1]
            return eqx.combine(tangent, zero_initial)

        return eqx.filter_vmap(differentiate, in_axes=0)(direction_basis)

    def _linearize_step(
        self, previous: FuelRodState, candidate: FuelRodState, dt: Array, /
    ) -> _StepPushforward:
        """Linearize one frozen step at its accepted endpoint.

        Gas is advanced from the accepted history at the joint field/stress
        point, so one stored linearization owns the residual and candidate-state
        derivatives, including the gas-pressure/hydrostatic-stress feedback.
        """
        time = previous.time
        joint_unknown = self.prepared.pack_joint(candidate)
        point, nonfloating = eqx.partition(
            (joint_unknown, previous, self.parameter_values), eqx.is_inexact_array
        )

        def residual_and_candidate(
            floating: tuple[Array, FuelRodState, Array],
        ) -> tuple[Array, FuelRodState]:
            unknown, history, parameters = eqx.combine(floating, nonfloating)
            residual, evaluation = self.prepared.kernel.joint_residual_with_aux(
                unknown,
                (
                    history,
                    time,
                    dt,
                    self.parameterization.bundle(
                        self.prepared.admission.bundle, parameters
                    ),
                ),
            )
            return residual, eqx.filter(evaluation.candidate, eqx.is_inexact_array)

        linearization = prepare_linearization(
            residual_and_candidate,
            point,
            policy=self.prepared.nonlinear_template.method.jacobian_policy.linearization_policy,
            linearization_id=f"{self.plan_id}:joint-candidate",
        )
        return _StepPushforward(
            linearization=linearization,
            zero_unknown=jnp.zeros_like(joint_unknown),
            zero_history=tree_zeros_like(point[1]),
            zero_parameters=jnp.zeros_like(self.parameter_values),
        )

    def _step_tangents(
        self,
        previous: FuelRodState,
        previous_tangents: FuelRodState,
        candidate: FuelRodState,
        dt: Array,
        direction_basis: Array,
        /,
    ) -> tuple[FuelRodState, Array, Array]:
        """Candidate-state tangents of one step and the status of their solve.

        Solves ``J dx = -(dR/dhistory dhistory + dR/dparameters e)`` for every
        direction ``e`` and maps ``(dx, dhistory, e)`` to the candidate tangent.
        """
        coordinate_count = self.parameterization.layout.size
        pushforward = self._linearize_step(previous, candidate, dt)

        def jacobian_action(unknown_direction: Array) -> Array:
            return pushforward(
                unknown_direction, pushforward.zero_history, pushforward.zero_parameters
            )[0]

        jacobian = FunctionLinearOperator(
            jacobian_action,
            source=self.prepared.kernel.joint_space,
            target=self.prepared.kernel.joint_space,
            operator_id=f"{self.plan_id}:joint-jacobian",
        )
        linear_solve = self.prepared.prepare_joint_solve(
            jacobian, rhs_count=coordinate_count
        )

        def history_residual_tangent(
            history_tangent: FuelRodState, direction: Array
        ) -> Array:
            return pushforward(pushforward.zero_unknown, history_tangent, direction)[0]

        rhs = -eqx.filter_vmap(history_residual_tangent, in_axes=(0, 0))(
            previous_tangents, direction_basis
        ).T
        solved = solve_linear(
            linear_solve, rhs, rhs_layout=RHSLayout((coordinate_count,))
        )

        def candidate_tangent(
            unknown_tangent: Array, history_tangent: FuelRodState, direction: Array
        ) -> FuelRodState:
            return pushforward(unknown_tangent, history_tangent, direction)[1]

        candidate_tangents = eqx.filter_vmap(candidate_tangent, in_axes=(0, 0, 0))(
            solved.value.T, previous_tangents, direction_basis
        )
        return candidate_tangents, jnp.all(solved.successful), solved.status


type FuelParameterName = Literal[
    "fuel-phonon-resistivity",
    "fuel-phonon-temperature-coefficient",
    "fuel-electronic-coefficient",
    "fuel-electronic-temperature",
    "fuel-porosity-normalization",
    "fuel-pore-shape-coefficient",
    "gas-intrinsic-diffusivity-prefactor",
    "gas-intrinsic-activation-energy",
]


def _coordinate(
    name: FuelParameterName, /
) -> tuple[Callable[[FuelModelBundle], Array], str]:
    match name:
        case "fuel-phonon-resistivity":
            return lambda bundle: bundle.fuel_conductivity.phonon_constant, "m*K/W"
        case "fuel-phonon-temperature-coefficient":
            return (
                lambda bundle: bundle.fuel_conductivity.phonon_temperature_coefficient,
                "m/W",
            )
        case "fuel-electronic-coefficient":
            return lambda bundle: bundle.fuel_conductivity.electronic_coefficient, "W*K/m"
        case "fuel-electronic-temperature":
            return lambda bundle: bundle.fuel_conductivity.electronic_temperature, "K"
        case "fuel-porosity-normalization":
            return lambda bundle: bundle.fuel_conductivity.porosity_normalization, "1"
        case "fuel-pore-shape-coefficient":
            return lambda bundle: bundle.fuel_conductivity.pore_shape_coefficient, "1"
        case "gas-intrinsic-diffusivity-prefactor":
            return (
                lambda bundle: bundle.fission_gas.diffusivity.intrinsic_prefactor,
                "m^2/s",
            )
        case "gas-intrinsic-activation-energy":
            return (
                lambda bundle: bundle.fission_gas.diffusivity.intrinsic_activation_energy,
                "J",
            )
        case _:
            assert_never(name)


def _validate_coordinate(name: FuelParameterName, value: Array, /) -> Array:
    """Preserve source-constructor domains when replacing dynamic law leaves."""
    value = eqx.error_if(value, ~jnp.isfinite(value), f"{name} must be finite.")
    match name:
        case "fuel-phonon-resistivity" | "fuel-porosity-normalization":
            return eqx.error_if(value, value <= 0.0, f"{name} must be positive.")
        case "gas-intrinsic-diffusivity-prefactor" | "gas-intrinsic-activation-energy":
            return eqx.error_if(value, value < 0.0, f"{name} must be nonnegative.")
        case (
            "fuel-phonon-temperature-coefficient"
            | "fuel-electronic-coefficient"
            | "fuel-electronic-temperature"
            | "fuel-pore-shape-coefficient"
        ):
            return value
        case _:
            assert_never(name)


class _FuelBundleCoordinateUpdate(StrictModule):
    names: tuple[FuelParameterName, ...] = eqx.field(static=True)

    def __call__(self, bundle: FuelModelBundle, values: Array, /) -> FuelModelBundle:
        for index, name in enumerate(self.names):
            select, _ = _coordinate(name)
            bundle = eqx.tree_at(
                select, bundle, _validate_coordinate(name, values[index])
            )
        return bundle


def fuel_parameterization(
    bundle: FuelModelBundle, names: tuple[FuelParameterName, ...], /
) -> tuple[FuelParameterization, Array]:
    """Bind explicitly named existing law scalars; return their nominal values."""
    selected = tuple(
        parse(name, FuelParameterName, "fuel parameter name") for name in names
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(
            "Fuel physical parameter coordinates must be nonempty and unique."
        )
    coordinates = tuple(_coordinate(name) for name in selected)
    parameterization = FuelParameterization(
        CoordinateLayout(selected),
        tuple(unit for _, unit in coordinates),
        _FuelBundleCoordinateUpdate(selected),
        parameterization_id=canonical_fingerprint(
            {
                "kind": "fuel-source-law-coordinates",
                "names": selected,
                "bundle": bundle.bundle_id,
            }
        ),
    )
    return parameterization, jnp.stack(tuple(select(bundle) for select, _ in coordinates))
