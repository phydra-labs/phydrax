#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any, Literal, Protocol, runtime_checkable

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import HeldInputPolicy, TimeGrid
from ...linalg import prepare_real_coordinate_tree
from ...qualification import CapabilityProfile, SupportTuple
from ...solver import (
    DAEConsistencyPolicy,
    DAEEventPlan,
    DAEEventStatus,
    DAEInitializationSpec,
    DAEInitializationStatus,
    DAEResetMap,
    DAESolvePolicy,
    DAEStatus,
    DAETerminationStatus,
    DifferentialAlgebraicProblem,
    DifferentialProblem,
    DifferentialSolution,
    HybridGuardPlan,
    HybridSchedulePlan,
    initialize_dae,
    prepare_dae,
    ScheduledHybridGuard,
    solve_dae,
    solve_diffrax,
)
from ...solver._dae_events import empty_dae_event_result
from ._admission import validate_battery_execution_admission
from ._protocol import (
    BatteryProtocolPlan,
    BatteryProtocolValues,
    PASSIVE_TERMINAL_CONVENTION,
)
from ._qualification import validate_battery_candidate_profile
from ._results import (
    BatteryDAEReplayEvidence,
    BatteryDAERestartEvidence,
    BatteryDAESolution,
    BatteryExperimentResult,
    BatteryModelOutput,
    BatteryRunStatus,
    BatterySelectedOutputs,
    BatteryTermination,
)


@runtime_checkable
class AbstractBatteryModelAdapter(Protocol):
    """Minimal structural interface between battery orchestration and one real model."""

    model_id: str
    equation_form: Literal["ode", "dae"]
    observable_names: tuple[str, ...]
    observable_units: tuple[str, ...]

    def prepare(self, /) -> Any: ...

    def initial_state(
        self, prepared_model: Any, parameters: PyTree, initial_condition: PyTree, /
    ) -> PyTree: ...

    def problem(
        self,
        prepared_model: Any,
        initial_state: PyTree,
        runtime_inputs: "BatteryRuntimeInputs",
        /,
    ) -> DifferentialProblem | DifferentialAlgebraicProblem: ...

    def observe(
        self,
        prepared_model: Any,
        times_s: Array,
        states: PyTree,
        runtime_inputs: "BatteryRuntimeInputs",
        /,
    ) -> BatteryModelOutput: ...

    def ledger(
        self,
        prepared_model: Any,
        native_solution: Any,
        runtime_inputs: "BatteryRuntimeInputs",
        /,
    ) -> Any: ...


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _type_id(value: Any, /) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


class BatteryOutputPlan(StrictModule, NonTrainableState):
    """Static ordered selection of model observables and passive terminal current."""

    names: tuple[str, ...] = eqx.field(static=True)
    output_plan_id: str = eqx.field(static=True)

    def __init__(self, names: Sequence[str], /):
        resolved = tuple(_identifier(name, "Battery output name") for name in names)
        if not resolved:
            raise ValueError("BatteryOutputPlan requires at least one selected output.")
        if len(set(resolved)) != len(resolved):
            raise ValueError("Battery output names must be unique.")
        self.names = resolved
        self.output_plan_id = canonical_fingerprint(
            {"kind": "battery-output-selection", "names": list(resolved)}
        )


class BatteryDiffraxSolvePlan(StrictModule, NonTrainableState):
    """Immutable native Diffrax configuration for an ODE battery experiment."""

    solver: Any
    stepsize_controller: Any
    adjoint: Any | None
    dt0: float | None = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    event_relative_tolerance: float = eqx.field(static=True)
    event_absolute_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    solve_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        solver: Any | None = None,
        stepsize_controller: Any | None = None,
        adjoint: Any | None = None,
        dt0: float | None = None,
        relative_tolerance: float = 1.0e-6,
        absolute_tolerance: float = 1.0e-8,
        event_relative_tolerance: float = 1.0e-7,
        event_absolute_tolerance: float = 1.0e-9,
        maximum_steps: int = 4096,
    ):
        solver_ = dfx.Tsit5() if solver is None else solver
        if not isinstance(solver_, dfx.AbstractSolver):
            raise TypeError("Battery ODE solver must be a Diffrax AbstractSolver.")
        controller = (
            dfx.PIDController(rtol=relative_tolerance, atol=absolute_tolerance)
            if stepsize_controller is None
            else stepsize_controller
        )
        if isinstance(controller, dfx.ClipStepSizeController):
            raise ValueError(
                "Battery preparation owns transition clipping; provide the underlying controller."
            )
        if not isinstance(
            controller, (dfx.AbstractAdaptiveStepSizeController, dfx.StepTo)
        ):
            raise ValueError(
                "Battery ODE execution supports adaptive controllers or explicit diffrax.StepTo."
            )
        if dt0 is None:
            dt0_ = None
        else:
            dt0_ = float(dt0)
            if not isfinite(dt0_) or dt0_ <= 0.0:
                raise ValueError("Battery ODE dt0 must be finite and positive or None.")
        if isinstance(controller, dfx.StepTo) and dt0_ is not None:
            raise ValueError("diffrax.StepTo requires dt0=None.")
        tolerances = tuple(
            float(value)
            for value in (
                relative_tolerance,
                absolute_tolerance,
                event_relative_tolerance,
                event_absolute_tolerance,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError(
                "Battery solve and event tolerances must be finite and positive."
            )
        if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int):
            raise TypeError("maximum_steps must be an integer.")
        if maximum_steps < 1:
            raise ValueError("maximum_steps must be positive.")
        self.solver = solver_
        self.stepsize_controller = controller
        self.adjoint = adjoint
        self.dt0 = dt0_
        (
            self.relative_tolerance,
            self.absolute_tolerance,
            self.event_relative_tolerance,
            self.event_absolute_tolerance,
        ) = tolerances
        self.maximum_steps = maximum_steps
        self.solve_plan_id = canonical_fingerprint(
            {
                "kind": "battery-diffrax-solve-plan",
                "solver": _type_id(solver_),
                "controller": _type_id(controller),
                "controller_configuration": repr(controller),
                "adjoint": None if adjoint is None else _type_id(adjoint),
                "dt0": dt0_,
                "rtol": tolerances[0],
                "atol": tolerances[1],
                "event_rtol": tolerances[2],
                "event_atol": tolerances[3],
                "initial_guard_policy": "terminate-nondifferentiable",
                "maximum_steps": maximum_steps,
            }
        )

    @property
    def adaptive(self) -> bool:
        return isinstance(
            self.stepsize_controller, dfx.AbstractAdaptiveStepSizeController
        )


class BatteryDAESolvePlan(StrictModule, NonTrainableState):
    """Native DAE policy with terminal guards and mandatory physical-state restarts."""

    policy: DAESolvePolicy
    event_consistency: DAEConsistencyPolicy
    guard_ids: tuple[str, ...] = eqx.field(static=True)
    maximum_events: int = eqx.field(static=True)
    event_tolerance: float = eqx.field(static=True)
    grazing_tolerance: float = eqx.field(static=True)
    localization_iterations: int = eqx.field(static=True)
    restart_policy: str = eqx.field(static=True)
    initial_guard_policy: str = eqx.field(static=True)
    solve_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: DAESolvePolicy | None = None,
        /,
        *,
        guard_ids: Sequence[str] = (),
        maximum_events: int = 1,
        event_tolerance: float = 1.0e-9,
        grazing_tolerance: float = 1.0e-8,
        localization_iterations: int = 32,
        restart_policy: Literal[
            "fixed-differential-state-reset-bdf"
        ] = "fixed-differential-state-reset-bdf",
        initial_guard_policy: Literal[
            "terminate-nondifferentiable"
        ] = "terminate-nondifferentiable",
    ):
        resolved = DAESolvePolicy() if policy is None else policy
        if not isinstance(resolved, DAESolvePolicy):
            raise TypeError("Battery DAE policy must be DAESolvePolicy.")
        identifiers = tuple(
            _identifier(value, "Battery native guard ID") for value in guard_ids
        )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Battery native guard IDs must be unique.")
        for value, name in (
            (maximum_events, "maximum_events"),
            (localization_iterations, "localization_iterations"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Battery DAE {name} must be a positive integer.")
        if any(
            not isfinite(float(value)) or value <= 0.0
            for value in (event_tolerance, grazing_tolerance)
        ):
            raise ValueError("Battery DAE event tolerances must be finite and positive.")
        if resolved.failure != "status":
            raise ValueError("Battery DAE execution requires native failure='status'.")
        if restart_policy != "fixed-differential-state-reset-bdf":
            raise ValueError(
                "Battery DAE transitions require fixed differential states and fresh BDF history."
            )
        if initial_guard_policy != "terminate-nondifferentiable":
            raise ValueError(
                "Battery DAE initial guards must terminate without a crossing derivative."
            )
        self.policy = resolved
        self.event_consistency = DAEConsistencyPolicy(0.0, 0.0, 0.0)
        self.guard_ids = identifiers
        self.maximum_events = maximum_events
        self.event_tolerance = float(event_tolerance)
        self.grazing_tolerance = float(grazing_tolerance)
        self.localization_iterations = localization_iterations
        self.restart_policy = restart_policy
        self.initial_guard_policy = initial_guard_policy
        self.solve_plan_id = canonical_fingerprint(
            {
                "kind": "battery-dae-solve-plan",
                "policy": repr(resolved),
                "guard_ids": identifiers,
                "maximum_events": maximum_events,
                "event_tolerance": self.event_tolerance,
                "grazing_tolerance": self.grazing_tolerance,
                "localization_iterations": localization_iterations,
                "restart_policy": restart_policy,
                "initial_guard_policy": initial_guard_policy,
            }
        )


class BatteryExperimentPlan(StrictModule, NonTrainableState):
    """Model/protocol/output/native-solve topology for one private candidate."""

    model: AbstractBatteryModelAdapter
    protocol: BatteryProtocolPlan
    outputs: BatteryOutputPlan
    native_solve_plan: BatteryDiffraxSolvePlan | BatteryDAESolvePlan
    save_times_s: Array
    capability_profile: CapabilityProfile
    support_tuple: SupportTuple
    experiment_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractBatteryModelAdapter,
        protocol: BatteryProtocolPlan,
        outputs: BatteryOutputPlan,
        native_solve_plan: BatteryDiffraxSolvePlan | BatteryDAESolvePlan,
        save_times_s: ArrayLike,
        capability_profile: CapabilityProfile,
        support_tuple: SupportTuple,
        /,
    ):
        if not isinstance(model, AbstractBatteryModelAdapter):
            raise TypeError("model must implement AbstractBatteryModelAdapter.")
        if not isinstance(protocol, BatteryProtocolPlan):
            raise TypeError("protocol must be BatteryProtocolPlan.")
        if not isinstance(outputs, BatteryOutputPlan):
            raise TypeError("outputs must be BatteryOutputPlan.")
        model_id = _identifier(model.model_id, "Battery model ID")
        if model.equation_form not in ("ode", "dae"):
            raise ValueError("Battery model equation_form must be 'ode' or 'dae'.")
        if model.equation_form == "ode" and not isinstance(
            native_solve_plan, BatteryDiffraxSolvePlan
        ):
            raise TypeError("ODE battery models require BatteryDiffraxSolvePlan.")
        if model.equation_form == "dae" and not isinstance(
            native_solve_plan, BatteryDAESolvePlan
        ):
            raise TypeError("DAE battery models require BatteryDAESolvePlan.")
        observable_names = tuple(model.observable_names)
        observable_units = tuple(model.observable_units)
        if len(observable_names) != len(observable_units):
            raise ValueError("Battery adapter observable names and units must align.")
        if any(
            not name or not unit
            for name, unit in zip(observable_names, observable_units, strict=True)
        ):
            raise ValueError(
                "Battery adapter observable names and units must be non-empty."
            )
        if (
            len(set(observable_names)) != len(observable_names)
            or "current_a" in observable_names
        ):
            raise ValueError(
                "Battery adapter observable names must be unique; current_a is protocol-owned."
            )
        times = jnp.asarray(save_times_s)
        if times.ndim != 1 or int(times.size) < 2:
            raise ValueError("Battery save_times_s must contain at least two nodes.")
        if jnp.issubdtype(times.dtype, jnp.complexfloating):
            raise TypeError("Battery save times must be real-valued.")
        host_times = np.asarray(times, dtype=float)
        if not np.all(np.isfinite(host_times)) or np.any(np.diff(host_times) <= 0.0):
            raise ValueError("Battery save times must be finite and strictly increasing.")
        if host_times[0] != protocol.t0_s or host_times[-1] != protocol.t1_s:
            raise ValueError(
                "Battery save times must span the complete protocol support."
            )
        validate_battery_candidate_profile(capability_profile, support_tuple)
        coordinates = dict(support_tuple.attributes)
        if (
            coordinates.get("model_id") != model_id
            or coordinates.get("equation_form") != model.equation_form
        ):
            raise ValueError(
                "Battery execution must match the admitted exact model ID and equation form."
            )
        solve_id = native_solve_plan.solve_plan_id
        self.model = model
        self.protocol = protocol
        self.outputs = outputs
        self.native_solve_plan = native_solve_plan
        self.save_times_s = jax.lax.stop_gradient(
            times.astype(jnp.result_type(times, protocol.boundary_times_s))
        )
        self.capability_profile = capability_profile
        self.support_tuple = support_tuple
        self.experiment_plan_id = canonical_fingerprint(
            {
                "kind": "battery-experiment-plan",
                "model_id": model_id,
                "protocol_id": protocol.protocol_id,
                "output_plan_id": outputs.output_plan_id,
                "native_solve_plan_id": solve_id,
                "save_times_s": host_times.tolist(),
                "profile_id": capability_profile.profile_id,
                "support_tuple_id": support_tuple.support_tuple_id,
                "terminal_convention": PASSIVE_TERMINAL_CONVENTION.convention_id,
            }
        )

    def prepare(
        self, /, *, admission: Any = None, distribution_id: str | None = None
    ) -> "PreparedBatteryExperiment":
        return prepare_battery_experiment(
            self, admission=admission, distribution_id=distribution_id
        )


class BatteryRuntimeInputs(StrictModule):
    """Dynamic parameters with distinct forcing and observed held-current policies."""

    parameters: PyTree
    input_policy: HeldInputPolicy
    observation_input_policy: HeldInputPolicy
    stop_thresholds: Array
    protocol_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: PyTree,
        input_policy: HeldInputPolicy,
        stop_thresholds: ArrayLike,
        /,
        *,
        protocol_id: str,
        observation_input_policy: HeldInputPolicy,
    ):
        if not isinstance(input_policy, HeldInputPolicy):
            raise TypeError("input_policy must be HeldInputPolicy.")
        if input_policy.node_side != "right":
            raise ValueError(
                "Battery execution input policy must use the forward right limit."
            )
        if not isinstance(observation_input_policy, HeldInputPolicy):
            raise TypeError("observation_input_policy must be HeldInputPolicy.")
        if (
            observation_input_policy.input_layout.layout_id
            != input_policy.input_layout.layout_id
        ):
            raise ValueError("Battery forcing and observation input layouts must match.")
        thresholds = jnp.asarray(stop_thresholds)
        if thresholds.ndim != 1:
            raise ValueError("Battery runtime stop thresholds must be rank one.")
        self.parameters = parameters
        self.input_policy = input_policy
        self.observation_input_policy = observation_input_policy
        self.stop_thresholds = thresholds
        self.protocol_id = _identifier(protocol_id, "Battery runtime protocol ID")

    def current(self, time_s: ArrayLike, state: PyTree, /) -> Array:
        return self.input_policy.evaluate(time_s, state)[0]

    def observed_current(self, time_s: ArrayLike, /) -> Array:
        return self.observation_input_policy.evaluate(time_s, jnp.asarray(0.0))[0]


class PreparedBatteryExperiment(StrictModule, NonTrainableState):
    """Prepared model topology, forced time topology, and static output/guard bindings."""

    plan: BatteryExperimentPlan
    prepared_model: Any
    transition_times_s: Array
    integration_time_grid: TimeGrid
    save_indices: Array
    output_indices: Array
    guard_output_indices: Array
    guard_step_indices: Array
    guard_polarities: Array
    output_units: tuple[str, ...] = eqx.field(static=True)
    guard_reason_names: tuple[str, ...] = eqx.field(static=True)
    stepsize_controller: Any | None
    native_guards: tuple[HybridGuardPlan, ...]
    admission: Any = eqx.field(static=True)
    distribution_id: str | None = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: BatteryExperimentPlan,
        prepared_model: Any,
        transition_times_s: Array,
        integration_time_grid: TimeGrid,
        save_indices: Array,
        output_indices: Array,
        guard_output_indices: Array,
        guard_step_indices: Array,
        guard_polarities: Array,
        output_units: tuple[str, ...],
        guard_reason_names: tuple[str, ...],
        stepsize_controller: Any | None,
        preparation_id: str,
        /,
        *,
        native_guards: tuple[HybridGuardPlan, ...],
        admission: Any,
        distribution_id: str | None,
    ):
        self.plan = plan
        self.prepared_model = prepared_model
        self.transition_times_s = transition_times_s
        self.integration_time_grid = integration_time_grid
        self.save_indices = save_indices
        self.output_indices = output_indices
        self.guard_output_indices = guard_output_indices
        self.guard_step_indices = guard_step_indices
        self.guard_polarities = guard_polarities
        self.output_units = output_units
        self.guard_reason_names = guard_reason_names
        self.stepsize_controller = stepsize_controller
        self.preparation_id = preparation_id
        self.native_guards = native_guards
        self.admission = admission
        self.distribution_id = distribution_id

    @property
    def time_grid(self) -> TimeGrid:
        return self.integration_time_grid

    def run(
        self,
        parameters: PyTree,
        initial_condition: PyTree,
        protocol_values: BatteryProtocolValues,
        /,
    ) -> BatteryExperimentResult:
        return run_battery_experiment(
            self, parameters, initial_condition, protocol_values
        )


def _model_current(adapter, prepared_model, times, states, runtime, /):
    current = getattr(adapter, "current", None)
    if callable(current):
        values = jnp.asarray(current(prepared_model, times, states, runtime))
        if values.shape != jnp.asarray(times).shape:
            raise ValueError(
                "Battery adapter current must match the observation time shape."
            )
        return values
    return (
        runtime.observed_current(times)
        if jnp.asarray(times).ndim == 0
        else jax.vmap(runtime.observed_current)(times)
    )


class _BatteryGuardCondition(StrictModule, NonTrainableState):
    adapter: AbstractBatteryModelAdapter
    prepared_model: Any
    protocol: BatteryProtocolPlan
    output_index: int = eqx.field(static=True)
    threshold_index: int = eqx.field(static=True)
    step_index: int = eqx.field(static=True)
    polarity: int = eqx.field(static=True)

    def margin(self, t: Array, y: PyTree, args: BatteryRuntimeInputs, /) -> Array:
        if self.output_index < 0:
            value = _model_current(self.adapter, self.prepared_model, t, y, args)
        else:
            observed = self.adapter.observe(self.prepared_model, t, y, args)
            if not isinstance(observed, BatteryModelOutput):
                raise TypeError("Battery adapter observe must return BatteryModelOutput.")
            if observed.values.shape != (len(self.adapter.observable_names),):
                raise ValueError("Scalar battery guard observation has an invalid shape.")
            value = observed.values[self.output_index]
        threshold = args.stop_thresholds[self.threshold_index]
        margin = self.polarity * (value - threshold)
        return margin

    def __call__(
        self,
        t: Array,
        y: PyTree,
        args: BatteryRuntimeInputs,
        **kwargs: Any,
    ) -> Array:
        del kwargs
        margin = self.margin(t, y, args)
        active = self.protocol.interval_index(t) == self.step_index
        return jnp.where(active, margin, jnp.ones_like(margin))


class _BatteryInitialGuardCondition(StrictModule, NonTrainableState):
    """Boolean startup stop: it needs no positive crossing or fabricated root."""

    condition: _BatteryGuardCondition
    segment_index: int | None = eqx.field(static=True)

    def __call__(
        self, t: Array, y: PyTree, args: BatteryRuntimeInputs, **kwargs: Any
    ) -> Array:
        del kwargs
        step = self.condition.step_index
        at_start = t == self.condition.protocol.boundary_times_s[step]
        if self.segment_index is not None:
            at_start = at_start & (step == self.segment_index)

        def check():
            restarted_runtime = eqx.tree_at(
                lambda value: value.observation_input_policy, args, args.input_policy
            )
            margin = self.condition.margin(t, y, restarted_runtime)
            return jnp.isfinite(margin) & (margin <= 0.0)

        return jax.lax.cond(at_start, check, lambda: jnp.asarray(False))


def _observable_index(name: str, model_names: tuple[str, ...], /) -> int:
    if name == "current_a":
        return -1
    if name not in model_names:
        raise ValueError(f"Battery model does not provide required observable {name!r}.")
    return model_names.index(name)


def _fixed_grid_covers(step_times: np.ndarray, required_times: np.ndarray, /) -> bool:
    locations = np.searchsorted(step_times, required_times)
    inside = locations < step_times.size
    safe = np.minimum(locations, step_times.size - 1)
    return bool(np.all(inside & (step_times[safe] == required_times)))


def prepare_battery_experiment(
    plan: BatteryExperimentPlan,
    /,
    *,
    admission: Any = None,
    distribution_id: str | None = None,
) -> PreparedBatteryExperiment:
    """Prepare static bindings and force every protocol transition into native time topology."""
    if not isinstance(plan, BatteryExperimentPlan):
        raise TypeError("plan must be BatteryExperimentPlan.")
    admission = validate_battery_execution_admission(
        plan.capability_profile,
        plan.support_tuple,
        admission,
        distribution_id=distribution_id,
    )
    if admission is not None:
        admission.validate_adapter(plan.model)
    prepared_model = plan.model.prepare()
    native_guards = ()
    if isinstance(plan.native_solve_plan, BatteryDAESolvePlan):
        factory = getattr(plan.model, "native_guards", None)
        if not callable(factory):
            raise TypeError("DAE battery adapters require native_guards(prepared_model).")
        native_guards = tuple(factory(prepared_model))
        if any(
            not isinstance(guard, HybridGuardPlan)
            or not guard.terminal
            or guard.direction != -1
            for guard in native_guards
        ):
            raise ValueError(
                "Battery native DAE guards must be terminal decreasing margins."
            )
        if (
            tuple(guard.guard_id for guard in native_guards)
            != plan.native_solve_plan.guard_ids
        ):
            raise ValueError(
                "Battery native DAE guard IDs must match the solve plan exactly."
            )
    protocol_times = np.asarray(plan.protocol.transition_times_s, dtype=float)
    transition_times = np.unique(protocol_times)
    if transition_times.size != protocol_times.size:
        raise ValueError("Battery protocol transition times must be unique.")
    save_times = np.asarray(plan.save_times_s, dtype=float)
    if transition_times[0] != save_times[0] or transition_times[-1] != save_times[-1]:
        raise ValueError("Battery transitions and save grid must have identical support.")
    # Protocol boundaries are authoritative. Snap only save nodes within local
    # floating-point roundoff (four machine epsilons), not a solver tolerance.
    # There is no absolute floor: short protocols retain their sampling scale,
    # and distinct protocol boundaries are never merged.
    positions = np.searchsorted(transition_times, save_times)
    lower = transition_times[np.maximum(positions - 1, 0)]
    upper = transition_times[np.minimum(positions, transition_times.size - 1)]
    nearest = np.where(save_times - lower <= upper - save_times, lower, upper)
    tolerance = (
        4.0
        * np.finfo(plan.save_times_s.dtype).eps
        * np.maximum(np.abs(save_times), np.abs(nearest))
    )
    normalized_save_times = np.where(
        np.abs(save_times - nearest) <= tolerance, nearest, save_times
    )
    integration_times = np.unique(
        np.concatenate((normalized_save_times, transition_times))
    )
    time_grid = TimeGrid(
        jnp.asarray(integration_times, dtype=plan.save_times_s.dtype),
        time_id=f"battery:{plan.experiment_plan_id}:integration-time",
    )
    save_indices_host = np.searchsorted(integration_times, normalized_save_times)
    if not np.all(integration_times[save_indices_host] == normalized_save_times):
        raise ValueError(
            "Battery save times do not map exactly into integration topology."
        )
    model_names = tuple(plan.model.observable_names)
    model_units = tuple(plan.model.observable_units)
    output_indices_host = np.asarray(
        [_observable_index(name, model_names) + 1 for name in plan.outputs.names],
        dtype=np.int32,
    )
    output_units = tuple(
        "A" if name == "current_a" else model_units[model_names.index(name)]
        for name in plan.outputs.names
    )
    guards = plan.protocol.stop_guards
    guard_indices_host = np.asarray(
        [_observable_index(guard.quantity, model_names) for guard in guards],
        dtype=np.int32,
    )
    guard_polarities_host = np.asarray(
        [1 if guard.direction == "below" else -1 for guard in guards],
        dtype=np.int32,
    )
    controller: Any | None
    if isinstance(plan.native_solve_plan, BatteryDiffraxSolvePlan):
        base = plan.native_solve_plan.stepsize_controller
        internal_transitions = jnp.asarray(
            transition_times[1:-1], dtype=plan.save_times_s.dtype
        )
        if isinstance(base, dfx.AbstractAdaptiveStepSizeController):
            controller = (
                base
                if internal_transitions.size == 0
                else dfx.ClipStepSizeController(
                    base,
                    step_ts=internal_transitions,
                    jump_ts=internal_transitions,
                )
            )
        else:
            step_times = np.asarray(base.ts, dtype=float)
            if (
                step_times.ndim != 1
                or step_times[0] != integration_times[0]
                or step_times[-1] != integration_times[-1]
                or not _fixed_grid_covers(step_times, transition_times)
            ):
                raise ValueError(
                    "diffrax.StepTo grid must span and exactly contain every protocol transition."
                )
            controller = base
    else:
        controller = None
    preparation_id = canonical_fingerprint(
        {
            "kind": "prepared-battery-experiment",
            "experiment_plan_id": plan.experiment_plan_id,
            "integration_times_s": integration_times.tolist(),
            "save_indices": save_indices_host.tolist(),
            "output_indices": output_indices_host.tolist(),
            "guard_indices": guard_indices_host.tolist(),
            "model_id": plan.model.model_id,
            "native_guard_ids": [guard.guard_id for guard in native_guards],
            "admission_id": None
            if admission is None
            else admission.numerical_admission_id,
            "distribution_id": distribution_id,
        }
    )
    return PreparedBatteryExperiment(
        plan,
        prepared_model,
        jax.lax.stop_gradient(
            jnp.asarray(transition_times, dtype=plan.save_times_s.dtype)
        ),
        time_grid,
        jax.lax.stop_gradient(jnp.asarray(save_indices_host, dtype=jnp.int32)),
        jax.lax.stop_gradient(jnp.asarray(output_indices_host, dtype=jnp.int32)),
        jax.lax.stop_gradient(jnp.asarray(guard_indices_host, dtype=jnp.int32)),
        plan.protocol.guard_step_indices,
        jax.lax.stop_gradient(jnp.asarray(guard_polarities_host, dtype=jnp.int32)),
        output_units,
        tuple(guard.reason for guard in guards)
        + tuple(guard.guard_id for guard in native_guards),
        controller,
        preparation_id,
        native_guards=native_guards,
        admission=admission,
        distribution_id=distribution_id,
    )


def _event(
    prepared: PreparedBatteryExperiment,
    /,
    *,
    segment_index: int | None = None,
) -> dfx.Event | None:
    guards = prepared.plan.protocol.stop_guards
    if not guards:
        return None
    conditions = tuple(
        _BatteryGuardCondition(
            prepared.plan.model,
            prepared.prepared_model,
            prepared.plan.protocol,
            int(prepared.guard_output_indices[index]),
            index,
            int(prepared.guard_step_indices[index]),
            int(prepared.guard_polarities[index]),
        )
        for index in range(len(guards))
    )
    initial_conditions = tuple(
        _BatteryInitialGuardCondition(condition, segment_index)
        for condition in conditions
    )
    solve_plan = prepared.plan.native_solve_plan
    if not isinstance(solve_plan, BatteryDiffraxSolvePlan):
        raise ValueError(
            "Native DAE events must be supplied by a concrete DAE model plan."
        )
    return dfx.Event(
        initial_conditions + conditions,
        root_finder=optx.Newton(
            rtol=solve_plan.event_relative_tolerance,
            atol=solve_plan.event_absolute_tolerance,
        ),
        direction=(None,) * len(initial_conditions) + (False,) * len(conditions),
    )


def _checked_ode_bounds(
    problem: DifferentialProblem, start: float, end: float, /
) -> DifferentialProblem:
    actual = jnp.stack((problem.t0, problem.t1))
    expected = jnp.asarray((start, end), dtype=actual.dtype)
    checked = eqx.error_if(
        actual,
        jnp.any(actual != expected),
        "Battery model problem must span the complete protocol interval.",
    )
    return eqx.tree_at(
        lambda value: (value.t0, value.t1),
        problem,
        (checked[0], checked[1]),
    )


def _validated_problem(
    prepared: PreparedBatteryExperiment,
    initial_state: PyTree,
    runtime: BatteryRuntimeInputs,
    /,
) -> DifferentialProblem | DifferentialAlgebraicProblem:
    problem = prepared.plan.model.problem(prepared.prepared_model, initial_state, runtime)
    if problem.args is not runtime:
        raise ValueError(
            "Battery model problem must retain the supplied BatteryRuntimeInputs."
        )
    if prepared.plan.model.equation_form == "ode":
        if not isinstance(problem, DifferentialProblem):
            raise TypeError("ODE battery adapter must return DifferentialProblem.")
        problem = _checked_ode_bounds(
            problem,
            prepared.plan.protocol.t0_s,
            prepared.plan.protocol.t1_s,
        )
    elif not isinstance(problem, DifferentialAlgebraicProblem):
        raise TypeError("DAE battery adapter must return DifferentialAlgebraicProblem.")
    return problem


def _safe_states(states: PyTree, initial_state: PyTree, valid: Array, /) -> PyTree:
    def sanitize(values, initial):
        values_ = jnp.asarray(values)
        initial_ = jnp.asarray(initial)
        mask = valid.reshape(valid.shape + (1,) * (values_.ndim - valid.ndim))
        return jnp.where(mask, values_, jnp.broadcast_to(initial_, values_.shape))

    return jax.tree.map(sanitize, states, initial_state)


def _localized_event_time(
    prepared: PreparedBatteryExperiment,
    native_solution: Any,
    runtime: BatteryRuntimeInputs,
    winner: Array,
    terminated: Array,
    /,
) -> Array:
    event = _event(prepared)
    if event is None:
        raise ValueError("Event localization requires battery stop guards.")
    interpolation = native_solution.interpolation
    backend = interpolation.interpolation

    def locate(_: None) -> Array:
        size = backend.ts_size[0]

        def initial_event(_: None) -> Array:
            direction = backend.direction[0]
            return backend.ts[0, jnp.maximum(size - 1, 0)] * direction

        def bracketed_event(_: None) -> Array:
            direction = backend.direction[0]
            lower = backend.ts[0, size - 2] * direction
            upper = backend.ts[0, size - 1] * direction

            def residual(time, unused):
                del unused
                state = interpolation.evaluate(time)
                values = jnp.stack(
                    tuple(
                        condition(t=time, y=state, args=runtime)
                        for condition in event.cond_fn[
                            prepared.plan.protocol.guard_count :
                        ]
                    )
                )
                return values[winner]

            return optx.root_find(
                residual,
                event.root_finder,
                y0=upper,
                options={"lower": lower, "upper": upper},
                throw=False,
            ).value

        initial_mask = jnp.stack(tuple(jax.tree.leaves(native_solution.event_mask)))[
            : prepared.plan.protocol.guard_count
        ]
        return jax.lax.cond(
            jnp.any(initial_mask) | (size <= 1),
            initial_event,
            bracketed_event,
            operand=None,
        )

    return jax.lax.cond(
        terminated,
        locate,
        lambda _: jnp.asarray(jnp.nan, dtype=prepared.plan.save_times_s.dtype),
        operand=None,
    )


def _termination(
    prepared: PreparedBatteryExperiment,
    native_solution: Any,
    runtime: BatteryRuntimeInputs,
    /,
) -> BatteryTermination:
    if isinstance(native_solution, BatteryDAESolution):
        terminated = jnp.asarray(False)
        event_time = jnp.asarray(jnp.nan, dtype=prepared.plan.save_times_s.dtype)
        winner = jnp.asarray(-1, dtype=jnp.int32)
        step = jnp.asarray(-1, dtype=jnp.int32)
        derivative_valid = jnp.asarray(True)
        for index, (active, segment) in enumerate(
            zip(
                native_solution.replay.segment_active,
                native_solution.segments,
                strict=True,
            )
        ):
            if segment.events is None:
                continue
            _, indices = _dae_event_plan(prepared, index)
            event = segment.events
            slot = jnp.maximum(event.event_count - 1, 0)
            hit = active & event.terminal & event.valid[slot]
            local_winner = jnp.maximum(event.event_indices[slot], 0)
            winner = jnp.where(hit, jnp.asarray(indices)[local_winner], winner)
            event_time = jnp.where(hit, event.event_times[slot], event_time)
            step = jnp.where(hit, index, step)
            terminated = terminated | hit
            derivative_valid = jnp.where(
                hit, event.derivative_valid[slot], derivative_valid
            )
        return BatteryTermination(
            terminated,
            event_time,
            winner,
            step,
            reason_names=prepared.guard_reason_names,
            derivative_valid=derivative_valid,
        )
    if not prepared.guard_reason_names:
        return BatteryTermination(
            jnp.asarray(False),
            jnp.asarray(jnp.nan, dtype=prepared.plan.save_times_s.dtype),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
            reason_names=(),
        )
    mask = jnp.stack(
        tuple(
            jnp.asarray(value, dtype=bool)
            for value in jax.tree.leaves(native_solution.event_mask)
        )
    )
    guard_count = prepared.plan.protocol.guard_count
    initial_mask, crossing_mask = mask[:guard_count], mask[guard_count:]
    mask = initial_mask | crossing_mask
    terminated = jnp.asarray(native_solution.event_terminated, dtype=bool)
    winner = jnp.argmax(mask).astype(jnp.int32)
    event_time = _localized_event_time(
        prepared, native_solution, runtime, winner, terminated
    )
    winner = jnp.where(terminated, winner, jnp.asarray(-1, dtype=jnp.int32))
    safe_winner = jnp.maximum(winner, 0)
    step = jnp.where(
        terminated,
        prepared.guard_step_indices[safe_winner],
        jnp.asarray(-1, dtype=jnp.int32),
    )
    return BatteryTermination(
        terminated,
        event_time,
        winner,
        step,
        reason_names=prepared.guard_reason_names,
        derivative_valid=~jnp.any(initial_mask),
    )


def _selected_outputs(
    prepared: PreparedBatteryExperiment,
    initial_state: PyTree,
    runtime: BatteryRuntimeInputs,
    native_solution: Any,
    termination: BatteryTermination,
    /,
) -> tuple[BatterySelectedOutputs, Array, Array]:
    times = native_solution.times[prepared.save_indices]
    states = jax.tree.map(
        lambda values: values[prepared.save_indices], native_solution.states
    )
    native_valid = native_solution.valid[prepared.save_indices]
    if prepared.plan.model.equation_form == "ode" and prepared.guard_reason_names:
        solve_plan = prepared.plan.native_solve_plan
        if not isinstance(solve_plan, BatteryDiffraxSolvePlan):
            raise TypeError("ODE event repair requires BatteryDiffraxSolvePlan.")
        requested = prepared.plan.save_times_s
        tolerance = (
            solve_plan.event_absolute_tolerance
            + solve_plan.event_relative_tolerance * jnp.abs(termination.time_s)
        )
        coincident = (
            termination.terminated
            & (~native_valid)
            & (jnp.abs(requested - termination.time_s) <= tolerance)
        )
        backend = native_solution.interpolation.interpolation
        fallback_query = backend.ts[0, 0] * backend.direction[0]
        root_query = jnp.where(
            termination.terminated,
            termination.time_s,
            fallback_query,
        )

        def immediate_state():
            valid_count = jnp.sum(native_solution.valid.astype(jnp.int32))
            last = jnp.maximum(valid_count - 1, 0)
            return jax.tree.map(
                lambda values, initial: jnp.where(valid_count > 0, values[last], initial),
                native_solution.states,
                initial_state,
            )

        root_state = jax.lax.cond(
            termination.derivative_valid,
            lambda: native_solution.interpolation.evaluate(root_query),
            immediate_state,
        )

        def repair(values, root):
            values_ = jnp.asarray(values)
            mask = coincident.reshape(
                coincident.shape + (1,) * (values_.ndim - coincident.ndim)
            )
            return jnp.where(mask, jnp.broadcast_to(root, values_.shape), values_)

        states = jax.tree.map(repair, states, root_state)
        native_valid = native_valid | coincident
        times = jnp.where(native_valid, requested, jnp.inf)
    if isinstance(native_solution, BatteryDAESolution):
        for active, segment in zip(
            native_solution.replay.segment_active, native_solution.segments, strict=True
        ):
            events = segment.events
            if events is None:
                continue
            slot = jnp.maximum(events.event_count - 1, 0)
            coincident = (
                active
                & events.terminal
                & events.valid[slot]
                & (~native_valid)
                & (
                    jnp.abs(prepared.plan.save_times_s - events.event_times[slot])
                    <= prepared.plan.native_solve_plan.event_tolerance
                )
            )
            mask = coincident.reshape(coincident.shape + (1,) * (states.ndim - 1))
            states = jnp.where(mask, events.states_before[slot], states)
            native_valid = native_valid | coincident
            times = jnp.where(coincident, prepared.plan.save_times_s, times)
    safe_times = jnp.where(native_valid, times, prepared.plan.save_times_s[0])
    safe_states = _safe_states(states, initial_state, native_valid)
    if isinstance(native_solution, BatteryDAESolution):
        forcing_currents = native_solution.forcing_currents_a[prepared.save_indices]

        def observe_one(time, state, current):
            policy = HeldInputPolicy(
                runtime.input_policy.times,
                jnp.broadcast_to(current, (len(prepared.plan.protocol.steps), 1)),
                input_layout=runtime.input_policy.input_layout,
                node_side="right",
                policy_id=f"{runtime.input_policy.policy_id}:one-sided-observation",
            )
            observation_runtime = BatteryRuntimeInputs(
                runtime.parameters,
                policy,
                runtime.stop_thresholds,
                protocol_id=runtime.protocol_id,
                observation_input_policy=policy,
            )
            observed = prepared.plan.model.observe(
                prepared.prepared_model, time, state, observation_runtime
            )
            current_value = _model_current(
                prepared.plan.model,
                prepared.prepared_model,
                time,
                state,
                observation_runtime,
            )
            return observed, current_value

        observed, current = jax.vmap(observe_one)(
            safe_times, safe_states, forcing_currents
        )
    else:
        observed = prepared.plan.model.observe(
            prepared.prepared_model, safe_times, safe_states, runtime
        )
        current = _model_current(
            prepared.plan.model, prepared.prepared_model, safe_times, safe_states, runtime
        )
    if not isinstance(observed, BatteryModelOutput):
        raise TypeError("Battery adapter observe must return BatteryModelOutput.")
    expected = safe_times.shape + (len(prepared.plan.model.observable_names),)
    if observed.values.shape != expected:
        raise ValueError(f"Battery adapter observation shape must be {expected}.")
    available = jnp.concatenate((current[:, None], observed.values), axis=-1)
    selected = jnp.take(available, prepared.output_indices, axis=-1)
    selected_finite = jnp.all(jnp.isfinite(selected), axis=-1)
    valid = native_valid & observed.domain_valid & selected_finite
    filled_times = jnp.where(native_valid, prepared.plan.save_times_s, jnp.inf)
    filled_values = jnp.where(valid[:, None], selected, jnp.zeros_like(selected))
    outputs = BatterySelectedOutputs(
        filled_times,
        filled_values,
        valid,
        names=prepared.plan.outputs.names,
        units=prepared.output_units,
    )
    all_observables_finite = jnp.all(jnp.isfinite(observed.values), axis=-1)
    domain_ok = jnp.all((~native_valid) | observed.domain_valid)
    finite_ok = jnp.all((~native_valid) | all_observables_finite)
    return outputs, domain_ok, finite_ok


def _identity_coordinates(state: PyTree, /):
    leaves, treedef = jax.tree.flatten(state)
    maps = jax.tree.unflatten(treedef, [None] * len(leaves))
    return prepare_real_coordinate_tree(state, maps)


def _solve_fixed_segments(
    prepared: PreparedBatteryExperiment,
    problem: DifferentialProblem,
    initial_state: PyTree,
    solve_plan: BatteryDiffraxSolvePlan,
    runtime: BatteryRuntimeInputs,
    /,
) -> DifferentialSolution:
    controller = prepared.stepsize_controller
    if not isinstance(controller, dfx.StepTo):
        raise TypeError("Fixed segmented battery execution requires diffrax.StepTo.")
    boundaries = np.asarray(prepared.transition_times_s, dtype=float)
    requested = np.asarray(prepared.integration_time_grid.times, dtype=float)
    step_times = np.asarray(controller.ts, dtype=float)
    sample_count = requested.size
    stitched_times = jnp.full_like(prepared.integration_time_grid.times, jnp.inf)
    stitched_valid = jnp.zeros((sample_count,), dtype=bool)
    stitched_states = jax.tree.map(
        lambda value: jnp.zeros(
            (sample_count,) + jnp.asarray(value).shape,
            dtype=jnp.asarray(value).dtype,
        ),
        initial_state,
    )
    current_state = initial_state
    segment_solutions: list[DifferentialSolution] = []
    for segment_index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        local_steps = step_times[(step_times >= start) & (step_times <= end)]
        if local_steps.size < 2 or local_steps[0] != start or local_steps[-1] != end:
            raise ValueError(
                "Every fixed protocol segment requires exact StepTo endpoints."
            )
        owns = ((requested >= start) if segment_index == 0 else (requested > start)) & (
            requested <= end
        )
        global_indices = np.nonzero(owns)[0]
        local_save_times = np.unique(
            np.concatenate(([start], requested[global_indices], [end]))
        )
        segment_current = runtime.current(
            jnp.asarray(0.5 * (start + end), dtype=prepared.plan.save_times_s.dtype),
            current_state,
        )
        local_input_policy = HeldInputPolicy(
            jnp.asarray((start, end), dtype=prepared.plan.save_times_s.dtype),
            jnp.reshape(segment_current, (1, 1)),
            input_layout=runtime.input_policy.input_layout,
            node_side="right",
            policy_id=(
                f"{runtime.input_policy.policy_id}:protocol-segment:{segment_index}"
            ),
        )
        local_runtime = BatteryRuntimeInputs(
            runtime.parameters,
            local_input_policy,
            runtime.stop_thresholds,
            protocol_id=runtime.protocol_id,
            observation_input_policy=runtime.observation_input_policy,
        )
        local_problem = prepared.plan.model.problem(
            prepared.prepared_model, current_state, local_runtime
        )
        if not isinstance(local_problem, DifferentialProblem):
            raise TypeError(
                "ODE battery adapter must return DifferentialProblem per segment."
            )
        if local_problem.args is not local_runtime:
            raise ValueError(
                "Segmented battery problem must retain its local runtime inputs."
            )
        local_problem = _checked_ode_bounds(local_problem, start, end)
        if local_problem.stochastic:
            raise ValueError(
                "Fixed segmented battery execution does not support stochastic ODEs."
            )
        event = _event(prepared, segment_index=segment_index)
        local_solution = solve_diffrax(
            local_problem,
            save_times=jnp.asarray(
                local_save_times, dtype=prepared.plan.save_times_s.dtype
            ),
            solver=solve_plan.solver,
            stepsize_controller=dfx.StepTo(
                ts=jnp.asarray(local_steps, dtype=prepared.plan.save_times_s.dtype)
            ),
            adjoint=solve_plan.adjoint,
            dt0=None,
            event=event,
            rtol=solve_plan.relative_tolerance,
            atol=solve_plan.absolute_tolerance,
            dense=event is not None,
            max_steps=solve_plan.maximum_steps,
            throw=False,
            solver_configuration_id=(
                f"{solve_plan.solve_plan_id}:protocol-segment:{segment_index}"
            ),
            state_coordinates=_identity_coordinates(current_state),
        )
        segment_solutions.append(local_solution)
        local_indices = np.searchsorted(local_save_times, requested[global_indices])
        global_index_array = jnp.asarray(global_indices, dtype=jnp.int32)
        local_index_array = jnp.asarray(local_indices, dtype=jnp.int32)
        stitched_times = stitched_times.at[global_index_array].set(
            local_solution.times[local_index_array]
        )
        stitched_valid = stitched_valid.at[global_index_array].set(
            local_solution.valid[local_index_array]
        )
        stitched_states = jax.tree.map(
            lambda target, source: target.at[global_index_array].set(
                source[local_index_array]
            ),
            stitched_states,
            local_solution.states,
        )
        if bool(np.asarray(local_solution.event_terminated)) or not bool(
            np.asarray(local_solution.backend_successful)
        ):
            break
        current_state = jax.tree.map(lambda values: values[-1], local_solution.states)
    last = segment_solutions[-1]
    backend_successful = all(
        bool(np.asarray(solution.backend_successful)) for solution in segment_solutions
    )
    event_terminated = any(
        bool(np.asarray(solution.event_terminated)) for solution in segment_solutions
    )
    return DifferentialSolution(
        times=stitched_times,
        states=stitched_states,
        valid=stitched_valid,
        interpolation=last.interpolation,
        backend_result=last.backend_result,
        stats={
            "protocol_segments": len(segment_solutions),
            "num_steps": sum(
                solution.stats["num_steps"] for solution in segment_solutions
            ),
            "num_accepted_steps": sum(
                solution.stats["num_accepted_steps"] for solution in segment_solutions
            ),
        },
        event_mask=last.event_mask,
        realization=None,
        wiener_term_slices=problem.wiener_term_slices,
        solver_name=last.solver_name,
        interpretation=problem.interpretation,
        state_geometry_id=problem.state_geometry_id,
        solver_id=last.solver_id,
        resolved_method=last.resolved_method,
        discretization_bundle=problem.discretization_bundle,
        backend_successful=backend_successful,
        event_terminated=event_terminated,
        temporal_evidence=last.temporal_evidence,
        problem_id=problem.problem_id,
    )


class _BatteryDAEProtocolGuard(StrictModule, NonTrainableState):
    adapter: AbstractBatteryModelAdapter
    prepared_model: Any
    output_index: int = eqx.field(static=True)
    threshold_index: int = eqx.field(static=True)
    polarity: int = eqx.field(static=True)

    def __call__(self, time, state, runtime, /):
        if self.output_index < 0:
            value = _model_current(
                self.adapter, self.prepared_model, time, state, runtime
            )
        else:
            output = self.adapter.observe(self.prepared_model, time, state, runtime)
            if not isinstance(output, BatteryModelOutput):
                raise TypeError("Battery adapter observe must return BatteryModelOutput.")
            value = output.values[self.output_index]
        return self.polarity * (value - runtime.stop_thresholds[self.threshold_index])


def _identity_dae_reset(time, state, rate, args, /):
    del time, args
    return state, rate


def _dae_event_plan(prepared, segment_index, /):
    solve_plan = prepared.plan.native_solve_plan
    guards = []
    indices = []
    for index, guard in enumerate(prepared.plan.protocol.stop_guards):
        if int(prepared.guard_step_indices[index]) != segment_index:
            continue
        guards.append(
            HybridGuardPlan(
                _BatteryDAEProtocolGuard(
                    prepared.plan.model,
                    prepared.prepared_model,
                    int(prepared.guard_output_indices[index]),
                    index,
                    int(prepared.guard_polarities[index]),
                ),
                direction=-1,
                terminal=True,
                guard_id=f"battery:{prepared.plan.protocol.protocol_id}:stop:{index}",
            )
        )
        indices.append(index)
    guards.extend(prepared.native_guards)
    indices.extend(
        range(
            prepared.plan.protocol.guard_count,
            prepared.plan.protocol.guard_count + len(prepared.native_guards),
        )
    )
    if not guards:
        return None, ()
    return DAEEventPlan(
        HybridSchedulePlan(
            tuple(ScheduledHybridGuard(guard) for guard in guards),
            maximum_events=solve_plan.maximum_events,
            simultaneous_tolerance=solve_plan.event_tolerance,
        ),
        tuple(
            DAEResetMap(
                _identity_dae_reset,
                DAEInitializationSpec.check_only(),
                reset_id=f"battery-stop:{guard.guard_id}",
            )
            for guard in guards
        ),
        solve_plan.event_consistency,
        event_tolerance=solve_plan.event_tolerance,
        grazing_tolerance=solve_plan.grazing_tolerance,
        localization_iterations=solve_plan.localization_iterations,
    ), tuple(indices)


def _neutral_dae_solution(native_prepared, runtime, /):
    """Allocate inactive native slots without evaluating a physical trajectory."""
    shape = eqx.filter_eval_shape(lambda: solve_dae(native_prepared, args=runtime))
    return jax.tree.map(
        lambda value: jnp.zeros(value.shape, value.dtype)
        if isinstance(value, jax.ShapeDtypeStruct)
        else value,
        shape,
    )


def _initial_guard_margins(event_plan, time, initialization, runtime, /):
    return jax.lax.cond(
        initialization.valid,
        lambda: jnp.stack(
            tuple(
                scheduled.guard.guard(time, initialization.state, runtime)
                for scheduled in event_plan.schedule.events
            )
        ),
        lambda: jnp.ones(
            (len(event_plan.schedule.events),), dtype=initialization.state.real.dtype
        ),
    )


def _initial_dae_solution(native_prepared, runtime, initialization, event_plan, /):
    """Represent a genuine initial consistency failure or immediate terminal guard."""
    solution = _neutral_dae_solution(native_prepared, runtime)
    time = native_prepared.time_grid.times[0]
    count = int(native_prepared.time_grid.times.size)
    state, rate = initialization.state, initialization.state_rate
    valid = jnp.zeros((count,), dtype=bool).at[0].set(initialization.valid)
    states = jnp.zeros((count,) + state.shape, state.dtype).at[0].set(state)
    rates = jnp.zeros_like(states).at[0].set(rate)
    status = (
        jnp.full((count,), int(DAEStatus.NOT_RUN), dtype=jnp.int32)
        .at[0]
        .set(
            jnp.where(
                initialization.valid,
                int(DAEStatus.SUCCESS),
                int(DAEStatus.INITIALIZATION_FAILED),
            )
        )
    )
    events = None
    if event_plan is not None:
        events = empty_dae_event_result(event_plan, state)
        margins = _initial_guard_margins(event_plan, time, initialization, runtime)
        hits = jnp.isfinite(margins) & (margins <= 0.0)
        priorities = jnp.asarray(
            tuple(guard.guard.priority for guard in event_plan.schedule.events),
            dtype=jnp.int32,
        )
        winner = jnp.argmin(jnp.where(hits, priorities, jnp.iinfo(jnp.int32).max)).astype(
            jnp.int32
        )
        occurred = initialization.valid & jnp.any(hits)
        active = events.replay.active.at[0].set(occurred)
        recorded_states = events.states_before.at[0].set(state)
        recorded_rates = events.state_rates_before.at[0].set(rate)
        event_times = events.event_times.at[0].set(time)
        replay = eqx.tree_at(
            lambda value: (
                value.bracket_left_times,
                value.bracket_right_times,
                value.recorded_event_times,
                value.recorded_states_before,
                value.recorded_states_after,
                value.active,
            ),
            events.replay,
            (
                event_times,
                event_times,
                event_times,
                recorded_states,
                recorded_states,
                active,
            ),
        )
        events = eqx.tree_at(
            lambda value: (
                value.event_times,
                value.event_indices,
                value.states_before,
                value.states_after,
                value.state_rates_before,
                value.state_rates_after,
                value.guard_residuals,
                value.pre_residual_norms,
                value.post_residual_norms,
                value.valid,
                value.status,
                value.event_count,
                value.terminal,
                value.simultaneous,
                value.replay,
            ),
            events,
            (
                event_times,
                events.event_indices.at[0].set(winner),
                recorded_states,
                recorded_states,
                recorded_rates,
                recorded_rates,
                events.guard_residuals.at[0].set(margins[winner]),
                events.pre_residual_norms.at[0].set(initialization.residual_norm),
                events.post_residual_norms.at[0].set(initialization.residual_norm),
                active,
                events.status.at[0].set(
                    jnp.where(
                        occurred, int(DAEEventStatus.SUCCESS), int(DAEEventStatus.NOT_RUN)
                    )
                ),
                occurred.astype(jnp.int32),
                occurred,
                events.simultaneous.at[0].set(jnp.sum(hits) > 1),
                replay,
            ),
        )
    termination = jnp.where(
        initialization.valid,
        int(DAETerminationStatus.EVENT_TERMINATED),
        int(DAETerminationStatus.INITIALIZATION_FAILED),
    ).astype(jnp.int32)
    step_size = native_prepared.time_grid.durations[0]
    continuation = eqx.tree_at(
        lambda value: (
            value.time,
            value.states,
            value.state_rates,
            value.times,
            value.step_sizes,
            value.history_depth,
            value.accepted_order,
            value.previous_error_ratio,
            value.proposed_step_size,
            value.last_alpha,
        ),
        solution.continuation,
        (
            time,
            jnp.broadcast_to(state, solution.continuation.states.shape),
            jnp.broadcast_to(rate, solution.continuation.state_rates.shape),
            jnp.full_like(solution.continuation.times, time),
            jnp.full_like(solution.continuation.step_sizes, step_size),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1.0, dtype=state.real.dtype),
            step_size,
            1.0 / step_size,
        ),
    )
    if continuation.nonlinear_solve is not None:
        dynamic_stage, _ = eqx.partition(native_prepared.stage_solve, eqx.is_array)
        continuation = eqx.tree_at(
            lambda value: value.nonlinear_solve, continuation, dynamic_stage
        )
    solution = eqx.tree_at(
        lambda value: (value.continuation, value.temporal_mesh.initial_time),
        solution,
        (continuation, time),
    )
    return eqx.tree_at(
        lambda value: (
            value.times,
            value.states,
            value.state_rates,
            value.valid,
            value.rate_valid,
            value.status,
            value.initialization,
            value.events,
            value.termination_status,
            value.residual_norm,
            value.residual_threshold,
            value.differential_residual_norm,
            value.constraint_norm,
        ),
        solution,
        (
            jnp.full_like(native_prepared.time_grid.times, jnp.inf).at[0].set(time),
            states,
            rates,
            valid,
            jnp.zeros_like(states, dtype=bool).at[0].set(initialization.rate_valid),
            status,
            initialization,
            events,
            termination,
            jnp.zeros((count,), state.real.dtype).at[0].set(initialization.residual_norm),
            jnp.zeros((count,), state.real.dtype)
            .at[0]
            .set(initialization.residual_threshold),
            jnp.zeros((count,), state.real.dtype)
            .at[0]
            .set(initialization.differential_residual_norm),
            jnp.zeros((count,), state.real.dtype)
            .at[0]
            .set(initialization.constraint_norm),
        ),
        is_leaf=lambda value: value is None,
    )


def _solve_dae_segments(prepared, problem, initial_state, runtime, /):
    solve_plan = prepared.plan.native_solve_plan
    boundaries = np.asarray(prepared.transition_times_s)
    requested = np.asarray(prepared.integration_time_grid.times)
    sample_count = requested.size
    state = jnp.asarray(initial_state)
    rate = problem.initial_state_rate
    differential = problem.system.structure.differential_variable_mask(state.shape)
    sample_fields = {
        "times": jnp.full_like(prepared.integration_time_grid.times, jnp.inf),
        "states": jnp.zeros((sample_count,) + state.shape, state.dtype),
        "state_rates": jnp.zeros((sample_count,) + state.shape, state.dtype),
        "valid": jnp.zeros((sample_count,), dtype=bool),
        "rate_valid": jnp.zeros((sample_count,) + state.shape, dtype=bool),
        "status": jnp.full((sample_count,), int(DAEStatus.NOT_RUN), dtype=jnp.int32),
    }
    forcing_currents = jnp.zeros((sample_count,), state.real.dtype)
    for name in (
        "residual_norm",
        "residual_threshold",
        "differential_residual_norm",
        "constraint_norm",
    ):
        sample_fields[name] = jnp.zeros((sample_count,), state.real.dtype)
    segments, restarts, active_segments = [], [], []
    active = jnp.asarray(True)
    successful = jnp.asarray(True)
    termination_status = jnp.asarray(int(DAETerminationStatus.SUCCESS), dtype=jnp.int32)
    for index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        local_times = requested[(requested >= start) & (requested <= end)]
        current = runtime.current(jnp.asarray(0.5 * (start + end)), state)
        local_policy = HeldInputPolicy(
            jnp.asarray((start, end), dtype=prepared.plan.save_times_s.dtype),
            jnp.reshape(current, (1, 1)),
            input_layout=runtime.input_policy.input_layout,
            node_side="right",
            policy_id=f"{runtime.input_policy.policy_id}:protocol-segment:{index}",
        )
        local_runtime = BatteryRuntimeInputs(
            runtime.parameters,
            local_policy,
            runtime.stop_thresholds,
            protocol_id=runtime.protocol_id,
            observation_input_policy=local_policy,
        )
        local_problem = prepared.plan.model.problem(
            prepared.prepared_model, state, local_runtime
        )
        if (
            not isinstance(local_problem, DifferentialAlgebraicProblem)
            or local_problem.args is not local_runtime
        ):
            raise TypeError("DAE adapter must retain the segment BatteryRuntimeInputs.")
        if local_problem.system.system_id != problem.system.system_id:
            raise ValueError("Battery DAE segments must use the identical model system.")
        spec = local_problem.initialization
        if spec.mode != "index-one":
            if spec.mode != "custom":
                raise ValueError(
                    "Battery DAE initialization must fix differential states and solve rates."
                )
            fixed_state = jnp.asarray(spec.fixed_state).reshape(state.shape)
            fixed_rate = jnp.asarray(spec.fixed_rate).reshape(state.shape)
            state = eqx.error_if(
                state,
                jnp.any(differential & ((~fixed_state) | fixed_rate)),
                "Battery DAE custom initialization must fix every differential state and free its rate.",
            )
        local_problem = eqx.tree_at(
            lambda value: (value.initial_state, value.initial_state_rate),
            local_problem,
            (state, rate),
        )
        event_plan, _ = _dae_event_plan(prepared, index)
        native_prepared = prepare_dae(
            local_problem,
            TimeGrid(
                jnp.asarray(local_times),
                time_id=f"{prepared.time_grid.time_id}:segment:{index}",
            ),
            policy=solve_plan.policy,
            event_plan=event_plan,
        )

        def run_segment():
            initialization = initialize_dae(
                local_problem,
                jnp.asarray(start),
                policy=solve_plan.policy,
                args=local_runtime,
            )
            unchanged = jnp.all(
                jnp.where(differential, initialization.state == state, True)
            )
            initialization = eqx.tree_at(
                lambda value: value.valid,
                initialization,
                initialization.valid & unchanged,
            )
            immediate = jnp.asarray(False)
            if event_plan is not None:
                margins = _initial_guard_margins(
                    event_plan, jnp.asarray(start), initialization, local_runtime
                )
                immediate = jnp.any(jnp.isfinite(margins) & (margins <= 0.0))
            solution = jax.lax.cond(
                initialization.valid & (~immediate),
                lambda: solve_dae(
                    native_prepared,
                    args=local_runtime,
                    initial_state=initialization.state,
                    initial_state_rate=initialization.state_rate,
                ),
                lambda: _initial_dae_solution(
                    native_prepared,
                    local_runtime,
                    initialization,
                    event_plan,
                ),
            )
            return solution, initialization, unchanged

        shape = eqx.filter_eval_shape(run_segment)
        neutral = jax.tree.map(
            lambda value: jnp.zeros(value.shape, value.dtype)
            if isinstance(value, jax.ShapeDtypeStruct)
            else value,
            shape,
        )
        local_solution, initialization, unchanged = jax.lax.cond(
            active, lambda: run_segment(), lambda: neutral
        )
        active_segments.append(active)
        segments.append(local_solution)
        if index == 0:
            initial_initialization = initialization
        if index:
            restarts.append(
                BatteryDAERestartEvidence(
                    jnp.asarray(start),
                    state,
                    rate,
                    initialization,
                    unchanged,
                    active,
                    index,
                    local_policy.policy_id,
                )
            )
        owns = ((requested >= start) if index == 0 else (requested > start)) & (
            requested <= end
        )
        if prepared.plan.protocol.node_side == "right" and index:
            owns = (requested >= start) & (requested <= end)
        global_indices_host = np.nonzero(owns)[0]
        global_indices = jnp.asarray(global_indices_host, dtype=jnp.int32)
        local_indices = jnp.asarray(
            np.searchsorted(local_times, requested[global_indices_host]), dtype=jnp.int32
        )
        forcing_currents = forcing_currents.at[global_indices].set(
            jnp.where(active, current, forcing_currents[global_indices])
        )
        for name, target in sample_fields.items():
            source = getattr(local_solution, name)[local_indices]
            existing = target[global_indices]
            sample_fields[name] = target.at[global_indices].set(
                jnp.where(active, source, existing)
            )
        successful = successful & ((~active) | local_solution.successful)
        termination_status = jnp.where(
            active, local_solution.termination_status, termination_status
        )
        continuing = (
            active
            & local_solution.successful
            & (local_solution.termination_status == int(DAETerminationStatus.SUCCESS))
        )
        state = jnp.where(continuing, local_solution.states[-1], state)
        rate = jnp.where(continuing, local_solution.state_rates[-1], rate)
        active = continuing
    segment_active = jnp.stack(active_segments)
    replay = BatteryDAEReplayEvidence(
        initial_initialization,
        tuple(segment.replay for segment in segments),
        tuple(segment.events for segment in segments),
        tuple(restarts),
        segment_active,
        successful,
        canonical_fingerprint(
            {
                "kind": "battery-segmented-dae-replay",
                "preparation_id": prepared.preparation_id,
                "solve_plan_id": solve_plan.solve_plan_id,
                "segment_prepared_ids": [segment.prepared_id for segment in segments],
                "restart_input_ids": [restart.input_policy_id for restart in restarts],
            }
        ),
        solve_plan.restart_policy,
    )
    return BatteryDAESolution(
        **sample_fields,
        forcing_currents_a=forcing_currents,
        segments=tuple(segments),
        replay=replay,
        termination_status=termination_status,
        problem_id=problem.problem_id,
    )


def _contains_tracer(tree: PyTree, /) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(tree))


def _concrete_value_digest(tree: PyTree, /) -> str:
    return str(array_tree_fingerprint(tree)["sha256"])


def run_battery_experiment(
    prepared: PreparedBatteryExperiment,
    parameters: PyTree,
    initial_condition: PyTree,
    protocol_values: BatteryProtocolValues,
    /,
) -> BatteryExperimentResult:
    """Execute one dynamic candidate run through native ODE or DAE solvers."""
    if not isinstance(prepared, PreparedBatteryExperiment):
        raise TypeError("prepared must be PreparedBatteryExperiment.")
    admission = validate_battery_execution_admission(
        prepared.plan.capability_profile,
        prepared.plan.support_tuple,
        prepared.admission,
        distribution_id=prepared.distribution_id,
    )
    prepared.plan.protocol._validate_values(protocol_values)
    if admission is not None:
        admission.validate_adapter(prepared.plan.model)
        admission.validate_run(
            prepared.prepared_model,
            parameters,
            initial_condition,
            prepared.plan.protocol,
            protocol_values,
        )
    forcing_policy = prepared.plan.protocol.input_policy(
        protocol_values, node_side="right"
    )
    observation_policy = prepared.plan.protocol.input_policy(protocol_values)
    runtime = BatteryRuntimeInputs(
        parameters,
        forcing_policy,
        protocol_values.stop_thresholds,
        protocol_id=prepared.plan.protocol.protocol_id,
        observation_input_policy=observation_policy,
    )
    initial_state = prepared.plan.model.initial_state(
        prepared.prepared_model, parameters, initial_condition
    )
    problem = _validated_problem(prepared, initial_state, runtime)
    if isinstance(problem, DifferentialProblem):
        solve_plan = prepared.plan.native_solve_plan
        if not isinstance(solve_plan, BatteryDiffraxSolvePlan):
            raise TypeError(
                "Prepared ODE battery experiment lost its Diffrax solve plan."
            )
        if isinstance(prepared.stepsize_controller, dfx.StepTo):
            native_solution = _solve_fixed_segments(
                prepared, problem, initial_state, solve_plan, runtime
            )
        else:
            event = _event(prepared)
            native_solution = solve_diffrax(
                problem,
                save_times=prepared.integration_time_grid.times,
                solver=solve_plan.solver,
                stepsize_controller=prepared.stepsize_controller,
                adjoint=solve_plan.adjoint,
                dt0=solve_plan.dt0,
                event=event,
                rtol=solve_plan.relative_tolerance,
                atol=solve_plan.absolute_tolerance,
                dense=event is not None,
                max_steps=solve_plan.maximum_steps,
                throw=False,
                solver_configuration_id=solve_plan.solve_plan_id,
                state_coordinates=_identity_coordinates(initial_state),
            )
        native_success = jnp.asarray(native_solution.backend_successful, dtype=bool)
    else:
        if not isinstance(prepared.plan.native_solve_plan, BatteryDAESolvePlan):
            raise TypeError("Prepared DAE battery experiment lost BatteryDAESolvePlan.")
        native_solution = _solve_dae_segments(prepared, problem, initial_state, runtime)
        native_success = jnp.asarray(native_solution.successful, dtype=bool)
    termination = _termination(prepared, native_solution, runtime)
    outputs, domain_ok, finite_ok = _selected_outputs(
        prepared, initial_state, runtime, native_solution, termination
    )
    ledger = prepared.plan.model.ledger(prepared.prepared_model, native_solution, runtime)
    if not hasattr(ledger, "successful"):
        raise TypeError("Every battery model ledger must expose scalar successful.")
    ledger_success = jnp.asarray(ledger.successful)
    if ledger_success.shape != () or ledger_success.dtype != jnp.dtype(bool):
        raise ValueError("Battery ledger successful must be one scalar Boolean.")
    native_failure_status = jnp.asarray(int(BatteryRunStatus.NATIVE_SOLVE_FAILED))
    if isinstance(native_solution, BatteryDAESolution):
        domain_failure = jnp.any(
            jnp.stack(
                tuple(
                    active
                    & (
                        segment.initialization.status
                        == int(DAEInitializationStatus.DOMAIN_FAILURE)
                    )
                    for active, segment in zip(
                        native_solution.replay.segment_active,
                        native_solution.segments,
                        strict=True,
                    )
                )
            )
        )
        native_failure_status = jnp.where(
            domain_failure, int(BatteryRunStatus.DOMAIN_ERROR), native_failure_status
        )
        capacity = native_solution.termination_status == int(
            DAETerminationStatus.EVENT_CAPACITY_EXCEEDED
        )
        native_failure_status = jnp.where(
            capacity, int(BatteryRunStatus.CAPACITY_EXCEEDED), native_failure_status
        )
    status = jnp.where(
        ~native_success,
        native_failure_status,
        jnp.where(
            ~domain_ok,
            int(BatteryRunStatus.DOMAIN_ERROR),
            jnp.where(
                ~finite_ok,
                int(BatteryRunStatus.NONFINITE_OUTPUT),
                jnp.where(
                    ledger_success,
                    int(BatteryRunStatus.SUCCESS),
                    int(BatteryRunStatus.MODEL_LEDGER_FAILED),
                ),
            ),
        ),
    ).astype(jnp.int32)
    protocol_value_tree = {
        "current_amplitudes_a": protocol_values.current_amplitudes_a,
        "stop_thresholds": protocol_values.stop_thresholds,
    }
    traced_run = any(
        _contains_tracer(tree)
        for tree in (parameters, initial_condition, protocol_value_tree, initial_state)
    )
    if traced_run:
        protocol_values_digest = None
        initial_state_digest = None
        parameters_digest = None
    else:
        protocol_values_digest = _concrete_value_digest(protocol_value_tree)
        initial_state_digest = _concrete_value_digest(initial_state)
        parameters_digest = _concrete_value_digest(parameters)
    run_id = (
        None
        if traced_run
        else canonical_fingerprint(
            {
                "kind": "battery-experiment-run",
                "preparation_id": prepared.preparation_id,
                "problem_id": problem.problem_id,
                "protocol_id": prepared.plan.protocol.protocol_id,
                "profile_id": prepared.plan.capability_profile.profile_id,
                "support_tuple_id": prepared.plan.support_tuple.support_tuple_id,
                "protocol_values_digest": protocol_values_digest,
                "initial_state_digest": initial_state_digest,
                "parameters_digest": parameters_digest,
                "numerical_admission_id": None
                if admission is None
                else admission.numerical_admission_id,
                "predictive_admission_id": None
                if admission is None
                else admission.predictive_admission_id,
                "replay_evidence_digest": (
                    _concrete_value_digest(native_solution.replay)
                    if isinstance(native_solution, BatteryDAESolution)
                    else None
                ),
            }
        )
    )
    return BatteryExperimentResult(
        native_solution,
        outputs,
        ledger,
        status,
        termination,
        run_id=run_id,
        protocol_values_digest=protocol_values_digest,
        initial_state_digest=initial_state_digest,
        parameters_digest=parameters_digest,
        experiment_plan_id=prepared.plan.experiment_plan_id,
        preparation_id=prepared.preparation_id,
        protocol_id=prepared.plan.protocol.protocol_id,
        model_id=prepared.plan.model.model_id,
        profile_id=prepared.plan.capability_profile.profile_id,
        support_tuple_id=prepared.plan.support_tuple.support_tuple_id,
        problem_id=problem.problem_id,
        numerical_admission_id=None
        if admission is None
        else admission.numerical_admission_id,
        predictive_admission_id=None
        if admission is None
        else admission.predictive_admission_id,
        distribution_id=prepared.distribution_id,
        validity_envelope_id=None if admission is None else admission.envelope_id,
    )


__all__ = [
    "AbstractBatteryModelAdapter",
    "BatteryDAESolvePlan",
    "BatteryDiffraxSolvePlan",
    "BatteryExperimentPlan",
    "BatteryOutputPlan",
    "BatteryRuntimeInputs",
    "PreparedBatteryExperiment",
    "prepare_battery_experiment",
    "run_battery_experiment",
]
