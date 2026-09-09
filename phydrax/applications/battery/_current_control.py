#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...control import (
    ControlProblem,
    ControlResult,
    DifferentialControlDynamics,
    evaluate_sampled_cost,
    evaluate_sampled_feasibility,
    PiecewiseConstantControlParameterization,
)
from ...dynamics import ContinuousSystem, HeldInputPolicy, StateLayout, TimeGrid
from ...solver import DifferentialProblem
from ._experiment import (
    BatteryDiffraxSolvePlan,
    BatteryRuntimeInputs,
    PreparedBatteryExperiment,
)
from ._protocol import (
    BATTERY_CURRENT_INPUT_LAYOUT,
    BatteryProtocolValues,
    CurrentStepPlan,
    PASSIVE_TERMINAL_CONVENTION,
)
from ._results import BatteryExperimentResult, BatteryRunStatus


class BatteryCurrentControlReplayStatus(IntEnum):
    """Fail-closed disposition of an independent battery control replay."""

    SUCCESS = 0
    RUNTIME_FAILED = 1
    LEDGER_FAILED = 2
    NONFINITE_REPLAY = 3
    PATH_CONSTRAINT_VIOLATED = 4
    TERMINAL_TARGET_MISSED = 5


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{owner} must be a non-empty canonical identifier.")
    return value


def _real_scalar(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValueError(f"{owner} must be one real scalar.")
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _ordered_bounds(lower: float, upper: float, owner: str, /) -> tuple[float, float]:
    lower_, upper_ = float(lower), float(upper)
    if not isfinite(lower_) or not isfinite(upper_) or lower_ > upper_:
        raise ValueError(f"{owner} bounds must be finite and ordered.")
    return lower_, upper_


def _finite_residual(value: Array, bound: float, side: int, /) -> Array:
    residual = (
        (jnp.asarray(bound, dtype=value.dtype) - value)
        if side < 0
        else (value - jnp.asarray(bound, dtype=value.dtype))
    )
    return jnp.where(jnp.isfinite(value), residual, jnp.asarray(jnp.inf, value.dtype))


class BatteryCurrentControlBounds(StrictModule, NonTrainableState):
    """Declared SI path bounds for one fixed-topology current-control problem."""

    minimum_current_a: float = eqx.field(static=True)
    maximum_current_a: float = eqx.field(static=True)
    minimum_voltage_v: float = eqx.field(static=True)
    maximum_voltage_v: float = eqx.field(static=True)
    minimum_stoichiometry: float = eqx.field(static=True)
    maximum_stoichiometry: float = eqx.field(static=True)
    minimum_temperature_k: float = eqx.field(static=True)
    maximum_temperature_k: float = eqx.field(static=True)
    voltage_name: str = eqx.field(static=True)
    stoichiometry_names: tuple[str, ...] = eqx.field(static=True)
    temperature_name: str = eqx.field(static=True)
    bounds_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        current_a: tuple[float, float],
        voltage_v: tuple[float, float],
        stoichiometry: tuple[float, float],
        temperature_k: tuple[float, float],
        voltage_name: str = "voltage_v",
        stoichiometry_names: Sequence[str] = (
            "stoichiometry:negative",
            "stoichiometry:positive",
        ),
        temperature_name: str = "temperature_k",
    ):
        current = _ordered_bounds(*current_a, "Current")
        voltage = _ordered_bounds(*voltage_v, "Voltage")
        stoichiometry_ = _ordered_bounds(*stoichiometry, "Stoichiometry")
        temperature = _ordered_bounds(*temperature_k, "Temperature")
        voltage_name_ = _identifier(voltage_name, "Voltage observable name")
        stoichiometry_names_ = tuple(
            _identifier(name, "Stoichiometry observable name")
            for name in stoichiometry_names
        )
        if not stoichiometry_names_ or len(set(stoichiometry_names_)) != len(
            stoichiometry_names_
        ):
            raise ValueError(
                "stoichiometry_names must be a non-empty sequence of unique names."
            )
        temperature_name_ = _identifier(temperature_name, "Temperature observable name")
        observable_names = (voltage_name_, *stoichiometry_names_, temperature_name_)
        if len(set(observable_names)) != len(observable_names):
            raise ValueError("Voltage, stoichiometry, and temperature names must differ.")
        (
            self.minimum_current_a,
            self.maximum_current_a,
        ) = current
        (
            self.minimum_voltage_v,
            self.maximum_voltage_v,
        ) = voltage
        (
            self.minimum_stoichiometry,
            self.maximum_stoichiometry,
        ) = stoichiometry_
        (
            self.minimum_temperature_k,
            self.maximum_temperature_k,
        ) = temperature
        self.voltage_name = voltage_name_
        self.stoichiometry_names = stoichiometry_names_
        self.temperature_name = temperature_name_
        self.bounds_id = canonical_fingerprint(
            {
                "kind": "battery-current-control-bounds",
                "passive_terminal_convention": PASSIVE_TERMINAL_CONVENTION.convention_id,
                "current_a": list(current),
                "voltage": [voltage_name_, *voltage],
                "stoichiometry": [list(stoichiometry_names_), *stoichiometry_],
                "temperature": [temperature_name_, *temperature],
                "domain_constraint": "adapter-domain-valid-and-finite",
            }
        )

    @property
    def observable_names(self) -> tuple[str, ...]:
        return (
            self.voltage_name,
            *self.stoichiometry_names,
            self.temperature_name,
        )


class BatteryCurrentControlTerminalTarget(StrictModule, NonTrainableState):
    """Exact equality target for one flattened terminal state coordinate."""

    value: float = eqx.field(static=True)
    state_index: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        /,
        *,
        state_index: int,
        name: str,
    ):
        value_array = _real_scalar(value, "Terminal target")
        value_ = float(np.asarray(value_array))
        if not isfinite(value_):
            raise ValueError("Terminal target must be finite.")
        if isinstance(state_index, bool) or int(state_index) != state_index:
            raise TypeError("Terminal target state_index must be an integer.")
        index = int(state_index)
        if index < 0:
            raise ValueError("Terminal target state_index must be nonnegative.")
        name_ = _identifier(name, "Terminal target name")
        self.value = value_
        self.state_index = index
        self.name = name_
        self.target_id = canonical_fingerprint(
            {
                "kind": "battery-current-control-terminal-target",
                "name": name_,
                "state_index": index,
                "value": value_,
                "unit": "model-state-coordinate",
                "relation": "equality",
            }
        )

    def coordinate(self, state: ArrayLike, /) -> Array:
        return jnp.asarray(state).reshape((-1,))[self.state_index]


class BatteryCurrentControlObjective(StrictModule, NonTrainableState):
    """Explicit current-effort and terminal-target objective terms."""

    current_squared_weight: float = eqx.field(static=True)
    terminal_target_squared_weight: float = eqx.field(static=True)
    objective_id: str = eqx.field(static=True)
    term_names: tuple[str, str] = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        current_squared_weight: float,
        terminal_target_squared_weight: float = 0.0,
    ):
        current_weight = float(current_squared_weight)
        terminal_weight = float(terminal_target_squared_weight)
        if any(
            not isfinite(value) or value < 0.0
            for value in (current_weight, terminal_weight)
        ):
            raise ValueError(
                "Battery current-control objective weights must be finite and nonnegative."
            )
        if current_weight == 0.0 and terminal_weight == 0.0:
            raise ValueError(
                "At least one battery current-control objective term must be active."
            )
        names = (
            "integral-current-squared-a2-s",
            "terminal-target-residual-squared",
        )
        self.current_squared_weight = current_weight
        self.terminal_target_squared_weight = terminal_weight
        self.term_names = names
        self.objective_id = canonical_fingerprint(
            {
                "kind": "battery-current-control-objective",
                "terms": [
                    {"name": names[0], "weight": current_weight},
                    {"name": names[1], "weight": terminal_weight},
                ],
            }
        )


class BatteryPiecewiseCurrent(StrictModule):
    """Dynamic passive-current amplitudes bound to immutable physical knots."""

    amplitudes_a: Array
    knot_times_s: tuple[float, ...] = eqx.field(static=True)
    lowering_id: str = eqx.field(static=True)

    def __init__(
        self,
        amplitudes_a: ArrayLike,
        /,
        *,
        knot_times_s: tuple[float, ...],
        lowering_id: str,
    ):
        amplitudes = jnp.asarray(amplitudes_a)
        if jnp.issubdtype(amplitudes.dtype, jnp.complexfloating):
            raise TypeError("Battery current amplitudes must be real-valued.")
        amplitudes = (
            amplitudes
            if jnp.issubdtype(amplitudes.dtype, jnp.inexact)
            else amplitudes.astype(float)
        )
        expected = (len(knot_times_s) - 1,)
        if amplitudes.shape != expected:
            raise ValueError(
                f"Battery current amplitudes must have shape {expected}; got {amplitudes.shape}."
            )
        self.amplitudes_a = eqx.error_if(
            amplitudes,
            jnp.any(~jnp.isfinite(amplitudes)),
            "Battery current amplitudes must be finite.",
        )
        self.knot_times_s = knot_times_s
        self.lowering_id = _identifier(lowering_id, "Current lowering ID")

    @property
    def coefficients(self) -> Array:
        return self.amplitudes_a[:, None]


class _BatteryCurrentControlArgs(StrictModule):
    parameters: PyTree[Any]


class _BatteryControlledVectorField(StrictModule, NonTrainableState):
    prepared: PreparedBatteryExperiment
    drift: Callable[..., Any]
    runtime_policy_id: str = eqx.field(static=True)

    def __call__(
        self,
        time_s: Array,
        state: Array,
        current_a: Array,
        args: _BatteryCurrentControlArgs,
        /,
    ) -> Array:
        runtime = _single_current_runtime(
            self.prepared, args.parameters, current_a, self.runtime_policy_id
        )
        return jnp.asarray(self.drift(time_s, state, runtime))


class _BatteryArrayObserver(StrictModule, NonTrainableState):
    prepared: PreparedBatteryExperiment
    runtime_policy_id: str = eqx.field(static=True)

    def __call__(
        self,
        time_s: Array,
        state: Array,
        current_a: Array,
        args: _BatteryCurrentControlArgs,
        /,
    ) -> tuple[Array, Array]:
        runtime = _single_current_runtime(
            self.prepared, args.parameters, current_a, self.runtime_policy_id
        )
        observation = self.prepared.plan.model.observe(
            self.prepared.prepared_model,
            time_s.reshape((1,)),
            jnp.expand_dims(state, axis=0),
            runtime,
        )
        expected = (1, len(self.prepared.plan.model.observable_names))
        if observation.values.shape != expected or observation.domain_valid.shape != (1,):
            raise ValueError(
                "Battery adapter single-state observation does not match its declared observable topology."
            )
        return observation.values[0], observation.domain_valid[0]


class _CurrentBoundConstraint(StrictModule, NonTrainableState):
    bound: float = eqx.field(static=True)
    side: int = eqx.field(static=True)

    def __call__(self, time: Array, state: Array, control: Array, args: Any, /) -> Array:
        del time, state, args
        return _finite_residual(control[0], self.bound, self.side)


class _ObservableBoundConstraint(StrictModule, NonTrainableState):
    observer: _BatteryArrayObserver
    observable_index: int = eqx.field(static=True)
    bound: float = eqx.field(static=True)
    side: int = eqx.field(static=True)

    def __call__(
        self,
        time: Array,
        state: Array,
        control: Array,
        args: _BatteryCurrentControlArgs,
        /,
    ) -> Array:
        values, domain_valid = self.observer(time, state, control, args)
        value = values[self.observable_index]
        residual = _finite_residual(value, self.bound, self.side)
        return jnp.where(domain_valid, residual, jnp.asarray(jnp.inf, residual.dtype))


class _DomainConstraint(StrictModule, NonTrainableState):
    observer: _BatteryArrayObserver

    def __call__(
        self,
        time: Array,
        state: Array,
        control: Array,
        args: _BatteryCurrentControlArgs,
        /,
    ) -> Array:
        values, domain_valid = self.observer(time, state, control, args)
        valid = domain_valid & jnp.all(jnp.isfinite(values))
        return jnp.where(
            valid, -jnp.ones((), values.dtype), jnp.asarray(jnp.inf, values.dtype)
        )


class _TerminalTargetConstraint(StrictModule, NonTrainableState):
    target: BatteryCurrentControlTerminalTarget
    side: int = eqx.field(static=True)

    def __call__(self, time: Array, state: Array, args: Any, /) -> Array:
        del time, args
        difference = self.target.coordinate(state) - self.target.value
        residual = difference if self.side > 0 else -difference
        return jnp.where(
            jnp.isfinite(difference),
            residual,
            jnp.asarray(jnp.inf, difference.dtype),
        )


class _RunningCurrentObjective(StrictModule, NonTrainableState):
    weight: float = eqx.field(static=True)

    def __call__(self, time: Array, state: Array, control: Array, args: Any, /) -> Array:
        del time, state, args
        return jnp.asarray(self.weight, dtype=control.dtype) * control[0] ** 2


class _TerminalTargetObjective(StrictModule, NonTrainableState):
    target: BatteryCurrentControlTerminalTarget
    weight: float = eqx.field(static=True)

    def __call__(self, time: Array, state: Array, args: Any, /) -> Array:
        del time, args
        difference = self.target.coordinate(state) - self.target.value
        return jnp.asarray(self.weight, dtype=difference.dtype) * difference**2


def _single_current_runtime(
    prepared: PreparedBatteryExperiment,
    parameters: PyTree[Any],
    current_a: Array,
    runtime_policy_id: str,
    /,
) -> BatteryRuntimeInputs:
    protocol = prepared.plan.protocol
    values = jnp.broadcast_to(current_a, (len(protocol.steps), 1))
    policy = HeldInputPolicy(
        protocol.boundary_times_s,
        values,
        input_layout=BATTERY_CURRENT_INPUT_LAYOUT,
        node_side="right",
        policy_id=runtime_policy_id,
    )
    observation_policy = HeldInputPolicy(
        protocol.boundary_times_s,
        values,
        input_layout=BATTERY_CURRENT_INPUT_LAYOUT,
        node_side=protocol.node_side,
        policy_id=f"{runtime_policy_id}:observation",
    )
    return BatteryRuntimeInputs(
        parameters,
        policy,
        jnp.empty((0,), dtype=values.dtype),
        protocol_id=protocol.protocol_id,
        observation_input_policy=observation_policy,
    )


def _path_topology(
    observer: _BatteryArrayObserver,
    bounds: BatteryCurrentControlBounds,
    model_names: tuple[str, ...],
    /,
) -> tuple[tuple[Callable[..., Array], ...], tuple[str, ...]]:
    constraints: list[Callable[..., Array]] = [
        _CurrentBoundConstraint(bounds.minimum_current_a, -1),
        _CurrentBoundConstraint(bounds.maximum_current_a, 1),
    ]
    names = ["current-a:lower", "current-a:upper"]
    declared = (
        (bounds.voltage_name, bounds.minimum_voltage_v, bounds.maximum_voltage_v),
        *(
            (
                name,
                bounds.minimum_stoichiometry,
                bounds.maximum_stoichiometry,
            )
            for name in bounds.stoichiometry_names
        ),
        (
            bounds.temperature_name,
            bounds.minimum_temperature_k,
            bounds.maximum_temperature_k,
        ),
    )
    for name, lower, upper in declared:
        index = model_names.index(name)
        constraints.extend(
            (
                _ObservableBoundConstraint(observer, index, lower, -1),
                _ObservableBoundConstraint(observer, index, upper, 1),
            )
        )
        names.extend((f"{name}:lower", f"{name}:upper"))
    constraints.append(_DomainConstraint(observer))
    names.append("model-domain")
    return tuple(constraints), tuple(names)


class BatteryCurrentControlPlan(StrictModule, NonTrainableState):
    """Fixed-horizon current-control construction over one prepared experiment."""

    experiment: PreparedBatteryExperiment
    parameters: PyTree[Any]
    initial_condition: PyTree[Any]
    bounds: BatteryCurrentControlBounds
    terminal_target: BatteryCurrentControlTerminalTarget
    objective: BatteryCurrentControlObjective
    ledger_success: Callable[[Any], ArrayLike]
    parameter_source_id: str = eqx.field(static=True)
    ledger_criterion_id: str = eqx.field(static=True)
    replay_constraint_tolerance: float = eqx.field(static=True)
    control_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        experiment: PreparedBatteryExperiment,
        parameters: PyTree[Any],
        initial_condition: PyTree[Any],
        bounds: BatteryCurrentControlBounds,
        terminal_target: BatteryCurrentControlTerminalTarget,
        objective: BatteryCurrentControlObjective,
        /,
        *,
        parameter_source_id: str,
        ledger_success: Callable[[Any], ArrayLike],
        ledger_criterion_id: str,
        replay_constraint_tolerance: float = 1.0e-6,
    ):
        if not isinstance(experiment, PreparedBatteryExperiment):
            raise TypeError("experiment must be a PreparedBatteryExperiment.")
        if not isinstance(bounds, BatteryCurrentControlBounds):
            raise TypeError("bounds must be BatteryCurrentControlBounds.")
        if not isinstance(terminal_target, BatteryCurrentControlTerminalTarget):
            raise TypeError(
                "terminal_target must be BatteryCurrentControlTerminalTarget."
            )
        if not isinstance(objective, BatteryCurrentControlObjective):
            raise TypeError("objective must be BatteryCurrentControlObjective.")
        if not callable(ledger_success):
            raise TypeError("ledger_success must be callable.")
        parameter_source_id_ = _identifier(parameter_source_id, "Parameter source ID")
        ledger_criterion_id_ = _identifier(ledger_criterion_id, "Ledger criterion ID")
        tolerance = float(replay_constraint_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "Replay constraint tolerance must be finite and nonnegative."
            )
        protocol = experiment.plan.protocol
        if protocol.guard_count:
            raise ValueError(
                "Battery current control requires a smooth fixed-horizon protocol without stop guards."
            )
        if any(not isinstance(step, CurrentStepPlan) for step in protocol.steps):
            raise ValueError(
                "Every fixed current-control phase must be a CurrentStepPlan."
            )
        if experiment.plan.model.equation_form != "ode" or not isinstance(
            experiment.plan.native_solve_plan, BatteryDiffraxSolvePlan
        ):
            raise TypeError(
                "Battery ControlProblem lowering requires a prepared smooth ODE experiment."
            )
        model_names = tuple(experiment.plan.model.observable_names)
        missing = tuple(
            name for name in bounds.observable_names if name not in model_names
        )
        if missing:
            raise ValueError(
                f"Battery model does not declare required control observables {missing}."
            )
        replay_names = tuple(experiment.plan.outputs.names)
        replay_required = ("current_a", *bounds.observable_names)
        replay_missing = tuple(
            name for name in replay_required if name not in replay_names
        )
        if replay_missing:
            raise ValueError(
                "Independent replay output selection is missing required observables "
                f"{replay_missing}."
            )
        phase_times = np.asarray(protocol.boundary_times_s, dtype=float)
        replay_times = np.asarray(experiment.plan.save_times_s, dtype=float)
        interior_per_phase = tuple(
            np.any((replay_times > left) & (replay_times < right))
            for left, right in zip(phase_times[:-1], phase_times[1:], strict=True)
        )
        if not all(interior_per_phase):
            raise ValueError(
                "Independent replay requires at least one finer interior save time per phase."
            )
        self.experiment = experiment
        self.parameters = parameters
        self.initial_condition = initial_condition
        self.bounds = bounds
        self.terminal_target = terminal_target
        self.objective = objective
        self.ledger_success = ledger_success
        self.parameter_source_id = parameter_source_id_
        self.ledger_criterion_id = ledger_criterion_id_
        self.replay_constraint_tolerance = tolerance
        self.control_plan_id = canonical_fingerprint(
            {
                "kind": "battery-fixed-horizon-current-control-plan",
                "preparation_id": experiment.preparation_id,
                "experiment_plan_id": experiment.plan.experiment_plan_id,
                "model_id": experiment.plan.model.model_id,
                "protocol_id": protocol.protocol_id,
                "profile_id": experiment.plan.capability_profile.profile_id,
                "support_tuple_id": experiment.plan.support_tuple.support_tuple_id,
                "parameter_source_id": parameter_source_id_,
                "parameters": array_tree_fingerprint(parameters),
                "initial_condition": array_tree_fingerprint(initial_condition),
                "bounds_id": bounds.bounds_id,
                "target_id": terminal_target.target_id,
                "objective_id": objective.objective_id,
                "ledger_criterion_id": ledger_criterion_id_,
                "replay_constraint_tolerance": tolerance,
                "horizon_s": [phase_times[0], phase_times[-1]],
                "phase_times_s": phase_times.tolist(),
                "replay_times_s": replay_times.tolist(),
                "variable_duration": False,
                "terminal_convention": PASSIVE_TERMINAL_CONVENTION.convention_id,
            }
        )

    def prepare(self, /) -> "PreparedBatteryCurrentControl":
        return prepare_battery_current_control(self)


class BatteryCurrentControlReplayEvidence(StrictModule):
    """Independent finer-grid causal replay and fail-closed feasibility evidence."""

    experiment_result: BatteryExperimentResult
    path_residuals: Array
    terminal_residuals: Array
    execution_residuals: Array
    objective_terms: Array
    objective: Array
    maximum_violation: Array
    feasible: Array
    status: Array
    path_constraint_names: tuple[str, ...] = eqx.field(static=True)
    terminal_constraint_names: tuple[str, str] = eqx.field(static=True)
    execution_constraint_names: tuple[str, str, str] = eqx.field(static=True)
    objective_term_names: tuple[str, str] = eqx.field(static=True)
    control_plan_id: str = eqx.field(static=True)
    replay_plan_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    parameter_source_id: str = eqx.field(static=True)
    ledger_criterion_id: str = eqx.field(static=True)
    experiment_plan_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.feasible


class PreparedBatteryCurrentControl(StrictModule, NonTrainableState):
    """Native control problem, lowering, and independent replay binding."""

    plan: BatteryCurrentControlPlan
    initial_state: Array
    phase_time_grid: TimeGrid
    parameterization: PiecewiseConstantControlParameterization
    control_problem: ControlProblem
    replay_output_indices: Array
    path_constraint_names: tuple[str, ...] = eqx.field(static=True)
    terminal_constraint_names: tuple[str, str] = eqx.field(static=True)
    replay_plan_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)
    lowering_id: str = eqx.field(static=True)

    @property
    def current_knot_times_s(self) -> Array:
        return self.phase_time_grid.times

    @property
    def horizon_s(self) -> Array:
        return self.phase_time_grid.t1 - self.phase_time_grid.t0

    def lower(self, amplitudes_a: ArrayLike, /) -> BatteryPiecewiseCurrent:
        return BatteryPiecewiseCurrent(
            amplitudes_a,
            knot_times_s=tuple(
                float(value) for value in np.asarray(self.phase_time_grid.times)
            ),
            lowering_id=self.lowering_id,
        )

    def protocol_values(self, amplitudes_a: ArrayLike, /) -> BatteryProtocolValues:
        current = self.lower(amplitudes_a)
        return BatteryProtocolValues(
            self.plan.experiment.plan.protocol,
            current.amplitudes_a,
            jnp.empty((0,), dtype=current.amplitudes_a.dtype),
        )

    def evaluate(self, amplitudes_a: ArrayLike, /) -> ControlResult:
        current = self.lower(amplitudes_a)
        solve_plan = self.plan.experiment.plan.native_solve_plan
        if not isinstance(solve_plan, BatteryDiffraxSolvePlan):
            raise RuntimeError(
                "Prepared battery current control lost its ODE solve plan."
            )
        trajectory = self.control_problem.rollout(
            self.parameterization,
            current.coefficients,
            solver=solve_plan.solver,
            stepsize_controller=self.plan.experiment.stepsize_controller,
            adjoint=solve_plan.adjoint,
            dt0=solve_plan.dt0,
            rtol=solve_plan.relative_tolerance,
            atol=solve_plan.absolute_tolerance,
            max_steps=solve_plan.maximum_steps,
            throw=False,
        )
        sampled_loss = evaluate_sampled_cost(self.control_problem, trajectory)
        feasibility = evaluate_sampled_feasibility(
            self.control_problem,
            trajectory,
            tolerance=self.plan.replay_constraint_tolerance,
        )
        return ControlResult(
            trajectory=trajectory,
            parameters=current.coefficients,
            sampled_loss=sampled_loss,
            feasibility=feasibility,
            result_id=f"control-result:{self.control_problem.problem_id}",
            method_id=trajectory.method_id,
        )

    def replay(self, amplitudes_a: ArrayLike, /) -> BatteryCurrentControlReplayEvidence:
        return replay_battery_current_control(self, amplitudes_a)


def prepare_battery_current_control(
    plan: BatteryCurrentControlPlan, /
) -> PreparedBatteryCurrentControl:
    """Lower a concrete smooth battery experiment to native fixed-horizon control."""

    if not isinstance(plan, BatteryCurrentControlPlan):
        raise TypeError("plan must be BatteryCurrentControlPlan.")
    experiment = plan.experiment
    protocol = experiment.plan.protocol
    dtype = jnp.result_type(protocol.boundary_times_s, plan.terminal_target.value, float)
    zero_values = BatteryProtocolValues(
        protocol,
        jnp.zeros((protocol.current_step_count,), dtype=dtype),
        jnp.empty((0,), dtype=dtype),
    )
    runtime = BatteryRuntimeInputs(
        plan.parameters,
        protocol.input_policy(zero_values, node_side="right"),
        zero_values.stop_thresholds,
        protocol_id=protocol.protocol_id,
        observation_input_policy=protocol.input_policy(zero_values),
    )
    initial_state_value = experiment.plan.model.initial_state(
        experiment.prepared_model,
        plan.parameters,
        plan.initial_condition,
    )
    if not eqx.is_array_like(initial_state_value):
        raise TypeError(
            "Battery ControlProblem lowering requires one concrete array-valued model state."
        )
    initial_state = jnp.asarray(initial_state_value)
    if jnp.issubdtype(initial_state.dtype, jnp.complexfloating):
        raise TypeError("Battery current-control state must be real-valued.")
    if not jnp.issubdtype(initial_state.dtype, jnp.inexact):
        initial_state = initial_state.astype(float)
    if not bool(np.all(np.isfinite(np.asarray(initial_state)))):
        raise ValueError("Battery current-control initial state must be finite.")
    if plan.terminal_target.state_index >= initial_state.size:
        raise ValueError("Terminal target state_index lies outside the model state.")
    native_problem = experiment.plan.model.problem(
        experiment.prepared_model,
        initial_state,
        runtime,
    )
    if not isinstance(native_problem, DifferentialProblem):
        raise TypeError(
            "Battery current-control lowering requires DifferentialProblem dynamics."
        )
    if native_problem.stochastic:
        raise ValueError(
            "Battery current-control lowering requires deterministic smooth dynamics."
        )
    native_bounds = np.asarray((native_problem.t0, native_problem.t1), dtype=float)
    phase_bounds = np.asarray(
        (protocol.boundary_times_s[0], protocol.boundary_times_s[-1]), dtype=float
    )
    if not np.array_equal(native_bounds, phase_bounds):
        raise ValueError(
            "Battery model and fixed control horizon must have identical bounds."
        )

    runtime_policy_id = (
        f"battery-current-control:{plan.control_plan_id}:single-current-runtime"
    )
    vector_field = _BatteryControlledVectorField(
        experiment,
        native_problem.drift,
        runtime_policy_id,
    )
    derivative_sample = vector_field(
        protocol.boundary_times_s[0],
        initial_state,
        jnp.zeros((1,), dtype=initial_state.dtype),
        _BatteryCurrentControlArgs(plan.parameters),
    )
    if derivative_sample.shape != initial_state.shape:
        raise ValueError(
            "Battery model derivative shape does not match its concrete array state."
        )
    state_layout = StateLayout(
        initial_state.shape,
        layout_id=f"battery-current-control:{plan.control_plan_id}:state",
    )
    system = ContinuousSystem(
        vector_field,
        state_layout=state_layout,
        input_layout=BATTERY_CURRENT_INPUT_LAYOUT,
        system_id=f"battery-current-control:{plan.control_plan_id}:system",
    )
    dynamics = DifferentialControlDynamics(
        system,
        method_id="battery-current-control:native-differential",
    )
    phase_grid = TimeGrid(
        protocol.boundary_times_s,
        time_id=f"battery-current-control:{plan.control_plan_id}:phase-time",
    )
    parameterization = PiecewiseConstantControlParameterization(
        phase_grid,
        (1,),
        parameterization_id=f"battery-current-control:{plan.control_plan_id}:piecewise-current",
    )
    observer = _BatteryArrayObserver(experiment, runtime_policy_id)
    model_names = tuple(experiment.plan.model.observable_names)
    path_constraints, path_names = _path_topology(observer, plan.bounds, model_names)
    terminal_constraints = (
        _TerminalTargetConstraint(plan.terminal_target, 1),
        _TerminalTargetConstraint(plan.terminal_target, -1),
    )
    terminal_names = (
        f"terminal:{plan.terminal_target.name}:target-minus-upper",
        f"terminal:{plan.terminal_target.name}:target-minus-lower",
    )
    problem = ControlProblem(
        dynamics,
        phase_grid,
        initial_state,
        running_cost=_RunningCurrentObjective(plan.objective.current_squared_weight),
        terminal_cost=_TerminalTargetObjective(
            plan.terminal_target,
            plan.objective.terminal_target_squared_weight,
        ),
        path_constraints=path_constraints,
        terminal_constraints=terminal_constraints,
        args=_BatteryCurrentControlArgs(plan.parameters),
        problem_id=f"battery-current-control:{plan.control_plan_id}",
    )
    selected_names = tuple(experiment.plan.outputs.names)
    replay_indices = jax.lax.stop_gradient(
        jnp.asarray(
            [
                selected_names.index(name)
                for name in ("current_a", *plan.bounds.observable_names)
            ],
            dtype=jnp.int32,
        )
    )
    replay_plan_id = canonical_fingerprint(
        {
            "kind": "battery-current-control-independent-replay",
            "control_plan_id": plan.control_plan_id,
            "experiment_plan_id": experiment.plan.experiment_plan_id,
            "preparation_id": experiment.preparation_id,
            "native_solve_plan_id": experiment.plan.native_solve_plan.solve_plan_id,
            "save_times_s": np.asarray(
                experiment.plan.save_times_s, dtype=float
            ).tolist(),
            "path_constraints": list(path_names),
            "terminal_constraints": list(terminal_names),
            "ledger_criterion_id": plan.ledger_criterion_id,
            "constraint_tolerance": plan.replay_constraint_tolerance,
            "causal": True,
            "independent_of_collocation_interpolant": True,
        }
    )
    preparation_id = canonical_fingerprint(
        {
            "kind": "prepared-battery-current-control",
            "control_plan_id": plan.control_plan_id,
            "problem_id": problem.problem_id,
            "parameterization_id": parameterization.parameterization_id,
            "replay_plan_id": replay_plan_id,
        }
    )
    return PreparedBatteryCurrentControl(
        plan,
        jax.lax.stop_gradient(initial_state),
        phase_grid,
        parameterization,
        problem,
        replay_indices,
        path_names,
        terminal_names,
        replay_plan_id,
        preparation_id,
        f"battery-current-control:{plan.control_plan_id}:lowering",
    )


def _replay_path_residuals(
    prepared: PreparedBatteryCurrentControl,
    result: BatteryExperimentResult,
    /,
) -> Array:
    bounds = prepared.plan.bounds
    selected = jnp.take(result.outputs.values, prepared.replay_output_indices, axis=-1)
    current = selected[:, 0]
    columns = [
        _finite_residual(current, bounds.minimum_current_a, -1),
        _finite_residual(current, bounds.maximum_current_a, 1),
    ]
    declarations = (
        (bounds.minimum_voltage_v, bounds.maximum_voltage_v),
        *(
            (bounds.minimum_stoichiometry, bounds.maximum_stoichiometry)
            for _ in bounds.stoichiometry_names
        ),
        (bounds.minimum_temperature_k, bounds.maximum_temperature_k),
    )
    for index, (lower, upper) in enumerate(declarations, start=1):
        value = selected[:, index]
        columns.extend(
            (
                _finite_residual(value, lower, -1),
                _finite_residual(value, upper, 1),
            )
        )
    sample_valid = result.outputs.valid & jnp.all(jnp.isfinite(selected), axis=-1)
    columns.append(
        jnp.where(
            sample_valid,
            -jnp.ones_like(current),
            jnp.full_like(current, jnp.inf),
        )
    )
    return jnp.stack(tuple(columns), axis=-1)


def replay_battery_current_control(
    prepared: PreparedBatteryCurrentControl,
    amplitudes_a: ArrayLike,
    /,
) -> BatteryCurrentControlReplayEvidence:
    """Run and audit an independent finer causal replay without failure penalties."""

    if not isinstance(prepared, PreparedBatteryCurrentControl):
        raise TypeError("prepared must be PreparedBatteryCurrentControl.")
    current = prepared.lower(amplitudes_a)
    protocol_values = BatteryProtocolValues(
        prepared.plan.experiment.plan.protocol,
        current.amplitudes_a,
        jnp.empty((0,), dtype=current.amplitudes_a.dtype),
    )
    result = prepared.plan.experiment.run(
        prepared.plan.parameters,
        prepared.plan.initial_condition,
        protocol_values,
    )
    path = _replay_path_residuals(prepared, result)
    states = jnp.asarray(result.native_solution.states)
    expected_states = (
        prepared.plan.experiment.plan.save_times_s.size,
    ) + prepared.initial_state.shape
    if states.shape != expected_states:
        raise ValueError(
            "Independent replay state history does not match the declared save topology."
        )
    terminal_difference = (
        prepared.plan.terminal_target.coordinate(states[-1])
        - prepared.plan.terminal_target.value
    )
    terminal = jnp.stack((terminal_difference, -terminal_difference))
    ledger_success = jnp.asarray(prepared.plan.ledger_success(result.ledger), dtype=bool)
    if ledger_success.shape != ():
        raise ValueError("ledger_success must return one scalar boolean.")
    model_ledger_failed = result.application_status == int(
        BatteryRunStatus.MODEL_LEDGER_FAILED
    )
    ledger_success = ledger_success & ~model_ledger_failed
    output_finite = (
        jnp.all(jnp.isfinite(result.outputs.times_s))
        & jnp.all(jnp.isfinite(result.outputs.values))
        & jnp.all(jnp.isfinite(states))
    )
    runtime_success = (
        jnp.asarray(result.successful, dtype=bool)
        & jnp.all(result.outputs.valid)
        & ~jnp.asarray(result.termination.terminated, dtype=bool)
    )
    execution = jnp.stack(
        (
            jnp.where(runtime_success, -1.0, 1.0),
            jnp.where(ledger_success, -1.0, 1.0),
            jnp.where(output_finite, -1.0, 1.0),
        )
    ).astype(path.dtype)
    durations = prepared.phase_time_grid.durations.astype(current.amplitudes_a.dtype)
    current_term = jnp.asarray(
        prepared.plan.objective.current_squared_weight,
        dtype=current.amplitudes_a.dtype,
    ) * jnp.sum(durations * current.amplitudes_a**2)
    terminal_term = (
        jnp.asarray(
            prepared.plan.objective.terminal_target_squared_weight,
            dtype=terminal_difference.dtype,
        )
        * terminal_difference**2
    )
    objective_terms = jnp.stack((current_term, terminal_term))
    objective = jnp.sum(objective_terms)
    finite_objective = jnp.all(jnp.isfinite(objective_terms))
    execution = execution.at[2].set(
        jnp.where(output_finite & finite_objective, -1.0, 1.0)
    )
    tolerance = prepared.plan.replay_constraint_tolerance
    path_ok = jnp.all(path <= tolerance)
    terminal_ok = jnp.all(terminal <= tolerance)
    execution_ok = jnp.all(execution <= 0.0)
    feasible = path_ok & terminal_ok & execution_ok
    maximum_violation = jnp.max(
        jnp.maximum(
            jnp.concatenate((path.reshape((-1,)), terminal, execution)),
            0.0,
        ),
        initial=jnp.asarray(0.0, dtype=path.dtype),
    )
    status = jnp.where(
        ~runtime_success & ~model_ledger_failed,
        int(BatteryCurrentControlReplayStatus.RUNTIME_FAILED),
        jnp.where(
            ~ledger_success,
            int(BatteryCurrentControlReplayStatus.LEDGER_FAILED),
            jnp.where(
                ~(output_finite & finite_objective),
                int(BatteryCurrentControlReplayStatus.NONFINITE_REPLAY),
                jnp.where(
                    ~path_ok,
                    int(BatteryCurrentControlReplayStatus.PATH_CONSTRAINT_VIOLATED),
                    jnp.where(
                        ~terminal_ok,
                        int(BatteryCurrentControlReplayStatus.TERMINAL_TARGET_MISSED),
                        int(BatteryCurrentControlReplayStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    replay_id = canonical_fingerprint(
        {
            "kind": "battery-current-control-replay-evidence",
            "replay_plan_id": prepared.replay_plan_id,
            "battery_run_id": result.require_evidence_identity(),
            "control_lowering_id": current.lowering_id,
        }
    )
    experiment = prepared.plan.experiment
    return BatteryCurrentControlReplayEvidence(
        result,
        path,
        terminal,
        execution,
        objective_terms,
        objective,
        maximum_violation,
        feasible,
        status,
        prepared.path_constraint_names,
        prepared.terminal_constraint_names,
        ("runtime-success", "ledger-success", "finite-replay-and-objective"),
        prepared.plan.objective.term_names,
        prepared.plan.control_plan_id,
        prepared.replay_plan_id,
        replay_id,
        prepared.plan.parameter_source_id,
        prepared.plan.ledger_criterion_id,
        experiment.plan.experiment_plan_id,
        experiment.preparation_id,
        experiment.plan.protocol.protocol_id,
        experiment.plan.model.model_id,
        experiment.plan.capability_profile.profile_id,
        experiment.plan.support_tuple.support_tuple_id,
    )


__all__ = [
    "BatteryCurrentControlBounds",
    "BatteryCurrentControlObjective",
    "BatteryCurrentControlPlan",
    "BatteryCurrentControlReplayEvidence",
    "BatteryCurrentControlReplayStatus",
    "BatteryCurrentControlTerminalTarget",
    "BatteryPiecewiseCurrent",
    "PreparedBatteryCurrentControl",
    "prepare_battery_current_control",
    "replay_battery_current_control",
]
