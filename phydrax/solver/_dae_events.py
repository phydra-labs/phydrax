#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import DenseLinearOperator, FactorizationPolicy, factorize
from ..metrix import AbstractStateGeometry
from ..nonlinear import (
    implicit_root_result,
    NonlinearStatus,
    NonlinearSystemProblem,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
)
from ..nonlinear._domain_cond import domain_cond as _domain_cond
from ._bdf_method import bdf_predict, bdf_shift_offset
from ._dae_initialization import (
    _initialize_dae,
    _masked_rms,
    _prepare_dae_initialization,
    _PreparedDAEInitialization,
    _scaled_space,
    DAEInitializationResult,
    DAEInitializationSpec,
)
from ._differential_algebraic import (
    _dense_prepared_initialization_regularity,
    DAERegularityStatus,
    DAESolvePolicy,
    DifferentialAlgebraicProblem,
    initialize_dae,
)
from ._hybrid_schedule import HybridSchedulePlan


class DAEConsistencyPolicy(StrictModule, NonTrainableState):
    """Explicit admissibility limits for a computed DAE consistency candidate."""

    maximum_state_correction: float = eqx.field(static=True)
    maximum_rate_correction: float = eqx.field(static=True)
    weighted_correction_tolerance: float = eqx.field(static=True)
    failure: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_state_correction: float,
        maximum_rate_correction: float,
        weighted_correction_tolerance: float,
        /,
        *,
        failure: int = -1,
    ):
        values = tuple(
            float(value)
            for value in (
                maximum_state_correction,
                maximum_rate_correction,
                weighted_correction_tolerance,
            )
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "DAE consistency correction limits must be finite and nonnegative."
            )
        if not isinstance(failure, int) or isinstance(failure, bool):
            raise TypeError("failure must be an integer status.")
        self.maximum_state_correction = values[0]
        self.maximum_rate_correction = values[1]
        self.weighted_correction_tolerance = values[2]
        self.failure = failure
        self.policy_id = canonical_fingerprint(
            {
                "kind": "dae-consistency-policy",
                "maximum_state_correction": values[0],
                "maximum_rate_correction": values[1],
                "weighted_correction_tolerance": values[2],
                "failure": failure,
            }
        )


class DAEConsistencyCandidate(StrictModule, NonTrainableState):
    """A consistency result that never mutates or silently repairs its source."""

    initialization: DAEInitializationResult
    original_state: Array
    original_rate: Array
    state_correction_norm: Array
    rate_correction_norm: Array
    weighted_correction_norm: Array
    admissible: Array
    status: Array
    policy_id: str = eqx.field(static=True)

    def apply(
        self, problem: DifferentialAlgebraicProblem, /
    ) -> DifferentialAlgebraicProblem:
        """Construct a new problem only when the candidate passes every limit."""

        if not isinstance(problem, DifferentialAlgebraicProblem):
            raise TypeError("problem must be a DifferentialAlgebraicProblem.")
        admissible = bool(np.asarray(jax.device_get(self.admissible)))
        if not admissible:
            raise ValueError(
                "An inadmissible DAE consistency candidate cannot be applied."
            )
        return DifferentialAlgebraicProblem(
            problem.system,
            self.initialization.state,
            initial_state_rate=self.initialization.state_rate,
            args=problem.args,
            input_policy=problem.input_policy,
            initialization=problem.initialization,
            discretization_bundle=problem.discretization_bundle,
            problem_id=f"consistent:{problem.problem_id}",
        )


def dae_consistency_candidate(
    problem: DifferentialAlgebraicProblem,
    time: ArrayLike,
    policy: DAEConsistencyPolicy,
    /,
    *,
    solve_policy: DAESolvePolicy | None = None,
    args: Any = None,
    state_guess: ArrayLike | None = None,
    rate_guess: ArrayLike | None = None,
) -> DAEConsistencyCandidate:
    """Compute and bound one candidate; adoption remains a separate explicit call."""

    if not isinstance(problem, DifferentialAlgebraicProblem):
        raise TypeError("problem must be a DifferentialAlgebraicProblem.")
    if not isinstance(policy, DAEConsistencyPolicy):
        raise TypeError("policy must be a DAEConsistencyPolicy.")
    original_state = (
        problem.initial_state if state_guess is None else jnp.asarray(state_guess)
    )
    original_rate = (
        problem.initial_state_rate if rate_guess is None else jnp.asarray(rate_guess)
    )
    result = initialize_dae(
        problem,
        time,
        policy=solve_policy,
        args=problem.args if args is None else args,
        initial_state=original_state,
        initial_state_rate=original_rate,
    )
    state_norm = jnp.sqrt(jnp.mean(jnp.square(jnp.abs(result.state_correction))))
    rate_norm = jnp.sqrt(jnp.mean(jnp.square(jnp.abs(result.rate_correction))))
    state_scale = problem.system.state_scale
    rate_scale = problem.system.state_rate_scale
    weighted = jnp.sqrt(
        0.5
        * (
            jnp.mean(jnp.square(jnp.abs(result.state_correction / state_scale)))
            + jnp.mean(jnp.square(jnp.abs(result.rate_correction / rate_scale)))
        )
    )
    admissible = (
        result.valid
        & jnp.isfinite(state_norm)
        & jnp.isfinite(rate_norm)
        & jnp.isfinite(weighted)
        & (state_norm <= policy.maximum_state_correction)
        & (rate_norm <= policy.maximum_rate_correction)
        & (weighted <= policy.weighted_correction_tolerance)
    )
    return DAEConsistencyCandidate(
        result,
        original_state,
        original_rate,
        state_norm,
        rate_norm,
        weighted,
        admissible,
        jnp.where(admissible, result.status, policy.failure).astype(jnp.int32),
        policy.policy_id,
    )


class DAEResetMap(StrictModule, NonTrainableState):
    """A reset returning explicit post-event state/rate guesses and mask contract."""

    reset: Callable[[Array, Array, Array, Any], tuple[Array, Array]]
    initialization: DAEInitializationSpec
    reset_id: str = eqx.field(static=True)

    def __init__(
        self,
        reset: Callable[[Array, Array, Array, Any], tuple[Array, Array]],
        initialization: DAEInitializationSpec,
        /,
        *,
        reset_id: str,
    ):
        if not callable(reset):
            raise TypeError("DAEResetMap reset must be callable.")
        if not isinstance(initialization, DAEInitializationSpec):
            raise TypeError("initialization must be a DAEInitializationSpec.")
        if not isinstance(reset_id, str) or not reset_id:
            raise ValueError("reset_id must be non-empty.")
        self.reset = reset
        self.initialization = initialization
        self.reset_id = canonical_fingerprint(
            {
                "kind": "dae-reset-map",
                "user_id": reset_id,
                "initialization": initialization.initialization_id,
            }
        )


class DAEEventStatus(IntEnum):
    SUCCESS = 0
    NOT_RUN = 1
    NONFINITE = 2
    LOCALIZATION_FAILED = 3
    GRAZING = 4
    CONSISTENCY_FAILED = 5
    CAPACITY_EXCEEDED = 6


class DAEEventPlan(StrictModule, NonTrainableState):
    """Guard-only schedule with one authoritative DAE reset per guard."""

    schedule: HybridSchedulePlan
    reset_maps: tuple[DAEResetMap, ...]
    consistency_policy: DAEConsistencyPolicy
    grazing_tolerance: float = eqx.field(static=True)
    event_tolerance: float = eqx.field(static=True)
    localization_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule: HybridSchedulePlan,
        reset_maps: Sequence[DAEResetMap],
        consistency_policy: DAEConsistencyPolicy,
        /,
        *,
        grazing_tolerance: float = 1.0e-8,
        event_tolerance: float = 1.0e-10,
        localization_iterations: int = 32,
    ):
        resets = tuple(reset_maps)
        if not isinstance(schedule, HybridSchedulePlan):
            raise TypeError("schedule must be a HybridSchedulePlan.")
        if any(scheduled.event is not None for scheduled in schedule.events):
            raise ValueError(
                "DAE schedules must contain guard metadata only; the DAE reset map "
                "is the sole event action."
            )
        if len(resets) != len(schedule.events) or any(
            not isinstance(value, DAEResetMap) for value in resets
        ):
            raise ValueError(
                "DAE event schedules require exactly one reset map per guard."
            )
        if not isinstance(consistency_policy, DAEConsistencyPolicy):
            raise TypeError("consistency_policy must be a DAEConsistencyPolicy.")
        grazing = float(grazing_tolerance)
        tolerance = float(event_tolerance)
        iterations = int(localization_iterations)
        if (
            not np.isfinite(grazing)
            or grazing <= 0.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or iterations < 1
        ):
            raise ValueError("DAE event localization settings are invalid.")
        self.schedule = schedule
        self.reset_maps = resets
        self.consistency_policy = consistency_policy
        self.grazing_tolerance = grazing
        self.event_tolerance = tolerance
        self.localization_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dae-event-plan",
                "schedule": schedule.plan_id,
                "resets": [value.reset_id for value in resets],
                "consistency": consistency_policy.policy_id,
                "grazing_tolerance": grazing,
                "event_tolerance": tolerance,
                "localization_iterations": iterations,
            }
        )


class DAEEventReplayEvidence(StrictModule):
    """Frozen bracket/winner topology plus audit-only recorded values."""

    bracket_step_indices: Array
    bracket_left_times: Array
    bracket_right_times: Array
    recorded_event_times: Array
    recorded_states_before: Array
    recorded_states_after: Array
    active: Array


class DAEEventResult(StrictModule):
    """Fixed-capacity scalar/index event evidence for one DAE solve."""

    event_times: Array
    event_indices: Array
    states_before: Array
    state_rates_before: Array
    states_after: Array
    state_rates_after: Array
    guard_residuals: Array
    pre_residual_norms: Array
    post_residual_norms: Array
    consistency_correction_norms: Array
    consistency_regularity_status: Array
    consistency_regularity_rank: Array
    consistency_regularity_condition: Array
    consistency_regularity_valid: Array
    simultaneous: Array
    derivative_valid: Array
    valid: Array
    status: Array
    event_count: Array
    terminal: Array
    capacity_exceeded: Array
    replay: DAEEventReplayEvidence
    plan_id: str = eqx.field(static=True)
    domain_failures: Array

    @property
    def successful(self) -> Array:
        return (
            (~self.capacity_exceeded)
            & jnp.all((~self.replay.active) | self.valid)
            & jnp.all(
                (~self.replay.active) | (self.status == int(DAEEventStatus.SUCCESS))
            )
        )


class _DAEEventRootArguments(StrictModule):
    """Event unknown ``[z, time]`` about the newest retained physical state."""

    state_history: Array
    rate_history: Array
    history_times: Array
    order: Array
    model_args: Any
    active: Array
    inactive_reference: Array
    bracket_left: Array
    bracket_right: Array
    guard: Callable[[Array, Array, Any], Array]
    guard_scale: Array

    @property
    def rate_reference(self) -> Array:
        return self.state_history[0]

    def physical_state(self, augmented: Array, /) -> Array:
        increment = augmented[:-1].reshape(self.rate_reference.shape)
        return self.rate_reference + increment

    def state_rate(self, augmented: Array, /) -> Array:
        increment = augmented[:-1].reshape(self.rate_reference.shape)
        shift, offset = bdf_shift_offset(
            self.state_history - self.rate_reference,
            self.history_times,
            augmented[-1],
            self.order,
        )
        return shift * increment + offset


class _DAEEventRootResidual(StrictModule, NonTrainableState):
    system: Any
    input_policy: Any
    guard: Callable[[Array, Array, Any], Array]
    guard_scale: float = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)

    def _state_time_rate(self, augmented, arguments, /):
        state = arguments.physical_state(augmented)
        time = augmented[-1]
        rate = arguments.state_rate(augmented)
        return state, time, rate

    def trial_valid(self, augmented, arguments, /):
        time = augmented[-1]
        in_bracket = (
            jnp.isfinite(time)
            & (time > arguments.bracket_left)
            & (time <= arguments.bracket_right)
        )

        def active(_):
            state, time, rate = self._state_time_rate(augmented, arguments)
            inputs = (
                None
                if self.input_policy is None
                else self.input_policy.evaluate(time, state, arguments.model_args)
            )
            return self.system.trial_valid(
                time, state, rate, arguments.model_args, inputs=inputs
            )

        return _domain_cond(
            arguments.active,
            lambda _: _domain_cond(
                in_bracket, active, lambda _: jnp.asarray(False), None
            ),
            lambda _: jnp.all(jnp.isfinite(augmented)),
            None,
        )

    def _active_residual(self, augmented, arguments, /):
        state, time, rate = self._state_time_rate(augmented, arguments)
        inputs = (
            None
            if self.input_policy is None
            else self.input_policy.evaluate(time, state, arguments.model_args)
        )
        residual = self.system.scaled_residual(
            time,
            state,
            rate,
            arguments.model_args,
            inputs=inputs,
        ).reshape((self.state_size,))
        guard = (
            jnp.asarray(
                self.guard(time, state, arguments.model_args),
                dtype=residual.real.dtype,
            ).reshape(())
            / self.guard_scale
        )
        return jnp.concatenate((residual, guard[None]))

    def __call__(self, augmented, arguments, /):
        return _domain_cond(
            arguments.active,
            lambda _: self._active_residual(augmented, arguments),
            lambda _: augmented - arguments.inactive_reference,
            None,
        )


class PreparedDAEEventPlan(StrictModule, NonTrainableState):
    plan: DAEEventPlan
    root_problems: tuple[NonlinearSystemProblem, ...]
    root_solves: tuple[PreparedNonlinearSolve, ...]
    consistency_solves: tuple[_PreparedDAEInitialization, ...]
    preparation_id: str = eqx.field(static=True)


class DAEEventTransition(StrictModule):
    occurred: Array
    successful: Array
    terminal: Array
    event_index: Array
    event_time: Array
    state_before: Array
    state_rate_before: Array
    state_after: Array
    state_rate_after: Array
    guard_residual: Array
    pre_residual_norm: Array
    post_residual_norm: Array
    consistency_correction_norm: Array
    consistency_regularity_status: Array
    consistency_regularity_rank: Array
    consistency_regularity_condition: Array
    consistency_regularity_valid: Array
    simultaneous: Array
    derivative_valid: Array
    status: Array
    domain_failures: Array


def prepare_dae_event_plan(
    plan: DAEEventPlan,
    problem: DifferentialAlgebraicProblem,
    time_grid: Any,
    solve_policy: DAESolvePolicy,
    /,
) -> PreparedDAEEventPlan:
    """Prepare augmented BDF-guard and post-reset consistency roots."""

    if not isinstance(plan, DAEEventPlan):
        raise TypeError("plan must be a DAEEventPlan.")
    state = problem.initial_state
    state_size = int(state.size)
    augmented_shape = (state_size + 1,)
    augmented_scale = jnp.concatenate(
        (problem.system.state_scale.reshape((state_size,)), jnp.ones((1,)))
    )
    state_space = _scaled_space(
        augmented_shape,
        state.dtype,
        augmented_scale,
        space_id=f"{problem.system.system_id}:dae-event-augmented-increment",
    )
    residual_space = _scaled_space(
        augmented_shape,
        state.dtype,
        jnp.ones(augmented_shape, dtype=state.real.dtype),
        space_id=f"{problem.system.system_id}:dae-event-augmented-residual",
    )
    initial_time = time_grid.times[0]
    target_time = time_grid.times[1]
    history = jnp.broadcast_to(state, (5,) + state.shape)
    rates = jnp.broadcast_to(problem.initial_state_rate, (5,) + state.shape)
    history_times = jnp.full((5,), initial_time, dtype=time_grid.times.dtype)
    order = jnp.asarray(1, dtype=jnp.int32)
    predictor = bdf_predict(
        history,
        rates,
        history_times,
        target_time,
        order,
        jnp.asarray(1, dtype=jnp.int32),
    )
    guess = jnp.concatenate(
        ((predictor - state).reshape((state_size,)), target_time[None])
    )
    root_problems = []
    root_solves = []
    consistency_solves = []
    for scheduled, reset in zip(plan.schedule.events, plan.reset_maps, strict=True):
        arguments = _DAEEventRootArguments(
            history,
            rates,
            history_times,
            order,
            problem.args,
            jnp.asarray(True),
            jnp.zeros(augmented_shape, dtype=state.dtype),
            initial_time,
            target_time,
            scheduled.guard.guard,
            jnp.asarray(np.sqrt(plan.event_tolerance), dtype=state.real.dtype),
        )
        guard_sample = eqx.filter_eval_shape(
            lambda time, value, args: jnp.asarray(
                scheduled.guard.guard(time, value, args)
            ),
            initial_time,
            state,
            problem.args,
        )
        if guard_sample.shape != ():
            raise ValueError("A DAE event guard must return a scalar.")
        residual = _DAEEventRootResidual(
            problem.system,
            problem.input_policy,
            scheduled.guard.guard,
            float(np.sqrt(plan.event_tolerance)),
            problem.system.state_shape,
            state_size,
        )
        root_problem = NonlinearSystemProblem(
            residual,
            state_space=state_space,
            residual_space=residual_space,
            problem_id=f"{problem.system.system_id}:{problem.system.trial_validity_id}:dae-event-increment:{scheduled.guard.guard_id}",
            **problem.system.root_options("event", residual, state_space, residual_space),
        )
        root_solve = prepare_nonlinear(
            root_problem,
            guess,
            method=solve_policy.nonlinear_method,
            termination=solve_policy.nonlinear_termination,
            args=arguments,
        )
        consistency = _prepare_dae_initialization(
            problem.system,
            state,
            problem.initial_state_rate,
            initial_time,
            args=problem.args,
            input_policy=problem.input_policy,
            spec=reset.initialization,
            method=solve_policy.initialization_method,
            termination=solve_policy.initialization_termination,
        )
        root_problems.append(root_problem)
        root_solves.append(root_solve)
        consistency_solves.append(consistency)
    return PreparedDAEEventPlan(
        plan,
        tuple(root_problems),
        tuple(root_solves),
        tuple(consistency_solves),
        canonical_fingerprint(
            {
                "kind": "prepared-dae-events",
                "plan": plan.plan_id,
                "problem": problem.problem_id,
                "time": time_grid.time_id,
            }
        ),
    )


def empty_dae_event_result(
    plan: DAEEventPlan,
    state_template: ArrayLike,
    /,
) -> DAEEventResult:
    state = jnp.asarray(state_template)
    capacity = plan.schedule.maximum_events
    scalar_shape = (capacity,)
    states = jnp.zeros(scalar_shape + state.shape, dtype=state.dtype)
    real_dtype = state.real.dtype
    active = jnp.zeros(scalar_shape, dtype=bool)
    replay = DAEEventReplayEvidence(
        jnp.full(scalar_shape, -1, dtype=jnp.int32),
        jnp.zeros(scalar_shape, dtype=real_dtype),
        jnp.zeros(scalar_shape, dtype=real_dtype),
        jnp.zeros(scalar_shape, dtype=real_dtype),
        states,
        states,
        active,
    )
    return DAEEventResult(
        jnp.zeros(scalar_shape, dtype=real_dtype),
        jnp.full(scalar_shape, -1, dtype=jnp.int32),
        states,
        states,
        states,
        states,
        jnp.zeros(scalar_shape, dtype=real_dtype),
        jnp.zeros(scalar_shape, dtype=real_dtype),
        jnp.zeros(scalar_shape, dtype=real_dtype),
        jnp.zeros(scalar_shape, dtype=real_dtype),
        jnp.full(
            scalar_shape,
            int(DAERegularityStatus.NOT_RUN),
            dtype=jnp.int32,
        ),
        jnp.full(scalar_shape, -1, dtype=jnp.int32),
        jnp.full(scalar_shape, jnp.nan, dtype=real_dtype),
        active,
        active,
        active,
        active,
        jnp.full(scalar_shape, int(DAEEventStatus.NOT_RUN), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(False),
        replay,
        plan.plan_id,
        jnp.asarray(0, dtype=jnp.int32),
    )


def _consistency_candidate(
    prepared: _PreparedDAEInitialization,
    system: Any,
    policy: DAEConsistencyPolicy,
    state_guess: Array,
    rate_guess: Array,
    time: Array,
    args: Any,
    solve_policy: DAESolvePolicy,
    /,
) -> DAEConsistencyCandidate:
    result = _initialize_dae(
        prepared,
        state_guess,
        rate_guess,
        time,
        args=args,
        termination=solve_policy.initialization_termination,
    )
    state_norm = jnp.sqrt(jnp.mean(jnp.square(jnp.abs(result.state_correction))))
    rate_norm = jnp.sqrt(jnp.mean(jnp.square(jnp.abs(result.rate_correction))))
    weighted = jnp.sqrt(
        0.5
        * (
            jnp.mean(jnp.square(jnp.abs(result.state_correction / system.state_scale)))
            + jnp.mean(
                jnp.square(jnp.abs(result.rate_correction / system.state_rate_scale))
            )
        )
    )
    admissible = (
        result.valid
        & jnp.isfinite(state_norm)
        & jnp.isfinite(rate_norm)
        & jnp.isfinite(weighted)
        & (state_norm <= policy.maximum_state_correction)
        & (rate_norm <= policy.maximum_rate_correction)
        & (weighted <= policy.weighted_correction_tolerance)
    )
    return DAEConsistencyCandidate(
        result,
        state_guess,
        rate_guess,
        state_norm,
        rate_norm,
        weighted,
        admissible,
        jnp.where(admissible, result.status, policy.failure).astype(jnp.int32),
        policy.policy_id,
    )


def localize_dae_event(
    prepared: PreparedDAEEventPlan,
    event_index: int,
    left_time: Array,
    right_time: Array,
    state_history: Array,
    rate_history: Array,
    history_times: Array,
    history_depth: Array,
    order: Array,
    right_state: Array,
    args: Any,
    solve_policy: DAESolvePolicy,
    /,
) -> DAEEventTransition:
    """Solve the shortened BDF stage and scalar guard as one implicit root."""

    scheduled = prepared.plan.schedule.events[event_index]
    guard = scheduled.guard.guard
    left_state = state_history[0]
    left_guard = jnp.asarray(guard(left_time, left_state, args)).reshape(())
    right_guard = jnp.asarray(guard(right_time, right_state, args)).reshape(())
    finite_guards = jnp.isfinite(left_guard) & jnp.isfinite(right_guard)
    direction_ok = (
        (scheduled.guard.direction == 0)
        | ((scheduled.guard.direction > 0) & (right_guard > left_guard))
        | ((scheduled.guard.direction < 0) & (right_guard < left_guard))
    )
    bracketed = (
        finite_guards
        & (left_guard != 0.0)
        & (left_guard * right_guard <= 0.0)
        & direction_ok
    )
    occurred = (~finite_guards) | bracketed
    safe_left_guard = jnp.where(finite_guards, left_guard, -1.0)
    safe_right_guard = jnp.where(finite_guards, right_guard, 1.0)
    fraction = -safe_left_guard / jnp.where(
        safe_right_guard == safe_left_guard,
        1.0,
        safe_right_guard - safe_left_guard,
    )
    secant_time = left_time + fraction * (right_time - left_time)
    guess_time = jnp.where(
        bracketed,
        secant_time,
        0.5 * (left_time + right_time),
    )
    predictor = bdf_predict(
        state_history,
        rate_history,
        history_times,
        guess_time,
        order,
        history_depth,
    )
    state_size = int(left_state.size)
    guess = jnp.concatenate(
        ((predictor - left_state).reshape((state_size,)), guess_time[None])
    )
    arguments = _DAEEventRootArguments(
        state_history,
        rate_history,
        history_times,
        order,
        args,
        bracketed,
        guess,
        left_time,
        right_time,
        guard,
        jnp.asarray(np.sqrt(prepared.plan.event_tolerance), dtype=right_time.dtype),
    )
    seeded = refresh_nonlinear(
        prepared.root_solves[event_index],
        prepared.root_problems[event_index],
        guess,
        args=arguments,
    )
    root = implicit_root_result(seeded)
    augmented = jnp.asarray(root.state)
    raw_event_time = augmented[-1]
    raw_state_before = arguments.physical_state(augmented)
    root_time_valid = (
        jnp.isfinite(raw_event_time)
        & (raw_event_time > left_time)
        & (raw_event_time <= right_time)
    )
    raw_state_finite = jnp.all(jnp.isfinite(raw_state_before))
    event_time = raw_event_time
    state_before = raw_state_before
    computed_rate = arguments.state_rate(augmented)
    rate_finite = jnp.all(jnp.isfinite(computed_rate))
    state_rate_before = computed_rate
    root_admissible = (
        arguments.active
        & root_time_valid
        & prepared.root_problems[event_index].trial_valid(augmented, arguments)
    )

    def guard_diagnostics(_):
        return (
            jnp.abs(guard(event_time, state_before, args)),
            jax.jvp(
                lambda time, state: guard(time, state, args),
                (event_time, state_before),
                (jnp.ones_like(event_time), state_rate_before),
            )[1],
        )

    guard_residual, transversality = _domain_cond(
        root_admissible,
        guard_diagnostics,
        lambda _: (
            jnp.asarray(jnp.inf, dtype=event_time.dtype),
            jnp.zeros_like(event_time),
        ),
        None,
    )
    grazing = jnp.abs(transversality) <= prepared.plan.grazing_tolerance
    localized = (
        bracketed
        & (root.status == int(NonlinearStatus.SUCCESS))
        & root_time_valid
        & raw_state_finite
        & rate_finite
        & (guard_residual <= prepared.plan.event_tolerance)
        & (~grazing)
    )
    reset = prepared.plan.reset_maps[event_index]
    state_guess, rate_guess = _domain_cond(
        localized,
        lambda _: reset.reset(event_time, state_before, state_rate_before, args),
        lambda _: (state_before, state_rate_before),
        None,
    )
    state_guess = jnp.asarray(state_guess)
    rate_guess = jnp.asarray(rate_guess)
    reset_finite = (
        jnp.all(jnp.isfinite(state_guess))
        & jnp.all(jnp.isfinite(rate_guess))
        & (state_guess.shape == state_before.shape)
        & (rate_guess.shape == state_rate_before.shape)
    )
    system = prepared.root_problems[event_index].residual_function.system
    consistency = _consistency_candidate(
        prepared.consistency_solves[event_index],
        system,
        prepared.plan.consistency_policy,
        state_guess,
        rate_guess,
        event_time,
        args,
        solve_policy,
    )
    if solve_policy.regularity.mode == "periodic":
        (
            consistency_regularity_status,
            consistency_regularity_rank,
            consistency_regularity_condition,
        ) = _dense_prepared_initialization_regularity(
            prepared.consistency_solves[event_index],
            consistency.initialization,
            event_time,
            args,
            int(state_before.size),
            solve_policy.regularity.condition_limit,
        )
        consistency_regularity_valid = localized & consistency.admissible
    else:
        consistency_regularity_status = jnp.asarray(
            int(DAERegularityStatus.NOT_RUN),
            dtype=jnp.int32,
        )
        consistency_regularity_rank = jnp.asarray(-1, dtype=jnp.int32)
        consistency_regularity_condition = jnp.asarray(jnp.nan, dtype=event_time.dtype)
        consistency_regularity_valid = jnp.asarray(False)
    input_policy = prepared.root_problems[event_index].residual_function.input_policy
    pre_inputs = (
        None
        if input_policy is None
        else input_policy.evaluate(event_time, state_before, args)
    )
    post_inputs = (
        None
        if input_policy is None
        else input_policy.evaluate(event_time, consistency.initialization.state, args)
    )
    pre = system.scaled_residual(
        event_time,
        state_before,
        state_rate_before,
        args,
        inputs=pre_inputs,
    )
    post = system.scaled_residual(
        event_time,
        consistency.initialization.state,
        consistency.initialization.state_rate,
        args,
        inputs=post_inputs,
    )
    pre_norm = _masked_rms(pre, jnp.ones(pre.shape, dtype=bool))
    post_norm = _masked_rms(post, jnp.ones(post.shape, dtype=bool))
    successful = localized & consistency.admissible
    finite = (
        finite_guards
        & root_time_valid
        & raw_state_finite
        & rate_finite
        & reset_finite
        & jnp.isfinite(pre_norm)
        & jnp.isfinite(post_norm)
        & jnp.all(jnp.isfinite(consistency.initialization.state))
        & jnp.all(jnp.isfinite(consistency.initialization.state_rate))
    )
    successful = successful & finite
    status = jnp.where(
        ~occurred,
        int(DAEEventStatus.NOT_RUN),
        jnp.where(
            ~finite,
            int(DAEEventStatus.NONFINITE),
            jnp.where(
                grazing,
                int(DAEEventStatus.GRAZING),
                jnp.where(
                    ~localized,
                    int(DAEEventStatus.LOCALIZATION_FAILED),
                    jnp.where(
                        ~consistency.admissible,
                        int(DAEEventStatus.CONSISTENCY_FAILED),
                        int(DAEEventStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return DAEEventTransition(
        occurred,
        successful,
        jnp.asarray(scheduled.guard.terminal),
        jnp.asarray(event_index, dtype=jnp.int32),
        event_time,
        state_before,
        state_rate_before,
        consistency.initialization.state,
        consistency.initialization.state_rate,
        guard_residual,
        pre_norm,
        post_norm,
        consistency.weighted_correction_norm,
        consistency_regularity_status,
        consistency_regularity_rank,
        consistency_regularity_condition,
        consistency_regularity_valid,
        jnp.asarray(False),
        successful,
        status,
        root.diagnostics.domain_failures + consistency.initialization.domain_failures,
    )


def _select_transition(
    choose: Array,
    candidate: DAEEventTransition,
    current: DAEEventTransition,
    /,
) -> DAEEventTransition:
    return DAEEventTransition(
        jnp.where(choose, candidate.occurred, current.occurred),
        jnp.where(choose, candidate.successful, current.successful),
        jnp.where(choose, candidate.terminal, current.terminal),
        jnp.where(choose, candidate.event_index, current.event_index),
        jnp.where(choose, candidate.event_time, current.event_time),
        jnp.where(choose, candidate.state_before, current.state_before),
        jnp.where(choose, candidate.state_rate_before, current.state_rate_before),
        jnp.where(choose, candidate.state_after, current.state_after),
        jnp.where(choose, candidate.state_rate_after, current.state_rate_after),
        jnp.where(choose, candidate.guard_residual, current.guard_residual),
        jnp.where(choose, candidate.pre_residual_norm, current.pre_residual_norm),
        jnp.where(choose, candidate.post_residual_norm, current.post_residual_norm),
        jnp.where(
            choose,
            candidate.consistency_correction_norm,
            current.consistency_correction_norm,
        ),
        jnp.where(
            choose,
            candidate.consistency_regularity_status,
            current.consistency_regularity_status,
        ),
        jnp.where(
            choose,
            candidate.consistency_regularity_rank,
            current.consistency_regularity_rank,
        ),
        jnp.where(
            choose,
            candidate.consistency_regularity_condition,
            current.consistency_regularity_condition,
        ),
        jnp.where(
            choose,
            candidate.consistency_regularity_valid,
            current.consistency_regularity_valid,
        ),
        jnp.where(choose, candidate.simultaneous, current.simultaneous),
        jnp.where(choose, candidate.derivative_valid, current.derivative_valid),
        jnp.where(choose, candidate.status, current.status),
        jnp.where(choose, candidate.domain_failures, current.domain_failures),
    )


def resolve_dae_event(
    prepared: PreparedDAEEventPlan,
    left_time: Array,
    right_time: Array,
    state_history: Array,
    rate_history: Array,
    history_times: Array,
    history_depth: Array,
    order: Array,
    right_state: Array,
    right_state_rate: Array,
    args: Any,
    solve_policy: DAESolvePolicy,
    /,
) -> DAEEventTransition:
    """Select the earliest successful crossing, then priority for simultaneous roots."""

    candidates = tuple(
        localize_dae_event(
            prepared,
            event_index,
            left_time,
            right_time,
            state_history,
            rate_history,
            history_times,
            history_depth,
            order,
            right_state,
            args,
            solve_policy,
        )
        for event_index in range(len(prepared.plan.schedule.events))
    )
    first = candidates[0]
    neutral = DAEEventTransition(
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(-1, dtype=jnp.int32),
        right_time,
        right_state,
        right_state_rate,
        right_state,
        right_state_rate,
        jnp.asarray(jnp.inf, dtype=right_time.dtype),
        jnp.asarray(jnp.inf, dtype=right_time.dtype),
        jnp.asarray(jnp.inf, dtype=right_time.dtype),
        jnp.asarray(jnp.inf, dtype=right_time.dtype),
        jnp.asarray(int(DAERegularityStatus.NOT_RUN), dtype=jnp.int32),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(jnp.nan, dtype=right_time.dtype),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(int(DAEEventStatus.NOT_RUN), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    winner = neutral
    winner_exists = jnp.asarray(False)
    winner_priority = jnp.asarray(np.iinfo(np.int32).min, dtype=jnp.int32)
    failed = neutral
    failed_exists = jnp.asarray(False)
    for event_index, candidate in enumerate(candidates):
        priority = prepared.plan.schedule.events[event_index].guard.priority
        simultaneous = (
            jnp.abs(candidate.event_time - winner.event_time)
            <= prepared.plan.schedule.simultaneous_tolerance
        )
        earlier = (
            candidate.event_time
            < winner.event_time - prepared.plan.schedule.simultaneous_tolerance
        )
        choose = candidate.successful & (
            (~winner_exists) | earlier | (simultaneous & (priority > winner_priority))
        )
        winner = _select_transition(choose, candidate, winner)
        winner_priority = jnp.where(choose, priority, winner_priority)
        winner_exists = winner_exists | candidate.successful
        choose_failed = candidate.occurred & (~candidate.successful) & (~failed_exists)
        failed = _select_transition(choose_failed, candidate, failed)
        failed_exists = failed_exists | (candidate.occurred & (~candidate.successful))
    selected = _select_transition(
        winner_exists & (~failed_exists),
        winner,
        failed,
    )
    exists = winner_exists | failed_exists
    selected = _select_transition(exists, selected, neutral)
    simultaneous = jnp.asarray(False)
    for event_index, candidate in enumerate(candidates):
        competing = (
            candidate.successful
            & (selected.event_index != event_index)
            & (
                jnp.abs(candidate.event_time - selected.event_time)
                <= prepared.plan.schedule.simultaneous_tolerance
            )
        )
        simultaneous = simultaneous | competing
    return DAEEventTransition(
        selected.occurred,
        selected.successful,
        selected.terminal,
        selected.event_index,
        selected.event_time,
        selected.state_before,
        selected.state_rate_before,
        selected.state_after,
        selected.state_rate_after,
        selected.guard_residual,
        selected.pre_residual_norm,
        selected.post_residual_norm,
        selected.consistency_correction_norm,
        selected.consistency_regularity_status,
        selected.consistency_regularity_rank,
        selected.consistency_regularity_condition,
        selected.consistency_regularity_valid,
        simultaneous,
        selected.derivative_valid & (~simultaneous),
        selected.status,
        sum(
            (candidate.domain_failures for candidate in candidates),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )


def record_dae_event(
    result: DAEEventResult,
    transition: DAEEventTransition,
    bracket_step_index: Array,
    bracket_left_time: Array,
    bracket_right_time: Array,
    /,
) -> DAEEventResult:
    """Append one transition without reallocating fixed-capacity evidence."""

    capacity = result.event_times.shape[0]
    if capacity == 0:
        return result
    room = result.event_count < capacity
    write = transition.occurred & room
    slot = jnp.minimum(result.event_count, capacity - 1)
    overflow = result.capacity_exceeded | (transition.occurred & (~room))
    status = jnp.where(
        transition.occurred & (~room),
        int(DAEEventStatus.CAPACITY_EXCEEDED),
        transition.status,
    ).astype(jnp.int32)
    replay = DAEEventReplayEvidence(
        result.replay.bracket_step_indices.at[slot].set(
            jnp.where(write, bracket_step_index, result.replay.bracket_step_indices[slot])
        ),
        result.replay.bracket_left_times.at[slot].set(
            jnp.where(write, bracket_left_time, result.replay.bracket_left_times[slot])
        ),
        result.replay.bracket_right_times.at[slot].set(
            jnp.where(write, bracket_right_time, result.replay.bracket_right_times[slot])
        ),
        result.replay.recorded_event_times.at[slot].set(
            jnp.where(
                write, transition.event_time, result.replay.recorded_event_times[slot]
            )
        ),
        result.replay.recorded_states_before.at[slot].set(
            jnp.where(
                write, transition.state_before, result.replay.recorded_states_before[slot]
            )
        ),
        result.replay.recorded_states_after.at[slot].set(
            jnp.where(
                write, transition.state_after, result.replay.recorded_states_after[slot]
            )
        ),
        result.replay.active.at[slot].set(write | result.replay.active[slot]),
    )
    return DAEEventResult(
        result.event_times.at[slot].set(
            jnp.where(write, transition.event_time, result.event_times[slot])
        ),
        result.event_indices.at[slot].set(
            jnp.where(write, transition.event_index, result.event_indices[slot])
        ),
        result.states_before.at[slot].set(
            jnp.where(write, transition.state_before, result.states_before[slot])
        ),
        result.state_rates_before.at[slot].set(
            jnp.where(
                write, transition.state_rate_before, result.state_rates_before[slot]
            )
        ),
        result.states_after.at[slot].set(
            jnp.where(write, transition.state_after, result.states_after[slot])
        ),
        result.state_rates_after.at[slot].set(
            jnp.where(write, transition.state_rate_after, result.state_rates_after[slot])
        ),
        result.guard_residuals.at[slot].set(
            jnp.where(write, transition.guard_residual, result.guard_residuals[slot])
        ),
        result.pre_residual_norms.at[slot].set(
            jnp.where(
                write, transition.pre_residual_norm, result.pre_residual_norms[slot]
            )
        ),
        result.post_residual_norms.at[slot].set(
            jnp.where(
                write, transition.post_residual_norm, result.post_residual_norms[slot]
            )
        ),
        result.consistency_correction_norms.at[slot].set(
            jnp.where(
                write,
                transition.consistency_correction_norm,
                result.consistency_correction_norms[slot],
            )
        ),
        result.consistency_regularity_status.at[slot].set(
            jnp.where(
                write,
                transition.consistency_regularity_status,
                result.consistency_regularity_status[slot],
            )
        ),
        result.consistency_regularity_rank.at[slot].set(
            jnp.where(
                write,
                transition.consistency_regularity_rank,
                result.consistency_regularity_rank[slot],
            )
        ),
        result.consistency_regularity_condition.at[slot].set(
            jnp.where(
                write,
                transition.consistency_regularity_condition,
                result.consistency_regularity_condition[slot],
            )
        ),
        result.consistency_regularity_valid.at[slot].set(
            jnp.where(
                write,
                transition.consistency_regularity_valid,
                result.consistency_regularity_valid[slot],
            )
        ),
        result.simultaneous.at[slot].set(
            jnp.where(write, transition.simultaneous, result.simultaneous[slot])
        ),
        result.derivative_valid.at[slot].set(
            jnp.where(write, transition.derivative_valid, result.derivative_valid[slot])
        ),
        result.valid.at[slot].set(
            jnp.where(write, transition.successful, result.valid[slot])
        ),
        result.status.at[slot].set(jnp.where(write, status, result.status[slot])),
        result.event_count + write.astype(jnp.int32),
        result.terminal | (write & transition.successful & transition.terminal),
        overflow,
        replay,
        result.plan_id,
        result.domain_failures + transition.domain_failures,
    )


class DAERegularityDomain(StrictModule, NonTrainableState):
    """Finite coordinate cells over which one operator enclosure is claimed."""

    lower: Array
    upper: Array
    domain_id: str = eqx.field(static=True)

    def __init__(self, lower: ArrayLike, upper: ArrayLike, /, *, domain_id: str):
        lower_ = jnp.asarray(lower)
        upper_ = jnp.asarray(upper, dtype=lower_.dtype)
        if lower_.ndim != 2 or lower_.shape != upper_.shape or lower_.shape[0] == 0:
            raise ValueError(
                "DAE regularity cells require matching nonempty shape (cells,coordinates)."
            )
        if (
            np.any(~np.isfinite(np.asarray(lower_)))
            or np.any(~np.isfinite(np.asarray(upper_)))
            or np.any(np.asarray(lower_) >= np.asarray(upper_))
        ):
            raise ValueError(
                "DAE regularity cells must be finite and have positive width."
            )
        if not isinstance(domain_id, str) or not domain_id:
            raise ValueError("domain_id must be non-empty.")
        self.lower = lower_
        self.upper = upper_
        self.domain_id = canonical_fingerprint(
            {
                "kind": "dae-regularity-domain",
                "user_id": domain_id,
                "cells": int(lower_.shape[0]),
                "coordinates": int(lower_.shape[1]),
                "bounds": array_tree_fingerprint((lower_, upper_)),
            }
        )


class DAERegularityCertificatePlan(StrictModule, NonTrainableState):
    """Typed center operator and certified variation bound provider."""

    domain: DAERegularityDomain
    enclosure: Callable[[Array, Any], tuple[Array, Array]]
    operator_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: DAERegularityDomain,
        enclosure: Callable[[Array, Any], tuple[Array, Array]],
        /,
        *,
        operator_id: str,
    ):
        if not isinstance(domain, DAERegularityDomain):
            raise TypeError("domain must be a DAERegularityDomain.")
        if not callable(enclosure):
            raise TypeError("enclosure must be callable.")
        if not isinstance(operator_id, str) or not operator_id:
            raise ValueError("operator_id must be non-empty.")
        self.domain = domain
        self.enclosure = enclosure
        self.operator_id = operator_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dae-regularity-certificate-plan",
                "domain": domain.domain_id,
                "operator": operator_id,
            }
        )


class DAERegularityCertificate(StrictModule, NonTrainableState):
    lower_singular_value_bounds: Array
    center_singular_values: Array
    variation_bounds: Array
    covered: Array
    uncovered_cells: Array
    certified: Array
    status: Array
    hypotheses: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def certify_dae_regularity(
    plan: DAERegularityCertificatePlan,
    /,
    *,
    args: Any = None,
) -> DAERegularityCertificate:
    """Certify only cells with sigma_min(A(center)) minus variation strictly positive."""

    if not isinstance(plan, DAERegularityCertificatePlan):
        raise TypeError("plan must be a DAERegularityCertificatePlan.")
    centers = 0.5 * (plan.domain.lower + plan.domain.upper)
    minimum = []
    variation = []
    for cell in range(centers.shape[0]):
        matrix, bound = plan.enclosure(centers[cell], args)
        matrix_ = jnp.asarray(matrix)
        bound_ = jnp.asarray(bound, dtype=matrix_.real.dtype).reshape(())
        if matrix_.ndim != 2 or matrix_.shape[0] != matrix_.shape[1]:
            raise ValueError(
                "DAE regularity enclosure operators must be square matrices."
            )
        factorization = factorize(
            DenseLinearOperator(matrix_, operator_id=f"{plan.operator_id}:cell:{cell}"),
            FactorizationPolicy("svd"),
        )
        singular_values = factorization.singular_values()
        minimum.append(jnp.min(singular_values))
        variation.append(bound_)
    center_values = jnp.stack(tuple(minimum))
    variation_values = jnp.stack(tuple(variation))
    lower_bounds = center_values - variation_values
    covered = (
        jnp.isfinite(center_values)
        & jnp.isfinite(variation_values)
        & (variation_values >= 0)
        & (lower_bounds > 0)
    )
    uncovered = jnp.nonzero(~covered, size=covered.shape[0], fill_value=-1)[0]
    certified = jnp.all(covered)
    return DAERegularityCertificate(
        lower_bounds,
        center_values,
        variation_values,
        covered,
        uncovered,
        certified,
        jnp.where(certified, 0, -1).astype(jnp.int32),
        "finite declared cells; center singular value minus certified operator variation",
        plan.plan_id,
    )


class ManifoldBDFMethod(StrictModule, NonTrainableState):
    """Prepared local-coordinate BDF1/BDF2 method; higher orders fail closed."""

    order: int = eqx.field(static=True)
    coefficients: tuple[float, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self, order: int = 1, /):
        if order not in (1, 2):
            raise ValueError("Manifold BDF currently supports only orders one and two.")
        coefficients = (1.0, -1.0) if order == 1 else (1.5, -2.0, 0.5)
        self.order = order
        self.coefficients = coefficients
        self.method_id = f"manifold-bdf:{order}"


class ManifoldBDFStage(StrictModule, NonTrainableState):
    state: Array
    state_rate: Array
    local_coordinate: Array
    local_rate: Array
    contained: Array
    chart_valid: Array
    method_id: str = eqx.field(static=True)


def manifold_bdf_stage(
    method: ManifoldBDFMethod,
    geometry: AbstractStateGeometry,
    base_state: ArrayLike,
    history: Sequence[ArrayLike],
    candidate_local: ArrayLike,
    step_size: ArrayLike,
    /,
) -> ManifoldBDFStage:
    """Construct one fixed-chart manifold BDF endpoint and physical tangent."""

    if not isinstance(method, ManifoldBDFMethod):
        raise TypeError("method must be a ManifoldBDFMethod.")
    if not isinstance(geometry, AbstractStateGeometry):
        raise TypeError("geometry must be an AbstractStateGeometry.")
    states = tuple(jnp.asarray(value) for value in history)
    if len(states) != method.order:
        raise ValueError("history length must equal the prepared manifold BDF order.")
    base = jnp.asarray(base_state)
    local = jnp.asarray(candidate_local)
    dt = jnp.asarray(step_size, dtype=base.real.dtype).reshape(())
    history_local = tuple(
        jnp.asarray(geometry.inverse_retract(base, value)) for value in states
    )
    local_rate = method.coefficients[0] * local
    for coefficient, value in zip(method.coefficients[1:], history_local, strict=True):
        local_rate = local_rate + coefficient * value
    local_rate = local_rate / dt
    state, state_rate = jax.jvp(
        lambda tangent: geometry.retract(base, tangent),
        (local,),
        (local_rate,),
    )
    contained = jnp.asarray(geometry.contains(state), dtype=bool)
    chart_valid = (
        contained
        & jnp.all(jnp.isfinite(state))
        & jnp.all(jnp.isfinite(state_rate))
        & (dt > 0)
    )
    return ManifoldBDFStage(
        state,
        jnp.where(chart_valid, state_rate, jnp.nan),
        local,
        local_rate,
        contained,
        chart_valid,
        method.method_id,
    )


__all__ = [
    "DAEConsistencyCandidate",
    "DAEConsistencyPolicy",
    "DAEEventPlan",
    "DAEEventReplayEvidence",
    "DAEEventResult",
    "DAEEventStatus",
    "DAERegularityCertificate",
    "DAERegularityCertificatePlan",
    "DAERegularityDomain",
    "DAEResetMap",
    "ManifoldBDFMethod",
    "ManifoldBDFStage",
    "PreparedDAEEventPlan",
    "certify_dae_regularity",
    "dae_consistency_candidate",
    "empty_dae_event_result",
    "localize_dae_event",
    "manifold_bdf_stage",
    "prepare_dae_event_plan",
]
