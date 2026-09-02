#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import DenseLinearOperator, FactorizationPolicy, factorize


class HybridEventPlan(StrictModule, NonTrainableState):
    """One time-aware guard/reset and its fixed-epoch saltation dynamics."""

    guard: Callable[[Array, Array, Any], Array]
    reset: Callable[[Array, Array, Any], Array]
    vector_field_before: Callable[[Array, Array, Any], Array]
    vector_field_after: Callable[[Array, Array, Any], Array]
    competing_guards: tuple[Callable[[Array, Array, Any], Array], ...]
    event_kind: str = eqx.field(static=True)
    grazing_tolerance: float = eqx.field(static=True)
    event_tolerance: float = eqx.field(static=True)
    bisection_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        guard,
        reset,
        vector_field_before,
        vector_field_after,
        /,
        *,
        event_kind: str,
        competing_guards: Sequence[Callable[[Array, Array, Any], Array]] = (),
        grazing_tolerance: float = 1.0e-8,
        event_tolerance: float = 1.0e-10,
        bisection_iterations: int = 48,
        plan_id: str,
    ):
        callables = (guard, reset, vector_field_before, vector_field_after)
        if any(not callable(value) for value in callables):
            raise TypeError("Hybrid guard/reset/vector fields must be callable.")
        competing = tuple(competing_guards)
        if any(not callable(value) for value in competing):
            raise TypeError("competing_guards must contain callables.")
        kind = str(event_kind)
        grazing = float(grazing_tolerance)
        tolerance = float(event_tolerance)
        iterations = int(bisection_iterations)
        identifier = str(plan_id)
        if (
            not kind
            or not np.isfinite(grazing)
            or grazing <= 0.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or iterations < 8
            or not identifier
        ):
            raise ValueError(
                "Hybrid event kind, tolerances, iterations, or ID are invalid."
            )
        self.guard = guard
        self.reset = reset
        self.vector_field_before = vector_field_before
        self.vector_field_after = vector_field_after
        self.competing_guards = competing
        self.event_kind = kind
        self.grazing_tolerance = grazing
        self.event_tolerance = tolerance
        self.bisection_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hybrid-event-plan",
                "user_id": identifier,
                "event_kind": kind,
                "grazing_tolerance": grazing,
                "event_tolerance": tolerance,
                "bisection_iterations": iterations,
                "competing_count": len(competing),
            }
        )


class HybridReplayPolicy(StrictModule, NonTrainableState):
    """Fixed event storage and fail-closed replay tolerances."""

    maximum_events: int = eqx.field(static=True)
    grazing_tolerance: float = eqx.field(static=True)
    simultaneous_tolerance: float = eqx.field(static=True)
    event_tolerance: float = eqx.field(static=True)
    failure: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_events: int,
        /,
        *,
        grazing_tolerance: float = 1.0e-8,
        simultaneous_tolerance: float = 1.0e-10,
        event_tolerance: float = 1.0e-10,
        failure: int = -1,
    ):
        if not isinstance(maximum_events, int) or isinstance(maximum_events, bool):
            raise TypeError("maximum_events must be an integer.")
        if maximum_events < 0:
            raise ValueError("maximum_events must be nonnegative.")
        tolerances = tuple(
            float(value)
            for value in (
                grazing_tolerance,
                simultaneous_tolerance,
                event_tolerance,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("Hybrid replay tolerances must be positive and finite.")
        if not isinstance(failure, int) or isinstance(failure, bool):
            raise TypeError("failure must be an integer status.")
        self.maximum_events = maximum_events
        self.grazing_tolerance = tolerances[0]
        self.simultaneous_tolerance = tolerances[1]
        self.event_tolerance = tolerances[2]
        self.failure = failure
        self.policy_id = canonical_fingerprint(
            {
                "kind": "hybrid-replay-policy",
                "maximum_events": maximum_events,
                "grazing_tolerance": tolerances[0],
                "simultaneous_tolerance": tolerances[1],
                "event_tolerance": tolerances[2],
                "failure": failure,
            }
        )


class HybridEventSensitivityResult(StrictModule):
    event_time: Array
    state_before: Array
    state_after: Array
    saltation_matrix: Array
    guard_residual: Array
    transversality: Array
    grazing: Array
    simultaneous: Array
    successful: Array
    determinant_sign: Array
    log_abs_determinant: Array
    log_jacobian_valid: Array
    plan_id: str = eqx.field(static=True)


class HybridEventActionResult(StrictModule):
    """Matrix-free jump action and its validity evidence."""

    action: Any
    transversality: Array
    grazing: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class HybridEventTape(StrictModule, NonTrainableState):
    """Canonical fixed-capacity event/reset/saltation replay record."""

    event_indices: Array
    event_times: Array
    states_before: Array
    states_after: Array
    guard_residuals: Array
    transversality: Array
    saltation_valid: Array
    determinant_signs: Array
    log_abs_determinants: Array
    log_jacobian_valid: Array
    active: Array
    event_count: Array
    terminal: Array
    capacity_exceeded: Array
    status: Array
    policy_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    @property
    def total_log_abs_determinant(self) -> Array:
        values = jnp.where(
            self.active & self.log_jacobian_valid, self.log_abs_determinants, 0
        )
        return jnp.where(
            jnp.all((~self.active) | self.log_jacobian_valid),
            jnp.sum(values),
            jnp.nan,
        )

    @property
    def total_determinant_sign(self) -> Array:
        values = jnp.where(
            self.active & self.log_jacobian_valid, self.determinant_signs, 1
        )
        return jnp.where(
            jnp.all((~self.active) | self.log_jacobian_valid),
            jnp.prod(values),
            jnp.nan,
        )


def empty_hybrid_event_tape(
    policy: HybridReplayPolicy,
    state_template: ArrayLike,
    /,
    *,
    schedule_id: str,
) -> HybridEventTape:
    """Allocate neutral fixed-shape tape storage for a prepared schedule."""

    if not isinstance(policy, HybridReplayPolicy):
        raise TypeError("policy must be a HybridReplayPolicy.")
    if not isinstance(schedule_id, str) or not schedule_id:
        raise ValueError("schedule_id must be non-empty.")
    state = jnp.asarray(state_template)
    n = policy.maximum_events
    states = jnp.zeros((n,) + state.shape, dtype=state.dtype)
    real_dtype = state.real.dtype
    return HybridEventTape(
        event_indices=jnp.full((n,), -1, dtype=jnp.int32),
        event_times=jnp.zeros((n,), dtype=real_dtype),
        states_before=states,
        states_after=states,
        guard_residuals=jnp.zeros((n,), dtype=real_dtype),
        transversality=jnp.zeros((n,), dtype=real_dtype),
        saltation_valid=jnp.zeros((n,), dtype=bool),
        determinant_signs=jnp.ones((n,), dtype=real_dtype),
        log_abs_determinants=jnp.zeros((n,), dtype=real_dtype),
        log_jacobian_valid=jnp.zeros((n,), dtype=bool),
        active=jnp.zeros((n,), dtype=bool),
        event_count=jnp.asarray(0, dtype=jnp.int32),
        terminal=jnp.asarray(False),
        capacity_exceeded=jnp.asarray(False),
        status=jnp.asarray(0, dtype=jnp.int32),
        policy_id=policy.policy_id,
        schedule_id=schedule_id,
    )


def record_hybrid_event(
    tape: HybridEventTape,
    policy: HybridReplayPolicy,
    event_index: ArrayLike,
    result: HybridEventSensitivityResult,
    /,
    *,
    terminal: ArrayLike = False,
    status: ArrayLike = 0,
) -> HybridEventTape:
    """Append one localized event without changing tape capacity or topology."""

    if not isinstance(tape, HybridEventTape):
        raise TypeError("tape must be a HybridEventTape.")
    if not isinstance(policy, HybridReplayPolicy) or tape.policy_id != policy.policy_id:
        raise ValueError("Hybrid tape and replay policy identities do not match.")
    if not isinstance(result, HybridEventSensitivityResult):
        raise TypeError("result must be HybridEventSensitivityResult.")
    slot = tape.event_count
    room = slot < policy.maximum_events
    safe_slot = jnp.minimum(slot, max(policy.maximum_events - 1, 0))

    if policy.maximum_events == 0:
        return eqx.tree_at(
            lambda value: (value.capacity_exceeded, value.status, value.terminal),
            tape,
            (
                jnp.asarray(True),
                jnp.asarray(policy.failure, dtype=jnp.int32),
                jnp.asarray(terminal),
            ),
        )

    valid = room & result.successful
    event_indices = tape.event_indices.at[safe_slot].set(
        jnp.where(valid, jnp.asarray(event_index, dtype=jnp.int32), -1)
    )
    event_times = tape.event_times.at[safe_slot].set(result.event_time)
    states_before = tape.states_before.at[safe_slot].set(result.state_before)
    states_after = tape.states_after.at[safe_slot].set(result.state_after)
    guard_residuals = tape.guard_residuals.at[safe_slot].set(result.guard_residual)
    transversality = tape.transversality.at[safe_slot].set(result.transversality)
    saltation_valid = tape.saltation_valid.at[safe_slot].set(valid)
    determinant_signs = tape.determinant_signs.at[safe_slot].set(result.determinant_sign)
    log_abs_determinants = tape.log_abs_determinants.at[safe_slot].set(
        result.log_abs_determinant
    )
    log_jacobian_valid = tape.log_jacobian_valid.at[safe_slot].set(
        valid & result.log_jacobian_valid
    )
    active = tape.active.at[safe_slot].set(valid)
    failed = tape.capacity_exceeded | (~room) | (~result.successful)
    next_status = jnp.where(failed, policy.failure, jnp.asarray(status, dtype=jnp.int32))
    return HybridEventTape(
        event_indices,
        event_times,
        states_before,
        states_after,
        guard_residuals,
        transversality,
        saltation_valid,
        determinant_signs,
        log_abs_determinants,
        log_jacobian_valid,
        active,
        tape.event_count + jnp.where(room, 1, 0),
        jnp.asarray(terminal, dtype=bool) & valid,
        failed,
        next_status,
        tape.policy_id,
        tape.schedule_id,
    )


def _event_differentials(
    plan: HybridEventPlan, time: Array, state_before: Array, args: Any, /
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    state_after = jnp.asarray(plan.reset(time, state_before, args))
    guard_time = jax.jvp(
        lambda value: plan.guard(value, state_before, args),
        (time,),
        (jnp.ones_like(time),),
    )[1]
    normal = jax.grad(lambda state: plan.guard(time, state, args))(state_before)
    reset_time = jax.jvp(
        lambda value: plan.reset(value, state_before, args),
        (time,),
        (jnp.ones_like(time),),
    )[1]
    reset_jacobian = jax.jacfwd(lambda state: plan.reset(time, state, args))(state_before)
    before = jnp.asarray(plan.vector_field_before(time, state_before, args))
    after = jnp.asarray(plan.vector_field_after(time, state_after, args))
    denominator = guard_time + oe.contract("...i,...i->", normal, before)
    reset_before = oe.contract("...ij,...j->...i", reset_jacobian, before)
    jump = after - reset_time - reset_before
    return state_after, normal, reset_jacobian, before, jump, denominator, guard_time


def localize_hybrid_event(
    plan: HybridEventPlan,
    state_at_time: Callable[[Array, Any], Array],
    left_time: Array,
    right_time: Array,
    /,
    *,
    args: Any = None,
) -> HybridEventSensitivityResult:
    """Localize one bracketed transverse event and construct dense jump evidence."""

    if not isinstance(plan, HybridEventPlan):
        raise TypeError("plan must be a HybridEventPlan.")
    if not callable(state_at_time):
        raise TypeError("state_at_time must be callable.")
    left = jnp.asarray(left_time)
    right = jnp.asarray(right_time, dtype=left.dtype)
    left_state = jnp.asarray(state_at_time(left, args))
    right_state = jnp.asarray(state_at_time(right, args))
    left_guard = jnp.asarray(plan.guard(left, left_state, args))
    right_guard = jnp.asarray(plan.guard(right, right_state, args))
    if left_guard.shape != () or right_guard.shape != ():
        raise ValueError("Hybrid guard must return a scalar.")
    bracketed = (
        jnp.isfinite(left_guard)
        & jnp.isfinite(right_guard)
        & (left < right)
        & (left_guard * right_guard <= 0.0)
    )

    def iteration(_: int, carry: tuple[Array, Array, Array]):
        lower, upper, lower_guard = carry
        midpoint = 0.5 * (lower + upper)
        midpoint_state = state_at_time(midpoint, args)
        midpoint_guard = plan.guard(midpoint, midpoint_state, args)
        same_side = lower_guard * midpoint_guard > 0.0
        return (
            jnp.where(same_side, midpoint, lower),
            jnp.where(same_side, upper, midpoint),
            jnp.where(same_side, midpoint_guard, lower_guard),
        )

    lower, upper, _ = jax.lax.fori_loop(
        0,
        plan.bisection_iterations,
        iteration,
        (left, right, left_guard),
    )
    event_time = 0.5 * (lower + upper)
    state_before = jnp.asarray(state_at_time(event_time, args))
    (
        state_after,
        normal,
        reset_jacobian,
        _,
        jump,
        transversality,
        _,
    ) = _event_differentials(plan, event_time, state_before, args)
    guard_residual = jnp.abs(plan.guard(event_time, state_before, args))
    grazing = jnp.abs(transversality) <= plan.grazing_tolerance
    saltation = reset_jacobian + oe.contract(
        "...i,...j->...ij", jump, normal
    ) / jnp.where(grazing, 1.0, transversality)
    simultaneous = jnp.asarray(False)
    for competing in plan.competing_guards:
        simultaneous = simultaneous | (
            jnp.abs(competing(event_time, state_before, args)) <= plan.event_tolerance
        )
    finite = (
        jnp.all(jnp.isfinite(state_before))
        & jnp.all(jnp.isfinite(state_after))
        & jnp.all(jnp.isfinite(saltation))
    )
    successful = (
        bracketed
        & (~grazing)
        & (~simultaneous)
        & finite
        & (guard_residual <= plan.event_tolerance)
    )
    saltation = jnp.where(successful, saltation, jnp.nan)
    square = (
        reset_jacobian.ndim == 2 and reset_jacobian.shape[0] == reset_jacobian.shape[1]
    )
    if square:
        factorization = factorize(
            DenseLinearOperator(
                saltation,
                operator_id=f"{plan.plan_id}:saltation-matrix",
            ),
            FactorizationPolicy("lu"),
        )
        determinant_sign = factorization.determinant_sign()
        log_abs_determinant = factorization.log_abs_determinant()
        log_valid = (
            successful & jnp.isfinite(log_abs_determinant) & (determinant_sign != 0)
        )
    else:
        determinant_sign = jnp.asarray(jnp.nan, dtype=state_before.real.dtype)
        log_abs_determinant = jnp.asarray(jnp.nan, dtype=state_before.real.dtype)
        log_valid = jnp.asarray(False)
    return HybridEventSensitivityResult(
        event_time,
        state_before,
        state_after,
        saltation,
        guard_residual,
        transversality,
        grazing,
        simultaneous,
        successful,
        determinant_sign,
        log_abs_determinant,
        log_valid,
        plan.plan_id,
    )


def hybrid_event_jvp(
    plan: HybridEventPlan,
    event_time: ArrayLike,
    state_before: ArrayLike,
    state_tangent: Any,
    /,
    *,
    args: Any = None,
    time_tangent: ArrayLike = 0.0,
    args_tangent: Any = None,
) -> HybridEventActionResult:
    """Apply the saltation/JVP action without materializing its state matrix."""

    if not isinstance(plan, HybridEventPlan):
        raise TypeError("plan must be a HybridEventPlan.")
    time = jnp.asarray(event_time)
    state = jnp.asarray(state_before)
    state_tangent_ = jnp.asarray(state_tangent)
    time_tangent_ = jnp.asarray(time_tangent, dtype=time.dtype)
    if args is None:
        direct_guard = jax.jvp(
            lambda t, y: plan.guard(t, y, None),
            (time, state),
            (time_tangent_, state_tangent_),
        )[1]
        direct_reset = jax.jvp(
            lambda t, y: plan.reset(t, y, None),
            (time, state),
            (time_tangent_, state_tangent_),
        )[1]
    else:
        tangent_args = (
            jax.tree.map(jnp.zeros_like, args) if args_tangent is None else args_tangent
        )
        direct_guard = jax.jvp(
            plan.guard,
            (time, state, args),
            (time_tangent_, state_tangent_, tangent_args),
        )[1]
        direct_reset = jax.jvp(
            plan.reset,
            (time, state, args),
            (time_tangent_, state_tangent_, tangent_args),
        )[1]
    _, _, _, _, jump, denominator, _ = _event_differentials(plan, time, state, args)
    grazing = jnp.abs(denominator) <= plan.grazing_tolerance
    action = direct_reset + jump * direct_guard / jnp.where(grazing, 1.0, denominator)
    finite = jnp.all(jnp.isfinite(action)) & jnp.isfinite(denominator)
    successful = (~grazing) & finite
    return HybridEventActionResult(
        jax.tree.map(lambda value: jnp.where(successful, value, jnp.nan), action),
        denominator,
        grazing,
        finite,
        successful,
        plan.plan_id,
    )


def hybrid_event_vjp(
    plan: HybridEventPlan,
    event_time: ArrayLike,
    state_before: ArrayLike,
    cotangent: Any,
    /,
    *,
    args: Any = None,
) -> tuple[Any, Any, Any, HybridEventActionResult]:
    """Transpose the identical matrix-free event action for reverse replay."""

    time = jnp.asarray(event_time)
    state = jnp.asarray(state_before)
    zero_time = jnp.zeros_like(time)
    zero_state = jnp.zeros_like(state)
    if args is None:

        def action(time_tangent, state_tangent):
            return hybrid_event_jvp(
                plan,
                time,
                state,
                state_tangent,
                args=None,
                time_tangent=time_tangent,
            ).action

        _, pullback = jax.vjp(action, zero_time, zero_state)
        time_cotangent, state_cotangent = pullback(cotangent)
        args_cotangent = None
    else:
        zero_args = jax.tree.map(jnp.zeros_like, args)

        def action(time_tangent, state_tangent, args_tangent):
            return hybrid_event_jvp(
                plan,
                time,
                state,
                state_tangent,
                args=args,
                time_tangent=time_tangent,
                args_tangent=args_tangent,
            ).action

        _, pullback = jax.vjp(action, zero_time, zero_state, zero_args)
        time_cotangent, state_cotangent, args_cotangent = pullback(cotangent)
    evidence = hybrid_event_jvp(plan, time, state, zero_state, args=args)
    return time_cotangent, state_cotangent, args_cotangent, evidence


class HybridReplayResult(StrictModule):
    state: Array
    valid: Array
    replayed_events: Array
    event_log_abs_determinant: Array
    determinant_sign: Array
    status: Array
    schedule_id: str = eqx.field(static=True)


def replay_hybrid_events(
    events: Sequence[HybridEventPlan],
    tape: HybridEventTape,
    initial_state: ArrayLike,
    /,
    *,
    args: Any = None,
) -> HybridReplayResult:
    """Replay only the event indices/times declared by an identical fixed tape."""

    plans = tuple(events)
    if not plans or any(not isinstance(plan, HybridEventPlan) for plan in plans):
        raise TypeError("events must be a non-empty sequence of HybridEventPlan values.")
    if not isinstance(tape, HybridEventTape):
        raise TypeError("tape must be a HybridEventTape.")
    initial = jnp.asarray(initial_state)
    if initial.shape != tape.states_before.shape[1:]:
        raise ValueError(
            "initial_state shape does not match HybridEventTape state shape."
        )
    branches = tuple(
        (lambda payload, plan=plan: plan.reset(payload[0], payload[1], payload[2]))
        for plan in plans
    )

    def body(index: int, carry: tuple[Array, Array, Array]):
        state, valid, count = carry
        active = tape.active[index]

        def apply(_: None):
            event_index = tape.event_indices[index]
            index_valid = (event_index >= 0) & (event_index < len(plans))
            safe_index = jnp.clip(event_index, 0, len(plans) - 1)
            before_match = jnp.all(
                jnp.abs(state - tape.states_before[index])
                <= jnp.sqrt(jnp.finfo(state.real.dtype).eps)
            )
            reset_state = jax.lax.switch(
                safe_index,
                branches,
                (tape.event_times[index], state, args),
            )
            after_match = jnp.all(
                jnp.abs(reset_state - tape.states_after[index])
                <= jnp.sqrt(jnp.finfo(state.real.dtype).eps)
            )
            event_valid = (
                index_valid & before_match & after_match & tape.saltation_valid[index]
            )
            return reset_state, valid & event_valid, count + 1

        return jax.lax.cond(active, apply, lambda _: carry, None)

    state, valid, count = jax.lax.fori_loop(
        0,
        tape.active.shape[0],
        body,
        (initial, ~tape.capacity_exceeded, jnp.asarray(0, dtype=jnp.int32)),
    )
    valid = valid & (count == tape.event_count)
    status = jnp.where(valid, tape.status, -1)
    return HybridReplayResult(
        state,
        valid,
        count,
        tape.total_log_abs_determinant,
        tape.total_determinant_sign,
        status,
        tape.schedule_id,
    )


__all__ = [
    "HybridEventActionResult",
    "HybridEventPlan",
    "HybridEventSensitivityResult",
    "HybridEventTape",
    "HybridReplayPolicy",
    "HybridReplayResult",
    "empty_hybrid_event_tape",
    "hybrid_event_jvp",
    "hybrid_event_vjp",
    "localize_hybrid_event",
    "record_hybrid_event",
    "replay_hybrid_events",
]
