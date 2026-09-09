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
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

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


class NumericalEventResult(StrictModule):
    """Scalar numerical root and separate primal/derivative qualification."""

    event_time: Array
    state: Any
    guard_residual: Array
    transversality: Array
    bracketed: Array
    crossing: Array
    grazing: Array
    finite: Array
    successful: Array
    derivative_valid: Array


class HybridEventRootResult(StrictModule):
    """Localized physical guard/reset evidence, without a dense state Jacobian."""

    event_time: Array
    state_before: Array
    state_after: Array
    guard_residual: Array
    transversality: Array
    grazing: Array
    simultaneous: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


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


def _tree_finite(tree: Any) -> Array:
    finite = jnp.asarray(True)
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        if value.dtype != jax.dtypes.float0:
            finite = finite & jnp.all(jnp.isfinite(value))
    return finite


def _zero_tangent(value: Any) -> Array:
    value = jnp.asarray(value)
    dtype = value.dtype if jnp.issubdtype(value.dtype, jnp.inexact) else jax.dtypes.float0
    return jnp.zeros(value.shape, dtype=dtype)


def localize_numerical_event(
    state_at_time: Callable[[Array], Any],
    guard: Callable[[Array, Any], Array],
    t0: ArrayLike,
    t1: ArrayLike,
    /,
    *,
    iterations: int = 48,
    tolerance: float = 1.0e-10,
    grazing_tolerance: float = 1.0e-8,
) -> NumericalEventResult:
    """Localize a scalar root on a numerical trajectory at absolute physical time.

    For ``F(t, p) = guard(t, state_at_time(t))``, the branchwise derivative is
    ``dt/dp = -partial_p F / partial_t F``. Both partials differentiate the
    supplied numerical callback, including captured parameters, not a continuous
    vector field. Bracket bounds select a branch; they contribute no derivative
    unless captured by the callbacks. The returned state differentiates through
    the callback and the implicit root time.

    ``successful`` qualifies the primal root; ``derivative_valid`` additionally
    requires transversality. Invalid or grazing derivatives are NaN, never an
    epsilon-regularized slope. No uniqueness or hidden interior crossing is
    certified. Callers must keep the selected event branch fixed.
    """

    if not callable(state_at_time) or not callable(guard):
        raise TypeError("state_at_time and guard must be callable.")
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError("iterations must be an integer.")
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if any(not np.isfinite(v) or v <= 0 for v in (tolerance, grazing_tolerance)):
        raise ValueError("Root tolerances must be positive and finite.")
    dtype = jnp.result_type(t0, t1, 0.0)
    left = jnp.asarray(t0, dtype=dtype)
    right = jnp.asarray(t1, dtype=dtype)
    if left.shape != () or right.shape != ():
        raise ValueError("Event bracket bounds must be scalars.")

    def residual(time):
        value = jnp.asarray(guard(time, state_at_time(time)))
        if value.shape != ():
            raise ValueError("Event guard must return a scalar.")
        return value

    left_guard, right_guard = residual(left), residual(right)
    bracketed = (
        jnp.isfinite(left)
        & jnp.isfinite(right)
        & jnp.isfinite(left_guard)
        & jnp.isfinite(right_guard)
        & (left < right)
        & (
            (left_guard == 0)
            | (right_guard == 0)
            | (jnp.signbit(left_guard) != jnp.signbit(right_guard))
        )
    )

    def solve(function, initial):
        del initial

        def iteration(_, carry):
            lower, upper, lower_guard = carry
            midpoint = lower + 0.5 * (upper - lower)
            midpoint_guard = function(midpoint)
            same_side = jnp.signbit(lower_guard) == jnp.signbit(midpoint_guard)
            exact = midpoint_guard == 0
            return (
                jnp.where(same_side | exact, midpoint, lower),
                jnp.where((~same_side) | exact, midpoint, upper),
                jnp.where(same_side | exact, midpoint_guard, lower_guard),
            )

        lower, upper, _ = jax.lax.fori_loop(
            0, iterations, iteration, (left, right, left_guard)
        )
        root = lower + 0.5 * (upper - lower)
        return jnp.where(left_guard == 0, left, jnp.where(right_guard == 0, right, root))

    # Qualification is held fixed during differentiation of the root branch.
    # custom_root ignores the solve algorithm and differentiates only residual.
    primal_time = jax.lax.stop_gradient(solve(residual, left))
    primal_valid = (
        bracketed
        & (jnp.abs(residual(primal_time)) <= tolerance)
        & _tree_finite(state_at_time(primal_time))
    )

    def tangent_solve(linearized, rhs):
        slope = linearized(jnp.ones_like(rhs))
        valid = primal_valid & jnp.isfinite(slope) & (jnp.abs(slope) > grazing_tolerance)
        return rhs / jnp.where(valid, slope, jnp.nan)

    event_time = jax.lax.custom_root(
        residual, primal_time, lambda function, initial: initial, tangent_solve
    )
    state = state_at_time(event_time)
    guard_residual = jnp.abs(residual(event_time))
    transversality = jax.jvp(residual, (event_time,), (jnp.ones_like(event_time),))[1]
    grazing = jnp.abs(transversality) <= grazing_tolerance
    finite = _tree_finite(state) & jnp.isfinite(event_time) & jnp.isfinite(guard_residual)
    successful = bracketed & finite & (guard_residual <= tolerance)
    derivative_valid = successful & jnp.isfinite(transversality) & (~grazing)
    crossing = jnp.where(bracketed, jnp.sign(right_guard - left_guard), 0).astype(
        jnp.int32
    )
    return NumericalEventResult(
        event_time,
        state,
        guard_residual,
        transversality,
        bracketed,
        crossing,
        grazing,
        finite,
        successful,
        derivative_valid,
    )


def _event_directional_data(
    plan: HybridEventPlan, time: Array, state: Array, args: Any, /
) -> tuple[Array, Array, Array]:
    before = jnp.asarray(plan.vector_field_before(time, state, args))
    state_after, reset_flow = jax.jvp(
        lambda t, y: plan.reset(t, y, args),
        (time, state),
        (jnp.ones_like(time), before),
    )
    guard_value, denominator = jax.jvp(
        lambda t, y: plan.guard(t, y, args),
        (time, state),
        (jnp.ones_like(time), before),
    )
    if jnp.shape(guard_value) != ():
        raise ValueError("Hybrid guard must return a scalar.")
    after = jnp.asarray(plan.vector_field_after(time, state_after, args))
    return state_after, after - reset_flow, denominator


def _simultaneous_event(plan, time, state, args):
    simultaneous = jnp.asarray(False)
    for competing in plan.competing_guards:
        simultaneous = simultaneous | (
            jnp.abs(competing(time, state, args)) <= plan.event_tolerance
        )
    return simultaneous


def localize_hybrid_event_root(
    plan: HybridEventPlan,
    state_at_time: Callable[[Array, Any], Array],
    left_time: ArrayLike,
    right_time: ArrayLike,
    /,
    *,
    args: Any = None,
) -> HybridEventRootResult:
    """Localize/reset without dense saltation or density/logdet construction.

    Root derivatives use the numerical callback; physical saltation eligibility
    uses the declared vector fields. These are distinct derivative contracts.
    """

    if not isinstance(plan, HybridEventPlan):
        raise TypeError("plan must be a HybridEventPlan.")
    if not callable(state_at_time):
        raise TypeError("state_at_time must be callable.")
    root = localize_numerical_event(
        lambda time: state_at_time(time, args),
        lambda time, state: plan.guard(time, state, args),
        left_time,
        right_time,
        iterations=plan.bisection_iterations,
        tolerance=plan.event_tolerance,
        grazing_tolerance=plan.grazing_tolerance,
    )
    state_before = jnp.asarray(root.state)
    state_after, jump, transversality = _event_directional_data(
        plan, root.event_time, state_before, args
    )
    grazing = jnp.abs(transversality) <= plan.grazing_tolerance
    simultaneous = _simultaneous_event(plan, root.event_time, state_before, args)
    successful = (
        root.successful
        & (~grazing)
        & (~simultaneous)
        & _tree_finite((state_after, jump, transversality))
    )
    return HybridEventRootResult(
        root.event_time,
        state_before,
        state_after,
        root.guard_residual,
        transversality,
        grazing,
        simultaneous,
        successful,
        plan.plan_id,
    )


def localize_hybrid_event(
    plan: HybridEventPlan,
    state_at_time: Callable[[Array, Any], Array],
    left_time: Array,
    right_time: Array,
    /,
    *,
    args: Any = None,
) -> HybridEventSensitivityResult:
    """Localize a root and explicitly construct dense physical saltation evidence.

    Density consumers retain determinant evidence here. Forward-only callers
    should use ``localize_hybrid_event_root``; directional derivatives should use
    ``hybrid_event_jvp``/``hybrid_event_vjp``, neither of which builds this matrix.
    """

    root = localize_hybrid_event_root(
        plan, state_at_time, left_time, right_time, args=args
    )
    time, state = root.event_time, root.state_before
    _, jump, denominator = _event_directional_data(plan, time, state, args)
    normal = jax.grad(lambda value: plan.guard(time, value, args))(state)
    reset_jacobian = jax.jacfwd(lambda value: plan.reset(time, value, args))(state)
    saltation = reset_jacobian + ein.contract(
        "...i,...j->...ij", jump, normal
    ) / jnp.where(root.successful, denominator, jnp.nan)
    successful = root.successful & _tree_finite(saltation)
    saltation = jnp.where(successful, saltation, jnp.nan)
    square = (
        reset_jacobian.ndim == 2 and reset_jacobian.shape[0] == reset_jacobian.shape[1]
    )
    if square:
        factorization = factorize(
            DenseLinearOperator(
                saltation, operator_id=f"{plan.plan_id}:saltation-matrix"
            ),
            FactorizationPolicy("lu"),
        )
        determinant_sign = factorization.determinant_sign()
        log_abs_determinant = factorization.log_abs_determinant()
        log_valid = (
            successful & jnp.isfinite(log_abs_determinant) & (determinant_sign != 0)
        )
    else:
        determinant_sign = jnp.asarray(jnp.nan, dtype=state.real.dtype)
        log_abs_determinant = jnp.asarray(jnp.nan, dtype=state.real.dtype)
        log_valid = jnp.asarray(False)
    return HybridEventSensitivityResult(
        time,
        state,
        root.state_after,
        saltation,
        root.guard_residual,
        denominator,
        root.grazing,
        root.simultaneous,
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
    """Apply physical fixed-epoch saltation, including time/parameter directions.

    Uses directional reset and guard derivatives only, with O(state size)
    storage. Invalid primal evidence or nonfinite tangents produce NaN actions.
    This is not the derivative of a numerical integrator's interpolant.
    """

    if not isinstance(plan, HybridEventPlan):
        raise TypeError("plan must be a HybridEventPlan.")
    time = jnp.asarray(event_time, dtype=jnp.result_type(event_time, 0.0))
    state = jnp.asarray(state_before)
    state_tangent_ = jnp.asarray(state_tangent, dtype=state.dtype)
    time_tangent_ = jnp.asarray(time_tangent, dtype=time.dtype)
    if args is None:
        if args_tangent is not None:
            raise ValueError("args_tangent requires args.")
        primals = (time, state)
        tangents = (time_tangent_, state_tangent_)
        function = lambda t, y: (plan.reset(t, y, None), plan.guard(t, y, None))
    else:
        tangent_args = (
            jax.tree.map(_zero_tangent, args) if args_tangent is None else args_tangent
        )
        primals = (time, state, args)
        tangents = (time_tangent_, state_tangent_, tangent_args)
        function = lambda t, y, a: (plan.reset(t, y, a), plan.guard(t, y, a))
    _, (direct_reset, direct_guard) = jax.jvp(function, primals, tangents)
    state_after, jump, denominator = _event_directional_data(plan, time, state, args)
    grazing = jnp.abs(denominator) <= plan.grazing_tolerance
    primal_valid = (
        _tree_finite((time, state, state_after, jump, denominator))
        & (jnp.abs(plan.guard(time, state, args)) <= plan.event_tolerance)
        & (~_simultaneous_event(plan, time, state, args))
        & (~grazing)
    )
    action = direct_reset + jump * direct_guard / jnp.where(
        primal_valid, denominator, jnp.nan
    )
    finite = _tree_finite((tangents, action, denominator))
    successful = primal_valid & finite
    return HybridEventActionResult(
        jnp.where(successful, action, jnp.nan),
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
    """Transpose physical saltation with direct guard/reset pullbacks.

    Returns time, state and parameter cotangents and qualification evidence.
    Invalid evidence explicitly poisons inexact cotangents, including at grazing;
    transposing a masked JVP would incorrectly return zeros on invalid branches.
    """

    if not isinstance(plan, HybridEventPlan):
        raise TypeError("plan must be a HybridEventPlan.")
    time = jnp.asarray(event_time, dtype=jnp.result_type(event_time, 0.0))
    state = jnp.asarray(state_before)
    cotangent = jnp.asarray(cotangent)
    state_after, jump, denominator = _event_directional_data(plan, time, state, args)
    grazing = jnp.abs(denominator) <= plan.grazing_tolerance
    finite = _tree_finite((time, state, state_after, jump, denominator, cotangent))
    valid = (
        finite
        & (~grazing)
        & (jnp.abs(plan.guard(time, state, args)) <= plan.event_tolerance)
        & (~_simultaneous_event(plan, time, state, args))
    )
    guard_cotangent = jnp.vdot(jump, cotangent) / jnp.where(valid, denominator, jnp.nan)
    if args is None:
        _, pullback = jax.vjp(
            lambda t, y: (plan.reset(t, y, None), plan.guard(t, y, None)), time, state
        )
        time_cotangent, state_cotangent = pullback((cotangent, guard_cotangent))
        args_cotangent = None
    else:
        _, pullback = jax.vjp(
            lambda t, y, a: (plan.reset(t, y, a), plan.guard(t, y, a)),
            time,
            state,
            args,
        )
        time_cotangent, state_cotangent, args_cotangent = pullback(
            (cotangent, guard_cotangent)
        )
    cotangents = (time_cotangent, state_cotangent, args_cotangent)
    finite = finite & _tree_finite(cotangents)
    successful = valid & finite

    def qualify(value):
        if value.dtype == jax.dtypes.float0:
            return value
        return jnp.where(successful, value, jnp.nan)

    time_cotangent, state_cotangent, args_cotangent = jax.tree.map(qualify, cotangents)
    evidence = HybridEventActionResult(
        state_cotangent, denominator, grazing, finite, successful, plan.plan_id
    )
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
    "HybridEventRootResult",
    "HybridEventSensitivityResult",
    "HybridEventTape",
    "HybridReplayPolicy",
    "HybridReplayResult",
    "NumericalEventResult",
    "empty_hybrid_event_tape",
    "hybrid_event_jvp",
    "hybrid_event_vjp",
    "localize_hybrid_event",
    "localize_hybrid_event_root",
    "localize_numerical_event",
    "record_hybrid_event",
    "replay_hybrid_events",
]
