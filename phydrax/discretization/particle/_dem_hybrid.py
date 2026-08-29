#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class DEMHybridEventKind(StrEnum):
    CONTACT_ONSET = "contact_onset"
    CONTACT_SEPARATION = "contact_separation"
    STICK_TO_SLIP = "stick_to_slip"
    SLIP_TO_STICK = "slip_to_stick"
    USER = "user"


class DEMHybridEventPlan(StrictModule, NonTrainableState):
    guard: Callable[[Array, Any], Array]
    reset: Callable[[Array, Any], Array]
    vector_field_before: Callable[[Array, Any], Array]
    vector_field_after: Callable[[Array, Any], Array]
    competing_guards: tuple[Callable[[Array, Any], Array], ...]
    kind: DEMHybridEventKind = eqx.field(static=True)
    grazing_tolerance: float = eqx.field(static=True)
    event_tolerance: float = eqx.field(static=True)
    bisection_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        guard: Callable[[Array, Any], Array],
        reset: Callable[[Array, Any], Array],
        vector_field_before: Callable[[Array, Any], Array],
        vector_field_after: Callable[[Array, Any], Array],
        /,
        *,
        kind: DEMHybridEventKind,
        competing_guards: Sequence[Callable[[Array, Any], Array]] = (),
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
        if not isinstance(kind, DEMHybridEventKind):
            raise TypeError("kind must be a DEMHybridEventKind.")
        grazing = float(grazing_tolerance)
        tolerance = float(event_tolerance)
        iterations = int(bisection_iterations)
        identifier = str(plan_id)
        if (
            not np.isfinite(grazing)
            or grazing <= 0.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or iterations < 8
            or not identifier
        ):
            raise ValueError("Hybrid-event tolerances, iterations, or ID are invalid.")
        self.guard = guard
        self.reset = reset
        self.vector_field_before = vector_field_before
        self.vector_field_after = vector_field_after
        self.competing_guards = competing
        self.kind = kind
        self.grazing_tolerance = grazing
        self.event_tolerance = tolerance
        self.bisection_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dem-hybrid-event-plan",
                "user_id": identifier,
                "event_kind": kind.value,
                "grazing_tolerance": grazing,
                "event_tolerance": tolerance,
                "bisection_iterations": iterations,
                "competing_count": len(competing),
            }
        )


class DEMHybridSensitivityResult(StrictModule):
    event_time: Array
    state_before: Array
    state_after: Array
    saltation_matrix: Array
    guard_residual: Array
    transversality: Array
    grazing: Array
    simultaneous: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def localize_and_differentiate_hybrid_event(
    plan: DEMHybridEventPlan,
    state_at_time: Callable[[Array, Any], Array],
    left_time: Array,
    right_time: Array,
    /,
    *,
    args: Any = None,
) -> DEMHybridSensitivityResult:
    """Localize one transverse event and return its saltation matrix."""

    if not isinstance(plan, DEMHybridEventPlan):
        raise TypeError("plan must be a DEMHybridEventPlan.")
    if not callable(state_at_time):
        raise TypeError("state_at_time must be callable.")
    left = jnp.asarray(left_time)
    right = jnp.asarray(right_time, dtype=left.dtype)
    left_state = state_at_time(left, args)
    right_state = state_at_time(right, args)
    left_guard = jnp.asarray(plan.guard(left_state, args))
    right_guard = jnp.asarray(plan.guard(right_state, args))
    if left_guard.shape != () or right_guard.shape != ():
        raise ValueError("Hybrid guard must return a scalar.")
    bracketed = (
        jnp.isfinite(left_guard)
        & jnp.isfinite(right_guard)
        & (left < right)
        & (left_guard * right_guard <= 0.0)
    )

    def iteration(_, carry):
        lower, upper, lower_guard = carry
        midpoint = 0.5 * (lower + upper)
        midpoint_guard = plan.guard(state_at_time(midpoint, args), args)
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
    state_before = state_at_time(event_time, args)
    state_after = plan.reset(state_before, args)
    guard_residual = jnp.abs(plan.guard(state_before, args))
    normal = jax.grad(lambda state: plan.guard(state, args))(state_before)
    reset_jacobian = jax.jacfwd(lambda state: plan.reset(state, args))(state_before)
    before = plan.vector_field_before(state_before, args)
    after = plan.vector_field_after(state_after, args)
    transversality = jnp.dot(normal, before)
    grazing = jnp.abs(transversality) <= plan.grazing_tolerance
    reset_before = reset_jacobian @ before
    saltation = reset_jacobian + jnp.outer(after - reset_before, normal) / jnp.where(
        grazing, 1.0, transversality
    )
    simultaneous = jnp.asarray(False)
    for competing in plan.competing_guards:
        simultaneous = simultaneous | (
            jnp.abs(competing(state_before, args)) <= plan.event_tolerance
        )
    finite = (
        jnp.all(jnp.isfinite(state_before))
        & jnp.all(jnp.isfinite(state_after))
        & jnp.all(jnp.isfinite(saltation))
    )
    successful = (
        bracketed
        & ~grazing
        & ~simultaneous
        & finite
        & (guard_residual <= plan.event_tolerance)
    )
    saltation = jnp.where(successful, saltation, jnp.nan)
    return DEMHybridSensitivityResult(
        event_time,
        state_before,
        state_after,
        saltation,
        guard_residual,
        transversality,
        grazing,
        simultaneous,
        successful,
        plan.plan_id,
    )


__all__ = [
    "DEMHybridEventKind",
    "DEMHybridEventPlan",
    "DEMHybridSensitivityResult",
    "localize_and_differentiate_hybrid_event",
]
