#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import HybridEventPlan, HybridEventSensitivityResult, localize_hybrid_event
from ._context import AstrodynamicsContext
from ._status import AstrodynamicsStatus


class AstrodynamicsEventPlan(StrictModule, NonTrainableState):
    hybrid: HybridEventPlan
    context: AstrodynamicsContext
    direction: int = eqx.field(static=True)
    terminal: bool = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        guard: Callable[[Array, Any], Array],
        reset: Callable[[Array, Any], Array],
        vector_field_before: Callable[[Array, Any], Array],
        vector_field_after: Callable[[Array, Any], Array],
        context: AstrodynamicsContext,
        /,
        *,
        event_kind: str,
        direction: int = 0,
        terminal: bool = False,
        competing_guards: Sequence[Callable[[Array, Any], Array]] = (),
        grazing_tolerance: float = 1.0e-10,
        event_tolerance: float = 1.0e-10,
        bisection_iterations: int = 64,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        direction_ = int(direction)
        if direction_ not in (-1, 0, 1):
            raise ValueError("event direction must be -1, 0, or +1.")
        if not isinstance(terminal, bool):
            raise TypeError("terminal must be a bool.")
        kind = str(event_kind).strip()
        if not kind:
            raise ValueError("event_kind must be non-empty.")
        hybrid = HybridEventPlan(
            guard,
            reset,
            vector_field_before,
            vector_field_after,
            event_kind=kind,
            competing_guards=competing_guards,
            grazing_tolerance=grazing_tolerance,
            event_tolerance=event_tolerance,
            bisection_iterations=bisection_iterations,
        )
        self.hybrid = hybrid
        self.context = context
        self.direction = direction_
        self.terminal = terminal
        self.event_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-event",
                "hybrid": hybrid.plan_id,
                "context": context.context_id,
                "direction": direction_,
                "terminal": terminal,
            }
        )


class AstrodynamicsEventResult(StrictModule):
    sensitivity: HybridEventSensitivityResult
    direction_valid: Array
    valid: Array
    status: Array
    event_id: str = eqx.field(static=True)
    context_id: str = eqx.field(static=True)


class IdentityReset(StrictModule):
    def __call__(self, state: Array, args: Any, /) -> Array:
        del args
        return state


class ImpulsiveVelocityReset(StrictModule):
    delta_velocity: Array

    def __init__(self, delta_velocity: ArrayLike, /):
        value = jnp.asarray(delta_velocity)
        if value.shape != (3,):
            raise ValueError("delta_velocity must have shape (3,).")
        self.delta_velocity = value

    def __call__(self, state: Array, args: Any, /) -> Array:
        del args
        if state.shape != (6,):
            raise ValueError("Impulsive reset state must have shape (6,).")
        return state.at[3:].add(self.delta_velocity)


class RadiusGuard(StrictModule):
    radius: Array

    def __init__(self, radius: ArrayLike, /):
        value = jnp.asarray(radius).reshape(())
        self.radius = value

    def __call__(self, state: Array, args: Any, /) -> Array:
        del args
        return jnp.sqrt(jnp.sum(state[:3] ** 2)) - self.radius


class ApsisGuard(StrictModule):
    def __call__(self, state: Array, args: Any, /) -> Array:
        del args
        return jnp.sum(state[:3] * state[3:])


class PlaneGuard(StrictModule):
    normal: Array
    offset: Array

    def __init__(
        self,
        normal: ArrayLike,
        offset: ArrayLike | tuple[float, float, float] = (0.0, 0.0, 0.0),
        /,
    ):
        normal_host = np.asarray(normal, dtype=float)
        offset_host = np.asarray(offset, dtype=float)
        if normal_host.shape != (3,) or offset_host.shape != (3,):
            raise ValueError("Plane normal and offset must have shape (3,).")
        norm = float(np.sqrt(np.sum(normal_host * normal_host)))
        if not np.isfinite(norm) or norm <= 0.0 or np.any(~np.isfinite(offset_host)):
            raise ValueError("Plane geometry must be finite and nondegenerate.")
        self.normal = jnp.asarray(normal_host / norm)
        self.offset = jnp.asarray(offset_host)

    def __call__(self, state: Array, args: Any, /) -> Array:
        del args
        return jnp.sum(self.normal * (state[:3] - self.offset))


def localize_astrodynamics_event(
    plan: AstrodynamicsEventPlan,
    state_at_time: Callable[[Array, Any], Array],
    left_time: ArrayLike,
    right_time: ArrayLike,
    /,
    *,
    args: Any = None,
) -> AstrodynamicsEventResult:
    if not isinstance(plan, AstrodynamicsEventPlan):
        raise TypeError("plan must be an AstrodynamicsEventPlan.")
    left = jnp.asarray(left_time)
    right = jnp.asarray(right_time, dtype=left.dtype)
    left_guard = plan.hybrid.guard(state_at_time(left, args), args)
    right_guard = plan.hybrid.guard(state_at_time(right, args), args)
    sensitivity = localize_hybrid_event(
        plan.hybrid,
        state_at_time,
        left,
        right,
        args=args,
    )
    direction_valid = jnp.asarray(plan.direction == 0) | (
        (jnp.asarray(plan.direction == 1) & (right_guard > left_guard))
        | (jnp.asarray(plan.direction == -1) & (right_guard < left_guard))
    )
    valid = sensitivity.successful & direction_valid
    status = jnp.where(
        valid,
        int(AstrodynamicsStatus.SUCCESS),
        jnp.where(
            sensitivity.grazing | sensitivity.simultaneous,
            int(AstrodynamicsStatus.SINGULAR_GEOMETRY),
            int(AstrodynamicsStatus.NO_SOLUTION),
        ),
    ).astype(jnp.int32)
    return AstrodynamicsEventResult(
        sensitivity,
        direction_valid,
        valid,
        status,
        plan.event_id,
        plan.context.context_id,
    )


__all__ = [
    "ApsisGuard",
    "AstrodynamicsEventPlan",
    "AstrodynamicsEventResult",
    "IdentityReset",
    "ImpulsiveVelocityReset",
    "PlaneGuard",
    "RadiusGuard",
    "localize_astrodynamics_event",
]
