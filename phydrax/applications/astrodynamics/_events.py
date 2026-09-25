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
from ..._identity import callable_payload
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import (
    HybridEventPlan,
    HybridEventSensitivityResult,
    HybridGuardPlan,
    localize_hybrid_event,
)
from ._context import AstrodynamicsContext
from ._status import AstrodynamicsStatus


def _callable_identity(
    value: Callable[..., Any],
    semantic_id: str | None,
    numeric_id: str | None,
    /,
) -> dict[str, str]:
    payload = callable_payload(value, semantic_id=semantic_id, numeric_id=numeric_id)
    return {
        "semantic": payload["semantic_content_id"],
        "numeric": payload["numeric_content_id"],
    }


def _competing_guard_ids(
    ids: Sequence[str] | None,
    count: int,
    name: str,
    /,
) -> tuple[str | None, ...]:
    if ids is None:
        return (None,) * count
    if isinstance(ids, str):
        raise TypeError(f"{name} must be a sequence of strings.")
    values = tuple(ids)
    if len(values) != count:
        raise ValueError(f"{name} must align with competing_guards.")
    return values


class AstrodynamicsEventPlan(StrictModule, NonTrainableState):
    hybrid: HybridEventPlan
    context: AstrodynamicsContext
    direction: int = eqx.field(static=True)
    terminal: bool = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        guard: Callable[[Array, Array, Any], Array],
        reset: Callable[[Array, Array, Any], Array],
        vector_field_before: Callable[[Array, Array, Any], Array],
        vector_field_after: Callable[[Array, Array, Any], Array],
        context: AstrodynamicsContext,
        /,
        *,
        event_kind: str,
        direction: int = 0,
        terminal: bool = False,
        competing_guards: Sequence[Callable[[Array, Array, Any], Array]] = (),
        grazing_tolerance: float = 1.0e-10,
        event_tolerance: float = 1.0e-10,
        bisection_iterations: int = 64,
        guard_semantic_id: str | None = None,
        guard_numeric_id: str | None = None,
        reset_semantic_id: str | None = None,
        reset_numeric_id: str | None = None,
        vector_field_before_semantic_id: str | None = None,
        vector_field_before_numeric_id: str | None = None,
        vector_field_after_semantic_id: str | None = None,
        vector_field_after_numeric_id: str | None = None,
        competing_guard_semantic_ids: Sequence[str] | None = None,
        competing_guard_numeric_ids: Sequence[str] | None = None,
    ):
        """Bind one hybrid event; opaque callables require semantic and numeric IDs.

        StrictModule callables and plain module-level functions are identified by
        content. Every other callable must be declared through its role's
        ``*_semantic_id``/``*_numeric_id`` pair (index-aligned sequences for
        ``competing_guards``); missing identities raise ``TypeError``.
        """
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        if isinstance(direction, bool) or not isinstance(direction, int):
            raise TypeError("event direction must be an integer.")
        direction_ = direction
        if direction_ not in (-1, 0, 1):
            raise ValueError("event direction must be -1, 0, or +1.")
        if not isinstance(terminal, bool):
            raise TypeError("terminal must be a bool.")
        kind = str(event_kind).strip()
        if not kind:
            raise ValueError("event_kind must be non-empty.")
        competing = tuple(competing_guards)
        competing_semantic = _competing_guard_ids(
            competing_guard_semantic_ids,
            len(competing),
            "competing_guard_semantic_ids",
        )
        competing_numeric = _competing_guard_ids(
            competing_guard_numeric_ids,
            len(competing),
            "competing_guard_numeric_ids",
        )
        callable_identity = canonical_fingerprint(
            {
                "kind": "astrodynamics-event-callables",
                "guard": _callable_identity(guard, guard_semantic_id, guard_numeric_id),
                "reset": _callable_identity(reset, reset_semantic_id, reset_numeric_id),
                "vector_field_before": _callable_identity(
                    vector_field_before,
                    vector_field_before_semantic_id,
                    vector_field_before_numeric_id,
                ),
                "vector_field_after": _callable_identity(
                    vector_field_after,
                    vector_field_after_semantic_id,
                    vector_field_after_numeric_id,
                ),
                "competing_guards": [
                    _callable_identity(candidate, semantic_id, numeric_id)
                    for candidate, semantic_id, numeric_id in zip(
                        competing, competing_semantic, competing_numeric, strict=True
                    )
                ],
            }
        )
        guard_plan = HybridGuardPlan(
            guard,
            direction=direction_,
            terminal=terminal,
            guard_id=canonical_fingerprint(
                {
                    "kind": "astrodynamics-hybrid-guard",
                    "context": context.context_id,
                    "event_kind": kind,
                    "callables": callable_identity,
                }
            ),
        )
        hybrid = HybridEventPlan(
            guard_plan,
            reset,
            vector_field_before,
            vector_field_after,
            competing_guards=competing,
            grazing_tolerance=grazing_tolerance,
            event_tolerance=event_tolerance,
            bisection_iterations=bisection_iterations,
            dense_diagnostics=True,
            max_dense_dimension=32,
            plan_id=canonical_fingerprint(
                {
                    "kind": "astrodynamics-hybrid-event",
                    "context": context.context_id,
                    "event_kind": kind,
                    "callables": callable_identity,
                }
            ),
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
                "callables": callable_identity,
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
    def __call__(self, time: Array, state: Array, args: Any, /) -> Array:
        del time, args
        return state


class ImpulsiveVelocityReset(StrictModule):
    delta_velocity: Array

    def __init__(self, delta_velocity: ArrayLike, /):
        value = jnp.asarray(delta_velocity)
        if value.shape != (3,) or not bool(jnp.all(jnp.isfinite(value))):
            raise ValueError("delta_velocity must be a finite vector with shape (3,).")
        self.delta_velocity = value

    def __call__(self, time: Array, state: Array, args: Any, /) -> Array:
        del time, args
        if state.shape != (6,):
            raise ValueError("Impulsive reset state must have shape (6,).")
        return state.at[3:].add(self.delta_velocity)


class RadiusGuard(StrictModule):
    radius: Array

    def __init__(self, radius: ArrayLike, /):
        value = jnp.asarray(radius).reshape(())
        if not bool(jnp.isfinite(value)) or not bool(value > 0.0):
            raise ValueError("radius must be finite and positive.")
        self.radius = value

    def __call__(self, time: Array, state: Array, args: Any, /) -> Array:
        del time, args
        return jnp.sqrt(jnp.sum(state[:3] ** 2)) - self.radius


class ApsisGuard(StrictModule):
    def __call__(self, time: Array, state: Array, args: Any, /) -> Array:
        del time, args
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
        normal_host = np.asarray(normal, dtype=np.float64)
        offset_host = np.asarray(offset, dtype=np.float64)
        if normal_host.shape != (3,) or offset_host.shape != (3,):
            raise ValueError("Plane normal and offset must have shape (3,).")
        norm = float(np.sqrt(np.sum(normal_host * normal_host)))
        if not np.isfinite(norm) or norm <= 0.0 or np.any(~np.isfinite(offset_host)):
            raise ValueError("Plane geometry must be finite and nondegenerate.")
        self.normal = jnp.asarray(normal_host / norm)
        self.offset = jnp.asarray(offset_host)

    def __call__(self, time: Array, state: Array, args: Any, /) -> Array:
        del time, args
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
    left_guard = plan.hybrid.guard_plan.guard(left, state_at_time(left, args), args)
    right_guard = plan.hybrid.guard_plan.guard(right, state_at_time(right, args), args)
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
    finite = (
        jnp.isfinite(left)
        & jnp.isfinite(right)
        & jnp.isfinite(left_guard)
        & jnp.isfinite(right_guard)
        & jnp.isfinite(sensitivity.event_time)
        & jnp.all(jnp.isfinite(sensitivity.state_before))
        & jnp.all(jnp.isfinite(sensitivity.state_after))
        & jnp.isfinite(sensitivity.guard_residual)
        & jnp.isfinite(sensitivity.transversality)
    )
    bracketed = (
        (left_guard == 0.0)
        | (right_guard == 0.0)
        | (jnp.signbit(left_guard) != jnp.signbit(right_guard))
    )
    valid = sensitivity.successful & direction_valid & finite
    status = jnp.where(
        ~finite,
        int(AstrodynamicsStatus.NONFINITE_INPUT),
        jnp.where(
            sensitivity.grazing | sensitivity.simultaneous,
            int(AstrodynamicsStatus.SINGULAR_GEOMETRY),
            jnp.where(
                ~bracketed | ~direction_valid,
                int(AstrodynamicsStatus.NO_SOLUTION),
                jnp.where(
                    sensitivity.successful,
                    int(AstrodynamicsStatus.SUCCESS),
                    int(AstrodynamicsStatus.NONCONVERGED),
                ),
            ),
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
