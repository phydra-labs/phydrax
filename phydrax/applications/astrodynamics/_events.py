#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import functools
import hashlib
import inspect
import marshal
from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from types import ModuleType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import HybridEventPlan, HybridEventSensitivityResult, localize_hybrid_event
from ._context import AstrodynamicsContext
from ._status import AstrodynamicsStatus


def _stable_value_payload(value: Any, /) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return {
            "numpy_scalar": {
                "dtype": value.dtype.str,
                "value": _stable_value_payload(value.item()),
            }
        }
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, (np.ndarray, jax.Array)):
        return {"array": array_tree_fingerprint(value)}
    if isinstance(value, (tuple, list)):
        return [_stable_value_payload(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_stable_value_payload(item) for item in value]
        return {"set": sorted(items, key=canonical_fingerprint)}
    if isinstance(value, dict):
        items = [
            (_stable_value_payload(key), _stable_value_payload(item))
            for key, item in value.items()
        ]
        return {
            "mapping": sorted(
                items,
                key=lambda item: canonical_fingerprint(item[0]),
            )
        }
    if isinstance(value, functools.partial):
        return {
            "partial": _callable_payload(value.func),
            "args": _stable_value_payload(value.args),
            "keywords": _stable_value_payload(value.keywords),
        }
    if inspect.ismethod(value):
        return {
            "method": _callable_payload(value.__func__),
            "self": _stable_value_payload(value.__self__),
        }
    if inspect.isfunction(value):
        return _callable_payload(value)
    if is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _stable_value_payload(getattr(value, field.name))
                for field in fields(value)
            },
        }
    raise ValueError(
        "Event callables with opaque state require a stable_id supplied by the caller."
    )


def _global_value_payload(value: Any, /) -> Any:
    if isinstance(value, ModuleType):
        return {"module": value.__name__}
    if inspect.isfunction(value):
        return {
            "function": f"{value.__module__}.{value.__qualname__}",
            "code": hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
        }
    if inspect.isbuiltin(value):
        return {
            "builtin": f"{value.__module__}.{value.__qualname__}",
        }
    if inspect.isclass(value):
        return {"class": f"{value.__module__}.{value.__qualname__}"}
    return _stable_value_payload(value)


def _callable_payload(value: Callable[..., Any], /) -> dict[str, Any]:
    if inspect.isfunction(value):
        closure = (
            ()
            if value.__closure__ is None
            else tuple(cell.cell_contents for cell in value.__closure__)
        )
        closure_variables = inspect.getclosurevars(value)
        return {
            "kind": "function",
            "module": value.__module__,
            "qualname": value.__qualname__,
            "code": hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
            "defaults": _stable_value_payload(value.__defaults__),
            "kwdefaults": _stable_value_payload(value.__kwdefaults__),
            "closure": _stable_value_payload(closure),
            "globals": {
                name: _global_value_payload(item)
                for name, item in sorted(closure_variables.globals.items())
            },
        }
    payload = _stable_value_payload(value)
    if not isinstance(payload, dict):
        raise ValueError("Event callable identity could not be represented.")
    return payload


def _event_callable_identity(
    guard: Callable[..., Any],
    reset: Callable[..., Any],
    vector_field_before: Callable[..., Any],
    vector_field_after: Callable[..., Any],
    competing_guards: tuple[Callable[..., Any], ...],
    stable_id: str | None,
    /,
) -> str:
    if stable_id is not None:
        identifier = str(stable_id).strip()
        if not identifier:
            raise ValueError("stable_id must be non-empty when supplied.")
        return canonical_fingerprint({"kind": "caller-event-id", "id": identifier})
    return canonical_fingerprint(
        {
            "kind": "astrodynamics-event-callables",
            "guard": _callable_payload(guard),
            "reset": _callable_payload(reset),
            "vector_field_before": _callable_payload(vector_field_before),
            "vector_field_after": _callable_payload(vector_field_after),
            "competing_guards": [
                _callable_payload(candidate) for candidate in competing_guards
            ],
        }
    )


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
        stable_id: str | None = None,
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
        competing = tuple(competing_guards)
        callable_identity = _event_callable_identity(
            guard,
            reset,
            vector_field_before,
            vector_field_after,
            competing,
            stable_id,
        )
        hybrid = HybridEventPlan(
            guard,
            reset,
            vector_field_before,
            vector_field_after,
            event_kind=kind,
            competing_guards=competing,
            grazing_tolerance=grazing_tolerance,
            event_tolerance=event_tolerance,
            bisection_iterations=bisection_iterations,
            plan_id=canonical_fingerprint(
                {
                    "kind": "astrodynamics-hybrid-event",
                    "context": context.context_id,
                    "event_kind": kind,
                    "callables": callable_identity,
                    "direction": direction_,
                    "terminal": terminal,
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
        normal_host = np.asarray(normal, dtype=float)
        offset_host = np.asarray(offset, dtype=float)
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
    left_guard = plan.hybrid.guard(left, state_at_time(left, args), args)
    right_guard = plan.hybrid.guard(right, state_at_time(right, args), args)
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
