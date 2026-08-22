#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule


def _event_shape(value, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("event_shape dimensions must be positive.")
    return shape


def _leading_shape(array: Array, event_shape: tuple[int, ...], /) -> tuple[int, ...]:
    rank = len(event_shape)
    if rank and (array.ndim < rank or tuple(array.shape[-rank:]) != event_shape):
        raise ValueError(
            f"Endpoint values must end in event shape {event_shape}; got {array.shape}."
        )
    return tuple(array.shape[:-rank]) if rank else tuple(array.shape)


def _event_axes(array: Array, event_shape: tuple[int, ...], /) -> tuple[int, ...]:
    rank = len(event_shape)
    return tuple(range(array.ndim - rank, array.ndim)) if rank else ()


class EndpointInterpolantEvaluation(StrictModule):
    """One endpoint-interpolant evaluation without density or inverse claims."""

    time: Array
    state: Array
    conditional_velocity: Array
    valid: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)
    interpolant_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        time: ArrayLike,
        state: ArrayLike,
        conditional_velocity: ArrayLike,
        valid: ArrayLike,
        event_shape,
        interpolant_id: str,
    ):
        events = _event_shape(event_shape)
        state_array = jnp.asarray(state)
        velocity = jnp.asarray(conditional_velocity)
        if state_array.shape != velocity.shape:
            raise ValueError("Interpolant state and velocity shapes must match.")
        leading = _leading_shape(state_array, events)
        time_array = jnp.asarray(time)
        if time_array.shape not in ((), leading):
            raise ValueError(
                f"Interpolant time must be scalar or have leading shape {leading}; "
                f"got {time_array.shape}."
            )
        validity = jnp.asarray(valid, dtype=bool)
        if validity.shape != leading:
            raise ValueError(
                f"Interpolant validity must have shape {leading}; got {validity.shape}."
            )
        if not isinstance(interpolant_id, str) or not interpolant_id:
            raise ValueError("interpolant_id must be a non-empty string.")
        self.time = time_array
        self.state = state_array
        self.conditional_velocity = velocity
        self.valid = validity
        self.event_shape = events
        self.interpolant_id = interpolant_id

    @property
    def event_size(self) -> int:
        return prod(self.event_shape)


class AbstractEndpointInterpolant(StrictModule):
    """Deterministic state/velocity interpolation between paired endpoints."""

    event_shape: AbstractAttribute[tuple[int, ...]]
    source_coordinate: AbstractAttribute[Array]
    target_coordinate: AbstractAttribute[Array]
    interpolant_id: AbstractAttribute[str]

    @abstractmethod
    def evaluate(
        self,
        time: ArrayLike,
        source: ArrayLike,
        target: ArrayLike,
        /,
    ) -> EndpointInterpolantEvaluation:
        raise NotImplementedError


class LinearEndpointInterpolant(AbstractEndpointInterpolant):
    """Affine endpoint path with constant conditional velocity."""

    event_shape: tuple[int, ...] = eqx.field(static=True)
    source_coordinate: Array
    target_coordinate: Array
    interpolant_id: str = eqx.field(static=True)

    def __init__(
        self,
        event_shape,
        /,
        *,
        source_coordinate: ArrayLike = 0.0,
        target_coordinate: ArrayLike = 1.0,
        interpolant_id: str | None = None,
    ):
        events = _event_shape(event_shape)
        source = jnp.asarray(source_coordinate, dtype=float).reshape(())
        target = jnp.asarray(target_coordinate, dtype=float).reshape(())
        if not bool(jnp.isfinite(source) & jnp.isfinite(target)):
            raise ValueError("Interpolant coordinates must be finite.")
        if not bool(target > source):
            raise ValueError(
                "Interpolant target_coordinate must exceed source_coordinate."
            )
        resolved_id = (
            canonical_fingerprint(
                {
                    "kind": "linear-endpoint-interpolant-v1",
                    "event_shape": list(events),
                    "source_coordinate": float(source),
                    "target_coordinate": float(target),
                }
            )
            if interpolant_id is None
            else str(interpolant_id)
        )
        if not resolved_id:
            raise ValueError("interpolant_id must be non-empty.")
        self.event_shape = events
        self.source_coordinate = source
        self.target_coordinate = target
        self.interpolant_id = resolved_id

    def evaluate(
        self,
        time: ArrayLike,
        source: ArrayLike,
        target: ArrayLike,
        /,
    ) -> EndpointInterpolantEvaluation:
        source_array = jnp.asarray(source)
        target_array = jnp.asarray(target, dtype=source_array.dtype)
        if source_array.shape != target_array.shape:
            raise ValueError("Interpolant source and target shapes must match.")
        leading = _leading_shape(source_array, self.event_shape)
        if not jnp.issubdtype(source_array.dtype, jnp.inexact):
            source_array = source_array.astype(float)
            target_array = target_array.astype(float)
        time_array = jnp.asarray(time, dtype=source_array.real.dtype)
        if time_array.shape == ():
            time_array = jnp.broadcast_to(time_array, leading)
        elif time_array.shape != leading:
            raise ValueError(
                f"Interpolant time must be scalar or have leading shape {leading}; "
                f"got {time_array.shape}."
            )
        time_array = eqx.error_if(
            time_array,
            jnp.any(
                (time_array < self.source_coordinate)
                | (time_array > self.target_coordinate)
            ),
            "Interpolant time lies outside its declared coordinate interval.",
        )
        duration = self.target_coordinate - self.source_coordinate
        weight = (time_array - self.source_coordinate) / duration
        expanded = weight.reshape(leading + (1,) * len(self.event_shape))
        state = (1.0 - expanded) * source_array + expanded * target_array
        velocity = (target_array - source_array) / duration
        axes = _event_axes(source_array, self.event_shape)
        finite = jnp.isfinite(time_array)
        endpoint_finite = jnp.isfinite(source_array) & jnp.isfinite(target_array)
        if axes:
            endpoint_finite = jnp.all(endpoint_finite, axis=axes)
        valid = finite & endpoint_finite
        return EndpointInterpolantEvaluation(
            time=time_array,
            state=state,
            conditional_velocity=velocity,
            valid=valid,
            event_shape=self.event_shape,
            interpolant_id=self.interpolant_id,
        )


__all__ = [
    "AbstractEndpointInterpolant",
    "EndpointInterpolantEvaluation",
    "LinearEndpointInterpolant",
]
