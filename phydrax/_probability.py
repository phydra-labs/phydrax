#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import prod

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from ._strict import StrictModule
from .domain._measure import MeasureKind


class AbstractProbabilityLaw(StrictModule):
    """Probability law with explicit sample, batch, event, and measure semantics."""

    @property
    @abstractmethod
    def event_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def batch_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def density_measure_kind(self) -> MeasureKind:
        raise NotImplementedError

    @abstractmethod
    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def contains(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError


def _positive_shape(value, /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"{owner} must contain positive dimensions.")
    return shape


class DiagonalNormalLaw(AbstractProbabilityLaw):
    """Full-rank diagonal Normal law over one explicit array event."""

    location: Array
    scale: Array
    _event_shape: tuple[int, ...] = eqx.field(static=True)
    _batch_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        location: ArrayLike,
        scale: ArrayLike,
        /,
        *,
        event_shape,
    ):
        events = _positive_shape(event_shape, owner="event_shape")
        raw_location = jnp.asarray(location)
        raw_scale = jnp.asarray(scale)
        if jnp.iscomplexobj(raw_location) or jnp.iscomplexobj(raw_scale):
            raise TypeError("Diagonal Normal parameters must be real-valued.")
        dtype = jnp.result_type(raw_location, raw_scale, float)
        location_array = raw_location.astype(dtype)
        if (
            location_array.ndim < len(events)
            or tuple(location_array.shape[-len(events) :]) != events
        ):
            raise ValueError(
                f"location must end in event_shape={events}; got {location_array.shape}."
            )
        scale_array = jnp.broadcast_to(raw_scale.astype(dtype), location_array.shape)
        location_array = eqx.error_if(
            location_array,
            jnp.any(~jnp.isfinite(location_array)),
            "Diagonal Normal location must be finite.",
        )
        scale_array = eqx.error_if(
            scale_array,
            jnp.any(~jnp.isfinite(scale_array) | (scale_array <= 0.0)),
            "Diagonal Normal scale must be finite and strictly positive.",
        )
        self.location = location_array
        self.scale = scale_array
        self._event_shape = events
        self._batch_shape = tuple(location_array.shape[: -len(events)])

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def density_measure_kind(self) -> MeasureKind:
        return "lebesgue"

    @property
    def mean(self) -> Array:
        return self.location

    @property
    def variance(self) -> Array:
        return self.scale**2

    @property
    def event_size(self) -> int:
        return prod(self.event_shape)

    def _value(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value, dtype=self.location.dtype)
        trailing = self.batch_shape + self.event_shape
        if array.ndim < len(trailing) or tuple(array.shape[-len(trailing) :]) != trailing:
            raise ValueError(
                f"value must end in batch_shape + event_shape {trailing}; got {array.shape}."
            )
        return array

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("sample_shape dimensions must be positive.")
        noise = jr.normal(
            key,
            samples + self.batch_shape + self.event_shape,
            dtype=self.location.dtype,
        )
        return self.location + self.scale * noise

    def log_prob(self, value: ArrayLike, /) -> Array:
        array = self._value(value)
        standardized = (array - self.location) / self.scale
        elementwise = (
            -0.5 * standardized**2
            - jnp.log(self.scale)
            - 0.5 * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=array.dtype))
        )
        axes = tuple(range(elementwise.ndim - len(self.event_shape), elementwise.ndim))
        return jnp.sum(elementwise, axis=axes)

    def score(self, value: ArrayLike, /) -> Array:
        array = self._value(value)
        return -(array - self.location) / self.variance

    def contains(self, value: ArrayLike, /) -> Array:
        array = self._value(value)
        axes = tuple(range(array.ndim - len(self.event_shape), array.ndim))
        return jnp.all(jnp.isfinite(array), axis=axes)


__all__ = ["AbstractProbabilityLaw", "DiagonalNormalLaw"]
