#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._base import _as_point, _pairwise_matrix, AbstractUnitDiagonalKernel


class AbstractStationaryKernel(AbstractUnitDiagonalKernel):
    """Unit-diagonal stationary kernel with scalar or coordinatewise scale."""

    length_scale: Array

    def __init__(self, *, length_scale: ArrayLike = 1.0):
        scale = jnp.asarray(length_scale, dtype=float)
        if scale.ndim > 1 or (scale.ndim == 1 and scale.shape[0] == 0):
            raise ValueError("length_scale must be scalar or a nonempty vector.")
        self.length_scale = eqx.error_if(
            scale,
            jnp.any(~jnp.isfinite(scale)) | jnp.any(scale <= 0.0),
            "length_scale must be finite and strictly positive.",
        )

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_point = _as_point(left, name="left")
        right_point = _as_point(right, name="right")
        if left_point.shape != right_point.shape:
            raise ValueError("Kernel points must have equal coordinate shape.")
        if self.length_scale.ndim == 1 and self.length_scale.shape[0] not in (
            1,
            left_point.shape[0],
        ):
            raise ValueError(
                "Vector length_scale must have size one or match coordinate size."
            )
        delta = (left_point - right_point) / self.length_scale
        return self._from_squared_distance(jnp.sum(delta * delta))

    @property
    def kernel_id(self) -> str:
        return type(self).__name__

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return _pairwise_matrix(self, left, right)

    @abstractmethod
    def _from_squared_distance(self, squared_distance: Array, /) -> Array:
        raise NotImplementedError


class SquaredExponentialKernel(AbstractStationaryKernel):
    """Squared-exponential correlation kernel."""

    def _from_squared_distance(self, squared_distance: Array, /) -> Array:
        return jnp.exp(-0.5 * squared_distance)

    @property
    def max_derivative_order(self) -> int | None:
        return None


class Matern32Kernel(AbstractStationaryKernel):
    """Matérn-3/2 correlation kernel."""

    def _from_squared_distance(self, squared_distance: Array, /) -> Array:
        return _matern32_from_squared_distance(squared_distance)

    @property
    def max_derivative_order(self) -> int | None:
        return 1


class Matern52Kernel(AbstractStationaryKernel):
    """Matérn-5/2 correlation kernel."""

    def _from_squared_distance(self, squared_distance: Array, /) -> Array:
        return _matern52_from_squared_distance(squared_distance)

    @property
    def max_derivative_order(self) -> int | None:
        return 2


class InverseMultiquadricKernel(AbstractStationaryKernel):
    """Inverse-multiquadric correlation kernel."""

    def _from_squared_distance(self, squared_distance: Array, /) -> Array:
        return jax_lax_rsqrt(1.0 + squared_distance)

    @property
    def max_derivative_order(self) -> int | None:
        return None


def _safe_distance(squared_distance: Array, /) -> Array:
    positive = squared_distance > 0.0
    safe_squared = jnp.where(positive, squared_distance, 1.0)
    return jnp.where(positive, jnp.sqrt(safe_squared), 0.0)


@jax.custom_jvp
def _matern32_from_squared_distance(squared_distance: Array, /) -> Array:
    scaled = jnp.sqrt(3.0) * _safe_distance(squared_distance)
    return (1.0 + scaled) * jnp.exp(-scaled)


@_matern32_from_squared_distance.defjvp
def _matern32_from_squared_distance_jvp(primals, tangents):
    (squared_distance,) = primals
    (squared_distance_tangent,) = tangents
    scaled = jnp.sqrt(3.0) * _safe_distance(squared_distance)
    value = _matern32_from_squared_distance(squared_distance)
    derivative = -1.5 * jnp.exp(-scaled)
    return value, derivative * squared_distance_tangent


@jax.custom_jvp
def _matern52_from_squared_distance(squared_distance: Array, /) -> Array:
    scaled = jnp.sqrt(5.0) * _safe_distance(squared_distance)
    return (1.0 + scaled + scaled * scaled / 3.0) * jnp.exp(-scaled)


@_matern52_from_squared_distance.defjvp
def _matern52_from_squared_distance_jvp(primals, tangents):
    (squared_distance,) = primals
    (squared_distance_tangent,) = tangents
    value = _matern52_from_squared_distance(squared_distance)
    derivative = _matern52_first_squared_distance_derivative(squared_distance)
    return value, derivative * squared_distance_tangent


@jax.custom_jvp
def _matern52_first_squared_distance_derivative(
    squared_distance: Array,
    /,
) -> Array:
    scaled = jnp.sqrt(5.0) * _safe_distance(squared_distance)
    return -(5.0 / 6.0) * (1.0 + scaled) * jnp.exp(-scaled)


@_matern52_first_squared_distance_derivative.defjvp
def _matern52_first_squared_distance_derivative_jvp(primals, tangents):
    (squared_distance,) = primals
    (squared_distance_tangent,) = tangents
    scaled = jnp.sqrt(5.0) * _safe_distance(squared_distance)
    value = _matern52_first_squared_distance_derivative(squared_distance)
    derivative = (25.0 / 12.0) * jnp.exp(-scaled)
    return value, derivative * squared_distance_tangent


def jax_lax_rsqrt(value: Array, /) -> Array:
    return jnp.reciprocal(jnp.sqrt(value))


__all__ = [
    "AbstractStationaryKernel",
    "InverseMultiquadricKernel",
    "Matern32Kernel",
    "Matern52Kernel",
    "SquaredExponentialKernel",
]
