#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._manifold import (
    _array_with_trailing_shape,
    _finite_residual,
    _same_shape,
    AbstractRiemannianManifold,
)


def _positive_scalar(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _dimension(value: int, name: str, /, *, minimum: int = 1) -> int:
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


class PoincareBallManifold(AbstractRiemannianManifold):
    """Poincaré ball of constant sectional curvature ``-curvature``."""

    dimension: int = eqx.field(static=True)
    curvature: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        curvature: float = 1.0,
        tolerance: float = 1e-6,
    ):
        self.dimension = _dimension(dimension, "Poincare dimension")
        self.curvature = _positive_scalar(curvature, "curvature")
        self.tolerance = _positive_scalar(tolerance, "tolerance")
        self.manifold_id = f"manifold:poincare-ball:{self.dimension}:{self.curvature}"
        self.point_shape = (self.dimension,)
        self.retraction_method = "exponential"
        self.transport_method = "coordinate-identity"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    def _point(self, value: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(value, self.point_shape, name)

    def _radius_squared(self, point: Array, /) -> Array:
        return jnp.sum(point * point, axis=-1)

    def _conformal_factor(self, point: Array, /) -> Array:
        return 2.0 / (1.0 - self.curvature * self._radius_squared(point))

    def contains(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Poincare point")
        radius = self.curvature * self._radius_squared(value)
        return jnp.all(jnp.isfinite(value)) & jnp.all(radius < 1.0 - self.tolerance)

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Poincare point")
        excess = jnp.maximum(self.curvature * self._radius_squared(value) - 1.0, 0.0)
        return _finite_residual(value, jnp.max(excess, initial=0.0))

    def project_tangent(self, point: ArrayLike, ambient_vector: ArrayLike, /) -> Array:
        value = self._point(point, "Poincare point")
        vector = self._point(ambient_vector, "Poincare tangent")
        _same_shape(vector, value, "Poincare tangent")
        return vector

    def egrad_to_rgrad(self, point: ArrayLike, ambient_cotangent: ArrayLike, /) -> Array:
        value = self._point(point, "Poincare point")
        cotangent = self.project_tangent(value, ambient_cotangent)
        factor = self._conformal_factor(value)
        return cotangent / factor[..., None] ** 2

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Poincare point")
        left = self.project_tangent(value, left_tangent)
        right = self.project_tangent(value, right_tangent)
        factor = self._conformal_factor(value)
        return jnp.sum(factor**2 * jnp.sum(left * right, axis=-1))

    def _mobius_add(self, left: Array, right: Array, /) -> Array:
        curvature = self.curvature
        left_squared = jnp.sum(left * left, axis=-1, keepdims=True)
        right_squared = jnp.sum(right * right, axis=-1, keepdims=True)
        product = jnp.sum(left * right, axis=-1, keepdims=True)
        numerator = (
            1.0 + 2.0 * curvature * product + curvature * right_squared
        ) * left + (1.0 - curvature * left_squared) * right
        denominator = (
            1.0 + 2.0 * curvature * product + curvature**2 * left_squared * right_squared
        )
        return numerator / denominator

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        value = self._point(point, "Poincare point")
        step = self.project_tangent(value, tangent_step)
        norm = jnp.linalg.norm(step, axis=-1, keepdims=True)
        factor = self._conformal_factor(value)[..., None]
        root = jnp.sqrt(jnp.asarray(self.curvature, dtype=value.dtype))
        denominator = jnp.where(norm > 0.0, root * norm, 1.0)
        scaled = jnp.tanh(0.5 * root * factor * norm) * step / denominator
        scaled = jnp.where(norm > 0.0, scaled, jnp.zeros_like(step))
        return self._mobius_add(value, scaled)

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Poincare point")
        self.project_tangent(value, tangent_step)
        target = self._point(destination, "Poincare destination")
        vector = self._point(tangent, "Poincare transported tangent")
        _same_shape(target, value, "Poincare destination")
        _same_shape(vector, value, "Poincare transported tangent")
        return vector


class HyperboloidManifold(AbstractRiemannianManifold):
    """Future sheet of the Lorentz hyperboloid with curvature ``-curvature``."""

    dimension: int = eqx.field(static=True)
    curvature: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        curvature: float = 1.0,
        tolerance: float = 1e-6,
    ):
        self.dimension = _dimension(dimension, "Hyperboloid dimension")
        self.curvature = _positive_scalar(curvature, "curvature")
        self.tolerance = _positive_scalar(tolerance, "tolerance")
        self.manifold_id = f"manifold:hyperboloid:{self.dimension}:{self.curvature}"
        self.point_shape = (self.dimension + 1,)
        self.retraction_method = "exponential"
        self.transport_method = "tangent-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    def _point(self, value: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(value, self.point_shape, name)

    @staticmethod
    def _lorentz_inner(left: Array, right: Array, /) -> Array:
        return -left[..., 0] * right[..., 0] + jnp.sum(
            left[..., 1:] * right[..., 1:], axis=-1
        )

    def contains(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Hyperboloid point")
        residual = jnp.abs(self._lorentz_inner(value, value) + 1.0 / self.curvature)
        return (
            jnp.all(jnp.isfinite(value))
            & jnp.all(value[..., 0] > 0.0)
            & jnp.all(residual <= self.tolerance)
        )

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Hyperboloid point")
        sheet = jnp.maximum(-value[..., 0], 0.0)
        quadratic = jnp.abs(self._lorentz_inner(value, value) + 1.0 / self.curvature)
        residual = jnp.maximum(
            jnp.max(sheet, initial=0.0), jnp.max(quadratic, initial=0.0)
        )
        return _finite_residual(value, residual)

    def project_tangent(self, point: ArrayLike, ambient_vector: ArrayLike, /) -> Array:
        value = self._point(point, "Hyperboloid point")
        vector = self._point(ambient_vector, "Hyperboloid tangent")
        _same_shape(vector, value, "Hyperboloid tangent")
        coefficient = self.curvature * self._lorentz_inner(value, vector)
        return vector + coefficient[..., None] * value

    def egrad_to_rgrad(self, point: ArrayLike, ambient_cotangent: ArrayLike, /) -> Array:
        value = self._point(point, "Hyperboloid point")
        cotangent = self._point(ambient_cotangent, "Hyperboloid cotangent")
        _same_shape(cotangent, value, "Hyperboloid cotangent")
        raised = cotangent.at[..., 0].multiply(-1.0)
        return self.project_tangent(value, raised)

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Hyperboloid point")
        left = self.project_tangent(value, left_tangent)
        right = self.project_tangent(value, right_tangent)
        return jnp.sum(self._lorentz_inner(left, right))

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        value = self._point(point, "Hyperboloid point")
        step = self.project_tangent(value, tangent_step)
        norm = jnp.sqrt(jnp.maximum(self._lorentz_inner(step, step), 0.0))
        root = jnp.sqrt(jnp.asarray(self.curvature, dtype=value.dtype))
        argument = root * norm
        coefficient = jnp.where(
            norm > 0.0,
            jnp.sinh(argument) / (root * norm),
            jnp.asarray(1.0, dtype=value.dtype),
        )
        return jnp.cosh(argument)[..., None] * value + coefficient[..., None] * step

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Hyperboloid point")
        self.project_tangent(value, tangent_step)
        target = self._point(destination, "Hyperboloid destination")
        vector = self._point(tangent, "Hyperboloid transported tangent")
        _same_shape(target, value, "Hyperboloid destination")
        _same_shape(vector, value, "Hyperboloid transported tangent")
        return self.project_tangent(target, vector)


class ProbabilitySimplexManifold(AbstractRiemannianManifold):
    """Open probability simplex with the Fisher–Rao metric."""

    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, tolerance: float = 1e-7):
        self.dimension = _dimension(dimension, "Simplex dimension", minimum=2)
        self.tolerance = _positive_scalar(tolerance, "tolerance")
        self.manifold_id = f"manifold:probability-simplex:{self.dimension}"
        self.point_shape = (self.dimension,)
        self.retraction_method = "multiplicative"
        self.transport_method = "tangent-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    def _point(self, value: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(value, self.point_shape, name)

    def contains(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Simplex point")
        sums = jnp.sum(value, axis=-1)
        return (
            jnp.all(jnp.isfinite(value))
            & jnp.all(value > 0.0)
            & jnp.all(jnp.abs(sums - 1.0) <= self.tolerance)
        )

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Simplex point")
        normalization = jnp.abs(jnp.sum(value, axis=-1) - 1.0)
        negativity = jnp.maximum(-value, 0.0)
        residual = jnp.maximum(
            jnp.max(normalization, initial=0.0),
            jnp.max(negativity, initial=0.0),
        )
        return _finite_residual(value, residual)

    def project_tangent(self, point: ArrayLike, ambient_vector: ArrayLike, /) -> Array:
        value = self._point(point, "Simplex point")
        vector = self._point(ambient_vector, "Simplex tangent")
        _same_shape(vector, value, "Simplex tangent")
        return vector - jnp.mean(vector, axis=-1, keepdims=True)

    def egrad_to_rgrad(self, point: ArrayLike, ambient_cotangent: ArrayLike, /) -> Array:
        value = self._point(point, "Simplex point")
        cotangent = self._point(ambient_cotangent, "Simplex cotangent")
        _same_shape(cotangent, value, "Simplex cotangent")
        expectation = jnp.sum(value * cotangent, axis=-1, keepdims=True)
        return value * (cotangent - expectation)

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Simplex point")
        left = self.project_tangent(value, left_tangent)
        right = self.project_tangent(value, right_tangent)
        return jnp.sum(left * right / value)

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        value = self._point(point, "Simplex point")
        step = self.project_tangent(value, tangent_step)
        logits = jnp.log(value) + step / value
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        candidate = jnp.exp(logits)
        return candidate / jnp.sum(candidate, axis=-1, keepdims=True)

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Simplex point")
        self.project_tangent(value, tangent_step)
        target = self._point(destination, "Simplex destination")
        vector = self._point(tangent, "Simplex transported tangent")
        _same_shape(target, value, "Simplex destination")
        _same_shape(vector, value, "Simplex transported tangent")
        return self.project_tangent(target, vector)


__all__ = [
    "HyperboloidManifold",
    "PoincareBallManifold",
    "ProbabilitySimplexManifold",
]
