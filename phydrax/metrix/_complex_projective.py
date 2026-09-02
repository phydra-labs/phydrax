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
    AbstractGeodesicManifold,
)


class ComplexProjectiveManifold(AbstractGeodesicManifold):
    """Complex projective space in normalized homogeneous coordinates."""

    ambient_dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, ambient_dimension: int, /, *, tolerance: float = 1e-7):
        dimension = int(ambient_dimension)
        if dimension < 2:
            raise ValueError("Complex projective ambient dimension must be at least two.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Complex projective tolerance must be finite and positive.")
        self.ambient_dimension = dimension
        self.tolerance = float(tolerance)
        self.manifold_id = f"manifold:complex-projective:{dimension - 1}"
        self.point_shape = (dimension,)
        self.retraction_method = "homogeneous-normalization"
        self.transport_method = "horizontal-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    @property
    def scalar_field(self) -> str:
        return "complex"

    def _point(self, point: ArrayLike, name: str, /) -> Array:
        value = _array_with_trailing_shape(point, self.point_shape, name)
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError(f"{name} must use complex floating-point coordinates.")
        return value

    def contains(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Complex-projective point")
        norms = jnp.linalg.norm(value, axis=-1)
        return jnp.all(jnp.isfinite(value)) & jnp.all(
            jnp.abs(norms - 1.0) <= self.tolerance
        )

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Complex-projective point")
        residual = jnp.max(
            jnp.abs(jnp.linalg.norm(value, axis=-1) - 1.0),
            initial=0.0,
        )
        return _finite_residual(value, residual)

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Complex-projective point")
        vector = self._point(ambient_vector, "Complex-projective tangent")
        _same_shape(vector, value, "Complex-projective tangent")
        coefficient = jnp.sum(jnp.conj(value) * vector, axis=-1, keepdims=True)
        return vector - value * coefficient

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(point, jnp.conj(ambient_cotangent))

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        left = self.project_tangent(point, left_tangent)
        right = self.project_tangent(point, right_tangent)
        return jnp.real(jnp.vdot(left, right))

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        value = self._point(point, "Complex-projective point")
        step = self.project_tangent(value, tangent_step)
        candidate = value + step
        norm = jnp.linalg.norm(candidate, axis=-1, keepdims=True)
        return candidate / norm

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Complex-projective point")
        self.project_tangent(value, tangent_step)
        target = self._point(destination, "Complex-projective destination")
        vector = self._point(tangent, "Complex-projective transported tangent")
        _same_shape(target, value, "Complex-projective destination")
        _same_shape(vector, value, "Complex-projective transported tangent")
        return self.project_tangent(target, vector)

    def exp(self, point: ArrayLike, tangent: ArrayLike, /) -> Array:
        value = self._point(point, "Complex-projective point")
        step = self.project_tangent(value, tangent)
        norm = jnp.linalg.norm(step, axis=-1, keepdims=True)
        safe_norm = jnp.where(norm > 0.0, norm, 1.0)
        coefficient = jnp.where(norm > 0.0, jnp.sin(norm) / safe_norm, 1.0)
        return jnp.cos(norm) * value + coefficient * step

    def log(self, point: ArrayLike, destination: ArrayLike, /) -> Array:
        value = self._point(point, "Complex-projective point")
        target = self._point(destination, "Complex-projective destination")
        _same_shape(target, value, "Complex-projective destination")
        overlap = jnp.sum(jnp.conj(value) * target, axis=-1, keepdims=True)
        magnitude = jnp.abs(overlap)
        target = eqx.error_if(
            target,
            jnp.any(magnitude <= self.tolerance),
            "Complex-projective logarithm is nonunique at orthogonal rays.",
        )
        phase = overlap / magnitude
        aligned = target * jnp.conj(phase)
        cosine = jnp.clip(magnitude, 0.0, 1.0)
        angle = jnp.arccos(cosine)
        tangent = aligned - cosine * value
        norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
        factor = jnp.where(norm > self.tolerance, angle / norm, 1.0)
        return factor * tangent

    def squared_distance(
        self,
        left: ArrayLike,
        right: ArrayLike,
        /,
    ) -> Array:
        value = self._point(left, "Complex-projective point")
        target = self._point(right, "Complex-projective destination")
        _same_shape(target, value, "Complex-projective destination")
        overlap = jnp.abs(jnp.sum(jnp.conj(value) * target, axis=-1))
        return jnp.sum(jnp.arccos(jnp.clip(overlap, 0.0, 1.0)) ** 2)


__all__ = ["ComplexProjectiveManifold"]
