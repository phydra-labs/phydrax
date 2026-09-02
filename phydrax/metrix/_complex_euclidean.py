#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._manifold import AbstractGeodesicManifold


class ComplexEuclideanManifold(AbstractGeodesicManifold):
    """Unconstrained complex leaves with the real Hermitian metric.

    Ambient cotangents use JAX's convention for real-valued objectives and are
    consumed exactly once by ``egrad_to_rgrad`` without caller conjugation.
    """

    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, point_shape: Sequence[int], /):
        shape = tuple(int(size) for size in point_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError(
                "ComplexEuclideanManifold point_shape must be nonempty and positive."
            )
        self.point_shape = shape
        self.manifold_id = f"manifold:complex-euclidean:{'x'.join(map(str, shape))}"
        self.retraction_method = "addition"
        self.transport_method = "identity"
        self.transport_is_isometric = True
        self.transport_is_parallel = True

    @property
    def scalar_field(self) -> str:
        return "complex"

    def _point(self, point: ArrayLike, name: str, /) -> Array:
        value = jnp.asarray(point)
        if value.shape[-len(self.point_shape) :] != self.point_shape:
            raise ValueError(f"{name} must end in shape {self.point_shape}.")
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError(f"{name} must use complex floating-point coordinates.")
        return value

    def contains(self, point: ArrayLike, /) -> Array:
        return jnp.all(jnp.isfinite(self._point(point, "Complex Euclidean point")))

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Complex Euclidean point")
        return jnp.where(
            jnp.all(jnp.isfinite(value)),
            jnp.asarray(0.0, dtype=value.real.dtype),
            jnp.asarray(jnp.inf, dtype=value.real.dtype),
        )

    def project_tangent(self, point: ArrayLike, ambient_vector: ArrayLike, /) -> Array:
        value = self._point(point, "Complex Euclidean point")
        tangent = self._point(ambient_vector, "Complex Euclidean tangent")
        if tangent.shape != value.shape:
            raise ValueError(
                "Complex Euclidean tangent must match the complete point shape."
            )
        return tangent

    def egrad_to_rgrad(self, point: ArrayLike, ambient_cotangent: ArrayLike, /) -> Array:
        return self.project_tangent(point, jnp.conj(ambient_cotangent))

    def inner(
        self, point: ArrayLike, left_tangent: ArrayLike, right_tangent: ArrayLike, /
    ) -> Array:
        left = self.project_tangent(point, left_tangent)
        right = self.project_tangent(point, right_tangent)
        return jnp.real(jnp.vdot(left, right))

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        value = self._point(point, "Complex Euclidean point")
        return value + self.project_tangent(value, tangent_step)

    exp = retract

    def log(self, point: ArrayLike, destination: ArrayLike, /) -> Array:
        value = self._point(point, "Complex Euclidean point")
        target = self._point(destination, "Complex Euclidean destination")
        if target.shape != value.shape:
            raise ValueError("Complex Euclidean destination must match point shape.")
        return target - value

    def squared_distance(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        difference = self.log(left, right)
        return jnp.real(jnp.vdot(difference, difference))

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        del point, tangent_step
        return self.project_tangent(destination, tangent)


__all__ = ["ComplexEuclideanManifold"]
