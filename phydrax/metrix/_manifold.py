#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule


def _point_shape(value: Sequence[int], /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("Manifold point_shape dimensions must be positive.")
    return shape


def _array_with_trailing_shape(
    value: ArrayLike,
    trailing_shape: tuple[int, ...],
    name: str,
    /,
) -> Array:
    array = jnp.asarray(value)
    rank = len(trailing_shape)
    if array.ndim < rank or (rank and array.shape[-rank:] != trailing_shape):
        raise ValueError(
            f"{name} must have trailing shape {trailing_shape}; got {array.shape}."
        )
    return array


def _same_shape(value: Array, reference: Array, name: str, /) -> None:
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} must preserve point shape {reference.shape}; got {value.shape}."
        )


def _real_inner(left: Array, right: Array, /) -> Array:
    return jnp.real(jnp.vdot(left, right))


def _finite_residual(value: Array, residual: Array, /) -> Array:
    infinity = jnp.asarray(jnp.inf, dtype=jnp.result_type(value.dtype, float))
    return jnp.where(jnp.all(jnp.isfinite(value)), residual, infinity)


class AbstractRiemannianManifold(StrictModule):
    """Metric and retraction contract for array-valued optimization parameters.

    ``point_shape`` describes the trailing dimensions of one manifold point. Any
    leading dimensions are interpreted as a product of independent points. Tangent
    vectors use the same ambient shape as their base points.
    """

    manifold_id: AbstractAttribute[str]
    point_shape: AbstractAttribute[tuple[int, ...]]
    retraction_method: AbstractAttribute[str]
    transport_method: AbstractAttribute[str]
    transport_is_isometric: AbstractAttribute[bool]
    transport_is_parallel: AbstractAttribute[bool]

    @abstractmethod
    def contains(self, point: ArrayLike, /) -> Array:
        """Return one scalar boolean for membership of the complete product point."""
        raise NotImplementedError

    @abstractmethod
    def constraint_residual(self, point: ArrayLike, /) -> Array:
        """Return one nonnegative scalar measuring the maximum membership defect."""
        raise NotImplementedError

    @abstractmethod
    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        """Project an ambient vector onto the tangent space at ``point``."""
        raise NotImplementedError

    @abstractmethod
    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        """Convert an ambient autodiff cotangent to the metric gradient."""
        raise NotImplementedError

    @abstractmethod
    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        """Return the product-manifold inner product of two tangents."""
        raise NotImplementedError

    def norm(self, point: ArrayLike, tangent: ArrayLike, /) -> Array:
        """Return the product-manifold norm of ``tangent``."""
        squared = jnp.asarray(self.inner(point, tangent, tangent))
        if squared.shape != ():
            raise ValueError("Manifold inner() must return a scalar array.")
        return jnp.sqrt(jnp.maximum(jnp.real(squared), 0.0))

    @abstractmethod
    def retract(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        /,
    ) -> Array:
        """Retract a tangent step at ``point`` onto the manifold."""
        raise NotImplementedError

    @abstractmethod
    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        """Transport a tangent from ``point`` to ``destination``."""
        raise NotImplementedError


class EuclideanManifold(AbstractRiemannianManifold):
    """Euclidean geometry for one or a product of unconstrained array points."""

    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, point_shape: Sequence[int] = (), /):
        self.point_shape = _point_shape(point_shape)
        shape_id = (
            "scalar" if not self.point_shape else "x".join(map(str, self.point_shape))
        )
        self.manifold_id = f"manifold:euclidean:{shape_id}"
        self.retraction_method = "addition"
        self.transport_method = "identity"
        self.transport_is_isometric = True
        self.transport_is_parallel = True

    def contains(self, point: ArrayLike, /) -> Array:
        value = _array_with_trailing_shape(point, self.point_shape, "Euclidean point")
        return jnp.all(jnp.isfinite(value))

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = _array_with_trailing_shape(point, self.point_shape, "Euclidean point")
        return jnp.where(
            jnp.all(jnp.isfinite(value)),
            jnp.asarray(0.0, dtype=jnp.result_type(value.dtype, float)),
            jnp.asarray(jnp.inf, dtype=jnp.result_type(value.dtype, float)),
        )

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        value = _array_with_trailing_shape(point, self.point_shape, "Euclidean point")
        vector = _array_with_trailing_shape(
            ambient_vector, self.point_shape, "Euclidean tangent"
        )
        _same_shape(vector, value, "Euclidean tangent")
        return vector

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(point, ambient_cotangent)

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        left = self.project_tangent(point, left_tangent)
        right = self.project_tangent(point, right_tangent)
        return _real_inner(left, right)

    def retract(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        /,
    ) -> Array:
        value = _array_with_trailing_shape(point, self.point_shape, "Euclidean point")
        step = self.project_tangent(value, tangent_step)
        return value + step

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        value = _array_with_trailing_shape(point, self.point_shape, "Euclidean point")
        self.project_tangent(value, tangent_step)
        target = _array_with_trailing_shape(
            destination, self.point_shape, "Euclidean destination"
        )
        _same_shape(target, value, "Euclidean destination")
        return self.project_tangent(target, tangent)


class SphereManifold(AbstractRiemannianManifold):
    """Unit sphere with the induced Euclidean metric and normalization retraction."""

    ambient_dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, ambient_dimension: int, /, *, tolerance: float = 1e-6):
        dimension = int(ambient_dimension)
        if dimension < 2:
            raise ValueError("Sphere ambient_dimension must be at least two.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Sphere tolerance must be finite and positive.")
        self.ambient_dimension = dimension
        self.tolerance = float(tolerance)
        self.manifold_id = f"manifold:sphere:{dimension}"
        self.point_shape = (dimension,)
        self.retraction_method = "normalization"
        self.transport_method = "tangent-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    def _point(self, point: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(point, self.point_shape, name)

    def contains(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Sphere point")
        norms = jnp.linalg.norm(value, axis=-1)
        return jnp.all(
            jnp.all(jnp.isfinite(value), axis=-1)
            & (jnp.abs(norms - 1.0) <= self.tolerance)
        )

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point, "Sphere point")
        defects = jnp.abs(jnp.linalg.norm(value, axis=-1) - 1.0)
        residual = jnp.max(defects, initial=0.0)
        return _finite_residual(value, residual)

    def project_tangent(
        self,
        point: ArrayLike,
        ambient_vector: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Sphere point")
        vector = self._point(ambient_vector, "Sphere tangent")
        _same_shape(vector, value, "Sphere tangent")
        coefficient = jnp.sum(jnp.conj(value) * vector, axis=-1, keepdims=True)
        return vector - value * coefficient

    def egrad_to_rgrad(
        self,
        point: ArrayLike,
        ambient_cotangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(point, ambient_cotangent)

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Sphere point")
        left = self._point(left_tangent, "Sphere left tangent")
        right = self._point(right_tangent, "Sphere right tangent")
        _same_shape(left, value, "Sphere left tangent")
        _same_shape(right, value, "Sphere right tangent")
        return _real_inner(left, right)

    def retract(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        /,
    ) -> Array:
        value = self._point(point, "Sphere point")
        step = self._point(tangent_step, "Sphere tangent step")
        _same_shape(step, value, "Sphere tangent step")
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
        value = self._point(point, "Sphere point")
        step = self._point(tangent_step, "Sphere tangent step")
        target = self._point(destination, "Sphere destination")
        vector = self._point(tangent, "Sphere transported tangent")
        _same_shape(step, value, "Sphere tangent step")
        _same_shape(target, value, "Sphere destination")
        _same_shape(vector, value, "Sphere transported tangent")
        return self.project_tangent(target, vector)


__all__ = [
    "AbstractRiemannianManifold",
    "EuclideanManifold",
    "SphereManifold",
]
