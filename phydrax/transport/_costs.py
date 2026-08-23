#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import AbstractGeodesicManifold


class AbstractGroundCost(StrictModule):
    """Real nonnegative ground cost between finite coordinate vectors."""

    @abstractmethod
    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Evaluate the cost between two individual points."""
        raise NotImplementedError

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Evaluate the complete pairwise cost matrix."""
        left_points = jnp.asarray(left)
        right_points = jnp.asarray(right, dtype=left_points.dtype)
        if left_points.ndim < 2 or right_points.ndim < 2:
            raise ValueError("Ground-cost designs require a leading sample axis.")
        if left_points.shape[1:] != right_points.shape[1:]:
            raise ValueError("Ground-cost point designs must share point shape.")
        return jax.vmap(
            lambda point: jax.vmap(lambda other: self.pairwise(point, other))(
                right_points
            )
        )(left_points)

    @property
    @abstractmethod
    def cost_id(self) -> str:
        """Return stable diagnostic identity for the cost."""
        raise NotImplementedError


class SquaredEuclideanCost(AbstractGroundCost):
    """Squared Euclidean ground cost."""

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_point, right_point = _point_pair(left, right)
        difference = left_point - right_point
        return jnp.real(jnp.vdot(difference, difference))

    @property
    def cost_id(self) -> str:
        return "squared-euclidean"


class WeightedSquaredEuclideanCost(AbstractGroundCost):
    """Squared Euclidean cost in explicit positive component scales."""

    scales: Array

    def __init__(self, scales: ArrayLike, /):
        values = jnp.asarray(scales, dtype=float)
        if values.ndim != 1 or values.shape[0] == 0:
            raise ValueError("scales must be a nonempty rank-one array.")
        self.scales = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)) | jnp.any(values <= 0.0),
            "scales must contain only finite positive values.",
        )

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_point, right_point = _point_pair(left, right)
        if left_point.shape != self.scales.shape:
            raise ValueError(
                "Point feature size must match WeightedSquaredEuclideanCost scales."
            )
        difference = (left_point - right_point) / self.scales
        return jnp.real(jnp.vdot(difference, difference))

    @property
    def cost_id(self) -> str:
        return "weighted-squared-euclidean"


class PeriodicSquaredEuclideanCost(AbstractGroundCost):
    """Squared shortest-displacement cost on explicitly periodic coordinates."""

    periods: Array

    def __init__(self, periods: ArrayLike, /):
        values = jnp.asarray(periods, dtype=float)
        if values.ndim != 1 or values.shape[0] == 0:
            raise ValueError("periods must be a nonempty rank-one array.")
        self.periods = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)) | jnp.any(values <= 0.0),
            "periods must contain only finite positive values.",
        )

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_point, right_point = _point_pair(left, right)
        if left_point.shape != self.periods.shape:
            raise ValueError(
                "Point feature size must match PeriodicSquaredEuclideanCost periods."
            )
        displacement = jnp.mod(jnp.abs(left_point - right_point), self.periods)
        shortest = jnp.minimum(displacement, self.periods - displacement)
        return jnp.real(jnp.vdot(shortest, shortest))

    @property
    def cost_id(self) -> str:
        return "periodic-squared-euclidean"


class IntrinsicSquaredDistanceCost(AbstractGroundCost):
    """Squared geodesic distance supplied by an intrinsic manifold."""

    geometry: AbstractGeodesicManifold

    def __init__(self, geometry: AbstractGeodesicManifold, /):
        if not isinstance(geometry, AbstractGeodesicManifold):
            raise TypeError("geometry must be an AbstractGeodesicManifold.")
        self.geometry = geometry

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_point = jnp.asarray(left)
        right_point = jnp.asarray(right, dtype=left_point.dtype)
        expected = self.geometry.point_shape
        if left_point.shape != expected or right_point.shape != expected:
            raise ValueError(f"Intrinsic ground-cost points must have shape {expected}.")
        return jnp.asarray(
            self.geometry.squared_distance(left_point, right_point),
            dtype=float,
        )

    @property
    def cost_id(self) -> str:
        return f"intrinsic-squared-distance:{self.geometry.manifold_id}"


class PrecomputedCost(StrictModule):
    """Validated finite nonnegative precomputed ground-cost matrix."""

    values: Array
    cost_id: str = eqx.field(static=True)

    def __init__(self, values: ArrayLike, /, *, cost_id: str = "precomputed"):
        matrix = jnp.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Precomputed costs must be a nonempty rank-two matrix.")
        identifier = str(cost_id)
        if not identifier:
            raise ValueError("cost_id must be nonempty.")
        self.values = eqx.error_if(
            matrix,
            jnp.any(~jnp.isfinite(matrix)) | jnp.any(matrix < 0.0),
            "Precomputed costs must contain only finite nonnegative values.",
        )
        self.cost_id = identifier

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.values.shape[0]), int(self.values.shape[1])

    def block(
        self,
        source_start: int | Array,
        target_start: int | Array,
        source_size: int,
        target_size: int,
        /,
    ) -> Array:
        """Return a fixed-shape cost block."""
        return jax.lax.dynamic_slice(
            self.values,
            (source_start, target_start),
            (source_size, target_size),
        )


GroundCost = AbstractGroundCost | PrecomputedCost


def _point_pair(left: ArrayLike, right: ArrayLike, /) -> tuple[Array, Array]:
    left_point = _as_point(left, name="left")
    right_point = _as_point(right, name="right")
    if left_point.shape != right_point.shape:
        raise ValueError("Ground-cost points must have equal shape.")
    return left_point, right_point


def _as_point(value: ArrayLike, /, *, name: str) -> Array:
    point = jnp.asarray(value, dtype=float)
    if point.ndim == 0:
        point = point.reshape((1,))
    if point.ndim != 1 or point.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty coordinate vector.")
    return eqx.error_if(
        point,
        jnp.any(~jnp.isfinite(point)),
        f"{name} must contain only finite coordinates.",
    )


def _as_points(value: ArrayLike, /, *, name: str) -> Array:
    points = jnp.asarray(value, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(f"{name} must have shape (point, feature) with nonempty axes.")
    return eqx.error_if(
        points,
        jnp.any(~jnp.isfinite(points)),
        f"{name} must contain only finite coordinates.",
    )


__all__ = [
    "AbstractGroundCost",
    "GroundCost",
    "IntrinsicSquaredDistanceCost",
    "PeriodicSquaredEuclideanCost",
    "PrecomputedCost",
    "SquaredEuclideanCost",
    "WeightedSquaredEuclideanCost",
]
