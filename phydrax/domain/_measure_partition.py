#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._strict import StrictModule


class GeometryMeasurePartition(StrictModule):
    """A finite simplex partition of a CAD geometry measure.

    ``vertices`` has shape ``(num_strata, simplex_size, spatial_dim)``. Segments
    represent boundary arclength; triangles represent planar area or surface area.
    ``measures`` stores the physical measure of every stratum.
    """

    vertices: Array
    measures: Array
    kind: Literal["segment", "triangle"]

    def __init__(
        self,
        vertices: Array,
        measures: Array,
        *,
        kind: Literal["segment", "triangle"],
    ):
        vertices_ = jnp.asarray(vertices, dtype=float)
        measures_ = jnp.asarray(measures, dtype=float).reshape((-1,))
        simplex_size = 2 if kind == "segment" else 3
        if vertices_.ndim != 3 or int(vertices_.shape[1]) != simplex_size:
            raise ValueError(
                f"{kind!r} partition vertices must have shape "
                f"(num_strata, {simplex_size}, spatial_dim)."
            )
        if int(vertices_.shape[0]) == 0:
            raise ValueError(
                "Geometry measure partitions must contain at least one stratum."
            )
        if measures_.shape != vertices_.shape[:1]:
            raise ValueError("Partition measures must have shape (num_strata,).")
        if bool(jnp.any(~jnp.isfinite(measures_))) or bool(jnp.any(measures_ <= 0.0)):
            raise ValueError("Partition measures must be finite and strictly positive.")
        self.vertices = vertices_
        self.measures = measures_
        self.kind = kind

    @property
    def num_strata(self) -> int:
        return int(self.measures.shape[0])

    @property
    def total_measure(self) -> Array:
        return jnp.sum(self.measures)

    def sample(
        self,
        num_points: int,
        /,
        *,
        key: Key[Array, ""],
        stratum_weights: Array | None = None,
        minimum_per_stratum: int = 0,
    ) -> tuple[Array, Array, Array]:
        """Sample a fixed-size stratified rule.

        Returns ``(points, stratum_index, base_mass)``. ``base_mass`` is the
        normalized physical measure represented by each sampled point and sums
        to one. Allocation uses largest remainders, so calls are fixed-shape and
        deterministic apart from point locations and final row order.
        """
        n = int(num_points)
        minimum = int(minimum_per_stratum)
        if n <= 0:
            raise ValueError("num_points must be positive.")
        if minimum < 0:
            raise ValueError("minimum_per_stratum must be non-negative.")
        if n < minimum * self.num_strata:
            raise ValueError(
                f"num_points={n} cannot allocate minimum_per_stratum={minimum} "
                f"across {self.num_strata} strata."
            )

        target_mass = self.measures / self.total_measure
        if stratum_weights is None:
            allocation_mass = target_mass
        else:
            weights = jnp.asarray(stratum_weights, dtype=float).reshape((-1,))
            if weights.shape != self.measures.shape:
                raise ValueError("stratum_weights must have shape (num_strata,).")
            weights = jnp.maximum(jnp.nan_to_num(weights, nan=0.0), 0.0)
            allocation_mass = target_mass * weights
            allocation_mass = jnp.where(
                jnp.sum(allocation_mass) > 0.0,
                allocation_mass / jnp.sum(allocation_mass),
                target_mass,
            )

        remaining = n - minimum * self.num_strata
        quotas = allocation_mass * remaining
        extras = jnp.floor(quotas).astype(jnp.int32)
        leftover = remaining - int(jnp.sum(extras))
        order = jnp.argsort(quotas - extras)[::-1]
        extras = extras.at[order[:leftover]].add(1)
        counts = extras + minimum
        strata = jnp.repeat(
            jnp.arange(self.num_strata, dtype=jnp.int32),
            counts,
            total_repeat_length=n,
        )

        vertices = self.vertices[strata]
        location_key, order_key = jr.split(key)
        if self.kind == "segment":
            coordinate = jr.uniform(location_key, (n, 1), dtype=vertices.dtype)
            points = vertices[:, 0] + coordinate * (vertices[:, 1] - vertices[:, 0])
        else:
            coordinate = jr.uniform(location_key, (n, 2), dtype=vertices.dtype)
            coordinate = jnp.where(
                jnp.sum(coordinate, axis=1, keepdims=True) > 1.0,
                1.0 - coordinate,
                coordinate,
            )
            points = (
                vertices[:, 0]
                + coordinate[:, :1] * (vertices[:, 1] - vertices[:, 0])
                + coordinate[:, 1:] * (vertices[:, 2] - vertices[:, 0])
            )

        represented = target_mass[strata] / counts[strata]
        represented = represented / jnp.sum(represented)
        permutation = jr.permutation(order_key, n)
        return points[permutation], strata[permutation], represented[permutation]


__all__ = ["GeometryMeasurePartition"]
