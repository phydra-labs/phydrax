#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from .._strict import StrictModule
from ._atlas import BoundaryAtlas


class GeometryMeasurePartition(StrictModule):
    """A finite simplex partition of a geometric measure.

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


class BoundaryAtlasPartition(StrictModule):
    """Physical-measure strata induced by a representation-independent atlas."""

    atlas: BoundaryAtlas
    measures: Array
    seed_reference: Array
    candidate_count: int

    def __init__(
        self,
        atlas: BoundaryAtlas,
        *,
        quadrature_order: int = 12,
        candidate_count: int = 64,
    ):
        if not isinstance(atlas, BoundaryAtlas):
            raise TypeError("atlas must be a BoundaryAtlas.")
        if quadrature_order < 2 or candidate_count < 2:
            raise ValueError("quadrature_order and candidate_count must be at least two.")
        nodes_host, weights_host = np.polynomial.legendre.leggauss(quadrature_order)
        nodes_host = 0.5 * (nodes_host + 1.0)
        weights_host = 0.5 * weights_host
        if atlas.reference_dimension == 1:
            reference_host = nodes_host[:, None]
            reference_weights_host = weights_host
        elif atlas.reference_dimension == 2:
            first, second = np.meshgrid(nodes_host, nodes_host, indexing="ij")
            first_weight, second_weight = np.meshgrid(
                weights_host, weights_host, indexing="ij"
            )
            reference_host = np.stack((first.ravel(), second.ravel()), axis=-1)
            reference_weights_host = (first_weight * second_weight).ravel()
        else:
            raise ValueError(
                "BoundaryAtlasPartition supports reference dimensions one and two."
            )
        reference = jnp.asarray(reference_host, dtype=float)
        reference_weights = jnp.asarray(reference_weights_host, dtype=float)
        chart_indices = jnp.broadcast_to(
            jnp.arange(atlas.num_charts, dtype=jnp.int32)[:, None],
            (atlas.num_charts, reference.shape[0]),
        )
        chart_reference = jnp.broadcast_to(
            reference[None, ...],
            (atlas.num_charts, *reference.shape),
        )
        density = atlas.jacobian(chart_indices, chart_reference)
        active = atlas.reference_mask(chart_indices, chart_reference)
        density = jnp.where(
            active & atlas.seam_owner[:, None],
            density,
            0.0,
        )
        measures = jnp.sum(density * reference_weights[None, :], axis=1)
        measures_host = np.asarray(measures)
        if np.any(~np.isfinite(measures_host)) or np.any(measures_host <= 0.0):
            raise ValueError(
                "Every selected atlas chart must have positive finite measure."
            )
        seed_indices = np.asarray(jnp.argmax(density, axis=1), dtype=np.int32)
        self.atlas = atlas
        self.measures = measures
        self.seed_reference = reference[jnp.asarray(seed_indices)]
        self.candidate_count = int(candidate_count)

    @property
    def num_strata(self) -> int:
        return self.atlas.num_charts

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
        n = int(num_points)
        minimum = int(minimum_per_stratum)
        if n <= 0:
            raise ValueError("num_points must be positive.")
        if minimum < 0 or n < minimum * self.num_strata:
            raise ValueError("Requested points cannot satisfy minimum_per_stratum.")
        target_mass = self.measures / self.total_measure
        if stratum_weights is None:
            allocation_mass = target_mass
        else:
            weights = jnp.asarray(stratum_weights, dtype=float).reshape((-1,))
            if weights.shape != self.measures.shape:
                raise ValueError("stratum_weights must have shape (num_strata,).")
            weighted = target_mass * jnp.maximum(jnp.nan_to_num(weights, nan=0.0), 0.0)
            allocation_mass = jnp.where(
                jnp.sum(weighted) > 0.0,
                weighted / jnp.sum(weighted),
                target_mass,
            )
        remaining = n - minimum * self.num_strata
        quotas = allocation_mass * remaining
        extras = jnp.floor(quotas).astype(jnp.int32)
        leftover = remaining - int(jnp.sum(extras))
        order = jnp.argsort(quotas - extras)[::-1]
        counts = extras.at[order[:leftover]].add(1) + minimum
        strata = jnp.repeat(
            jnp.arange(self.num_strata, dtype=jnp.int32),
            counts,
            total_repeat_length=n,
        )
        reference_key, selection_key, permutation_key = jr.split(key, 3)
        random_reference = jr.uniform(
            reference_key,
            (
                n,
                self.candidate_count - 1,
                self.atlas.reference_dimension,
            ),
            dtype=self.seed_reference.dtype,
        )
        candidates = jnp.concatenate(
            (self.seed_reference[strata, None, :], random_reference),
            axis=1,
        )
        candidate_charts = jnp.broadcast_to(strata[:, None], candidates.shape[:2])
        density = self.atlas.jacobian(candidate_charts, candidates)
        density = jnp.where(
            self.atlas.reference_mask(candidate_charts, candidates),
            density,
            0.0,
        )
        selection = jr.categorical(
            selection_key,
            jnp.log(jnp.maximum(density, jnp.finfo(density.dtype).tiny)),
            axis=1,
        )
        selected_reference = candidates[jnp.arange(n), selection]
        points = self.atlas.map(strata, selected_reference)
        represented = target_mass[strata] / counts[strata]
        represented = represented / jnp.sum(represented)
        permutation = jr.permutation(permutation_key, n)
        return points[permutation], strata[permutation], represented[permutation]


__all__ = ["BoundaryAtlasPartition", "GeometryMeasurePartition"]
