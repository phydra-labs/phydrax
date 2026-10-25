#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..spatial._morton import canonical_morton_order, morton_encode_integer
from ._image_neighborhood import (
    _AbstractCandidateImageRouteSearch,
    _Candidates,
    _host_reach,
    _INT32_LIMIT,
    _LatticeFrame,
    _ResolvedReceivers,
    _within_radius,
    CellListImageRouteSearch,
    ParticleImageCapacity,
)


_MORTON_CODE_BITS = 30


class ClusterTileImageRouteSearch(_AbstractCandidateImageRouteSearch):
    """Morton cluster-tile image search with exact pair queries inside tiles.

    Usable particles are wrapped into the lattice frame and put in the
    canonical Morton order of their quantized fractional coordinates (stable
    IDs break ties, inactive particles last).  Consecutive runs of
    ``cluster_width`` particles form clusters with fractional centroid
    ``c_g`` (kept inside the cell) and bounding radius
    ``rho_g = max_i |x_i - c_g|``.

    Candidate tiles are the directed image routes ``(a, b, n)`` between
    cluster centroids found by a prepared cluster-level cell-list image search
    of radius ``radius + 2 * maximum_cluster_radius``, plus every cluster's own
    zero-shift tile.  A tile survives when
    ``|c_b - c_a + n H| < radius + rho_a + rho_b``, and every particle pair
    ``(j in a, i in b)`` of a surviving tile is tested exactly.  Any route
    within ``radius`` passes the sphere test, and its centroid separation is
    below ``radius + 2 * maximum_cluster_radius`` whenever every cluster
    radius is bounded by ``maximum_cluster_radius``; a larger cluster is a
    ``cluster_extent_overflow`` (prepared-envelope failure, not repaired by a
    capacity retry).

    ``tile_capacity`` charges the intermediate tile search (cluster cell
    occupancy, per-case tile pairs through its required
    ``maximum_case_edges``, tile degree and images) independently of the final
    route ``capacity``: ``tile_overflow`` is reported separately from, and
    possibly before, edge overflow.  Exact work is
    ``case * (tile pairs + clusters) * cluster_width**2`` candidate slots plus
    the tile search, all admitted against ``maximum_candidate_slots``.
    Positions keep their floating dtype (fp32 or fp64); triclinic and
    partially periodic row lattices of rank 1-3 are supported.
    """

    cluster_search: CellListImageRouteSearch
    radius: float = eqx.field(static=True)
    cluster_width: int = eqx.field(static=True)
    cluster_count: int = eqx.field(static=True)
    maximum_cluster_radius: float = eqx.field(static=True)
    morton_depth: int = eqx.field(static=True)
    case_count: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    lattice_rank: int = eqx.field(static=True)
    capacity: ParticleImageCapacity
    candidate_slot_count: int = eqx.field(static=True)
    search_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius: float,
        reference_vectors: ArrayLike,
        capacity: ParticleImageCapacity,
        /,
        *,
        particle_capacity: int,
        cluster_width: int,
        maximum_cluster_radius: float,
        tile_capacity: ParticleImageCapacity,
        maximum_candidate_slots: int,
        deformation_margin: float = 0.0,
    ) -> None:
        value = float(radius)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Image search radius must be finite and positive.")
        width = int(cluster_width)
        if width <= 0:
            raise ValueError("cluster_width must be positive.")
        bound = float(maximum_cluster_radius)
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError("maximum_cluster_radius must be finite and nonnegative.")
        if not isinstance(capacity, ParticleImageCapacity) or not isinstance(
            tile_capacity, ParticleImageCapacity
        ):
            raise TypeError("capacity and tile_capacity must be ParticleImageCapacity.")
        if tile_capacity.maximum_case_edges is None:
            raise ValueError(
                "tile_capacity must declare maximum_case_edges, the per-case tile "
                "pair budget."
            )
        vectors = np.asarray(reference_vectors)
        reach = _host_reach(vectors)
        cases, rank = map(int, reach.shape)
        if rank > 3:
            raise ValueError("Cluster tile search supports lattice rank 1, 2 or 3.")
        particles = int(particle_capacity)
        if particles <= 0:
            raise ValueError("particle_capacity must be positive.")
        capacity.route_buffer(cases)
        clusters = -(-particles // width)
        cluster_search = CellListImageRouteSearch(
            value + 2.0 * bound,
            vectors,
            tile_capacity,
            particle_capacity=clusters,
            maximum_candidate_slots=maximum_candidate_slots,
            deformation_margin=deformation_margin,
        )
        tiles = tile_capacity.maximum_case_edges + clusters
        exact_slots = cases * tiles * width * width
        slots = cluster_search.candidate_slot_count + exact_slots
        if slots > min(int(maximum_candidate_slots), _INT32_LIMIT):
            raise ValueError(
                f"Cluster tile search requires {slots} candidate slots, exceeding "
                f"maximum_candidate_slots={int(maximum_candidate_slots)} or the "
                "int32 candidate index space."
            )
        depth = _MORTON_CODE_BITS // rank
        self.cluster_search = cluster_search
        self.radius = value
        self.cluster_width = width
        self.cluster_count = clusters
        self.maximum_cluster_radius = bound
        self.morton_depth = depth
        self.case_count = cases
        self.particle_capacity = particles
        self.lattice_rank = rank
        self.capacity = capacity
        self.candidate_slot_count = slots
        self.search_id = canonical_fingerprint(
            {
                "kind": "cluster-tile-image-route-search",
                "radius": value,
                "cluster_width": width,
                "maximum_cluster_radius": bound,
                "morton_depth": depth,
                "case_count": cases,
                "particle_capacity": particles,
                "capacity": capacity.capacity_id,
                "tile_search": cluster_search.search_id,
            }
        )

    @property
    def tile_capacity(self) -> ParticleImageCapacity:
        return self.cluster_search.capacity

    def _candidates(
        self,
        frame: _LatticeFrame,
        vectors: Array,
        periodic: Array,
        stable_ids: Array,
        receivers: _ResolvedReceivers,
        /,
    ) -> _Candidates:
        cases = self.case_count
        particles = self.particle_capacity
        width = self.cluster_width
        clusters = self.cluster_count
        rank = self.lattice_rank
        dtype = frame.wrapped.dtype
        scale = float(2**self.morton_depth)
        quantized = jnp.clip(
            jax.lax.stop_gradient(jnp.floor(frame.fractional * scale)), 0.0, scale - 1.0
        ).astype(jnp.int32)
        codes = morton_encode_integer(quantized, self.morton_depth)
        order = jax.vmap(canonical_morton_order)(codes, stable_ids, frame.usable)
        padding = clusters * width - particles
        member_slots = jnp.concatenate(
            (order, jnp.zeros((cases, padding), dtype=jnp.int32)), axis=1
        ).reshape((cases, clusters, width))
        member_valid = jnp.concatenate(
            (
                jnp.take_along_axis(frame.usable, order, axis=1),
                jnp.zeros((cases, padding), dtype=jnp.bool_),
            ),
            axis=1,
        ).reshape((cases, clusters, width))
        flat_members = member_slots.reshape((cases, -1, 1))
        member_fraction = jnp.take_along_axis(
            frame.fractional, flat_members, axis=1
        ).reshape((cases, clusters, width, rank))
        member_positions = jax.lax.stop_gradient(
            jnp.take_along_axis(frame.wrapped, flat_members, axis=1).reshape(
                (cases, clusters, width, -1)
            )
        )
        weights = member_valid.astype(dtype)
        counts = jnp.sum(weights, axis=-1)
        mean = (
            jnp.sum(member_fraction * weights[..., None], axis=2)
            / jnp.maximum(counts, 1.0)[..., None]
        )
        # A centroid strictly inside the half-open cell keeps the tile search in
        # its domain on nonperiodic axes despite the lattice solve's roundoff;
        # the bounding radius is measured from this centre, so completeness holds.
        margin = 1024.0 * float(jnp.finfo(dtype).eps)
        center_fraction = jax.lax.stop_gradient(jnp.clip(mean, margin, 1.0 - margin))
        centers = frame.origin[:, None, :] + contract(
            "cgr,crd->cgd", center_fraction, vectors, backend="jax"
        )
        offsets = member_positions - centers[:, :, None, :]
        radii = jnp.sqrt(
            jnp.max(
                jnp.where(member_valid, jnp.sum(offsets * offsets, axis=-1), 0.0),
                axis=-1,
            )
        )
        cluster_active = counts > 0.0
        extent_overflow = jnp.any(
            cluster_active & (radii > self.maximum_cluster_radius), axis=-1
        )
        cluster_ids = jnp.broadcast_to(
            jnp.arange(clusters, dtype=jnp.int32)[None, :], (cases, clusters)
        )
        tiles = self.cluster_search.routes(
            centers,
            cluster_active,
            cluster_ids,
            vectors,
            frame.origin,
            periodic,
            support_id="cluster-tile-centroids",
            relation_schema_id=self.search_id,
        )
        tile_relation = tiles.relation
        tile_budget = self.tile_capacity.maximum_case_edges
        if tile_budget is None:
            raise ValueError("tile_capacity must declare maximum_case_edges.")
        tile_sources = (tile_relation.source_indices % clusters).reshape(
            (cases, tile_budget)
        )
        tile_receivers = (tile_relation.receiver_indices % clusters).reshape(
            (cases, tile_budget)
        )
        tile_shifts = tile_relation.image_shifts.reshape((cases, tile_budget, rank))
        separation = tile_relation.displacement(
            centers.reshape((cases * clusters, -1)), vectors
        ).reshape((cases, tile_budget, -1))
        reach = (
            self.radius
            + jnp.take_along_axis(radii, tile_sources, axis=1)
            + jnp.take_along_axis(radii, tile_receivers, axis=1)
        )
        tile_valid = tile_relation.valid.reshape((cases, tile_budget)) & (
            jnp.sum(separation * separation, axis=-1) < reach * reach
        )
        source_clusters = jnp.concatenate((tile_sources, cluster_ids), axis=1)
        receiver_clusters = jnp.concatenate((tile_receivers, cluster_ids), axis=1)
        shifts = jnp.concatenate(
            (tile_shifts, jnp.zeros((cases, clusters, rank), dtype=jnp.int32)), axis=1
        )
        active_tiles = jnp.concatenate((tile_valid, cluster_active), axis=1)
        tile_count = tile_budget + clusters
        member_shape = (cases, tile_count, width)

        def gather(values: Array, cluster: Array, /) -> Array:
            return jnp.take_along_axis(
                values, jnp.broadcast_to(cluster[..., None], member_shape), axis=1
            )

        receiver_members = gather(member_slots, receiver_clusters)
        receiver_members_valid = gather(member_valid, receiver_clusters)
        source_members = gather(member_slots, source_clusters)
        source_members_valid = gather(member_valid, source_clusters)
        case_index = jnp.arange(cases, dtype=jnp.int32)[:, None]
        requested = (
            jnp.zeros((cases, particles), dtype=jnp.bool_)
            .at[case_index, jnp.where(receivers.valid, receivers.slots, particles)]
            .set(True, mode="drop")
        )
        receiver_requested = jnp.take_along_axis(
            requested, receiver_members.reshape((cases, -1)), axis=1
        ).reshape(member_shape)
        pair_valid = (
            active_tiles[:, :, None, None]
            & (receiver_members_valid & receiver_requested)[:, :, :, None]
            & source_members_valid[:, :, None, :]
        )
        rows = tile_count * width
        row_sources = jnp.broadcast_to(
            source_members[:, :, None, :], (cases, tile_count, width, width)
        ).reshape((cases, rows, width))
        row_shifts = jnp.broadcast_to(
            shifts[:, :, None, None, :], (cases, tile_count, width, width, rank)
        ).reshape((cases, rows, width, rank))
        row_receivers = receiver_members.reshape((cases, rows))
        candidate = _within_radius(
            frame,
            vectors,
            row_receivers,
            row_sources,
            row_shifts,
            pair_valid.reshape((cases, rows, width)),
            self.radius,
        )
        evidence = tiles.evidence
        envelope = (
            extent_overflow
            | evidence.domain_violation
            | evidence.representation_overflow
            | (evidence.nonfinite & frame.solved)
        )
        return _Candidates(
            candidate,
            row_receivers,
            row_sources,
            row_shifts,
            jnp.zeros((cases,), dtype=jnp.int32),
            jnp.zeros((cases,), dtype=jnp.bool_),
            evidence.required_offset_extents,
            evidence.stencil_overflow,
            evidence.required_edges + jnp.sum(cluster_active, axis=-1, dtype=jnp.int32),
            evidence.capacity_failure,
            envelope,
        )


__all__ = ["ClusterTileImageRouteSearch"]
