#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...sparse import EdgeRelation
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
    resolved_identifier,
)
from ._core import ParticleDiscretization
from ._neighborhood import (
    AbstractParticleNeighborhoodPlan,
    AbstractPreparedParticleNeighborhood,
    ParticleNeighborhoodState,
)
from ._pairwise import ParticleBox, ParticlePairRelation
from ._precision import ParticleRealization


def _cell_strides(shape: tuple[int, ...], /) -> tuple[int, ...]:
    return tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))


class HierarchicalRadiusParticleNeighborhoodPlan(AbstractParticleNeighborhoodPlan):
    """Sparse bottom-up multilevel grid for bounded interaction radii."""

    interaction_radii: Array
    level_edges: Array
    level_ids: Array
    maximum_particles_per_cell: int = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    maximum_candidate_slots: int = eqx.field(static=True)
    skin: float = eqx.field(static=True)
    box: ParticleBox
    backend: ParticleRealization = eqx.field(static=True)
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        interaction_radii: ArrayLike,
        level_edges: ArrayLike,
        maximum_particles_per_cell: int,
        maximum_pairs: int,
        box: ParticleBox,
        /,
        *,
        skin: float = 0.0,
        maximum_candidate_slots: int = 10_000_000,
        name: str = "hierarchical-radius-particle-neighborhood",
        plan_id: str | None = None,
    ):
        radii = np.asarray(interaction_radii)
        edges = np.asarray(level_edges)
        cell_capacity = int(maximum_particles_per_cell)
        pair_capacity = int(maximum_pairs)
        candidate_capacity = int(maximum_candidate_slots)
        skin_ = float(skin)
        if radii.ndim != 1 or radii.size == 0:
            raise ValueError("interaction_radii must be a nonempty rank-1 array.")
        if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("interaction_radii must be finite and positive.")
        if (
            edges.ndim != 1
            or edges.size < 2
            or np.any(~np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("level_edges must be finite and strictly increasing.")
        if edges[0] > np.min(radii) or edges[-1] < np.max(radii):
            raise ValueError("level_edges must cover every interaction radius.")
        if cell_capacity <= 0 or pair_capacity <= 0 or candidate_capacity <= 0:
            raise ValueError("Cell, pair, and candidate capacities must be positive.")
        if not np.isfinite(skin_) or skin_ < 0.0:
            raise ValueError("skin must be finite and nonnegative.")
        if not isinstance(box, ParticleBox):
            raise TypeError("box must be a ParticleBox.")
        if box.ambient_dimension not in (1, 2, 3):
            raise ValueError("Hierarchical neighborhoods support dimensions 1 through 3.")
        levels = np.searchsorted(edges[1:-1], radii, side="right").astype(np.int32)
        key = DiscretizationKey(
            name,
            DiscretizationRole.AUXILIARY,
            domain_labels=("material_point", "pair_relation", "radius_hierarchy"),
        )
        self.interaction_radii = jnp.asarray(radii)
        self.level_edges = jnp.asarray(edges)
        self.level_ids = jnp.asarray(levels)
        self.maximum_particles_per_cell = cell_capacity
        self.maximum_pairs = pair_capacity
        self.maximum_candidate_slots = candidate_capacity
        self.skin = skin_
        self.box = box
        self.backend = "cell_edge_list"
        self.key = key
        self.plan_id = resolved_identifier(
            "plan_id",
            plan_id,
            {
                "kind": "hierarchical-radius-particle-neighborhood-plan",
                "interaction_radii": array_tree_fingerprint(radii),
                "level_edges": edges.tolist(),
                "maximum_particles_per_cell": cell_capacity,
                "maximum_pairs": pair_capacity,
                "maximum_candidate_slots": candidate_capacity,
                "skin": skin_,
                "box": box.box_id,
                "key": key.key_id,
            },
        )

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedHierarchicalRadiusParticleNeighborhood:
        return PreparedHierarchicalRadiusParticleNeighborhood(self, particles)


class PreparedHierarchicalRadiusParticleNeighborhood(
    AbstractPreparedParticleNeighborhood
):
    plan: HierarchicalRadiusParticleNeighborhoodPlan
    particle_ids: Array
    active_mask: Array
    cell_shapes: Array
    cell_widths: Array
    cell_strides: Array
    cell_counts_by_level: Array
    level_cell_offsets: Array
    neighbor_offsets: Array
    level_members: tuple[Array, ...]
    preparation: PreparationReport
    key: DiscretizationKey
    box: ParticleBox
    backend: ParticleRealization = eqx.field(static=True)
    level_count: int = eqx.field(static=True)
    neighbor_width: int = eqx.field(static=True)
    pair_capacity: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    candidate_slot_count: int = eqx.field(static=True)
    particle_discretization_id: str = eqx.field(static=True)
    source_support_id: str = eqx.field(static=True)
    relation_schema_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    artifact_kind: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: HierarchicalRadiusParticleNeighborhoodPlan,
        particles: ParticleDiscretization,
        /,
    ):
        if not isinstance(plan, HierarchicalRadiusParticleNeighborhoodPlan):
            raise TypeError("plan must be a HierarchicalRadiusParticleNeighborhoodPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if plan.interaction_radii.shape != (particles.capacity,):
            raise ValueError("Interaction radii must match particle capacity.")
        if plan.box.ambient_dimension != particles.ambient_dimension:
            raise ValueError("ParticleBox dimension does not match particle support.")
        level_count = int(plan.level_edges.shape[0] - 1)
        dimension = particles.ambient_dimension
        lengths = np.asarray(plan.box.lengths, dtype=float)
        upper_edges = np.asarray(plan.level_edges[1:], dtype=float)
        shapes = []
        widths = []
        strides = []
        cell_counts = []
        for upper in upper_edges:
            desired_width = 2.0 * upper + plan.skin
            shape = tuple(
                max(int(np.floor(length / desired_width)), 1) for length in lengths
            )
            width = lengths / np.asarray(shape, dtype=float)
            shapes.append(shape)
            widths.append(width)
            strides.append(_cell_strides(shape))
            cell_counts.append(prod(shape))
        offsets = np.asarray(tuple(product((-1, 0, 1), repeat=dimension)), dtype=np.int32)
        level_population = np.bincount(np.asarray(plan.level_ids), minlength=level_count)
        candidate_slots = int(
            sum(
                int(level_population[level])
                * (level_count - level)
                * offsets.shape[0]
                * plan.maximum_particles_per_cell
                for level in range(level_count)
            )
        )
        if candidate_slots > plan.maximum_candidate_slots:
            raise ValueError(
                f"Hierarchical relation requires {candidate_slots} candidate slots, "
                f"exceeding maximum_candidate_slots={plan.maximum_candidate_slots}."
            )
        relation_schema_id = canonical_fingerprint(
            {
                "kind": "hierarchical-radius-pair-relation-schema",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "pair_capacity": plan.maximum_pairs,
            }
        )
        level_offsets = np.cumsum((0, *cell_counts[:-1]), dtype=np.int64)
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
                DiscretizationCapability.MATRIX_FREE,
                DiscretizationCapability.TOPOLOGY_REFRESH_FIXED_CAPACITY,
            ),
            diagnostics=(
                "immutable interaction-radius classes",
                "sparse occupied-cell lookup by stable sorted keys",
                "bottom-up cross-level traversal",
                "pair-specific interaction-radius fine filtering",
                "cell, candidate, and pair overflow fail closed",
            ),
            resource_counts={
                "particle_capacity": particles.capacity,
                "radius_levels": level_count,
                "maximum_level_population": int(np.max(level_population)),
                "neighbor_width": int(offsets.shape[0]),
                "maximum_particles_per_cell": plan.maximum_particles_per_cell,
                "candidate_slot_count": candidate_slots,
                "pair_capacity": plan.maximum_pairs,
                "sorted_cell_slots": particles.capacity * level_count,
                "dense_cell_slots": 0,
            },
        )
        self.plan = plan
        self.particle_ids = particles.particle_ids
        self.active_mask = particles.active_mask
        self.cell_shapes = jnp.asarray(shapes, dtype=jnp.int32)
        self.cell_widths = jnp.asarray(widths, dtype=plan.box.lengths.dtype)
        self.cell_strides = jnp.asarray(strides, dtype=jnp.int64)
        self.cell_counts_by_level = jnp.asarray(cell_counts, dtype=jnp.int64)
        self.level_cell_offsets = jnp.asarray(level_offsets, dtype=jnp.int64)
        self.neighbor_offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.level_members = tuple(
            jnp.asarray(
                np.flatnonzero(np.asarray(plan.level_ids) == level),
                dtype=jnp.int32,
            )
            for level in range(level_count)
        )
        self.preparation = preparation
        self.key = plan.key
        self.box = plan.box
        self.backend = plan.backend
        self.level_count = level_count
        self.neighbor_width = int(offsets.shape[0])
        self.pair_capacity = plan.maximum_pairs
        self.particle_capacity = particles.capacity
        self.ambient_dimension = dimension
        self.candidate_slot_count = candidate_slots
        self.particle_discretization_id = particles.prepared_id
        self.source_support_id = particles.support.support_id
        self.relation_schema_id = relation_schema_id
        self.numeric_version = particles.numeric_version
        self.artifact_kind = "hierarchical-radius-particle-neighborhood"
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-hierarchical-radius-particle-neighborhood",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "cell_shapes": shapes,
                "candidate_slot_count": candidate_slots,
                "relation_schema": relation_schema_id,
                "preparation": preparation.report_id,
            }
        )

    def _level_cells(
        self, position: Array, active: Array, level: int, /
    ) -> tuple[Array, Array, Array]:
        finite = jnp.all(jnp.isfinite(position), axis=-1)
        safe = jnp.where(finite[:, None], position, self.box.lower)
        relative = (safe - self.box.lower.astype(safe.dtype)) / self.cell_widths[
            level
        ].astype(safe.dtype)
        coordinates = jax.lax.stop_gradient(jnp.floor(relative).astype(jnp.int32))
        domain_valid = finite
        resolved = []
        for axis in range(self.ambient_dimension):
            coordinate = coordinates[:, axis]
            size = self.cell_shapes[level, axis]
            if self.box.periodic_axes[axis]:
                coordinate = jnp.mod(coordinate, size)
            else:
                axis_valid = (safe[:, axis] >= self.box.lower[axis]) & (
                    safe[:, axis] < self.box.upper[axis]
                )
                domain_valid = domain_valid & axis_valid
                coordinate = jnp.clip(coordinate, 0, size - 1)
            resolved.append(coordinate)
        coordinate_array = jnp.stack(resolved, axis=-1)
        cell_ids = jnp.sum(
            coordinate_array.astype(jnp.int64) * self.cell_strides[level], axis=-1
        )
        return coordinate_array, cell_ids, active & ~domain_valid

    def _neighbor_keys(
        self, coordinates: Array, query_valid: Array, level: int, /
    ) -> tuple[Array, Array]:
        candidates = coordinates[:, None, :] + self.neighbor_offsets[None, :, :]
        valid = jnp.broadcast_to(query_valid[:, None], candidates.shape[:2])
        resolved = []
        for axis in range(self.ambient_dimension):
            coordinate = candidates[:, :, axis]
            size = self.cell_shapes[level, axis]
            if self.box.periodic_axes[axis]:
                coordinate = jnp.mod(coordinate, size)
            else:
                axis_valid = (coordinate >= 0) & (coordinate < size)
                valid = valid & axis_valid
                coordinate = jnp.clip(coordinate, 0, size - 1)
            resolved.append(coordinate)
        coordinate_array = jnp.stack(resolved, axis=-1)
        keys = jnp.sum(
            coordinate_array.astype(jnp.int64) * self.cell_strides[level], axis=-1
        )
        sentinel = self.cell_counts_by_level[level]
        sortable = jnp.where(valid, keys, sentinel)
        ordered = jnp.sort(sortable, axis=1)
        unique = (ordered < sentinel) & (
            (jnp.arange(self.neighbor_width)[None, :] == 0)
            | (ordered != jnp.roll(ordered, 1, axis=1))
        )
        return ordered, unique

    def build(
        self, positions: ArrayLike, /, *, active_mask: ArrayLike | None = None
    ) -> ParticleNeighborhoodState:
        position = jnp.asarray(positions)
        expected = (self.particle_capacity, self.ambient_dimension)
        if position.shape != expected:
            raise ValueError(f"Particle positions must have shape {expected}.")
        active = self.active_mask
        if active_mask is not None:
            requested = jnp.asarray(active_mask, dtype=bool)
            if requested.shape != (self.particle_capacity,):
                raise ValueError("active_mask must have particle-capacity shape.")
            active = active & requested
        level_coordinates = []
        level_cells = []
        sorted_indices = []
        sorted_keys = []
        domain_violations = jnp.zeros((self.particle_capacity,), dtype=bool)
        maximum_occupancy = jnp.zeros((), dtype=jnp.int32)
        cell_overflow_count = jnp.zeros((), dtype=jnp.int32)
        indices = jnp.arange(self.particle_capacity, dtype=jnp.int32)
        for level in range(self.level_count):
            coordinates, cell_ids, violations = self._level_cells(position, active, level)
            domain_violations = domain_violations | violations
            target_valid = active & ~violations & (self.plan.level_ids == level)
            sentinel = self.cell_counts_by_level[level]
            sortable = jnp.where(target_valid, cell_ids, sentinel)
            order = jax.lax.stop_gradient(
                jnp.lexsort((self.particle_ids, sortable)).astype(jnp.int32)
            )
            keys = sortable[order]
            valid_sorted = keys < sentinel
            starts = jnp.where((indices == 0) | (keys != jnp.roll(keys, 1)), indices, 0)
            starts = jax.lax.associative_scan(jnp.maximum, starts)
            rank = indices - starts
            occupancy = jnp.max(jnp.where(valid_sorted, rank + 1, 0), initial=0)
            overflow = jnp.sum(
                valid_sorted & (rank >= self.plan.maximum_particles_per_cell),
                dtype=jnp.int32,
            )
            maximum_occupancy = jnp.maximum(maximum_occupancy, occupancy)
            cell_overflow_count = cell_overflow_count + overflow
            level_coordinates.append(coordinates)
            level_cells.append(cell_ids)
            sorted_indices.append(order)
            sorted_keys.append(keys)

        candidate_left = []
        candidate_right = []
        candidate_valid = []
        broad_candidate_count = jnp.zeros((), dtype=jnp.int32)
        occupant_rank = jnp.arange(self.plan.maximum_particles_per_cell, dtype=jnp.int32)
        for query_level in range(self.level_count):
            query_indices = self.level_members[query_level]
            query_valid = active[query_indices] & ~domain_violations[query_indices]
            for target_level in range(query_level, self.level_count):
                keys, neighbor_valid = self._neighbor_keys(
                    level_coordinates[target_level][query_indices],
                    query_valid,
                    target_level,
                )
                starts = jnp.searchsorted(
                    sorted_keys[target_level], keys, side="left"
                ).astype(jnp.int32)
                slots = starts[:, :, None] + occupant_rank[None, None, :]
                in_bounds = slots < self.particle_capacity
                safe_slots = jnp.clip(slots, 0, self.particle_capacity - 1)
                right = sorted_indices[target_level][safe_slots]
                matched = (
                    in_bounds
                    & neighbor_valid[:, :, None]
                    & (sorted_keys[target_level][safe_slots] == keys[:, :, None])
                )
                left = jnp.broadcast_to(query_indices[:, None, None], right.shape)
                left_ids = self.particle_ids[left]
                right_ids = self.particle_ids[right]
                matched = matched & query_valid[:, None, None]
                if query_level == target_level:
                    matched = matched & (left_ids < right_ids)
                canonical_left = jnp.where(left_ids < right_ids, left, right)
                canonical_right = jnp.where(left_ids < right_ids, right, left)
                broad_candidate_count = broad_candidate_count + jnp.sum(
                    matched, dtype=jnp.int32
                )
                displacement = self.box.minimum_image(
                    position[canonical_left] - position[canonical_right]
                )
                distance_squared = jnp.sum(displacement * displacement, axis=-1)
                reach = (
                    self.plan.interaction_radii[canonical_left]
                    + self.plan.interaction_radii[canonical_right]
                    + self.plan.skin
                )
                valid = matched & (distance_squared < reach**2)
                candidate_left.append(canonical_left.reshape((-1,)))
                candidate_right.append(canonical_right.reshape((-1,)))
                candidate_valid.append(valid.reshape((-1,)))

        all_left = jnp.concatenate(candidate_left)
        all_right = jnp.concatenate(candidate_right)
        all_valid = jnp.concatenate(candidate_valid)
        pair_count_unclipped = jnp.sum(all_valid, dtype=jnp.int32)
        pair_overflow_count = jnp.maximum(pair_count_unclipped - self.pair_capacity, 0)
        selected = jax.lax.stop_gradient(
            jnp.nonzero(all_valid, size=self.pair_capacity, fill_value=0)[0]
        )
        pair_count = jnp.minimum(pair_count_unclipped, self.pair_capacity)
        route_valid = jnp.arange(self.pair_capacity, dtype=jnp.int32) < pair_count
        selected_left = all_left[selected]
        selected_right = all_right[selected]
        selected_left_ids = self.particle_ids[selected_left]
        selected_right_ids = self.particle_ids[selected_right]
        sentinel_id = jnp.iinfo(jnp.int64).max
        sort_left = jnp.where(route_valid, selected_left_ids, sentinel_id)
        sort_right = jnp.where(route_valid, selected_right_ids, sentinel_id)
        pair_order = jax.lax.stop_gradient(
            jnp.lexsort((sort_right, sort_left)).astype(jnp.int32)
        )
        selected_left = selected_left[pair_order]
        selected_right = selected_right[pair_order]
        route_valid = route_valid[pair_order]
        relation = EdgeRelation(
            selected_left,
            selected_right,
            source_size=self.particle_capacity,
            target_size=self.particle_capacity,
            valid=route_valid,
        )
        pairs = ParticlePairRelation(
            relation,
            self.particle_ids[selected_left],
            self.particle_ids[selected_right],
            source_support_id=self.source_support_id,
            target_support_id=self.source_support_id,
            same_set=True,
            unordered=True,
            relation_schema_id=self.relation_schema_id,
        )
        cells_by_level = jnp.stack(level_cells, axis=1)
        own_cells = jnp.take_along_axis(
            cells_by_level,
            self.plan.level_ids[:, None],
            axis=1,
        )[:, 0]
        global_cells = own_cells + self.level_cell_offsets[self.plan.level_ids]
        global_cells = jnp.where(active & ~domain_violations, global_cells, -1)
        sortable_global = jnp.where(
            global_cells >= 0,
            global_cells,
            jnp.sum(self.cell_counts_by_level),
        )
        storage_to_logical = jax.lax.stop_gradient(
            jnp.lexsort((self.particle_ids, sortable_global)).astype(jnp.int32)
        )
        logical_to_storage = (
            jnp.zeros((self.particle_capacity,), dtype=jnp.int32)
            .at[storage_to_logical]
            .set(indices)
        )
        level_counts = jnp.bincount(
            jnp.where(active & ~domain_violations, self.plan.level_ids, self.level_count),
            weights=(active & ~domain_violations).astype(jnp.int32),
            length=self.level_count + 1,
        )[: self.level_count].astype(jnp.int32)
        level_offsets = jnp.cumsum(level_counts, dtype=jnp.int32) - level_counts
        domain_violation_count = jnp.sum(domain_violations, dtype=jnp.int32)
        return ParticleNeighborhoodState(
            pairs,
            box=self.box,
            storage_to_logical=storage_to_logical,
            logical_to_storage=logical_to_storage,
            cell_ids=global_cells,
            cell_counts=level_counts,
            cell_offsets=level_offsets,
            candidate_pair_count=broad_candidate_count,
            pair_count=pair_count,
            maximum_cell_occupancy=maximum_occupancy,
            cell_overflow=cell_overflow_count > 0,
            cell_overflow_count=cell_overflow_count,
            pair_overflow=pair_overflow_count > 0,
            pair_overflow_count=pair_overflow_count,
            domain_violation=domain_violation_count > 0,
            domain_violation_count=domain_violation_count,
            prepared_neighborhood_id=self.prepared_id,
            relation_schema_id=self.relation_schema_id,
        )


__all__ = [
    "HierarchicalRadiusParticleNeighborhoodPlan",
    "PreparedHierarchicalRadiusParticleNeighborhood",
]
