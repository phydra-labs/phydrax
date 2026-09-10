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

from ..._fingerprint import canonical_fingerprint
from ...sparse import EdgeRelation, KeyGroupPlan
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


class CellListParticleNeighborhoodPlan(AbstractParticleNeighborhoodPlan):
    """Fixed-capacity cell-list candidate relation for one particle box."""

    search_radius: float = eqx.field(static=True)
    maximum_particles_per_cell: int = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    maximum_candidate_slots: int = eqx.field(static=True)
    box: ParticleBox
    backend: ParticleRealization = eqx.field(static=True)
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        search_radius: float,
        maximum_particles_per_cell: int,
        maximum_pairs: int,
        box: ParticleBox,
        *,
        maximum_candidate_slots: int = 10_000_000,
        name: str = "cell-list-particle-neighborhood",
        plan_id: str | None = None,
    ):
        radius = float(search_radius)
        cell_capacity = int(maximum_particles_per_cell)
        pair_capacity = int(maximum_pairs)
        candidate_limit = int(maximum_candidate_slots)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("search_radius must be finite and positive.")
        if cell_capacity <= 0 or pair_capacity <= 0 or candidate_limit <= 0:
            raise ValueError(
                "Cell, pair, and candidate capacities must be positive integers."
            )
        if not isinstance(box, ParticleBox):
            raise TypeError("box must be a ParticleBox.")
        if box.ambient_dimension not in (1, 2, 3):
            raise ValueError("Cell-list neighborhoods support dimensions 1, 2, and 3.")
        key = DiscretizationKey(
            name,
            DiscretizationRole.AUXILIARY,
            domain_labels=("material_point", "pair_relation", "cell_list"),
        )
        self.search_radius = radius
        self.maximum_particles_per_cell = cell_capacity
        self.maximum_pairs = pair_capacity
        self.maximum_candidate_slots = candidate_limit
        self.box = box
        self.backend = "cell_edge_list"
        self.key = key
        self.plan_id = resolved_identifier(
            "plan_id",
            plan_id,
            {
                "kind": "cell-list-particle-neighborhood-plan",
                "search_radius": radius,
                "maximum_particles_per_cell": cell_capacity,
                "maximum_pairs": pair_capacity,
                "maximum_candidate_slots": candidate_limit,
                "box": box.box_id,
                "key": key.key_id,
            },
        )

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedCellListParticleNeighborhood:
        return PreparedCellListParticleNeighborhood(self, particles)


class PreparedCellListParticleNeighborhood(AbstractPreparedParticleNeighborhood):
    """Prepared occupied-cell topology with pure-JAX edge construction."""

    plan: CellListParticleNeighborhoodPlan
    particle_ids: Array
    active_mask: Array
    cell_widths: Array
    cell_strides: Array
    neighbor_offsets: Array
    key_groups: KeyGroupPlan
    preparation: PreparationReport
    key: DiscretizationKey
    box: ParticleBox
    backend: ParticleRealization = eqx.field(static=True)
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    neighbor_cell_capacity: int = eqx.field(static=True)
    maximum_particles_per_cell: int = eqx.field(static=True)
    pair_capacity: int = eqx.field(static=True)
    candidate_slot_count: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    source_support_id: str = eqx.field(static=True)
    relation_schema_id: str = eqx.field(static=True)
    particle_discretization_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    artifact_kind: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CellListParticleNeighborhoodPlan,
        particles: ParticleDiscretization,
        /,
    ):
        if not isinstance(plan, CellListParticleNeighborhoodPlan):
            raise TypeError("plan must be a CellListParticleNeighborhoodPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if plan.box.ambient_dimension != particles.ambient_dimension:
            raise ValueError("ParticleBox dimension does not match particle support.")
        lengths = np.asarray(plan.box.lengths, dtype=float)
        shape = tuple(
            max(int(np.floor(length / plan.search_radius)), 1) for length in lengths
        )
        widths = lengths / np.asarray(shape, dtype=float)
        if any(
            cells > 1 and width < plan.search_radius
            for cells, width in zip(shape, widths, strict=True)
        ):
            raise AssertionError(
                "Multi-cell axes must cover the search radius in adjacent cells."
            )
        cell_count = prod(shape)
        neighbor_offsets = np.asarray(
            tuple(product((-1, 0, 1), repeat=particles.ambient_dimension)),
            dtype=np.int32,
        )
        neighbor_cell_capacity = int(neighbor_offsets.shape[0])
        candidate_slots = (
            particles.capacity * neighbor_cell_capacity * plan.maximum_particles_per_cell
        )
        if candidate_slots > plan.maximum_candidate_slots:
            raise ValueError(
                f"Cell-list relation requires {candidate_slots} candidate slots, "
                f"exceeding maximum_candidate_slots={plan.maximum_candidate_slots}."
            )
        occupied_cell_capacity = min(particles.capacity, cell_count)
        key_groups = KeyGroupPlan(
            particles.capacity,
            max(occupied_cell_capacity, 1),
            cell_count - 1,
            maximum_group_size=plan.maximum_particles_per_cell,
        )
        relation_schema_id = canonical_fingerprint(
            {
                "kind": "cell-list-particle-pair-relation-schema",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "source_support": particles.support.support_id,
                "pair_capacity": plan.maximum_pairs,
            }
        )
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
                DiscretizationCapability.MATRIX_FREE,
            ),
            diagnostics=(
                "cell and pair capacities are fixed",
                "only occupied cell keys are materialized",
                "cell and edge selection are frozen branchwise decisions",
                "overflow and nonperiodic domain violations fail closed",
                "public particle state remains in logical order",
            ),
            resource_counts={
                "particle_capacity": particles.capacity,
                "logical_cell_count": cell_count,
                "occupied_cell_capacity": occupied_cell_capacity,
                "neighbor_cell_capacity": neighbor_cell_capacity,
                "maximum_particles_per_cell": plan.maximum_particles_per_cell,
                "candidate_slot_count": candidate_slots,
                "pair_capacity": plan.maximum_pairs,
                "particle_table_slots": particles.capacity,
                "dense_cell_slots": 0,
            },
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cell-list-particle-neighborhood",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "cell_shape": list(shape),
                "cell_widths": widths.tolist(),
                "neighbor_offsets": neighbor_offsets.tolist(),
                "key_group_plan": key_groups.plan_id,
                "relation_schema": relation_schema_id,
                "preparation": preparation.report_id,
                "numeric_version": particles.numeric_version,
            }
        )
        self.plan = plan
        self.particle_ids = particles.particle_ids
        self.active_mask = particles.active_mask
        self.cell_widths = jnp.asarray(widths, dtype=plan.box.lengths.dtype)
        self.cell_strides = jnp.asarray(_cell_strides(shape), dtype=jnp.int32)
        self.neighbor_offsets = jnp.asarray(neighbor_offsets)
        self.key_groups = key_groups
        self.preparation = preparation
        self.key = plan.key
        self.box = plan.box
        self.backend = plan.backend
        self.cell_shape = shape
        self.cell_count = cell_count
        self.neighbor_cell_capacity = neighbor_cell_capacity
        self.maximum_particles_per_cell = plan.maximum_particles_per_cell
        self.pair_capacity = plan.maximum_pairs
        self.candidate_slot_count = candidate_slots
        self.particle_capacity = particles.capacity
        self.ambient_dimension = particles.ambient_dimension
        self.source_support_id = particles.support.support_id
        self.relation_schema_id = relation_schema_id
        self.particle_discretization_id = particles.prepared_id
        self.numeric_version = particles.numeric_version
        self.artifact_kind = "cell-list-particle-neighborhood"
        self.prepared_id = prepared_id

    def _logical_cell_ids(
        self, position: Array, active_mask: Array, /
    ) -> tuple[Array, Array, Array]:
        finite = jnp.all(jnp.isfinite(position), axis=-1)
        safe = jnp.where(finite[:, None], position, self.box.lower)
        relative = (safe - self.box.lower.astype(safe.dtype)) / self.cell_widths.astype(
            safe.dtype
        )
        coordinates = jax.lax.stop_gradient(jnp.floor(relative).astype(jnp.int32))
        domain_valid = finite
        resolved_coordinates = []
        for axis, size in enumerate(self.cell_shape):
            coordinate = coordinates[:, axis]
            if self.box.periodic_axes[axis]:
                coordinate = jnp.mod(coordinate, size)
            else:
                axis_valid = (safe[:, axis] >= self.box.lower[axis]) & (
                    safe[:, axis] < self.box.upper[axis]
                )
                domain_valid = domain_valid & axis_valid
                coordinate = jnp.clip(coordinate, 0, size - 1)
            resolved_coordinates.append(coordinate)
        coordinate_array = jnp.stack(resolved_coordinates, axis=-1)
        cell_ids = jnp.sum(coordinate_array * self.cell_strides, axis=-1)
        active_valid = active_mask & domain_valid
        return (
            jnp.where(active_valid, cell_ids, -1),
            coordinate_array,
            active_mask & ~domain_valid,
        )

    def _neighbor_cell_ids(
        self, coordinates: Array, active_valid: Array, /
    ) -> tuple[Array, Array]:
        candidates = coordinates[:, None, :] + self.neighbor_offsets[None, :, :]
        valid = jnp.broadcast_to(active_valid[:, None], candidates.shape[:-1])
        resolved = []
        for axis, size in enumerate(self.cell_shape):
            coordinate = candidates[..., axis]
            if self.box.periodic_axes[axis]:
                coordinate = jnp.mod(coordinate, size)
            else:
                valid = valid & (coordinate >= 0) & (coordinate < size)
                coordinate = jnp.clip(coordinate, 0, size - 1)
            resolved.append(coordinate)
        coordinate_array = jnp.stack(resolved, axis=-1)
        cell_ids = jnp.sum(coordinate_array * self.cell_strides, axis=-1)
        sortable = jnp.where(valid, cell_ids, self.cell_count)
        order = jnp.argsort(sortable, axis=-1)
        sorted_ids = jnp.take_along_axis(sortable, order, axis=-1)
        sorted_valid = sorted_ids < self.cell_count
        previous = jnp.concatenate(
            (
                jnp.full(sorted_ids.shape[:-1] + (1,), self.cell_count),
                sorted_ids[..., :-1],
            ),
            axis=-1,
        )
        unique = sorted_valid & (sorted_ids != previous)
        return jnp.where(unique, sorted_ids, 0), unique

    def build(
        self, position: ArrayLike, /, *, active_mask: ArrayLike | None = None
    ) -> ParticleNeighborhoodState:
        value = jnp.asarray(position)
        expected = (self.particle_capacity, self.ambient_dimension)
        if value.shape != expected:
            raise ValueError(f"Particle positions must have shape {expected}.")
        active = self.active_mask
        if active_mask is not None:
            requested = jnp.asarray(active_mask, dtype=bool)
            if requested.shape != (self.particle_capacity,):
                raise ValueError("active_mask must have particle-capacity shape.")
            active = active & requested
        cell_ids, cell_coordinates, domain_violations = self._logical_cell_ids(
            value, active
        )
        active_valid = active & ~domain_violations
        groups = self.key_groups.build(
            cell_ids,
            active_valid,
            stable_ids=self.particle_ids,
        )
        neighbor_ids, neighbor_valid = self._neighbor_cell_ids(
            cell_coordinates, active_valid
        )
        neighbor_lookup = groups.lookup(neighbor_ids, valid=neighbor_valid)
        safe_groups = neighbor_lookup.group_slots
        group_starts = groups.group_starts[safe_groups]
        group_counts = groups.group_counts[safe_groups]
        member_rank = jnp.arange(self.maximum_particles_per_cell, dtype=jnp.int32)
        sorted_positions = group_starts[..., None] + member_rank
        sorted_positions = jnp.clip(sorted_positions, 0, self.particle_capacity - 1)
        right_indices = groups.storage_to_logical[sorted_positions]
        right_valid = neighbor_lookup.supported[..., None] & (
            member_rank < group_counts[..., None]
        )
        left_indices = jnp.broadcast_to(
            jnp.arange(self.particle_capacity, dtype=jnp.int32)[:, None, None],
            right_indices.shape,
        )
        candidate_valid = active_valid[:, None, None] & right_valid
        left_ids = self.particle_ids[left_indices]
        right_ids = self.particle_ids[right_indices]
        candidate_valid = candidate_valid & (left_ids < right_ids)
        left_position = value[left_indices]
        right_position = value[right_indices]
        displacement = self.box.minimum_image(left_position - right_position)
        distance_squared = jnp.sum(displacement * displacement, axis=-1)
        candidate_valid = candidate_valid & (
            distance_squared < self.plan.search_radius**2
        )
        flat_valid = candidate_valid.reshape((-1,))
        candidate_pair_count = jnp.sum(flat_valid, dtype=jnp.int32)
        pair_overflow_count = jnp.maximum(candidate_pair_count - self.pair_capacity, 0)
        pair_overflow = pair_overflow_count > 0
        selected = jax.lax.stop_gradient(
            jnp.nonzero(flat_valid, size=self.pair_capacity, fill_value=0)[0]
        )
        flat_left = left_indices.reshape((-1,))
        flat_right = right_indices.reshape((-1,))
        selected_left = flat_left[selected]
        selected_right = flat_right[selected]
        pair_count = jnp.minimum(candidate_pair_count, self.pair_capacity)
        route_valid = jnp.arange(self.pair_capacity, dtype=jnp.int32) < pair_count
        relation = EdgeRelation(
            selected_left,
            selected_right,
            source_size=self.particle_capacity,
            target_size=self.particle_capacity,
            valid=route_valid,
        )
        pair_relation = ParticlePairRelation(
            relation,
            self.particle_ids[selected_left],
            self.particle_ids[selected_right],
            source_support_id=self.source_support_id,
            target_support_id=self.source_support_id,
            same_set=True,
            unordered=True,
            relation_schema_id=self.relation_schema_id,
        )
        excess = jnp.maximum(groups.group_counts - self.maximum_particles_per_cell, 0)
        cell_overflow_count = jnp.sum(excess, dtype=jnp.int32)
        cell_overflow = (
            groups.evidence.group_overflow
            | groups.evidence.member_overflow
            | (groups.evidence.duplicate_stable_ids > 0)
        )
        domain_violation_count = jnp.sum(domain_violations, dtype=jnp.int32)
        return ParticleNeighborhoodState(
            pair_relation,
            box=self.box,
            storage_to_logical=groups.storage_to_logical,
            logical_to_storage=groups.logical_to_storage,
            cell_ids=cell_ids,
            cell_counts=groups.group_counts,
            cell_offsets=groups.group_starts,
            candidate_pair_count=candidate_pair_count,
            pair_count=pair_count,
            maximum_cell_occupancy=groups.evidence.maximum_group_size,
            cell_overflow=cell_overflow,
            cell_overflow_count=cell_overflow_count,
            pair_overflow=pair_overflow,
            pair_overflow_count=pair_overflow_count,
            domain_violation=domain_violation_count > 0,
            domain_violation_count=domain_violation_count,
            prepared_neighborhood_id=self.prepared_id,
            relation_schema_id=self.relation_schema_id,
        )


__all__ = [
    "CellListParticleNeighborhoodPlan",
    "PreparedCellListParticleNeighborhood",
]
