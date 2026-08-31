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
from ...sparse import EdgeRelation, RowRelation
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


def _neighbor_cell_relation(
    shape: tuple[int, ...], periodic_axes: tuple[bool, ...], /
) -> RowRelation:
    count = prod(shape)
    rows: list[tuple[int, ...]] = []
    offsets = tuple(product((-1, 0, 1), repeat=len(shape)))
    for cell_id in range(count):
        coordinate = np.unravel_index(cell_id, shape)
        neighbors: set[int] = set()
        for offset in offsets:
            candidate = []
            valid = True
            for axis, (index, shift, size, periodic) in enumerate(
                zip(coordinate, offset, shape, periodic_axes, strict=True)
            ):
                del axis
                value = index + shift
                if periodic:
                    value %= size
                elif value < 0 or value >= size:
                    valid = False
                    break
                candidate.append(value)
            if valid:
                neighbors.add(int(np.ravel_multi_index(tuple(candidate), shape)))
        rows.append(tuple(sorted(neighbors)))
    width = max(len(row) for row in rows)
    indices = np.zeros((count, width), dtype=np.int32)
    valid = np.zeros((count, width), dtype=bool)
    for cell_id, row in enumerate(rows):
        indices[cell_id, : len(row)] = row
        valid[cell_id, : len(row)] = True
    return RowRelation(indices, source_size=count, valid=valid)


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
    """Prepared cell topology with pure-JAX runtime edge construction."""

    plan: CellListParticleNeighborhoodPlan
    particle_ids: Array
    active_mask: Array
    cell_widths: Array
    cell_strides: Array
    neighbor_cells: RowRelation
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
        neighbor_cells = _neighbor_cell_relation(shape, plan.box.periodic_axes)
        candidate_slots = (
            particles.capacity * neighbor_cells.width * plan.maximum_particles_per_cell
        )
        if candidate_slots > plan.maximum_candidate_slots:
            raise ValueError(
                f"Cell-list relation requires {candidate_slots} candidate slots, "
                f"exceeding maximum_candidate_slots={plan.maximum_candidate_slots}."
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
                "cell and edge selection are frozen branchwise decisions",
                "overflow and nonperiodic domain violations fail closed",
                "public particle state remains in logical order",
            ),
            resource_counts={
                "particle_capacity": particles.capacity,
                "cell_count": cell_count,
                "neighbor_cell_capacity": neighbor_cells.width,
                "maximum_particles_per_cell": plan.maximum_particles_per_cell,
                "candidate_slot_count": candidate_slots,
                "pair_capacity": plan.maximum_pairs,
                "particle_table_slots": cell_count * plan.maximum_particles_per_cell,
            },
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cell-list-particle-neighborhood",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "cell_shape": list(shape),
                "cell_widths": widths.tolist(),
                "neighbor_cells": {
                    "shape": list(neighbor_cells.route_shape),
                    "indices": np.asarray(neighbor_cells.source_indices).tolist(),
                    "valid": np.asarray(neighbor_cells.valid).tolist(),
                },
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
        self.neighbor_cells = neighbor_cells
        self.preparation = preparation
        self.key = plan.key
        self.box = plan.box
        self.backend = plan.backend
        self.cell_shape = shape
        self.cell_count = cell_count
        self.neighbor_cell_capacity = neighbor_cells.width
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
    ) -> tuple[Array, Array]:
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
        return jnp.where(active_valid, cell_ids, -1), active_mask & ~domain_valid

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
        cell_ids, domain_violations = self._logical_cell_ids(value, active)
        active_valid = active & ~domain_violations
        sentinel_cell = self.cell_count
        sortable_cells = jnp.where(active_valid, cell_ids, sentinel_cell)
        storage_to_logical = jax.lax.stop_gradient(
            jnp.lexsort((self.particle_ids, sortable_cells)).astype(jnp.int32)
        )
        logical_to_storage = (
            jnp.zeros((self.particle_capacity,), dtype=jnp.int32)
            .at[storage_to_logical]
            .set(jnp.arange(self.particle_capacity, dtype=jnp.int32))
        )
        sorted_cells = sortable_cells[storage_to_logical]
        sorted_valid = active_valid[storage_to_logical]
        counts_with_sentinel = jnp.bincount(
            sortable_cells,
            weights=active_valid.astype(jnp.int32),
            length=self.cell_count + 1,
        ).astype(jnp.int32)
        offsets_with_sentinel = (
            jnp.cumsum(counts_with_sentinel, dtype=jnp.int32) - counts_with_sentinel
        )
        rank = (
            jnp.arange(self.particle_capacity, dtype=jnp.int32)
            - offsets_with_sentinel[sorted_cells]
        )
        cell_counts = counts_with_sentinel[: self.cell_count]
        cell_offsets = offsets_with_sentinel[: self.cell_count]
        maximum_occupancy = jnp.max(cell_counts, initial=0)
        excess = jnp.maximum(cell_counts - self.maximum_particles_per_cell, 0)
        cell_overflow_count = jnp.sum(excess, dtype=jnp.int32)
        cell_overflow = cell_overflow_count > 0
        occupant_shape = (
            self.cell_count + 1,
            self.maximum_particles_per_cell + 1,
        )
        occupant_indices = jnp.zeros(occupant_shape, dtype=jnp.int32)
        occupant_valid = jnp.zeros(occupant_shape, dtype=bool)
        accepted_occupant = sorted_valid & (rank < self.maximum_particles_per_cell)
        write_cell = jnp.where(accepted_occupant, sorted_cells, self.cell_count)
        write_rank = jnp.where(accepted_occupant, rank, self.maximum_particles_per_cell)
        occupant_indices = occupant_indices.at[write_cell, write_rank].set(
            storage_to_logical
        )
        occupant_valid = occupant_valid.at[write_cell, write_rank].set(accepted_occupant)
        occupant_indices = occupant_indices[
            : self.cell_count, : self.maximum_particles_per_cell
        ]
        occupant_valid = occupant_valid[
            : self.cell_count, : self.maximum_particles_per_cell
        ]
        safe_particle_cells = jnp.where(cell_ids >= 0, cell_ids, 0)
        neighbor_ids = self.neighbor_cells.source_indices[safe_particle_cells]
        neighbor_valid = self.neighbor_cells.valid[safe_particle_cells] & (
            cell_ids[:, None] >= 0
        )
        right_indices = occupant_indices[neighbor_ids]
        right_valid = occupant_valid[neighbor_ids]
        left_indices = jnp.broadcast_to(
            jnp.arange(self.particle_capacity, dtype=jnp.int32)[:, None, None],
            right_indices.shape,
        )
        candidate_valid = (
            active_valid[:, None, None] & neighbor_valid[:, :, None] & right_valid
        )
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
        domain_violation_count = jnp.sum(domain_violations, dtype=jnp.int32)
        return ParticleNeighborhoodState(
            pair_relation,
            box=self.box,
            storage_to_logical=storage_to_logical,
            logical_to_storage=logical_to_storage,
            cell_ids=cell_ids,
            cell_counts=cell_counts,
            cell_offsets=cell_offsets,
            candidate_pair_count=candidate_pair_count,
            pair_count=pair_count,
            maximum_cell_occupancy=maximum_occupancy,
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
