#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from ._assembly import ParticlePopulation
from ._bipartite_neighborhood import (
    BipartiteNeighborhoodState,
    BipartiteParticleRelation,
)
from ._pairwise import ParticleBox


class ParticleSearchKey(StrictModule, NonTrainableState):
    target_population_id: str = eqx.field(static=True)
    source_population_id: str = eqx.field(static=True)
    search_radius: float = eqx.field(static=True)
    key_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_population_id: str,
        source_population_id: str,
        search_radius: float,
        /,
    ):
        radius = float(search_radius)
        if radius <= 0.0 or not np.isfinite(radius):
            raise ValueError("search_radius must be finite and positive.")
        self.target_population_id = str(target_population_id)
        self.source_population_id = str(source_population_id)
        self.search_radius = radius
        self.key_id = canonical_fingerprint(
            {
                "kind": "particle-search-key",
                "target": target_population_id,
                "source": source_population_id,
                "search_radius": radius,
            }
        )


class PopulationCellView(StrictModule, NonTrainableState):
    cell_ids: Array
    storage_to_logical: Array
    logical_to_storage: Array
    cell_counts: Array
    cell_offsets: Array
    cell_particles: Array
    cell_particle_valid: Array
    maximum_occupancy: Array
    overflow: Array
    domain_violation: Array


class MultiPopulationCellState(StrictModule, NonTrainableState):
    populations: tuple[PopulationCellView, ...]
    successful: Array
    state_id: str = eqx.field(static=True)


class MultiPopulationCellPlan(StrictModule, NonTrainableState):
    box: ParticleBox
    cell_size: float = eqx.field(static=True)
    maximum_particles_per_cell: tuple[int, ...] = eqx.field(static=True)
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        box: ParticleBox,
        cell_size: float,
        maximum_particles_per_cell: Sequence[int],
        /,
    ):
        size = float(cell_size)
        capacities = tuple(int(value) for value in maximum_particles_per_cell)
        if size <= 0.0 or not capacities or any(value <= 0 for value in capacities):
            raise ValueError("Multi-population cell parameters are invalid.")
        lengths = np.asarray(box.lengths, dtype=float)
        shape = tuple(max(int(np.floor(length / size)), 1) for length in lengths)
        self.box = box
        self.cell_size = size
        self.maximum_particles_per_cell = capacities
        self.cell_shape = shape
        self.cell_count = prod(shape)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multi-population-cell-plan",
                "box": box.box_id,
                "cell_size": size,
                "capacities": list(capacities),
                "cell_shape": list(shape),
            }
        )

    def prepare(
        self, populations: Sequence[ParticlePopulation], /
    ) -> "PreparedMultiPopulationCells":
        return PreparedMultiPopulationCells(self, populations)


class PreparedMultiPopulationCells(StrictModule, NonTrainableState):
    plan: MultiPopulationCellPlan
    populations: tuple[ParticlePopulation, ...]
    widths: Array
    strides: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MultiPopulationCellPlan,
        populations: Sequence[ParticlePopulation],
        /,
    ):
        values = tuple(populations)
        if len(values) != len(plan.maximum_particles_per_cell):
            raise ValueError("Cell capacity count must match populations.")
        if any(
            population.particles.ambient_dimension != plan.box.ambient_dimension
            for population in values
        ):
            raise ValueError("Population dimensions must match the cell box.")
        strides = tuple(
            prod(plan.cell_shape[axis + 1 :]) for axis in range(len(plan.cell_shape))
        )
        self.plan = plan
        self.populations = values
        self.widths = plan.box.lengths / jnp.asarray(plan.cell_shape)
        self.strides = jnp.asarray(strides, dtype=jnp.int32)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-multi-population-cells",
                "plan": plan.plan_id,
                "populations": [population.population_id for population in values],
            }
        )

    def _build_population(
        self, population_index: int, position: Array, /
    ) -> PopulationCellView:
        population = self.populations[population_index]
        particles = population.particles
        capacity = self.plan.maximum_particles_per_cell[population_index]
        finite = jnp.all(jnp.isfinite(position), axis=-1)
        safe = jnp.where(finite[:, None], position, self.plan.box.lower)
        relative = (safe - self.plan.box.lower) / self.widths
        coordinates = jax.lax.stop_gradient(jnp.floor(relative).astype(jnp.int32))
        domain_valid = finite
        resolved = []
        for axis, count in enumerate(self.plan.cell_shape):
            coordinate = coordinates[:, axis]
            if self.plan.box.periodic_axes[axis]:
                coordinate = jnp.mod(coordinate, count)
            else:
                valid = (safe[:, axis] >= self.plan.box.lower[axis]) & (
                    safe[:, axis] < self.plan.box.upper[axis]
                )
                domain_valid = domain_valid & valid
                coordinate = jnp.clip(coordinate, 0, count - 1)
            resolved.append(coordinate)
        cell = jnp.sum(jnp.stack(resolved, axis=-1) * self.strides, axis=-1)
        active_valid = particles.active_mask & domain_valid
        sentinel = self.plan.cell_count
        sortable = jnp.where(active_valid, cell, sentinel)
        order = jax.lax.stop_gradient(
            jnp.lexsort((particles.particle_ids, sortable)).astype(jnp.int32)
        )
        inverse = (
            jnp.zeros((particles.capacity,), dtype=jnp.int32)
            .at[order]
            .set(jnp.arange(particles.capacity, dtype=jnp.int32))
        )
        sorted_cell = sortable[order]
        sorted_valid = active_valid[order]
        counts_all = jnp.bincount(
            sortable,
            weights=active_valid.astype(jnp.int32),
            length=self.plan.cell_count + 1,
        ).astype(jnp.int32)
        offsets_all = jnp.cumsum(counts_all) - counts_all
        rank = jnp.arange(particles.capacity, dtype=jnp.int32) - offsets_all[sorted_cell]
        counts = counts_all[: self.plan.cell_count]
        offsets = offsets_all[: self.plan.cell_count]
        accepted = sorted_valid & (rank < capacity)
        table = jnp.zeros((self.plan.cell_count + 1, capacity + 1), dtype=jnp.int32)
        table_valid = jnp.zeros_like(table, dtype=bool)
        write_cell = jnp.where(accepted, sorted_cell, self.plan.cell_count)
        write_rank = jnp.where(accepted, rank, capacity)
        table = table.at[write_cell, write_rank].set(order)
        table_valid = table_valid.at[write_cell, write_rank].set(accepted)
        overflow = jnp.any(counts > capacity)
        violation = jnp.any(particles.active_mask & ~domain_valid)
        return PopulationCellView(
            jnp.where(active_valid, cell, -1),
            order,
            inverse,
            counts,
            offsets,
            table[: self.plan.cell_count, :capacity],
            table_valid[: self.plan.cell_count, :capacity],
            jnp.max(counts, initial=0),
            overflow,
            violation,
        )

    def build(self, positions: Sequence[ArrayLike], /) -> MultiPopulationCellState:
        values = tuple(jnp.asarray(position) for position in positions)
        if len(values) != len(self.populations):
            raise ValueError("Position count must match populations.")
        views = tuple(
            self._build_population(index, position)
            for index, position in enumerate(values)
        )
        successful = jnp.all(
            jnp.stack(tuple(~(view.overflow | view.domain_violation) for view in views))
        )
        return MultiPopulationCellState(views, successful, self.prepared_id)

    def bipartite_relation(
        self,
        state: MultiPopulationCellState,
        positions: Sequence[ArrayLike],
        key: ParticleSearchKey,
        maximum_pairs: int,
        /,
    ) -> BipartiteNeighborhoodState:
        population_ids = tuple(
            population.population_id for population in self.populations
        )
        target_index = population_ids.index(key.target_population_id)
        source_index = population_ids.index(key.source_population_id)
        target_population = self.populations[target_index]
        source_population = self.populations[source_index]
        target_view = state.populations[target_index]
        source_view = state.populations[source_index]
        target_position = jnp.asarray(positions[target_index])
        source_position = jnp.asarray(positions[source_index])
        radius_cells = tuple(
            int(np.ceil(key.search_radius / float(width)))
            for width in np.asarray(self.widths)
        )
        offsets = np.asarray(
            list(np.ndindex(tuple(2 * radius + 1 for radius in radius_cells))),
            dtype=np.int32,
        ) - np.asarray(radius_cells, dtype=np.int32)
        target_cell = jnp.where(target_view.cell_ids >= 0, target_view.cell_ids, 0)
        coordinates = []
        for axis, stride in enumerate(np.asarray(self.strides)):
            coordinates.append((target_cell // stride) % self.plan.cell_shape[axis])
        base = jnp.stack(coordinates, axis=-1)
        candidate_coordinates = base[:, None, :] + jnp.asarray(offsets)[None, :, :]
        neighbor_valid = jnp.ones(candidate_coordinates.shape[:-1], dtype=bool)
        resolved = []
        for axis, count in enumerate(self.plan.cell_shape):
            coordinate = candidate_coordinates[..., axis]
            if self.plan.box.periodic_axes[axis]:
                coordinate = jnp.mod(coordinate, count)
            else:
                valid = (coordinate >= 0) & (coordinate < count)
                neighbor_valid = neighbor_valid & valid
                coordinate = jnp.clip(coordinate, 0, count - 1)
            resolved.append(coordinate)
        neighbor_cell = jnp.sum(jnp.stack(resolved, axis=-1) * self.strides, axis=-1)
        source_candidates = source_view.cell_particles[neighbor_cell]
        source_valid = source_view.cell_particle_valid[neighbor_cell]
        target_candidates = jnp.broadcast_to(
            jnp.arange(target_population.particles.capacity, dtype=jnp.int32)[
                :, None, None
            ],
            source_candidates.shape,
        )
        valid = (
            target_population.particles.active_mask[:, None, None]
            & neighbor_valid[:, :, None]
            & source_valid
        )
        displacement = (
            target_position[target_candidates] - source_position[source_candidates]
        )
        displacement = self.plan.box.minimum_image(displacement)
        distance2 = jnp.sum(displacement * displacement, axis=-1)
        valid = valid & (distance2 < key.search_radius**2)
        flat_valid = valid.reshape((-1,))
        count = jnp.sum(flat_valid, dtype=jnp.int32)
        selected = jax.lax.stop_gradient(
            jnp.nonzero(flat_valid, size=int(maximum_pairs), fill_value=0)[0]
        )
        target_selected = target_candidates.reshape((-1,))[selected]
        source_selected = source_candidates.reshape((-1,))[selected]
        route_valid = jnp.arange(int(maximum_pairs)) < jnp.minimum(
            count, int(maximum_pairs)
        )
        edge = EdgeRelation(
            source_selected,
            target_selected,
            source_size=source_population.particles.capacity,
            target_size=target_population.particles.capacity,
            valid=route_valid,
        )
        schema = canonical_fingerprint(
            {
                "kind": "native-bipartite-relation",
                "prepared": self.prepared_id,
                "search": key.key_id,
                "maximum_pairs": int(maximum_pairs),
            }
        )
        relation = BipartiteParticleRelation(
            edge,
            target_population.particles.particle_ids[target_selected],
            source_population.particles.particle_ids[source_selected],
            target_population.population_id,
            source_population.population_id,
            schema,
        )
        overflow = count > int(maximum_pairs)
        return BipartiteNeighborhoodState(
            relation,
            jnp.minimum(count, int(maximum_pairs)),
            overflow,
            state.successful & ~overflow,
        )


class PreparedSearchGroup(StrictModule, NonTrainableState):
    prepared: PreparedMultiPopulationCells
    keys: tuple[ParticleSearchKey, ...]
    group_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedMultiPopulationCells,
        keys: Sequence[ParticleSearchKey],
        /,
    ):
        values = tuple(keys)
        self.prepared = prepared
        self.keys = values
        self.group_id = canonical_fingerprint(
            {
                "kind": "particle-search-group",
                "prepared": prepared.prepared_id,
                "keys": [key.key_id for key in values],
            }
        )


class FusedInteractionResult(StrictModule):
    target: Array
    source: Array
    pair_count: Array
    successful: Array


def fused_bipartite_interaction(
    relation: BipartiteNeighborhoodState,
    target_position: ArrayLike,
    source_position: ArrayLike,
    pair_function: Callable[[Array, Array, Array, Array], tuple[Array, Array]],
    /,
) -> FusedInteractionResult:
    """Reference fused interface: evaluate and reduce without exposing edge payloads."""

    target_position_ = jnp.asarray(target_position)
    source_position_ = jnp.asarray(source_position)
    routes = relation.relation
    target_index = routes.target_indices
    source_index = routes.source_indices
    target_pair, source_pair = pair_function(
        target_position_[target_index],
        source_position_[source_index],
        target_index,
        source_index,
    )
    valid = routes.valid
    target_pair = jnp.where(
        valid.reshape(valid.shape + (1,) * (target_pair.ndim - 1)), target_pair, 0.0
    )
    source_pair = jnp.where(
        valid.reshape(valid.shape + (1,) * (source_pair.ndim - 1)), source_pair, 0.0
    )
    target = (
        jnp.zeros((target_position_.shape[0],) + target_pair.shape[1:], target_pair.dtype)
        .at[target_index]
        .add(target_pair)
    )
    source = (
        jnp.zeros((source_position_.shape[0],) + source_pair.shape[1:], source_pair.dtype)
        .at[source_index]
        .add(source_pair)
    )
    return FusedInteractionResult(
        target, source, relation.pair_count, relation.successful
    )


__all__ = [
    "FusedInteractionResult",
    "MultiPopulationCellPlan",
    "MultiPopulationCellState",
    "ParticleSearchKey",
    "PopulationCellView",
    "PreparedMultiPopulationCells",
    "PreparedSearchGroup",
    "fused_bipartite_interaction",
]
