#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from ._cell_list import CellListParticleNeighborhoodPlan
from ._core import ParticleDiscretization, ParticleSetPlan
from ._pairwise import ParticleBox


class BipartiteParticleRelation(StrictModule, NonTrainableState):
    relation: EdgeRelation
    target_particle_ids: Array
    source_particle_ids: Array
    target_population_id: str = eqx.field(static=True)
    source_population_id: str = eqx.field(static=True)
    relation_schema_id: str = eqx.field(static=True)

    @property
    def target_indices(self) -> Array:
        return self.relation.target_indices

    @property
    def source_indices(self) -> Array:
        return self.relation.source_indices

    @property
    def valid(self) -> Array:
        return self.relation.valid

    @property
    def capacity(self) -> int:
        return self.relation.capacity


class BipartiteNeighborhoodState(StrictModule, NonTrainableState):
    relation: BipartiteParticleRelation
    pair_count: Array
    overflow: Array
    successful: Array


class DenseBipartiteParticleNeighborhoodPlan(StrictModule, NonTrainableState):
    maximum_pairs: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, maximum_pairs: int, /):
        maximum = int(maximum_pairs)
        if maximum <= 0:
            raise ValueError("maximum_pairs must be positive.")
        self.maximum_pairs = maximum
        self.plan_id = canonical_fingerprint(
            {"kind": "dense-bipartite-particle-neighborhood", "maximum": maximum}
        )

    def prepare(
        self,
        target: ParticleDiscretization,
        source: ParticleDiscretization,
        /,
        *,
        target_population_id: str,
        source_population_id: str,
    ) -> "PreparedDenseBipartiteParticleNeighborhood":
        return PreparedDenseBipartiteParticleNeighborhood(
            self,
            target,
            source,
            target_population_id=target_population_id,
            source_population_id=source_population_id,
        )


class PreparedDenseBipartiteParticleNeighborhood(StrictModule, NonTrainableState):
    relation: BipartiteParticleRelation
    target_capacity: int = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DenseBipartiteParticleNeighborhoodPlan,
        target: ParticleDiscretization,
        source: ParticleDiscretization,
        /,
        *,
        target_population_id: str,
        source_population_id: str,
    ):
        count = target.capacity * source.capacity
        if count > plan.maximum_pairs:
            raise ValueError("Dense bipartite relation exceeds maximum_pairs.")
        target_indices = np.repeat(np.arange(target.capacity), source.capacity)
        source_indices = np.tile(np.arange(source.capacity), target.capacity)
        target_active = np.asarray(target.active_mask)
        source_active = np.asarray(source.active_mask)
        valid = target_active[target_indices] & source_active[source_indices]
        edge = EdgeRelation(
            source_indices,
            target_indices,
            source_size=source.capacity,
            target_size=target.capacity,
            valid=valid,
        )
        schema = canonical_fingerprint(
            {
                "kind": "dense-bipartite-relation",
                "plan": plan.plan_id,
                "target": target.prepared_id,
                "source": source.prepared_id,
                "target_population": target_population_id,
                "source_population": source_population_id,
            }
        )
        self.relation = BipartiteParticleRelation(
            edge,
            target.particle_ids[target_indices],
            source.particle_ids[source_indices],
            str(target_population_id),
            str(source_population_id),
            schema,
        )
        self.target_capacity = target.capacity
        self.source_capacity = source.capacity
        self.prepared_id = schema

    def build(
        self, target_position: ArrayLike, source_position: ArrayLike, /
    ) -> BipartiteNeighborhoodState:
        target = jnp.asarray(target_position)
        source = jnp.asarray(source_position)
        if (
            target.shape[0] != self.target_capacity
            or source.shape[0] != self.source_capacity
        ):
            raise ValueError("Bipartite position capacity mismatch.")
        count = jnp.sum(self.relation.valid, dtype=jnp.int32)
        return BipartiteNeighborhoodState(
            self.relation, count, jnp.asarray(False), jnp.asarray(True)
        )


class CellListBipartiteParticleNeighborhoodPlan(StrictModule, NonTrainableState):
    search_radius: float = eqx.field(static=True)
    maximum_particles_per_cell: int = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    box: ParticleBox
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        search_radius: float,
        maximum_particles_per_cell: int,
        maximum_pairs: int,
        box: ParticleBox,
        /,
    ):
        radius = float(search_radius)
        if radius <= 0.0 or not np.isfinite(radius):
            raise ValueError("search_radius must be finite and positive.")
        if maximum_particles_per_cell <= 0 or maximum_pairs <= 0:
            raise ValueError("Bipartite cell capacities must be positive.")
        self.search_radius = radius
        self.maximum_particles_per_cell = int(maximum_particles_per_cell)
        self.maximum_pairs = int(maximum_pairs)
        self.box = box
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cell-list-bipartite-neighborhood",
                "search_radius": radius,
                "cell_capacity": int(maximum_particles_per_cell),
                "pair_capacity": int(maximum_pairs),
                "box": box.box_id,
            }
        )

    def prepare(
        self,
        target: ParticleDiscretization,
        source: ParticleDiscretization,
        /,
        *,
        target_population_id: str,
        source_population_id: str,
    ) -> "PreparedCellListBipartiteParticleNeighborhood":
        return PreparedCellListBipartiteParticleNeighborhood(
            self,
            target,
            source,
            target_population_id=target_population_id,
            source_population_id=source_population_id,
        )


class PreparedCellListBipartiteParticleNeighborhood(StrictModule, NonTrainableState):
    combined: Any
    target_ids: Array
    source_ids: Array
    target_population_id: str = eqx.field(static=True)
    source_population_id: str = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    relation_schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CellListBipartiteParticleNeighborhoodPlan,
        target: ParticleDiscretization,
        source: ParticleDiscretization,
        /,
        *,
        target_population_id: str,
        source_population_id: str,
    ):
        if target.ambient_dimension != source.ambient_dimension:
            raise ValueError("Bipartite particle dimensions must match.")
        total = target.capacity + source.capacity
        combined_particles = ParticleSetPlan(
            jnp.arange(total),
            jnp.ones((total,)),
            ambient_dimension=target.ambient_dimension,
            active_mask=jnp.concatenate((target.active_mask, source.active_mask)),
            name="bipartite-combined",
        ).prepare()
        combined_pair_capacity = total * (total - 1) // 2
        combined_plan = CellListParticleNeighborhoodPlan(
            plan.search_radius,
            plan.maximum_particles_per_cell,
            combined_pair_capacity,
            plan.box,
            maximum_candidate_slots=max(
                10_000_000,
                total * plan.maximum_particles_per_cell * 27,
            ),
        )
        self.combined = combined_plan.prepare(combined_particles)
        self.target_ids = target.particle_ids
        self.source_ids = source.particle_ids
        self.target_population_id = str(target_population_id)
        self.source_population_id = str(source_population_id)
        self.target_capacity = target.capacity
        self.source_capacity = source.capacity
        self.maximum_pairs = plan.maximum_pairs
        self.relation_schema_id = canonical_fingerprint(
            {
                "kind": "prepared-cell-list-bipartite",
                "plan": plan.plan_id,
                "target": target.prepared_id,
                "source": source.prepared_id,
                "target_population": target_population_id,
                "source_population": source_population_id,
            }
        )

    def build(
        self, target_position: ArrayLike, source_position: ArrayLike, /
    ) -> BipartiteNeighborhoodState:
        target = jnp.asarray(target_position)
        source = jnp.asarray(source_position)
        combined_state = self.combined.build(jnp.concatenate((target, source), axis=0))
        pairs = combined_state.pair_relation
        left = pairs.left_indices
        right = pairs.right_indices
        cross = (
            pairs.valid & (left < self.target_capacity) & (right >= self.target_capacity)
        )
        count = jnp.sum(cross, dtype=jnp.int32)
        overflow = count > self.maximum_pairs
        selected = jax.lax.stop_gradient(
            jnp.nonzero(cross, size=self.maximum_pairs, fill_value=0)[0]
        )
        selected_target = left[selected]
        selected_source = right[selected] - self.target_capacity
        valid = jnp.arange(self.maximum_pairs, dtype=jnp.int32) < jnp.minimum(
            count, self.maximum_pairs
        )
        edge = EdgeRelation(
            selected_source,
            selected_target,
            source_size=self.source_capacity,
            target_size=self.target_capacity,
            valid=valid,
        )
        relation = BipartiteParticleRelation(
            edge,
            self.target_ids[selected_target],
            self.source_ids[selected_source],
            self.target_population_id,
            self.source_population_id,
            self.relation_schema_id,
        )
        successful = combined_state.successful & ~overflow
        return BipartiteNeighborhoodState(
            relation, jnp.minimum(count, self.maximum_pairs), overflow, successful
        )


__all__ = [
    "BipartiteNeighborhoodState",
    "BipartiteParticleRelation",
    "CellListBipartiteParticleNeighborhoodPlan",
    "DenseBipartiteParticleNeighborhoodPlan",
    "PreparedCellListBipartiteParticleNeighborhood",
    "PreparedDenseBipartiteParticleNeighborhood",
]
