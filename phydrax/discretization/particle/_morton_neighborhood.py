#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
    resolved_identifier,
)
from ..spatial import (
    MortonAddressPlan,
    MortonRadiusRelationPlan,
    SpatialDistanceBackend,
)
from ._core import ParticleDiscretization
from ._neighborhood import (
    AbstractParticleNeighborhoodPlan,
    AbstractPreparedParticleNeighborhood,
    ParticleNeighborhoodState,
)
from ._pairwise import ParticleBox, ParticlePairRelation
from ._precision import ParticleRealization


class MortonTreeParticleNeighborhoodPlan(AbstractParticleNeighborhoodPlan):
    """Exact fixed-capacity radius pairs over compact Morton execution leaves."""

    search_radius: float = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    maximum_depth: int = eqx.field(static=True)
    maximum_nodes: int | None = eqx.field(static=True)
    maximum_leaf_occupancy: int = eqx.field(static=True)
    coarsening_factor: int = eqx.field(static=True)
    target_top_nodes: int = eqx.field(static=True)
    distance_backend: SpatialDistanceBackend = eqx.field(static=True)
    pallas_interpret: bool = eqx.field(static=True)
    box: ParticleBox
    backend: ParticleRealization = eqx.field(static=True)
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        search_radius: float,
        maximum_pairs: int,
        box: ParticleBox,
        *,
        maximum_depth: int = 21,
        maximum_nodes: int | None = None,
        maximum_leaf_occupancy: int = 32,
        coarsening_factor: int = 8,
        target_top_nodes: int = 1024,
        distance_backend: SpatialDistanceBackend = "jax",
        pallas_interpret: bool = False,
        name: str = "morton-tree-particle-neighborhood",
        plan_id: str | None = None,
    ) -> None:
        radius = float(search_radius)
        pair_capacity = int(maximum_pairs)
        depth = int(maximum_depth)
        nodes = None if maximum_nodes is None else int(maximum_nodes)
        leaf_occupancy = int(maximum_leaf_occupancy)
        coarse = int(coarsening_factor)
        top_nodes = int(target_top_nodes)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("search_radius must be finite and positive.")
        if pair_capacity <= 0:
            raise ValueError("maximum_pairs must be positive.")
        if not isinstance(box, ParticleBox):
            raise TypeError("box must be a ParticleBox.")
        if box.ambient_dimension not in (1, 2, 3):
            raise ValueError("Morton neighborhoods support dimensions 1, 2, and 3.")
        if distance_backend not in ("jax", "pallas"):
            raise ValueError("distance_backend must be 'jax' or 'pallas'.")
        key = DiscretizationKey(
            name,
            DiscretizationRole.AUXILIARY,
            domain_labels=("material_point", "pair_relation", "morton_tree"),
        )
        self.search_radius = radius
        self.maximum_pairs = pair_capacity
        self.maximum_depth = depth
        self.maximum_nodes = nodes
        self.maximum_leaf_occupancy = leaf_occupancy
        self.coarsening_factor = coarse
        self.target_top_nodes = top_nodes
        self.distance_backend = distance_backend
        self.pallas_interpret = bool(pallas_interpret)
        self.box = box
        self.backend = "morton_tree"
        self.key = key
        self.plan_id = resolved_identifier(
            "plan_id",
            plan_id,
            {
                "kind": "morton-tree-particle-neighborhood-plan",
                "search_radius": radius,
                "maximum_pairs": pair_capacity,
                "maximum_depth": depth,
                "maximum_nodes": nodes,
                "maximum_leaf_occupancy": leaf_occupancy,
                "coarsening_factor": coarse,
                "target_top_nodes": top_nodes,
                "distance_backend": distance_backend,
                "pallas_interpret": bool(pallas_interpret),
                "box": box.box_id,
                "key": key.key_id,
            },
        )

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedMortonTreeParticleNeighborhood:
        return PreparedMortonTreeParticleNeighborhood(self, particles)


class PreparedMortonTreeParticleNeighborhood(AbstractPreparedParticleNeighborhood):
    """Prepared exact Morton radius-relation backend."""

    plan: MortonTreeParticleNeighborhoodPlan
    particle_ids: Array
    active_mask: Array
    query_plan: MortonRadiusRelationPlan
    preparation: PreparationReport
    key: DiscretizationKey
    box: ParticleBox
    backend: ParticleRealization = eqx.field(static=True)
    pair_capacity: int = eqx.field(static=True)
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
        plan: MortonTreeParticleNeighborhoodPlan,
        particles: ParticleDiscretization,
        /,
    ) -> None:
        if not isinstance(plan, MortonTreeParticleNeighborhoodPlan):
            raise TypeError("plan must be a MortonTreeParticleNeighborhoodPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if plan.box.ambient_dimension != particles.ambient_dimension:
            raise ValueError("ParticleBox dimension does not match particle support.")
        address = MortonAddressPlan(
            tuple(np.asarray(plan.box.lower, dtype=float)),
            tuple(np.asarray(plan.box.upper, dtype=float)),
            plan.maximum_depth,
            periodic_axes=plan.box.periodic_axes,
        )
        query_plan = MortonRadiusRelationPlan(
            address,
            particles.capacity,
            particles.capacity,
            plan.maximum_pairs,
            inclusive=False,
            maximum_nodes=plan.maximum_nodes,
            maximum_leaf_occupancy=plan.maximum_leaf_occupancy,
            coarsening_factor=plan.coarsening_factor,
            target_top_nodes=plan.target_top_nodes,
            distance_backend=plan.distance_backend,
            pallas_interpret=plan.pallas_interpret,
        )
        relation_schema_id = canonical_fingerprint(
            {
                "kind": "morton-tree-particle-pair-relation-schema",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "source_support": particles.support.support_id,
                "pair_capacity": plan.maximum_pairs,
            }
        )
        schedule = query_plan.schedule_plan
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
                DiscretizationCapability.GEOMETRY_REFRESH,
                DiscretizationCapability.MATRIX_FREE,
            ),
            diagnostics=(
                "Morton planes and pair capacity are fixed",
                "point ordering is deterministic by stable physical ID",
                "pair selection is a frozen branchwise decision",
                "capacity and nonperiodic-domain failures fail closed",
                "public particle state remains in logical order",
            ),
            resource_counts={
                "particle_capacity": particles.capacity,
                "plane_count": schedule.plane_count,
                "leaf_capacity": schedule.plane_capacities[0],
                "node_capacity": schedule.node_capacity,
                "maximum_leaf_occupancy": schedule.maximum_leaf_occupancy,
                "pair_capacity": plan.maximum_pairs,
            },
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-morton-tree-particle-neighborhood",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "query_plan": query_plan.plan_id,
                "relation_schema": relation_schema_id,
                "preparation": preparation.report_id,
                "numeric_version": particles.numeric_version,
            }
        )
        self.plan = plan
        self.particle_ids = particles.particle_ids
        self.active_mask = particles.active_mask
        self.query_plan = query_plan
        self.preparation = preparation
        self.key = plan.key
        self.box = plan.box
        self.backend = plan.backend
        self.pair_capacity = plan.maximum_pairs
        self.particle_capacity = particles.capacity
        self.ambient_dimension = particles.ambient_dimension
        self.source_support_id = particles.support.support_id
        self.relation_schema_id = relation_schema_id
        self.particle_discretization_id = particles.prepared_id
        self.numeric_version = particles.numeric_version
        self.artifact_kind = "morton-tree-particle-neighborhood"
        self.prepared_id = prepared_id

    def build(
        self,
        position: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
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
        result = self.query_plan.query(
            value,
            value,
            self.plan.search_radius,
            source_mask=active,
            target_mask=active,
            source_stable_ids=self.particle_ids,
            target_stable_ids=self.particle_ids,
            exclude_self=True,
            pair_once=True,
        )
        relation = result.relation
        pair_relation = ParticlePairRelation(
            relation,
            self.particle_ids[relation.source_indices],
            self.particle_ids[relation.target_indices],
            source_support_id=self.source_support_id,
            target_support_id=self.source_support_id,
            same_set=True,
            unordered=True,
            relation_schema_id=self.relation_schema_id,
        )
        pair_count = jnp.sum(relation.valid, dtype=jnp.int32)
        pair_overflow_count = jnp.maximum(
            result.evidence.required_pairs - self.pair_capacity, 0
        )
        domain_violation_count = jnp.maximum(
            result.evidence.invalid_sources,
            result.evidence.invalid_targets,
        )
        cell_overflow_count = jnp.maximum(
            result.evidence.required_nodes - result.evidence.node_capacity, 0
        )
        cell_overflow = (~result.evidence.topology_successful) & (
            domain_violation_count == 0
        )
        return ParticleNeighborhoodState(
            pair_relation,
            box=self.box,
            storage_to_logical=result.storage_to_logical,
            logical_to_storage=result.logical_to_storage,
            cell_ids=result.logical_leaf_slots,
            cell_counts=result.leaf_counts,
            cell_offsets=result.leaf_offsets,
            candidate_pair_count=result.evidence.required_pairs,
            pair_count=pair_count,
            maximum_cell_occupancy=result.evidence.maximum_leaf_occupancy,
            cell_overflow=cell_overflow,
            cell_overflow_count=cell_overflow_count,
            pair_overflow=result.evidence.pair_overflow,
            pair_overflow_count=pair_overflow_count,
            domain_violation=domain_violation_count > 0,
            domain_violation_count=domain_violation_count,
            prepared_neighborhood_id=self.prepared_id,
            relation_schema_id=self.relation_schema_id,
        )


__all__ = [
    "MortonTreeParticleNeighborhoodPlan",
    "PreparedMortonTreeParticleNeighborhood",
]
