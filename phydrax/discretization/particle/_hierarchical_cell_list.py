#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...sparse import EdgeRelation
from .._core import DiscretizationCapability, DiscretizationKey, PreparationReport
from ._cell_list import (
    CellListParticleNeighborhoodPlan,
    PreparedCellListParticleNeighborhood,
)
from ._core import ParticleDiscretization
from ._neighborhood import (
    AbstractParticleNeighborhoodPlan,
    AbstractPreparedParticleNeighborhood,
    ParticleNeighborhoodState,
)
from ._pairwise import particle_pair_geometry, ParticleBox, ParticlePairRelation
from ._precision import ParticleRealization


class HierarchicalRadiusParticleNeighborhoodPlan(AbstractParticleNeighborhoodPlan):
    """Radius-class reference backend with conservative cell broad phase."""

    base: CellListParticleNeighborhoodPlan
    radii: Array
    level_edges: Array
    level_ids: Array
    skin: float = eqx.field(static=True)
    box: ParticleBox
    backend: ParticleRealization = eqx.field(static=True)
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: CellListParticleNeighborhoodPlan,
        radii: ArrayLike,
        level_edges: ArrayLike,
        /,
        *,
        skin: float = 0.0,
        name: str = "hierarchical-radius-particle-neighborhood",
        plan_id: str | None = None,
    ):
        if not isinstance(base, CellListParticleNeighborhoodPlan):
            raise TypeError("base must be a CellListParticleNeighborhoodPlan.")
        radii_host = np.asarray(radii)
        edges = np.asarray(level_edges)
        skin_ = float(skin)
        if radii_host.ndim != 1 or radii_host.size == 0:
            raise ValueError("radii must be a nonempty rank-1 array.")
        if np.any(~np.isfinite(radii_host)) or np.any(radii_host <= 0.0):
            raise ValueError("radii must be finite and positive.")
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("level_edges must be a strictly increasing rank-1 array.")
        if edges[0] > np.min(radii_host) or edges[-1] < np.max(radii_host):
            raise ValueError("level_edges must cover every radius.")
        if not np.isfinite(skin_) or skin_ < 0.0:
            raise ValueError("skin must be finite and nonnegative.")
        required = 2.0 * float(np.max(radii_host)) + skin_
        if base.search_radius < required:
            raise ValueError("Base search radius must cover maximum diameter plus skin.")
        levels = np.searchsorted(edges[1:-1], radii_host, side="right").astype(np.int32)
        key = DiscretizationKey(
            name,
            base.key.role,
            domain_labels=base.key.domain_labels + ("radius_hierarchy",),
        )
        identifier = canonical_fingerprint(
            {
                "kind": "hierarchical-radius-particle-neighborhood-plan",
                "base": base.plan_id,
                "radii": array_tree_fingerprint(radii_host),
                "level_edges": edges.tolist(),
                "skin": skin_,
                "key": key.key_id,
            }
        )
        self.base = base
        self.radii = jnp.asarray(radii_host)
        self.level_edges = jnp.asarray(edges)
        self.level_ids = jnp.asarray(levels)
        self.skin = skin_
        self.box = base.box
        self.backend = base.backend
        self.key = key
        self.plan_id = identifier if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedHierarchicalRadiusParticleNeighborhood:
        return PreparedHierarchicalRadiusParticleNeighborhood(self, particles)


class PreparedHierarchicalRadiusParticleNeighborhood(
    AbstractPreparedParticleNeighborhood
):
    plan: HierarchicalRadiusParticleNeighborhoodPlan
    base: PreparedCellListParticleNeighborhood
    key: DiscretizationKey
    box: ParticleBox
    backend: ParticleRealization = eqx.field(static=True)
    pair_capacity: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    particle_discretization_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    artifact_kind: str = eqx.field(static=True)
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: HierarchicalRadiusParticleNeighborhoodPlan,
        particles: ParticleDiscretization,
        /,
    ):
        if not isinstance(plan, HierarchicalRadiusParticleNeighborhoodPlan):
            raise TypeError("plan must be a HierarchicalRadiusParticleNeighborhoodPlan.")
        if plan.radii.shape != (particles.capacity,):
            raise ValueError("Radius hierarchy must match particle capacity.")
        base = plan.base.prepare(particles)
        counts = np.bincount(
            np.asarray(plan.level_ids), minlength=int(plan.level_edges.shape[0] - 1)
        )
        preparation = PreparationReport(
            capabilities=tuple(
                set(base.preparation.capabilities)
                | {DiscretizationCapability.TOPOLOGY_REFRESH_FIXED_CAPACITY}
            ),
            diagnostics=base.preparation.diagnostics
            + (
                "immutable radius classes",
                "pair-specific diameter-plus-skin fine filtering",
                "global cell broad phase remains correctness authority",
            ),
            resource_counts={
                **dict(base.preparation.resource_counts),
                "radius_levels": int(counts.size),
                "maximum_level_population": int(np.max(counts)),
            },
        )
        self.plan = plan
        self.base = base
        self.key = plan.key
        self.box = plan.box
        self.backend = plan.backend
        self.pair_capacity = base.pair_capacity
        self.particle_capacity = particles.capacity
        self.particle_discretization_id = particles.prepared_id
        self.numeric_version = particles.numeric_version
        self.artifact_kind = "hierarchical-radius-particle-neighborhood"
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-hierarchical-radius-particle-neighborhood",
                "plan": plan.plan_id,
                "base": base.prepared_id,
                "particles": particles.prepared_id,
                "preparation": preparation.report_id,
            }
        )

    def build(self, positions: ArrayLike, /) -> ParticleNeighborhoodState:
        state = self.base.build(positions)
        pairs = state.pair_relation
        geometry = particle_pair_geometry(positions, pairs, box=state.box)
        reach = (
            self.plan.radii[pairs.left_indices]
            + self.plan.radii[pairs.right_indices]
            + self.plan.skin
        )
        valid = pairs.valid & (geometry.distance < reach)
        relation = EdgeRelation(
            pairs.left_indices,
            pairs.right_indices,
            source_size=pairs.relation.source_size,
            target_size=pairs.relation.target_size,
            valid=valid,
        )
        filtered_pairs = ParticlePairRelation(
            relation,
            pairs.left_particle_ids,
            pairs.right_particle_ids,
            source_support_id=pairs.source_support_id,
            target_support_id=pairs.target_support_id,
            same_set=pairs.same_set,
            unordered=pairs.unordered,
            relation_schema_id=pairs.relation_schema_id,
        )
        return ParticleNeighborhoodState(
            filtered_pairs,
            box=state.box,
            storage_to_logical=state.storage_to_logical,
            logical_to_storage=state.logical_to_storage,
            cell_ids=state.cell_ids,
            cell_counts=state.cell_counts,
            cell_offsets=state.cell_offsets,
            candidate_pair_count=state.candidate_pair_count,
            pair_count=jnp.sum(valid, dtype=jnp.int32),
            maximum_cell_occupancy=state.maximum_cell_occupancy,
            cell_overflow=state.cell_overflow,
            cell_overflow_count=state.cell_overflow_count,
            pair_overflow=state.pair_overflow,
            pair_overflow_count=state.pair_overflow_count,
            domain_violation=state.domain_violation,
            domain_violation_count=state.domain_violation_count,
            prepared_neighborhood_id=self.prepared_id,
            relation_schema_id=state.relation_schema_id,
        )


__all__ = [
    "HierarchicalRadiusParticleNeighborhoodPlan",
    "PreparedHierarchicalRadiusParticleNeighborhood",
]
