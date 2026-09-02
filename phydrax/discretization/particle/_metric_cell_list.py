#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...sparse import EdgeRelation
from .._core import DiscretizationKey, PreparationReport
from .._periodic_cell import PeriodicCell
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
from ._pairwise import ParticleBox, ParticlePairRelation
from ._precision import ParticleRealization


class MetricCellListParticleNeighborhoodPlan(AbstractParticleNeighborhoodPlan):
    """Conservative fractional cell list filtered by physical triclinic distance."""

    search_radius: float = eqx.field(static=True)
    fractional_search_radius: float = eqx.field(static=True)
    maximum_particles_per_cell: int = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    maximum_candidate_slots: int = eqx.field(static=True)
    box: PeriodicCell
    fractional_base: CellListParticleNeighborhoodPlan
    backend: ParticleRealization = eqx.field(static=True)
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        search_radius: float,
        maximum_particles_per_cell: int,
        maximum_pairs: int,
        cell: PeriodicCell,
        /,
        *,
        maximum_candidate_slots: int = 10_000_000,
        name: str = "metric-cell-list-particle-neighborhood",
        plan_id: str | None = None,
    ):
        radius = float(search_radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("search_radius must be finite and positive.")
        if not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell.")
        cell.require_unique_image(radius)
        singular_values = np.linalg.svd(np.asarray(cell.vectors), compute_uv=False)
        fractional_radius = radius / float(singular_values[-1])
        unit_box = ParticleBox(
            np.zeros((cell.ambient_dimension,), dtype=float),
            np.ones((cell.ambient_dimension,), dtype=float),
            periodic_axes=cell.periodic_axes,
        )
        base = CellListParticleNeighborhoodPlan(
            fractional_radius,
            maximum_particles_per_cell,
            maximum_pairs,
            unit_box,
            maximum_candidate_slots=maximum_candidate_slots,
            name=f"{name}-fractional-authority",
        )
        identifier = canonical_fingerprint(
            {
                "kind": "metric-cell-list-neighborhood-plan",
                "search_radius": radius,
                "fractional_search_radius": fractional_radius,
                "cell": cell.cell_id,
                "base": base.plan_id,
            }
        )
        self.search_radius = radius
        self.fractional_search_radius = fractional_radius
        self.maximum_particles_per_cell = int(maximum_particles_per_cell)
        self.maximum_pairs = int(maximum_pairs)
        self.maximum_candidate_slots = int(maximum_candidate_slots)
        self.box = cell
        self.fractional_base = base
        self.backend = "cell_edge_list"
        self.key = base.key
        self.plan_id = identifier if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be non-empty.")

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> "PreparedMetricCellListParticleNeighborhood":
        return PreparedMetricCellListParticleNeighborhood(self, particles)


class PreparedMetricCellListParticleNeighborhood(AbstractPreparedParticleNeighborhood):
    plan: MetricCellListParticleNeighborhoodPlan
    base: PreparedCellListParticleNeighborhood
    box: PeriodicCell
    key: DiscretizationKey
    backend: ParticleRealization = eqx.field(static=True)
    pair_capacity: int = eqx.field(static=True)
    particle_discretization_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)
    artifact_kind: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MetricCellListParticleNeighborhoodPlan,
        particles: ParticleDiscretization,
        /,
    ):
        if not isinstance(plan, MetricCellListParticleNeighborhoodPlan):
            raise TypeError("plan must be MetricCellListParticleNeighborhoodPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be ParticleDiscretization.")
        if particles.ambient_dimension != plan.box.ambient_dimension:
            raise ValueError("Particle support and metric cell dimensions differ.")
        base = plan.fractional_base.prepare(particles)
        preparation = PreparationReport(
            capabilities=base.preparation.capabilities,
            diagnostics=base.preparation.diagnostics
            + (
                "fractional routes conservatively cover physical triclinic distance",
                "physical metric filtering never repairs an authority overflow",
            ),
            resource_counts=dict(base.preparation.resource_counts),
        )
        self.plan = plan
        self.base = base
        self.box = plan.box
        self.key = plan.key
        self.backend = plan.backend
        self.pair_capacity = base.pair_capacity
        self.particle_discretization_id = particles.prepared_id
        self.numeric_version = particles.numeric_version
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-metric-cell-list-neighborhood",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "base": base.prepared_id,
                "preparation": preparation.report_id,
            }
        )
        self.artifact_kind = "metric-cell-list-particle-neighborhood"

    def build(
        self, positions: ArrayLike, /, *, active_mask: ArrayLike | None = None
    ) -> ParticleNeighborhoodState:
        value = jnp.asarray(positions)
        fractional = self.box.fractional(value)
        fractional = jnp.where(
            self.box.periodic_mask,
            fractional - jnp.floor(fractional),
            fractional,
        )
        authority = self.base.build(fractional, active_mask=active_mask)
        pairs = authority.pair_relation
        displacement = value[pairs.left_indices] - value[pairs.right_indices]
        displacement = self.box.minimum_image(displacement)
        distance_squared = jnp.sum(displacement * displacement, axis=-1)
        physical_valid = pairs.valid & (distance_squared < self.plan.search_radius**2)
        relation = EdgeRelation(
            pairs.left_indices,
            pairs.right_indices,
            source_size=pairs.relation.source_size,
            target_size=pairs.relation.target_size,
            valid=physical_valid,
        )
        schema_id = canonical_fingerprint(
            {
                "kind": "metric-cell-list-pair-schema",
                "prepared": self.prepared_id,
                "base_schema": pairs.relation_schema_id,
            }
        )
        pair_relation = ParticlePairRelation(
            relation,
            pairs.left_particle_ids,
            pairs.right_particle_ids,
            source_support_id=pairs.source_support_id,
            target_support_id=pairs.target_support_id,
            same_set=True,
            unordered=True,
            relation_schema_id=schema_id,
        )
        pair_count = jnp.sum(physical_valid, dtype=jnp.int32)
        return ParticleNeighborhoodState(
            pair_relation,
            box=self.box,
            storage_to_logical=authority.storage_to_logical,
            logical_to_storage=authority.logical_to_storage,
            cell_ids=authority.cell_ids,
            cell_counts=authority.cell_counts,
            cell_offsets=authority.cell_offsets,
            candidate_pair_count=authority.candidate_pair_count,
            pair_count=pair_count,
            maximum_cell_occupancy=authority.maximum_cell_occupancy,
            cell_overflow=authority.cell_overflow,
            cell_overflow_count=authority.cell_overflow_count,
            pair_overflow=authority.pair_overflow,
            pair_overflow_count=authority.pair_overflow_count,
            domain_violation=authority.domain_violation,
            domain_violation_count=authority.domain_violation_count,
            prepared_neighborhood_id=self.prepared_id,
            relation_schema_id=schema_id,
        )


__all__ = [
    "MetricCellListParticleNeighborhoodPlan",
    "PreparedMetricCellListParticleNeighborhood",
]
