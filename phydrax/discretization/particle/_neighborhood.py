#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
    resolved_identifier,
)
from ._core import ParticleDiscretization
from ._pairwise import ParticleBox, ParticlePairRelation
from ._precision import ParticleRealization


class ParticleNeighborhoodState(StrictModule, NonTrainableState):
    """Fixed-shape runtime particle relation and complete capacity status."""

    pair_relation: ParticlePairRelation
    box: ParticleBox | None
    storage_to_logical: Array
    logical_to_storage: Array
    cell_ids: Array
    cell_counts: Array
    cell_offsets: Array
    candidate_pair_count: Array
    pair_count: Array
    maximum_cell_occupancy: Array
    cell_overflow: Array
    cell_overflow_count: Array
    pair_overflow: Array
    pair_overflow_count: Array
    domain_violation: Array
    domain_violation_count: Array
    prepared_neighborhood_id: str = eqx.field(static=True)
    relation_schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        pair_relation: ParticlePairRelation,
        /,
        *,
        box: ParticleBox | None,
        storage_to_logical: ArrayLike,
        logical_to_storage: ArrayLike,
        cell_ids: ArrayLike,
        cell_counts: ArrayLike,
        cell_offsets: ArrayLike,
        candidate_pair_count: ArrayLike,
        pair_count: ArrayLike,
        maximum_cell_occupancy: ArrayLike,
        cell_overflow: ArrayLike,
        cell_overflow_count: ArrayLike,
        pair_overflow: ArrayLike,
        pair_overflow_count: ArrayLike,
        domain_violation: ArrayLike,
        domain_violation_count: ArrayLike,
        prepared_neighborhood_id: str,
        relation_schema_id: str,
    ):
        if not isinstance(pair_relation, ParticlePairRelation):
            raise TypeError("pair_relation must be a ParticlePairRelation.")
        if box is not None and not isinstance(box, ParticleBox):
            raise TypeError("box must be a ParticleBox or None.")
        storage = jnp.asarray(storage_to_logical)
        logical = jnp.asarray(logical_to_storage)
        cells = jnp.asarray(cell_ids)
        if (
            storage.ndim != 1
            or logical.shape != storage.shape
            or cells.shape != storage.shape
        ):
            raise ValueError(
                "Particle permutations and cell IDs must be matching vectors."
            )
        if not jnp.issubdtype(storage.dtype, jnp.integer) or not jnp.issubdtype(
            logical.dtype, jnp.integer
        ):
            raise TypeError("Particle permutations must contain integers.")
        counts = jnp.asarray(cell_counts)
        offsets = jnp.asarray(cell_offsets)
        if counts.ndim != 1 or offsets.shape != counts.shape:
            raise ValueError("Cell counts and offsets must be matching vectors.")
        if not jnp.issubdtype(cells.dtype, jnp.integer) or not jnp.issubdtype(
            counts.dtype, jnp.integer
        ):
            raise TypeError("Cell IDs and counts must contain integers.")
        if pair_relation.relation_schema_id != str(relation_schema_id):
            raise ValueError("Pair relation and neighborhood schema IDs differ.")
        prepared_id = str(prepared_neighborhood_id)
        schema_id = str(relation_schema_id)
        if not prepared_id or not schema_id:
            raise ValueError("Neighborhood and relation schema IDs must be non-empty.")
        self.pair_relation = pair_relation
        self.box = box
        self.storage_to_logical = storage
        self.logical_to_storage = logical
        self.cell_ids = cells
        self.cell_counts = counts
        self.cell_offsets = offsets
        self.candidate_pair_count = jnp.asarray(candidate_pair_count, dtype=jnp.int32)
        self.pair_count = jnp.asarray(pair_count, dtype=jnp.int32)
        self.maximum_cell_occupancy = jnp.asarray(maximum_cell_occupancy, dtype=jnp.int32)
        self.cell_overflow = jnp.asarray(cell_overflow, dtype=bool)
        self.cell_overflow_count = jnp.asarray(cell_overflow_count, dtype=jnp.int32)
        self.pair_overflow = jnp.asarray(pair_overflow, dtype=bool)
        self.pair_overflow_count = jnp.asarray(pair_overflow_count, dtype=jnp.int32)
        self.domain_violation = jnp.asarray(domain_violation, dtype=bool)
        self.domain_violation_count = jnp.asarray(domain_violation_count, dtype=jnp.int32)
        for name, value in (
            ("candidate_pair_count", self.candidate_pair_count),
            ("pair_count", self.pair_count),
            ("maximum_cell_occupancy", self.maximum_cell_occupancy),
            ("cell_overflow", self.cell_overflow),
            ("cell_overflow_count", self.cell_overflow_count),
            ("pair_overflow", self.pair_overflow),
            ("pair_overflow_count", self.pair_overflow_count),
            ("domain_violation", self.domain_violation),
            ("domain_violation_count", self.domain_violation_count),
        ):
            if value.shape != ():
                raise ValueError(f"{name} must be scalar.")
        self.prepared_neighborhood_id = prepared_id
        self.relation_schema_id = schema_id

    @property
    def successful(self) -> Array:
        return ~(self.cell_overflow | self.pair_overflow | self.domain_violation)

    def require_success(self, value: ArrayLike, /) -> Array:
        return eqx.error_if(
            jnp.asarray(value),
            ~self.successful,
            "Particle neighborhood construction failed capacity or domain checks.",
        )


class AbstractParticleNeighborhoodPlan(StrictModule, NonTrainableState):
    """Structural plan for a geometry-dependent particle relation."""

    key: AbstractAttribute[DiscretizationKey]
    box: AbstractAttribute[ParticleBox | None]
    backend: AbstractAttribute[ParticleRealization]
    plan_id: AbstractAttribute[str]

    @abc.abstractmethod
    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> "AbstractPreparedParticleNeighborhood":
        raise NotImplementedError


class AbstractPreparedParticleNeighborhood(StrictModule, NonTrainableState):
    """Prepared fixed-shape neighborhood backend."""

    plan: AbstractAttribute[AbstractParticleNeighborhoodPlan]
    key: AbstractAttribute[DiscretizationKey]
    box: AbstractAttribute[ParticleBox | None]
    backend: AbstractAttribute[ParticleRealization]
    pair_capacity: AbstractAttribute[int]
    particle_discretization_id: AbstractAttribute[str]
    numeric_version: AbstractAttribute[str]
    preparation: AbstractAttribute[PreparationReport]
    prepared_id: AbstractAttribute[str]
    artifact_kind: AbstractAttribute[str]

    @abc.abstractmethod
    def build(
        self, position: ArrayLike, /, *, active_mask: ArrayLike | None = None
    ) -> ParticleNeighborhoodState:
        raise NotImplementedError

    @property
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id


class DenseParticleNeighborhoodPlan(AbstractParticleNeighborhoodPlan):
    """All canonical same-set pairs under an explicit allocation budget."""

    maximum_pairs: int = eqx.field(static=True)
    box: ParticleBox | None
    backend: ParticleRealization = eqx.field(static=True)
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_pairs: int,
        /,
        *,
        box: ParticleBox | None = None,
        name: str = "dense-particle-neighborhood",
        plan_id: str | None = None,
    ):
        maximum = int(maximum_pairs)
        if maximum < 0:
            raise ValueError("maximum_pairs must be non-negative.")
        if box is not None and not isinstance(box, ParticleBox):
            raise TypeError("box must be a ParticleBox or None.")
        key = DiscretizationKey(
            name,
            DiscretizationRole.AUXILIARY,
            domain_labels=("material_point", "pair_relation"),
        )
        self.maximum_pairs = maximum
        self.box = box
        self.backend = "dense_pairs"
        self.key = key
        self.plan_id = resolved_identifier(
            "plan_id",
            plan_id,
            {
                "kind": "dense-particle-neighborhood-plan",
                "maximum_pairs": maximum,
                "box": None if box is None else box.box_id,
                "key": key.key_id,
            },
        )

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> "PreparedDenseParticleNeighborhood":
        return PreparedDenseParticleNeighborhood(self, particles)


class PreparedDenseParticleNeighborhood(AbstractPreparedParticleNeighborhood):
    """Prepared canonical dense candidate relation."""

    plan: DenseParticleNeighborhoodPlan
    pair_relation: ParticlePairRelation
    preparation: PreparationReport
    key: DiscretizationKey
    box: ParticleBox | None
    backend: ParticleRealization = eqx.field(static=True)
    pair_capacity: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    particle_discretization_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    artifact_kind: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DenseParticleNeighborhoodPlan,
        particles: ParticleDiscretization,
        /,
    ):
        if not isinstance(plan, DenseParticleNeighborhoodPlan):
            raise TypeError("plan must be a DenseParticleNeighborhoodPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if (
            plan.box is not None
            and plan.box.ambient_dimension != particles.ambient_dimension
        ):
            raise ValueError("ParticleBox dimension does not match particle support.")
        capacity = particles.capacity
        pair_count = capacity * (capacity - 1) // 2
        if pair_count > plan.maximum_pairs:
            raise ValueError(
                f"Dense particle relation requires {pair_count} pairs, exceeding "
                f"maximum_pairs={plan.maximum_pairs}."
            )
        first, second = np.triu_indices(capacity, k=1)
        particle_ids = np.asarray(particles.particle_ids, dtype=np.int64)
        first_ids = particle_ids[first]
        second_ids = particle_ids[second]
        swap = first_ids > second_ids
        left = np.where(swap, second, first).astype(np.int64, copy=False)
        right = np.where(swap, first, second).astype(np.int64, copy=False)
        left_ids = particle_ids[left]
        right_ids = particle_ids[right]
        active = np.asarray(particles.active_mask, dtype=bool)
        valid = active[left] & active[right]
        relation = EdgeRelation(
            left,
            right,
            source_size=capacity,
            target_size=capacity,
            valid=valid,
        )
        pair_relation = ParticlePairRelation(
            relation,
            left_ids,
            right_ids,
            source_support_id=particles.support.support_id,
            target_support_id=particles.support.support_id,
            same_set=True,
            unordered=True,
        )
        active_pairs = int(np.count_nonzero(valid))
        preparation = PreparationReport(
            capabilities=(
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
                DiscretizationCapability.MATRIX_FREE,
            ),
            diagnostics=(
                "all candidate pairs are prepared",
                "pair endpoints use stable global IDs",
                "physical support remains a runtime mask",
            ),
            resource_counts={
                "particle_capacity": capacity,
                "pair_capacity": pair_count,
                "active_pairs": active_pairs,
                "pair_index_values": 2 * pair_count,
            },
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-dense-particle-neighborhood",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "relation_schema": pair_relation.relation_schema_id,
                "preparation": preparation.report_id,
                "numeric_version": particles.numeric_version,
            }
        )
        self.plan = plan
        self.pair_relation = pair_relation
        self.preparation = preparation
        self.key = plan.key
        self.box = plan.box
        self.backend = plan.backend
        self.pair_capacity = pair_count
        self.particle_capacity = capacity
        self.particle_discretization_id = particles.prepared_id
        self.numeric_version = particles.numeric_version
        self.artifact_kind = "dense-particle-neighborhood"
        self.prepared_id = prepared_id

    def build(
        self, position: ArrayLike, /, *, active_mask: ArrayLike | None = None
    ) -> ParticleNeighborhoodState:
        value = jnp.asarray(position)
        if value.ndim != 2 or value.shape[0] != self.particle_capacity:
            raise ValueError(
                "Particle positions must have shape (particle_capacity, dimension)."
            )
        base = self.pair_relation
        if active_mask is None:
            route_valid = base.valid
        else:
            active = jnp.asarray(active_mask, dtype=bool)
            if active.shape != (self.particle_capacity,):
                raise ValueError("active_mask must have particle-capacity shape.")
            route_valid = (
                base.valid & active[base.left_indices] & active[base.right_indices]
            )
        relation = EdgeRelation(
            base.left_indices,
            base.right_indices,
            source_size=self.particle_capacity,
            target_size=self.particle_capacity,
            valid=route_valid,
        )
        pair_relation = ParticlePairRelation(
            relation,
            base.left_particle_ids,
            base.right_particle_ids,
            source_support_id=base.source_support_id,
            target_support_id=base.target_support_id,
            same_set=True,
            unordered=True,
            relation_schema_id=base.relation_schema_id,
        )
        logical = jnp.arange(self.particle_capacity, dtype=jnp.int32)
        pair_count = jnp.sum(route_valid, dtype=jnp.int32)
        empty = jnp.zeros((0,), dtype=jnp.int32)
        zero = jnp.zeros((), dtype=jnp.int32)
        false = jnp.asarray(False)
        return ParticleNeighborhoodState(
            pair_relation,
            box=self.box,
            storage_to_logical=logical,
            logical_to_storage=logical,
            cell_ids=jnp.full((self.particle_capacity,), -1, dtype=jnp.int32),
            cell_counts=empty,
            cell_offsets=empty,
            candidate_pair_count=pair_count,
            pair_count=pair_count,
            maximum_cell_occupancy=zero,
            cell_overflow=false,
            cell_overflow_count=zero,
            pair_overflow=false,
            pair_overflow_count=zero,
            domain_violation=false,
            domain_violation_count=zero,
            prepared_neighborhood_id=self.prepared_id,
            relation_schema_id=pair_relation.relation_schema_id,
        )


__all__ = [
    "AbstractParticleNeighborhoodPlan",
    "AbstractPreparedParticleNeighborhood",
    "DenseParticleNeighborhoodPlan",
    "ParticleNeighborhoodState",
    "PreparedDenseParticleNeighborhood",
]
