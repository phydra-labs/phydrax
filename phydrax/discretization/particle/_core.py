#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._precision import PrecisionEvidenceEnvelope, real_precision_dtype_name
from ...linalg import ArraySpace, DiagonalPairing
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    normalized_capabilities,
    PreparationReport,
    resolved_identifier,
)
from .._lifecycle import (
    AbstractDiscretizationPlan,
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import EntitySet, EntitySubset, PointTopology


_PARTICLE_CAPABILITIES = normalized_capabilities(
    (
        DiscretizationCapability.GEOMETRY_REFRESH,
        DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        DiscretizationCapability.MATRIX_FREE,
    )
)


class ParticleSetPlan(AbstractDiscretizationPlan):
    """Stable material particles without current-position state."""

    particle_ids: Array
    masses: Array
    active_mask: Array
    subsets: tuple[EntitySubset, ...]
    ambient_dimension: int = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_ids: ArrayLike,
        masses: ArrayLike,
        /,
        *,
        ambient_dimension: int,
        active_mask: ArrayLike | None = None,
        subsets: Sequence[EntitySubset] = (),
        name: str = "particles",
        domain_labels: Sequence[str] = ("material_point",),
        coordinate_dtype: Any = "float64",
        plan_id: str | None = None,
    ):
        ids_host = np.asarray(particle_ids)
        if ids_host.ndim != 1 or ids_host.size == 0:
            raise ValueError("particle_ids must be a non-empty rank-1 array.")
        if not np.issubdtype(ids_host.dtype, np.integer):
            raise TypeError("particle_ids must contain integers.")
        ids_host = ids_host.astype(np.int64, copy=False)
        masses_host = np.asarray(masses)
        if masses_host.shape != ids_host.shape:
            raise ValueError("masses must have the particle_ids shape.")
        if not np.issubdtype(masses_host.dtype, np.inexact):
            masses_host = masses_host.astype(float)
        active_host = (
            np.ones(ids_host.shape, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active_host.shape != ids_host.shape:
            raise ValueError("active_mask must have the particle_ids shape.")
        if not np.any(active_host):
            raise ValueError("ParticleSetPlan requires at least one active particle.")
        active_masses = masses_host[active_host]
        if np.any(~np.isfinite(active_masses)) or np.any(active_masses <= 0.0):
            raise ValueError("Active particle masses must be finite and positive.")
        dimension = int(ambient_dimension)
        if dimension <= 0:
            raise ValueError("ambient_dimension must be positive.")
        dtype = real_precision_dtype_name(coordinate_dtype)
        subsets_ = tuple(subsets)
        entities = EntitySet(
            name,
            0,
            ids_host,
            active_mask=active_host,
            subsets=subsets_,
        )
        key = DiscretizationKey(
            name,
            DiscretizationRole.PHYSICAL,
            domain_labels=tuple(str(label) for label in domain_labels),
        )
        self.particle_ids = jnp.asarray(ids_host, dtype=jnp.int64)
        self.masses = jnp.asarray(masses_host)
        self.active_mask = jnp.asarray(active_host, dtype=bool)
        self.subsets = subsets_
        self.ambient_dimension = dimension
        self.coordinate_dtype = dtype
        self.key = key
        self.capabilities = _PARTICLE_CAPABILITIES
        self.plan_id = resolved_identifier(
            "plan_id",
            plan_id,
            {
                "kind": "particle-set-plan",
                "key": key.key_id,
                "entities": entities.entity_set_id,
                "arrays": array_tree_fingerprint(
                    {
                        "particle_ids": ids_host,
                        "masses": masses_host,
                        "active_mask": active_host,
                    }
                ),
                "ambient_dimension": dimension,
                "coordinate_dtype": dtype,
            },
        )

    def prepare(self, /, *, numeric_version: str = "0") -> ParticleDiscretization:
        return ParticleDiscretization(self, numeric_version=numeric_version)


class ParticleDiscretization(AbstractPreparedDiscretization):
    """Prepared material-particle support and mass measure."""

    plan: ParticleSetPlan
    entities: EntitySet
    safe_masses: Array
    geometry_layout_id: str = eqx.field(static=True)
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    preparation: PreparationReport
    plan_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ParticleSetPlan, /, *, numeric_version: str = "0"):
        if not isinstance(plan, ParticleSetPlan):
            raise TypeError("plan must be a ParticleSetPlan.")
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        ids_host = np.asarray(plan.particle_ids, dtype=np.int64)
        masses_host = np.asarray(plan.masses)
        active_host = np.asarray(plan.active_mask, dtype=bool)
        entities = EntitySet(
            plan.key.name,
            0,
            ids_host,
            active_mask=active_host,
            subsets=plan.subsets,
        )
        topology = PointTopology(entities)
        geometry_layout_id = canonical_fingerprint(
            {
                "kind": "particle-geometry-layout",
                "topology": topology.topology_id,
                "capacity": int(ids_host.size),
                "ambient_dimension": plan.ambient_dimension,
                "coordinate_dtype": plan.coordinate_dtype,
            }
        )
        support = DiscreteSupport(
            topology,
            plan.ambient_dimension,
            geometry_layout_id,
        )
        safe_masses_host = np.where(active_host, masses_host, 1.0)
        safe_masses = jnp.asarray(safe_masses_host, dtype=plan.coordinate_dtype)
        vector_weights = jnp.broadcast_to(
            safe_masses[:, None],
            (int(ids_host.size), plan.ambient_dimension),
        )
        vector_layout = EntityDofLayout(
            entities.entity_set_id,
            int(ids_host.size),
            int(ids_host.size),
            component_shape=(plan.ambient_dimension,),
        )
        vector_space = ArraySpace(
            (int(ids_host.size), plan.ambient_dimension),
            dtype=plan.coordinate_dtype,
            pairing=DiagonalPairing(vector_weights),
        )
        position_space = DiscreteFieldSpace(
            "position",
            support.support_id,
            vector_layout,
            vector_space,
            representation="particle_value",
        )
        velocity_space = DiscreteFieldSpace(
            "velocity",
            support.support_id,
            vector_layout,
            vector_space,
            representation="particle_value",
        )
        mass_measure = DiscreteMeasure(
            "material_mass",
            support.support_id,
            entities.entity_set_id,
            jnp.asarray(masses_host, dtype=plan.coordinate_dtype),
            active_mask=plan.active_mask,
            normalization="physical",
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "static material identity",
                "current positions remain temporal state",
                "active-mask changes are topology events",
            ),
            resource_counts={
                "particle_capacity": int(ids_host.size),
                "active_particles": int(np.count_nonzero(active_host)),
                "ambient_dimension": plan.ambient_dimension,
                "coordinate_values": int(ids_host.size) * plan.ambient_dimension,
            },
        )
        field_spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=support,
            field_spaces=(position_space, velocity_space),
            measures=(mass_measure,),
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "particle-discretization",
                "plan": plan.plan_id,
                "support": support.support_id,
                "fields": [position_space.field_space_id, velocity_space.field_space_id],
                "mass_measure": mass_measure.measure_id,
                "numeric_version": version,
            }
        )
        self.plan = plan
        self.entities = entities
        self.safe_masses = safe_masses
        self.geometry_layout_id = geometry_layout_id
        self.key = plan.key
        self.support = support
        self.field_spaces = field_spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.preparation = preparation
        self.numeric_version = version
        self.prepared_id = prepared_id

    @property
    def capacity(self) -> int:
        return self.entities.count

    @property
    def active_count(self) -> int:
        return self.entities.num_active

    @property
    def ambient_dimension(self) -> int:
        return self.support.ambient_dimension

    @property
    def particle_ids(self) -> Array:
        return self.entities.entity_ids

    @property
    def active_mask(self) -> Array:
        return self.entities.active_mask

    @property
    def masses(self) -> Array:
        return self.measures[0].weights

    @property
    def position_space(self) -> DiscreteFieldSpace:
        return self.field_spaces[0]

    @property
    def velocity_space(self) -> DiscreteFieldSpace:
        return self.field_spaces[1]

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope | None:
        return None

    @property
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id


__all__ = ["ParticleDiscretization", "ParticleSetPlan"]
