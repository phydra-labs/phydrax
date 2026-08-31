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

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope, real_precision_dtype_name
from .._strict import StrictModule
from ..linalg import ArraySpace, DiagonalPairing
from ._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    normalized_capabilities,
    PreparationReport,
    resolved_identifier,
)
from ._lifecycle import (
    AbstractDiscretizationPlan,
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from ._measure import DiscreteMeasure
from ._spaces import DiscreteFieldSpace, EntityDofLayout
from ._support import DiscreteSupport
from ._topology import EntitySet, EntitySubset, PointTopology


_MARKER_CAPABILITIES = normalized_capabilities(
    (
        DiscretizationCapability.GEOMETRY_REFRESH,
        DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        DiscretizationCapability.MATRIX_FREE,
    )
)


class LagrangianMarkerSetPlan(AbstractDiscretizationPlan):
    """Stable material markers with fixed reference quadrature and topology."""

    marker_ids: Array
    reference_position: Array
    quadrature_weight: Array
    active_mask: Array
    subsets: tuple[EntitySubset, ...]
    ambient_dimension: int = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        marker_ids: ArrayLike,
        reference_position: ArrayLike,
        quadrature_weight: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        subsets: Sequence[EntitySubset] = (),
        name: str = "lagrangian-markers",
        coordinate_dtype: Any = "float64",
        plan_id: str | None = None,
    ):
        ids = np.asarray(marker_ids)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("marker_ids must be a nonempty rank-1 integer array.")
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("marker_ids must contain integers.")
        ids = ids.astype(np.int64, copy=False)
        if np.unique(ids).size != ids.size:
            raise ValueError("marker_ids must be unique.")

        reference = np.asarray(reference_position)
        if reference.ndim != 2 or reference.shape[0] != ids.size:
            raise ValueError(
                "reference_position must have shape (marker_capacity,dimension)."
            )
        dimension = int(reference.shape[1])
        if dimension not in (2, 3):
            raise ValueError("Lagrangian markers currently require dimension two or three.")
        if not np.issubdtype(reference.dtype, np.inexact):
            reference = reference.astype(float)

        weights = np.asarray(quadrature_weight)
        if weights.shape != ids.shape:
            raise ValueError("quadrature_weight must have the marker-capacity shape.")
        if not np.issubdtype(weights.dtype, np.inexact):
            weights = weights.astype(float)
        active = (
            np.ones(ids.shape, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != ids.shape:
            raise ValueError("active_mask must have the marker-capacity shape.")
        if not np.any(active):
            raise ValueError("At least one Lagrangian marker must be active.")
        if (
            np.any(~np.isfinite(reference[active]))
            or np.any(~np.isfinite(weights[active]))
            or np.any(weights[active] <= 0.0)
        ):
            raise ValueError(
                "Active marker reference positions and quadrature weights must be finite, "
                "with strictly positive weights."
            )

        dtype = real_precision_dtype_name(coordinate_dtype)
        subsets_ = tuple(subsets)
        entities = EntitySet(
            name,
            0,
            ids,
            active_mask=active,
            subsets=subsets_,
        )
        key = DiscretizationKey(
            name,
            DiscretizationRole.PHYSICAL,
            domain_labels=("lagrangian_marker",),
        )
        self.marker_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.reference_position = jnp.asarray(reference, dtype=dtype)
        self.quadrature_weight = jnp.asarray(weights, dtype=dtype)
        self.active_mask = jnp.asarray(active, dtype=bool)
        self.subsets = subsets_
        self.ambient_dimension = dimension
        self.coordinate_dtype = dtype
        self.key = key
        self.capabilities = _MARKER_CAPABILITIES
        self.plan_id = resolved_identifier(
            "plan_id",
            plan_id,
            {
                "kind": "lagrangian-marker-set-plan",
                "key": key.key_id,
                "entities": entities.entity_set_id,
                "arrays": array_tree_fingerprint(
                    {
                        "marker_ids": ids,
                        "reference_position": reference,
                        "quadrature_weight": weights,
                        "active_mask": active,
                    }
                ),
                "ambient_dimension": dimension,
                "coordinate_dtype": dtype,
            },
        )

    def prepare(
        self, /, *, numeric_version: str = "0"
    ) -> LagrangianMarkerDiscretization:
        return LagrangianMarkerDiscretization(self, numeric_version=numeric_version)


class LagrangianMarkerKinematics(StrictModule):
    """One fixed-topology marker configuration and material velocity."""

    position: Array
    velocity: Array
    markers_id: str = eqx.field(static=True)


class LagrangianMarkerDiscretization(AbstractPreparedDiscretization):
    """Prepared marker support, material measure, and compact active coordinates."""

    plan: LagrangianMarkerSetPlan
    entities: EntitySet
    reference_position: Array
    active_indices: Array
    stable_active_order: Array
    geometry_layout_id: str = eqx.field(static=True)
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    active_velocity_space: ArraySpace
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    preparation: PreparationReport
    plan_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: LagrangianMarkerSetPlan, /, *, numeric_version: str = "0"
    ):
        if not isinstance(plan, LagrangianMarkerSetPlan):
            raise TypeError("plan must be a LagrangianMarkerSetPlan.")
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be nonempty.")
        ids = np.asarray(plan.marker_ids, dtype=np.int64)
        active = np.asarray(plan.active_mask, dtype=bool)
        weights = np.asarray(plan.quadrature_weight)
        active_indices = np.flatnonzero(active).astype(np.int32)
        stable_order = np.argsort(ids[active_indices], kind="stable").astype(np.int32)
        entities = EntitySet(
            plan.key.name,
            0,
            ids,
            active_mask=active,
            subsets=plan.subsets,
        )
        topology = PointTopology(entities)
        layout_id = canonical_fingerprint(
            {
                "kind": "lagrangian-marker-geometry-layout",
                "topology": topology.topology_id,
                "capacity": int(ids.size),
                "active": int(active_indices.size),
                "ambient_dimension": plan.ambient_dimension,
                "coordinate_dtype": plan.coordinate_dtype,
            }
        )
        support = DiscreteSupport(topology, plan.ambient_dimension, layout_id)
        safe_weights = np.where(active, weights, 1.0)
        full_vector_weights = jnp.broadcast_to(
            jnp.asarray(safe_weights, dtype=plan.coordinate_dtype)[:, None],
            (int(ids.size), plan.ambient_dimension),
        )
        vector_layout = EntityDofLayout(
            entities.entity_set_id,
            int(ids.size),
            int(ids.size),
            component_shape=(plan.ambient_dimension,),
        )
        full_space = ArraySpace(
            (int(ids.size), plan.ambient_dimension),
            dtype=plan.coordinate_dtype,
            pairing=DiagonalPairing(full_vector_weights),
        )
        position_space = DiscreteFieldSpace(
            "position",
            support.support_id,
            vector_layout,
            full_space,
            representation="particle_value",
        )
        velocity_space = DiscreteFieldSpace(
            "velocity",
            support.support_id,
            vector_layout,
            full_space,
            representation="particle_value",
        )
        measure = DiscreteMeasure(
            "lagrangian_material_measure",
            support.support_id,
            entities.entity_set_id,
            jnp.asarray(weights, dtype=plan.coordinate_dtype),
            active_mask=plan.active_mask,
            normalization="physical",
        )
        active_weights = jnp.broadcast_to(
            jnp.asarray(weights[active_indices], dtype=plan.coordinate_dtype)[:, None],
            (int(active_indices.size), plan.ambient_dimension),
        )
        active_space = ArraySpace(
            (int(active_indices.size), plan.ambient_dimension),
            dtype=plan.coordinate_dtype,
            pairing=DiagonalPairing(active_weights),
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "static Lagrangian marker identity and material quadrature",
                "current marker positions and velocities remain temporal state",
                "active-mask changes are topology events",
                "KKT coordinates contain active markers only",
            ),
            resource_counts={
                "marker_capacity": int(ids.size),
                "active_markers": int(active_indices.size),
                "ambient_dimension": plan.ambient_dimension,
                "active_constraint_values": int(active_indices.size)
                * plan.ambient_dimension,
            },
        )
        field_spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=support,
            field_spaces=(position_space, velocity_space),
            measures=(measure,),
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "lagrangian-marker-discretization",
                "plan": plan.plan_id,
                "support": support.support_id,
                "fields": [position_space.field_space_id, velocity_space.field_space_id],
                "measure": measure.measure_id,
                "numeric_version": version,
            }
        )
        self.plan = plan
        self.entities = entities
        self.reference_position = plan.reference_position
        self.active_indices = jnp.asarray(active_indices, dtype=jnp.int32)
        self.stable_active_order = jnp.asarray(stable_order, dtype=jnp.int32)
        self.geometry_layout_id = layout_id
        self.key = plan.key
        self.support = support
        self.field_spaces = field_spaces
        self.measures = measures
        self.active_velocity_space = active_space
        self.capabilities = capabilities
        self.preparation = preparation
        self.plan_id = plan.plan_id
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
    def marker_ids(self) -> Array:
        return self.entities.entity_ids

    @property
    def active_mask(self) -> Array:
        return self.entities.active_mask

    @property
    def material_measure(self) -> DiscreteMeasure:
        return self.measures[0]

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

    def active_values(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values, dtype=self.reference_position.dtype)
        expected = (self.capacity, self.ambient_dimension)
        if array.shape != expected:
            raise ValueError(f"Marker values must have shape {expected}.")
        return array[self.active_indices]

    def expand_active(self, values: ArrayLike, /) -> Array:
        array = self.active_velocity_space.validate(jnp.asarray(values))
        output = jnp.zeros(
            (self.capacity, self.ambient_dimension), dtype=array.dtype
        )
        return output.at[self.active_indices].set(array)

    def kinematics(
        self, position: ArrayLike, velocity: ArrayLike, /
    ) -> LagrangianMarkerKinematics:
        position_ = jnp.asarray(position, dtype=self.reference_position.dtype)
        velocity_ = jnp.asarray(velocity, dtype=self.reference_position.dtype)
        expected = (self.capacity, self.ambient_dimension)
        if position_.shape != expected or velocity_.shape != expected:
            raise ValueError(f"Marker position and velocity must have shape {expected}.")
        active = self.active_mask[:, None]
        checked = eqx.error_if(
            position_,
            jnp.any(active & (~jnp.isfinite(position_) | ~jnp.isfinite(velocity_))),
            "Active marker kinematics must be finite.",
        )
        return LagrangianMarkerKinematics(
            jnp.where(active, checked, 0.0),
            jnp.where(active, velocity_, 0.0),
            self.prepared_id,
        )

    def validate_kinematics(
        self, kinematics: LagrangianMarkerKinematics, /
    ) -> LagrangianMarkerKinematics:
        if not isinstance(kinematics, LagrangianMarkerKinematics):
            raise TypeError("kinematics must be LagrangianMarkerKinematics.")
        if kinematics.markers_id != self.prepared_id:
            raise ValueError("Marker kinematics belongs to another discretization.")
        return self.kinematics(kinematics.position, kinematics.velocity)


__all__ = [
    "LagrangianMarkerDiscretization",
    "LagrangianMarkerKinematics",
    "LagrangianMarkerSetPlan",
]
