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
from ..._precision import real_precision_dtype_name
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


_SURFEL_CAPABILITIES = normalized_capabilities(
    (
        DiscretizationCapability.RECONSTRUCTION,
        DiscretizationCapability.BOUNDARY_INTEGRAL,
        DiscretizationCapability.GEOMETRY_REFRESH,
        DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        DiscretizationCapability.MATRIX_FREE,
    )
)


class SurfelSetPlan(AbstractDiscretizationPlan):
    """Stable point ownership and reference surface measure for surfels."""

    surfel_ids: Array
    reference_position: Array
    reference_surface_weight: Array
    active_mask: Array
    source_entity_ids: Array
    subsets: tuple[EntitySubset, ...]
    ambient_dimension: int = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        surfel_ids: ArrayLike,
        reference_position: ArrayLike,
        reference_surface_weight: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        source_entity_ids: ArrayLike | None = None,
        subsets: Sequence[EntitySubset] = (),
        name: str = "surfels",
        coordinate_dtype: Any = "float64",
        plan_id: str | None = None,
    ) -> None:
        ids = np.asarray(surfel_ids)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("surfel_ids must be a nonempty rank-1 integer array.")
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("surfel_ids must contain integers.")
        ids = ids.astype(np.int64, copy=False)
        if np.unique(ids).size != ids.size:
            raise ValueError("surfel_ids must be unique.")
        reference = np.asarray(reference_position)
        if reference.ndim != 2 or reference.shape[0] != ids.size:
            raise ValueError(
                "reference_position must have shape (surfel_capacity,dimension)."
            )
        dimension = int(reference.shape[1])
        if dimension not in (2, 3):
            raise ValueError("Surfels require ambient dimension two or three.")
        if not np.issubdtype(reference.dtype, np.inexact):
            reference = reference.astype(float)
        weights = np.asarray(reference_surface_weight)
        if weights.shape != ids.shape:
            raise ValueError(
                "reference_surface_weight must have the surfel-capacity shape."
            )
        if not np.issubdtype(weights.dtype, np.inexact):
            weights = weights.astype(float)
        active = (
            np.ones(ids.shape, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != ids.shape:
            raise ValueError("active_mask must have the surfel-capacity shape.")
        if not np.any(active):
            raise ValueError("At least one surfel must be active.")
        source_ids = ids if source_entity_ids is None else np.asarray(source_entity_ids)
        if source_ids.shape != ids.shape or not np.issubdtype(
            source_ids.dtype, np.integer
        ):
            raise ValueError(
                "source_entity_ids must be an integer surfel-capacity array."
            )
        source_ids = source_ids.astype(np.int64, copy=False)
        if (
            np.any(~np.isfinite(reference[active]))
            or np.any(~np.isfinite(weights[active]))
            or np.any(weights[active] <= 0.0)
        ):
            raise ValueError(
                "Active surfel reference positions and surface weights must be "
                "finite, with strictly positive weights."
            )
        dtype = real_precision_dtype_name(coordinate_dtype)
        name_value = str(name).strip()
        if not name_value:
            raise ValueError("name must be nonempty.")
        subsets_value = tuple(subsets)
        entities = EntitySet(
            name_value,
            0,
            ids,
            active_mask=active,
            subsets=subsets_value,
        )
        key = DiscretizationKey(
            name_value,
            DiscretizationRole.PHYSICAL,
            domain_labels=("surfel", "surface"),
        )
        self.surfel_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.reference_position = jnp.asarray(reference, dtype=dtype)
        self.reference_surface_weight = jnp.asarray(weights, dtype=dtype)
        self.active_mask = jnp.asarray(active, dtype=bool)
        self.source_entity_ids = jnp.asarray(source_ids, dtype=jnp.int64)
        self.subsets = subsets_value
        self.ambient_dimension = dimension
        self.coordinate_dtype = dtype
        self.key = key
        self.capabilities = _SURFEL_CAPABILITIES
        self.plan_id = resolved_identifier(
            "plan_id",
            plan_id,
            {
                "kind": "surfel-set-plan",
                "key": key.key_id,
                "entities": entities.entity_set_id,
                "arrays": array_tree_fingerprint(
                    {
                        "surfel_ids": ids,
                        "reference_position": reference,
                        "reference_surface_weight": weights,
                        "active_mask": active,
                        "source_entity_ids": source_ids,
                    }
                ),
                "ambient_dimension": dimension,
                "coordinate_dtype": dtype,
            },
        )

    def prepare(self, /, *, numeric_version: str = "0") -> PreparedSurfelDiscretization:
        return PreparedSurfelDiscretization(self, numeric_version=numeric_version)


class PreparedSurfelDiscretization(AbstractPreparedDiscretization):
    """Prepared surfel identities, point topology, and reference surface measure."""

    plan: SurfelSetPlan
    entities: EntitySet
    reference_position: Array
    reference_surface_weight: Array
    source_entity_ids: Array
    active_indices: Array
    stable_active_order: Array
    position_space: DiscreteFieldSpace
    normal_space: DiscreteFieldSpace
    tangent_axes_space: DiscreteFieldSpace
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

    def __init__(
        self,
        plan: SurfelSetPlan,
        /,
        *,
        numeric_version: str = "0",
    ) -> None:
        if not isinstance(plan, SurfelSetPlan):
            raise TypeError("plan must be a SurfelSetPlan.")
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be nonempty.")
        ids = np.asarray(plan.surfel_ids, dtype=np.int64)
        active = np.asarray(plan.active_mask, dtype=bool)
        weights = np.asarray(plan.reference_surface_weight)
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
                "kind": "surfel-geometry-layout",
                "topology": topology.topology_id,
                "capacity": int(ids.size),
                "active": int(active_indices.size),
                "ambient_dimension": plan.ambient_dimension,
                "coordinate_dtype": plan.coordinate_dtype,
            }
        )
        support = DiscreteSupport(topology, plan.ambient_dimension, layout_id)
        safe_weights = np.where(active, weights, 1.0)
        vector_weights = jnp.broadcast_to(
            jnp.asarray(safe_weights, dtype=plan.coordinate_dtype)[:, None],
            (int(ids.size), plan.ambient_dimension),
        )
        vector_layout = EntityDofLayout(
            entities.entity_set_id,
            int(ids.size),
            int(ids.size),
            component_shape=(plan.ambient_dimension,),
        )
        vector_space = ArraySpace(
            (int(ids.size), plan.ambient_dimension),
            dtype=plan.coordinate_dtype,
            pairing=DiagonalPairing(vector_weights),
        )
        position_space = DiscreteFieldSpace(
            "surfel_position",
            support.support_id,
            vector_layout,
            vector_space,
            representation="point_value",
        )
        normal_space = DiscreteFieldSpace(
            "surfel_normal",
            support.support_id,
            vector_layout,
            vector_space,
            representation="point_value",
        )
        tangent_shape = (
            int(ids.size),
            plan.ambient_dimension,
            plan.ambient_dimension - 1,
        )
        tangent_weights = jnp.broadcast_to(
            jnp.asarray(safe_weights, dtype=plan.coordinate_dtype)[:, None, None],
            tangent_shape,
        )
        tangent_layout = EntityDofLayout(
            entities.entity_set_id,
            int(ids.size),
            int(ids.size),
            component_shape=(plan.ambient_dimension, plan.ambient_dimension - 1),
        )
        tangent_axes_space = DiscreteFieldSpace(
            "surfel_tangent_axes",
            support.support_id,
            tangent_layout,
            ArraySpace(
                tangent_shape,
                dtype=plan.coordinate_dtype,
                pairing=DiagonalPairing(tangent_weights),
            ),
            representation="point_value",
        )
        measure = DiscreteMeasure(
            "surfel_reference_surface_measure",
            support.support_id,
            entities.entity_set_id,
            jnp.asarray(weights, dtype=plan.coordinate_dtype),
            active_mask=plan.active_mask,
            normalization="physical",
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "surfel ownership is a zero-dimensional point topology",
                "surface measure and reconstruction footprint remain distinct",
                "normal and tangent geometry are realization state",
                "active-mask changes are topology events",
            ),
            resource_counts={
                "surfel_capacity": int(ids.size),
                "active_surfels": int(active_indices.size),
                "ambient_dimension": plan.ambient_dimension,
            },
        )
        field_spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=support,
            field_spaces=(position_space, normal_space, tangent_axes_space),
            measures=(measure,),
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-surfel-discretization",
                "plan": plan.plan_id,
                "support": support.support_id,
                "fields": [space.field_space_id for space in field_spaces],
                "measure": measure.measure_id,
                "numeric_version": version,
            }
        )
        self.plan = plan
        self.entities = entities
        self.reference_position = plan.reference_position
        self.reference_surface_weight = plan.reference_surface_weight
        self.source_entity_ids = plan.source_entity_ids
        self.active_indices = jnp.asarray(active_indices, dtype=jnp.int32)
        self.stable_active_order = jnp.asarray(stable_order, dtype=jnp.int32)
        self.position_space = position_space
        self.normal_space = normal_space
        self.tangent_axes_space = tangent_axes_space
        self.geometry_layout_id = layout_id
        self.key = plan.key
        self.support = support
        self.field_spaces = field_spaces
        self.measures = measures
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
    def surfel_ids(self) -> Array:
        return self.entities.entity_ids

    @property
    def active_mask(self) -> Array:
        return self.entities.active_mask


__all__ = ["PreparedSurfelDiscretization", "SurfelSetPlan"]
