#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike as JaxArrayLike
from numpy.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._result import CellMeshingResult


class EntityLineageKind(IntEnum):
    PRESERVED = 0
    REFINED_FROM = 1
    COARSENED_INTO = 2
    SPLIT_FROM = 3
    MERGED_INTO = 4


class MeshTransitionKind(StrEnum):
    REFINE = "refine"
    COARSEN = "coarsen"
    REMESH = "remesh"
    REPAIR = "repair"
    PARTITION = "partition"


class EntityLineage(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    source_entity_set_id: str = eqx.field(static=True)
    target_entity_set_id: str = eqx.field(static=True)
    source_global_ids: Array
    target_global_ids: Array
    relation_kinds: Array
    created_target_ids: Array
    deleted_source_ids: Array
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        source_entity_set_id: str,
        target_entity_set_id: str,
        source_global_ids: ArrayLike,
        target_global_ids: ArrayLike,
        relation_kinds: ArrayLike,
        /,
        *,
        created_target_ids: ArrayLike = (),
        deleted_source_ids: ArrayLike = (),
    ):
        source_set = str(source_entity_set_id).strip()
        target_set = str(target_entity_set_id).strip()
        if not source_set or not target_set:
            raise ValueError("Entity lineage set identities must be non-empty.")
        dimension_ = int(dimension)
        if dimension_ < 0:
            raise ValueError("Entity lineage dimension must be non-negative.")
        source = np.asarray(source_global_ids, dtype=np.int64)
        target = np.asarray(target_global_ids, dtype=np.int64)
        kinds = np.asarray(relation_kinds, dtype=np.int32)
        created = np.asarray(created_target_ids, dtype=np.int64)
        deleted = np.asarray(deleted_source_ids, dtype=np.int64)
        if (
            source.ndim != 1
            or target.shape != source.shape
            or kinds.shape != source.shape
        ):
            raise ValueError("Entity lineage relations must be aligned rank-one arrays.")
        if created.ndim != 1 or deleted.ndim != 1:
            raise ValueError("Created and deleted entity IDs must be rank-one arrays.")
        if (
            np.any(source < 0)
            or np.any(target < 0)
            or np.any(created < 0)
            or np.any(deleted < 0)
        ):
            raise ValueError("Entity lineage IDs must be non-negative.")
        supported = np.asarray(
            [int(value) for value in EntityLineageKind], dtype=np.int32
        )
        if np.any(~np.isin(kinds, supported)):
            raise ValueError("Entity lineage contains an unsupported relation kind.")
        if np.intersect1d(target, created).size or np.intersect1d(source, deleted).size:
            raise ValueError(
                "Created/deleted IDs cannot also appear in lineage relations."
            )
        self.dimension = dimension_
        self.source_entity_set_id = source_set
        self.target_entity_set_id = target_set
        self.source_global_ids = jnp.asarray(source)
        self.target_global_ids = jnp.asarray(target)
        self.relation_kinds = jnp.asarray(kinds)
        self.created_target_ids = jnp.asarray(created)
        self.deleted_source_ids = jnp.asarray(deleted)
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "entity-lineage",
                "dimension": dimension_,
                "source_entity_set_id": source_set,
                "target_entity_set_id": target_set,
                "source_global_ids": array_tree_fingerprint(source),
                "target_global_ids": array_tree_fingerprint(target),
                "relation_kinds": array_tree_fingerprint(kinds),
                "created_target_ids": array_tree_fingerprint(created),
                "deleted_source_ids": array_tree_fingerprint(deleted),
            }
        )


class MeshLineage(StrictModule, NonTrainableState):
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    entities: tuple[EntityLineage, ...]
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_topology_id: str,
        target_topology_id: str,
        entities: tuple[EntityLineage, ...],
        /,
    ):
        source = str(source_topology_id).strip()
        target = str(target_topology_id).strip()
        records = tuple(entities)
        if not source or not target:
            raise ValueError("Mesh lineage topology identities must be non-empty.")
        if not records or not all(isinstance(value, EntityLineage) for value in records):
            raise ValueError("Mesh lineage requires EntityLineage records.")
        dimensions = tuple(value.dimension for value in records)
        if dimensions != tuple(sorted(set(dimensions))):
            raise ValueError(
                "Mesh lineage must contain one ordered record per dimension."
            )
        self.source_topology_id = source
        self.target_topology_id = target
        self.entities = records
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "mesh-lineage",
                "source_topology": source,
                "target_topology": target,
                "entities": [value.lineage_id for value in records],
            }
        )

    def entity_lineage(self, dimension: int, /) -> EntityLineage:
        target = int(dimension)
        for value in self.entities:
            if value.dimension == target:
                return value
        raise KeyError(f"No lineage record for dimension {target}.")


class VertexInterpolationStencil(StrictModule, NonTrainableState):
    source_entity_set_id: str = eqx.field(static=True)
    target_entity_set_id: str = eqx.field(static=True)
    target_global_ids: Array
    source_global_ids: Array
    weights: Array
    valid: Array
    preserves_constants: bool = eqx.field(static=True)
    stencil_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_entity_set_id: str,
        target_entity_set_id: str,
        target_global_ids: ArrayLike,
        source_global_ids: ArrayLike,
        weights: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        preserves_constants: bool = True,
    ):
        source_set = str(source_entity_set_id).strip()
        target_set = str(target_entity_set_id).strip()
        targets = np.asarray(target_global_ids, dtype=np.int64)
        sources = np.asarray(source_global_ids, dtype=np.int64)
        coefficients = np.asarray(weights, dtype=float)
        valid_ = np.asarray(valid, dtype=bool)
        if not source_set or not target_set:
            raise ValueError("Interpolation entity set identities must be non-empty.")
        if targets.ndim != 1 or sources.ndim != 2:
            raise ValueError("Interpolation targets and sources must be rank one/two.")
        if sources.shape != coefficients.shape or valid_.shape != sources.shape:
            raise ValueError(
                "Interpolation source, weight, and validity shapes must match."
            )
        if sources.shape[0] != targets.shape[0] or targets.size == 0:
            raise ValueError("Every interpolation target requires one stencil row.")
        if np.any(targets < 0) or np.any(sources[valid_] < 0):
            raise ValueError("Interpolation entity IDs must be non-negative.")
        if np.any(~np.isfinite(coefficients[valid_])):
            raise ValueError("Interpolation weights must be finite.")
        row_sums = np.sum(np.where(valid_, coefficients, 0.0), axis=1)
        if preserves_constants and not np.allclose(row_sums, 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError("Constant-preserving interpolation rows must sum to one.")
        self.source_entity_set_id = source_set
        self.target_entity_set_id = target_set
        self.target_global_ids = jnp.asarray(targets)
        self.source_global_ids = jnp.asarray(sources)
        self.weights = jnp.asarray(coefficients)
        self.valid = jnp.asarray(valid_)
        self.preserves_constants = bool(preserves_constants)
        self.stencil_id = canonical_fingerprint(
            {
                "kind": "vertex-interpolation-stencil",
                "source_entity_set_id": source_set,
                "target_entity_set_id": target_set,
                "target_global_ids": array_tree_fingerprint(targets),
                "source_global_ids": array_tree_fingerprint(sources),
                "weights": array_tree_fingerprint(coefficients),
                "valid": array_tree_fingerprint(valid_),
                "preserves_constants": bool(preserves_constants),
            }
        )

    def apply(
        self,
        source_global_ids: ArrayLike,
        values: JaxArrayLike,
        /,
    ) -> Array:
        identifiers = np.asarray(source_global_ids, dtype=np.int64)
        source_values = jnp.asarray(values)
        if identifiers.ndim != 1 or source_values.shape[0] != identifiers.shape[0]:
            raise ValueError("Source values must align with source_global_ids.")
        lookup = {int(identifier): index for index, identifier in enumerate(identifiers)}
        routes = np.zeros(self.source_global_ids.shape, dtype=np.int32)
        for row, column in zip(*np.nonzero(np.asarray(self.valid)), strict=True):
            identifier = int(np.asarray(self.source_global_ids)[row, column])
            if identifier not in lookup:
                raise ValueError(
                    "Interpolation stencil references an unavailable source ID."
                )
            routes[row, column] = lookup[identifier]
        gathered = source_values[jnp.asarray(routes)]
        weights = jnp.where(self.valid, self.weights, 0.0)
        return jnp.sum(
            gathered * weights[(...,) + (None,) * (source_values.ndim - 1)], axis=1
        )


class CellMeshTransition(StrictModule, NonTrainableState):
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target: CellMeshingResult
    lineage: MeshLineage
    vertex_stencil: VertexInterpolationStencil | None
    transition_kind: MeshTransitionKind = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_mesh_id: str,
        source_topology_id: str,
        target: CellMeshingResult,
        lineage: MeshLineage,
        transition_kind: MeshTransitionKind,
        /,
        *,
        vertex_stencil: VertexInterpolationStencil | None = None,
    ):
        mesh_id = str(source_mesh_id).strip()
        topology_id = str(source_topology_id).strip()
        if not mesh_id or not topology_id:
            raise ValueError("Transition source identities must be non-empty.")
        if not isinstance(target, CellMeshingResult):
            raise TypeError("target must be CellMeshingResult.")
        if not isinstance(lineage, MeshLineage):
            raise TypeError("lineage must be MeshLineage.")
        if (
            lineage.source_topology_id != topology_id
            or lineage.target_topology_id != target.mesh.topology_id
        ):
            raise ValueError(
                "Transition lineage endpoints do not match source/target topology."
            )
        if not isinstance(transition_kind, MeshTransitionKind):
            raise TypeError("transition_kind must be MeshTransitionKind.")
        if vertex_stencil is not None and not isinstance(
            vertex_stencil, VertexInterpolationStencil
        ):
            raise TypeError("vertex_stencil must be VertexInterpolationStencil or None.")
        self.source_mesh_id = mesh_id
        self.source_topology_id = topology_id
        self.target = target
        self.lineage = lineage
        self.vertex_stencil = vertex_stencil
        self.transition_kind = transition_kind
        self.transition_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-transition",
                "source_mesh": mesh_id,
                "source_topology": topology_id,
                "target": target.result_id,
                "lineage": lineage.lineage_id,
                "vertex_stencil": None
                if vertex_stencil is None
                else vertex_stencil.stencil_id,
                "transition_kind": transition_kind.value,
            }
        )


__all__ = [
    "CellMeshTransition",
    "EntityLineage",
    "EntityLineageKind",
    "MeshLineage",
    "MeshTransitionKind",
    "VertexInterpolationStencil",
]
