#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike as JaxArrayLike
from numpy.typing import ArrayLike

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    logical_array_value_collection_digest,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._cell_geometry_transfer import (
    CellGeometryTransition,
    NestedReferenceWitnesses,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization.fem import (
    FiniteElementTopologyTransfer,
    vertex_interpolation_transfer,
)
from ..sparse import RowRelation, SparseLinearMap
from ..typing import checked
from ._organization import MeshAttribute, MeshLabel, MeshPatch, MeshZone
from ._result import CellMeshingResult
from ._scope import _mesh_global_ids, _selected_ids, MeshingEntityKind, MeshingScope


class EntityLineageKind(IntEnum):
    PRESERVED = 0
    REFINED_FROM = 1
    COARSENED_INTO = 2
    SPLIT_FROM = 3
    MERGED_INTO = 4
    COLLAPSED_INTO = 5
    SWAPPED_FROM = 6
    RELOCATED = 7
    GENERATED_ON_GEOMETRY = 8
    UNKNOWN = 9


class MeshTransitionKind(StrEnum):
    REFINE = "refine"
    COARSEN = "coarsen"
    REMESH = "remesh"
    REPAIR = "repair"
    PARTITION = "partition"
    GEOMETRY_REALIZATION = "geometry_realization"


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
    ) -> None:
        source_set = str(source_entity_set_id).strip()
        target_set = str(target_entity_set_id).strip()
        if not source_set or not target_set:
            raise ValueError("Entity lineage set identities must be non-empty.")
        dimension_ = int(dimension)
        if dimension_ < 0:
            raise ValueError("Entity lineage dimension must be non-negative.")
        source = jnp.asarray(source_global_ids, dtype=jnp.int64)
        target = jnp.asarray(target_global_ids, dtype=jnp.int64)
        kinds = jnp.asarray(relation_kinds, dtype=jnp.int32)
        created = jnp.asarray(created_target_ids, dtype=jnp.int64)
        deleted = jnp.asarray(deleted_source_ids, dtype=jnp.int64)
        if (
            source.ndim != 1
            or target.shape != source.shape
            or kinds.shape != source.shape
        ):
            raise ValueError("Entity lineage relations must be aligned rank-one arrays.")
        if created.ndim != 1 or deleted.ndim != 1:
            raise ValueError("Created and deleted entity IDs must be rank-one arrays.")
        if bool(
            jax.device_get(
                jnp.any(source < 0)
                | jnp.any(target < 0)
                | jnp.any(created < 0)
                | jnp.any(deleted < 0)
            )
        ):
            raise ValueError("Entity lineage IDs must be non-negative.")
        supported = jnp.asarray(
            [int(value) for value in EntityLineageKind], dtype=jnp.int32
        )
        if bool(jax.device_get(jnp.any(~jnp.isin(kinds, supported)))):
            raise ValueError("Entity lineage contains an unsupported relation kind.")
        if bool(
            jax.device_get(
                jnp.any(jnp.isin(target, created)) | jnp.any(jnp.isin(source, deleted))
            )
        ):
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
                "relations": logical_array_value_collection_digest(
                    {
                        "source_global_ids": source,
                        "target_global_ids": target,
                        "relation_kinds": kinds,
                        "created_target_ids": created,
                        "deleted_source_ids": deleted,
                    }
                ),
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
    ) -> None:
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
    ) -> None:
        source_set = str(source_entity_set_id).strip()
        target_set = str(target_entity_set_id).strip()
        targets = np.asarray(target_global_ids, dtype=np.int64)
        sources = np.asarray(source_global_ids, dtype=np.int64)
        coefficients = np.asarray(weights, dtype=np.float64)
        valid_ = np.asarray(valid, dtype=np.bool_)
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

    def _source_routes(self, source_global_ids: ArrayLike, /) -> np.ndarray:
        """Resolve stencil source IDs to positions in one caller-supplied ID order."""

        identifiers = np.asarray(source_global_ids, dtype=np.int64)
        if identifiers.ndim != 1 or identifiers.size == 0:
            raise ValueError("source_global_ids must be one non-empty rank-1 array.")
        order = np.argsort(identifiers, kind="stable")
        ordered = identifiers[order]
        if np.any(ordered[1:] == ordered[:-1]):
            raise ValueError("source_global_ids must be unique.")
        references = np.asarray(self.source_global_ids)
        valid = np.asarray(self.valid)
        position = np.minimum(np.searchsorted(ordered, references), identifiers.size - 1)
        if np.any(valid & (ordered[position] != references)):
            raise ValueError("Interpolation stencil references an unavailable source ID.")
        return np.where(valid, order[position], 0).astype(np.int32)

    def as_transfer(
        self,
        source_global_ids: ArrayLike,
        /,
        *,
        source_topology_id: str,
        target_topology_id: str,
        preserves_linear: bool = False,
        conservative: bool = False,
        source_coordinates: ArrayLike | None = None,
        target_coordinates: ArrayLike | None = None,
        source_measures: ArrayLike | None = None,
        target_measures: ArrayLike | None = None,
    ) -> FiniteElementTopologyTransfer:
        """Resolve this stencil into a sparse transfer over one source DOF order.

        Linear-preservation and conservation claims are certified by the transfer
        owner from the supplied row-aligned coordinates and P1 DOF measures.
        """

        identifiers = np.asarray(source_global_ids, dtype=np.int64)
        return vertex_interpolation_transfer(
            self._source_routes(identifiers),
            np.asarray(self.weights),
            np.asarray(self.valid),
            source_size=identifiers.size,
            source_topology_id=source_topology_id,
            target_topology_id=target_topology_id,
            preserves_linear=preserves_linear,
            conservative=conservative,
            # ty: ignore[invalid-argument-type]
            source_coordinates=source_coordinates,
            # ty: ignore[invalid-argument-type]
            target_coordinates=target_coordinates,
            # ty: ignore[invalid-argument-type]
            source_measures=source_measures,
            # ty: ignore[invalid-argument-type]
            target_measures=target_measures,
        )

    def apply(
        self,
        source_global_ids: ArrayLike,
        values: JaxArrayLike,
        /,
    ) -> Array:
        """Interpolate source values given in ``source_global_ids`` order in one gather."""

        identifiers = np.asarray(source_global_ids, dtype=np.int64)
        source_values = jnp.asarray(values)
        routes = self._source_routes(identifiers)
        if source_values.ndim == 0 or source_values.shape[0] != identifiers.size:
            raise ValueError("Source values must align with source_global_ids.")
        relation = RowRelation(routes, source_size=identifiers.size, valid=self.valid)
        return SparseLinearMap(
            relation,
            self.weights,
            operator_id=canonical_fingerprint(
                {
                    "kind": "vertex-interpolation-stencil-map",
                    "stencil": self.stencil_id,
                    "source_global_ids": array_tree_fingerprint(identifiers),
                }
            ),
        ).mv(source_values)


def _parent_witnesses(
    target: CellMeshingResult, parents: NestedReferenceWitnesses, /
) -> tuple[np.ndarray, np.ndarray]:
    """Parent IDs and reference corners in target concatenated block order."""

    mesh = target.mesh
    ids = np.concatenate(
        [np.asarray(block.global_ids, np.int64) for block in mesh.blocks]
    )
    dimension = mesh.topological_dimension
    fine = np.asarray(parents.fine_cell_ids, dtype=np.int64)
    vertices = np.asarray(parents.fine_reference_vertices, dtype=np.float64)
    if (
        vertices.ndim != 3
        or vertices.shape[0] != fine.size
        or vertices.shape[2] != dimension
    ):
        raise ValueError("Parent witnesses need finite (C, padded corners, d) records.")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Parent reference witnesses must be finite.")
    order = np.argsort(ids, kind="stable")
    position = np.minimum(np.searchsorted(ids[order], fine), ids.size - 1)
    if np.any(ids[order][position] != fine) or np.unique(fine).size != fine.size:
        raise ValueError("Parent witnesses must name distinct target cells.")
    rows = order[position]
    parent_ids = np.full(ids.shape, -1, dtype=np.int64)
    parent_ids[rows] = np.asarray(parents.coarse_cell_ids, dtype=np.int64)
    reference = np.zeros((ids.size, vertices.shape[1], dimension), dtype=np.float64)
    reference[rows] = vertices
    return parent_ids, reference


def _coarsening_witnesses(
    target: CellMeshingResult,
    lineage: MeshLineage,
    witnesses: NestedReferenceWitnesses,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate exact fine-to-coarse charts against accepted cell lineage."""

    fine = np.asarray(witnesses.fine_cell_ids, dtype=np.int64)
    coarse = np.asarray(witnesses.coarse_cell_ids, dtype=np.int64)
    reference = np.asarray(witnesses.fine_reference_vertices, dtype=np.float64)
    dimension = target.mesh.topological_dimension
    if (
        coarse.shape != fine.shape
        or reference.ndim != 3
        or reference.shape[0] != fine.size
        or reference.shape[2] != dimension
        or np.unique(fine).size != fine.size
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError(
            "Coarsening witnesses need distinct fine cells and finite "
            "(C, padded corners, d) records."
        )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target.mesh.blocks]
    )
    if not np.all(np.isin(coarse, target_ids)):
        raise ValueError("Coarsening witnesses must name accepted target cells.")
    cells = lineage.entity_lineage(dimension)
    relations = {
        (int(source), int(successor))
        for source, successor, kind in zip(
            np.asarray(cells.source_global_ids, dtype=np.int64),
            np.asarray(cells.target_global_ids, dtype=np.int64),
            np.asarray(cells.relation_kinds, dtype=np.int32),
            strict=True,
        )
        if kind
        in (
            int(EntityLineageKind.PRESERVED),
            int(EntityLineageKind.COARSENED_INTO),
        )
    }
    if any(
        (int(source), int(successor)) not in relations
        for source, successor in zip(fine, coarse, strict=True)
    ):
        raise ValueError(
            "Coarsening witnesses must bind accepted preserved or coarsened cell lineage."
        )
    return fine, coarse, reference


class CellMeshTransition(StrictModule, NonTrainableState):
    """One accepted topology transition of a certified mesh.

    ``geometry_transition`` records how the target coordinate map was built from
    a non-affine source map (absent when both maps are affine carriers).
    ``parent_cell_ids``/``parent_reference_vertices`` are the nested witnesses in
    target concatenated block order: the source cell containing each target cell
    (``-1`` for a target assembled from several source cells) and the target
    corners in that parent's reference coordinates. They are absent for
    non-nested transitions.
    """

    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target: CellMeshingResult
    lineage: MeshLineage
    vertex_stencil: VertexInterpolationStencil | None
    geometry_transition: CellGeometryTransition | None
    parent_cell_ids: Array | None
    parent_reference_vertices: Array | None
    coarsened_cell_ids: Array | None
    coarsened_into_ids: Array | None
    coarsened_reference_vertices: Array | None
    transition_kind: MeshTransitionKind = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    @checked
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
        geometry_transition: CellGeometryTransition | None = None,
        parents: NestedReferenceWitnesses | None = None,
        coarsening: NestedReferenceWitnesses | None = None,
    ) -> None:
        mesh_id = str(source_mesh_id).strip()
        topology_id = str(source_topology_id).strip()
        if not mesh_id or not topology_id:
            raise ValueError("Transition source identities must be non-empty.")
        if (
            lineage.source_topology_id != topology_id
            or lineage.target_topology_id != target.mesh.topology_id
        ):
            raise ValueError(
                "Transition lineage endpoints do not match source/target topology."
            )
        if not isinstance(transition_kind, MeshTransitionKind):
            raise TypeError("transition_kind must be MeshTransitionKind.")
        if geometry_transition is not None:
            if (
                geometry_transition.source_topology_id != topology_id
                or geometry_transition.target_topology_id != target.mesh.topology_id
                or geometry_transition.target_geometry_id
                != cell_geometry_id(target.geometry)
            ):
                raise ValueError(
                    "The geometry transition must produce the target coordinate map."
                )
        witnesses = None if parents is None else _parent_witnesses(target, parents)
        coarsened = (
            None
            if coarsening is None or np.asarray(coarsening.fine_cell_ids).size == 0
            else _coarsening_witnesses(target, lineage, coarsening)
        )
        self.source_mesh_id = mesh_id
        self.source_topology_id = topology_id
        self.target = target
        self.lineage = lineage
        self.vertex_stencil = vertex_stencil
        self.geometry_transition = geometry_transition
        self.parent_cell_ids = None if witnesses is None else jnp.asarray(witnesses[0])
        self.parent_reference_vertices = (
            None if witnesses is None else jnp.asarray(witnesses[1])
        )
        self.coarsened_cell_ids = None if coarsened is None else jnp.asarray(coarsened[0])
        self.coarsened_into_ids = None if coarsened is None else jnp.asarray(coarsened[1])
        self.coarsened_reference_vertices = (
            None if coarsened is None else jnp.asarray(coarsened[2])
        )
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
                "geometry_transition": None
                if geometry_transition is None
                else geometry_transition.transition_id,
                "parents": None
                if witnesses is None
                else [array_tree_fingerprint(value) for value in witnesses],
                "coarsening": None
                if coarsened is None
                else [array_tree_fingerprint(value) for value in coarsened],
                "transition_kind": transition_kind.value,
            }
        )


def identity_lineage(source: CellMesh, target: CellMesh, /) -> MeshLineage:
    """PRESERVED lineage of every entity of a same-topology coordinate change."""

    if not isinstance(source, CellMesh) or not isinstance(target, CellMesh):
        raise TypeError("identity_lineage requires two CellMesh values.")
    if source.topology_id != target.topology_id:
        raise ValueError("Identity lineage requires one shared topology.")
    records = []
    for dimension in range(source.topological_dimension + 1):
        ids = _mesh_global_ids(source, dimension)
        records.append(
            EntityLineage(
                dimension,
                source.entity_set(dimension).entity_set_id,
                target.entity_set(dimension).entity_set_id,
                ids,
                ids,
                jnp.full(ids.shape, int(EntityLineageKind.PRESERVED), dtype=jnp.int32),
                created_target_ids=jnp.zeros((0,), dtype=jnp.int64),
                deleted_source_ids=jnp.zeros((0,), dtype=jnp.int64),
            )
        )
    return MeshLineage(source.topology_id, target.topology_id, tuple(records))


# Relations through which organization membership (patches/zones/labels) is inherited.
_INHERITING_KINDS = np.asarray(
    [
        EntityLineageKind.PRESERVED,
        EntityLineageKind.REFINED_FROM,
        EntityLineageKind.COARSENED_INTO,
        EntityLineageKind.MERGED_INTO,
        EntityLineageKind.COLLAPSED_INTO,
        EntityLineageKind.SWAPPED_FROM,
        EntityLineageKind.RELOCATED,
    ],
    dtype=np.int32,
)


def _inherited_ids(scope: MeshingScope, record: EntityLineage, /) -> Array:
    """Target IDs inheriting one organization scope through one lineage record.

    COLLAPSED_INTO sources are non-dominant: a target with any other inheriting
    relation takes its membership from those; otherwise the collapsed sources
    decide. Mixed membership among deciding sources is a route error.
    """

    arrays = (
        record.relation_kinds,
        record.source_global_ids,
        record.target_global_ids,
        scope.global_entity_ids,
    )
    if all(
        not isinstance(value, Tracer)
        and value.is_fully_addressable
        and isinstance(value.sharding, jax.sharding.SingleDeviceSharding)
        for value in arrays
    ):
        # Organization is immutable host preparation. Retain the collective
        # array route below when any scientific inventory is not addressable.
        kinds_host, sources_host, targets_host, members_host = (
            np.asarray(value) for value in jax.device_get(arrays)
        )
        inheriting_host = np.isin(kinds_host, _INHERITING_KINDS)
        targets_host = targets_host[inheriting_host]
        if targets_host.size == 0:
            return jnp.zeros((0,), dtype=jnp.int64)
        order_host = np.argsort(targets_host, kind="stable")
        targets_host = targets_host[order_host]
        members = np.isin(sources_host[inheriting_host][order_host], members_host)
        collapsed = (
            kinds_host[inheriting_host][order_host] == EntityLineageKind.COLLAPSED_INTO
        )
        unique, inverse = np.unique(targets_host, return_inverse=True)
        dominant = np.zeros(unique.shape, dtype=np.int64)
        np.add.at(dominant, inverse, (~collapsed).astype(np.int64))
        deciding = ~collapsed | (dominant[inverse] == 0)
        total = np.zeros(unique.shape, dtype=np.int64)
        member = np.zeros(unique.shape, dtype=np.int64)
        np.add.at(total, inverse, deciding.astype(np.int64))
        np.add.at(member, inverse, (deciding & members).astype(np.int64))
        if np.any((member > 0) & (member < total)):
            raise ValueError(
                "A topology transition merged entities of different organization "
                "membership; the region evidence is ambiguous."
            )
        return jnp.asarray(unique[(member > 0) & (member == total)], dtype=jnp.int64)

    kinds = record.relation_kinds
    inheriting = jnp.isin(kinds, jnp.asarray(_INHERITING_KINDS))
    targets = _selected_ids(record.target_global_ids, inheriting)
    sources = _selected_ids(record.source_global_ids, inheriting)
    selected_kinds = _selected_ids(kinds, inheriting)
    if targets.size == 0:
        return jnp.zeros((0,), dtype=jnp.int64)
    order = jnp.argsort(targets, stable=True)
    targets = targets[order]
    members = jnp.isin(sources[order], scope.global_entity_ids)
    collapsed = selected_kinds[order] == EntityLineageKind.COLLAPSED_INTO
    fresh = jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), targets[1:] != targets[:-1])
    )
    unique = _selected_ids(targets, fresh)
    inverse = jnp.cumsum(fresh, dtype=jnp.int32) - 1
    dominant = (
        jnp.zeros(unique.shape, dtype=jnp.int64)
        .at[inverse]
        .add((~collapsed).astype(jnp.int64))
        > 0
    )
    deciding = ~collapsed | ~dominant[inverse]
    total = (
        jnp.zeros(unique.shape, dtype=jnp.int64)
        .at[inverse]
        .add(deciding.astype(jnp.int64))
    )
    member = (
        jnp.zeros(unique.shape, dtype=jnp.int64)
        .at[inverse]
        .add((deciding & members).astype(jnp.int64))
    )
    if bool(jax.device_get(jnp.any((member > 0) & (member < total)))):
        raise ValueError(
            "A topology transition merged entities of different organization "
            "membership; the region evidence is ambiguous."
        )
    return _selected_ids(unique, (member > 0) & (member == total))


def inherit_scope(
    scope: MeshingScope, lineage: MeshLineage, target: CellMesh, name: str, /
) -> MeshingScope:
    """Rebind one mesh scope onto ``target`` through ``lineage``; ambiguity refuses."""

    dimension = scope.entity_dimension
    record = lineage.entity_lineage(dimension)
    if (
        scope.entity_kind is not MeshingEntityKind.MESH
        or scope.entity_set_id != record.source_entity_set_id
        or record.target_entity_set_id != target.entity_set(dimension).entity_set_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError(
            "Organization inheritance requires exact source and target entity bindings."
        )
    identifiers = _inherited_ids(scope, record)
    if identifiers.size == 0:
        raise ValueError(
            f"A topology transition removed every entity of organization {name!r}."
        )
    return MeshingScope(
        target.mesh_id,
        target.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        target.entity_set(dimension).entity_set_id,
        identifiers,
        local_mesh=target,
    )


def inherit_mesh_attributes(
    source: CellMeshingResult,
    target: CellMesh,
    lineage: MeshLineage,
    /,
    *,
    excluded_names: tuple[str, ...] = (),
) -> tuple[MeshAttribute, ...]:
    """Carry exact entity data through the same deciding-source organization lineage.

    Refinement copies parent values. Merged deciding sources must agree exactly;
    a numeric dtype alone does not authorize interpolation or averaging. Physical
    finite-element coefficients use their separately declared field transfer.
    Distributed publication consumes its globally prepared attribute banks.
    """
    if source.mesh.storage is not None or target.storage is not None:
        raise ValueError(
            "Distributed attributes require their accepted global numerical publication receipts."
        )
    if (
        lineage.source_topology_id != source.mesh.topology_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError(
            "Attribute inheritance requires the actual source and target topology lineage."
        )
    result = []
    for attribute in source.attributes:
        if attribute.name in excluded_names:
            continue
        scope = inherit_scope(attribute.scope, lineage, target, attribute.name)
        record = lineage.entity_lineage(scope.entity_dimension)
        source_ids = np.asarray(attribute.scope.global_entity_ids, dtype=np.int64)
        values = np.asarray(attribute.global_values)
        route_sources = np.asarray(record.source_global_ids, dtype=np.int64)
        route_targets = np.asarray(record.target_global_ids, dtype=np.int64)
        kinds = np.asarray(record.relation_kinds, dtype=np.int32)
        inheriting = np.isin(kinds, _INHERITING_KINDS)
        target_values = np.empty(
            (scope.global_entity_ids.shape[0], *attribute.component_shape),
            dtype=values.dtype,
        )
        for row, identifier in enumerate(
            np.asarray(scope.global_entity_ids, dtype=np.int64)
        ):
            routes = inheriting & (route_targets == identifier)
            dominant = routes & (kinds != EntityLineageKind.COLLAPSED_INTO)
            deciding = dominant if np.any(dominant) else routes
            selected = route_sources[deciding]
            positions = np.searchsorted(source_ids, selected)
            if (
                selected.size == 0
                or np.any(positions >= source_ids.size)
                or not np.array_equal(
                    source_ids[positions],
                    selected,
                )
            ):
                raise ValueError(
                    "An inherited attribute lacks exact deciding scientific source entities."
                )
            candidates = values[positions]
            if not np.all(candidates == candidates[0]):
                raise ValueError(
                    "A topology transition merged different scientific attribute values."
                )
            target_values[row] = candidates[0]
        result.append(
            MeshAttribute(
                attribute.name,
                attribute.role,
                scope,
                target_values,
                unit=attribute.unit,
            )
        )
    return tuple(result)


def inherit_mesh_organization(
    source: CellMeshingResult, target: CellMesh, lineage: MeshLineage, /
) -> tuple[tuple[MeshPatch, ...], tuple[MeshZone, ...], tuple[MeshLabel, ...]]:
    """Rebind every patch, zone, and label of ``source`` onto ``target``.

    Membership follows the lineage relations; region identity (material and
    region role), zone adjacency, and oriented source-region sides are kept.
    Ambiguous membership refuses the whole remap, so no partial region evidence
    is published.
    """

    if lineage.target_topology_id != target.topology_id or (
        lineage.source_topology_id != source.mesh.topology_id
    ):
        raise ValueError("The lineage must join the source and target topologies.")
    for record in (*source.patches, *source.zones, *source.labels):
        scope = record.scope
        if (
            scope.source_id != source.mesh.mesh_id
            or scope.source_revision != source.mesh.numeric_version
            or scope.entity_kind is not MeshingEntityKind.MESH
        ):
            raise ValueError(
                "Organization inheritance requires the exact source mesh revision."
            )
    zone_ids = {}
    zones = []
    for zone in source.zones:
        inherited = MeshZone(
            zone.name,
            zone.role,
            inherit_scope(zone.scope, lineage, target, zone.name),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        zone_ids[zone.zone_id] = inherited.zone_id
        zones.append(inherited)
    patches = tuple(
        MeshPatch(
            patch.name,
            inherit_scope(patch.scope, lineage, target, patch.name),
            connected=patch.connected,
            adjacent_zone_ids=tuple(zone_ids[value] for value in patch.adjacent_zone_ids),
            source_adjacent_region_ids=patch.source_adjacent_region_ids,
        )
        for patch in source.patches
    )
    labels = tuple(
        MeshLabel(label.name, inherit_scope(label.scope, lineage, target, label.name))
        for label in source.labels
    )
    return patches, tuple(zones), labels


__all__ = [
    "CellMeshTransition",
    "EntityLineage",
    "EntityLineageKind",
    "MeshLineage",
    "MeshTransitionKind",
    "VertexInterpolationStencil",
    "identity_lineage",
    "inherit_mesh_organization",
    "inherit_mesh_attributes",
]
