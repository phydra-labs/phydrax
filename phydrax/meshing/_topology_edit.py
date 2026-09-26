#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host topology-edit records shared by the native adaptation routes.

A route describes its target mesh relative to the source in global identities only:
vertices and cells by global ID, intermediate entities (edges, faces) by their sorted
vertex-global-ID keys. `assemble_topology_edit` owns the one conversion into a
canonical `CellMesh` with deterministic intermediate entity IDs, a complete
`MeshLineage`, and the sparse vertex interpolation stencil.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..discretization import CellBlock, CellMesh, PolyhedralConnectivity
from ..discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ._lineage import (
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    VertexInterpolationStencil,
)


class EntityRelations(NamedTuple):
    """Explicit lineage relations of one entity dimension.

    Vertex and cell keys are ``(r, 1)`` global IDs; edge and face keys are
    ``(r, dimension + 1)`` ascending vertex global IDs. A key present in both
    meshes also receives the identity PRESERVED relation unless an explicit
    relation already maps that source key onto the same target key.
    """

    dimension: int
    source_keys: np.ndarray
    target_keys: np.ndarray
    kinds: np.ndarray


class PrescribedEntityIds(NamedTuple):
    """Global IDs a route restores for new target edges or faces.

    Coarsening restores retired entities under their recorded IDs; ``keys`` are
    ascending vertex-global-ID rows that must not name source entities.
    """

    dimension: int
    keys: np.ndarray
    ids: np.ndarray


class SimplexTopologyEdit(NamedTuple):
    """Target simplex mesh of one native topology edit, relative to its source.

    ``block_cells[b]`` are target vertex rows of the cells kept in source block
    ``b`` (same name and kind), with ``block_cell_ids[b]`` their global IDs. The
    stencil rows interpolate every target vertex from source vertex global IDs.
    ``relations`` holds one record per dimension ``0..D`` in ascending order.
    ``prescribed_entity_ids`` fixes IDs of restored intermediate entities; other
    new intermediate entities receive fresh IDs above every source and prescribed
    ID, in sorted key order.
    """

    coordinates: np.ndarray
    vertex_global_ids: np.ndarray
    block_cells: tuple[np.ndarray, ...]
    block_cell_ids: tuple[np.ndarray, ...]
    stencil_sources: np.ndarray
    stencil_weights: np.ndarray
    stencil_valid: np.ndarray
    relations: tuple[EntityRelations, ...]
    prescribed_entity_ids: tuple[PrescribedEntityIds, ...] = ()


def entity_keys(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Keys aligned with ``mesh.entity_set(dimension)`` rows.

    Vertices and cells give ``(n, 1)`` global IDs; edges and faces of simplex
    meshes give ascending vertex-global-ID rows.
    """

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    target = int(dimension)
    if target == 0:
        return np.asarray(mesh.vertex_global_ids, dtype=np.int64)[:, None]
    if target == mesh.topological_dimension:
        return np.asarray(mesh.entity_set(target).entity_ids, dtype=np.int64)[:, None]
    connectivity = mesh.connectivity
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if target == 1 and isinstance(
        connectivity,
        (PolygonalConnectivity, TetrahedralConnectivity, PolyhedralConnectivity),
    ):
        rows = np.asarray(connectivity.edges, dtype=np.int64)
    elif target == 2 and isinstance(connectivity, TetrahedralConnectivity):
        rows = np.asarray(connectivity.faces, dtype=np.int64)
    elif target == 2 and isinstance(connectivity, PolyhedralConnectivity):
        offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        if np.any(np.diff(offsets) != 3):
            raise ValueError("Simplex face keys require triangular faces.")
        rows = np.asarray(connectivity.face_vertex_values, dtype=np.int64).reshape(
            (-1, 3)
        )
    else:
        raise ValueError(f"Entity keys of dimension {target} are unavailable.")
    return np.sort(vertex_ids[rows], axis=1)


def key_rows(table: np.ndarray, queries: np.ndarray, /) -> np.ndarray:
    """Row of each query key in ``table`` (unique rows), or -1 when absent."""

    table_ = np.asarray(table, dtype=np.int64)
    queries_ = np.asarray(queries, dtype=np.int64)
    if queries_.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if table_.shape[0] == 0:
        return np.full((queries_.shape[0],), -1, dtype=np.int64)
    width = table_.shape[1]
    if queries_.ndim != 2 or queries_.shape[1] != width:
        raise ValueError("Key queries must match the table key width.")
    combined = np.concatenate((table_, queries_), axis=0)
    _, inverse = np.unique(combined, axis=0, return_inverse=True)
    inverse = inverse.reshape((-1,))
    owner = np.full((int(np.max(inverse)) + 1,), -1, dtype=np.int64)
    owner[inverse[: table_.shape[0]]] = np.arange(table_.shape[0], dtype=np.int64)
    return owner[inverse[table_.shape[0] :]]


def _target_entity_ids(
    source: CellMesh,
    probe: CellMesh,
    dimension: int,
    prescribed: tuple[PrescribedEntityIds, ...],
    /,
) -> np.ndarray:
    """Preserved keys keep source IDs, restored keys their prescribed IDs, and the
    remaining new keys fresh IDs in sorted key order."""

    source_keys = entity_keys(source, dimension)
    source_ids = np.asarray(source.entity_set(dimension).entity_ids, dtype=np.int64)
    target_keys = entity_keys(probe, dimension)
    rows = key_rows(source_keys, target_keys)
    identifiers = np.where(rows >= 0, source_ids[np.maximum(rows, 0)], -1)
    records = tuple(value for value in prescribed if value.dimension == dimension)
    keys = np.concatenate(
        [np.asarray(value.keys, dtype=np.int64) for value in records]
        or [np.zeros((0, dimension + 1), dtype=np.int64)]
    )
    restored = np.concatenate(
        [np.asarray(value.ids, dtype=np.int64) for value in records]
        or [np.zeros((0,), dtype=np.int64)]
    )
    if (
        keys.shape != (restored.size, dimension + 1)
        or np.unique(restored).size != restored.size
        or np.any(key_rows(source_keys, keys) >= 0)
        or np.intersect1d(restored, source_ids).size
    ):
        raise ValueError("Prescribed entity IDs must name new, unique entities.")
    matched = key_rows(keys, target_keys)
    restoring = np.flatnonzero(matched >= 0)
    identifiers[restoring] = restored[matched[restoring]]
    new = np.flatnonzero(identifiers < 0)
    order = np.lexsort(target_keys[new].T[::-1])
    start = int(max(np.max(source_ids, initial=-1), np.max(restored, initial=-1))) + 1
    identifiers[new[order]] = np.arange(start, start + new.size, dtype=np.int64)
    return identifiers


def _build_mesh(
    source: CellMesh,
    edit: SimplexTopologyEdit,
    numeric_version: str,
    entity_ids: dict[int, np.ndarray] | None,
    /,
) -> CellMesh:
    blocks = []
    for block, cells, identifiers in zip(
        source.blocks, edit.block_cells, edit.block_cell_ids, strict=True
    ):
        order = np.argsort(identifiers, kind="stable")
        blocks.append(
            CellBlock(
                block.name,
                block.cell_kind,
                np.asarray(cells, dtype=np.int32)[order],
                global_ids=np.asarray(identifiers, dtype=np.int64)[order],
            )
        )
    return CellMesh(
        edit.coordinates,
        tuple(blocks),
        vertex_global_ids=edit.vertex_global_ids,
        entity_global_ids=entity_ids,
        numeric_version=numeric_version,
    )


def _entity_lineage(
    source: CellMesh,
    target: CellMesh,
    relations: EntityRelations,
    /,
) -> EntityLineage:
    dimension = relations.dimension
    source_keys = entity_keys(source, dimension)
    target_keys = entity_keys(target, dimension)
    source_ids = np.asarray(source.entity_set(dimension).entity_ids, dtype=np.int64)
    target_ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
    explicit_sources = key_rows(source_keys, relations.source_keys)
    explicit_targets = key_rows(target_keys, relations.target_keys)
    if np.any(explicit_sources < 0) or np.any(explicit_targets < 0):
        raise ValueError(
            f"Dimension-{dimension} relations reference entities outside the meshes."
        )
    related_targets = np.zeros((target_ids.size,), dtype=np.bool_)
    related_targets[explicit_targets] = True
    related_sources = np.zeros((source_ids.size,), dtype=np.bool_)
    related_sources[explicit_sources] = True
    # A key present in both meshes names the same entity; it keeps an identity
    # relation unless the route already states that identity explicitly.
    implicit = key_rows(source_keys, target_keys)
    explicit_identity = np.zeros((target_ids.size,), dtype=np.bool_)
    explicit_identity[
        explicit_targets[implicit[explicit_targets] == explicit_sources]
    ] = True
    preserved = np.flatnonzero((implicit >= 0) & ~explicit_identity)
    related_sources[implicit[preserved]] = True
    related_targets[preserved] = True
    sources = np.concatenate(
        (source_ids[explicit_sources], source_ids[implicit[preserved]])
    )
    targets = np.concatenate((target_ids[explicit_targets], target_ids[preserved]))
    kinds = np.concatenate(
        (
            np.asarray(relations.kinds, dtype=np.int32),
            np.full(preserved.shape, int(EntityLineageKind.PRESERVED), dtype=np.int32),
        )
    )
    order = np.lexsort((kinds, targets, sources))
    return EntityLineage(
        dimension,
        source.entity_set(dimension).entity_set_id,
        target.entity_set(dimension).entity_set_id,
        sources[order],
        targets[order],
        kinds[order],
        created_target_ids=np.sort(target_ids[~related_targets]),
        deleted_source_ids=np.sort(source_ids[~related_sources]),
    )


def assemble_topology_edit(
    source: CellMesh,
    edit: SimplexTopologyEdit,
    /,
    *,
    numeric_version: str,
) -> tuple[CellMesh, MeshLineage, VertexInterpolationStencil]:
    """Build the canonical target mesh, its complete lineage, and the vertex stencil."""

    if not isinstance(source, CellMesh):
        raise TypeError("source must be CellMesh.")
    if not isinstance(edit, SimplexTopologyEdit):
        raise TypeError("edit must be SimplexTopologyEdit.")
    dimension = source.topological_dimension
    if tuple(value.dimension for value in edit.relations) != tuple(range(dimension + 1)):
        raise ValueError("A topology edit needs one relation record per dimension.")
    probe = _build_mesh(source, edit, numeric_version, None)
    entity_ids = {
        degree: _target_entity_ids(source, probe, degree, edit.prescribed_entity_ids)
        for degree in range(1, dimension)
    }
    target = _build_mesh(source, edit, numeric_version, entity_ids or None)
    lineage = MeshLineage(
        source.topology_id,
        target.topology_id,
        tuple(_entity_lineage(source, target, value) for value in edit.relations),
    )
    stencil = VertexInterpolationStencil(
        source.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        edit.vertex_global_ids,
        edit.stencil_sources,
        edit.stencil_weights,
        edit.stencil_valid,
        preserves_constants=True,
    )
    return target, lineage, stencil


__all__ = [
    "EntityRelations",
    "SimplexTopologyEdit",
    "PrescribedEntityIds",
    "assemble_topology_edit",
    "entity_keys",
    "key_rows",
]
