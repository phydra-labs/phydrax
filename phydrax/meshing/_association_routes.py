#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host topology and lineage invariants shared by association routes."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.meshing._entity_keys as entity_keys_owner

from ..discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ..typing import checked
from . import _lineage_records as lineage_owner
from ._association_state import (
    entity_rows,
    GeometryAssociation,
    GeometryAssociationKind,
    target_dimension,
)
from ._lineage_records import MeshLineage


class PlcEntityClasses(NamedTuple):
    """Source strata and explicit namespace codes of one mesh dimension."""

    dimensions: np.ndarray
    indices: np.ndarray
    resolved: np.ndarray
    codes: np.ndarray


class AssociationSource(Protocol):
    mesh: cell_mesh_owner.CellMesh
    associations: tuple[GeometryAssociation, ...]


@checked
def require_transfer_lineage(
    source: cell_mesh_owner.CellMesh,
    lineage: MeshLineage,
    target: cell_mesh_owner.CellMesh,
    /,
    *,
    degrees: tuple[int, ...] = (),
    message: str,
    unknown_message: str = "AssociationTransfer.lineage.relation_kinds must identify source strata; got an unknown relation",
) -> None:
    """Require source-to-target topology identity before checking relation status."""
    if (
        lineage.source_topology_id != source.topology_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError(message)
    for degree in degrees:
        if np.any(
            np.asarray(lineage.entity_lineage(degree).relation_kinds)
            == lineage_owner.EntityLineageKind.UNKNOWN
        ):
            raise ValueError(unknown_message)


def association_for_dimension(
    source: AssociationSource,
    dimension: int,
    /,
    *,
    namespace: tuple[str, str] | None = None,
) -> GeometryAssociation:
    """Find the declared table without changing namespace or source row order."""
    return next(
        item
        for item in source.associations
        if target_dimension(source.mesh, item) == dimension
        and (namespace is None or (item.source_id, item.source_revision) == namespace)
    )


def association_lookup(source: AssociationSource, /) -> dict[int, GeometryAssociation]:
    """Index present tables by dimension, preserving source traversal order."""
    return {target_dimension(source.mesh, item): item for item in source.associations}


class AssociationPropagationError(ValueError):
    """A topology change violates the B-Rep classification rules."""

    def __init__(self, message: str, target_ids: ArrayLike, /) -> None:
        super().__init__(message)
        self.target_ids = np.asarray(target_ids, dtype=np.int64)


# -- mesh topology helpers -------------------------------------------------------


def compose_incidence_pairs(
    first: np.ndarray, second: np.ndarray, /, *, unique: bool = True
) -> np.ndarray:
    """Compose ``(a, b)`` pairs with ``(b, c)`` pairs into ``(a, c)`` pairs.

    With ``unique=False`` every path is kept, so multiplicities count the
    intermediate ``b`` connecting each ``(a, c)``.
    """
    if first.shape[0] == 0 or second.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    order = np.argsort(second[:, 0], kind="stable")
    keys = second[order, 0]
    start = np.searchsorted(keys, first[:, 1], side="left")
    counts = np.searchsorted(keys, first[:, 1], side="right") - start
    owner = np.repeat(np.arange(first.shape[0]), counts)
    offsets = np.cumsum(counts) - counts
    position = np.repeat(start - offsets, counts) + np.arange(owner.size)
    pairs = np.stack((first[owner, 0], second[order[position], 1]), axis=1)
    return np.unique(pairs, axis=0).reshape(-1, 2) if unique else pairs


def incidence_pairs(mesh: cell_mesh_owner.CellMesh, low: int, high: int, /) -> np.ndarray:
    """``(low_row, high_row)`` pairs with the low entity in the high entity's closure."""
    if low == high:
        rows = np.arange(mesh.entity_set(low).count, dtype=np.int64)
        return np.stack((rows, rows), axis=1)
    pairs = None
    for degree in range(low, high):
        relation = mesh.topology.incidences[degree].relation
        valid = np.asarray(relation.valid, dtype=np.bool_).reshape(-1)
        step = np.stack(
            (
                np.asarray(relation.source_indices, dtype=np.int64).reshape(-1)[valid],
                np.asarray(relation.target_indices, dtype=np.int64).reshape(-1)[valid],
            ),
            axis=1,
        )
        pairs = step if pairs is None else compose_incidence_pairs(pairs, step)
    # ty: ignore[no-matching-overload]
    return np.unique(pairs, axis=0).reshape(-1, 2)


def boundary_mask(mesh: cell_mesh_owner.CellMesh, dimension: int, /) -> np.ndarray:
    """Entities of ``dimension`` in the closure of mesh boundary facets."""
    top = mesh.topological_dimension
    facets = incidence_pairs(mesh, top - 1, top)
    boundary_facets = np.flatnonzero(
        np.bincount(facets[:, 0], minlength=mesh.entity_set(top - 1).count) == 1
    )
    mask = np.zeros((mesh.entity_set(dimension).count,), dtype=np.bool_)
    if dimension == top:
        return mask
    pairs = incidence_pairs(mesh, dimension, top - 1)
    mask[pairs[np.isin(pairs[:, 1], boundary_facets), 0]] = True
    return mask


def ordered_vertices(
    mesh: cell_mesh_owner.CellMesh, dimension: int, /
) -> np.ndarray | None:
    """Return incidence-ordered vertex rows for entities of ``dimension`` on simplex routes."""
    connectivity = mesh.connectivity
    if dimension == 1 and isinstance(
        connectivity, (PolygonalConnectivity, TetrahedralConnectivity)
    ):
        return np.asarray(connectivity.edges, dtype=np.int64)
    if dimension == 2 and isinstance(connectivity, TetrahedralConnectivity):
        return np.asarray(connectivity.faces, dtype=np.int64)
    if dimension == mesh.topological_dimension == 2 and all(
        block.cell_kind == "triangle" for block in mesh.blocks
    ):
        identifiers = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
        )
        rows = entity_rows(mesh, dimension, identifiers)
        ordered = np.empty_like(cells)
        ordered[rows] = cells
        return ordered
    return None


def child_sources(
    lineage: MeshLineage,
    source_mesh: cell_mesh_owner.CellMesh,
    target_mesh: cell_mesh_owner.CellMesh,
    created: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest-dimensional source entity ``(dimension, row)`` each created vertex lies in.

    A vertex on a target edge (face) refined from a source edge (face) lies in
    that source entity; otherwise its split parents span the source simplex.
    Interior vertices without split-parent records inherit a source cell only
    when every incident target cell refines that same source cell.
    """

    top = source_mesh.topological_dimension
    dims = np.full((created.size,), -1, dtype=np.int64)
    rows = np.full((created.size,), -1, dtype=np.int64)
    target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    refined = (
        int(lineage_owner.EntityLineageKind.REFINED_FROM),
        int(lineage_owner.EntityLineageKind.SPLIT_FROM),
    )
    for dimension in range(1, top):
        record = lineage.entity_lineage(dimension)
        kinds = np.asarray(record.relation_kinds)
        mask = np.isin(kinds, refined)
        source_rows = entity_rows(
            source_mesh, dimension, np.asarray(record.source_global_ids)[mask]
        )
        target_rows = entity_rows(
            target_mesh, dimension, np.asarray(record.target_global_ids)[mask]
        )
        vertices = incidence_pairs(target_mesh, 0, dimension)
        entity_vertices = compose_incidence_pairs(
            np.stack((np.arange(target_rows.size), target_rows), axis=1),
            vertices[:, ::-1],
        )
        vertex_ids = target_ids[entity_vertices[:, 1]]
        position = np.searchsorted(created, vertex_ids)
        position = np.minimum(position, created.size - 1)
        hit = (created.size > 0) & (created[position] == vertex_ids)
        open_ = dims[position[hit]] < 0
        chosen = position[hit][open_]
        dims[chosen] = dimension
        rows[chosen] = source_rows[entity_vertices[hit, 0][open_]]
    record = lineage.entity_lineage(0)
    split = (
        np.asarray(record.relation_kinds) == lineage_owner.EntityLineageKind.SPLIT_FROM
    )
    parents = np.stack(
        (
            np.asarray(record.target_global_ids)[split],
            np.asarray(record.source_global_ids)[split],
        ),
        axis=1,
    )
    open_rows = np.flatnonzero(dims < 0)
    if open_rows.size and parents.size:
        parents = parents[np.isin(parents[:, 0], created[open_rows])]
        parents = parents[np.lexsort((parents[:, 1], parents[:, 0]))]
        owners, starts, sizes = np.unique(
            parents[:, 0], return_index=True, return_counts=True
        )
        for size in np.unique(sizes).tolist():
            if size - 1 > top or size < 2:
                continue
            selected = sizes == size
            keys = np.stack(
                [parents[starts[selected] + offset, 1] for offset in range(size)], axis=1
            )
            if size - 1 == top:
                cells = np.concatenate(
                    [
                        np.asarray(block.vertices, dtype=np.int64)
                        for block in source_mesh.blocks
                    ]
                )
                identifiers = np.concatenate(
                    [
                        np.asarray(block.global_ids, dtype=np.int64)
                        for block in source_mesh.blocks
                    ]
                )
                table = np.sort(
                    np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)[cells],
                    axis=1,
                )
                found = entity_keys_owner.key_rows(table, keys)
                entity = np.where(
                    found >= 0,
                    entity_rows(source_mesh, top, identifiers[np.maximum(found, 0)]),
                    -1,
                )
            else:
                entity = entity_keys_owner.key_rows(
                    entity_keys_owner.entity_keys(source_mesh, size - 1), keys
                )
            position = np.searchsorted(created, owners[selected])
            dims[position] = np.where(entity >= 0, size - 1, -1)
            rows[position] = entity
    if np.any(dims < 0) and any(
        record.dimension == top for record in lineage.entities
    ):
        record = lineage.entity_lineage(top)
        mask = np.isin(np.asarray(record.relation_kinds), refined)
        parents = np.unique(
            np.stack(
                (
                    entity_rows(
                        target_mesh, top, np.asarray(record.target_global_ids)[mask]
                    ),
                    entity_rows(
                        source_mesh, top, np.asarray(record.source_global_ids)[mask]
                    ),
                ),
                axis=1,
            ),
            axis=0,
        )
        owners, counts = np.unique(parents[:, 0], return_counts=True)
        cell_parents = np.full(target_mesh.entity_set(top).count, -1, dtype=np.int64)
        single = counts == 1
        cell_parents[owners[single]] = parents[
            np.searchsorted(parents[:, 0], owners[single]), 1
        ]
        vertex_parents = stratum_region_consensus(target_mesh, 0, cell_parents, top)
        target_rows = entity_rows(target_mesh, 0, created)
        interior = ~boundary_mask(target_mesh, 0)[target_rows]
        inherited = (dims < 0) & interior & (vertex_parents[target_rows] >= 0)
        dims[inherited] = top
        rows[inherited] = vertex_parents[target_rows[inherited]]
    return dims, rows


def surface_source_association_kind(
    source_kinds: tuple[str, ...], /
) -> GeometryAssociationKind:
    """Resolve the parametric stratum namespace from its declared source kinds."""
    match source_kinds:
        case ("corner", "curve", "surface"):
            return GeometryAssociationKind.SURFACE
        case ("vertex", "edge", "face"):
            return GeometryAssociationKind.BREP
        case _:
            raise ValueError(
                f"SurfaceAssociationTransfer.source_kinds must declare surface or B-Rep strata; got {source_kinds!r}"
            )


@checked
def stratum_region_consensus(
    mesh: cell_mesh_owner.CellMesh,
    dimension: int,
    regions: np.ndarray,
    top_dimension: int,
    /,
) -> np.ndarray:
    """Return a region only when all incident top cells agree."""
    if dimension == top_dimension:
        return regions.copy()
    links = incidence_pairs(mesh, dimension, top_dimension)
    pairs = np.unique(np.stack((links[:, 0], regions[links[:, 1]]), axis=1), axis=0)
    rows, counts = np.unique(pairs[:, 0], return_counts=True)
    result = np.full(mesh.entity_set(dimension).count, -1, dtype=np.int64)
    single = rows[counts == 1]
    result[single] = pairs[np.searchsorted(pairs[:, 0], single), 1]
    return result
