#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host entity banks, ancestry, and source keys for mesh construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import phydrax.meshing._failures as _leaf_phydrax_meshing__failures

from . import _entity_keys as entity_keys_owner, _scope as scope_owner


if TYPE_CHECKING:
    from ..discretization._cell_mesh import CellMesh
    from ._adaptation import MarkedMeshAdaptation
    from ._result import CellMeshingResult
    from ._scope import MeshingScope
import numpy as np
from jax.typing import DTypeLike

from .._meshcore import NativeExecutionScope
from ..discretization._cell_complex import (
    dense_rows,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._hexahedral import HexahedralConnectivity


def failure(
    message: str, /, *, resource: bool = False
) -> _leaf_phydrax_meshing__failures.MeshingFailure:
    return _leaf_phydrax_meshing__failures.MeshingFailure(
        _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
        if resource
        else _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
        message,
        stage="family-extraction",
    )


def family_host_array(
    shape: tuple[int, ...], dtype: DTypeLike, /, *, scope: NativeExecutionScope
) -> np.ndarray:
    """Allocate family host banks in the explicit native pool."""
    return scope.allocate_host_array(shape, dtype)


def entities(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    if dimension == 0:
        return np.arange(mesh.coordinates.shape[0], dtype=np.int64)[:, None]
    if dimension == mesh.topology.dimension:
        width = max(block.vertices.shape[1] for block in mesh.blocks)
        return np.concatenate(
            tuple(
                np.pad(
                    np.asarray(block.vertices, dtype=np.int64),
                    ((0, 0), (0, width - block.vertices.shape[1])),
                    constant_values=-1,
                )
                for block in mesh.blocks
            )
        )
    if dimension == 1:
        if not isinstance(
            mesh.connectivity,
            (
                PolygonalConnectivity,
                TetrahedralConnectivity,
                HexahedralConnectivity,
                PolyhedralConnectivity,
            ),
        ):
            raise ValueError(
                f"entities.mesh.connectivity must provide surface or volume edges; got {type(mesh.connectivity).__name__}."
            )
        return np.asarray(mesh.connectivity.edges, dtype=np.int64)
    if dimension == 2 and mesh.topology.dimension == 3:
        if isinstance(mesh.connectivity, PolyhedralConnectivity):
            offsets = np.asarray(mesh.connectivity.face_vertex_offsets, dtype=np.int64)
            values = np.asarray(mesh.connectivity.face_vertex_values, dtype=np.int64)
            lengths = np.diff(offsets)
            width = int(np.max(lengths))
            dense = dense_rows(offsets, values, width)
            return np.where(np.arange(width)[None, :] < lengths[:, None], dense, -1)
        if not isinstance(
            mesh.connectivity, (TetrahedralConnectivity, HexahedralConnectivity)
        ):
            raise ValueError(
                f"entities.mesh.connectivity must provide volume faces; got {type(mesh.connectivity).__name__}."
            )
        return np.asarray(mesh.connectivity.faces, dtype=np.int64)
    raise ValueError(
        f"entities.dimension must have an entity-vertex route; got {dimension!r}."
    )


def ancestry(
    source: CellMesh,
    target: CellMesh,
    supports: np.ndarray,
    /,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    lookup: dict[tuple[int, ...], tuple[int, int]] = {}
    for dimension in range(source.topology.dimension + 1):
        for row, vertices in enumerate(entities(source, dimension).tolist()):
            key = tuple(sorted(value for value in vertices if value >= 0))
            lookup[key] = (dimension, row)
    dimensions, rows = [], []
    for dimension in range(target.topology.dimension + 1):
        vertices = entities(target, dimension)
        parents = []
        for entity in vertices.tolist():
            key = tuple(
                sorted(
                    {
                        value
                        for vertex in entity
                        if vertex >= 0
                        for value in supports[vertex].tolist()
                        if value >= 0
                    }
                )
            )
            if key not in lookup:
                raise RuntimeError(
                    f"ancestry.entity_key must have a source ancestor; got {key!r}."
                )
            parents.append(lookup[key])
        parent = np.asarray(parents, dtype=np.int64).reshape((-1, 2))
        dimensions.append(parent[:, 0])
        rows.append(parent[:, 1])
    return tuple(dimensions), tuple(rows)


def entity_key(revision: str, kind: str, index: int, /) -> str:
    # Matches the native source entity convention of the provider bindings.
    return f"{revision}:{kind}:{index}"


def closure_rows(
    mesh: CellMesh, dimension: int, rows: np.ndarray, target: int, /
) -> np.ndarray:
    """Rows of the dimension-``target`` entities in the closure of selected rows."""

    for degree in range(dimension, target, -1):
        relation = mesh.topology.incidences[degree - 1].relation
        valid = np.asarray(relation.valid) & np.isin(
            np.asarray(relation.target_indices), rows
        )
        rows = np.unique(np.asarray(relation.source_indices)[valid])
    return rows


def organization_scopes(source: CellMeshingResult, /) -> tuple[MeshingScope, ...]:
    return tuple(
        value.scope for value in (*source.patches, *source.zones, *source.labels)
    )


def row_classes(rows: np.ndarray, /) -> np.ndarray:
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    return inverse.reshape((-1,)).astype(np.int64)


def mesh_cell_ids(mesh: CellMesh, /) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )


def selected_rows(mesh: CellMesh, scope: MeshingScope, /) -> np.ndarray:
    return np.flatnonzero(np.asarray(scope_owner.resolve_mesh_scope(mesh, scope).mask))


def membership(
    mesh: CellMesh,
    scopes: tuple[MeshingScope, ...],
    dimension: int,
    /,
) -> np.ndarray:
    columns = [
        np.asarray(
            scope_owner.resolve_mesh_scope(mesh, scope).mask,
            dtype=np.bool_,
        )
        for scope in scopes
        if scope.entity_dimension == dimension
    ]
    count = mesh.entity_set(dimension).count
    if not columns:
        return np.zeros((count, 0), dtype=np.bool_)
    return np.stack(columns, axis=1)


def mesh_cell_classes(
    mesh: CellMesh,
    scopes: tuple[MeshingScope, ...],
    /,
    *,
    family_identity: bool = False,
    scientific_block_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Organization classes; mixed templates retain family, not coordinate-block identity."""

    dimension = mesh.topological_dimension
    families = {
        kind: index
        for index, kind in enumerate(sorted({b.cell_kind for b in mesh.blocks}))
    }
    block_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    block_rows = (
        scientific_block_rows
        if scientific_block_rows is not None
        else np.concatenate(
            [
                np.full(
                    (block.cell_count,),
                    families[block.cell_kind] if family_identity else index,
                    dtype=np.int64,
                )
                for index, block in enumerate(mesh.blocks)
            ]
        )
    )
    cells = entity_keys_owner.key_rows(
        block_ids[:, None],
        entity_keys_owner.entity_keys(mesh, dimension),
    )
    return row_classes(
        np.column_stack(
            (block_rows[cells], membership(mesh, scopes, dimension).astype(np.int64))
        )
    )


def edge_classes(
    mesh: CellMesh,
    scopes: tuple[MeshingScope, ...],
    cell_classes: np.ndarray,
    /,
) -> np.ndarray:
    """Planar feature classes: 0 for free interior edges, positive otherwise.

    Boundary edges, region interfaces, and edges of any edge organization are
    features; distinct incident region pairs or memberships are distinct classes.
    """

    count = mesh.entity_set(1).count
    relation = mesh.topology.incidences[1].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    edges = np.asarray(relation.source_indices)[valid]
    cells = cell_classes[np.asarray(relation.target_indices)[valid]]
    incident = np.bincount(edges, minlength=count)
    low = np.full((count,), np.iinfo(np.int64).max, dtype=np.int64)
    high = np.full((count,), -1, dtype=np.int64)
    np.minimum.at(low, edges, cells)
    np.maximum.at(high, edges, cells)
    member = membership(mesh, scopes, 1)
    feature = (incident != 2) | (low != high) | np.any(member, axis=1)
    classes = np.zeros((count,), dtype=np.int64)
    if np.any(feature):
        keys = np.column_stack(
            (incident == 1, low, high, member.astype(np.int64))
        ).astype(np.int64)
        classes[feature] = 1 + row_classes(keys[feature])
    return classes


def check_marked_source(
    source: CellMeshingResult,
    request: MarkedMeshAdaptation,
    /,
) -> None:
    identifiers = mesh_cell_ids(source.mesh)
    marked = np.concatenate(
        (np.asarray(request.refine_cell_ids), np.asarray(request.coarsen_cell_ids))
    )
    if not np.all(np.isin(marked, identifiers)):
        raise ValueError(
            f"check_marked_source.marked_cell_ids must belong to the source mesh; got {marked!r}."
        )
    if request.layer_columns is not None and not np.all(
        np.isin(request.layer_columns.cell_ids, identifiers)
    ):
        raise ValueError(
            f"check_marked_source.layer_columns.cell_ids must belong to the source mesh; got {request.layer_columns.cell_ids!r}."
        )
