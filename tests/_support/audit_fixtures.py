"""Mesh ordering and bound organization fixtures for audit contracts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    reference_cell_topology,
)
from phydrax.discretization.fem import lagrange_element
from phydrax.meshing import (
    GeometryAssociation,
    GeometryAssociationKind,
    MeshingEntityKind,
    MeshingScope,
    MeshLabel,
)


def unordered_reference_cells(cell_kind: str = "triangle") -> CellMesh:
    """Return two disjoint cells with descending cell IDs and stable entity IDs.

    The fixture preserves the coordinates and IDs of test_audit_quality._two_cells.
    """
    points = np.asarray(reference_cell_topology(cell_kind).vertices)
    count = len(points)
    mesh = CellMesh(
        np.concatenate((points, points + 3.0)),
        (
            CellBlock(
                "cells",
                cell_kind,
                np.arange(2 * count, dtype=np.int64).reshape(2, count),
                global_ids=np.asarray([20, 10], dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(2 * count, dtype=np.int64) + 100,
    )
    return CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        entity_global_ids={
            dimension: np.arange(mesh.entity_set(dimension).count, dtype=np.int64)
            + 1000 * dimension
            for dimension in range(1, mesh.topological_dimension)
        },
    )


def quadratic_triangle_geometry(mesh: CellMesh) -> CellGeometrySpec:
    """Raise edge nodes by 0.05 while retaining the mesh's triangle corners.

    The interpolation and operation order preserve test_audit_quality._quadratic_geometry.
    """
    element = lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes)
    points = np.asarray(mesh.coordinates)
    local = points[np.asarray(mesh.blocks[0].vertices)]
    coordinates = (
        local[:, :1]
        + nodes[None, :, :1] * (local[:, 1:2] - local[:, :1])
        + nodes[None, :, 1:] * (local[:, 2:3] - local[:, :1])
    )
    vertex_nodes = np.asarray(reference_cell_topology("triangle").vertices)
    edge_nodes = ~np.any(np.all(nodes[:, None] == vertex_nodes[None], axis=-1), axis=1)
    coordinates[:, edge_nodes, 1] += 0.05
    routes = np.arange(
        coordinates.shape[0] * coordinates.shape[1], dtype=np.int64
    ).reshape(coordinates.shape[:2])
    return CellGeometrySpec(
        {"cells": element},
        {"cells": routes},
        coordinates.reshape(-1, coordinates.shape[-1]),
    )


def quadratic_triangle_with_edge_nodes(
    edge_nodes: Sequence[Sequence[float]],
) -> CellGeometrySpec:
    """Retain unit-triangle corners and the three supplied quadratic edge nodes.

    Node order is the Lagrange order used by test_audit_quality._p2_triangle.
    """
    element = lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    nodes[3:] = edge_nodes
    return CellGeometrySpec(
        {"cells": element}, {"cells": np.arange(6, dtype=np.int64)[None, :]}, nodes
    )


def cell_label(
    mesh: CellMesh, ids: Sequence[int], *, source_id: str | None = None
) -> MeshLabel:
    """Bind the selection label to cell IDs, optionally using a stale source ID."""
    entities = mesh.entity_set(mesh.topological_dimension)
    return MeshLabel(
        "selection",
        MeshingScope(
            mesh.mesh_id if source_id is None else source_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            mesh.topological_dimension,
            entities.entity_set_id,
            np.asarray(ids, dtype=np.int64),
        ),
    )


def surface_cell_association(
    mesh: CellMesh,
    ids: Sequence[int],
    *,
    entity_set_id: str | None = None,
    residual: float = 0.0,
) -> GeometryAssociation:
    """Bind cells to source-face identity with an explicit residual for every row."""
    return GeometryAssociation(
        GeometryAssociationKind.SURFACE,
        "source",
        "revision",
        mesh.entity_set(mesh.topological_dimension).entity_set_id
        if entity_set_id is None
        else entity_set_id,
        np.asarray(ids, dtype=np.int64),
        tuple("source-face" for _ in ids),
        np.full(len(ids), residual, dtype=np.float64),
    )
