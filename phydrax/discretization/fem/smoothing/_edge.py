#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from ..._cell_complex import PolygonalConnectivity
from ..._cell_mesh import CellMesh
from ._common import SmoothingPatchLayout


def _cell_incidence(connectivity: PolygonalConnectivity):
    face_count = int(connectivity.edges.shape[0])
    owner = np.full((face_count,), -1, dtype=np.int32)
    neighbour = np.full((face_count,), -1, dtype=np.int32)
    for cell in range(connectivity.cell_edges.shape[0]):
        for local in range(connectivity.cell_edges.shape[1]):
            if not bool(connectivity.cell_edge_valid[cell, local]):
                continue
            edge = int(connectivity.cell_edges[cell, local])
            if owner[edge] < 0:
                owner[edge] = cell
            else:
                neighbour[edge] = cell
    return owner, neighbour


def edge_smoothing_layout(mesh: CellMesh, /) -> SmoothingPatchLayout:
    """Construct 2-D T3 ES-FEM edge-star patches with two-point boundary rules."""

    if not isinstance(mesh, CellMesh) or mesh.topological_dimension != 2:
        raise TypeError("ES-FEM requires a two-dimensional CellMesh.")
    if any(block.cell_kind != "triangle" for block in mesh.blocks):
        raise ValueError("Initial ES-FEM supports triangular cell blocks only.")
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("ES-FEM requires polygonal connectivity.")
    coordinates = np.asarray(mesh.coordinates)
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks),
        axis=0,
    )
    owner, neighbour = _cell_incidence(connectivity)
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    patch_count = edges.shape[0]
    max_vertices = 4
    max_dofs = 4
    max_sources = 3
    rule_points = np.asarray([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
    rule_weights = np.asarray([0.5, 0.5])
    owners = np.arange(patch_count, dtype=np.int32)
    dof_routes = np.zeros((patch_count, max_dofs), dtype=np.int32)
    dof_valid = np.zeros_like(dof_routes, dtype=bool)
    vertex_sources = np.zeros((patch_count, max_vertices, max_sources), dtype=np.int32)
    vertex_coefficients = np.zeros_like(vertex_sources, dtype=float)
    vertex_valid = np.zeros((patch_count, max_vertices), dtype=bool)
    boundary_edges = np.zeros((patch_count, max_vertices, 2), dtype=np.int32)
    boundary_valid = np.zeros((patch_count, max_vertices), dtype=bool)
    boundary_shape_values = np.zeros(
        (patch_count, max_vertices, rule_points.size, max_dofs), dtype=float
    )
    for edge in range(patch_count):
        incident = [owner[edge]]
        if neighbour[edge] >= 0:
            incident.append(neighbour[edge])
        if owner[edge] < 0 or len(incident) > 2:
            raise ValueError("Every ES-FEM edge must have one or two incident cells.")
        edge_vertices = list(edges[edge])
        stencil = sorted(
            set(edge_vertices).union(*(set(cells[cell]) for cell in incident))
        )
        if len(stencil) > max_dofs:
            raise ValueError("T3 ES-FEM edge stencil exceeded four vertices.")
        dof_routes[edge, : len(stencil)] = stencil
        dof_valid[edge, : len(stencil)] = True
        local_index = {vertex: index for index, vertex in enumerate(stencil)}

        def endpoint(vertex):
            sources = [vertex]
            coefficients = [1.0]
            shapes = np.zeros((max_dofs,))
            shapes[local_index[vertex]] = 1.0
            return sources, coefficients, shapes

        def centroid(cell):
            sources = list(cells[cell])
            coefficients = [1.0 / 3.0] * 3
            shapes = np.zeros((max_dofs,))
            for vertex in cells[cell]:
                shapes[local_index[int(vertex)]] = 1.0 / 3.0
            return sources, coefficients, shapes

        patch = [
            endpoint(edge_vertices[0]),
            centroid(owner[edge]),
            endpoint(edge_vertices[1]),
        ]
        if neighbour[edge] >= 0:
            patch.append(centroid(neighbour[edge]))
        patch_coordinates = []
        for sources, coefficients, _ in patch:
            patch_coordinates.append(
                sum(
                    coefficient * coordinates[source]
                    for source, coefficient in zip(sources, coefficients, strict=True)
                )
            )
        patch_coordinates = np.asarray(patch_coordinates)
        area2 = np.sum(
            patch_coordinates[:, 0] * np.roll(patch_coordinates[:, 1], -1)
            - np.roll(patch_coordinates[:, 0], -1) * patch_coordinates[:, 1]
        )
        if area2 < 0.0:
            patch = [patch[0]] + list(reversed(patch[1:]))
        count = len(patch)
        vertex_valid[edge, :count] = True
        for vertex_index, (sources, coefficients, _) in enumerate(patch):
            vertex_sources[edge, vertex_index, : len(sources)] = sources
            vertex_coefficients[edge, vertex_index, : len(sources)] = coefficients
        for piece in range(count):
            start = piece
            end = (piece + 1) % count
            boundary_edges[edge, piece] = (start, end)
            boundary_valid[edge, piece] = True
            start_shape = patch[start][2]
            end_shape = patch[end][2]
            boundary_shape_values[edge, piece] = (1.0 - rule_points)[
                :, None
            ] * start_shape[None, :] + rule_points[:, None] * end_shape[None, :]
    return SmoothingPatchLayout(
        "edge",
        owners,
        dof_routes,
        dof_valid,
        vertex_sources,
        vertex_coefficients,
        vertex_valid,
        boundary_edges,
        boundary_valid,
        boundary_shape_values,
        rule_points,
        rule_weights,
    )


__all__ = ["edge_smoothing_layout"]
