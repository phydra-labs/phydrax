#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..._cell_mesh import CellMesh
from ._common import SmoothingPatchLayout


def _key(coefficients: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(coefficients, 14))


def node_smoothing_layout(mesh: CellMesh, /) -> SmoothingPatchLayout:
    """Construct 2-D T3 NS-FEM node-star patches from centroid/mid-edge pieces."""

    if not isinstance(mesh, CellMesh) or mesh.topological_dimension != 2:
        raise TypeError("NS-FEM requires a two-dimensional CellMesh.")
    if any(block.cell_kind != "triangle" for block in mesh.blocks):
        raise ValueError("Initial NS-FEM supports triangular cell blocks only.")
    coordinates = np.asarray(mesh.coordinates)
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks),
        axis=0,
    )
    vertex_count = coordinates.shape[0]
    incident = [[] for _ in range(vertex_count)]
    for cell_index, cell in enumerate(cells):
        for vertex in cell:
            incident[int(vertex)].append(cell_index)
    rule_points = np.asarray([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
    rule_weights = np.asarray([0.5, 0.5])
    patch_records = []
    for node in range(vertex_count):
        if not incident[node]:
            raise ValueError("Every NS-FEM node must belong to at least one cell.")
        stencil = sorted(set(cells[incident[node]].reshape((-1,))))
        local = {vertex: index for index, vertex in enumerate(stencil)}
        coefficient_by_key = {}
        shape_by_key = {}
        edge_counter = Counter()
        directed_edges = defaultdict(list)

        def register(coefficients):
            key = _key(coefficients)
            coefficient_by_key[key] = coefficients
            shape = np.zeros((len(stencil),))
            for vertex, coefficient in enumerate(coefficients):
                if coefficient != 0.0 and vertex in local:
                    shape[local[vertex]] = coefficient
            shape_by_key[key] = shape
            return key

        node_coefficients = np.zeros((vertex_count,))
        node_coefficients[node] = 1.0
        node_key = register(node_coefficients)
        for cell_index in incident[node]:
            cell = cells[cell_index]
            position = int(np.flatnonzero(cell == node)[0])
            next_vertex = int(cell[(position + 1) % 3])
            previous_vertex = int(cell[(position - 1) % 3])
            next_mid = np.zeros((vertex_count,))
            next_mid[node] = 0.5
            next_mid[next_vertex] = 0.5
            previous_mid = np.zeros((vertex_count,))
            previous_mid[node] = 0.5
            previous_mid[previous_vertex] = 0.5
            centroid = np.zeros((vertex_count,))
            centroid[cell] = 1.0 / 3.0
            piece = (
                node_key,
                register(next_mid),
                register(centroid),
                register(previous_mid),
            )
            for first, second in zip(piece, piece[1:] + piece[:1], strict=True):
                undirected = tuple(sorted((first, second)))
                edge_counter[undirected] += 1
                directed_edges[undirected].append((first, second))
        boundary_directed = [
            directed_edges[edge][0] for edge, count in edge_counter.items() if count == 1
        ]
        if not boundary_directed:
            raise ValueError("NS-FEM node star produced no exterior boundary.")
        successor = {first: second for first, second in boundary_directed}
        if len(successor) != len(boundary_directed):
            raise ValueError("NS-FEM node-star boundary is not a simple loop.")
        start = boundary_directed[0][0]
        ordered = [start]
        while True:
            next_key = successor[ordered[-1]]
            if next_key == start:
                break
            if next_key in ordered:
                raise ValueError("NS-FEM node-star boundary contains a cycle defect.")
            ordered.append(next_key)
        patch_records.append((stencil, ordered, coefficient_by_key, shape_by_key))
    max_dofs = max(len(record[0]) for record in patch_records)
    max_vertices = max(len(record[1]) for record in patch_records)
    max_sources = 3
    dof_routes = np.zeros((vertex_count, max_dofs), dtype=np.int32)
    dof_valid = np.zeros_like(dof_routes, dtype=bool)
    vertex_sources = np.zeros((vertex_count, max_vertices, max_sources), dtype=np.int32)
    vertex_coefficients = np.zeros_like(vertex_sources, dtype=float)
    vertex_valid = np.zeros((vertex_count, max_vertices), dtype=bool)
    boundary_edges = np.zeros((vertex_count, max_vertices, 2), dtype=np.int32)
    boundary_valid = np.zeros((vertex_count, max_vertices), dtype=bool)
    boundary_shape_values = np.zeros(
        (vertex_count, max_vertices, rule_points.size, max_dofs), dtype=float
    )
    for node, (stencil, ordered, coefficients_by_key, shapes_by_key) in enumerate(
        patch_records
    ):
        dof_routes[node, : len(stencil)] = stencil
        dof_valid[node, : len(stencil)] = True
        vertex_valid[node, : len(ordered)] = True
        padded_shapes = []
        for vertex_index, key in enumerate(ordered):
            coefficients = coefficients_by_key[key]
            active = np.flatnonzero(coefficients)
            vertex_sources[node, vertex_index, : active.size] = active
            vertex_coefficients[node, vertex_index, : active.size] = coefficients[active]
            shape = np.zeros((max_dofs,))
            shape[: len(stencil)] = shapes_by_key[key]
            padded_shapes.append(shape)
        for piece in range(len(ordered)):
            start = piece
            end = (piece + 1) % len(ordered)
            boundary_edges[node, piece] = (start, end)
            boundary_valid[node, piece] = True
            boundary_shape_values[node, piece] = (1.0 - rule_points)[
                :, None
            ] * padded_shapes[start][None, :] + rule_points[:, None] * padded_shapes[end][
                None, :
            ]
    return SmoothingPatchLayout(
        "node",
        np.arange(vertex_count, dtype=np.int32),
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


__all__ = ["node_smoothing_layout"]
