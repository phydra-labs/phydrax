#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from ..._cell_mesh import CellMesh
from ._common import SmoothingPatchLayout


def _partition_shape(count: int) -> tuple[int, int]:
    shapes = {
        1: (1, 1),
        2: (2, 1),
        3: (3, 1),
        4: (2, 2),
        8: (4, 2),
        16: (4, 4),
    }
    if count not in shapes:
        raise ValueError("Q4 smoothing-cell count must be 1, 2, 3, 4, 8, or 16.")
    return shapes[count]


def _q1_shape(xi: float, eta: float) -> np.ndarray:
    return np.asarray(
        [
            (1.0 - xi) * (1.0 - eta),
            xi * (1.0 - eta),
            xi * eta,
            (1.0 - xi) * eta,
        ]
    )


def q4_cell_smoothing_layout(
    mesh: CellMesh,
    smoothing_cells: int,
    /,
) -> SmoothingPatchLayout:
    """Uniform reference Q4 cell-smoothed partitions for source SC counts."""

    if not isinstance(mesh, CellMesh) or mesh.topological_dimension != 2:
        raise TypeError("Q4 cell smoothing requires a two-dimensional CellMesh.")
    if any(block.cell_kind != "quadrilateral" for block in mesh.blocks):
        raise ValueError("Q4 cell smoothing requires quadrilateral blocks only.")
    n_xi, n_eta = _partition_shape(int(smoothing_cells))
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks),
        axis=0,
    )
    rule_points = np.asarray([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
    rule_weights = np.asarray([0.5, 0.5])
    patch_count = cells.shape[0] * n_xi * n_eta
    dof_routes = np.zeros((patch_count, 4), dtype=np.int32)
    dof_valid = np.ones((patch_count, 4), dtype=bool)
    vertex_sources = np.zeros((patch_count, 4, 4), dtype=np.int32)
    vertex_coefficients = np.zeros((patch_count, 4, 4), dtype=float)
    vertex_valid = np.ones((patch_count, 4), dtype=bool)
    boundary_edges = np.tile(
        np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32),
        (patch_count, 1, 1),
    )
    boundary_valid = np.ones((patch_count, 4), dtype=bool)
    boundary_shape_values = np.zeros((patch_count, 4, rule_points.size, 4), dtype=float)
    owners = np.zeros((patch_count,), dtype=np.int32)
    patch = 0
    for cell_index, cell in enumerate(cells):
        for j in range(n_eta):
            eta_0 = j / n_eta
            eta_1 = (j + 1) / n_eta
            for i in range(n_xi):
                xi_0 = i / n_xi
                xi_1 = (i + 1) / n_xi
                corners = (
                    (xi_0, eta_0),
                    (xi_1, eta_0),
                    (xi_1, eta_1),
                    (xi_0, eta_1),
                )
                owners[patch] = cell_index
                dof_routes[patch] = cell
                for vertex, (xi, eta) in enumerate(corners):
                    vertex_sources[patch, vertex] = cell
                    vertex_coefficients[patch, vertex] = _q1_shape(xi, eta)
                for piece, (start, end) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
                    start_xi, start_eta = corners[start]
                    end_xi, end_eta = corners[end]
                    for q, parameter in enumerate(rule_points):
                        xi = (1.0 - parameter) * start_xi + parameter * end_xi
                        eta = (1.0 - parameter) * start_eta + parameter * end_eta
                        boundary_shape_values[patch, piece, q] = _q1_shape(xi, eta)
                patch += 1
    return SmoothingPatchLayout(
        "cell",
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


__all__ = ["q4_cell_smoothing_layout"]
