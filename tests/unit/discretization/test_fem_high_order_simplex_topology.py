#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

import phydrax as phx


def _grid_triangles(rng: Any, count: Any) -> Any:
    axis = np.linspace(0.0, 1.0, count + 1)
    coordinates = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(
        (-1, 2)
    )
    cells = []
    for row in range(count):
        for column in range(count):
            a = row * (count + 1) + column
            b, c, d = a + 1, a + count + 2, a + count + 1
            if rng.random() < 0.5:
                cells.extend(((a, b, c), (a, c, d)))
            else:
                cells.extend(((a, b, d), (b, c, d)))
    return coordinates, np.asarray(cells, dtype=np.int32)


def _kuhn_tetrahedra(count: Any) -> Any:
    axis = np.linspace(0.0, 1.0, count + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    coordinates = grid.transpose(2, 1, 0, 3).reshape((-1, 3))

    def vertex(x: Any, y: Any, z: Any) -> Any:
        return (z * (count + 1) + y) * (count + 1) + x

    cells = []
    for x, y, z in itertools.product(range(count), repeat=3):
        for axes in itertools.permutations(range(3)):
            corner = [x, y, z]
            path = [vertex(*corner)]
            for axis_index in axes:
                corner[axis_index] += 1
                path.append(vertex(*corner))
            cells.append(path)
    return coordinates, np.asarray(cells, dtype=np.int32)


def _randomly_oriented_mesh(cell_kind: Any, seed: Any) -> Any:
    """Relabel vertices and permute every cell's local vertex order at random."""

    rng = np.random.default_rng(seed)
    if cell_kind == "triangle":
        coordinates, cells = _grid_triangles(rng, 3)
    else:
        coordinates, cells = _kuhn_tetrahedra(2)
    relabel = rng.permutation(coordinates.shape[0]).astype(np.int32)
    relabeled = np.empty_like(coordinates)
    relabeled[relabel] = coordinates
    cells = relabel[cells]
    oriented = []
    for cell in cells:
        while True:
            candidate = cell[rng.permutation(cell.shape[0])]
            corners = relabeled[candidate]
            if np.linalg.det((corners[1:] - corners[0]).T) > 0.0:
                break
        oriented.append(candidate)
    cells = np.asarray(oriented, dtype=np.int32)
    if cell_kind == "triangle":
        return phx.discretization.CellMesh.from_triangles(relabeled, cells)
    return phx.discretization.CellMesh.from_tetrahedra(relabeled, cells)


def _affine_points(mesh: Any, reference_points: Any) -> Any:
    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    jacobians = np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
    return corners[:, None, 0] + np.matmul(
        jacobians[:, None], np.asarray(reference_points)[None, :, :, None]
    ).squeeze(-1)


def _routed_dof_points(mesh: Any, element: Any) -> Any:
    dof_map = phx.discretization.FiniteElementDofMap(mesh, (element,))
    routes = np.asarray(dof_map.cell_dofs[0])
    local_points = _affine_points(mesh, element.reference_nodes)
    dimension = local_points.shape[-1]
    totals = np.zeros((dof_map.global_dof_count, dimension))
    counts = np.zeros((dof_map.global_dof_count,))
    np.add.at(totals, routes.reshape((-1,)), local_points.reshape((-1, dimension)))
    np.add.at(counts, routes.reshape((-1,)), 1.0)
    return routes, local_points, totals / counts[:, None], counts


def _cubic(points: Any) -> Any:
    x, y = points[..., 0], points[..., 1]
    value = 1.0 + x - 2.0 * y + x**2 * y - x**3 + 3.0 * y**3 - 0.5 * x * y**2
    if points.shape[-1] == 3:
        z = points[..., 2]
        value = value + z**3 - 2.0 * x * y * z + y * z**2
    return value


@pytest.mark.parametrize("cell_kind", ("triangle", "tetrahedron"))
@pytest.mark.parametrize("order", (3, 4))
@pytest.mark.parametrize("seed", (0, 1))
def test_randomly_oriented_simplex_dofs_route_to_one_physical_point(
    cell_kind: Any, order: Any, seed: Any
) -> None:
    mesh = _randomly_oriented_mesh(cell_kind, seed)
    element = phx.discretization.lagrange_element(cell_kind, order)

    routes, local_points, global_points, counts = _routed_dof_points(mesh, element)

    assert np.all(counts > 0)
    np.testing.assert_allclose(local_points, global_points[routes], atol=1.0e-12)
    # Distinct global DOFs are distinct physical nodes (no collapsed routes).
    rounded = np.round(global_points, decimals=9)
    assert np.unique(rounded, axis=0).shape[0] == global_points.shape[0]


@pytest.mark.parametrize("cell_kind", ("triangle", "tetrahedron"))
def test_p3_interpolation_of_cubic_is_exact_and_continuous(cell_kind: Any) -> None:
    mesh = _randomly_oriented_mesh(cell_kind, 3)
    element = phx.discretization.lagrange_element(cell_kind, 3)
    routes, _local_points, global_points, _counts = _routed_dof_points(mesh, element)
    coefficients = _cubic(global_points)
    dimension = element.topological_dimension

    rng = np.random.default_rng(7)
    interior = rng.dirichlet(np.ones((dimension + 1,)), size=16)[:, 1:]
    topology = phx.discretization.reference_cell_topology(cell_kind)
    corners = np.asarray(topology.vertices)
    parameters = np.asarray((0.17, 0.5, 0.71))
    edge_points = np.concatenate(
        tuple(
            (1.0 - parameters)[:, None] * corners[start]
            + parameters[:, None] * corners[stop]
            for start, stop in topology.entities[1]
        )
    )
    reference = np.concatenate((interior, edge_points))
    basis, _gradients = element.tabulate(reference)
    values = np.asarray(basis) @ coefficients[routes].T
    physical = _affine_points(mesh, reference)

    np.testing.assert_allclose(values.T, _cubic(physical), atol=1.0e-11)

    # Continuity: every physical edge sample receives one value from all cells.
    edge_values = values.T[:, interior.shape[0] :].reshape((-1,))
    edge_locations = np.round(
        physical[:, interior.shape[0] :].reshape((-1, dimension)), decimals=9
    )
    _unique, inverse = np.unique(edge_locations, axis=0, return_inverse=True)
    inverse = inverse.reshape((-1,))
    shared = np.bincount(inverse) > 1
    assert np.any(shared)
    lower = np.full(shared.shape, np.inf)
    upper = np.full(shared.shape, -np.inf)
    np.minimum.at(lower, inverse, edge_values)
    np.maximum.at(upper, inverse, edge_values)
    assert np.max(upper[shared] - lower[shared]) <= 1.0e-11
