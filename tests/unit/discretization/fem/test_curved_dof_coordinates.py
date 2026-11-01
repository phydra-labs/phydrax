#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import itertools
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_mesh import CellBlock, CellMesh, SimplicialConnectivity


# "triangle" meshes carry polygonal connectivity; "simplex:2" meshes carry the
# simplicial complex the native mesher publishes.
_KINDS = ("triangle", "simplex:2")


def _triangles(n: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, n + 1)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((first.ravel(), second.ravel()), axis=-1)
    corner = (np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]).ravel()
    cells = np.concatenate(
        (
            np.stack((corner, corner + n + 1, corner + n + 2), axis=-1),
            np.stack((corner, corner + n + 2, corner + 1), axis=-1),
        )
    )
    return points, cells.astype(np.int32)


def _tetrahedra(n: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, n + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    stride = np.asarray(((n + 1) ** 2, n + 1, 1))
    origins = np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing="ij"), -1).reshape(
        -1, 3
    )
    cells = []
    for order in itertools.permutations(range(3)):
        path = [np.zeros(3, dtype=np.int64)]
        for axis_index in order:
            path.append(path[-1] + np.eye(3, dtype=np.int64)[axis_index])
        cells.append(np.stack([(origins + step) @ stride for step in path], axis=-1))
    cells = np.concatenate(cells)
    corners = points[cells]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
    cells[negative] = cells[negative][:, (0, 1, 3, 2)]
    return points, cells.astype(np.int32)


def _mesh(points: np.ndarray, cells: np.ndarray, kind: str) -> Any:
    if kind == "triangle":
        return CellMesh.from_triangles(jnp.asarray(points), jnp.asarray(cells))
    return CellMesh(
        points,
        (
            CellBlock(
                "cells",
                kind,
                cells,
                global_ids=tuple(range(cells.shape[0])),
            ),
        ),
    )


def _annulus(parameters: np.ndarray) -> np.ndarray:
    radius = 1.0 + parameters[..., 0]
    angle = 0.5 * np.pi * parameters[..., 1]
    return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)


def _warp(points: np.ndarray) -> np.ndarray:
    return points + 0.1 * np.sin(np.pi * points[..., ::-1])


def _curved(
    points: np.ndarray, cells: np.ndarray, kind: str, chart: Any
) -> tuple[Any, Any, np.ndarray]:
    """Mesh whose P2 coordinate map interpolates ``chart`` at P2 nodes."""
    parameter_mesh = _mesh(points, cells, kind)
    nodes = phx.discretization.FiniteElementDofMap(
        parameter_mesh, (phx.discretization.lagrange_element(kind, 2),)
    )
    node_parameters = np.asarray(nodes.dof_coordinates)
    mesh = _mesh(chart(points), cells, kind)
    geometry = phx.discretization.CellGeometrySpec(
        {mesh.blocks[0].name: phx.discretization.lagrange_element(kind, 2)},
        {mesh.blocks[0].name: np.asarray(nodes.cell_dofs[0])},
        chart(node_parameters),
    )
    return mesh, geometry, node_parameters


def _curved_annulus(kind: str) -> tuple[Any, Any, np.ndarray]:
    return _curved(*_triangles(3), kind, _annulus)


def _space(mesh: Any, geometry: Any, degree: int) -> Any:
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element(mesh.blocks[0].cell_kind, degree)
        ),
        coordinate_spec=geometry,
    ).prepare()


def _field_at_quadrature(space: Any, coefficients: Any) -> tuple[np.ndarray, np.ndarray]:
    (geometry,) = space.block_geometries[0]
    routes = np.asarray(space.dof_maps[0].cell_dofs[0])
    values = np.einsum(
        "qi,ci->cq",
        np.asarray(geometry.basis_values),
        np.asarray(coefficients)[routes],
    )
    return values, np.asarray(geometry.physical_points)


def _harmonic_x(space: Any) -> np.ndarray:
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "curved-harmonic", "u", (phx.equations.DiffusionAction("u", 1.0),)
        ),
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=lambda points: points[..., 0],
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(
        operator, rhs, policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    )
    assert bool(jnp.all(solved.successful))
    return np.asarray(problem.expand(solved.value))


@pytest.mark.parametrize("kind", _KINDS)
def test_curved_dof_points_lie_on_the_coordinate_map(kind: str) -> None:
    mesh, geometry, node_parameters = _curved_annulus(kind)
    space = _space(mesh, geometry, 2)
    dof_map = space.dof_maps[0]
    np.testing.assert_allclose(
        dof_map.dof_coordinates, _annulus(node_parameters), rtol=0.0, atol=4.0e-16
    )
    # Outer-arc edge nodes sit on r = 2, not on the straight chord.
    outer = np.isclose(node_parameters[:, 0], 1.0)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(dof_map.dof_coordinates)[outer], axis=1),
        2.0,
        rtol=4.0e-16,
    )
    np.testing.assert_array_equal(
        dof_map.evaluate_coordinates(space.default_runtime.coordinates),
        dof_map.dof_coordinates,
    )
    # Vertex nodes are the exact mesh vertices.
    np.testing.assert_array_equal(
        np.asarray(dof_map.dof_coordinates)[: mesh.coordinates.shape[0]],
        np.asarray(mesh.coordinates),
    )


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    ("degree", "function"),
    [
        (2, lambda x: x[..., 0]),
        (2, lambda x: 2.0 + 3.0 * x[..., 0] - x[..., 1]),
        # The P2 map makes x and y quadratic in reference coordinates, so this
        # quadratic is quartic there and lies exactly in the P4 space.
        (4, lambda x: 3.0 - x[..., 0] ** 2 + 0.5 * x[..., 0] * x[..., 1]),
    ],
    ids=("p2-x", "p2-linear", "p4-quadratic"),
)
def test_nodal_interpolation_reproduces_mapped_polynomials(
    degree: int, function: Any, kind: str
) -> None:
    mesh, geometry, _ = _curved_annulus(kind)
    space = _space(mesh, geometry, degree)
    coefficients = space.project("u", lambda points, _args: function(points))
    values, points = _field_at_quadrature(space, coefficients)
    np.testing.assert_allclose(values, function(points), rtol=0.0, atol=2.0e-14)


@pytest.mark.parametrize("kind", _KINDS)
def test_isoparametric_dirichlet_solve_reproduces_harmonic_x(kind: str) -> None:
    mesh, geometry, _ = _curved_annulus(kind)
    space = _space(mesh, geometry, 2)
    np.testing.assert_allclose(
        _harmonic_x(space),
        np.asarray(space.dof_maps[0].dof_coordinates)[:, 0],
        atol=2.0e-13,
    )


def test_simplicial_tetrahedra_carry_isoparametric_p2_fields() -> None:
    mesh, geometry, node_parameters = _curved(*_tetrahedra(2), "simplex:3", _warp)
    assert isinstance(mesh.connectivity, SimplicialConnectivity)
    space = _space(mesh, geometry, 2)
    dof_map = space.dof_maps[0]
    edge_count = mesh.connectivity.entities[1].shape[0]
    assert dof_map.global_dof_count == mesh.coordinates.shape[0] + edge_count
    np.testing.assert_allclose(
        dof_map.dof_coordinates, _warp(node_parameters), rtol=0.0, atol=4.0e-16
    )
    coefficients = space.project("u", lambda points, _args: points[..., 2])
    values, points = _field_at_quadrature(space, coefficients)
    np.testing.assert_allclose(values, points[..., 2], rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        _harmonic_x(space), np.asarray(dof_map.dof_coordinates)[:, 0], atol=2.0e-13
    )

    # P3 adds one interior DOF per triangular face, routed by simplicial faces;
    # z is quadratic in reference coordinates on the P2 map, so P3 reproduces it.
    cubic = _space(mesh, geometry, 3)
    face_count = mesh.connectivity.entities[2].shape[0]
    assert cubic.dof_maps[0].global_dof_count == (
        mesh.coordinates.shape[0] + 2 * edge_count + face_count
    )
    coefficients = cubic.project("u", lambda points, _args: points[..., 2])
    values, points = _field_at_quadrature(cubic, coefficients)
    np.testing.assert_allclose(values, points[..., 2], rtol=0.0, atol=2.0e-14)
