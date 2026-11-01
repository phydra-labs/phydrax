#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


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


def _annulus(parameters: np.ndarray) -> np.ndarray:
    radius = 1.0 + parameters[..., 0]
    angle = 0.5 * np.pi * parameters[..., 1]
    return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)


def _curved_annulus(n: int) -> tuple[Any, Any, np.ndarray]:
    """Quarter annulus whose P2 coordinate map interpolates the polar map."""
    parameters, cells = _triangles(n)
    parameter_mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(parameters), jnp.asarray(cells)
    )
    nodes = phx.discretization.FiniteElementDofMap(
        parameter_mesh, (phx.discretization.lagrange_element("triangle", 2),)
    )
    node_parameters = np.asarray(nodes.dof_coordinates)
    mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(_annulus(parameters)), jnp.asarray(cells)
    )
    geometry = phx.discretization.CellGeometrySpec(
        {mesh.blocks[0].name: phx.discretization.lagrange_element("triangle", 2)},
        {mesh.blocks[0].name: np.asarray(nodes.cell_dofs[0])},
        _annulus(node_parameters),
    )
    return mesh, geometry, node_parameters


def _space(mesh: Any, geometry: Any, degree: int) -> Any:
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", degree)
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


def test_curved_dof_points_lie_on_the_coordinate_map() -> None:
    mesh, geometry, node_parameters = _curved_annulus(3)
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
    degree: int, function: Any
) -> None:
    mesh, geometry, _ = _curved_annulus(3)
    space = _space(mesh, geometry, degree)
    coefficients = space.project("u", lambda points, _args: function(points))
    values, points = _field_at_quadrature(space, coefficients)
    np.testing.assert_allclose(values, function(points), rtol=0.0, atol=2.0e-14)


def test_isoparametric_dirichlet_solve_reproduces_harmonic_x() -> None:
    mesh, geometry, _ = _curved_annulus(3)
    space = _space(mesh, geometry, 2)
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "curved-annulus-harmonic", "u", (phx.equations.DiffusionAction("u", 1.0),)
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
    values = np.asarray(problem.expand(solved.value))
    np.testing.assert_allclose(
        values, np.asarray(space.dof_maps[0].dof_coordinates)[:, 0], atol=2.0e-13
    )
