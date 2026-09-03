#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _two_tetrahedra():
    coordinates = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    cells = jnp.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=jnp.int32)
    return phx.discretization.CellMesh.from_tetrahedra(coordinates, cells)


def test_p2_tetrahedral_entity_routes_share_vertices_and_edges():
    mesh = _two_tetrahedra()
    element = phx.discretization.lagrange_element("tetrahedron", 2)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec("u", element),
    ).prepare()
    dof_map = discretization.dof_maps[0]
    routes = np.asarray(dof_map.cell_dofs[0])

    assert dof_map.association == "entity"
    assert dof_map.global_dof_count == 14
    assert dof_map.entity_dof_counts == (5, 9, 0, 0)
    assert dof_map.entity_dofs_per_entity == (1, 1, 1, 1)
    assert len(np.intersect1d(routes[0], routes[1])) == 6
    assert np.all(routes >= 0)
    assert np.all(routes < dof_map.global_dof_count)

    edges = np.asarray(mesh.connectivity.edges)
    expected_midpoints = np.mean(np.asarray(mesh.coordinates)[edges], axis=1)
    np.testing.assert_allclose(
        np.asarray(dof_map.dof_coordinates)[mesh.coordinates.shape[0] :],
        expected_midpoints,
    )


def test_curved_p2_tetrahedral_coordinates_drive_tensor_diffusion():
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    )
    element = phx.discretization.lagrange_element("tetrahedron", 2)
    reference = jnp.asarray(element.reference_nodes)
    curved = reference.at[:, 2].add(0.05 * reference[:, 0] * (1.0 - reference[:, 0]))
    coordinate_spec = phx.discretization.FiniteElementCoordinateSpec(
        {"tetrahedra": element},
        {"tetrahedra": jnp.arange(10, dtype=jnp.int32)[None, :]},
        curved,
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec("u", element),
        coordinate_spec=coordinate_spec,
    ).prepare(numeric_version="curved-p2-tetrahedron")
    properties = phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
        },
    )
    action = phx.equations.TensorDiffusionAction(
        "u",
        jnp.diag(jnp.asarray((2.0, 1.0, 0.5))),
        properties=properties,
        action_id="curved-p2-tetrahedral-diffusion",
    )
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(action.action_id, "u", (action,)),
        discretization,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(realization="sparse"),
    )
    operator = compiled.affine_operator()
    state = jnp.linspace(-0.5, 0.7, 10)

    assert discretization.dof_maps[0].global_dof_count == 10
    np.testing.assert_allclose(discretization.default_runtime.coordinates, curved)
    assert jnp.all(jnp.isfinite(operator.mv(state)))
    assert jnp.max(jnp.abs(operator.mv(jnp.ones((10,))))) <= 2.0e-6
    assert operator.properties.self_adjoint is True
    assert operator.properties.positive_semidefinite is True
