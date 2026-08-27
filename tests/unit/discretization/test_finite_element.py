#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _square_mesh():
    vertices = jnp.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    cells = jnp.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=jnp.int32,
    )
    return phx.discretization.CellMesh.from_triangles(vertices, cells)


def _square_discretization(*, degree=1):
    mesh = _square_mesh()
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", degree),
    )
    return phx.discretization.FiniteElementPlan(mesh, field).prepare()


def test_reference_elements_partition_unity_and_reproduce_coordinates():
    cases = (
        ("triangle", 1, jnp.asarray([[0.2, 0.3]])),
        ("triangle", 2, jnp.asarray([[0.2, 0.3]])),
        ("quadrilateral", 1, jnp.asarray([[0.2, 0.3]])),
        ("tetrahedron", 1, jnp.asarray([[0.1, 0.2, 0.3]])),
    )
    for cell, degree, points in cases:
        element = phx.discretization.lagrange_element(cell, degree)
        values, gradients = element.tabulate(points)
        reconstructed = values @ element.reference_nodes

        assert jnp.allclose(jnp.sum(values, axis=-1), 1.0)
        assert jnp.allclose(jnp.sum(gradients, axis=1), 0.0)
        assert jnp.allclose(reconstructed, points)


def test_generic_preparation_assembles_mass_stiffness_and_p2_dofs():
    p1 = _square_discretization()
    p2 = _square_discretization(degree=2)
    ones = jnp.ones((p1.dof_maps[0].global_dof_count,))

    assert p1.field_spaces[0].conformity == "H1"
    assert jnp.allclose(jnp.sum(p1.measures[0].weights), 1.0)
    assert jnp.allclose(jnp.sum(p1.mass.mv(ones)), 1.0)
    assert jnp.allclose(p1.stiffness.mv(ones), 0.0, atol=1e-12)
    assert p1.mass.sparse_storage().nnz > 0
    assert p2.dof_maps[0].global_dof_count == 13
    assert jnp.count_nonzero(p2.boundary_dof_mask) == 8


def test_variational_compiler_reproduces_affine_dirichlet_solution():
    discretization = _square_discretization()
    constraint = phx.discretization.dirichlet_constraint(discretization, "u")
    form = phx.equations.WeakForm(
        "affine-laplace",
        "u",
        (
            phx.equations.DiffusionTerm("u"),
            phx.equations.SourceTerm("u", 0.0),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=lambda points: points[..., 0] + points[..., 1],
    )
    system, right_hand_side = compiled.linear_system()
    result = phx.linalg.solve(system, right_hand_side)
    solution = compiled.expand(result.value)
    expected = discretization.vertices[:, 0] + discretization.vertices[:, 1]

    assert jnp.all(result.successful)
    assert jnp.allclose(solution, expected, rtol=1e-10, atol=1e-10)
    assert (
        compiled.discretization_bundle.record(discretization.key).artifact_id
        == discretization.prepared_id
    )
    assert isinstance(compiled.residual_space, phx.linalg.DualSpace)


def test_boundary_loading_reconstruction_and_functional_preserve_integrals():
    discretization = _square_discretization()
    form = phx.equations.WeakForm(
        "boundary-load",
        "u",
        (phx.equations.BoundaryLoadTerm("u", 1.0),),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    load = -compiled.full_residual(jnp.zeros((5,)), None)
    field = discretization.vertices[:, 0] - 2.0 * discretization.vertices[:, 1]
    reconstructed = discretization.reconstruct(
        "u", field, "triangles", jnp.asarray([[1.0 / 3.0, 1.0 / 3.0]])
    )
    centroids = jnp.mean(
        discretization.vertices[discretization.mesh.blocks[0].vertices], axis=1
    )
    functional = phx.equations.FiniteElementFunctional(
        "integral-u",
        "u",
        lambda values, gradients, points, args: values,
    )

    assert jnp.allclose(jnp.sum(load), 4.0)
    assert jnp.allclose(
        reconstructed[:, 0],
        centroids[:, 0] - 2.0 * centroids[:, 1],
    )
    assert jnp.allclose(functional.evaluate(discretization, jnp.ones((5,))), 1.0)


def test_fixed_topology_geometry_is_differentiable():
    discretization = _square_discretization()
    direction = jnp.zeros_like(discretization.vertices).at[2, 0].set(1.0)

    area, tangent = jax.jvp(
        lambda vertices: jnp.sum(
            discretization.evaluate_geometry("u", vertices)[0].measure
        ),
        (discretization.vertices,),
        (direction,),
    )

    assert jnp.allclose(area, 1.0)
    assert jnp.isfinite(tangent)
    assert tangent != 0.0


def test_native_dae_adapter_preserves_constant_heat_state():
    discretization = _square_discretization()
    form = phx.equations.WeakForm(
        "heat",
        "u",
        (phx.equations.DiffusionTerm("u", 0.25),),
    )
    dae = phx.equations.compile_finite_element_problem(
        form, discretization
    ).as_dae_system()

    residual = dae(
        jnp.asarray(0.0),
        jnp.ones((5,)),
        jnp.zeros((5,)),
        None,
    )

    assert jnp.allclose(residual, 0.0, atol=1e-10)


def test_finite_elements_reject_degenerate_and_unconstrained_components():
    degenerate = phx.discretization.CellMesh.from_triangles(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="positive finite metric determinant"):
        phx.discretization.FiniteElementPlan(
            degenerate,
            phx.discretization.FiniteElementFieldSpec(
                "u", phx.discretization.lagrange_element("triangle", 1)
            ),
        ).prepare()

    disconnected_mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [3.0, 1.0],
            ]
        ),
        jnp.asarray([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32),
    )
    disconnected = phx.discretization.FiniteElementPlan(
        disconnected_mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    with pytest.raises(ValueError, match="connected mesh component"):
        phx.discretization.dirichlet_constraint(
            disconnected,
            "u",
            boundary_mask=jnp.asarray([True, False, False, False, False, False]),
        )

    with pytest.raises(ValueError, match="proper subset"):
        phx.discretization.dirichlet_constraint(
            _square_discretization(),
            "u",
            boundary_mask=jnp.ones((5,), dtype=bool),
        )
