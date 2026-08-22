#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _square_plan():
    vertices = jnp.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    faces = jnp.asarray(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=jnp.int32,
    )
    return phx.discretization.P1FiniteElementPlan(
        vertices,
        faces,
        field_name="u",
    )


def test_p1_preparation_assembles_measure_pairing_and_sparse_forms():
    discretization = _square_plan().prepare()
    ones = jnp.ones((5,))

    assert isinstance(
        discretization,
        phx.discretization.P1FiniteElementDiscretization,
    )
    assert discretization.field_spaces[0].conformity == "H1"
    assert jnp.allclose(jnp.sum(discretization.areas), 1.0)
    assert jnp.allclose(jnp.sum(discretization.mass.mv(ones)), 1.0)
    assert jnp.allclose(discretization.stiffness.mv(ones), 0.0, atol=1e-12)
    assert discretization.mass.relation.route_shape == (36,)
    assert discretization.mass.sparse_storage().nnz == 21
    assert discretization.stiffness.sparse_storage().nnz == 21
    assert jnp.count_nonzero(discretization.boundary_vertex_mask) == 4


def test_variational_compiler_reproduces_affine_dirichlet_solution():
    discretization = _square_plan().prepare()
    problem = phx.equations.VariationalProblemIR(
        "affine-laplace",
        "u",
        diffusion=1.0,
        source=lambda points: jnp.zeros(points.shape[:-1]),
        dirichlet=lambda points: points[..., 0] + points[..., 1],
    )
    compiled = phx.equations.compile_variational_problem(problem, discretization)

    solution, linear_result = compiled.solve()
    expected = discretization.vertices[:, 0] + discretization.vertices[:, 1]

    assert jnp.all(linear_result.successful)
    assert jnp.allclose(solution, expected, rtol=1e-10, atol=1e-10)
    assert (
        compiled.discretization_bundle.record(discretization.key).artifact_id
        == discretization.prepared_id
    )


def test_p1_neumann_loading_and_reconstruction_preserve_integrals_and_affine_fields():
    discretization = _square_plan().prepare()
    boundary_load = discretization.assemble_boundary_load(
        jnp.ones((discretization.boundary_edges.shape[0],))
    )
    field = discretization.vertices[:, 0] - 2.0 * discretization.vertices[:, 1]
    barycentric = jnp.full((discretization.faces.shape[0], 3), 1.0 / 3.0)
    reconstructed = discretization.reconstruct(field, barycentric)
    centroids = jnp.mean(discretization.vertices[discretization.faces], axis=1)

    assert jnp.allclose(jnp.sum(boundary_load), 4.0)
    assert jnp.allclose(
        reconstructed,
        centroids[:, 0] - 2.0 * centroids[:, 1],
    )


def test_p1_geometry_kernels_are_differentiable_under_fixed_topology():
    plan = _square_plan()
    direction = jnp.zeros_like(plan.vertices).at[2, 0].set(1.0)

    area, tangent = jax.jvp(
        lambda vertices: jnp.sum(
            phx.discretization.p1_local_matrices(vertices, plan.faces)[0]
        ),
        (plan.vertices,),
        (direction,),
    )

    assert jnp.allclose(area, 1.0)
    assert jnp.isfinite(tangent)
    assert tangent != 0.0


def test_p1_heat_dynamics_preserves_constant_states_without_source():
    discretization = _square_plan().prepare()
    heat = discretization.heat_dynamics(0.25)

    drift = heat(jnp.asarray(0.0), jnp.ones((5,)), None)

    assert jnp.allclose(drift, 0.0, atol=1e-10)


def test_p1_rejects_degenerate_and_unconstrained_components():
    with pytest.raises(ValueError, match="positive metric determinant|degenerate"):
        phx.discretization.P1FiniteElementPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        ).prepare()

    discretization = _square_plan().prepare()
    mask = jnp.asarray([True, False, False, False, False, False])
    with pytest.raises(ValueError, match="connected mesh component"):
        disconnected = phx.discretization.P1FiniteElementPlan(
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
        ).prepare()
        disconnected.dirichlet(boundary_mask=mask)

    with pytest.raises(ValueError, match="proper subset"):
        discretization.dirichlet(boundary_mask=jnp.ones((5,), dtype=bool))
