#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _square_plan():
    return phx.discretization.FiniteVolumePlan(
        jnp.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32),
        field_name="u",
    )


def _transport_problem(velocity=jnp.asarray([0.7, -0.2])):
    return phx.equations.ConservationProblemIR(
        "linear-transport",
        "u",
        lambda time, state, points, args: state[:, None] * velocity,
        lambda left, right, normals, args: jnp.abs(normals @ velocity),
        exterior_state=lambda time, interior, points, normals, args: interior,
    )


def test_finite_volume_preparation_orients_faces_and_cell_measures():
    discretization = _square_plan().prepare()
    interior = ~discretization.boundary_face_mask
    displacement = (
        discretization.cell_centroids[discretization.right_cells[interior]]
        - discretization.cell_centroids[discretization.left_cells[interior]]
    )

    assert discretization.field_spaces[0].representation == "cell_average"
    assert jnp.allclose(jnp.sum(discretization.cell_areas), 1.0)
    assert jnp.all(discretization.face_lengths > 0.0)
    assert jnp.all(
        jnp.sum(displacement * discretization.face_normals[interior], axis=-1) > 0.0
    )
    assert jnp.count_nonzero(interior) == 1


def test_first_order_finite_volume_preserves_constant_state():
    discretization = _square_plan().prepare()
    compiled = phx.equations.compile_conservation_problem(
        _transport_problem(),
        discretization,
    )

    derivative = compiled(jnp.asarray(0.0), jnp.full((2,), 3.0))

    assert jnp.allclose(derivative, 0.0, atol=1e-12)
    assert (
        compiled.discretization_bundle.record(discretization.key).artifact_id
        == discretization.prepared_id
    )


def test_finite_volume_internal_fluxes_cancel_globally():
    discretization = _square_plan().prepare()
    compiled = phx.equations.compile_conservation_problem(
        _transport_problem(),
        discretization,
    )
    state = jnp.asarray([1.0, -0.25])

    face_flux = compiled.face_flux(jnp.asarray(0.0), state)
    derivative = compiled(jnp.asarray(0.0), state)
    boundary_outflow = jnp.sum(
        jnp.where(
            discretization.boundary_face_mask,
            face_flux * discretization.face_lengths,
            0.0,
        )
    )

    assert jnp.allclose(
        jnp.sum(discretization.cell_areas * derivative),
        -boundary_outflow,
        atol=1e-12,
    )


def test_finite_volume_requires_explicit_boundary_state():
    discretization = _square_plan().prepare()
    with pytest.raises(ValueError, match="Boundary faces"):
        discretization.first_order_dynamics(
            lambda time, state, points, args: jnp.stack((state, state), axis=-1),
            lambda left, right, normals, args: 1.0,
        )


def test_finite_volume_geometry_is_differentiable_at_fixed_topology():
    plan = _square_plan()
    direction = jnp.zeros_like(plan.vertices).at[2, 0].set(1.0)

    area, tangent = jax.jvp(
        lambda vertices: jnp.sum(
            phx.discretization.triangular_finite_volume_geometry(
                vertices,
                plan.faces,
            )[0]
        ),
        (plan.vertices,),
        (direction,),
    )

    assert jnp.allclose(area, 1.0)
    assert jnp.isfinite(tangent)
    assert tangent != 0.0


def test_finite_volume_rejects_nonmanifold_faces():
    with pytest.raises(ValueError, match="edge-manifold"):
        phx.discretization.FiniteVolumePlan(
            jnp.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, -1.0],
                    [0.5, 1.0],
                ]
            ),
            jnp.asarray(
                [[0, 1, 2], [1, 0, 3], [0, 1, 4]],
                dtype=jnp.int32,
            ),
        )
