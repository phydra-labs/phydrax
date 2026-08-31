#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _mesh():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = jnp.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=jnp.int32)
    return phx.discretization.CellMesh.from_triangles(vertices, cells)


def test_allen_cahn_accepted_step_decreases_free_energy():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _mesh(), phx.discretization.FiniteElementFieldSpec("eta", element)
    ).prepare()
    parameters = phx.applications.phase_field.AllenCahnParameters(
        1.0,
        phx.equations.BinaryThermodynamicParameters(1.0, 0.02),
    )
    result = phx.applications.phase_field.solve_allen_cahn_step(
        discretization,
        "eta",
        jnp.full((5,), 0.2),
        0.01,
        parameters,
    )

    assert bool(result.successful)
    assert result.energy_after < result.energy_before


def test_cahn_hilliard_step_preserves_mass():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _mesh(),
        (
            phx.discretization.FiniteElementFieldSpec("c", element),
            phx.discretization.FiniteElementFieldSpec("mu", element),
        ),
    ).prepare()
    parameters = phx.applications.phase_field.CahnHilliardParameters(
        1.0,
        phx.equations.BinaryThermodynamicParameters(1.0, 0.02),
    )
    result = phx.applications.phase_field.solve_cahn_hilliard_step(
        discretization,
        "c",
        "mu",
        jnp.full((5,), 0.2),
        jnp.full((5,), -0.192),
        0.01,
        parameters,
    )

    assert bool(result.successful)
    assert jnp.abs(result.mass_after - result.mass_before) < 1.0e-12
