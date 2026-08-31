#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _parameters():
    return phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
        3.0, 11.0
    )


def _embedded(deformation):
    if deformation.shape[-2:] == (3, 3):
        return deformation
    return jnp.eye(3, dtype=deformation.dtype).at[:2, :2].set(deformation)


@pytest.mark.parametrize("dimension", [2, 3])
def test_neo_hookean_form_density_and_ad_residual_match_constitutive_model(dimension):
    parameters = _parameters()
    form = phx.applications.solid_mechanics.neo_hookean_form("u", parameters)
    action = form.actions[0]
    assert isinstance(action, phx.equations.CellEnergyAction)

    displacement_gradient = (
        jnp.asarray([[0.08, 0.02], [0.05, -0.04]])
        if dimension == 2
        else jnp.asarray([[0.08, 0.02, 0.01], [0.05, -0.04, 0.03], [0.0, 0.02, 0.06]])
    )
    executor_gradient = displacement_gradient.T[None, None]
    values = jnp.zeros((1, 1, dimension))
    points = jnp.zeros((1, 1, dimension))

    def total_energy(gradient):
        return jnp.sum(action.density(values, gradient, points, None))

    actual_energy = total_energy(executor_gradient)
    actual_derivative = jax.grad(total_energy)(executor_gradient)[0, 0]
    deformation = jnp.eye(dimension) + displacement_gradient
    deformation_3d = _embedded(deformation)
    expected_energy = phx.applications.solid_mechanics.neo_hookean_reference_energy(
        deformation_3d, parameters
    )
    expected_piola = phx.applications.solid_mechanics.neo_hookean_first_piola(
        deformation_3d, parameters
    )[:dimension, :dimension]

    np.testing.assert_allclose(actual_energy, expected_energy, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        actual_derivative, expected_piola.T, rtol=2e-11, atol=2e-11
    )


def test_neo_hookean_form_compiles_vector_plane_strain_identity_residual():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = jnp.asarray([[0, 1, 3], [1, 2, 3]], dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_triangles(vertices, cells)
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", 1),
        component_shape=(2,),
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.applications.solid_mechanics.neo_hookean_form("u", _parameters())
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    residual = compiled.residual(compiled.state_space.zeros())

    for leaf in jax.tree.leaves(residual):
        np.testing.assert_allclose(leaf, 0.0, atol=2e-12)


def test_neo_hookean_form_rejects_incompatible_component_dimension():
    form = phx.applications.solid_mechanics.neo_hookean_form("u", _parameters())
    action = form.actions[0]
    values = jnp.zeros((1, 1, 3))
    gradients = jnp.zeros((1, 1, 2, 3))
    points = jnp.zeros((1, 1, 2))

    with pytest.raises(ValueError, match="2x2 or 3x3"):
        action.density(values, gradients, points, None)
