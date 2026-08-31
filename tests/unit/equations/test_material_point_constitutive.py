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


def _empty(batch_shape):
    return jnp.empty(batch_shape + (0,), dtype=jnp.float64)


def test_neo_hookean_physical_bulk_factory_and_identity_response():
    parameters = _parameters()
    assert parameters.shear_modulus == pytest.approx(3.0)
    assert parameters.bulk_modulus == pytest.approx(11.0)
    assert parameters.lame_lambda == pytest.approx(9.0)

    plan = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
    deformation = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))
    response = plan.evaluate(
        deformation,
        _empty((2,)),
        jnp.asarray((2.0, 8.0)),
        parameters,
        0.0,
        0.01,
    )

    assert jnp.all(response.successful)
    assert jnp.all(response.admissible)
    np.testing.assert_allclose(response.first_piola, 0.0, atol=1e-13)
    np.testing.assert_allclose(response.reference_energy_density, 0.0, atol=1e-13)
    expected = jnp.sqrt((11.0 + 4.0) / jnp.asarray((2.0, 8.0)))
    np.testing.assert_allclose(response.maximum_wave_speed, expected, rtol=1e-13)
    assert response.trial_state.shape == (2, 0)


def test_neo_hookean_energy_gradient_is_first_piola_and_parameters_differentiate():
    parameters = _parameters()
    deformation = jnp.asarray([[1.08, 0.07, 0.0], [0.02, 0.94, 0.03], [0.0, 0.01, 1.04]])
    stress = phx.applications.solid_mechanics.neo_hookean_first_piola(
        deformation, parameters
    )
    gradient = jax.grad(
        lambda value: phx.applications.solid_mechanics.neo_hookean_reference_energy(
            value, parameters
        )
    )(deformation)
    np.testing.assert_allclose(stress, gradient, rtol=2e-11, atol=2e-11)

    plan = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)

    def energy(material):
        return plan.evaluate(
            deformation[None],
            _empty((1,)),
            jnp.asarray((2.5,)),
            material,
            0.0,
            0.01,
        ).reference_energy_density[0]

    material_gradient = jax.grad(energy)(parameters)
    assert jnp.isfinite(material_gradient.shear_modulus)
    assert jnp.isfinite(material_gradient.lame_lambda)


def test_plane_strain_matches_embedded_three_dimensional_response():
    parameters = _parameters()
    deformation_2d = jnp.asarray([[[1.05, 0.08], [0.03, 0.97]]])
    embedded = jnp.asarray([[[1.05, 0.08, 0.0], [0.03, 0.97, 0.0], [0.0, 0.0, 1.0]]])
    plane = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
    spatial = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
    plane_result = plane.evaluate(
        deformation_2d,
        _empty((1,)),
        jnp.asarray((4.0,)),
        parameters,
        0.0,
        0.01,
    )
    spatial_result = spatial.evaluate(
        embedded,
        _empty((1,)),
        jnp.asarray((4.0,)),
        parameters,
        0.0,
        0.01,
    )

    np.testing.assert_allclose(
        plane_result.first_piola,
        spatial_result.first_piola[..., :2, :2],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        plane_result.reference_energy_density,
        spatial_result.reference_energy_density,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_finite_deformation_wave_bound_dominates_sampled_acoustic_speeds(dimension):
    parameters = _parameters()
    deformation = (
        jnp.asarray([[1.15, 0.12], [0.04, 0.88]])
        if dimension == 2
        else jnp.asarray([[1.15, 0.12, 0.02], [0.04, 0.88, 0.03], [0.01, 0.02, 1.07]])
    )
    density = jnp.asarray((2.7,))
    plan = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(dimension)
    response = plan.evaluate(
        deformation[None],
        _empty((1,)),
        density,
        parameters,
        0.0,
        0.01,
    )

    if dimension == 2:
        embedded = jnp.eye(3).at[:2, :2].set(deformation)
    else:
        embedded = deformation
    inverse_transpose = jnp.linalg.inv(embedded).T[:dimension, :dimension]
    coefficient = (
        parameters.lame_lambda * (1.0 - jnp.log(jnp.linalg.det(embedded)))
        + parameters.shear_modulus
    )
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 257)[:-1]
    if dimension == 2:
        directions = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
    else:
        z = jnp.linspace(-1.0, 1.0, 33)
        azimuth = jnp.linspace(0.0, 2.0 * jnp.pi, 65)[:-1]
        zz, aa = jnp.meshgrid(z, azimuth, indexing="ij")
        radial = jnp.sqrt(jnp.maximum(1.0 - zz**2, 0.0))
        directions = jnp.stack(
            (radial * jnp.cos(aa), radial * jnp.sin(aa), zz), axis=-1
        ).reshape((-1, 3))

    def acoustic_speed(direction):
        mapped = inverse_transpose @ direction
        acoustic = parameters.shear_modulus * jnp.eye(
            dimension
        ) + coefficient * jnp.outer(mapped, mapped)
        eigenvalue = jnp.max(jnp.linalg.eigvalsh(acoustic))
        return jnp.sqrt(jnp.maximum(eigenvalue, 0.0) / density[0])

    sampled_maximum = jnp.max(jax.vmap(acoustic_speed)(directions))
    assert response.maximum_wave_speed[0] >= sampled_maximum - 1e-12


def test_neo_hookean_mpm_rejects_nonpositive_jacobian():
    plan = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
    response = plan.evaluate(
        jnp.asarray([[[-1.0, 0.0], [0.0, 1.0]]]),
        _empty((1,)),
        jnp.asarray((1.0,)),
        _parameters(),
        0.0,
        0.01,
    )

    assert not bool(response.successful[0])
    assert not bool(response.admissible[0])
    np.testing.assert_allclose(response.first_piola, 0.0)
    np.testing.assert_allclose(response.reference_energy_density, 0.0)
