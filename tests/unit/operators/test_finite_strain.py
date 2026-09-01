#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.operators.mechanics import (
    cauchy_to_first_piola,
    finite_strain_kinematics,
    first_piola_to_cauchy,
    inverse_nanson_transform,
    nanson_response,
    nanson_transform,
    neo_hookean_first_piola,
    neo_hookean_response,
    neo_hookean_response_from_moduli,
    NeoHookeanLaw,
    NeoHookeanParameters,
    VolumetricConstraint,
)


def _deformation(dimension):
    if dimension == 2:
        return jnp.asarray([[1.08, 0.07], [0.02, 0.94]])
    return jnp.asarray([[1.08, 0.07, 0.01], [0.02, 0.94, 0.03], [0.0, 0.01, 1.04]])


def _parameters():
    return NeoHookeanParameters.from_shear_bulk(3.0, 11.0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_finite_strain_kinematics_energy_stress_and_tangent_are_ad_consistent(
    dimension,
):
    deformation = _deformation(dimension)
    parameters = _parameters()
    kinematics = finite_strain_kinematics(deformation)
    response = NeoHookeanLaw(parameters).evaluate(deformation)

    assert isinstance(response, type(neo_hookean_response(deformation, parameters)))
    assert bool(response.admissible)
    assert kinematics.dimension == dimension
    assert kinematics.kinematics == (
        "plane_strain" if dimension == 2 else "three_dimensional"
    )
    assert kinematics.deformation_gradient.shape == (3, 3)
    if dimension == 2:
        np.testing.assert_allclose(
            kinematics.deformation_gradient,
            jnp.eye(3).at[:2, :2].set(deformation),
            rtol=0.0,
            atol=0.0,
        )

    energy_gradient = jax.grad(
        lambda value: NeoHookeanLaw(parameters).evaluate(value).reference_energy_density
    )(deformation)
    np.testing.assert_allclose(
        energy_gradient,
        response.first_piola[:dimension, :dimension],
        rtol=2e-11,
        atol=2e-11,
    )

    ad_tangent = jax.jacfwd(lambda value: neo_hookean_first_piola(value, parameters))(
        deformation
    )
    np.testing.assert_allclose(
        ad_tangent,
        response.tangent[..., :dimension, :dimension],
        rtol=3e-11,
        atol=3e-11,
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_nanson_area_and_stress_transforms_are_exact_inverses(dimension):
    deformation = _deformation(dimension)
    parameters = _parameters()
    reference_area = (
        jnp.asarray((0.6, -0.8)) if dimension == 2 else jnp.asarray((0.3, -0.4, 0.5))
    )
    response = neo_hookean_response(deformation, parameters)

    current_area = nanson_transform(deformation, reference_area)
    restored_area = inverse_nanson_transform(deformation, current_area)
    evidence = nanson_response(deformation, reference_area)
    np.testing.assert_allclose(restored_area, reference_area, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(evidence.current_area_vector, current_area)
    assert bool(evidence.admissible)
    assert evidence.area_ratio > 0.0

    cauchy = first_piola_to_cauchy(response.kinematics, response.first_piola)
    restored_piola = cauchy_to_first_piola(response.kinematics, cauchy)
    np.testing.assert_allclose(cauchy, response.cauchy_stress, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        restored_piola, response.first_piola, rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize("kind", ["jacobian", "logarithmic"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_volumetric_constraint_derivative_matches_forward_ad(kind, dimension):
    deformation = _deformation(dimension)
    constraint = VolumetricConstraint(kind)
    value, derivative = constraint.evaluate(deformation)
    ad_derivative = jax.jacfwd(constraint.value)(deformation)

    assert jnp.isfinite(value)
    np.testing.assert_allclose(
        derivative[..., :dimension, :dimension],
        ad_derivative,
        rtol=2e-12,
        atol=2e-12,
    )


def test_finite_strain_and_material_admissibility_are_explicit():
    inverted = jnp.diag(jnp.asarray((-1.0, 1.0)))
    inverted_response = neo_hookean_response(inverted, _parameters())
    assert not bool(inverted_response.kinematic_admissible)
    assert bool(inverted_response.material_admissible)
    assert not bool(inverted_response.admissible)
    assert not bool(jnp.isfinite(inverted_response.reference_energy_density))
    assert not bool(jnp.all(jnp.isfinite(inverted_response.first_piola)))
    assert not bool(jnp.all(jnp.isfinite(inverted_response.tangent)))

    invalid_material = neo_hookean_response_from_moduli(jnp.eye(3), -1.0, 3.0)
    assert bool(invalid_material.kinematic_admissible)
    assert not bool(invalid_material.material_admissible)
    assert not bool(invalid_material.admissible)

    with pytest.raises(ValueError, match="implied bulk modulus"):
        NeoHookeanParameters(2.0, -2.0)
    with pytest.raises(ValueError, match="2x2 or 3x3"):
        finite_strain_kinematics(jnp.ones((2, 3)))


def test_canonical_moduli_kernels_support_batched_scalar_material_fields():
    deformations = jnp.stack((_deformation(2), 1.03 * _deformation(2)))
    shear = jnp.asarray((2.0, 4.0))
    lambda_ = jnp.asarray((3.0, 5.0))
    batched = neo_hookean_response_from_moduli(deformations, shear, lambda_)
    individual = tuple(
        neo_hookean_response_from_moduli(
            deformations[index],
            shear[index],
            lambda_[index],
        )
        for index in range(2)
    )

    np.testing.assert_allclose(
        batched.reference_energy_density,
        jnp.stack(tuple(response.reference_energy_density for response in individual)),
    )
    np.testing.assert_allclose(
        batched.first_piola,
        jnp.stack(tuple(response.first_piola for response in individual)),
    )
    np.testing.assert_allclose(
        batched.tangent,
        jnp.stack(tuple(response.tangent for response in individual)),
    )
