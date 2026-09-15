#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import CylindricalHankelPlan, CylindricalHankelStatus


def test_bessel_zero_hankel_pair_is_invertible_and_parseval_consistent():
    prepared = CylindricalHankelPlan(2.5, 24).prepare()
    evidence = prepared.evidence
    assert bool(evidence.successful)
    assert int(evidence.status) == int(CylindricalHankelStatus.SUCCESS)

    radius = prepared.radial_coordinates
    values = jnp.exp(-0.8 * radius * radius) * (1.0 + 0.2j * radius)
    transformed = prepared.forward(values)
    reconstructed = prepared.inverse(transformed)
    np.testing.assert_allclose(reconstructed, values, rtol=3.0e-5, atol=3.0e-6)

    physical_norm = jnp.sum(prepared.radial_weights * jnp.abs(values) ** 2)
    spectral_norm = jnp.sum(prepared.spectral_weights * jnp.abs(transformed) ** 2)
    np.testing.assert_allclose(spectral_norm, physical_norm, rtol=3.0e-5, atol=3.0e-6)


def test_hankel_application_preserves_unselected_axes_and_has_fixed_gradient():
    prepared = CylindricalHankelPlan(1.0, 10, order=1).prepare()
    values = jnp.arange(30.0).reshape(3, 10)
    transformed = prepared.normalized(values, axis=1)
    reconstructed = prepared.normalized(transformed, axis=1)
    assert transformed.shape == values.shape
    np.testing.assert_allclose(reconstructed, values, rtol=3.0e-5, atol=3.0e-5)

    gradient = jax.grad(
        lambda scale: jnp.real(jnp.sum(prepared.normalized(scale * values)))
    )(1.0)
    assert bool(jnp.isfinite(gradient))


def test_hankel_plan_refuses_invalid_geometry_and_resource_requests():
    with pytest.raises(ValueError):
        CylindricalHankelPlan(0.0, 8)
    with pytest.raises(ValueError):
        CylindricalHankelPlan(1.0, 1)
    with pytest.raises(ValueError):
        CylindricalHankelPlan(1.0, 8, order=-1)
    with pytest.raises(ValueError):
        CylindricalHankelPlan(1.0, 8, maximum_matrix_elements=63)


def test_hankel_plan_rejects_wrong_runtime_axis():
    prepared = CylindricalHankelPlan(1.0, 8).prepare()
    with pytest.raises(ValueError):
        prepared.forward(jnp.ones((7,)))
    with pytest.raises(ValueError):
        prepared.forward(jnp.ones((8,)), axis=2)
