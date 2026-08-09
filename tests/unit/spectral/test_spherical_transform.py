#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import s2fft
from s2fft.transforms import spherical as s2fft_spherical
from scipy.special import sph_harm_y

from phydrax._spectral._spherical import SphericalHarmonicPlan


def _real_bandlimited_field(plan):
    theta, phi = np.meshgrid(
        np.asarray(plan.theta),
        np.asarray(plan.phi),
        indexing="ij",
    )
    return jnp.asarray(
        np.real(sph_harm_y(2, 1, theta, phi))
        + 0.3 * np.real(sph_harm_y(3, -2, theta, phi))
        - 0.2 * np.real(sph_harm_y(1, 0, theta, phi))
    )


@pytest.mark.parametrize("sampling", ("mw", "mwss", "dh", "gl"))
def test_spherical_plan_roundtrips_sampling_theorems_and_integrates_constants(sampling):
    plan = SphericalHarmonicPlan(4, sampling=sampling)
    values = _real_bandlimited_field(plan)

    coefficients = plan.analysis(values)
    reconstructed = plan.synthesis(coefficients)
    sphere_measure = jnp.sum(plan.theta_quadrature_weights) * jnp.sum(
        plan.phi_quadrature_weights
    )

    assert values.shape == plan.sample_shape
    assert coefficients.shape == plan.coefficient_shape
    assert jnp.allclose(reconstructed, values, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(sphere_measure, 4.0 * jnp.pi, rtol=1e-12, atol=1e-12)


def test_spherical_plan_roundtrips_complex_spin_coefficients():
    plan = SphericalHarmonicPlan(5, spin=1, reality=False)
    degree = jnp.arange(plan.bandlimit)[:, None]
    order = jnp.arange(-(plan.bandlimit - 1), plan.bandlimit)[None, :]
    valid = (jnp.abs(order) <= degree) & (degree >= abs(plan.spin))
    coefficients = (
        jr.normal(jr.key(1), plan.coefficient_shape)
        + 1j * jr.normal(jr.key(2), plan.coefficient_shape)
    ) * valid

    actual = plan.analysis(plan.synthesis(coefficients))

    assert jnp.allclose(actual, coefficients, rtol=1e-11, atol=1e-11)


def test_spherical_plan_matches_s2fft_and_handles_batch_channel_axes():
    plan = SphericalHarmonicPlan(4, sampling="mw")
    first = _real_bandlimited_field(plan)
    values = jnp.stack(
        (
            jnp.stack((first, -0.5 * first), axis=-1),
            jnp.stack((0.25 * first, 2.0 * first), axis=-1),
        )
    )
    forward_precomputes = tuple(
        s2fft.generate_precomputes_jax(4, 0, "mw", None, True)
    )
    inverse_precomputes = tuple(
        s2fft.generate_precomputes_jax(4, 0, "mw", None, False)
    )
    expected_first = s2fft_spherical.forward_jax(
        first,
        4,
        0,
        None,
        "mw",
        True,
        forward_precomputes,
    )

    actual = eqx.filter_jit(lambda transform, field: transform.analysis(field))(
        plan, values
    )
    reconstructed = eqx.filter_jit(
        lambda transform, coefficients: transform.synthesis(coefficients)
    )(plan, actual)
    expected_reconstruction = s2fft_spherical.inverse_jax(
        expected_first,
        4,
        0,
        None,
        "mw",
        True,
        inverse_precomputes,
    )
    gradient = jax.grad(lambda field: jnp.sum(jnp.abs(plan.analysis(field)) ** 2))(
        first
    )

    assert actual.shape == (2, *plan.coefficient_shape, 2)
    assert jnp.allclose(actual[0, ..., 0], expected_first, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(reconstructed, values, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(expected_reconstruction, first, rtol=1e-11, atol=1e-11)
    assert jnp.all(jnp.isfinite(gradient))


def test_recursive_and_precomputed_spherical_plans_share_semantic_identity():
    recursive = SphericalHarmonicPlan(4, execution="recursive")
    precomputed = SphericalHarmonicPlan(4, execution="precomputed")
    values = _real_bandlimited_field(recursive)

    expected = recursive.analysis(values)
    actual = precomputed.analysis(values)

    assert recursive.fingerprint == precomputed.fingerprint
    assert jnp.allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(
        precomputed.synthesis(actual),
        recursive.synthesis(expected),
        rtol=1e-11,
        atol=1e-11,
    )


def test_spherical_plan_rejects_invalid_configuration_shapes_and_memory():
    with pytest.raises(ValueError, match="exceed the absolute spin"):
        SphericalHarmonicPlan(2, spin=2, reality=False)
    with pytest.raises(ValueError, match="spin-zero"):
        SphericalHarmonicPlan(4, spin=1, reality=True)
    with pytest.raises(ValueError, match="sampling"):
        SphericalHarmonicPlan(4, sampling="healpix")
    with pytest.raises(ValueError, match="execution"):
        SphericalHarmonicPlan(4, execution="dense")
    with pytest.raises(ValueError, match="max_precompute_bytes"):
        SphericalHarmonicPlan(4, max_precompute_bytes=1)

    plan = SphericalHarmonicPlan(4)
    with pytest.raises(ValueError, match="Spherical analysis expects"):
        plan.analysis(jnp.ones((4, 8)))
    with pytest.raises(ValueError, match="Spherical synthesis expects"):
        plan.synthesis(jnp.ones((4, 8)))
    with pytest.raises(TypeError, match="requires real values"):
        plan.analysis(jnp.ones(plan.sample_shape, dtype=complex))
