import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

import phydrax as phx


cosmology = phx.applications.cosmology
geophysics = phx.applications.geophysics


def test_flat_radial_kernel_table_matches_scipy_including_tiny_higher_modes():
    plan = cosmology.FlatRadialKernelPlan(12)
    argument = np.asarray([0.0, 1.0e-8, 0.3, 2.0, 15.0])
    actual = np.asarray(plan.evaluate(jnp.asarray(argument)))
    expected = np.stack(
        [scipy.special.spherical_jn(order, argument) for order in range(13)]
    )

    assert actual.shape == (13, 5)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-11, atol=2.0e-14)
    np.testing.assert_array_equal(actual[:, 0], [1.0] + [0.0] * 12)
    assert np.all(actual[1:7, 1] != 0.0)

    compiled = jax.jit(plan.evaluate)(jnp.asarray(argument))
    np.testing.assert_allclose(compiled, expected, rtol=3.0e-11, atol=2.0e-14)


def test_digital_hankel_transform_matches_independent_cylindrical_bessel_sum():
    wave = np.asarray([0.1, 0.35, 0.8, 1.6, 3.0, 5.0])
    weights = np.asarray([0.07, 0.11, 0.19, 0.23, 0.17, 0.09])
    kernel = np.asarray(
        [
            np.exp(-0.4 * wave),
            (1.0 + 0.2 * wave) * np.exp(-0.7 * wave),
        ]
    )
    offsets = np.asarray([0.0, 0.2, 1.3, 4.0])

    for order in (0, 1):
        plan = geophysics.DigitalHankelTransformPlan(wave, weights, order)
        actual = np.asarray(plan.evaluate(jnp.asarray(kernel), jnp.asarray(offsets)))
        bessel = scipy.special.jv(order, offsets[:, None] * wave[None, :])
        expected = np.sum(kernel[:, None, :] * bessel[None, :, :] * weights, axis=-1)

        assert actual.shape == (2, 4)
        assert plan.wavenumbers_m_inverse.shape == wave.shape
        assert plan.weights_m_inverse.shape == weights.shape
        np.testing.assert_allclose(actual, expected, rtol=3.0e-11, atol=3.0e-14)

        compiled = eqx.filter_jit(plan.evaluate)(
            jnp.asarray(kernel), jnp.asarray(offsets)
        )
        np.testing.assert_allclose(compiled, expected, rtol=3.0e-11, atol=3.0e-14)
