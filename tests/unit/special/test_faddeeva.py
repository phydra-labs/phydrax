import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import scipy.integrate
import scipy.special

import phydrax as phx


def _mp_wofz(value: complex) -> complex:
    with mp.workdps(80):
        z = mp.mpc(value.real, value.imag)
        return complex(mp.exp(-(z**2)) * mp.erfc(-1j * z))


def _mp_dawsn(value: float) -> mp.mpf:
    x = mp.mpf(value)
    return mp.exp(-(x**2)) * mp.quad(lambda t: mp.exp(t**2), [0, x])


def _mp_voigt(x: mp.mpf, sigma: mp.mpf, gamma: mp.mpf) -> mp.mpf:
    z = (x + 1j * gamma) / (sigma * mp.sqrt(2))
    return mp.re(mp.exp(-(z**2)) * mp.erfc(-1j * z)) / (sigma * mp.sqrt(2 * mp.pi))


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (jnp.complex64, 8e-7, 2e-7),
        (jnp.complex128, 5e-13, 5e-14),
    ],
)
def test_wofz_matches_scipy_across_complex_plane(dtype, rtol, atol):
    real = np.asarray([-40.0, -12.0, -6.0, -2.0, 0.0, 2.0, 6.0, 12.0, 40.0])
    imaginary = np.asarray([-6.0, -2.0, -0.1, 0.0, 0.1, 2.0, 6.0])
    arguments = (real[:, None] + 1j * imaginary[None, :]).astype(dtype)
    actual = np.asarray(phx.special.wofz(jnp.asarray(arguments)))
    expected = scipy.special.wofz(arguments)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_wofz_matches_high_precision_hard_points():
    arguments = np.asarray(
        [0.0j, 1.0 + 2.0j, -2.0 + 0.01j, 3.0 - 0.5j, -3.5 - 2.0j, -5.0j, 12.0 + 0.001j]
    )
    expected = np.asarray([_mp_wofz(value) for value in arguments])
    actual = np.asarray(phx.special.wofz(jnp.asarray(arguments)))
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=4e-14)


@pytest.mark.parametrize(
    ("dtype", "maximum", "rtol", "atol"),
    [
        (jnp.float32, 1e30, 4e-7, 2e-8),
        (jnp.float64, 1e300, 5e-15, 5e-16),
    ],
)
def test_dawsn_matches_scipy_across_real_domain(dtype, maximum, rtol, atol):
    positive = np.geomspace(1.0 / maximum, maximum, 241)
    values = np.concatenate(
        [
            -positive[::-1],
            np.linspace(-10.0, 10.0, 257),
            positive,
        ]
    ).astype(dtype)
    actual = np.asarray(phx.special.dawsn(jnp.asarray(values)))
    expected = scipy.special.dawsn(values)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("threshold", [3.25, 6.25])
def test_dawsn_regime_switches_are_value_and_derivative_continuous(threshold):
    values = jnp.asarray(
        [
            np.nextafter(threshold, -np.inf),
            threshold,
            np.nextafter(threshold, np.inf),
        ]
    )
    actual = np.asarray(phx.special.dawsn(values))
    expected = scipy.special.dawsn(np.asarray(values))
    np.testing.assert_allclose(actual, expected, rtol=4e-15, atol=5e-16)

    actual_derivative = np.asarray(jax.vmap(jax.grad(phx.special.dawsn))(values))
    expected_derivative = 1.0 - 2.0 * np.asarray(values) * expected
    np.testing.assert_allclose(
        actual_derivative, expected_derivative, rtol=2e-14, atol=2e-15
    )


def test_faddeeva_reflection_conjugation_and_real_axis_identities():
    z = jnp.asarray([0.2 + 0.4j, 2.0 + 1.0j, -1.5 + 0.2j])
    reflected = phx.special.wofz(-z)
    expected_reflected = 2.0 * jnp.exp(-(z**2)) - phx.special.wofz(z)
    np.testing.assert_allclose(
        np.asarray(reflected), np.asarray(expected_reflected), rtol=8e-13, atol=6e-14
    )

    conjugate_identity = jnp.conj(phx.special.wofz(z))
    np.testing.assert_allclose(
        np.asarray(conjugate_identity),
        np.asarray(phx.special.wofz(-jnp.conj(z))),
        rtol=8e-13,
        atol=6e-14,
    )

    x = jnp.linspace(-5.0, 5.0, 101)
    real_axis = phx.special.wofz(x)
    np.testing.assert_allclose(
        np.asarray(jnp.real(real_axis)),
        np.asarray(jnp.exp(-(x**2))),
        rtol=7e-12,
        atol=4e-14,
    )
    np.testing.assert_allclose(
        np.asarray(jnp.imag(real_axis)),
        np.asarray(2.0 * phx.special.dawsn(x) / math.sqrt(math.pi)),
        rtol=8e-13,
        atol=4e-14,
    )


def test_wofz_forward_reverse_and_higher_order_derivatives_agree():
    coefficient = 0.4 - 0.2j

    def observable(coordinates):
        z = coordinates[0] + 1j * coordinates[1]
        return jnp.real(coefficient * phx.special.wofz(z))

    coordinates = jnp.asarray([0.7, -0.3])
    forward = jax.jacfwd(observable)(coordinates)
    reverse = jax.jacrev(observable)(coordinates)
    np.testing.assert_allclose(np.asarray(forward), np.asarray(reverse), rtol=2e-14)

    reference_value = _mp_wofz(complex(*np.asarray(coordinates)))
    reference_derivative = -2.0 * complex(*np.asarray(coordinates)) * reference_value + (
        2.0j / math.sqrt(math.pi)
    )
    weighted_derivative = coefficient * reference_derivative
    expected = np.asarray([weighted_derivative.real, -weighted_derivative.imag])
    np.testing.assert_allclose(np.asarray(forward), expected, rtol=4e-13, atol=5e-14)

    forward_hessian = jax.jacfwd(jax.jacrev(observable))(coordinates)
    reverse_hessian = jax.jacrev(jax.jacfwd(observable))(coordinates)
    np.testing.assert_allclose(
        np.asarray(forward_hessian), np.asarray(reverse_hessian), rtol=3e-13, atol=5e-14
    )


@pytest.mark.parametrize("value", [-8.0, -3.25, -0.2, 0.0, 0.2, 6.25])
def test_dawsn_first_and_second_derivatives_match_high_precision(value):
    x = jnp.asarray(value)
    first = jax.grad(phx.special.dawsn)(x)
    second = jax.grad(jax.grad(phx.special.dawsn))(x)
    with mp.workdps(80):
        expected_first = float(mp.diff(_mp_dawsn, mp.mpf(value), 1))
        expected_second = float(mp.diff(_mp_dawsn, mp.mpf(value), 2))
    np.testing.assert_allclose(np.asarray(first), expected_first, rtol=3e-13, atol=5e-14)
    np.testing.assert_allclose(
        np.asarray(second), expected_second, rtol=8e-13, atol=8e-14
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (jnp.float32, 8e-7, 2e-7),
        (jnp.float64, 7e-13, 8e-14),
    ],
)
def test_voigt_profile_matches_scipy(dtype, rtol, atol):
    x = np.linspace(-30.0, 30.0, 241, dtype=np.dtype(dtype))[:, None]
    sigma = np.asarray([0.2, 0.8, 3.0], dtype=np.dtype(dtype))[None, :]
    gamma = np.asarray([0.0, 0.3, 2.0], dtype=np.dtype(dtype))[None, :]
    actual = np.asarray(
        phx.special.voigt_profile(jnp.asarray(x), jnp.asarray(sigma), jnp.asarray(gamma))
    )
    expected = scipy.special.voigt_profile(x, sigma, gamma)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    assert np.all(actual >= 0.0)


def test_voigt_profile_gradient_matches_high_precision_reference():
    parameters = jnp.asarray([0.7, 1.2, 0.4])
    actual = jax.grad(lambda args: phx.special.voigt_profile(*args))(parameters)
    with mp.workdps(80):
        x, sigma, gamma = (mp.mpf(value) for value in np.asarray(parameters))
        expected = np.asarray(
            [
                float(mp.diff(lambda value: _mp_voigt(value, sigma, gamma), x)),
                float(mp.diff(lambda value: _mp_voigt(x, value, gamma), sigma)),
                float(mp.diff(lambda value: _mp_voigt(x, sigma, value), gamma)),
            ]
        )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=8e-13, atol=8e-14)


def test_voigt_boundary_derivative_contract():
    x = jnp.asarray(0.5)
    gamma = jnp.asarray(1.0)
    sigma_tangent = jax.jvp(
        lambda sigma: phx.special.voigt_profile(x, sigma, gamma),
        (jnp.asarray(0.0),),
        (jnp.asarray(1.0),),
    )[1]
    assert sigma_tangent == 0.0

    x_derivative = jax.grad(lambda value: phx.special.voigt_profile(value, 0.0, gamma))(x)
    gamma_derivative = jax.grad(lambda value: phx.special.voigt_profile(x, 0.0, value))(
        gamma
    )
    denominator = math.pi * (float(x) ** 2 + float(gamma) ** 2) ** 2
    expected_x = -2.0 * float(x) * float(gamma) / denominator
    expected_gamma = (float(x) ** 2 - float(gamma) ** 2) / denominator
    np.testing.assert_allclose(np.asarray(x_derivative), expected_x, rtol=2e-15)
    np.testing.assert_allclose(np.asarray(gamma_derivative), expected_gamma, rtol=2e-15)

    point_tangent = jax.grad(lambda value: phx.special.voigt_profile(value, 0.0, 0.0))(
        1.0
    )
    invalid_tangent = jax.grad(lambda value: phx.special.voigt_profile(0.5, value, 1.0))(
        -1.0
    )
    assert jnp.isnan(point_tangent)
    assert jnp.isnan(invalid_tangent)


def test_voigt_profile_is_normalized():
    profile = jax.jit(lambda value: phx.special.voigt_profile(value, 0.8, 0.3))
    profile(jnp.asarray(0.0)).block_until_ready()
    area, error = scipy.integrate.quad(
        lambda value: float(profile(value)),
        -np.inf,
        np.inf,
        epsabs=2e-10,
        epsrel=2e-10,
        limit=200,
    )
    assert error < 2e-9
    assert area == pytest.approx(1.0, rel=2e-10, abs=2e-10)
