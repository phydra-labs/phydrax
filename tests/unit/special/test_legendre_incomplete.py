import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import scipy.special

import phydrax as phx


def test_incomplete_first_and_second_kind_match_scipy_over_periods():
    amplitudes = np.asarray([-20.0, -7.2, -math.pi / 2, -0.3, 0.0, 0.8, 4.0, 19.0])
    parameters = np.asarray([-2.0, -0.5, 0.0, 0.2, 0.8, 0.99])
    phi, m = np.meshgrid(amplitudes, parameters, indexing="ij")
    for function, reference in [
        (phx.special.ellipkinc, scipy.special.ellipkinc),
        (phx.special.ellipeinc, scipy.special.ellipeinc),
    ]:
        actual = np.asarray(function(jnp.asarray(phi), jnp.asarray(m)))
        expected = reference(phi, m)
        np.testing.assert_allclose(actual, expected, rtol=8e-13, atol=3e-14)


def test_incomplete_third_kind_matches_high_precision_reference():
    characteristics = np.asarray([-3.0, -0.2, 0.0, 0.4, 0.9])
    amplitudes = np.asarray([-7.0, -1.2, 0.2, 1.4, 8.0])
    parameters = np.asarray([-1.0, 0.0, 0.3, 0.8, 0.99])
    with mp.workdps(70):
        expected = np.asarray(
            [
                float(mp.ellippi(n, phi, m))
                for n, phi, m in zip(characteristics, amplitudes, parameters, strict=True)
            ]
        )
    actual = np.asarray(
        phx.special.ellippiinc(
            jnp.asarray(characteristics),
            jnp.asarray(amplitudes),
            jnp.asarray(parameters),
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=3e-14)


def test_incomplete_third_kind_large_negative_characteristic_is_factored():
    characteristic = -1e300
    amplitudes = np.asarray([0.3, 4.0, -7.0])
    periods = np.floor((amplitudes + 0.5 * math.pi) / math.pi)
    reduced = amplitudes - periods * math.pi
    scale = np.sqrt(1.0 - characteristic)
    expected = (
        np.arctan2(scale * np.sin(reduced), np.cos(reduced)) + periods * math.pi
    ) / scale
    actual = phx.special.ellippiinc(
        characteristic,
        jnp.asarray(amplitudes),
        0.0,
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3e-15)

    amplitude = 0.3
    expected_amplitude_derivative = 1.0 / (
        1.0 - characteristic * math.sin(amplitude) ** 2
    )
    function = lambda phi: phx.special.ellippiinc(characteristic, phi, 0.0)
    np.testing.assert_allclose(
        jax.jacfwd(function)(amplitude),
        expected_amplitude_derivative,
        rtol=3e-15,
    )
    np.testing.assert_allclose(
        jax.jacrev(function)(amplitude),
        expected_amplitude_derivative,
        rtol=3e-15,
    )


def test_incomplete_legendre_amplitude_derivatives_are_integrands():
    phi = jnp.asarray([-4.0, -0.7, 0.3, 1.2, 5.0])
    m = jnp.asarray([-0.5, 0.0, 0.2, 0.8, 0.95])
    n = jnp.asarray([-1.0, -0.2, 0.0, 0.4, 0.8])
    sine_squared = np.sin(np.asarray(phi)) ** 2
    root = np.sqrt(1.0 - np.asarray(m) * sine_squared)

    first = jax.vmap(jax.grad(phx.special.ellipkinc, argnums=0))(phi, m)
    second = jax.vmap(jax.grad(phx.special.ellipeinc, argnums=0))(phi, m)
    third = jax.vmap(jax.grad(phx.special.ellippiinc, argnums=1))(n, phi, m)
    np.testing.assert_allclose(np.asarray(first), 1.0 / root, rtol=4e-13)
    np.testing.assert_allclose(np.asarray(second), root, rtol=4e-13)
    np.testing.assert_allclose(
        np.asarray(third), 1.0 / ((1.0 - np.asarray(n) * sine_squared) * root), rtol=7e-13
    )


def test_incomplete_legendre_parameter_derivatives_compose_across_modes():
    point = jnp.asarray([0.2, 4.0, 0.4])

    def observable(arguments):
        n, phi, m = arguments
        return (
            phx.special.ellipkinc(phi, m)
            + 0.3 * phx.special.ellipeinc(phi, m)
            - 0.2 * phx.special.ellippiinc(n, phi, m)
        )

    forward = jax.jacfwd(observable)(point)
    reverse = jax.jacrev(observable)(point)
    np.testing.assert_allclose(
        np.asarray(forward), np.asarray(reverse), rtol=7e-13, atol=7e-14
    )

    forward_hessian = jax.jacfwd(jax.jacrev(observable))(point)
    reverse_hessian = jax.jacrev(jax.jacfwd(observable))(point)
    assert np.all(np.isfinite(np.asarray(forward_hessian)))
    np.testing.assert_allclose(
        np.asarray(forward_hessian),
        np.asarray(reverse_hessian),
        rtol=2e-11,
        atol=2e-12,
    )


def test_incomplete_second_kind_endpoint_parameter_derivatives():
    amplitude = 0.5
    for parameter, direction in ((0.0, 1), (1.0, -1)):
        function = lambda value: phx.special.ellipeinc(amplitude, value)
        with mp.workdps(70):
            reference = lambda value: mp.ellipe(mp.mpf(str(amplitude)), value)
            expected_first = float(
                mp.diff(reference, mp.mpf(str(parameter)), 1, direction=direction)
            )
            expected_second = float(
                mp.diff(reference, mp.mpf(str(parameter)), 2, direction=direction)
            )
        np.testing.assert_allclose(
            jax.grad(function)(parameter), expected_first, rtol=3e-13
        )
        np.testing.assert_allclose(
            jax.grad(jax.grad(function))(parameter), expected_second, rtol=3e-12
        )


def test_incomplete_second_kind_endpoint_period_amplitude_derivative():
    function = lambda phi: phx.special.ellipeinc(phi, 1.0)
    np.testing.assert_allclose(jax.jacfwd(function)(math.pi), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(jax.jacrev(function)(math.pi), 1.0, rtol=0.0, atol=0.0)


def test_incomplete_legendre_period_reductions_and_oddness():
    phi = jnp.asarray([-2.0, -0.4, 0.3, 1.7])
    m = 0.6
    n = 0.3
    functions = [
        (lambda value: phx.special.ellipkinc(value, m), 2.0 * phx.special.ellipk(m)),
        (lambda value: phx.special.ellipeinc(value, m), 2.0 * phx.special.ellipe(m)),
        (
            lambda value: phx.special.ellippiinc(n, value, m),
            2.0 * phx.special.ellippi(n, m),
        ),
    ]
    for function, increment in functions:
        np.testing.assert_allclose(
            np.asarray(function(phi + math.pi) - function(phi)),
            np.asarray(jnp.full_like(phi, increment)),
            rtol=8e-13,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            np.asarray(function(-phi)), -np.asarray(function(phi)), rtol=8e-13, atol=2e-14
        )


def test_incomplete_legendre_domains_and_broadcasting():
    values = phx.special.ellipkinc(
        jnp.asarray([0.2, 0.4])[:, None], jnp.asarray([0.0, 0.5])
    )
    assert values.shape == (2, 2)
    third_values = phx.special.ellippiinc(
        jnp.asarray([0.1, 0.2])[:, None], 0.3, jnp.asarray([0.0, 0.5])
    )
    assert third_values.shape == (2, 2)
    assert np.isnan(phx.special.ellipkinc(0.2, 1.1))
    assert np.isnan(phx.special.ellipeinc(0.2, 1.1))
    assert np.isnan(phx.special.ellippiinc(1.0, 0.2, 0.5))
    assert np.isposinf(phx.special.ellipkinc(math.pi / 2.0, 1.0))
