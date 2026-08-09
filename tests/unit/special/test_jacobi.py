import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import scipy.special

import phydrax as phx


def test_jacobi_functions_match_scipy_on_standard_domain():
    u = np.linspace(-30.0, 30.0, 161)[:, None]
    m = np.asarray([0.0, 1e-8, 0.2, 0.8, 0.99, 1.0 - 1e-6])[None, :]
    actual = [
        np.asarray(value) for value in phx.special.ellipj(jnp.asarray(u), jnp.asarray(m))
    ]
    expected = scipy.special.ellipj(u, m)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=2e-12, atol=8e-14)


def test_jacobi_algebraic_and_amplitude_identities_include_negative_parameters():
    u = jnp.linspace(-12.0, 12.0, 101)[:, None]
    m = jnp.asarray([-20.0, -1.0, -0.1, 0.0, 0.3, 0.9, 1.0 - 1e-10])[None, :]
    sn, cn, dn, amplitude = phx.special.ellipj(u, m)
    np.testing.assert_allclose(np.asarray(sn * sn + cn * cn), 1.0, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        np.asarray(dn * dn + m * sn * sn), 1.0, rtol=3e-14, atol=3e-14
    )
    np.testing.assert_allclose(
        np.asarray(jnp.sin(amplitude)), np.asarray(sn), rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        np.asarray(jnp.cos(amplitude)), np.asarray(cn), rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        np.asarray(phx.special.ellipam(u, m)), np.asarray(amplitude)
    )


def test_jacobi_argument_derivatives_match_closed_system():
    u = jnp.asarray([-5.0, -0.2, 0.0, 1.3, 8.0])
    m = jnp.asarray([-1.0, 0.0, 0.2, 0.8, 1.0])

    def evaluate(argument, parameter):
        return jnp.stack(phx.special.ellipj(argument, parameter))

    derivatives = jax.vmap(jax.jacfwd(evaluate, argnums=0))(u, m)
    sn, cn, dn, _ = phx.special.ellipj(u, m)
    expected = jnp.stack((cn * dn, -sn * dn, -m * sn * cn, dn), axis=1)
    np.testing.assert_allclose(
        np.asarray(derivatives), np.asarray(expected), rtol=3e-13, atol=3e-14
    )


def test_jacobi_parameter_derivatives_compose_across_modes():
    point = jnp.asarray([1.3, 0.4])

    def observable(arguments):
        sn, cn, dn, amplitude = phx.special.ellipj(arguments[0], arguments[1])
        return sn + 0.2 * cn - 0.3 * dn + 0.1 * amplitude

    forward = jax.jacfwd(observable)(point)
    reverse = jax.jacrev(observable)(point)
    np.testing.assert_allclose(
        np.asarray(forward), np.asarray(reverse), rtol=3e-13, atol=3e-14
    )

    forward_hessian = jax.jacfwd(jax.jacrev(observable))(point)
    reverse_hessian = jax.jacrev(jax.jacfwd(observable))(point)
    np.testing.assert_allclose(
        np.asarray(forward_hessian), np.asarray(reverse_hessian), rtol=2e-11, atol=2e-12
    )


def test_jacobi_extreme_negative_parameter_preserves_complement():
    parameter = -1e20
    argument = 10.0 / np.sqrt(1.0 - parameter)
    actual = [
        float(value)
        for value in phx.special.ellipj(jnp.asarray(argument), jnp.asarray(parameter))[:3]
    ]
    with mp.workdps(70):
        expected = [
            float(
                mp.re(
                    mp.ellipfun(
                        name,
                        mp.mpf(str(argument)),
                        mp.mpf(str(parameter)),
                    )
                )
            )
            for name in ("sn", "cn", "dn")
        ]
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-15)


def test_jacobi_endpoint_parameter_hessians_match_high_precision():
    argument = 1.3
    with mp.workdps(70):
        reference = lambda parameter: mp.asin(
            mp.ellipfun("sn", mp.mpf(str(argument)), parameter)
        )
        expected_zero = float(mp.diff(reference, mp.mpf("0"), 2, direction=1))
        expected_one = float(mp.diff(reference, mp.mpf("1"), 2, direction=-1))

    second = lambda parameter: jax.grad(
        jax.grad(lambda value: phx.special.ellipam(argument, value))
    )(parameter)
    np.testing.assert_allclose(second(0.0), expected_zero, rtol=3e-13)
    np.testing.assert_allclose(second(1.0), expected_one, rtol=3e-12)

    assert np.isfinite(
        jax.grad(lambda parameter: phx.special.ellipam(jnp.asarray(500.0), parameter))(
            1.0
        )
    )


def test_jacobi_near_endpoint_expansion_respects_large_argument_scale():
    def parameter_derivative(dtype):
        argument = jnp.asarray(20.0, dtype=dtype)
        parameter = jnp.asarray(0.9995, dtype=dtype)
        function = lambda value: jnp.stack(phx.special.ellipj(argument, value))
        return jax.jacfwd(function)(parameter)

    actual = parameter_derivative(jnp.float32)
    reference = parameter_derivative(jnp.float64)
    assert np.isfinite(np.asarray(actual)).all()
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(reference), rtol=2e-4, atol=2e-4
    )


def test_jacobi_large_argument_endpoint_parameter_derivatives_are_safe():
    argument = jnp.asarray(100.0, dtype=jnp.float32)
    parameter = jnp.asarray(1.0, dtype=jnp.float32)
    expected = (-0.25, np.inf, -np.inf, -np.inf)
    for component, reference in enumerate(expected):
        function = lambda value, component=component: phx.special.ellipj(argument, value)[
            component
        ]
        forward = jax.jacfwd(function)(parameter)
        reverse = jax.jacrev(function)(parameter)
        assert not np.isnan(forward)
        assert not np.isnan(reverse)
        if np.isfinite(reference):
            np.testing.assert_allclose(forward, reference, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(reverse, reference, rtol=0.0, atol=0.0)
        else:
            assert np.isinf(forward)
            assert np.signbit(forward) == np.signbit(reference)
            assert np.isinf(reverse)
            assert np.signbit(reverse) == np.signbit(reference)


def test_jacobi_large_argument_endpoint_amplitude_curvature_is_nonzero():
    argument_value = 30.0
    argument = jnp.asarray(argument_value, dtype=jnp.float32)
    parameter = jnp.asarray(1.0, dtype=jnp.float32)
    hyperbolic_sine = math.sinh(argument_value)
    hyperbolic_tangent = math.tanh(argument_value)
    hyperbolic_secant = 1.0 / math.cosh(argument_value)
    expected = (
        hyperbolic_sine * (9.0 - 4.0 * argument_value * hyperbolic_tangent)
        - hyperbolic_secant
        * (
            9.0 * argument_value
            + 2.0 * argument_value * argument_value * hyperbolic_tangent
        )
    ) / 32.0
    function = lambda value: phx.special.ellipam(argument, value)
    forward_reverse = jax.jacfwd(jax.jacrev(function))(parameter)
    reverse_forward = jax.jacrev(jax.jacfwd(function))(parameter)
    assert np.isfinite(forward_reverse)
    assert forward_reverse != 0.0
    np.testing.assert_allclose(forward_reverse, expected, rtol=2e-6)
    np.testing.assert_allclose(reverse_forward, expected, rtol=2e-6)


def test_jacobi_endpoint_and_invalid_contracts():
    u = jnp.asarray([-3.0, -0.2, 0.0, 1.0, 4.0])
    zero = phx.special.ellipj(u, 0.0)
    np.testing.assert_allclose(np.asarray(zero[0]), np.sin(np.asarray(u)), rtol=2e-15)
    np.testing.assert_allclose(np.asarray(zero[1]), np.cos(np.asarray(u)), rtol=2e-15)
    np.testing.assert_array_equal(np.asarray(zero[2]), np.ones(u.shape))
    np.testing.assert_allclose(np.asarray(zero[3]), np.asarray(u), rtol=0.0, atol=0.0)

    one = phx.special.ellipj(u, 1.0)
    np.testing.assert_allclose(np.asarray(one[0]), np.tanh(np.asarray(u)), rtol=2e-15)
    np.testing.assert_allclose(
        np.asarray(one[1]), 1.0 / np.cosh(np.asarray(u)), rtol=2e-15
    )
    np.testing.assert_allclose(np.asarray(one[2]), np.asarray(one[1]), rtol=0.0, atol=0.0)

    invalid = phx.special.ellipj(jnp.asarray([0.2, 0.2]), jnp.asarray([0.5, 1.1]))
    for value in invalid:
        assert np.isfinite(np.asarray(value)[0])
        assert np.isnan(np.asarray(value)[1])
