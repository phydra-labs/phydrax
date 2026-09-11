import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import scipy.special

import phydrax as phx


mpmath.mp.dps = 60


def test_riemann_zeta_values_and_trivial_zero_derivative():
    arguments = jnp.asarray([-7.0, -4.0, -3.0, -2.0, 0.0, 0.5, 2.0, 10.0])
    actual = phx.special.zeta(arguments)
    expected = scipy.special.zeta(np.asarray(arguments))
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)
    derivative = jax.grad(phx.special.zeta)(-2.0)
    expected_derivative = float(mpmath.diff(mpmath.zeta, -2.0))
    assert derivative != 0.0
    np.testing.assert_allclose(derivative, expected_derivative, rtol=2e-10, atol=2e-13)
    assert jnp.isposinf(phx.special.zeta(1.0))


def test_hurwitz_zeta_values_shift_and_parameter_derivative():
    orders = jnp.asarray([2.0, 3.5, -1.0])
    parameters = jnp.asarray([1.25, 2.5, 0.75])
    actual = phx.special.hurwitz_zeta(orders, parameters)
    expected = np.asarray(
        [
            float(mpmath.zeta(float(order), float(parameter)))
            for order, parameter in zip(np.asarray(orders), np.asarray(parameters))
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=4e-13)
    shifted = phx.special.hurwitz_zeta(orders, parameters + 1.0)
    np.testing.assert_allclose(
        actual - shifted,
        parameters ** (-orders),
        rtol=4e-12,
        atol=4e-13,
    )
    derivative = jax.grad(lambda a: phx.special.hurwitz_zeta(2.4, a))(1.7)
    np.testing.assert_allclose(
        derivative,
        -2.4 * phx.special.hurwitz_zeta(3.4, 1.7),
        rtol=2e-11,
        atol=2e-12,
    )


def _mp_polylog(order, argument):
    return complex(mpmath.polylog(complex(order), complex(argument)))


def test_principal_dilog_spence_and_cut_lips():
    arguments = np.asarray([0.0, 0.2 + 0.3j, -1.0, 0.7 - 0.4j])
    actual = np.asarray(phx.special.dilog(jnp.asarray(arguments)))
    expected = np.asarray([_mp_polylog(2.0, value) for value in arguments])
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(
        phx.special.spence(jnp.asarray(arguments)),
        phx.special.dilog(1.0 - jnp.asarray(arguments)),
        rtol=0.0,
        atol=0.0,
    )
    upper = phx.special.dilog(jnp.asarray(complex(2.0, 0.0)))
    lower = phx.special.dilog(jnp.asarray(complex(2.0, -0.0)))
    np.testing.assert_allclose(upper, jnp.conj(lower), rtol=2e-13, atol=2e-13)
    assert jnp.signbit(upper.imag) != jnp.signbit(lower.imag)
    np.testing.assert_allclose(jax.grad(lambda x: phx.special.dilog(x).real)(0.0), 1.0)


def test_polylog_special_values_noninteger_order_and_order_derivative():
    cases = (
        (0.0, 0.3 + 0.2j),
        (1.0, -0.4 + 0.1j),
        (2.0, 0.6 - 0.2j),
        (2.5, -0.7 + 0.15j),
        (-1.25, 0.45 + 0.1j),
    )
    for order, argument in cases:
        actual = phx.special.polylog(order, argument)
        expected = _mp_polylog(order, argument)
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)

    order, argument = 2.3, 0.45
    derivative = jax.grad(lambda s: phx.special.polylog(s, argument).real)(order)
    expected_derivative = float(
        mpmath.re(mpmath.diff(lambda s: mpmath.polylog(s, argument), order))
    )
    np.testing.assert_allclose(derivative, expected_derivative, rtol=3e-7, atol=3e-9)
    compiled = jax.jit(phx.special.polylog)(
        jnp.asarray([2.5, 1.0]), jnp.asarray([0.4, -0.2])
    )
    assert compiled.shape == (2,)
    assert jnp.all(jnp.isfinite(compiled))


def test_polylog_unsupported_envelope_is_explicit_nan():
    unsupported = phx.special.polylog(25.0, 0.9)
    assert jnp.isnan(unsupported.real) and jnp.isnan(unsupported.imag)
