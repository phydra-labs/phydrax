import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

import phydrax as phx


def test_cylindrical_bessel_family_matches_scipy_across_regimes():
    orders = np.asarray([0.0, 1e-8, 0.3, 0.5, 1.0, 2.3, 10.0, 49.0, 100.0, 499.0, 500.0])
    arguments = np.geomspace(1e-12, 1e4, 55)
    v, x = np.meshgrid(orders, arguments, indexing="ij")
    for function, reference in [
        (phx.special.jv, scipy.special.jv),
        (phx.special.yv, scipy.special.yv),
        (phx.special.hankel1, scipy.special.hankel1),
        (phx.special.hankel2, scipy.special.hankel2),
    ]:
        actual = np.asarray(function(jnp.asarray(v), jnp.asarray(x)))
        expected = reference(v, x)
        finite = np.isfinite(expected)
        np.testing.assert_allclose(
            actual[finite], expected[finite], rtol=3e-10, atol=3e-13
        )
        assert np.all(~np.isfinite(expected) | np.isfinite(actual))


def test_cylindrical_bessel_float32_values_match_scipy():
    orders = np.asarray([0.0, 0.3, 1.0, 2.3, 10.0, 49.0], dtype=np.float32)
    arguments = np.geomspace(1e-4, 100.0, 50, dtype=np.float32)
    v, x = np.meshgrid(orders, arguments, indexing="ij")
    for function, reference in [
        (phx.special.jv, scipy.special.jv),
        (phx.special.yv, scipy.special.yv),
    ]:
        actual = np.asarray(function(jnp.asarray(v), jnp.asarray(x)))
        expected = reference(v, x)
        finite = np.isfinite(expected)
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize(
    ("dtype", "rtol"),
    [
        (jnp.float32, 3e-6),
        (jnp.float64, 3e-15),
    ],
)
def test_cylindrical_zero_order_at_smallest_positive_argument(dtype, rtol):
    x = jnp.nextafter(jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype))
    j_value = np.asarray(phx.special.jv(0.0, x))
    y_value = np.asarray(phx.special.yv(0.0, x))
    assert j_value.dtype == np.dtype(dtype)
    assert y_value.dtype == np.dtype(dtype)
    assert j_value == 1.0
    assert np.isfinite(y_value)
    expected_y = (2.0 / math.pi) * (np.log(np.asarray(x)) - np.log(2.0) + np.euler_gamma)
    np.testing.assert_allclose(y_value, expected_y, rtol=rtol)


def test_cylindrical_half_order_gradient_at_smallest_float64_is_finite():
    x = jnp.nextafter(
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    expected = np.exp(
        np.log(0.5)
        - 0.5 * np.log(2.0)
        - scipy.special.gammaln(1.5)
        - 0.5 * np.log(np.asarray(x))
    )
    derivative = np.asarray(jax.grad(lambda argument: phx.special.jv(0.5, argument))(x))
    assert np.isfinite(derivative)
    np.testing.assert_allclose(derivative, expected, rtol=3e-14)


def test_hankel_composition_conjugacy_and_half_integer_values():
    v = jnp.asarray([0.0, 0.3, 2.3, 10.0])[:, None]
    x = jnp.geomspace(0.1, 100.0, 30)[None, :]
    first = phx.special.hankel1(v, x)
    second = phx.special.hankel2(v, x)
    np.testing.assert_allclose(
        np.asarray(second), np.conj(np.asarray(first)), rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        np.asarray(first),
        np.asarray(phx.special.jv(v, x) + 1j * phx.special.yv(v, x)),
        rtol=0.0,
        atol=0.0,
    )

    arguments = jnp.asarray([0.2, 1.0, 10.0, 100.0])
    scale = jnp.sqrt(2.0 / (math.pi * arguments))
    np.testing.assert_allclose(
        np.asarray(phx.special.jv(0.5, arguments)),
        np.asarray(scale * jnp.sin(arguments)),
        rtol=3e-13,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        np.asarray(phx.special.yv(0.5, arguments)),
        np.asarray(-scale * jnp.cos(arguments)),
        rtol=3e-13,
        atol=2e-14,
    )


def test_cylindrical_half_order_large_argument_retains_phase():
    argument = jnp.asarray(1e20, dtype=jnp.float64)
    expected = jnp.sqrt(2.0 / (math.pi * argument)) * jnp.sin(argument)
    np.testing.assert_allclose(
        np.asarray(phx.special.jv(0.5, argument)),
        np.asarray(expected),
        rtol=float(8.0 * np.finfo(np.float64).eps),
        atol=0.0,
    )


def test_cylindrical_recurrence_wronskian_and_argument_derivatives():
    orders = jnp.asarray([0.0, 0.3, 2.3, 10.0, 100.0, 500.0])
    arguments = jnp.asarray([0.2, 1.0, 3.0, 10.0, 300.0, 500.0])
    for function in (phx.special.jv, phx.special.yv):
        actual = jax.vmap(jax.grad(function, argnums=1))(orders, arguments)
        expected = orders * function(orders, arguments) / arguments - function(
            orders + 1.0, arguments
        )
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=2e-13, atol=2e-14
        )

        forward = jax.jacfwd(lambda value: function(2.3, value))(jnp.asarray(3.0))
        reverse = jax.jacrev(lambda value: function(2.3, value))(jnp.asarray(3.0))
        np.testing.assert_allclose(np.asarray(forward), np.asarray(reverse), rtol=2e-15)
        assert np.isfinite(jax.grad(jax.grad(lambda value: function(2.3, value)))(3.0))

    j = phx.special.jv(orders, arguments)
    j_next = phx.special.jv(orders + 1.0, arguments)
    y = phx.special.yv(orders, arguments)
    y_next = phx.special.yv(orders + 1.0, arguments)
    np.testing.assert_allclose(
        np.asarray(j_next * y - j * y_next),
        np.asarray(2.0 / (math.pi * arguments)),
        rtol=2e-9,
        atol=2e-13,
    )


def test_cylindrical_zero_argument_derivatives_and_hankel_limits():
    assert jax.grad(jax.grad(lambda x: phx.special.jv(0.0, x)))(0.0) == pytest.approx(
        -0.5
    )
    assert jax.grad(jax.grad(lambda x: phx.special.jv(2.0, x)))(0.0) == pytest.approx(
        0.25
    )
    assert np.isposinf(jax.grad(lambda x: phx.special.yv(0.0, x))(0.0))
    assert np.isneginf(jax.grad(jax.grad(lambda x: phx.special.yv(0.0, x)))(0.0))

    first = np.asarray(phx.special.hankel1(0.0, 0.0))
    second = np.asarray(phx.special.hankel2(0.0, 0.0))
    assert first.real == 1.0
    assert second.real == 1.0
    assert np.isneginf(first.imag)
    assert np.isposinf(second.imag)


def test_cylindrical_order_derivatives_match_scipy():
    order = 0.3
    argument = 2.0
    step = np.cbrt(np.finfo(np.float64).eps) * (1.0 + abs(order))
    cases = (
        (
            phx.special.jv,
            phx.special.jv_order_derivative,
            scipy.special.jv,
        ),
        (
            phx.special.yv,
            phx.special.yv_order_derivative,
            scipy.special.yv,
        ),
        (
            phx.special.hankel1,
            lambda value, x: (
                phx.special.jv_order_derivative(value, x)
                + 1j * phx.special.yv_order_derivative(value, x)
            ),
            scipy.special.hankel1,
        ),
        (
            phx.special.hankel2,
            lambda value, x: (
                phx.special.jv_order_derivative(value, x)
                - 1j * phx.special.yv_order_derivative(value, x)
            ),
            scipy.special.hankel2,
        ),
    )
    for function, explicit_derivative, reference in cases:
        actual = jax.jacfwd(lambda value: function(value, argument))(jnp.asarray(order))
        explicit = explicit_derivative(order, argument)
        expected = (
            reference(order + step, argument) - reference(order - step, argument)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(explicit), rtol=2e-12, atol=2e-13
        )
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-7, atol=2e-9)


def test_cylindrical_boundaries_domains_broadcasting_and_dtype():
    orders = jnp.asarray([0.0, 0.5, 1.0, 2.0])[:, None]
    arguments = jnp.asarray([0.0, 1.0])[None, :]
    assert phx.special.jv(orders, arguments).shape == (4, 2)
    np.testing.assert_array_equal(
        np.asarray(phx.special.jv(orders[:, 0], 0.0)), [1.0, 0.0, 0.0, 0.0]
    )
    assert np.isneginf(np.asarray(phx.special.yv(orders[:, 0], 0.0))).all()

    for function in (phx.special.jv, phx.special.yv):
        invalid = np.asarray(
            function(jnp.asarray([-1.0, 0.0, jnp.inf]), jnp.asarray([1.0, -1.0, 1.0]))
        )
        assert np.isnan(invalid).all()
        np.testing.assert_array_equal(
            np.asarray(function(orders[:, 0], jnp.inf)), np.zeros(4)
        )
        complex_value = function(0.0, 1.0 + 0.2j)
        assert jnp.iscomplexobj(complex_value)
        assert jnp.all(jnp.isfinite(complex_value))

    assert phx.special.hankel1(0.3, 1.0).dtype == jnp.complex128
    assert (
        phx.special.hankel2(
            jnp.asarray(0.3, dtype=jnp.float32), jnp.asarray(1.0, dtype=jnp.float32)
        ).dtype
        == jnp.complex64
    )
