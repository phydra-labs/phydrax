import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

import phydrax as phx


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (jnp.float32, 4e-5, 2e-7),
        (jnp.float64, 8e-11, 2e-13),
    ],
)
def test_modified_bessel_family_matches_scipy(dtype, rtol, atol):
    orders = np.asarray([0.0, 1e-8, 0.3, 0.5, 1.0, 2.3, 10.0, 29.9, 30.0, 50.0, 100.0])
    arguments = np.geomspace(1e-8, 1e3, 60)
    v, x = np.meshgrid(orders, arguments, indexing="ij")
    v = v.astype(np.dtype(dtype))
    x = x.astype(np.dtype(dtype))
    for function, reference in [
        (phx.special.iv, scipy.special.iv),
        (phx.special.ive, scipy.special.ive),
        (phx.special.kv, scipy.special.kv),
        (phx.special.kve, scipy.special.kve),
    ]:
        actual = np.asarray(function(jnp.asarray(v), jnp.asarray(x)))
        expected = reference(v, x)
        finite = np.isfinite(expected)
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=rtol, atol=atol)
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))


@pytest.mark.parametrize(
    ("dtype", "rtol"),
    [
        (jnp.float32, 3e-6),
        (jnp.float64, 3e-15),
    ],
)
def test_modified_bessel_small_orders_at_smallest_positive_argument(dtype, rtol):
    x = jnp.nextafter(jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype))
    for function in (phx.special.iv, phx.special.ive):
        actual = np.asarray(function(0.0, x))
        assert actual.dtype == np.dtype(dtype)
        assert actual == 1.0

    expected_k = -np.log(np.asarray(x)) + np.log(2.0) - np.euler_gamma
    for function in (phx.special.kv, phx.special.kve):
        actual = np.asarray(function(0.0, x))
        assert actual.dtype == np.dtype(dtype)
        assert np.isfinite(actual)
        np.testing.assert_allclose(actual, expected_k, rtol=rtol)

    expected_half = np.exp(0.5 * (np.log(np.pi / 2.0) - np.log(np.asarray(x))))
    for function in (phx.special.kv, phx.special.kve):
        actual = np.asarray(function(0.5, x))
        assert np.isfinite(actual)
        np.testing.assert_allclose(actual, expected_half, rtol=rtol)

    ive_derivative = np.asarray(
        jax.grad(lambda argument: phx.special.ive(0.0, argument))(x)
    )
    kve_derivative = np.asarray(
        jax.grad(lambda argument: phx.special.kve(0.0, argument))(x)
    )
    np.testing.assert_allclose(ive_derivative, -1.0, rtol=rtol)
    assert np.isneginf(kve_derivative)


def test_modified_bessel_half_order_gradient_at_smallest_float64_is_finite():
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
    for function in (phx.special.iv, phx.special.ive):
        derivative = np.asarray(jax.grad(lambda argument: function(0.5, argument))(x))
        assert np.isfinite(derivative)
        np.testing.assert_allclose(derivative, expected, rtol=3e-14)


def test_scaled_and_unscaled_modified_bessel_values_agree_when_representable():
    v = jnp.asarray([0.0, 0.3, 2.0, 10.0, 30.0])[:, None]
    x = jnp.geomspace(1e-4, 100.0, 60)[None, :]
    np.testing.assert_allclose(
        np.asarray(phx.special.ive(v, x) * jnp.exp(x)),
        np.asarray(phx.special.iv(v, x)),
        rtol=3e-13,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        np.asarray(phx.special.kve(v, x) * jnp.exp(-x)),
        np.asarray(phx.special.kv(v, x)),
        rtol=3e-13,
        atol=1e-300,
    )


def test_modified_bessel_argument_derivatives_follow_recurrences():
    orders = jnp.asarray([0.0, 0.3, 2.3, 10.0, 30.0, 50.0])
    arguments = jnp.asarray([0.2, 1.0, 3.0, 10.0, 20.0, 100.0])
    cases = [
        (
            phx.special.iv,
            lambda v, x: phx.special.iv(v + 1.0, x) + v * phx.special.iv(v, x) / x,
        ),
        (
            phx.special.ive,
            lambda v, x: (
                phx.special.ive(v + 1.0, x) + (v / x - 1.0) * phx.special.ive(v, x)
            ),
        ),
        (
            phx.special.kv,
            lambda v, x: v * phx.special.kv(v, x) / x - phx.special.kv(v + 1.0, x),
        ),
        (
            phx.special.kve,
            lambda v, x: (
                (1.0 + v / x) * phx.special.kve(v, x) - phx.special.kve(v + 1.0, x)
            ),
        ),
    ]
    for function, reference in cases:
        actual = jax.vmap(jax.grad(function, argnums=1))(orders, arguments)
        expected = reference(orders, arguments)
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=2e-13, atol=2e-14
        )

        forward = jax.jacfwd(lambda value: function(2.3, value))(jnp.asarray(3.0))
        reverse = jax.jacrev(lambda value: function(2.3, value))(jnp.asarray(3.0))
        np.testing.assert_allclose(np.asarray(forward), np.asarray(reverse), rtol=2e-15)
        assert np.isfinite(jax.grad(jax.grad(lambda value: function(2.3, value)))(3.0))


def test_scaled_modified_bessel_extreme_gradients_do_not_cancel():
    argument = jnp.asarray(1e200)
    for order in (0.0, 30.0):
        for function in (phx.special.ive, phx.special.kve):
            value = function(order, argument)
            derivative = jax.grad(lambda x: function(order, x))(argument)
            expected = -0.5 * value / argument
            assert derivative != 0.0
            np.testing.assert_allclose(
                np.asarray(derivative), np.asarray(expected), rtol=2e-13
            )


def test_modified_bessel_zero_argument_derivatives_compose():
    assert jax.grad(jax.grad(lambda x: phx.special.iv(0.0, x)))(0.0) == pytest.approx(0.5)
    assert jax.grad(jax.grad(lambda x: phx.special.ive(0.0, x)))(0.0) == pytest.approx(
        1.5
    )
    assert jax.grad(jax.grad(lambda x: phx.special.iv(2.0, x)))(0.0) == pytest.approx(
        0.25
    )
    for function in (phx.special.kv, phx.special.kve):
        assert np.isneginf(jax.grad(lambda x: function(0.0, x))(0.0))
        assert np.isposinf(jax.grad(jax.grad(lambda x: function(0.0, x)))(0.0))


def test_modified_bessel_order_derivatives_match_scipy():
    order = 0.3
    argument = 2.0
    step = np.cbrt(np.finfo(np.float64).eps) * (1.0 + abs(order))
    cases = (
        (phx.special.iv, phx.special.iv_order_derivative, scipy.special.iv),
        (phx.special.ive, phx.special.ive_order_derivative, scipy.special.ive),
        (phx.special.kv, phx.special.kv_order_derivative, scipy.special.kv),
        (phx.special.kve, phx.special.kve_order_derivative, scipy.special.kve),
    )
    for function, explicit_derivative, reference in cases:
        actual = jax.grad(lambda value: function(value, argument))(jnp.asarray(order))
        explicit = jnp.real(explicit_derivative(order, argument))
        expected = (
            reference(order + step, argument) - reference(order - step, argument)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(explicit), rtol=2e-12, atol=2e-13
        )
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-7, atol=2e-9)


def test_modified_bessel_boundaries_domains_and_broadcasting():
    orders = jnp.asarray([0.0, 1.0, 2.0])[:, None]
    arguments = jnp.asarray([0.0, 1.0])[None, :]
    assert phx.special.iv(orders, arguments).shape == (3, 2)
    np.testing.assert_array_equal(
        np.asarray(phx.special.iv(orders[:, 0], 0.0)), [1.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(
        np.asarray(phx.special.ive(orders[:, 0], 0.0)), [1.0, 0.0, 0.0]
    )
    assert np.isposinf(np.asarray(phx.special.kv(orders[:, 0], 0.0))).all()
    assert np.isposinf(np.asarray(phx.special.kve(orders[:, 0], 0.0))).all()

    for function in (phx.special.iv, phx.special.ive, phx.special.kv, phx.special.kve):
        invalid = np.asarray(function(jnp.asarray([-1.0, 0.0]), jnp.asarray([1.0, -1.0])))
        assert np.isnan(invalid).all()
        complex_value = function(0.0, 1.0 + 0.2j)
        assert jnp.iscomplexobj(complex_value)
        assert jnp.all(jnp.isfinite(complex_value))

    assert np.isposinf(phx.special.iv(0.0, jnp.inf))
    assert phx.special.ive(0.0, jnp.inf) == 0.0
    assert phx.special.kv(0.0, jnp.inf) == 0.0
    assert phx.special.kve(0.0, jnp.inf) == 0.0
