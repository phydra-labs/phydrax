import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (jnp.float16, jnp.float32),
        (jnp.bfloat16, jnp.float32),
        (jnp.float32, jnp.float32),
        (jnp.float64, jnp.float64),
    ],
)
def test_real_dtype_contract(dtype, expected):
    value = jnp.asarray(0.5, dtype=dtype)
    assert phx.special.dawsn(value).dtype == expected
    assert phx.special.voigt_profile(value, value, value).dtype == expected
    real_values = [
        *phx.special.airy(value),
        *phx.special.airye(value),
        phx.special.elliprc(value, value),
        phx.special.elliprd(value, value, value),
        phx.special.elliprf(value, value, value),
        phx.special.elliprg(value, value, value),
        phx.special.elliprj(value, value, value, value),
        phx.special.ellipe(value),
        phx.special.ellipeinc(value, value),
        *phx.special.ellipj(value, value),
        phx.special.ellipk(value),
        phx.special.ellipkinc(value, value),
        phx.special.ellipkm1(value),
        phx.special.ellippi(value, value),
        phx.special.ellippiinc(value, value, value),
        phx.special.iv(value, value),
        phx.special.ive(value, value),
        phx.special.jv(value, value),
        phx.special.kv(value, value),
        phx.special.kve(value, value),
        phx.special.yv(value, value),
    ]
    assert all(result.dtype == expected for result in real_values)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (jnp.float16, jnp.complex64),
        (jnp.bfloat16, jnp.complex64),
        (jnp.float32, jnp.complex64),
        (jnp.float64, jnp.complex128),
        (jnp.complex64, jnp.complex64),
        (jnp.complex128, jnp.complex128),
    ],
)
def test_faddeeva_dtype_contract(dtype, expected):
    assert phx.special.wofz(jnp.asarray(0.5, dtype=dtype)).dtype == expected


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (jnp.float16, jnp.complex64),
        (jnp.bfloat16, jnp.complex64),
        (jnp.float32, jnp.complex64),
        (jnp.float64, jnp.complex128),
    ],
)
def test_hankel_dtype_contract(dtype, expected):
    value = jnp.asarray(0.5, dtype=dtype)
    assert phx.special.hankel1(value, value).dtype == expected
    assert phx.special.hankel2(value, value).dtype == expected


def test_python_integer_uses_configured_default_float_dtype():
    assert phx.special.dawsn(1).dtype == jnp.float64
    assert phx.special.wofz(1).dtype == jnp.complex128


def test_mixed_voigt_arguments_use_common_inexact_dtype():
    value = phx.special.voigt_profile(
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float64),
        0,
    )
    assert value.dtype == jnp.float64


@pytest.mark.parametrize(
    "function",
    [
        lambda value: phx.special.voigt_profile(value, 1.0, 1.0),
        lambda value: phx.special.voigt_profile(1.0, value, 1.0),
        lambda value: phx.special.voigt_profile(1.0, 1.0, value),
    ],
)
def test_nonholomorphic_voigt_arguments_reject_complex_inputs(function):
    with pytest.raises(TypeError, match="does not support complex-valued inputs"):
        function(1.0 + 0.5j)


def test_voigt_arguments_broadcast():
    x = jnp.asarray([-1.0, 0.0, 1.0])[:, None]
    sigma = jnp.asarray([0.5, 1.0])
    value = phx.special.voigt_profile(x, sigma, 0.25)
    assert value.shape == (3, 2)
    assert np.all(np.asarray(value) >= 0.0)


def test_jit_and_vmap_compose():
    points = jnp.linspace(-2.0, 2.0, 9)
    compiled = jax.jit(
        lambda values: (
            phx.special.dawsn(values),
            phx.special.wofz(values + 0.5j),
            phx.special.voigt_profile(values, 0.8, 0.2),
        )
    )
    dawson, faddeeva, voigt = compiled(points)
    mapped = jax.vmap(lambda value: phx.special.voigt_profile(value, 0.8, 0.2))(points)
    assert dawson.shape == points.shape
    assert faddeeva.shape == points.shape
    np.testing.assert_allclose(
        np.asarray(voigt), np.asarray(mapped), rtol=3e-16, atol=2e-17
    )


def test_dawson_nan_infinity_and_signed_zero_contract():
    values = np.asarray(
        phx.special.dawsn(jnp.asarray([jnp.nan, -jnp.inf, -0.0, 0.0, jnp.inf]))
    )
    assert np.isnan(values[0])
    np.testing.assert_array_equal(values[1:], np.zeros(4))
    assert np.signbit(values[1])
    assert np.signbit(values[2])
    assert not np.signbit(values[3])
    assert not np.signbit(values[4])
    derivatives = np.asarray(
        jax.vmap(jax.grad(phx.special.dawsn))(jnp.asarray([-jnp.inf, jnp.inf]))
    )
    np.testing.assert_array_equal(derivatives, np.zeros(2))


def test_faddeeva_nan_and_complex_infinity_contract():
    arguments = jax.lax.complex(
        jnp.asarray([jnp.nan, -jnp.inf, jnp.inf, 0.0, 0.0]),
        jnp.asarray([0.0, 0.0, 0.0, jnp.inf, -jnp.inf]),
    )
    values = np.asarray(phx.special.wofz(arguments))
    assert np.isnan(values[0].real) or np.isnan(values[0].imag)
    np.testing.assert_array_equal(values[1:4], np.zeros(3, dtype=np.complex128))
    assert np.isposinf(values[4].real)
    assert values[4].imag == 0.0

    derivatives = np.asarray(
        jax.jvp(
            phx.special.wofz,
            (arguments[1:4],),
            (jnp.ones_like(arguments[1:4]),),
        )[1]
    )
    np.testing.assert_array_equal(derivatives, np.zeros(3, dtype=np.complex128))


def test_voigt_scale_boundaries_match_limiting_densities():
    x = jnp.asarray([-jnp.inf, -2.0, 0.0, 2.0, jnp.inf])
    sigma = 1.25
    gamma = 0.75

    gaussian = np.asarray(phx.special.voigt_profile(x, sigma, 0.0))
    expected_gaussian = np.exp(-(np.asarray(x) ** 2) / (2.0 * sigma**2)) / (
        sigma * math.sqrt(2.0 * math.pi)
    )
    np.testing.assert_allclose(gaussian, expected_gaussian, rtol=5e-13, atol=5e-15)

    cauchy = np.asarray(phx.special.voigt_profile(x, 0.0, gamma))
    expected_cauchy = gamma / (math.pi * (np.asarray(x) ** 2 + gamma**2))
    np.testing.assert_allclose(cauchy, expected_cauchy, rtol=2e-15, atol=0.0)

    point_mass = np.asarray(phx.special.voigt_profile(x, 0.0, 0.0))
    np.testing.assert_array_equal(point_mass[[0, 1, 3, 4]], np.zeros(4))
    assert np.isposinf(point_mass[2])
    boundary_x = jnp.asarray([-jnp.inf, jnp.inf])
    gaussian_derivatives = jax.vmap(
        jax.grad(lambda value: phx.special.voigt_profile(value, sigma, 0.0))
    )(boundary_x)
    cauchy_derivatives = jax.vmap(
        jax.grad(lambda value: phx.special.voigt_profile(value, 0.0, gamma))
    )(boundary_x)
    np.testing.assert_array_equal(np.asarray(gaussian_derivatives), np.zeros(2))
    np.testing.assert_array_equal(np.asarray(cauchy_derivatives), np.zeros(2))


def test_voigt_invalid_scales_and_nans_propagate():
    values = np.asarray(
        phx.special.voigt_profile(
            jnp.asarray([0.0, 0.0, jnp.nan, 0.0]),
            jnp.asarray([-1.0, 1.0, 1.0, jnp.nan]),
            jnp.asarray([1.0, -1.0, 1.0, 1.0]),
        )
    )
    assert np.all(np.isnan(values))
