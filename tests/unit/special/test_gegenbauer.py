import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

import phydrax as phx


@pytest.mark.parametrize("degree", (0, 1, 2, 5, 12))
def test_gegenbauer_matches_scipy_for_broadcast_real_arguments(degree):
    alpha = np.asarray([0.2, 0.5, 1.0, 2.75])[:, None]
    points = np.linspace(-1.3, 1.3, 17)[None, :]

    actual = phx.special.gegenbauer_c(degree, alpha, points)
    expected = special.eval_gegenbauer(degree, alpha, points)

    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)


@pytest.mark.parametrize("degree", (0, 1, 2, 7))
def test_gegenbauer_supports_complex_arguments(degree):
    alpha = np.asarray([0.25, 1.5])[:, None]
    points = np.asarray([-0.4 + 0.7j, 0.0 - 0.3j, 1.2 + 0.1j])[None, :]

    actual = phx.special.gegenbauer_c(degree, alpha, points)
    expected = special.eval_gegenbauer(degree, alpha, points)

    assert np.iscomplexobj(actual)
    np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=4e-13)


def test_gegenbauer_recurrence_and_parity_hold_lane_wise():
    alpha = jnp.asarray([0.1, 0.7, 2.0])[:, None]
    points = jnp.linspace(-0.9, 0.9, 11)[None, :]
    degree = 9
    previous = phx.special.gegenbauer_c(degree - 1, alpha, points)
    current = phx.special.gegenbauer_c(degree, alpha, points)
    following = phx.special.gegenbauer_c(degree + 1, alpha, points)
    recurrence = (
        2.0 * (degree + alpha) * points * current
        - (degree + 2.0 * alpha - 1.0) * previous
    ) / (degree + 1.0)

    assert jnp.allclose(following, recurrence, rtol=2e-12, atol=2e-13)
    assert jnp.allclose(
        phx.special.gegenbauer_c(degree, alpha, -points),
        (-1) ** degree * current,
        rtol=2e-12,
        atol=2e-13,
    )


def test_gegenbauer_contains_legendre_and_chebyshev_u_families():
    points = np.linspace(-0.97, 0.97, 31)
    for degree in range(9):
        np.testing.assert_allclose(
            phx.special.gegenbauer_c(degree, 0.5, points),
            special.eval_legendre(degree, points),
            rtol=2e-12,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            phx.special.gegenbauer_c(degree, 1.0, points),
            special.eval_chebyu(degree, points),
            rtol=2e-12,
            atol=2e-13,
        )


def test_standard_alpha_zero_collapse_retains_parameter_derivative_limit():
    points = np.linspace(-0.95, 0.95, 29)
    assert np.all(phx.special.gegenbauer_c(0, 0.0, points) == 1.0)
    for degree in (1, 2, 5, 11):
        np.testing.assert_array_equal(
            phx.special.gegenbauer_c(degree, 0.0, points),
            np.zeros_like(points),
        )
        np.testing.assert_allclose(
            phx.special.gegenbauer_alpha_derivative(degree, 0.0, points),
            2.0 * special.eval_chebyt(degree, points) / degree,
            rtol=3e-12,
            atol=3e-13,
        )


def test_argument_and_parameter_autodiff_match_polynomial_identities():
    degree = 8
    alpha = jnp.asarray(0.73)
    point = jnp.asarray(0.21)
    x_derivative = jax.grad(lambda value: phx.special.gegenbauer_c(degree, alpha, value))(
        point
    )
    expected_x_derivative = (
        2.0 * alpha * phx.special.gegenbauer_c(degree - 1, alpha + 1.0, point)
    )
    alpha_derivative = jax.grad(
        lambda value: phx.special.gegenbauer_c(degree, value, point)
    )(alpha)

    assert jnp.allclose(x_derivative, expected_x_derivative, rtol=2e-12, atol=2e-13)
    assert jnp.allclose(
        alpha_derivative,
        phx.special.gegenbauer_alpha_derivative(degree, alpha, point),
        rtol=3e-12,
        atol=3e-13,
    )


def test_vandermonde_is_modes_last_after_broadcasting():
    alpha = jnp.asarray([0.25, 0.75])[:, None, None]
    points = jnp.asarray([[-0.8, -0.1, 0.4], [0.2, 0.6, 1.1], [-0.7, 0.0, 0.9]])[
        None, :, :
    ]
    degree = 6
    values = phx.special.gegenbauer_vander(alpha, points, degree)
    expected = jnp.stack(
        [phx.special.gegenbauer_c(index, alpha, points) for index in range(degree + 1)],
        axis=-1,
    )

    assert values.shape == (2, 3, 3, degree + 1)
    assert jnp.allclose(values, expected, rtol=2e-12, atol=2e-13)


def test_gegenbauer_degree_and_alpha_boundaries_are_explicit():
    with pytest.raises(TypeError, match="n must be an integer"):
        phx.special.gegenbauer_c(2.5, 0.7, 0.2)
    with pytest.raises(TypeError, match="degree must be an integer"):
        phx.special.gegenbauer_vander(0.7, 0.2, True)
    with pytest.raises(ValueError, match="n must be nonnegative"):
        phx.special.gegenbauer_c(-1, 0.7, 0.2)
    with pytest.raises(TypeError, match="real alpha"):
        phx.special.gegenbauer_c(2, 0.7 + 0.1j, 0.2)

    alpha = jnp.asarray([-0.5000001, -0.5, jnp.nan, jnp.inf, -jnp.inf])
    assert jnp.all(jnp.isnan(phx.special.gegenbauer_c(3, alpha, 0.2)))
    assert jnp.all(jnp.isnan(phx.special.gegenbauer_vander(alpha, 0.2, 3)))


def test_low_precision_inputs_widen_without_narrowing_complex_values():
    real = phx.special.gegenbauer_c(
        4,
        jnp.asarray(0.7, dtype=jnp.float16),
        jnp.asarray(0.2, dtype=jnp.float16),
    )
    complex_value = phx.special.gegenbauer_c(
        4,
        jnp.asarray(0.7, dtype=jnp.float64),
        jnp.asarray(0.2 + 0.1j, dtype=jnp.complex64),
    )

    assert real.dtype == jnp.float32
    assert complex_value.dtype == jnp.complex128
