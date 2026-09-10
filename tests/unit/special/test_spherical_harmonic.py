import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

import phydrax as phx


def _cartesian(theta, phi):
    sine = np.sin(theta)
    return np.stack((sine * np.cos(phi), sine * np.sin(phi), np.cos(theta)), axis=-1)


def _scipy_cartesian(n, m, vector):
    vector = np.asarray(vector)
    radius = np.linalg.norm(vector, axis=-1)
    theta = np.arccos(vector[..., 2] / radius)
    phi = np.mod(np.arctan2(vector[..., 1], vector[..., 0]), 2.0 * np.pi)
    return scipy.special.sph_harm_y(n, m, theta, phi)


def test_scalar_harmonics_match_scipy_at_poles_and_representative_modes():
    theta = np.asarray(
        [
            0.0,
            1.0e-10,
            1.0e-7,
            0.37,
            np.pi / 2,
            2.41,
            np.pi - 1.0e-7,
            np.pi - 1.0e-10,
            np.pi,
        ]
    )
    phi = np.asarray([0.0, 0.2, 1.1, 2.0, 3.4, 4.7, 5.8, 0.9, 2.8])
    modes = ((0, 0), (1, 1), (2, -1), (4, 2), (7, -3), (12, 7))

    for n, m in modes:
        legendre = np.asarray(phx.special.sph_legendre_p(n, m, jnp.asarray(theta)))
        harmonic = np.asarray(
            phx.special.sph_harm_y(n, m, jnp.asarray(theta), jnp.asarray(phi))
        )
        # The absolute tolerance covers exact pole zeros; the relative tolerance
        # allows accumulated roundoff through twelve recurrence levels.
        np.testing.assert_allclose(
            legendre,
            scipy.special.sph_legendre_p(n, m, theta)[0],
            rtol=5.0e-12,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            harmonic,
            scipy.special.sph_harm_y(n, m, theta, phi),
            rtol=5.0e-12,
            atol=2.0e-14,
        )


def test_normalization_and_negative_order_identities():
    nodes, weights = np.polynomial.legendre.leggauss(32)
    theta = jnp.asarray(np.arccos(nodes))
    for m in (0, 2, 5):
        values = np.asarray(phx.special.sph_legendre_p(5, m, theta))
        integral = 2.0 * np.pi * np.sum(weights * np.abs(values) ** 2)
        # Gauss-Legendre is exact here because |P_5^m|^2 is a degree-10
        # polynomial in cos(theta), leaving only floating-point recurrence error.
        np.testing.assert_allclose(integral, 1.0, rtol=2.0e-12, atol=2.0e-14)

    theta = jnp.asarray([0.3, 1.0, 2.2])[:, None]
    phi = jnp.asarray([0.2, 1.7, 5.4])
    positive_legendre = phx.special.sph_legendre_p(7, 3, theta)
    negative_legendre = phx.special.sph_legendre_p(7, -3, theta)
    positive = phx.special.sph_harm_y(7, 3, theta, phi)
    negative = phx.special.sph_harm_y(7, -3, theta, phi)
    np.testing.assert_allclose(negative_legendre, -positive_legendre, rtol=2e-13)
    np.testing.assert_allclose(negative, -jnp.conj(positive), rtol=2e-13)


def test_high_order_normalized_diagonal_avoids_raw_legendre_overflow():
    theta = jnp.asarray(jnp.pi / 2.0)
    actual = phx.special.sph_legendre_p(200, 200, theta)
    expected = scipy.special.sph_legendre_p(200, 200, float(theta))[0]

    assert jnp.isfinite(actual)
    assert np.isfinite(expected)
    np.testing.assert_allclose(actual, expected, rtol=8e-13, atol=2e-14)


def test_broadcasting_and_dtype_promotion():
    theta = jnp.asarray([[0.3], [1.2]], dtype=jnp.float16)
    phi = jnp.asarray([0.2, 1.0, 2.4], dtype=jnp.float32)
    harmonic = phx.special.sph_harm_y(3, -2, theta, phi)
    assert harmonic.shape == (2, 3)
    assert harmonic.dtype == jnp.complex64
    assert phx.special.sph_legendre_p(3, 2, theta).dtype == jnp.float32
    assert (
        phx.special.sph_harm_y_cart(3, 2, jnp.ones((2, 3), dtype=jnp.bfloat16)).dtype
        == jnp.complex64
    )

    promoted = phx.special.sph_harm_y(
        2,
        1,
        jnp.asarray(0.4, dtype=jnp.float32),
        jnp.asarray([0.1, 0.7], dtype=jnp.float64),
    )
    assert promoted.shape == (2,)
    assert promoted.dtype == jnp.complex128
    assert (
        phx.special.sph_harm_y_cart(
            2, 1, jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
        ).dtype
        == jnp.complex128
    )


def test_structural_degree_order_and_cartesian_shape_validation():
    with pytest.raises(ValueError):
        phx.special.sph_legendre_p(-1, 0, 0.4)
    with pytest.raises(ValueError):
        phx.special.sph_harm_y(2, 3, 0.4, 0.7)
    with pytest.raises(ValueError):
        phx.special.sph_harm_y_cart(2, -3, jnp.ones(3))
    for call in (
        lambda: phx.special.sph_legendre_p(True, 0, 0.4),
        lambda: phx.special.sph_harm_y(2.0, 1, 0.4, 0.7),
        lambda: phx.special.sph_harm_y(2, False, 0.4, 0.7),
        lambda: phx.special.sph_harm_y_cart(2, 1.0, jnp.ones(3)),
    ):
        with pytest.raises(TypeError):
            call()
    with pytest.raises(ValueError):
        phx.special.sph_harm_y_cart(2, 1, jnp.ones((4, 2)))

    accepted = phx.special.sph_harm_y(np.int64(2), np.int32(-1), 0.4, 0.7)
    expected = scipy.special.sph_harm_y(2, -1, 0.4, 0.7)
    np.testing.assert_allclose(accepted, expected, rtol=2e-13)

    with pytest.raises(TypeError):
        jax.jit(lambda degree: phx.special.sph_harm_y(degree, 0, 0.4, 0.7))(
            jnp.asarray(2)
        )


def test_cartesian_matches_angular_and_respects_scale_and_antipodal_parity():
    theta = np.asarray([0.2, 0.8, 1.6, 2.9])
    phi = np.asarray([0.1, 2.0, 4.2, 5.8])
    directions = _cartesian(theta, phi)
    scales = np.asarray([1.0e-200, 0.25, 7.0, 1.0e200])[:, None]

    cartesian = phx.special.sph_harm_y_cart(5, -2, jnp.asarray(directions))
    angular = phx.special.sph_harm_y(5, -2, jnp.asarray(theta), jnp.asarray(phi))
    scaled = phx.special.sph_harm_y_cart(5, -2, jnp.asarray(scales * directions))
    antipodal = phx.special.sph_harm_y_cart(5, -2, jnp.asarray(-directions))
    np.testing.assert_allclose(cartesian, angular, rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(scaled, cartesian, rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(antipodal, -cartesian, rtol=3e-12, atol=3e-14)


def test_cartesian_invalid_lanes_are_isolated():
    directions = jnp.asarray(
        [
            [0.2, -0.3, 0.9],
            [0.0, 0.0, 0.0],
            [jnp.nan, 0.0, 1.0],
            [0.0, jnp.inf, 1.0],
            [-0.4, 0.1, 0.7],
        ]
    )
    values = np.asarray(phx.special.sph_harm_y_cart(4, 1, directions))
    expected = _scipy_cartesian(4, 1, np.asarray(directions)[[0, 4]])

    np.testing.assert_allclose(values[[0, 4]], expected, rtol=3e-12, atol=3e-14)
    assert np.isnan(values[1:4].real).all()
    assert np.isnan(values[1:4].imag).all()


def test_cartesian_axis_gradients_and_forward_reverse_agree():
    basis = np.eye(3)
    coefficient = math.sqrt(3.0 / (8.0 * math.pi))

    def degree_one_components(vector):
        value = phx.special.sph_harm_y_cart(1, 1, vector)
        return jnp.stack((value.real, value.imag))

    for axis in basis:
        actual = jax.jacfwd(degree_one_components)(jnp.asarray(axis))
        expected = -coefficient * np.stack(
            (basis[0] - axis[0] * axis, basis[1] - axis[1] * axis)
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-14)

    point = jnp.asarray([0.4, -0.7, 1.3])
    direction = jnp.asarray([-0.2, 0.5, 0.1])

    def components(vector):
        value = phx.special.sph_harm_y_cart(6, -3, vector)
        return jnp.stack((value.real, value.imag))

    forward = jax.jacfwd(components)(point)
    reverse = jax.jacrev(components)(point)
    _, tangent = jax.jvp(components, (point,), (direction,))
    step = 2.0e-6
    finite_difference = (
        _scipy_cartesian(6, -3, np.asarray(point + step * direction))
        - _scipy_cartesian(6, -3, np.asarray(point - step * direction))
    ) / (2.0 * step)
    np.testing.assert_allclose(forward, reverse, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        tangent,
        [finite_difference.real, finite_difference.imag],
        rtol=3e-8,
        atol=3e-10,
    )


def test_jit_vmap_and_jvp_compose_with_scalar_harmonics():
    theta = jnp.asarray([0.2, 1.1, 2.7])
    phi = jnp.asarray([0.4, 2.2, 5.1])
    theta_tangent = jnp.asarray([0.3, -0.2, 0.1])
    phi_tangent = jnp.asarray([-0.1, 0.4, 0.2])

    def evaluate(one_theta, one_phi, one_theta_tangent, one_phi_tangent):
        return jax.jvp(
            lambda polar, azimuth: phx.special.sph_harm_y(6, -3, polar, azimuth),
            (one_theta, one_phi),
            (one_theta_tangent, one_phi_tangent),
        )

    value, tangent = jax.jit(jax.vmap(evaluate))(theta, phi, theta_tangent, phi_tangent)
    expected = scipy.special.sph_harm_y(6, -3, np.asarray(theta), np.asarray(phi))
    step = 2.0e-6
    expected_tangent = (
        scipy.special.sph_harm_y(
            6,
            -3,
            np.asarray(theta + step * theta_tangent),
            np.asarray(phi + step * phi_tangent),
        )
        - scipy.special.sph_harm_y(
            6,
            -3,
            np.asarray(theta - step * theta_tangent),
            np.asarray(phi - step * phi_tangent),
        )
    ) / (2.0 * step)
    np.testing.assert_allclose(value, expected, rtol=5e-12, atol=2e-14)
    # The finite-difference reference is limited by O(h^2) truncation and
    # O(eps/h) cancellation rather than the harmonic evaluation itself.
    np.testing.assert_allclose(tangent, expected_tangent, rtol=3e-8, atol=3e-10)
