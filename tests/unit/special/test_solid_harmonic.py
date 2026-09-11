#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _components(function, vector):
    value = function(vector)
    return jnp.stack((jnp.real(value), jnp.imag(value)))


def test_low_degree_regular_and_irregular_closed_forms():
    vector = jnp.asarray(
        [[0.3, -0.4, 0.8], [-1.2, 0.5, -0.7], [0.0, 0.6, 1.1]],
        dtype=jnp.float64,
    )
    x, y, z = vector.T
    radius_squared = jnp.sum(vector * vector, axis=-1)
    radius = jnp.sqrt(radius_squared)

    regular = {
        (0, 0): jnp.full_like(x, 1.0 / math.sqrt(4.0 * math.pi)),
        (1, 0): math.sqrt(3.0 / (4.0 * math.pi)) * z,
        (1, 1): -math.sqrt(3.0 / (8.0 * math.pi)) * (x + 1j * y),
        (1, -1): math.sqrt(3.0 / (8.0 * math.pi)) * (x - 1j * y),
        (2, 0): math.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * z * z - radius_squared),
        (2, 2): math.sqrt(15.0 / (32.0 * math.pi)) * (x + 1j * y) ** 2,
    }
    for (degree, order), expected in regular.items():
        actual = phx.special.solid_harmonic_regular(degree, order, vector)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-14)
        irregular = phx.special.solid_harmonic_irregular(degree, order, vector)
        np.testing.assert_allclose(
            irregular,
            expected / radius ** (2 * degree + 1),
            rtol=5e-13,
            atol=3e-14,
        )


def test_homogeneity_antipodal_parity_and_conjugacy():
    vector = jnp.asarray([[0.4, -0.7, 1.1], [-0.2, 0.9, 0.5]])
    scale = 3.25
    for degree, order in ((0, 0), (1, 1), (4, 2), (5, 3)):
        regular = phx.special.solid_harmonic_regular(degree, order, vector)
        irregular = phx.special.solid_harmonic_irregular(degree, order, vector)
        np.testing.assert_allclose(
            phx.special.solid_harmonic_regular(degree, order, scale * vector),
            scale**degree * regular,
            rtol=2e-12,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            phx.special.solid_harmonic_irregular(degree, order, scale * vector),
            scale ** (-degree - 1) * irregular,
            rtol=2e-12,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            phx.special.solid_harmonic_regular(degree, order, -vector),
            (-1) ** degree * regular,
            rtol=2e-12,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            phx.special.solid_harmonic_irregular(degree, order, -vector),
            (-1) ** degree * irregular,
            rtol=2e-12,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            phx.special.solid_harmonic_regular(degree, -order, vector),
            (-1) ** order * jnp.conj(regular),
            rtol=2e-12,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            phx.special.solid_harmonic_irregular(degree, -order, vector),
            (-1) ** order * jnp.conj(irregular),
            rtol=2e-12,
            atol=2e-14,
        )


def test_euler_homogeneity_identities_hold_for_coordinate_derivatives():
    point = jnp.asarray([0.4, -0.65, 1.2], dtype=jnp.float64)
    for degree, order in ((1, 0), (3, -2), (6, 3)):
        regular = lambda vector: phx.special.solid_harmonic_regular(degree, order, vector)
        irregular = lambda vector: phx.special.solid_harmonic_irregular(
            degree, order, vector
        )
        regular_value = _components(regular, point)
        irregular_value = _components(irregular, point)
        regular_jacobian = jax.jacfwd(lambda value: _components(regular, value))(point)
        irregular_jacobian = jax.jacrev(lambda value: _components(irregular, value))(
            point
        )
        np.testing.assert_allclose(
            regular_jacobian @ point,
            degree * regular_value,
            rtol=2e-11,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            irregular_jacobian @ point,
            -(degree + 1) * irregular_value,
            rtol=3e-11,
            atol=3e-12,
        )


def test_regular_and_irregular_modes_are_harmonic_under_coordinate_hessians():
    point = jnp.asarray([0.37, -0.61, 1.09], dtype=jnp.float64)
    for function in (
        lambda vector: phx.special.solid_harmonic_regular(5, -3, vector),
        lambda vector: phx.special.solid_harmonic_irregular(5, -3, vector),
    ):
        real_hessian = jax.hessian(lambda value: jnp.real(function(value)))(point)
        imaginary_hessian = jax.hessian(lambda value: jnp.imag(function(value)))(point)
        np.testing.assert_allclose(jnp.trace(real_hessian), 0.0, atol=2e-10)
        np.testing.assert_allclose(jnp.trace(imaginary_hessian), 0.0, atol=2e-10)


def test_origin_semantics_and_irregular_singularities_are_lane_local():
    origin = jnp.zeros(3, dtype=jnp.float64)
    np.testing.assert_allclose(
        phx.special.solid_harmonic_regular(0, 0, origin),
        1.0 / math.sqrt(4.0 * math.pi),
        rtol=0.0,
        atol=0.0,
    )
    for degree, order in ((1, 0), (2, -1), (6, 4)):
        np.testing.assert_array_equal(
            phx.special.solid_harmonic_regular(degree, order, origin),
            0.0j,
        )

    vectors = jnp.asarray(
        [
            [0.3, -0.4, 0.8],
            [0.0, 0.0, 0.0],
            [jnp.nan, 0.0, 1.0],
            [0.0, jnp.inf, 1.0],
            [-0.6, 0.2, 0.9],
        ]
    )
    values = phx.special.solid_harmonic_irregular(3, 2, vectors)
    assert jnp.all(jnp.isfinite(values[jnp.asarray([0, 4])]))
    assert jnp.all(jnp.isnan(jnp.real(values[1:4])))
    assert jnp.all(jnp.isnan(jnp.imag(values[1:4])))


def test_irregular_reciprocal_scaling_handles_extreme_finite_radii():
    direction = jnp.asarray([2.0, -3.0, 5.0], dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)
    radii = jnp.asarray([1.0e-100, 1.0e100], dtype=jnp.float64)
    vectors = radii[:, None] * direction
    angular = phx.special.sph_harm_y_cart(1, -1, direction)
    actual = phx.special.solid_harmonic_irregular(1, -1, vectors)
    expected = angular * radii**-2
    assert jnp.all(jnp.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)


def test_laplace_addition_theorem_recovers_inverse_distance():
    source = jnp.asarray([0.08, -0.11, 0.17], dtype=jnp.float64)
    target = jnp.asarray([1.1, 0.7, -0.5], dtype=jnp.float64)
    expansion = 0.0j
    for degree in range(10):
        shell = 0.0j
        for order in range(-degree, degree + 1):
            shell = shell + phx.special.solid_harmonic_regular(
                degree, order, source
            ) * jnp.conj(phx.special.solid_harmonic_irregular(degree, order, target))
        expansion = expansion + shell / (2 * degree + 1)
    expected = 1.0 / (4.0 * math.pi * jnp.linalg.norm(target - source))
    np.testing.assert_allclose(expansion, expected, rtol=2e-9, atol=2e-12)
    np.testing.assert_allclose(jnp.imag(expansion), 0.0, atol=2e-14)


def test_dtype_static_structure_jit_and_directional_ad_compose():
    low = jnp.asarray([[0.2, -0.3, 0.7]], dtype=jnp.float16)
    high = jnp.asarray([0.2, -0.3, 0.7], dtype=jnp.float64)
    assert phx.special.solid_harmonic_regular(3, 2, low).dtype == jnp.complex64
    assert phx.special.solid_harmonic_irregular(3, 2, high).dtype == jnp.complex128

    points = jnp.asarray([[0.4, -0.2, 0.8], [-0.5, 0.3, 1.1]])
    compiled = jax.jit(lambda value: phx.special.solid_harmonic_regular(4, -2, value))(
        points
    )
    expected = phx.special.solid_harmonic_regular(4, -2, points)
    np.testing.assert_allclose(compiled, expected, rtol=2e-13, atol=2e-14)

    tangent = jnp.asarray([0.1, 0.3, -0.2])
    _, forward = jax.jvp(
        lambda value: _components(
            lambda vector: phx.special.solid_harmonic_irregular(3, 1, vector),
            value,
        ),
        (high,),
        (tangent,),
    )
    reverse = (
        jax.jacrev(
            lambda value: _components(
                lambda vector: phx.special.solid_harmonic_irregular(3, 1, vector),
                value,
            )
        )(high)
        @ tangent
    )
    np.testing.assert_allclose(forward, reverse, rtol=2e-12, atol=2e-13)

    with pytest.raises(ValueError):
        phx.special.solid_harmonic_regular(2, 3, high)
    with pytest.raises(ValueError):
        phx.special.solid_harmonic_irregular(2, 1, jnp.ones((4, 2)))
    with pytest.raises(TypeError):
        phx.special.solid_harmonic_regular(2, 1, jnp.ones(3, dtype=complex))
    with pytest.raises(TypeError):
        jax.jit(lambda degree: phx.special.solid_harmonic_regular(degree, 0, high))(
            jnp.asarray(2)
        )
