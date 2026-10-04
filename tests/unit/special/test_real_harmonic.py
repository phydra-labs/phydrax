#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import legendre

from phydrax.special import (
    real_harmonic_basis,
    RealCartesianHarmonics,
    sph_harm_y_cart,
)


_DIRECTIONS = np.asarray(
    [
        [0.3, -0.4, 0.8],
        [-1.2, 0.5, -0.7],
        [0.0, 0.6, 1.1],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, -3.0],
    ],
    dtype=np.float64,
)


def _units(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def test_low_degree_values_match_closed_form_real_polynomials() -> None:
    x, y, z = _units(_DIRECTIONS).T
    values = np.asarray(RealCartesianHarmonics(2)(_DIRECTIONS))
    c0 = 0.5 / math.sqrt(math.pi)
    c1 = math.sqrt(3.0 / (4.0 * math.pi))
    c2 = 0.5 * math.sqrt(15.0 / math.pi)
    expected = np.stack(
        [
            np.full_like(x, c0),
            c1 * y,
            c1 * z,
            c1 * x,
            c2 * x * y,
            c2 * y * z,
            0.25 * math.sqrt(5.0 / math.pi) * (3.0 * z * z - 1.0),
            c2 * x * z,
            0.5 * c2 * (x * x - y * y),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(values, expected, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("degree", [0, 1, 2, 5, 8])
def test_conversion_to_complex_condon_shortley_harmonics_is_unitary(degree: int) -> None:
    basis = real_harmonic_basis(degree)
    np.testing.assert_allclose(
        basis @ jnp.conj(basis).T, jnp.eye(2 * degree + 1), atol=1e-15
    )
    complex_values = jnp.stack(
        [
            sph_harm_y_cart(degree, order, _DIRECTIONS)
            for order in range(-degree, degree + 1)
        ],
        axis=-1,
    )
    converted = complex_values @ basis.T
    real = RealCartesianHarmonics(degree)(_DIRECTIONS)[:, degree * degree :]
    np.testing.assert_allclose(jnp.imag(converted), 0.0, atol=1e-14)
    np.testing.assert_allclose(jnp.real(converted), real, rtol=1e-12, atol=1e-14)


def test_high_degree_harmonics_obey_the_addition_theorem() -> None:
    maximum = 10
    harmonics = np.asarray(RealCartesianHarmonics(maximum)(_DIRECTIONS))
    units = _units(_DIRECTIONS)
    cosine = units @ units.T
    for degree in range(maximum + 1):
        block = harmonics[:, degree * degree : (degree + 1) ** 2]
        coefficients = np.zeros(degree + 1)
        coefficients[-1] = 1.0
        expected = (
            (2 * degree + 1) / (4.0 * math.pi) * legendre.legval(cosine, coefficients)
        )
        np.testing.assert_allclose(block @ block.T, expected, rtol=1e-11, atol=1e-12)


def test_harmonics_are_orthonormal_under_exact_sphere_quadrature() -> None:
    maximum = 6
    nodes, weights = legendre.leggauss(maximum + 1)
    azimuths = 2.0 * math.pi * np.arange(2 * maximum + 2) / (2 * maximum + 2)
    sine = np.sqrt(1.0 - nodes * nodes)
    points = np.stack(
        [
            (sine[:, None] * np.cos(azimuths)[None, :]).ravel(),
            (sine[:, None] * np.sin(azimuths)[None, :]).ravel(),
            np.repeat(nodes, azimuths.size),
        ],
        axis=-1,
    )
    quadrature = np.repeat(weights, azimuths.size) * (2.0 * math.pi / azimuths.size)
    values = np.asarray(RealCartesianHarmonics(maximum)(points))
    gram = values.T @ (quadrature[:, None] * values)
    np.testing.assert_allclose(gram, np.eye((maximum + 1) ** 2), atol=1e-13)


@pytest.mark.parametrize(
    ("normalization", "degree_sum"),
    [("fully_normalized", lambda degree: 2 * degree + 1), ("schmidt", lambda degree: 1)],
    ids=["fully-normalized", "schmidt"],
)
def test_normalizations_scale_every_degree_uniformly(
    normalization: Any, degree_sum: Any
) -> None:
    values = np.asarray(
        RealCartesianHarmonics(6, normalization=normalization)(_DIRECTIONS)
    )
    for degree in range(7):
        block = values[:, degree * degree : (degree + 1) ** 2]
        np.testing.assert_allclose(
            np.sum(block * block, axis=-1), degree_sum(degree), rtol=1e-13
        )


def test_zero_direction_is_nan_without_poisoning_other_derivatives() -> None:
    harmonics = RealCartesianHarmonics(3)
    vectors = jnp.asarray([[0.0, 0.0, 0.0], [0.3, -0.4, 0.8], [jnp.inf, 0.0, 1.0]])
    values = harmonics(vectors)
    assert bool(jnp.all(jnp.isnan(values[0])))
    assert bool(jnp.all(jnp.isnan(values[2])))
    assert bool(jnp.all(jnp.isfinite(values[1])))
    gradient = jax.grad(lambda v: jnp.sum(harmonics(v)[1] ** 3))(vectors)
    np.testing.assert_array_equal(gradient[0], 0.0)
    assert bool(jnp.all(jnp.isfinite(gradient[1])))


@pytest.mark.parametrize("pole", [1.0, -1.0], ids=["north", "south"])
def test_pole_directions_have_finite_exact_derivatives(pole: float) -> None:
    harmonics = RealCartesianHarmonics(5)
    point = jnp.asarray([0.0, 0.0, 2.0 * pole])
    tangent = jnp.asarray([0.7, -0.3, 0.2])
    _, derivative = jax.jvp(harmonics, (point,), (tangent,))
    step = 1e-6
    finite = (harmonics(point + step * tangent) - harmonics(point - step * tangent)) / (
        2.0 * step
    )
    np.testing.assert_allclose(derivative, finite, rtol=1e-6, atol=1e-8)
    cotangent = jnp.linspace(-1.0, 1.0, harmonics.component_count)
    _, pullback = jax.vjp(harmonics, point)
    np.testing.assert_allclose(
        jnp.dot(pullback(cotangent)[0], tangent),
        jnp.dot(cotangent, derivative),
        rtol=1e-12,
    )


def test_regular_solid_harmonics_are_homogeneous_and_smooth_at_zero() -> None:
    solid = RealCartesianHarmonics(4, argument="regular_solid")
    direction = RealCartesianHarmonics(4)
    vectors = jnp.asarray(_DIRECTIONS)
    radius = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    degrees = jnp.repeat(jnp.arange(5), 2 * jnp.arange(5) + 1)
    np.testing.assert_allclose(
        solid(vectors), radius**degrees * direction(vectors), rtol=1e-13, atol=1e-14
    )
    origin = solid(jnp.zeros(3))
    np.testing.assert_allclose(origin[0], 0.5 / math.sqrt(math.pi))
    np.testing.assert_array_equal(origin[1:], 0.0)
    jacobian = jax.jacfwd(solid)(jnp.zeros(3))
    np.testing.assert_allclose(
        jacobian[1:4],
        math.sqrt(3.0 / (4.0 * math.pi))
        * jnp.eye(3, dtype=jnp.float64)[jnp.asarray([1, 2, 0], dtype=jnp.int32)],
    )


def test_construction_and_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        RealCartesianHarmonics(-1)
    with pytest.raises(TypeError, match="static integer"):
        RealCartesianHarmonics(True)
    with pytest.raises(ValueError):
        RealCartesianHarmonics(2, normalization="unnormalized")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="dimension 3"):
        RealCartesianHarmonics(2)(jnp.ones((4, 2)))
    harmonics = RealCartesianHarmonics(3)
    assert harmonics.component_count == 16
    assert harmonics.component_offset(2) == 4
    with pytest.raises(ValueError, match="exceeds"):
        harmonics.component_offset(4)
