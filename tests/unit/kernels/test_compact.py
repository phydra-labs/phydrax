#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _rotation(angle):
    return jnp.asarray(
        [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
    )


def _assert_psd(matrix, tolerance=1e-9):
    assert np.min(np.linalg.eigvalsh(np.asarray(matrix))) >= -tolerance


def test_sphere_circle_levels_match_fourier_chebyshev_closed_form():
    points = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    time = 0.3
    kernel = phx.kernels.SphereSpectralKernel(
        1,
        2,
        phx.kernels.HeatSpectralMultiplier(time),
        normalize=False,
    )
    similarity = points @ points.T
    expected = (
        1.0
        + 2.0 * jnp.exp(-time) * similarity
        + 2.0 * jnp.exp(-4.0 * time) * (2.0 * similarity**2 - 1.0)
    )

    assert jnp.allclose(kernel.matrix(points, points), expected)
    assert kernel.spectrum.mode_count == 5
    _assert_psd(kernel.matrix(points, points))


def test_sphere_two_levels_match_legendre_addition_theorem():
    points = jnp.eye(3)
    time = 0.2
    kernel = phx.kernels.SphereSpectralKernel(
        2,
        2,
        phx.kernels.HeatSpectralMultiplier(time),
        normalize=False,
    )
    similarity = points @ points.T
    expected = (
        1.0
        + 3.0 * jnp.exp(-2.0 * time) * similarity
        + 5.0 * jnp.exp(-6.0 * time) * (0.5 * (3.0 * similarity**2 - 1.0))
    )

    assert jnp.allclose(kernel.matrix(points, points), expected)
    assert kernel.spectrum.mode_count == 9


def test_sphere_level_multiplicities_preserve_large_integer_exactness():
    spectrum = phx.metrix.SphereLaplacianLevels(16, 64)
    expected = math.comb(80, 16) - math.comb(78, 16)

    assert expected > 2**53
    assert isinstance(spectrum.multiplicities, tuple)
    assert spectrum.multiplicities[-1] == expected
    assert spectrum.mode_count == sum(spectrum.multiplicities)


def test_sphere_kernel_is_rotation_invariant_unit_diagonal_and_differentiable():
    diagonal = 1.0 / jnp.sqrt(3.0)
    points = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [diagonal, diagonal, diagonal]]
    )
    rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def objective(length_scale):
        kernel = phx.kernels.SphereSpectralKernel(
            2,
            5,
            phx.kernels.MaternSpectralMultiplier(length_scale, 1.4),
        )
        return jnp.sum(kernel.matrix(points, points))

    kernel = phx.kernels.SphereSpectralKernel(
        2,
        5,
        phx.kernels.MaternSpectralMultiplier(0.7, 1.4),
    )
    matrix = kernel.matrix(points, points)

    assert jnp.allclose(jnp.diag(matrix), 1.0)
    assert jnp.allclose(matrix, kernel.matrix(points @ rotation.T, points @ rotation.T))
    assert jnp.isfinite(jax.jit(jax.grad(objective))(jnp.asarray(0.7)))
    _assert_psd(matrix)
    with pytest.raises(Exception, match="unit vectors"):
        kernel.matrix(jnp.asarray([[2.0, 0.0, 0.0]]), points)


def test_tolerance_near_sphere_points_are_canonicalized_before_expansion():
    squared_radii = jnp.asarray([0.9995, 1.0, 1.0005])
    points = jnp.sqrt(squared_radii)[:, None] * jnp.asarray([[1.0, 0.0, 0.0]])
    kernel = phx.kernels.SphereSpectralKernel(
        2,
        2,
        phx.kernels.HeatSpectralMultiplier(0.0),
        normalize=False,
        membership_tolerance=1e-3,
    )
    matrix = kernel.matrix(points, points)

    assert jnp.allclose(jnp.diag(matrix), kernel.diagonal(points))
    _assert_psd(matrix)
    with pytest.raises(ValueError, match="one sphere point"):
        kernel.pairwise(points[:2], points[0])


def test_special_orthogonal_character_kernel_is_biinvariant_and_psd():
    points = jnp.stack([_rotation(0.0), _rotation(0.4), _rotation(-0.8)])
    left = _rotation(0.3)
    right = _rotation(-0.2)
    transformed = jnp.einsum("ij,bjk,kl->bil", left, points, right)
    kernel = phx.kernels.SpecialOrthogonalCharacterKernel(
        2,
        4,
        phx.kernels.MaternSpectralMultiplier(0.6, 1.2),
    )
    matrix = kernel.matrix(points, points)

    assert jnp.allclose(matrix, kernel.matrix(transformed, transformed))
    assert jnp.allclose(jnp.diag(matrix), 1.0)
    _assert_psd(matrix)


def test_homogeneous_diagonal_tracks_tolerance_near_input_and_pairwise_is_scalar():
    point = jnp.sqrt(1.0005) * jnp.eye(2)
    kernel = phx.kernels.SpecialOrthogonalCharacterKernel(
        2,
        1,
        phx.kernels.HeatSpectralMultiplier(0.0),
        membership_tolerance=1e-3,
    )

    assert jnp.allclose(kernel.diagonal(point), jnp.diag(kernel.matrix(point, point)))
    assert not kernel.is_unit_diagonal
    with pytest.raises(ValueError, match="one homogeneous-space point"):
        kernel.pairwise(jnp.stack((jnp.eye(2), point)), point)


def test_special_unitary_character_kernel_is_biinvariant_real_and_psd():
    angles = jnp.asarray([0.0, 0.3, -0.7])
    points = jax.vmap(
        lambda angle: jnp.diag(jnp.asarray([jnp.exp(1j * angle), jnp.exp(-1j * angle)]))
    )(angles)
    left = points[1]
    right = points[2]
    transformed = jnp.einsum("ij,bjk,kl->bil", left, points, right)
    kernel = phx.kernels.SpecialUnitaryCharacterKernel(
        2,
        4,
        phx.kernels.HeatSpectralMultiplier(0.2),
    )
    matrix = kernel.matrix(points, points)

    assert not jnp.iscomplexobj(matrix)
    assert jnp.allclose(matrix, kernel.matrix(transformed, transformed), atol=1e-10)
    assert jnp.allclose(jnp.diag(matrix), 1.0)
    _assert_psd(matrix)


def test_stiefel_kernel_is_left_invariant_and_grassmann_kernel_is_quotient_invariant():
    first = jnp.eye(3)[:, :2]
    second = jnp.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    frames = jnp.stack((first, second))
    ambient_rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = jnp.einsum("ij,bjk->bik", ambient_rotation, frames)
    multiplier = phx.kernels.MaternSpectralMultiplier(0.8, 1.5)
    stiefel = phx.kernels.StiefelSpectralKernel(3, 2, 3, multiplier)
    grassmann = phx.kernels.GrassmannSpectralKernel(3, 2, 3, multiplier)

    stiefel_matrix = stiefel.matrix(frames, frames)
    grassmann_matrix = grassmann.matrix(frames, frames)
    assert jnp.allclose(stiefel_matrix, stiefel.matrix(transformed, transformed))
    assert jnp.allclose(grassmann_matrix, grassmann.matrix(transformed, transformed))

    first_gauge = _rotation(0.4)
    second_gauge = _rotation(-0.7)
    regauged = jnp.stack((frames[0] @ first_gauge, frames[1] @ second_gauge))
    assert jnp.allclose(grassmann_matrix, grassmann.matrix(regauged, regauged))
    _assert_psd(stiefel_matrix)
    _assert_psd(grassmann_matrix)
    assert jnp.allclose(jnp.diag(stiefel_matrix), 1.0)
    assert jnp.allclose(jnp.diag(grassmann_matrix), 1.0)


def test_compact_matrix_kernel_hyperparameter_gradients_are_finite():
    points = jnp.stack([_rotation(0.0), _rotation(0.5), _rotation(-0.4)])

    def objective(length_scale, smoothness):
        kernel = phx.kernels.SpecialOrthogonalCharacterKernel(
            2,
            5,
            phx.kernels.MaternSpectralMultiplier(length_scale, smoothness),
        )
        return jnp.sum(kernel.matrix(points, points))

    gradients = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        jnp.asarray(0.5), jnp.asarray(1.2)
    )
    assert jnp.all(jnp.isfinite(jnp.asarray(gradients)))
