#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _hamming_points(dimension, alphabet_size):
    return jnp.asarray(
        tuple(itertools.product(range(alphabet_size), repeat=dimension)),
        dtype=jnp.int32,
    )


def _hamming_laplacian(points):
    distance = np.sum(
        np.asarray(points)[:, None, :] != np.asarray(points)[None, :, :], axis=-1
    )
    adjacency = (distance == 1).astype(float)
    return np.diag(np.sum(adjacency, axis=1)) - adjacency


def test_hamming_heat_kernel_matches_explicit_full_laplacian_spectrum():
    points = _hamming_points(2, 3)
    diffusion_time = 0.23
    kernel = phx.kernels.HammingSpectralKernel(
        2,
        3,
        phx.kernels.HeatSpectralMultiplier(diffusion_time),
    )
    laplacian = _hamming_laplacian(points)
    values, vectors = np.linalg.eigh(laplacian)
    raw = points.shape[0] * (
        (vectors * np.exp(-diffusion_time * values)[None, :]) @ vectors.T
    )
    expected = raw / np.mean(np.diag(raw))

    assert jnp.allclose(kernel.matrix(points, points), expected, atol=1e-10)
    assert jnp.allclose(kernel.diagonal(points), 1.0)


def test_hypercube_is_exact_binary_hamming_specialization():
    points = _hamming_points(4, 2)
    multiplier = phx.kernels.MaternSpectralMultiplier(0.7, 1.3)
    hamming = phx.kernels.HammingSpectralKernel(4, 2, multiplier, max_level=3)
    hypercube = phx.kernels.HypercubeSpectralKernel(4, multiplier, max_level=3)

    assert jnp.allclose(hamming.matrix(points, points), hypercube.matrix(points, points))


def test_hamming_kernel_is_invariant_to_coordinate_and_symbol_permutations():
    points = _hamming_points(3, 3)
    kernel = phx.kernels.HammingSpectralKernel(
        3,
        3,
        phx.kernels.HeatSpectralMultiplier(0.15),
    )
    coordinate_permutation = jnp.asarray([2, 0, 1])
    symbol_permutation = jnp.asarray([2, 0, 1])
    transformed = symbol_permutation[points[:, coordinate_permutation]]

    assert jnp.allclose(
        kernel.matrix(points, points), kernel.matrix(transformed, transformed)
    )


def test_truncated_hamming_levels_remain_positive_semidefinite():
    points = _hamming_points(4, 3)
    kernel = phx.kernels.HammingSpectralKernel(
        4,
        3,
        phx.kernels.MaternSpectralMultiplier(0.5, 1.1),
        max_level=2,
    )
    matrix = np.asarray(kernel.matrix(points, points))

    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-9
    assert np.allclose(np.diag(matrix), 1.0)


def test_high_dimensional_hamming_recurrence_and_gradients_are_finite():
    points = jnp.stack(
        (
            jnp.zeros((256,), dtype=jnp.int32),
            jnp.ones((256,), dtype=jnp.int32),
            jnp.arange(256, dtype=jnp.int32) % 7,
        )
    )

    def objective(length_scale, smoothness):
        kernel = phx.kernels.HammingSpectralKernel(
            256,
            7,
            phx.kernels.MaternSpectralMultiplier(length_scale, smoothness),
        )
        return jnp.sum(kernel.matrix(points, points))

    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
        jnp.asarray(0.6), jnp.asarray(1.4)
    )

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(jnp.asarray(gradients)))


def test_hamming_kernel_rejects_invalid_symbols_and_level_cutoffs():
    multiplier = phx.kernels.HeatSpectralMultiplier(0.2)
    with pytest.raises(ValueError, match="max_level"):
        phx.kernels.HammingSpectralKernel(3, 2, multiplier, max_level=4)

    kernel = phx.kernels.HammingSpectralKernel(3, 2, multiplier)
    for invalid in (
        jnp.asarray([[0.0, 1.0, 0.5]]),
        jnp.asarray([[0, 1, 2]]),
    ):
        with pytest.raises(Exception, match="in-range integers"):
            kernel.matrix(invalid, invalid)
    with pytest.raises(ValueError, match="one Hamming point"):
        kernel.pairwise(jnp.asarray([[0, 0, 0], [1, 1, 1]]), jnp.zeros((3,)))
