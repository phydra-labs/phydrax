#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (
            phx.kernels.SquaredExponentialKernel(length_scale=0.5),
            jnp.exp(-0.5 * 0.8**2),
        ),
        (
            phx.kernels.Matern32Kernel(length_scale=0.5),
            (1.0 + jnp.sqrt(3.0) * 0.8) * jnp.exp(-jnp.sqrt(3.0) * 0.8),
        ),
        (
            phx.kernels.Matern52Kernel(length_scale=0.5),
            (1.0 + jnp.sqrt(5.0) * 0.8 + 5.0 * 0.8**2 / 3.0)
            * jnp.exp(-jnp.sqrt(5.0) * 0.8),
        ),
        (
            phx.kernels.InverseMultiquadricKernel(length_scale=0.5),
            1.0 / jnp.sqrt(1.0 + 0.8**2),
        ),
    ],
)
def test_stationary_pairwise_values_match_closed_forms(kernel, expected):
    assert jnp.allclose(kernel.pairwise(jnp.asarray([0.1]), jnp.asarray([0.5])), expected)


def test_stationary_ard_matrices_are_symmetric_positive_semidefinite():
    coordinate = jnp.linspace(-1.0, 1.0, 13)
    points = jnp.stack((coordinate, coordinate**2), axis=1)

    for kernel in (
        phx.kernels.SquaredExponentialKernel(length_scale=jnp.array([0.3, 0.8])),
        phx.kernels.Matern32Kernel(length_scale=jnp.array([0.3, 0.8])),
        phx.kernels.Matern52Kernel(length_scale=jnp.array([0.3, 0.8])),
        phx.kernels.InverseMultiquadricKernel(length_scale=jnp.array([0.3, 0.8])),
    ):
        matrix = kernel.matrix(points, points)

        assert matrix.shape == (13, 13)
        assert jnp.allclose(matrix, matrix.T)
        assert jnp.allclose(kernel.diagonal(points), jnp.ones(13))
        assert jnp.linalg.eigvalsh(matrix).min() > -1e-10

    gradient = jax.grad(
        lambda scale: (
            phx.kernels.Matern32Kernel(length_scale=scale).matrix(points, points).sum()
        )
    )(jnp.array([0.3, 0.8]))
    assert jnp.all(jnp.isfinite(gradient))


def test_matern_origin_derivatives_match_process_regularity_moments():
    length_scale = 0.4
    matern32 = phx.kernels.Matern32Kernel(length_scale=length_scale)
    matern52 = phx.kernels.Matern52Kernel(length_scale=length_scale)

    def scalar_covariance(kernel, left, right):
        return kernel.pairwise(jnp.asarray([left]), jnp.asarray([right]))

    matern32_cross = jax.grad(
        jax.grad(lambda left, right: scalar_covariance(matern32, left, right), argnums=0),
        argnums=1,
    )(0.2, 0.2)
    matern52_cross = jax.grad(
        jax.grad(lambda left, right: scalar_covariance(matern52, left, right), argnums=0),
        argnums=1,
    )(0.2, 0.2)
    matern52_fourth = jax.grad(
        jax.grad(
            jax.grad(
                jax.grad(
                    lambda left, right: scalar_covariance(matern52, left, right),
                    argnums=0,
                ),
                argnums=0,
            ),
            argnums=1,
        ),
        argnums=1,
    )(0.2, 0.2)

    assert matern32.max_derivative_order == 1
    assert matern52.max_derivative_order == 2
    assert jnp.allclose(matern32_cross, 3.0 / length_scale**2)
    assert jnp.allclose(matern52_cross, 5.0 / (3.0 * length_scale**2))
    assert jnp.allclose(matern52_fourth, 25.0 / length_scale**4)
