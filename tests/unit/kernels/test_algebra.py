#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_kernel_algebra_preserves_covariance_and_flattens_associative_nodes():
    points = jnp.linspace(-1.0, 1.0, 15)[:, None]
    squared_exponential = phx.kernels.SquaredExponentialKernel(length_scale=0.3)
    matern32 = phx.kernels.Matern32Kernel(length_scale=0.5)
    matern52 = phx.kernels.Matern52Kernel(length_scale=0.8)

    kernel = 1.7 * squared_exponential + matern32 * matern52 + squared_exponential
    matrix = kernel.matrix(points, points)
    expected = (
        1.7 * squared_exponential.matrix(points, points)
        + matern32.matrix(points, points) * matern52.matrix(points, points)
        + squared_exponential.matrix(points, points)
    )

    assert isinstance(kernel, phx.kernels.SumKernel)
    assert len(kernel.kernels) == 3
    assert isinstance(kernel.kernels[1], phx.kernels.ProductKernel)
    assert jnp.allclose(matrix, expected)
    assert jnp.linalg.eigvalsh(matrix).min() > -1e-10
    assert kernel.max_derivative_order == 1
    assert kernel.kernel_id == (
        "SumKernel[ScaleKernel[SquaredExponentialKernel],"
        "ProductKernel[Matern32Kernel,Matern52Kernel],"
        "SquaredExponentialKernel]"
    )


def test_scale_and_amplitude_have_distinct_explicit_semantics():
    points = jnp.linspace(0.0, 1.0, 8)[:, None]
    correlation = phx.kernels.Matern52Kernel(length_scale=0.25)
    scaled = phx.kernels.ScaleKernel(correlation, 0.4)
    amplituded = phx.kernels.AmplitudeKernel(correlation, 0.4)

    assert jnp.allclose(scaled.diagonal(points), 0.4)
    assert jnp.allclose(amplituded.diagonal(points), 0.4**2)
    assert not scaled.is_unit_diagonal
    assert not amplituded.is_unit_diagonal

    with pytest.raises(TypeError, match="only be added"):
        correlation + 1.0
    with pytest.raises(TypeError):
        _ = correlation - correlation
