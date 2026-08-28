#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


@pytest.mark.parametrize(
    "kernel_type",
    [
        phx.discretization.WendlandC2SPHKernel,
        phx.discretization.CubicSplineSPHKernel,
    ],
)
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_sph_kernel_is_radially_normalized(kernel_type, dimension):
    kernel = kernel_type(dimension)
    radius = np.linspace(0.0, 2.0, 20_001)
    values = np.asarray(kernel.value(jnp.asarray(radius), 1.0))
    surface_area = {1: 2.0, 2: 2.0 * np.pi, 3: 4.0 * np.pi}[dimension]
    integral = surface_area * np.trapezoid(radius ** (dimension - 1) * values, radius)

    assert integral == pytest.approx(1.0, rel=2e-8, abs=2e-8)


@pytest.mark.parametrize(
    "kernel",
    [
        phx.discretization.WendlandC2SPHKernel(2),
        phx.discretization.CubicSplineSPHKernel(2),
    ],
)
def test_sph_kernel_derivatives_match_automatic_differentiation(kernel):
    distance = jnp.asarray(0.73)
    smoothing_length = jnp.asarray(0.61)
    radial_reference = jax.grad(lambda radius: kernel.value(radius, smoothing_length))(
        distance
    )
    smoothing_reference = jax.grad(lambda length: kernel.value(distance, length))(
        smoothing_length
    )

    assert jnp.allclose(
        kernel.radial_derivative(distance, smoothing_length),
        radial_reference,
        rtol=2e-12,
        atol=2e-12,
    )
    assert jnp.allclose(
        kernel.smoothing_length_derivative(distance, smoothing_length),
        smoothing_reference,
        rtol=2e-12,
        atol=2e-12,
    )


def test_sph_kernel_gradient_is_zero_safe_and_compact():
    kernel = phx.discretization.WendlandC2SPHKernel(2)
    zero = kernel.gradient(jnp.zeros((2,)), jnp.asarray(0.0), 0.5)
    boundary = kernel.value(jnp.asarray(1.0), 0.5)
    outside = kernel.value(jnp.asarray(1.01), 0.5)

    assert jnp.array_equal(zero, jnp.zeros((2,)))
    assert jnp.all(jnp.isfinite(zero))
    assert boundary == pytest.approx(0.0)
    assert outside == pytest.approx(0.0)
    with pytest.raises(Exception, match="non-negative"):
        kernel.value(jnp.asarray(-0.1), 0.5).block_until_ready()
    with pytest.raises(Exception, match="positive"):
        kernel.value(jnp.asarray(0.1), 0.0).block_until_ready()
