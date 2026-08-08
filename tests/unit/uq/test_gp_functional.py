#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _design(points, functional, name):
    return phx.uq.FunctionalDesign.from_points(points, functional, name=name)


def test_squared_exponential_functional_covariances_match_closed_forms():
    length_scale = jnp.array([0.4, 0.7])
    kernel = phx.kernels.SquaredExponentialKernel(length_scale=length_scale)
    left = jnp.array([[-0.2, 0.3], [0.4, -0.1]])
    right = jnp.array([[0.1, -0.2], [0.7, 0.5], [-0.3, 0.8]])
    value = phx.uq.value_functional(2)
    direction = jnp.array([0.6, -0.8])
    directional = phx.uq.directional_derivative_functional(direction)
    laplacian = phx.uq.laplacian_functional(2)

    base = kernel.matrix(left, right)
    delta = left[:, None, :] - right[None, :, :]
    inverse_variance = 1.0 / length_scale**2
    expected_directional = (
        jnp.sum(
            delta * inverse_variance * direction,
            axis=-1,
        )
        * base
    )
    expected_laplacian = (
        jnp.sum(delta * delta * inverse_variance**2, axis=-1) - jnp.sum(inverse_variance)
    ) * base
    value_directional = phx.uq.functional_kernel_matrix(
        kernel,
        _design(left, value, "left-value"),
        _design(right, directional, "right-direction"),
    )
    laplacian_value = phx.uq.functional_kernel_matrix(
        kernel,
        _design(left, laplacian, "left-laplacian"),
        _design(right, value, "right-value"),
    )
    laplacian_diagonal = phx.uq.functional_kernel_diagonal(
        kernel,
        _design(left, laplacian, "laplacian"),
    )
    expected_laplacian_diagonal = jnp.sum(inverse_variance) ** 2 + 2.0 * jnp.sum(
        inverse_variance**2
    )

    assert jnp.allclose(value_directional, expected_directional)
    assert jnp.allclose(laplacian_value, expected_laplacian)
    assert jnp.allclose(laplacian_diagonal, expected_laplacian_diagonal)


def test_functional_exact_and_fitc_conditioning_are_finite_and_block_ordered():
    points = jnp.linspace(0.05, 0.95, 9)
    derivative_points = jnp.linspace(0.1, 0.9, 7)
    value = phx.uq.value_functional(1)
    derivative = phx.uq.partial_derivative_functional(1, 0)
    blocks = (
        phx.uq.FunctionalObservationBlock(points, value, name="values"),
        phx.uq.FunctionalObservationBlock(
            derivative_points,
            derivative,
            name="derivatives",
        ),
    )
    observations = (
        jnp.sin(2.0 * jnp.pi * points),
        2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * derivative_points),
    )
    model = phx.uq.FunctionalGaussianProcessDiscrepancy(blocks, observations)
    kernel = phx.kernels.SquaredExponentialKernel(length_scale=0.25)
    exact_state = phx.uq.FunctionalGaussianProcessLikelihoodState(
        kernel=kernel,
        noise_scale=jnp.array([0.02, 0.03]),
    )
    inducing = _design(jnp.linspace(0.0, 1.0, 6), value, "inducing")
    sparse_state = phx.uq.FunctionalGaussianProcessLikelihoodState(
        kernel=kernel,
        noise_scale=jnp.array([0.02, 0.03]),
        inducing_design=inducing,
    )
    query = phx.uq.FunctionalDesign(
        (
            phx.uq.FunctionalObservationBlock(
                jnp.linspace(0.0, 1.0, 5),
                value,
                name="query-values",
            ),
            phx.uq.FunctionalObservationBlock(
                jnp.linspace(0.1, 0.9, 4),
                derivative,
                name="query-derivatives",
            ),
        )
    )
    zero_mean = (jnp.zeros_like(points), jnp.zeros_like(derivative_points))
    exact = model.condition(zero_mean, query, state=exact_state)
    sparse = model.condition(zero_mean, query, state=sparse_state)

    assert jnp.isfinite(model.log_marginal_likelihood(zero_mean, state=exact_state))
    assert jnp.isfinite(model.log_marginal_likelihood(zero_mean, state=sparse_state))
    assert tuple(value.shape for value in exact.split_mean()) == ((5,), (4,))
    assert exact.covariance.shape == (9, 9)
    assert sparse.covariance.shape == (9, 9)
    assert jnp.all(exact.variance >= 0.0)
    assert jnp.all(sparse.variance >= 0.0)


def test_dynamic_differential_operator_coefficients_are_jittable_and_differentiable():
    points = jnp.linspace(0.1, 0.9, 6)
    laplacian = phx.uq.laplacian_functional(1)
    derivative = phx.uq.partial_derivative_functional(1, 0)
    kernel = phx.kernels.SquaredExponentialKernel(length_scale=0.35)

    def objective(coefficients):
        diffusion, advection = coefficients
        operator = diffusion * laplacian + advection * derivative
        design = _design(points, operator, "dynamic-pde")
        return jnp.sum(phx.uq.functional_kernel_diagonal(kernel, design))

    coefficients = jnp.array([0.4, -0.2])
    eager = objective(coefficients)
    compiled = jax.jit(objective)(coefficients)
    gradient = jax.grad(objective)(coefficients)

    assert jnp.allclose(eager, compiled)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0.0)


def test_functional_regularity_gate_rejects_unsupported_laplacian():
    design = _design(
        jnp.array([0.2, 0.5]),
        phx.uq.laplacian_functional(1),
        "laplacian",
    )

    with pytest.raises(ValueError, match="requires order 2"):
        phx.uq.functional_kernel_diagonal(
            phx.kernels.Matern32Kernel(length_scale=0.4),
            design,
        )

    supported = phx.uq.functional_kernel_diagonal(
        phx.kernels.Matern52Kernel(length_scale=0.4),
        design,
    )
    assert jnp.allclose(supported, 25.0 / 0.4**4)
