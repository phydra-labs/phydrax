# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp
import pytest

import phydrax as phx


def _dense_condition(kernel, train_times, query_times, values, noise):
    observation = kernel.matrix(train_times[:, None], train_times[:, None])
    observation = observation + noise**2 * jnp.eye(train_times.size)
    cross = kernel.matrix(query_times[:, None], train_times[:, None])
    query = kernel.matrix(query_times[:, None], query_times[:, None])
    factor = jnp.linalg.cholesky(observation)
    alpha = jnp.linalg.solve(factor.T, jnp.linalg.solve(factor, values))
    projection = jnp.linalg.solve(factor, cross.T)
    return cross @ alpha, jnp.diag(query - projection.T @ projection)


@pytest.mark.parametrize(
    "kernel",
    [
        phx.kernels.Matern32Kernel(length_scale=0.7),
        phx.kernels.Matern52Kernel(length_scale=0.7),
        phx.kernels.SHOKernel(frequency=1.2, quality_factor=0.8, variance=1.3),
    ],
)
def test_bounded_temporal_components_match_dense_gp(kernel):
    train_times = jnp.asarray([0.0, 0.25, 0.8, 1.4])
    query_times = jnp.asarray([-0.2, 0.5, 1.9])
    values = jnp.asarray([0.4, -0.1, 0.7, 0.2])
    noise = jnp.asarray(0.05)
    plan = phx.uq.compile_state_space_kernel(
        kernel,
        phx.uq.StateSpaceGaussianProcessDesign(train_times, query_times),
    )
    result = phx.uq.fit_state_space_gaussian_process(plan, values, noise_scale=noise)
    expected_mean, expected_variance = _dense_condition(
        kernel, train_times, query_times, values, noise
    )
    assert jnp.allclose(result.posterior_mean, expected_mean, atol=2e-4)
    assert jnp.allclose(result.posterior_variance, expected_variance, atol=2e-4)


def test_sums_repeated_rows_vector_noise_and_parallel_covariance():
    kernel = phx.kernels.Matern32Kernel(
        length_scale=0.4
    ) + 0.3 * phx.kernels.Matern52Kernel(length_scale=0.9)
    design = phx.uq.StateSpaceGaussianProcessDesign(
        jnp.asarray([0.0, 0.0, 0.7]),
        jnp.asarray([0.2, 1.0]),
    )
    plan = phx.uq.compile_state_space_kernel(
        kernel,
        design,
        max_observations_per_time=2,
    )
    values = jnp.asarray([0.1, 0.2, -0.3])
    noise = jnp.asarray([0.03, 0.04, 0.05])
    sequential = phx.uq.fit_state_space_gaussian_process(
        plan,
        values,
        noise_scale=noise,
        temporal_method="sequential",
        covariance_form="covariance",
    )
    parallel = phx.uq.fit_state_space_gaussian_process(
        plan,
        values,
        noise_scale=noise,
        temporal_method="parallel",
        covariance_form="covariance",
    )
    assert jnp.allclose(sequential.posterior_mean, parallel.posterior_mean, atol=2e-4)
    assert jnp.allclose(
        sequential.posterior_variance, parallel.posterior_variance, atol=2e-4
    )
    with pytest.raises(ValueError, match="Parallel square-root"):
        phx.uq.fit_state_space_gaussian_process(
            plan,
            values,
            noise_scale=noise,
            temporal_method="parallel",
            covariance_form="square_root",
        )


def test_time_derivative_and_state_capacity_guards():
    design = phx.uq.StateSpaceGaussianProcessDesign(
        jnp.asarray([0.0, 0.4]),
        jnp.asarray([0.2]),
        train_time_derivative_order=jnp.asarray([0, 1]),
        query_time_derivative_order=jnp.asarray([1]),
    )
    plan = phx.uq.compile_state_space_kernel(
        phx.kernels.Matern32Kernel(length_scale=0.5),
        design,
    )
    assert plan.observation_matrices.shape[-1] == 2
    with pytest.raises(ValueError, match="max_state_dimension"):
        phx.uq.compile_state_space_kernel(
            phx.kernels.Matern32Kernel(length_scale=0.5)
            + phx.kernels.Matern52Kernel(length_scale=0.8),
            design,
            max_state_dimension=2,
        )


def test_carma_stability_and_supported_algebra_fail_closed():
    stable = phx.kernels.CARMAKernel([2.0, 1.0], [1.0], 0.5)
    assert jnp.all(
        jnp.isfinite(stable.matrix(jnp.asarray([0.0, 0.5]), jnp.asarray([0.0, 0.5])))
    )
    with pytest.raises(ValueError, match="stable"):
        phx.kernels.CARMAKernel([-1.0], [1.0], 1.0)
    design = phx.uq.StateSpaceGaussianProcessDesign(
        jnp.asarray([0.0]), jnp.asarray([0.5])
    )
    with pytest.raises(TypeError, match="supports"):
        phx.uq.compile_state_space_kernel(
            phx.kernels.Matern32Kernel() * phx.kernels.Matern52Kernel(),
            design,
        )
