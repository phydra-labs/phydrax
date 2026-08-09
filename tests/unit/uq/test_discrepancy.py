#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_exact_gp_discrepancy_marginalizes_and_conditions_coherent_functions():
    observation_x = jnp.linspace(0.0, 1.0, 9)
    physical_mean = 2.0 * observation_x
    discrepancy = 0.15 * jnp.sin(2.0 * jnp.pi * observation_x)
    observations = physical_mean + discrepancy
    model = phx.uq.ExactGaussianProcessDiscrepancy(
        observation_x,
        observations,
    )
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.AmplitudeKernel(
            phx.kernels.SquaredExponentialKernel(length_scale=0.25),
            0.2,
        ),
        noise_scale=0.01,
        jitter=1e-9,
    )
    query = cx.Field(jnp.linspace(0.0, 1.0, 31), dims=("x",))

    log_probability = model.log_marginal_likelihood(
        physical_mean,
        state=state,
    )
    conditioned = model.condition(
        physical_mean,
        query,
        state=state,
    )
    samples = conditioned.sample(jr.key(0), num_samples=16)
    prediction = conditioned.predictive_field(
        2.0 * query.data,
        jr.key(1),
        num_samples=12,
        observation_variance=state.noise_scale**2,
    )

    assert jnp.isfinite(log_probability)
    assert conditioned.output_dims == ("x",)
    assert conditioned.mean.shape == (31,)
    assert conditioned.covariance.shape == (31, 31)
    assert jnp.all(conditioned.variance >= 0.0)
    assert samples.shape == (16, 31)
    assert jnp.array_equal(samples, conditioned.sample(jr.key(0), num_samples=16))
    assert prediction.samples.dims == ("__phydra_uq_discrepancy", "x")
    assert prediction.samples.shape == (12, 31)
    assert jnp.allclose(
        prediction.observation_variance().data,
        state.noise_scale**2,
    )
    expected_discrepancy = 0.15 * jnp.sin(2.0 * jnp.pi * query.data)
    assert jnp.sqrt(jnp.mean((conditioned.mean - expected_discrepancy) ** 2)) < 0.02


def test_gp_discrepancy_improves_a_misspecified_physical_model():
    truth = lambda x: 4.0 * 0.5 * x * (1.0 - x) + 0.03 * jnp.sin(2.0 * jnp.pi * x)
    observation_x = jnp.linspace(0.0, 1.0, 15)
    base_at_observations = 4.0 * 0.5 * observation_x * (1.0 - observation_x)
    model = phx.uq.ExactGaussianProcessDiscrepancy(
        observation_x,
        truth(observation_x),
    )
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.AmplitudeKernel(
            phx.kernels.Matern32Kernel(length_scale=0.3),
            0.03,
        ),
        noise_scale=0.003,
    )
    query = jnp.linspace(0.0, 1.0, 101)
    base = 4.0 * 0.5 * query * (1.0 - query)
    conditioned = model.condition(
        base_at_observations,
        query,
        state=state,
        output_dim="x",
    )

    base_rmse = jnp.sqrt(jnp.mean((base - truth(query)) ** 2))
    corrected_rmse = jnp.sqrt(jnp.mean((base + conditioned.mean - truth(query)) ** 2))

    assert corrected_rmse < 0.1 * base_rmse


def test_gp_discrepancy_rejects_misaligned_scalar_observations():
    with pytest.raises(ValueError, match="align"):
        phx.uq.ExactGaussianProcessDiscrepancy(
            jnp.linspace(0.0, 1.0, 4),
            jnp.ones(3),
        )
    with pytest.raises(ValueError, match="scalar-output"):
        phx.uq.ExactGaussianProcessDiscrepancy(
            jnp.linspace(0.0, 1.0, 4),
            jnp.ones((4, 2)),
        )


def test_exact_scalar_gp_supports_path_valued_kernel_inputs():
    observation_paths = jnp.cumsum(
        jr.normal(jr.key(30), (7, 5, 2)) * 0.25,
        axis=1,
    )
    query_paths = jnp.cumsum(
        jr.normal(jr.key(31), (4, 7, 2)) * 0.25,
        axis=1,
    )
    observations = 0.3 * observation_paths[:, -1, 0] - 0.2 * observation_paths[:, -1, 1]
    kernel = phx.kernels.SignaturePDEKernel(
        phx.kernels.LinearKernel(),
        polynomial_order=4,
        pair_block_size=3,
    )
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=kernel,
        noise_scale=0.05,
        jitter=1e-9,
    )
    model = phx.uq.ExactGaussianProcessDiscrepancy(
        observation_paths,
        observations,
    )
    residual = model.residual(jnp.zeros_like(observations))
    factor = model.factor(state=state)
    conditioned = factor.condition(residual, query_paths, output_dim="path")

    observation_covariance = kernel.matrix(observation_paths, observation_paths)
    observation_covariance = observation_covariance + (
        state.noise_scale**2 + state.jitter
    ) * jnp.eye(observation_paths.shape[0])
    cross_covariance = kernel.matrix(query_paths, observation_paths)
    expected_projection = jnp.linalg.solve(observation_covariance, cross_covariance.T).T
    expected_covariance = (
        kernel.matrix(query_paths, query_paths) - expected_projection @ cross_covariance.T
    )

    assert jnp.isfinite(factor.log_probability(residual))
    assert conditioned.query_points.shape == query_paths.shape
    assert conditioned.output_dims == ("path",)
    assert jnp.allclose(conditioned.mean, expected_projection @ residual)
    assert jnp.allclose(conditioned.covariance, expected_covariance)
    assert jnp.all(conditioned.variance >= 0.0)


def test_sparse_scalar_gp_supports_different_path_design_lengths():
    observation_paths = jnp.cumsum(
        jr.normal(jr.key(40), (8, 6, 2)) * 0.2,
        axis=1,
    )
    inducing_paths = jnp.cumsum(
        jr.normal(jr.key(41), (3, 4, 2)) * 0.2,
        axis=1,
    )
    query_paths = jnp.cumsum(
        jr.normal(jr.key(42), (5, 7, 2)) * 0.2,
        axis=1,
    )
    observations = observation_paths[:, -1, 0]
    kernel = phx.kernels.SignaturePDEKernel(
        phx.kernels.SquaredExponentialKernel(length_scale=0.8),
        polynomial_order=4,
        pair_block_size=3,
    )
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=kernel,
        noise_scale=0.08,
    )
    model = phx.uq.SparseGaussianProcessDiscrepancy(
        observation_paths,
        observations,
        inducing_paths,
    )
    factor = model.factor(state=state)
    conditioned = factor.condition(
        model.residual(jnp.zeros_like(observations)),
        query_paths,
        output_dim="path",
    )

    assert jnp.isfinite(
        model.log_marginal_likelihood(
            jnp.zeros_like(observations),
            state=state,
        )
    )
    assert factor.observation_points.shape == observation_paths.shape
    assert factor.inducing_points.shape == inducing_paths.shape
    assert conditioned.mean.shape == (5,)
    assert conditioned.covariance.shape == (5, 5)
    assert jnp.all(conditioned.variance >= 0.0)


def test_path_valued_gp_likelihood_is_differentiable_and_functional_gp_rejects_it():
    paths = jnp.cumsum(jr.normal(jr.key(50), (5, 4, 2)) * 0.15, axis=1)
    observations = paths[:, -1, 0]
    model = phx.uq.ExactGaussianProcessDiscrepancy(paths, observations)

    def objective(scale):
        state = phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.SignaturePDEKernel(
                phx.kernels.ScaleKernel(phx.kernels.LinearKernel(), scale),
                polynomial_order=3,
            ),
            noise_scale=0.05,
        )
        return model.log_marginal_likelihood(
            jnp.zeros_like(observations),
            state=state,
        )

    gradient = jax.grad(objective)(jnp.asarray(0.7))
    assert jnp.isfinite(gradient)

    with pytest.raises(ValueError, match="input_ndim == 1"):
        phx.uq.FunctionalGaussianProcessLikelihoodState(
            kernel=phx.kernels.SignaturePDEKernel(phx.kernels.LinearKernel()),
            noise_scale=0.1,
        )
