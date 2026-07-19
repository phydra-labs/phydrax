#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
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
        kernel="exp_squared",
        jitter=1e-9,
    )
    hyperparameters = {
        "amplitude": 0.2,
        "length_scale": 0.25,
        "noise_scale": 0.01,
    }
    query = cx.Field(jnp.linspace(0.0, 1.0, 31), dims=("x",))

    log_probability = model.log_marginal_likelihood(
        physical_mean,
        **hyperparameters,
    )
    conditioned = model.condition(
        physical_mean,
        query,
        **hyperparameters,
    )
    samples = conditioned.sample(jr.key(0), num_samples=16)
    prediction = conditioned.predictive_field(
        2.0 * query.data,
        jr.key(1),
        num_samples=12,
        observation_variance=hyperparameters["noise_scale"] ** 2,
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
        hyperparameters["noise_scale"] ** 2,
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
        kernel="matern32",
    )
    query = jnp.linspace(0.0, 1.0, 101)
    base = 4.0 * 0.5 * query * (1.0 - query)
    conditioned = model.condition(
        base_at_observations,
        query,
        amplitude=0.03,
        length_scale=0.3,
        noise_scale=0.003,
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
