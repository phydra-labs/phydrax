#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import coordax as cx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _conjugate_poisson_problem():
    sensor_x = jnp.linspace(0.05, 0.95, 24)
    basis = 0.5 * sensor_x * (1.0 - sensor_x)
    observation_scale = 0.02
    observations = 4.0 * basis
    prior_scale = 3.0
    posterior_variance = 1.0 / (
        1.0 / prior_scale**2 + jnp.vdot(basis, basis) / observation_scale**2
    )
    posterior_mean = posterior_variance * (
        jnp.vdot(basis, observations) / observation_scale**2
    )
    space = phx.uq.ParameterSpace(
        {"source": jnp.asarray(3.8)},
        priors={"source": phx.uq.Normal(0.0, prior_scale)},
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda parameters: (
            -0.5
            * jnp.sum(
                ((observations - parameters["source"] * basis) / observation_scale) ** 2
            )
        ),
        predict=lambda parameters, query: {
            "u": cx.Field(
                parameters["source"] * 0.5 * query * (1.0 - query),
                dims=("x",),
            )
        },
    )
    return problem, posterior_mean, posterior_variance


def test_nuts_and_dense_laplace_recover_the_conjugate_poisson_posterior():
    problem, exact_mean, exact_variance = _conjugate_poisson_problem()
    nuts = phx.uq.sample_nuts(
        problem,
        key=jr.key(10),
        num_chains=4,
        num_warmup=180,
        num_samples=300,
        target_acceptance_rate=0.9,
        max_num_doublings=8,
    )
    laplace = phx.uq.fit_laplace(problem, {"source": exact_mean})
    query = jnp.linspace(0.0, 1.0, 17)
    predictions = nuts.predict(query, batch_size=73)
    assert not isinstance(predictions, phx.uq.PredictiveField)
    prediction = predictions["u"]

    estimated_mean = jnp.mean(nuts.samples["source"])
    estimated_variance = jnp.var(nuts.samples["source"])
    report = nuts.convergence_report(
        max_rhat=1.05,
        min_bulk_ess=100,
        min_tail_ess=50,
    )

    assert nuts.algorithm == "nuts"
    assert nuts.samples["source"].shape == (4, 300)
    assert nuts.unconstrained_samples["source"].shape == (4, 300)
    assert nuts.log_density.shape == (4, 300)
    assert nuts.diagnostics.max_rhat < 1.05
    assert nuts.diagnostics.divergence_count == 0
    assert nuts.diagnostics.min_bulk_ess > 100
    assert report.passed
    assert jnp.abs(estimated_mean - exact_mean) < 0.01
    assert jnp.abs(estimated_variance - exact_variance) < 4e-4
    assert jnp.allclose(laplace.map_parameters["source"], exact_mean)
    assert jnp.allclose(laplace.covariance[0, 0], exact_variance)
    assert prediction.samples.dims == (
        "__phydra_uq_chain",
        "__phydra_uq_draw",
        "x",
    )
    assert prediction.samples.shape == (4, 300, 17)
    assert jnp.allclose(jnp.asarray(prediction.samples.data)[..., (0, -1)], 0.0)


def test_fixed_hmc_retains_trajectory_configuration_and_replays_from_root_key():
    problem, exact_mean, _ = _conjugate_poisson_problem()
    settings: dict[str, Any] = dict(
        num_integration_steps=7,
        num_chains=2,
        num_warmup=80,
        num_samples=100,
        target_acceptance_rate=0.9,
    )

    first = phx.uq.sample_hmc(problem, key=jr.key(20), **settings)
    replay = phx.uq.sample_hmc(problem, key=jr.key(20), **settings)

    assert first.algorithm == "hmc"
    assert jnp.array_equal(first.samples["source"], replay.samples["source"])
    assert jnp.array_equal(first.log_density, replay.log_density)
    assert jnp.array_equal(first.chain_keys, replay.chain_keys)
    assert jnp.all(first.num_integration_steps == 7)
    assert all(warmup.num_integration_steps == 7 for warmup in first.warmup)
    assert first.diagnostics.divergence_count == 0
    assert jnp.abs(jnp.mean(first.samples["source"]) - exact_mean) < 0.03
