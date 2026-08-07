#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_linearized_eiv_posterior_matches_explicit_latent_input_nuts_reference():
    true_slope = 2.0
    input_scale = 0.5
    observation_scale = 0.12
    latent_inputs = jnp.linspace(-2.0, 2.0, 12)
    input_key, observation_key = jr.split(jr.key(46))
    measured_inputs = latent_inputs + input_scale * jr.normal(
        input_key,
        latent_inputs.shape,
    )
    measured_targets = true_slope * latent_inputs + observation_scale * jr.normal(
        observation_key,
        latent_inputs.shape,
    )

    slope_space = phx.uq.ParameterSpace(
        jnp.asarray(1.8),
        priors=phx.uq.Normal(0.0, 3.0),
    )
    eiv_term = phx.uq.LinearizedGaussianMeasurementLikelihood(
        lambda slope, value: slope * value[0],
        measured_inputs[:, None],
        measured_targets,
        input_covariance=jnp.asarray([[input_scale**2]]),
        observation_covariance=jnp.asarray([[observation_scale**2]]),
    )
    eiv_problem = phx.uq.PosteriorProblem.from_terms(slope_space, (eiv_term,))

    latent_space = phx.uq.ParameterSpace(
        {"slope": jnp.asarray(1.8), "input_error": jnp.zeros_like(measured_inputs)},
        priors={
            "slope": phx.uq.Normal(0.0, 3.0),
            "input_error": phx.uq.Normal(0.0, input_scale),
        },
    )
    observation_likelihood = phx.uq.GaussianLikelihood(observation_scale)
    latent_problem = phx.uq.PosteriorProblem(
        latent_space,
        lambda parameters: jnp.sum(
            observation_likelihood.log_prob(
                parameters["slope"]
                * (measured_inputs + parameters["input_error"]),
                measured_targets,
            )
        ),
    )
    settings = dict(
        num_chains=4,
        num_warmup=300,
        num_samples=500,
        initial_step_size=0.15,
        target_acceptance_rate=0.9,
        max_num_doublings=8,
        chain_method="vectorized",
    )
    eiv = phx.uq.sample_nuts(eiv_problem, key=jr.key(81), **settings)
    latent = phx.uq.sample_nuts(latent_problem, key=jr.key(82), **settings)
    eiv_draws = eiv.samples.reshape((-1,))
    latent_slope_draws = latent.samples["slope"].reshape((-1,))
    eiv_mean = jnp.mean(eiv_draws)
    latent_mean = jnp.mean(latent_slope_draws)

    ordinary_precision = (
        jnp.sum(measured_inputs**2) / observation_scale**2 + 1.0 / 3.0**2
    )
    ordinary_mean = (
        jnp.sum(measured_inputs * measured_targets) / observation_scale**2
    ) / ordinary_precision

    assert eiv.convergence_report(
        max_rhat=1.04,
        min_bulk_ess=150.0,
        min_tail_ess=100.0,
    ).passed
    assert latent.diagnostics.rhat["slope"] < 1.04
    assert latent.diagnostics.bulk_ess["slope"] > 150.0
    assert latent.diagnostics.tail_ess["slope"] > 100.0
    assert not jnp.any(latent.divergent)
    assert jnp.abs(eiv_mean - latent_mean) < 0.06
    assert jnp.abs(ordinary_mean - latent_mean) > 0.25
    assert jnp.abs(eiv_mean - true_slope) < jnp.abs(ordinary_mean - true_slope)
    assert jnp.allclose(
        jnp.var(eiv_draws, ddof=1),
        jnp.var(latent_slope_draws, ddof=1),
        rtol=0.20,
        atol=0.005,
    )
