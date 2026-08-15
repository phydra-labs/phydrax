#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _conjugate_normal_problem():
    observations = 1.2 + jnp.linspace(-0.3, 0.3, 64)
    prior_scale = 2.0
    source = phx.uq.ArrayMinibatchSource(
        observations,
        batch_size=8,
        seed=101,
    )
    parameter_space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        priors=phx.uq.Normal(0.0, prior_scale),
    )

    def factors(parameter, batch):
        return -0.5 * (batch.data - parameter) ** 2

    def full_likelihood(parameter):
        return jnp.sum(-0.5 * (observations - parameter) ** 2)

    minibatch_problem = phx.uq.MinibatchPosteriorProblem(
        parameter_space,
        factors,
        num_factors=source.num_factors,
        full_log_likelihood=full_likelihood,
    )
    exact_problem = phx.uq.PosteriorProblem(parameter_space, full_likelihood)
    precision = observations.shape[0] + 1.0 / prior_scale**2
    posterior_mean = jnp.sum(observations) / precision
    posterior_variance = 1.0 / precision
    return (
        minibatch_problem,
        exact_problem,
        source,
        posterior_mean,
        posterior_variance,
    )


def test_sgld_recovers_conjugate_posterior_and_step_refinement_reduces_bias():
    problem, exact_problem, source, expected_mean, expected_variance = (
        _conjugate_normal_problem()
    )
    control = phx.uq.build_sgmcmc_control_variate(
        problem,
        source,
        expected_mean,
    )
    coarse = phx.uq.sample_sgld(
        problem,
        source,
        key=jr.key(501),
        step_size=1.0e-2,
        num_chains=4,
        num_burnin=300,
        num_samples=800,
        control_variate=control,
    )
    refined = phx.uq.sample_sgld(
        problem,
        source,
        key=jr.key(502),
        step_size=2.0e-3,
        num_chains=4,
        num_burnin=500,
        num_samples=1200,
        control_variate=control,
    )
    nuts = phx.uq.sample_nuts(
        exact_problem,
        key=jr.key(506),
        num_chains=4,
        num_warmup=200,
        num_samples=400,
        initial_step_size=0.1,
        max_num_doublings=6,
    )
    laplace = phx.uq.fit_laplace(exact_problem, expected_mean)

    coarse_variance_error = jnp.abs(jnp.var(coarse.samples) - expected_variance)
    refined_variance_error = jnp.abs(jnp.var(refined.samples) - expected_variance)
    assert refined_variance_error < coarse_variance_error
    assert jnp.abs(jnp.mean(refined.samples) - expected_mean) < 0.03
    assert refined_variance_error / expected_variance < 0.15
    assert jnp.abs(jnp.mean(nuts.samples) - expected_mean) < 0.03
    assert jnp.abs(jnp.var(nuts.samples) - expected_variance) / expected_variance < 0.2
    assert jnp.allclose(laplace.map_parameters, expected_mean, atol=1.0e-10)
    assert jnp.allclose(laplace.covariance[0, 0], expected_variance, rtol=1.0e-10)


def test_sgnht_recovers_conjugate_posterior_with_thermostat_diagnostics():
    problem, _, source, expected_mean, expected_variance = _conjugate_normal_problem()
    control = phx.uq.build_sgmcmc_control_variate(
        problem,
        source,
        expected_mean,
    )
    result = phx.uq.sample_sgnht(
        problem,
        source,
        key=jr.key(505),
        step_size=5.0e-3,
        diffusion=0.01,
        num_chains=4,
        num_burnin=2000,
        num_samples=5000,
        control_variate=control,
    )
    assert result.thermostat is not None
    assert result.momentum_norm is not None

    assert jnp.abs(jnp.mean(result.samples) - expected_mean) < 0.03
    assert jnp.abs(jnp.var(result.samples) - expected_variance) / expected_variance < 0.25
    assert result.diagnostics.max_rhat < 1.1
    assert result.thermostat.shape == (4, 5000)
    assert result.momentum_norm.shape == (4, 5000)
    assert jnp.all(jnp.isfinite(result.thermostat))
