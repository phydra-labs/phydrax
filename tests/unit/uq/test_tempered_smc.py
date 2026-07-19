#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_parameter_space_draws_prior_particles_in_unconstrained_coordinates():
    space = phx.uq.ParameterSpace(
        {"location": jnp.zeros(2), "rate": jnp.asarray(0.0)},
        priors={
            "location": phx.uq.Normal(1.0, 2.0),
            "rate": phx.uq.LogNormal(0.2, 0.4),
        },
        bijectors={
            "location": phx.uq.IdentityBijector(),
            "rate": phx.uq.ExpBijector(),
        },
    )
    position = space.sample_prior(jr.key(1), num_samples=20_000)
    physical = space.constrain(position)

    assert position["location"].shape == (20_000, 2)
    assert position["rate"].shape == (20_000,)
    assert jnp.mean(position["location"]) == pytest.approx(1.0, abs=0.04)
    assert jnp.std(position["location"]) == pytest.approx(2.0, rel=0.03)
    assert jnp.mean(position["rate"]) == pytest.approx(0.2, abs=0.015)
    assert jnp.all(physical["rate"] > 0.0)


def test_tempered_smc_recovers_conjugate_gaussian_and_reports_schedule():
    observation = 1.2
    prior_scale = 2.0
    observation_scale = 0.5
    posterior_variance = 1.0 / (1.0 / prior_scale**2 + 1.0 / observation_scale**2)
    posterior_mean = posterior_variance * observation / observation_scale**2
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        priors=phx.uq.Normal(0.0, prior_scale),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: -0.5 * ((value - observation) / observation_scale) ** 2,
    )
    result = phx.uq.sample_tempered_smc(
        problem,
        key=jr.key(2),
        num_particles=1_000,
        target_ess=0.8,
        num_mcmc_steps=5,
        step_size=0.15,
        num_integration_steps=8,
    )

    assert result.num_particles == 1_000
    assert result.num_tempering_steps >= 2
    assert result.temperatures[0] == 0.0
    assert result.temperatures[-1] == pytest.approx(1.0)
    assert jnp.all(jnp.diff(result.temperatures) > 0.0)
    assert result.effective_sample_sizes.shape == result.temperatures.shape
    assert result.acceptance_rates.shape == (result.num_tempering_steps,)
    assert jnp.all(result.divergence_rates == 0.0)
    assert result.num_unique_initial_particles > 100
    assert result.duration_seconds > 0.0
    assert jnp.mean(result.samples) == pytest.approx(posterior_mean, abs=0.04)
    assert jnp.var(result.samples) == pytest.approx(posterior_variance, rel=0.15)


def test_tempered_smc_requires_prior_particles_for_custom_prior_density():
    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.asarray(0.0),
            log_prior=lambda value: -0.5 * value**2,
        ),
        lambda value: -0.5 * value**2,
    )
    with pytest.raises(ValueError, match="Prior sampling"):
        phx.uq.sample_tempered_smc(problem, key=jr.key(3), num_particles=20)

    with pytest.raises(ValueError, match="leading axis"):
        phx.uq.sample_tempered_smc(
            problem,
            key=jr.key(3),
            num_particles=20,
            prior_position_sampler=lambda key, count: jr.normal(key, (count - 1,)),
        )


def test_tempered_smc_observation_prediction_preserves_particle_axes():
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        priors=phx.uq.Normal(0.0, 1.0),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda parameter: -0.5 * (parameter - 0.5) ** 2,
        sample_observation=lambda key, parameter, query: cx.Field(
            parameter * query + 0.1 * jr.normal(key, query.shape),
            dims=("x",),
        ),
    )
    result = phx.uq.sample_tempered_smc(
        problem,
        key=jr.key(14),
        num_particles=32,
        num_mcmc_steps=1,
        num_integration_steps=3,
    )

    observations = result.predict_observations(
        jr.key(15),
        jnp.linspace(0.0, 1.0, 4),
        num_observation_samples=3,
        observation_dim="measurement",
        batch_size=7,
    )

    assert observations.samples.dims == (
        "__phydra_uq_particle",
        "measurement",
        "x",
    )
    assert observations.samples.shape == (32, 3, 4)
    assert tuple(axis.source for axis in observations.sample_axes) == (
        "epistemic",
        "observation",
    )
