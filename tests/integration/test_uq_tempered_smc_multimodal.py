#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import pytest

import phydrax as phx


def test_tempered_smc_preserves_both_modes_of_symmetric_inverse_posterior():
    prior_scale = 3.0
    likelihood_scale = 0.3
    mode_location = 2.0
    component_variance = 1.0 / (1.0 / likelihood_scale**2 + 1.0 / prior_scale**2)
    component_mean = component_variance * mode_location / likelihood_scale**2
    expected_variance = component_variance + component_mean**2

    def log_likelihood(value):
        return jsp.special.logsumexp(
            jnp.stack(
                [
                    -0.5 * ((value - mode_location) / likelihood_scale) ** 2,
                    -0.5 * ((value + mode_location) / likelihood_scale) ** 2,
                ]
            )
        )

    space = phx.uq.ParameterSpace(
        jnp.asarray(1.8),
        priors=phx.uq.Normal(0.0, prior_scale),
    )
    problem = phx.uq.PosteriorProblem(space, log_likelihood)
    smc = phx.uq.sample_tempered_smc(
        problem,
        key=jr.key(30),
        num_particles=1_500,
        target_ess=0.8,
        num_mcmc_steps=5,
        step_size=0.15,
        num_integration_steps=8,
    )
    local_pathfinder = phx.uq.fit_pathfinder(
        problem,
        key=jr.key(31),
        num_samples=1_500,
        num_elbo_samples=100,
        max_steps=50,
    )

    positive_mass = jnp.mean(smc.samples > 0.0)
    local_positive_mass = jnp.mean(local_pathfinder.samples > 0.0)
    assert jnp.abs(positive_mass - 0.5) < 0.07
    assert jnp.abs(jnp.mean(smc.samples)) < 0.2
    assert jnp.var(smc.samples) == pytest.approx(expected_variance, rel=0.08)
    assert local_positive_mass > 0.98
    assert smc.num_unique_initial_particles > 250
    assert smc.temperatures[-1] == 1.0
    assert jnp.max(smc.divergence_rates) < 0.01
