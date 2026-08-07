#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

import phydrax as phx


def test_flow_nuts_recovers_represented_asymmetric_modes():
    prior_scale = 4.0
    likelihood_scale = 0.35
    mode_location = 2.2
    positive_weight = 0.7
    component_variance = 1.0 / (1.0 / likelihood_scale**2 + 1.0 / prior_scale**2)
    component_mean = component_variance * mode_location / likelihood_scale**2
    expected_mean = (2.0 * positive_weight - 1.0) * component_mean
    expected_variance = component_variance + 4.0 * (
        positive_weight * (1.0 - positive_weight) * component_mean**2
    )

    def log_likelihood(value):
        return jsp.special.logsumexp(
            jnp.stack(
                (
                    jnp.log(1.0 - positive_weight)
                    - 0.5 * ((value + mode_location) / likelihood_scale) ** 2,
                    jnp.log(positive_weight)
                    - 0.5 * ((value - mode_location) / likelihood_scale) ** 2,
                )
            )
        )

    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.asarray(0.0),
            priors=phx.uq.Normal(0.0, prior_scale),
        ),
        log_likelihood,
    )
    config = phx.uq.FlowNUTSConfig(
        num_adaptation_rounds=2,
        num_local_adaptation_steps=30,
        num_global_adaptation_steps=8,
        num_stabilization_steps=20,
        num_local_steps=2,
        num_global_steps=1,
        history_capacity_per_chain=60,
        flow_layers=3,
        num_knots=8,
        nn_width=24,
        nn_depth=1,
        learning_rate=1e-3,
        max_epochs=20,
        max_patience=20,
        batch_size=32,
        validation_fraction=0.2,
    )
    result = phx.uq.sample_flow_nuts(
        problem,
        key=jr.key(41),
        num_chains=4,
        num_warmup=100,
        num_samples=250,
        initial_positions=jnp.asarray([-2.2, -2.0, 2.0, 2.2]),
        target_acceptance_rate=0.9,
        max_num_doublings=7,
        config=config,
        chain_method="vectorized",
    )
    samples = result.samples
    positive_mass = jnp.mean(samples > 0.0)
    transitions = jnp.sum((samples[:, 1:] > 0.0) != (samples[:, :-1] > 0.0))

    assert jnp.abs(positive_mass - positive_weight) < 0.1
    assert jnp.abs(jnp.mean(samples) - expected_mean) < 0.25
    assert jnp.abs(jnp.var(samples) - expected_variance) / expected_variance < 0.2
    assert transitions >= 4
    assert result.diagnostics.max_rhat < 1.15
    assert result.diagnostics.divergence_count == 0
    assert jnp.sum(result.global_accepted_count) > 0
