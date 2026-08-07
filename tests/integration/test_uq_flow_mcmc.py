#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_flow_nuts_recovers_conjugate_posterior_and_predictive_axes():
    sensor_x = jnp.linspace(0.05, 0.95, 16)
    basis = 0.5 * sensor_x * (1.0 - sensor_x)
    observation_scale = 0.05
    observations = 2.5 * basis
    prior_scale = 2.0
    exact_variance = 1.0 / (
        1.0 / prior_scale**2 + jnp.vdot(basis, basis) / observation_scale**2
    )
    exact_mean = exact_variance * (jnp.vdot(basis, observations) / observation_scale**2)
    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            {"source": jnp.asarray(2.0)},
            priors={"source": phx.uq.Normal(0.0, prior_scale)},
        ),
        lambda parameters: (
            -0.5
            * jnp.sum(
                ((observations - parameters["source"] * basis) / observation_scale) ** 2
            )
        ),
        predict=lambda parameters, query: cx.Field(
            parameters["source"] * 0.5 * query * (1.0 - query),
            dims=("x",),
        ),
    )
    config = phx.uq.FlowNUTSConfig(
        num_adaptation_rounds=2,
        num_local_adaptation_steps=16,
        num_global_adaptation_steps=4,
        num_stabilization_steps=8,
        num_local_steps=2,
        num_global_steps=1,
        history_capacity_per_chain=32,
        flow_layers=2,
        num_knots=6,
        nn_width=16,
        nn_depth=1,
        max_epochs=8,
        max_patience=8,
        batch_size=16,
        validation_fraction=0.2,
    )

    result = phx.uq.sample_flow_nuts(
        problem,
        key=jr.key(40),
        num_chains=4,
        num_warmup=80,
        num_samples=120,
        target_acceptance_rate=0.9,
        max_num_doublings=7,
        config=config,
        chain_method="vectorized",
    )
    prediction = result.predict(jnp.linspace(0.0, 1.0, 9), batch_size=64)
    estimated_mean = jnp.mean(result.samples["source"])
    estimated_variance = jnp.var(result.samples["source"])

    assert result.samples["source"].shape == (4, 120)
    assert jnp.abs(estimated_mean - exact_mean) < 0.025
    assert jnp.abs(estimated_variance - exact_variance) < 0.003
    assert result.diagnostics.max_rhat < 1.1
    assert result.diagnostics.divergence_count == 0
    assert jnp.sum(result.global_accepted_count) > 0
    assert prediction.samples.dims == (
        "__phydra_uq_chain",
        "__phydra_uq_draw",
        "x",
    )
    assert prediction.samples.shape == (4, 120, 9)
