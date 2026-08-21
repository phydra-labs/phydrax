#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _problem():
    prior = phx.uq.Normal(0.0, 1.0)
    likelihood = phx.uq.Normal(1.5, 0.5)
    return phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(jnp.asarray(0.0), priors=prior),
        lambda value: likelihood.log_prob(value),
    )


def _config():
    return phx.uq.FlowVariationalConfig(
        initialization=phx.uq.VariationalConfig(
            num_steps=30,
            samples_per_step=8,
            learning_rate=0.03,
            record_every=5,
        ),
        optimization=phx.uq.VariationalConfig(
            num_steps=30,
            samples_per_step=8,
            learning_rate=0.005,
            record_every=5,
        ),
        initialization_samples=64,
        flow_layers=2,
        num_knots=4,
        nn_width=8,
        nn_depth=1,
    )


def test_flow_variational_family_preserves_pytree_sample_density_contract():
    problem = _problem()
    result = phx.uq.fit_flow_variational(
        problem,
        key=jax.random.key(1),
        config=_config(),
        num_samples=32,
    )
    samples, sampled_log_prob = result.family.sample_and_log_prob(
        jax.random.key(2),
        sample_shape=(16,),
    )

    assert result.family.family_id == "flowjax-spline"
    assert samples.shape == (16,)
    assert sampled_log_prob.shape == (16,)
    assert jnp.allclose(sampled_log_prob, result.family.log_prob(samples))
    assert result.samples.shape == (32,)
    assert jnp.all(jnp.isfinite(result.log_target))
    assert jnp.all(jnp.isfinite(result.log_variational))


def test_flow_variational_reuses_explicit_mean_field_initialization():
    problem = _problem()
    config = _config()
    initialization = phx.uq.fit_variational(
        problem,
        key=jax.random.key(3),
        config=config.initialization,
        num_samples=config.initialization_samples,
    )
    result = phx.uq.fit_flow_variational(
        problem,
        key=jax.random.key(4),
        config=config,
        num_samples=16,
        initialization=initialization,
    )

    assert result.initialization is initialization
    assert result.approximation_id == "reverse-kl/flowjax-spline"
    assert jnp.all(result.diagnostics.finite)
