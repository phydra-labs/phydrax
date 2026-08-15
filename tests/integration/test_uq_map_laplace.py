#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_map_to_laplace_pipeline_recovers_nonlinear_transformed_inverse_problem():
    x = jnp.linspace(0.0, 2.0, 30)
    observation_scale = 0.03
    true_amplitude = 1.7
    true_rate = 0.8
    observations = true_amplitude * jnp.exp(-true_rate * x)
    space = phx.uq.ParameterSpace(
        {"amplitude": jnp.asarray(1.4), "rate": jnp.log(jnp.asarray(0.6))},
        priors={
            "amplitude": phx.uq.Normal(0.0, 3.0),
            "rate": phx.uq.LogNormal(jnp.log(0.8), 0.5),
        },
        bijectors={
            "amplitude": phx.uq.IdentityBijector(),
            "rate": phx.uq.ExpBijector(),
        },
    )
    likelihood = phx.uq.GaussianLikelihood(observation_scale)
    problem = phx.uq.PosteriorProblem(
        space,
        lambda parameters: jnp.sum(
            likelihood.log_prob(
                parameters["amplitude"] * jnp.exp(-parameters["rate"] * x),
                observations,
            )
        ),
        predict=lambda parameters, query: cx.Field(
            parameters["amplitude"] * jnp.exp(-parameters["rate"] * query),
            dims=("x",),
        ),
    )

    mode = phx.uq.find_map(problem, gradient_tolerance=1e-7)
    laplace = phx.uq.fit_laplace(problem, mode.position)
    prediction = laplace.predict(
        jr.key(0),
        x,
        num_samples=128,
        batch_size=31,
    )
    assert isinstance(prediction, phx.uq.PredictiveField)

    assert mode.converged
    assert jnp.abs(mode.parameters["amplitude"] - true_amplitude) < 2e-3
    assert jnp.abs(mode.parameters["rate"] - true_rate) < 2e-3
    assert laplace.gradient_norm < 1e-7
    assert prediction.samples.shape == (128, 30)
    assert (
        jnp.sqrt(jnp.mean((jnp.asarray(prediction.mean().data) - observations) ** 2))
        < 3e-3
    )
