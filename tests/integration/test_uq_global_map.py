#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_global_map_to_local_map_and_laplace_prediction_pipeline():
    x = jnp.linspace(0.0, 2.0, 24)
    observation_scale = 0.03
    true_amplitude = 1.7
    true_rate = 0.8
    observations = true_amplitude * jnp.exp(-true_rate * x)
    space = phx.uq.ParameterSpace(
        {
            "amplitude": jnp.asarray(-1.5),
            "rate": jnp.log(jnp.asarray(1.8)),
        },
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
    search = phx.optim.DifferentialEvolutionSearch(
        16,
        10,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        design=phx.sampling.SobolDesign(scrambled=True),
    )

    global_mode = phx.uq.search_map(
        problem,
        search,
        key=jr.key(30),
        position_bounds=(
            {"amplitude": -3.0, "rate": -3.0},
            {"amplitude": 3.0, "rate": 1.0},
        ),
    )
    local_mode = phx.uq.find_map(
        problem,
        global_mode.position,
        gradient_tolerance=1e-7,
    )
    laplace = phx.uq.fit_laplace(problem, local_mode.position)
    prediction = laplace.predict(
        jr.key(31),
        x,
        num_samples=32,
        batch_size=11,
    )

    assert jnp.isfinite(global_mode.objective)
    assert local_mode.converged
    assert local_mode.objective <= global_mode.objective + 1e-8
    assert jnp.abs(local_mode.parameters["amplitude"] - true_amplitude) < 3e-3
    assert jnp.abs(local_mode.parameters["rate"] - true_rate) < 3e-3
    assert laplace.gradient_norm < 1e-7
    assert isinstance(prediction, phx.uq.PredictiveField)
    assert prediction.samples.shape == (32, 24)
    assert (
        jnp.sqrt(jnp.mean((jnp.asarray(prediction.mean().data) - observations) ** 2))
        < 4e-3
    )
