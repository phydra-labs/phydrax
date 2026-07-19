#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_eki_matches_nuts_on_elliptic_coefficient_inverse_benchmark():
    """Infer diffusivity in -k u''=1 with homogeneous Dirichlet boundaries."""
    true_coefficient = 1.7
    noise_scale = 0.006
    sensors = jnp.linspace(0.08, 0.92, 12)

    def solution(coefficient, coordinates):
        return coordinates * (1.0 - coordinates) / (2.0 * coefficient)

    observations = solution(true_coefficient, sensors) + noise_scale * jr.normal(
        jr.key(960), sensors.shape
    )
    space = phx.uq.ParameterSpace(
        jnp.log(jnp.asarray(1.2)),
        priors=phx.uq.LogNormal(jnp.log(1.2), 0.5),
        bijectors=phx.uq.ExpBijector(),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda coefficient: (
            -0.5
            * jnp.sum(
                ((solution(coefficient, sensors) - observations) / noise_scale) ** 2
            )
        ),
        gauss_newton_residual=lambda coefficient: (
            (solution(coefficient, sensors) - observations) / noise_scale
        ),
    )

    eki = phx.uq.fit_eki(
        problem,
        key=jr.key(961),
        ensemble_size=384,
        target_ess=0.8,
        max_steps=20,
    )
    nuts = phx.uq.sample_nuts(
        problem,
        key=jr.key(962),
        num_chains=2,
        num_warmup=80,
        num_samples=160,
        initial_step_size=0.15,
        max_num_doublings=6,
        chain_method="vectorized",
    )

    eki_mean = jnp.mean(eki.ensemble)
    eki_scale = jnp.std(eki.ensemble)
    nuts_mean = jnp.mean(nuts.samples)
    nuts_scale = jnp.std(nuts.samples)
    prior_prediction = solution(jnp.asarray(1.2), sensors)
    eki_prediction = solution(eki_mean, sensors)
    prior_rmse = jnp.sqrt(jnp.mean((prior_prediction - observations) ** 2))
    eki_rmse = jnp.sqrt(jnp.mean((eki_prediction - observations) ** 2))

    assert eki.converged
    assert eki.diagnostics.forward_solve_count <= 5_000
    assert jnp.abs(eki_mean - nuts_mean) < 0.06
    assert 0.6 < eki_scale / nuts_scale < 1.4
    assert jnp.abs(eki_mean - true_coefficient) < 0.08
    assert eki_rmse < 0.35 * prior_rmse
