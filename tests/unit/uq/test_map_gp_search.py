# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _posterior():
    space = phx.uq.ParameterSpace(
        jnp.asarray([0.0]),
        log_prior=lambda value: -0.5 * jnp.sum(value**2),
    )
    return phx.uq.PosteriorProblem(
        space,
        lambda value: -0.5 * jnp.sum((value - 0.3) ** 2),
    )


def test_search_map_consumes_canonical_q1_bayesian_optimization():
    problem = _posterior()
    search = phx.uq.GaussianProcessBayesianOptimization(
        8,
        objective_surrogate=phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.Matern52Kernel(length_scale=0.25),
            noise_scale=0.0,
        ),
        initial_evaluations=4,
        batch_size=1,
        candidate_tuple_count=16,
        fantasy_count=8,
    )
    result = phx.uq.search_map(
        problem,
        search,
        key=jr.key(4),
        position_bounds=(jnp.asarray([-2.0]), jnp.asarray([2.0])),
    )
    assert isinstance(result, phx.uq.BayesianOptimizationMAPResult)
    assert result.objective_evaluations == 8
    assert result.evidence.evaluation_count == 8
    assert result.evidence.globally_optimal is False
