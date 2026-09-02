# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_q_batch_mixed_constrained_bo_has_exact_budget_and_replay():
    categorical = phx.optim.FiniteProductSpace(
        {"model": phx.optim.FiniteAxis(jnp.asarray([0.0, 1.0, 2.0]))}
    )
    domain = phx.uq.BayesianOptimizationDomain(
        jnp.asarray([0.4]),
        lower_bounds=jnp.asarray([0.0]),
        upper_bounds=jnp.asarray([1.0]),
        categorical=categorical,
    )

    def objective(point):
        category = point.categorical["model"]
        return (point.continuous[0] - 0.25) ** 2 + 0.1 * category

    pending = domain.decode(jnp.asarray([0.6]), jnp.asarray(1))
    problem = phx.uq.BayesianOptimizationProblem(
        objective,
        domain,
        constraints=(lambda point: point.continuous[0] - 0.8,),
        pending=(pending,),
    )
    surrogate = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.SquaredExponentialKernel(length_scale=0.5),
        noise_scale=0.01,
    )
    plan = phx.uq.GaussianProcessBayesianOptimization(
        8,
        objective_surrogate=surrogate,
        constraint_surrogates=(surrogate,),
        initial_evaluations=4,
        batch_size=2,
        candidate_tuple_count=16,
        fantasy_count=8,
    )
    first = phx.uq.bayesian_optimize(problem, plan, jr.key(7))
    second = phx.uq.bayesian_optimize(problem, plan, jr.key(7))
    assert first.pending_count == 1
    assert first.fantasy_keys.shape[0] == 2
    assert first.evaluation_count == 8
    assert first.constraints.shape == (8, 1)
    assert jnp.array_equal(first.evaluated_encoded, second.evaluated_encoded)
    assert first.globally_optimal is False
    assert jnp.all(jnp.isfinite(first.acquisition_standard_errors))


def test_no_pending_bo_starts_and_reports_finite_acquisition_error():
    domain = phx.uq.BayesianOptimizationDomain(
        jnp.asarray([0.5]),
        lower_bounds=jnp.asarray([0.0]),
        upper_bounds=jnp.asarray([1.0]),
    )
    problem = phx.uq.BayesianOptimizationProblem(
        lambda point: (point.continuous[0] - 0.3) ** 2,
        domain,
    )
    surrogate = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.SquaredExponentialKernel(length_scale=0.4),
        noise_scale=0.01,
    )
    plan = phx.uq.GaussianProcessBayesianOptimization(
        4,
        objective_surrogate=surrogate,
        initial_evaluations=2,
        candidate_tuple_count=8,
        fantasy_count=4,
    )

    result = phx.uq.bayesian_optimize(problem, plan, jr.key(11))

    assert result.pending_count == 0
    assert result.evaluation_count == 4
    assert jnp.all(jnp.isfinite(result.acquisition_standard_errors))


def test_bo_requires_two_fantasies_for_sample_standard_error():
    surrogate = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.SquaredExponentialKernel(length_scale=0.4),
        noise_scale=0.01,
    )

    with pytest.raises(ValueError, match="fantasy_count must be at least 2"):
        phx.uq.GaussianProcessBayesianOptimization(
            2,
            objective_surrogate=surrogate,
            initial_evaluations=1,
            fantasy_count=1,
        )


def test_pending_point_is_excluded_from_initial_and_space_filling_proposals():
    categorical = phx.optim.FiniteProductSpace(
        {"choice": phx.optim.FiniteAxis(jnp.arange(4.0))}
    )
    domain = phx.uq.BayesianOptimizationDomain(categorical=categorical)
    pending = domain.decode(jnp.empty((0,)), jnp.asarray(3))
    problem = phx.uq.BayesianOptimizationProblem(
        lambda point: point.categorical["choice"],
        domain,
        pending=(pending,),
    )
    surrogate = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.SquaredExponentialKernel(length_scale=1.0),
        noise_scale=0.01,
    )
    plan = phx.uq.GaussianProcessBayesianOptimization(
        2,
        objective_surrogate=surrogate,
        initial_evaluations=1,
        candidate_tuple_count=64,
        fantasy_count=4,
        minimum_separation=0.1,
    )
    result = phx.uq.bayesian_optimize(problem, plan, jr.key(19))
    distances = jnp.sqrt(
        jnp.sum(jnp.square(result.evaluated_encoded - pending.encoded), axis=1)
    )
    assert jnp.all(distances >= plan.minimum_separation)
    assert result.proposal_kinds[-1] == phx.uq.BAYESIAN_OPTIMIZATION_SPACE_FILLING
