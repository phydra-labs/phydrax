# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from itertools import combinations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.optim._pareto import dominance_matrix, hypervolume, nondominated_mask
from phydrax.uq._multiobjective_bayesian_optimization import (
    _acquisition_scores,
    _PreparedGP,
    _psd_factors,
    _sample_baseline,
    _sample_hvi,
    GaussianProcessMultiObjectiveBayesianOptimization,
    multiobjective_bayesian_optimize,
    MultiObjectiveBayesianOptimizationProblem,
)


def _domain():
    return phx.uq.BayesianOptimizationDomain(
        jnp.asarray([0.5]),
        lower_bounds=jnp.asarray([0.0]),
        upper_bounds=jnp.asarray([1.0]),
    )


def _state(names=("cost", "loss"), *, noise=0.2):
    count = len(names)
    coregionalization = phx.uq.Coregionalization(
        jnp.ones((count, 1)) * 0.8,
        jnp.ones((count,)) * 0.4,
        output_names=names,
    )
    return phx.uq.MultiOutputGaussianProcessLikelihoodState(
        kernel=phx.uq.IntrinsicCoregionalizationKernel(
            phx.kernels.SquaredExponentialKernel(length_scale=0.25), coregionalization
        ),
        noise_scale=jnp.full((count,), noise),
        jitter=1e-6,
    )


def _problem(*, pending=(), objective=None, validity=None, constraints=()):
    return MultiObjectiveBayesianOptimizationProblem(
        (
            lambda point, key: jnp.stack(
                (point.continuous[0] ** 2, (1 - point.continuous[0]) ** 2)
            )
        )
        if objective is None
        else objective,
        _domain(),
        objective_names=("cost", "loss"),
        directions=("min", "min"),
        scales=jnp.ones((2,)),
        reference=jnp.asarray([3.0, 3.0]),
        pending=pending,
        validity=validity,
        constraints=constraints,
    )


def _plan(**kwargs):
    options = dict(
        objective_surrogate=_state(),
        initial_evaluations=2,
        batch_size=2,
        candidate_tuple_count=8,
        fantasy_count=16,
    )
    options.update(kwargs)
    return GaussianProcessMultiObjectiveBayesianOptimization(5, **options)


def test_exact_hypervolume_and_tied_finite_minimization_dominance():
    front = jnp.asarray([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
    reference = jnp.asarray([5.0, 5.0])
    assert float(hypervolume(front, reference)) == 11.0
    augmented = jnp.concatenate((front, jnp.asarray([[1.0, 1.0]])))
    assert float(hypervolume(augmented, reference) - hypervolume(front, reference)) == 5.0
    rows = jnp.concatenate(
        (front, front[:1], jnp.asarray([[6.0, 0.0], [jnp.nan, 0.0], [3.0, 3.0]]))
    )
    assert float(hypervolume(rows, reference)) == 11.0
    assert jnp.array_equal(
        nondominated_mask(rows), jnp.asarray([True, True, True, True, True, False, False])
    )
    assert not bool(dominance_matrix(rows)[0, 3])
    assert bool(dominance_matrix(rows)[1, 6])
    assert float(hypervolume(jnp.empty((0, 2)), reference)) == 0.0


def test_exact_3d_hypervolume_matches_box_inclusion_exclusion():
    points = np.asarray(
        [[1.0, 4.0, 1.0], [2.0, 2.0, 2.0], [4.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    )
    reference = np.asarray([5.0, 5.0, 4.0])
    expected = 0.0
    for count in range(1, len(points) + 1):
        for subset in combinations(points, count):
            intersection = np.maximum(reference - np.max(subset, axis=0), 0.0).prod()
            expected += (-1) ** (count + 1) * intersection
    assert float(hypervolume(points, reference)) == expected
    assert float(jax.jit(hypervolume)(points, reference)) == expected


def test_qhvi_filters_each_member_instead_of_rejecting_a_partly_feasible_batch():
    baseline = jnp.asarray([[[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]]])
    candidates = jnp.asarray([[[1.0, 1.0], [0.0, 0.0]]])
    gains = _sample_hvi(
        baseline,
        candidates,
        jnp.ones((1, 3), dtype=bool),
        jnp.asarray([[True, False]]),
        jnp.asarray([5.0, 5.0]),
    )
    assert float(gains[0]) == 5.0
    # An infeasible historical member must not suppress the other baseline boxes.
    gains = _sample_hvi(
        baseline,
        candidates,
        jnp.asarray([[False, True, True]]),
        jnp.asarray([[True, False]]),
        jnp.asarray([5.0, 5.0]),
    )
    assert float(gains[0]) == 6.0


def test_pending_is_in_the_sampled_attained_set_not_only_a_conditioning_location():
    point = _domain().decode(jnp.asarray([0.9]))
    plan = _plan(fantasy_count=64)
    encoded = jnp.asarray([[0.1]])
    values = jnp.asarray([[1.0, 1.0]])
    candidates = jnp.asarray([[[0.9]]])
    arguments = (
        plan,
        encoded,
        values,
        jnp.empty((1, 0)),
        jnp.asarray([True]),
        candidates,
        jr.key(19),
    )
    with_pending, _ = _acquisition_scores(_problem(pending=(point,)), *arguments)
    without_pending, _ = _acquisition_scores(_problem(), *arguments)
    assert float(with_pending[0]) < 1e-4
    assert float(without_pending[0]) > 0.1


def test_historical_baseline_is_joint_latent_not_the_noisy_observed_front():
    plan = _plan(objective_surrogate=_state(noise=1.0), fantasy_count=64)
    # A candidate already in B cannot improve B, despite high observation noise.
    # Plugging the noisy observed vector into the baseline gives spurious gain.
    estimates, errors = _acquisition_scores(
        _problem(),
        plan,
        jnp.asarray([[0.2]]),
        jnp.asarray([[0.0, 0.0]]),
        jnp.empty((1, 0)),
        jnp.asarray([True]),
        jnp.asarray([[[0.2]]]),
        jr.key(23),
    )
    assert float(estimates[0]) < 1e-4
    assert float(errors[0]) < 1e-4


def test_correlated_latent_draws_preserve_output_covariance_and_exclude_observation_noise():
    state = _state(noise=2.0)
    plan = _plan(objective_surrogate=state, fantasy_count=8192)
    gp = _PreparedGP(
        jnp.asarray([[0.0]]), jnp.asarray([[0.0, 0.0]]), state, plan.max_working_bytes
    )
    baseline = _sample_baseline(gp, jnp.asarray([[1.0]]), jr.key(31), plan)
    covariance = np.cov(np.asarray(baseline.draws).T)
    prior = np.asarray(state.kernel.coregionalization.covariance)
    np.testing.assert_allclose(covariance, prior, atol=0.05, rtol=0.05)
    assert covariance[0, 1] > 0.5
    assert covariance[0, 0] < 1.0  # Adding observation variance would exceed four.
    factor, inverse_root = _psd_factors(jnp.asarray([[1.0, 1.0], [1.0, 1.0]]), 1e-6)
    np.testing.assert_allclose(factor @ factor.T, [[1.0, 1.0], [1.0, 1.0]], atol=1e-6)
    assert bool(jnp.all(jnp.isfinite(inverse_root)))
    tiny = 1.0e-12 * jnp.asarray([[1.0, 0.5], [0.5, 1.0]])
    tiny_factor, _ = _psd_factors(tiny, 1e-6)
    np.testing.assert_allclose(
        tiny_factor @ tiny_factor.T,
        tiny,
        atol=1e-20,
        rtol=1e-8,
    )
    with pytest.raises((RuntimeError, ValueError), match="positive semidefinite"):
        _psd_factors(jnp.asarray([[1.0, 2.0], [2.0, 1.0]]), 1e-6)


def test_seeded_mixed_noisy_constrained_q_batches_replay_and_keep_vector_front():
    categorical = phx.optim.FiniteProductSpace(
        {"material": phx.optim.FiniteAxis(jnp.asarray([0.0, 1.0]))}
    )
    domain = phx.uq.BayesianOptimizationDomain(
        jnp.asarray([0.5]),
        lower_bounds=jnp.asarray([0.0]),
        upper_bounds=jnp.asarray([1.0]),
        categorical=categorical,
    )

    def objective(point, key):
        x, material = point.continuous[0], point.categorical["material"]
        return jnp.stack(
            ((x - 0.15) ** 2 + 0.1 * material, (x - 0.85) ** 2 - 0.05 * material)
        ) + 0.02 * jr.normal(key, (2,))

    pending = domain.decode(jnp.asarray([0.7]), jnp.asarray(1))
    problem = MultiObjectiveBayesianOptimizationProblem(
        objective,
        domain,
        objective_names=("cost", "loss"),
        directions=("min", "min"),
        scales=jnp.ones((2,)),
        reference=jnp.asarray([2.0, 2.0]),
        pending=(pending,),
        constraints=(
            lambda point, key: point.continuous[0] - 0.8 + 0.01 * jr.normal(key),
        ),
    )
    constraint_state = phx.uq.GaussianProcessLikelihoodState(
        noise_scale=0.01, jitter=1e-6
    )
    plan = _plan(
        objective_surrogate=_state(noise=0.02), constraint_surrogates=(constraint_state,)
    )
    first = multiobjective_bayesian_optimize(problem, plan, jr.key(7))
    replay = multiobjective_bayesian_optimize(problem, plan, jr.key(7))
    assert first.evaluation_count == 5
    assert first.acquisition_batch_sizes.tolist() == [2, 1]
    np.testing.assert_array_equal(first.evaluated_encoded, replay.evaluated_encoded)
    np.testing.assert_array_equal(first.objectives, replay.objectives)
    np.testing.assert_array_equal(
        jr.key_data(first.evaluation_keys), jr.key_data(replay.evaluation_keys)
    )
    assert first.work_id == replay.work_id
    assert bool(jnp.all(jnp.isfinite(first.acquisition_standard_errors)))
    assert not bool(jnp.any(jnp.all(first.evaluated_encoded == pending.encoded, axis=1)))
    expected = np.asarray(first.feasible).copy()
    values = np.asarray(first.objectives)
    for index in range(len(values)):
        if expected[index]:
            expected[index] = not any(
                bool(first.feasible[other])
                and np.all(values[other] <= values[index])
                and np.any(values[other] < values[index])
                for other in range(len(values))
            )
    np.testing.assert_array_equal(first.observed_pareto_mask, expected)
    assert first.globally_optimal is False


def test_invalid_physics_is_guarded_and_excluded_from_gp_training():
    calls = []

    def objective(point, key):
        calls.append(float(point.continuous[0]))
        return jnp.asarray([point.continuous[0], 1 - point.continuous[0]])

    problem = _problem(
        objective=objective, validity=lambda point: point.continuous[0] != 0.5
    )
    result = multiobjective_bayesian_optimize(problem, _plan(), jr.key(41))
    assert 0.5 not in calls
    assert result.invalid_evaluation_count == 1
    assert bool(jnp.all(jnp.isnan(result.objectives[0])))
    assert not bool(result.observed_pareto_mask[0])
    assert bool(jnp.all(jnp.isfinite(result.acquisition_estimates)))
    assert len(calls) == result.evaluation_count - 1


@pytest.mark.parametrize(
    "limits",
    [
        {"max_training_points": 4},
        {"max_pending_points": 1},
        {"max_baseline_points": 5},
        {"max_hypervolume_points": 5},
        {"max_working_bytes": 1},
        {"max_hypervolume_work": 1},
    ],
)
def test_all_resource_limits_fail_before_physical_evaluation(limits):
    calls = []

    def objective(point, key):
        calls.append(point)
        return jnp.zeros((2,))

    problem = _problem(
        objective=objective,
        pending=(
            _domain().decode(jnp.asarray([0.1])),
            _domain().decode(jnp.asarray([0.9])),
        ),
    )
    with pytest.raises(ValueError, match="capacity|bytes|max_hypervolume_work"):
        multiobjective_bayesian_optimize(problem, _plan(**limits), jr.key(5))
    assert calls == []


def test_unsupported_geometry_and_hypervolume_capacity_are_explicit():
    with pytest.raises(ValueError, match="two or three"):
        hypervolume(jnp.zeros((2, 4)), jnp.ones((4,)))
    with pytest.raises(ValueError, match="capacity"):
        hypervolume(jnp.zeros((3, 2)), jnp.ones((2,)), max_points=2)
    with pytest.raises(ValueError, match="strictly positive"):
        MultiObjectiveBayesianOptimizationProblem(
            lambda point, key: jnp.zeros((2,)),
            _domain(),
            objective_names=("a", "b"),
            directions=("min", "max"),
            scales=[1.0, 0.0],
            reference=[2.0, -2.0],
        )


def test_three_objectives_respect_maximization_and_physical_scales():
    names = ("mass", "efficiency", "cost")
    problem = MultiObjectiveBayesianOptimizationProblem(
        lambda point, key: jnp.stack(
            (point.continuous[0], 2 * point.continuous[0], 3 * point.continuous[0])
        ),
        _domain(),
        objective_names=names,
        directions=("min", "max", "min"),
        scales=[1.0, 2.0, 3.0],
        reference=[2.0, -2.0, 6.0],
    )
    plan = GaussianProcessMultiObjectiveBayesianOptimization(
        3,
        objective_surrogate=_state(names),
        initial_evaluations=2,
        candidate_tuple_count=3,
        fantasy_count=4,
    )
    result = multiobjective_bayesian_optimize(problem, plan, jr.key(37))
    # All distinct points trade increasing mass/cost against increasing efficiency.
    assert bool(jnp.all(result.observed_pareto_mask))
    assert float(result.observed_hypervolume) > 0
    np.testing.assert_allclose(
        problem.canonical(result.objectives),
        result.objectives * np.asarray([1.0, -0.5, 1 / 3]),
    )
