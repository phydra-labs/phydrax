#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control.stochastic._evaluation import (
    ControlledTransitionProblem,
    PreparedControlledNoise,
    rollout_feedback,
)
from phydrax.control.stochastic._fitted_bellman import (
    bridge_fitted_bellman_to_bsde,
    evaluate_fitted_bellman,
    fit_frozen_policy_bellman,
    FittedBellmanPlan,
    FittedBellmanProblem,
    FittedBellmanStatus,
    FROZEN_POLICY_FITTED_BELLMAN,
    prepare_fitted_bellman,
)
from phydrax.dynamics import TimeGrid


def _controlled_problem(
    *,
    num_steps=2,
    policy_scale=0.0,
    terminal_cost=None,
    problem_id="fitted:linear",
):
    grid = TimeGrid(jnp.arange(num_steps + 1, dtype=float), time_id=f"{problem_id}:time")

    def transition(context, state, action, noise, args):
        del context, args
        return state + action + noise

    def stage_cost(context, state, action, args):
        del context, args
        return state[0] + action[0]

    def default_terminal(time, state, args):
        del time, args
        return 2.0 * state[0]

    problem = ControlledTransitionProblem(
        transition,
        grid,
        jnp.asarray([1.0]),
        state_shape=(1,),
        action_shape=(1,),
        noise_shape=(1,),
        stage_cost=stage_cost,
        terminal_cost=default_terminal if terminal_cost is None else terminal_cost,
        problem_id=problem_id,
    )

    def policy(context, state, args):
        del context, args
        return policy_scale * state

    return problem, policy


def _paths(
    problem,
    policy,
    increments,
    *,
    role,
    validity=None,
    policy_id="frozen:linear",
    realization_ids=None,
):
    values = jnp.asarray(increments, dtype=float)
    count = values.shape[0]
    prepared = PreparedControlledNoise(
        values,
        valid=(
            jnp.ones((count,), dtype=bool)
            if validity is None
            else jnp.asarray(validity, dtype=bool)
        ),
        realization_ids=(
            tuple(f"{role}:realization:{index}" for index in range(count))
            if realization_ids is None
            else realization_ids
        ),
        coupling_id=f"{role}:coupling",
        independence_labels=jnp.arange(count, dtype=jnp.int32),
        noise_shape=(1,),
    )
    return rollout_feedback(problem, policy, prepared, policy_id=policy_id)


def _problem_and_plan(
    *,
    train_noise=None,
    holdout_noise=None,
    feature_map=None,
    num_features=1,
    ridge=0.0,
    training_validity=None,
    holdout_validity=None,
    policy_scale=0.0,
    controlled_problem=None,
    policy=None,
    training_weights=None,
    holdout_weights=None,
):
    if controlled_problem is None or policy is None:
        controlled_problem, policy = _controlled_problem(policy_scale=policy_scale)
    steps = controlled_problem.time_grid.num_steps
    if train_noise is None:
        train_noise = jnp.zeros((4, steps, 1))
    if holdout_noise is None:
        holdout_noise = jnp.zeros((3, steps, 1))
    training = _paths(
        controlled_problem,
        policy,
        train_noise,
        role="training",
        validity=training_validity,
    )
    holdout = _paths(
        controlled_problem,
        policy,
        holdout_noise,
        role="holdout",
        validity=holdout_validity,
    )
    features = (lambda time, state, args: state) if feature_map is None else feature_map
    fitted_problem = FittedBellmanProblem(
        training,
        holdout,
        features,
        num_features=num_features,
        feature_id="features:fixed",
        problem_id="fitted:linear:evaluation",
        training_weights=training_weights,
        holdout_weights=holdout_weights,
    )
    plan = FittedBellmanPlan(
        ridge=ridge,
        plan_id=f"fitted:plan:ridge:{ridge}",
    )
    return controlled_problem, policy, fitted_problem, plan


def test_exact_linear_feature_value_recovery_and_deterministic_reduction():
    _, _, problem, plan = _problem_and_plan()

    result = fit_frozen_policy_bellman(problem, plan)

    assert bool(result.valid)
    assert int(result.status) == FittedBellmanStatus.SUCCESS
    assert result.result_label == FROZEN_POLICY_FITTED_BELLMAN
    np.testing.assert_allclose(result.coefficients[:, 0], [4.0, 3.0, 2.0])
    np.testing.assert_allclose(
        result.training_value_predictions,
        jnp.broadcast_to(jnp.asarray([4.0, 3.0, 2.0]), (4, 3)),
    )
    np.testing.assert_allclose(result.training_regression_residuals, 0.0, atol=1e-6)
    np.testing.assert_allclose(result.holdout_bellman_residuals, 0.0, atol=1e-6)
    np.testing.assert_allclose(
        result.training_value_predictions[:, 0], problem.training_paths.returns
    )
    np.testing.assert_array_equal(result.design_ranks, [1, 1, 1])
    assert result.frozen_policy_evaluation
    assert not result.policy_improvement_performed
    assert not result.optimality_claimed


def test_backward_regression_uses_explicit_training_weights():
    controlled, policy = _controlled_problem(num_steps=1)
    _, _, problem, plan = _problem_and_plan(
        controlled_problem=controlled,
        policy=policy,
        train_noise=jnp.asarray([[[-1.0]], [[1.0]]]),
        holdout_noise=jnp.asarray([[[0.0]], [[2.0]]]),
        feature_map=lambda time, state, args: jnp.ones((1,)),
        training_weights=jnp.asarray([1.0, 3.0]),
        holdout_weights=jnp.asarray([2.0, 1.0]),
    )

    result = fit_frozen_policy_bellman(problem, plan)

    np.testing.assert_allclose(result.coefficients[-1, 0], 3.0, atol=1e-6)
    np.testing.assert_allclose(
        result.training_regression_residuals[:, -1], [-3.0, 1.0], atol=1e-6
    )
    np.testing.assert_allclose(
        jnp.average(
            result.training_regression_residuals[:, -1],
            weights=problem.training_weights,
        ),
        0.0,
        atol=1e-6,
    )


def test_rank_deficiency_without_ridge_is_reported_without_pseudoinverse():
    feature_map = lambda time, state, args: jnp.asarray([1.0, state[0]])
    _, _, problem, plan = _problem_and_plan(
        feature_map=feature_map,
        num_features=2,
        ridge=0.0,
    )

    result = fit_frozen_policy_bellman(problem, plan)

    assert not bool(result.valid)
    assert int(result.status) == FittedBellmanStatus.RANK_DEFICIENT
    assert int(result.stage_status[-1]) == FittedBellmanStatus.RANK_DEFICIENT
    assert int(result.design_ranks[-1]) == 1
    assert np.isinf(result.design_condition_numbers[-1])
    assert np.isnan(result.coefficients[-1]).all()
    assert int(result.stage_status[0]) == FittedBellmanStatus.DEPENDENCY_FAILED


def test_ridge_regularizes_solve_but_does_not_hide_original_normal_residual():
    _, _, problem, plan = _problem_and_plan(ridge=0.5)

    result = fit_frozen_policy_bellman(problem, plan)

    assert bool(result.valid)
    assert np.all(np.asarray(result.ridge_normal_equation_residuals) < 1e-5)
    assert np.all(np.asarray(result.original_normal_equation_residuals) > 0.0)
    np.testing.assert_allclose(
        result.original_normal_equation_residuals,
        plan.ridge * jnp.abs(result.coefficients[:, 0]),
        rtol=2e-5,
        atol=2e-5,
    )
    assert np.all(np.asarray(result.training_weighted_rmse) > 0.0)


def test_training_fit_and_holdout_bellman_identity_are_separate():
    train_noise = jnp.zeros((4, 2, 1))
    holdout_noise = jnp.asarray(
        [
            [[1.0], [0.0]],
            [[-0.5], [0.0]],
            [[2.0], [0.0]],
        ]
    )
    _, _, problem, plan = _problem_and_plan(
        train_noise=train_noise,
        holdout_noise=holdout_noise,
    )

    result = fit_frozen_policy_bellman(problem, plan)

    assert result.training_realization_ids == problem.training_paths.realization_ids
    assert result.holdout_realization_ids == problem.holdout_paths.realization_ids
    assert set(result.training_realization_ids).isdisjoint(result.holdout_realization_ids)
    assert result.training_coupling_id != result.holdout_coupling_id
    np.testing.assert_allclose(result.training_regression_residuals, 0.0, atol=1e-6)
    assert np.max(np.abs(np.asarray(result.holdout_bellman_residuals[:, 0]))) > 0.5
    np.testing.assert_allclose(
        result.holdout_bellman_residuals,
        result.holdout_targets - result.holdout_value_predictions,
        atol=1e-6,
    )


def test_invalid_paths_are_excluded_case_locally_from_both_roles():
    train_noise = jnp.asarray(
        [
            [[0.0], [0.0]],
            [[jnp.nan], [0.0]],
            [[0.0], [0.0]],
        ]
    )
    holdout_noise = jnp.zeros((3, 2, 1))
    _, _, problem, plan = _problem_and_plan(
        train_noise=train_noise,
        holdout_noise=holdout_noise,
        training_validity=[True, True, False],
        holdout_validity=[True, False, True],
    )

    result = fit_frozen_policy_bellman(problem, plan)

    np.testing.assert_array_equal(result.training_path_valid, [True, False, False])
    np.testing.assert_array_equal(result.holdout_path_valid, [True, False, True])
    np.testing.assert_array_equal(result.sample_counts, [1, 1, 1])
    assert np.isnan(result.training_regression_residuals[1:]).all()
    assert np.isnan(result.holdout_bellman_residuals[1]).all()
    assert bool(result.valid)


def test_problem_rejects_reused_or_overlapping_holdout_identity():
    controlled, policy = _controlled_problem()
    training = _paths(controlled, policy, jnp.zeros((2, 2, 1)), role="training")

    with pytest.raises(ValueError, match="separate"):
        FittedBellmanProblem(
            training,
            training,
            lambda time, state, args: state,
            num_features=1,
            feature_id="feature",
        )

    overlapping = _paths(
        controlled,
        policy,
        jnp.zeros((2, 2, 1)),
        role="holdout",
        realization_ids=training.realization_ids,
    )
    with pytest.raises(ValueError, match="disjoint"):
        FittedBellmanProblem(
            training,
            overlapping,
            lambda time, state, args: state,
            num_features=1,
            feature_id="feature",
        )


def test_terminal_value_is_regressed_on_terminal_features():
    def affine_terminal(time, state, args):
        del time, args
        return 3.0 + 2.0 * state[0]

    controlled, policy = _controlled_problem(
        num_steps=1,
        terminal_cost=affine_terminal,
        problem_id="fitted:terminal",
    )
    feature_map = lambda time, state, args: jnp.asarray([1.0, state[0]])
    _, _, problem, plan = _problem_and_plan(
        controlled_problem=controlled,
        policy=policy,
        train_noise=jnp.asarray([[[-1.0]], [[0.0]], [[1.0]], [[2.0]]]),
        holdout_noise=jnp.asarray([[[-0.5]], [[0.5]], [[1.5]]]),
        feature_map=feature_map,
        num_features=2,
    )

    result = fit_frozen_policy_bellman(problem, plan)

    np.testing.assert_allclose(result.coefficients[-1], [3.0, 2.0], atol=2e-5)
    np.testing.assert_allclose(
        result.training_regression_residuals[:, -1], 0.0, atol=2e-5
    )
    assert int(result.design_ranks[-1]) == 2
    assert int(result.stage_status[0]) == FittedBellmanStatus.RANK_DEFICIENT


def test_prepared_evaluation_is_filter_jittable_and_prediction_vmaps_over_cases():
    _, _, problem, plan = _problem_and_plan()
    prepared = prepare_fitted_bellman(problem, plan)

    eager = evaluate_fitted_bellman(prepared)
    compiled = eqx.filter_jit(evaluate_fitted_bellman)(prepared)
    times = jnp.asarray([0.0, 1.0, 2.0])
    states = jnp.asarray([[1.0], [2.0], [3.0]])
    case_values = jax.vmap(eager.predict)(times, states)

    np.testing.assert_allclose(compiled.coefficients, eager.coefficients, atol=1e-6)
    np.testing.assert_allclose(case_values, [4.0, 6.0, 6.0], atol=1e-6)


def test_bsde_bridge_preserves_ids_and_never_conflates_action_with_z():
    controlled, policy = _controlled_problem(policy_scale=0.25)
    _, _, problem, plan = _problem_and_plan(
        controlled_problem=controlled,
        policy=policy,
        policy_scale=0.25,
    )
    result = fit_frozen_policy_bellman(problem, plan)

    def controlled_drift(time, state, action, args):
        del time, state, args
        return action

    def controlled_diffusion(time, state, action, args):
        del time, state, action, args
        return jnp.ones((1, 1))

    def z_predictor(time, state):
        del time, state
        return jnp.asarray([[7.0]])

    bridge = bridge_fitted_bellman_to_bsde(
        result,
        controlled,
        policy,
        controlled_drift,
        controlled_diffusion,
        z_predictor=z_predictor,
    )

    source = problem.holdout_paths
    assert bridge.realization_ids == source.realization_ids
    assert bridge.coupling_id == source.coupling_id
    assert bridge.path_id == source.coupling_id
    assert bridge.process_id == controlled.problem_id
    assert bridge.time_id == controlled.time_grid.time_id
    assert bridge.policy_id == source.policy_id
    assert bridge.sample_role == "holdout"
    assert bridge.bsde_problem.process_id == controlled.problem_id
    assert bridge.bsde_paths.path_id == source.coupling_id
    assert bridge.bsde_paths.metadata["realization_ids"] == source.realization_ids
    assert bridge.feynman_kac_labels.metadata["realization_ids"] == source.realization_ids
    np.testing.assert_array_equal(
        bridge.feynman_kac_labels.cluster_ids,
        jnp.repeat(source.independence_labels, source.time_grid.num_times),
    )
    np.testing.assert_allclose(bridge.physical_actions, source.actions)
    np.testing.assert_allclose(bridge.martingale_integrands, 7.0)
    assert bridge.physical_actions.shape[-1:] == bridge.action_shape
    assert bridge.martingale_integrands.shape[-2:] == bridge.z_shape
    assert not bridge.action_is_martingale_integrand
    assert not bridge.policy_improvement_performed
    np.testing.assert_allclose(
        bridge.evaluation.local_residuals[..., 0],
        result.holdout_bellman_residuals[:, :-1],
        atol=2e-5,
    )
    expected_path_values = jnp.asarray([5.9375, 4.6875, 3.125])
    np.testing.assert_allclose(
        bridge.feynman_kac_labels.value_targets[:3, 0],
        expected_path_values,
        atol=2e-5,
    )
