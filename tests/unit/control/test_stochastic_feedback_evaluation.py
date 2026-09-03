#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.control.stochastic import (
    compare_feedback_policies,
    ControlledTransitionProblem,
    evaluate_feedback_policy,
    FeedbackPolicyEvaluationStatus,
    PreparedControlledNoise,
    rollout_feedback,
)
from phydrax.dynamics import TimeGrid
from phydrax.optim import CVaRRisk, EntropicRisk, ExpectationRisk, MeanVarianceRisk
from phydrax.stochastic import WienerRealization


def _noise(
    increments,
    *,
    validity=None,
    realization_ids=None,
    coupling_id="coupling:shared",
    independence_labels=None,
):
    values = jnp.asarray(increments, dtype=float)
    path_count = values.shape[0]
    return PreparedControlledNoise(
        values,
        valid=(jnp.ones((path_count,), dtype=bool) if validity is None else validity),
        realization_ids=(
            tuple(f"realization:{index}" for index in range(path_count))
            if realization_ids is None
            else realization_ids
        ),
        coupling_id=coupling_id,
        independence_labels=(
            jnp.arange(path_count, dtype=jnp.int32)
            if independence_labels is None
            else independence_labels
        ),
        noise_shape=(1,),
    )


def _problem(*, num_steps=2, stage_cost=None, terminal_cost=None):
    grid = TimeGrid(
        jnp.arange(num_steps + 1, dtype=float), time_id=f"feedback:{num_steps}:time"
    )

    def transition(context, state, action, noise, args):
        del context, args
        return state + action + noise

    def zero_stage(context, state, action, args):
        del context, state, action, args
        return jnp.asarray(0.0)

    def final_state_cost(time, state, args):
        del time, args
        return state[0]

    return ControlledTransitionProblem(
        transition,
        grid,
        jnp.asarray([1.0]),
        state_shape=(1,),
        action_shape=(1,),
        noise_shape=(1,),
        stage_cost=zero_stage if stage_cost is None else stage_cost,
        terminal_cost=final_state_cost if terminal_cost is None else terminal_cost,
        problem_id=f"feedback:{num_steps}",
    )


def test_exact_deterministic_feedback_rollout():
    def stage_cost(context, state, action, args):
        del context, args
        return state[0] + action[0]

    problem = _problem(num_steps=2, stage_cost=stage_cost)
    prepared = _noise(jnp.zeros((1, 2, 1)))

    paths = rollout_feedback(
        problem,
        lambda context, state, args: -0.5 * state,
        prepared,
        policy_id="half-state",
    )

    np.testing.assert_allclose(paths.states[0, :, 0], [1.0, 0.5, 0.25])
    np.testing.assert_allclose(paths.actions[0, :, 0], [-0.5, -0.25])
    np.testing.assert_allclose(paths.stage_costs, [[0.5, 0.25]])
    np.testing.assert_allclose(paths.terminal_costs, [0.25])
    np.testing.assert_allclose(paths.returns, [1.0])
    assert bool(paths.valid[0])
    assert int(paths.status[0]) == FeedbackPolicyEvaluationStatus.SUCCESS


def test_policy_cannot_observe_current_noise():
    problem = _problem(num_steps=1)
    prepared = _noise(jnp.asarray([[[2.0]], [[-3.0]]]))

    def state_feedback(context, state, args):
        del context, args
        return 2.0 * state

    paths = rollout_feedback(problem, state_feedback, prepared, policy_id="state-only")

    np.testing.assert_allclose(paths.actions[:, 0], [[2.0], [2.0]])
    np.testing.assert_allclose(paths.states[:, 1], [[5.0], [0.0]])


def test_realization_replay_ids_and_antithetic_cluster_labels():
    grid = TimeGrid(jnp.asarray([0.0, 0.5, 1.0]), time_id="wiener-grid")
    realization = WienerRealization.antithetic(
        jr.key(4),
        (1,),
        support=(0.0, 1.0),
        num_pairs=2,
        coupling_id="wiener:paired",
    )
    increments = realization.increments(grid.times[:-1], grid.times[1:])
    prepared = PreparedControlledNoise.from_realization(
        increments,
        realization,
        valid=jnp.ones((4,), dtype=bool),
        noise_shape=(1,),
    )
    problem = _problem(num_steps=2)
    paths = rollout_feedback(
        problem,
        lambda context, state, args: jnp.zeros((1,)),
        prepared,
        policy_id="zero",
    )
    evaluation = evaluate_feedback_policy(
        problem,
        lambda context, state, args: jnp.zeros((1,)),
        prepared,
        policy_id="zero",
        method="asymptotic-normal",
    )

    assert paths.realization_ids == prepared.realization_ids
    assert paths.coupling_id == realization.coupling_id
    np.testing.assert_array_equal(paths.independence_labels, [0, 0, 1, 1])
    assert int(evaluation.evidence.valid_path_count) == 4
    assert int(evaluation.evidence.independent_cluster_count) == 2


def test_common_random_number_comparison_retains_pairing():
    problem = _problem(num_steps=1)
    prepared = _noise(jnp.asarray([[[0.5]], [[-0.25]], [[1.0]]]))
    left = rollout_feedback(
        problem,
        lambda context, state, args: jnp.zeros((1,)),
        prepared,
        policy_id="left",
    )
    right = rollout_feedback(
        problem,
        lambda context, state, args: jnp.ones((1,)),
        prepared,
        policy_id="right",
    )

    comparison = compare_feedback_policies(left, right)

    np.testing.assert_allclose(comparison.paired_differences, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(comparison.mean_difference, 1.0)
    assert comparison.realization_ids == prepared.realization_ids
    assert comparison.coupling_id == prepared.coupling_id
    assert comparison.left_policy_id == "left"
    assert comparison.right_policy_id == "right"


def test_comparison_rejects_mismatched_coupling():
    problem = _problem(num_steps=1)
    increments = jnp.asarray([[[0.5]], [[-0.25]]])
    left_noise = _noise(increments, coupling_id="coupling:left")
    right_noise = _noise(increments, coupling_id="coupling:right")
    policy = lambda context, state, args: jnp.zeros((1,))
    left = rollout_feedback(problem, policy, left_noise, policy_id="left")
    right = rollout_feedback(problem, policy, right_noise, policy_id="right")

    with pytest.raises(ValueError, match="coupling IDs"):
        compare_feedback_policies(left, right)


def test_invalid_noise_paths_remain_case_local():
    problem = _problem(num_steps=1)
    prepared = _noise(
        jnp.asarray([[[0.0]], [[1.0]], [[jnp.nan]]]),
        validity=jnp.asarray([True, False, True]),
    )

    paths = rollout_feedback(
        problem,
        lambda context, state, args: jnp.zeros((1,)),
        prepared,
        policy_id="zero",
    )

    np.testing.assert_array_equal(paths.valid, [True, False, False])
    np.testing.assert_array_equal(
        paths.status,
        [
            FeedbackPolicyEvaluationStatus.SUCCESS,
            FeedbackPolicyEvaluationStatus.INVALID_NOISE_PATH,
            FeedbackPolicyEvaluationStatus.INVALID_NOISE_PATH,
        ],
    )
    evaluation = evaluate_feedback_policy(
        problem,
        lambda context, state, args: jnp.zeros((1,)),
        prepared,
        policy_id="zero",
    )
    assert int(evaluation.status) == FeedbackPolicyEvaluationStatus.PARTIAL_PATH_FAILURE
    assert evaluation.evidence.coverage == "none"


@pytest.mark.parametrize(
    "risk",
    [
        ExpectationRisk(),
        MeanVarianceRisk(0.25),
        CVaRRisk(0.5),
        EntropicRisk(0.2),
    ],
)
def test_empirical_risk_uses_existing_optim_risk_exactly(risk):
    problem = _problem(num_steps=1)
    prepared = _noise(jnp.asarray([[[0.0]], [[1.0]], [[3.0]]]))
    evaluation = evaluate_feedback_policy(
        problem,
        lambda context, state, args: jnp.zeros((1,)),
        prepared,
        policy_id="zero",
        risk=risk,
        method="none",
    )
    weights = jnp.full((3,), 1.0 / 3.0)

    np.testing.assert_allclose(
        evaluation.empirical_risk,
        risk.evaluate(evaluation.paths.returns, weights),
    )
    np.testing.assert_allclose(evaluation.paths.returns, [1.0, 2.0, 4.0])
    if not isinstance(risk, ExpectationRisk):
        assert evaluation.evidence.coverage == "none"


def test_hoeffding_requires_bounds_and_training_data_has_no_coverage_claim():
    problem = _problem(num_steps=1)
    prepared = _noise(jnp.asarray([[[0.0]], [[1.0]], [[-0.5]]]))
    policy = lambda context, state, args: jnp.zeros((1,))

    with pytest.raises(ValueError, match="requires declared finite return_bounds"):
        evaluate_feedback_policy(
            problem,
            policy,
            prepared,
            policy_id="zero",
            method="hoeffding",
        )

    holdout = evaluate_feedback_policy(
        problem,
        policy,
        prepared,
        policy_id="zero",
        method="hoeffding",
        return_bounds=(-1.0, 3.0),
    )
    training = evaluate_feedback_policy(
        problem,
        policy,
        prepared,
        policy_id="zero",
        method="hoeffding",
        return_bounds=(-1.0, 3.0),
        sample_role="training",
    )

    assert holdout.evidence.coverage == "hoeffding"
    assert bool(jnp.isfinite(holdout.evidence.lower))
    assert bool(jnp.isfinite(holdout.evidence.upper))
    assert training.evidence.coverage == "none"
    assert not training.evidence.has_coverage_claim
