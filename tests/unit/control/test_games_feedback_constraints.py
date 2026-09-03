#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.control._trajectory_optimization import BoundedPathConstraint
from phydrax.control.games._constraints import (
    GameConstraintBlock,
    GameConstraintScope,
    GameConstraintSite,
    OpenLoopGameConstraints,
)
from phydrax.control.games._feedback_constraints import (
    CONSTRAINED_FEEDBACK_QUASI_NASH_MODEL,
    ConstrainedFeedbackGameProblem,
    FeedbackQuasiNashPlan,
    FeedbackQuasiNashStatus,
    solve_feedback_quasi_nash_model,
)
from phydrax.control.games._layout import PlayerControlPartition
from phydrax.control.games._local_lq import suggest_local_affine_game_policy
from phydrax.control.games._nonlinear import (
    DeterministicFeedbackGameProblem,
    evaluate_game_policy,
    ILQGameScaling,
)


def _local_suggestion(
    partition,
    control_costs,
    control_linear,
    *,
    state_control_cross=None,
    suggestion_id="feedback-constraint-test",
):
    players = partition.num_players
    controls = partition.joint_control_size
    control_costs = jnp.asarray(control_costs, dtype=float)
    control_linear = jnp.asarray(control_linear, dtype=float)
    if state_control_cross is None:
        state_control_cross = jnp.zeros((players, 1, controls))
    state_control_cross = jnp.asarray(state_control_cross, dtype=float)

    input_layout = phx.dynamics.InputLayout((controls,), roles="control")

    def transition(context, state, control, args):
        del context, control, args
        return state

    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=input_layout,
        system_id=f"{suggestion_id}:dynamics",
    )
    time_grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0]), time_id=f"{suggestion_id}:time"
    )
    args = {
        "R": control_costs,
        "r": control_linear,
        "N": state_control_cross,
    }

    def stage_cost(player):
        def cost(context, state, control, callback_args):
            del context
            return (
                0.5 * control @ callback_args["R"][player] @ control
                + callback_args["r"][player] @ control
                + state @ callback_args["N"][player] @ control
            )

        return cost

    def terminal_cost(time, state, callback_args):
        del time, state, callback_args
        return 0.0

    game = DeterministicFeedbackGameProblem(
        phx.control.DiscreteControlDynamics(system),
        time_grid,
        jnp.zeros((1,)),
        partition,
        stage_costs=tuple(stage_cost(player) for player in range(players)),
        terminal_costs=(terminal_cost,) * players,
        args=args,
        problem_id=f"{suggestion_id}:game",
    )
    nominal = phx.dynamics.CallableInputPolicy(
        lambda context, state, callback_args: jnp.zeros((controls,)),
        input_layout=input_layout,
        policy_id=f"{suggestion_id}:nominal",
    )
    evaluation = evaluate_game_policy(game, nominal)
    scaling = ILQGameScaling(
        jnp.ones((1,)),
        jnp.ones((controls,)),
        jnp.ones((players,)),
        scaling_id=f"{suggestion_id}:scaling",
    )
    suggestion = suggest_local_affine_game_policy(
        game,
        evaluation,
        scaling,
        symmetry_tolerance=2.0e-5,
        curvature_tolerance=1.0e-6,
        rank_relative_tolerance=1.0e-6,
        rank_absolute_tolerance=1.0e-7,
        maximum_condition=1.0e7,
        suggestion_id=suggestion_id,
    )
    assert bool(suggestion.successful)
    return suggestion


def _block(
    constraint_id,
    *,
    owner,
    participants,
    scope=GameConstraintScope.PLAYER_LOCAL,
    equality=False,
    control_dependencies=None,
):
    if control_dependencies is None:
        control_dependencies = participants
    return GameConstraintBlock(
        BoundedPathConstraint(
            lambda time, state, control, args: jnp.asarray(0.0),
            lower=0.0 if equality else -jnp.inf,
            upper=0.0,
            constraint_id=constraint_id,
        ),
        scope=scope,
        participants=participants,
        owner=owner,
        site=GameConstraintSite.PATH,
        equality=equality,
        residual_shape=(),
        time_dependent=False,
        state_dependent=False,
        control_dependencies=control_dependencies,
    )


def _plan():
    return FeedbackQuasiNashPlan(
        residual_tolerance=2.0e-5,
        feasibility_tolerance=2.0e-5,
        strict_complementarity_tolerance=1.0e-6,
        curvature_tolerance=1.0e-6,
        rank_relative_tolerance=1.0e-6,
        rank_absolute_tolerance=1.0e-7,
        maximum_condition=1.0e7,
    )


def _solve(
    suggestion,
    blocks=(),
    *,
    residuals=None,
    control_jacobians=None,
    state_jacobians=None,
    active_set=None,
    variational=False,
    problem_id="feedback-constraint-case",
):
    constraints = OpenLoopGameConstraints(suggestion.model.partition, blocks)
    problem = ConstrainedFeedbackGameProblem(
        suggestion,
        constraints,
        residuals,
        state_jacobians,
        control_jacobians,
        active_set,
        variational=variational,
        problem_id=problem_id,
    )
    return solve_feedback_quasi_nash_model(problem, plan=_plan())


def test_unconstrained_model_reduces_to_the_local_lq_suggestion():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    suggestion = _local_suggestion(
        partition,
        jnp.asarray(
            [
                [[2.0, 0.3], [0.3, 1.0]],
                [[1.2, -0.2], [-0.2, 3.0]],
            ]
        ),
        jnp.asarray([[-0.5, 0.2], [0.1, 0.75]]),
        state_control_cross=jnp.asarray([[[0.4, -0.1]], [[0.25, 0.6]]]),
    )

    result = _solve(suggestion)

    assert result.status == int(FeedbackQuasiNashStatus.SUCCESS)
    assert bool(result.policy_authoritative)
    np.testing.assert_allclose(
        result.feedback_gain, suggestion.feedback_gain, rtol=2.0e-5, atol=2.0e-5
    )
    np.testing.assert_allclose(
        result.feedforward, suggestion.feedforward, rtol=2.0e-5, atol=2.0e-5
    )
    assert result.multipliers.shape == (1, 0)
    assert result.kkt_ranks[0] == partition.joint_control_size


def test_active_bound_returns_affine_control_and_positive_multiplier():
    partition = PlayerControlPartition(("one",), (1,))
    suggestion = _local_suggestion(
        partition,
        jnp.asarray([[[1.0]]]),
        jnp.asarray([[-2.0]]),
    )
    bound = _block("upper-one", owner="one", participants=("one",))

    result = _solve(
        suggestion,
        (bound,),
        residuals=jnp.asarray([[-1.0]]),
        state_jacobians=jnp.zeros((1, 1, 1)),
        control_jacobians=jnp.asarray([[[1.0]]]),
        active_set=jnp.asarray([[True]]),
    )

    assert result.status == int(FeedbackQuasiNashStatus.SUCCESS)
    np.testing.assert_allclose(result.feedforward, [[1.0]], atol=2.0e-5)
    np.testing.assert_allclose(result.multipliers, [[1.0]], atol=2.0e-5)
    np.testing.assert_allclose(result.active_residuals, [[0.0]], atol=2.0e-5)
    assert result.licq_ranks[0] == 1
    assert result.own_minimum_curvatures[0, 0] > 0.0


def test_private_and_shared_variational_multipliers_have_distinct_layouts():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    suggestion = _local_suggestion(
        partition,
        jnp.stack((jnp.eye(2), jnp.eye(2))),
        jnp.asarray([[-2.0, 0.0], [0.0, -1.0]]),
    )
    private = _block(
        "left-private",
        owner="left",
        participants=("left",),
    )
    shared = _block(
        "shared-capacity",
        owner=None,
        participants=("left", "right"),
        scope=GameConstraintScope.SHARED,
    )

    result = _solve(
        suggestion,
        (private, shared),
        residuals=jnp.zeros((1, 2)),
        state_jacobians=jnp.zeros((1, 2, 1)),
        control_jacobians=jnp.asarray([[[1.0, 0.0], [1.0, 1.0]]]),
        active_set=jnp.asarray([[True, True]]),
        variational=True,
    )

    assert result.status == int(FeedbackQuasiNashStatus.SUCCESS)
    assert result.problem.multiplier_player_indices == (0, -1)
    assert result.private_multipliers[0].shape == (1, 1)
    assert result.private_multipliers[1].shape == (1, 0)
    assert result.shared_player_multipliers[0].shape == (1, 0)
    assert result.shared_player_multipliers[1].shape == (1, 0)
    assert result.variational_multipliers.shape == (1, 1)
    np.testing.assert_allclose(result.private_multipliers[0], [[1.0]], atol=2.0e-5)
    np.testing.assert_allclose(result.variational_multipliers, [[1.0]], atol=2.0e-5)


def test_generic_shared_rows_keep_player_multiplier_copies_and_report_nonisolation():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    suggestion = _local_suggestion(
        partition,
        jnp.stack((jnp.eye(2), jnp.eye(2))),
        jnp.asarray([[-1.0, 0.0], [0.0, -2.0]]),
    )
    shared = _block(
        "generic-shared",
        owner=None,
        participants=("left", "right"),
        scope=GameConstraintScope.SHARED,
    )

    result = _solve(
        suggestion,
        (shared,),
        residuals=jnp.zeros((1, 1)),
        state_jacobians=jnp.zeros((1, 1, 1)),
        control_jacobians=jnp.asarray([[[1.0, 1.0]]]),
        active_set=jnp.asarray([[True]]),
        variational=False,
    )

    assert result.problem.multiplier_constraint_indices == (0, 0)
    assert result.problem.multiplier_player_indices == (0, 1)
    assert result.shared_player_multipliers[0].shape == (1, 1)
    assert result.shared_player_multipliers[1].shape == (1, 1)
    assert result.licq_ranks[0] == 2
    assert result.status == int(FeedbackQuasiNashStatus.COUPLED_KKT_RANK_DEFICIENT)
    assert not bool(result.policy_authoritative)
    assert not bool(result.unique_feedback_sensitivity_available)
    assert not result.generic_gne_existence_rejected


def test_linearly_dependent_active_private_rows_fail_licq():
    partition = PlayerControlPartition(("one",), (1,))
    suggestion = _local_suggestion(
        partition,
        jnp.asarray([[[1.0]]]),
        jnp.asarray([[0.0]]),
    )
    first = _block(
        "first-equality",
        owner="one",
        participants=("one",),
        equality=True,
    )
    second = _block(
        "second-equality",
        owner="one",
        participants=("one",),
        equality=True,
    )

    result = _solve(
        suggestion,
        (first, second),
        residuals=jnp.zeros((1, 2)),
        state_jacobians=jnp.zeros((1, 2, 1)),
        control_jacobians=jnp.asarray([[[1.0], [2.0]]]),
        active_set=jnp.asarray([[True, True]]),
    )

    assert result.licq_ranks[0] == 1
    assert result.active_multiplier_counts[0] == 2
    assert result.status == int(FeedbackQuasiNashStatus.LICQ_FAILURE)


def test_zero_active_inequality_multiplier_fails_strict_complementarity():
    partition = PlayerControlPartition(("one",), (1,))
    suggestion = _local_suggestion(
        partition,
        jnp.asarray([[[1.0]]]),
        jnp.asarray([[0.0]]),
    )
    bound = _block("zero-multiplier", owner="one", participants=("one",))

    result = _solve(
        suggestion,
        (bound,),
        residuals=jnp.zeros((1, 1)),
        state_jacobians=jnp.zeros((1, 1, 1)),
        control_jacobians=jnp.ones((1, 1, 1)),
        active_set=jnp.asarray([[True]]),
    )

    np.testing.assert_allclose(result.multipliers, [[0.0]], atol=2.0e-5)
    assert result.status == int(FeedbackQuasiNashStatus.STRICT_COMPLEMENTARITY_FAILURE)


def test_inactive_violation_is_reported_without_an_active_set_switch():
    partition = PlayerControlPartition(("one",), (1,))
    suggestion = _local_suggestion(
        partition,
        jnp.asarray([[[1.0]]]),
        jnp.asarray([[-2.0]]),
    )
    bound = _block("inactive-upper", owner="one", participants=("one",))

    result = _solve(
        suggestion,
        (bound,),
        residuals=jnp.asarray([[-1.0]]),
        state_jacobians=jnp.zeros((1, 1, 1)),
        control_jacobians=jnp.ones((1, 1, 1)),
        active_set=jnp.asarray([[False]]),
    )

    np.testing.assert_allclose(result.feedforward, [[2.0]], atol=2.0e-5)
    np.testing.assert_allclose(result.multipliers, [[0.0]], atol=2.0e-5)
    np.testing.assert_allclose(result.maximum_inactive_violations, [1.0], atol=2.0e-5)
    assert result.status == int(FeedbackQuasiNashStatus.INACTIVE_CONSTRAINT_VIOLATION)
    assert not bool(result.active_set[0, 0])


def test_nonsymmetric_coupled_private_kkt_can_be_singular_despite_licq():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    suggestion = _local_suggestion(
        partition,
        jnp.asarray(
            [
                [[1.0, 2.0], [2.0, 5.0]],
                [[2.0, 1.0], [1.0, 1.0]],
            ]
        ),
        jnp.zeros((2, 2)),
    )
    coupled = _block(
        "left-owned-coupled",
        owner="left",
        participants=("left", "right"),
        scope=GameConstraintScope.PLAYER_OWNED_COUPLED,
        equality=True,
        control_dependencies=("left", "right"),
    )

    result = _solve(
        suggestion,
        (coupled,),
        residuals=jnp.zeros((1, 1)),
        state_jacobians=jnp.zeros((1, 1, 1)),
        control_jacobians=jnp.asarray([[[1.0, 1.0]]]),
        active_set=jnp.asarray([[True]]),
    )

    assert result.licq_ranks[0] == 1
    assert result.kkt_ranks[0] == 2
    assert result.status == int(FeedbackQuasiNashStatus.COUPLED_KKT_RANK_DEFICIENT)
    assert not bool(result.policy_authoritative)


def test_player_permutation_permutes_policy_and_owned_multiplier_layout():
    original_partition = PlayerControlPartition(("left", "right"), (1, 1))
    original = _local_suggestion(
        original_partition,
        jnp.stack((jnp.eye(2), jnp.eye(2))),
        jnp.asarray([[-1.0, 0.0], [0.0, 0.5]]),
        suggestion_id="feedback-permutation-original",
    )
    original_bound = _block(
        "left-active",
        owner="left",
        participants=("left",),
    )
    original_result = _solve(
        original,
        (original_bound,),
        residuals=jnp.zeros((1, 1)),
        state_jacobians=jnp.zeros((1, 1, 1)),
        control_jacobians=jnp.asarray([[[1.0, 0.0]]]),
        active_set=jnp.asarray([[True]]),
        problem_id="feedback-permutation-original",
    )

    permuted_partition = PlayerControlPartition(("right", "left"), (1, 1))
    permutation = jnp.asarray([1, 0])
    original_costs = jnp.stack((jnp.eye(2), jnp.eye(2)))
    original_linear = jnp.asarray([[-1.0, 0.0], [0.0, 0.5]])
    permuted = _local_suggestion(
        permuted_partition,
        original_costs[permutation][:, permutation][:, :, permutation],
        original_linear[permutation][:, permutation],
        suggestion_id="feedback-permutation-swapped",
    )
    permuted_bound = _block(
        "left-active",
        owner="left",
        participants=("left",),
    )
    permuted_result = _solve(
        permuted,
        (permuted_bound,),
        residuals=jnp.zeros((1, 1)),
        state_jacobians=jnp.zeros((1, 1, 1)),
        control_jacobians=jnp.asarray([[[0.0, 1.0]]]),
        active_set=jnp.asarray([[True]]),
        problem_id="feedback-permutation-swapped",
    )

    assert original_result.status == int(FeedbackQuasiNashStatus.SUCCESS)
    assert permuted_result.status == int(FeedbackQuasiNashStatus.SUCCESS)
    np.testing.assert_allclose(
        permuted_result.feedforward[..., permutation],
        original_result.feedforward,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        permuted_result.private_multipliers[1],
        original_result.private_multipliers[0],
        atol=2.0e-5,
    )


def test_result_labels_only_a_fixed_active_local_quasi_nash_model():
    partition = PlayerControlPartition(("one",), (1,))
    suggestion = _local_suggestion(
        partition,
        jnp.asarray([[[1.0]]]),
        jnp.asarray([[-2.0]]),
    )
    result = _solve(suggestion)

    assert result.model_label == CONSTRAINED_FEEDBACK_QUASI_NASH_MODEL
    assert result.fixed_active_set
    assert result.local_piecewise_affine_suggestion
    assert not result.exact_nonlinear_feedback_nash_claim
    assert not result.global_gne_claim
    assert not result.off_trajectory_feasibility_claim
    assert not result.active_switch_derivative_available
