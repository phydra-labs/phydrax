#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.control._lqr import AffineFeedbackPolicy
from phydrax.control.games._ilq import (
    ILQFeedbackGameStatus,
    ILQFeedbackGameTrialReason,
    plan_ilq_feedback_game,
    prepare_ilq_feedback_game,
    refresh_ilq_feedback_game,
    solve_ilq_feedback_game,
    solve_prepared_ilq_feedback_game,
)
from phydrax.control.games._layout import PlayerControlPartition
from phydrax.control.games._linear_quadratic import (
    finite_horizon_lq_feedback_nash,
)
from phydrax.control.games._local_lq import (
    LocalAffineGamePolicy,
    LocalAffineGameSuggestionStatus,
)
from phydrax.control.games._nonlinear import (
    DeterministicFeedbackGameProblem,
    ILQGameScaling,
)


def _affine_policy(problem, controls, *, policy_id="initial-affine"):
    values = jnp.asarray(controls)
    horizon = problem.time_grid.num_steps
    cases = problem.case_shape
    expected = cases + (horizon, problem.control_size)
    values = jnp.broadcast_to(values, expected)
    return AffineFeedbackPolicy(
        jnp.zeros(cases + (horizon, problem.control_size, problem.state_size)),
        values,
        time_grid=problem.time_grid,
        state_size=problem.state_size,
        case_shape=cases,
        policy_id=policy_id,
    )


def _static_game(
    stage_costs,
    *,
    initial_controls=None,
    initial_state=0.0,
    case_shape=(),
    problem_id="static-ilq-game",
):
    players = len(stage_costs)
    input_layout = phx.dynamics.InputLayout((players,), roles="control")
    system = phx.dynamics.DiscreteSystem(
        lambda context, state, control, args: state,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=input_layout,
        system_id=f"{problem_id}:system",
    )
    problem = DeterministicFeedbackGameProblem(
        phx.control.DiscreteControlDynamics(system),
        phx.dynamics.TimeGrid(jnp.asarray([0.0, 1.0]), time_id=f"{problem_id}:grid"),
        jnp.broadcast_to(jnp.asarray([initial_state]), case_shape + (1,)),
        PlayerControlPartition(
            tuple(f"p{index}" for index in range(players)), (1,) * players
        ),
        stage_costs=stage_costs,
        terminal_costs=tuple(
            (lambda time, state, args: jnp.asarray(0.0)) for _ in range(players)
        ),
        problem_id=problem_id,
    )
    if initial_controls is None:
        initial_controls = jnp.zeros(case_shape + (1, players))
    policy = _affine_policy(problem, initial_controls)
    scaling = ILQGameScaling(jnp.ones(1), jnp.ones(players), jnp.ones(players))
    return problem, policy, scaling


def _separable_affine_lq_game(*, case_shape=()):
    horizon = 2
    state_size = 2
    control_size = 2
    target = jnp.asarray([0.75, -1.25])
    input_layout = phx.dynamics.InputLayout((control_size,), roles="control")

    def transition(context, state, control, args):
        del context, args
        return state + control

    def stage(player):
        def cost(context, state, control, args):
            del context, state, args
            return 0.5 * control[player] ** 2

        return cost

    def terminal(player):
        def cost(time, state, args):
            del time, args
            return 0.5 * (state[player] - target[player]) ** 2

        return cost

    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((state_size,)),
        input_layout=input_layout,
        system_id="separable-affine-lq-system",
    )
    problem = DeterministicFeedbackGameProblem(
        phx.control.DiscreteControlDynamics(system),
        phx.dynamics.TimeGrid(
            jnp.arange(horizon + 1, dtype=float), time_id="separable-affine-lq-grid"
        ),
        jnp.broadcast_to(jnp.asarray([2.0, -0.25]), case_shape + (state_size,)),
        PlayerControlPartition(("left", "right"), (1, 1)),
        stage_costs=(stage(0), stage(1)),
        terminal_costs=(terminal(0), terminal(1)),
        problem_id="separable-affine-lq-game",
    )
    policy = _affine_policy(problem, jnp.zeros(case_shape + (horizon, control_size)))
    scaling = ILQGameScaling(jnp.ones(state_size), jnp.ones(control_size), jnp.ones(2))
    return problem, policy, scaling, target


def test_exact_affine_lq_converges_to_the_exact_feedback_law():
    problem, policy, scaling, target = _separable_affine_lq_game()
    result = solve_ilq_feedback_game(
        problem,
        scaling,
        policy,
        maximum_iterations=4,
        residual_tolerance=2.0e-6,
        step_tolerance=2.0e-6,
        dynamics_tolerance=2.0e-7,
    )

    horizon = problem.time_grid.num_steps
    A = jnp.broadcast_to(jnp.eye(2), (horizon, 2, 2))
    B = jnp.broadcast_to(jnp.eye(2), (horizon, 2, 2))
    Q = jnp.zeros((2, horizon, 2, 2))
    R = jnp.zeros((2, horizon, 2, 2))
    R = R.at[0, :, 0, 0].set(1.0)
    R = R.at[1, :, 1, 1].set(1.0)
    terminal_Q = jnp.zeros((2, 2, 2))
    terminal_Q = terminal_Q.at[0, 0, 0].set(1.0)
    terminal_Q = terminal_Q.at[1, 1, 1].set(1.0)
    terminal_linear = jnp.zeros((2, 2))
    terminal_linear = terminal_linear.at[0, 0].set(-target[0])
    terminal_linear = terminal_linear.at[1, 1].set(-target[1])
    exact = finite_horizon_lq_feedback_nash(
        A,
        B,
        Q,
        R,
        terminal_Q,
        problem.partition,
        terminal_linear=terminal_linear,
        terminal_constants=0.5 * target**2,
        time_grid=problem.time_grid,
    )

    assert bool(result.successful)
    assert int(result.status) == int(ILQFeedbackGameStatus.SUCCESS)
    np.testing.assert_allclose(
        result.policy.feedback_gain, exact.feedback_gain, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        result.policy.absolute_feedforward,
        exact.feedforward,
        rtol=2e-6,
        atol=2e-6,
    )
    assert result.diagnostics.accepted_iterations >= 1


def test_compatible_local_affine_initial_profile_is_preserved_by_preparation():
    problem, _, scaling, _ = _separable_affine_lq_game()
    states = jnp.broadcast_to(
        problem.initial_state, (problem.time_grid.num_times, problem.state_size)
    )
    controls = jnp.zeros((problem.time_grid.num_steps, problem.control_size))
    initial = LocalAffineGamePolicy(
        states,
        controls,
        jnp.zeros(
            (
                problem.time_grid.num_steps,
                problem.control_size,
                problem.state_size,
            )
        ),
        jnp.zeros_like(controls),
        feedforward_scale=0.0,
        time_grid=problem.time_grid,
        input_layout=problem.dynamics.system.input_layout,
        partition=problem.partition,
        policy_id="local-affine-initial-profile",
    )
    plan = plan_ilq_feedback_game(problem, scaling, maximum_iterations=2)
    prepared = prepare_ilq_feedback_game(plan, problem, initial)
    result = solve_prepared_ilq_feedback_game(prepared)

    assert prepared.initial_policy_id == "local-affine-initial-profile"
    assert bool(result.successful)


def test_cubic_cross_terms_keep_independent_whole_control_owned_gradients():
    def player_zero(context, state, control, args):
        del context, state, args
        return 0.5 * (control[0] - 1.0) ** 2 + control[1] ** 3

    def player_one(context, state, control, args):
        del context, state, args
        return 0.5 * (control[1] + 2.0) ** 2 + control[0] ** 3

    problem, policy, scaling = _static_game(
        (player_zero, player_one), problem_id="cubic-whole-control"
    )
    result = solve_ilq_feedback_game(
        problem,
        scaling,
        policy,
        maximum_iterations=4,
        residual_tolerance=2.0e-6,
        step_tolerance=2.0e-6,
    )

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.trajectory.controls[0], jnp.asarray([1.0, -2.0]), atol=2e-6
    )
    np.testing.assert_allclose(result.residual.raw_owned_stationarity, 0.0, atol=2e-6)


def test_residual_armijo_accepts_when_one_players_cost_increases():
    def player_zero(context, state, control, args):
        del context, state, args
        return 0.5 * (control[0] - 1.0) ** 2 + 10.0 * control[1]

    def player_one(context, state, control, args):
        del context, state, args
        return 0.5 * (control[1] - 1.0) ** 2

    problem, policy, scaling = _static_game(
        (player_zero, player_one), problem_id="cost-increase-residual-decrease"
    )
    result = solve_ilq_feedback_game(
        problem,
        scaling,
        policy,
        maximum_iterations=2,
        residual_tolerance=2e-6,
        step_tolerance=2e-6,
    )
    diagnostics = result.diagnostics

    assert bool(diagnostics.accepted_history[0])
    assert int(diagnostics.trial_reason_history[0, 0]) == int(
        ILQFeedbackGameTrialReason.ACCEPTED
    )
    assert (
        diagnostics.trial_residual_merit_history[0, 0]
        < diagnostics.residual_merit_history[0]
    )
    assert (
        diagnostics.trial_player_cost_history[0, 0, 0]
        > diagnostics.player_cost_history[0, 0]
    )
    assert not diagnostics.player_costs_used_for_acceptance


def test_all_cost_decrease_cannot_override_residual_armijo_and_incumbent_is_preserved():
    def cubic(context, state, control, args):
        del context, state, args
        value = control[0]
        return 0.5 * value**2 + (2.0 / 3.0) * value**3 + value

    problem, policy, scaling = _static_game((cubic,), problem_id="reject-cost-merit")
    result = solve_ilq_feedback_game(
        problem,
        scaling,
        policy,
        maximum_iterations=1,
        maximum_line_search_steps=1,
    )
    diagnostics = result.diagnostics

    assert int(result.status) == int(ILQFeedbackGameStatus.LINE_SEARCH_FAILED)
    assert int(diagnostics.trial_reason_history[0, 0]) == int(
        ILQFeedbackGameTrialReason.ORIGINAL_RESIDUAL_ARMIJO_FAILED
    )
    assert (
        diagnostics.trial_player_cost_history[0, 0, 0]
        < diagnostics.player_cost_history[0, 0]
    )
    assert (
        diagnostics.trial_residual_merit_history[0, 0]
        > diagnostics.residual_merit_history[0]
    )
    np.testing.assert_array_equal(result.trajectory.controls, jnp.zeros((1, 1)))
    assert int(diagnostics.accepted_iterations) == 0


def test_zero_state_motion_is_not_a_convergence_test():
    def shifted_control(context, state, control, args):
        del context, state, args
        return 0.5 * (control[0] - 1.0) ** 2

    problem, policy, scaling = _static_game(
        (shifted_control,), problem_id="zero-state-motion"
    )
    result = solve_ilq_feedback_game(
        problem,
        scaling,
        policy,
        maximum_iterations=1,
        residual_tolerance=2e-6,
        step_tolerance=2e-6,
    )

    assert bool(result.successful)
    assert int(result.diagnostics.accepted_iterations) == 1
    assert result.diagnostics.stationarity_infinity_history[0] > 0.5
    np.testing.assert_allclose(
        result.diagnostics.trial_state_step_infinity_history[0, 0], 0.0
    )
    assert result.diagnostics.trial_control_step_infinity_history[0, 0] > 0.5


def test_regularized_stationary_direction_never_substitutes_for_unregularized_model():
    def concave(context, state, control, args):
        del context, state, args
        return -0.5 * control[0] ** 2

    problem, policy, scaling = _static_game((concave,), problem_id="regularized-only")
    result = solve_ilq_feedback_game(
        problem,
        scaling,
        policy,
        maximum_iterations=2,
        initial_proximal_regularization=2.0,
        maximum_proximal_regularization=2.0,
        residual_tolerance=1e-8,
        step_tolerance=1e-8,
    )

    assert not bool(result.successful)
    assert int(result.status) == int(
        ILQFeedbackGameStatus.FINAL_UNREGULARIZED_LOCAL_LQ_FAILED
    )
    assert not bool(result.diagnostics.final_unregularized_local_valid)
    assert int(result.local_suggestion.status) == int(
        LocalAffineGameSuggestionStatus.OWN_CURVATURE_NOT_POSITIVE_DEFINITE
    )
    assert result.certificate == "LOCAL_NOMINAL_NASH_STATIONARY"
    assert not bool(result.diagnostics.certificate_valid)


def test_coupled_rank_failure_is_retained_without_pseudoinverse_or_fallback():
    def player_zero(context, state, control, args):
        del context, state, args
        return 0.5 * control[0] ** 2 + control[0] * control[1]

    def player_one(context, state, control, args):
        del context, state, args
        return 0.5 * control[1] ** 2 + control[0] * control[1]

    problem, policy, scaling = _static_game(
        (player_zero, player_one), problem_id="rank-deficient-local-game"
    )
    result = solve_ilq_feedback_game(problem, scaling, policy, maximum_iterations=1)

    assert not bool(result.successful)
    assert int(result.local_suggestion.status) == int(
        LocalAffineGameSuggestionStatus.COUPLED_SYSTEM_RANK_DEFICIENT
    )
    assert int(result.local_suggestion.lq_diagnostics.coupled_ranks[0]) < 2


def test_physical_scaling_transforms_leave_dimensionless_result_equivalent():
    def make_problem(*, transformed):
        state_factor = 2.0 if transformed else 1.0
        control_factor = 4.0 if transformed else 1.0
        cost_factor = 5.0 if transformed else 1.0
        input_layout = phx.dynamics.InputLayout((1,), roles="control")

        def transition(context, state, control, args):
            del context, args
            return state + (control_factor / state_factor) * control

        def stage(context, state, control, args):
            del context, state, args
            return 0.5 * (control_factor * control[0]) ** 2 / cost_factor

        def terminal(time, state, args):
            del time, args
            return 0.5 * (state_factor * state[0]) ** 2 / cost_factor

        label = "transformed" if transformed else "physical"
        system = phx.dynamics.DiscreteSystem(
            transition,
            state_layout=phx.dynamics.StateLayout((1,)),
            input_layout=input_layout,
            system_id=f"scaling-{label}:system",
        )
        problem = DeterministicFeedbackGameProblem(
            phx.control.DiscreteControlDynamics(system),
            phx.dynamics.TimeGrid(
                jnp.asarray([0.0, 1.0]), time_id=f"scaling-{label}:grid"
            ),
            jnp.asarray([2.0 / state_factor]),
            PlayerControlPartition(("player",), (1,)),
            stage_costs=(stage,),
            terminal_costs=(terminal,),
            problem_id=f"scaling-{label}:problem",
        )
        policy = _affine_policy(problem, jnp.zeros((1, 1)))
        scaling = ILQGameScaling(
            jnp.asarray([3.0 / state_factor]),
            jnp.asarray([7.0 / control_factor]),
            jnp.asarray([11.0 / cost_factor]),
        )
        return problem, policy, scaling

    physical = make_problem(transformed=False)
    transformed = make_problem(transformed=True)
    first = solve_ilq_feedback_game(
        *physical,
        maximum_iterations=3,
        residual_tolerance=2e-6,
        step_tolerance=2e-6,
    )
    second = solve_ilq_feedback_game(
        *transformed,
        maximum_iterations=3,
        residual_tolerance=2e-6,
        step_tolerance=2e-6,
    )

    assert bool(first.successful) and bool(second.successful)
    np.testing.assert_allclose(
        first.trajectory.states, 2.0 * second.trajectory.states, atol=2e-6
    )
    np.testing.assert_allclose(
        first.trajectory.controls, 4.0 * second.trajectory.controls, atol=2e-6
    )
    np.testing.assert_allclose(
        first.residual.dimensionless_owned_stationarity,
        second.residual.dimensionless_owned_stationarity,
        atol=2e-6,
    )


def test_case_axes_filter_jit_and_fixed_histories_are_preserved():
    case_shape = (2, 2)
    problem, policy, scaling, _ = _separable_affine_lq_game(case_shape=case_shape)
    plan = plan_ilq_feedback_game(
        problem,
        scaling,
        maximum_iterations=3,
        maximum_line_search_steps=4,
        residual_tolerance=2e-6,
        step_tolerance=2e-6,
    )
    prepared = prepare_ilq_feedback_game(plan, problem, policy)
    result = eqx.filter_jit(solve_prepared_ilq_feedback_game)(prepared)
    diagnostics = result.diagnostics

    assert result.status.shape == case_shape
    assert result.trajectory.states.shape == case_shape + (3, 2)
    assert diagnostics.residual_merit_history.shape == case_shape + (3,)
    assert diagnostics.player_cost_history.shape == case_shape + (3, 2)
    assert diagnostics.trial_reason_history.shape == case_shape + (3, 4)
    assert diagnostics.trial_player_cost_history.shape == case_shape + (3, 4, 2)
    assert np.all(np.asarray(result.successful))
    padding = ~np.asarray(diagnostics.history_valid)
    assert np.all(np.isnan(np.asarray(diagnostics.residual_merit_history)[padding]))


def test_refresh_updates_materialization_ids_but_preserves_plan_topology():
    problem, first_policy, scaling = _static_game(
        (lambda context, state, control, args: 0.5 * (control[0] - 1.0) ** 2,),
        problem_id="refresh-ilq",
    )
    plan = plan_ilq_feedback_game(problem, scaling, maximum_iterations=2)
    prepared = prepare_ilq_feedback_game(plan, problem, first_policy)
    second_policy = _affine_policy(
        problem, jnp.asarray([[0.25]]), policy_id=first_policy.parameterization_id
    )
    refreshed = refresh_ilq_feedback_game(prepared, initial_policy=second_policy)

    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.prepared_id != prepared.prepared_id
    assert refreshed.materialization_id != prepared.materialization_id
    assert int(refreshed.materialization_version) == 1
    np.testing.assert_allclose(refreshed.initial_policy.feedforward, 0.25)

    different_grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0]), time_id="different-coordinate-provenance"
    )
    incompatible = AffineFeedbackPolicy(
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1)),
        time_grid=different_grid,
        state_size=1,
        policy_id="incompatible-coordinate-policy",
    )
    with pytest.raises(ValueError, match="time-grid identity"):
        refresh_ilq_feedback_game(prepared, initial_policy=incompatible)


def test_certificate_wording_and_claim_boundaries_are_exact():
    problem, policy, scaling = _static_game(
        (lambda context, state, control, args: 0.5 * control[0] ** 2,),
        problem_id="certificate-wording",
    )
    result = solve_ilq_feedback_game(
        problem,
        policy,
        scaling,
        maximum_iterations=1,
        residual_tolerance=1e-8,
        step_tolerance=1e-8,
    )

    assert bool(result.successful)
    assert result.certificate == "LOCAL_NOMINAL_NASH_STATIONARY"
    assert result.residual.certificate == "LOCAL_NOMINAL_NASH_STATIONARY"
    assert result.diagnostics.certificate == "LOCAL_NOMINAL_NASH_STATIONARY"
    assert result.diagnostics.acceptance_method == (
        "original-unregularized-dimensionless-residual-armijo"
    )
    assert not result.diagnostics.feedback_nash_claimed
    assert not result.diagnostics.global_convergence_claimed
    assert not result.diagnostics.implicit_differentiation
