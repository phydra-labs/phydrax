#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.control.games._coupled_hjb import (
    CoupledHJBPolicyIterationPlan,
    DiscreteCoupledHJBProblem,
    DiscreteCoupledHJBStatus,
    LOCAL_FEEDBACK_FIXED_POINT,
    solve_coupled_hjb_reference,
)
from phydrax.control.stochastic import (
    BoundedUniformGrid1D,
    DiscreteHJBProblem,
    solve_discrete_hjb_reference,
)
from phydrax.dynamics import TimeGrid


def _zero_coefficient(player, time, state, joint_action, args):
    del player, time, state, joint_action, args
    return 0.0


def _static_problem(cost, *, actions=None, problem_id="static-coupled-hjb"):
    if actions is None:
        actions = (jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 1.0]))
    grid = BoundedUniformGrid1D(-1.0, 1.0, 5)
    time_grid = TimeGrid(jnp.asarray([0.0, 0.05, 0.1]), time_id=problem_id)
    players = len(actions)
    return DiscreteCoupledHJBProblem(
        grid,
        time_grid,
        actions,
        jnp.zeros((players, grid.num_points)),
        jnp.zeros((players, time_grid.num_times, 2)),
        _zero_coefficient,
        _zero_coefficient,
        cost,
        problem_id=problem_id,
    )


def _plan(
    *,
    maximum_iterations=3,
    damping=1.0,
    update="jacobi",
    plan_id="coupled-policy-iteration",
):
    return CoupledHJBPolicyIterationPlan(
        maximum_iterations=maximum_iterations,
        fixed_point_tolerance=0.0,
        damping=damping,
        update=update,
        plan_id=plan_id,
    )


def _uniform_selectors(problem, selectors):
    selectors = np.asarray(selectors, dtype=np.int32)
    return np.broadcast_to(
        selectors[:, None, None],
        (
            problem.num_players,
            problem.time_grid.num_times - 1,
            problem.spatial_grid.num_points - 2,
        ),
    ).copy()


def test_decoupled_players_reduce_to_independent_discrete_hjb_tables():
    actions = (jnp.asarray([-1.0, 1.0]), jnp.asarray([0.0, 2.0, 3.0]))
    bases = (1.0, 2.0)
    targets = (1.0, 2.0)

    def coupled_cost(player, time, state, joint_action, args):
        del time, state, args
        return bases[player] + (joint_action[player] - targets[player]) ** 2

    problem = _static_problem(
        coupled_cost, actions=actions, problem_id="decoupled-reduction"
    )
    coupled = solve_coupled_hjb_reference(problem, _plan(maximum_iterations=2))

    for player in range(problem.num_players):

        def independent_drift(time, state, action, args):
            del time, state, action, args
            return 0.0

        def independent_diffusion(time, state, action, args):
            del time, state, action, args
            return 0.0

        def independent_cost(time, state, action, args, *, player=player):
            del time, state, args
            return bases[player] + (action - targets[player]) ** 2

        independent_problem = DiscreteHJBProblem(
            problem.spatial_grid,
            problem.time_grid,
            actions[player],
            problem.terminal_values[player],
            problem.boundary_values[player],
            independent_drift,
            independent_diffusion,
            independent_cost,
            problem_id=f"independent-{player}",
        )
        independent = solve_discrete_hjb_reference(independent_problem)
        np.testing.assert_allclose(coupled.values[player], independent.values)
        np.testing.assert_array_equal(
            coupled.action_selectors[player], independent.action_selectors
        )

    assert bool(coupled.successful)
    assert coupled.joint_action_profiles.shape == (6, 2)
    assert float(coupled.evidence.maximum_policy_evaluation_residual) < 1.0e-7


def test_two_player_joint_profiles_converge_to_own_hamiltonian_fixed_point():
    def cost(player, time, state, joint_action, args):
        del time, state, args
        if player == 0:
            return (joint_action[0] - 1.0) ** 2 + 0.1 * joint_action[1]
        return (joint_action[1] - joint_action[0]) ** 2 + 0.2

    problem = _static_problem(cost, problem_id="two-player-fixed-point")
    result = solve_coupled_hjb_reference(problem, _plan(maximum_iterations=2))

    np.testing.assert_array_equal(result.action_selectors[0], 1)
    np.testing.assert_array_equal(result.action_selectors[1], 1)
    np.testing.assert_array_equal(result.action_selectors, result.best_response_selectors)
    assert bool(result.successful)
    assert float(result.evidence.maximum_own_action_hamiltonian_gap) == 0.0
    assert float(result.evidence.maximum_fixed_point_residual) == 0.0
    np.testing.assert_allclose(result.own_action_hamiltonian_gaps, 0.0)
    np.testing.assert_allclose(
        result.player_policy_evaluation_residuals, 0.0, rtol=0.0, atol=1.0e-14
    )
    assert float(result.evidence.maximum_boundary_residual) == 0.0
    assert float(result.evidence.maximum_terminal_residual) == 0.0
    assert bool(result.evidence.boundary_passed)
    assert bool(result.evidence.terminal_passed)
    assert bool(result.evidence.refinement_passed)
    assert result.evidence.scope == (
        "declared-bounded-grid-local-feedback-fixed-point-evidence-only"
    )


def test_multiple_starts_preserve_distinct_coordination_branches_and_ids():
    def coordination_cost(player, time, state, joint_action, args):
        del time, state, args
        opponent = 1 - player
        return (joint_action[player] - joint_action[opponent]) ** 2

    problem = _static_problem(coordination_cost, problem_id="coordination-branches")
    zero = _uniform_selectors(problem, [0, 0])
    one = _uniform_selectors(problem, [1, 1])
    starts = np.stack((zero, one))
    result = solve_coupled_hjb_reference(
        problem,
        _plan(maximum_iterations=1),
        initial_policy_selectors=starts,
        branch_ids=("lower-coordination", "upper-coordination"),
    )

    assert result.evidence.branch.branch_ids == (
        "lower-coordination",
        "upper-coordination",
    )
    np.testing.assert_array_equal(result.branch_action_selectors[0], 0)
    np.testing.assert_array_equal(result.branch_action_selectors[1], 1)
    np.testing.assert_array_equal(result.evidence.branch.converged, [True, True])
    assert bool(result.evidence.branch.branch_dependence_detected)
    assert result.selected_branch_id == "lower-coordination"
    assert result.evidence.branch.history_capacity == 1


def test_jacobi_and_gauss_seidel_have_distinct_declared_one_sweep_updates():
    def anti_coordination_cost(player, time, state, joint_action, args):
        del time, state, args
        opponent = 1 - player
        return (joint_action[player] + joint_action[opponent] - 1.0) ** 2

    problem = _static_problem(anti_coordination_cost, problem_id="update-order")
    start = _uniform_selectors(problem, [0, 0])
    jacobi = solve_coupled_hjb_reference(
        problem,
        _plan(maximum_iterations=1, update="jacobi", plan_id="jacobi"),
        initial_policy_selectors=start,
    )
    gauss_seidel = solve_coupled_hjb_reference(
        problem,
        _plan(
            maximum_iterations=1,
            update="gauss_seidel",
            plan_id="gauss-seidel",
        ),
        initial_policy_selectors=start,
    )

    np.testing.assert_array_equal(jacobi.action_selectors[0], 1)
    np.testing.assert_array_equal(jacobi.action_selectors[1], 1)
    np.testing.assert_array_equal(gauss_seidel.action_selectors[0], 1)
    np.testing.assert_array_equal(gauss_seidel.action_selectors[1], 0)
    assert not bool(jacobi.successful)
    assert bool(gauss_seidel.successful)
    assert jacobi.update == "jacobi"
    assert gauss_seidel.update == "gauss_seidel"


def test_fixed_damping_relaxes_policy_probabilities_and_is_recorded():
    def dominant_one_cost(player, time, state, joint_action, args):
        del time, state, args
        return (joint_action[player] - 1.0) ** 2

    problem = _static_problem(dominant_one_cost, problem_id="fixed-damping")
    result = solve_coupled_hjb_reference(
        problem,
        _plan(maximum_iterations=1, damping=0.5),
        initial_policy_selectors=_uniform_selectors(problem, [0, 0]),
    )

    for probability in result.policy_probabilities:
        np.testing.assert_allclose(probability, 0.5)
    np.testing.assert_allclose(
        result.evidence.branch.fixed_point_residual_history[0, 0], 1.0
    )
    np.testing.assert_allclose(result.evidence.branch.update_residual_history[0, 0], 0.5)
    assert float(result.evidence.maximum_fixed_point_residual) == 0.5
    assert result.damping == 0.5
    assert result.history_capacity == 1
    assert not bool(result.successful)


def test_selector_ties_use_lowest_declared_index_and_expose_tie_identity():
    def tied_cost(player, time, state, joint_action, args):
        del player, time, state, joint_action, args
        return 0.0

    problem = _static_problem(tied_cost, problem_id="selector-tie")
    result = solve_coupled_hjb_reference(
        problem,
        _plan(maximum_iterations=1),
        initial_policy_selectors=_uniform_selectors(problem, [1, 1]),
    )

    np.testing.assert_array_equal(result.action_selectors, 0)
    assert result.selector_id == "own-hamiltonian-argmin"
    assert result.tie_break_id == "lowest-declared-action-index"
    assert int(result.evidence.maximum_tie_count) == 2
    assert bool(result.successful)


def test_nonconvergence_fills_fixed_capacity_history_without_success_label():
    def anti_coordination_cost(player, time, state, joint_action, args):
        del time, state, args
        opponent = 1 - player
        return (joint_action[player] + joint_action[opponent] - 1.0) ** 2

    problem = _static_problem(anti_coordination_cost, problem_id="jacobi-cycle")
    result = solve_coupled_hjb_reference(
        problem,
        _plan(maximum_iterations=2, update="jacobi"),
        initial_policy_selectors=_uniform_selectors(problem, [0, 0]),
    )

    assert int(result.status) == int(DiscreteCoupledHJBStatus.MAXIMUM_POLICY_ITERATIONS)
    assert not bool(result.successful)
    assert result.evidence.branch.fixed_point_residual_history.shape == (1, 2)
    np.testing.assert_array_equal(
        result.evidence.branch.iteration_validity_history, [[True, True]]
    )
    assert result.status_label == "MAXIMUM_POLICY_ITERATIONS"


def test_nested_refinement_is_explicit_and_can_gate_a_local_fixed_point():
    grid = BoundedUniformGrid1D(-1.0, 1.0, 7)
    time_grid = TimeGrid(jnp.asarray([0.0, 0.02, 0.04]), time_id="coupled-refinement")
    terminal = np.stack(
        (
            np.asarray(grid.points) ** 2,
            2.0 * np.asarray(grid.points) ** 2,
        )
    )
    boundary = np.empty((2, time_grid.num_times, 2))
    boundary[0] = 1.0
    boundary[1] = 2.0

    def drift(player, time, state, joint_action, args):
        del time, state, joint_action, args
        return 0.2 if player == 0 else -0.15

    problem = DiscreteCoupledHJBProblem(
        grid,
        time_grid,
        (jnp.asarray([0.0]), jnp.asarray([0.0])),
        terminal,
        boundary,
        drift,
        _zero_coefficient,
        _zero_coefficient,
        problem_id="coupled-refinement",
    )
    strict = solve_coupled_hjb_reference(
        problem,
        _plan(maximum_iterations=1),
        refinement_absolute_tolerance=0.0,
        refinement_relative_tolerance=0.0,
        residual_tolerance=1.0e-7,
    )
    loose = solve_coupled_hjb_reference(
        problem,
        _plan(maximum_iterations=1),
        refinement_absolute_tolerance=1.0,
        refinement_relative_tolerance=0.0,
        residual_tolerance=1.0e-7,
    )

    assert strict.common_grid_difference.shape == strict.values.shape
    assert float(strict.evidence.maximum_refinement_difference) > 0.0
    assert int(strict.status) == int(DiscreteCoupledHJBStatus.REFINEMENT_GATE_FAILED)
    assert not bool(strict.evidence.refinement_passed)
    assert bool(loose.evidence.refinement_passed)
    assert bool(loose.successful)


def test_success_label_is_local_only_and_carries_no_broad_solution_claim():
    def cost(player, time, state, joint_action, args):
        del player, time, state, joint_action, args
        return 0.0

    result = solve_coupled_hjb_reference(
        _static_problem(cost, problem_id="local-claim-only"),
        _plan(maximum_iterations=1),
    )

    assert result.status_label == LOCAL_FEEDBACK_FIXED_POINT
    assert result.certificate_label == LOCAL_FEEDBACK_FIXED_POINT
    assert bool(result.local_feedback_fixed_point)
    assert result.candidate_evaluation_only
    assert not result.viscosity_solution_claimed
    assert not result.unique_solution_claimed
    assert not result.global_nash_equilibrium_claimed
    for forbidden in ("GLOBAL", "UNIQUE", "VISCOSITY", "NASH"):
        assert forbidden not in result.status_label
