#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.control.games import (
    DeterministicFeedbackGameProblem,
    evaluate_game_policy,
    GamePolicyEvaluation,
    GamePolicyEvaluationStatus,
    ILQGameScaling,
    nominal_nash_residual,
)


def _player_stage_cost(player):
    def cost(context, state, control, args):
        step = context.step_index
        return (
            0.5 * state @ args["Q"][player, step] @ state
            + state @ args["N"][player, step] @ control
            + 0.5 * control @ args["R"][player, step] @ control
            + args["q"][player, step] @ state
            + args["r"][player, step] @ control
            + args["stage_constant"][player, step]
        )

    return cost


def _player_terminal_cost(player):
    def cost(time, state, args):
        del time
        return (
            0.5 * state @ args["Q_terminal"][player] @ state
            + args["q_terminal"][player] @ state
            + args["terminal_constant"][player]
        )

    return cost


def _affine_data():
    return {
        "A": jnp.asarray(
            [
                [[1.0, 0.2], [-0.1, 0.9]],
                [[0.8, -0.1], [0.3, 1.1]],
                [[1.05, 0.0], [-0.2, 0.95]],
            ]
        ),
        "B": jnp.asarray(
            [
                [[0.4, -0.2, 0.1], [0.3, 0.5, -0.4]],
                [[0.2, 0.6, -0.3], [-0.5, 0.1, 0.2]],
                [[0.7, -0.1, 0.2], [0.0, 0.4, 0.6]],
            ]
        ),
        "dynamics_bias": jnp.asarray([[0.05, -0.02], [-0.04, 0.03], [0.02, 0.01]]),
        "Q": jnp.asarray(
            [
                [
                    [[1.2, 0.1], [0.1, 0.8]],
                    [[0.9, -0.2], [-0.2, 1.4]],
                    [[1.1, 0.0], [0.0, 0.7]],
                ],
                [
                    [[0.6, -0.1], [-0.1, 1.0]],
                    [[1.3, 0.2], [0.2, 0.5]],
                    [[0.8, 0.15], [0.15, 1.2]],
                ],
            ]
        ),
        "R": jnp.asarray(
            [
                [
                    [[1.4, 0.1, -0.2], [0.1, 1.0, 0.3], [-0.2, 0.3, 0.9]],
                    [[1.0, -0.1, 0.2], [-0.1, 1.6, 0.0], [0.2, 0.0, 1.1]],
                    [[1.2, 0.2, 0.1], [0.2, 0.8, -0.1], [0.1, -0.1, 1.5]],
                ],
                [
                    [[0.9, -0.2, 0.0], [-0.2, 1.3, 0.1], [0.0, 0.1, 1.7]],
                    [[1.5, 0.0, -0.1], [0.0, 0.7, 0.2], [-0.1, 0.2, 1.0]],
                    [[0.8, 0.1, 0.3], [0.1, 1.4, 0.0], [0.3, 0.0, 1.2]],
                ],
            ]
        ),
        "N": jnp.asarray(
            [
                [
                    [[0.1, -0.2, 0.0], [0.3, 0.0, 0.2]],
                    [[-0.1, 0.2, 0.1], [0.0, 0.1, -0.2]],
                    [[0.2, 0.0, -0.1], [-0.1, 0.3, 0.1]],
                ],
                [
                    [[-0.2, 0.1, 0.3], [0.1, -0.1, 0.0]],
                    [[0.0, -0.2, 0.1], [0.2, 0.1, 0.3]],
                    [[0.1, 0.2, 0.0], [0.0, -0.2, 0.2]],
                ],
            ]
        ),
        "q": jnp.asarray(
            [
                [[0.2, -0.1], [-0.3, 0.4], [0.1, 0.2]],
                [[-0.1, 0.3], [0.2, -0.2], [0.4, -0.1]],
            ]
        ),
        "r": jnp.asarray(
            [
                [[0.1, -0.2, 0.3], [0.0, 0.2, -0.1], [-0.3, 0.1, 0.2]],
                [[-0.2, 0.1, 0.0], [0.3, -0.1, 0.2], [0.1, 0.0, -0.2]],
            ]
        ),
        "stage_constant": jnp.asarray([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]]),
        "Q_terminal": jnp.asarray([[[1.5, 0.2], [0.2, 1.0]], [[0.8, -0.1], [-0.1, 1.7]]]),
        "q_terminal": jnp.asarray([[0.3, -0.2], [-0.1, 0.4]]),
        "terminal_constant": jnp.asarray([0.25, -0.15]),
        "K": jnp.asarray(
            [
                [[0.2, -0.1], [-0.3, 0.4], [0.1, 0.2]],
                [[-0.1, 0.3], [0.2, -0.2], [0.4, 0.1]],
                [[0.3, 0.0], [-0.2, 0.1], [0.1, -0.3]],
            ]
        ),
        "feedforward": jnp.asarray(
            [[0.05, -0.1, 0.2], [-0.2, 0.15, 0.05], [0.1, 0.0, -0.05]]
        ),
    }


def _permute_affine_data(data):
    players = jnp.asarray([1, 0])
    controls = jnp.asarray([2, 0, 1])
    return {
        "A": data["A"],
        "B": data["B"][:, :, controls],
        "dynamics_bias": data["dynamics_bias"],
        "Q": data["Q"][players],
        "R": data["R"][players][:, :, controls][:, :, :, controls],
        "N": data["N"][players][:, :, :, controls],
        "q": data["q"][players],
        "r": data["r"][players][:, :, controls],
        "stage_constant": data["stage_constant"][players],
        "Q_terminal": data["Q_terminal"][players],
        "q_terminal": data["q_terminal"][players],
        "terminal_constant": data["terminal_constant"][players],
        "K": data["K"][:, controls],
        "feedforward": data["feedforward"][:, controls],
    }


def _affine_problem(initial_state, *, permuted=False, failing_policy=False):
    data = _affine_data()
    if permuted:
        data = _permute_affine_data(data)
    state_layout = phx.dynamics.StateLayout((2,))
    input_layout = phx.dynamics.InputLayout((3,), roles="control")

    def transition(context, state, control, args):
        step = context.step_index
        return (
            args["A"][step] @ state
            + args["B"][step] @ control
            + args["dynamics_bias"][step]
        )

    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=state_layout,
        input_layout=input_layout,
        system_id="permuted-affine-game" if permuted else "affine-game",
    )
    dynamics = phx.control.DiscreteControlDynamics(system)
    time_grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.2, 1.7, 4.0]),
        time_id="nonuniform-affine-game",
    )
    partition = (
        phx.control.games.PlayerControlPartition(("right", "left"), (1, 2))
        if permuted
        else phx.control.games.PlayerControlPartition(("left", "right"), (2, 1))
    )
    problem = DeterministicFeedbackGameProblem(
        dynamics,
        time_grid,
        initial_state,
        partition,
        stage_costs=(_player_stage_cost(0), _player_stage_cost(1)),
        terminal_costs=(_player_terminal_cost(0), _player_terminal_cost(1)),
        args=data,
        problem_id="permuted-affine-game" if permuted else "affine-game",
    )

    def feedback_policy(context, state, args):
        step = context.step_index
        control = args["K"][step] @ state + args["feedforward"][step]
        if failing_policy:
            control = jnp.where(
                state[0] < 0.0,
                jnp.full_like(control, jnp.nan),
                control,
            )
        return control

    policy = phx.dynamics.CallableInputPolicy(
        feedback_policy,
        input_layout=input_layout,
        policy_id="permuted-affine-feedback" if permuted else "affine-feedback",
    )
    return problem, policy


def _direct_player_objectives(problem, joint_controls):
    def step(state, step_control):
        step_index, control = step_control
        context = phx.dynamics.DiscreteStepContext(
            problem.time_grid.times[step_index],
            problem.time_grid.times[step_index + 1],
            step_index,
        )
        stage = jnp.stack(
            tuple(
                callback(context, state, control, problem.args)
                for callback in problem.stage_costs
            )
        )
        next_state = problem.dynamics.system.evaluate(
            context, state, problem.args, inputs=control
        )
        return next_state, stage

    terminal_state, stage_costs = jax.lax.scan(
        step,
        problem.initial_state,
        (jnp.arange(problem.time_grid.num_steps), joint_controls),
    )
    terminal = jnp.stack(
        tuple(
            callback(problem.time_grid.times[-1], terminal_state, problem.args)
            for callback in problem.terminal_costs
        )
    )
    return jnp.sum(stage_costs, axis=0) + terminal


def test_affine_lq_callbacks_are_unweighted_and_keep_physical_time_axes():
    problem, policy = _affine_problem(jnp.asarray([0.35, -0.25]))
    evaluation = evaluate_game_policy(problem, policy)
    trajectory = evaluation.trajectory
    expected_stage = jnp.stack(
        tuple(
            jnp.stack(
                tuple(
                    callback(
                        phx.dynamics.DiscreteStepContext(
                            problem.time_grid.times[step],
                            problem.time_grid.times[step + 1],
                            step,
                        ),
                        trajectory.states[step],
                        trajectory.controls[step],
                        problem.args,
                    )
                    for step in range(problem.time_grid.num_steps)
                )
            )
            for callback in problem.stage_costs
        )
    )
    expected_terminal = jnp.stack(
        tuple(
            callback(problem.time_grid.times[-1], trajectory.states[-1], problem.args)
            for callback in problem.terminal_costs
        )
    )

    assert bool(evaluation.successful)
    assert evaluation.stage_cost_semantics == "unweighted-discrete-stage-sum"
    assert trajectory.states.shape == (problem.time_grid.num_steps + 1, 2)
    assert trajectory.controls.shape == (problem.time_grid.num_steps, 3)
    assert evaluation.stage_costs.shape == (2, problem.time_grid.num_steps)
    assert evaluation.terminal_costs.shape == (2,)
    np.testing.assert_allclose(evaluation.stage_costs, expected_stage, rtol=2e-6)
    np.testing.assert_allclose(evaluation.terminal_costs, expected_terminal, rtol=2e-6)
    np.testing.assert_allclose(
        evaluation.total_costs,
        jnp.sum(expected_stage, axis=-1) + expected_terminal,
        rtol=2e-6,
    )
    duration_weighted = (
        jnp.sum(expected_stage * problem.time_grid.durations, axis=-1) + expected_terminal
    )
    assert not np.allclose(np.asarray(evaluation.total_costs), duration_weighted)


def test_owned_adjoint_rows_equal_whole_horizon_complete_objective_gradients():
    problem, policy = _affine_problem(jnp.asarray([0.35, -0.25]))
    evaluation = evaluate_game_policy(problem, policy)
    scaling = ILQGameScaling(
        jnp.ones((2,)),
        jnp.ones((3,)),
        jnp.ones((2,)),
        scaling_id="unit-game-scaling",
    )
    residual = nominal_nash_residual(problem, evaluation, scaling)
    complete_gradient = jax.jacrev(
        lambda controls: _direct_player_objectives(problem, controls)
    )(evaluation.trajectory.controls)
    ownership = jax.nn.one_hot(
        jnp.asarray(problem.partition.control_owner),
        problem.num_players,
        dtype=complete_gradient.dtype,
    )
    expected_owned = jnp.einsum("ptm,mp->tm", complete_gradient, ownership)

    assert bool(residual.valid)
    assert residual.player_costates.shape == (2, problem.time_grid.num_steps + 1, 2)
    assert residual.raw_owned_stationarity.shape == (problem.time_grid.num_steps, 3)
    assert residual.dynamics_defect.shape == (problem.time_grid.num_steps, 2)
    np.testing.assert_allclose(
        residual.raw_owned_stationarity, expected_owned, rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(residual.dynamics_defect, 0.0, atol=2e-7)


def test_player_and_owned_control_permutation_is_equivariant():
    initial = jnp.asarray([0.35, -0.25])
    problem, policy = _affine_problem(initial)
    permuted_problem, permuted_policy = _affine_problem(initial, permuted=True)
    evaluation = evaluate_game_policy(problem, policy)
    permuted_evaluation = evaluate_game_policy(permuted_problem, permuted_policy)
    residual = nominal_nash_residual(
        problem,
        evaluation,
        ILQGameScaling(jnp.ones(2), jnp.ones(3), jnp.ones(2)),
    )
    permuted_residual = nominal_nash_residual(
        permuted_problem,
        permuted_evaluation,
        ILQGameScaling(jnp.ones(2), jnp.ones(3), jnp.ones(2)),
    )
    control_permutation = jnp.asarray([2, 0, 1])
    player_permutation = jnp.asarray([1, 0])

    np.testing.assert_allclose(
        permuted_evaluation.trajectory.states,
        evaluation.trajectory.states,
    )
    np.testing.assert_allclose(
        permuted_evaluation.trajectory.controls,
        evaluation.trajectory.controls[:, control_permutation],
    )
    np.testing.assert_allclose(
        permuted_evaluation.total_costs, evaluation.total_costs[player_permutation]
    )
    np.testing.assert_allclose(
        permuted_residual.player_costates,
        residual.player_costates[player_permutation],
        rtol=3e-5,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        permuted_residual.raw_owned_stationarity,
        residual.raw_owned_stationarity[:, control_permutation],
        rtol=3e-5,
        atol=3e-6,
    )


def test_explicit_dimensionless_scaling_controls_all_reported_norms():
    problem, policy = _affine_problem(jnp.asarray([0.35, -0.25]))
    evaluation = evaluate_game_policy(problem, policy)
    scaling = ILQGameScaling(
        jnp.asarray([2.0, 0.5]),
        jnp.asarray([4.0, 3.0, 0.25]),
        jnp.asarray([8.0, 0.5]),
        scaling_id="physical-game-scales",
    )
    residual = nominal_nash_residual(problem, evaluation, scaling)
    owners = jnp.asarray(problem.partition.control_owner)
    expected_stationarity = (
        residual.raw_owned_stationarity
        * scaling.control_scales
        / scaling.cost_scales[owners]
    )
    expected_defect = residual.dynamics_defect / scaling.state_scales
    combined = jnp.concatenate(
        (expected_stationarity.reshape(-1), expected_defect.reshape(-1))
    )

    np.testing.assert_allclose(
        residual.dimensionless_owned_stationarity, expected_stationarity
    )
    np.testing.assert_allclose(
        residual.dimensionless_dynamics_defect,
        expected_defect,
    )
    np.testing.assert_allclose(
        residual.stationarity_rms_norm,
        jnp.sqrt(jnp.mean(jnp.square(expected_stationarity))),
    )
    np.testing.assert_allclose(
        residual.stationarity_infinity_norm, jnp.max(jnp.abs(expected_stationarity))
    )
    np.testing.assert_allclose(
        residual.rms_norm,
        jnp.sqrt(jnp.mean(jnp.square(combined))),
    )
    np.testing.assert_allclose(
        residual.infinity_norm,
        jnp.max(jnp.abs(combined)),
    )


def test_mixed_case_failure_is_local_and_preserves_the_first_cause():
    problem, policy = _affine_problem(
        jnp.asarray([[0.35, -0.25], [-0.2, 0.1]]), failing_policy=True
    )
    evaluation = evaluate_game_policy(problem, policy)

    np.testing.assert_array_equal(evaluation.valid, [True, False])
    np.testing.assert_array_equal(
        evaluation.status,
        [
            int(GamePolicyEvaluationStatus.SUCCESS),
            int(GamePolicyEvaluationStatus.NONFINITE_POLICY_CONTROL),
        ],
    )
    np.testing.assert_array_equal(evaluation.first_failed_step, [-1, 0])
    np.testing.assert_array_equal(evaluation.first_failed_player, [-1, -1])
    np.testing.assert_array_equal(
        evaluation.trajectory.valid[0],
        [True, True, True, True],
    )
    np.testing.assert_array_equal(
        evaluation.trajectory.valid[1],
        [True, False, False, False],
    )
    assert np.all(np.isfinite(np.asarray(evaluation.total_costs[0])))
    assert np.all(np.isnan(np.asarray(evaluation.total_costs[1])))


def test_certificate_is_only_local_nominal_stationarity_evidence():
    problem, policy = _affine_problem(jnp.asarray([0.35, -0.25]))
    evaluation = evaluate_game_policy(problem, policy)
    residual = nominal_nash_residual(
        problem,
        evaluation,
        ILQGameScaling(jnp.ones(2), jnp.ones(3), jnp.ones(2)),
    )

    assert residual.certificate == "LOCAL_NOMINAL_NASH_STATIONARY"
    assert "equilibrium" not in (GamePolicyEvaluation.__doc__ or "").lower()
    assert "equilibrium" not in evaluation.evaluation_id.lower()
    assert "equilibrium" not in evaluation.method_id.lower()
