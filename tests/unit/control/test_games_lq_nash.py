#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.control.games import (
    finite_horizon_lq_feedback_nash,
    LQFeedbackNashStatus,
    PlayerControlPartition,
)


def _rational_game(beta=1.0):
    partition = PlayerControlPartition(("player-1", "player-2"), (1, 1))
    time_grid = phx.dynamics.TimeGrid(jnp.asarray([0.0, 1.0]), time_id="rational-game")
    return {
        "dynamics_matrices": jnp.asarray([[[1.0]]]),
        "control_matrices": jnp.asarray([[[beta, 1.0]]]),
        "state_costs": jnp.asarray([[[[2.0]]], [[[-1.0]]]]),
        "control_costs": jnp.asarray(
            [
                [[[1.0, -0.5], [-0.5, 2.0]]],
                [[[2.0, -3.0], [-3.0, 1.0]]],
            ]
        ),
        "terminal_state_costs": jnp.asarray([[[1.0]], [[2.0]]]),
        "partition": partition,
        "dynamics_bias": jnp.asarray([[1.0]]),
        "state_control_cross": jnp.asarray([[[[1.0, -1.0]]], [[[0.5, -0.5]]]]),
        "state_linear": jnp.asarray([[[1.0]], [[-2.0]]]),
        "control_linear": jnp.asarray([[[0.5, -1.0]], [[0.25, 1.0]]]),
        "stage_constants": jnp.asarray([[3.0], [-1.0]]),
        "terminal_linear": jnp.asarray([[0.5], [-1.0]]),
        "terminal_constants": jnp.asarray([0.25, 2.0]),
        "time_grid": time_grid,
    }


def _solve_rational(beta=1.0, values=None):
    values = _rational_game(beta) if values is None else values
    control_matrices = values["control_matrices"].at[0, 0, 0].set(beta)
    return finite_horizon_lq_feedback_nash(
        values["dynamics_matrices"],
        control_matrices,
        values["state_costs"],
        values["control_costs"],
        values["terminal_state_costs"],
        values["partition"],
        dynamics_bias=values["dynamics_bias"],
        state_control_cross=values["state_control_cross"],
        state_linear=values["state_linear"],
        control_linear=values["control_linear"],
        stage_constants=values["stage_constants"],
        terminal_linear=values["terminal_linear"],
        terminal_constants=values["terminal_constants"],
        time_grid=values["time_grid"],
    )


def _stage_cost(values, player, state, control, step):
    return (
        0.5 * state @ values["state_costs"][player, step] @ state
        + state @ values["state_control_cross"][player, step] @ control
        + 0.5 * control @ values["control_costs"][player, step] @ control
        + values["state_linear"][player, step] @ state
        + values["control_linear"][player, step] @ control
        + values["stage_constants"][player, step]
    )


def _terminal_cost(values, player, state):
    return (
        0.5 * state @ values["terminal_state_costs"][player] @ state
        + values["terminal_linear"][player] @ state
        + values["terminal_constants"][player]
    )


def test_player_control_partition_round_trips_leading_axes_and_rejects_invalid_data():
    partition = PlayerControlPartition(("pursuer", "evader"), (2, 1))
    controls = jnp.arange(24.0).reshape(2, 4, 3)
    pursuer, evader = partition.split_controls(controls)

    assert partition.num_players == 2
    assert partition.joint_control_size == 3
    assert partition.control_slices == ((0, 2), (2, 3))
    assert pursuer.shape == (2, 4, 2)
    assert evader.shape == (2, 4, 1)
    np.testing.assert_array_equal(partition.join_controls((pursuer, evader)), controls)

    gain = jnp.arange(30.0).reshape(5, 3, 2)
    split_gain = partition.split_feedback_gain(gain)
    assert split_gain[0].shape == (5, 2, 2)
    assert split_gain[1].shape == (5, 1, 2)

    with pytest.raises(ValueError, match="at least one"):
        PlayerControlPartition((), ())
    with pytest.raises(ValueError, match="unique"):
        PlayerControlPartition(("same", "same"), (1, 1))
    with pytest.raises(ValueError, match="positive"):
        PlayerControlPartition(("player",), (0,))
    with pytest.raises(ValueError, match="one size per player"):
        PlayerControlPartition(("left", "right"), (1,))
    with pytest.raises(ValueError, match="trailing shape"):
        partition.split_controls(jnp.zeros((2,)))
    with pytest.raises(ValueError, match="share leading axes"):
        partition.join_controls((jnp.zeros((2, 2)), jnp.zeros((3, 1))))


def test_rational_affine_two_player_game_matches_closed_form_feedback_and_values():
    result = _solve_rational()

    assert bool(result.valid)
    assert int(result.status) == int(LQFeedbackNashStatus.SUCCESS)
    np.testing.assert_allclose(
        result.feedback_gain[..., 0],
        [[-21.0 / 26.0, -10.0 / 13.0]],
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        result.feedforward,
        [[-10.0 / 13.0, -12.0 / 13.0]],
        rtol=2e-13,
        atol=2e-13,
    )
    expected_values = (
        (1173.0 / 338.0, 1015.0 / 338.0, 3025.0 / 676.0),
        (-745.0 / 338.0, -4837.0 / 1352.0, -19.0 / 338.0),
    )
    for value, (matrix, linear, constant) in zip(
        result.values,
        expected_values,
        strict=True,
    ):
        np.testing.assert_allclose(value.matrices[0, 0, 0], matrix, rtol=2e-13)
        np.testing.assert_allclose(value.linear[0, 0], linear, rtol=2e-13)
        np.testing.assert_allclose(value.constants[0], constant, rtol=2e-13)
    np.testing.assert_allclose(
        result.diagnostics.own_control_minimum_eigenvalues,
        [[2.0], [3.0]],
        rtol=2e-13,
    )
    np.testing.assert_array_equal(result.diagnostics.coupled_ranks, [2])
    assert result.diagnostics.linear_method == "dense-lu"
    assert result.diagnostics.linear_backend == "jax-dense"
    assert result.diagnostics.maximum_stationarity_residual < 1e-13
    assert result.diagnostics.maximum_bellman_residual < 1e-13


def test_feedback_and_feedforward_have_exact_jitted_gradients_through_coupled_solve():
    values = _rational_game()

    def feedback(beta):
        return _solve_rational(beta, values).feedback_gain[0, 0, 0]

    def feedforward(beta):
        return _solve_rational(beta, values).feedforward[0, 0]

    feedback_gradient = jax.jit(jax.grad(feedback))(jnp.asarray(1.0))
    feedforward_gradient = jax.jit(jax.grad(feedforward))(jnp.asarray(1.0))
    np.testing.assert_allclose(feedback_gradient, 87.0 / 169.0, rtol=2e-11)
    np.testing.assert_allclose(feedforward_gradient, 55.0 / 169.0, rtol=2e-11)


def test_one_player_game_matches_finite_horizon_lqr_on_shared_affine_domain():
    horizon = 3
    time_grid = phx.dynamics.TimeGrid(
        jnp.arange(horizon + 1, dtype=float),
        time_id="one-player-reduction",
    )
    a = jnp.asarray([[[1.0]], [[0.9]], [[1.1]]])
    b = jnp.asarray([[[0.8]], [[1.0]], [[0.7]]])
    q = jnp.asarray([[[1.2]], [[0.8]], [[1.4]]])
    r = jnp.asarray([[[2.0]], [[1.6]], [[2.4]]])
    qf = jnp.asarray([[2.5]])
    c = jnp.asarray([[0.2], [-0.1], [0.3]])
    cross = jnp.asarray([[[0.1]], [[-0.05]], [[0.08]]])
    q_linear = jnp.asarray([[0.3], [-0.2], [0.1]])
    r_linear = jnp.asarray([[0.2], [0.1], [-0.1]])
    constants = jnp.asarray([0.5, -0.2, 0.4])
    qf_linear = jnp.asarray([0.25])

    game = finite_horizon_lq_feedback_nash(
        a,
        b,
        q[None, ...],
        r[None, ...],
        qf[None, ...],
        PlayerControlPartition(("controller",), (1,)),
        dynamics_bias=c,
        state_control_cross=cross[None, ...],
        state_linear=q_linear[None, ...],
        control_linear=r_linear[None, ...],
        stage_constants=constants[None, ...],
        terminal_linear=qf_linear[None, ...],
        terminal_constants=jnp.asarray([0.7]),
        time_grid=time_grid,
    )
    lqr = phx.control.finite_horizon_lqr(
        a,
        b,
        q,
        r,
        qf,
        dynamics_bias=c,
        state_control_cross=cross,
        state_linear=q_linear,
        control_linear=r_linear,
        stage_constants=constants,
        terminal_linear=qf_linear,
        terminal_constant=0.7,
        time_grid=time_grid,
    )

    assert bool(game.valid)
    np.testing.assert_allclose(
        game.feedback_gain, lqr.feedback_gain, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(game.feedforward, lqr.feedforward, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        game.values[0].matrices, lqr.value.matrices, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        game.values[0].linear, lqr.value.linear, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        game.values[0].constants, lqr.value.constants, rtol=2e-12, atol=2e-12
    )


def test_multistage_policy_satisfies_unilateral_bellman_conditions_and_rolls_out():
    horizon = 3
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    time_grid = phx.dynamics.TimeGrid(
        jnp.arange(horizon + 1, dtype=float),
        time_id="multistage-game",
    )
    a = jnp.asarray(
        [
            [[0.9, 0.1], [0.0, 0.8]],
            [[0.8, -0.1], [0.1, 0.85]],
            [[0.95, 0.0], [0.05, 0.9]],
        ]
    )
    b = jnp.asarray(
        [
            [[0.5, 0.1], [0.0, 0.4]],
            [[0.4, 0.0], [0.1, 0.5]],
            [[0.45, 0.05], [0.05, 0.35]],
        ]
    )
    c = jnp.asarray([[0.1, -0.05], [0.0, 0.08], [-0.04, 0.02]])
    q = jnp.stack(
        (
            jnp.broadcast_to(jnp.diag(jnp.asarray([1.2, 0.6])), (horizon, 2, 2)),
            jnp.broadcast_to(jnp.diag(jnp.asarray([0.5, 1.4])), (horizon, 2, 2)),
        )
    )
    r = jnp.stack(
        (
            jnp.broadcast_to(jnp.asarray([[2.0, 0.2], [0.2, 1.5]]), (horizon, 2, 2)),
            jnp.broadcast_to(jnp.asarray([[1.5, -0.1], [-0.1, 2.2]]), (horizon, 2, 2)),
        )
    )
    cross = jnp.asarray(
        [
            [[[0.08, -0.02], [0.01, 0.04]]] * horizon,
            [[[-0.03, 0.05], [0.02, -0.04]]] * horizon,
        ]
    )
    qf = jnp.asarray([[[1.5, 0.1], [0.1, 0.8]], [[0.7, -0.05], [-0.05, 1.8]]])
    q_linear = jnp.asarray([[[0.1, -0.05]] * horizon, [[-0.08, 0.12]] * horizon])
    r_linear = jnp.asarray([[[0.04, -0.02]] * horizon, [[-0.03, 0.05]] * horizon])
    constants = jnp.asarray([[0.2, -0.1, 0.3], [-0.2, 0.15, 0.05]])
    qf_linear = jnp.asarray([[0.1, -0.2], [-0.05, 0.15]])
    terminal_constants = jnp.asarray([0.4, -0.25])
    values = {
        "dynamics_matrices": a,
        "control_matrices": b,
        "state_costs": q,
        "control_costs": r,
        "terminal_state_costs": qf,
        "partition": partition,
        "dynamics_bias": c,
        "state_control_cross": cross,
        "state_linear": q_linear,
        "control_linear": r_linear,
        "stage_constants": constants,
        "terminal_linear": qf_linear,
        "terminal_constants": terminal_constants,
        "time_grid": time_grid,
    }
    result = finite_horizon_lq_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        dynamics_bias=c,
        state_control_cross=cross,
        state_linear=q_linear,
        control_linear=r_linear,
        stage_constants=constants,
        terminal_linear=qf_linear,
        terminal_constants=terminal_constants,
        time_grid=time_grid,
    )
    assert bool(result.valid)

    probe_states = (
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.4, -0.3]),
        jnp.asarray([-0.6, 0.2]),
    )
    for step, state in enumerate(probe_states):
        control = result.feedback_gain[step] @ state + result.feedforward[step]
        next_state = a[step] @ state + b[step] @ control + c[step]
        for player, (start, stop) in enumerate(partition.control_slices):

            def unilateral(
                local_control,
                *,
                control=control,
                player=player,
                start=start,
                state=state,
                step=step,
                stop=stop,
            ):
                joint = control.at[start:stop].set(local_control)
                following = a[step] @ state + b[step] @ joint + c[step]
                return _stage_cost(values, player, state, joint, step) + result.values[
                    player
                ](time_grid.times[step + 1], following)

            local = control[start:stop]
            gradient = jax.grad(unilateral)(local)
            hessian = jax.hessian(unilateral)(local)
            np.testing.assert_allclose(gradient, 0.0, atol=2e-10)
            assert np.linalg.eigvalsh(np.asarray(hessian)).min() > 0.0
            bellman = _stage_cost(values, player, state, control, step) + result.values[
                player
            ](time_grid.times[step + 1], next_state)
            np.testing.assert_allclose(
                result.values[player](time_grid.times[step], state),
                bellman,
                rtol=2e-11,
                atol=2e-11,
            )
    for player in range(partition.num_players):
        terminal_state = jnp.asarray([0.3, -0.4])
        np.testing.assert_allclose(
            result.values[player](time_grid.times[-1], terminal_state),
            _terminal_cost(values, player, terminal_state),
            rtol=2e-13,
            atol=2e-13,
        )

    def transition(context, state, control, args):
        step = context.step_index
        return args["A"][step] @ state + args["B"][step] @ control + args["c"][step]

    dynamics = phx.control.DiscreteControlDynamics(
        phx.dynamics.DiscreteSystem(
            transition,
            state_layout=phx.dynamics.StateLayout((2,)),
            input_layout=phx.dynamics.InputLayout((2,), roles="control"),
            system_id="multistage-lq-game",
        )
    )
    initial_state = jnp.asarray([0.35, -0.25])
    problem = phx.control.ControlProblem(
        dynamics,
        time_grid,
        initial_state,
        args={"A": a, "B": b, "c": c},
        problem_id="multistage-lq-game",
    )
    evaluation = problem.evaluate(result.policy, jnp.zeros(()))
    assert bool(evaluation.successful)
    trajectory = evaluation.trajectory
    np.testing.assert_allclose(
        trajectory.states[1:],
        jax.vmap(lambda A, B, c_, x, u: A @ x + B @ u + c_)(
            a,
            b,
            c,
            trajectory.states[:-1],
            trajectory.controls,
        ),
        rtol=2e-13,
        atol=2e-13,
    )
    split = partition.split_controls(trajectory.controls)
    np.testing.assert_allclose(partition.join_controls(split), trajectory.controls)
    for player in range(partition.num_players):
        direct = sum(
            _stage_cost(
                values,
                player,
                trajectory.states[step],
                trajectory.controls[step],
                step,
            )
            for step in range(horizon)
        ) + _terminal_cost(values, player, trajectory.states[-1])
        np.testing.assert_allclose(
            result.values[player](time_grid.times[0], initial_state),
            direct,
            rtol=2e-11,
            atol=2e-11,
        )


def test_player_permutation_preserves_physical_policy_and_permutes_values():
    values = _rational_game()
    baseline = _solve_rational()
    control_permutation = jnp.asarray([1, 0])
    player_permutation = jnp.asarray([1, 0])
    permuted = finite_horizon_lq_feedback_nash(
        values["dynamics_matrices"],
        values["control_matrices"][..., control_permutation],
        values["state_costs"][player_permutation],
        values["control_costs"][player_permutation][..., control_permutation, :][
            ..., control_permutation
        ],
        values["terminal_state_costs"][player_permutation],
        PlayerControlPartition(("player-2", "player-1"), (1, 1)),
        dynamics_bias=values["dynamics_bias"],
        state_control_cross=values["state_control_cross"][player_permutation][
            ..., control_permutation
        ],
        state_linear=values["state_linear"][player_permutation],
        control_linear=values["control_linear"][player_permutation][
            ..., control_permutation
        ],
        stage_constants=values["stage_constants"][player_permutation],
        terminal_linear=values["terminal_linear"][player_permutation],
        terminal_constants=values["terminal_constants"][player_permutation],
        time_grid=values["time_grid"],
    )

    assert bool(permuted.valid)
    np.testing.assert_allclose(
        permuted.feedback_gain[..., control_permutation, :],
        baseline.feedback_gain,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        permuted.feedforward[..., control_permutation],
        baseline.feedforward,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(permuted.values[1].matrices, baseline.values[0].matrices)
    np.testing.assert_allclose(permuted.values[0].matrices, baseline.values[1].matrices)


def test_jitted_mixed_cases_report_independent_numeric_failures():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    time_grid = phx.dynamics.TimeGrid(jnp.asarray([0.0, 1.0]), time_id="mixed-cases")
    cases = 5
    a = jnp.ones((cases, 1, 1, 1))
    a = a.at[4, 0, 0, 0].set(jnp.nan)
    b = jnp.zeros((cases, 1, 1, 2))
    q = jnp.zeros((cases, 2, 1, 1, 1))
    qf = jnp.zeros((cases, 2, 1, 1))
    identity_costs = jnp.broadcast_to(jnp.eye(2), (2, 1, 2, 2))
    r = jnp.broadcast_to(identity_costs, (cases, 2, 1, 2, 2))
    r = r.at[1, 0, 0].set(jnp.asarray([[1.0, 1.0], [0.0, 1.0]]))
    r = r.at[2, 0, 0].set(jnp.asarray([[1.0, 1.0], [1.0, 2.0]]))
    r = r.at[2, 1, 0].set(jnp.asarray([[2.0, 1.0], [1.0, 1.0]]))
    r = r.at[3, 0, 0].set(jnp.asarray([[-1.0, 0.0], [0.0, 1.0]]))

    solve = eqx.filter_jit(finite_horizon_lq_feedback_nash)
    result = solve(a, b, q, r, qf, partition, time_grid=time_grid)
    expected = jnp.asarray(
        [
            LQFeedbackNashStatus.SUCCESS,
            LQFeedbackNashStatus.NONSYMMETRIC_COST,
            LQFeedbackNashStatus.COUPLED_SYSTEM_RANK_DEFICIENT,
            LQFeedbackNashStatus.OWN_CURVATURE_NOT_POSITIVE_DEFINITE,
            LQFeedbackNashStatus.NONFINITE_INPUT,
        ],
        dtype=jnp.int32,
    )
    np.testing.assert_array_equal(result.status, expected)
    np.testing.assert_array_equal(result.diagnostics.first_failed_stage, [-1, 0, 0, 0, 0])
    assert bool(result.valid[0])
    assert not bool(jnp.any(result.valid[1:]))
    assert bool(result.diagnostics.diagnostic_available[1, 0])
    assert bool(result.diagnostics.diagnostic_available[2, 0])
    assert bool(result.diagnostics.diagnostic_available[3, 0])
    assert not bool(result.diagnostics.diagnostic_available[4, 0])
    assert np.all(np.isfinite(np.asarray(result.feedback_gain[0])))
    assert np.all(np.isfinite(np.asarray(result.feedback_gain[3])))
    assert np.any(~np.isfinite(np.asarray(result.feedback_gain[2])))

    valid = finite_horizon_lq_feedback_nash(
        a[0],
        b[0],
        q[0],
        r[0],
        qf[0],
        partition,
        time_grid=time_grid,
    )
    np.testing.assert_allclose(result.feedback_gain[0], valid.feedback_gain)


def test_rank_cutoff_and_condition_limit_are_distinct_from_lu_success():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    time_grid = phx.dynamics.TimeGrid(jnp.asarray([0.0, 1.0]), time_id="rank-policy")
    delta = 2.0**-20
    a = jnp.ones((1, 1, 1))
    b = jnp.zeros((1, 1, 2))
    q = jnp.zeros((2, 1, 1, 1))
    qf = jnp.zeros((2, 1, 1))
    r = jnp.asarray(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 0.0], [0.0, delta]]],
        ]
    )

    rank_rejected = finite_horizon_lq_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        time_grid=time_grid,
        rank_relative_tolerance=1e-4,
    )
    accepted = finite_horizon_lq_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        time_grid=time_grid,
        rank_relative_tolerance=1e-12,
    )
    condition_rejected = finite_horizon_lq_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        time_grid=time_grid,
        rank_relative_tolerance=1e-12,
        maximum_condition=1e5,
    )

    assert int(rank_rejected.status) == int(
        LQFeedbackNashStatus.COUPLED_SYSTEM_RANK_DEFICIENT
    )
    assert bool(rank_rejected.diagnostics.finite)
    np.testing.assert_array_equal(
        rank_rejected.diagnostics.linear_status,
        [[int(phx.linalg.LinearSolveStatus.SUCCESS)] * 2],
    )
    assert bool(accepted.valid)
    assert int(condition_rejected.status) == int(
        LQFeedbackNashStatus.CONDITION_LIMIT_REACHED
    )
    np.testing.assert_allclose(
        accepted.diagnostics.coupled_condition_numbers, [1.0 / delta]
    )


def test_structural_validation_rejects_shape_dtype_grid_and_tolerance_errors():
    values = _rational_game()
    required = (
        values["dynamics_matrices"],
        values["control_matrices"],
        values["state_costs"],
        values["control_costs"],
        values["terminal_state_costs"],
        values["partition"],
    )
    with pytest.raises(TypeError, match="real-valued"):
        finite_horizon_lq_feedback_nash(
            values["dynamics_matrices"].astype(complex),
            *required[1:],
            time_grid=values["time_grid"],
        )
    with pytest.raises(ValueError, match="partition joint control size"):
        finite_horizon_lq_feedback_nash(
            required[0],
            required[1][..., :1],
            *required[2:],
            time_grid=values["time_grid"],
        )
    with pytest.raises(ValueError, match="state_costs must have shape"):
        finite_horizon_lq_feedback_nash(
            required[0],
            required[1],
            required[2][:1],
            *required[3:],
            time_grid=values["time_grid"],
        )
    wrong_grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.5, 1.0]),
        time_id="wrong-game-grid",
    )
    with pytest.raises(ValueError, match="must contain 2 times"):
        finite_horizon_lq_feedback_nash(*required, time_grid=wrong_grid)
    with pytest.raises(ValueError, match="tolerance must be finite and positive"):
        finite_horizon_lq_feedback_nash(
            *required,
            time_grid=values["time_grid"],
            tolerance=0.0,
        )
    with pytest.raises(ValueError, match="maximum_condition"):
        finite_horizon_lq_feedback_nash(
            *required,
            time_grid=values["time_grid"],
            maximum_condition=1.0,
        )


def test_reverse_failures_preserve_causal_stage_and_terminal_index():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    time_grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0, 2.0]),
        time_id="causal-game-failure",
    )
    a = jnp.ones((2, 1, 1))
    b = jnp.zeros((2, 1, 2))
    q = jnp.zeros((2, 2, 1, 1))
    qf = jnp.zeros((2, 1, 1))
    identity = jnp.eye(2)
    r = jnp.broadcast_to(identity, (2, 2, 2, 2))
    r = r.at[0, 1].set(jnp.asarray([[1.0, 1.0], [1.0, 2.0]]))
    r = r.at[1, 1].set(jnp.asarray([[2.0, 1.0], [1.0, 1.0]]))
    stage_failure = finite_horizon_lq_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        time_grid=time_grid,
    )

    assert int(stage_failure.status) == int(
        LQFeedbackNashStatus.COUPLED_SYSTEM_RANK_DEFICIENT
    )
    assert int(stage_failure.diagnostics.first_failed_stage) == 1
    np.testing.assert_array_equal(
        stage_failure.diagnostics.stage_status,
        [
            int(LQFeedbackNashStatus.DEPENDENCY_FAILED),
            int(LQFeedbackNashStatus.COUPLED_SYSTEM_RANK_DEFICIENT),
        ],
    )
    np.testing.assert_array_equal(
        stage_failure.diagnostics.diagnostic_available,
        [False, True],
    )

    terminal_grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0]),
        time_id="terminal-game-failure",
    )
    terminal_failure = finite_horizon_lq_feedback_nash(
        jnp.eye(2)[None, ...],
        jnp.ones((1, 2, 1)),
        jnp.zeros((1, 1, 2, 2)),
        jnp.ones((1, 1, 1, 1)),
        jnp.asarray([[[1.0, 1.0], [0.0, 1.0]]]),
        PlayerControlPartition(("controller",), (1,)),
        time_grid=terminal_grid,
    )
    assert int(terminal_failure.status) == int(LQFeedbackNashStatus.NONSYMMETRIC_COST)
    assert int(terminal_failure.diagnostics.first_failed_stage) == 1
    assert int(terminal_failure.diagnostics.stage_status[0]) == int(
        LQFeedbackNashStatus.DEPENDENCY_FAILED
    )
    assert not bool(terminal_failure.diagnostics.diagnostic_available[0])


def test_public_surface_is_namespaced_under_control_games():
    assert phx.control.games.__all__ == [
        "FiniteHorizonLQFeedbackNashDiagnostics",
        "FiniteHorizonLQFeedbackNashResult",
        "LQFeedbackNashStatus",
        "PlayerControlPartition",
        "finite_horizon_lq_feedback_nash",
    ]
    assert "games" in phx.control.__all__
    assert "finite_horizon_lq_feedback_nash" not in phx.control.__all__
