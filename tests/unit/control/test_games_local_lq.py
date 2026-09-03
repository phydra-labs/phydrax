#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.control.games import (
    DeterministicFeedbackGameProblem,
    evaluate_game_policy,
    finite_horizon_lq_feedback_nash,
    ILQGameScaling,
    LQFeedbackNashStatus,
    PlayerControlPartition,
)
from phydrax.control.games._local_lq import (
    LocalAffineGameSuggestionStatus,
    suggest_local_affine_game_policy,
)


def _stage_cost(player):
    def cost(context, state, control, args):
        step = context.step_index
        cross = args["beta"] * args["N"][player, step]
        return (
            0.5 * state @ args["Q"][player, step] @ state
            + state @ cross @ control
            + 0.5 * control @ args["R"][player, step] @ control
            + args["q"][player, step] @ state
            + args["r"][player, step] @ control
            + args["stage_constants"][player, step]
        )

    return cost


def _terminal_cost(player):
    def cost(time, state, args):
        del time
        return (
            0.5 * state @ args["terminal_Q"][player] @ state
            + args["terminal_q"][player] @ state
            + args["terminal_constants"][player]
        )

    return cost


def _affine_data():
    return {
        "A": jnp.asarray(
            [
                [[1.0, 0.2], [-0.1, 0.9]],
                [[0.85, -0.15], [0.25, 1.05]],
            ]
        ),
        "B": jnp.asarray(
            [
                [[0.4, -0.2], [0.3, 0.5]],
                [[0.2, 0.6], [-0.4, 0.1]],
            ]
        ),
        "dynamics_bias": jnp.asarray([[0.05, -0.02], [-0.04, 0.03]]),
        "Q": jnp.asarray(
            [
                [
                    [[1.2, 0.1], [0.1, 0.8]],
                    [[0.9, -0.2], [-0.2, 1.4]],
                ],
                [
                    [[0.6, -0.1], [-0.1, 1.0]],
                    [[1.3, 0.2], [0.2, 0.5]],
                ],
            ]
        ),
        "R": jnp.asarray(
            [
                [
                    [[2.0, 0.1], [0.1, 0.8]],
                    [[1.7, -0.1], [-0.1, 0.9]],
                ],
                [
                    [[1.1, -0.2], [-0.2, 1.7]],
                    [[0.9, 0.15], [0.15, 2.1]],
                ],
            ]
        ),
        "N": jnp.asarray(
            [
                [
                    [[0.1, -0.2], [0.3, 0.05]],
                    [[-0.1, 0.2], [0.0, 0.1]],
                ],
                [
                    [[-0.2, 0.1], [0.1, -0.1]],
                    [[0.0, -0.2], [0.2, 0.1]],
                ],
            ]
        ),
        "q": jnp.asarray(
            [
                [[0.2, -0.1], [-0.3, 0.4]],
                [[-0.1, 0.3], [0.2, -0.2]],
            ]
        ),
        "r": jnp.asarray(
            [
                [[0.1, -0.2], [0.0, 0.2]],
                [[-0.2, 0.1], [0.3, -0.1]],
            ]
        ),
        "stage_constants": jnp.asarray([[0.2, -0.1], [-0.2, 0.4]]),
        "terminal_Q": jnp.asarray([[[1.5, 0.2], [0.2, 1.0]], [[0.8, -0.1], [-0.1, 1.7]]]),
        "terminal_q": jnp.asarray([[0.3, -0.2], [-0.1, 0.4]]),
        "terminal_constants": jnp.asarray([0.25, -0.15]),
        "nominal_K": jnp.asarray(
            [
                [[0.2, -0.1], [-0.3, 0.4]],
                [[-0.1, 0.3], [0.2, -0.2]],
            ]
        ),
        "nominal_k": jnp.asarray([[0.05, -0.1], [-0.2, 0.15]]),
        "beta": jnp.asarray(1.0),
    }


def _permute_data(data):
    players = jnp.asarray([1, 0])
    controls = jnp.asarray([1, 0])
    return {
        "A": data["A"],
        "B": data["B"][:, :, controls],
        "dynamics_bias": data["dynamics_bias"],
        "Q": data["Q"][players],
        "R": data["R"][players][:, :, controls][:, :, :, controls],
        "N": data["N"][players][:, :, :, controls],
        "q": data["q"][players],
        "r": data["r"][players][:, :, controls],
        "stage_constants": data["stage_constants"][players],
        "terminal_Q": data["terminal_Q"][players],
        "terminal_q": data["terminal_q"][players],
        "terminal_constants": data["terminal_constants"][players],
        "nominal_K": data["nominal_K"][:, controls],
        "nominal_k": data["nominal_k"][:, controls],
        "beta": data["beta"],
    }


def _affine_problem(*, permuted=False):
    data = _affine_data()
    if permuted:
        data = _permute_data(data)
    input_layout = phx.dynamics.InputLayout((2,), roles="control")

    def transition(context, state, control, args):
        step = context.step_index
        return (
            args["A"][step] @ state
            + args["B"][step] @ control
            + args["dynamics_bias"][step]
        )

    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((2,)),
        input_layout=input_layout,
        system_id="permuted-local-affine" if permuted else "local-affine",
    )
    dynamics = phx.control.DiscreteControlDynamics(system)
    time_grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.3, 1.4]),
        time_id="local-affine-grid",
    )
    partition = (
        PlayerControlPartition(("right", "left"), (1, 1))
        if permuted
        else PlayerControlPartition(("left", "right"), (1, 1))
    )
    problem = DeterministicFeedbackGameProblem(
        dynamics,
        time_grid,
        jnp.asarray([0.35, -0.25]),
        partition,
        stage_costs=(_stage_cost(0), _stage_cost(1)),
        terminal_costs=(_terminal_cost(0), _terminal_cost(1)),
        args=data,
        problem_id="permuted-local-affine" if permuted else "local-affine",
    )

    def nominal_policy(context, state, args):
        step = context.step_index
        return args["nominal_K"][step] @ state + args["nominal_k"][step]

    policy = phx.dynamics.CallableInputPolicy(
        nominal_policy,
        input_layout=input_layout,
        policy_id="permuted-nominal" if permuted else "nominal",
    )
    return problem, evaluate_game_policy(problem, policy)


def _scaling(problem, *, permuted=False):
    state = jnp.asarray([2.0, 0.5])
    control = jnp.asarray([3.0, 0.25])
    cost = jnp.asarray([4.0, 0.75])
    if permuted:
        control = control[jnp.asarray([1, 0])]
        cost = cost[jnp.asarray([1, 0])]
    return ILQGameScaling(
        state,
        control,
        cost,
        scaling_id="permuted-local-scales" if permuted else "local-scales",
    )


def _suggest(problem, evaluation, scaling, *, suggestion_id="local-test"):
    return suggest_local_affine_game_policy(
        problem,
        evaluation,
        scaling,
        symmetry_tolerance=2e-6,
        curvature_tolerance=1e-8,
        rank_relative_tolerance=1e-8,
        rank_absolute_tolerance=1e-10,
        maximum_condition=1e8,
        suggestion_id=suggestion_id,
    )


def test_exact_affine_lq_identity_from_a_nonzero_nominal():
    problem, evaluation = _affine_problem()
    suggestion = _suggest(problem, evaluation, _scaling(problem))
    data = problem.args
    exact = finite_horizon_lq_feedback_nash(
        data["A"],
        data["B"],
        data["Q"],
        data["R"],
        data["terminal_Q"],
        problem.partition,
        dynamics_bias=data["dynamics_bias"],
        state_control_cross=data["beta"] * data["N"],
        state_linear=data["q"],
        control_linear=data["r"],
        stage_constants=data["stage_constants"],
        terminal_linear=data["terminal_q"],
        terminal_constants=data["terminal_constants"],
        time_grid=problem.time_grid,
        symmetry_tolerance=2e-6,
        curvature_tolerance=1e-8,
        rank_relative_tolerance=1e-8,
        rank_absolute_tolerance=1e-10,
        maximum_condition=1e8,
    )

    assert bool(suggestion.successful)
    assert suggestion.scope == "LOCAL_QUADRATIC_SUGGESTION"
    assert "equilibrium" not in suggestion.scope.lower()
    np.testing.assert_allclose(suggestion.feedback_gain, exact.feedback_gain, rtol=2e-5)
    np.testing.assert_allclose(
        suggestion.policy.absolute_feedforward,
        exact.feedforward,
        rtol=3e-5,
        atol=3e-6,
    )
    np.testing.assert_allclose(suggestion.model.A, data["A"], rtol=2e-6)
    np.testing.assert_allclose(suggestion.model.B, data["B"], rtol=2e-6)
    np.testing.assert_allclose(suggestion.model.Q, data["Q"], rtol=2e-6)
    np.testing.assert_allclose(suggestion.model.R, data["R"], rtol=2e-6)
    np.testing.assert_allclose(suggestion.model.terminal_Q, data["terminal_Q"], rtol=2e-6)


def test_state_control_cross_derivative_keeps_n_by_m_orientation():
    problem, evaluation = _affine_problem()
    suggestion = _suggest(problem, evaluation, _scaling(problem))

    assert suggestion.model.N.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(
        suggestion.model.N,
        problem.args["beta"] * problem.args["N"],
        rtol=2e-6,
        atol=2e-7,
    )
    state = suggestion.model.nominal_states[0]
    control = suggestion.model.nominal_controls[0]
    context = phx.dynamics.DiscreteStepContext(
        problem.time_grid.times[0], problem.time_grid.times[1], jnp.asarray(0)
    )
    reverse_mixed = jax.jacrev(
        jax.jacrev(
            lambda x, u: jnp.stack(
                tuple(cost(context, x, u, problem.args) for cost in problem.stage_costs)
            ),
            argnums=1,
        ),
        argnums=0,
    )(state, control)
    np.testing.assert_allclose(
        suggestion.model.N[:, 0],
        jnp.swapaxes(reverse_mixed, -1, -2),
        rtol=2e-6,
        atol=2e-7,
    )


def test_dynamics_defect_uses_nominal_next_minus_nonlinear_transition():
    problem, evaluation = _affine_problem()
    displacement = jnp.asarray([0.3, -0.4])
    altered_states = evaluation.trajectory.states.at[1].add(displacement)
    altered_evaluation = eqx.tree_at(
        lambda value: value.trajectory.states,
        evaluation,
        altered_states,
    )
    scaling = _scaling(problem)
    suggestion = _suggest(problem, altered_evaluation, scaling)

    np.testing.assert_allclose(suggestion.model.dynamics_defects[0], displacement)
    np.testing.assert_allclose(suggestion.model.dynamics_bias[0], -displacement)
    np.testing.assert_allclose(
        suggestion.dimensionless_dynamics_defects[0],
        displacement / scaling.state_scales,
    )
    np.testing.assert_allclose(
        suggestion.dynamics_defect_infinity_norm,
        jnp.max(jnp.abs(suggestion.dimensionless_dynamics_defects)),
    )


def test_model_and_policy_preserve_explicit_T_and_T_plus_one_axes():
    problem, evaluation = _affine_problem()
    suggestion = _suggest(problem, evaluation, _scaling(problem))
    model = suggestion.model
    horizon = problem.time_grid.num_steps

    assert model.nominal_states.shape == (horizon + 1, 2)
    assert model.nominal_controls.shape == (horizon, 2)
    assert model.nominal_dynamics.shape == (horizon, 2)
    assert model.dynamics_defects.shape == (horizon, 2)
    assert model.A.shape == (horizon, 2, 2)
    assert model.B.shape == (horizon, 2, 2)
    assert model.q.shape == (2, horizon, 2)
    assert model.r.shape == (2, horizon, 2)
    assert model.Q.shape == (2, horizon, 2, 2)
    assert model.R.shape == (2, horizon, 2, 2)
    assert model.N.shape == (2, horizon, 2, 2)
    assert model.stage_constants.shape == (2, horizon)
    assert model.terminal_q.shape == (2, 2)
    assert model.terminal_Q.shape == (2, 2, 2)
    assert suggestion.policy.feedback_gain.shape == (horizon, 2, 2)
    assert suggestion.policy.feedforward.shape == (horizon, 2)


def test_deviation_policy_converts_to_absolute_control_and_rolls_out_physically():
    problem, evaluation = _affine_problem()
    suggestion = _suggest(problem, evaluation, _scaling(problem))
    policy = suggestion.policy.with_feedforward_scale(
        jnp.asarray(0.25),
        policy_id="quarter-step-local-policy",
    )
    step = 1
    deviation = jnp.asarray([0.12, -0.07])
    state = policy.nominal_states[step] + deviation
    context = phx.dynamics.DiscreteStepContext(
        problem.time_grid.times[step],
        problem.time_grid.times[step + 1],
        jnp.asarray(step),
    )
    expected = (
        policy.nominal_controls[step]
        + policy.feedback_gain[step] @ deviation
        + 0.25 * policy.feedforward[step]
    )

    np.testing.assert_allclose(policy.evaluate_step(context, state), expected)
    np.testing.assert_allclose(
        policy.absolute_feedforward[step],
        policy.nominal_controls[step]
        - policy.feedback_gain[step] @ policy.nominal_states[step]
        + 0.25 * policy.feedforward[step],
    )
    rollout = policy.rollout(problem)
    assert bool(rollout.successful)
    assert rollout.trajectory.states.shape == (problem.time_grid.num_steps + 1, 2)
    assert rollout.trajectory.controls.shape == (problem.time_grid.num_steps, 2)
    assert rollout.trajectory.control_id == "quarter-step-local-policy"


def test_player_and_control_permutation_is_equivariant():
    problem, evaluation = _affine_problem()
    permuted_problem, permuted_evaluation = _affine_problem(permuted=True)
    suggestion = _suggest(problem, evaluation, _scaling(problem), suggestion_id="base")
    permuted = _suggest(
        permuted_problem,
        permuted_evaluation,
        _scaling(permuted_problem, permuted=True),
        suggestion_id="permuted",
    )
    players = jnp.asarray([1, 0])
    controls = jnp.asarray([1, 0])

    np.testing.assert_allclose(
        permuted.model.nominal_states,
        suggestion.model.nominal_states,
        rtol=2e-6,
    )
    np.testing.assert_allclose(permuted.model.q, suggestion.model.q[players], rtol=2e-5)
    np.testing.assert_allclose(permuted.model.Q, suggestion.model.Q[players], rtol=2e-5)
    np.testing.assert_allclose(
        permuted.model.r,
        suggestion.model.r[players][:, :, controls],
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        permuted.model.R,
        suggestion.model.R[players][:, :, controls][:, :, :, controls],
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        permuted.model.N,
        suggestion.model.N[players][:, :, :, controls],
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        permuted.feedback_gain,
        suggestion.feedback_gain[:, controls],
        rtol=3e-5,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        permuted.policy.absolute_feedforward,
        suggestion.policy.absolute_feedforward[:, controls],
        rtol=3e-5,
        atol=3e-6,
    )


def _one_player_problem():
    input_layout = phx.dynamics.InputLayout((1,), roles="control")
    time_grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.4, 1.0]),
        time_id="one-player-local-grid",
    )
    args = {
        "A": jnp.asarray([[[1.0]], [[0.9]]]),
        "B": jnp.asarray([[[0.8]], [[1.1]]]),
        "c": jnp.asarray([[0.2], [-0.1]]),
        "Q": jnp.asarray([[[[1.2]], [[0.8]]]]),
        "R": jnp.asarray([[[[2.0]], [[1.6]]]]),
        "N": jnp.asarray([[[[0.1]], [[-0.05]]]]),
        "q": jnp.asarray([[[0.3], [-0.2]]]),
        "r": jnp.asarray([[[0.2], [0.1]]]),
        "d": jnp.asarray([[0.5, -0.2]]),
        "Qf": jnp.asarray([[[2.5]]]),
        "qf": jnp.asarray([[0.25]]),
        "df": jnp.asarray([0.7]),
    }

    def transition(context, state, control, data):
        step = context.step_index
        return data["A"][step] @ state + data["B"][step] @ control + data["c"][step]

    def stage(context, state, control, data):
        step = context.step_index
        return (
            0.5 * state @ data["Q"][0, step] @ state
            + state @ data["N"][0, step] @ control
            + 0.5 * control @ data["R"][0, step] @ control
            + data["q"][0, step] @ state
            + data["r"][0, step] @ control
            + data["d"][0, step]
        )

    def terminal(time, state, data):
        del time
        return 0.5 * state @ data["Qf"][0] @ state + data["qf"][0] @ state + data["df"][0]

    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=input_layout,
        system_id="one-player-local-system",
    )
    problem = DeterministicFeedbackGameProblem(
        phx.control.DiscreteControlDynamics(system),
        time_grid,
        jnp.asarray([0.4]),
        PlayerControlPartition(("controller",), (1,)),
        stage_costs=(stage,),
        terminal_costs=(terminal,),
        args=args,
        problem_id="one-player-local",
    )
    policy = phx.dynamics.CallableInputPolicy(
        lambda context, state, data: jnp.asarray([0.2 * state[0] - 0.1]),
        input_layout=input_layout,
        policy_id="one-player-nominal",
    )
    return problem, evaluate_game_policy(problem, policy)


def test_one_player_local_game_reduces_to_finite_horizon_lqr():
    problem, evaluation = _one_player_problem()
    suggestion = _suggest(
        problem,
        evaluation,
        ILQGameScaling(jnp.ones(1), jnp.ones(1), jnp.ones(1)),
        suggestion_id="one-player",
    )
    model = suggestion.model
    lqr = phx.control.finite_horizon_lqr(
        model.A,
        model.B,
        model.Q[0],
        model.R[0],
        model.terminal_Q[0],
        dynamics_bias=model.dynamics_bias,
        state_control_cross=model.N[0],
        state_linear=model.q[0],
        control_linear=model.r[0],
        stage_constants=model.stage_constants[0],
        terminal_linear=model.terminal_q[0],
        terminal_constant=model.terminal_constants[0],
        time_grid=problem.time_grid,
        cost_tolerance=2e-6,
    )

    assert bool(suggestion.successful)
    np.testing.assert_allclose(suggestion.feedback_gain, lqr.feedback_gain, rtol=2e-5)
    np.testing.assert_allclose(suggestion.feedforward, lqr.feedforward, rtol=2e-5)
    np.testing.assert_allclose(
        suggestion.lq_result.values[0].matrices,
        lqr.value.matrices,
        rtol=2e-5,
    )


def test_exact_derivative_blocks_are_jittable_and_differentiable():
    problem, evaluation = _affine_problem()
    scaling = _scaling(problem)

    def blocks(beta):
        varied = eqx.tree_at(lambda value: value.args["beta"], problem, beta)
        model = _suggest(
            varied,
            evaluation,
            scaling,
            suggestion_id="jitted-blocks",
        ).model
        return model.A, model.B, model.Q, model.R, model.N

    A, B, Q, R, N = jax.jit(blocks)(jnp.asarray(1.3))
    np.testing.assert_allclose(A, problem.args["A"], rtol=2e-6)
    np.testing.assert_allclose(B, problem.args["B"], rtol=2e-6)
    np.testing.assert_allclose(Q, problem.args["Q"], rtol=2e-6)
    np.testing.assert_allclose(R, problem.args["R"], rtol=2e-6)
    np.testing.assert_allclose(N, 1.3 * problem.args["N"], rtol=2e-6)

    cross_jacobian = jax.jit(jax.jacrev(lambda beta: blocks(beta)[-1]))(jnp.asarray(1.3))
    np.testing.assert_allclose(cross_jacobian, problem.args["N"], rtol=2e-6)


def _failure_problem(control_costs):
    input_layout = phx.dynamics.InputLayout((2,), roles="control")
    time_grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0]),
        time_id="local-failure-grid",
    )

    def transition(context, state, control, args):
        del context, control, args
        return state

    def stage(player):
        def cost(context, state, control, args):
            del context, state
            return 0.5 * control @ args[player] @ control

        return cost

    def terminal(time, state, args):
        del time, state, args
        return jnp.asarray(0.0)

    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=input_layout,
        system_id="local-failure-system",
    )
    problem = DeterministicFeedbackGameProblem(
        phx.control.DiscreteControlDynamics(system),
        time_grid,
        jnp.asarray([0.0]),
        PlayerControlPartition(("left", "right"), (1, 1)),
        stage_costs=(stage(0), stage(1)),
        terminal_costs=(terminal, terminal),
        args=control_costs,
        problem_id="local-failure-problem",
    )
    policy = phx.dynamics.CallableInputPolicy(
        lambda context, state, args: jnp.zeros((2,)),
        input_layout=input_layout,
        policy_id="zero-nominal",
    )
    evaluation = evaluate_game_policy(problem, policy)
    scaling = ILQGameScaling(jnp.ones(1), jnp.ones(2), jnp.ones(2))
    return problem, evaluation, scaling


def test_lq_curvature_failure_status_and_evidence_propagate_exactly():
    costs = (
        jnp.asarray([[-1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    problem, evaluation, scaling = _failure_problem(costs)
    suggestion = _suggest(problem, evaluation, scaling, suggestion_id="curvature")

    expected = int(LQFeedbackNashStatus.OWN_CURVATURE_NOT_POSITIVE_DEFINITE)
    assert int(suggestion.status) == expected
    assert int(suggestion.lq_result.status) == expected
    assert int(suggestion.status) == int(
        LocalAffineGameSuggestionStatus.OWN_CURVATURE_NOT_POSITIVE_DEFINITE
    )
    assert suggestion.lq_diagnostics is suggestion.lq_result.diagnostics
    assert not bool(suggestion.valid)


def test_lq_rank_failure_status_and_evidence_propagate_exactly():
    singular = jnp.ones((2, 2))
    problem, evaluation, scaling = _failure_problem((singular, singular))
    suggestion = _suggest(problem, evaluation, scaling, suggestion_id="rank")

    expected = int(LQFeedbackNashStatus.COUPLED_SYSTEM_RANK_DEFICIENT)
    assert int(suggestion.status) == expected
    assert int(suggestion.lq_result.status) == expected
    assert int(suggestion.status) == int(
        LocalAffineGameSuggestionStatus.COUPLED_SYSTEM_RANK_DEFICIENT
    )
    np.testing.assert_array_equal(
        suggestion.lq_diagnostics.coupled_ranks,
        suggestion.lq_result.diagnostics.coupled_ranks,
    )
    assert not bool(suggestion.valid)
