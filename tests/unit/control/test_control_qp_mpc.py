#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_compiler_matches_finite_lqr_and_preserves_exact_primal_policy():
    horizon = 3
    dynamics = jnp.array([[[1.0]], [[0.9]], [[1.1]]])
    controls = jnp.ones((horizon, 1, 1))
    state_costs = jnp.array([[[1.0]], [[1.5]], [[0.75]]])
    control_costs = jnp.array([[[0.5]], [[0.8]], [[1.2]]])
    terminal_cost = jnp.array([[2.0]])
    bias = jnp.array([[0.1], [-0.2], [0.05]])
    cross = jnp.array([[[0.1]], [[-0.05]], [[0.2]]])
    state_linear = jnp.array([[0.2], [-0.1], [0.3]])
    control_linear = jnp.array([[-0.4], [0.25], [0.1]])
    initial = jnp.array([1.3])
    specification = phx.control.LinearQuadraticControlProblem(
        dynamics,
        controls,
        initial,
        state_costs,
        control_costs,
        terminal_cost,
        dynamics_bias=bias,
        state_control_cross=cross,
        state_linear=state_linear,
        control_linear=control_linear,
        terminal_linear=jnp.array([-0.15]),
        stage_constants=jnp.array([0.5, 0.25, -0.1]),
        terminal_constant=0.75,
    )
    qp_solution = phx.control.solve_linear_quadratic_control(
        specification,
        policy=phx.optim.ConvexSolvePolicy(
            termination=phx.optim.ConvexTermination(absolute=1e-9)
        ),
    )
    lqr = phx.control.finite_horizon_lqr(
        dynamics,
        controls,
        state_costs,
        control_costs,
        terminal_cost,
        dynamics_bias=bias,
        state_control_cross=cross,
        state_linear=state_linear,
        control_linear=control_linear,
        terminal_linear=jnp.array([-0.15]),
    )
    lqr_states = [initial]
    lqr_controls = []
    state = initial
    for stage in range(horizon):
        control = lqr.feedback_gain[stage] @ state + lqr.feedforward[stage]
        lqr_controls.append(control)
        state = dynamics[stage] @ state + controls[stage] @ control + bias[stage]
        lqr_states.append(state)

    assert qp_solution.valid
    np.testing.assert_allclose(
        qp_solution.states, jnp.stack(lqr_states), atol=2e-6, rtol=2e-6
    )
    np.testing.assert_allclose(
        qp_solution.controls, jnp.stack(lqr_controls), atol=2e-6, rtol=2e-6
    )
    decoded_states, decoded_controls = qp_solution.compilation.decode(
        qp_solution.qp_result.primal
    )
    np.testing.assert_array_equal(decoded_states, qp_solution.states)
    np.testing.assert_array_equal(decoded_controls, qp_solution.controls)
    np.testing.assert_array_equal(
        qp_solution.compilation.decision_layout.encode(decoded_states, decoded_controls),
        qp_solution.qp_result.primal,
    )
    assert qp_solution.trajectory.problem_id == specification.problem_id
    assert qp_solution.trajectory.backend_id == qp_solution.qp_result.backend
    assert qp_solution.policy.parameterization_id.endswith(":qp-policy")
    np.testing.assert_allclose(
        qp_solution.objective,
        qp_solution.qp_result.objective + 1.4,
        atol=1e-8,
    )


def test_control_qp_decoder_rejects_foreign_numeric_binding():
    first_specification = phx.control.LinearQuadraticControlProblem(
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.asarray([0.0]),
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1)),
        problem_id="binding-a",
    )
    second_specification = phx.control.LinearQuadraticControlProblem(
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.asarray([2.0]),
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1)),
        problem_id="binding-b",
    )
    prepared = phx.control.prepare_linear_quadratic_control(first_specification)
    foreign = phx.control.solve_linear_quadratic_control(second_specification)

    with pytest.raises(ValueError, match="provenance does not match"):
        phx.control.decode_linear_control_solution(prepared, foreign.qp_result)


def test_decision_and_constraint_layouts_identify_every_compiled_block():
    dynamics = jnp.array(
        [
            [[1.0, 0.2], [0.0, 1.0]],
            [[0.9, 0.1], [0.0, 1.1]],
        ]
    )
    controls = jnp.array([[[0.0], [1.0]], [[0.1], [1.0]]])
    cross = jnp.array([[[0.2], [-0.1]], [[0.3], [0.4]]])
    specification = phx.control.LinearQuadraticControlProblem(
        dynamics,
        controls,
        jnp.array([1.0, -1.0]),
        jnp.stack((jnp.eye(2), 2.0 * jnp.eye(2))),
        jnp.array([[[3.0]], [[4.0]]]),
        5.0 * jnp.eye(2),
        dynamics_bias=jnp.array([[0.5, -0.5], [0.25, 0.75]]),
        state_control_cross=cross,
        state_linear=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        control_linear=jnp.array([[5.0], [6.0]]),
        state_lower_bounds=-10.0 * jnp.ones((3, 2)),
        state_upper_bounds=10.0 * jnp.ones((3, 2)),
        control_lower_bounds=-2.0 * jnp.ones((2, 1)),
        control_upper_bounds=2.0 * jnp.ones((2, 1)),
        stage_equality_state_matrix=jnp.ones((2, 1, 2)),
        stage_equality_control_matrix=jnp.zeros((2, 1, 1)),
        stage_equality_rhs=jnp.array([[0.0], [1.0]]),
        stage_inequality_state_matrix=jnp.ones((2, 2, 2)),
        stage_inequality_control_matrix=jnp.ones((2, 2, 1)),
        stage_inequality_rhs=20.0 * jnp.ones((2, 2)),
        terminal_equality_matrix=jnp.array([[1.0, 0.0]]),
        terminal_equality_rhs=jnp.array([0.0]),
        terminal_inequality_matrix=jnp.array([[0.0, 1.0]]),
        terminal_inequality_rhs=jnp.array([3.0]),
    )
    compilation = phx.control.compile_linear_quadratic_control(specification)
    decision = compilation.decision_layout
    constraints_layout = compilation.constraint_layout
    bound_layout = compilation.bound_layout
    qp = compilation.program

    assert decision.initial_state_slice == slice(0, 2)
    assert decision.state_stage_slices == (slice(2, 4), slice(4, 6))
    assert decision.control_stage_slices == (slice(6, 7), slice(7, 8))
    assert decision.num_variables == 8
    assert constraints_layout.initial_condition_slice == slice(0, 2)
    assert constraints_layout.dynamics_slices == (slice(2, 4), slice(4, 6))
    assert constraints_layout.stage_equality_slices == (slice(6, 7), slice(7, 8))
    assert constraints_layout.terminal_equality_slice == slice(8, 9)
    assert bound_layout.state_lower_slices == (
        slice(0, 2),
        slice(2, 4),
        slice(4, 6),
    )
    assert bound_layout.state_upper_slices == bound_layout.state_lower_slices
    assert bound_layout.control_lower_slices == (slice(6, 7), slice(7, 8))
    assert bound_layout.control_upper_slices == bound_layout.control_lower_slices
    assert constraints_layout.stage_inequality_slices == (
        slice(0, 2),
        slice(2, 4),
    )
    assert constraints_layout.terminal_inequality_slice == slice(4, 5)
    assert qp.num_equalities == 9
    assert qp.num_user_equalities == 9
    assert qp.num_user_inequalities == 5
    assert qp.num_inequalities == 21

    np.testing.assert_array_equal(
        qp.quadratic[decision.state_slice(0), decision.control_slice(0)], cross[0]
    )
    np.testing.assert_array_equal(
        qp.quadratic[decision.control_slice(0), decision.state_slice(0)], cross[0].T
    )
    np.testing.assert_array_equal(
        qp.equality_matrix[
            constraints_layout.initial_condition_slice,
            decision.initial_state_slice,
        ],
        jnp.eye(2),
    )
    first_dynamics = constraints_layout.dynamics_slices[0]
    np.testing.assert_array_equal(
        qp.equality_matrix[first_dynamics, decision.state_slice(0)], -dynamics[0]
    )
    np.testing.assert_array_equal(
        qp.equality_matrix[first_dynamics, decision.control_slice(0)], -controls[0]
    )
    np.testing.assert_array_equal(
        qp.equality_matrix[first_dynamics, decision.state_slice(1)], jnp.eye(2)
    )
    np.testing.assert_array_equal(qp.equality_rhs[first_dynamics], [0.5, -0.5])
    np.testing.assert_array_equal(
        qp.quadratic[decision.state_slice(1), decision.state_slice(1)],
        2.0 * jnp.eye(2),
    )
    np.testing.assert_array_equal(
        qp.quadratic[decision.control_slice(1), decision.control_slice(1)],
        jnp.array([[4.0]]),
    )
    np.testing.assert_array_equal(
        qp.quadratic[decision.state_slice(2), decision.state_slice(2)],
        5.0 * jnp.eye(2),
    )
    np.testing.assert_array_equal(
        qp.linear[decision.state_slice(1)], jnp.array([3.0, 4.0])
    )
    np.testing.assert_array_equal(qp.linear[decision.control_slice(1)], jnp.array([6.0]))
    first_stage_equality = constraints_layout.stage_equality_slices[0]
    np.testing.assert_array_equal(
        qp.equality_matrix[first_stage_equality, decision.state_slice(0)],
        jnp.ones((1, 2)),
    )
    np.testing.assert_array_equal(
        qp.equality_matrix[first_stage_equality, decision.control_slice(0)],
        jnp.zeros((1, 1)),
    )
    np.testing.assert_array_equal(qp.equality_rhs[first_stage_equality], [0.0])
    np.testing.assert_array_equal(
        qp.equality_matrix[
            constraints_layout.terminal_equality_slice,
            decision.state_slice(2),
        ],
        jnp.array([[1.0, 0.0]]),
    )
    first_state = bound_layout.state_lower_slices[0]
    first_control = bound_layout.control_lower_slices[0]
    np.testing.assert_array_equal(
        qp.lower_bounds[first_state],
        -10.0 * jnp.ones(2),
    )
    np.testing.assert_array_equal(
        qp.upper_bounds[first_state],
        10.0 * jnp.ones(2),
    )
    np.testing.assert_array_equal(qp.lower_bounds[first_control], [-2.0])
    np.testing.assert_array_equal(qp.upper_bounds[first_control], [2.0])
    first_stage_inequality = constraints_layout.stage_inequality_slices[0]
    np.testing.assert_array_equal(
        qp.inequality_matrix[first_stage_inequality, decision.state_slice(0)],
        jnp.ones((2, 2)),
    )
    np.testing.assert_array_equal(
        qp.inequality_matrix[first_stage_inequality, decision.control_slice(0)],
        jnp.ones((2, 1)),
    )
    np.testing.assert_array_equal(qp.inequality_rhs[first_stage_inequality], [20.0, 20.0])
    np.testing.assert_array_equal(
        qp.inequality_matrix[
            constraints_layout.terminal_inequality_slice,
            decision.state_slice(2),
        ],
        jnp.array([[0.0, 1.0]]),
    )


def test_box_polyhedral_and_terminal_constraints_are_enforced_without_repair():
    specification = phx.control.LinearQuadraticControlProblem(
        jnp.ones((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.array([1.0]),
        jnp.zeros((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.zeros((1, 1)),
        control_linear=jnp.array([[10.0], [0.0]]),
        state_lower_bounds=-2.0 * jnp.ones((3, 1)),
        state_upper_bounds=2.0 * jnp.ones((3, 1)),
        control_lower_bounds=-0.5 * jnp.ones((2, 1)),
        control_upper_bounds=0.5 * jnp.ones((2, 1)),
        stage_inequality_control_matrix=jnp.ones((2, 1, 1)),
        stage_inequality_rhs=jnp.array([[0.0], [-0.2]]),
        terminal_equality_matrix=jnp.ones((1, 1)),
        terminal_equality_rhs=jnp.array([0.25]),
        terminal_inequality_matrix=jnp.ones((1, 1)),
        terminal_inequality_rhs=jnp.array([0.3]),
    )
    solution = phx.control.solve_linear_quadratic_control(
        specification,
        policy=phx.optim.ConvexSolvePolicy(
            termination=phx.optim.ConvexTermination(
                absolute=2e-8,
                maximum_steps=200,
            )
        ),
    )
    assert solution.valid
    np.testing.assert_allclose(solution.controls[:, 0], [-0.5, -0.25], atol=2e-6)
    np.testing.assert_allclose(solution.states[:, 0], [1.0, 0.5, 0.25], atol=2e-6)
    assert jnp.max(solution.qp_result.inequality_violation) <= 2e-8
    assert jnp.max(jnp.abs(solution.qp_result.equality_residual)) <= 2e-8
    assert jnp.all(solution.controls >= specification.control_lower_bounds - 2e-8)
    assert jnp.all(solution.controls <= specification.control_upper_bounds + 2e-8)


@pytest.mark.parametrize(
    ("control_cost", "cross"),
    [
        (jnp.array([[[-1.0]]]), None),
        (jnp.zeros((1, 1, 1)), jnp.ones((1, 1, 1))),
    ],
)
def test_compiler_rejects_indefinite_joint_stage_costs(control_cost, cross):
    specification = phx.control.LinearQuadraticControlProblem(
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1, 1, 1)),
        control_cost,
        jnp.zeros((1, 1)),
        state_control_cross=cross,
    )

    with pytest.raises(
        ValueError,
        match="joint stage costs must be positive semidefinite",
    ):
        phx.control.solve_linear_quadratic_control(specification)


def test_mpc_rejects_complex_initial_state_before_real_dtype_conversion():
    specification = phx.control.LinearQuadraticControlProblem(
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1)),
    )

    with pytest.raises(TypeError, match="initial_state must be real-valued"):
        phx.control.solve_receding_horizon_mpc(
            specification,
            prediction_horizon=1,
            terminal_policy="global",
            initial_state=jnp.array([1.0 + 2.0j]),
        )


def test_batched_cases_and_failure_statuses_remain_case_explicit():
    horizon = 2
    specification = phx.control.LinearQuadraticControlProblem(
        jnp.ones((2, horizon, 1, 1)),
        jnp.ones((2, horizon, 1, 1)),
        jnp.array([[1.0], [2.0]]),
        jnp.ones((2, horizon, 1, 1)),
        jnp.ones((2, horizon, 1, 1)),
        jnp.ones((2, 1, 1)),
    )
    compilation = phx.control.compile_linear_quadratic_control(specification)
    solution = phx.control.solve_linear_quadratic_control(specification)
    assert compilation.program.batch_shape == (2,)
    assert solution.states.shape == (2, horizon + 1, 1)
    assert solution.controls.shape == (2, horizon, 1)
    assert solution.valid.shape == (2,)
    assert jnp.all(solution.valid)

    infeasible = phx.control.LinearQuadraticControlProblem(
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.array([1.0]),
        jnp.zeros((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1)),
        terminal_equality_matrix=jnp.ones((1, 1)),
        terminal_equality_rhs=jnp.array([0.0]),
    )
    infeasible_solution = phx.control.solve_linear_quadratic_control(infeasible)
    assert infeasible_solution.status == phx.optim.ConvexProgramStatus.PRIMAL_INFEASIBLE
    assert not infeasible_solution.valid

    nonfinite = phx.control.LinearQuadraticControlProblem(
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.array([1.0]),
        jnp.array([[[jnp.nan]]]),
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1)),
    )
    nonfinite_solution = phx.control.solve_linear_quadratic_control(nonfinite)
    assert nonfinite_solution.status == phx.optim.ConvexProgramStatus.NONFINITE_INPUT
    assert not nonfinite_solution.valid
    assert jnp.isnan(nonfinite_solution.qp_result.primal).any()


def test_receding_horizon_state_handoff_and_terminal_policy_are_explicit():
    horizon = 3
    specification = phx.control.LinearQuadraticControlProblem(
        jnp.ones((horizon, 1, 1)),
        jnp.ones((horizon, 1, 1)),
        jnp.array([1.0]),
        jnp.zeros((horizon, 1, 1)),
        jnp.ones((horizon, 1, 1)),
        10.0 * jnp.ones((1, 1)),
    )
    policy = phx.optim.ConvexSolvePolicy(
        termination=phx.optim.ConvexTermination(absolute=1e-9)
    )
    global_result = phx.control.solve_receding_horizon_mpc(
        specification,
        prediction_horizon=1,
        terminal_policy="global",
        policy=policy,
    )
    always_result = phx.control.solve_receding_horizon_mpc(
        specification,
        prediction_horizon=1,
        terminal_policy="always",
        policy=policy,
    )
    no_terminal_result = phx.control.solve_receding_horizon_mpc(
        specification,
        prediction_horizon=1,
        terminal_policy="none",
        policy=policy,
    )
    assert global_result.valid
    assert always_result.valid
    assert no_terminal_result.valid
    np.testing.assert_allclose(global_result.controls[:2, 0], 0.0, atol=1e-8)
    assert always_result.controls[0, 0] < -0.8
    np.testing.assert_allclose(no_terminal_result.controls, 0.0, atol=1e-8)
    for stage, subproblem in enumerate(global_result.subproblem_solutions):
        np.testing.assert_allclose(
            subproblem.compilation.specification.initial_state,
            global_result.states[stage],
            atol=1e-9,
        )
    np.testing.assert_allclose(
        global_result.states[1:, 0],
        global_result.states[:-1, 0] + global_result.controls[:, 0],
        atol=1e-9,
    )
    with pytest.raises(TypeError, match="LinearControlQPSolution"):
        phx.control.solve_receding_horizon_mpc(
            specification,
            prediction_horizon=1,
            terminal_policy="global",
            warm_start=jnp.zeros((horizon, 1)),
        )


def test_mpc_propagates_infeasible_qp_and_nonfinite_rollout_failures():
    infeasible = phx.control.LinearQuadraticControlProblem(
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.array([1.0]),
        jnp.zeros((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1)),
        terminal_equality_matrix=jnp.ones((1, 1)),
        terminal_equality_rhs=jnp.zeros((1,)),
    )
    infeasible_result = phx.control.solve_receding_horizon_mpc(
        infeasible,
        prediction_horizon=1,
        terminal_policy="global",
    )
    assert (
        infeasible_result.qp_results[0].status
        == phx.optim.ConvexProgramStatus.PRIMAL_INFEASIBLE
    )
    assert infeasible_result.status == phx.optim.ConvexProgramStatus.PRIMAL_INFEASIBLE
    assert not infeasible_result.valid

    dynamics = jnp.array([[[1.0]], [[jnp.nan]]])
    specification = phx.control.LinearQuadraticControlProblem(
        dynamics,
        jnp.ones((2, 1, 1)),
        jnp.array([1.0]),
        jnp.zeros((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.ones((1, 1)),
    )
    result = phx.control.solve_receding_horizon_mpc(
        specification,
        prediction_horizon=1,
        terminal_policy="global",
    )
    assert result.qp_results[0].status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert result.qp_results[1].status == phx.optim.ConvexProgramStatus.NONFINITE_INPUT
    assert result.status == phx.optim.ConvexProgramStatus.NONFINITE_INPUT
    assert not result.valid
    assert not result.trajectory.successful
    assert jnp.isnan(result.states[-1]).any()


_TIGHT = phx.optim.ConvexSolvePolicy(
    termination=phx.optim.ConvexTermination(
        absolute=1e-12, relative=1e-12, maximum_steps=200
    )
)


def _shifted(problem, tangent, step):
    def at(name):
        return getattr(problem, name) + step * getattr(tangent, name)

    return phx.control.LinearQuadraticControlProblem(
        at("dynamics_matrices"),
        at("control_matrices"),
        at("initial_state"),
        at("state_costs"),
        at("control_costs"),
        at("terminal_state_cost"),
        dynamics_bias=at("dynamics_bias"),
        state_linear=at("state_linear"),
        control_linear=at("control_linear"),
        control_lower_bounds=at("control_lower_bounds"),
        control_upper_bounds=at("control_upper_bounds"),
    )


def test_mpc_sensitivity_matches_finite_differences_away_from_active_set_changes():
    horizon = 4
    dynamics = jnp.stack(
        (
            jnp.broadcast_to(jnp.array([[1.0, 0.1], [0.0, 0.95]]), (horizon, 2, 2)),
            jnp.broadcast_to(jnp.array([[0.9, 0.2], [-0.1, 1.0]]), (horizon, 2, 2)),
        )
    )
    specification = phx.control.LinearQuadraticControlProblem(
        dynamics,
        jnp.broadcast_to(jnp.array([[0.0], [0.1]]), (2, horizon, 2, 1)),
        jnp.array([[1.0, -0.5], [-0.8, 0.3]]),
        jnp.broadcast_to(jnp.eye(2), (2, horizon, 2, 2)),
        0.1 * jnp.ones((2, horizon, 1, 1)),
        5.0 * jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
        dynamics_bias=jnp.broadcast_to(jnp.array([0.01, -0.02]), (2, horizon, 2)),
        state_linear=jnp.broadcast_to(jnp.array([0.05, 0.0]), (2, horizon, 2)),
        control_linear=jnp.zeros((2, horizon, 1)),
        control_lower_bounds=-0.4 * jnp.ones((2, horizon, 1)),
        control_upper_bounds=0.6 * jnp.ones((2, horizon, 1)),
    )
    controller = phx.control.RecedingHorizonMPC(
        specification,
        prediction_horizon=2,
        terminal_policy="global",
        policy=_TIGHT,
    )
    sensitivity = phx.control.prepare_receding_horizon_mpc_sensitivity(controller)
    audited = controller.solve()

    assert sensitivity.refusal is None
    assert bool(sensitivity.regular)
    np.testing.assert_array_equal(sensitivity.states, audited.states)
    np.testing.assert_array_equal(sensitivity.controls, audited.controls)
    # Both bound types are strictly active somewhere, so the derivative crosses
    # active constraints without any active-set change.
    assert jnp.any(jnp.abs(audited.controls - 0.6) < 1e-8)
    assert jnp.any(jnp.abs(audited.controls + 0.4) < 1e-8)

    keys = jax.random.split(jax.random.PRNGKey(7), 9)

    def symmetric(key, shape):
        value = jax.random.normal(key, shape)
        return 0.5 * (value + jnp.swapaxes(value, -1, -2))

    tangent = eqx.tree_at(
        lambda problem: (
            problem.dynamics_matrices,
            problem.control_matrices,
            problem.initial_state,
            problem.state_costs,
            problem.control_costs,
            problem.terminal_state_cost,
            problem.dynamics_bias,
            problem.state_linear,
            problem.control_upper_bounds,
        ),
        jax.tree.map(jnp.zeros_like, specification),
        (
            0.1 * jax.random.normal(keys[0], (2, horizon, 2, 2)),
            jax.random.normal(keys[1], (2, horizon, 2, 1)),
            jax.random.normal(keys[2], (2, 2)),
            0.1 * symmetric(keys[3], (2, horizon, 2, 2)),
            0.01 * symmetric(keys[4], (2, horizon, 1, 1)),
            symmetric(keys[5], (2, 2, 2)),
            jax.random.normal(keys[6], (2, horizon, 2)),
            jax.random.normal(keys[7], (2, horizon, 2)),
            jax.random.normal(keys[8], (2, horizon, 1)),
        ),
    )
    state_tangent, control_tangent = sensitivity.jvp(tangent)

    step = 1e-6
    plus = phx.control.RecedingHorizonMPC(
        _shifted(specification, tangent, step),
        prediction_horizon=2,
        terminal_policy="global",
        policy=_TIGHT,
    ).solve()
    minus = phx.control.RecedingHorizonMPC(
        _shifted(specification, tangent, -step),
        prediction_horizon=2,
        terminal_policy="global",
        policy=_TIGHT,
    ).solve()
    np.testing.assert_allclose(
        state_tangent, (plus.states - minus.states) / (2 * step), atol=2e-5
    )
    np.testing.assert_allclose(
        control_tangent, (plus.controls - minus.controls) / (2 * step), atol=2e-5
    )

    state_weights = jax.random.normal(jax.random.PRNGKey(11), state_tangent.shape)
    control_weights = jax.random.normal(jax.random.PRNGKey(12), control_tangent.shape)
    cotangent = sensitivity.vjp(state_weights, control_weights)
    assert isinstance(cotangent, phx.control.LinearQuadraticControlProblem)
    pairing = sum(
        jnp.vdot(left, right)
        for left, right in zip(
            jax.tree.leaves(cotangent), jax.tree.leaves(tangent), strict=True
        )
    )
    np.testing.assert_allclose(
        pairing,
        jnp.vdot(state_weights, state_tangent)
        + jnp.vdot(control_weights, control_tangent),
        rtol=1e-10,
    )
    np.testing.assert_array_equal(cotangent.time_grid.times, 0.0)


def test_mpc_sensitivity_refuses_the_complete_derivative_for_one_weak_window():
    specification = phx.control.LinearQuadraticControlProblem(
        jnp.ones((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.array([0.0]),
        jnp.zeros((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.zeros((1, 1)),
        # Window 0's unconstrained optimum u = 0.5 lies exactly on its bound
        # (zero multiplier); window 1's optimum u = 0.2 is strictly interior.
        control_linear=jnp.array([[-0.5], [-0.2]]),
        control_upper_bounds=0.5 * jnp.ones((2, 1)),
    )
    controller = phx.control.RecedingHorizonMPC(
        specification,
        prediction_horizon=1,
        terminal_policy="none",
        policy=_TIGHT,
    )
    sensitivity = phx.control.prepare_receding_horizon_mpc_sensitivity(controller)

    assert sensitivity.result.valid
    np.testing.assert_allclose(sensitivity.controls[:, 0], [0.5, 0.2], atol=1e-5)
    np.testing.assert_array_equal(sensitivity.stage_optimal, [True, True])
    np.testing.assert_array_equal(sensitivity.stage_regular, [False, True])
    assert not bool(sensitivity.regular)
    zero = jax.tree.map(jnp.zeros_like, specification)
    with pytest.raises(ValueError, match=r"refused: windows \[0\] have nonregular"):
        sensitivity.jvp(zero)
    with pytest.raises(ValueError, match="refused"):
        sensitivity.vjp(
            jnp.zeros_like(sensitivity.states), jnp.zeros_like(sensitivity.controls)
        )

    infeasible = phx.control.LinearQuadraticControlProblem(
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.array([1.0]),
        jnp.zeros((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1)),
        terminal_equality_matrix=jnp.ones((1, 1)),
        terminal_equality_rhs=jnp.zeros((1,)),
    )
    refused = phx.control.prepare_receding_horizon_mpc_sensitivity(
        phx.control.RecedingHorizonMPC(
            infeasible, prediction_horizon=1, terminal_policy="global"
        )
    )
    assert refused.linearization is None
    assert "windows [0] are not valid and OPTIMAL" in refused.refusal


def test_mpc_sensitivity_admits_only_dense_cold_unregularized_qp_sensitivities():
    specification = phx.control.LinearQuadraticControlProblem(
        jnp.ones((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.array([1.0]),
        jnp.ones((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.ones((1, 1)),
    )

    def refused(match, error=ValueError, differentiation=None, **options):
        controller = phx.control.RecedingHorizonMPC(
            specification,
            prediction_horizon=1,
            terminal_policy="global",
            **options,
        )
        with pytest.raises(error, match=match):
            phx.control.prepare_receding_horizon_mpc_sensitivity(
                controller, differentiation=differentiation
            )

    refused(
        "dense-only",
        compilation_policy=phx.control.LinearControlCompilationPolicy("sparse"),
    )
    refused(
        "no warm-start derivative", warm_start_policy=phx.control.MPCWarmStartPolicy()
    )
    refused(
        "zero solver regularization",
        policy=phx.optim.ConvexSolvePolicy(regularization=1e-8),
    )
    refused(
        "has no dense QP sensitivity",
        policy=phx.optim.ConvexSolvePolicy(phx.optim.ClarabelInteriorPoint()),
    )
    refused(
        "requires a dense unrolled plan",
        policy=phx.optim.ConvexSolvePolicy(phx.optim.MPAXraPDHG(unroll=False)),
    )
    refused(
        "active-set-kkt or barrier-kkt",
        differentiation=phx.optim.ConvexDifferentiationPolicy("algorithmic"),
    )
    with pytest.raises(TypeError, match="RecedingHorizonMPC"):
        phx.control.prepare_receding_horizon_mpc_sensitivity(specification)
