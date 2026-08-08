#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
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
        specification, tolerance=1e-9
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
    qp = compilation.quadratic_program

    assert decision.initial_state_slice == slice(0, 2)
    assert decision.state_stage_slices == (slice(2, 4), slice(4, 6))
    assert decision.control_stage_slices == (slice(6, 7), slice(7, 8))
    assert decision.num_variables == 8
    assert constraints_layout.initial_condition_slice == slice(0, 2)
    assert constraints_layout.dynamics_slices == (slice(2, 4), slice(4, 6))
    assert constraints_layout.stage_equality_slices == (slice(6, 7), slice(7, 8))
    assert constraints_layout.terminal_equality_slice == slice(8, 9)
    assert constraints_layout.state_lower_slices == (
        slice(0, 2),
        slice(2, 4),
        slice(4, 6),
    )
    assert constraints_layout.state_upper_slices == (
        slice(6, 8),
        slice(8, 10),
        slice(10, 12),
    )
    assert constraints_layout.control_lower_slices == (slice(12, 13), slice(13, 14))
    assert constraints_layout.control_upper_slices == (slice(14, 15), slice(15, 16))
    assert constraints_layout.stage_inequality_slices == (
        slice(16, 18),
        slice(18, 20),
    )
    assert constraints_layout.terminal_inequality_slice == slice(20, 21)
    assert qp.num_equalities == 9
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
    first_state_lower = constraints_layout.state_lower_slices[0]
    first_state_upper = constraints_layout.state_upper_slices[0]
    first_control_lower = constraints_layout.control_lower_slices[0]
    first_control_upper = constraints_layout.control_upper_slices[0]
    np.testing.assert_array_equal(
        qp.inequality_matrix[first_state_lower, decision.state_slice(0)],
        -jnp.eye(2),
    )
    np.testing.assert_array_equal(qp.inequality_rhs[first_state_lower], [10.0, 10.0])
    np.testing.assert_array_equal(
        qp.inequality_matrix[first_state_upper, decision.state_slice(0)],
        jnp.eye(2),
    )
    np.testing.assert_array_equal(qp.inequality_rhs[first_state_upper], [10.0, 10.0])
    np.testing.assert_array_equal(
        qp.inequality_matrix[first_control_lower, decision.control_slice(0)],
        -jnp.ones((1, 1)),
    )
    np.testing.assert_array_equal(qp.inequality_rhs[first_control_lower], [2.0])
    np.testing.assert_array_equal(
        qp.inequality_matrix[first_control_upper, decision.control_slice(0)],
        jnp.ones((1, 1)),
    )
    np.testing.assert_array_equal(qp.inequality_rhs[first_control_upper], [2.0])
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
        specification, tolerance=2e-8, max_iterations=200
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
        eqx.EquinoxRuntimeError,
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
    assert compilation.quadratic_program.batch_shape == (2,)
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
    assert infeasible_solution.status == phx.optim.QP_INFEASIBLE
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
    assert nonfinite_solution.status == phx.optim.QP_NONFINITE
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
    global_result = phx.control.solve_receding_horizon_mpc(
        specification,
        prediction_horizon=1,
        terminal_policy="global",
        tolerance=1e-9,
    )
    always_result = phx.control.solve_receding_horizon_mpc(
        specification,
        prediction_horizon=1,
        terminal_policy="always",
        tolerance=1e-9,
    )
    no_terminal_result = phx.control.solve_receding_horizon_mpc(
        specification,
        prediction_horizon=1,
        terminal_policy="none",
        tolerance=1e-9,
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
    with pytest.raises(NotImplementedError, match="warm starts"):
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
    assert infeasible_result.qp_results[0].status == phx.optim.QP_INFEASIBLE
    assert infeasible_result.status == phx.optim.QP_INFEASIBLE
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
    assert result.qp_results[0].status == phx.optim.QP_SUCCESS
    assert result.qp_results[1].status == phx.optim.QP_NONFINITE
    assert result.status == phx.optim.QP_NONFINITE
    assert not result.valid
    assert not result.trajectory.successful
    assert jnp.isnan(result.states[-1]).any()
