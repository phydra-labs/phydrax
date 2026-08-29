#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from tests._control_systems import (
    make_differential_control_dynamics,
    make_discrete_control_dynamics,
)


def _linear_problem(*, num_steps=2, path_constraints=(), terminal_constraints=()):
    grid = phx.dynamics.TimeGrid(
        jnp.arange(num_steps + 1, dtype=float),
        time_id=f"multiple-shooting-linear-{num_steps}",
    )
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state + control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="multiple-shooting-linear-integrator",
    )
    return phx.control.ControlProblem(
        dynamics,
        grid,
        jnp.asarray([0.0]),
        running_cost=lambda time, state, control, args: (
            0.5 * (state[0] ** 2 + control[0] ** 2)
        ),
        terminal_cost=lambda time, state, args: 0.5 * (state[0] - 1.0) ** 2,
        path_constraints=path_constraints,
        terminal_constraints=terminal_constraints,
        problem_id=f"multiple-shooting-linear-problem-{num_steps}",
    )


def test_linear_subproblem_matches_kkt_oracle_and_exact_derivatives():
    problem = _linear_problem()
    states = jnp.zeros((3, 1))
    controls = jnp.zeros((2, 1))
    local = phx.control.linearize_multiple_shooting(problem, states, controls)
    qp = local.quadratic_program

    kkt = np.block(
        [
            [np.asarray(qp.quadratic), np.asarray(qp.equality_matrix).T],
            [
                np.asarray(qp.equality_matrix),
                np.zeros((qp.num_equalities, qp.num_equalities)),
            ],
        ]
    )
    rhs = np.concatenate((-np.asarray(qp.linear), np.asarray(qp.equality_rhs)))
    oracle = np.linalg.solve(kkt, rhs)[: qp.num_variables]
    qp_result = phx.optim.solve_quadratic_program(qp)

    assert qp_result.successful
    np.testing.assert_allclose(qp_result.primal, oracle, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        local.objective_gradient,
        jnp.asarray([0.0, 0.0, -1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        jnp.diag(local.objective_hessian),
        jnp.ones((5,)),
    )
    np.testing.assert_allclose(
        local.equality_jacobian,
        jnp.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 1.0],
            ]
        ),
    )
    assert local.equality_provenance == (
        "boundary:initial:state[0]",
        "continuity:segment[0]:state[0]",
        "continuity:segment[1]:state[0]",
    )

    result = phx.control.solve_multiple_shooting(
        problem, states, controls, max_iterations=3
    )
    assert result.last_qp_result is not None
    assert result.status == phx.control.MULTIPLE_SHOOTING_SUCCESS
    np.testing.assert_allclose(result.layout.pack(states, controls), jnp.zeros((5,)))
    np.testing.assert_allclose(result.last_qp_result.primal, oracle, atol=1e-6)
    np.testing.assert_allclose(result.trajectory.states, result.state_nodes, atol=1e-6)


def test_exact_boundary_continuity_path_and_terminal_defect_accounting():
    def path(time, state, control, args):
        return state[0] + 2.0 * control[0] - 0.4

    def terminal(time, state, args):
        return state[0] - 0.2

    problem = _linear_problem(path_constraints=(path,), terminal_constraints=(terminal,))
    states = jnp.asarray([[0.2], [0.9], [-0.1]])
    controls = jnp.asarray([[0.3], [-0.2]])
    local = phx.control.linearize_multiple_shooting(problem, states, controls)

    np.testing.assert_allclose(local.boundary_defect, jnp.asarray([0.2]))
    np.testing.assert_allclose(local.continuity_defects, jnp.asarray([[-0.4], [0.8]]))
    np.testing.assert_allclose(local.path_residuals, jnp.asarray([[0.4], [0.1]]))
    np.testing.assert_allclose(local.terminal_residuals, jnp.asarray([-0.3]))
    np.testing.assert_allclose(local.equality_residuals, jnp.asarray([0.2, -0.4, 0.8]))
    np.testing.assert_allclose(local.inequality_residuals, jnp.asarray([0.4, 0.1, -0.3]))
    assert local.inequality_provenance == (
        "path:segment[0]:constraint[0]",
        "path:segment[1]:constraint[0]",
        "terminal:constraint[0]",
    )


def test_nonlinear_constrained_problem_converges_without_projection_or_repair():
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0, 2.0]), time_id="multiple-shooting-nonlinear"
    )
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state + control + 0.1 * control**2,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="multiple-shooting-nonlinear-dynamics",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        grid,
        jnp.asarray([0.0]),
        running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
        path_constraints=(lambda time, state, control, args: control[0] ** 2 - 1.0,),
        terminal_constraints=(lambda time, state, args: 0.8 - state[0],),
        problem_id="multiple-shooting-nonlinear-constrained",
    )

    result = phx.control.solve_multiple_shooting(
        problem,
        jnp.zeros((3, 1)),
        jnp.zeros((2, 1)),
        hessian_regularization=1e-9,
        max_iterations=10,
    )

    assert result.successful
    assert result.maximum_defect <= 1e-6
    assert result.maximum_constraint_violation <= 1e-6
    assert jnp.all(result.path_residuals <= 0.0)
    assert jnp.all(result.terminal_residuals <= 1e-6)
    assert result.rollout_state_error <= 1e-5
    assert jnp.all(result.history.accepted)
    assert np.all(np.diff(np.asarray(result.history.merit)) <= 1e-10)


def test_rejected_merit_line_search_is_explicit():
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0]), time_id="multiple-shooting-rejected-line"
    )
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state + control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="multiple-shooting-rejected-line-dynamics",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        grid,
        jnp.asarray([0.0]),
        terminal_cost=lambda time, state, args: jnp.log(jnp.cosh(state[0])),
        problem_id="multiple-shooting-rejected-line-problem",
    )

    result = phx.control.solve_multiple_shooting(
        problem,
        jnp.asarray([[0.0], [3.0]]),
        jnp.asarray([[3.0]]),
        max_line_search_iterations=1,
        max_iterations=2,
    )
    assert result.last_qp_result is not None

    assert result.status == phx.control.MULTIPLE_SHOOTING_LINE_SEARCH_FAILED
    assert result.last_qp_result.successful
    np.testing.assert_array_equal(result.history.accepted, jnp.asarray([False]))
    np.testing.assert_allclose(result.history.step_size, jnp.asarray([0.0]))
    np.testing.assert_allclose(result.control_nodes, jnp.asarray([[3.0]]))


def test_infeasible_dense_qp_status_is_propagated():
    def impossible(time, state, control, args):
        return jnp.asarray(1.0)

    problem = _linear_problem(num_steps=1, path_constraints=(impossible,))

    result = phx.control.solve_multiple_shooting(
        problem, jnp.zeros((2, 1)), jnp.zeros((1, 1)), max_iterations=2
    )
    assert result.last_qp_result is not None

    assert result.status == phx.control.MULTIPLE_SHOOTING_QP_FAILED
    assert result.last_qp_result.status == phx.optim.ConvexProgramStatus.PRIMAL_INFEASIBLE
    np.testing.assert_array_equal(
        result.history.qp_status,
        jnp.asarray([phx.optim.ConvexProgramStatus.PRIMAL_INFEASIBLE]),
    )
    assert not result.valid


def test_concave_qp_model_at_stationary_maximum_is_rejected():
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0]), time_id="multiple-shooting-concave"
    )
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="multiple-shooting-concave-dynamics",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        grid,
        jnp.asarray([0.0]),
        running_cost=lambda time, state, control, args: -0.5 * control[0] ** 2,
        problem_id="multiple-shooting-concave-problem",
    )

    result = phx.control.solve_multiple_shooting(
        problem,
        jnp.zeros((2, 1)),
        jnp.zeros((1, 1)),
        hessian_regularization=0.25,
        max_iterations=1,
    )

    assert result.status == phx.control.MULTIPLE_SHOOTING_QP_FAILED
    assert not result.successful
    assert result.last_qp_result is None
    assert result.history.num_iterations == 0


def test_differential_segments_use_canonical_solver_and_report_failed_integration():
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.5, 1.0]), time_id="multiple-shooting-ode"
    )
    dynamics = make_differential_control_dynamics(
        lambda time, state, control, args: control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="multiple-shooting-ode-dynamics",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        grid,
        jnp.asarray([0.0]),
        running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
        terminal_cost=lambda time, state, args: 0.5 * (state[0] - 1.0) ** 2,
        problem_id="multiple-shooting-ode-problem",
    )
    result = phx.control.solve_multiple_shooting(
        problem,
        jnp.zeros((3, 1)),
        jnp.zeros((2, 1)),
        hessian_regularization=1e-10,
        max_iterations=5,
    )
    assert result.successful
    np.testing.assert_allclose(result.control_nodes, 0.5, rtol=1e-5, atol=1e-6)
    assert result.trajectory.backend_id == "backend:diffrax"

    failed = phx.control.solve_multiple_shooting(
        problem,
        jnp.zeros((3, 1)),
        jnp.zeros((2, 1)),
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.01,
        integration_max_steps=1,
    )
    assert failed.status == phx.control.MULTIPLE_SHOOTING_INTEGRATION_FAILED
    assert failed.history.num_iterations == 0
    assert not failed.trajectory.successful


def test_differential_rollout_audit_matches_segments_at_control_jumps():
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0, 2.5]), time_id="multiple-shooting-control-jump"
    )
    dynamics = make_differential_control_dynamics(
        lambda time, state, control, args: control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="multiple-shooting-control-jump-dynamics",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        grid,
        jnp.asarray([0.0]),
        problem_id="multiple-shooting-control-jump-problem",
    )
    states = jnp.asarray([[0.0], [0.0], [1.5]])
    controls = jnp.asarray([[0.0], [1.0]])

    result = phx.control.solve_multiple_shooting(
        problem,
        states,
        controls,
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=2.0,
        max_iterations=1,
    )

    assert result.status == phx.control.MULTIPLE_SHOOTING_SUCCESS
    np.testing.assert_allclose(result.trajectory.states, states)
    np.testing.assert_allclose(result.rollout_state_error, 0.0)


def test_solver_is_deterministic_and_rejects_batched_optimization():
    problem = _linear_problem()
    kwargs: dict[str, Any] = dict(
        initial_states=jnp.zeros((3, 1)),
        initial_controls=jnp.zeros((2, 1)),
        max_iterations=4,
    )
    first = phx.control.solve_multiple_shooting(problem, **kwargs)
    second = phx.control.solve_multiple_shooting(problem, **kwargs)

    np.testing.assert_array_equal(first.state_nodes, second.state_nodes)
    np.testing.assert_array_equal(first.control_nodes, second.control_nodes)
    np.testing.assert_array_equal(first.history.objective, second.history.objective)
    np.testing.assert_array_equal(first.history.step_size, second.history.step_size)
    np.testing.assert_array_equal(first.history.qp_status, second.history.qp_status)

    batched = phx.control.ControlProblem(
        problem.dynamics,
        problem.time_grid,
        jnp.zeros((2, 1)),
        running_cost=problem.running_cost,
        terminal_cost=problem.terminal_cost,
        problem_id="multiple-shooting-batched-rejected",
    )
    with pytest.raises(ValueError, match="one optimization case"):
        phx.control.solve_multiple_shooting(batched)


def test_global_control_search_trajectory_is_a_native_seed():
    problem = _linear_problem()
    parameterization = phx.control.PiecewiseConstantControlParameterization(
        problem.time_grid,
        problem.control_shape,
        parameterization_id="multiple-shooting-global-seed-controls",
    )
    search = phx.optim.DifferentialEvolutionSearch(
        8,
        2,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        design=phx.sampling.SobolDesign(scrambled=True),
    )
    global_result = phx.control.search_control(
        problem,
        parameterization,
        search,
        key=jr.key(912),
        coefficient_bounds=(-jnp.ones((2, 1)), jnp.ones((2, 1))),
    )

    positional = phx.control.solve_multiple_shooting(
        problem, global_result.trajectory, max_iterations=5
    )
    keyword = phx.control.solve_multiple_shooting(
        problem, initial_trajectory=global_result.trajectory, max_iterations=5
    )

    assert positional.successful
    assert keyword.successful
    np.testing.assert_allclose(positional.state_nodes, keyword.state_nodes)
    np.testing.assert_allclose(positional.control_nodes, keyword.control_nodes)
    assert positional.trajectory.problem_id == global_result.trajectory.problem_id


def test_multiple_shooting_lowers_to_structured_nlp_and_solves_natively():
    problem = _linear_problem()
    compilation = phx.control.compile_structured_multiple_shooting(
        problem,
        jnp.zeros((3, 1)),
        jnp.zeros((2, 1)),
    )
    result = phx.control.solve_structured_multiple_shooting(
        compilation,
        method=phx.optim.PrimalDualInteriorPoint(mode="sparse-augmented"),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1e-6,
            relative_optimality=0.0,
            maximum_steps=80,
        ),
    )
    assert bool(result.successful)
    assert result.maximum_defect <= 1e-6
    assert result.maximum_constraint_violation <= 1e-6
