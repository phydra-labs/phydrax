import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.control._batched_trajectory import (
    plan_ilqr,
    prepare_ilqr,
    solve_prepared_ilqr,
)
from tests._control_systems import make_discrete_control_dynamics


class _ScaledTransition(eqx.Module):
    scale: jax.Array

    def __call__(self, context, state, control, args):
        del context, args
        accepted = state + self.scale * control
        return phx.dynamics.DiscreteTransitionResult(
            accepted + 100.0,
            accepted,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        )


def test_prepared_ilqr_preserves_two_dimensional_case_axes_and_statuses():
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state + control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="batched-integrator",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        phx.dynamics.TimeGrid(jnp.asarray([0.0, 1.0, 2.0]), time_id="batch-grid"),
        jnp.asarray([[[1.0], [2.0]], [[-1.0], [-2.0]]]),
        running_cost=lambda time, state, control, args: 0.5 * jnp.sum(control**2),
        terminal_cost=lambda time, state, args: 0.5 * jnp.sum(state**2),
        problem_id="batched-ilqr",
    )
    controls = jnp.zeros(problem.case_shape + (2, 1))
    plan = plan_ilqr(problem, max_iterations=8)
    prepared = prepare_ilqr(plan, problem, controls)
    result = eqx.filter_jit(solve_prepared_ilqr)(prepared)
    assert result.trajectory.states.shape == problem.case_shape + (3, 1)
    assert result.policy.feedback.shape == problem.case_shape + (2, 1, 1)
    assert result.diagnostics.status.shape == problem.case_shape
    assert result.diagnostics.objective_history.shape == problem.case_shape + (8,)


def test_prepared_ilqr_uses_six_pose_feedback_coordinates_across_signs():
    geometry = phx.metrix.QuaternionPoseStateGeometry()
    local_space = phx.linalg.ArraySpace((6,), dtype=jnp.float32)
    state_layout = phx.dynamics.StateLayout(
        (7,),
        geometry=geometry,
        local_space=local_space,
        tangent_space=local_space,
        layout_id="test:batched-ilqr-pose",
    )
    system = phx.dynamics.DiscreteSystem(
        lambda context, state, control, args: state,
        state_layout=state_layout,
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="test:batched-ilqr-pose-identity",
    )
    pose = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.2, -0.4, 0.7])
    equivalent = pose.at[:4].multiply(-1.0)
    problem = phx.control.ControlProblem(
        phx.control.DiscreteControlDynamics(system),
        phx.dynamics.TimeGrid(
            jnp.asarray([0.0, 1.0]),
            time_id="test:batched-ilqr-pose-grid",
        ),
        jnp.stack((pose, equivalent)),
        terminal_cost=lambda time, state, args: (
            0.5 * jnp.sum(geometry.inverse_retract(pose, state) ** 2)
        ),
        problem_id="test:batched-ilqr-pose-problem",
    )
    controls = jnp.zeros((2, 1, 1))
    result = eqx.filter_jit(solve_prepared_ilqr)(
        prepare_ilqr(
            plan_ilqr(problem, max_iterations=1, gradient_tolerance=1.0e6),
            problem,
            controls,
        )
    )

    assert result.policy.feedback.shape == (2, 1, 1, 6)
    assert result.trajectory.states.shape == (2, 2, 7)
    assert jnp.all(result.trajectory.successful)
    evidence = result.trajectory.transition_evidence
    assert evidence is not None
    assert jnp.all(evidence.attempted)
    np.testing.assert_allclose(
        jax.vmap(geometry.inverse_retract)(
            result.trajectory.states[0],
            result.trajectory.states[1],
        ),
        0.0,
    )


def test_prepared_ilqr_retains_rejected_line_search_attempt_metrics():
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="batched-line-search-rejection",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        phx.dynamics.TimeGrid(
            jnp.asarray([0.0, 1.0, 2.0]),
            time_id="batched-line-search-rejection-grid",
        ),
        jnp.asarray([[0.0]]),
        running_cost=lambda time, state, control, args: (control[0] - 1.0) ** 4,
        problem_id="batched-line-search-rejection",
    )
    controls = jnp.zeros((1, 2, 1))
    result = eqx.filter_jit(solve_prepared_ilqr)(
        prepare_ilqr(
            plan_ilqr(
                problem,
                max_iterations=1,
                regularization=0.0,
                initial_step_size=10.0,
                line_search_steps=1,
            ),
            problem,
            controls,
        )
    )

    assert int(result.status[0]) == phx.control.ILQRStatus.LINE_SEARCH_FAILED
    assert int(result.diagnostics.line_search_evaluations_history[0, 0]) == 1
    assert jnp.isfinite(result.diagnostics.expected_reduction_history[0, 0])
    assert jnp.isfinite(result.diagnostics.actual_reduction_history[0, 0])
    assert result.diagnostics.actual_reduction_history[0, 0] < 0.0
    np.testing.assert_array_equal(result.trajectory.controls, controls)


def test_prepared_ilqr_retains_selected_transition_evidence_per_case():
    failure_status = 73

    def transition(context, state, control, args):
        del context, args
        successful = state[0] >= 0.0
        candidate = state + control + 100.0
        accepted = jnp.where(successful, state + control, state)
        return phx.dynamics.DiscreteTransitionResult(
            candidate,
            accepted,
            successful,
            jnp.where(successful, 0, failure_status),
        )

    dynamics = make_discrete_control_dynamics(
        transition,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="batched-status-aware-integrator",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        phx.dynamics.TimeGrid(
            jnp.asarray([0.0, 1.0, 2.0]),
            time_id="batched-status-aware-grid",
        ),
        jnp.asarray([[1.0], [-1.0]]),
        terminal_cost=lambda time, state, args: 0.5 * jnp.sum(state**2),
        problem_id="batched-status-aware-ilqr",
    )
    controls = jnp.zeros(problem.case_shape + (2, 1))
    result = eqx.filter_jit(solve_prepared_ilqr)(
        prepare_ilqr(
            plan_ilqr(
                problem,
                max_iterations=1,
                gradient_tolerance=1.0e6,
            ),
            problem,
            controls,
        )
    )

    evidence = result.trajectory.transition_evidence
    assert evidence is not None
    np.testing.assert_allclose(
        evidence.candidate_states,
        jnp.asarray([[[101.0], [101.0]], [[99.0], [jnp.nan]]]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        evidence.accepted_states,
        jnp.asarray([[[1.0], [1.0]], [[-1.0], [jnp.nan]]]),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        evidence.attempted,
        jnp.asarray([[True, True], [True, False]]),
    )
    np.testing.assert_array_equal(
        evidence.successful,
        jnp.asarray([[True, True], [False, False]]),
    )
    np.testing.assert_array_equal(
        evidence.status,
        jnp.asarray([[0, 0], [failure_status, 0]], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        result.trajectory.backend_status,
        jnp.asarray([0, failure_status], dtype=jnp.int32),
    )


def test_prepared_ilqr_keeps_transition_parameters_dynamic():
    dynamics = make_discrete_control_dynamics(
        _ScaledTransition(jnp.asarray(1.0)),
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="batched-parameterized-integrator",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        phx.dynamics.TimeGrid(
            jnp.asarray([0.0, 1.0, 2.0]),
            time_id="batched-parameterized-grid",
        ),
        jnp.asarray([[0.0]]),
        terminal_cost=lambda time, state, args: 0.5 * jnp.sum(state**2),
        problem_id="batched-parameterized-ilqr",
    )
    prepared = prepare_ilqr(
        plan_ilqr(
            problem,
            max_iterations=1,
            gradient_tolerance=1.0e6,
        ),
        problem,
        jnp.ones((1, 2, 1)),
    )
    refreshed = eqx.tree_at(
        lambda value: value.problem.dynamics.system.transition.scale,
        prepared,
        jnp.asarray(2.0),
    )

    solve = eqx.filter_jit(solve_prepared_ilqr)
    first = solve(prepared)
    second = solve(refreshed)

    np.testing.assert_allclose(first.trajectory.final_state, jnp.asarray([[2.0]]))
    np.testing.assert_allclose(second.trajectory.final_state, jnp.asarray([[4.0]]))
