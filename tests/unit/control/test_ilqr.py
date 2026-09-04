#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.control import (
    DifferentialControlFlow,
    finite_horizon_lqr,
    ILQRStatus,
    solve_ilqr,
)
from tests._control_systems import (
    make_differential_control_dynamics,
    make_discrete_control_dynamics,
)


def test_ilqr_policy_feedback_uses_quaternion_pose_local_error():
    geometry = phx.metrix.QuaternionPoseStateGeometry()
    local_space = phx.linalg.ArraySpace((6,), dtype=jnp.float32)
    state_layout = phx.dynamics.StateLayout(
        (7,),
        geometry=geometry,
        local_space=local_space,
        tangent_space=local_space,
        layout_id="test:ilqr-quaternion-pose",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0]), time_id="test:ilqr-quaternion-grid"
    )
    pose = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.2, -0.4, 0.7])
    policy = phx.control.ILQRPolicy(
        grid,
        jnp.stack((pose, pose)),
        jnp.asarray([[0.3]]),
        jnp.ones((1, 1, 6)),
        state_layout=state_layout,
        control_shape=(1,),
        policy_id="test:ilqr-quaternion-policy",
    )

    nominal = policy.evaluate(jnp.empty((0,)), 0.0, state=pose)
    equivalent = policy.evaluate(
        jnp.empty((0,)),
        0.0,
        state=pose.at[:4].multiply(-1.0),
    )

    np.testing.assert_allclose(nominal, jnp.asarray([0.3]))
    np.testing.assert_allclose(equivalent, nominal)
    assert policy.feedback.shape == (1, 1, 6)


def _problem(
    transition,
    times,
    initial_state,
    running_cost,
    terminal_cost,
    *,
    state_shape,
    control_shape,
    problem_id,
    args=None,
):
    grid = phx.dynamics.TimeGrid(times, time_id=f"{problem_id}:time")
    dynamics = make_discrete_control_dynamics(
        transition,
        state_shape=state_shape,
        control_shape=control_shape,
        dynamics_id=f"{problem_id}:dynamics",
    )
    return phx.control.ControlProblem(
        dynamics,
        grid,
        initial_state,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        args=args,
        problem_id=problem_id,
    )


def test_ilqr_reduces_exactly_to_affine_finite_horizon_lqr():
    horizon = 5
    a = jnp.array(
        [
            [[1.0, 0.2], [0.0, 1.0]],
            [[1.0, 0.15], [0.0, 0.95]],
            [[1.0, 0.25], [0.0, 1.0]],
            [[1.0, 0.2], [0.0, 0.9]],
            [[1.0, 0.1], [0.0, 0.95]],
        ]
    )
    b = jnp.array(
        [
            [[0.02], [0.2]],
            [[0.01], [0.15]],
            [[0.03], [0.25]],
            [[0.02], [0.2]],
            [[0.01], [0.1]],
        ]
    )
    bias = jnp.array(
        [[0.03, -0.01], [0.0, 0.02], [-0.02, 0.01], [0.01, 0.0], [0.0, -0.02]]
    )
    q = jnp.stack([jnp.diag(jnp.array([1.0 + 0.1 * i, 0.4])) for i in range(horizon)])
    r = jnp.asarray([[[0.7 + 0.05 * i]] for i in range(horizon)])
    cross = jnp.asarray([[[0.02], [-0.01]]] * horizon)
    q_linear = jnp.asarray([[-0.3 + 0.02 * i, 0.1] for i in range(horizon)])
    r_linear = jnp.asarray([[0.04 - 0.01 * i] for i in range(horizon)])
    q_terminal = jnp.diag(jnp.array([3.0, 0.8]))
    terminal_linear = jnp.array([-0.6, 0.2])
    times = jnp.arange(horizon + 1, dtype=float)
    initial_state = jnp.array([0.8, -0.2])
    args = (a, b, bias, q, r, cross, q_linear, r_linear, q_terminal, terminal_linear)

    def transition(context, state, control, data):
        index = context.step_index
        return data[0][index] @ state + data[1][index] @ control + data[2][index]

    def running_cost(time, state, control, data):
        index = jnp.asarray(time, dtype=jnp.int32)
        return (
            0.5 * state @ data[3][index] @ state
            + 0.5 * control @ data[4][index] @ control
            + state @ data[5][index] @ control
            + data[6][index] @ state
            + data[7][index] @ control
        )

    def terminal_cost(time, state, data):
        del time
        return 0.5 * state @ data[8] @ state + data[9] @ state

    problem = _problem(
        transition,
        times,
        initial_state,
        running_cost,
        terminal_cost,
        state_shape=(2,),
        control_shape=(1,),
        problem_id="ilqr-linear-reduction",
        args=args,
    )
    lqr = finite_horizon_lqr(
        a,
        b,
        q,
        r,
        q_terminal,
        dynamics_bias=bias,
        state_control_cross=cross,
        state_linear=q_linear,
        control_linear=r_linear,
        terminal_linear=terminal_linear,
        time_grid=problem.time_grid,
    )
    ilqr = solve_ilqr(
        problem,
        jnp.zeros((horizon, 1)),
        regularization=0.0,
        gradient_tolerance=1e-7,
        cost_tolerance=1e-10,
    )
    lqr_trajectory = problem.rollout(lqr.policy, jnp.asarray(0.0))

    assert int(ilqr.status) == ILQRStatus.SUCCESS
    np.testing.assert_allclose(
        ilqr.trajectory.states, lqr_trajectory.states, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        ilqr.trajectory.controls, lqr_trajectory.controls, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        ilqr.policy.feedback, lqr.feedback_gain, rtol=2e-6, atol=2e-6
    )


def _nonlinear_problem(kind):
    dt = 0.08
    horizon = 24
    times = jnp.arange(horizon + 1) * dt
    if kind == "pendulum":
        initial_state = jnp.array([0.9, 0.0])

        def transition(time, state, control, args):
            del time, args
            angle, velocity = state
            torque = control[0]
            acceleration = -jnp.sin(angle) - 0.08 * velocity + torque
            return state + dt * jnp.array([velocity, acceleration])

        def running_cost(time, state, control, args):
            del time, args
            return 0.5 * (
                4.0 * state[0] ** 2 + 0.3 * state[1] ** 2 + 0.08 * control[0] ** 2
            )

        def terminal_cost(time, state, args):
            del time, args
            return 15.0 * state[0] ** 2 + 2.0 * state[1] ** 2

        state_shape = (2,)
    else:
        initial_state = jnp.array([0.0, 0.45, 0.0, 0.0])
        cart_mass = 1.0
        pole_mass = 0.15
        length = 0.5
        gravity = 9.81

        def transition(time, state, control, args):
            del time, args
            position, angle, velocity, angular_velocity = state
            force = control[0]
            total_mass = cart_mass + pole_mass
            sine = jnp.sin(angle)
            cosine = jnp.cos(angle)
            temporary = (
                force + pole_mass * length * angular_velocity**2 * sine
            ) / total_mass
            angular_acceleration = (gravity * sine - cosine * temporary) / (
                length * (4.0 / 3.0 - pole_mass * cosine**2 / total_mass)
            )
            cart_acceleration = (
                temporary
                - pole_mass * length * angular_acceleration * cosine / total_mass
            )
            return state + dt * jnp.array(
                [velocity, angular_velocity, cart_acceleration, angular_acceleration]
            )

        def running_cost(time, state, control, args):
            del time, args
            return 0.5 * (
                0.2 * state[0] ** 2
                + 8.0 * state[1] ** 2
                + 0.1 * state[2] ** 2
                + 0.4 * state[3] ** 2
                + 0.05 * control[0] ** 2
            )

        def terminal_cost(time, state, args):
            del time, args
            return (
                2.0 * state[0] ** 2
                + 30.0 * state[1] ** 2
                + state[2] ** 2
                + 4.0 * state[3] ** 2
            )

        state_shape = (4,)
    return _problem(
        transition,
        times,
        initial_state,
        running_cost,
        terminal_cost,
        state_shape=state_shape,
        control_shape=(1,),
        problem_id=f"ilqr-{kind}",
    )


@pytest.mark.parametrize("kind", ["pendulum", "cartpole"])
def test_ilqr_improves_nonlinear_pendulum_and_cartpole(kind):
    problem = _nonlinear_problem(kind)
    result = solve_ilqr(
        problem,
        jnp.zeros((problem.time_grid.num_steps, 1)),
        max_iterations=12,
        regularization=1e-4,
        gradient_tolerance=1e-5,
    )

    history = np.asarray(result.diagnostics.objective_history)
    assert int(result.diagnostics.accepted_iterations) > 0
    assert history[-1] < history[0]
    assert np.all(np.diff(history) < 0.0)
    assert np.all(np.isfinite(result.trajectory.states))
    assert np.all(np.isfinite(result.trajectory.controls))


def test_ilqr_reported_gradient_agrees_with_direct_open_loop_gradient():
    dt = 0.2
    horizon = 6

    def transition(time, state, control, args):
        del time, args
        return state + dt * jnp.array(
            [state[1], -jnp.sin(state[0]) + control[0] + 0.1 * control[0] ** 3]
        )

    def running_cost(time, state, control, args):
        del time, args
        return (
            0.5 * (state @ state + 0.3 * control @ control) + 0.02 * state[0] * control[0]
        )

    def terminal_cost(time, state, args):
        del time, args
        return state @ jnp.diag(jnp.array([2.0, 0.5])) @ state

    problem = _problem(
        transition,
        jnp.arange(horizon + 1) * dt,
        jnp.array([0.6, -0.1]),
        running_cost,
        terminal_cost,
        state_shape=(2,),
        control_shape=(1,),
        problem_id="ilqr-gradient",
    )
    initial_controls = jnp.linspace(-0.2, 0.25, horizon).reshape((horizon, 1))

    def objective(controls):
        state = problem.initial_state
        total = jnp.asarray(0.0)
        for step in range(horizon):
            total = total + dt * running_cost(
                problem.time_grid.times[step], state, controls[step], None
            )
            state = transition(problem.time_grid.times[step], state, controls[step], None)
        return total + terminal_cost(problem.time_grid.times[-1], state, None)

    direct_norm = jnp.linalg.norm(jax.grad(objective)(initial_controls))
    result = solve_ilqr(problem, initial_controls, max_iterations=2, regularization=1e-5)
    np.testing.assert_allclose(
        result.diagnostics.gradient_norm_history[0], direct_norm, rtol=2e-6, atol=2e-6
    )


def test_ilqr_rejects_non_positive_definite_backward_pass_without_fallback():
    problem = _problem(
        lambda time, state, control, args: state,
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([0.0]),
        lambda time, state, control, args: -0.5 * control[0] ** 2,
        None,
        state_shape=(1,),
        control_shape=(1,),
        problem_id="ilqr-indefinite",
    )
    result = solve_ilqr(problem, jnp.zeros((2, 1)), regularization=0.0)

    assert int(result.status) == ILQRStatus.BACKWARD_PASS_NOT_POSITIVE_DEFINITE
    assert int(result.diagnostics.failed_step) == 1
    assert result.diagnostics.regularized_minimum_curvature_history[0] < 0.0
    assert not bool(result.successful)


def test_differential_ilqr_requires_selected_flow_and_propagates_failed_integration():
    grid = phx.dynamics.TimeGrid(
        jnp.array([0.0, 0.5, 1.0, 1.5]), time_id="failed-flow-time"
    )
    dynamics = make_differential_control_dynamics(
        lambda time, state, control, args: -state + control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="failed-flow-dynamics",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        grid,
        jnp.array([1.0]),
        running_cost=lambda time, state, control, args: state[0] ** 2 + control[0] ** 2,
        problem_id="failed-flow-problem",
    )
    controls = jnp.zeros((3, 1))
    with pytest.raises(ValueError, match="explicit DifferentialControlFlow"):
        solve_ilqr(problem, controls)

    flow = DifferentialControlFlow(
        lambda t0, t1, state, control, args: jnp.where(
            t0 >= 0.5,
            jnp.full_like(state, jnp.nan),
            state + (t1 - t0) * (-state + control),
        ),
        flow_id="selected-euler-that-reports-failure",
    )
    result = solve_ilqr(problem, controls, differential_flow=flow)

    assert int(result.status) == ILQRStatus.INITIAL_ROLLOUT_FAILED
    assert int(result.diagnostics.failed_step) == 1
    assert not bool(result.trajectory.successful)
    assert result.trajectory.discretization_id == flow.flow_id
    assert not bool(result.control_result.sampled_loss.valid)


def test_ilqr_rejects_explicit_finite_rollback_and_retains_transition_evidence():
    failure_status = 59

    def transition(context, state, control, args):
        del context, args
        return phx.dynamics.DiscreteTransitionResult(
            state + control + 100.0,
            state,
            jnp.asarray(False),
            jnp.asarray(failure_status, dtype=jnp.int32),
        )

    problem = _problem(
        transition,
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([2.0]),
        lambda time, state, control, args: state[0] ** 2 + control[0] ** 2,
        None,
        state_shape=(1,),
        control_shape=(1,),
        problem_id="ilqr-finite-rollback",
    )
    result = solve_ilqr(problem, jnp.asarray([[1.0], [2.0]]))

    assert int(result.status) == ILQRStatus.INITIAL_ROLLOUT_FAILED
    assert int(result.diagnostics.failed_step) == 0
    assert not bool(result.trajectory.successful)
    assert float(result.trajectory.states[1, 0]) == 2.0
    assert bool(jnp.isnan(result.trajectory.states[2, 0]))
    assert int(result.trajectory.backend_status) == failure_status
    evidence = result.trajectory.transition_evidence
    assert evidence is not None
    np.testing.assert_allclose(evidence.candidate_states[0], jnp.asarray([103.0]))
    np.testing.assert_allclose(evidence.accepted_states[0], jnp.asarray([2.0]))
    np.testing.assert_array_equal(evidence.attempted, jnp.asarray([True, False]))
    assert int(evidence.first_failure_step) == 0
    assert int(evidence.first_failure_status) == failure_status


def test_ilqr_reports_line_search_rejection_without_changing_nominal_controls():
    problem = _problem(
        lambda time, state, control, args: state,
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([0.0]),
        lambda time, state, control, args: (control[0] - 1.0) ** 4,
        None,
        state_shape=(1,),
        control_shape=(1,),
        problem_id="ilqr-line-search-rejection",
    )
    initial = jnp.zeros((2, 1))
    result = solve_ilqr(
        problem,
        initial,
        regularization=0.0,
        initial_step_size=10.0,
        line_search_steps=1,
    )

    assert int(result.status) == ILQRStatus.LINE_SEARCH_FAILED
    assert int(result.diagnostics.line_search_evaluations_history[-1]) == 1
    np.testing.assert_array_equal(result.trajectory.controls, initial)
    assert not bool(result.successful)


def test_ilqr_is_deterministic_and_policy_rolls_out_through_public_control_problem():
    problem = _nonlinear_problem("pendulum")
    initial = jnp.zeros((problem.time_grid.num_steps, 1))
    first = solve_ilqr(problem, initial, max_iterations=8, regularization=1e-4)
    second = solve_ilqr(problem, initial, max_iterations=8, regularization=1e-4)

    np.testing.assert_array_equal(first.trajectory.states, second.trajectory.states)
    np.testing.assert_array_equal(first.trajectory.controls, second.trajectory.controls)
    np.testing.assert_array_equal(
        first.diagnostics.objective_history, second.diagnostics.objective_history
    )
    replay = problem.rollout(first.policy, jnp.empty((0,)))
    assert bool(replay.successful)
    np.testing.assert_allclose(
        replay.states, first.trajectory.states, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        replay.controls, first.trajectory.controls, rtol=2e-6, atol=2e-6
    )
