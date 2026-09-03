import equinox as eqx
import jax.numpy as jnp

import phydrax as phx
from phydrax.control._batched_trajectory import (
    plan_ilqr,
    prepare_ilqr,
    solve_prepared_ilqr,
)
from tests._control_systems import make_discrete_control_dynamics


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
