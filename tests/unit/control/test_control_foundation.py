#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control import (
    AffineFeedbackPolicy,
    CONTROL_DYNAMICS_FAILED,
    CONTROL_INFEASIBLE,
    CONTROL_SUCCESS,
    ControlProblem,
    PiecewiseConstantControlParameterization,
    PiecewiseLinearControlParameterization,
)
from phydrax.dynamics import TimeGrid
from tests._control_systems import (
    make_differential_control_dynamics,
    make_discrete_control_dynamics,
)


def _grid():
    return TimeGrid(jnp.asarray([0.0, 0.5, 1.0]), time_id="time:grid")


def test_control_foundation_constructor_guards_are_explicit():
    with pytest.raises(ValueError, match="at least two"):
        TimeGrid(jnp.asarray([0.0]), time_id="short")
    with pytest.raises(ValueError, match="strictly increasing"):
        TimeGrid(jnp.asarray([0.0, 0.0, 1.0]), time_id="repeated")
    with pytest.raises(ValueError, match="non-empty"):
        TimeGrid(jnp.asarray([0.0, 1.0]), time_id="")
    invalid_dynamics: Any = 3
    with pytest.raises(TypeError, match="callable"):
        make_discrete_control_dynamics(
            invalid_dynamics,
            state_shape=(1,),
            control_shape=(1,),
            dynamics_id="bad",
        )

    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state + control,
        state_shape=(2,),
        control_shape=(1,),
        dynamics_id="shape",
    )
    with pytest.raises(ValueError, match="state_shape"):
        ControlProblem(
            dynamics,
            _grid(),
            jnp.zeros((3,)),
            problem_id="bad-state",
        )


def test_discrete_rollout_preserves_case_time_axes_and_gradients():
    grid = _grid()
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state + control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="discrete-integrator",
    )
    parameterization = PiecewiseConstantControlParameterization(
        grid,
        (1,),
        parameterization_id="piecewise-constant",
    )
    problem = ControlProblem(
        dynamics,
        grid,
        jnp.asarray([[0.0], [10.0]]),
        problem_id="batched-discrete",
    )
    coefficients = jnp.asarray([[[1.0], [2.0]], [[-1.0], [-2.0]]])

    trajectory = problem.rollout(parameterization, coefficients)

    assert trajectory.states.shape == (2, 3, 1)
    assert trajectory.controls.shape == (2, 2, 1)
    assert trajectory.valid.shape == (2, 3)
    assert np.all(np.asarray(trajectory.status) == CONTROL_SUCCESS)
    assert np.allclose(
        np.asarray(trajectory.states[..., 0]),
        np.asarray([[0.0, 1.0, 3.0], [10.0, 9.0, 7.0]]),
    )
    assert trajectory.problem_id == "batched-discrete"
    assert trajectory.dynamics_id == "discrete-integrator"
    assert trajectory.discretization_id == grid.time_id

    scalar_problem = ControlProblem(
        dynamics,
        grid,
        jnp.asarray([0.0]),
        running_cost=lambda time, state, control, args: jnp.sum(control**2),
        terminal_cost=lambda time, state, args: jnp.sum(state**2),
        problem_id="gradient",
    )
    scalar_coefficients = jnp.asarray([[1.0], [2.0]])
    gradient = jax.grad(
        lambda values: (
            scalar_problem.evaluate(parameterization, values).sampled_loss.total
        )
    )(scalar_coefficients)
    assert np.allclose(np.asarray(gradient), np.asarray([[7.0], [8.0]]))


def test_discrete_rollout_masks_failed_feedback_policy_cases():
    grid = _grid()
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: jnp.full_like(state, jnp.nan),
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="failed-feedback-transition",
    )
    policy = AffineFeedbackPolicy(
        jnp.zeros((grid.num_steps, 1, 1)),
        jnp.ones((grid.num_steps, 1)),
        time_grid=grid,
        state_size=1,
        policy_id="failed-feedback-policy",
    )
    problem = ControlProblem(
        dynamics,
        grid,
        jnp.zeros((1,)),
        problem_id="failed-feedback-discrete",
    )

    trajectory = problem.rollout(policy, jnp.asarray(0.0))

    np.testing.assert_array_equal(trajectory.valid, jnp.array([True, False, False]))
    assert int(trajectory.status) == CONTROL_DYNAMICS_FAILED
    assert np.isfinite(np.asarray(trajectory.controls[0])).all()
    assert np.isnan(np.asarray(trajectory.controls[1])).all()


def test_sampled_loss_and_sampled_feasibility_remain_distinct():
    grid = _grid()
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state + control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="constrained",
    )
    parameterization = PiecewiseConstantControlParameterization(
        grid,
        (1,),
        parameterization_id="infeasible-control",
    )
    problem = ControlProblem(
        dynamics,
        grid,
        jnp.asarray([0.0]),
        running_cost=lambda time, state, control, args: control[0] ** 2,
        path_constraints=(lambda time, state, control, args: control[0] - 0.5,),
        problem_id="sampled-semantics",
    )

    result = problem.evaluate(parameterization, jnp.ones((2, 1)))

    assert bool(result.sampled_loss.valid)
    assert float(result.sampled_loss.total) == pytest.approx(1.0)
    assert not bool(result.feasibility.feasible)
    assert not result.feasibility.certified
    assert result.feasibility.method_id == "control-constraint:sampled-grid-noncertifying"
    assert int(result.status) == CONTROL_INFEASIBLE
    assert bool(result.valid)


def test_differential_rollout_is_differentiable_and_propagates_backend_failure():
    grid = _grid()
    dynamics = make_differential_control_dynamics(
        lambda time, state, control, args: control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="controlled-ode",
    )
    parameterization = PiecewiseLinearControlParameterization(
        grid,
        (1,),
        parameterization_id="linear-control",
    )
    problem = ControlProblem(
        dynamics,
        grid,
        jnp.asarray([0.0]),
        terminal_cost=lambda time, state, args: state[0] ** 2,
        problem_id="differential",
    )
    coefficients = jnp.ones((3, 1))

    trajectory = problem.rollout(parameterization, coefficients, dt0=0.05)
    assert np.allclose(np.asarray(trajectory.states[:, 0]), [0.0, 0.5, 1.0])
    assert int(trajectory.status) == CONTROL_SUCCESS
    assert trajectory.backend_id == "backend:diffrax"
    assert trajectory.method_id.startswith("temporal-configuration:")

    batched_problem = ControlProblem(
        dynamics,
        grid,
        jnp.asarray([[0.0], [10.0]]),
        problem_id="batched-differential",
    )
    batched_coefficients = jnp.stack((coefficients, -coefficients))
    batched = batched_problem.rollout(
        parameterization,
        batched_coefficients,
        dt0=0.05,
    )
    assert batched.states.shape == (2, 3, 1)
    assert np.allclose(
        np.asarray(batched.states[..., 0]),
        np.asarray([[0.0, 0.5, 1.0], [10.0, 9.5, 9.0]]),
    )
    assert np.all(np.asarray(batched.status) == CONTROL_SUCCESS)

    gradient = jax.grad(
        lambda values: (
            problem.evaluate(
                parameterization,
                values,
                dt0=0.05,
            ).sampled_loss.total
        )
    )(coefficients)
    assert gradient.shape == coefficients.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0

    failed = problem.rollout(
        parameterization,
        coefficients,
        dt0=0.01,
        max_steps=1,
        throw=False,
    )
    assert int(failed.status) == CONTROL_DYNAMICS_FAILED
    assert not bool(failed.successful)
    assert not bool(jnp.all(failed.valid))
    assert "maximum number of solver steps" in str(failed.backend_status)


def test_differential_failed_feedback_reconstruction_masks_invalid_states():
    grid = _grid()
    dynamics = make_differential_control_dynamics(
        lambda time, state, control, args: -state + control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="failed-feedback-ode",
    )
    policy = AffineFeedbackPolicy(
        jnp.zeros((grid.num_steps, 1, 1)),
        jnp.zeros((grid.num_steps, 1)),
        time_grid=grid,
        state_size=1,
        policy_id="failed-differential-feedback-policy",
    )
    problem = ControlProblem(
        dynamics,
        grid,
        jnp.ones((1,)),
        problem_id="failed-feedback-differential",
    )

    trajectory = problem.rollout(
        policy,
        jnp.asarray(0.0),
        dt0=0.01,
        max_steps=1,
        throw=False,
    )

    assert int(trajectory.status) == CONTROL_DYNAMICS_FAILED
    assert np.isfinite(np.asarray(trajectory.controls[0])).all()
    assert np.isnan(np.asarray(trajectory.controls[1])).all()


def test_differential_rollout_solves_declared_cases_independently():
    grid = _grid()

    def vector_field(time, state, control, args):
        del time, control, args
        return jnp.where(
            state[0] > 5.0,
            jnp.full_like(state, jnp.nan),
            jnp.zeros_like(state),
        )

    dynamics = make_differential_control_dynamics(
        vector_field,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="independent-case-ode",
    )
    parameterization = PiecewiseConstantControlParameterization(
        grid,
        (1,),
        parameterization_id="independent-case-control",
    )
    problem = ControlProblem(
        dynamics,
        grid,
        jnp.array([[[0.0], [10.0]], [[0.0], [10.0]]]),
        problem_id="independent-differential-cases",
    )

    trajectory = problem.rollout(
        parameterization,
        jnp.zeros((2, 2, grid.num_steps, 1)),
        dt0=0.05,
        max_steps=16,
        throw=False,
    )

    expected_success = jnp.array([[True, False], [True, False]])
    np.testing.assert_array_equal(trajectory.status == CONTROL_SUCCESS, expected_success)
    np.testing.assert_array_equal(jnp.all(trajectory.valid, axis=-1), expected_success)
    np.testing.assert_array_equal(
        trajectory.backend_status == dfx.RESULTS.successful,
        expected_success,
    )
    assert trajectory.states.shape == (2, 2, grid.num_times, 1)
    assert trajectory.controls.shape == (2, 2, grid.num_steps, 1)
    np.testing.assert_allclose(trajectory.states[:, 0, :, 0], 0.0)
