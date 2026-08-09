#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

import phydrax as phx
from phydrax.control import (
    continuous_lqr,
    discrete_lqr,
    finite_horizon_lqr,
    RiccatiStatus,
    solve_continuous_are,
    solve_discrete_are,
)
from tests._control_systems import make_discrete_control_dynamics


def _policy_rollout(a, b, c, policy, initial_state):
    state = jnp.asarray(initial_state)
    states = [state]
    controls = []
    for step, time in enumerate(np.asarray(policy.time_grid.times[:-1])):
        control = policy(float(time), state)
        state = a[step] @ state + b[step] @ control + c[step]
        controls.append(control)
        states.append(state)
    return jnp.stack(states), jnp.stack(controls)


def _equality_kkt_solution(a, b, c, q, r, n_cross, q_linear, r_linear, qf, qf_linear, x0):
    horizon, n, _ = a.shape
    m = b.shape[-1]
    num_states = (horizon + 1) * n
    size = num_states + horizon * m
    hessian = np.zeros((size, size))
    linear = np.zeros(size)
    for step in range(horizon):
        xs = slice(step * n, (step + 1) * n)
        us = slice(num_states + step * m, num_states + (step + 1) * m)
        hessian[xs, xs] += np.asarray(q[step])
        hessian[us, us] += np.asarray(r[step])
        hessian[xs, us] += np.asarray(n_cross[step])
        hessian[us, xs] += np.asarray(n_cross[step]).T
        linear[xs] += np.asarray(q_linear[step])
        linear[us] += np.asarray(r_linear[step])
    terminal = slice(horizon * n, (horizon + 1) * n)
    hessian[terminal, terminal] += np.asarray(qf)
    linear[terminal] += np.asarray(qf_linear)

    constraint = np.zeros(((horizon + 1) * n, size))
    right_hand_side = np.zeros((horizon + 1) * n)
    constraint[:n, :n] = np.eye(n)
    right_hand_side[:n] = np.asarray(x0)
    for step in range(horizon):
        rows = slice((step + 1) * n, (step + 2) * n)
        current = slice(step * n, (step + 1) * n)
        following = slice((step + 1) * n, (step + 2) * n)
        control = slice(num_states + step * m, num_states + (step + 1) * m)
        constraint[rows, current] = -np.asarray(a[step])
        constraint[rows, following] = np.eye(n)
        constraint[rows, control] = -np.asarray(b[step])
        right_hand_side[rows] = np.asarray(c[step])
    kkt = np.block(
        [
            [hessian, constraint.T],
            [constraint, np.zeros((constraint.shape[0], constraint.shape[0]))],
        ]
    )
    solution = np.linalg.solve(kkt, np.concatenate((-linear, right_hand_side)))[:size]
    states = solution[:num_states].reshape(horizon + 1, n)
    controls = solution[num_states:].reshape(horizon, m)
    return states, controls


def test_continuous_are_matches_textbook_and_scipy_for_stable_and_unstable_systems():
    b = jnp.ones((1, 1))
    q = jnp.ones((1, 1))
    r = jnp.ones((1, 1))
    stable = solve_continuous_are(jnp.array([[-1.0]]), b, q, r)
    unstable = solve_continuous_are(jnp.array([[1.0]]), b, q, r)

    np.testing.assert_allclose(stable.matrix, [[np.sqrt(2.0) - 1.0]], rtol=1e-10)
    np.testing.assert_allclose(unstable.matrix, [[np.sqrt(2.0) + 1.0]], rtol=1e-10)
    assert bool(stable.valid)
    assert bool(unstable.valid)

    a = np.array([[0.0, 1.0], [-2.0, 0.4]])
    b2 = np.array([[0.0], [1.0]])
    q2 = np.diag([2.0, 1.0])
    r2 = np.array([[0.7]])
    result = solve_continuous_are(a, b2, q2, r2)
    reference = scipy.linalg.solve_continuous_are(a, b2, q2, r2)
    np.testing.assert_allclose(result.matrix, reference, rtol=2e-9, atol=2e-9)
    assert result.diagnostics.equation.system_type == "continuous"
    assert result.diagnostics.relative_residual < 1e-9


def test_discrete_are_matches_scipy_and_stabilizes_unstable_system():
    a = np.array([[1.15, 0.2], [0.0, 0.8]])
    b = np.array([[1.0], [0.4]])
    q = np.diag([1.5, 0.5])
    r = np.array([[0.8]])
    result = solve_discrete_are(a, b, q, r, tolerance=1e-10)
    reference = scipy.linalg.solve_discrete_are(a, b, q, r)
    np.testing.assert_allclose(result.matrix, reference, rtol=2e-8, atol=2e-8)
    lqr = discrete_lqr(a, b, q, r, tolerance=1e-10)
    closed_loop = a + b @ np.asarray(lqr.feedback_gain)
    assert np.max(np.abs(np.linalg.eigvals(closed_loop))) < 1.0
    assert bool(result.valid)
    assert result.diagnostics.equation.system_type == "discrete"


def test_unstabilizable_and_undetectable_modes_have_explicit_status():
    continuous_unstabilizable = solve_continuous_are(
        jnp.diag(jnp.array([1.0, -1.0])),
        jnp.array([[0.0], [1.0]]),
        jnp.eye(2),
        jnp.ones((1, 1)),
    )
    assert not bool(continuous_unstabilizable.diagnostics.stabilizable)
    assert int(continuous_unstabilizable.status) == RiccatiStatus.UNSTABILIZABLE
    assert not bool(continuous_unstabilizable.valid)

    discrete_undetectable = solve_discrete_are(
        jnp.diag(jnp.array([1.1, 0.5])),
        jnp.eye(2),
        jnp.diag(jnp.array([0.0, 1.0])),
        jnp.eye(2),
    )
    assert bool(discrete_undetectable.diagnostics.stabilizable)
    assert not bool(discrete_undetectable.diagnostics.detectable)
    assert int(discrete_undetectable.status) == RiccatiStatus.UNDETECTABLE
    assert not bool(discrete_undetectable.valid)


def test_invalid_infinite_lqr_results_retain_only_raw_policy_evidence():
    one = jnp.ones((1, 1))
    zero = jnp.zeros((1, 1))

    continuous = eqx.filter_jit(continuous_lqr)(zero, zero, one, one)
    discrete = eqx.filter_jit(discrete_lqr)(
        jnp.array([[1.1]]),
        zero,
        one,
        one,
        max_iterations=8,
    )

    for result in (continuous, discrete):
        assert not bool(result.valid)
        assert result.policy is not None
        assert result.feedback_gain is not None
        assert result.feedback_gain.shape == (1, 1)
    assert int(continuous.status) == RiccatiStatus.UNSTABILIZABLE
    assert int(discrete.status) == RiccatiStatus.UNSTABILIZABLE


def test_batched_infinite_lqr_retains_independent_valid_case_policy():
    result = continuous_lqr(
        jnp.array([[[-1.0]], [[0.0]]]),
        jnp.array([[[1.0]], [[0.0]]]),
        jnp.ones((2, 1, 1)),
        jnp.ones((2, 1, 1)),
    )

    np.testing.assert_array_equal(result.valid, jnp.array([True, False]))
    assert result.policy is not None
    assert result.feedback_gain is not None
    assert bool(jnp.isfinite(result.feedback_gain[0, 0, 0]))
    assert int(result.status[1]) == RiccatiStatus.UNSTABILIZABLE


def test_riccati_equation_convergence_excludes_outer_structural_diagnosis():
    result = solve_continuous_are(
        jnp.ones((1, 1)),
        jnp.ones((1, 1)),
        jnp.zeros((1, 1)),
        jnp.ones((1, 1)),
    )

    assert bool(result.diagnostics.equation.converged)
    assert bool(result.diagnostics.equation.successful)
    assert not bool(result.diagnostics.detectable)
    assert not bool(result.diagnostics.converged)
    assert not bool(result.valid)
    assert int(result.status) == RiccatiStatus.UNDETECTABLE


def test_finite_lqr_nonfinite_value_constants_and_residuals_have_nonfinite_status():
    horizon = 2
    stage = jnp.ones((horizon, 1, 1))
    huge = jnp.asarray(0.75 * np.finfo(np.float64).max)
    result = finite_horizon_lqr(
        stage,
        jnp.zeros_like(stage),
        stage,
        stage,
        jnp.ones((1, 1)),
        stage_constants=jnp.full((horizon,), huge),
    )

    assert bool(jnp.any(~jnp.isfinite(result.value.constants)))
    assert bool(jnp.any(~jnp.isfinite(result.diagnostics.riccati_residuals)))
    assert not bool(result.diagnostics.finite)
    assert not bool(result.valid)
    assert int(result.status) == RiccatiStatus.NONFINITE


def test_singular_or_indefinite_costs_are_rejected_without_regularization():
    stage = jnp.ones((2, 1, 1))
    with pytest.raises(eqx.EquinoxRuntimeError, match="singular control costs"):
        finite_horizon_lqr(stage, stage, stage, jnp.zeros_like(stage), jnp.ones((1, 1)))
    with pytest.raises(eqx.EquinoxRuntimeError, match="indefinite costs"):
        solve_continuous_are(
            jnp.array([[-1.0]]),
            jnp.ones((1, 1)),
            jnp.array([[-1.0]]),
            jnp.ones((1, 1)),
        )


def test_affine_time_varying_riccati_matches_full_kkt_block_solve():
    a = jnp.array(
        [
            [[1.0, 0.2], [0.0, 1.0]],
            [[1.0, 0.25], [0.0, 0.95]],
            [[1.0, 0.15], [0.0, 0.9]],
        ]
    )
    b = jnp.array([[[0.02], [0.2]], [[0.03], [0.25]], [[0.01], [0.15]]])
    c = jnp.array([[0.1, -0.03], [0.0, 0.04], [-0.05, 0.02]])
    q = jnp.stack((jnp.diag(jnp.array([1.0, 0.4])),) * 3)
    r = jnp.array([[[1.2]], [[0.9]], [[1.1]]])
    cross = jnp.array([[[0.04], [0.01]], [[0.02], [-0.01]], [[0.03], [0.0]]])
    q_linear = jnp.array([[-1.0, 0.2], [-0.8, 0.1], [-0.6, -0.1]])
    r_linear = jnp.array([[0.2], [-0.1], [0.15]])
    qf = jnp.diag(jnp.array([2.0, 0.7]))
    qf_linear = jnp.array([-1.5, 0.3])
    x0 = jnp.array([0.4, -0.2])
    time_grid = phx.dynamics.TimeGrid(
        jnp.array([0.0, 0.3, 0.7, 1.0]), time_id="affine-kkt-grid"
    )
    result = finite_horizon_lqr(
        a,
        b,
        q,
        r,
        qf,
        dynamics_bias=c,
        state_control_cross=cross,
        state_linear=q_linear,
        control_linear=r_linear,
        terminal_linear=qf_linear,
        terminal_constant=0.7,
        time_grid=time_grid,
    )
    states, controls = _policy_rollout(a, b, c, result.policy, x0)
    oracle_states, oracle_controls = _equality_kkt_solution(
        a, b, c, q, r, cross, q_linear, r_linear, qf, qf_linear, x0
    )
    np.testing.assert_allclose(states, oracle_states, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(controls, oracle_controls, rtol=2e-10, atol=2e-10)
    assert result.diagnostics.method == "sequential-riccati"
    assert result.diagnostics.maximum_kkt_residual < 1e-10
    assert bool(result.valid)


def test_affine_tracking_feedback_rolls_out_through_control_foundation():
    horizon = 8
    time_grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 1.0, horizon + 1), time_id="lqr-rollout-grid"
    )
    a = jnp.ones((horizon, 1, 1))
    b = jnp.ones((horizon, 1, 1))
    q = jnp.ones((horizon, 1, 1))
    r = 0.2 * jnp.ones((horizon, 1, 1))
    target = 2.0
    result = finite_horizon_lqr(
        a,
        b,
        q,
        r,
        4.0 * jnp.ones((1, 1)),
        state_linear=-target * q[..., 0],
        terminal_linear=jnp.array([-4.0 * target]),
        time_grid=time_grid,
        policy_id="tracking-policy",
    )
    dynamics = make_discrete_control_dynamics(
        lambda time, state, control, args: state + control,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="tracking-integrator",
    )
    problem = phx.control.ControlProblem(
        dynamics,
        time_grid,
        jnp.array([0.0]),
        problem_id="tracking-rollout",
    )
    trajectory = problem.rollout(result.policy, jnp.asarray(0.0))
    assert trajectory.states.shape == (horizon + 1, 1)
    assert trajectory.controls.shape == (horizon, 1)
    assert bool(trajectory.valid[-1])
    assert abs(float(trajectory.states[-1, 0]) - target) < 0.1
    assert result.policy.parameter_shape == ()
    with pytest.raises(ValueError, match="cannot be sampled without states"):
        result.policy.sample(jnp.asarray(0.0), time_grid.times)


def test_long_finite_horizon_converges_to_discrete_infinite_horizon_gain():
    horizon = 100
    a = jnp.broadcast_to(jnp.array([[1.1]]), (horizon, 1, 1))
    b = jnp.ones((horizon, 1, 1))
    q = jnp.ones((horizon, 1, 1))
    r = 0.7 * jnp.ones((horizon, 1, 1))
    finite = finite_horizon_lqr(a, b, q, r, jnp.ones((1, 1)))
    infinite = discrete_lqr(a[0], b[0], q[0], r[0], tolerance=1e-11)
    np.testing.assert_allclose(
        finite.feedback_gain[0], infinite.feedback_gain, rtol=2e-9, atol=2e-9
    )


def test_finite_lqr_preserves_explicit_case_and_time_axes():
    horizon = 3
    case_shape = (2,)
    a = jnp.array([1.0, 0.8])[:, None, None, None] * jnp.ones(
        case_shape + (horizon, 1, 1)
    )
    b = jnp.ones(case_shape + (horizon, 1, 1))
    q = jnp.ones(case_shape + (horizon, 1, 1))
    r = jnp.ones(case_shape + (horizon, 1, 1))
    qf = jnp.ones(case_shape + (1, 1))
    result = finite_horizon_lqr(a, b, q, r, qf)

    assert result.feedback_gain.shape == case_shape + (horizon, 1, 1)
    assert result.feedforward.shape == case_shape + (horizon, 1)
    assert result.value.matrices.shape == case_shape + (horizon + 1, 1, 1)
    assert result.diagnostics.kkt_residuals.shape == case_shape + (horizon,)
    assert result.status.shape == case_shape
    assert bool(jnp.all(result.valid))
    controls = result.policy.evaluate(
        jnp.zeros(case_shape),
        jnp.asarray(0.0),
        case_shape=case_shape,
        state=jnp.array([[1.0], [2.0]]),
    )
    assert controls.shape == case_shape + (1,)


def test_are_implicit_gradients_match_centered_direct_differences():
    b = jnp.ones((1, 1))
    q = jnp.ones((1, 1))
    r = jnp.ones((1, 1))

    def continuous_value(rate):
        return solve_continuous_are(rate.reshape(1, 1), b, q, r).matrix[0, 0]

    def discrete_value(rate):
        return solve_discrete_are(rate.reshape(1, 1), b, q, r, tolerance=1e-11).matrix[
            0, 0
        ]

    step = 1e-4
    continuous_gradient = jax.grad(continuous_value)(jnp.asarray(0.7))
    continuous_reference = (
        continuous_value(jnp.asarray(0.7 + step))
        - continuous_value(jnp.asarray(0.7 - step))
    ) / (2.0 * step)
    discrete_gradient = jax.grad(discrete_value)(jnp.asarray(1.1))
    discrete_reference = (
        discrete_value(jnp.asarray(1.1 + step)) - discrete_value(jnp.asarray(1.1 - step))
    ) / (2.0 * step)
    np.testing.assert_allclose(continuous_gradient, continuous_reference, rtol=2e-6)
    np.testing.assert_allclose(discrete_gradient, discrete_reference, rtol=2e-5)
