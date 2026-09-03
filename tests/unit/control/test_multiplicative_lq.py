#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control import finite_horizon_lqr
from phydrax.control.stochastic._multiplicative_lq import (
    finite_horizon_multiplicative_lq_state_feedback,
    MultiplicativeLQStateFeedbackStatus,
)
from phydrax.dynamics import TimeGrid


def test_scalar_affine_recursion_matches_direct_correlated_noise_algebra():
    a = 1.2
    b = 0.7
    c = 0.3
    q = 0.8
    r = 1.1
    cross = 0.1
    q_linear = -0.2
    r_linear = 0.05
    stage_constant = 0.4
    terminal_p = 1.5
    terminal_linear = 0.2
    terminal_constant = 0.7
    state_noise = np.asarray([0.4, -0.2])
    control_noise = np.asarray([0.5, 0.25])
    noise_bias = np.asarray([0.1, -0.15])
    gamma = np.asarray([[1.0, 0.3], [0.3, 2.0]])

    h = (
        r
        + b * terminal_p * b
        + np.einsum("r,s,rs->", control_noise * terminal_p, control_noise, gamma)
    )
    w = (
        cross
        + b * terminal_p * a
        + np.einsum("r,s,rs->", control_noise * terminal_p, state_noise, gamma)
    )
    g = (
        r_linear
        + b * (terminal_p * c + terminal_linear)
        + np.einsum("r,s,rs->", control_noise * terminal_p, noise_bias, gamma)
    )
    expected_feedback = -w / h
    expected_feedforward = -g / h
    state_hessian = (
        q
        + a * terminal_p * a
        + np.einsum("r,s,rs->", state_noise * terminal_p, state_noise, gamma)
    )
    state_affine = (
        q_linear
        + a * (terminal_p * c + terminal_linear)
        + np.einsum("r,s,rs->", state_noise * terminal_p, noise_bias, gamma)
    )
    delta = (
        stage_constant
        + terminal_constant
        + 0.5 * c * terminal_p * c
        + terminal_linear * c
        + 0.5 * np.einsum("r,s,rs->", noise_bias * terminal_p, noise_bias, gamma)
    )
    expected_p = state_hessian + w * expected_feedback
    expected_linear = state_affine + w * expected_feedforward
    expected_constant = delta + 0.5 * g * expected_feedforward
    closed_noise_bias = noise_bias + control_noise * expected_feedforward
    expected_trace = 0.5 * np.einsum(
        "r,s,rs->", closed_noise_bias * terminal_p, closed_noise_bias, gamma
    )

    result = finite_horizon_multiplicative_lq_state_feedback(
        jnp.asarray([[[a]]]),
        jnp.asarray([[[b]]]),
        jnp.asarray([[[q]]]),
        jnp.asarray([[[r]]]),
        jnp.asarray([[terminal_p]]),
        state_noise_matrices=jnp.asarray(state_noise)[None, :, None, None],
        control_noise_matrices=jnp.asarray(control_noise)[None, :, None, None],
        noise_covariances=jnp.asarray(gamma)[None, ...],
        dynamics_bias=jnp.asarray([[c]]),
        noise_bias=jnp.asarray(noise_bias)[None, :, None],
        state_control_cross=jnp.asarray([[[cross]]]),
        state_linear=jnp.asarray([[q_linear]]),
        control_linear=jnp.asarray([[r_linear]]),
        stage_constants=jnp.asarray([stage_constant]),
        terminal_linear=jnp.asarray([terminal_linear]),
        terminal_constant=terminal_constant,
    )

    assert bool(result.valid)
    np.testing.assert_allclose(result.feedback_gain[0, 0, 0], expected_feedback)
    np.testing.assert_allclose(result.feedforward[0, 0], expected_feedforward)
    np.testing.assert_allclose(result.value.matrices[0, 0, 0], expected_p)
    np.testing.assert_allclose(result.value.linear[0, 0], expected_linear)
    np.testing.assert_allclose(result.value.constants[0], expected_constant)
    np.testing.assert_allclose(result.trace_increments[0], expected_trace)
    assert result.diagnostics.maximum_stationarity_residual < 1e-12
    assert result.diagnostics.maximum_bellman_residual < 1e-12


def test_zero_noise_reduces_to_exact_finite_horizon_lqr():
    horizon = 3
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

    result = finite_horizon_multiplicative_lq_state_feedback(
        a,
        b,
        q,
        r,
        qf,
        state_noise_matrices=jnp.zeros((horizon, 2, 1, 1)),
        control_noise_matrices=jnp.zeros((horizon, 2, 1, 1)),
        noise_covariances=jnp.broadcast_to(jnp.eye(2), (horizon, 2, 2)),
        dynamics_bias=c,
        noise_bias=jnp.zeros((horizon, 2, 1)),
        state_control_cross=cross,
        state_linear=q_linear,
        control_linear=r_linear,
        stage_constants=constants,
        terminal_linear=qf_linear,
        terminal_constant=0.7,
    )
    deterministic = finite_horizon_lqr(
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
    )

    assert bool(result.valid)
    np.testing.assert_allclose(result.feedback_gain, deterministic.feedback_gain)
    np.testing.assert_allclose(result.feedforward, deterministic.feedforward)
    np.testing.assert_allclose(result.value.matrices, deterministic.value.matrices)
    np.testing.assert_allclose(result.value.linear, deterministic.value.linear)
    np.testing.assert_allclose(result.value.constants, deterministic.value.constants)
    np.testing.assert_array_equal(result.trace_increments, jnp.zeros(horizon))


def test_additive_bias_noise_has_certainty_equivalent_gains_and_exact_traces():
    horizon = 3
    a = jnp.asarray([[[1.0]], [[0.9]], [[1.1]]])
    b = jnp.asarray([[[0.8]], [[1.0]], [[0.7]]])
    q = jnp.asarray([[[1.2]], [[0.8]], [[1.4]]])
    r = jnp.asarray([[[2.0]], [[1.6]], [[2.4]]])
    qf = jnp.asarray([[2.5]])
    noise_bias = jnp.asarray([[[0.2], [-0.1]], [[0.3], [0.4]], [[-0.2], [0.1]]])
    gamma = jnp.broadcast_to(jnp.asarray([[1.0, 0.25], [0.25, 0.5]]), (horizon, 2, 2))

    result = finite_horizon_multiplicative_lq_state_feedback(
        a,
        b,
        q,
        r,
        qf,
        state_noise_matrices=jnp.zeros((horizon, 2, 1, 1)),
        control_noise_matrices=jnp.zeros((horizon, 2, 1, 1)),
        noise_covariances=gamma,
        noise_bias=noise_bias,
    )
    deterministic = finite_horizon_lqr(a, b, q, r, qf)
    expected_traces = 0.5 * jnp.einsum(
        "tri,tij,tsj,trs->t",
        noise_bias,
        deterministic.value.matrices[1:],
        noise_bias,
        gamma,
    )
    expected_correction = jnp.flip(jnp.cumsum(jnp.flip(expected_traces)))

    np.testing.assert_allclose(result.feedback_gain, deterministic.feedback_gain)
    np.testing.assert_allclose(result.feedforward, deterministic.feedforward)
    np.testing.assert_allclose(result.trace_increments, expected_traces)
    np.testing.assert_allclose(
        result.value.constants[:-1],
        deterministic.value.constants[:-1] + expected_correction,
    )
    np.testing.assert_allclose(
        result.value.constants[-1], deterministic.value.constants[-1]
    )


def test_control_noise_curvature_and_state_noise_change_the_expected_policy():
    common = dict(
        dynamics_matrices=jnp.ones((2, 1, 1)),
        control_matrices=jnp.ones((2, 1, 1)),
        state_costs=jnp.zeros((2, 1, 1)),
        control_costs=jnp.ones((2, 1, 1)),
        terminal_state_cost=jnp.ones((1, 1)),
        noise_covariances=jnp.ones((2, 1, 1)),
    )
    no_noise = finite_horizon_multiplicative_lq_state_feedback(
        common["dynamics_matrices"],
        common["control_matrices"],
        common["state_costs"],
        common["control_costs"],
        common["terminal_state_cost"],
        state_noise_matrices=jnp.zeros((2, 1, 1, 1)),
        control_noise_matrices=jnp.zeros((2, 1, 1, 1)),
        noise_covariances=common["noise_covariances"],
    )
    state_noise_only = finite_horizon_multiplicative_lq_state_feedback(
        common["dynamics_matrices"],
        common["control_matrices"],
        common["state_costs"],
        common["control_costs"],
        common["terminal_state_cost"],
        state_noise_matrices=jnp.full((2, 1, 1, 1), 0.6),
        control_noise_matrices=jnp.zeros((2, 1, 1, 1)),
        noise_covariances=common["noise_covariances"],
    )
    control_noise = finite_horizon_multiplicative_lq_state_feedback(
        common["dynamics_matrices"],
        common["control_matrices"],
        common["state_costs"],
        common["control_costs"],
        common["terminal_state_cost"],
        state_noise_matrices=jnp.full((2, 1, 1, 1), 0.6),
        control_noise_matrices=jnp.full((2, 1, 1, 1), 0.8),
        noise_covariances=common["noise_covariances"],
    )

    assert not np.isclose(
        state_noise_only.feedback_gain[0, 0, 0], no_noise.feedback_gain[0, 0, 0]
    )
    assert not np.isclose(
        control_noise.feedback_gain[-1, 0, 0],
        state_noise_only.feedback_gain[-1, 0, 0],
    )
    np.testing.assert_allclose(
        control_noise.diagnostics.control_minimum_eigenvalues[-1],
        1.0 + 1.0 + 0.8**2,
    )


def test_case_axes_jit_and_autodiff_preserve_noise_dependence():
    cases = 2
    a = jnp.ones((cases, 1, 1, 1))
    b = jnp.ones((cases, 1, 1, 1))
    q = jnp.zeros((cases, 1, 1, 1))
    r = jnp.ones((cases, 1, 1, 1))
    qf = jnp.ones((cases, 1, 1))
    state_noise = jnp.asarray([0.3, 0.5])[:, None, None, None, None]
    gamma = jnp.ones((cases, 1, 1, 1))
    time_grid = TimeGrid(jnp.asarray([0.0, 1.0]), time_id="multiplicative-cases")

    def solve(control_scale):
        control_noise = control_scale * jnp.asarray([0.2, 0.4])[:, None, None, None, None]
        return finite_horizon_multiplicative_lq_state_feedback(
            a,
            b,
            q,
            r,
            qf,
            state_noise_matrices=state_noise,
            control_noise_matrices=control_noise,
            noise_covariances=gamma,
            time_grid=time_grid,
        )

    result = eqx.filter_jit(solve)(jnp.asarray(1.0))
    gradient = jax.jit(jax.grad(lambda scale: solve(scale).feedback_gain[0, 0, 0, 0]))(
        jnp.asarray(1.0)
    )

    assert result.feedback_gain.shape == (cases, 1, 1, 1)
    assert result.value.matrices.shape == (cases, 2, 1, 1)
    assert result.noise_covariances.shape == (cases, 1, 1, 1)
    np.testing.assert_array_equal(result.valid, jnp.ones(cases, dtype=bool))
    assert np.isfinite(gradient)
    assert not np.isclose(gradient, 0.0)


def test_jitted_cases_report_covariance_curvature_and_nonfinite_failures():
    cases = 5
    a = jnp.ones((cases, 1, 1, 1)).at[3, 0, 0, 0].set(jnp.nan)
    b = jnp.ones((cases, 1, 1, 1))
    q = jnp.zeros((cases, 1, 1, 1))
    r = jnp.ones((cases, 1, 1, 1)).at[4, 0, 0, 0].set(-2.0)
    qf = jnp.ones((cases, 1, 1))
    gamma = jnp.broadcast_to(jnp.eye(2), (cases, 1, 2, 2))
    gamma = gamma.at[1, 0].set(jnp.asarray([[1.0, 0.2], [0.0, 1.0]]))
    gamma = gamma.at[2, 0].set(jnp.asarray([[1.0, 0.0], [0.0, -1.0]]))
    time_grid = TimeGrid(jnp.asarray([0.0, 1.0]), time_id="multiplicative-failures")

    result = eqx.filter_jit(finite_horizon_multiplicative_lq_state_feedback)(
        a,
        b,
        q,
        r,
        qf,
        state_noise_matrices=jnp.zeros((cases, 1, 2, 1, 1)),
        control_noise_matrices=jnp.zeros((cases, 1, 2, 1, 1)),
        noise_covariances=gamma,
        time_grid=time_grid,
    )
    expected = jnp.asarray(
        [
            MultiplicativeLQStateFeedbackStatus.SUCCESS,
            MultiplicativeLQStateFeedbackStatus.NOISE_COVARIANCE_NONSYMMETRIC,
            MultiplicativeLQStateFeedbackStatus.NOISE_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE,
            MultiplicativeLQStateFeedbackStatus.NONFINITE_INPUT,
            MultiplicativeLQStateFeedbackStatus.CONTROL_CURVATURE_NOT_POSITIVE_DEFINITE,
        ],
        dtype=jnp.int32,
    )

    np.testing.assert_array_equal(result.status, expected)
    np.testing.assert_array_equal(result.diagnostics.first_failed_stage, [-1, 0, 0, 0, 0])
    assert bool(result.valid[0])
    assert not bool(jnp.any(result.valid[1:]))


def test_structural_validation_rejects_implicit_noise_axes_and_bad_tolerances():
    required = (
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1)),
    )
    with pytest.raises(ValueError, match="state_noise_matrices must have shape"):
        finite_horizon_multiplicative_lq_state_feedback(
            *required,
            state_noise_matrices=jnp.zeros((1, 1, 1)),
            control_noise_matrices=jnp.zeros((1, 1, 1, 1)),
            noise_covariances=jnp.ones((1, 1, 1)),
        )
    with pytest.raises(ValueError, match="noise_covariances must have shape"):
        finite_horizon_multiplicative_lq_state_feedback(
            *required,
            state_noise_matrices=jnp.zeros((1, 1, 1, 1)),
            control_noise_matrices=jnp.zeros((1, 1, 1, 1)),
            noise_covariances=jnp.ones((1, 1)),
        )
    with pytest.raises(ValueError, match="covariance_tolerance"):
        finite_horizon_multiplicative_lq_state_feedback(
            *required,
            state_noise_matrices=jnp.zeros((1, 1, 1, 1)),
            control_noise_matrices=jnp.zeros((1, 1, 1, 1)),
            noise_covariances=jnp.ones((1, 1, 1)),
            covariance_tolerance=-1.0,
        )
