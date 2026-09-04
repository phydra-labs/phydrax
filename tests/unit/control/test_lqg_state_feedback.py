#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control._lqr import finite_horizon_lqr
from phydrax.control.stochastic._lqg import (
    finite_horizon_lqg_state_feedback,
    LQGStateFeedbackStatus,
)


def _scalar_problem():
    return (
        jnp.asarray([[[1.0]], [[1.0]]]),
        jnp.asarray([[[1.0]], [[1.0]]]),
        jnp.zeros((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.asarray([[2.0]]),
    )


def test_zero_noise_is_an_exact_reduction_to_affine_finite_horizon_lqr():
    a, b, q, r, qf = _scalar_problem()
    q = jnp.ones_like(q)
    c = jnp.asarray([[0.2], [-0.1]])
    cross = jnp.asarray([[[0.1]], [[-0.05]]])
    q_linear = jnp.asarray([[0.3], [-0.2]])
    r_linear = jnp.asarray([[0.2], [0.1]])
    constants = jnp.asarray([0.5, -0.2])
    terminal_linear = jnp.asarray([0.25])
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
        terminal_linear=terminal_linear,
        terminal_constant=0.7,
        policy_id="zero-noise-reduction",
    )

    result = finite_horizon_lqg_state_feedback(
        a,
        b,
        q,
        r,
        qf,
        process_noise_factors=jnp.ones((2, 1, 1)),
        process_noise_covariances=jnp.zeros((2, 1, 1)),
        dynamics_bias=c,
        state_control_cross=cross,
        state_linear=q_linear,
        control_linear=r_linear,
        stage_constants=constants,
        terminal_linear=terminal_linear,
        terminal_constant=0.7,
        policy_id="zero-noise-reduction",
    )

    assert bool(result.valid)
    assert int(result.status) == int(LQGStateFeedbackStatus.SUCCESS)
    np.testing.assert_array_equal(
        result.deterministic_result.status,
        deterministic.status,
    )
    np.testing.assert_array_equal(
        result.deterministic_result.valid,
        deterministic.valid,
    )
    np.testing.assert_array_equal(
        result.feedback_gain,
        result.deterministic_result.feedback_gain,
    )
    np.testing.assert_array_equal(result.feedback_gain, deterministic.feedback_gain)
    np.testing.assert_array_equal(result.feedforward, deterministic.feedforward)
    np.testing.assert_array_equal(
        result.deterministic_result.value.matrices,
        deterministic.value.matrices,
    )
    np.testing.assert_array_equal(
        result.deterministic_result.value.linear,
        deterministic.value.linear,
    )
    np.testing.assert_array_equal(result.value.matrices, deterministic.value.matrices)
    np.testing.assert_array_equal(result.value.linear, deterministic.value.linear)
    np.testing.assert_array_equal(result.trace_increments, jnp.zeros((2,)))
    np.testing.assert_array_equal(
        result.value_constant_corrections,
        jnp.zeros((3,)),
    )
    np.testing.assert_array_equal(result.value.constants, deterministic.value.constants)


def test_scalar_trace_recursion_and_initial_gaussian_cost_are_analytic():
    a, b, q, r, qf = _scalar_problem()
    result = finite_horizon_lqg_state_feedback(
        a,
        b,
        q,
        r,
        qf,
        process_noise_factors=jnp.asarray([[[2.0]], [[1.0]]]),
        process_noise_covariances=jnp.asarray([[[0.75]], [[5.0]]]),
        initial_mean=jnp.asarray([1.5]),
        initial_covariance=jnp.asarray([[0.25]]),
    )

    np.testing.assert_allclose(
        result.deterministic_result.value.matrices[:, 0, 0],
        [2.0 / 5.0, 2.0 / 3.0, 2.0],
        rtol=2e-7,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        result.process_covariances[:, 0, 0],
        [3.0, 5.0],
        rtol=2e-7,
        atol=2e-7,
    )
    np.testing.assert_allclose(result.trace_increments, [1.0, 5.0], rtol=2e-7)
    np.testing.assert_allclose(
        result.value_constant_corrections,
        [6.0, 5.0, 0.0],
        rtol=2e-7,
    )
    np.testing.assert_allclose(result.value.constants, [6.0, 5.0, 0.0], rtol=2e-7)
    np.testing.assert_allclose(result.initial_covariance_cost, 0.05, rtol=2e-7)
    np.testing.assert_allclose(result.initial_expected_cost, 6.5, rtol=2e-7)
    np.testing.assert_array_equal(result.covariance_symmetry_residuals, [0.0, 0.0])
    np.testing.assert_allclose(
        result.covariance_minimum_eigenvalues,
        [0.75, 5.0],
    )


def test_case_axes_are_preserved_without_time_or_case_broadcasting():
    case_shape = (2, 3)
    horizon = 2
    a = jnp.broadcast_to(jnp.ones((horizon, 1, 1)), case_shape + (horizon, 1, 1))
    b = jnp.broadcast_to(jnp.ones((horizon, 1, 1)), case_shape + (horizon, 1, 1))
    q = jnp.broadcast_to(jnp.zeros((horizon, 1, 1)), case_shape + (horizon, 1, 1))
    r = jnp.broadcast_to(jnp.ones((horizon, 1, 1)), case_shape + (horizon, 1, 1))
    qf = jnp.broadcast_to(jnp.asarray([[2.0]]), case_shape + (1, 1))
    factors = jnp.broadcast_to(
        jnp.ones((horizon, 1, 1)),
        case_shape + (horizon, 1, 1),
    )
    covariances = jnp.broadcast_to(
        jnp.ones((horizon, 1, 1)),
        case_shape + (horizon, 1, 1),
    )
    initial_mean = jnp.zeros(case_shape + (1,))
    initial_covariance = jnp.broadcast_to(jnp.eye(1), case_shape + (1, 1))

    result = finite_horizon_lqg_state_feedback(
        a,
        b,
        q,
        r,
        qf,
        process_noise_factors=factors,
        process_noise_covariances=covariances,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
    )

    assert result.feedback_gain.shape == case_shape + (horizon, 1, 1)
    assert result.trace_increments.shape == case_shape + (horizon,)
    assert result.process_covariances.shape == case_shape + (horizon, 1, 1)
    assert result.value_constant_corrections.shape == case_shape + (horizon + 1,)
    assert result.initial_expected_cost.shape == case_shape
    assert result.covariance_finite.shape == case_shape
    assert result.initial_covariance_finite.shape == case_shape

    with pytest.raises(ValueError, match="process_noise_covariances must have shape"):
        finite_horizon_lqg_state_feedback(
            a,
            b,
            q,
            r,
            qf,
            process_noise_factors=factors,
            process_noise_covariances=jnp.ones((horizon, 1, 1)),
        )


def test_malformed_indefinite_and_nonfinite_covariance_data_are_rejected():
    a, b, q, r, qf = _scalar_problem()
    factors = jnp.ones((2, 1, 1))
    covariances = jnp.ones((2, 1, 1))

    with pytest.raises(ValueError, match="process_noise_factors must have shape"):
        finite_horizon_lqg_state_feedback(
            a,
            b,
            q,
            r,
            qf,
            process_noise_factors=jnp.ones((2, 1, 1, 1)),
            process_noise_covariances=covariances,
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="positive semidefinite"):
        finite_horizon_lqg_state_feedback(
            a,
            b,
            q,
            r,
            qf,
            process_noise_factors=factors,
            process_noise_covariances=jnp.asarray([[[1.0]], [[-0.1]]]),
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="symmetric"):
        finite_horizon_lqg_state_feedback(
            a,
            b,
            q,
            r,
            qf,
            process_noise_factors=jnp.ones((2, 1, 2)),
            process_noise_covariances=jnp.asarray(
                [
                    [[1.0, 0.2], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ]
            ),
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite"):
        finite_horizon_lqg_state_feedback(
            a,
            b,
            q,
            r,
            qf,
            process_noise_factors=factors.at[0, 0, 0].set(jnp.inf),
            process_noise_covariances=covariances,
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite"):
        finite_horizon_lqg_state_feedback(
            a,
            b,
            q,
            r,
            qf,
            process_noise_factors=factors,
            process_noise_covariances=covariances.at[1, 0, 0].set(jnp.nan),
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="positive semidefinite"):
        finite_horizon_lqg_state_feedback(
            a,
            b,
            q,
            r,
            qf,
            process_noise_factors=factors,
            process_noise_covariances=covariances,
            initial_covariance=jnp.asarray([[-1.0]]),
        )
