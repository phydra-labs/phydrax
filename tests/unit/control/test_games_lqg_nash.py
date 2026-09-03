#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control.games._layout import PlayerControlPartition
from phydrax.control.games._linear_quadratic import finite_horizon_lq_feedback_nash
from phydrax.control.games._lqg import (
    finite_horizon_lqg_feedback_nash,
    LQGFeedbackNashStatus,
)


def _two_player_scalar_game(case_shape=()):
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    a = jnp.broadcast_to(jnp.asarray([[[1.0]]]), case_shape + (1, 1, 1))
    b = jnp.broadcast_to(jnp.asarray([[[1.0, 1.0]]]), case_shape + (1, 1, 2))
    q = jnp.broadcast_to(jnp.zeros((2, 1, 1, 1)), case_shape + (2, 1, 1, 1))
    r = jnp.broadcast_to(
        jnp.stack((jnp.eye(2), jnp.eye(2)))[:, None, :, :],
        case_shape + (2, 1, 2, 2),
    )
    qf = jnp.broadcast_to(
        jnp.asarray([[[2.0]], [[4.0]]]),
        case_shape + (2, 1, 1),
    )
    return a, b, q, r, qf, partition


def test_zero_noise_exactly_preserves_the_deterministic_feedback_nash_result():
    a, b, q, r, qf, partition = _two_player_scalar_game()
    state_linear = jnp.asarray([[[0.2]], [[-0.3]]])
    control_linear = jnp.asarray([[[0.1, -0.2]], [[0.3, 0.4]]])
    stage_constants = jnp.asarray([[0.5], [-0.25]])
    terminal_linear = jnp.asarray([[0.4], [-0.6]])
    terminal_constants = jnp.asarray([0.7, 1.2])
    deterministic = finite_horizon_lq_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        state_linear=state_linear,
        control_linear=control_linear,
        stage_constants=stage_constants,
        terminal_linear=terminal_linear,
        terminal_constants=terminal_constants,
        policy_id="zero-noise-nash",
    )

    result = finite_horizon_lqg_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        process_noise_factors=jnp.ones((1, 1, 1)),
        process_noise_covariances=jnp.zeros((1, 1, 1)),
        state_linear=state_linear,
        control_linear=control_linear,
        stage_constants=stage_constants,
        terminal_linear=terminal_linear,
        terminal_constants=terminal_constants,
        policy_id="zero-noise-nash",
    )

    assert bool(result.valid)
    assert int(result.status) == int(LQGFeedbackNashStatus.SUCCESS)
    np.testing.assert_array_equal(
        result.deterministic_result.status,
        deterministic.status,
    )
    np.testing.assert_array_equal(
        result.deterministic_result.valid,
        deterministic.valid,
    )
    np.testing.assert_array_equal(result.feedback_gain, deterministic.feedback_gain)
    np.testing.assert_array_equal(result.feedforward, deterministic.feedforward)
    np.testing.assert_array_equal(result.trace_increments, jnp.zeros((2, 1)))
    np.testing.assert_array_equal(
        result.value_constant_corrections,
        jnp.zeros((2, 2)),
    )
    for stochastic_value, nested_value, deterministic_value in zip(
        result.values,
        result.deterministic_result.values,
        deterministic.values,
        strict=True,
    ):
        np.testing.assert_array_equal(
            stochastic_value.matrices,
            deterministic_value.matrices,
        )
        np.testing.assert_array_equal(
            stochastic_value.linear,
            deterministic_value.linear,
        )
        np.testing.assert_array_equal(
            stochastic_value.constants,
            deterministic_value.constants,
        )
        np.testing.assert_array_equal(
            nested_value.constants,
            deterministic_value.constants,
        )


def test_common_noise_corrects_each_player_constant_and_initial_cost_separately():
    a, b, q, r, qf, partition = _two_player_scalar_game()
    result = finite_horizon_lqg_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        process_noise_factors=jnp.asarray([[[2.0]]]),
        process_noise_covariances=jnp.asarray([[[0.25]]]),
        initial_mean=jnp.asarray([0.5]),
        initial_covariance=jnp.asarray([[0.5]]),
    )

    np.testing.assert_allclose(result.process_covariances[..., 0, 0], [1.0])
    np.testing.assert_allclose(result.trace_increments, [[1.0], [2.0]])
    np.testing.assert_allclose(
        result.value_constant_corrections,
        [[1.0, 0.0], [2.0, 0.0]],
    )
    for player, expected_increment in enumerate((1.0, 2.0)):
        np.testing.assert_allclose(
            result.values[player].constants,
            result.deterministic_result.values[player].constants
            + jnp.asarray([expected_increment, 0.0]),
        )
    initial_matrices = jnp.asarray(
        [value.matrices[0, 0, 0] for value in result.deterministic_result.values]
    )
    np.testing.assert_allclose(
        result.initial_covariance_cost,
        0.25 * initial_matrices,
    )
    expected = (
        jnp.asarray([value.evaluate(0.0, jnp.asarray([0.5])) for value in result.values])
        + result.initial_covariance_cost
    )
    np.testing.assert_allclose(result.initial_expected_cost, expected)


def test_game_case_axes_remain_distinct_from_the_player_axis():
    case_shape = (2, 3)
    a, b, q, r, qf, partition = _two_player_scalar_game(case_shape)
    factors = jnp.ones(case_shape + (1, 1, 1))
    covariances = jnp.ones(case_shape + (1, 1, 1))
    result = finite_horizon_lqg_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        process_noise_factors=factors,
        process_noise_covariances=covariances,
        initial_mean=jnp.zeros(case_shape + (1,)),
        initial_covariance=jnp.broadcast_to(jnp.eye(1), case_shape + (1, 1)),
    )

    assert result.feedback_gain.shape == case_shape + (1, 2, 1)
    assert result.process_covariances.shape == case_shape + (1, 1, 1)
    assert result.trace_increments.shape == case_shape + (2, 1)
    assert result.value_constant_corrections.shape == case_shape + (2, 2)
    assert result.initial_covariance_cost.shape == case_shape + (2,)
    assert result.initial_expected_cost.shape == case_shape + (2,)
    assert all(value.case_shape == case_shape for value in result.values)

    with pytest.raises(ValueError, match="process_noise_factors must have shape"):
        finite_horizon_lqg_feedback_nash(
            a,
            b,
            q,
            r,
            qf,
            partition,
            process_noise_factors=jnp.ones(case_shape + (2, 1, 1, 1)),
            process_noise_covariances=covariances,
        )
    with pytest.raises(ValueError, match="process_noise_covariances must have shape"):
        finite_horizon_lqg_feedback_nash(
            a,
            b,
            q,
            r,
            qf,
            partition,
            process_noise_factors=factors,
            process_noise_covariances=jnp.ones(case_shape + (2, 1, 1, 1)),
        )


def test_game_rejects_non_psd_and_nonfinite_noise_covariances_before_solving():
    a, b, q, r, qf, partition = _two_player_scalar_game()
    factors = jnp.ones((1, 1, 1))

    with pytest.raises(eqx.EquinoxRuntimeError, match="positive semidefinite"):
        finite_horizon_lqg_feedback_nash(
            a,
            b,
            q,
            r,
            qf,
            partition,
            process_noise_factors=factors,
            process_noise_covariances=jnp.asarray([[[-1.0]]]),
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite"):
        finite_horizon_lqg_feedback_nash(
            a,
            b,
            q,
            r,
            qf,
            partition,
            process_noise_factors=factors,
            process_noise_covariances=jnp.asarray([[[jnp.nan]]]),
        )
