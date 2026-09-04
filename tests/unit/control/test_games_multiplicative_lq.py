#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.control.games._layout import PlayerControlPartition
from phydrax.control.games._linear_quadratic import finite_horizon_lq_feedback_nash
from phydrax.control.games._multiplicative_lq import (
    finite_horizon_multiplicative_lq_feedback_nash,
    MultiplicativeLQFeedbackNashStatus,
)
from phydrax.control.stochastic._multiplicative_lq import (
    finite_horizon_multiplicative_lq_state_feedback,
)
from phydrax.dynamics import TimeGrid


def test_correlated_noise_rows_match_direct_two_player_stationarity_system():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    a = 1.1
    b = np.asarray([0.8, -0.4])
    c = 0.2
    state_noise = np.asarray([0.3, -0.15])
    control_noise = np.asarray([[0.5, -0.2], [0.1, 0.4]])
    noise_bias = np.asarray([0.25, -0.1])
    gamma = np.asarray([[1.0, 0.35], [0.35, 0.7]])
    terminal_p = np.asarray([1.2, 2.0])
    terminal_linear = np.asarray([0.1, -0.3])
    control_costs = np.asarray([[[1.4, 0.2], [0.2, 1.1]], [[0.9, -0.1], [-0.1, 1.6]]])
    cross = np.asarray([[0.15, -0.05], [-0.2, 0.3]])
    control_linear = np.asarray([[0.1, -0.2], [0.25, 0.05]])

    h = []
    w = []
    g = []
    for player in range(2):
        h.append(
            control_costs[player]
            + terminal_p[player] * np.outer(b, b)
            + terminal_p[player]
            * np.einsum("ri,sj,rs->ij", control_noise, control_noise, gamma)
        )
        w.append(
            cross[player]
            + terminal_p[player] * b * a
            + terminal_p[player]
            * np.einsum("ri,s,rs->i", control_noise, state_noise, gamma)
        )
        g.append(
            control_linear[player]
            + b * (terminal_p[player] * c + terminal_linear[player])
            + terminal_p[player]
            * np.einsum("ri,s,rs->i", control_noise, noise_bias, gamma)
        )
    coupled = np.stack((h[0][0], h[1][1]))
    expected_feedback = -np.linalg.solve(coupled, np.asarray([w[0][0], w[1][1]]))
    expected_feedforward = -np.linalg.solve(coupled, np.asarray([g[0][0], g[1][1]]))

    result = finite_horizon_multiplicative_lq_feedback_nash(
        jnp.asarray([[[a]]]),
        jnp.asarray(b)[None, None, :],
        jnp.zeros((2, 1, 1, 1)),
        jnp.asarray(control_costs)[:, None, :, :],
        jnp.asarray(terminal_p)[:, None, None],
        partition,
        state_noise_matrices=jnp.asarray(state_noise)[None, :, None, None],
        control_noise_matrices=jnp.asarray(control_noise)[None, :, None, :],
        noise_covariances=jnp.asarray(gamma)[None, ...],
        dynamics_bias=jnp.asarray([[c]]),
        noise_bias=jnp.asarray(noise_bias)[None, :, None],
        state_control_cross=jnp.asarray(cross)[:, None, None, :],
        control_linear=jnp.asarray(control_linear)[:, None, :],
        terminal_linear=jnp.asarray(terminal_linear)[:, None],
    )

    assert bool(result.valid)
    np.testing.assert_allclose(result.feedback_gain[0, :, 0], expected_feedback)
    np.testing.assert_allclose(result.feedforward[0], expected_feedforward)
    for player in range(2):
        np.testing.assert_allclose(
            result.diagnostics.own_control_minimum_eigenvalues[player, 0],
            h[player][player, player],
        )
    assert result.diagnostics.maximum_stationarity_residual < 1e-12
    assert result.diagnostics.maximum_bellman_residual < 1e-12


def test_control_noise_dtpd_term_changes_coupled_nash_answer():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    a = jnp.ones((1, 1, 1))
    b = jnp.asarray([[[1.0, 0.5]]])
    q = jnp.zeros((2, 1, 1, 1))
    r = jnp.broadcast_to(jnp.eye(2), (2, 1, 2, 2))
    qf = jnp.asarray([[[1.0]], [[1.5]]])
    common = dict(
        state_noise_matrices=jnp.full((1, 1, 1, 1), 0.25),
        noise_covariances=jnp.ones((1, 1, 1)),
    )
    no_control_noise = finite_horizon_multiplicative_lq_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        control_noise_matrices=jnp.zeros((1, 1, 1, 2)),
        **common,
    )
    with_control_noise = finite_horizon_multiplicative_lq_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        control_noise_matrices=jnp.asarray([[[[0.8, -0.3]]]]),
        **common,
    )

    assert bool(no_control_noise.valid)
    assert bool(with_control_noise.valid)
    assert not np.allclose(
        with_control_noise.feedback_gain, no_control_noise.feedback_gain
    )
    assert not np.allclose(
        with_control_noise.diagnostics.own_control_minimum_eigenvalues,
        no_control_noise.diagnostics.own_control_minimum_eigenvalues,
    )


def test_zero_noise_reduces_to_exact_deterministic_feedback_nash():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    horizon = 2
    a = jnp.asarray([[[1.0]], [[0.9]]])
    b = jnp.asarray([[[0.7, 0.4]], [[0.6, -0.2]]])
    q = jnp.asarray([[[[1.0]], [[0.8]]], [[[1.4]], [[0.6]]]])
    r = jnp.broadcast_to(jnp.eye(2), (2, horizon, 2, 2)) + jnp.asarray([[0.4], [0.7]])[
        ..., None, None
    ] * jnp.eye(2)
    qf = jnp.asarray([[[1.5]], [[2.0]]])
    c = jnp.asarray([[0.2], [-0.1]])
    cross = jnp.asarray([[[[0.1, -0.05]], [[0.0, 0.08]]], [[[-0.1, 0.2]], [[0.05, 0.0]]]])
    q_linear = jnp.asarray([[[0.1], [-0.2]], [[0.3], [0.1]]])
    r_linear = jnp.asarray([[[0.1, -0.1], [0.2, 0.0]], [[-0.2, 0.3], [0.1, -0.1]]])
    constants = jnp.asarray([[0.4, -0.2], [0.7, 0.1]])
    terminal_linear = jnp.asarray([[0.2], [-0.3]])
    terminal_constants = jnp.asarray([0.5, -0.25])

    stochastic = finite_horizon_multiplicative_lq_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        state_noise_matrices=jnp.zeros((horizon, 2, 1, 1)),
        control_noise_matrices=jnp.zeros((horizon, 2, 1, 2)),
        noise_covariances=jnp.broadcast_to(jnp.eye(2), (horizon, 2, 2)),
        noise_bias=jnp.zeros((horizon, 2, 1)),
        dynamics_bias=c,
        state_control_cross=cross,
        state_linear=q_linear,
        control_linear=r_linear,
        stage_constants=constants,
        terminal_linear=terminal_linear,
        terminal_constants=terminal_constants,
    )
    deterministic = finite_horizon_lq_feedback_nash(
        a,
        b,
        q,
        r,
        qf,
        partition,
        dynamics_bias=c,
        state_control_cross=cross,
        state_linear=q_linear,
        control_linear=r_linear,
        stage_constants=constants,
        terminal_linear=terminal_linear,
        terminal_constants=terminal_constants,
    )

    assert bool(stochastic.valid)
    np.testing.assert_allclose(stochastic.feedback_gain, deterministic.feedback_gain)
    np.testing.assert_allclose(stochastic.feedforward, deterministic.feedforward)
    for stochastic_value, deterministic_value in zip(
        stochastic.values, deterministic.values, strict=True
    ):
        np.testing.assert_allclose(
            stochastic_value.matrices, deterministic_value.matrices
        )
        np.testing.assert_allclose(stochastic_value.linear, deterministic_value.linear)
        np.testing.assert_allclose(
            stochastic_value.constants, deterministic_value.constants
        )
    np.testing.assert_array_equal(stochastic.trace_increments, jnp.zeros((2, horizon)))


def test_one_player_game_matches_multiplicative_lq_control():
    partition = PlayerControlPartition(("controller",), (1,))
    horizon = 2
    a = jnp.asarray([[[1.0]], [[0.9]]])
    b = jnp.asarray([[[0.8]], [[0.7]]])
    q = jnp.asarray([[[1.2]], [[0.8]]])
    r = jnp.asarray([[[1.5]], [[2.0]]])
    qf = jnp.asarray([[2.2]])
    state_noise = jnp.asarray([[[[0.3]], [[-0.2]]], [[[0.1]], [[0.4]]]])
    control_noise = jnp.asarray([[[[0.2]], [[0.1]]], [[[0.5]], [[-0.1]]]])
    gamma = jnp.broadcast_to(jnp.asarray([[1.0, 0.2], [0.2, 0.6]]), (horizon, 2, 2))
    noise_bias = jnp.asarray([[[0.1], [-0.2]], [[0.3], [0.1]]])
    c = jnp.asarray([[0.2], [-0.1]])
    cross = jnp.asarray([[[0.1]], [[-0.05]]])
    q_linear = jnp.asarray([[0.2], [-0.1]])
    r_linear = jnp.asarray([[0.15], [0.05]])
    constants = jnp.asarray([0.4, -0.2])
    terminal_linear = jnp.asarray([0.3])

    control = finite_horizon_multiplicative_lq_state_feedback(
        a,
        b,
        q,
        r,
        qf,
        state_noise_matrices=state_noise,
        control_noise_matrices=control_noise,
        noise_covariances=gamma,
        noise_bias=noise_bias,
        dynamics_bias=c,
        state_control_cross=cross,
        state_linear=q_linear,
        control_linear=r_linear,
        stage_constants=constants,
        terminal_linear=terminal_linear,
        terminal_constant=0.7,
    )
    game = finite_horizon_multiplicative_lq_feedback_nash(
        a,
        b,
        q[None, ...],
        r[None, ...],
        qf[None, ...],
        partition,
        state_noise_matrices=state_noise,
        control_noise_matrices=control_noise,
        noise_covariances=gamma,
        noise_bias=noise_bias,
        dynamics_bias=c,
        state_control_cross=cross[None, ...],
        state_linear=q_linear[None, ...],
        control_linear=r_linear[None, ...],
        stage_constants=constants[None, ...],
        terminal_linear=terminal_linear[None, ...],
        terminal_constants=jnp.asarray([0.7]),
    )

    np.testing.assert_allclose(game.feedback_gain, control.feedback_gain)
    np.testing.assert_allclose(game.feedforward, control.feedforward)
    np.testing.assert_allclose(game.values[0].matrices, control.value.matrices)
    np.testing.assert_allclose(game.values[0].linear, control.value.linear)
    np.testing.assert_allclose(game.values[0].constants, control.value.constants)
    np.testing.assert_allclose(game.trace_increments[0], control.trace_increments)


def test_additive_noise_trace_evidence_keeps_player_and_time_axes():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    noise_bias = jnp.asarray([[[0.2], [-0.1]]])
    gamma = jnp.asarray([[[1.0, 0.4], [0.4, 0.5]]])
    terminal_p = jnp.asarray([1.0, 3.0])
    result = finite_horizon_multiplicative_lq_feedback_nash(
        jnp.ones((1, 1, 1)),
        jnp.zeros((1, 1, 2)),
        jnp.zeros((2, 1, 1, 1)),
        jnp.broadcast_to(jnp.eye(2), (2, 1, 2, 2)),
        terminal_p[:, None, None],
        partition,
        state_noise_matrices=jnp.zeros((1, 2, 1, 1)),
        control_noise_matrices=jnp.zeros((1, 2, 1, 2)),
        noise_covariances=gamma,
        noise_bias=noise_bias,
    )
    scalar_variance = jnp.einsum("ri,sj,rs->", noise_bias[0], noise_bias[0], gamma[0])
    expected = 0.5 * terminal_p * scalar_variance

    assert result.trace_increments.shape == (2, 1)
    np.testing.assert_allclose(result.trace_increments[:, 0], expected)
    np.testing.assert_allclose(result.values[0].constants, [expected[0], 0.0])
    np.testing.assert_allclose(result.values[1].constants, [expected[1], 0.0])


def test_case_axes_jit_and_autodiff_preserve_coupled_noise_dependence():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    cases = 2
    a = jnp.ones((cases, 1, 1, 1))
    b = jnp.broadcast_to(jnp.asarray([[[1.0, 0.5]]]), (cases, 1, 1, 2))
    q = jnp.zeros((cases, 2, 1, 1, 1))
    r = jnp.broadcast_to(jnp.eye(2), (cases, 2, 1, 2, 2))
    qf = jnp.broadcast_to(jnp.asarray([[[1.0]], [[1.5]]]), (cases, 2, 1, 1))
    state_noise = jnp.asarray([0.2, 0.4])[:, None, None, None, None]
    gamma = jnp.ones((cases, 1, 1, 1))
    time_grid = TimeGrid(jnp.asarray([0.0, 1.0]), time_id="multiplicative-game-cases")

    def solve(control_scale):
        channel = jnp.asarray([[0.3, -0.1], [0.5, 0.2]])
        control_noise = control_scale * channel[:, None, None, None, :]
        return finite_horizon_multiplicative_lq_feedback_nash(
            a,
            b,
            q,
            r,
            qf,
            partition,
            state_noise_matrices=state_noise,
            control_noise_matrices=control_noise,
            noise_covariances=gamma,
            time_grid=time_grid,
        )

    result = eqx.filter_jit(solve)(jnp.asarray(1.0))
    gradient = jax.jit(jax.grad(lambda scale: solve(scale).feedback_gain[0, 0, 0, 0]))(
        jnp.asarray(1.0)
    )

    assert result.feedback_gain.shape == (cases, 1, 2, 1)
    assert result.trace_increments.shape == (cases, 2, 1)
    assert result.diagnostics.own_control_minimum_eigenvalues.shape == (
        cases,
        2,
        1,
    )
    assert result.values[0].matrices.shape == (cases, 2, 1, 1)
    np.testing.assert_array_equal(result.valid, jnp.ones(cases, dtype=bool))
    assert np.isfinite(gradient)
    assert not np.isclose(gradient, 0.0)


def test_jitted_cases_report_covariance_curvature_rank_condition_and_nonfinite_failures():
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    cases = 7
    a = jnp.ones((cases, 1, 1, 1)).at[3, 0, 0, 0].set(jnp.nan)
    b = jnp.zeros((cases, 1, 1, 2))
    q = jnp.zeros((cases, 2, 1, 1, 1))
    qf = jnp.zeros((cases, 2, 1, 1))
    r = jnp.broadcast_to(jnp.eye(2), (cases, 2, 1, 2, 2))
    r = r.at[4, 0, 0, 0, 0].set(-1.0)
    singular_left = jnp.asarray([[1.0, 1.0], [1.0, 2.0]])
    singular_right = jnp.asarray([[2.0, 1.0], [1.0, 1.0]])
    r = r.at[5, 0, 0].set(singular_left)
    r = r.at[5, 1, 0].set(singular_right)
    r = r.at[6, 1, 0, 1, 1].set(1e-5)
    gamma = jnp.broadcast_to(jnp.eye(2), (cases, 1, 2, 2))
    gamma = gamma.at[1, 0].set(jnp.asarray([[1.0, 0.2], [0.0, 1.0]]))
    gamma = gamma.at[2, 0].set(jnp.asarray([[1.0, 0.0], [0.0, -1.0]]))
    time_grid = TimeGrid(jnp.asarray([0.0, 1.0]), time_id="multiplicative-game-failures")

    result = eqx.filter_jit(finite_horizon_multiplicative_lq_feedback_nash)(
        a,
        b,
        q,
        r,
        qf,
        partition,
        state_noise_matrices=jnp.zeros((cases, 1, 2, 1, 1)),
        control_noise_matrices=jnp.zeros((cases, 1, 2, 1, 2)),
        noise_covariances=gamma,
        time_grid=time_grid,
        maximum_condition=1e4,
    )
    expected = jnp.asarray(
        [
            MultiplicativeLQFeedbackNashStatus.SUCCESS,
            MultiplicativeLQFeedbackNashStatus.NOISE_COVARIANCE_NONSYMMETRIC,
            MultiplicativeLQFeedbackNashStatus.NOISE_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE,
            MultiplicativeLQFeedbackNashStatus.NONFINITE_INPUT,
            MultiplicativeLQFeedbackNashStatus.OWN_CURVATURE_NOT_POSITIVE_DEFINITE,
            MultiplicativeLQFeedbackNashStatus.COUPLED_SYSTEM_RANK_DEFICIENT,
            MultiplicativeLQFeedbackNashStatus.CONDITION_LIMIT_REACHED,
        ],
        dtype=jnp.int32,
    )

    np.testing.assert_array_equal(result.status, expected)
    np.testing.assert_array_equal(
        result.diagnostics.first_failed_stage, [-1, 0, 0, 0, 0, 0, 0]
    )
    assert bool(result.diagnostics.diagnostic_available[5, 0])
    assert result.diagnostics.coupled_ranks[5, 0] == 1
    assert result.diagnostics.coupled_condition_numbers[6, 0] > 1e4
