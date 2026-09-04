#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.control.stochastic._belief_lqg as belief_lqg
from phydrax.control._lqr import finite_horizon_lqr
from phydrax.control.games._information import GaussianBelief
from phydrax.control.stochastic._belief_lqg import (
    CentralizedLQGProblem,
    CentralizedLQGStatus,
    finite_horizon_centralized_lqg,
)
from phydrax.control.stochastic._lqg import finite_horizon_lqg_state_feedback
from phydrax.dynamics import DiscreteStepContext
from phydrax.linalg import LinearSolveStatus


def _scalar_problem(
    *,
    horizon=1,
    prior_mean=1.0,
    prior_covariance=2.0,
    observation=1.0,
    measurement_covariance=3.0,
    process_covariance=4.0,
    case_shape=(),
    belief_id="shared-posterior",
    covariance_tolerance=1.0e-7,
):
    a = jnp.broadcast_to(jnp.ones((horizon, 1, 1)), case_shape + (horizon, 1, 1))
    b = jnp.broadcast_to(jnp.ones((horizon, 1, 1)), case_shape + (horizon, 1, 1))
    g = jnp.broadcast_to(jnp.ones((horizon, 1, 1)), case_shape + (horizon, 1, 1))
    omega = jnp.broadcast_to(
        jnp.full((horizon, 1, 1), process_covariance),
        case_shape + (horizon, 1, 1),
    )
    c = jnp.broadcast_to(
        jnp.full((horizon, 1, 1), observation),
        case_shape + (horizon, 1, 1),
    )
    v = jnp.broadcast_to(
        jnp.full((horizon, 1, 1), measurement_covariance),
        case_shape + (horizon, 1, 1),
    )
    q = jnp.broadcast_to(jnp.ones((horizon, 1, 1)), case_shape + (horizon, 1, 1))
    r = jnp.broadcast_to(jnp.ones((horizon, 1, 1)), case_shape + (horizon, 1, 1))
    qf = jnp.broadcast_to(jnp.asarray([[2.0]]), case_shape + (1, 1))
    prior = GaussianBelief(
        jnp.asarray([prior_mean]),
        jnp.asarray([[prior_covariance]]),
        belief_id=belief_id,
    )
    return CentralizedLQGProblem(
        a,
        b,
        g,
        omega,
        c,
        v,
        prior,
        q,
        r,
        qf,
        information_id="centralized-test-observation",
        problem_id="scalar-belief-lqg",
        covariance_tolerance=covariance_tolerance,
    )


def test_perfect_observation_reduces_exactly_to_full_state_additive_lqg():
    horizon = 2
    process_covariances = jnp.asarray([[[0.5]], [[0.25]]])
    problem = CentralizedLQGProblem(
        jnp.ones((horizon, 1, 1)),
        jnp.ones((horizon, 1, 1)),
        jnp.ones((horizon, 1, 1)),
        process_covariances,
        jnp.ones((horizon, 1, 1)),
        jnp.zeros((horizon, 1, 1)),
        GaussianBelief(
            jnp.asarray([1.25]),
            jnp.asarray([[2.0]]),
            belief_id="perfect-state-belief",
        ),
        jnp.ones((horizon, 1, 1)),
        jnp.ones((horizon, 1, 1)),
        jnp.asarray([[2.0]]),
    )
    result = finite_horizon_centralized_lqg(problem, policy_id="perfect")
    full_state = finite_horizon_lqg_state_feedback(
        problem.dynamics_matrices,
        problem.control_matrices,
        problem.state_costs,
        problem.control_costs,
        problem.terminal_state_cost,
        process_noise_factors=problem.process_noise_factors,
        process_noise_covariances=problem.process_noise_covariances,
        initial_mean=problem.initial_belief.mean,
        initial_covariance=problem.initial_belief.covariance,
        policy_id="perfect",
    )

    assert bool(result.valid)
    assert int(result.status) == int(CentralizedLQGStatus.SUCCESS)
    assert result.result_label == "CENTRALIZED_GAUSSIAN_BELIEF_LQG"
    np.testing.assert_allclose(result.posterior_covariances, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(result.feedback_gain, full_state.feedback_gain)
    np.testing.assert_allclose(result.feedforward, full_state.feedforward)
    np.testing.assert_allclose(
        result.initial_expected_cost,
        full_state.initial_expected_cost,
        rtol=2.0e-6,
        atol=2.0e-6,
    )


def test_no_observation_limit_is_the_exact_open_loop_mean_policy():
    problem = _scalar_problem(
        observation=0.0,
        measurement_covariance=1.0,
        process_covariance=3.0,
    )
    result = finite_horizon_centralized_lqg(problem)

    np.testing.assert_array_equal(result.kalman_gains, jnp.zeros((1, 1, 1)))
    np.testing.assert_allclose(result.predicted_covariances[..., 0, 0], [2.0])
    np.testing.assert_allclose(result.posterior_covariances[..., 0, 0], [2.0])
    np.testing.assert_allclose(result.terminal_covariance[..., 0, 0], 5.0)
    np.testing.assert_allclose(result.expected_actions[..., 0], [-2.0 / 3.0])
    # E[.5 x0² + .5 u0² + x1²] for x0 ~ N(1, 2), u0 = -2/3.
    np.testing.assert_allclose(result.initial_expected_cost, 41.0 / 6.0, rtol=2e-6)


def test_scalar_kalman_schedule_and_lqg_trace_terms_are_analytic():
    result = finite_horizon_centralized_lqg(_scalar_problem(prior_mean=1.5))

    np.testing.assert_allclose(result.innovation_covariances[..., 0, 0], [5.0])
    np.testing.assert_allclose(result.kalman_gains[..., 0, 0], [2.0 / 5.0])
    np.testing.assert_allclose(result.posterior_covariances[..., 0, 0], [6.0 / 5.0])
    np.testing.assert_allclose(result.terminal_covariance[..., 0, 0], 26.0 / 5.0)
    np.testing.assert_allclose(
        result.posterior_mean_innovation_covariances[..., 0, 0], [4.0 / 5.0]
    )
    np.testing.assert_allclose(result.state_covariance_costs, [3.0 / 5.0, 26.0 / 5.0])
    np.testing.assert_allclose(result.initial_covariance_cost, 5.0 / 3.0)
    np.testing.assert_allclose(result.initial_observation_trace_cost, 2.0 / 3.0)
    np.testing.assert_allclose(
        result.value_constant_corrections, [29.0 / 5.0, 26.0 / 5.0]
    )
    np.testing.assert_allclose(
        result.initial_expected_cost,
        1001.0 / 120.0,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(result.innovation_symmetry_residuals, [0.0])
    np.testing.assert_array_equal(result.posterior_covariance_symmetry_residuals, [0.0])
    np.testing.assert_allclose(result.innovation_solve_residuals, 0.0)
    np.testing.assert_array_equal(
        result.innovation_solve_statuses,
        [[int(LinearSolveStatus.SUCCESS)]],
    )
    np.testing.assert_allclose(
        result.innovation_solve_relative_residuals, [[0.0]], atol=1.0e-15
    )
    np.testing.assert_array_equal(result.innovation_solve_successful, [True])


def test_finite_uncertified_innovation_gain_propagates_failure(monkeypatch):
    native_solve = belief_lqg.solve

    def uncertified_solve(*args, **kwargs):
        result = native_solve(*args, **kwargs)
        result = eqx.tree_at(
            lambda item: item.status,
            result,
            jnp.full_like(
                result.status,
                int(LinearSolveStatus.RESIDUAL_TOO_LARGE),
            ),
        )
        return eqx.tree_at(
            lambda item: item.diagnostics.relative_residual,
            result,
            jnp.full_like(result.diagnostics.relative_residual, 0.25),
        )

    monkeypatch.setattr(belief_lqg, "solve", uncertified_solve)
    result = finite_horizon_centralized_lqg(_scalar_problem())

    assert bool(jnp.all(jnp.isfinite(result.kalman_gains)))
    np.testing.assert_array_equal(
        result.innovation_solve_statuses,
        [[int(LinearSolveStatus.RESIDUAL_TOO_LARGE)]],
    )
    np.testing.assert_allclose(result.innovation_solve_relative_residuals, [[0.25]])
    np.testing.assert_array_equal(result.innovation_solve_successful, [False])
    assert not bool(result.valid)
    assert int(result.status) == int(CentralizedLQGStatus.INNOVATION_SOLVE_FAILED)

    inactive = finite_horizon_centralized_lqg(
        _scalar_problem(observation=0.0, measurement_covariance=0.0)
    )
    np.testing.assert_array_equal(
        inactive.innovation_solve_statuses,
        [[int(LinearSolveStatus.SUCCESS)]],
    )
    np.testing.assert_array_equal(
        inactive.innovation_solve_relative_residuals,
        [[0.0]],
    )
    np.testing.assert_array_equal(inactive.innovation_solve_successful, [True])
    assert bool(inactive.valid)
    assert int(inactive.status) == int(CentralizedLQGStatus.SUCCESS)


def test_policy_sees_only_the_posterior_belief_not_the_latent_world():
    problem = _scalar_problem(
        prior_mean=0.0,
        prior_covariance=1.0,
        measurement_covariance=1.0,
        process_covariance=0.0,
    )
    result = finite_horizon_centralized_lqg(problem)
    context = DiscreteStepContext(0.0, 1.0, 0)
    observation = 0.5
    posterior_mean = 0.5 * observation
    posterior_covariance = 0.5

    latent_world_left = -2.0
    latent_world_right = 3.0
    del latent_world_left, latent_world_right
    left_history_belief = GaussianBelief(
        jnp.asarray([posterior_mean]),
        jnp.asarray([[posterior_covariance]]),
        belief_id=problem.initial_belief.belief_id,
    )
    right_history_belief = GaussianBelief(
        jnp.asarray([posterior_mean]),
        jnp.asarray([[posterior_covariance]]),
        belief_id=problem.initial_belief.belief_id,
    )

    left_action = result.policy.action(context, left_history_belief, None)
    right_action = result.policy.action(context, right_history_belief, {"unused": 1})
    np.testing.assert_array_equal(left_action, right_action)
    with pytest.raises(TypeError, match="GaussianBelief"):
        result.policy.action(context, jnp.asarray([-2.0]), None)


def test_covariance_assumptions_and_unsupported_correlations_are_rejected():
    base = _scalar_problem()
    constructor_args = (
        base.dynamics_matrices,
        base.control_matrices,
        base.process_noise_factors,
        base.process_noise_covariances,
        base.observation_matrices,
        base.measurement_covariances,
        base.initial_belief,
        base.state_costs,
        base.control_costs,
        base.terminal_state_cost,
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match="positive semidefinite"):
        CentralizedLQGProblem(
            *constructor_args[:5],
            jnp.asarray([[[-1.0]]]),
            *constructor_args[6:],
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="symmetric"):
        CentralizedLQGProblem(
            base.dynamics_matrices,
            base.control_matrices,
            base.process_noise_factors,
            base.process_noise_covariances,
            jnp.asarray([[[1.0], [0.0]]]),
            jnp.asarray([[[1.0, 0.25], [0.0, 1.0]]]),
            base.initial_belief,
            base.state_costs,
            base.control_costs,
            base.terminal_state_cost,
        )
    with pytest.raises(ValueError, match="action-independent sensing"):
        CentralizedLQGProblem(
            *constructor_args,
            observation_control_matrices=jnp.zeros((1, 1, 1)),
        )
    with pytest.raises(ValueError, match="independent process and measurement"):
        CentralizedLQGProblem(
            *constructor_args,
            process_measurement_cross_covariances=jnp.zeros((1, 1, 1)),
        )


def test_nontrivial_singular_innovation_is_invalid_without_a_pseudoinverse():
    base = _scalar_problem()
    problem = CentralizedLQGProblem(
        base.dynamics_matrices,
        base.control_matrices,
        base.process_noise_factors,
        base.process_noise_covariances,
        jnp.asarray([[[1.0], [1.0]]]),
        jnp.zeros((1, 2, 2)),
        base.initial_belief,
        base.state_costs,
        base.control_costs,
        base.terminal_state_cost,
    )
    result = finite_horizon_centralized_lqg(problem)

    assert not bool(result.valid)
    assert int(result.status) == int(CentralizedLQGStatus.INNOVATION_SOLVE_FAILED)
    assert bool(
        jnp.all(result.innovation_solve_statuses != int(LinearSolveStatus.SUCCESS))
    )
    np.testing.assert_array_equal(result.innovation_well_posed, False)


def test_successful_innovation_solve_cannot_override_well_posedness_failure():
    result = finite_horizon_centralized_lqg(
        _scalar_problem(
            prior_covariance=0.0,
            measurement_covariance=1.0e-4,
            process_covariance=0.0,
            covariance_tolerance=1.0e-2,
        )
    )

    np.testing.assert_array_equal(
        result.innovation_solve_statuses,
        [[int(LinearSolveStatus.SUCCESS)]],
    )
    assert not bool(result.innovation_well_posed)
    assert not bool(result.valid)
    assert int(result.status) == int(
        CentralizedLQGStatus.INNOVATION_COVARIANCE_NOT_POSITIVE_DEFINITE
    )


def test_problem_and_policy_reject_observation_timing_mismatches():
    base = _scalar_problem()
    with pytest.raises(ValueError, match="pre-action observation timing"):
        CentralizedLQGProblem(
            base.dynamics_matrices,
            base.control_matrices,
            base.process_noise_factors,
            base.process_noise_covariances,
            base.observation_matrices,
            base.measurement_covariances,
            base.initial_belief,
            base.state_costs,
            base.control_costs,
            base.terminal_state_cost,
            observation_timing="post-action",
        )

    result = finite_horizon_centralized_lqg(base)
    belief = GaussianBelief(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        belief_id=base.initial_belief.belief_id,
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="time-grid interval"):
        result.policy.action(DiscreteStepContext(0.0, 2.0, 0), belief, None)


def test_zero_process_and_state_uncertainty_reduce_to_affine_lqr():
    problem = _scalar_problem(
        horizon=2,
        prior_mean=1.25,
        prior_covariance=0.0,
        observation=0.0,
        measurement_covariance=0.0,
        process_covariance=0.0,
    )
    deterministic = finite_horizon_lqr(
        problem.dynamics_matrices,
        problem.control_matrices,
        problem.state_costs,
        problem.control_costs,
        problem.terminal_state_cost,
    )
    result = finite_horizon_centralized_lqg(problem)
    assert bool(result.valid)
    np.testing.assert_array_equal(result.innovation_inactive, [True, True])
    np.testing.assert_array_equal(
        result.innovation_solve_statuses,
        jnp.full((2, 1), int(LinearSolveStatus.SUCCESS)),
    )
    np.testing.assert_array_equal(
        result.innovation_solve_relative_residuals,
        jnp.zeros((2, 1)),
    )
    np.testing.assert_array_equal(result.innovation_solve_successful, [True, True])

    np.testing.assert_array_equal(result.feedback_gain, deterministic.feedback_gain)
    np.testing.assert_array_equal(result.feedforward, deterministic.feedforward)
    np.testing.assert_array_equal(
        result.value_constant_corrections, jnp.zeros((problem.horizon + 1,))
    )
    np.testing.assert_array_equal(result.value.constants, deterministic.value.constants)
    np.testing.assert_allclose(
        result.initial_expected_cost,
        deterministic.value.evaluate(0.0, problem.initial_belief.mean),
    )


def test_case_axes_and_filtered_jit_are_preserved():
    case_shape = (2, 3)
    problem = _scalar_problem(
        horizon=2,
        prior_mean=0.5,
        prior_covariance=1.0,
        measurement_covariance=1.0,
        process_covariance=0.5,
        case_shape=case_shape,
    )
    result = eqx.filter_jit(finite_horizon_centralized_lqg)(problem)

    assert result.feedback_gain.shape == case_shape + (2, 1, 1)
    assert result.kalman_gains.shape == case_shape + (2, 1, 1)
    assert result.predicted_means.shape == case_shape + (2, 1)
    assert result.posterior_means.shape == case_shape + (2, 1)
    assert result.expected_actions.shape == case_shape + (2, 1)
    assert result.predicted_covariances.shape == case_shape + (2, 1, 1)
    assert result.posterior_covariances.shape == case_shape + (2, 1, 1)
    assert result.innovation_symmetry_residuals.shape == case_shape + (2,)
    assert result.innovation_solve_statuses.shape == case_shape + (2, 1)
    assert result.innovation_solve_relative_residuals.shape == case_shape + (2, 1)
    assert result.innovation_solve_successful.shape == case_shape + (2,)
    assert result.value_constant_corrections.shape == case_shape + (3,)
    assert result.initial_expected_cost.shape == case_shape
    assert result.valid.shape == case_shape
    assert bool(jnp.all(result.valid))

    posterior = GaussianBelief(
        jnp.asarray([0.25]),
        jnp.asarray([[0.5]]),
        belief_id=problem.initial_belief.belief_id,
    )
    action = eqx.filter_jit(result.policy.action)(
        DiscreteStepContext(0.0, 1.0, 0), posterior, None
    )
    assert action.shape == case_shape + (1,)
