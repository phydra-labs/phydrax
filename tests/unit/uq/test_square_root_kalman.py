from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.uq._square_root import (
    _forecast_factor,
    _smoothing_factor,
    _update_factors,
)


def _problem(
    *,
    case_shape=(),
    mask=None,
    step_valid=None,
    prior_covariance=None,
    process_covariance=None,
    observation_covariance=None,
):
    if case_shape:
        values = jnp.asarray(
            [
                [[0.8, -0.1], [1.0, 0.2], [1.2, 0.4], [1.1, 0.5]],
                [[-0.4, 0.7], [-0.2, 0.5], [0.0, 0.3], [0.0, 0.0]],
            ]
        )
        times = jnp.asarray([[0.25, 0.5, 0.75, 1.0], [0.25, 0.5, 0.75, 0.75]])
        mean = jnp.asarray([[0.2, -0.1], [-0.3, 0.4]])
        case_axes = ("case",)
        case_ids = ("first", "second")
    else:
        values = jnp.asarray([[0.8, -0.1], [1.0, 0.2], [1.2, 0.4], [1.1, 0.5]])
        times = jnp.asarray([0.25, 0.5, 0.75, 1.0])
        mean = jnp.asarray([0.2, -0.1])
        case_axes = ()
        case_ids = ("only",)
    observations = phx.stochastic.ObservationSequence(
        times,
        values,
        case_axes=case_axes,
        case_shape=case_shape,
        observation_mask=mask,
        step_valid=step_valid,
        case_ids=case_ids,
        sequence_id="square-root-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        mean,
        (
            jnp.asarray([[1.0, 0.2], [0.2, 0.5]])
            if prior_covariance is None
            else prior_covariance
        ),
        state_shape=(2,),
        prior_id="square-root-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0, 0.1], [0.0, 0.9]]),
        (
            jnp.asarray([[0.08, 0.01], [0.01, 0.04]])
            if process_covariance is None
            else process_covariance
        ),
        state_shape=(2,),
        process_id="square-root-process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0, 0.0], [0.2, 1.0]]),
        (
            jnp.asarray([[0.3, 0.02], [0.02, 0.2]])
            if observation_covariance is None
            else observation_covariance
        ),
        state_shape=(2,),
        observation_shape=(2,),
        observation_id="square-root-observation",
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="square-root-linear",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="square-root-problem",
    )


def _scalar_problem(*, process_variance=0.1, observation_variance=0.2):
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0, 1.5]),
        jnp.asarray([[1.0], [1.5], [1.25]]),
        case_axes=(),
        case_shape=(),
        case_ids=("only",),
        sequence_id="scalar-square-root-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="scalar-square-root-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[process_variance]]),
        state_shape=(1,),
        process_id="scalar-square-root-process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[observation_variance]]),
        state_shape=(1,),
        observation_shape=(1,),
        observation_id="scalar-square-root-observation",
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="scalar-square-root-linear",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="scalar-square-root-problem",
    )


def _assert_filter_equivalent(covariance, square_root, *, atol=2e-5):
    assert jnp.allclose(
        square_root.predicted_means, covariance.predicted_means, atol=atol
    )
    assert jnp.allclose(
        square_root.predicted_covariances,
        covariance.predicted_covariances,
        atol=atol,
    )
    assert jnp.allclose(square_root.filtered_means, covariance.filtered_means, atol=atol)
    assert jnp.allclose(
        square_root.filtered_covariances,
        covariance.filtered_covariances,
        atol=atol,
    )
    assert jnp.allclose(
        square_root.innovation_covariances,
        covariance.innovation_covariances,
        atol=atol,
    )
    assert jnp.allclose(
        square_root.incremental_log_likelihood,
        covariance.incremental_log_likelihood,
        atol=atol,
    )
    assert jnp.array_equal(square_root.valid, covariance.valid)
    assert jnp.array_equal(square_root.status, covariance.status)


def test_regular_filter_and_rts_match_covariance_form_with_provenance():
    problem = _problem()
    covariance = phx.uq.kalman_filter(
        problem,
        method="sequential",
        covariance_form="covariance",
        covariance_regularization=0.03,
    )
    square_root = phx.uq.kalman_filter(
        problem,
        method="sequential",
        covariance_form="square_root",
        covariance_regularization=0.03,
    )
    _assert_filter_equivalent(covariance, square_root)

    covariance_smoother = phx.uq.rts_smoother(
        covariance,
        method="sequential",
        covariance_form="covariance",
    )
    square_root_smoother = phx.uq.rts_smoother(
        square_root,
        method="sequential",
        covariance_form="square_root",
    )
    assert jnp.allclose(square_root_smoother.means, covariance_smoother.means, atol=3e-5)
    assert jnp.allclose(
        square_root_smoother.covariances,
        covariance_smoother.covariances,
        atol=3e-5,
    )
    assert square_root.covariance_form == "square_root"
    assert square_root.execution_method == "sequential"
    assert square_root.covariance_regularization == 0.03
    assert square_root_smoother.covariance_form == "square_root"
    assert square_root_smoother.execution_method == "sequential"


def test_square_root_smoother_propagates_factor_diagnostics_backward():
    result = phx.uq.kalman_filter(
        _problem(),
        method="sequential",
        covariance_form="square_root",
    )
    invalid_filtered_covariances = result.filtered_covariances.at[3].set(
        jnp.asarray([[1.0, 0.5], [0.0, 1.0]])
    )
    invalid_filtered = eqx.tree_at(
        lambda node: node.filtered_covariances,
        result,
        invalid_filtered_covariances,
    )

    filtered_smoother = phx.uq.rts_smoother(
        invalid_filtered,
        covariance_form="square_root",
    )

    assert jnp.array_equal(
        filtered_smoother.valid,
        jnp.asarray([False, False, False, False]),
    )
    assert jnp.all(jnp.isfinite(filtered_smoother.covariances))

    invalid_predicted_covariances = result.predicted_covariances.at[2].set(
        jnp.asarray([[1.0, 0.0], [0.0, -1.0]])
    )
    invalid_predicted = eqx.tree_at(
        lambda node: node.predicted_covariances,
        result,
        invalid_predicted_covariances,
    )

    predicted_smoother = phx.uq.rts_smoother(
        invalid_predicted,
        covariance_form="square_root",
    )

    assert jnp.array_equal(
        predicted_smoother.valid,
        jnp.asarray([False, False, True, True]),
    )
    assert jnp.all(jnp.isfinite(predicted_smoother.covariances))

    inconsistent_transitions = result.transition_matrices.at[2].set(1e6 * jnp.eye(2))
    invalid_proposal = eqx.tree_at(
        lambda node: node.transition_matrices,
        result,
        inconsistent_transitions,
    )

    proposal_smoother = phx.uq.rts_smoother(
        invalid_proposal,
        covariance_form="square_root",
    )

    assert jnp.array_equal(
        proposal_smoother.valid,
        jnp.asarray([False, False, True, True]),
    )
    assert jnp.all(jnp.isfinite(proposal_smoother.covariances))


def test_square_root_smoother_rejects_nonfinite_proposed_moments():
    result = phx.uq.kalman_filter(
        _problem(),
        method="sequential",
        covariance_form="square_root",
    )
    extreme = 0.75 * jnp.finfo(result.filtered_means.dtype).max
    filtered_means = result.filtered_means.at[2].set(jnp.full((2,), extreme))
    predicted_means = result.predicted_means.at[2].set(jnp.full((2,), -extreme))
    extreme_result = eqx.tree_at(
        lambda node: (node.filtered_means, node.predicted_means),
        result,
        (filtered_means, predicted_means),
    )

    smoother = phx.uq.rts_smoother(
        extreme_result,
        covariance_form="square_root",
    )

    assert jnp.array_equal(
        smoother.valid,
        jnp.asarray([False, False, True, True]),
    )
    assert jnp.all(jnp.isfinite(smoother.means))
    assert jnp.all(jnp.isfinite(smoother.covariances))


def test_singular_psd_state_covariance_is_preserved_through_filter_and_smoother():
    problem = _problem(
        prior_covariance=jnp.asarray([[1.0, 0.0], [0.0, 0.0]]),
        process_covariance=jnp.asarray([[0.1, 0.0], [0.0, 0.0]]),
        observation_covariance=jnp.asarray([[0.2, 0.0], [0.0, 0.3]]),
    )
    covariance = phx.uq.kalman_filter(
        problem, method="sequential", covariance_form="covariance"
    )
    square_root = phx.uq.kalman_filter(
        problem, method="sequential", covariance_form="square_root"
    )
    _assert_filter_equivalent(covariance, square_root, atol=3e-5)
    square_root_smoother = phx.uq.rts_smoother(
        square_root, method="sequential", covariance_form="square_root"
    )
    covariance_smoother = phx.uq.rts_smoother(
        covariance, method="sequential", covariance_form="covariance"
    )
    assert jnp.allclose(
        square_root_smoother.covariances,
        covariance_smoother.covariances,
        atol=4e-5,
    )
    assert jnp.all(jnp.linalg.eigvalsh(square_root.filtered_covariances) >= -1e-6)


def test_zero_observation_noise_matches_exact_covariance_update():
    problem = _scalar_problem(observation_variance=0.0)
    covariance = phx.uq.kalman_filter(
        problem, method="sequential", covariance_form="covariance"
    )
    square_root = phx.uq.kalman_filter(
        problem, method="sequential", covariance_form="square_root"
    )
    _assert_filter_equivalent(covariance, square_root, atol=2e-6)
    assert jnp.allclose(square_root.filtered_covariances, 0.0, atol=2e-6)
    assert jnp.all(square_root.status == phx.uq.KALMAN_SUCCESS)


def test_missing_batched_and_padded_cases_match_without_hidden_updates():
    mask = jnp.asarray(
        [
            [[True, True], [True, False], [False, False], [True, True]],
            [[True, False], [False, True], [False, False], [False, False]],
        ]
    )
    step_valid = jnp.asarray([[True, True, True, True], [True, True, True, False]])
    problem = _problem(
        case_shape=(2,),
        mask=mask,
        step_valid=step_valid,
    )
    covariance = phx.uq.kalman_filter(
        problem, method="sequential", covariance_form="covariance"
    )
    square_root = phx.uq.kalman_filter(
        problem, method="auto", covariance_form="square_root"
    )
    _assert_filter_equivalent(covariance, square_root, atol=3e-5)
    assert jnp.allclose(
        square_root.filtered_means[0, 2], square_root.predicted_means[0, 2]
    )
    assert jnp.allclose(
        square_root.filtered_covariances[0, 2],
        square_root.predicted_covariances[0, 2],
    )
    assert square_root.observed_counts[0, 2] == 0
    assert square_root.incremental_log_likelihood[0, 2] == 0.0
    assert jnp.allclose(
        square_root.filtered_means[1, 3], square_root.filtered_means[1, 2]
    )


def test_square_root_filter_gradient_matches_covariance_filter_gradient():
    def objective(scale, covariance_form):
        observations = phx.stochastic.ObservationSequence(
            jnp.asarray([0.5, 1.0]),
            jnp.asarray([[0.7], [1.1]]),
            case_axes=(),
            case_shape=(),
            case_ids=("only",),
            sequence_id="gradient-sequence",
        )
        prior = phx.stochastic.GaussianStatePrior(
            jnp.asarray([0.0]),
            jnp.asarray([[0.8]]),
            state_shape=(1,),
            prior_id="gradient-prior",
        )

        def process_covariance(t0, t1, context):
            del t0, t1, context
            return jnp.reshape(scale**2, (1, 1))

        transition = phx.stochastic.LinearGaussianTransitionKernel(
            jnp.asarray([[1.0]]),
            process_covariance,
            state_shape=(1,),
            process_id="gradient-process",
        )
        observation = phx.stochastic.LinearGaussianObservationModel(
            jnp.asarray([[1.0]]),
            jnp.asarray([[0.25]]),
            state_shape=(1,),
            observation_shape=(1,),
            observation_id="gradient-observation",
        )
        problem = phx.stochastic.StateSpaceProblem(
            phx.stochastic.StateSpaceModel(
                prior,
                transition,
                observation,
                model_id="gradient-model",
            ),
            observations,
            initial_time=0.0,
            problem_id="gradient-problem",
        )
        result = phx.uq.kalman_filter(
            problem,
            method="sequential",
            covariance_form=covariance_form,
        )
        return result.filtered_means[-1, 0]

    covariance_gradient = jax.grad(lambda scale: objective(scale, "covariance"))(
        jnp.asarray(0.3)
    )
    square_root_gradient = jax.grad(lambda scale: objective(scale, "square_root"))(
        jnp.asarray(0.3)
    )
    assert jnp.isfinite(square_root_gradient)
    assert jnp.allclose(square_root_gradient, covariance_gradient, atol=3e-5)


def test_qr_factor_algebra_uses_conjugate_transposes_for_complex_values():
    filtered = phx.uq.GaussianFactor(
        jnp.asarray([[1.0 + 0.0j, 0.0j], [0.1j, 0.7 + 0.0j]])
    )
    process = phx.uq.GaussianFactor(
        jnp.asarray([[0.3 + 0.0j, 0.0j], [0.05j, 0.2 + 0.0j]])
    )
    transition = jnp.asarray([[1.0 + 0.0j, 0.2j], [0.1 - 0.1j, 0.9 + 0.0j]])
    predicted = _forecast_factor(transition, filtered, process)
    expected_prediction = (
        transition @ filtered.covariance @ jnp.conj(transition.T) + process.covariance
    )
    assert jnp.allclose(predicted.covariance, expected_prediction, atol=2e-6)

    observation_matrix = jnp.asarray([[1.0 + 0.0j, 0.3j], [0.2 - 0.1j, 1.0 + 0.0j]])
    observation_noise = phx.uq.GaussianFactor(
        jnp.asarray([[0.5 + 0.0j, 0.0j], [0.1j, 0.4 + 0.0j]])
    )
    innovation, updated, gain = _update_factors(
        predicted, observation_matrix, observation_noise
    )
    expected_innovation = (
        observation_matrix @ predicted.covariance @ jnp.conj(observation_matrix.T)
        + observation_noise.covariance
    )
    expected_gain = (
        predicted.covariance
        @ jnp.conj(observation_matrix.T)
        @ jnp.linalg.inv(expected_innovation)
    )
    expected_updated = (
        predicted.covariance
        - expected_gain @ expected_innovation @ jnp.conj(expected_gain.T)
    )
    assert jnp.allclose(innovation.covariance, expected_innovation, atol=3e-6)
    assert jnp.allclose(gain, expected_gain, atol=3e-6)
    assert jnp.allclose(updated.covariance, expected_updated, atol=3e-6)

    smoothed, smoothing_gain = _smoothing_factor(filtered, predicted, transition, updated)
    expected_smoothing_gain = (
        filtered.covariance
        @ jnp.conj(transition.T)
        @ jnp.linalg.inv(predicted.covariance)
    )
    expected_smoothed = filtered.covariance + expected_smoothing_gain @ (
        updated.covariance - predicted.covariance
    ) @ jnp.conj(expected_smoothing_gain.T)
    assert jnp.allclose(smoothing_gain, expected_smoothing_gain, atol=4e-6)
    assert jnp.allclose(smoothed.covariance, expected_smoothed, atol=4e-6)


def test_invalid_covariance_form_and_square_root_parallel_dispatch_are_explicit():
    problem = _problem()
    invalid_form: Any = "factor"
    invalid_method: Any = "invalid"
    with pytest.raises(ValueError, match="covariance_form"):
        phx.uq.kalman_filter(problem, covariance_form=invalid_form)
    with pytest.raises(ValueError, match="method"):
        phx.uq.kalman_filter(
            problem, covariance_form="square_root", method=invalid_method
        )
    with pytest.raises(ValueError, match="does not support method='parallel'"):
        phx.uq.kalman_filter(problem, covariance_form="square_root", method="parallel")

    result = phx.uq.kalman_filter(
        problem, covariance_form="square_root", method="sequential"
    )
    with pytest.raises(ValueError, match="covariance_form"):
        phx.uq.rts_smoother(result, covariance_form=invalid_form)
    with pytest.raises(ValueError, match="does not support method='parallel'"):
        phx.uq.rts_smoother(result, covariance_form="square_root", method="parallel")
