import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _problem(*, mask=None, case_shape=(), step_valid=None):
    if case_shape:
        values = jnp.asarray([[[1.0], [2.0]], [[-1.0], [-1.0]]])
        times = jnp.asarray([[0.5, 1.0], [0.5, 0.5]])
        mean = jnp.asarray([[0.0], [0.0]])
        case_axes = ("case",)
        case_ids = ("positive", "negative")
    else:
        values = jnp.asarray([[1.0], [2.0]])
        times = jnp.asarray([0.5, 1.0])
        mean = jnp.asarray([0.0])
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
        sequence_id="kalman-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        mean,
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        process_id="latent",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="linear"
    )
    return phx.stochastic.StateSpaceProblem(
        model, observations, initial_time=0.0, problem_id="linear-problem"
    )


def test_scalar_kalman_matches_hand_calculation_and_streaming():
    problem = _problem()
    result = phx.uq.kalman_filter(problem)

    predicted_variance = 1.1
    gain = predicted_variance / (predicted_variance + 0.2)
    expected_mean = gain
    expected_variance = (1.0 - gain) * predicted_variance

    assert jnp.allclose(result.filtered_means[0, 0], expected_mean)
    assert jnp.allclose(result.filtered_covariances[0, 0, 0], expected_variance)
    assert jnp.all(result.status == phx.uq.KALMAN_SUCCESS)

    state = phx.uq.initialize_kalman_filter(problem)
    records = []
    for _ in range(problem.observations.num_steps):
        state, record = phx.uq.kalman_filter_step(problem, state)
        records.append(record)
    assert jnp.allclose(state.mean, result.final_state.mean)
    assert jnp.allclose(
        jnp.stack([record.filtered_mean for record in records]),
        result.filtered_means,
    )


def test_missing_observation_is_exact_forecast_only_update():
    problem = _problem(mask=jnp.asarray([[True], [False]]))
    result = phx.uq.kalman_filter(problem)

    assert jnp.allclose(result.filtered_means[1], result.predicted_means[1])
    assert jnp.allclose(result.filtered_covariances[1], result.predicted_covariances[1])
    assert result.observed_counts[1] == 0
    assert result.incremental_log_likelihood[1] == 0.0


def test_rts_terminal_identity_covariance_contraction_and_coherent_paths():
    problem = _problem()
    filtered = phx.uq.kalman_filter(problem)
    smoothed = phx.uq.rts_smoother(filtered)
    paths = phx.uq.sample_kalman_smoother_paths(jr.key(4), smoothed, sample_shape=(4096,))

    assert jnp.allclose(smoothed.means[-1], filtered.filtered_means[-1])
    assert jnp.allclose(smoothed.covariances[-1], filtered.filtered_covariances[-1])
    assert jnp.all(
        jnp.linalg.eigvalsh(filtered.filtered_covariances - smoothed.covariances)
        >= -1e-10
    )
    assert paths.shape == (4096, 2, 1)
    assert jnp.allclose(jnp.mean(paths[:, 0, 0]), smoothed.means[0, 0], atol=0.04)
    assert jnp.allclose(jnp.var(paths[:, 0, 0]), smoothed.covariances[0, 0, 0], atol=0.04)
    assert jnp.corrcoef(paths[:, 0, 0], paths[:, 1, 0])[0, 1] > 0.0


def test_irregular_padded_cases_preserve_last_valid_state():
    problem = _problem(
        case_shape=(2,),
        step_valid=jnp.asarray([[True, True], [True, False]]),
        mask=jnp.asarray([[[True], [True]], [[True], [False]]]),
    )
    result = phx.uq.kalman_filter(problem)
    smoothed = phx.uq.rts_smoother(result)

    assert result.filtered_means.shape == (2, 2, 1)
    assert jnp.allclose(result.filtered_means[1, 1], result.filtered_means[1, 0])
    assert jnp.allclose(smoothed.means[1, 1], result.filtered_means[1, 1])
    assert jnp.all(result.successful)


def test_kalman_diagnostics_detect_finite_psd_results():
    result = phx.uq.kalman_filter(_problem())
    diagnostics = phx.uq.kalman_innovation_diagnostics(result)

    assert diagnostics.passed
    assert diagnostics.valid_steps == 2
    assert diagnostics.minimum_filtered_covariance_eigenvalue.shape == (2,)
