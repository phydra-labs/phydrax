import jax
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


def _input_driven_problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([[0.5, 1.0], [0.5, 1.0]]),
        jnp.zeros((2, 2, 1)),
        case_axes=("case",),
        case_shape=(2,),
        case_ids=("low-input", "high-input"),
        sequence_id="input-driven-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((2, 1)),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="input-driven-prior",
    )
    input_signal = phx.stochastic.SampledStateSpaceInput(
        jnp.asarray([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]),
        jnp.asarray([[[1.0], [2.0], [4.0]], [[10.0], [20.0], [40.0]]]),
        interpolation="linear",
        input_id="time-varying-input",
    )

    def transition_matrix(t0, t1, context):
        del t0, t1
        return context.transition_end_input.reshape((1, 1))

    def transition_offset(t0, t1, context):
        del t0, t1
        return context.args["transition_offset_scale"] * context.transition_start_input

    transition = phx.stochastic.LinearGaussianTransitionKernel(
        transition_matrix,
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        offset=transition_offset,
        process_id="input-driven-process",
    )

    def observation_matrix(time, context):
        del time
        return context.observation_input.reshape((1, 1))

    def observation_offset(time, context):
        del time
        return context.args["observation_offset_scale"] * context.observation_input

    observation = phx.stochastic.LinearGaussianObservationModel(
        observation_matrix,
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
        offset=observation_offset,
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="input-driven-linear",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="input-driven-problem",
        args={
            "transition_offset_scale": jnp.asarray(0.5),
            "observation_offset_scale": jnp.asarray(0.25),
        },
        input_signal=input_signal,
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


def test_typed_input_drives_kalman_transition_and_observation_parameters():
    result = phx.uq.kalman_filter(_input_driven_problem())
    expected_end_input = jnp.asarray([[2.0, 4.0], [20.0, 40.0]])

    assert result.input_id == "time-varying-input"
    assert jnp.allclose(
        result.transition_matrices[..., 0, 0],
        expected_end_input,
    )
    assert jnp.allclose(result.predicted_means[:, 0, 0], jnp.asarray([0.5, 5.0]))
    expected_prediction = (
        expected_end_input * result.predicted_means[..., 0] + 0.25 * expected_end_input
    )
    assert jnp.allclose(result.innovations[..., 0], -expected_prediction)


def test_context_indices_and_input_parameters_survive_jit_vmap_and_scan():
    problem = _input_driven_problem()
    indices = jnp.asarray([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=jnp.int32)

    def inspect(index):
        context = problem.step_context(index[0], index[1])
        transition = problem.model.transition.parameters(0.0, 1.0, context)
        observation_matrix, observation_offset, _ = problem.model.observation.parameters(
            1.0, context
        )
        return jnp.stack(
            (
                context.case_index,
                context.step_index,
                context.transition_start_input[0],
                context.transition_end_input[0],
                transition.transition[0, 0],
                transition.offset[0],
                observation_matrix[0, 0],
                observation_offset[0],
            )
        )

    @jax.jit
    def transformed(index_pairs):
        vmapped = jax.vmap(inspect)(index_pairs)
        _, scanned = jax.lax.scan(
            lambda carry, index: (carry, inspect(index)),
            None,
            index_pairs,
        )
        return vmapped, scanned

    vmapped, scanned = transformed(indices)
    expected = jnp.asarray(
        [
            [0.0, 1.0, 2.0, 4.0, 4.0, 1.0, 4.0, 1.0],
            [1.0, 0.0, 10.0, 20.0, 20.0, 5.0, 20.0, 5.0],
            [1.0, 1.0, 20.0, 40.0, 40.0, 10.0, 40.0, 10.0],
            [0.0, 0.0, 1.0, 2.0, 2.0, 0.5, 2.0, 0.5],
        ]
    )

    assert jnp.array_equal(vmapped[:, :2], indices)
    assert jnp.array_equal(scanned[:, :2], indices)
    assert jnp.allclose(vmapped, expected)
    assert jnp.allclose(scanned, expected)


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
