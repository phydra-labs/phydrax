import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[1.0], [2.0]]),
        case_ids=("only",),
        sequence_id="kalman-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
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


def _three_step_problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0, 1.5]),
        jnp.asarray([[1.0], [2.0], [1.5]]),
        case_ids=("only",),
        sequence_id="extended-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
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


def test_resampling_utilities_are_fixed_size_bounded_and_weighted():
    log_weights = jnp.log(jnp.asarray([0.8, 0.15, 0.05]))
    for method in ("systematic", "stratified", "multinomial", "residual"):
        indices = phx.uq.resample_indices(jr.key(2), log_weights, method=method)
        assert indices.shape == (3,)
        assert jnp.all((indices >= 0) & (indices < 3))

    draws = jnp.concatenate(
        [
            phx.uq.resample_indices(
                jr.fold_in(jr.key(3), index), log_weights, method="multinomial"
            )
            for index in range(256)
        ]
    )
    frequencies = jnp.bincount(draws, length=3) / draws.size
    assert jnp.allclose(frequencies, jnp.exp(log_weights), atol=0.05)
    assert jnp.allclose(phx.uq.effective_sample_size(jnp.zeros(8)), 8.0)
    with pytest.raises(ValueError, match="degenerate"):
        phx.uq.resample_indices(jr.key(0), jnp.full((3,), -jnp.inf))


def test_bootstrap_filter_matches_linear_gaussian_marginals():
    problem = _problem()
    kalman = phx.uq.kalman_filter(problem)
    particles = phx.uq.bootstrap_particle_filter(
        jr.key(10),
        problem,
        num_particles=512,
        resampling_policy="never",
    )
    weights = jnp.exp(particles.posterior_log_weights)
    means = jnp.sum(weights[..., None] * particles.predicted_particles, axis=-2)

    assert jnp.allclose(means, kalman.filtered_means, atol=0.12)
    assert jnp.allclose(
        particles.final_state.log_likelihood,
        kalman.final_state.log_likelihood,
        atol=0.2,
    )
    assert phx.uq.particle_filter_diagnostics(particles).passed


def test_bootstrap_filter_propagates_sampled_inputs_without_changing_noise_stream():
    base = _problem()
    input_signal = phx.stochastic.SampledStateSpaceInput(
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.asarray([[0.0], [1.0], [2.0]]),
        interpolation="linear",
        input_id="particle-input",
    )

    def input_driven_sample(key, state, t0, t1, context):
        sample = base.model.transition.sample(key, state, t0, t1, context)
        return sample.values + context.transition_end_input

    driven_transition = phx.stochastic.CallableTransitionKernel(
        input_driven_sample,
        state_shape=(1,),
        process_id=base.model.transition.process_id,
        approximation_id="input-driven",
    )
    driven_problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            base.model.prior,
            driven_transition,
            base.model.observation,
            model_id=base.model.model_id,
        ),
        base.observations,
        initial_time=base.initial_time,
        input_signal=input_signal,
        problem_id=base.problem_id,
    )

    baseline = phx.uq.bootstrap_particle_filter(
        jr.key(21), base, num_particles=8, resampling_policy="never"
    )
    driven = phx.uq.bootstrap_particle_filter(
        jr.key(21), driven_problem, num_particles=8, resampling_policy="never"
    )

    assert jnp.allclose(
        driven.predicted_particles - baseline.predicted_particles,
        jnp.asarray([1.0, 3.0])[:, None, None],
    )


def test_genealogy_backward_smoothing_and_predictive_conversion():
    result = phx.uq.bootstrap_particle_filter(
        jr.key(11),
        _problem(),
        num_particles=64,
        resampling_policy="always",
    )
    ancestry = phx.uq.sample_particle_ancestry_paths(
        jr.key(12), result, sample_shape=(16,)
    )
    backward = phx.uq.sample_particle_backward_paths(
        jr.key(13), result, sample_shape=(16,)
    )
    predictive = phx.uq.particle_filter_predictive(jr.key(14), result)

    assert ancestry.shape == backward.shape == (16, 2, 1)
    assert jnp.all(jnp.isfinite(ancestry))
    assert jnp.all(jnp.isfinite(backward))
    assert result.ancestor_indices.shape == (2, 64)
    assert predictive.samples.data.shape == (2, 64, 1)
    assert predictive.sample_axes[0].source == "process"


def test_replay_is_exact_and_schedule_extension_preserves_prefix():
    short = phx.uq.bootstrap_particle_filter(
        jr.key(15), _problem(), num_particles=32, resampling_policy="always"
    )
    replay = phx.uq.bootstrap_particle_filter(
        jr.key(15), _problem(), num_particles=32, resampling_policy="always"
    )
    extended = phx.uq.bootstrap_particle_filter(
        jr.key(15),
        _three_step_problem(),
        num_particles=32,
        resampling_policy="always",
    )

    assert jnp.array_equal(short.particles, replay.particles)
    assert jnp.array_equal(short.ancestor_indices, replay.ancestor_indices)
    assert jnp.array_equal(short.particles, extended.particles[:2])
    assert jnp.array_equal(short.ancestor_indices, extended.ancestor_indices[:2])


def test_particle_checkpoint_resumes_exactly_and_rejects_wrong_settings(tmp_path):
    problem = _problem()
    state = phx.uq.initialize_particle_filter(
        jr.key(16), problem, num_particles=32, resampling_policy="always"
    )
    state, _ = phx.uq.particle_filter_step(problem, state)
    path = tmp_path / "particle-filter.zip"
    phx.uq.write_particle_filter_checkpoint(path, problem, state)
    restored = phx.uq.read_particle_filter_checkpoint(
        path, problem, num_particles=32, resampling_policy="always"
    )
    expected, expected_step = phx.uq.particle_filter_step(problem, state)
    resumed, resumed_step = phx.uq.particle_filter_step(problem, restored)

    assert jnp.array_equal(expected.particles, resumed.particles)
    assert jnp.array_equal(expected_step.ancestor_indices, resumed_step.ancestor_indices)
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        phx.uq.read_particle_filter_checkpoint(
            path, problem, num_particles=32, resampling_policy="never"
        )


def test_particle_filter_reports_all_invalid_likelihoods():
    base = _problem()
    invalid_observation = phx.stochastic.CallableObservationModel(
        lambda state, time, context: jnp.zeros((1,)),
        lambda value, state, time, mask, context: jnp.asarray(-jnp.inf),
        lambda key, state, time, sample_shape, context: jnp.zeros(sample_shape + (1,)),
        state_shape=(1,),
        observation_shape=(1,),
        observation_id="invalid",
    )
    model = phx.stochastic.StateSpaceModel(
        base.model.prior,
        base.model.transition,
        invalid_observation,
        model_id="invalid-model",
    )
    problem = phx.stochastic.StateSpaceProblem(
        model,
        base.observations,
        initial_time=0.0,
        problem_id="invalid-problem",
    )
    result = phx.uq.bootstrap_particle_filter(jr.key(17), problem, num_particles=16)

    assert not result.successful
    assert result.status[0] == phx.uq.PARTICLE_FILTER_WEIGHT_DEGENERACY
    with pytest.raises(RuntimeError, match="failed"):
        phx.uq.bootstrap_particle_filter(
            jr.key(17), problem, num_particles=16, raise_on_failure=True
        )


def test_particle_posterior_measure_matches_weighted_filtering_marginals():
    result = phx.uq.bootstrap_particle_filter(
        jr.key(18),
        _problem(),
        num_particles=128,
        resampling_policy="never",
    )
    target = phx.uq.particle_posterior_measure(result)

    estimate = phx.integration.integrate(lambda particles: particles, target)

    weights = jnp.exp(result.posterior_log_weights)
    expected = jnp.sum(weights[..., None] * result.predicted_particles, axis=-2)
    assert estimate.value.dims == ("time", None)
    assert jnp.allclose(estimate.value.data, expected)
    assert jnp.all(estimate.successful)
    assert jnp.array_equal(
        estimate.diagnostics.active_samples,
        jnp.full((2,), result.num_particles),
    )
    assert not estimate.diagnostics.independent
    assert estimate.error_estimate is None
    assert jnp.array_equal(
        estimate.diagnostics.ancestry_ids,
        result.ancestor_indices,
    )


def test_particle_posterior_measure_masks_failed_filtering_steps():
    base = _problem()
    invalid_observation = phx.stochastic.CallableObservationModel(
        lambda state, time, context: jnp.zeros((1,)),
        lambda value, state, time, mask, context: jnp.asarray(-jnp.inf),
        lambda key, state, time, sample_shape, context: jnp.zeros(sample_shape + (1,)),
        state_shape=(1,),
        observation_shape=(1,),
        observation_id="integration-invalid",
    )
    problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            base.model.prior,
            base.model.transition,
            invalid_observation,
            model_id="integration-invalid-model",
        ),
        base.observations,
        initial_time=0.0,
        problem_id="integration-invalid-problem",
    )
    result = phx.uq.bootstrap_particle_filter(
        jr.key(19),
        problem,
        num_particles=16,
    )

    estimate = phx.integration.integrate(
        lambda particles: particles,
        phx.uq.particle_posterior_measure(result),
    )

    assert jnp.all(
        estimate.status == int(phx.integration.IntegrationStatus.NO_VALID_SAMPLES)
    )
    assert jnp.all(jnp.isnan(estimate.value.data))
