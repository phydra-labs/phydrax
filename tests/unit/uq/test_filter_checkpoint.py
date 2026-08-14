import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[1.0], [2.0]]),
        case_ids=("only",),
        sequence_id="checkpoint-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="checkpoint-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        process_id="checkpoint-process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="checkpoint-model"
    )
    return phx.stochastic.StateSpaceProblem(
        model, observations, initial_time=0.0, problem_id="checkpoint-problem"
    )


def test_kalman_filter_checkpoint_replays_streaming_state(tmp_path):
    problem = _problem()
    initial = phx.uq.initialize_kalman_filter(problem, covariance_regularization=1e-8)
    state, _ = phx.uq.kalman_filter_step(problem, initial)
    path = tmp_path / "kalman.phxckpt"

    phx.uq.write_filter_checkpoint(path, problem, state)
    restored = phx.uq.read_filter_checkpoint(
        path,
        problem,
        "kalman",
        covariance_regularization=1e-8,
    )
    expected, _ = phx.uq.kalman_filter_step(problem, state)
    replay, _ = phx.uq.kalman_filter_step(problem, restored)

    assert restored.step_index == 1
    assert jnp.array_equal(restored.mean, state.mean)
    assert jnp.array_equal(replay.mean, expected.mean)
    assert jnp.array_equal(replay.covariance, expected.covariance)
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        phx.uq.read_filter_checkpoint(
            path,
            problem,
            "kalman",
            covariance_regularization=1e-6,
        )


def test_bellman_filter_checkpoint_replays_and_rejects_changed_numerics(tmp_path):
    problem = _problem()
    initial = phx.uq.initialize_bellman_filter(problem)
    state, _ = phx.uq.bellman_filter_step(problem, initial)
    path = tmp_path / "bellman.phxckpt"

    phx.uq.write_filter_checkpoint(path, problem, state)
    restored = phx.uq.read_filter_checkpoint(path, problem, "bellman")
    expected, expected_step = phx.uq.bellman_filter_step(problem, state)
    replay, replay_step = phx.uq.bellman_filter_step(problem, restored)

    assert jnp.array_equal(restored.mode, state.mode)
    assert jnp.array_equal(restored.information, state.information)
    assert jnp.array_equal(replay.mode, expected.mode)
    assert jnp.array_equal(replay.pseudo_log_likelihood, expected.pseudo_log_likelihood)
    assert jnp.array_equal(replay_step.filtered_mode, expected_step.filtered_mode)
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        phx.uq.read_filter_checkpoint(
            path,
            problem,
            "bellman",
            bellman_method="optimization",
        )


def test_ensemble_filter_checkpoint_replays_key_and_members(tmp_path):
    problem = _problem()
    initial = phx.uq.initialize_ensemble_filter(
        jr.key(60),
        problem,
        ensemble_size=16,
        inflation=1.01,
        covariance_regularization=1e-8,
    )
    state, _ = phx.uq.ensemble_filter_step(problem, initial)
    path = tmp_path / "ensemble.phxckpt"

    phx.uq.write_ensemble_filter_checkpoint(path, problem, state)
    restored = phx.uq.read_ensemble_filter_checkpoint(
        path,
        problem,
        ensemble_size=16,
        inflation=1.01,
        covariance_regularization=1e-8,
    )
    expected, _ = phx.uq.ensemble_filter_step(problem, state)
    replay, _ = phx.uq.ensemble_filter_step(problem, restored)

    assert jnp.array_equal(restored.ensemble, state.ensemble)
    assert jnp.array_equal(jr.key_data(restored.root_key), jr.key_data(state.root_key))
    assert jnp.array_equal(replay.ensemble, expected.ensemble)
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        phx.uq.read_ensemble_filter_checkpoint(
            path,
            problem,
            ensemble_size=8,
            inflation=1.01,
            covariance_regularization=1e-8,
        )


def test_unified_particle_checkpoint_dispatch(tmp_path):
    problem = _problem()
    state = phx.uq.initialize_particle_filter(
        jr.key(61),
        problem,
        num_particles=16,
        resampling_policy="always",
    )
    path = tmp_path / "particle.phxckpt"

    phx.uq.write_filter_checkpoint(path, problem, state)
    restored = phx.uq.read_filter_checkpoint(
        path,
        problem,
        "particle",
        num_particles=16,
        resampling_policy="always",
    )

    assert jnp.array_equal(restored.particles, state.particles)
    assert jnp.array_equal(restored.log_weights, state.log_weights)
