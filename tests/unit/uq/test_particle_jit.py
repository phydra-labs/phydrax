#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([[0.5, 1.0], [0.5, 1.0]]),
        jnp.asarray([[[1.0], [2.0]], [[-1.0], [-2.0]]]),
        case_axes=("case",),
        case_shape=(2,),
        case_ids=("positive", "negative"),
        sequence_id="particle-jit",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((2, 1)),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[0.9]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        process_id="latent",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.3]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="particle-jit-model",
        ),
        observations,
        initial_time=0.0,
        problem_id="particle-jit-problem",
    )


def test_bootstrap_particle_filter_runs_as_one_jitted_scan():
    problem = _problem()
    compiled = jax.jit(
        lambda key: phx.uq.bootstrap_particle_filter(
            key,
            problem,
            num_particles=16,
            resampling_policy="ess",
            resampling_threshold=0.8,
        )
    )
    result = compiled(jax.random.key(1))
    replay = compiled(jax.random.key(1))

    assert result.particles.shape == (2, 2, 16, 1)
    assert int(result.final_state.step_index) == 2
    assert jnp.array_equal(result.initial_particles, replay.initial_particles)
    assert jnp.array_equal(result.particles, replay.particles)
    assert jnp.array_equal(result.ancestor_indices, replay.ancestor_indices)
    assert jnp.array_equal(result.resampled, replay.resampled)


def test_jitted_scan_matches_streaming_particle_steps_exactly():
    problem = _problem()
    key = jax.random.key(2)
    scan = phx.uq.bootstrap_particle_filter(
        key,
        problem,
        num_particles=12,
        resampling_policy="always",
    )
    state = phx.uq.initialize_particle_filter(
        key,
        problem,
        num_particles=12,
        resampling_policy="always",
    )
    records = []
    for _ in range(problem.observations.num_steps):
        state, record = jax.jit(phx.uq.particle_filter_step)(problem, state)
        records.append(record)

    assert jnp.array_equal(scan.final_state.particles, state.particles)
    assert jnp.array_equal(scan.final_state.log_weights, state.log_weights)
    assert jnp.array_equal(
        scan.ancestor_indices,
        jnp.stack([record.ancestor_indices for record in records], axis=1),
    )
    assert jnp.array_equal(
        scan.cumulative_log_likelihood,
        jnp.stack([record.cumulative_log_likelihood for record in records], axis=1),
    )
