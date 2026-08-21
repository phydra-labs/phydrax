#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0, 1.5]),
        jnp.asarray([[1.0], [2.0], [1.5]]),
        case_ids=("only",),
        sequence_id="genealogical-score",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        offset=jnp.asarray([0.1]),
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
            model_id="linear-score-model",
        ),
        observations,
        initial_time=0.0,
        problem_id="linear-score-problem",
    )


@pytest.mark.parametrize("resampling_policy", ("never", "always", "ess"))
def test_genealogical_score_covers_complete_model_and_resampling(resampling_policy):
    filtered = phx.uq.bootstrap_particle_filter(
        jax.random.key(1),
        _problem(),
        num_particles=32,
        resampling_policy=resampling_policy,
        resampling_threshold=0.8,
    )
    score = phx.uq.particle_genealogical_score(filtered)

    assert bool(score.valid)
    assert score.flat_score.shape == (score.parameter_size,)
    assert score.case_scores.shape == (score.parameter_size,)
    assert jnp.all(jnp.isfinite(score.flat_score))
    assert any(path.startswith(".prior") for path in score.parameter_paths)
    assert any(path.startswith(".transition") for path in score.parameter_paths)
    assert any(path.startswith(".observation") for path in score.parameter_paths)
    assert score.method_id == "particle-complete-model-genealogical-score"
    assert score.ancestry_gradient == "stopped-realized-ancestry"


def test_genealogical_score_replays_exactly_with_semantic_particle_keys():
    problem = _problem()
    first_filter = phx.uq.bootstrap_particle_filter(
        jax.random.key(2),
        problem,
        num_particles=24,
        resampling_policy="always",
    )
    second_filter = phx.uq.bootstrap_particle_filter(
        jax.random.key(2),
        problem,
        num_particles=24,
        resampling_policy="always",
    )
    first = phx.uq.particle_genealogical_score(first_filter)
    second = phx.uq.particle_genealogical_score(second_filter)

    assert jnp.array_equal(
        first_filter.initial_particles, second_filter.initial_particles
    )
    assert jnp.array_equal(first_filter.ancestor_indices, second_filter.ancestor_indices)
    assert jnp.array_equal(first.flat_score, second.flat_score)


def test_genealogical_score_cost_state_has_linear_particle_shape():
    problem = _problem()
    small = phx.uq.particle_genealogical_score(
        phx.uq.bootstrap_particle_filter(
            jax.random.key(3),
            problem,
            num_particles=8,
            resampling_policy="never",
        )
    )
    large = phx.uq.particle_genealogical_score(
        phx.uq.bootstrap_particle_filter(
            jax.random.key(3),
            problem,
            num_particles=16,
            resampling_policy="never",
        )
    )

    assert small.case_scores.shape == large.case_scores.shape
    assert small.parameter_size == large.parameter_size
    assert small.filter_result.initial_particles.shape[-2] == 8
    assert large.filter_result.initial_particles.shape[-2] == 16
