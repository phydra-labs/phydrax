#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _problem(*, step_valid=None):
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0, 1.5]),
        jnp.asarray([[1.0], [2.0], [2.0]]),
        step_valid=step_valid,
        case_ids=("only",),
        sequence_id="state-space-vi",
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
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="linear",
        ),
        observations,
        initial_time=0.0,
        problem_id="linear-state-space-vi",
    )


def test_state_space_path_density_decomposes_normalized_model_terms():
    problem = _problem()
    states = jnp.asarray([[0.0], [0.5], [1.0], [1.5]])
    result = jax.jit(phx.uq.state_space_path_log_density)(problem, states)

    assert bool(result.valid)
    assert result.prior.shape == ()
    assert result.transition.shape == (3,)
    assert result.observation.shape == (3,)
    assert jnp.allclose(
        result.log_density,
        result.prior + jnp.sum(result.transition) + jnp.sum(result.observation),
    )


def test_state_space_path_density_requires_frozen_padding():
    problem = _problem(step_valid=jnp.asarray([True, True, False]))
    frozen = phx.uq.state_space_path_log_density(
        problem,
        jnp.asarray([[0.0], [0.5], [1.0], [1.0]]),
    )
    changed = phx.uq.state_space_path_log_density(
        problem,
        jnp.asarray([[0.0], [0.5], [1.0], [1.2]]),
    )

    assert bool(frozen.valid)
    assert not bool(changed.valid)
    assert jnp.isneginf(changed.log_density)
    assert frozen.transition[-1] == 0.0
    assert frozen.observation[-1] == 0.0


def test_gaussian_markov_family_sampling_and_density_are_consistent():
    problem = _problem(step_valid=jnp.asarray([True, True, False]))
    family = phx.uq.GaussianMarkovVariationalFamily.from_problem(problem)
    states, sampled_log_prob = family.sample_and_log_prob(
        jax.random.key(1),
        sample_shape=(32,),
    )

    assert states.shape == (32, 4, 1)
    assert sampled_log_prob.shape == (32,)
    assert jnp.array_equal(sampled_log_prob, family.log_prob(states))
    assert jnp.array_equal(states[:, -1], states[:, -2])


def test_full_path_variational_matches_linear_gaussian_smoother_means():
    problem = _problem()
    exact = phx.uq.rts_smoother(phx.uq.kalman_filter(problem))
    result = phx.uq.fit_state_space_variational(
        problem,
        key=jax.random.key(2),
        config=phx.uq.StateSpaceVariationalConfig(
            optimization=phx.uq.VariationalConfig(
                num_steps=500,
                samples_per_step=32,
                learning_rate=0.02,
                record_every=25,
            ),
            initial_scale=0.5,
        ),
        num_samples=2000,
    )
    variational_means = jnp.mean(result.states[:, 1:], axis=0)

    assert jnp.all(result.diagnostics.finite)
    assert jnp.allclose(variational_means, exact.means, atol=0.15, rtol=0.1)
    assert jnp.all(jnp.isfinite(result.log_model))
    assert jnp.all(jnp.isfinite(result.log_variational))
