#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _problem(values, *, problem_id):
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray(values).reshape((2, 1)),
        case_ids=("only",),
        sequence_id=problem_id,
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
            model_id="amortized-model",
        ),
        observations,
        initial_time=0.0,
        problem_id=problem_id,
    )


def test_amortized_family_is_normalized_and_conditions_on_observations():
    first_problem = _problem([1.0, 2.0], problem_id="first")
    second_problem = _problem([-1.0, -2.0], problem_id="second")
    family = phx.uq.AmortizedGaussianMarkovFamily.from_problem(
        first_problem,
        hidden_size=8,
        key=jax.random.key(1),
    )
    conditioned = family.condition(second_problem)
    first_states, first_log_prob = family.sample_and_log_prob(
        jax.random.key(2), sample_shape=(16,)
    )
    second_states, second_log_prob = conditioned.sample_and_log_prob(
        jax.random.key(2), sample_shape=(16,)
    )

    assert first_states.shape == second_states.shape == (16, 3, 1)
    assert jnp.array_equal(first_log_prob, family.log_prob(first_states))
    assert jnp.array_equal(second_log_prob, conditioned.log_prob(second_states))
    assert not jnp.array_equal(
        family.conditional_family.offsets,
        conditioned.conditional_family.offsets,
    )
    assert jax.tree.structure(family.encoder) == jax.tree.structure(conditioned.encoder)


def test_amortized_full_path_training_returns_reusable_encoder():
    problem = _problem([1.0, 2.0], problem_id="training")
    result = phx.uq.fit_amortized_state_space_variational(
        problem,
        key=jax.random.key(3),
        config=phx.uq.AmortizedStateSpaceVariationalConfig(
            optimization=phx.uq.VariationalConfig(
                num_steps=80,
                samples_per_step=8,
                learning_rate=0.01,
                record_every=10,
            ),
            hidden_size=8,
        ),
        num_samples=32,
    )
    new_family = result.family.condition(_problem([0.0, 0.5], problem_id="deployment"))
    new_states, new_log_prob = new_family.sample_and_log_prob(
        jax.random.key(4),
        sample_shape=(8,),
    )

    assert result.states.shape == (32, 3, 1)
    assert result.family.family_id == "amortized-gaussian-markov-path"
    assert jnp.all(result.diagnostics.finite)
    assert new_states.shape == (8, 3, 1)
    assert jnp.all(jnp.isfinite(new_log_prob))
