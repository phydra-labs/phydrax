#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0, 1.5, 2.0]),
        jnp.asarray([[1.0], [2.0], [1.5], [1.8]]),
        case_ids=("only",),
        sequence_id="buffered",
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
            model_id="buffered-model",
        ),
        observations,
        initial_time=0.0,
        problem_id="buffered-problem",
    )


def test_window_inclusion_weights_reconstruct_every_full_path_factor():
    plan = phx.uq.StateSpaceWindowPlan(
        6,
        target_length=3,
        left_buffer=2,
        right_buffer=1,
    )
    masks = []
    for start in range(plan.num_starts):
        indices = jnp.arange(plan.num_steps)
        masks.append((indices >= start) & (indices < start + plan.target_length))
    target_masks = jnp.stack(masks)
    weighted_average = jnp.mean(
        target_masks / plan.inclusion_probability[None, :],
        axis=0,
    )

    assert jnp.allclose(weighted_average, jnp.ones((plan.num_steps,)))
    assert jnp.all(plan.inclusion_probability > 0.0)


def test_buffer_context_changes_bidirectional_amortized_conditioning():
    problem = _problem()
    family = phx.uq.AmortizedGaussianMarkovFamily.from_problem(
        problem,
        hidden_size=8,
        key=jax.random.key(1),
    )
    short_context = jnp.asarray([True, True, False, False])
    long_context = jnp.asarray([True, True, True, True])
    short = eqx.tree_at(
        lambda value: value.context_mask,
        family,
        short_context,
    )
    long = eqx.tree_at(
        lambda value: value.context_mask,
        family,
        long_context,
    )

    assert not jnp.array_equal(
        short.conditional_family.offsets,
        long.conditional_family.offsets,
    )


def test_buffered_variational_training_replays_and_returns_full_context_family():
    problem = _problem()
    config = phx.uq.BufferedStateSpaceVariationalConfig(
        target_length=2,
        left_buffer=1,
        right_buffer=1,
        hidden_size=8,
        optimization=phx.uq.VariationalConfig(
            num_steps=30,
            samples_per_step=4,
            learning_rate=0.01,
            record_every=5,
        ),
    )
    first = phx.uq.fit_buffered_state_space_variational(
        problem,
        key=jax.random.key(2),
        config=config,
        num_samples=16,
    )
    replay = phx.uq.fit_buffered_state_space_variational(
        problem,
        key=jax.random.key(2),
        config=config,
        num_samples=16,
    )

    assert jnp.array_equal(first.states, replay.states)
    assert jnp.array_equal(
        first.diagnostics.target_start,
        replay.diagnostics.target_start,
    )
    assert jnp.all(first.diagnostics.finite)
    assert first.approximation_id == "buffered-amortized-gaussian-markov-path"
    assert jnp.array_equal(first.family.context_mask, problem.observations.step_valid)


def test_full_length_target_has_unit_inclusion_and_full_context():
    plan = phx.uq.StateSpaceWindowPlan(
        4,
        target_length=4,
        left_buffer=0,
        right_buffer=0,
    )
    window = plan.sample(jax.random.key(3))

    assert int(window.target_start) == 0
    assert jnp.all(window.target_mask)
    assert jnp.all(window.context_mask)
    assert jnp.all(plan.inclusion_probability == 1.0)
