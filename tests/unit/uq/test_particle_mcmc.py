import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _problem(*, offset=0.0):
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[0.5], [1.0]]),
        case_ids=("only",),
        sequence_id="particle-mcmc-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="particle-mcmc-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        process_id="particle-mcmc-process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.3]]),
        state_shape=(1,),
        observation_shape=(1,),
        offset=jnp.asarray(offset),
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="particle-mcmc-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="particle-mcmc-problem",
    )


def test_conditional_smc_retains_reference_and_samples_complete_paths():
    reference = jnp.asarray([[0.0], [0.4], [0.9]])
    fixed = phx.uq.conditional_particle_filter(
        jr.key(2),
        _problem(),
        reference,
        num_particles=16,
        ancestor_sampling=False,
    )
    pgas = phx.uq.conditional_particle_filter(
        jr.key(3),
        _problem(),
        reference,
        num_particles=16,
        ancestor_sampling=True,
    )
    sampled = phx.uq.sample_conditional_particle_path(jr.key(4), pgas)

    assert fixed.successful
    assert pgas.successful
    assert jnp.array_equal(fixed.initial_particles[0], reference[0])
    assert jnp.array_equal(fixed.particles[:, 0], reference[1:])
    assert jnp.all(fixed.ancestor_indices[:, 0] == 0)
    assert jnp.array_equal(pgas.particles[:, 0], reference[1:])
    assert sampled.shape == reference.shape
    assert jnp.all(jnp.isfinite(sampled))
    assert jnp.allclose(jnp.sum(jnp.exp(pgas.log_weights), axis=-1), 1.0)


def test_particle_gibbs_returns_reproducible_pgas_path_chain():
    reference = jnp.asarray([[0.0], [0.4], [0.9]])
    first = phx.uq.particle_gibbs(
        jr.key(10),
        _problem(),
        reference,
        num_particles=12,
        num_samples=3,
        num_warmup=1,
    )
    second = phx.uq.particle_gibbs(
        jr.key(10),
        _problem(),
        reference,
        num_particles=12,
        num_samples=3,
        num_warmup=1,
    )

    assert first.paths.shape == (3, 3, 1)
    assert jnp.array_equal(first.paths, second.paths)
    assert jnp.array_equal(first.moved, second.moved)
    assert jnp.all(jnp.isfinite(first.log_likelihood_estimates))
    assert 0.0 <= first.movement_rate <= 1.0


def test_particle_marginal_metropolis_hastings_tracks_pseudo_marginal_chain():
    result = phx.uq.particle_marginal_metropolis_hastings(
        jr.key(20),
        {"offset": jnp.asarray(0.0)},
        lambda parameters: _problem(offset=parameters["offset"]),
        lambda parameters: -0.5 * parameters["offset"] ** 2,
        phx.uq.GaussianRandomWalkProposal(0.1),
        num_particles=16,
        num_samples=3,
        num_warmup=1,
    )

    assert result.samples["offset"].shape == (3,)
    assert result.log_likelihoods.shape == (3,)
    assert jnp.all(jnp.isfinite(result.log_posteriors))
    assert result.accepted.dtype == jnp.bool_
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.filter_evaluations == 5
