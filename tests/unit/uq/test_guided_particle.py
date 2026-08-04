import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([1.0]),
        jnp.asarray([[1.0]]),
        case_ids=("only",),
        sequence_id="guided-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="guided-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.25]]),
        state_shape=(1,),
        process_id="guided-process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="guided-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="guided-problem",
    )


def test_fully_adapted_auxiliary_filter_has_exact_incremental_correction():
    problem = _problem()
    proposal = phx.uq.LinearGaussianGuidedParticleProposal((1,))
    result = phx.uq.guided_particle_filter(
        jr.key(4),
        problem,
        proposal,
        num_particles=512,
        auxiliary_resampling_policy="always",
        resampling_policy="never",
    )
    exact = phx.uq.exact_state_space_log_likelihood(problem)
    predicted = result.predicted_particles[0]
    observation_log_prob = jax.vmap(
        lambda state: problem.model.observation.log_prob(
            jnp.asarray([1.0]),
            state,
            jnp.asarray(1.0),
            jnp.asarray([True]),
        )
    )(predicted)
    parent_lookahead = result.auxiliary_log_weights[0][
        result.proposal_ancestor_indices[0]
    ]

    assert result.successful
    assert result.auxiliary_resampled[0]
    assert jnp.allclose(
        observation_log_prob + result.proposal_log_corrections[0],
        parent_lookahead,
        atol=1e-5,
    )
    assert jnp.allclose(jnp.sum(jnp.exp(result.log_weights[0])), 1.0)
    assert jnp.allclose(
        result.final_state.log_likelihood,
        exact.total_log_likelihood,
        atol=0.08,
    )


def test_callable_guided_proposal_computes_target_density_correction():
    problem = _problem()

    def sample(key, current_problem, previous, t0, t1, observation, mask):
        del observation, mask
        return current_problem.model.transition.sample(key, previous, t0, t1)

    def log_prob(
        proposed,
        current_problem,
        previous,
        t0,
        t1,
        observation,
        mask,
    ):
        del observation, mask
        return current_problem.model.transition.log_prob(proposed, previous, t0, t1)

    proposal = phx.uq.CallableGuidedParticleProposal(
        sample,
        log_prob,
        state_shape=(1,),
        proposal_id="transition-equivalent",
    )
    result = phx.uq.guided_particle_filter(
        jr.key(8),
        problem,
        proposal,
        num_particles=32,
        auxiliary_resampling_policy="never",
        resampling_policy="never",
    )

    assert result.successful
    assert result.proposal_id == "transition-equivalent"
    assert jnp.allclose(result.proposal_log_corrections, 0.0)
    assert jnp.array_equal(result.proposal_ancestor_indices[0], jnp.arange(32))
    assert not result.auxiliary_resampled[0]


def test_guided_filter_rejects_proposal_shape_mismatch():
    with jax.disable_jit():
        proposal = phx.uq.BootstrapParticleProposal((2,))
        try:
            phx.uq.guided_particle_filter(
                jr.key(1), _problem(), proposal, num_particles=4
            )
        except ValueError as error:
            assert "state shapes do not match" in str(error)
        else:
            raise AssertionError("Expected a proposal/model shape mismatch.")
