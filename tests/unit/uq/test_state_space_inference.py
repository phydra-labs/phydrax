import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _linear_problem(*, observation_offset=0.0):
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[1.0], [2.0]]),
        case_ids=("only",),
        sequence_id="exact-linear-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="exact-linear-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        process_id="exact-linear-process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
        offset=jnp.asarray(observation_offset),
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="exact-linear-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="exact-linear-problem",
    )


def _finite_problem():
    process = phx.stochastic.JumpProcess(
        lambda time, state, args: jnp.where(
            state[0] == 0, jnp.asarray([1.0]), jnp.asarray([0.0])
        ),
        lambda state, channel, mark, args: jnp.ones_like(state),
        state_shape=(1,),
        num_channels=1,
        process_id="finite-birth",
    )
    states = jnp.asarray([[0], [1]])
    generator = phx.solver.finite_state_generator(process, states)
    prior = phx.stochastic.CategoricalStatePrior(
        states,
        jnp.asarray([1.0, 0.0]),
        prior_id="finite-prior",
    )
    transition = phx.stochastic.FiniteStateTransitionKernel(generator)
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([1.0]),
        jnp.asarray([[1.0]]),
        case_ids=("only",),
        sequence_id="finite-sequence",
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="finite-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="finite-problem",
    )


def test_exact_linear_likelihood_matches_canonical_kalman_filter():
    problem = _linear_problem()
    expected = phx.uq.kalman_filter(problem)
    result = phx.uq.exact_state_space_log_likelihood(problem)

    assert result.method == "kalman"
    assert result.successful
    assert jnp.allclose(
        result.per_case_log_likelihood,
        expected.final_state.log_likelihood,
    )
    assert jnp.allclose(
        result.incremental_log_likelihood,
        expected.incremental_log_likelihood,
    )


def test_exact_finite_state_likelihood_matches_enumerated_mixture():
    problem = _finite_problem()
    result = phx.uq.exact_state_space_log_likelihood(problem)
    transition_mass = jnp.asarray([jnp.exp(-1.0), 1.0 - jnp.exp(-1.0)])
    observation_log_mass = jax.vmap(
        lambda state: problem.model.observation.log_prob(
            jnp.asarray([1.0]), state, jnp.asarray(1.0), jnp.asarray([True])
        )
    )(problem.model.prior.states)
    expected = jax.scipy.special.logsumexp(
        jnp.log(transition_mass) + observation_log_mass
    )

    assert result.method == "finite-state"
    assert result.successful
    assert result.status[0] == phx.uq.EXACT_STATE_SPACE_SUCCESS
    assert result.total_log_likelihood == pytest.approx(expected)
    assert jnp.allclose(result.backend.predicted_probabilities[0], transition_mass)
    assert jnp.allclose(jnp.sum(result.backend.filtered_probabilities[0]), 1.0)


def test_state_space_term_is_differentiable_and_reports_identifiability():
    template = _linear_problem()

    def build_problem(parameters):
        return eqx.tree_at(
            lambda problem: problem.model.observation.offset,
            template,
            parameters["offset"],
        )

    term = phx.uq.StateSpaceMarginalLikelihood(build_problem)
    parameters = {"offset": jnp.asarray(0.0)}
    gradient = jax.grad(term.log_prob)(parameters)
    report = phx.uq.state_space_identifiability(term, parameters)
    parameter_space = phx.uq.ParameterSpace(
        parameters,
        log_prior=lambda values: -0.5 * values["offset"] ** 2,
    )
    posterior = phx.uq.PosteriorProblem.from_terms(parameter_space, (term,))
    value, posterior_gradient = posterior.validate()

    assert jnp.isfinite(gradient["offset"])
    assert jnp.isfinite(value)
    assert jnp.isfinite(posterior_gradient["offset"])
    assert report.finite
    assert report.dimension == 1
    assert report.numerical_rank == 1
    assert report.full_rank
    assert report.observed_information.shape == (1, 1)


def test_exact_dispatch_rejects_incompatible_backend():
    with pytest.raises(TypeError, match="finite-state likelihood requires"):
        phx.uq.exact_state_space_log_likelihood(_linear_problem(), method="finite-state")
