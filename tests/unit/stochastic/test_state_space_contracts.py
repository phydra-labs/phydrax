import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _linear_problem(*, values=None, mask=None):
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[1.0], [2.0]]) if values is None else values,
        observation_mask=mask,
        sequence_id="sequence",
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
        process_id="process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="model"
    )
    return phx.stochastic.StateSpaceProblem(
        model, observations, initial_time=0.0, problem_id="problem"
    )


def test_observation_sequence_requires_prefix_validity_and_explicit_missingness():
    sequence = phx.stochastic.ObservationSequence(
        jnp.asarray([[0.2, 0.5, 1.0], [0.3, 0.7, 0.7]]),
        jnp.ones((2, 3, 2)),
        case_axes=("case",),
        case_shape=(2,),
        observation_axes=("sensor",),
        step_valid=jnp.asarray([[True, True, True], [True, True, False]]),
        observation_mask=jnp.asarray(
            [
                [[True, True], [True, False], [False, False]],
                [[True, True], [True, True], [False, False]],
            ]
        ),
        case_ids=("a", "b"),
        sequence_id="irregular",
    )

    assert sequence.case_shape == (2,)
    assert sequence.observation_shape == (2,)
    assert not jnp.any(sequence.observation_mask[0, 2])

    with pytest.raises(ValueError, match="prefix"):
        phx.stochastic.ObservationSequence(
            jnp.asarray([0.0, 0.5, 1.0]),
            jnp.ones((3, 1)),
            step_valid=jnp.asarray([True, False, True]),
        )
    with pytest.raises(ValueError, match="finite"):
        phx.stochastic.ObservationSequence(jnp.asarray([0.0]), jnp.asarray([[jnp.nan]]))


def test_gaussian_and_categorical_priors_expose_density_semantics():
    gaussian = phx.stochastic.GaussianStatePrior(
        jnp.zeros((2, 1)),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="gaussian",
    )
    draws = gaussian.sample(jr.key(0), (5,))
    categorical = phx.stochastic.CategoricalStatePrior(
        jnp.asarray([[0.0], [1.0]]), jnp.asarray([0.25, 0.75])
    )

    assert draws.shape == (5, 2, 1)
    assert gaussian.log_prob(jnp.zeros((2, 1))).shape == (2,)
    assert jnp.allclose(categorical.log_prob(jnp.asarray([1.0])), jnp.log(0.75))

    singular = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]), jnp.asarray([[0.0]]), state_shape=(1,)
    )
    assert not singular.has_log_density
    with pytest.raises(ValueError, match="singular"):
        singular.log_prob(jnp.asarray([0.0]))


def test_linear_gaussian_roles_sample_and_normalize_masked_observations():
    problem = _linear_problem(mask=jnp.asarray([[True], [False]]))
    transition = problem.model.transition
    observation = problem.model.observation
    context = phx.stochastic.StateSpaceStepContext.empty()

    sample = transition.sample(jr.key(1), jnp.zeros((4, 1)), 0.0, 0.5, context)
    observed = observation.log_prob(
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        0.5,
        jnp.asarray([True]),
        context,
    )
    missing = observation.log_prob(
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        0.5,
        jnp.asarray([False]),
        context,
    )

    assert sample.values.shape == (4, 1)
    assert jnp.all(sample.valid)
    assert jnp.isfinite(observed)
    assert jnp.allclose(missing, 0.0)


def test_likelihood_observation_adapter_reduces_only_observed_components():
    model = phx.uq.LikelihoodObservationModel(
        phx.uq.GaussianLikelihood(0.5),
        lambda state, time, context: state + 0.0 * time + 0.0 * context.step_index,
        state_shape=(2,),
        observation_shape=(2,),
        observation_id="sensor",
    )
    context = phx.stochastic.StateSpaceStepContext.empty()
    value = model.log_prob(
        jnp.asarray([1.0, 50.0]),
        jnp.asarray([1.0, 0.0]),
        0.0,
        jnp.asarray([True, False]),
        context,
    )
    expected = phx.uq.GaussianLikelihood(0.5).log_prob(1.0, 1.0)

    assert jnp.allclose(value, expected)
    assert model.sample(
        jr.key(2), jnp.zeros(2), 0.0, context, sample_shape=(3,)
    ).shape == (3, 2)


def test_callable_adapters_receive_context_as_the_final_callback_argument():
    transition = phx.stochastic.CallableTransitionKernel(
        lambda key, state, t0, t1, context: state + (t1 - t0) * context.args["rate"],
        state_shape=(1,),
        process_id="context-transition",
        approximation_id="hand-checked",
        log_prob_fn=lambda next_state, state, t0, t1, context: (
            -jnp.sum((next_state - state - (t1 - t0) * context.args["rate"]) ** 2)
        ),
    )

    def observation_sample(key, state, time, sample_shape, context):
        del key
        location = state + time * context.args["slope"]
        return jnp.broadcast_to(location, sample_shape + location.shape)

    observation = phx.stochastic.CallableObservationModel(
        lambda state, time, context: state + time * context.args["slope"],
        lambda value, state, time, mask, context: (
            -jnp.sum(
                jnp.where(
                    mask,
                    (value - state - time * context.args["slope"]) ** 2,
                    0.0,
                )
            )
        ),
        observation_sample,
        state_shape=(1,),
        observation_shape=(1,),
        observation_id="context-observation",
    )
    context = phx.stochastic.StateSpaceStepContext.empty(
        args={"rate": jnp.asarray(2.0), "slope": jnp.asarray(3.0)}
    )

    transition_sample = transition.sample(
        jr.key(3), jnp.asarray([1.0]), 0.5, 1.0, context
    )
    transition_log_prob = transition.log_prob(
        jnp.asarray([2.0]), jnp.asarray([1.0]), 0.5, 1.0, context
    )
    location = observation.location(jnp.asarray([1.0]), 2.0, context)
    observation_log_prob = observation.log_prob(
        jnp.asarray([7.0]),
        jnp.asarray([1.0]),
        2.0,
        jnp.asarray([True]),
        context,
    )
    draws = observation.sample(
        jr.key(4),
        jnp.asarray([1.0]),
        2.0,
        context,
        sample_shape=(2,),
    )

    assert jnp.allclose(transition_sample.values, jnp.asarray([2.0]))
    assert jnp.allclose(transition_log_prob, 0.0)
    assert jnp.allclose(location, jnp.asarray([7.0]))
    assert jnp.allclose(observation_log_prob, 0.0)
    assert jnp.allclose(draws, jnp.asarray([[7.0], [7.0]]))


def test_state_space_keys_are_case_identity_and_prefix_stable():
    root = jr.key(4)
    first = phx.stochastic.state_space_key(
        root, "transition", "physical-case", 3, member=7
    )
    repeated = phx.stochastic.state_space_key(
        root, "transition", "physical-case", 3, member=7
    )
    other_step = phx.stochastic.state_space_key(
        root, "transition", "physical-case", 4, member=7
    )

    assert jnp.array_equal(first, repeated)
    assert not jnp.array_equal(first, other_step)


def test_state_space_problem_rejects_shape_mismatch():
    problem = _linear_problem()
    assert problem.model.state_shape == (1,)

    mismatched = phx.stochastic.ObservationSequence(
        jnp.asarray([1.0]),
        jnp.ones((1, 2)),
        observation_axes=("sensor",),
    )
    with pytest.raises(ValueError, match="observation shapes"):
        phx.stochastic.StateSpaceProblem(
            problem.model, mismatched, initial_time=0.0, problem_id="bad"
        )
