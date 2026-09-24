import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.models import MLP


def _problem(*, mask=None):
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[1.0], [2.0]]),
        observation_mask=mask,
        case_ids=("only",),
        sequence_id="ensemble-sequence",
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
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="linear"
    )
    return phx.stochastic.StateSpaceProblem(
        model, observations, initial_time=0.0, problem_id="ensemble-problem"
    )


def test_etkf_matches_linear_gaussian_mean_and_variance():
    problem = _problem()
    exact = phx.uq.kalman_filter(problem)
    ensemble = phx.uq.ensemble_transform_kalman_filter(
        jr.key(20), problem, ensemble_size=512
    )
    means = jnp.mean(ensemble.analysis_ensembles, axis=-2)
    variances = jnp.var(ensemble.analysis_ensembles, axis=-2, ddof=1)

    assert jnp.allclose(means, exact.filtered_means, atol=0.08)
    assert jnp.allclose(
        variances[..., 0], exact.filtered_covariances[..., 0, 0], atol=0.04
    )
    assert jnp.all(ensemble.status == phx.uq.ENSEMBLE_FILTER_SUCCESS)


def test_streaming_and_batch_etkf_are_identical():
    problem = _problem()
    batch = phx.uq.ensemble_transform_kalman_filter(jr.key(21), problem, ensemble_size=32)
    state = phx.uq.initialize_ensemble_filter(jr.key(21), problem, ensemble_size=32)
    records = []
    for _ in range(problem.observations.num_steps):
        state, record = phx.uq.ensemble_filter_step(problem, state)
        records.append(record)

    assert jnp.array_equal(state.ensemble, batch.final_state.ensemble)
    assert jnp.array_equal(
        jnp.stack([record.analysis_ensemble for record in records]),
        batch.analysis_ensembles,
    )


def test_missing_observation_is_forecast_only_and_smoother_is_terminally_exact():
    problem = _problem(mask=jnp.asarray([[True], [False]]))
    filtered = phx.uq.ensemble_transform_kalman_filter(
        jr.key(22), problem, ensemble_size=64
    )
    smoothed = phx.uq.ensemble_kalman_smoother(filtered)
    predictive = phx.uq.ensemble_filter_predictive(smoothed)

    assert jnp.allclose(filtered.analysis_ensembles[1], filtered.forecast_ensembles[1])
    assert filtered.incremental_log_likelihood[1] == 0.0
    assert jnp.allclose(smoothed.ensembles[-1], filtered.analysis_ensembles[-1])
    assert predictive.samples.data.shape == (2, 64, 1)


def test_nonlinear_gaussian_observation_and_diagnostics():
    base = _problem()
    observation = phx.stochastic.GaussianObservationModel(
        lambda state, time, context: state**2,
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        base.model.prior,
        base.model.transition,
        observation,
        model_id="nonlinear",
    )
    problem = phx.stochastic.StateSpaceProblem(
        model,
        base.observations,
        initial_time=0.0,
        problem_id="nonlinear-problem",
    )
    result = phx.uq.ensemble_transform_kalman_filter(
        jr.key(23), problem, ensemble_size=64
    )
    diagnostics = phx.uq.ensemble_filter_diagnostics(result)

    assert diagnostics.passed
    assert jnp.all(diagnostics.effective_rank <= 1)
    assert jnp.all(diagnostics.ensemble_spread >= 0.0)


def _with_observation(base, location, /):
    observation = phx.stochastic.GaussianObservationModel(
        location,
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        base.model.prior, base.model.transition, observation, model_id="learned"
    )
    return phx.stochastic.StateSpaceProblem(
        model, base.observations, initial_time=0.0, problem_id="learned-problem"
    )


def test_learned_observation_location_runs_unchanged_etkf_numerics():
    base = _problem()
    network = MLP(in_size=1, out_size=1, width_size=4, depth=1, key=jr.key(3))
    location = phx.stochastic.ModelObservationLocation(
        network, state_shape=(1,), observation_shape=(1,)
    )
    learned = _with_observation(base, location)
    reference = _with_observation(base, lambda state, time, context: network(state))

    result = phx.uq.ensemble_transform_kalman_filter(jr.key(5), learned, ensemble_size=16)
    expected = phx.uq.ensemble_transform_kalman_filter(
        jr.key(5), reference, ensemble_size=16
    )

    assert jnp.array_equal(result.analysis_ensembles, expected.analysis_ensembles)
    assert jnp.all(result.status == phx.uq.ENSEMBLE_FILTER_SUCCESS)
    contract = location.component_contract()
    assert contract.authority is phx.ComponentAuthority.MODEL

    observation = learned.model.observation
    parameters, model_state, fixed = phx.partition_parameters(observation)
    assert {id(leaf) for leaf in jax.tree.leaves(parameters)} == {
        id(leaf) for leaf in jax.tree.leaves(eqx.filter(network, eqx.is_array))
    }
    context = phx.stochastic.StateSpaceStepContext.empty()

    def log_likelihood(parameters):
        bound = phx.combine_parameters(parameters, model_state, fixed)
        return bound.log_prob(
            jnp.asarray([1.0]), jnp.asarray([0.3]), 0.5, jnp.asarray([True]), context
        )

    gradient = jax.grad(log_likelihood)(parameters)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
    assert any(jnp.any(leaf != 0.0) for leaf in jax.tree.leaves(gradient))


def test_model_observation_location_requires_exact_pointwise_sizes():
    network = MLP(in_size=2, out_size=1, width_size=4, depth=1, key=jr.key(4))
    location = phx.stochastic.ModelObservationLocation(
        network, state_shape=(1,), observation_shape=(1,), time_input=True
    )
    states = jnp.asarray([[0.1], [0.4], [-0.2]])

    batched = location(states, 0.5, None)
    single = jnp.stack([location(state, 0.5, None) for state in states])
    assert batched.shape == (3, 1)
    assert jnp.allclose(batched, single, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="in_size must be 1"):
        phx.stochastic.ModelObservationLocation(
            network, state_shape=(1,), observation_shape=(1,)
        )


def test_high_dimensional_path_uses_ensemble_rank_not_state_covariance():
    state_size = 128
    ensemble_size = 12
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((state_size,)),
        jnp.eye(state_size),
        state_shape=(state_size,),
        prior_id="high-dimensional-prior",
    )
    transition = phx.stochastic.CallableTransitionKernel(
        lambda key, state, t0, t1, context: state + 0.05 * jr.normal(key, state.shape),
        state_shape=(state_size,),
        process_id="high-dimensional-process",
        approximation_id="explicit",
    )
    matrix = jnp.zeros((4, state_size)).at[jnp.arange(4), jnp.arange(4)].set(1.0)
    observation = phx.stochastic.LinearGaussianObservationModel(
        matrix,
        0.1 * jnp.eye(4),
        state_shape=(state_size,),
        observation_shape=(4,),
    )
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([1.0]),
        jnp.ones((1, 4)),
        case_ids=("field",),
        sequence_id="high-dimensional",
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="high-dimensional"
    )
    problem = phx.stochastic.StateSpaceProblem(
        model, observations, initial_time=0.0, problem_id="high-dimensional"
    )
    result = phx.uq.ensemble_transform_kalman_filter(
        jr.key(24), problem, ensemble_size=ensemble_size
    )
    diagnostics = phx.uq.ensemble_filter_diagnostics(result)

    assert result.analysis_ensembles.shape == (1, ensemble_size, state_size)
    assert diagnostics.effective_rank[0] <= ensemble_size - 1
    assert not hasattr(result, "filtered_covariances")
