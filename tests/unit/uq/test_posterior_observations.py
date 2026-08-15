#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _observation_problem(*, sample_observation=True):
    query_scale = 0.2
    space = phx.uq.ParameterSpace(jnp.asarray(0.9), priors=phx.uq.Normal(0.0, 2.0))

    def noise_scale(parameter):
        return 0.1 + 0.05 * parameter

    sampler = None
    if sample_observation:
        sampler = lambda key, parameter, query: cx.Field(
            parameter * query
            + noise_scale(parameter) * jr.normal(key, query.shape, dtype=query.dtype),
            dims=("x",),
        )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda parameter: -0.5 * ((parameter - 1.0) / query_scale) ** 2,
        predict=lambda parameter, query: cx.Field(parameter * query, dims=("x",)),
        observation_variance=lambda parameter, query: cx.Field(
            jnp.full(query.shape, noise_scale(parameter) ** 2),
            dims=("x",),
        ),
        sample_observation=sampler,
    )
    return problem


def _laplace(problem):
    return phx.uq.fit_laplace(
        problem,
        jnp.asarray(0.99),
        stationarity_tolerance=None,
    )


def test_laplace_prediction_propagates_conditional_variance_and_total_variance():
    problem = _observation_problem()
    result = _laplace(problem)
    query = jnp.linspace(0.5, 1.5, 5)
    prediction = result.predict(
        jr.key(0),
        query,
        num_samples=128,
        batch_size=19,
    )

    assert prediction.conditional_variance is not None
    assert prediction.conditional_variance.dims == ("__phydra_uq_draw", "x")
    assert prediction.conditional_variance.shape == prediction.samples.shape
    assert jnp.allclose(
        jnp.asarray(prediction.total_variance().data),
        prediction.epistemic_variance().data + prediction.observation_variance().data,
    )
    assert jnp.all(prediction.observation_variance().data > 0.0)


def test_laplace_observation_draws_are_reproducible_chunk_invariant_and_separated():
    problem = _observation_problem()
    result = _laplace(problem)
    query = jnp.linspace(0.5, 1.5, 4)
    kwargs = dict(num_samples=9, num_observation_samples=32)

    first = result.predict_observations(jr.key(1), query, batch_size=None, **kwargs)
    replay = result.predict_observations(jr.key(1), query, batch_size=4, **kwargs)
    changed = result.predict_observations(jr.key(2), query, batch_size=4, **kwargs)

    assert first.samples.dims == (
        "__phydra_uq_draw",
        "__phydra_uq_observation",
        "x",
    )
    assert first.samples.shape == (9, 32, 4)
    assert tuple(axis.source for axis in first.sample_axes) == (
        "epistemic",
        "observation",
    )
    assert jnp.array_equal(jnp.asarray(first.samples.data), replay.samples.data)
    assert not jnp.array_equal(jnp.asarray(first.samples.data), changed.samples.data)
    assert first.mean(sources="observation").dims == ("__phydra_uq_draw", "x")
    assert first.variance(sources="observation").dims == (
        "__phydra_uq_draw",
        "x",
    )


def test_observation_draws_recover_declared_gaussian_variance():
    problem = _observation_problem()
    result = _laplace(problem)
    query = jnp.asarray([0.5, 1.0])
    prediction = result.predict(jr.key(3), query, num_samples=6)
    observations = result.predict_observations(
        jr.key(3),
        query,
        num_samples=6,
        num_observation_samples=2048,
        batch_size=2,
    )

    empirical = observations.variance(sources="observation").data
    expected = prediction.conditional_variance.data
    assert jnp.allclose(empirical, expected, rtol=0.12, atol=3e-3)


def test_observation_prediction_requires_explicit_sampler_and_valid_draws():
    query = jnp.ones((2,))
    missing = _laplace(_observation_problem(sample_observation=False))
    with pytest.raises(ValueError, match="no observation-sampling function"):
        missing.predict_observations(
            jr.key(4),
            query,
            num_samples=2,
            num_observation_samples=3,
        )

    space = phx.uq.ParameterSpace(jnp.asarray(0.0), priors=phx.uq.Normal(0.0, 1.0))
    invalid_problem = phx.uq.PosteriorProblem(
        space,
        lambda parameter: -0.5 * parameter**2,
        predict=lambda parameter, query: cx.Field(parameter + query, dims=("x",)),
        sample_observation=lambda key, parameter, query: cx.Field(
            jnp.full(query.shape, jnp.nan),
            dims=("x",),
        ),
    )
    invalid = _laplace(invalid_problem)
    with pytest.raises(FloatingPointError, match="invalid sample indices"):
        invalid.predict_observations(
            jr.key(5),
            query,
            num_samples=2,
            num_observation_samples=3,
            valid_policy="raise",
        )


def test_mcmc_observation_prediction_preserves_chain_and_draw_axes():
    problem = _observation_problem()
    query = jnp.asarray([0.5, 1.0])
    result = phx.uq.sample_nuts(
        problem,
        key=jr.key(6),
        num_chains=2,
        num_warmup=12,
        num_samples=8,
        initial_step_size=0.1,
    )

    latent = result.predict(query, batch_size=3)
    observations = result.predict_observations(
        jr.key(7),
        query,
        num_observation_samples=4,
        batch_size=3,
    )
    assert isinstance(latent, phx.uq.PredictiveField)
    assert isinstance(observations, phx.uq.PredictiveField)
    assert latent.conditional_variance is not None
    assert latent.samples.shape == (2, 8, 2)
    assert observations.samples.shape == (2, 8, 4, 2)
    assert observations.samples.dims == (
        "__phydra_uq_chain",
        "__phydra_uq_draw",
        "__phydra_uq_observation",
        "x",
    )
