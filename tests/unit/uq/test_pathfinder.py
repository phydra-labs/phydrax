#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _gaussian_problem():
    mean = jnp.array([0.4, -0.8])
    covariance = jnp.array([[0.5, 0.15], [0.15, 0.8]])
    precision = jnp.linalg.inv(covariance)
    space = phx.uq.ParameterSpace(
        jnp.array([2.0, 2.0]),
        log_prior=lambda _: jnp.zeros(()),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: -0.5 * (value - mean) @ precision @ (value - mean),
    )
    return problem, mean, covariance


def test_pathfinder_recovers_correlated_gaussian_and_reports_density_ratios():
    problem, mean, covariance = _gaussian_problem()
    result = phx.uq.fit_pathfinder(
        problem,
        key=jr.key(0),
        num_samples=5_000,
        num_elbo_samples=100,
        max_steps=50,
    )

    assert result.num_samples == 5_000
    assert result.optimization_steps > 0
    assert result.duration_seconds > 0.0
    assert result.sample_memory_bytes == 2 * result.samples.nbytes
    assert jnp.isfinite(result.elbo)
    assert result.log_density.shape == (5_000,)
    assert result.importance_log_weights.shape == (5_000,)
    assert jnp.allclose(jnp.mean(result.samples, axis=0), mean, atol=0.04)
    assert jnp.allclose(
        jnp.cov(result.samples, rowvar=False),
        covariance,
        atol=0.055,
    )


def test_pathfinder_is_key_deterministic_and_supports_fresh_constrained_draws():
    problem, _, _ = _gaussian_problem()
    settings: dict[str, Any] = dict(
        key=jr.key(8),
        num_samples=32,
        num_elbo_samples=20,
        max_steps=30,
    )
    first = phx.uq.fit_pathfinder(problem, **settings)
    second = phx.uq.fit_pathfinder(problem, **settings)
    fresh = first.sample_approximation(jr.key(9), num_samples=7)

    assert jnp.array_equal(first.samples, second.samples)
    assert jnp.array_equal(first.log_density, second.log_density)
    assert fresh.shape == (7, 2)
    assert jnp.all(jnp.isfinite(fresh))


def test_pathfinder_rejects_invalid_configuration_and_nonfinite_initial_density():
    problem, _, _ = _gaussian_problem()
    with pytest.raises(ValueError, match="num_samples"):
        phx.uq.fit_pathfinder(problem, key=jr.key(0), num_samples=0)
    with pytest.raises(ValueError, match="history_size"):
        phx.uq.fit_pathfinder(problem, key=jr.key(0), history_size=0)

    invalid = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(jnp.asarray(0.0), log_prior=lambda _: jnp.zeros(())),
        lambda _: jnp.asarray(jnp.nan),
    )
    with pytest.raises(FloatingPointError, match="Initial Pathfinder"):
        phx.uq.fit_pathfinder(invalid, key=jr.key(1))


def test_pathfinder_observation_prediction_preserves_draw_and_observation_axes():
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        priors=phx.uq.Normal(0.0, 1.0),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda parameter: -0.5 * (parameter - 0.5) ** 2,
        sample_observation=lambda key, parameter, query: cx.Field(
            parameter * query + 0.1 * jr.normal(key, query.shape),
            dims=("x",),
        ),
    )
    result = phx.uq.fit_pathfinder(
        problem,
        key=jr.key(12),
        num_samples=10,
        num_elbo_samples=20,
    )

    observations = result.predict_observations(
        jr.key(13),
        jnp.linspace(0.0, 1.0, 4),
        num_observation_samples=3,
        observation_dim="measurement",
        batch_size=3,
    )
    assert isinstance(observations, phx.uq.PredictiveField)

    assert observations.samples.dims == (
        "__phydra_uq_draw",
        "measurement",
        "x",
    )
    assert observations.samples.shape == (10, 3, 4)
    assert tuple(axis.source for axis in observations.sample_axes) == (
        "epistemic",
        "observation",
    )
