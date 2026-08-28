#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import pytest

import phydrax as phx


@pytest.fixture(scope="module")
def gaussian_nested_result():
    observation = 1.2
    prior_scale = 2.0
    observation_scale = 0.5
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        priors=phx.uq.Normal(0.0, prior_scale),
    )

    def log_likelihood(value):
        standardized = (value - observation) / observation_scale
        return (
            -0.5 * standardized**2
            - jnp.log(observation_scale)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )

    problem = phx.uq.PosteriorProblem(
        space,
        log_likelihood,
        predict=lambda value, coordinates: cx.Field(
            value * coordinates,
            dims=("coordinate",),
        ),
    )
    result = phx.uq.sample_nested(
        problem,
        key=jr.key(17),
        num_live=60,
        num_inner_steps=6,
        num_delete=5,
        remaining_evidence_tolerance=0.08,
        max_dead_points=1_200,
        num_volume_replicates=64,
    )
    return result, observation, prior_scale, observation_scale


def test_static_nested_sampling_recovers_gaussian_evidence_and_posterior(
    gaussian_nested_result,
):
    result, observation, prior_scale, observation_scale = gaussian_nested_result
    evidence_truth = jsp.stats.norm.logpdf(
        observation,
        loc=0.0,
        scale=jnp.sqrt(prior_scale**2 + observation_scale**2),
    )
    posterior_variance = 1.0 / (1.0 / prior_scale**2 + 1.0 / observation_scale**2)
    posterior_mean = posterior_variance * observation / observation_scale**2
    weights = jnp.exp(result.posterior_log_weights)
    recovered_mean = jnp.sum(weights * result.samples)
    recovered_variance = jnp.sum(weights * (result.samples - recovered_mean) ** 2)

    assert result.converged
    assert result.valid
    assert result.diagnostics.passed
    assert result.num_dead > result.num_live
    assert jnp.all(jnp.diff(result.log_likelihood) >= 0.0)
    assert jsp.special.logsumexp(result.posterior_log_weights) == pytest.approx(0.0)
    assert result.log_evidence == pytest.approx(evidence_truth, abs=0.35)
    assert recovered_mean == pytest.approx(posterior_mean, abs=0.15)
    assert recovered_variance == pytest.approx(posterior_variance, rel=0.35)
    assert result.posterior_effective_sample_size > 20


def test_nested_result_preserves_weighted_measure_and_resampling(
    gaussian_nested_result,
):
    result = gaussian_nested_result[0]
    measure = result.posterior_measure()
    draws = result.resample_posterior(jr.key(18), num_samples=37)
    prediction = result.predict(jnp.asarray([1.0, 2.0]))

    assert measure.normalized
    assert not measure.independent
    assert measure.ancestry.shape == result.posterior_log_weights.shape
    assert measure.stratum_ids.shape == result.posterior_log_weights.shape
    assert draws.shape == (37,)
    assert jnp.all(jnp.isfinite(draws))
    assert prediction.samples.data.shape == (result.num_samples, 2)


def test_nested_sampling_requires_prior_sampler_for_custom_prior():
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        log_prior=lambda value: -0.5 * value**2,
    )
    problem = phx.uq.PosteriorProblem(space, lambda value: -0.5 * value**2)

    with pytest.raises(ValueError, match="requires prior_position_sampler"):
        phx.uq.sample_nested(problem, key=jr.key(19), num_live=10)


def test_nested_sampling_rejects_counting_measure_prior():
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        priors=phx.uq.EmpiricalDistribution(jnp.asarray([0.0, 1.0])),
    )
    problem = phx.uq.PosteriorProblem(space, lambda value: -0.5 * value**2)

    with pytest.raises(TypeError, match="Lebesgue-density priors"):
        phx.uq.sample_nested(problem, key=jr.key(20), num_live=10)


def test_nested_sampling_returns_explicit_status_when_every_live_point_is_zero_mass():
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        priors=phx.uq.Normal(0.0, 1.0),
    )
    problem = phx.uq.PosteriorProblem(space, lambda _value: -jnp.inf)
    result = phx.uq.sample_nested(
        problem,
        key=jr.key(21),
        num_live=10,
        method="slice-within-gibbs",
        num_inner_steps=1,
        max_dead_points=20,
        num_volume_replicates=8,
    )

    assert int(result.status) == phx.uq.NESTED_SAMPLING_NO_FINITE_LIVE_POINT
    assert not result.valid
    assert not result.converged


def test_nested_checkpoint_replays_completed_run(tmp_path):
    prior_calls = []

    def prior_position_sampler(key, count):
        prior_calls.append(count)
        return jr.normal(key, (count,))

    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        log_prior=lambda value: -0.5 * value**2 - 0.5 * jnp.log(2.0 * jnp.pi),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: -0.5 * ((value - 0.5) / 0.4) ** 2,
    )
    checkpoint = tmp_path / "nested.phxckpt"
    kwargs = {
        "key": jr.key(22),
        "num_live": 24,
        "num_inner_steps": 4,
        "num_delete": 3,
        "remaining_evidence_tolerance": 0.2,
        "max_dead_points": 400,
        "num_volume_replicates": 16,
        "checkpoint_id": "nested-test",
        "prior_position_sampler": prior_position_sampler,
    }
    first = phx.uq.sample_nested(
        problem,
        checkpoint_path=checkpoint,
        checkpoint_every=5,
        **kwargs,
    )
    assert prior_calls == [24]
    resumed = phx.uq.sample_nested(
        problem,
        resume_from=checkpoint,
        **kwargs,
    )
    assert prior_calls == [24]

    assert jnp.array_equal(first.log_likelihood, resumed.log_likelihood)
    assert jnp.array_equal(
        first.log_evidence_replicates,
        resumed.log_evidence_replicates,
    )
    assert jnp.array_equal(first.sample_ids, resumed.sample_ids)
    assert first.num_likelihood_evaluations == resumed.num_likelihood_evaluations


def test_nested_result_exports_portable_weighted_record(
    gaussian_nested_result,
    tmp_path,
):
    result = gaussian_nested_result[0]
    destination = tmp_path / "nested.phxuq"
    phx.uq.export_result(result, destination)
    archive = phx.uq.read_result_archive(destination)

    assert archive.kind == "nested_sampling"
    assert archive.metadata["status_name"] == "success"
    assert archive.metadata["num_samples"] == result.num_samples
    assert archive.array("log_evidence") == result.log_evidence
    assert set(archive.tree("samples")) == {"<root>"}
