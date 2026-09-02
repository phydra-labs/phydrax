#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import pytest

import phydrax as phx


def _continuous_plan(
    *,
    initial_live=32,
    max_live=None,
    max_dead_points=400,
    max_likelihood_evaluations=8_000,
    maximum_attempts=8,
    rejection_fallback=False,
):
    live_capacity = initial_live if max_live is None else max_live
    return phx.uq.NestedSamplingPlan(
        phx.uq.NestedSamplingCapacity(
            max_live=live_capacity,
            max_dead_points=max_dead_points,
            max_likelihood_evaluations=max_likelihood_evaluations,
            max_dynamic_batches=2,
            max_clusters=2,
            max_phantoms=16,
        ),
        phx.uq.NestedPriorPlan(continuous_paths=("<root>",)),
        phx.uq.NestedProposalPlan(
            "hit-and-run",
            maximum_attempts=maximum_attempts,
            rejection_fallback=rejection_fallback,
        ),
        initial_live=initial_live,
    )


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
        plan=_continuous_plan(),
        remaining_evidence_tolerance=0.08,
    )
    return result, observation, prior_scale, observation_scale


def test_prepared_nested_sampling_recovers_gaussian_evidence_and_posterior(
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
    assert result.num_dead > result.num_live
    assert jnp.all(jnp.diff(result.log_likelihood) >= 0.0)
    assert jsp.special.logsumexp(result.posterior_log_weights) == pytest.approx(0.0)
    assert result.log_evidence == pytest.approx(evidence_truth, abs=0.5)
    assert recovered_mean == pytest.approx(posterior_mean, abs=0.25)
    assert recovered_variance == pytest.approx(posterior_variance, rel=0.6)
    assert result.final_state.adaptation.base_attempts > 0


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


def test_all_prepared_nested_kernels_dynamic_allocation_and_phantoms_execute():
    support = jnp.asarray([0.0, 1.0])
    masses = jnp.asarray([0.4, 0.6])
    space = phx.uq.ParameterSpace(
        {"angle": jnp.asarray(0.0), "finite": jnp.asarray(0.0), "x": jnp.asarray(0.0)},
        priors={
            "angle": phx.uq.Uniform(0.0, 2.0 * jnp.pi),
            "finite": phx.uq.EmpiricalDistribution(support, masses),
            "x": phx.uq.Normal(0.0, 1.0),
        },
    )

    def log_likelihood(value):
        angular = jnp.mod(value["angle"] - 0.2 + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        return -0.1 * (value["x"] - 0.4) ** 2 - 0.05 * angular**2 + 0.1 * value["finite"]

    problem = phx.uq.PosteriorProblem(space, log_likelihood)
    capacity = phx.uq.NestedSamplingCapacity(
        max_live=10,
        max_dead_points=24,
        max_likelihood_evaluations=2_000,
        max_dynamic_batches=2,
        max_clusters=2,
        max_phantoms=16,
    )
    plan = phx.uq.NestedSamplingPlan(
        capacity,
        phx.uq.NestedPriorPlan(
            continuous_paths=("['angle']", "['x']"),
            finite_supports={"['finite']": (support, masses)},
            periodic=(phx.uq.PeriodicNestedCoordinate("['angle']", 0.0, 2.0 * jnp.pi),),
        ),
        phx.uq.NestedProposalPlan(
            "slice-within-gibbs",
            ellipsoid=True,
            discrete_gibbs=True,
            periodic_slice=True,
            phantom_recycling=True,
            learned_flow=True,
            gradient_guided=True,
            maximum_attempts=8,
        ),
        initial_live=8,
        dynamic=phx.uq.DynamicNestedPolicy(
            pilot_dead_points=1,
            additional_live_per_batch=1,
            allocation_cadence=1,
        ),
    )
    result = phx.uq.sample_nested(
        problem,
        key=jr.key(93),
        plan=plan,
        remaining_evidence_tolerance=0.35,
    )
    evidence = result.final_state.adaptation

    assert result.num_live <= capacity.max_live
    assert result.num_dead <= capacity.max_dead_points
    assert int(result.status) != phx.uq.NESTED_SAMPLING_INNER_KERNEL_FAILURE
    assert result.valid
    assert result.num_likelihood_evaluations <= capacity.max_likelihood_evaluations
    assert evidence.base_attempts > 0
    assert evidence.ellipsoid_attempts > 0
    assert evidence.flow_attempts > 0
    assert evidence.gradient_attempts > 0
    assert evidence.discrete_updates > 0
    assert evidence.periodic_updates > 0
    assert evidence.dynamic_additions > 0
    assert evidence.phantom_creations > 0
    assert evidence.phantom_revalidations > 0
    assert jnp.all(
        result.final_state.phantom.birth_log_likelihood[result.final_state.phantom.mask]
        < jnp.inf
    )
    assert result.num_samples == result.num_dead + result.num_live


def test_nested_checkpoint_resume_restores_exact_full_prepared_state(tmp_path):
    prior_calls = []

    def prior_position_sampler(key, count):
        prior_calls.append(count)
        return jr.normal(key, (count,))

    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(jnp.asarray(0.0), priors=phx.uq.Normal(0.0, 1.0)),
        lambda value: -0.5 * ((value - 0.5) / 0.4) ** 2,
    )
    checkpoint = tmp_path / "nested.phxckpt"
    plan = _continuous_plan(
        initial_live=16,
        max_dead_points=120,
        max_likelihood_evaluations=3_000,
        maximum_attempts=6,
    )
    kwargs = {
        "key": jr.key(22),
        "plan": plan,
        "remaining_evidence_tolerance": 0.2,
        "prior_position_sampler": prior_position_sampler,
        "checkpoint_id": "nested-test",
    }
    first = phx.uq.sample_nested(
        problem,
        checkpoint_path=checkpoint,
        checkpoint_every=5,
        **kwargs,
    )
    resumed = phx.uq.sample_nested(
        problem,
        resume_from=checkpoint,
        **kwargs,
    )
    assert prior_calls == [plan.initial_live]

    equality = jax.tree.map(jnp.array_equal, first.final_state, resumed.final_state)
    assert all(bool(value) for value in jax.tree_util.tree_leaves(equality))
    assert jnp.array_equal(first.log_likelihood, resumed.log_likelihood)
    assert jnp.array_equal(first.posterior_log_weights, resumed.posterior_log_weights)
    assert jnp.array_equal(first.sample_ids, resumed.sample_ids)
    assert first.num_likelihood_evaluations == resumed.num_likelihood_evaluations

    incompatible = _continuous_plan(
        initial_live=16,
        max_dead_points=121,
        max_likelihood_evaluations=3_000,
        maximum_attempts=6,
    )
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        phx.uq.sample_nested(
            problem,
            key=jr.key(22),
            plan=incompatible,
            resume_from=checkpoint,
            checkpoint_id="nested-test",
            prior_position_sampler=prior_position_sampler,
            remaining_evidence_tolerance=0.2,
        )


def test_exact_rejection_is_only_an_explicit_failed_geometry_fallback():
    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(jnp.zeros(2), priors=phx.uq.Normal(0.0, 1.0)),
        lambda value: -0.5 * jnp.sum((value - jnp.asarray([0.5, -0.5])) ** 2),
    )

    def plan(fallback):
        return phx.uq.NestedSamplingPlan(
            phx.uq.NestedSamplingCapacity(
                max_live=2,
                max_dead_points=4,
                max_likelihood_evaluations=256,
                max_dynamic_batches=1,
                max_clusters=1,
                max_phantoms=2,
            ),
            phx.uq.NestedPriorPlan(continuous_paths=("<root>",)),
            phx.uq.NestedProposalPlan(
                ellipsoid=True,
                maximum_attempts=8,
                rejection_fallback=fallback,
            ),
            initial_live=2,
        )

    failed = phx.uq.sample_nested(
        problem,
        key=jr.key(31),
        plan=plan(False),
        remaining_evidence_tolerance=0.9,
    )
    recovered = phx.uq.sample_nested(
        problem,
        key=jr.key(31),
        plan=plan(True),
        remaining_evidence_tolerance=0.9,
    )

    assert jnp.isinf(failed.final_state.proposal.ellipsoid_condition)
    assert failed.final_state.adaptation.proposal_failures > 0
    assert int(failed.status) == phx.uq.NESTED_SAMPLING_INNER_KERNEL_FAILURE
    assert int(recovered.status) != phx.uq.NESTED_SAMPLING_INNER_KERNEL_FAILURE
    assert recovered.final_state.adaptation.fallback_draws > 0


def test_nested_sampling_rejects_incomplete_prior_topology():
    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            {"x": jnp.asarray(0.0), "y": jnp.asarray(0.0)},
            priors={"x": phx.uq.Normal(0.0, 1.0), "y": phx.uq.Normal(0.0, 1.0)},
        ),
        lambda value: -0.5 * (value["x"] ** 2 + value["y"] ** 2),
    )
    plan = phx.uq.NestedSamplingPlan(
        phx.uq.NestedSamplingCapacity(
            max_live=4,
            max_dead_points=8,
            max_likelihood_evaluations=64,
            max_dynamic_batches=1,
            max_clusters=1,
            max_phantoms=2,
        ),
        phx.uq.NestedPriorPlan(continuous_paths=("['x']",)),
        phx.uq.NestedProposalPlan(maximum_attempts=4),
        initial_live=4,
    )
    with pytest.raises(ValueError, match="classify every parameter leaf exactly"):
        phx.uq.sample_nested(problem, key=jr.key(32), plan=plan)


def test_nested_sampling_rejects_nondeterministic_likelihood():
    calls = {"count": 0}

    def changing_likelihood(_value):
        calls["count"] += 1
        return jnp.asarray(calls["count"], dtype=float)

    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(jnp.asarray(0.0), priors=phx.uq.Normal(0.0, 1.0)),
        changing_likelihood,
    )
    with pytest.raises(ValueError, match="deterministic prior and likelihood"):
        phx.uq.sample_nested(
            problem,
            key=jr.key(33),
            plan=_continuous_plan(
                initial_live=4,
                max_dead_points=8,
                max_likelihood_evaluations=64,
                maximum_attempts=4,
            ),
        )


def test_nested_sampling_returns_explicit_status_when_every_live_point_is_zero_mass():
    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(jnp.asarray(0.0), priors=phx.uq.Normal(0.0, 1.0)),
        lambda _value: -jnp.inf,
    )
    result = phx.uq.sample_nested(
        problem,
        key=jr.key(21),
        plan=_continuous_plan(
            initial_live=8,
            max_dead_points=16,
            max_likelihood_evaluations=64,
            maximum_attempts=4,
        ),
    )

    assert int(result.status) == phx.uq.NESTED_SAMPLING_NO_FINITE_LIVE_POINT
    assert not result.valid
    assert not result.converged


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
