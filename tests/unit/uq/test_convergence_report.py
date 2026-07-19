#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.uq._diagnostics import mcmc_diagnostics


def _result(
    *,
    rhat=(1.0, 1.0),
    bulk_ess=(800.0, 700.0),
    tail_ess=(750.0, 650.0),
    divergent=None,
    trajectory_expansions=None,
):
    chains, draws = 2, 12
    samples = {"layer": {"weight": jnp.zeros((chains, draws, 2))}}
    space = phx.uq.ParameterSpace(
        {"layer": {"weight": jnp.zeros((2,))}},
        priors={"layer": {"weight": phx.uq.Normal(0.0, 1.0)}},
    )
    problem = phx.uq.PosteriorProblem(space, lambda _: jnp.zeros(()))
    divergent_array = (
        jnp.zeros((chains, draws), dtype=bool)
        if divergent is None
        else jnp.asarray(divergent, dtype=bool)
    )
    expansions = (
        jnp.zeros((chains, draws), dtype=jnp.int32)
        if trajectory_expansions is None
        else jnp.asarray(trajectory_expansions, dtype=jnp.int32)
    )
    diagnostics = phx.uq.MCMCDiagnostics(
        rhat={"layer": {"weight": jnp.asarray(rhat)}},
        bulk_ess={"layer": {"weight": jnp.asarray(bulk_ess)}},
        tail_ess={"layer": {"weight": jnp.asarray(tail_ess)}},
        acceptance_rate=jnp.full((chains, draws), 0.82),
        divergent=divergent_array,
    )
    return phx.uq.MCMCResult(
        problem=problem,
        samples=samples,
        unconstrained_samples=samples,
        log_density=jnp.zeros((chains, draws)),
        acceptance_rate=jnp.full((chains, draws), 0.82),
        divergent=divergent_array,
        energy=jnp.ones((chains, draws)),
        num_integration_steps=jnp.full((chains, draws), 7),
        num_trajectory_expansions=expansions,
        final_states=(),
        warmup=(),
        diagnostics=diagnostics,
        root_key=jr.key(0),
        chain_keys=jr.split(jr.key(0), chains),
        algorithm="nuts",
        duration_seconds=1.25,
        max_num_doublings=4,
        chain_method="sequential",
        adaptation_duration_seconds=0.75,
        sampling_duration_seconds=0.25,
    )


def test_convergence_report_passes_and_serializes_complete_summary():
    result = _result()
    report = result.convergence_report()

    assert report.passed
    assert report.failures == ()
    assert report.num_chains == 2
    assert report.num_draws == 12
    assert report.sample_memory_bytes == result.sample_memory_bytes
    assert report.max_integration_steps == 7
    assert report.mean_acceptance_rate == pytest.approx(0.82)
    assert json.loads(json.dumps(report.as_dict()))["passed"] is True
    report.raise_for_failure()


def test_convergence_report_identifies_nested_rhat_and_ess_failures():
    result = _result(
        rhat=(1.0, 1.08),
        bulk_ess=(800.0, 30.0),
        tail_ess=(20.0, 650.0),
    )
    report = result.convergence_report(
        max_rhat=1.01,
        min_bulk_ess=400,
        min_tail_ess=400,
    )

    assert not report.passed
    assert report.failures == ("rhat", "bulk_ess", "tail_ess")
    assert report.rhat_failures == ("['layer']['weight'][1]",)
    assert report.bulk_ess_failures == ("['layer']['weight'][1]",)
    assert report.tail_ess_failures == ("['layer']['weight'][0]",)
    with pytest.raises(phx.uq.MCMCConvergenceError) as caught:
        report.raise_for_failure()
    assert caught.value.report is report


def test_convergence_report_gates_divergences_and_tree_saturation_independently():
    divergent = jnp.zeros((2, 12), dtype=bool).at[1, 3].set(True)
    expansions = jnp.zeros((2, 12), dtype=jnp.int32).at[0, 5].set(4)
    result = _result(divergent=divergent, trajectory_expansions=expansions)

    strict = result.convergence_report()
    permissive = result.convergence_report(
        allow_divergences=True,
        allow_trajectory_saturation=True,
    )
    assert strict.failures == ("divergences", "trajectory_saturation")
    assert strict.divergence_indices == ((1, 3),)
    assert strict.trajectory_saturation_count == 1
    assert permissive.passed
    assert permissive.divergence_count == 1


def test_convergence_report_rejects_invalid_thresholds_and_nonfinite_metrics():
    with pytest.raises(ValueError, match="max_rhat"):
        phx.uq.MCMCConvergenceThresholds(max_rhat=0.99)
    with pytest.raises(ValueError, match="min_bulk_ess"):
        phx.uq.MCMCConvergenceThresholds(min_bulk_ess=0.0)

    result = _result(rhat=(1.0, jnp.nan))
    report = result.convergence_report()
    assert not report.passed
    assert report.rhat_failures == ("['layer']['weight'][1]",)


def test_reports_detect_shifted_chains_and_autocorrelated_draws():
    chains, draws = 4, 256
    noise = jr.normal(jr.key(10), (chains, draws))
    shifted = noise + 2.0 * jnp.arange(chains)[:, None]
    innovations = jr.normal(jr.key(11), (chains, draws))

    def ar1(values):
        _, draws_ = jax.lax.scan(
            lambda previous, innovation: (
                0.995 * previous + innovation,
                0.995 * previous + innovation,
            ),
            jnp.zeros(()),
            values,
        )
        return draws_

    autocorrelated = jax.vmap(ar1)(innovations)

    def report(samples, thresholds):
        diagnostics = mcmc_diagnostics(
            {"coefficient": samples},
            acceptance_rate=jnp.full((chains, draws), 0.8),
            divergent=jnp.zeros((chains, draws), dtype=bool),
        )
        return phx.uq.MCMCConvergenceReport(
            diagnostics=diagnostics,
            thresholds=thresholds,
            divergent=jnp.zeros((chains, draws), dtype=bool),
            num_integration_steps=jnp.ones((chains, draws), dtype=jnp.int32),
            num_trajectory_expansions=jnp.zeros((chains, draws), dtype=jnp.int32),
            max_num_doublings=10,
            num_chains=chains,
            num_draws=draws,
            sample_memory_bytes=int(samples.nbytes),
            duration_seconds=0.0,
            adaptation_duration_seconds=0.0,
            sampling_duration_seconds=0.0,
            samples_per_second=0.0,
        )

    shifted_report = report(
        shifted,
        phx.uq.MCMCConvergenceThresholds(
            max_rhat=1.01,
            min_bulk_ess=1.0,
            min_tail_ess=1.0,
        ),
    )
    autocorrelated_report = report(
        autocorrelated,
        phx.uq.MCMCConvergenceThresholds(
            max_rhat=100.0,
            min_bulk_ess=100.0,
            min_tail_ess=1.0,
        ),
    )
    assert shifted_report.rhat_failures == ("['coefficient']",)
    assert autocorrelated_report.bulk_ess_failures == ("['coefficient']",)
