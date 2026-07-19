#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.uq._checkpoint import _json_value
from phydrax.uq._result_export import _adapt_result


def _problem():
    target = {
        "offset": jnp.asarray(0.5),
        "slope": jnp.asarray([-0.3, 0.8]),
    }
    space = phx.uq.ParameterSpace(
        {
            "offset": jnp.asarray(0.0),
            "slope": jnp.zeros(2),
        },
        priors={
            "offset": phx.uq.Normal(0.0, 1.0),
            "slope": phx.uq.Normal(0.0, 1.0),
        },
    )
    return phx.uq.PosteriorProblem(
        space,
        lambda value: (
            -0.5 * ((value["offset"] - target["offset"]) / 0.4) ** 2
            - 0.5 * jnp.sum(((value["slope"] - target["slope"]) / 0.5) ** 2)
        ),
        gauss_newton_residual=lambda value: jnp.concatenate(
            (
                ((value["offset"] - target["offset"]) / 0.4).reshape(1),
                (value["slope"] - target["slope"]) / 0.5,
            )
        ),
    )


def _assert_archive_matches_adapter(result, destination):
    expected_arrays = {}
    expected_fields = {}
    expected_trees = {}
    expected_kind, expected_metadata, expected_excluded = _adapt_result(
        result,
        expected_arrays,
        expected_fields,
        expected_trees,
    )
    phx.uq.export_result(result, destination)
    archive = phx.uq.read_result_archive(destination)

    assert archive.kind == expected_kind
    assert dict(archive.metadata) == _json_value(expected_metadata, path="metadata")
    assert dict(archive.fields) == expected_fields
    assert dict(archive.trees) == expected_trees
    assert archive.excluded == tuple(expected_excluded)
    assert set(archive.arrays) == set(expected_arrays)
    for name, expected in expected_arrays.items():
        assert np.array_equal(archive.arrays[name], expected, equal_nan=True)
        assert not archive.arrays[name].flags.writeable
    return archive


def test_portable_result_archives_cover_declared_uq_result_types(tmp_path):
    problem = _problem()
    mode = phx.uq.find_map(problem, gradient_tolerance=1e-8)
    dense_laplace = phx.uq.fit_laplace(problem, mode.position)
    structured_laplace = phx.uq.fit_laplace(
        problem,
        mode.position,
        curvature="diagonal",
    )
    pathfinder = phx.uq.fit_pathfinder(
        problem,
        key=jr.key(940),
        num_samples=16,
        num_elbo_samples=8,
        max_steps=15,
    )
    mcmc = phx.uq.sample_hmc(
        problem,
        key=jr.key(941),
        num_integration_steps=5,
        num_chains=2,
        num_warmup=25,
        num_samples=16,
        initial_step_size=0.15,
        chain_method="vectorized",
    )
    convergence = mcmc.convergence_report(
        max_rhat=2.0,
        min_bulk_ess=1.0,
        min_tail_ess=1.0,
        allow_divergences=True,
    )
    smc = phx.uq.sample_tempered_smc(
        problem,
        key=jr.key(942),
        num_particles=48,
        target_ess=0.75,
        num_mcmc_steps=1,
        step_size=0.1,
        num_integration_steps=4,
    )
    eki = phx.uq.fit_eki(
        problem,
        key=jr.key(944),
        ensemble_size=48,
        target_ess=0.75,
    )
    discrepancy = phx.uq.DiscrepancyIdentifiabilityReport(
        failures=("coverage",),
        num_repeats=6,
        baseline_parameter_bias=0.4,
        fixed_gp_parameter_bias=0.2,
        joint_gp_parameter_bias=0.1,
        nll_improvement=0.3,
        crps_improvement=0.25,
        mean_coverage=0.8,
        max_abs_parameter_gp_correlation=0.5,
    )

    results = (
        mode,
        dense_laplace,
        structured_laplace,
        pathfinder,
        mcmc,
        convergence,
        smc,
        eki,
        discrepancy,
    )
    kinds = []
    for index, result in enumerate(results):
        archive = _assert_archive_matches_adapter(
            result,
            tmp_path / f"result-{index}.phxuq",
        )
        kinds.append(archive.kind)

    assert kinds == [
        "map",
        "laplace",
        "structured_laplace",
        "pathfinder",
        "mcmc",
        "mcmc_convergence_report",
        "tempered_smc",
        "ensemble_kalman_inversion",
        "discrepancy_identifiability_report",
    ]


def test_arviz_adapter_preserves_chain_draw_parameter_and_sampler_dimensions():
    problem = _problem()
    result = phx.uq.sample_nuts(
        problem,
        key=jr.key(943),
        num_chains=2,
        num_warmup=25,
        num_samples=16,
        initial_step_size=0.15,
        max_num_doublings=5,
        chain_method="vectorized",
    )
    inference_data = phx.uq.to_arviz(result)
    posterior = inference_data["posterior"].dataset
    sample_stats = inference_data["sample_stats"].dataset

    assert posterior.sizes["chain"] == 2
    assert posterior.sizes["draw"] == 16
    recovered_paths = {
        phx.uq.decode_parameter_name(str(name)) for name in posterior.data_vars
    }
    assert recovered_paths == {"['offset']", "['slope']"}
    slope_name = phx.uq.encode_parameter_name("['slope']")
    assert posterior[slope_name].dims == (
        "chain",
        "draw",
        f"{slope_name}_dim_0",
    )
    assert set(sample_stats.data_vars) == {
        "lp",
        "acceptance_rate",
        "diverging",
        "energy",
        "n_steps",
        "tree_depth",
    }
    assert sample_stats["lp"].dims == ("chain", "draw")
    assert "observed_data" not in inference_data.children


def test_result_archive_reader_rejects_truncated_archives(tmp_path):
    result = phx.uq.find_map(_problem(), gradient_tolerance=1e-8)
    destination = phx.uq.export_result(result, tmp_path / "result.phxuq")
    destination.write_bytes(destination.read_bytes()[:48])

    with pytest.raises(phx.uq.CheckpointCorruptionError, match="Cannot read"):
        phx.uq.read_result_archive(destination)
