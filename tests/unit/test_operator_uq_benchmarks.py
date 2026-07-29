#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import jax.numpy as jnp
import polars as pl

from tools.operator_benchmarks import (
    calibration_case_checksum,
    green_function_scenario,
    OperatorUQBenchmarkProfile,
    periodic_burgers_scenario,
    run_operator_uq_benchmark,
    run_operator_uq_suite,
    save_operator_uq_artifacts,
    split_operator_scenario,
)


def _split(scenario):
    return split_operator_scenario(
        scenario,
        seed=17,
        train_fraction=0.5,
        validation_fraction=0.25,
    )


def test_operator_uq_suite_runs_grid_and_point_cloud_and_serializes(tmp_path):
    periodic = _split(
        periodic_burgers_scenario(
            train_resolution=6,
            test_resolution=8,
            num_cases=8,
            rollout_steps=2,
        )
    )
    point_cloud = _split(
        green_function_scenario(source_points=5, query_points=7, num_cases=8)
    )
    suite = run_operator_uq_suite(
        (
            OperatorUQBenchmarkProfile(periodic, "fno"),
            OperatorUQBenchmarkProfile(point_cloud, "deeponet"),
        ),
        seeds=(2, 3),
        steps=0,
        repeats=1,
        alpha=0.5,
        quick=True,
        fit_projection_laplace=False,
        commit_identity="unit-test",
    )

    assert suite.metadata.commit_identity == "unit-test"
    assert suite.metadata.scenario_checksums[0][0] == periodic.name
    assert suite.calibration_case_checksums == (
        (periodic.name, calibration_case_checksum(periodic)),
        (point_cloud.name, calibration_case_checksum(point_cloud)),
    )
    assert {result.architecture for result in suite.results} == {"fno", "deeponet"}
    assert all(result.ensemble_size == 2 for result in suite.results)
    assert all(
        jnp.isfinite(evaluation.crps)
        and jnp.isfinite(evaluation.energy_score)
        and 0.0 <= evaluation.pointwise_coverage <= 1.0
        for result in suite.results
        for evaluation in result.evaluations
    )
    assert any(
        evaluation.rollout_steps == 2 for evaluation in suite.results[0].evaluations
    )
    assert any(
        evaluation.nominal_coverage_compatible is not None
        for result in suite.results
        for evaluation in result.evaluations
    )

    json_path, parquet_path = save_operator_uq_artifacts(tmp_path, suite)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    table = pl.read_parquet(parquet_path)
    assert payload["metadata"]["commit_identity"] == "unit-test"
    assert set(table["architecture"]) == {"fno", "deeponet"}
    assert table.height == sum(len(result.evaluations) for result in suite.results)


def test_operator_uq_projection_laplace_preserves_operator_geometry():
    scenario = _split(
        periodic_burgers_scenario(
            train_resolution=6,
            test_resolution=8,
            num_cases=8,
        )
    )
    _, result = run_operator_uq_benchmark(
        scenario,
        architecture="fno",
        seeds=(5,),
        steps=0,
        repeats=1,
        alpha=0.5,
        quick=True,
        fit_projection_laplace=True,
        posterior_samples=3,
    )

    assert result.laplace is not None
    assert result.laplace.parameter_dimension > 0
    assert result.laplace.parameter_variance_mean > 0.0
    assert result.laplace.output_variance_mean >= 0.0
    assert result.laplace.posterior_sample_count == 3
    assert result.laplace.geometry_preserved
