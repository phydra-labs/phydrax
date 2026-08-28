#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import os

import pytest

from tools.uq_benchmarks.gp_scaling import run_gp_scaling_benchmark
from tools.uq_benchmarks.nuts_scheduling import run_nuts_scheduling_benchmark
from tools.uq_benchmarks.runner import run_benchmark_matrix
from tools.uq_benchmarks.scenarios import SCENARIOS


@pytest.mark.skipif(
    os.environ.get("PHYDRAX_RUN_UQ_BENCHMARKS") != "1",
    reason="set PHYDRAX_RUN_UQ_BENCHMARKS=1 to run the complete UQ release matrix",
)
def test_complete_uq_benchmark_matrix_passes_and_writes_machine_report(tmp_path):
    report = run_benchmark_matrix(profile="smoke")
    destination = report.write_json(tmp_path / "uq-benchmark-smoke.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert report.passed
    assert tuple(scenario.name for scenario in report.scenarios) == tuple(SCENARIOS)
    assert "schema_version" not in payload
    assert payload["summary"]["scenario_count"] == len(SCENARIOS)
    assert payload["summary"]["scenarios_failed"] == 0
    assert all(
        category["failed"] == 0
        for category in payload["summary"]["metric_categories"].values()
    )


@pytest.mark.skipif(
    os.environ.get("PHYDRAX_RUN_UQ_BENCHMARKS") != "1",
    reason="set PHYDRAX_RUN_UQ_BENCHMARKS=1 to run the UQ GP scaling gates",
)
def test_gp_scaling_benchmark_passes_accuracy_and_reuse_speedup_gates(tmp_path):
    report = run_gp_scaling_benchmark(profile="smoke")
    destination = report.write_json(tmp_path / "uq-gp-scaling-smoke.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert report.passed
    assert payload["suite"] == "phydrax-uq-gp-scaling"
    assert payload["summary"]["scenario_count"] == 2
    assert payload["summary"]["scenarios_failed"] == 0
    assert payload["configuration"]["regression_gates"] == {
        "fitc_fixed_factor_gate_minimum_observations": 256,
        "maximum_cagp_conservative_violation": 1e-08,
        "maximum_cagp_exact_storage_ratio": 0.75,
        "maximum_fitc_exact_mean_rmse": 0.01,
        "maximum_fitc_exact_variance_rmse": 0.003,
        "minimum_exact_fixed_factor_speedup": 2.0,
        "minimum_fitc_fixed_factor_speedup": 1.2,
    }


@pytest.mark.skipif(
    os.environ.get("PHYDRAX_RUN_UQ_BENCHMARKS") != "1",
    reason="set PHYDRAX_RUN_UQ_BENCHMARKS=1 to run the NUTS scheduling benchmark",
)
def test_nuts_scheduling_benchmark_preserves_results_and_writes_report(tmp_path):
    report = run_nuts_scheduling_benchmark(
        profile="smoke",
        targets=("correlated",),
        chain_counts=(4,),
    )
    destination = report.write_json(tmp_path / "uq-nuts-scheduling-smoke.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert report.passed
    assert payload["suite"] == "phydrax-uq-nuts-scheduling"
    assert payload["summary"]["scenario_count"] == 1
    assert payload["summary"]["scenarios_failed"] == 0
    assert (
        payload["scenarios"][0]["metrics"]["integration_step_mismatches"]["value"] == 0.0
    )
    metrics = payload["scenarios"][0]["metrics"]
    assert metrics["maximum_integration_steps"]["value"] >= 1.0
    assert metrics["trajectory_saturation_count"]["value"] >= 0.0
