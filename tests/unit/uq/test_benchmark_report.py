#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import pytest

from tools.uq_benchmarks import runner
from tools.uq_benchmarks.report import BenchmarkReport, Metric, ScenarioResult


def _scenario(name, value=0.5):
    def run(configuration, seed):
        return ScenarioResult(
            name=name,
            description=f"{name} benchmark",
            seed=seed,
            metrics={
                "accuracy": Metric(
                    value,
                    "accuracy",
                    minimum=0.0,
                    maximum=1.0,
                )
            },
            metadata={"profile": configuration.profile},
        )

    return run


def test_metric_uses_inclusive_finite_release_gates():
    lower = Metric(0.0, "accuracy", minimum=0.0, maximum=1.0)
    upper = Metric(1.0, "accuracy", minimum=0.0, maximum=1.0)
    failed = Metric(1.1, "accuracy", minimum=0.0, maximum=1.0)

    assert lower.passed
    assert upper.passed
    assert not failed.passed
    assert failed.as_dict()["gate"]["inclusive"]
    with pytest.raises(ValueError, match="finite"):
        Metric(float("nan"), "accuracy")
    with pytest.raises(ValueError, match="exceed"):
        Metric(0.5, "accuracy", minimum=1.0, maximum=0.0)


def test_report_serialization_is_strict_atomic_and_category_aggregated(tmp_path):
    passed = _scenario("passed")(runner.get_configuration("smoke"), 10)
    failed = _scenario("failed", 2.0)(runner.get_configuration("smoke"), 20)
    report = BenchmarkReport(
        profile="smoke",
        root_seed=7,
        started_at_utc="2026-07-18T00:00:00+00:00",
        duration_seconds=1.25,
        configuration={"profile": "smoke"},
        environment={"jax_backend": "cpu"},
        scenarios=(passed, failed),
    )

    destination = report.write_json(tmp_path / "nested" / "report.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert not report.passed
    assert "schema_version" not in payload
    assert payload["summary"]["scenarios_passed"] == 1
    assert payload["summary"]["scenarios_failed"] == 1
    assert payload["summary"]["metric_categories"]["accuracy"] == {
        "failed": 1,
        "gated": 2,
        "passed": 1,
        "total": 2,
    }
    assert payload["summary"]["metric_aggregates"]["accuracy"] == {
        "categories": ["accuracy"],
        "count": 2,
        "failed": 1,
        "maximum": 2.0,
        "mean": 1.25,
        "minimum": 0.5,
        "units": [],
    }
    assert not (destination.parent / f".{destination.name}.tmp").exists()


def test_runner_keeps_registry_order_and_seed_when_selecting_subsets(monkeypatch):
    scenarios = {
        "first": _scenario("first"),
        "second": _scenario("second"),
        "third": _scenario("third"),
    }
    monkeypatch.setattr(runner, "SCENARIOS", scenarios)

    full = runner.run_benchmark_matrix(profile="smoke", root_seed=11)
    subset = runner.run_benchmark_matrix(
        profile="smoke",
        root_seed=11,
        scenario_names=("third", "first"),
    )

    assert tuple(result.name for result in subset.scenarios) == ("first", "third")
    assert full.scenarios[0].seed == subset.scenarios[0].seed == 10_011
    assert full.scenarios[2].seed == subset.scenarios[1].seed == 30_011


def test_runner_records_scenario_exceptions_without_losing_the_report(monkeypatch):
    def broken(configuration, seed):
        raise RuntimeError(f"failure for {configuration.profile} at {seed}")

    monkeypatch.setattr(runner, "SCENARIOS", {"broken": broken})
    report = runner.run_benchmark_matrix(profile="smoke", root_seed=5)

    assert not report.passed
    assert report.scenarios[0].error_type == "RuntimeError"
    assert report.scenarios[0].failures == ("scenario_error",)
    assert "failure for smoke" in report.scenarios[0].error_message


def test_stochastic_gradient_benchmark_controls_are_profiled_and_registered():
    smoke = runner.get_configuration("smoke")
    standard = runner.get_configuration("standard")

    assert tuple(runner.SCENARIOS)[-1] == "stochastic_gradient_regression"
    assert smoke.sgmcmc_burnin > 0
    assert smoke.sgmcmc_draws > 0
    assert smoke.sgmcmc_batch_size > 0
    assert smoke.sgmcmc_steps_per_sample > 0
    assert standard.sgmcmc_burnin >= smoke.sgmcmc_burnin
    assert standard.sgmcmc_draws >= smoke.sgmcmc_draws
