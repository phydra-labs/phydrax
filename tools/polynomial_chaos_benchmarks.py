#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


@dataclass(frozen=True)
class PolynomialChaosBenchmarkRecord:
    scenario: str
    method: str
    model_calls: int
    degree: int | None
    feature_count: int | None
    mean: float
    variance: float
    mean_absolute_error: float
    variance_absolute_error: float
    elapsed_seconds: float
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class PolynomialChaosBenchmarkSummary:
    method: str
    scenario_count: int
    total_model_calls: int
    maximum_mean_absolute_error: float
    maximum_variance_absolute_error: float
    total_elapsed_seconds: float


@dataclass(frozen=True)
class _Scenario:
    name: str
    factors: tuple[phx.domain.ProbabilityDomain, ...]
    degree: int
    quadrature_order: int
    model: Callable[..., Any]
    reference_mean: float
    reference_variance: float
    reference_provenance: str

    @property
    def budget(self) -> int:
        return self.quadrature_order ** len(self.factors)


class _ModelCounter:
    def __init__(self, model: Callable[..., Any]):
        self.model = model
        self.calls = 0

    def __call__(self, *values):
        self.calls += 1
        return self.model(*values)


def _ishigami() -> _Scenario:
    factors = tuple(
        phx.domain.ProbabilityDomain(phx.uq.Uniform(-jnp.pi, jnp.pi), label=label)
        for label in ("x1", "x2", "x3")
    )
    a = 7.0
    b = 0.1

    def model(x1, x2, x3):
        return jnp.sin(x1) + a * jnp.sin(x2) ** 2 + b * x3**4 * jnp.sin(x1)

    variance = 0.5 + a**2 / 8.0 + b * math.pi**4 / 5.0 + b**2 * math.pi**8 / 18.0
    return _Scenario(
        "ishigami",
        factors,
        8,
        10,
        model,
        a / 2.0,
        variance,
        "analytic Ishigami moments",
    )


def _gaussian_polynomial() -> _Scenario:
    x = phx.domain.ProbabilityDomain(phx.uq.Normal(1.0, 2.0), label="x")
    y = phx.domain.ProbabilityDomain(phx.uq.Normal(-0.5, 1.5), label="y")

    def model(x_value, y_value):
        first = (x_value - 1.0) / 2.0
        second = (y_value + 0.5) / 1.5
        return (
            1.0
            + 2.0 * first
            - 0.5 * second
            + 0.75 * (first**2 - 1.0)
            + first * second
        )

    return _Scenario(
        "exact-gaussian-polynomial",
        (x, y),
        4,
        4,
        model,
        1.0,
        4.0 + 0.25 + 2.0 * 0.75**2 + 1.0,
        "analytic standard-normal polynomial moments",
    )


def _solver_model(first, second):
    matrix = jnp.asarray(
        [[2.0 + 0.2 * first, 0.25], [0.25, 1.5 + 0.1 * second]]
    )
    right_hand_side = jnp.asarray([1.0 + 0.1 * second, 2.0 - 0.1 * first])
    result = phx.linalg.solve(
        phx.linalg.LinearSystem(phx.linalg.DenseLinearOperator(matrix)),
        right_hand_side,
    )
    return result.value[0]


def _solver_scenario() -> _Scenario:
    first = phx.domain.ProbabilityDomain(phx.uq.Uniform(-1.0, 1.0), label="u")
    second = phx.domain.ProbabilityDomain(phx.uq.Uniform(-1.0, 1.0), label="v")
    reference_basis = phx.uq.PolynomialChaosBasis((first, second), 10)
    reference = phx.uq.PolynomialChaosProjectionPlan(
        reference_basis,
        phx.integration.ProductIntegrationPlan(
            {
                "u": phx.integration.FixedQuadraturePlan(
                    phx.integration.GaussLegendreRule(16)
                ),
                "v": phx.integration.FixedQuadraturePlan(
                    phx.integration.GaussLegendreRule(16)
                ),
            }
        ),
    ).fit(_solver_model)
    return _Scenario(
        "solver-derived-observable",
        (first, second),
        4,
        5,
        _solver_model,
        float(reference.expansion.mean),
        float(reference.expansion.variance),
        "independent order-16 tensor projection reference",
    )


def _projection_plan(scenario: _Scenario, basis: phx.uq.PolynomialChaosBasis):
    plans = {}
    for factor in scenario.factors:
        if isinstance(factor.distribution, phx.uq.Uniform):
            rule = phx.integration.GaussLegendreRule(scenario.quadrature_order)
        elif isinstance(factor.distribution, phx.uq.Normal):
            rule = phx.integration.GaussHermiteRule(scenario.quadrature_order)
        else:
            raise TypeError("Benchmark scenarios support only Uniform and Normal laws.")
        plans[factor.label] = phx.integration.FixedQuadraturePlan(rule)
    return phx.uq.PolynomialChaosProjectionPlan(
        basis, phx.integration.ProductIntegrationPlan(plans)
    )


def _samples(scenario: _Scenario, *, sampler: str, key):
    distributions = {
        factor.label: factor.distribution for factor in scenario.factors
    }
    return phx.uq.sample_joint(
        distributions,
        num_samples=scenario.budget,
        key=key,
        sampler=sampler,
    )


def _point_matrix(scenario: _Scenario, samples: phx.uq.RandomSampleBatch):
    return jnp.stack(
        tuple(samples.values[factor.label] for factor in scenario.factors), axis=-1
    )


def _evaluate_samples(scenario: _Scenario, points: jax.Array):
    return jax.vmap(scenario.model)(
        *tuple(points[:, index] for index in range(points.shape[1]))
    )


def _record(
    scenario: _Scenario,
    method: str,
    model_calls: int,
    elapsed: float,
    mean: Any,
    variance: Any,
    /,
    *,
    degree: int | None,
    feature_count: int | None,
    provenance: Mapping[str, Any],
) -> PolynomialChaosBenchmarkRecord:
    mean_value = float(mean)
    variance_value = float(variance)
    return PolynomialChaosBenchmarkRecord(
        scenario=scenario.name,
        method=method,
        model_calls=model_calls,
        degree=degree,
        feature_count=feature_count,
        mean=mean_value,
        variance=variance_value,
        mean_absolute_error=abs(mean_value - scenario.reference_mean),
        variance_absolute_error=abs(variance_value - scenario.reference_variance),
        elapsed_seconds=elapsed,
        provenance={
            **dict(provenance),
            "reference": scenario.reference_provenance,
        },
    )


def _run_scenario(
    scenario: _Scenario, key
) -> tuple[PolynomialChaosBenchmarkRecord, ...]:
    basis = phx.uq.PolynomialChaosBasis(scenario.factors, scenario.degree)

    projection_counter = _ModelCounter(scenario.model)
    started = time.perf_counter()
    projection = _projection_plan(scenario, basis).fit(projection_counter)
    jax.block_until_ready(projection.expansion.coefficient_leaves)
    projection_elapsed = time.perf_counter() - started
    projection_record = _record(
        scenario,
        "projection-pce",
        projection.model_evaluations,
        projection_elapsed,
        projection.expansion.mean,
        projection.expansion.variance,
        degree=basis.degree,
        feature_count=basis.feature_count,
        provenance={
            "fit_method": projection.method,
            "fit_evidence": dict(projection.evidence),
            "pointwise_python_traces": projection_counter.calls,
        },
    )

    regression_key, qmc_key, mc_key = jr.split(key, 3)
    started = time.perf_counter()
    regression_samples = _samples(
        scenario, sampler="sobol_scrambled", key=regression_key
    )
    regression_points = _point_matrix(scenario, regression_samples)
    jax.block_until_ready(regression_points)
    regression_values = _evaluate_samples(scenario, regression_points)
    regression = phx.uq.PolynomialChaosRegressionPlan(basis).fit(
        regression_points, regression_values
    )
    jax.block_until_ready(regression.expansion.coefficient_leaves)
    regression_elapsed = time.perf_counter() - started
    regression_record = _record(
        scenario,
        "regression-pce",
        scenario.budget,
        regression_elapsed,
        regression.expansion.mean,
        regression.expansion.variance,
        degree=basis.degree,
        feature_count=basis.feature_count,
        provenance={
            "fit_method": regression.method,
            "fit_evidence": dict(regression.evidence),
            "design": "joint scrambled Sobol",
        },
    )

    records = [projection_record, regression_record]
    for method, sampler, sample_key in (
        ("qmc", "sobol_scrambled", qmc_key),
        ("mc", "uniform", mc_key),
    ):
        started = time.perf_counter()
        samples = _samples(scenario, sampler=sampler, key=sample_key)
        points = _point_matrix(scenario, samples)
        jax.block_until_ready(points)
        values = _evaluate_samples(scenario, points)
        mean = jnp.mean(values)
        variance = jnp.mean(jnp.real((values - mean) * jnp.conj(values - mean)))
        jax.block_until_ready((mean, variance))
        elapsed = time.perf_counter() - started
        records.append(
            _record(
                scenario,
                method,
                scenario.budget,
                elapsed,
                mean,
                variance,
                degree=None,
                feature_count=None,
                provenance={
                    "design": sampler,
                    "moment_variance_normalization": "population",
                },
            )
        )
    return tuple(records)


def _summaries(
    records: Sequence[PolynomialChaosBenchmarkRecord],
) -> tuple[PolynomialChaosBenchmarkSummary, ...]:
    methods = tuple(dict.fromkeys(record.method for record in records))
    summaries = []
    for method in methods:
        selected = tuple(record for record in records if record.method == method)
        summaries.append(
            PolynomialChaosBenchmarkSummary(
                method=method,
                scenario_count=len(selected),
                total_model_calls=sum(record.model_calls for record in selected),
                maximum_mean_absolute_error=max(
                    record.mean_absolute_error for record in selected
                ),
                maximum_variance_absolute_error=max(
                    record.variance_absolute_error for record in selected
                ),
                total_elapsed_seconds=sum(record.elapsed_seconds for record in selected),
            )
        )
    return tuple(summaries)


def _gate(records: Sequence[PolynomialChaosBenchmarkRecord]) -> Mapping[str, Any]:
    thresholds = {
        "ishigami": (0.08, 2.0),
        "exact-gaussian-polynomial": (2e-8, 2e-7),
        "solver-derived-observable": (2e-4, 2e-4),
    }
    checks = []
    for scenario in thresholds:
        selected = tuple(record for record in records if record.scenario == scenario)
        budgets = {record.model_calls for record in selected}
        checks.append(
            {
                "name": f"{scenario}:matched-model-calls",
                "passed": len(selected) == 4 and len(budgets) == 1,
                "observed": tuple(sorted(budgets)),
            }
        )
        mean_limit, variance_limit = thresholds[scenario]
        for method in ("projection-pce", "regression-pce"):
            record = next(row for row in selected if row.method == method)
            passed = (
                math.isfinite(record.mean)
                and math.isfinite(record.variance)
                and record.mean_absolute_error <= mean_limit
                and record.variance_absolute_error <= variance_limit
            )
            checks.append(
                {
                    "name": f"{scenario}:{method}:accuracy",
                    "passed": passed,
                    "mean_absolute_error": record.mean_absolute_error,
                    "variance_absolute_error": record.variance_absolute_error,
                    "limits": {
                        "mean_absolute_error": mean_limit,
                        "variance_absolute_error": variance_limit,
                    },
                }
            )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _source_provenance() -> Mapping[str, Any]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "precision": "jax-runtime-default",
    }


def run_campaign(seed: int = 20260829) -> Mapping[str, Any]:
    scenarios = (_ishigami(), _gaussian_polynomial(), _solver_scenario())
    records = tuple(
        record
        for scenario, key in zip(
            scenarios,
            jr.split(jr.key(seed), len(scenarios)),
            strict=True,
        )
        for record in _run_scenario(scenario, key)
    )
    summaries = _summaries(records)
    gate = _gate(records)
    return {
        "passed": gate["passed"],
        "configuration": {
            "seed": seed,
            "comparison": "matched model calls per scenario",
            "scenarios": [
                {
                    "name": scenario.name,
                    "degree": scenario.degree,
                    "quadrature_order": scenario.quadrature_order,
                    "model_call_budget": scenario.budget,
                    "reference": scenario.reference_provenance,
                }
                for scenario in scenarios
            ],
        },
        "source_provenance": _source_provenance(),
        "gate": gate,
        "summaries": [asdict(summary) for summary in summaries],
        "records": [asdict(record) for record in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched-call nonintrusive polynomial-chaos benchmark campaign."
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    campaign = run_campaign(arguments.seed)
    payload = json.dumps(campaign, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
