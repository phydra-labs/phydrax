#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _block(estimate: phx.integration.IntegrationEstimate, /) -> None:
    jax.block_until_ready(estimate.value.data)
    jax.block_until_ready(estimate.status)
    if estimate.error_estimate is not None:
        jax.block_until_ready(estimate.error_estimate)


def _measure(
    operation: Callable[[], phx.integration.IntegrationEstimate],
    /,
    *,
    repeats: int,
) -> tuple[phx.integration.IntegrationEstimate, float]:
    estimate = operation()
    _block(estimate)
    started = time.perf_counter()
    for _ in range(repeats):
        estimate = operation()
        _block(estimate)
    return estimate, 1.0e3 * (time.perf_counter() - started) / repeats


def _record(
    scenario: str,
    method: str,
    budget: int,
    analytic: float,
    estimate: phx.integration.IntegrationEstimate,
    elapsed_ms: float,
    /,
) -> dict[str, Any]:
    value = float(estimate.value.data)
    return {
        "scenario": scenario,
        "method": method,
        "requested_budget": int(budget),
        "num_evaluations": int(estimate.num_evaluations),
        "value": value,
        "analytic_value": float(analytic),
        "absolute_error": abs(value - analytic),
        "reported_error": (
            None
            if estimate.error_estimate is None
            else float(estimate.error_estimate)
        ),
        "error_kind": estimate.error_kind,
        "status": int(estimate.status),
        "mean_wall_ms": float(elapsed_ms),
        "method_provenance": estimate.provenance.method,
        "realization_provenance": estimate.provenance.realization,
    }


def _summaries(records: list[dict[str, Any]], /) -> list[dict[str, Any]]:
    methods = sorted({str(record["method"]) for record in records})
    summaries: list[dict[str, Any]] = []
    for method in methods:
        selected = [record for record in records if record["method"] == method]
        errors = np.asarray([record["absolute_error"] for record in selected])
        walls = np.asarray([record["mean_wall_ms"] for record in selected])
        summaries.append(
            {
                "method": method,
                "record_count": len(selected),
                "mean_absolute_error": float(np.mean(errors)),
                "maximum_absolute_error": float(np.max(errors)),
                "median_wall_ms": float(np.median(walls)),
                "all_successful": all(record["status"] == 0 for record in selected),
            }
        )
    return summaries


def _gate(records: list[dict[str, Any]], budgets: tuple[int, ...], /) -> dict[str, Any]:
    largest = max(budgets)
    terminal = [record for record in records if record["requested_budget"] == largest]
    by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    for record in terminal:
        scenario = str(record["scenario"])
        by_scenario.setdefault(scenario, {})[str(record["method"])] = record
    checks: list[dict[str, Any]] = []
    for scenario, methods in sorted(by_scenario.items()):
        bq = methods["bayesian-quadrature"]
        iid = methods["iid-monte-carlo"]
        checks.append(
            {
                "scenario": scenario,
                "criterion": "largest-budget BQ absolute error <= IID absolute error",
                "passed": bq["absolute_error"] <= iid["absolute_error"],
                "bq_absolute_error": bq["absolute_error"],
                "iid_absolute_error": iid["absolute_error"],
            }
        )
        checks.append(
            {
                "scenario": scenario,
                "criterion": "BQ reports posterior standard deviation",
                "passed": (
                    bq["error_kind"]
                    == "bayesian-posterior-standard-deviation"
                    and bq["reported_error"] is not None
                    and math.isfinite(bq["reported_error"])
                ),
                "reported_error": bq["reported_error"],
            }
        )
    checks.append(
        {
            "scenario": "all",
            "criterion": "every integration status converged",
            "passed": all(record["status"] == 0 for record in records),
        }
    )
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
    }


def run_bayesian_quadrature_benchmarks(
    budgets: Sequence[int] = (8, 16, 32, 64),
    /,
    *,
    repeats: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare fixed-design BQ, IID MC, and randomized QMC Gaussian expectations."""
    budgets_ = tuple(int(value) for value in budgets)
    if not budgets_ or any(value < 4 for value in budgets_):
        raise ValueError("budgets must contain integers of at least four.")
    if any(value & (value - 1) for value in budgets_):
        raise ValueError("budgets must be powers of two for the randomized Sobol path.")
    repeats_ = int(repeats)
    if repeats_ < 1:
        raise ValueError("repeats must be positive.")

    location, scale = 0.35, 1.2
    probability = phx.domain.ProbabilityDomain(
        phx.uq.Normal(location, scale), label="z"
    )
    target = phx.integration.expectation(
        probability, target_id="benchmark-gaussian-expectation"
    )
    scenarios = (
        (
            "exp-linear",
            probability.Function("z")(lambda z: jnp.exp(0.25 * z)),
            math.exp(0.25 * location + 0.5 * 0.25**2 * scale**2),
        ),
        (
            "quadratic",
            probability.Function("z")(lambda z: z**2),
            location**2 + scale**2,
        ),
    )
    root_key = jr.key(seed)
    records: list[dict[str, Any]] = []
    for budget_index, budget in enumerate(budgets_):
        kernel = phx.kernels.SquaredExponentialKernel(length_scale=0.9)
        kernel_mean = phx.integration.GaussianKernelMean(target, kernel)
        bq_plan = phx.integration.BayesianQuadraturePlan(
            kernel_mean,
            phx.domain.PointSampling(budget, design="hammersley"),
            solve_regularization=1.0e-10,
            max_points=max(budgets_),
        )
        iid_plan = phx.integration.MonteCarloPlan(budget)
        qmc_plan = phx.integration.QuasiMonteCarloPlan(
            budget,
            sequence="sobol",
            scrambled=True,
            num_replicates=8,
        )
        for scenario_index, (name, integrand, analytic) in enumerate(scenarios):
            bq, bq_ms = _measure(
                lambda f=integrand, p=bq_plan: phx.integration.integrate(f, target, p),
                repeats=repeats_,
            )
            iid_key = jr.fold_in(root_key, 2 * budget_index + scenario_index)
            iid, iid_ms = _measure(
                lambda f=integrand, p=iid_plan, key=iid_key: phx.integration.integrate(
                    f, target, p, key=key
                ),
                repeats=repeats_,
            )
            qmc_key = jr.fold_in(
                root_key,
                2 * len(budgets_) + 2 * budget_index + scenario_index,
            )
            qmc, qmc_ms = _measure(
                lambda f=integrand, p=qmc_plan, key=qmc_key: phx.integration.integrate(
                    f, target, p, key=key
                ),
                repeats=repeats_,
            )
            records.extend(
                (
                    _record(name, "bayesian-quadrature", budget, analytic, bq, bq_ms),
                    _record(name, "iid-monte-carlo", budget, analytic, iid, iid_ms),
                    _record(name, "randomized-sobol", budget, analytic, qmc, qmc_ms),
                )
            )

    return {
        "provenance": {
            "tool": "tools/bayesian_quadrature_benchmarks.py",
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
            "seed": int(seed),
            "repeats": repeats_,
            "budgets": list(budgets_),
            "measure": {
                "distribution": "Normal",
                "location": location,
                "scale": scale,
            },
            "bq_kernel": "SquaredExponentialKernel(length_scale=0.9)",
            "bq_design": "hammersley",
            "bq_solve_regularization": 1.0e-10,
            "qmc_design": "randomized-sobol:8-replicates",
        },
        "records": records,
        "summaries": _summaries(records),
        "gate": _gate(records, budgets_),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Phydrax fixed-design Bayesian quadrature against IID and "
            "randomized QMC on analytic Gaussian expectations."
        )
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=(8, 16, 32, 64))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = run_bayesian_quadrature_benchmarks(
        arguments.budgets,
        repeats=arguments.repeats,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
