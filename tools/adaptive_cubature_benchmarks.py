#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Qualify adaptive cubature accuracy, cost, compilation, and memory evidence.

Run: python -m tools.adaptive_cubature_benchmarks [--quick]
Compiler memory estimates are not process or device peak-memory measurements.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    synchronize,
)


@dataclass(frozen=True)
class Case:
    name: str
    dimension: int
    integrand: Callable[[jax.Array], jax.Array]
    exact: float | complex
    breakpoints: tuple[tuple[float, ...], ...] | None = None


def _cases(*, quick: bool) -> tuple[Case, ...]:
    gaussian_dimensions = (2, 3) if quick else (2, 3, 4, 5)
    cases = [
        Case(
            f"gaussian-{dimension}d",
            dimension,
            lambda points: jnp.exp(-jnp.sum(points**2, axis=-1)),
            (math.sqrt(math.pi) * math.erf(1.0)) ** dimension,
        )
        for dimension in gaussian_dimensions
    ]
    cases.extend(
        (
            Case(
                "ridge-3d",
                3,
                lambda points: jnp.exp(-100.0 * (points[:, 0] - 0.2) ** 2),
                4.0 * math.sqrt(math.pi) / 20.0 * (math.erf(8.0) + math.erf(12.0)),
            ),
            Case(
                "diagonal-oscillation-2d",
                2,
                lambda points: jnp.cos(10.0 * jnp.sum(points, axis=-1)),
                (2.0 * math.sin(10.0) / 10.0) ** 2,
            ),
            Case(
                "marked-kink-2d",
                2,
                lambda points: jnp.abs(points[:, 0] - 0.2) * jnp.abs(points[:, 1] + 0.3),
                (1.0 + 0.2**2) * (1.0 + 0.3**2),
                ((0.2,), (-0.3,)),
            ),
        )
    )
    if not quick:
        coefficients = jnp.asarray(
            [0.2660088941584163, 0.4430043456880922, 0.4918098187571035]
        )
        centers = jnp.asarray(
            [0.25354825920421914, 0.7997477160722586, 0.19929440330793777]
        )
        half_width = 0.475
        midpoint = 0.495
        cases.append(
            Case(
                "cusp-regression-3d",
                3,
                lambda points: (
                    half_width**3
                    * jnp.exp(
                        -jnp.sum(
                            coefficients
                            * jnp.abs(half_width * points + midpoint - centers),
                            axis=-1,
                        )
                    )
                ),
                0.5882689256199701,
            )
        )
    return tuple(cases)


def _device_peak_bytes() -> int | None:
    stats = jax.devices()[0].memory_stats()
    if stats is None:
        return None
    value = stats.get("peak_bytes_in_use")
    return None if value is None else int(value)


def _cubature_record(
    case: Case,
    rule,
    /,
    *,
    max_cells: int,
    repeats: int,
) -> dict[str, Any]:
    plan = phx.integration.AdaptiveCubaturePlan(
        rule,
        breakpoints=case.breakpoints,
        max_batch_points=min(64, rule.num_points),
        absolute_tolerance=1e-8,
        relative_tolerance=1e-8,
        max_cells=max_cells,
        max_evaluations=2_000_000,
        collect_partition=True,
        throw=False,
    )
    precision = phx.integration.IntegrationPrecisionPolicy()

    def solve(scale):
        return phx.integration.adaptive_cubature_callable(
            lambda points: scale * case.integrand(points),
            plan,
            precision=precision,
        )

    def arrays(scale):
        estimate = solve(scale)
        return (
            estimate.value,
            estimate.error_estimate,
            estimate.status,
            estimate.num_evaluations,
            estimate.diagnostics.partition.count,
        )

    scale = jnp.asarray(1.0)
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(arrays).lower(scale),
        lambda lowered: lowered.compile(),
    )
    first = compiled(scale)
    synchronize(first)
    _, steady = measure_repeated(lambda: compiled(scale), warmup=0, repeats=repeats)
    estimate = solve(scale)
    synchronize(estimate)
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Backend did not expose compiler cost or memory analysis.",
    )
    value = complex(jnp.asarray(estimate.value))
    exact = complex(case.exact)
    reported = float(estimate.error_estimate)
    true_error = abs(value - exact)
    return {
        "case": case.name,
        "dimension": case.dimension,
        "method": rule.family,
        "exact_degree": rule.exact_degree,
        "embedded_degree": rule.embedded_degree,
        "num_rule_points": rule.num_points,
        "rule_storage_bytes": logical_array_bytes(rule.prepared),
        "negative_weight_mass": rule.negative_weight_mass,
        "value_real": value.real,
        "value_imag": value.imag,
        "true_error": true_error,
        "reported_error": reported,
        "true_to_reported_error": (None if reported == 0.0 else true_error / reported),
        "status": int(estimate.status),
        "successful": bool(estimate.successful),
        "num_evaluations": int(estimate.num_evaluations),
        "num_cells": int(estimate.diagnostics.partition.count),
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "steady": steady.to_seconds_dict(),
        "compiler": asdict(evidence),
        "compiler_estimated_device_memory_bytes": (
            evidence.estimated_device_memory_bytes
        ),
        "observed_device_peak_bytes": _device_peak_bytes(),
        "max_batch_points": plan.max_batch_points,
    }


def _declared_problem(case: Case):
    labels = tuple(f"x{axis}" for axis in range(case.dimension))
    factors = tuple(phx.domain.ScalarInterval(-1.0, 1.0, label=label) for label in labels)
    domain = phx.domain.ProductDomain(*factors)
    function = domain.Function(*labels)(
        lambda *coordinates: case.integrand(jnp.stack(coordinates, axis=-1)[None, :])[0]
    )
    return function, phx.integration.over(domain.component())


def _comparison_record(
    case: Case,
    method: str,
    /,
    *,
    repeats: int,
) -> dict[str, Any]:
    function, target = _declared_problem(case)
    if method == "sparse-grid":
        plan = phx.integration.SparseGridPlan(case.dimension, 5)
        operation = lambda: phx.integration.integrate(function, target, plan)
    elif method == "randomized-sobol":
        plan = phx.integration.QuasiMonteCarloPlan(
            4096, num_replicates=8, sequence="sobol", scrambled=True
        )
        key = jr.key(2026)
        operation = lambda: phx.integration.integrate(function, target, plan, key=key)
    else:
        raise ValueError(f"Unsupported comparison method {method!r}.")
    estimate, steady = measure_repeated(operation, warmup=1, repeats=repeats)
    value = complex(jnp.asarray(estimate.value.data))
    reported = None if estimate.error_estimate is None else float(estimate.error_estimate)
    true_error = abs(value - complex(case.exact))
    return {
        "case": case.name,
        "dimension": case.dimension,
        "method": method,
        "value_real": value.real,
        "value_imag": value.imag,
        "true_error": true_error,
        "reported_error": reported,
        "true_to_reported_error": (
            None if reported in (None, 0.0) else true_error / reported
        ),
        "status": int(estimate.status),
        "successful": bool(estimate.successful),
        "num_evaluations": int(estimate.num_evaluations),
        "steady": steady.to_seconds_dict(),
    }


def run(*, quick: bool, repeats: int) -> dict[str, Any]:
    cases = _cases(quick=quick)
    degrees = (9,) if quick else (7, 9, 11, 13)
    max_cells = 64 if quick else 256
    records: list[dict[str, Any]] = []
    for case in cases:
        for degree in degrees:
            records.append(
                _cubature_record(
                    case,
                    phx.integration.GenzMalikRule(case.dimension, degree),
                    max_cells=max_cells,
                    repeats=repeats,
                )
            )
        if case.dimension <= 3:
            records.append(
                _cubature_record(
                    case,
                    phx.integration.TensorProductCubatureRule(
                        phx.integration.GaussKronrodRule(15),
                        dimension=case.dimension,
                    ),
                    max_cells=max_cells,
                    repeats=repeats,
                )
            )
        if not quick and case.breakpoints is None:
            records.append(_comparison_record(case, "sparse-grid", repeats=repeats))
            records.append(_comparison_record(case, "randomized-sobol", repeats=repeats))
    baseline_path = Path(".tmp/legacy_cubature_baseline.json")
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    return {
        "environment": asdict(capture_environment()),
        "settings": {
            "quick": quick,
            "repeats": repeats,
            "absolute_tolerance": 1e-8,
            "relative_tolerance": 1e-8,
            "max_cells": max_cells,
        },
        "legacy_baseline": baseline,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    payload = run(quick=arguments.quick, repeats=arguments.repeats)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(text, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)
        print(arguments.output)


if __name__ == "__main__":
    main()
