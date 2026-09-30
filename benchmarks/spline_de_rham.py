#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Cold preparation, compiler, warm action, and retained-state spline campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import Array

from phydrax.discretization.iga import BSplineGrid, SplineDeRhamComplex

from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


def _mapped(point: Array) -> Array:
    return jnp.stack((point[0] + 0.1 * point[0] * point[1], point[1]))


def _case(intervals: int, mapped: bool, repeats: int, /) -> dict[str, object]:
    grid = BSplineGrid.open_uniform(2, intervals)

    def prepare_complex() -> SplineDeRhamComplex:
        return SplineDeRhamComplex(
            (grid, grid),
            geometry=_mapped if mapped else None,
            geometry_id="bilinear-chart" if mapped else None,
        )

    complex_, preparation = measure_synchronized(prepare_complex)
    values = jnp.linspace(-0.5, 0.8, complex_.dof_count(1), dtype=jnp.float64)

    def action(value: Array) -> tuple[Array, Array, Array]:
        return (
            complex_.hodge_star(1, value),
            complex_.codifferential(1, value),
            complex_.hodge_laplacian(1, value),
        )

    executable, compilation = measure_lower_and_compile(
        lambda: jax.jit(action).lower(values), lambda lowered: lowered.compile()
    )
    _, first = measure_synchronized(lambda: executable(values))
    _, warm = measure_repeated(lambda: executable(values), warmup=1, repeats=repeats)
    compiler = compiler_evidence(
        executable.cost_analysis(), executable.memory_analysis(), source="xla"
    )
    return {
        "intervals": intervals,
        "geometry": "mapped" if mapped else "separable",
        "dof_counts": complex_.dof_counts,
        "preparation_seconds": preparation,
        "compilation": asdict(compilation),
        "first_execution_seconds": first,
        "warm": warm.to_dict(unit="seconds"),
        "compiler": asdict(compiler),
        "retained_array_bytes": logical_array_bytes(complex_),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/spline_de_rham.json")
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.intervals):
        raise ValueError("Benchmark intervals and repeats must be positive.")
    cases = [
        _case(intervals, mapped, args.repeats)
        for intervals in args.intervals
        for mapped in (False, True)
    ]
    identity = capture_benchmark_identity(
        Path(__file__).resolve().parent.parent, Path(__file__).resolve(), cases[0].keys()
    )
    report = {
        "kind": "spline-de-rham-capacity-campaign",
        "identity": identity.to_dict(),
        "environment": capture_environment().to_dict(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
