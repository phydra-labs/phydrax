#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Cold preparation, compiler, warm action, and retained-state spline campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from phydrax.discretization.iga import BSplineGrid, SplineDeRhamComplex
from phydrax.discretization.iga._compatible import (
    _differentials,
    _reduced_grid,
    SplineDifferentialSpace,
    SplineFormComponent,
)
from phydrax.exterior._basis import exterior_indices

from ._io import write_json_atomic
from ._memory import fitted_scaling_exponent, host_allocation_peak, process_peak_rss_bytes
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


def _case(intervals: int, mapped: bool, repeats: int, /) -> dict[str, Any]:
    grid = BSplineGrid.open_uniform(2, intervals)

    def prepare_complex() -> SplineDeRhamComplex:
        return SplineDeRhamComplex(
            (grid, grid),
            geometry=_mapped if mapped else None,
            geometry_id="bilinear-chart" if mapped else None,
        )

    measured, host_peak = host_allocation_peak(
        lambda: measure_synchronized(prepare_complex)
    )
    complex_, preparation = measured
    refresh_peak = None
    refresh_seconds = None
    if mapped:
        refreshed, refresh_peak = host_allocation_peak(
            lambda: measure_synchronized(lambda: complex_.refresh_geometry(_mapped))
        )
        _, refresh_seconds = refreshed
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
        "host_allocation_peak_bytes": host_peak,
        "refresh_host_peak_bytes": refresh_peak,
        "refresh_seconds": refresh_seconds,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "compilation": asdict(compilation),
        "first_execution_seconds": first,
        "warm": warm.to_dict(unit="seconds"),
        "compiler": asdict(compiler),
        "retained_array_bytes": logical_array_bytes(complex_),
        "realization_id": complex_.realization_id,
    }


def _incidence_case(intervals: int, /) -> dict[str, Any]:
    grid = BSplineGrid.open_uniform(2, intervals)
    reduced = _reduced_grid(grid)
    spaces = tuple(
        SplineDifferentialSpace(
            k,
            2,
            tuple(
                SplineFormComponent(
                    k,
                    axes,
                    tuple(reduced if axis in axes else grid for axis in range(2)),
                    periodic=(False, False),
                )
                for axes in exterior_indices(2, k)
            ),
        )
        for k in range(3)
    )
    matrices, peak = host_allocation_peak(lambda: _differentials(spaces, (False, False)))
    return {
        "intervals": intervals,
        "dofs": sum(space.dof_count for space in spaces),
        "host_allocation_peak_bytes": peak,
        "retained_bytes": sum(
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
            for matrix in matrices
        ),
        "nnz": sum(matrix.nnz for matrix in matrices),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/spline_de_rham.json")
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.intervals):
        raise ValueError("Benchmark intervals and repeats must be positive.")
    if args.smoke:
        args.intervals, args.repeats = [2, 4], 1
    cases = [
        _case(intervals, mapped, args.repeats)
        for intervals in args.intervals
        for mapped in (False, True)
    ]
    identity = capture_benchmark_identity(
        Path(__file__).resolve().parent.parent, Path(__file__).resolve(), cases[0].keys()
    )
    report: dict[str, Any] = {
        "kind": "spline-de-rham-capacity-campaign",
        "identity": identity.to_dict(),
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "incidence_cases": [_incidence_case(size) for size in (16, 32, 64, 128)],
    }
    for mapped in (False, True):
        selected = [case for case in cases if (case["geometry"] == "mapped") == mapped]
        report[f"{'mapped' if mapped else 'separable'}_host_scaling_exponent"] = (
            fitted_scaling_exponent(
                [sum(case["dof_counts"]) for case in selected],
                [case["host_allocation_peak_bytes"] for case in selected],
            )
        )
    incidence = report["incidence_cases"]
    report["incidence_host_scaling_exponent"] = fitted_scaling_exponent(
        [case["dofs"] for case in incidence],
        [case["host_allocation_peak_bytes"] for case in incidence],
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
