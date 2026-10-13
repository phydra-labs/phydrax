#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host preparation scaling for exact structured transfer routes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    fitted_scaling_exponent,
    host_allocation_peak,
    process_peak_rss_bytes,
)
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def _restrict(
    plan: phx.discretization.StructuredTransferPlan, values: jax.Array
) -> jax.Array:
    return plan.restrict_array(values)


def run(sizes: tuple[int, ...]) -> dict[str, Any]:
    records = []
    host_peaks: list[int] = []
    retained_bytes: list[int] = []
    for n in sizes:
        fine = phx.discretization.TensorGridPlan(
            (phx.discretization.UniformCellAxisSpec(n),), axis_names=("x",)
        ).prepare(jnp.asarray([[0.0], [1.0]], dtype=jnp.float64))
        coarse = phx.discretization.TensorGridPlan(
            (phx.discretization.UniformCellAxisSpec(n // 2),), axis_names=("x",)
        ).prepare(jnp.asarray([[0.0], [1.0]], dtype=jnp.float64))
        plan, peak = host_allocation_peak(
            lambda: phx.discretization.StructuredTransferPlan(fine, coarse)
        )
        values = jnp.arange(n, dtype=jnp.float64)
        action = jax.jit(_restrict)
        compiled, timing = measure_lower_and_compile(
            lambda: action.lower(plan, values), lambda lowered: lowered.compile()
        )
        evidence = compiler_evidence(
            compiled.cost_analysis(),
            compiled.memory_analysis(),
            source="Compiled.memory_analysis",
        )
        _, warm = measure_repeated(lambda: compiled(plan, values), warmup=1, repeats=3)
        host_peaks.append(peak)
        retained_bytes.append(logical_array_bytes(plan))
        records.append(
            {
                "n": n,
                "host_peak_bytes": peak,
                "retained_bytes": retained_bytes[-1],
                "conservation_residual": plan.report.conservation_residual,
                "compiler": asdict(evidence),
                "compilation": asdict(timing),
                "warm": warm.to_seconds_dict(),
            }
        )
    root = Path(__file__).resolve().parents[1]
    return {
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            root,
            Path(__file__).resolve(),
            (
                "n",
                "host_peak_bytes",
                "retained_bytes",
                "conservation_residual",
                "compiler",
                "compilation",
                "warm",
            ),
        ).to_dict(),
        "records": records,
        "host_exponent": fitted_scaling_exponent(sizes, host_peaks),
        "retained_exponent": fitted_scaling_exponent(sizes, retained_bytes),
        "rss_bytes": process_peak_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/memory_structured_transfer.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    write_json_atomic(
        args.output, run((32, 64, 128) if args.smoke else (32, 64, 128, 256))
    )


if __name__ == "__main__":
    main()
