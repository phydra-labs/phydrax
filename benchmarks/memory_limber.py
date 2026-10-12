"""Compile-only dense versus exclusive-tail Limber workspace evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    MemoryGatePolicy,
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
from phydrax.ein import contract


def dense(radial: Array, weighted_density: Array, /) -> Array:
    safe = jnp.where(radial > 0.0, radial, 1.0)
    separation = jnp.maximum(radial[None, :] - radial[:, None], 0.0)
    return contract("ij,j->i", separation / safe[None, :], weighted_density)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    sizes = args.sizes or ([64, 256] if args.smoke else [64, 256, 1024, 4096])
    if args.baseline:
        kernel = dense
    else:
        from phydrax.applications.cosmology._observables import _lensing_efficiency

        kernel = _lensing_efficiency
    records: list[dict[str, Any]] = []
    for size in sizes:
        radial = jnp.linspace(0.0, 2.0, size, dtype=jnp.float64) ** 1.3
        weighted = jnp.linspace(0.1, 1.0, size, dtype=jnp.float64) / size
        compiled, timing = measure_lower_and_compile(
            lambda: jax.jit(kernel).lower(radial, weighted),
            lambda lowered: lowered.compile(),
        )
        evidence = compiler_evidence(
            compiled.cost_analysis(), compiled.memory_analysis(), source="jax"
        )
        result, warm = measure_repeated(
            lambda: compiled(radial, weighted), warmup=1, repeats=3
        )
        defect = float(jnp.max(jnp.abs(result - dense(radial, weighted))))
        records.append(
            {
                "capacity": size,
                "compiler": asdict(evidence),
                "compilation": asdict(timing),
                "warm": warm.to_dict(),
                "reference_defect": defect,
                "input_bytes": logical_array_bytes((radial, weighted)),
                "logical_workspace_bytes": 8 * size * size
                if args.baseline
                else 32 * size,
            }
        )
    root = Path(__file__).resolve().parents[1]
    record = {
        "route": "dense" if args.baseline else "exclusive-tail",
        "records": records,
        "logical_scaling_exponent": fitted_scaling_exponent(
            sizes, [r["logical_workspace_bytes"] for r in records]
        ),
        "environment": capture_environment().to_dict(),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "identity": capture_benchmark_identity(
            root,
            Path(__file__),
            ("compiler", "logical_workspace_bytes", "reference_defect"),
        ).to_dict(),
    }
    if not args.baseline:
        baseline = json.loads(
            (root / "benchmarks/memory_limber_baseline.json").read_text()
        )
        baseline_rows = {row["capacity"]: row for row in baseline["records"]}
        record["memory_gate"] = evaluate_memory_gate(
            sizes,
            [baseline_rows[size]["compiler"]["temporary_bytes"] for size in sizes],
            [row["compiler"]["temporary_bytes"] for row in records],
            MemoryGatePolicy(),
        ).to_dict()
        record["compiler_scaling_exponent"] = fitted_scaling_exponent(
            sizes, [row["compiler"]["temporary_bytes"] for row in records]
        )
    path = args.output or (
        root
        / "benchmarks"
        / ("memory_limber_baseline.json" if args.baseline else "memory_limber.json")
    )
    write_json_atomic(path, record)
    print(json.dumps({"record": str(path), "records": records}, indent=2))


if __name__ == "__main__":
    main()
