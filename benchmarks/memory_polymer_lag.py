"""Compiler gate for exact sequence contour-lag accumulation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
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
from phydrax.applications.polymer_liquids._mixture import SequenceFormFactorPlan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 256, 1024, 4096])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = [64, 128] if args.smoke else args.sizes
    rows: list[dict[str, Any]] = []
    for size in sizes:
        plan, peak = host_allocation_peak(
            lambda: SequenceFormFactorPlan(
                "gaussian-chain", np.arange(size, dtype=np.int32) % 3, 1.2
            )
        )
        wave = jnp.linspace(0.0, 4.0, 16, dtype=jnp.float64)
        compiled, timing = measure_lower_and_compile(
            # Equinox's decorator stub preserves the callable but omits its lower API.
            # ty: ignore[unresolved-attribute]
            lambda: eqx.filter_jit(SequenceFormFactorPlan.evaluate).lower(plan, wave),
            lambda lowered: lowered.compile(),
        )
        evidence = compiler_evidence(
            compiled.compiled.cost_analysis(),
            compiled.compiled.memory_analysis(),
            source="Compiled.memory_analysis",
        )
        _, duration = measure_repeated(lambda: compiled(plan, wave), warmup=1, repeats=3)
        rows.append(
            {
                "capacity": size,
                "compiler": asdict(evidence),
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "warm": duration.to_dict(),
                "host_peak_bytes": peak,
                "retained_bytes": logical_array_bytes(plan),
                "logical_workspace_bytes": (
                    16 * size * size * 8 if args.baseline is None else size * 32
                ),
            }
        )
    record: dict[str, Any] = {
        "rows": rows,
        "environment": capture_environment().to_dict(),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }
    record["identity"] = capture_benchmark_identity(
        Path.cwd(), Path(__file__), record.keys()
    ).to_dict()
    record["logical_scaling_exponent"] = fitted_scaling_exponent(
        sizes, [row["logical_workspace_bytes"] for row in rows]
    )
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        record["gate"] = evaluate_memory_gate(
            sizes,
            [row["compiler"]["temporary_bytes"] for row in baseline["rows"]],
            [row["compiler"]["temporary_bytes"] for row in rows],
            MemoryGatePolicy(),
        ).to_dict()
    write_json_atomic(args.output, record)


if __name__ == "__main__":
    main()
