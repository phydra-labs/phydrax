"""Compiler-gated product graph and exact polynomial preparation evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

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
from phydrax._holomorphic_linear import HolomorphicMultiIndexSet
from phydrax._polynomial._product_graph import ProductGraphPlan
from phydrax._polynomial._scaled_monomial import ScaledMonomialBasis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/memory_polynomial_preparation.json"),
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    capacities = [32, 128, 512]
    rows: list[dict[str, Any]] = []
    for lanes in capacities:
        plan, peak = host_allocation_peak(
            lambda: ProductGraphPlan.total_degree(
                4, 8, include_constant=True, maximum_nodes=4096
            )
        )
        values = jnp.full((lanes, 4), 0.75, dtype=jnp.float32)

        def evaluate(inputs: jax.Array) -> jax.Array:
            return plan.evaluate(inputs)

        compiled, timing = measure_lower_and_compile(
            lambda: jax.jit(evaluate).lower(values), lambda lowered: lowered.compile()
        )
        evidence = compiler_evidence(
            compiled.cost_analysis(),
            compiled.memory_analysis(),
            source="jax.Compiled.memory_analysis",
        )
        result, warm = measure_repeated(
            lambda: compiled(values), warmup=1, repeats=2 if args.smoke else 5
        )
        rows.append(
            {
                "lanes": lanes,
                "node_count": plan.node_count,
                "term_count": len(plan.terms),
                "live_bytes": plan.live_bytes(lanes, 4),
                "retained_bytes": logical_array_bytes(plan),
                "host_peak_bytes": peak,
                "compiler": asdict(evidence),
                "timing": asdict(timing),
                "warm": warm.to_dict(),
                "output_bytes": result.nbytes,
            }
        )
    preparations = []
    for dimension, degree in [(2, 4), (3, 4), (4, 4)]:
        holomorphic, hp = host_allocation_peak(
            lambda: HolomorphicMultiIndexSet.total_degree(dimension, degree)
        )
        scaled, sp = host_allocation_peak(lambda: ScaledMonomialBasis(dimension, degree))
        preparations.append(
            {
                "dimension": dimension,
                "degree": degree,
                "count": holomorphic.count,
                "index_set_id": holomorphic.index_set_id,
                "basis_id": scaled.basis_id,
                "holomorphic_host_peak_bytes": hp,
                "scaled_host_peak_bytes": sp,
            }
        )
    record: dict[str, Any] = {
        "rows": rows,
        "preparations": preparations,
        "environment": capture_environment().to_dict(),
        "rss_bytes": process_peak_rss_bytes(),
        "logical_scaling_exponent": fitted_scaling_exponent(
            capacities, [row["live_bytes"] for row in rows]
        ),
        "compiler_scaling_exponent": fitted_scaling_exponent(
            capacities, [row["compiler"]["temporary_bytes"] for row in rows]
        ),
        "declared_scaling_exponent": 1.0,
    }
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        record["gate"] = evaluate_memory_gate(
            capacities,
            [row["compiler"]["temporary_bytes"] for row in baseline["rows"]],
            [row["compiler"]["temporary_bytes"] for row in rows],
            MemoryGatePolicy(),
        ).to_dict()
    record["identity"] = capture_benchmark_identity(
        Path.cwd(), Path(__file__), record.keys()
    ).to_dict()
    write_json_atomic(args.output, record)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gate": record.get("gate"),
                "temporary_bytes": [row["compiler"]["temporary_bytes"] for row in rows],
            }
        )
    )


if __name__ == "__main__":
    main()
