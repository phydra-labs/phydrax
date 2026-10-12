"""Host graph and compiled exact-neighborhood memory scaling (small defaults)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.graph import query_neighbors
from phydrax.graph._derived import _line_graph_pairs
from phydrax.graph._geometry import _host_query_pairs, _point_cloud_edges

from ._io import write_json_atomic
from ._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
    MemoryGatePolicy,
    process_peak_rss_bytes,
)
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def _query(source: Array, target: Array) -> tuple[Array, ...]:
    result = query_neighbors(source, target, max_neighbors=8)
    return (
        result.indices,
        result.relative,
        result.distance,
        result.distance_squared,
        result.mask,
        result.count,
        result.evidence.successful,
    )


_QUERY = jax.jit(_query)


def _whole_axis(source: Array, target: Array) -> tuple[Array, ...]:
    result = query_neighbors(
        source,
        target,
        max_neighbors=8,
        target_chunk_size=target.shape[0],
        source_chunk_size=source.shape[0],
    )
    return (
        result.indices,
        result.relative,
        result.distance,
        result.distance_squared,
        result.mask,
        result.count,
        result.evidence.successful,
    )


_WHOLE_AXIS = jax.jit(_whole_axis)


def _case(size: int, repeats: int) -> dict[str, Any]:
    points = np.arange(size, dtype=np.float64)[:, None]
    edges, host_peak = host_allocation_peak(
        lambda: _point_cloud_edges(points, radius=1.0, k=2, add_self_edges=False)
    )
    senders = np.arange(size, dtype=np.int32)
    receivers = senders + 1
    joins, join_peak = host_allocation_peak(
        lambda: _line_graph_pairs(
            senders,
            receivers,
            connectivity="directed_path",
            include_self_transitions=False,
        )
    )
    query_pairs, query_peak = host_allocation_peak(
        lambda: _host_query_pairs(
            points,
            points + 0.25,
            None,
            radius=1.0,
            k=None,
            same_points=False,
            include_self=False,
        )
    )
    source = jnp.asarray(points)
    target = source + 0.25
    compiled, timing = measure_lower_and_compile(
        lambda: _QUERY.lower(source, target), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(
        lambda: compiled(source, target), warmup=1, repeats=repeats
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-graph-query",
        unavailable_reason="Compiler estimate unavailable.",
    )
    whole_compiled, whole_timing = measure_lower_and_compile(
        lambda: _WHOLE_AXIS.lower(source, target), lambda lowered: lowered.compile()
    )
    whole_result, whole_warm = measure_repeated(
        lambda: whole_compiled(source, target), warmup=1, repeats=repeats
    )
    for blocked, whole in zip(result, whole_result, strict=True):
        np.testing.assert_array_equal(blocked, whole)
    whole_evidence = compiler_evidence(
        whole_compiled.cost_analysis(),
        whole_compiled.memory_analysis(),
        source="jax-compiled-whole-axis-graph-query",
        unavailable_reason="Compiler estimate unavailable.",
    )
    return {
        "capacity": size,
        "host_graph_peak_bytes": host_peak,
        "host_graph_output_bytes": edges.nbytes,
        "host_join_peak_bytes": join_peak,
        "host_join_output_bytes": sum(array.nbytes for array in joins),
        "host_radius_query_peak_bytes": query_peak,
        "host_radius_query_output_bytes": sum(array.nbytes for array in query_pairs),
        "whole_axis": {
            "target_chunk_size": size,
            "source_chunk_size": size,
            "lowering_seconds": whole_timing.lowering_seconds,
            "compilation_seconds": whole_timing.compilation_seconds,
            "warm": whole_warm.to_seconds_dict(),
            "compiler": asdict(whole_evidence),
        },
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "output_bytes": logical_array_bytes(result),
        "logical_dense_pair_bytes": size * size * (8 * 2 + 1),
        "logical_block_pair_bytes": min(size, 64) * (min(size, 256) + 8) * (8 + 4),
        "target_chunk_size": 64,
        "source_chunk_size": 256,
        "neighbor_count": 8,
    }


def _source_scan(size: int, repeats: int) -> dict[str, Any]:
    source = jnp.arange(size, dtype=jnp.float64)[:, None]
    target = jnp.arange(64, dtype=jnp.float64)[:, None] + 0.25
    compiled, timing = measure_lower_and_compile(
        lambda: _QUERY.lower(source, target), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(
        lambda: compiled(source, target), warmup=1, repeats=repeats
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-fixed-target-graph-query",
        unavailable_reason="Compiler estimate unavailable.",
    )
    return {
        "source_capacity": size,
        "target_capacity": 64,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "output_bytes": logical_array_bytes(result),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    cases = [_case(size, args.repeats) for size in args.sizes]
    temporary = [case["compiler"]["temporary_bytes"] for case in cases]
    source_scan = [_source_scan(size, args.repeats) for size in args.sizes]
    # Include the one-tile/loop crossover above, then measure steady tiling
    # beyond it with a fixed required output (64 targets by eight neighbors).
    steady_scan = [_source_scan(size * 4, args.repeats) for size in args.sizes]
    record = {
        "identity": asdict(
            capture_benchmark_identity(
                Path(__file__).resolve().parents[1],
                Path(__file__),
                ("environment", "cases", "gate"),
            )
        ),
        "environment": asdict(capture_environment()),
        "cases": cases,
        "source_scan_cases": source_scan,
        "source_scan_temporary_scaling_exponent": fitted_scaling_exponent(
            args.sizes,
            [case["compiler"]["temporary_bytes"] for case in source_scan],
        ),
        "steady_source_scan_cases": steady_scan,
        "steady_source_scan_temporary_scaling_exponent": fitted_scaling_exponent(
            [case["source_capacity"] for case in steady_scan],
            [case["compiler"]["temporary_bytes"] for case in steady_scan],
        ),
        "steady_source_scan_temporary_growth_factors": [
            later["compiler"]["temporary_bytes"] / earlier["compiler"]["temporary_bytes"]
            for earlier, later in zip(steady_scan, steady_scan[1:])
        ],
        "temporary_scaling_exponent": fitted_scaling_exponent(args.sizes, temporary),
        "host_scaling_exponent": fitted_scaling_exponent(
            args.sizes, [case["host_graph_peak_bytes"] for case in cases]
        ),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "gate": None,
    }
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        if [case["capacity"] for case in baseline["cases"]] != args.sizes:
            raise ValueError("Baseline capacities must match the requested sizes.")
        record["gate"] = evaluate_memory_gate(
            args.sizes,
            [case["compiler"]["temporary_bytes"] for case in baseline["cases"]],
            temporary,
            MemoryGatePolicy(),
        ).to_dict()
    write_json_atomic(args.output, record)
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "gate",
                    "temporary_scaling_exponent",
                    "host_scaling_exponent",
                    "process_peak_rss_bytes",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
