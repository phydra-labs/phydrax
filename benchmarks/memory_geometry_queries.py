#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Phase-separated dense, automatic, and bounded geometry query evidence.

The untouched pre-cutover record is memory_geometry_queries_baseline.json.
--route baseline records the frozen dense reference with dynamic query points;
--route candidate measures both routes in one process, plus explicit BVH scaling.
Prepared topology/output bytes are reported separately from compiler temporaries.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.geometry._closest_point import segment_query_evidence
from phydrax.geometry.simplicial import SegmentBVH, SegmentTopology
from phydrax.typing import parse

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


GeometryQueryBenchmarkRoute: TypeAlias = Literal["baseline", "candidate"]


def dense_segments(vertices: Array, edges: Array, points: Array) -> tuple[Array, ...]:
    """Frozen pre-cutover dense projection and closest-segment evidence."""
    segments = vertices[edges]
    direction = segments[:, 1] - segments[:, 0]
    relative = points[:, None] - segments[:, 0]
    coordinate = jnp.clip(
        jnp.sum(relative * direction, axis=-1) / jnp.sum(direction**2, axis=-1),
        0.0,
        1.0,
    )
    closest = segments[:, 0] + coordinate[..., None] * direction
    squared = jnp.sum((points[:, None] - closest) ** 2, axis=-1)
    selected = jnp.argmin(squared, axis=-1).astype(jnp.int32)
    first = jnp.min(squared, axis=-1)
    second = jnp.sort(squared, axis=-1)[:, 1]
    selected_coordinate = jnp.take_along_axis(coordinate, selected[:, None], axis=-1)[
        :, 0
    ]
    length = jnp.linalg.norm(direction[selected], axis=-1)
    gap = jnp.sqrt(second) - jnp.sqrt(first)
    feature = jnp.minimum(selected_coordinate, 1.0 - selected_coordinate) * length
    tolerance = (
        64.0
        * jnp.finfo(points.dtype).eps
        * jnp.maximum(jnp.max(jnp.abs(points), axis=-1), 1.0)
    )
    unique = (gap > tolerance) & (feature > tolerance)
    margin = jnp.where(unique, jnp.maximum(jnp.minimum(gap, feature), 0.0), 0.0)
    ordered = jnp.argsort(squared, axis=-1, stable=True)[:, :2]
    return ordered, jnp.take_along_axis(squared, ordered, axis=-1), unique, unique, margin


def candidate_segments(index: SegmentBVH, points: Array) -> tuple[Array, ...]:
    nearest = index.nearest_segments(points, k=2)
    selected = index.vertices[index.edges[nearest.items[:, 0]]]
    direction = selected[:, 1] - selected[:, 0]
    coordinate = jnp.clip(
        jnp.sum((points - selected[:, 0]) * direction, axis=-1)
        / jnp.sum(direction**2, axis=-1),
        0.0,
        1.0,
    )
    closest = selected[:, 0] + coordinate[:, None] * direction
    unique, regular, margin = segment_query_evidence(
        points,
        closest,
        coordinate,
        jnp.linalg.norm(direction, axis=-1),
        nearest.distance_squared[:, 1],
    )
    return nearest.items, nearest.distance_squared, unique, regular, margin


def measure(
    operation: Callable[[Array], tuple[Array, ...]], points: Array, repeats: int
) -> tuple[tuple[Array, ...], dict[str, Any]]:
    entry = jax.jit(operation)
    executable, timing = measure_lower_and_compile(
        lambda: entry.lower(points), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(lambda: executable(points), warmup=2, repeats=repeats)
    evidence = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-geometry-segment-query",
        unavailable_reason="Compiler memory unavailable.",
    )
    return result, {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "output_bytes": logical_array_bytes(result),
    }


def pair_preparation(count: int, dense: bool) -> int:
    """Old retained nonadjacent pair table versus lexicographic tile enumeration."""
    pairs = (
        (first, second)
        for first in range(count)
        for second in range(first + 2, count)
        if not (first == 0 and second == count - 1)
    )
    if dense:
        table = np.asarray(list(pairs), dtype=np.int32).reshape((-1, 2))
        return table.shape[0]
    evaluated = 0
    for first in range(count):
        stop = count - 1 if first == 0 else count
        for start in range(first + 2, stop, 256):
            second = np.arange(start, min(start + 256, stop), dtype=np.intp)
            evaluated += second.shape[0]
    return evaluated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route", choices=get_args(GeometryQueryBenchmarkRoute), default="baseline"
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--pair-sizes", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.route = parse(args.route, GeometryQueryBenchmarkRoute, "route")
    cases: list[dict[str, Any]] = []
    pair_cases: list[dict[str, int]] = []
    for count in args.pair_sizes:
        pair_count, peak = host_allocation_peak(
            lambda: pair_preparation(count, args.route == "baseline")
        )
        pair_cases.append(
            {
                "capacity": count,
                "pair_count": pair_count,
                "peak_bytes": peak,
                "retained_pair_bytes": pair_count * 8 if args.route == "baseline" else 0,
            }
        )
    for count in args.sizes:
        angle = np.arange(count, dtype=np.float64) * (2.0 * np.pi / count)
        vertices = jnp.asarray(
            np.stack((np.cos(angle), np.sin(angle)), axis=-1), dtype=jnp.float64
        )
        edge_vertices = jnp.arange(count, dtype=jnp.int32)
        edges = jnp.stack((edge_vertices, (edge_vertices + 1) % count), axis=-1)
        points = jnp.asarray(
            np.random.default_rng(7).normal(size=(args.queries, 2)), dtype=jnp.float64
        )
        dense_result, dense_evidence = measure(
            lambda query_points: dense_segments(vertices, edges, query_points),
            points,
            args.repeats,
        )
        case: dict[str, Any] = {"capacity": count, "dense": dense_evidence}
        if args.route == "baseline":
            retained, preparation_peak = host_allocation_peak(
                lambda: (vertices, SegmentTopology(edges, num_vertices=count))
            )
            case.update(
                {
                    "prepared_bytes": logical_array_bytes(retained),
                    "preparation_peak_bytes": preparation_peak,
                }
            )
        else:
            index, preparation_peak = host_allocation_peak(
                lambda: SegmentBVH(vertices, edges)
            )
            candidate, evidence = measure(
                lambda query_points: candidate_segments(index, query_points),
                points,
                args.repeats,
            )
            bounded = SegmentBVH(vertices, edges, execution="bvh")
            bvh_result, bvh_evidence = measure(
                lambda query_points: candidate_segments(bounded, query_points),
                points,
                args.repeats,
            )
            for actual, reference in zip(candidate, dense_result, strict=True):
                np.testing.assert_allclose(actual, reference, atol=1.0e-12, rtol=1.0e-12)
            for actual, reference in zip(bvh_result, dense_result, strict=True):
                np.testing.assert_allclose(actual, reference, atol=1.0e-12, rtol=1.0e-12)
            case.update(
                {
                    "candidate": evidence,
                    "bvh": bvh_evidence,
                    "selected_route": "dense"
                    if 64 * args.queries * count <= index.dense_workspace_bytes
                    else "bvh",
                    "dense_workspace_bytes": index.dense_workspace_bytes,
                    "prepared_bytes": logical_array_bytes(index),
                    "preparation_peak_bytes": preparation_peak,
                    "paired_warm_median_ratio": evidence["warm"]["median_seconds"]
                    / dense_evidence["warm"]["median_seconds"],
                }
            )
        cases.append(case)
    root = Path(__file__).resolve().parents[1]
    result: dict[str, Any] = {
        "identity": asdict(
            capture_benchmark_identity(
                root,
                Path(__file__).resolve(),
                ("environment", "cases", "route", "gate", "polygon_pairs"),
            )
        ),
        "environment": asdict(capture_environment()),
        "route": args.route,
        "cases": cases,
        "polygon_pairs": pair_cases,
        "polygon_pair_doubling_ratios": [
            later["peak_bytes"] / earlier["peak_bytes"]
            for earlier, later in zip(pair_cases, pair_cases[1:])
        ],
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "workspace_formula": {
            "auto": "dense when 64*Q*E <= dense_workspace_bytes, BVH otherwise",
            "bvh": "O(min(Q,64)*(leaf_size + k + log2(E))); prepared topology and outputs excluded",
        },
    }
    if args.route == "candidate":
        baseline_bytes = [case["dense"]["compiler"]["temporary_bytes"] for case in cases]
        candidate_bytes = [
            case["candidate"]["compiler"]["temporary_bytes"] for case in cases
        ]
        bvh_bytes = [case["bvh"]["compiler"]["temporary_bytes"] for case in cases]
        result["gate"] = evaluate_memory_gate(
            args.sizes, baseline_bytes, candidate_bytes, MemoryGatePolicy()
        ).to_dict()
        result["bvh_doubling_ratios"] = [b / a for a, b in zip(bvh_bytes, bvh_bytes[1:])]
        result["bvh_scaling_exponent"] = fitted_scaling_exponent(args.sizes, bvh_bytes)
        result["smallest_latency_adopts"] = cases[0]["paired_warm_median_ratio"] <= 1.2
        result["timing_context"] = "Same-process paired samples; contention may remain."
    destination = args.output or Path(__file__).with_name(
        "memory_geometry_queries_dynamic_baseline.json"
        if args.route == "baseline"
        else "memory_geometry_queries.json"
    )
    write_json_atomic(destination, result)
    print(
        json.dumps(
            {
                "record": str(destination),
                "gate": result.get("gate"),
                "smallest_latency_adopts": result.get("smallest_latency_adopts"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
