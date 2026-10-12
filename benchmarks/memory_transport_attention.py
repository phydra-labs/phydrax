"""Compare dense and bounded transport/attention routes in one process.

Run with --smoke for small cases; compiler bytes, not contended timings, own gates.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

import phydrax as phx
from phydrax.nn.layers import MeasureAwareAttention

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


def _measure(
    operation: Callable[[Array], Array], argument: Array, repeats: int
) -> dict[str, Any]:
    entry = jax.jit(operation)
    executable, timing = measure_lower_and_compile(
        lambda: entry.lower(argument), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(
        lambda: executable(argument), warmup=1, repeats=repeats
    )
    evidence = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-compiled-memory-route",
        unavailable_reason="Compiler estimates unavailable.",
    )
    return {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "output_bytes": logical_array_bytes(result),
        "checksum": float(jnp.sum(result)),
    }


def _measure_case(capacity: int, block_size: int, repeats: int) -> list[dict[str, Any]]:
    query_count = 256
    points = jnp.linspace(-1.0, 1.0, query_count, dtype=jnp.float64)[:, None]
    target_points = jnp.linspace(-0.8, 1.2, capacity, dtype=jnp.float64)[:, None]
    source = phx.integration.discrete(
        points,
        phx.axes.AxisArray(jnp.ones(query_count, dtype=jnp.float64), dims=("atom",)),
        axes="atom",
        normalized=True,
    )
    target = phx.integration.discrete(
        target_points,
        phx.axes.AxisArray(jnp.ones(capacity, dtype=jnp.float64), dims=("atom",)),
        axes="atom",
        normalized=True,
    )
    balanced = phx.transport.discrete_problem(
        source, target, cost=phx.transport.SquaredEuclideanCost()
    )
    unbalanced = phx.transport.unbalanced_problem(
        source,
        target,
        cost=phx.transport.SquaredEuclideanCost(),
        source_marginal_penalty=1.3,
        target_marginal_penalty=2.1,
    )
    q = jnp.sin(jnp.arange(query_count * 16, dtype=jnp.float64)).reshape(
        1, query_count, 2, 8
    )
    k = jnp.cos(jnp.arange(capacity * 16, dtype=jnp.float64)).reshape(1, capacity, 2, 8)
    v = jnp.sin(k)
    weights = jnp.linspace(0.1, 1.0, capacity, dtype=jnp.float64)
    rows: list[dict[str, Any]] = []
    for route, tile in (("dense", None), ("blockwise", block_size)):
        for family, problem, solver in (
            (
                "balanced",
                balanced,
                phx.transport.Sinkhorn(0.5, max_iterations=10, block_size=tile),
            ),
            (
                "unbalanced",
                unbalanced,
                phx.transport.UnbalancedSinkhorn(0.5, max_iterations=10, block_size=tile),
            ),
        ):

            def transport(epsilon: Array) -> Array:
                if isinstance(problem, phx.transport.DiscreteTransportProblem):
                    result = phx.transport.Sinkhorn(
                        epsilon, max_iterations=10, block_size=tile
                    )(problem)
                else:
                    result = phx.transport.UnbalancedSinkhorn(
                        epsilon, max_iterations=10, block_size=tile
                    )(problem)
                return jnp.stack(
                    (
                        result.regularized_cost,
                        result.diagnostics.status.astype(jnp.float64),
                    )
                )

            record = _measure(transport, solver.epsilon, repeats)
            record.update(
                family=family,
                route=route,
                capacity=capacity,
                block_size=tile,
                logical_pair_workspace_bytes=query_count
                * (capacity if tile is None else min(tile, capacity))
                * 8,
                retained_input_bytes=logical_array_bytes(problem),
            )
            rows.append(record)
        layer = MeasureAwareAttention(
            source_channels=16,
            query_channels=16,
            out_channels=16,
            num_heads=2,
            head_dim=8,
            execution=route,
            block_size=block_size,
        )
        record = _measure(
            lambda query: layer.attend_projected(query, k, v, weights), q, repeats
        )
        record.update(
            family="attention",
            route=route,
            capacity=capacity,
            block_size=tile,
            logical_pair_workspace_bytes=2
            * query_count
            * (capacity if tile is None else min(tile, capacity))
            * 8,
            retained_input_bytes=logical_array_bytes((layer, q, k, v, weights)),
        )
        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    args = parser.parse_args()
    sizes = [128, 256] if args.smoke else args.sizes
    rows: list[dict[str, Any]] = []
    for capacity in sizes:
        cases, peak = host_allocation_peak(
            lambda: _measure_case(capacity, args.block_size, args.repeats)
        )
        for case in cases:
            case["host_allocation_peak_bytes"] = peak
        rows.extend(cases)
    gates: dict[str, Any] = {}
    for family in ("balanced", "unbalanced", "attention"):
        baseline = [
            row["compiler"]["temporary_bytes"]
            for row in rows
            if row["family"] == family and row["route"] == "dense"
        ]
        candidate = [
            row["compiler"]["temporary_bytes"]
            for row in rows
            if row["family"] == family and row["route"] == "blockwise"
        ]
        gates[family] = {
            "memory": evaluate_memory_gate(
                sizes, baseline, candidate, MemoryGatePolicy()
            ).to_dict(),
            "baseline_exponent": fitted_scaling_exponent(sizes, baseline),
            "candidate_exponent": fitted_scaling_exponent(sizes, candidate),
            "declared_fixed_query_fixed_tile_exponent": 0.0,
        }
    fields = ("environment", "configuration", "cases", "gates", "process_peak_rss_bytes")
    record = {
        "identity": capture_benchmark_identity(
            Path.cwd(), Path(__file__), fields
        ).to_dict(),
        "environment": asdict(capture_environment()),
        "configuration": {
            "capacities": sizes,
            "block_size": args.block_size,
            "repeats": args.repeats,
            "timing_acceptance": "deferred-quiet-serialized-wave",
        },
        "cases": rows,
        "gates": gates,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }
    write_json_atomic(args.output, record)
    print(gates)


if __name__ == "__main__":
    main()
