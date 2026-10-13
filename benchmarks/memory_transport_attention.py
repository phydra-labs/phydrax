"""Compare explicit dense and automatic transport/attention routes in one run.

Moderate cases execute; larger cases compile only. Above-budget compiler bytes
own adoption gates. Historical baseline records are not the gate comparison.
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
from phydrax.transport._geometry import resolve_sinkhorn_block_size

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
    operation: Callable[[Array], Array],
    argument: Array,
    repeats: int,
    *,
    compile_only: bool,
) -> dict[str, Any]:
    entry = jax.jit(operation)
    executable, timing = measure_lower_and_compile(
        lambda: entry.lower(argument), lambda lowered: lowered.compile()
    )
    evidence = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-compiled-memory-route",
        unavailable_reason="Compiler estimates unavailable.",
    )
    record = {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "compile_only": compile_only,
        "warm": None,
        "compiler": asdict(evidence),
        "output_bytes": evidence.output_bytes,
        "checksum": None,
    }
    if not compile_only:
        result, warm = measure_repeated(
            lambda: executable(argument), warmup=1, repeats=repeats
        )
        record.update(
            warm=warm.to_seconds_dict(),
            output_bytes=logical_array_bytes(result),
            checksum=float(jnp.sum(result)),
        )
    return record


def _measure_case(
    capacity: int,
    repeats: int,
    *,
    compile_only: bool,
    sinkhorn_workspace_bytes: int,
    attention_workspace_bytes: int,
) -> list[dict[str, Any]]:
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
    for requested_route in ("dense", "default"):
        for family, problem in (("balanced", balanced), ("unbalanced", unbalanced)):
            options: dict[str, Any] = {
                "max_iterations": 10,
                "sinkhorn_workspace_bytes": sinkhorn_workspace_bytes,
            }
            if requested_route == "dense":
                options["block_size"] = None
            # Deliberately omit block_size for the default candidate.
            solver = (
                phx.transport.Sinkhorn(0.5, **options)
                if family == "balanced"
                else phx.transport.UnbalancedSinkhorn(0.5, **options)
            )

            def transport(epsilon: Array) -> Array:
                if isinstance(problem, phx.transport.DiscreteTransportProblem):
                    result = phx.transport.Sinkhorn(epsilon, **options)(problem)
                else:
                    result = phx.transport.UnbalancedSinkhorn(epsilon, **options)(problem)
                return jnp.stack(
                    (
                        result.regularized_cost,
                        result.diagnostics.status.astype(jnp.float64),
                    )
                )

            tile = resolve_sinkhorn_block_size(
                solver.block_size,
                solver.sinkhorn_workspace_bytes,
                query_count,
                capacity,
                solver.epsilon.dtype.itemsize,
            )
            full_workspace = 4 * query_count * capacity * 8
            record = _measure(
                transport, solver.epsilon, repeats, compile_only=compile_only
            )
            record.update(
                family=family,
                requested_route=requested_route,
                requested_block_size=solver.block_size,
                route="dense" if tile is None else "blockwise",
                capacity=capacity,
                query_count=query_count,
                block_size=tile,
                sinkhorn_workspace_bytes=sinkhorn_workspace_bytes,
                full_pair_workspace_bytes=full_workspace,
                above_budget=full_workspace > sinkhorn_workspace_bytes,
                logical_pair_workspace_bytes=(
                    full_workspace if tile is None else 4 * tile * tile * 8
                ),
                retained_input_bytes=logical_array_bytes(problem),
            )
            rows.append(record)
        layer = MeasureAwareAttention(
            source_channels=16,
            query_channels=16,
            out_channels=16,
            num_heads=2,
            head_dim=8,
            execution="dense" if requested_route == "dense" else "auto",
            attention_workspace_bytes=attention_workspace_bytes,
        )
        bytes_per_source = q.shape[0] * layer.num_heads * query_count * q.dtype.itemsize
        full_workspace = bytes_per_source * capacity
        dense = (
            requested_route == "dense"
            or full_workspace <= layer.attention_workspace_bytes
        )
        tile = (
            None
            if dense
            else min(
                capacity, max(1, layer.attention_workspace_bytes // bytes_per_source)
            )
        )
        record = _measure(
            lambda query: layer.attend_projected(query, k, v, weights),
            q,
            repeats,
            compile_only=compile_only,
        )
        record.update(
            family="attention",
            requested_route=requested_route,
            requested_execution=layer.execution,
            requested_block_size=layer.block_size,
            route="dense" if tile is None else "blockwise",
            capacity=capacity,
            query_count=query_count,
            block_size=tile,
            attention_workspace_bytes=layer.attention_workspace_bytes,
            full_pair_workspace_bytes=full_workspace,
            above_budget=full_workspace > layer.attention_workspace_bytes,
            logical_pair_workspace_bytes=(
                full_workspace if tile is None else bytes_per_source * tile
            ),
            retained_input_bytes=logical_array_bytes((layer, q, k, v, weights)),
        )
        rows.append(record)
    return rows


def _family_gate(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    dense_rows = [
        row
        for row in rows
        if row["family"] == family and row["requested_route"] == "dense"
    ]
    default_rows = [
        row
        for row in rows
        if row["family"] == family and row["requested_route"] == "default"
    ]
    bounded_dense = [row for row in dense_rows if row["above_budget"]]
    bounded_default = [row for row in default_rows if row["above_budget"]]
    capacities = [row["capacity"] for row in bounded_dense]
    baseline = [row["compiler"]["temporary_bytes"] for row in bounded_dense]
    candidate = [row["compiler"]["temporary_bytes"] for row in bounded_default]
    return {
        "basis": "same-run current-code explicit dense versus automatic default above budget",
        "capacities": capacities,
        "memory": (
            evaluate_memory_gate(
                capacities, baseline, candidate, MemoryGatePolicy()
            ).to_dict()
            if capacities
            else None
        ),
        "baseline_exponent": (
            fitted_scaling_exponent(capacities, baseline) if len(capacities) > 1 else None
        ),
        "candidate_exponent": (
            fitted_scaling_exponent(capacities, candidate)
            if len(capacities) > 1
            else None
        ),
        "logical_scratch_formula": (
            "4 * itemsize * tile**2 plus linear marginal buffers; "
            "tile=floor(budget/(4*itemsize*max(n,m))) clamped to at least one"
            if family != "attention"
            else "B * H * Q * tile * itemsize plus linear input/output buffers"
        ),
        "full_fit_comparisons": [
            {
                "capacity": dense["capacity"],
                "expectation": "identical dense route; excluded from adoption gate",
                "dense_temporary_bytes": dense["compiler"]["temporary_bytes"],
                "default_temporary_bytes": default["compiler"]["temporary_bytes"],
                "equal_temporary_bytes": (
                    dense["compiler"]["temporary_bytes"]
                    == default["compiler"]["temporary_bytes"]
                ),
            }
            for dense, default in zip(dense_rows, default_rows, strict=True)
            if not default["above_budget"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument(
        "--compile-only-sizes", type=int, nargs="*", default=[8192, 16384, 32768, 65536]
    )
    parser.add_argument("--sinkhorn-workspace-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--attention-workspace-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    args = parser.parse_args()
    executed_sizes = [128, 256] if args.smoke else args.sizes
    compile_only_sizes = [] if args.smoke else args.compile_only_sizes
    sizes = sorted(set(executed_sizes + compile_only_sizes))
    rows: list[dict[str, Any]] = []
    for capacity in sizes:
        cases, peak = host_allocation_peak(
            lambda: _measure_case(
                capacity,
                args.repeats,
                compile_only=capacity not in executed_sizes,
                sinkhorn_workspace_bytes=args.sinkhorn_workspace_bytes,
                attention_workspace_bytes=args.attention_workspace_bytes,
            )
        )
        for case in cases:
            case["host_allocation_peak_bytes"] = peak
        rows.extend(cases)
    gates = {
        family: _family_gate(rows, family)
        for family in ("balanced", "unbalanced", "attention")
    }
    fields = ("environment", "configuration", "cases", "gates", "process_peak_rss_bytes")
    record = {
        "identity": capture_benchmark_identity(
            Path.cwd(), Path(__file__), fields
        ).to_dict(),
        "environment": asdict(capture_environment()),
        "configuration": {
            "capacities": sizes,
            "executed_capacities": executed_sizes,
            "compile_only_capacities": [
                size for size in sizes if size not in executed_sizes
            ],
            "sinkhorn_workspace_bytes": args.sinkhorn_workspace_bytes,
            "attention_workspace_bytes": args.attention_workspace_bytes,
            "candidate_selection": "public automatic defaults; no explicit tile",
            "gate_basis": "current-code same-run explicit dense comparison above budget",
            "historical_baseline": "memory_transport_attention_baseline.json unchanged",
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
