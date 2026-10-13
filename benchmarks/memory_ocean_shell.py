"""Compile-only gates for ocean nearest lookup and the NR boundary shell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    MemoryGatePolicy,
)
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    measure_lower_and_compile,
)


def nearest(nodes: Array, values: Array, period: Array | None = None) -> Array:
    """Adjacent candidates for the original one-period, not modulo, distance."""
    queries = (values,) if period is None else (values, values - period, values + period)
    best = jnp.zeros(values.shape, dtype=jnp.int32)
    distance = jnp.full(values.shape, jnp.inf, dtype=values.dtype)
    for query in queries:
        upper = jnp.searchsorted(nodes, query, side="left")
        for index in (
            jnp.clip(upper - 1, 0, nodes.size - 1),
            jnp.clip(upper, 0, nodes.size - 1),
        ):
            delta = jnp.abs(values - nodes[index])
            if period is not None:
                delta = jnp.minimum(delta, jnp.abs(period - delta))
            update = (delta < distance) | ((delta == distance) & (index < best))
            best = jnp.where(update, index, best)
            distance = jnp.where(update, delta, distance)
    return best


def dense(nodes: Array, values: Array, period: Array) -> Array:
    delta = jnp.abs(values[:, None] - nodes[None, :])
    return jnp.argmin(jnp.minimum(delta, jnp.abs(period - delta)), axis=1)


def shell(size: int, candidate: bool) -> Array:
    indices = tuple(jnp.arange(size, dtype=jnp.int32) for _ in range(3))
    if not candidate:
        indices = tuple(jnp.meshgrid(*indices, indexing="ij"))
        return (
            (indices[0] < 1)
            | (indices[0] >= size - 1)
            | (indices[1] < 1)
            | (indices[1] >= size - 1)
            | (indices[2] < 1)
            | (indices[2] >= size - 1)
        )
    predicates = tuple((index < 1) | (index >= size - 1) for index in indices)
    return (
        predicates[0][:, None, None]
        | predicates[1][None, :, None]
        | predicates[2][None, None, :]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    root = Path(__file__).parent
    capacities = [32, 128] if args.smoke else [32, 128, 512, 2048]
    shell_capacities = [8, 16] if args.smoke else [8, 16, 32, 64]
    records: dict[str, list[dict[str, object]]] = {"ocean": [], "shell": []}
    temporary_bytes: dict[str, list[int]] = {"ocean": [], "shell": []}
    for capacity_index, size in enumerate(capacities):
        nodes = jnp.linspace(-3.0, 3.0, size, dtype=jnp.float64)
        values = jnp.linspace(-12.0, 12.0, size // 2, dtype=jnp.float64)
        for area in records:
            if area == "ocean":
                operation = jax.jit(dense if args.baseline else nearest)
                lower = lambda: operation.lower(nodes, values, jnp.asarray(6.0))
            else:
                shell_size = shell_capacities[capacity_index]
                operation = jax.jit(lambda: shell(shell_size, not args.baseline))
                lower = operation.lower
            compiled, timing = measure_lower_and_compile(
                lower, lambda lowered: lowered.compile()
            )
            evidence = compiler_evidence(
                compiled.cost_analysis(),
                compiled.memory_analysis(),
                source="Compiled.memory_analysis",
            )
            if evidence.temporary_bytes is None:
                raise RuntimeError("Compiler temporary-byte evidence is unavailable.")
            temporary_bytes[area].append(evidence.temporary_bytes)
            records[area].append(
                {
                    "capacity": size
                    if area == "ocean"
                    else shell_capacities[capacity_index],
                    "temporary_bytes": evidence.temporary_bytes,
                    "argument_bytes": evidence.argument_bytes,
                    "output_bytes": evidence.output_bytes,
                    "lowering_seconds": timing.lowering_seconds,
                    "compilation_seconds": timing.compilation_seconds,
                }
            )
    result: dict[str, object] = {
        "environment": capture_environment().to_dict(),
        "records": records,
    }
    if not args.baseline:
        baseline = json.loads((root / "memory_ocean_shell_baseline.json").read_text())
        result["gates"] = {
            area: evaluate_memory_gate(
                capacities if area == "ocean" else shell_capacities,
                [
                    {
                        row["capacity"]: row["temporary_bytes"]
                        for row in baseline["records"][area]
                    }[size]
                    for size in (capacities if area == "ocean" else shell_capacities)
                ],
                temporary_bytes[area],
                MemoryGatePolicy(),
            ).to_dict()
            for area in records
        }
        result["logical_workspace_exponent"] = fitted_scaling_exponent(
            capacities, [24 * (size // 2) for size in capacities]
        )
    write_json_atomic(
        args.output
        or root
        / (
            "memory_ocean_shell_baseline.json"
            if args.baseline
            else "memory_ocean_shell.json"
        ),
        result,
    )


if __name__ == "__main__":
    main()
