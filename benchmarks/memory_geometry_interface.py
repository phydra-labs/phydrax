#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reproduce G4's budget-derived interface workspace gate.

Small cases execute and compare against the frozen source-only reference.
Above-budget gate cases compile the current default and explicit full-tile
routes without execution. Committed baseline records remain untouched;
timing qualification is deferred to the quiet serialized wave.
Compiler temporaries exclude arguments and outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.ein as ein
from phydrax.geometry._interface import _directed_nearest_distances, _interface_tiles
from phydrax.typing import parse

from ._io import write_json_atomic
from ._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    MemoryGatePolicy,
    process_peak_rss_bytes,
)
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    DurationDistribution,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


GeometryInterfaceBenchmarkPhase: TypeAlias = Literal["baseline", "candidate"]


def baseline_kernel(
    source: Array, target: Array, target_mask: Array, *, chunk_size: int
) -> Array:
    """Frozen untouched source-only tiling for reproducible comparisons."""
    target_square = jnp.sum(target * target, axis=-1)
    blocks = []
    for start in range(0, source.shape[-2], chunk_size):
        stop = min(start + chunk_size, source.shape[-2])
        block = source[..., start:stop, :]
        block_square = jnp.sum(block * block, axis=-1, keepdims=True)
        cross = ein.contract("...id,...jd->...ij", block, target)
        squared = jnp.maximum(
            block_square + target_square[..., None, :] - 2.0 * cross, 0.0
        )
        squared = jnp.where(target_mask[..., None, :], squared, jnp.inf)
        blocks.append(jnp.sqrt(jnp.min(squared, axis=-1)))
    return jnp.concatenate(blocks, axis=-1)


def _paired_samples(
    candidate: Callable[[], Array],
    baseline: Callable[[], Array],
    repeats: int,
) -> tuple[Array, DurationDistribution, Array, DurationDistribution]:
    result, _ = measure_repeated(candidate, warmup=1, repeats=1)
    reference, _ = measure_repeated(baseline, warmup=1, repeats=1)
    candidate_samples: list[float] = []
    baseline_samples: list[float] = []
    # Alternating order prevents measuring each route in a different time window.
    for repeat in range(repeats):
        if repeat % 2 == 0:
            result, candidate_time = measure_repeated(candidate, warmup=0, repeats=1)
            reference, baseline_time = measure_repeated(baseline, warmup=0, repeats=1)
        else:
            reference, baseline_time = measure_repeated(baseline, warmup=0, repeats=1)
            result, candidate_time = measure_repeated(candidate, warmup=0, repeats=1)
        candidate_samples.extend(candidate_time.samples_seconds)
        baseline_samples.extend(baseline_time.samples_seconds)
    return (
        result,
        DurationDistribution(tuple(candidate_samples)),
        reference,
        DurationDistribution(tuple(baseline_samples)),
    )


def _compile_gate_case(source: Array, capacity: int) -> dict[str, Any]:
    target = jnp.arange(capacity * 3, dtype=jnp.float64).reshape(capacity, 3)
    mask = jnp.ones((capacity,), dtype=jnp.bool_)
    kernel = jax.jit(
        _directed_nearest_distances, static_argnames=("chunk_size", "target_chunk_size")
    )
    row: dict[str, Any] = {"capacity": capacity}
    for name, source_tile, target_tile in (
        ("default", None, None),
        ("explicit_full_tile", source.shape[-2], capacity),
    ):
        compiled, timing = measure_lower_and_compile(
            lambda: kernel.lower(
                source,
                target,
                mask,
                chunk_size=source_tile,
                target_chunk_size=target_tile,
            ),
            lambda lowered: lowered.compile(),
        )
        row[name] = {
            "compiler": asdict(
                compiler_evidence(
                    compiled.cost_analysis(),
                    compiled.memory_analysis(),
                    source="jax-compiled-interface-current-route",
                )
            ),
            "lowering_seconds": timing.lowering_seconds,
            "compilation_seconds": timing.compilation_seconds,
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=get_args(GeometryInterfaceBenchmarkPhase), required=True
    )
    parser.add_argument("--targets", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--sources", type=int, default=256)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--gate-targets", nargs="+", type=int, default=[8192, 16384])
    parser.add_argument("--target-chunk-size", type=int)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    args.phase = parse(args.phase, GeometryInterfaceBenchmarkPhase, "phase")
    if args.smoke:
        args.targets = [64, 128]
        args.sources = 32
    if min(args.sources, args.repeats) <= 0 or (
        args.chunk_size is not None and args.chunk_size <= 0
    ):
        parser.error("Counts, tiles, and repeats must be positive.")
    if args.target_chunk_size is not None and args.target_chunk_size <= 0:
        parser.error("Target tile must be positive.")
    if len(args.targets) < 2 or any(
        a >= b for a, b in zip(args.targets, args.targets[1:])
    ):
        parser.error("Target capacities must be positive and strictly increasing.")
    if args.targets[0] <= 0:
        parser.error("Target capacities must be positive.")
    root = Path(__file__).resolve().parents[1]
    baseline_path = (
        args.baseline or root / "benchmarks/memory_geometry_interface_baseline.json"
    )
    output = (
        args.output
        or root
        / f"benchmarks/memory_geometry_interface{'_baseline' if args.phase == 'baseline' else ''}.json"
    )
    rng = np.random.default_rng(715)
    source = jnp.asarray(rng.normal(size=(args.sources, 3)), dtype=jnp.float64)
    source_tile = min(args.chunk_size or args.sources, args.sources)
    if args.phase == "candidate" and args.target_chunk_size is None:
        source_tile = min(source_tile, max(1, (2 * 1024 * 1024) // source.dtype.itemsize))
    cases: list[dict[str, Any]] = []
    for capacity in args.targets:
        target = jnp.asarray(rng.normal(size=(capacity, 3)), dtype=jnp.float64)
        mask = jnp.asarray(np.arange(capacity) % 7 != 0, dtype=jnp.bool_)
        if args.phase == "baseline":
            kernel = jax.jit(baseline_kernel, static_argnames=("chunk_size",))
            keywords = {"chunk_size": args.chunk_size or args.sources}
        else:
            kernel = jax.jit(
                _directed_nearest_distances,
                static_argnames=("chunk_size", "target_chunk_size"),
            )
            keywords = {
                "chunk_size": args.chunk_size,
                "target_chunk_size": args.target_chunk_size,
            }
        executable, timing = measure_lower_and_compile(
            lambda: kernel.lower(source, target, mask, **keywords),
            lambda lowered: lowered.compile(),
        )
        evidence = compiler_evidence(
            executable.cost_analysis(),
            executable.memory_analysis(),
            source="jax-compiled-directed-interface",
        )
        baseline_executable, baseline_timing = measure_lower_and_compile(
            lambda: jax.jit(baseline_kernel, static_argnames=("chunk_size",)).lower(
                source, target, mask, chunk_size=args.chunk_size or args.sources
            ),
            lambda lowered: lowered.compile(),
        )
        result, warm, reference, baseline_warm = _paired_samples(
            lambda: executable(source, target, mask),
            lambda: baseline_executable(source, target, mask),
            args.repeats,
        )
        baseline_evidence = compiler_evidence(
            baseline_executable.cost_analysis(),
            baseline_executable.memory_analysis(),
            source="jax-compiled-source-only-interface",
        )
        np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)
        executed_tiles = (
            (source_tile, capacity)
            if args.phase == "baseline"
            else _interface_tiles(
                source,
                target,
                args.chunk_size,
                args.target_chunk_size,
                2 * 1024 * 1024,
            )
        )
        cases.append(
            {
                "capacity": capacity,
                "source_count": args.sources,
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "warm": warm.to_seconds_dict(),
                "compiler": asdict(evidence),
                "required_output_bytes": logical_array_bytes(result),
                "logical_pair_tile_bytes": executed_tiles[0]
                * executed_tiles[1]
                * source.dtype.itemsize,
                "executed_tiles": executed_tiles,
                "full_fit_comparison": (
                    {
                        "expectation": "identical dense route; excluded from adoption gate",
                        "equal_temporary_bytes": evidence.temporary_bytes
                        == baseline_evidence.temporary_bytes,
                    }
                    if executed_tiles == (args.sources, capacity)
                    else None
                ),
                "reference_max_abs_error": float(jnp.max(jnp.abs(result - reference))),
                "paired_baseline": {
                    "compiler": asdict(baseline_evidence),
                    "warm": baseline_warm.to_seconds_dict(),
                    "lowering_seconds": baseline_timing.lowering_seconds,
                    "compilation_seconds": baseline_timing.compilation_seconds,
                },
            }
        )
        jax.clear_caches()
    configuration = {
        "sources": args.sources,
        "targets": args.targets,
        "chunk_size": args.chunk_size,
        "target_chunk_size": args.target_chunk_size,
        "repeats": args.repeats,
        "interface_workspace_bytes": 2 * 1024 * 1024,
    }
    record: dict[str, Any] = {
        "item": "G4",
        "phase": args.phase,
        "configuration": configuration,
        "environment": capture_environment().to_dict(),
        "command": sys.argv,
        "cases": cases,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "identity": capture_benchmark_identity(
            root,
            Path(__file__).resolve(),
            ("cases", "configuration", "environment", "gate", "timing_acceptance"),
        ).to_dict(),
        "temporary_scaling_exponent": fitted_scaling_exponent(
            args.targets, [row["compiler"]["temporary_bytes"] for row in cases]
        ),
    }
    if args.phase == "candidate":
        baseline = json.loads(baseline_path.read_text())
        record["captured_baseline_configuration"] = baseline["configuration"]
        gate_cases = [_compile_gate_case(source, size) for size in args.gate_targets]
        temporary = [row["default"]["compiler"]["temporary_bytes"] for row in gate_cases]
        gate = evaluate_memory_gate(
            args.gate_targets,
            [
                row["explicit_full_tile"]["compiler"]["temporary_bytes"]
                for row in gate_cases
            ],
            temporary,
            MemoryGatePolicy(),
        ).to_dict()
        scaling_ratios = [b / a for a, b in zip(temporary, temporary[1:])]
        scaling_passes = all(ratio <= 1.25 for ratio in scaling_ratios)
        record["gate"] = gate
        record["gate_cases"] = gate_cases
        record["gate_basis"] = (
            "Compile-only current default versus explicit full tile above the workspace budget; committed baseline is unchanged."
        )
        record["fixed_tile_temporary_ratios"] = scaling_ratios
        record["fixed_tile_scaling_acceptance"] = scaling_passes
        record["decision"] = (
            "adopt"
            if gate["decision"] == "adopt" and scaling_passes
            else "retain-baseline"
        )
        record["timing_pending_quiet_wave"] = True
    write_json_atomic(output, record)
    print(output)


if __name__ == "__main__":
    main()
