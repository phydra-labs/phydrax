"""Phase-separated exact gravity pair workspace and adoption evidence."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
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


def context() -> Any:
    astro = phx.applications.astrodynamics
    return astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0), "TT")),
        astro.FrameDefinition("barycenter", "icrf", pseudo_inertial=True),
    )


def run(
    route: str, count: int, candidate: bool, steps: int = 0, *, compile_only: bool = False
) -> dict[str, Any]:
    astro = phx.applications.astrodynamics
    positions = jnp.asarray(
        np.random.default_rng(count).normal(size=(count, 3)), dtype=jnp.float64
    )
    masses = jnp.linspace(1.0e-4, 2.0e-4, count)
    start = time.perf_counter()
    if route == "direct":
        particles = phx.discretization.particle.ParticleSetPlan(
            jnp.arange(count), masses, ambient_dimension=3
        ).prepare()
        plan, peak = host_allocation_peak(
            lambda: astro.DirectNBodyGravityPlan(
                particles,
                context(),
                softening=0.01,
                target_chunk_size=None if candidate else count,
            )
        )
        operation = plan.evaluate
    else:
        plan, peak = host_allocation_peak(
            lambda: astro.NearlyKeplerianPlan(
                1.0,
                masses,
                jnp.linspace(0.0, 0.01, max(steps, 1) + 1),
                context(),
                target_chunk_size=None if candidate else count,
            )
        )
        operation = (
            plan._perturbation
            if steps == 0
            else lambda p: plan.rollout(
                astro.NearlyKeplerianState(p, jnp.zeros_like(p), plan.context)
            )
        )
    preparation = time.perf_counter() - start
    compiled, timing = measure_lower_and_compile(
        lambda: jax.jit(lambda p: operation(p)).lower(positions),
        lambda lowered: lowered.compile(),
    )
    if compile_only:
        result, warm = None, None
    else:
        result, warm = measure_repeated(lambda: compiled(positions), warmup=1, repeats=3)
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="Compiled.memory_analysis",
    )
    defect = None
    if not steps and result is not None:
        p = np.asarray(positions)
        m = np.asarray(masses)
        delta = p[None, :, :] - p[:, None, :]
        squared = np.sum(delta * delta, axis=-1)
        mask = ~np.eye(count, dtype=np.bool_)
        safe = np.where(mask, squared + (0.0001 if route == "direct" else 0.0), 1.0)
        acceleration = np.sum(
            m[None, :, None] * delta * np.where(mask, safe**-1.5, 0.0)[..., None], axis=1
        )
        if route != "direct":
            acceleration -= np.sum(
                m[:, None] * p / np.linalg.norm(p, axis=1)[:, None] ** 3, axis=0
            )
        actual = result.acceleration if route == "direct" else result[0]
        defect = float(np.max(np.abs(np.asarray(actual) - acceleration)))
    return {
        "route": route,
        "capacity": count,
        "steps": steps,
        "preparation_seconds": preparation,
        "host_peak_bytes": peak,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": None if warm is None else warm.to_dict(),
        "compiler": asdict(evidence),
        "retained_bytes": logical_array_bytes(plan),
        "logical_pair_workspace_bytes": 8
        * 6
        * count
        * (max(1, min(count, (4 * 1024 * 1024) // (48 * count))) if candidate else count),
        "reference_defect": defect,
        "valid": None
        if result is None
        else bool(result.valid)
        if route == "direct"
        else bool(result[2])
        if not steps
        else bool(result.successful),
        "compile_only": compile_only,
        "pair_workspace_bytes": 4 * 1024 * 1024,
        "plan_id": plan.plan_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    sizes = [32, 64] if args.smoke else [32, 64, 128, 256]
    near_sizes = [4, 8] if args.smoke else [4, 8, 16, 32]
    rows = [
        run(route, n, args.candidate)
        for route, capacities in (("direct", sizes), ("near", near_sizes))
        for n in capacities
    ]
    rows += [
        run("near", near_sizes[0], args.candidate, steps)
        for steps in ([4] if args.smoke else [64, 256])
    ]
    # Above-budget comparisons use the current parallel route; historical
    # baseline records remain untouched and below-budget rows are expected equal.
    gate_rows = [
        run(route, n, args.candidate, compile_only=True)
        for route in ("direct", "near")
        for n in (1024, 2048, 4096)
    ]
    comparison_rows = [
        run(route, n, False, compile_only=True)
        for route in ("direct", "near")
        for n in (1024, 2048, 4096)
    ]
    fitting_comparisons = [
        run(row["route"], row["capacity"], False, compile_only=True)
        for row in rows
        if not row["steps"]
    ]
    record = {
        "candidate": args.candidate,
        "rows": rows,
        "above_budget_rows": gate_rows,
        "parallel_comparison_rows": comparison_rows,
        "fitting_parallel_comparison_rows": fitting_comparisons,
        "fitting_memory_equal": [
            row["compiler"]["temporary_bytes"]
            == comparison["compiler"]["temporary_bytes"]
            for row, comparison in zip(
                (row for row in rows if not row["steps"]),
                fitting_comparisons,
                strict=True,
            )
        ],
        "historical_baseline": None if args.baseline is None else str(args.baseline),
        "gate_basis": "same-run current default versus explicit full-tile parallel, compile-only above budget",
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            Path.cwd(), Path(__file__), rows[0].keys()
        ).to_dict(),
        "rss_bytes": process_peak_rss_bytes(),
        "gates": {},
    }
    if args.baseline:
        # The committed historical baseline is provenance, not the gate basis.
        for route in ("direct", "near"):
            selected = [row for row in gate_rows if row["route"] == route]
            old = [row for row in comparison_rows if row["route"] == route]
            gate = evaluate_memory_gate(
                [r["capacity"] for r in selected],
                [r["compiler"]["temporary_bytes"] for r in old],
                [r["compiler"]["temporary_bytes"] for r in selected],
                MemoryGatePolicy(),
            ).to_dict()
            gate["logical_scaling_exponent"] = fitted_scaling_exponent(
                [r["capacity"] for r in selected],
                [r["logical_pair_workspace_bytes"] for r in selected],
            )
            gate["temporary_scaling_exponent"] = fitted_scaling_exponent(
                [r["capacity"] for r in selected],
                [r["compiler"]["temporary_bytes"] for r in selected],
            )
            gate["doubling_ratios"] = [
                b["logical_pair_workspace_bytes"] / a["logical_pair_workspace_bytes"]
                for a, b in zip(selected, selected[1:])
            ]
            record["gates"][route] = gate
    write_json_atomic(args.output, record)
    print(json.dumps(record["gates"]))


if __name__ == "__main__":
    main()
