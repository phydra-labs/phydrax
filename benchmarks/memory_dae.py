"""Fixed-grid DAE refresh and periodic-probe compiler-memory evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

import phydrax as phx
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


def residual(time: Array, state: Array, rate: Array, parameter: Array) -> Array:
    del time
    return rate + parameter * state


def run(*, baseline: bool, horizons: list[int], dimension: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        system = phx.dynamics.DifferentialAlgebraicSystem(
            residual,
            state_shape=(dimension,),
            structure=phx.dynamics.DAEStructure(
                tuple("differential" for _ in range(dimension))
            ),
            system_id="memory-dae",
        )
        grid = phx.dynamics.TimeGrid(
            jnp.linspace(0.0, horizon * 0.01, horizon + 1), time_id="memory-dae"
        )
        for reuse in (False, True):
            for batch in [None] if baseline else [None, 1, 4]:
                policy = phx.solver.DAESolvePolicy(
                    method=phx.solver.BDFMethod(1),
                    temporal_reuse=phx.solver.DAETemporalReusePolicy(enabled=reuse),
                    regularity=phx.solver.DAERegularityPolicy(
                        "periodic",
                        interval=2,
                        probe_batch_size=batch,
                    ),
                )
                problem = phx.solver.DifferentialAlgebraicProblem(
                    system,
                    jnp.ones(dimension, dtype=jnp.float64),
                    initial_state_rate=-jnp.ones(dimension, dtype=jnp.float64),
                    args=jnp.asarray(1.0),
                    problem_id="memory-dae",
                )
                prepared = phx.solver.prepare_dae(problem, grid, policy=policy)

                def operation(
                    parameter: Array,
                ) -> tuple[Array, ...]:
                    solution = phx.solver.solve_dae(prepared, args=parameter)
                    return (
                        solution.states,
                        solution.status,
                        solution.regularity.stage_status,
                        jnp.sum(solution.attempt_history.jacobian_preparations),
                        jnp.sum(solution.attempt_history.stale_jacobian_retries),
                        jnp.sum(solution.attempt_history.numeric_refreshes),
                        jnp.sum(solution.attempt_history.linear_solves),
                        solution.regularity.stage_rank,
                        solution.regularity.stage_condition_estimate,
                        solution.regularity.stage_valid,
                        solution.attempt_history.nonlinear_iterations,
                        solution.residual_norm,
                        solution.residual_threshold,
                        solution.regularity.consistency_status,
                        solution.regularity.consistency_rank,
                        solution.regularity.consistency_condition_estimate,
                    )

                compiled, timing = measure_lower_and_compile(
                    lambda: jax.jit(operation).lower(jnp.asarray(1.0)),
                    lambda lowered: lowered.compile(),
                )
                value, latency = measure_repeated(
                    lambda: compiled(jnp.asarray(1.0)), warmup=1, repeats=3
                )
                evidence = compiler_evidence(
                    compiled.cost_analysis(),
                    compiled.memory_analysis(),
                    source="jax-compiled",
                )
                rows.append(
                    {
                        "horizon": horizon,
                        "dimension": dimension,
                        "reuse": reuse,
                        "probe_batch_size": batch,
                        "compiler": asdict(evidence),
                        "lowering_seconds": timing.lowering_seconds,
                        "compilation_seconds": timing.compilation_seconds,
                        "warm_latency": latency.to_dict(),
                        "jacobian_preparations": int(value[3]),
                        "stale_retries": int(value[4]),
                        "numeric_refreshes": int(value[5]),
                        "linear_solves": int(value[6]),
                        "status": value[1].tolist(),
                        "regularity_status": value[2].tolist(),
                        "final_state": value[0][-1].tolist(),
                        "regularity_rank": value[7].tolist(),
                        "regularity_condition": [
                            float(condition) if valid else None
                            for condition, valid in zip(
                                value[8].tolist(), value[9].tolist(), strict=True
                            )
                        ],
                        "regularity_valid": value[9].tolist(),
                        "nonlinear_iterations": value[10].tolist(),
                        "residual_norm": value[11].tolist(),
                        "residual_threshold": value[12].tolist(),
                        "consistency_status": int(value[13]),
                        "consistency_rank": int(value[14]),
                        "consistency_condition": float(value[15]),
                        "logical_evidence_output_bytes": logical_array_bytes(value),
                    }
                )
    record = {
        "identity": capture_benchmark_identity(
            Path.cwd(), Path(__file__), ("rows", "gate")
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "rows": rows,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "probe_baseline": "Exact dev periodic probe vmap code path; temporal reuse is held fixed across probe routes.",
        "declared_probe_scratch": "O(batch_size * dimension**2 + horizon) bounded scratch and evidence archive",
    }
    if not baseline:
        dense = [
            row["compiler"]["temporary_bytes"]
            for row in rows
            if not row["reuse"] and row["probe_batch_size"] is None
        ]
        bounded = [
            row["compiler"]["temporary_bytes"]
            for row in rows
            if not row["reuse"] and row["probe_batch_size"] == 1
        ]
        record["gate"] = evaluate_memory_gate(
            horizons, dense, bounded, MemoryGatePolicy()
        ).to_dict()
        record["probe_scaling_exponent"] = fitted_scaling_exponent(horizons, bounded)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[8, 16])
    args = parser.parse_args()
    write_json_atomic(
        Path(
            "benchmarks/memory_dae_baseline.json"
            if args.baseline
            else "benchmarks/memory_dae.json"
        ),
        run(
            baseline=args.baseline,
            horizons=[2, 4] if args.smoke else args.horizons,
            dimension=2 if args.smoke else args.dimension,
        ),
    )


if __name__ == "__main__":
    main()
