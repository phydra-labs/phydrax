"""Causal implicit derivative workspace and same-run route latency evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._memory import fitted_scaling_exponent, process_peak_rss_bytes
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax import nonlinear as nl
from phydrax.nonlinear._causal_adjoint import attach_causal_implicit_derivative


def transition(parameter: Array, previous: Array, driver: Array) -> Array:
    return jnp.tanh(parameter @ previous + driver)


def run(*, baseline: bool, horizons: list[int], dimension: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    routes: list[nl.CausalAdjointRoute] = (
        ["dense-associative"]
        if baseline
        else ["dense-associative", "action-scan", "checkpointed-action-scan"]
    )
    for horizon in horizons:
        matrix = 0.4 * jnp.eye(dimension, dtype=jnp.float64) + 0.01 / dimension
        initial = jnp.zeros(dimension, dtype=jnp.float64)
        drivers = jnp.full((horizon, dimension), 0.1, dtype=jnp.float64)

        def step(state: Array, driver: Array) -> tuple[Array, Array]:
            value = transition(matrix, state, driver)
            return value, value

        trajectory = jax.lax.scan(step, initial, drivers)[1]
        certified = nl.solve_causal_recurrence(
            nl.CausalRecurrenceProblem(transition, initial, drivers, parameters=matrix),
            initial_trajectory=trajectory,
            method=nl.CausalNewton(),
        )
        if not bool(certified.successful):
            raise RuntimeError("The independently generated primal must certify.")
        for route in routes:

            def objective(parameter: Array) -> Array:
                problem = nl.CausalRecurrenceProblem(
                    transition, initial, drivers, parameters=parameter
                )
                result = (
                    nl.solve_causal_recurrence(
                        problem,
                        initial_trajectory=trajectory,
                        method=nl.CausalNewton(),
                    )
                    if baseline
                    else attach_causal_implicit_derivative(
                        problem,
                        certified,
                        nl.CausalAdjointPolicy(route, checkpoint_block_size=8),
                    )
                )
                return jnp.sum(result.states)

            operation = jax.jit(jax.grad(objective))
            compiled, timing = measure_lower_and_compile(
                lambda: operation.lower(matrix), lambda lowered: lowered.compile()
            )
            value, latency = measure_repeated(
                lambda: compiled(matrix), warmup=1, repeats=3
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
                    "route": route,
                    "lowering_seconds": timing.lowering_seconds,
                    "compilation_seconds": timing.compilation_seconds,
                    "warm_latency": latency.to_dict(),
                    "compiler": asdict(evidence),
                    "logical_transition_bytes": horizon
                    * dimension
                    * dimension
                    * matrix.dtype.itemsize
                    if route == "dense-associative"
                    else 0,
                    "logical_trajectory_bytes": horizon
                    * dimension
                    * matrix.dtype.itemsize,
                    "logical_action_carry_bytes": dimension * matrix.dtype.itemsize,
                    "logical_output_bytes": logical_array_bytes(value),
                    "gradient_norm": float(jnp.linalg.norm(value)),
                }
            )
    return {
        "identity": capture_benchmark_identity(
            Path.cwd(), Path(__file__), ("rows", "scaling")
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "rows": rows,
        "measurement": "end-to-end primal and derivative"
        if baseline
        else "derivative attachment only; primal independently generated and certified outside the measured executable",
        "comparison_baseline": "The dense-associative row in this same run is the preserved exact dev adjoint code path; the untouched baseline artifact additionally records the full original solve.",
        "scaling": {
            route: fitted_scaling_exponent(
                horizons,
                [
                    row["compiler"]["temporary_bytes"]
                    for row in rows
                    if row["route"] == route
                ],
            )
            for route in routes
        },
        "crossover": [
            {
                "horizon": horizon,
                "fastest_route": min(
                    (row for row in rows if row["horizon"] == horizon),
                    key=lambda row: row["warm_latency"]["median_seconds"],
                )["route"],
            }
            for horizon in horizons
        ],
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 32, 64])
    args = parser.parse_args()
    record = run(
        baseline=args.baseline,
        horizons=[4, 8] if args.smoke else args.horizons,
        dimension=4 if args.smoke else args.dimension,
    )
    write_json_atomic(
        Path(
            "benchmarks/memory_causal_adjoint_baseline.json"
            if args.baseline
            else "benchmarks/memory_causal_adjoint.json"
        ),
        record,
    )


if __name__ == "__main__":
    main()
