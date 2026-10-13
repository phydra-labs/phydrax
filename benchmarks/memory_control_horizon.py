"""Phase-separated iLQR workspace and control-QP horizon scaling."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

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
from phydrax.control._ilqr import _local_model
from phydrax.control._prepared_ilqr import _backward


def problem(horizon: int, dimension: int) -> phx.control.ControlProblem:
    def transition(
        context: phx.dynamics.DiscreteStepContext,
        state: jax.Array,
        control: jax.Array,
        args: Any,
    ) -> jax.Array:
        return 0.9 * state + 0.04 * jnp.sin(state) + control

    def running(
        time: jax.Array, state: jax.Array, control: jax.Array, args: Any
    ) -> jax.Array:
        return 0.5 * (jnp.sum(state**2) + jnp.sum(control**2))

    def terminal(time: jax.Array, state: jax.Array, args: Any) -> jax.Array:
        return jnp.sum(state**2)

    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((dimension,)),
        input_layout=phx.dynamics.InputLayout((dimension,)),
        system_id="memory-control:system",
    )
    return phx.control.ControlProblem(
        phx.control.DiscreteControlDynamics(system),
        phx.dynamics.TimeGrid(
            jnp.arange(horizon + 1, dtype=jnp.float64), time_id="memory-control:time"
        ),
        jnp.ones((dimension,), dtype=jnp.float64),
        running_cost=running,
        terminal_cost=terminal,
        problem_id="memory-control:problem",
    )


def run(horizons: list[int], dimension: int, baseline: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    routes = (
        ["batched"]
        if baseline
        else ["batched", "reverse-scan", "checkpointed-reverse-scan"]
    )
    for horizon in horizons:
        p = problem(horizon, dimension)
        states = jnp.ones((horizon + 1, dimension), dtype=jnp.float64)
        controls = jnp.zeros((horizon, dimension), dtype=jnp.float64)
        for route in routes:
            if route == "batched":

                def kernel(x: jax.Array, u: jax.Array) -> Any:
                    model = _local_model(p, x, u, None)
                    return _backward(model, 1e-6), model.control_gradient
            else:
                from phydrax.control._prepared_ilqr import _backward_scan

                def kernel(x: jax.Array, u: jax.Array) -> Any:
                    return _backward_scan(
                        p, x, u, None, 1e-6, route == "checkpointed-reverse-scan"
                    )

            compiled, timing = measure_lower_and_compile(
                lambda: jax.jit(kernel).lower(states, controls),
                lambda lowered: lowered.compile(),
            )
            result, warm = measure_repeated(
                lambda: compiled(states, controls), warmup=1, repeats=3
            )
            evidence = compiler_evidence(
                compiled.cost_analysis(),
                compiled.memory_analysis(),
                source="jax-compiled-memory-analysis",
            )
            rows.append(
                {
                    "item": "S10",
                    "horizon": horizon,
                    "dimension": dimension,
                    "route": route,
                    "compiler": asdict(evidence),
                    "timing": asdict(timing),
                    "warm": warm.to_dict(),
                    "output_bytes": logical_array_bytes(result),
                    "backward_valid": bool(jax.device_get(result[0][5])),
                    "failed_step": int(jax.device_get(result[0][6])),
                    "gradient_norm": float(jax.device_get(jnp.linalg.norm(result[1]))),
                    "logical_stage_model_bytes": horizon
                    * (5 * dimension**2 + 2 * dimension)
                    * 8
                    if route == "batched"
                    else (5 * dimension**2 + 2 * dimension) * 8,
                }
            )
        qp_dimension = min(dimension, 2)
        eye = jnp.eye(qp_dimension, dtype=jnp.float64)
        blocks = jnp.broadcast_to(eye, (horizon, qp_dimension, qp_dimension))
        specification = phx.control.LinearQuadraticControlProblem(
            blocks,
            blocks,
            jnp.ones((qp_dimension,), dtype=jnp.float64),
            blocks,
            blocks,
            eye,
            problem_id="memory-control:qp",
        )
        for route in ["dense", "sparse"]:
            policy = phx.control.LinearControlCompilationPolicy(route)
            compilation, peak = host_allocation_peak(
                lambda: phx.control.compile_linear_quadratic_control(
                    specification, compilation_policy=policy
                )
            )
            _, warm = measure_repeated(
                lambda: phx.control.compile_linear_quadratic_control(
                    specification, compilation_policy=policy
                ),
                warmup=1,
                repeats=3,
            )
            rows.append(
                {
                    "item": "S11",
                    "horizon": horizon,
                    "dimension": qp_dimension,
                    "route": route,
                    "host_peak_bytes": peak,
                    "retained_bytes": logical_array_bytes(compilation),
                    "warm": warm.to_dict(),
                }
            )
    gates: dict[str, Any] = {}
    crossover: list[dict[str, Any]] = []
    if not baseline:
        for route in routes[1:]:
            base = [
                r["compiler"]["temporary_bytes"]
                for r in rows
                if r["item"] == "S10" and r["route"] == "batched"
            ]
            candidate = [
                r["compiler"]["temporary_bytes"]
                for r in rows
                if r["item"] == "S10" and r["route"] == route
            ]
            gates[route] = {
                **evaluate_memory_gate(
                    horizons, base, candidate, MemoryGatePolicy()
                ).to_dict(),
                "fitted_exponent": fitted_scaling_exponent(horizons, candidate),
                "declared_workspace_formula": "constant stage scratch plus linear required outputs",
                "successive_capacity_ratios": [
                    candidate[i] / candidate[i - 1] for i in range(1, len(candidate))
                ],
                "linear_scaling_admitted": all(
                    candidate[i] / candidate[i - 1]
                    <= 1.25 * horizons[i] / horizons[i - 1]
                    for i in range(1, len(candidate))
                ),
            }
        for horizon in horizons:
            choices = [r for r in rows if r["item"] == "S10" and r["horizon"] == horizon]
            crossover.append(
                {
                    "horizon": horizon,
                    "lowest_latency_route": min(
                        choices, key=lambda r: r["warm"]["median_seconds"]
                    )["route"],
                }
            )
    root = Path(__file__).resolve().parents[1]
    return {
        "identity": capture_benchmark_identity(
            root, Path(__file__), ("records", "gates", "crossover")
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "records": rows,
        "gates": gates,
        "crossover": crossover,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--dimension", type=int, default=8)
    args = parser.parse_args()
    horizons = [4, 8] if args.smoke else args.horizons
    result = run(horizons, 2 if args.smoke else args.dimension, args.baseline)
    destination = Path(__file__).with_name(
        "memory_control_horizon_baseline.json"
        if args.baseline
        else "memory_control_horizon.json"
    )
    write_json_atomic(destination, result)
    print(destination)


if __name__ == "__main__":
    main()
