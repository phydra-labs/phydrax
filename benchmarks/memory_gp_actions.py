"""GP default and restricted-budget actions; above-budget compile-only gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    evaluate_memory_gate,
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
from phydrax.uq._gp_computation_backend import _kernel_matrix_action


def _dense_lanczos(points: jax.Array, state: Any) -> jax.Array:
    covariance = state.kernel.matrix(points, points) + jnp.diag(
        jnp.broadcast_to(state.noise_scale**2 + state.jitter, (points.shape[0],))
    )
    covariance = eqx.error_if(
        covariance,
        jnp.any(~jnp.isfinite(covariance)),
        "Iterative GP actions require finite covariance entries.",
    )
    vector = jnp.ones(points.shape[0], dtype=points.dtype)
    norm = jnp.linalg.vector_norm(vector)
    active = jnp.isfinite(norm) & (norm > 1e-10)
    vector = jnp.where(active, vector / jnp.where(active, norm, 1.0), 0.0)
    previous = jnp.zeros_like(vector)
    beta = jnp.zeros((), dtype=points.dtype)
    basis = []
    for _ in range(4):
        basis.append(jnp.where(active, vector, 0.0))
        candidate = covariance @ vector - beta * previous
        for column in basis:
            candidate = candidate - jnp.vdot(column, candidate).real * column
        next_norm = jnp.linalg.vector_norm(candidate)
        next_active = active & jnp.isfinite(next_norm) & (next_norm > 1e-10)
        next_vector = jnp.where(
            next_active, candidate / jnp.where(next_active, next_norm, 1.0), 0.0
        )
        previous, vector, beta, active = vector, next_vector, next_norm, next_active
    return jnp.stack(tuple(basis), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gate-sizes", type=int, nargs="+", default=[4096, 8192])
    args = parser.parse_args()
    sizes = [32, 64, 128] if args.smoke else [128, 256, 512]
    budget = phx.uq.GaussianProcessComputationPolicy().max_workspace_bytes
    records: list[dict[str, Any]] = []
    for n in sizes + args.gate_sizes:
        compile_only = n in args.gate_sizes
        if compile_only and n * (n + 1) * 8 <= budget:
            raise ValueError("Gate capacities must exceed the default workspace budget.")
        points = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)[:, None]
        state = phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.SquaredExponentialKernel(length_scale=0.4),
            noise_scale=0.05,
        )
        policy = phx.uq.LanczosGaussianProcessActionPolicy(
            jnp.ones(n, dtype=jnp.float64), max_actions=4
        )
        action_matrix = jnp.stack(
            (jnp.ones(n), points[:, 0], points[:, 0] ** 2, points[:, 0] ** 3), axis=1
        )
        record: dict[str, Any] = dict(n=n, compile_only=compile_only)
        for family in ("lanczos", "variational"):
            rows: dict[str, Any] = {}
            for route in ("dense", "default_policy", "restricted_budget"):
                workspace = (
                    budget
                    if route == "default_policy"
                    else (n + (1 if family == "lanczos" else 4)) * 8
                )

                def operation(x: jax.Array, matrix: jax.Array) -> jax.Array:
                    if family == "variational":
                        if route == "dense":
                            return state.kernel.matrix(x, x) @ matrix
                        return _kernel_matrix_action(
                            state.kernel,
                            x,
                            x,
                            phx.linalg.DenseLinearOperator(matrix),
                            max_workspace_bytes=workspace,
                            checkpoint=False,
                        )[0]
                    if route == "dense":
                        return _dense_lanczos(x, state)
                    resolved = policy.resolve(
                        x, state=state, max_workspace_bytes=workspace
                    )
                    return phx.linalg.materialize(
                        resolved.operator,
                        phx.linalg.MaterializationPolicy(
                            max_entries=n * 4, max_bytes=n * 4 * 8
                        ),
                    )

                executable, timing = measure_lower_and_compile(
                    lambda: jax.jit(operation).lower(points, action_matrix),
                    lambda lowered: lowered.compile(),
                )
                row: dict[str, Any] = dict(
                    route=route,
                    compiler=asdict(
                        compiler_evidence(
                            executable.cost_analysis(),
                            executable.memory_analysis(),
                            source="jax",
                        )
                    ),
                    compilation=asdict(timing),
                    workspace_budget_bytes=None if route == "dense" else workspace,
                    row_batch_size=n
                    if route == "dense"
                    else min(
                        n, workspace // ((n + (1 if family == "lanczos" else 4)) * 8)
                    ),
                )
                if not compile_only:
                    output, execution = measure_repeated(
                        lambda: executable(points, action_matrix), warmup=1, repeats=2
                    )
                    reference = (
                        _dense_lanczos(points, state)
                        if family == "lanczos"
                        else state.kernel.matrix(points, points) @ action_matrix
                    )
                    row.update(
                        execution=execution.to_dict(),
                        retained_bytes=logical_array_bytes(output),
                        parity_error=float(jnp.max(jnp.abs(output - reference))),
                    )
                rows[route] = row
                jax.clear_caches()
            record[family] = rows
        records.append(record)
    gates = [r for r in records if r["compile_only"]]
    result: dict[str, Any] = dict(
        records=records,
        default_workspace_bytes=budget,
        gate_basis="Current-code dense versus default_policy at compile-only capacities above the default budget; below-budget records are single-tile equivalence cases.",
        environment=capture_environment().to_dict(),
        identity=capture_benchmark_identity(
            Path(__file__).resolve().parents[1],
            Path(__file__),
            ("compiler", "retained_bytes"),
        ).to_dict(),
        peak_rss_bytes=process_peak_rss_bytes(),
    )
    for family, key in (("lanczos", "gate"), ("variational", "variational_gate")):
        result[key] = evaluate_memory_gate(
            [r["n"] for r in gates],
            [r[family]["dense"]["compiler"]["temporary_bytes"] for r in gates],
            [r[family]["default_policy"]["compiler"]["temporary_bytes"] for r in gates],
            MemoryGatePolicy(),
        ).to_dict()
    write_json_atomic(Path(__file__).with_suffix(".json"), result)


if __name__ == "__main__":
    main()
