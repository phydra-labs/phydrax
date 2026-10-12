"""Fixed-capacity GP action compiler memory scaling (small default)."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sizes = [32, 64, 128] if args.smoke else [128, 256, 512]
    records: list[dict[str, Any]] = []
    for n in sizes:
        points = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)[:, None]
        state = phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.SquaredExponentialKernel(length_scale=0.4),
            noise_scale=0.05,
        )
        policy = phx.uq.LanczosGaussianProcessActionPolicy(
            jnp.ones(n, dtype=jnp.float64), max_actions=4
        )

        def run(x: jax.Array) -> jax.Array:
            if args.baseline:
                return _dense_lanczos(x, state)
            resolved = policy.resolve(x, state=state, max_workspace_bytes=(n + 1) * 8)
            return phx.linalg.materialize(
                resolved.operator,
                phx.linalg.MaterializationPolicy(max_entries=n * 4, max_bytes=n * 4 * 8),
            )

        compiled, timing = measure_lower_and_compile(
            lambda: jax.jit(run).lower(points), lambda lowered: lowered.compile()
        )
        result, samples = measure_repeated(lambda: compiled(points), warmup=1, repeats=2)
        evidence = asdict(
            compiler_evidence(
                compiled.cost_analysis(), compiled.memory_analysis(), source="jax"
            )
        )
        parity_error = float(jnp.max(jnp.abs(result - _dense_lanczos(points, state))))
        action_matrix = jnp.stack(
            (jnp.ones(n), points[:, 0], points[:, 0] ** 2, points[:, 0] ** 3),
            axis=1,
        )

        def variational_dense(x: jax.Array, matrix: jax.Array) -> jax.Array:
            return state.kernel.matrix(x, x) @ matrix

        def variational_bounded(x: jax.Array, matrix: jax.Array) -> jax.Array:
            return _kernel_matrix_action(
                state.kernel,
                x,
                x,
                phx.linalg.DenseLinearOperator(matrix),
                max_workspace_bytes=(n + 4) * 8,
                checkpoint=False,
            )[0]

        variational_evidence = {}
        for name, operation in (
            ("dense", variational_dense),
            ("bounded", variational_bounded),
        ):
            executable, compile_timing = measure_lower_and_compile(
                lambda: jax.jit(operation).lower(points, action_matrix),
                lambda lowered: lowered.compile(),
            )
            action_result, execution = measure_repeated(
                lambda: executable(points, action_matrix),
                warmup=1,
                repeats=2,
            )
            variational_evidence[name] = dict(
                compiler=asdict(
                    compiler_evidence(
                        executable.cost_analysis(),
                        executable.memory_analysis(),
                        source="jax",
                    )
                ),
                compilation=asdict(compile_timing),
                execution=execution.to_dict(),
                retained_bytes=logical_array_bytes(action_result),
            )
        records.append(
            dict(
                n=n,
                compiler=evidence,
                compilation=asdict(timing),
                execution=samples.to_dict(),
                retained_bytes=logical_array_bytes(result),
                variational=variational_evidence,
                action_parity_error=parity_error,
                logical_workspace_bytes=(n + 4) * 8,
            )
        )
        jax.clear_caches()
    root = Path(__file__).resolve().parents[1]
    result: dict[str, Any] = dict(
        records=records,
        environment=capture_environment().to_dict(),
        identity=capture_benchmark_identity(
            root, Path(__file__), ("compiler", "retained_bytes")
        ).to_dict(),
        retained_exponent=fitted_scaling_exponent(
            sizes, [r["retained_bytes"] for r in records]
        ),
        peak_rss_bytes=process_peak_rss_bytes(),
    )
    baseline = root / "benchmarks/memory_gp_actions_baseline.json"
    if not args.baseline:
        old = json.loads(baseline.read_text())
        result["gate"] = evaluate_memory_gate(
            sizes,
            [r["compiler"]["temporary_bytes"] for r in old["records"]],
            [r["compiler"]["temporary_bytes"] for r in records],
            MemoryGatePolicy(),
        ).to_dict()
        result["variational_gate"] = evaluate_memory_gate(
            sizes,
            [r["variational"]["dense"]["compiler"]["temporary_bytes"] for r in records],
            [r["variational"]["bounded"]["compiler"]["temporary_bytes"] for r in records],
            MemoryGatePolicy(),
        ).to_dict()
        result["workspace_exponent"] = fitted_scaling_exponent(
            sizes,
            [r["logical_workspace_bytes"] for r in records],
        )
        result["compiler_workspace_exponent"] = fitted_scaling_exponent(
            sizes,
            [r["compiler"]["temporary_bytes"] for r in records],
        )
        result["variational_compiler_workspace_exponent"] = fitted_scaling_exponent(
            sizes,
            [r["variational"]["bounded"]["compiler"]["temporary_bytes"] for r in records],
        )
    write_json_atomic(
        baseline if args.baseline else root / "benchmarks/memory_gp_actions.json", result
    )


if __name__ == "__main__":
    main()
