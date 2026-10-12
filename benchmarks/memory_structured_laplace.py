"""Compile route-specific retained-state construction and covariance actions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax.uq._structured_laplace import _native_eigh


def state_action(
    diagonal: Array, basis: Array, probe: Array, *, mode: str, baseline: bool
) -> tuple[Array, ...]:
    p0 = jnp.asarray(1.0, dtype=diagonal.dtype)
    if mode == "diagonal":
        if not baseline:
            return diagonal, diagonal + p0, probe / (diagonal + p0)
        hessian = jnp.diag(diagonal)
    else:
        hessian = (basis * diagonal[: basis.shape[1]]) @ basis.T
        if mode != "full" and not baseline:
            values = diagonal[: basis.shape[1]]
            return (
                values,
                basis,
                values + p0,
                probe + basis @ ((1 / (values + p0) - 1) * (basis.T @ probe)),
            )
    precision = hessian + p0 * jnp.eye(probe.size, dtype=probe.dtype)
    values, vectors = jnp.linalg.eigh(precision)
    return hessian, precision, values, vectors, vectors @ ((vectors.T @ probe) / values)


def diagonal_fit_replay(matrix: Array, *, baseline: bool) -> tuple[Array, ...]:
    """Replay the old/new diagonal HVP, retained-state, and covariance kernels."""
    dimension = matrix.shape[0]
    if baseline:
        hessian = jax.vmap(lambda vector: matrix @ vector)(
            jnp.eye(dimension, dtype=matrix.dtype)
        ).T
        hessian = 0.5 * (hessian + hessian.T)
        _, _, full_successful = _native_eigh(hessian)
        diagonal = jnp.diag(hessian)
        precision = jnp.diag(diagonal) + jnp.eye(dimension, dtype=matrix.dtype)
        values, vectors, precision_successful = _native_eigh(precision)
        probe = jnp.ones(dimension, dtype=matrix.dtype)
        action = vectors @ ((vectors.T @ probe) / values)
        return (
            hessian,
            diagonal,
            diagonal,
            jnp.eye(dimension, dtype=matrix.dtype),
            full_successful,
            precision,
            values,
            vectors,
            precision_successful,
            action,
        )

    def diagonal_entry(index: Array) -> Array:
        unit = jnp.zeros(dimension, dtype=matrix.dtype).at[index].set(1)
        return (matrix @ unit)[index]

    diagonal = jax.lax.map(diagonal_entry, jnp.arange(dimension, dtype=jnp.int32))
    precision = diagonal + jnp.asarray(1.0, dtype=matrix.dtype)
    return diagonal, precision, jnp.ones(dimension, dtype=matrix.dtype) / precision


def diagonal_latency() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dimension in (16, 64, 256):
        matrix = jnp.diag(jnp.arange(1, dimension + 1, dtype=jnp.float64))
        matrix = matrix + jnp.ones((dimension, dimension), dtype=jnp.float64) / dimension
        routes: dict[str, Any] = {}
        reference: Array | None = None
        for route, baseline in (("dev-dense", True), ("candidate", False)):
            kernel = jax.jit(lambda value: diagonal_fit_replay(value, baseline=baseline))
            compiled, timing = measure_lower_and_compile(
                lambda: kernel.lower(matrix), lambda lowered: lowered.compile()
            )
            result, durations = measure_repeated(
                lambda: compiled(matrix), warmup=3, repeats=15
            )
            if reference is None:
                reference = result[-1]
            else:
                if not bool(jnp.allclose(result[-1], reference, rtol=1e-12, atol=1e-12)):
                    raise ValueError("Diagonal replay covariance actions disagree.")
            routes[route] = {
                "warm_execution": durations.to_seconds_dict(),
                "compilation": asdict(timing),
                "compiler": asdict(
                    compiler_evidence(
                        compiled.cost_analysis(), compiled.memory_analysis(), source="jax"
                    )
                ),
            }
        ratio = (
            routes["candidate"]["warm_execution"]["median_seconds"]
            / routes["dev-dense"]["warm_execution"]["median_seconds"]
        )
        records.append(
            {
                "dimension": dimension,
                "routes": routes,
                "candidate_to_dev_warm_ratio": ratio,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    for dimension in (16, 32, 64) if args.smoke else (64, 128, 256):
        for mode in ("diagonal", "lanczos", "lobpcg", "full"):
            rank = dimension if mode == "full" else 4
            diagonal = jnp.arange(1, dimension + 1, dtype=jnp.float64)
            basis = jnp.eye(dimension, rank, dtype=jnp.float64)
            probe = jnp.ones(dimension, dtype=jnp.float64)
            kernel = jax.jit(
                lambda d, u, v: state_action(d, u, v, mode=mode, baseline=args.baseline)
            )
            compiled, timing = measure_lower_and_compile(
                lambda: kernel.lower(diagonal, basis, probe),
                lambda lowered: lowered.compile(),
            )
            output = compiled(diagonal, basis, probe)
            jax.block_until_ready(output)
            records.append(
                {
                    "dimension": dimension,
                    "rank": rank,
                    "mode": mode,
                    "logical_output_bytes": logical_array_bytes(output),
                    "compilation": {
                        "lowering_seconds": timing.lowering_seconds,
                        "compilation_seconds": timing.compilation_seconds,
                    },
                    "compiler": asdict(
                        compiler_evidence(
                            compiled.cost_analysis(),
                            compiled.memory_analysis(),
                            source="jax",
                        )
                    ),
                }
            )
    root = Path(__file__).resolve().parents[1]
    suffix = "_baseline" if args.baseline else ""
    write_json_atomic(
        root / f"benchmarks/memory_structured_laplace{suffix}.json",
        {
            "scope": "retained-state/action substrate; baseline replays removed dense representation",
            "identity": capture_benchmark_identity(
                root, Path(__file__), ("logical_output_bytes", "compiler")
            ).to_dict(),
            "environment": capture_environment().to_dict(),
            "records": records,
            "diagonal_fit_latency": diagonal_latency(),
        },
    )


if __name__ == "__main__":
    main()
