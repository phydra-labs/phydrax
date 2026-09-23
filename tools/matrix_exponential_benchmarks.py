#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Prepared Taylor versus fresh Arnoldi exponential-action scaling.

Run: python -m tools.matrix_exponential_benchmarks --quick
Timings separate preparation, lowering, compilation, first execution, and warmed
execution. Compiler memory is official executable evidence, not process peak RSS.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import phydrax as phx
from benchmarks._runtime import (
    compiler_evidence,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


def _measure(function, argument, *, repetitions: int) -> tuple[jax.Array, dict]:
    compiled_function = jax.jit(function)
    compiled, compilation = measure_lower_and_compile(
        lambda: compiled_function.lower(argument),
        lambda lowered: lowered.compile(),
    )
    first, first_seconds = measure_synchronized(lambda: compiled(argument))
    _, steady = measure_repeated(
        lambda: compiled(argument),
        warmup=0,
        repeats=repetitions,
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Backend did not expose compiler cost or memory analysis.",
    )
    return first, {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_execution_seconds": first_seconds,
        "steady": steady.to_seconds_dict(),
        "compiler": asdict(evidence),
        "compiler_estimated_device_memory_bytes": (
            evidence.estimated_device_memory_bytes
        ),
    }


def _operator_matrix(size: int) -> jax.Array:
    diagonal = -jnp.linspace(0.5, 4.0, size, dtype=jnp.float64)
    upper = jnp.linspace(1.0, 3.0, size - 1, dtype=jnp.float64)
    second = jnp.linspace(-0.3, 0.2, max(size - 2, 0), dtype=jnp.float64)
    matrix = jnp.diag(diagonal) + jnp.diag(upper, 1)
    if size > 2:
        matrix = matrix + jnp.diag(second, 2)
    return matrix


def benchmark_case(*, size: int, repetitions: int) -> dict:
    matrix = _operator_matrix(size)
    operator = phx.linalg.DenseLinearOperator(
        matrix,
        operator_id=f"matrix-exponential-benchmark:{size}",
    )
    rhs = jnp.cos(jnp.arange(size, dtype=jnp.float64) + 0.25)
    scale = jnp.asarray(0.2, dtype=jnp.float64)
    taylor_policy = phx.linalg.TaylorExponentialPolicy(error_tolerance=1e-9)
    prepared, preparation_seconds = measure_synchronized(
        lambda: phx.linalg.prepare_taylor_exponential_action(
            operator,
            taylor_policy,
        )
    )

    def taylor_action(vector):
        return phx.linalg.matrix_exponential_action(
            prepared,
            vector,
            scale,
        ).value

    def arnoldi_action(vector):
        return phx.linalg.matrix_exponential_action(
            operator,
            vector,
            scale,
            policy=phx.linalg.MatrixFunctionPolicy(
                "arnoldi",
                max_dimension=min(size, 32),
                error_tolerance=1e-9,
            ),
        ).value

    taylor_value, taylor_timing = _measure(
        taylor_action,
        rhs,
        repetitions=repetitions,
    )
    arnoldi_value, arnoldi_timing = _measure(
        arnoldi_action,
        rhs,
        repetitions=repetitions,
    )
    reference = jsp.linalg.expm(scale * matrix) @ rhs
    reference_scale = jnp.maximum(jnp.linalg.norm(reference), 1e-30)
    taylor_error = jnp.linalg.norm(taylor_value - reference) / reference_scale
    arnoldi_error = jnp.linalg.norm(arnoldi_value - reference) / reference_scale
    evidence = phx.linalg.matrix_exponential_action(prepared, rhs, scale)
    passed = bool(
        evidence.successful
        & jnp.isfinite(taylor_error)
        & (taylor_error <= 5e-8)
        & jnp.isfinite(arnoldi_error)
    )
    return {
        "dimension": size,
        "dtype": str(matrix.dtype),
        "preparation_seconds": preparation_seconds,
        "prepared_retained_storage_bytes": prepared.plan.retained_storage_bytes,
        "prepared_workspace_bytes": prepared.plan.workspace_bytes,
        "selected_degree": int(evidence.diagnostics.selected_degree),
        "scaling_count": int(evidence.diagnostics.scaling_count),
        "setup_matvec_count": int(evidence.diagnostics.setup_matvec_count),
        "action_matvec_count": int(evidence.diagnostics.action_matvec_count),
        "transpose_matvec_count": int(evidence.diagnostics.transpose_matvec_count),
        "taylor_relative_error": float(taylor_error),
        "arnoldi_relative_error": float(arnoldi_error),
        "taylor": taylor_timing,
        "arnoldi": arnoldi_timing,
        "passed": passed,
    }


def run_benchmarks(*, quick: bool = False) -> dict:
    sizes = (8, 16) if quick else (16, 32, 64, 128)
    repetitions = 2 if quick else 7
    cases = [benchmark_case(size=size, repetitions=repetitions) for size in sizes]
    return {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "quick": quick,
        "workload": "fixed nonnormal operator with changing right-hand sides",
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_benchmarks(quick=arguments.quick)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
