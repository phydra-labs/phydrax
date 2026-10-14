"""Synchronized value, JVP, and VJP phases for batched 3-by-3 functions.

Run with ``python -m benchmarks.hermitian_spectral --contended`` when other
workloads are active; the record explicitly retains that measurement condition.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import config as pxconfig
from phydrax.linalg import (
    hermitian_exp,
    hermitian_inverse_sqrt,
    hermitian_log,
    hermitian_sqrt,
    HermitianFunctionResult,
)


def benchmark_case(
    batch_size: int,
    function: Callable[[Array], HermitianFunctionResult],
    function_id: str,
    repeats: int,
) -> dict[str, Any]:
    base = jnp.asarray(
        ((1.4, 0.2, 0.1), (0.2, 1.7, -0.1), (0.1, -0.1, 2.0)), dtype=jnp.float64
    )
    matrices = jnp.broadcast_to(base, (batch_size, 3, 3))
    directions = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), matrices.shape)
    cotangents = jnp.ones_like(matrices)

    def value(matrix: Array) -> Array:
        return function(matrix).value

    def jvp(matrix: Array, direction: Array) -> tuple[Array, Array]:
        return jax.jvp(value, (matrix,), (direction,))

    def vjp(matrix: Array, cotangent: Array) -> tuple[Array, Array]:
        primal, pullback = jax.vjp(value, matrix)
        return primal, pullback(cotangent)[0]

    phases: dict[str, Any] = {}
    all_finite = True
    for phase, operation, arguments in (
        ("value", jax.jit(value), (matrices,)),
        ("jvp", jax.jit(jvp), (matrices, directions)),
        ("vjp", jax.jit(vjp), (matrices, cotangents)),
    ):
        compiled, compilation = measure_lower_and_compile(
            lambda operation=operation, arguments=arguments: operation.lower(*arguments),
            lambda lowered: lowered.compile(),
        )
        _, first_seconds = measure_synchronized(
            lambda compiled=compiled, arguments=arguments: compiled(*arguments)
        )
        result, timing = measure_repeated(
            lambda compiled=compiled, arguments=arguments: compiled(*arguments),
            warmup=1,
            repeats=repeats,
        )
        finite = all(
            bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(result)
        )
        all_finite = all_finite and finite
        phases[phase] = {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "first_execution_seconds": first_seconds,
            "steady": timing.to_seconds_dict(),
            "output_logical_bytes": logical_array_bytes(result),
            "finite": finite,
        }
    return {
        "axes": {"batch_size": batch_size, "matrix_size": 3},
        "function_id": function_id,
        "dtype": str(matrices.dtype),
        "input_logical_bytes": logical_array_bytes(matrices),
        "phases": phases,
        "successful": all_finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(1, 1_000, 100_000))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--contended", action="store_true")
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or any(size < 1 for size in arguments.batch_sizes):
        raise ValueError("Batch sizes and repeats must be positive.")
    pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
    functions = (
        ("hermitian-log", hermitian_log),
        ("hermitian-exp", hermitian_exp),
        ("hermitian-sqrt", hermitian_sqrt),
        ("hermitian-inverse-sqrt", hermitian_inverse_sqrt),
    )
    cases = [
        benchmark_case(size, function, identifier, arguments.repeats)
        for size in arguments.batch_sizes
        for identifier, function in functions
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "measurement_conditions": {"contended": arguments.contended},
        "cases": cases,
        "passed": all(case["successful"] for case in cases),
    }
    if arguments.output:
        write_json_atomic(arguments.output, payload)
    else:
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
