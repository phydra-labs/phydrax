#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx


_DTYPES = {"float32": jnp.float32, "float64": jnp.float64}


@dataclass(frozen=True)
class _Case:
    backend: str
    operation: str
    size: int
    dtype_name: str
    workload: str


@dataclass(frozen=True)
class _Configuration:
    native_epsilon: float
    fast_temperature: float
    warmups: int
    repeats: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare native Sinkhorn and fast PAV soft ordering."
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("native-sinkhorn", "fast-pav"),
        default=("native-sinkhorn", "fast-pav"),
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("sort", "rank"),
        default=("sort", "rank"),
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=(64, 256, 1024))
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=tuple(_DTYPES),
        default=("float32",),
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=("sinusoidal", "nearly-tied", "repeated", "heavy-tailed"),
        default=("sinusoidal",),
    )
    parser.add_argument("--native-epsilon", type=float, default=0.1)
    parser.add_argument("--fast-temperature", type=float, default=0.4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-native-pairs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _validate(arguments: argparse.Namespace) -> None:
    if any(size < 2 for size in arguments.sizes):
        raise ValueError("sizes must be at least two.")
    if arguments.native_epsilon <= 0.0:
        raise ValueError("native-epsilon must be positive.")
    if arguments.fast_temperature <= 0.0:
        raise ValueError("fast-temperature must be positive.")
    if arguments.warmups < 0 or arguments.repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats positive.")
    if arguments.max_native_pairs is not None and arguments.max_native_pairs < 1:
        raise ValueError("max-native-pairs must be positive or omitted.")


def _cases(arguments: argparse.Namespace) -> list[_Case]:
    if arguments.smoke:
        return [
            _Case("native-sinkhorn", "sort", 8, "float32", "sinusoidal"),
            _Case("fast-pav", "sort", 8, "float32", "sinusoidal"),
            _Case("fast-pav", "rank", 8, "float64", "nearly-tied"),
        ]
    return [
        _Case(backend, operation, size, dtype_name, workload)
        for backend in arguments.backends
        for operation in arguments.operations
        for size in arguments.sizes
        for dtype_name in arguments.dtypes
        for workload in arguments.workloads
    ]


def _values(case: _Case, seed: int) -> jax.Array:
    dtype = _DTYPES[case.dtype_name]
    index = jnp.arange(case.size, dtype=dtype)
    phase = jnp.asarray(1.61803398875, dtype=dtype) * index
    base = jnp.sin(phase) + 0.1 * jnp.cos(0.37 * phase)
    if case.workload == "sinusoidal":
        return base
    if case.workload == "nearly-tied":
        scale = 1e-3 if dtype == jnp.float32 else 1e-7
        return jnp.ones_like(base) + scale * (base + index / case.size)
    if case.workload == "repeated":
        return jnp.mod(jnp.arange(case.size), min(case.size, 8)).astype(dtype)
    key = jax.random.fold_in(jax.random.key(seed), case.size)
    uniform = jax.random.uniform(
        key,
        (case.size,),
        dtype=dtype,
        minval=1e-3,
        maxval=1.0 - 1e-3,
    )
    return jnp.tan(jnp.pi * (uniform - 0.5))


def _operation(case: _Case, configuration: _Configuration):
    if case.backend == "native-sinkhorn":
        solver = phx.transport.Sinkhorn(
            configuration.native_epsilon,
            max_iterations=300,
            min_iterations=1,
            tolerance=1e-7,
            check_every=5,
            early_stop=False,
            store_history=False,
        )
        if case.operation == "sort":
            return lambda values: phx.transport.soft_sort(values, solver=solver)
        return lambda values: phx.transport.soft_rank(values, solver=solver)
    if case.operation == "sort":
        return lambda values: phx.transport.fast_soft_sort(
            values,
            temperature=configuration.fast_temperature,
        )
    return lambda values: phx.transport.fast_soft_rank(
        values,
        temperature=configuration.fast_temperature,
    )


def _hard_output(case: _Case, values: jax.Array) -> jax.Array:
    if case.operation == "sort":
        return jnp.sort(values)
    order = jnp.argsort(values, stable=True)
    return (
        jnp.zeros((case.size,), dtype=values.dtype)
        .at[order]
        .set(jnp.arange(case.size, dtype=values.dtype))
    )


def _block(value: Any) -> Any:
    return jax.tree.map(jax.block_until_ready, value)


def _compile(function, *arguments):
    started = time.perf_counter_ns()
    compiled = jax.jit(function).lower(*arguments).compile()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return compiled, elapsed_ms


def _execute(compiled, *arguments):
    started = time.perf_counter_ns()
    output = _block(compiled(*arguments))
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return output, elapsed_ms


def _steady(compiled, arguments, *, warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        _block(compiled(*arguments))
    started = time.perf_counter_ns()
    for _ in range(repeats):
        _block(compiled(*arguments))
    return (time.perf_counter_ns() - started) / (1e6 * repeats)


def _memory(compiled) -> dict[str, int | str]:
    analysis = compiled.memory_analysis()
    if analysis is None:
        return {"status": "unavailable"}
    return {
        "status": "available",
        "argument_bytes": int(analysis.argument_size_in_bytes),
        "output_bytes": int(analysis.output_size_in_bytes),
        "temporary_bytes": int(analysis.temp_size_in_bytes),
        "alias_bytes": int(analysis.alias_size_in_bytes),
        "generated_code_bytes": int(analysis.generated_code_size_in_bytes),
    }


def _accuracy(case: _Case, values: jax.Array, output: jax.Array) -> dict[str, Any]:
    hard = _hard_output(case, values).astype(output.dtype)
    relative_error = jnp.linalg.norm(output - hard) / jnp.maximum(
        jnp.linalg.norm(hard),
        jnp.asarray(1e-12, dtype=output.dtype),
    )
    tolerance = 20.0 * jnp.finfo(output.dtype).eps
    order = jnp.argsort(values, stable=True)
    ordered_output = output if case.operation == "sort" else output[order]
    maximum = jnp.max(values) if case.operation == "sort" else case.size - 1
    minimum = jnp.min(values) if case.operation == "sort" else 0.0
    expected_sum = (
        jnp.sum(values) if case.operation == "sort" else case.size * (case.size - 1) / 2
    )
    return {
        "relative_hard_error": float(relative_error),
        "monotonicity_violations": int(jnp.sum(jnp.diff(ordered_output) < -tolerance)),
        "range_violations": int(
            jnp.sum((output < minimum - tolerance) | (output > maximum + tolerance))
        ),
        "sum_error": float(jnp.abs(jnp.sum(output) - expected_sum)),
    }


def _equivariance(case: _Case, operation, values, output) -> dict[str, float]:
    transformed = operation(3.0 * values - 2.0)
    if case.operation == "sort":
        residual = transformed - (3.0 * output - 2.0)
    else:
        residual = transformed - output
    permutation = jnp.roll(jnp.arange(case.size), max(1, case.size // 3))
    permuted = operation(values[permutation])
    if case.operation == "sort":
        permutation_residual = permuted - output
    else:
        permutation_residual = permuted - output[permutation]
    return {
        "positive_affine_linf": float(jnp.max(jnp.abs(residual))),
        "permutation_linf": float(jnp.max(jnp.abs(permutation_residual))),
    }


def _convergence(
    case: _Case,
    values: jax.Array,
    configuration: _Configuration,
) -> dict[str, Any]:
    if case.backend != "native-sinkhorn":
        return {"status": "not-applicable"}
    solver = phx.transport.Sinkhorn(
        configuration.native_epsilon,
        max_iterations=300,
        min_iterations=1,
        tolerance=1e-7,
        check_every=5,
        early_stop=False,
        store_history=False,
    )
    solve = jax.jit(
        lambda candidate: phx.transport.soft_order_transport(
            candidate,
            solver=solver,
        )
    )
    result = _block(solve(values))
    diagnostics = result.diagnostics
    return {
        "status": "available",
        "converged": bool(result.converged),
        "iterations": int(diagnostics.num_iterations),
        "normalized_marginal_residual": float(diagnostics.normalized_marginal_residual),
        "physical_marginal_residual": float(diagnostics.physical_marginal_residual),
    }


def _record(
    case: _Case,
    configuration: _Configuration,
    *,
    seed: int,
) -> dict[str, Any]:
    values = _values(case, seed)
    operation = _operation(case, configuration)
    forward, forward_compile_ms = _compile(operation, values)
    output, first_forward_ms = _execute(forward, values)
    steady_forward_ms = _steady(
        forward,
        (values,),
        warmups=configuration.warmups,
        repeats=configuration.repeats,
    )

    coefficients = jnp.linspace(0.5, 1.5, case.size, dtype=values.dtype)

    def objective(candidate):
        return jnp.sum(coefficients * operation(candidate))

    reverse, reverse_compile_ms = _compile(jax.grad(objective), values)
    gradient, first_reverse_ms = _execute(reverse, values)
    steady_reverse_ms = _steady(
        reverse,
        (values,),
        warmups=configuration.warmups,
        repeats=configuration.repeats,
    )
    direction = jnp.cos(jnp.arange(case.size, dtype=values.dtype))

    def jvp(candidate, tangent):
        return jax.jvp(operation, (candidate,), (tangent,))[1]

    forward_mode, jvp_compile_ms = _compile(jvp, values, direction)
    tangent, first_jvp_ms = _execute(forward_mode, values, direction)
    steady_jvp_ms = _steady(
        forward_mode,
        (values, direction),
        warmups=configuration.warmups,
        repeats=configuration.repeats,
    )
    return {
        "backend": case.backend,
        "operation": case.operation,
        "size": case.size,
        "dtype": case.dtype_name,
        "output_dtype": str(output.dtype),
        "workload": case.workload,
        "parameter": {
            "name": ("epsilon" if case.backend == "native-sinkhorn" else "temperature"),
            "value": (
                configuration.native_epsilon
                if case.backend == "native-sinkhorn"
                else configuration.fast_temperature
            ),
        },
        "status": "ok",
        "timing_ms": {
            "forward_compile": forward_compile_ms,
            "first_forward_execution": first_forward_ms,
            "steady_forward": steady_forward_ms,
            "reverse_compile": reverse_compile_ms,
            "first_reverse_execution": first_reverse_ms,
            "steady_reverse": steady_reverse_ms,
            "jvp_compile": jvp_compile_ms,
            "first_jvp_execution": first_jvp_ms,
            "steady_jvp": steady_jvp_ms,
        },
        "compiled_memory": {
            "forward": _memory(forward),
            "reverse": _memory(reverse),
            "jvp": _memory(forward_mode),
        },
        "accuracy": _accuracy(case, values, output),
        "equivariance": _equivariance(case, operation, values, output),
        "convergence": _convergence(case, values, configuration),
        "gradient_norm": float(jnp.linalg.norm(gradient)),
        "jvp_norm": float(jnp.linalg.norm(tangent)),
        "nonfinite_gradient_count": int(jnp.sum(~jnp.isfinite(gradient))),
        "nonfinite_jvp_count": int(jnp.sum(~jnp.isfinite(tangent))),
    }


def _metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "default_backend": jax.default_backend(),
        "devices": [
            {
                "id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
                "process_index": device.process_index,
            }
            for device in jax.devices()
        ],
    }


def main() -> None:
    arguments = _parser().parse_args()
    _validate(arguments)
    configuration = _Configuration(
        native_epsilon=arguments.native_epsilon,
        fast_temperature=arguments.fast_temperature,
        warmups=0 if arguments.smoke else arguments.warmups,
        repeats=1 if arguments.smoke else arguments.repeats,
    )
    records = []
    for case in _cases(arguments):
        if (
            case.backend == "native-sinkhorn"
            and arguments.max_native_pairs is not None
            and case.size * case.size > arguments.max_native_pairs
        ):
            records.append(
                {
                    "backend": case.backend,
                    "operation": case.operation,
                    "size": case.size,
                    "dtype": case.dtype_name,
                    "workload": case.workload,
                    "status": "resource-limit",
                    "reason": (
                        f"size^2={case.size * case.size} exceeds explicit "
                        f"max_native_pairs={arguments.max_native_pairs}"
                    ),
                }
            )
            continue
        records.append(_record(case, configuration, seed=arguments.seed))
    report = {
        "schema": "phydrax.fast-order-benchmark.v1",
        "configuration": {
            "backends": list(arguments.backends),
            "operations": list(arguments.operations),
            "sizes": list(arguments.sizes),
            "dtypes": list(arguments.dtypes),
            "workloads": list(arguments.workloads),
            "native_epsilon": configuration.native_epsilon,
            "fast_temperature": configuration.fast_temperature,
            "warmups": configuration.warmups,
            "repeats": configuration.repeats,
            "seed": arguments.seed,
            "max_native_pairs": arguments.max_native_pairs,
            "smoke": arguments.smoke,
        },
        "metadata": _metadata(),
        "records": records,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
