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

import coordax as cx
import jax
import jax.numpy as jnp

import phydrax as phx


_OPERATIONS = ("sort", "rank", "topk", "quantile")
_WORKLOADS = (
    "sinusoidal",
    "nearly-tied",
    "repeated",
    "heavy-tailed",
    "nonuniform-zero-weight",
    "batched",
    "named-field",
)
_DTYPES = {"float32": jnp.float32, "float64": jnp.float64}
_QUANTILES = jnp.asarray([0.1, 0.5, 0.9])


@dataclass(frozen=True)
class _Workload:
    values: jax.Array
    weights: jax.Array | None
    named: bool


@dataclass(frozen=True)
class _Case:
    operation: str
    size: int
    epsilon: float
    block_size: int | None
    dtype_name: str
    workload: str


def _block_size(value: str) -> int | None:
    if value.lower() in ("dense", "none"):
        return None
    size = int(value)
    if size < 1:
        raise argparse.ArgumentTypeError("block sizes must be positive or 'dense'.")
    return size


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark native differentiable transport ordering operators."
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=_OPERATIONS,
        default=_OPERATIONS,
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=(64, 256, 1024))
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=(0.2, 0.1, 0.05, 0.02),
    )
    parser.add_argument(
        "--block-sizes",
        type=_block_size,
        nargs="+",
        default=(None, 64, 128),
        metavar="{dense,N}",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=tuple(_DTYPES),
        default=tuple(_DTYPES),
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=_WORKLOADS,
        default=_WORKLOADS,
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Explicit resource limit; skipped cases remain in JSON records.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if any(size < 2 for size in arguments.sizes):
        raise ValueError("sizes must contain integers of at least two.")
    if any(not jnp.isfinite(value) or value <= 0.0 for value in arguments.epsilons):
        raise ValueError("epsilons must be finite and positive.")
    if arguments.warmups < 0:
        raise ValueError("warmups must be nonnegative.")
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    if arguments.max_pairs is not None and arguments.max_pairs < 1:
        raise ValueError("max-pairs must be positive or omitted.")


def _normal_cases(arguments: argparse.Namespace) -> list[_Case]:
    return [
        _Case(operation, size, epsilon, block_size, dtype_name, workload)
        for operation in arguments.operations
        for size in arguments.sizes
        for epsilon in arguments.epsilons
        for block_size in arguments.block_sizes
        for dtype_name in arguments.dtypes
        for workload in arguments.workloads
    ]


def _smoke_cases() -> list[_Case]:
    return [
        _Case("sort", 8, 0.2, None, "float32", "sinusoidal"),
        _Case("rank", 8, 0.2, 4, "float32", "nearly-tied"),
        _Case("topk", 8, 0.2, None, "float32", "nonuniform-zero-weight"),
        _Case("quantile", 8, 0.2, 4, "float32", "named-field"),
    ]


def _workload(name: str, size: int, dtype: jnp.dtype, seed: int) -> _Workload:
    index = jnp.arange(size, dtype=dtype)
    phase = jnp.asarray(1.61803398875, dtype=dtype) * index
    base = jnp.sin(phase) + jnp.asarray(0.1, dtype=dtype) * jnp.cos(0.37 * phase)

    if name == "sinusoidal":
        return _Workload(base, None, False)
    if name == "nearly-tied":
        scale = jnp.asarray(1e-3 if dtype == jnp.float32 else 1e-7, dtype=dtype)
        values = jnp.ones((size,), dtype=dtype) + scale * (base + index / size)
        return _Workload(values, None, False)
    if name == "repeated":
        values = jnp.mod(jnp.arange(size), min(size, 8)).astype(dtype)
        return _Workload(values, None, False)
    if name == "heavy-tailed":
        key = jax.random.fold_in(jax.random.key(seed), size)
        uniform = jax.random.uniform(
            key,
            (size,),
            dtype=dtype,
            minval=jnp.asarray(1e-3, dtype=dtype),
            maxval=jnp.asarray(1.0 - 1e-3, dtype=dtype),
        )
        values = jnp.tan(jnp.pi * (uniform - 0.5))
        return _Workload(values, None, False)
    if name == "nonuniform-zero-weight":
        weights = 1.0 + jnp.mod(jnp.arange(size), 11).astype(dtype)
        weights = jnp.where(jnp.mod(jnp.arange(size), 7) == 0, 0.0, weights)
        return _Workload(base, weights, False)
    if name == "batched":
        values = jnp.stack((base, base + 0.2, -0.7 * base + 0.4))
        return _Workload(values, None, False)
    if name == "named-field":
        values = jnp.stack((base, jnp.roll(base, 1)))
        return _Workload(values, None, True)
    raise ValueError(f"Unsupported workload {name!r}.")


def _solver(epsilon: float, block_size: int | None) -> phx.transport.Sinkhorn:
    return phx.transport.Sinkhorn(
        epsilon,
        max_iterations=300,
        min_iterations=1,
        tolerance=1e-7,
        check_every=5,
        block_size=block_size,
        early_stop=False,
        store_history=False,
    )


def _operation(
    case: _Case,
    workload: _Workload,
    solver: phx.transport.Sinkhorn,
):
    count = case.size
    selected = max(1, count // 4)

    def apply(values):
        if workload.named:
            value = cx.Field(values, dims=("case", "sample"))
            weights = (
                None
                if workload.weights is None
                else cx.Field(workload.weights, dims=("case", "sample"))
            )
            axis: int | str = "sample"
        else:
            value = values
            weights = workload.weights
            axis = -1

        if case.operation == "sort":
            output = phx.transport.soft_sort(
                value,
                weights=weights,
                axis=axis,
                solver=solver,
            )
        elif case.operation == "rank":
            output = phx.transport.soft_rank(
                value,
                weights=weights,
                axis=axis,
                solver=solver,
            )
        elif case.operation == "topk":
            output = phx.transport.soft_topk_mask(
                value,
                selected,
                weights=weights,
                axis=axis,
                solver=solver,
            )
        else:
            output = phx.transport.soft_quantile(
                value,
                _QUANTILES.astype(values.dtype),
                weights=weights,
                axis=axis,
                solver=solver,
                quantile_dim="quantile",
            )
        return output.data if isinstance(output, cx.Field) else output

    return apply


def _block(value: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf.block_until_ready(), value)


def _compile(function, example: jax.Array):
    started = time.perf_counter_ns()
    compiled = jax.jit(function).lower(example).compile()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return compiled, elapsed_ms


def _execute(compiled, example: jax.Array):
    started = time.perf_counter_ns()
    result = _block(compiled(example))
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return result, elapsed_ms


def _steady_ms(compiled, example: jax.Array, warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        _block(compiled(example))
    started = time.perf_counter_ns()
    for _ in range(repeats):
        _block(compiled(example))
    return (time.perf_counter_ns() - started) / (1e6 * repeats)


def _memory(compiled) -> dict[str, int | str]:
    analysis = compiled.memory_analysis()
    if analysis is None:
        return {"status": "unavailable"}
    return {
        "status": "available",
        "generated_code_bytes": int(analysis.generated_code_size_in_bytes),
        "argument_bytes": int(analysis.argument_size_in_bytes),
        "output_bytes": int(analysis.output_size_in_bytes),
        "alias_bytes": int(analysis.alias_size_in_bytes),
        "temporary_bytes": int(analysis.temp_size_in_bytes),
        "host_generated_code_bytes": int(analysis.host_generated_code_size_in_bytes),
        "host_argument_bytes": int(analysis.host_argument_size_in_bytes),
        "host_output_bytes": int(analysis.host_output_size_in_bytes),
        "host_alias_bytes": int(analysis.host_alias_size_in_bytes),
        "host_temporary_bytes": int(analysis.host_temp_size_in_bytes),
    }


def _rows(values: jax.Array, size: int) -> jax.Array:
    return values.reshape((-1, size))


def _weight_rows(workload: _Workload, size: int) -> jax.Array:
    rows = _rows(workload.values, size)
    if workload.weights is None:
        return jnp.ones_like(rows)
    return jnp.broadcast_to(workload.weights, workload.values.shape).reshape((-1, size))


def _hard_output(case: _Case, workload: _Workload) -> jax.Array:
    rows = _rows(workload.values, case.size)
    weights = _weight_rows(workload, case.size)
    target_index = jnp.arange(case.size, dtype=workload.values.dtype)
    target_lower = target_index / case.size
    target_upper = (target_index + 1.0) / case.size

    def hard_row(row, row_weights):
        probabilities = row_weights / jnp.sum(row_weights)
        active = probabilities > 0.0
        order = jnp.argsort(jnp.where(active, row, jnp.inf), stable=True)
        ordered_values = row[order]
        ordered_probabilities = probabilities[order]
        source_upper = jnp.cumsum(ordered_probabilities)
        source_lower = source_upper - ordered_probabilities
        overlap = jnp.maximum(
            0.0,
            jnp.minimum(source_upper[:, None], target_upper[None, :])
            - jnp.maximum(source_lower[:, None], target_lower[None, :]),
        )

        if case.operation == "sort":
            return case.size * jnp.sum(
                overlap * ordered_values[:, None],
                axis=0,
            )

        target_payload = target_index
        if case.operation == "topk":
            selected = max(1, case.size // 4)
            target_payload = target_index >= case.size - selected
        source_payload = jnp.sum(
            overlap * target_payload.astype(row.dtype)[None, :],
            axis=1,
        )
        safe_probabilities = jnp.where(
            ordered_probabilities > 0.0,
            ordered_probabilities,
            1.0,
        )
        source_payload = jnp.where(
            ordered_probabilities > 0.0,
            source_payload / safe_probabilities,
            0.0,
        )
        restored = jnp.zeros_like(row).at[order].set(source_payload)
        if case.operation in ("rank", "topk"):
            return restored

        sorted_values = case.size * jnp.sum(
            overlap * ordered_values[:, None],
            axis=0,
        )
        quantiles = _QUANTILES.astype(row.dtype)
        result = jnp.interp(
            quantiles * (case.size - 1),
            target_index,
            sorted_values,
        )
        lower = jnp.min(jnp.where(active, row, jnp.inf))
        upper = jnp.max(jnp.where(active, row, -jnp.inf))
        result = jnp.where(quantiles == 0.0, lower, result)
        return jnp.where(quantiles == 1.0, upper, result)

    result = jax.vmap(hard_row)(rows, weights)
    leading = workload.values.shape[:-1]
    return result.reshape(leading + result.shape[1:])


def _accuracy(
    case: _Case,
    workload: _Workload,
    output: jax.Array,
) -> dict[str, float | int | None]:
    hard = _hard_output(case, workload)
    denominator = jnp.maximum(jnp.linalg.norm(hard), jnp.asarray(1e-12))
    relative_error = jnp.linalg.norm(output - hard) / denominator
    values = _rows(workload.values, case.size)
    output_rows = output.reshape((values.shape[0], -1))
    tolerance = 10.0 * jnp.finfo(workload.values.dtype).eps
    weights = _weight_rows(workload, case.size)
    active = weights > 0.0

    if case.operation == "sort":
        monotonicity = jnp.sum(jnp.diff(output_rows, axis=-1) < -tolerance)
        lower = jnp.min(jnp.where(active, values, jnp.inf), axis=-1, keepdims=True)
        upper = jnp.max(jnp.where(active, values, -jnp.inf), axis=-1, keepdims=True)
        range_violations = jnp.sum(
            (output_rows < lower - tolerance) | (output_rows > upper + tolerance)
        )
    elif case.operation == "quantile":
        monotonicity = jnp.sum(jnp.diff(output_rows, axis=-1) < -tolerance)
        lower = jnp.min(jnp.where(active, values, jnp.inf), axis=-1, keepdims=True)
        upper = jnp.max(jnp.where(active, values, -jnp.inf), axis=-1, keepdims=True)
        range_violations = jnp.sum(
            (output_rows < lower - tolerance) | (output_rows > upper + tolerance)
        )
    else:
        order = jnp.argsort(jnp.where(active, values, jnp.inf), axis=-1, stable=True)
        ordered_output = jnp.take_along_axis(output_rows, order, axis=-1)
        ordered_weights = jnp.take_along_axis(weights, order, axis=-1)
        active_pairs = (ordered_weights[:, :-1] > 0.0) & (ordered_weights[:, 1:] > 0.0)
        monotonicity = jnp.sum(
            active_pairs & (jnp.diff(ordered_output, axis=-1) < -tolerance)
        )
        maximum = float(case.size - 1) if case.operation == "rank" else 1.0
        range_violations = jnp.sum(
            (output_rows < -tolerance) | (output_rows > maximum + tolerance)
        )

    rank_sum_error: float | None = None
    if case.operation == "rank":
        probabilities = weights / jnp.sum(weights, axis=-1, keepdims=True)
        expected = 0.5 * (case.size - 1)
        conserved = jnp.sum(probabilities * output_rows, axis=-1)
        rank_sum_error = float(jnp.max(jnp.abs(conserved - expected)))

    topk_mass_error: float | None = None
    if case.operation == "topk":
        selected = max(1, case.size // 4)
        probabilities = weights / jnp.sum(weights, axis=-1, keepdims=True)
        conserved = jnp.sum(probabilities * output_rows, axis=-1)
        topk_mass_error = float(jnp.max(jnp.abs(conserved - selected / case.size)))

    return {
        "relative_hard_error": float(relative_error),
        "monotonicity_violations": int(monotonicity),
        "range_violations": int(range_violations),
        "rank_sum_error": rank_sum_error,
        "topk_mass_error": topk_mass_error,
    }


def _convergence(
    workload: _Workload,
    size: int,
    solver: phx.transport.Sinkhorn,
) -> dict[str, Any]:
    values = _rows(workload.values, size)
    weights = _weight_rows(workload, size)
    solve = jax.jit(
        jax.vmap(
            lambda row, row_weights: phx.transport.soft_order_transport(
                row,
                weights=row_weights,
                solver=solver,
            )
        )
    )
    result = _block(solve(values, weights))
    diagnostics = result.diagnostics
    return {
        "all_converged": bool(jnp.all(result.converged)),
        "status_codes": [int(value) for value in diagnostics.status],
        "max_iterations_executed": int(jnp.max(diagnostics.num_iterations)),
        "max_normalized_marginal_residual": float(
            jnp.max(diagnostics.normalized_marginal_residual)
        ),
        "max_physical_marginal_residual": float(
            jnp.max(diagnostics.physical_marginal_residual)
        ),
    }


def _identity(case: _Case) -> dict[str, Any]:
    return {
        "operation": case.operation,
        "size": case.size,
        "epsilon": case.epsilon,
        "block_size": case.block_size,
        "execution": "dense" if case.block_size is None else "blockwise",
        "dtype": case.dtype_name,
        "workload": case.workload,
    }


def _record(case: _Case, *, seed: int, warmups: int, repeats: int) -> dict[str, Any]:
    dtype = _DTYPES[case.dtype_name]
    workload = _workload(case.workload, case.size, dtype, seed)
    solver = _solver(case.epsilon, case.block_size)
    operation = _operation(case, workload, solver)
    values = workload.values

    forward, forward_compile_ms = _compile(operation, values)
    output, first_forward_ms = _execute(forward, values)
    steady_forward_ms = _steady_ms(forward, values, warmups, repeats)

    def objective(candidate):
        transformed = operation(candidate)
        coefficients = jnp.linspace(
            0.5,
            1.5,
            transformed.size,
            dtype=transformed.dtype,
        ).reshape(transformed.shape)
        return jnp.sum(coefficients * transformed)

    reverse, reverse_compile_ms = _compile(jax.grad(objective), values)
    gradient, first_reverse_ms = _execute(reverse, values)
    steady_reverse_ms = _steady_ms(reverse, values, warmups, repeats)

    direction = jnp.cos(jnp.arange(values.size, dtype=dtype)).reshape(values.shape)

    def jvp(candidate):
        return jax.jvp(operation, (candidate,), (direction,))[1]

    forward_mode, jvp_compile_ms = _compile(jvp, values)
    tangent, first_jvp_ms = _execute(forward_mode, values)
    steady_jvp_ms = _steady_ms(forward_mode, values, warmups, repeats)

    return {
        **_identity(case),
        "status": "ok",
        "seed": seed,
        "warmups": warmups,
        "repeats": repeats,
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
        "compiled_memory": _memory(forward),
        "convergence": _convergence(workload, case.size, solver),
        "accuracy": _accuracy(case, workload, output),
        "gradient_norm": float(jnp.linalg.norm(gradient)),
        "nonfinite_gradient_count": int(jnp.sum(~jnp.isfinite(gradient))),
        "jvp_norm": float(jnp.linalg.norm(tangent)),
        "nonfinite_jvp_count": int(jnp.sum(~jnp.isfinite(tangent))),
    }


def _metadata() -> dict[str, Any]:
    devices = [
        {
            "id": device.id,
            "platform": device.platform,
            "device_kind": device.device_kind,
            "process_index": device.process_index,
        }
        for device in jax.devices()
    ]
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "default_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "devices": devices,
    }


def main() -> None:
    arguments = _parser().parse_args()
    _validate_arguments(arguments)
    cases = _smoke_cases() if arguments.smoke else _normal_cases(arguments)
    warmups = 0 if arguments.smoke else int(arguments.warmups)
    repeats = 1 if arguments.smoke else int(arguments.repeats)
    records: list[dict[str, Any]] = []

    for case in cases:
        if (
            arguments.max_pairs is not None
            and case.size * case.size > arguments.max_pairs
        ):
            records.append(
                {
                    **_identity(case),
                    "status": "resource-limit",
                    "reason": (
                        f"size^2={case.size * case.size} exceeds explicit "
                        f"max_pairs={arguments.max_pairs}"
                    ),
                }
            )
            continue
        try:
            records.append(
                _record(
                    case,
                    seed=int(arguments.seed),
                    warmups=warmups,
                    repeats=repeats,
                )
            )
        except Exception as error:
            records.append(
                {
                    **_identity(case),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    payload = {
        "schema": "phydrax.soft-order-benchmark.v1",
        "metadata": _metadata(),
        "configuration": {
            "smoke": bool(arguments.smoke),
            "operations": list(arguments.operations),
            "sizes": list(arguments.sizes),
            "epsilons": list(arguments.epsilons),
            "block_sizes": list(arguments.block_sizes),
            "dtypes": list(arguments.dtypes),
            "workloads": list(arguments.workloads),
            "warmups": warmups,
            "repeats": repeats,
            "seed": int(arguments.seed),
            "max_pairs": arguments.max_pairs,
        },
        "records": records,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
