#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx
from phydrax.linalg._dense_pseudoinverse import (
    apply_pseudoinverse,
    factor_pseudoinverse,
)


la = phx.linalg


def _timed_case(
    name: str,
    function,
    argument: jax.Array,
    /,
    *,
    warmup: int,
    repeats: int,
    residual,
) -> dict[str, Any]:
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(argument),
        lambda lowered: lowered.compile(),
    )
    value, execution = measure_repeated(
        lambda: compiled(argument),
        warmup=warmup,
        repeats=repeats,
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    return {
        "name": name,
        "input_shape": list(argument.shape),
        "input_dtype": str(argument.dtype),
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "logical_input_bytes": logical_array_bytes(argument),
        "logical_output_bytes": logical_array_bytes(value),
        "compiler": {
            "flops": evidence.flops,
            "bytes_accessed": evidence.bytes_accessed,
            "argument_bytes": evidence.argument_bytes,
            "output_bytes": evidence.output_bytes,
            "temporary_bytes": evidence.temporary_bytes,
            "generated_code_bytes": evidence.generated_code_bytes,
            "source": evidence.source,
            "unavailable_reason": evidence.unavailable_reason,
        },
        "residual": float(residual(argument, value)),
    }


def _general_matrix(key: jax.Array, batch: int, size: int) -> jax.Array:
    shape = ((batch,) if batch > 1 else ()) + (size, size)
    matrix = jax.random.normal(key, shape, dtype=jnp.float64)
    return matrix + float(size + 1) * jnp.eye(size, dtype=matrix.dtype)


def _rectangular_matrix(key: jax.Array, batch: int, size: int) -> jax.Array:
    rows = size + max(1, size // 2)
    shape = ((batch,) if batch > 1 else ()) + (rows, size)
    matrix = jax.random.normal(key, shape, dtype=jnp.float64)
    if size > 1:
        matrix = matrix.at[..., -1].set(matrix[..., 0])
    return matrix


def _inverse_residual(matrix: jax.Array, inverse: jax.Array) -> jax.Array:
    identity = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    return jnp.max(jnp.abs(matrix @ inverse - identity))


def _pseudoinverse_residual(matrix: jax.Array, pseudoinverse: jax.Array) -> jax.Array:
    scale = jnp.maximum(jnp.max(jnp.abs(matrix)), 1.0)
    return jnp.max(jnp.abs(matrix @ pseudoinverse @ matrix - matrix)) / scale


def _inverse_cases(
    key: jax.Array,
    sizes: tuple[int, ...],
    batches: tuple[int, ...],
    /,
    *,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    cases = []
    for batch in batches:
        for size in sizes:
            key, matrix_key = jax.random.split(key)
            matrix = _general_matrix(matrix_key, batch, size)
            phydrax_inverse = jax.jit(lambda value: la.inverse(value).value)
            jax_inverse = jax.jit(jnp.linalg.inv)
            cases.append(
                _timed_case(
                    "phydrax_inverse",
                    phydrax_inverse,
                    matrix,
                    warmup=warmup,
                    repeats=repeats,
                    residual=_inverse_residual,
                )
            )
            cases.append(
                _timed_case(
                    "jax_inverse",
                    jax_inverse,
                    matrix,
                    warmup=warmup,
                    repeats=repeats,
                    residual=_inverse_residual,
                )
            )
    return cases


def _pseudoinverse_cases(
    key: jax.Array,
    sizes: tuple[int, ...],
    batches: tuple[int, ...],
    /,
    *,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    cases = []
    for batch in batches:
        for size in sizes:
            key, matrix_key = jax.random.split(key)
            matrix = _rectangular_matrix(matrix_key, batch, size)
            phydrax_pseudoinverse = jax.jit(lambda value: la.pseudoinverse(value).value)
            jax_pseudoinverse = jax.jit(jnp.linalg.pinv)
            cases.append(
                _timed_case(
                    "phydrax_pseudoinverse",
                    phydrax_pseudoinverse,
                    matrix,
                    warmup=warmup,
                    repeats=repeats,
                    residual=_pseudoinverse_residual,
                )
            )
            cases.append(
                _timed_case(
                    "jax_pseudoinverse",
                    jax_pseudoinverse,
                    matrix,
                    warmup=warmup,
                    repeats=repeats,
                    residual=_pseudoinverse_residual,
                )
            )

            right_hand_side = jax.random.normal(
                matrix_key,
                matrix.shape[:-2] + (matrix.shape[-2], 4),
                dtype=matrix.dtype,
            )
            rank_policy = la.RankPolicy()
            fused = jax.jit(
                lambda value, rhs=right_hand_side, policy=rank_policy: (
                    apply_pseudoinverse(
                        factor_pseudoinverse(value, policy),
                        rhs,
                    )
                )
            )
            materialized = jax.jit(
                lambda value, rhs=right_hand_side: la.pseudoinverse(value).value @ rhs
            )
            action_residual = lambda value, result, reference=materialized: jnp.max(
                jnp.abs(value @ result - value @ reference(value))
            )
            cases.append(
                _timed_case(
                    "phydrax_fused_pseudoinverse_apply",
                    fused,
                    matrix,
                    warmup=warmup,
                    repeats=repeats,
                    residual=action_residual,
                )
            )
            materialized_residual = lambda value, result, reference=fused: jnp.max(
                jnp.abs(value @ result - value @ reference(value))
            )
            cases.append(
                _timed_case(
                    "phydrax_materialized_pseudoinverse_apply",
                    materialized,
                    matrix,
                    warmup=warmup,
                    repeats=repeats,
                    residual=materialized_residual,
                )
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 8, 32, 128])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 256])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    sizes = tuple(int(value) for value in arguments.sizes)
    batches = tuple(int(value) for value in arguments.batch_sizes)
    if any(value < 1 for value in sizes + batches):
        raise ValueError("sizes and batch sizes must be positive.")
    key = jax.random.PRNGKey(20260901)
    inverse_cases = _inverse_cases(
        key,
        sizes,
        batches,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    pseudoinverse_cases = _pseudoinverse_cases(
        jax.random.fold_in(key, 1),
        sizes,
        batches,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    cases = inverse_cases + pseudoinverse_cases
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "maximum_residual": max(case["residual"] for case in cases),
        "all_finite": all(jnp.isfinite(case["residual"]) for case in cases),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
