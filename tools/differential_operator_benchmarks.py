#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from phydrax.operators.differential import evaluate_fused_coordinate_derivatives


def _seconds(function, argument, repeats: int, /) -> tuple[float, float]:
    started = time.perf_counter()
    compiled = function.lower(argument).compile()
    compile_seconds = time.perf_counter() - started
    compiled(argument).block_until_ready()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        compiled(argument).block_until_ready()
        samples.append(time.perf_counter() - started)
    return compile_seconds, min(samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare action-based and dense exact coordinate derivatives."
    )
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    dimension = int(arguments.dimension)
    repeats = int(arguments.repeats)
    if dimension < 2 or repeats < 1:
        raise ValueError("dimension must exceed one and repeats must be positive.")

    point = jnp.linspace(-0.7, 0.9, dimension)

    def function(value):
        return jnp.sum(jnp.sin(value) + 0.05 * value**3)

    def action_laplacian(value):
        evaluated = evaluate_fused_coordinate_derivatives(
            function,
            value,
            second_axes=tuple(range(dimension)),
        )
        total = jnp.asarray(0.0, dtype=value.dtype)
        for derivative in evaluated.diagonal_second_derivatives:
            total = total + derivative
        return total

    def dense_laplacian(value):
        return jnp.trace(jax.hessian(function)(value))

    action = jax.jit(action_laplacian)
    dense = jax.jit(dense_laplacian)
    action_value = action(point)
    dense_value = dense(point)
    action_compile, action_run = _seconds(action, point, repeats)
    dense_compile, dense_run = _seconds(dense, point, repeats)
    payload = {
        "dimension": dimension,
        "repeats": repeats,
        "absolute_error": float(jnp.abs(action_value - dense_value)),
        "action": {
            "compile_seconds": action_compile,
            "best_execution_seconds": action_run,
        },
        "dense": {
            "compile_seconds": dense_compile,
            "best_execution_seconds": dense_run,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
