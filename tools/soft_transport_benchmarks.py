#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark differentiable native transport ordering operators."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=(64, 256, 1024))
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _record(size: int, epsilon: float, repeats: int):
    values = jnp.sin(jnp.arange(size, dtype=float) * 1.61803398875)
    sort = jax.jit(lambda candidate: phx.transport.soft_sort(candidate, epsilon=epsilon))

    started = time.perf_counter()
    ordered = sort(values)
    jax.block_until_ready(ordered)
    compile_first_ms = 1e3 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repeats):
        ordered = sort(values)
        jax.block_until_ready(ordered)
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats

    gradient_function = jax.jit(
        jax.grad(
            lambda candidate: jnp.sum(
                phx.transport.soft_sort(candidate, epsilon=epsilon) ** 2
            )
        )
    )
    gradient = gradient_function(values)
    jax.block_until_ready(gradient)
    started = time.perf_counter()
    gradient = gradient_function(values)
    jax.block_until_ready(gradient)
    backward_ms = 1e3 * (time.perf_counter() - started)

    hard = jnp.sort(values)
    ranks = phx.transport.soft_rank(values, epsilon=epsilon)
    quantiles = phx.transport.soft_quantile(
        values,
        jnp.asarray([0.1, 0.5, 0.9]),
        epsilon=epsilon,
    )
    return {
        "size": size,
        "epsilon": epsilon,
        "compile_first_ms": compile_first_ms,
        "steady_ms": steady_ms,
        "backward_ms": backward_ms,
        "relative_hard_error": float(
            jnp.linalg.norm(ordered - hard) / jnp.maximum(jnp.linalg.norm(hard), 1e-12)
        ),
        "monotonicity_violations": int(jnp.sum(jnp.diff(ordered) < 0.0)),
        "range_violations": int(
            jnp.sum((ordered < jnp.min(values)) | (ordered > jnp.max(values)))
        ),
        "rank_sum_error": float(
            jnp.abs(jnp.sum(ranks) - 0.5 * size * (size - 1))
        ),
        "quantiles": [float(value) for value in quantiles],
        "gradient_norm": float(jnp.linalg.norm(gradient)),
    }


def main() -> None:
    arguments = _parser().parse_args()
    sizes = (8, 16) if arguments.smoke else tuple(arguments.sizes)
    repeats = 1 if arguments.smoke else int(arguments.repeats)
    records = [
        _record(size, float(arguments.epsilon), repeats)
        for size in sizes
    ]
    print(json.dumps({"records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
