#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark native dense and blockwise balanced Sinkhorn transport."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=(64, 256, 1024))
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _problem(size: int, dimension: int):
    source_points = jnp.reshape(
        jnp.sin(jnp.arange(size * dimension, dtype=float) * 0.013),
        (size, dimension),
    )
    target_points = jnp.reshape(
        jnp.cos(jnp.arange(size * dimension, dtype=float) * 0.017),
        (size, dimension),
    )
    weights = cx.Field(jnp.ones((size,), dtype=float), dims=("atom",))
    source = phx.integration.discrete(
        source_points,
        weights,
        axes="atom",
        normalized=True,
        provenance="transport-benchmark-source",
    )
    target = phx.integration.discrete(
        target_points,
        weights,
        axes="atom",
        normalized=True,
        provenance="transport-benchmark-target",
    )
    return phx.transport.discrete_problem(
        source,
        target,
        cost=phx.transport.SquaredEuclideanCost(),
    )


def _bytes(tree) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _record(size, dimension, block_size, iterations, repeats, *, blockwise):
    problem = _problem(size, dimension)
    solver = phx.transport.Sinkhorn(
        0.5,
        max_iterations=iterations,
        tolerance=1e-6,
        check_every=5,
        block_size=block_size if blockwise else None,
        early_stop=False,
    )
    compiled = eqx.filter_jit(lambda candidate: solver(candidate))
    started = time.perf_counter()
    result = compiled(problem)
    jax.block_until_ready(result.regularized_cost)
    compile_first_ms = 1e3 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repeats):
        result = compiled(problem)
        jax.block_until_ready(result.regularized_cost)
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats

    payload = jnp.ones((size, 4), dtype=float)
    apply = eqx.filter_jit(lambda solved, values: solved.apply_source_to_target(values))
    applied = apply(result, payload)
    jax.block_until_ready(applied)
    started = time.perf_counter()
    for _ in range(repeats):
        applied = apply(result, payload)
        jax.block_until_ready(applied)
    apply_ms = 1e3 * (time.perf_counter() - started) / repeats

    def scalar(points):
        weights = cx.Field(jnp.ones((size,), dtype=float), dims=("atom",))
        source = phx.integration.discrete(
            points,
            weights,
            axes="atom",
            normalized=True,
        )
        candidate = phx.transport.discrete_problem(
            source,
            phx.integration.discrete(
                problem.target.points,
                weights,
                axes="atom",
                normalized=True,
            ),
            cost=phx.transport.SquaredEuclideanCost(),
        )
        return solver(candidate).regularized_cost

    differentiated = eqx.filter_jit(jax.grad(scalar))
    gradient = differentiated(problem.source.points)
    jax.block_until_ready(gradient)
    started = time.perf_counter()
    gradient = differentiated(problem.source.points)
    jax.block_until_ready(gradient)
    backward_ms = 1e3 * (time.perf_counter() - started)

    return {
        "size": size,
        "dimension": dimension,
        "execution": "blockwise" if blockwise else "dense",
        "block_size": block_size if blockwise else None,
        "iterations": int(result.diagnostics.num_iterations),
        "status": int(result.diagnostics.status),
        "normalized_marginal_residual": float(
            result.diagnostics.normalized_marginal_residual
        ),
        "primal_dual_gap": float(result.diagnostics.primal_dual_gap),
        "compile_first_ms": compile_first_ms,
        "steady_ms": steady_ms,
        "plan_apply_ms": apply_ms,
        "backward_ms": backward_ms,
        "result_bytes": _bytes(result),
        "gradient_norm": float(jnp.linalg.norm(gradient)),
    }


def main() -> None:
    arguments = _parser().parse_args()
    sizes = (16, 32) if arguments.smoke else tuple(arguments.sizes)
    repeats = 1 if arguments.smoke else int(arguments.repeats)
    iterations = min(int(arguments.iterations), 30) if arguments.smoke else int(arguments.iterations)
    records = []
    for size in sizes:
        records.append(
            _record(
                size,
                arguments.dimension,
                arguments.block_size,
                iterations,
                repeats,
                blockwise=False,
            )
        )
        records.append(
            _record(
                size,
                arguments.dimension,
                arguments.block_size,
                iterations,
                repeats,
                blockwise=True,
            )
        )
    print(json.dumps({"records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
