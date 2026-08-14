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
        description="Benchmark approximate positive-feature balanced Sinkhorn."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=(128, 512, 2048))
    parser.add_argument("--ranks", type=int, nargs="+", default=(32, 128))
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _problem(size: int, dimension: int):
    indices = jnp.arange(size * dimension, dtype=float)
    source_points = jnp.reshape(jnp.sin(0.013 * indices), (size, dimension))
    target_points = jnp.reshape(jnp.cos(0.017 * indices), (size, dimension))
    weights = cx.Field(jnp.ones((size,), dtype=float), dims=("atom",))
    source = phx.integration.discrete(
        source_points,
        weights,
        axes="atom",
        normalized=True,
        provenance="positive-feature-benchmark-source",
    )
    target = phx.integration.discrete(
        target_points,
        weights,
        axes="atom",
        normalized=True,
        provenance="positive-feature-benchmark-target",
    )
    return phx.transport.discrete_problem(
        source,
        target,
        cost=phx.transport.SquaredEuclideanCost(),
    )


def _array_bytes(tree) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _record(
    size: int,
    rank: int,
    dimension: int,
    iterations: int,
    repeats: int,
    seed: int,
):
    problem = _problem(size, dimension)
    feature_map = phx.transport.GaussianPositiveFeatures(
        jax.random.key(seed),
        rank,
        num_probes=64,
    )
    solver = phx.transport.PositiveFeatureSinkhorn(
        0.5,
        feature_map,
        max_iterations=iterations,
        tolerance=1e-6,
        check_every=5,
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

    payload = jnp.reshape(
        jnp.sin(jnp.arange(size * 4, dtype=float)),
        (size, 4),
    )
    apply = eqx.filter_jit(lambda solved, values: solved.apply_source_to_target(values))
    applied = apply(result, payload)
    jax.block_until_ready(applied)
    started = time.perf_counter()
    for _ in range(repeats):
        applied = apply(result, payload)
        jax.block_until_ready(applied)
    plan_action_ms = 1e3 * (time.perf_counter() - started) / repeats

    def scalar(points):
        weights = cx.Field(jnp.ones((size,), dtype=float), dims=("atom",))
        source = phx.integration.discrete(
            points,
            weights,
            axes="atom",
            normalized=True,
            provenance="positive-feature-gradient-source",
        )
        target = phx.integration.discrete(
            problem.target.points,
            weights,
            axes="atom",
            normalized=True,
            provenance="positive-feature-gradient-target",
        )
        candidate = phx.transport.discrete_problem(
            source,
            target,
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

    if size <= 256:
        approximate_kernel = result.factors.kernel_matrix()
        exact_kernel = jnp.exp(-problem.cost_matrix() / result.epsilon)
        kernel_relative_error = jnp.linalg.norm(
            approximate_kernel - exact_kernel
        ) / jnp.linalg.norm(exact_kernel)
        kernel_error = float(kernel_relative_error)
        exact_solver = phx.transport.Sinkhorn(
            0.5,
            max_iterations=iterations,
            tolerance=1e-6,
            check_every=5,
        )
        exact_result = exact_solver(problem)
        approximate_plan = result.dense_plan()
        exact_plan = exact_result.dense_plan()
        plan_relative_error = jnp.linalg.norm(
            approximate_plan - exact_plan
        ) / jnp.linalg.norm(exact_plan)
        plan_error = float(plan_relative_error)
    else:
        kernel_error = None
        plan_error = None

    factor_bytes = _array_bytes(result.factors)
    result_bytes = _array_bytes(result)
    dense_plan_bytes = size * size * problem.source.points.dtype.itemsize
    return {
        "size": size,
        "dimension": dimension,
        "rank": rank,
        "seed": seed,
        "probe_count": int(result.approximation.num_probes),
        "approximation_status": int(result.approximation.status),
        "relative_probe_error": float(result.approximation.relative_probe_error),
        "maximum_relative_probe_error": float(
            result.approximation.maximum_relative_probe_error
        ),
        "kernel_relative_error": kernel_error,
        "dense_plan_relative_error": plan_error,
        "transport_status": int(result.diagnostics.status),
        "normalized_marginal_residual": float(
            result.diagnostics.normalized_marginal_residual
        ),
        "compile_first_ms": compile_first_ms,
        "steady_solve_ms": steady_ms,
        "plan_action_ms": plan_action_ms,
        "backward_ms": backward_ms,
        "factor_bytes": factor_bytes,
        "result_bytes": result_bytes,
        "dense_plan_bytes": dense_plan_bytes,
        "factor_to_dense_memory_ratio": factor_bytes / dense_plan_bytes,
        "plan_action_norm": float(jnp.linalg.norm(applied)),
        "gradient_norm": float(jnp.linalg.norm(gradient)),
        "gradient_finite": bool(jnp.all(jnp.isfinite(gradient))),
        "provenance_approximation": result.provenance.approximation,
    }


def main() -> None:
    arguments = _parser().parse_args()
    sizes = (16, 32) if arguments.smoke else tuple(arguments.sizes)
    ranks = (8, 16) if arguments.smoke else tuple(arguments.ranks)
    repeats = 1 if arguments.smoke else int(arguments.repeats)
    iterations = (
        min(int(arguments.iterations), 30)
        if arguments.smoke
        else int(arguments.iterations)
    )
    records = [
        _record(
            size,
            rank,
            int(arguments.dimension),
            iterations,
            repeats,
            int(arguments.seed),
        )
        for size in sizes
        for rank in ranks
    ]
    print(
        json.dumps(
            {
                "benchmark": "positive-feature-balanced-sinkhorn",
                "smoke": bool(arguments.smoke),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
