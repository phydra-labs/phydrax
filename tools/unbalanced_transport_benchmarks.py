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
        description="Benchmark dense and blockwise generalized unbalanced Sinkhorn."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=(64, 256, 1024))
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _measure(points, weights, *, provenance):
    return phx.integration.discrete(
        points,
        cx.Field(weights, dims=("atom",)),
        axes="atom",
        normalized=False,
        provenance=provenance,
    )


def _problem(size: int, dimension: int):
    coordinates = jnp.arange(size * dimension, dtype=float)
    source_points = jnp.sin(0.013 * coordinates).reshape((size, dimension))
    target_points = jnp.cos(0.017 * coordinates).reshape((size, dimension))
    source_weights = 0.5 + jnp.linspace(0.0, 1.0, size)
    target_weights = 0.25 + jnp.linspace(1.0, 0.0, size)
    return phx.transport.unbalanced_problem(
        _measure(
            source_points,
            source_weights,
            provenance="unbalanced-benchmark-source",
        ),
        _measure(
            target_points,
            target_weights,
            provenance="unbalanced-benchmark-target",
        ),
        cost=phx.transport.SquaredEuclideanCost(),
        source_marginal_penalty=1.0,
        target_marginal_penalty=2.0,
    )


def _bytes(tree) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _record(size, dimension, block_size, iterations, repeats, *, blockwise):
    problem = _problem(size, dimension)
    solver = phx.transport.UnbalancedSinkhorn(
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
        candidate = phx.transport.UnbalancedTransportProblem(
            eqx.tree_at(lambda measure: measure.points, problem.source, points),
            problem.target,
            problem.cost,
            source_marginal_penalty=problem.source_marginal_penalty,
            target_marginal_penalty=problem.target_marginal_penalty,
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
        "converged": bool(result.converged),
        "mass_collapsed": bool(result.mass_collapsed),
        "source_mass": float(problem.source_mass),
        "target_mass": float(problem.target_mass),
        "transported_mass": float(result.transported_mass),
        "transport_cost": float(result.transport_cost),
        "entropy_regularization": float(result.entropy_regularization),
        "source_marginal_regularization": float(result.source_marginal_regularization),
        "target_marginal_regularization": float(result.target_marginal_regularization),
        "regularized_cost": float(result.regularized_cost),
        "fixed_point_residual": float(result.diagnostics.fixed_point_residual),
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
    iterations = (
        min(int(arguments.iterations), 30)
        if arguments.smoke
        else int(arguments.iterations)
    )
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
    print(json.dumps({"schema_version": 1, "records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
