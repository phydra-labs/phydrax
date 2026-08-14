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
        description="Benchmark native fixed- and free-support transport barycenters."
    )
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--support-atoms", type=int, default=64)
    parser.add_argument("--measures", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--outer-iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _measure(points, weights, *, provenance):
    return phx.integration.discrete(
        points,
        cx.Field(weights, dims=("atom",)),
        axes="atom",
        normalized=True,
        provenance=provenance,
    )


def _problem(
    atoms: int,
    support_atoms: int,
    num_measures: int,
    dimension: int,
):
    measures = []
    for index in range(num_measures):
        count = atoms - index % max(1, min(atoms // 4, num_measures))
        coordinates = jnp.arange(count * dimension, dtype=float).reshape(
            (count, dimension)
        )
        points = jnp.sin(0.013 * coordinates + 0.2 * index)
        raw_weights = 1.0 + jnp.cos(0.017 * jnp.arange(count, dtype=float) + index) ** 2
        measures.append(
            _measure(
                points,
                raw_weights,
                provenance=f"barycenter-benchmark-measure-{index}",
            )
        )
    support_coordinates = jnp.arange(support_atoms * dimension, dtype=float).reshape(
        (support_atoms, dimension)
    )
    support = _measure(
        jnp.cos(0.019 * support_coordinates),
        jnp.ones((support_atoms,), dtype=float),
        provenance="barycenter-benchmark-initial-support",
    )
    return phx.transport.fixed_support_barycenter_problem(
        tuple(measures),
        support,
        measure_weights=jnp.ones((num_measures,), dtype=float) / num_measures,
        cost=phx.transport.SquaredEuclideanCost(),
    )


def _bytes(tree) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _fixed_record(problem, iterations, block_size, repeats, *, blockwise):
    solver = phx.transport.SinkhornBarycenter(
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
    jax.block_until_ready(result.objective)
    compile_first_ms = 1e3 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repeats):
        result = compiled(problem)
        jax.block_until_ready(result.objective)
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats

    def objective(points):
        candidate = eqx.tree_at(
            lambda item: item.measure_points,
            problem,
            problem.measure_points.at[0].set(points),
        )
        return solver(candidate).objective

    differentiated = eqx.filter_jit(jax.grad(objective))
    gradient = differentiated(problem.measure_points[0])
    jax.block_until_ready(gradient)
    started = time.perf_counter()
    gradient = differentiated(problem.measure_points[0])
    jax.block_until_ready(gradient)
    backward_ms = 1e3 * (time.perf_counter() - started)

    return {
        "family": "fixed-support",
        "execution": "blockwise" if blockwise else "dense",
        "block_size": block_size if blockwise else None,
        "measures": problem.num_measures,
        "padded_atoms": problem.padded_atom_count,
        "support_atoms": problem.support_atom_count,
        "dimension": problem.feature_size,
        "iterations": int(result.diagnostics.num_iterations),
        "status": int(result.diagnostics.status),
        "normalized_marginal_residual": float(
            result.diagnostics.normalized_marginal_residual
        ),
        "objective": float(result.objective),
        "compile_first_ms": compile_first_ms,
        "steady_ms": steady_ms,
        "backward_ms": backward_ms,
        "gradient_norm": float(jnp.linalg.norm(gradient)),
        "result_bytes": _bytes(result),
        "approximate": result.approximate,
    }


def _free_record(problem, inner_iterations, outer_iterations):
    inner = phx.transport.SinkhornBarycenter(
        0.5,
        max_iterations=inner_iterations,
        tolerance=1e-6,
        check_every=5,
        early_stop=False,
    )
    solver = phx.transport.FreeSupportBarycenter(
        inner,
        max_iterations=outer_iterations,
        tolerance=1e-5,
    )
    compiled = eqx.filter_jit(lambda candidate: solver(candidate))
    started = time.perf_counter()
    result = compiled(problem)
    jax.block_until_ready(result.barycenter.objective)
    elapsed_ms = 1e3 * (time.perf_counter() - started)
    return {
        "family": "free-support",
        "execution": "dense",
        "measures": problem.num_measures,
        "padded_atoms": problem.padded_atom_count,
        "support_atoms": problem.support_atom_count,
        "dimension": problem.feature_size,
        "inner_iterations": inner_iterations,
        "outer_iterations": int(result.diagnostics.num_iterations),
        "retained_inner_solves": len(result.inner_results),
        "status": int(result.diagnostics.status),
        "objective": float(result.barycenter.objective),
        "support_displacement": float(
            result.diagnostics.support_displacement_history[-1]
        ),
        "compile_and_solve_ms": elapsed_ms,
        "result_bytes": _bytes(result),
        "local_optimization": result.provenance.local_optimization,
        "approximate": result.approximate,
    }


def main() -> None:
    arguments = _parser().parse_args()
    atoms = min(arguments.atoms, 16) if arguments.smoke else arguments.atoms
    support_atoms = (
        min(arguments.support_atoms, 4) if arguments.smoke else arguments.support_atoms
    )
    num_measures = min(arguments.measures, 3) if arguments.smoke else arguments.measures
    iterations = (
        min(arguments.iterations, 20) if arguments.smoke else arguments.iterations
    )
    outer_iterations = (
        min(arguments.outer_iterations, 2)
        if arguments.smoke
        else arguments.outer_iterations
    )
    repeats = 1 if arguments.smoke else arguments.repeats
    problem = _problem(
        atoms,
        support_atoms,
        num_measures,
        arguments.dimension,
    )
    records = [
        _fixed_record(
            problem,
            iterations,
            arguments.block_size,
            repeats,
            blockwise=False,
        ),
        _fixed_record(
            problem,
            iterations,
            arguments.block_size,
            repeats,
            blockwise=True,
        ),
        _free_record(problem, iterations, outer_iterations),
    ]
    print(
        json.dumps(
            {
                "benchmark": "transport-barycenters",
                "smoke": bool(arguments.smoke),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
