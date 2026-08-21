#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark dense and sparse relative-entropy moment calibration."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=(1_000, 10_000))
    parser.add_argument("--moments", type=int, nargs="+", default=(4, 16, 64))
    parser.add_argument(
        "--execution",
        choices=("dense", "sparse", "both"),
        default="both",
    )
    parser.add_argument(
        "--target-kind",
        choices=("exact", "quadratic", "both"),
        default="both",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _dense_operator(size: int, moments: int, seed: int):
    key = jr.fold_in(jr.key(seed), size * 10_000 + moments)
    return jr.normal(key, (size, moments)) / jnp.sqrt(float(moments))


def _sparse_operator(size: int, moments: int):
    source_indices = jnp.arange(size, dtype=jnp.int32)
    target_indices = source_indices % moments
    relation = phx.sparse.EdgeRelation(
        source_indices,
        target_indices,
        source_size=size,
        target_size=moments,
    )
    return phx.sparse.SparseLinearMap(
        relation,
        jnp.ones((size,), dtype=float),
    )


def _problem(size: int, moments: int, execution: str, target_kind: str, seed: int):
    moment_map = (
        _dense_operator(size, moments, seed)
        if execution == "dense"
        else _sparse_operator(size, moments)
    )
    prior_logits = -0.1 * jnp.sin(jnp.arange(size, dtype=float) * 0.017)
    known_dual = 0.15 * jnp.cos(jnp.arange(moments, dtype=float) * 0.31)
    known_weights = jax.nn.softmax(
        prior_logits + moment_map.transpose_mv(known_dual)
        if not isinstance(moment_map, jax.Array)
        else prior_logits + moment_map @ known_dual
    )
    achieved = (
        moment_map.mv(known_weights)
        if not isinstance(moment_map, jax.Array)
        else moment_map.T @ known_weights
    )
    if target_kind == "exact":
        target = phx.weighting.ExactMoments(achieved)
    else:
        scale = jnp.full((moments,), 0.2)
        target = phx.weighting.QuadraticMoments(
            achieved + scale**2 * known_dual,
            scale=scale,
        )
    return phx.weighting.MomentCalibrationProblem(
        moment_map,
        target,
        prior_log_weights=prior_logits,
    )


def _nearby_problem(problem, target_kind: str):
    perturbation = 0.002 * jnp.cos(
        jnp.arange(problem.moment_count, dtype=problem.target.values.dtype) * 0.23
    )
    if target_kind == "exact":
        nearby_dual = 0.152 * jnp.cos(
            jnp.arange(
                problem.moment_count,
                dtype=problem.target.values.dtype,
            )
            * 0.31
        )
        nearby_weights = jax.nn.softmax(
            problem.prior_log_weights + problem.moment_map.transpose_mv(nearby_dual)
        )
        target = phx.weighting.ExactMoments(problem.moment_map.mv(nearby_weights))
    else:
        target = phx.weighting.QuadraticMoments(
            problem.target.values + perturbation,
            scale=problem.target.scale,
        )
    return phx.weighting.MomentCalibrationProblem(
        problem.moment_map,
        target,
        prior_log_weights=problem.prior_log_weights,
        mask=problem.mask,
    )


def _bytes(tree) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _record(size, moments, execution, target_kind, repeats, seed):
    setup_started = time.perf_counter()
    problem = _problem(size, moments, execution, target_kind, seed)
    nearby = _nearby_problem(problem, target_kind)
    setup_ms = 1e3 * (time.perf_counter() - setup_started)
    termination = phx.optim.OptimizationTermination(
        absolute_optimality=1e-7,
        relative_optimality=0.0,
        maximum_steps=100,
    )
    method = phx.optim.NewtonKrylov()
    compiled = eqx.filter_jit(
        lambda candidate, initial: phx.weighting.calibrate_moments(
            candidate,
            method=method,
            termination=termination,
            initial_dual=initial,
        )
    )
    initial = jnp.zeros((moments,), dtype=problem.target.values.dtype)

    started = time.perf_counter()
    result = compiled(problem, initial)
    jax.block_until_ready(result.log_weights)
    compile_first_ms = 1e3 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repeats):
        result = compiled(problem, initial)
        jax.block_until_ready(result.log_weights)
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats

    started = time.perf_counter()
    cold_nearby = compiled(nearby, initial)
    jax.block_until_ready(cold_nearby.log_weights)
    cold_nearby_ms = 1e3 * (time.perf_counter() - started)

    started = time.perf_counter()
    warm_nearby = compiled(nearby, result.dual_variables)
    jax.block_until_ready(warm_nearby.log_weights)
    warm_nearby_ms = 1e3 * (time.perf_counter() - started)

    diagnostics = result.diagnostics
    return {
        "source_points": size,
        "moment_count": moments,
        "execution": execution,
        "target_kind": target_kind,
        "setup_ms": setup_ms,
        "compile_first_ms": compile_first_ms,
        "steady_ms": steady_ms,
        "cold_nearby_ms": cold_nearby_ms,
        "warm_nearby_ms": warm_nearby_ms,
        "iterations": int(diagnostics.optimization.iterations),
        "cold_nearby_iterations": int(cold_nearby.diagnostics.optimization.iterations),
        "warm_nearby_iterations": int(warm_nearby.diagnostics.optimization.iterations),
        "linear_iterations": int(diagnostics.optimization.linear_iterations),
        "status": int(result.status),
        "optimizer_status": int(diagnostics.optimizer_status),
        "affine_rank": int(diagnostics.numerical_affine_rank),
        "active_support": int(diagnostics.active_support),
        "maximum_scaled_residual": float(diagnostics.maximum_scaled_residual),
        "relative_entropy": float(diagnostics.relative_entropy),
        "effective_sample_size": float(diagnostics.effective_sample_size),
        "problem_bytes": _bytes(problem),
        "result_bytes": _bytes(result),
    }


def main() -> None:
    arguments = _parser().parse_args()
    sizes = (128,) if arguments.smoke else tuple(arguments.sizes)
    moments = (4,) if arguments.smoke else tuple(arguments.moments)
    repeats = 1 if arguments.smoke else int(arguments.repeats)
    executions = (
        ("dense", "sparse") if arguments.execution == "both" else (arguments.execution,)
    )
    target_kinds = (
        ("exact", "quadratic")
        if arguments.target_kind == "both"
        else (arguments.target_kind,)
    )
    records = [
        _record(size, count, execution, target_kind, repeats, arguments.seed)
        for size in sizes
        for count in moments
        for execution in executions
        for target_kind in target_kinds
    ]
    print(json.dumps({"records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
