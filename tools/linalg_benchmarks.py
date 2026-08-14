#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _block(tree: Any, /) -> None:
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(
    operation: Callable[[], Any], /, *, repeats: int
) -> tuple[Any, float, float]:
    result = operation()
    _block(result)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        _block(result)
        durations.append(1e3 * (time.perf_counter() - started))
    values = jnp.asarray(durations)
    return result, float(jnp.mean(values)), float(jnp.std(values))


def _environment() -> dict[str, Any]:
    device = jax.devices()[0]
    return {
        "backend": jax.default_backend(),
        "device_kind": device.device_kind,
        "jax_version": jax.__version__,
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "system": platform.system(),
        "system_release": platform.release(),
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
    }


def run_benchmarks(
    *,
    dense_size: int,
    right_hand_sides: int,
    iterative_size: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Benchmark direct, prepared, matrix-free, and sparse-derivative solve paths."""
    key = jr.key(seed)
    dense_key, rhs_key = jr.split(key)
    raw = jr.normal(dense_key, (dense_size, dense_size), dtype=jnp.float64)
    matrix = raw @ raw.T + dense_size * jnp.eye(dense_size)
    rhs = jr.normal(
        rhs_key,
        (dense_size, right_hand_sides),
        dtype=jnp.float64,
    )
    properties = phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )
    operator = phx.linalg.DenseLinearOperator(matrix, properties=properties)
    problem = phx.linalg.LinearSystem(operator)
    policy = phx.linalg.LinearSolvePolicy(phx.linalg.DenseCholesky())

    direct = jax.jit(jnp.linalg.solve)
    direct_value, direct_ms, direct_std = _measure(
        lambda: direct(matrix, rhs),
        repeats=repeats,
    )
    cold = jax.jit(
        lambda coefficients, targets: (
            phx.linalg.solve(
                phx.linalg.LinearSystem(
                    phx.linalg.DenseLinearOperator(
                        coefficients,
                        properties=properties,
                    )
                ),
                targets,
                policy=policy,
            ).value
        )
    )
    cold_started = time.perf_counter()
    cold_value = cold(matrix, rhs)
    _block(cold_value)
    cold_ms = 1e3 * (time.perf_counter() - cold_started)
    prepare_started = time.perf_counter()
    prepared = phx.linalg.prepare(problem, policy)
    _block(prepared)
    prepare_ms = 1e3 * (time.perf_counter() - prepare_started)
    reuse = jax.jit(lambda targets: phx.linalg.solve(prepared, targets).value)
    prepared_value, prepared_ms, prepared_std = _measure(
        lambda: reuse(rhs),
        repeats=repeats,
    )
    dense_residual = jnp.linalg.norm(matrix @ prepared_value - rhs) / jnp.linalg.norm(rhs)

    iterative_space = phx.linalg.ArraySpace((iterative_size,), dtype=jnp.float64)

    def stencil(vector):
        padded = jnp.pad(vector, (1, 1))
        return 4.0 * vector - padded[:-2] - padded[2:]

    iterative_operator = phx.linalg.FunctionLinearOperator(
        stencil,
        source=iterative_space,
        target=iterative_space,
        transpose_action=stencil,
        properties=properties,
    )
    iterative_problem = phx.linalg.LinearSystem(iterative_operator)
    iterative_policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.ConjugateGradient(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1e-9,
            absolute=1e-11,
            max_steps=100,
        ),
        preconditioner=phx.linalg.DiagonalPreconditioner(
            jnp.full((iterative_size,), 4.0),
            space=iterative_space,
        ),
    )
    iterative_prepared = phx.linalg.prepare(iterative_problem, iterative_policy)
    iterative_rhs = jnp.linspace(-1.0, 1.0, iterative_size)
    iterative_solve = jax.jit(
        lambda targets: phx.linalg.solve(iterative_prepared, targets)
    )
    iterative_result, iterative_ms, iterative_std = _measure(
        lambda: iterative_solve(iterative_rhs),
        repeats=repeats,
    )

    sparse_space = phx.linalg.ArraySpace((iterative_size,), dtype=jnp.float64)
    sparse_target = phx.linalg.ArraySpace((iterative_size - 1,), dtype=jnp.float64)

    def residual(values, _):
        return (values[1:] - values[:-1]) ** 2

    point = jnp.linspace(0.0, 1.0, iterative_size)
    direction = jnp.linspace(1.0, 2.0, iterative_size)
    residual_rows = jnp.repeat(jnp.arange(iterative_size - 1), 2)
    residual_cols = jnp.stack(
        (jnp.arange(iterative_size - 1), jnp.arange(1, iterative_size)),
        axis=1,
    ).reshape((-1,))
    jacobian_pattern = phx.sparse.SparsePattern.from_coo(
        residual_rows,
        residual_cols,
        (iterative_size - 1, iterative_size),
        origin="structural",
    )

    dense_jacobian_evaluate = jax.jit(lambda values: jax.jacfwd(residual)(values, None))
    dense_jacobian, dense_jacobian_ms, dense_jacobian_std = _measure(
        lambda: dense_jacobian_evaluate(point),
        repeats=repeats,
    )

    compile_started = time.perf_counter()
    native_jacobian = phx.sparse.compile_sparse_jacobian(
        residual,
        point,
        source=sparse_space,
        target=sparse_target,
        structure=jacobian_pattern,
        compiler="native",
    )
    _block(native_jacobian)
    native_jacobian_compile_ms = 1e3 * (time.perf_counter() - compile_started)

    compile_started = time.perf_counter()
    asdex_jacobian = phx.sparse.compile_sparse_jacobian(
        residual,
        point,
        source=sparse_space,
        target=sparse_target,
        compiler="asdex",
    )
    _block(asdex_jacobian)
    asdex_jacobian_compile_ms = 1e3 * (time.perf_counter() - compile_started)

    native_jacobian_evaluate = jax.jit(
        lambda values: native_jacobian.coefficients(values)
    )
    native_coefficients, native_jacobian_ms, native_jacobian_std = _measure(
        lambda: native_jacobian_evaluate(point),
        repeats=repeats,
    )
    asdex_jacobian_evaluate = jax.jit(lambda values: asdex_jacobian.coefficients(values))
    asdex_coefficients, asdex_jacobian_ms, asdex_jacobian_std = _measure(
        lambda: asdex_jacobian_evaluate(point),
        repeats=repeats,
    )
    native_jacobian_action = jax.jit(
        lambda values, vector: native_jacobian.operator(values).mv(vector)
    )
    native_action, native_action_ms, native_action_std = _measure(
        lambda: native_jacobian_action(point, direction),
        repeats=repeats,
    )
    asdex_jacobian_action = jax.jit(
        lambda values, vector: asdex_jacobian.operator(values).mv(vector)
    )
    asdex_action, asdex_action_ms, asdex_action_std = _measure(
        lambda: asdex_jacobian_action(point, direction),
        repeats=repeats,
    )
    dense_pattern_coefficients = dense_jacobian[
        jacobian_pattern.rows, jacobian_pattern.cols
    ]
    dense_action = dense_jacobian @ direction
    jacobian_maximum_difference = jnp.max(
        jnp.abs(
            jnp.concatenate(
                (
                    native_coefficients - dense_pattern_coefficients,
                    asdex_coefficients - dense_pattern_coefficients,
                    native_action - dense_action,
                    asdex_action - dense_action,
                )
            )
        )
    )

    def energy(values, _):
        return jnp.sum((values[1:] - values[:-1]) ** 2) + jnp.sum(values**2)

    coordinates = jnp.arange(iterative_size)
    hessian_pattern = phx.sparse.SparsePattern.from_coo(
        jnp.concatenate((coordinates, coordinates[:-1], coordinates[1:])),
        jnp.concatenate((coordinates, coordinates[1:], coordinates[:-1])),
        (iterative_size, iterative_size),
        symmetric=True,
        origin="structural",
    )
    compile_started = time.perf_counter()
    native_hessian = phx.sparse.compile_sparse_hessian(
        energy,
        point,
        space=sparse_space,
        structure=hessian_pattern,
        compiler="native",
        properties=properties,
    )
    _block(native_hessian)
    native_hessian_compile_ms = 1e3 * (time.perf_counter() - compile_started)

    compile_started = time.perf_counter()
    asdex_hessian = phx.sparse.compile_sparse_hessian(
        energy,
        point,
        space=sparse_space,
        compiler="asdex",
        properties=properties,
    )
    _block(asdex_hessian)
    asdex_hessian_compile_ms = 1e3 * (time.perf_counter() - compile_started)

    native_hessian_evaluate = jax.jit(lambda values: native_hessian.coefficients(values))
    native_hessian_coefficients, native_hessian_ms, native_hessian_std = _measure(
        lambda: native_hessian_evaluate(point),
        repeats=repeats,
    )
    asdex_hessian_evaluate = jax.jit(lambda values: asdex_hessian.coefficients(values))
    asdex_hessian_coefficients, asdex_hessian_ms, asdex_hessian_std = _measure(
        lambda: asdex_hessian_evaluate(point),
        repeats=repeats,
    )
    native_hessian_action = jax.jit(
        lambda values, vector: native_hessian.operator(values).mv(vector)
    )
    native_hessian_value, native_hessian_action_ms, native_hessian_action_std = _measure(
        lambda: native_hessian_action(point, direction),
        repeats=repeats,
    )
    direct_hessian_action = jax.jit(
        lambda values, vector: jax.jvp(
            jax.grad(lambda current: energy(current, None)),
            (values,),
            (vector,),
        )[1]
    )
    direct_hessian_value, direct_hessian_ms, direct_hessian_std = _measure(
        lambda: direct_hessian_action(point, direction),
        repeats=repeats,
    )
    hessian_maximum_difference = jnp.maximum(
        jnp.max(jnp.abs(native_hessian_coefficients - asdex_hessian_coefficients)),
        jnp.max(jnp.abs(native_hessian_value - direct_hessian_value)),
    )

    hessian_operator = native_hessian.operator(point)
    hessian_diagonal = jnp.full((iterative_size,), 6.0).at[0].set(4.0).at[-1].set(4.0)
    sparse_policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.ConjugateGradient(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1e-9,
            absolute=1e-11,
            max_steps=100,
        ),
        preconditioner=phx.linalg.DiagonalPreconditioner(
            hessian_diagonal,
            space=sparse_space,
        ),
    )
    sparse_rhs = jnp.linspace(-1.0, 1.0, iterative_size)
    sparse_solve = jax.jit(
        lambda targets: phx.linalg.solve(
            phx.linalg.LinearSystem(hessian_operator),
            targets,
            policy=sparse_policy,
        )
    )
    sparse_solve_result, sparse_solve_ms, sparse_solve_std = _measure(
        lambda: sparse_solve(sparse_rhs),
        repeats=repeats,
    )
    sparse_solve_residual = jnp.linalg.norm(
        hessian_operator.mv(sparse_solve_result.value) - sparse_rhs
    ) / jnp.linalg.norm(sparse_rhs)
    sparse_maximum_difference = jnp.maximum(
        jacobian_maximum_difference,
        hessian_maximum_difference,
    )
    sparse_result = {
        "jacobian": {
            "shape": list(jacobian_pattern.shape),
            "nnz": jacobian_pattern.nnz,
            "dense_jax_evaluation_mean_ms": dense_jacobian_ms,
            "dense_jax_evaluation_standard_deviation_ms": dense_jacobian_std,
            "native_compile_ms": native_jacobian_compile_ms,
            "native_num_colors": native_jacobian.num_colors,
            "native_evaluation_mean_ms": native_jacobian_ms,
            "native_evaluation_standard_deviation_ms": native_jacobian_std,
            "native_action_mean_ms": native_action_ms,
            "native_action_standard_deviation_ms": native_action_std,
            "asdex_compile_ms": asdex_jacobian_compile_ms,
            "asdex_num_colors": asdex_jacobian.num_colors,
            "asdex_evaluation_mean_ms": asdex_jacobian_ms,
            "asdex_evaluation_standard_deviation_ms": asdex_jacobian_std,
            "asdex_action_mean_ms": asdex_action_ms,
            "asdex_action_standard_deviation_ms": asdex_action_std,
            "maximum_value_difference": float(jacobian_maximum_difference),
        },
        "hessian": {
            "shape": list(hessian_pattern.shape),
            "nnz": hessian_pattern.nnz,
            "native_compile_ms": native_hessian_compile_ms,
            "native_num_colors": native_hessian.num_colors,
            "native_evaluation_mean_ms": native_hessian_ms,
            "native_evaluation_standard_deviation_ms": native_hessian_std,
            "native_action_mean_ms": native_hessian_action_ms,
            "native_action_standard_deviation_ms": native_hessian_action_std,
            "asdex_compile_ms": asdex_hessian_compile_ms,
            "asdex_num_colors": asdex_hessian.num_colors,
            "asdex_evaluation_mean_ms": asdex_hessian_ms,
            "asdex_evaluation_standard_deviation_ms": asdex_hessian_std,
            "direct_hvp_mean_ms": direct_hessian_ms,
            "direct_hvp_standard_deviation_ms": direct_hessian_std,
            "maximum_value_difference": float(hessian_maximum_difference),
        },
        "solve": {
            "mean_ms": sparse_solve_ms,
            "standard_deviation_ms": sparse_solve_std,
            "iterations": int(sparse_solve_result.diagnostics.iterations),
            "matvec_count": int(sparse_solve_result.diagnostics.matvec_count),
            "adjoint_matvec_count": int(
                sparse_solve_result.diagnostics.adjoint_matvec_count
            ),
            "relative_residual": float(sparse_solve_residual),
            "status": int(sparse_solve_result.status),
        },
    }

    maximum_dense_difference = jnp.max(
        jnp.abs(jnp.stack((direct_value - cold_value, direct_value - prepared_value)))
    )
    return {
        "configuration": {
            "dense_size": dense_size,
            "right_hand_sides": right_hand_sides,
            "iterative_size": iterative_size,
            "repeats": repeats,
            "seed": seed,
        },
        "environment": _environment(),
        "dense": {
            "direct_jax_mean_ms": direct_ms,
            "direct_jax_standard_deviation_ms": direct_std,
            "phydrax_cold_compile_and_execute_ms": cold_ms,
            "prepare_ms": prepare_ms,
            "prepared_reuse_mean_ms": prepared_ms,
            "prepared_reuse_standard_deviation_ms": prepared_std,
            "maximum_value_difference": float(maximum_dense_difference),
            "relative_residual": float(dense_residual),
        },
        "matrix_free": {
            "dimension": iterative_size,
            "mean_ms": iterative_ms,
            "standard_deviation_ms": iterative_std,
            "iterations": int(iterative_result.diagnostics.iterations),
            "matvec_count": int(iterative_result.diagnostics.matvec_count),
            "adjoint_matvec_count": int(
                iterative_result.diagnostics.adjoint_matvec_count
            ),
            "relative_residual": float(iterative_result.diagnostics.relative_residual),
            "status": int(iterative_result.status),
        },
        "sparse_derivatives": sparse_result,
        "passed": bool(
            maximum_dense_difference < 1e-10
            and dense_residual < 1e-10
            and iterative_result.successful
            and sparse_maximum_difference < 1e-10
            and sparse_solve_result.successful
            and sparse_solve_residual < 1e-8
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the shared Phydrax linear algebra runtime."
    )
    parser.add_argument("--dense-size", type=int, default=128)
    parser.add_argument("--right-hand-sides", type=int, default=16)
    parser.add_argument("--iterative-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if (
        min(
            args.dense_size,
            args.right_hand_sides,
            args.iterative_size,
            args.repeats,
        )
        <= 0
    ):
        parser.error("all benchmark sizes and repeats must be positive")
    if args.iterative_size < 2:
        parser.error("iterative-size must be at least two")
    print(
        json.dumps(
            run_benchmarks(
                dense_size=args.dense_size,
                right_hand_sides=args.right_hand_sides,
                iterative_size=args.iterative_size,
                repeats=args.repeats,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
