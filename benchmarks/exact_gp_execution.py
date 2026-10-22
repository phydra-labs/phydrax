"""Same-process exact GP execution evidence; raw JAX is benchmark-only reference.

Run as ``python -m benchmarks.exact_gp_execution --smoke``. ``--baseline-only``
never resolves the new execution-policy or prepared-mean exports and can run
against the pre-cutover package. No reference implementation lives in phydrax.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


_JITTER = 1e-8
_RTOL = 2e-7
_ATOL = 2e-8


def _kernel(parameters: jax.Array) -> Any:
    return phx.kernels.SquaredExponentialKernel(length_scale=jnp.exp(parameters[0]))


def _raw_factor(points: jax.Array, parameters: jax.Array) -> jax.Array:
    """Original raw-JAX covariance-plus-diagonal Cholesky, benchmark only."""
    kernel = _kernel(parameters)
    diagonal = jnp.broadcast_to(jnp.exp(parameters[1]) ** 2 + _JITTER, (len(points),))
    return jnp.linalg.cholesky(kernel.matrix(points, points) + jnp.diag(diagonal))


def _raw_likelihood(
    points: jax.Array, residual: jax.Array, parameters: jax.Array
) -> jax.Array:
    factor = _raw_factor(points, parameters)
    whitened = jsp.linalg.solve_triangular(factor, residual, lower=True)
    return -0.5 * (
        residual.size * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64))
        + 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))
        + whitened @ whitened
    )


def _raw_conditioner(
    points: jax.Array,
    query: jax.Array,
    factor: jax.Array,
    parameters: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Original query-side conditioner, including query covariance outputs."""
    kernel = _kernel(parameters)
    cross = kernel.matrix(query, points)
    projection = jsp.linalg.cho_solve((factor, True), cross.T).T
    covariance = kernel.matrix(query, query) - projection @ cross.T
    covariance = 0.5 * (covariance + covariance.T)
    return projection, covariance, jnp.maximum(jnp.diag(covariance), 0.0)


def _raw_query(
    points: jax.Array,
    query: jax.Array,
    factor: jax.Array,
    residual: jax.Array,
    parameters: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    projection, covariance, variance = _raw_conditioner(points, query, factor, parameters)
    return projection @ residual, projection, covariance, variance


def _mean_and_query_gradient(
    operation: Callable[[jax.Array], Any], query: jax.Array
) -> tuple[Any, jax.Array]:
    def summed(value: jax.Array) -> tuple[jax.Array, Any]:
        output = operation(value)
        mean = output[0] if isinstance(output, tuple) else output
        return jnp.sum(mean), output

    (_, output), gradient = jax.value_and_grad(summed, has_aux=True)(query)
    return output, gradient


def _profile(
    operation: Callable[..., Any],
    arguments: tuple[Any, ...],
    *,
    warmup: int,
    repeats: int,
) -> tuple[Any, dict[str, Any]]:
    # Clear in-memory tracing/executable caches before each cold phase. Persistent
    # cache policy is controlled externally and captured in the environment.
    jax.clear_caches()
    executable, compilation = measure_lower_and_compile(
        lambda: jax.jit(operation).lower(*arguments), lambda lowered: lowered.compile()
    )
    output, execution = measure_repeated(
        lambda: executable(*arguments), warmup=warmup, repeats=repeats
    )
    return output, {
        "compilation": asdict(compilation),
        "execution": execution.to_dict(),
        "compiler": asdict(
            compiler_evidence(
                executable.cost_analysis(), executable.memory_analysis(), source="jax"
            )
        ),
        "retained_bytes": logical_array_bytes(output),
    }


def _parity(actual: Any, expected: Any) -> dict[str, Any]:
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    if len(actual_leaves) != len(expected_leaves):
        raise ValueError("Parity comparison requires matching output leaves.")
    errors: list[float] = []
    relative_errors: list[float] = []
    passed = True
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        left = np.asarray(actual_leaf)
        right = np.asarray(expected_leaf)
        if left.shape != right.shape:
            raise ValueError("Parity comparison requires matching output shapes.")
        difference = np.abs(left - right)
        errors.append(float(np.max(difference, initial=0.0)))
        relative_errors.append(
            float(np.max(difference / np.maximum(np.abs(right), _ATOL), initial=0.0))
        )
        passed = passed and bool(np.allclose(left, right, rtol=_RTOL, atol=_ATOL))
    result = {
        "max_absolute_error": max(errors, default=0.0),
        "max_relative_error": max(relative_errors, default=0.0),
        "rtol": _RTOL,
        "atol": _ATOL,
        "passed": passed,
    }
    if not passed:
        raise ValueError(f"Exact GP numerical parity failed: {result}")
    return result


def _design(count: int, dimensions: int, *, offset: float) -> jax.Array:
    index = jnp.arange(count, dtype=jnp.float64)[:, None] + offset
    frequency = jnp.arange(1, dimensions + 1, dtype=jnp.float64)[None, :]
    return jnp.sin(index * frequency * 0.173) / jnp.sqrt(jnp.float64(dimensions))


def _case(
    n: int,
    d: int,
    q: int,
    *,
    capacities: tuple[int, ...],
    baseline_only: bool,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    points = _design(n, d, offset=0.0)
    query = _design(q, d, offset=0.37)
    residual = jnp.sin(points[:, 0] * 3.0) + jnp.float64(0.1) * jnp.cos(
        jnp.arange(n, dtype=jnp.float64)
    )
    parameters = jnp.log(jnp.asarray([0.4, 0.05], dtype=jnp.float64))
    rows: dict[str, Any] = {}

    def profile(operation: Callable[..., Any], *args: Any) -> tuple[Any, dict[str, Any]]:
        return _profile(operation, args, warmup=warmup, repeats=repeats)

    raw_factor, rows["raw_factor_preparation"] = profile(_raw_factor, points, parameters)
    raw_mll, rows["raw_mll"] = profile(_raw_likelihood, points, residual, parameters)
    raw_mll_gradient, rows["raw_mll_hyperparameter_gradient"] = profile(
        jax.value_and_grad(_raw_likelihood, argnums=2), points, residual, parameters
    )
    conditioner, rows["raw_query_conditioner_preparation"] = profile(
        _raw_conditioner, points, query, raw_factor, parameters
    )
    prepared_mean, rows["raw_prepared_conditioner_mean"] = profile(
        lambda projection, value: projection @ value, conditioner[0], residual
    )
    full_mean, rows["raw_full_conditioner_query"] = profile(
        _raw_query, points, query, raw_factor, residual, parameters
    )
    rows["raw_prepared_conditioner_mean"]["parity"] = _parity(prepared_mean, full_mean[0])
    full_gradient, rows["raw_full_conditioner_query_gradient"] = profile(
        lambda x, z, factor, value, theta: _mean_and_query_gradient(
            lambda design: _raw_query(x, design, factor, value, theta), z
        ),
        points,
        query,
        raw_factor,
        residual,
        parameters,
    )
    selected_mean, rows["raw_compiled_mean_only_query"] = profile(
        lambda x, z, factor, value, theta: _raw_query(x, z, factor, value, theta)[0],
        points,
        query,
        raw_factor,
        residual,
        parameters,
    )
    selected_gradient, rows["raw_compiled_mean_only_query_gradient"] = profile(
        lambda x, z, factor, value, theta: _mean_and_query_gradient(
            lambda design: _raw_query(x, design, factor, value, theta)[0], z
        ),
        points,
        query,
        raw_factor,
        residual,
        parameters,
    )
    rows["raw_compiled_mean_only_query"]["parity"] = _parity(selected_mean, full_mean[0])
    rows["raw_compiled_mean_only_query_gradient"]["parity"] = _parity(
        selected_gradient, (full_gradient[0][0], full_gradient[1])
    )
    candidates: list[dict[str, Any]] = []
    if not baseline_only:
        # Deliberately resolve new exports only in the candidate branch.
        policy_type = phx.uq.ExactGaussianProcessExecutionPolicy
        prepare_mean = phx.uq.prepare_gaussian_process_mean
        for capacity in capacities:
            workspace = 8 * ((8 * n + 16) + capacity * (10 * n + 2 * d + 8))
            mean_workspace = capacity * (n + d + 1) * 8
            execution = policy_type(
                max_workspace_bytes=workspace,
                max_factor_storage_bytes=512 * 1024 * 1024,
            )

            def native_factor(x: jax.Array, theta: jax.Array) -> Any:
                state = phx.uq.GaussianProcessLikelihoodState(
                    kernel=_kernel(theta),
                    noise_scale=jnp.exp(theta[1]),
                    jitter=jnp.float64(_JITTER),
                    execution=execution,
                )
                return phx.uq.ExactGaussianProcessFactor(x, state=state)

            def native_mll(x: jax.Array, value: jax.Array, theta: jax.Array) -> jax.Array:
                state = phx.uq.GaussianProcessLikelihoodState(
                    kernel=_kernel(theta),
                    noise_scale=jnp.exp(theta[1]),
                    jitter=jnp.float64(_JITTER),
                    execution=execution,
                )
                model = phx.uq.ExactGaussianProcessDiscrepancy(x, value)
                return model.log_marginal_likelihood(jnp.zeros_like(value), state=state)

            candidate: dict[str, Any] = {
                "requested_block_capacity": capacity,
                "workspace_budget_bytes": workspace,
                "mean_workspace_budget_bytes": mean_workspace,
                "resolved_covariance_column_capacity": execution.column_batch_size(
                    points
                ),
                "expected_mean_query_row_capacity": min(q, capacity),
            }
            factor, candidate["factor_preparation"] = profile(
                native_factor, points, parameters
            )
            mll, candidate["mll"] = profile(native_mll, points, residual, parameters)
            candidate["mll"]["parity"] = _parity(mll, raw_mll)
            mll_gradient, candidate["mll_hyperparameter_gradient"] = profile(
                jax.value_and_grad(native_mll, argnums=2), points, residual, parameters
            )
            candidate["mll_hyperparameter_gradient"]["parity"] = _parity(
                mll_gradient, raw_mll_gradient
            )
            predictor, candidate["mean_preparation"] = profile(
                lambda prepared, value: prepare_mean(
                    prepared, value, max_workspace_bytes=mean_workspace
                ),
                factor,
                residual,
            )
            candidate["mean_preparation"]["declared_retained_storage_bytes"] = (
                predictor.retained_storage_bytes
            )
            mean, candidate["mean_query"] = profile(
                lambda prepared, z: prepared(z), predictor, query
            )
            candidate["mean_query"]["parity"] = _parity(mean, full_mean[0])
            mean_gradient, candidate["mean_query_gradient"] = profile(
                lambda prepared, z: _mean_and_query_gradient(prepared, z),
                predictor,
                query,
            )
            candidate["mean_query_gradient"]["parity"] = _parity(
                mean_gradient, (full_gradient[0][0], full_gradient[1])
            )
            candidates.append(candidate)
    return {"n": n, "d": d, "q": q, "baseline": rows, "bounded_native": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--block-capacities", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    args = parser.parse_args()
    if len(set(args.block_capacities)) < 2 or min(args.block_capacities) < 1:
        parser.error("At least two distinct positive block capacities are required.")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats must be positive.")
    if os.environ.get("JAX_ENABLE_COMPILATION_CACHE", "").lower() not in ("false", "0"):
        parser.error(
            "Set JAX_ENABLE_COMPILATION_CACHE=false before process launch for cold compilation."
        )
    # One-axis controls distinguish observation, coordinate, and query scaling.
    cases = (
        [(16, 1, 7), (32, 1, 7), (16, 4, 7), (16, 1, 15)]
        if args.smoke
        else [
            (128, 4, 128),
            (256, 4, 128),
            (512, 4, 128),
            (256, 1, 128),
            (256, 16, 128),
            (256, 4, 32),
            (256, 4, 512),
        ]
    )
    records = [
        _case(
            n,
            d,
            q,
            capacities=tuple(args.block_capacities),
            baseline_only=args.baseline_only,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for n, d, q in cases
    ]
    write_json_atomic(
        args.output,
        {
            "baseline_only": args.baseline_only,
            "dtype": "float64",
            "hyperparameters": ["log_length_scale", "log_noise_scale"],
            "query_gradient_objective": "sum of residual predictive means",
            "jitter": _JITTER,
            "protocol": {
                "execution": "same process, sequential routes; no concurrent competitors",
                "cold": "clear in-memory JAX caches; lower and compile measured separately",
                "persistent_compilation_cache": "disabled externally via JAX_ENABLE_COMPILATION_CACHE=false",
                "preparation": "factorization separate from residual-to-predictor solve",
                "raw_reference": "benchmark-only original dense JAX formulas, not a production backend",
                "mean_only_baseline": "full old conditioner traced with only mean returned; permits compiler dead-output elimination",
                "retention": "logical unique array bytes, not allocator or physical RSS",
                "query_gradient": "reverse-mode gradient with respect to every query coordinate",
            },
            "environment": capture_environment().to_dict(),
            "identity": capture_benchmark_identity(
                Path(__file__).resolve().parents[1],
                Path(__file__),
                ("compiler", "compilation", "execution", "retained_bytes", "parity"),
            ).to_dict(),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
