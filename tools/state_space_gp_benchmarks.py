#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark exact dense and temporal state-space Matérn Gaussian processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


@dataclass(frozen=True)
class StateSpaceGaussianProcessBenchmarkRecord:
    kernel: str
    training_size: int
    query_size: int
    schedule_size: int
    state_dimension: int
    dense_compile_seconds: float
    state_space_compile_seconds: float
    dense_execution_seconds: float
    state_space_execution_seconds: float
    dense_retained_elements: int
    state_space_retained_elements: int
    dense_retained_bytes: int
    state_space_retained_bytes: int
    log_marginal_likelihood_absolute_error: float
    posterior_mean_maximum_absolute_error: float
    posterior_variance_maximum_absolute_error: float
    evaluated_kernel_content_id: str | None
    prepared_kernel_content_id: str
    schedule_id: str
    method_id: str
    precision_evidence_id: str
    valid: bool


def _dense_gp(kernel, train_times, train_values, query_times, noise_scale):
    covariance = kernel.matrix(train_times, train_times)
    covariance = covariance + noise_scale**2 * jnp.eye(
        train_times.size, dtype=covariance.dtype
    )
    factor = jnp.linalg.cholesky(covariance)
    policy = phx.linalg.LinearSolvePolicy(phx.linalg.DenseCholesky())

    def solve(right_hand_side):
        operator = phx.linalg.DenseLinearOperator(
            covariance,
            properties=phx.linalg.OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                positive_semidefinite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                    "positive_semidefinite": "construction",
                },
            ),
        )
        return phx.linalg.solve(
            phx.linalg.LinearSystem(operator),
            right_hand_side,
            policy=policy,
        ).value

    alpha = solve(train_values)
    cross = kernel.matrix(query_times, train_times)
    solved_cross = jax.vmap(solve)(cross)
    posterior_mean = cross @ alpha
    posterior_variance = kernel.diagonal(query_times) - jnp.sum(
        cross * solved_cross, axis=1
    )
    log_marginal_likelihood = -0.5 * (
        train_values @ alpha
        + 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))
        + train_times.size * jnp.log(2.0 * jnp.pi)
    )
    return factor, log_marginal_likelihood, posterior_mean, posterior_variance


def _compile(function, *arguments, **keywords):
    started = perf_counter()
    compiled = function.lower(*arguments, **keywords).compile()
    elapsed = perf_counter() - started
    return compiled, elapsed


def _execution_seconds(function, *arguments, repeats, **keywords):
    samples = []
    value = None
    for _ in range(repeats):
        started = perf_counter()
        value = function(*arguments, **keywords)
        jax.block_until_ready(value)
        samples.append(perf_counter() - started)
    return value, float(median(samples))


def _retained_storage(value, /) -> tuple[int, int]:
    """Count each retained JAX array object once, including all nested results."""
    identities: set[int] = set()
    elements = 0
    bytes_ = 0
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array) and id(leaf) not in identities:
            identities.add(id(leaf))
            elements += int(leaf.size)
            bytes_ += int(leaf.size) * int(leaf.dtype.itemsize)
    return elements, bytes_


def _scaling_slope(sizes, values):
    coordinates = np.log(np.asarray(sizes, dtype=float))
    measurements = np.log(np.asarray(values, dtype=float))
    return float(np.polyfit(coordinates, measurements, 1)[0])


def _summaries(records):
    sizes = [record.training_size for record in records]
    return {
        "record_count": len(records),
        "training_sizes": sizes,
        "dense_execution_scaling_exponent": _scaling_slope(
            sizes, [record.dense_execution_seconds for record in records]
        ),
        "state_space_execution_scaling_exponent": _scaling_slope(
            sizes, [record.state_space_execution_seconds for record in records]
        ),
        "dense_storage_scaling_exponent": _scaling_slope(
            sizes, [record.dense_retained_elements for record in records]
        ),
        "state_space_storage_scaling_exponent": _scaling_slope(
            sizes, [record.state_space_retained_elements for record in records]
        ),
        "maximum_log_marginal_likelihood_absolute_error": max(
            record.log_marginal_likelihood_absolute_error for record in records
        ),
        "maximum_posterior_mean_absolute_error": max(
            record.posterior_mean_maximum_absolute_error for record in records
        ),
        "maximum_posterior_variance_absolute_error": max(
            record.posterior_variance_maximum_absolute_error for record in records
        ),
        "largest_dense_retained_bytes": records[-1].dense_retained_bytes,
        "largest_state_space_retained_bytes": records[-1].state_space_retained_bytes,
        "all_valid": all(record.valid for record in records),
    }


def _gate(summary):
    checks = {
        "all_state_space_results_valid": summary["all_valid"],
        "log_marginal_likelihood_absolute_error_at_most_2e-4": (
            summary["maximum_log_marginal_likelihood_absolute_error"] <= 2e-4
        ),
        "posterior_mean_absolute_error_at_most_2e-4": (
            summary["maximum_posterior_mean_absolute_error"] <= 2e-4
        ),
        "posterior_variance_absolute_error_at_most_2e-4": (
            summary["maximum_posterior_variance_absolute_error"] <= 2e-4
        ),
        "state_space_storage_exponent_at_most_1_1": (
            summary["state_space_storage_scaling_exponent"] <= 1.1
        ),
        "dense_storage_exponent_at_least_1_9": (
            summary["dense_storage_scaling_exponent"] >= 1.9
        ),
        "state_space_execution_exponent_at_most_1_5": (
            summary["state_space_execution_scaling_exponent"] <= 1.5
        ),
        "largest_state_space_storage_below_dense": (
            summary["largest_state_space_retained_bytes"]
            < summary["largest_dense_retained_bytes"]
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _source_provenance():
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "git_revision": revision,
        "working_tree_clean": not bool(diff),
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def run_state_space_gp_benchmarks(
    *,
    sizes=(128, 256, 512, 1024, 2048),
    repeats=5,
):
    resolved_sizes = tuple(int(size) for size in sizes)
    if len(resolved_sizes) < 3 or any(size <= 0 for size in resolved_sizes):
        raise ValueError("sizes must contain at least three positive schedule sizes.")
    if tuple(sorted(set(resolved_sizes))) != resolved_sizes:
        raise ValueError("sizes must be strictly increasing.")
    repeat_count = int(repeats)
    if repeat_count <= 0:
        raise ValueError("repeats must be positive.")

    kernel = phx.kernels.ScaleKernel(phx.kernels.Matern52Kernel(length_scale=0.37), 1.6)
    noise_scale = jnp.asarray(0.035)
    query_times = jnp.linspace(-0.2, 1.2, 65)
    records = []
    for size in resolved_sizes:
        increments = 0.2 + jnp.square(jnp.linspace(0.1, 1.0, size))
        train_times = jnp.cumsum(increments)
        train_times = train_times / train_times[-1]
        train_values = (
            jnp.sin(7.0 * train_times)
            + 0.2 * jnp.cos(19.0 * train_times)
            + 0.05 * train_times
        )
        plan = phx.uq.compile_state_space_kernel(
            kernel,
            phx.uq.StateSpaceGaussianProcessDesign(
                train_times[::-1],
                query_times[::-1],
            ),
            max_schedule_size=size + query_times.size,
        )
        reversed_values = train_values[::-1]
        dense_function = jax.jit(
            lambda times, values, queries, noise: _dense_gp(
                kernel, times, values, queries, noise
            )
        )
        state_space_function = eqx.filter_jit(phx.uq.fit_state_space_gaussian_process)
        dense_compiled, dense_compile_seconds = _compile(
            dense_function,
            train_times,
            train_values,
            query_times,
            noise_scale,
        )
        state_space_compiled, state_space_compile_seconds = _compile(
            state_space_function,
            plan,
            reversed_values,
            noise_scale=noise_scale,
        )
        dense_result, dense_execution_seconds = _execution_seconds(
            dense_compiled,
            train_times,
            train_values,
            query_times,
            noise_scale,
            repeats=repeat_count,
        )
        state_space_result, state_space_execution_seconds = _execution_seconds(
            state_space_compiled,
            plan,
            reversed_values,
            noise_scale=noise_scale,
            repeats=repeat_count,
        )
        dense_factor, dense_likelihood, dense_mean, dense_variance = dense_result
        state_mean = state_space_result.posterior_mean[::-1]
        state_variance = state_space_result.posterior_variance[::-1]
        dense_elements, dense_bytes = _retained_storage(dense_result)
        state_elements, state_bytes = _retained_storage(state_space_result)
        records.append(
            StateSpaceGaussianProcessBenchmarkRecord(
                kernel=kernel.kernel_id,
                training_size=size,
                query_size=int(query_times.size),
                schedule_size=plan.schedule_size,
                state_dimension=plan.state_dimension,
                dense_compile_seconds=dense_compile_seconds,
                state_space_compile_seconds=state_space_compile_seconds,
                dense_execution_seconds=dense_execution_seconds,
                state_space_execution_seconds=state_space_execution_seconds,
                dense_retained_elements=dense_elements,
                state_space_retained_elements=state_elements,
                dense_retained_bytes=dense_bytes,
                state_space_retained_bytes=state_bytes,
                log_marginal_likelihood_absolute_error=float(
                    jnp.abs(state_space_result.log_marginal_likelihood - dense_likelihood)
                ),
                posterior_mean_maximum_absolute_error=float(
                    jnp.max(jnp.abs(state_mean - dense_mean))
                ),
                posterior_variance_maximum_absolute_error=float(
                    jnp.max(jnp.abs(state_variance - dense_variance))
                ),
                evaluated_kernel_content_id=state_space_result.kernel_content_id,
                prepared_kernel_content_id=(
                    state_space_result.prepared_kernel_content_id
                ),
                schedule_id=state_space_result.schedule_id,
                method_id=state_space_result.method_id,
                precision_evidence_id=(state_space_result.precision_evidence.evidence_id),
                valid=bool(state_space_result.successful),
            )
        )

    summary = _summaries(records)
    return {
        "configuration": {
            "sizes": list(resolved_sizes),
            "query_size": int(query_times.size),
            "repeats": repeat_count,
            "kernel": kernel.kernel_id,
            "noise_scale": float(noise_scale),
            "timing_statistic": "median wall-clock seconds after explicit compilation",
            "storage_contract": (
                "complete unique retained JAX array objects for each returned "
                "dense/state-space result; repeated PyTree aliases count once"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "device": jax.devices()[0].device_kind,
            "x64": bool(jax.config.x64_enabled),
        },
        "source_provenance": _source_provenance(),
        "records": [asdict(record) for record in records],
        "summaries": summary,
        "gate": _gate(summary),
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser


def main():
    arguments = _parser().parse_args()
    sizes = (64, 128, 256) if arguments.quick else (128, 256, 512, 1024, 2048)
    report = run_state_space_gp_benchmarks(sizes=sizes, repeats=arguments.repeats)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
