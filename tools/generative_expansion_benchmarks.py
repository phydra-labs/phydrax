#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Statistical correctness and runtime qualification for generative expansions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _ready(value):
    return jax.block_until_ready(value)


def _empirical_covariance(samples):
    centered = samples - jnp.mean(samples, axis=0)
    return centered.T @ centered / samples.shape[0]


def benchmark_matrix_transition(*, sample_count: int):
    process = phx.stochastic.MatrixGaussianDiffusion(
        jnp.asarray([[-0.2, 0.1], [-0.05, -0.4]]),
        jnp.asarray([[0.7, 0.1], [0.0, 0.5]]),
        offset=jnp.asarray([0.1, -0.05]),
        process_id="generative-benchmark-matrix",
    )
    law = process.marginal_transition(jnp.asarray([0.4, -0.3]), t0=0.0, t1=0.8)
    start = perf_counter()
    samples = law.sample(jr.key(0), (sample_count,))
    _ready(samples)
    seconds = perf_counter() - start
    mean_error = jnp.linalg.vector_norm(jnp.mean(samples, axis=0) - law.mean)
    covariance_error = jnp.linalg.matrix_norm(
        _empirical_covariance(samples) - law.covariance
    ) / jnp.linalg.matrix_norm(law.covariance)
    return {
        "case": "exact-matrix-gaussian-transition",
        "sample_count": sample_count,
        "mean_error": float(mean_error),
        "covariance_relative_error": float(covariance_error),
        "seconds": seconds,
        "passed": bool(mean_error < 0.05 and covariance_error < 0.08),
    }


def benchmark_categorical_corruption(*, sample_count: int):
    schedule = phx.stochastic.CategoricalDiffusionSchedule.uniform(
        32, 5, beta_start=0.01, beta_end=0.08
    )
    clean = jnp.zeros((sample_count,), dtype=jnp.int32)
    timestep = 23
    expected = schedule.marginal_probabilities(jnp.asarray(0, dtype=jnp.int32), timestep)
    start = perf_counter()
    samples = schedule.corrupt(clean, timestep, jr.key(1))
    _ready(samples)
    seconds = perf_counter() - start
    frequency = jnp.bincount(samples, length=schedule.num_classes) / sample_count
    maximum_error = jnp.max(jnp.abs(frequency - expected))
    return {
        "case": "categorical-kernel-frequency",
        "sample_count": sample_count,
        "maximum_probability_error": float(maximum_error),
        "seconds": seconds,
        "passed": bool(maximum_error < 0.035),
    }


def benchmark_subspace_law(*, sample_count: int):
    basis = jnp.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0], [0.5, 0.5]])
    layout = phx.stochastic.AffineSubspaceLayout(
        jnp.asarray([0.2, -0.1, 0.3, 0.0]), basis, event_shape=(4,)
    )
    covariance = jnp.asarray([[0.8, 0.1], [0.1, 0.6]])
    coefficient_law = phx.uq.GaussianFactorLaw(
        jnp.asarray([0.4, -0.2]),
        phx.uq.GaussianFactor(jnp.linalg.cholesky(covariance)),
        event_shape=(2,),
    )
    law = phx.stochastic.SubspaceGaussianLaw(layout, coefficient_law)
    start = perf_counter()
    samples = law.sample(jr.key(2), (sample_count,))
    log_prob = law.log_prob(samples)
    _ready(log_prob)
    seconds = perf_counter() - start
    coefficients, residual = layout.project(samples)
    maximum_residual = jnp.max(residual)
    reconstruction_error = jnp.max(
        jnp.abs(layout.synthesize(coefficients) - samples)
    )
    return {
        "case": "hausdorff-subspace-gaussian",
        "sample_count": sample_count,
        "maximum_support_residual": float(maximum_residual),
        "maximum_reconstruction_error": float(reconstruction_error),
        "seconds": seconds,
        "passed": bool(
            jnp.all(jnp.isfinite(log_prob))
            and maximum_residual < 1e-5
            and reconstruction_error < 1e-5
        ),
    }


def benchmark_complex_law(*, sample_count: int):
    location = jnp.asarray([0.3 + 0.2j, -0.1 + 0.4j])
    variance = jnp.asarray([1.0, 0.6])
    law = phx.stochastic.ComplexNormalLaw(location, variance, event_shape=(2,))
    start = perf_counter()
    samples = law.sample(jr.key(3), (sample_count,))
    log_prob = law.log_prob(samples)
    _ready(log_prob)
    seconds = perf_counter() - start
    mean_error = jnp.linalg.vector_norm(jnp.mean(samples, axis=0) - location)
    empirical_variance = jnp.mean(jnp.abs(samples - jnp.mean(samples, axis=0)) ** 2, axis=0)
    variance_error = jnp.linalg.vector_norm(empirical_variance - variance) / jnp.linalg.vector_norm(
        variance
    )
    return {
        "case": "proper-complex-gaussian",
        "sample_count": sample_count,
        "mean_error": float(mean_error),
        "variance_relative_error": float(variance_error),
        "seconds": seconds,
        "passed": bool(
            jnp.all(jnp.isfinite(log_prob)) and mean_error < 0.05 and variance_error < 0.06
        ),
    }


def run_generative_expansion_benchmarks(*, quick: bool = False):
    sample_count = 2_048 if quick else 16_384
    cases = (
        benchmark_matrix_transition(sample_count=sample_count),
        benchmark_categorical_corruption(sample_count=sample_count),
        benchmark_subspace_law(sample_count=sample_count),
        benchmark_complex_law(sample_count=sample_count),
    )
    return {
        "benchmark": "generative-expansion",
        "backend": jax.default_backend(),
        "quick": bool(quick),
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    report = run_generative_expansion_benchmarks(quick=arguments.quick)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n")
        print(arguments.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
