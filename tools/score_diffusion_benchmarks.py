#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic correctness and runtime qualification for score diffusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


class _GaussianMarginalScore(eqx.Module):
    process: phx.stochastic.AbstractGaussianDiffusion
    mean: jax.Array
    variance: jax.Array

    def __call__(self, state, time):
        if isinstance(self.process, phx.stochastic.VariancePreservingDiffusion):
            slope = (
                self.process.beta_maximum - self.process.beta_minimum
            ) / self.process.terminal_time
            integrated = self.process.beta_minimum * time + 0.5 * slope * time**2
            mean_scale = jnp.exp(-0.5 * integrated)
            noise_variance = -jnp.expm1(-integrated)
        else:
            mean_scale = jnp.asarray(1.0, dtype=state.dtype)
            reference = self.process.initial_scale * jnp.exp(
                self.process.log_scale_ratio * time / self.process.terminal_time
            )
            noise_variance = reference**2 - self.process.initial_scale**2
        marginal_mean = mean_scale * self.mean
        marginal_variance = mean_scale**2 * self.variance + noise_variance
        return -(state - marginal_mean) / marginal_variance


def _score_function(process, mean, variance):
    dimension = process.dimension
    state = phx.domain.HyperRectangle(
        jnp.full((dimension,), -100.0),
        jnp.full((dimension,), 100.0),
        label="x",
    )
    domain = state @ phx.domain.TimeInterval(0.0, process.terminal_time)
    return domain.Function("x", "t")(_GaussianMarginalScore(process, mean, variance))


def _exact_terminal(process, mean, variance):
    terminal = process.terminal_time
    if isinstance(process, phx.stochastic.VariancePreservingDiffusion):
        mean_scale = process.transition_mean_scale(0.0, terminal)
        noise_scale = process.transition_scale(0.0, terminal)
    else:
        mean_scale = jnp.asarray(1.0)
        noise_scale = process.transition_scale(0.0, terminal)
    law = phx.uq.DiagonalNormalLaw(
        mean_scale * mean,
        jnp.sqrt(mean_scale**2 * variance + noise_scale**2),
        event_shape=process.state_shape,
    )
    return phx.stochastic.DiffusionTerminalReference(
        law,
        relationship="exact",
        residual_signal_scale=mean_scale,
        reference_id=f"exact-terminal:{process.process_id}",
        process_id=process.process_id,
    )


def _moments(samples):
    mean = jnp.mean(samples, axis=0)
    centered = samples - mean
    variance = jnp.mean(centered**2, axis=0)
    return mean, variance


def _ready(value):
    return jax.block_until_ready(value)


def benchmark_process(process, *, sample_count: int, seed: int, quick: bool):
    dimension = process.dimension
    mean = jnp.linspace(-0.4, 0.6, dimension)
    variance = jnp.linspace(0.6, 1.2, dimension)
    data_law = phx.uq.DiagonalNormalLaw(
        mean,
        jnp.sqrt(variance),
        event_shape=process.state_shape,
    )
    score = _score_function(process, mean, variance)
    terminal = _exact_terminal(process, mean, variance)

    data = data_law.sample(jr.key(seed), (max(64, sample_count // 2),))
    denoising_data = jnp.broadcast_to(mean, data.shape)
    target = phx.integration.weighted(
        denoising_data,
        jnp.zeros((data.shape[0],)),
        normalized=True,
        independent=True,
        provenance="deterministic-clean-state",
    )
    term = phx.terms.DenoisingScoreMatchingTerm(
        "score",
        target,
        process,
        phx.terms.UniformTimeSamplingPolicy(0.01, process.terminal_time),
    )
    conditional_score = _score_function(process, mean, jnp.zeros_like(variance))
    score_diagnostics = term.diagnostics(
        {"score": conditional_score}, key=jr.key(seed + 1)
    )

    reverse = phx.transport.ReverseDiffusion(
        process,
        score,
        terminal,
        score_id=f"analytic-score:{process.process_id}",
        dt0=0.01 if quick else 0.005,
        wiener_tolerance=1e-4,
        rtol=1e-6,
        atol=1e-8,
        max_steps=4096,
    )
    reverse_start = perf_counter()
    reverse_result = reverse.sample_with_diagnostics(jr.key(seed + 2), (sample_count,))
    _ready(reverse_result.final_states)
    reverse_seconds = perf_counter() - reverse_start
    sample_mean, sample_variance = _moments(reverse_result.final_states)
    mean_error = jnp.linalg.vector_norm(sample_mean - mean)
    variance_error = jnp.linalg.vector_norm(sample_variance - variance) / jnp.linalg.vector_norm(
        variance
    )

    system = phx.transport.probability_flow_system(
        process,
        score,
        state_layout=phx.dynamics.StateLayout(process.state_shape),
        score_id=f"analytic-score:{process.process_id}",
    )
    evolution = phx.solver.DiffraxEvolution(
        system,
        rtol=1e-8,
        atol=1e-10,
        max_steps=4096,
    )
    transport = phx.transport.ContinuousTransport(
        terminal.law,
        evolution,
        source_coordinate=0.0,
        target_coordinate=process.terminal_time,
        transport_id=f"probability-flow:{process.process_id}",
    )
    flow = phx.transport.ContinuousFlowLaw(
        transport,
        max_exact_dimension=max(8, dimension),
    )
    points = data[: min(8, data.shape[0])]
    density_start = perf_counter()
    density = flow.log_prob_with_diagnostics(points)
    _ready(density.log_prob)
    density_seconds = perf_counter() - density_start
    exact_density = data_law.log_prob(points)
    maximum_density_error = jnp.max(jnp.abs(density.log_prob - exact_density))

    mean_limit = 0.35 if quick else 0.12
    variance_limit = 0.45 if quick else 0.18
    passed = bool(
        score_diagnostics.passed
        and score_diagnostics.objective < 1e-20
        and reverse_result.successful
        and mean_error < mean_limit
        and variance_error < variance_limit
        and density.successful
        and maximum_density_error < 2e-5
    )
    return {
        "process": type(process).__name__,
        "dimension": dimension,
        "sample_count": sample_count,
        "denoising_objective": float(score_diagnostics.objective),
        "reverse_valid_fraction": float(jnp.mean(reverse_result.valid.astype(float))),
        "reverse_mean_error": float(mean_error),
        "reverse_variance_relative_error": float(variance_error),
        "reverse_seconds": reverse_seconds,
        "maximum_probability_flow_density_error": float(maximum_density_error),
        "probability_flow_density_seconds": density_seconds,
        "terminal_relationship": terminal.relationship,
        "passed": passed,
    }


def run_score_diffusion_benchmarks(*, quick: bool = False):
    sample_count = 256 if quick else 2048
    dimension = 2 if quick else 4
    processes = (
        phx.stochastic.VariancePreservingDiffusion(
            dimension,
            beta_minimum=0.1,
            beta_maximum=3.0,
        ),
        phx.stochastic.VarianceExplodingDiffusion(
            dimension,
            initial_scale=0.01,
            terminal_scale=2.0,
        ),
    )
    cases = tuple(
        benchmark_process(
            process,
            sample_count=sample_count,
            seed=20260830 + index * 10,
            quick=quick,
        )
        for index, process in enumerate(processes)
    )
    return {
        "benchmark": "score-diffusion",
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
    report = run_score_diffusion_benchmarks(quick=arguments.quick)
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
