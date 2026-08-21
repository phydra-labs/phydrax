#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic sequential-versus-causal fixed-trajectory HMC benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.uq._causal_hmc import build_causal_hmc_kernel


def _ready(value: Any) -> Any:
    return jax.block_until_ready(value)


def _timings(function, arguments, *, repetitions: int):
    compiled = jax.jit(function)
    started = time.perf_counter()
    result = compiled(*arguments)
    _ready(result)
    cold = time.perf_counter() - started
    warm = []
    for _ in range(int(repetitions)):
        started = time.perf_counter()
        value = compiled(*arguments)
        _ready(value)
        warm.append(time.perf_counter() - started)
    execution = statistics.median(warm)
    return result, {
        "cold_seconds": cold,
        "compile_seconds": max(0.0, cold - execution),
        "execute_seconds": execution,
    }


def _target(name: str, dimension: int):
    scales = jnp.linspace(1.0, 4.0, dimension)
    if name == "gaussian":
        return lambda value: -0.5 * jnp.sum(scales * value * value)
    if name == "curved":

        def logdensity(value):
            leading = value[:-1]
            trailing = value[1:]
            rosenbrock = jnp.sum(
                100.0 * jnp.square(trailing - jnp.square(leading))
                + jnp.square(1.0 - leading)
            )
            return -0.02 * rosenbrock - 0.01 * jnp.sum(value * value)

        return logdensity
    raise ValueError(f"Unknown HMC benchmark target {name!r}.")


def benchmark_case(
    *,
    target_name: str,
    dimension: int,
    leapfrog_steps: int,
    linearization: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    logdensity = _target(target_name, dimension)
    position = jnp.linspace(-0.2, 0.3, dimension)
    state = blackjax.hmc.init(position, logdensity)
    key = jr.key(seed)
    step_size = jnp.asarray(0.02 if target_name == "curved" else 0.08)
    inverse_mass = jnp.ones((dimension,))
    sequential_kernel = blackjax.hmc.build_kernel()
    config = phx.uq.CausalHMCConfig(
        trajectory_block_size=min(128, leapfrog_steps),
        linearization=linearization,
        probe_count=4,
        absolute_residual=2e-6,
        relative_residual=2e-6,
        maximum_outer_iterations=min(160, max(16, leapfrog_steps)),
    )
    causal_kernel = build_causal_hmc_kernel(config)

    def sequential(current_key, current_state):
        return sequential_kernel(
            current_key,
            current_state,
            logdensity,
            step_size,
            inverse_mass,
            leapfrog_steps,
        )

    def causal(current_key, current_state):
        return causal_kernel(
            current_key,
            current_state,
            logdensity,
            step_size,
            inverse_mass,
            leapfrog_steps,
        )

    sequential_result, sequential_timing = _timings(
        sequential,
        (key, state),
        repetitions=repetitions,
    )
    causal_result, causal_timing = _timings(
        causal,
        (key, state),
        repetitions=repetitions,
    )
    sequential_state, sequential_info = sequential_result
    causal_state, causal_info = causal_result
    return {
        "target": target_name,
        "dimension": dimension,
        "leapfrog_steps": leapfrog_steps,
        "linearization": linearization,
        "sequential": sequential_timing,
        "causal": causal_timing,
        "warm_speedup": (
            sequential_timing["execute_seconds"] / causal_timing["execute_seconds"]
        ),
        "maximum_endpoint_error": float(
            jnp.max(
                jnp.abs(sequential_info.proposal.position - causal_info.proposal.position)
            )
        ),
        "energy_error": float(jnp.abs(sequential_info.energy - causal_info.energy)),
        "acceptance_rate_error": float(
            jnp.abs(sequential_info.acceptance_rate - causal_info.acceptance_rate)
        ),
        "acceptance_decision_matches": bool(
            sequential_info.is_accepted == causal_info.is_accepted
        ),
        "returned_state_error": float(
            jnp.max(jnp.abs(sequential_state.position - causal_state.position))
        ),
        "converged": bool(causal_info.causal_converged),
        "fallback_used": bool(causal_info.causal_fallback_used),
        "outer_iterations": int(causal_info.causal_outer_iterations),
        "maximum_direct_residual": float(causal_info.causal_maximum_residual),
        "transition_evaluations": int(causal_info.causal_transition_evaluations),
    }


def run_causal_hmc_benchmarks(*, quick: bool = False) -> dict[str, Any]:
    dimensions = (2,) if quick else (2, 16, 64)
    steps = (8,) if quick else (8, 32, 128)
    targets = ("gaussian",) if quick else ("gaussian", "curved")
    linearizations = (
        ("dense-exact",)
        if quick
        else (
            "dense-exact",
            "pair-hutchinson",
        )
    )
    repetitions = 2 if quick else 5
    cases = []
    index = 0
    for target_name in targets:
        for dimension in dimensions:
            for leapfrog_steps in steps:
                for linearization in linearizations:
                    if linearization == "dense-exact" and dimension > 16:
                        continue
                    index += 1
                    cases.append(
                        benchmark_case(
                            target_name=target_name,
                            dimension=dimension,
                            leapfrog_steps=leapfrog_steps,
                            linearization=linearization,
                            repetitions=repetitions,
                            seed=20260821 + index,
                        )
                    )
    return {
        "schema": "phydrax-causal-hmc-benchmark-v1",
        "backend": jax.default_backend(),
        "quick": bool(quick),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    report = run_causal_hmc_benchmarks(quick=arguments.quick)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n")
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
