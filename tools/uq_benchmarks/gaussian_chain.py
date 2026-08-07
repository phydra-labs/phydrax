#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp

import phydrax as phx


def _problem(num_steps: int, state_size: int) -> phx.stochastic.StateSpaceProblem:
    times = jnp.linspace(0.01, 2.0, num_steps)
    observations = phx.stochastic.ObservationSequence(
        times,
        jnp.sin(times)[:, None],
        case_ids=("benchmark",),
        sequence_id=f"gaussian-chain-{num_steps}-{state_size}",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((state_size,)),
        jnp.eye(state_size),
        state_shape=(state_size,),
        prior_id="gaussian-chain-benchmark-prior",
    )
    drift = -0.15 * jnp.eye(state_size) + 0.02 * jnp.diag(
        jnp.ones((max(state_size - 1, 0),)), 1
    )
    dynamics = phx.stochastic.LinearGaussianDynamics(
        drift,
        0.2 * jnp.eye(state_size),
        state_shape=(state_size,),
        dynamics_id="gaussian-chain-benchmark-dynamics",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(dynamics)
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.ones((1, state_size)) / state_size,
        jnp.asarray([[0.1]]),
        state_shape=(state_size,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="gaussian-chain-benchmark-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="gaussian-chain-benchmark-problem",
    )


def _measure(problem, method: str, repeats: int) -> dict[str, float | str]:
    filtered = phx.uq.kalman_filter(problem, method=method)
    smoothed = phx.uq.rts_smoother(filtered, method=method)
    jax.block_until_ready(smoothed.means)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        filtered = phx.uq.kalman_filter(problem, method=method)
        smoothed = phx.uq.rts_smoother(filtered, method=method)
        jax.block_until_ready(smoothed.means)
        durations.append(time.perf_counter() - started)
    return {
        "requested_method": method,
        "resolved_filter_method": filtered.execution_method,
        "resolved_smoother_method": smoothed.execution_method,
        "minimum_seconds": min(durations),
        "mean_seconds": sum(durations) / len(durations),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark sequential and associative Gaussian-chain inference."
    )
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--state-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.state_size <= 0 or args.repeats <= 0:
        parser.error("steps, state-size, and repeats must be positive")
    problem = _problem(args.steps, args.state_size)
    report = {
        "benchmark_id": "linear-gaussian-chain",
        "num_steps": args.steps,
        "state_size": args.state_size,
        "repeats": args.repeats,
        "sequential": _measure(problem, "sequential", args.repeats),
        "parallel": _measure(problem, "parallel", args.repeats),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
