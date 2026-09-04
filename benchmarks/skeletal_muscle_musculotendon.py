"""Benchmark fixed-capacity De Groote--Fregly 2016 constitutive evaluations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.musculotendon import (
    de_groote_fregly_2016_active_force_length,
    de_groote_fregly_2016_force_velocity,
    de_groote_fregly_2016_inverse_tendon_force_length,
    DeGrooteFregly2016ImplicitTendonForcePlan,
    DeGrooteFregly2016Parameters,
    DeGrooteFregly2016Plan,
    DeGrooteFregly2016State,
)


def _case(capacity: int):
    parameters = DeGrooteFregly2016Parameters(
        jnp.linspace(400.0, 5000.0, capacity),
        jnp.linspace(0.04, 0.16, capacity),
        jnp.linspace(0.08, 0.45, capacity),
        jnp.linspace(0.0, 0.35, capacity),
        jnp.linspace(0.4, 1.6, capacity),
    )
    activation = jnp.linspace(0.05, 0.95, capacity)
    one = jnp.ones((capacity,))
    cosine = jnp.cos(parameters.pennation_angle_at_optimum_rad)
    tendon_force = (
        activation
        * de_groote_fregly_2016_active_force_length(parameters, one)
        * de_groote_fregly_2016_force_velocity(parameters, jnp.zeros_like(one))
        * cosine
    )
    tendon_length = (
        de_groote_fregly_2016_inverse_tendon_force_length(parameters, tendon_force)
        * parameters.tendon_slack_length_m
    )
    length = tendon_length + parameters.optimal_fiber_length_m * cosine
    state = DeGrooteFregly2016State(activation, tendon_force)
    prepared = DeGrooteFregly2016Plan(
        parameters, tuple(f"muscle-{index:03d}" for index in range(capacity))
    ).prepare(state)
    return prepared, state, length, jnp.zeros_like(length)


def benchmark(
    capacity: int, iterations: int, implicit_iterations: int
) -> dict[str, object]:
    prepared, state, length, velocity = _case(capacity)
    evaluate = eqx.filter_jit(prepared.evaluate)
    output = evaluate(state, state.activation, length, velocity)
    jax.block_until_ready(output.tendon_force_N)
    start = time.perf_counter()
    for _ in range(iterations):
        output = evaluate(state, state.activation, length, velocity)
    jax.block_until_ready(output.tendon_force_N)
    elapsed = time.perf_counter() - start

    def total_force(musculotendon_length):
        return jnp.sum(
            prepared.evaluate(
                state, state.activation, musculotendon_length, velocity
            ).tendon_force_N
        )

    jvp = jax.jit(
        lambda value: jax.jvp(total_force, (value,), (jnp.ones_like(value),))[1]
    )
    derivative = jvp(length)
    jax.block_until_ready(derivative)
    derivative_start = time.perf_counter()
    for _ in range(iterations):
        derivative = jvp(length)
    jax.block_until_ready(derivative)
    derivative_elapsed = time.perf_counter() - derivative_start
    implicit = DeGrooteFregly2016ImplicitTendonForcePlan(
        prepared.plan.parameters, prepared.plan.muscle_names
    ).prepare(state)
    implicit_step = eqx.filter_jit(implicit.candidate)
    implicit_output = implicit_step(
        state, state.activation, length, velocity, jnp.asarray(1.0e-6)
    )
    jax.block_until_ready(implicit_output.evidence.algebraic_residual)
    implicit_start = time.perf_counter()
    for _ in range(implicit_iterations):
        implicit_output = implicit_step(
            state, state.activation, length, velocity, jnp.asarray(1.0e-6)
        )
    jax.block_until_ready(implicit_output.evidence.algebraic_residual)
    implicit_elapsed = time.perf_counter() - implicit_start
    return {
        "muscle_capacity": capacity,
        "iterations": iterations,
        "evaluation_seconds": elapsed,
        "evaluations_per_second": iterations / elapsed,
        "jvp_seconds": derivative_elapsed,
        "jvps_per_second": iterations / derivative_elapsed,
        "implicit_iterations": implicit_iterations,
        "implicit_roots_per_second": implicit_iterations / implicit_elapsed,
        "implicit_successful": bool(implicit_output.successful),
        "all_successful": bool(jnp.all(output.successful)),
        "force_checksum_N": float(jnp.sum(output.tendon_force_N)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--implicit-iterations", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/skeletal_muscle_musculotendon.json"),
    )
    arguments = parser.parse_args()
    if (
        arguments.capacity <= 0
        or arguments.iterations <= 0
        or arguments.implicit_iterations <= 0
    ):
        raise ValueError("capacity and iterations must be positive.")
    payload = benchmark(
        arguments.capacity,
        arguments.iterations,
        arguments.implicit_iterations,
    )
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"] or not payload["implicit_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
