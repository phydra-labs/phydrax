#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from benchmarks._runtime import capture_environment
from phydrax.applications.skeletal_muscle.motor_units import (
    PotvinFuglevand2017Plan,
    PotvinFuglevand2017State,
)


def _synchronize(value) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _case(batch_size: int, rollout_steps: int, repetitions: int):
    runtime = PotvinFuglevand2017Plan().prepare()
    source = runtime.initialize()
    duration = jnp.broadcast_to(
        source.recruitment_duration_s, (batch_size, runtime.plan.unit_count)
    )
    capacity = jnp.broadcast_to(
        source.current_twitch_force, (batch_size, runtime.plan.unit_count)
    )
    excitation = jnp.full((batch_size,), 40.125, dtype=capacity.dtype)

    def one_step(duration_value, capacity_value, drive):
        state = PotvinFuglevand2017State(duration_value, capacity_value)
        candidate = runtime.candidate(state, drive, 0.1)
        accepted = candidate.commit()
        return (
            accepted.recruitment_duration_s,
            accepted.current_twitch_force,
            candidate.output.total_force,
            candidate.evidence.successful,
            candidate.evidence.minimum_recruitment_margin,
        )

    action = jax.jit(jax.vmap(one_step))
    start = time.perf_counter()
    first = action(duration, capacity, excitation)
    _synchronize(first)
    compile_and_first_ms = 1000.0 * (time.perf_counter() - start)

    start = time.perf_counter()
    result = first
    for _ in range(repetitions):
        result = action(duration, capacity, excitation)
    _synchronize(result)
    execution_ms = 1000.0 * (time.perf_counter() - start) / repetitions

    def rollout(initial):
        def step(state, _):
            next_duration, next_capacity, total_force, successful, _ = action(
                state[0], state[1], excitation
            )
            return (next_duration, next_capacity), (total_force, successful)

        return jax.lax.scan(step, initial, xs=None, length=rollout_steps)

    compiled_rollout = jax.jit(rollout)
    start = time.perf_counter()
    rollout_result = compiled_rollout((duration, capacity))
    _synchronize(rollout_result)
    rollout_compile_and_first_ms = 1000.0 * (time.perf_counter() - start)

    start = time.perf_counter()
    rollout_result = compiled_rollout((duration, capacity))
    _synchronize(rollout_result)
    rollout_execution_ms = 1000.0 * (time.perf_counter() - start)
    final_state, history = rollout_result
    itemsize = capacity.dtype.itemsize
    return {
        "unit_count": runtime.plan.unit_count,
        "batch_size": batch_size,
        "rollout_steps": rollout_steps,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "transitions_per_second": 1000.0 * batch_size / execution_ms,
        "rollout_compile_and_first_ms": rollout_compile_and_first_ms,
        "rollout_execution_ms": rollout_execution_ms,
        "rollout_transitions_per_second": (
            1000.0 * batch_size * rollout_steps / rollout_execution_ms
        ),
        "logical_state_bytes": int(batch_size * runtime.plan.unit_count * 2 * itemsize),
        "minimum_recruitment_margin": float(jnp.min(result[4])),
        "minimum_final_capacity": float(jnp.min(final_state[1])),
        "all_successful": bool(jnp.all(result[3]) & jnp.all(history[1])),
    }

def _gradient_case(repetitions: int):
    runtime = PotvinFuglevand2017Plan().prepare()
    source = runtime.initialize()
    state = PotvinFuglevand2017State(
        jnp.full_like(source.recruitment_duration_s, 15.0),
        source.current_twitch_force,
    )
    scale = runtime.parameters.adaptation_scale

    def objective(value):
        selected = eqx.tree_at(
            lambda model: model.parameters.adaptation_scale,
            runtime,
            value,
        )
        return selected.evaluate(state, 20.125).total_force

    def derivatives(value):
        primal, tangent = jax.jvp(objective, (value,), (jnp.ones_like(value),))
        reverse_primal, pullback = jax.vjp(objective, value)
        reverse = pullback(jnp.ones_like(reverse_primal))[0]
        return primal, tangent, reverse

    action = jax.jit(derivatives)
    start = time.perf_counter()
    first = action(scale)
    _synchronize(first)
    compile_and_first_ms = 1000.0 * (time.perf_counter() - start)

    start = time.perf_counter()
    result = first
    for _ in range(repetitions):
        result = action(scale)
    _synchronize(result)
    execution_ms = 1000.0 * (time.perf_counter() - start) / repetitions
    duality_error = jnp.abs(result[1] - result[2])
    return {
        "parameter": "adaptation_scale",
        "common_excitation": 20.125,
        "recruitment_duration_s": 15.0,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "jvp": float(result[1]),
        "vjp": float(result[2]),
        "duality_error": float(duality_error),
        "successful": bool(jnp.all(jnp.isfinite(jnp.stack(result))) & (duality_error <= 1.0e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/skeletal_muscle_motor_units.json"))
    arguments = parser.parse_args()
    batch_sizes = (1,) if arguments.smoke else (1, 128, 1024)
    rollout_steps = 10 if arguments.smoke else 1_000
    repetitions = 2 if arguments.smoke else 10
    cases = [
        _case(batch_size, rollout_steps, repetitions)
        for batch_size in batch_sizes
    ]
    gradient = _gradient_case(repetitions)
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "gradient": gradient,
        "all_successful": (
            all(case["all_successful"] for case in cases)
            and gradient["successful"]
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
