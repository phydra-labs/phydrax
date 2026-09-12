"""End-to-end record/replay timing and numerical evidence for RA34PW2."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def _compiler_record(compiled):
    cost = compiled.cost_analysis()
    memory = compiled.memory_analysis()
    unavailable = (
        "Backend did not expose compiler cost or memory analysis."
        if not cost and memory is None
        else None
    )
    return asdict(
        compiler_evidence(
            cost,
            memory,
            source="jax-compiled-executable",
            unavailable_reason=unavailable,
        )
    )


def _problem(kind: str, parameter):
    if kind == "scaled_decay":

        def drift(time, state, rate):
            del time
            return -rate * state

        initial = jnp.asarray((1.0, 1.0e6))
    elif kind == "van_der_pol":

        def drift(time, state, rate):
            del time
            return jnp.stack(
                (
                    state[1],
                    rate * (1.0 - state[0] ** 2) * state[1] - state[0],
                )
            )

        initial = jnp.asarray((2.0, 0.0))
    else:
        raise ValueError("Unknown Rosenbrock replay benchmark problem.")
    return phx.solver.DifferentialProblem(
        drift,
        initial,
        t0=0.0,
        t1=1.0,
        args=parameter,
        problem_id=f"rosenbrock-replay-benchmark:{kind}",
    )


def _benchmark_case(kind: str, parameter: float, repeats: int, warmup: int):
    base = jnp.asarray(parameter)
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.5, 1.0)),
        time_id=f"rosenbrock-replay-benchmark:{kind}",
    )
    prepared = phx.solver.prepare_rosenbrock(
        _problem(kind, base),
        grid,
        adaptive=phx.solver.RosenbrockAdaptivePolicy(
            relative_tolerance=1e-4,
            absolute_tolerance=1e-9,
            initial_step=0.05,
            maximum_step=0.2,
            maximum_accepted_steps=512,
            maximum_attempts=1024,
        ),
    )
    source_solve = jax.jit(
        lambda value: phx.solver.solve_rosenbrock(prepared, args=value)
    )
    source_compiled, source_compilation = measure_lower_and_compile(
        lambda: source_solve.lower(base),
        lambda lowered: lowered.compile(),
    )
    source, record_timing = measure_repeated(
        lambda: source_compiled(base),
        warmup=warmup,
        repeats=repeats,
    )
    scheduled, schedule_timing = measure_repeated(
        lambda: phx.solver.schedule_rosenbrock(
            prepared,
            source,
            reference_args=base,
        ),
        warmup=1,
        repeats=repeats,
    )

    fresh_value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda value: phx.solver.solve_rosenbrock(
                prepared,
                args=value,
            ).states[-1, 0]
        )
    )
    replay_value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda value: phx.solver.solve_scheduled_rosenbrock(
                scheduled,
                args=value,
            ).states[-1, 0]
        )
    )
    fresh_compiled, fresh_compilation = measure_lower_and_compile(
        lambda: fresh_value_and_grad.lower(base),
        lambda lowered: lowered.compile(),
    )
    replay_compiled, replay_compilation = measure_lower_and_compile(
        lambda: replay_value_and_grad.lower(base),
        lambda lowered: lowered.compile(),
    )
    _, fresh_timing = measure_repeated(
        lambda: fresh_compiled(base),
        warmup=warmup,
        repeats=repeats,
    )
    _, replay_timing = measure_repeated(
        lambda: replay_compiled(base),
        warmup=warmup,
        repeats=repeats,
    )
    fresh_value, fresh_gradient = fresh_compiled(base)
    replay_value, replay_gradient = replay_compiled(base)
    value_difference = float(jnp.abs(fresh_value - replay_value))
    gradient_difference = float(jnp.abs(fresh_gradient - replay_gradient))
    saved_seconds = fresh_timing.median_seconds - replay_timing.median_seconds
    setup_seconds = record_timing.median_seconds + schedule_timing.median_seconds
    break_even = None if saved_seconds <= 0.0 else setup_seconds / saved_seconds
    replay_adequacy = phx.solver.solve_scheduled_rosenbrock(
        scheduled,
        args=base,
    ).stats["replay_adequacy"]
    passed = (
        bool(source.successful)
        and bool(replay_adequacy.completed)
        and value_difference < 1e-6
        and gradient_difference < 1e-5
        and replay_timing.median_seconds < fresh_timing.median_seconds
        and break_even is not None
        and break_even <= 8.0
    )
    return {
        "problem": kind,
        "parameter": parameter,
        "accepted_steps": int(source.stats["accepted_steps"]),
        "attempts": int(source.stats["attempts"]),
        "capacity": prepared.adaptive.maximum_accepted_steps,
        "capacity_utilization": float(
            source.stats["accepted_steps"] / prepared.adaptive.maximum_accepted_steps
        ),
        "record": record_timing.to_dict(),
        "schedule": schedule_timing.to_dict(),
        "fresh_gradient": fresh_timing.to_dict(),
        "scheduled_gradient": replay_timing.to_dict(),
        "source_compilation": {
            "lowering_seconds": source_compilation.lowering_seconds,
            "compilation_seconds": source_compilation.compilation_seconds,
        },
        "fresh_compilation": {
            "lowering_seconds": fresh_compilation.lowering_seconds,
            "compilation_seconds": fresh_compilation.compilation_seconds,
        },
        "scheduled_compilation": {
            "lowering_seconds": replay_compilation.lowering_seconds,
            "compilation_seconds": replay_compilation.compilation_seconds,
        },
        "fresh_compiler": _compiler_record(fresh_compiled),
        "scheduled_compiler": _compiler_record(replay_compiled),
        "source_mesh_bytes": logical_array_bytes(source.temporal_mesh),
        "scheduled_mesh_bytes": logical_array_bytes(scheduled.temporal_mesh),
        "value_difference": value_difference,
        "gradient_difference": gradient_difference,
        "maximum_error_ratio": float(replay_adequacy.maximum_error_ratio),
        "amortized_break_even_replays": break_even,
        "passed": passed,
    }


def benchmark(repeats: int, warmup: int):
    cases = (
        _benchmark_case("scaled_decay", 2.0, repeats, warmup),
        _benchmark_case("van_der_pol", 10.0, repeats, warmup),
    )
    return {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/rosenbrock_replay.json"),
    )
    arguments = parser.parse_args()
    payload = benchmark(arguments.repeats, arguments.warmup)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
