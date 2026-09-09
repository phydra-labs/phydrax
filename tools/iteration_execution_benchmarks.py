#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measure disabled, streaming, and bounded-trace iteration execution costs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import (
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def _step(step_index, time, state, step_size, forcing):
    del step_index, time
    candidate = state + step_size * forcing
    return phx.solver.FixedStepResult(
        candidate_state=candidate,
        accepted_state=candidate,
        successful=jnp.asarray(True),
        residual=jnp.max(jnp.abs(forcing)),
        iterations=jnp.asarray(1, dtype=jnp.int32),
        work=jnp.asarray(state.size, dtype=jnp.int32),
        transform_applied=jnp.asarray(False),
        transform_correction_norm=jnp.zeros((), dtype=state.dtype),
    )


def _operation(iteration):
    method = phx.solver.CallableFixedStepMethod(_step, "iteration-benchmark-step")
    rollout = phx.solver.FixedStepRolloutPlan(
        retention="final",
        iteration=iteration,
    )

    def run(initial):
        problem = phx.solver.FixedStepProblem(
            method,
            initial,
            t0=0.0,
            t1=1.0,
            step_size=1.0 / 256.0,
            args=jnp.linspace(0.5, 1.5, initial.size),
        )
        return rollout.rollout(problem)

    return jax.jit(run)


def _case(name, iteration, initial, *, warmup, repeats):
    operation = _operation(iteration)
    compiled, compilation = measure_lower_and_compile(
        lambda: operation.lower(initial),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(initial),
        warmup=warmup,
        repeats=repeats,
    )
    compiler = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    return {
        "name": name,
        "compilation": asdict(compilation),
        "execution": execution.to_milliseconds_dict(),
        "compiler": asdict(compiler),
        "logical_result_bytes": logical_array_bytes(result),
        "successful": bool(result.successful),
        "iteration_plan_id": None if iteration is None else iteration.plan_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    warmup = 1 if arguments.smoke else 5
    repeats = 2 if arguments.smoke else 20
    initial = jnp.zeros((128 if arguments.smoke else 4096,), dtype=jnp.float64)
    cases = (
        ("disabled", None),
        (
            "constant-memory-counts",
            phx.execution.IterationPlan(
                observers=(phx.execution.IterationCountObserver(),)
            ),
        ),
        (
            "bounded-trace-64",
            phx.execution.IterationPlan(
                observers=(phx.execution.IterationTraceObserver(64),)
            ),
        ),
    )
    report = {
        "benchmark": "iteration-execution",
        "state_size": int(initial.size),
        "step_count": 256,
        "warmup": warmup,
        "repeats": repeats,
        "cases": [
            _case(name, iteration, initial, warmup=warmup, repeats=repeats)
            for name, iteration in cases
        ],
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
