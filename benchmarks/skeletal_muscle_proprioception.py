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
from phydrax.applications.skeletal_muscle.proprioception import (
    MileusnicSpindle2006Plan,
    MileusnicSpindleInput,
)


def _case(step_count: int) -> dict[str, object]:
    runtime = MileusnicSpindle2006Plan().prepare()
    input_value = MileusnicSpindleInput(1.02, 0.1, 0.0, 70.0, 70.0)

    def rollout(initial):
        def step(state, _):
            candidate = runtime.candidate(state, input_value, 1.0e-4)
            return candidate.commit(), candidate.evidence.successful

        return jax.lax.scan(step, initial, xs=None, length=step_count)

    action = eqx.filter_jit(rollout)
    initial = runtime.initialize()
    start = time.perf_counter()
    first = action(initial)
    first[0].branch_tension_force_unit.block_until_ready()
    compile_and_first_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    result = action(initial)
    result[0].branch_tension_force_unit.block_until_ready()
    execution_ms = 1000.0 * (time.perf_counter() - start)
    return {
        "step_count": step_count,
        "simulated_duration_s": step_count * 1.0e-4,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "steps_per_second": 1000.0 * step_count / execution_ms,
        "all_successful": bool(jnp.all(result[1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/skeletal_muscle_proprioception.json"),
    )
    arguments = parser.parse_args()
    payload = {
        "environment": capture_environment().to_dict(),
        "case": _case(10 if arguments.smoke else 10_000),
    }
    payload["all_successful"] = payload["case"]["all_successful"]
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
