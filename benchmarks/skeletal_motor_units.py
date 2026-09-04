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
import jax.random as jr

from benchmarks._runtime import capture_environment
from phydrax.applications.skeletal_muscle.motor_units import (
    FuglevandWinterPatla1993Plan,
    FuglevandWinterPatla1993RandomInput,
)


def _case(unit_count: int, capacity: int, repetitions: int):
    prepared = FuglevandWinterPatla1993Plan(
        unit_count,
        event_capacity_per_unit=capacity,
        random_stream_id=f"benchmark/{unit_count}/{capacity}",
    ).prepare()
    state = prepared.initialize()
    random_input = FuglevandWinterPatla1993RandomInput(
        jr.key(0), state.random_step, stream_id=prepared.plan.random_stream_id
    )
    action = eqx.filter_jit(prepared.evaluate)
    start = time.perf_counter()
    first = action(state, prepared.maximum_excitation, 5.0, random_input)
    jax.block_until_ready(first.proposed.motor_unit_force)
    compile_and_first_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    for _ in range(repetitions):
        result = action(state, prepared.maximum_excitation, 5.0, random_input)
    jax.block_until_ready(result.proposed.motor_unit_force)
    execution_ms = 1000.0 * (time.perf_counter() - start) / repetitions
    return {
        "unit_count": unit_count,
        "event_capacity_per_unit": capacity,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "unit_event_slots_per_second": 1000.0 * unit_count * capacity / execution_ms,
        "successful": bool(result.evidence.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-counts", nargs="+", type=int, default=[120, 512])
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/skeletal_motor_units.json")
    )
    arguments = parser.parse_args()
    cases = [
        _case(count, arguments.capacity, arguments.repeats)
        for count in arguments.unit_counts
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_successful": all(case["successful"] for case in cases),
    }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
