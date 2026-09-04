#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

from benchmarks._runtime import capture_environment
from phydrax.applications.skeletal_muscle.electromyography import (
    MotorUnitActionPotentialTemplatePlan,
)


def _case(unit_count: int, channel_count: int, event_capacity: int) -> dict[str, object]:
    template = jnp.zeros((unit_count, channel_count, 64)).at[..., 8].set(1.0e-4)
    prepared = MotorUnitActionPotentialTemplatePlan(
        template,
        0.001,
        0,
        tuple(f"unit-{index}" for index in range(unit_count)),
        tuple(f"channel-{index}" for index in range(channel_count)),
        template_source_id="benchmark-explicit-template",
    ).prepare()
    events = jnp.arange(event_capacity, dtype=float)[None, :] * 0.01
    events = jnp.broadcast_to(events, (unit_count, event_capacity))
    mask = jnp.ones(events.shape, dtype=bool)
    times = jnp.arange(1024) * 0.001
    action = eqx.filter_jit(prepared.synthesize)
    start = time.perf_counter()
    first = action(events, mask, times)
    first.voltage_V.block_until_ready()
    compile_and_first_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    result = action(events, mask, times)
    result.voltage_V.block_until_ready()
    execution_ms = 1000.0 * (time.perf_counter() - start)
    return {
        "unit_count": unit_count,
        "channel_count": channel_count,
        "event_capacity": event_capacity,
        "sample_count": times.size,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "successful": bool(result.evidence.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/skeletal_muscle_emg.json")
    )
    arguments = parser.parse_args()
    case = _case(2, 1, 2) if arguments.smoke else _case(120, 16, 16)
    payload = {
        "environment": capture_environment().to_dict(),
        "case": case,
        "all_successful": case["successful"],
    }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
