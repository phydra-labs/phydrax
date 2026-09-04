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

from benchmarks._runtime import capture_environment
from phydrax.applications.skeletal_muscle.fatigue import (
    LiuBrownYue2002Parameters,
    LiuBrownYue2002Plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/skeletal_fatigue.json")
    )
    arguments = parser.parse_args()
    prepared = LiuBrownYue2002Plan(
        LiuBrownYue2002Parameters(
            fatigue_rate_per_s=0.0206,
            recovery_rate_per_s=0.0084,
        ),
        muscle_id="benchmark-muscle",
        protocol_id="benchmark-constant-effort",
    ).prepare()
    state = prepared.initialize()
    action = eqx.filter_jit(prepared.evaluate)
    start = time.perf_counter()
    first = action(state, 1.0, 0.1)
    jax.block_until_ready(first.proposed.active_fraction)
    compile_and_first_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    for _ in range(arguments.repeats):
        result = action(state, 1.0, 0.1)
    jax.block_until_ready(result.proposed.active_fraction)
    execution_us = 1.0e6 * (time.perf_counter() - start) / arguments.repeats
    payload = {
        "environment": capture_environment().to_dict(),
        "compile_and_first_ms": compile_and_first_ms,
        "execution_us": execution_us,
        "steps_per_second": 1.0e6 / execution_us,
        "successful": bool(result.evidence.successful),
    }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
