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
from phydrax.applications.skeletal_muscle.energetics import (
    UchidaUmberger2010Parameters,
    UchidaUmberger2010Plan,
)


def _case(muscle_count: int) -> dict[str, object]:
    plan = UchidaUmberger2010Plan(
        UchidaUmberger2010Parameters(
            jnp.full((muscle_count,), 0.5),
            jnp.full((muscle_count,), 0.5),
            jnp.full((muscle_count,), 0.1),
            jnp.full((muscle_count,), 10.0),
        ),
        tuple(f"muscle-{index}" for index in range(muscle_count)),
    )
    values = (
        jnp.full((muscle_count,), 0.7),
        jnp.full((muscle_count,), 0.6),
        jnp.full((muscle_count,), 100.0),
        jnp.full((muscle_count,), 0.9),
        jnp.full((muscle_count,), 0.1),
        jnp.full((muscle_count,), -0.01),
    )
    action = eqx.filter_jit(plan.evaluate)
    start = time.perf_counter()
    first = action(*values)
    first.muscle_metabolic_power_W.block_until_ready()
    compile_and_first_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    result = action(*values)
    result.muscle_metabolic_power_W.block_until_ready()
    execution_ms = 1000.0 * (time.perf_counter() - start)
    return {
        "muscle_count": muscle_count,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "muscles_per_second": 1000.0 * muscle_count / execution_ms,
        "successful": bool(result.evidence.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/skeletal_muscle_energetics.json"),
    )
    arguments = parser.parse_args()
    cases = [_case(2)] if arguments.smoke else [_case(126), _case(416), _case(4096)]
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
