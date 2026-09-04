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
from phydrax.applications.skeletal_muscle.personalization import (
    PhysicalRelativeForceCalibrationPlan,
)


def _case(sample_count: int, nuisance_count: int, repetitions: int):
    relative = jnp.linspace(0.0, 1.0, sample_count)
    columns = [jnp.ones_like(relative)]
    for power in range(1, nuisance_count):
        columns.append((relative - 0.5) ** (power + 1))
    nuisance = (
        jnp.stack(columns, axis=1)
        if nuisance_count
        else jnp.zeros((sample_count, 0))
    )
    plan = PhysicalRelativeForceCalibrationPlan(
        nuisance,
        tuple(f"nuisance-{index}" for index in range(nuisance_count)),
        protocol_id=f"benchmark/{sample_count}/{nuisance_count}",
        asset_id="synthetic-benchmark-load-cell",
    ).prepare()
    state = plan.initialize(100.0)
    observed = 500.0 * relative + (2.0 if nuisance_count else 0.0)
    uncertainty = jnp.ones_like(relative)
    action = eqx.filter_jit(plan.evaluate)
    start = time.perf_counter()
    first = action(state, relative, observed, uncertainty)
    jax.block_until_ready(first.proposed.scale_newton_per_relative_force)
    compile_and_first_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    for _ in range(repetitions):
        result = action(state, relative, observed, uncertainty)
    jax.block_until_ready(result.proposed.scale_newton_per_relative_force)
    execution_ms = 1000.0 * (time.perf_counter() - start) / repetitions
    return {
        "sample_count": sample_count,
        "nuisance_count": nuisance_count,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "successful": bool(result.evidence.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/skeletal_force_calibration.json"),
    )
    arguments = parser.parse_args()
    cases = [
        _case(samples, nuisances, arguments.repeats)
        for samples, nuisances in ((32, 0), (128, 1), (512, 3))
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
