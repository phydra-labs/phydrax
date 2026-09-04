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
from phydrax.applications.robotics import SphereRouteWrapPlan


def _case(batch_size: int, repetitions: int) -> dict[str, object]:
    prepared = SphereRouteWrapPlan(32).prepare(jnp.zeros(3), 1.0)
    offsets = jnp.linspace(-0.1, 0.1, batch_size)
    starts = jnp.stack(
        (-2.0 * jnp.ones_like(offsets), 0.4 + offsets, jnp.zeros_like(offsets)),
        axis=-1,
    )
    ends = jnp.stack(
        (2.1 * jnp.ones_like(offsets), 0.7 - offsets, jnp.zeros_like(offsets)),
        axis=-1,
    )
    action = eqx.filter_jit(jax.vmap(prepared.evaluate))
    start = time.perf_counter()
    first = action(starts, ends)
    first.total_length_m.block_until_ready()
    compile_and_first_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    result = first
    for _ in range(repetitions):
        result = action(starts, ends)
    result.total_length_m.block_until_ready()
    execution_ms = 1000.0 * (time.perf_counter() - start) / repetitions
    return {
        "batch_size": batch_size,
        "sample_count": prepared.plan.sample_count,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "routes_per_second": 1000.0 * batch_size / execution_ms,
        "maximum_tangency_residual": float(
            jnp.max(result.evidence.endpoint_tangency_residual)
        ),
        "maximum_surface_residual": float(
            jnp.max(result.evidence.surface_residual)
        ),
        "all_successful": bool(jnp.all(result.evidence.successful)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/robotics_analytic_wrap.json")
    )
    arguments = parser.parse_args()
    sizes = (1,) if arguments.smoke else (1, 128, 1024)
    repetitions = 2 if arguments.smoke else 10
    cases = [_case(size, repetitions) for size in sizes]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_successful": all(case["all_successful"] for case in cases),
    }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
