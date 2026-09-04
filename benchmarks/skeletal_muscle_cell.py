#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark the complete Shorten 2007 RHS and prepared stiff route."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.cellular import (
    ShortenFastTwitchModel,
    ShortenIntegrationPlan,
)


def _measure(callable_, arguments, repeats: int) -> tuple[float, object]:
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = callable_(*arguments)
        jax.block_until_ready(result)
    return (time.perf_counter() - start) / repeats, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--integration-repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/skeletal_muscle_cell.json"),
    )
    arguments = parser.parse_args()
    if (
        arguments.cells <= 0
        or arguments.repeats <= 0
        or arguments.integration_repeats <= 0
    ):
        raise ValueError("Benchmark repeat counts and --cells must be positive.")

    model = ShortenFastTwitchModel()
    states = model.initialize((arguments.cells,))
    times = jnp.full((arguments.cells,), 0.75, dtype=states.dtype)
    function = eqx.filter_jit(
        lambda configured, time, state: configured.rhs(time, state)
    )

    compile_start = time.perf_counter()
    compiled = function.lower(model, times, states).compile()
    compile_seconds = time.perf_counter() - compile_start
    seconds, rates = _measure(compiled, (model, times, states), arguments.repeats)
    prepared = ShortenIntegrationPlan(model, [0.0, 0.5, 1.0]).prepare()
    integration_function = eqx.filter_jit(
        lambda integrator: integrator.integrate().states
    )
    integration_compile_start = time.perf_counter()
    compiled_integration = integration_function.lower(prepared).compile()
    integration_compile_seconds = time.perf_counter() - integration_compile_start
    integration_seconds, integration_states = _measure(
        compiled_integration,
        (prepared,),
        arguments.integration_repeats,
    )
    payload = {
        "benchmark": "skeletal-muscle-shorten-2007-complete-rhs",
        "model_id": model.model_id,
        "environment": {
            "backend": jax.default_backend(),
            "jax": jax.__version__,
            "platform": platform.platform(),
        },
        "problem": {
            "cell_count": arguments.cells,
            "state_count": model.state_layout.count,
            "algebraic_count": model.algebraic_layout.count,
            "repeats": arguments.repeats,
            "integration_repeats": arguments.integration_repeats,
        },
        "timings_seconds": {
            "lower_and_compile": compile_seconds,
            "integration_lower_and_compile": integration_compile_seconds,
            "complete_rhs": seconds,
            "integrate_source_pulse_1_ms": integration_seconds,
        },
        "throughput_cell_rhs_per_second": arguments.cells / seconds,
        "evidence": {
            "finite": bool(jnp.all(jnp.isfinite(rates))),
            "rate_norm": float(jnp.linalg.norm(rates)),
            "integration_finite": bool(
                jnp.all(jnp.isfinite(integration_states))
            ),
        },
    }
    payload["all_successful"] = (
        payload["evidence"]["finite"] and payload["evidence"]["integration_finite"]
    )
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
