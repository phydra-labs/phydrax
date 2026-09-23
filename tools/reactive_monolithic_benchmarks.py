#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import contextlib
import io
import json
import runpy
import statistics
import time
from pathlib import Path

import jax.numpy as jnp


def run():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        return runpy.run_path(
            Path(__file__).resolve().parents[1]
            / "examples/monolithic_reactive_cfd_dem.py"
        )


warm = run()
durations = []
for _ in range(3):
    started = time.perf_counter()
    latest = run()
    durations.append(time.perf_counter() - started)
result = latest["result"]
evaluation = result.evaluation
maximum_residual = max(
    float(jnp.linalg.norm(evaluation.momentum_residual)),
    float(jnp.abs(evaluation.energy_residual)),
    float(jnp.max(jnp.abs(evaluation.species_residual))),
)
passed = bool(
    warm["result"].successful
    & result.successful
    & jnp.isfinite(maximum_residual)
    & (maximum_residual <= 1.0e-12)
)
payload = {
    "benchmark": "reactive-monolithic-newton",
    "passed": passed,
    "median_seconds": statistics.median(durations),
    "minimum_seconds": min(durations),
    "nonlinear_iterations": int(result.nonlinear.diagnostics.iterations),
    "linear_iterations": int(result.nonlinear.diagnostics.linear_iterations),
    "preconditioner": result.preconditioner.mode.value,
    "maximum_balance_residual": maximum_residual,
}
print(json.dumps(payload, indent=2, allow_nan=False))
if not passed:
    raise SystemExit(1)
