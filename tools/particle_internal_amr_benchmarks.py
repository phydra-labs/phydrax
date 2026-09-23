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
            Path(__file__).resolve().parents[1] / "examples/adaptive_catalyst_pellet.py"
        )


warm = run()
durations = []
for _ in range(5):
    started = time.perf_counter()
    latest = run()
    durations.append(time.perf_counter() - started)
evidence = latest["result"].evidence
maximum_balance = max(
    float(jnp.max(jnp.abs(evidence.energy_residual))),
    float(jnp.max(jnp.abs(evidence.species_residual))),
    float(jnp.max(jnp.abs(evidence.pore_volume_residual))),
    float(jnp.max(jnp.abs(evidence.surface_area_residual))),
)
passed = bool(
    warm["result"].successful
    & latest["result"].successful
    & jnp.isfinite(maximum_balance)
    & (maximum_balance <= 1.0e-10)
)
payload = {
    "benchmark": "particle-internal-amr",
    "passed": passed,
    "median_seconds": statistics.median(durations),
    "minimum_seconds": min(durations),
    "active_fine_cells": int(latest["result"].accepted_state.fine_active.sum()),
    "maximum_balance_residual": maximum_balance,
}
print(json.dumps(payload, indent=2, allow_nan=False))
if not passed:
    raise SystemExit(1)
