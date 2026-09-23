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


def run():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        return runpy.run_path(
            Path(__file__).resolve().parents[1]
            / "examples/growing_reactive_particle_pool.py"
        )


warm = run()
durations = []
for _ in range(3):
    started = time.perf_counter()
    latest = run()
    durations.append(time.perf_counter() - started)
result = latest["result"]
passed = bool(
    warm["result"].successful
    & result.successful
    & (result.epoch.dynamics.bodies.capacity > latest["particles"].capacity)
    & (abs(result.transition.mass_residual) <= 1.0e-12)
)
payload = {
    "benchmark": "particle-capacity-growth",
    "passed": passed,
    "median_seconds": statistics.median(durations),
    "minimum_seconds": min(durations),
    "initial_capacity": latest["particles"].capacity,
    "target_capacity": result.epoch.dynamics.bodies.capacity,
    "mass_residual": float(result.transition.mass_residual),
}
print(json.dumps(payload, indent=2, allow_nan=False))
if not passed:
    raise SystemExit(1)
