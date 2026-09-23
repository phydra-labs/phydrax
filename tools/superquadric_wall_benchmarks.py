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
            Path(__file__).resolve().parents[1] / "examples/superquadric_triangle_wall.py"
        )


warm = run()
durations = []
for _ in range(3):
    started = time.perf_counter()
    latest = run()
    durations.append(time.perf_counter() - started)
response = latest["result"].evaluation.walls[0]
force_residual = jnp.linalg.norm(
    jnp.sum(response.particle_load.force, axis=0) + response.reaction_force
)
witness_residual = float(response.geometry.witness_residual.max())
passed = bool(
    warm["result"].successful
    & latest["result"].successful
    & jnp.isfinite(force_residual)
    & jnp.isfinite(witness_residual)
    & (force_residual <= 1.0e-12)
    & (witness_residual <= 1.0e-8)
)
payload = {
    "benchmark": "superquadric-triangle-wall",
    "passed": passed,
    "median_seconds": statistics.median(durations),
    "minimum_seconds": min(durations),
    "candidate_count": response.geometry.geometry.valid.shape[0],
    "force_residual": float(force_residual),
    "witness_residual": witness_residual,
}
print(json.dumps(payload, indent=2, allow_nan=False))
if not passed:
    raise SystemExit(1)
