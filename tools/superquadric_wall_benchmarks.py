#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import contextlib
import io
import json
import runpy
import statistics
import time


def run():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        return runpy.run_path("examples/superquadric_triangle_wall.py")


warm = run()
durations = []
for _ in range(3):
    started = time.perf_counter()
    latest = run()
    durations.append(time.perf_counter() - started)
response = latest["result"].evaluation.walls[0]
print(
    json.dumps(
        {
            "benchmark": "superquadric-triangle-wall",
            "passed": bool(warm["result"].successful & latest["result"].successful),
            "median_seconds": statistics.median(durations),
            "minimum_seconds": min(durations),
            "candidate_count": int(response.geometry.geometry.valid.shape[0]),
            "witness_residual": float(response.geometry.witness_residual.max()),
        },
        indent=2,
    )
)
