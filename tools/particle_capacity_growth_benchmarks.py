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
        return runpy.run_path("examples/growing_reactive_particle_pool.py")


warm = run()
durations = []
for _ in range(3):
    started = time.perf_counter()
    latest = run()
    durations.append(time.perf_counter() - started)
print(
    json.dumps(
        {
            "benchmark": "particle-capacity-growth",
            "passed": bool(warm["result"].successful & latest["result"].successful),
            "median_seconds": statistics.median(durations),
            "minimum_seconds": min(durations),
            "target_capacity": latest["result"].epoch.dynamics.bodies.capacity,
        },
        indent=2,
    )
)
