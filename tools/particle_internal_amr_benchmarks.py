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
        return runpy.run_path("examples/adaptive_catalyst_pellet.py")


warm = run()
durations = []
for _ in range(5):
    started = time.perf_counter()
    latest = run()
    durations.append(time.perf_counter() - started)
print(
    json.dumps(
        {
            "benchmark": "particle-internal-amr",
            "passed": bool(warm["result"].successful & latest["result"].successful),
            "median_seconds": statistics.median(durations),
            "minimum_seconds": min(durations),
            "active_fine_cells": int(latest["result"].accepted_state.fine_active.sum()),
        },
        indent=2,
    )
)
