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
        return runpy.run_path("examples/monolithic_reactive_cfd_dem.py")


warm = run()
durations = []
for _ in range(3):
    started = time.perf_counter()
    latest = run()
    durations.append(time.perf_counter() - started)
result = latest["result"]
print(
    json.dumps(
        {
            "benchmark": "reactive-monolithic-newton",
            "passed": bool(warm["result"].successful & result.successful),
            "median_seconds": statistics.median(durations),
            "minimum_seconds": min(durations),
            "nonlinear_iterations": int(result.nonlinear.diagnostics.iterations),
            "linear_iterations": int(result.nonlinear.diagnostics.linear_iterations),
            "preconditioner": result.preconditioner.mode.value,
        },
        indent=2,
    )
)
