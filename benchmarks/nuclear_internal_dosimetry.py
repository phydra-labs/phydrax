#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Cold/warm timing for analytic internal-dosimetry numerical contracts."""

import json
import time

from tools.nuclear_internal_dosimetry_qualification import qualify


def run() -> dict[str, object]:
    start = time.perf_counter()
    cold = qualify()
    middle = time.perf_counter()
    warm = qualify()
    end = time.perf_counter()
    if not cold["successful"] or not warm["successful"]:
        raise RuntimeError("Internal-dosimetry benchmark path failed qualification.")
    return {
        "cold_seconds": middle - start,
        "warm_seconds": end - middle,
        "time_integral_bq_s": warm["time_integral_bq_s"],
        "spatial_nonzero_dose_gy": warm["spatial_nonzero_dose_gy"],
        "scope": "Performance only; no S-value-data or clinical accuracy claim.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
