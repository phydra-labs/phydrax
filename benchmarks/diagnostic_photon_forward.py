#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Cold/warm timing for the analytic diagnostic-photon forward route."""

import json
import time

from tools.diagnostic_photon_qualification import qualify


def run() -> dict[str, object]:
    start = time.perf_counter()
    cold = qualify()
    middle = time.perf_counter()
    warm = qualify()
    end = time.perf_counter()
    if not cold["accepted"] or not warm["accepted"]:
        raise RuntimeError("Diagnostic-photon benchmark path failed qualification.")
    return {
        "case": cold["case"],
        "cold_seconds": middle - start,
        "warm_seconds": end - middle,
        "coefficient_table": warm["identities"]["coefficient_table"],
        "detector_plan": warm["identities"]["detector_plan"],
        "scope": "Performance only; no transport, scanner, or clinical accuracy claim.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
