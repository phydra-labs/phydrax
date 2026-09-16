#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Cold/warm timing for deterministic and stochastic blood-dose mechanics."""

import json
import time

from tools.circulating_blood_dose_qualification import qualify


def run() -> dict[str, object]:
    start = time.perf_counter()
    cold = qualify()
    middle = time.perf_counter()
    warm = qualify()
    end = time.perf_counter()
    if not cold["successful"] or not warm["successful"]:
        raise RuntimeError("Circulating-blood benchmark path failed qualification.")
    return {
        "cold_seconds": middle - start,
        "warm_seconds": end - middle,
        "path_count": warm["path_count"],
        "maximum_events_used": warm["maximum_events_used"],
        "model_id": warm["model_id"],
        "scope": "Performance only; no physiological or clinical validation claim.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
