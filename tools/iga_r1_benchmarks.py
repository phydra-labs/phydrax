#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from iga_closure_qualification import run as run_closure
from iga_h1_benchmarks import run as run_h1_benchmarks


def run(*, smoke: bool, warmup: int, repeats: int) -> dict[str, object]:
    start = time.perf_counter()
    h1 = run_h1_benchmarks(smoke=smoke, warmup=warmup, repeats=repeats)
    closure_start = time.perf_counter()
    closure = run_closure()
    closure_seconds = time.perf_counter() - closure_start
    return {
        "kind": "iga-r1-record-only-benchmark",
        "record_only": True,
        "wall_seconds": time.perf_counter() - start,
        "h1": h1,
        "closure": {
            "seconds": closure_seconds,
            "passed": closure["passed"],
            "case_ids": sorted(closure["cases"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/iga_r1.json"),
    )
    arguments = parser.parse_args()
    report = run(
        smoke=arguments.smoke,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
