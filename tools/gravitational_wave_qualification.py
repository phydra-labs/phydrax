# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Run the end-to-end gravitational-wave inference qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import phydrax as phx
from examples.gravitational_wave_inference import run as run_inference
from tools.uq_benchmarks.configuration import get_configuration
from tools.uq_benchmarks.gravitational_waves import GRAVITATIONAL_WAVE_SCENARIOS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tmp/gravitational-wave-qualification"),
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    inference = run_inference(output=arguments.output_dir / "inference.phxresult")
    archive = phx.uq.read_result_archive(inference["result"])
    configuration = get_configuration("smoke")
    scenarios = tuple(
        scenario(configuration, 20260911 + index)
        for index, scenario in enumerate(GRAVITATIONAL_WAVE_SCENARIOS.values())
    )
    payload = {
        "passed": bool(
            inference["valid"]
            and inference["absolute_error"] <= 0.2
            and all(scenario.passed for scenario in scenarios)
            and archive.kind == "nested_sampling"
            and archive.metadata["context"]["normalization"] == "absolute"
        ),
        "inference": inference,
        "archive": {
            "kind": archive.kind,
            "context_id": archive.metadata["context"]["context_id"],
        },
        "scenarios": [scenario.as_dict() for scenario in scenarios],
    }
    destination = arguments.output_dir / "qualification.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise RuntimeError(f"Gravitational-wave qualification failed; see {destination}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
