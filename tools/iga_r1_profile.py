#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import phydrax as phx


def assemble(report: dict[str, object]) -> phx.qualification.CapabilityProfile:
    if report.get("kind") != "iga-r1-candidate-qualification":
        raise ValueError("Input is not an IGA R1 qualification report.")
    cases = report.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("IGA R1 qualification report has no cases.")
    supports = []
    for case_id in sorted(cases):
        case = cases[case_id]
        if not isinstance(case, dict) or not case.get("passed"):
            raise ValueError(f"Qualification case {case_id!r} did not pass.")
        attributes = case.get("support")
        if not isinstance(attributes, dict):
            raise ValueError(f"Qualification case {case_id!r} has no support tuple.")
        supports.append(phx.qualification.SupportTuple("IGA.Core.Tensor", attributes))
    return phx.qualification.CapabilityProfile(
        "IGA.Core.Tensor",
        "phydrax",
        "canonical",
        tuple(supports),
        required_gates=("code-verification", "solution-verification", "operational"),
        release_evidence=(),
        released=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "qualification",
        type=Path,
        nargs="?",
        default=Path("benchmarks/iga_r1_qualification.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/iga_r1_candidate_profile.json"),
    )
    arguments = parser.parse_args()
    report = json.loads(arguments.qualification.read_text())
    profile = assemble(report)
    payload = {
        "kind": "candidate-capability-profile",
        "release_ready": False,
        "profile": profile.to_record(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
