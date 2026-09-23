#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmarks._io import write_json_atomic
from benchmarks._runtime import capture_environment

from .cases import qualification_setups
from .contracts import DirectCollocationQualificationArtifact
from .graduation import evaluate_direct_collocation_graduation
from .runner import run_qualification_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        action="append",
        choices=("native", "ipopt"),
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/direct_collocation_qualification.json"),
    )
    parser.add_argument(
        "--documentation-evidence",
        type=Path,
        default=None,
        help="Existing governed documentation evidence required for product graduation.",
    )
    arguments = parser.parse_args()
    backends = tuple(arguments.backend or ("native",))
    setups = qualification_setups()
    records = tuple(
        run_qualification_case(setup, backend) for setup in setups for backend in backends
    )
    documentation_complete = bool(
        arguments.documentation_evidence is not None
        and arguments.documentation_evidence.is_file()
    )
    graduation = evaluate_direct_collocation_graduation(
        records,
        documentation_complete=documentation_complete,
        artifact_present=False,
    )
    runtime = capture_environment().to_dict()
    metadata = {
        "qualification": "direct-collocation",
        "source_id": os.environ.get("GITHUB_SHA", "working-tree"),
        "package_fingerprint": runtime["package_fingerprint"],
        "platform": runtime["platform"],
        "python": runtime["python_version"],
        "jax": runtime["jax"]["version"],
        "numpy": runtime["numpy_version"],
        "dtype": runtime["default_float_dtype"],
        "backends": list(backends),
        "runtime": runtime,
        "setups": [
            {
                "case_id": setup.case.case_id,
                "problem_id": setup.problem.problem_id,
                "plan_id": setup.plan.plan_id,
            }
            for setup in setups
        ],
    }
    artifact = DirectCollocationQualificationArtifact.create(
        metadata=metadata,
        cases=tuple(setup.case for setup in setups),
        records=records,
        graduation=graduation,
    )
    required = tuple(setup.case.case_id for setup in setups)
    artifact.verify(required_case_ids=required)
    write_json_atomic(arguments.output, artifact.to_dict())
    persisted = DirectCollocationQualificationArtifact.from_dict(
        json.loads(arguments.output.read_text(encoding="utf-8"))
    )
    graduation = evaluate_direct_collocation_graduation(
        records,
        documentation_complete=documentation_complete,
        artifact_present=True,
    )
    artifact = DirectCollocationQualificationArtifact.create(
        metadata=metadata,
        cases=tuple(setup.case for setup in setups),
        records=records,
        graduation=graduation,
    )
    artifact.verify(required_case_ids=required)
    write_json_atomic(arguments.output, artifact.to_dict())
    persisted = DirectCollocationQualificationArtifact.from_dict(
        json.loads(arguments.output.read_text(encoding="utf-8"))
    )
    print(json.dumps(persisted.to_dict(), indent=2, allow_nan=False))
    return 0 if bool(persisted.graduation["production_ready"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
