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


def main() -> None:
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
    arguments = parser.parse_args()
    backends = tuple(arguments.backend or ("native",))
    setups = qualification_setups()
    records = tuple(
        run_qualification_case(setup, backend) for setup in setups for backend in backends
    )
    graduation = evaluate_direct_collocation_graduation(
        records,
        documentation_complete=True,
        artifact_present=True,
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
    print(json.dumps(artifact.to_dict(), indent=2))


if __name__ == "__main__":
    main()
