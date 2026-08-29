#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint

from .cases import qualification_setups
from .contracts import DirectCollocationQualificationArtifact
from .graduation import evaluate_direct_collocation_graduation
from .runner import run_qualification_case


def _package_fingerprint() -> str:
    packages = sorted(
        (distribution.metadata["Name"].lower(), distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    )
    return canonical_fingerprint(packages)


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
        run_qualification_case(setup, backend)
        for setup in setups
        for backend in backends
    )
    graduation = evaluate_direct_collocation_graduation(
        records,
        documentation_complete=True,
        artifact_present=True,
    )
    metadata = {
        "qualification": "direct-collocation",
        "source_id": os.environ.get("GITHUB_SHA", "working-tree"),
        "package_fingerprint": _package_fingerprint(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "dtype": str(jnp.asarray(0.0).dtype),
        "backends": list(backends),
    }
    artifact = DirectCollocationQualificationArtifact.create(
        metadata=metadata,
        cases=tuple(setup.case for setup in setups),
        records=records,
        graduation=graduation,
    )
    required = tuple(setup.case.case_id for setup in setups)
    artifact.verify(required_case_ids=required)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact.to_dict(), indent=2) + "\n")
    print(json.dumps(artifact.to_dict(), indent=2))


if __name__ == "__main__":
    main()
