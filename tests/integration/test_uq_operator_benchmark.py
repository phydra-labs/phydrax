#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl


_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_operator_uq_cli_runs_end_to_end_and_writes_artifacts(tmp_path):
    environment = os.environ.copy()
    environment.setdefault("JAX_PLATFORM_NAME", "cpu")
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.operator_benchmarks",
            "--uq",
            "--quick",
            "--steps",
            "0",
            "--repeats",
            "1",
            "--resolution",
            "6",
            "--seeds",
            "0",
            "--alpha",
            "0.5",
            "--posterior-samples",
            "2",
            "--skip-laplace",
            "--output",
            str(tmp_path),
            "--commit-identity",
            "integration-test",
        ],
        cwd=str(_REPO_ROOT),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=5 * 60,
    )

    printed = json.loads(completed.stdout)
    persisted = json.loads(
        (tmp_path / "operator_uq_benchmarks.json").read_text(encoding="utf-8")
    )
    table = pl.read_parquet(tmp_path / "operator_uq_benchmarks.parquet")
    assert printed == persisted
    assert persisted["metadata"]["commit_identity"] == "integration-test"
    assert [result["architecture"] for result in persisted["results"]] == [
        "fno",
        "deeponet",
    ]
    assert "long_rollout" in set(table["name"])
    assert set(table["architecture"]) == {"fno", "deeponet"}
