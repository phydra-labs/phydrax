#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import math
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
            "1",
            "--repeats",
            "1",
            "--resolution",
            "6",
            "--seeds",
            "0",
            "--alpha",
            "0.5",
            "--posterior-samples",
            "4",
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
    for result in persisted["results"]:
        assert result["seeds"] == [0]
        assert result["ensemble_size"] == 1
        assert result["training_steps"] == [1]
        assert all(
            math.isfinite(value)
            for field in ("initial_losses", "final_losses", "validation_losses")
            for value in result[field]
        )
        assert result["evaluations"]
        for evaluation in result["evaluations"]:
            assert evaluation["valid_draw_count"] > 0
            assert evaluation["valid_draw_count"] == evaluation["total_draw_count"]
            assert all(
                math.isfinite(evaluation[field])
                for field in (
                    "relative_l2",
                    "epistemic_standard_deviation",
                    "crps",
                    "energy_score",
                    "pointwise_coverage",
                    "simultaneous_coverage",
                    "interval_width",
                )
            )
    assert persisted["results"][0]["laplace"]["posterior_sample_count"] == 4
    assert persisted["results"][0]["laplace"]["geometry_preserved"]
    assert "long_rollout" in set(table["name"])
    assert set(table["architecture"]) == {"fno", "deeponet"}
