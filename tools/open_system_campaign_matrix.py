#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax

from benchmarks._io import write_json_atomic
from tools.open_system_campaigns import (
    dense_trajectory_campaign,
    distillation_campaign,
    gaussian_campaign,
    heom_campaign,
    lpdo_campaign,
    memory_campaign,
    mps_campaign,
    neural_campaign,
    process_recovery_campaign,
    run_open_system_graduation,
    verify_open_system_artifact,
    write_open_system_artifact,
)


def _runner_fingerprint(runner) -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = [
        *root.joinpath("phydrax").rglob("*.py"),
        *root.joinpath("tools", "open_system_campaigns").rglob("*.py"),
        root / "tools" / "open_system_campaign_matrix.py",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    digest.update(f"{runner.__module__}:{runner.__name__}".encode("utf-8"))
    return digest.hexdigest()


def run_campaign_matrix(output_directory: str):
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    runners = (
        gaussian_campaign,
        dense_trajectory_campaign,
        mps_campaign,
        lpdo_campaign,
        heom_campaign,
        memory_campaign,
        process_recovery_campaign,
        distillation_campaign,
        neural_campaign,
    )
    verified_campaigns = []
    summaries = {}
    for runner in runners:
        runner_id = f"{runner.__module__}:{runner.__name__}"
        try:
            record = runner()
            problem_id = record.campaign_id
            plan_id = f"{record.campaign_id}:runner"
            backend = jax.default_backend()
            code_fingerprint = _runner_fingerprint(runner)
            path = directory / f"{record.campaign_id}.zip"
            write_open_system_artifact(
                path,
                record,
                problem_id=problem_id,
                plan_id=plan_id,
                backend=backend,
                runner_id=runner_id,
                code_fingerprint=code_fingerprint,
            )
            reproduced = runner()
            verified = verify_open_system_artifact(
                path,
                reproduced,
                expected_runner_id=runner_id,
                expected_problem_id=problem_id,
                expected_plan_id=plan_id,
                expected_backend=backend,
                expected_code_fingerprint=code_fingerprint,
            )
            verified_campaigns.append(verified)
            summaries[record.campaign_id] = {
                "artifact": str(path),
                "artifact_sha256": verified.artifact_sha256,
                "approximation_valid": bool(record.approximation.valid),
                "physicality_status": record.physicality.status,
                "replay_valid": bool(record.replay.valid),
                "capacity_exhausted": bool(record.capacity_exhausted),
                "execution_success": bool(record.execution_success),
                "provider_status": "success",
                "provider_error": None,
                "reproduction_verified": bool(verified.reproduction_verified),
            }
        except Exception as error:
            failure = {
                "runner_id": runner_id,
                "provider_status": "failed",
                "provider_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            write_json_atomic(
                directory / f"{runner.__name__}.failure.json",
                failure,
            )
            summaries[runner.__name__] = failure
    decisions = {}
    promoted = False
    stop_claims = []
    if len(verified_campaigns) == len(runners):
        graduation = run_open_system_graduation(verified_campaigns)
        for campaign_id, decision in zip(
            graduation.campaign_ids, graduation.decisions, strict=True
        ):
            decisions[campaign_id] = {
                "promoted": bool(decision.promoted),
                "missing_axes": list(decision.missing_axes),
                "missing_quantities": list(decision.missing_quantities),
                "missing_physicality": list(decision.missing_physicality),
                "physicality_satisfied": bool(decision.physicality_satisfied),
                "capacity_available": bool(decision.capacity_available),
                "precision_satisfied": bool(decision.precision_satisfied),
                "archive_verified": bool(decision.archive_verified),
            }
        promoted = bool(graduation.promoted)
        stop_claims = list(graduation.stop_claims)
    return {
        "promoted": promoted,
        "complete": len(verified_campaigns) == len(runners),
        "campaigns": summaries,
        "decisions": decisions,
        "permanent_stop_claims": stop_claims,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args()
    report = run_campaign_matrix(arguments.output_directory)
    output = Path(arguments.output_directory) / "campaign-matrix.json"
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["promoted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
