#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import jax

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
    return hashlib.sha256(inspect.getsource(runner).encode("utf-8")).hexdigest()


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
        record = runner()
        runner_id = f"{runner.__module__}:{runner.__name__}"
        path = directory / f"{record.campaign_id}.zip"
        write_open_system_artifact(
            path,
            record,
            problem_id=record.campaign_id,
            plan_id=f"{record.campaign_id}:runner",
            backend=jax.default_backend(),
            runner_id=runner_id,
            code_fingerprint=_runner_fingerprint(runner),
        )
        reproduced = runner()
        verified = verify_open_system_artifact(
            path,
            reproduced,
            expected_runner_id=runner_id,
        )
        verified_campaigns.append(verified)
        summaries[record.campaign_id] = {
            "artifact": str(path),
            "artifact_sha256": verified.artifact_sha256,
            "approximation_valid": bool(record.approximation.valid),
            "physicality_status": record.physicality.status,
            "replay_valid": bool(record.replay.valid),
            "capacity_exhausted": bool(record.capacity_exhausted),
            "reproduction_verified": bool(verified.reproduction_verified),
        }
    graduation = run_open_system_graduation(verified_campaigns)
    decisions = {}
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
    return {
        "promoted": bool(graduation.promoted),
        "campaigns": summaries,
        "decisions": decisions,
        "permanent_stop_claims": list(graduation.stop_claims),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_campaign_matrix(arguments.output_directory),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
