#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.fatigue import (
    commit_liu_brown_yue_2002,
    LiuBrownYue2002Parameters,
    LiuBrownYue2002Plan,
    LiuBrownYue2002QualificationPlan,
)


def _record(state):
    return jnp.stack(
        (state.uncommitted_fraction, state.active_fraction, state.fatigued_fraction)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/skeletal_fatigue_qualification.json"),
    )
    arguments = parser.parse_args()
    prepared = LiuBrownYue2002Plan(
        LiuBrownYue2002Parameters(
            fatigue_rate_per_s=0.0206,
            recovery_rate_per_s=0.0084,
        ),
        muscle_id="healthy-handgrip-cohort-example",
        protocol_id="source-sustained-mvc-and-recovery-controls",
    ).prepare()

    state = prepared.initialize()
    sustained = [_record(state)]
    for _ in range(180):
        candidate = prepared.evaluate(state, 1.0, 1.0)
        if not bool(candidate.evidence.successful):
            raise RuntimeError(
                f"sustained fatigue trial failed with status {int(candidate.evidence.status)}"
            )
        state = commit_liu_brown_yue_2002(candidate, state)
        sustained.append(_record(state))

    recovery_state = prepared.initialize(
        uncommitted_fraction=0.0,
        active_fraction=0.05,
        fatigued_fraction=0.95,
    )
    recovery = [_record(recovery_state)]
    for _ in range(120):
        candidate = prepared.evaluate(recovery_state, 0.0, 1.0)
        if not bool(candidate.evidence.successful):
            raise RuntimeError(
                f"recovery trial failed with status {int(candidate.evidence.status)}"
            )
        recovery_state = commit_liu_brown_yue_2002(candidate, recovery_state)
        recovery.append(_record(recovery_state))

    evidence = LiuBrownYue2002QualificationPlan().evaluate(
        prepared,
        jnp.stack(sustained),
        jnp.stack(recovery),
    )
    payload = {
        "model_id": prepared.plan.model_id,
        "source_doi": "10.1016/S0006-3495(02)75580-X",
        "maximum_conservation_error": float(evidence.maximum_conservation_error),
        "minimum_compartment_fraction": float(evidence.minimum_compartment_fraction),
        "sustained_effort_fatigue_increases": bool(
            evidence.sustained_effort_fatigue_increases
        ),
        "zero_effort_fatigued_nonincreasing": bool(
            evidence.zero_effort_fatigued_nonincreasing
        ),
        "zero_effort_active_non_decreasing": bool(
            evidence.zero_effort_active_non_decreasing
        ),
        "intermittent_task_fidelity_released": False,
        "valid": bool(evidence.valid),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
