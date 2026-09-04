#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.personalization import (
    PhysicalRelativeForceCalibrationPlan,
    PhysicalRelativeForceCalibrationQualificationPlan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/skeletal_force_calibration_qualification.json"),
    )
    arguments = parser.parse_args()
    relative = jnp.asarray([0.0, 0.08, 0.2, 0.35, 0.55, 0.78, 1.0])
    uncertainty = jnp.full_like(relative, 0.5)
    expected_scale = 540.0

    identifiable = PhysicalRelativeForceCalibrationPlan(
        jnp.ones((relative.shape[0], 1)),
        ("load-cell-zero",),
        protocol_id="qualification-mvc-ramp",
        asset_id="qualification-load-cell",
    ).prepare()
    identifiable_state = identifiable.initialize(100.0)
    identifiable_candidate = identifiable.evaluate(
        identifiable_state,
        relative,
        expected_scale * relative + 2.5,
        uncertainty,
    )

    confounded = PhysicalRelativeForceCalibrationPlan(
        relative[:, None],
        ("gain-shaped-nuisance",),
        protocol_id="qualification-confounded-negative-control",
        asset_id="qualification-load-cell",
    ).prepare()
    confounded_state = confounded.initialize(100.0)
    confounded_candidate = confounded.evaluate(
        confounded_state,
        relative,
        expected_scale * relative,
        uncertainty,
    )
    evidence = PhysicalRelativeForceCalibrationQualificationPlan().evaluate(
        identifiable_candidate,
        confounded_candidate,
        expected_scale,
    )
    payload = {
        "protocol_id": identifiable.plan.protocol_id,
        "asset_id": identifiable.plan.asset_id,
        "measurement_equation": identifiable_candidate.evidence.measurement_equation,
        "recovered_scale_newton_per_relative_force": float(
            evidence.recovered_scale_newton_per_relative_force
        ),
        "relative_scale_error": float(evidence.relative_scale_error),
        "identifiable_control_accepted": bool(
            evidence.identifiable_control_accepted
        ),
        "confounded_control_rejected": bool(evidence.confounded_control_rejected),
        "confounded_scale_flagged": bool(evidence.confounded_scale_flagged),
        "valid": bool(evidence.valid),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
