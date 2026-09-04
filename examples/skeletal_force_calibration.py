#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fit a protocol-bound relative-force scale and observe physical newtons."""

from __future__ import annotations

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.personalization import (
    commit_physical_relative_force_calibration,
    PhysicalRelativeForceCalibrationPlan,
)


def main() -> None:
    relative_force = jnp.asarray([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    prepared = PhysicalRelativeForceCalibrationPlan(
        jnp.ones((relative_force.shape[0], 1)),
        ("load-cell-zero",),
        protocol_id="example-isometric-ramp",
        asset_id="example-load-cell-calibration-2026-08",
    ).prepare()
    state = prepared.initialize(100.0)
    observed_force_newton = 480.0 * relative_force + 1.75
    candidate = prepared.evaluate(
        state,
        relative_force,
        observed_force_newton,
        jnp.full_like(relative_force, 0.5),
    )
    if not bool(candidate.evidence.successful):
        raise RuntimeError(f"calibration status={int(candidate.evidence.status)}")
    state = commit_physical_relative_force_calibration(candidate, state)
    observation = prepared.observe(state, jnp.asarray([0.2, 0.8]))
    print(
        "scale_newton_per_relative_force="
        f"{float(state.scale_newton_per_relative_force):.6f}"
    )
    print(f"force_newton={observation.force_newton}")
    print(f"protocol_id={observation.protocol_id}")
    print(f"asset_id={observation.asset_id}")
    print("semantic=observation model, not quantity conversion")


if __name__ == "__main__":
    main()
