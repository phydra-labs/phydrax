#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.fibers import (
    PrescribedFiberStimulusSchedule,
    SkeletalFiberBundlePlan,
)


def qualify() -> dict[str, object]:
    mask = jnp.zeros((1, 2, 7), dtype=bool).at[0, 0, 0].set(True)
    schedule = PrescribedFiberStimulusSchedule(
        jnp.asarray([0.0]),
        jnp.asarray([0.05]),
        jnp.asarray([150.0]),
        mask,
    )
    runtime = SkeletalFiberBundlePlan(
        ("stimulated", "control"),
        7,
        jnp.asarray((12.0, 12.0)),
        jnp.asarray((0.2, 0.2)),
        schedule,
        maximum_step_ms=0.05,
    ).prepare()
    initial = runtime.initialize()
    candidate = runtime.candidate(initial, 0.05)
    final = candidate.commit()
    output = runtime.output(final)
    stimulated_change = output.membrane_potential_mV[0] - initial.values[0, :, 0]
    control_change = output.membrane_potential_mV[1] - initial.values[1, :, 0]
    neighbor_transfer = stimulated_change[1]
    target_selectivity = stimulated_change[0] - control_change[0]
    passed = (
        candidate.evidence.successful
        & jnp.all(jnp.isfinite(output.membrane_potential_mV))
        & (stimulated_change[0] > 0.0)
        & (neighbor_transfer > control_change[1])
        & (target_selectivity > 0.0)
    )
    return {
        "qualification": "shorten-2007-prescribed-stimulus-fiber-bundle",
        "passed": bool(passed),
        "plan_id": runtime.plan.plan_id,
        "prepared_id": runtime.prepared_id,
        "solver_steps": int(candidate.evidence.solver_steps),
        "stimulated_node_delta_mV": float(stimulated_change[0]),
        "neighbor_delta_mV": float(neighbor_transfer),
        "control_node_delta_mV": float(control_change[0]),
        "target_selectivity_mV": float(target_selectivity),
        "event_aligned": bool(candidate.evidence.event_aligned),
    }


def main() -> None:
    payload = qualify()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
