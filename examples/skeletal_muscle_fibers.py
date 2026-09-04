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


def main() -> None:
    target = jnp.zeros((1, 1, 5), dtype=bool).at[0, 0, 0].set(True)
    stimulus = PrescribedFiberStimulusSchedule(
        jnp.asarray([0.0]),
        jnp.asarray([0.05]),
        jnp.asarray([150.0]),
        target,
    )
    runtime = SkeletalFiberBundlePlan(
        ("fiber-0",),
        5,
        jnp.asarray([10.0]),
        jnp.asarray([0.1]),
        stimulus,
        maximum_step_ms=0.05,
    ).prepare()
    initial = runtime.initialize()
    candidate = runtime.candidate(initial, 0.05)
    final = candidate.commit()
    output = runtime.output(final)
    payload = {
        "prepared_id": runtime.prepared_id,
        "successful": bool(candidate.evidence.successful),
        "time_ms": float(final.time_ms),
        "surface_voltage_mV": output.membrane_potential_mV[0].tolist(),
        "cytosolic_calcium_uM": output.cytosolic_calcium_uM[0].tolist(),
        "force_bearing_crossbridge_uM": output.force_bearing_crossbridge_uM[0].tolist(),
        "solver_steps": int(candidate.evidence.solver_steps),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
