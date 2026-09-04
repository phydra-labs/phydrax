#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run a sustained-isometric motor-unit fatigue protocol through public APIs."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from phydrax.applications import skeletal_muscle


def main() -> None:
    plan = skeletal_muscle.motor_units.PotvinFuglevand2017Plan()
    runtime = plan.prepare()
    initial = runtime.initialize()
    step_s = 0.02
    step_count = 1_000

    def step(state, _):
        candidate = runtime.candidate(state, 40.0, step_s)
        return candidate.commit(), jnp.stack(
            (
                candidate.output.total_force,
                candidate.output.total_force_capacity_fraction,
                candidate.evidence.successful.astype(candidate.output.total_force.dtype),
            )
        )

    final, history = jax.lax.scan(step, initial, xs=None, length=step_count)
    initial_output = runtime.evaluate(initial, 40.0)
    final_output = runtime.evaluate(final, 40.0)
    payload = {
        "model": skeletal_muscle.motor_units.POTVIN_FUGLEVAND_2017_MODEL_ID,
        "source_doi": skeletal_muscle.motor_units.POTVIN_FUGLEVAND_2017_DOI,
        "reference_sha": (
            skeletal_muscle.motor_units.POTVIN_FUGLEVAND_2017_REFERENCE_SHA
        ),
        "plan_id": plan.plan_id,
        "prepared_id": runtime.prepared_id,
        "duration_s": step_s * step_count,
        "common_excitation": 40.0,
        "initial_total_force": float(initial_output.total_force),
        "final_total_force": float(final_output.total_force),
        "final_capacity_fraction": float(
            final_output.total_force_capacity_fraction
        ),
        "recruited_unit_count": int(
            jnp.sum(runtime.evaluate(initial, 40.0).recruited)
        ),
        "all_steps_successful": bool(jnp.all(history[:, 2] == 1.0)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
