#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run the source-named stochastic isometric motor-unit pool."""

from __future__ import annotations

import jax.random as jr

from phydrax.applications.skeletal_muscle.motor_units import (
    commit_fuglevand_winter_patla_1993,
    FuglevandWinterPatla1993Plan,
    FuglevandWinterPatla1993RandomInput,
)


def main() -> None:
    prepared = FuglevandWinterPatla1993Plan(
        120,
        event_capacity_per_unit=4,
        random_stream_id="example/fuglevand-1993/discharge",
    ).prepare()
    state = prepared.initialize()
    key = jr.key(20260903)
    for _ in range(200):
        random_input = FuglevandWinterPatla1993RandomInput(
            key,
            state.random_step,
            stream_id=prepared.plan.random_stream_id,
        )
        candidate = prepared.evaluate(
            state,
            0.6 * prepared.maximum_excitation,
            5.0,
            random_input,
        )
        if not bool(candidate.evidence.successful):
            raise RuntimeError(f"motor-unit step status={int(candidate.evidence.status)}")
        state = commit_fuglevand_winter_patla_1993(candidate, state)
    force = prepared.force(state)
    print(f"time_ms={float(force.time_ms):.1f}")
    print(f"terminal_force_arbitrary={float(force.total_force_arbitrary):.6g}")
    print("force_owner=FuglevandWinterPatla1993")


if __name__ == "__main__":
    main()
