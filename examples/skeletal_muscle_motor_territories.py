#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.fibers import MotorUnitTerritoryPlan


def main() -> None:
    territory = MotorUnitTerritoryPlan(
        ("unit-0", "unit-1"),
        ("fiber-0", "fiber-1", "fiber-2"),
        jnp.asarray((0, 1, 0)),
        jnp.asarray((1, 2, 3)),
        5,
        jnp.asarray((100.0, 150.0)),
        jnp.asarray((0.2, 0.1)),
        stimulus_source_id="declared-example-endplate-pulse",
    )
    stimulus = territory.bind_events(
        jnp.asarray(((1.0,), (1.05,))),
        jnp.asarray(((True,), (True,))),
        event_source_id="example-motor-unit-events",
    )
    payload = {
        "plan_id": territory.plan_id,
        "schedule_id": stimulus.schedule_id,
        "fiber_count_per_unit": territory.evidence.fiber_count_per_unit.tolist(),
        "current_at_1_08_ms_uA_per_cm2": stimulus.current(1.08).tolist(),
        "valid": bool(territory.evidence.valid),
        "claim_scope": "explicit event routing; no universal neuromuscular-junction law",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
