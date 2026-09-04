#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.energetics import (
    UchidaUmberger2010Parameters,
    UchidaUmberger2010Plan,
)


def main() -> None:
    plan = UchidaUmberger2010Plan(
        UchidaUmberger2010Parameters(
            jnp.asarray((0.5, 0.8)),
            jnp.asarray((0.5, 0.7)),
            jnp.asarray((0.1, 0.12)),
            jnp.asarray((10.0, 10.0)),
        ),
        ("muscle-a", "muscle-b"),
    )
    result = plan.evaluate(
        jnp.asarray((0.8, 0.7)),
        jnp.asarray((0.7, 0.6)),
        jnp.asarray((100.0, 120.0)),
        jnp.asarray((1.0, 0.9)),
        jnp.asarray((0.1, 0.12)),
        jnp.asarray((-0.01, 0.01)),
    )
    payload = {
        "model_id": result.model_id,
        "successful": bool(result.evidence.successful),
        "muscle_metabolic_power_W": result.muscle_metabolic_power_W.tolist(),
        "total_muscle_metabolic_power_W": float(
            result.total_muscle_metabolic_power_W
        ),
        "claim_scope": "muscle-only phenomenological power; basal and thermal fields excluded",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
