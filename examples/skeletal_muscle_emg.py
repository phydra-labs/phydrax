#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.electromyography import (
    MotorUnitActionPotentialTemplatePlan,
)


def main() -> None:
    templates = MotorUnitActionPotentialTemplatePlan(
        jnp.asarray([[[0.0, 1.0e-4, -5.0e-5, 0.0]]]),
        0.001,
        0,
        ("unit-0",),
        ("bipolar-channel",),
        template_source_id="explicit-example-template",
    ).prepare()
    result = templates.synthesize(
        jnp.asarray(((0.0, 0.004),)),
        jnp.asarray(((True, True),)),
        jnp.arange(10) * 0.001,
    )
    payload = {
        "plan_id": result.plan_id,
        "successful": bool(result.evidence.successful),
        "sample_times_s": result.sample_times_s.tolist(),
        "surface_voltage_V": result.voltage_V[0].tolist(),
        "claim_scope": "supplied MUAP template superposition; not activation-to-EMG",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
