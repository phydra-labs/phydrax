#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.proprioception import (
    MILEUSNIC_SPINDLE_2006_DOI,
    MileusnicSpindle2006Plan,
    MileusnicSpindleInput,
)


def qualify() -> dict[str, object]:
    runtime = MileusnicSpindle2006Plan().prepare()
    rest = MileusnicSpindleInput(1.0, 0.0, 0.0, 0.0, 0.0)
    state = runtime.initialize(rest)
    equilibrium = runtime.rates(state, rest)
    driven = MileusnicSpindleInput(1.02, 0.1, 0.0, 70.0, 70.0)
    step_s = runtime.plan.maximum_step_s
    step_count = round(runtime.parameters.dynamic_time_constant_s.item() / step_s)

    def step(current, _):
        candidate = runtime.candidate(current, driven, step_s)
        return candidate.commit(), candidate.evidence.successful

    final, successful = jax.lax.scan(step, state, xs=None, length=step_count)
    output = runtime.output(final, driven)
    dynamic_target = runtime._fusimotor_targets(driven)[0]
    expected_dynamic = dynamic_target * (1.0 - jnp.exp(-1.0))
    dynamic_error = jnp.abs(final.bag1_dynamic_activation - expected_dynamic)
    acceleration_residual = jnp.max(
        jnp.abs(equilibrium.branch_tension_acceleration_force_unit_per_s2)
    )
    tolerance = 2.0e-4
    passed = (
        jnp.all(successful)
        & (acceleration_residual <= 1.0e-9)
        & (dynamic_error <= tolerance)
        & (output.primary_afferent_pps > 0.0)
        & (output.secondary_afferent_pps > 0.0)
    )
    return {
        "qualification": "mileusnic-brown-lan-loeb-2006-feline-spindle",
        "source_doi": MILEUSNIC_SPINDLE_2006_DOI,
        "passed": bool(passed),
        "equilibrium_acceleration_residual": float(acceleration_residual),
        "one_tau_dynamic_activation_error": float(dynamic_error),
        "primary_afferent_pps": float(output.primary_afferent_pps),
        "secondary_afferent_pps": float(output.secondary_afferent_pps),
        "claim_scope": (
            "feline soleus fit and feline medial-gastrocnemius validation; "
            "no human or closed-loop reflex claim"
        ),
        "gto_status": (
            "not implemented: the source requires per-fiber collagen topology and "
            "its printed nonlinear collagen exponent is not unambiguous enough for "
            "an independently reproducible implementation"
        ),
    }


def main() -> None:
    payload = qualify()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
