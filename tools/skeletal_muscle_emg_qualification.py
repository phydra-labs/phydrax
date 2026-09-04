#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.electromyography import (
    MotorUnitActionPotentialTemplatePlan,
    PETERSEN_ROSTALSKI_2019_DOI,
    PetersenRostalski2019PlanarConductorPlan,
    PlanarConductorParameters,
)


def qualify() -> dict[str, object]:
    template = jnp.asarray([[[0.0, 1.0e-4, -5.0e-5, 0.0]]])
    templates = MotorUnitActionPotentialTemplatePlan(
        template,
        0.001,
        0,
        ("unit-0",),
        ("channel-0",),
        template_source_id="qualification-explicit-template",
    ).prepare()
    times = jnp.arange(8) * 0.001
    single = templates.synthesize(
        jnp.asarray(((0.0,),)), jnp.asarray(((True,),)), times
    )
    double = templates.synthesize(
        jnp.asarray(((0.0, 0.002),)), jnp.asarray(((True, True),)), times
    )
    expected_double = single.voltage_V + jnp.pad(
        single.voltage_V[:, :-2], ((0, 0), (2, 0))
    )
    superposition_error = jnp.max(jnp.abs(double.voltage_V - expected_double))

    frequency = 2.0 * jnp.pi * jnp.fft.fftfreq(8, d=0.005)
    source = jnp.zeros((8, 8), dtype=jnp.complex128).at[1, 0].set(1.0).at[-1, 0].set(1.0)

    def conductor(depth):
        return PetersenRostalski2019PlanarConductorPlan(
            frequency,
            frequency,
            jnp.ones((8, 8), dtype=jnp.complex128),
            jnp.asarray(((0.0, 0.0), (0.01, 0.0))),
            jnp.asarray((1.0, -1.0)),
            PlanarConductorParameters(0.5, 0.1, 0.04, 0.2, 0.003, 0.001, depth),
        ).evaluate(source)

    shallow = conductor(-0.005)
    deep = conductor(-0.02)
    depth_ratio = jnp.max(jnp.abs(deep.surface_voltage_V)) / jnp.max(
        jnp.abs(shallow.surface_voltage_V)
    )
    tolerance = 1.0e-12
    passed = (
        single.evidence.successful
        & double.evidence.successful
        & (superposition_error <= tolerance)
        & shallow.evidence.successful
        & deep.evidence.successful
        & (depth_ratio < 1.0)
    )
    return {
        "qualification": "skeletal-surface-emg",
        "physical_source_doi": PETERSEN_ROSTALSKI_2019_DOI,
        "passed": bool(passed),
        "template_superposition_error_V": float(superposition_error),
        "planar_conductor_deep_to_shallow_amplitude_ratio": float(depth_ratio),
        "planar_conductor_real_signal_residual": float(
            shallow.evidence.real_signal_residual
        ),
        "claim_scope": (
            "explicit supplied MUAP templates plus infinite planar surface conductor; "
            "no activation-to-EMG or intramuscular/limb-geometry claim"
        ),
    }


def main() -> None:
    payload = qualify()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
