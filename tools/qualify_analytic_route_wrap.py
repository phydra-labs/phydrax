#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from phydrax.applications.robotics import (
    PlanarCylinderRouteWrapPlan,
    SphereRouteWrapPlan,
)


def qualify() -> dict[str, object]:
    start = jnp.asarray((-2.0, 0.35, 0.0))
    end = jnp.asarray((2.1, 0.65, 0.0))
    sphere = SphereRouteWrapPlan(64).prepare(jnp.zeros(3), 1.0)
    cylinder = PlanarCylinderRouteWrapPlan(64).prepare(
        jnp.zeros(3), jnp.asarray((0.0, 0.0, 1.0)), 1.0, 4.0
    )
    sphere_result = sphere.evaluate(start, end)
    cylinder_result = cylinder.evaluate(start, end)
    direction = jnp.asarray((0.01, -0.02, 0.005))
    sphere_jvp = jax.jvp(
        lambda value: sphere.evaluate(value, end).total_length_m,
        (start,),
        (direction,),
    )[1]
    epsilon = 1.0e-5
    sphere_difference = (
        sphere.evaluate(start + epsilon * direction, end).total_length_m
        - sphere.evaluate(start - epsilon * direction, end).total_length_m
    ) / (2.0 * epsilon)
    derivative_error = jnp.abs(sphere_jvp - sphere_difference)
    tolerance = 2.0e-6
    passed = (
        sphere_result.evidence.successful
        & cylinder_result.evidence.successful
        & sphere_result.evidence.applied
        & cylinder_result.evidence.applied
        & (sphere_result.evidence.endpoint_tangency_residual <= tolerance)
        & (sphere_result.evidence.surface_residual <= tolerance)
        & (cylinder_result.evidence.endpoint_tangency_residual <= tolerance)
        & (cylinder_result.evidence.surface_residual <= tolerance)
        & (derivative_error <= tolerance)
    )
    return {
        "qualification": "opensim-derived-bounded-analytic-route-wrap",
        "source_revision": "86b30588374650fbaf012a345a836a64f6855522",
        "passed": bool(passed),
        "sphere_total_length_m": float(sphere_result.total_length_m),
        "sphere_surface_length_m": float(sphere_result.surface_length_m),
        "sphere_tangency_residual": float(
            sphere_result.evidence.endpoint_tangency_residual
        ),
        "sphere_surface_residual": float(sphere_result.evidence.surface_residual),
        "cylinder_total_length_m": float(cylinder_result.total_length_m),
        "cylinder_surface_length_m": float(cylinder_result.surface_length_m),
        "cylinder_tangency_residual": float(
            cylinder_result.evidence.endpoint_tangency_residual
        ),
        "cylinder_surface_residual": float(
            cylinder_result.evidence.surface_residual
        ),
        "sphere_directional_derivative_error": float(derivative_error),
        "tolerance": tolerance,
    }


def main() -> None:
    payload = qualify()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
