# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Trace a bounded line of Kerr shadow rays with explicit terminal events."""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    astro = phx.applications.astrophysics
    scale = phx.RelativityScaleContract.geometric(phx.units.SOLAR_MASS)
    convention = phx.metrix.RelativityConvention.canonical()
    mass = 1.0
    spin = 0.7
    chart = phx.metrix.CoordinateChart("ingoing-kerr", ("v", "r", "theta", "phi_tilde"))
    metric = phx.metrix.ingoing_kerr_metric(mass, spin, chart=chart)
    outer_horizon = float(phx.metrix.kerr_horizon_radii(mass, spin)[1])

    horizontal = jnp.asarray((-0.5, -0.25, 0.0, 0.25, 0.5))
    pixels = jnp.stack((horizontal, jnp.zeros_like(horizontal)), axis=-1)
    screen = astro.GRObserverScreenPlan(
        metric,
        jnp.asarray((0.0, 20.0, jnp.pi / 2.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0, 0.0)),
        jnp.asarray((0.0, -1.0, 0.0, 0.0)),
        jnp.asarray((0.0, 0.0, -1.0, 0.0)),
        pixels,
        scale=scale,
        convention=convention,
        coordinate_unit=scale.dimensional_scale.length_unit,
        affine_parameter_unit=scale.dimensional_scale.length_unit,
        temporal_direction="past",
    ).initialize()
    if not bool(jnp.all(screen.valid)):
        raise RuntimeError("The Kerr observer tetrad did not initialize all ray lanes.")

    capture_radius = outer_horizon + 0.25
    escape_radius = 30.0
    events = astro.GRRayEventSurfaces(
        capture_margin=lambda affine, point, tangent: point[1] - capture_radius,
        escape_margin=lambda affine, point, tangent: escape_radius - point[1],
        event_id="kerr-shadow-capture-or-escape",
    )
    plan = astro.GRRayPlan.from_screen(
        metric,
        screen,
        jnp.linspace(0.0, 80.0, 65),
        scale=scale,
        convention=convention,
        coordinate_unit=scale.dimensional_scale.length_unit,
        affine_parameter_unit=scale.dimensional_scale.length_unit,
        events=events,
        transport_screen_basis=True,
        constants_of_motion=(
            astro.GRCoordinateMomentumConstant(0, name="energy", sign=-1.0),
            astro.GRCoordinateMomentumConstant(3, name="axial-angular-momentum"),
        ),
        maximum_steps=2048,
    )
    rays = plan.trace()

    event_names = [
        astro.GRRayEventCode(int(code)).name.lower() for code in rays.event_code
    ]
    captured = rays.event_code == int(astro.GRRayEventCode.CAPTURE)
    escaped = rays.event_code == int(astro.GRRayEventCode.ESCAPE)
    if not bool(jnp.any(captured) & jnp.any(escaped)):
        raise RuntimeError("The bounded ray fan did not bracket the Kerr shadow.")
    print("pixel_horizontal", [float(value) for value in horizontal])
    print("terminal_events", event_names)
    print("capture_escape_counts", int(jnp.sum(captured)), int(jnp.sum(escaped)))
    print("qualified_lanes", int(jnp.sum(rays.qualified)), "of", rays.status.shape[0])
    print("max_killing_drift", float(jnp.max(rays.constant_relative_drift)))
    print("max_null_residual", float(jnp.max(rays.null_residual)))


if __name__ == "__main__":
    main()
