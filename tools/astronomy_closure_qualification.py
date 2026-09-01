"""Deterministic qualification evidence for astronomy closure systems."""

import json

import jax.numpy as jnp

import phydrax as phx


def main():
    astro = phx.applications.astrodynamics
    solver = phx.solver.IAS15Plan(relative_tolerance=1e-10, absolute_tolerance=1e-12)
    trajectory = solver.solve(
        lambda time, position, velocity, args: -position,
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        jnp.linspace(0.0, 1.0, 5),
    )
    harmonic_error = jnp.abs(trajectory.position[-1, 0] - jnp.cos(1.0))

    physics = phx.applications.astrophysics
    wcs = physics.TangentSipWcsPlan(
        jnp.asarray([1.0, 0.5]),
        jnp.asarray([100.0, 100.0]),
        jnp.eye(2),
        jnp.zeros((2, 2)),
        jnp.zeros((2, 2)),
    )
    sky = jnp.asarray([1.001, 0.5005])
    round_trip = wcs.pixel_to_world(wcs.world_to_pixel(sky).coordinates)
    wcs_error = jnp.max(jnp.abs(round_trip.coordinates - sky))

    compact = phx.applications.compact_objects
    pressure = jnp.linspace(1.0e-6, 0.2, 128)
    eos = compact.EquationOfStateTable(pressure, 1.0 + 2.0 * pressure)
    tov = compact.TovPlan(eos, jnp.linspace(1.0e-4, 2.0, 512)).solve(0.1)

    report = {
        "kind": "astronomy-closure-qualification",
        "ias15_harmonic_error": float(harmonic_error),
        "ias15_accepted_steps": int(jnp.sum(trajectory.accepted_steps)),
        "ias15_rejected_steps": int(jnp.sum(trajectory.rejected_steps)),
        "wcs_round_trip_error": float(wcs_error),
        "tov_mass": float(tov.mass),
        "tov_radius": float(tov.radius),
        "passed": bool(
            jnp.all(trajectory.valid)
            & (harmonic_error < 2.0e-7)
            & round_trip.valid
            & (wcs_error < 1.0e-9)
            & tov.valid
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
