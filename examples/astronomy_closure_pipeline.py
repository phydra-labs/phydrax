"""Exact time semantics, adaptive orbit propagation, and astronomical WCS."""

import jax.numpy as jnp

import phydrax as phx


def main():
    astro = phx.applications.astrodynamics
    context = astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0), "TT")),
        astro.FrameDefinition("earth", "icrf", pseudo_inertial=True),
    )
    bundled = astro.bundled_astronomy_data_store()
    leap_seconds = astro.load_bundled_leap_seconds(bundled)
    earth_orientation = astro.load_bundled_earth_orientation(bundled)
    if not bool(earth_orientation.evaluate(0.0).valid):
        raise RuntimeError("Bundled EOP query lies outside declared coverage.")
    if int(leap_seconds.tai_minus_utc[-1]) != 37:
        raise RuntimeError("Bundled leap-second product failed integrity checks.")
    initial = astro.CartesianOrbitState(
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
        context,
    )
    trajectory = phx.solver.IAS15Plan(
        relative_tolerance=1.0e-10, absolute_tolerance=1.0e-12
    ).solve(
        lambda time, position, velocity, args: (
            -position / jnp.maximum(jnp.sqrt(jnp.sum(position * position)) ** 3, 1.0e-30)
        ),
        initial.position,
        initial.velocity,
        jnp.linspace(0.0, 0.2, 5),
    )
    if not bool(jnp.all(trajectory.valid)):
        raise RuntimeError("Adaptive orbit propagation failed.")

    wcs = phx.applications.astrophysics.TangentSipWcsPlan(
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([32.0, 32.0]),
        jnp.eye(2),
        jnp.zeros((2, 2)),
        jnp.zeros((2, 2)),
    )
    final = trajectory.position[-1]
    sky = jnp.asarray([jnp.arctan2(final[1], final[0]), 0.0])
    pixel = wcs.world_to_pixel(sky)
    if not bool(pixel.valid):
        raise RuntimeError("WCS projection failed.")
    print(trajectory.position[-1])
    print(pixel.coordinates)
    print(leap_seconds.table_id)


if __name__ == "__main__":
    main()
