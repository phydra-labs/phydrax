"""Branch-explicit Lambert transfer between two unit-circle positions."""

import jax.numpy as jnp

import phydrax as phx


def main():
    astro = phx.applications.astrodynamics
    context = astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0, 0.0), "TT")),
        astro.FrameDefinition("central-body", "inertial", pseudo_inertial=True),
    )
    result = astro.solve_lambert(
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
        0.5 * jnp.pi,
        1.0,
        context,
        astro.LambertPlan(max_revolutions=1),
    )
    if not bool(result.valid[0]):
        raise RuntimeError("Zero-revolution Lambert branch failed.")
    print(result.departure_velocity[result.valid])
    print(result.arrival_velocity[result.valid])


if __name__ == "__main__":
    main()
