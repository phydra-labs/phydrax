import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_orbit_to_calibrated_pixel_workflow():
    astro = phx.applications.astrodynamics
    context = astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0), "TT")),
        astro.FrameDefinition("earth", "icrf", pseudo_inertial=True),
    )
    initial = astro.CartesianOrbitState(
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
        context,
    )
    trajectory = phx.solver.IAS15Plan(
        relative_tolerance=1e-10, absolute_tolerance=1e-12
    ).solve(
        lambda time, position, velocity, args: (
            -position / jnp.maximum(jnp.sqrt(jnp.sum(position * position)) ** 3, 1e-30)
        ),
        initial.position,
        initial.velocity,
        jnp.linspace(0.0, 0.2, 5),
    )
    assert bool(jnp.all(trajectory.valid))

    physics = phx.applications.astrophysics
    wcs = physics.TangentSipWcsPlan(
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([32.0, 32.0]),
        jnp.eye(2),
        jnp.zeros((2, 2)),
        jnp.zeros((2, 2)),
    )
    final = trajectory.position[-1]
    sky = jnp.asarray([jnp.arctan2(final[1], final[0]), 0.0])
    pixel = wcs.world_to_pixel(sky)
    assert bool(pixel.valid)
    restored = wcs.pixel_to_world(pixel.coordinates)
    np.testing.assert_allclose(restored.coordinates, sky, atol=1e-9)
