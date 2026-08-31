#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _discretization(*, periodic=(True, True), shape=(5, 4)):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic[axis])
            for axis, count in enumerate(shape)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, shape[1] / shape[0]))))
    return phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()


def test_periodic_pull_routes_every_population_direction():
    discretization = _discretization()
    boundary = phx.discretization.LatticeBoltzmannBoundaryPlan().prepare(discretization)
    populations = jnp.arange(
        np.prod(discretization.population_shape), dtype=float
    ).reshape(discretization.population_shape)
    density = jnp.ones(discretization.grid.shape)
    routed = boundary.route(populations, density, jnp.empty((0, 2)))

    expected = jnp.stack(
        tuple(
            jnp.roll(
                populations[..., direction],
                shift=tuple(int(value) for value in velocity),
                axis=(0, 1),
            )
            for direction, velocity in enumerate(
                np.asarray(discretization.velocity_set.velocities)
            )
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(routed, expected)


def test_nonperiodic_and_interior_solid_links_reflect_local_opposites():
    discretization = _discretization(periodic=(False, False))
    fluid = np.ones(discretization.grid.shape, dtype=bool)
    fluid[2, 2] = False
    snapshot = phx.discretization.LatticeBoltzmannGeometrySnapshot(discretization, fluid)
    boundary = phx.discretization.LatticeBoltzmannBoundaryPlan(geometry=snapshot).prepare(
        discretization
    )
    populations = jnp.arange(
        np.prod(discretization.population_shape), dtype=float
    ).reshape(discretization.population_shape)
    density = jnp.ones(discretization.grid.shape)
    routed = boundary.route(populations, density, jnp.empty((0, 2)))
    velocities = np.asarray(discretization.velocity_set.velocities)
    opposite = np.asarray(discretization.velocity_set.opposite)

    east = int(np.flatnonzero(np.all(velocities == (1, 0), axis=1))[0])
    np.testing.assert_array_equal(routed[0, :, east], populations[0, :, opposite[east]])
    np.testing.assert_array_equal(routed[3, 2, east], populations[3, 2, opposite[east]])
    np.testing.assert_array_equal(routed[2, 2], populations[2, 2])


def test_tangential_moving_wall_adds_documented_link_momentum():
    discretization = _discretization(periodic=(True, False))
    boundary = phx.discretization.LatticeBoltzmannBoundaryPlan(
        moving_faces=(("y", "upper"),)
    ).prepare(discretization)
    populations = jnp.zeros(discretization.population_shape)
    density = jnp.ones(discretization.grid.shape)
    wall_velocity = jnp.asarray(((0.1, 0.0),))
    routed = boundary.route(populations, density, wall_velocity)
    velocities = np.asarray(discretization.velocity_set.velocities)
    direction = int(np.flatnonzero(np.all(velocities == (1, -1), axis=1))[0])
    expected = (
        2.0
        * float(discretization.velocity_set.weights[direction])
        * 0.1
        / float(discretization.velocity_set.sound_speed_squared)
    )

    np.testing.assert_allclose(routed[:, -1, direction], expected, atol=1e-14)
    np.testing.assert_allclose(routed[:, 0, direction], 0.0, atol=1e-14)


def test_geometry_snapshot_is_detached_from_source_mask():
    discretization = _discretization()
    source = np.ones(discretization.grid.shape, dtype=bool)
    snapshot = phx.discretization.LatticeBoltzmannGeometrySnapshot(discretization, source)
    source[0, 0] = False

    assert bool(snapshot.fluid_mask[0, 0])
