#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
import pytest

import phydrax as phx
from phydrax.discretization.lattice_boltzmann._collision import (
    collide_bgk,
    collide_trt,
    quadratic_equilibrium,
)
from phydrax.discretization.lattice_boltzmann._forcing import guo_raw_source


def _cell_grid(shape, *, periodic=None, lengths=None):
    dimension = len(shape)
    periodic = (True,) * dimension if periodic is None else tuple(periodic)
    lengths = (1.0,) * dimension if lengths is None else tuple(lengths)
    plan = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic[axis])
            for axis, count in enumerate(shape)
        ),
        axis_names=tuple("xyz"[:dimension]),
    )
    return plan.prepare(
        jnp.asarray(
            (
                (0.0,) * dimension,
                lengths,
            )
        )
    )


@pytest.mark.parametrize(
    "velocity_set", [phx.discretization.D2Q9(), phx.discretization.D3Q19()]
)
def test_lattice_velocity_sets_satisfy_hydrodynamic_moments(velocity_set):
    c = np.asarray(velocity_set.velocities, dtype=float)
    w = np.asarray(velocity_set.weights)
    opposite = np.asarray(velocity_set.opposite)
    dimension = velocity_set.dimension
    identity = np.eye(dimension)
    cs2 = float(velocity_set.sound_speed_squared)

    assert np.array_equal(opposite[opposite], np.arange(velocity_set.population_count))
    assert np.array_equal(c[opposite], -c)
    np.testing.assert_allclose(np.sum(w), 1.0, atol=1e-14)
    np.testing.assert_allclose(oe.contract("q,qa->a", w, c), 0.0, atol=1e-14)
    np.testing.assert_allclose(
        oe.contract("q,qa,qb->ab", w, c, c), cs2 * identity, atol=1e-14
    )
    expected_fourth = cs2**2 * (
        oe.contract("ab,cd->abcd", identity, identity)
        + oe.contract("ac,bd->abcd", identity, identity)
        + oe.contract("ad,bc->abcd", identity, identity)
    )
    np.testing.assert_allclose(
        oe.contract("q,qa,qb,qc,qd->abcd", w, c, c, c, c),
        expected_fourth,
        atol=1e-14,
    )


def test_lattice_discretization_requires_isotropic_cell_centres():
    grid = _cell_grid((8, 8))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()

    assert discretization.population_shape == (8, 8, 9)
    assert discretization.population_space.layout.value_shape == (8, 8, 9)
    assert discretization.velocity_space.layout.value_shape == (8, 8, 2)
    assert discretization.preparation.resource_counts
    assert discretization.precision_evidence_id is not None

    anisotropic = _cell_grid((8, 8), lengths=(1.0, 2.0))
    with pytest.raises(ValueError, match="equal cell size"):
        phx.discretization.LatticeBoltzmannPlan(
            anisotropic, phx.discretization.D2Q9()
        ).prepare()

    point_grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(8),
            phx.discretization.UniformAxisSpec(8),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    with pytest.raises(ValueError, match="cell-centred"):
        phx.discretization.LatticeBoltzmannPlan(
            point_grid, phx.discretization.D2Q9()
        ).prepare()

    float32 = phx.discretization.LatticeBoltzmannPrecisionPolicy(
        population_dtype=jnp.float32
    )
    assert float32.population_dtype == "float32"


def test_lattice_scaling_round_trips_and_derives_relaxation():
    scaling = phx.discretization.LatticeBoltzmannScaling(0.02, 0.001, 2.0)
    velocity = jnp.asarray((0.4, -0.2))
    viscosity = jnp.asarray(0.015)
    acceleration = jnp.asarray((0.1, 0.0))

    np.testing.assert_allclose(
        scaling.physical_velocity(scaling.lattice_velocity(velocity)), velocity
    )
    np.testing.assert_allclose(
        scaling.physical_viscosity(scaling.lattice_viscosity(viscosity)), viscosity
    )
    np.testing.assert_allclose(
        scaling.physical_acceleration(scaling.lattice_acceleration(acceleration)),
        acceleration,
    )
    rate = scaling.relaxation_rate(viscosity)
    lattice_viscosity = scaling.lattice_viscosity(viscosity)
    np.testing.assert_allclose(
        lattice_viscosity,
        scaling.sound_speed_squared * (1.0 / rate - 0.5),
    )


def test_equilibrium_collision_and_guo_moments_are_consistent():
    velocity_set = phx.discretization.D2Q9()
    precision = phx.discretization.LatticeBoltzmannPrecisionPolicy()
    density = jnp.asarray([[1.0, 1.2], [0.9, 1.1]])
    velocity = jnp.broadcast_to(jnp.asarray((0.04, -0.025)), (2, 2, 2))
    equilibrium = quadratic_equilibrium(density, velocity, velocity_set, precision)
    c = jnp.asarray(velocity_set.velocities, dtype=equilibrium.dtype)

    np.testing.assert_allclose(jnp.sum(equilibrium, axis=-1), density, atol=1e-14)
    np.testing.assert_allclose(
        oe.contract("...q,qd->...d", equilibrium, c),
        density[..., None] * velocity,
        atol=1e-14,
    )

    force = jnp.broadcast_to(jnp.asarray((1e-5, -2e-5)), velocity.shape)
    source = guo_raw_source(velocity, force, velocity_set, precision)
    np.testing.assert_allclose(jnp.sum(source, axis=-1), 0.0, atol=1e-14)
    np.testing.assert_allclose(oe.contract("...q,qd->...d", source, c), force, atol=1e-14)

    rate = jnp.asarray(1.3)
    bgk = collide_bgk(equilibrium, equilibrium, source, rate)
    trt = collide_trt(
        equilibrium,
        equilibrium,
        source,
        rate,
        rate,
        velocity_set.opposite,
    )
    np.testing.assert_allclose(trt, bgk, atol=1e-14)
