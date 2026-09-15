#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.compact_objects._accretion import (
    AccretionInitialDataStatus,
    FishboneMoncriefTorusPlan,
    MichelBondiAccretionPlan,
)
from phydrax.equations._relativistic_eos import GammaLawEOS
from phydrax.units import KILOGRAM, METER, SECOND


def _eos():
    scale = RelativityScaleContract(
        DimensionalScaleContract(METER, KILOGRAM, SECOND), 1, 1, 1, 1
    )
    return GammaLawEOS(scale, 4.0 / 3.0, minimum_density=0.0)


def test_michel_bondi_solution_satisfies_both_relativistic_integrals():
    plan = MichelBondiAccretionPlan(
        _eos(),
        1.0,
        1.0,
        0.01,
        root_iterations=56,
        bracket_samples=96,
    )
    radii = jnp.asarray(
        (0.7 * plan.critical_radius, plan.critical_radius, 2.0 * plan.critical_radius)
    )
    solution = plan.evaluate(radii)
    assert bool(jnp.all(solution.finite))
    assert bool(jnp.all(solution.converged))
    assert bool(jnp.all(solution.physically_valid))
    assert bool(jnp.all(solution.qualified))
    assert bool(jnp.all(solution.status == int(AccretionInitialDataStatus.SUCCESS)))
    np.testing.assert_allclose(
        solution.continuity_residual,
        0.0,
        atol=3.0e-5 * plan.mass_accretion_rate,
    )
    np.testing.assert_allclose(
        solution.bernoulli_residual,
        0.0,
        atol=3.0e-5,
    )
    assert float(solution.radial_four_velocity[0]) < -plan.critical_radial_speed
    assert float(solution.radial_four_velocity[2]) > -plan.critical_radial_speed


def test_fishbone_moncrief_torus_has_constant_first_integral_and_magnetic_seed():
    plan = FishboneMoncriefTorusPlan(
        _eos(),
        1.0,
        0.5,
        6.0,
        12.0,
        0.01,
        atmosphere_density=1.0e-8,
        atmosphere_pressure=1.0e-10,
        magnetic_seed_amplitude=0.2,
        magnetic_seed_cutoff=0.2,
    )
    points = jnp.asarray(
        (
            (12.0, jnp.pi / 2.0, 0.0),
            (0.999 * plan.outer_radius, jnp.pi / 2.0, 0.4),
            (1.001 * plan.outer_radius, jnp.pi / 2.0, 0.0),
        )
    )
    data = plan.evaluate(points)
    assert bool(data.inside_torus[0])
    assert bool(data.inside_torus[1])
    assert not bool(data.inside_torus[2])
    assert bool(jnp.all(data.finite))
    assert bool(jnp.all(data.physically_valid))
    assert bool(jnp.all(data.qualified))
    assert bool(jnp.all(data.converged))
    np.testing.assert_allclose(
        data.equilibrium_residual[data.inside_torus],
        0.0,
        atol=3.0e-5,
    )
    assert float(data.vector_potential_covector[0, 2]) > 0.0
    np.testing.assert_allclose(data.vector_potential_covector[:, :2], 0.0)
    np.testing.assert_allclose(data.primitive[..., 5:8], 0.0)
    assert float(data.primitive[0, 4]) > plan.atmosphere_pressure
