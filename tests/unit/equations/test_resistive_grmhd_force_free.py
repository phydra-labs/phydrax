#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax._physical import RelativityScaleContract
from phydrax.equations._force_free import GRForceFreeSystem
from phydrax.equations._resistive_grmhd import ResistiveGRMHDOhmicClosure
from phydrax.metrix._adm_exchange import ADMGridGeometry
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.units import KILOGRAM


def _scale():
    return RelativityScaleContract.geometric(KILOGRAM)


def _geometry(scale, convention):
    dtype = jnp.float32
    identity = jnp.eye(3, dtype=dtype)
    return ADMGridGeometry(
        jnp.asarray(1.0, dtype=dtype),
        jnp.zeros(3, dtype=dtype),
        identity,
        identity,
        jnp.asarray(1.0, dtype=dtype),
        jnp.zeros((3, 3), dtype=dtype),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id="single-cell",
        geometry_lineage_id="flat-single-cell",
    )


def test_relativistic_ohm_closure_recovers_resistive_and_ideal_limits():
    scale = _scale()
    convention = RelativityConvention.canonical()
    geometry = _geometry(scale, convention)
    magnetic = jnp.asarray((0.0, 0.0, 2.0))
    velocity = jnp.asarray((0.2, 0.0, 0.0))
    charge = jnp.asarray(3.0)
    vacuum = ResistiveGRMHDOhmicClosure(scale, convention, conductivity=0.0)

    vacuum_result = jax.jit(vacuum.evaluate)(
        jnp.asarray((0.0, 0.1, 0.0)), magnetic, velocity, charge, geometry
    )
    np.testing.assert_allclose(
        vacuum_result.spatial_current, charge * velocity, rtol=1.0e-6
    )
    np.testing.assert_allclose(vacuum_result.conduction_current, 0.0)
    np.testing.assert_allclose(vacuum_result.entropy_production, 0.0)

    conducting = ResistiveGRMHDOhmicClosure(scale, convention, conductivity=1.0e4)
    ideal_electric = conducting.ideal_electric_field(magnetic, velocity, geometry)
    ideal = conducting.evaluate(ideal_electric, magnetic, velocity, charge, geometry)
    np.testing.assert_allclose(ideal.conduction_current, 0.0, atol=1.0e-5)
    assert bool(ideal.ideal_limit)
    assert bool(ideal.qualified)

    resistive = conducting.evaluate(
        ideal_electric + jnp.asarray((0.0, 0.05, 0.0)),
        magnetic,
        velocity,
        charge,
        geometry,
    )
    assert float(resistive.entropy_production) > 0.0
    assert not bool(resistive.ideal_limit)


def test_force_free_projection_enforces_degeneracy_and_magnetic_dominance():
    scale = _scale()
    convention = RelativityConvention.canonical()
    geometry = _geometry(scale, convention)
    system = GRForceFreeSystem(scale, convention, dominance_margin=1.0e-4)
    magnetic = jnp.asarray((0.0, 0.0, 2.0))
    electric = jnp.asarray((3.0, 0.0, 1.0))

    before = system.constraint_evaluation(electric, magnetic, geometry)
    projected = jax.jit(system.project_constraints)(electric, magnetic, geometry)

    assert not bool(before.physically_valid)
    np.testing.assert_allclose(
        projected.constraints.degeneracy_residual, 0.0, atol=1.0e-6
    )
    assert float(projected.constraints.magnetic_dominance) > 0.0
    assert bool(projected.qualified)
    assert not bool(projected.derivative_valid)


def test_force_free_current_closure_preserves_parallel_constraint_current():
    scale = _scale()
    convention = RelativityConvention.canonical()
    geometry = _geometry(scale, convention)
    system = GRForceFreeSystem(scale, convention)
    electric = jnp.zeros(3)
    magnetic = jnp.asarray((0.0, 0.0, 2.0))
    electric_gradient = jnp.zeros((3, 3))
    magnetic_gradient = jnp.zeros((3, 3)).at[0, 1].set(1.0)

    result = system.constraint_current(
        electric,
        magnetic,
        electric_gradient,
        magnetic_gradient,
        geometry,
    )

    np.testing.assert_allclose(result.charge_density, 0.0)
    np.testing.assert_allclose(result.current, jnp.asarray((0.0, 0.0, 1.0)))
    state = jnp.concatenate((electric, magnetic, jnp.zeros(2)))
    source = system.coordinate_source(state, result, geometry)
    np.testing.assert_allclose(source[:3], -result.current)
    np.testing.assert_allclose(source[3:], 0.0)
    assert bool(result.qualified)


def test_force_free_coordinate_characteristics_remain_on_adm_light_cone():
    scale = _scale()
    convention = RelativityConvention.canonical()
    geometry = _geometry(scale, convention)
    system = GRForceFreeSystem(scale, convention)

    lower, upper = system.coordinate_characteristic_bounds(
        jnp.asarray((0.0, 1.0, 0.0)), geometry
    )
    state = jnp.asarray((0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.1, -0.2))
    np.testing.assert_allclose(
        system.coordinate_flux(state, 0, geometry),
        system.local_physical_flux(state, 0),
    )

    np.testing.assert_allclose(lower, -1.0)
    np.testing.assert_allclose(upper, 1.0)
