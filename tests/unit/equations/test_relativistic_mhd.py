#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.equations._relativistic_eos import GammaLawEOS
from phydrax.equations._relativistic_mhd import (
    IdealValenciaGRMHDSystem,
    ValenciaRecoveryStatus,
)
from phydrax.metrix._adm_exchange import ADMGridGeometry
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.units import KILOGRAM, METER, SECOND


def _scale():
    return RelativityScaleContract(
        DimensionalScaleContract(METER, KILOGRAM, SECOND),
        1,
        1,
        1,
        1,
    )


def _geometry(shape, scale, convention):
    identity = jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    return ADMGridGeometry(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        identity,
        identity,
        jnp.ones(shape),
        jnp.zeros(shape + (3, 3)),
        jnp.ones(shape, dtype="bool"),
        jnp.ones(shape, dtype="bool"),
        chart_id="minkowski-cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id="pointwise",
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        geometry_lineage_id="minkowski-pointwise",
    )


def _system(*, maximum_magnetization=1.0e6):
    scale = _scale()
    convention = RelativityConvention.canonical()
    eos = GammaLawEOS(scale, 4.0 / 3.0, minimum_density=1.0e-15)
    system = IdealValenciaGRMHDSystem(
        eos,
        scale,
        convention=convention,
        pressure_ceiling=1.0e6,
        maximum_magnetization=maximum_magnetization,
        recovery_iterations=32,
        enthalpy_iterations=32,
    )
    return system, _geometry((), scale, convention)


def test_minkowski_zero_field_valencia_limit_and_bounded_recovery():
    system, geometry = _system()
    primitive = jnp.asarray((1.2, 0.25, -0.1, 0.05, 0.18, 0.0, 0.0, 0.0))
    conserved = system.primitive_to_conserved(primitive, geometry)
    gamma = system.eos.adiabatic_index
    lorentz = 1.0 / jnp.sqrt(1.0 - jnp.sum(primitive[1:4] ** 2))
    enthalpy = 1.0 + gamma * primitive[4] / ((gamma - 1.0) * primitive[0])
    expected = jnp.concatenate(
        (
            jnp.asarray((primitive[0] * lorentz,)),
            primitive[0] * enthalpy * lorentz**2 * primitive[1:4],
            jnp.asarray(
                (
                    primitive[0] * enthalpy * lorentz**2
                    - primitive[4]
                    - primitive[0] * lorentz,
                )
            ),
            jnp.zeros((3,)),
        )
    )
    np.testing.assert_allclose(conserved, expected, rtol=2.0e-6, atol=2.0e-6)

    recovered = system.recover(conserved, geometry)
    assert bool(recovered.finite)
    assert bool(recovered.converged)
    assert bool(recovered.physically_valid)
    assert bool(recovered.qualified)
    assert int(recovered.status) == int(ValenciaRecoveryStatus.SUCCESS)
    np.testing.assert_allclose(recovered.primitive, primitive, rtol=3.0e-5, atol=3.0e-6)

    projection = system.stress_energy(recovered.primitive, geometry, conserved=conserved)
    np.testing.assert_allclose(
        projection.energy_density,
        conserved[0] + conserved[4],
        rtol=3.0e-5,
        atol=3.0e-6,
    )
    assert float(projection.projection_defect) < 2.0e-6
    assert float(projection.conservation_defect) < 3.0e-5


def test_parallel_alfven_momentum_and_rotor_covariance_are_preserved():
    system, geometry = _system()
    primitive = jnp.asarray((0.9, 0.22, 0.0, 0.0, 0.08, 0.7, 0.0, 0.0))
    conserved = system.primitive_to_conserved(primitive, geometry)
    eos = system.eos.evaluate_pressure(primitive[0], primitive[4])
    lorentz = 1.0 / jnp.sqrt(1.0 - primitive[1] ** 2)
    expected_parallel_momentum = (
        primitive[0] * eos.specific_enthalpy * lorentz**2 * primitive[1]
    )
    np.testing.assert_allclose(
        conserved[1], expected_parallel_momentum, rtol=3.0e-6, atol=3.0e-6
    )

    rotation = jnp.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    rotated = primitive.at[1:4].set(rotation @ primitive[1:4])
    rotated = rotated.at[5:8].set(rotation @ primitive[5:8])
    first = system.stress_energy(primitive, geometry)
    second = system.stress_energy(rotated, geometry)
    np.testing.assert_allclose(second.energy_density, first.energy_density, rtol=2.0e-6)
    np.testing.assert_allclose(
        second.momentum_covector, rotation @ first.momentum_covector, rtol=3.0e-6
    )
    np.testing.assert_allclose(
        second.stress_covariant,
        rotation @ first.stress_covariant @ rotation.T,
        rtol=3.0e-6,
        atol=3.0e-6,
    )


def test_hlle_shock_bounds_and_high_magnetization_status_are_explicit():
    system, geometry = _system()
    left_primitive = jnp.asarray((1.0, 0.1, 0.0, 0.0, 1.0, 0.2, 0.3, 0.0))
    right_primitive = jnp.asarray((0.125, -0.1, 0.0, 0.0, 0.1, 0.2, -0.3, 0.0))
    left = system.primitive_to_conserved(left_primitive, geometry)
    right = system.primitive_to_conserved(right_primitive, geometry)
    flux = system.hlle_flux(left, right, geometry, 0)
    assert bool(flux.finite)
    assert bool(flux.physically_valid)
    assert float(flux.bounds.lower) < 0.0 < float(flux.bounds.upper)
    assert float(flux.bounds.left_fast_speed) < 1.0
    assert float(flux.bounds.right_fast_speed) < 1.0

    limited, limited_geometry = _system(maximum_magnetization=0.05)
    magnetized_primitive = jnp.asarray((1.0, 0.0, 0.0, 0.0, 0.01, 1.0, 0.0, 0.0))
    recovered = limited.recover(
        limited.primitive_to_conserved(magnetized_primitive, limited_geometry),
        limited_geometry,
    )
    assert bool(recovered.finite)
    assert bool(recovered.converged)
    assert bool(recovered.physically_valid)
    assert not bool(recovered.qualified)
    assert int(recovered.status) == int(ValenciaRecoveryStatus.MAGNETIZATION_LIMIT)
