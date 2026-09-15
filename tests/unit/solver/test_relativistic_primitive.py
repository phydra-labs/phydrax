import jax
import jax.numpy as jnp

from phydrax._physical import RelativityScaleContract
from phydrax.equations._relativistic_eos import GammaLawEOS
from phydrax.equations._relativistic_hydrodynamics import ValenciaGRHDSystem
from phydrax.metrix._adm_exchange import ADMGridGeometry
from phydrax.solver._relativistic_primitive import (
    AtmosphereFloorPolicy,
    AtmosphereFloorStatus,
    GRHDC2PPolicy,
    GRHDC2PStatus,
)
from phydrax.units import KILOGRAM


def _system():
    scale = RelativityScaleContract.geometric(KILOGRAM)
    return ValenciaGRHDSystem(GammaLawEOS(scale, 5.0 / 3.0))


def _geometry(shape, system, *, valid=True):
    identity = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), shape + (3, 3))
    return ADMGridGeometry(
        jnp.ones(shape, dtype=jnp.float64),
        jnp.zeros(shape + (3,), dtype=jnp.float64),
        identity,
        identity,
        jnp.ones(shape, dtype=jnp.float64),
        jnp.zeros(shape + (3, 3), dtype=jnp.float64),
        jnp.ones(shape, dtype=bool),
        jnp.full(shape, valid, dtype=bool),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="cartesian",
        convention_id=system.convention.convention_id,
        scale_id=system.eos.scale.scale_id,
        topology_id="fixed-grid",
        geometry_lineage_id="minkowski",
    )


def test_c2p_primary_round_trip_certifies_recomposition_and_evidence():
    system = _system()
    geometry = _geometry((4,), system)
    primitive = jnp.asarray(
        (
            (1.0, 0.2, 0.1, -0.05, 0.02),
            (0.3, 0.5, 0.8, 0.1, -0.1),
            (2.0, 0.01, 0.0, 0.0, 0.0),
            (0.02, 2.0, -0.2, 0.3, 0.1),
        ),
        dtype=jnp.float64,
    )
    conserved = system.primitive_to_conserved(primitive, geometry)
    result = GRHDC2PPolicy(system, implicit_differentiation=True).recover(
        conserved, geometry
    )

    assert bool(jnp.all(result.successful))
    assert bool(jnp.all(result.status == int(GRHDC2PStatus.PRIMARY_SUCCESS)))
    assert bool(jnp.all(result.finite))
    assert bool(jnp.all(result.converged))
    assert bool(jnp.all(result.physically_valid))
    assert bool(jnp.all(result.qualified))
    assert bool(jnp.all(result.derivative_valid))
    assert jnp.allclose(result.primitive, primitive, rtol=1.0e-8, atol=1.0e-10)
    assert jnp.allclose(result.conservative_state, conserved)
    assert result.snapshot_token == geometry.snapshot_token
    assert result.geometry_lineage_id == geometry.geometry_lineage_id
    assert result.candidates.attempted.shape == (4, 3)
    assert result.candidates.nonlinear_status.shape == (4, 3)
    assert result.candidates.iterations.shape == (4, 3)


def test_c2p_fixed_bracket_is_a_visible_secondary_candidate():
    system = _system()
    geometry = _geometry((1,), system)
    primitive = jnp.asarray(((0.4, 5.0, 0.92, 0.0, 0.0),), dtype=jnp.float64)
    conserved = system.primitive_to_conserved(primitive, geometry)
    policy = GRHDC2PPolicy(
        system,
        maximum_primary_iterations=1,
        absolute_tolerance=1.0e-13,
        relative_tolerance=0.0,
    )
    result = policy.recover(conserved, geometry, warm_pressure=jnp.asarray((1.0e-12,)))

    assert bool(result.successful[0])
    assert result.status[0] == int(GRHDC2PStatus.BRACKET_SUCCESS)
    assert bool(result.candidates.attempted[0, 1])
    assert result.candidates.selected_branch[0] == 1
    assert not bool(result.derivative_valid[0])
    assert jnp.allclose(result.primitive, primitive, rtol=1.0e-8, atol=1.0e-9)


def test_near_vacuum_atmosphere_and_hard_budget_failure_are_not_silent():
    system = _system()
    geometry = _geometry((2,), system)
    vacuum = jnp.zeros((2, 5), dtype=jnp.float64)
    permissive = GRHDC2PPolicy(
        system,
        atmosphere=AtmosphereFloorPolicy(
            rest_mass_density=1.0e-8,
            specific_internal_energy=1.0e-6,
            maximum_mass_addition=1.0,
            maximum_energy_addition=1.0,
        ),
    ).recover(vacuum, geometry)

    assert bool(jnp.all(permissive.successful))
    assert bool(jnp.all(permissive.status == int(GRHDC2PStatus.ATMOSPHERE_APPLIED)))
    assert bool(jnp.all(permissive.atmosphere.applied))
    assert bool(
        jnp.all(permissive.atmosphere.status == int(AtmosphereFloorStatus.NEAR_VACUUM))
    )
    assert permissive.atmosphere.total_positive_mass_addition > 0.0
    assert jnp.any(permissive.atmosphere.conservative_increment != 0.0)
    assert not bool(jnp.any(permissive.derivative_valid))

    strict = GRHDC2PPolicy(
        system,
        atmosphere=AtmosphereFloorPolicy(
            rest_mass_density=1.0e-8,
            specific_internal_energy=1.0e-6,
            maximum_mass_addition=0.0,
            maximum_energy_addition=0.0,
        ),
    ).recover(vacuum, geometry)
    assert not bool(jnp.any(strict.successful))
    assert bool(jnp.all(strict.status == int(GRHDC2PStatus.ATMOSPHERE_BUDGET_EXCEEDED)))
    assert bool(
        jnp.all(strict.atmosphere.status == int(AtmosphereFloorStatus.BUDGET_EXCEEDED))
    )
    assert jnp.all(strict.atmosphere.conservative_increment == 0.0)


def test_c2p_invalid_geometry_status_and_jit_fixed_shapes():
    system = _system()
    valid_geometry = _geometry((3,), system)
    invalid_geometry = _geometry((3,), system, valid=False)
    primitive = jnp.broadcast_to(jnp.asarray((1.0, 0.2, 0.1, 0.0, 0.0)), (3, 5))
    conserved = system.primitive_to_conserved(primitive, valid_geometry)
    policy = GRHDC2PPolicy(system)
    invalid = policy.recover(conserved, invalid_geometry)

    assert not bool(jnp.any(invalid.successful))
    assert bool(jnp.all(invalid.status == int(GRHDC2PStatus.INVALID_GEOMETRY)))

    compiled = jax.jit(lambda state: policy.recover(state, valid_geometry))
    result = compiled(conserved)
    assert result.primitive.shape == (3, 5)
    assert result.candidates.attempted.shape == (3, 3)
    assert result.candidates.recomposition_defect.shape == (3, 3)
