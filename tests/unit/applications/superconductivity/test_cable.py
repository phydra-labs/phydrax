#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _manifest():
    return phx.qualification.ReferenceArtifactManifest(
        "superconducting-material",
        checksum_algorithm="sha256",
        checksum="9" * 64,
        size_bytes=1,
        license_id="synthetic-permissive",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"identity": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=("synthetic:superconducting-material",),
    )


def _plan():
    material = phx.equations.SuperconductingMaterialLawPlan(
        jnp.asarray((2.0, 10.0, 19.0)),
        jnp.asarray((0.0, 5.0)),
        jnp.asarray((0.0, 0.5 * jnp.pi)),
        1.0e8 * jnp.ones((3, 2, 2)),
        _manifest(),
        critical_temperature=20.0,
        stabilizer_resistivity=1.0e-4,
        criterion_electric_field=1.0e-4,
        power_law_exponent=20.0,
    )
    return phx.applications.superconductivity.SuperconductingCablePlan(
        material,
        jnp.asarray((1.0, 1.0, 1.0)),
        jnp.asarray((1.0, 1.0, 1.0)),
        jnp.zeros((3,)),
        jnp.full((3,), 1000.0),
        jnp.full((3,), 1000.0),
        superconductor_area=1.0e-6,
        stabilizer_area=1.0e-6,
        inductance=1.0,
        axial_thermal_conductance=1.0,
        coolant_heat_transfer_per_length=0.0,
        coolant_mass_flow_heat_capacity=0.0,
        coolant_inlet_temperature=4.0,
        dump_resistance=1.0,
        protection_trigger_temperature=4.0001,
        tolerance=1.0e-7,
    )


def test_cable_current_sharing_joule_heat_and_energy_ledgers_close():
    plan = _plan()
    state = plan.initialize(150.0, jnp.full((3,), 4.0), jnp.full((3,), 4.0))
    result = plan.advance(state, 1.0e-6, 0.0)

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.candidate.superconducting_current + result.candidate.stabilizer_current,
        result.candidate.current,
        rtol=1.0e-10,
    )
    assert jnp.all(result.candidate.stabilizer_current > 0.0)
    assert jnp.all(result.joule_power > 0.0)
    assert jnp.max(result.candidate.solid_temperature) > 4.0
    assert bool(result.candidate.protection_active)
    np.testing.assert_allclose(result.evidence.electrical_energy_residual, 0.0, atol=1e-7)
    np.testing.assert_allclose(result.evidence.thermal_energy_residual, 0.0, atol=1e-7)


def test_invalid_cable_step_rolls_back_circuit_and_thermal_state():
    plan = _plan()
    state = plan.initialize(50.0, jnp.full((3,), 4.0), jnp.full((3,), 4.0))
    result = plan.advance(state, -1.0e-3, 0.0)

    assert not bool(result.successful)
    np.testing.assert_array_equal(result.accepted.current, state.current)
    np.testing.assert_array_equal(
        result.accepted.solid_temperature, state.solid_temperature
    )
    np.testing.assert_array_equal(result.accepted.time, state.time)
