#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.atmosphere._column import (
    conservative_radiation,
    conservative_vertical_mixing,
    MoistColumnPlan,
    precipitate,
)
from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan


def test_dry_limit_loading_and_both_caloric_constraints():
    thermo = MoistThermodynamicPlan()
    temperature = jnp.asarray((240.0, thermo.reference_temperature, 290.0, 330.0))
    density = jnp.asarray((0.8, 1.0, 1.1, 1.4))
    energy = thermo.dry_cv * (temperature - thermo.reference_temperature)
    result = thermo.adjust(density, 0.0, energy)
    assert jnp.all(result.successful)
    np.testing.assert_allclose(result.temperature, temperature, atol=2e-8)
    np.testing.assert_allclose(result.vapor + result.liquid + result.ice, 0.0)
    np.testing.assert_allclose(
        result.pressure, density * thermo.dry_gas_constant * temperature
    )
    h = energy + thermo.dry_gas_constant * temperature
    isobaric = thermo.adjust_isobaric(result.pressure, 0.0, h)
    assert jnp.all(isobaric.successful)
    np.testing.assert_allclose(isobaric.temperature, temperature, atol=2e-8)
    np.testing.assert_allclose(isobaric.density, density)
    dry_derivative = jax.grad(lambda e: thermo.adjust(1.0, 0.0, e).temperature)(0.0)
    np.testing.assert_allclose(dry_derivative, 1.0 / thermo.dry_cv, rtol=1e-12)
    loaded_pressure = thermo.pressure(1.0, 280.0, 0.01, 0.2, 0.1)
    np.testing.assert_allclose(
        loaded_pressure,
        ((1 - 0.31) * thermo.dry_gas_constant + 0.01 * thermo.vapor_gas_constant) * 280,
    )


@pytest.mark.parametrize("phase,temperature", [("liquid", 290.0), ("ice", 250.0)])
def test_saturation_uses_declared_latent_enthalpy(phase, temperature):
    thermo = MoistThermodynamicPlan(latent_vaporization=2.6e6, vapor_cv=1450.0)
    slope = jax.grad(lambda t: jnp.log(thermo.saturation_pressure(t, phase=phase)))(
        temperature
    )
    _, hv, hl, hi = thermo.phase_enthalpies(temperature)
    latent = hv - (hl if phase == "liquid" else hi)
    np.testing.assert_allclose(
        slope, latent / (thermo.vapor_gas_constant * temperature**2), rtol=2e-12
    )


def test_closed_adjustment_conserves_phase_water_energy_and_saturation():
    thermo = MoistThermodynamicPlan()
    rho = jnp.asarray((0.9, 1.2, 0.7))
    temperature = jnp.asarray((248.0, 288.0, 315.0))
    qt = jnp.asarray((0.02, 0.025, 0.001))
    equilibrium = thermo.equilibrium(rho, temperature, qt)
    energy = thermo.energy(
        rho, temperature, equilibrium.vapor, equilibrium.liquid, equilibrium.ice
    )
    result = jax.jit(thermo.adjust)(rho, qt, energy)
    assert jnp.all(result.successful)
    np.testing.assert_allclose(result.temperature, temperature, atol=2e-8)
    np.testing.assert_allclose(result.vapor + result.liquid + result.ice, qt, atol=1e-15)
    np.testing.assert_allclose(
        thermo.energy(rho, result.temperature, result.vapor, result.liquid, result.ice),
        energy,
        atol=2e-6,
    )
    vapor_pressure = rho * result.vapor * thermo.vapor_gas_constant * result.temperature
    np.testing.assert_allclose(
        vapor_pressure[0], thermo.saturation_pressure(temperature[0], phase="ice")
    )
    np.testing.assert_allclose(
        vapor_pressure[1], thermo.saturation_pressure(temperature[1])
    )
    assert result.ice[0] > 0 and result.liquid[0] == 0
    assert result.liquid[1] > 0 and result.ice[1] == 0
    assert result.vapor[2] == qt[2] and result.liquid[2] + result.ice[2] == 0
    pressure = result.pressure
    enthalpy = thermo.enthalpy(temperature, result.vapor, result.liquid, result.ice)
    isobaric = thermo.adjust_isobaric(pressure, qt, enthalpy)
    assert jnp.all(isobaric.successful)
    np.testing.assert_allclose(isobaric.temperature, temperature, atol=2e-8)
    np.testing.assert_allclose(isobaric.density, rho, atol=2e-10)


def test_supersaturated_vapor_condenses_and_warms_without_external_energy():
    thermo = MoistThermodynamicPlan()
    rho, temperature, qt = 1.1, 270.0, 0.025
    energy = thermo.energy(rho, temperature, qt, 0.0, 0.0)
    adjusted = thermo.adjust(rho, qt, energy)
    assert adjusted.successful
    assert adjusted.temperature > temperature and adjusted.liquid > 0
    np.testing.assert_allclose(
        adjusted.vapor + adjusted.liquid + adjusted.ice, qt, atol=1e-15
    )
    np.testing.assert_allclose(
        thermo.energy(
            rho, adjusted.temperature, adjusted.vapor, adjusted.liquid, adjusted.ice
        ),
        energy,
        atol=2e-6,
    )
    warmed = thermo.adjust(rho, qt, energy + 10000.0)
    assert warmed.successful and warmed.temperature > adjusted.temperature
    returned = thermo.adjust(
        rho,
        qt,
        thermo.energy(rho, warmed.temperature, warmed.vapor, warmed.liquid, warmed.ice)
        - 10000.0,
    )
    assert returned.successful
    np.testing.assert_allclose(returned.temperature, adjusted.temperature, atol=2e-8)


def test_freezing_coexistence_absorbs_latent_energy_without_temperature_jump():
    thermo = MoistThermodynamicPlan()
    rho, qt, t = 1.1, 0.025, thermo.reference_temperature
    at_freezing = thermo.equilibrium(rho, t, qt)
    qv = at_freezing.vapor
    condensate = qt - qv
    frozen_energy = thermo.energy(rho, t, qv, 0.0, condensate)
    energy = frozen_energy + 0.35 * condensate * thermo.latent_fusion
    mixed = thermo.adjust(rho, qt, energy)
    assert mixed.successful and not mixed.derivative_valid
    np.testing.assert_allclose(mixed.temperature, t, atol=1e-12)
    np.testing.assert_allclose(mixed.liquid, 0.35 * condensate, atol=1e-14)
    np.testing.assert_allclose(mixed.ice, 0.65 * condensate, atol=1e-14)
    np.testing.assert_allclose(
        thermo.energy(rho, mixed.temperature, mixed.vapor, mixed.liquid, mixed.ice),
        energy,
        atol=1e-9,
    )
    below = thermo.adjust(rho, qt, frozen_energy - 20.0)
    above = thermo.adjust(
        rho, qt, frozen_energy + condensate * thermo.latent_fusion + 20.0
    )
    assert below.successful and above.successful
    assert below.temperature < t < above.temperature
    assert below.liquid == 0.0 and above.ice == 0.0


@pytest.mark.parametrize(
    "temperature,total_water", [(250.0, 0.018), (292.0, 0.028), (305.0, 0.002)]
)
def test_branch_regular_implicit_derivatives_match_centered_perturbations(
    temperature, total_water
):
    thermo = MoistThermodynamicPlan()
    rho = 1.0
    initial = thermo.equilibrium(rho, temperature, total_water)
    energy = thermo.energy(rho, temperature, initial.vapor, initial.liquid, initial.ice)
    args = jnp.asarray((rho, total_water, energy))

    def observable(parameters):
        result = thermo.adjust(parameters[0], parameters[1], parameters[2])
        return jnp.stack((result.temperature, result.vapor))

    assert thermo.adjust(*args).derivative_valid
    derivative = jax.jacrev(observable)(args)
    increments = np.asarray((1e-5, 1e-7, 0.1))
    finite_difference = np.column_stack(
        [
            np.asarray(
                (
                    observable(args.at[i].add(increment))
                    - observable(args.at[i].add(-increment))
                )
                / (2 * increment)
            )
            for i, increment in enumerate(increments)
        ]
    )
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-5, atol=2e-7)


def test_out_of_domain_energy_cannot_be_certified():
    thermo = MoistThermodynamicPlan()
    result = thermo.adjust(1.0, 0.02, 1e8)
    assert not result.successful and not result.derivative_valid
    assert not thermo.adjust(1.0, -0.01, 0.0).successful


def test_precipitation_transfers_mass_and_condensate_enthalpy_and_rejects_atomically():
    thermo = MoistThermodynamicPlan()
    rho, temperature, qt, volume = 1.1, 255.0, 0.03, 120.0
    initial = thermo.equilibrium(rho, temperature, qt)
    energy = thermo.energy(rho, temperature, initial.vapor, initial.liquid, initial.ice)
    result = precipitate(thermo, rho, qt, energy, volume, fraction=0.4)
    assert result.successful and result.precipitated_water > 0
    mass, final_mass = rho * volume, result.density * volume
    np.testing.assert_allclose(
        final_mass * result.total_water + result.precipitated_water, mass * qt, atol=1e-12
    )
    np.testing.assert_allclose(
        final_mass * result.specific_internal_energy + result.precipitated_energy,
        mass * energy,
        atol=2e-9,
    )
    _, _, hl, hi = thermo.phase_enthalpies(temperature)
    expected_enthalpy = 0.4 * mass * (initial.liquid * hl + initial.ice * hi)
    np.testing.assert_allclose(result.precipitated_energy, expected_enthalpy, atol=2e-6)
    rejected = precipitate(thermo, rho, qt, energy, volume, fraction=1.1)
    assert not rejected.successful
    assert rejected.density == rho and rejected.total_water == qt
    assert rejected.specific_internal_energy == energy
    assert rejected.precipitated_water == 0 and rejected.precipitated_energy == 0


def test_radiation_and_vertical_mixing_have_real_opposite_budgets():
    mass = jnp.asarray((20.0, 40.0, 60.0))
    quantity = jnp.asarray((2.0, 5.0, 1.0))
    tendency = conservative_vertical_mixing(quantity, mass, 0.01)
    np.testing.assert_allclose(jnp.sum(tendency), 0.0, atol=1e-15)
    assert tendency[1] < 0 and tendency[0] > 0 and tendency[2] > 0
    radiation, reservoir = conservative_radiation(
        jnp.asarray((250.0, 280.0, 300.0)), mass, 1000.0, 260.0, 5000.0
    )
    np.testing.assert_allclose(jnp.sum(radiation) + reservoir, 0.0, atol=1e-15)
    assert radiation[0] > 0 and radiation[2] < 0


def test_isobaric_unsaturated_derivatives_remain_finite_above_saturation_pressure():
    thermo = MoistThermodynamicPlan()
    temperature, qt, pressure = 350.0, 0.02, 1000.0
    enthalpy = thermo.enthalpy(temperature, qt, 0.0, 0.0)
    result = thermo.adjust_isobaric(pressure, qt, enthalpy)
    assert result.successful and result.derivative_valid
    np.testing.assert_allclose(result.temperature, temperature, atol=2e-8)
    np.testing.assert_allclose(result.vapor, qt, atol=1e-15)
    derivative = jax.grad(lambda h: thermo.adjust_isobaric(pressure, qt, h).temperature)(
        enthalpy
    )
    np.testing.assert_allclose(
        derivative,
        1.0 / thermo.heat_capacity(qt, 0.0, 0.0, at_constant_pressure=True),
        rtol=1e-12,
    )


def test_water_only_mixing_carries_vapor_energy_without_splitting_equal_temperatures():
    plan = MoistColumnPlan(mixing_rate=0.01, radiation_temperature=300.0)
    initial = plan.initialize(
        jnp.ones(2), jnp.full(2, 300.0), jnp.asarray((0.002, 0.010)), 100.0
    )
    result = plan.step(initial, 1.0)
    assert result.successful
    diagnosed = plan.diagnose(result.state)
    np.testing.assert_allclose(diagnosed.temperature, 300.0, atol=2e-8)
    np.testing.assert_array_equal(result.state.dry_mass, initial.dry_mass)
    water_gain = result.state.water_mass[0] - initial.water_mass[0]
    assert water_gain > 0 and result.state.water_mass[1] < initial.water_mass[1]
    vapor_energy = plan.thermodynamics.phase_energies(300.0)[1]
    np.testing.assert_allclose(
        result.state.internal_energy[0] - initial.internal_energy[0],
        water_gain * vapor_energy,
        atol=2e-6,
    )
    np.testing.assert_allclose(result.state.total_water, initial.total_water, atol=1e-13)
    np.testing.assert_allclose(result.state.total_energy, initial.total_energy, atol=2e-8)


def test_column_restart_cadence_closed_budgets_and_rejection_no_commit(tmp_path):
    plan = MoistColumnPlan(forcing_cadence=3, mixing_rate=1e-4)
    initial = plan.initialize(
        jnp.asarray((0.8, 1.0, 1.2)),
        jnp.asarray((250.0, 270.0, 290.0)),
        0.015,
        100.0,
        reservoir_water=20.0,
        reservoir_energy=5e7,
    )
    first = plan.step(
        initial, 5.0, heating_rate=3.0, surface_vapor_flux=1e-5, surface_energy_flux=2.0
    )
    assert first.successful and first.precipitated_water > 0
    np.testing.assert_allclose(first.state.total_water, initial.total_water, atol=2e-12)
    np.testing.assert_allclose(
        first.state.total_energy - initial.total_energy, 45.0, atol=2e-7
    )
    checkpoint = plan.save_checkpoint(tmp_path / "column.phx", first.state)
    restart = plan.load_checkpoint(checkpoint)
    with pytest.raises(ValueError):
        MoistColumnPlan(forcing_cadence=4, mixing_rate=1e-4).load_checkpoint(checkpoint)
    uninterrupted = plan.step(
        first.state,
        5.0,
        heating_rate=3.0,
        surface_vapor_flux=1e-5,
        surface_energy_flux=2.0,
    )
    resumed = plan.step(
        restart,
        5.0,
        heating_rate=1000.0,
        surface_vapor_flux=100.0,
        surface_energy_flux=1000.0,
    )
    assert uninterrupted.successful and resumed.successful
    for expected, actual in zip(
        jax.tree.leaves(uninterrupted.state), jax.tree.leaves(resumed.state)
    ):
        np.testing.assert_array_equal(actual, expected)
    # Cadence is still held here; negative dt tests rollback independent of forcing refresh.
    rejected = plan.step(resumed.state, -1.0)
    assert not rejected.successful
    for expected, actual in zip(
        jax.tree.leaves(resumed.state), jax.tree.leaves(rejected.state)
    ):
        np.testing.assert_array_equal(actual, expected)
    assert rejected.precipitated_water == 0 and rejected.precipitated_energy == 0
    # Exhaust the surface reservoir on an actual refresh, and reject all inventories.
    third = plan.step(resumed.state, 5.0)
    assert third.successful and third.state.cadence_phase == 0
    overdrawn = plan.step(third.state, 5.0, surface_vapor_flux=100.0)
    assert not overdrawn.successful
    for expected, actual in zip(
        jax.tree.leaves(third.state), jax.tree.leaves(overdrawn.state)
    ):
        np.testing.assert_array_equal(actual, expected)
