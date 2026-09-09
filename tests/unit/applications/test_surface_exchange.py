#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan
from phydrax.applications.atmosphere._surface import (
    BulkSurfaceExchangePlan,
    paired_surface_transfer,
    WetSlabPlan,
    WetSlabState,
)


def _saturation_humidity(thermo, temperature, pressure=1e5):
    es = thermo.saturation_pressure(temperature)
    epsilon = thermo.dry_gas_constant / thermo.vapor_gas_constant
    return epsilon * es / (pressure - (1 - epsilon) * es)


def _air(thermo, temperature, vapor, volume=100.0):
    density = 1e5 / (thermo.gas_constant(vapor, 0.0, 0.0) * temperature)
    mass = density * volume
    energy = mass * thermo.energy(density, temperature, vapor, 0.0, 0.0)
    return density, mass * (1 - vapor), mass * vapor, energy


def test_neutral_bulk_analytic_fluxes_no_gradient_and_no_wind():
    thermo = MoistThermodynamicPlan()
    plan = BulkSurfaceExchangePlan(
        heat_transfer_coefficient=1.5e-3,
        moisture_transfer_coefficient=9e-4,
        stability="neutral",
    )
    surface_t, air_t, density, vapor, wind = 300.0, 295.0, 1.2, 0.005, 6.0
    result = plan.evaluate(thermo, air_t, density, vapor, 1e5, surface_t, wind, 10.0)
    cp = thermo.heat_capacity(vapor, 0.0, 0.0, at_constant_pressure=True)
    assert result.successful
    np.testing.assert_allclose(result.sensible_heat, density * cp * 1.5e-3 * wind * 5.0)
    np.testing.assert_allclose(
        result.water_mass,
        density * 9e-4 * wind * (_saturation_humidity(thermo, surface_t) - vapor),
    )
    for exchange in (plan, BulkSurfaceExchangePlan()):
        quiet = exchange.evaluate(
            thermo, air_t, density, vapor, 1e5, surface_t, 0.0, 10.0
        )
        balanced = exchange.evaluate(
            thermo,
            surface_t,
            density,
            _saturation_humidity(thermo, surface_t),
            1e5,
            surface_t,
            wind,
            10.0,
        )
        assert quiet.successful and balanced.successful
        for rates in (quiet, balanced):
            np.testing.assert_array_equal(
                jnp.stack((rates.sensible_heat, rates.water_mass, rates.water_enthalpy)),
                jnp.zeros(3),
            )


def test_stability_reduces_stable_and_enhances_unstable_ventilation():
    thermo = MoistThermodynamicPlan()
    neutral = BulkSurfaceExchangePlan(stability="neutral")
    stability = BulkSurfaceExchangePlan()
    for air_t, vapor, enhanced in ((305.0, 0.022, False), (295.0, 0.005, True)):
        arguments = (thermo, air_t, 1.1, vapor, 1e5, 300.0, 5.0, 10.0)
        base, adjusted = neutral.evaluate(*arguments), stability.evaluate(*arguments)
        assert base.successful and adjusted.successful
        ratio = adjusted.sensible_heat / base.sensible_heat
        assert ratio > 1 if enhanced else 0 < ratio < 1
        np.testing.assert_allclose(adjusted.water_mass / base.water_mass, ratio)


@pytest.mark.parametrize(
    "surface_t,air_t,vapor", [(300.0, 295.0, 0.004), (290.0, 300.0, 0.017)]
)
def test_evaporation_cools_dew_warms_with_exact_donor_energy(surface_t, air_t, vapor):
    thermo = MoistThermodynamicPlan(
        latent_vaporization=2.6e6, liquid_heat_capacity=4200.0
    )
    slab_plan = WetSlabPlan(thermo, dry_heat_capacity=5e5)
    slab = slab_plan.initialize(surface_t, 20.0)
    exchange = BulkSurfaceExchangePlan(heat_transfer_coefficient=0.0, stability="neutral")
    density, dry, water, energy = _air(thermo, air_t, vapor)
    rates = exchange.evaluate(thermo, air_t, density, vapor, 1e5, surface_t, 5.0, 10.0)
    evaporation = surface_t > air_t
    assert rates.successful
    assert rates.water_mass > 0 if evaporation else rates.water_mass < 0
    donor_t = surface_t if evaporation else air_t
    hv = thermo.phase_enthalpies(donor_t)[1]
    np.testing.assert_allclose(rates.water_enthalpy, rates.water_mass * hv)
    result = paired_surface_transfer(
        thermo,
        slab_plan,
        slab,
        dry,
        water,
        energy,
        100.0,
        water_mass=100.0 * rates.water_mass,
        energy=100.0 * (rates.sensible_heat + rates.water_enthalpy),
    )
    assert result.successful
    final_t = slab_plan.temperature(result.slab_state, thermo)
    assert final_t < surface_t if evaporation else final_t > surface_t
    # Phase-change cooling/warming follows the varying inventory caloric law,
    # rather than adding a second latent heat to the advected vapor enthalpy.
    hl = thermo.phase_enthalpies(surface_t)[2]
    final_capacity = (
        slab_plan.dry_heat_capacity
        + result.slab_state.water_mass * thermo.liquid_heat_capacity
    )
    np.testing.assert_allclose(
        final_t - surface_t,
        -result.water_mass * (hv - hl) / final_capacity,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.air_water_mass + result.slab_state.water_mass,
        water + slab.water_mass,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.air_internal_energy + result.slab_state.energy,
        energy + slab.energy,
        atol=1e-8,
    )


def test_water_inventory_changes_heat_capacity_without_duplicate_temperature():
    thermo = MoistThermodynamicPlan()
    plan = WetSlabPlan(thermo, dry_heat_capacity=1e5)
    state = plan.initialize(300.0, jnp.asarray((0.0, 100.0)))
    heated = WetSlabState(state.water_mass, state.energy + 1000.0)
    np.testing.assert_allclose(
        plan.temperature(heated, thermo) - 300.0,
        1000.0 / (1e5 + state.water_mass * thermo.liquid_heat_capacity),
        atol=1e-12,
    )
    assert jnp.all(plan.admissible(heated, thermo))
    with pytest.raises(ValueError, match="reference"):
        plan.temperature(state, MoistThermodynamicPlan(reference_temperature=274.0))
    with pytest.raises(ValueError, match="liquid"):
        WetSlabPlan(thermo, minimum_temperature=250.0)
    assert not plan.admissible(WetSlabState(1.0, -1.0), thermo)


def test_batched_exhaustion_rejects_both_inventories_without_capping():
    thermo = MoistThermodynamicPlan()
    plan = WetSlabPlan(thermo)
    slab = plan.initialize(300.0, jnp.asarray((1.0, 0.001)))
    _, dry, water, energy = _air(thermo, 295.0, 0.005)
    integrated_water = 0.002
    integrated_energy = integrated_water * thermo.phase_enthalpies(300.0)[1] + 20.0
    result = eqx.filter_jit(paired_surface_transfer)(
        thermo,
        plan,
        slab,
        dry,
        water,
        energy,
        100.0,
        water_mass=integrated_water,
        energy=integrated_energy,
    )
    np.testing.assert_array_equal(result.successful, [True, False])
    assert result.water_mass[0] == integrated_water and result.water_mass[1] == 0
    assert result.energy[0] == integrated_energy and result.energy[1] == 0
    assert result.slab_state.water_mass[1] == slab.water_mass[1]
    assert result.slab_state.energy[1] == slab.energy[1]
    assert result.air_water_mass[1] == water
    assert result.air_internal_energy[1] == energy
    np.testing.assert_allclose(
        result.air_water_mass + result.slab_state.water_mass,
        water + slab.water_mass,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.air_internal_energy + result.slab_state.energy,
        energy + slab.energy,
        atol=1e-8,
    )


def test_dew_cannot_remove_unavailable_vapor_and_freezing_rejects():
    thermo = MoistThermodynamicPlan()
    plan = WetSlabPlan(thermo)
    slab = plan.initialize(thermo.reference_temperature, 1.0)
    _, dry, water, energy = _air(thermo, 290.0, 0.002)
    for transfer_water, transfer_energy in (
        (-2 * water, -2 * water * thermo.phase_enthalpies(290.0)[1]),
        (0.0, 1.0),
    ):
        result = paired_surface_transfer(
            thermo,
            plan,
            slab,
            dry,
            water,
            energy,
            100.0,
            water_mass=transfer_water,
            energy=transfer_energy,
        )
        assert not result.successful
        assert result.water_mass == 0 and result.energy == 0
        assert result.air_internal_energy == energy
        assert result.air_water_mass == water
        assert result.slab_state.energy == slab.energy
        assert result.slab_state.water_mass == slab.water_mass


def test_bulk_rejects_boiling_and_nonphysical_ventilation():
    thermo = MoistThermodynamicPlan()
    plan = BulkSurfaceExchangePlan()
    result = plan.evaluate(
        thermo,
        290.0,
        1.0,
        0.005,
        jnp.asarray((1000.0, 1e5, 1e5)),
        300.0,
        jnp.asarray((5.0, -1.0, 5.0)),
        jnp.asarray((10.0, 10.0, 0.0)),
    )
    assert not jnp.any(result.successful)


def test_flux_and_slab_response_derivatives_match_finite_differences():
    thermo = MoistThermodynamicPlan()
    exchange = BulkSurfaceExchangePlan()
    slab_plan = WetSlabPlan(thermo)

    def observable(parameters):
        surface_t, vapor, speed, inventory, heat_scale, moisture_scale, capacity_scale = (
            parameters
        )
        surface_plan = eqx.tree_at(
            lambda p: p.dry_heat_capacity,
            slab_plan,
            slab_plan.dry_heat_capacity * jnp.exp(capacity_scale),
        )
        exchange_plan = eqx.tree_at(
            lambda p: (p.heat_transfer_coefficient, p.moisture_transfer_coefficient),
            exchange,
            (
                exchange.heat_transfer_coefficient * jnp.exp(heat_scale),
                exchange.moisture_transfer_coefficient * jnp.exp(moisture_scale),
            ),
        )
        slab = surface_plan.initialize(surface_t, inventory)
        rates = exchange_plan.evaluate(
            thermo, 295.0, 1.1, vapor, 1e5, surface_t, speed, 10.0
        )
        candidate = WetSlabState(
            slab.water_mass - 60 * rates.water_mass,
            slab.energy - 60 * (rates.sensible_heat + rates.water_enthalpy),
        )
        return jnp.stack(
            (
                rates.sensible_heat,
                rates.water_mass,
                rates.water_enthalpy,
                surface_plan.temperature(candidate, thermo),
            )
        )

    parameters = jnp.asarray((300.0, 0.005, 5.0, 20.0, 0.0, 0.0, 0.0))
    derivative = jax.jacrev(observable)(parameters)
    increments = (1e-3, 1e-6, 1e-4, 1e-2, 1e-4, 1e-4, 1e-4)
    finite_difference = jnp.stack(
        [
            (
                observable(parameters.at[i].add(step))
                - observable(parameters.at[i].add(-step))
            )
            / (2 * step)
            for i, step in enumerate(increments)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(derivative, finite_difference, rtol=3e-6, atol=3e-8)
    batched = jax.vmap(observable)(jnp.stack((parameters, parameters.at[0].add(1.0))))
    np.testing.assert_allclose(batched[0], observable(parameters))
    np.testing.assert_allclose(batched[1], observable(parameters.at[0].add(1.0)))


def test_nonsmooth_donor_and_calm_branches_do_not_certify_ad():
    thermo = MoistThermodynamicPlan()
    exchange = BulkSurfaceExchangePlan(stability="neutral")
    vapor = _saturation_humidity(thermo, 300.0)
    donor_switch = exchange.evaluate(thermo, 295.0, 1.1, vapor, 1e5, 300.0, 5.0, 10.0)
    equal_donors = exchange.evaluate(thermo, 300.0, 1.1, vapor, 1e5, 300.0, 5.0, 10.0)
    assert donor_switch.successful and not donor_switch.derivative_valid
    assert equal_donors.successful and equal_donors.derivative_valid
    unstable = BulkSurfaceExchangePlan()
    calm = unstable.evaluate(thermo, 295.0, 1.1, 0.005, 1e5, 300.0, 0.0, 10.0)
    assert calm.successful and not calm.derivative_valid
    assert calm.sensible_heat == 0 and calm.water_mass == 0
    invalid = eqx.tree_at(
        lambda p: p.moisture_transfer_coefficient, exchange, jnp.asarray(-0.001)
    )
    assert not invalid.evaluate(
        thermo, 295.0, 1.1, vapor, 1e5, 300.0, 5.0, 10.0
    ).successful


def test_zero_moisture_coefficient_retains_dew_donor_directional_derivative():
    thermo = MoistThermodynamicPlan()
    exchange = BulkSurfaceExchangePlan(
        moisture_transfer_coefficient=0.0, stability="neutral"
    )

    def rates(coefficient):
        plan = eqx.tree_at(
            lambda p: p.moisture_transfer_coefficient, exchange, coefficient
        )
        return plan.evaluate(thermo, 300.0, 1.1, 0.017, 1e5, 290.0, 5.0, 10.0)

    zero = rates(jnp.asarray(0.0))
    assert zero.successful and zero.derivative_valid
    assert zero.water_mass == 0 and zero.water_enthalpy == 0
    derivative = jax.grad(lambda coefficient: rates(coefficient).water_enthalpy)(0.0)
    step = 1e-7
    one_sided = (rates(jnp.asarray(step)).water_enthalpy - zero.water_enthalpy) / step
    expected = (
        1.1
        * 5.0
        * (_saturation_humidity(thermo, 290.0) - 0.017)
        * thermo.phase_enthalpies(300.0)[1]
    )
    assert derivative < 0
    np.testing.assert_allclose(derivative, expected, rtol=1e-12)
    np.testing.assert_allclose(derivative, one_sided, rtol=1e-12)
