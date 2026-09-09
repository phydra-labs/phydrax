#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measured liquid-surface fluxes, closed transfers and branch-regular derivatives.

PYTHONPATH=. python tools/surface_exchange_qualification.py --steps 120 --time-step 10
Reports observations for the declared forced surface-layer closure, not a
universal turbulent-flux qualification or a momentum-dynamics experiment.
"""

import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.atmosphere import (
    BulkSurfaceExchangePlan,
    MoistThermodynamicPlan,
    paired_surface_transfer,
    WetSlabPlan,
    WetSlabState,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--time-step", type=float, default=10.0)
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.time_step) or args.time_step <= 0:
        parser.error("steps and time-step must be positive and finite")
    jax.config.update("jax_enable_x64", True)
    thermo = MoistThermodynamicPlan()
    exchange = BulkSurfaceExchangePlan()
    slab_plan = WetSlabPlan(thermo, dry_heat_capacity=5e5)
    volume, pressure, wind, height = 100.0, 1e5, 5.0, 10.0
    reports = {}

    for name, surface_t, air_t, vapor in (
        ("evaporation", 300.0, 295.0, 0.004),
        ("dew", 290.0, 300.0, 0.017),
    ):
        slab = slab_plan.initialize(surface_t, 20.0)
        density = pressure / (thermo.gas_constant(vapor, 0.0, 0.0) * air_t)
        mass = density * volume
        dry_mass, water_mass = mass * (1 - vapor), mass * vapor
        air_energy = mass * thermo.energy(density, air_t, vapor, 0.0, 0.0)
        initial_flux = exchange.evaluate(
            thermo, air_t, density, vapor, pressure, surface_t, wind, height
        )

        def rollout(initial):
            def step(carry, _):
                surface, water, energy = carry
                air_mass = dry_mass + water
                air = thermo.adjust(
                    air_mass / volume, water / air_mass, energy / air_mass
                )
                gas_fraction = 1 - air.liquid - air.ice
                rates = exchange.evaluate(
                    thermo,
                    air.temperature,
                    air.density * gas_fraction,
                    air.vapor / gas_fraction,
                    air.pressure,
                    slab_plan.temperature(surface, thermo),
                    wind,
                    height,
                )
                transfer = paired_surface_transfer(
                    thermo,
                    slab_plan,
                    surface,
                    dry_mass,
                    water,
                    energy,
                    volume,
                    water_mass=args.time_step * rates.water_mass,
                    energy=args.time_step * (rates.sensible_heat + rates.water_enthalpy),
                )
                successful = rates.successful & transfer.successful
                candidate = (
                    transfer.slab_state,
                    transfer.air_water_mass,
                    transfer.air_internal_energy,
                )
                committed = jax.tree.map(
                    lambda new, old: jnp.where(successful, new, old), candidate, carry
                )
                return committed, successful

            return jax.lax.scan(step, initial, None, length=args.steps)

        (final_slab, final_water, final_energy), successful = eqx.filter_jit(rollout)(
            (slab, water_mass, air_energy)
        )
        if not bool(jnp.all(successful)):
            raise RuntimeError(f"{name} rejected a surface transaction.")
        water_residual = float(
            final_slab.water_mass + final_water - slab.water_mass - water_mass
        )
        energy_residual = float(
            final_slab.energy + final_energy - slab.energy - air_energy
        )
        final_temperature = float(slab_plan.temperature(final_slab, thermo))
        np.testing.assert_allclose(water_residual, 0.0, atol=2e-11)
        np.testing.assert_allclose(energy_residual, 0.0, atol=2e-6)
        if (name == "evaporation" and final_temperature >= surface_t) or (
            name == "dew" and final_temperature <= surface_t
        ):
            raise RuntimeError(f"{name} produced the wrong slab thermal response.")
        reports[name] = {
            "initial_sensible_heat_W_m2": float(initial_flux.sensible_heat),
            "initial_water_mass_kg_m2_s": float(initial_flux.water_mass),
            "initial_water_enthalpy_W_m2": float(initial_flux.water_enthalpy),
            "initial_total_energy_flux_W_m2": float(
                initial_flux.sensible_heat + initial_flux.water_enthalpy
            ),
            "final_surface_temperature_K": final_temperature,
            "surface_temperature_change_K": final_temperature - surface_t,
            "closed_water_residual_kg_m2": water_residual,
            "closed_energy_residual_J_m2": energy_residual,
        }

    def observable(parameters):
        temperature, vapor, speed, water, heat_scale, moisture_scale, capacity_scale = (
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
        slab = surface_plan.initialize(temperature, water)
        rates = exchange_plan.evaluate(
            thermo, 295.0, 1.1, vapor, pressure, temperature, speed, height
        )
        candidate = WetSlabState(
            slab.water_mass - args.time_step * rates.water_mass,
            slab.energy - args.time_step * (rates.sensible_heat + rates.water_enthalpy),
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
    increments = (1e-3, 1e-6, 1e-4, 1e-2, 1e-4, 1e-4, 1e-4)
    derivative = jax.jacrev(observable)(parameters)
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
    relative_error = float(
        jnp.max(
            jnp.abs(derivative - finite_difference)
            / jnp.maximum(jnp.abs(finite_difference), 1.0)
        )
    )
    saturated = thermo.saturation_pressure(300.0)
    epsilon = thermo.dry_gas_constant / thermo.vapor_gas_constant
    saturated_vapor = epsilon * saturated / (pressure - (1 - epsilon) * saturated)
    balanced = exchange.evaluate(
        thermo, 300.0, 1.1, saturated_vapor, pressure, 300.0, wind, height
    )
    calm = exchange.evaluate(thermo, 295.0, 1.1, 0.005, pressure, 300.0, 0.0, height)
    for rates in (balanced, calm):
        if not bool(rates.successful):
            raise RuntimeError("Analytic limiting state rejected.")
        np.testing.assert_array_equal(
            jnp.stack((rates.sensible_heat, rates.water_mass, rates.water_enthalpy)),
            jnp.zeros(3),
        )
    if bool(calm.derivative_valid):
        raise RuntimeError("Nonregular zero-wind derivative incorrectly certified.")
    finite_slab = slab_plan.initialize(300.0, 0.001)
    rejected = paired_surface_transfer(
        thermo,
        slab_plan,
        finite_slab,
        dry_mass,
        water_mass,
        air_energy,
        volume,
        water_mass=0.002,
        energy=0.002 * thermo.phase_enthalpies(300.0)[1],
    )
    if bool(rejected.successful):
        raise RuntimeError("Exhausted surface proposal was committed.")
    np.testing.assert_array_equal(rejected.slab_state.water_mass, finite_slab.water_mass)
    np.testing.assert_array_equal(rejected.slab_state.energy, finite_slab.energy)
    np.testing.assert_array_equal(rejected.air_water_mass, water_mass)
    np.testing.assert_array_equal(rejected.air_internal_energy, air_energy)
    np.testing.assert_array_equal(
        jnp.stack((rejected.water_mass, rejected.energy)), jnp.zeros(2)
    )

    print(
        json.dumps(
            {
                "steps": args.steps,
                "time_step_s": args.time_step,
                "exchange_plan_id": exchange.plan_id,
                "slab_plan_id": slab_plan.plan_id,
                "stability_law": exchange.stability,
                "heat_transfer_coefficient": float(exchange.heat_transfer_coefficient),
                "moisture_transfer_coefficient": float(
                    exchange.moisture_transfer_coefficient
                ),
                "dry_heat_capacity_J_m2_K": float(slab_plan.dry_heat_capacity),
                "scenarios": reports,
                "no_gradient_and_no_wind_fluxes_exactly_zero": True,
                "exhaustion_atomically_rejected": True,
                "calm_derivative_valid": bool(calm.derivative_valid),
                "gradient_inputs": [
                    "surface_temperature_K",
                    "specific_humidity",
                    "wind_speed_m_s",
                    "surface_water_kg_m2",
                    "log_heat_coefficient_scale",
                    "log_moisture_coefficient_scale",
                    "log_dry_heat_capacity_scale",
                ],
                "gradient_outputs": [
                    "sensible_heat_W_m2",
                    "water_mass_kg_m2_s",
                    "water_enthalpy_W_m2",
                    "candidate_surface_temperature_K",
                ],
                "automatic_derivative": np.asarray(derivative).tolist(),
                "centered_finite_difference": np.asarray(finite_difference).tolist(),
                "gradient_scaled_max_error": relative_error,
                "momentum_dynamics": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
