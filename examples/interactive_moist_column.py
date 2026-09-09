#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Forced radiative-convective moist column with an interactive finite wet slab.

PYTHONPATH=. python examples/interactive_moist_column.py --steps 1800 --dt 2
A longer experiment can use --spinup-steps 43200; report rejection rather than
silently changing the physical timestep. Grey coefficients below are explicit
illustrative calibration choices, not observed optical constants.
"""

import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.applications.atmosphere._interactive_column import InteractiveMoistColumnPlan
from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan
from phydrax.applications.atmosphere._radiation import (
    ColumnOpticalProperties,
    ColumnRadiationPlan,
)
from phydrax.applications.atmosphere._surface import BulkSurfaceExchangePlan, WetSlabPlan


def build_column(layers=8):
    if int(layers) != layers or layers < 2:
        raise ValueError("The forced-column example requires at least two layers.")
    thermo = MoistThermodynamicPlan()
    optics = ColumnOpticalProperties(
        shortwave_absorption=(1e-5, 0.002, 0.04, 0.03),
        shortwave_scattering=(0.0, 0.0, 60.0, 30.0),
        longwave_absorption=(1e-4, 0.08, 50.0, 25.0),
        shortwave_asymmetry=(0.0, 0.0, 0.85, 0.7),
        reference_id="illustrative-grey-forced-column-calibration",
    )
    plan = InteractiveMoistColumnPlan(
        thermo,
        radiation=ColumnRadiationPlan(optics, surface_albedo=0.08),
        surface_exchange=BulkSurfaceExchangePlan(),
        slab=WetSlabPlan(thermo, dry_heat_capacity=2e6),
        background_diffusivity=0.5,
        mixing_length=50.0,
    )
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    thickness = 4000.0 / layers
    height = thickness * (
        jnp.arange(layers - 1, -1, -1, dtype=dtype) + jnp.asarray(0.5, dtype)
    )
    temperature = jnp.asarray(295.0, dtype) - jnp.asarray(0.0065, dtype) * height
    volume = jnp.full((layers,), thickness, dtype=dtype)
    dry = (
        jnp.asarray(1.18, dtype) * jnp.exp(-height / jnp.asarray(8500.0, dtype)) * volume
    )
    saturation = jnp.where(
        temperature < thermo.reference_temperature,
        thermo.saturation_pressure(temperature, phase="ice"),
        thermo.saturation_pressure(temperature),
    )
    vapor = 0.8 * saturation * volume / (thermo.vapor_gas_constant * temperature)
    # A resolved cloud seed is advected and autoconverted, not instantaneously
    # projected to a lower reservoir. Subsequent water comes only from the slab.
    cloud = jnp.zeros_like(dry).at[layers // 2].set(0.08)
    initial = plan.initialize(
        dry,
        vapor,
        temperature,
        volume,
        cloud_liquid_mass=cloud,
        surface_temperature=jnp.asarray(298.0, dtype),
        surface_water_mass=jnp.asarray(1000.0, dtype),
    )
    return plan, initial


def physical_report(plan, initial, final, dt, forcing):
    view = plan.diagnose(final)
    flux = plan.step(final, dt, **forcing)
    if not bool(flux.successful):
        raise RuntimeError(
            f"Endpoint flux probe rejected; stable_step={float(flux.stable_step):.6g} s"
        )
    elapsed = final.time - initial.time
    water_budget = final.total_water - initial.total_water
    energy_budget = (
        final.total_energy
        - initial.total_energy
        - (final.external_energy - initial.external_energy)
    )
    return {
        "elapsed_seconds": float(elapsed),
        "accepted_steps": int(final.step_count - initial.step_count),
        "temperature_K": view.temperature.tolist(),
        "pressure_Pa": view.pressure.tolist(),
        "relative_humidity": view.relative_humidity.tolist(),
        "surface_temperature_K": float(view.surface_temperature),
        "slab_water_kg_m2": float(final.slab.water_mass),
        "slab_energy_J_m2": float(final.slab.energy),
        "cloud_liquid_kg_m2": final.cloud_liquid_mass.tolist(),
        "cloud_ice_kg_m2": final.cloud_ice_mass.tolist(),
        "rain_kg_m2": final.rain_mass.tolist(),
        "snow_kg_m2": final.snow_mass.tolist(),
        "cumulative_precipitation_kg_m2": float(
            final.precipitated_water - initial.precipitated_water
        ),
        "mean_precipitation_mm_day": float(
            (final.precipitated_water - initial.precipitated_water) * 86400 / elapsed
        ),
        "mean_evaporation_mm_day": float(
            (final.evaporated_water - initial.evaporated_water) * 86400 / elapsed
        ),
        "environment_radiant_energy_J_m2": float(
            final.environment_energy - initial.environment_energy
        ),
        "external_heat_J_m2": float(final.external_energy - initial.external_energy),
        "closed_water_residual_kg_m2": float(water_budget),
        "energy_residual_J_m2": float(energy_budget),
        "endpoint_fluxes": {
            "sensible_heat_W_m2": float(flux.sensible_heat),
            "water_enthalpy_W_m2": float(flux.surface_water_enthalpy),
            "evaporation_kg_m2_s": float(flux.surface_water_flux),
            "radiative_heating_W_m2": flux.radiative_heating.tolist(),
            "surface_radiative_heating_W_m2": float(flux.surface_radiative_heating),
            "space_net_upward_W_m2": float(flux.space_radiative_heating),
            "upward_radiation_W_m2": flux.upward_radiative_flux.tolist(),
            "downward_radiation_W_m2": flux.downward_radiative_flux.tolist(),
            "diffusivity_m2_s": flux.turbulent_diffusivity.tolist(),
            "buoyancy_frequency_squared_s_minus2": flux.buoyancy_frequency_squared.tolist(),
            "omitted_terminal_fall_gravitational_power_W_m2": float(
                flux.fall_potential_power
            ),
            "mixing_temperature_variance_dissipation_K2_m_s": float(
                flux.mixing_temperature_variance_dissipation
            ),
            "derivative_valid": bool(flux.derivative_valid),
            "stable_step_seconds": float(flux.stable_step),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--spinup-steps", type=int, default=0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--solar-down", type=float, default=340.0)
    parser.add_argument("--wind-speed", type=float, default=5.0)
    parser.add_argument("--restart", help="Write the final native array checkpoint.")
    parser.add_argument(
        "--resume", help="Continue a checkpoint with the same numeric physics."
    )
    args = parser.parse_args()
    if args.layers < 2 or args.steps < 1 or args.spinup_steps < 0 or args.dt <= 0:
        parser.error("layers>=2, steps>=1, spinup>=0 and dt>0 are required")
    jax.config.update("jax_enable_x64", True)
    plan, initial = build_column(args.layers)
    if args.resume:
        initial = plan.load_checkpoint(args.resume)
    forcing = dict(
        solar_down=args.solar_down,
        wind_speed=args.wind_speed,
        ventilation=0.05,
        shear=0.01,
    )
    run = eqx.filter_jit(
        lambda state, steps: plan.advance(state, args.dt, steps, **forcing)
    )
    if args.spinup_steps:
        initial, successful = run(initial, args.spinup_steps)
        if not bool(jnp.all(successful)):
            raise RuntimeError(
                "Spinup rejected; reduce dt or inspect physical domain and inventories."
            )
    final, successful = run(initial, args.steps)
    if not bool(jnp.all(successful)):
        raise RuntimeError(
            "Column rejected; reduce dt or inspect physical domain and inventories."
        )
    report = physical_report(plan, initial, final, args.dt, forcing)
    report.update(
        forcing=forcing,
        optical_reference="illustrative grey; not observational calibration",
        scope="forced fixed-volume caloric column, not a hydrostatic or momentum solver",
    )
    print(json.dumps(report, indent=2))
    if args.restart:
        plan.save_checkpoint(args.restart, final)


if __name__ == "__main__":
    main()
