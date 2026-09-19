#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded engineering controls spanning mechanics, fluids, energy, Earth, and bio."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...qualification import CapabilityProfile, SupportTuple


def miner_damage(cycle_counts: ArrayLike, cycles_to_failure: ArrayLike, /) -> Array:
    return jnp.sum(jnp.asarray(cycle_counts) / jnp.asarray(cycles_to_failure), axis=-1)


def jeffcott_critical_speed_rad_s(
    stiffness_n_m: ArrayLike, mass_kg: ArrayLike, /
) -> Array:
    return jnp.sqrt(jnp.asarray(stiffness_n_m) / jnp.asarray(mass_kg))


def euler_turbomachinery_specific_work(
    blade_speed_m_s: ArrayLike,
    tangential_velocity_out_m_s: ArrayLike,
    tangential_velocity_in_m_s: ArrayLike,
    /,
) -> Array:
    return jnp.asarray(blade_speed_m_s) * (
        jnp.asarray(tangential_velocity_out_m_s) - jnp.asarray(tangential_velocity_in_m_s)
    )


def ideal_otto_efficiency(
    compression_ratio: ArrayLike, heat_capacity_ratio: float, /
) -> Array:
    return 1.0 - jnp.asarray(compression_ratio) ** (1.0 - float(heat_capacity_ratio))


def messinger_freezing_fraction(
    convective_loss_w_m2: ArrayLike, sensible_and_latent_load_w_m2: ArrayLike, /
) -> Array:
    return jnp.clip(
        jnp.asarray(convective_loss_w_m2)
        / jnp.maximum(jnp.asarray(sensible_and_latent_load_w_m2), 1.0e-30),
        0.0,
        1.0,
    )


def darcy_well_rate(
    permeability_m2: ArrayLike,
    thickness_m: float,
    pressure_drawdown_pa: ArrayLike,
    viscosity_pa_s: float,
    logarithmic_radius_ratio: float,
    /,
) -> Array:
    return (
        2.0
        * jnp.pi
        * jnp.asarray(permeability_m2)
        * float(thickness_m)
        * jnp.asarray(pressure_drawdown_pa)
        / (float(viscosity_pa_s) * float(logarithmic_radius_ratio))
    )


def black_oil_tank_pressure(
    initial_pressure_pa: ArrayLike,
    initial_oil_m3: ArrayLike,
    cumulative_oil_production_m3: ArrayLike,
    total_compressibility_pa_inv: float,
    /,
) -> Array:
    if total_compressibility_pa_inv <= 0.0:
        raise ValueError("Total compressibility must be positive.")
    fraction = jnp.asarray(cumulative_oil_production_m3) / jnp.asarray(initial_oil_m3)
    return jnp.asarray(initial_pressure_pa) + jnp.log(
        jnp.maximum(1.0 - fraction, 1.0e-30)
    ) / float(total_compressibility_pa_inv)


def manning_discharge(
    area_m2: ArrayLike,
    hydraulic_radius_m: ArrayLike,
    slope: ArrayLike,
    roughness_s_m13: float,
    /,
) -> Array:
    return (
        jnp.asarray(area_m2)
        * jnp.asarray(hydraulic_radius_m) ** (2.0 / 3.0)
        * jnp.sqrt(jnp.asarray(slope))
        / float(roughness_s_m13)
    )


def hydrostatic_heave_stiffness(
    waterplane_area_m2: ArrayLike,
    density_kg_m3: float = 1025.0,
    gravity_m_s2: float = 9.80665,
    /,
) -> Array:
    return float(density_kg_m3) * float(gravity_m_s2) * jnp.asarray(waterplane_area_m2)


def wind_turbine_power(
    wind_speed_m_s: ArrayLike,
    swept_area_m2: float,
    power_coefficient: ArrayLike,
    density_kg_m3: float = 1.225,
    /,
) -> Array:
    return (
        0.5
        * float(density_kg_m3)
        * float(swept_area_m2)
        * jnp.asarray(power_coefficient)
        * jnp.asarray(wind_speed_m_s) ** 3
    )


def windkessel_pressure_rate(
    pressure_pa: ArrayLike,
    inflow_m3_s: ArrayLike,
    resistance_pa_s_m3: float,
    compliance_m3_pa: float,
    venous_pressure_pa: float = 0.0,
    /,
) -> Array:
    return (
        jnp.asarray(inflow_m3_s)
        - (jnp.asarray(pressure_pa) - float(venous_pressure_pa))
        / float(resistance_pa_s_m3)
    ) / float(compliance_m3_pa)


def rocket_delta_v(
    exhaust_velocity_m_s: ArrayLike,
    initial_mass_kg: ArrayLike,
    final_mass_kg: ArrayLike,
    /,
) -> Array:
    return jnp.asarray(exhaust_velocity_m_s) * jnp.log(
        jnp.asarray(initial_mass_kg) / jnp.asarray(final_mass_kg)
    )


def time_to_collision(
    relative_distance_m: ArrayLike, closing_speed_m_s: ArrayLike, /
) -> Array:
    speed = jnp.asarray(closing_speed_m_s)
    return jnp.where(speed > 0.0, jnp.asarray(relative_distance_m) / speed, jnp.inf)


def pendulum_angular_acceleration(
    angle_rad: ArrayLike,
    angular_velocity_rad_s: ArrayLike,
    length_m: float,
    damping_s_inv: float = 0.0,
    gravity_m_s2: float = 9.80665,
    /,
) -> Array:
    return -float(gravity_m_s2) / float(length_m) * jnp.sin(
        jnp.asarray(angle_rad)
    ) - float(damping_s_inv) * jnp.asarray(angular_velocity_rad_s)


def impact_energy_j(mass_kg: ArrayLike, velocity_m_s: ArrayLike, /) -> Array:
    return 0.5 * jnp.asarray(mass_kg) * jnp.asarray(velocity_m_s) ** 2


def composite_longitudinal_modulus(
    fiber_modulus_pa: ArrayLike,
    matrix_modulus_pa: ArrayLike,
    fiber_fraction: ArrayLike,
    /,
) -> Array:
    fraction = jnp.asarray(fiber_fraction)
    return fraction * jnp.asarray(fiber_modulus_pa) + (1.0 - fraction) * jnp.asarray(
        matrix_modulus_pa
    )


def sdof_natural_frequency_rad_s(
    stiffness_n_m: ArrayLike, mass_kg: ArrayLike, /
) -> Array:
    return jnp.sqrt(jnp.asarray(stiffness_n_m) / jnp.asarray(mass_kg))


def aeroelastic_divergence_speed_m_s(
    torsional_stiffness_n_m_rad: ArrayLike,
    density_kg_m3: float,
    area_m2: float,
    moment_slope_m: float,
    /,
) -> Array:
    denominator = float(density_kg_m3) * float(area_m2) * float(moment_slope_m)
    return jnp.sqrt(2.0 * jnp.asarray(torsional_stiffness_n_m_rad) / denominator)


def fire_heat_release_w(
    oxygen_consumption_kg_s: ArrayLike,
    heat_per_oxygen_j_kg: float = 13.1e6,
    /,
) -> Array:
    return float(heat_per_oxygen_j_kg) * jnp.asarray(oxygen_consumption_kg_s)


def hvac_zone_temperature_rate_k_s(
    zone_temperature_k: ArrayLike,
    supply_temperature_k: ArrayLike,
    mass_flow_kg_s: ArrayLike,
    heat_capacity_j_kg_k: float,
    zone_thermal_capacity_j_k: float,
    internal_gain_w: ArrayLike = 0.0,
    /,
) -> Array:
    sensible = (
        jnp.asarray(mass_flow_kg_s)
        * float(heat_capacity_j_kg_k)
        * (jnp.asarray(supply_temperature_k) - jnp.asarray(zone_temperature_k))
    )
    return (sensible + jnp.asarray(internal_gain_w)) / float(zone_thermal_capacity_j_k)


def joule_power_w(current_a: ArrayLike, resistance_ohm: ArrayLike, /) -> Array:
    return jnp.asarray(current_a) ** 2 * jnp.asarray(resistance_ohm)


def effective_stress_pa(
    total_stress_pa: ArrayLike, pore_pressure_pa: ArrayLike, biot: float = 1.0, /
) -> Array:
    stress = jnp.asarray(total_stress_pa)
    identity = jnp.eye(stress.shape[-1], dtype=stress.dtype)
    return (
        stress - float(biot) * jnp.asarray(pore_pressure_pa)[..., None, None] * identity
    )


def first_order_chemistry_rate(
    concentration_mol_m3: ArrayLike, rate_s_inv: float, /
) -> Array:
    return -float(rate_s_inv) * jnp.asarray(concentration_mol_m3)


def sea_ice_growth_rate_m_s(
    conductive_flux_w_m2: ArrayLike,
    ocean_flux_w_m2: ArrayLike,
    density_kg_m3: float,
    latent_heat_j_kg: float,
    /,
) -> Array:
    return (jnp.asarray(conductive_flux_w_m2) - jnp.asarray(ocean_flux_w_m2)) / (
        float(density_kg_m3) * float(latent_heat_j_kg)
    )


def darcy_weisbach_pressure_drop_pa(
    friction_factor: ArrayLike,
    length_m: float,
    diameter_m: float,
    density_kg_m3: float,
    velocity_m_s: ArrayLike,
    /,
) -> Array:
    return (
        jnp.asarray(friction_factor)
        * float(length_m)
        / float(diameter_m)
        * 0.5
        * float(density_kg_m3)
        * jnp.asarray(velocity_m_s) ** 2
    )


def mineral_recovery(recovered_mass_kg: ArrayLike, feed_mass_kg: ArrayLike, /) -> Array:
    return jnp.asarray(recovered_mass_kg) / jnp.asarray(feed_mass_kg)


def three_phase_branch_current(
    sending_voltage_v: ArrayLike,
    receiving_voltage_v: ArrayLike,
    admittance_s: ArrayLike,
    /,
) -> Array:
    return jnp.asarray(admittance_s) @ (
        jnp.asarray(sending_voltage_v) - jnp.asarray(receiving_voltage_v)
    )


def debye_length_m(
    temperature_k: ArrayLike,
    electron_density_m3: ArrayLike,
    permittivity_f_m: float,
    /,
) -> Array:
    boltzmann = 1.380649e-23
    elementary_charge = 1.602176634e-19
    return jnp.sqrt(
        float(permittivity_f_m)
        * boltzmann
        * jnp.asarray(temperature_k)
        / (jnp.asarray(electron_density_m3) * elementary_charge**2)
    )


def ideal_diode_current_a(
    saturation_current_a: ArrayLike,
    voltage_v: ArrayLike,
    temperature_k: ArrayLike,
    ideality_factor: float = 1.0,
    /,
) -> Array:
    thermal_voltage = 1.380649e-23 * jnp.asarray(temperature_k) / 1.602176634e-19
    return jnp.asarray(saturation_current_a) * (
        jnp.exp(jnp.asarray(voltage_v) / (float(ideality_factor) * thermal_voltage)) - 1.0
    )


def cstr_conversion(
    rate_constant_s_inv: ArrayLike, residence_time_s: ArrayLike, /
) -> Array:
    damkohler = jnp.asarray(rate_constant_s_inv) * jnp.asarray(residence_time_s)
    return damkohler / (1.0 + damkohler)


def engineering_system_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("durability.miner", "linear-damage"),
        ("rotordynamics.jeffcott", "single-disk"),
        ("turbomachinery.euler-work", "one-dimensional"),
        ("engine.otto-cycle", "ideal-gas"),
        ("icing.messinger", "bounded-freezing-fraction"),
        ("reservoir.darcy-well", "radial-steady"),
        ("reservoir.black-oil-tank", "slightly-compressible-material-balance"),
        ("hydrology.manning", "uniform-open-channel"),
        ("marine.heave-stiffness", "linear-hydrostatic"),
        ("wind.power", "actuator-disk-power"),
        ("biomedical.windkessel", "two-element"),
        ("aerospace.rocket-delta-v", "ideal-rocket"),
        ("autonomy.time-to-collision", "constant-relative-speed"),
        ("multibody.pendulum", "single-revolute"),
        ("crash.impact-energy", "rigid-kinetic"),
        ("composite-structures.rule-of-mixtures", "longitudinal"),
        ("civil-structures.sdof", "linear-natural-frequency"),
        ("aeroelasticity.divergence", "linear-static"),
        ("fire.oxygen-consumption", "heat-release"),
        ("hvac.zone-energy", "single-zone"),
        ("electronics.joule-heating", "lumped-resistance"),
        ("geotechnical.effective-stress", "biot-isotropic"),
        ("atmosphere.first-order-chemistry", "box-model"),
        ("cryosphere.sea-ice-growth", "thermodynamic-column"),
        ("flow-assurance.darcy-weisbach", "single-phase-pipe-control"),
        ("mineral-processing.recovery", "mass-ratio"),
        ("power.unbalanced-branch", "three-phase-admittance"),
        ("plasma.debye-length", "ideal-electron-screening"),
        ("tcad.ideal-diode", "shockley"),
        ("multiphase-reactor.cstr", "first-order-conversion"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("analytic-control", "public-workflow", "nonclaim"),
        )
        for name, formulation in specs
    )


__all__ = [
    "aeroelastic_divergence_speed_m_s",
    "black_oil_tank_pressure",
    "composite_longitudinal_modulus",
    "cstr_conversion",
    "darcy_weisbach_pressure_drop_pa",
    "darcy_well_rate",
    "debye_length_m",
    "effective_stress_pa",
    "engineering_system_candidate_profiles",
    "euler_turbomachinery_specific_work",
    "fire_heat_release_w",
    "first_order_chemistry_rate",
    "hydrostatic_heave_stiffness",
    "hvac_zone_temperature_rate_k_s",
    "ideal_diode_current_a",
    "ideal_otto_efficiency",
    "impact_energy_j",
    "jeffcott_critical_speed_rad_s",
    "joule_power_w",
    "manning_discharge",
    "messinger_freezing_fraction",
    "mineral_recovery",
    "miner_damage",
    "pendulum_angular_acceleration",
    "rocket_delta_v",
    "sdof_natural_frequency_rad_s",
    "sea_ice_growth_rate_m_s",
    "three_phase_branch_current",
    "time_to_collision",
    "wind_turbine_power",
    "windkessel_pressure_rate",
]
