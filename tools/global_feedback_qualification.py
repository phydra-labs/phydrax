# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Chunked native global feedback experiment; execution is not climate qualification.

Long recipe (420 days per case, 180-day excluded spinup):
JAX_ENABLE_X64=1 python tools/global_feedback_qualification.py --sensitivity
Short path exercise, never statistically qualified:
JAX_ENABLE_X64=1 python tools/global_feedback_qualification.py --smoke --scenarios baseline
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import t as student_t

from phydrax.applications.atmosphere._equilibration import (
    global_flux_residuals,
    precondition_global_fluxes,
)
from phydrax.applications.atmosphere._global import (
    GlobalPrimitiveEquationPlan,
    read_global_atmosphere_checkpoint,
    write_global_atmosphere_checkpoint,
)
from phydrax.applications.atmosphere._global_surface import GlobalSurfacePhysics
from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan
from phydrax.applications.atmosphere._processes import GlobalAtmosphereProcesses
from phydrax.applications.atmosphere._radiation import (
    ColumnOpticalProperties,
    ColumnRadiationPlan,
)
from phydrax.applications.atmosphere._surface import BulkSurfaceExchangePlan, WetSlabPlan
from phydrax.applications.geophysics._vertical import HybridPressureCoordinate
from phydrax.discretization.spectral._spherical import SphericalSpectralPlan


DAY = 86400.0
SCENARIOS = {
    "baseline": (1.0, 1.0, 1.0),
    "greenhouse": (1.2, 1.0, 1.0),
    "solar": (1.0, 1.02, 1.0),
    "surface_capacity": (1.0, 1.0, 2.0),
}


RESPONSE_TOLERANCES = {
    "sst_k": 0.1,
    "atmosphere_temperature_k": 0.1,
    "eddy_kinetic_energy_m2_s2": 0.5,
    "precipitation_kg_m2_s": 1e-7,
    "toa_net_in_w_m2": 0.1,
}
REQUIRED_SENSITIVITIES = ("half_dt", "higher_resolution", "half_filter_rate")


def _checkpoint_path(directory, scenario, bandlimit, dt, filter_days):
    return Path(directory) / f"{scenario}-L{bandlimit}-dt{dt:g}-filter{filter_days:g}.npz"


def make_model(
    *,
    scenario="baseline",
    bandlimit=6,
    levels=4,
    dt=300.0,
    filter_days=2.0,
    initialization="flux-preconditioned",
):
    """Declared synthetic parameters, not a calibrated terrestrial climatology."""
    greenhouse, solar, capacity = SCENARIOS[scenario]
    thermo = MoistThermodynamicPlan()
    optics = ColumnOpticalProperties(
        shortwave_absorption=(1e-5, 0.002, 0.01, 0.01),
        shortwave_scattering=(0.0, 0.0, 0.1, 0.1),
        longwave_absorption=(1e-4, 0.03, 0.1, 0.1),
        reference_id="declared-synthetic-global-feedback-grey-coefficients-not-Earth-calibration",
    )
    boundary = GlobalSurfacePhysics(
        WetSlabPlan(thermo, dry_heat_capacity=2e7 * capacity),
        BulkSurfaceExchangePlan(stability="neutral"),
        ColumnRadiationPlan(
            optics, surface_albedo=0.25, longwave_absorption_scale=greenhouse
        ),
        solar_constant=1361.0 * solar,
    )
    processes = GlobalAtmosphereProcesses(
        thermodynamics=thermo,
        surface_physics=boundary,
        cadence=3,
        condensation_timescale=600.0,
        precipitation_timescale=1800.0,
        mixing_rate=1e-6,
    )
    space = SphericalSpectralPlan(bandlimit, sampling="gl").prepare(radius=6.371e6)
    sigma = np.linspace(0, 1, levels + 1)
    model = GlobalPrimitiveEquationPlan(
        space,
        HybridPressureCoordinate(0.1 * (1 - sigma), sigma),
        dt=dt,
        processes=processes,
        filter_rate=1 / (filter_days * DAY),
        water_limiter="conservative",
        angular_momentum_projection="energy-neutral",
    ).prepare()
    theta = model.work_space.transform.theta[:, None, None]
    phi = model.work_space.transform.phi[None, :, None]
    layer = jnp.asarray(0.5 * (sigma[1:] + sigma[:-1]))
    initial = model.initialize(
        temperature=250
        + 35 * layer
        + 4 * jnp.sin(theta) ** 2
        + 0.05 * jnp.sin(theta) ** 2 * jnp.cos(phi),
        vapor=0.001 + 0.002 * layer,
        liquid=1e-5,
        ice=1e-5,
        east=5 * jnp.sin(theta) * (1 - layer),
        surface_temperature=290.0,
        surface_water=1000.0,
    )
    initial_residual, initial_valid = global_flux_residuals(model, initial)
    if initialization == "flux-preconditioned" and scenario == "baseline":
        prepared = precondition_global_fluxes(model, initial)
        current = prepared.continuation
        evidence = {
            "method": initialization,
            "successful": bool(prepared.successful),
            "air_temperature_offset_k": float(prepared.air_temperature_offset),
            "surface_temperature_offset_k": float(prepared.surface_temperature_offset),
            "initial_flux_residual_w_m2": np.asarray(initial_residual).tolist(),
            "final_flux_residual_w_m2": np.asarray(
                prepared.final_flux_residual_w_per_m2
            ).tolist(),
            "accepted_iterations": int(prepared.accepted_iterations),
            "jacobian_condition": float(prepared.jacobian_condition),
            "preparation_energy_j": float(prepared.preparation_energy_j),
            "preparation_id": prepared.preparation_id,
            "claim": (
                "Two uniform temperature offsets reduce instantaneous global-mean "
                "fluxes; local heating, dynamical balance and stationarity remain "
                "unqualified."
            ),
        }
    elif initialization == "flux-preconditioned":
        reference_model, reference, reference_evidence = make_model(
            scenario="baseline",
            bandlimit=bandlimit,
            levels=levels,
            dt=dt,
            filter_days=filter_days,
            initialization="flux-preconditioned",
        )
        reference_view = reference_model.view(reference.state)
        reference_boundary = reference_model.plan.processes.surface_physics
        reference_surface_temperature = reference_boundary.temperature(
            reference.state.surface_water,
            reference.state.surface_energy,
            reference_model.plan.processes.thermodynamics,
        )
        current = model.initialize(
            temperature=reference_view.temperature,
            surface_pressure=reference_view.surface_pressure,
            east=reference_view.east,
            north=reference_view.north,
            vapor=reference_view.water[0],
            liquid=reference_view.water[1],
            ice=reference_view.water[2],
            surface_water=reference.state.surface_water,
            surface_temperature=reference_surface_temperature,
        )
        residual, valid = global_flux_residuals(model, current)
        evidence = {
            "method": "common-baseline-flux-preconditioned-state",
            "successful": bool(valid) and bool(reference_evidence["successful"]),
            "air_temperature_offset_k": reference_evidence["air_temperature_offset_k"],
            "surface_temperature_offset_k": reference_evidence[
                "surface_temperature_offset_k"
            ],
            "initial_flux_residual_w_m2": np.asarray(initial_residual).tolist(),
            "final_flux_residual_w_m2": np.asarray(residual).tolist(),
            "accepted_iterations": reference_evidence["accepted_iterations"],
            "jacobian_condition": reference_evidence["jacobian_condition"],
            "preparation_energy_j": float(
                model.inventories(current.state)[2] - model.inventories(initial.state)[2]
            ),
            "preparation_id": reference_evidence["preparation_id"],
            "source_baseline_preparation_id": reference_evidence["preparation_id"],
            "claim": (
                "Every intervention starts from the same baseline-preconditioned "
                "physical temperature, water, pressure and wind fields. Its own "
                "instantaneous forcing residual is not preconditioned away."
            ),
        }
    elif initialization == "raw":
        residual, valid = initial_residual, initial_valid
        current = initial
        evidence = {
            "method": initialization,
            "successful": bool(valid),
            "air_temperature_offset_k": 0.0,
            "surface_temperature_offset_k": 0.0,
            "initial_flux_residual_w_m2": np.asarray(residual).tolist(),
            "final_flux_residual_w_m2": np.asarray(residual).tolist(),
            "accepted_iterations": 0,
            "jacobian_condition": None,
            "preparation_energy_j": 0.0,
            "preparation_id": None,
            "claim": "Unmodified declared initial state; no flux-balance claim.",
        }
    else:
        raise ValueError("initialization must be 'raw' or 'flux-preconditioned'.")
    return model, current, evidence


def snapshot(model, continuation):
    """Resolved diagnostics using native quadrature, operators and modal layout."""
    state = continuation.state
    view = model.view(state)
    processes = model.plan.processes
    boundary = processes.surface_physics
    flux = boundary.evaluate(
        processes.thermodynamics,
        view,
        state.surface_water,
        state.surface_energy,
        continuation.held_forcing.solar_down,
    )
    budget = model.budget_rates(state, continuation.held_forcing)
    area = 4 * jnp.pi * model.plan.space.radius**2
    mean = lambda x: model.work_space.integral(x) / area
    mass = view.layer_mass
    enthalpy = processes.thermodynamics.enthalpy(view.temperature, *view.water)
    water = sum(view.water)
    ubar, vbar = jnp.mean(view.east, axis=1), jnp.mean(view.north, axis=1)
    up, vp = view.east - ubar[:, None, :], view.north - vbar[:, None, :]
    circumference = (
        2 * jnp.pi * model.plan.space.radius * jnp.sin(model.work_space.transform.theta)
    )
    mass_v = mass * view.north
    meridional_heat = jnp.mean(mass_v * enthalpy, axis=1)
    meridional_water = jnp.mean(mass_v * water, axis=1)
    eddy_heat = meridional_heat - jnp.mean(mass_v, axis=1) * jnp.mean(enthalpy, axis=1)
    eddy_water = meridional_water - jnp.mean(mass_v, axis=1) * jnp.mean(water, axis=1)
    rain = (
        jnp.sum(mass * (view.water[1] + view.water[2]), axis=-1)
        / processes.precipitation_timescale
    )
    mechanical = view.geopotential + 0.5 * (view.east**2 + view.north**2)
    _, _, hl, hi = processes.thermodynamics.phase_enthalpies(view.temperature)
    rain_energy = (
        jnp.sum(
            mass
            * (view.water[1] * (hl + mechanical) + view.water[2] * (hi + mechanical)),
            axis=-1,
        )
        / processes.precipitation_timescale
    )
    surface_power = (
        flux.radiation.surface_heating
        - flux.exchange.sensible_heat
        - flux.exchange.water_enthalpy
        - flux.exchange.water_mass * mechanical[..., -1]
        + rain_energy
    )
    k2 = model.plan.space.negative_laplacian_levels()
    inverse_k2 = jnp.where(k2 > 0, 1 / jnp.where(k2 > 0, k2, 1.0), 0.0)
    # Orthonormal scalar harmonics + native inverse-Laplacian wind inversion:
    # sum_ell E_ell = global mean 1/2*(u^2+v^2), per layer, not mass weighted.
    spectrum = (
        jnp.sum(jnp.abs(state.vorticity) ** 2 + jnp.abs(state.divergence) ** 2, axis=1)
        * inverse_k2[:, None]
        / (8 * jnp.pi)
    )
    atmospheric_energy = model.inventories(state)[2] - model.work_space.integral(
        state.surface_energy + state.environment_energy
    )
    zonal_wind = jnp.sum(jnp.mean(mass, axis=1) * ubar, axis=-1) / jnp.sum(
        jnp.mean(mass, axis=1), axis=-1
    )
    return {
        "sst_k": mean(flux.surface_temperature),
        "atmosphere_temperature_k": jnp.sum(mean(mass * view.temperature))
        / jnp.sum(mean(mass)),
        "eddy_kinetic_energy_m2_s2": jnp.sum(mean(mass * 0.5 * (up**2 + vp**2)))
        / jnp.sum(mean(mass)),
        "precipitation_kg_m2_s": mean(rain),
        "evaporation_kg_m2_s": mean(flux.exchange.water_mass),
        "toa_net_in_w_m2": -mean(flux.radiation.space_heating),
        "surface_net_in_w_m2": mean(surface_power),
        "surface_radiation_in_w_m2": mean(flux.radiation.surface_heating),
        "sensible_to_air_w_m2": mean(flux.exchange.sensible_heat),
        "donor_enthalpy_to_air_w_m2": mean(flux.exchange.water_enthalpy),
        "boundary_mechanical_to_air_w_m2": mean(
            flux.exchange.water_mass * mechanical[..., -1]
        ),
        "precipitation_energy_to_slab_w_m2": mean(rain_energy),
        "radiative_closure_max_w_m2": jnp.max(jnp.abs(flux.radiation.budget_residual)),
        "closed_rhs_w_m2": budget.energy_residual_w_per_m2,
        "process_rhs_w_m2": budget.process_energy_w_per_m2,
        "angular_momentum_kg_m2_s": budget.angular_momentum_kg_m2_per_s,
        "angular_momentum_tendency_nm": budget.angular_momentum_tendency_nm,
        "mountain_torque_nm": budget.terrain_torque_nm,
        "process_torque_nm": budget.process_torque_nm,
        "torque_residual_nm": budget.torque_residual_nm,
        "raw_torque_residual_nm": budget.raw_torque_residual_nm,
        "angular_momentum_projection_torque_nm": budget.projection_torque_nm,
        "angular_momentum_projection_energy_w_m2": (
            budget.projection_energy_power_w_per_m2
        ),
        "maximum_angular_momentum_projection_acceleration_m_per_s2": (
            budget.maximum_projection_acceleration_m_per_s2
        ),
        "angular_momentum_projection_fraction": budget.projection_fraction,
        "atmosphere_energy_j_m2": atmospheric_energy / area,
        "slab_energy_j_m2": mean(state.surface_energy),
        "environment_energy_j_m2": mean(state.environment_energy),
        "zonal_east_m_s": ubar,
        "zonal_mass_weighted_east_m_s": zonal_wind,
        "northern_jet_speed_m_s": jnp.max(
            jnp.where(
                model.work_space.transform.theta <= jnp.pi / 2, zonal_wind, -jnp.inf
            )
        ),
        "southern_jet_speed_m_s": jnp.max(
            jnp.where(
                model.work_space.transform.theta >= jnp.pi / 2, zonal_wind, -jnp.inf
            )
        ),
        "zonal_temperature_k": jnp.mean(view.temperature, axis=1),
        "zonal_sst_k": jnp.mean(flux.surface_temperature, axis=1),
        "eddy_momentum_flux_m2_s2": jnp.mean(up * vp, axis=1),
        "meridional_heat_transport_w": circumference * jnp.sum(meridional_heat, axis=-1),
        "eddy_heat_transport_w": circumference * jnp.sum(eddy_heat, axis=-1),
        "meridional_water_transport_kg_s": circumference
        * jnp.sum(meridional_water, axis=-1),
        "eddy_water_transport_kg_s": circumference * jnp.sum(eddy_water, axis=-1),
        "mass_overturning_kg_s": circumference[:, None]
        * jnp.cumsum(jnp.mean(mass_v, axis=1), axis=-1),
        "kinetic_spectrum_m2_s2": spectrum,
    }


@eqx.filter_jit
def native_chunk(model, continuation, *, steps_per_sample, samples, spinup_time=0.0):
    """Only native model.advance inside compiled scans; no alternate state/solver."""

    def sample(current, _):
        sample_start = current.time

        def step(carry, _):
            result = model.advance(carry)
            return result.continuation, result.evidence

        current, evidence = jax.lax.scan(step, current, None, length=steps_per_sample)
        step_times = sample_start + model.plan.dt * jnp.arange(1, steps_per_sample + 1)
        production = evidence.accepted & (step_times > spinup_time)
        return current, (
            current.time,
            snapshot(model, current),
            {
                "accepted": jnp.all(evidence.accepted),
                "admissible": jnp.all(evidence.admissible),
                "process_successful": jnp.all(evidence.process_successful),
                "courant": jnp.max(evidence.advective_courant),
                "max_abs_step_drift_w_m2": jnp.max(
                    jnp.abs(model.step_energy_flux(evidence))
                ),
                "filter_energy_j": jnp.sum(evidence.filter_energy),
                "production_seconds": model.plan.dt * jnp.sum(production),
                "production_filter_energy_j": jnp.sum(
                    jnp.where(production, evidence.filter_energy, 0.0)
                ),
                "production_absolute_filter_energy_j": jnp.sum(
                    jnp.where(
                        production,
                        jnp.abs(evidence.filter_kinetic_energy)
                        + jnp.abs(
                            evidence.filter_energy - evidence.filter_kinetic_energy
                        ),
                        0.0,
                    )
                ),
                "filter_angular_momentum": jnp.sum(
                    jnp.where(evidence.accepted, evidence.filter_angular_momentum, 0.0)
                ),
                "water_total_redistribution_mass_kg": jnp.sum(
                    jnp.where(
                        evidence.accepted,
                        evidence.water_total_redistribution_mass,
                        0.0,
                    )
                ),
                "water_phase_repartition_mass_kg": jnp.sum(
                    jnp.where(
                        evidence.accepted,
                        evidence.water_phase_repartition_mass,
                        0.0,
                    )
                ),
                "water_projection_activated": jnp.any(
                    evidence.accepted & evidence.water_projection_active
                ),
                "maximum_water_total_redistribution_fraction": jnp.max(
                    jnp.where(
                        evidence.accepted,
                        evidence.water_total_redistribution_fraction,
                        0.0,
                    )
                ),
                "maximum_water_phase_repartition_fraction": jnp.max(
                    jnp.where(
                        evidence.accepted,
                        evidence.water_phase_repartition_fraction,
                        0.0,
                    )
                ),
                "physical_phase_conversion_mass_kg": jnp.sum(
                    jnp.where(
                        evidence.accepted,
                        evidence.physical_phase_conversion_mass,
                        0.0,
                    )
                ),
                "maximum_water_projection_energy_w_m2": jnp.max(
                    jnp.where(
                        evidence.accepted,
                        evidence.maximum_water_projection_energy_residual,
                        0.0,
                    )
                )
                / (4 * jnp.pi * model.plan.space.radius**2 * model.plan.dt),
                "maximum_raw_torque_residual_nm": jnp.max(
                    jnp.where(
                        evidence.accepted,
                        jnp.abs(evidence.raw_torque_residual_nm),
                        0.0,
                    )
                ),
                "maximum_corrected_torque_residual_nm": jnp.max(
                    jnp.where(
                        evidence.accepted,
                        jnp.abs(
                            evidence.raw_torque_residual_nm
                            + evidence.angular_momentum_projection_torque_nm
                        ),
                        0.0,
                    )
                ),
                "maximum_angular_momentum_projection_fraction": jnp.max(
                    jnp.where(
                        evidence.accepted,
                        evidence.angular_momentum_projection_fraction,
                        0.0,
                    )
                ),
                "maximum_angular_momentum_projection_energy_w_m2": jnp.max(
                    jnp.where(
                        evidence.accepted,
                        jnp.abs(evidence.angular_momentum_projection_energy_power_w),
                        0.0,
                    )
                )
                / (4 * jnp.pi * model.plan.space.radius**2),
                "maximum_angular_momentum_projection_acceleration_m_per_s2": (
                    jnp.max(
                        jnp.where(
                            evidence.accepted,
                            evidence.maximum_angular_momentum_projection_acceleration,
                            0.0,
                        )
                    )
                ),
                "angular_momentum_projection_successful": jnp.all(
                    evidence.angular_momentum_projection_successful
                    & evidence.angular_momentum_projection_evaluated
                ),
                "energy_residual_j": jnp.sum(evidence.energy_residual),
            },
        )

    return jax.lax.scan(sample, continuation, None, length=samples)


def block_statistics(times_days, values, *, spinup_days, block_days):
    """Nonoverlapping batch means; incomplete final blocks are excluded explicitly."""
    times, values = np.asarray(times_days), np.asarray(values)
    selected = times > spinup_days
    times, values = times[selected], values[selected]
    complete = (
        max(0, int(np.floor((times[-1] - spinup_days) / block_days + 1e-10)))
        if times.size
        else 0
    )
    blocks = []
    for index in range(complete):
        mask = (times > spinup_days + index * block_days) & (
            times <= spinup_days + (index + 1) * block_days + 1e-10
        )
        if np.any(mask):
            blocks.append(values[mask].mean(axis=0))
    block = np.asarray(blocks)
    result = {"samples_after_spinup": int(times.size), "complete_blocks": len(blocks)}
    if len(blocks) < 2:
        return {
            **result,
            "mean": None,
            "standard_error": None,
            "ci95_half_width": None,
            "lag1_block_correlation": None,
            "sampling_adequate": False,
            "stationarity_consistent": False,
            "half_record_change": None,
        }
    stderr = block.std(axis=0, ddof=1) / np.sqrt(len(block))
    halfwidth = student_t.ppf(0.975, len(block) - 1) * stderr
    centered = block - block.mean(axis=0)
    denominator = np.sum(centered**2, axis=0)
    lag = np.divide(
        np.sum(centered[1:] * centered[:-1], axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    split = len(block) // 2
    change = block[split:].mean(axis=0) - block[:split].mean(axis=0)
    enough = len(block) >= 8 and bool(np.all(np.abs(lag) < 0.3))
    return {
        **result,
        "mean": block.mean(axis=0).tolist(),
        "standard_error": stderr.tolist(),
        "ci95_half_width": halfwidth.tolist(),
        "lag1_block_correlation": lag.tolist(),
        "sampling_adequate": enough,
        "stationarity_consistent": enough
        and bool(np.all(np.abs(change) <= 2 * halfwidth)),
        "half_record_change": change.tolist(),
    }


def deterministic_equilibrium_assessment(
    statistics,
    last_record,
    *,
    numerical_accounting,
    radiative_equilibrium,
    duration_adequate,
    temperature_change_k,
    flux_change_w_m2,
    endpoint_flux_w_m2,
    eke_change_m2_s2,
    precipitation_change_kg_m2_s,
):
    """Qualify a near-fixed equilibrium without inventing independent samples."""
    limits = {
        "sst_k": temperature_change_k,
        "atmosphere_temperature_k": temperature_change_k,
        "eddy_kinetic_energy_m2_s2": eke_change_m2_s2,
        "precipitation_kg_m2_s": precipitation_change_kg_m2_s,
        "toa_net_in_w_m2": flux_change_w_m2,
        "surface_net_in_w_m2": flux_change_w_m2,
    }
    changes = {key: statistics[key]["half_record_change"] for key in limits}
    finite_changes = all(
        value is not None
        and np.ndim(value) == 0
        and np.isfinite(value)
        and abs(float(value)) <= limits[key]
        for key, value in changes.items()
    )
    endpoint = {
        key: float(last_record[key]) for key in ("toa_net_in_w_m2", "surface_net_in_w_m2")
    }
    endpoint_balanced = all(
        np.isfinite(value) and abs(value) <= endpoint_flux_w_m2
        for value in endpoint.values()
    )
    qualified = bool(
        numerical_accounting
        and radiative_equilibrium
        and duration_adequate
        and finite_changes
        and endpoint_balanced
    )
    return {
        "qualified": qualified,
        "half_record_changes": changes,
        "change_limits": limits,
        "endpoint_flux_w_m2": endpoint,
        "endpoint_flux_limit_w_m2": endpoint_flux_w_m2,
        "duration_adequate": bool(duration_adequate),
        "scope": (
            "Deterministic near-fixed-equilibrium evidence; no independent "
            "sample, uncertainty, variability, climate, or response claim."
        ),
    }


def _bitwise(left, right):
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
        )
    )


def qualify(
    *,
    scenario,
    bandlimit,
    levels,
    dt,
    filter_days,
    days,
    spinup_days,
    sample_days,
    block_days,
    chunk_samples,
    maximum_drift_w_m2,
    deterministic_temperature_change_k=0.2,
    deterministic_flux_change_w_m2=0.5,
    deterministic_endpoint_flux_w_m2=0.1,
    deterministic_eke_change_m2_s2=0.01,
    deterministic_precipitation_change_kg_m2_s=1e-7,
    checkpoint_dir=None,
    initialization="flux-preconditioned",
    resume_checkpoint=None,
):
    steps_per_sample = round(sample_days * DAY / dt)
    samples_total = round(days / sample_days)
    samples_per_block = round(block_days / sample_days)
    if samples_per_block < 2 or not np.isclose(
        samples_per_block * sample_days, block_days
    ):
        raise ValueError(
            "A statistical block must contain an integer number of at least two sampling intervals."
        )
    if (
        steps_per_sample < 1
        or samples_total < 1
        or not np.isclose(steps_per_sample * dt, sample_days * DAY)
        or not np.isclose(samples_total * sample_days, days)
    ):
        raise ValueError(
            "sample_days must be an integer number of timesteps; days an integer number of samples."
        )
    model, current, initialization_evidence = make_model(
        scenario=scenario,
        bandlimit=bandlimit,
        levels=levels,
        dt=dt,
        filter_days=filter_days,
        initialization=initialization,
    )
    if resume_checkpoint is not None:
        current = read_global_atmosphere_checkpoint(resume_checkpoint, model, current)
        resumed_residual, resumed_valid = global_flux_residuals(model, current)
        initialization_evidence = {
            "method": "native-checkpoint-resume",
            "successful": bool(resumed_valid),
            "checkpoint": str(resume_checkpoint),
            "segment_initial_flux_residual_w_m2": np.asarray(resumed_residual).tolist(),
            "claim": (
                "Exact continuation begins a new statistical segment; prior "
                "samples are not silently reused."
            ),
        }
    initial = current
    segment_start_days = float(initial.time) / DAY
    spinup_end_days = segment_start_days + spinup_days
    times, records, gates = [], [], []
    for start in range(0, samples_total, chunk_samples):
        count = min(chunk_samples, samples_total - start)
        current, (time, record, gate) = native_chunk(
            model,
            current,
            steps_per_sample=steps_per_sample,
            samples=count,
            spinup_time=spinup_end_days * DAY,
        )
        time, record, gate = jax.device_get((time, record, gate))
        # Rejected attempts are reported, not included as repeated stationary samples.
        accepted_count = (
            int(np.argmax(~gate["accepted"])) if not np.all(gate["accepted"]) else count
        )
        times.extend(np.asarray(time[:accepted_count]) / DAY)
        records.append(
            {key: np.asarray(value[:accepted_count]) for key, value in record.items()}
        )
        gates.append(gate)
        if checkpoint_dir is not None:
            write_global_atmosphere_checkpoint(
                _checkpoint_path(checkpoint_dir, scenario, bandlimit, dt, filter_days),
                model,
                current,
            )
        if not np.all(gate["accepted"]):
            break
    all_records = {
        key: np.concatenate([record[key] for record in records]) for key in records[0]
    }
    all_gates = {key: np.concatenate([gate[key] for gate in gates]) for key in gates[0]}
    execution = (
        bool(initialization_evidence["successful"])
        and bool(np.all(all_gates["accepted"]))
        and len(times) == samples_total
    )
    statistics = {
        key: block_statistics(
            times, value, spinup_days=spinup_end_days, block_days=block_days
        )
        for key, value in all_records.items()
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "continuation.npz"
        write_global_atmosphere_checkpoint(path, model, current)
        restored = read_global_atmosphere_checkpoint(path, model, initial)
        uninterrupted, _ = native_chunk(model, current, steps_per_sample=1, samples=1)
        restarted, _ = native_chunk(model, restored, steps_per_sample=1, samples=1)
        exact_restart = _bitwise(current, restored) and _bitwise(uninterrupted, restarted)
    elapsed = float(current.time - initial.time)
    area = 4 * np.pi * model.plan.space.radius**2
    drift = (
        float(current.ledger.energy_residual - initial.ledger.energy_residual)
        / (area * elapsed)
        if elapsed > 0
        else None
    )
    filter_work = (
        float(current.ledger.filter_energy - initial.ledger.filter_energy)
        / (area * elapsed)
        if elapsed > 0
        else None
    )
    filter_torque = (
        float(np.sum(all_gates["filter_angular_momentum"])) / elapsed
        if elapsed > 0
        else None
    )
    production_seconds = float(np.sum(all_gates["production_seconds"]))
    production_filter_work = (
        float(np.sum(all_gates["production_filter_energy_j"]))
        / (area * production_seconds)
        if production_seconds > 0
        else None
    )
    production_absolute_filter_work = (
        float(np.sum(all_gates["production_absolute_filter_energy_j"]))
        / (area * production_seconds)
        if production_seconds > 0
        else None
    )
    peak_drift = float(np.max(all_gates["max_abs_step_drift_w_m2"]))
    before, after = model.inventories(initial.state), model.inventories(current.state)
    atmospheric_water_scale = max(
        float(before[1] - model.work_space.integral(initial.state.surface_water)),
        1.0,
    )
    segment_total_redistribution_fraction = float(
        (
            current.ledger.water_total_redistribution_mass
            - initial.ledger.water_total_redistribution_mass
        )
        / atmospheric_water_scale
    )
    lifetime_total_redistribution_fraction = float(
        current.ledger.water_total_redistribution_mass / atmospheric_water_scale
    )
    segment_phase_repartition_fraction = float(
        (
            current.ledger.water_phase_repartition_mass
            - initial.ledger.water_phase_repartition_mass
        )
        / atmospheric_water_scale
    )
    lifetime_phase_repartition_fraction = float(
        current.ledger.water_phase_repartition_mass / atmospheric_water_scale
    )
    segment_physical_phase_conversion = float(
        current.ledger.physical_phase_conversion_mass
        - initial.ledger.physical_phase_conversion_mass
    )
    lifetime_physical_phase_conversion = float(
        current.ledger.physical_phase_conversion_mass
    )
    segment_phase_repartition_ratio = float(
        (
            current.ledger.water_phase_repartition_mass
            - initial.ledger.water_phase_repartition_mass
        )
        / max(segment_physical_phase_conversion, 1.0)
    )
    lifetime_phase_repartition_ratio = float(
        current.ledger.water_phase_repartition_mass
        / max(lifetime_physical_phase_conversion, 1.0)
    )
    projection_activations = int(
        np.count_nonzero(all_gates["water_projection_activated"])
    )
    maximum_total_redistribution_fraction = float(
        np.max(all_gates["maximum_water_total_redistribution_fraction"])
    )
    maximum_phase_repartition_fraction = float(
        np.max(all_gates["maximum_water_phase_repartition_fraction"])
    )
    maximum_water_projection_energy = float(
        np.max(all_gates["maximum_water_projection_energy_w_m2"])
    )
    finite = all(np.all(np.isfinite(value)) for value in all_records.values())
    relative_mass = float((after[0] - before[0]) / before[0])
    relative_water = float((after[1] - before[1]) / before[1])
    torque_scale = np.maximum.reduce(
        (
            np.abs(all_records["angular_momentum_kg_m2_s"]) / DAY,
            np.abs(all_records["mountain_torque_nm"])
            + np.abs(all_records["process_torque_nm"]),
            np.ones(len(times)),
        )
    )
    raw_relative_torque = (
        float(np.max(np.abs(all_records["raw_torque_residual_nm"]) / torque_scale))
        if times
        else None
    )
    maximum_projection_fraction = float(
        np.max(all_gates["maximum_angular_momentum_projection_fraction"])
    )
    maximum_projection_energy = float(
        np.max(all_gates["maximum_angular_momentum_projection_energy_w_m2"])
    )
    maximum_projection_acceleration = float(
        np.max(all_gates["maximum_angular_momentum_projection_acceleration_m_per_s2"])
    )
    projection_successful = bool(
        np.all(all_gates["angular_momentum_projection_successful"])
    )
    angular_momentum_scale = max(abs(float(model.angular_momentum(initial.state))), 1.0)
    segment_projection_fraction = float(
        (
            current.ledger.absolute_angular_momentum_projection_impulse
            - initial.ledger.absolute_angular_momentum_projection_impulse
        )
        / angular_momentum_scale
    )
    lifetime_projection_fraction = float(
        current.ledger.absolute_angular_momentum_projection_impulse
        / angular_momentum_scale
    )
    relative_torque = (
        float(np.max(np.abs(all_records["torque_residual_nm"]) / torque_scale))
        if times
        else None
    )
    radiation_closure = (
        float(np.max(all_records["radiative_closure_max_w_m2"])) if times else None
    )
    numerical = (
        execution
        and finite
        and exact_restart
        and abs(drift) <= maximum_drift_w_m2
        and peak_drift <= maximum_drift_w_m2
    )
    numerical = numerical and (
        abs(relative_mass) <= model.plan.budget_tolerance
        and abs(relative_water) <= model.plan.budget_tolerance
        and relative_torque is not None
        and relative_torque <= 1e-10
        and raw_relative_torque is not None
        and raw_relative_torque <= model.plan.maximum_angular_momentum_projection_fraction
        and maximum_projection_fraction
        <= model.plan.maximum_angular_momentum_projection_fraction
        and lifetime_projection_fraction <= 0.01
        and maximum_projection_energy <= maximum_drift_w_m2
        and projection_successful
        and radiation_closure is not None
        and radiation_closure <= 1e-9
        and lifetime_total_redistribution_fraction <= 1e-8
        and lifetime_phase_repartition_ratio <= 0.1
        and maximum_total_redistribution_fraction
        <= 2 * model.plan.water_projection_tolerance
        and maximum_phase_repartition_fraction
        <= 2 * model.plan.maximum_water_phase_repartition_fraction
        and maximum_water_projection_energy <= maximum_drift_w_m2
    )
    climate_metrics = (
        "sst_k",
        "atmosphere_temperature_k",
        "eddy_kinetic_energy_m2_s2",
        "precipitation_kg_m2_s",
        "toa_net_in_w_m2",
    )
    sampling = all(statistics[key]["sampling_adequate"] for key in climate_metrics)
    stationary = all(
        statistics[key]["stationarity_consistent"] for key in climate_metrics
    )
    # A consistent split mean is not evidence of radiative equilibrium by itself.
    equilibrium = all(
        statistics[key]["mean"] is not None
        and abs(statistics[key]["mean"]) + statistics[key]["ci95_half_width"] <= 1.0
        for key in ("toa_net_in_w_m2", "surface_net_in_w_m2")
    )
    last_record = {
        key: value[-1].tolist() if len(value) else None
        for key, value in all_records.items()
    }
    stationary_sample_qualified = bool(
        numerical
        and sampling
        and stationary
        and equilibrium
        and days - spinup_days >= 180
    )
    deterministic_equilibrium = deterministic_equilibrium_assessment(
        statistics,
        last_record,
        numerical_accounting=numerical,
        radiative_equilibrium=equilibrium,
        duration_adequate=days - spinup_days >= 180,
        temperature_change_k=deterministic_temperature_change_k,
        flux_change_w_m2=deterministic_flux_change_w_m2,
        endpoint_flux_w_m2=deterministic_endpoint_flux_w_m2,
        eke_change_m2_s2=deterministic_eke_change_m2_s2,
        precipitation_change_kg_m2_s=(deterministic_precipitation_change_kg_m2_s),
    )
    greenhouse, solar, capacity = SCENARIOS[scenario]
    return {
        "scenario": scenario,
        "model_id": model.prepared_id,
        "parameters": {
            "bandlimit": bandlimit,
            "levels": levels,
            "dt_seconds": dt,
            "filter_days": filter_days,
            "longwave_opacity_scale": greenhouse,
            "solar_constant_w_m2": 1361 * solar,
            "dry_slab_capacity_j_m2_k": 2e7 * capacity,
            "water_initial_kg_m2": 1000,
            "solar_p2": -0.48,
            "water_projection": {
                "method": "joint-phase-Dykstra-simplex-fixed-total-water",
                "iterations": model.plan.water_projection_iterations,
                "feasibility_tolerance": model.plan.water_projection_tolerance,
                "maximum_step_phase_repartition_fraction": (
                    model.plan.maximum_water_phase_repartition_fraction
                ),
            },
            "angular_momentum_projection": {
                "method": model.plan.angular_momentum_projection,
                "maximum_step_fraction": (
                    model.plan.maximum_angular_momentum_projection_fraction
                ),
            },
        },
        "optical_provenance": {
            "reference_id": model.plan.processes.surface_physics.radiation.optics.reference_id,
            "species_order": ["dry", "vapor", "liquid", "ice"],
            "coefficient_units": "m2/kg",
            "shortwave_absorption": np.asarray(
                model.plan.processes.surface_physics.radiation.optics.shortwave_absorption
            ).tolist(),
            "shortwave_scattering": np.asarray(
                model.plan.processes.surface_physics.radiation.optics.shortwave_scattering
            ).tolist(),
            "longwave_absorption": np.asarray(
                model.plan.processes.surface_physics.radiation.optics.longwave_absorption
            ).tolist(),
        },
        "initialization": initialization_evidence,
        "sampling_policy": {
            "requested_days": days,
            "completed_days": elapsed / DAY,
            "excluded_spinup_days": spinup_days,
            "segment_start_days": segment_start_days,
            "segment_end_days": float(current.time) / DAY,
            "sample_days": sample_days,
            "block_days": block_days,
            "minimum_blocks": 8,
            "maximum_abs_block_lag1": 0.3,
            "maximum_drift_w_m2": maximum_drift_w_m2,
            "equilibrium_flux_bound_w_m2": 1.0,
        },
        "numerical_policy": {
            "maximum_relative_inventory_change": model.plan.budget_tolerance,
            "maximum_corrected_relative_torque_residual": 1e-10,
            "maximum_raw_relative_torque_residual": (
                model.plan.maximum_angular_momentum_projection_fraction
            ),
            "maximum_lifetime_projection_impulse_fraction": 0.01,
            "maximum_projection_energy_w_m2": maximum_drift_w_m2,
            "torque_scale": (
                "max(abs(atmospheric angular momentum)/day, "
                "abs(terrain torque)+abs(process torque), 1 Nm)"
            ),
            "maximum_step_total_water_redistribution_fraction": (
                2 * model.plan.water_projection_tolerance
            ),
            "maximum_lifetime_total_water_redistribution_fraction": 1e-8,
            "maximum_step_water_phase_repartition_fraction": (
                2 * model.plan.maximum_water_phase_repartition_fraction
            ),
            "maximum_lifetime_phase_repartition_to_physical_conversion": 0.1,
            "maximum_water_projection_energy_w_m2": maximum_drift_w_m2,
        },
        "execution_success": execution,
        "numerical_accounting_adequacy": numerical,
        "sampling_adequacy": sampling,
        "statistical_stationarity": stationary,
        "stationary_sample_qualified": stationary_sample_qualified,
        "deterministic_equilibrium_qualified": deterministic_equilibrium["qualified"],
        "deterministic_equilibrium": deterministic_equilibrium,
        "bitwise_restart": exact_restart,
        "segment_accepted_steps": int(current.accepted_steps - initial.accepted_steps),
        "segment_rejected_attempts": int(current.rejected_steps - initial.rejected_steps),
        "cumulative_accepted_steps": int(current.accepted_steps),
        "cumulative_rejected_attempts": int(current.rejected_steps),
        "continuation_checkpoint": (
            None
            if checkpoint_dir is None
            else str(
                _checkpoint_path(checkpoint_dir, scenario, bandlimit, dt, filter_days)
            )
        ),
        "closed_energy_drift_w_m2": drift,
        "filter_work_w_m2": filter_work,
        "production_filter_work_w_m2": production_filter_work,
        "production_absolute_filter_work_w_m2": production_absolute_filter_work,
        "filter_torque_nm": filter_torque,
        "relative_mass_change": relative_mass,
        "relative_water_change": relative_water,
        "water_projection_activated_samples": projection_activations,
        "maximum_step_total_water_redistribution_fraction": (
            maximum_total_redistribution_fraction
        ),
        "segment_total_water_redistribution_fraction": (
            segment_total_redistribution_fraction
        ),
        "lifetime_total_water_redistribution_fraction": (
            lifetime_total_redistribution_fraction
        ),
        "maximum_step_water_phase_repartition_fraction": (
            maximum_phase_repartition_fraction
        ),
        "segment_water_phase_repartition_fraction": (segment_phase_repartition_fraction),
        "lifetime_water_phase_repartition_fraction": (
            lifetime_phase_repartition_fraction
        ),
        "segment_physical_phase_conversion_mass_kg": (segment_physical_phase_conversion),
        "lifetime_physical_phase_conversion_mass_kg": (
            lifetime_physical_phase_conversion
        ),
        "segment_phase_repartition_to_physical_conversion": (
            segment_phase_repartition_ratio
        ),
        "lifetime_phase_repartition_to_physical_conversion": (
            lifetime_phase_repartition_ratio
        ),
        "maximum_corrected_relative_torque_residual": relative_torque,
        "maximum_raw_relative_torque_residual": raw_relative_torque,
        "maximum_projection_fraction": maximum_projection_fraction,
        "segment_projection_impulse_fraction": segment_projection_fraction,
        "lifetime_projection_impulse_fraction": lifetime_projection_fraction,
        "maximum_projection_energy_w_m2": maximum_projection_energy,
        "maximum_projection_acceleration_m_per_s2": (maximum_projection_acceleration),
        "maximum_water_projection_energy_w_m2": (maximum_water_projection_energy),
        "maximum_radiative_closure_w_m2": radiation_closure,
        "last_admitted_sample": {
            key: value[-1].tolist() if len(value) else None
            for key, value in all_records.items()
        },
        "peak_abs_step_drift_w_m2": peak_drift,
        "maximum_courant": float(np.max(all_gates["courant"])),
        "stage_admissibility": bool(np.all(all_gates["admissible"])),
        "process_success": bool(np.all(all_gates["process_successful"])),
        "latitude_degrees": (
            90 - np.rad2deg(np.asarray(model.work_space.transform.theta))
        ).tolist(),
        "spectral_degree": list(range(bandlimit)),
        "statistics": statistics,
        "limitations": [
            (
                "Synthetic grey coefficients and P2 solar forcing are declared "
                "experiment parameters, not Earth calibration."
            ),
            (
                "Global cloud liquid/ice relax to isobaric equilibrium; "
                "precipitation falls instantly, unlike InteractiveMoistColumnPlan "
                "finite-rate species."
            ),
            (
                "Joint phase projection preserves represented total water but "
                "changes vapor/liquid/ice composition; its mass is compared with "
                "actual physical phase conversion and never hidden."
            ),
            (
                "No sea ice, boiling, deep convection, gustiness closure or "
                "surface momentum stress; invalid slabs reject."
            ),
            (
                "Filter and mixing momentum work are explicit measured losses to "
                "filter/environment, not hidden slab heating."
            ),
            (
                "Accounting removes explicitly measured filter work; a small "
                "residual is not a bound on filter bias or physical response error."
            ),
            (
                "Batch-mean Student intervals assume blocks sufficiently "
                "independent; lag-one testing is necessary, not sufficient."
            ),
            (
                "No observational skill, resolved turbulent inertial range, "
                "orbital insolation or resolution-independent climatology claim."
            ),
        ],
    }


def compare(
    left,
    right,
    *,
    sensitivity_pairs=None,
    tolerances=None,
    maximum_response_sensitivity_fraction=0.25,
    maximum_filter_work_w_m2=0.1,
    target_signal_w_m2=1.0,
    maximum_filter_signal_fraction=0.1,
):
    """Measured paired-response robustness, not a rigorous continuum-error bound.

    Statistical significance alone never licenses a physical-response claim.
    All declared paired numerical perturbations must retain qualified sampling,
    and the upper uncertainty bound on response change must fit both dimensional
    and signal-relative limits. Wide sampling uncertainty therefore cannot make
    an unresolved sensitivity look harmless.
    """
    pairs = {} if sensitivity_pairs is None else sensitivity_pairs
    limits = RESPONSE_TOLERANCES if tolerances is None else tolerances
    policy_values = (
        maximum_response_sensitivity_fraction,
        maximum_filter_work_w_m2,
        target_signal_w_m2,
        maximum_filter_signal_fraction,
        *limits.values(),
    )
    if set(limits) != set(RESPONSE_TOLERANCES) or any(
        not np.isfinite(value) or value <= 0 for value in policy_values
    ):
        raise ValueError(
            "Response policy requires all finite positive dimensional tolerances and signal scales."
        )
    if maximum_response_sensitivity_fraction >= 1 or maximum_filter_signal_fraction >= 1:
        raise ValueError(
            "Sensitivity and filter fractions must be smaller than the declared signal."
        )

    def difference(a, b, key):
        first, second = a["statistics"][key], b["statistics"][key]
        if first["mean"] is None or second["mean"] is None:
            return None, None, False
        delta = second["mean"] - first["mean"]
        # Conservative sum allows common-initial-condition dependence.
        width = first["ci95_half_width"] + second["ci95_half_width"]
        qualified = a["stationary_sample_qualified"] and b["stationary_sample_qualified"]
        return delta, width, qualified and abs(delta) > width

    filter_limit = min(
        maximum_filter_work_w_m2, target_signal_w_m2 * maximum_filter_signal_fraction
    )

    def filter_budget(a, b):
        first, second = a["production_filter_work_w_m2"], b["production_filter_work_w_m2"]
        first_abs = a["production_absolute_filter_work_w_m2"]
        second_abs = b["production_absolute_filter_work_w_m2"]
        finite = all(
            value is not None and np.isfinite(value)
            for value in (first, second, first_abs, second_abs)
        )
        maximum = max(first_abs, second_abs) if finite else None
        contrast = second - first if finite else None
        return {
            "baseline_work_w_m2": first,
            "forced_work_w_m2": second,
            "forced_minus_baseline_work_w_m2": contrast,
            "maximum_mean_absolute_step_work_w_m2": maximum,
            "allowed_work_w_m2": filter_limit,
            "window": (
                "Accepted post-spinup steps; sum of absolute kinetic and remaining "
                "filter work prevents signed/channel cancellation."
            ),
            "within_declared_budget": bool(
                finite and maximum <= filter_limit and abs(contrast) <= filter_limit
            ),
        }

    reference_filter = filter_budget(left, right)
    results = {}
    for key in RESPONSE_TOLERANCES:
        delta, width, resolved = difference(left, right, key)
        signal_lower = max(0.0, abs(delta) - width) if delta is not None else 0.0
        sensitivity_limit = min(
            limits[key], maximum_response_sensitivity_fraction * signal_lower
        )
        evidence = {}
        for name in REQUIRED_SENSITIVITIES:
            if name not in pairs:
                evidence[name] = {"available": False, "within_declared_tolerance": False}
                continue
            baseline, forced = pairs[name]
            trial_delta, trial_width, trial_resolved = difference(baseline, forced, key)
            shift = (
                trial_delta - delta
                if trial_delta is not None and delta is not None
                else None
            )
            uncertainty = trial_width + width if shift is not None else None
            upper = abs(shift) + uncertainty if shift is not None else None
            pair_filter = filter_budget(baseline, forced)
            stable = (
                resolved
                and trial_resolved
                and upper is not None
                and upper <= sensitivity_limit
                and pair_filter["within_declared_budget"]
            )
            evidence[name] = {
                "available": True,
                "response_difference": trial_delta,
                "response_ci95_half_width": trial_width,
                "response_shift": shift,
                "combined_ci95_half_width": uncertainty,
                "response_shift_upper_bound": upper,
                "within_declared_tolerance": bool(stable),
                "filter_work_budget": pair_filter,
            }
        empirically_stable = reference_filter["within_declared_budget"] and all(
            item["within_declared_tolerance"] for item in evidence.values()
        )
        results[key] = {
            "difference": delta,
            "ci95_half_width": width,
            "statistically_resolved_difference": resolved,
            "dimensional_sensitivity_tolerance": limits[key],
            "maximum_sensitivity_fraction_of_resolved_signal": maximum_response_sensitivity_fraction,
            "resolved_signal_lower_bound": signal_lower,
            "allowed_response_shift": sensitivity_limit,
            "reference_filter_work_budget": reference_filter,
            "declared_target_signal_w_m2": target_signal_w_m2,
            "maximum_filter_signal_fraction": maximum_filter_signal_fraction,
            "paired_sensitivity": evidence,
            "empirical_response_numerical_adequacy": empirically_stable,
            "response_claim_supported": bool(
                resolved and left["scenario"] != right["scenario"] and empirically_stable
            ),
            "scope": (
                "Empirical robustness across these tested configurations only; no "
                "rigorous continuum or universal filter-bias bound."
            ),
        }
    return results


def json_ready(value):
    """Report nonfinite measurements as null, without turning a failed gate green."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS)
    )
    parser.add_argument("--bandlimit", type=int, default=6)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--filter-days", type=float, default=2.0)
    parser.add_argument("--days", type=float, default=420.0)
    parser.add_argument("--spinup-days", type=float, default=180.0)
    parser.add_argument("--sample-days", type=float, default=1.0)
    parser.add_argument("--block-days", type=float, default=30.0)
    parser.add_argument("--chunk-samples", type=int, default=10)
    parser.add_argument("--maximum-drift-w-m2", type=float, default=0.1)
    parser.add_argument("--maximum-filter-work-w-m2", type=float, default=0.1)
    parser.add_argument("--target-signal-w-m2", type=float, default=1.0)
    parser.add_argument("--maximum-filter-signal-fraction", type=float, default=0.1)
    parser.add_argument("--response-temperature-tolerance-k", type=float, default=0.1)
    parser.add_argument("--response-eke-tolerance-m2-s2", type=float, default=0.5)
    parser.add_argument(
        "--response-precipitation-tolerance-kg-m2-s", type=float, default=1e-7
    )
    parser.add_argument("--response-toa-tolerance-w-m2", type=float, default=0.1)
    parser.add_argument("--deterministic-temperature-change-k", type=float, default=0.2)
    parser.add_argument("--deterministic-flux-change-w-m2", type=float, default=0.5)
    parser.add_argument("--deterministic-endpoint-flux-w-m2", type=float, default=0.1)
    parser.add_argument("--deterministic-eke-change-m2-s2", type=float, default=0.01)
    parser.add_argument(
        "--deterministic-precipitation-change-kg-m2-s",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--maximum-response-sensitivity-fraction", type=float, default=0.25
    )
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument(
        "--initialization",
        choices=("raw", "flux-preconditioned"),
        default="flux-preconditioned",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--maximum-segments", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.dt,
            args.filter_days,
            args.days,
            args.sample_days,
            args.block_days,
            args.maximum_drift_w_m2,
        )
        <= 0
        or args.spinup_days < 0
        or args.chunk_samples < 1
        or args.bandlimit < 3
        or args.levels < 2
    ):
        parser.error(
            "Positive duration/scales, nonnegative spinup, bandlimit >= 3 and levels >= 2 required."
        )
    response_scales = (
        args.maximum_filter_work_w_m2,
        args.target_signal_w_m2,
        args.maximum_filter_signal_fraction,
        args.maximum_response_sensitivity_fraction,
        args.response_temperature_tolerance_k,
        args.response_eke_tolerance_m2_s2,
        args.response_precipitation_tolerance_kg_m2_s,
        args.response_toa_tolerance_w_m2,
        args.deterministic_temperature_change_k,
        args.deterministic_flux_change_w_m2,
        args.deterministic_endpoint_flux_w_m2,
        args.deterministic_eke_change_m2_s2,
        args.deterministic_precipitation_change_kg_m2_s,
    )
    if any(not np.isfinite(value) or value <= 0 for value in response_scales):
        parser.error("Response tolerances and signal scales must be finite and positive.")
    if (
        args.maximum_filter_signal_fraction >= 1
        or args.maximum_response_sensitivity_fraction >= 1
    ):
        parser.error("Filter and sensitivity fractions must be smaller than one.")
    if args.maximum_segments < 1:
        parser.error("maximum-segments must be a positive integer.")
    if args.maximum_segments > 1 and (
        len(args.scenarios) != 1 or args.sensitivity or args.checkpoint_dir is None
    ):
        parser.error(
            "Sequential equilibration requires one scenario, no sensitivity "
            "campaign, and a checkpoint directory."
        )
    if args.resume_checkpoint is not None and (
        len(args.scenarios) != 1 or args.sensitivity
    ):
        parser.error(
            "A native checkpoint resumes exactly one scenario without a parallel "
            "sensitivity campaign."
        )
    if args.smoke:
        args.dt, args.days, args.spinup_days = 20.0, 40.0 / DAY, 0.0
        args.sample_days, args.chunk_samples = 20.0 / DAY, 2
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        bandlimit=args.bandlimit,
        levels=args.levels,
        dt=args.dt,
        filter_days=args.filter_days,
        days=args.days,
        spinup_days=args.spinup_days,
        sample_days=args.sample_days,
        block_days=args.block_days,
        chunk_samples=args.chunk_samples,
        maximum_drift_w_m2=args.maximum_drift_w_m2,
        deterministic_temperature_change_k=(args.deterministic_temperature_change_k),
        deterministic_flux_change_w_m2=(args.deterministic_flux_change_w_m2),
        deterministic_endpoint_flux_w_m2=(args.deterministic_endpoint_flux_w_m2),
        deterministic_eke_change_m2_s2=(args.deterministic_eke_change_m2_s2),
        deterministic_precipitation_change_kg_m2_s=(
            args.deterministic_precipitation_change_kg_m2_s
        ),
        checkpoint_dir=args.checkpoint_dir,
        initialization=(
            "raw" if args.resume_checkpoint is not None else args.initialization
        ),
        resume_checkpoint=args.resume_checkpoint,
    )
    equilibration_segments = []
    if args.maximum_segments == 1:
        runs = {
            scenario: qualify(scenario=scenario, **common) for scenario in args.scenarios
        }
    else:
        scenario = args.scenarios[0]
        segment_common = common
        for _ in range(args.maximum_segments):
            segment = qualify(scenario=scenario, **segment_common)
            equilibration_segments.append(segment)
            if not segment["execution_success"] or (
                segment["stationary_sample_qualified"]
                or segment["deterministic_equilibrium_qualified"]
            ):
                break
            segment_common = {
                **common,
                "initialization": "raw",
                "resume_checkpoint": Path(segment["continuation_checkpoint"]),
                "spinup_days": 0.0,
            }
        runs = {scenario: equilibration_segments[-1]}
    sensitivity = {}
    if args.sensitivity:
        if "baseline" not in runs:
            runs["baseline"] = qualify(scenario="baseline", **common)
        for name, changes in {
            "half_dt": {"dt": args.dt / 2},
            "higher_resolution": {
                "bandlimit": args.bandlimit + 2,
                "levels": args.levels + 2,
            },
            "half_filter_rate": {"filter_days": args.filter_days * 2},
        }.items():
            variants = {
                scenario: qualify(scenario=scenario, **{**common, **changes})
                for scenario in runs
            }
            sensitivity[name] = {"runs": variants}
    response_policy = dict(
        tolerances={
            "sst_k": args.response_temperature_tolerance_k,
            "atmosphere_temperature_k": args.response_temperature_tolerance_k,
            "eddy_kinetic_energy_m2_s2": args.response_eke_tolerance_m2_s2,
            "precipitation_kg_m2_s": args.response_precipitation_tolerance_kg_m2_s,
            "toa_net_in_w_m2": args.response_toa_tolerance_w_m2,
        },
        maximum_response_sensitivity_fraction=args.maximum_response_sensitivity_fraction,
        maximum_filter_work_w_m2=args.maximum_filter_work_w_m2,
        target_signal_w_m2=args.target_signal_w_m2,
        maximum_filter_signal_fraction=args.maximum_filter_signal_fraction,
    )
    responses = (
        {
            scenario: compare(
                runs["baseline"],
                value,
                sensitivity_pairs={
                    name: (case["runs"]["baseline"], case["runs"][scenario])
                    for name, case in sensitivity.items()
                },
                **response_policy,
            )
            for scenario, value in runs.items()
            if scenario != "baseline"
        }
        if "baseline" in runs
        else {}
    )
    result = {
        "mode": "smoke" if args.smoke else "long-run",
        "execution_success": all(run["execution_success"] for run in runs.values())
        and all(
            run["execution_success"]
            for case in sensitivity.values()
            for run in case["runs"].values()
        ),
        "runs": runs,
        "forcing_responses": responses,
        "sensitivity": sensitivity,
        "response_policy": response_policy,
        "equilibration_campaign": {
            "maximum_segments": args.maximum_segments,
            "completed_segments": len(equilibration_segments),
            "stopped_on_stationary_sample": bool(
                equilibration_segments
                and equilibration_segments[-1]["stationary_sample_qualified"]
            ),
            "stopped_on_deterministic_equilibrium": bool(
                equilibration_segments
                and equilibration_segments[-1]["deterministic_equilibrium_qualified"]
            ),
            "initialization": (
                equilibration_segments[0]["initialization"]
                if equilibration_segments
                else None
            ),
            "segment_gates": [
                {
                    "segment_start_days": segment["sampling_policy"][
                        "segment_start_days"
                    ],
                    "segment_end_days": segment["sampling_policy"]["segment_end_days"],
                    "execution_success": segment["execution_success"],
                    "numerical_accounting_adequacy": segment[
                        "numerical_accounting_adequacy"
                    ],
                    "sampling_adequacy": segment["sampling_adequacy"],
                    "statistical_stationarity": segment["statistical_stationarity"],
                    "radiative_surface_equilibrium": segment[
                        "radiative_surface_equilibrium"
                    ],
                    "stationary_sample_qualified": segment["stationary_sample_qualified"],
                    "deterministic_equilibrium_qualified": segment[
                        "deterministic_equilibrium_qualified"
                    ],
                    "segment_total_water_redistribution_fraction": segment[
                        "segment_total_water_redistribution_fraction"
                    ],
                    "lifetime_total_water_redistribution_fraction": segment[
                        "lifetime_total_water_redistribution_fraction"
                    ],
                    "segment_water_phase_repartition_fraction": segment[
                        "segment_water_phase_repartition_fraction"
                    ],
                    "lifetime_water_phase_repartition_fraction": segment[
                        "lifetime_water_phase_repartition_fraction"
                    ],
                    "lifetime_phase_repartition_to_physical_conversion": segment[
                        "lifetime_phase_repartition_to_physical_conversion"
                    ],
                    "maximum_corrected_relative_torque_residual": segment[
                        "maximum_corrected_relative_torque_residual"
                    ],
                    "maximum_raw_relative_torque_residual": segment[
                        "maximum_raw_relative_torque_residual"
                    ],
                    "segment_projection_impulse_fraction": segment[
                        "segment_projection_impulse_fraction"
                    ],
                    "lifetime_projection_impulse_fraction": segment[
                        "lifetime_projection_impulse_fraction"
                    ],
                }
                for segment in equilibration_segments
            ],
            "claim": (
                "Segments stop on the full stationary-sample gate, the separate "
                "deterministic near-fixed-equilibrium gate, or the user-specified "
                "maximum. The deterministic gate never creates independent "
                "samples or response support."
            ),
        }
        if args.maximum_segments > 1
        else None,
        "claim_boundary": (
            "Statistical differences do not imply physical responses. Response "
            "support requires actual paired dt/resolution/filter sensitivity and "
            "declared filter-work budgets; these are empirical tested-configuration "
            "checks, not continuum error bounds."
        ),
    }
    text = json.dumps(json_ready(result), indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["execution_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
