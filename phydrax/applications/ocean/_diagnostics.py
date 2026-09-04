#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._array_archive import write_array_archive
from ..._strict import StrictModule
from ...equations._ksgs import DynamicKSGSPlan
from ._boussinesq import PreparedCartesianBoussinesqOcean
from ._hydrostatic import HydrostaticOceanState, PreparedHydrostaticOcean
from ._hydrostatic_step import (
    HydrostaticContinuationState,
    HydrostaticOceanLedger,
)
from ._step import OceanBoussinesqContinuationState


class OceanDiagnosticView(StrictModule):
    """Named derived ocean fields and coupled semidiscrete evidence."""

    velocity: tuple[Array, ...]
    temperature: Array
    salinity: Array
    density_anomaly: Array
    sgs_kinetic_energy: Array | None
    ksgs_rate: Array | None
    ksgs_eddy_viscosity: Array | None
    ksgs_low_re_dissipation: Array | None
    ksgs_dynamic_coefficient: Array | None
    ksgs_dynamic_updates: Array | None
    ksgs_wall_distance: Array | None
    buoyancy: Array
    pressure: Array
    kinetic_energy: Array
    divergence_norm: Array
    pressure_residual_norm: Array
    temperature_content: Array
    salinity_content: Array
    temperature_sgs_content_rate: Array
    salinity_sgs_content_rate: Array
    temperature_sgs_variance_rate: Array
    salinity_sgs_variance_rate: Array
    coriolis_power: Array
    surface_stress_power: Array
    sgs_dissipation: Array
    molecular_potential_energy_mixing: Array
    sgs_potential_energy_mixing: Array
    boundary_potential_energy_rate: Array
    potential_energy_mixing_available: Array
    buoyancy_exchange_defect: Array
    energy_balance_defect: Array
    successful: Array
    ocean_id: str = eqx.field(static=True)


def ocean_diagnostic_view(
    ocean: PreparedCartesianBoussinesqOcean,
    time: ArrayLike,
    state: OceanBoussinesqContinuationState | ArrayLike,
    args: Any = None,
    /,
) -> OceanDiagnosticView:
    if not isinstance(ocean, PreparedCartesianBoussinesqOcean):
        raise TypeError("ocean must be PreparedCartesianBoussinesqOcean.")
    continuation = state if isinstance(state, OceanBoussinesqContinuationState) else None
    coordinates = (
        continuation.coordinates if continuation is not None else jnp.asarray(state)
    )
    physical = ocean.state_view(state)
    stage = ocean.dynamics.stage(
        time,
        coordinates,
        args,
        ksgs_state=None if continuation is None else continuation.ksgs_state,
    )
    diagnostics = ocean.dynamics.diagnostics_from_stage(stage)
    pressure = stage.pressure
    vertical = ocean.plan.axes.vertical_axis
    gravity = ocean.plan.axes.gravity(ocean.plan.reference.gravity_magnitude)
    buoyancy = (
        gravity[vertical]
        * physical.density_anomaly
        / ocean.plan.reference.reference_density
    )
    temperature = diagnostics.scalars.fields[ocean.plan.reference.temperature_name]
    salinity = diagnostics.scalars.fields[ocean.plan.reference.salinity_name]
    coriolis_power = (
        jnp.asarray(0.0, dtype=coordinates.dtype)
        if stage.ocean_forcing is None
        else stage.ocean_forcing.coriolis_power
    )
    stress_power = (
        jnp.asarray(0.0, dtype=coordinates.dtype)
        if stage.ocean_forcing is None
        else stage.ocean_forcing.surface_stress_power
    )
    return OceanDiagnosticView(
        velocity=physical.velocity,
        temperature=physical.temperature,
        salinity=physical.salinity,
        density_anomaly=physical.density_anomaly,
        sgs_kinetic_energy=physical.sgs_kinetic_energy,
        ksgs_rate=(
            None
            if ocean.plan.ksgs_field_name is None
            else stage.scalar_rates[ocean.plan.ksgs_field_name]
        ),
        ksgs_eddy_viscosity=(
            None if stage.ksgs is None else stage.ksgs.result.eddy_viscosity
        ),
        ksgs_low_re_dissipation=(
            None
            if stage.ksgs is None
            else stage.ksgs.result.contributions.low_re_dissipation
        ),
        ksgs_dynamic_coefficient=(
            stage.ksgs.result.state.eddy_viscosity_coefficient
            if stage.ksgs is not None and isinstance(ocean.plan.ksgs, DynamicKSGSPlan)
            else None
        ),
        ksgs_dynamic_updates=(
            stage.ksgs.result.state.dynamic_updates
            if stage.ksgs is not None and isinstance(ocean.plan.ksgs, DynamicKSGSPlan)
            else None
        ),
        ksgs_wall_distance=(
            None if ocean.prepared_ksgs is None else ocean.prepared_ksgs.wall_distance
        ),
        buoyancy=buoyancy,
        pressure=pressure,
        kinetic_energy=diagnostics.kinetic_energy,
        divergence_norm=diagnostics.divergence_norm,
        pressure_residual_norm=diagnostics.pressure_residual_norm,
        temperature_content=temperature.content,
        salinity_content=salinity.content,
        temperature_sgs_content_rate=temperature.sgs_diffusive_content_rate,
        salinity_sgs_content_rate=salinity.sgs_diffusive_content_rate,
        temperature_sgs_variance_rate=temperature.sgs_diffusive_variance_rate,
        salinity_sgs_variance_rate=salinity.sgs_diffusive_variance_rate,
        coriolis_power=coriolis_power,
        surface_stress_power=stress_power,
        sgs_dissipation=diagnostics.sgs_dissipation,
        molecular_potential_energy_mixing=(
            stage.buoyancy.molecular_potential_energy_mixing
        ),
        sgs_potential_energy_mixing=stage.buoyancy.sgs_potential_energy_mixing,
        boundary_potential_energy_rate=stage.buoyancy.boundary_potential_energy_rate,
        potential_energy_mixing_available=jnp.asarray(
            stage.buoyancy.potential_energy_mixing_available
        ),
        buoyancy_exchange_defect=stage.buoyancy.exchange_defect,
        energy_balance_defect=diagnostics.energy_balance_defect,
        successful=diagnostics.success,
        ocean_id=ocean.prepared_id,
    )


def write_ocean_output(
    path: str | Path,
    ocean: PreparedCartesianBoussinesqOcean,
    time: ArrayLike,
    state: OceanBoussinesqContinuationState | ArrayLike,
    args: Any = None,
    /,
) -> Path:
    view = ocean_diagnostic_view(ocean, time, state, args)
    arrays: dict[str, object] = {
        "time": jnp.asarray(time),
        "temperature": view.temperature,
        "salinity": view.salinity,
        "density_anomaly": view.density_anomaly,
        "buoyancy": view.buoyancy,
        "pressure": view.pressure,
        "kinetic_energy": view.kinetic_energy,
        "divergence_norm": view.divergence_norm,
        "pressure_residual_norm": view.pressure_residual_norm,
        "temperature_content": view.temperature_content,
        "salinity_content": view.salinity_content,
        "temperature_sgs_content_rate": view.temperature_sgs_content_rate,
        "salinity_sgs_content_rate": view.salinity_sgs_content_rate,
        "temperature_sgs_variance_rate": view.temperature_sgs_variance_rate,
        "salinity_sgs_variance_rate": view.salinity_sgs_variance_rate,
        "coriolis_power": view.coriolis_power,
        "surface_stress_power": view.surface_stress_power,
        "sgs_dissipation": view.sgs_dissipation,
        "molecular_potential_energy_mixing": (view.molecular_potential_energy_mixing),
        "sgs_potential_energy_mixing": view.sgs_potential_energy_mixing,
        "boundary_potential_energy_rate": view.boundary_potential_energy_rate,
        "potential_energy_mixing_available": (view.potential_energy_mixing_available),
        "buoyancy_exchange_defect": view.buoyancy_exchange_defect,
        "energy_balance_defect": view.energy_balance_defect,
        "successful": view.successful,
    }
    if view.sgs_kinetic_energy is not None:
        arrays["sgs_kinetic_energy"] = view.sgs_kinetic_energy
    if view.ksgs_rate is not None:
        arrays["ksgs_rate"] = view.ksgs_rate
    if view.ksgs_eddy_viscosity is not None:
        arrays["ksgs_eddy_viscosity"] = view.ksgs_eddy_viscosity
    if view.ksgs_low_re_dissipation is not None:
        arrays["ksgs_low_re_dissipation"] = view.ksgs_low_re_dissipation
    if view.ksgs_dynamic_coefficient is not None:
        arrays["ksgs_dynamic_coefficient"] = view.ksgs_dynamic_coefficient
    if view.ksgs_dynamic_updates is not None:
        arrays["ksgs_dynamic_updates"] = view.ksgs_dynamic_updates
    if view.ksgs_wall_distance is not None:
        arrays["ksgs_wall_distance"] = view.ksgs_wall_distance
    for axis, component in enumerate(view.velocity):
        arrays[f"velocity/{axis}"] = component
    return write_array_archive(
        path,
        manifest={
            "kind": "ocean-boussinesq-output",
            "ocean_id": ocean.prepared_id,
            "field_names": list(ocean.transport.layout.field_names),
        },
        arrays=arrays,
    )


class HydrostaticDiagnosticView(StrictModule):
    eta: Array
    total_depth: Array
    layer_volume: Array
    velocity: tuple[Array, Array]
    tracers: dict[str, Array]
    density: Array
    hydrostatic_pressure: Array
    vertical_flux: Array
    wet_column: Array
    kinetic_energy: Array
    free_surface_energy: Array
    volume: Array
    tracer_content: dict[str, Array]
    ledger: HydrostaticOceanLedger | None
    eos_valid: Array
    eos_finite: Array
    eos_successful: Array
    successful: Array
    ocean_id: str = eqx.field(static=True)


def hydrostatic_diagnostic_view(
    ocean: PreparedHydrostaticOcean,
    state: HydrostaticContinuationState | HydrostaticOceanState,
    /,
) -> HydrostaticDiagnosticView:
    if not isinstance(ocean, PreparedHydrostaticOcean):
        raise TypeError("ocean must be PreparedHydrostaticOcean.")
    physical = state.state if isinstance(state, HydrostaticContinuationState) else state
    ledger = state.ledger if isinstance(state, HydrostaticContinuationState) else None
    epoch = ocean.geometry.metric_epoch(physical.eta)
    view = ocean.view(physical)
    kinetic = (
        0.5
        * ocean.plan.reference_density
        * (
            jnp.sum(
                jnp.where(
                    epoch.x_face_area > 0.0,
                    physical.transports[0] ** 2 / epoch.x_face_area,
                    0.0,
                )
            )
            + jnp.sum(
                jnp.where(
                    epoch.y_face_area > 0.0,
                    physical.transports[1] ** 2 / epoch.y_face_area,
                    0.0,
                )
            )
        )
    )
    free_surface = (
        0.5
        * ocean.plan.reference_density
        * ocean.plan.gravity
        * jnp.sum(ocean.geometry.cell_area * physical.eta**2)
    )
    return HydrostaticDiagnosticView(
        eta=physical.eta,
        total_depth=epoch.total_depth,
        layer_volume=epoch.cell_volume,
        velocity=view.velocity,
        tracers=view.tracers,
        density=view.density,
        hydrostatic_pressure=view.hydrostatic_pressure,
        vertical_flux=view.vertical_flux,
        wet_column=epoch.wet_column,
        kinetic_energy=kinetic,
        free_surface_energy=free_surface,
        volume=jnp.sum(epoch.cell_volume),
        tracer_content={
            name: jnp.sum(value) for name, value in physical.tracer_inventory.items()
        },
        ledger=ledger,
        eos_valid=view.eos_valid,
        eos_finite=view.eos_finite,
        eos_successful=view.eos_successful,
        successful=epoch.valid & view.eos_successful,
        ocean_id=ocean.prepared_id,
    )


def write_hydrostatic_output(
    path: str | Path,
    ocean: PreparedHydrostaticOcean,
    state: HydrostaticContinuationState | HydrostaticOceanState,
    /,
) -> Path:
    view = hydrostatic_diagnostic_view(ocean, state)
    arrays: dict[str, object] = {
        "eta": view.eta,
        "total_depth": view.total_depth,
        "layer_volume": view.layer_volume,
        "density": view.density,
        "hydrostatic_pressure": view.hydrostatic_pressure,
        "vertical_flux": view.vertical_flux,
        "wet_column": view.wet_column,
        "kinetic_energy": view.kinetic_energy,
        "free_surface_energy": view.free_surface_energy,
        "volume": view.volume,
        "eos_valid": view.eos_valid,
        "eos_finite": view.eos_finite,
        "eos_successful": view.eos_successful,
        "successful": view.successful,
    }
    for axis, component in enumerate(view.velocity):
        arrays[f"velocity/{axis}"] = component
    for name, value in view.tracers.items():
        arrays[f"tracer/{name}"] = value
    for name, value in view.tracer_content.items():
        arrays[f"tracer_content/{name}"] = value
    return write_array_archive(
        path,
        manifest={
            "kind": "hydrostatic-ocean-output",
            "ocean_id": ocean.prepared_id,
            "tracer_names": sorted(view.tracers),
        },
        arrays=arrays,
    )


__all__ = [
    "HydrostaticDiagnosticView",
    "OceanDiagnosticView",
    "hydrostatic_diagnostic_view",
    "ocean_diagnostic_view",
    "write_hydrostatic_output",
    "write_ocean_output",
]
