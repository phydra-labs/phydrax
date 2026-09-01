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
from ._boussinesq import PreparedCartesianBoussinesqOcean
from ._step import OceanBoussinesqContinuationState


class OceanDiagnosticView(StrictModule):
    """Named derived ocean fields and coupled semidiscrete evidence."""

    velocity: tuple[Array, ...]
    temperature: Array
    salinity: Array
    density_anomaly: Array
    buoyancy: Array
    pressure: Array
    kinetic_energy: Array
    divergence_norm: Array
    pressure_residual_norm: Array
    temperature_content: Array
    salinity_content: Array
    coriolis_power: Array
    surface_stress_power: Array
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
    coordinates = (
        state.coordinates
        if isinstance(state, OceanBoussinesqContinuationState)
        else jnp.asarray(state)
    )
    physical = ocean.state_view(coordinates)
    diagnostics = ocean.dynamics.diagnostics(time, coordinates, args)
    stage = ocean.dynamics.stage(time, coordinates, args)
    pressure = ocean.dynamics.pressure_field(time, coordinates, args)
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
        buoyancy=buoyancy,
        pressure=pressure,
        kinetic_energy=diagnostics.kinetic_energy,
        divergence_norm=diagnostics.divergence_norm,
        pressure_residual_norm=diagnostics.pressure_residual_norm,
        temperature_content=temperature.content,
        salinity_content=salinity.content,
        coriolis_power=coriolis_power,
        surface_stress_power=stress_power,
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
        "coriolis_power": view.coriolis_power,
        "surface_stress_power": view.surface_stress_power,
        "buoyancy_exchange_defect": view.buoyancy_exchange_defect,
        "energy_balance_defect": view.energy_balance_defect,
        "successful": view.successful,
    }
    for axis, component in enumerate(view.velocity):
        arrays[f"velocity/{axis}"] = component
    return write_array_archive(
        path,
        manifest={
            "kind": "ocean-boussinesq-output",
            "ocean_id": ocean.prepared_id,
            "field_names": [
                ocean.plan.reference.temperature_name,
                ocean.plan.reference.salinity_name,
            ],
        },
        arrays=arrays,
    )


__all__ = ["OceanDiagnosticView", "ocean_diagnostic_view", "write_ocean_output"]
