#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class MPMFieldRole(IntEnum):
    MECHANICAL_VELOCITY = 0
    PORE_PRESSURE = 1
    FLUID_SATURATION = 2
    TEMPERATURE = 3
    DAMAGE = 4
    SPECIES = 5


class MPMPhysicalFieldPlan(StrictModule, NonTrainableState):
    role: MPMFieldRole = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    units: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, role: MPMFieldRole, field_id: str, units: str, /):
        role_ = MPMFieldRole(role)
        identifier = str(field_id)
        units_ = str(units)
        if not identifier or not units_:
            raise ValueError("Physical field ID and units must be non-empty.")
        self.role = role_
        self.field_id = identifier
        self.units = units_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-physical-field",
                "role": int(role_),
                "field_id": identifier,
                "units": units_,
            }
        )


class MPMCoupledFieldState(StrictModule):
    pore_pressure: Array
    saturation: Array
    temperature: Array
    damage: Array
    time: Array


class BiotPoromechanicsParameters(StrictModule, NonTrainableState):
    biot_coefficient: Array
    storage_coefficient: Array
    permeability: Array
    fluid_viscosity: Array

    def __init__(
        self,
        biot_coefficient: ArrayLike,
        storage_coefficient: ArrayLike,
        permeability: ArrayLike,
        fluid_viscosity: ArrayLike,
        /,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                biot_coefficient,
                storage_coefficient,
                permeability,
                fluid_viscosity,
            )
        )
        if any(value.shape != () for value in values) or any(
            not bool(jnp.isfinite(value)) for value in values
        ):
            raise ValueError("Biot parameters must be finite scalars.")
        if not 0.0 <= values[0] <= 1.0 or any(value <= 0.0 for value in values[1:]):
            raise ValueError("Biot parameters are inadmissible.")
        (
            self.biot_coefficient,
            self.storage_coefficient,
            self.permeability,
            self.fluid_viscosity,
        ) = values


class ThermalMPMParameters(StrictModule, NonTrainableState):
    density_heat_capacity: Array
    conductivity: Array
    thermal_expansion: Array
    plastic_heat_fraction: Array
    reference_temperature: Array

    def __init__(
        self,
        density_heat_capacity: ArrayLike,
        conductivity: ArrayLike,
        thermal_expansion: ArrayLike,
        plastic_heat_fraction: ArrayLike,
        reference_temperature: ArrayLike,
        /,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                density_heat_capacity,
                conductivity,
                thermal_expansion,
                plastic_heat_fraction,
                reference_temperature,
            )
        )
        if any(value.shape != () for value in values) or any(
            not bool(jnp.isfinite(value)) for value in values
        ):
            raise ValueError("Thermal MPM parameters must be finite scalars.")
        if (
            values[0] <= 0.0
            or values[1] <= 0.0
            or values[2] < 0.0
            or not 0.0 <= values[3] <= 1.0
        ):
            raise ValueError("Thermal MPM parameters are inadmissible.")
        (
            self.density_heat_capacity,
            self.conductivity,
            self.thermal_expansion,
            self.plastic_heat_fraction,
            self.reference_temperature,
        ) = values


class MPMCoupledBoundaryPlan(StrictModule, NonTrainableState):
    pressure_mask: Array
    pressure_values: Array
    pressure_flux: Array
    temperature_mask: Array
    temperature_values: Array
    heat_flux: Array
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        pressure_mask: ArrayLike,
        pressure_values: ArrayLike,
        pressure_flux: ArrayLike,
        temperature_mask: ArrayLike,
        temperature_values: ArrayLike,
        heat_flux: ArrayLike,
    ):
        pressure_mask_ = np.asarray(pressure_mask, dtype=bool)
        temperature_mask_ = np.asarray(temperature_mask, dtype=bool)
        if pressure_mask_.shape != temperature_mask_.shape:
            raise ValueError("Coupled boundary masks must share grid shape.")
        shape = pressure_mask_.shape
        arrays = tuple(
            np.broadcast_to(np.asarray(value, dtype=float), shape)
            for value in (
                pressure_values,
                pressure_flux,
                temperature_values,
                heat_flux,
            )
        )
        if any(np.any(~np.isfinite(value)) for value in arrays):
            raise ValueError("Coupled boundary values/fluxes must be finite.")
        self.pressure_mask = jnp.asarray(pressure_mask_)
        self.pressure_values = jnp.asarray(arrays[0])
        self.pressure_flux = jnp.asarray(arrays[1])
        self.temperature_mask = jnp.asarray(temperature_mask_)
        self.temperature_values = jnp.asarray(arrays[2])
        self.heat_flux = jnp.asarray(arrays[3])
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "mpm-coupled-boundary",
                "shape": shape,
                "pressure_dirichlet_count": int(np.sum(pressure_mask_)),
                "temperature_dirichlet_count": int(np.sum(temperature_mask_)),
            }
        )


class MPMCoupledResidual(StrictModule):
    pressure: Array
    temperature: Array
    effective_stress_correction: Array
    darcy_flux: Array
    heat_flux: Array
    finite: Array


class MPMCoupledLinearization(StrictModule):
    residual: MPMCoupledResidual
    jvp: MPMCoupledResidual
    transpose: tuple[Array, Array]
    successful: Array


class PreparedMPMCoupledFieldOperator(StrictModule, NonTrainableState):
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    biot: BiotPoromechanicsParameters
    thermal: ThermalMPMParameters
    boundaries: MPMCoupledBoundaryPlan
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid_shape,
        spacing,
        periodic,
        biot: BiotPoromechanicsParameters,
        thermal: ThermalMPMParameters,
        boundaries: MPMCoupledBoundaryPlan,
        /,
    ):
        shape = tuple(int(value) for value in grid_shape)
        spacing_ = tuple(float(value) for value in spacing)
        periodic_ = tuple(bool(value) for value in periodic)
        if (
            not shape
            or len(shape) != len(spacing_)
            or len(shape) != len(periodic_)
            or any(value <= 0 for value in shape)
            or any(not np.isfinite(value) or value <= 0.0 for value in spacing_)
            or boundaries.pressure_mask.shape != shape
        ):
            raise ValueError("Coupled field grid geometry is invalid.")
        self.grid_shape = shape
        self.spacing = spacing_
        self.periodic = periodic_
        self.biot = biot
        self.thermal = thermal
        self.boundaries = boundaries
        self.operator_id = canonical_fingerprint(
            {
                "kind": "prepared-mpm-coupled-fields",
                "grid_shape": shape,
                "spacing": spacing_,
                "periodic": periodic_,
                "boundary": boundaries.boundary_id,
            }
        )

    def _neighbors(self, value, axis):
        if self.periodic[axis]:
            return jnp.roll(value, 1, axis=axis), jnp.roll(value, -1, axis=axis)
        indices = jnp.arange(value.shape[axis])
        lower = jnp.take(value, jnp.maximum(indices - 1, 0), axis=axis)
        upper = jnp.take(
            value, jnp.minimum(indices + 1, value.shape[axis] - 1), axis=axis
        )
        return lower, upper

    def gradient(self, value):
        components = []
        for axis, spacing in enumerate(self.spacing):
            lower, upper = self._neighbors(value, axis)
            components.append((upper - lower) / (2.0 * spacing))
        return jnp.stack(components, axis=-1)

    def laplacian(self, value):
        result = jnp.zeros_like(value)
        for axis, spacing in enumerate(self.spacing):
            lower, upper = self._neighbors(value, axis)
            result = result + (upper - 2.0 * value + lower) / spacing**2
        return result

    def residual(
        self,
        state: MPMCoupledFieldState,
        pressure_rate: ArrayLike,
        temperature_rate: ArrayLike,
        volumetric_strain_rate: ArrayLike,
        plastic_dissipation_rate: ArrayLike,
        /,
    ) -> MPMCoupledResidual:
        pressure = jnp.asarray(state.pore_pressure)
        temperature = jnp.asarray(state.temperature)
        values = tuple(
            jnp.broadcast_to(jnp.asarray(value, dtype=pressure.dtype), self.grid_shape)
            for value in (
                pressure_rate,
                temperature_rate,
                volumetric_strain_rate,
                plastic_dissipation_rate,
            )
        )
        darcy_flux = -(
            self.biot.permeability / self.biot.fluid_viscosity
        ) * self.gradient(pressure)
        pressure_residual = (
            self.biot.storage_coefficient * values[0]
            + self.biot.biot_coefficient * values[2]
            - (self.biot.permeability / self.biot.fluid_viscosity)
            * self.laplacian(pressure)
            - self.boundaries.pressure_flux
        )
        pressure_residual = jnp.where(
            self.boundaries.pressure_mask,
            pressure - self.boundaries.pressure_values,
            pressure_residual,
        )
        heat_flux = -self.thermal.conductivity * self.gradient(temperature)
        temperature_residual = (
            self.thermal.density_heat_capacity * values[1]
            - self.thermal.conductivity * self.laplacian(temperature)
            - self.thermal.plastic_heat_fraction * values[3]
            - self.boundaries.heat_flux
        )
        temperature_residual = jnp.where(
            self.boundaries.temperature_mask,
            temperature - self.boundaries.temperature_values,
            temperature_residual,
        )
        dimension = len(self.grid_shape)
        identity = jnp.eye(dimension, dtype=pressure.dtype)
        effective = (
            -self.biot.biot_coefficient * pressure[..., None, None] * identity
            - 3.0
            * self.thermal.thermal_expansion
            * (temperature - self.thermal.reference_temperature)[..., None, None]
            * identity
        )
        finite = (
            jnp.all(jnp.isfinite(pressure_residual))
            & jnp.all(jnp.isfinite(temperature_residual))
            & jnp.all(jnp.isfinite(effective))
        )
        return MPMCoupledResidual(
            pressure_residual,
            temperature_residual,
            effective,
            darcy_flux,
            heat_flux,
            finite,
        )

    def linearize(
        self,
        state: MPMCoupledFieldState,
        state_direction: MPMCoupledFieldState,
        pressure_rate,
        temperature_rate,
        volumetric_strain_rate,
        plastic_dissipation_rate,
        cotangent: tuple[ArrayLike, ArrayLike],
        /,
    ):
        def function(pressure, temperature):
            current = MPMCoupledFieldState(
                pressure,
                state.saturation,
                temperature,
                state.damage,
                state.time,
            )
            result = self.residual(
                current,
                pressure_rate,
                temperature_rate,
                volumetric_strain_rate,
                plastic_dissipation_rate,
            )
            return result.pressure, result.temperature

        primals = (state.pore_pressure, state.temperature)
        tangents = (
            state_direction.pore_pressure,
            state_direction.temperature,
        )
        residual_values, tangent_values = jax.jvp(function, primals, tangents)
        _, pullback = jax.vjp(function, *primals)
        transpose = pullback(
            (
                jnp.asarray(cotangent[0]),
                jnp.asarray(cotangent[1]),
            )
        )
        full = self.residual(
            state,
            pressure_rate,
            temperature_rate,
            volumetric_strain_rate,
            plastic_dissipation_rate,
        )
        tangent = MPMCoupledResidual(
            tangent_values[0],
            tangent_values[1],
            jnp.zeros_like(full.effective_stress_correction),
            jnp.zeros_like(full.darcy_flux),
            jnp.zeros_like(full.heat_flux),
            jnp.all(jnp.isfinite(tangent_values[0]))
            & jnp.all(jnp.isfinite(tangent_values[1])),
        )
        return MPMCoupledLinearization(
            full,
            tangent,
            transpose,
            full.finite & tangent.finite,
        )


__all__ = [
    "BiotPoromechanicsParameters",
    "MPMCoupledBoundaryPlan",
    "MPMCoupledFieldState",
    "MPMCoupledLinearization",
    "MPMCoupledResidual",
    "MPMFieldRole",
    "MPMPhysicalFieldPlan",
    "PreparedMPMCoupledFieldOperator",
    "ThermalMPMParameters",
]
