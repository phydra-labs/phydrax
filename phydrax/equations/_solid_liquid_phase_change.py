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

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..nonlinear import LocalRootPlan


class SolidLiquidPhaseStatus(IntEnum):
    SOLID = 0
    PHASE_CHANGE = 1
    LIQUID = 2


class SolidLiquidEnthalpyState(StrictModule):
    temperature: Array
    liquid_fraction: Array
    volumetric_heat_capacity: Array
    conductivity: Array
    viscosity: Array
    mushy_resistance: Array
    temperature_enthalpy_derivative: Array
    phase_status: Array
    solidus_margin: Array
    liquidus_margin: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class SolidLiquidEnthalpyPlan(StrictModule, NonTrainableState):
    reference_density: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    solidus_temperature: float = eqx.field(static=True)
    liquidus_temperature: float = eqx.field(static=True)
    solid_heat_capacity: float = eqx.field(static=True)
    liquid_heat_capacity: float = eqx.field(static=True)
    latent_heat: float = eqx.field(static=True)
    solid_conductivity: float = eqx.field(static=True)
    liquid_conductivity: float = eqx.field(static=True)
    liquid_kinematic_viscosity: float = eqx.field(static=True)
    thermal_expansion: float = eqx.field(static=True)
    buoyancy_reference_temperature: float = eqx.field(static=True)
    mushy_resistance_coefficient: float = eqx.field(static=True)
    mushy_regularization: float = eqx.field(static=True)
    solidus_enthalpy: float = eqx.field(static=True)
    liquidus_enthalpy: float = eqx.field(static=True)
    mushy_volumetric_capacity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_density,
        reference_temperature,
        solidus_temperature,
        liquidus_temperature,
        solid_heat_capacity,
        liquid_heat_capacity,
        latent_heat,
        solid_conductivity,
        liquid_conductivity,
        liquid_kinematic_viscosity,
        /,
        *,
        thermal_expansion=0.0,
        buoyancy_reference_temperature=None,
        mushy_resistance_coefficient=1.0e6,
        mushy_regularization=1.0e-3,
    ):
        values = tuple(
            float(value)
            for value in (
                reference_density,
                reference_temperature,
                solidus_temperature,
                liquidus_temperature,
                solid_heat_capacity,
                liquid_heat_capacity,
                latent_heat,
                solid_conductivity,
                liquid_conductivity,
                liquid_kinematic_viscosity,
                thermal_expansion,
                mushy_resistance_coefficient,
                mushy_regularization,
            )
        )
        (
            density,
            reference,
            solidus,
            liquidus,
            solid_cp,
            liquid_cp,
            latent,
            solid_k,
            liquid_k,
            viscosity,
            expansion,
            resistance,
            regularization,
        ) = values
        buoyancy_reference = (
            reference
            if buoyancy_reference_temperature is None
            else float(buoyancy_reference_temperature)
        )
        if (
            any(not np.isfinite(value) for value in values)
            or not np.isfinite(buoyancy_reference)
            or density <= 0.0
            or solidus > liquidus
            or solid_cp <= 0.0
            or liquid_cp <= 0.0
            or latent <= 0.0
            or solid_k <= 0.0
            or liquid_k <= 0.0
            or viscosity <= 0.0
            or expansion < 0.0
            or resistance < 0.0
            or regularization <= 0.0
        ):
            raise ValueError("Solid-liquid enthalpy parameters are invalid.")
        width = liquidus - solidus
        sensible_mushy_capacity = density * 0.5 * (solid_cp + liquid_cp)
        latent_mushy_capacity = 0.0 if width == 0.0 else density * latent / width
        mushy_capacity = sensible_mushy_capacity + latent_mushy_capacity
        solidus_enthalpy = density * solid_cp * (solidus - reference)
        liquidus_enthalpy = (
            solidus_enthalpy + density * latent
            if width == 0.0
            else solidus_enthalpy + mushy_capacity * width
        )
        self.reference_density = density
        self.reference_temperature = reference
        self.solidus_temperature = solidus
        self.liquidus_temperature = liquidus
        self.solid_heat_capacity = solid_cp
        self.liquid_heat_capacity = liquid_cp
        self.latent_heat = latent
        self.solid_conductivity = solid_k
        self.liquid_conductivity = liquid_k
        self.liquid_kinematic_viscosity = viscosity
        self.thermal_expansion = expansion
        self.buoyancy_reference_temperature = buoyancy_reference
        self.mushy_resistance_coefficient = resistance
        self.mushy_regularization = regularization
        self.solidus_enthalpy = solidus_enthalpy
        self.liquidus_enthalpy = liquidus_enthalpy
        self.mushy_volumetric_capacity = mushy_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "solid-liquid-enthalpy",
                "values": values,
                "buoyancy_reference_temperature": buoyancy_reference,
            }
        )

    @property
    def is_isothermal_transition(self) -> bool:
        return self.solidus_temperature == self.liquidus_temperature

    def enthalpy_from_temperature(
        self,
        temperature: ArrayLike,
        /,
        *,
        liquid_fraction: ArrayLike | None = None,
    ) -> Array:
        temperature_ = jnp.asarray(temperature)
        solid = (
            self.reference_density
            * self.solid_heat_capacity
            * (temperature_ - self.reference_temperature)
        )
        liquid = self.liquidus_enthalpy + (
            self.reference_density
            * self.liquid_heat_capacity
            * (temperature_ - self.liquidus_temperature)
        )
        if self.is_isothermal_transition:
            fraction = (
                jnp.where(temperature_ > self.solidus_temperature, 1.0, 0.0)
                if liquid_fraction is None
                else jnp.asarray(liquid_fraction)
            )
            phase_change = self.solidus_enthalpy + (
                self.reference_density * self.latent_heat * fraction
            )
            return jnp.where(
                temperature_ < self.solidus_temperature,
                solid,
                jnp.where(temperature_ > self.liquidus_temperature, liquid, phase_change),
            )
        fraction = jnp.clip(
            (temperature_ - self.solidus_temperature)
            / (self.liquidus_temperature - self.solidus_temperature),
            0.0,
            1.0,
        )
        mushy = self.solidus_enthalpy + self.mushy_volumetric_capacity * (
            temperature_ - self.solidus_temperature
        )
        return jnp.where(
            fraction <= 0.0,
            solid,
            jnp.where(fraction >= 1.0, liquid, mushy),
        )

    def evaluate(self, enthalpy: ArrayLike, /) -> SolidLiquidEnthalpyState:
        enthalpy_ = jnp.asarray(enthalpy)
        solid = enthalpy_ < self.solidus_enthalpy
        liquid = enthalpy_ > self.liquidus_enthalpy
        solid_temperature = self.reference_temperature + enthalpy_ / (
            self.reference_density * self.solid_heat_capacity
        )
        liquid_temperature = self.liquidus_temperature + (
            enthalpy_ - self.liquidus_enthalpy
        ) / (self.reference_density * self.liquid_heat_capacity)
        if self.is_isothermal_transition:
            changing_temperature = jnp.full_like(enthalpy_, self.solidus_temperature)
            changing_fraction = (enthalpy_ - self.solidus_enthalpy) / (
                self.reference_density * self.latent_heat
            )
            changing_derivative = jnp.zeros_like(enthalpy_)
        else:
            changing_temperature = (
                self.solidus_temperature
                + (enthalpy_ - self.solidus_enthalpy) / self.mushy_volumetric_capacity
            )
            changing_fraction = (changing_temperature - self.solidus_temperature) / (
                self.liquidus_temperature - self.solidus_temperature
            )
            changing_derivative = jnp.full_like(
                enthalpy_, 1.0 / self.mushy_volumetric_capacity
            )
        temperature = jnp.where(
            solid,
            solid_temperature,
            jnp.where(liquid, liquid_temperature, changing_temperature),
        )
        fraction = jnp.where(solid, 0.0, jnp.where(liquid, 1.0, changing_fraction))
        fraction = jnp.clip(fraction, 0.0, 1.0)
        heat_capacity = self.reference_density * (
            (1.0 - fraction) * self.solid_heat_capacity
            + fraction * self.liquid_heat_capacity
        )
        conductivity = (
            1.0 - fraction
        ) * self.solid_conductivity + fraction * self.liquid_conductivity
        viscosity = jnp.full_like(enthalpy_, self.liquid_kinematic_viscosity)
        resistance = (
            self.mushy_resistance_coefficient
            * ((1.0 - fraction) ** 2)
            / (fraction**3 + self.mushy_regularization)
        )
        derivative = jnp.where(
            solid,
            1.0 / (self.reference_density * self.solid_heat_capacity),
            jnp.where(
                liquid,
                1.0 / (self.reference_density * self.liquid_heat_capacity),
                changing_derivative,
            ),
        )
        status = jnp.where(
            solid,
            int(SolidLiquidPhaseStatus.SOLID),
            jnp.where(
                liquid,
                int(SolidLiquidPhaseStatus.LIQUID),
                int(SolidLiquidPhaseStatus.PHASE_CHANGE),
            ),
        ).astype(jnp.int32)
        finite = (
            jnp.isfinite(enthalpy_)
            & jnp.isfinite(temperature)
            & jnp.isfinite(fraction)
            & jnp.isfinite(conductivity)
            & jnp.isfinite(resistance)
        )
        return SolidLiquidEnthalpyState(
            temperature,
            fraction,
            heat_capacity,
            conductivity,
            viscosity,
            resistance,
            derivative,
            status,
            temperature - self.solidus_temperature,
            self.liquidus_temperature - temperature,
            finite,
            finite,
            self.plan_id,
        )

    def buoyancy_anomaly(self, temperature: ArrayLike, /) -> Array:
        return self.thermal_expansion * (
            jnp.asarray(temperature) - self.buoyancy_reference_temperature
        )


class BinaryAlloyEnthalpyState(StrictModule):
    temperature: Array
    liquid_fraction: Array
    liquid_concentration: Array
    solid_concentration: Array
    conductivity: Array
    viscosity: Array
    mushy_resistance: Array
    solute_diffusivity: Array
    temperature_enthalpy_derivative: Array
    liquidus_temperature: Array
    equilibrium_margin: Array
    residual: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class BinaryAlloyPhaseDiagramPlan(StrictModule, NonTrainableState):
    reference_density: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    melting_temperature: float = eqx.field(static=True)
    liquidus_slope: float = eqx.field(static=True)
    partition_coefficient: float = eqx.field(static=True)
    smoothing_width: float = eqx.field(static=True)
    heat_capacity: float = eqx.field(static=True)
    latent_heat: float = eqx.field(static=True)
    solid_conductivity: float = eqx.field(static=True)
    liquid_conductivity: float = eqx.field(static=True)
    liquid_kinematic_viscosity: float = eqx.field(static=True)
    solid_solute_diffusivity: float = eqx.field(static=True)
    liquid_solute_diffusivity: float = eqx.field(static=True)
    mushy_resistance_coefficient: float = eqx.field(static=True)
    mushy_regularization: float = eqx.field(static=True)
    root: LocalRootPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_density,
        reference_temperature,
        melting_temperature,
        liquidus_slope,
        partition_coefficient,
        smoothing_width,
        heat_capacity,
        latent_heat,
        solid_conductivity,
        liquid_conductivity,
        liquid_kinematic_viscosity,
        solid_solute_diffusivity,
        liquid_solute_diffusivity,
        /,
        *,
        mushy_resistance_coefficient=1.0e6,
        mushy_regularization=1.0e-3,
        maximum_root_steps=30,
        root_tolerance=1.0e-9,
    ):
        values = tuple(
            float(value)
            for value in (
                reference_density,
                reference_temperature,
                melting_temperature,
                liquidus_slope,
                partition_coefficient,
                smoothing_width,
                heat_capacity,
                latent_heat,
                solid_conductivity,
                liquid_conductivity,
                liquid_kinematic_viscosity,
                solid_solute_diffusivity,
                liquid_solute_diffusivity,
                mushy_resistance_coefficient,
                mushy_regularization,
            )
        )
        (
            density,
            reference,
            melting,
            slope,
            partition,
            width,
            capacity,
            latent,
            solid_k,
            liquid_k,
            viscosity,
            solid_diffusivity,
            liquid_diffusivity,
            resistance,
            regularization,
        ) = values
        if (
            any(not np.isfinite(value) for value in values)
            or density <= 0.0
            or partition <= 0.0
            or partition > 1.0
            or width <= 0.0
            or capacity <= 0.0
            or latent <= 0.0
            or solid_k <= 0.0
            or liquid_k <= 0.0
            or viscosity <= 0.0
            or solid_diffusivity < 0.0
            or liquid_diffusivity <= 0.0
            or resistance < 0.0
            or regularization <= 0.0
        ):
            raise ValueError("Binary-alloy phase-diagram parameters are invalid.")
        identifier = canonical_fingerprint(
            {"kind": "binary-alloy-phase-diagram", "values": values}
        )
        self.reference_density = density
        self.reference_temperature = reference
        self.melting_temperature = melting
        self.liquidus_slope = slope
        self.partition_coefficient = partition
        self.smoothing_width = width
        self.heat_capacity = capacity
        self.latent_heat = latent
        self.solid_conductivity = solid_k
        self.liquid_conductivity = liquid_k
        self.liquid_kinematic_viscosity = viscosity
        self.solid_solute_diffusivity = solid_diffusivity
        self.liquid_solute_diffusivity = liquid_diffusivity
        self.mushy_resistance_coefficient = resistance
        self.mushy_regularization = regularization
        self.root = LocalRootPlan(
            maximum_steps=maximum_root_steps,
            tolerance=root_tolerance,
            plan_id=f"{identifier}/temperature",
        )
        self.plan_id = identifier

    def _liquid_fraction(self, temperature: Array, concentration: Array, /) -> Array:
        liquidus = self.melting_temperature + self.liquidus_slope * concentration
        return 0.5 * (1.0 + jnp.tanh((temperature - liquidus) / self.smoothing_width))

    def evaluate(
        self, enthalpy: ArrayLike, total_concentration: ArrayLike, /
    ) -> BinaryAlloyEnthalpyState:
        enthalpy_ = jnp.asarray(enthalpy)
        concentration = jnp.asarray(total_concentration, dtype=enthalpy_.dtype)
        if concentration.shape != enthalpy_.shape:
            raise ValueError("Binary-alloy enthalpy and concentration shapes must match.")
        flat_h = enthalpy_.reshape((-1,))
        flat_c = concentration.reshape((-1,))

        def solve_one(target_enthalpy, composition):
            def residual(temperature):
                fraction = self._liquid_fraction(temperature, composition)
                return (
                    self.reference_density
                    * self.heat_capacity
                    * (temperature - self.reference_temperature)
                    + self.reference_density * self.latent_heat * fraction
                    - target_enthalpy
                )

            initial = self.melting_temperature + self.liquidus_slope * composition
            root, diagnostics = self.root.solve_with_diagnostics(residual, initial)
            return (
                root,
                diagnostics.residual,
                diagnostics.derivative,
                diagnostics.finite,
                diagnostics.converged,
            )

        temperature, residual, derivative, root_finite, converged = jax.vmap(solve_one)(
            flat_h, flat_c
        )
        temperature = temperature.reshape(enthalpy_.shape)
        residual = residual.reshape(enthalpy_.shape)
        derivative = derivative.reshape(enthalpy_.shape)
        root_finite = root_finite.reshape(enthalpy_.shape)
        converged = converged.reshape(enthalpy_.shape)
        liquidus = self.melting_temperature + self.liquidus_slope * concentration
        fraction = self._liquid_fraction(temperature, concentration)
        denominator = fraction + self.partition_coefficient * (1.0 - fraction)
        liquid_concentration = concentration / jnp.maximum(
            denominator, jnp.finfo(enthalpy_.dtype).tiny
        )
        solid_concentration = self.partition_coefficient * liquid_concentration
        conductivity = (
            1.0 - fraction
        ) * self.solid_conductivity + fraction * self.liquid_conductivity
        viscosity = jnp.full_like(enthalpy_, self.liquid_kinematic_viscosity)
        resistance = (
            self.mushy_resistance_coefficient
            * ((1.0 - fraction) ** 2)
            / (fraction**3 + self.mushy_regularization)
        )
        diffusivity = (
            1.0 - fraction
        ) * self.solid_solute_diffusivity + fraction * self.liquid_solute_diffusivity
        finite = (
            root_finite
            & jnp.isfinite(concentration)
            & jnp.isfinite(fraction)
            & jnp.isfinite(liquid_concentration)
            & jnp.isfinite(conductivity)
            & jnp.isfinite(resistance)
            & jnp.isfinite(diffusivity)
            & jnp.isfinite(derivative)
            & (derivative > 0.0)
        )
        successful = finite & converged & (concentration >= 0.0) & (concentration <= 1.0)
        return BinaryAlloyEnthalpyState(
            temperature,
            fraction,
            liquid_concentration,
            solid_concentration,
            conductivity,
            viscosity,
            resistance,
            diffusivity,
            1.0 / derivative,
            liquidus,
            temperature - liquidus,
            residual,
            finite,
            successful,
            self.plan_id,
        )
