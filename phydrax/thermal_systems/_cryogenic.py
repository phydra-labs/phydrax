#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rigid cryogenic tank boil-off and pressure-limited venting."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


@dataclass(frozen=True, slots=True)
class CryogenicTankState:
    liquid_mass_kg: Array
    vapor_mass_kg: Array
    temperature_k: Array


@dataclass(frozen=True, slots=True)
class CryogenicTankStep:
    state: CryogenicTankState
    pressure_pa: Array
    vaporized_mass_kg: Array
    vented_mass_kg: Array
    mass_balance_residual_kg: Array
    energy_balance_residual_j: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class CryogenicTankModel:
    tank_volume_m3: float
    liquid_density_kg_m3: float
    saturation_temperature_k: float
    latent_heat_j_kg: float
    liquid_heat_capacity_j_kg_k: float
    vapor_heat_capacity_j_kg_k: float
    vapor_gas_constant_j_kg_k: float
    vent_set_pressure_pa: float

    def __post_init__(self):
        values = (
            self.tank_volume_m3,
            self.liquid_density_kg_m3,
            self.saturation_temperature_k,
            self.latent_heat_j_kg,
            self.liquid_heat_capacity_j_kg_k,
            self.vapor_heat_capacity_j_kg_k,
            self.vapor_gas_constant_j_kg_k,
            self.vent_set_pressure_pa,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Cryogenic tank parameters must be finite and positive.")

    def _volume(self, liquid_mass_kg: Array) -> Array:
        volume = self.tank_volume_m3 - liquid_mass_kg / self.liquid_density_kg_m3
        if bool(volume <= 0):
            raise ValueError("Cryogenic liquid mass fills or exceeds tank volume.")
        return volume

    def pressure(self, state: CryogenicTankState, /) -> Array:
        return (
            state.vapor_mass_kg
            * self.vapor_gas_constant_j_kg_k
            * state.temperature_k
            / self._volume(state.liquid_mass_kg)
        )

    def _energy(self, state: CryogenicTankState, /) -> Array:
        temperature_offset = state.temperature_k - self.saturation_temperature_k
        return (
            state.liquid_mass_kg * self.liquid_heat_capacity_j_kg_k * temperature_offset
            + state.vapor_mass_kg
            * (
                self.latent_heat_j_kg
                + self.vapor_heat_capacity_j_kg_k * temperature_offset
            )
        )

    def advance(
        self,
        state: CryogenicTankState,
        heat_leak_w: float,
        step_size_s: float,
        /,
    ) -> CryogenicTankStep:
        liquid = jnp.asarray(state.liquid_mass_kg)
        vapor = jnp.asarray(state.vapor_mass_kg)
        temperature = jnp.asarray(state.temperature_k)
        if any(value.shape != () for value in (liquid, vapor, temperature)):
            raise ValueError("Cryogenic tank state variables must be scalar.")
        if bool((liquid < 0) | (vapor < 0) | (liquid + vapor <= 0) | (temperature <= 0)):
            raise ValueError("Cryogenic tank state is inadmissible.")
        if bool((liquid > 0) & (temperature > self.saturation_temperature_k + 1e-10)):
            raise ValueError(
                "Equilibrium cryogenic liquid cannot exceed saturation temperature."
            )
        if heat_leak_w < 0 or step_size_s <= 0:
            raise ValueError(
                "Cryogenic boil-off requires non-negative heat and positive time."
            )
        initial_mass = liquid + vapor
        initial_energy = self._energy(state)
        heat = jnp.asarray(float(heat_leak_w) * float(step_size_s))

        total_capacity = (
            liquid * self.liquid_heat_capacity_j_kg_k
            + vapor * self.vapor_heat_capacity_j_kg_k
        )
        warming_need = (
            jnp.maximum(self.saturation_temperature_k - temperature, 0) * total_capacity
        )
        warming = jnp.minimum(heat, warming_need)
        temperature = temperature + jnp.where(
            total_capacity > 0, warming / total_capacity, 0
        )
        remaining = heat - warming

        vaporized = jnp.minimum(liquid, remaining / self.latent_heat_j_kg)
        liquid = liquid - vaporized
        vapor = vapor + vaporized
        remaining = remaining - vaporized * self.latent_heat_j_kg
        temperature = jnp.where(
            (liquid <= 0) & (vapor > 0),
            temperature + remaining / (vapor * self.vapor_heat_capacity_j_kg_k),
            temperature,
        )

        vapor_volume = self._volume(liquid)
        maximum_vapor_mass = (
            self.vent_set_pressure_pa
            * vapor_volume
            / (self.vapor_gas_constant_j_kg_k * temperature)
        )
        vented = jnp.maximum(vapor - maximum_vapor_mass, 0)
        vapor = vapor - vented
        next_state = CryogenicTankState(liquid, vapor, temperature)
        pressure = self.pressure(next_state)
        vent_specific_energy = self.latent_heat_j_kg + self.vapor_heat_capacity_j_kg_k * (
            temperature - self.saturation_temperature_k
        )
        mass_residual = liquid + vapor + vented - initial_mass
        energy_residual = (
            self._energy(next_state)
            + vented * vent_specific_energy
            - initial_energy
            - heat
        )
        successful = (
            jnp.all(jnp.isfinite(jnp.asarray((liquid, vapor, temperature, pressure))))
            & (liquid >= 0)
            & (vapor >= 0)
            & (pressure <= self.vent_set_pressure_pa * (1 + 1e-10))
        )
        return CryogenicTankStep(
            next_state,
            pressure,
            vaporized,
            vented,
            mass_residual,
            energy_residual,
            successful,
        )


__all__ = ["CryogenicTankModel", "CryogenicTankState", "CryogenicTankStep"]
