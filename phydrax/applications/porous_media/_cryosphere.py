#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class FreezeThawMaterial(StrictModule):
    freezing_temperature_K: Array
    mushy_width_K: Array
    liquid_heat_capacity_J_kg_K: Array
    ice_heat_capacity_J_kg_K: Array
    latent_heat_J_kg: Array
    reference_temperature_K: Array

    def __init__(
        self,
        *,
        freezing_temperature_K: ArrayLike = 273.15,
        mushy_width_K: ArrayLike = 1.0,
        liquid_heat_capacity_J_kg_K: ArrayLike = 4180.0,
        ice_heat_capacity_J_kg_K: ArrayLike = 2100.0,
        latent_heat_J_kg: ArrayLike = 333700.0,
        reference_temperature_K: ArrayLike = 273.15,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                freezing_temperature_K,
                mushy_width_K,
                liquid_heat_capacity_J_kg_K,
                ice_heat_capacity_J_kg_K,
                latent_heat_J_kg,
                reference_temperature_K,
            )
        )
        invalid = any(jnp.any(~jnp.isfinite(value)) for value in values)
        invalid = (
            invalid
            | jnp.any(values[0] <= 0)
            | jnp.any(values[1] < 0)
            | jnp.any(values[2] <= 0)
            | jnp.any(values[3] <= 0)
            | jnp.any(values[4] <= 0)
            | jnp.any(values[5] <= 0)
        )
        self.freezing_temperature_K = eqx.error_if(
            values[0], invalid, "Freeze-thaw material values must be finite and physical."
        )
        (
            self.mushy_width_K,
            self.liquid_heat_capacity_J_kg_K,
            self.ice_heat_capacity_J_kg_K,
            self.latent_heat_J_kg,
            self.reference_temperature_K,
        ) = values[1:]

    def liquid_fraction(self, temperature_K: ArrayLike, /) -> Array:
        temperature = jnp.asarray(temperature_K)
        half = 0.5 * self.mushy_width_K
        exact = jnp.where(temperature >= self.freezing_temperature_K, 1.0, 0.0)
        smooth = (temperature - (self.freezing_temperature_K - half)) / jnp.where(
            self.mushy_width_K > 0, self.mushy_width_K, 1.0
        )
        return jnp.where(
            self.mushy_width_K > 0, jnp.minimum(jnp.maximum(smooth, 0.0), 1.0), exact
        )

    def specific_enthalpy(self, temperature_K: ArrayLike, /) -> Array:
        temperature = jnp.asarray(temperature_K)
        fraction = self.liquid_fraction(temperature)
        heat_capacity = (
            fraction * self.liquid_heat_capacity_J_kg_K
            + (1.0 - fraction) * self.ice_heat_capacity_J_kg_K
        )
        return (
            heat_capacity * (temperature - self.reference_temperature_K)
            + fraction * self.latent_heat_J_kg
        )

    @property
    def derivative_available(self) -> bool:
        return bool(np.all(np.asarray(self.mushy_width_K) > 0))


class VaporEquilibrium(StrictModule):
    reference_pressure_Pa: Array
    reference_temperature_K: Array
    latent_heat_J_kg: Array
    vapor_gas_constant_J_kg_K: Array

    def __init__(
        self,
        reference_pressure_Pa: ArrayLike = 611.657,
        reference_temperature_K: ArrayLike = 273.16,
        latent_heat_J_kg: ArrayLike = 2.5e6,
        vapor_gas_constant_J_kg_K: ArrayLike = 461.5,
        /,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                reference_pressure_Pa,
                reference_temperature_K,
                latent_heat_J_kg,
                vapor_gas_constant_J_kg_K,
            )
        )
        invalid = any(
            jnp.any(~jnp.isfinite(value)) | jnp.any(value <= 0) for value in values
        )
        self.reference_pressure_Pa = eqx.error_if(
            values[0], invalid, "Vapor-equilibrium constants must be positive and finite."
        )
        (
            self.reference_temperature_K,
            self.latent_heat_J_kg,
            self.vapor_gas_constant_J_kg_K,
        ) = values[1:]

    def saturation_pressure(self, temperature_K: ArrayLike, /) -> Array:
        temperature = jnp.asarray(temperature_K)
        temperature = eqx.error_if(
            temperature,
            jnp.any(~jnp.isfinite(temperature)) | jnp.any(temperature <= 0),
            "Vapor temperature must be positive and finite.",
        )
        exponent = (
            -self.latent_heat_J_kg
            / self.vapor_gas_constant_J_kg_K
            * (1.0 / temperature - 1.0 / self.reference_temperature_K)
        )
        return self.reference_pressure_Pa * jnp.exp(exponent)

    def density(self, temperature_K: ArrayLike, relative_humidity: ArrayLike, /) -> Array:
        temperature, humidity = jnp.broadcast_arrays(
            jnp.asarray(temperature_K), jnp.asarray(relative_humidity)
        )
        humidity = eqx.error_if(
            humidity,
            jnp.any(~jnp.isfinite(humidity)) | jnp.any((humidity < 0) | (humidity > 1)),
            "Relative humidity must lie in [0,1].",
        )
        return (
            humidity
            * self.saturation_pressure(temperature)
            / (self.vapor_gas_constant_J_kg_K * temperature)
        )


class AtmosphericExchangeResult(StrictModule):
    evaporation_kg_s: Array
    sensible_heat_W: Array
    latent_heat_W: Array
    net_radiation_W: Array
    total_energy_W: Array
    limited: Array
    derivative_available: Array


class AtmosphericExchangePlan(StrictModule):
    area_m2: Array
    aerodynamic_conductance_m_s: Array
    air_density_kg_m3: Array
    air_heat_capacity_J_kg_K: Array
    emissivity: Array
    vapor: VaporEquilibrium

    def __init__(
        self,
        area_m2: ArrayLike,
        aerodynamic_conductance_m_s: ArrayLike,
        /,
        *,
        air_density_kg_m3: ArrayLike = 1.225,
        air_heat_capacity_J_kg_K: ArrayLike = 1005.0,
        emissivity: ArrayLike = 0.98,
        vapor: VaporEquilibrium | None = None,
    ):
        area, conductance, density, capacity, emissivity_ = jnp.broadcast_arrays(
            jnp.asarray(area_m2),
            jnp.asarray(aerodynamic_conductance_m_s),
            jnp.asarray(air_density_kg_m3),
            jnp.asarray(air_heat_capacity_J_kg_K),
            jnp.asarray(emissivity),
        )
        invalid = (
            jnp.any(~jnp.isfinite(area))
            | jnp.any(area <= 0)
            | jnp.any(~jnp.isfinite(conductance))
            | jnp.any(conductance < 0)
            | jnp.any(~jnp.isfinite(density))
            | jnp.any(density <= 0)
            | jnp.any(~jnp.isfinite(capacity))
            | jnp.any(capacity <= 0)
            | jnp.any(~jnp.isfinite(emissivity_))
            | jnp.any((emissivity_ <= 0) | (emissivity_ > 1))
        )
        self.area_m2 = eqx.error_if(
            area, invalid, "Atmospheric exchange parameters must be finite and physical."
        )
        self.aerodynamic_conductance_m_s = conductance
        self.air_density_kg_m3, self.air_heat_capacity_J_kg_K = density, capacity
        self.emissivity = emissivity_
        self.vapor = VaporEquilibrium() if vapor is None else vapor

    def evaluate(
        self,
        surface_temperature_K: ArrayLike,
        air_temperature_K: ArrayLike,
        relative_humidity: ArrayLike,
        incoming_shortwave_W_m2: ArrayLike,
        incoming_longwave_W_m2: ArrayLike,
        available_water_kg: ArrayLike,
        dt_s: ArrayLike,
        /,
    ) -> AtmosphericExchangeResult:
        surface, air, shortwave, longwave, available = jnp.broadcast_arrays(
            jnp.asarray(surface_temperature_K),
            jnp.asarray(air_temperature_K),
            jnp.asarray(incoming_shortwave_W_m2),
            jnp.asarray(incoming_longwave_W_m2),
            jnp.asarray(available_water_kg),
        )
        dt = jnp.asarray(dt_s)
        if dt.shape != ():
            raise ValueError("Atmospheric exchange timestep must be scalar.")
        surface = eqx.error_if(
            surface,
            jnp.any(~jnp.isfinite(surface))
            | jnp.any(surface <= 0)
            | jnp.any(~jnp.isfinite(air))
            | jnp.any(air <= 0)
            | jnp.any(~jnp.isfinite(shortwave))
            | jnp.any(~jnp.isfinite(longwave))
            | jnp.any(~jnp.isfinite(available))
            | jnp.any(available < 0)
            | ~jnp.isfinite(dt)
            | (dt <= 0),
            "Atmospheric forcing/state/timestep must be finite and physical.",
        )
        surface_vapor = self.vapor.density(surface, 1.0)
        air_vapor = self.vapor.density(air, relative_humidity)
        demand = (
            self.area_m2 * self.aerodynamic_conductance_m_s * (surface_vapor - air_vapor)
        )
        maximum = available / dt
        evaporation = jnp.minimum(demand, maximum)
        limited = jnp.any(demand > maximum)
        sensible = (
            self.area_m2
            * self.aerodynamic_conductance_m_s
            * self.air_density_kg_m3
            * self.air_heat_capacity_J_kg_K
            * (air - surface)
        )
        latent = -self.vapor.latent_heat_J_kg * evaporation
        stefan_boltzmann = 5.670374419e-8
        radiation = self.area_m2 * (
            shortwave
            + self.emissivity * longwave
            - self.emissivity * stefan_boltzmann * surface**4
        )
        return AtmosphericExchangeResult(
            evaporation,
            sensible,
            latent,
            radiation,
            sensible + latent + radiation,
            limited,
            ~limited,
        )


__all__ = [
    "AtmosphericExchangePlan",
    "AtmosphericExchangeResult",
    "FreezeThawMaterial",
    "VaporEquilibrium",
]
