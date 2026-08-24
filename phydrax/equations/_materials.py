#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class AbstractThermodynamicMaterial(StrictModule, NonTrainableState):
    """Caloric and thermal closure independent of a conservation layout."""

    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def pressure(self, density: Array, specific_internal_energy: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def specific_internal_energy(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def temperature(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def sound_speed(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def specific_enthalpy(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def specific_heat_cp(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def admissible(self, density: Array, pressure: Array, /) -> Array:
        raise NotImplementedError


class IdealGasMaterial(AbstractThermodynamicMaterial):
    """Calorically perfect ideal gas."""

    gamma: float = eqx.field(static=True)
    gas_constant: float = eqx.field(static=True)

    def __init__(
        self,
        gamma: float = 1.4,
        gas_constant: float = 1.0,
        /,
        *,
        density_floor: float = 1e-12,
        pressure_floor: float = 1e-12,
    ):
        gamma_ = float(gamma)
        gas_constant_ = float(gas_constant)
        density_floor_ = float(density_floor)
        pressure_floor_ = float(pressure_floor)
        if (
            not np.isfinite(gamma_)
            or gamma_ <= 1.0
            or not np.isfinite(gas_constant_)
            or gas_constant_ <= 0.0
            or density_floor_ <= 0.0
            or pressure_floor_ <= 0.0
        ):
            raise ValueError(
                "Ideal-gas parameters and floors must be finite and positive."
            )
        self.gamma = gamma_
        self.gas_constant = gas_constant_
        self.density_floor = density_floor_
        self.pressure_floor = pressure_floor_
        self.material_id = canonical_fingerprint(
            {
                "kind": "ideal-gas-material",
                "gamma": gamma_,
                "gas_constant": gas_constant_,
                "density_floor": density_floor_,
                "pressure_floor": pressure_floor_,
            }
        )

    def pressure(self, density: Array, specific_internal_energy: Array, /) -> Array:
        return (self.gamma - 1.0) * density * specific_internal_energy

    def specific_internal_energy(self, density: Array, pressure: Array, /) -> Array:
        return pressure / ((self.gamma - 1.0) * density)

    def temperature(self, density: Array, pressure: Array, /) -> Array:
        return pressure / (density * self.gas_constant)

    def sound_speed(self, density: Array, pressure: Array, /) -> Array:
        return jnp.sqrt(self.gamma * pressure / density)

    def specific_enthalpy(self, density: Array, pressure: Array, /) -> Array:
        return self.gamma * pressure / ((self.gamma - 1.0) * density)

    def specific_heat_cp(self, density: Array, pressure: Array, /) -> Array:
        del pressure
        return jnp.full_like(
            density,
            self.gamma * self.gas_constant / (self.gamma - 1.0),
        )

    def admissible(self, density: Array, pressure: Array, /) -> Array:
        return (density >= self.density_floor) & (pressure >= self.pressure_floor)


class StiffenedGasMaterial(AbstractThermodynamicMaterial):
    """Calorically perfect stiffened-gas closure."""

    gamma: float = eqx.field(static=True)
    pressure_offset: float = eqx.field(static=True)
    reference_energy: float = eqx.field(static=True)
    heat_capacity: float = eqx.field(static=True)

    def __init__(
        self,
        gamma: float,
        pressure_offset: float,
        heat_capacity: float,
        /,
        *,
        reference_energy: float = 0.0,
        density_floor: float = 1e-12,
        pressure_floor: float = 1e-12,
    ):
        values = tuple(
            float(value)
            for value in (
                gamma,
                pressure_offset,
                heat_capacity,
                reference_energy,
                density_floor,
                pressure_floor,
            )
        )
        gamma_, offset, capacity, reference, density_floor_, pressure_floor_ = values
        if (
            any(not np.isfinite(value) for value in values)
            or gamma_ <= 1.0
            or capacity <= 0.0
            or density_floor_ <= 0.0
            or pressure_floor_ + offset <= 0.0
        ):
            raise ValueError("Stiffened-gas parameters are invalid.")
        self.gamma = gamma_
        self.pressure_offset = offset
        self.heat_capacity = capacity
        self.reference_energy = reference
        self.density_floor = density_floor_
        self.pressure_floor = pressure_floor_
        self.material_id = canonical_fingerprint(
            {
                "kind": "stiffened-gas-material",
                "gamma": gamma_,
                "pressure_offset": offset,
                "heat_capacity": capacity,
                "reference_energy": reference,
                "density_floor": density_floor_,
                "pressure_floor": pressure_floor_,
            }
        )

    def pressure(self, density: Array, specific_internal_energy: Array, /) -> Array:
        return (self.gamma - 1.0) * density * (
            specific_internal_energy - self.reference_energy
        ) - self.gamma * self.pressure_offset

    def specific_internal_energy(self, density: Array, pressure: Array, /) -> Array:
        return self.reference_energy + (pressure + self.gamma * self.pressure_offset) / (
            (self.gamma - 1.0) * density
        )

    def temperature(self, density: Array, pressure: Array, /) -> Array:
        return (
            self.specific_internal_energy(density, pressure) - self.reference_energy
        ) / self.heat_capacity

    def sound_speed(self, density: Array, pressure: Array, /) -> Array:
        return jnp.sqrt(self.gamma * (pressure + self.pressure_offset) / density)

    def specific_enthalpy(self, density: Array, pressure: Array, /) -> Array:
        return self.specific_internal_energy(density, pressure) + pressure / density

    def specific_heat_cp(self, density: Array, pressure: Array, /) -> Array:
        del pressure
        return jnp.full_like(density, self.gamma * self.heat_capacity)

    def admissible(self, density: Array, pressure: Array, /) -> Array:
        return (density >= self.density_floor) & (pressure + self.pressure_offset > 0.0)


__all__ = [
    "AbstractThermodynamicMaterial",
    "IdealGasMaterial",
    "StiffenedGasMaterial",
]
