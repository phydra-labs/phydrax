#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_species import ChemicalSpeciesSchema
from ._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT


@jax.custom_jvp
def _implicit_mode_temperature(
    temperature: Array,
    target_energy: Array,
    evaluated_energy: Array,
    heat_capacity: Array,
    /,
) -> Array:
    del target_energy, evaluated_energy, heat_capacity
    return temperature


@_implicit_mode_temperature.defjvp
def _implicit_mode_temperature_jvp(primals, tangents):
    temperature, _, _, heat_capacity = primals
    _, target_tangent, evaluated_tangent, _ = tangents
    tangent = (target_tangent - evaluated_tangent) / heat_capacity
    return temperature, tangent


class ThermalModeSpec(StrictModule, NonTrainableState):
    """One harmonic-oscillator energy pool over a static species subset."""

    characteristic_temperatures: Array
    name: str = eqx.field(static=True)
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        characteristic_temperatures: ArrayLike,
        /,
        *,
        minimum_temperature: float = 50.0,
        maximum_temperature: float = 50000.0,
    ):
        name_ = str(name)
        theta = np.asarray(characteristic_temperatures, dtype=float)
        minimum = float(minimum_temperature)
        maximum = float(maximum_temperature)
        if (
            not name_
            or theta.ndim != 1
            or theta.size == 0
            or np.any(~np.isfinite(theta))
            or np.any(theta < 0.0)
            or not np.any(theta > 0.0)
            or not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or not 0.0 < minimum < maximum
        ):
            raise ValueError("Thermal-mode name, temperatures, or bounds are invalid.")
        value = jnp.asarray(theta)
        self.name = name_
        self.characteristic_temperatures = value
        self.minimum_temperature = minimum
        self.maximum_temperature = maximum
        self.spec_id = canonical_fingerprint(
            {
                "kind": "harmonic-oscillator-thermal-mode",
                "name": name_,
                "characteristic_temperatures": array_tree_fingerprint(value),
                "bounds": (minimum, maximum),
            }
        )


class ThermalModeEvaluation(StrictModule):
    temperatures: Array
    species_molar_energies: Array
    species_molar_heat_capacities: Array
    energy_densities: Array
    heat_capacity_densities: Array
    finite: Array
    successful: Array
    schema_id: str = eqx.field(static=True)


class ThermalModeTemperatureResult(StrictModule):
    temperatures: Array
    energy_residual: Array
    heat_capacity_densities: Array
    bracket_margin: Array
    iterations: Array
    finite: Array
    successful: Array
    schema_id: str = eqx.field(static=True)


class ThermalModeSchema(StrictModule, NonTrainableState):
    """Ordered static thermal-mode pools bound to one chemical schema."""

    species: ChemicalSpeciesSchema
    modes: tuple[ThermalModeSpec, ...]
    characteristic_temperatures: Array
    minimum_temperatures: Array
    maximum_temperatures: Array
    mode_names: tuple[str, ...] = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: ChemicalSpeciesSchema,
        modes: Sequence[ThermalModeSpec],
        /,
    ):
        mode_tuple = tuple(modes)
        if (
            not isinstance(species, ChemicalSpeciesSchema)
            or not mode_tuple
            or any(not isinstance(mode, ThermalModeSpec) for mode in mode_tuple)
            or len({mode.name for mode in mode_tuple}) != len(mode_tuple)
            or any(
                mode.characteristic_temperatures.shape != (species.species_count,)
                for mode in mode_tuple
            )
        ):
            raise ValueError("Thermal modes must uniquely match the chemical species.")
        characteristic = jnp.stack(
            tuple(mode.characteristic_temperatures for mode in mode_tuple), axis=0
        )
        minimum = jnp.asarray(tuple(mode.minimum_temperature for mode in mode_tuple))
        maximum = jnp.asarray(tuple(mode.maximum_temperature for mode in mode_tuple))
        self.species = species
        self.modes = mode_tuple
        self.characteristic_temperatures = characteristic
        self.minimum_temperatures = minimum
        self.maximum_temperatures = maximum
        self.mode_names = tuple(mode.name for mode in mode_tuple)
        self.schema_id = canonical_fingerprint(
            {
                "kind": "thermal-mode-schema",
                "species": species.schema_id,
                "modes": tuple(mode.spec_id for mode in mode_tuple),
            }
        )

    @property
    def mode_count(self) -> int:
        return len(self.modes)

    def species_calorics(self, temperatures: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(temperatures)
        if value.ndim < 1 or value.shape[-1] != self.mode_count:
            raise ValueError("Mode temperatures must end in the complete mode axis.")
        theta = self.characteristic_temperatures.astype(value.dtype)
        temperature = value[..., :, None]
        active = theta > 0.0
        safe_theta = jnp.where(active, theta, 1.0)
        x = safe_theta / temperature
        exponential = jnp.exp(-x)
        denominator = jnp.maximum(1.0 - exponential, jnp.finfo(value.dtype).tiny)
        energy_mode_species = (
            UNIVERSAL_GAS_CONSTANT * safe_theta * exponential / denominator
        )
        heat_capacity_mode_species = (
            UNIVERSAL_GAS_CONSTANT * x * x * exponential / (denominator * denominator)
        )
        energy_mode_species = jnp.where(active, energy_mode_species, 0.0)
        heat_capacity_mode_species = jnp.where(active, heat_capacity_mode_species, 0.0)
        return (
            jnp.swapaxes(energy_mode_species, -2, -1),
            jnp.swapaxes(heat_capacity_mode_species, -2, -1),
        )

    def evaluate(
        self,
        species_mass_density: ArrayLike,
        temperatures: ArrayLike,
        /,
    ) -> ThermalModeEvaluation:
        density = jnp.asarray(species_mass_density)
        temperature = jnp.asarray(temperatures, dtype=density.dtype)
        if density.ndim < 1 or density.shape[-1] != self.species.species_count:
            raise ValueError("Species density must end in the complete species axis.")
        if temperature.shape != density.shape[:-1] + (self.mode_count,):
            raise ValueError("Mode temperatures must match the species-density cells.")
        molar_energy, molar_capacity = self.species_calorics(temperature)
        amount_density = density / self.species.molar_masses.astype(density.dtype)
        energy_density = contract(
            "...s,...sm->...m", amount_density, molar_energy, backend="jax"
        )
        capacity_density = contract(
            "...s,...sm->...m", amount_density, molar_capacity, backend="jax"
        )
        within_bounds = jnp.all(
            (temperature >= self.minimum_temperatures)
            & (temperature <= self.maximum_temperatures),
            axis=-1,
        )
        finite = (
            jnp.all(jnp.isfinite(density), axis=-1)
            & jnp.all(jnp.isfinite(temperature), axis=-1)
            & jnp.all(jnp.isfinite(energy_density), axis=-1)
            & jnp.all(jnp.isfinite(capacity_density), axis=-1)
        )
        successful = (
            finite
            & within_bounds
            & jnp.all(density >= 0.0, axis=-1)
            & jnp.all(capacity_density > 0.0, axis=-1)
        )
        return ThermalModeEvaluation(
            temperature,
            molar_energy,
            molar_capacity,
            energy_density,
            capacity_density,
            finite,
            successful,
            self.schema_id,
        )

    def solve_temperatures(
        self,
        species_mass_density: ArrayLike,
        energy_densities: ArrayLike,
        /,
        *,
        maximum_iterations: int = 80,
    ) -> ThermalModeTemperatureResult:
        density = jnp.asarray(species_mass_density)
        target = jnp.asarray(energy_densities, dtype=density.dtype)
        iterations = int(maximum_iterations)
        if (
            density.ndim < 1
            or density.shape[-1] != self.species.species_count
            or target.shape != density.shape[:-1] + (self.mode_count,)
            or iterations <= 0
        ):
            raise ValueError("Mode inversion density, energy, or capacity is invalid.")
        lower = jnp.broadcast_to(
            self.minimum_temperatures.astype(density.dtype), target.shape
        )
        upper = jnp.broadcast_to(
            self.maximum_temperatures.astype(density.dtype), target.shape
        )
        lower_energy = self.evaluate(density, lower).energy_densities
        upper_energy = self.evaluate(density, upper).energy_densities
        bracketed = jnp.all((target >= lower_energy) & (target <= upper_energy), axis=-1)

        def body(_, bounds):
            low, high = bounds
            midpoint = 0.5 * (low + high)
            energy = self.evaluate(density, midpoint).energy_densities
            choose_lower = energy < target
            return (
                jnp.where(choose_lower, midpoint, low),
                jnp.where(choose_lower, high, midpoint),
            )

        lower, upper = jax.lax.fori_loop(0, iterations, body, (lower, upper))
        raw_temperature = 0.5 * (lower + upper)
        evaluated = self.evaluate(density, raw_temperature)
        temperature = _implicit_mode_temperature(
            raw_temperature,
            target,
            evaluated.energy_densities,
            evaluated.heat_capacity_densities,
        )
        final = self.evaluate(density, temperature)
        residual = final.energy_densities - target
        margin = jnp.minimum(
            temperature - self.minimum_temperatures,
            self.maximum_temperatures - temperature,
        )
        scale = jnp.maximum(jnp.max(jnp.abs(target), axis=-1), 1.0)
        finite = (
            final.finite
            & jnp.all(jnp.isfinite(residual), axis=-1)
            & jnp.all(jnp.isfinite(margin), axis=-1)
        )
        successful = (
            finite
            & bracketed
            & final.successful
            & jnp.all(
                jnp.abs(residual)
                <= 128.0 * jnp.finfo(density.dtype).eps * scale[..., None],
                axis=-1,
            )
        )
        return ThermalModeTemperatureResult(
            temperature,
            residual,
            final.heat_capacity_densities,
            margin,
            jnp.asarray(iterations, dtype=jnp.int32),
            finite,
            successful,
            self.schema_id,
        )


__all__ = [
    "ThermalModeEvaluation",
    "ThermalModeSchema",
    "ThermalModeSpec",
    "ThermalModeTemperatureResult",
]
