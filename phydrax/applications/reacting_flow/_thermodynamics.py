#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from ...equations._chemical_thermodynamics import (
    AbstractSpeciesThermodynamicsPlan,
    SpeciesThermodynamicEvaluation,
    UNIVERSAL_GAS_CONSTANT,
)


class IdealMixtureThermodynamicState(StrictModule):
    """Caloric and ideal-gas state of one or more reacting-gas cells."""

    temperature: Array
    pressure: Array
    density: Array
    mass_fractions: Array
    mole_fractions: Array
    mixture_molar_mass: Array
    gas_constant: Array
    specific_heat_capacity_pressure: Array
    specific_heat_capacity_volume: Array
    specific_enthalpy: Array
    specific_internal_energy: Array
    speed_of_sound: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class TemperatureInversionEvidence(StrictModule):
    temperature: Array
    energy_residual: Array
    bracket_width: Array
    iterations: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class _FormationAdjustedSpeciesThermodynamicsPlan(AbstractSpeciesThermodynamicsPlan):
    base: AbstractSpeciesThermodynamicsPlan
    formation_molar_enthalpies: Array

    def __init__(
        self,
        base: AbstractSpeciesThermodynamicsPlan,
        formation_molar_enthalpies: Array,
        /,
    ):
        self.base = base
        self.schema = base.schema
        self.formation_molar_enthalpies = formation_molar_enthalpies
        self.minimum_temperature = base.minimum_temperature
        self.maximum_temperature = base.maximum_temperature
        self.thermodynamics_id = canonical_fingerprint(
            {
                "kind": "formation-adjusted-species-thermodynamics",
                "base": base.thermodynamics_id,
                "formation_enthalpies": array_tree_fingerprint(
                    np.asarray(formation_molar_enthalpies)
                ),
            }
        )

    def evaluate(self, temperature: ArrayLike, /) -> SpeciesThermodynamicEvaluation:
        base = self.base.evaluate(temperature)
        offset = self.formation_molar_enthalpies
        return SpeciesThermodynamicEvaluation(
            base.molar_heat_capacity_pressure,
            base.molar_heat_capacity_volume,
            base.molar_enthalpy + offset,
            base.molar_internal_energy + offset,
            base.molar_entropy,
            base.molar_gibbs_energy + offset,
            base.active_interval,
            base.temperature_margin,
            base.successful,
            self.thermodynamics_id,
        )


class ReactingGasModel(StrictModule, NonTrainableState):
    """Ideal-gas mixture thermodynamics with explicit formation enthalpies.

    Species thermodynamics are molar. ``formation_molar_enthalpies`` is an
    optional constant offset used when a caloric plan stores sensible energy
    only. NASA plans already containing heats of formation should leave it at
    its default zero value.
    """

    schema: ChemicalSpeciesSchema
    thermodynamics: AbstractSpeciesThermodynamicsPlan
    formation_molar_enthalpies: Array
    composition_tolerance: float = eqx.field(static=True)
    inversion_tolerance: float = eqx.field(static=True)
    inversion_iterations: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: ChemicalSpeciesSchema,
        thermodynamics: AbstractSpeciesThermodynamicsPlan,
        /,
        *,
        formation_molar_enthalpies: ArrayLike | None = None,
        composition_tolerance: float = 1.0e-10,
        inversion_tolerance: float = 1.0e-10,
        inversion_iterations: int = 24,
        model_id: str | None = None,
    ):
        if not isinstance(schema, ChemicalSpeciesSchema):
            raise TypeError("schema must be ChemicalSpeciesSchema.")
        if not isinstance(thermodynamics, AbstractSpeciesThermodynamicsPlan):
            raise TypeError(
                "thermodynamics must implement AbstractSpeciesThermodynamicsPlan."
            )
        if thermodynamics.schema.schema_id != schema.schema_id:
            raise ValueError("Gas-model species and thermodynamic schemas must match.")
        if any(phase is not ChemicalPhaseKind.GAS for phase in schema.phases):
            raise ValueError("ReactingGasModel supports gas-phase species only.")
        formation = (
            np.zeros(schema.species_count, dtype=float)
            if formation_molar_enthalpies is None
            else np.asarray(formation_molar_enthalpies, dtype=float)
        )
        tolerance = float(composition_tolerance)
        root_tolerance = float(inversion_tolerance)
        iterations = int(inversion_iterations)
        if formation.shape != (schema.species_count,) or np.any(~np.isfinite(formation)):
            raise ValueError(
                "formation_molar_enthalpies must contain one finite value per species."
            )
        if (
            not isfinite(tolerance)
            or tolerance <= 0.0
            or not isfinite(root_tolerance)
            or root_tolerance <= 0.0
            or iterations < 2
        ):
            raise ValueError("Gas-model tolerances/iteration count are invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "reacting-ideal-gas-model",
                "schema": schema.schema_id,
                "thermodynamics": thermodynamics.thermodynamics_id,
                "formation_enthalpies": array_tree_fingerprint(formation),
                "composition_tolerance": tolerance,
                "inversion_tolerance": root_tolerance,
                "inversion_iterations": iterations,
            }
        )
        identifier = generated if model_id is None else str(model_id)
        if not identifier:
            raise ValueError("model_id must be nonempty.")
        self.schema = schema
        self.thermodynamics = thermodynamics
        self.formation_molar_enthalpies = jnp.asarray(formation)
        self.composition_tolerance = tolerance
        self.inversion_tolerance = root_tolerance
        self.inversion_iterations = iterations
        self.model_id = identifier

    def mechanism_thermodynamics(self) -> AbstractSpeciesThermodynamicsPlan:
        """Return species thermodynamics with formation offsets materialized."""
        return _FormationAdjustedSpeciesThermodynamicsPlan(
            self.thermodynamics,
            self.formation_molar_enthalpies,
        )

    def _validate_shapes(self, temperature: Array, mass_fractions: Array, /) -> None:
        if (
            mass_fractions.ndim < 1
            or mass_fractions.shape[-1] != self.schema.species_count
        ):
            raise ValueError("mass_fractions must end in the model species axis.")
        if temperature.shape != mass_fractions.shape[:-1]:
            raise ValueError(
                "temperature must match the leading mass-fraction cell shape."
            )

    def composition_valid(self, mass_fractions: ArrayLike, /) -> Array:
        value = jnp.asarray(mass_fractions)
        if value.ndim < 1 or value.shape[-1] != self.schema.species_count:
            raise ValueError("mass_fractions must end in the model species axis.")
        closure = jnp.sum(value, axis=-1)
        return (
            jnp.all(jnp.isfinite(value), axis=-1)
            & jnp.all(value >= 0.0, axis=-1)
            & (jnp.abs(closure - 1.0) <= self.composition_tolerance)
        )

    def mixture_molar_mass(self, mass_fractions: ArrayLike, /) -> Array:
        value = jnp.asarray(mass_fractions)
        if value.ndim < 1 or value.shape[-1] != self.schema.species_count:
            raise ValueError("mass_fractions must end in the model species axis.")
        reciprocal = contract("...s,s->...", value, 1.0 / self.schema.molar_masses)
        return 1.0 / reciprocal

    def mole_fractions(self, mass_fractions: ArrayLike, /) -> Array:
        value = jnp.asarray(mass_fractions)
        molar_mass = self.mixture_molar_mass(value)
        return value * molar_mass[..., None] / self.schema.molar_masses

    def caloric_properties(
        self, temperature: ArrayLike, mass_fractions: ArrayLike, /
    ) -> tuple[Array, Array, Array, Array, Array]:
        temperature_ = jnp.asarray(temperature)
        mass = jnp.asarray(mass_fractions, dtype=temperature_.dtype)
        self._validate_shapes(temperature_, mass)
        species = self.thermodynamics.evaluate(temperature_)
        inverse_mass = 1.0 / self.schema.molar_masses
        formation = self.formation_molar_enthalpies
        cp = contract(
            "...s,...s,s->...",
            mass,
            species.molar_heat_capacity_pressure,
            inverse_mass,
        )
        cv = contract(
            "...s,...s,s->...",
            mass,
            species.molar_heat_capacity_volume,
            inverse_mass,
        )
        enthalpy = contract(
            "...s,...s,s->...",
            mass,
            species.molar_enthalpy + formation,
            inverse_mass,
        )
        internal_energy = contract(
            "...s,...s,s->...",
            mass,
            species.molar_internal_energy + formation,
            inverse_mass,
        )
        successful = species.successful & self.composition_valid(mass)
        return cp, cv, enthalpy, internal_energy, successful

    def evaluate_pressure(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mass_fractions: ArrayLike,
        /,
    ) -> IdealMixtureThermodynamicState:
        temperature_ = jnp.asarray(temperature)
        pressure_ = jnp.asarray(pressure, dtype=temperature_.dtype)
        mass = jnp.asarray(mass_fractions, dtype=temperature_.dtype)
        self._validate_shapes(temperature_, mass)
        if pressure_.shape not in ((), temperature_.shape):
            raise ValueError(
                "pressure must be scalar or match the temperature cell shape."
            )
        pressure_ = jnp.broadcast_to(pressure_, temperature_.shape)
        molar_mass = self.mixture_molar_mass(mass)
        gas_constant = UNIVERSAL_GAS_CONSTANT / molar_mass
        density = pressure_ / (gas_constant * temperature_)
        return self._state(
            temperature_, pressure_, density, mass, molar_mass, gas_constant
        )

    def evaluate_density(
        self,
        temperature: ArrayLike,
        density: ArrayLike,
        mass_fractions: ArrayLike,
        /,
    ) -> IdealMixtureThermodynamicState:
        temperature_ = jnp.asarray(temperature)
        density_ = jnp.asarray(density, dtype=temperature_.dtype)
        mass = jnp.asarray(mass_fractions, dtype=temperature_.dtype)
        self._validate_shapes(temperature_, mass)
        if density_.shape not in ((), temperature_.shape):
            raise ValueError(
                "density must be scalar or match the temperature cell shape."
            )
        density_ = jnp.broadcast_to(density_, temperature_.shape)
        molar_mass = self.mixture_molar_mass(mass)
        gas_constant = UNIVERSAL_GAS_CONSTANT / molar_mass
        pressure = density_ * gas_constant * temperature_
        return self._state(
            temperature_, pressure, density_, mass, molar_mass, gas_constant
        )

    def _state(
        self,
        temperature: Array,
        pressure: Array,
        density: Array,
        mass_fractions: Array,
        molar_mass: Array,
        gas_constant: Array,
        /,
    ) -> IdealMixtureThermodynamicState:
        cp, cv, enthalpy, internal_energy, caloric_success = self.caloric_properties(
            temperature, mass_fractions
        )
        gamma = cp / cv
        sound = jnp.sqrt(gamma * gas_constant * temperature)
        successful = (
            caloric_success
            & jnp.isfinite(temperature)
            & (temperature >= self.thermodynamics.minimum_temperature)
            & (temperature <= self.thermodynamics.maximum_temperature)
            & jnp.isfinite(pressure)
            & (pressure > 0.0)
            & jnp.isfinite(density)
            & (density > 0.0)
            & jnp.isfinite(cp)
            & jnp.isfinite(cv)
            & (cp > cv)
            & (cv > 0.0)
            & jnp.isfinite(sound)
        )
        return IdealMixtureThermodynamicState(
            temperature,
            pressure,
            density,
            mass_fractions,
            self.mole_fractions(mass_fractions),
            molar_mass,
            gas_constant,
            cp,
            cv,
            enthalpy,
            internal_energy,
            sound,
            successful,
            self.model_id,
        )

    def temperature_from_internal_energy(
        self,
        specific_internal_energy: ArrayLike,
        mass_fractions: ArrayLike,
        /,
        *,
        initial_temperature: ArrayLike | None = None,
    ) -> TemperatureInversionEvidence:
        target = jnp.asarray(specific_internal_energy)
        mass = jnp.asarray(mass_fractions, dtype=target.dtype)
        if mass.ndim < 1 or mass.shape[-1] != self.schema.species_count:
            raise ValueError("mass_fractions must end in the model species axis.")
        if target.shape != mass.shape[:-1]:
            raise ValueError("specific_internal_energy must match the cell shape.")
        lower = jnp.full_like(target, self.thermodynamics.minimum_temperature)
        upper = jnp.full_like(target, self.thermodynamics.maximum_temperature)
        if initial_temperature is None:
            temperature = 0.5 * (lower + upper)
        else:
            temperature = jnp.asarray(initial_temperature, dtype=target.dtype)
            if temperature.shape != target.shape:
                raise ValueError("initial_temperature must match the cell shape.")
        _, _, _, energy_lower, lower_success = self.caloric_properties(lower, mass)
        _, _, _, energy_upper, upper_success = self.caloric_properties(upper, mass)

        def inversion_step(_, bracket):
            lower_value, upper_value, temperature_value = bracket
            _, cv, _, energy, _ = self.caloric_properties(temperature_value, mass)
            residual_value = energy - target
            lower_value = jnp.where(residual_value <= 0.0, temperature_value, lower_value)
            upper_value = jnp.where(residual_value > 0.0, temperature_value, upper_value)
            newton = temperature_value - residual_value / cv
            inside = (
                jnp.isfinite(newton) & (newton >= lower_value) & (newton <= upper_value)
            )
            temperature_value = jnp.where(
                inside, newton, 0.5 * (lower_value + upper_value)
            )
            return lower_value, upper_value, temperature_value

        lower, upper, temperature = jax.lax.fori_loop(
            0,
            self.inversion_iterations,
            inversion_step,
            (lower, upper, temperature),
        )
        _, _, _, final_energy, final_caloric_success = self.caloric_properties(
            temperature, mass
        )
        residual = final_energy - target
        scale = jnp.maximum(jnp.abs(target), 1.0)
        bracketed = (target >= energy_lower) & (target <= energy_upper)
        successful = (
            lower_success
            & upper_success
            & final_caloric_success
            & bracketed
            & jnp.isfinite(residual)
            & (jnp.abs(residual) <= self.inversion_tolerance * scale)
        )
        return TemperatureInversionEvidence(
            temperature,
            residual,
            upper - lower,
            jnp.asarray(self.inversion_iterations, dtype=jnp.int32),
            successful,
            self.model_id,
        )

    def state_from_density_internal_energy(
        self,
        density: ArrayLike,
        specific_internal_energy: ArrayLike,
        mass_fractions: ArrayLike,
        /,
        *,
        initial_temperature: ArrayLike | None = None,
    ) -> tuple[IdealMixtureThermodynamicState, TemperatureInversionEvidence]:
        inversion = self.temperature_from_internal_energy(
            specific_internal_energy,
            mass_fractions,
            initial_temperature=initial_temperature,
        )
        state = self.evaluate_density(
            inversion.temperature,
            density,
            mass_fractions,
        )
        state = eqx.tree_at(
            lambda value: value.successful,
            state,
            state.successful & inversion.successful,
        )
        return state, inversion


__all__ = [
    "IdealMixtureThermodynamicState",
    "ReactingGasModel",
    "TemperatureInversionEvidence",
]
