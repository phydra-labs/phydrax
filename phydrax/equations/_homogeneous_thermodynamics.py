#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from ._chemical_thermodynamics import (
    AbstractSpeciesThermodynamicsPlan,
    UNIVERSAL_GAS_CONSTANT,
)


class ThermodynamicDomainEvidence(StrictModule):
    temperature_margin: Array
    composition_margin: Array
    density_margin: Array
    heat_capacity_margin: Array
    mechanical_stability_margin: Array
    sound_speed_squared_margin: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class HomogeneousThermodynamicEvaluation(StrictModule):
    temperature: Array
    molar_density: Array
    mole_fraction: Array
    mass_density: Array
    molar_mass: Array
    molar_helmholtz_energy: Array
    molar_internal_energy: Array
    molar_enthalpy: Array
    molar_entropy: Array
    molar_gibbs_energy: Array
    molar_heat_capacity_volume: Array
    molar_heat_capacity_pressure: Array
    pressure: Array
    pressure_temperature_derivative: Array
    pressure_molar_density_derivative: Array
    isothermal_compressibility: Array
    thermal_expansion: Array
    frozen_sound_speed_squared: Array
    evidence: ThermodynamicDomainEvidence
    model_id: str = eqx.field(static=True)


class HomogeneousChemicalEvaluation(StrictModule):
    state: HomogeneousThermodynamicEvaluation
    chemical_potential: Array
    log_fugacity_coefficient: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class DensityEnergyStateResult(StrictModule):
    state: HomogeneousThermodynamicEvaluation
    species_mass_density: Array
    internal_energy_density: Array
    energy_residual: Array
    temperature_bracket_margin: Array
    iteration_count: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class AbstractMolarHelmholtzTerm(StrictModule, NonTrainableState, abc.ABC):
    schema: ChemicalSpeciesSchema
    term_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def molar_helmholtz_energy(
        self,
        temperature: ArrayLike,
        molar_density: ArrayLike,
        mole_fraction: ArrayLike,
        /,
    ) -> Array:
        raise NotImplementedError


class IdealGasReferenceHelmholtzTerm(AbstractMolarHelmholtzTerm):
    """Ideal-gas species reference calorics plus ideal mixing."""

    thermodynamics: AbstractSpeciesThermodynamicsPlan
    standard_pressure: float = eqx.field(static=True)

    def __init__(
        self,
        schema: ChemicalSpeciesSchema,
        thermodynamics: AbstractSpeciesThermodynamicsPlan,
        /,
    ) -> None:
        if not isinstance(schema, ChemicalSpeciesSchema):
            raise TypeError("schema must be ChemicalSpeciesSchema.")
        if not isinstance(thermodynamics, AbstractSpeciesThermodynamicsPlan):
            raise TypeError(
                "thermodynamics must implement AbstractSpeciesThermodynamicsPlan."
            )
        if thermodynamics.schema.schema_id != schema.schema_id:
            raise ValueError("Species thermodynamics and schema must match exactly.")
        if any(phase is not ChemicalPhaseKind.GAS for phase in schema.phases):
            raise ValueError(
                "Ideal gas mixture thermodynamics requires only gas species."
            )
        if len(set(int(value) for value in np.asarray(schema.phase_ids))) != 1:
            raise ValueError(
                "Ideal gas mixture species must share one gas phase instance."
            )
        if schema.species_count != schema.component_count or not np.array_equal(
            np.asarray(schema.species_component_indices),
            np.arange(schema.species_count),
        ):
            raise ValueError(
                "Ideal gas mixture thermodynamics requires one species occurrence "
                "per component in catalog order."
            )
        phase = schema.phase_specs[int(np.asarray(schema.phase_ids)[0])]
        if phase.standard_pressure is None:
            raise ValueError("Gas phase standard pressure is required.")
        generated = canonical_fingerprint(
            {
                "kind": "ideal-gas-reference-helmholtz",
                "schema": schema.schema_id,
                "thermodynamics": thermodynamics.thermodynamics_id,
                "standard_pressure": phase.standard_pressure,
            }
        )
        self.schema = schema
        self.thermodynamics = thermodynamics
        self.standard_pressure = phase.standard_pressure
        self.term_id = generated

    def molar_helmholtz_energy(
        self,
        temperature: ArrayLike,
        molar_density: ArrayLike,
        mole_fraction: ArrayLike,
        /,
    ) -> Array:
        temperature_value = jnp.asarray(temperature)
        density = jnp.asarray(molar_density)
        composition = jnp.asarray(mole_fraction)
        thermo = self.thermodynamics.evaluate(temperature_value)
        safe_fraction = jnp.where(composition > 0.0, composition, 1.0)
        partial_pressure_ratio = (
            safe_fraction
            * density[..., None]
            * UNIVERSAL_GAS_CONSTANT
            * temperature_value[..., None]
            / self.standard_pressure
        )
        term = (
            thermo.molar_gibbs_energy
            - UNIVERSAL_GAS_CONSTANT * temperature_value[..., None]
            + UNIVERSAL_GAS_CONSTANT
            * temperature_value[..., None]
            * jnp.log(partial_pressure_ratio)
        )
        return jnp.sum(jnp.where(composition > 0.0, composition * term, 0.0), axis=-1)

    def standard_gibbs_energy(self, temperature: ArrayLike, /) -> Array:
        return self.thermodynamics.evaluate(temperature).molar_gibbs_energy


class ZeroResidualHelmholtzTerm(AbstractMolarHelmholtzTerm):
    """Zero residual term for ideal mixtures."""

    def __init__(self, schema: ChemicalSpeciesSchema, /) -> None:
        if not isinstance(schema, ChemicalSpeciesSchema):
            raise TypeError("schema must be ChemicalSpeciesSchema.")
        self.schema = schema
        self.term_id = canonical_fingerprint(
            {"kind": "zero-residual-helmholtz", "schema": schema.schema_id}
        )

    def molar_helmholtz_energy(
        self,
        temperature: ArrayLike,
        molar_density: ArrayLike,
        mole_fraction: ArrayLike,
        /,
    ) -> Array:
        temperature_value = jnp.asarray(temperature)
        density = jnp.asarray(molar_density)
        composition = jnp.asarray(mole_fraction)
        return jnp.zeros(
            jnp.broadcast_shapes(
                temperature_value.shape,
                density.shape,
                composition.shape[:-1],
            ),
            dtype=jnp.result_type(temperature_value, density, composition),
        )


class HomogeneousHelmholtzPlan(StrictModule, NonTrainableState):
    """Complete homogeneous phase thermodynamics derived from Helmholtz energy."""

    ideal: IdealGasReferenceHelmholtzTerm
    residual: AbstractMolarHelmholtzTerm
    minimum_molar_density: float = eqx.field(static=True)
    maximum_molar_density: float = eqx.field(static=True)
    composition_tolerance: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        ideal: IdealGasReferenceHelmholtzTerm,
        residual: AbstractMolarHelmholtzTerm,
        /,
        *,
        minimum_molar_density: float = 1.0e-12,
        maximum_molar_density: float = 1.0e8,
        composition_tolerance: float = 1.0e-10,
    ) -> None:
        if not isinstance(ideal, IdealGasReferenceHelmholtzTerm):
            raise TypeError("ideal must be IdealGasReferenceHelmholtzTerm.")
        if not isinstance(residual, AbstractMolarHelmholtzTerm):
            raise TypeError("residual must implement AbstractMolarHelmholtzTerm.")
        if ideal.schema.schema_id != residual.schema.schema_id:
            raise ValueError("Ideal and residual Helmholtz schemas must match exactly.")
        lower = float(minimum_molar_density)
        upper = float(maximum_molar_density)
        tolerance = float(composition_tolerance)
        if not 0.0 < lower < upper or not np.isfinite(upper):
            raise ValueError(
                "Molar-density bounds must be finite, positive, and ordered."
            )
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("composition_tolerance must be finite and positive.")
        generated = canonical_fingerprint(
            {
                "kind": "homogeneous-helmholtz",
                "ideal": ideal.term_id,
                "residual": residual.term_id,
                "density_bounds": [lower, upper],
                "composition_tolerance": tolerance,
            }
        )
        self.ideal = ideal
        self.residual = residual
        self.minimum_molar_density = lower
        self.maximum_molar_density = upper
        self.composition_tolerance = tolerance
        self.model_id = generated

    @property
    def schema(self) -> ChemicalSpeciesSchema:
        return self.ideal.schema

    @property
    def thermodynamics(self) -> AbstractSpeciesThermodynamicsPlan:
        return self.ideal.thermodynamics

    def molar_helmholtz_energy(
        self,
        temperature: ArrayLike,
        molar_density: ArrayLike,
        mole_fraction: ArrayLike,
        /,
    ) -> Array:
        return self.ideal.molar_helmholtz_energy(
            temperature, molar_density, mole_fraction
        ) + self.residual.molar_helmholtz_energy(
            temperature, molar_density, mole_fraction
        )

    def evaluate(
        self,
        temperature: ArrayLike,
        molar_density: ArrayLike,
        mole_fraction: ArrayLike,
        /,
    ) -> HomogeneousThermodynamicEvaluation:
        temperature_value, density, composition = _broadcast_state(
            temperature,
            molar_density,
            mole_fraction,
            self.schema.species_count,
        )
        shape = temperature_value.shape
        flat_temperature = temperature_value.reshape((-1,))
        flat_density = density.reshape((-1,))
        flat_composition = composition.reshape((-1, self.schema.species_count))
        outputs = jax.vmap(self._evaluate_scalar)(
            flat_temperature,
            flat_density,
            flat_composition,
        )
        return jax.tree_util.tree_map(
            lambda value: value.reshape(shape + value.shape[1:]),
            outputs,
        )

    def evaluate_chemical(
        self,
        temperature: ArrayLike,
        molar_density: ArrayLike,
        mole_fraction: ArrayLike,
        /,
    ) -> HomogeneousChemicalEvaluation:
        state = self.evaluate(temperature, molar_density, mole_fraction)
        temperature_value = state.temperature
        density = state.molar_density
        composition = state.mole_fraction
        shape = temperature_value.shape
        flat_temperature = temperature_value.reshape((-1,))
        flat_density = density.reshape((-1,))
        flat_composition = composition.reshape((-1, self.schema.species_count))
        chemical_potential = jax.vmap(self._chemical_potential_scalar)(
            flat_temperature,
            flat_density,
            flat_composition,
        ).reshape(shape + (self.schema.species_count,))
        standard_gibbs = self.ideal.standard_gibbs_energy(temperature_value)
        safe_fraction = jnp.where(composition > 0.0, composition, 1.0)
        ideal_pressure_potential = standard_gibbs + (
            UNIVERSAL_GAS_CONSTANT
            * temperature_value[..., None]
            * jnp.log(
                safe_fraction * state.pressure[..., None] / self.ideal.standard_pressure
            )
        )
        log_fugacity = (chemical_potential - ideal_pressure_potential) / (
            UNIVERSAL_GAS_CONSTANT * temperature_value[..., None]
        )
        log_fugacity = jnp.where(composition > 0.0, log_fugacity, jnp.inf)
        successful = state.evidence.successful & jnp.all(
            jnp.isfinite(chemical_potential), axis=-1
        )
        return HomogeneousChemicalEvaluation(
            state,
            chemical_potential,
            log_fugacity,
            successful,
            self.model_id,
        )

    def evaluate_density_temperature(
        self,
        species_mass_density: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> HomogeneousThermodynamicEvaluation:
        density = jnp.asarray(species_mass_density)
        if density.ndim < 1 or density.shape[-1] != self.schema.species_count:
            raise ValueError("species_mass_density must end in the species axis.")
        masses = self.schema.molar_masses.astype(density.dtype)
        concentration = density / masses
        molar_density = jnp.sum(concentration, axis=-1)
        safe_density = jnp.maximum(
            molar_density,
            jnp.asarray(self.minimum_molar_density, dtype=density.dtype),
        )
        composition = concentration / safe_density[..., None]
        return self.evaluate(temperature, safe_density, composition)

    def solve_density_energy(
        self,
        species_mass_density: ArrayLike,
        internal_energy_density: ArrayLike,
        /,
        *,
        maximum_iterations: int = 80,
    ) -> DensityEnergyStateResult:
        density = jnp.asarray(species_mass_density)
        target = jnp.asarray(internal_energy_density)
        if density.ndim < 1 or density.shape[-1] != self.schema.species_count:
            raise ValueError("species_mass_density must end in the species axis.")
        if target.shape != density.shape[:-1]:
            raise ValueError(
                "internal_energy_density must match the leading state shape."
            )
        iterations = int(maximum_iterations)
        if iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        lower = jnp.full_like(target, self.thermodynamics.minimum_temperature)
        upper = jnp.full_like(target, self.thermodynamics.maximum_temperature)

        def residual(temperature_value):
            state_value = self.evaluate_density_temperature(density, temperature_value)
            concentration = density / self.schema.molar_masses.astype(density.dtype)
            return (
                jnp.sum(concentration, axis=-1) * state_value.molar_internal_energy
                - target
            )

        lower_residual = residual(lower)
        upper_residual = residual(upper)
        bracketed = (
            jnp.isfinite(lower_residual)
            & jnp.isfinite(upper_residual)
            & (lower_residual <= 0.0)
            & (upper_residual >= 0.0)
        )

        def body(_, bounds):
            low, high = bounds
            midpoint = 0.5 * (low + high)
            value = residual(midpoint)
            use_upper = value >= 0.0
            return (
                jnp.where(use_upper, low, midpoint),
                jnp.where(use_upper, midpoint, high),
            )

        lower_final, upper_final = jax.lax.fori_loop(
            0,
            iterations,
            body,
            (lower, upper),
        )
        temperature_value = 0.5 * (lower_final + upper_final)
        state = self.evaluate_density_temperature(density, temperature_value)
        energy_residual = residual(temperature_value)
        tolerance = (
            256.0 * jnp.finfo(target.dtype).eps * jnp.maximum(jnp.abs(target), 1.0)
        )
        successful = (
            bracketed
            & state.evidence.successful
            & jnp.isfinite(energy_residual)
            & (jnp.abs(energy_residual) <= tolerance)
        )
        return DensityEnergyStateResult(
            state,
            density,
            target,
            energy_residual,
            0.5 * (upper_final - lower_final),
            jnp.full(target.shape, iterations, dtype=jnp.int32),
            successful,
            self.model_id,
        )

    def _evaluate_scalar(self, temperature, density, composition):
        safe_temperature = jnp.clip(
            temperature,
            self.thermodynamics.minimum_temperature,
            self.thermodynamics.maximum_temperature,
        )
        safe_density = jnp.clip(
            density,
            self.minimum_molar_density,
            self.maximum_molar_density,
        )
        clipped = jnp.maximum(composition, 0.0)
        composition_sum = jnp.sum(clipped)
        safe_composition = clipped / jnp.maximum(composition_sum, 1.0)

        def energy(t, c):
            return self.molar_helmholtz_energy(t, c, safe_composition)

        a = energy(safe_temperature, safe_density)
        a_t = jax.grad(energy, argnums=0)(safe_temperature, safe_density)
        a_c = jax.grad(energy, argnums=1)(safe_temperature, safe_density)
        a_tt = jax.grad(jax.grad(energy, argnums=0), argnums=0)(
            safe_temperature, safe_density
        )
        a_ct = jax.grad(jax.grad(energy, argnums=1), argnums=0)(
            safe_temperature, safe_density
        )
        a_cc = jax.grad(jax.grad(energy, argnums=1), argnums=1)(
            safe_temperature, safe_density
        )
        pressure = safe_density**2 * a_c
        pressure_t = safe_density**2 * a_ct
        pressure_c = 2.0 * safe_density * a_c + safe_density**2 * a_cc
        entropy = -a_t
        internal_energy = a - safe_temperature * a_t
        enthalpy = internal_energy + pressure / safe_density
        gibbs = a + pressure / safe_density
        cv = -safe_temperature * a_tt
        safe_pressure_c = jnp.where(pressure_c > 0.0, pressure_c, 1.0)
        cp = cv + safe_temperature * pressure_t**2 / (safe_density**2 * safe_pressure_c)
        molar_mass = contract(
            "s,s->",
            safe_composition,
            self.schema.molar_masses.astype(safe_temperature.dtype),
        )
        sound_numerator = pressure_c + safe_temperature * pressure_t**2 / (
            safe_density**2 * jnp.where(cv > 0.0, cv, 1.0)
        )
        sound_squared = sound_numerator / jnp.where(molar_mass > 0.0, molar_mass, 1.0)
        tolerance = jnp.asarray(self.composition_tolerance, dtype=safe_temperature.dtype)
        thermo_success = self.thermodynamics.evaluate(safe_temperature).successful
        temperature_margin = jnp.minimum(
            safe_temperature - self.thermodynamics.minimum_temperature,
            self.thermodynamics.maximum_temperature - safe_temperature,
        )
        composition_margin = jnp.min(safe_composition)
        density_margin = jnp.minimum(
            safe_density - self.minimum_molar_density,
            self.maximum_molar_density - safe_density,
        )
        successful = (
            jnp.isfinite(temperature)
            & jnp.isfinite(density)
            & jnp.all(jnp.isfinite(composition))
            & (temperature >= self.thermodynamics.minimum_temperature)
            & (temperature <= self.thermodynamics.maximum_temperature)
            & (density >= self.minimum_molar_density)
            & (density <= self.maximum_molar_density)
            & jnp.all(composition >= 0.0)
            & (jnp.abs(jnp.sum(composition) - 1.0) <= tolerance)
            & thermo_success
            & jnp.isfinite(a)
            & jnp.isfinite(pressure)
            & (cv > 0.0)
            & (pressure_c > 0.0)
            & (sound_squared > 0.0)
        )
        evidence = ThermodynamicDomainEvidence(
            temperature_margin,
            composition_margin,
            density_margin,
            cv,
            pressure_c,
            sound_squared,
            successful,
            self.model_id,
        )
        return HomogeneousThermodynamicEvaluation(
            safe_temperature,
            safe_density,
            safe_composition,
            safe_density * molar_mass,
            molar_mass,
            a,
            internal_energy,
            enthalpy,
            entropy,
            gibbs,
            cv,
            cp,
            pressure,
            pressure_t,
            pressure_c,
            1.0 / (safe_density * safe_pressure_c),
            pressure_t / (safe_density * safe_pressure_c),
            sound_squared,
            evidence,
            self.model_id,
        )

    def _chemical_potential_scalar(self, temperature, density, composition):
        def energy(c, x):
            return self.molar_helmholtz_energy(temperature, c, x)

        molar_energy = energy(density, composition)
        density_derivative = jax.grad(energy, argnums=0)(density, composition)
        composition_derivative = jax.grad(energy, argnums=1)(density, composition)
        mean_composition_derivative = contract(
            "s,s->", composition, composition_derivative
        )
        return (
            molar_energy
            + density * density_derivative
            + composition_derivative
            - mean_composition_derivative
        )


def _broadcast_state(
    temperature: ArrayLike,
    molar_density: ArrayLike,
    mole_fraction: ArrayLike,
    species_count: int,
):
    temperature_value = jnp.asarray(temperature)
    density = jnp.asarray(molar_density)
    composition = jnp.asarray(mole_fraction)
    if composition.ndim < 1 or composition.shape[-1] != species_count:
        raise ValueError("mole_fraction must end in the species axis.")
    if not jnp.issubdtype(temperature_value.dtype, jnp.inexact):
        raise TypeError("temperature must have inexact dtype.")
    dtype = jnp.result_type(temperature_value, density, composition)
    shape = jnp.broadcast_shapes(
        temperature_value.shape,
        density.shape,
        composition.shape[:-1],
    )
    return (
        jnp.broadcast_to(temperature_value, shape).astype(dtype),
        jnp.broadcast_to(density, shape).astype(dtype),
        jnp.broadcast_to(composition, shape + (species_count,)).astype(dtype),
    )


__all__ = [
    "AbstractMolarHelmholtzTerm",
    "DensityEnergyStateResult",
    "HomogeneousChemicalEvaluation",
    "HomogeneousHelmholtzPlan",
    "HomogeneousThermodynamicEvaluation",
    "IdealGasReferenceHelmholtzTerm",
    "ThermodynamicDomainEvidence",
    "ZeroResidualHelmholtzTerm",
]
