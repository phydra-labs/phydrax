#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._chemical_mechanism import (
    ChemicalRateEvaluation,
    PreparedChemicalMechanism,
)
from ...equations._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ...equations._homogeneous_thermodynamics import (
    HomogeneousHelmholtzPlan,
    HomogeneousThermodynamicEvaluation,
    ZeroResidualHelmholtzTerm,
)


class LowMachReactiveState(StrictModule):
    velocity: Array
    temperature: Array
    mass_fractions: Array
    thermodynamic_pressure: Array
    state_id: str = eqx.field(static=True)


class LowMachConstraintEvidence(StrictModule):
    divergence_source: Array
    thermal_expansion: Array
    compositional_expansion: Array
    pressure_expansion: Array
    density: Array
    mass_fraction_closure: Array
    mass_fraction_rate_closure: Array
    successful: Array
    formulation_id: str = eqx.field(static=True)


class LowMachReactiveEvaluation(StrictModule):
    mass_fractions: Array
    thermodynamics: HomogeneousThermodynamicEvaluation
    divergence: LowMachConstraintEvidence
    chemistry: ChemicalRateEvaluation
    species_mass_production_rate: Array
    diagnostic_heat_release_rate: Array
    successful: Array
    formulation_id: str = eqx.field(static=True)


class LowMachReactingFormulation(StrictModule, NonTrainableState):
    """Variable-density low-Mach constraint at uniform thermodynamic pressure."""

    thermodynamics: HomogeneousHelmholtzPlan
    mechanism: PreparedChemicalMechanism | None
    dimension: int = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: HomogeneousHelmholtzPlan,
        dimension: int,
        /,
        *,
        mechanism: PreparedChemicalMechanism | None = None,
        constraint_tolerance: float = 1.0e-10,
    ):
        if not isinstance(thermodynamics, HomogeneousHelmholtzPlan):
            raise TypeError("thermodynamics must be HomogeneousHelmholtzPlan.")
        if not isinstance(thermodynamics.residual, ZeroResidualHelmholtzTerm):
            raise TypeError(
                "Low-Mach reacting flow currently requires ideal-mixture thermodynamics."
            )
        dimension_ = int(dimension)
        tolerance = float(constraint_tolerance)
        if dimension_ not in (1, 2, 3):
            raise ValueError("Low-Mach reacting dimension must be one to three.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("constraint_tolerance must be finite and positive.")
        if mechanism is not None:
            if not isinstance(mechanism, PreparedChemicalMechanism):
                raise TypeError("mechanism must be PreparedChemicalMechanism or None.")
            if (
                mechanism.schema.schema_id != thermodynamics.schema.schema_id
                or mechanism.thermodynamics.thermodynamics_id
                != thermodynamics.thermodynamics.thermodynamics_id
            ):
                raise ValueError(
                    "Low-Mach chemistry and thermodynamics must match exactly."
                )
        self.thermodynamics = thermodynamics
        self.mechanism = mechanism
        self.dimension = dimension_
        self.constraint_tolerance = tolerance
        self.formulation_id = canonical_fingerprint(
            {
                "kind": "low-mach-reacting-formulation",
                "thermodynamics": thermodynamics.model_id,
                "mechanism": None if mechanism is None else mechanism.mechanism_id,
                "dimension": dimension_,
                "constraint_tolerance": tolerance,
                "pressure_role": "spatially-uniform-thermodynamic-pressure",
            }
        )

    def initial_state(
        self,
        velocity: ArrayLike,
        temperature: ArrayLike,
        mass_fractions: ArrayLike,
        thermodynamic_pressure: ArrayLike,
        /,
    ) -> LowMachReactiveState:
        velocity_ = jnp.asarray(velocity)
        temperature_ = jnp.asarray(temperature, dtype=velocity_.dtype)
        mass = jnp.asarray(mass_fractions, dtype=velocity_.dtype)
        pressure = jnp.asarray(thermodynamic_pressure, dtype=velocity_.dtype)
        if velocity_.ndim < 1 or velocity_.shape[-1] != self.dimension:
            raise ValueError("velocity must end in the formulation dimension.")
        cell_shape = velocity_.shape[:-1]
        if temperature_.shape != cell_shape:
            raise ValueError("temperature must match velocity cell shape.")
        if mass.shape != cell_shape + (self.thermodynamics.schema.species_count,):
            raise ValueError("mass_fractions must contain every species slot.")
        if pressure.shape != ():
            raise ValueError(
                "thermodynamic_pressure must be one spatially uniform scalar."
            )
        identifier = canonical_fingerprint(
            {
                "kind": "low-mach-reacting-state",
                "formulation": self.formulation_id,
                "cell_shape": list(cell_shape),
            }
        )
        return LowMachReactiveState(velocity_, temperature_, mass, pressure, identifier)

    def _pressure_state(
        self,
        temperature: Array,
        pressure: Array,
        mass_fractions: Array,
        /,
    ) -> HomogeneousThermodynamicEvaluation:
        molar_masses = self.thermodynamics.schema.molar_masses.astype(
            mass_fractions.dtype
        )
        reciprocal_molar_mass = jnp.sum(mass_fractions / molar_masses, axis=-1)
        mixture_molar_mass = 1.0 / reciprocal_molar_mass
        mole_fraction = mass_fractions * mixture_molar_mass[..., None] / molar_masses
        molar_density = pressure / (UNIVERSAL_GAS_CONSTANT * temperature)
        return self.thermodynamics.evaluate(temperature, molar_density, mole_fraction)

    def divergence_source(
        self,
        temperature: ArrayLike,
        mass_fractions: ArrayLike,
        temperature_rate: ArrayLike,
        mass_fraction_rate: ArrayLike,
        thermodynamic_pressure: ArrayLike,
        /,
        *,
        thermodynamic_pressure_rate: ArrayLike = 0.0,
    ) -> LowMachConstraintEvidence:
        temperature_ = jnp.asarray(temperature)
        mass = jnp.asarray(mass_fractions, dtype=temperature_.dtype)
        temperature_rate_ = jnp.asarray(temperature_rate, dtype=temperature_.dtype)
        mass_rate = jnp.asarray(mass_fraction_rate, dtype=temperature_.dtype)
        pressure = jnp.asarray(thermodynamic_pressure, dtype=temperature_.dtype)
        pressure_rate = jnp.asarray(thermodynamic_pressure_rate, dtype=temperature_.dtype)
        cell_shape = temperature_.shape
        species_count = self.thermodynamics.schema.species_count
        if mass.shape != cell_shape + (species_count,) or mass_rate.shape != mass.shape:
            raise ValueError("Mass fractions/rates have invalid cell or species shape.")
        if temperature_rate_.shape != cell_shape:
            raise ValueError("temperature_rate must match the cell shape.")
        if pressure.shape != () or pressure_rate.shape != ():
            raise ValueError(
                "Thermodynamic pressure and its rate must be spatially uniform scalars."
            )
        thermo = self._pressure_state(temperature_, pressure, mass)
        molar_masses = self.thermodynamics.schema.molar_masses.astype(mass.dtype)
        thermal = thermo.thermal_expansion * temperature_rate_
        composition = thermo.molar_mass * contract(
            "...s,s->...", mass_rate, 1.0 / molar_masses, backend="jax"
        )
        pressure_expansion = -thermo.isothermal_compressibility * pressure_rate
        source = thermal + composition + pressure_expansion
        mass_closure = jnp.sum(mass, axis=-1) - 1.0
        rate_closure = jnp.sum(mass_rate, axis=-1)
        successful = (
            thermo.evidence.successful
            & jnp.isfinite(temperature_rate_)
            & jnp.all(jnp.isfinite(mass_rate), axis=-1)
            & jnp.isfinite(pressure_rate)
            & jnp.isfinite(source)
            & (jnp.abs(mass_closure) <= self.constraint_tolerance)
            & (jnp.abs(rate_closure) <= self.constraint_tolerance)
        )
        return LowMachConstraintEvidence(
            source,
            thermal,
            composition,
            pressure_expansion,
            thermo.mass_density,
            mass_closure,
            rate_closure,
            successful,
            self.formulation_id,
        )

    def evaluate_chemistry(
        self,
        state: LowMachReactiveState,
        /,
        *,
        thermodynamic_pressure_rate: ArrayLike = 0.0,
    ) -> LowMachReactiveEvaluation:
        if not isinstance(state, LowMachReactiveState):
            raise TypeError("state must be LowMachReactiveState.")
        if self.mechanism is None:
            raise ValueError("Chemistry evaluation requires a prepared mechanism.")
        mass = state.mass_fractions
        thermo = self._pressure_state(
            state.temperature, state.thermodynamic_pressure, mass
        )
        species_density = thermo.mass_density[..., None] * mass
        concentrations = species_density / self.mechanism.schema.molar_masses.astype(
            mass.dtype
        )
        chemistry = self.mechanism.evaluate(
            concentrations,
            state.temperature,
            jnp.broadcast_to(state.thermodynamic_pressure, state.temperature.shape),
        )
        species_mass_rate = (
            chemistry.species_amount_rate
            * self.mechanism.schema.molar_masses.astype(mass.dtype)
        )
        mass_fraction_rate = species_mass_rate / thermo.mass_density[..., None]
        heat_release = -contract(
            "...s,...s->...",
            chemistry.species_amount_rate,
            chemistry.thermodynamics.molar_enthalpy,
            backend="jax",
        )
        specific_heat_capacity_pressure = (
            thermo.molar_heat_capacity_pressure / thermo.molar_mass
        )
        temperature_rate = heat_release / (
            thermo.mass_density * specific_heat_capacity_pressure
        )
        divergence = self.divergence_source(
            state.temperature,
            mass,
            temperature_rate,
            mass_fraction_rate,
            state.thermodynamic_pressure,
            thermodynamic_pressure_rate=thermodynamic_pressure_rate,
        )
        mass_defect = jnp.abs(jnp.sum(species_mass_rate, axis=-1))
        mass_scale = jnp.maximum(jnp.max(jnp.abs(species_mass_rate), axis=-1), 1.0)
        successful = (
            chemistry.successful
            & divergence.successful
            & jnp.isfinite(heat_release)
            & (mass_defect <= self.constraint_tolerance * mass_scale)
        )
        return LowMachReactiveEvaluation(
            mass,
            thermo,
            divergence,
            chemistry,
            species_mass_rate,
            heat_release,
            successful,
            self.formulation_id,
        )


__all__ = [
    "LowMachConstraintEvidence",
    "LowMachReactiveEvaluation",
    "LowMachReactiveState",
    "LowMachReactingFormulation",
]
