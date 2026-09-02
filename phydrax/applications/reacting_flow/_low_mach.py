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
from ._mechanism import CompiledChemicalMechanism, CompiledMechanismEvaluation
from ._thermodynamics import IdealMixtureThermodynamicState, ReactingGasModel


class LowMachReactiveState(StrictModule):
    velocity: Array
    temperature: Array
    independent_mass_fractions: Array
    thermodynamic_pressure: Array
    state_id: str = eqx.field(static=True)


class LowMachConstraintEvidence(StrictModule):
    divergence_source: Array
    thermal_expansion: Array
    compositional_expansion: Array
    pressure_expansion: Array
    density: Array
    mass_fraction_closure: Array
    successful: Array
    formulation_id: str = eqx.field(static=True)


class LowMachReactiveEvaluation(StrictModule):
    mass_fractions: Array
    thermodynamics: IdealMixtureThermodynamicState
    divergence: LowMachConstraintEvidence
    chemistry: CompiledMechanismEvaluation | None
    formulation_id: str = eqx.field(static=True)


class LowMachReactingFormulation(StrictModule, NonTrainableState):
    """Low-Mach reacting constraint with a separate thermodynamic pressure.

    This formulation deliberately does not inherit the incompressible MAC
    equation: density follows the reacting ideal-mixture EOS and velocity is
    constrained by the thermochemical divergence source.
    """

    gas_model: ReactingGasModel
    mechanism: CompiledChemicalMechanism | None
    dimension: int = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)

    def __init__(
        self,
        gas_model: ReactingGasModel,
        dimension: int,
        /,
        *,
        mechanism: CompiledChemicalMechanism | None = None,
        constraint_tolerance: float = 1.0e-10,
    ):
        if not isinstance(gas_model, ReactingGasModel):
            raise TypeError("gas_model must be ReactingGasModel.")
        dimension_ = int(dimension)
        tolerance = float(constraint_tolerance)
        if dimension_ not in (1, 2, 3):
            raise ValueError("Low-Mach reacting dimension must be one to three.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("constraint_tolerance must be finite and positive.")
        if mechanism is not None:
            if not isinstance(mechanism, CompiledChemicalMechanism):
                raise TypeError("mechanism must be CompiledChemicalMechanism or None.")
            if mechanism.gas_model.model_id != gas_model.model_id:
                raise ValueError("Low-Mach chemistry and gas model must match exactly.")
        self.gas_model = gas_model
        self.mechanism = mechanism
        self.dimension = dimension_
        self.constraint_tolerance = tolerance
        self.formulation_id = canonical_fingerprint(
            {
                "kind": "low-mach-reacting-formulation",
                "gas_model": gas_model.model_id,
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
        independent_mass_fractions: ArrayLike,
        thermodynamic_pressure: ArrayLike,
        /,
    ) -> LowMachReactiveState:
        velocity_ = jnp.asarray(velocity)
        temperature_ = jnp.asarray(temperature, dtype=velocity_.dtype)
        independent = jnp.asarray(independent_mass_fractions, dtype=velocity_.dtype)
        pressure = jnp.asarray(thermodynamic_pressure, dtype=velocity_.dtype)
        if velocity_.ndim < 1 or velocity_.shape[-1] != self.dimension:
            raise ValueError("velocity must end in the formulation dimension.")
        cell_shape = velocity_.shape[:-1]
        if temperature_.shape != cell_shape:
            raise ValueError("temperature must match velocity cell shape.")
        if independent.shape != cell_shape + (self.gas_model.schema.species_count - 1,):
            raise ValueError("independent_mass_fractions has an invalid shape.")
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
        return LowMachReactiveState(
            velocity_, temperature_, independent, pressure, identifier
        )

    def complete_mass_fractions(self, independent_mass_fractions: ArrayLike, /) -> Array:
        independent = jnp.asarray(independent_mass_fractions)
        if independent.ndim < 1 or (
            independent.shape[-1] != self.gas_model.schema.species_count - 1
        ):
            raise ValueError("independent_mass_fractions has an invalid species axis.")
        dependent = 1.0 - jnp.sum(independent, axis=-1)
        return jnp.concatenate((independent, dependent[..., None]), axis=-1)

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
        species_count = self.gas_model.schema.species_count
        if mass.shape != cell_shape + (species_count,) or mass_rate.shape != mass.shape:
            raise ValueError("Mass fractions/rates have invalid cell or species shape.")
        if temperature_rate_.shape != cell_shape:
            raise ValueError("temperature_rate must match the cell shape.")
        if pressure.shape != () or pressure_rate.shape != ():
            raise ValueError(
                "Thermodynamic pressure and its rate must be spatially uniform scalars."
            )
        mixture_molar_mass = self.gas_model.mixture_molar_mass(mass)
        thermal = temperature_rate_ / temperature_
        composition = mixture_molar_mass * contract(
            "...s,s->...", mass_rate, 1.0 / self.gas_model.schema.molar_masses
        )
        pressure_expansion = -pressure_rate / pressure
        source = thermal + composition + pressure_expansion
        thermo = self.gas_model.evaluate_pressure(temperature_, pressure, mass)
        closure = jnp.sum(mass_rate, axis=-1)
        successful = (
            thermo.successful
            & jnp.isfinite(temperature_rate_)
            & jnp.all(jnp.isfinite(mass_rate), axis=-1)
            & jnp.isfinite(pressure_rate)
            & jnp.isfinite(source)
            & (jnp.abs(closure) <= self.constraint_tolerance)
        )
        return LowMachConstraintEvidence(
            source,
            thermal,
            composition,
            pressure_expansion,
            thermo.density,
            closure,
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
            raise ValueError("Chemistry evaluation requires a compiled mechanism.")
        mass = self.complete_mass_fractions(state.independent_mass_fractions)
        thermo = self.gas_model.evaluate_pressure(
            state.temperature, state.thermodynamic_pressure, mass
        )
        chemistry = self.mechanism.source_from_density_mass_fractions(
            thermo.density,
            state.temperature,
            state.thermodynamic_pressure,
            mass,
        )
        mass_fraction_rate = (
            chemistry.species_mass_production_rate / thermo.density[..., None]
        )
        temperature_rate = chemistry.heat_release_rate / (
            thermo.density * thermo.specific_heat_capacity_pressure
        )
        divergence = self.divergence_source(
            state.temperature,
            mass,
            temperature_rate,
            mass_fraction_rate,
            state.thermodynamic_pressure,
            thermodynamic_pressure_rate=thermodynamic_pressure_rate,
        )
        return LowMachReactiveEvaluation(
            mass,
            thermo,
            divergence,
            chemistry,
            self.formulation_id,
        )


__all__ = [
    "LowMachConstraintEvidence",
    "LowMachReactiveEvaluation",
    "LowMachReactiveState",
    "LowMachReactingFormulation",
]
