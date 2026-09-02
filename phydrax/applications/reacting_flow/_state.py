#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ...equations._hyperbolic_systems import (
    AbstractAdmissibleSystem,
    AbstractNormalReflectionSystem,
)
from ._thermodynamics import (
    IdealMixtureThermodynamicState,
    ReactingGasModel,
    TemperatureInversionEvidence,
)


class ReactiveConservedFields(StrictModule):
    density: Array
    independent_species_density: Array
    species_density: Array
    mass_fractions: Array
    momentum: Array
    velocity: Array
    total_energy_density: Array
    specific_internal_energy: Array


class ReactiveConservedEvidence(StrictModule):
    minimum_species_density: Array
    species_closure_defect: Array
    internal_energy: Array
    pressure: Array
    temperature: Array
    finite: Array
    species_positive: Array
    thermodynamics_successful: Array
    successful: Array
    layout_id: str = eqx.field(static=True)


class ReactivePrimitiveState(StrictModule):
    density: Array
    independent_mass_fractions: Array
    mass_fractions: Array
    velocity: Array
    pressure: Array
    temperature: Array
    thermodynamics: IdealMixtureThermodynamicState
    inversion: TemperatureInversionEvidence
    evidence: ReactiveConservedEvidence


class ReactiveConservedLayout(StrictModule, NonTrainableState):
    """Conserved layout with total density and S-1 independent species."""

    gas_model: ReactingGasModel
    dimension: int = eqx.field(static=True)
    density_floor: float = eqx.field(static=True)
    species_density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        gas_model: ReactingGasModel,
        dimension: int,
        /,
        *,
        density_floor: float = 1.0e-12,
        species_density_floor: float = 0.0,
        pressure_floor: float = 1.0e-12,
    ):
        if not isinstance(gas_model, ReactingGasModel):
            raise TypeError("gas_model must be ReactingGasModel.")
        dimension_ = int(dimension)
        floors = (
            float(density_floor),
            float(species_density_floor),
            float(pressure_floor),
        )
        if dimension_ not in (1, 2, 3):
            raise ValueError(
                "Reactive conserved layouts support one to three dimensions."
            )
        if (
            any(not isfinite(value) for value in floors)
            or floors[0] <= 0.0
            or floors[1] < 0.0
            or floors[2] <= 0.0
        ):
            raise ValueError("Reactive density/pressure floors are invalid.")
        independent_names = gas_model.schema.species_names[:-1]
        names = (
            "density",
            *(f"species_density_{name}" for name in independent_names),
            *(f"momentum_{axis}" for axis in range(dimension_)),
            "total_energy",
        )
        self.gas_model = gas_model
        self.dimension = dimension_
        self.density_floor = floors[0]
        self.species_density_floor = floors[1]
        self.pressure_floor = floors[2]
        self.component_names = names
        self.layout_id = canonical_fingerprint(
            {
                "kind": "reactive-conserved-layout",
                "gas_model": gas_model.model_id,
                "dimension": dimension_,
                "density_floor": floors[0],
                "species_density_floor": floors[1],
                "pressure_floor": floors[2],
                "components": list(names),
            }
        )

    @property
    def species_count(self) -> int:
        return self.gas_model.schema.species_count

    @property
    def independent_species_count(self) -> int:
        return self.species_count - 1

    @property
    def component_count(self) -> int:
        return len(self.component_names)

    @property
    def momentum_slice(self) -> slice:
        start = self.species_count
        return slice(start, start + self.dimension)

    def assemble(
        self,
        density: ArrayLike,
        independent_species_density: ArrayLike,
        momentum: ArrayLike,
        total_energy_density: ArrayLike,
        /,
    ) -> Array:
        density_ = jnp.asarray(density)
        species = jnp.asarray(independent_species_density, dtype=density_.dtype)
        momentum_ = jnp.asarray(momentum, dtype=density_.dtype)
        energy = jnp.asarray(total_energy_density, dtype=density_.dtype)
        cell_shape = density_.shape
        if species.shape != cell_shape + (self.independent_species_count,):
            raise ValueError("Independent species density has an invalid shape.")
        if momentum_.shape != cell_shape + (self.dimension,):
            raise ValueError("Momentum has an invalid shape.")
        if energy.shape != cell_shape:
            raise ValueError("Total energy density has an invalid shape.")
        return jnp.concatenate(
            (
                density_[..., None],
                species,
                momentum_,
                energy[..., None],
            ),
            axis=-1,
        )

    def split(self, conserved: ArrayLike, /) -> ReactiveConservedFields:
        state = jnp.asarray(conserved)
        if state.ndim < 1 or state.shape[-1] != self.component_count:
            raise ValueError("Conserved state has an invalid component axis.")
        density = state[..., 0]
        independent = state[..., 1 : self.species_count]
        dependent = density - jnp.sum(independent, axis=-1)
        species = jnp.concatenate((independent, dependent[..., None]), axis=-1)
        mass_fractions = species / density[..., None]
        momentum = state[..., self.momentum_slice]
        velocity = momentum / density[..., None]
        total_energy = state[..., -1]
        kinetic = 0.5 * jnp.sum(momentum * velocity, axis=-1)
        internal = (total_energy - kinetic) / density
        return ReactiveConservedFields(
            density,
            independent,
            species,
            mass_fractions,
            momentum,
            velocity,
            total_energy,
            internal,
        )

    def from_thermodynamic_state(
        self,
        density: ArrayLike,
        velocity: ArrayLike,
        temperature: ArrayLike,
        mass_fractions: ArrayLike,
        /,
    ) -> Array:
        density_ = jnp.asarray(density)
        velocity_ = jnp.asarray(velocity, dtype=density_.dtype)
        temperature_ = jnp.asarray(temperature, dtype=density_.dtype)
        mass = jnp.asarray(mass_fractions, dtype=density_.dtype)
        if velocity_.shape != density_.shape + (self.dimension,):
            raise ValueError("velocity must match density and layout dimension.")
        if mass.shape != density_.shape + (self.species_count,):
            raise ValueError("mass_fractions must match density and species count.")
        if temperature_.shape != density_.shape:
            raise ValueError("temperature must match density cell shape.")
        thermo = self.gas_model.evaluate_density(temperature_, density_, mass)
        momentum = density_[..., None] * velocity_
        specific_total = thermo.specific_internal_energy + 0.5 * jnp.sum(
            velocity_**2, axis=-1
        )
        return self.assemble(
            density_,
            density_[..., None] * mass[..., :-1],
            momentum,
            density_ * specific_total,
        )

    def primitive(self, conserved: ArrayLike, /) -> ReactivePrimitiveState:
        fields = self.split(conserved)
        thermo, inversion = self.gas_model.state_from_density_internal_energy(
            fields.density,
            fields.specific_internal_energy,
            fields.mass_fractions,
        )
        evidence = self.evidence_from_fields(fields, thermo, inversion)
        return ReactivePrimitiveState(
            fields.density,
            fields.mass_fractions[..., :-1],
            fields.mass_fractions,
            fields.velocity,
            thermo.pressure,
            thermo.temperature,
            thermo,
            inversion,
            evidence,
        )

    def evidence(self, conserved: ArrayLike, /) -> ReactiveConservedEvidence:
        fields = self.split(conserved)
        thermo, inversion = self.gas_model.state_from_density_internal_energy(
            fields.density,
            fields.specific_internal_energy,
            fields.mass_fractions,
        )
        return self.evidence_from_fields(fields, thermo, inversion)

    def evidence_from_fields(
        self,
        fields: ReactiveConservedFields,
        thermo: IdealMixtureThermodynamicState,
        inversion: TemperatureInversionEvidence,
        /,
    ) -> ReactiveConservedEvidence:
        species_sum = jnp.sum(fields.species_density, axis=-1)
        closure = species_sum - fields.density
        finite = (
            jnp.isfinite(fields.density)
            & jnp.all(jnp.isfinite(fields.species_density), axis=-1)
            & jnp.all(jnp.isfinite(fields.momentum), axis=-1)
            & jnp.isfinite(fields.total_energy_density)
        )
        positive = jnp.all(fields.species_density >= self.species_density_floor, axis=-1)
        tolerance = self.gas_model.composition_tolerance * jnp.maximum(
            jnp.abs(fields.density), 1.0
        )
        thermo_success = thermo.successful & inversion.successful
        successful = (
            finite
            & (fields.density > self.density_floor)
            & positive
            & (jnp.abs(closure) <= tolerance)
            & jnp.isfinite(fields.specific_internal_energy)
            & thermo_success
            & (thermo.pressure > self.pressure_floor)
        )
        return ReactiveConservedEvidence(
            jnp.min(fields.species_density, axis=-1),
            closure,
            fields.specific_internal_energy,
            thermo.pressure,
            thermo.temperature,
            finite,
            positive,
            thermo_success,
            successful,
            self.layout_id,
        )


class ReactiveEulerSystem(AbstractAdmissibleSystem, AbstractNormalReflectionSystem):
    """Thermally-perfect reactive Euler system for structured FV contracts."""

    layout: ReactiveConservedLayout

    def __init__(self, layout: ReactiveConservedLayout, /):
        if not isinstance(layout, ReactiveConservedLayout):
            raise TypeError("layout must be ReactiveConservedLayout.")
        self.layout = layout
        self.dimension = layout.dimension
        self.component_names = layout.component_names
        self.system_id = canonical_fingerprint(
            {
                "kind": "reactive-euler-system",
                "layout": layout.layout_id,
            }
        )

    def conserved_to_primitive(self, state: Array, /) -> Array:
        primitive = self.layout.primitive(state)
        return jnp.concatenate(
            (
                primitive.density[..., None],
                primitive.independent_mass_fractions,
                primitive.velocity,
                primitive.pressure[..., None],
            ),
            axis=-1,
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        if value.ndim < 1 or value.shape[-1] != self.component_count:
            raise ValueError("Reactive primitive state has an invalid component axis.")
        density = value[..., 0]
        independent = value[..., 1 : self.layout.species_count]
        dependent = 1.0 - jnp.sum(independent, axis=-1)
        mass = jnp.concatenate((independent, dependent[..., None]), axis=-1)
        velocity = value[..., self.layout.species_count : -1]
        pressure = value[..., -1]
        molar_mass = self.layout.gas_model.mixture_molar_mass(mass)
        gas_constant = UNIVERSAL_GAS_CONSTANT / molar_mass
        temperature = pressure / (density * gas_constant)
        return self.layout.from_thermodynamic_state(density, velocity, temperature, mass)

    def pressure(self, state: Array, /) -> Array:
        return self.layout.primitive(state).pressure

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        axis_ = int(axis)
        if axis_ < 0 or axis_ >= self.dimension:
            raise ValueError("Flux axis is outside the reactive-system dimension.")
        fields = self.layout.split(state)
        pressure = self.layout.primitive(state).pressure
        normal_velocity = fields.velocity[..., axis_]
        density_flux = fields.density * normal_velocity
        species_flux = fields.independent_species_density * normal_velocity[..., None]
        momentum_flux = fields.momentum * normal_velocity[..., None]
        momentum_flux = momentum_flux.at[..., axis_].add(pressure)
        energy_flux = (fields.total_energy_density + pressure) * normal_velocity
        return self.layout.assemble(
            density_flux,
            species_flux,
            momentum_flux,
            energy_flux,
        )

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        del args
        axis_ = int(axis)
        if axis_ < 0 or axis_ >= self.dimension:
            raise ValueError("Wave-speed axis is outside the system dimension.")
        left_primitive = self.layout.primitive(left)
        right_primitive = self.layout.primitive(right)
        left_speed = (
            jnp.abs(left_primitive.velocity[..., axis_])
            + left_primitive.thermodynamics.speed_of_sound
        )
        right_speed = (
            jnp.abs(right_primitive.velocity[..., axis_])
            + right_primitive.thermodynamics.speed_of_sound
        )
        return jnp.maximum(left_speed, right_speed)

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        axis_ = int(axis)
        left_primitive = self.layout.primitive(left)
        right_primitive = self.layout.primitive(right)
        sound = jnp.maximum(
            left_primitive.thermodynamics.speed_of_sound,
            right_primitive.thermodynamics.speed_of_sound,
        )
        lower = jnp.minimum(
            left_primitive.velocity[..., axis_] - sound,
            right_primitive.velocity[..., axis_] - sound,
        )
        upper = jnp.maximum(
            left_primitive.velocity[..., axis_] + sound,
            right_primitive.velocity[..., axis_] + sound,
        )
        return lower, upper

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        normal_ = jnp.asarray(normal)
        if normal_.ndim < 1 or normal_.shape[-1] != self.dimension:
            raise ValueError("normal must end in the system dimension.")
        left_primitive = self.layout.primitive(left)
        right_primitive = self.layout.primitive(right)
        left_velocity = jnp.sum(left_primitive.velocity * normal_, axis=-1)
        right_velocity = jnp.sum(right_primitive.velocity * normal_, axis=-1)
        sound = jnp.maximum(
            left_primitive.thermodynamics.speed_of_sound,
            right_primitive.thermodynamics.speed_of_sound,
        )
        return (
            jnp.minimum(left_velocity - sound, right_velocity - sound),
            jnp.maximum(left_velocity + sound, right_velocity + sound),
        )

    def admissible(self, state: Array, /) -> Array:
        fields = self.layout.split(state)
        gas = self.layout.gas_model
        molar_mass = gas.mixture_molar_mass(fields.mass_fractions)
        gas_constant = UNIVERSAL_GAS_CONSTANT / molar_mass
        pressure_temperature = self.layout.pressure_floor / (
            fields.density * gas_constant
        )
        minimum_temperature = jnp.maximum(
            jnp.asarray(
                gas.thermodynamics.minimum_temperature,
                dtype=fields.density.dtype,
            ),
            pressure_temperature,
        )
        maximum_temperature = jnp.full_like(
            fields.density,
            gas.thermodynamics.maximum_temperature,
        )
        _, _, _, minimum_energy, minimum_success = gas.caloric_properties(
            minimum_temperature,
            fields.mass_fractions,
        )
        _, _, _, maximum_energy, maximum_success = gas.caloric_properties(
            maximum_temperature,
            fields.mass_fractions,
        )
        return (
            jnp.isfinite(fields.density)
            & (fields.density > self.layout.density_floor)
            & jnp.all(jnp.isfinite(fields.species_density), axis=-1)
            & jnp.all(
                fields.species_density >= self.layout.species_density_floor,
                axis=-1,
            )
            & jnp.all(jnp.isfinite(fields.momentum), axis=-1)
            & jnp.isfinite(fields.total_energy_density)
            & gas.composition_valid(fields.mass_fractions)
            & minimum_success
            & maximum_success
            & (minimum_temperature <= maximum_temperature)
            & (fields.specific_internal_energy >= minimum_energy)
            & (fields.specific_internal_energy <= maximum_energy)
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        axis_ = int(axis)
        if axis_ < 0 or axis_ >= self.dimension:
            raise ValueError("Reflection axis is outside the system dimension.")
        return (
            jnp.asarray(state).at[..., self.layout.species_count + axis_].multiply(-1.0)
        )

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        value = jnp.asarray(state)
        normal_ = jnp.asarray(normal)
        if normal_.ndim < 1 or normal_.shape[-1] != self.dimension:
            raise ValueError("normal must end in the system dimension.")
        norm = jnp.sqrt(contract("...d,...d->...", normal_, normal_))
        unit = normal_ / norm[..., None]
        momentum = value[..., self.layout.momentum_slice]
        normal_momentum = contract("...d,...d->...", momentum, unit)
        reflected = momentum - 2.0 * normal_momentum[..., None] * unit
        return value.at[..., self.layout.momentum_slice].set(reflected)


__all__ = [
    "ReactiveConservedEvidence",
    "ReactiveConservedFields",
    "ReactiveConservedLayout",
    "ReactiveEulerSystem",
    "ReactivePrimitiveState",
]
