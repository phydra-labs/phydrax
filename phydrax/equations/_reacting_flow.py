#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._entropy_pair import ConvexEntropyPair
from ._hyperbolic_systems import (
    AbstractAdmissibleSystem,
    AbstractNormalReflectionSystem,
)


UNIVERSAL_GAS_CONSTANT = 8.31446261815324


class ReactingMixture(StrictModule, NonTrainableState):
    species_names: tuple[str, ...] = eqx.field(static=True)
    molecular_weights: Array
    heat_capacities: Array
    formation_energies: Array
    mixture_id: str = eqx.field(static=True)

    def __init__(
        self,
        species_names: Sequence[str],
        molecular_weights: ArrayLike,
        heat_capacities: ArrayLike,
        formation_energies: ArrayLike,
        /,
    ):
        names = tuple(str(value) for value in species_names)
        molecular = np.asarray(molecular_weights, dtype=float)
        heat = np.asarray(heat_capacities, dtype=float)
        formation = np.asarray(formation_energies, dtype=float)
        if (
            len(names) < 2
            or len(set(names)) != len(names)
            or any(not value for value in names)
            or molecular.shape != (len(names),)
            or heat.shape != molecular.shape
            or formation.shape != molecular.shape
            or np.any(molecular <= 0.0)
            or np.any(heat <= UNIVERSAL_GAS_CONSTANT / molecular)
            or np.any(~np.isfinite(formation))
        ):
            raise ValueError("Reacting mixture thermodynamics are invalid.")
        self.species_names = names
        self.molecular_weights = jnp.asarray(molecular)
        self.heat_capacities = jnp.asarray(heat)
        self.formation_energies = jnp.asarray(formation)
        self.mixture_id = canonical_fingerprint(
            {
                "kind": "reacting-mixture",
                "species": names,
                "molecular_weights": array_tree_fingerprint(molecular),
                "heat_capacities": array_tree_fingerprint(heat),
                "formation_energies": array_tree_fingerprint(formation),
            }
        )

    @property
    def species_count(self) -> int:
        return len(self.species_names)

    def gas_constants(self, /) -> Array:
        return UNIVERSAL_GAS_CONSTANT / self.molecular_weights

    def mixture_gas_constant(self, mass_fractions: ArrayLike, /) -> Array:
        return oe.contract(
            "...s,s->...",
            jnp.asarray(mass_fractions),
            self.gas_constants(),
            backend="jax",
        )

    def mixture_heat_capacity(self, mass_fractions: ArrayLike, /) -> Array:
        return oe.contract(
            "...s,s->...",
            jnp.asarray(mass_fractions),
            self.heat_capacities,
            backend="jax",
        )

    def mixture_internal_heat_capacity(self, mass_fractions: ArrayLike, /) -> Array:
        return self.mixture_heat_capacity(mass_fractions) - self.mixture_gas_constant(
            mass_fractions
        )


class ArrheniusReaction(StrictModule, NonTrainableState):
    stoichiometry: Array
    reactant_orders: Array
    pre_exponential: float = eqx.field(static=True)
    temperature_exponent: float = eqx.field(static=True)
    activation_temperature: float = eqx.field(static=True)
    reaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        stoichiometry: ArrayLike,
        reactant_orders: ArrayLike,
        /,
        *,
        pre_exponential: float,
        temperature_exponent: float = 0.0,
        activation_temperature: float,
    ):
        stoichiometry_ = np.asarray(stoichiometry, dtype=float)
        orders = np.asarray(reactant_orders, dtype=float)
        pre = float(pre_exponential)
        exponent = float(temperature_exponent)
        activation = float(activation_temperature)
        if (
            stoichiometry_.ndim != 1
            or orders.shape != stoichiometry_.shape
            or np.any(orders < 0.0)
            or pre <= 0.0
            or activation < 0.0
            or not np.isfinite(exponent)
        ):
            raise ValueError("Arrhenius reaction parameters are invalid.")
        self.stoichiometry = jnp.asarray(stoichiometry_)
        self.reactant_orders = jnp.asarray(orders)
        self.pre_exponential = pre
        self.temperature_exponent = exponent
        self.activation_temperature = activation
        self.reaction_id = canonical_fingerprint(
            {
                "kind": "arrhenius-reaction",
                "stoichiometry": array_tree_fingerprint(stoichiometry_),
                "orders": array_tree_fingerprint(orders),
                "pre_exponential": pre,
                "temperature_exponent": exponent,
                "activation_temperature": activation,
            }
        )


class ReactingEulerSystem(
    AbstractAdmissibleSystem,
    AbstractNormalReflectionSystem,
):
    mixture: ReactingMixture
    reactions: tuple[ArrheniusReaction, ...]
    density_floor: float = eqx.field(static=True)
    temperature_floor: float = eqx.field(static=True)

    def __init__(
        self,
        mixture: ReactingMixture,
        reactions: Sequence[ArrheniusReaction],
        dimension: int,
        /,
        *,
        density_floor: float = 1.0e-12,
        temperature_floor: float = 1.0e-10,
    ):
        if not isinstance(mixture, ReactingMixture):
            raise TypeError("mixture must be ReactingMixture.")
        reactions_ = tuple(reactions)
        if any(
            not isinstance(value, ArrheniusReaction)
            or value.stoichiometry.shape != (mixture.species_count,)
            for value in reactions_
        ):
            raise ValueError("Reaction mechanisms must match mixture species.")
        for reaction in reactions_:
            mass_defect = float(
                np.asarray(reaction.stoichiometry @ mixture.molecular_weights)
            )
            if abs(mass_defect) > 1.0e-10:
                raise ValueError("Reaction stoichiometry must conserve mass.")
        dimension_ = int(dimension)
        if dimension_ not in (1, 2, 3):
            raise ValueError("Reacting Euler dimension must be one, two, or three.")
        names = (
            "density",
            *(f"momentum_{axis}" for axis in range(dimension_)),
            "total_energy",
            *(f"species_{name}" for name in mixture.species_names[:-1]),
        )
        super().__init__(
            dimension_,
            names,
            system_id=canonical_fingerprint(
                {
                    "kind": "reacting-euler-system",
                    "mixture": mixture.mixture_id,
                    "reactions": tuple(value.reaction_id for value in reactions_),
                    "dimension": dimension_,
                }
            ),
        )
        self.mixture = mixture
        self.reactions = reactions_
        self.density_floor = float(density_floor)
        self.temperature_floor = float(temperature_floor)

    def mass_fractions(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        independent = value[..., self.dimension + 2 :] / density[..., None]
        final = 1.0 - jnp.sum(independent, axis=-1, keepdims=True)
        return jnp.concatenate((independent, final), axis=-1)

    def temperature(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum = value[..., 1 : 1 + self.dimension]
        kinetic = (
            0.5
            * oe.contract("...d,...d->...", momentum, momentum, backend="jax")
            / density**2
        )
        fractions = self.mass_fractions(value)
        formation = oe.contract(
            "...s,s->...", fractions, self.mixture.formation_energies, backend="jax"
        )
        internal = value[..., self.dimension + 1] / density - kinetic
        return (internal - formation) / self.mixture.mixture_internal_heat_capacity(
            fractions
        )

    def pressure(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        return (
            value[..., 0]
            * self.mixture.mixture_gas_constant(self.mass_fractions(value))
            * self.temperature(value)
        )

    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        fractions = self.mass_fractions(value)
        return (
            jnp.isfinite(value).all(axis=-1)
            & (value[..., 0] > self.density_floor)
            & jnp.all(fractions > 0.0, axis=-1)
            & (self.temperature(value) > self.temperature_floor)
            & (self.pressure(value) > 0.0)
        )

    def primitive_to_conserved(self, primitive: ArrayLike, /) -> Array:
        value = jnp.asarray(primitive)
        expected = self.dimension + 2 + self.mixture.species_count
        if value.shape[-1] != expected:
            raise ValueError("Reacting primitive shape is incompatible.")
        density = value[..., 0]
        velocity = value[..., 1 : 1 + self.dimension]
        temperature = value[..., 1 + self.dimension]
        fractions = value[..., 2 + self.dimension :]
        internal = self.mixture.mixture_internal_heat_capacity(
            fractions
        ) * temperature + oe.contract(
            "...s,s->...",
            fractions,
            self.mixture.formation_energies,
            backend="jax",
        )
        energy = density * (
            internal
            + 0.5 * oe.contract("...d,...d->...", velocity, velocity, backend="jax")
        )
        return jnp.concatenate(
            (
                density[..., None],
                density[..., None] * velocity,
                energy[..., None],
                density[..., None] * fractions[..., :-1],
            ),
            axis=-1,
        )

    def conserved_to_primitive(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        return jnp.concatenate(
            (
                density[..., None],
                value[..., 1 : 1 + self.dimension] / density[..., None],
                self.temperature(value)[..., None],
                self.mass_fractions(value),
            ),
            axis=-1,
        )

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        lower = []
        upper = []
        for state in (left, right):
            fractions = self.mass_fractions(state)
            gas = self.mixture.mixture_gas_constant(fractions)
            cp = self.mixture.mixture_heat_capacity(fractions)
            gamma = cp / (cp - gas)
            velocity = state[..., 1 + int(axis)] / state[..., 0]
            sound = jnp.sqrt(gamma * self.pressure(state) / state[..., 0])
            lower.append(velocity - sound)
            upper.append(velocity + sound)
        return jnp.minimum(*lower), jnp.maximum(*upper)

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
        lower = []
        upper = []
        for state in (left, right):
            fractions = self.mass_fractions(state)
            gas = self.mixture.mixture_gas_constant(fractions)
            cp = self.mixture.mixture_heat_capacity(fractions)
            gamma = cp / (cp - gas)
            velocity = state[..., 1 : 1 + self.dimension] / state[..., 0, None]
            normal_velocity = oe.contract(
                "...d,...d->...", velocity, normal_, backend="jax"
            )
            normal_norm = jnp.sqrt(
                oe.contract("...d,...d->...", normal_, normal_, backend="jax")
            )
            sound = jnp.sqrt(gamma * self.pressure(state) / state[..., 0])
            lower.append(normal_velocity - normal_norm * sound)
            upper.append(normal_velocity + normal_norm * sound)
        return jnp.minimum(*lower), jnp.maximum(*upper)

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum = value[..., 1 : 1 + self.dimension]
        velocity = momentum / density[..., None]
        pressure = self.pressure(value)
        flux = value * velocity[..., axis, None]
        flux = flux.at[..., 1 : 1 + self.dimension].add(
            pressure[..., None] * jax.nn.one_hot(axis, self.dimension)
        )
        return flux.at[..., self.dimension + 1].add(pressure * velocity[..., axis])

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        del args
        speeds = []
        for state in (left, right):
            fractions = self.mass_fractions(state)
            gas = self.mixture.mixture_gas_constant(fractions)
            cp = self.mixture.mixture_heat_capacity(fractions)
            gamma = cp / (cp - gas)
            velocity = state[..., 1 + axis] / state[..., 0]
            sound = jnp.sqrt(gamma * self.pressure(state) / state[..., 0])
            speeds.append(jnp.abs(velocity) + sound)
        return jnp.maximum(*speeds)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return jnp.asarray(state).at[..., 1 + int(axis)].multiply(-1.0)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        value = jnp.asarray(state)
        unit = jnp.asarray(normal)
        unit = (
            unit
            / jnp.sqrt(oe.contract("...d,...d->...", unit, unit, backend="jax"))[
                ..., None
            ]
        )
        momentum = value[..., 1 : 1 + self.dimension]
        reflected = (
            momentum
            - 2.0
            * oe.contract("...d,...d->...", momentum, unit, backend="jax")[..., None]
            * unit
        )
        return value.at[..., 1 : 1 + self.dimension].set(reflected)

    def reaction_source(self, state: ArrayLike, args: Any = None, /) -> Array:
        del args
        value = jnp.asarray(state)
        density = value[..., 0]
        fractions = self.mass_fractions(value)
        temperature = self.temperature(value)
        concentrations = density[..., None] * fractions / self.mixture.molecular_weights
        molar_rate = jnp.zeros_like(fractions)
        for reaction in self.reactions:
            progress = (
                reaction.pre_exponential
                * temperature**reaction.temperature_exponent
                * jnp.exp(-reaction.activation_temperature / temperature)
                * jnp.prod(
                    jnp.maximum(concentrations, 1.0e-30) ** reaction.reactant_orders,
                    axis=-1,
                )
            )
            molar_rate = molar_rate + progress[..., None] * reaction.stoichiometry
        mass_rate = molar_rate * self.mixture.molecular_weights
        source = jnp.zeros_like(value)
        return source.at[..., self.dimension + 2 :].set(mass_rate[..., :-1])

    def reaction_source_term(
        self,
        time: Array,
        state: Array,
        coordinates: Array,
        args: Any = None,
        /,
    ) -> Array:
        del time, coordinates
        return self.reaction_source(state, args)


def reacting_mixture_entropy_pair(system: ReactingEulerSystem, /) -> ConvexEntropyPair:
    if not isinstance(system, ReactingEulerSystem):
        raise TypeError("system must be ReactingEulerSystem.")

    def entropy(state):
        density = state[..., 0]
        fractions = system.mass_fractions(state)
        temperature = system.temperature(state)
        species_density = density[..., None] * fractions
        species_entropy = system.mixture.heat_capacities * jnp.log(
            temperature[..., None]
        ) - system.mixture.gas_constants() * jnp.log(
            jnp.maximum(species_density, 1.0e-30)
        )
        return -density * oe.contract(
            "...s,...s->...", fractions, species_entropy, backend="jax"
        )

    point_entropy = lambda point: entropy(point)
    entropy_variables = jax.vmap(jax.grad(point_entropy))

    def variables(state):
        value = jnp.asarray(state)
        flat = value.reshape((-1, value.shape[-1]))
        return entropy_variables(flat).reshape(value.shape)

    def entropy_flux(state, axis, args):
        del args
        return entropy(state) * state[..., 1 + int(axis)] / state[..., 0]

    return ConvexEntropyPair(
        system,
        entropy,
        variables,
        entropy_flux,
        system.admissible,
        entropy_id="reacting-mixture-thermodynamic-entropy",
    )


__all__ = [
    "ArrheniusReaction",
    "ReactingEulerSystem",
    "ReactingMixture",
    "reacting_mixture_entropy_pair",
]
