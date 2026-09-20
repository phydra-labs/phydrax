#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import DSMCParticleState, DSMCSpeciesPlan


class DSMCReactionChannelPlan(StrictModule, NonTrainableState):
    """A bounded, endothermic two-to-two reaction channel."""

    reactant_species: Array
    product_species: Array
    threshold_energy: float = eqx.field(static=True)
    probability: float = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        reactant_species: tuple[int, int],
        product_species: tuple[int, int],
        /,
        *,
        threshold_energy: float,
        probability: float,
    ) -> None:
        reactants = np.asarray(reactant_species, dtype=np.int32)
        products = np.asarray(product_species, dtype=np.int32)
        threshold = float(threshold_energy)
        chance = float(probability)
        if (
            reactants.shape != (2,)
            or products.shape != (2,)
            or np.any(reactants < 0)
            or np.any(products < 0)
            or not np.isfinite(threshold)
            or threshold < 0.0
            or not np.isfinite(chance)
            or not 0.0 <= chance <= 1.0
        ):
            raise ValueError(
                "DSMC reaction channel indices, threshold, or probability are invalid."
            )
        self.reactant_species = jnp.asarray(reactants)
        self.product_species = jnp.asarray(products)
        self.threshold_energy = threshold
        self.probability = chance
        self.channel_id = canonical_fingerprint(
            {
                "kind": "dsmc-reaction-channel",
                "reactants": tuple(reactants),
                "products": tuple(products),
                "threshold_energy": threshold,
                "probability": chance,
            }
        )


class DSMCInternalReactionEventResult(StrictModule):
    state: DSMCParticleState
    reacted: Array
    relaxed: Array
    energy_defect: Array
    chemical_energy_consumed: Array
    finite: Array


class DSMCInternalReactionResult(StrictModule):
    state: DSMCParticleState
    reacted: Array
    relaxed: Array
    energy_defect: Array
    chemical_energy_consumed: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DSMCInternalReactionPlan(StrictModule, NonTrainableState):
    """Accepted-pair rotational relaxation and bounded two-to-two chemistry.

    Vibrational energy is deliberately frozen. Rotational redistribution samples
    the microcanonical degree-of-freedom partition with beta variates.
    """

    species: DSMCSpeciesPlan
    channels: tuple[DSMCReactionChannelPlan, ...]
    rotational_relaxation_probability: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: DSMCSpeciesPlan,
        channels: tuple[DSMCReactionChannelPlan, ...] = (),
        /,
        *,
        rotational_relaxation_probability: ArrayLike,
    ) -> None:
        rotational = np.asarray(rotational_relaxation_probability, dtype=np.float64)
        channels_ = tuple(channels)
        invalid_channels = any(
            not isinstance(value, DSMCReactionChannelPlan) for value in channels_
        )
        if (
            not isinstance(species, DSMCSpeciesPlan)
            or rotational.shape != (species.species_count,)
            or np.any(~np.isfinite(rotational))
            or np.any((rotational < 0.0) | (rotational > 1.0))
            or invalid_channels
            or any(
                int(np.max(np.asarray(value.product_species))) >= species.species_count
                or int(np.max(np.asarray(value.reactant_species)))
                >= species.species_count
                for value in channels_
            )
        ):
            raise ValueError("DSMC internal relaxation or reaction data are invalid.")
        if channels_ and not species.element_names:
            raise ValueError(
                "Reactive DSMC species require explicit elemental composition."
            )
        masses = np.asarray(species.molecular_masses)
        charges = np.asarray(species.charges)
        composition = np.asarray(species.elemental_composition)
        for channel in channels_:
            reactants = np.asarray(channel.reactant_species)
            products = np.asarray(channel.product_species)
            mass_scale = max(float(np.sum(masses[reactants])), 1.0e-300)
            if not np.isclose(
                np.sum(masses[reactants]),
                np.sum(masses[products]),
                rtol=256.0 * np.finfo(np.float64).eps,
                atol=256.0 * np.finfo(np.float64).eps * mass_scale,
            ):
                raise ValueError("A DSMC reaction channel does not conserve mass.")
            if not np.isclose(np.sum(charges[reactants]), np.sum(charges[products])):
                raise ValueError("A DSMC reaction channel does not conserve charge.")
            if not np.array_equal(
                np.sum(composition[reactants], axis=0),
                np.sum(composition[products], axis=0),
            ):
                raise ValueError("A DSMC reaction channel does not conserve elements.")
        self.species = species
        self.channels = channels_
        self.rotational_relaxation_probability = jnp.asarray(rotational)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-internal-reaction",
                "species": species.plan_id,
                "channels": tuple(value.channel_id for value in channels_),
                "rotational": array_tree_fingerprint(rotational),
                "vibrational": "frozen",
            }
        )

    def apply_one(
        self,
        state: DSMCParticleState,
        first_index: ArrayLike,
        second_index: ArrayLike,
        accepted_collision: ArrayLike,
        key: PRNGKeyArray,
        /,
    ) -> DSMCInternalReactionEventResult:
        first = jnp.asarray(first_index, dtype=jnp.int32)
        second = jnp.asarray(second_index, dtype=jnp.int32)
        enabled = jnp.asarray(accepted_collision, dtype=jnp.bool_)
        if (
            first.shape != ()
            or second.shape != ()
            or enabled.shape != ()
            or jax.random.key_data(key).shape != (2,)
        ):
            raise ValueError(
                "One DSMC internal event requires scalar indices and one key."
            )
        capacity = state.capacity
        in_bounds = (
            (first >= 0) & (first < capacity) & (second >= 0) & (second < capacity)
        )
        first_safe = jnp.clip(first, 0, capacity - 1)
        second_safe = jnp.clip(second, 0, capacity - 1)
        valid = (
            enabled
            & in_bounds
            & (first != second)
            & state.active[first_safe]
            & state.active[second_safe]
            & (state.cell_id[first_safe] == state.cell_id[second_safe])
        )
        species_first = state.species_index[first_safe]
        species_second = state.species_index[second_safe]
        mass_first = self.species.molecular_masses[species_first]
        mass_second = self.species.molecular_masses[species_second]
        velocity_first_before = state.velocity[first_safe]
        velocity_second_before = state.velocity[second_safe]
        momentum = (
            mass_first * velocity_first_before + mass_second * velocity_second_before
        )
        total_mass_before = mass_first + mass_second
        relative = velocity_first_before - velocity_second_before
        relative_speed_squared = jnp.sum(relative * relative)
        reduced_mass = mass_first * mass_second / total_mass_before
        kinetic_before = 0.5 * reduced_mass * relative_speed_squared
        rotational_first_before = state.rotational_energy[first_safe]
        rotational_second_before = state.rotational_energy[second_safe]
        rotational_before = rotational_first_before + rotational_second_before
        vibrational_before = (
            state.vibrational_energy[first_safe] + state.vibrational_energy[second_safe]
        )
        available_before = kinetic_before + rotational_before

        reacted = jnp.asarray(False)
        product_first = species_first
        product_second = species_second
        threshold = jnp.asarray(0.0, dtype=state.velocity.dtype)
        for channel_index, channel in enumerate(self.channels):
            matches = (
                (species_first == channel.reactant_species[0])
                & (species_second == channel.reactant_species[1])
            ) | (
                (species_first == channel.reactant_species[1])
                & (species_second == channel.reactant_species[0])
            )
            selected = (
                valid
                & matches
                & ~reacted
                & (available_before >= channel.threshold_energy)
                & (
                    jax.random.uniform(
                        jax.random.fold_in(key, channel_index),
                        (),
                        dtype=state.velocity.dtype,
                    )
                    < channel.probability
                )
            )
            product_first = jnp.where(selected, channel.product_species[0], product_first)
            product_second = jnp.where(
                selected, channel.product_species[1], product_second
            )
            threshold = jnp.where(selected, channel.threshold_energy, threshold)
            reacted = reacted | selected

        remaining = jnp.maximum(available_before - threshold, 0.0)
        first_dof = self.species.rotational_degrees[product_first]
        second_dof = self.species.rotational_degrees[product_second]
        rotational_dof = first_dof + second_dof
        relaxation_probability = 0.5 * (
            self.rotational_relaxation_probability[product_first]
            + self.rotational_relaxation_probability[product_second]
        )
        relaxation_selected = (
            valid
            & (rotational_dof > 0.0)
            & (
                jax.random.uniform(
                    jax.random.fold_in(key, len(self.channels) + 1),
                    (),
                    dtype=state.velocity.dtype,
                )
                < relaxation_probability
            )
        )
        redistributed = reacted | relaxation_selected
        relaxed = redistributed & (rotational_dof > 0.0)
        translational_dof = jnp.asarray(
            state.velocity.shape[-1], dtype=state.velocity.dtype
        )
        rotational_fraction = jax.random.beta(
            jax.random.fold_in(key, len(self.channels) + 2),
            jnp.maximum(0.5 * rotational_dof, 0.5),
            jnp.maximum(0.5 * translational_dof, 0.5),
            dtype=state.velocity.dtype,
        )
        rotational_fraction = jnp.where(rotational_dof > 0.0, rotational_fraction, 0.0)
        redistributed_rotational = remaining * rotational_fraction
        rotational_total = jnp.where(
            redistributed, redistributed_rotational, rotational_before
        )
        kinetic_after = jnp.where(
            redistributed,
            remaining - redistributed_rotational,
            kinetic_before,
        )
        split = jax.random.beta(
            jax.random.fold_in(key, len(self.channels) + 3),
            jnp.maximum(0.5 * first_dof, 0.5),
            jnp.maximum(0.5 * second_dof, 0.5),
            dtype=state.velocity.dtype,
        )
        split = jnp.where(first_dof <= 0.0, 0.0, split)
        split = jnp.where(second_dof <= 0.0, 1.0, split)
        rotational_first = jnp.where(
            redistributed, rotational_total * split, rotational_first_before
        )
        rotational_second = jnp.where(
            redistributed,
            rotational_total * (1.0 - split),
            rotational_second_before,
        )

        new_mass_first = self.species.molecular_masses[product_first]
        new_mass_second = self.species.molecular_masses[product_second]
        new_total_mass = new_mass_first + new_mass_second
        new_reduced_mass = new_mass_first * new_mass_second / new_total_mass
        random_direction = jax.random.normal(
            jax.random.fold_in(key, len(self.channels) + 4),
            (state.velocity.shape[-1],),
            dtype=state.velocity.dtype,
        )
        random_direction = random_direction / jnp.maximum(
            jnp.sqrt(jnp.sum(random_direction**2)),
            jnp.finfo(state.velocity.dtype).tiny,
        )
        direction = relative / jnp.maximum(
            jnp.sqrt(relative_speed_squared), jnp.finfo(state.velocity.dtype).tiny
        )
        direction = jnp.where(relative_speed_squared > 0.0, direction, random_direction)
        new_relative_speed = jnp.sqrt(
            2.0
            * kinetic_after
            / jnp.maximum(new_reduced_mass, jnp.finfo(state.velocity.dtype).tiny)
        )
        new_relative = new_relative_speed * direction
        new_center = momentum / new_total_mass
        velocity_first = new_center + new_mass_second / new_total_mass * new_relative
        velocity_second = new_center - new_mass_first / new_total_mass * new_relative
        transformed = redistributed
        velocity_first = jnp.where(transformed, velocity_first, velocity_first_before)
        velocity_second = jnp.where(transformed, velocity_second, velocity_second_before)
        product_first = jnp.where(transformed, product_first, species_first)
        product_second = jnp.where(transformed, product_second, species_second)
        rotational_first = jnp.where(
            transformed, rotational_first, rotational_first_before
        )
        rotational_second = jnp.where(
            transformed, rotational_second, rotational_second_before
        )

        velocity = state.velocity.at[first_safe].set(velocity_first)
        velocity = velocity.at[second_safe].set(velocity_second)
        species_index = state.species_index.at[first_safe].set(product_first)
        species_index = species_index.at[second_safe].set(product_second)
        rotational_energy = state.rotational_energy.at[first_safe].set(rotational_first)
        rotational_energy = rotational_energy.at[second_safe].set(rotational_second)
        updated = DSMCParticleState(
            state.position,
            velocity,
            species_index,
            rotational_energy,
            state.vibrational_energy,
            state.statistical_weight,
            state.cell_id,
            state.active,
            state.incarnation,
        )
        relative_after = velocity_first - velocity_second
        kinetic_after_check = 0.5 * new_reduced_mass * jnp.sum(relative_after**2)
        energy_after = (
            kinetic_after_check
            + rotational_first
            + rotational_second
            + vibrational_before
            + threshold
        )
        energy_before = available_before + vibrational_before
        defect = jnp.where(transformed, energy_after - energy_before, 0.0)
        finite = (~valid) | (
            jnp.all(jnp.isfinite(velocity_first))
            & jnp.all(jnp.isfinite(velocity_second))
            & jnp.isfinite(defect)
            & jnp.isfinite(rotational_first)
            & jnp.isfinite(rotational_second)
        )
        return DSMCInternalReactionEventResult(
            updated,
            reacted,
            relaxed,
            defect,
            jnp.where(reacted, threshold, 0.0),
            finite,
        )

    def apply(
        self,
        state: DSMCParticleState,
        first_indices: ArrayLike,
        second_indices: ArrayLike,
        accepted_collisions: ArrayLike,
        keys: ArrayLike,
        /,
    ) -> DSMCInternalReactionResult:
        first = jnp.asarray(first_indices, dtype=jnp.int32)
        second = jnp.asarray(second_indices, dtype=jnp.int32)
        accepted = jnp.asarray(accepted_collisions, dtype=jnp.bool_)
        key_values = keys
        if (
            first.ndim != 1
            or second.shape != first.shape
            or accepted.shape != first.shape
            or jax.random.key_data(key_values).shape != (first.size, 2)
        ):
            raise ValueError("DSMC internal event arrays are incompatible.")

        def body(particles, event):
            first_, second_, accepted_, key_ = event
            result = self.apply_one(
                particles,
                first_,
                second_,
                accepted_,
                key_,
            )
            return result.state, (
                result.reacted,
                result.relaxed,
                result.energy_defect,
                result.chemical_energy_consumed,
                result.finite,
            )

        updated, diagnostics = jax.lax.scan(
            body,
            state,
            (first, second, accepted, key_values),
        )
        reacted, relaxed, defect, consumed, finite = diagnostics
        scale = jnp.maximum(jnp.max(jnp.abs(consumed)), 1.0)
        tolerance = 512.0 * jnp.finfo(defect.dtype).eps * scale
        successful = jnp.all(finite) & jnp.all(jnp.abs(defect) <= tolerance)
        return DSMCInternalReactionResult(
            updated,
            reacted,
            relaxed,
            defect,
            consumed,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "DSMCInternalReactionEventResult",
    "DSMCInternalReactionPlan",
    "DSMCInternalReactionResult",
    "DSMCReactionChannelPlan",
]
