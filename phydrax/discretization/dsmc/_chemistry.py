#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import DSMCParticleState, DSMCSpeciesPlan


class DSMCReactionChannelPlan(StrictModule, NonTrainableState):
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
    ):
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
                "reactants": tuple(int(x) for x in reactants),
                "products": tuple(int(x) for x in products),
                "threshold_energy": threshold,
                "probability": chance,
            }
        )


class DSMCInternalReactionResult(StrictModule):
    state: DSMCParticleState
    reacted: Array
    relaxed: Array
    energy_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DSMCInternalReactionPlan(StrictModule, NonTrainableState):
    """Pairwise Larsen-Borgnakke redistribution plus bounded two-to-two chemistry."""

    species: DSMCSpeciesPlan
    channels: tuple[DSMCReactionChannelPlan, ...]
    rotational_relaxation_probability: Array
    vibrational_relaxation_probability: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: DSMCSpeciesPlan,
        channels: tuple[DSMCReactionChannelPlan, ...] = (),
        /,
        *,
        rotational_relaxation_probability: ArrayLike,
        vibrational_relaxation_probability: ArrayLike,
    ):
        rotational = np.asarray(rotational_relaxation_probability, dtype=float)
        vibrational = np.asarray(vibrational_relaxation_probability, dtype=float)
        channels_ = tuple(channels)
        if (
            not isinstance(species, DSMCSpeciesPlan)
            or rotational.shape != (species.species_count,)
            or vibrational.shape != rotational.shape
            or np.any(~np.isfinite(rotational))
            or np.any((rotational < 0.0) | (rotational > 1.0))
            or np.any(~np.isfinite(vibrational))
            or np.any((vibrational < 0.0) | (vibrational > 1.0))
            or any(not isinstance(value, DSMCReactionChannelPlan) for value in channels_)
            or any(
                int(jnp.max(value.product_species)) >= species.species_count
                or int(jnp.max(value.reactant_species)) >= species.species_count
                for value in channels_
            )
        ):
            raise ValueError("DSMC internal relaxation or reaction data are invalid.")
        self.species = species
        self.channels = channels_
        self.rotational_relaxation_probability = jnp.asarray(rotational)
        self.vibrational_relaxation_probability = jnp.asarray(vibrational)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-internal-reaction",
                "species": species.plan_id,
                "channels": tuple(value.channel_id for value in channels_),
                "rotational": array_tree_fingerprint(
                    self.rotational_relaxation_probability
                ),
                "vibrational": array_tree_fingerprint(
                    self.vibrational_relaxation_probability
                ),
            }
        )

    def apply(
        self,
        state: DSMCParticleState,
        first_indices: ArrayLike,
        second_indices: ArrayLike,
        uniforms: ArrayLike,
        /,
    ) -> DSMCInternalReactionResult:
        first = jnp.asarray(first_indices, dtype=jnp.int32)
        second = jnp.asarray(second_indices, dtype=jnp.int32)
        random = jnp.asarray(uniforms, dtype=state.velocity.dtype)
        event_count = first.size
        if (
            first.shape != second.shape
            or first.ndim != 1
            or random.shape != (event_count, 4)
        ):
            raise ValueError("DSMC internal event indices or random draws are invalid.")
        species_first = state.species_index[first]
        species_second = state.species_index[second]
        mass_first = self.species.molecular_masses[species_first]
        mass_second = self.species.molecular_masses[species_second]
        relative = state.velocity[first] - state.velocity[second]
        relative_speed_squared = jnp.sum(relative * relative, axis=-1)
        reduced_mass = mass_first * mass_second / (mass_first + mass_second)
        kinetic = 0.5 * reduced_mass * relative_speed_squared
        internal_before = (
            state.rotational_energy[first]
            + state.rotational_energy[second]
            + state.vibrational_energy[first]
            + state.vibrational_energy[second]
        )
        total_available = kinetic + internal_before
        reacted = jnp.zeros((event_count,), dtype=bool)
        product_first = species_first
        product_second = species_second
        threshold_used = jnp.zeros_like(total_available)
        for channel_index, channel in enumerate(self.channels):
            matches = (
                (species_first == channel.reactant_species[0])
                & (species_second == channel.reactant_species[1])
            ) | (
                (species_first == channel.reactant_species[1])
                & (species_second == channel.reactant_species[0])
            )
            selected = (
                matches
                & ~reacted
                & (total_available >= channel.threshold_energy)
                & (random[:, 0] < channel.probability)
            )
            product_first = jnp.where(selected, channel.product_species[0], product_first)
            product_second = jnp.where(
                selected, channel.product_species[1], product_second
            )
            threshold_used = jnp.where(selected, channel.threshold_energy, threshold_used)
            reacted = reacted | selected
        remaining = jnp.maximum(total_available - threshold_used, 0.0)
        rotational_probability = 0.5 * (
            self.rotational_relaxation_probability[product_first]
            + self.rotational_relaxation_probability[product_second]
        )
        vibrational_probability = 0.5 * (
            self.vibrational_relaxation_probability[product_first]
            + self.vibrational_relaxation_probability[product_second]
        )
        rotational_active = random[:, 1] < rotational_probability
        vibrational_active = random[:, 2] < vibrational_probability
        rotational_fraction = jnp.where(rotational_active, 0.25 * random[:, 3], 0.0)
        vibrational_fraction = jnp.where(
            vibrational_active, 0.25 * (1.0 - random[:, 3]), 0.0
        )
        translational_fraction = jnp.maximum(
            1.0 - rotational_fraction - vibrational_fraction, 0.0
        )
        rotational_total = remaining * rotational_fraction
        vibrational_total = remaining * vibrational_fraction
        kinetic_after = remaining * translational_fraction
        new_mass_first = self.species.molecular_masses[product_first]
        new_mass_second = self.species.molecular_masses[product_second]
        new_reduced = (
            new_mass_first * new_mass_second / (new_mass_first + new_mass_second)
        )
        new_relative_speed = jnp.sqrt(
            2.0
            * kinetic_after
            / jnp.maximum(new_reduced, jnp.finfo(state.velocity.dtype).tiny)
        )
        direction = relative / jnp.maximum(
            jnp.sqrt(relative_speed_squared)[..., None],
            jnp.finfo(state.velocity.dtype).tiny,
        )
        new_relative = new_relative_speed[:, None] * direction
        center = (
            mass_first[:, None] * state.velocity[first]
            + mass_second[:, None] * state.velocity[second]
        ) / (mass_first + mass_second)[:, None]
        velocity_first = (
            center
            + new_mass_second[:, None]
            / (new_mass_first + new_mass_second)[:, None]
            * new_relative
        )
        velocity_second = (
            center
            - new_mass_first[:, None]
            / (new_mass_first + new_mass_second)[:, None]
            * new_relative
        )
        velocity = (
            state.velocity.at[first].set(velocity_first).at[second].set(velocity_second)
        )
        species_index = (
            state.species_index.at[first]
            .set(product_first)
            .at[second]
            .set(product_second)
        )
        rotational_energy = (
            state.rotational_energy.at[first]
            .set(0.5 * rotational_total)
            .at[second]
            .set(0.5 * rotational_total)
        )
        vibrational_energy = (
            state.vibrational_energy.at[first]
            .set(0.5 * vibrational_total)
            .at[second]
            .set(0.5 * vibrational_total)
        )
        energy_after = (
            kinetic_after + rotational_total + vibrational_total + threshold_used
        )
        energy_defect = energy_after - total_available
        updated = DSMCParticleState(
            state.position,
            velocity,
            species_index,
            rotational_energy,
            vibrational_energy,
            state.statistical_weight,
            state.cell_id,
            state.active,
            state.incarnation,
        )
        finite = jnp.all(jnp.isfinite(velocity)) & jnp.all(jnp.isfinite(energy_defect))
        tolerance = (
            512.0 * jnp.finfo(energy_defect.dtype).eps * jnp.maximum(total_available, 1.0)
        )
        successful = finite & jnp.all(jnp.abs(energy_defect) <= tolerance)
        return DSMCInternalReactionResult(
            updated,
            reacted,
            rotational_active | vibrational_active,
            energy_defect,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "DSMCInternalReactionPlan",
    "DSMCInternalReactionResult",
    "DSMCReactionChannelPlan",
]
