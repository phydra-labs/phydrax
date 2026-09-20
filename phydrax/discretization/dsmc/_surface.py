#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import DSMCParticleState, DSMCSpeciesPlan


class DSMCSurfaceReactionPlan(StrictModule, NonTrainableState):
    product_species: Array
    probabilities: Array
    reaction_energies: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        product_species: ArrayLike,
        probabilities: ArrayLike,
        reaction_energies: ArrayLike,
        /,
    ) -> None:
        products = np.asarray(product_species, dtype=np.int32)
        probabilities_ = np.asarray(probabilities, dtype=np.float64)
        energies = np.asarray(reaction_energies, dtype=np.float64)
        if (
            products.ndim != 1
            or probabilities_.shape != products.shape
            or energies.shape != products.shape
            or np.any(products < -1)
            or np.any(~np.isfinite(probabilities_))
            or np.any((probabilities_ < 0.0) | (probabilities_ > 1.0))
            or np.any(~np.isfinite(energies))
        ):
            raise ValueError(
                "DSMC surface reaction products, probabilities, or energies are invalid."
            )
        self.product_species = jnp.asarray(products)
        self.probabilities = jnp.asarray(probabilities_)
        self.reaction_energies = jnp.asarray(energies)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-surface-reaction",
                "products": array_tree_fingerprint(products),
                "probabilities": array_tree_fingerprint(probabilities_),
                "energies": array_tree_fingerprint(energies),
            }
        )


class DSMCSurfaceResult(StrictModule):
    state: DSMCParticleState
    mass_to_surface: Array
    reacted: Array
    absorbed: Array
    momentum_to_surface: Array
    energy_to_surface: Array
    total_mass: Array
    total_force_impulse: Array
    total_heat: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DSMCSurfaceInteractionPlan(StrictModule, NonTrainableState):
    """Specular or fully diffuse Maxwell gas-surface interaction."""

    species: DSMCSpeciesPlan
    wall_velocity: Array
    kind: Literal["specular", "diffuse"] = eqx.field(static=True)
    wall_temperature: float = eqx.field(static=True)
    reaction: DSMCSurfaceReactionPlan | None
    boltzmann_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: DSMCSpeciesPlan,
        kind: Literal["specular", "diffuse"],
        wall_velocity: ArrayLike,
        /,
        *,
        wall_temperature: float,
        reaction: DSMCSurfaceReactionPlan | None = None,
        boltzmann_constant: float = 1.380649e-23,
    ) -> None:
        velocity = np.asarray(wall_velocity, dtype=np.float64)
        temperature = float(wall_temperature)
        boltzmann = float(boltzmann_constant)
        if (
            not isinstance(species, DSMCSpeciesPlan)
            or kind not in ("specular", "diffuse")
            or velocity.ndim != 1
            or velocity.size not in (1, 2, 3)
            or np.any(~np.isfinite(velocity))
            or not np.isfinite(temperature)
            or temperature <= 0.0
            or (
                reaction is not None
                and (
                    not isinstance(reaction, DSMCSurfaceReactionPlan)
                    or reaction.product_species.shape != (species.species_count,)
                    or int(np.max(np.asarray(reaction.product_species)))
                    >= species.species_count
                )
            )
            or not np.isfinite(boltzmann)
            or boltzmann <= 0.0
        ):
            raise ValueError("DSMC surface interaction data are invalid.")
        self.species = species
        self.wall_velocity = jnp.asarray(velocity)
        self.kind = kind
        self.wall_temperature = temperature
        self.reaction = reaction
        self.boltzmann_constant = boltzmann
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-surface-interaction",
                "species": species.plan_id,
                "interaction": kind,
                "wall_velocity": tuple(float(value) for value in velocity),
                "wall_temperature": temperature,
                "reaction": None if reaction is None else reaction.plan_id,
            }
        )

    def _outgoing_velocity(
        self,
        incoming: Array,
        species_index: Array,
        outward_normal: Array,
        key: Array,
        /,
    ) -> Array:
        wall_velocity = self.wall_velocity.astype(incoming.dtype)
        relative = incoming - wall_velocity
        normal_speed = jnp.sum(relative * outward_normal)
        if self.kind == "specular":
            return wall_velocity + relative - 2.0 * normal_speed * outward_normal
        mass = self.species.molecular_masses[species_index].astype(incoming.dtype)
        thermal_standard_deviation = jnp.sqrt(
            self.boltzmann_constant * self.wall_temperature / mass
        )
        normal_uniform = jax.random.uniform(
            jax.random.fold_in(key, 0),
            (),
            minval=jnp.finfo(incoming.dtype).tiny,
            maxval=1.0,
            dtype=incoming.dtype,
        )
        inward_speed = thermal_standard_deviation * jnp.sqrt(
            -2.0 * jnp.log(normal_uniform)
        )
        gaussian = jax.random.normal(
            jax.random.fold_in(key, 1), incoming.shape, dtype=incoming.dtype
        )
        tangential = gaussian - jnp.sum(gaussian * outward_normal) * outward_normal
        return (
            wall_velocity
            - inward_speed * outward_normal
            + thermal_standard_deviation * tangential
        )

    def interact(
        self,
        state: DSMCParticleState,
        particle_indices: ArrayLike,
        outward_normals: ArrayLike,
        keys: ArrayLike,
        /,
    ) -> DSMCSurfaceResult:
        indices = jnp.asarray(particle_indices, dtype=jnp.int32)
        normals = jnp.asarray(outward_normals, dtype=state.velocity.dtype)
        key_values = keys
        count = indices.size
        dimension = state.velocity.shape[-1]
        if (
            indices.ndim != 1
            or normals.shape != (count, dimension)
            or jax.random.key_data(key_values).shape != (count, 2)
            or self.wall_velocity.shape != (dimension,)
        ):
            raise ValueError(
                "DSMC surface indices, normals, keys, or dimension are invalid."
            )

        def body(particles, event):
            index, normal, key = event
            safe_index = jnp.clip(index, 0, particles.capacity - 1)
            valid = (
                (index >= 0) & (index < particles.capacity) & particles.active[safe_index]
            )
            incoming = particles.velocity[safe_index]
            species_index = particles.species_index[safe_index]
            mass = self.species.molecular_masses[species_index]
            normal_norm = jnp.sqrt(jnp.sum(normal * normal))
            normal_valid = jnp.abs(normal_norm - 1.0) <= 1.0e-6
            surface_velocity = self.wall_velocity.astype(incoming.dtype)
            wall_normal_velocity = jnp.abs(jnp.sum(surface_velocity * normal))
            normal_valid = normal_valid & (wall_normal_velocity <= 1.0e-12)
            outgoing = self._outgoing_velocity(incoming, species_index, normal, key)
            reacted = jnp.asarray(False)
            absorbed = jnp.asarray(False)
            product = species_index
            reaction_energy = jnp.asarray(0.0, dtype=incoming.dtype)
            if self.reaction is not None:
                reacted = valid & (
                    jax.random.uniform(
                        jax.random.fold_in(key, 2), (), dtype=incoming.dtype
                    )
                    < self.reaction.probabilities[species_index]
                )
                product = jnp.where(
                    reacted, self.reaction.product_species[species_index], species_index
                )
                absorbed = reacted & (product < 0)
                reaction_energy = jnp.where(
                    reacted, self.reaction.reaction_energies[species_index], 0.0
                )
            apply = valid & normal_valid
            product_safe = jnp.maximum(product, 0)
            product_mass = self.species.molecular_masses[product_safe]
            outgoing = jnp.where(apply, outgoing, incoming)
            absorbed = absorbed & apply
            reacted = reacted & apply
            active = particles.active.at[safe_index].set(
                jnp.where(
                    apply,
                    particles.active[safe_index] & ~absorbed,
                    particles.active[safe_index],
                )
            )
            velocity_value = jnp.where(absorbed, jnp.zeros_like(outgoing), outgoing)
            velocity = particles.velocity.at[safe_index].set(velocity_value)
            species = particles.species_index.at[safe_index].set(
                jnp.where(absorbed, species_index, product_safe)
            )
            cell = particles.cell_id.at[safe_index].set(
                jnp.where(absorbed, -1, particles.cell_id[safe_index])
            )
            weight = particles.statistical_weight[safe_index]
            momentum_before = mass * incoming
            momentum_after = jnp.where(absorbed, 0.0, product_mass * outgoing)
            energy_before = (
                0.5 * mass * jnp.sum(incoming**2)
                + particles.rotational_energy[safe_index]
                + particles.vibrational_energy[safe_index]
            )
            energy_after = jnp.where(
                absorbed,
                0.0,
                0.5 * product_mass * jnp.sum(outgoing**2)
                + particles.rotational_energy[safe_index]
                + particles.vibrational_energy[safe_index],
            )
            mass_to_surface = jnp.where(
                apply,
                weight * (mass - jnp.where(absorbed, 0.0, product_mass)),
                0.0,
            )
            momentum_to_surface = jnp.where(
                apply, weight * (momentum_before - momentum_after), 0.0
            )
            energy_to_surface = jnp.where(
                apply,
                weight * (energy_before - energy_after + reaction_energy),
                0.0,
            )
            rotational_energy = particles.rotational_energy.at[safe_index].set(
                jnp.where(absorbed, 0.0, particles.rotational_energy[safe_index])
            )
            vibrational_energy = particles.vibrational_energy.at[safe_index].set(
                jnp.where(absorbed, 0.0, particles.vibrational_energy[safe_index])
            )
            statistical_weight = particles.statistical_weight.at[safe_index].set(
                jnp.where(absorbed, 0.0, particles.statistical_weight[safe_index])
            )
            updated = DSMCParticleState(
                particles.position,
                velocity,
                species,
                rotational_energy,
                vibrational_energy,
                statistical_weight,
                cell,
                active,
                particles.incarnation,
            )
            finite = (~valid) | (
                normal_valid
                & jnp.all(jnp.isfinite(outgoing))
                & jnp.all(jnp.isfinite(momentum_to_surface))
                & jnp.isfinite(energy_to_surface)
            )
            return updated, (
                reacted,
                mass_to_surface,
                absorbed,
                momentum_to_surface,
                energy_to_surface,
                finite,
            )

        updated, diagnostics = jax.lax.scan(body, state, (indices, normals, key_values))
        reacted, mass, absorbed, momentum, energy, finite = diagnostics
        return DSMCSurfaceResult(
            updated,
            mass,
            reacted,
            absorbed,
            momentum,
            energy,
            jnp.sum(mass),
            jnp.sum(momentum, axis=0),
            jnp.sum(energy),
            finite,
            jnp.all(finite),
            self.plan_id,
        )


__all__ = [
    "DSMCSurfaceInteractionPlan",
    "DSMCSurfaceReactionPlan",
    "DSMCSurfaceResult",
]
