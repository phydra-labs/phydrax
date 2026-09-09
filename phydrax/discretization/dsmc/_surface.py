#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
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
    ):
        products = np.asarray(product_species, dtype=np.int32)
        probabilities_ = np.asarray(probabilities, dtype=float)
        energies = np.asarray(reaction_energies, dtype=float)
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
                "products": array_tree_fingerprint(self.product_species),
                "probabilities": array_tree_fingerprint(self.probabilities),
                "energies": array_tree_fingerprint(self.reaction_energies),
            }
        )


class DSMCSurfaceResult(StrictModule):
    state: DSMCParticleState
    reacted: Array
    absorbed: Array
    momentum_to_surface: Array
    energy_to_surface: Array
    total_force_impulse: Array
    total_heat: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DSMCSurfaceInteractionPlan(StrictModule, NonTrainableState):
    """Specular, diffuse, or CLL-like particle surface exchange and chemistry."""

    species: DSMCSpeciesPlan
    kind: Literal["specular", "diffuse", "cll"] = eqx.field(static=True)
    wall_temperature: float = eqx.field(static=True)
    normal_accommodation: float = eqx.field(static=True)
    tangential_accommodation: float = eqx.field(static=True)
    reaction: DSMCSurfaceReactionPlan | None
    boltzmann_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: DSMCSpeciesPlan,
        kind: Literal["specular", "diffuse", "cll"],
        /,
        *,
        wall_temperature: float,
        normal_accommodation: float = 1.0,
        tangential_accommodation: float = 1.0,
        reaction: DSMCSurfaceReactionPlan | None = None,
        boltzmann_constant: float = 1.380649e-23,
    ):
        temperature = float(wall_temperature)
        normal = float(normal_accommodation)
        tangential = float(tangential_accommodation)
        boltzmann = float(boltzmann_constant)
        if (
            not isinstance(species, DSMCSpeciesPlan)
            or kind not in ("specular", "diffuse", "cll")
            or not np.isfinite(temperature)
            or temperature <= 0.0
            or not 0.0 <= normal <= 1.0
            or not 0.0 <= tangential <= 1.0
            or (
                reaction is not None
                and (
                    not isinstance(reaction, DSMCSurfaceReactionPlan)
                    or reaction.product_species.shape != (species.species_count,)
                )
            )
            or not np.isfinite(boltzmann)
            or boltzmann <= 0.0
        ):
            raise ValueError("DSMC surface interaction data are invalid.")
        self.species = species
        self.kind = kind
        self.wall_temperature = temperature
        self.normal_accommodation = normal
        self.tangential_accommodation = tangential
        self.reaction = reaction
        self.boltzmann_constant = boltzmann
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-surface-interaction",
                "species": species.plan_id,
                "interaction": kind,
                "wall_temperature": temperature,
                "normal_accommodation": normal,
                "tangential_accommodation": tangential,
                "reaction": None if reaction is None else reaction.plan_id,
            }
        )

    def interact(
        self,
        state: DSMCParticleState,
        particle_indices: ArrayLike,
        outward_normals: ArrayLike,
        uniforms: ArrayLike,
        /,
    ) -> DSMCSurfaceResult:
        indices = jnp.asarray(particle_indices, dtype=jnp.int32)
        normals = jnp.asarray(outward_normals, dtype=state.velocity.dtype)
        random = jnp.asarray(uniforms, dtype=state.velocity.dtype)
        count = indices.size
        dimension = state.velocity.shape[-1]
        if (
            indices.ndim != 1
            or normals.shape != (count, dimension)
            or random.shape != (count, 4)
        ):
            raise ValueError(
                "DSMC surface indices, normals, or random draws are invalid."
            )
        incoming = state.velocity[indices]
        species_index = state.species_index[indices]
        masses = self.species.molecular_masses[species_index]
        normal_speed = jnp.sum(incoming * normals, axis=-1)
        specular = incoming - 2.0 * normal_speed[:, None] * normals
        thermal_speed = jnp.sqrt(
            2.0 * self.boltzmann_constant * self.wall_temperature / masses
        )
        if dimension == 1:
            diffuse_direction = -normals
        elif dimension == 2:
            tangent = jnp.stack((-normals[:, 1], normals[:, 0]), axis=-1)
            diffuse_direction = (
                -jnp.sqrt(random[:, 1])[:, None] * normals
                + jnp.sqrt(1.0 - random[:, 1])[:, None]
                * jnp.sign(random[:, 2] - 0.5)[:, None]
                * tangent
            )
        else:
            reference = jnp.where(
                jnp.abs(normals[:, :1]) < 0.9,
                jnp.asarray((1.0, 0.0, 0.0)),
                jnp.asarray((0.0, 1.0, 0.0)),
            )
            tangent_1 = jnp.cross(normals, reference)
            tangent_1 = tangent_1 / jnp.maximum(
                jnp.sqrt(jnp.sum(tangent_1 * tangent_1, axis=-1))[..., None],
                jnp.finfo(incoming.dtype).tiny,
            )
            tangent_2 = jnp.cross(normals, tangent_1)
            azimuth = 2.0 * jnp.pi * random[:, 2]
            cosine = jnp.sqrt(random[:, 1])
            sine = jnp.sqrt(1.0 - cosine * cosine)
            diffuse_direction = -cosine[:, None] * normals + sine[:, None] * (
                jnp.cos(azimuth)[:, None] * tangent_1
                + jnp.sin(azimuth)[:, None] * tangent_2
            )
        diffuse = (
            thermal_speed[:, None]
            * jnp.sqrt(
                -jnp.log(jnp.maximum(random[:, 3], jnp.finfo(incoming.dtype).tiny))
            )[:, None]
            * diffuse_direction
        )
        if self.kind == "specular":
            outgoing = specular
        elif self.kind == "diffuse":
            outgoing = diffuse
        else:
            outgoing = (
                specular
                + self.normal_accommodation
                * jnp.sum((diffuse - specular) * normals, axis=-1)[:, None]
                * normals
                + self.tangential_accommodation
                * (
                    (diffuse - specular)
                    - jnp.sum((diffuse - specular) * normals, axis=-1)[:, None] * normals
                )
            )
        reacted = jnp.zeros((count,), dtype=bool)
        absorbed = jnp.zeros((count,), dtype=bool)
        product = species_index
        reaction_energy = jnp.zeros((count,), dtype=incoming.dtype)
        if self.reaction is not None:
            reacted = random[:, 0] < self.reaction.probabilities[species_index]
            product = jnp.where(
                reacted, self.reaction.product_species[species_index], species_index
            )
            absorbed = reacted & (product < 0)
            reaction_energy = jnp.where(
                reacted, self.reaction.reaction_energies[species_index], 0.0
            )
        active = state.active.at[indices].set(state.active[indices] & ~absorbed)
        updated_species = state.species_index.at[indices].set(
            jnp.where(absorbed, species_index, product)
        )
        velocity = state.velocity.at[indices].set(
            jnp.where(absorbed[:, None], 0.0, outgoing)
        )
        momentum_before = masses[:, None] * incoming
        product_mass = self.species.molecular_masses[jnp.maximum(product, 0)]
        momentum_after = jnp.where(
            absorbed[:, None], 0.0, product_mass[:, None] * outgoing
        )
        energy_before = (
            0.5 * masses * jnp.sum(incoming * incoming, axis=-1)
            + state.rotational_energy[indices]
            + state.vibrational_energy[indices]
        )
        energy_after = jnp.where(
            absorbed, 0.0, 0.5 * product_mass * jnp.sum(outgoing * outgoing, axis=-1)
        )
        momentum_to_surface = state.statistical_weight[indices, None] * (
            momentum_before - momentum_after
        )
        energy_to_surface = state.statistical_weight[indices] * (
            energy_before - energy_after + reaction_energy
        )
        updated = DSMCParticleState(
            state.position,
            velocity,
            updated_species,
            state.rotational_energy,
            state.vibrational_energy,
            state.statistical_weight,
            state.cell_id,
            active,
            state.incarnation,
        )
        finite = (
            jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(momentum_to_surface))
            & jnp.all(jnp.isfinite(energy_to_surface))
        )
        successful = finite & jnp.all(
            jnp.abs(jnp.sqrt(jnp.sum(normals * normals, axis=-1)) - 1.0) <= 1.0e-6
        )
        return DSMCSurfaceResult(
            updated,
            reacted,
            absorbed,
            momentum_to_surface,
            energy_to_surface,
            jnp.sum(momentum_to_surface, axis=0),
            jnp.sum(energy_to_surface),
            finite,
            successful,
            self.plan_id,
        )


__all__ = ["DSMCSurfaceInteractionPlan", "DSMCSurfaceReactionPlan", "DSMCSurfaceResult"]
