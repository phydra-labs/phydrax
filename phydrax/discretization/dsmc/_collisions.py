#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import DSMCParticleState, DSMCSpeciesPlan


class DSMCCollisionResult(StrictModule):
    state: DSMCParticleState
    accepted: Array
    acceptance_probability: Array
    momentum_defect: Array
    energy_defect: Array
    majorant_violation: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class VSSVHSCollisionPlan(StrictModule, NonTrainableState):
    """Pairwise elastic VSS/VHS collision kernel with explicit majorants."""

    species: DSMCSpeciesPlan
    reference_temperature: float = eqx.field(static=True)
    scattering_parameters: Array
    boltzmann_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: DSMCSpeciesPlan,
        scattering_parameters: ArrayLike,
        /,
        *,
        reference_temperature: float = 273.15,
        boltzmann_constant: float = 1.380649e-23,
    ):
        scattering = np.asarray(scattering_parameters, dtype=float)
        temperature = float(reference_temperature)
        boltzmann = float(boltzmann_constant)
        if (
            not isinstance(species, DSMCSpeciesPlan)
            or scattering.shape != (species.species_count, species.species_count)
            or np.any(~np.isfinite(scattering))
            or np.any(scattering < 1.0)
            or not np.allclose(scattering, scattering.T)
            or not np.isfinite(temperature)
            or temperature <= 0.0
            or not np.isfinite(boltzmann)
            or boltzmann <= 0.0
        ):
            raise ValueError("VSS/VHS scattering or reference data are invalid.")
        self.species = species
        self.scattering_parameters = jnp.asarray(scattering)
        self.reference_temperature = temperature
        self.boltzmann_constant = boltzmann
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vss-vhs-collision",
                "species": species.plan_id,
                "scattering_parameters": tuple(
                    tuple(float(x) for x in row) for row in scattering
                ),
                "reference_temperature": temperature,
                "boltzmann_constant": boltzmann,
            }
        )

    def collision_cross_section(
        self, first_species: Array, second_species: Array, relative_speed: Array, /
    ) -> Array:
        first_mass = self.species.molecular_masses[first_species]
        second_mass = self.species.molecular_masses[second_species]
        reduced_mass = first_mass * second_mass / (first_mass + second_mass)
        diameter = 0.5 * (
            self.species.reference_diameters[first_species]
            + self.species.reference_diameters[second_species]
        )
        omega = 0.5 * (
            self.species.viscosity_exponents[first_species]
            + self.species.viscosity_exponents[second_species]
        )
        reference_speed = jnp.sqrt(
            2.0 * self.boltzmann_constant * self.reference_temperature / reduced_mass
        )
        speed = jnp.maximum(relative_speed, jnp.finfo(relative_speed.dtype).tiny)
        return (
            jnp.pi
            * diameter
            * diameter
            * (reference_speed / speed) ** (2.0 * (omega - 0.5))
        )

    def collide(
        self,
        state: DSMCParticleState,
        first_indices: ArrayLike,
        second_indices: ArrayLike,
        uniforms: ArrayLike,
        majorant_sigma_speed: ArrayLike,
        /,
    ) -> DSMCCollisionResult:
        first = jnp.asarray(first_indices, dtype=jnp.int32)
        second = jnp.asarray(second_indices, dtype=jnp.int32)
        random = jnp.asarray(uniforms, dtype=state.velocity.dtype)
        majorant = jnp.asarray(majorant_sigma_speed, dtype=state.velocity.dtype)
        event_count = first.size
        dimension = state.velocity.shape[-1]
        if (
            first.shape != second.shape
            or first.ndim != 1
            or random.shape != (event_count, 3)
            or majorant.shape not in ((), (event_count,))
            or dimension not in (1, 2, 3)
        ):
            raise ValueError(
                "DSMC collision pairs, random draws, or majorants are invalid."
            )
        majorant = jnp.broadcast_to(majorant, (event_count,))
        first_velocity = state.velocity[first]
        second_velocity = state.velocity[second]
        first_species = state.species_index[first]
        second_species = state.species_index[second]
        relative = first_velocity - second_velocity
        relative_speed = jnp.sqrt(jnp.sum(relative * relative, axis=-1))
        cross_section = self.collision_cross_section(
            first_species, second_species, relative_speed
        )
        sigma_speed = cross_section * relative_speed
        probability = sigma_speed / jnp.maximum(
            majorant, jnp.finfo(relative_speed.dtype).tiny
        )
        valid_pair = (
            (first >= 0)
            & (first < state.capacity)
            & (second >= 0)
            & (second < state.capacity)
            & (first != second)
            & state.active[first]
            & state.active[second]
            & (state.cell_id[first] == state.cell_id[second])
        )
        accepted = valid_pair & (random[:, 0] < jnp.minimum(probability, 1.0))
        first_mass = self.species.molecular_masses[first_species]
        second_mass = self.species.molecular_masses[second_species]
        total_mass = first_mass + second_mass
        center = (
            first_mass[:, None] * first_velocity + second_mass[:, None] * second_velocity
        ) / total_mass[:, None]
        if dimension == 1:
            direction = jnp.ones((event_count, 1), dtype=state.velocity.dtype)
        elif dimension == 2:
            angle = 2.0 * jnp.pi * random[:, 1]
            direction = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
        else:
            cosine = 2.0 * random[:, 1] - 1.0
            sine = jnp.sqrt(jnp.maximum(1.0 - cosine * cosine, 0.0))
            angle = 2.0 * jnp.pi * random[:, 2]
            direction = jnp.stack(
                (sine * jnp.cos(angle), sine * jnp.sin(angle), cosine), axis=-1
            )
        scattered = relative_speed[:, None] * direction
        candidate_first = center + second_mass[:, None] / total_mass[:, None] * scattered
        candidate_second = center - first_mass[:, None] / total_mass[:, None] * scattered
        first_after = jnp.where(accepted[:, None], candidate_first, first_velocity)
        second_after = jnp.where(accepted[:, None], candidate_second, second_velocity)
        velocity = state.velocity.at[first].set(first_after).at[second].set(second_after)
        momentum_before = (
            first_mass[:, None] * first_velocity + second_mass[:, None] * second_velocity
        )
        momentum_after = (
            first_mass[:, None] * first_after + second_mass[:, None] * second_after
        )
        energy_before = 0.5 * first_mass * jnp.sum(
            first_velocity * first_velocity, axis=-1
        ) + 0.5 * second_mass * jnp.sum(second_velocity * second_velocity, axis=-1)
        energy_after = 0.5 * first_mass * jnp.sum(
            first_after * first_after, axis=-1
        ) + 0.5 * second_mass * jnp.sum(second_after * second_after, axis=-1)
        momentum_defect = jnp.where(
            accepted[:, None], momentum_after - momentum_before, 0.0
        )
        energy_defect = jnp.where(accepted, energy_after - energy_before, 0.0)
        updated = DSMCParticleState(
            state.position,
            velocity,
            state.species_index,
            state.rotational_energy,
            state.vibrational_energy,
            state.statistical_weight,
            state.cell_id,
            state.active,
            state.incarnation,
        )
        finite = jnp.all(jnp.isfinite(velocity)) & jnp.all(jnp.isfinite(probability))
        scale = jnp.maximum(jnp.max(jnp.abs(energy_before)), 1.0)
        successful = (
            finite
            & ~jnp.any(probability > 1.0 + 64.0 * jnp.finfo(probability.dtype).eps)
            & jnp.all(
                jnp.abs(energy_defect)
                <= 512.0 * jnp.finfo(energy_defect.dtype).eps * scale
            )
        )
        return DSMCCollisionResult(
            updated,
            accepted,
            probability,
            momentum_defect,
            energy_defect,
            probability > 1.0,
            finite,
            successful,
            self.plan_id,
        )


__all__ = ["DSMCCollisionResult", "VSSVHSCollisionPlan"]
