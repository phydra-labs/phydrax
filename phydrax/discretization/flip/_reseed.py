#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..particle import ParticlePopulationState
from ._types import FLIPParticleState


class FLIPReseedingResult(StrictModule):
    candidate_population: ParticlePopulationState
    accepted_population: ParticlePopulationState
    candidate_particles: FLIPParticleState
    accepted_particles: FLIPParticleState
    inserted: Array
    merged: Array
    mass_defect: Array
    momentum_defect: Array
    energy_defect: Array
    capacity_available: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class FLIPReseedingPlan(StrictModule, NonTrainableState):
    cell_count: int = eqx.field(static=True)
    target_per_cell: int = eqx.field(static=True)
    minimum_per_cell: int = eqx.field(static=True)
    maximum_per_cell: int = eqx.field(static=True)
    maximum_events: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_count: int,
        /,
        *,
        target_per_cell: int,
        minimum_per_cell: int,
        maximum_per_cell: int,
        maximum_events: int,
    ):
        cells = int(cell_count)
        target = int(target_per_cell)
        minimum = int(minimum_per_cell)
        maximum = int(maximum_per_cell)
        events = int(maximum_events)
        if (
            cells <= 0
            or not 0 <= minimum <= target <= maximum
            or target <= 0
            or events <= 0
        ):
            raise ValueError("FLIP reseeding policy is invalid.")
        self.cell_count = cells
        self.target_per_cell = target
        self.minimum_per_cell = minimum
        self.maximum_per_cell = maximum
        self.maximum_events = events
        self.plan_id = canonical_fingerprint(
            {
                "kind": "flip-reseeding",
                "cell_count": cells,
                "target": target,
                "minimum": minimum,
                "maximum": maximum,
                "events": events,
            }
        )

    def apply(
        self,
        population: ParticlePopulationState,
        particles: FLIPParticleState,
        cell_ids: ArrayLike,
        cell_centers: ArrayLike,
        /,
    ) -> FLIPReseedingResult:
        cells = jnp.asarray(cell_ids, dtype=jnp.int32)
        centers = jnp.asarray(cell_centers, dtype=particles.position.dtype)
        if cells.shape != population.active.shape or centers.shape != (
            self.cell_count,
            particles.position.shape[1],
        ):
            raise ValueError("FLIP reseeding cell arrays are incompatible.")
        initial_mass = jnp.sum(population.mass)
        initial_momentum = jnp.sum(population.mass[:, None] * particles.velocity, axis=0)
        initial_energy = 0.5 * jnp.sum(
            population.mass * jnp.sum(particles.velocity**2, axis=-1)
        )
        active = population.active
        mass = population.mass
        position = particles.position
        velocity = particles.velocity
        incarnation = population.incarnation
        ever = population.ever_occupied
        retired = population.retired
        inserted = jnp.zeros_like(active)
        merged = jnp.zeros_like(active)
        event_count = jnp.asarray(0, dtype=jnp.int32)
        for cell in range(self.cell_count):
            members = active & (cells == cell)
            count = jnp.sum(members, dtype=jnp.int32)
            member_indices = jnp.nonzero(members, size=active.size, fill_value=-1)[0]
            receiver = jnp.maximum(member_indices[0], 0)
            excess = jnp.maximum(count - self.target_per_cell, 0)
            for local in range(1, self.maximum_per_cell + 1):
                slot = jnp.maximum(member_indices[local], 0)
                use = (
                    (local <= excess)
                    & (member_indices[local] >= 0)
                    & (event_count < self.maximum_events)
                )
                combined_mass = mass[receiver] + mass[slot]
                combined_velocity = (
                    mass[receiver] * velocity[receiver] + mass[slot] * velocity[slot]
                ) / jnp.maximum(combined_mass, 1.0e-30)
                combined_position = (
                    mass[receiver] * position[receiver] + mass[slot] * position[slot]
                ) / jnp.maximum(combined_mass, 1.0e-30)
                mass = mass.at[receiver].set(
                    jnp.where(use, combined_mass, mass[receiver])
                )
                velocity = velocity.at[receiver].set(
                    jnp.where(use, combined_velocity, velocity[receiver])
                )
                position = position.at[receiver].set(
                    jnp.where(use, combined_position, position[receiver])
                )
                active = active.at[slot].set(jnp.where(use, False, active[slot]))
                mass = mass.at[slot].set(jnp.where(use, 0.0, mass[slot]))
                velocity = velocity.at[slot].set(
                    jnp.where(use, jnp.zeros_like(velocity[slot]), velocity[slot])
                )
                merged = merged.at[slot].set(use)
                event_count = event_count + use.astype(jnp.int32)
            deficit = jnp.maximum(self.target_per_cell - count, 0)
            free_indices = jnp.nonzero(
                ~active & ~retired, size=active.size, fill_value=-1
            )[0]
            donor_mass = jnp.where(count > 0, mass[receiver], 0.0)
            split_mass = donor_mass / jnp.maximum(deficit + 1, 1)
            for local in range(self.target_per_cell):
                slot = jnp.maximum(free_indices[local], 0)
                use = (
                    (local < deficit)
                    & (count > 0)
                    & (free_indices[local] >= 0)
                    & (event_count < self.maximum_events)
                )
                mass = mass.at[receiver].set(
                    jnp.where(use, mass[receiver] - split_mass, mass[receiver])
                )
                active = active.at[slot].set(jnp.where(use, True, active[slot]))
                mass = mass.at[slot].set(jnp.where(use, split_mass, mass[slot]))
                position = position.at[slot].set(
                    jnp.where(use, centers[cell], position[slot])
                )
                velocity = velocity.at[slot].set(
                    jnp.where(use, velocity[receiver], velocity[slot])
                )
                incarnation = incarnation.at[slot].set(
                    jnp.where(use, incarnation[slot] + 1, incarnation[slot])
                )
                ever = ever.at[slot].set(jnp.where(use, True, ever[slot]))
                inserted = inserted.at[slot].set(use)
                event_count = event_count + use.astype(jnp.int32)
        candidate_population = ParticlePopulationState(
            active, mass, incarnation, ever, retired
        )
        candidate_particles = FLIPParticleState(position, velocity)
        final_mass = jnp.sum(mass)
        final_momentum = jnp.sum(mass[:, None] * velocity, axis=0)
        final_energy = 0.5 * jnp.sum(mass * jnp.sum(velocity**2, axis=-1))
        mass_defect = final_mass - initial_mass
        momentum_defect = jnp.sqrt(jnp.sum((final_momentum - initial_momentum) ** 2))
        energy_defect = final_energy - initial_energy
        finite = (
            jnp.all(jnp.isfinite(mass))
            & jnp.all(jnp.isfinite(position))
            & jnp.all(jnp.isfinite(velocity))
        )
        tolerance = 256.0 * jnp.finfo(mass.dtype).eps
        conservative = (
            jnp.abs(mass_defect) <= tolerance * jnp.maximum(1.0, initial_mass)
        ) & (
            momentum_defect
            <= tolerance * jnp.maximum(1.0, jnp.sqrt(jnp.sum(initial_momentum**2)))
        )
        capacity_available = event_count <= self.maximum_events
        successful = finite & conservative & capacity_available
        accepted_population = jax_tree_where(successful, candidate_population, population)
        accepted_particles = jax_tree_where(successful, candidate_particles, particles)
        return FLIPReseedingResult(
            candidate_population,
            accepted_population,
            candidate_particles,
            accepted_particles,
            inserted,
            merged,
            mass_defect,
            momentum_defect,
            energy_defect,
            capacity_available,
            finite,
            successful,
            self.plan_id,
        )


def jax_tree_where(predicate, candidate, current):
    import jax

    return jax.tree.map(
        lambda proposed, old: jnp.where(predicate, proposed, old), candidate, current
    )


__all__ = ["FLIPReseedingPlan", "FLIPReseedingResult"]
