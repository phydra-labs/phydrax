#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
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
        particle_count = active.shape[0]
        particle_indices = jnp.arange(particle_count, dtype=jnp.int32)
        valid_cell = active & (cells >= 0) & (cells < self.cell_count)
        safe_cells = jnp.where(valid_cell, cells, 0)
        counts = (
            jnp.zeros((self.cell_count,), dtype=jnp.int32)
            .at[safe_cells]
            .add(valid_cell.astype(jnp.int32))
        )
        order = jnp.lexsort(
            (
                particle_indices,
                jnp.where(valid_cell, cells, self.cell_count),
            )
        )
        sorted_particles = particle_indices[order]
        offsets = jnp.cumsum(counts) - counts
        receiver_positions = jnp.minimum(offsets, particle_count - 1)
        receivers = sorted_particles[receiver_positions]
        receiver_valid = counts > 0
        safe_receivers = jnp.where(receiver_valid, receivers, 0)

        merge_local = jnp.arange(1, self.maximum_per_cell + 1, dtype=jnp.int32)
        merge_positions = jnp.minimum(
            offsets[:, None] + merge_local[None, :],
            particle_count - 1,
        )
        merge_slots = sorted_particles[merge_positions]
        excess = jnp.maximum(counts - self.target_per_cell, 0)
        merge_requested = (merge_local[None, :] <= excess[:, None]) & (
            merge_local[None, :] < counts[:, None]
        )
        merge_rank = jnp.cumsum(merge_requested.reshape((-1,)), dtype=jnp.int32) - 1
        merge_use = merge_requested & (
            merge_rank.reshape(merge_requested.shape) < self.maximum_events
        )
        merge_slot_mask = (
            jnp.zeros((particle_count,), dtype=jnp.int32)
            .at[merge_slots.reshape((-1,))]
            .add(merge_use.reshape((-1,)).astype(jnp.int32))
            > 0
        )
        merged_mass = jnp.sum(
            jnp.where(merge_use, mass[merge_slots], 0.0),
            axis=-1,
        )
        merged_momentum = jnp.sum(
            jnp.where(
                merge_use[..., None],
                mass[merge_slots, None] * velocity[merge_slots],
                0.0,
            ),
            axis=1,
        )
        merged_position_moment = jnp.sum(
            jnp.where(
                merge_use[..., None],
                mass[merge_slots, None] * position[merge_slots],
                0.0,
            ),
            axis=1,
        )
        receiver_mass = mass[safe_receivers]
        combined_mass = receiver_mass + merged_mass
        combined_velocity = (
            receiver_mass[:, None] * velocity[safe_receivers] + merged_momentum
        ) / jnp.maximum(combined_mass[:, None], 1.0e-30)
        combined_position = (
            receiver_mass[:, None] * position[safe_receivers] + merged_position_moment
        ) / jnp.maximum(combined_mass[:, None], 1.0e-30)
        mass = mass.at[safe_receivers].add(
            jnp.where(receiver_valid, combined_mass - receiver_mass, 0.0)
        )
        velocity = velocity.at[safe_receivers].add(
            jnp.where(
                receiver_valid[:, None],
                combined_velocity - velocity[safe_receivers],
                0.0,
            )
        )
        position = position.at[safe_receivers].add(
            jnp.where(
                receiver_valid[:, None],
                combined_position - position[safe_receivers],
                0.0,
            )
        )
        active = active & ~merge_slot_mask
        mass = jnp.where(merge_slot_mask, 0.0, mass)
        velocity = jnp.where(merge_slot_mask[:, None], 0.0, velocity)

        free_mask = ~active & ~retired
        free_count = jnp.sum(free_mask, dtype=jnp.int32)
        free_indices = jnp.nonzero(
            free_mask,
            size=particle_count,
            fill_value=0,
        )[0]
        deficit = jnp.maximum(self.target_per_cell - counts, 0)
        split_local = jnp.arange(self.target_per_cell, dtype=jnp.int32)
        split_requested = (split_local[None, :] < deficit[:, None]) & receiver_valid[
            :, None
        ]
        merge_events = jnp.sum(merge_requested, dtype=jnp.int32)
        remaining_events = jnp.maximum(self.maximum_events - merge_events, 0)
        split_rank = jnp.cumsum(split_requested.reshape((-1,)), dtype=jnp.int32) - 1
        split_rank = split_rank.reshape(split_requested.shape)
        split_use = (
            split_requested & (split_rank < remaining_events) & (split_rank < free_count)
        )
        safe_free_rank = jnp.clip(split_rank, 0, particle_count - 1)
        split_slots = free_indices[safe_free_rank]
        donor_mass = jnp.where(receiver_valid, mass[safe_receivers], 0.0)
        split_mass = donor_mass / jnp.maximum(deficit + 1, 1)
        split_slot_mask = (
            jnp.zeros((particle_count,), dtype=jnp.int32)
            .at[split_slots.reshape((-1,))]
            .add(split_use.reshape((-1,)).astype(jnp.int32))
            > 0
        )
        split_mass_payload = (
            jnp.zeros((particle_count,), dtype=mass.dtype)
            .at[split_slots.reshape((-1,))]
            .add(
                jnp.where(
                    split_use,
                    split_mass[:, None],
                    0.0,
                ).reshape((-1,))
            )
        )
        split_position_payload = (
            jnp.zeros_like(position)
            .at[split_slots.reshape((-1,))]
            .add(
                jnp.where(
                    split_use[..., None],
                    centers[:, None, :],
                    0.0,
                ).reshape((-1, position.shape[-1]))
            )
        )
        split_velocity_payload = (
            jnp.zeros_like(velocity)
            .at[split_slots.reshape((-1,))]
            .add(
                jnp.where(
                    split_use[..., None],
                    velocity[safe_receivers, None, :],
                    0.0,
                ).reshape((-1, velocity.shape[-1]))
            )
        )
        splits_per_cell = jnp.sum(split_use, axis=-1, dtype=jnp.int32)
        mass = mass.at[safe_receivers].add(
            jnp.where(receiver_valid, -splits_per_cell * split_mass, 0.0)
        )
        active = active | split_slot_mask
        mass = jnp.where(split_slot_mask, split_mass_payload, mass)
        position = jnp.where(
            split_slot_mask[:, None],
            split_position_payload,
            position,
        )
        velocity = jnp.where(
            split_slot_mask[:, None],
            split_velocity_payload,
            velocity,
        )
        incarnation = incarnation + split_slot_mask.astype(incarnation.dtype)
        ever = ever | split_slot_mask
        inserted = split_slot_mask
        merged = merge_slot_mask
        required_events = merge_events + jnp.sum(split_requested, dtype=jnp.int32)
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
        capacity_available = (required_events <= self.maximum_events) & (
            jnp.sum(split_requested, dtype=jnp.int32) <= free_count
        )
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

    return jax.tree.map(
        lambda proposed, old: jnp.where(predicate, proposed, old), candidate, current
    )


__all__ = ["FLIPReseedingPlan", "FLIPReseedingResult"]
