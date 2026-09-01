#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._types import MPMParticleState


class MPMLifecycleState(StrictModule):
    particle_ids: Array
    masses: Array
    active: Array
    parent_ids: Array
    generation: Array


class MPMLifecycleEvidence(StrictModule):
    mass_defect: Array
    momentum_defect: Array
    volume_defect: Array
    id_unique: Array
    capacity_remaining: Array
    successful: Array


class MPMLifecycleResult(StrictModule):
    particles: MPMParticleState
    lifecycle: MPMLifecycleState
    evidence: MPMLifecycleEvidence


class MPMParticleLifecyclePlan(StrictModule, NonTrainableState):
    capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, capacity: int, /):
        capacity_ = int(capacity)
        if capacity_ <= 0:
            raise ValueError("Particle lifecycle capacity must be positive.")
        self.capacity = capacity_
        self.plan_id = canonical_fingerprint(
            {"kind": "mpm-particle-lifecycle", "capacity": capacity_}
        )

    def initialize(self, particle_ids, masses, active, /):
        identifiers = jnp.asarray(particle_ids, dtype=jnp.int64)
        mass = jnp.asarray(masses)
        active_ = jnp.asarray(active, dtype=bool)
        if (
            identifiers.shape != (self.capacity,)
            or mass.shape != identifiers.shape
            or active_.shape != identifiers.shape
        ):
            raise ValueError("Lifecycle arrays must have prepared capacity shape.")
        unique = jnp.unique(jnp.where(active_, identifiers, -1), size=self.capacity)
        valid = jnp.all(jnp.where(active_, mass > 0.0, True)) & (
            jnp.sum(unique >= 0) == jnp.sum(active_)
        )
        state = MPMLifecycleState(
            identifiers,
            mass,
            active_,
            -jnp.ones_like(identifiers),
            jnp.zeros((), dtype=jnp.int32),
        )
        return state, valid

    @staticmethod
    def _evidence(before, after, particles_before, particles_after):
        before_mass = jnp.sum(jnp.where(before.active, before.masses, 0.0))
        after_mass = jnp.sum(jnp.where(after.active, after.masses, 0.0))
        before_momentum = jnp.sum(
            jnp.where(
                before.active[:, None],
                before.masses[:, None] * particles_before.velocity,
                0.0,
            ),
            axis=0,
        )
        after_momentum = jnp.sum(
            jnp.where(
                after.active[:, None],
                after.masses[:, None] * particles_after.velocity,
                0.0,
            ),
            axis=0,
        )
        before_volume = jnp.sum(
            jnp.where(before.active, particles_before.reference_volume, 0.0)
        )
        after_volume = jnp.sum(
            jnp.where(after.active, particles_after.reference_volume, 0.0)
        )
        active_ids = jnp.where(after.active, after.particle_ids, -1)
        unique = jnp.unique(active_ids, size=active_ids.size)
        id_unique = jnp.sum(unique >= 0) == jnp.sum(after.active)
        mass_defect = jnp.abs(after_mass - before_mass) / jnp.maximum(1.0, before_mass)
        momentum_defect = jnp.linalg.norm(after_momentum - before_momentum) / jnp.maximum(
            1.0, jnp.linalg.norm(before_momentum)
        )
        volume_defect = jnp.abs(after_volume - before_volume) / jnp.maximum(
            1.0, before_volume
        )
        remaining = jnp.sum(~after.active, dtype=jnp.int32)
        return MPMLifecycleEvidence(
            mass_defect,
            momentum_defect,
            volume_defect,
            id_unique,
            remaining,
            id_unique
            & (mass_defect <= 1.0e-12)
            & (momentum_defect <= 1.0e-12)
            & (volume_defect <= 1.0e-12),
        )

    def activate(
        self,
        particles: MPMParticleState,
        lifecycle: MPMLifecycleState,
        slots: ArrayLike,
        particle_ids: ArrayLike,
        masses: ArrayLike,
        /,
    ):
        slots_ = jnp.asarray(slots, dtype=jnp.int32)
        identifiers = jnp.asarray(particle_ids, dtype=jnp.int64)
        masses_ = jnp.asarray(masses, dtype=lifecycle.masses.dtype)
        if slots_.shape != identifiers.shape or slots_.shape != masses_.shape:
            raise ValueError("Activation slot/ID/mass arrays must share shape.")
        available = jnp.all(~lifecycle.active[slots_]) & jnp.all(masses_ > 0.0)
        next_state = MPMLifecycleState(
            lifecycle.particle_ids.at[slots_].set(identifiers),
            lifecycle.masses.at[slots_].set(masses_),
            lifecycle.active.at[slots_].set(True),
            lifecycle.parent_ids.at[slots_].set(-1),
            lifecycle.generation + available.astype(jnp.int32),
        )
        evidence = self._evidence(lifecycle, next_state, particles, particles)
        evidence = eqx.tree_at(
            lambda value: value.successful,
            evidence,
            available & evidence.id_unique,
        )
        return MPMLifecycleResult(particles, next_state, evidence)

    def retire(
        self,
        particles: MPMParticleState,
        lifecycle: MPMLifecycleState,
        slots: ArrayLike,
        /,
    ):
        slots_ = jnp.asarray(slots, dtype=jnp.int32)
        next_state = MPMLifecycleState(
            lifecycle.particle_ids,
            lifecycle.masses,
            lifecycle.active.at[slots_].set(False),
            lifecycle.parent_ids,
            lifecycle.generation + 1,
        )
        return MPMLifecycleResult(
            particles,
            next_state,
            self._evidence(lifecycle, next_state, particles, particles),
        )

    def split(
        self,
        particles: MPMParticleState,
        lifecycle: MPMLifecycleState,
        parent_slot: int,
        child_slots: ArrayLike,
        child_ids: ArrayLike,
        mass_fractions: ArrayLike,
        position_offsets: ArrayLike,
        /,
    ):
        parent = int(parent_slot)
        children = jnp.asarray(child_slots, dtype=jnp.int32)
        identifiers = jnp.asarray(child_ids, dtype=jnp.int64)
        fractions = jnp.asarray(mass_fractions, dtype=lifecycle.masses.dtype)
        offsets = jnp.asarray(position_offsets, dtype=particles.position.dtype)
        if (
            children.ndim != 1
            or children.shape != identifiers.shape
            or children.shape != fractions.shape
            or offsets.shape != (children.size, particles.position.shape[1])
        ):
            raise ValueError("Particle split inputs have incompatible shapes.")
        available = (
            lifecycle.active[parent]
            & jnp.all(~lifecycle.active[children])
            & jnp.all(fractions > 0.0)
            & jnp.isclose(jnp.sum(fractions), 1.0)
        )

        def copy_parent(array):
            return array.at[children].set(
                jnp.broadcast_to(array[parent], array[children].shape)
            )

        next_particles = MPMParticleState(
            particles.position.at[children].set(particles.position[parent] + offsets),
            copy_parent(particles.velocity),
            copy_parent(particles.deformation_gradient),
            copy_parent(particles.affine_velocity),
            particles.reference_volume.at[children].set(
                particles.reference_volume[parent] * fractions
            ),
            copy_parent(particles.first_piola),
            copy_parent(particles.reference_energy_density),
            copy_parent(particles.maximum_wave_speed),
            copy_parent(particles.material_state),
        )
        next_active = lifecycle.active.at[parent].set(False).at[children].set(True)
        next_mass = lifecycle.masses.at[children].set(
            lifecycle.masses[parent] * fractions
        )
        next_state = MPMLifecycleState(
            lifecycle.particle_ids.at[children].set(identifiers),
            next_mass,
            next_active,
            lifecycle.parent_ids.at[children].set(lifecycle.particle_ids[parent]),
            lifecycle.generation + available.astype(jnp.int32),
        )
        evidence = self._evidence(lifecycle, next_state, particles, next_particles)
        evidence = eqx.tree_at(
            lambda value: value.successful, evidence, available & evidence.successful
        )
        return MPMLifecycleResult(next_particles, next_state, evidence)

    def merge(
        self,
        particles: MPMParticleState,
        lifecycle: MPMLifecycleState,
        source_slots: ArrayLike,
        target_slot: int,
        target_id: int,
        /,
    ):
        sources = jnp.asarray(source_slots, dtype=jnp.int32)
        target = int(target_slot)
        masses = lifecycle.masses[sources]
        total = jnp.sum(masses)
        weights = masses / jnp.where(total > 0.0, total, 1.0)

        def merge_field(array):
            merged = jnp.tensordot(weights, array[sources], axes=(0, 0))
            return array.at[target].set(merged)

        next_particles = MPMParticleState(
            merge_field(particles.position),
            merge_field(particles.velocity),
            merge_field(particles.deformation_gradient),
            merge_field(particles.affine_velocity),
            particles.reference_volume.at[target].set(
                jnp.sum(particles.reference_volume[sources])
            ),
            merge_field(particles.first_piola),
            merge_field(particles.reference_energy_density),
            merge_field(particles.maximum_wave_speed),
            merge_field(particles.material_state),
        )
        next_active = lifecycle.active.at[sources].set(False).at[target].set(True)
        next_state = MPMLifecycleState(
            lifecycle.particle_ids.at[target].set(int(target_id)),
            lifecycle.masses.at[target].set(total),
            next_active,
            lifecycle.parent_ids.at[target].set(-1),
            lifecycle.generation + 1,
        )
        evidence = self._evidence(lifecycle, next_state, particles, next_particles)
        return MPMLifecycleResult(next_particles, next_state, evidence)


class MPMCapacityBucketPlan(StrictModule, NonTrainableState):
    buckets: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, buckets: Sequence[int], /):
        values = tuple(sorted(set(int(value) for value in buckets)))
        if not values or any(value <= 0 for value in values):
            raise ValueError("Capacity buckets must be positive.")
        self.buckets = values
        self.plan_id = canonical_fingerprint(
            {"kind": "mpm-capacity-buckets", "buckets": values}
        )

    def select(self, required: int, /):
        required_ = int(required)
        for value in self.buckets:
            if required_ <= value:
                return value
        raise OverflowError("No MPM capacity bucket admits the requested state.")


class MPMPageTableState(StrictModule):
    keys: Array
    values: Array
    occupied: Array
    count: Array
    overflow: Array


class MPMPageTablePlan(StrictModule, NonTrainableState):
    capacity: int = eqx.field(static=True)
    maximum_probes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, capacity: int, /, *, maximum_probes: int = 16):
        capacity_ = int(capacity)
        probes = int(maximum_probes)
        if capacity_ <= 0 or probes <= 0:
            raise ValueError("Page-table capacity/probes must be positive.")
        self.capacity = capacity_
        self.maximum_probes = probes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-page-table",
                "capacity": capacity_,
                "maximum_probes": probes,
            }
        )

    def empty(self):
        return MPMPageTableState(
            -jnp.ones((self.capacity,), dtype=jnp.int64),
            -jnp.ones((self.capacity,), dtype=jnp.int32),
            jnp.zeros((self.capacity,), dtype=bool),
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(False),
        )

    def insert(self, state: MPMPageTableState, keys: ArrayLike, values: ArrayLike, /):
        keys_ = jnp.asarray(keys, dtype=jnp.int64)
        values_ = jnp.asarray(values, dtype=jnp.int32)
        if keys_.shape != values_.shape:
            raise ValueError("Page-table keys and values must share shape.")

        def insert_one(carry, item):
            current, overflow = carry
            key, value = item
            start = jnp.mod(key, self.capacity).astype(jnp.int32)

            def probe_body(index, probe_carry):
                table, inserted = probe_carry
                slot = jnp.mod(start + index, self.capacity)
                available = ~table.occupied[slot] | (table.keys[slot] == key)
                write = ~inserted & available
                next_table = MPMPageTableState(
                    table.keys.at[slot].set(jnp.where(write, key, table.keys[slot])),
                    table.values.at[slot].set(
                        jnp.where(write, value, table.values[slot])
                    ),
                    table.occupied.at[slot].set(table.occupied[slot] | write),
                    table.count + (write & ~table.occupied[slot]).astype(jnp.int32),
                    table.overflow,
                )
                return next_table, inserted | write

            table, inserted = jax.lax.fori_loop(
                0, self.maximum_probes, probe_body, (current, jnp.asarray(False))
            )
            return (table, overflow | ~inserted), inserted

        (table, overflow), inserted = jax.lax.scan(
            insert_one, (state, state.overflow), (keys_.reshape(-1), values_.reshape(-1))
        )
        table = eqx.tree_at(lambda value: value.overflow, table, overflow)
        return table, inserted.reshape(keys_.shape)


class MPMAMRPlan(StrictModule, NonTrainableState):
    level_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
    maximum_blocks: tuple[int, ...] = eqx.field(static=True)
    subcycles: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        level_shapes: Sequence[Sequence[int]],
        maximum_blocks: Sequence[int],
        /,
        *,
        refinement_ratio: int = 2,
    ):
        shapes = tuple(tuple(int(value) for value in shape) for shape in level_shapes)
        blocks = tuple(int(value) for value in maximum_blocks)
        ratio = int(refinement_ratio)
        if (
            not shapes
            or len(shapes) != len(blocks)
            or ratio != 2
            or any(value <= 0 for shape in shapes for value in shape)
            or any(value <= 0 for value in blocks)
            or any(
                any(
                    fine != ratio * coarse
                    for fine, coarse in zip(shapes[level], shapes[level - 1], strict=True)
                )
                for level in range(1, len(shapes))
            )
        ):
            raise ValueError("MPM AMR hierarchy must be ratio-two nested grids.")
        self.level_shapes = shapes
        self.refinement_ratio = ratio
        self.maximum_blocks = blocks
        self.subcycles = tuple(ratio**level for level in range(len(shapes)))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-amr-plan",
                "level_shapes": shapes,
                "maximum_blocks": blocks,
                "refinement_ratio": ratio,
                "subcycles": self.subcycles,
            }
        )

    def restrict(self, fine: ArrayLike, /):
        value = jnp.asarray(fine)
        dimension = len(self.level_shapes[0])
        result = value
        for axis in range(dimension):
            shape = result.shape
            result = result.reshape(
                shape[:axis] + (shape[axis] // 2, 2) + shape[axis + 1 :]
            ).mean(axis=axis + 1)
        return result

    def prolong(self, coarse: ArrayLike, /):
        value = jnp.asarray(coarse)
        result = value
        for axis in range(len(self.level_shapes[0])):
            result = jnp.repeat(result, 2, axis=axis)
        return result

    def select_particle_level(self, half_extent_cells: ArrayLike, /):
        extent = jnp.asarray(half_extent_cells)
        maximum = jnp.max(extent, axis=-1)
        level = jnp.floor(-jnp.log2(jnp.maximum(maximum, 1.0e-30))).astype(jnp.int32)
        return jnp.clip(level, 0, len(self.level_shapes) - 1)


class MPMAMRTopologyJournal(StrictModule):
    attempted: Array
    accepted: Array
    generations: Array
    refined_blocks: Array
    coarsened_blocks: Array
    level_counts: Array
    capacity_overflow: Array


__all__ = [
    "MPMAMRPlan",
    "MPMAMRTopologyJournal",
    "MPMCapacityBucketPlan",
    "MPMLifecycleEvidence",
    "MPMLifecycleResult",
    "MPMLifecycleState",
    "MPMPageTablePlan",
    "MPMPageTableState",
    "MPMParticleLifecyclePlan",
]
