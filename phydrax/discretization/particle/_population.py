#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization


class ParticleSlotReusePolicy(IntEnum):
    NEVER_REUSE = 0
    REUSE_WITH_INCARNATION = 1


class ParticlePopulationStatus(IntEnum):
    SUCCESS = 0
    CAPACITY_EXCEEDED = 1
    INCARNATION_OVERFLOW = 2
    INVALID_REQUEST = 3
    NONFINITE = 4


class ParticlePopulationState(StrictModule):
    active: Array
    mass: Array
    incarnation: Array
    ever_occupied: Array
    retired: Array


class ParticleAllocationRequest(StrictModule):
    event_ids: Array
    masses: Array
    valid: Array


class ParticleAllocationResult(StrictModule):
    candidate_state: ParticlePopulationState
    accepted_state: ParticlePopulationState
    slots: Array
    allocated: Array
    requested_count: Array
    allocated_count: Array
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    @property
    def capacity_available(self) -> Array:
        return self.status != int(ParticlePopulationStatus.CAPACITY_EXCEEDED)


class ParticleDeactivationResult(StrictModule):
    candidate_state: ParticlePopulationState
    accepted_state: ParticlePopulationState
    removed_mass: Array
    removed_count: Array
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ParticlePopulationPlan(StrictModule, NonTrainableState):
    particles: ParticleDiscretization
    reuse_policy: ParticleSlotReusePolicy = eqx.field(static=True)
    allocation_capacity: int = eqx.field(static=True)
    incarnation_maximum: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        /,
        *,
        reuse_policy: ParticleSlotReusePolicy = ParticleSlotReusePolicy.REUSE_WITH_INCARNATION,
        allocation_capacity: int | None = None,
        incarnation_maximum: int = 2**31 - 1,
    ):
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be ParticleDiscretization.")
        reuse = ParticleSlotReusePolicy(reuse_policy)
        capacity = (
            particles.capacity
            if allocation_capacity is None
            else int(allocation_capacity)
        )
        maximum = int(incarnation_maximum)
        if capacity <= 0 or capacity > particles.capacity:
            raise ValueError("allocation_capacity must lie in [1, particle capacity].")
        if maximum <= 0 or maximum > np.iinfo(np.int32).max:
            raise ValueError("incarnation_maximum is invalid.")
        self.particles = particles
        self.reuse_policy = reuse
        self.allocation_capacity = capacity
        self.incarnation_maximum = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-population-plan",
                "particles": particles.prepared_id,
                "reuse": int(reuse),
                "allocation_capacity": capacity,
                "incarnation_maximum": maximum,
            }
        )

    def initialize(
        self,
        *,
        active_mask: ArrayLike | None = None,
        masses: ArrayLike | None = None,
    ) -> ParticlePopulationState:
        structural = self.particles.active_mask
        active = (
            structural if active_mask is None else jnp.asarray(active_mask, dtype=bool)
        )
        mass = self.particles.masses if masses is None else jnp.asarray(masses)
        if active.shape != structural.shape or mass.shape != structural.shape:
            raise ValueError("Population arrays must have particle-capacity shape.")
        active = structural & active
        valid = jnp.all(jnp.where(active, jnp.isfinite(mass) & (mass > 0.0), mass == 0.0))
        mass = eqx.error_if(
            jnp.where(active, mass, 0.0), ~valid, "Population mass/activity is invalid."
        )
        return ParticlePopulationState(
            active,
            mass,
            jnp.where(active, 1, 0).astype(jnp.int32),
            active,
            jnp.zeros_like(active),
        )

    def allocate(
        self,
        state: ParticlePopulationState,
        request: ParticleAllocationRequest,
        /,
    ) -> ParticleAllocationResult:
        if not isinstance(state, ParticlePopulationState):
            raise TypeError("state must be ParticlePopulationState.")
        if not isinstance(request, ParticleAllocationRequest):
            raise TypeError("request must be ParticleAllocationRequest.")
        width = request.valid.shape[0]
        if (
            width > self.allocation_capacity
            or request.event_ids.shape != (width,)
            or request.masses.shape != (width,)
        ):
            raise ValueError("Allocation request exceeds its prepared capacity.")
        finite_request = jnp.all(
            jnp.where(
                request.valid, jnp.isfinite(request.masses) & (request.masses > 0.0), True
            )
        )
        reusable = self.particles.active_mask & ~state.active
        if self.reuse_policy is ParticleSlotReusePolicy.NEVER_REUSE:
            reusable = reusable & ~state.ever_occupied & ~state.retired
        order = jnp.argsort(
            jnp.where(request.valid, request.event_ids, jnp.iinfo(jnp.int64).max)
        )
        ordered_valid = request.valid[order]
        ordered_mass = request.masses[order]
        requested = jnp.sum(ordered_valid, dtype=jnp.int32)
        slots = jnp.nonzero(reusable, size=width, fill_value=-1)[0].astype(jnp.int32)
        safe_slots = jnp.maximum(slots, 0)
        available = jnp.sum(reusable, dtype=jnp.int32) >= requested
        allocation_mask = ordered_valid & (slots >= 0) & available & finite_request
        previous_incarnation = state.incarnation[safe_slots]
        next_incarnation = previous_incarnation + allocation_mask.astype(jnp.int32)
        overflow = jnp.any(
            allocation_mask & (next_incarnation > self.incarnation_maximum)
        )
        successful = available & finite_request & ~overflow
        use = allocation_mask & successful
        candidate_active = state.active.at[safe_slots].set(
            jnp.where(use, True, state.active[safe_slots])
        )
        candidate_mass = state.mass.at[safe_slots].set(
            jnp.where(use, ordered_mass, state.mass[safe_slots])
        )
        candidate_incarnation = state.incarnation.at[safe_slots].set(
            jnp.where(use, next_incarnation, state.incarnation[safe_slots])
        )
        candidate_ever = state.ever_occupied.at[safe_slots].set(
            jnp.where(use, True, state.ever_occupied[safe_slots])
        )
        candidate_retired = state.retired.at[safe_slots].set(
            jnp.where(use, False, state.retired[safe_slots])
        )
        candidate = ParticlePopulationState(
            candidate_active,
            candidate_mass,
            candidate_incarnation,
            candidate_ever,
            candidate_retired,
        )
        accepted = ParticlePopulationState(
            jnp.where(successful, candidate.active, state.active),
            jnp.where(successful, candidate.mass, state.mass),
            jnp.where(successful, candidate.incarnation, state.incarnation),
            jnp.where(successful, candidate.ever_occupied, state.ever_occupied),
            jnp.where(successful, candidate.retired, state.retired),
        )
        status = jnp.where(
            overflow,
            int(ParticlePopulationStatus.INCARNATION_OVERFLOW),
            jnp.where(
                ~finite_request,
                int(ParticlePopulationStatus.INVALID_REQUEST),
                jnp.where(
                    available,
                    int(ParticlePopulationStatus.SUCCESS),
                    int(ParticlePopulationStatus.CAPACITY_EXCEEDED),
                ),
            ),
        ).astype(jnp.int32)
        return ParticleAllocationResult(
            candidate,
            accepted,
            jnp.where(use, safe_slots, -1),
            use,
            requested,
            jnp.sum(use, dtype=jnp.int32),
            status,
            successful,
            self.plan_id,
        )

    def deactivate(
        self, state: ParticlePopulationState, mask: ArrayLike, /
    ) -> ParticleDeactivationResult:
        requested = jnp.asarray(mask, dtype=bool)
        if requested.shape != state.active.shape:
            raise ValueError("Deactivation mask must have particle-capacity shape.")
        remove = state.active & requested
        removed_mass = jnp.sum(jnp.where(remove, state.mass, 0.0))
        retired = state.retired | (
            remove & (self.reuse_policy is ParticleSlotReusePolicy.NEVER_REUSE)
        )
        candidate = ParticlePopulationState(
            state.active & ~remove,
            jnp.where(remove, 0.0, state.mass),
            state.incarnation,
            state.ever_occupied,
            retired,
        )
        finite = jnp.isfinite(removed_mass)
        accepted = ParticlePopulationState(
            jnp.where(finite, candidate.active, state.active),
            jnp.where(finite, candidate.mass, state.mass),
            jnp.where(finite, candidate.incarnation, state.incarnation),
            jnp.where(finite, candidate.ever_occupied, state.ever_occupied),
            jnp.where(finite, candidate.retired, state.retired),
        )
        return ParticleDeactivationResult(
            candidate,
            accepted,
            removed_mass,
            jnp.sum(remove, dtype=jnp.int32),
            jnp.where(
                finite,
                int(ParticlePopulationStatus.SUCCESS),
                int(ParticlePopulationStatus.NONFINITE),
            ).astype(jnp.int32),
            finite,
            self.plan_id,
        )


def update_particle_population(
    previous: ParticlePopulationState,
    active_mask: ArrayLike,
    masses: ArrayLike,
    /,
) -> ParticlePopulationState:
    """Update runtime activity/mass while preserving incarnation identity."""

    active = jnp.asarray(active_mask, dtype=bool)
    mass = jnp.asarray(masses)
    if active.shape != previous.active.shape or mass.shape != previous.mass.shape:
        raise ValueError("Updated population arrays must preserve capacity.")
    born = active & ~previous.active
    incarnation = previous.incarnation + born.astype(jnp.int32)
    valid = jnp.all(jnp.where(active, jnp.isfinite(mass) & (mass > 0.0), mass == 0.0))
    mass = eqx.error_if(
        jnp.where(active, mass, 0.0),
        ~valid,
        "Updated population mass/activity is invalid.",
    )
    return ParticlePopulationState(
        active,
        mass,
        incarnation,
        previous.ever_occupied | active,
        previous.retired & ~born,
    )


__all__ = [
    "ParticleAllocationRequest",
    "ParticleAllocationResult",
    "ParticleDeactivationResult",
    "ParticlePopulationPlan",
    "ParticlePopulationState",
    "ParticlePopulationStatus",
    "ParticleSlotReusePolicy",
    "update_particle_population",
]
