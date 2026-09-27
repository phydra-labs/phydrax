#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization


# Both words all-ones: the reserved 64-bit identity. It is never allocated, marks
# never-occupied slots, and marks an absent parent.
_NO_IDENTITY_WORD = np.uint32(0xFFFFFFFF)


class ParticleSlotReusePolicy(IntEnum):
    NEVER_REUSE = 0
    REUSE_WITH_INCARNATION = 1


class ParticlePopulationStatus(IntEnum):
    SUCCESS = 0
    CAPACITY_EXCEEDED = 1
    INCARNATION_OVERFLOW = 2
    INVALID_REQUEST = 3
    NONFINITE = 4
    IDENTITY_EXHAUSTED = 5


class ParticlePopulationState(StrictModule):
    """Runtime activity, mass, slot incarnation, and persistent particle identity.

    Every particle created by a population carries a global 64-bit identity
    stored as two `uint32` words `(id_hi, id_lo)` and the identity of the
    particle it was created from as `(parent_hi, parent_lo)`. Identity is
    independent of the storage slot: it survives deactivation, slot reuse,
    and slot permutation, and a deactivated slot keeps its last occupant's
    identity until the slot is reallocated. `(next_id_hi, next_id_lo)` is the
    monotone population counter; each creating event assigns consecutive
    identities from it in the event's request order and advances it by the
    number of particles created. The all-ones identity is reserved: it is
    never allocated and marks never-occupied slots and absent parents.
    """

    active: Array
    mass: Array
    incarnation: Array
    ever_occupied: Array
    retired: Array
    id_hi: Array
    id_lo: Array
    parent_hi: Array
    parent_lo: Array
    next_id_hi: Array
    next_id_lo: Array

    @property
    def has_parent(self) -> Array:
        """Mask of slots whose occupant was created from a parent particle."""
        return ~(
            (self.parent_hi == _NO_IDENTITY_WORD) & (self.parent_lo == _NO_IDENTITY_WORD)
        )


class ParticleAllocationRequest(StrictModule):
    """Fixed-width particle creation request.

    `event_ids` define the creating-event order: accepted requests receive
    slots and consecutive identities in ascending event-ID order. `parents`
    gives the `(hi, lo)` identity words of each request's parent particle;
    omitting it creates particles without a parent.
    """

    event_ids: Array
    masses: Array
    valid: Array
    parent_hi: Array
    parent_lo: Array

    def __init__(
        self,
        event_ids: ArrayLike,
        masses: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        parents: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> None:
        valid_ = jnp.asarray(valid, dtype=jnp.bool_)
        if valid_.ndim != 1:
            raise ValueError("Allocation request valid mask must be rank one.")
        if parents is None:
            parent_hi = jnp.full(valid_.shape, _NO_IDENTITY_WORD, dtype=jnp.uint32)
            parent_lo = parent_hi
        else:
            parent_hi = jnp.asarray(parents[0])
            parent_lo = jnp.asarray(parents[1])
            if parent_hi.dtype != jnp.uint32 or parent_lo.dtype != jnp.uint32:
                raise TypeError("Allocation request parent words must be uint32.")
            if parent_hi.shape != valid_.shape or parent_lo.shape != valid_.shape:
                raise ValueError(
                    "Allocation request parents must match the request width."
                )
        self.event_ids = jnp.asarray(event_ids)
        self.masses = jnp.asarray(masses)
        self.valid = valid_
        self.parent_hi = parent_hi
        self.parent_lo = parent_lo


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


def _offset_identity(hi: Array, lo: Array, offset: Array, /) -> tuple[Array, Array]:
    """Add a `uint32` offset to a `(hi, lo)` identity with explicit carry."""
    sum_lo = lo + offset
    return hi + (sum_lo < lo).astype(jnp.uint32), sum_lo


def assign_particle_identities(
    state: ParticlePopulationState,
    created: ArrayLike,
    rank: ArrayLike,
    parent_hi: ArrayLike,
    parent_lo: ArrayLike,
    /,
) -> tuple[ParticlePopulationState, Array]:
    """Assign fresh identities to the particles of one creating event.

    `created` marks the slots filled by the event and `rank` gives each
    created slot's position in the event's request order; the ranks of the
    created slots must be exactly `0, ..., count - 1`. Created slot `s`
    receives identity `next_id + rank[s]` and parent `(parent_hi[s],
    parent_lo[s])`; the counter advances by `count`. Returns the updated state
    and a validity flag that is false when the ranks are not a permutation of
    `0, ..., count - 1` or the identity space below the reserved all-ones
    identity is exhausted. Callers must reject the event when it is false.
    """
    created_ = jnp.asarray(created, dtype=jnp.bool_)
    rank_ = jnp.asarray(rank)
    parent_hi_ = jnp.asarray(parent_hi)
    parent_lo_ = jnp.asarray(parent_lo)
    if not jnp.issubdtype(rank_.dtype, jnp.integer):
        raise TypeError("Identity ranks must be integers.")
    rank_ = rank_.astype(jnp.int32)
    if parent_hi_.dtype != jnp.uint32 or parent_lo_.dtype != jnp.uint32:
        raise TypeError("Parent identity words must be uint32.")
    shape = state.active.shape
    if (
        created_.shape != shape
        or rank_.shape != shape
        or parent_hi_.shape != shape
        or parent_lo_.shape != shape
    ):
        raise ValueError("Identity assignment arrays must have particle-capacity shape.")
    capacity = shape[0]
    count = jnp.sum(created_, dtype=jnp.int32)
    in_range = jnp.all(~created_ | ((rank_ >= 0) & (rank_ < count)))
    rank_hits = (
        jnp.zeros((capacity,), dtype=jnp.int32)
        .at[jnp.where(created_, rank_, capacity)]
        .add(1, mode="drop")
    )
    dense = in_range & jnp.all(rank_hits <= 1)
    # Identities below the reserved all-ones value remain available while the
    # high word is below its maximum; otherwise only `max - next_id_lo` remain.
    exhausted = (state.next_id_hi == _NO_IDENTITY_WORD) & (
        count.astype(jnp.uint32) > _NO_IDENTITY_WORD - state.next_id_lo
    )
    offset = jnp.where(created_, rank_, 0).astype(jnp.uint32)
    new_hi, new_lo = _offset_identity(state.next_id_hi, state.next_id_lo, offset)
    next_hi, next_lo = _offset_identity(
        state.next_id_hi, state.next_id_lo, count.astype(jnp.uint32)
    )
    assigned = ParticlePopulationState(
        state.active,
        state.mass,
        state.incarnation,
        state.ever_occupied,
        state.retired,
        jnp.where(created_, new_hi, state.id_hi),
        jnp.where(created_, new_lo, state.id_lo),
        jnp.where(created_, parent_hi_, state.parent_hi),
        jnp.where(created_, parent_lo_, state.parent_lo),
        next_hi,
        next_lo,
    )
    return assigned, dense & ~exhausted


def _select_state(
    predicate: Array,
    candidate: ParticlePopulationState,
    current: ParticlePopulationState,
    /,
) -> ParticlePopulationState:
    return jax.tree.map(
        lambda proposed, old: jnp.where(predicate, proposed, old), candidate, current
    )


def _initial_population_state(active: Array, mass: Array, /) -> ParticlePopulationState:
    """First population of validated activity/mass; identities follow slot order."""
    no_identity = jnp.full(active.shape, _NO_IDENTITY_WORD, dtype=jnp.uint32)
    zero = jnp.zeros((), dtype=jnp.uint32)
    empty = ParticlePopulationState(
        active,
        mass,
        jnp.where(active, 1, 0).astype(jnp.int32),
        active,
        jnp.zeros_like(active),
        no_identity,
        no_identity,
        no_identity,
        no_identity,
        zero,
        zero,
    )
    # The capacity is far below the 64-bit identity space, so the first
    # assignment cannot exhaust it and its ranks are dense by construction.
    state, _ = assign_particle_identities(
        empty,
        active,
        jnp.cumsum(active, dtype=jnp.int32) - 1,
        no_identity,
        no_identity,
    )
    return state


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
    ) -> None:
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
        """Initial population; active particles receive identities in slot order."""
        structural = self.particles.active_mask
        active = (
            structural
            if active_mask is None
            else jnp.asarray(active_mask, dtype=jnp.bool_)
        )
        mass = self.particles.masses if masses is None else jnp.asarray(masses)
        if active.shape != structural.shape or mass.shape != structural.shape:
            raise ValueError("Population arrays must have particle-capacity shape.")
        active = structural & active
        valid = jnp.all(jnp.where(active, jnp.isfinite(mass) & (mass > 0.0), mass == 0.0))
        mass = eqx.error_if(
            jnp.where(active, mass, 0.0), ~valid, "Population mass/activity is invalid."
        )
        return _initial_population_state(active, mass)

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
        inverse_order = (
            jnp.empty_like(order).at[order].set(jnp.arange(width, dtype=order.dtype))
        )
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
        # Identities follow ascending event-ID order, which is the order of `slots`.
        capacity = state.active.shape[0]
        identity_slots = jnp.where(allocation_mask, slots, capacity)
        identified, identity_valid = assign_particle_identities(
            state,
            jnp.zeros((capacity,), dtype=jnp.bool_)
            .at[identity_slots]
            .set(True, mode="drop"),
            jnp.zeros((capacity,), dtype=jnp.int32)
            .at[identity_slots]
            .set(jnp.cumsum(allocation_mask, dtype=jnp.int32) - 1, mode="drop"),
            jnp.full((capacity,), _NO_IDENTITY_WORD, dtype=jnp.uint32)
            .at[identity_slots]
            .set(request.parent_hi[order], mode="drop"),
            jnp.full((capacity,), _NO_IDENTITY_WORD, dtype=jnp.uint32)
            .at[identity_slots]
            .set(request.parent_lo[order], mode="drop"),
        )
        exhausted = ~identity_valid
        successful = available & finite_request & ~overflow & ~exhausted
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
        identity = _select_state(successful, identified, state)
        candidate = ParticlePopulationState(
            candidate_active,
            candidate_mass,
            candidate_incarnation,
            candidate_ever,
            candidate_retired,
            identity.id_hi,
            identity.id_lo,
            identity.parent_hi,
            identity.parent_lo,
            identity.next_id_hi,
            identity.next_id_lo,
        )
        accepted = _select_state(successful, candidate, state)
        status = jnp.where(
            overflow,
            int(ParticlePopulationStatus.INCARNATION_OVERFLOW),
            jnp.where(
                ~finite_request,
                int(ParticlePopulationStatus.INVALID_REQUEST),
                jnp.where(
                    ~available,
                    int(ParticlePopulationStatus.CAPACITY_EXCEEDED),
                    jnp.where(
                        exhausted,
                        int(ParticlePopulationStatus.IDENTITY_EXHAUSTED),
                        int(ParticlePopulationStatus.SUCCESS),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return ParticleAllocationResult(
            candidate,
            accepted,
            jnp.where(use, safe_slots, -1)[inverse_order],
            use[inverse_order],
            requested,
            jnp.sum(use, dtype=jnp.int32),
            status,
            successful,
            self.plan_id,
        )

    def deactivate(
        self, state: ParticlePopulationState, mask: ArrayLike, /
    ) -> ParticleDeactivationResult:
        """Deactivate particles; their slots keep the retired identities."""
        requested = jnp.asarray(mask, dtype=jnp.bool_)
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
            state.id_hi,
            state.id_lo,
            state.parent_hi,
            state.parent_lo,
            state.next_id_hi,
            state.next_id_lo,
        )
        finite = jnp.isfinite(removed_mass)
        accepted = _select_state(finite, candidate, state)
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
    """Update runtime activity/mass while preserving incarnation identity.

    Newly active slots are one creating event: they receive fresh identities
    without a parent in ascending slot order.
    """

    active = jnp.asarray(active_mask, dtype=jnp.bool_)
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
    no_parent = jnp.full(active.shape, _NO_IDENTITY_WORD, dtype=jnp.uint32)
    updated, identity_valid = assign_particle_identities(
        ParticlePopulationState(
            active,
            mass,
            incarnation,
            previous.ever_occupied | active,
            previous.retired & ~born,
            previous.id_hi,
            previous.id_lo,
            previous.parent_hi,
            previous.parent_lo,
            previous.next_id_hi,
            previous.next_id_lo,
        ),
        born,
        jnp.cumsum(born, dtype=jnp.int32) - 1,
        no_parent,
        no_parent,
    )
    return eqx.error_if(updated, ~identity_valid, "Particle identity space is exhausted.")


__all__ = [
    "ParticleAllocationRequest",
    "ParticleAllocationResult",
    "ParticleDeactivationResult",
    "ParticlePopulationPlan",
    "ParticlePopulationState",
    "ParticlePopulationStatus",
    "ParticleSlotReusePolicy",
    "assign_particle_identities",
    "update_particle_population",
]
