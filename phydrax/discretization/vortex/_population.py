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


_EVENT_NONE = 0
_EVENT_INSERT = 1
_EVENT_DEACTIVATE = 2
_EVENT_MERGE = 3
_EVENT_SPLIT = 4
_EVENT_PRUNE = 5


class VortexPopulationState(StrictModule):
    positions: Array
    strength: Array
    core_radius: Array
    volume: Array
    active_mask: Array
    stable_ids: Array
    parent_ids: Array
    source_codes: Array
    age: Array
    next_stable_id: Array


class VortexPopulationEventJournal(StrictModule):
    event_kind: Array
    primary_id: Array
    secondary_id: Array
    created_id: Array
    strength_residual: Array
    accepted: Array
    write_index: Array


class VortexPopulationEvidence(StrictModule):
    active_count_before: Array
    active_count_after: Array
    total_strength_before: Array
    total_strength_after: Array
    strength_residual: Array
    impulse_residual: Array
    duplicate_id_count: Array
    capacity_remaining: Array
    finite: Array


class VortexPopulationTransition(StrictModule):
    candidate: VortexPopulationState
    accepted: VortexPopulationState
    journal: VortexPopulationEventJournal
    evidence: VortexPopulationEvidence
    successful: Array
    transition_id: str = eqx.field(static=True)


class VortexPopulationPlan(StrictModule, NonTrainableState):
    capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    journal_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, capacity: int, dimension: int, /, *, journal_capacity: int = 256):
        capacity_, dimension_, journal_capacity_ = (
            int(capacity),
            int(dimension),
            int(journal_capacity),
        )
        if capacity_ <= 0 or dimension_ not in (2, 3) or journal_capacity_ <= 0:
            raise ValueError(
                "Vortex population capacity/dimension/journal capacity is invalid."
            )
        self.capacity = capacity_
        self.dimension = dimension_
        self.journal_capacity = journal_capacity_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vortex-population-plan",
                "capacity": capacity_,
                "dimension": dimension_,
                "journal_capacity": journal_capacity_,
            }
        )

    def initialize(
        self,
        positions: ArrayLike,
        strength: ArrayLike,
        core_radius: ArrayLike,
        volume: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        stable_ids: ArrayLike | None = None,
        source_codes: ArrayLike | None = None,
    ) -> tuple[VortexPopulationState, VortexPopulationEventJournal]:
        position = jnp.asarray(positions)
        strength_ = jnp.asarray(strength, dtype=position.dtype)
        core = jnp.asarray(core_radius, dtype=position.dtype)
        volume_ = jnp.asarray(volume, dtype=position.dtype)
        expected_strength = (
            (self.capacity,) if self.dimension == 2 else (self.capacity, 3)
        )
        if (
            position.shape != (self.capacity, self.dimension)
            or strength_.shape != expected_strength
        ):
            raise ValueError("Vortex population position/strength shapes are invalid.")
        if core.shape != (self.capacity,) or volume_.shape != (self.capacity,):
            raise ValueError("Vortex population core/volume shapes are invalid.")
        active = (
            jnp.ones((self.capacity,), dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        ids = (
            jnp.arange(self.capacity, dtype=jnp.int64)
            if stable_ids is None
            else jnp.asarray(stable_ids, dtype=jnp.int64)
        )
        codes = (
            jnp.zeros((self.capacity,), dtype=jnp.int32)
            if source_codes is None
            else jnp.asarray(source_codes, dtype=jnp.int32)
        )
        if (
            active.shape != (self.capacity,)
            or ids.shape != active.shape
            or codes.shape != active.shape
        ):
            raise ValueError("Vortex population active/ID/source shapes are invalid.")
        ids = jnp.where(active, ids, -1)
        finite = (
            jnp.all(jnp.where(active[:, None], jnp.isfinite(position), True))
            & jnp.all(
                jnp.where(
                    active if self.dimension == 2 else active[:, None],
                    jnp.isfinite(strength_),
                    True,
                )
            )
            & jnp.all(
                jnp.where(
                    active,
                    jnp.isfinite(core)
                    & (core > 0.0)
                    & jnp.isfinite(volume_)
                    & (volume_ > 0.0),
                    True,
                )
            )
        )
        position = eqx.error_if(
            position,
            ~finite,
            "Active vortex population values must be finite and positive where required.",
        )
        state = VortexPopulationState(
            jnp.where(active[:, None], position, 0.0),
            jnp.where(active if self.dimension == 2 else active[:, None], strength_, 0.0),
            jnp.where(active, core, 1.0),
            jnp.where(active, volume_, 1.0),
            active,
            ids,
            jnp.full((self.capacity,), -1, dtype=jnp.int64),
            codes,
            jnp.zeros((self.capacity,), dtype=position.dtype),
            jnp.max(jnp.where(active, ids, -1), initial=-1) + 1,
        )
        journal = VortexPopulationEventJournal(
            jnp.zeros((self.journal_capacity,), dtype=jnp.int8),
            jnp.full((self.journal_capacity,), -1, dtype=jnp.int64),
            jnp.full((self.journal_capacity,), -1, dtype=jnp.int64),
            jnp.full((self.journal_capacity,), -1, dtype=jnp.int64),
            jnp.zeros(
                (self.journal_capacity, self.dimension if self.dimension == 3 else 1),
                dtype=position.dtype,
            ),
            jnp.zeros((self.journal_capacity,), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
        )
        return state, journal

    def _total(self, state: VortexPopulationState, /) -> Array:
        mask = state.active_mask if self.dimension == 2 else state.active_mask[:, None]
        return jnp.sum(jnp.where(mask, state.strength, 0.0), axis=0)

    def _impulse(self, state: VortexPopulationState, /) -> Array:
        if self.dimension == 2:
            return jnp.sum(
                jnp.where(
                    state.active_mask[:, None],
                    state.strength[:, None]
                    * jnp.stack((-state.positions[:, 1], state.positions[:, 0]), axis=-1),
                    0.0,
                ),
                axis=0,
            )
        return 0.5 * jnp.sum(
            jnp.where(
                state.active_mask[:, None],
                jnp.cross(state.positions, state.strength),
                0.0,
            ),
            axis=0,
        )

    def _journal(
        self,
        journal: VortexPopulationEventJournal,
        event_kind: int,
        primary_id: Array,
        secondary_id: Array,
        created_id: Array,
        residual: Array,
        accepted: Array,
        /,
    ) -> VortexPopulationEventJournal:
        slot = journal.write_index % self.journal_capacity
        residual_vector = jnp.atleast_1d(residual)
        if self.dimension == 2:
            residual_vector = residual_vector.reshape((1,))
        return VortexPopulationEventJournal(
            journal.event_kind.at[slot].set(event_kind),
            journal.primary_id.at[slot].set(primary_id),
            journal.secondary_id.at[slot].set(secondary_id),
            journal.created_id.at[slot].set(created_id),
            journal.strength_residual.at[slot].set(residual_vector),
            journal.accepted.at[slot].set(accepted),
            journal.write_index + 1,
        )

    def _transition(
        self,
        previous: VortexPopulationState,
        candidate: VortexPopulationState,
        journal: VortexPopulationEventJournal,
        /,
        *,
        event_kind: int,
        primary_id: Array,
        secondary_id: Array,
        created_id: Array,
        precondition: Array,
        expected_strength_delta: Array | None = None,
        expected_impulse_delta: Array | None = None,
    ) -> VortexPopulationTransition:
        before = self._total(previous)
        after = self._total(candidate)
        expected_strength = (
            jnp.zeros_like(before)
            if expected_strength_delta is None
            else jnp.asarray(expected_strength_delta, dtype=before.dtype)
        )
        impulse_before = self._impulse(previous)
        impulse_after = self._impulse(candidate)
        expected_impulse = (
            jnp.zeros_like(impulse_before)
            if expected_impulse_delta is None
            else jnp.asarray(expected_impulse_delta, dtype=impulse_before.dtype)
        )
        residual = after - before - expected_strength
        impulse_residual = impulse_after - impulse_before - expected_impulse
        active_ids = jnp.where(candidate.active_mask, candidate.stable_ids, -1)
        duplicate_count = (
            jnp.sum(
                (active_ids[:, None] == active_ids[None, :])
                & (active_ids[:, None] >= 0)
                & ~jnp.eye(self.capacity, dtype=bool),
                dtype=jnp.int32,
            )
            // 2
        )
        finite = (
            jnp.all(jnp.isfinite(candidate.positions))
            & jnp.all(jnp.isfinite(candidate.strength))
            & jnp.all(jnp.isfinite(candidate.core_radius))
            & jnp.all(jnp.isfinite(candidate.volume))
        )
        scale = jnp.maximum(jnp.max(jnp.abs(before)), 1.0)
        impulse_scale = jnp.maximum(jnp.max(jnp.abs(impulse_before)), 1.0)
        epsilon = jnp.finfo(candidate.positions.dtype).eps
        conservative = jnp.max(jnp.abs(residual)) <= 256 * epsilon * scale
        impulse_conservative = (
            jnp.max(jnp.abs(impulse_residual)) <= 512 * epsilon * impulse_scale
        )
        successful = (
            precondition
            & finite
            & conservative
            & impulse_conservative
            & (duplicate_count == 0)
        )
        accepted = jax_tree_select(successful, candidate, previous)
        evidence = VortexPopulationEvidence(
            jnp.sum(previous.active_mask, dtype=jnp.int32),
            jnp.sum(accepted.active_mask, dtype=jnp.int32),
            before,
            self._total(accepted),
            self._total(accepted) - before - expected_strength,
            self._impulse(accepted) - impulse_before - expected_impulse,
            duplicate_count,
            self.capacity - jnp.sum(accepted.active_mask, dtype=jnp.int32),
            finite,
        )
        updated_journal = self._journal(
            journal,
            event_kind,
            primary_id,
            secondary_id,
            created_id,
            residual,
            successful,
        )
        return VortexPopulationTransition(
            candidate,
            accepted,
            updated_journal,
            evidence,
            successful,
            canonical_fingerprint(
                {
                    "kind": "vortex-population-transition",
                    "plan": self.plan_id,
                    "event_kind": event_kind,
                }
            ),
        )

    def insert(
        self,
        state: VortexPopulationState,
        journal: VortexPopulationEventJournal,
        position: ArrayLike,
        strength: ArrayLike,
        core_radius: ArrayLike,
        volume: ArrayLike,
        /,
        *,
        parent_id: ArrayLike = -1,
        source_code: ArrayLike = 0,
    ) -> VortexPopulationTransition:
        free = jnp.nonzero(~state.active_mask, size=1, fill_value=-1)[0][0]
        valid = free >= 0
        slot = jnp.where(valid, free, 0)
        position_ = jnp.asarray(position, dtype=state.positions.dtype)
        strength_ = jnp.asarray(strength, dtype=state.strength.dtype)
        core = jnp.asarray(core_radius, dtype=state.core_radius.dtype)
        volume_ = jnp.asarray(volume, dtype=state.volume.dtype)
        expected_strength = () if self.dimension == 2 else (3,)
        if (
            position_.shape != (self.dimension,)
            or strength_.shape != expected_strength
            or core.shape != ()
            or volume_.shape != ()
        ):
            raise ValueError("Inserted vortex element shapes are invalid.")
        new_id = state.next_stable_id
        candidate = VortexPopulationState(
            state.positions.at[slot].set(position_),
            state.strength.at[slot].set(strength_),
            state.core_radius.at[slot].set(core),
            state.volume.at[slot].set(volume_),
            state.active_mask.at[slot].set(valid),
            state.stable_ids.at[slot].set(
                jnp.where(valid, new_id, state.stable_ids[slot])
            ),
            state.parent_ids.at[slot].set(
                jnp.where(
                    valid, jnp.asarray(parent_id, dtype=jnp.int64), state.parent_ids[slot]
                )
            ),
            state.source_codes.at[slot].set(
                jnp.where(
                    valid,
                    jnp.asarray(source_code, dtype=jnp.int32),
                    state.source_codes[slot],
                )
            ),
            state.age.at[slot].set(0.0),
            state.next_stable_id + jnp.where(valid, 1, 0),
        )
        inserted_impulse = (
            strength_ * jnp.stack((-position_[1], position_[0]))
            if self.dimension == 2
            else 0.5 * jnp.cross(position_, strength_)
        )
        return self._transition(
            state,
            candidate,
            journal,
            event_kind=_EVENT_INSERT,
            primary_id=jnp.asarray(parent_id, dtype=jnp.int64),
            secondary_id=jnp.asarray(-1, dtype=jnp.int64),
            created_id=new_id,
            precondition=(
                valid
                & jnp.all(jnp.isfinite(position_))
                & jnp.all(jnp.isfinite(strength_))
                & jnp.isfinite(core)
                & (core > 0.0)
                & jnp.isfinite(volume_)
                & (volume_ > 0.0)
            ),
            expected_strength_delta=strength_,
            expected_impulse_delta=inserted_impulse,
        )

    def deactivate(
        self,
        state: VortexPopulationState,
        journal: VortexPopulationEventJournal,
        stable_id: ArrayLike,
        /,
    ) -> VortexPopulationTransition:
        identifier = jnp.asarray(stable_id, dtype=jnp.int64)
        match = state.active_mask & (state.stable_ids == identifier)
        count = jnp.sum(match, dtype=jnp.int32)
        # Deactivation is conservative only for exactly zero-strength carriers.
        strength_mask = match if self.dimension == 2 else match[:, None]
        zero_strength = (
            jnp.max(jnp.abs(jnp.where(strength_mask, state.strength, 0.0))) == 0.0
        )
        candidate = VortexPopulationState(
            state.positions,
            state.strength,
            state.core_radius,
            state.volume,
            state.active_mask & ~match,
            jnp.where(match, -1, state.stable_ids),
            state.parent_ids,
            state.source_codes,
            state.age,
            state.next_stable_id,
        )
        return self._transition(
            state,
            candidate,
            journal,
            event_kind=_EVENT_DEACTIVATE,
            primary_id=identifier,
            secondary_id=jnp.asarray(-1, dtype=jnp.int64),
            created_id=jnp.asarray(-1, dtype=jnp.int64),
            precondition=(count == 1) & zero_strength,
        )

    def merge(
        self,
        state: VortexPopulationState,
        journal: VortexPopulationEventJournal,
        first_id: ArrayLike,
        second_id: ArrayLike,
        /,
    ) -> VortexPopulationTransition:
        first_identifier = jnp.asarray(first_id, dtype=jnp.int64)
        second_identifier = jnp.asarray(second_id, dtype=jnp.int64)
        first_match = state.active_mask & (state.stable_ids == first_identifier)
        second_match = state.active_mask & (state.stable_ids == second_identifier)
        first = jnp.argmax(first_match)
        second = jnp.argmax(second_match)
        first_strength, second_strength = state.strength[first], state.strength[second]
        magnitude_first = jnp.linalg.norm(jnp.atleast_1d(first_strength))
        magnitude_second = jnp.linalg.norm(jnp.atleast_1d(second_strength))
        weight_total = jnp.maximum(
            magnitude_first + magnitude_second, jnp.finfo(state.positions.dtype).tiny
        )
        merged_position = (
            magnitude_first * state.positions[first]
            + magnitude_second * state.positions[second]
        ) / weight_total
        merged_strength = first_strength + second_strength
        merged_volume = state.volume[first] + state.volume[second]
        merged_core = jnp.sqrt(
            (
                state.volume[first] * state.core_radius[first] ** 2
                + state.volume[second] * state.core_radius[second] ** 2
            )
            / merged_volume
        )
        candidate = VortexPopulationState(
            state.positions.at[first].set(merged_position),
            state.strength.at[first].set(merged_strength).at[second].set(0.0),
            state.core_radius.at[first].set(merged_core),
            state.volume.at[first].set(merged_volume),
            state.active_mask.at[second].set(False),
            state.stable_ids.at[second].set(-1),
            state.parent_ids.at[first].set(first_identifier),
            state.source_codes,
            state.age.at[first].set(jnp.maximum(state.age[first], state.age[second])),
            state.next_stable_id,
        )
        valid = (
            (first_identifier != second_identifier)
            & (jnp.sum(first_match) == 1)
            & (jnp.sum(second_match) == 1)
        )
        return self._transition(
            state,
            candidate,
            journal,
            event_kind=_EVENT_MERGE,
            primary_id=first_identifier,
            secondary_id=second_identifier,
            created_id=first_identifier,
            precondition=valid,
        )

    def split(
        self,
        state: VortexPopulationState,
        journal: VortexPopulationEventJournal,
        stable_id: ArrayLike,
        offset: ArrayLike,
        /,
    ) -> VortexPopulationTransition:
        identifier = jnp.asarray(stable_id, dtype=jnp.int64)
        match = state.active_mask & (state.stable_ids == identifier)
        source_slot = jnp.argmax(match)
        free_slot = jnp.nonzero(~state.active_mask, size=1, fill_value=-1)[0][0]
        valid = (jnp.sum(match) == 1) & (free_slot >= 0)
        target_slot = jnp.where(free_slot >= 0, free_slot, 0)
        offset_ = jnp.asarray(offset, dtype=state.positions.dtype)
        if offset_.shape != (self.dimension,):
            raise ValueError("Split offset must match population dimension.")
        new_id = state.next_stable_id
        half_strength = 0.5 * state.strength[source_slot]
        half_volume = 0.5 * state.volume[source_slot]
        candidate = VortexPopulationState(
            state.positions.at[source_slot]
            .set(state.positions[source_slot] - offset_)
            .at[target_slot]
            .set(state.positions[source_slot] + offset_),
            state.strength.at[source_slot]
            .set(half_strength)
            .at[target_slot]
            .set(half_strength),
            state.core_radius.at[target_slot].set(state.core_radius[source_slot]),
            state.volume.at[source_slot]
            .set(half_volume)
            .at[target_slot]
            .set(half_volume),
            state.active_mask.at[target_slot].set(valid),
            state.stable_ids.at[target_slot].set(
                jnp.where(valid, new_id, state.stable_ids[target_slot])
            ),
            state.parent_ids.at[target_slot].set(identifier),
            state.source_codes.at[target_slot].set(state.source_codes[source_slot]),
            state.age.at[target_slot].set(state.age[source_slot]),
            state.next_stable_id + jnp.where(valid, 1, 0),
        )
        return self._transition(
            state,
            candidate,
            journal,
            event_kind=_EVENT_SPLIT,
            primary_id=identifier,
            secondary_id=jnp.asarray(-1, dtype=jnp.int64),
            created_id=new_id,
            precondition=valid & jnp.all(jnp.isfinite(offset_)),
        )


def jax_tree_select(
    condition: ArrayLike,
    candidate: VortexPopulationState,
    previous: VortexPopulationState,
    /,
) -> VortexPopulationState:
    choose = jnp.asarray(condition, dtype=bool)
    return VortexPopulationState(
        jnp.where(choose, candidate.positions, previous.positions),
        jnp.where(choose, candidate.strength, previous.strength),
        jnp.where(choose, candidate.core_radius, previous.core_radius),
        jnp.where(choose, candidate.volume, previous.volume),
        jnp.where(choose, candidate.active_mask, previous.active_mask),
        jnp.where(choose, candidate.stable_ids, previous.stable_ids),
        jnp.where(choose, candidate.parent_ids, previous.parent_ids),
        jnp.where(choose, candidate.source_codes, previous.source_codes),
        jnp.where(choose, candidate.age, previous.age),
        jnp.where(choose, candidate.next_stable_id, previous.next_stable_id),
    )


__all__ = [
    "VortexPopulationEventJournal",
    "VortexPopulationEvidence",
    "VortexPopulationPlan",
    "VortexPopulationState",
    "VortexPopulationTransition",
]
