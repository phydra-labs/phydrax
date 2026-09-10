#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class KeyGroupEvidence(NonTrainableState, StrictModule):
    """Auditable counts and independent capacity failures for key grouping."""

    requested_items: Array
    active_items: Array
    invalid_keys: Array
    duplicate_stable_ids: Array
    required_groups: Array
    group_capacity: Array
    maximum_group_size: Array
    member_capacity: Array
    group_overflow: Array
    member_overflow: Array
    successful: Array


class KeyGroupLookup(NonTrainableState, StrictModule):
    """Fixed-shape lookup from logical keys to compact group slots."""

    group_slots: Array
    supported: Array


class KeyGroupPlan(StrictModule):
    """Plan a canonical fixed-capacity grouping of integer-keyed items."""

    item_capacity: int = eqx.field(static=True)
    group_capacity: int = eqx.field(static=True)
    maximum_group_size: int | None = eqx.field(static=True)
    key_upper_bound: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        item_capacity: int,
        group_capacity: int,
        key_upper_bound: int,
        *,
        maximum_group_size: int | None = None,
        case_shape: tuple[int, ...] = (),
    ) -> None:
        if item_capacity < 0:
            raise ValueError("item_capacity must be nonnegative.")
        if group_capacity <= 0:
            raise ValueError("group_capacity must be positive.")
        if key_upper_bound < 0:
            raise ValueError("key_upper_bound must be nonnegative.")
        if maximum_group_size is not None and maximum_group_size <= 0:
            raise ValueError("maximum_group_size must be positive when provided.")
        if any(size <= 0 for size in case_shape):
            raise ValueError("case_shape entries must be positive.")
        self.item_capacity = int(item_capacity)
        self.group_capacity = int(group_capacity)
        self.maximum_group_size = (
            None if maximum_group_size is None else int(maximum_group_size)
        )
        self.key_upper_bound = int(key_upper_bound)
        self.case_shape = tuple(int(size) for size in case_shape)
        self.plan_id = canonical_fingerprint(
            {
                "type": "key-group-plan",
                "item_capacity": self.item_capacity,
                "group_capacity": self.group_capacity,
                "maximum_group_size": self.maximum_group_size,
                "key_upper_bound": self.key_upper_bound,
                "case_shape": self.case_shape,
            }
        )

    def build(
        self,
        keys: ArrayLike,
        valid: ArrayLike,
        *,
        stable_ids: ArrayLike | None = None,
    ) -> KeyGroupState:
        """Group valid items by key in canonical ``(key, stable_id)`` order."""
        key_array = jnp.asarray(keys)
        if not jnp.issubdtype(key_array.dtype, jnp.integer):
            raise TypeError("keys must have an integer dtype.")
        expected_shape = self.case_shape + (self.item_capacity,)
        if key_array.shape != expected_shape:
            raise ValueError(
                f"keys must have shape {expected_shape}; got {key_array.shape}."
            )
        valid_array = jnp.asarray(valid, dtype=bool)
        if valid_array.shape != expected_shape:
            raise ValueError(
                f"valid must have shape {expected_shape}; got {valid_array.shape}."
            )
        if stable_ids is None:
            ids = jnp.broadcast_to(
                jnp.arange(self.item_capacity, dtype=jnp.int32), expected_shape
            )
        else:
            ids = jnp.asarray(stable_ids)
            if not jnp.issubdtype(ids.dtype, jnp.integer):
                raise TypeError("stable_ids must have an integer dtype.")
            if ids.shape != expected_shape:
                raise ValueError(
                    f"stable_ids must have shape {expected_shape}; got {ids.shape}."
                )

        batch_size = 1
        for size in self.case_shape:
            batch_size *= size
        flat_keys = key_array.reshape((batch_size, self.item_capacity))
        flat_valid = valid_array.reshape((batch_size, self.item_capacity))
        flat_ids = ids.reshape((batch_size, self.item_capacity))
        grouped = jax.vmap(
            lambda case_keys, case_valid, case_ids: _build_one(
                case_keys,
                case_valid,
                case_ids,
                group_capacity=self.group_capacity,
                maximum_group_size=self.maximum_group_size,
                key_upper_bound=self.key_upper_bound,
            )
        )(flat_keys, flat_valid, flat_ids)

        item_shape = self.case_shape + (self.item_capacity,)
        group_shape = self.case_shape + (self.group_capacity,)
        scalar_shape = self.case_shape
        return KeyGroupState(
            plan=self,
            item_keys=key_array,
            item_valid=valid_array,
            stable_ids=ids,
            storage_to_logical=grouped.storage_to_logical.reshape(item_shape),
            logical_to_storage=grouped.logical_to_storage.reshape(item_shape),
            sorted_keys=grouped.sorted_keys.reshape(item_shape),
            sorted_item_valid=grouped.sorted_item_valid.reshape(item_shape),
            item_group_slots=grouped.item_group_slots.reshape(item_shape),
            group_keys=grouped.group_keys.reshape(group_shape),
            group_active=grouped.group_active.reshape(group_shape),
            group_starts=grouped.group_starts.reshape(group_shape),
            group_counts=grouped.group_counts.reshape(group_shape),
            evidence=KeyGroupEvidence(
                requested_items=grouped.requested_items.reshape(scalar_shape),
                active_items=grouped.active_items.reshape(scalar_shape),
                invalid_keys=grouped.invalid_keys.reshape(scalar_shape),
                duplicate_stable_ids=grouped.duplicate_stable_ids.reshape(scalar_shape),
                required_groups=grouped.required_groups.reshape(scalar_shape),
                group_capacity=jnp.full(
                    scalar_shape, self.group_capacity, dtype=jnp.int32
                ),
                maximum_group_size=grouped.maximum_group_size.reshape(scalar_shape),
                member_capacity=jnp.full(
                    scalar_shape,
                    self.maximum_group_size or self.item_capacity,
                    dtype=jnp.int32,
                ),
                group_overflow=grouped.group_overflow.reshape(scalar_shape),
                member_overflow=grouped.member_overflow.reshape(scalar_shape),
                successful=grouped.successful.reshape(scalar_shape),
            ),
        )


class KeyGroupState(NonTrainableState, StrictModule):
    """Canonical compact groups and reversible item placement."""

    plan: KeyGroupPlan
    item_keys: Array
    item_valid: Array
    stable_ids: Array
    storage_to_logical: Array
    logical_to_storage: Array
    sorted_keys: Array
    sorted_item_valid: Array
    item_group_slots: Array
    group_keys: Array
    group_active: Array
    group_starts: Array
    group_counts: Array
    evidence: KeyGroupEvidence

    def lookup(
        self,
        keys: ArrayLike,
        *,
        valid: ArrayLike | None = None,
    ) -> KeyGroupLookup:
        """Look up keys without allocating a dense reverse map."""
        query = jnp.asarray(keys)
        if not jnp.issubdtype(query.dtype, jnp.integer):
            raise TypeError("lookup keys must have an integer dtype.")
        if query.shape[: len(self.plan.case_shape)] != self.plan.case_shape:
            raise ValueError(
                "lookup keys must begin with the grouping case shape "
                f"{self.plan.case_shape}; got {query.shape}."
            )
        if valid is None:
            query_valid = jnp.ones(query.shape, dtype=bool)
        else:
            query_valid = jnp.asarray(valid, dtype=bool)
            if query_valid.shape != query.shape:
                raise ValueError(
                    f"lookup valid must have shape {query.shape}; got "
                    f"{query_valid.shape}."
                )

        batch_size = 1
        for size in self.plan.case_shape:
            batch_size *= size
        query_tail = query.shape[len(self.plan.case_shape) :]
        flat_query = query.reshape((batch_size, -1))
        flat_valid = query_valid.reshape((batch_size, -1))
        flat_group_keys = self.group_keys.reshape((batch_size, self.plan.group_capacity))
        flat_group_active = self.group_active.reshape(
            (batch_size, self.plan.group_capacity)
        )
        slots, supported = jax.vmap(
            lambda case_groups, case_active, case_query, case_valid: _lookup_one(
                case_groups,
                case_active,
                case_query,
                case_valid,
                key_upper_bound=self.plan.key_upper_bound,
            )
        )(flat_group_keys, flat_group_active, flat_query, flat_valid)
        result_shape = self.plan.case_shape + query_tail
        supported = supported.reshape(result_shape)
        successful = self.evidence.successful.reshape(
            self.plan.case_shape + (1,) * len(query_tail)
        )
        supported = supported & successful
        slots = slots.reshape(result_shape)
        return KeyGroupLookup(
            group_slots=jnp.where(supported, slots, 0),
            supported=supported,
        )


class KeyGroupTransition(NonTrainableState, StrictModule):
    """Logical-key alignment between two fixed-capacity group states."""

    previous: KeyGroupState
    candidate: KeyGroupState
    previous_to_candidate: Array
    previous_retained: Array
    candidate_to_previous: Array
    candidate_retained: Array
    topology_changed: Array
    successful: Array


def align_key_groups(
    previous: KeyGroupState,
    candidate: KeyGroupState,
) -> KeyGroupTransition:
    """Align group slots by key without treating storage position as identity."""
    if previous.plan.case_shape != candidate.plan.case_shape:
        raise ValueError("previous and candidate case shapes must match.")
    previous_lookup = candidate.lookup(previous.group_keys, valid=previous.group_active)
    candidate_lookup = previous.lookup(candidate.group_keys, valid=candidate.group_active)
    changed = (
        previous.evidence.required_groups != candidate.evidence.required_groups
    ) | jnp.any(
        (previous.group_active != candidate.group_active)
        | (
            previous.group_active
            & candidate.group_active
            & (previous.group_keys != candidate.group_keys)
        ),
        axis=-1,
    )
    return KeyGroupTransition(
        previous=previous,
        candidate=candidate,
        previous_to_candidate=previous_lookup.group_slots,
        previous_retained=previous_lookup.supported,
        candidate_to_previous=candidate_lookup.group_slots,
        candidate_retained=candidate_lookup.supported,
        topology_changed=changed,
        successful=candidate.evidence.successful,
    )


class _GroupedArrays(NamedTuple):
    storage_to_logical: Array
    logical_to_storage: Array
    sorted_keys: Array
    sorted_item_valid: Array
    item_group_slots: Array
    group_keys: Array
    group_active: Array
    group_starts: Array
    group_counts: Array
    requested_items: Array
    active_items: Array
    invalid_keys: Array
    duplicate_stable_ids: Array
    required_groups: Array
    maximum_group_size: Array
    group_overflow: Array
    member_overflow: Array
    successful: Array


def _build_one(
    keys: Array,
    valid: Array,
    stable_ids: Array,
    *,
    group_capacity: int,
    maximum_group_size: int | None,
    key_upper_bound: int,
) -> _GroupedArrays:
    item_capacity = keys.shape[0]
    sentinel = jnp.asarray(key_upper_bound + 1, dtype=keys.dtype)
    if item_capacity == 0:
        group_shape = (group_capacity,)
        return _GroupedArrays(
            storage_to_logical=jnp.zeros((0,), dtype=jnp.int32),
            logical_to_storage=jnp.zeros((0,), dtype=jnp.int32),
            sorted_keys=jnp.zeros((0,), dtype=keys.dtype),
            sorted_item_valid=jnp.zeros((0,), dtype=bool),
            item_group_slots=jnp.zeros((0,), dtype=jnp.int32),
            group_keys=jnp.full(group_shape, sentinel),
            group_active=jnp.zeros(group_shape, dtype=bool),
            group_starts=jnp.zeros(group_shape, dtype=jnp.int32),
            group_counts=jnp.zeros(group_shape, dtype=jnp.int32),
            requested_items=jnp.asarray(0, dtype=jnp.int32),
            active_items=jnp.asarray(0, dtype=jnp.int32),
            invalid_keys=jnp.asarray(0, dtype=jnp.int32),
            duplicate_stable_ids=jnp.asarray(0, dtype=jnp.int32),
            required_groups=jnp.asarray(0, dtype=jnp.int32),
            maximum_group_size=jnp.asarray(0, dtype=jnp.int32),
            group_overflow=jnp.asarray(False),
            member_overflow=jnp.asarray(False),
            successful=jnp.asarray(True),
        )
    key_valid = valid & (keys >= 0) & (keys <= key_upper_bound)
    safe_keys = jnp.where(key_valid, keys, sentinel)
    original = jnp.arange(item_capacity, dtype=jnp.int32)
    order = jnp.lexsort(
        (
            original,
            stable_ids,
            safe_keys,
            (~key_valid).astype(jnp.int32),
        )
    ).astype(jnp.int32)
    sorted_keys = safe_keys[order]
    sorted_valid = key_valid[order]

    previous_valid = jnp.concatenate((jnp.zeros((1,), dtype=bool), sorted_valid[:-1]))
    previous_keys = jnp.concatenate((jnp.full((1,), sentinel), sorted_keys[:-1]))
    group_start_mask = sorted_valid & ((~previous_valid) | (sorted_keys != previous_keys))
    sorted_group_slots = jnp.cumsum(group_start_mask.astype(jnp.int32)) - 1
    sorted_group_slots = jnp.where(sorted_valid, sorted_group_slots, -1)
    required_groups = jnp.sum(group_start_mask, dtype=jnp.int32)
    active_items = jnp.sum(sorted_valid, dtype=jnp.int32)
    start_positions = jnp.nonzero(group_start_mask, size=group_capacity, fill_value=0)[
        0
    ].astype(jnp.int32)
    group_active = jnp.arange(group_capacity, dtype=jnp.int32) < required_groups
    group_keys = jnp.where(group_active, sorted_keys[start_positions], sentinel)
    group_starts = jnp.where(group_active, start_positions, 0)
    following_starts = jnp.concatenate((group_starts[1:], active_items[None]))
    following_active = jnp.concatenate((group_active[1:], jnp.zeros((1,), dtype=bool)))
    next_starts = jnp.where(following_active, following_starts, active_items)
    group_counts = jnp.where(group_active, next_starts - group_starts, 0)
    maximum_size = jnp.max(group_counts, initial=0)

    logical_to_storage = (
        jnp.zeros((item_capacity,), dtype=jnp.int32).at[order].set(original)
    )
    item_group_slots = (
        jnp.full((item_capacity,), -1, dtype=jnp.int32).at[order].set(sorted_group_slots)
    )

    id_order = jnp.lexsort(
        (
            original,
            stable_ids,
            (~key_valid).astype(jnp.int32),
        )
    ).astype(jnp.int32)
    ids_by_id = stable_ids[id_order]
    valid_by_id = key_valid[id_order]
    duplicate_ids = valid_by_id[1:] & valid_by_id[:-1] & (ids_by_id[1:] == ids_by_id[:-1])
    duplicate_stable_ids = jnp.sum(duplicate_ids, dtype=jnp.int32)

    requested_items = jnp.sum(valid, dtype=jnp.int32)
    invalid_keys = requested_items - active_items
    group_overflow = required_groups > group_capacity
    member_limit = maximum_group_size or item_capacity
    member_overflow = maximum_size > member_limit
    successful = (
        (invalid_keys == 0)
        & (duplicate_stable_ids == 0)
        & (~group_overflow)
        & (~member_overflow)
    )
    return _GroupedArrays(
        storage_to_logical=order,
        logical_to_storage=logical_to_storage,
        sorted_keys=sorted_keys,
        sorted_item_valid=sorted_valid,
        item_group_slots=item_group_slots,
        group_keys=group_keys,
        group_active=group_active,
        group_starts=group_starts,
        group_counts=group_counts,
        requested_items=requested_items,
        active_items=active_items,
        invalid_keys=invalid_keys,
        duplicate_stable_ids=duplicate_stable_ids,
        required_groups=required_groups,
        maximum_group_size=maximum_size,
        group_overflow=group_overflow,
        member_overflow=member_overflow,
        successful=successful,
    )


def _lookup_one(
    group_keys: Array,
    group_active: Array,
    query: Array,
    query_valid: Array,
    *,
    key_upper_bound: int,
) -> tuple[Array, Array]:
    positions = jnp.searchsorted(group_keys, query, side="left").astype(jnp.int32)
    safe_positions = jnp.clip(positions, 0, group_keys.shape[0] - 1)
    valid_keys = query_valid & (query >= 0) & (query <= key_upper_bound)
    supported = (
        valid_keys
        & (positions < group_keys.shape[0])
        & group_active[safe_positions]
        & (group_keys[safe_positions] == query)
    )
    return jnp.where(supported, safe_positions, 0), supported


__all__ = [
    "KeyGroupEvidence",
    "KeyGroupLookup",
    "KeyGroupPlan",
    "KeyGroupState",
    "KeyGroupTransition",
    "align_key_groups",
]
