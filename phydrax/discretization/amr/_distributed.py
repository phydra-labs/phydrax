#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike

from ..._execution_runtime import ExecutionGroup
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._distributed_field import _color_directed_pairs
from ._core import (
    BlockHierarchyPlan,
    BlockHierarchyState,
    BlockHierarchyTopology,
    BlockLevelState,
)
from ._fd_halo import (
    FDAMRFillPatchPlan,
    FDAMRFillPatchResult,
    FDAMRFillPatchWorkspace,
    FDAMRPhysicalBoundaryRequest,
    FillPatchSource,
)
from ._fd_runtime import PreparedFDAMRHierarchy
from ._topology_compiler import BlockTopologyCompileResult


RouteKind = Literal["same_level", "coarse_fine", "interface"]


def _part_spec(axis_name: str, rank: int, /) -> PartitionSpec:
    return PartitionSpec(axis_name, *((None,) * (rank - 1)))


def _place_by_part(
    value: ArrayLike,
    mesh: Mesh | None,
    axis_name: str,
    /,
    *,
    dtype: Any | None = None,
) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if mesh is None:
        return array
    return jax.device_put(array, NamedSharding(mesh, _part_spec(axis_name, array.ndim)))


def _morton_key(logical_index: Sequence[int], lattice: Sequence[int], /) -> int:
    """Interleave logical-index bits without imposing a machine-word limit."""

    bit_count = max(1, max(int(extent - 1).bit_length() for extent in lattice))
    result = 0
    dimension = len(logical_index)
    for bit in range(bit_count):
        for axis, value in enumerate(logical_index):
            result |= ((int(value) >> bit) & 1) << (bit * dimension + axis)
    return result


def _active_costs(
    value: ArrayLike | None,
    active_count: int,
    maximum_blocks: int,
    /,
) -> np.ndarray:
    if value is None:
        return np.ones((active_count,), dtype=np.float64)
    costs = np.asarray(value, dtype=np.float64)
    if costs.shape == (maximum_blocks,):
        costs = costs[:active_count]
    elif costs.shape != (active_count,):
        raise ValueError(
            "Each block-cost vector must match either active blocks or level capacity."
        )
    if np.any(~np.isfinite(costs)) or np.any(costs <= 0.0):
        raise ValueError("Active AMR block costs must be finite and strictly positive.")
    return costs


def _contiguous_owners(costs: np.ndarray, part_count: int, /) -> np.ndarray:
    """Split one locality order into deterministic, nonempty contiguous ranges."""

    count = costs.size
    if count == 0:
        return np.empty((0,), dtype=np.int32)
    used_parts = min(count, part_count)
    prefix = np.concatenate((np.zeros((1,), dtype=np.float64), np.cumsum(costs)))
    total = float(prefix[-1])
    boundaries = [0]
    previous = 0
    for part in range(1, used_parts):
        first = previous + 1
        last = count - (used_parts - part)
        target = total * part / used_parts
        candidates = np.arange(first, last + 1, dtype=np.int32)
        errors = np.abs(prefix[candidates] - target)
        boundary = int(candidates[int(np.argmin(errors))])
        boundaries.append(boundary)
        previous = boundary
    boundaries.append(count)
    owners = np.empty((count,), dtype=np.int32)
    for part, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        owners[start:stop] = part
    return owners


class _BlockAMRLevelLayout(StrictModule, NonTrainableState):
    block_owner: Array
    local_block_slots: Array
    local_block_valid: Array
    canonical_to_local: Array
    stable_block_ids: Array
    slot_identities: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    active_count: int = eqx.field(static=True)
    maximum_blocks: int = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    local_block_capacity: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        block_owner: np.ndarray,
        local_block_slots: np.ndarray,
        local_block_valid: np.ndarray,
        canonical_to_local: np.ndarray,
        stable_block_ids: np.ndarray,
        /,
        *,
        mesh: Mesh | None,
        axis_name: str,
    ):
        identities = tuple(
            tuple(local_block_slots[part, local_block_valid[part]])
            for part in range(local_block_slots.shape[0])
        )
        self.block_owner = jnp.asarray(block_owner, dtype=jnp.int32)
        self.local_block_slots = _place_by_part(
            local_block_slots, mesh, axis_name, dtype=jnp.int32
        )
        self.local_block_valid = _place_by_part(
            local_block_valid, mesh, axis_name, dtype=jnp.bool_
        )
        self.canonical_to_local = jnp.asarray(canonical_to_local, dtype=jnp.int32)
        self.stable_block_ids = jnp.asarray(stable_block_ids, dtype=jnp.int32)
        self.slot_identities = identities
        self.active_count = int(np.count_nonzero(block_owner >= 0))
        self.maximum_blocks = block_owner.size
        self.part_count = local_block_slots.shape[0]
        self.local_block_capacity = local_block_slots.shape[1]
        self.layout_id = canonical_fingerprint(
            {
                "kind": "distributed-block-amr-level-layout",
                "owner": array_tree_fingerprint(block_owner),
                "local_slots": array_tree_fingerprint(local_block_slots),
                "stable_ids": array_tree_fingerprint(stable_block_ids),
            }
        )

    def pack(self, canonical_values: ArrayLike, /) -> Array:
        values = jnp.asarray(canonical_values)
        if values.ndim == 0 or values.shape[0] != self.maximum_blocks:
            raise ValueError("Canonical block values must begin with level capacity.")
        safe_slots = jnp.maximum(self.local_block_slots, 0)
        packed = values[safe_slots]
        mask = self.local_block_valid.reshape(
            self.local_block_valid.shape + (1,) * (values.ndim - 1)
        )
        return jnp.where(mask, packed, jnp.zeros((), dtype=values.dtype))

    def unpack(self, packed_values: ArrayLike, /) -> Array:
        values = jnp.asarray(packed_values)
        if values.shape[:2] != (self.part_count, self.local_block_capacity):
            raise ValueError("Packed block values disagree with the level partition.")
        result = jnp.zeros((self.maximum_blocks,) + values.shape[2:], dtype=values.dtype)
        for part, identities in enumerate(self.slot_identities):
            if identities:
                slots = jnp.asarray(identities, dtype=jnp.int32)
                result = result.at[slots].set(values[part, : len(identities)])
        return result


class _PackedBlockRoutePlan(StrictModule, NonTrainableState):
    send_local_indices: Array
    receive_local_indices: Array
    send_valid: Array
    receive_valid: Array
    received_block_slots: Array
    received_block_valid: Array
    permutations: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    reverse_permutations: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    phase_count: int = eqx.field(static=True)
    message_capacity: int = eqx.field(static=True)
    receive_capacity: int = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        send_local_indices: np.ndarray,
        receive_local_indices: np.ndarray,
        send_valid: np.ndarray,
        receive_valid: np.ndarray,
        received_block_slots: np.ndarray,
        received_block_valid: np.ndarray,
        permutations: tuple[tuple[tuple[int, int], ...], ...],
        route_count: int,
        /,
        *,
        mesh: Mesh | None,
        axis_name: str,
    ):
        self.send_local_indices = _place_by_part(
            send_local_indices, mesh, axis_name, dtype=jnp.int32
        )
        self.receive_local_indices = _place_by_part(
            receive_local_indices, mesh, axis_name, dtype=jnp.int32
        )
        self.send_valid = _place_by_part(send_valid, mesh, axis_name, dtype=jnp.bool_)
        self.receive_valid = _place_by_part(
            receive_valid, mesh, axis_name, dtype=jnp.bool_
        )
        self.received_block_slots = _place_by_part(
            received_block_slots, mesh, axis_name, dtype=jnp.int32
        )
        self.received_block_valid = _place_by_part(
            received_block_valid, mesh, axis_name, dtype=jnp.bool_
        )
        self.permutations = permutations
        self.reverse_permutations = tuple(
            tuple((target, source) for source, target in phase) for phase in permutations
        )
        self.part_count = send_local_indices.shape[0]
        self.phase_count = send_local_indices.shape[1]
        self.message_capacity = send_local_indices.shape[2]
        self.receive_capacity = received_block_slots.shape[1]
        self.route_count = int(route_count)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "packed-block-route-plan",
                "send": array_tree_fingerprint(send_local_indices),
                "receive": array_tree_fingerprint(receive_local_indices),
                "send_valid": array_tree_fingerprint(send_valid),
                "receive_valid": array_tree_fingerprint(receive_valid),
                "received_slots": array_tree_fingerprint(received_block_slots),
                "permutations": permutations,
            }
        )

    @property
    def dynamic_arrays(self) -> tuple[Array, ...]:
        return (
            self.send_local_indices,
            self.receive_local_indices,
            self.send_valid,
            self.receive_valid,
            self.received_block_slots,
            self.received_block_valid,
        )

    def serial_exchange(self, source_values: ArrayLike, /) -> Array:
        source = jnp.asarray(source_values)
        if source.ndim < 2 or source.shape[0] != self.part_count:
            raise ValueError("Route source values must begin with the partition axis.")
        received = jnp.zeros(
            (self.part_count, self.receive_capacity) + source.shape[2:],
            dtype=source.dtype,
        )
        for phase, permutation in enumerate(self.permutations):
            for source_part, target_part in permutation:
                send_indices = self.send_local_indices[source_part, phase]
                receive_indices = self.receive_local_indices[target_part, phase]
                valid = self.receive_valid[target_part, phase]
                mask = valid.reshape(valid.shape + (1,) * (source.ndim - 2))
                payload = jnp.where(
                    mask,
                    source[source_part, send_indices],
                    jnp.zeros((), dtype=source.dtype),
                )
                received = received.at[target_part, receive_indices].add(payload)
        return received

    def serial_accumulate(
        self,
        owned_cotangent: ArrayLike,
        received_cotangent: ArrayLike,
        /,
    ) -> Array:
        owned = jnp.asarray(owned_cotangent)
        received = jnp.asarray(received_cotangent)
        if (
            owned.ndim < 2
            or owned.shape[0] != self.part_count
            or received.shape[:2] != (self.part_count, self.receive_capacity)
            or owned.shape[2:] != received.shape[2:]
        ):
            raise ValueError("Route reverse values disagree with packed capacities.")
        result = owned
        for phase, permutation in reversed(tuple(enumerate(self.permutations))):
            for source_part, target_part in permutation:
                send_indices = self.send_local_indices[source_part, phase]
                receive_indices = self.receive_local_indices[target_part, phase]
                valid = self.receive_valid[target_part, phase]
                mask = valid.reshape(valid.shape + (1,) * (received.ndim - 2))
                payload = jnp.where(
                    mask,
                    received[target_part, receive_indices],
                    jnp.zeros((), dtype=received.dtype),
                )
                result = result.at[source_part, send_indices].add(payload)
        return result


def _level_layout(
    topology: BlockHierarchyTopology,
    level: int,
    part_count: int,
    costs: ArrayLike | None,
    /,
    *,
    mesh: Mesh | None,
    axis_name: str,
) -> _BlockAMRLevelLayout:
    metadata = topology.levels[level]
    level_plan = topology.plan.levels[level]
    active = np.asarray(metadata.active, dtype=np.bool_)
    active_slots = np.flatnonzero(active)
    logical = np.asarray(metadata.logical_indices, dtype=np.int32)
    stable_ids = np.asarray(metadata.block_ids, dtype=np.int32)
    weights = _active_costs(costs, active_slots.size, level_plan.maximum_blocks)
    locality_order = sorted(
        range(active_slots.size),
        key=lambda index: (
            _morton_key(
                logical[active_slots[index]], topology.plan.block_lattice_shapes[level]
            ),
            int(stable_ids[active_slots[index]]),
        ),
    )
    ordered_slots = active_slots[np.asarray(locality_order, dtype=np.int32)]
    ordered_costs = weights[np.asarray(locality_order, dtype=np.int32)]
    ordered_owners = _contiguous_owners(ordered_costs, part_count)
    owner = np.full((level_plan.maximum_blocks,), -1, dtype=np.int32)
    owner[ordered_slots] = ordered_owners
    per_part = [
        sorted(
            np.flatnonzero(owner == part).tolist(),
            key=lambda slot: int(stable_ids[slot]),
        )
        for part in range(part_count)
    ]
    local_capacity = max(1, max((len(slots) for slots in per_part), default=0))
    local_slots = np.full((part_count, local_capacity), -1, dtype=np.int32)
    local_valid = np.zeros((part_count, local_capacity), dtype=np.bool_)
    canonical_to_local = np.full((level_plan.maximum_blocks,), -1, dtype=np.int32)
    for part, slots in enumerate(per_part):
        local_slots[part, : len(slots)] = slots
        local_valid[part, : len(slots)] = True
        for local, slot in enumerate(slots):
            canonical_to_local[slot] = local
    return _BlockAMRLevelLayout(
        owner,
        local_slots,
        local_valid,
        canonical_to_local,
        stable_ids,
        mesh=mesh,
        axis_name=axis_name,
    )


def _route_plan(
    source_layout: _BlockAMRLevelLayout,
    requirements: Sequence[set[int]],
    /,
    *,
    mesh: Mesh | None,
    axis_name: str,
) -> tuple[_PackedBlockRoutePlan, tuple[dict[int, int], ...]]:
    parts = source_layout.part_count
    if len(requirements) != parts:
        raise ValueError("Route requirements must provide one set per partition.")
    owner = np.asarray(source_layout.block_owner, dtype=np.int32)
    local = np.asarray(source_layout.canonical_to_local, dtype=np.int32)
    stable_ids = np.asarray(source_layout.stable_block_ids, dtype=np.int32)
    received_by_part: list[list[int]] = []
    lookups: list[dict[int, int]] = []
    pair_messages: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for target in range(parts):
        remote = sorted(
            {
                int(slot)
                for slot in requirements[target]
                if int(slot) >= 0 and int(owner[int(slot)]) != target
            },
            key=lambda slot: int(stable_ids[slot]),
        )
        received_by_part.append(remote)
        lookup = {slot: index for index, slot in enumerate(remote)}
        lookups.append(lookup)
        for slot in remote:
            source = int(owner[slot])
            if source < 0:
                raise ValueError(
                    "A distributed route references an inactive source block."
                )
            pair_messages.setdefault((source, target), []).append(
                (int(local[slot]), lookup[slot], int(stable_ids[slot]))
            )
    for messages in pair_messages.values():
        messages.sort(key=lambda value: value[2])
    permutations = _color_directed_pairs(pair_messages, include_empty_phase=True)
    message_capacity = max(
        1, max((len(messages) for messages in pair_messages.values()), default=0)
    )
    phase_count = len(permutations)
    send = np.zeros((parts, phase_count, message_capacity), dtype=np.int32)
    receive = np.zeros_like(send)
    send_valid = np.zeros_like(send, dtype=np.bool_)
    receive_valid = np.zeros_like(send, dtype=np.bool_)
    for phase, permutation in enumerate(permutations):
        for source, target in permutation:
            messages = pair_messages[(source, target)]
            count = len(messages)
            send[source, phase, :count] = [message[0] for message in messages]
            receive[target, phase, :count] = [message[1] for message in messages]
            send_valid[source, phase, :count] = True
            receive_valid[target, phase, :count] = True
    receive_capacity = max(1, max((len(slots) for slots in received_by_part), default=0))
    received_slots = np.full((parts, receive_capacity), -1, dtype=np.int32)
    received_valid = np.zeros((parts, receive_capacity), dtype=np.bool_)
    for part, slots in enumerate(received_by_part):
        received_slots[part, : len(slots)] = slots
        received_valid[part, : len(slots)] = True
    return (
        _PackedBlockRoutePlan(
            send,
            receive,
            send_valid,
            receive_valid,
            received_slots,
            received_valid,
            permutations,
            sum(len(slots) for slots in received_by_part),
            mesh=mesh,
            axis_name=axis_name,
        ),
        tuple(lookups),
    )


class _DistributedFillPatchLevel(StrictModule, NonTrainableState):
    source_class: Array
    same_local_indices: Array
    same_remote_indices: Array
    same_remote: Array
    same_cell_indices: Array
    coarse_local_indices: Array
    coarse_remote_indices: Array
    coarse_remote: Array
    coarse_cell_indices: Array
    coarse_donor_valid: Array
    coarse_child_indices: Array
    physical_boundary_mask: Array
    target_valid: Array
    level: int = eqx.field(static=True)
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: int,
        padded_shape: tuple[int, ...],
        arrays: tuple[np.ndarray, ...],
        /,
        *,
        mesh: Mesh | None,
        axis_name: str,
    ):
        placed = tuple(_place_by_part(value, mesh, axis_name) for value in arrays)
        (
            self.source_class,
            self.same_local_indices,
            self.same_remote_indices,
            self.same_remote,
            self.same_cell_indices,
            self.coarse_local_indices,
            self.coarse_remote_indices,
            self.coarse_remote,
            self.coarse_donor_valid,
            self.coarse_cell_indices,
            self.coarse_child_indices,
            self.physical_boundary_mask,
            self.target_valid,
        ) = placed
        self.level = int(level)
        self.padded_shape = padded_shape
        self.route_id = canonical_fingerprint(
            {
                "kind": "distributed-fill-patch-level-routes",
                "level": int(level),
                "arrays": array_tree_fingerprint(arrays),
            }
        )

    @property
    def dynamic_arrays(self) -> tuple[Array, ...]:
        return (
            self.source_class,
            self.same_local_indices,
            self.same_remote_indices,
            self.same_remote,
            self.same_cell_indices,
            self.coarse_local_indices,
            self.coarse_remote_indices,
            self.coarse_remote,
            self.coarse_donor_valid,
            self.coarse_cell_indices,
            self.coarse_child_indices,
            self.physical_boundary_mask,
            self.target_valid,
        )


def _route_requirements(
    fill: FDAMRFillPatchPlan,
    target_layout: _BlockAMRLevelLayout,
    source_slots: np.ndarray,
    valid: np.ndarray,
    /,
) -> list[set[int]]:
    requirements = [set() for _ in range(target_layout.part_count)]
    target_identities = target_layout.slot_identities
    for part, slots in enumerate(target_identities):
        for target_slot in slots:
            selected = source_slots[target_slot][valid[target_slot]]
            requirements[part].update(
                int(value) for value in np.asarray(selected).reshape(-1)
            )
    return requirements


def _distributed_fill_routes(
    fill: FDAMRFillPatchPlan,
    target_layout: _BlockAMRLevelLayout,
    same_lookup: tuple[dict[int, int], ...],
    coarse_layout: _BlockAMRLevelLayout | None,
    coarse_lookup: tuple[dict[int, int], ...] | None,
    /,
    *,
    mesh: Mesh | None,
    axis_name: str,
) -> _DistributedFillPatchLevel:
    source_class_global = np.asarray(fill.source_class, dtype=np.int8)
    source_slots_global = np.asarray(fill.source_slots, dtype=np.int32)
    source_local_global = np.asarray(fill.source_local_indices, dtype=np.int32)
    coarse_slots_global = np.asarray(fill.coarse_donor_slots, dtype=np.int32)
    coarse_valid_global = np.asarray(fill.coarse_donor_valid, dtype=np.bool_)
    coarse_child_global = np.asarray(fill.coarse_child_indices, dtype=np.int32)
    coarse_local_global = np.asarray(fill.coarse_donor_local_indices, dtype=np.int32)
    physical_global = np.asarray(fill.physical_boundary_mask, dtype=np.bool_)
    parts = target_layout.part_count
    local_capacity = target_layout.local_block_capacity
    padded_shape = tuple(source_class_global.shape[1:])
    donor_count = coarse_slots_global.shape[-1]
    dimension = coarse_child_global.shape[-1]
    route_shape = (parts, local_capacity) + padded_shape
    source_class = np.full(route_shape, int(FillPatchSource.INACTIVE), dtype=np.int8)
    same_local = np.zeros(route_shape, dtype=np.int32)
    same_remote_indices = np.zeros(route_shape, dtype=np.int32)
    same_remote = np.zeros(route_shape, dtype=np.bool_)
    same_cell = np.zeros(route_shape + (dimension,), dtype=np.int32)
    coarse_shape = route_shape + (donor_count,)
    coarse_local = np.zeros(coarse_shape, dtype=np.int32)
    coarse_remote_indices = np.zeros(coarse_shape, dtype=np.int32)
    coarse_remote = np.zeros(coarse_shape, dtype=np.bool_)
    coarse_valid = np.zeros(coarse_shape, dtype=np.bool_)
    coarse_cell = np.zeros(coarse_shape + (dimension,), dtype=np.int32)
    coarse_child = np.zeros(route_shape + (dimension,), dtype=np.int32)
    physical = np.zeros(route_shape, dtype=np.bool_)
    target_valid = np.zeros((parts, local_capacity), dtype=np.bool_)
    same_owner = np.asarray(target_layout.block_owner, dtype=np.int32)
    same_canonical_to_local = np.asarray(target_layout.canonical_to_local, dtype=np.int32)
    coarse_owner = (
        None
        if coarse_layout is None
        else np.asarray(coarse_layout.block_owner, dtype=np.int32)
    )
    coarse_canonical_to_local = (
        None
        if coarse_layout is None
        else np.asarray(coarse_layout.canonical_to_local, dtype=np.int32)
    )
    for part, slots in enumerate(target_layout.slot_identities):
        for target_local, target_slot in enumerate(slots):
            target_valid[part, target_local] = True
            destination = (part, target_local)
            source_class[destination] = source_class_global[target_slot]
            physical[destination] = physical_global[target_slot]
            coarse_child[destination] = np.maximum(coarse_child_global[target_slot], 0)
            same_mask = (
                (source_class_global[target_slot] == int(FillPatchSource.INTERIOR))
                | (source_class_global[target_slot] == int(FillPatchSource.SAME_LEVEL))
                | (source_class_global[target_slot] == int(FillPatchSource.PERIODIC))
            )
            for padded_index in np.argwhere(same_mask):
                index = destination + tuple(padded_index)
                source_slot = int(
                    source_slots_global[(target_slot,) + tuple(padded_index)]
                )
                owner = int(same_owner[source_slot])
                if owner == part:
                    same_local[index] = int(same_canonical_to_local[source_slot])
                else:
                    same_remote[index] = True
                    same_remote_indices[index] = same_lookup[part][source_slot]
                same_cell[index] = np.maximum(
                    source_local_global[(target_slot,) + tuple(padded_index)], 0
                )
            if coarse_layout is None or coarse_lookup is None:
                continue
            for route_index in np.argwhere(coarse_valid_global[target_slot]):
                padded_index = tuple(route_index[:-1])
                donor = int(route_index[-1])
                index = destination + padded_index + (donor,)
                source_slot = int(
                    coarse_slots_global[(target_slot,) + padded_index + (donor,)]
                )
                coarse_valid[index] = True
                owner = int(coarse_owner[source_slot])
                if owner == part:
                    coarse_local[index] = int(coarse_canonical_to_local[source_slot])
                else:
                    coarse_remote[index] = True
                    coarse_remote_indices[index] = coarse_lookup[part][source_slot]
                coarse_cell[index] = np.maximum(
                    coarse_local_global[(target_slot,) + padded_index + (donor,)],
                    0,
                )
    return _DistributedFillPatchLevel(
        fill.level,
        padded_shape,
        (
            source_class,
            same_local,
            same_remote_indices,
            same_remote,
            same_cell,
            coarse_local,
            coarse_remote_indices,
            coarse_remote,
            coarse_valid,
            coarse_cell,
            coarse_child,
            physical,
            target_valid,
        ),
        mesh=mesh,
        axis_name=axis_name,
    )


def _execute_fill_local(
    current: Array,
    same_ghost: Array,
    coarse_old: Array,
    coarse_new: Array,
    coarse_old_ghost: Array,
    coarse_new_ghost: Array,
    boundary: Array,
    source_class: Array,
    same_local_indices: Array,
    same_remote_indices: Array,
    same_remote: Array,
    same_cell_indices: Array,
    coarse_local_indices: Array,
    coarse_remote_indices: Array,
    coarse_remote: Array,
    coarse_donor_valid: Array,
    coarse_cell_indices: Array,
    coarse_child_indices: Array,
    physical_boundary_mask: Array,
    target_valid: Array,
    transfer: Any,
    coarse_old_time: ArrayLike,
    coarse_new_time: ArrayLike,
    fill_time: ArrayLike,
    boundary_supplied: bool,
    /,
) -> tuple[Array, Array]:
    same_cells = jnp.maximum(same_cell_indices, 0)
    same_local_index = (jnp.maximum(same_local_indices, 0),) + tuple(
        same_cells[..., axis] for axis in range(same_cells.shape[-1])
    )
    same_remote_index = (jnp.maximum(same_remote_indices, 0),) + tuple(
        same_cells[..., axis] for axis in range(same_cells.shape[-1])
    )
    same_local_values = current[same_local_index]
    same_remote_values = same_ghost[same_remote_index]
    component_rank = same_local_values.ndim - source_class.ndim
    same_values = jnp.where(
        same_remote.reshape(same_remote.shape + (1,) * component_rank),
        same_remote_values,
        same_local_values,
    )
    same_mask = (
        (source_class == int(FillPatchSource.INTERIOR))
        | (source_class == int(FillPatchSource.SAME_LEVEL))
        | (source_class == int(FillPatchSource.PERIODIC))
    )
    values = jnp.where(
        same_mask.reshape(same_mask.shape + (1,) * component_rank),
        same_values,
        jnp.zeros((), dtype=current.dtype),
    )
    valid = same_mask
    if transfer is not None:
        coarse_cells = jnp.maximum(coarse_cell_indices, 0)
        coarse_local_index = (jnp.maximum(coarse_local_indices, 0),) + tuple(
            coarse_cells[..., axis] for axis in range(coarse_cells.shape[-1])
        )
        coarse_remote_index = (jnp.maximum(coarse_remote_indices, 0),) + tuple(
            coarse_cells[..., axis] for axis in range(coarse_cells.shape[-1])
        )
        old_local_values = coarse_old[coarse_local_index]
        new_local_values = coarse_new[coarse_local_index]
        old_remote_values = coarse_old_ghost[coarse_remote_index]
        new_remote_values = coarse_new_ghost[coarse_remote_index]
        donor_component_rank = old_local_values.ndim - coarse_donor_valid.ndim
        remote_mask = coarse_remote.reshape(
            coarse_remote.shape + (1,) * donor_component_rank
        )
        old_donors = jnp.where(remote_mask, old_remote_values, old_local_values)
        new_donors = jnp.where(remote_mask, new_remote_values, new_local_values)
        donor_valid = coarse_donor_valid.reshape(
            coarse_donor_valid.shape + (1,) * donor_component_rank
        )
        old_donors = jnp.where(donor_valid, old_donors, 0)
        new_donors = jnp.where(donor_valid, new_donors, 0)
        dtype = jnp.result_type(coarse_old_time, coarse_new_time, fill_time, 1.0)
        old_time = jnp.asarray(coarse_old_time, dtype=dtype)
        new_time = jnp.asarray(coarse_new_time, dtype=dtype)
        target_time = jnp.asarray(fill_time, dtype=dtype)
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.abs(old_time),
                jnp.maximum(jnp.abs(new_time), jnp.abs(target_time)),
            ),
        )
        tolerance = 16.0 * jnp.finfo(dtype).eps * scale
        span = new_time - old_time
        same_time = jnp.abs(span) <= tolerance
        denominator = jnp.where(same_time, 1.0, span)
        alpha = jnp.where(same_time, 0.0, (target_time - old_time) / denominator)
        invalid_time = (
            ~jnp.isfinite(old_time)
            | ~jnp.isfinite(new_time)
            | ~jnp.isfinite(target_time)
            | (span < -tolerance)
            | (target_time < old_time - tolerance)
            | (target_time > new_time + tolerance)
            | (same_time & (jnp.abs(target_time - old_time) > tolerance))
        )
        alpha = eqx.error_if(
            alpha,
            invalid_time,
            "FillPatch time must lie within the coarse old/new interval.",
        )
        alpha = jnp.clip(alpha, 0.0, 1.0)
        donors = old_donors + alpha * (new_donors - old_donors)
        dimension = coarse_child_indices.shape[-1]
        component_shape = donors.shape[coarse_donor_valid.ndim :]
        donor_patches = donors.reshape(
            source_class.shape + (3,) * dimension + component_shape
        )
        route_count = prod(source_class.shape)
        flat_patches = donor_patches.reshape(
            (route_count,) + (3,) * dimension + component_shape
        )
        prolonged = jax.vmap(transfer.prolong)(flat_patches)
        child = coarse_child_indices.reshape((route_count, dimension))
        ratio = transfer.refinement_ratio
        fine_index = (jnp.arange(route_count, dtype=jnp.int32),) + tuple(
            ratio + child[:, axis] for axis in range(dimension)
        )
        coarse_values = prolonged[fine_index].reshape(
            source_class.shape + component_shape
        )
        coarse_mask = source_class == int(FillPatchSource.COARSE_TIME_INTERPOLATED)
        values = jnp.where(
            coarse_mask.reshape(coarse_mask.shape + (1,) * component_rank),
            coarse_values,
            values,
        )
        valid = valid | coarse_mask
    if boundary_supplied:
        values = jnp.where(
            physical_boundary_mask.reshape(
                physical_boundary_mask.shape + (1,) * component_rank
            ),
            boundary,
            values,
        )
        valid = valid | physical_boundary_mask
    active = target_valid.reshape(target_valid.shape + (1,) * (valid.ndim - 1))
    valid = valid & active
    values = jnp.where(
        valid.reshape(valid.shape + (1,) * component_rank),
        values,
        jnp.zeros((), dtype=values.dtype),
    )
    return values, valid


class DistributedBlockAMRResourceEvidence(StrictModule, NonTrainableState):
    active_blocks: tuple[int, ...] = eqx.field(static=True)
    local_block_capacities: tuple[int, ...] = eqx.field(static=True)
    allocated_block_slots: tuple[int, ...] = eqx.field(static=True)
    same_level_routes: tuple[int, ...] = eqx.field(static=True)
    coarse_fine_routes: tuple[int, ...] = eqx.field(static=True)
    interface_routes: tuple[int, ...] = eqx.field(static=True)
    same_level_phases: tuple[int, ...] = eqx.field(static=True)
    coarse_fine_phases: tuple[int, ...] = eqx.field(static=True)
    interface_phases: tuple[int, ...] = eqx.field(static=True)
    dynamic_route_array_entries: int = eqx.field(static=True)
    dynamic_route_array_bytes: int = eqx.field(static=True)
    static_permutation_pairs: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        layouts: Sequence[_BlockAMRLevelLayout],
        same_level: Sequence[_PackedBlockRoutePlan],
        coarse_fine: Sequence[_PackedBlockRoutePlan],
        interface: Sequence[_PackedBlockRoutePlan],
        fill_routes: Sequence[_DistributedFillPatchLevel],
        /,
    ):
        layouts_ = tuple(layouts)
        same = tuple(same_level)
        coarse = tuple(coarse_fine)
        interfaces = tuple(interface)
        fill = tuple(fill_routes)
        dynamic_arrays = tuple(
            array
            for route in (*same, *coarse, *interfaces)
            for array in route.dynamic_arrays
        ) + tuple(array for route in fill for array in route.dynamic_arrays)
        self.active_blocks = tuple(layout.active_count for layout in layouts_)
        self.local_block_capacities = tuple(
            layout.local_block_capacity for layout in layouts_
        )
        self.allocated_block_slots = tuple(
            layout.part_count * layout.local_block_capacity for layout in layouts_
        )
        self.same_level_routes = tuple(route.route_count for route in same)
        self.coarse_fine_routes = tuple(route.route_count for route in coarse)
        self.interface_routes = tuple(route.route_count for route in interfaces)
        self.same_level_phases = tuple(route.phase_count for route in same)
        self.coarse_fine_phases = tuple(route.phase_count for route in coarse)
        self.interface_phases = tuple(route.phase_count for route in interfaces)
        self.dynamic_route_array_entries = sum(array.size for array in dynamic_arrays)
        self.dynamic_route_array_bytes = sum(
            array.size * array.dtype.itemsize for array in dynamic_arrays
        )
        self.static_permutation_pairs = sum(
            len(phase)
            for route in (*same, *coarse, *interfaces)
            for phase in route.permutations
        )
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "distributed-block-amr-resource-evidence",
                "active_blocks": self.active_blocks,
                "local_capacities": self.local_block_capacities,
                "allocated_slots": self.allocated_block_slots,
                "same_routes": self.same_level_routes,
                "coarse_routes": self.coarse_fine_routes,
                "interface_routes": self.interface_routes,
                "same_phases": self.same_level_phases,
                "coarse_phases": self.coarse_fine_phases,
                "interface_phases": self.interface_phases,
                "dynamic_entries": self.dynamic_route_array_entries,
                "dynamic_bytes": self.dynamic_route_array_bytes,
                "static_permutation_pairs": self.static_permutation_pairs,
            }
        )

    @property
    def resource_counts(self) -> tuple[tuple[str, int], ...]:
        return (
            ("active_blocks", sum(self.active_blocks)),
            ("allocated_block_slots", sum(self.allocated_block_slots)),
            ("same_level_routes", sum(self.same_level_routes)),
            ("coarse_fine_routes", sum(self.coarse_fine_routes)),
            ("interface_routes", sum(self.interface_routes)),
            ("dynamic_route_array_entries", self.dynamic_route_array_entries),
            ("dynamic_route_array_bytes", self.dynamic_route_array_bytes),
            ("static_permutation_pairs", self.static_permutation_pairs),
        )


class BlockAMRPartitionPlan(StrictModule, NonTrainableState):
    """Host partition policy for one fixed-block hierarchy.

    Ownership is prepared from canonical logical indices, never prior ownership or
    storage history. Costs affect only deterministic contiguous cuts of a Morton
    locality order.
    """

    hierarchy: BlockHierarchyPlan
    part_count: int = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hierarchy: BlockHierarchyPlan,
        part_count: int,
        /,
        *,
        axis_name: str = "block_parts",
    ):
        parts = int(part_count)
        axis = str(axis_name).strip()
        if not isinstance(hierarchy, BlockHierarchyPlan):
            raise TypeError("Distributed block AMR requires BlockHierarchyPlan.")
        if parts <= 0 or not axis:
            raise ValueError("AMR partition count and mesh axis name must be valid.")
        self.hierarchy = hierarchy
        self.part_count = parts
        self.axis_name = axis
        self.plan_id = canonical_fingerprint(
            {
                "kind": "block-amr-partition-plan",
                "hierarchy": hierarchy.plan_id,
                "part_count": parts,
                "axis_name": axis,
                "locality": "morton-contiguous-weighted",
            }
        )

    def prepare(
        self,
        topology_result: BlockTopologyCompileResult | BlockHierarchyTopology,
        fd_hierarchy: PreparedFDAMRHierarchy,
        /,
        *,
        costs: Sequence[ArrayLike | None] | None = None,
        execution_group: ExecutionGroup | None = None,
    ) -> PreparedDistributedBlockAMRHierarchy:
        return PreparedDistributedBlockAMRHierarchy(
            self,
            topology_result,
            fd_hierarchy,
            costs=costs,
            execution_group=execution_group,
        )


class PreparedDistributedBlockAMRHierarchy(StrictModule, NonTrainableState):
    """Packed owner-computes fixed-block AMR and exact route transposes."""

    partition: BlockAMRPartitionPlan
    topology: BlockHierarchyTopology
    fd_hierarchy: PreparedFDAMRHierarchy
    layouts: tuple[_BlockAMRLevelLayout, ...]
    fill_patch_plans: tuple[FDAMRFillPatchPlan, ...]
    same_level_routes: tuple[_PackedBlockRoutePlan, ...]
    coarse_fine_routes: tuple[_PackedBlockRoutePlan, ...]
    interface_routes: tuple[_PackedBlockRoutePlan, ...]
    fill_routes: tuple[_DistributedFillPatchLevel, ...]
    resources: DistributedBlockAMRResourceEvidence
    mesh: Mesh | None = eqx.field(static=True)
    compilation_result_id: str | None = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        partition: BlockAMRPartitionPlan,
        topology_result: BlockTopologyCompileResult | BlockHierarchyTopology,
        fd_hierarchy: PreparedFDAMRHierarchy,
        /,
        *,
        costs: Sequence[ArrayLike | None] | None = None,
        execution_group: ExecutionGroup | None = None,
    ):
        if not isinstance(partition, BlockAMRPartitionPlan):
            raise TypeError("partition must be BlockAMRPartitionPlan.")
        if isinstance(topology_result, BlockTopologyCompileResult):
            if not topology_result.status.successful:
                raise ValueError("Cannot distribute an unsuccessful topology result.")
            topology = topology_result.topology
            compilation_result_id = topology_result.result_id
        elif isinstance(topology_result, BlockHierarchyTopology):
            topology = topology_result
            compilation_result_id = None
        else:
            raise TypeError(
                "Distributed preparation requires BlockTopologyCompileResult or BlockHierarchyTopology."
            )
        if topology.plan.plan_id != partition.hierarchy.plan_id:
            raise ValueError("Partition policy and realized topology hierarchy differ.")
        if (
            not isinstance(fd_hierarchy, PreparedFDAMRHierarchy)
            or fd_hierarchy.plan.hierarchy.plan_id != partition.hierarchy.plan_id
        ):
            raise ValueError("Prepared FD AMR hierarchy does not match partition policy.")
        cost_values = (
            (None,) * len(topology.plan.levels) if costs is None else tuple(costs)
        )
        if len(cost_values) != len(topology.plan.levels):
            raise ValueError("Block costs require one optional vector per AMR level.")
        if execution_group is not None and not isinstance(
            execution_group, ExecutionGroup
        ):
            raise TypeError("execution_group must be an ExecutionGroup or None.")
        if (
            execution_group is not None
            and len(execution_group.devices) != partition.part_count
        ):
            raise ValueError(
                "Distributed AMR requires one execution-group device per part."
            )
        mesh = (
            None
            if execution_group is None
            else Mesh(
                np.asarray(execution_group.devices, dtype=object),
                (partition.axis_name,),
            )
        )
        layouts = tuple(
            _level_layout(
                topology,
                level,
                partition.part_count,
                cost_values[level],
                mesh=mesh,
                axis_name=partition.axis_name,
            )
            for level in range(len(topology.plan.levels))
        )
        fill_plans = fd_hierarchy.prepare_fill_patch(topology)
        same_routes: list[_PackedBlockRoutePlan] = []
        coarse_routes: list[_PackedBlockRoutePlan] = []
        interface_routes: list[_PackedBlockRoutePlan] = []
        same_lookups: list[tuple[dict[int, int], ...]] = []
        coarse_lookups: list[tuple[dict[int, int], ...]] = []
        for level, (fill, layout) in enumerate(zip(fill_plans, layouts, strict=True)):
            source_class = np.asarray(fill.source_class, dtype=np.int8)
            source_slots = np.asarray(fill.source_slots, dtype=np.int32)
            same_valid = (
                (source_class == int(FillPatchSource.INTERIOR))
                | (source_class == int(FillPatchSource.SAME_LEVEL))
                | (source_class == int(FillPatchSource.PERIODIC))
            )
            same_requirements = _route_requirements(
                fill, layout, source_slots, same_valid
            )
            same_route, same_lookup = _route_plan(
                layout,
                same_requirements,
                mesh=mesh,
                axis_name=partition.axis_name,
            )
            same_routes.append(same_route)
            same_lookups.append(same_lookup)
            if level == 0:
                coarse_requirements = [set() for _ in range(partition.part_count)]
                coarse_source_layout = layout
            else:
                donor_slots = np.asarray(fill.coarse_donor_slots, dtype=np.int32)
                donor_valid = np.asarray(fill.coarse_donor_valid, dtype=np.bool_)
                coarse_requirements = _route_requirements(
                    fill, layout, donor_slots, donor_valid
                )
                coarse_source_layout = layouts[level - 1]
            coarse_route, coarse_lookup = _route_plan(
                coarse_source_layout,
                coarse_requirements,
                mesh=mesh,
                axis_name=partition.axis_name,
            )
            coarse_routes.append(coarse_route)
            coarse_lookups.append(coarse_lookup)
            interface_requirements = [set() for _ in range(partition.part_count)]
            interface_source_layout = layout
            if level > 0:
                fine_interfaces = np.asarray(topology.interfaces[level], dtype=np.bool_)
                fine_parents = np.asarray(
                    topology.levels[level].parent_ids, dtype=np.int32
                )
                coarse_ids = np.asarray(
                    topology.levels[level - 1].block_ids, dtype=np.int32
                )
                coarse_by_id = {
                    int(value): slot
                    for slot, value in enumerate(coarse_ids)
                    if value >= 0
                }
                coarse_owner = np.asarray(layouts[level - 1].block_owner, dtype=np.int32)
                for fine_slot in np.flatnonzero(np.any(fine_interfaces, axis=(1, 2))):
                    parent_slot = coarse_by_id[int(fine_parents[fine_slot])]
                    target_part = int(coarse_owner[parent_slot])
                    interface_requirements[target_part].add(int(fine_slot))
            interface_route, _ = _route_plan(
                interface_source_layout,
                interface_requirements,
                mesh=mesh,
                axis_name=partition.axis_name,
            )
            interface_routes.append(interface_route)
        fill_routes = tuple(
            _distributed_fill_routes(
                fill,
                layouts[level],
                same_lookups[level],
                None if level == 0 else layouts[level - 1],
                None if level == 0 else coarse_lookups[level],
                mesh=mesh,
                axis_name=partition.axis_name,
            )
            for level, fill in enumerate(fill_plans)
        )
        resources = DistributedBlockAMRResourceEvidence(
            layouts, same_routes, coarse_routes, interface_routes, fill_routes
        )
        self.partition = partition
        self.topology = topology
        self.fd_hierarchy = fd_hierarchy
        self.layouts = layouts
        self.fill_patch_plans = fill_plans
        self.same_level_routes = tuple(same_routes)
        self.coarse_fine_routes = tuple(coarse_routes)
        self.interface_routes = tuple(interface_routes)
        self.fill_routes = fill_routes
        self.resources = resources
        self.mesh = mesh
        self.compilation_result_id = compilation_result_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-distributed-block-amr-hierarchy",
                "partition": partition.plan_id,
                "epoch": topology.epoch.epoch_id,
                "fd_hierarchy": fd_hierarchy.prepared_id,
                "layouts": [layout.layout_id for layout in layouts],
                "same_routes": [route.plan_id for route in same_routes],
                "coarse_routes": [route.plan_id for route in coarse_routes],
                "interface_routes": [route.plan_id for route in interface_routes],
                "fill_routes": [route.route_id for route in fill_routes],
                "resources": resources.evidence_id,
            }
        )

    @property
    def resource_evidence_id(self) -> str:
        return self.resources.evidence_id

    @property
    def local_block_capacities(self) -> tuple[int, ...]:
        return tuple(layout.local_block_capacity for layout in self.layouts)

    def pack(self, state: BlockHierarchyState, /) -> tuple[Array, ...]:
        self._validate_state(state)
        return tuple(
            layout.pack(level.values)
            for layout, level in zip(self.layouts, state.levels, strict=True)
        )

    def unpack(self, packed_values: Sequence[ArrayLike], /) -> BlockHierarchyState:
        values = tuple(packed_values)
        if len(values) != len(self.layouts):
            raise ValueError("Packed hierarchy requires one value array per level.")
        levels = tuple(
            BlockLevelState(
                level_plan,
                metadata,
                layout.unpack(value),
            )
            for level_plan, metadata, layout, value in zip(
                self.topology.plan.levels,
                self.topology.levels,
                self.layouts,
                values,
                strict=True,
            )
        )
        return BlockHierarchyState(self.topology, levels)

    def _validate_state(self, state: BlockHierarchyState, /) -> None:
        if (
            not isinstance(state, BlockHierarchyState)
            or state.topology.epoch.epoch_id != self.topology.epoch.epoch_id
        ):
            raise ValueError(
                "Distributed AMR state must share the prepared topology epoch."
            )
        expected_dtype = jnp.dtype(self.fd_hierarchy.plan.precision.field_dtype)
        for level in state.levels:
            if level.values.dtype != expected_dtype:
                raise TypeError(
                    "Distributed FillPatch state dtype must match FD field precision."
                )

    def _routes(self, kind: RouteKind, level: int, /) -> _PackedBlockRoutePlan:
        level_ = int(level)
        if level_ < 0 or level_ >= len(self.layouts):
            raise ValueError("Distributed AMR route level is out of range.")
        if kind == "same_level":
            return self.same_level_routes[level_]
        if kind == "coarse_fine":
            return self.coarse_fine_routes[level_]
        if kind == "interface":
            return self.interface_routes[level_]
        raise ValueError("Unknown distributed AMR route kind.")

    def route_exchange(
        self,
        kind: RouteKind,
        level: int,
        source_values: ArrayLike,
        /,
        *,
        distributed: bool = False,
    ) -> Array:
        route = self._routes(kind, level)
        source = jnp.asarray(source_values)
        if distributed:
            return self._distributed_exchange(route, source)
        return route.serial_exchange(source)

    def route_reverse(
        self,
        kind: RouteKind,
        level: int,
        owned_cotangent: ArrayLike,
        received_cotangent: ArrayLike,
        /,
        *,
        distributed: bool = False,
    ) -> Array:
        route = self._routes(kind, level)
        owned = jnp.asarray(owned_cotangent)
        received = jnp.asarray(received_cotangent)
        if distributed:
            return self._distributed_accumulate(route, owned, received)
        return route.serial_accumulate(owned, received)

    def _distributed_exchange(
        self, route: _PackedBlockRoutePlan, source_values: Array, /
    ) -> Array:
        if self.mesh is None:
            raise RuntimeError(
                "Distributed execution requires preparation with an ExecutionGroup."
            )
        axis = self.partition.axis_name
        source = jax.device_put(
            source_values,
            NamedSharding(self.mesh, _part_spec(axis, source_values.ndim)),
        )
        route_arrays = tuple(
            jax.device_put(array, NamedSharding(self.mesh, _part_spec(axis, array.ndim)))
            for array in (
                route.send_local_indices,
                route.receive_local_indices,
                route.send_valid,
                route.receive_valid,
            )
        )

        def exchange_local(local, send, receive, send_valid, receive_valid):
            current = local[0]
            ghost = jnp.zeros(
                (route.receive_capacity,) + current.shape[1:], dtype=current.dtype
            )
            for phase, permutation in enumerate(route.permutations):
                indices = send[0, phase]
                valid = send_valid[0, phase]
                payload = current[indices]
                payload = jnp.where(
                    valid.reshape(valid.shape + (1,) * (payload.ndim - 1)),
                    payload,
                    0,
                )
                incoming = jax.lax.ppermute(payload, axis_name=axis, perm=permutation)
                target_indices = receive[0, phase]
                target_valid = receive_valid[0, phase]
                incoming = jnp.where(
                    target_valid.reshape(target_valid.shape + (1,) * (incoming.ndim - 1)),
                    incoming,
                    0,
                )
                ghost = ghost.at[target_indices].add(incoming)
            return ghost[None]

        source_spec = _part_spec(axis, source.ndim)
        route_specs = tuple(_part_spec(axis, array.ndim) for array in route_arrays)
        output_spec = _part_spec(axis, source.ndim)
        mapped = jax.shard_map(
            exchange_local,
            mesh=self.mesh,
            in_specs=(source_spec, *route_specs),
            out_specs=output_spec,
            check_vma=False,
        )
        return mapped(source, *route_arrays)

    def _distributed_accumulate(
        self,
        route: _PackedBlockRoutePlan,
        owned_cotangent: Array,
        received_cotangent: Array,
        /,
    ) -> Array:
        if self.mesh is None:
            raise RuntimeError(
                "Distributed execution requires preparation with an ExecutionGroup."
            )
        axis = self.partition.axis_name
        owned = jax.device_put(
            owned_cotangent,
            NamedSharding(self.mesh, _part_spec(axis, owned_cotangent.ndim)),
        )
        received = jax.device_put(
            received_cotangent,
            NamedSharding(self.mesh, _part_spec(axis, received_cotangent.ndim)),
        )
        route_arrays = tuple(
            jax.device_put(array, NamedSharding(self.mesh, _part_spec(axis, array.ndim)))
            for array in (
                route.send_local_indices,
                route.receive_local_indices,
                route.send_valid,
                route.receive_valid,
            )
        )

        def accumulate_local(
            local_owned, local_received, send, receive, send_valid, receive_valid
        ):
            result = local_owned[0]
            ghost = local_received[0]
            for phase, permutation in reversed(
                tuple(enumerate(route.reverse_permutations))
            ):
                indices = receive[0, phase]
                valid = receive_valid[0, phase]
                payload = ghost[indices]
                payload = jnp.where(
                    valid.reshape(valid.shape + (1,) * (payload.ndim - 1)),
                    payload,
                    0,
                )
                incoming = jax.lax.ppermute(payload, axis_name=axis, perm=permutation)
                target_indices = send[0, phase]
                target_valid = send_valid[0, phase]
                incoming = jnp.where(
                    target_valid.reshape(target_valid.shape + (1,) * (incoming.ndim - 1)),
                    incoming,
                    0,
                )
                result = result.at[target_indices].add(incoming)
            return result[None]

        owned_spec = _part_spec(axis, owned.ndim)
        received_spec = _part_spec(axis, received.ndim)
        route_specs = tuple(_part_spec(axis, array.ndim) for array in route_arrays)
        mapped = jax.shard_map(
            accumulate_local,
            mesh=self.mesh,
            in_specs=(owned_spec, received_spec, *route_specs),
            out_specs=owned_spec,
            check_vma=False,
        )
        return mapped(owned, received, *route_arrays)

    def _boundary_inputs(
        self,
        packed_state: tuple[Array, ...],
        physical_boundary_values: Sequence[ArrayLike | None] | None,
        /,
    ) -> tuple[tuple[Array, ...], tuple[bool, ...]]:
        boundaries = (
            (None,) * len(self.layouts)
            if physical_boundary_values is None
            else tuple(physical_boundary_values)
        )
        if len(boundaries) != len(self.layouts):
            raise ValueError("Physical boundary values require one entry per AMR level.")
        packed_boundaries: list[Array] = []
        supplied: list[bool] = []
        dimension = len(self.topology.plan.grid.shape)
        for level, (boundary, packed, layout, fill_route) in enumerate(
            zip(boundaries, packed_state, self.layouts, self.fill_routes, strict=True)
        ):
            component_shape = packed.shape[2 + dimension :]
            expected = (
                self.topology.plan.levels[level].maximum_blocks,
                *fill_route.padded_shape,
                *component_shape,
            )
            if boundary is None:
                packed_boundaries.append(
                    jnp.zeros(
                        (
                            self.partition.part_count,
                            layout.local_block_capacity,
                            *fill_route.padded_shape,
                            *component_shape,
                        ),
                        dtype=packed.dtype,
                    )
                )
                supplied.append(False)
            else:
                value = jnp.asarray(boundary, dtype=packed.dtype)
                if value.shape != expected:
                    raise ValueError(
                        f"Physical boundary level {level} must have shape {expected}."
                    )
                packed_boundaries.append(layout.pack(value))
                supplied.append(True)
        return tuple(packed_boundaries), tuple(supplied)

    def _packed_fill_values(
        self,
        packed_state: tuple[Array, ...],
        packed_old: tuple[Array, ...],
        packed_new: tuple[Array, ...],
        packed_boundaries: tuple[Array, ...],
        boundary_supplied: tuple[bool, ...],
        coarse_old_time: ArrayLike,
        coarse_new_time: ArrayLike,
        fill_time: ArrayLike,
        /,
        *,
        distributed: bool,
    ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        exchange = (
            self._distributed_exchange
            if distributed
            else lambda route, values: route.serial_exchange(values)
        )
        same_ghost = tuple(
            exchange(route, values)
            for route, values in zip(self.same_level_routes, packed_state, strict=True)
        )
        coarse_old_ghost: list[Array] = []
        coarse_new_ghost: list[Array] = []
        for level, route in enumerate(self.coarse_fine_routes):
            source_level = max(0, level - 1)
            coarse_old_ghost.append(exchange(route, packed_old[source_level]))
            coarse_new_ghost.append(exchange(route, packed_new[source_level]))
        if distributed:
            return self._distributed_fill_levels(
                packed_state,
                packed_old,
                packed_new,
                same_ghost,
                tuple(coarse_old_ghost),
                tuple(coarse_new_ghost),
                packed_boundaries,
                boundary_supplied,
                coarse_old_time,
                coarse_new_time,
                fill_time,
            )
        outputs: list[Array] = []
        validity: list[Array] = []
        for level, route in enumerate(self.fill_routes):
            level_outputs = []
            level_valid = []
            coarse_level = max(0, level - 1)
            transfer = self.fill_patch_plans[level].transfer
            for part in range(self.partition.part_count):
                result = _execute_fill_local(
                    packed_state[level][part],
                    same_ghost[level][part],
                    packed_old[coarse_level][part],
                    packed_new[coarse_level][part],
                    coarse_old_ghost[level][part],
                    coarse_new_ghost[level][part],
                    packed_boundaries[level][part],
                    route.source_class[part],
                    route.same_local_indices[part],
                    route.same_remote_indices[part],
                    route.same_remote[part],
                    route.same_cell_indices[part],
                    route.coarse_local_indices[part],
                    route.coarse_remote_indices[part],
                    route.coarse_remote[part],
                    route.coarse_donor_valid[part],
                    route.coarse_cell_indices[part],
                    route.coarse_child_indices[part],
                    route.physical_boundary_mask[part],
                    route.target_valid[part],
                    transfer,
                    coarse_old_time,
                    coarse_new_time,
                    fill_time,
                    boundary_supplied[level],
                )
                level_outputs.append(result[0])
                level_valid.append(result[1])
            outputs.append(jnp.stack(level_outputs))
            validity.append(jnp.stack(level_valid))
        return tuple(outputs), tuple(validity)

    def _distributed_fill_levels(
        self,
        packed_state: tuple[Array, ...],
        packed_old: tuple[Array, ...],
        packed_new: tuple[Array, ...],
        same_ghost: tuple[Array, ...],
        coarse_old_ghost: tuple[Array, ...],
        coarse_new_ghost: tuple[Array, ...],
        packed_boundaries: tuple[Array, ...],
        boundary_supplied: tuple[bool, ...],
        coarse_old_time: ArrayLike,
        coarse_new_time: ArrayLike,
        fill_time: ArrayLike,
        /,
    ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        if self.mesh is None:
            raise RuntimeError(
                "Distributed execution requires preparation with an ExecutionGroup."
            )
        axis = self.partition.axis_name
        outputs: list[Array] = []
        validity: list[Array] = []
        for level, route in enumerate(self.fill_routes):
            coarse_level = max(0, level - 1)
            transfer = self.fill_patch_plans[level].transfer
            boundary_is_supplied = boundary_supplied[level]
            data = (
                packed_state[level],
                same_ghost[level],
                packed_old[coarse_level],
                packed_new[coarse_level],
                coarse_old_ghost[level],
                coarse_new_ghost[level],
                packed_boundaries[level],
                *route.dynamic_arrays,
            )
            placed = tuple(
                jax.device_put(
                    value,
                    NamedSharding(self.mesh, _part_spec(axis, value.ndim)),
                )
                for value in data
            )
            specs = tuple(_part_spec(axis, value.ndim) for value in placed)

            def execute_local(
                *local_values,
                transfer=transfer,
                boundary_is_supplied=boundary_is_supplied,
            ):
                values = tuple(value[0] for value in local_values)
                result = _execute_fill_local(
                    *values[:7],
                    *values[7:],
                    transfer,
                    coarse_old_time,
                    coarse_new_time,
                    fill_time,
                    boundary_is_supplied,
                )
                return result[0][None], result[1][None]

            output_rank = packed_boundaries[level].ndim
            valid_rank = route.source_class.ndim
            mapped = jax.shard_map(
                execute_local,
                mesh=self.mesh,
                in_specs=specs,
                out_specs=(
                    _part_spec(axis, output_rank),
                    _part_spec(axis, valid_rank),
                ),
                check_vma=False,
            )
            level_output, level_valid = mapped(*placed)
            outputs.append(level_output)
            validity.append(level_valid)
        return tuple(outputs), tuple(validity)

    def _result(
        self,
        packed_values: tuple[Array, ...],
        packed_valid: tuple[Array, ...],
        boundary_supplied: tuple[bool, ...],
        /,
    ) -> FDAMRFillPatchResult:
        workspaces = tuple(
            FDAMRFillPatchWorkspace(
                layout.unpack(values),
                layout.unpack(valid),
                fill.source_class,
                fill.plan_id,
            )
            for layout, values, valid, fill in zip(
                self.layouts,
                packed_values,
                packed_valid,
                self.fill_patch_plans,
                strict=True,
            )
        )
        requests = tuple(
            FDAMRPhysicalBoundaryRequest(level, fill.physical_boundary_mask, fill.plan_id)
            for level, fill in enumerate(self.fill_patch_plans)
        )
        complete_by_level = []
        for metadata, workspace in zip(self.topology.levels, workspaces, strict=True):
            active = metadata.active.reshape(
                metadata.active.shape + (1,) * (workspace.valid.ndim - 1)
            )
            complete_by_level.append(jnp.all(workspace.valid | ~active))
        return FDAMRFillPatchResult(
            workspaces=workspaces,
            physical_boundary_requests=requests,
            complete=jnp.all(jnp.stack(tuple(complete_by_level))),
            result_id=canonical_fingerprint(
                {
                    "kind": "distributed-block-amr-fill-patch-result",
                    "prepared": self.prepared_id,
                    "epoch": self.topology.epoch.epoch_id,
                    "physical_values_supplied": boundary_supplied,
                }
            ),
        )

    def _fill_patch(
        self,
        state: BlockHierarchyState,
        /,
        *,
        coarse_old: BlockHierarchyState | None,
        coarse_new: BlockHierarchyState | None,
        coarse_old_time: ArrayLike,
        coarse_new_time: ArrayLike,
        fill_time: ArrayLike,
        physical_boundary_values: Sequence[ArrayLike | None] | None,
        distributed: bool,
    ) -> FDAMRFillPatchResult:
        old = state if coarse_old is None else coarse_old
        new = state if coarse_new is None else coarse_new
        self._validate_state(state)
        self._validate_state(old)
        self._validate_state(new)
        packed_state = self.pack(state)
        packed_old = self.pack(old)
        packed_new = self.pack(new)
        boundaries, supplied = self._boundary_inputs(
            packed_state, physical_boundary_values
        )
        values, valid = self._packed_fill_values(
            packed_state,
            packed_old,
            packed_new,
            boundaries,
            supplied,
            coarse_old_time,
            coarse_new_time,
            fill_time,
            distributed=distributed,
        )
        return self._result(values, valid, supplied)

    def serial_fill_patch(
        self,
        state: BlockHierarchyState,
        /,
        *,
        coarse_old: BlockHierarchyState | None = None,
        coarse_new: BlockHierarchyState | None = None,
        coarse_old_time: ArrayLike = 0.0,
        coarse_new_time: ArrayLike = 0.0,
        fill_time: ArrayLike = 0.0,
        physical_boundary_values: Sequence[ArrayLike | None] | None = None,
    ) -> FDAMRFillPatchResult:
        return self._fill_patch(
            state,
            coarse_old=coarse_old,
            coarse_new=coarse_new,
            coarse_old_time=coarse_old_time,
            coarse_new_time=coarse_new_time,
            fill_time=fill_time,
            physical_boundary_values=physical_boundary_values,
            distributed=False,
        )

    def distributed_fill_patch(
        self,
        state: BlockHierarchyState,
        /,
        *,
        coarse_old: BlockHierarchyState | None = None,
        coarse_new: BlockHierarchyState | None = None,
        coarse_old_time: ArrayLike = 0.0,
        coarse_new_time: ArrayLike = 0.0,
        fill_time: ArrayLike = 0.0,
        physical_boundary_values: Sequence[ArrayLike | None] | None = None,
    ) -> FDAMRFillPatchResult:
        return self._fill_patch(
            state,
            coarse_old=coarse_old,
            coarse_new=coarse_new,
            coarse_old_time=coarse_old_time,
            coarse_new_time=coarse_new_time,
            fill_time=fill_time,
            physical_boundary_values=physical_boundary_values,
            distributed=True,
        )

    def _fill_patch_reverse(
        self,
        cotangents: Sequence[ArrayLike],
        state: BlockHierarchyState,
        /,
        *,
        coarse_old: BlockHierarchyState | None,
        coarse_new: BlockHierarchyState | None,
        coarse_old_time: ArrayLike,
        coarse_new_time: ArrayLike,
        fill_time: ArrayLike,
        physical_boundary_values: Sequence[ArrayLike | None] | None,
        distributed: bool,
    ) -> tuple[
        BlockHierarchyState,
        BlockHierarchyState,
        BlockHierarchyState,
        tuple[Array | None, ...],
    ]:
        old = state if coarse_old is None else coarse_old
        new = state if coarse_new is None else coarse_new
        for value in (state, old, new):
            self._validate_state(value)
        state_values = tuple(level.values for level in state.levels)
        old_values = tuple(level.values for level in old.levels)
        new_values = tuple(level.values for level in new.levels)
        template_packed = tuple(
            layout.pack(value)
            for layout, value in zip(self.layouts, state_values, strict=True)
        )
        boundary_values, supplied = self._boundary_inputs(
            template_packed, physical_boundary_values
        )
        canonical_boundaries = tuple(
            layout.unpack(value)
            for layout, value in zip(self.layouts, boundary_values, strict=True)
        )

        def action(current_values, coarse_old_values, coarse_new_values, boundaries):
            current_packed = tuple(
                layout.pack(value)
                for layout, value in zip(self.layouts, current_values, strict=True)
            )
            old_packed = tuple(
                layout.pack(value)
                for layout, value in zip(self.layouts, coarse_old_values, strict=True)
            )
            new_packed = tuple(
                layout.pack(value)
                for layout, value in zip(self.layouts, coarse_new_values, strict=True)
            )
            packed_boundaries = tuple(
                layout.pack(value)
                for layout, value in zip(self.layouts, boundaries, strict=True)
            )
            output, _ = self._packed_fill_values(
                current_packed,
                old_packed,
                new_packed,
                packed_boundaries,
                supplied,
                coarse_old_time,
                coarse_new_time,
                fill_time,
                distributed=distributed,
            )
            return tuple(
                layout.unpack(value)
                for layout, value in zip(self.layouts, output, strict=True)
            )

        _, pullback = jax.vjp(
            action,
            state_values,
            old_values,
            new_values,
            canonical_boundaries,
        )
        cotangent_values = tuple(jnp.asarray(value) for value in cotangents)
        if len(cotangent_values) != len(self.layouts):
            raise ValueError(
                "FillPatch reverse requires one workspace cotangent per level."
            )
        state_grad, old_grad, new_grad, boundary_grad = pullback(cotangent_values)

        def hierarchy(values: tuple[Array, ...]) -> BlockHierarchyState:
            return BlockHierarchyState(
                self.topology,
                tuple(
                    BlockLevelState(level, metadata, value)
                    for level, metadata, value in zip(
                        self.topology.plan.levels,
                        self.topology.levels,
                        values,
                        strict=True,
                    )
                ),
            )

        return (
            hierarchy(state_grad),
            hierarchy(old_grad),
            hierarchy(new_grad),
            tuple(
                gradient if was_supplied else None
                for gradient, was_supplied in zip(boundary_grad, supplied, strict=True)
            ),
        )

    def serial_fill_patch_reverse(
        self,
        cotangents: Sequence[ArrayLike],
        state: BlockHierarchyState,
        /,
        *,
        coarse_old: BlockHierarchyState | None = None,
        coarse_new: BlockHierarchyState | None = None,
        coarse_old_time: ArrayLike = 0.0,
        coarse_new_time: ArrayLike = 0.0,
        fill_time: ArrayLike = 0.0,
        physical_boundary_values: Sequence[ArrayLike | None] | None = None,
    ):
        return self._fill_patch_reverse(
            cotangents,
            state,
            coarse_old=coarse_old,
            coarse_new=coarse_new,
            coarse_old_time=coarse_old_time,
            coarse_new_time=coarse_new_time,
            fill_time=fill_time,
            physical_boundary_values=physical_boundary_values,
            distributed=False,
        )

    def distributed_fill_patch_reverse(
        self,
        cotangents: Sequence[ArrayLike],
        state: BlockHierarchyState,
        /,
        *,
        coarse_old: BlockHierarchyState | None = None,
        coarse_new: BlockHierarchyState | None = None,
        coarse_old_time: ArrayLike = 0.0,
        coarse_new_time: ArrayLike = 0.0,
        fill_time: ArrayLike = 0.0,
        physical_boundary_values: Sequence[ArrayLike | None] | None = None,
    ):
        return self._fill_patch_reverse(
            cotangents,
            state,
            coarse_old=coarse_old,
            coarse_new=coarse_new,
            coarse_old_time=coarse_old_time,
            coarse_new_time=coarse_new_time,
            fill_time=fill_time,
            physical_boundary_values=physical_boundary_values,
            distributed=True,
        )

    def migration_to(
        self, target: PreparedDistributedBlockAMRHierarchy, /
    ) -> BlockAMRStableIDMigrationPlan:
        return BlockAMRStableIDMigrationPlan(self, target)

    def manifest_compatibility_data(self, /) -> dict[str, object]:
        """Canonical restore identity; partition changes still require migration."""

        return {
            "kind": "distributed-block-amr-compatibility",
            "hierarchy_plan_id": self.topology.plan.plan_id,
            "fd_hierarchy_plan_id": self.fd_hierarchy.plan.plan_id,
            "geometry_id": self.topology.epoch.geometry_id,
            "topology_id": self.topology.epoch.topology_id,
            "canonical_partition_id": self.topology.epoch.partition_id,
            "topology_epoch_id": self.topology.epoch.epoch_id,
            "distributed_partition_plan_id": self.partition.plan_id,
            "prepared_partition_id": canonical_fingerprint(
                {
                    "partition": self.partition.plan_id,
                    "layouts": [layout.layout_id for layout in self.layouts],
                }
            ),
            "part_count": self.partition.part_count,
            "local_block_capacities": list(self.local_block_capacities),
            "active_stable_block_ids": [
                [
                    int(value)
                    for value in np.asarray(layout.stable_block_ids)[
                        : layout.active_count
                    ]
                ]
                for layout in self.layouts
            ],
        }


class BlockAMRStableIDMigrationPlan(StrictModule, NonTrainableState):
    """Explicit packed repartition between equal topologies by stable block ID."""

    source: PreparedDistributedBlockAMRHierarchy
    target: PreparedDistributedBlockAMRHierarchy
    stable_block_ids: tuple[Array, ...]
    source_parts: tuple[Array, ...]
    source_local_indices: tuple[Array, ...]
    target_parts: tuple[Array, ...]
    target_local_indices: tuple[Array, ...]
    moved: tuple[Array, ...]
    migration_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: PreparedDistributedBlockAMRHierarchy,
        target: PreparedDistributedBlockAMRHierarchy,
        /,
    ):
        if not isinstance(source, PreparedDistributedBlockAMRHierarchy) or not isinstance(
            target, PreparedDistributedBlockAMRHierarchy
        ):
            raise TypeError(
                "Stable-ID migration requires two distributed AMR preparations."
            )
        if (
            source.topology.epoch.geometry_id != target.topology.epoch.geometry_id
            or source.topology.topology_id != target.topology.topology_id
            or source.topology.plan.plan_id != target.topology.plan.plan_id
            or source.partition.part_count != target.partition.part_count
        ):
            raise ValueError(
                "AMR repartition migration requires one unchanged block topology."
            )
        ids_by_level: list[Array] = []
        source_parts: list[Array] = []
        source_local: list[Array] = []
        target_parts: list[Array] = []
        target_local: list[Array] = []
        moved: list[Array] = []
        for source_layout, target_layout in zip(
            source.layouts, target.layouts, strict=True
        ):
            count = source_layout.active_count
            source_ids = np.asarray(source_layout.stable_block_ids)[:count]
            target_ids = np.asarray(target_layout.stable_block_ids)[
                : target_layout.active_count
            ]
            if not np.array_equal(source_ids, target_ids):
                raise ValueError(
                    "AMR stable block identities changed across repartition."
                )
            source_owner = np.asarray(source_layout.block_owner, dtype=np.int32)[:count]
            target_owner = np.asarray(target_layout.block_owner, dtype=np.int32)[:count]
            source_index = np.asarray(source_layout.canonical_to_local, dtype=np.int32)[
                :count
            ]
            target_index = np.asarray(target_layout.canonical_to_local, dtype=np.int32)[
                :count
            ]
            ids_by_level.append(jnp.asarray(source_ids, dtype=jnp.int32))
            source_parts.append(jnp.asarray(source_owner, dtype=jnp.int32))
            source_local.append(jnp.asarray(source_index, dtype=jnp.int32))
            target_parts.append(jnp.asarray(target_owner, dtype=jnp.int32))
            target_local.append(jnp.asarray(target_index, dtype=jnp.int32))
            moved.append(jnp.asarray(source_owner != target_owner))
        self.source = source
        self.target = target
        self.stable_block_ids = tuple(ids_by_level)
        self.source_parts = tuple(source_parts)
        self.source_local_indices = tuple(source_local)
        self.target_parts = tuple(target_parts)
        self.target_local_indices = tuple(target_local)
        self.moved = tuple(moved)
        self.migration_id = canonical_fingerprint(
            {
                "kind": "block-amr-stable-id-migration",
                "source": source.prepared_id,
                "target": target.prepared_id,
                "stable_ids": [array_tree_fingerprint(value) for value in ids_by_level],
                "source_parts": [array_tree_fingerprint(value) for value in source_parts],
                "target_parts": [array_tree_fingerprint(value) for value in target_parts],
            }
        )

    @property
    def moved_block_counts(self) -> tuple[int, ...]:
        return tuple(int(np.count_nonzero(np.asarray(value))) for value in self.moved)

    def migrate(self, packed_values: Sequence[ArrayLike], /) -> tuple[Array, ...]:
        values = tuple(jnp.asarray(value) for value in packed_values)
        if len(values) != len(self.stable_block_ids):
            raise ValueError("Migration requires one packed value array per AMR level.")
        migrated: list[Array] = []
        for level, value in enumerate(values):
            source_layout = self.source.layouts[level]
            target_layout = self.target.layouts[level]
            if value.shape[:2] != (
                self.source.partition.part_count,
                source_layout.local_block_capacity,
            ):
                raise ValueError("Migration source does not match packed level capacity.")
            result = jnp.zeros(
                (
                    self.target.partition.part_count,
                    target_layout.local_block_capacity,
                    *value.shape[2:],
                ),
                dtype=value.dtype,
            )
            source_values = value[
                self.source_parts[level], self.source_local_indices[level]
            ]
            result = result.at[
                self.target_parts[level], self.target_local_indices[level]
            ].set(source_values)
            migrated.append(result)
        return tuple(migrated)


__all__ = [
    "BlockAMRPartitionPlan",
    "BlockAMRStableIDMigrationPlan",
    "DistributedBlockAMRResourceEvidence",
    "PreparedDistributedBlockAMRHierarchy",
]
