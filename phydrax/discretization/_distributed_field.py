#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from .._execution_runtime import ExecutionGroup
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation, RowRelation
from ..typing import checked, ConvertibleToArray


def _color_directed_pairs(
    pairs: Iterable[tuple[int, int]],
    /,
    *,
    include_empty_phase: bool = False,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Deterministically color peer routes for one-source/one-target phases."""

    remaining = set(pairs)
    phases: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        phase: list[tuple[int, int]] = []
        used_source: set[int] = set()
        used_target: set[int] = set()
        for source, target in sorted(remaining):
            if source not in used_source and target not in used_target:
                phase.append((source, target))
                used_source.add(source)
                used_target.add(target)
        phases.append(tuple(phase))
        remaining.difference_update(phase)
    if not phases and include_empty_phase:
        return ((),)
    return tuple(phases)


class DistributedHaloPlan(StrictModule, NonTrainableState):
    """Padded owned/halo layout and colored peer permutations.

    Every communication phase has at most one outgoing and incoming peer per
    partition, so ``lax.ppermute`` routes fixed-size packed messages without an
    all-gather. Canonical global IDs determine ownership and transpose routing.
    A group partition may own no entities; its padded rows remain invalid, so
    it participates in collectives without inventing owners or messages.
    """

    entity_owner: Array
    local_global_ids: Array
    local_valid: Array
    local_owned: Array
    phase_send_indices: Array
    phase_receive_indices: Array
    phase_send_valid: Array
    phase_receive_valid: Array
    permutations: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    reverse_permutations: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    local_capacity: int = eqx.field(static=True)
    message_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, entity_owner: ArrayLike, adjacency: ArrayLike, part_count: int, /
    ) -> None:
        owner = np.asarray(entity_owner)
        pairs = np.asarray(adjacency)
        if isinstance(part_count, (bool, np.bool_)) or not isinstance(
            part_count, (int, np.integer)
        ):
            raise TypeError(
                "Distributed partition count must be an integer excluding bool."
            )
        parts = int(part_count)
        if (
            owner.ndim != 1
            or owner.size == 0
            or not np.issubdtype(owner.dtype, np.integer)
            or parts > np.iinfo(np.int32).max + 1
            or owner.size > np.iinfo(np.int32).max + 1
            or parts <= 0
            or np.any(owner < 0)
            or np.any(owner >= parts)
        ):
            raise ValueError(
                "Distributed entity ownership must contain owned rows within the partition range."
            )
        if (
            pairs.ndim != 2
            or pairs.shape[1:] != (2,)
            or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs < 0)
            or np.any(pairs >= owner.size)
            or np.any(pairs[:, 0] == pairs[:, 1])
        ):
            raise ValueError(
                "Distributed adjacency must contain valid distinct entity pairs."
            )
        local_ids: list[np.ndarray] = []
        local_owned: list[np.ndarray] = []
        maps: list[dict[int, int]] = []
        for part in range(parts):
            owned = set(np.flatnonzero(owner == part).tolist())
            halo: set[int] = set()
            for left, right in pairs:
                if int(left) in owned and owner[right] != part:
                    halo.add(int(right))
                if int(right) in owned and owner[left] != part:
                    halo.add(int(left))
            ids = np.asarray((*sorted(owned), *sorted(halo)), dtype=np.int32)
            flags = np.asarray([value in owned for value in ids], dtype=np.bool_)
            local_ids.append(ids)
            local_owned.append(flags)
            maps.append({int(value): index for index, value in enumerate(ids)})
        capacity = max(values.size for values in local_ids)
        global_ids = np.zeros((parts, capacity), dtype=np.int32)
        valid = np.zeros((parts, capacity), dtype=np.bool_)
        owned_mask = np.zeros((parts, capacity), dtype=np.bool_)
        for part, (ids, flags) in enumerate(zip(local_ids, local_owned, strict=True)):
            global_ids[part, : ids.size] = ids
            valid[part, : ids.size] = True
            owned_mask[part, : ids.size] = flags

        pair_messages: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for receiver in range(parts):
            for local_index, global_id in enumerate(local_ids[receiver]):
                source = int(owner[global_id])
                if source == receiver:
                    continue
                pair_messages.setdefault((source, receiver), []).append(
                    (maps[source][int(global_id)], local_index)
                )
        phases = _color_directed_pairs(pair_messages)
        message_capacity = max(
            (len(message) for message in pair_messages.values()), default=1
        )
        send = np.zeros((len(phases), parts, message_capacity), dtype=np.int32)
        receive = np.zeros_like(send)
        send_valid = np.zeros_like(send, dtype=np.bool_)
        receive_valid = np.zeros_like(send, dtype=np.bool_)
        permutations: list[tuple[tuple[int, int], ...]] = []
        for phase_index, phase in enumerate(phases):
            permutations.append(tuple(phase))
            for source, target in phase:
                message = pair_messages[(source, target)]
                count = len(message)
                send[phase_index, source, :count] = [value[0] for value in message]
                receive[phase_index, target, :count] = [value[1] for value in message]
                send_valid[phase_index, source, :count] = True
                receive_valid[phase_index, target, :count] = True
        self.entity_owner = jnp.asarray(owner, dtype=jnp.int32)
        self.local_global_ids = jnp.asarray(global_ids)
        self.local_valid = jnp.asarray(valid)
        self.local_owned = jnp.asarray(owned_mask)
        self.phase_send_indices = jnp.asarray(send)
        self.phase_receive_indices = jnp.asarray(receive)
        self.phase_send_valid = jnp.asarray(send_valid)
        self.phase_receive_valid = jnp.asarray(receive_valid)
        self.permutations = tuple(permutations)
        self.reverse_permutations = tuple(
            tuple((target, source) for source, target in phase) for phase in permutations
        )
        self.entity_count, self.part_count = owner.size, parts
        self.local_capacity, self.message_capacity = capacity, message_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-halo-plan",
                "owner": owner,
                "adjacency": pairs,
                "local_global_ids": global_ids,
                "local_owned": owned_mask,
                "permutations": permutations,
            }
        )

    def pack_owned(self, global_values: ArrayLike, /) -> Array:
        values = jnp.asarray(global_values)
        if values.ndim == 0 or values.shape[0] != self.entity_count:
            raise ValueError("Global field leading axis must match distributed entities.")
        packed = values[self.local_global_ids]
        mask = self.local_owned.reshape(self.local_owned.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, packed, jnp.zeros_like(packed))

    def pack_reference(self, global_values: ArrayLike, /) -> Array:
        values = jnp.asarray(global_values)
        if values.ndim == 0 or values.shape[0] != self.entity_count:
            raise ValueError("Global field leading axis must match distributed entities.")
        packed = values[self.local_global_ids]
        mask = self.local_valid.reshape(self.local_valid.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, packed, jnp.zeros_like(packed))

    def unpack_owned(self, local_values: ArrayLike, /) -> Array:
        values = jnp.asarray(local_values)
        if values.shape[:2] != (self.part_count, self.local_capacity):
            raise ValueError("Local distributed field shape disagrees with halo plan.")
        result = jnp.zeros((self.entity_count,) + values.shape[2:], dtype=values.dtype)
        for part in range(self.part_count):
            ids = self.local_global_ids[part]
            mask = self.local_owned[part].reshape(
                (self.local_capacity,) + (1,) * (values.ndim - 2)
            )
            result = result.at[ids].add(jnp.where(mask, values[part], 0))
        return result

    def exchange(self, local_values: Array, part: Array, /, *, axis_name: str) -> Array:
        values = local_values
        for phase, permutation in enumerate(self.permutations):
            send_indices = self.phase_send_indices[phase, part]
            send_valid = self.phase_send_valid[phase, part]
            payload = values[send_indices]
            mask = send_valid.reshape(send_valid.shape + (1,) * (values.ndim - 1))
            payload = jnp.where(mask, payload, 0)
            received = jax.lax.ppermute(payload, axis_name=axis_name, perm=permutation)
            receive_indices = self.phase_receive_indices[phase, part]
            receive_valid = self.phase_receive_valid[phase, part].reshape(
                send_valid.shape + (1,) * (values.ndim - 1)
            )
            # Padded slots are masked; an ownerless partition has no valid routes.
            destination = jnp.any(
                jnp.asarray(
                    [target == part for _, target in permutation], dtype=jnp.bool_
                )
            )
            values = jax.lax.cond(
                destination,
                lambda current: current.at[receive_indices].set(
                    jnp.where(receive_valid, received, current[receive_indices])
                ),
                lambda current: current,
                values,
            )
        return values

    def accumulate_halo(
        self, local_values: Array, part: Array, /, *, axis_name: str
    ) -> Array:
        values = local_values
        for phase, permutation in reversed(tuple(enumerate(self.reverse_permutations))):
            receive_indices = self.phase_receive_indices[phase, part]
            source_valid = self.phase_receive_valid[phase, part]
            payload = values[receive_indices]
            mask = source_valid.reshape(source_valid.shape + (1,) * (values.ndim - 1))
            payload = jnp.where(mask, payload, 0)
            received = jax.lax.ppermute(payload, axis_name=axis_name, perm=permutation)
            send_indices = self.phase_send_indices[phase, part]
            destination = jnp.any(
                jnp.asarray(
                    [target == part for _, target in permutation], dtype=jnp.bool_
                )
            )
            values = jax.lax.cond(
                destination,
                lambda current: current.at[send_indices].add(received),
                lambda current: current,
                values,
            )
        return values


class _TransferExchange(NamedTuple):
    owned_global_ids: Array
    owned_valid: Array
    local_global_ids: Array
    local_valid: Array
    phase_send_indices: Array
    phase_receive_indices: Array
    phase_send_valid: Array
    phase_receive_valid: Array
    permutations: tuple[tuple[tuple[int, int], ...], ...]
    owned_capacity: int
    local_capacity: int


def _transfer_exchange_layout(
    owner: np.ndarray,
    needed_by_part: list[set[int]],
    part_count: int,
    /,
) -> _TransferExchange:
    owned = [np.flatnonzero(owner == part).astype(np.int64) for part in range(part_count)]
    local: list[np.ndarray] = []
    owned_slots: list[dict[int, int]] = []
    local_slots: list[dict[int, int]] = []
    for part in range(part_count):
        own = owned[part]
        own_set = set(int(value) for value in own)
        halo = sorted(needed_by_part[part].difference(own_set))
        values = np.asarray((*own.tolist(), *halo), dtype=np.int64)
        local.append(values)
        owned_slots.append({int(value): index for index, value in enumerate(own)})
        local_slots.append({int(value): index for index, value in enumerate(values)})
    owned_capacity = max(1, max((values.size for values in owned), default=0))
    local_capacity = max(1, max((values.size for values in local), default=0))
    owned_ids = np.zeros((part_count, owned_capacity), dtype=np.int32)
    owned_valid = np.zeros((part_count, owned_capacity), dtype=np.bool_)
    local_ids = np.zeros((part_count, local_capacity), dtype=np.int32)
    local_valid = np.zeros((part_count, local_capacity), dtype=np.bool_)
    for part in range(part_count):
        owned_ids[part, : owned[part].size] = owned[part]
        owned_valid[part, : owned[part].size] = True
        local_ids[part, : local[part].size] = local[part]
        local_valid[part, : local[part].size] = True
    messages: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for receiver in range(part_count):
        for entity in local[receiver][owned[receiver].size :]:
            source = int(owner[entity])
            messages.setdefault((source, receiver), []).append(
                (
                    owned_slots[source][int(entity)],
                    local_slots[receiver][int(entity)],
                )
            )
    phases = _color_directed_pairs(messages)
    message_capacity = max((len(message) for message in messages.values()), default=1)
    send = np.zeros((len(phases), part_count, message_capacity), dtype=np.int32)
    receive = np.zeros_like(send)
    send_valid = np.zeros_like(send, dtype=np.bool_)
    receive_valid = np.zeros_like(send, dtype=np.bool_)
    for phase_index, phase in enumerate(phases):
        for source, target in phase:
            message = messages[(source, target)]
            count = len(message)
            send[phase_index, source, :count] = [entry[0] for entry in message]
            receive[phase_index, target, :count] = [entry[1] for entry in message]
            send_valid[phase_index, source, :count] = True
            receive_valid[phase_index, target, :count] = True
    return _TransferExchange(
        jnp.asarray(owned_ids),
        jnp.asarray(owned_valid),
        jnp.asarray(local_ids),
        jnp.asarray(local_valid),
        jnp.asarray(send),
        jnp.asarray(receive),
        jnp.asarray(send_valid),
        jnp.asarray(receive_valid),
        phases,
        owned_capacity,
        local_capacity,
    )


def _exchange_transfer_values(
    layout: _TransferExchange,
    values: Array,
    part: Array,
    axis_name: str,
    /,
) -> Array:
    result = values
    for phase, permutation in enumerate(layout.permutations):
        send_indices = layout.phase_send_indices[phase, part]
        send_valid = layout.phase_send_valid[phase, part]
        payload = result[send_indices]
        mask = send_valid.reshape(send_valid.shape + (1,) * (result.ndim - 1))
        received = jax.lax.ppermute(
            jnp.where(mask, payload, 0),
            axis_name=axis_name,
            perm=permutation,
        )
        receive_indices = layout.phase_receive_indices[phase, part]
        receive_valid = layout.phase_receive_valid[phase, part].reshape(
            send_valid.shape + (1,) * (result.ndim - 1)
        )
        destination = jnp.any(
            jnp.asarray([target == part for _, target in permutation], dtype=jnp.bool_)
        )
        result = jax.lax.cond(
            destination,
            lambda current: current.at[receive_indices].set(
                jnp.where(
                    receive_valid,
                    received,
                    current[receive_indices],
                )
            ),
            lambda current: current,
            result,
        )
    return result


def _pad_transfer_routes(
    part_count: int,
    routes: list[list[tuple[int, int, object]]],
    dtype: np.dtype,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    capacity = max(1, max((len(rows) for rows in routes), default=0))
    targets = np.zeros((part_count, capacity), dtype=np.int32)
    sources = np.zeros((part_count, capacity), dtype=np.int32)
    coefficients = np.zeros((part_count, capacity), dtype=dtype)
    valid = np.zeros((part_count, capacity), dtype=np.bool_)
    for part, rows in enumerate(routes):
        if not rows:
            continue
        targets[part, : len(rows)] = [row[0] for row in rows]
        sources[part, : len(rows)] = [row[1] for row in rows]
        coefficients[part, : len(rows)] = [row[2] for row in rows]
        valid[part, : len(rows)] = True
    return targets, sources, coefficients, valid


def _local_sparse_route_action(
    values: Array,
    rows: Array,
    columns: Array,
    coefficients: Array,
    valid: Array,
    output_capacity: int,
    /,
) -> Array:
    payload_shape = values.shape[1:]
    dtype = jnp.result_type(values.dtype, coefficients.dtype)
    routed = values[columns].astype(dtype) * coefficients.astype(dtype).reshape(
        (coefficients.shape[0],) + (1,) * len(payload_shape)
    )
    mask = valid.reshape(valid.shape + (1,) * len(payload_shape))
    result = jnp.zeros((output_capacity,) + payload_shape, dtype=dtype)
    return result.at[rows].add(jnp.where(mask, routed, 0))


class DistributedSparseTransfer(StrictModule, NonTrainableState):
    """Owner-local sparse primal and algebraic-transpose field transfer.

    Source and target DOFs have independent ownership maps. Only sparse routes
    requested by target owners are exchanged; no global dense map or gathered
    global field is needed by the distributed actions. ``transpose_distributed``
    is the algebraic transpose, not a Hilbert adjoint.
    """

    source_exchange: _TransferExchange
    target_exchange: _TransferExchange
    forward_rows: Array
    forward_columns: Array
    forward_coefficients: Array
    forward_valid: Array
    transpose_rows: Array
    transpose_columns: Array
    transpose_coefficients: Array
    transpose_valid: Array
    source_owner: Array
    target_owner: Array
    source_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        relation: EdgeRelation | RowRelation,
        coefficients: ConvertibleToArray,
        source_owner: ConvertibleToArray,
        target_owner: ConvertibleToArray,
        part_count: int,
        /,
        *,
        operator_id: str | None = None,
    ) -> None:
        if not isinstance(relation, (EdgeRelation, RowRelation)):
            raise TypeError("relation must be an EdgeRelation or RowRelation.")
        route_relation = (
            relation
            if isinstance(relation, EdgeRelation)
            else relation.as_edge_relation()
        )
        values = jnp.asarray(coefficients)
        if isinstance(relation, RowRelation):
            if values.shape != relation.route_shape:
                raise ValueError("Transfer coefficients must match sparse row routes.")
            values = values.reshape(route_relation.route_shape)
        elif values.shape != route_relation.route_shape:
            raise ValueError("Transfer coefficients must match sparse edge routes.")
        owner_source = np.asarray(source_owner)
        owner_target = np.asarray(target_owner)
        if isinstance(part_count, (bool, np.bool_)) or not isinstance(
            part_count, (int, np.integer)
        ):
            raise TypeError("part_count must be an integer excluding bool.")
        parts = int(part_count)
        if (
            parts <= 0
            or parts > np.iinfo(np.int32).max + 1
            or route_relation.source_size > np.iinfo(np.int32).max + 1
            or route_relation.target_size > np.iinfo(np.int32).max + 1
            or owner_source.shape != (route_relation.source_size,)
            or owner_target.shape != (route_relation.target_size,)
            or not np.issubdtype(owner_source.dtype, np.integer)
            or not np.issubdtype(owner_target.dtype, np.integer)
            or np.any(owner_source < 0)
            or np.any(owner_source >= parts)
            or np.any(owner_target < 0)
            or np.any(owner_target >= parts)
        ):
            raise ValueError(
                "Source/target ownership must align with their sparse DOF spaces."
            )
        route_valid = np.asarray(route_relation.valid, dtype=np.bool_)
        host_values = np.asarray(values)
        if not np.issubdtype(host_values.dtype, np.inexact) or np.any(
            ~np.isfinite(host_values[route_valid])
        ):
            raise ValueError("Active distributed transfer coefficients must be finite.")
        sources = np.asarray(route_relation.source_indices, dtype=np.int64)[route_valid]
        targets = np.asarray(route_relation.target_indices, dtype=np.int64)[route_valid]
        weights = host_values[route_valid]
        needed_source = [set() for _ in range(parts)]
        needed_target = [set() for _ in range(parts)]
        for source, target in zip(sources, targets, strict=True):
            needed_source[int(owner_target[target])].add(int(source))
            needed_target[int(owner_source[source])].add(int(target))
        source_exchange = _transfer_exchange_layout(
            owner_source.astype(np.int64, copy=False), needed_source, parts
        )
        target_exchange = _transfer_exchange_layout(
            owner_target.astype(np.int64, copy=False), needed_target, parts
        )
        source_owned_slots = []
        target_owned_slots = []
        source_local_slots = []
        target_local_slots = []
        for part in range(parts):
            source_owned_ids = np.asarray(
                source_exchange.owned_global_ids[part][
                    np.asarray(source_exchange.owned_valid[part], dtype=np.bool_)
                ],
                dtype=np.int64,
            )
            target_owned_ids = np.asarray(
                target_exchange.owned_global_ids[part][
                    np.asarray(target_exchange.owned_valid[part], dtype=np.bool_)
                ],
                dtype=np.int64,
            )
            source_local_ids = np.asarray(
                source_exchange.local_global_ids[part][
                    np.asarray(source_exchange.local_valid[part], dtype=np.bool_)
                ],
                dtype=np.int64,
            )
            target_local_ids = np.asarray(
                target_exchange.local_global_ids[part][
                    np.asarray(target_exchange.local_valid[part], dtype=np.bool_)
                ],
                dtype=np.int64,
            )
            source_owned_slots.append(
                {int(value): index for index, value in enumerate(source_owned_ids)}
            )
            target_owned_slots.append(
                {int(value): index for index, value in enumerate(target_owned_ids)}
            )
            source_local_slots.append(
                {int(value): index for index, value in enumerate(source_local_ids)}
            )
            target_local_slots.append(
                {int(value): index for index, value in enumerate(target_local_ids)}
            )
        forward_routes = [[] for _ in range(parts)]
        transpose_routes = [[] for _ in range(parts)]
        for source, target, weight in zip(sources, targets, weights, strict=True):
            source_id, target_id = int(source), int(target)
            target_part = int(owner_target[target_id])
            source_part = int(owner_source[source_id])
            forward_routes[target_part].append(
                (
                    target_owned_slots[target_part][target_id],
                    source_local_slots[target_part][source_id],
                    weight,
                )
            )
            transpose_routes[source_part].append(
                (
                    source_owned_slots[source_part][source_id],
                    target_local_slots[source_part][target_id],
                    weight,
                )
            )
        (
            forward_rows,
            forward_columns,
            forward_coefficients,
            forward_valid,
        ) = _pad_transfer_routes(parts, forward_routes, host_values.dtype)
        (
            transpose_rows,
            transpose_columns,
            transpose_coefficients,
            transpose_valid,
        ) = _pad_transfer_routes(parts, transpose_routes, host_values.dtype)
        self.source_exchange, self.target_exchange = source_exchange, target_exchange
        self.source_owner = jnp.asarray(owner_source, dtype=jnp.int32)
        self.target_owner = jnp.asarray(owner_target, dtype=jnp.int32)
        self.forward_rows = jnp.asarray(forward_rows)
        self.forward_columns = jnp.asarray(forward_columns)
        self.forward_coefficients = jnp.asarray(forward_coefficients)
        self.forward_valid = jnp.asarray(forward_valid)
        self.transpose_rows = jnp.asarray(transpose_rows)
        self.transpose_columns = jnp.asarray(transpose_columns)
        self.transpose_coefficients = jnp.asarray(transpose_coefficients)
        self.transpose_valid = jnp.asarray(transpose_valid)
        self.source_size, self.target_size, self.part_count = (
            route_relation.source_size,
            route_relation.target_size,
            parts,
        )
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "distributed-sparse-field-transfer",
                "operator": operator_id,
                "source_owner": owner_source,
                "target_owner": owner_target,
                "source_indices": sources,
                "target_indices": targets,
                "coefficients": array_tree_fingerprint(host_values[route_valid]),
                "source_local_ids": np.asarray(source_exchange.local_global_ids),
                "target_local_ids": np.asarray(target_exchange.local_global_ids),
            }
        )

    def pack_source_owned(self, global_values: ArrayLike, /) -> Array:
        values = jnp.asarray(global_values)
        if values.ndim == 0 or values.shape[0] != self.source_size:
            raise ValueError("Global source fields must match the source DOF extent.")
        if self.source_size == 0:
            return jnp.zeros(
                (self.part_count, self.source_exchange.owned_capacity) + values.shape[1:],
                dtype=values.dtype,
            )
        identifiers = jnp.where(
            self.source_exchange.owned_valid,
            self.source_exchange.owned_global_ids,
            0,
        )
        packed = values[identifiers]
        mask = self.source_exchange.owned_valid.reshape(
            self.source_exchange.owned_valid.shape + (1,) * (values.ndim - 1)
        )
        return jnp.where(mask, packed, 0)

    def pack_target_owned(self, global_values: ArrayLike, /) -> Array:
        values = jnp.asarray(global_values)
        if values.ndim == 0 or values.shape[0] != self.target_size:
            raise ValueError("Global target fields must match the target DOF extent.")
        if self.target_size == 0:
            return jnp.zeros(
                (self.part_count, self.target_exchange.owned_capacity) + values.shape[1:],
                dtype=values.dtype,
            )
        identifiers = jnp.where(
            self.target_exchange.owned_valid,
            self.target_exchange.owned_global_ids,
            0,
        )
        packed = values[identifiers]
        mask = self.target_exchange.owned_valid.reshape(
            self.target_exchange.owned_valid.shape + (1,) * (values.ndim - 1)
        )
        return jnp.where(mask, packed, 0)

    def unpack_target_owned(self, local_values: ArrayLike, /) -> Array:
        values = jnp.asarray(local_values)
        if values.shape[:2] != (
            self.part_count,
            self.target_exchange.owned_capacity,
        ):
            raise ValueError(
                "Target-owned field layout disagrees with its transfer plan."
            )
        if self.target_size == 0:
            return jnp.zeros((0,) + values.shape[2:], dtype=values.dtype)
        result = jnp.zeros((self.target_size,) + values.shape[2:], dtype=values.dtype)
        for part in range(self.part_count):
            mask = self.target_exchange.owned_valid[part].reshape(
                (self.target_exchange.owned_capacity,) + (1,) * (values.ndim - 2)
            )
            result = result.at[self.target_exchange.owned_global_ids[part]].add(
                jnp.where(mask, values[part], 0)
            )
        return result

    def unpack_source_owned(self, local_values: ArrayLike, /) -> Array:
        values = jnp.asarray(local_values)
        if values.shape[:2] != (
            self.part_count,
            self.source_exchange.owned_capacity,
        ):
            raise ValueError(
                "Source-owned field layout disagrees with its transfer plan."
            )
        if self.source_size == 0:
            return jnp.zeros((0,) + values.shape[2:], dtype=values.dtype)
        result = jnp.zeros((self.source_size,) + values.shape[2:], dtype=values.dtype)
        for part in range(self.part_count):
            mask = self.source_exchange.owned_valid[part].reshape(
                (self.source_exchange.owned_capacity,) + (1,) * (values.ndim - 2)
            )
            result = result.at[self.source_exchange.owned_global_ids[part]].add(
                jnp.where(mask, values[part], 0)
            )
        return result

    def serial_reference(self, global_source: ArrayLike, /) -> Array:
        values = jnp.asarray(global_source)
        if values.ndim == 0 or values.shape[0] != self.source_size:
            raise ValueError("Global source fields must match the source DOF extent.")
        outputs = []
        for part in range(self.part_count):
            valid = self.source_exchange.local_valid[part]
            identifiers = self.source_exchange.local_global_ids[part]
            if self.source_size == 0:
                local = jnp.zeros(
                    (self.source_exchange.local_capacity,) + values.shape[1:],
                    dtype=values.dtype,
                )
            else:
                mask = valid.reshape((valid.size,) + (1,) * (values.ndim - 1))
                local = jnp.where(mask, values[identifiers], 0)
            outputs.append(
                _local_sparse_route_action(
                    local,
                    self.forward_rows[part],
                    self.forward_columns[part],
                    self.forward_coefficients[part],
                    self.forward_valid[part],
                    self.target_exchange.owned_capacity,
                )
            )
        return self.unpack_target_owned(jnp.stack(outputs))

    def serial_transpose_reference(self, global_target: ArrayLike, /) -> Array:
        values = jnp.asarray(global_target)
        if values.ndim == 0 or values.shape[0] != self.target_size:
            raise ValueError("Global target fields must match the target DOF extent.")
        outputs = []
        for part in range(self.part_count):
            valid = self.target_exchange.local_valid[part]
            identifiers = self.target_exchange.local_global_ids[part]
            if self.target_size == 0:
                local = jnp.zeros(
                    (self.target_exchange.local_capacity,) + values.shape[1:],
                    dtype=values.dtype,
                )
            else:
                mask = valid.reshape((valid.size,) + (1,) * (values.ndim - 1))
                local = jnp.where(mask, values[identifiers], 0)
            outputs.append(
                _local_sparse_route_action(
                    local,
                    self.transpose_rows[part],
                    self.transpose_columns[part],
                    self.transpose_coefficients[part],
                    self.transpose_valid[part],
                    self.source_exchange.owned_capacity,
                )
            )
        return self.unpack_source_owned(jnp.stack(outputs))

    def _distributed_action(
        self,
        owned_values: ArrayLike,
        transpose_action: bool,
        /,
        *,
        axis_name: str,
        execution_group: ExecutionGroup | None,
    ) -> Array:
        values = jnp.asarray(owned_values)
        exchange = self.target_exchange if transpose_action else self.source_exchange
        output_exchange = (
            self.source_exchange if transpose_action else self.target_exchange
        )
        expected = (self.part_count, exchange.owned_capacity)
        if values.ndim < 2 or values.shape[:2] != expected:
            raise ValueError(
                "Distributed transfer input must use its owned local DOF layout."
            )
        devices = (
            tuple(jax.devices()) if execution_group is None else execution_group.devices
        )
        if self.part_count != len(devices):
            raise ValueError(
                "Distributed transfer requires one assigned JAX device per partition."
            )
        mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
        input_spec = PartitionSpec(axis_name, *(None for _ in range(values.ndim - 1)))
        part_spec = PartitionSpec(axis_name)
        sharded_values = jax.device_put(values, NamedSharding(mesh, input_spec))
        parts = jax.device_put(
            jnp.arange(self.part_count, dtype=jnp.int32),
            NamedSharding(mesh, part_spec),
        )

        def action(local: Array, part: Array) -> Array:
            local_owned = local[0]
            part_index = part[0]
            owned_mask = exchange.owned_valid[part_index].reshape(
                (exchange.owned_capacity,) + (1,) * (local_owned.ndim - 1)
            )
            local_values = jnp.zeros(
                (exchange.local_capacity,) + local_owned.shape[1:],
                dtype=local_owned.dtype,
            )
            local_values = local_values.at[: exchange.owned_capacity].set(
                jnp.where(owned_mask, local_owned, 0)
            )
            local_values = _exchange_transfer_values(
                exchange, local_values, part_index, axis_name
            )
            if transpose_action:
                rows, columns = self.transpose_rows, self.transpose_columns
                coefficients, valid = (
                    self.transpose_coefficients,
                    self.transpose_valid,
                )
            else:
                rows, columns = self.forward_rows, self.forward_columns
                coefficients, valid = (
                    self.forward_coefficients,
                    self.forward_valid,
                )
            result = _local_sparse_route_action(
                local_values,
                rows[part_index],
                columns[part_index],
                coefficients[part_index],
                valid[part_index],
                output_exchange.owned_capacity,
            )
            return result[None]

        output_spec = PartitionSpec(axis_name, *(None for _ in range(values.ndim - 1)))
        mapped = jax.shard_map(
            action,
            mesh=mesh,
            in_specs=(input_spec, part_spec),
            out_specs=output_spec,
            check_vma=False,
        )
        return mapped(sharded_values, parts)

    def distributed(
        self,
        source_owned_values: ArrayLike,
        /,
        *,
        axis_name: str = "parts",
        execution_group: ExecutionGroup | None = None,
    ) -> Array:
        """Apply the sparse primal to source-owned shards and return target-owned shards."""
        return self._distributed_action(
            source_owned_values,
            False,
            axis_name=axis_name,
            execution_group=execution_group,
        )

    def distributed_transpose(
        self,
        target_owned_values: ArrayLike,
        /,
        *,
        axis_name: str = "parts",
        execution_group: ExecutionGroup | None = None,
    ) -> Array:
        """Apply the algebraic transpose to target-owned shards."""
        return self._distributed_action(
            target_owned_values,
            True,
            axis_name=axis_name,
            execution_group=execution_group,
        )


class DistributedLocalOperator(StrictModule, NonTrainableState):
    """Local owned-row operator with an explicit transpose and halo plan."""

    halo: DistributedHaloPlan
    local_action: Callable[[Array, Array, Array, Array, Array], Array] = eqx.field(
        static=True
    )
    local_transpose: Callable[[Array, Array, Array, Array, Array], Array] = eqx.field(
        static=True
    )
    operator_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        halo: DistributedHaloPlan,
        local_action: Callable[[Array, Array, Array, Array, Array], Array],
        local_transpose: Callable[[Array, Array, Array, Array, Array], Array],
        /,
        *,
        operator_name: str,
    ) -> None:
        if not callable(local_action) or not callable(local_transpose):
            raise TypeError(
                "Distributed local forward and transpose actions are required."
            )
        name = str(operator_name).strip()
        if not name:
            raise ValueError("Distributed operator name is required.")
        self.halo, self.local_action, self.local_transpose = (
            halo,
            local_action,
            local_transpose,
        )
        self.operator_id = canonical_fingerprint(
            {"kind": "distributed-local-operator", "halo": halo.plan_id, "name": name}
        )

    def serial_reference(self, global_values: ArrayLike, /) -> Array:
        local = self.halo.pack_reference(global_values)
        outputs = []
        for part in range(self.halo.part_count):
            outputs.append(
                self.local_action(
                    jnp.asarray(part, dtype=jnp.int32),
                    local[part],
                    self.halo.local_global_ids[part],
                    self.halo.local_valid[part],
                    self.halo.local_owned[part],
                )
            )
        return self.halo.unpack_owned(jnp.stack(outputs))

    def distributed(
        self,
        global_values: ArrayLike,
        /,
        *,
        axis_name: str = "parts",
        execution_group: ExecutionGroup | None = None,
    ) -> Array:
        devices = (
            tuple(jax.devices()) if execution_group is None else execution_group.devices
        )
        if self.halo.part_count != len(devices):
            raise ValueError(
                "Distributed execution requires one assigned JAX device per partition."
            )
        mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
        packed = self.halo.pack_owned(global_values)
        parts = jnp.arange(self.halo.part_count, dtype=jnp.int32)
        packed_spec = PartitionSpec(
            axis_name,
            *(None for _ in range(packed.ndim - 1)),
        )
        part_spec = PartitionSpec(axis_name)
        packed = jax.device_put(packed, NamedSharding(mesh, packed_spec))
        parts = jax.device_put(parts, NamedSharding(mesh, part_spec))

        def action(local: Array, part: Array) -> Array:
            part_index = part[0]
            exchanged = self.halo.exchange(
                local[0],
                part_index,
                axis_name=axis_name,
            )
            result = self.local_action(
                part_index,
                exchanged,
                self.halo.local_global_ids[part_index],
                self.halo.local_valid[part_index],
                self.halo.local_owned[part_index],
            )
            return result[None, ...]

        mapped = jax.shard_map(
            action,
            mesh=mesh,
            in_specs=(packed_spec, part_spec),
            out_specs=packed_spec,
            check_vma=False,
        )
        return self.halo.unpack_owned(mapped(packed, parts))

    def serial_transpose_reference(self, global_values: ArrayLike, /) -> Array:
        values = jnp.asarray(global_values)
        local = self.halo.pack_owned(values)
        result = jnp.zeros(
            (self.halo.entity_count,) + values.shape[1:], dtype=values.dtype
        )
        for part in range(self.halo.part_count):
            contribution = self.local_transpose(
                jnp.asarray(part, dtype=jnp.int32),
                local[part],
                self.halo.local_global_ids[part],
                self.halo.local_valid[part],
                self.halo.local_owned[part],
            )
            mask = self.halo.local_valid[part].reshape(
                (self.halo.local_capacity,) + (1,) * (values.ndim - 1)
            )
            result = result.at[self.halo.local_global_ids[part]].add(
                jnp.where(mask, contribution, 0)
            )
        return result

    def distributed_transpose(
        self,
        global_values: ArrayLike,
        /,
        *,
        axis_name: str = "parts",
        execution_group: ExecutionGroup | None = None,
    ) -> Array:
        devices = (
            tuple(jax.devices()) if execution_group is None else execution_group.devices
        )
        if self.halo.part_count != len(devices):
            raise ValueError(
                "Distributed transpose requires one assigned JAX device per partition."
            )
        mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
        packed = self.halo.pack_owned(global_values)
        parts = jnp.arange(self.halo.part_count, dtype=jnp.int32)
        packed_spec = PartitionSpec(
            axis_name,
            *(None for _ in range(packed.ndim - 1)),
        )
        part_spec = PartitionSpec(axis_name)
        packed = jax.device_put(packed, NamedSharding(mesh, packed_spec))
        parts = jax.device_put(parts, NamedSharding(mesh, part_spec))

        def action(local: Array, part: Array) -> Array:
            part_index = part[0]
            local_result = self.local_transpose(
                part_index,
                local[0],
                self.halo.local_global_ids[part_index],
                self.halo.local_valid[part_index],
                self.halo.local_owned[part_index],
            )
            accumulated = self.halo.accumulate_halo(
                local_result,
                part_index,
                axis_name=axis_name,
            )
            return accumulated[None, ...]

        mapped = jax.shard_map(
            action,
            mesh=mesh,
            in_specs=(packed_spec, part_spec),
            out_specs=packed_spec,
            check_vma=False,
        )
        return self.halo.unpack_owned(mapped(packed, parts))


__all__ = [
    "DistributedHaloPlan",
    "DistributedLocalOperator",
    "DistributedSparseTransfer",
]
