#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike

from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class DistributedHaloPlan(StrictModule, NonTrainableState):
    """Padded owned/halo layout and colored peer permutations.

    Every communication phase has at most one outgoing and incoming peer per
    partition, so ``lax.ppermute`` routes fixed-size packed messages without an
    all-gather. Canonical global IDs determine ownership and transpose routing.
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

    def __init__(self, entity_owner: ArrayLike, adjacency: ArrayLike, part_count: int, /):
        owner = np.asarray(entity_owner)
        pairs = np.asarray(adjacency)
        parts = int(part_count)
        if (
            owner.ndim != 1
            or owner.size == 0
            or not np.issubdtype(owner.dtype, np.integer)
            or parts <= 0
            or np.any(owner < 0)
            or np.any(owner >= parts)
            or np.unique(owner).size != parts
        ):
            raise ValueError("Distributed entity ownership must cover every partition.")
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
            flags = np.asarray([value in owned for value in ids], dtype=bool)
            local_ids.append(ids)
            local_owned.append(flags)
            maps.append({int(value): index for index, value in enumerate(ids)})
        capacity = max(values.size for values in local_ids)
        global_ids = np.zeros((parts, capacity), dtype=np.int32)
        valid = np.zeros((parts, capacity), dtype=bool)
        owned_mask = np.zeros((parts, capacity), dtype=bool)
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
        remaining = dict(pair_messages)
        phases: list[list[tuple[int, int]]] = []
        while remaining:
            phase: list[tuple[int, int]] = []
            used_source: set[int] = set()
            used_target: set[int] = set()
            for pair in sorted(remaining):
                source, target = pair
                if source not in used_source and target not in used_target:
                    phase.append(pair)
                    used_source.add(source)
                    used_target.add(target)
            phases.append(phase)
            for pair in phase:
                del remaining[pair]
        message_capacity = max(
            (len(message) for message in pair_messages.values()), default=1
        )
        send = np.zeros((len(phases), parts, message_capacity), dtype=np.int32)
        receive = np.zeros_like(send)
        send_valid = np.zeros_like(send, dtype=bool)
        receive_valid = np.zeros_like(send, dtype=bool)
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
            # Only destinations write; sources with no incoming permutation receive zeros.
            destination = jnp.any(
                jnp.asarray([target == part for _, target in permutation], dtype=bool)
            )
            values = jax.lax.cond(
                destination,
                lambda current: current.at[receive_indices].set(received),
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
                jnp.asarray([target == part for _, target in permutation], dtype=bool)
            )
            values = jax.lax.cond(
                destination,
                lambda current: current.at[send_indices].add(received),
                lambda current: current,
                values,
            )
        return values


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

    def __init__(
        self,
        halo: DistributedHaloPlan,
        local_action: Callable[[Array, Array, Array, Array, Array], Array],
        local_transpose: Callable[[Array, Array, Array, Array, Array], Array],
        /,
        *,
        operator_name: str,
    ):
        if not isinstance(halo, DistributedHaloPlan):
            raise TypeError("Distributed local operator requires a halo plan.")
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

        def action(local, part):
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

        def action(local, part):
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


__all__ = ["DistributedHaloPlan", "DistributedLocalOperator"]
