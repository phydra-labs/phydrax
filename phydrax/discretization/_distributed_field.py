#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._cell_mesh import CellMesh


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
    owner_local: bool = eqx.field(static=True)
    partition_index: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        entity_owner: ArrayLike | None,
        adjacency: ArrayLike | None,
        part_count: int,
        /,
        *,
        local_global_ids: ArrayLike | None = None,
        local_owned: ArrayLike | None = None,
        local_valid: ArrayLike | None = None,
        global_entity_count: int | None = None,
        partition_index: int | None = None,
        phase_send_indices: ArrayLike | None = None,
        phase_receive_indices: ArrayLike | None = None,
        phase_send_valid: ArrayLike | None = None,
        phase_receive_valid: ArrayLike | None = None,
        permutations: tuple[tuple[tuple[int, int], ...], ...] | None = None,
        evidence_id: str | None = None,
    ) -> None:
        if local_global_ids is not None:
            if entity_owner is not None or adjacency is not None:
                raise ValueError("Owner-local halos cannot capture global ownership.")
            self._initialize_owner_local(
                local_global_ids,
                local_owned,
                local_valid,
                global_entity_count,
                partition_index,
                part_count,
                phase_send_indices,
                phase_receive_indices,
                phase_send_valid,
                phase_receive_valid,
                permutations,
                evidence_id,
            )
            return
        if any(
            value is not None
            for value in (
                local_owned,
                global_entity_count,
                partition_index,
                phase_send_indices,
                phase_receive_indices,
                phase_send_valid,
                phase_receive_valid,
                permutations,
                evidence_id,
            )
        ):
            raise ValueError("Owner-local halo metadata requires local_global_ids.")
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
        self.owner_local = False
        self.partition_index = jnp.asarray(-1, dtype=jnp.int32)
        self.evidence_id = ""
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

    def _initialize_owner_local(
        self,
        local_global_ids: ArrayLike,
        local_owned: ArrayLike | None,
        local_valid: ArrayLike | None,
        global_entity_count: int | None,
        partition_index: int | None,
        part_count: int,
        phase_send_indices: ArrayLike | None,
        phase_receive_indices: ArrayLike | None,
        phase_send_valid: ArrayLike | None,
        phase_receive_valid: ArrayLike | None,
        permutations: tuple[tuple[tuple[int, int], ...], ...] | None,
        evidence_id: str | None,
        /,
    ) -> None:
        ids = np.asarray(local_global_ids)
        owned = np.asarray(local_owned, dtype=np.bool_)
        parts = int(part_count)
        valid = (
            np.ones(ids.shape, dtype=np.bool_)
            if local_valid is None
            else np.asarray(local_valid, dtype=np.bool_)
        )
        active = (
            ids[valid] if valid.shape == ids.shape else np.empty((0,), dtype=np.int64)
        )
        if (
            ids.ndim != 1
            or ids.size == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or valid.shape != ids.shape
            or not np.array_equal(valid, np.arange(ids.size) < np.count_nonzero(valid))
            or np.any(active < 0)
            or np.unique(active).size != active.size
            or not np.array_equal(ids[~valid], np.arange(-np.count_nonzero(~valid), 0))
            or owned.shape != ids.shape
            or np.any(owned & ~valid)
            or global_entity_count is None
            or int(global_entity_count) < active.size
            or partition_index is None
            or not 0 <= int(partition_index) < parts
            or evidence_id is None
            or not str(evidence_id).strip()
            or permutations is None
        ):
            raise ValueError("Owner-local halo identity and coverage are incomplete.")
        index = int(partition_index)
        phases = tuple(
            tuple((int(a), int(b)) for a, b in phase) for phase in permutations
        )
        send = np.asarray(phase_send_indices)
        receive = np.asarray(phase_receive_indices)
        send_valid = np.asarray(phase_send_valid, dtype=np.bool_)
        receive_valid = np.asarray(phase_receive_valid, dtype=np.bool_)
        if (
            send.ndim != 2
            or send.shape[0] != len(phases)
            or send.shape[1] < 1
            or receive.shape != send.shape
            or send_valid.shape != send.shape
            or receive_valid.shape != send.shape
            or not np.issubdtype(send.dtype, np.integer)
            or not np.issubdtype(receive.dtype, np.integer)
            or np.any(send[send_valid] < 0)
            or np.any(send[send_valid] >= ids.size)
            or np.any(receive[receive_valid] < 0)
            or np.any(receive[receive_valid] >= ids.size)
        ):
            raise ValueError("Owner-local halo phases require valid local index arrays.")
        for phase_index, phase in enumerate(phases):
            sources = [a for a, _ in phase]
            targets = [b for _, b in phase]
            if (
                len(set(sources)) != len(sources)
                or len(set(targets)) != len(targets)
                or any(
                    a == b or not 0 <= a < parts or not 0 <= b < parts for a, b in phase
                )
                or (np.any(send_valid[phase_index]) and index not in sources)
                or (np.any(receive_valid[phase_index]) and index not in targets)
                or np.unique(send[phase_index][send_valid[phase_index]]).size
                != np.count_nonzero(send_valid[phase_index])
            ):
                raise ValueError(
                    "Owner-local halo coloring or phase participation is invalid."
                )
        ghost_rows = receive[receive_valid]
        if (
            np.any(~owned[send[send_valid]])
            or np.any(owned[ghost_rows])
            or np.unique(ghost_rows).size != ghost_rows.size
            or not np.array_equal(np.sort(ghost_rows), np.flatnonzero(valid & ~owned))
        ):
            raise ValueError(
                "Halo routes must send owners and cover every ghost exactly once."
            )
        send = np.where(send_valid, send, 0).astype(np.int32)
        receive = np.where(receive_valid, receive, 0).astype(np.int32)
        self.entity_owner = jnp.empty((0,), dtype=jnp.int32)
        self.local_global_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.local_valid = jnp.asarray(valid)
        self.local_owned = jnp.asarray(owned, dtype=jnp.bool_)
        self.phase_send_indices = jnp.asarray(send, dtype=jnp.int32)
        self.phase_receive_indices = jnp.asarray(receive, dtype=jnp.int32)
        self.phase_send_valid = jnp.asarray(send_valid, dtype=jnp.bool_)
        self.phase_receive_valid = jnp.asarray(receive_valid, dtype=jnp.bool_)
        self.permutations = phases
        self.reverse_permutations = tuple(
            tuple((b, a) for a, b in phase) for phase in phases
        )
        self.entity_count = int(global_entity_count)
        self.part_count = parts
        self.local_capacity = ids.size
        self.message_capacity = send.shape[1]
        self.owner_local = True
        self.partition_index = jnp.asarray(index, dtype=jnp.int32)
        self.evidence_id = str(evidence_id)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "owner-local-distributed-halo",
                "ids": ids,
                "owned": owned,
                "send": send,
                "receive": receive,
                "local_valid": valid,
                "send_valid": send_valid,
                "receive_valid": receive_valid,
                "permutations": phases,
                "partition_index": index,
                "global_entity_count": self.entity_count,
                "part_count": parts,
                "evidence_id": self.evidence_id,
            }
        )

    def collective_certificate(self, /, *, axis_name: str) -> Array:
        """Collectively reconcile stable IDs, masks, coverage, and participants.

        The mesh evidence supplies the ownership proof. This certificate additionally
        checks the executable sparse routes; a claimed evidence ID alone never passes.
        No entity vectors or mesh arrays are gathered.
        """
        if not self.owner_local:
            raise ValueError(
                "Collective route certificates require owner-local metadata."
            )
        valid = (
            (jax.lax.axis_index(axis_name) == self.partition_index)
            & (
                jax.lax.psum(jnp.asarray(1, dtype=jnp.int32), axis_name)
                == self.part_count
            )
            & (
                jax.lax.psum(jnp.sum(self.local_owned, dtype=jnp.int64), axis_name)
                == self.entity_count
            )
        )
        agreement = canonical_fingerprint(
            {
                "permutations": self.permutations,
                "entity_count": self.entity_count,
                "part_count": self.part_count,
                "evidence_id": self.evidence_id,
            }
        )
        # Compare the full digest in bounded scalar metadata, not a global ID array.
        digest = jnp.asarray(
            [int(agreement[offset : offset + 7], 16) for offset in range(0, 64, 7)],
            dtype=jnp.int32,
        )
        valid = valid & jnp.all(
            jax.lax.pmin(digest, axis_name) == jax.lax.pmax(digest, axis_name)
        )
        for phase, permutation in enumerate(self.permutations):
            send_ids = jnp.where(
                self.phase_send_valid[phase],
                self.local_global_ids[self.phase_send_indices[phase]],
                -1,
            )
            received = jax.lax.ppermute(send_ids, axis_name, permutation)
            destination = jnp.asarray(False, dtype=jnp.bool_)
            for _, target in permutation:
                destination = destination | (self.partition_index == target)
            expected = jnp.where(
                self.phase_receive_valid[phase],
                self.local_global_ids[self.phase_receive_indices[phase]],
                -1,
            )
            valid = valid & (~destination | jnp.all(received == expected))
        return jax.lax.pmin(valid.astype(jnp.int32), axis_name).astype(jnp.bool_)

    def _phase_rows(self, table: Array, phase: int, part: Array, /) -> Array:
        return table[phase] if self.owner_local else table[phase, part]

    def _require_global_layout(self, /) -> None:
        if self.owner_local:
            raise ValueError(
                "Owner-local halos cannot pack or unpack global arrays; use an explicit bounded export."
            )

    def pack_owned(self, global_values: ArrayLike, /) -> Array:
        self._require_global_layout()
        values = jnp.asarray(global_values)
        if values.ndim == 0 or values.shape[0] != self.entity_count:
            raise ValueError("Global field leading axis must match distributed entities.")
        packed = values[self.local_global_ids]
        mask = self.local_owned.reshape(self.local_owned.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, packed, jnp.zeros_like(packed))

    def pack_reference(self, global_values: ArrayLike, /) -> Array:
        self._require_global_layout()
        values = jnp.asarray(global_values)
        if values.ndim == 0 or values.shape[0] != self.entity_count:
            raise ValueError("Global field leading axis must match distributed entities.")
        packed = values[self.local_global_ids]
        mask = self.local_valid.reshape(self.local_valid.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, packed, jnp.zeros_like(packed))

    def unpack_owned(self, local_values: ArrayLike, /) -> Array:
        self._require_global_layout()
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
            send_indices = self._phase_rows(self.phase_send_indices, phase, part)
            send_valid = self._phase_rows(self.phase_send_valid, phase, part)
            payload = values[send_indices]
            mask = send_valid.reshape(send_valid.shape + (1,) * (values.ndim - 1))
            payload = jnp.where(mask, payload, 0)
            received = jax.lax.ppermute(payload, axis_name=axis_name, perm=permutation)
            receive_indices = self._phase_rows(self.phase_receive_indices, phase, part)
            receive_valid = self._phase_rows(
                self.phase_receive_valid, phase, part
            ).reshape(send_valid.shape + (1,) * (values.ndim - 1))
            # Only destinations write, and only their valid message slots: padded
            # slots alias local row zero, which is always an owned entity.
            destination = jnp.any(
                jnp.asarray(
                    [target == part for _, target in permutation], dtype=jnp.bool_
                )
            )
            delta = jnp.where(receive_valid, received - values[receive_indices], 0)
            values = jax.lax.cond(
                destination,
                lambda current: current.at[receive_indices].add(delta),
                lambda current: current,
                values,
            )
        return values

    def accumulate_halo(
        self, local_values: Array, part: Array, /, *, axis_name: str
    ) -> Array:
        values = local_values
        for phase, permutation in reversed(tuple(enumerate(self.reverse_permutations))):
            receive_indices = self._phase_rows(self.phase_receive_indices, phase, part)
            source_valid = self._phase_rows(self.phase_receive_valid, phase, part)
            payload = values[receive_indices]
            mask = source_valid.reshape(source_valid.shape + (1,) * (values.ndim - 1))
            payload = jnp.where(mask, payload, 0)
            received = jax.lax.ppermute(payload, axis_name=axis_name, perm=permutation)
            send_indices = self._phase_rows(self.phase_send_indices, phase, part)
            destination = jnp.any(
                jnp.asarray(
                    [target == part for _, target in permutation], dtype=jnp.bool_
                )
            )
            target_valid = self._phase_rows(self.phase_send_valid, phase, part).reshape(
                source_valid.shape + (1,) * (values.ndim - 1)
            )
            values = jax.lax.cond(
                destination,
                lambda current: current.at[send_indices].add(
                    jnp.where(target_valid, received, 0)
                ),
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


def make_owner_local_field_array(
    local_values: Sequence[ArrayLike],
    partition_indices: Sequence[int],
    devices: Sequence[jax.Device],
    /,
    *,
    axis_name: str,
) -> Array:
    """Place uniform local buckets without materializing the global field on host."""
    assigned = tuple(devices)
    ranks = tuple(int(index) for index in partition_indices)
    values = tuple(jnp.asarray(value) for value in local_values)
    if any(not value.is_fully_addressable for value in values):
        raise ValueError(
            "Local field buckets must be addressable; global fields require explicit local shard selection."
        )
    if not assigned or len(set(assigned)) != len(assigned):
        raise ValueError(
            "Owner-local execution requires distinct declared actual devices."
        )
    expected = tuple(
        index
        for index, device in enumerate(assigned)
        if device.process_index == jax.process_index()
    )
    if ranks != expected or len(values) != len(ranks) or not values:
        raise ValueError(
            "Local buckets must cover exactly the process-addressable declared partitions."
        )
    shape = values[0].shape
    dtype = values[0].dtype
    if any(value.shape != shape or value.dtype != dtype for value in values):
        raise ValueError(
            "Owner-local execution requires uniform local shape/dtype buckets; padding is not implicit."
        )
    mesh = Mesh(np.asarray(assigned, dtype=object), (axis_name,))
    sharding = NamedSharding(mesh, PartitionSpec(axis_name))
    arrays = [
        jax.device_put(value[None, ...], assigned[rank])
        for rank, value in zip(ranks, values, strict=True)
    ]
    return jax.make_array_from_single_device_arrays(
        (len(assigned),) + shape, sharding, arrays
    )


def _exchange_owner_local_packets(
    packets: Array,
    permutations: tuple[tuple[tuple[int, int], ...], ...],
    mesh: Mesh,
    axis_name: str,
    /,
) -> Array:
    """Execute the shared bounded per-phase peer packet convention."""
    spec = PartitionSpec(axis_name)

    def exchange(local_packet: Array) -> Array:
        messages = tuple(
            jax.lax.ppermute(local_packet[0, phase], axis_name, permutation)
            for phase, permutation in enumerate(permutations)
        )
        result = (
            jnp.stack(messages)
            if messages
            else jnp.empty(local_packet.shape[1:], dtype=local_packet.dtype)
        )
        return result[None, ...]

    return jax.shard_map(
        exchange,
        mesh=mesh,
        in_specs=spec,
        out_specs=spec,
        check_vma=False,
    )(packets)


def prepare_owner_local_halo_plans(
    meshes: Sequence[CellMesh],
    entity_degree: int,
    /,
    *,
    devices: Sequence[jax.Device],
    axis_name: str,
    message_capacity: int,
) -> tuple[DistributedHaloPlan, ...]:
    """Derive sparse owner routes from actual bounded peer stable-ID requests.

    Only process-local closures are inspected. Ring-colored request packets are
    exchanged by real JAX permutations; received IDs are resolved against owned
    rows, never a complete global entity table. ``message_capacity`` is an
    explicit admission bound, not an allocation of that many entity rows.
    """
    local_meshes = tuple(meshes)
    assigned = tuple(devices)
    degree = int(entity_degree)
    capacity = int(message_capacity)
    axis = str(axis_name).strip()
    if not local_meshes or capacity < 1 or not axis:
        raise ValueError(
            "Halo preparation requires local meshes, a positive bound, and a named axis."
        )
    if any(
        not isinstance(mesh, CellMesh) or mesh.storage is None for mesh in local_meshes
    ):
        raise ValueError(
            "Halo preparation requires canonical owner-local CellMesh storage."
        )
    storages = tuple(mesh.storage for mesh in local_meshes)
    if any(
        storage is None
        or not 0 <= degree < len(storage.global_entity_counts)
        or storage.partition_count != len(assigned)
        for storage in storages
    ):
        raise ValueError(
            "Halo entity degree and declared placement must match canonical storage."
        )
    ranks = tuple(storage.partition_index for storage in storages if storage is not None)
    parts = len(assigned)
    request_permutations = tuple(
        tuple((rank, (rank + offset) % parts) for rank in range(parts))
        for offset in range(1, parts)
    )
    forward_permutations = tuple(
        tuple((target, source) for source, target in phase)
        for phase in request_permutations
    )
    requests_by_rank = []
    rows_by_rank = []
    maxima = []
    identities = []
    for storage in storages:
        if storage is None:
            raise ValueError("Owner-local storage is required.")
        ids = np.asarray(storage.entity_global_ids[degree], dtype=np.int64)
        owners = np.asarray(storage.entity_owner[degree], dtype=np.int32)
        phase_rows = []
        for offset in range(1, parts):
            owner = (storage.partition_index + offset) % parts
            rows = np.flatnonzero(owners == owner)
            rows = rows[np.argsort(ids[rows], kind="stable")].astype(np.int32)
            phase_rows.append(rows)
        rows_by_rank.append(tuple(phase_rows))
        maxima.append(
            jnp.asarray(max((row.size for row in phase_rows), default=0), dtype=jnp.int32)
        )
        identity = canonical_fingerprint(
            {
                "kind": "owner-local-halo-request",
                "degree": degree,
                "global_count": storage.global_entity_counts[degree],
                "topology": storage.logical_topology_id,
                "evidence": storage.evidence_id,
                "parts": parts,
                "message_bound": capacity,
            }
        )
        identities.append(
            jnp.asarray(
                [int(identity[offset : offset + 7], 16) for offset in range(0, 64, 7)],
                dtype=jnp.int32,
            )
        )
        requests_by_rank.append(ids)
    mesh = Mesh(np.asarray(assigned, dtype=object), (axis,))
    spec = PartitionSpec(axis)
    maximum_array = make_owner_local_field_array(maxima, ranks, assigned, axis_name=axis)
    identity_array = make_owner_local_field_array(
        identities, ranks, assigned, axis_name=axis
    )

    def admit(maximum: Array, identity: Array) -> tuple[Array, Array]:
        count = jax.lax.pmax(maximum[0], axis)
        agreed = jnp.all(
            jax.lax.pmin(identity[0], axis) == jax.lax.pmax(identity[0], axis)
        )
        return count, agreed

    required, agreed = jax.shard_map(
        admit,
        mesh=mesh,
        in_specs=(spec, spec),
        out_specs=(PartitionSpec(), PartitionSpec()),
        check_vma=False,
    )(maximum_array, identity_array)
    if not bool(np.asarray(agreed)) or int(np.asarray(required)) > capacity:
        raise ValueError("Collective halo identity or bounded message admission failed.")
    width = max(int(np.asarray(required)), 1)
    packets = []
    receives = []
    receives_valid = []
    for ids, phase_rows in zip(requests_by_rank, rows_by_rank, strict=True):
        packet = np.full((parts - 1, width), -1, dtype=np.int64)
        receive = np.zeros((parts - 1, width), dtype=np.int32)
        valid = np.zeros((parts - 1, width), dtype=np.bool_)
        for phase, rows in enumerate(phase_rows):
            packet[phase, : rows.size] = ids[rows]
            receive[phase, : rows.size] = rows
            valid[phase, : rows.size] = True
        packets.append(jnp.asarray(packet, dtype=jnp.int64))
        receives.append(receive)
        receives_valid.append(valid)
    packet_array = make_owner_local_field_array(packets, ranks, assigned, axis_name=axis)

    incoming = _exchange_owner_local_packets(
        packet_array,
        request_permutations,
        mesh,
        axis,
    )
    local_incoming = {
        (0 if shard.index[0].start is None else int(shard.index[0].start)): np.asarray(
            shard.data
        )[0]
        for shard in incoming.addressable_shards
    }
    plans = []
    missing = []
    for storage, receive, receive_valid in zip(
        storages, receives, receives_valid, strict=True
    ):
        if storage is None:
            raise ValueError("Owner-local storage is required.")
        ids = np.asarray(storage.entity_global_ids[degree], dtype=np.int64)
        owned = np.asarray(storage.entity_owned[degree], dtype=np.bool_)
        messages = local_incoming[storage.partition_index]
        lookup = {
            int(identifier): row for row, identifier in enumerate(ids) if owned[row]
        }
        send = np.zeros(messages.shape, dtype=np.int32)
        send_valid = messages >= 0
        absent = False
        for phase, column in zip(*np.nonzero(send_valid), strict=True):
            row = lookup.get(int(messages[phase, column]))
            if row is None:
                absent = True
            else:
                send[phase, column] = row
        missing.append(jnp.asarray(absent, dtype=jnp.bool_))
        plans.append((storage, ids, owned, send, send_valid, receive, receive_valid))
    missing_array = make_owner_local_field_array(missing, ranks, assigned, axis_name=axis)
    failed = jax.shard_map(
        lambda value: jax.lax.pmax(value[0].astype(jnp.int32), axis),
        mesh=mesh,
        in_specs=spec,
        out_specs=PartitionSpec(),
        check_vma=False,
    )(missing_array)
    if bool(np.asarray(failed)):
        raise ValueError(
            "A peer requested a stable ID absent from its declared owner's local closure."
        )
    return tuple(
        DistributedHaloPlan(
            None,
            None,
            parts,
            local_global_ids=ids,
            local_owned=owned,
            global_entity_count=storage.global_entity_counts[degree],
            partition_index=storage.partition_index,
            phase_send_indices=send,
            phase_receive_indices=receive,
            phase_send_valid=send_valid,
            phase_receive_valid=receive_valid,
            permutations=forward_permutations,
            evidence_id=storage.evidence_id,
        )
        for storage, ids, owned, send, send_valid, receive, receive_valid in plans
    )


def resolve_owner_local_field_values(
    source_meshes: Sequence[CellMesh],
    source_values: Sequence[ArrayLike],
    query_global_ids: Sequence[ArrayLike],
    query_owners: Sequence[ArrayLike],
    entity_degree: int,
    /,
    *,
    source_topology_id: str,
    devices: Sequence[jax.Device],
    axis_name: str,
    message_capacity: int,
) -> tuple[Array, ...]:
    """Resolve actual carried fields in exact predecessor-transfer query ID order.

    ``query_owners`` must come from the accepted immediate predecessor ownership
    proof. Values are read only from authoritative owned predecessor rows, never
    ghost copies, complete global field arrays, or an analytic seed. Queries and
    responses use the same bounded peer packet execution as canonical halos.
    Returned arrays remain on assigned local devices and may have different query
    counts; only communication packets are padded.
    """
    meshes = tuple(source_meshes)
    values = tuple(jnp.asarray(value) for value in source_values)
    if any(not value.is_fully_addressable for value in values) or any(
        isinstance(value, Array) and not value.is_fully_addressable
        for group in (query_global_ids, query_owners)
        for value in group
    ):
        raise ValueError(
            "Predecessor query IDs, owners, and values must be process-local addressable arrays."
        )
    query_ids = tuple(np.asarray(identifiers) for identifiers in query_global_ids)
    owners = tuple(np.asarray(placement) for placement in query_owners)
    assigned = tuple(devices)
    degree = int(entity_degree)
    bound = int(message_capacity)
    axis = str(axis_name).strip()
    if (
        not meshes
        or bound < 1
        or not axis
        or len(values) != len(meshes)
        or len(query_ids) != len(meshes)
        or len(owners) != len(meshes)
    ):
        raise ValueError(
            "Field resolution requires aligned local sources, queries, and a positive peer bound."
        )
    if any(
        not isinstance(mesh, CellMesh)
        or mesh.storage is None
        or not 0 <= degree < len(mesh.storage.global_entity_counts)
        or mesh.storage.partition_count != len(assigned)
        or mesh.storage.logical_topology_id != source_topology_id
        for mesh in meshes
    ):
        raise ValueError(
            "Field queries must bind the actual accepted predecessor topology and placement."
        )
    storages = tuple(mesh.storage for mesh in meshes)
    ranks = tuple(storage.partition_index for storage in storages if storage is not None)
    values = tuple(
        jax.device_put(value, assigned[rank])
        for rank, value in zip(ranks, values, strict=True)
    )
    parts = len(assigned)
    requests = tuple(
        tuple((rank, (rank + offset) % parts) for rank in range(parts))
        for offset in range(1, parts)
    )
    responses = tuple(
        tuple((target, source) for source, target in phase) for phase in requests
    )
    component_shape = values[0].shape[1:]
    dtype = values[0].dtype
    lookups = []
    query_rows = []
    own_rows = []
    own_source_rows = []
    maxima = []
    identities = []
    local_missing = []
    for storage, field, ids, placement in zip(
        storages, values, query_ids, owners, strict=True
    ):
        if storage is None:
            raise ValueError("Accepted predecessor storage is required.")
        source_ids = np.asarray(storage.entity_global_ids[degree], dtype=np.int64)
        owned = np.asarray(storage.entity_owned[degree], dtype=np.bool_)
        if (
            not field.is_fully_addressable
            or field.shape != (source_ids.size,) + component_shape
            or field.dtype != dtype
            or ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids > np.iinfo(np.int64).max)
            or placement.shape != ids.shape
            or placement.dtype.kind not in "iu"
            or np.any(placement < 0)
            or np.any(placement >= parts)
        ):
            raise ValueError(
                "Local field components and accepted query IDs/owners are incompatible."
            )
        lookup = {
            int(identifier): row
            for row, identifier in enumerate(source_ids)
            if owned[row]
        }
        lookups.append(lookup)
        phase_rows = tuple(
            np.flatnonzero(
                placement == (storage.partition_index + offset) % parts
            ).astype(np.int32)
            for offset in range(1, parts)
        )
        query_rows.append(phase_rows)
        own = np.flatnonzero(placement == storage.partition_index).astype(np.int32)
        source_rows = np.asarray(
            [lookup.get(int(ids[row]), -1) for row in own], dtype=np.int32
        )
        own_rows.append(own)
        own_source_rows.append(source_rows)
        local_missing.append(bool(np.any(source_rows < 0)))
        maxima.append(
            jnp.asarray(max((row.size for row in phase_rows), default=0), dtype=jnp.int32)
        )
        identity = canonical_fingerprint(
            {
                "kind": "accepted-owner-local-field-query",
                "topology": source_topology_id,
                "degree": degree,
                "count": storage.global_entity_counts[degree],
                "evidence": storage.evidence_id,
                "parts": parts,
                "component_shape": component_shape,
                "dtype": str(dtype),
                "message_bound": bound,
            }
        )
        identities.append(
            jnp.asarray(
                [int(identity[offset : offset + 7], 16) for offset in range(0, 64, 7)],
                dtype=jnp.int32,
            )
        )
    mesh = Mesh(np.asarray(assigned, dtype=object), (axis,))
    spec = PartitionSpec(axis)
    maximum_array = make_owner_local_field_array(maxima, ranks, assigned, axis_name=axis)
    identity_array = make_owner_local_field_array(
        identities, ranks, assigned, axis_name=axis
    )

    def admit(maximum: Array, identity: Array) -> tuple[Array, Array]:
        return (
            jax.lax.pmax(maximum[0], axis),
            jnp.all(jax.lax.pmin(identity[0], axis) == jax.lax.pmax(identity[0], axis)),
        )

    required, agreed = jax.shard_map(
        admit,
        mesh=mesh,
        in_specs=(spec, spec),
        out_specs=(PartitionSpec(), PartitionSpec()),
        check_vma=False,
    )(maximum_array, identity_array)
    if not bool(np.asarray(agreed)) or int(np.asarray(required)) > bound:
        raise ValueError(
            "Collective predecessor field identity or bounded query admission failed."
        )
    width = max(int(np.asarray(required)), 1)
    packets = []
    for ids, phase_rows in zip(query_ids, query_rows, strict=True):
        packet = np.full((parts - 1, width), -1, dtype=np.int64)
        for phase, rows in enumerate(phase_rows):
            packet[phase, : rows.size] = ids[rows]
        packets.append(jnp.asarray(packet, dtype=jnp.int64))
    incoming = _exchange_owner_local_packets(
        make_owner_local_field_array(packets, ranks, assigned, axis_name=axis),
        requests,
        mesh,
        axis,
    )
    local_messages = {
        int(shard.index[0].start): np.asarray(shard.data)[0]
        for shard in incoming.addressable_shards
    }
    response_values = []
    missing = []
    for rank, field, lookup, absent in zip(
        ranks, values, lookups, local_missing, strict=True
    ):
        messages = local_messages[rank]
        valid = messages >= 0
        rows = np.zeros(messages.shape, dtype=np.int32)
        for phase, column in zip(*np.nonzero(valid), strict=True):
            row = lookup.get(int(messages[phase, column]))
            if row is None:
                absent = True
            else:
                rows[phase, column] = row
        missing.append(jnp.asarray(absent, dtype=jnp.bool_))
        mask = jnp.asarray(valid).reshape(valid.shape + (1,) * len(component_shape))
        response_values.append(
            jnp.where(mask, field[jnp.asarray(rows, dtype=jnp.int32)], 0)
        )
    failed = jax.shard_map(
        lambda value: jax.lax.pmax(value[0].astype(jnp.int32), axis),
        mesh=mesh,
        in_specs=spec,
        out_specs=PartitionSpec(),
        check_vma=False,
    )(make_owner_local_field_array(missing, ranks, assigned, axis_name=axis))
    if bool(np.asarray(failed)):
        raise ValueError(
            "A transfer query ID is absent from its accepted authoritative owner closure."
        )
    returned = _exchange_owner_local_packets(
        make_owner_local_field_array(response_values, ranks, assigned, axis_name=axis),
        responses,
        mesh,
        axis,
    )
    local_responses = {
        int(shard.index[0].start): shard.data[0] for shard in returned.addressable_shards
    }
    results = []
    for rank, field, ids, phase_rows, own, source_rows in zip(
        ranks,
        values,
        query_ids,
        query_rows,
        own_rows,
        own_source_rows,
        strict=True,
    ):
        result = jax.device_put(
            jnp.zeros(ids.shape + component_shape, dtype=dtype),
            assigned[rank],
        )
        result = result.at[jnp.asarray(own, dtype=jnp.int32)].set(
            field[jnp.asarray(source_rows, dtype=jnp.int32)]
        )
        for phase, rows in enumerate(phase_rows):
            result = result.at[jnp.asarray(rows, dtype=jnp.int32)].set(
                local_responses[rank][phase, : rows.size]
            )
        results.append(result)
    return tuple(results)


__all__ = [
    "DistributedHaloPlan",
    "DistributedLocalOperator",
    "make_owner_local_field_array",
    "prepare_owner_local_halo_plans",
    "resolve_owner_local_field_values",
]
