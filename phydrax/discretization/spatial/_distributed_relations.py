#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Arbitrary-owner point layouts, shell-certified queries, halos, and packets.

Ownership is explicit. Every active point carries one stable global ID, one
owner along a one-dimensional execution-group mesh axis, and one owner-local
slot; partitions may be arbitrarily uneven up to the declared per-owner
capacity. Queries never replicate targets. Every owner publishes the bounding
box and population of its sources; each target first searches its own owner,
derives a certified search radius from that local result and the published
owner populations, and then contacts only the owners whose (periodic) boxes
intersect that ball. A contacted owner answers only with its sources inside
the ball, so its candidate buffer is bounded by the ball's stencil cells, not
by an unbounded nearest selection for a target outside its region. Remote
candidates return through bounded all-to-all packets and merge
deterministically by ``(distance, stable_id, owner)``.

Capacity, halo, owner, and identity failures are per-target statuses rather
than silently truncated relations. Forced CPU devices exercise the collective
semantics only; they are not performance evidence for accelerators.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from ..._execution_runtime import ExecutionGroup
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...backends.distributed import JaxCollectiveProvider
from ._morton import canonical_morton_order, MortonAddressPlan
from ._neighbor_query import (
    _minimum_image,
    _rounding_margin,
    MortonNeighborQueryPlan,
    MortonNeighborQueryStatus,
    SpatialDistanceBackend,
)


class DistributedRelationStatus(IntEnum):
    """Per-target outcome of a distributed shell query.

    Values ``0``-``5`` coincide with :class:`MortonNeighborQueryStatus` and
    keep their meaning when reported by the target's own or a remote owner.
    """

    COMPLETE = 0
    INACTIVE_TARGET = 1
    INVALID_TARGET = 2
    INVALID_SOURCES = 3
    CANDIDATE_OVERFLOW = 4
    UNCERTIFIED = 5
    MISSING_OWNER = 6
    OWNER_OVERFLOW = 7
    HALO_OVERFLOW = 8
    ROW_OVERFLOW = 9


_STATUS_COUNT = len(DistributedRelationStatus)


def _spec(axis: str, ndim: int) -> PartitionSpec:
    return PartitionSpec(axis, *([None] * (ndim - 1)))


def _static_positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _require_x64() -> None:
    """Refuse runtimes that cannot represent 64-bit Morton codes and IDs.

    Morton addresses are ``uint64`` and stable IDs ``int64``. With JAX x64
    disabled they would be silently truncated to 32 bits, changing neighbor
    identity. Float32 coordinates are supported with x64 enabled.
    """
    if not bool(jax.config.read("jax_enable_x64")):
        raise ValueError(
            "Distributed point relations require jax_enable_x64=True for uint64 "
            "Morton codes and int64 stable IDs; float32 coordinates are "
            "supported with x64 enabled."
        )


def _bucket_ranks(destination: Array, valid: Array, buckets: int) -> tuple[Array, Array]:
    """Stable rank of every valid item inside its destination bucket, and loads."""
    count = destination.shape[0]
    key = jnp.where(valid, destination, buckets).astype(jnp.int32)
    order = jnp.argsort(key, stable=True)
    ordered = key[order]
    first = jnp.searchsorted(ordered, ordered, side="left").astype(jnp.int32)
    rank = (
        jnp.zeros((count,), dtype=jnp.int32)
        .at[order]
        .set(jnp.arange(count, dtype=jnp.int32) - first)
    )
    loads = jax.ops.segment_sum(
        valid.astype(jnp.int32), jnp.where(valid, destination, 0), buckets
    )
    return rank, loads


def _pack(
    values: Array,
    destination: Array,
    rank: Array,
    fits: Array,
    buckets: int,
    capacity: int,
    fill: ArrayLike,
) -> Array:
    """Scatter items into a ``(buckets, capacity, ...)`` packet buffer."""
    sentinel = buckets * capacity
    index = jnp.where(fits, destination * capacity + rank, sentinel)
    buffer = jnp.full((sentinel + 1,) + values.shape[1:], fill, dtype=values.dtype)
    buffer = buffer.at[index].set(values)
    return buffer[:sentinel].reshape((buckets, capacity) + values.shape[1:])


def _owner_box_distances(
    points: Array,
    lower: Array,
    upper: Array,
    address_plan: MortonAddressPlan,
) -> tuple[Array, Array]:
    """Lower and upper bounds of target-to-owner-box minimum-image distances.

    The working set is ``targets x owners x 3 images x dimension``: bounded by
    the owner count, never by the global point count.
    """
    dtype = points.dtype
    domain_lower = jnp.asarray(address_plan.lower, dtype=dtype)
    lengths = jnp.asarray(address_plan.upper, dtype=dtype) - domain_lower
    periodic = jnp.asarray(address_plan.periodic_axes, dtype=jnp.bool_)
    shifts = jnp.asarray((-1.0, 0.0, 1.0), dtype=dtype)[:, None] * jnp.where(
        periodic, lengths, 0
    )
    images = points[:, None, None, :] + shifts[None, None, :, :]
    box_lower = lower[None, :, None, :]
    box_upper = upper[None, :, None, :]
    gap = jnp.maximum(jnp.maximum(box_lower - images, images - box_upper), 0)
    spread = jnp.maximum(jnp.abs(images - box_lower), jnp.abs(images - box_upper))
    near_axis = jnp.min(gap, axis=2)
    far_axis = jnp.min(spread, axis=2)
    far_axis = jnp.where(periodic, jnp.minimum(far_axis, 0.5 * lengths), far_axis)
    near = jnp.sqrt(jnp.sum(near_axis * near_axis, axis=-1))
    far = jnp.sqrt(jnp.sum(far_axis * far_axis, axis=-1))
    return near, far


@final
class DistributedOwnershipPlan(StrictModule):
    """Owner axis, per-owner slot capacity, and owner-to-process map.

    Owner ``r`` is device ``r`` of a one-dimensional execution-group mesh; its
    process is recorded in ``owner_processes``. Point arrays are owner-blocked:
    rows ``[r * local_capacity, (r + 1) * local_capacity)`` belong to owner
    ``r``.
    """

    address_plan: MortonAddressPlan
    execution_group: ExecutionGroup = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    local_capacity: int = eqx.field(static=True)
    owner_processes: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        execution_group: ExecutionGroup,
        local_capacity: int,
        /,
    ) -> None:
        if not isinstance(address_plan, MortonAddressPlan):
            raise TypeError("address_plan must be a MortonAddressPlan.")
        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be a live ExecutionGroup.")
        axes = tuple(execution_group.mesh.axis_names)
        if len(axes) != 1:
            raise ValueError(
                "Distributed point ownership requires a one-dimensional owner mesh."
            )
        capacity = _static_positive(local_capacity, "local_capacity")
        _require_x64()
        owners = len(execution_group.devices)
        processes = tuple(device.process_index for device in execution_group.devices)
        self.address_plan = address_plan
        self.execution_group = execution_group
        self.axis_name = str(axes[0])
        self.owner_count = owners
        self.local_capacity = capacity
        self.owner_processes = processes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-ownership-plan",
                "address_plan_id": address_plan.plan_id,
                "group_id": execution_group.spec.group_id,
                "axis_name": self.axis_name,
                "owner_count": owners,
                "local_capacity": capacity,
                "owner_processes": list(processes),
            }
        )

    @property
    def total_capacity(self) -> int:
        return self.owner_count * self.local_capacity

    def sharding(self, ndim: int, /) -> NamedSharding:
        """Owner-blocked sharding of an array whose leading axis is owner-major."""
        return self.execution_group.named_sharding(_spec(self.axis_name, ndim))

    def place(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        return jax.device_put(array, self.sharding(array.ndim))

    def map(
        self,
        function: Callable[..., Any],
        in_specs: Any,
        out_specs: Any,
    ) -> Callable[..., Any]:
        """Bind ``function`` to one owner per device along the ownership axis."""
        return jax.shard_map(
            function,
            mesh=self.execution_group.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )

    def compatible(self, other: DistributedOwnershipPlan, /) -> bool:
        """Whether two layouts share devices, axis, and addressing."""
        return (
            self.execution_group.spec.group_id == other.execution_group.spec.group_id
            and self.axis_name == other.axis_name
            and self.address_plan.plan_id == other.address_plan.plan_id
        )


class DistributedMigrationEvidence(NonTrainableState, StrictModule):
    """Packet, capacity, and epoch evidence of one atomic ownership transfer."""

    committed: Array
    invalid_destinations: Array
    maximum_packet: Array
    packet_capacity: Array
    maximum_received: Array
    local_capacity: Array
    migrated: Array
    epochs_consistent: Array
    epoch_before: Array
    epoch_after: Array


@final
class DistributedPointLayout(NonTrainableState, StrictModule):
    """Owner-blocked point rows with stable IDs, logical rows, and owner epochs.

    ``owner_epochs`` holds one epoch per owner. Owners whose epoch lags the
    newest one have not committed the latest ownership transfer; queries
    report every target as ``MISSING_OWNER`` while any owner is stale.
    """

    plan: DistributedOwnershipPlan
    points: Array
    stable_ids: Array
    active: Array
    logical_indices: Array
    owner_epochs: Array
    stable_ids_unique: Array
    logical_count: int = eqx.field(static=True)

    def __init__(
        self,
        plan: DistributedOwnershipPlan,
        points: ArrayLike,
        stable_ids: ArrayLike,
        active: ArrayLike,
        logical_indices: ArrayLike,
        owner_epochs: ArrayLike,
        stable_ids_unique: ArrayLike,
        logical_count: int,
        /,
    ) -> None:
        if not isinstance(plan, DistributedOwnershipPlan):
            raise TypeError("plan must be a DistributedOwnershipPlan.")
        rows = plan.total_capacity
        dimension = plan.address_plan.dimension
        coordinates = jnp.asarray(points)
        identifiers = jnp.asarray(stable_ids)
        mask = jnp.asarray(active)
        logical = jnp.asarray(logical_indices)
        epochs = jnp.asarray(owner_epochs)
        unique = jnp.asarray(stable_ids_unique)
        if coordinates.shape != (rows, dimension) or not jnp.issubdtype(
            coordinates.dtype, jnp.floating
        ):
            raise ValueError(f"points must be floating with shape {(rows, dimension)}.")
        if identifiers.shape != (rows,) or not jnp.issubdtype(
            identifiers.dtype, jnp.integer
        ):
            raise ValueError("stable_ids must be one integer per owner slot.")
        if mask.shape != (rows,) or mask.dtype != jnp.bool_:
            raise ValueError("active must be one boolean per owner slot.")
        if logical.shape != (rows,) or logical.dtype != jnp.int32:
            raise ValueError("logical_indices must be one int32 per owner slot.")
        if epochs.shape != (plan.owner_count,) or not jnp.issubdtype(
            epochs.dtype, jnp.integer
        ):
            raise ValueError("owner_epochs must hold one integer per owner.")
        if unique.shape != () or unique.dtype != jnp.bool_:
            raise ValueError("stable_ids_unique must be a boolean scalar.")
        count = _static_positive(logical_count, "logical_count")
        self.plan = plan
        self.points = plan.place(coordinates)
        self.stable_ids = plan.place(identifiers)
        self.active = plan.place(mask)
        self.logical_indices = plan.place(logical)
        self.owner_epochs = plan.place(epochs)
        self.stable_ids_unique = unique
        self.logical_count = count

    @classmethod
    def from_global(
        cls,
        plan: DistributedOwnershipPlan,
        points: ArrayLike,
        owners: ArrayLike,
        /,
        *,
        stable_ids: ArrayLike | None = None,
        epoch: int = 0,
    ) -> DistributedPointLayout:
        """Host-prepare an arbitrary uneven ownership in owner-local Morton order.

        Ingress validates the owner map, per-owner capacity, and global stable
        ID uniqueness once on the host and refuses violations.
        """
        coordinates = np.asarray(points)
        owner = np.asarray(owners)
        count = coordinates.shape[0]
        dimension = plan.address_plan.dimension
        if coordinates.ndim != 2 or coordinates.shape[1] != dimension:
            raise ValueError(f"points must have shape (N, {dimension}).")
        if not np.issubdtype(coordinates.dtype, np.floating):
            raise TypeError("points must have a floating dtype.")
        if owner.shape != (count,) or not np.issubdtype(owner.dtype, np.integer):
            raise ValueError("owners must hold one integer owner per point.")
        if count == 0 or np.any(owner < 0) or np.any(owner >= plan.owner_count):
            raise ValueError("Every owner must lie in [0, owner_count).")
        identifiers = (
            np.arange(count, dtype=np.int64)
            if stable_ids is None
            else np.asarray(stable_ids)
        )
        if identifiers.shape != (count,) or not np.issubdtype(
            identifiers.dtype, np.integer
        ):
            raise ValueError("stable_ids must hold one integer per point.")
        if np.unique(identifiers).size != count:
            raise ValueError("Stable global IDs must be unique.")
        loads = np.bincount(owner, minlength=plan.owner_count)
        if np.max(loads) > plan.local_capacity:
            raise ValueError(
                f"Owner load {int(np.max(loads))} exceeds local_capacity "
                f"{plan.local_capacity}."
            )
        codes = np.asarray(plan.address_plan.encode(jnp.asarray(coordinates)).codes)
        order = np.lexsort((identifiers, codes, owner))
        ordered_owner = owner[order]
        starts = np.concatenate(([0], np.cumsum(loads)[:-1]))
        slots = np.arange(count) - starts[ordered_owner]
        rows = ordered_owner * plan.local_capacity + slots
        total = plan.total_capacity
        blocked = np.zeros((total, dimension), dtype=coordinates.dtype)
        blocked[rows] = coordinates[order]
        blocked_ids = np.zeros((total,), dtype=identifiers.dtype)
        blocked_ids[rows] = identifiers[order]
        blocked_active = np.zeros((total,), dtype=np.bool_)
        blocked_active[rows] = True
        blocked_logical = np.full((total,), -1, dtype=np.int32)
        blocked_logical[rows] = order.astype(np.int32)
        return cls(
            plan,
            blocked,
            blocked_ids,
            blocked_active,
            blocked_logical,
            np.full((plan.owner_count,), int(epoch), dtype=np.int64),
            np.asarray(True),
            count,
        )

    @classmethod
    def from_blocks(
        cls,
        plan: DistributedOwnershipPlan,
        points: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
        stable_ids: ArrayLike | None = None,
        logical_indices: ArrayLike | None = None,
        logical_count: int | None = None,
        epoch: int = 0,
    ) -> DistributedPointLayout:
        """Adopt already owner-blocked device rows in their given owner order.

        Global stable-ID uniqueness is audited on device by hashing every ID
        to one bucket owner through a bounded all-to-all exchange.
        """
        total = plan.total_capacity
        coordinates = jnp.asarray(points)
        mask = (
            jnp.ones((total,), dtype=jnp.bool_)
            if active is None
            else jnp.asarray(active, dtype=jnp.bool_)
        )
        identifiers = (
            jnp.arange(total, dtype=jnp.int64)
            if stable_ids is None
            else jnp.asarray(stable_ids)
        )
        logical = (
            jnp.arange(total, dtype=jnp.int32)
            if logical_indices is None
            else jnp.asarray(logical_indices)
        )
        if logical.shape != (total,) or not jnp.issubdtype(logical.dtype, jnp.integer):
            raise ValueError("logical_indices must hold one integer per owner slot.")
        if mask.shape != (total,):
            raise ValueError("active must hold one boolean per owner slot.")
        if identifiers.shape != (total,) or not jnp.issubdtype(
            identifiers.dtype, jnp.integer
        ):
            raise ValueError("stable_ids must hold one integer per owner slot.")
        logical = jnp.where(mask, logical, -1).astype(jnp.int32)
        identifiers = plan.place(identifiers)
        mask = plan.place(mask)
        unique = _audit_unique_ids(plan, identifiers, mask)
        return cls(
            plan,
            coordinates,
            identifiers,
            mask,
            logical,
            jnp.full((plan.owner_count,), epoch, dtype=jnp.int64),
            unique,
            total if logical_count is None else logical_count,
        )

    @property
    def slot_owners(self) -> Array:
        """Owner index of every owner-blocked row."""
        return jnp.repeat(
            jnp.arange(self.plan.owner_count, dtype=jnp.int32), self.plan.local_capacity
        )

    def distribute(self, values: ArrayLike, /) -> Array:
        """Place logical-order rows into this owner-blocked slot layout."""
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.logical_count:
            raise ValueError("Values must begin with the logical point axis.")
        safe = jnp.where(self.active, self.logical_indices, 0)
        mask = self.active.reshape(self.active.shape + (1,) * (array.ndim - 1))
        blocked = jnp.where(mask, array[safe], jnp.zeros((), dtype=array.dtype))
        return self.plan.place(blocked)

    def collect(self, values: ArrayLike, /) -> Array:
        """Return owner-blocked rows in logical order (an explicit egress)."""
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.plan.total_capacity:
            raise ValueError("Values must begin with the owner-blocked slot axis.")
        index = jnp.where(self.active, self.logical_indices, self.logical_count)
        output = jnp.zeros((self.logical_count,) + array.shape[1:], dtype=array.dtype)
        return output.at[index].set(array, mode="drop")

    def host_addresses(self) -> tuple[np.ndarray, np.ndarray]:
        """Host map from logical row to ``(owner, slot)``; ``-1`` when absent.

        This is an explicit preparation boundary for host-built bindings.
        """
        logical = np.asarray(jax.device_get(self.logical_indices))
        active = np.asarray(jax.device_get(self.active))
        rows = np.flatnonzero(active)
        owners = np.full((self.logical_count,), -1, dtype=np.int32)
        slots = np.full((self.logical_count,), -1, dtype=np.int32)
        owners[logical[rows]] = (rows // self.plan.local_capacity).astype(np.int32)
        slots[logical[rows]] = (rows % self.plan.local_capacity).astype(np.int32)
        return owners, slots

    def partition_fingerprint(self) -> str:
        """Host identity of the current partition (plan, IDs, owners, epochs)."""
        active = np.asarray(jax.device_get(self.active))
        rows = np.flatnonzero(active)
        identifiers = np.asarray(jax.device_get(self.stable_ids))[rows]
        order = np.argsort(identifiers, kind="stable")
        return canonical_fingerprint(
            {
                "kind": "distributed-point-partition",
                "plan_id": self.plan.plan_id,
                "stable_ids": identifiers[order],
                "owners": (rows // self.plan.local_capacity)[order],
                "owner_epochs": np.asarray(jax.device_get(self.owner_epochs)),
            }
        )

    def migrate(
        self,
        destinations: ArrayLike,
        /,
        *,
        packet_capacity: int,
        points: ArrayLike | None = None,
        payload: Any = None,
    ) -> DistributedMigrationResult:
        """Move every active row exactly once to its destination owner.

        Packets carry stable IDs, logical rows, coordinates, payload leaves, and
        the sender epoch. The transfer commits atomically: any invalid
        destination, packet overflow, receive overflow, or epoch inconsistency
        on any owner rolls every owner back to its previous rows and payload.
        Committed owners compact received rows in owner-local Morton order and
        advance their epoch by one.
        """
        capacity = _static_positive(packet_capacity, "packet_capacity")
        destination = jnp.asarray(destinations)
        if destination.shape != (self.plan.total_capacity,) or not jnp.issubdtype(
            destination.dtype, jnp.integer
        ):
            raise ValueError("destinations must hold one integer owner per slot.")
        moved = self.points if points is None else jnp.asarray(points)
        if moved.shape != self.points.shape:
            raise ValueError("Migrated points must preserve the owner-blocked shape.")
        leaves, structure = jax.tree.flatten(payload)
        arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
        for leaf in arrays:
            if leaf.ndim == 0 or leaf.shape[0] != self.plan.total_capacity:
                raise ValueError("Payload leaves must begin with the owner slot axis.")
        return _migrate(
            self,
            self.plan.place(destination.astype(jnp.int32)),
            self.plan.place(moved),
            tuple(self.plan.place(leaf) for leaf in arrays),
            structure,
            capacity,
        )


class DistributedMigrationResult(NonTrainableState, StrictModule):
    """Committed (or rolled-back) layout, payload, and transfer evidence."""

    layout: DistributedPointLayout
    payload: Any
    evidence: DistributedMigrationEvidence


@eqx.filter_jit
def _audit_unique_ids(
    plan: DistributedOwnershipPlan, identifiers: Array, active: Array
) -> Array:
    owners = plan.owner_count
    capacity = plan.local_capacity
    axis = plan.axis_name

    def audit(local_ids: Array, local_active: Array) -> Array:
        provider = JaxCollectiveProvider(axis)
        bucket = jnp.mod(local_ids, owners).astype(jnp.int32)
        rank, _ = _bucket_ranks(bucket, local_active, owners)
        sent_ids = _pack(local_ids, bucket, rank, local_active, owners, capacity, 0)
        sent_valid = _pack(
            local_active, bucket, rank, local_active, owners, capacity, False
        )
        received_ids = provider.all_to_all(sent_ids, split_axis=0, concat_axis=0)
        received_valid = provider.all_to_all(sent_valid, split_axis=0, concat_axis=0)
        flat_ids = received_ids.reshape((-1,))
        flat_valid = received_valid.reshape((-1,))
        order = jnp.lexsort((flat_ids, ~flat_valid))
        ordered_ids = flat_ids[order]
        ordered_valid = flat_valid[order]
        duplicate = jnp.any(
            ordered_valid[1:] & ordered_valid[:-1] & (ordered_ids[1:] == ordered_ids[:-1])
        )
        return provider.maximum(duplicate.astype(jnp.int32)) == 0

    return plan.map(
        audit,
        (_spec(axis, 1), _spec(axis, 1)),
        PartitionSpec(),
    )(identifiers, active)


@eqx.filter_jit
def _migrate(
    layout: DistributedPointLayout,
    destination: Array,
    moved: Array,
    payload: tuple[Array, ...],
    structure: Any,
    capacity: int,
) -> DistributedMigrationResult:
    plan = layout.plan
    owners = plan.owner_count
    local = plan.local_capacity
    axis = plan.axis_name
    address = plan.address_plan
    leaf_count = len(payload)

    def exchange(
        old_points: Array,
        new_points: Array,
        ids: Array,
        active: Array,
        logical: Array,
        epoch: Array,
        target: Array,
        *leaves: Array,
    ) -> tuple[Array, ...]:
        provider = JaxCollectiveProvider(axis)
        rank_index = jax.lax.axis_index(axis).astype(jnp.int32)
        in_range = (target >= 0) & (target < owners)
        invalid = jnp.sum(active & ~in_range, dtype=jnp.int32)
        valid = active & in_range
        rank, loads = _bucket_ranks(target, valid, owners)
        fits = valid & (rank < capacity)

        def route(values: Array, fill: ArrayLike) -> Array:
            packet = _pack(values, target, rank, fits, owners, capacity, fill)
            received = provider.all_to_all(packet, split_axis=0, concat_axis=0)
            return received.reshape((owners * capacity,) + values.shape[1:])

        sender_epoch = jnp.broadcast_to(epoch, active.shape)
        received_points = route(new_points, 0)
        received_ids = route(ids, 0)
        received_logical = route(logical, -1)
        received_valid = route(fits, False)
        received_epoch = route(sender_epoch, -1)
        received_leaves = tuple(route(leaf, 0) for leaf in leaves)
        received_count = jnp.sum(received_valid, dtype=jnp.int32)
        epochs_consistent = jnp.all(~received_valid | (received_epoch == epoch[0]))

        padding = max(local - owners * capacity, 0)

        def padded(values: Array, fill: ArrayLike) -> Array:
            if padding == 0:
                return values
            extra = jnp.full((padding,) + values.shape[1:], fill, dtype=values.dtype)
            return jnp.concatenate((values, extra), axis=0)

        all_points = padded(received_points, 0)
        all_ids = padded(received_ids, 0)
        all_valid = padded(received_valid, False)
        codes = address.encode(all_points).codes
        order = canonical_morton_order(codes, all_ids, all_valid)[:local]

        local_ok = (
            (invalid == 0)
            & jnp.all(loads <= capacity)
            & (received_count <= local)
            & epochs_consistent
        )
        committed = provider.minimum(local_ok.astype(jnp.int32)) == 1

        def commit(candidate: Array, previous: Array) -> Array:
            return jnp.where(committed, candidate, previous)

        moved_count = jnp.sum(valid & (target != rank_index), dtype=jnp.int32)
        new_leaves = tuple(
            commit(padded(value, 0)[order], previous)
            for value, previous in zip(received_leaves, leaves, strict=True)
        )
        return (
            commit(all_points[order], old_points),
            commit(all_ids[order], ids),
            commit(all_valid[order], active),
            commit(
                jnp.where(all_valid[order], padded(received_logical, -1)[order], -1),
                logical,
            ),
            commit(epoch + 1, epoch),
            committed,
            provider.sum(invalid),
            provider.maximum(jnp.max(loads)),
            provider.maximum(received_count),
            provider.sum(moved_count),
            provider.minimum(epochs_consistent.astype(jnp.int32)) == 1,
            provider.maximum(epoch[0]),
            *new_leaves,
        )

    row = _spec(axis, 1)
    point = _spec(axis, 2)
    replicated = PartitionSpec()
    leaf_specs = tuple(_spec(axis, leaf.ndim) for leaf in payload)
    outputs = plan.map(
        exchange,
        (point, point, row, row, row, row, row, *leaf_specs),
        (
            point,
            row,
            row,
            row,
            row,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
            *leaf_specs,
        ),
    )(
        layout.points,
        moved,
        layout.stable_ids,
        layout.active,
        layout.logical_indices,
        layout.owner_epochs,
        destination,
        *payload,
    )
    (
        new_points,
        new_ids,
        new_active,
        new_logical,
        new_epochs,
        committed,
        invalid,
        maximum_packet,
        maximum_received,
        migrated,
        epochs_consistent,
        epoch_before,
    ) = outputs[:12]
    new_payload = jax.tree.unflatten(structure, list(outputs[12 : 12 + leaf_count]))
    result_layout = DistributedPointLayout(
        plan,
        new_points,
        new_ids,
        new_active,
        new_logical,
        new_epochs,
        layout.stable_ids_unique,
        layout.logical_count,
    )
    evidence = DistributedMigrationEvidence(
        committed=committed,
        invalid_destinations=invalid,
        maximum_packet=maximum_packet,
        packet_capacity=jnp.asarray(capacity, dtype=jnp.int32),
        maximum_received=maximum_received,
        local_capacity=jnp.asarray(local, dtype=jnp.int32),
        migrated=jnp.where(committed, migrated, 0),
        epochs_consistent=epochs_consistent,
        epoch_before=epoch_before,
        epoch_after=jnp.where(committed, epoch_before + 1, epoch_before),
    )
    return DistributedMigrationResult(
        layout=result_layout, payload=new_payload, evidence=evidence
    )


class DistributedNeighborEvidence(NonTrainableState, StrictModule):
    """Global completeness, capacity, identity, and communication evidence."""

    successful: Array
    status_counts: Array
    finite: Array
    sources_valid: Array
    stable_ids_unique: Array
    owners_current: Array
    epoch: Array
    owner_count: Array
    maximum_required_owners: Array
    remote_owner_capacity: Array
    maximum_halo_load: Array
    halo_capacity: Array
    required_candidates: Array
    candidate_capacity: Array
    communicated_targets: Array


class DistributedNeighborResult(NonTrainableState, StrictModule):
    """Target-owner-blocked neighbor rows with global source addresses.

    Rows follow the target layout's owner blocks. ``source_owners`` and
    ``source_slots`` address the source layout; ``source_logical`` is the
    logical row of the source and ``source_stable_ids`` its stable global ID.
    Entries are valid only for rows whose status is ``COMPLETE``.
    """

    source_stable_ids: Array
    source_owners: Array
    source_slots: Array
    source_logical: Array
    distance_squared: Array
    valid: Array
    counts: Array
    status: Array
    evidence: DistributedNeighborEvidence


class _ShellConfig(NamedTuple):
    owners: int
    source_slots: int
    target_slots: int
    remote_owners: int
    halo_capacity: int
    selected: int
    width: int
    knn: bool


class _OwnerEvidence(NamedTuple):
    lower: Array
    upper: Array
    counts: Array
    sources_valid: Array
    current: Array
    finite: Array
    epoch: Array


class _Candidates(NamedTuple):
    distance: Array
    ids: Array
    owners: Array
    slots: Array
    logical: Array
    valid: Array


def _owner_evidence(
    provider: JaxCollectiveProvider,
    address: MortonAddressPlan,
    points: Array,
    active: Array,
    epoch: Array,
    dtype: Any,
) -> _OwnerEvidence:
    encoding = address.encode(points)
    usable = active & encoding.in_domain
    coordinates = encoding.coordinates.astype(dtype)
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    lower = jnp.min(jnp.where(usable[:, None], coordinates, infinity), axis=0)
    upper = jnp.max(jnp.where(usable[:, None], coordinates, -infinity), axis=0)
    count = jnp.sum(usable, dtype=jnp.int32)
    valid = ~jnp.any(active & ~encoding.in_domain)
    finite = jnp.all(encoding.finite | ~active)
    latest = provider.maximum(epoch[0])
    return _OwnerEvidence(
        lower=provider.all_gather(lower[None], axis=0, tiled=True),
        upper=provider.all_gather(upper[None], axis=0, tiled=True),
        counts=provider.all_gather(count[None], axis=0, tiled=True),
        sources_valid=provider.minimum(valid.astype(jnp.int32)) == 1,
        current=provider.minimum((epoch[0] == latest).astype(jnp.int32)) == 1,
        finite=provider.minimum(finite.astype(jnp.int32)) == 1,
        epoch=latest,
    )


def _engine_candidates(
    engine: MortonNeighborQueryPlan,
    address: MortonAddressPlan,
    sources: tuple[Array, Array, Array, Array],
    targets: Array,
    target_active: Array,
    target_ids: Array,
    owner: Array,
    *,
    exclude_self: bool,
    radius: float | None,
    target_radii: Array | None = None,
) -> tuple[_Candidates, Array, Array]:
    points, ids, active, logical = sources
    result = engine.query(
        points,
        targets,
        source_mask=active,
        target_mask=target_active,
        source_stable_ids=ids,
        target_stable_ids=target_ids,
        exclude_self=exclude_self,
        radius=radius,
        target_radii=target_radii,
    )
    slots = result.source_indices
    source_coordinates = address.encode(points).coordinates
    target_coordinates = address.encode(targets).coordinates
    relative = _minimum_image(
        target_coordinates[:, None, :] - source_coordinates[slots], address
    )
    distance = jnp.sum(relative * relative, axis=-1)
    candidates = _Candidates(
        distance=distance,
        ids=ids[slots],
        owners=jnp.broadcast_to(owner, slots.shape).astype(jnp.int32),
        slots=slots.astype(jnp.int32),
        logical=logical[slots],
        valid=result.valid,
    )
    return candidates, result.status, result.evidence.required_candidates


def _search_radius(
    config: _ShellConfig,
    local: _Candidates,
    owner: _OwnerEvidence,
    far: Array,
    margin: float,
    *,
    exclude_self: bool,
    radius: float | None,
) -> Array:
    dtype = far.dtype
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    if not config.knn:
        if radius is None:
            raise ValueError("Radius queries require a declared search radius.")
        return jnp.full(far.shape[:1], radius, dtype=dtype)
    found = jnp.sum(local.valid, axis=1)
    if local.distance.shape[1] >= config.selected:
        kth = jnp.sqrt(local.distance[:, config.selected - 1]).astype(dtype)
        local_bound = jnp.where(found >= config.selected, kth, infinity)
    else:
        local_bound = jnp.full(found.shape, jnp.inf, dtype=dtype)
    # Owners whose farthest box point lies within R jointly hold every point of
    # their boxes; once they hold enough sources R bounds the k-th distance.
    needed = config.selected + int(exclude_self)
    populated = owner.counts > 0
    reach = jnp.where(populated[None, :], far + margin, infinity)
    order = jnp.argsort(reach, axis=1, stable=True)
    ordered_reach = jnp.take_along_axis(reach, order, axis=1)
    cumulative = jnp.cumsum(owner.counts[order], axis=1)
    enough = cumulative >= needed
    first = jnp.argmax(enough, axis=1)
    box_bound = jnp.where(
        jnp.any(enough, axis=1),
        jnp.take_along_axis(ordered_reach, first[:, None], axis=1)[:, 0],
        infinity,
    )
    bound = jnp.minimum(local_bound, box_bound)
    if radius is not None:
        bound = jnp.minimum(bound, jnp.asarray(radius, dtype=dtype))
    return bound


def _shell_body(
    config: _ShellConfig,
    address: MortonAddressPlan,
    axis: str,
    local_engine: MortonNeighborQueryPlan,
    remote_engine: MortonNeighborQueryPlan | None,
    exclude_self: bool,
    radius: float | None,
    pair_once: bool,
) -> Callable[..., tuple[Array, ...]]:
    owners = config.owners

    def body(
        source_points: Array,
        source_ids: Array,
        source_active: Array,
        source_logical: Array,
        source_epoch: Array,
        ids_unique: Array,
        target_points: Array,
        target_ids: Array,
        target_active: Array,
    ) -> tuple[Array, ...]:
        provider = JaxCollectiveProvider(axis)
        rank = jax.lax.axis_index(axis).astype(jnp.int32)
        dtype = jnp.result_type(source_points.dtype, target_points.dtype)
        margin = _rounding_margin(address, dtype)
        owner = _owner_evidence(
            provider, address, source_points, source_active, source_epoch, dtype
        )
        sources = (source_points, source_ids, source_active, source_logical)
        local, local_status, local_required = _engine_candidates(
            local_engine,
            address,
            sources,
            target_points,
            target_active,
            target_ids,
            rank,
            exclude_self=exclude_self,
            radius=radius,
        )
        target_encoding = address.encode(target_points)
        target_valid = target_active & target_encoding.in_domain
        target_finite = jnp.all(target_encoding.finite | ~target_active)
        near, far = _owner_box_distances(
            target_encoding.coordinates.astype(dtype), owner.lower, owner.upper, address
        )
        bound = _search_radius(
            config,
            local,
            owner,
            far,
            margin,
            exclude_self=exclude_self,
            radius=radius,
        )
        owner_index = jnp.arange(owners, dtype=jnp.int32)
        required = (
            target_valid[:, None]
            & (owner.counts > 0)[None, :]
            & (owner_index != rank)[None, :]
            & (near - margin <= bound[:, None])
        )
        required_count = jnp.sum(required, axis=1, dtype=jnp.int32)
        owner_overflow = required_count > config.remote_owners
        if remote_engine is None:
            merged = local
            answers = _RemoteAnswers(
                candidates=local,
                status=jnp.zeros(target_valid.shape, dtype=jnp.int32),
                halo_overflow=jnp.zeros(target_valid.shape, dtype=jnp.bool_),
                maximum_load=jnp.asarray(0, dtype=jnp.int32),
                required_candidates=jnp.asarray(0, dtype=jnp.int32),
                communicated=jnp.asarray(0, dtype=jnp.int32),
            )
        else:
            # A remote owner answers only within the requester's certified
            # search radius (padded by the rounding margin): every k-th
            # neighbor lies inside it, and its candidate buffer then holds only
            # the stencil cells within that radius rather than an unbounded
            # k-nearest selection for a target outside the owner's region.
            answers = _remote_answers(
                config,
                provider,
                address,
                remote_engine,
                sources,
                rank,
                jnp.where(required, near, jnp.asarray(jnp.inf, dtype=dtype)),
                required_count,
                owner_overflow,
                target_points,
                target_ids,
                bound + margin,
                exclude_self=exclude_self,
                radius=radius,
            )
            merged = _Candidates(
                *(
                    jnp.concatenate((first, second), axis=1)
                    for first, second in zip(local, answers.candidates, strict=True)
                )
            )
        rows = _merge_rows(config, merged, target_ids, pair_once=pair_once)
        merged_rows, row_overflow = rows
        status = _resolve_status(
            local_status,
            answers.status,
            answers.halo_overflow,
            owner_overflow,
            row_overflow,
            target_valid,
            owner.sources_valid & ids_unique,
            owner.current,
        )
        complete = status == DistributedRelationStatus.COMPLETE
        final_valid = merged_rows.valid & complete[:, None]

        def masked(values: Array) -> Array:
            return jnp.where(final_valid, values, jnp.zeros((), dtype=values.dtype))

        histogram = provider.sum(
            jnp.zeros((_STATUS_COUNT,), dtype=jnp.int32).at[status].add(1)
        )
        return (
            masked(merged_rows.ids),
            masked(merged_rows.owners),
            masked(merged_rows.slots),
            masked(merged_rows.logical),
            masked(merged_rows.distance),
            final_valid,
            jnp.sum(final_valid, axis=1, dtype=jnp.int32),
            status,
            histogram,
            owner.finite & (provider.minimum(target_finite.astype(jnp.int32)) == 1),
            owner.sources_valid,
            owner.current,
            owner.epoch,
            provider.maximum(jnp.max(required_count, initial=0)),
            provider.maximum(answers.maximum_load),
            provider.maximum(jnp.maximum(local_required, answers.required_candidates)),
            provider.sum(answers.communicated),
        )

    return body


class _RemoteAnswers(NamedTuple):
    candidates: _Candidates
    status: Array
    halo_overflow: Array
    maximum_load: Array
    required_candidates: Array
    communicated: Array


def _remote_answers(
    config: _ShellConfig,
    provider: JaxCollectiveProvider,
    address: MortonAddressPlan,
    engine: MortonNeighborQueryPlan,
    sources: tuple[Array, Array, Array, Array],
    rank: Array,
    required_near: Array,
    required_count: Array,
    owner_overflow: Array,
    target_points: Array,
    target_ids: Array,
    search_radius: Array,
    *,
    exclude_self: bool,
    radius: float | None,
) -> _RemoteAnswers:
    """Ship targets to their nearest required owners and gather their answers.

    Each target contacts at most ``remote_owners`` owners, nearest box first;
    each owner accepts at most ``halo_capacity`` targets from each requester.
    """
    owners = config.owners
    remote = config.remote_owners
    halo = config.halo_capacity
    count = target_ids.shape[0]
    _, selected_owners = jax.lax.sort(
        (
            required_near,
            jnp.broadcast_to(jnp.arange(owners, dtype=jnp.int32), required_near.shape),
        ),
        dimension=1,
        num_keys=2,
    )
    selected_valid = (
        jnp.arange(remote, dtype=jnp.int32)[None, :] < required_count[:, None]
    ) & ~owner_overflow[:, None]
    destination = selected_owners[:, :remote].reshape((-1,))
    pair_valid = selected_valid.reshape((-1,))
    rank_in_bucket, loads = _bucket_ranks(destination, pair_valid, owners)
    fits = pair_valid & (rank_in_bucket < halo)
    pair_fits = fits.reshape(selected_valid.shape)
    target_rows = jnp.repeat(jnp.arange(count, dtype=jnp.int32), remote)

    def send(values: Array, fill: ArrayLike) -> Array:
        packet = _pack(values, destination, rank_in_bucket, fits, owners, halo, fill)
        received = provider.all_to_all(packet, split_axis=0, concat_axis=0)
        return received.reshape((owners * halo,) + values.shape[1:])

    answers, answer_status, required_candidates = _engine_candidates(
        engine,
        address,
        sources,
        send(target_points[target_rows], 0),
        send(jnp.ones(destination.shape, dtype=jnp.bool_), False),
        send(target_ids[target_rows], 0),
        rank,
        exclude_self=exclude_self,
        radius=radius,
        target_radii=send(search_radius[target_rows], 0),
    )
    slot = jnp.where(fits, rank_in_bucket, 0)
    source_owner = jnp.where(fits, destination, 0)

    def reply(values: Array) -> Array:
        """Return answers to requesters and pick each target's pair rows."""
        shaped = values.reshape((owners, halo) + values.shape[1:])
        returned = provider.all_to_all(shaped, split_axis=0, concat_axis=0)
        return returned[source_owner, slot].reshape(
            selected_valid.shape + values.shape[1:]
        )

    pair_status = jnp.where(pair_fits, reply(answer_status), 0)
    gathered = _Candidates(*(reply(value) for value in answers))
    valid = gathered.valid & pair_fits[:, :, None] & (pair_status == 0)[:, :, None]
    gathered = gathered._replace(valid=valid)
    return _RemoteAnswers(
        candidates=_Candidates(
            *(value.reshape((count, -1) + value.shape[3:]) for value in gathered)
        ),
        status=jnp.max(pair_status, axis=1),
        halo_overflow=jnp.any(selected_valid & ~pair_fits, axis=1),
        maximum_load=jnp.max(loads),
        required_candidates=required_candidates,
        communicated=jnp.sum(fits, dtype=jnp.int32),
    )


def _merge_rows(
    config: _ShellConfig,
    candidates: _Candidates,
    target_ids: Array,
    *,
    pair_once: bool,
) -> tuple[_Candidates, Array]:
    """Deterministic ``(distance, stable_id, owner)`` merge of owner answers."""
    infinity = jnp.asarray(jnp.inf, dtype=candidates.distance.dtype)
    id_sentinel = jnp.asarray(jnp.iinfo(candidates.ids.dtype).max, candidates.ids.dtype)
    owner_sentinel = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    shortfall = config.selected - candidates.valid.shape[1]
    if shortfall > 0:
        # Fewer owner answers than requested slots: pad with invalid entries.
        candidates = _Candidates(
            *(jnp.pad(value, ((0, 0), (0, shortfall))) for value in candidates)
        )
    valid = candidates.valid
    ordered = jax.lax.sort(
        (
            jnp.where(valid, candidates.distance, infinity),
            jnp.where(valid, candidates.ids, id_sentinel),
            jnp.where(valid, candidates.owners, owner_sentinel),
            candidates.slots,
            candidates.logical,
            valid,
        ),
        dimension=1,
        num_keys=3,
    )
    kept = _Candidates(
        distance=ordered[0][:, : config.selected],
        ids=ordered[1][:, : config.selected],
        owners=ordered[2][:, : config.selected],
        slots=ordered[3][:, : config.selected],
        logical=ordered[4][:, : config.selected],
        valid=ordered[5][:, : config.selected],
    )
    if config.knn:
        return kept, jnp.zeros(target_ids.shape, dtype=jnp.bool_)
    # Radius rows fetch one extra slot: a valid extra neighbor proves overflow.
    row_overflow = jnp.sum(kept.valid, axis=1) > config.width
    if pair_once:
        retained = kept.valid & (kept.ids > target_ids[:, None])
        ordered = jax.lax.sort(
            (
                (~retained).astype(jnp.int32),
                jnp.where(retained, kept.distance, infinity),
                jnp.where(retained, kept.ids, id_sentinel),
                kept.owners,
                kept.slots,
                kept.logical,
                retained,
            ),
            dimension=1,
            num_keys=3,
        )
        kept = _Candidates(
            distance=ordered[1],
            ids=ordered[2],
            owners=ordered[3],
            slots=ordered[4],
            logical=ordered[5],
            valid=ordered[6],
        )
    return _Candidates(*(value[:, : config.width] for value in kept)), row_overflow


def _resolve_status(
    local_status: Array,
    remote_status: Array,
    halo_overflow: Array,
    owner_overflow: Array,
    row_overflow: Array,
    target_valid: Array,
    sources_valid: Array,
    owners_current: Array,
) -> Array:
    """Later conditions take precedence; target validity is decided last."""
    status = jnp.full(local_status.shape, DistributedRelationStatus.COMPLETE, jnp.int32)
    status = jnp.where(
        row_overflow, jnp.int32(DistributedRelationStatus.ROW_OVERFLOW), status
    )
    status = jnp.where(remote_status != 0, remote_status, status)
    status = jnp.where(
        halo_overflow, jnp.int32(DistributedRelationStatus.HALO_OVERFLOW), status
    )
    status = jnp.where(
        owner_overflow, jnp.int32(DistributedRelationStatus.OWNER_OVERFLOW), status
    )
    status = jnp.where(
        local_status != MortonNeighborQueryStatus.COMPLETE, local_status, status
    )
    status = jnp.where(
        owners_current, status, jnp.int32(DistributedRelationStatus.MISSING_OWNER)
    )
    status = jnp.where(
        sources_valid, status, jnp.int32(DistributedRelationStatus.INVALID_SOURCES)
    )
    status = jnp.where(
        local_status == MortonNeighborQueryStatus.INVALID_TARGET,
        jnp.int32(DistributedRelationStatus.INVALID_TARGET),
        status,
    )
    return jnp.where(
        local_status == MortonNeighborQueryStatus.INACTIVE_TARGET,
        jnp.int32(DistributedRelationStatus.INACTIVE_TARGET),
        status,
    ).astype(jnp.int32)


class _ShellQueryCore(StrictModule):
    """Shared validated capacities and local engines of both query kinds."""

    source_ownership: DistributedOwnershipPlan
    target_ownership: DistributedOwnershipPlan
    local_engine: MortonNeighborQueryPlan
    remote_engine: MortonNeighborQueryPlan | None
    config: _ShellConfig = eqx.field(static=True)
    core_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_ownership: DistributedOwnershipPlan,
        target_ownership: DistributedOwnershipPlan,
        selected: int,
        width: int,
        knn: bool,
        maximum_remote_owners: int | None,
        halo_capacity: int | None,
        maximum_candidates: int | None,
        target_chunk_size: int | None,
        distance_backend: SpatialDistanceBackend,
    ) -> None:
        if not isinstance(source_ownership, DistributedOwnershipPlan) or not isinstance(
            target_ownership, DistributedOwnershipPlan
        ):
            raise TypeError("Ownership plans must be DistributedOwnershipPlan values.")
        _require_x64()
        if not source_ownership.compatible(target_ownership):
            raise ValueError(
                "Source and target ownership must share devices, axis, and addressing."
            )
        owners = source_ownership.owner_count
        sources = source_ownership.local_capacity
        targets = target_ownership.local_capacity
        remote = (
            owners - 1 if maximum_remote_owners is None else int(maximum_remote_owners)
        )
        if remote < 0 or remote > owners - 1:
            raise ValueError("maximum_remote_owners must lie in [0, owner_count - 1].")
        halo = targets if halo_capacity is None else int(halo_capacity)
        if halo < 1:
            raise ValueError("halo_capacity must be positive.")
        engine_neighbors = min(selected, sources)
        candidates = (
            None
            if maximum_candidates is None
            else max(engine_neighbors, min(int(maximum_candidates), sources))
        )
        address = source_ownership.address_plan
        self.local_engine = MortonNeighborQueryPlan(
            address,
            sources,
            targets,
            engine_neighbors,
            maximum_candidates=candidates,
            target_chunk_size=target_chunk_size,
            distance_backend=distance_backend,
        )
        self.remote_engine = (
            None
            if remote == 0
            else MortonNeighborQueryPlan(
                address,
                sources,
                owners * halo,
                engine_neighbors,
                maximum_candidates=candidates,
                target_chunk_size=target_chunk_size,
                distance_backend=distance_backend,
            )
        )
        self.source_ownership = source_ownership
        self.target_ownership = target_ownership
        self.config = _ShellConfig(
            owners=owners,
            source_slots=sources,
            target_slots=targets,
            remote_owners=remote,
            halo_capacity=halo,
            selected=selected,
            width=width,
            knn=knn,
        )
        self.core_id = canonical_fingerprint(
            {
                "kind": "distributed-shell-query",
                "source_ownership": source_ownership.plan_id,
                "target_ownership": target_ownership.plan_id,
                "local_engine": self.local_engine.plan_id,
                "remote_engine": (
                    None if self.remote_engine is None else self.remote_engine.plan_id
                ),
                "selected": selected,
                "width": width,
                "knn": knn,
                "maximum_remote_owners": remote,
                "halo_capacity": halo,
            }
        )

    def run(
        self,
        sources: DistributedPointLayout,
        targets: DistributedPointLayout,
        *,
        exclude_self: bool,
        radius: float | None,
        pair_once: bool,
    ) -> DistributedNeighborResult:
        if not isinstance(sources, DistributedPointLayout) or not isinstance(
            targets, DistributedPointLayout
        ):
            raise TypeError("Queries require DistributedPointLayout sources and targets.")
        if sources.plan.plan_id != self.source_ownership.plan_id:
            raise ValueError("sources do not use this plan's source ownership.")
        if targets.plan.plan_id != self.target_ownership.plan_id:
            raise ValueError("targets do not use this plan's target ownership.")
        _require_x64()
        return _run_shell_query(self, sources, targets, exclude_self, radius, pair_once)

    def _execute(
        self,
        sources: DistributedPointLayout,
        targets: DistributedPointLayout,
        exclude_self: bool,
        radius: float | None,
        pair_once: bool,
    ) -> DistributedNeighborResult:
        axis = self.source_ownership.axis_name
        body = _shell_body(
            self.config,
            self.source_ownership.address_plan,
            axis,
            self.local_engine,
            self.remote_engine,
            exclude_self,
            radius,
            pair_once,
        )
        row = _spec(axis, 1)
        matrix = _spec(axis, 2)
        replicated = PartitionSpec()
        outputs = self.source_ownership.map(
            body,
            (matrix, row, row, row, row, replicated, matrix, row, row),
            (matrix,) * 6 + (row, row) + (replicated,) * 9,
        )(
            sources.points,
            sources.stable_ids,
            sources.active,
            sources.logical_indices,
            sources.owner_epochs,
            sources.stable_ids_unique,
            targets.points,
            targets.stable_ids,
            targets.active,
        )
        (
            ids,
            owners,
            slots,
            logical,
            distance,
            valid,
            counts,
            status,
            histogram,
            finite,
            sources_valid,
            current,
            epoch,
            maximum_required,
            maximum_load,
            required_candidates,
            communicated,
        ) = outputs
        failures = jnp.sum(histogram[DistributedRelationStatus.INVALID_TARGET :])
        evidence = DistributedNeighborEvidence(
            successful=(failures == 0) & sources.stable_ids_unique,
            status_counts=histogram,
            finite=finite,
            sources_valid=sources_valid,
            stable_ids_unique=sources.stable_ids_unique,
            owners_current=current,
            epoch=epoch,
            owner_count=jnp.asarray(self.config.owners, dtype=jnp.int32),
            maximum_required_owners=maximum_required,
            remote_owner_capacity=jnp.asarray(self.config.remote_owners, dtype=jnp.int32),
            maximum_halo_load=maximum_load,
            halo_capacity=jnp.asarray(self.config.halo_capacity, dtype=jnp.int32),
            required_candidates=required_candidates,
            candidate_capacity=jnp.asarray(
                self.local_engine.maximum_candidates, dtype=jnp.int32
            ),
            communicated_targets=communicated,
        )
        return DistributedNeighborResult(
            source_stable_ids=ids,
            source_owners=owners,
            source_slots=slots,
            source_logical=logical,
            distance_squared=distance,
            valid=valid,
            counts=counts,
            status=status,
            evidence=evidence,
        )


@eqx.filter_jit
def _run_shell_query(
    core: _ShellQueryCore,
    sources: DistributedPointLayout,
    targets: DistributedPointLayout,
    exclude_self: bool,
    radius: float | None,
    pair_once: bool,
) -> DistributedNeighborResult:
    return core._execute(sources, targets, exclude_self, radius, pair_once)


@final
class DistributedNeighborQueryPlan(StrictModule):
    """Exact distributed k-nearest neighbors with shell-certified owner contact.

    ``maximum_remote_owners`` bounds how many other owners one target may
    contact; ``halo_capacity`` bounds how many targets one owner may send to
    one other owner. Exceeding either refuses the affected targets with
    ``OWNER_OVERFLOW`` or ``HALO_OVERFLOW``.
    """

    core: _ShellQueryCore
    maximum_neighbors: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_ownership: DistributedOwnershipPlan,
        target_ownership: DistributedOwnershipPlan,
        maximum_neighbors: int,
        /,
        *,
        maximum_remote_owners: int | None = None,
        halo_capacity: int | None = None,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
        distance_backend: SpatialDistanceBackend = "jax",
    ) -> None:
        neighbors = _static_positive(maximum_neighbors, "maximum_neighbors")
        self.core = _ShellQueryCore(
            source_ownership,
            target_ownership,
            neighbors,
            neighbors,
            True,
            maximum_remote_owners,
            halo_capacity,
            maximum_candidates,
            target_chunk_size,
            distance_backend,
        )
        self.maximum_neighbors = neighbors
        self.plan_id = canonical_fingerprint(
            {"kind": "distributed-neighbor-query-plan", "core": self.core.core_id}
        )

    def query(
        self,
        sources: DistributedPointLayout,
        targets: DistributedPointLayout,
        /,
        *,
        exclude_self: bool = False,
        radius: float | None = None,
    ) -> DistributedNeighborResult:
        radius_value = None if radius is None else float(radius)
        if radius_value is not None and (
            not np.isfinite(radius_value) or radius_value <= 0
        ):
            raise ValueError("radius must be finite and positive when supplied.")
        return self.core.run(
            sources,
            targets,
            exclude_self=bool(exclude_self),
            radius=radius_value,
            pair_once=False,
        )


@final
class DistributedRadiusQueryPlan(StrictModule):
    """Exact distributed inclusive radius rows of bounded width.

    Every source within ``radius`` of a target is returned, ordered by
    ``(distance, stable_id, owner)``. A row holding more than
    ``maximum_row_neighbors`` sources is refused with ``ROW_OVERFLOW``.
    """

    core: _ShellQueryCore
    radius: float = eqx.field(static=True)
    maximum_row_neighbors: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_ownership: DistributedOwnershipPlan,
        target_ownership: DistributedOwnershipPlan,
        radius: float,
        maximum_row_neighbors: int,
        /,
        *,
        maximum_remote_owners: int | None = None,
        halo_capacity: int | None = None,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
        distance_backend: SpatialDistanceBackend = "jax",
    ) -> None:
        radius_value = float(radius)
        if not np.isfinite(radius_value) or radius_value <= 0:
            raise ValueError("radius must be finite and positive.")
        width = _static_positive(maximum_row_neighbors, "maximum_row_neighbors")
        self.core = _ShellQueryCore(
            source_ownership,
            target_ownership,
            width + 1,
            width,
            False,
            maximum_remote_owners,
            halo_capacity,
            maximum_candidates,
            target_chunk_size,
            distance_backend,
        )
        self.radius = radius_value
        self.maximum_row_neighbors = width
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-radius-query-plan",
                "core": self.core.core_id,
                "radius": radius_value,
            }
        )

    def query(
        self,
        sources: DistributedPointLayout,
        targets: DistributedPointLayout,
        /,
        *,
        exclude_self: bool = False,
        pair_once: bool = False,
    ) -> DistributedNeighborResult:
        """Radius rows; ``pair_once`` keeps each unordered pair in one row.

        With ``pair_once`` the sources and targets must be the same layout and a
        pair is reported only by its smaller-stable-ID endpoint, so every
        unordered pair is owned exactly once across all owners.
        """
        if pair_once and sources is not targets:
            raise ValueError(
                "pair_once requires the same layout for sources and targets."
            )
        return self.core.run(
            sources,
            targets,
            exclude_self=bool(exclude_self),
            radius=self.radius,
            pair_once=bool(pair_once),
        )


class DistributedHaloEvidence(NonTrainableState, StrictModule):
    """Halo completeness and capacity evidence."""

    successful: Array
    refused_routes: Array
    maximum_halo_load: Array
    halo_capacity: Array
    halo_columns: Array


@final
class DistributedHaloPlan(NonTrainableState, StrictModule):
    """Owned-row routes bound to owned columns plus deduplicated halo columns.

    Each owner's local column space has ``local_capacity`` owned columns
    followed by ``owner_count * halo_capacity`` halo columns; halo column
    ``local_capacity + o * halo_capacity + h`` receives slot ``send_slots[o, h]``
    of owner ``o``. Every remote source slot appears at most once per
    requesting owner, so the transpose returns each contribution exactly once.
    Transposed halo contributions accumulate after owned contributions in
    ascending requesting-owner order.
    """

    ownership: DistributedOwnershipPlan
    send_slots: Array
    send_valid: Array
    route_columns: Array
    route_valid: Array
    evidence: DistributedHaloEvidence
    route_capacity: int = eqx.field(static=True)
    halo_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        ownership: DistributedOwnershipPlan,
        route_owners: ArrayLike,
        route_slots: ArrayLike,
        route_valid: ArrayLike,
        /,
        *,
        halo_capacity: int,
    ) -> None:
        if not isinstance(ownership, DistributedOwnershipPlan):
            raise TypeError("ownership must be a DistributedOwnershipPlan.")
        owners_array = jnp.asarray(route_owners)
        slots_array = jnp.asarray(route_slots)
        valid_array = jnp.asarray(route_valid, dtype=jnp.bool_)
        total = owners_array.shape[0] if owners_array.ndim == 1 else -1
        if (
            total < 0
            or total % ownership.owner_count
            or slots_array.shape != owners_array.shape
            or valid_array.shape != owners_array.shape
        ):
            raise ValueError(
                "Route owners, slots, and validity must be owner-blocked vectors."
            )
        halo = _static_positive(halo_capacity, "halo_capacity")
        routes = total // ownership.owner_count
        outputs = _prepare_halo(
            ownership,
            ownership.place(owners_array.astype(jnp.int32)),
            ownership.place(slots_array.astype(jnp.int32)),
            ownership.place(valid_array),
            halo,
        )
        (
            send_slots,
            send_valid,
            columns,
            kept,
            refused,
            maximum_load,
            halo_columns,
        ) = outputs
        self.ownership = ownership
        self.send_slots = send_slots
        self.send_valid = send_valid
        self.route_columns = columns
        self.route_valid = kept
        self.evidence = DistributedHaloEvidence(
            successful=refused == 0,
            refused_routes=refused,
            maximum_halo_load=maximum_load,
            halo_capacity=jnp.asarray(halo, dtype=jnp.int32),
            halo_columns=halo_columns,
        )
        self.route_capacity = routes
        self.halo_capacity = halo

    @property
    def column_count(self) -> int:
        """Owned plus halo local columns of one owner."""
        return self.ownership.local_capacity + self.ownership.owner_count * (
            self.halo_capacity
        )

    def local_gather(
        self, values: Array, send_slots: Array, send_valid: Array, /
    ) -> Array:
        """Owned plus halo columns of one owner, inside a mapped owner region."""
        owners = self.ownership.owner_count
        halo = self.halo_capacity
        provider = JaxCollectiveProvider(self.ownership.axis_name)
        slots = send_slots.reshape((owners, halo))
        valid = send_valid.reshape((owners, halo))
        packet = values[jnp.where(valid, slots, 0)]
        mask = valid.reshape(valid.shape + (1,) * (values.ndim - 1))
        packet = jnp.where(mask, packet, jnp.zeros((), dtype=values.dtype))
        received = provider.all_to_all(packet, split_axis=0, concat_axis=0)
        return jnp.concatenate(
            (values, received.reshape((owners * halo,) + values.shape[1:])), axis=0
        )

    def local_transpose(
        self, contributions: Array, send_slots: Array, send_valid: Array, /
    ) -> Array:
        """Sum owned and returned halo contributions onto owned slots exactly once."""
        owners = self.ownership.owner_count
        halo = self.halo_capacity
        local = self.ownership.local_capacity
        provider = JaxCollectiveProvider(self.ownership.axis_name)
        owned = contributions[:local]
        returned = provider.all_to_all(
            contributions[local:].reshape((owners, halo) + contributions.shape[1:]),
            split_axis=0,
            concat_axis=0,
        )
        slots = jnp.where(
            send_valid.reshape((owners, halo)), send_slots.reshape((owners, halo)), local
        )

        def accumulate(requester: Array, total: Array) -> Array:
            return total.at[slots[requester]].add(returned[requester], mode="drop")

        return jax.lax.fori_loop(0, owners, accumulate, owned)

    def gather(self, values: ArrayLike, /) -> Array:
        """Owner-blocked ``(owner_count * column_count, ...)`` local columns."""
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.ownership.total_capacity:
            raise ValueError("Values must begin with the owner-blocked slot axis.")
        return _halo_exchange(self, self.ownership.place(array), False)

    def transpose(self, contributions: ArrayLike, /) -> Array:
        """Owner-blocked transpose of :meth:`gather`."""
        array = jnp.asarray(contributions)
        if (
            array.ndim == 0
            or array.shape[0] != self.ownership.owner_count * self.column_count
        ):
            raise ValueError("Contributions must begin with the local column axis.")
        return _halo_exchange(self, self.ownership.place(array), True)


@eqx.filter_jit
def _halo_exchange(halo: DistributedHaloPlan, values: Array, transpose: bool) -> Array:
    axis = halo.ownership.axis_name
    return halo.ownership.map(
        halo.local_transpose if transpose else halo.local_gather,
        (_spec(axis, values.ndim), _spec(axis, 2), _spec(axis, 2)),
        _spec(axis, values.ndim),
    )(values, halo.send_slots, halo.send_valid)


@eqx.filter_jit
def _prepare_halo(
    ownership: DistributedOwnershipPlan,
    route_owners: Array,
    route_slots: Array,
    route_valid: Array,
    halo: int,
) -> tuple[Array, ...]:
    owners = ownership.owner_count
    local = ownership.local_capacity
    axis = ownership.axis_name

    def prepare(owner: Array, slot: Array, valid: Array) -> tuple[Array, ...]:
        provider = JaxCollectiveProvider(axis)
        rank = jax.lax.axis_index(axis).astype(jnp.int32)
        in_range = (owner >= 0) & (owner < owners) & (slot >= 0) & (slot < local)
        usable = valid & in_range
        is_remote = usable & (owner != rank)
        count = owner.shape[0]
        sentinel = jnp.asarray(owners * local, dtype=jnp.int32)
        key = jnp.where(is_remote, owner * local + slot, sentinel)
        order = jnp.argsort(key, stable=True)
        ordered = key[order]
        head = jnp.concatenate(
            (jnp.ones((1,), dtype=jnp.bool_), ordered[1:] != ordered[:-1])
        ) & (ordered < sentinel)
        unique_index = jnp.cumsum(head, dtype=jnp.int32) - 1
        # Compact unique remote columns; each owner's columns form one sorted
        # group whose offset is the column's position in that owner's halo.
        unique_keys = (
            jnp.full((count,), sentinel, dtype=jnp.int32)
            .at[jnp.where(head, unique_index, count)]
            .set(ordered, mode="drop")
        )
        unique_valid = unique_keys < sentinel
        unique_owners = jnp.where(unique_valid, unique_keys // local, owners)
        group_start = jnp.searchsorted(unique_owners, unique_owners, side="left")
        position = jnp.arange(count, dtype=jnp.int32) - group_start.astype(jnp.int32)
        fits_unique = unique_valid & (position < halo)
        request_slots = _pack(
            unique_keys % local,
            unique_owners,
            position,
            fits_unique,
            owners,
            halo,
            0,
        )
        request_valid = _pack(
            fits_unique, unique_owners, position, fits_unique, owners, halo, False
        )
        send_slots = provider.all_to_all(request_slots, split_axis=0, concat_axis=0)
        send_valid = provider.all_to_all(request_valid, split_axis=0, concat_axis=0)
        route_unique = jnp.zeros((count,), dtype=jnp.int32).at[order].set(unique_index)
        route_position = position[jnp.clip(route_unique, 0, count - 1)]
        route_fits = fits_unique[jnp.clip(route_unique, 0, count - 1)]
        remote_column = local + owner * halo + route_position
        columns = jnp.where(is_remote, remote_column, slot)
        kept = usable & (~is_remote | route_fits)
        columns = jnp.where(kept, columns, 0).astype(jnp.int32)
        refused = jnp.sum(valid & ~kept, dtype=jnp.int32)
        loads = jax.ops.segment_sum(
            unique_valid.astype(jnp.int32),
            jnp.where(unique_valid, unique_owners, 0),
            owners,
        )
        return (
            send_slots.reshape((owners, halo)),
            send_valid.reshape((owners, halo)),
            columns,
            kept,
            provider.sum(refused),
            provider.maximum(jnp.max(loads)),
            provider.sum(jnp.sum(fits_unique, dtype=jnp.int32)),
        )

    row = _spec(axis, 1)
    replicated = PartitionSpec()
    return ownership.map(
        prepare,
        (row, row, row),
        (_spec(axis, 2), _spec(axis, 2), row, row, replicated, replicated, replicated),
    )(route_owners, route_slots, route_valid)


__all__ = [
    "DistributedHaloEvidence",
    "DistributedHaloPlan",
    "DistributedMigrationEvidence",
    "DistributedMigrationResult",
    "DistributedNeighborEvidence",
    "DistributedNeighborQueryPlan",
    "DistributedNeighborResult",
    "DistributedOwnershipPlan",
    "DistributedPointLayout",
    "DistributedRadiusQueryPlan",
    "DistributedRelationStatus",
]
