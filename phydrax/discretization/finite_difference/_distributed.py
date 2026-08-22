#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._boundary import HaloPlan


class DistributedStencilPartition(StrictModule, NonTrainableState):
    """Modern NamedSharding plan plus explicit physical/inter-device halo semantics."""

    global_shape: tuple[int, ...] = eqx.field(static=True)
    partition_axis: int = eqx.field(static=True)
    device_axis_name: str = eqx.field(static=True)
    device_count: int = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    halo_plan: HaloPlan
    sharding: NamedSharding = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        global_shape: Sequence[int],
        partition_axis: int,
        halo_plan: HaloPlan,
        /,
        *,
        devices: Sequence[jax.Device] | None = None,
        device_axis_name: str = "partitions",
        periodic: bool = False,
    ):
        shape = tuple(int(size) for size in global_shape)
        axis = int(partition_axis)
        if (
            not shape
            or any(size <= 0 for size in shape)
            or axis < 0
            or axis >= len(shape)
        ):
            raise ValueError("Invalid distributed global shape or partition axis.")
        if not isinstance(halo_plan, HaloPlan):
            raise TypeError("halo_plan must be a HaloPlan.")
        selected = tuple(jax.devices() if devices is None else devices)
        if not selected or shape[axis] % len(selected):
            raise ValueError("Partition axis size must divide the selected device count.")
        axis_name = str(device_axis_name)
        if not axis_name:
            raise ValueError("device_axis_name must be non-empty.")
        mesh = Mesh(np.asarray(selected, dtype=object), (axis_name,))
        partitions: list[str | None] = [None] * len(shape)
        partitions[axis] = axis_name
        sharding = NamedSharding(mesh, PartitionSpec(*partitions))
        self.global_shape = shape
        self.partition_axis = axis
        self.device_axis_name = axis_name
        self.device_count = len(selected)
        self.periodic = bool(periodic)
        self.halo_plan = halo_plan
        self.sharding = sharding
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-stencil-partition",
                "global_shape": list(shape),
                "partition_axis": axis,
                "device_axis_name": axis_name,
                "device_count": len(selected),
                "periodic": bool(periodic),
                "halo": halo_plan.halo_id,
            }
        )

    def shard(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.shape != self.global_shape:
            raise ValueError("Distributed value shape must match global_shape.")
        return jax.device_put(array, self.sharding)

    def permutation(self, direction: int, /) -> tuple[tuple[int, int], ...]:
        offset = int(direction)
        if offset not in (-1, 1):
            raise ValueError("Permutation direction must be -1 or +1.")
        pairs = []
        for source in range(self.device_count):
            target = source + offset
            if self.periodic:
                target %= self.device_count
            if 0 <= target < self.device_count:
                pairs.append((source, target))
        return tuple(pairs)

    def ppermute_halo(self, values: Array, direction: int, /) -> Array:
        """Exchange one mapped halo payload inside ``shard_map``/collective context."""
        return jax.lax.ppermute(
            values,
            axis_name=self.device_axis_name,
            perm=self.permutation(direction),
        )

    def exchange_block_halos_1d(self, blocks: ArrayLike, /) -> Array:
        """Reference leading-partition halo exchange for deterministic verification."""
        values = jnp.asarray(blocks)
        if values.ndim < 2 or values.shape[0] != self.device_count:
            raise ValueError("blocks must begin with one entry per partition.")
        width = self.halo_plan.lower_widths[self.partition_axis]
        if width != self.halo_plan.upper_widths[self.partition_axis]:
            raise ValueError("Reference exchange requires symmetric halo width.")
        if width == 0:
            return values
        left_indices = jnp.arange(self.device_count) - 1
        right_indices = jnp.arange(self.device_count) + 1
        if self.periodic:
            left_indices %= self.device_count
            right_indices %= self.device_count
            left_valid = right_valid = jnp.ones((self.device_count,), dtype=bool)
        else:
            left_valid = left_indices >= 0
            right_valid = right_indices < self.device_count
            left_indices = jnp.clip(left_indices, 0, self.device_count - 1)
            right_indices = jnp.clip(right_indices, 0, self.device_count - 1)
        left = values[left_indices, -width:]
        right = values[right_indices, :width]
        left_mask = left_valid.reshape((-1,) + (1,) * (left.ndim - 1))
        right_mask = right_valid.reshape((-1,) + (1,) * (right.ndim - 1))
        padded = jnp.pad(
            values,
            ((0, 0), (width, width)) + ((0, 0),) * (values.ndim - 2),
        )
        padded = padded.at[:, :width].set(jnp.where(left_mask, left, 0.0))
        padded = padded.at[:, -width:].set(jnp.where(right_mask, right, 0.0))
        return padded


class HaloExchangeDescriptor(StrictModule, NonTrainableState):
    """One face/edge/corner neighbor offset and per-axis payload width."""

    offset: tuple[int, ...] = eqx.field(static=True)
    widths: tuple[int, ...] = eqx.field(static=True)
    codimension: int = eqx.field(static=True)
    descriptor_id: str = eqx.field(static=True)

    def __init__(self, offset: Sequence[int], widths: Sequence[int], /):
        offset_ = tuple(int(value) for value in offset)
        widths_ = tuple(int(value) for value in widths)
        if (
            not offset_
            or len(offset_) != len(widths_)
            or all(value == 0 for value in offset_)
            or any(value not in (-1, 0, 1) for value in offset_)
            or any(value < 0 for value in widths_)
            or any(
                value and width == 0
                for value, width in zip(offset_, widths_, strict=True)
            )
        ):
            raise ValueError("Halo exchange offset and widths are invalid.")
        self.offset = offset_
        self.widths = widths_
        self.codimension = sum(value != 0 for value in offset_)
        self.descriptor_id = canonical_fingerprint(
            {
                "kind": "halo-exchange-descriptor",
                "offset": list(offset_),
                "widths": list(widths_),
            }
        )


class DistributedHaloSchedule(StrictModule, NonTrainableState):
    """Multi-axis NamedSharding and deterministic face/edge/corner halo schedule."""

    global_shape: tuple[int, ...] = eqx.field(static=True)
    partition_shape: tuple[int, ...] = eqx.field(static=True)
    local_shape: tuple[int, ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    halo_plan: HaloPlan
    mesh_axis_names: tuple[str, ...] = eqx.field(static=True)
    sharding: NamedSharding = eqx.field(static=True)
    exchanges: tuple[HaloExchangeDescriptor, ...]
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        global_shape: Sequence[int],
        partition_shape: Sequence[int],
        halo_plan: HaloPlan,
        /,
        *,
        periodic_axes: Sequence[bool] | None = None,
        devices: Sequence[jax.Device] | None = None,
        mesh_axis_prefix: str = "fd",
    ):
        shape = tuple(int(value) for value in global_shape)
        partitions = tuple(int(value) for value in partition_shape)
        if (
            not shape
            or len(partitions) != len(shape)
            or any(value <= 0 for value in shape + partitions)
            or any(size % count for size, count in zip(shape, partitions, strict=True))
        ):
            raise ValueError("Distributed halo global/partition shapes are invalid.")
        if not isinstance(halo_plan, HaloPlan) or len(halo_plan.axis_names) != len(shape):
            raise TypeError("Halo plan rank must match the distributed tensor rank.")
        periodic = (
            (False,) * len(shape)
            if periodic_axes is None
            else tuple(bool(value) for value in periodic_axes)
        )
        if len(periodic) != len(shape):
            raise ValueError("periodic_axes must align with global_shape.")
        selected = tuple(jax.devices() if devices is None else devices)
        if int(np.prod(partitions)) != len(selected):
            raise ValueError("partition_shape product must equal selected device count.")
        prefix = str(mesh_axis_prefix)
        if not prefix:
            raise ValueError("mesh_axis_prefix must be non-empty.")
        mesh_names = tuple(f"{prefix}_{axis}" for axis in range(len(shape)))
        mesh = Mesh(np.asarray(selected, dtype=object).reshape(partitions), mesh_names)
        spec = PartitionSpec(
            *tuple(
                name if count > 1 else None
                for name, count in zip(mesh_names, partitions, strict=True)
            )
        )
        offsets = tuple(
            offset
            for offset in np.ndindex(*(3,) * len(shape))
            if any(value != 1 for value in offset)
        )
        descriptors = []
        for encoded in offsets:
            offset = tuple(value - 1 for value in encoded)
            widths = tuple(
                halo_plan.lower_widths[axis]
                if direction < 0
                else halo_plan.upper_widths[axis]
                if direction > 0
                else 0
                for axis, direction in enumerate(offset)
            )
            if all(
                direction == 0 or widths[axis] > 0
                for axis, direction in enumerate(offset)
            ):
                descriptors.append(HaloExchangeDescriptor(offset, widths))
        self.global_shape = shape
        self.partition_shape = partitions
        self.local_shape = tuple(
            size // count for size, count in zip(shape, partitions, strict=True)
        )
        self.periodic_axes = periodic
        self.halo_plan = halo_plan
        self.mesh_axis_names = mesh_names
        self.sharding = NamedSharding(mesh, spec)
        self.exchanges = tuple(descriptors)
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "distributed-halo-schedule",
                "global_shape": list(shape),
                "partition_shape": list(partitions),
                "periodic_axes": list(periodic),
                "halo": halo_plan.halo_id,
                "exchanges": [value.descriptor_id for value in descriptors],
            }
        )

    def permutation(
        self,
        axis: int,
        direction: int,
        /,
    ) -> tuple[tuple[int, int], ...]:
        axis_ = int(axis)
        direction_ = int(direction)
        if axis_ < 0 or axis_ >= len(self.partition_shape) or direction_ not in (-1, 1):
            raise ValueError("Collective halo axis/direction is invalid.")
        count = self.partition_shape[axis_]
        pairs = []
        for source in range(count):
            target = source + direction_
            if self.periodic_axes[axis_]:
                target %= count
            if 0 <= target < count:
                pairs.append((source, target))
        return tuple(pairs)

    def ppermute_halo(
        self,
        values: Array,
        axis: int,
        direction: int,
        /,
    ) -> Array:
        """Exchange one local payload on a selected mesh axis inside shard_map."""
        axis_ = int(axis)
        return jax.lax.ppermute(
            values,
            axis_name=self.mesh_axis_names[axis_],
            perm=self.permutation(axis_, direction),
        )

    def shard(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape != self.global_shape:
            raise ValueError("Distributed halo value must match global_shape.")
        return jax.device_put(value, self.sharding)

    def interior_slices(self, /) -> tuple[slice, ...]:
        return tuple(
            slice(lower, size - upper if upper else size)
            for size, lower, upper in zip(
                self.local_shape,
                self.halo_plan.lower_widths,
                self.halo_plan.upper_widths,
                strict=True,
            )
        )

    def exchange_reference(self, blocks: ArrayLike, /) -> Array:
        """Deterministic all-codimension exchange for partition-indexed local blocks."""
        value = jnp.asarray(blocks)
        expected = self.partition_shape + self.local_shape
        if value.shape != expected:
            raise ValueError(
                f"Reference blocks must have shape {expected}; got {value.shape}."
            )
        global_value = jnp.zeros(self.global_shape, dtype=value.dtype)
        for partition_index in np.ndindex(*self.partition_shape):
            global_index = tuple(
                slice(index * local, (index + 1) * local)
                for index, local in zip(partition_index, self.local_shape, strict=True)
            )
            global_value = global_value.at[global_index].set(value[partition_index])
        padded = global_value
        for axis, (lower, upper, periodic) in enumerate(
            zip(
                self.halo_plan.lower_widths,
                self.halo_plan.upper_widths,
                self.periodic_axes,
                strict=True,
            )
        ):
            padding = [(0, 0)] * padded.ndim
            padding[axis] = (lower, upper)
            padded = jnp.pad(
                padded,
                tuple(padding),
                mode="wrap" if periodic else "constant",
            )
        exchanged = []
        for partition_index in np.ndindex(*self.partition_shape):
            index = tuple(
                slice(index * local, (index + 1) * local + lower + upper)
                for index, local, lower, upper in zip(
                    partition_index,
                    self.local_shape,
                    self.halo_plan.lower_widths,
                    self.halo_plan.upper_widths,
                    strict=True,
                )
            )
            exchanged.append(padded[index])
        output_shape = self.partition_shape + tuple(
            local + lower + upper
            for local, lower, upper in zip(
                self.local_shape,
                self.halo_plan.lower_widths,
                self.halo_plan.upper_widths,
                strict=True,
            )
        )
        return jnp.stack(exchanged).reshape(output_shape)


__all__ = [
    "DistributedHaloSchedule",
    "DistributedStencilPartition",
    "HaloExchangeDescriptor",
]
