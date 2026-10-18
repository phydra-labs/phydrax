#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Live JAX resource inventories, meshes, and nested execution groups."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ._execution_device import device_vendor
from ._execution_plan import ExecutionPlan
from ._execution_resources import (
    DeviceResource,
    ExecutionGroupSpec,
    ResourceInventory,
)
from ._execution_sampling import sample_device_memory, sample_host_memory
from ._fingerprint import canonical_fingerprint


def _local_host_boot_id() -> bytes | None:
    """Read kernel boot identity, never a hostname or a rank-derived guess."""

    if not sys.platform.startswith("linux"):
        return None
    try:
        identity = UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    except (OSError, UnicodeError, ValueError):
        return None
    return identity.bytes if identity.int else None


def _discover_process_hosts() -> tuple[tuple[int, str], ...]:
    """Exchange fixed-size kernel identities; incomplete evidence stays unknown."""

    identity = _local_host_boot_id()
    local = np.frombuffer(identity if identity is not None else bytes(16), dtype=np.uint8)
    identities = np.asarray(multihost_utils.process_allgather(local, tiled=False))
    if any(not np.any(record) for record in identities):
        return ()
    return tuple(
        (process, f"linux-boot-{UUID(bytes=record.tobytes())}")
        for process, record in enumerate(identities)
    )


def discover_resource_inventory() -> ResourceInventory:
    """Collectively describe every JAX device after runtime bootstrap.

    Device capacity is the allocator ``bytes_limit`` reported by its owning
    process. All processes exchange capacity in canonical device order so the
    inventory identity is shared. Linux kernel boot identities establish physical
    host ownership across processes; unavailable identity leaves the host mapping
    empty so multi-process resource admission refuses it. Timestamped host and
    device memory samples remain local evidence; unavailable owner capacity
    remains unknown. Single-process inventory identity is unchanged.

    Every process must call discovery in the same collective sequence. For
    rank-local memory observations, use ``sample_host_memory`` and
    ``sample_device_memory`` instead.
    """

    devices = sorted(jax.devices(), key=lambda item: (item.process_index, item.id))
    process = jax.process_index()
    local = tuple(device for device in devices if device.process_index == process)
    device_memory = sample_device_memory(local)
    capacity = {
        sample.key: sample.allocator_limit.value_bytes for sample in device_memory
    }
    process_hosts: tuple[tuple[int, str], ...] = ()
    if jax.process_count() > 1:
        local_capacity = np.asarray(
            [capacity.get((device.process_index, device.id)) or 0 for device in devices],
            dtype=np.int64,
        )
        capacities = np.asarray(
            multihost_utils.process_allgather(local_capacity, tiled=False)
        )
        capacity = {
            (device.process_index, device.id): int(
                capacities[device.process_index, index]
            )
            for index, device in enumerate(devices)
        }
        process_hosts = _discover_process_hosts()
    resources = tuple(
        DeviceResource(
            process_index=device.process_index,
            device_id=device.id,
            local_device_id=device.local_hardware_id,
            platform=device.platform,
            kind=device.device_kind,
            vendor=device_vendor(device),
            # A zero allocator limit reports no usable capacity, not a capacity.
            memory_bytes=capacity.get((device.process_index, device.id)) or None,
        )
        for device in devices
    )
    return ResourceInventory(
        jax.process_count(),
        process,
        resources,
        process_host_ids=process_hosts,
        host_memory=(sample_host_memory(),),
        device_memory=device_memory,
    )


def root_execution_group_spec(
    inventory: ResourceInventory,
    *,
    mesh_axes: Sequence[tuple[str, int]] | None = None,
) -> ExecutionGroupSpec:
    """Reserve the complete initialized runtime as one coupled group."""

    axes = (
        (("device", len(inventory.devices)),) if mesh_axes is None else tuple(mesh_axes)
    )
    processes = tuple(sorted({device.process_index for device in inventory.devices}))
    inventory_hosts = dict(inventory.process_host_ids)
    process_hosts = tuple(
        (process, inventory_hosts[process])
        for process in processes
        if process in inventory_hosts
    )
    payload = {
        "inventory_id": inventory.inventory_id,
        "devices": [list(device.key) for device in inventory.devices],
        "mesh_axes": [[name, size] for name, size in axes],
    }
    group_id = canonical_fingerprint(payload)
    return ExecutionGroupSpec(
        group_id,
        processes,
        tuple(device.key for device in inventory.devices),
        mesh_axes=axes,
        process_host_ids=process_hosts,
    )


def partition_execution_group_specs(
    parent: ExecutionGroupSpec,
    group_count: int,
    *,
    mesh_axis: str = "device",
) -> tuple[ExecutionGroupSpec, ...]:
    """Split every process's devices equally into process-symmetric groups."""

    if group_count <= 0:
        raise ValueError("group_count must be positive")
    by_process = {
        process: tuple(key for key in parent.device_keys if key[0] == process)
        for process in parent.process_indices
    }
    local_counts = {len(keys) for keys in by_process.values()}
    if len(local_counts) != 1:
        raise ValueError("process-symmetric groups require equal device counts")
    local_count = next(iter(local_counts))
    if local_count % group_count:
        raise ValueError("each process device count must be divisible by group_count")
    devices_per_process = local_count // group_count
    groups: list[ExecutionGroupSpec] = []
    for group_index in range(group_count):
        start = group_index * devices_per_process
        stop = start + devices_per_process
        device_keys = tuple(
            key
            for process in parent.process_indices
            for key in by_process[process][start:stop]
        )
        group_id = canonical_fingerprint(
            {
                "parent_group_id": parent.group_id,
                "group_index": group_index,
                "device_keys": [list(key) for key in device_keys],
            }
        )
        groups.append(
            ExecutionGroupSpec(
                group_id,
                parent.process_indices,
                device_keys,
                mesh_axes=((mesh_axis, len(device_keys)),),
                parent_group_id=parent.group_id,
                process_host_ids=parent.process_host_ids,
            )
        )
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class ExecutionGroup:
    """Live devices and JAX mesh bound to a serializable group specification."""

    spec: ExecutionGroupSpec
    devices: tuple[jax.Device, ...]
    mesh: Mesh

    def __init__(
        self,
        spec: ExecutionGroupSpec,
        devices: Sequence[jax.Device],
    ) -> None:
        devices_ = tuple(devices)
        actual_keys = tuple((device.process_index, device.id) for device in devices_)
        if actual_keys != spec.device_keys:
            raise ValueError(
                "live devices do not match the execution group specification"
            )
        axis_names = tuple(name for name, _ in spec.mesh_axes)
        axis_shape = tuple(size for _, size in spec.mesh_axes)
        if not axis_names:
            axis_names = ("device",)
            axis_shape = (len(devices_),)
        device_array = np.asarray(devices_, dtype=object).reshape(axis_shape)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "devices", devices_)
        object.__setattr__(self, "mesh", Mesh(device_array, axis_names))

    @property
    def is_member(self) -> bool:
        return jax.process_index() in self.spec.process_indices

    @property
    def is_coordinator(self) -> bool:
        return jax.process_index() == min(self.spec.process_indices)

    def named_sharding(
        self,
        partition_spec: PartitionSpec | Sequence[str | None],
    ) -> NamedSharding:
        spec = (
            partition_spec
            if isinstance(partition_spec, PartitionSpec)
            else PartitionSpec(*partition_spec)
        )
        return NamedSharding(self.mesh, spec)


@dataclass(frozen=True, slots=True)
class ExecutionRuntime:
    """Live root allocation for one initialized JAX process set."""

    inventory: ResourceInventory
    root_group: ExecutionGroup

    @classmethod
    def current(
        cls,
        *,
        mesh_axes: Sequence[tuple[str, int]] | None = None,
    ) -> ExecutionRuntime:
        """Collectively discover and bind the initialized process set.

        Every process must enter this call in the same collective sequence.
        """

        inventory = discover_resource_inventory()
        spec = root_execution_group_spec(inventory, mesh_axes=mesh_axes)
        return cls(inventory, bind_execution_group(spec))

    def bind_plan(self, plan: ExecutionPlan) -> ExecutionGroup:
        if (
            plan.inventory_id is not None
            and plan.inventory_id != self.inventory.inventory_id
        ):
            raise ValueError("execution plan inventory does not match this runtime")
        if plan.group is None:
            raise ValueError("execution plan does not contain an execution group")
        return bind_execution_group(plan.group)

    def child_groups(
        self,
        group_count: int,
        *,
        mesh_axis: str = "device",
    ) -> tuple[ExecutionGroup, ...]:
        specs = partition_execution_group_specs(
            self.root_group.spec,
            group_count,
            mesh_axis=mesh_axis,
        )
        return tuple(bind_execution_group(spec) for spec in specs)


def bind_execution_group(spec: ExecutionGroupSpec) -> ExecutionGroup:
    """Bind a serializable group to the matching devices in this process set."""

    devices_by_key: Mapping[tuple[int, int], jax.Device] = {
        (device.process_index, device.id): device for device in jax.devices()
    }
    missing = tuple(key for key in spec.device_keys if key not in devices_by_key)
    if missing:
        raise ValueError(f"execution group references unavailable devices: {missing}")
    devices = tuple(devices_by_key[key] for key in spec.device_keys)
    return ExecutionGroup(spec, devices)


__all__ = (
    "ExecutionGroup",
    "ExecutionRuntime",
    "bind_execution_group",
    "discover_resource_inventory",
    "partition_execution_group_specs",
    "root_execution_group_spec",
)
