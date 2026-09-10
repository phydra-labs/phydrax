#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-neutral resource requests, inventories, and allocations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from ._fingerprint import canonical_fingerprint


def _identifier(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


class DistributionMode(str, Enum):
    """How an operation may use the initialized resource allocation."""

    SINGLE = "single"
    AUTO = "auto"
    DISTRIBUTED = "distributed"


class DeterminismScope(str, Enum):
    """Reproducibility contract requested from an execution plan."""

    LOGICAL = "logical"
    TOPOLOGY = "topology"
    BITWISE = "bitwise"


class RecoveryPolicy(str, Enum):
    """Failure boundary for an execution group."""

    FAIL_FAST = "fail_fast"
    CHECKPOINT_RESTART = "checkpoint_restart"


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Total resources requested for one root execution allocation."""

    cpu_cores: int
    memory_bytes: int
    accelerator_count: int = 0
    host_count: int = 1
    process_count: int = 1
    accelerator_platform: str | None = None
    accelerator_vendor: str | None = None
    minimum_accelerator_memory_bytes: int | None = None
    exclusive: bool = True

    def __post_init__(self) -> None:
        if self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        if self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        if self.accelerator_count < 0:
            raise ValueError("accelerator_count must be non-negative")
        if self.host_count <= 0:
            raise ValueError("host_count must be positive")
        if self.process_count < self.host_count:
            raise ValueError("process_count must be at least host_count")
        if self.accelerator_platform is not None:
            object.__setattr__(
                self,
                "accelerator_platform",
                _identifier(self.accelerator_platform, "accelerator_platform"),
            )
        if self.accelerator_vendor is not None:
            object.__setattr__(
                self,
                "accelerator_vendor",
                _identifier(self.accelerator_vendor, "accelerator_vendor"),
            )
        if (
            self.minimum_accelerator_memory_bytes is not None
            and self.minimum_accelerator_memory_bytes <= 0
        ):
            raise ValueError("minimum_accelerator_memory_bytes must be positive")
        accelerator_constraints = (
            self.accelerator_platform is not None
            or self.accelerator_vendor is not None
            or self.minimum_accelerator_memory_bytes is not None
        )
        if self.accelerator_count == 0 and accelerator_constraints:
            raise ValueError("accelerator constraints require accelerator_count > 0")

    @property
    def resource_id(self) -> str:
        return canonical_fingerprint(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_bytes": self.memory_bytes,
            "accelerator_count": self.accelerator_count,
            "host_count": self.host_count,
            "process_count": self.process_count,
            "accelerator_platform": self.accelerator_platform,
            "accelerator_vendor": self.accelerator_vendor,
            "minimum_accelerator_memory_bytes": (self.minimum_accelerator_memory_bytes),
            "exclusive": self.exclusive,
        }


@dataclass(frozen=True, slots=True)
class DeviceResource:
    """Serializable description of one JAX-visible device."""

    process_index: int
    device_id: int
    local_device_id: int | None
    platform: str
    kind: str

    def __post_init__(self) -> None:
        if self.process_index < 0 or self.device_id < 0:
            raise ValueError("process_index and device_id must be non-negative")
        if self.local_device_id is not None and self.local_device_id < 0:
            raise ValueError("local_device_id must be non-negative")
        object.__setattr__(self, "platform", _identifier(self.platform, "platform"))
        object.__setattr__(self, "kind", _identifier(self.kind, "kind"))

    @property
    def key(self) -> tuple[int, int]:
        return (self.process_index, self.device_id)

    def to_payload(self) -> dict[str, object]:
        return {
            "process_index": self.process_index,
            "device_id": self.device_id,
            "local_device_id": self.local_device_id,
            "platform": self.platform,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class ResourceInventory:
    """Observed resources in one initialized JAX runtime."""

    process_count: int
    process_index: int
    devices: tuple[DeviceResource, ...]

    def __init__(
        self,
        process_count: int,
        process_index: int,
        devices: Sequence[DeviceResource],
    ) -> None:
        devices_ = tuple(devices)
        if process_count <= 0:
            raise ValueError("process_count must be positive")
        if not 0 <= process_index < process_count:
            raise ValueError("process_index must be in [0, process_count)")
        if not devices_:
            raise ValueError("devices cannot be empty")
        if len({device.key for device in devices_}) != len(devices_):
            raise ValueError("device resources must have unique process/device keys")
        if any(device.process_index >= process_count for device in devices_):
            raise ValueError("device process_index exceeds process_count")
        object.__setattr__(self, "process_count", process_count)
        object.__setattr__(self, "process_index", process_index)
        object.__setattr__(self, "devices", devices_)

    @property
    def inventory_id(self) -> str:
        return canonical_fingerprint(self.to_payload())

    @property
    def platforms(self) -> tuple[str, ...]:
        return tuple(sorted({device.platform for device in self.devices}))

    @property
    def local_devices(self) -> tuple[DeviceResource, ...]:
        return tuple(
            device
            for device in self.devices
            if device.process_index == self.process_index
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "process_count": self.process_count,
            "devices": [
                {
                    "process_index": device.process_index,
                    "device_id": device.device_id,
                    "platform": device.platform,
                    "kind": device.kind,
                }
                for device in self.devices
            ],
        }


@dataclass(frozen=True, slots=True)
class ExecutionGroupSpec:
    """Serializable process/device reservation for one coupled computation."""

    group_id: str
    process_indices: tuple[int, ...]
    device_keys: tuple[tuple[int, int], ...]
    mesh_axes: tuple[tuple[str, int], ...] = ()
    parent_group_id: str | None = None

    def __init__(
        self,
        group_id: str,
        process_indices: Sequence[int],
        device_keys: Sequence[tuple[int, int]],
        *,
        mesh_axes: Sequence[tuple[str, int]] = (),
        parent_group_id: str | None = None,
    ) -> None:
        group = _identifier(group_id, "group_id")
        processes = tuple(int(value) for value in process_indices)
        devices = tuple((int(process), int(device)) for process, device in device_keys)
        axes = tuple(
            (_identifier(name, "mesh axis"), int(size)) for name, size in mesh_axes
        )
        if not processes or len(set(processes)) != len(processes):
            raise ValueError("process_indices must be non-empty and unique")
        if min(processes) < 0:
            raise ValueError("process_indices must be non-negative")
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("device_keys must be non-empty and unique")
        if any(process not in processes for process, _ in devices):
            raise ValueError("every device must belong to a group process")
        if len({name for name, _ in axes}) != len(axes):
            raise ValueError("mesh axis names must be unique")
        if any(size <= 0 for _, size in axes):
            raise ValueError("mesh axis sizes must be positive")
        mesh_size = 1
        for _, size in axes:
            mesh_size *= size
        if axes and mesh_size != len(devices):
            raise ValueError("mesh axis product must equal the number of devices")
        object.__setattr__(self, "group_id", group)
        object.__setattr__(self, "process_indices", processes)
        object.__setattr__(self, "device_keys", devices)
        object.__setattr__(self, "mesh_axes", axes)
        object.__setattr__(
            self,
            "parent_group_id",
            None
            if parent_group_id is None
            else _identifier(parent_group_id, "parent_group_id"),
        )

    @property
    def device_count(self) -> int:
        return len(self.device_keys)

    def to_payload(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "process_indices": list(self.process_indices),
            "device_keys": [list(key) for key in self.device_keys],
            "mesh_axes": [[name, size] for name, size in self.mesh_axes],
            "parent_group_id": self.parent_group_id,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """User intent resolved against owner-supplied execution candidates."""

    distribution: DistributionMode = DistributionMode.SINGLE
    resources: ResourceRequest | None = None
    provider_preferences: tuple[str, ...] = ()
    strict_providers: bool = False
    determinism: DeterminismScope = DeterminismScope.LOGICAL
    recovery: RecoveryPolicy = RecoveryPolicy.FAIL_FAST
    maximum_gather_bytes: int | None = None

    def __init__(
        self,
        distribution: DistributionMode = DistributionMode.SINGLE,
        *,
        resources: ResourceRequest | None = None,
        provider_preferences: Sequence[str] = (),
        strict_providers: bool = False,
        determinism: DeterminismScope = DeterminismScope.LOGICAL,
        recovery: RecoveryPolicy = RecoveryPolicy.FAIL_FAST,
        maximum_gather_bytes: int | None = None,
    ) -> None:
        providers = tuple(
            _identifier(provider, "provider preference")
            for provider in provider_preferences
        )
        if len(set(providers)) != len(providers):
            raise ValueError("provider_preferences must be unique")
        if strict_providers and not providers:
            raise ValueError("strict_providers requires provider_preferences")
        if maximum_gather_bytes is not None and maximum_gather_bytes < 0:
            raise ValueError("maximum_gather_bytes must be non-negative")
        object.__setattr__(self, "distribution", DistributionMode(distribution))
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "provider_preferences", providers)
        object.__setattr__(self, "strict_providers", bool(strict_providers))
        object.__setattr__(self, "determinism", DeterminismScope(determinism))
        object.__setattr__(self, "recovery", RecoveryPolicy(recovery))
        object.__setattr__(self, "maximum_gather_bytes", maximum_gather_bytes)

    @classmethod
    def single(cls) -> ExecutionPolicy:
        return cls(DistributionMode.SINGLE)

    @classmethod
    def auto(
        cls,
        *,
        resources: ResourceRequest | None = None,
        provider_preferences: Sequence[str] = (),
        determinism: DeterminismScope = DeterminismScope.LOGICAL,
        recovery: RecoveryPolicy = RecoveryPolicy.FAIL_FAST,
        maximum_gather_bytes: int | None = None,
    ) -> ExecutionPolicy:
        return cls(
            DistributionMode.AUTO,
            resources=resources,
            provider_preferences=provider_preferences,
            determinism=determinism,
            recovery=recovery,
            maximum_gather_bytes=maximum_gather_bytes,
        )

    @property
    def policy_id(self) -> str:
        return canonical_fingerprint(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "distribution": self.distribution.value,
            "resources": None if self.resources is None else self.resources.to_payload(),
            "provider_preferences": list(self.provider_preferences),
            "strict_providers": self.strict_providers,
            "determinism": self.determinism.value,
            "recovery": self.recovery.value,
            "maximum_gather_bytes": self.maximum_gather_bytes,
        }


__all__ = (
    "DeterminismScope",
    "DeviceResource",
    "DistributionMode",
    "ExecutionGroupSpec",
    "ExecutionPolicy",
    "RecoveryPolicy",
    "ResourceInventory",
    "ResourceRequest",
)
