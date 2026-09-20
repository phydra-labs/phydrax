#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-neutral resource requests, inventories, and allocations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._fingerprint import canonical_fingerprint
from ._validation import normalized_identifier


def _identifier(value: str, name: str) -> str:
    return normalized_identifier(value, name)


def _identifiers(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} values must be a sequence of strings")
    normalized = tuple(sorted(_identifier(value, name) for value in values))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} values must be unique")
    return normalized


def _optional_byte_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _process_host_records(
    values: Mapping[int, str] | Sequence[tuple[int, str]],
    name: str,
    /,
) -> tuple[tuple[int, str], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    records = tuple(
        sorted(
            (int(process), _identifier(host, f"{name} host ID"))
            for process, host in items
        )
    )
    if len({process for process, _ in records}) != len(records):
        raise ValueError(f"{name} process indices must be unique")
    return records


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
    """Total resources and fail-closed component budgets for one allocation."""

    cpu_cores: int
    memory_bytes: int
    accelerator_count: int = 0
    host_count: int = 1
    process_count: int = 1
    accelerator_platform: str | None = None
    accelerator_vendor: str | None = None
    minimum_accelerator_memory_bytes: int | None = None
    exclusive: bool = True
    maximum_device_bytes: int | None = None
    maximum_host_bytes: int | None = None
    maximum_compilation_cache_bytes: int | None = None
    maximum_halo_collective_bytes: int | None = None
    maximum_checkpoint_staging_bytes: int | None = None
    maximum_output_backlog_bytes: int | None = None
    required_dtypes: tuple[str, ...] = ()
    required_backends: tuple[str, ...] = ()
    required_collectives: tuple[str, ...] = ()

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
        for name in (
            "maximum_device_bytes",
            "maximum_host_bytes",
            "maximum_compilation_cache_bytes",
            "maximum_halo_collective_bytes",
            "maximum_checkpoint_staging_bytes",
            "maximum_output_backlog_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _optional_byte_count(object.__getattribute__(self, name), name),
            )
        object.__setattr__(
            self,
            "required_dtypes",
            _identifiers(self.required_dtypes, "required dtype"),
        )
        object.__setattr__(
            self,
            "required_backends",
            _identifiers(self.required_backends, "required backend"),
        )
        object.__setattr__(
            self,
            "required_collectives",
            _identifiers(self.required_collectives, "required collective"),
        )

    @property
    def resource_id(self) -> str:
        return canonical_fingerprint(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
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
        if (
            self.maximum_device_bytes is not None
            or self.maximum_host_bytes is not None
            or self.maximum_compilation_cache_bytes is not None
            or self.maximum_halo_collective_bytes is not None
            or self.maximum_checkpoint_staging_bytes is not None
            or self.maximum_output_backlog_bytes is not None
            or self.required_dtypes
            or self.required_backends
            or self.required_collectives
        ):
            payload.update(
                {
                    "maximum_device_bytes": self.maximum_device_bytes,
                    "maximum_host_bytes": self.maximum_host_bytes,
                    "maximum_compilation_cache_bytes": (
                        self.maximum_compilation_cache_bytes
                    ),
                    "maximum_halo_collective_bytes": (self.maximum_halo_collective_bytes),
                    "maximum_checkpoint_staging_bytes": (
                        self.maximum_checkpoint_staging_bytes
                    ),
                    "maximum_output_backlog_bytes": (self.maximum_output_backlog_bytes),
                    "required_dtypes": list(self.required_dtypes),
                    "required_backends": list(self.required_backends),
                    "required_collectives": list(self.required_collectives),
                }
            )
        return payload

    @classmethod
    def from_payload(cls, value: Mapping[str, Any], /) -> ResourceRequest:
        return cls(
            value["cpu_cores"],
            value["memory_bytes"],
            accelerator_count=value.get("accelerator_count", 0),
            host_count=value.get("host_count", 1),
            process_count=value.get("process_count", 1),
            accelerator_platform=value.get("accelerator_platform"),
            accelerator_vendor=value.get("accelerator_vendor"),
            minimum_accelerator_memory_bytes=value.get(
                "minimum_accelerator_memory_bytes"
            ),
            exclusive=value.get("exclusive", True),
            maximum_device_bytes=value.get("maximum_device_bytes"),
            maximum_host_bytes=value.get("maximum_host_bytes"),
            maximum_compilation_cache_bytes=value.get("maximum_compilation_cache_bytes"),
            maximum_halo_collective_bytes=value.get("maximum_halo_collective_bytes"),
            maximum_checkpoint_staging_bytes=value.get(
                "maximum_checkpoint_staging_bytes"
            ),
            maximum_output_backlog_bytes=value.get("maximum_output_backlog_bytes"),
            required_dtypes=tuple(value.get("required_dtypes", ())),
            required_backends=tuple(value.get("required_backends", ())),
            required_collectives=tuple(value.get("required_collectives", ())),
        )


@dataclass(frozen=True, slots=True)
class ExecutionResourceEvidence:
    """Static candidate estimates and explicitly attested capabilities."""

    per_device_peak_bytes: int | None = None
    per_device_reserve_bytes: int | None = None
    per_host_peak_bytes: int | None = None
    per_host_reserve_bytes: int | None = None
    compilation_cache_bytes: int | None = None
    halo_collective_bytes: int | None = None
    checkpoint_staging_bytes: int | None = None
    output_backlog_bytes: int | None = None
    dtypes: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    collectives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "per_device_peak_bytes",
            "per_device_reserve_bytes",
            "per_host_peak_bytes",
            "per_host_reserve_bytes",
            "compilation_cache_bytes",
            "halo_collective_bytes",
            "checkpoint_staging_bytes",
            "output_backlog_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _optional_byte_count(object.__getattribute__(self, name), name),
            )
        object.__setattr__(self, "dtypes", _identifiers(self.dtypes, "dtype"))
        object.__setattr__(self, "backends", _identifiers(self.backends, "backend"))
        object.__setattr__(
            self,
            "collectives",
            _identifiers(self.collectives, "collective"),
        )

    @property
    def evidence_id(self) -> str:
        return canonical_fingerprint(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "per_device_peak_bytes": self.per_device_peak_bytes,
            "per_device_reserve_bytes": self.per_device_reserve_bytes,
            "per_host_peak_bytes": self.per_host_peak_bytes,
            "per_host_reserve_bytes": self.per_host_reserve_bytes,
            "compilation_cache_bytes": self.compilation_cache_bytes,
            "halo_collective_bytes": self.halo_collective_bytes,
            "checkpoint_staging_bytes": self.checkpoint_staging_bytes,
            "output_backlog_bytes": self.output_backlog_bytes,
            "dtypes": list(self.dtypes),
            "backends": list(self.backends),
            "collectives": list(self.collectives),
        }

    @classmethod
    def from_payload(
        cls,
        value: Mapping[str, Any],
        /,
    ) -> ExecutionResourceEvidence:
        return cls(
            per_device_peak_bytes=value.get("per_device_peak_bytes"),
            per_device_reserve_bytes=value.get("per_device_reserve_bytes"),
            per_host_peak_bytes=value.get("per_host_peak_bytes"),
            per_host_reserve_bytes=value.get("per_host_reserve_bytes"),
            compilation_cache_bytes=value.get("compilation_cache_bytes"),
            halo_collective_bytes=value.get("halo_collective_bytes"),
            checkpoint_staging_bytes=value.get("checkpoint_staging_bytes"),
            output_backlog_bytes=value.get("output_backlog_bytes"),
            dtypes=tuple(value.get("dtypes", ())),
            backends=tuple(value.get("backends", ())),
            collectives=tuple(value.get("collectives", ())),
        )


@dataclass(frozen=True, slots=True)
class DeviceResource:
    """Serializable description of one JAX-visible device."""

    process_index: int
    device_id: int
    local_device_id: int | None
    platform: str
    kind: str
    memory_bytes: int | None = None
    vendor: str | None = None

    def __post_init__(self) -> None:
        if self.process_index < 0 or self.device_id < 0:
            raise ValueError("process_index and device_id must be non-negative")
        if self.local_device_id is not None and self.local_device_id < 0:
            raise ValueError("local_device_id must be non-negative")
        if self.memory_bytes is not None and (
            isinstance(self.memory_bytes, bool)
            or not isinstance(self.memory_bytes, int)
            or self.memory_bytes <= 0
        ):
            raise ValueError("memory_bytes must be a positive integer or None")
        object.__setattr__(self, "platform", _identifier(self.platform, "platform"))
        object.__setattr__(self, "kind", _identifier(self.kind, "kind"))
        if self.vendor is not None:
            object.__setattr__(self, "vendor", _identifier(self.vendor, "vendor"))

    @property
    def key(self) -> tuple[int, int]:
        return (self.process_index, self.device_id)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "process_index": self.process_index,
            "device_id": self.device_id,
            "local_device_id": self.local_device_id,
            "platform": self.platform,
            "kind": self.kind,
        }
        if self.memory_bytes is not None:
            payload["memory_bytes"] = self.memory_bytes
        if self.vendor is not None:
            payload["vendor"] = self.vendor
        return payload


@dataclass(frozen=True, slots=True)
class ResourceInventory:
    """Observed resources in one initialized JAX runtime."""

    process_count: int
    process_index: int
    devices: tuple[DeviceResource, ...]
    process_host_ids: tuple[tuple[int, str], ...]

    def __init__(
        self,
        process_count: int,
        process_index: int,
        devices: Sequence[DeviceResource],
        *,
        process_host_ids: Mapping[int, str] | Sequence[tuple[int, str]] = (),
    ) -> None:
        devices_ = tuple(devices)
        hosts = _process_host_records(process_host_ids, "inventory")
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
        if hosts and tuple(process for process, _ in hosts) != tuple(
            range(process_count)
        ):
            raise ValueError(
                "inventory process_host_ids must map every process exactly once"
            )
        object.__setattr__(self, "process_count", process_count)
        object.__setattr__(self, "process_index", process_index)
        object.__setattr__(self, "devices", devices_)
        object.__setattr__(self, "process_host_ids", hosts)

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
        devices: list[dict[str, object]] = []
        for device in self.devices:
            record: dict[str, object] = {
                "process_index": device.process_index,
                "device_id": device.device_id,
                "platform": device.platform,
                "kind": device.kind,
            }
            if device.memory_bytes is not None:
                record["memory_bytes"] = device.memory_bytes
            if device.vendor is not None:
                record["vendor"] = device.vendor
            devices.append(record)
        payload: dict[str, object] = {
            "process_count": self.process_count,
            "devices": devices,
        }
        if self.process_host_ids:
            payload["process_host_ids"] = [
                [process, host] for process, host in self.process_host_ids
            ]
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionGroupSpec:
    """Serializable process/device reservation for one coupled computation."""

    group_id: str
    process_indices: tuple[int, ...]
    device_keys: tuple[tuple[int, int], ...]
    mesh_axes: tuple[tuple[str, int], ...] = ()
    parent_group_id: str | None = None
    process_host_ids: tuple[tuple[int, str], ...] = ()

    def __init__(
        self,
        group_id: str,
        process_indices: Sequence[int],
        device_keys: Sequence[tuple[int, int]],
        *,
        mesh_axes: Sequence[tuple[str, int]] = (),
        parent_group_id: str | None = None,
        process_host_ids: Mapping[int, str] | Sequence[tuple[int, str]] = (),
    ) -> None:
        group = _identifier(group_id, "group_id")
        processes = tuple(process_indices)
        devices = tuple((int(process), int(device)) for process, device in device_keys)
        axes = tuple(
            (_identifier(name, "mesh axis"), int(size)) for name, size in mesh_axes
        )
        hosts = _process_host_records(process_host_ids, "group")
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
        if hosts and tuple(process for process, _ in hosts) != tuple(sorted(processes)):
            raise ValueError("group process_host_ids must map every group process")
        object.__setattr__(self, "group_id", group)
        object.__setattr__(self, "process_indices", processes)
        object.__setattr__(self, "device_keys", devices)
        object.__setattr__(self, "mesh_axes", axes)
        object.__setattr__(self, "process_host_ids", hosts)
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
        payload: dict[str, object] = {
            "group_id": self.group_id,
            "process_indices": list(self.process_indices),
            "device_keys": [list(key) for key in self.device_keys],
            "mesh_axes": [[name, size] for name, size in self.mesh_axes],
            "parent_group_id": self.parent_group_id,
        }
        if self.process_host_ids:
            payload["process_host_ids"] = [
                [process, host] for process, host in self.process_host_ids
            ]
        return payload


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
    "ExecutionResourceEvidence",
    "RecoveryPolicy",
    "ResourceInventory",
    "ResourceRequest",
)
