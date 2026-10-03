#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-neutral host, process, device, and compiler memory measurements.

Each quantity is a distinct measurement with its own provenance: process resident
set size (RSS), the kernel-maintained RSS high-water mark, host physical and
available memory, device allocator limit/reservation/live allocation counters,
JAX-visible live array bytes, and XLA compiled-memory estimates are never merged
into one number. Unavailable quantities are explicit records with a reason; a
missing probe never becomes zero or free memory.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from types import TracebackType
from typing import assert_never, Literal, TypeAlias

import jax

from ._validation import canonical_identifier
from .typing import parse


MemoryMeasurementScope: TypeAlias = Literal["baseline", "peak"]
"""`baseline` is an instantaneous value; `peak` is a maximum over a window."""

MemoryMeasurementMethod: TypeAlias = Literal[
    "getrusage_max_rss",
    "procfs_statm",
    "mach_task_basic_info",
    "sysconf_physical_pages",
    "procfs_meminfo_available",
    "mach_host_vm_statistics",
    "jax_memory_stats",
    "jax_live_arrays",
    "interval_sampling",
]
HostMemoryProvider: TypeAlias = Literal["linux_procfs", "darwin_mach", "unsupported"]

_MACH_TASK_BASIC_INFO = 20
_HOST_VM_INFO64 = 4
_KIB = 1024


def _byte_count(value: object, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _optional_byte_count(value: object, name: str, /) -> int | None:
    return None if value is None else _byte_count(value, name)


def _require_measurement(value: object, name: str, /) -> None:
    if not isinstance(value, MemoryMeasurement):
        raise TypeError(f"{name} must be a MemoryMeasurement.")


@dataclass(frozen=True, slots=True)
class MemoryMeasurement:
    """One memory quantity with its method, scope, and resolution, or its absence.

    ``value_bytes`` and ``resolution_bytes`` are both present for a measured value
    and both ``None`` when ``unavailable_reason`` explains why the provider cannot
    report the quantity. ``resolution_bytes`` is the granularity of the source
    counter, not a statistical error.
    """

    method: MemoryMeasurementMethod
    scope: MemoryMeasurementScope
    value_bytes: int | None
    resolution_bytes: int | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        parse(self.method, MemoryMeasurementMethod, "method")
        parse(self.scope, MemoryMeasurementScope, "scope")
        _optional_byte_count(self.value_bytes, "value_bytes")
        _optional_byte_count(self.resolution_bytes, "resolution_bytes")
        if self.value_bytes is None:
            if self.resolution_bytes is not None:
                raise ValueError("An unavailable measurement has no resolution.")
            if not isinstance(self.unavailable_reason, str) or not (
                self.unavailable_reason.strip()
            ):
                raise ValueError("An unavailable measurement requires a reason.")
        else:
            if self.resolution_bytes is None or self.resolution_bytes == 0:
                raise ValueError("A measured value requires a positive resolution.")
            if self.unavailable_reason is not None:
                raise ValueError("A measured value cannot carry an unavailable reason.")

    @classmethod
    def measured(
        cls,
        method: MemoryMeasurementMethod,
        scope: MemoryMeasurementScope,
        value_bytes: int,
        resolution_bytes: int,
        /,
    ) -> MemoryMeasurement:
        return cls(method, scope, value_bytes, resolution_bytes)

    @classmethod
    def unavailable(
        cls,
        method: MemoryMeasurementMethod,
        scope: MemoryMeasurementScope,
        reason: str,
        /,
    ) -> MemoryMeasurement:
        return cls(method, scope, None, None, reason)

    @property
    def available(self) -> bool:
        return self.value_bytes is not None

    @property
    def is_upper_bound(self) -> bool:
        """Whether this value bounds every instantaneous value in its window.

        Kernel and allocator high-water counters are maintained on every change,
        so they bound the true peak. A maximum over discrete samples misses
        transients between samples and is never an upper bound.
        """

        if self.value_bytes is None or self.scope != "peak":
            return False
        match self.method:
            case "getrusage_max_rss" | "jax_memory_stats":
                return True
            case (
                "interval_sampling"
                | "procfs_statm"
                | "mach_task_basic_info"
                | "sysconf_physical_pages"
                | "procfs_meminfo_available"
                | "mach_host_vm_statistics"
                | "jax_live_arrays"
            ):
                return False
            case _:
                assert_never(self.method)

    def to_payload(self) -> dict[str, object]:
        return {
            "method": self.method,
            "scope": self.scope,
            "value_bytes": self.value_bytes,
            "resolution_bytes": self.resolution_bytes,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class HostMemorySample:
    """Process RSS, RSS high-water mark, and host memory at one instant.

    ``process_peak_resident`` is the kernel high-water mark since process start;
    it bounds the RSS peak of every phase that ran in this process.
    ``host_available`` is the provider's reclaimable-memory estimate
    (Linux ``MemAvailable``; Darwin free plus inactive pages).
    """

    provider: HostMemoryProvider
    process_index: int
    timestamp_ns: int
    process_resident: MemoryMeasurement
    process_peak_resident: MemoryMeasurement
    host_total: MemoryMeasurement
    host_available: MemoryMeasurement

    def __post_init__(self) -> None:
        parse(self.provider, HostMemoryProvider, "provider")
        _byte_count(self.process_index, "process_index")
        _byte_count(self.timestamp_ns, "timestamp_ns")
        for name in (
            "process_resident",
            "process_peak_resident",
            "host_total",
            "host_available",
        ):
            _require_measurement(object.__getattribute__(self, name), name)

    def to_payload(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "process_index": self.process_index,
            "timestamp_ns": self.timestamp_ns,
            "process_resident": self.process_resident.to_payload(),
            "process_peak_resident": self.process_peak_resident.to_payload(),
            "host_total": self.host_total.to_payload(),
            "host_available": self.host_available.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class DeviceMemorySample:
    """Allocator counters and JAX live-array bytes for one addressable device.

    Allocator ``limit``, ``reserved`` (pool bytes obtained from the device),
    ``in_use`` (live allocations), and ``peak_in_use`` (allocator high-water
    mark) come from the PJRT provider when it exposes them. ``live_array_bytes``
    counts only buffers of live JAX arrays on this device.
    """

    process_index: int
    device_id: int
    platform: str
    provider_id: str
    provider_version: str
    timestamp_ns: int
    allocator_limit: MemoryMeasurement
    allocator_reserved: MemoryMeasurement
    allocator_in_use: MemoryMeasurement
    allocator_peak_in_use: MemoryMeasurement
    live_array_bytes: MemoryMeasurement

    def __post_init__(self) -> None:
        _byte_count(self.process_index, "process_index")
        _byte_count(self.device_id, "device_id")
        canonical_identifier(self.platform, "platform")
        canonical_identifier(self.provider_id, "provider_id")
        canonical_identifier(self.provider_version, "provider_version")
        _byte_count(self.timestamp_ns, "timestamp_ns")
        for name in (
            "allocator_limit",
            "allocator_reserved",
            "allocator_in_use",
            "allocator_peak_in_use",
            "live_array_bytes",
        ):
            _require_measurement(object.__getattribute__(self, name), name)

    @property
    def key(self) -> tuple[int, int]:
        return (self.process_index, self.device_id)

    @property
    def allocator_headroom_bytes(self) -> int | None:
        """Allocator limit minus live allocations, or ``None`` when either is unknown."""

        limit = self.allocator_limit.value_bytes
        in_use = self.allocator_in_use.value_bytes
        if limit is None or in_use is None:
            return None
        return max(limit - in_use, 0)

    def to_payload(self) -> dict[str, object]:
        return {
            "process_index": self.process_index,
            "device_id": self.device_id,
            "platform": self.platform,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "timestamp_ns": self.timestamp_ns,
            "allocator_limit": self.allocator_limit.to_payload(),
            "allocator_reserved": self.allocator_reserved.to_payload(),
            "allocator_in_use": self.allocator_in_use.to_payload(),
            "allocator_peak_in_use": self.allocator_peak_in_use.to_payload(),
            "live_array_bytes": self.live_array_bytes.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class CompiledMemoryEstimate:
    """XLA buffer-assignment memory estimate of one compiled executable.

    This is a compiler plan, not a measurement: it excludes allocator
    fragmentation, other live buffers, and runtime-provider workspaces.
    """

    argument_bytes: int
    output_bytes: int
    alias_bytes: int
    temporary_bytes: int
    generated_code_bytes: int
    host_argument_bytes: int
    host_output_bytes: int
    host_alias_bytes: int
    host_temporary_bytes: int
    host_generated_code_bytes: int
    peak_bytes: int | None

    def __post_init__(self) -> None:
        for name in (
            "argument_bytes",
            "output_bytes",
            "alias_bytes",
            "temporary_bytes",
            "generated_code_bytes",
            "host_argument_bytes",
            "host_output_bytes",
            "host_alias_bytes",
            "host_temporary_bytes",
            "host_generated_code_bytes",
        ):
            _byte_count(object.__getattribute__(self, name), name)
        _optional_byte_count(self.peak_bytes, "peak_bytes")
        if self.alias_bytes > self.argument_bytes + self.output_bytes:
            raise ValueError("alias_bytes cannot exceed argument plus output bytes.")

    @property
    def device_bytes(self) -> int:
        """Arguments, outputs, temporaries, and code, counting aliased buffers once."""

        return (
            self.argument_bytes
            + self.output_bytes
            - self.alias_bytes
            + self.temporary_bytes
            + self.generated_code_bytes
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "argument_bytes": self.argument_bytes,
            "output_bytes": self.output_bytes,
            "alias_bytes": self.alias_bytes,
            "temporary_bytes": self.temporary_bytes,
            "generated_code_bytes": self.generated_code_bytes,
            "host_argument_bytes": self.host_argument_bytes,
            "host_output_bytes": self.host_output_bytes,
            "host_alias_bytes": self.host_alias_bytes,
            "host_temporary_bytes": self.host_temporary_bytes,
            "host_generated_code_bytes": self.host_generated_code_bytes,
            "peak_bytes": self.peak_bytes,
        }


@dataclass(frozen=True, slots=True)
class DevicePhaseMemory:
    """Device allocator evidence bracketing one sampled phase."""

    baseline: DeviceMemorySample
    final: DeviceMemorySample
    sampled_peak_in_use: MemoryMeasurement

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, DeviceMemorySample) or not isinstance(
            self.final, DeviceMemorySample
        ):
            raise TypeError("baseline and final must be DeviceMemorySample values.")
        if self.baseline.key != self.final.key:
            raise ValueError("baseline and final samples must describe one device.")
        _require_measurement(self.sampled_peak_in_use, "sampled_peak_in_use")

    @property
    def key(self) -> tuple[int, int]:
        return self.baseline.key

    def to_payload(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_payload(),
            "final": self.final.to_payload(),
            "sampled_peak_in_use": self.sampled_peak_in_use.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class PhaseMemoryEvidence:
    """Bracketing samples and the sampled peak of one measured phase.

    ``sampled_peak_resident`` is the maximum RSS over ``sample_count`` discrete
    samples at the declared interval; transients between samples are missed, so
    it is reported as sampled and is never an upper bound. ``interval_peak_resident``
    is exact (to counter resolution) only when the kernel high-water mark rose
    during the phase; ``resident_peak_upper_bound`` always bounds the phase peak.
    """

    phase: str
    interval_seconds: float
    duration_seconds: float
    sample_count: int
    maximum_sample_gap_seconds: float
    baseline: HostMemorySample
    final: HostMemorySample
    sampled_peak_resident: MemoryMeasurement
    devices: tuple[DevicePhaseMemory, ...] = ()

    def __post_init__(self) -> None:
        canonical_identifier(self.phase, "phase")
        for name in (
            "interval_seconds",
            "duration_seconds",
            "maximum_sample_gap_seconds",
        ):
            value = object.__getattribute__(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, float)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative float.")
        if self.interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be positive.")
        if _byte_count(self.sample_count, "sample_count") < 2:
            raise ValueError("A phase is bracketed by at least two samples.")
        if not isinstance(self.baseline, HostMemorySample) or not isinstance(
            self.final, HostMemorySample
        ):
            raise TypeError("baseline and final must be HostMemorySample values.")
        _require_measurement(self.sampled_peak_resident, "sampled_peak_resident")
        if not isinstance(self.devices, tuple) or any(
            not isinstance(device, DevicePhaseMemory) for device in self.devices
        ):
            raise TypeError("devices must be a tuple of DevicePhaseMemory values.")
        if len({device.key for device in self.devices}) != len(self.devices):
            raise ValueError("device phase records must have unique device keys.")

    @property
    def sampled_peak_increase_bytes(self) -> int | None:
        """Sampled peak RSS above the baseline RSS, when both are measured."""

        peak = self.sampled_peak_resident.value_bytes
        baseline = self.baseline.process_resident.value_bytes
        if peak is None or baseline is None:
            return None
        return max(peak - baseline, 0)

    @property
    def resident_peak_upper_bound(self) -> MemoryMeasurement:
        """Kernel RSS high-water mark at phase end; bounds the phase peak."""

        return self.final.process_peak_resident

    @property
    def interval_peak_resident(self) -> MemoryMeasurement:
        """Exact phase RSS peak when the process high-water mark rose during it."""

        before = self.baseline.process_peak_resident.value_bytes
        after = self.final.process_peak_resident.value_bytes
        if before is None or after is None:
            return MemoryMeasurement.unavailable(
                "getrusage_max_rss",
                "peak",
                "the provider does not report the process RSS high-water mark",
            )
        if after <= before:
            return MemoryMeasurement.unavailable(
                "getrusage_max_rss",
                "peak",
                "the process RSS high-water mark predates the phase; "
                "only its upper bound is known",
            )
        return self.final.process_peak_resident

    def to_payload(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "interval_seconds": self.interval_seconds,
            "duration_seconds": self.duration_seconds,
            "sample_count": self.sample_count,
            "maximum_sample_gap_seconds": self.maximum_sample_gap_seconds,
            "baseline": self.baseline.to_payload(),
            "final": self.final.to_payload(),
            "sampled_peak_resident": self.sampled_peak_resident.to_payload(),
            "interval_peak_resident": self.interval_peak_resident.to_payload(),
            "devices": [device.to_payload() for device in self.devices],
        }


class _MachTaskBasicInfo(ctypes.Structure):
    _fields_ = (
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", ctypes.c_int32 * 2),
        ("system_time", ctypes.c_int32 * 2),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    )


class _MachVmStatistics64(ctypes.Structure):
    _fields_ = (
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    )


@cache
def _darwin_system() -> ctypes.CDLL:
    library = ctypes.CDLL(None)
    count = ctypes.POINTER(ctypes.c_uint32)
    library.task_info.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, count)
    library.task_info.restype = ctypes.c_int
    library.mach_host_self.argtypes = ()
    library.mach_host_self.restype = ctypes.c_uint32
    library.host_statistics64.argtypes = (
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_void_p,
        count,
    )
    library.host_statistics64.restype = ctypes.c_int
    library.host_page_size.argtypes = (ctypes.c_uint32, ctypes.POINTER(ctypes.c_size_t))
    library.host_page_size.restype = ctypes.c_int
    library.mach_port_deallocate.argtypes = (ctypes.c_uint32, ctypes.c_uint32)
    library.mach_port_deallocate.restype = ctypes.c_int
    return library


def _host_provider() -> HostMemoryProvider:
    if sys.platform.startswith("linux"):
        return "linux_procfs"
    if sys.platform == "darwin":
        return "darwin_mach"
    return "unsupported"


def _page_size() -> int:
    return os.sysconf("SC_PAGE_SIZE")


def _current_resident(provider: HostMemoryProvider, /) -> MemoryMeasurement:
    match provider:
        case "linux_procfs":
            try:
                with open("/proc/self/statm", encoding="ascii") as stream:
                    fields = stream.read().split()
            except OSError as error:
                return MemoryMeasurement.unavailable(
                    "procfs_statm", "baseline", f"/proc/self/statm unreadable: {error}"
                )
            page = _page_size()
            return MemoryMeasurement.measured(
                "procfs_statm", "baseline", int(fields[1]) * page, page
            )
        case "darwin_mach":
            library = _darwin_system()
            info = _MachTaskBasicInfo()
            count = ctypes.c_uint32(ctypes.sizeof(_MachTaskBasicInfo) // 4)
            task = ctypes.c_uint32.in_dll(library, "mach_task_self_").value
            status = library.task_info(
                task, _MACH_TASK_BASIC_INFO, ctypes.byref(info), ctypes.byref(count)
            )
            if status != 0:
                return MemoryMeasurement.unavailable(
                    "mach_task_basic_info",
                    "baseline",
                    f"task_info returned kern_return_t {status}",
                )
            return MemoryMeasurement.measured(
                "mach_task_basic_info", "baseline", info.resident_size, _page_size()
            )
        case "unsupported":
            return MemoryMeasurement.unavailable(
                "procfs_statm",
                "baseline",
                f"process RSS is not exposed on platform {sys.platform!r}",
            )
        case _:
            assert_never(provider)


def _peak_resident(provider: HostMemoryProvider, /) -> MemoryMeasurement:
    match provider:
        case "linux_procfs" | "darwin_mach":
            import resource

            maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Darwin reports ru_maxrss in bytes and Linux in KiB.
            unit = 1 if provider == "darwin_mach" else _KIB
            return MemoryMeasurement.measured(
                "getrusage_max_rss", "peak", maximum * unit, unit
            )
        case "unsupported":
            return MemoryMeasurement.unavailable(
                "getrusage_max_rss",
                "peak",
                f"getrusage is not available on platform {sys.platform!r}",
            )
        case _:
            assert_never(provider)


def _host_total(provider: HostMemoryProvider, /) -> MemoryMeasurement:
    match provider:
        case "linux_procfs" | "darwin_mach":
            page = _page_size()
            return MemoryMeasurement.measured(
                "sysconf_physical_pages",
                "baseline",
                os.sysconf("SC_PHYS_PAGES") * page,
                page,
            )
        case "unsupported":
            return MemoryMeasurement.unavailable(
                "sysconf_physical_pages",
                "baseline",
                f"physical memory is not exposed on platform {sys.platform!r}",
            )
        case _:
            assert_never(provider)


def _linux_available() -> MemoryMeasurement:
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            lines = stream.read().splitlines()
    except OSError as error:
        return MemoryMeasurement.unavailable(
            "procfs_meminfo_available", "baseline", f"/proc/meminfo unreadable: {error}"
        )
    for line in lines:
        name, _, value = line.partition(":")
        if name == "MemAvailable":
            amount, unit = value.split()
            if unit != "kB":
                raise ValueError(f"/proc/meminfo MemAvailable has unit {unit!r}.")
            return MemoryMeasurement.measured(
                "procfs_meminfo_available", "baseline", int(amount) * _KIB, _KIB
            )
    return MemoryMeasurement.unavailable(
        "procfs_meminfo_available",
        "baseline",
        "the kernel does not report MemAvailable",
    )


def _darwin_available() -> MemoryMeasurement:
    library = _darwin_system()
    host = library.mach_host_self()
    try:
        page = ctypes.c_size_t()
        page_status = library.host_page_size(host, ctypes.byref(page))
        statistics = _MachVmStatistics64()
        count = ctypes.c_uint32(ctypes.sizeof(_MachVmStatistics64) // 4)
        status = library.host_statistics64(
            host, _HOST_VM_INFO64, ctypes.byref(statistics), ctypes.byref(count)
        )
    finally:
        task = ctypes.c_uint32.in_dll(library, "mach_task_self_").value
        library.mach_port_deallocate(task, host)
    if page_status != 0 or status != 0:
        return MemoryMeasurement.unavailable(
            "mach_host_vm_statistics",
            "baseline",
            f"host VM statistics returned kern_return_t {status or page_status}",
        )
    # free_count already includes speculative pages; inactive pages are
    # reclaimable without swapping. Purgeable and file-backed active pages are
    # excluded, so the estimate is conservative.
    pages = statistics.free_count + statistics.inactive_count
    return MemoryMeasurement.measured(
        "mach_host_vm_statistics", "baseline", pages * page.value, page.value
    )


def _host_available(provider: HostMemoryProvider, /) -> MemoryMeasurement:
    match provider:
        case "linux_procfs":
            return _linux_available()
        case "darwin_mach":
            return _darwin_available()
        case "unsupported":
            return MemoryMeasurement.unavailable(
                "procfs_meminfo_available",
                "baseline",
                f"available memory is not exposed on platform {sys.platform!r}",
            )
        case _:
            assert_never(provider)


def sample_host_memory() -> HostMemorySample:
    """Measure this process's RSS and RSS high-water mark and host memory now."""

    provider = _host_provider()
    return HostMemorySample(
        provider,
        jax.process_index(),
        time.time_ns(),
        _current_resident(provider),
        _peak_resident(provider),
        _host_total(provider),
        _host_available(provider),
    )


def _addressable_devices(devices: Sequence[jax.Device], /) -> tuple[jax.Device, ...]:
    if isinstance(devices, (str, bytes)):
        raise TypeError("devices must be a sequence of jax.Device values.")
    devices_ = tuple(devices)
    process = jax.process_index()
    for device in devices_:
        if not isinstance(device, jax.Device):
            raise TypeError("devices must contain jax.Device values.")
        if device.process_index != process:
            raise ValueError("memory can be sampled only on addressable devices.")
    if len({device.id for device in devices_}) != len(devices_):
        raise ValueError("devices must be unique.")
    return devices_


def _allocator_measurement(
    statistics: dict[str, int] | None,
    key: str,
    scope: MemoryMeasurementScope,
    platform: str,
    /,
) -> MemoryMeasurement:
    if statistics is None:
        return MemoryMeasurement.unavailable(
            "jax_memory_stats",
            scope,
            f"the JAX {platform} provider does not expose allocator statistics",
        )
    value = statistics.get(key)
    if value is None:
        return MemoryMeasurement.unavailable(
            "jax_memory_stats",
            scope,
            f"the JAX {platform} allocator does not report {key!r}",
        )
    return MemoryMeasurement.measured("jax_memory_stats", scope, value, 1)


def sample_device_memory(
    devices: Sequence[jax.Device],
) -> tuple[DeviceMemorySample, ...]:
    """Measure allocator counters and live JAX array bytes on addressable devices."""

    devices_ = _addressable_devices(devices)
    live = {device.id: 0 for device in devices_}
    for array in jax.live_arrays():
        for shard in array.addressable_shards:
            if shard.device.id in live:
                live[shard.device.id] += shard.data.nbytes
    samples: list[DeviceMemorySample] = []
    for device in devices_:
        statistics = device.memory_stats()
        platform = device.platform
        samples.append(
            DeviceMemorySample(
                device.process_index,
                device.id,
                platform,
                device.client.platform,
                device.client.platform_version,
                time.time_ns(),
                _allocator_measurement(statistics, "bytes_limit", "baseline", platform),
                _allocator_measurement(statistics, "pool_bytes", "baseline", platform),
                _allocator_measurement(statistics, "bytes_in_use", "baseline", platform),
                _allocator_measurement(statistics, "peak_bytes_in_use", "peak", platform),
                MemoryMeasurement.measured(
                    "jax_live_arrays", "baseline", live[device.id], 1
                ),
            )
        )
    return tuple(samples)


def compiled_memory_estimate(
    compiled: jax.stages.Compiled,
) -> CompiledMemoryEstimate | None:
    """Return XLA's memory analysis, or ``None`` when the provider exposes none."""

    if not isinstance(compiled, jax.stages.Compiled):
        raise TypeError("compiled must be a jax.stages.Compiled executable.")
    statistics = compiled.memory_analysis()
    if statistics is None:
        return None
    peak = statistics.peak_memory_in_bytes
    return CompiledMemoryEstimate(
        argument_bytes=statistics.argument_size_in_bytes,
        output_bytes=statistics.output_size_in_bytes,
        alias_bytes=statistics.alias_size_in_bytes,
        temporary_bytes=statistics.temp_size_in_bytes,
        generated_code_bytes=statistics.generated_code_size_in_bytes,
        host_argument_bytes=statistics.host_argument_size_in_bytes,
        host_output_bytes=statistics.host_output_size_in_bytes,
        host_alias_bytes=statistics.host_alias_size_in_bytes,
        host_temporary_bytes=statistics.host_temp_size_in_bytes,
        host_generated_code_bytes=statistics.host_generated_code_size_in_bytes,
        peak_bytes=peak if peak > 0 else None,
    )


class PhaseMemorySampler:
    """Single-use context that samples RSS on a background thread during a phase.

    State is constant-size (running maxima, count, and largest gap), so the
    sampler is bounded for arbitrarily long phases. Device allocator usage is
    sampled at the same cadence for the given addressable devices.
    """

    __slots__ = (
        "_baseline",
        "_device_baseline",
        "_device_peaks",
        "_devices",
        "_evidence",
        "_failure",
        "_interval",
        "_last",
        "_maximum_gap",
        "_peak",
        "_phase",
        "_provider",
        "_sample_count",
        "_started",
        "_stop",
        "_thread",
    )

    def __init__(
        self,
        phase: str,
        *,
        interval_seconds: float,
        devices: Sequence[jax.Device] = (),
    ) -> None:
        canonical_identifier(phase, "phase")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0.0
        ):
            raise ValueError("interval_seconds must be a finite positive number.")
        self._phase = phase
        self._interval = float(interval_seconds)
        self._devices = _addressable_devices(devices)
        self._provider = _host_provider()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._evidence: PhaseMemoryEvidence | None = None
        self._failure: Exception | None = None
        self._baseline: HostMemorySample | None = None
        self._device_baseline: tuple[DeviceMemorySample, ...] = ()
        self._device_peaks: dict[int, int | None] = {}
        self._peak: int | None = None
        self._sample_count = 0
        self._maximum_gap = 0.0
        self._started = 0.0
        self._last = 0.0

    def __enter__(self) -> PhaseMemorySampler:
        if self._thread is not None:
            raise RuntimeError("PhaseMemorySampler is single-use.")
        self._baseline = sample_host_memory()
        self._device_baseline = sample_device_memory(self._devices)
        self._peak = self._baseline.process_resident.value_bytes
        self._device_peaks = {
            sample.device_id: sample.allocator_in_use.value_bytes
            for sample in self._device_baseline
        }
        self._sample_count = 1
        self._started = time.perf_counter()
        self._last = self._started
        self._thread = threading.Thread(
            target=self._run, name=f"phydrax-memory-{self._phase}", daemon=True
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self._stop.wait(self._interval):
                self._record()
        except Exception as error:  # Re-raised by __exit__ on the caller thread.
            self._failure = error

    def _record(self) -> None:
        resident = _current_resident(self._provider).value_bytes
        now = time.perf_counter()
        self._maximum_gap = max(self._maximum_gap, now - self._last)
        self._last = now
        self._sample_count += 1
        if resident is not None and self._peak is not None:
            self._peak = max(self._peak, resident)
        for device in self._devices:
            previous = self._device_peaks[device.id]
            statistics = device.memory_stats()
            if previous is not None and statistics is not None:
                self._device_peaks[device.id] = max(previous, statistics["bytes_in_use"])

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        thread = self._thread
        baseline = self._baseline
        if thread is None or baseline is None or self._evidence is not None:
            raise RuntimeError("PhaseMemorySampler exited without an active phase.")
        self._stop.set()
        thread.join()
        if self._failure is not None:
            if exc is None:
                raise RuntimeError(
                    f"memory sampling of phase {self._phase!r} failed"
                ) from self._failure
            return
        self._record()
        final = sample_host_memory()
        device_final = {
            sample.device_id: sample for sample in sample_device_memory(self._devices)
        }
        self._evidence = PhaseMemoryEvidence(
            phase=self._phase,
            interval_seconds=self._interval,
            duration_seconds=self._last - self._started,
            sample_count=self._sample_count,
            maximum_sample_gap_seconds=self._maximum_gap,
            baseline=baseline,
            final=final,
            sampled_peak_resident=self._sampled_peak(
                self._peak, baseline.process_resident
            ),
            devices=tuple(
                DevicePhaseMemory(
                    sample,
                    device_final[sample.device_id],
                    self._sampled_peak(
                        self._device_peaks[sample.device_id], sample.allocator_in_use
                    ),
                )
                for sample in self._device_baseline
            ),
        )

    def _sampled_peak(
        self, peak: int | None, baseline: MemoryMeasurement, /
    ) -> MemoryMeasurement:
        if peak is None or baseline.resolution_bytes is None:
            return MemoryMeasurement.unavailable(
                "interval_sampling",
                "peak",
                f"baseline {baseline.method} is unavailable: {baseline.unavailable_reason}",
            )
        return MemoryMeasurement.measured(
            "interval_sampling", "peak", peak, baseline.resolution_bytes
        )

    @property
    def evidence(self) -> PhaseMemoryEvidence:
        if self._evidence is None:
            raise RuntimeError("The sampled phase has not completed.")
        return self._evidence


__all__ = (
    "CompiledMemoryEstimate",
    "DeviceMemorySample",
    "DevicePhaseMemory",
    "HostMemoryProvider",
    "HostMemorySample",
    "MemoryMeasurement",
    "MemoryMeasurementMethod",
    "MemoryMeasurementScope",
    "PhaseMemoryEvidence",
    "PhaseMemorySampler",
    "compiled_memory_estimate",
    "sample_device_memory",
    "sample_host_memory",
)
