#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Process-global JAX runtime bootstrap for local and distributed execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Literal

import jax


_PHYDRAX_COORDINATOR_ADDRESS: Final = "PHYDRAX_COORDINATOR_ADDRESS"
_PHYDRAX_COORDINATOR_BIND_ADDRESS: Final = "PHYDRAX_COORDINATOR_BIND_ADDRESS"
_PHYDRAX_NUM_PROCESSES: Final = "PHYDRAX_NUM_PROCESSES"
_PHYDRAX_PROCESS_ID: Final = "PHYDRAX_PROCESS_ID"
_PHYDRAX_LOCAL_DEVICE_IDS: Final = "PHYDRAX_LOCAL_DEVICE_IDS"
_PHYDRAX_CPU_COLLECTIVES: Final = "PHYDRAX_CPU_COLLECTIVES"

_CLUSTER_SIZE_ENVIRONMENTS: Final = (
    "SLURM_NTASKS",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
)


@dataclass(frozen=True, slots=True)
class RuntimeBootstrap:
    """Configuration consumed before JAX initializes any device backend."""

    coordinator_address: str | None = None
    num_processes: int | None = None
    process_id: int | None = None
    local_device_ids: tuple[int, ...] | None = None
    cluster_detection_method: str | None = None
    autodetect: bool = False
    coordinator_bind_address: str | None = None
    heartbeat_timeout_seconds: int = 100
    initialization_timeout_seconds: int = 300
    cpu_collectives_implementation: Literal["gloo", "mpi"] | None = None

    def __post_init__(self) -> None:
        explicit = (
            self.coordinator_address is not None,
            self.num_processes is not None,
            self.process_id is not None,
        )
        if any(explicit) and not all(explicit):
            raise ValueError(
                "coordinator_address, num_processes, and process_id must be "
                "provided together"
            )
        selection_count = sum(
            (
                self.coordinator_address is not None,
                self.cluster_detection_method is not None,
                self.autodetect,
            )
        )
        if selection_count > 1:
            raise ValueError(
                "explicit coordinator configuration, cluster_detection_method, "
                "and autodetect are mutually exclusive"
            )
        if self.num_processes is not None and self.num_processes < 1:
            raise ValueError("num_processes must be positive")
        if self.process_id is not None:
            assert self.num_processes is not None
            if not 0 <= self.process_id < self.num_processes:
                raise ValueError("process_id must be in [0, num_processes)")
        if self.local_device_ids is not None:
            if not self.local_device_ids:
                raise ValueError("local_device_ids cannot be empty")
            if min(self.local_device_ids) < 0:
                raise ValueError("local_device_ids must be non-negative")
            if len(set(self.local_device_ids)) != len(self.local_device_ids):
                raise ValueError("local_device_ids must be unique")
        if self.heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat_timeout_seconds must be positive")
        if self.initialization_timeout_seconds <= 0:
            raise ValueError("initialization_timeout_seconds must be positive")
        if self.cpu_collectives_implementation not in (None, "gloo", "mpi"):
            raise ValueError(
                "cpu_collectives_implementation must be 'gloo', 'mpi', or None"
            )

    @property
    def requests_distributed_initialization(self) -> bool:
        return (
            self.coordinator_address is not None
            or self.cluster_detection_method is not None
            or self.autodetect
        )

    @classmethod
    def from_environment(cls) -> RuntimeBootstrap | None:
        """Resolve explicit PHYDRAX rank settings or a launcher-detected runtime."""

        coordinator = os.environ.get(_PHYDRAX_COORDINATOR_ADDRESS)
        num_processes = os.environ.get(_PHYDRAX_NUM_PROCESSES)
        process_id = os.environ.get(_PHYDRAX_PROCESS_ID)
        local_device_ids = os.environ.get(_PHYDRAX_LOCAL_DEVICE_IDS)
        bind_address = os.environ.get(_PHYDRAX_COORDINATOR_BIND_ADDRESS)
        cpu_collectives = os.environ.get(_PHYDRAX_CPU_COLLECTIVES)

        explicit_values = (coordinator, num_processes, process_id)
        if any(value is not None for value in explicit_values):
            if not all(value is not None for value in explicit_values):
                raise ValueError(
                    "PHYDRAX_COORDINATOR_ADDRESS, PHYDRAX_NUM_PROCESSES, and "
                    "PHYDRAX_PROCESS_ID must be set together"
                )
            parsed_device_ids = None
            if local_device_ids:
                parsed_device_ids = tuple(
                    int(value.strip()) for value in local_device_ids.split(",")
                )
            assert coordinator is not None
            assert num_processes is not None
            assert process_id is not None
            return cls(
                coordinator_address=coordinator,
                num_processes=int(num_processes),
                process_id=int(process_id),
                local_device_ids=parsed_device_ids,
                coordinator_bind_address=bind_address,
                cpu_collectives_implementation=cpu_collectives,
            )

        for variable in _CLUSTER_SIZE_ENVIRONMENTS:
            raw_size = os.environ.get(variable)
            if raw_size is not None and int(raw_size) > 1:
                return cls(
                    autodetect=True,
                    cpu_collectives_implementation=cpu_collectives,
                )
        if cpu_collectives is not None:
            return cls(cpu_collectives_implementation=cpu_collectives)
        return None


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """Observed immutable facts about the initialized JAX runtime."""

    distributed_initialized: bool
    process_count: int
    process_index: int
    local_device_count: int
    global_device_count: int
    local_device_ids: tuple[int, ...]
    platforms: tuple[str, ...]

    @property
    def is_multi_process(self) -> bool:
        return self.process_count > 1

    @property
    def is_multi_device(self) -> bool:
        return self.global_device_count > 1


def runtime_info() -> RuntimeInfo:
    """Return runtime facts, initializing the local backend if necessary."""

    local_devices = tuple(jax.local_devices())
    global_devices = tuple(jax.devices())
    return RuntimeInfo(
        distributed_initialized=jax.distributed.is_initialized(),
        process_count=jax.process_count(),
        process_index=jax.process_index(),
        local_device_count=len(local_devices),
        global_device_count=len(global_devices),
        local_device_ids=tuple(
            device.id
            if device.local_hardware_id is None
            else int(device.local_hardware_id)
            for device in local_devices
        ),
        platforms=tuple(sorted({device.platform for device in global_devices})),
    )


def initialize(bootstrap: RuntimeBootstrap | None = None) -> RuntimeInfo:
    """Initialize a configured distributed runtime, or observe a local runtime.

    This function must be called before any JAX device computation when a
    distributed bootstrap is requested. Calling it without a bootstrap is a
    local no-op followed by runtime discovery.
    """

    if bootstrap is not None and bootstrap.cpu_collectives_implementation is not None:
        jax.config.update(
            "jax_cpu_collectives_implementation",
            bootstrap.cpu_collectives_implementation,
        )
    if jax.distributed.is_initialized():
        if bootstrap is not None and bootstrap.requests_distributed_initialization:
            raise RuntimeError("the JAX distributed runtime is already initialized")
        return runtime_info()

    if bootstrap is not None and bootstrap.requests_distributed_initialization:
        jax.distributed.initialize(
            coordinator_address=bootstrap.coordinator_address,
            num_processes=bootstrap.num_processes,
            process_id=bootstrap.process_id,
            local_device_ids=bootstrap.local_device_ids,
            cluster_detection_method=bootstrap.cluster_detection_method,
            coordinator_bind_address=bootstrap.coordinator_bind_address,
            heartbeat_timeout_seconds=bootstrap.heartbeat_timeout_seconds,
            initialization_timeout=bootstrap.initialization_timeout_seconds,
        )
    return runtime_info()


def initialize_from_environment() -> RuntimeInfo:
    """Initialize from explicit PHYDRAX variables or a supported launcher."""
    if jax.distributed.is_initialized():
        return runtime_info()

    bootstrap = RuntimeBootstrap.from_environment()
    return initialize(bootstrap)
