#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Distributed coordination and observed runtime facts.

Coordination consumes a configuration-owned `RuntimeBootstrap` and never
discovers devices; `runtime_info` is the explicit discovery action.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

from ._config_types import RuntimeBootstrap


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


def coordinate(bootstrap: RuntimeBootstrap, /) -> None:
    """Join the distributed process set; must precede any backend initialization."""

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


__all__ = ["RuntimeInfo", "coordinate", "runtime_info"]
