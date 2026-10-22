#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Single-process MG admission within the canonical cuDSS lifecycle."""

from __future__ import annotations

import jax
from jax import Array, core as jax_core

from ..._execution_device import device_platform
from .._policies import CudssExecutionPolicy


def mg_device_indices(execution: CudssExecutionPolicy, /) -> tuple[int, ...]:
    """Resolve the declared ordered reservation without discovering a new world."""
    if execution.mode == "local":
        return ()
    group = execution.group
    if group is None:
        raise ValueError("cuDSS MG requires an execution group.")
    if group.process_indices != (jax.process_index(),):
        raise ValueError("cuDSS MG group must belong to the initiating process.")
    devices = {
        (device.process_index, device.id): device for device in jax.local_devices()
    }
    if any(key not in devices for key in group.device_keys):
        raise ValueError("cuDSS MG reservation references unavailable local devices.")
    selected = tuple(devices[key] for key in group.device_keys)
    if any(device_platform(device) != "cuda" for device in selected):
        raise ValueError("cuDSS MG reservation requires NVIDIA CUDA devices.")
    indices = tuple(device.local_hardware_id for device in selected)
    if any(index is None for index in indices):
        raise ValueError("cuDSS MG requires explicit CUDA local hardware indices.")
    return indices


def require_mg_placement(execution: CudssExecutionPolicy, *arrays: Array) -> None:
    """Refuse sharded inputs; no implicit gather or duplicate per-shard solve."""
    if execution.mode != "mg":
        return
    group = execution.group
    if group is None:
        raise ValueError("cuDSS MG requires an execution group.")
    initiating = group.device_keys[0]
    for array in arrays:
        sharding = getattr(array, "sharding", None)
        if sharding is None:
            sharding = getattr(getattr(array, "aval", None), "sharding", None)
        mesh = getattr(sharding, "mesh", None)
        if mesh is not None and not mesh.empty and mesh.size != 1:
            raise ValueError(
                "cuDSS MG refuses distributed or replicated multi-device inputs."
            )
        if isinstance(array, jax_core.Tracer):
            # Native entry also checks the initiating device and every input
            # pointer before allocating or invoking the vendor.
            continue
        keys = tuple((device.process_index, device.id) for device in array.devices())
        if keys != (initiating,):
            raise ValueError(
                "cuDSS MG full CSR/RHS must reside on its initiating device."
            )


__all__ = ["mg_device_indices", "require_mg_placement"]
