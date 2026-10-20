#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical accelerator identity for the existing execution substrate."""

from __future__ import annotations

import jax


def device_platform(device: jax.Device, /) -> str:
    """Identify CUDA, ROCm, and oneAPI from the loaded runtime.

    JAX's public GPU platform is generic on current releases. Its client
    platform/version describes the loaded runtime; device display names do not
    own scientific or provider identity. Unknown plugins remain unqualified.
    """
    platform = device.platform.lower()
    if platform != "gpu":
        return platform
    client = device.client.platform.lower()
    if client in ("cuda", "rocm", "oneapi"):
        return client
    runtime = device.client.platform_version.lower().split()
    if runtime and runtime[0] in ("cuda", "rocm", "oneapi"):
        return runtime[0]
    if client == "sycl" or (runtime and runtime[0] == "sycl"):
        return "oneapi"
    return platform


def device_vendor(device: jax.Device, /) -> str | None:
    """Return the accelerator vendor identified by the loaded runtime, if any."""
    match device_platform(device):
        case "cuda":
            return "nvidia"
        case "rocm":
            return "amd"
        case "tpu":
            return "google"
        case "oneapi":
            return "intel"
        case "mps":
            runtime = device.client.platform_version.split()
            return "apple" if runtime and runtime[0] == "phydrax-metal" else None
        case _:
            return None
