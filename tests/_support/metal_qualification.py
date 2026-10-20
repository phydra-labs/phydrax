#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Actual MPS placement evidence for Metal installed-wheel qualification cases.

Device test files run in a process started with ``METAL_STARTUP_ENVIRONMENT``.
Every device case records the runtime identity of the device that holds its
actual result arrays as the ``metal_runtime`` user property; the wheel
qualifier compares that record with the installed companion's identity.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from importlib import metadata
from typing import Final

import jax
import pytest


METAL_STARTUP_ENVIRONMENT: Final = {"JAX_PLATFORMS": "mps", "JAX_ENABLE_X64": "1"}
METAL_RUNTIME_PROPERTY: Final = "metal_runtime"
METAL_RUNTIME_KEYS: Final = frozenset(
    {"platform", "platform_version", "device_kind", "device_id", "x64_enabled"}
)


def metal_device() -> jax.Device:
    """Return the one MPS device of this explicitly selected Metal process.

    Skips only when the optional companion distribution is absent. An installed
    companion with an unsuccessful environment, a non-MPS process or several
    devices fails instead of silently running elsewhere.
    """
    try:
        metadata.distribution("phydrax-metal")
    except metadata.PackageNotFoundError:
        pytest.skip("Install phydrax[metal] to run Apple GPU cases.")
    from phydrax import config as pxconfig
    from phydrax.backends import metal_environment

    assert pxconfig.snapshot().platforms == ("mps",), (
        "Metal cases need a process started with JAX_PLATFORMS=mps."
    )
    report = metal_environment()
    issues = [f"{issue.code}: {issue.message}" for issue in report.issues]
    assert report.successful and report.facts.device_probed, issues
    assert jax.default_backend() == "mps", f"Default backend is {jax.default_backend()}."
    assert pxconfig.x64_enabled(), "Metal qualification requires x64 enabled."
    devices = jax.devices("mps")
    assert len(devices) == 1, f"Expected exactly one MPS device, observed {devices}."
    return devices[0]


def _device_identity(device: jax.Device, /) -> dict[str, object]:
    from phydrax import config as pxconfig

    return {
        "platform": device.platform,
        "platform_version": device.client.platform_version,
        "device_kind": device.device_kind,
        "device_id": device.id,
        "x64_enabled": bool(pxconfig.x64_enabled()),
    }


def _placed_evidence(
    device: jax.Device, arrays: tuple[jax.Array, ...], /
) -> dict[str, object]:
    if not arrays:
        raise ValueError("Metal runtime evidence requires at least one result array.")
    for array in arrays:
        if not isinstance(array, jax.Array):
            raise TypeError(f"Expected a jax.Array result, got {type(array).__name__}.")
        placed = array.devices()
        assert placed == {device}, f"Result placed on {placed}, expected {device}."
    return _device_identity(device)


def _record(
    record_property: Callable[[str, object], None],
    device: jax.Device,
    evidence: Mapping[str, object],
    /,
) -> None:
    observed = dict(evidence)
    if set(observed) != METAL_RUNTIME_KEYS:
        raise ValueError(f"Metal runtime evidence keys differ: {sorted(observed)}.")
    expected = {**_device_identity(device), "x64_enabled": True}
    assert observed == expected, f"Work ran on {observed}, expected {expected}."
    record_property(METAL_RUNTIME_PROPERTY, json.dumps(observed, sort_keys=True))


def metal_runtime_evidence(*arrays: jax.Array) -> dict[str, object]:
    """Assert that every result lives on the one MPS device and describe it."""
    return _placed_evidence(metal_device(), arrays)


def record_metal_evidence(
    record_property: Callable[[str, object], None], evidence: Mapping[str, object]
) -> None:
    """Record evidence produced by a child process after checking its identity."""
    _record(record_property, metal_device(), evidence)


def record_metal_execution(
    record_property: Callable[[str, object], None], *arrays: jax.Array
) -> None:
    """Record the identity of the MPS device holding this case's actual results."""
    device = metal_device()
    _record(record_property, device, _placed_evidence(device, arrays))


__all__ = [
    "METAL_RUNTIME_KEYS",
    "METAL_RUNTIME_PROPERTY",
    "METAL_STARTUP_ENVIRONMENT",
    "metal_device",
    "metal_runtime_evidence",
    "record_metal_evidence",
    "record_metal_execution",
]
