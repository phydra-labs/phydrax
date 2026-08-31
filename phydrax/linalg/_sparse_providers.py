#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import platform
import sys
from importlib.util import find_spec
from typing import Literal, TypeAlias

import equinox as eqx

from .._strict import StrictModule


SparseProviderName: TypeAlias = Literal[
    "jax-cuda",
    "scipy-superlu",
    "spineax-cudss",
    "umfpack",
    "cholmod",
    "spqr",
]


class SparseProviderCapabilities(StrictModule):
    """One immutable direct-provider capability declaration."""

    name: SparseProviderName = eqx.field(static=True)
    factorization: Literal["lu", "cholesky", "qr", "ldlt"] = eqx.field(static=True)
    placement: Literal["device", "host"] = eqx.field(static=True)
    package: str | None = eqx.field(static=True)
    jit: bool = eqx.field(static=True)
    transpose_solve: bool = eqx.field(static=True)
    complex: bool = eqx.field(static=True)
    batched_shared_pattern: bool = eqx.field(static=True, default=False)
    numeric_refactorization: bool = eqx.field(static=True, default=False)
    inertia: bool = eqx.field(static=True, default=False)
    reliable_zero_inertia: bool = eqx.field(static=True, default=False)
    multiple_rhs: bool = eqx.field(static=True, default=True)
    explicit_release: bool = eqx.field(static=True, default=False)


class SparseProviderAvailability(StrictModule):
    """Capability declaration paired with deterministic environment availability."""

    capabilities: SparseProviderCapabilities
    available: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)


SPARSE_PROVIDER_CATALOG = (
    SparseProviderCapabilities(
        name="jax-cuda",
        factorization="qr",
        placement="device",
        package=None,
        jit=True,
        transpose_solve=False,
        complex=False,
    ),
    SparseProviderCapabilities(
        name="scipy-superlu",
        factorization="lu",
        placement="host",
        package="scipy",
        jit=False,
        transpose_solve=True,
        complex=True,
    ),
    SparseProviderCapabilities(
        name="spineax-cudss",
        factorization="ldlt",
        placement="device",
        package="spineax",
        jit=True,
        transpose_solve=True,
        complex=False,
        batched_shared_pattern=True,
        numeric_refactorization=True,
        inertia=True,
        reliable_zero_inertia=False,
        multiple_rhs=True,
        explicit_release=True,
    ),
    SparseProviderCapabilities(
        name="umfpack",
        factorization="lu",
        placement="host",
        package="scikits.umfpack",
        jit=False,
        transpose_solve=True,
        complex=True,
    ),
    SparseProviderCapabilities(
        name="cholmod",
        factorization="cholesky",
        placement="host",
        package="sksparse.cholmod",
        jit=False,
        transpose_solve=True,
        complex=True,
    ),
    SparseProviderCapabilities(
        name="spqr",
        factorization="qr",
        placement="host",
        package="sparseqr",
        jit=False,
        transpose_solve=True,
        complex=True,
    ),
)


def sparse_provider_capabilities(
    name: SparseProviderName,
    /,
) -> SparseProviderCapabilities:
    """Return one provider declaration from the immutable built-in catalog."""
    for capabilities in SPARSE_PROVIDER_CATALOG:
        if capabilities.name == name:
            return capabilities
    raise ValueError(f"Unknown sparse provider {name!r}.")


def _package_available(package: str, /) -> bool:
    root = package.partition(".")[0]
    return find_spec(root) is not None and find_spec(package) is not None


def sparse_provider_availability(
    name: SparseProviderName,
    /,
) -> SparseProviderAvailability:
    """Inspect one provider without mutating global selection state."""
    capabilities = sparse_provider_capabilities(name)
    if name == "jax-cuda":
        import jax

        available = any(device.platform == "gpu" for device in jax.devices())
        reason = (
            "CUDA device available." if available else "No JAX CUDA device is available."
        )
    elif name == "spineax-cudss":
        import jax

        package_available = _package_available("spineax")
        platform_supported = sys.platform.startswith("linux")
        architecture_supported = platform.machine().lower() in ("x86_64", "amd64")
        cuda_available = any(device.platform == "gpu" for device in jax.devices())
        available = (
            package_available
            and platform_supported
            and architecture_supported
            and cuda_available
        )
        missing = []
        if not package_available:
            missing.append("spineax is not installed")
        if not platform_supported:
            missing.append("Linux is required")
        if not architecture_supported:
            missing.append("x86-64 is required")
        if not cuda_available:
            missing.append("no JAX CUDA device is available")
        reason = (
            "Spineax cuDSS is available."
            if available
            else "Spineax cuDSS unavailable: " + "; ".join(missing) + "."
        )
    elif capabilities.package is None:
        available = True
        reason = "Provider has no optional package dependency."
    else:
        available = _package_available(capabilities.package)
        reason = (
            f"Optional package {capabilities.package!r} is available."
            if available
            else f"Optional package {capabilities.package!r} is not installed."
        )
    return SparseProviderAvailability(
        capabilities=capabilities,
        available=available,
        reason=reason,
    )


def available_sparse_providers() -> tuple[SparseProviderAvailability, ...]:
    """Return deterministic availability evidence for every built-in provider."""
    return tuple(
        sparse_provider_availability(capabilities.name)
        for capabilities in SPARSE_PROVIDER_CATALOG
    )


__all__ = [
    "SPARSE_PROVIDER_CATALOG",
    "SparseProviderAvailability",
    "SparseProviderCapabilities",
    "SparseProviderName",
    "available_sparse_providers",
    "sparse_provider_availability",
    "sparse_provider_capabilities",
]
