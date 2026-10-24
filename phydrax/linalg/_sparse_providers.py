#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
from typing import assert_never, Literal, TypeAlias

import equinox as eqx

from .._cuda_capabilities import cuda_compiler_capabilities, selected_device
from .._strict import StrictModule
from ..backends.cudss import CudssExecutionMode
from ..typing import parse


SparseProviderName: TypeAlias = Literal[
    "jax-cuda",
    "jax-cpu",
    "scipy-superlu",
    "cudss",
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
    execution_mode: CudssExecutionMode = eqx.field(static=True, default="local")
    requires_execution_group: bool = eqx.field(static=True, default=False)
    requires_collective_lease: bool = eqx.field(static=True, default=False)
    certified_memory_bounds: bool = eqx.field(static=True, default=False)


class SparseProviderAvailability(StrictModule):
    """Capability declaration paired with deterministic environment availability."""

    capabilities: SparseProviderCapabilities
    available: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)


SPARSE_PROVIDER_CATALOG = (
    SparseProviderCapabilities(
        name="jax-cpu",
        factorization="lu",
        placement="host",
        package="scipy",
        jit=True,
        transpose_solve=False,
        complex=True,
        batched_shared_pattern=True,
    ),
    SparseProviderCapabilities(
        name="jax-cuda",
        factorization="qr",
        placement="device",
        package=None,
        jit=True,
        transpose_solve=False,
        complex=True,
        batched_shared_pattern=True,
    ),
    SparseProviderCapabilities(
        name="scipy-superlu",
        factorization="lu",
        placement="host",
        package="scipy",
        jit=False,
        transpose_solve=True,
        complex=True,
        batched_shared_pattern=True,
    ),
    SparseProviderCapabilities(
        name="cudss",
        factorization="ldlt",
        placement="device",
        package=None,
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
        batched_shared_pattern=True,
    ),
    SparseProviderCapabilities(
        name="cholmod",
        factorization="cholesky",
        placement="host",
        package="sksparse.cholmod",
        jit=False,
        transpose_solve=True,
        complex=True,
        batched_shared_pattern=True,
    ),
    SparseProviderCapabilities(
        name="spqr",
        factorization="qr",
        placement="host",
        package="sparseqr",
        jit=False,
        transpose_solve=True,
        complex=True,
        batched_shared_pattern=True,
    ),
)


def sparse_provider_capabilities(
    name: SparseProviderName,
    /,
    *,
    execution_mode: CudssExecutionMode = "local",
) -> SparseProviderCapabilities:
    """Return one provider declaration from the immutable built-in catalog.

    The JAX CUDA QR provider supports real and complex square systems, but does
    not expose singular-system failure status. cuSolver reports a singularity
    index; JAX converts it into an XLA runtime error instead of returning it.
    Consequently, even ``FailurePolicy("status")`` cannot turn a singular CUDA
    QR solve into a ``LinearSolveResult``. No rank-deficient recovery is offered.
    Automatic selection therefore considers this provider only with
    ``FailurePolicy("error")``; explicit CUDA QR selection remains available
    with either failure mode and still raises on singular systems.
    """
    execution_mode = parse(execution_mode, CudssExecutionMode, "execution_mode")
    if execution_mode != "local" and name != "cudss":
        raise ValueError(f"Sparse provider {name!r} does not support {execution_mode!r}.")
    for capabilities in SPARSE_PROVIDER_CATALOG:
        if capabilities.name == name:
            if execution_mode == "local":
                return capabilities
            return replace(
                capabilities,
                execution_mode=execution_mode,
                batched_shared_pattern=False,
                inertia=False,
                requires_execution_group=True,
                requires_collective_lease=execution_mode == "mgmn",
            )
    raise ValueError(f"Unknown sparse provider {name!r}.")


def _package_available(package: str, /) -> bool:
    root = package.partition(".")[0]
    return find_spec(root) is not None and find_spec(package) is not None


def sparse_provider_availability(
    name: SparseProviderName,
    /,
    *,
    execution_mode: CudssExecutionMode = "local",
) -> SparseProviderAvailability:
    """Inspect one provider without mutating global selection state."""
    name = parse(name, SparseProviderName, "name")
    capabilities = sparse_provider_capabilities(name, execution_mode=execution_mode)
    match name:
        case "jax-cpu":
            import jax

            package_available = _package_available("scipy")
            cpu_default = jax.default_backend() == "cpu"
            available = package_available and cpu_default
            missing = []
            if not package_available:
                missing.append("scipy is not installed")
            if not cpu_default:
                missing.append("the default JAX backend is not CPU")
            reason = (
                "JAX CPU sparse LU is available."
                if available
                else "JAX CPU sparse LU unavailable: " + "; ".join(missing) + "."
            )
        case "jax-cuda":
            compiler = cuda_compiler_capabilities(selected_device().client)
            available = compiler.sparse_qr
            reason = (
                "The selected CUDA compiler provides the sparse QR native kernel."
                if available
                else f"The selected runtime has no admitted sparse QR kernel: {compiler.reason}"
            )
        case "cudss":
            from ..backends.cudss import cudss_availability

            discovery = cudss_availability(execution_mode=execution_mode)
            available = discovery.available
            reason = discovery.reason
        case "scipy-superlu" | "umfpack" | "cholmod" | "spqr":
            if capabilities.package is None:
                available = True
                reason = "Provider has no optional package dependency."
            else:
                available = _package_available(capabilities.package)
                reason = (
                    f"Optional package {capabilities.package!r} is available."
                    if available
                    else f"Optional package {capabilities.package!r} is not installed."
                )
        case _:
            assert_never(name)
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
