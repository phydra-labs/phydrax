#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from types import ModuleType

from ._availability import import_backend_module, probe_backend
from ._types import AbstractExternalBackend, BackendAvailability, BackendCapabilities


IREE_CAPABILITIES = BackendCapabilities(
    backend="iree",
    problem_kinds=("compiled-inference",),
    execution="host",
    host_only=False,
    supports_matrix_free=False,
    supports_assembled=False,
    coordinate_dtypes=("float32", "float64", "complex64", "complex128"),
)


def iree_availability() -> BackendAvailability:
    """Probe matched compiler and runtime Python packages without eager import."""

    compiler = probe_backend(
        IREE_CAPABILITIES,
        module="iree.compiler",
        requirement="install phydrax[iree]",
        distributions=("iree-base-compiler", "iree-base-runtime"),
    )
    if not compiler.available:
        return compiler
    runtime = probe_backend(
        IREE_CAPABILITIES,
        module="iree.runtime",
        requirement="install phydrax[iree]",
        distributions=("iree-base-compiler", "iree-base-runtime"),
    )
    if not runtime.available:
        return runtime
    versions = dict(runtime.versions)
    compiler_version = versions.get("iree-base-compiler")
    runtime_version = versions.get("iree-base-runtime")
    if compiler_version is None or runtime_version is None:
        return BackendAvailability(
            capabilities=IREE_CAPABILITIES,
            available=False,
            requirement="install phydrax[iree]",
            reason="compiler and runtime distribution versions must both be discoverable",
            versions=runtime.versions,
        )
    if compiler_version != runtime_version:
        return BackendAvailability(
            capabilities=IREE_CAPABILITIES,
            available=False,
            requirement="install matched iree-base-compiler and iree-base-runtime",
            reason=(
                f"compiler version {compiler_version!r} differs from runtime "
                f"version {runtime_version!r}"
            ),
            versions=runtime.versions,
        )
    return runtime


def import_iree() -> tuple[ModuleType, ModuleType]:
    """Import matched IREE compiler and runtime modules after capability validation."""

    availability = iree_availability()
    compiler = import_backend_module(
        availability,
        "compiled-inference",
        "iree.compiler",
    )
    runtime = import_backend_module(
        availability,
        "compiled-inference",
        "iree.runtime",
    )
    return compiler, runtime


class IREEBackend(AbstractExternalBackend):
    """Inspection boundary for IREE compilation and inference runtime."""

    @property
    def name(self) -> str:
        return "iree"

    @property
    def capabilities(self) -> BackendCapabilities:
        return IREE_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return iree_availability()


__all__ = ["IREE_CAPABILITIES", "IREEBackend", "import_iree", "iree_availability"]
