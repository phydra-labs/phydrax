#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ._availability import probe_backend
from ._types import AbstractExternalBackend, BackendAvailability, BackendCapabilities


SPINEAX_CAPABILITIES = BackendCapabilities(
    backend="spineax-cudss",
    problem_kinds=(
        "linear.sparse-system",
        "linear.symmetric-indefinite-system",
        "optimization.kkt-system",
    ),
    execution="device",
    host_only=False,
    supports_matrix_free=False,
    supports_assembled=True,
    coordinate_dtypes=("float32", "float64"),
    supports_plan_prepare_solve_refresh=True,
    requires_explicit_release=True,
)


def spineax_availability() -> BackendAvailability:
    """Probe the optional Linux CUDA cuDSS bridge without base-package import."""
    return probe_backend(
        SPINEAX_CAPABILITIES,
        module="spineax.cudss",
        requirement=(
            "install the phydrax[cudss] extra on Linux x86-64 with CUDA 13 and "
            "an NVIDIA Turing-or-newer device"
        ),
        distributions=("spineax",),
        supported_platforms=("linux",),
    )


class SpineaxBackend(AbstractExternalBackend):
    """Lazy optional cuDSS sparse-direct backend boundary."""

    @property
    def name(self) -> str:
        return "spineax-cudss"

    @property
    def capabilities(self) -> BackendCapabilities:
        return SPINEAX_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return spineax_availability()


__all__ = [
    "SPINEAX_CAPABILITIES",
    "SpineaxBackend",
    "spineax_availability",
]
