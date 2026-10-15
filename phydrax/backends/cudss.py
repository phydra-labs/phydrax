#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

from ._availability import probe_cuda_component
from ._types import AbstractExternalBackend, BackendAvailability, BackendCapabilities


CUDSS_CAPABILITIES = BackendCapabilities(
    backend="cudss",
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
    # Native state is a capacity-bounded memo; release only retires entries early.
    requires_explicit_release=False,
)


def cudss_availability() -> BackendAvailability:
    """Probe the lockstep CUDA companion before the first cuDSS call."""
    return probe_cuda_component(
        CUDSS_CAPABILITIES,
        component="cudss",
        requirement=(
            "install phydrax[cuda12] or phydrax[cuda13] on Linux x86-64 "
            "with an NVIDIA Turing-or-newer device"
        ),
    )


@final
class CudssBackend(AbstractExternalBackend):
    """Lazy native cuDSS sparse-direct backend boundary."""

    @property
    def name(self) -> str:
        return "cudss"

    @property
    def capabilities(self) -> BackendCapabilities:
        return CUDSS_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return cudss_availability()


__all__ = ["CUDSS_CAPABILITIES", "CudssBackend", "cudss_availability"]
