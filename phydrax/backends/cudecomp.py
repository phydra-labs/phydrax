#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Site-built cuDecomp discovery; runtime admission belongs to spectral preparation."""

from __future__ import annotations

from typing import final

from .._cuda_capabilities import windows_topology_refusal
from ._availability import probe_cuda_component
from ._types import (
    AbstractExternalBackend,
    BackendAvailability,
    BackendCapabilities,
    BackendDistributionCapabilities,
)


CUDECOMP_CAPABILITIES = BackendCapabilities(
    backend="cudecomp",
    problem_kinds=("spectral.direct-slab-transpose",),
    execution="device",
    host_only=False,
    supports_matrix_free=False,
    supports_assembled=True,
    coordinate_dtypes=("complex64", "complex128"),
    supports_plan_prepare_solve_refresh=False,
    requires_explicit_release=True,
    distribution=BackendDistributionCapabilities(
        array_model="rank_local",
        scopes=("multi_process",),
        platforms=("cuda",),
        collectives=("direct-slab-transpose",),
        transformations=("jit", "operation_specific_jvp", "operation_specific_vjp"),
        communicator_model="mpi",
        supports_process_local_input=True,
        supports_distributed_output=True,
        supports_topology_change_restart=True,
    ),
)


def cudecomp_availability() -> BackendAvailability:
    """Discover the exact site component without claiming a live MPI transport."""
    refusal = windows_topology_refusal(site_mpi=True)
    if refusal is not None:
        return BackendAvailability(
            capabilities=CUDECOMP_CAPABILITIES,
            available=False,
            requirement="a Linux site-built cuDecomp component and GPU-aware MPI",
            reason=refusal,
        )
    return probe_cuda_component(
        CUDECOMP_CAPABILITIES,
        component="cudecomp",
        requirement=(
            "an explicit CUDA13 cuDecomp site component and matching Linux "
            "same-host CUDA-aware Open MPI 5 ABI with a live one-GPU-per-rank lease"
        ),
    )


@final
class CudecompBackend(AbstractExternalBackend):
    """Lazy discovery for explicitly prepared regular three-dimensional slabs."""

    @property
    def name(self) -> str:
        return "cudecomp"

    @property
    def capabilities(self) -> BackendCapabilities:
        return CUDECOMP_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return cudecomp_availability()


__all__ = ["CUDECOMP_CAPABILITIES", "CudecompBackend", "cudecomp_availability"]
