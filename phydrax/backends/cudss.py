#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import replace
from typing import assert_never, final, Literal, TypeAlias

from .._cuda_capabilities import windows_topology_refusal
from ..typing import parse
from ._availability import probe_cuda_component
from ._types import (
    AbstractExternalBackend,
    BackendAvailability,
    BackendCapabilities,
    BackendDistributionCapabilities,
)


CudssExecutionMode: TypeAlias = Literal["local", "mg", "mgmn"]


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


def cudss_availability(
    *, execution_mode: CudssExecutionMode = "local"
) -> BackendAvailability:
    """Probe artifacts for one mode, not group, memory, or transport admission.

    Installing cuDSS alone does not reserve peers or establish a live collective
    lease. MGMN additionally requires the exact site-linked bridge/shim.
    """
    execution_mode = parse(execution_mode, CudssExecutionMode, "execution_mode")
    match execution_mode:
        case "local":
            capabilities = CUDSS_CAPABILITIES
            component = "cudss"
        case "mg":
            capabilities = replace(
                CUDSS_CAPABILITIES,
                distribution=BackendDistributionCapabilities(
                    array_model="global",
                    scopes=("single_process_multi_device",),
                    platforms=("cuda",),
                    transformations=("jit",),
                    supports_process_local_input=True,
                ),
            )
            component = "cudss"
        case "mgmn":
            capabilities = replace(
                CUDSS_CAPABILITIES,
                requires_explicit_release=True,
                distribution=BackendDistributionCapabilities(
                    array_model="rank_local",
                    scopes=("same_host_multi_process_gpu",),
                    platforms=("cuda",),
                    collectives=("site_mpi",),
                    transformations=("jit",),
                    communicator_model="mpi",
                    supports_process_local_input=True,
                    supports_distributed_output=True,
                ),
            )
            component = "cudss_mgmn"
        case _:
            assert_never(execution_mode)
    refusal = windows_topology_refusal(
        device_count=1 if execution_mode == "local" else 2,
        site_mpi=execution_mode == "mgmn",
    )
    if refusal is not None:
        return BackendAvailability(
            capabilities=capabilities,
            available=False,
            requirement="a CUDA runtime supporting the requested cuDSS execution topology",
            reason=refusal,
        )
    return probe_cuda_component(
        capabilities,
        component=component,
        requirement=(
            "install phydrax[cuda13] on Linux x86-64/aarch64 or supported Windows x86-64 "
            "with NVIDIA Turing-or-newer devices"
            + (
                " and exact site-built collective/cuDSS MGMN components using GPU-aware MPI"
                if execution_mode == "mgmn"
                else ""
            )
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


__all__ = [
    "CUDSS_CAPABILITIES",
    "CudssBackend",
    "CudssExecutionMode",
    "cudss_availability",
]
