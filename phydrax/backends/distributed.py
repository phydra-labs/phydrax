#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualified native-JAX and optional rank-local MPI collective providers."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ._availability import import_backend_module, probe_backend
from ._types import (
    BackendAvailability,
    BackendCapabilities,
    BackendDistributionCapabilities,
)


JAX_NATIVE_DISTRIBUTION = BackendDistributionCapabilities(
    array_model="global",
    scopes=("single_device", "multi_device", "multi_host"),
    platforms=("cpu", "cuda", "rocm", "tpu", "plugin"),
    collectives=(
        "sum",
        "mean",
        "maximum",
        "minimum",
        "permute",
        "all_gather",
        "all_to_all",
    ),
    transformations=("jit", "vmap", "jvp", "vjp", "rematerialization"),
    communicator_model="jax",
    supports_process_local_input=True,
    supports_distributed_output=True,
    supports_topology_change_restart=True,
)


@dataclass(frozen=True, slots=True)
class VendorExecutionProfile:
    """Admissible platform envelope; exact support still requires qualification."""

    provider_id: str
    platform: str
    vendor: str
    scopes: tuple[str, ...]
    dtypes: tuple[str, ...]
    experimental: bool = False

    def __post_init__(self) -> None:
        values = (
            self.provider_id,
            self.platform,
            self.vendor,
            *self.scopes,
            *self.dtypes,
        )
        if any(not str(value).strip() for value in values):
            raise ValueError("vendor execution profile values must be non-empty")

    @property
    def profile_id(self) -> str:
        return canonical_fingerprint(
            {
                "provider_id": self.provider_id,
                "platform": self.platform,
                "vendor": self.vendor,
                "scopes": list(self.scopes),
                "dtypes": list(self.dtypes),
                "experimental": self.experimental,
            }
        )

    def require(self, *, scope: str, dtype: str) -> None:
        if scope not in self.scopes:
            raise ValueError(
                f"{self.provider_id} does not admit execution scope {scope!r}"
            )
        if dtype not in self.dtypes:
            raise ValueError(
                f"{self.provider_id} does not admit coordinate dtype {dtype!r}"
            )


NVIDIA_JAX_PROFILE = VendorExecutionProfile(
    "jax-cuda",
    "cuda",
    "nvidia",
    ("single_device", "multi_device", "multi_host"),
    ("float16", "bfloat16", "float32", "float64", "complex64", "complex128"),
)

AMD_JAX_PROFILE = VendorExecutionProfile(
    "jax-rocm",
    "rocm",
    "amd",
    ("single_device", "multi_device", "multi_host"),
    ("float16", "bfloat16", "float32", "float64", "complex64", "complex128"),
)

TPU_JAX_PROFILE = VendorExecutionProfile(
    "jax-tpu",
    "tpu",
    "google",
    ("single_device", "multi_device", "multi_host"),
    ("float32", "bfloat16", "float64", "complex64", "complex128"),
)

INTEL_JAX_PROFILE = VendorExecutionProfile(
    "jax-intel-gpu",
    "xpu",
    "intel",
    ("single_device", "multi_device"),
    ("float16", "bfloat16", "float32", "float64"),
    experimental=True,
)

APPLE_METAL_PROFILE = VendorExecutionProfile(
    "jax-metal",
    "metal",
    "apple",
    ("single_device",),
    ("float16", "float32"),
    experimental=True,
)


def vendor_execution_profiles() -> tuple[VendorExecutionProfile, ...]:
    return (
        NVIDIA_JAX_PROFILE,
        AMD_JAX_PROFILE,
        TPU_JAX_PROFILE,
        INTEL_JAX_PROFILE,
        APPLE_METAL_PROFILE,
    )


MPI4JAX_CAPABILITIES = BackendCapabilities(
    backend="mpi4jax",
    problem_kinds=(
        "collective.sum",
        "collective.mean",
        "collective.maximum",
        "collective.minimum",
        "collective.allgather",
        "collective.alltoall",
        "collective.broadcast",
        "collective.gather",
        "collective.scatter",
        "collective.sendrecv",
    ),
    execution="device",
    host_only=False,
    supports_matrix_free=True,
    supports_assembled=False,
    coordinate_dtypes=(
        "bool",
        "int32",
        "int64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ),
    supports_plan_prepare_solve_refresh=False,
    requires_explicit_release=False,
    distribution=BackendDistributionCapabilities(
        array_model="rank_local",
        scopes=("multi_process", "multi_host"),
        platforms=("cpu", "cuda", "xpu"),
        collectives=(
            "sum",
            "mean",
            "maximum",
            "minimum",
            "allgather",
            "alltoall",
            "broadcast",
            "gather",
            "scatter",
            "sendrecv",
        ),
        transformations=("jit", "operation_specific_jvp", "operation_specific_vjp"),
        communicator_model="mpi",
        supports_process_local_input=True,
        supports_distributed_output=True,
        supports_topology_change_restart=True,
    ),
)


def mpi4jax_availability() -> BackendAvailability:
    return probe_backend(
        MPI4JAX_CAPABILITIES,
        module="mpi4jax",
        requirement=(
            "install phydrax[mpi] with one ABI-compatible MPI implementation; "
            "GPU buffers additionally require explicitly qualified GPU-aware MPI"
        ),
        distributions=("mpi4jax", "mpi4py"),
    )


class JaxCollectiveProvider:
    """Named-axis collectives used inside a JAX mapped execution region."""

    __slots__ = ("axis_name", "provider_id")

    capabilities = JAX_NATIVE_DISTRIBUTION

    def __init__(self, axis_name: str, /) -> None:
        axis = str(axis_name).strip()
        if not axis:
            raise ValueError("axis_name must be non-empty")
        self.axis_name = axis
        self.provider_id = f"jax-native:{axis}"

    def sum(self, value: Array, /) -> Array:
        return jax.lax.psum(value, self.axis_name)

    def mean(self, value: Array, /) -> Array:
        return jax.lax.pmean(value, self.axis_name)

    def maximum(self, value: Array, /) -> Array:
        return jax.lax.pmax(value, self.axis_name)

    def minimum(self, value: Array, /) -> Array:
        return jax.lax.pmin(value, self.axis_name)

    def permute(
        self,
        value: Array,
        permutation: Sequence[tuple[int, int]],
        /,
    ) -> Array:
        return jax.lax.ppermute(value, self.axis_name, tuple(permutation))

    def all_gather(
        self,
        value: Array,
        /,
        *,
        axis: int = 0,
        tiled: bool = True,
    ) -> Array:
        return jax.lax.all_gather(
            value,
            self.axis_name,
            axis=axis,
            tiled=bool(tiled),
        )

    def all_to_all(
        self,
        value: Array,
        /,
        *,
        split_axis: int,
        concat_axis: int,
    ) -> Array:
        return jax.lax.all_to_all(
            value,
            self.axis_name,
            split_axis=split_axis,
            concat_axis=concat_axis,
            tiled=True,
        )


class Mpi4JaxCollectiveProvider:
    """Explicit rank-local MPI operations with caller-owned communicator lifetime."""

    __slots__ = (
        "_mpi",
        "_mpi4jax",
        "communicator",
        "communicator_id",
        "gpu_direct",
        "provider_id",
    )

    capabilities = MPI4JAX_CAPABILITIES.distribution

    def __init__(
        self,
        communicator: Any,
        communicator_id: str,
        /,
        *,
        gpu_direct: bool = False,
    ) -> None:
        identifier = str(communicator_id).strip()
        if communicator is None or not identifier:
            raise ValueError("MPI provider requires a communicator and stable identity")
        self.communicator = communicator
        self.communicator_id = identifier
        self.gpu_direct = bool(gpu_direct)
        self.provider_id = f"mpi4jax:{identifier}"
        availability = mpi4jax_availability()
        self._mpi4jax = import_backend_module(
            availability,
            "collective.sum",
            "mpi4jax",
        )
        self._mpi = importlib.import_module("mpi4py.MPI")

    def _modules(self):
        return self._mpi4jax, self._mpi

    def sum(self, value: Array, /) -> Array:
        mpi4jax, mpi = self._modules()
        return mpi4jax.allreduce(value, op=mpi.SUM, comm=self.communicator)

    def maximum(self, value: Array, /) -> Array:
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("MPI maximum is undefined for complex values")
        mpi4jax, mpi = self._modules()
        return mpi4jax.allreduce(value, op=mpi.MAX, comm=self.communicator)

    def minimum(self, value: Array, /) -> Array:
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("MPI minimum is undefined for complex values")
        mpi4jax, mpi = self._modules()
        return mpi4jax.allreduce(value, op=mpi.MIN, comm=self.communicator)

    def mean(self, value: Array, /) -> Array:
        total = self.sum(value)
        return total / self.communicator.Get_size()

    def sendrecv(
        self,
        value: Array,
        receive_buffer: Array | None = None,
        /,
        *,
        source: int,
        destination: int,
        sendtag: int = 0,
        recvtag: int = 0,
    ) -> Array:
        mpi4jax, _ = self._modules()
        receive = jnp.zeros_like(value) if receive_buffer is None else receive_buffer
        return mpi4jax.sendrecv(
            value,
            receive,
            source,
            destination,
            sendtag=sendtag,
            recvtag=recvtag,
            comm=self.communicator,
        )


__all__ = (
    "AMD_JAX_PROFILE",
    "APPLE_METAL_PROFILE",
    "INTEL_JAX_PROFILE",
    "JAX_NATIVE_DISTRIBUTION",
    "MPI4JAX_CAPABILITIES",
    "NVIDIA_JAX_PROFILE",
    "TPU_JAX_PROFILE",
    "VendorExecutionProfile",
    "JaxCollectiveProvider",
    "Mpi4JaxCollectiveProvider",
    "mpi4jax_availability",
    "vendor_execution_profiles",
)
