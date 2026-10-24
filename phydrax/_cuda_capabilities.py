# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Positive compiler support for a destination client, never a default-backend guess.

Records contain only immutable provenance, not clients or numerical arrays. Runtime
inspection is separate from the metadata-only Windows topology admission boundary.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, replace
from importlib.metadata import distribution, PackageNotFoundError, version
from pathlib import Path
from threading import RLock
from types import ModuleType
from weakref import WeakKeyDictionary

import jax
from jax._src.lib import xla_client

from ._execution_device import device_platform
from ._fingerprint import canonical_fingerprint
from ._jax_config import default_device_selection


@dataclass(frozen=True)
class CudaCompilerCapabilities:
    host_platform: str
    distribution: str
    version: str
    build_id: str
    jax_version: str
    jaxlib_version: str
    platform_version: str
    mosaic_gpu: bool = False
    pallas_triton: bool = False
    sparse_qr: bool = False
    reason: str = "No admitted CUDA compiler implementation."

    @property
    def identity(self) -> str:
        """Stable compiler/build identity, independent of a client object's lifetime."""
        return canonical_fingerprint(self.identity_payload)

    @property
    def identity_payload(self) -> dict[str, str | bool]:
        return {
            "host_platform": self.host_platform,
            "distribution": self.distribution,
            "version": self.version,
            "build_id": self.build_id,
            "jax": self.jax_version,
            "jaxlib": self.jaxlib_version,
            "platform_version": self.platform_version,
            "mosaic_gpu": self.mosaic_gpu,
            "pallas_triton": self.pallas_triton,
            "sparse_qr": self.sparse_qr,
        }


_CLIENT_CAPABILITIES: WeakKeyDictionary[xla_client.Client, CudaCompilerCapabilities] = (
    WeakKeyDictionary()
)
_CLIENT_CAPABILITIES_LOCK = RLock()


def windows_topology_refusal(
    *, device_count: int = 1, process_count: int = 1, site_mpi: bool = False
) -> str | None:
    """Metadata-only admission; callers select CUDA intent before applying this.

    Explicit CPU reference-device simulation is not a CUDA topology request.
    """
    if sys.platform == "win32" and (device_count != 1 or process_count != 1 or site_mpi):
        return (
            "The Windows CUDA runtime supports one process and one device only; "
            "multi-device, multi-process, site MPI, cuDSS MG/MGMN and cuDecomp "
            "execution are unsupported."
        )
    return None


def selected_device() -> jax.Device:
    """Resolve the scoped selection used to prepare a numerical operation."""
    selection = default_device_selection()
    return selection if isinstance(selection, jax.Device) else jax.devices(selection)[0]


def _optional_module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None


def _native_support(package: str) -> tuple[bool, bool, bool]:
    # Use the modules JAX actually bound, not a second independently loaded kernel
    # package. In particular sparse QR lives in _solver, not _sparse.
    from jaxlib import gpu_solver, gpu_triton

    solver = gpu_solver._cusolver
    triton = gpu_triton._cuda_triton
    sparse = bool(
        solver is not None
        and solver.__name__.startswith(package + ".")
        and "cusolver_csrlsvqr_ffi" in solver.registrations()
    )
    pallas = bool(
        triton is not None
        and triton.__name__.startswith(package + ".")
        and callable(getattr(triton, "get_custom_call", None))
        and _optional_module("jax._src.lib.triton") is not None
    )
    bindings = _optional_module("jax._src.lib.mosaic_gpu")
    extension = getattr(bindings, "_mosaic_gpu_ext", None)
    mosaic = bool(
        extension is not None
        and extension.__name__.startswith(package + ".")
        and callable(getattr(extension, "_get_ptxas_isa_version", None))
    )
    if mosaic and sys.platform == "linux":
        mosaic = bool(
            extension.__file__
            and (Path(extension.__file__).parent / "libmosaic_gpu_runtime.so").is_file()
        )
    return mosaic, pallas, sparse


def cuda_compiler_capabilities(client: xla_client.Client, /) -> CudaCompilerCapabilities:
    """Inspect the actual destination client without running a probe kernel.

    Windows requires both the validated owned build declaration and the actual
    JAX-bound native entry point. Linux retains upstream CUDA 13 plugin support,
    but mere package presence or compute capability is not compiler support.
    Foreign CUDA majors and unknown implementations are not positively admitted.
    """
    with _CLIENT_CAPABILITIES_LOCK:
        cached = _CLIENT_CAPABILITIES.get(client)
        if cached is None:
            cached = _inspect_cuda_compiler(client)
            _CLIENT_CAPABILITIES[client] = cached
        return cached


def _inspect_cuda_compiler(client: xla_client.Client, /) -> CudaCompilerCapabilities:
    base = CudaCompilerCapabilities(
        host_platform=sys.platform,
        distribution="",
        version="",
        build_id="",
        jax_version=jax.__version__,
        jaxlib_version=version("jaxlib"),
        platform_version=client.platform_version,
    )
    devices = client.devices()
    if not devices or any(device_platform(device) != "cuda" for device in devices):
        return base
    if sys.platform == "win32":
        try:
            from phydrax_cuda13_pjrt.runtime import installation_record

            record = installation_record()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            return replace(
                base,
                distribution="phydrax-cuda13-pjrt",
                reason=f"Windows CUDA runtime capability inspection failed: {error}",
            )
        reason = "; ".join(record.issues)
        if (
            record.jax_version != jax.__version__
            or record.jaxlib_version != base.jaxlib_version
        ):
            reason = reason or "Windows CUDA runtime requires its pinned JAX/JAXlib pair."
        mosaic, triton, sparse = (
            (False, False, False) if reason else _native_support("jax_cuda13_plugin")
        )
        return replace(
            base,
            distribution=record.distribution,
            version=record.version,
            build_id=record.build_id,
            mosaic_gpu=mosaic and "mosaic_gpu" in record.capabilities,
            pallas_triton=triton and "pallas_triton" in record.capabilities,
            sparse_qr=sparse and "sparse_qr" in record.capabilities,
            reason=reason
            or "Windows CUDA compiler support requires declared build capabilities and native entry points.",
        )
    if sys.platform == "linux":
        for name in ("jax-cuda12-plugin", "jax-cuda12-pjrt"):
            try:
                foreign = distribution(name)
            except PackageNotFoundError:
                continue
            return replace(
                base,
                distribution=name,
                version=foreign.version,
                reason="CUDA 12 is unsupported; Phydrax requires the CUDA 13 runtime and kernels.",
            )
        try:
            kernels = distribution("jax-cuda13-plugin")
            runtime = distribution("jax-cuda13-pjrt")
        except PackageNotFoundError:
            pass
        else:
            records = (kernels.read_text("RECORD"), runtime.read_text("RECORD"))
            matched = kernels.version == runtime.version == base.jaxlib_version
            mosaic, triton, sparse = (
                _native_support("jax_cuda13_plugin")
                if matched and all(records)
                else (False, False, False)
            )
            return replace(
                base,
                distribution="jax-cuda13-pjrt",
                version=runtime.version,
                build_id=canonical_fingerprint(
                    {"kernel_record": records[0], "runtime_record": records[1]}
                ),
                mosaic_gpu=mosaic,
                pallas_triton=triton,
                sparse_qr=sparse,
                reason="Linux CUDA compiler support requires matched plugin/runtime builds and native entry points.",
            )
    return replace(
        base,
        reason="No unique, matched CUDA compiler implementation is admitted for this client.",
    )
