#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lazy, lockstep CUDA companions using the public JAX FFI and a plain C ABI."""

from __future__ import annotations

import atexit
import ctypes
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final, Literal

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.core import Tracer

from ._config_environment import LIBRARY_ENVIRONMENT
from ._config_state import library_input
from ._config_types import LibrarySetting
from ._cuda_environment import (
    assess_cuda_environment,
    collect_cuda_environment,
    CudaEnvironmentFacts,
    CudaEnvironmentIssue,
    CudaIssueCode,
)
from ._elf_dependencies import read_dynamic_section, resolve_dependency_closure
from ._execution_device import device_platform
from ._strict import StrictModule
from .logging import emit
from .typing import parse


type CudaNativeComponentName = Literal["cudss", "amgx", "quda"]

# This table owns target identity; Python callers use these same native names.
CUDA_NATIVE_HANDLERS: dict[CudaNativeComponentName, tuple[str, ...]] = {
    "cudss": tuple(
        f"phydrax_cudss_{operation}_{dtype}"
        for operation in ("analyze", "factorize", "refactorize", "solve", "diagonal")
        for dtype in ("f32", "f64")
    )
    + ("phydrax_cudss_release",),
    "amgx": tuple(
        f"phydrax_amgx_{op}" for op in ("setup", "refresh", "solve", "release")
    ),
    "quda": tuple(
        f"phydrax_quda_{operation}_{dtype}"
        for operation in ("gauge_action", "gauge_force", "wilson_dslash")
        for dtype in ("c64", "c128")
    ),
}
_SETTINGS: dict[CudaNativeComponentName, LibrarySetting] = {
    "cudss": "cuda_cudss",
    "amgx": "cuda_amgx",
    "quda": "cuda_quda",
}
# Explicit family -> wheel names; math-library versions do not encode toolkit major.
_NVIDIA_DISTRIBUTIONS = {
    "cudart": "nvidia-cuda-runtime",
    "cublas": "nvidia-cublas",
    "cublasLt": "nvidia-cublas",
    "cusparse": "nvidia-cusparse",
    "cusolver": "nvidia-cusolver",
    "cufft": "nvidia-cufft",
    "curand": "nvidia-curand",
    "nvJitLink": "nvidia-nvjitlink",
    "nvrtc": "nvidia-cuda-nvrtc",
    "cupti": "nvidia-cuda-cupti",
    "cudnn": "nvidia-cudnn-cu13",
    "nccl": "nvidia-nccl-cu13",
    "nvshmem_host": "nvidia-nvshmem-cu13",
    "cudss": "nvidia-cudss-cu13",
}
_CUDA_SONAME = re.compile(
    r"lib(cudart|cublasLt|cublas|cusparse|cusolver|cufft|curand|nvJitLink|nvrtc|cupti|cudnn[^/]*|nccl|nvshmem_host|cudss)\.so(?:\.\d+)*\Z"
)
_I32 = ctypes.c_int32
_I64 = ctypes.c_int64
_P = ctypes.c_void_p
_SIGNATURES: dict[str, tuple[Any, tuple[Any, ...]]] = {
    "phx_cuda_version": (ctypes.c_char_p, ()),
    "phx_cuda_build_hash": (ctypes.c_char_p, ()),
    "phx_cuda_component": (ctypes.c_char_p, ()),
    "phx_cuda_last_error": (ctypes.c_char_p, ()),
    "phx_cuda_major": (_I32, ()),
    "phx_cuda_handler_count": (_I32, ()),
    "phx_cuda_handler_name": (ctypes.c_char_p, (_I32,)),
    "phx_cuda_handler": (_P, (_I32,)),
    "phx_cuda_registry_partitions": (_I32, ()),
    "phx_cuda_registry_partition": (_I32, (_I32, _P, _P, _P, _P, _P)),
    "phx_cuda_release": (_I32, (_I64, _P, _I64, _P)),
    "phx_cuda_release_all": (_I32, (_P,)),
    "phx_cuda_shutdown": (_I32, ()),
}


class CudaNativeUnavailableError(ImportError):
    """A companion refusal with structured, actionable environment evidence."""

    def __init__(self, issue: CudaEnvironmentIssue, /) -> None:
        super().__init__(issue.message)
        self.issue = issue


@final
class CudaNativeComponent(StrictModule):
    """Identity and absolute dependency paths of one registered companion."""

    name: CudaNativeComponentName = eqx.field(static=True)
    path: str = eqx.field(static=True)
    version: str = eqx.field(static=True)
    build_hash: str = eqx.field(static=True)
    major: int = eqx.field(static=True)
    preloaded: tuple[str, ...] = eqx.field(static=True)


@final
class CudaNativePartition(StrictModule):
    """Bounded native memo evidence for a component/device/plan partition."""

    component: CudaNativeComponentName = eqx.field(static=True)
    device: int = eqx.field(static=True)
    partition: int = eqx.field(static=True)
    live: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    rebuilds: int = eqx.field(static=True)


@final
class CudaNativeResources(StrictModule):
    """Snapshot of loaded component identities and their live memo partitions."""

    components: tuple[CudaNativeComponent, ...] = eqx.field(static=True)
    partitions: tuple[CudaNativePartition, ...] = eqx.field(static=True)


@dataclass(frozen=True)
class _Loaded:
    identity: CudaNativeComponent
    library: ctypes.CDLL
    functions: dict[str, Any]
    dependencies: tuple[ctypes.CDLL, ...]


_LOCK = threading.RLock()
_LOADED: dict[CudaNativeComponentName, _Loaded] = {}
_EXIT_REGISTERED = False


def _refuse(code: CudaIssueCode, message: str, *paths: str) -> CudaNativeUnavailableError:
    return CudaNativeUnavailableError(
        CudaEnvironmentIssue(code=code, severity="error", message=message, paths=paths)
    )


def _reinstall() -> str:
    return 'Reinstall with pip install --upgrade --force-reinstall "phydrax[cuda13]".'


@functools.cache
def _library_location(name: CudaNativeComponentName, configured: str) -> Path:
    setting = LIBRARY_ENVIRONMENT[_SETTINGS[name]]
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise _refuse(
                "companion-missing",
                f"{setting} names a missing library: {configured!r}; correct or unset {setting}.",
                str(path),
            )
        return path
    module = "phydrax_cuda13"
    if importlib.util.find_spec(module) is None:
        raise _refuse(
            "companion-missing",
            "CUDA companion phydrax-cuda13 is missing; install phydrax[cuda13].",
        )
    try:
        return Path(importlib.import_module(module).component_library(name)).resolve()
    except (OSError, ValueError) as error:
        raise _refuse(
            "companion-missing",
            f"CUDA {name} companion is missing: {error}. {_reinstall()}",
        ) from error


def _cuda_facts() -> CudaEnvironmentFacts:
    facts = collect_cuda_environment(initialize_backend=False)
    for issue in assess_cuda_environment(facts).issues:
        if issue.severity == "error":
            raise CudaNativeUnavailableError(issue)
    # JAX owns its wheel preloads and must initialize before any companion dlopen.
    try:
        devices = jax.local_devices()
    except RuntimeError as error:
        raise _refuse(
            "backend-unavailable",
            f"No JAX CUDA device is available: {error}; check the NVIDIA driver and JAX CUDA installation.",
        ) from error
    cuda_devices = tuple(
        device for device in devices if device_platform(device) == "cuda"
    )
    if not cuda_devices:
        # A probed assessment names the cause (e.g. LD_LIBRARY_PATH shadowing that
        # stopped the CUDA plugin); otherwise CUDA was not selected or present.
        probed = assess_cuda_environment(
            collect_cuda_environment(initialize_backend=True)
        )
        for issue in probed.issues:
            if issue.severity == "error":
                raise CudaNativeUnavailableError(issue)
        raise _refuse(
            "backend-unavailable",
            "No JAX CUDA device is available; enable the CUDA JAX platform and install phydrax[cuda13].",
        )
    if len(facts.plugin_majors) != 1:
        raise _refuse(
            "no-cuda-plugin",
            "No unique JAX CUDA13 plugin is installed; install phydrax[cuda13].",
        )
    runtime = cuda_devices[0].client.platform_version
    match = re.fullmatch(r"cuda (\d+)", runtime)
    if match is None or int(match[1]) // 1000 != 13:
        raise _refuse(
            "companion-major-mismatch",
            f"JAX runtime {runtime!r} does not match CUDA 13. {_reinstall()}",
        )
    return facts


def _missing_dependency(soname: str, path: str) -> CudaNativeUnavailableError:
    match = _CUDA_SONAME.fullmatch(soname)
    if match is None:
        raise ValueError(f"Not a CUDA-family soname: {soname}.")
    family = "cudnn" if match[1].startswith("cudnn") else match[1]
    distribution = _NVIDIA_DISTRIBUTIONS[family]
    return _refuse(
        "native-dependency-missing",
        f"CUDA dependency {soname}, requested by {path}, is absent from the CUDA 13 NVIDIA wheels; install or reinstall {distribution}. {_reinstall()}",
        path,
    )


def _dependency_paths(path: Path, facts: CudaEnvironmentFacts) -> tuple[str, ...]:
    wheels: dict[str, str] = {}
    for filename, owner_major in facts.wheel_library_majors:
        if owner_major != 13 or _CUDA_SONAME.fullmatch(Path(filename).name) is None:
            continue
        candidate = Path(filename)
        if candidate.is_file():
            dynamic = read_dynamic_section(candidate)
            if dynamic.soname is not None:
                wheels[dynamic.soname] = str(candidate.resolve())
    # Seeds force every CUDA soname to its matching RECORD file. The ELF resolver
    # treats seeds as already loaded, so visit each seeded object's own closure too.
    seeds = tuple(sorted(wheels.items()))
    visited: set[str] = set()
    ordered: list[str] = []

    def visit(filename: str) -> None:
        if filename in visited:
            return
        visited.add(filename)
        # Refuse a missing wheel before the resolver can follow a system copy's
        # transitive dependencies and misidentify the distribution to reinstall.
        for soname in read_dynamic_section(filename).needed:
            if _CUDA_SONAME.fullmatch(soname) is not None and soname not in wheels:
                raise _missing_dependency(soname, filename)
        closure = resolve_dependency_closure(
            filename, library_path=(), executable=sys.executable, preloaded=seeds
        )
        for dependency in (*closure.resolved, *closure.unresolved):
            if _CUDA_SONAME.fullmatch(dependency.soname) is None:
                continue
            wheel = wheels.get(dependency.soname)
            if wheel is None:
                raise _missing_dependency(dependency.soname, dependency.requested_by)
            visit(wheel)
        if filename != str(path):
            ordered.append(filename)

    visit(str(path))
    return tuple(ordered)


def _identity_text(function: Any, symbol: str, path: Path) -> str:
    raw = function()
    if isinstance(raw, bytes):
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            text = ""
        if text:
            return text
    raise _refuse(
        "companion-invalid",
        f"Companion {path} has invalid {symbol}. {_reinstall()}",
        str(path),
    )


def _bind(
    library: ctypes.CDLL,
    path: Path,
    name: CudaNativeComponentName,
    preloaded: tuple[str, ...],
) -> tuple[CudaNativeComponent, dict[str, Any], tuple[tuple[str, int], ...]]:
    functions: dict[str, Any] = {}
    for symbol, (restype, argtypes) in _SIGNATURES.items():
        try:
            function = library[symbol]
        except AttributeError as error:
            raise _refuse(
                "companion-invalid",
                f"Companion {path} lacks C ABI symbol {symbol}. {_reinstall()}",
                str(path),
            ) from error
        function.restype = restype
        function.argtypes = argtypes
        functions[symbol] = function
    version = _identity_text(functions["phx_cuda_version"], "version", path)
    try:
        required = importlib.metadata.version("phydrax")
    except importlib.metadata.PackageNotFoundError as error:
        raise _refuse(
            "companion-invalid",
            f"Cannot verify companion {path}: phydrax distribution metadata is missing. {_reinstall()}",
            str(path),
        ) from error
    if version != required:
        raise _refuse(
            "companion-version-mismatch",
            f"Companion {path} is release {version}; phydrax {required} requires phydrax-cuda13=={required}. {_reinstall()}",
            str(path),
        )
    native_major = functions["phx_cuda_major"]()
    if native_major != 13:
        raise _refuse(
            "companion-major-mismatch",
            f"Companion {path} uses CUDA {native_major}, but Phydrax requires CUDA 13. {_reinstall()}",
            str(path),
        )
    component = _identity_text(functions["phx_cuda_component"], "component", path)
    build_hash = _identity_text(functions["phx_cuda_build_hash"], "build hash", path)
    if component != name or re.fullmatch(r"[0-9a-f]{64}", build_hash) is None:
        raise _refuse(
            "companion-invalid",
            f"Companion {path} has invalid identity (component {component!r}, build hash {build_hash!r}); expected {name}. {_reinstall()}",
            str(path),
        )
    count = functions["phx_cuda_handler_count"]()
    handlers: list[tuple[str, int]] = []
    if count == len(CUDA_NATIVE_HANDLERS[name]):
        for index in range(count):
            raw_name = functions["phx_cuda_handler_name"](index)
            pointer = functions["phx_cuda_handler"](index)
            if not isinstance(raw_name, bytes) or not pointer:
                break
            try:
                target = raw_name.decode("ascii")
            except UnicodeDecodeError:
                break
            handlers.append((target, pointer))
    if len(handlers) != count or tuple(sorted(target for target, _ in handlers)) != tuple(
        sorted(CUDA_NATIVE_HANDLERS[name])
    ):
        raise _refuse(
            "companion-invalid",
            f"Companion {path} has missing, duplicate, null, or unexpected FFI handlers; required: {', '.join(CUDA_NATIVE_HANDLERS[name])}. {_reinstall()}",
            str(path),
        )
    for target, _ in handlers:
        try:
            library[target]
        except AttributeError as error:
            raise _refuse(
                "companion-invalid",
                f"Companion {path} lacks exported FFI handler {target}. {_reinstall()}",
                str(path),
            ) from error
    return (
        CudaNativeComponent(
            name=name,
            path=str(path),
            version=version,
            build_hash=build_hash,
            major=13,
            preloaded=preloaded,
        ),
        functions,
        tuple(handlers),
    )


def load_cuda_component(name: CudaNativeComponentName, /) -> CudaNativeComponent:
    """Validate, preload and register a component exactly once in this process."""
    global _EXIT_REGISTERED
    name = parse(name, CudaNativeComponentName, "name")
    with _LOCK:
        if name in _LOADED:
            return _LOADED[name].identity
        facts = _cuda_facts()
        path = _library_location(name, library_input(_SETTINGS[name]))
        try:
            preloaded = _dependency_paths(path, facts)
            dependencies = tuple(ctypes.CDLL(filename) for filename in preloaded)
            library = ctypes.CDLL(str(path))
        except (OSError, ValueError) as error:
            raise _refuse(
                "companion-invalid",
                f"Cannot load CUDA {name} companion {path}: {error}. {_reinstall()}",
                str(path),
            ) from error
        identity, functions, handlers = _bind(library, path, name, preloaded)
        for target, pointer in handlers:
            jax.ffi.register_ffi_target(
                target, jax.ffi.pycapsule(ctypes.c_void_p(pointer)), platform="CUDA"
            )
        _LOADED[name] = _Loaded(identity, library, functions, dependencies)
        if not _EXIT_REGISTERED:
            atexit.register(_shutdown)
            _EXIT_REGISTERED = True
        return identity


def cuda_component_status(
    name: CudaNativeComponentName, /
) -> tuple[bool, str, tuple[CudaEnvironmentIssue, ...], tuple[tuple[str, str], ...]]:
    """Return a component's availability and identity without raising refusals."""
    try:
        component = load_cuda_component(name)
    except CudaNativeUnavailableError as error:
        return False, str(error), (error.issue,), ()
    versions = [
        ("phydrax-cuda13", component.version),
        (f"phydrax-cuda-{component.name}", component.build_hash),
    ]
    if component.name == "cudss":
        distribution = _NVIDIA_DISTRIBUTIONS["cudss"]
        try:
            versions.append((distribution, importlib.metadata.version(distribution)))
        except importlib.metadata.PackageNotFoundError:
            pass  # Development C ABI fixtures need no NVIDIA dependencies.
    return (
        True,
        f"Phydrax CUDA {component.major} {component.name} companion is available.",
        (),
        tuple(versions),
    )


def cuda_partition_key(identity: str, /) -> int:
    """Map an explicit plan identity to a stable positive signed 64-bit key."""
    if not isinstance(identity, str):
        raise TypeError("CUDA partition identity must be text.")
    if not identity:
        raise ValueError("CUDA partition identity must be nonempty.")
    return (
        int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")
        & ((1 << 63) - 1)
    ) or 1


def _check_status(loaded: _Loaded, status: int, operation: str) -> None:
    if status != 0:
        error = loaded.functions["phx_cuda_last_error"]()
        text = (
            error.decode("utf-8", errors="replace")
            if isinstance(error, bytes)
            else "no native error text"
        )
        raise RuntimeError(
            f"CUDA {loaded.identity.name} {operation} failed (status {status}): {text}"
        )


def release_cuda_entries(
    name: CudaNativeComponentName, partition: int, ids: tuple[int, ...] | Array, /
) -> int:
    """Eagerly release host IDs; traced releases must use the component FFI target."""
    if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves((partition, ids))):
        raise TypeError(
            f"Eager CUDA release cannot accept traced inputs; use the traced phydrax_{name}_release FFI target."
        )
    if type(partition) is not int:
        raise TypeError("CUDA partition must be an integer.")
    if not 0 < partition < (1 << 63):
        raise ValueError("CUDA partition must be a positive int63.")
    host = np.asarray(jax.device_get(ids))
    if host.ndim != 1 or (host.size and not np.issubdtype(host.dtype, np.integer)):
        raise TypeError("CUDA release IDs must be a one-dimensional integer sequence.")
    if host.size and (np.any(host < 0) or np.any(host > np.iinfo(np.int32).max)):
        raise ValueError("CUDA release IDs must fit non-negative int32.")
    host = np.ascontiguousarray(host, dtype=np.int32)
    load_cuda_component(name)
    with _LOCK:
        loaded = _LOADED[name]
        released = _I64()
        _check_status(
            loaded,
            loaded.functions["phx_cuda_release"](
                partition, host.ctypes.data, host.size, ctypes.byref(released)
            ),
            "release",
        )
        return released.value


def cuda_native_resources() -> CudaNativeResources:
    """Inspect loaded components only; never discover or load another component."""
    partitions: list[CudaNativePartition] = []
    with _LOCK:
        for name, loaded in sorted(_LOADED.items()):
            count = loaded.functions["phx_cuda_registry_partitions"]()
            _check_status(loaded, min(count, 0), "registry partitions")
            for index in range(count):
                device = _I32()
                partition, live, capacity, rebuilds = (_I64() for _ in range(4))
                _check_status(
                    loaded,
                    loaded.functions["phx_cuda_registry_partition"](
                        index,
                        ctypes.byref(device),
                        ctypes.byref(partition),
                        ctypes.byref(live),
                        ctypes.byref(capacity),
                        ctypes.byref(rebuilds),
                    ),
                    "registry partition",
                )
                partitions.append(
                    CudaNativePartition(
                        component=name,
                        device=device.value,
                        partition=partition.value,
                        live=live.value,
                        capacity=capacity.value,
                        rebuilds=rebuilds.value,
                    )
                )
        return CudaNativeResources(
            components=tuple(value.identity for _, value in sorted(_LOADED.items())),
            partitions=tuple(
                sorted(
                    partitions,
                    key=lambda item: (item.component, item.device, item.partition),
                )
            ),
        )


def _synchronize_devices() -> None:
    facts = collect_cuda_environment(initialize_backend=False)
    majors = {loaded.identity.major for loaded in _LOADED.values()}
    runtime_paths = tuple(
        path
        for path, major in facts.wheel_library_majors
        if major in majors and Path(path).name.startswith("libcudart.so.")
    )
    if not runtime_paths:
        # Fixtures without any CUDA dependency have no device work to drain.
        if any(loaded.identity.preloaded for loaded in _LOADED.values()):
            raise RuntimeError(
                "Cannot synchronize CUDA devices: NVIDIA runtime wheel is missing."
            )
        return
    runtime = ctypes.CDLL(runtime_paths[0])
    set_device = runtime.cudaSetDevice
    set_device.argtypes = [_I32]
    set_device.restype = _I32
    synchronize = runtime.cudaDeviceSynchronize
    synchronize.argtypes = []
    synchronize.restype = _I32
    failures: list[str] = []
    for device in jax.local_devices():
        if device_platform(device) == "cuda":
            status = set_device(device.local_hardware_id)
            if status == 0:
                status = synchronize()
            if status != 0:
                failures.append(
                    f"CUDA device {device.local_hardware_id} exit synchronization failed (status {status})."
                )
    if failures:
        raise RuntimeError("; ".join(failures))


def _shutdown() -> None:
    # Registered after JAX's own handlers, hence executed before JAX tears down.
    with _LOCK:
        for operation in (jax.effects_barrier, _synchronize_devices):
            try:
                operation()
            except Exception as error:
                emit(
                    "ERROR",
                    "cuda.native.shutdown_failed",
                    f"CUDA exit barrier failed: {error}",
                    operation=operation.__name__,
                )
        for loaded in _LOADED.values():
            for symbol in ("phx_cuda_release_all", "phx_cuda_shutdown"):
                try:
                    function = loaded.functions[symbol]
                    status = (
                        function(ctypes.byref(_I64()))
                        if symbol == "phx_cuda_release_all"
                        else function()
                    )
                    _check_status(loaded, status, symbol)
                except Exception as error:
                    emit(
                        "ERROR",
                        "cuda.native.shutdown_failed",
                        f"CUDA {loaded.identity.name} exit cleanup failed: {error}",
                        component=loaded.identity.name,
                        operation=symbol,
                    )
