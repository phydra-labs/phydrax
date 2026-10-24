# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Sole JAX discovery entry point for the source-pinned Windows CUDA runtime.

JAX catches initialize() exceptions. CUDA is ordered before other default
factories so its refusal cannot leave a usable partial CPU backend cache.
Admission failures also guard the top-level backend lookup, including when
the registration hook itself is untrusted.
"""

from __future__ import annotations

import ctypes
import functools
import importlib
import os
import sys
from threading import RLock
from typing import Any, Literal, NamedTuple, NoReturn

from . import _jax_contract, runtime


_LOCK = RLock()
_ERROR: Exception | None = None
_STATE: Literal[
    "uninitialized", "cpu_selected", "initializing", "registered", "failed"
] = "uninitialized"
_CAPSULE: Any = None
_ALLOCATOR: str | None = None
_PREALLOCATE: bool | None = None
_CAPABILITIES: tuple[str, ...] = ()


class BootstrapState(NamedTuple):
    status: str
    error: str | None
    allocator: str | None
    preallocate: bool | None
    capabilities: tuple[str, ...]


def bootstrap_state() -> BootstrapState:
    """Return process-local registration evidence, not numerical qualification."""
    return BootstrapState(
        _STATE,
        str(_ERROR) if _ERROR is not None else None,
        _ALLOCATOR,
        _PREALLOCATE,
        _CAPABILITIES,
    )


def _refuse() -> NoReturn:
    message = "The installed Windows CUDA 13 runtime could not initialize"
    if _ERROR is not None:
        raise RuntimeError(f"{message}: {_ERROR}") from _ERROR
    raise RuntimeError(f"{message}: bootstrap did not complete")


class _RefusalRegistration:
    """Failure-only structural JAX registration; never invokes an untrusted hook.

    JAX 0.12's registry consumers read these five BackendRegistration fields.
    This entry is only used after admission failed, not to register a native API.
    """

    factory = staticmethod(_refuse)
    priority = 500
    fail_quietly = False
    experimental = False
    c_api = None


def _order_cuda_first(xb: Any) -> None:
    # Discovery precedes JAX's backend lock. Preserve the dictionary identity and
    # every other plugin registration while preventing a CPU-first partial cache.
    with xb._backend_lock:
        registrations = xb._backend_factories
        cuda = registrations["cuda"]
        others = [(name, value) for name, value in registrations.items() if name != "cuda"]
        registrations.clear()
        registrations["cuda"] = cuda
        registrations.update(others)


def _retain_admission_refusal(xb: Any, hardware_utils: Any) -> None:
    # Discovery catches initialize() errors. Guard future lookups, and also
    # install a refusal for a stock backends() already executing discovery.
    # No replaced factory is called on this failure path.
    vars(xb)["backends"] = _refuse
    vars(hardware_utils)["has_visible_nvidia_gpu"] = _route_cuda_selection_to_backend
    with xb._backend_lock:
        xb._backend_factories["cuda"] = _RefusalRegistration()
    _order_cuda_first(xb)


def _guard_client_factory(xb: Any) -> None:
    # Client creation happens later, under JAX's backend lock. Retain refusal
    # there too: an explicit "cpu,cuda" order can already have populated CPU.
    with xb._backend_lock:
        registration = xb._backend_factories["cuda"]
        factory = registration.factory

        def create_client() -> Any:
            global _STATE, _ERROR
            try:
                return factory()
            except Exception as error:
                _ERROR = error
                _STATE = "failed"
                # Do not reacquire JAX's non-reentrant lock from its factory.
                vars(xb)["backends"] = _refuse
                _refuse()

        registration.factory = create_client



def _route_cuda_selection_to_backend() -> bool:
    # This is selection routing, not a claim about observed GPU presence. The
    # real backend/driver owns detection and refusal, including a zero-device host.
    return True


def _options() -> dict[str, str | int | float | list[int]]:
    global _ALLOCATOR, _PREALLOCATE
    from jax._src.lib import xla_client

    options = dict(xla_client.generate_pjrt_gpu_plugin_options())
    if (
        "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ
        and "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
    ):
        options["preallocate"] = False
    allocator, preallocate = options["allocator"], options.get("preallocate")
    if not isinstance(allocator, str):
        raise RuntimeError("JAX returned a non-string CUDA allocator option")
    if preallocate is not None and not isinstance(preallocate, bool):
        raise RuntimeError("JAX returned a non-boolean CUDA preallocation option")
    _ALLOCATOR = allocator
    _PREALLOCATE = preallocate
    return options


def _register_kernels(c_api: Any) -> None:
    from jax._src.lib import xla_client

    extension = importlib.import_module("jax_cuda13_plugin.cuda_plugin_extension")
    xla_client.register_custom_type_handler(
        "CUDA", functools.partial(extension.register_custom_type, c_api)
    )
    xla_client.register_custom_call_handler(
        "CUDA", functools.partial(extension.register_custom_call_target, c_api)
    )
    for name, value in extension.ffi_types().items():
        xla_client.register_custom_type(name, value, platform="CUDA")
    for name, value in extension.ffi_handlers().items():
        xla_client.register_custom_call_target(
            name, value, platform="CUDA", api_version=1
        )
    # Ordinary XLA fusion/code generation is compiled into PJRT. This runtime
    # does not declare JAX Pallas/Triton or Mosaic Python lowering support.


def _register_native(xb: Any) -> tuple[str, ...]:
    global _CAPSULE
    from jax._src import lib
    from jax._src.lib import _profiler

    record = runtime.installation_record()
    runtime.driver_version()
    runtime.prepare_kernels()
    root = runtime.compiler_root()
    # _cuda_path is source admitted above. The Windows compiler assets are not
    # found by stock JAX's POSIX executable-name probe; use the admitted wheel.
    lib.cuda_path = root
    pjrt = next(asset.path for asset in record.assets if asset.role == "pjrt")
    runtime.load_libraries(pjrt)
    library = runtime.library_handle(pjrt)
    get_api = library.GetPjrtApi
    get_api.argtypes = []
    get_api.restype = ctypes.c_void_p
    pointer = get_api()
    if not pointer:
        raise RuntimeError("Windows PJRT GetPjrtApi returned a null pointer")
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule_new.restype = ctypes.py_object
    _CAPSULE = capsule_new(pointer, b"pjrt_c_api", None)
    # This replaces the refusal factory synchronously, while discovery is still
    # running; JAX's later callbacks see the real registered C API.
    c_api = xb.register_plugin("cuda", priority=500, c_api=_CAPSULE, options=_options)
    if c_api is None:
        raise RuntimeError("JAX did not register the Windows CUDA PJRT C API")
    _profiler.register_plugin_profiler(c_api)
    _register_kernels(c_api)
    if "sparse_qr" in record.capabilities:
        registered = lib.gpu_solver.registrations()["CUDA"]
        if not any(
            name == "cusolver_csrlsvqr_ffi" and capsule is not None and version == 1
            for name, capsule, version in registered
        ):
            raise RuntimeError(
                "The runtime declares sparse QR but JAX has no bound cusolver_csrlsvqr_ffi handler"
            )
    return record.capabilities


def initialize() -> None:
    """Register CUDA or retain an actionable, non-quiet refusal backend."""
    global _STATE, _ERROR, _CAPABILITIES
    # This standalone discovery owner only reads selection; importing core's
    # configuration owner here would introduce the prohibited JAX/core cycle.
    from jax._src import config, hardware_utils, lib, xla_bridge as xb
    from jax._src.lib import xla_client

    with _LOCK:
        if _STATE == "registered":
            return
        if _STATE == "failed":
            _refuse()
        platforms = config.jax_platforms.value
        if platforms and not (
            {part.strip() for part in platforms.split(",")} & {"cuda", "gpu"}
        ):
            _STATE = "cpu_selected"
            return
        if sys.platform != "win32" or sys.version_info[:2] != (3, 13):
            raise RuntimeError(
                "The CUDA PJRT runtime requires standard CPython 3.13 on native Windows x64"
            )
        try:
            # Authenticate the critical callable before invoking it. Remaining
            # checks run only after refusal and Windows selection routing exist.
            _jax_contract.validate_loaded_hook(
                xb, "register_backend_factory", xb.register_backend_factory
            )
            if "cuda" in xb._backend_factories:
                raise RuntimeError(
                    "A competing CUDA backend is already registered; install exactly one CUDA plugin"
                )
            xb.register_backend_factory("cuda", _refuse, priority=500, fail_quietly=False)
            _order_cuda_first(xb)
            original_visibility = hardware_utils.has_visible_nvidia_gpu
            vars(hardware_utils)["has_visible_nvidia_gpu"] = _route_cuda_selection_to_backend
            _STATE = "initializing"
            _jax_contract.validate_installed_sources()
            _jax_contract.validate_loaded_hooks(xb, lib, xla_client)
            _jax_contract.validate_loaded_hook(
                hardware_utils, "has_visible_nvidia_gpu", original_visibility
            )
            _CAPABILITIES = _register_native(xb)
            _guard_client_factory(xb)
        except Exception as error:
            _ERROR = error
            _STATE = "failed"
            # Also replaces a partially registered native factory after profiler
            # or FFI registration fails. No partial capability remains callable.
            _retain_admission_refusal(xb, hardware_utils)
            raise RuntimeError(f"Windows CUDA PJRT bootstrap failed: {error}") from error
        _STATE = "registered"
