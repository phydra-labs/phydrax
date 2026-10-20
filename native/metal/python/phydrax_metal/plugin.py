#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""JAX plugin entry point for the phydrax-metal ``mps`` PJRT backend.

JAX's plugin discovery imports this module and calls :func:`initialize`. It
requires standard-GIL CPython 3.13, validates the package-local build
(``build.json``: exact JAX/JAXlib pair, asset hashes, platform), refuses
competing ``mps``/``metal`` plugins, registers the
package-local PJRT library, installs the native host-callback bridge, and
registers ``mps`` lowering rules for JAX's host-callback and gather primitives.

There is no library search outside this package, no global JAX callable
replacement, no Shardy or other configuration change, and no CPU fallback.
"""

from __future__ import annotations

import ctypes
import hashlib
import itertools
import json
import sys
import sysconfig
import threading
import traceback
import weakref
from collections.abc import Callable, Sequence
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import Any, Final, TYPE_CHECKING

import ml_dtypes
import numpy as np


__all__ = ["PLATFORM", "PluginError", "initialize"]

PLATFORM: Final = "mps"
DISTRIBUTION: Final = "phydrax-metal"
HOST_CALLBACK_TARGET: Final = "phydrax_metal.host_callback"
# Below the CPU backend (priority 0): installing the companion never changes the
# default device; the Apple runtime is selected explicitly (jax_platforms).
PRIORITY: Final = -1
_PACKAGE: Final = Path(__file__).resolve().parent
_COMPETING_PLUGIN_NAMES: Final = frozenset(("mps", "metal"))
_ERROR_CAPACITY_FLOOR: Final = 2

# PJRT_Buffer_Type values (xla/pjrt/c/pjrt_c_api.h) for the host-callback ABI.
_PJRT_NUMPY_DTYPES: Final[dict[int, np.dtype[Any]]] = {
    1: np.dtype(np.bool_),
    2: np.dtype(np.int8),
    3: np.dtype(np.int16),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    6: np.dtype(np.uint8),
    7: np.dtype(np.uint16),
    8: np.dtype(np.uint32),
    9: np.dtype(np.uint64),
    10: np.dtype(np.float16),
    11: np.dtype(np.float32),
    13: np.dtype(ml_dtypes.bfloat16),
    14: np.dtype(np.complex64),
}


class PluginError(RuntimeError):
    """The installed companion cannot provide the ``mps`` runtime."""


class _HostBuffer(ctypes.Structure):
    _fields_ = (
        ("data", ctypes.c_void_p),
        ("element_type", ctypes.c_int32),
        ("rank", ctypes.c_int32),
        ("dims", ctypes.POINTER(ctypes.c_int64)),
    )


_TRAMPOLINE_TYPE: Final = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.c_uint64,
    ctypes.c_int32,
    ctypes.POINTER(_HostBuffer),
    ctypes.c_int32,
    ctypes.POINTER(_HostBuffer),
    ctypes.c_void_p,
    ctypes.c_int32,
)

_lock = threading.Lock()
_callback_ids = itertools.count(1)
# Weak references only: each executable that can invoke a callback owns it
# through JAX's host-callback list, so the registry never extends its lifetime.
_callbacks: dict[int, weakref.ReferenceType[Callable[..., Sequence[np.ndarray]]]] = {}
_registered_library: Path | None = None


def _require_supported_interpreter() -> None:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13):
        raise PluginError(f"{DISTRIBUTION} requires standard-GIL CPython 3.13")
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise PluginError(
            f"{DISTRIBUTION} requires standard-GIL CPython 3.13; "
            "free-threaded CPython builds are not supported"
        )


def _manifest() -> dict[str, Any]:
    path = _PACKAGE / "build.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PluginError(
            f"{DISTRIBUTION} is missing its build manifest {path}"
        ) from error
    if not isinstance(record, dict):
        raise PluginError(f"{path} is not a build manifest")
    identity = {key: value for key, value in record.items() if key != "build_id"}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != record.get("build_id"):
        raise PluginError(f"{path} does not match its build_id")
    return record


def _validate(record: dict[str, Any]) -> Path:
    from jax._src.lib import version_str as imported_jaxlib
    from jax.version import __version__ as imported_jax

    if record.get("platform") != PLATFORM:
        raise PluginError(
            f"{DISTRIBUTION} manifest names platform {record.get('platform')!r}"
        )
    installed = metadata.version(DISTRIBUTION)
    if record.get("version") != installed:
        raise PluginError(
            f"{DISTRIBUTION} {installed} carries a build manifest for {record.get('version')}"
        )
    for distribution, key, imported in (
        ("jax", "jax_version", imported_jax),
        ("jaxlib", "jaxlib_version", imported_jaxlib),
    ):
        found = metadata.version(distribution)
        if found != record.get(key) or imported != record.get(key):
            raise PluginError(
                f"{DISTRIBUTION} {installed} was built for {distribution}=="
                f"{record.get(key)}, but {distribution} {found} is installed "
                f"(imported {imported})"
            )
    assets = record.get("assets")
    library = record.get("library")
    if (
        not isinstance(assets, dict)
        or "_jax_indexing.py" not in assets
        or not isinstance(library, str)
        or library not in assets
    ):
        raise PluginError(
            f"{DISTRIBUTION} manifest does not identify its library and indexing helper"
        )
    for relative, expected in assets.items():
        path = (_PACKAGE / relative).resolve()
        if not path.is_relative_to(_PACKAGE) or not path.is_file():
            raise PluginError(f"{DISTRIBUTION} asset {relative} is missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise PluginError(f"{DISTRIBUTION} asset {relative} does not match its build")
    return _PACKAGE / library


def _refuse_competing_plugins() -> None:
    own = metadata.distribution(DISTRIBUTION)
    for entry_point in metadata.entry_points(group="jax_plugins"):
        foreign = entry_point.dist is None or entry_point.dist.name != own.name
        if foreign and entry_point.name.lower() in _COMPETING_PLUGIN_NAMES:
            owner = entry_point.dist.name if entry_point.dist is not None else "unknown"
            raise PluginError(
                f"JAX plugin {entry_point.name!r} from {owner} competes with {DISTRIBUTION}; "
                "uninstall it so exactly one Apple GPU provider is installed"
            )
    try:
        import jax_plugins  # ty: ignore[unresolved-import]
    except ImportError:
        return
    import pkgutil

    for module in pkgutil.iter_modules(jax_plugins.__path__):
        if module.name.lower() in _COMPETING_PLUGIN_NAMES:
            raise PluginError(
                f"JAX plugin module jax_plugins.{module.name} competes with {DISTRIBUTION}; "
                "uninstall it so exactly one Apple GPU provider is installed"
            )


def _write_error(error: int, capacity: int, message: str) -> None:
    if capacity < _ERROR_CAPACITY_FLOOR or not error:
        return
    data = message.encode("utf-8", "replace")
    # Keep the end of a long traceback: it names the raised exception.
    data = data[-(capacity - 1) :]
    ctypes.memmove(error, data, len(data))
    ctypes.memset(error + len(data), 0, 1)


def _trampoline(
    callback_id: int,
    num_inputs: int,
    inputs: Any,
    num_outputs: int,
    outputs: Any,
    error: int,
    capacity: int,
) -> int:
    try:
        with _lock:
            reference = _callbacks.get(callback_id)
        callback = reference() if reference is not None else None
        if callback is None:
            raise RuntimeError(
                f"phydrax-metal host callback {callback_id} is no longer alive"
            )
        arguments = []
        for index in range(num_inputs):
            buffer = inputs[index]
            dtype = _PJRT_NUMPY_DTYPES[buffer.element_type]
            shape = tuple(buffer.dims[axis] for axis in range(buffer.rank))
            count = int(np.prod(shape, dtype=np.int64))
            if count == 0:
                arguments.append(np.zeros(shape, dtype))
                continue
            raw = (ctypes.c_char * (count * dtype.itemsize)).from_address(buffer.data)
            arguments.append(np.frombuffer(raw, dtype).reshape(shape).copy())
        results = callback(*arguments)
        if len(results) != num_outputs:
            raise RuntimeError(
                f"Mismatched number of outputs from callback. Expected: {num_outputs}, "
                f"Actual: {len(results)}"
            )
        for index, value in enumerate(results):
            buffer = outputs[index]
            dtype = _PJRT_NUMPY_DTYPES[buffer.element_type]
            shape = tuple(buffer.dims[axis] for axis in range(buffer.rank))
            array = np.asarray(value, order="C")
            if array.shape != shape:
                raise RuntimeError(
                    f"Incorrect output shape for return value #{index}: "
                    f"Expected: {shape}, Actual: {array.shape}"
                )
            if array.dtype != dtype:
                raise RuntimeError(
                    f"Incorrect output dtype for return value #{index}: "
                    f"Expected: {dtype}, Actual: {array.dtype}"
                )
            if array.nbytes:
                ctypes.memmove(buffer.data, array.ctypes.data, array.nbytes)
    except BaseException as exception:
        _write_error(error, capacity, "".join(traceback.format_exception(exception)))
        return 1
    return 0


_native_trampoline: Final = _TRAMPOLINE_TYPE(_trampoline)


def _forget(callback_id: int, _reference: object) -> None:
    with _lock:
        _callbacks.pop(callback_id, None)


def _install_bridge(library: Path) -> None:
    native = ctypes.CDLL(str(library))
    setter = native.PhydraxMetal_SetHostCallbackTrampoline
    setter.argtypes = (_TRAMPOLINE_TYPE,)
    setter.restype = ctypes.c_int32
    if setter(_native_trampoline) != 0:
        raise PluginError(
            f"{library} already has a different host-callback bridge installed"
        )


def _emit_host_callback(
    ctx: Any,
    callback: Callable[..., Sequence[np.ndarray]],
    token: Any,
    operands: Sequence[Any],
    *,
    has_side_effect: bool,
) -> tuple[list[Any], Any]:
    from jax._src.interpreters import mlir
    from jax._src.lib.mlir.dialects import hlo

    callback_id = next(_callback_ids)
    with _lock:
        _callbacks[callback_id] = weakref.ref(callback, partial(_forget, callback_id))
    # The executable owns the callable from here on (and through in-flight runs).
    ctx.module_context.add_host_callback(callback)
    result_types = [
        mlir.aval_to_ir_type(ctx.module_context, aval) for aval in ctx.avals_out
    ]
    if token is not None:
        result_types = [hlo.TokenType.get(), *result_types]
        operands = [token, *operands]
    call = mlir.custom_call(
        HOST_CALLBACK_TARGET,
        result_types=result_types,
        operands=list(operands),
        backend_config=json.dumps({"callback_id": callback_id}),
        has_side_effect=has_side_effect,
    )
    results = list(call.results)
    if token is not None:
        return results[1:], results[0]
    return results, None


def _flat(callback: Callable[..., Any]) -> Callable[..., tuple[np.ndarray, ...]]:
    def run(*arguments: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(np.asarray(value) for value in callback(*arguments))

    return run


def _pure_callback_rule(ctx: Any, *args: Any, callback: Any, **params: Any) -> list[Any]:
    del params
    results, _ = _emit_host_callback(
        ctx, _flat(callback), None, args, has_side_effect=False
    )
    return results


def _io_callback_rule(
    ctx: Any, *args: Any, callback: Any, ordered: bool, **params: Any
) -> list[Any]:
    from jax._src import callback as jax_callback
    from jax._src.interpreters import mlir

    del params
    if not ordered:
        results, _ = _emit_host_callback(
            ctx, _flat(callback), None, args, has_side_effect=True
        )
        return results
    effect = jax_callback._OrderedIOEffect
    results, token = _emit_host_callback(
        ctx, _flat(callback), ctx.tokens_in.get(effect), args, has_side_effect=True
    )
    ctx.set_tokens_out(ctx.tokens_in.update_tokens(mlir.TokenSet({effect: token})))
    return results


def _debug_callback_rule(
    ctx: Any, *args: Any, effect: Any, partitioned: bool, callback: Any
) -> list[Any]:
    from jax._src import effects
    from jax._src.interpreters import mlir

    del partitioned  # A single device: partitioned and global callbacks coincide.

    def run(*arguments: np.ndarray) -> tuple[np.ndarray, ...]:
        callback(*arguments)
        return ()

    if not effects.ordered_effects.contains(effect):
        results, _ = _emit_host_callback(ctx, run, None, args, has_side_effect=True)
        return results
    results, token = _emit_host_callback(
        ctx, run, ctx.tokens_in.get(effect), args, has_side_effect=True
    )
    ctx.set_tokens_out(ctx.tokens_in.update_tokens(mlir.TokenSet({effect: token})))
    return results


def _debug_print_rule(
    ctx: Any,
    *args: Any,
    fmt: str,
    ordered: bool,
    partitioned: bool,
    in_tree: Any,
    static_args: Any,
    np_printoptions: Any,
    has_placeholders: bool,
    logging_record: Any,
) -> list[Any]:
    from jax._src import debugging

    callback = partial(
        debugging._format_print_callback,
        fmt,
        dict(np_printoptions),
        has_placeholders,
        logging_record,
    )
    callback = debugging._make_flat_callback(in_tree, callback, static_args)
    effect = debugging.ordered_debug_effect if ordered else debugging.debug_effect
    return _debug_callback_rule(
        ctx, *args, effect=effect, partitioned=partitioned, callback=callback
    )


def _register_lowerings() -> None:
    from jax._src import callback as jax_callback, debugging
    from jax._src.interpreters import mlir

    if TYPE_CHECKING:
        from phydrax._jax_indexing import register_gather_lowering
    else:
        # The wheel packages this exact canonical source under its private owner.
        from ._jax_indexing import register_gather_lowering

    register_gather_lowering(PLATFORM)

    # Platform-specific rules only; CPU/GPU lowerings and every public JAX
    # callable stay untouched. Not cacheable: each lowering owns its callable.
    for primitive, rule in (
        (jax_callback.pure_callback_p, _pure_callback_rule),
        (jax_callback.io_callback_p, _io_callback_rule),
        (debugging.debug_callback_p, _debug_callback_rule),
        (debugging.debug_print_p, _debug_print_rule),
    ):
        mlir.register_lowering(primitive, rule, platform=PLATFORM, cacheable=False)


def initialize() -> None:
    """Register the package-local ``mps`` PJRT plugin with JAX.

    Called by JAX plugin discovery. Raises :class:`PluginError` for an unsupported
    interpreter, when the build does not match the installed JAX pair, a native
    asset differs from its build, or another Apple GPU plugin is installed.
    """
    global _registered_library
    _require_supported_interpreter()
    from jax._src import xla_bridge

    record = _manifest()
    library = _validate(record)
    _refuse_competing_plugins()
    with _lock:
        if _registered_library is not None:
            if _registered_library == library:
                return
            raise PluginError(
                f"{PLATFORM} is already registered from {_registered_library}"
            )
        if PLATFORM in xla_bridge._backend_factories:
            raise PluginError(
                f"another provider already registered the {PLATFORM!r} platform"
            )
        xla_bridge.register_plugin(PLATFORM, priority=PRIORITY, library_path=str(library))
        _registered_library = library
    _install_bridge(library)
    _register_lowerings()
