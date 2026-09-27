#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host binding of the native Phydrax meshcore library.

The optional ``phydrax-meshcore`` distribution ships a C++ shared library with a
plain C ABI (``native/meshcore/include/phydrax_meshcore.h``) that is loaded with
``ctypes``.  It owns exact geometric predicates (Shewchuk adaptive expansion
arithmetic with index-ordered symbolic perturbation), exactly classified convex
clipping moments, and exact-predicate Delaunay/regular/constrained
triangulations.  Every function here is host-only NumPy interchange: inputs are
validated and converted to contiguous float64/int32 arrays, results are fresh
arrays, and native statuses reach the caller (per-item status arrays for batched
clipping, exceptions for rejected calls).

Library lookup order: the ``PHYDRAX_MESHCORE_LIBRARY`` environment variable
(path to the shared library), then ``phydrax_meshcore.library_path()``.  A
library is usable only when it exports every C ABI symbol bound here and its
version equals the installed ``phydrax`` release; anything else is reported as
:class:`MeshcoreUnavailableError` with the reason.
"""

from __future__ import annotations

import ctypes
import functools
import importlib
import importlib.metadata
import importlib.util
import os
from collections.abc import Callable, Mapping
from enum import IntEnum
from pathlib import Path
from typing import Any, final

import numpy as np


_ENVIRONMENT = "PHYDRAX_MESHCORE_LIBRARY"
_MAX_POINTS = 2147483646


class MeshcoreUnavailableError(ImportError):
    """The native meshcore library is not installed or cannot be loaded."""


class MeshcoreError(RuntimeError):
    """The native meshcore library refused a call for resource or internal reasons."""

    def __init__(self, status: MeshcoreStatus, message: str, /) -> None:
        super().__init__(f"{message} (meshcore status {status.name})")
        self.status = status


class MeshcoreStatus(IntEnum):
    """Call and item status codes of the meshcore C ABI."""

    OK = 0
    INVALID_ARGUMENT = 1
    NONFINITE_INPUT = 2
    RANGE_ERROR = 3
    DEGENERATE_INPUT = 4
    INVALID_INPUT = 5
    CAPACITY_EXCEEDED = 6
    CONSTRAINT_INTERSECTION = 7
    REFINEMENT_LIMIT = 8
    INTERNAL_ERROR = 9


_P = ctypes.c_void_p
_I32 = ctypes.c_int32
_I64 = ctypes.c_int64
_F64 = ctypes.c_double

_SIGNATURES: dict[str, tuple[object, tuple[object, ...]]] = {
    "phx_mc_version": (ctypes.c_char_p, ()),
    "phx_mc_build_hash": (ctypes.c_char_p, ()),
    "phx_mc_exact_domain": (None, (_P, _P)),
    "phx_mc_orient2d": (_I32, (_I64, _P, _P, _P, _P)),
    "phx_mc_orient3d": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_incircle": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_insphere": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_orient2d_sos": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_orient3d_sos": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_incircle_sos": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_insphere_sos": (_I32, (_I64, _P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_polygon_intersection_moments": (
        _I32,
        (_I64, _I32, _P, _P, _I32, _P, _P, _P, _P, _P),
    ),
    "phx_mc_tetrahedron_intersection_moments": (
        _I32,
        (_I64, _P, _P, _I32, _P, _P, _P),
    ),
    "phx_mc_polygon_intersection_simplices": (
        _I32,
        (_I64, _I32, _P, _P, _I32, _P, _P, _I32, _P, _P, _P, _P, _P),
    ),
    "phx_mc_tetrahedron_intersection_simplices": (
        _I32,
        (_I64, _P, _P, _I32, _I32, _P, _P, _P, _P, _P),
    ),
    "phx_mc_polyhedron_clip_moments": (
        _I32,
        (_I64, _I32, _P, _P, _P, _P, _I32, _P, _P, _P),
    ),
    "phx_mc_clip_box_halfplanes": (
        _I32,
        (_I64, _P, _P, _I32, _P, _P, _P, _I32, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_clip_box_halfspaces": (
        _I32,
        (
            _I64,
            _P,
            _P,
            _I32,
            _P,
            _P,
            _P,
            _I32,
            _I32,
            _I32,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
        ),
    ),
    "phx_mc_delaunay_2d": (_I32, (_I64, _P, _I64, _P)),
    "phx_mc_regular_2d": (_I32, (_I64, _P, _P, _I64, _P)),
    "phx_mc_delaunay_3d": (_I32, (_I64, _P, _I64, _P)),
    "phx_mc_regular_3d": (_I32, (_I64, _P, _P, _I64, _P)),
    "phx_mc_constrained_delaunay_2d": (
        _I32,
        (_I64, _P, _I64, _P, _I64, _P, _I32, _F64, _F64, _I64, _I64, _P),
    ),
    "phx_mc_mesh_dimension": (_I32, (_P,)),
    "phx_mc_mesh_point_count": (_I64, (_P,)),
    "phx_mc_mesh_cell_count": (_I64, (_P,)),
    "phx_mc_mesh_input_point_count": (_I64, (_P,)),
    "phx_mc_mesh_copy_points": (None, (_P, _P)),
    "phx_mc_mesh_copy_cells": (None, (_P, _P)),
    "phx_mc_mesh_copy_vertex_map": (None, (_P, _P)),
    "phx_mc_mesh_copy_cell_segments": (None, (_P, _P)),
    "phx_mc_mesh_free": (None, (_P,)),
}


def _library_location(configured: str, /) -> Path | str:
    """Return the library path, or the reason it is unavailable."""

    if configured:
        path = Path(configured)
        if not path.is_file():
            return f"{_ENVIRONMENT} names a missing meshcore library: {configured!r}."
        return path
    if importlib.util.find_spec("phydrax_meshcore") is None:
        return (
            "The meshcore library is unavailable: install the optional "
            "'phydrax[meshcore]' extra or set "
            f"{_ENVIRONMENT} to libphydrax_meshcore."
        )
    return Path(importlib.import_module("phydrax_meshcore").library_path())


@final
class MeshcoreLibrary:
    """Loaded meshcore shared library with typed C entry points."""

    __slots__ = ("_functions", "_library", "build_hash", "path", "version")

    def __init__(
        self,
        library: ctypes.CDLL,
        functions: Mapping[str, Callable[..., object]],
        path: Path,
        version: str,
        build_hash: str,
        /,
    ) -> None:
        self._library = library
        self._functions = functions
        self.path = path
        self.version = version
        self.build_hash = build_hash

    def __getitem__(self, name: str, /) -> Any:
        return self._functions[name]


def _identity_text(function: Callable[[], object], symbol: str, path: Path, /) -> Any:
    """Decode one required non-null ASCII identity string, or return its error."""

    raw = function()
    if not isinstance(raw, bytes):
        return None, f"meshcore library {str(path)!r} returned null from {symbol}."
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError:
        return None, f"meshcore library {str(path)!r} returned non-ASCII {symbol}."
    if not value:
        return None, f"meshcore library {str(path)!r} returned an empty {symbol}."
    return value, None


def _bind(library: ctypes.CDLL, path: Path, /) -> MeshcoreLibrary | str:
    """Bind every C ABI entry point and verify the release, or return the reason."""

    functions = {}
    missing = []
    for name, (restype, argtypes) in _SIGNATURES.items():
        # ctypes reports an unresolved symbol only as AttributeError.
        try:
            function = library[name]
        except AttributeError:
            missing.append(name)
            continue
        # ty: ignore[invalid-assignment]
        function.restype = restype
        # ty: ignore[invalid-assignment]
        function.argtypes = argtypes
        functions[name] = function
    if missing:
        return (
            f"meshcore library {str(path)!r} lacks C ABI symbols "
            f"{', '.join(missing)}; install the phydrax-meshcore release of this phydrax."
        )
    version, error = _identity_text(functions["phx_mc_version"], "phx_mc_version", path)
    if error is not None:
        return error
    build_hash, error = _identity_text(
        functions["phx_mc_build_hash"], "phx_mc_build_hash", path
    )
    if error is not None:
        return error
    if len(build_hash) != 64 or any(
        character not in "0123456789abcdef" for character in build_hash
    ):
        return f"meshcore library {str(path)!r} returned an invalid phx_mc_build_hash."
    try:
        required = importlib.metadata.version("phydrax")
    except importlib.metadata.PackageNotFoundError:
        return (
            "The meshcore release cannot be verified: phydrax distribution "
            "metadata is not installed."
        )
    if version != required:
        return (
            f"meshcore library {str(path)!r} is release {version}, but phydrax "
            f"{required} requires phydrax-meshcore=={required}."
        )
    return MeshcoreLibrary(library, functions, path, version, build_hash)


@functools.cache
def _resolve(configured: str, /) -> MeshcoreLibrary | str:
    location = _library_location(configured)
    if isinstance(location, str):
        return location
    # ctypes reports every loader failure as OSError; it is the only signal.
    try:
        library = ctypes.CDLL(str(location))
    except OSError as error:
        return f"meshcore library {str(location)!r} cannot be loaded: {error}"
    return _bind(library, location)


def _current() -> MeshcoreLibrary | str:
    return _resolve(os.environ.get(_ENVIRONMENT, "").strip())


def meshcore_available() -> bool:
    """Whether the native meshcore library can be loaded."""

    return not isinstance(_current(), str)


def load_meshcore() -> MeshcoreLibrary:
    """Return the loaded meshcore library or raise :class:`MeshcoreUnavailableError`."""

    resolved = _current()
    if isinstance(resolved, str):
        raise MeshcoreUnavailableError(resolved)
    return resolved


def meshcore_identity() -> str:
    """Library release and source build hash: ``phydrax-meshcore <release> <sha256>``."""

    library = load_meshcore()
    return f"phydrax-meshcore {library.version} {library.build_hash}"


def meshcore_exact_domain() -> tuple[int, int]:
    """Binary exponent range ``(minimum, maximum)`` of exactly handled coordinates.

    Coordinates must be zero or have magnitude in ``[2**minimum, 2**maximum]``;
    weights use the doubled range.
    """

    library = load_meshcore()
    minimum = np.zeros((1,), dtype=np.int32)
    maximum = np.zeros((1,), dtype=np.int32)
    library["phx_mc_exact_domain"](minimum.ctypes.data, maximum.ctypes.data)
    return int(minimum[0]), int(maximum[0])


# ---------------------------------------------------------------- validation


def _real_array(value: object, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.bool_) or not (
        np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(f"{name} must be a real floating or integer array.")
    if np.issubdtype(array.dtype, np.integer) and np.any(
        (array > 2**53) | (array < -(2**53))
    ):
        raise ValueError(f"{name} integer values must be exactly representable.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _index_array(value: object, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be an integer array.")
    return array


def _points(value: object, name: str, width: int, /) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width}).")
    return array


def _batched_points(
    arrays: tuple[object, ...], names: tuple[str, ...], width: int, /
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    converted = tuple(
        _real_array(value, name) for value, name in zip(arrays, names, strict=True)
    )
    for array, name in zip(converted, names, strict=True):
        if array.ndim < 1 or array.shape[-1] != width:
            raise ValueError(f"{name} must have shape (..., {width}).")
    leading = np.broadcast_shapes(*(array.shape[:-1] for array in converted))
    flat = tuple(
        np.ascontiguousarray(
            np.broadcast_to(array, leading + (width,)).reshape((-1, width)),
            dtype=np.float64,
        )
        for array in converted
    )
    return flat, leading


def _raise_for_call(status: int, operation: str, /) -> None:
    code = MeshcoreStatus(status)
    match code:
        case MeshcoreStatus.OK:
            return
        case MeshcoreStatus.NONFINITE_INPUT:
            raise ValueError(f"{operation}: inputs must be finite.")
        case MeshcoreStatus.RANGE_ERROR:
            raise ValueError(
                f"{operation}: coordinates must be zero or have magnitude in "
                "[2**-120, 2**120] (weights [2**-240, 2**240]) for exact evaluation."
            )
        case MeshcoreStatus.DEGENERATE_INPUT:
            raise ValueError(f"{operation}: the distinct points do not span the space.")
        case MeshcoreStatus.INVALID_ARGUMENT | MeshcoreStatus.INVALID_INPUT:
            raise ValueError(f"{operation}: invalid input ({code.name}).")
        case MeshcoreStatus.CONSTRAINT_INTERSECTION:
            raise ValueError(
                f"{operation}: constraint segments cross in their interiors."
            )
        case MeshcoreStatus.CAPACITY_EXCEEDED:
            raise MeshcoreError(code, f"{operation}: explicit capacity exceeded")
        case MeshcoreStatus.REFINEMENT_LIMIT | MeshcoreStatus.INTERNAL_ERROR:
            raise MeshcoreError(code, f"{operation}: native failure")
        case _:
            raise ValueError(f"Unknown meshcore status {status}.")


# ---------------------------------------------------------------- predicates


def _predicate(
    name: str, arrays: tuple[object, ...], width: int, /, ids: object = None
) -> np.ndarray:
    library = load_meshcore()
    names = tuple(f"point {index}" for index in range(len(arrays)))
    flat, leading = _batched_points(arrays, names, width)
    count = flat[0].shape[0]
    signs = np.zeros((count,), dtype=np.int8)
    pointers = [array.ctypes.data for array in flat]
    if ids is None:
        status = library[name](count, *pointers, signs.ctypes.data)
    else:
        index = _index_array(ids, "ids")
        if index.shape[-1:] != (len(arrays),):
            raise ValueError(f"ids must have shape (..., {len(arrays)}).")
        index_flat = np.ascontiguousarray(
            np.broadcast_to(index, leading + (len(arrays),)).reshape((-1, len(arrays))),
            dtype=np.int64,
        )
        status = library[name](
            count, *pointers, index_flat.ctypes.data, signs.ctypes.data
        )
    _raise_for_call(status, name.removeprefix("phx_mc_"))
    return signs.reshape(leading)


def exact_orient2d(a: object, b: object, c: object, /) -> np.ndarray:
    """Exact ``sign det[b - a, c - a]`` for ``(..., 2)`` arrays (int8 in {-1, 0, 1})."""

    return _predicate("phx_mc_orient2d", (a, b, c), 2)


def exact_orient3d(a: object, b: object, c: object, d: object, /) -> np.ndarray:
    """Exact ``sign det[b - a, c - a, d - a]`` for ``(..., 3)`` arrays."""

    return _predicate("phx_mc_orient3d", (a, b, c, d), 3)


def exact_incircle(a: object, b: object, c: object, d: object, /) -> np.ndarray:
    """Exact in-circle sign: positive when ``d`` is inside the circle of CCW ``(a, b, c)``."""

    return _predicate("phx_mc_incircle", (a, b, c, d), 2)


def exact_insphere(
    a: object, b: object, c: object, d: object, e: object, /
) -> np.ndarray:
    """Exact in-sphere sign: positive when ``e`` is inside the sphere of positive ``(a..d)``."""

    return _predicate("phx_mc_insphere", (a, b, c, d, e), 3)


def symbolic_orient2d(a: object, b: object, c: object, ids: object, /) -> np.ndarray:
    """Index-ordered Simulation-of-Simplicity ``orient2d``; never zero for distinct ids."""

    return _predicate("phx_mc_orient2d_sos", (a, b, c), 2, ids)


def symbolic_orient3d(
    a: object, b: object, c: object, d: object, ids: object, /
) -> np.ndarray:
    """Index-ordered Simulation-of-Simplicity ``orient3d``; never zero for distinct ids."""

    return _predicate("phx_mc_orient3d_sos", (a, b, c, d), 3, ids)


def symbolic_incircle(
    a: object, b: object, c: object, d: object, ids: object, /
) -> np.ndarray:
    """Lifted-perturbation ``incircle``; zero only for four collinear points."""

    return _predicate("phx_mc_incircle_sos", (a, b, c, d), 2, ids)


def symbolic_insphere(
    a: object, b: object, c: object, d: object, e: object, ids: object, /
) -> np.ndarray:
    """Lifted-perturbation ``insphere``; zero only for five coplanar points."""

    return _predicate("phx_mc_insphere_sos", (a, b, c, d, e), 3, ids)


# ---------------------------------------------------------------- clipping


def _counted_polygons(
    vertices: object, counts: object, name: str, /
) -> tuple[np.ndarray, np.ndarray]:
    array = _real_array(vertices, name)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError(f"{name} must have shape (n, k, 2).")
    count_array = _index_array(counts, f"{name} counts")
    if count_array.shape != array.shape[:1]:
        raise ValueError(f"{name} counts must have shape (n,).")
    if np.any(count_array < 0) or np.any(count_array > array.shape[1]):
        raise ValueError(f"{name} counts must lie in [0, {array.shape[1]}].")
    return array, np.ascontiguousarray(count_array, dtype=np.int32)


def polygon_intersection_moments(
    first_vertices: object,
    first_counts: object,
    second_vertices: object,
    second_counts: object,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly classified intersection moments of convex polygon pairs.

    Polygons are ``(n, k, 2)`` padded vertex arrays with per-row counts, in either
    orientation.  Returns ``(area (n,), first_moment (n, 2), status (n,))`` where
    ``first_moment`` is the integral of ``x`` over the intersection and ``status``
    holds :class:`MeshcoreStatus` item codes (empty and contact overlaps are ``OK``
    with zero area).
    """

    library = load_meshcore()
    first, first_count = _counted_polygons(first_vertices, first_counts, "first_vertices")
    second, second_count = _counted_polygons(
        second_vertices, second_counts, "second_vertices"
    )
    if first.shape[0] != second.shape[0]:
        raise ValueError("first and second polygon batches must have equal length.")
    count = first.shape[0]
    area = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 2), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_polygon_intersection_moments"](
        count,
        first.shape[1],
        first.ctypes.data,
        first_count.ctypes.data,
        second.shape[1],
        second.ctypes.data,
        second_count.ctypes.data,
        area.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "polygon_intersection_moments")
    return area, moment, status


def polygon_intersection_simplices(
    first_vertices: object,
    first_counts: object,
    second_vertices: object,
    second_counts: object,
    /,
    *,
    simplex_capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convex polygon-pair intersections as triangle partitions plus their moments.

    Returns ``(simplices (n, S, 3, 2), simplex_counts (n,), area, first_moment,
    status)``.  Each nonempty intersection is fanned from the vertex average of
    its counterclockwise loop, the fan of ``polygon_intersection_moments``, so the
    signed triangle areas sum to ``area`` term by term.  The default capacity
    ``k1 + k2`` bounds every intersection.
    """

    library = load_meshcore()
    first, first_count = _counted_polygons(first_vertices, first_counts, "first_vertices")
    second, second_count = _counted_polygons(
        second_vertices, second_counts, "second_vertices"
    )
    if first.shape[0] != second.shape[0]:
        raise ValueError("first and second polygon batches must have equal length.")
    capacity = _capacity(
        simplex_capacity, max(1, first.shape[1] + second.shape[1]), "simplex_capacity"
    )
    count = first.shape[0]
    simplices = np.zeros((count, capacity, 3, 2), dtype=np.float64)
    simplex_counts = np.zeros((count,), dtype=np.int32)
    area = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 2), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_polygon_intersection_simplices"](
        count,
        first.shape[1],
        first.ctypes.data,
        first_count.ctypes.data,
        second.shape[1],
        second.ctypes.data,
        second_count.ctypes.data,
        capacity,
        simplices.ctypes.data,
        simplex_counts.ctypes.data,
        area.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "polygon_intersection_simplices")
    return simplices, simplex_counts, area, moment, status


def _tetrahedra(value: object, name: str, /) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 3 or array.shape[1:] != (4, 3):
        raise ValueError(f"{name} must have shape (n, 4, 3).")
    return array


def _capacity(value: int | None, default: int, name: str, /) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1 or value > 2**31 - 1:
        raise ValueError(f"{name} must lie in [1, 2**31 - 1].")
    return int(value)


def tetrahedron_intersection_moments(
    first: object,
    second: object,
    /,
    *,
    vertex_capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly classified intersection moments of tetrahedron pairs ``(n, 4, 3)``.

    Returns ``(volume (n,), first_moment (n, 3), status (n,))``.  The intersection
    has at most eight faces, so the default vertex capacity (48) is never binding
    for valid input.
    """

    library = load_meshcore()
    first_array = _tetrahedra(first, "first")
    second_array = _tetrahedra(second, "second")
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("first and second tetrahedron batches must have equal length.")
    capacity = _capacity(vertex_capacity, 48, "vertex_capacity")
    count = first_array.shape[0]
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_tetrahedron_intersection_moments"](
        count,
        first_array.ctypes.data,
        second_array.ctypes.data,
        capacity,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "tetrahedron_intersection_moments")
    return volume, moment, status


def tetrahedron_intersection_simplices(
    first: object,
    second: object,
    /,
    *,
    vertex_capacity: int | None = None,
    simplex_capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tetrahedron-pair intersections as tetrahedron partitions plus their moments.

    Returns ``(simplices (n, S, 4, 3), simplex_counts (n,), volume, first_moment,
    status)``.  Each nonempty intersection is coned from the vertex average of its
    vertices over the fan triangles of every outward face, the fan of
    ``tetrahedron_intersection_moments``, so the signed simplex volumes sum to
    ``volume`` term by term.  The default capacity (20) bounds every intersection.
    """

    library = load_meshcore()
    first_array = _tetrahedra(first, "first")
    second_array = _tetrahedra(second, "second")
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("first and second tetrahedron batches must have equal length.")
    vertices = _capacity(vertex_capacity, 48, "vertex_capacity")
    capacity = _capacity(simplex_capacity, 20, "simplex_capacity")
    count = first_array.shape[0]
    simplices = np.zeros((count, capacity, 4, 3), dtype=np.float64)
    simplex_counts = np.zeros((count,), dtype=np.int32)
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_tetrahedron_intersection_simplices"](
        count,
        first_array.ctypes.data,
        second_array.ctypes.data,
        vertices,
        capacity,
        simplices.ctypes.data,
        simplex_counts.ctypes.data,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "tetrahedron_intersection_simplices")
    return simplices, simplex_counts, volume, moment, status


def _halfspaces(
    normals: object, offsets: object, plane_counts: object, width: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_array = _real_array(normals, "normals")
    if normal_array.ndim != 3 or normal_array.shape[2] != width:
        raise ValueError(f"normals must have shape (n, m, {width}).")
    offset_array = _real_array(offsets, "offsets")
    if offset_array.shape != normal_array.shape[:2]:
        raise ValueError("offsets must have shape (n, m).")
    count_array = _index_array(plane_counts, "plane_counts")
    if count_array.shape != normal_array.shape[:1]:
        raise ValueError("plane_counts must have shape (n,).")
    if np.any(count_array < 0) or np.any(count_array > normal_array.shape[1]):
        raise ValueError(f"plane_counts must lie in [0, {normal_array.shape[1]}].")
    return normal_array, offset_array, np.ascontiguousarray(count_array, dtype=np.int32)


def polyhedron_clip_moments(
    normals: object,
    offsets: object,
    plane_counts: object,
    tetrahedra: object,
    /,
    *,
    vertex_capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moments of convex polyhedra ``{x : n . x <= h}`` intersected with tetrahedra.

    ``normals`` is ``(n, m, 3)``, ``offsets`` ``(n, m)``, ``plane_counts`` ``(n,)``
    and ``tetrahedra`` ``(n, 4, 3)``.  Returns ``(volume, first_moment, status)``.
    The default vertex capacity ``4 (m + 4)`` bounds every intermediate polytope.
    """

    library = load_meshcore()
    normal_array, offset_array, count_array = _halfspaces(
        normals, offsets, plane_counts, 3
    )
    tetrahedron_array = _tetrahedra(tetrahedra, "tetrahedra")
    if tetrahedron_array.shape[0] != normal_array.shape[0]:
        raise ValueError("halfspace and tetrahedron batches must have equal length.")
    capacity = _capacity(
        vertex_capacity, 4 * (normal_array.shape[1] + 4), "vertex_capacity"
    )
    count = normal_array.shape[0]
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_polyhedron_clip_moments"](
        count,
        normal_array.shape[1],
        normal_array.ctypes.data,
        offset_array.ctypes.data,
        count_array.ctypes.data,
        tetrahedron_array.ctypes.data,
        capacity,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "polyhedron_clip_moments")
    return volume, moment, status


def _box(lower: object, upper: object, width: int, /) -> tuple[np.ndarray, np.ndarray]:
    lower_array = _real_array(lower, "box_lower")
    upper_array = _real_array(upper, "box_upper")
    if lower_array.shape != (width,) or upper_array.shape != (width,):
        raise ValueError(f"box bounds must have shape ({width},).")
    if not np.all(lower_array < upper_array):
        raise ValueError("box_lower must be strictly below box_upper.")
    return lower_array, upper_array


def clip_box_halfplanes(
    box_lower: object,
    box_upper: object,
    normals: object,
    offsets: object,
    plane_counts: object,
    /,
    *,
    vertex_capacity: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Clip one axis-aligned 2D box by per-item halfplane sets ``{x : n . x <= h}``.

    Returns ``(vertices (n, V, 2), edge_labels (n, V), vertex_counts (n,), area (n,),
    first_moment (n, 2), status (n,))``.  Edge ``k`` joins vertices ``k`` and
    ``k + 1`` (cyclic, counterclockwise) and is labeled with the item-local plane
    index or ``-(1 + 2 axis + side)`` for box sides.  The default capacity
    ``m + 4`` is exact for convex polygons.
    """

    library = load_meshcore()
    lower, upper = _box(box_lower, box_upper, 2)
    normal_array, offset_array, count_array = _halfspaces(
        normals, offsets, plane_counts, 2
    )
    capacity = _capacity(vertex_capacity, normal_array.shape[1] + 4, "vertex_capacity")
    count = normal_array.shape[0]
    vertices = np.zeros((count, capacity, 2), dtype=np.float64)
    labels = np.zeros((count, capacity), dtype=np.int32)
    vertex_counts = np.zeros((count,), dtype=np.int32)
    area = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 2), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_clip_box_halfplanes"](
        count,
        lower.ctypes.data,
        upper.ctypes.data,
        normal_array.shape[1],
        normal_array.ctypes.data,
        offset_array.ctypes.data,
        count_array.ctypes.data,
        capacity,
        vertices.ctypes.data,
        labels.ctypes.data,
        vertex_counts.ctypes.data,
        area.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "clip_box_halfplanes")
    return vertices, labels, vertex_counts, area, moment, status


def clip_box_halfspaces(
    box_lower: object,
    box_upper: object,
    normals: object,
    offsets: object,
    plane_counts: object,
    /,
    *,
    vertex_capacity: int | None = None,
    face_capacity: int | None = None,
    face_vertex_capacity: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Clip one axis-aligned 3D box by per-item halfspace sets ``{x : n . x <= h}``.

    Returns ``(vertices (n, V, 3), vertex_counts (n,), face_offsets (n, F + 1),
    face_labels (n, F), face_vertices (n, W), face_counts (n,), volume (n,),
    first_moment (n, 3), status (n,))``.  Faces are sorted by label, oriented
    counterclockwise from outside, and start at their smallest vertex.  Default
    capacities follow the convex-polytope bounds ``F <= m + 6``, ``V <= 2F - 4``
    and ``W <= 2 (V + F - 2)`` (twice the edge count).
    """

    library = load_meshcore()
    lower, upper = _box(box_lower, box_upper, 3)
    normal_array, offset_array, count_array = _halfspaces(
        normals, offsets, plane_counts, 3
    )
    faces_bound = normal_array.shape[1] + 6
    vertices_bound = 2 * faces_bound - 4
    face_capacity_ = _capacity(face_capacity, faces_bound, "face_capacity")
    vertex_capacity_ = _capacity(vertex_capacity, vertices_bound, "vertex_capacity")
    face_vertex_capacity_ = _capacity(
        face_vertex_capacity,
        2 * (vertices_bound + faces_bound - 2),
        "face_vertex_capacity",
    )
    count = normal_array.shape[0]
    vertices = np.zeros((count, vertex_capacity_, 3), dtype=np.float64)
    vertex_counts = np.zeros((count,), dtype=np.int32)
    face_offsets = np.zeros((count, face_capacity_ + 1), dtype=np.int32)
    face_labels = np.zeros((count, face_capacity_), dtype=np.int32)
    face_vertices = np.zeros((count, face_vertex_capacity_), dtype=np.int32)
    face_counts = np.zeros((count,), dtype=np.int32)
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_clip_box_halfspaces"](
        count,
        lower.ctypes.data,
        upper.ctypes.data,
        normal_array.shape[1],
        normal_array.ctypes.data,
        offset_array.ctypes.data,
        count_array.ctypes.data,
        vertex_capacity_,
        face_capacity_,
        face_vertex_capacity_,
        vertices.ctypes.data,
        vertex_counts.ctypes.data,
        face_offsets.ctypes.data,
        face_labels.ctypes.data,
        face_vertices.ctypes.data,
        face_counts.ctypes.data,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "clip_box_halfspaces")
    return (
        vertices,
        vertex_counts,
        face_offsets,
        face_labels,
        face_vertices,
        face_counts,
        volume,
        moment,
        status,
    )


# ---------------------------------------------------------------- triangulations


def _read_mesh(
    library: MeshcoreLibrary, handle: ctypes.c_void_p, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # The native handle is released even when a copy fails.
    try:
        dimension = library["phx_mc_mesh_dimension"](handle)
        point_count = library["phx_mc_mesh_point_count"](handle)
        cell_count = library["phx_mc_mesh_cell_count"](handle)
        input_count = library["phx_mc_mesh_input_point_count"](handle)
        points = np.zeros((point_count, dimension), dtype=np.float64)
        cells = np.zeros((cell_count, dimension + 1), dtype=np.int32)
        vertex_map = np.zeros((input_count,), dtype=np.int32)
        cell_segments = np.zeros((cell_count, 3), dtype=np.int32)
        library["phx_mc_mesh_copy_points"](handle, points.ctypes.data)
        library["phx_mc_mesh_copy_cells"](handle, cells.ctypes.data)
        library["phx_mc_mesh_copy_vertex_map"](handle, vertex_map.ctypes.data)
        library["phx_mc_mesh_copy_cell_segments"](handle, cell_segments.ctypes.data)
    finally:
        library["phx_mc_mesh_free"](handle)
    return points, cells, vertex_map, cell_segments


def _limit(value: int | None, default: int, name: str, /) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _triangulate(
    name: str,
    points: object,
    weights: object | None,
    width: int,
    max_cells: int | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    library = load_meshcore()
    point_array = _points(points, "points", width)
    count = point_array.shape[0]
    if count > _MAX_POINTS:
        raise ValueError("meshcore triangulations support fewer than 2**31 - 1 points.")
    # Worst-case simplex counts: 2n - 5 triangles; (n^2 - 3n - 2) / 2 tetrahedra.
    default = (
        max(1, 2 * count) if width == 2 else max(1, (count * count - 3 * count) // 2)
    )
    limit = _limit(max_cells, default, "max_cells")
    handle = ctypes.c_void_p()
    if weights is None:
        status = library[name](
            count, point_array.ctypes.data, limit, ctypes.addressof(handle)
        )
    else:
        weight_array = _real_array(weights, "weights")
        if weight_array.shape != (count,):
            raise ValueError("weights must have shape (n,).")
        status = library[name](
            count,
            point_array.ctypes.data,
            weight_array.ctypes.data,
            limit,
            ctypes.addressof(handle),
        )
    operation = name.removeprefix("phx_mc_")
    if status != MeshcoreStatus.OK:
        _raise_for_call(status, operation)
    if not handle.value:
        raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, f"{operation}: missing mesh")
    _, cells, vertex_map, _ = _read_mesh(library, handle)
    return cells, vertex_map


def delaunay_2d(
    points: object, /, *, max_triangles: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Delaunay triangulation of ``(n, 2)`` points.

    Returns ``(triangles (T, 3) int32, vertex_map (n,) int32)``; triangles are
    counterclockwise, canonically ordered, and cocircular ties are resolved by
    index-ordered symbolic perturbation.  ``vertex_map`` sends duplicate points
    to their smallest identical index.
    """

    return _triangulate("phx_mc_delaunay_2d", points, None, 2, max_triangles)


def delaunay_3d(
    points: object, /, *, max_tetrahedra: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Delaunay tetrahedralization of ``(n, 3)`` points.

    Returns ``(tetrahedra (T, 4) int32, vertex_map (n,) int32)`` with positively
    oriented, canonically ordered tetrahedra.
    """

    return _triangulate("phx_mc_delaunay_3d", points, None, 3, max_tetrahedra)


def regular_2d(
    points: object, weights: object, /, *, max_triangles: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact regular (weighted Delaunay) triangulation; redundant points map to -1."""

    return _triangulate("phx_mc_regular_2d", points, weights, 2, max_triangles)


def regular_3d(
    points: object, weights: object, /, *, max_tetrahedra: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact regular (weighted Delaunay) tetrahedralization; redundant points map to -1."""

    return _triangulate("phx_mc_regular_3d", points, weights, 3, max_tetrahedra)


def constrained_delaunay_2d(
    points: object,
    segments: object,
    /,
    *,
    holes: object | None = None,
    keep_convex_hull: bool = False,
    min_angle: float = 0.0,
    max_area: float = float("inf"),
    max_steiner: int = 0,
    max_triangles: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MeshcoreStatus]:
    """Constrained Delaunay triangulation with Ruppert/Chew refinement.

    ``segments`` are ``(m, 2)`` point indices.  Without ``keep_convex_hull`` the
    triangles outside the segment-bounded region and inside hole seeds are
    removed.  ``min_angle`` (degrees, ``[0, 60)``), ``max_area`` and
    ``max_steiner`` bound the refinement.  Returns ``(points (N, 2), triangles
    (T, 3), segment_ids (T, 3), status)`` where input points come first,
    ``segment_ids[t, k]`` is the input segment carrying the edge opposite vertex
    ``k`` (or -1), and ``status`` is ``OK`` or ``REFINEMENT_LIMIT`` (valid mesh,
    quality targets unmet within ``max_steiner``).
    """

    library = load_meshcore()
    point_array = _points(points, "points", 2)
    segment_array = _index_array(segments, "segments")
    if segment_array.ndim != 2 or segment_array.shape[1] != 2:
        raise ValueError("segments must have shape (m, 2).")
    if np.any(segment_array < 0) or np.any(segment_array >= point_array.shape[0]):
        raise ValueError("segments reference a missing point.")
    segment_int = np.ascontiguousarray(segment_array, dtype=np.int32)
    hole_array = (
        np.zeros((0, 2), dtype=np.float64)
        if holes is None
        else _points(holes, "holes", 2)
    )
    if not isinstance(keep_convex_hull, bool):
        raise TypeError("keep_convex_hull must be a bool.")
    angle = float(min_angle)
    area = float(max_area)
    if not (0.0 <= angle < 60.0):
        raise ValueError("min_angle must lie in [0, 60) degrees.")
    if not area > 0.0:
        raise ValueError("max_area must be positive (inf disables the bound).")
    if isinstance(max_steiner, bool) or not isinstance(max_steiner, (int, np.integer)):
        raise TypeError("max_steiner must be an integer.")
    if max_steiner < 0:
        raise ValueError("max_steiner must be nonnegative.")
    total = point_array.shape[0] + int(max_steiner)
    if total > _MAX_POINTS:
        raise ValueError("meshcore triangulations support fewer than 2**31 - 1 points.")
    limit = _limit(max_triangles, max(1, 2 * total), "max_triangles")
    handle = ctypes.c_void_p()
    status = library["phx_mc_constrained_delaunay_2d"](
        point_array.shape[0],
        point_array.ctypes.data,
        segment_int.shape[0],
        segment_int.ctypes.data,
        hole_array.shape[0],
        hole_array.ctypes.data,
        1 if keep_convex_hull else 0,
        angle,
        area,
        int(max_steiner),
        limit,
        ctypes.addressof(handle),
    )
    code = MeshcoreStatus(status)
    if code not in (MeshcoreStatus.OK, MeshcoreStatus.REFINEMENT_LIMIT):
        _raise_for_call(status, "constrained_delaunay_2d")
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "constrained_delaunay_2d: missing mesh"
        )
    mesh_points, cells, _, cell_segments = _read_mesh(library, handle)
    return mesh_points, cells, cell_segments, code


__all__ = [
    "MeshcoreError",
    "MeshcoreLibrary",
    "MeshcoreStatus",
    "MeshcoreUnavailableError",
    "clip_box_halfplanes",
    "clip_box_halfspaces",
    "constrained_delaunay_2d",
    "delaunay_2d",
    "delaunay_3d",
    "exact_incircle",
    "exact_insphere",
    "exact_orient2d",
    "exact_orient3d",
    "load_meshcore",
    "meshcore_available",
    "meshcore_exact_domain",
    "meshcore_identity",
    "polygon_intersection_moments",
    "polygon_intersection_simplices",
    "polyhedron_clip_moments",
    "regular_2d",
    "regular_3d",
    "symbolic_incircle",
    "symbolic_insphere",
    "symbolic_orient2d",
    "symbolic_orient3d",
    "tetrahedron_intersection_moments",
    "tetrahedron_intersection_simplices",
]
