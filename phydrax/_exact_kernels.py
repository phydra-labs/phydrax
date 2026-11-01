#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native exact and filtered kernels of Bernstein and box-pair proofs.

The single Python surface of the meshcore ``exact kernels`` C ABI section.
Interval kernels reproduce their host reference bit for bit; exact kernels
work on integer numerators over one shared denominator in checked 256-bit
words. A value outside that representation is a typed refusal (``None`` for
the item): the caller recomputes that item on its own exact path, so the
native layer never caps, clamps, or changes a certificate. Work is split into
fixed chunks by item index, so every ``parallelism`` gives identical results.
"""

from __future__ import annotations

import ctypes
import os
from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from math import comb, lcm, prod
from typing import Literal

import numpy as np

from ._meshcore import _raise_for_call, load_meshcore, MeshcoreStatus


type BernsteinDomain = Literal["simplex", "prism", "box"]
type HostTree = tuple[np.ndarray, ...]
type _IntegerRows = tuple[tuple[int, ...], ...]

_WORD_BYTES = 32
_DOMAIN_CODES = {"simplex": 0, "box": 1, "prism": 2}
_RANGE_ERROR = int(MeshcoreStatus.RANGE_ERROR)
_OK = int(MeshcoreStatus.OK)
_PARALLELISM_ENVIRONMENT = "PHX_MESHCORE_THREADS"


def exact_kernel_parallelism() -> int:
    """Worker count of the native kernels: ``PHX_MESHCORE_THREADS``, else one."""
    configured = os.environ.get(_PARALLELISM_ENVIRONMENT, "").strip()
    if not configured:
        return 1
    try:
        value = int(configured)
    except ValueError as error:
        raise ValueError(
            f"{_PARALLELISM_ENVIRONMENT} must be a positive integer, got {configured!r}."
        ) from error
    if value < 1:
        raise ValueError(
            f"{_PARALLELISM_ENVIRONMENT} must be a positive integer, got {configured!r}."
        )
    return value


def _workers(parallelism: int | None, /) -> int:
    if parallelism is None:
        return exact_kernel_parallelism()
    if (
        isinstance(parallelism, bool)
        or not isinstance(parallelism, int)
        or parallelism < 1
    ):
        raise ValueError("parallelism must be a positive integer.")
    return parallelism


def _words(values: Sequence[int], /) -> bytes | None:
    """256-bit little-endian two's-complement words, or ``None`` when out of range."""
    try:
        return b"".join(
            value.to_bytes(_WORD_BYTES, "little", signed=True) for value in values
        )
    except OverflowError:
        return None


def _word(buffer: bytes, index: int, /) -> int:
    start = index * _WORD_BYTES
    return int.from_bytes(buffer[start : start + _WORD_BYTES], "little", signed=True)


def _pointer(array: np.ndarray, /) -> int:
    return array.ctypes.data


# ------------------------------------------------------ interval restriction


def interval_bernstein_restrict(
    lower: np.ndarray,
    upper: np.ndarray,
    parameters: np.ndarray,
    sides: np.ndarray,
    /,
    *,
    parallelism: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict interval Bernstein nets along axis 1 of ``(count, degree + 1, ...)``.

    ``parameters`` is ``(count, 4)``: the enclosure of ``last`` and of
    ``first / last``; ``sides`` bit 0 applies the split at ``last`` (``last <
    1``) and bit 1 the split at the ratio (``first > 0``). Bit-identical to the
    host rounding-bound de Casteljau reference.
    """
    source_lower = np.ascontiguousarray(lower, dtype=np.float64)
    source_upper = np.ascontiguousarray(upper, dtype=np.float64)
    if source_lower.shape != source_upper.shape or source_lower.ndim < 2:
        raise ValueError("interval nets must share one (count, degree + 1, ...) shape.")
    count, rows = source_lower.shape[:2]
    width = int(prod(source_lower.shape[2:]))
    if rows < 1 or width < 1:
        raise ValueError("interval nets need at least one row and one component.")
    parameter_array = np.ascontiguousarray(parameters, dtype=np.float64)
    side_array = np.ascontiguousarray(sides, dtype=np.int32)
    if parameter_array.shape != (count, 4) or side_array.shape != (count,):
        raise ValueError("parameters must be (count, 4) and sides (count,).")
    out_lower = np.empty_like(source_lower)
    out_upper = np.empty_like(source_upper)
    status = load_meshcore()["phx_mc_interval_bernstein_restrict"](
        count,
        rows - 1,
        width,
        _pointer(source_lower),
        _pointer(source_upper),
        _pointer(parameter_array),
        _pointer(side_array),
        _pointer(out_lower),
        _pointer(out_upper),
        _workers(parallelism),
    )
    _raise_for_call(status, "interval_bernstein_restrict")
    return out_lower, out_upper


# --------------------------------------------------------- exact Bernstein


def _control_count(
    domain: BernsteinDomain, dimension: int, degree: Sequence[int], /
) -> int:
    match domain:
        case "simplex":
            return comb(int(degree[0]) + dimension, dimension)
        case "box":
            return prod(int(value) + 1 for value in degree)
        case "prism":
            return comb(int(degree[0]) + 2, 2) * (int(degree[1]) + 1)
    raise ValueError(f"Unsupported Bernstein domain {domain!r}.")


@dataclass(frozen=True, slots=True)
class IntegerBernsteinNet:
    """Exact Bernstein controls ``numerator(i) / scale`` of one polynomial.

    ``least`` and ``greatest`` index a least and a greatest control (the first
    in control order). Numerators decode lazily from their native words.
    """

    words: bytes
    count: int
    scale: int
    least: int
    greatest: int

    def numerator(self, index: int, /) -> int:
        if not 0 <= index < self.count:
            raise IndexError("control index out of range.")
        return _word(self.words, index)

    def numerators(self) -> tuple[int, ...]:
        return tuple(_word(self.words, index) for index in range(self.count))

    def control(self, index: int, /) -> Fraction:
        return Fraction(self.numerator(index), self.scale)

    def controls(self) -> tuple[Fraction, ...]:
        scale = self.scale
        return tuple(Fraction(value, scale) for value in self.numerators())


def rational_bernstein_nets(
    polynomials: Sequence[Mapping[tuple[int, ...], object]],
    domain: BernsteinDomain,
    dimension: int,
    degrees: Sequence[Sequence[int]],
    /,
    *,
    parallelism: int | None = None,
) -> list[IntegerBernsteinNet | None]:
    """Exact Bernstein controls of rational polynomials, in `bernstein_coefficients` order.

    Every ``degrees[i]`` must be at least the least Bernstein degree of
    ``polynomials[i]``. An item is ``None`` (typed refusal) when a coefficient
    is not rational or a value leaves the native exact representation; the
    caller then evaluates it on its Python exact path.
    """
    code = _DOMAIN_CODES.get(domain)
    if code is None:
        raise ValueError(f"Unsupported Bernstein domain {domain!r}.")
    if len(polynomials) != len(degrees):
        raise ValueError("every polynomial needs one Bernstein degree.")
    width = 1 if domain == "simplex" else (2 if domain == "prism" else dimension)
    results: list[IntegerBernsteinNet | None] = [None] * len(polynomials)
    rows: list[int] = []
    scales: list[int] = []
    degree_values: list[int] = []
    exponents: list[int] = []
    numerators: list[int] = []
    term_offsets = [0]
    control_offsets = [0]
    for row, (polynomial, degree) in enumerate(zip(polynomials, degrees, strict=True)):
        if len(degree) != width:
            raise ValueError("Bernstein degree width does not match the domain.")
        coefficients = [
            value
            for value in polynomial.values()
            if isinstance(value, (Fraction, int)) and not isinstance(value, bool)
        ]
        if len(coefficients) != len(polynomial):
            continue
        scale = lcm(*(value.denominator for value in coefficients)) if coefficients else 1
        item_numerators = [
            value.numerator * (scale // value.denominator) for value in coefficients
        ]
        if any(value.bit_length() >= 8 * _WORD_BYTES - 1 for value in item_numerators):
            continue
        for index in polynomial:
            if len(index) != dimension:
                raise ValueError(
                    "Bernstein dimension must match polynomial monomial dimensions."
                )
            exponents.extend(index)
        numerators.extend(item_numerators)
        rows.append(row)
        scales.append(scale)
        degree_values.extend(int(value) for value in degree)
        term_offsets.append(term_offsets[-1] + len(coefficients))
        control_offsets.append(
            control_offsets[-1] + _control_count(domain, dimension, degree)
        )
    if not rows:
        return results
    numerator_words = _words(numerators)
    if numerator_words is None:
        raise AssertionError("checked numerators must fit their native words.")
    count = len(rows)
    # Small calls dominate: bytes inputs and ctypes buffers avoid per-call
    # NumPy array construction.
    controls = ctypes.create_string_buffer(control_offsets[-1] * _WORD_BYTES)
    denominators = ctypes.create_string_buffer(8 * count)
    extremes = ctypes.create_string_buffer(16 * count)
    status = ctypes.create_string_buffer(4 * count)
    call = load_meshcore()["phx_mc_bernstein_controls"](
        count,
        code,
        dimension,
        array("i", degree_values).tobytes(),
        array("q", term_offsets).tobytes(),
        array("i", exponents).tobytes(),
        numerator_words,
        array("q", control_offsets).tobytes(),
        controls,
        denominators,
        extremes,
        status,
        _workers(parallelism),
    )
    _raise_for_call(call, "bernstein_controls")
    payload = controls.raw
    denominator_values = array("q", denominators.raw)
    extreme_values = array("q", extremes.raw)
    status_values = array("i", status.raw)
    for position, row in enumerate(rows):
        item_status = status_values[position]
        if item_status == _RANGE_ERROR:
            continue
        _raise_for_call(item_status, "bernstein_controls")
        start = control_offsets[position] * _WORD_BYTES
        stop = control_offsets[position + 1] * _WORD_BYTES
        results[row] = IntegerBernsteinNet(
            payload[start:stop],
            control_offsets[position + 1] - control_offsets[position],
            scales[position] * denominator_values[position],
            extreme_values[2 * position],
            extreme_values[2 * position + 1],
        )
    return results


def simplex_blossom_extremes(
    nets: Sequence[Sequence[int]],
    corners: Sequence[Sequence[Sequence[int]]],
    degrees: Sequence[int],
    parts: int,
    /,
    *,
    parallelism: int | None = None,
) -> list[tuple[int, int] | None]:
    """Least and greatest restricted controls of integer simplex Bernstein nets.

    ``nets[i]`` lists the integer controls of degree ``degrees[i]`` in
    composition order (first barycentric exponent descending); ``corners[i][j]``
    lists the integer barycentric weights of subsimplex corner ``j`` over one
    common positive scale ``s``. Each returned value is a restricted control
    times ``s ** degree``; ``None`` is the typed refusal of an item whose exact
    blossoms leave the native representation.
    """
    count = len(nets)
    if len(corners) != count or len(degrees) != count:
        raise ValueError("every net needs its corners and degree.")
    offsets = [0]
    flat: list[int] = []
    corner_values: list[int] = []
    for net, corner_rows, degree in zip(nets, corners, degrees, strict=True):
        if len(net) != comb(int(degree) + parts - 1, parts - 1):
            raise ValueError("a simplex net must hold every control of its degree.")
        if len(corner_rows) != parts or any(len(row) != parts for row in corner_rows):
            raise ValueError("corners must be (parts, parts) integer weights.")
        flat.extend(net)
        corner_values.extend(value for row in corner_rows for value in row)
        offsets.append(offsets[-1] + len(net))
    if not count:
        return []
    net_words = _words(flat)
    corner_words = _words(corner_values)
    if net_words is None or corner_words is None:
        return [None] * count
    values = np.empty((offsets[-1] * _WORD_BYTES,), dtype=np.uint8)
    extremes = np.empty((count, 2), dtype=np.int64)
    status = np.empty((count,), dtype=np.int32)
    degree_array = np.asarray(degrees, dtype=np.int32)
    offset_array = np.asarray(offsets, dtype=np.int64)
    call = load_meshcore()["phx_mc_simplex_bernstein_blossoms"](
        count,
        parts,
        _pointer(degree_array),
        _pointer(offset_array),
        net_words,
        corner_words,
        _pointer(values),
        _pointer(extremes),
        _pointer(status),
        _workers(parallelism),
    )
    _raise_for_call(call, "simplex_bernstein_blossoms")
    payload = values.tobytes()
    results: list[tuple[int, int] | None] = []
    for item in range(count):
        item_status = int(status[item])
        if item_status == _RANGE_ERROR:
            results.append(None)
            continue
        _raise_for_call(item_status, "simplex_bernstein_blossoms")
        results.append(
            (
                _word(payload, offsets[item] + int(extremes[item, 0])),
                _word(payload, offsets[item] + int(extremes[item, 1])),
            )
        )
    return results


@dataclass(frozen=True, slots=True)
class BisectedNets:
    """Halves of integer Bernstein nets over ``2 ** exponent`` times the input scale.

    ``halves[i]`` is ``(left, right)`` as ``(controls, components)`` integer rows,
    or ``None`` (typed refusal) when item ``i`` left the native representation.
    """

    exponent: int
    halves: tuple[tuple[_IntegerRows, _IntegerRows] | None, ...]


def bisect_integer_nets(
    nets: Sequence[Sequence[Sequence[int]]],
    indices: Sequence[Sequence[int]],
    first: int,
    second: int,
    /,
    *,
    parallelism: int | None = None,
) -> BisectedNets:
    """Exact de Casteljau halves of integer multi-index nets along slots ``(first, second)``.

    The left half keeps slot ``first`` (its vertex) and moves ``second`` to the
    midpoint; the right half keeps ``second``. Each line of controls sharing all
    other slots must hold every split of its length.
    """
    count = len(nets)
    control_count = len(indices)
    if not control_count:
        raise ValueError("bisection needs at least one control.")
    slot_count = len(indices[0])
    components = len(nets[0][0]) if count else 1
    flat: list[int] = []
    for net in nets:
        if len(net) != control_count or any(len(row) != components for row in net):
            raise ValueError("nets must be (controls, components) integer rows.")
        for row in net:
            flat.extend(row)
    index_array = np.asarray(indices, dtype=np.int32)
    if index_array.shape != (control_count, slot_count):
        raise ValueError("indices must share one slot count.")
    words = _words(flat) if count else b""
    if words is None:
        exponent = max(int(row[first]) + int(row[second]) for row in indices)
        return BisectedNets(exponent, (None,) * count)
    size = count * control_count * components * _WORD_BYTES
    left = np.empty((max(size, 1),), dtype=np.uint8)
    right = np.empty((max(size, 1),), dtype=np.uint8)
    exponent_out = np.zeros((1,), dtype=np.int32)
    status = np.empty((max(count, 1),), dtype=np.int32)
    call = load_meshcore()["phx_mc_bernstein_bisect"](
        count,
        control_count,
        slot_count,
        components,
        _pointer(index_array),
        first,
        second,
        words if count else None,
        _pointer(left),
        _pointer(right),
        _pointer(exponent_out),
        _pointer(status),
        _workers(parallelism),
    )
    _raise_for_call(call, "bernstein_bisect")
    left_bytes = left.tobytes()
    right_bytes = right.tobytes()
    halves: list[tuple[_IntegerRows, _IntegerRows] | None] = []
    for item in range(count):
        item_status = int(status[item])
        if item_status == _RANGE_ERROR:
            halves.append(None)
            continue
        _raise_for_call(item_status, "bernstein_bisect")
        base = item * control_count * components
        left_rows, right_rows = (
            tuple(
                tuple(
                    _word(payload, base + row * components + component)
                    for component in range(components)
                )
                for row in range(control_count)
            )
            for payload in (left_bytes, right_bytes)
        )
        halves.append((left_rows, right_rows))
    return BisectedNets(int(exponent_out[0]), tuple(halves))


# ---------------------------------------------------------- separation screen


class SeparationDecision(IntEnum):
    """Filtered outcome of one plane separation screen."""

    UNDECIDED = 0
    SEPARATED = 1
    NOT_SEPARATED = 2


def separation_screen(
    first_lower: np.ndarray,
    first_upper: np.ndarray,
    first_offsets: np.ndarray,
    second_lower: np.ndarray,
    second_upper: np.ndarray,
    second_offsets: np.ndarray,
    normals: np.ndarray,
    /,
    *,
    parallelism: int | None = None,
) -> np.ndarray:
    """Decide ``max n.b < min n.a`` per pair from interval point boxes.

    Pair ``i`` owns first points ``first_offsets[i]:first_offsets[i + 1]`` and
    second points likewise; boxes are ``(points, dimension)``. Returns int8
    `SeparationDecision` codes; both decided codes equal the exact outcome and
    ``UNDECIDED`` pairs need the caller's exact check.
    """
    normal_array = np.ascontiguousarray(normals, dtype=np.float64)
    if normal_array.ndim != 2:
        raise ValueError("normals must be (pairs, dimension).")
    count, dimension = normal_array.shape
    arrays = [
        np.ascontiguousarray(value, dtype=np.float64).reshape((-1, dimension))
        for value in (first_lower, first_upper, second_lower, second_upper)
    ]
    first_index = np.ascontiguousarray(first_offsets, dtype=np.int64)
    second_index = np.ascontiguousarray(second_offsets, dtype=np.int64)
    if (
        first_index.shape != (count + 1,)
        or second_index.shape != (count + 1,)
        or arrays[0].shape != arrays[1].shape
        or arrays[2].shape != arrays[3].shape
        or (count and (first_index[-1] != arrays[0].shape[0]))
        or (count and (second_index[-1] != arrays[2].shape[0]))
    ):
        raise ValueError("separation screen offsets must cover their point boxes.")
    decisions = np.empty((count,), dtype=np.int8)
    status = load_meshcore()["phx_mc_separation_screen"](
        count,
        dimension,
        _pointer(first_index),
        _pointer(arrays[0]),
        _pointer(arrays[1]),
        _pointer(second_index),
        _pointer(arrays[2]),
        _pointer(arrays[3]),
        _pointer(normal_array),
        _pointer(decisions),
        _workers(parallelism),
    )
    _raise_for_call(status, "separation_screen")
    return decisions


# ------------------------------------------------------------- BVH pair stream


class _TreeView(ctypes.Structure):
    _fields_ = (
        ("node_count", ctypes.c_int64),
        ("leaf_count", ctypes.c_int64),
        ("leaf_width", ctypes.c_int64),
        ("item_count", ctypes.c_int64),
        ("node_min", ctypes.c_void_p),
        ("node_max", ctypes.c_void_p),
        ("node_size", ctypes.c_void_p),
        ("left", ctypes.c_void_p),
        ("right", ctypes.c_void_p),
        ("leaf", ctypes.c_void_p),
        ("leaf_items", ctypes.c_void_p),
        ("item_min", ctypes.c_void_p),
        ("item_max", ctypes.c_void_p),
    )


def _tree_view(tree: HostTree, /) -> tuple[_TreeView, tuple[np.ndarray, ...]]:
    node_min, node_max, left, right, leaf, leaf_items, item_min, item_max = (
        np.ascontiguousarray(value) for value in tree
    )
    # The descent key is the reference's own float64 extent sum.
    node_size = np.ascontiguousarray(np.sum(node_max - node_min, axis=-1))
    arrays = (
        node_min,
        node_max,
        node_size,
        left,
        right,
        leaf,
        leaf_items,
        item_min,
        item_max,
    )
    view = _TreeView(
        int(node_min.shape[0]),
        int(leaf_items.shape[0]),
        int(leaf_items.shape[1]),
        int(item_min.shape[0]),
        *(_pointer(array) for array in arrays),
    )
    return view, arrays


def bvh_pair_blocks(
    first: HostTree,
    second: HostTree,
    /,
    *,
    include_touching: bool,
    absolute_tolerance: float,
    relative_tolerance: float,
    item_block: int,
    node_block: int,
    parallelism: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Overlapping item pairs of two host trees in the reference's block sequence.

    Trees are ``(node_min, node_max, left, right, leaf, leaf_items, item_min,
    item_max)`` float64/int64 host arrays. The yielded blocks equal the host
    LIFO traversal's blocks one for one; the native stream exists only while
    the generator runs and is released when it finishes or is closed.
    """
    library = load_meshcore()
    # The views and their arrays stay referenced by this frame for the
    # stream's lifetime.
    first_view, first_arrays = _tree_view(first)
    second_view, second_arrays = _tree_view(second)
    leaf_pairs = int(first_view.leaf_width) * int(second_view.leaf_width)
    capacity = max(1, int(item_block) // leaf_pairs) * leaf_pairs
    block_capacity = 64
    first_items = np.empty((capacity,), dtype=np.int64)
    second_items = np.empty((capacity,), dtype=np.int64)
    ends = np.empty((block_capacity,), dtype=np.int64)
    blocks = np.zeros((1,), dtype=np.int64)
    stream = ctypes.c_void_p()
    status = library["phx_mc_bvh_pair_stream_create"](
        ctypes.addressof(first_view),
        ctypes.addressof(second_view),
        int(first_arrays[0].shape[1]),
        int(include_touching),
        float(absolute_tolerance),
        float(relative_tolerance),
        int(item_block),
        int(node_block),
        _workers(parallelism),
        ctypes.addressof(stream),
    )
    _raise_for_call(status, "bvh_pair_stream_create")
    try:
        while True:
            status = library["phx_mc_bvh_pair_stream_next"](
                stream,
                capacity,
                block_capacity,
                _pointer(first_items),
                _pointer(second_items),
                _pointer(ends),
                _pointer(blocks),
            )
            _raise_for_call(status, "bvh_pair_stream_next")
            count = int(blocks[0])
            if count == 0:
                return
            start = 0
            for stop in ends[:count].tolist():
                yield first_items[start:stop].copy(), second_items[start:stop].copy()
                start = stop
    finally:
        library["phx_mc_bvh_pair_stream_free"](stream)
        del first_arrays, second_arrays


__all__ = [
    "BisectedNets",
    "IntegerBernsteinNet",
    "SeparationDecision",
    "bisect_integer_nets",
    "bvh_pair_blocks",
    "exact_kernel_parallelism",
    "interval_bernstein_restrict",
    "rational_bernstein_nets",
    "separation_screen",
    "simplex_blossom_extremes",
]
