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
from typing import cast, Literal

import numpy as np

from ._meshcore import _raise_for_call, load_meshcore, MeshcoreStatus


type BernsteinDomain = Literal["simplex", "prism", "box"]
type HostTree = tuple[np.ndarray, ...]
type _IntegerRows = tuple[tuple[int, ...], ...]

# Native exact word widths in 64-bit limbs (256 to 4096 bits).
_WORD_LIMBS = (4, 8, 16, 32, 64)
_DOMAIN_CODES = {"simplex": 0, "box": 1, "prism": 2}
_RANGE_ERROR = int(MeshcoreStatus.RANGE_ERROR)
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


def _limbs_for(bits: int, /) -> int | None:
    """Least native word width (in 64-bit limbs) holding ``bits``-bit magnitudes."""
    for limbs in _WORD_LIMBS:
        if bits < 64 * limbs - 1:
            return limbs
    return None


def _words(values: Sequence[int], limbs: int, /) -> bytes:
    """Little-endian two's-complement native words of ``limbs`` limbs."""
    width = 8 * limbs
    return b"".join(value.to_bytes(width, "little", signed=True) for value in values)


def _word(buffer: bytes, index: int, width: int, /) -> int:
    start = index * width
    return int.from_bytes(buffer[start : start + width], "little", signed=True)


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
    in control order). Numerators decode lazily from their native words of
    ``word_bytes`` bytes.
    """

    words: bytes
    word_bytes: int
    count: int
    scale: int
    least: int
    greatest: int

    def numerator(self, index: int, /) -> int:
        if not 0 <= index < self.count:
            raise IndexError("control index out of range.")
        return _word(self.words, index, self.word_bytes)

    def numerators(self) -> tuple[int, ...]:
        return tuple(
            _word(self.words, index, self.word_bytes) for index in range(self.count)
        )

    def control(self, index: int, /) -> Fraction:
        return Fraction(self.numerator(index), self.scale)

    def controls(self) -> tuple[Fraction, ...]:
        scale = self.scale
        return tuple(Fraction(value, scale) for value in self.numerators())


@dataclass(frozen=True, slots=True)
class _RationalItem:
    row: int
    scale: int
    degree: tuple[int, ...]
    exponents: tuple[int, ...]
    numerators: tuple[int, ...]
    controls: int
    limbs: int


def _rational_item(
    row: int,
    polynomial: Mapping[tuple[int, ...], object],
    domain: BernsteinDomain,
    dimension: int,
    degree: Sequence[int],
    /,
) -> _RationalItem | None:
    coefficients = [
        value
        for value in polynomial.values()
        if isinstance(value, (Fraction, int)) and not isinstance(value, bool)
    ]
    if len(coefficients) != len(polynomial):
        return None
    scale = lcm(*(value.denominator for value in coefficients)) if coefficients else 1
    numerators = tuple(
        value.numerator * (scale // value.denominator) for value in coefficients
    )
    exponents: list[int] = []
    for index in polynomial:
        if len(index) != dimension:
            raise ValueError(
                "Bernstein dimension must match polynomial monomial dimensions."
            )
        exponents.extend(index)
    # Integer weights stay below 2**64; the sum adds the term-count bits.
    bits = max((value.bit_length() for value in numerators), default=0)
    limbs = _limbs_for(bits + 64 + len(numerators).bit_length())
    if limbs is None:
        return None
    return _RationalItem(
        row,
        scale,
        tuple(int(value) for value in degree),
        tuple(exponents),
        numerators,
        _control_count(domain, dimension, degree),
        limbs,
    )


def _rational_call(
    items: Sequence[_RationalItem], code: int, dimension: int, limbs: int, workers: int, /
) -> list[IntegerBernsteinNet | None]:
    width = 8 * limbs
    term_offsets = [0]
    control_offsets = [0]
    for item in items:
        term_offsets.append(term_offsets[-1] + len(item.numerators))
        control_offsets.append(control_offsets[-1] + item.controls)
    count = len(items)
    # Small calls dominate: bytes inputs and ctypes buffers avoid per-call
    # NumPy array construction.
    controls = ctypes.create_string_buffer(control_offsets[-1] * width)
    denominators = ctypes.create_string_buffer(8 * count)
    extremes = ctypes.create_string_buffer(16 * count)
    status = ctypes.create_string_buffer(4 * count)
    call = load_meshcore()["phx_mc_bernstein_controls"](
        count,
        code,
        dimension,
        array("i", [value for item in items for value in item.degree]).tobytes(),
        array("q", term_offsets).tobytes(),
        array("i", [value for item in items for value in item.exponents]).tobytes(),
        _words([value for item in items for value in item.numerators], limbs),
        array("q", control_offsets).tobytes(),
        controls,
        denominators,
        extremes,
        status,
        limbs,
        workers,
    )
    _raise_for_call(call, "bernstein_controls")
    payload = controls.raw
    denominator_values = array("q", denominators.raw)
    extreme_values = array("q", extremes.raw)
    status_values = array("i", status.raw)
    results: list[IntegerBernsteinNet | None] = []
    for position, item in enumerate(items):
        item_status = status_values[position]
        if item_status == _RANGE_ERROR:
            results.append(None)
            continue
        _raise_for_call(item_status, "bernstein_controls")
        results.append(
            IntegerBernsteinNet(
                payload[
                    control_offsets[position] * width : control_offsets[position + 1]
                    * width
                ],
                width,
                item.controls,
                item.scale * denominator_values[position],
                extreme_values[2 * position],
                extreme_values[2 * position + 1],
            )
        )
    return results


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
    ``polynomials[i]``. Each item runs at the least word width its numerators
    allow and is retried wider on a range refusal. An item is ``None`` (typed
    refusal) when a coefficient is not rational or its exact controls exceed
    the widest native word; the caller then evaluates it on its Python exact
    path.
    """
    code = _DOMAIN_CODES.get(domain)
    if code is None:
        raise ValueError(f"Unsupported Bernstein domain {domain!r}.")
    if len(polynomials) != len(degrees):
        raise ValueError("every polynomial needs one Bernstein degree.")
    width = 1 if domain == "simplex" else (2 if domain == "prism" else dimension)
    results: list[IntegerBernsteinNet | None] = [None] * len(polynomials)
    pending: list[_RationalItem] = []
    for row, (polynomial, degree) in enumerate(zip(polynomials, degrees, strict=True)):
        if len(degree) != width:
            raise ValueError("Bernstein degree width does not match the domain.")
        item = _rational_item(row, polynomial, domain, dimension, degree)
        if item is not None:
            pending.append(item)
    workers = _workers(parallelism)
    for limbs in _WORD_LIMBS:
        batch = [item for item in pending if item.limbs <= limbs]
        if not batch:
            continue
        refused = []
        for item, net in zip(
            batch, _rational_call(batch, code, dimension, limbs, workers), strict=True
        ):
            if net is None:
                refused.append(item)
            else:
                results[item.row] = net
        pending = [item for item in pending if item.limbs > limbs] + refused
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
    blossoms exceed the widest native word.
    """
    count = len(nets)
    if len(corners) != count or len(degrees) != count:
        raise ValueError("every net needs its corners and degree.")
    starts: list[int | None] = []
    for net, corner_rows, degree in zip(nets, corners, degrees, strict=True):
        if len(net) != comb(int(degree) + parts - 1, parts - 1):
            raise ValueError("a simplex net must hold every control of its degree.")
        if len(corner_rows) != parts or any(len(row) != parts for row in corner_rows):
            raise ValueError("corners must be (parts, parts) integer weights.")
        net_bits = max((value.bit_length() for value in net), default=0)
        corner_bits = max(
            (value.bit_length() for row in corner_rows for value in row), default=0
        )
        starts.append(
            _limbs_for(net_bits + int(degree) * (corner_bits + parts.bit_length()) + 2)
        )
    results: list[tuple[int, int] | None] = [None] * count
    pending = [item for item in range(count) if starts[item] is not None]
    workers = _workers(parallelism)
    for limbs in _WORD_LIMBS:
        batch = [item for item in pending if cast(int, starts[item]) <= limbs]
        if not batch:
            continue
        width = 8 * limbs
        offsets = [0]
        for item in batch:
            offsets.append(offsets[-1] + len(nets[item]))
        values = ctypes.create_string_buffer(offsets[-1] * width)
        extremes = ctypes.create_string_buffer(16 * len(batch))
        status = ctypes.create_string_buffer(4 * len(batch))
        call = load_meshcore()["phx_mc_simplex_bernstein_blossoms"](
            len(batch),
            parts,
            array("i", [int(degrees[item]) for item in batch]).tobytes(),
            array("q", offsets).tobytes(),
            _words([value for item in batch for value in nets[item]], limbs),
            _words(
                [value for item in batch for row in corners[item] for value in row], limbs
            ),
            values,
            extremes,
            status,
            limbs,
            workers,
        )
        _raise_for_call(call, "simplex_bernstein_blossoms")
        payload = values.raw
        extreme_values = array("q", extremes.raw)
        status_values = array("i", status.raw)
        refused = []
        for position, item in enumerate(batch):
            if status_values[position] == _RANGE_ERROR:
                refused.append(item)
                continue
            _raise_for_call(status_values[position], "simplex_bernstein_blossoms")
            results[item] = (
                _word(payload, offsets[position] + extreme_values[2 * position], width),
                _word(
                    payload, offsets[position] + extreme_values[2 * position + 1], width
                ),
            )
        pending = [item for item in pending if cast(int, starts[item]) > limbs] + refused
    return results


@dataclass(frozen=True, slots=True)
class BisectedNets:
    """Halves of integer Bernstein nets over ``2 ** exponent`` times the input scale.

    ``halves[i]`` is ``(left, right)`` as ``(controls, components)`` integer rows,
    or ``None`` (typed refusal) when item ``i`` exceeds the widest native word.
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
    for net in nets:
        if len(net) != control_count or any(len(row) != components for row in net):
            raise ValueError("nets must be (controls, components) integer rows.")
    index_bytes = array("i", [int(value) for row in indices for value in row]).tobytes()
    if len(index_bytes) != 4 * control_count * slot_count:
        raise ValueError("indices must share one slot count.")
    exponent = max(int(row[first]) + int(row[second]) for row in indices)
    starts = [
        _limbs_for(
            max((value.bit_length() for row in net for value in row), default=0)
            + exponent
            + 2
        )
        for net in nets
    ]
    halves: list[tuple[_IntegerRows, _IntegerRows] | None] = [None] * count
    pending = [item for item in range(count) if starts[item] is not None]
    workers = _workers(parallelism)
    for limbs in _WORD_LIMBS:
        batch = [item for item in pending if cast(int, starts[item]) <= limbs]
        if not batch:
            continue
        width = 8 * limbs
        size = len(batch) * control_count * components * width
        left = ctypes.create_string_buffer(size)
        right = ctypes.create_string_buffer(size)
        exponent_out = ctypes.create_string_buffer(4)
        status = ctypes.create_string_buffer(4 * len(batch))
        call = load_meshcore()["phx_mc_bernstein_bisect"](
            len(batch),
            control_count,
            slot_count,
            components,
            index_bytes,
            first,
            second,
            _words(
                [value for item in batch for row in nets[item] for value in row], limbs
            ),
            left,
            right,
            exponent_out,
            status,
            limbs,
            workers,
        )
        _raise_for_call(call, "bernstein_bisect")
        if array("i", exponent_out.raw)[0] != exponent:
            raise AssertionError("native bisection exponent disagrees with the indices.")
        left_bytes = left.raw
        right_bytes = right.raw
        status_values = array("i", status.raw)
        refused = []
        for position, item in enumerate(batch):
            if status_values[position] == _RANGE_ERROR:
                refused.append(item)
                continue
            _raise_for_call(status_values[position], "bernstein_bisect")
            base = position * control_count * components
            left_rows, right_rows = (
                tuple(
                    tuple(
                        _word(payload, base + row * components + component, width)
                        for component in range(components)
                    )
                    for row in range(control_count)
                )
                for payload in (left_bytes, right_bytes)
            )
            halves[item] = (left_rows, right_rows)
        pending = [item for item in pending if cast(int, starts[item]) > limbs] + refused
    return BisectedNets(exponent, tuple(halves))


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
