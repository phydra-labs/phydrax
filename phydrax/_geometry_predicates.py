#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Filtered and exact geometric predicates.

Three evaluation modes share one sign convention:

- ``orient2d(a, b, c)`` is the sign of ``det[b - a, c - a]`` (positive for a
  counterclockwise triangle);
- ``orient3d(a, b, c, d)`` is the sign of ``det[b - a, c - a, d - a]`` (positive
  for a right-handed tetrahedron);
- ``incircle(a, b, c, d)`` is positive when ``d`` lies inside the circle through a
  counterclockwise ``(a, b, c)``;
- ``insphere(a, b, c, d, e)`` is positive when ``e`` lies inside the sphere
  through a positively oriented ``(a, b, c, d)``.

``FILTERED`` (host NumPy) and ``FILTERED_DEVICE`` (pure JAX, jittable) evaluate
the determinant in floating point and certify its sign with Shewchuk's static
stage-A error bounds (Adaptive Precision Floating-Point Arithmetic and Fast
Robust Geometric Predicates, 1997) derived for the input dtype, extended by an
absolute term covering gradual underflow and flush-to-zero.  A certified sign is
never wrong; unresolved entries are ``UNCERTAIN``.  Exact zeros are certified
structurally, when every monomial of the determinant contains a coordinate
difference of equal inputs.  ``EXACT`` resolves the uncertain entries of the host
filter with the native meshcore adaptive expansion arithmetic and raises
:class:`~phydrax._meshcore.MeshcoreUnavailableError` when the library is absent.

``segment_intersections_2d`` and ``polygon_simplicity_2d`` are host algorithms
built on ``orient2d`` and exact coordinate comparisons; their classifications
are exact wherever every contributing sign is certified.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum, StrEnum
from types import ModuleType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ._bvh import bvh_overlap_pair_blocks, prepare_bvh
from ._meshcore import (
    exact_incircle,
    exact_insphere,
    exact_orient2d,
    exact_orient3d,
    load_meshcore,
    meshcore_available,
)
from ._strict import StrictModule
from ._trainable import NonTrainableState


class PredicateMode(StrEnum):
    """Evaluation route of a geometric predicate."""

    FILTERED = "filtered"
    EXACT = "exact"
    FILTERED_DEVICE = "filtered_device"


class PredicateSign(IntEnum):
    """Predicate outcome; ``UNCERTAIN`` marks an unresolved filtered sign."""

    NEGATIVE = -1
    ZERO = 0
    POSITIVE = 1
    UNCERTAIN = 2


class PredicateResult(StrictModule, NonTrainableState):
    """Batched predicate signs with certification.

    ``signs`` is int8 with :class:`PredicateSign` values and ``certain`` marks the
    entries whose sign is exact.  Host modes carry NumPy arrays; the device mode
    carries JAX arrays.
    """

    signs: Array
    certain: Array
    mode: PredicateMode = eqx.field(static=True)

    def __init__(self, signs: Array, certain: Array, mode: PredicateMode) -> None:
        if not isinstance(mode, PredicateMode):
            raise TypeError("mode must be a PredicateMode.")
        if signs.dtype != np.int8:
            raise TypeError("Predicate signs must be int8.")
        if certain.dtype != np.bool_:
            raise TypeError("Predicate certainty must be boolean.")
        if signs.shape != certain.shape:
            raise ValueError("Predicate signs and certainty must share one shape.")
        self.signs = signs
        self.certain = certain
        self.mode = mode


def resolve_host_predicate_mode(mode: PredicateMode, /) -> PredicateMode:
    """Effective mode of a host geometric algorithm that reports unresolved decisions.

    ``EXACT`` requires meshcore; without it the host filter is used and its
    unresolved entries surface as the caller's uncertain-predicate status.
    """

    match mode:
        case PredicateMode.EXACT:
            return PredicateMode.EXACT if meshcore_available() else PredicateMode.FILTERED
        case PredicateMode.FILTERED:
            return PredicateMode.FILTERED
        case PredicateMode.FILTERED_DEVICE:
            raise ValueError(
                "Host geometric algorithms require the FILTERED or EXACT predicate mode."
            )
        case _:
            raise TypeError("mode must be a PredicateMode.")


# ------------------------------------------------------------------ filter kernels
#
# Each kernel returns the floating determinant in the package convention, its
# Shewchuk permanent, the structural exact-zero mask, the largest coordinate
# difference, and the static constants (relative bound coefficient, polynomial
# degree, absolute-term multiplier).  Kernels are written once for NumPy and JAX.


def _orient2d_kernel(xp: ModuleType, a: Any, b: Any, c: Any, /) -> Any:
    acx = a[..., 0] - c[..., 0]
    bcx = b[..., 0] - c[..., 0]
    acy = a[..., 1] - c[..., 1]
    bcy = b[..., 1] - c[..., 1]
    detleft = acx * bcy
    detright = acy * bcx
    det = detleft - detright
    permanent = xp.abs(detleft) + xp.abs(detright)
    ex = a[..., 0] == c[..., 0]
    ey = a[..., 1] == c[..., 1]
    fx = b[..., 0] == c[..., 0]
    fy = b[..., 1] == c[..., 1]
    zero = (ex | fy) & (ey | fx)
    largest = xp.maximum(
        xp.maximum(xp.abs(acx), xp.abs(bcx)), xp.maximum(xp.abs(acy), xp.abs(bcy))
    )
    return det, permanent, zero, largest


def _minor_zero(x_equal: Any, y_equal: Any, first: int, second: int, /) -> Any:
    # det[[x_i, y_i], [x_j, y_j]] = x_i y_j - x_j y_i is structurally zero when
    # each monomial has a vanishing difference.
    return (x_equal[first] | y_equal[second]) & (x_equal[second] | y_equal[first])


def _orient3d_kernel(xp: ModuleType, a: Any, b: Any, c: Any, d: Any, /) -> Any:
    rows = (a, b, c)
    x = tuple(row[..., 0] - d[..., 0] for row in rows)
    y = tuple(row[..., 1] - d[..., 1] for row in rows)
    z = tuple(row[..., 2] - d[..., 2] for row in rows)
    bxcy = x[1] * y[2]
    cxby = x[2] * y[1]
    cxay = x[2] * y[0]
    axcy = x[0] * y[2]
    axby = x[0] * y[1]
    bxay = x[1] * y[0]
    shewchuk = z[0] * (bxcy - cxby) + z[1] * (cxay - axcy) + z[2] * (axby - bxay)
    permanent = (
        (xp.abs(bxcy) + xp.abs(cxby)) * xp.abs(z[0])
        + (xp.abs(cxay) + xp.abs(axcy)) * xp.abs(z[1])
        + (xp.abs(axby) + xp.abs(bxay)) * xp.abs(z[2])
    )
    xe = tuple((row[..., 0] == d[..., 0]) for row in rows)
    ye = tuple((row[..., 1] == d[..., 1]) for row in rows)
    ze = tuple((row[..., 2] == d[..., 2]) for row in rows)
    zero = (
        (ze[0] | _minor_zero(xe, ye, 1, 2))
        & (ze[1] | _minor_zero(xe, ye, 2, 0))
        & (ze[2] | _minor_zero(xe, ye, 0, 1))
    )
    largest = xp.max(xp.abs(xp.stack(x + y + z, axis=-1)), axis=-1)
    return -shewchuk, permanent, zero, largest


def _incircle_kernel(xp: ModuleType, a: Any, b: Any, c: Any, d: Any, /) -> Any:
    rows = (a, b, c)
    x = tuple(row[..., 0] - d[..., 0] for row in rows)
    y = tuple(row[..., 1] - d[..., 1] for row in rows)
    bxcy = x[1] * y[2]
    cxby = x[2] * y[1]
    cxay = x[2] * y[0]
    axcy = x[0] * y[2]
    axby = x[0] * y[1]
    bxay = x[1] * y[0]
    lift = tuple(x[k] * x[k] + y[k] * y[k] for k in range(3))
    det = lift[0] * (bxcy - cxby) + lift[1] * (cxay - axcy) + lift[2] * (axby - bxay)
    permanent = (
        (xp.abs(bxcy) + xp.abs(cxby)) * lift[0]
        + (xp.abs(cxay) + xp.abs(axcy)) * lift[1]
        + (xp.abs(axby) + xp.abs(bxay)) * lift[2]
    )
    xe = tuple((row[..., 0] == d[..., 0]) for row in rows)
    ye = tuple((row[..., 1] == d[..., 1]) for row in rows)
    lift_zero = tuple(xe[k] & ye[k] for k in range(3))
    zero = (
        (lift_zero[0] | _minor_zero(xe, ye, 1, 2))
        & (lift_zero[1] | _minor_zero(xe, ye, 2, 0))
        & (lift_zero[2] | _minor_zero(xe, ye, 0, 1))
    )
    largest = xp.max(xp.abs(xp.stack(x + y, axis=-1)), axis=-1)
    return det, permanent, zero, largest


def _insphere_kernel(xp: ModuleType, a: Any, b: Any, c: Any, d: Any, e: Any, /) -> Any:
    rows = (a, b, c, d)
    x = tuple(row[..., 0] - e[..., 0] for row in rows)
    y = tuple(row[..., 1] - e[..., 1] for row in rows)
    z = tuple(row[..., 2] - e[..., 2] for row in rows)
    products = {(i, j): x[i] * y[j] for i in range(4) for j in range(4) if i != j}
    ab = products[0, 1] - products[1, 0]
    bc = products[1, 2] - products[2, 1]
    cd = products[2, 3] - products[3, 2]
    da = products[3, 0] - products[0, 3]
    ac = products[0, 2] - products[2, 0]
    bd = products[1, 3] - products[3, 1]
    abc = z[0] * bc - z[1] * ac + z[2] * ab
    bcd = z[1] * cd - z[2] * bd + z[3] * bc
    cda = z[2] * da + z[3] * ac + z[0] * cd
    dab = z[3] * ab + z[0] * bd + z[1] * da
    lift = tuple(x[k] * x[k] + y[k] * y[k] + z[k] * z[k] for k in range(4))
    shewchuk = (lift[3] * abc - lift[2] * dab) + (lift[1] * cda - lift[0] * bcd)

    def plus(i: int, j: int, /) -> Any:
        return xp.abs(products[i, j]) + xp.abs(products[j, i])

    zp = tuple(xp.abs(value) for value in z)
    permanent = (
        (plus(2, 3) * zp[1] + plus(3, 1) * zp[2] + plus(1, 2) * zp[3]) * lift[0]
        + (plus(3, 0) * zp[2] + plus(0, 2) * zp[3] + plus(2, 3) * zp[0]) * lift[1]
        + (plus(0, 1) * zp[3] + plus(1, 3) * zp[0] + plus(3, 0) * zp[1]) * lift[2]
        + (plus(1, 2) * zp[0] + plus(2, 0) * zp[1] + plus(0, 1) * zp[2]) * lift[3]
    )
    xe = tuple((row[..., 0] == e[..., 0]) for row in rows)
    ye = tuple((row[..., 1] == e[..., 1]) for row in rows)
    ze = tuple((row[..., 2] == e[..., 2]) for row in rows)

    def triple_zero(i: int, j: int, k: int, /) -> Any:
        return (
            (ze[i] | _minor_zero(xe, ye, j, k))
            & (ze[j] | _minor_zero(xe, ye, i, k))
            & (ze[k] | _minor_zero(xe, ye, i, j))
        )

    lift_zero = tuple(xe[k] & ye[k] & ze[k] for k in range(4))
    zero = (
        (lift_zero[3] | triple_zero(0, 1, 2))
        & (lift_zero[2] | triple_zero(3, 0, 1))
        & (lift_zero[1] | triple_zero(2, 3, 0))
        & (lift_zero[0] | triple_zero(1, 2, 3))
    )
    largest = xp.max(xp.abs(xp.stack(x + y + z, axis=-1)), axis=-1)
    return -shewchuk, permanent, zero, largest


class _Filter:
    """Static description of one predicate filter."""

    __slots__ = (
        "arity",
        "coefficient",
        "degree",
        "kernel",
        "multiplier",
        "name",
        "width",
    )

    def __init__(
        self,
        name: str,
        kernel: Callable,
        width: int,
        arity: int,
        coefficient: tuple[float, float],
        degree: int,
        multiplier: float,
    ) -> None:
        self.name = name
        self.kernel = kernel
        self.width = width
        self.arity = arity
        # Shewchuk's stage-A bound (first + second * u) * u for unit roundoff u.
        self.coefficient = coefficient
        self.degree = degree
        # Absolute underflow term: every rounding may lose up to 4 * tiny, which
        # propagates through at most `degree` factors bounded by (1 + largest).
        self.multiplier = multiplier


_ORIENT2D = _Filter("orient2d", _orient2d_kernel, 2, 3, (3.0, 16.0), 2, 32.0)
_ORIENT3D = _Filter("orient3d", _orient3d_kernel, 3, 4, (7.0, 56.0), 3, 144.0)
_INCIRCLE = _Filter("incircle", _incircle_kernel, 2, 4, (10.0, 96.0), 4, 384.0)
_INSPHERE = _Filter("insphere", _insphere_kernel, 3, 5, (16.0, 224.0), 5, 2880.0)


def _filter(
    xp: ModuleType, spec: _Filter, points: tuple, dtype: Any, safety: float, /
) -> tuple:
    det, permanent, zero, largest = spec.kernel(xp, *points)
    info = np.finfo(dtype)
    unit = float(info.eps) / 2.0
    first, second = spec.coefficient
    relative = (first + second * unit) * unit
    absolute = spec.multiplier * float(info.tiny) * (1.0 + largest) ** (spec.degree - 1)
    bound = safety * (relative * permanent + absolute)
    finite = xp.isfinite(det) & xp.isfinite(bound)
    decided = finite & (xp.abs(det) > bound)
    for point in points:
        finite = finite & xp.all(xp.isfinite(point), axis=-1)
    zero = zero & finite
    certain = zero | decided
    sign = xp.where(zero, 0, xp.sign(det)).astype(xp.int8)
    signs = xp.where(certain, sign, xp.asarray(PredicateSign.UNCERTAIN, dtype=xp.int8))
    return signs.astype(xp.int8), certain


# ------------------------------------------------------------------ host routes


def _host_points(
    values: tuple, width: int, /
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    arrays = []
    for value in values:
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.integer):
            if np.any((array > 2**53) | (array < -(2**53))):
                raise ValueError("Integer coordinates must be exactly representable.")
        elif not np.issubdtype(array.dtype, np.floating):
            raise TypeError(
                "Predicate coordinates must be real floating or integer arrays."
            )
        if array.ndim < 1 or array.shape[-1] != width:
            raise ValueError(f"Predicate coordinates must have shape (..., {width}).")
        arrays.append(array.astype(np.float64, copy=False))
    leading = np.broadcast_shapes(*(array.shape[:-1] for array in arrays))
    flat = tuple(
        np.ascontiguousarray(
            np.broadcast_to(array, leading + (width,)).reshape((-1, width))
        )
        for array in arrays
    )
    return flat, leading


_EXACT_ROUTES = {
    "orient2d": exact_orient2d,
    "orient3d": exact_orient3d,
    "incircle": exact_incircle,
    "insphere": exact_insphere,
}


def _host(spec: _Filter, values: tuple, mode: PredicateMode, /) -> PredicateResult:
    if mode is PredicateMode.EXACT:
        load_meshcore()
    flat, leading = _host_points(values, spec.width)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        signs, certain = _filter(np, spec, flat, np.float64, 1.0)
    if mode is PredicateMode.EXACT:
        unresolved = np.flatnonzero(~certain)
        if unresolved.size:
            signs[unresolved] = _EXACT_ROUTES[spec.name](
                *(array[unresolved] for array in flat)
            )
            certain[unresolved] = True
    return PredicateResult(signs.reshape(leading), certain.reshape(leading), mode)


# ------------------------------------------------------------------ device route


_SUBNORMAL_MASKS = {
    np.dtype(np.float32): (jnp.uint32, 0x7F800000, 0x007FFFFF),
    np.dtype(np.float64): (jnp.uint64, 0x7FF0000000000000, 0x000FFFFFFFFFFFFF),
}


def _device_points(values: tuple, width: int, /) -> tuple[tuple[Array, ...], np.dtype]:
    arrays = []
    for value in values:
        array = jnp.asarray(value)
        if not isinstance(value, jax.Array):
            source = np.asarray(value).dtype
            if (
                np.issubdtype(source, np.floating)
                and source.itemsize > array.dtype.itemsize
            ):
                raise ValueError(
                    "Device predicates would round the input coordinates; enable "
                    "jax_enable_x64 or pass arrays of the device dtype."
                )
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError("Predicate coordinates must be real floating arrays.")
        if array.ndim < 1 or array.shape[-1] != width:
            raise ValueError(f"Predicate coordinates must have shape (..., {width}).")
        arrays.append(array)
    dtype = jnp.result_type(*arrays)
    if np.dtype(dtype) not in _SUBNORMAL_MASKS:
        raise TypeError("Device predicates support float32 and float64 coordinates.")
    broadcast = jnp.broadcast_arrays(*(array.astype(dtype) for array in arrays))
    return tuple(broadcast), np.dtype(dtype)


def _device(spec: _Filter, values: tuple, /) -> PredicateResult:
    points, dtype = _device_points(values, spec.width)
    # Safety factor 2 covers backend contraction of products into fused
    # multiply-adds, which only removes roundings.
    signs, certain = _filter(jnp, spec, points, dtype, 2.0)
    unsigned, exponent_mask, mantissa_mask = _SUBNORMAL_MASKS[dtype]
    # Denormals-are-zero backends see subnormal inputs as zero; such entries
    # are never certified.
    subnormal = jnp.zeros(signs.shape, dtype=jnp.bool_)
    for point in points:
        bits = jax.lax.bitcast_convert_type(point, unsigned)
        is_subnormal = ((bits & exponent_mask) == 0) & ((bits & mantissa_mask) != 0)
        subnormal = subnormal | jnp.any(is_subnormal, axis=-1)
    certain = certain & ~subnormal
    signs = jnp.where(certain, signs, jnp.int8(PredicateSign.UNCERTAIN)).astype(jnp.int8)
    return PredicateResult(signs, certain, PredicateMode.FILTERED_DEVICE)


def _evaluate(spec: _Filter, values: tuple, mode: PredicateMode, /) -> PredicateResult:
    match mode:
        case PredicateMode.FILTERED | PredicateMode.EXACT:
            return _host(spec, values, mode)
        case PredicateMode.FILTERED_DEVICE:
            return _device(spec, values)
        case _:
            raise TypeError("mode must be a PredicateMode.")


def orient2d(a: Any, b: Any, c: Any, /, *, mode: PredicateMode) -> PredicateResult:
    """Sign of ``det[b - a, c - a]`` for ``(..., 2)`` coordinates (leading axes broadcast)."""

    return _evaluate(_ORIENT2D, (a, b, c), mode)


def orient3d(
    a: Any, b: Any, c: Any, d: Any, /, *, mode: PredicateMode
) -> PredicateResult:
    """Sign of ``det[b - a, c - a, d - a]`` for ``(..., 3)`` coordinates."""

    return _evaluate(_ORIENT3D, (a, b, c, d), mode)


def incircle(
    a: Any, b: Any, c: Any, d: Any, /, *, mode: PredicateMode
) -> PredicateResult:
    """Positive when ``d`` lies inside the circle through counterclockwise ``(a, b, c)``."""

    return _evaluate(_INCIRCLE, (a, b, c, d), mode)


def insphere(
    a: Any, b: Any, c: Any, d: Any, e: Any, /, *, mode: PredicateMode
) -> PredicateResult:
    """Positive when ``e`` lies inside the sphere through positively oriented ``(a..d)``."""

    return _evaluate(_INSPHERE, (a, b, c, d, e), mode)


# ------------------------------------------------------------------ segments and polygons


class SegmentIntersectionStatus(IntEnum):
    """Contact class of two closed segments.

    ``ENDPOINT_CONTACT`` is a single common point that is an endpoint of at least
    one segment; ``COLLINEAR_OVERLAP`` is a common piece of positive length.
    """

    DISJOINT = 0
    PROPER_CROSSING = 1
    ENDPOINT_CONTACT = 2
    COLLINEAR_OVERLAP = 3
    UNCERTAIN = 4


class SegmentIntersectionResult(StrictModule, NonTrainableState):
    """Batched closed-segment contact classes (int8 :class:`SegmentIntersectionStatus`)."""

    status: Array | np.ndarray
    mode: PredicateMode = eqx.field(static=True)

    def __init__(self, status: Array | np.ndarray, mode: PredicateMode) -> None:
        if not isinstance(mode, PredicateMode):
            raise TypeError("mode must be a PredicateMode.")
        if status.dtype != np.int8:
            raise TypeError("Segment intersection status must be int8.")
        self.status = status
        self.mode = mode

    @property
    def certain(self) -> np.ndarray:
        return np.asarray(self.status) != SegmentIntersectionStatus.UNCERTAIN


class PolygonSimplicityStatus(IntEnum):
    """Whether a closed vertex loop bounds a simple polygon."""

    SIMPLE = 0
    SELF_INTERSECTING = 1
    UNCERTAIN = 2


class PolygonSimplicityResult(StrictModule, NonTrainableState):
    """Per-polygon simplicity, orientation, and the edge-pair evidence behind it.

    ``orientation`` is the :class:`PredicateSign` of the signed area of SIMPLE
    loops (never ``ZERO``: a simple loop encloses positive area) and
    ``UNCERTAIN`` for every other loop or an unresolved sign.
    ``intersecting_pairs`` counts certified offending edge pairs (non-adjacent
    edges in contact, or adjacent edges meeting beyond their shared vertex) and
    ``uncertain_pairs`` the candidate pairs whose class stayed unresolved.
    """

    status: Array
    orientation: Array
    intersecting_pairs: Array
    uncertain_pairs: Array
    candidate_pair_count: int = eqx.field(static=True)
    candidate_capacity_exceeded: bool = eqx.field(static=True)
    mode: PredicateMode = eqx.field(static=True)

    def __init__(
        self,
        status: Array,
        orientation: Array,
        intersecting_pairs: Array,
        uncertain_pairs: Array,
        mode: PredicateMode,
        *,
        candidate_pair_count: int,
        candidate_capacity_exceeded: bool,
    ) -> None:
        if not isinstance(mode, PredicateMode):
            raise TypeError("mode must be a PredicateMode.")
        if status.dtype != np.int8 or orientation.dtype != np.int8:
            raise TypeError("Polygon simplicity status and orientation must be int8.")
        if intersecting_pairs.dtype != np.int64 or uncertain_pairs.dtype != np.int64:
            raise TypeError("Polygon simplicity pair counts must be int64.")
        if not (
            status.shape
            == orientation.shape
            == intersecting_pairs.shape
            == uncertain_pairs.shape
        ):
            raise ValueError("Polygon simplicity arrays must share one shape.")
        count = int(candidate_pair_count)
        if count < 0:
            raise ValueError("candidate_pair_count must be non-negative.")
        if not isinstance(candidate_capacity_exceeded, (bool, np.bool_)):
            raise TypeError("candidate_capacity_exceeded must be bool.")
        self.status = status
        self.orientation = orientation
        self.intersecting_pairs = intersecting_pairs
        self.uncertain_pairs = uncertain_pairs
        self.candidate_pair_count = count
        self.candidate_capacity_exceeded = bool(candidate_capacity_exceeded)
        self.mode = mode


def _require_host_mode(mode: PredicateMode, /) -> None:
    match mode:
        case PredicateMode.FILTERED | PredicateMode.EXACT:
            return
        case PredicateMode.FILTERED_DEVICE:
            raise ValueError(
                "Segment and polygon predicates require the FILTERED or EXACT mode."
            )
        case _:
            raise TypeError("mode must be a PredicateMode.")


def _classify_segments(
    a: Any, b: Any, c: Any, d: Any, mode: PredicateMode, /
) -> np.ndarray:
    """Contact classes of closed segments ``ab`` and ``cd`` of shape ``(..., 2)``."""

    orientation = orient2d(
        np.stack((a, a, c, c)),
        np.stack((b, b, d, d)),
        np.stack((c, d, a, b)),
        mode=mode,
    )
    signs = orientation.signs.astype(np.int16)
    certain = orientation.certain
    first = signs[0] * signs[1]
    second = signs[2] * signs[3]
    # One certified supporting line with both ends of the other segment strictly
    # on one side separates the segments whatever the remaining signs are.
    separated = (certain[0] & certain[1] & (first > 0)) | (
        certain[2] & certain[3] & (second > 0)
    )
    # Coordinate comparisons are exact; for collinear segments the common box is
    # their common piece.
    lower = np.maximum(np.minimum(a, b), np.minimum(c, d))
    upper = np.minimum(np.maximum(a, b), np.maximum(c, d))
    boxes_apart = np.any(lower > upper, axis=-1)
    collinear = np.all(signs == 0, axis=0)
    collinear_status = np.where(
        np.all(lower == upper, axis=-1),
        SegmentIntersectionStatus.ENDPOINT_CONTACT,
        SegmentIntersectionStatus.COLLINEAR_OVERLAP,
    )
    general_status = np.where(
        (first < 0) & (second < 0),
        SegmentIntersectionStatus.PROPER_CROSSING,
        np.where(
            (first <= 0) & (second <= 0),
            SegmentIntersectionStatus.ENDPOINT_CONTACT,
            SegmentIntersectionStatus.DISJOINT,
        ),
    )
    status = np.where(
        np.all(certain, axis=0),
        np.where(collinear, collinear_status, general_status),
        SegmentIntersectionStatus.UNCERTAIN,
    )
    return np.where(
        separated | boxes_apart, SegmentIntersectionStatus.DISJOINT, status
    ).astype(np.int8)


def segment_intersections_2d(
    a: Any, b: Any, c: Any, d: Any, /, *, mode: PredicateMode
) -> SegmentIntersectionResult:
    """Classify the contact of closed segments ``ab`` and ``cd`` (``(..., 2)``, broadcast).

    Host-only (``FILTERED`` or ``EXACT``); unresolved filtered signs yield
    ``UNCERTAIN`` unless an exact coordinate or certified sign test separates
    the segments.
    """

    _require_host_mode(mode)
    flat, leading = _host_points((a, b, c, d), 2)
    status = _classify_segments(flat[0], flat[1], flat[2], flat[3], mode)
    return SegmentIntersectionResult(status.reshape(leading), mode)


def _edge_pair_evidence(
    points: np.ndarray, mode: PredicateMode, maximum_candidate_pairs: int, /
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Bounded offending and unresolved edge-pair evidence."""

    polygon_count, corner_count, _ = points.shape
    intersecting = np.zeros((polygon_count,), dtype=np.int64)
    uncertain = np.zeros((polygon_count,), dtype=np.int64)
    starts = points.reshape((-1, 2))
    stops = np.roll(points, -1, axis=1).reshape((-1, 2))
    # An exact polygon coordinate axis keeps the broad phase within polygons.
    group = np.repeat(np.arange(polygon_count, dtype=np.float64), corner_count)
    bvh = prepare_bvh(
        np.column_stack((np.minimum(starts, stops), group)),
        np.column_stack((np.maximum(starts, stops), group)),
        dtype=np.float64,
    )
    candidate_count = 0
    exceeded = False
    for first, second in bvh_overlap_pair_blocks(bvh, bvh, include_touching=True):
        keep = first < second
        first = first[keep]
        second = second[keep]
        remaining = maximum_candidate_pairs - candidate_count
        if first.size > remaining:
            first = first[:remaining]
            second = second[:remaining]
            exceeded = True
        candidate_count += first.size
        if first.size:
            status = _classify_segments(
                starts[first], stops[first], starts[second], stops[second], mode
            )
            gap = second - first
            adjacent = (gap == 1) | (gap == corner_count - 1)
            allowed = (status == SegmentIntersectionStatus.DISJOINT) | (
                adjacent & (status == SegmentIntersectionStatus.ENDPOINT_CONTACT)
            )
            unresolved = status == SegmentIntersectionStatus.UNCERTAIN
            polygon = first // corner_count
            np.add.at(intersecting, polygon[~allowed & ~unresolved], 1)
            np.add.at(uncertain, polygon[unresolved], 1)
        if exceeded:
            # A proven crossing remains invalid; every other loop is unresolved
            # because unvisited candidate pairs may intersect.
            uncertain += intersecting == 0
            break
    return intersecting, uncertain, candidate_count, exceeded


def _extreme_turn(points: np.ndarray, mode: PredicateMode, /) -> PredicateResult:
    """Turn at the lexicographically smallest vertex of ``(polygons, n, 2)`` loops.

    That vertex is extreme, so for a simple loop its turn is the orientation.
    """

    corner_count = points.shape[1]
    order = np.lexsort((points[..., 1], points[..., 0]), axis=-1)[:, 0]
    rows = np.arange(points.shape[0])
    return orient2d(
        points[rows, (order - 1) % corner_count],
        points[rows, order],
        points[rows, (order + 1) % corner_count],
        mode=mode,
    )


def polygon_simplicity_2d(
    vertices: Any,
    /,
    *,
    mode: PredicateMode,
    maximum_candidate_pairs: int = 5_000_000,
) -> PolygonSimplicityResult:
    """Certify that closed vertex loops ``(..., n, 2)`` bound simple polygons.

    A loop is simple when non-adjacent edges are disjoint and adjacent edges
    meet only at their shared vertex; repeated vertices therefore make a loop
    self-intersecting. Candidate edge pairs come from a bounded BVH block stream
    over exact edge boxes, partitioned by polygon; exhausting
    ``maximum_candidate_pairs`` leaves every unproved loop ``UNCERTAIN``.
    Host-only (``FILTERED`` or ``EXACT``).
    """

    _require_host_mode(mode)
    if isinstance(maximum_candidate_pairs, bool) or not isinstance(
        maximum_candidate_pairs, (int, np.integer)
    ):
        raise TypeError("maximum_candidate_pairs must be an integer.")
    capacity = int(maximum_candidate_pairs)
    if capacity <= 0:
        raise ValueError("maximum_candidate_pairs must be positive.")
    array = np.asarray(vertices)
    if array.ndim < 2 or array.shape[-2] < 3:
        raise ValueError("Polygon vertices must have shape (..., n >= 3, 2).")
    (flat,), _ = _host_points((array,), 2)
    if not np.all(np.isfinite(flat)):
        raise ValueError("Polygon vertices must be finite.")
    leading = array.shape[:-2]
    points = flat.reshape((-1, array.shape[-2], 2))
    if points.shape[0] == 0:
        empty = np.zeros(leading, dtype=np.int64)
        return PolygonSimplicityResult(
            empty.astype(np.int8),
            empty.astype(np.int8),
            empty,
            empty.copy(),
            mode,
            candidate_pair_count=0,
            candidate_capacity_exceeded=False,
        )
    intersecting, uncertain, candidate_count, exceeded = _edge_pair_evidence(
        points, mode, capacity
    )
    status = np.where(
        intersecting > 0,
        PolygonSimplicityStatus.SELF_INTERSECTING,
        np.where(
            uncertain > 0,
            PolygonSimplicityStatus.UNCERTAIN,
            PolygonSimplicityStatus.SIMPLE,
        ),
    ).astype(np.int8)
    turn = _extreme_turn(points, mode)
    orientation = np.where(
        (status == PolygonSimplicityStatus.SIMPLE) & turn.certain,
        turn.signs,
        np.int8(PredicateSign.UNCERTAIN),
    ).astype(np.int8)
    return PolygonSimplicityResult(
        status.reshape(leading),
        orientation.reshape(leading),
        intersecting.reshape(leading),
        uncertain.reshape(leading),
        mode,
        candidate_pair_count=candidate_count,
        candidate_capacity_exceeded=exceeded,
    )


__all__ = [
    "PolygonSimplicityResult",
    "PolygonSimplicityStatus",
    "PredicateMode",
    "PredicateResult",
    "PredicateSign",
    "SegmentIntersectionResult",
    "SegmentIntersectionStatus",
    "incircle",
    "insphere",
    "orient2d",
    "orient3d",
    "polygon_simplicity_2d",
    "resolve_host_predicate_mode",
    "segment_intersections_2d",
]
