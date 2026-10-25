#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evaluate filtered geometric predicates and resolve signs with exact arithmetic.

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
absolute term covering gradual underflow and flush-to-zero. A certified sign is
never wrong; unresolved entries are ``UNCERTAIN``. Exact zeros are certified
structurally, when every monomial of the determinant contains a coordinate
difference of equal inputs. ``EXACT`` resolves uncertain host entries with the
native meshcore adaptive expansion arithmetic when available and otherwise with
dyadic-rational evaluation of the input floating-point coordinates without rounding.

``segment_intersections_2d`` and ``polygon_simplicity_2d`` are host algorithms
built on ``orient2d`` and coordinate comparisons without rounding; their classifications
are certified wherever every contributing sign is certified.

Host ``orient2d``, ``orient3d`` and ``segment_intersections_2d`` also accept
integer banks: object arrays of Python integers, for example a
rational source scaled to one positive common denominator. Their kernel
determinant is evaluated without rounding and every sign is certain. Before
evaluation the explicit coordinate ledger counts the kernel visits and bounds
every bigint temporary from the measured input bits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from enum import IntEnum, StrEnum
from fractions import Fraction
from types import ModuleType
from typing import Any, cast

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
from ._resource_ticket import HostBatchTicket
from ._strict import StrictModule
from ._trainable import NonTrainableState
from ._validation import nonnegative_integer
from .discretization._coordinate_budget import CoordinateEnclosureBudget
from .typing import checked, ConvertibleToArray, parse


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
    entries whose sign is certified. All routes publish JAX arrays.
    """

    signs: Array
    certain: Array
    mode: PredicateMode = eqx.field(static=True)

    def __init__(self, signs: Array, certain: Array, mode: PredicateMode) -> None:
        if not isinstance(mode, PredicateMode):
            raise TypeError(
                f"PredicateResult.mode must be a PredicateMode; got {type(mode).__name__}"
            )
        if signs.dtype != np.int8:
            raise TypeError(
                f"PredicateResult.signs.dtype must be int8; got {signs.dtype}"
            )
        if certain.dtype != np.bool_:
            raise TypeError(
                f"PredicateResult.certain.dtype must be boolean; got {certain.dtype}"
            )
        if signs.shape != certain.shape:
            raise ValueError(
                f"PredicateResult.signs.shape must equal certain.shape; got {signs.shape} and {certain.shape}"
            )
        self.signs = signs
        self.certain = certain
        self.mode = mode


class HostPredicateResult(StrictModule, NonTrainableState):
    """Retain NumPy predicate signs and certainty without device placement."""

    signs: np.ndarray
    certain: np.ndarray
    mode: PredicateMode = eqx.field(static=True)


def resolve_host_predicate_mode(mode: PredicateMode, /) -> PredicateMode:
    """Return the effective mode of a host geometric algorithm.

    ``EXACT`` uses the optional meshcore adaptive predicates when available and
    the dyadic host fallback otherwise. Both evaluate without rounding;
    ``FILTERED`` alone may return unresolved entries.
    """

    match mode:
        case PredicateMode.EXACT | PredicateMode.FILTERED:
            return mode
        case PredicateMode.FILTERED_DEVICE:
            raise ValueError(
                f"resolve_host_predicate_mode.mode must be FILTERED or EXACT; got {mode!r}"
            )
        case _:
            raise TypeError(
                f"resolve_host_predicate_mode.mode must be a PredicateMode; got {type(mode).__name__}"
            )


# ------------------------------------------------------------------ filter kernels
#
# Each kernel returns the floating determinant in the package convention, its
# Shewchuk permanent, the structural zero mask, the largest coordinate
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
    values: tuple[np.ndarray, ...],
    width: int,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    arrays = []
    hosts = values
    integer_bank = any(array.dtype == object for array in hosts)
    for array in hosts:
        if integer_bank:
            # One integer bank representation: Python integers only. Fixed-width, boolean,
            # floating or rational entries would mix two arithmetics.
            if array.dtype != object:
                raise TypeError(
                    f"_host_points.coordinates.dtype must be object for integer banks; got {array.dtype}"
                )
            with (
                nullcontext()
                if budget is None
                else exact_charge(array.size, budget=budget)
            ):
                if any(type(item) is not int for item in array.flat):
                    raise TypeError(
                        "_host_points.coordinates.entries must be Python integers in object arrays; "
                        f"got {tuple(sorted({type(item).__name__ for item in array.flat if type(item) is not int}))}"
                    )
        elif np.issubdtype(array.dtype, np.integer):
            if np.any((array > 2**53) | (array < -(2**53))):
                raise ValueError(
                    f"_host_points.coordinates must lie in [-2**53, 2**53] for float64 conversion; got {array}"
                )
        elif not np.issubdtype(array.dtype, np.floating):
            raise TypeError(
                f"_host_points.coordinates.dtype must be real floating or integer; got {array.dtype}"
            )
        if array.ndim < 1 or array.shape[-1] != width:
            raise ValueError(
                f"_host_points.coordinates.shape must be (..., {width}); got {array.shape}"
            )
        arrays.append(array if integer_bank else array.astype(np.float64, copy=False))
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


def _normalize_dyadic(numerator: int, exponent: int, /) -> tuple[int, int]:
    """Normalize ``numerator * 2**exponent`` by removing even numerator factors."""
    if numerator == 0:
        return 0, 0
    magnitude = abs(numerator)
    shift = (magnitude & -magnitude).bit_length() - 1
    return numerator >> shift, exponent + shift


def _dyadic(value: np.float64, /) -> tuple[int, int]:
    """Exact integer significand and binary exponent of one finite binary64."""
    numerator, denominator = float(value).as_integer_ratio()
    return _normalize_dyadic(numerator, 1 - denominator.bit_length())


def _dyadic_add(first: tuple[int, int], second: tuple[int, int], /) -> tuple[int, int]:
    exponent = min(first[1], second[1])
    numerator = (first[0] << (first[1] - exponent)) + (
        second[0] << (second[1] - exponent)
    )
    return _normalize_dyadic(numerator, exponent)


def _dyadic_subtract(
    first: tuple[int, int], second: tuple[int, int], /
) -> tuple[int, int]:
    return _dyadic_add(first, (-second[0], second[1]))


def _dyadic_multiply(
    first: tuple[int, int], second: tuple[int, int], /
) -> tuple[int, int]:
    return _normalize_dyadic(first[0] * second[0], first[1] + second[1])


def _dyadic_sign(value: tuple[int, int], /) -> np.int8:
    return np.int8((value[0] > 0) - (value[0] < 0))


def _exact_dyadic_row(
    spec: _Filter, points: tuple[np.ndarray, ...], row: int, /
) -> np.int8:
    """Exact sign of one unresolved predicate on binary64 coordinates."""
    values = tuple(
        tuple(_dyadic(coordinate) for coordinate in point[row]) for point in points
    )

    def difference(first: int, second: int, axis: int, /) -> tuple[int, int]:
        return _dyadic_subtract(values[first][axis], values[second][axis])

    if spec.name == "orient2d":
        ab = (difference(1, 0, 0), difference(1, 0, 1))
        ac = (difference(2, 0, 0), difference(2, 0, 1))
        determinant = _dyadic_subtract(
            _dyadic_multiply(ab[0], ac[1]), _dyadic_multiply(ab[1], ac[0])
        )
        return _dyadic_sign(determinant)
    if spec.name == "orient3d":
        ab = tuple(difference(1, 0, axis) for axis in range(3))
        ac = tuple(difference(2, 0, axis) for axis in range(3))
        ad = tuple(difference(3, 0, axis) for axis in range(3))
        first = _dyadic_multiply(
            ab[0],
            _dyadic_subtract(
                _dyadic_multiply(ac[1], ad[2]), _dyadic_multiply(ac[2], ad[1])
            ),
        )
        second = _dyadic_multiply(
            ab[1],
            _dyadic_subtract(
                _dyadic_multiply(ac[0], ad[2]), _dyadic_multiply(ac[2], ad[0])
            ),
        )
        third = _dyadic_multiply(
            ab[2],
            _dyadic_subtract(
                _dyadic_multiply(ac[0], ad[1]), _dyadic_multiply(ac[1], ad[0])
            ),
        )
        return _dyadic_sign(_dyadic_add(_dyadic_subtract(first, second), third))

    x = tuple(difference(row_, spec.arity - 1, 0) for row_ in range(spec.arity - 1))
    y = tuple(difference(row_, spec.arity - 1, 1) for row_ in range(spec.arity - 1))
    products = {
        (first, second): _dyadic_multiply(x[first], y[second])
        for first in range(spec.arity - 1)
        for second in range(spec.arity - 1)
        if first != second
    }

    def cross(first: int, second: int, /) -> tuple[int, int]:
        return _dyadic_subtract(products[first, second], products[second, first])

    if spec.name == "incircle":
        lift = tuple(
            _dyadic_add(
                _dyadic_multiply(x[index], x[index]), _dyadic_multiply(y[index], y[index])
            )
            for index in range(3)
        )
        determinant = _dyadic_add(
            _dyadic_add(
                _dyadic_multiply(lift[0], cross(1, 2)),
                _dyadic_multiply(lift[1], cross(2, 0)),
            ),
            _dyadic_multiply(lift[2], cross(0, 1)),
        )
        return _dyadic_sign(determinant)
    if spec.name != "insphere":
        raise ValueError(
            f"_exact_dyadic_row.spec.name must name a supported exact predicate; got {spec.name!r}"
        )
    z = tuple(difference(row_, spec.arity - 1, 2) for row_ in range(spec.arity - 1))
    ab, bc, cd = cross(0, 1), cross(1, 2), cross(2, 3)
    da, ac, bd = cross(3, 0), cross(0, 2), cross(1, 3)
    abc = _dyadic_add(
        _dyadic_subtract(_dyadic_multiply(z[0], bc), _dyadic_multiply(z[1], ac)),
        _dyadic_multiply(z[2], ab),
    )
    bcd = _dyadic_add(
        _dyadic_subtract(_dyadic_multiply(z[1], cd), _dyadic_multiply(z[2], bd)),
        _dyadic_multiply(z[3], bc),
    )
    cda = _dyadic_add(
        _dyadic_add(_dyadic_multiply(z[2], da), _dyadic_multiply(z[3], ac)),
        _dyadic_multiply(z[0], cd),
    )
    dab = _dyadic_add(
        _dyadic_add(_dyadic_multiply(z[3], ab), _dyadic_multiply(z[0], bd)),
        _dyadic_multiply(z[1], da),
    )
    lift = tuple(
        _dyadic_add(
            _dyadic_add(
                _dyadic_multiply(x[index], x[index]), _dyadic_multiply(y[index], y[index])
            ),
            _dyadic_multiply(z[index], z[index]),
        )
        for index in range(4)
    )
    shewchuk = _dyadic_add(
        _dyadic_subtract(_dyadic_multiply(lift[3], abc), _dyadic_multiply(lift[2], dab)),
        _dyadic_subtract(_dyadic_multiply(lift[1], cda), _dyadic_multiply(lift[0], bcd)),
    )
    return -_dyadic_sign(shewchuk)


def _exact_dyadic(
    spec: _Filter, points: tuple[np.ndarray, ...], unresolved: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    signs = np.full((unresolved.size,), PredicateSign.UNCERTAIN, dtype=np.int8)
    certain = np.zeros((unresolved.size,), dtype=np.bool_)
    for output, row in enumerate(unresolved.tolist()):
        if all(np.all(np.isfinite(point[row])) for point in points):
            signs[output] = _exact_dyadic_row(spec, points, row)
            certain[output] = True
    return signs, certain


def bigint_bytes(bits: int, /) -> int:
    """Conservative CPython storage of one integer of ``bits`` bits and its slot.

    A PyLong stores 30-bit digits after a header below 32 bytes; an object
    array or tuple slot adds one 8-byte reference.
    """
    return 40 + 4 * ((max(bits, 1) + 29) // 30)


def exact_fraction_rows_key(
    rows: object,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    extra_storage: int = 0,
) -> bytes | None:
    """Pack complete exact row values, never rounded coordinates or a digest.

    Row lengths and integer lengths delimit signed numerators and positive
    denominators. Unsupported scalar forms decline this numerical cache route.
    The caller owns any scientific namespace and the returned key's lifetime.
    """
    extra_storage = nonnegative_integer(extra_storage, "exact_fraction_rows_key.extra_storage")
    if type(rows) is not tuple:
        return None
    size = 8
    for row in rows:
        budget.reserve(1)
        if type(row) is not tuple:
            return None
        size += 8
        for value in row:
            budget.reserve(1)
            if (
                type(value) is not Fraction
                or type(value.numerator) is not int
                or type(value.denominator) is not int
                or value.denominator <= 0
            ):
                return None
            size += (
                16
                + max(1, (value.numerator.bit_length() + 8) // 8)
                + max(1, (value.denominator.bit_length() + 7) // 8)
            )
    # The complete immutable row/scalar validation above establishes this shape.
    canonical = cast(tuple[tuple[Fraction, ...], ...], rows)
    budget.reserve(0, 1024 + 3 * size + extra_storage)
    payload = bytearray()
    payload.extend(len(canonical).to_bytes(8, "big"))
    for row in canonical:
        budget.reserve(1)
        payload.extend(len(row).to_bytes(8, "big"))
        for value in row:
            budget.reserve(1)
            numerator, denominator = value.numerator, value.denominator
            width = max(1, (numerator.bit_length() + 8) // 8)
            payload.extend(width.to_bytes(8, "big"))
            payload.extend(numerator.to_bytes(width, "big", signed=True))
            width = max(1, (denominator.bit_length() + 7) // 8)
            payload.extend(width.to_bytes(8, "big"))
            payload.extend(denominator.to_bytes(width, "big"))
    return bytes(payload)


def exact_bits(values: np.ndarray, /, *, budget: CoordinateEnclosureBudget) -> int:
    """Return the largest numerator or denominator bit length of an integer or rational bank.

    Python integers and Fractions share ``numerator``/``denominator``; the
    one-visit-per-entry scan is charged before it runs.
    """
    with exact_charge(values.size, budget=budget):
        return max(
            (
                max(value.numerator.bit_length(), value.denominator.bit_length())
                for value in values.flat
            ),
            default=0,
        )


def exact_orientation_preparation_bounds(
    row_count: int, dimension: int, integer_bits: int, /
) -> tuple[int, int, int]:
    """Bound charged integer orientation preparation from row and bit metadata."""
    from ._host_storage import declared_integer_storage_bytes

    if row_count < 0 or integer_bits < 0 or dimension not in (2, 3):
        raise ValueError(
            "exact_orientation_preparation_bounds.metadata must have nonnegative rows/bits and dimension 2 or 3; "
            f"got rows={row_count}, bits={integer_bits}, dimension={dimension}"
        )
    if row_count == 0:
        return 0, 0, 0
    bits = max(integer_bits, 1)
    points = dimension + 1
    normal_bits = (dimension - 1) * (bits + 1) + 1
    offset_bits = dimension * (bits + 1) + dimension.bit_length()
    profile_bits = max(normal_bits, offset_bits)
    temporary_bits = (48 if dimension == 3 else 16) * bits + 64
    result_bits = (dimension + 1) * (profile_bits + bits + 1) + dimension.bit_length() + 4
    # Outer/key/point/preparation/normal tuples and the eight-character
    # predicate name use _host_storage's 256-byte/16-byte-slot convention.
    graph_bytes = (
        256
        + 16 * 2
        + 256
        + 16 * points
        + dimension * (256 + 16 * dimension)
        + 256
        + 4 * 8
        + 256
        + 16 * 3
        + 256
        + 16 * dimension
    )
    cache_bytes = (
        graph_bytes
        + dimension**2 * declared_integer_storage_bytes(bits)
        + dimension * declared_integer_storage_bytes(normal_bits)
        + declared_integer_storage_bytes(offset_bits)
        + declared_integer_storage_bytes(profile_bits.bit_length())
    )
    retention_visits = 7 + dimension**2 + 2 * dimension
    visits = (
        points * dimension
        + points * dimension
        + dimension**2
        + 1
        + dimension**2
        + (20 if dimension == 3 else 5)
        + dimension
        + 1
        + retention_visits
        + dimension
        + 2 * dimension
        + 2
    )
    scratch = (
        128
        + 64 * points
        + max(
            512 + 24 * (128 + 2 * bigint_bytes(temporary_bits)),
            (dimension + 2) * (128 + 2 * bigint_bytes(result_bits)),
        )
    )
    # Ordered-input signs coexist with their shared line/plane profile.
    cache_bytes += (
        768
        + 16 * (points + dimension + 4)
        + dimension * declared_integer_storage_bytes(bits)
    )
    visits += 6 + 2 * dimension + points
    return row_count * visits, row_count * cache_bytes + row_count + scratch, 0


def polygon_simplicity_preparation_bounds(
    polygon_count: int,
    corner_count: int,
    /,
    *,
    integer_bits: int = 1075,
    maximum_candidate_pairs: int = 5_000_000,
) -> tuple[int, int, int]:
    """Bound the polygon edge-pair and extreme-turn orientation circuit."""
    if polygon_count < 0 or corner_count < 3 or maximum_candidate_pairs <= 0:
        raise ValueError(
            "polygon_simplicity_preparation_bounds.metadata must have nonnegative polygons, at least three corners, and positive pair capacity; "
            f"got polygons={polygon_count}, corners={corner_count}, capacity={maximum_candidate_pairs}"
        )
    pairs = min(
        maximum_candidate_pairs,
        polygon_count * corner_count * (corner_count - 1) // 2,
    )
    # Each edge-pair classification evaluates four orientations; the extreme
    # turn adds one per polygon. Native predicate queries are telemetry only.
    work, storage, queries = exact_orientation_preparation_bounds(
        4 * pairs + polygon_count, 2, integer_bits
    )
    return work + 2 * polygon_count * corner_count, storage, queries


def exact_reserve(
    work: int,
    storage: int = 0,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Count integer-arithmetic visits and storage retained by the supplied request."""
    budget.reserve(work, storage)


@contextmanager
def exact_charge(
    work: int,
    storage: int = 0,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Iterator[None]:
    """Count integer-arithmetic visits and transient bytes, releasing the transient bound."""
    with budget.temporary_scope():
        budget.reserve(work, storage)
        yield


def _prepared_exact_orientation_signs(
    spec: _Filter,
    points: tuple[np.ndarray, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Reuse immutable line/plane preparation in the supplied request."""
    dimension = spec.width
    count = points[0].shape[0]
    with exact_charge(0, count, budget=budget):
        signs = np.empty(count, dtype=np.int8)
    cache = budget.exact_orientation_preparation_cache
    sign_cache = budget.exact_orientation_sign_cache
    evaluations = budget.exact_predicate_evaluations
    hits = budget.exact_predicate_cache_hits
    for row in range(count):
        with exact_charge(
            len(points) * dimension + 1,
            128 + 64 * len(points),
            budget=budget,
        ):
            row_points = tuple(tuple(point[row]) for point in points)
            sign_key = (spec.name, *row_points)
            known = sign_cache.get(sign_key)
            if known is not None:
                signs[row] = known
                hits[spec.name] = hits.get(spec.name, 0) + 1
                continue
            key = sign_key[:-1]
            prepared = cache.get(key)
            if prepared is None:
                coordinates = row_points[:-1]
                values = tuple(value for point in coordinates for value in point)
                with exact_charge(len(values), budget=budget):
                    bits = max(
                        (
                            max(
                                value.numerator.bit_length(),
                                value.denominator.bit_length(),
                            )
                            for value in values
                        ),
                        default=0,
                    )
                # Rational differences/products may multiply independent source
                # denominators; integer inputs are the smaller special case.
                temporary_bits = (48 if dimension == 3 else 16) * max(bits, 1) + 64
                normal: tuple[int | Fraction, ...]
                with exact_charge(
                    20 if dimension == 3 else 5,
                    512 + 24 * (128 + 2 * bigint_bytes(temporary_bits)),
                    budget=budget,
                ):
                    if spec.name == "orient3d":
                        a, b, c = coordinates
                        first = tuple(q - p for p, q in zip(a, b, strict=True))
                        second = tuple(q - p for p, q in zip(a, c, strict=True))
                        normal = (
                            first[1] * second[2] - first[2] * second[1],
                            first[2] * second[0] - first[0] * second[2],
                            first[0] * second[1] - first[1] * second[0],
                        )
                        offset = normal[0] * a[0] + normal[1] * a[1] + normal[2] * a[2]
                    else:
                        a, b = coordinates
                        normal = (a[1] - b[1], b[0] - a[0])
                        offset = normal[0] * a[0] + normal[1] * a[1]
                    with exact_charge(dimension + 1, budget=budget):
                        normal_bits = max(
                            (
                                max(
                                    value.numerator.bit_length(),
                                    value.denominator.bit_length(),
                                )
                                for value in (*normal, offset)
                            ),
                            default=0,
                        )
                    prepared = (normal, offset, normal_bits)
                budget.retain_basis((key, prepared))
                cache[key] = prepared
        normal, offset, normal_bits = prepared
        with exact_charge(dimension, budget=budget):
            point = tuple(points[-1][row])
            point_bits = max(
                (
                    max(value.numerator.bit_length(), value.denominator.bit_length())
                    for value in point
                ),
                default=0,
            )
        bits = (
            (dimension + 1) * (normal_bits + point_bits + 1) + dimension.bit_length() + 4
        )
        with exact_charge(
            2 * dimension,
            (dimension + 2) * (128 + 2 * bigint_bytes(bits)),
            budget=budget,
        ):
            if len(normal) != dimension or len(point) != dimension:
                raise RuntimeError(
                    f"_prepared_exact_orientation_signs.dimension must match prepared normal and point lengths; got dimension={dimension}, normal={len(normal)}, point={len(point)}"
                )
            value = (
                sum(
                    (
                        coefficient * coordinate
                        for coefficient, coordinate in zip(normal, point, strict=True)
                    ),
                    Fraction(),
                )
                - offset
            )
            with exact_charge(2, budget=budget):
                signs[row] = int(value > 0) - int(value < 0)
            budget.retain_basis((sign_key, int(signs[row])))
            sign_cache[sign_key] = int(signs[row])
            evaluations[spec.name] = evaluations.get(spec.name, 0) + 1
    return signs


def _exact_integer(
    spec: _Filter,
    points: tuple[np.ndarray, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Evaluate ordered orientations with integer arithmetic in the supplied request."""
    if spec.name not in ("orient2d", "orient3d"):
        raise TypeError(
            f"_exact_integer.spec.name must be orient2d or orient3d; got {spec.name!r}"
        )
    return _prepared_exact_orientation_signs(spec, points, budget=budget)


def _host_signs(
    spec: _Filter,
    values: tuple[np.ndarray, ...],
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    native_exact = mode is PredicateMode.EXACT and meshcore_available()
    if native_exact:
        load_meshcore()
    flat, leading = _host_points(values, spec.width, budget=budget)
    return _host_signs_from_points(
        spec, flat, leading, mode, native_exact=native_exact, budget=budget
    )


def _memo_native_signs(
    spec: _Filter,
    flat: tuple[np.ndarray, ...],
    unresolved: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Evaluate each unresolved binary64 input once within the supplied request."""
    cache = budget.exact_native_sign_cache
    evaluations = budget.exact_predicate_evaluations
    hits = budget.exact_predicate_cache_hits
    keys: list[tuple[str, bytes]] = []
    missing: dict[tuple[str, bytes], int] = {}
    with budget.temporary_scope():
        budget.reserve(
            unresolved.size * spec.arity * spec.width,
            512 + unresolved.size * (512 + 8 * spec.arity * spec.width),
        )
        for row in unresolved:
            key = (spec.name, b"".join(point[row].tobytes() for point in flat))
            keys.append(key)
            if key in cache or key in missing:
                hits[spec.name] = hits.get(spec.name, 0) + 1
            else:
                missing[key] = int(row)
        if missing:
            rows = np.fromiter(missing.values(), dtype=np.int64, count=len(missing))

            with (
                budget.ticket.native_target()
                if budget.ticket.native_accounting
                else nullcontext()
            ):
                signs = _EXACT_ROUTES[spec.name](*(point[rows] for point in flat))
            for key, sign in zip(missing, signs, strict=True):
                value = int(sign)
                budget.retain_basis((key, value))
                cache[key] = value
            evaluations[spec.name] = evaluations.get(spec.name, 0) + len(missing)
        return np.fromiter((cache[key] for key in keys), dtype=np.int8, count=len(keys))


def _host_signs_from_points(
    spec: _Filter,
    flat: tuple[np.ndarray, ...],
    leading: tuple[int, ...],
    mode: PredicateMode,
    /,
    *,
    native_exact: bool,
    memoize: bool = True,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    if flat[0].dtype == object:
        return _exact_integer(spec, flat, budget=budget).reshape(leading), np.ones(
            leading, dtype=np.bool_
        )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        signs, certain = _filter(np, spec, flat, np.float64, 1.0)
    if mode is PredicateMode.EXACT:
        unresolved = np.flatnonzero(~certain)
        if unresolved.size and native_exact:
            # Standalone binary predicates admit no host work or retained storage.
            # Their native evaluation remains valid without a request-owned memo.
            if memoize and budget.maximum_work_units and budget.maximum_memory_bytes:
                signs[unresolved] = _memo_native_signs(
                    spec, flat, unresolved, budget=budget
                )
            else:
                with budget.temporary_scope():
                    if budget.maximum_memory_bytes:
                        budget.reserve(
                            0, 512 + unresolved.size * (8 * spec.arity * spec.width + 1)
                        )
                    with (
                        budget.ticket.native_target()
                        if budget.ticket.native_accounting
                        else nullcontext()
                    ):
                        signs[unresolved] = _EXACT_ROUTES[spec.name](
                            *(array[unresolved] for array in flat)
                        )
            certain[unresolved] = True
        elif unresolved.size:
            exact_signs, exact_certain = _exact_dyadic(spec, flat, unresolved)
            signs[unresolved] = exact_signs
            certain[unresolved] = exact_certain
    return signs.reshape(leading), certain.reshape(leading)


@contextmanager
def _predicate_owner(
    work_upper: int,
    bytes_upper: int,
    /,
    *,
    completed_metadata_work: int = 0,
) -> Iterator[CoordinateEnclosureBudget]:
    # Integer source preparation imports predicates; create the domain view only
    # after those modules have finished their mutual initialization.
    from .discretization._coordinate_budget import CoordinateEnclosureBudget

    ticket = HostBatchTicket(work_upper, bytes_upper=bytes_upper)
    budget = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=work_upper,
        maximum_memory_bytes=bytes_upper,
    )
    with ticket, budget.activate():
        budget.reserve(completed_metadata_work)
        yield budget


def _integer_input_bits(points: tuple[np.ndarray, ...], /) -> int:
    return max(
        (value.bit_length() for point in points for value in point.flat),
        default=0,
    )


def _host(
    spec: _Filter,
    values: tuple,
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
    memoize_native: bool = True,
) -> HostPredicateResult:
    native_exact = mode is PredicateMode.EXACT and meshcore_available()
    if native_exact:
        load_meshcore()
    hosts = tuple(np.asarray(value) for value in values)
    flat, leading = _host_points(hosts, spec.width, budget=budget)
    if budget is None:
        work_upper = bytes_upper = metadata_work = 0
        if flat[0].dtype == object:
            work_upper, bytes_upper, _ = exact_orientation_preparation_bounds(
                flat[0].shape[0], spec.width, _integer_input_bits(flat)
            )
            validation = sum(value.size for value in hosts)
            metadata_work = validation + sum(point.size for point in flat)
            work_upper += max(0, validation - flat[0].shape[0] * len(flat) * spec.width)
        with _predicate_owner(
            work_upper, bytes_upper, completed_metadata_work=metadata_work
        ) as owner:
            signs, certain = _host_signs_from_points(
                spec,
                flat,
                leading,
                mode,
                native_exact=native_exact,
                memoize=False,
                budget=owner,
            )
    else:
        signs, certain = _host_signs_from_points(
            spec, flat, leading, mode, native_exact=native_exact,
            memoize=memoize_native, budget=budget,
        )
    return HostPredicateResult(signs, certain, mode)


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
                    "_device_points.coordinates.dtype must preserve input precision; "
                    f"got source={source}, device={array.dtype}; enable jax_enable_x64 or pass the device dtype"
                )
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError(
                f"_device_points.coordinates.dtype must be real floating; got {array.dtype}"
            )
        if array.ndim < 1 or array.shape[-1] != width:
            raise ValueError(
                f"_device_points.coordinates.shape must be (..., {width}); got {array.shape}"
            )
        arrays.append(array)
    dtype = jnp.result_type(*arrays)
    if np.dtype(dtype) not in _SUBNORMAL_MASKS:
        raise TypeError(
            f"_device_points.coordinates.dtype must be float32 or float64; got {dtype}"
        )
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


def _evaluate(
    spec: _Filter,
    values: tuple,
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> PredicateResult:
    match mode:
        case PredicateMode.FILTERED | PredicateMode.EXACT:
            host = _host(spec, values, mode, budget=budget)
            return PredicateResult(
                jax.device_put(host.signs), jax.device_put(host.certain), host.mode
            )
        case PredicateMode.FILTERED_DEVICE:
            return _device(spec, values)
        case _:
            raise TypeError(
                f"_evaluate.mode must be a PredicateMode; got {type(mode).__name__}"
            )


@checked
def orient2d_host(
    a: ConvertibleToArray,
    b: ConvertibleToArray,
    c: ConvertibleToArray,
    /,
    *,
    mode: PredicateMode,
    budget: CoordinateEnclosureBudget | None = None,
    memoize_native: bool = True,
) -> HostPredicateResult:
    """Return certified host signs of ``det[b - a, c - a]``.

    ``memoize_native=False`` executes each filtered-uncertain binary64 row
    directly, without retaining coordinate-byte/sign keys. Native counter
    prices and exact-integer operand preparation are unchanged.
    """
    mode = parse(mode, PredicateMode, "orient2d_host.mode")
    resolve_host_predicate_mode(mode)
    return _host(
        _ORIENT2D, (a, b, c), mode, budget=budget,
        memoize_native=memoize_native,
    )


@checked
def orient3d_host(
    a: ConvertibleToArray,
    b: ConvertibleToArray,
    c: ConvertibleToArray,
    d: ConvertibleToArray,
    /,
    *,
    mode: PredicateMode,
    budget: CoordinateEnclosureBudget | None = None,
    memoize_native: bool = True,
) -> HostPredicateResult:
    """Return certified host signs of ``det[b - a, c - a, d - a]``.

    ``memoize_native=False`` executes every filtered-uncertain binary64 row
    directly, without building retained coordinate-byte/sign keys. The native
    work/query ticket and exact-integer prepared operands are unchanged.
    Primitive telemetry remains distinct from construction work/source queries
    under the original native counter contract.
    """
    mode = parse(mode, PredicateMode, "orient3d_host.mode")
    resolve_host_predicate_mode(mode)
    return _host(
        _ORIENT3D, (a, b, c, d), mode, budget=budget,
        memoize_native=memoize_native,
    )


def _orientation_enclosure_host(
    spec: _Filter,
    values: tuple[np.ndarray, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose the exact binary64 orientation with the existing static filter.

    This is the determinant enclosure underlying the host orientation's certified
    sign, not an enclosure of a rounded or sampled source. Callers lifting an
    exact source bank must first prove equality with its binary64 view.
    Nonfinite filter operands return an unbounded interval and require the
    caller's exact route. One filtered predicate visit is charged per row.
    """
    points, leading = _host_points(values, spec.width, budget=budget)
    if points[0].dtype == object:
        raise TypeError("orientation_enclosure_host requires binary64 predicate points")
    count = points[0].shape[0]
    budget.reserve(count, 1536 * count)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        determinant, permanent, zero, largest = spec.kernel(np, *points)
        info = np.finfo(np.float64)
        unit = float(info.eps) / 2.0
        first, second = spec.coefficient
        relative = np.nextafter((first + second * unit) * unit, np.inf)
        absolute = np.nextafter(
            spec.multiplier * float(info.tiny) * (1.0 + largest) ** (spec.degree - 1),
            np.inf,
        )
        bound = np.nextafter(
            np.nextafter(relative * permanent, np.inf) + absolute, np.inf
        )
        lower = np.nextafter(determinant - bound, -np.inf)
        upper = np.nextafter(determinant + bound, np.inf)
    finite = np.isfinite(determinant) & np.isfinite(bound)
    lower = np.where(finite, np.where(zero, 0.0, lower), -np.inf)
    upper = np.where(finite, np.where(zero, 0.0, upper), np.inf)
    return lower.reshape(leading), upper.reshape(leading)


def orient2d_enclosure_host(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose the exact binary64 determinant using the planar static filter."""
    return _orientation_enclosure_host(_ORIENT2D, (a, b, c), budget=budget)


def orient3d_enclosure_host(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose the exact binary64 determinant using the spatial static filter."""
    return _orientation_enclosure_host(_ORIENT3D, (a, b, c, d), budget=budget)


def orient2d(
    a: ConvertibleToArray,
    b: ConvertibleToArray,
    c: ConvertibleToArray,
    /,
    *,
    mode: PredicateMode,
    budget: CoordinateEnclosureBudget | None = None,
) -> PredicateResult:
    """Sign of ``det[b - a, c - a]`` for ``(..., 2)`` coordinates (leading axes broadcast).

    Host coordinates are real floating, fixed-width integer (at most ``2**53``
    in magnitude) or, together, object arrays of Python integers, which are
    evaluated exactly with certain signs. Any other entry kind raises
    ``TypeError``.
    """

    return _evaluate(_ORIENT2D, (a, b, c), mode, budget=budget)


def orient3d(
    a: ConvertibleToArray,
    b: ConvertibleToArray,
    c: ConvertibleToArray,
    d: ConvertibleToArray,
    /,
    *,
    mode: PredicateMode,
    budget: CoordinateEnclosureBudget | None = None,
) -> PredicateResult:
    """Sign of ``det[b - a, c - a, d - a]`` for ``(..., 3)`` coordinates.

    Accepts the same coordinate kinds as :func:`orient2d`.
    """

    return _evaluate(_ORIENT3D, (a, b, c, d), mode, budget=budget)


def incircle(
    a: Any,
    b: Any,
    c: Any,
    d: Any,
    /,
    *,
    mode: PredicateMode,
    budget: CoordinateEnclosureBudget | None = None,
) -> PredicateResult:
    """Positive when ``d`` lies inside the circle through counterclockwise ``(a, b, c)``."""

    return _evaluate(_INCIRCLE, (a, b, c, d), mode, budget=budget)


def insphere(
    a: Any,
    b: Any,
    c: Any,
    d: Any,
    e: Any,
    /,
    *,
    mode: PredicateMode,
    budget: CoordinateEnclosureBudget | None = None,
) -> PredicateResult:
    """Positive when ``e`` lies inside the sphere through positively oriented ``(a..d)``."""

    return _evaluate(_INSPHERE, (a, b, c, d, e), mode, budget=budget)


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
            raise TypeError(
                f"SegmentIntersectionResult.mode must be a PredicateMode; got {type(mode).__name__}"
            )
        if status.dtype != np.int8:
            raise TypeError(
                f"SegmentIntersectionResult.status.dtype must be int8; got {status.dtype}"
            )
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

    status: Array | np.ndarray
    orientation: Array | np.ndarray
    intersecting_pairs: Array | np.ndarray
    uncertain_pairs: Array | np.ndarray
    candidate_pair_count: int = eqx.field(static=True)
    candidate_capacity_exceeded: bool = eqx.field(static=True)
    mode: PredicateMode = eqx.field(static=True)

    def __init__(
        self,
        status: Array | np.ndarray,
        orientation: Array | np.ndarray,
        intersecting_pairs: Array | np.ndarray,
        uncertain_pairs: Array | np.ndarray,
        mode: PredicateMode,
        *,
        candidate_pair_count: int,
        candidate_capacity_exceeded: bool,
    ) -> None:
        if not isinstance(mode, PredicateMode):
            raise TypeError(
                f"PolygonSimplicityResult.mode must be a PredicateMode; got {type(mode).__name__}"
            )
        if status.dtype != np.int8 or orientation.dtype != np.int8:
            raise TypeError(
                f"PolygonSimplicityResult.status.dtype must be int8 along with orientation.dtype; got status={status.dtype}, orientation={orientation.dtype}"
            )
        if intersecting_pairs.dtype != np.int64 or uncertain_pairs.dtype != np.int64:
            raise TypeError(
                f"PolygonSimplicityResult.pair_counts.dtype must be int64; got intersecting={intersecting_pairs.dtype}, uncertain={uncertain_pairs.dtype}"
            )
        if not (
            status.shape
            == orientation.shape
            == intersecting_pairs.shape
            == uncertain_pairs.shape
        ):
            raise ValueError(
                f"PolygonSimplicityResult.arrays.shape must agree; got status={status.shape}, orientation={orientation.shape}, intersecting={intersecting_pairs.shape}, uncertain={uncertain_pairs.shape}"
            )
        count = int(candidate_pair_count)
        if count < 0:
            raise ValueError(
                f"PolygonSimplicityResult.candidate_pair_count must be non-negative; got {count}"
            )
        if not isinstance(candidate_capacity_exceeded, (bool, np.bool_)):
            raise TypeError(
                f"PolygonSimplicityResult.candidate_capacity_exceeded must be bool; got {type(candidate_capacity_exceeded).__name__}"
            )
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
                f"_require_host_mode.mode must be FILTERED or EXACT; got {mode!r}"
            )
        case _:
            raise TypeError(
                f"_require_host_mode.mode must be a PredicateMode; got {type(mode).__name__}"
            )


def _classify_segments(
    a: Any,
    b: Any,
    c: Any,
    d: Any,
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Contact classes of closed segments ``ab`` and ``cd`` of shape ``(..., 2)``."""

    signs_, certain = _host_signs(
        _ORIENT2D,
        (
            np.stack((a, a, c, c)),
            np.stack((b, b, d, d)),
            np.stack((c, d, a, b)),
        ),
        mode,
        budget=budget,
    )
    signs = signs_.astype(np.int16)
    first = signs[0] * signs[1]
    second = signs[2] * signs[3]
    # One certified supporting line with both ends of the other segment strictly
    # on one side separates the segments whatever the remaining signs are.
    separated = (certain[0] & certain[1] & (first > 0)) | (
        certain[2] & certain[3] & (second > 0)
    )
    # Coordinate comparisons introduce no rounding; for collinear segments the common box is
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
    a: ConvertibleToArray,
    b: ConvertibleToArray,
    c: ConvertibleToArray,
    d: ConvertibleToArray,
    /,
    *,
    mode: PredicateMode,
    budget: CoordinateEnclosureBudget | None = None,
) -> SegmentIntersectionResult:
    """Classify the contact of closed segments ``ab`` and ``cd`` (``(..., 2)``, broadcast).

    Host-only (``FILTERED`` or ``EXACT``); unresolved filtered signs yield
    ``UNCERTAIN`` unless a coordinate comparison or certified sign test separates
    the segments. Object arrays of Python integers are classified exactly.
    """

    _require_host_mode(mode)
    hosts = tuple(np.asarray(value) for value in (a, b, c, d))
    flat, leading = _host_points(hosts, 2, budget=budget)
    if budget is None:
        work_upper = bytes_upper = metadata_work = 0
        if flat[0].dtype == object:
            count = flat[0].shape[0]
            work_upper, bytes_upper, _ = exact_orientation_preparation_bounds(
                4 * count, 2, _integer_input_bits(flat)
            )
            validation = sum(value.size for value in hosts)
            metadata_work = validation + sum(point.size for point in flat)
            work_upper += 16 * count + validation
            bytes_upper += 8 * 36 * count
        with _predicate_owner(
            work_upper, bytes_upper, completed_metadata_work=metadata_work
        ) as owner:
            status = _segment_status(flat, mode, budget=owner)
    else:
        status = _segment_status(flat, mode, budget=budget)
    return SegmentIntersectionResult(status.reshape(leading), mode)


def _segment_status(
    flat: tuple[np.ndarray, ...],
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    if flat[0].dtype == object:
        with exact_charge(
            16 * flat[0].shape[0], 8 * 36 * flat[0].shape[0], budget=budget
        ):
            return _classify_segments(
                flat[0], flat[1], flat[2], flat[3], mode, budget=budget
            )
    return _classify_segments(flat[0], flat[1], flat[2], flat[3], mode, budget=budget)


def _edge_pair_evidence(
    points: np.ndarray,
    mode: PredicateMode,
    maximum_candidate_pairs: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Bounded offending and unresolved edge-pair evidence."""

    polygon_count, corner_count, _ = points.shape
    intersecting = np.zeros((polygon_count,), dtype=np.int64)
    uncertain = np.zeros((polygon_count,), dtype=np.int64)
    starts = points.reshape((-1, 2))
    stops = np.roll(points, -1, axis=1).reshape((-1, 2))
    # A polygon-index coordinate axis keeps the broad phase within polygons.
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
                starts[first],
                stops[first],
                starts[second],
                stops[second],
                mode,
                budget=budget,
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


def _extreme_turn(
    points: np.ndarray,
    mode: PredicateMode,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PredicateResult:
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
        budget=budget,
    )


def polygon_simplicity_2d(
    vertices: Any,
    /,
    *,
    mode: PredicateMode,
    maximum_candidate_pairs: int = 5_000_000,
    budget: CoordinateEnclosureBudget | None = None,
) -> PolygonSimplicityResult:
    """Certify that closed vertex loops ``(..., n, 2)`` bound simple polygons.

    A loop is simple when non-adjacent edges are disjoint and adjacent edges
    meet only at their shared vertex; repeated vertices therefore make a loop
    self-intersecting. Candidate edge pairs come from a bounded BVH block stream
    over edge boxes, partitioned by polygon; exhausting
    ``maximum_candidate_pairs`` leaves every unproved loop ``UNCERTAIN``.
    Host-only (``FILTERED`` or ``EXACT``).
    """

    _require_host_mode(mode)
    if isinstance(maximum_candidate_pairs, bool) or not isinstance(
        maximum_candidate_pairs, (int, np.integer)
    ):
        raise TypeError(
            f"polygon_simplicity_2d.maximum_candidate_pairs must be an integer; got {type(maximum_candidate_pairs).__name__}"
        )
    capacity = int(maximum_candidate_pairs)
    if capacity <= 0:
        raise ValueError(
            f"polygon_simplicity_2d.maximum_candidate_pairs must be positive; got {capacity}"
        )
    array = np.asarray(vertices)
    if array.ndim < 2 or array.shape[-2] < 3:
        raise ValueError(
            f"polygon_simplicity_2d.vertices.shape must be (..., n >= 3, 2); got {array.shape}"
        )
    (flat,), _ = _host_points((array,), 2, budget=budget)
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"polygon_simplicity_2d.vertices must be finite; got {flat}")
    leading = array.shape[:-2]
    points = flat.reshape((-1, array.shape[-2], 2))
    if budget is None:
        work_upper, bytes_upper, _ = polygon_simplicity_preparation_bounds(
            points.shape[0], points.shape[1], maximum_candidate_pairs=capacity
        )
        with _predicate_owner(work_upper, bytes_upper) as owner:
            return _polygon_simplicity(points, leading, mode, capacity, budget=owner)
    return _polygon_simplicity(points, leading, mode, capacity, budget=budget)


def _polygon_simplicity(
    points: np.ndarray,
    leading: tuple[int, ...],
    mode: PredicateMode,
    capacity: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PolygonSimplicityResult:
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
        points, mode, capacity, budget=budget
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
    turn = _extreme_turn(points, mode, budget=budget)
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
    "HostPredicateResult",
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
    "orient2d_host",
    "orient3d",
    "orient3d_host",
    "polygon_simplicity_2d",
    "resolve_host_predicate_mode",
    "segment_intersections_2d",
]
