#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certify closest points on trimmed analytic faces through their exact carriers.

A face lies on its untrimmed carrier, so the carrier distance is a lower bound
for the face distance. Planes, circular cylinders, cones, spheres and ring tori
have closed-form global carrier minimizers. When that minimizer is unique and
its outward parameter box is certified strictly inside the face trim, it is the
face minimizer and an interior point of the face. Rows with a non-isolated
carrier minimizer (an axis, center or core-circle query), a competing cone
nappe, a minimizer outside the trim, or an undecided trim box are left
unresolved for the general stationary-root route; their carrier distance lower
bound is retained.

All enclosures use outward-rounded binary64 interval arithmetic. Curved
carriers are admitted only for exactly orthonormal binary64 frames, so the frame
coordinates of a query are exact orthonormal coordinates of the authored
surface. Elementary functions carry the interval implementation's four-ulp
error allowance.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from ..._meshcore import NativeBatchTicket
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from .._interval_enclosure import (
    Interval,
    interval_add as _host_add,
    interval_divide as _host_divide,
    interval_multiply as _host_multiply,
    interval_subtract as _host_subtract,
    interval_tan as _host_tan,
    monotone,
)
from .._rational import fraction_endpoint, rational_dot, rational_vector
from ._analytic_source_regions import AnalyticClosestBatch
from ._intersection import NativePeriodEndpoint
from ._intersection_curve import affine_curve_coefficients, pcurve_periodic_source
from ._model import BRepModel
from ._patches import (
    AbstractCurve,
    ConePatch,
    CylinderPatch,
    PlanePatch,
    SpherePatch,
    TorusPatch,
)
from ._projection_contracts import BRepProjectionStatus
from ._query_types import FaceData, Tolerances
from ._request_preparations import TrimRequestPreparations


type _Interval = tuple[np.ndarray, np.ndarray]
type _Carrier = PlanePatch | CylinderPatch | ConePatch | SpherePatch | TorusPatch

_EPS = float(np.finfo(np.float64).eps)


def _arrays(value: Interval, /) -> _Interval:
    return np.asarray(value[0], dtype=np.float64), np.asarray(value[1], dtype=np.float64)


def interval_add(first: _Interval, second: _Interval, /) -> _Interval:
    return _arrays(_host_add(first, second))


def interval_subtract(first: _Interval, second: _Interval, /) -> _Interval:
    return _arrays(_host_subtract(first, second))


def interval_multiply(first: _Interval, second: _Interval, /) -> _Interval:
    return _arrays(_host_multiply(first, second))


def interval_divide(first: _Interval, second: _Interval, /) -> _Interval:
    return _arrays(_host_divide(first, second))


def _degenerate(value: np.ndarray | float, /) -> _Interval:
    array = np.asarray(value, dtype=np.float64)
    return array, array


# binary64 pi and 2*pi lie below the real constants by less than one ulp.
_PI: _Interval = (np.asarray(math.pi), np.nextafter(np.asarray(math.pi), np.inf))
_TWO_PI: _Interval = (
    np.asarray(2.0 * math.pi),
    np.nextafter(np.asarray(2.0 * math.pi), np.inf),
)
_SQRT = monotone(np.sqrt, True, 1)


def _square(value: _Interval, /) -> _Interval:
    lower, upper = value
    straddles = (lower <= 0.0) & (upper >= 0.0)
    low = np.where(straddles, 0.0, np.minimum(lower * lower, upper * upper))
    high = np.maximum(lower * lower, upper * upper)
    return np.maximum(np.nextafter(low, -np.inf), 0.0), np.nextafter(high, np.inf)


def _sqrt(value: _Interval, /) -> _Interval:
    return _arrays(_SQRT((np.maximum(value[0], 0.0), np.maximum(value[1], 0.0))))


def _tan(value: _Interval, /) -> _Interval:
    return _arrays(_host_tan(value))


def _absolute(value: _Interval, /) -> _Interval:
    lower, upper = value
    straddles = (lower <= 0.0) & (upper >= 0.0)
    low = np.where(straddles, 0.0, np.minimum(np.abs(lower), np.abs(upper)))
    return low, np.maximum(np.abs(lower), np.abs(upper))


def _coordinate(relative: _Interval, axis: np.ndarray, /) -> _Interval:
    """Enclose ``relative . axis`` row-wise for one binary64 axis."""
    total: _Interval = _degenerate(np.zeros(relative[0].shape[0], dtype=np.float64))
    for k in range(3):
        total = interval_add(
            total,
            interval_multiply(
                (relative[0][:, k], relative[1][:, k]), _degenerate(np.float64(axis[k]))
            ),
        )
    return total


def _combination(
    origin: np.ndarray, terms: tuple[tuple[_Interval, np.ndarray], ...], /
) -> _Interval:
    """Enclose ``origin + sum(coefficient * axis)`` as row-wise world boxes."""
    count = terms[0][0][0].shape[0]
    total: _Interval = _degenerate(np.broadcast_to(origin, (count, 3)).copy())
    for coefficient, axis in terms:
        total = interval_add(
            total,
            interval_multiply(
                (coefficient[0][:, None], coefficient[1][:, None]),
                _degenerate(np.broadcast_to(axis, (count, 3))),
            ),
        )
    return total


def _angle(y: _Interval, x: _Interval, /) -> tuple[_Interval, np.ndarray]:
    """Enclose ``atan2(y, x)`` over boxes excluding the origin.

    Along each box edge the polar angle is monotone, so its range is the hull of
    the corner angles taken on one branch. The pad covers the four-ulp
    elementary-function error, the branch shift and its subtraction rounding.
    """
    valid = (x[0] > 0.0) | (x[1] < 0.0) | (y[0] > 0.0) | (y[1] < 0.0)
    with np.errstate(invalid="ignore"):
        centre = np.arctan2(0.5 * (y[0] + y[1]), 0.5 * (x[0] + x[1]))
        corners = np.stack([np.arctan2(row, column) for row in y for column in x], axis=0)
        turns = np.round((corners - centre[None]) / _TWO_PI[0])
        unwrapped = corners - turns * _TWO_PI[0]
    pad = 16.0 * _EPS * (np.abs(unwrapped) + 2.0 * _TWO_PI[1])
    lower = np.min(unwrapped - pad, axis=0)
    upper = np.max(unwrapped + pad, axis=0)
    valid &= np.isfinite(lower) & np.isfinite(upper) & (upper - lower < 0.5 * _PI[0])
    return (lower, upper), valid


def _turn_representative(value: _Interval, /) -> tuple[_Interval, np.ndarray]:
    """Native-gauge representative of an angle box on a whole-turn axis.

    Boxes that may touch the seam are shifted to straddle zero and marked; the
    caller reports their lowest nonnegative representative.
    """
    with np.errstate(invalid="ignore"):
        turns = np.floor(value[0] / _TWO_PI[0])
    turns = np.where(np.isfinite(turns), turns, 0.0)
    shifted = interval_subtract(value, interval_multiply(_degenerate(turns), _TWO_PI))
    upper_seam = shifted[1] >= _TWO_PI[0]
    lowered = interval_subtract(shifted, _TWO_PI)
    shifted = (
        np.where(upper_seam, lowered[0], shifted[0]),
        np.where(upper_seam, lowered[1], shifted[1]),
    )
    return shifted, upper_seam | (shifted[0] <= 0.0)


def _into_range(
    value: _Interval, low: float, high: float, /
) -> tuple[_Interval, np.ndarray, np.ndarray]:
    """Shift an angle box by whole turns toward ``[low, high]``.

    Returns the shifted box, rows whose box lies in the range with both
    neighbouring turns certified outside it, and rows whose box and both
    neighbouring turns are certified disjoint from it.
    """
    with np.errstate(invalid="ignore"):
        turns = np.ceil((low - value[0]) / _TWO_PI[0])
    turns = np.where(np.isfinite(turns), turns, 0.0)
    shifted = interval_add(value, interval_multiply(_degenerate(turns), _TWO_PI))
    previous = interval_subtract(shifted, _TWO_PI)
    following = interval_add(shifted, _TWO_PI)
    if not np.isfinite(low) or not np.isfinite(high) or not high - low < 2.0 * _TWO_PI[0]:
        empty = np.zeros(value[0].shape, dtype=np.bool_)
        return shifted, empty, empty
    neighbours_outside = (previous[1] < low) & (following[0] > high)
    fits = (shifted[0] >= low) & (shifted[1] <= high) & neighbours_outside
    excluded = (shifted[0] > high) & neighbours_outside
    return shifted, fits, excluded


def _exact_orthonormal(*axes: np.ndarray) -> bool:
    exact = tuple(rational_vector(np.asarray(axis, dtype=np.float64)) for axis in axes)
    return all(
        rational_dot(first, second) == Fraction(int(i == j))
        for i, first in enumerate(exact)
        for j, second in enumerate(exact)
    )


def carrier_supported(patch: object, /) -> bool:
    """Whether a face patch admits closed-form certified carrier minimizers."""
    match patch:
        case PlanePatch():
            return type(patch) is PlanePatch
        case CylinderPatch() | ConePatch() | SpherePatch() | TorusPatch():
            if type(patch) not in (CylinderPatch, ConePatch, SpherePatch, TorusPatch):
                return False
            if not _exact_orthonormal(
                np.asarray(patch.first_axis),
                np.asarray(patch.second_axis),
                np.asarray(patch.axis),
            ):
                return False
            if isinstance(patch, TorusPatch):
                # Only ring tori are the exact minor-radius offsets of their core.
                return float(patch.minor_radius) < float(patch.major_radius)
            return True
        case _:
            return False


class _WallRectangle:
    """A trim proved to be one loop of four affine chart walls.

    ``whole`` marks periodic axes whose walls sit at native gauge zero and one
    turn later; their seam lies inside the surface face. ``interior`` is an
    open binary64 interval per bounded axis contained in every wall's extent,
    so each of its points has winding one about the loop however the authored
    wall ends are joined at the corners.
    """

    def __init__(
        self, whole: tuple[bool, bool], interior: tuple[tuple[float, float], ...]
    ) -> None:
        self.whole, self.interior = whole, interior


def _wall_rectangle(
    model: BRepModel, face: int, data: FaceData, /
) -> _WallRectangle | None:
    patch = data.patch
    periods = patch.periods
    if all(period is None for period in periods):
        return None
    geometry = model.require_geometry()
    loops = geometry.face_loops[face]
    if len(loops) != 1 or len(loops[0]) != 4:
        return None
    walls: tuple[list[tuple[Fraction, tuple[Fraction, Fraction] | None]], ...] = (
        [],
        [],
    )
    for coedge in loops[0]:
        if any(
            root is not None and not isinstance(root, NativePeriodEndpoint)
            for root in geometry.coedge_endpoint_roots[coedge]
        ):
            return None
        extracted = pcurve_periodic_source(geometry.pcurves[coedge], patch)
        if extracted is None:
            return None
        base, shifts = extracted
        if not isinstance(base, AbstractCurve):
            return None
        coefficients = affine_curve_coefficients(base, np.zeros((2,), dtype=np.float64))
        if coefficients is None:
            return None
        origin, direction = coefficients
        fixed = [axis for axis in range(2) if direction[axis] == 0]
        if len(fixed) != 1:
            return None
        axis = fixed[0]
        varying = 1 - axis
        if periods[axis] is not None:
            if origin[axis] != 0 or shifts[axis] not in (0, 1):
                return None
            position = Fraction(shifts[axis])
        elif shifts[axis] == 0:
            position = origin[axis]
        else:
            return None
        edge = geometry.coedge_edges[coedge]
        first, last = (float(value) for value in geometry.edge_ranges[edge])
        extent: tuple[Fraction, Fraction] | None
        period = periods[varying]
        if period is not None:
            # Periodic walls span exactly one native turn from gauge zero.
            if (
                direction[varying] != 1
                or shifts[varying] != 0
                or origin[varying] + Fraction(first) != 0
                or last != first + period
            ):
                return None
            extent = None
        elif shifts[varying] != 0:
            return None
        else:
            ends = tuple(
                origin[varying] + direction[varying] * Fraction(value)
                for value in (first, last)
            )
            extent = (min(ends), max(ends))
        walls[axis].append((position, extent))
    if any(len(axis_walls) != 2 for axis_walls in walls):
        return None
    whole = (periods[0] is not None, periods[1] is not None)
    interior: list[tuple[float, float]] = []
    for axis in range(2):
        positions = sorted(position for position, _ in walls[axis])
        if whole[axis]:
            if positions != [0, 1]:
                return None
            interior.append((-math.inf, math.inf))
            continue
        low, high = positions
        for _, extent in walls[1 - axis]:
            if extent is not None:
                low, high = max(low, extent[0]), min(high, extent[1])
        if not low < high:
            return None
        interior.append(
            (
                fraction_endpoint(low, upper=True, saturate=True),
                fraction_endpoint(high, upper=False, saturate=True),
            )
        )
    return _WallRectangle(whole, tuple(interior))


class _CarrierRows:
    """Interval carrier minimizers of one batch, before trim admission."""

    def __init__(
        self,
        u: _Interval,
        v: _Interval,
        point: _Interval,
        distance: _Interval,
        valid: np.ndarray,
    ) -> None:
        self.u, self.v, self.point, self.distance, self.valid = (
            u,
            v,
            point,
            distance,
            valid,
        )


def _frame_coordinates(
    patch: CylinderPatch | ConePatch | SpherePatch | TorusPatch, points: np.ndarray, /
) -> tuple[
    _Interval, _Interval, _Interval, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    origin = np.asarray(
        patch.center if isinstance(patch, (SpherePatch, TorusPatch)) else patch.origin,
        dtype=np.float64,
    )
    first = np.asarray(patch.first_axis, dtype=np.float64)
    second = np.asarray(patch.second_axis, dtype=np.float64)
    axis = np.asarray(patch.axis, dtype=np.float64)
    relative = interval_subtract(
        _degenerate(points), _degenerate(np.broadcast_to(origin, points.shape))
    )
    return (
        _coordinate(relative, first),
        _coordinate(relative, second),
        _coordinate(relative, axis),
        origin,
        first,
        second,
        axis,
    )


def _plane_rows(patch: PlanePatch, points: np.ndarray, /) -> _CarrierRows:
    origin = np.asarray(patch.origin, dtype=np.float64)
    first = np.asarray(patch.first_axis, dtype=np.float64)
    second = np.asarray(patch.second_axis, dtype=np.float64)
    relative = interval_subtract(
        _degenerate(points), _degenerate(np.broadcast_to(origin, points.shape))
    )
    count = points.shape[0]

    def constant(value: float) -> _Interval:
        return _degenerate(np.full((count,), value, dtype=np.float64))

    def dot(a: np.ndarray, b: np.ndarray) -> _Interval:
        total = constant(0.0)
        for k in range(3):
            total = interval_add(
                total, interval_multiply(constant(float(a[k])), constant(float(b[k])))
            )
        return total

    g00, g01, g11 = dot(first, first), dot(first, second), dot(second, second)
    b0, b1 = _coordinate(relative, first), _coordinate(relative, second)
    determinant = interval_subtract(interval_multiply(g00, g11), _square(g01))
    u = interval_divide(
        interval_subtract(interval_multiply(b0, g11), interval_multiply(b1, g01)),
        determinant,
    )
    v = interval_divide(
        interval_subtract(interval_multiply(g00, b1), interval_multiply(g01, b0)),
        determinant,
    )
    normal = tuple(
        interval_subtract(
            interval_multiply(
                constant(float(first[(k + 1) % 3])), constant(float(second[(k + 2) % 3]))
            ),
            interval_multiply(
                constant(float(first[(k + 2) % 3])), constant(float(second[(k + 1) % 3]))
            ),
        )
        for k in range(3)
    )
    offset = constant(0.0)
    length = constant(0.0)
    for k in range(3):
        offset = interval_add(
            offset, interval_multiply((relative[0][:, k], relative[1][:, k]), normal[k])
        )
        length = interval_add(length, _square(normal[k]))
    distance = interval_divide(_absolute(offset), _sqrt(length))
    point = _combination(origin, ((u, first), (v, second)))
    valid = determinant[0] > 0.0
    return _CarrierRows(u, v, point, distance, valid)


def _cylinder_rows(patch: CylinderPatch, points: np.ndarray, /) -> _CarrierRows:
    x, y, z, origin, first, second, axis = _frame_coordinates(patch, points)
    radius = float(patch.radius)
    rho = _sqrt(interval_add(_square(x), _square(y)))
    u, valid = _angle(y, x)
    valid &= rho[0] > 0.0
    distance = _absolute(interval_subtract(rho, _degenerate(np.float64(radius))))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = interval_divide(_degenerate(np.float64(radius)), rho)
    point = _combination(
        origin,
        (
            (interval_multiply(scale, x), first),
            (interval_multiply(scale, y), second),
            (z, axis),
        ),
    )
    return _CarrierRows(u, z, point, distance, valid)


def _sphere_rows(patch: SpherePatch, points: np.ndarray, /) -> _CarrierRows:
    x, y, z, center, first, second, axis = _frame_coordinates(patch, points)
    radius = float(patch.radius)
    planar = interval_add(_square(x), _square(y))
    rho = _sqrt(planar)
    length = _sqrt(interval_add(planar, _square(z)))
    u, valid = _angle(y, x)
    v, valid_v = _angle(z, rho)
    valid &= valid_v & (rho[0] > 0.0)
    distance = _absolute(interval_subtract(length, _degenerate(np.float64(radius))))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = interval_divide(_degenerate(np.float64(radius)), length)
    point = _combination(
        center,
        (
            (interval_multiply(scale, x), first),
            (interval_multiply(scale, y), second),
            (interval_multiply(scale, z), axis),
        ),
    )
    return _CarrierRows(u, v, point, distance, valid)


def _torus_rows(patch: TorusPatch, points: np.ndarray, /) -> _CarrierRows:
    x, y, z, center, first, second, axis = _frame_coordinates(patch, points)
    major, minor = float(patch.major_radius), float(patch.minor_radius)
    rho = _sqrt(interval_add(_square(x), _square(y)))
    radial = interval_subtract(rho, _degenerate(np.float64(major)))
    core = _sqrt(interval_add(_square(radial), _square(z)))
    u, valid = _angle(y, x)
    v, valid_v = _angle(z, radial)
    valid &= valid_v & (rho[0] > 0.0) & (core[0] > 0.0)
    distance = _absolute(interval_subtract(core, _degenerate(np.float64(minor))))
    with np.errstate(divide="ignore", invalid="ignore"):
        tube = interval_divide(_degenerate(np.float64(minor)), core)
        ring = interval_divide(
            interval_add(_degenerate(np.float64(major)), interval_multiply(tube, radial)),
            rho,
        )
    point = _combination(
        center,
        (
            (interval_multiply(ring, x), first),
            (interval_multiply(ring, y), second),
            (interval_multiply(tube, z), axis),
        ),
    )
    return _CarrierRows(u, v, point, distance, valid)


def _cone_rows(patch: ConePatch, points: np.ndarray, /) -> _CarrierRows:
    """Closest generator point in the query's meridian half-plane.

    For a query at radial distance ``rho > 0`` every cone point at another
    meridian is strictly farther than its rotation into the query half-plane,
    where the double cone is two rays from the apex. The distances to both full
    generator lines bound the rays from below; a ray foot with nonzero radius
    that is strictly nearer than the other line is the unique global minimizer.
    """
    x, y, z, origin, first, second, axis = _frame_coordinates(patch, points)
    reference = _degenerate(np.float64(float(patch.reference_radius)))
    angle = float(patch.semi_angle)
    slope = _tan(_degenerate(angle))
    rho = _sqrt(interval_add(_square(x), _square(y)))
    quadratic = interval_add(_degenerate(np.float64(1.0)), _square(slope))
    root = _sqrt(quadratic)
    near = interval_subtract(
        interval_subtract(rho, reference), interval_multiply(slope, z)
    )
    far = interval_add(interval_add(rho, reference), interval_multiply(slope, z))
    near_distance = interval_divide(_absolute(near), root)
    far_distance = interval_divide(_absolute(far), root)
    near_axial = interval_divide(
        interval_add(interval_multiply(slope, interval_subtract(rho, reference)), z),
        quadratic,
    )
    far_axial = interval_divide(
        interval_subtract(z, interval_multiply(slope, interval_add(rho, reference))),
        quadratic,
    )
    near_radius = interval_add(reference, interval_multiply(near_axial, slope))
    far_radius = interval_add(reference, interval_multiply(far_axial, slope))
    angle_box, valid = _angle(y, x)
    valid &= rho[0] > 0.0
    use_near = (near_radius[0] > 0.0) & (near_distance[1] < far_distance[0])
    use_far = (far_radius[1] < 0.0) & (far_distance[1] < near_distance[0])
    valid &= use_near | use_far
    opposite = interval_add(angle_box, _PI)
    u = (
        np.where(use_far, opposite[0], angle_box[0]),
        np.where(use_far, opposite[1], angle_box[1]),
    )
    v = (
        np.where(use_far, far_axial[0], near_axial[0]),
        np.where(use_far, far_axial[1], near_axial[1]),
    )
    # The physical radius |r| always points along the query meridian.
    magnitude = (
        np.where(use_far, -far_radius[1], near_radius[0]),
        np.where(use_far, -far_radius[0], near_radius[1]),
    )
    # Without a selected ray, both full generator lines still bound from below.
    distance = (
        np.where(
            use_far,
            far_distance[0],
            np.where(
                use_near, near_distance[0], np.minimum(near_distance[0], far_distance[0])
            ),
        ),
        np.where(use_far, far_distance[1], near_distance[1]),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = interval_divide(magnitude, rho)
    point = _combination(
        origin,
        (
            (interval_multiply(scale, x), first),
            (interval_multiply(scale, y), second),
            (v, axis),
        ),
    )
    return _CarrierRows(u, v, point, distance, valid)


def _carrier_rows(patch: _Carrier, points: np.ndarray, /) -> _CarrierRows:
    match patch:
        case PlanePatch():
            return _plane_rows(patch, points)
        case CylinderPatch():
            return _cylinder_rows(patch, points)
        case ConePatch():
            return _cone_rows(patch, points)
        case SpherePatch():
            return _sphere_rows(patch, points)
        case TorusPatch():
            return _torus_rows(patch, points)


def _derivatives(
    patch: _Carrier, parameters: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Representative chart derivatives at binary64 parameters."""
    u, v = parameters[:, 0:1], parameters[:, 1:2]
    first = np.asarray(patch.first_axis, dtype=np.float64)[None]
    second = np.asarray(patch.second_axis, dtype=np.float64)[None]
    if isinstance(patch, PlanePatch):
        return np.broadcast_to(
            first, parameters.shape[:1] + (3,)
        ).copy(), np.broadcast_to(second, parameters.shape[:1] + (3,)).copy()
    axis = np.asarray(patch.axis, dtype=np.float64)[None]
    radial = np.cos(u) * first + np.sin(u) * second
    tangent = -np.sin(u) * first + np.cos(u) * second
    match patch:
        case CylinderPatch():
            return float(patch.radius) * tangent, np.broadcast_to(
                axis, radial.shape
            ).copy()
        case ConePatch():
            slope = math.tan(float(patch.semi_angle))
            return (
                float(patch.reference_radius) + v * slope
            ) * tangent, axis + slope * radial
        case SpherePatch():
            radius = float(patch.radius)
            return radius * np.cos(v) * tangent, radius * (
                -np.sin(v) * radial + np.cos(v) * axis
            )
        case TorusPatch():
            major, minor = float(patch.major_radius), float(patch.minor_radius)
            return (major + minor * np.cos(v)) * tangent, minor * (
                -np.sin(v) * radial + np.cos(v) * axis
            )
    raise TypeError(
        f"carrier.patch must be an analytic carrier; got {type(patch).__name__}"
    )


def _midpoint_error(value: _Interval, /) -> tuple[np.ndarray, np.ndarray]:
    """Binary64 midpoints and outward componentwise distances to the box ends."""
    lower, upper = value
    # Unbounded rows yield NaN here and are rejected by the finiteness checks.
    with np.errstate(invalid="ignore"):
        middle = 0.5 * lower + 0.5 * upper
        error = np.maximum(
            np.nextafter(middle - lower, np.inf), np.nextafter(upper - middle, np.inf)
        )
    return middle, error


def _norm_upper(errors: np.ndarray, /) -> np.ndarray:
    squared = np.sum(np.nextafter(errors * errors, np.inf), axis=-1)
    return np.nextafter(
        np.sqrt(np.nextafter(squared * (1.0 + 4.0 * _EPS), np.inf)), np.inf
    )


def carrier_workspace_bytes(count: int, /) -> int:
    """Bound the vectorized interval workspace of one batch of rows."""
    # About 120 live float64 row vectors (interval ends, corner angles, boxes).
    return 1024 * count + 16384


def closest_carrier_face(
    model: BRepModel,
    face: int,
    data: FaceData,
    points: np.ndarray,
    tolerances: Tolerances,
    /,
    *,
    maximum_depth: int,
    preparations: TrimRequestPreparations,
    coordinate_budget: CoordinateEnclosureBudget,
) -> AnalyticClosestBatch | None:
    """Return certified interior closest points of one analytic face, row-wise.

    Rows resolved here have ``UNIQUE`` status, or ``SEAM`` when the minimizer
    may lie on the interior seam of a proven native whole-turn rectangle. Rows
    whose carrier minimizer is certified outside the face, or is undecided by
    the trim, keep their carrier lower bound with an infinite upper bound and no
    point. Rows without an isolated carrier minimizer or within-tolerance
    enclosures are ``unresolved``.
    ``refinements`` counts newly earned trim refinements.
    """
    patch = model.patches[face]
    if not isinstance(
        patch, (PlanePatch, CylinderPatch, ConePatch, SpherePatch, TorusPatch)
    ) or not carrier_supported(patch):
        return None
    count = points.shape[0]
    match coordinate_budget.ticket:
        case NativeBatchTicket() as ticket:
            if count > ticket.remaining_geometry_queries:
                ticket.refuse(0, queries_upper=count)
            ticket.consume(geometry_queries=count)
        case _:
            pass
    rows = _carrier_rows(patch, points)
    lower_parameters = np.asarray(data.lower, dtype=np.float64)
    upper_parameters = np.asarray(data.upper, dtype=np.float64)
    boxes: list[_Interval] = [rows.u, rows.v]
    # Angular chart axes of the analytic carriers; the plane has none.
    angular = (
        (False, False)
        if isinstance(patch, PlanePatch)
        else (True, isinstance(patch, TorusPatch))
    )
    rectangle = (
        None if isinstance(patch, PlanePatch) else _wall_rectangle(model, face, data)
    )
    in_range = np.ones((count,), dtype=np.bool_)
    excluded = np.zeros((count,), dtype=np.bool_)
    interior = np.ones((count,), dtype=np.bool_)
    seams = [np.zeros((count,), dtype=np.bool_), np.zeros((count,), dtype=np.bool_)]
    for axis in range(2):
        if rectangle is not None and rectangle.whole[axis]:
            boxes[axis], seams[axis] = _turn_representative(boxes[axis])
            continue
        low, high = float(lower_parameters[axis]), float(upper_parameters[axis])
        if angular[axis]:
            boxes[axis], fits, outside = _into_range(boxes[axis], low, high)
            in_range &= fits
            excluded |= outside
        if rectangle is not None:
            # Inside every wall extent the open rectangle is the face interior.
            first, last = rectangle.interior[axis]
            interior &= (boxes[axis][0] > first) & (boxes[axis][1] < last)
            excluded |= (boxes[axis][1] < low) | (boxes[axis][0] > high)
    u, v = boxes
    middles, errors = [], []
    for value, straddles in zip(boxes, seams, strict=True):
        middle, error = _midpoint_error(value)
        # A seam point reports its lowest nonnegative native representative.
        middle = np.where(straddles, np.maximum(middle, 0.0), middle)
        error = np.where(
            straddles,
            np.maximum(
                np.nextafter(middle - value[0], np.inf),
                np.nextafter(value[1] - middle, np.inf),
            ),
            error,
        )
        middles.append(middle)
        errors.append(error)
    seam = seams[0] | seams[1]
    parameters_middle = np.stack(middles, axis=1)
    parameter_error = _norm_upper(np.stack(errors, axis=1))
    point, point_errors = _midpoint_error(rows.point)
    point_error = _norm_upper(point_errors)
    distance_lower = np.maximum(
        np.where(np.isfinite(rows.distance[0]), rows.distance[0], 0.0), 0.0
    )
    distance_upper = rows.distance[1]
    # A certified row has an isolated carrier minimizer within every tolerance.
    valid = (
        rows.valid
        & np.isfinite(distance_upper)
        & np.all(np.isfinite(point), axis=1)
        & np.all(np.isfinite(parameters_middle), axis=1)
        & (distance_upper - distance_lower <= tolerances.distance)
        & (point_error <= tolerances.distance)
        & (parameter_error <= tolerances.parametric)
    )
    inside = np.zeros((count,), dtype=np.bool_)
    refinements = np.zeros((count,), dtype=np.int32)
    if rectangle is not None:
        inside = in_range & interior
    else:
        selected = np.flatnonzero(valid & in_range)
        if selected.size:
            classification = data.trim_domain.classify_boxes(
                np.stack(
                    (
                        np.stack((u[0][selected], v[0][selected]), axis=1),
                        np.stack((u[1][selected], v[1][selected]), axis=1),
                    ),
                    axis=1,
                ),
                maximum_depth=maximum_depth,
                budget=coordinate_budget,
                preparations=preparations,
            )
            inside[selected] = np.asarray(classification.inside) & np.asarray(
                classification.resolved
            )
            refinements[selected] = np.asarray(classification.refinements, dtype=np.int32)
    accepted = valid & inside
    first, second = _derivatives(
        patch, np.where(accepted[:, None], parameters_middle, 0.0)
    )
    normal = np.cross(first, second)
    length = np.linalg.norm(normal, axis=1)
    usable = accepted & np.isfinite(length) & (length > 0.0)
    normal = (
        np.where(usable[:, None], normal / np.where(usable, length, 1.0)[:, None], np.nan)
        * data.orientation
    )
    delta = point - points
    gradient = np.stack(
        (np.sum(delta * first, axis=1), np.sum(delta * second, axis=1)), axis=1
    )
    denominator = np.maximum(
        np.linalg.norm(np.stack((first, second), axis=1), axis=(1, 2))
        * np.maximum(distance_upper, tolerances.distance),
        np.finfo(np.float64).tiny,
    )
    nan3 = np.full((count, 3), np.nan, dtype=np.float64)
    return AnalyticClosestBatch(
        points=np.where(usable[:, None], point, nan3),
        parameters=np.where(usable[:, None], parameters_middle, np.nan),
        distances=np.where(usable, distance_upper, np.inf).astype(np.float64),
        faces=np.full((count,), face, dtype=np.int32),
        status=np.where(
            usable,
            np.where(
                seam, int(BRepProjectionStatus.SEAM), int(BRepProjectionStatus.UNIQUE)
            ),
            int(BRepProjectionStatus.FAILED),
        ).astype(np.int8),
        normals=np.where(usable[:, None], normal, nan3),
        first_derivatives=np.where(usable[:, None], first, nan3),
        second_derivatives=np.where(usable[:, None], second, nan3),
        stationarity=np.where(
            usable, np.linalg.norm(gradient, axis=1) / denominator, np.nan
        ),
        distance_lower_bounds=distance_lower.astype(np.float64),
        # A carrier minimizer certified outside the face, or an undecided trim,
        # still bounds the face distance below with an infinite upper bound; a
        # non-isolated or loose carrier minimizer, or an ambiguous chart turn,
        # leaves the row unresolved.
        unresolved=~(valid & (in_range | excluded)),
        refinements=refinements,
        parameter_bounds=np.where(usable, parameter_error, np.inf),
        edge_indices=np.full((count,), -1, dtype=np.int32),
        edge_parameters=np.full((count,), np.nan, dtype=np.float64),
        point_error_bounds=np.where(usable, point_error, np.inf),
        query_operations=(1 + refinements).astype(np.int64),
    )


__all__ = ["carrier_supported", "carrier_workspace_bytes", "closest_carrier_face"]
