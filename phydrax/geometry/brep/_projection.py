#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor, pi, sqrt, tau
from typing import cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._validation import positive_finite_float
from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
)
from ...linalg._exact import (
    ExactLinearFactorization,
    factor_exact,
)
from ...linalg._hermitian_spectral import fraction_sqrt_interval
from ...typing import checked
from .._planar_embedding import PlanarEmbedding
from .._rational import fraction_endpoint
from .._source_boundary_records import SourceBoundaryDistance
from ._model import BRepModel
from ._patches import (
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    ConicRootLinePCurve,
    CylinderPatch,
    ExactLineCurve,
    ExtrusionSurface,
    LineCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    TorusPatch,
)
from ._projection_contracts import (
    AbstractBRepProjection,
    brep_entity_id,
    BRepEntityDimension,
    BRepProjectionPolicy,
    BRepProjectionResult,
    BRepProjectionStatus,
    closure_codes,
)
from ._tspline import TSplineSurfacePatch


_UNIQUE = int(BRepProjectionStatus.UNIQUE)
_SEAM = int(BRepProjectionStatus.SEAM)
_AMBIGUOUS = int(BRepProjectionStatus.AMBIGUOUS)
_SINGULAR = int(BRepProjectionStatus.SINGULAR)
_FAILED = int(BRepProjectionStatus.FAILED)
_UNRESOLVED = int(BRepProjectionStatus.UNRESOLVED)
_NON_ISOLATED = int(BRepProjectionStatus.NON_ISOLATED)
_NAN3 = np.full((3,), np.nan, dtype=np.float64)
_NAN2 = np.full((2,), np.nan, dtype=np.float64)
_NAN_FRAME3 = np.full((2, 3), np.nan, dtype=np.float64)

_IDENTITY_ROTATION = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_ZERO_TRANSLATION = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class _Occurrence:
    path: tuple[str, ...]
    solid: int
    rotation: tuple[tuple[float, ...], ...]
    translation: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _BoundaryEntity:
    entity_id: str
    dimension: int
    index: int
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AffineProjectionFrame:
    origin: tuple[ExactScalar, ...]
    first: tuple[ExactScalar, ...]
    second: tuple[ExactScalar, ...]
    gram_determinant: ExactScalar
    factorization: ExactLinearFactorization[ExactScalar] | None


@dataclass(frozen=True, slots=True)
class _Hit:
    dimension: int
    index: int
    point: np.ndarray
    parameters: np.ndarray
    lower: float
    upper: float
    status: int
    normal: np.ndarray
    tangents: np.ndarray
    path: tuple[str, ...]
    piece_id: tuple[object, ...] | None = None
    root_id: tuple[object, ...] | None = None


def _distance_bounds(first: np.ndarray, second: np.ndarray, /) -> tuple[float, float]:
    distance = float(np.linalg.norm(first - second))
    scale = max(
        1.0,
        float(np.max(np.abs(first), initial=0.0)),
        float(np.max(np.abs(second), initial=0.0)),
        distance,
    )
    error = 64.0 * np.finfo(np.float64).eps * scale
    return max(0.0, float(np.nextafter(distance - error, -np.inf))), float(
        np.nextafter(distance + error, np.inf)
    )


def _unresolved(dimension: int, index: int, path: tuple[str, ...] = ()) -> _Hit:
    return _Hit(
        dimension,
        index,
        _NAN3.copy(),
        _NAN2.copy(),
        0.0,
        np.inf,
        _UNRESOLVED,
        _NAN3.copy(),
        _NAN_FRAME3.copy(),
        path,
    )


def _failed(ambient: int, /) -> _Hit:
    return _Hit(
        -1,
        -1,
        np.full((ambient,), np.nan, dtype=np.float64),
        _NAN2.copy(),
        0.0,
        np.inf,
        _FAILED,
        np.full((ambient,), np.nan, dtype=np.float64),
        np.full((2, ambient), np.nan, dtype=np.float64),
        (),
    )


@dataclass(frozen=True)
class _CircleRevolutionFrame:
    """Meridian coordinates of a circle profile whose plane contains the axis.

    A profile point is ``origin + x meridian + y axis`` with
    ``(x, y) = center + radius (cos t first + sin t second)``; the revolution
    turns it to ``origin + x (cos u meridian + sin u binormal) + y axis``.
    """

    origin: np.ndarray
    axis: np.ndarray
    meridian: np.ndarray
    binormal: np.ndarray
    center: np.ndarray
    first: np.ndarray
    second: np.ndarray
    radius: float

    def profile(self, parameter: float, /) -> np.ndarray:
        return self.center + self.radius * (
            np.cos(parameter) * self.first + np.sin(parameter) * self.second
        )

    def turned(self, angle: float, /) -> np.ndarray:
        return np.cos(angle) * self.meridian + np.sin(angle) * self.binormal


def _circle_revolution_frame(
    patch: RevolutionSurface, /
) -> _CircleRevolutionFrame | None:
    curve = patch.curve
    if not isinstance(curve, CircleCurve):
        return None
    tolerance = 256.0 * np.finfo(np.float64).eps
    axis = np.asarray(patch.axis_direction, dtype=np.float64)
    origin = np.asarray(patch.axis_origin, dtype=np.float64)
    first = np.asarray(curve.first_axis, dtype=np.float64)
    second = np.asarray(curve.second_axis, dtype=np.float64)
    center = np.asarray(curve.center, dtype=np.float64) - origin
    radius = float(np.asarray(curve.radius))
    normal = np.cross(first, second)
    scale = max(1.0, float(np.max(np.abs(center))), radius)
    if (
        not radius > 0.0
        or abs(float(axis @ axis) - 1.0) > tolerance
        or abs(float(first @ first) - 1.0) > tolerance
        or abs(float(second @ second) - 1.0) > tolerance
        or abs(float(first @ second)) > tolerance
        or abs(float(normal @ axis)) > tolerance
        or abs(float(normal @ center)) > tolerance * scale
    ):
        return None
    meridian = np.cross(normal, axis)
    meridian /= np.linalg.norm(meridian)
    return _CircleRevolutionFrame(
        origin,
        axis,
        meridian,
        np.cross(axis, meridian),
        np.asarray((center @ meridian, center @ axis)),
        np.asarray((first @ meridian, first @ axis)),
        np.asarray((second @ meridian, second @ axis)),
        radius,
    )


def _circle_revolution_closest(
    patch: RevolutionSurface, bounds: np.ndarray, point: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Closest point of a circle-profile revolution chart rectangle to ``point``.

    The distance from the point at meridian coordinates ``(rho, z)`` to the
    turned profile point ``(x(t), y(t))`` is least at the turn aligning their
    meridian half-planes, where it is the meridian distance from
    ``(sign(x(t)) rho, z)`` to the profile point. The closest point of the
    whole revolution is therefore the closer of the profile-circle points
    nearest to ``(rho, z)`` and ``(-rho, z)``. A tied closest point outside the
    chart's profile interval is not on the face; the face-closest point is a
    tied one inside it, and none inside leaves the query unresolved (the angle
    is checked against the rectangle by the caller). A point whose closest set
    is a circle about the axis or a whole profile returns a representative with
    the non-isolated flag.
    """
    frame = _circle_revolution_frame(patch)
    if frame is None:
        return None
    relative = point - frame.origin
    height = float(relative @ frame.axis)
    radial = relative - height * frame.axis
    rho = float(np.linalg.norm(radial))
    angle = (
        float(bounds[0, 0])
        if rho == 0.0
        else float(np.arctan2(radial @ frame.binormal, radial @ frame.meridian))
    )
    period = patch.periods[1]
    first_parameter, last_parameter = (float(value) for value in bounds[:, 1])
    tolerance = (
        256.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(relative))), frame.radius)
    )
    options: list[tuple[float, float | None, bool]] = []
    for sign in (1.0, -1.0):
        offset = np.asarray((sign * rho, height)) - frame.center
        length = float(np.linalg.norm(offset))
        if length == 0.0:
            # Every profile point is equidistant.
            options.append((frame.radius, first_parameter, True))
            continue
        parameter: float | None = float(
            np.arctan2(offset @ frame.second, offset @ frame.first)
        )
        if period is not None:
            parameter = _periodic_parameter(
                parameter, first_parameter, last_parameter, period
            )
        elif not first_parameter <= parameter <= last_parameter:
            parameter = None
        options.append((abs(length - frame.radius), parameter, False))
    nearest = min(distance for distance, _, _ in options)
    closest = [
        (parameter, whole)
        for distance, parameter, whole in options
        if distance <= nearest + tolerance and parameter is not None
    ]
    if not closest:
        return None
    located: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for parameter, whole in closest:
        profile = frame.profile(parameter)
        turn = angle if rho == 0.0 or profile[0] >= 0.0 else angle + pi
        spatial = frame.origin + profile[0] * frame.turned(turn) + profile[1] * frame.axis
        around_axis = rho == 0.0 and abs(float(profile[0])) > tolerance
        located.append(
            (
                np.asarray((turn, parameter), dtype=np.float64),
                spatial,
                whole or around_axis,
            )
        )
    uv, expected, nonisolated = located[0]
    if any(np.linalg.norm(value - expected) > tolerance for _, value, _ in located[1:]):
        return None
    candidate = np.asarray(
        patch.evaluate(jnp.asarray(uv, dtype=jnp.float64)), dtype=np.float64
    )
    if np.linalg.norm(candidate - expected) > tolerance:
        return None
    return uv, candidate, nonisolated


def _circle_revolution_normal(
    patch: RevolutionSurface,
    parameters: np.ndarray,
    profile_interval: tuple[float, float],
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Regular normal and turn tangent of a circle-profile revolution chart.

    ``X_u x X_v`` is ``x(t)`` times the turned profile normal, which stays
    regular on the axis where ``x`` vanishes. ``x`` keeps one sign over the
    open profile interval of a chart that does not cross the axis; its sign at
    the interval midpoint orients the normal also at an axis point.
    """
    frame = _circle_revolution_frame(patch)
    if frame is None:
        return None
    angle, parameter = (float(value) for value in parameters)
    middle = 0.5 * (profile_interval[0] + profile_interval[1])
    side = float(np.sign(frame.profile(middle)[0]))
    if side == 0.0:
        return None
    derivative = frame.radius * (
        -np.sin(parameter) * frame.first + np.cos(parameter) * frame.second
    )
    # (a . q') w - (w . q') a in meridian coordinates, turned by the angle.
    turned = frame.turned(angle)
    normal = side * (derivative[1] * turned - derivative[0] * frame.axis)
    length = float(np.linalg.norm(normal))
    if not length > 0.0:
        return None
    tangent = -np.sin(angle) * frame.meridian + np.cos(angle) * frame.binormal
    return normal / length, tangent


def _face_normal_and_frame(
    patch: AbstractSurfacePatch,
    parameters: np.ndarray,
    orientation: float,
    /,
    *,
    embedding: PlanarEmbedding | None,
    profile_interval: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    jacobian = np.asarray(
        jax.jacfwd(lambda uv: patch.evaluate(uv))(
            jnp.asarray(parameters, dtype=jnp.float64)
        ),
        dtype=np.float64,
    )
    normal = np.cross(jacobian[:, 0], jacobian[:, 1])
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length <= 128.0 * np.finfo(np.float64).eps:
        if isinstance(patch, SpherePatch):
            longitude, latitude = (float(value) for value in parameters)
            first_axis = np.asarray(patch.first_axis, dtype=np.float64)
            second_axis = np.asarray(patch.second_axis, dtype=np.float64)
            third_axis = np.asarray(patch.axis, dtype=np.float64)
            radial = (
                np.cos(latitude) * np.cos(longitude) * first_axis
                + np.cos(latitude) * np.sin(longitude) * second_axis
                + np.sin(latitude) * third_axis
            )
            radial_length = float(np.linalg.norm(radial))
            if radial_length > 0.0 and np.isfinite(radial_length):
                normal = orientation * radial / radial_length
                first = -np.sin(longitude) * first_axis + np.cos(longitude) * second_axis
                first /= np.linalg.norm(first)
                frame = np.stack((first, np.cross(normal, first)))
                if embedding is None:
                    return normal, frame, True
                basis = np.stack((embedding.x_axis, embedding.y_axis), axis=1)
                return np.full((2,), np.nan), frame @ basis, True
        if isinstance(patch, RevolutionSurface) and profile_interval is not None:
            regular = _circle_revolution_normal(patch, parameters, profile_interval)
            if regular is not None:
                direction, first = regular
                normal = orientation * direction
                frame = np.stack((first, np.cross(normal, first)))
                if embedding is None:
                    return normal, frame, True
                basis = np.stack((embedding.x_axis, embedding.y_axis), axis=1)
                return np.full((2,), np.nan), frame @ basis, True
        if embedding is not None:
            return np.full((2,), np.nan), np.full((2, 2), np.nan), False
        return _NAN3.copy(), _NAN_FRAME3.copy(), False
    normal = orientation * normal / length
    first_length = float(np.linalg.norm(jacobian[:, 0]))
    if first_length <= 128.0 * np.finfo(np.float64).eps:
        first = jacobian[:, 1] / np.linalg.norm(jacobian[:, 1])
    else:
        first = jacobian[:, 0] / first_length
    second = np.cross(normal, first)
    frame = np.stack((first, second))
    if embedding is None:
        return normal, frame, True
    return (
        np.full((2,), np.nan),
        frame @ np.stack((embedding.x_axis, embedding.y_axis), axis=1),
        True,
    )


def _periodic_parameter(
    value: float, first: float, last: float, period: float, /
) -> float | None:
    if last - first > period * (1.0 + 128.0 * np.finfo(np.float64).eps):
        return None
    low = ceil((first - value) / period)
    high = floor((last - value) / period)
    if low > high:
        return None
    return value + low * period


def _trim_classification(model: BRepModel, face: int, uv: np.ndarray, /) -> int | None:
    """Classify a chart point only when every authored trim is a line p-curve."""
    geometry = model.geometry
    domain = model.trim_domains[face]
    if domain is not None and domain.curve_loops:
        result = domain.classify(np.asarray(uv, dtype=np.float64)[None, :])
        if not bool(np.asarray(result.resolved[0])):
            return None
        if bool(np.asarray(result.boundary[0])):
            return -1
        return int(bool(np.asarray(result.inside[0])))
    loops = geometry.face_loops[face]
    polygons: list[np.ndarray] = []
    for loop in loops:
        points: list[np.ndarray] = []
        first_start: np.ndarray | None = None
        previous_end: np.ndarray | None = None
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            curve = geometry.pcurves[coedge]
            if (
                not isinstance(curve, (LineCurve, ExactLineCurve))
                or curve.ambient_dimension != 2
            ):
                return None
            first, last = np.asarray(geometry.edge_ranges[edge], dtype=np.float64)
            start = np.asarray(curve.evaluate(jnp.asarray(first)), dtype=np.float64)
            end = np.asarray(curve.evaluate(jnp.asarray(last)), dtype=np.float64)
            if geometry.coedge_senses[coedge] < 0:
                start, end = end, start
            if first_start is None:
                first_start = start
            if previous_end is not None:
                scale = max(
                    1.0, float(np.max(np.abs(previous_end))), float(np.max(np.abs(start)))
                )
                if (
                    np.linalg.norm(previous_end - start)
                    > 256.0 * np.finfo(np.float64).eps * scale
                ):
                    return None
            points.append(start)
            previous_end = end
        if first_start is None or previous_end is None:
            return None
        scale = max(
            1.0, float(np.max(np.abs(previous_end))), float(np.max(np.abs(first_start)))
        )
        if (
            np.linalg.norm(previous_end - first_start)
            > 256.0 * np.finfo(np.float64).eps * scale
        ):
            return None
        polygon = np.asarray(points, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[0] < 3:
            return None
        polygons.append(polygon)

    def loop_state(polygon: np.ndarray, /) -> int:
        inside = False
        scale = max(
            1.0, float(np.max(np.abs(polygon), initial=0.0)), float(np.max(np.abs(uv)))
        )
        tolerance = 128.0 * np.finfo(np.float64).eps * scale
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            direction = end - start
            squared = float(direction @ direction)
            if squared == 0.0:
                continue
            coordinate = float(np.clip((uv - start) @ direction / squared, 0.0, 1.0))
            if np.linalg.norm(uv - (start + coordinate * direction)) <= tolerance:
                return -1
            crosses = (start[1] > uv[1]) != (end[1] > uv[1])
            if crosses:
                x_crossing = start[0] + (uv[1] - start[1]) * direction[0] / direction[1]
                if uv[0] < x_crossing:
                    inside = not inside
        return int(inside)

    states = tuple(loop_state(polygon) for polygon in polygons)
    if any(state < 0 for state in states):
        return -1
    return int(bool(states) and states[0] == 1 and not any(states[1:]))


def _full_chart_rectangle(model: BRepModel, face: int, /) -> bool:
    """Prove a single line-trim loop covers the entire parameter rectangle.

    Trim sides are chart lines: rational lines or exact conic-root angle
    isolines, whose bounds are outward enclosures of the exact root. A seam
    side may carry an integer native-period gauge over its line, and a side
    may be an exact affine chart transport of a line, which is again a line;
    a doubly periodic face (a full torus) is the rectangle of one period in
    each direction traced by such sides.
    """
    from ._boolean_core import AffinePCurve, PeriodicPCurve

    geometry = model.geometry
    loops = geometry.face_loops[face]
    if len(loops) != 1:
        return False
    bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(bounds))))
    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    vertical: tuple[list[tuple[float, float]], list[tuple[float, float]]] = ([], [])
    horizontal: tuple[list[tuple[float, float]], list[tuple[float, float]]] = ([], [])
    first_start: np.ndarray | None = None
    previous_end: np.ndarray | None = None
    for coedge in loops[0]:
        edge = geometry.coedge_edges[coedge]
        curve = geometry.pcurves[coedge]
        line = curve
        while isinstance(line, (PeriodicPCurve, AffinePCurve)):
            line = line.source_curve if isinstance(line, PeriodicPCurve) else line.curve
        if (
            not isinstance(line, (LineCurve, ExactLineCurve, ConicRootLinePCurve))
            or curve.ambient_dimension != 2
        ):
            return False
        first, last = np.asarray(geometry.edge_ranges[edge], dtype=np.float64)
        start = np.asarray(curve.evaluate(jnp.asarray(first)), dtype=np.float64)
        end = np.asarray(curve.evaluate(jnp.asarray(last)), dtype=np.float64)
        if geometry.coedge_senses[coedge] < 0:
            start, end = end, start
        if np.any(start < bounds[0] - tolerance) or np.any(start > bounds[1] + tolerance):
            return False
        if np.any(end < bounds[0] - tolerance) or np.any(end > bounds[1] + tolerance):
            return False
        if first_start is None:
            first_start = start
        if previous_end is not None and np.linalg.norm(previous_end - start) > tolerance:
            return False
        if abs(start[0] - end[0]) <= tolerance:
            if abs(start[0] - bounds[0, 0]) <= tolerance:
                vertical[0].append((min(start[1], end[1]), max(start[1], end[1])))
            elif abs(start[0] - bounds[1, 0]) <= tolerance:
                vertical[1].append((min(start[1], end[1]), max(start[1], end[1])))
            else:
                return False
        elif abs(start[1] - end[1]) <= tolerance:
            if abs(start[1] - bounds[0, 1]) <= tolerance:
                horizontal[0].append((min(start[0], end[0]), max(start[0], end[0])))
            elif abs(start[1] - bounds[1, 1]) <= tolerance:
                horizontal[1].append((min(start[0], end[0]), max(start[0], end[0])))
            else:
                return False
        else:
            return False
        previous_end = end

    if (
        first_start is None
        or previous_end is None
        or np.linalg.norm(previous_end - first_start) > tolerance
    ):
        return False

    def covers(
        intervals: list[tuple[float, float]], first: float, last: float, /
    ) -> bool:
        ordered = sorted(intervals)
        cursor = first
        for lower, upper in ordered:
            if lower > cursor + tolerance:
                return False
            cursor = max(cursor, upper)
        return cursor >= last - tolerance

    return all(
        (covers(vertical[side], bounds[0, 1], bounds[1, 1]) for side in range(2))
    ) and all((covers(horizontal[side], bounds[0, 0], bounds[1, 0]) for side in range(2)))


def _full_period(model: BRepModel, face: int, axis: int, /) -> bool:
    patch = model.patches[face]
    if patch.native_period_symbols()[axis] != "two_pi":
        return False
    from ._carrier_pieces import carrier_payload
    from ._intersection import NativePeriodEndpoint

    geometry = model.geometry
    for loop in geometry.face_loops[face]:
        for coedge in loop:
            first, last = geometry.coedge_endpoint_roots[coedge]
            if not isinstance(first, NativePeriodEndpoint) or not isinstance(
                last, NativePeriodEndpoint
            ):
                continue
            if (
                first.patch is not None
                and last.patch is not None
                and first.axis == last.axis == axis
                and first.rational == last.rational
                and last.turns - first.turns == 1
                and carrier_payload(first.patch) == carrier_payload(patch)
                and carrier_payload(last.patch) == carrier_payload(patch)
                and carrier_payload(first.curve)
                == carrier_payload(geometry.pcurves[coedge])
                and carrier_payload(last.curve)
                == carrier_payload(geometry.pcurves[coedge])
            ):
                return True
    return False


def _line_curve_hit(
    model: BRepModel,
    edge: int,
    point: np.ndarray,
    path: tuple[str, ...],
    policy: BRepProjectionPolicy,
    /,
) -> _Hit:
    geometry = model.geometry
    carrier = geometry.edge_curves[edge]
    if carrier < 0:
        vertex = geometry.edge_vertices[edge][0]
        closest = np.asarray(geometry.vertex_points[vertex], dtype=np.float64)
        lower, upper = _distance_bounds(point, closest)
        return _Hit(
            1,
            edge,
            closest,
            np.asarray((np.nan, np.nan)),
            lower,
            upper,
            _SEAM,
            _NAN3.copy(),
            _NAN_FRAME3.copy(),
            path,
        )
    curve = geometry.curves[carrier]
    if (
        not isinstance(curve, (LineCurve, ExactLineCurve, CircleCurve, BSplineCurve))
        or curve.ambient_dimension != 3
    ):
        return _unresolved(1, edge, path)
    ranges = np.asarray(geometry.edge_ranges[edge], dtype=np.float64)
    first, last = float(ranges[0]), float(ranges[1])
    spline_projection = None
    if isinstance(curve, BSplineCurve):
        from ._spline_projection import project_bspline_curve

        spline_projection = project_bspline_curve(
            curve, (first, last), point, policy, edge_range=(first, last)
        )
        if spline_projection.status in (_UNRESOLVED, _FAILED):
            return _Hit(
                1,
                edge,
                _NAN3.copy(),
                _NAN2.copy(),
                spline_projection.lower,
                spline_projection.upper,
                spline_projection.status,
                _NAN3.copy(),
                _NAN_FRAME3.copy(),
                path,
            )
        if spline_projection.status == _NON_ISOLATED:
            return _Hit(
                1,
                edge,
                _NAN3.copy(),
                _NAN2.copy(),
                spline_projection.lower,
                spline_projection.upper,
                _NON_ISOLATED,
                _NAN3.copy(),
                _NAN_FRAME3.copy(),
                path,
            )
        parameter = spline_projection.parameter
        closest = spline_projection.point
        lower, upper = spline_projection.lower, spline_projection.upper
        status = spline_projection.status
    elif isinstance(curve, (LineCurve, ExactLineCurve)):
        if isinstance(curve, LineCurve):
            origin = np.asarray(curve.origin, dtype=np.float64)
            direction = np.asarray(curve.direction, dtype=np.float64)
        else:
            origin = np.asarray(curve.first, dtype=np.float64)
            direction = np.asarray(curve.last - curve.first, dtype=np.float64)
        squared = float(direction @ direction)
        if squared <= np.finfo(np.float64).tiny:
            return _unresolved(1, edge, path)
        parameter = float(np.clip((point - origin) @ direction / squared, first, last))
        closest = np.asarray(curve.evaluate(jnp.asarray(parameter)), dtype=np.float64)
        lower, upper = _distance_bounds(point, closest)
        status = _SEAM if parameter == first or parameter == last else _UNIQUE
    else:
        relative = point - np.asarray(curve.center, dtype=np.float64)
        x = float(relative @ np.asarray(curve.first_axis))
        y = float(relative @ np.asarray(curve.second_axis))
        radius = float(np.asarray(curve.radius))
        if x == 0.0 and y == 0.0:
            return _Hit(
                1,
                edge,
                _NAN3.copy(),
                _NAN2.copy(),
                max(0.0, radius),
                float(np.hypot(np.linalg.norm(relative), radius)),
                _NON_ISOLATED,
                _NAN3.copy(),
                _NAN_FRAME3.copy(),
                path,
            )
        candidate = float(np.arctan2(y, x))
        parameter = _periodic_parameter(candidate, first, last, tau)
        if parameter is None:
            endpoints = (first, last)
            values = [
                np.asarray(curve.evaluate(jnp.asarray(value)), dtype=np.float64)
                for value in endpoints
            ]
            parameter = endpoints[
                int(np.argmin([np.sum((value - point) ** 2) for value in values]))
            ]
        closest = np.asarray(curve.evaluate(jnp.asarray(parameter)), dtype=np.float64)
        lower, upper = _distance_bounds(point, closest)
        status = _SEAM if parameter == first or parameter == last else _UNIQUE
    parameter = float(parameter)
    derivative = np.asarray(
        jax.jacfwd(lambda value: curve.evaluate(value))(jnp.asarray(parameter)),
        dtype=np.float64,
    )
    derivative_length = float(np.linalg.norm(derivative))
    tangent = derivative / derivative_length if derivative_length else _NAN3.copy()
    tangents = np.stack((tangent, _NAN3))
    if derivative_length == 0.0 and status != _AMBIGUOUS:
        status = _SINGULAR
    return _Hit(
        1,
        edge,
        closest,
        np.asarray((parameter, np.nan)),
        lower,
        upper,
        status,
        _NAN3.copy(),
        tangents,
        path,
    )


def _natural_rectangle_trim(model: BRepModel, face: int, /) -> bool:
    """Prove that a face's single authored loop traces its parameter rectangle.

    Every coedge must be an exact axis-aligned chart line (possibly on a
    native period sheet) along one rectangle side, joined corner to corner.
    """
    from ._boolean_core import PeriodicPCurve

    geometry = model.geometry
    if geometry is None or len(geometry.face_loops[face]) != 1:
        return False
    bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    corners = {(float(bounds[i, 0]), float(bounds[j, 1])) for i in (0, 1) for j in (0, 1)}
    loop = geometry.face_loops[face][0]
    path: list[tuple[float, float]] = []
    for coedge in loop:
        pcurve = geometry.pcurves[coedge]
        while isinstance(pcurve, PeriodicPCurve):
            pcurve = pcurve.source_curve
        if not isinstance(pcurve, (LineCurve, ExactLineCurve)):
            return False
        edge = geometry.coedge_edges[coedge]
        first, last = np.asarray(geometry.edge_ranges[edge], dtype=np.float64)
        ends = np.asarray(
            geometry.pcurves[coedge].evaluate(jnp.asarray((first, last))),
            dtype=np.float64,
        )
        if geometry.coedge_senses[coedge] < 0:
            ends = ends[::-1]
        start, end = (
            (float(ends[0, 0]), float(ends[0, 1])),
            (
                float(ends[1, 0]),
                float(ends[1, 1]),
            ),
        )
        if start not in corners or end not in corners or start == end:
            return False
        if start[0] != end[0] and start[1] != end[1]:
            return False
        if path and path[-1] != start:
            return False
        path.append(start)
        path.append(end)
    return len(loop) == 4 and path[0] == path[-1] and set(path) == corners


def _plane_boundary_hit(
    model: BRepModel,
    face: int,
    point: np.ndarray,
    patch: AbstractSurfacePatch,
    path: tuple[str, ...],
    orientation: float,
    policy: BRepProjectionPolicy,
    /,
    *,
    embedding: PlanarEmbedding | None,
) -> _Hit | None:
    """Closest boundary point of a planar face bounded by line and circle edges.

    Each edge's closest point comes from its exact carrier; its p-curve must lift
    to that point, so the chart point names the same boundary location.
    """
    geometry = model.geometry
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for loop in geometry.face_loops[face]:
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            pcurve = geometry.pcurves[coedge]
            carrier = geometry.edge_curves[edge]
            if pcurve.ambient_dimension != 2 or carrier < 0:
                return None
            curve = geometry.curves[carrier]
            lines = isinstance(pcurve, (LineCurve, ExactLineCurve)) and isinstance(
                curve, (LineCurve, ExactLineCurve)
            )
            circles = isinstance(pcurve, CircleCurve) and isinstance(curve, CircleCurve)
            if not lines and not circles:
                return None
            hit = _line_curve_hit(model, edge, point, path, policy)
            if hit.status not in (_UNIQUE, _SEAM):
                return None
            parameter = float(hit.parameters[0])
            uv = np.asarray(pcurve.evaluate(jnp.asarray(parameter)), dtype=np.float64)
            lifted = np.asarray(patch.evaluate(jnp.asarray(uv)), dtype=np.float64)
            closest = np.asarray(hit.point, dtype=np.float64)
            scale = max(
                1.0,
                float(np.max(np.abs(point))),
                float(np.max(np.abs(closest))),
            )
            tolerance = 256.0 * np.finfo(np.float64).eps * scale
            if np.linalg.norm(closest - lifted) > tolerance:
                return None
            distance = float(np.sum((point - closest) ** 2))
            if best is None or distance < best[0]:
                best = distance, closest, uv
    if best is None:
        return None
    _, closest, uv = best
    lower, upper = _distance_bounds(point, closest)
    normal, tangents, regular = _face_normal_and_frame(
        patch, uv, orientation, embedding=embedding
    )
    if embedding is not None:
        closest = embedding.to_planar(closest)
    return _Hit(
        2,
        face,
        closest,
        uv,
        lower,
        upper,
        _SEAM if not regular else _UNIQUE,
        normal,
        tangents,
        path,
    )


def _tspline_affine_plane(
    patch: AbstractSurfacePatch, /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    """Prove that every exact T-spline extraction piece is one affine map."""
    from ._tspline import TSplineSurfacePatch

    if not isinstance(patch, TSplineSurfacePatch):
        return None
    patch.validate_exact_transfer_chain()
    common_weight: Fraction | None = None
    affine: (
        tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]] | None
    ) = None
    for extraction in patch.realization.extractions:
        bounds = extraction.parameter_bounds
        u0, u1, v0, v1 = bounds
        if u0 >= u1 or v0 >= v1:
            return None
        coefficients = patch.realization.exact_homogeneous_piece(
            extraction.piece_id, patch.exact_target_homogeneous
        )
        weights = tuple(row[3] for row in coefficients)
        if (
            not weights
            or weights[0] <= 0
            or any(value != weights[0] for value in weights)
        ):
            return None
        if common_weight is None:
            common_weight = weights[0]
        elif common_weight != weights[0]:
            return None
        controls = tuple(
            tuple(value / weights[0] for value in row[:3]) for row in coefficients
        )
        origin_control = controls[0]
        u_axis = tuple(
            (controls[12][axis] - origin_control[axis]) / (u1 - u0) for axis in range(3)
        )
        v_axis = tuple(
            (controls[3][axis] - origin_control[axis]) / (v1 - v0) for axis in range(3)
        )
        origin = tuple(
            origin_control[axis] - u_axis[axis] * u0 - v_axis[axis] * v0
            for axis in range(3)
        )
        candidate = origin, u_axis, v_axis
        if affine is None:
            affine = candidate
        elif affine != candidate:
            return None
        for index, control in enumerate(controls):
            u = u0 + Fraction(index // 4, 3) * (u1 - u0)
            v = v0 + Fraction(index % 4, 3) * (v1 - v0)
            expected = tuple(
                origin[axis] + u_axis[axis] * u + v_axis[axis] * v for axis in range(3)
            )
            if control != expected:
                return None
    if affine is None:
        return None
    _, first, second = affine
    first_norm = sum((value * value for value in first), Fraction(0))
    second_norm = sum((value * value for value in second), Fraction(0))
    mixed = sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))
    if first_norm * second_norm - mixed * mixed <= 0:
        return None
    return affine


def _prepare_affine_projection_frame(
    patch: AbstractSurfacePatch, /
) -> _AffineProjectionFrame | None:
    if isinstance(patch, PlanePatch):
        origin, first, second = (
            tuple(Fraction.from_float(float(value)) for value in np.asarray(vector))
            for vector in (patch.origin, patch.first_axis, patch.second_axis)
        )
    elif isinstance(patch, TSplineSurfacePatch):
        frame = _tspline_affine_plane(patch)
        if frame is None:
            return None
        origin, first, second = frame
    elif isinstance(patch, ExtrusionSurface) and isinstance(
        patch.curve, (LineCurve, ExactLineCurve)
    ):
        if isinstance(patch.curve, LineCurve):
            origin, first = patch.curve.origin_exact, patch.curve.direction_exact
        else:
            origin = patch.curve.first_exact
            first = tuple(
                last - start
                for start, last in zip(
                    patch.curve.first_exact,
                    patch.curve.last_exact,
                    strict=True,
                )
            )
        second = tuple(
            Fraction.from_float(float(value)) for value in np.asarray(patch.direction)
        )
    else:
        return None

    def dot(
        left: tuple[ExactScalar, ...], right: tuple[ExactScalar, ...], /
    ) -> ExactScalar:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    gram = (
        (dot(first, first), dot(first, second)),
        (dot(first, second), dot(second, second)),
    )
    try:
        factorization = factor_exact(gram)
    except ValueError:
        return _AffineProjectionFrame(origin, first, second, Fraction(0), None)
    determinant = factorization.determinant
    if determinant <= 0:
        return _AffineProjectionFrame(origin, first, second, determinant, None)
    return _AffineProjectionFrame(origin, first, second, determinant, factorization)


def _project_affine_frame_exact(
    frame: _AffineProjectionFrame,
    query: tuple[ExactScalar, ExactScalar, ExactScalar],
    /,
) -> (
    tuple[
        tuple[ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
        ExactScalar,
    ]
    | None
):
    if frame.factorization is None:
        return None
    relative = (
        query[0] - frame.origin[0],
        query[1] - frame.origin[1],
        query[2] - frame.origin[2],
    )
    right = (
        sum((a * b for a, b in zip(frame.first, relative, strict=True)), Fraction(0)),
        sum((a * b for a, b in zip(frame.second, relative, strict=True)), Fraction(0)),
    )
    solved = frame.factorization.solve(((right[0],), (right[1],)))
    uv = (solved[0][0], solved[1][0])
    closest = (
        frame.origin[0] + uv[0] * frame.first[0] + uv[1] * frame.second[0],
        frame.origin[1] + uv[0] * frame.first[1] + uv[1] * frame.second[1],
        frame.origin[2] + uv[0] * frame.first[2] + uv[1] * frame.second[2],
    )
    squared_distance = sum(
        ((query[axis] - closest[axis]) ** 2 for axis in range(3)),
        Fraction(0),
    )
    return (uv[0], uv[1]), closest, squared_distance


def _project_affine_frame(
    frame: _AffineProjectionFrame,
    point: np.ndarray,
    /,
) -> (
    tuple[
        tuple[ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
        ExactScalar,
    ]
    | None
):
    query = (
        Fraction.from_float(float(point[0])),
        Fraction.from_float(float(point[1])),
        Fraction.from_float(float(point[2])),
    )
    return _project_affine_frame_exact(frame, query)


def _affine_distance_bounds(value: ExactScalar, /) -> tuple[float, float]:
    lower_squared, upper_squared = coefficient_interval(value)
    lower_squared = max(Fraction(0), lower_squared)
    upper_squared = max(Fraction(0), upper_squared)
    lower_root = fraction_sqrt_interval(lower_squared)[0]
    upper_root = fraction_sqrt_interval(upper_squared)[1]
    return (
        fraction_endpoint(lower_root, upper=False, saturate=True),
        fraction_endpoint(upper_root, upper=True, saturate=True),
    )


def _tspline_graph_stationary_hit(
    model: BRepModel,
    face: int,
    point: np.ndarray,
    path: tuple[str, ...],
    orientation: float,
    /,
) -> _Hit | None:
    """Certify the vertical stationary projection of an exact T-spline graph."""
    from .._rational import fraction_down, fraction_up
    from ._tspline import TSplineSurfacePatch

    patch = model.patches[face]
    if not isinstance(patch, TSplineSurfacePatch):
        return None
    patch.validate_exact_transfer_chain()
    if not patch.is_c1_on(np.asarray(model.parameter_bounds[face], dtype=np.float64)):
        return None
    if not _full_chart_rectangle(model, face):
        return None
    model_box = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    model_bounds = (
        Fraction.from_float(float(model_box[0, 0])),
        Fraction.from_float(float(model_box[1, 0])),
        Fraction.from_float(float(model_box[0, 1])),
        Fraction.from_float(float(model_box[1, 1])),
    )
    domain = patch.mesh.parameter_domain
    if model_bounds != domain:
        return None
    common_weight: Fraction | None = None
    pieces: list[
        tuple[
            tuple[Fraction, Fraction, Fraction, Fraction],
            tuple[Fraction, ...],
        ]
    ] = []
    lower_height: Fraction | None = None
    upper_height: Fraction | None = None
    maximum_uu = Fraction(0)
    maximum_vv = Fraction(0)
    maximum_uv = Fraction(0)
    for extraction in patch.realization.extractions:
        u0, u1, v0, v1 = extraction.parameter_bounds
        homogeneous = patch.realization.exact_homogeneous_piece(
            extraction.piece_id, patch.exact_target_homogeneous
        )
        weights = tuple(row[3] for row in homogeneous)
        if (
            not weights
            or weights[0] <= 0
            or any(value != weights[0] for value in weights)
        ):
            return None
        if common_weight is None:
            common_weight = weights[0]
        elif common_weight != weights[0]:
            return None
        controls = tuple(
            tuple(value / weights[0] for value in row[:3]) for row in homogeneous
        )
        heights = tuple(control[2] for control in controls)
        for index, control in enumerate(controls):
            expected_u = u0 + Fraction(index // 4, 3) * (u1 - u0)
            expected_v = v0 + Fraction(index % 4, 3) * (v1 - v0)
            if control[0] != expected_u or control[1] != expected_v:
                return None
        pieces.append((extraction.parameter_bounds, heights))
        piece_minimum, piece_maximum = min(heights), max(heights)
        lower_height = (
            piece_minimum if lower_height is None else min(lower_height, piece_minimum)
        )
        upper_height = (
            piece_maximum if upper_height is None else max(upper_height, piece_maximum)
        )
        du, dv = u1 - u0, v1 - v0
        uu = tuple(
            abs(
                Fraction(6)
                * (
                    heights[(i + 2) * 4 + j]
                    - 2 * heights[(i + 1) * 4 + j]
                    + heights[i * 4 + j]
                )
                / (du * du)
            )
            for i in range(2)
            for j in range(4)
        )
        vv = tuple(
            abs(
                Fraction(6)
                * (
                    heights[i * 4 + j + 2]
                    - 2 * heights[i * 4 + j + 1]
                    + heights[i * 4 + j]
                )
                / (dv * dv)
            )
            for i in range(4)
            for j in range(2)
        )
        uv = tuple(
            abs(
                Fraction(9)
                * (
                    heights[(i + 1) * 4 + j + 1]
                    - heights[(i + 1) * 4 + j]
                    - heights[i * 4 + j + 1]
                    + heights[i * 4 + j]
                )
                / (du * dv)
            )
            for i in range(3)
            for j in range(3)
        )
        maximum_uu = max(maximum_uu, *uu)
        maximum_vv = max(maximum_vv, *vv)
        maximum_uv = max(maximum_uv, *uv)
    if lower_height is None or upper_height is None or not pieces:
        return _unresolved(2, face, path)
    parameter_u = Fraction.from_float(float(point[0]))
    parameter_v = Fraction.from_float(float(point[1]))
    if not (domain[0] < parameter_u < domain[1] and domain[2] < parameter_v < domain[3]):
        return _unresolved(2, face, path)
    selected = next(
        (
            (bounds, heights)
            for bounds, heights in pieces
            if bounds[0] <= parameter_u <= bounds[1]
            and bounds[2] <= parameter_v <= bounds[3]
        ),
        None,
    )
    if selected is None:
        return _unresolved(2, face, path)
    (u0, u1, v0, v1), heights = selected
    u_fraction = (parameter_u - u0) / (u1 - u0)
    v_fraction = (parameter_v - v0) / (v1 - v0)

    def basis(degree: int, value: Fraction, /) -> tuple[Fraction, ...]:
        if degree == 2:
            return ((1 - value) ** 2, 2 * value * (1 - value), value**2)
        return (
            (1 - value) ** 3,
            3 * value * (1 - value) ** 2,
            3 * value**2 * (1 - value),
            value**3,
        )

    bu3, bv3 = basis(3, u_fraction), basis(3, v_fraction)
    bu2, bv2 = basis(2, u_fraction), basis(2, v_fraction)
    height = sum(
        (heights[i * 4 + j] * bu3[i] * bv3[j] for i in range(4) for j in range(4)),
        Fraction(0),
    )
    derivative_u = sum(
        (
            3
            * (heights[(i + 1) * 4 + j] - heights[i * 4 + j])
            / (u1 - u0)
            * bu2[i]
            * bv3[j]
            for i in range(3)
            for j in range(4)
        ),
        Fraction(0),
    )
    derivative_v = sum(
        (
            3
            * (heights[i * 4 + j + 1] - heights[i * 4 + j])
            / (v1 - v0)
            * bu3[i]
            * bv2[j]
            for i in range(4)
            for j in range(3)
        ),
        Fraction(0),
    )
    if derivative_u != 0 or derivative_v != 0:
        return _unresolved(2, face, path)
    query_height = Fraction.from_float(float(point[2]))
    vertical_extent = max(
        abs(query_height - lower_height), abs(query_height - upper_height)
    )
    hessian_norm_squared = (
        maximum_uu * maximum_uu + maximum_vv * maximum_vv + 2 * maximum_uv * maximum_uv
    )
    if vertical_extent * vertical_extent * hessian_norm_squared >= 1:
        return None
    chart = np.asarray((float(parameter_u), float(parameter_v)), dtype=np.float64)
    if _trim_classification(model, face, chart) != 1:
        return _unresolved(2, face, path)
    distance = abs(query_height - height)
    lower = fraction_down(distance)
    upper = fraction_up(distance)
    candidate = np.asarray(
        (float(parameter_u), float(parameter_v), float(height)), dtype=np.float64
    )
    normal = np.asarray((0.0, 0.0, orientation), dtype=np.float64)
    tangents = np.asarray(((1.0, 0.0, 0.0), (0.0, orientation, 0.0)), dtype=np.float64)
    return _Hit(2, face, candidate, chart, lower, upper, _UNIQUE, normal, tangents, path)


def _project_face(
    model: BRepModel,
    face: int,
    point: np.ndarray,
    path: tuple[str, ...],
    orientation: float,
    policy: BRepProjectionPolicy,
    /,
    *,
    embedding: PlanarEmbedding | None,
    affine_frame: _AffineProjectionFrame | None,
) -> _Hit:
    patch = model.patches[face]
    candidate: np.ndarray | None = None
    uv: np.ndarray | None = None
    nonisolated = False
    affine_spline = False
    # Straight-ruled revolutions and cylinders: the candidate is the closest
    # point of the whole authored parameter rectangle, including its profile
    # endpoints, so it is the face-closest point of a natural-rectangle face.
    rectangle_closest = False
    affine_distance_squared: ExactScalar | None = None
    if isinstance(patch, PlanePatch) or (
        isinstance(patch, TSplineSurfacePatch) and affine_frame is not None
    ):
        if affine_frame is None:
            return _unresolved(2, face, path)
        projection = _project_affine_frame(affine_frame, point)
        if projection is None:
            return _unresolved(2, face, path)
        exact_uv, exact_candidate, affine_distance_squared = projection
        uv = np.asarray(tuple(float(value) for value in exact_uv), dtype=np.float64)
        candidate = np.asarray(
            tuple(float(value) for value in exact_candidate), dtype=np.float64
        )
        if not np.all(np.isfinite(uv)) or not np.all(np.isfinite(candidate)):
            return _unresolved(2, face, path)
        affine_spline = isinstance(patch, TSplineSurfacePatch)
    elif (
        graph_hit := _tspline_graph_stationary_hit(model, face, point, path, orientation)
    ) is not None:
        return graph_hit
    elif isinstance(patch, (BSplineSurfacePatch, TSplineSurfacePatch)):
        from ._spline_projection import project_spline_face

        projected = project_spline_face(
            model,
            face,
            point,
            policy,
            trim_classify=_trim_classification,
            full_rectangle=_full_chart_rectangle,
        )
        if projected.status in (_FAILED, _UNRESOLVED, _NON_ISOLATED):
            return _Hit(
                2,
                face,
                projected.point,
                projected.parameters,
                projected.lower,
                projected.upper,
                projected.status,
                _NAN3.copy(),
                _NAN_FRAME3.copy(),
                path,
                projected.piece_id,
                projected.root_id,
            )
        candidate = projected.point
        uv = np.array(projected.parameters, dtype=np.float64, copy=True)
        status = projected.status
        normal, tangents, regular = _face_normal_and_frame(
            patch, uv, orientation, embedding=embedding
        )
        if not regular and status != _SINGULAR:
            return _Hit(
                2,
                face,
                _NAN3.copy(),
                _NAN2.copy(),
                projected.lower,
                projected.upper,
                _UNRESOLVED,
                _NAN3.copy(),
                _NAN_FRAME3.copy(),
                path,
                projected.piece_id,
                projected.root_id,
            )
        parameter_bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
        for axis, period in enumerate(patch.periods):
            if period is None:
                continue
            representative = _periodic_parameter(
                float(uv[axis]),
                parameter_bounds[0, axis],
                parameter_bounds[1, axis],
                period,
            )
            if representative is None:
                return _Hit(
                    2,
                    face,
                    _NAN3.copy(),
                    _NAN2.copy(),
                    projected.lower,
                    projected.upper,
                    _UNRESOLVED,
                    _NAN3.copy(),
                    _NAN_FRAME3.copy(),
                    path,
                    projected.piece_id,
                    projected.root_id,
                )
            uv[axis] = representative
            seam_tolerance = (
                128.0 * np.finfo(np.float64).eps * max(1.0, abs(float(uv[axis])))
            )
            if (
                status == _UNIQUE
                and abs(float(uv[axis]) - parameter_bounds[0, axis]) <= seam_tolerance
            ):
                status = _SEAM
        if embedding is not None:
            candidate = embedding.to_planar(candidate)
        return _Hit(
            2,
            face,
            candidate,
            uv,
            projected.lower,
            projected.upper,
            status,
            normal,
            tangents,
            path,
            projected.piece_id,
            projected.root_id,
        )
    elif isinstance(patch, CylinderPatch):
        origin = np.asarray(patch.origin, dtype=np.float64)
        first = np.asarray(patch.first_axis, dtype=np.float64)
        second = np.asarray(patch.second_axis, dtype=np.float64)
        axis = np.asarray(patch.axis, dtype=np.float64)
        # The radial frame is orthonormal and normal to the axis; the axis
        # length is the height scale of the second parameter.
        axis_squared = float(axis @ axis)
        if not axis_squared > 0.0:
            return _unresolved(2, face, path)
        frame = np.stack((first, second, axis / np.sqrt(axis_squared)))
        if not np.allclose(
            frame @ frame.T,
            np.eye(3),
            rtol=0.0,
            atol=256.0 * np.finfo(np.float64).eps,
        ):
            return _unresolved(2, face, path)
        radius = float(patch.radius)
        if not radius > 0.0:
            return _unresolved(2, face, path)
        relative = point - origin
        axial = float(relative @ axis) / axis_squared
        x, y = float(relative @ first), float(relative @ second)
        radial = float(np.hypot(x, y))
        bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
        if radial == 0.0:
            if (
                not _full_chart_rectangle(model, face)
                or not _full_period(model, face, 0)
                or axial < bounds[0, 1]
                or axial > bounds[1, 1]
            ):
                return _unresolved(2, face, path)
            candidate = origin + radius * first + axial * axis
            lower, upper = _distance_bounds(point, candidate)
            return _Hit(
                2,
                face,
                _NAN3.copy(),
                _NAN2.copy(),
                lower,
                upper,
                _NON_ISOLATED,
                _NAN3.copy(),
                _NAN_FRAME3.copy(),
                path,
            )
        angle = float(np.arctan2(y, x))
        # Axial and angular distances separate on a cylinder: the free angle and
        # the height clamped to the authored range give the rectangle-closest
        # point, seam angles included.
        clamped = float(np.clip(axial, bounds[0, 1], bounds[1, 1]))
        rectangle_closest = True
        uv = np.asarray((angle, clamped), dtype=np.float64)
        candidate = (
            origin
            + radius * (np.cos(angle) * first + np.sin(angle) * second)
            + clamped * axis
        )
    elif isinstance(patch, SpherePatch):
        center = np.asarray(patch.center, dtype=np.float64)
        axes = tuple(
            np.asarray(axis, dtype=np.float64)
            for axis in (patch.first_axis, patch.second_axis, patch.axis)
        )
        frame = np.column_stack(axes)
        if (
            not np.allclose(
                frame.T @ frame,
                np.eye(3),
                rtol=0.0,
                atol=256.0 * np.finfo(np.float64).eps,
            )
            or np.linalg.det(frame) <= 0.0
        ):
            return _unresolved(2, face, path)
        radius = float(patch.radius)
        if not radius > 0.0:
            return _unresolved(2, face, path)
        relative = point - center
        coordinates = np.asarray([relative @ axis for axis in axes], dtype=np.float64)
        length = float(np.linalg.norm(coordinates))
        if length == 0.0:
            bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
            latitude_tolerance = 256.0 * np.finfo(np.float64).eps
            if (
                not _full_chart_rectangle(model, face)
                or not _full_period(model, face, 0)
                or bounds[0, 1] > -pi / 2.0 + latitude_tolerance
                or bounds[1, 1] < pi / 2.0 - latitude_tolerance
            ):
                return _unresolved(2, face, path)
            candidate = center + radius * axes[0]
            lower, upper = _distance_bounds(point, candidate)
            return _Hit(
                2,
                face,
                _NAN3.copy(),
                _NAN2.copy(),
                lower,
                upper,
                _NON_ISOLATED,
                _NAN3.copy(),
                _NAN_FRAME3.copy(),
                path,
            )
        longitude = float(np.arctan2(coordinates[1], coordinates[0]))
        latitude = float(
            np.arctan2(coordinates[2], np.hypot(coordinates[0], coordinates[1]))
        )
        if np.hypot(coordinates[0], coordinates[1]) == 0.0:
            longitude = float(np.asarray(model.parameter_bounds)[face, 0, 0])
        uv = np.asarray((longitude, latitude), dtype=np.float64)
        candidate = center + radius * relative / length
        # The radial point is the closest point of the whole sphere, so inside
        # the parameter rectangle it is also the rectangle's closest point,
        # including on the seam and at a pole.
        rectangle_closest = True
    elif isinstance(patch, ConePatch):
        origin = np.asarray(patch.origin, dtype=np.float64)
        first = np.asarray(patch.first_axis, dtype=np.float64)
        second = np.asarray(patch.second_axis, dtype=np.float64)
        axis = np.asarray(patch.axis, dtype=np.float64)
        frame = np.column_stack((first, second, axis))
        if (
            not np.allclose(
                frame.T @ frame,
                np.eye(3),
                rtol=0.0,
                atol=256.0 * np.finfo(np.float64).eps,
            )
            or np.linalg.det(frame) <= 0.0
        ):
            return _unresolved(2, face, path)
        relative = point - origin
        axial_point = float(relative @ axis)
        x, y = float(relative @ first), float(relative @ second)
        radial_point = float(np.hypot(x, y))
        tangent = float(np.tan(float(patch.semi_angle)))
        reference = float(patch.reference_radius)
        axial = (axial_point + tangent * (radial_point - reference)) / (
            1.0 + tangent * tangent
        )
        radius = reference + axial * tangent
        if radial_point == 0.0 and radius != 0.0:
            bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
            if (
                not _full_chart_rectangle(model, face)
                or not _full_period(model, face, 0)
                or axial < bounds[0, 1]
                or axial > bounds[1, 1]
            ):
                return _unresolved(2, face, path)
            candidate = origin + axial * axis + radius * first
            lower, upper = _distance_bounds(point, candidate)
            return _Hit(
                2,
                face,
                _NAN3.copy(),
                _NAN2.copy(),
                lower,
                upper,
                _NON_ISOLATED,
                _NAN3.copy(),
                _NAN_FRAME3.copy(),
                path,
            )
        angle = 0.0 if radial_point == 0.0 else float(np.arctan2(y, x))
        uv = np.asarray((angle, axial), dtype=np.float64)
        candidate = (
            origin
            + axial * axis
            + radius * (np.cos(angle) * first + np.sin(angle) * second)
        )
    elif isinstance(patch, TorusPatch):
        center = np.asarray(patch.center, dtype=np.float64)
        first = np.asarray(patch.first_axis, dtype=np.float64)
        second = np.asarray(patch.second_axis, dtype=np.float64)
        axis = np.asarray(patch.axis, dtype=np.float64)
        frame = np.column_stack((first, second, axis))
        if (
            not np.allclose(
                frame.T @ frame,
                np.eye(3),
                rtol=0.0,
                atol=256.0 * np.finfo(np.float64).eps,
            )
            or np.linalg.det(frame) <= 0.0
        ):
            return _unresolved(2, face, path)
        relative = point - center
        x, y, z = (
            float(relative @ first),
            float(relative @ second),
            float(relative @ axis),
        )
        radial = float(np.hypot(x, y))
        major, minor = float(patch.major_radius), float(patch.minor_radius)
        if not major > minor > 0.0:
            return _unresolved(2, face, path)
        tube_x, tube_y = radial - major, z
        tube_length = float(np.hypot(tube_x, tube_y))
        if tube_length == 0.0:
            bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
            if (
                not _full_chart_rectangle(model, face)
                or not _full_period(model, face, 0)
                or not _full_period(model, face, 1)
            ):
                return _unresolved(2, face, path)
            candidate = center + (major + minor) * first
            lower, upper = _distance_bounds(point, candidate)
            return _Hit(
                2,
                face,
                _NAN3.copy(),
                _NAN2.copy(),
                lower,
                upper,
                _NON_ISOLATED,
                _NAN3.copy(),
                _NAN_FRAME3.copy(),
                path,
            )
        tube_angle = float(np.arctan2(tube_y, tube_x))
        ring_radius = major + minor * tube_x / tube_length
        height = minor * tube_y / tube_length
        if radial == 0.0:
            bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
            if (
                not _full_chart_rectangle(model, face)
                or not _full_period(model, face, 0)
                or not _full_period(model, face, 1)
            ):
                return _unresolved(2, face, path)
            candidate = center + ring_radius * first + height * axis
            lower, upper = _distance_bounds(point, candidate)
            return _Hit(
                2,
                face,
                _NAN3.copy(),
                _NAN2.copy(),
                lower,
                upper,
                _NON_ISOLATED,
                _NAN3.copy(),
                _NAN_FRAME3.copy(),
                path,
            )
        longitude = float(np.arctan2(y, x))
        uv = np.asarray((longitude, tube_angle), dtype=np.float64)
        radial_direction = np.cos(longitude) * first + np.sin(longitude) * second
        candidate = center + ring_radius * radial_direction + height * axis
    elif isinstance(patch, ExtrusionSurface) and isinstance(
        patch.curve, (LineCurve, ExactLineCurve)
    ):
        if affine_frame is None:
            return _unresolved(2, face, path)
        projection = _project_affine_frame(affine_frame, point)
        if projection is None:
            return _unresolved(2, face, path)
        exact_uv, exact_candidate, affine_distance_squared = projection
        uv = np.asarray(tuple(float(value) for value in exact_uv), dtype=np.float64)
        candidate = np.asarray(
            tuple(float(value) for value in exact_candidate), dtype=np.float64
        )
        if not np.all(np.isfinite(uv)) or not np.all(np.isfinite(candidate)):
            return _unresolved(2, face, path)
    elif isinstance(patch, RevolutionSurface) and isinstance(
        patch.curve, (LineCurve, ExactLineCurve)
    ):
        from .._rational import rational_vector

        axis_origin = np.asarray(patch.axis_origin, dtype=np.float64)
        axis = np.asarray(patch.axis_direction, dtype=np.float64)
        if isinstance(patch.curve, LineCurve):
            line_origin = np.asarray(patch.curve.origin, dtype=np.float64)
            line_direction = np.asarray(patch.curve.direction, dtype=np.float64)
            exact_direction = patch.curve.direction_exact
            exact_line_origin = patch.curve.origin_exact
        else:
            line_origin = np.asarray(patch.curve.first, dtype=np.float64)
            line_direction = np.asarray(
                patch.curve.last - patch.curve.first, dtype=np.float64
            )
            exact_line_origin = patch.curve.first_exact
            exact_direction = tuple(
                last - first
                for first, last in zip(
                    patch.curve.first_exact, patch.curve.last_exact, strict=True
                )
            )
        exact_axis = rational_vector(axis)
        exact_axis_origin = rational_vector(axis_origin)
        exact_origin = tuple(
            first - second
            for first, second in zip(exact_line_origin, exact_axis_origin, strict=True)
        )
        dot_exact = lambda a, b: sum(
            (x * y for x, y in zip(a, b, strict=True)), Fraction(0)
        )
        if dot_exact(exact_axis, exact_axis) != 1:
            return _unresolved(2, face, path)
        axial_start = float((line_origin - axis_origin) @ axis)
        radial_start = line_origin - axis_origin - axial_start * axis
        radial_rate = line_direction - float(line_direction @ axis) * axis
        exact_axial_rate = dot_exact(exact_direction, exact_axis)
        exact_radial_start = tuple(
            value - dot_exact(exact_origin, exact_axis) * axis_value
            for value, axis_value in zip(exact_origin, exact_axis, strict=True)
        )
        exact_radial_rate = tuple(
            value - exact_axial_rate * axis_value
            for value, axis_value in zip(exact_direction, exact_axis, strict=True)
        )
        cross = (
            exact_radial_start[1] * exact_radial_rate[2]
            - exact_radial_start[2] * exact_radial_rate[1],
            exact_radial_start[2] * exact_radial_rate[0]
            - exact_radial_start[0] * exact_radial_rate[2],
            exact_radial_start[0] * exact_radial_rate[1]
            - exact_radial_start[1] * exact_radial_rate[0],
        )
        if any(value != 0 for value in cross):
            return _unresolved(2, face, path)
        radial_norm = float(np.linalg.norm(radial_start))
        rate_norm = float(np.linalg.norm(radial_rate))
        if radial_norm > 0.0:
            radial_axis = radial_start / radial_norm
        elif rate_norm > 0.0:
            radial_axis = radial_rate / rate_norm
        else:
            return _unresolved(2, face, path)
        second_axis = np.cross(axis, radial_axis)
        first_parameter, last_parameter = (
            float(value) for value in np.asarray(model.parameter_bounds[face])[:, 1]
        )
        profile_first = line_origin + first_parameter * line_direction
        profile_last = line_origin + last_parameter * line_direction
        relative_first = profile_first - axis_origin
        relative_last = profile_last - axis_origin
        z_first, z_last = float(relative_first @ axis), float(relative_last @ axis)
        rho_first = float(np.linalg.norm(relative_first - z_first * axis))
        rho_last = float(np.linalg.norm(relative_last - z_last * axis))
        relative = point - axis_origin
        axial_point = float(relative @ axis)
        radial_point = relative - axial_point * axis
        rho_point = float(np.linalg.norm(radial_point))
        delta = np.asarray((rho_last - rho_first, z_last - z_first))
        squared = float(delta @ delta)
        if squared == 0.0:
            return _unresolved(2, face, path)
        phase = np.asarray((rho_point - rho_first, axial_point - z_first))
        fraction = float(np.clip((phase @ delta) / squared, 0.0, 1.0))
        profile_parameter = first_parameter + fraction * (
            last_parameter - first_parameter
        )
        if rho_point == 0.0:
            bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
            angle = float(bounds[0, 0])
        else:
            angle = float(
                np.arctan2(radial_point @ second_axis, radial_point @ radial_axis)
            )
        uv = np.asarray((angle, profile_parameter), dtype=np.float64)
        candidate = np.asarray(
            patch.evaluate(jnp.asarray(uv, dtype=jnp.float64)), dtype=np.float64
        )
        candidate_relative = candidate - axis_origin
        candidate_height = float(candidate_relative @ axis)
        nonisolated = (
            rho_point == 0.0
            and float(np.linalg.norm(candidate_relative - candidate_height * axis)) > 0.0
        )
        rectangle_closest = True
    elif isinstance(patch, RevolutionSurface) and isinstance(patch.curve, CircleCurve):
        projected = _circle_revolution_closest(
            patch,
            np.asarray(model.parameter_bounds[face], dtype=np.float64),
            point,
        )
        if projected is None:
            return _unresolved(2, face, path)
        uv, candidate, nonisolated = projected
        rectangle_closest = True
    else:
        return _unresolved(2, face, path)
    if uv is None or candidate is None:
        return _unresolved(2, face, path)
    bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    periods = patch.periods
    for axis, period in enumerate(periods):
        if period is not None:
            representative = _periodic_parameter(
                float(uv[axis]), bounds[0, axis], bounds[1, axis], period
            )
            if representative is None:
                return _unresolved(2, face, path)
            uv[axis] = representative
    outside_bounds = np.any(uv < bounds[0]) or np.any(uv > bounds[1])
    if (
        rectangle_closest
        and not outside_bounds
        and (_natural_rectangle_trim(model, face) or _full_chart_rectangle(model, face))
    ):
        # The face is its full rectangle; the rectangle-closest point is the
        # face-closest point even where it lies on a seam or rim edge.
        trim = 1
    else:
        trim = 0 if outside_bounds else _trim_classification(model, face, uv)
    if trim is None or trim == 0:
        if (
            isinstance(patch, PlanePatch)
            or affine_spline
            or (
                isinstance(patch, ExtrusionSurface)
                and isinstance(patch.curve, (LineCurve, ExactLineCurve))
            )
        ):
            boundary_hit = _plane_boundary_hit(
                model,
                face,
                point,
                patch,
                path,
                orientation,
                policy,
                embedding=embedding,
            )
            if boundary_hit is not None and trim == 0:
                return boundary_hit
            if boundary_hit is not None:
                # An undecided trim point lies on or beside a curved boundary.
                # The face lies in its plane and contains its boundary, so its
                # distance lies between the plane's and the boundary point's;
                # within the tie band that boundary point is the closest.
                plane_lower = (
                    _distance_bounds(point, candidate)[0]
                    if affine_distance_squared is None
                    else _affine_distance_bounds(affine_distance_squared)[0]
                )
                if boundary_hit.upper - plane_lower <= policy.ambiguity_tolerance:
                    return _Hit(
                        2,
                        face,
                        boundary_hit.point,
                        boundary_hit.parameters,
                        min(plane_lower, boundary_hit.lower),
                        boundary_hit.upper,
                        _SEAM,
                        boundary_hit.normal,
                        boundary_hit.tangents,
                        path,
                    )
        return _unresolved(2, face, path)
    if nonisolated and trim > 0:
        if affine_distance_squared is None:
            lower, upper = _distance_bounds(point, candidate)
        else:
            lower, upper = _affine_distance_bounds(affine_distance_squared)
        return _Hit(
            2,
            face,
            _NAN3.copy(),
            _NAN2.copy(),
            lower,
            upper,
            _NON_ISOLATED,
            _NAN3.copy(),
            _NAN_FRAME3.copy(),
            path,
        )
    normal, tangents, regular = _face_normal_and_frame(
        patch,
        uv,
        orientation,
        embedding=embedding,
        profile_interval=(float(bounds[0, 1]), float(bounds[1, 1])),
    )
    if affine_distance_squared is None:
        lower, upper = _distance_bounds(point, candidate)
    else:
        lower, upper = _affine_distance_bounds(affine_distance_squared)
    if not regular and affine_distance_squared is not None:
        return _Hit(
            2,
            face,
            _NAN3.copy(),
            _NAN2.copy(),
            lower,
            upper,
            _UNRESOLVED,
            _NAN3.copy(),
            _NAN_FRAME3.copy(),
            path,
        )
    status = _SINGULAR if not regular else (_SEAM if trim < 0 else _UNIQUE)
    seam_tolerance = (
        128.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(uv))))
    )
    if status != _SINGULAR and any(
        period is not None and abs(uv[axis] - bounds[0, axis]) <= seam_tolerance
        for axis, period in enumerate(periods)
    ):
        status = _SEAM
    if embedding is not None:
        candidate = embedding.to_planar(candidate)
    return _Hit(2, face, candidate, uv, lower, upper, status, normal, tangents, path)


def _ray_cylinder_crossings(
    model: BRepModel,
    face: int,
    ray_origin: tuple[Fraction, Fraction, Fraction],
    ray_direction: tuple[Fraction, Fraction, Fraction],
    /,
) -> int | None:
    """Count transversal crossings of an exact ray with a trimmed cylinder face.

    The ray meets the infinite cylinder where ``A t^2 + B t + C = 0`` with
    exact rational coefficients; the discriminant and the signs of both roots
    are decided exactly. Each positive root is then classified against the
    face trim, and a tangent ray, a ray starting on the cylinder or a crossing
    on a trim boundary refuses.
    """
    patch = model.patches[face]
    if not isinstance(patch, CylinderPatch):
        return None
    vectors = (
        patch.origin_exact,
        patch.first_axis_exact,
        patch.second_axis_exact,
        patch.axis_exact,
    )
    if any(
        not all(isinstance(value, Fraction) for value in vector) for vector in vectors
    ) or not isinstance(patch.radius_exact, Fraction):
        return None
    origin, first, second, axis = (
        cast(tuple[Fraction, ...], vector) for vector in vectors
    )
    radius = patch.radius_exact

    def dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...], /) -> Fraction:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    axis_squared = dot(axis, axis)
    if (
        radius <= 0
        or axis_squared == 0
        or dot(first, first) != 1
        or dot(second, second) != 1
        or dot(first, second) != 0
        or dot(first, axis) != 0
        or dot(second, axis) != 0
    ):
        return None
    relative = tuple(ray_origin[index] - origin[index] for index in range(3))
    position = (dot(relative, first), dot(relative, second))
    rate = (dot(ray_direction, first), dot(ray_direction, second))
    quadratic = rate[0] * rate[0] + rate[1] * rate[1]
    linear = 2 * (position[0] * rate[0] + position[1] * rate[1])
    constant = position[0] * position[0] + position[1] * position[1] - radius * radius
    if quadratic == 0:
        # A ray parallel to the axis never crosses unless it lies on the wall.
        return None if constant == 0 else 0
    if constant == 0:
        return None
    discriminant = linear * linear - 4 * quadratic * constant
    if discriminant < 0:
        return 0
    if discriminant == 0:
        return None
    # With A > 0: t+ > 0 iff B < 0 or C < 0, and t- > 0 iff B < 0 and C > 0.
    roots: list[int] = []
    if linear < 0 or constant < 0:
        roots.append(1)
    if linear < 0 and constant > 0:
        roots.append(-1)
    if not roots:
        return 0
    bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    period = patch.periods[0]
    if period is None:
        return None
    root_discriminant = sqrt(float(discriminant))
    crossings = 0
    for sign in roots:
        parameter = (-float(linear) + sign * root_discriminant) / (2.0 * float(quadratic))
        x = float(position[0]) + parameter * float(rate[0])
        y = float(position[1]) + parameter * float(rate[1])
        height = (
            float(dot(relative, axis)) + parameter * float(dot(ray_direction, axis))
        ) / float(axis_squared)
        angle = float(np.arctan2(y, x))
        middle = 0.5 * (float(bounds[0, 0]) + float(bounds[1, 0]))
        angle += round((middle - angle) / float(period)) * float(period)
        uv = np.asarray((angle, height), dtype=np.float64)
        if not np.all(np.isfinite(uv)):
            return None
        trim = _trim_classification(model, face, uv)
        if trim is None or trim < 0:
            return None
        crossings += trim
    return crossings


_ExactVector = tuple[ExactScalar, ExactScalar, ExactScalar]


def _exact_dot(
    first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...], /
) -> ExactScalar:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _exact_cross(
    first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...], /
) -> _ExactVector:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _exact_sign(value: ExactScalar, /) -> int:
    return int(value > 0) - int(value < 0)


def _sign_of_root_sum(terms: tuple[tuple[ExactScalar, ExactScalar], ...], /) -> int:
    """Exact sign of ``sum(c * sqrt(p))`` over at most three ``(c, p)``, ``p >= 0``."""
    active = tuple((c, p) for c, p in terms if c != 0 and p != 0)
    if not active:
        return 0
    if len(active) == 1:
        return _exact_sign(active[0][0])
    if len(active) == 2:
        (c1, p1), (c2, p2) = active
        first, second = _exact_sign(c1), _exact_sign(c2)
        if first == second:
            return first
        return first * _exact_sign(c1 * c1 * p1 - c2 * c2 * p2)
    if len(active) != 3:
        raise ValueError("Root-sum signs take at most three terms.")
    (c1, p1), (c2, p2), (c3, p3) = active
    head = _sign_of_root_sum(((c1, p1), (c2, p2)))
    tail = _exact_sign(c3)
    if head == 0:
        return tail
    if head == tail:
        return head
    # sign(S + t) = sign(S) sign(S^2 - t^2) for opposite signs, with
    # S^2 - t^2 = (c1^2 p1 + c2^2 p2 - c3^2 p3) + 2 c1 c2 sqrt(p1 p2).
    return head * _sign_of_root_sum(
        (
            (c1 * c1 * p1 + c2 * c2 * p2 - c3 * c3 * p3, Fraction(1)),
            (2 * c1 * c2, p1 * p2),
        )
    )


@dataclass(frozen=True)
class _ProfileLine:
    """Profile segment in meridian form: exact end heights and radial vectors.

    Heights and radial vectors are affine along the segment; the radial vectors
    are parallel and point to one side of the axis.
    """

    heights: tuple[ExactScalar, ExactScalar]
    radials: tuple[_ExactVector, _ExactVector]


@dataclass(frozen=True)
class _ArcEnd:
    """End of a height-monotone circle piece.

    ``exact`` is the exact height of a turning point (top or bottom of the
    profile circle); a trim end carries only its binary64 ``height``.
    """

    height: float
    exact: ExactScalar | None


@dataclass(frozen=True)
class _ProfileArc:
    """Height-monotone piece of a profile circle in the meridian half-plane.

    At height ``h`` its distance from the axis is
    ``sqrt(center_radial_squared) + side * sqrt(radius_squared - (h - center_height)^2)``.
    """

    center_height: ExactScalar
    center_radial_squared: ExactScalar
    radius_squared: ExactScalar
    side: int
    ends: tuple[_ArcEnd, _ArcEnd]
    tolerance: float


@dataclass(frozen=True)
class _RevolutionProfile:
    """Meridian profile of a solid bounded only by full-turn revolution faces."""

    origin: _ExactVector
    axis: _ExactVector
    lines: tuple[_ProfileLine, ...]
    arcs: tuple[_ProfileArc, ...]


def _profile_line(
    curve: LineCurve | ExactLineCurve,
    bounds: np.ndarray,
    origin: _ExactVector,
    axis: _ExactVector,
    /,
) -> _ProfileLine | None:
    if isinstance(curve, LineCurve):
        start, direction = curve.origin_exact, curve.direction_exact
    else:
        start = curve.first_exact
        direction = tuple(
            last - first
            for first, last in zip(curve.first_exact, curve.last_exact, strict=True)
        )
    ends = tuple(
        tuple(
            start[index] + parameter * direction[index] - origin[index]
            for index in range(3)
        )
        for parameter in (Fraction(float(value)) for value in bounds[:, 1])
    )
    heights = (_exact_dot(ends[0], axis), _exact_dot(ends[1], axis))
    radials = tuple(
        (
            end[0] - height * axis[0],
            end[1] - height * axis[1],
            end[2] - height * axis[2],
        )
        for end, height in zip(ends, heights, strict=True)
    )
    # A profile segment lies in one meridian half-plane.
    if any(value != 0 for value in _exact_cross(radials[0], radials[1])) or (
        _exact_dot(radials[0], radials[1]) < 0
    ):
        return None
    return _ProfileLine(heights, (radials[0], radials[1]))


def _profile_arcs(
    patch: RevolutionSurface,
    curve: CircleCurve,
    bounds: np.ndarray,
    origin: _ExactVector,
    axis: _ExactVector,
    /,
) -> tuple[_ProfileArc, ...] | None:
    """Split a profile circle (arc) into height-monotone meridian pieces.

    The circle plane must contain the axis exactly. In the meridian frame
    ``(m, a)`` a circle point sits at angle ``theta = turn t + phase`` from the
    center; turning points ``theta = pi/2 + k pi`` have exact heights, trim
    ends of a partial arc binary64 ones.
    """
    center = tuple(
        value - base for value, base in zip(curve.center_exact, origin, strict=True)
    )
    normal = _exact_cross(curve.first_axis_exact, curve.second_axis_exact)
    if _exact_dot(normal, axis) != 0 or _exact_dot(normal, center) != 0:
        return None
    center_height = _exact_dot(center, axis)
    center_radial = tuple(
        center[index] - center_height * axis[index] for index in range(3)
    )
    center_radial_squared = _exact_dot(center_radial, center_radial)
    radius = float(curve.radius_exact)
    axis_ = np.asarray([float(value) for value in axis], dtype=np.float64)
    first = np.asarray(
        [float(value) for value in curve.first_axis_exact], dtype=np.float64
    )
    second = np.asarray(
        [float(value) for value in curve.second_axis_exact], dtype=np.float64
    )
    start, stop = (float(value) for value in bounds[:, 1])
    period = patch.periods[1]
    whole = period is not None and Fraction(stop) - Fraction(start) == Fraction(
        float(period)
    )
    if center_radial_squared != 0:
        meridian = np.asarray([float(value) for value in center_radial], dtype=np.float64)
    else:
        middle = 0.5 * (start + stop)
        offset = np.cos(middle) * first + np.sin(middle) * second
        meridian = offset - float(offset @ axis_) * axis_
    length = float(np.linalg.norm(meridian))
    if not length > 1.0e-6 * max(1.0, radius):
        return None
    meridian /= length
    orientation = float(
        (first @ meridian) * (second @ axis_) - (second @ meridian) * (first @ axis_)
    )
    if abs(abs(orientation) - 1.0) > 1.0e-9:
        return None
    turn = 1.0 if orientation > 0.0 else -1.0
    phase = float(np.arctan2(first @ axis_, first @ meridian))
    if whole:
        low, high = -0.5 * pi, 1.5 * pi
    else:
        low, high = sorted((phase + turn * start, phase + turn * stop))
    snap = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(low), abs(high))
    turns = [
        0.5 * pi + k * pi
        for k in range(
            ceil((low - snap - 0.5 * pi) / pi),
            floor((high + snap - 0.5 * pi) / pi) + 1,
        )
    ]
    angles: list[tuple[float, int | None]] = []
    if not turns or abs(turns[0] - low) > snap:
        angles.append((low, None))
    angles.extend((angle, round((angle - 0.5 * pi) / pi)) for angle in turns)
    if not turns or abs(turns[-1] - high) > snap:
        angles.append((high, None))
    center_height_ = float(center_height)
    center_distance = sqrt(float(center_radial_squared))
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(center_height_), center_distance, radius)
    )
    # The arc must stay in the half-plane of its meridian direction.
    lowest = min(float(np.cos(angle)) for angle, _ in angles)
    if any(low < pi + 2.0 * k * pi < high for k in range(-2, 3)):
        lowest = -1.0
    if center_distance + radius * lowest < -tolerance:
        return None
    radius_exact = curve.radius_exact

    def end(angle: float, k: int | None, /) -> _ArcEnd:
        if k is None:
            return _ArcEnd(center_height_ + radius * float(np.sin(angle)), None)
        exact = center_height + (radius_exact if k % 2 == 0 else -radius_exact)
        return _ArcEnd(float(exact), exact)

    pieces: list[_ProfileArc] = []
    for (first_angle, first_k), (last_angle, last_k) in zip(
        angles, angles[1:], strict=False
    ):
        cosine = float(np.cos(0.5 * (first_angle + last_angle)))
        if abs(cosine) < 1.0e-12:
            return None
        pieces.append(
            _ProfileArc(
                center_height,
                center_radial_squared,
                radius_exact * radius_exact,
                1 if cosine > 0.0 else -1,
                (end(first_angle, first_k), end(last_angle, last_k)),
                tolerance,
            )
        )
    return tuple(pieces)


def _closed_revolution_profile(
    model: BRepModel, solid: int, /
) -> _RevolutionProfile | None:
    """Meridian profile of a solid whose faces all turn a line or circle fully.

    Every face must be a full-turn revolution about one exact unit axis of a
    line or circle profile lying in a plane through the axis.
    """
    from .._rational import rational_vector

    origin: _ExactVector | None = None
    axis: _ExactVector | None = None
    lines: list[_ProfileLine] = []
    arcs: list[_ProfileArc] = []
    for face in model.topology.solid_faces[solid]:
        patch = model.patches[face]
        if (
            not isinstance(patch, RevolutionSurface)
            or patch.native_period_symbols()[0] != "two_pi"
        ):
            return None
        bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
        period = patch.periods[0]
        if (
            period is None
            or not np.isfinite(period)
            or Fraction(float(bounds[1, 0])) - Fraction(float(bounds[0, 0]))
            != Fraction(float(period))
        ):
            return None
        face_origin = rational_vector(np.asarray(patch.axis_origin, dtype=np.float64))
        face_axis = rational_vector(np.asarray(patch.axis_direction, dtype=np.float64))
        if len(face_origin) != 3 or len(face_axis) != 3:
            return None
        face_origin_ = (face_origin[0], face_origin[1], face_origin[2])
        face_axis_ = (face_axis[0], face_axis[1], face_axis[2])
        if origin is None or axis is None:
            if _exact_dot(face_axis_, face_axis_) != 1:
                return None
            origin, axis = face_origin_, face_axis_
        elif face_axis_ != axis or any(
            value != 0
            for value in _exact_cross(
                tuple(a - b for a, b in zip(face_origin_, origin, strict=True)), axis
            )
        ):
            return None
        curve = patch.curve
        if isinstance(curve, (LineCurve, ExactLineCurve)):
            line = _profile_line(curve, bounds, origin, axis)
            if line is None:
                return None
            lines.append(line)
        elif isinstance(curve, CircleCurve):
            pieces = _profile_arcs(patch, curve, bounds, origin, axis)
            if pieces is None:
                return None
            arcs.extend(pieces)
        else:
            return None
    if origin is None or axis is None:
        return None
    return _RevolutionProfile(origin, axis, tuple(lines), tuple(arcs))


def _revolution_profile_contains(
    profile: _RevolutionProfile, point: np.ndarray, /
) -> bool | None:
    """Whether ``point`` lies inside a closed revolution solid; ``None`` on its boundary.

    The point is inside iff its meridian image ``(rho, h)`` lies inside the
    profile region: the ray ``{(r, h) : r > rho}`` crosses the profile curves an
    odd number of times (half-open rule at piece ends). Distances compare
    exactly through their squares and square-root sums; a query within the
    binary64 tolerance of a trimmed arc end height is unresolved.
    """
    from .._rational import rational_vector

    values = rational_vector(np.asarray(point, dtype=np.float64))
    relative = tuple(
        value - base for value, base in zip(values, profile.origin, strict=True)
    )
    height = _exact_dot(relative, profile.axis)
    radial_squared = _exact_dot(relative, relative) - height * height
    crossings = 0
    for line in profile.lines:
        (first, last), (start, stop) = line.heights, line.radials
        squared = (_exact_dot(start, start), _exact_dot(stop, stop))
        if (first == height and squared[0] == radial_squared) or (
            last == height and squared[1] == radial_squared
        ):
            return None
        if first == last:
            if first == height and min(squared) <= radial_squared <= max(squared):
                return None
            continue
        if (first > height) == (last > height):
            continue
        fraction = (height - first) / (last - first)
        hit = tuple(
            start[index] + fraction * (stop[index] - start[index]) for index in range(3)
        )
        hit_squared = _exact_dot(hit, hit)
        if hit_squared == radial_squared:
            return None
        crossings += int(hit_squared > radial_squared)
    approximate = float(height)
    for arc in profile.arcs:
        above: list[bool] = []
        for end in arc.ends:
            if end.exact is None:
                if abs(end.height - approximate) <= arc.tolerance:
                    return None
                above.append(end.height > approximate)
                continue
            if end.exact == height and (
                _sign_of_root_sum(
                    (
                        (Fraction(1), arc.center_radial_squared),
                        (Fraction(-1), radial_squared),
                    )
                )
                == 0
            ):
                return None
            above.append(end.exact > height)
        if above[0] == above[1]:
            continue
        offset = height - arc.center_height
        discriminant = arc.radius_squared - offset * offset
        if discriminant < 0:
            return None
        sign = _sign_of_root_sum(
            (
                (Fraction(1), arc.center_radial_squared),
                (Fraction(arc.side), discriminant),
                (Fraction(-1), radial_squared),
            )
        )
        if sign == 0:
            return None
        crossings += int(sign > 0)
    return bool(crossings % 2)


@final
class NativeBRepProjection(AbstractBRepProjection):
    """Source-evaluating closest-point and point-location queries without tessellation."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    policy: BRepProjectionPolicy = eqx.field(static=True)
    embedding: PlanarEmbedding | None = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    entity_counts: tuple[int, int, int, int] = eqx.field(static=True)
    closure_codes: Array
    entity_boxes: Array
    edge_degenerate: Array
    affine_projection_frames: tuple[_AffineProjectionFrame | None, ...] = eqx.field(
        static=True
    )
    projection_id: str = eqx.field(static=True)
    model: BRepModel
    occurrences: tuple[_Occurrence, ...] = eqx.field(static=True)
    boundary_entities: tuple[_BoundaryEntity, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: BRepModel,
        policy: BRepProjectionPolicy,
        embedding: PlanarEmbedding | None,
        /,
    ) -> None:
        if not isinstance(model, BRepModel):
            raise TypeError("model must be a BRepModel.")
        if not isinstance(policy, BRepProjectionPolicy):
            raise TypeError("policy must be a BRepProjectionPolicy.")
        affine_projection_frames = tuple(
            _prepare_affine_projection_frame(patch) for patch in model.patches
        )
        topology = model.topology
        if embedding is not None and topology.num_solids:
            raise ValueError("embedding is only valid for face-only B-Rep sources.")
        if embedding is not None:
            residual = np.asarray(
                embedding.plane_residual(np.asarray(model.geometry.vertex_points))
            )
            if np.max(np.abs(residual), initial=0.0) > policy.classifier_tolerance:
                raise ValueError(
                    "embedding must contain every source vertex within classifier_tolerance."
                )
        occurrences: list[_Occurrence] = []
        if topology.num_solids:
            for item in model.geometry.assembly_graph.iter_occurrences():
                matrix = np.asarray(
                    [[float(value) for value in row] for row in item.placement.matrix],
                    dtype=np.float64,
                )
                rotation_matrix = matrix[:3, :3]
                translation_vector = matrix[:3, 3]
                scale = max(1.0, float(np.max(np.abs(rotation_matrix))))
                tolerance = 256.0 * np.finfo(np.float64).eps * scale
                if (
                    matrix.shape != (4, 4)
                    or not np.all(np.isfinite(matrix))
                    or not np.allclose(
                        rotation_matrix.T @ rotation_matrix,
                        np.eye(3),
                        rtol=0.0,
                        atol=tolerance,
                    )
                    or not np.isclose(
                        np.linalg.det(rotation_matrix), 1.0, rtol=0.0, atol=tolerance
                    )
                    or not np.array_equal(matrix[3], np.asarray((0.0, 0.0, 0.0, 1.0)))
                ):
                    rotation_values = ((np.nan, np.nan, np.nan),) * 3
                    translation_values = (np.nan, np.nan, np.nan)
                else:
                    rotation_values = tuple(
                        tuple(float(value) for value in row) for row in rotation_matrix
                    )
                    translation_values = tuple(
                        float(value) for value in translation_vector
                    )
                occurrences.append(
                    _Occurrence(
                        item.path,
                        item.solid_index,
                        rotation_values,
                        translation_values,
                    )
                )
        occurrence_rows = tuple(sorted(occurrences, key=lambda value: value.path))
        counts = (
            topology.num_vertices,
            topology.num_edges,
            topology.num_faces,
            topology.num_solids,
        )
        total = sum(counts)
        boxes = np.empty((total, 2, 3), dtype=np.float64)
        boxes[:, 0, :] = -np.inf
        boxes[:, 1, :] = np.inf
        vertices = np.asarray(model.geometry.vertex_points)
        if vertices.size:
            boxes[: counts[0], 0] = vertices
            boxes[: counts[0], 1] = vertices
        closure = closure_codes(
            topology,
            model.geometry.edge_vertices,
            counts,
        )
        self.source_id = model.source_id
        self.source_revision = model.source_revision
        self.model_id = model.model_id
        self.frame_id = model.coordinate_contract.spatial_id
        self.policy = policy
        self.embedding = embedding
        self.ambient_dimension = 2 if embedding is not None else 3
        self.entity_counts = counts
        self.closure_codes = jnp.asarray(closure, dtype=jnp.int64)
        self.entity_boxes = jnp.asarray(boxes, dtype=jnp.float64)
        self.edge_degenerate = jnp.asarray(
            np.asarray(model.geometry.edge_curves, dtype=np.int32) < 0,
            dtype=jnp.bool_,
        )
        self.affine_projection_frames = affine_projection_frames
        self.model = model
        self.occurrences = occurrence_rows
        boundary_entities: list[_BoundaryEntity] = []
        for dimension, count in enumerate(counts[:3]):
            dimensions = np.full((count,), dimension, dtype=np.int8)
            indices = np.arange(count, dtype=np.int64)
            for index, paths in enumerate(
                self.occurrence_paths_for_entities(dimensions, indices)
            ):
                for path in paths:
                    boundary_entities.append(
                        _BoundaryEntity(
                            brep_entity_id(
                                model.source_id,
                                model.source_revision,
                                dimension,
                                index,
                                occurrence_path=path,
                            ),
                            dimension,
                            index,
                            path,
                        )
                    )
        self.boundary_entities = tuple(boundary_entities)
        self.projection_id = canonical_fingerprint(
            {
                "kind": "native-brep-projection",
                "source_id": model.source_id,
                "source_revision": model.source_revision,
                "model_id": model.model_id,
                "affine_frame_solver": "exact-prepared-lu-v1",
                "policy_id": policy.policy_id,
                "embedding": None if embedding is None else embedding.embedding_id,
                "frame_id": self.frame_id,
                "occurrences": tuple(
                    (value.path, value.solid, value.rotation, value.translation)
                    for value in occurrence_rows
                ),
            }
        )

    def _world(self, points: np.ndarray, /) -> np.ndarray:
        return points if self.embedding is None else self.embedding.to_world(points)

    def _ambient(self, points: np.ndarray, /) -> np.ndarray:
        return points if self.embedding is None else self.embedding.to_planar(points)

    def _belongs(self, dimension: int, index: int, solid: int, /) -> bool:
        geometry = self.model.geometry
        if dimension == BRepEntityDimension.SOLID:
            return index == solid
        if dimension == BRepEntityDimension.FACE:
            return index in self.model.topology.solid_faces[solid]
        if dimension == BRepEntityDimension.EDGE:
            return any(
                index in self.model.topology.face_edges[face]
                for face in self.model.topology.solid_faces[solid]
            )
        if dimension == BRepEntityDimension.VERTEX:
            edges = {
                edge
                for face in self.model.topology.solid_faces[solid]
                for edge in self.model.topology.face_edges[face]
            }
            return any(index in geometry.edge_vertices[edge] for edge in edges)
        return False

    def _target_occurrences(
        self,
        dimension: int,
        index: int,
        occurrence_path: tuple[str, ...] | None,
        /,
    ) -> tuple[_Occurrence, ...]:
        definition_only = self.entity_counts[3] == 0 or (
            dimension < BRepEntityDimension.SOLID
            and not any(
                self._belongs(dimension, index, solid)
                for solid in range(self.entity_counts[3])
            )
        )
        if occurrence_path == ():
            if definition_only:
                return (_Occurrence((), -1, _IDENTITY_ROTATION, _ZERO_TRANSLATION),)
            raise ValueError(
                "An empty occurrence path cannot select an instantiated solid entity."
            )
        available = tuple(
            value
            for value in self.occurrences
            if occurrence_path is None or value.path == occurrence_path
            if self._belongs(dimension, index, value.solid)
        )
        if occurrence_path is not None and not available:
            raise ValueError("occurrence_path must own the requested source entity.")
        if available:
            return available
        if occurrence_path is not None:
            raise ValueError("occurrence_path does not belong to the B-Rep source.")
        if definition_only:
            return (_Occurrence((), -1, _IDENTITY_ROTATION, _ZERO_TRANSLATION),)
        return ()

    def occurrence_paths_for_entities(
        self,
        dimensions: ArrayLike,
        indices: ArrayLike,
        /,
    ) -> tuple[tuple[tuple[str, ...], ...], ...]:
        """Return exact owning occurrence paths for each definition entity row."""
        dims = np.asarray(dimensions)
        rows = np.asarray(indices)
        if not np.issubdtype(dims.dtype, np.integer) or not np.issubdtype(
            rows.dtype, np.integer
        ):
            raise TypeError("Entity dimensions and indices must be integer arrays.")
        if dims.ndim != 1 or rows.shape != dims.shape:
            raise ValueError("Entity dimensions and indices must be aligned vectors.")
        if np.any((dims < 0) | (dims > 3)) or any(
            int(index) >= self.entity_counts[int(dimension)]
            for dimension, index in zip(dims, rows, strict=True)
        ):
            raise ValueError("Entity dimension or index lies outside the B-Rep source.")
        return tuple(
            tuple(
                occurrence.path
                for occurrence in self._target_occurrences(
                    int(dimension), int(index), None
                )
            )
            for dimension, index in zip(dims, rows, strict=True)
        )

    def entity_codes(self, dimensions: ArrayLike, indices: ArrayLike, /) -> np.ndarray:
        """Encode aligned definition-entity rows in the canonical topology bank."""
        dims = np.asarray(dimensions)
        rows = np.asarray(indices)
        if not np.issubdtype(dims.dtype, np.integer) or not np.issubdtype(
            rows.dtype, np.integer
        ):
            raise TypeError("Entity dimensions and indices must be integer arrays.")
        if dims.ndim != 1 or rows.shape != dims.shape:
            raise ValueError("Entity dimensions and indices must be aligned vectors.")
        if np.any((dims < 0) | (dims > 3)):
            raise ValueError("Entity dimensions must lie between vertex and solid.")
        if any(
            index < 0 or index >= self.entity_counts[int(dimension)]
            for dimension, index in zip(dims, rows, strict=True)
        ):
            raise ValueError("Entity index lies outside its B-Rep topology bank.")
        offsets = np.cumsum((0, *self.entity_counts), dtype=np.int64)
        return offsets[dims] + rows

    def code_entities(self, codes: ArrayLike, /) -> tuple[np.ndarray, np.ndarray]:
        """Decode canonical topology-bank codes to dimensions and row indices."""
        values = np.asarray(codes)
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("Entity codes must be an integer array.")
        if values.ndim != 1:
            raise ValueError("Entity codes must be a vector.")
        total = sum(self.entity_counts)
        if np.any(values < 0) or np.any(values >= total):
            raise ValueError("Entity code lies outside the B-Rep topology banks.")
        values = values.astype(np.int64, copy=False)
        offsets = np.cumsum((0, *self.entity_counts), dtype=np.int64)
        dimensions = np.searchsorted(offsets[1:], values, side="right").astype(
            np.int64, copy=False
        )
        return dimensions, values - offsets[dimensions]

    def members(
        self, dimensions: ArrayLike, indices: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return each input entity's reflexive definition-topology closure."""
        codes = self.entity_codes(dimensions, indices)
        total = sum(self.entity_counts)
        if not len(codes):
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy(), empty.copy()
        if total == 0:
            raise ValueError("An empty topology has no member entities.")
        encoded = np.asarray(self.closure_codes, dtype=np.int64)
        owners, members = encoded // total, encoded % total
        owner_rows: list[int] = []
        member_codes: list[np.ndarray] = []
        for row, code in enumerate(codes):
            start = int(np.searchsorted(owners, code, side="left"))
            stop = int(np.searchsorted(owners, code, side="right"))
            if stop > start:
                owner_rows.extend([row] * (stop - start))
                member_codes.append(members[start:stop])
        if not member_codes:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy(), empty.copy()
        member_dimensions, member_indices = self.code_entities(
            np.concatenate(member_codes)
        )
        return np.asarray(owner_rows, dtype=np.int64), member_dimensions, member_indices

    def containers(
        self, dimensions: ArrayLike, indices: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return each input entity's reflexive definition-topology containers."""
        codes = self.entity_codes(dimensions, indices)
        total = sum(self.entity_counts)
        if not len(codes):
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy(), empty.copy()
        if total == 0:
            raise ValueError("An empty topology has no container entities.")
        encoded = np.asarray(self.closure_codes, dtype=np.int64)
        owners, members = encoded // total, encoded % total
        order = np.argsort(members, kind="stable")
        sorted_members, sorted_owners = members[order], owners[order]
        owner_rows: list[int] = []
        container_codes: list[np.ndarray] = []
        for row, code in enumerate(codes):
            start = int(np.searchsorted(sorted_members, code, side="left"))
            stop = int(np.searchsorted(sorted_members, code, side="right"))
            if stop > start:
                owner_rows.extend([row] * (stop - start))
                container_codes.append(sorted_owners[start:stop])
        if not container_codes:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy(), empty.copy()
        container_dimensions, container_indices = self.code_entities(
            np.concatenate(container_codes)
        )
        return (
            np.asarray(owner_rows, dtype=np.int64),
            container_dimensions,
            container_indices,
        )

    def contains(
        self,
        container_dimensions: ArrayLike,
        container_indices: ArrayLike,
        child_dimensions: ArrayLike,
        child_indices: ArrayLike,
        /,
    ) -> np.ndarray:
        """Test reflexive definition-topology containment for aligned entity rows.

        Occurrence identity is intentionally outside this relation; callers that
        materialize instances must additionally match their occurrence paths.
        """
        containers = self.entity_codes(container_dimensions, container_indices)
        children = self.entity_codes(child_dimensions, child_indices)
        if containers.shape != children.shape:
            raise ValueError("Container and child entity rows must be aligned.")
        if not len(containers):
            return np.empty(0, dtype=bool)
        total = sum(self.entity_counts)
        encoded = containers * total + children
        return np.isin(encoded, np.asarray(self.closure_codes, dtype=np.int64))

    def _face_orientation(self, face: int, solid: int, /) -> float:
        orientation = float(self.model.orientation[face])
        if solid < 0:
            return orientation
        faces = self.model.topology.solid_faces[solid]
        signs = self.model.topology.solid_face_orientations[solid]
        return orientation * signs[faces.index(face)]

    def _project_one(
        self,
        dimension: int,
        index: int,
        query: np.ndarray,
        occurrence_path: tuple[str, ...] | None,
        /,
    ) -> _Hit:
        count = self.entity_counts[dimension]
        if index < 0 or index >= count:
            raise ValueError("Entity index lies outside its B-Rep dimension.")
        targets = self._target_occurrences(dimension, index, occurrence_path)
        if not targets:
            return _failed(self.ambient_dimension)
        hits: list[_Hit] = []
        for occurrence in targets:
            rotation = np.asarray(occurrence.rotation, dtype=np.float64)
            translation = np.asarray(occurrence.translation, dtype=np.float64)
            local = rotation.T @ (query - translation)
            if not np.all(np.isfinite(local)):
                hits.append(_unresolved(dimension, index, occurrence.path))
                continue
            if dimension == BRepEntityDimension.VERTEX:
                point = np.asarray(
                    self.model.geometry.vertex_points[index], dtype=np.float64
                )
                lower, upper = _distance_bounds(local, point)
                hit = _Hit(
                    dimension,
                    index,
                    point,
                    _NAN2.copy(),
                    lower,
                    upper,
                    _UNIQUE,
                    _NAN3.copy(),
                    _NAN_FRAME3.copy(),
                    occurrence.path,
                )
            elif dimension == BRepEntityDimension.EDGE:
                hit = _line_curve_hit(
                    self.model, index, local, occurrence.path, self.policy
                )
            elif dimension == BRepEntityDimension.FACE:
                hit = _project_face(
                    self.model,
                    index,
                    local,
                    occurrence.path,
                    self._face_orientation(index, occurrence.solid),
                    self.policy,
                    embedding=None,
                    affine_frame=self.affine_projection_frames[index],
                )
            else:
                return _failed(self.ambient_dimension)
            if hit.status not in (_FAILED, _UNRESOLVED) and occurrence.path:
                point = (
                    rotation @ hit.point + translation
                    if np.all(np.isfinite(hit.point))
                    else hit.point
                )
                normal = (
                    rotation @ hit.normal
                    if np.all(np.isfinite(hit.normal))
                    else hit.normal
                )
                tangents = (
                    hit.tangents @ rotation.T
                    if np.all(np.isfinite(hit.tangents))
                    else hit.tangents
                )
                hit = _Hit(
                    hit.dimension,
                    hit.index,
                    point,
                    hit.parameters,
                    hit.lower,
                    hit.upper,
                    hit.status,
                    normal,
                    tangents,
                    occurrence.path,
                    hit.piece_id,
                    hit.root_id,
                )
            hits.append(hit)
        if len(hits) == 1:
            hit = hits[0]
            return hit
        if any(hit.status in (_FAILED, _UNRESOLVED) for hit in hits):
            return _unresolved(dimension, index)
        best = min(hits, key=lambda value: value.upper)
        tied = [
            hit
            for hit in hits
            if hit.upper <= best.lower + self.policy.ambiguity_tolerance
            and (
                hit.path != best.path
                or (
                    np.all(np.isfinite(hit.point))
                    and np.all(np.isfinite(best.point))
                    and np.linalg.norm(hit.point - best.point)
                    > self.policy.ambiguity_tolerance
                )
            )
        ]
        if tied:
            return _Hit(
                best.dimension,
                best.index,
                best.point,
                best.parameters,
                min(hit.lower for hit in hits),
                best.upper,
                _AMBIGUOUS,
                best.normal,
                best.tangents,
                best.path,
                best.piece_id,
                best.root_id,
            )
        return _Hit(
            best.dimension,
            best.index,
            best.point,
            best.parameters,
            min(hit.lower for hit in hits),
            best.upper,
            best.status,
            best.normal,
            best.tangents,
            best.path,
            best.piece_id,
            best.root_id,
        )

    def _result(self, queries: np.ndarray, hits: list[_Hit], /) -> BRepProjectionResult:
        dimensions = np.asarray([hit.dimension for hit in hits], dtype=np.int8)
        indices = np.asarray([hit.index for hit in hits], dtype=np.int64)
        points_3d = np.stack([hit.point for hit in hits]) if hits else np.empty((0, 3))
        parameters = (
            np.stack([hit.parameters for hit in hits]) if hits else np.empty((0, 2))
        )
        lower = np.asarray([hit.lower for hit in hits], dtype=np.float64)
        upper = np.asarray([hit.upper for hit in hits], dtype=np.float64)
        status = np.asarray([hit.status for hit in hits], dtype=np.int8)
        normals_3d = np.stack([hit.normal for hit in hits]) if hits else np.empty((0, 3))
        tangents_3d = (
            np.stack([hit.tangents for hit in hits]) if hits else np.empty((0, 2, 3))
        )
        if self.embedding is None:
            points, normals, tangents = points_3d, normals_3d, tangents_3d
        else:
            points = np.full((len(hits), 2), np.nan, dtype=np.float64)
            finite_points = np.all(np.isfinite(points_3d), axis=1)
            if np.any(finite_points):
                points[finite_points] = self.embedding.to_planar(points_3d[finite_points])
            normals = np.full((len(hits), 2), np.nan, dtype=np.float64)
            tangents = np.full((len(hits), 2, 2), np.nan, dtype=np.float64)
            finite_tangents = np.all(np.isfinite(tangents_3d), axis=(1, 2))
            if np.any(finite_tangents):
                basis = np.stack((self.embedding.x_axis, self.embedding.y_axis), axis=1)
                tangents[finite_tangents] = tangents_3d[finite_tangents] @ basis
        occurrence_paths = tuple(hit.path for hit in hits)
        return BRepProjectionResult(
            self.source_id,
            self.source_revision,
            dimensions,
            indices,
            queries,
            points,
            parameters,
            upper,
            status,
            normals,
            tangents,
            distance_lower_bounds=lower,
            frame_id=self.frame_id,
            occurrence_paths=occurrence_paths,
        )

    @checked
    def project(
        self,
        points: ArrayLike,
        dimensions: ArrayLike,
        indices: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        queries = np.asarray(points, dtype=np.float64)
        dims = np.asarray(dimensions)
        rows = np.asarray(indices)
        if (
            queries.ndim != 2
            or queries.shape[1] != self.ambient_dimension
            or not np.all(np.isfinite(queries))
        ):
            raise ValueError("Projection queries must be finite ambient points.")
        if not np.issubdtype(dims.dtype, np.integer) or not np.issubdtype(
            rows.dtype, np.integer
        ):
            raise TypeError("Projection dimensions and indices must be integer arrays.")
        if dims.shape != (queries.shape[0],) or rows.shape != dims.shape:
            raise ValueError(
                "Projection dimensions and indices must align with query rows."
            )
        if np.any((dims < 0) | (dims > 2)):
            raise ValueError(
                "Projection dimensions must select vertices, edges, or faces."
            )
        if occurrence_paths is not None and (
            not isinstance(occurrence_paths, tuple)
            or len(occurrence_paths) != queries.shape[0]
            or any(not isinstance(path, tuple) for path in occurrence_paths)
        ):
            raise ValueError("occurrence_paths must contain one tuple path per query.")
        requested = (
            (None,) * queries.shape[0] if occurrence_paths is None else occurrence_paths
        )
        world = self._world(queries)
        hits = [
            self._project_one(int(dimension), int(index), world[row], requested[row])
            for row, (dimension, index) in enumerate(zip(dims, rows, strict=True))
        ]
        return self._result(queries, hits)

    @checked
    def project_boundary(
        self,
        points: ArrayLike,
        /,
        *,
        entity_ids: tuple[str, ...] | None = None,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        """Project to the complete source boundary or an exact entity scope."""
        queries = np.asarray(points, dtype=np.float64)
        if (
            queries.ndim != 2
            or queries.shape[1] != self.ambient_dimension
            or not np.all(np.isfinite(queries))
        ):
            raise ValueError("Boundary projection queries must be finite ambient points.")
        if entity_ids is None:
            candidates = self.boundary_entities
        else:
            if not isinstance(entity_ids, tuple) or any(
                not isinstance(value, str) or not value or value.strip() != value
                for value in entity_ids
            ):
                raise TypeError(
                    "entity_ids must be a tuple of canonical source entity IDs."
                )
            if len(set(entity_ids)) != len(entity_ids):
                raise ValueError("entity_ids must not repeat a source entity.")
            by_id = {value.entity_id: value for value in self.boundary_entities}
            unknown = tuple(value for value in entity_ids if value not in by_id)
            if unknown:
                raise ValueError(f"entity_ids do not belong to this source: {unknown!r}.")
            candidates = tuple(by_id[value] for value in entity_ids)
        if occurrence_paths is not None and (
            not isinstance(occurrence_paths, tuple)
            or len(occurrence_paths) != queries.shape[0]
            or any(
                not isinstance(path, tuple)
                or any(not isinstance(part, str) or not part for part in path)
                for path in occurrence_paths
            )
        ):
            raise ValueError(
                "occurrence_paths must contain one exact path tuple per query."
            )
        known_paths = {value.path for value in self.occurrences}
        if occurrence_paths is not None and any(
            path not in known_paths and path != () for path in occurrence_paths
        ):
            raise ValueError("occurrence_paths must select an occurrence of this source.")
        world = self._world(queries)
        hits: list[_Hit] = []
        for row, query in enumerate(world):
            path = None if occurrence_paths is None else occurrence_paths[row]
            eligible = tuple(
                entity for entity in candidates if path is None or entity.path == path
            )
            if not eligible:
                if path is not None and candidates:
                    raise ValueError(
                        "occurrence_path does not own the selected source entities."
                    )
                hits.append(_failed(self.ambient_dimension))
                continue
            projected: list[_Hit] = []
            unresolved: list[_Hit] = []
            for entity in eligible:
                hit = self._project_one(
                    entity.dimension,
                    entity.index,
                    query,
                    entity.path,
                )
                if hit.status == _UNRESOLVED:
                    unresolved.append(hit)
                elif hit.status != _FAILED:
                    projected.append(hit)
            if unresolved:
                hits.append(unresolved[0])
                continue
            if not projected:
                hits.append(_failed(self.ambient_dimension))
                continue
            best = min(projected, key=lambda hit: hit.upper)
            lower = min(hit.lower for hit in projected)
            overlaps = [
                hit
                for hit in projected
                if hit is not best
                and hit.lower <= best.upper + self.policy.ambiguity_tolerance
                and best.lower <= hit.upper + self.policy.ambiguity_tolerance
                and (
                    hit.dimension != best.dimension
                    or hit.index != best.index
                    or hit.path != best.path
                )
            ]
            status = (
                _AMBIGUOUS if overlaps and best.status != _NON_ISOLATED else best.status
            )
            hits.append(
                _Hit(
                    best.dimension,
                    best.index,
                    best.point,
                    best.parameters,
                    lower,
                    best.upper,
                    status,
                    best.normal,
                    best.tangents,
                    best.path,
                )
            )
        return self._result(queries, hits)

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        """Return certified physical distance bounds to this source boundary."""
        result = self.project_boundary(points)
        return SourceBoundaryDistance(
            np.asarray(result.distance_lower_bounds),
            np.asarray(result.residuals),
            "certified",
        )

    @staticmethod
    def _choose(
        candidates: list[_Hit],
        unresolved: list[_Hit],
        ambiguity_tolerance: float,
        ambient: int,
        /,
    ) -> _Hit:
        if unresolved:
            return unresolved[0]
        if not candidates:
            return _failed(ambient)
        dimension = min(hit.dimension for hit in candidates)
        candidates = [hit for hit in candidates if hit.dimension == dimension]
        best = min(candidates, key=lambda hit: hit.upper)
        ties = [
            hit
            for hit in candidates
            if hit.upper <= best.lower + ambiguity_tolerance
            and (
                hit.index != best.index
                or hit.path != best.path
                or (
                    np.all(np.isfinite(hit.point))
                    and np.all(np.isfinite(best.point))
                    and np.linalg.norm(hit.point - best.point) > ambiguity_tolerance
                )
            )
        ]
        if ties:
            return _Hit(
                best.dimension,
                best.index,
                best.point,
                best.parameters,
                min(hit.lower for hit in candidates),
                best.upper,
                _AMBIGUOUS,
                best.normal,
                best.tangents,
                best.path,
            )
        return _Hit(
            best.dimension,
            best.index,
            best.point,
            best.parameters,
            min(hit.lower for hit in candidates),
            best.upper,
            best.status,
            best.normal,
            best.tangents,
            best.path,
        )

    @checked
    def classify(
        self,
        points: ArrayLike,
        /,
        *,
        tolerance: float,
        maximum_dimension: int = 2,
        groups: ArrayLike | None = None,
        allowed_groups: ArrayLike | None = None,
        allowed_dimensions: ArrayLike | None = None,
        allowed_indices: ArrayLike | None = None,
        allowed_occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        queries = np.asarray(points, dtype=np.float64)
        if (
            queries.ndim != 2
            or queries.shape[1] != self.ambient_dimension
            or not np.all(np.isfinite(queries))
        ):
            raise ValueError("Classification queries must be finite ambient points.")
        radius = positive_finite_float(tolerance, "tolerance")
        if isinstance(maximum_dimension, bool) or maximum_dimension not in (0, 1, 2):
            raise ValueError("maximum_dimension must be 0, 1, or 2.")

        restrictions = (
            groups,
            allowed_groups,
            allowed_dimensions,
            allowed_indices,
            allowed_occurrence_paths,
        )
        restricted = any(value is not None for value in restrictions)
        if restricted and any(value is None for value in restrictions):
            raise ValueError(
                "Occurrence-restricted classification requires groups, allowed groups, dimensions, indices, and paths."
            )
        group_values: np.ndarray | None = None
        allowed_values: dict[int, tuple[tuple[int, int, tuple[str, ...]], ...]] | None = (
            None
        )
        if restricted:
            assert groups is not None
            assert allowed_groups is not None
            assert allowed_dimensions is not None
            assert allowed_indices is not None
            assert allowed_occurrence_paths is not None
            raw_groups = np.asarray(groups)
            raw_allowed_groups = np.asarray(allowed_groups)
            raw_dimensions = np.asarray(allowed_dimensions)
            raw_indices = np.asarray(allowed_indices)
            if any(
                not np.issubdtype(values.dtype, np.integer)
                for values in (
                    raw_groups,
                    raw_allowed_groups,
                    raw_dimensions,
                    raw_indices,
                )
            ):
                raise TypeError(
                    "Occurrence-restricted classification IDs must be integer arrays."
                )
            if (
                raw_groups.shape != (queries.shape[0],)
                or raw_allowed_groups.ndim != 1
                or raw_dimensions.shape != raw_allowed_groups.shape
                or raw_indices.shape != raw_allowed_groups.shape
                or len(allowed_occurrence_paths) != raw_allowed_groups.size
            ):
                raise ValueError(
                    "Occurrence-restricted classification filters must be row-aligned."
                )
            if not isinstance(allowed_occurrence_paths, tuple) or any(
                not isinstance(path, tuple)
                or any(not isinstance(part, str) or not part for part in path)
                for path in allowed_occurrence_paths
            ):
                raise TypeError(
                    "allowed_occurrence_paths must contain exact occurrence-path tuples."
                )
            if np.any((raw_dimensions < 0) | (raw_dimensions > 2)) or np.any(
                raw_indices < 0
            ):
                raise ValueError(
                    "Allowed entity dimensions and indices must select B-Rep entities."
                )
            if any(
                int(index) >= self.entity_counts[int(dimension)]
                for dimension, index in zip(raw_dimensions, raw_indices, strict=True)
            ):
                raise ValueError("Allowed entity index lies outside its B-Rep dimension.")
            group_values = raw_groups.astype(np.int64, copy=False)
            allowed_lists: dict[int, list[tuple[int, int, tuple[str, ...]]]] = {}
            for group, dimension, index, path in zip(
                raw_allowed_groups,
                raw_dimensions,
                raw_indices,
                allowed_occurrence_paths,
                strict=True,
            ):
                allowed_lists.setdefault(int(group), []).append(
                    (int(dimension), int(index), path)
                )
            allowed_values = {
                group: tuple(dict.fromkeys(values))
                for group, values in allowed_lists.items()
            }

        world = self._world(queries)
        offsets = np.cumsum((0,) + self.entity_counts)
        entity_boxes = np.asarray(self.entity_boxes)
        occurrence_by_path = {value.path: value for value in self.occurrences}
        hits: list[_Hit] = []

        def unresolved_is_near(hit: _Hit, query: np.ndarray, /) -> bool:
            box = entity_boxes[offsets[hit.dimension] + hit.index]
            if not np.all(np.isfinite(box)):
                return True
            if hit.path:
                occurrence = occurrence_by_path.get(hit.path)
                if occurrence is None:
                    return True
                if not (
                    np.array_equal(np.asarray(occurrence.rotation), np.eye(3))
                    and np.array_equal(np.asarray(occurrence.translation), np.zeros(3))
                ):
                    return True
            delta = np.maximum(np.maximum(box[0] - query, query - box[1]), 0.0)
            return float(np.linalg.norm(delta)) <= radius

        def add_hit(
            hit: _Hit,
            query: np.ndarray,
            candidates: list[_Hit],
            unresolved: list[_Hit],
            /,
        ) -> None:
            if hit.status == _UNRESOLVED:
                if unresolved_is_near(hit, query):
                    unresolved.append(hit)
            elif hit.status == _NON_ISOLATED:
                if hit.upper <= radius:
                    candidates.append(hit)
            elif (
                hit.status
                in (
                    _UNIQUE,
                    _SEAM,
                    _AMBIGUOUS,
                    int(BRepProjectionStatus.SINGULAR),
                )
                and hit.upper <= radius
            ):
                candidates.append(hit)

        for row, query in enumerate(world):
            candidates: list[_Hit] = []
            unresolved: list[_Hit] = []
            if group_values is None:
                entity_requests = (
                    (dimension, index, None)
                    for dimension in range(maximum_dimension + 1)
                    for index in range(self.entity_counts[dimension])
                )
            else:
                assert allowed_values is not None
                entity_requests = (
                    (dimension, index, path)
                    for dimension, index, path in allowed_values.get(
                        int(group_values[row]), ()
                    )
                    if dimension <= maximum_dimension
                )
            seen: set[tuple[int, int, tuple[str, ...] | None]] = set()
            for dimension, index, path in entity_requests:
                key = dimension, index, path
                if key in seen:
                    continue
                seen.add(key)
                hit = self._project_one(dimension, index, query, path)
                add_hit(hit, query, candidates, unresolved)
            hits.append(
                self._choose(
                    candidates,
                    unresolved,
                    self.policy.ambiguity_tolerance,
                    self.ambient_dimension,
                )
            )
        return self._result(queries, hits)

    def _contains_solid(self, point: np.ndarray, solid: int, /) -> bool | None:
        profile = _closed_revolution_profile(self.model, solid)
        if profile is not None:
            return _revolution_profile_contains(profile, point)
        direction = np.asarray((1.0, sqrt(2.0), sqrt(3.0)), dtype=np.float64)
        direction /= np.linalg.norm(direction)
        from .._rational import rational_cross, rational_vector

        origin_values = rational_vector(np.asarray(point, dtype=np.float64))
        ray_origin = (origin_values[0], origin_values[1], origin_values[2])
        direction_values = rational_vector(direction)
        ray_direction = (
            direction_values[0],
            direction_values[1],
            direction_values[2],
        )
        crossings = 0
        for face in self.model.topology.solid_faces[solid]:
            patch = self.model.patches[face]
            if isinstance(patch, CylinderPatch):
                cylinder_crossings = _ray_cylinder_crossings(
                    self.model, face, ray_origin, ray_direction
                )
                if cylinder_crossings is None:
                    return None
                crossings += cylinder_crossings
                continue
            if not (
                isinstance(patch, PlanePatch)
                or (
                    isinstance(patch, ExtrusionSurface)
                    and isinstance(patch.curve, (LineCurve, ExactLineCurve))
                )
            ):
                return None
            frame = self.affine_projection_frames[face]
            if frame is None or frame.factorization is None:
                return None
            normal = rational_cross(frame.first, frame.second)
            denominator = sum(
                (
                    value * component
                    for value, component in zip(normal, ray_direction, strict=True)
                ),
                Fraction(0),
            )
            if denominator == 0:
                continue
            numerator = sum(
                (
                    value * (origin - query)
                    for value, origin, query in zip(
                        normal, frame.origin, ray_origin, strict=True
                    )
                ),
                Fraction(0),
            )
            ray_parameter = numerator / denominator
            if ray_parameter <= 0:
                continue
            intersection = (
                ray_origin[0] + ray_parameter * ray_direction[0],
                ray_origin[1] + ray_parameter * ray_direction[1],
                ray_origin[2] + ray_parameter * ray_direction[2],
            )
            projected = _project_affine_frame_exact(frame, intersection)
            if projected is None:
                return None
            uv = np.asarray(tuple(float(value) for value in projected[0]))
            if not np.all(np.isfinite(uv)):
                return None
            trim = _trim_classification(self.model, face, uv)
            if trim is None or trim < 0:
                return None
            crossings += trim
        return bool(crossings % 2)

    @checked
    def locate_solids(
        self,
        points: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        queries = np.asarray(points, dtype=np.float64)
        if (
            queries.ndim != 2
            or queries.shape[1] != self.ambient_dimension
            or not np.all(np.isfinite(queries))
        ):
            raise ValueError("Solid-location queries must be finite ambient points.")
        if not self.entity_counts[3]:
            raise ValueError("The B-Rep source contains no solids.")
        if occurrence_paths is not None and (
            not isinstance(occurrence_paths, tuple)
            or len(occurrence_paths) != queries.shape[0]
            or any(not isinstance(path, tuple) for path in occurrence_paths)
        ):
            raise ValueError("occurrence_paths must contain one tuple path per query.")
        requested = (
            (None,) * queries.shape[0] if occurrence_paths is None else occurrence_paths
        )
        world = self._world(queries)
        hits: list[_Hit] = []
        for query, selected_path in zip(world, requested, strict=True):
            if selected_path == ():
                targets = tuple(
                    _Occurrence((), solid, _IDENTITY_ROTATION, _ZERO_TRANSLATION)
                    for solid in range(self.entity_counts[3])
                )
            else:
                targets = tuple(
                    occurrence
                    for occurrence in self.occurrences
                    if selected_path is None or occurrence.path == selected_path
                )
            if selected_path is not None and selected_path and not targets:
                raise ValueError("occurrence_path does not belong to the B-Rep source.")
            inside: list[_Hit] = []
            boundary: list[_Hit] = []
            unresolved: list[_Hit] = []
            for occurrence in targets:
                rotation = np.asarray(occurrence.rotation, dtype=np.float64)
                translation = np.asarray(occurrence.translation, dtype=np.float64)
                local = rotation.T @ (query - translation)
                if not np.all(np.isfinite(local)):
                    unresolved.append(_unresolved(3, occurrence.solid, occurrence.path))
                    continue
                face_hits = [
                    self._project_one(
                        BRepEntityDimension.FACE,
                        face,
                        query,
                        occurrence.path,
                    )
                    for face in self.model.topology.solid_faces[occurrence.solid]
                ]
                near = [
                    hit
                    for hit in face_hits
                    if hit.status in (_UNIQUE, _SEAM, _AMBIGUOUS)
                    and hit.upper <= self.policy.classifier_tolerance
                ]
                if near:
                    face = min(near, key=lambda hit: hit.upper)
                    boundary.append(
                        _Hit(
                            3,
                            occurrence.solid,
                            face.point,
                            _NAN2.copy(),
                            face.lower,
                            face.upper,
                            _AMBIGUOUS,
                            face.normal,
                            face.tangents,
                            occurrence.path,
                        )
                    )
                    continue
                if any(hit.status == _UNRESOLVED for hit in face_hits):
                    unresolved.append(_unresolved(3, occurrence.solid, occurrence.path))
                    continue
                contained = self._contains_solid(local, occurrence.solid)
                if contained is None:
                    unresolved.append(_unresolved(3, occurrence.solid, occurrence.path))
                elif contained:
                    inside.append(
                        _Hit(
                            3,
                            occurrence.solid,
                            query,
                            _NAN2.copy(),
                            0.0,
                            0.0,
                            _UNIQUE,
                            _NAN3.copy(),
                            _NAN_FRAME3.copy(),
                            occurrence.path,
                        )
                    )
            if unresolved:
                hits.append(unresolved[0])
            elif boundary:
                hits.append(
                    self._choose(boundary, [], self.policy.ambiguity_tolerance, 3)
                )
            elif len(inside) > 1:
                first = inside[0]
                hits.append(
                    _Hit(
                        3,
                        first.index,
                        first.point,
                        first.parameters,
                        0.0,
                        0.0,
                        _AMBIGUOUS,
                        first.normal,
                        first.tangents,
                        first.path,
                    )
                )
            elif inside:
                hits.append(inside[0])
            else:
                hits.append(_failed(self.ambient_dimension))
        return self._result(queries, hits)


@checked
def prepare_brep_projection(
    model: BRepModel,
    /,
    *,
    policy: BRepProjectionPolicy | None = None,
    embedding: PlanarEmbedding | None = None,
) -> NativeBRepProjection:
    """Prepare source-native B-Rep projection without consulting a mesh or CAD engine."""
    policy_ = BRepProjectionPolicy() if policy is None else policy
    if not isinstance(policy_, BRepProjectionPolicy):
        raise TypeError("policy must be a BRepProjectionPolicy or None.")
    if embedding is not None and not isinstance(embedding, PlanarEmbedding):
        raise TypeError("embedding must be a PlanarEmbedding or None.")
    return NativeBRepProjection(model, policy_, embedding)


__all__ = [
    "NativeBRepProjection",
    "prepare_brep_projection",
]
