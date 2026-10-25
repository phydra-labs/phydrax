#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Construct B-Reps from primitive carriers, sketches and sweeps.

Every construction builds carriers (analytic surfaces, lines, circles),
oriented coedges with same-parameter p-curves, closed loops, shells and solids,
then derives a query tessellation with its own identity. Sweeps recognize the
analytic family of each swept edge (plane, cylinder, cone, sphere, torus)
and use generating extrusion/revolution surfaces otherwise. Primitives are
profile sweeps (box: extruded rectangle; cylinder, cone, sphere,
torus: revolved profiles), so their topology, orientation and identity follow
one construction route.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, InitVar
from fractions import Fraction
from math import atan2, ceil, cos, pi, sin
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint, tree_content_equal
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import (
    finite_real_scalar,
    nonnegative_real_scalar,
    positive_finite_float,
)
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ...discretization._exact_algebraic_geometry import rational_coefficient
from ...typing import checked, ConvertibleToArray
from .._atlas import AbstractTrimCurve, CurveTrimLoop, TrimDomain
from .._planar_primitives import rational_points, reserve_fraction_work
from .._rational import fraction_down, fraction_up, rational_cross_2d, rational_vector
from ._carrier_pieces import (
    admit_carrier_fingerprint,
    AffineTrimChordEvidence,
    carrier_payload,
)
from ._conic_gaps import endpoint_gap_bounds
from ._coordinate_request import coordinate_source_request
from ._intersection import (
    intersect_curve_ranges,
    NativePeriodEndpoint,
    original_curve_surface_root_preparation,
    RootEndpoint,
)
from ._intersection_curve import (
    affine_curve_coefficients,
    AffinePCurve,
    CurveRange,
    CurveTrimSegment,
    encode_geometry,
    exact_line_pcurve,
    IntersectionPCurve,
    PeriodicPCurve,
)
from ._intersection_kernel import point_times_interval
from ._model import (
    _certify_edge_face_gap_bounds,
    brep_physical_scale,
    BRepCurve,
    BRepGeometry,
    BRepImportReport,
    BRepModel,
    BRepOccurrence,
    BRepPCurve,
)
from ._offset_result import BRepOffsetResult, exact_operation_offset_result
from ._parameter_range import attach_parameter_range, unknown_parameter_range
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    axis_frame,
    BernsteinRestrictionBank,
    BSplineCurve,
    CircleCurve,
    closure_tangent_continuous,
    ConePatch,
    convex_meridian_offset,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    LineCurve,
    MeridianOverlaps,
    OffsetSurface,
    PlanePatch,
    RevolutionSurface,
    sphere_source_equivalence,
    SpherePatch,
    surface_value_bounds,
    SurfaceIsoparametricCurve,
    TorusPatch,
)
from ._planar_arrangement import area, inside_loop, intersections, Point, sub
from ._request_preparations import TrimRequestPreparations
from ._root_bindings import BRepVertexRoot, interval_jacobian_full_rank


_TWO_PI = 2.0 * pi
_GEOMETRIC_TOLERANCE = 1.0e-10


# ------------------------------------------------------------------ policies


class BRepTessellationPolicy(StrictModule, NonTrainableState):
    """Derived query-tessellation resolution and resource bound.

    ``linear_deflection`` bounds the chord deviation of edge and surface
    samples, ``angular_deflection`` the normal turn between neighboring samples
    (radians); ``trim_samples_per_edge`` is the minimum number of samples per
    coedge in the polygonal trim charts. ``maximum_triangles`` bounds the
    tessellation size. ``realize=False`` retains source carriers and topology
    without constructing the optional query tessellation. None of these
    settings changes the source model identity.
    """

    linear_deflection: float = eqx.field(static=True)
    angular_deflection: float = eqx.field(static=True)
    trim_samples_per_edge: int = eqx.field(static=True)
    maximum_triangles: int = eqx.field(static=True)
    realize: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_deflection: float = 1.0e-3,
        angular_deflection: float = 0.1,
        trim_samples_per_edge: int = 33,
        maximum_triangles: int = 2_000_000,
        realize: bool = True,
    ) -> None:
        linear = positive_finite_float(linear_deflection, "linear_deflection")
        angular = positive_finite_float(angular_deflection, "angular_deflection")
        if not isinstance(realize, bool):
            raise TypeError(
                f"BRepTessellationPolicy.realize must be a boolean derived-artifact request; got {realize!r}"
            )
        if isinstance(trim_samples_per_edge, bool) or not isinstance(
            trim_samples_per_edge, int
        ):
            raise TypeError(
                f"BRepTessellationPolicy.trim_samples_per_edge must be an integer; got {trim_samples_per_edge!r}"
            )
        if trim_samples_per_edge < 3:
            raise ValueError(
                f"BRepTessellationPolicy.trim_samples_per_edge must be at least three; got {trim_samples_per_edge}"
            )
        if isinstance(maximum_triangles, bool) or not isinstance(maximum_triangles, int):
            raise TypeError(
                f"BRepTessellationPolicy.maximum_triangles must be an integer; got {maximum_triangles!r}"
            )
        if maximum_triangles <= 0:
            raise ValueError(
                f"BRepTessellationPolicy.maximum_triangles must be positive; got {maximum_triangles}"
            )
        if angular >= pi:
            raise ValueError(
                f"BRepTessellationPolicy.angular_deflection must be below pi; got {angular}"
            )
        self.linear_deflection = linear
        self.angular_deflection = angular
        self.trim_samples_per_edge = trim_samples_per_edge
        self.maximum_triangles = maximum_triangles
        self.realize = realize
        self.policy_id = canonical_fingerprint(
            {
                "kind": "native-brep-tessellation-policy",
                "linear_deflection": linear,
                "angular_deflection": angular,
                "trim_samples_per_edge": trim_samples_per_edge,
                "maximum_triangles": maximum_triangles,
                "realize": realize,
            }
        )


# ------------------------------------------------------------------ sketches


def _point2(value: ConvertibleToArray, name: str, /) -> tuple[float, float]:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(
            f"ProfilePlane.coordinate must be a finite two-dimensional point; got {name} shape {point.shape}"
        )
    return float(point[0]), float(point[1])


def _point3(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(
            f"ProfilePlane.coordinate must be a finite three-dimensional vector; got {name} shape {point.shape}"
        )
    return point


def _unit3(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    vector = _point3(value, name)
    norm = float(np.linalg.norm(vector))
    if not norm > 0.0:
        raise ValueError(
            f"ProfilePlane.direction must be nonzero; got {name} norm {norm}"
        )
    return vector / norm


@dataclass(frozen=True, slots=True)
class ProfileLine:
    """Straight segment from a loop vertex to the next."""


@dataclass(frozen=True, slots=True)
class ProfileArc:
    """Circular arc about ``center`` from a loop vertex to the next.

    A loop with one vertex and one arc is a full circle.
    """

    center: tuple[float, float]
    counterclockwise: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point2(self.center, "center"))
        if not isinstance(self.counterclockwise, bool):
            raise TypeError(
                f"ProfileArc.counterclockwise must be a bool; got {self.counterclockwise!r}"
            )


class ProfileCurveUse(StrictModule, NonTrainableState):
    """Directed angular use of an unchanged planar analytic conic.

    ``first`` and ``last`` are authored binary64 parameters, not angles inferred
    from loop vertices. Decreasing ranges reverse traversal. Endpoint coincidence
    is proved by ``PlanarProfile`` against its optional declared gap rows.
    """

    curve: CircleCurve | EllipseCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)

    @checked
    def __init__(
        self, curve: CircleCurve | EllipseCurve, first: float, last: float
    ) -> None:
        first_ = finite_real_scalar(first, "ProfileCurveUse.first")
        last_ = finite_real_scalar(last, "ProfileCurveUse.last")
        if curve.ambient_dimension != 2:
            raise ValueError(
                "ProfileCurveUse.curve must be a planar conic; "
                f"got dimension {curve.ambient_dimension}"
            )
        if first_ == last_ or abs(last_ - first_) > _TWO_PI:
            raise ValueError(
                "ProfileCurveUse.range must cover a positive, at-most-full turn; "
                f"got {(first_, last_)}"
            )
        self.curve = curve
        self.first = first_
        self.last = last_


type ProfileSegment = ProfileLine | ProfileArc | BSplineCurve | ProfileCurveUse


@dataclass(frozen=True, slots=True)
class ProfileLoop:
    """Closed loop; segment ``i`` joins vertex ``i`` to ``i + 1``.

    A native ``BSplineCurve`` segment uses its full clamped parameter
    domain, finite planar controls and positive rational weights. Its boundary
    controls must exactly equal the declared vertices. Closed regular splines
    may form a one-vertex loop. Continuum topology certification belongs to
    ``PlanarProfile``; mixed native families use the source/chord
    certificate. Unproved source joins and singular boundaries remain explicit.
    ``ProfileCurveUse`` retains a native conic and its directed angular range;
    vertices remain independently authored. ``PlanarProfile`` owns the endpoint
    gap replay and source-topology certificate for these uses.
    """

    vertices: tuple[tuple[float, float], ...]
    segments: tuple[ProfileSegment, ...]

    def __post_init__(self) -> None:
        vertices = tuple(_point2(vertex, "vertex") for vertex in self.vertices)
        segments = tuple(self.segments)
        if not vertices or len(segments) != len(vertices):
            raise ValueError(
                f"ProfileLoop.segments must provide one segment per vertex; got {len(segments)} segments and {len(vertices)} vertices"
            )
        if not all(
            isinstance(segment, (ProfileLine, ProfileArc, BSplineCurve, ProfileCurveUse))
            for segment in segments
        ):
            raise TypeError(
                "ProfileLoop.segments must be lines, arcs or planar BSplineCurve values; got an unsupported segment"
            )
        if len(vertices) == 1 and isinstance(segments[0], ProfileLine):
            raise ValueError(
                "ProfileLoop.segments must use a closed curved carrier for a one-vertex loop; got a line"
            )
        if len(vertices) == 2 and all(isinstance(s, ProfileLine) for s in segments):
            raise ValueError(
                "ProfileLoop.segments must include a curved segment for a two-vertex loop; got two lines"
            )
        if len(set(vertices)) != len(vertices):
            raise ValueError(
                "ProfileLoop.vertices must be distinct; got repeated vertices"
            )
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "segments", segments)

    @classmethod
    def polygon(cls, vertices: ConvertibleToArray) -> ProfileLoop:
        """Closed polygon through ``vertices``."""
        points = np.asarray(vertices, dtype=np.float64)
        return cls(
            tuple((float(x), float(y)) for x, y in points),
            tuple(ProfileLine() for _ in range(points.shape[0])),
        )

    @classmethod
    def circle(cls, center: ConvertibleToArray, radius: float) -> ProfileLoop:
        """Full counterclockwise circle starting at angle zero."""
        cx, cy = _point2(center, "center")
        radius_ = positive_finite_float(radius, "radius")
        return cls(((cx + radius_, cy),), (ProfileArc((cx, cy)),))


@dataclass(frozen=True, slots=True)
class ProfilePlane:
    """Orthonormal sketch frame; sketch ``(x, y)`` maps to ``origin + x X + y Y``."""

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    x_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    y_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        origin = _point3(self.origin, "origin")
        x_axis = _point3(self.x_axis, "x_axis")
        y_axis = _point3(self.y_axis, "y_axis")
        if (
            abs(np.linalg.norm(x_axis) - 1.0) > _GEOMETRIC_TOLERANCE
            or abs(np.linalg.norm(y_axis) - 1.0) > _GEOMETRIC_TOLERANCE
            or abs(float(x_axis @ y_axis)) > _GEOMETRIC_TOLERANCE
        ):
            raise ValueError(
                "ProfilePlane.axes must be orthonormal; got a nonorthonormal frame"
            )
        object.__setattr__(self, "origin", tuple(float(v) for v in origin))
        object.__setattr__(self, "x_axis", tuple(float(v) for v in x_axis))
        object.__setattr__(self, "y_axis", tuple(float(v) for v in y_axis))

    @property
    def normal(self) -> np.ndarray:
        return np.cross(np.asarray(self.x_axis), np.asarray(self.y_axis))

    def point(self, xy: ConvertibleToArray, /) -> np.ndarray:
        values = np.asarray(xy, dtype=np.float64)
        return (
            np.asarray(self.origin)
            + values[..., :1] * np.asarray(self.x_axis)
            + values[..., 1:2] * np.asarray(self.y_axis)
        )

    def vector(self, xy: ConvertibleToArray, /) -> np.ndarray:
        values = np.asarray(xy, dtype=np.float64)
        return values[..., :1] * np.asarray(self.x_axis) + values[..., 1:2] * np.asarray(
            self.y_axis
        )


@dataclass(frozen=True, slots=True)
class _ProfileEdge:
    """Traversal-ordered segment: 2D p-curve, range and end points.

    ``turn`` is +1 (-1) for counterclockwise (clockwise) arcs and 0 otherwise.
    """

    pcurve: AbstractCurve
    first: float
    last: float
    start: tuple[float, float]
    end: tuple[float, float]
    turn: float
    radius: float
    authored_range: bool = False


def _arc_angle(
    point: tuple[float, float], center: tuple[float, float], sign: float
) -> float:
    return atan2(sign * (point[1] - center[1]), point[0] - center[0])


def rational_circular_arc(
    start: tuple[float, float],
    end: tuple[float, float],
    center: tuple[float, float],
    radius: float,
    /,
) -> BSplineCurve:
    """Rational quadratic conic spline of a counterclockwise circular arc.

    Each Bernstein piece spans at most a quarter turn of the circle
    ``(center, radius)`` with its tangent-intersection control and weight
    ``cos(step / 2)``. The boundary controls are exactly ``start`` and
    ``end``, so the carrier is a rational conic spline through the
    authored vertices: interior junctions follow the authored circle, and an
    end point off that circle by a solver residual perturbs only its own
    conic piece. Nothing is sampled or refitted.
    """
    start_, end_ = _point2(start, "start"), _point2(end, "end")
    cx, cy = _point2(center, "center")
    radius_ = positive_finite_float(radius, "radius")
    first = atan2(start_[1] - cy, start_[0] - cx)
    sweep = (atan2(end_[1] - cy, end_[0] - cx) - first) % _TWO_PI
    if start_ == end_ or not sweep > 0.0:
        raise ValueError(
            "rational_circular_arc.endpoints must have distinct end directions; got a zero sweep"
        )
    pieces = max(1, ceil(sweep / (0.5 * pi)))
    step = sweep / pieces
    weight = cos(0.5 * step)
    controls = [start_]
    for index in range(pieces):
        middle = first + (index + 0.5) * step
        controls.append(
            (cx + radius_ / weight * cos(middle), cy + radius_ / weight * sin(middle))
        )
        following = first + (index + 1) * step
        controls.append(
            end_
            if index == pieces - 1
            else (cx + radius_ * cos(following), cy + radius_ * sin(following))
        )
    knots = (
        [0.0] * 3
        + [float(knot) for knot in range(1, pieces) for _ in range(2)]
        + [float(pieces)] * 3
    )
    return BSplineCurve(
        np.asarray(controls, dtype=np.float64),
        np.asarray([1.0] + [weight, 1.0] * pieces, dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        2,
    )


def _endpoint_line_curve(
    origin: np.ndarray,
    direction: np.ndarray,
    first: float,
    last: float,
    start: np.ndarray,
    end: np.ndarray,
    /,
) -> LineCurve | BSplineCurve:
    """Publish a line specialization when endpoint images agree in rational arithmetic.

    Otherwise the endpoint controls define the carrier, on the same
    declared interval; a topology certificate never substitutes another curve.
    """
    exact_origin = tuple(Fraction(float(value)) for value in origin)
    exact_direction = tuple(Fraction(float(value)) for value in direction)
    for parameter, point in ((first, start), (last, end)):
        scalar = Fraction(parameter)
        if any(
            a + scalar * b != Fraction(float(value))
            for a, b, value in zip(exact_origin, exact_direction, point, strict=True)
        ):
            return BSplineCurve(
                np.stack((start, end)),
                np.ones((2,), dtype=np.float64),
                np.asarray((first, first, last, last), dtype=np.float64),
                1,
            )
    return LineCurve(origin, direction)


def _profile_conic_edge(
    start: tuple[float, float], end: tuple[float, float], segment: ProfileCurveUse, /
) -> _ProfileEdge:
    """Lower directed source parameters by an algebraic reflection when needed."""
    curve, first, last = segment.curve, segment.first, segment.last
    pcurve = curve if first < last else _reflected_conic(curve)
    first_, last_ = (first, last) if first < last else (-first, -last)
    first_axis = np.asarray(pcurve.first_axis)
    second_axis = np.asarray(pcurve.second_axis)
    determinant = rational_cross_2d(
        (Fraction(float(first_axis[0])), Fraction(float(first_axis[1]))),
        (Fraction(float(second_axis[0])), Fraction(float(second_axis[1]))),
    )
    turn = 1.0 if determinant > 0 else -1.0
    radius = float(pcurve.radius) if isinstance(pcurve, CircleCurve) else 0.0
    return _ProfileEdge(pcurve, first_, last_, start, end, turn, radius, True)


def _profile_spline_edge(
    start: tuple[float, float], end: tuple[float, float], segment: BSplineCurve, /
) -> _ProfileEdge:
    """Validate the source spline's planar, positive-weight, clamped boundary."""
    if segment.ambient_dimension != 2:
        raise ValueError(
            f"ProfileLoop.spline must have two-dimensional control points; got dimension {segment.ambient_dimension}"
        )
    controls = np.asarray(segment.control_points)
    weights = np.asarray(segment.weights)
    knots = np.asarray(segment.knots)
    if not np.all(np.isfinite(controls)) or not np.all(np.isfinite(weights)):
        raise ValueError(
            "ProfileLoop.spline must have finite controls and weights; got nonfinite coefficients"
        )
    if np.any(weights <= 0.0):
        raise ValueError(
            "ProfileLoop.spline_weights must be positive rational weights; got nonpositive weights"
        )
    first, last = segment.validate_range(
        float(knots[segment.degree]), float(knots[-segment.degree - 1])
    )
    if not (
        np.all(knots[: segment.degree + 1] == first)
        and np.all(knots[-segment.degree - 1 :] == last)
    ):
        raise ValueError(
            "ProfileLoop.spline must be clamped at its segment boundaries; got an unclamped knot bank"
        )
    if tuple(controls[0]) != start or tuple(controls[-1]) != end:
        raise ValueError(
            "ProfileLoop.spline_boundary_controls must exactly match its loop vertices; got mismatched endpoints"
        )
    if np.array_equal(controls[0], controls[1]) or np.array_equal(
        controls[-1], controls[-2]
    ):
        raise ValueError(
            "ProfileLoop.spline_boundary must have nonzero derivative; got an unresolved zero-derivative boundary"
        )
    return _ProfileEdge(segment, first, last, start, end, 0.0, 0.0)


def _profile_arc_edge(
    start: tuple[float, float], end: tuple[float, float], segment: ProfileArc, /
) -> _ProfileEdge:
    """Retain the legacy rational radial endpoint predicate and native-turn range."""
    center = segment.center
    sign = 1.0 if segment.counterclockwise else -1.0
    radius = float(np.hypot(start[0] - center[0], start[1] - center[1]))
    first_radius_squared = sum(
        ((Fraction(a) - Fraction(b)) ** 2 for a, b in zip(start, center, strict=True)),
        Fraction(),
    )
    last_radius_squared = sum(
        ((Fraction(a) - Fraction(b)) ** 2 for a, b in zip(end, center, strict=True)),
        Fraction(),
    )
    if not radius > 0.0 or first_radius_squared != last_radius_squared:
        raise ValueError(
            "ProfileArc.endpoints must be equidistant from its center with positive radius; got mismatched endpoint radii"
        )
    first = _arc_angle(start, center, sign)
    last = _arc_angle(end, center, sign)
    while last <= first:
        last += _TWO_PI
    if start != end and last - first >= _TWO_PI:
        last -= _TWO_PI
    if start == end:
        last = first + _TWO_PI
    pcurve = CircleCurve(center, (1.0, 0.0), (0.0, sign), radius)
    return _ProfileEdge(pcurve, first, last, start, end, sign, radius)


def _profile_edge(
    start: tuple[float, float], end: tuple[float, float], segment: ProfileSegment, /
) -> _ProfileEdge:
    match segment:
        case ProfileCurveUse():
            return _profile_conic_edge(start, end, segment)
        case ProfileLine():
            start_, end_ = (
                np.asarray(start, dtype=np.float64),
                np.asarray(end, dtype=np.float64),
            )
            delta = end_ - start_
            length = float(np.linalg.norm(delta))
            if not length > 0.0:
                raise ValueError(f"ProfileLine.length must be positive; got {length}")
            return _ProfileEdge(
                _endpoint_line_curve(start_, delta / length, 0.0, length, start_, end_),
                0.0,
                length,
                start,
                end,
                0.0,
                0.0,
            )
        case ProfileArc():
            return _profile_arc_edge(start, end, segment)
        case BSplineCurve():
            return _profile_spline_edge(start, end, segment)
        case _:
            raise TypeError(
                f"ProfileLoop.segment must be a supported sketch segment; got {type(segment).__name__}"
            )


def _loop_edges(loop: ProfileLoop, /) -> tuple[_ProfileEdge, ...]:
    count = len(loop.vertices)
    return tuple(
        _profile_edge(loop.vertices[index], loop.vertices[(index + 1) % count], segment)
        for index, segment in enumerate(loop.segments)
    )


@dataclass(frozen=True, slots=True)
class _ProfileHull:
    """Source Bernstein restriction used for topology certification."""

    controls: tuple[tuple[Fraction, ...], ...]

    @property
    def points(self) -> tuple[Point, ...]:
        return tuple((row[0] / row[2], row[1] / row[2]) for row in self.controls)

    def split(self) -> tuple[_ProfileHull, _ProfileHull]:
        rows = self.controls
        left, right = [rows[0]], [rows[-1]]
        while len(rows) > 1:
            rows = tuple(
                tuple((a + b) / 2 for a, b in zip(first, second, strict=True))
                for first, second in zip(rows[:-1], rows[1:], strict=True)
            )
            left.append(rows[0])
            right.append(rows[-1])
        return _ProfileHull(tuple(left)), _ProfileHull(tuple(reversed(right)))


def _source_profile_hulls(edge: _ProfileEdge, /) -> list[_ProfileHull]:
    match edge.pcurve:
        case LineCurve():
            return [
                _ProfileHull(
                    tuple(
                        (Fraction(x), Fraction(y), Fraction(1))
                        for x, y in (edge.start, edge.end)
                    )
                )
            ]
        case BSplineCurve():
            result = []
            for piece in edge.pcurve.exact_bezier_pieces():
                controls = piece.homogeneous_controls
                rows = []
                for row in range(controls.shape[0]):
                    values = []
                    for column in range(3):
                        value = controls[row, column]
                        if not isinstance(value, Fraction):
                            raise TypeError(
                                f"PlanarProfile.spline_coefficients must be source rational coefficients; got {type(value).__name__}"
                            )
                        values.append(value)
                    rows.append(tuple(values))
                hull = _ProfileHull(tuple(rows))
                points = hull.points
                if points[0] == points[1] or points[-1] == points[-2]:
                    raise ValueError(
                        "PlanarProfile.spline_boundary must have nonzero derivative; got an unresolved zero-derivative boundary"
                    )
                result.append(hull)
            return result
        case _:
            raise TypeError(
                "PlanarProfile.hull_carrier must be a native line or spline carrier; got an unsupported carrier"
            )


def _regular_profile_hull(hull: _ProfileHull, /) -> bool:
    points = hull.points
    first, last = points[0], points[-1]
    direction = (last[0] - first[0], last[1] - first[1])
    projections = tuple(
        point[0] * direction[0] + point[1] * direction[1] for point in points
    )
    # Positive weights and total positivity preserve this strict projection.
    # The endpoint inequalities also exclude zero-derivative boundary poles.
    return (
        projections[1] > projections[0]
        and projections[-1] > projections[-2]
        and all(a <= b for a, b in zip(projections[:-1], projections[1:], strict=True))
    )


def _supporting_line_traces(
    points: tuple[Point, ...], values: tuple[Fraction, ...], plane: Fraction, /
) -> tuple[tuple[Point, Point], ...]:
    """Return contact of a regular restriction's isotopy family with a line.

    The hull lies on one closed side of the line. A straight-line homotopy
    from the source curve to its control polygon is on the line only where
    both are; interior Bernstein values are strictly positive combinations
    of every control, so that set is the polygon's on-line edges and
    vertices. Tangent (G1) junction controls therefore count only through
    their on-line polygon edge, never as hull-wide contact.
    """
    on_line = tuple(value == plane for value in values)
    edges = tuple(
        (points[index], points[index + 1])
        for index in range(len(points) - 1)
        if on_line[index] and on_line[index + 1]
    )
    covered = {point for edge in edges for point in edge}
    return edges + tuple(
        (point, point)
        for point, touching in zip(points, on_line, strict=True)
        if touching and point not in covered
    )


def _profile_plane_separates(
    left: tuple[Point, ...],
    right: tuple[Point, ...],
    left_projection: tuple[Fraction, ...],
    right_projection: tuple[Fraction, ...],
    allowed: frozenset[Point],
) -> bool:
    plane = max(left_projection)
    boundary = min(right_projection)
    if plane < boundary:
        return True
    if plane != boundary:
        return False
    contacts = [
        intersections(first_trace, second_trace)
        for first_trace in _supporting_line_traces(left, left_projection, plane)
        for second_trace in _supporting_line_traces(right, right_projection, plane)
    ]
    return all(
        len(found) <= 1 and all(point in allowed for point in found) for found in contacts
    )


def _separated_profile_hulls(
    first: _ProfileHull, second: _ProfileHull, allowed: frozenset[Point], /
) -> bool:
    a, b = first.points, second.points
    for points in (a, b):
        for index, start in enumerate(points):
            for end in points[index + 1 :]:
                normal = (start[1] - end[1], end[0] - start[0])
                if normal == (0, 0):
                    continue
                pa = tuple(p[0] * normal[0] + p[1] * normal[1] for p in a)
                pb = tuple(p[0] * normal[0] + p[1] * normal[1] for p in b)
                if _profile_plane_separates(a, b, pa, pb, allowed) or (
                    _profile_plane_separates(b, a, pb, pa, allowed)
                ):
                    return True
    return False


class _ProfileHullWorkset(StrictModule):
    hulls: list[list[_ProfileHull]]


class _ProfileSubdivision(StrictModule):
    loop: int
    hull: int


def _prepare_profile_hulls(
    edges: tuple[tuple[_ProfileEdge, ...], ...],
) -> _ProfileHullWorkset:
    hulls = [
        [hull for edge in loop for hull in _source_profile_hulls(edge)] for loop in edges
    ]
    for loop in hulls:
        if any(
            first.points[-1] != second.points[0]
            for first, second in zip(loop, (*loop[1:], loop[0]), strict=True)
        ):
            raise ValueError(
                "PlanarProfile.spline_restrictions must close exactly at every profile boundary; got an open source junction"
            )
    return _ProfileHullWorkset(hulls)


def _profile_regularity_split(
    workset: _ProfileHullWorkset,
) -> _ProfileSubdivision | None:
    for loop_index, loop in enumerate(workset.hulls):
        for index, hull in enumerate(loop):
            if not _regular_profile_hull(hull):
                return _ProfileSubdivision(loop_index, index)
    return None


def _profile_pair_split(
    workset: _ProfileHullWorkset,
    li: int,
    hi: int,
    first: _ProfileHull,
    lj: int,
    hj: int,
    second: _ProfileHull,
) -> _ProfileSubdivision | None:
    allowed: set[Point] = set()
    if li == lj:
        if (hi + 1) % len(workset.hulls[li]) == hj:
            allowed.add(first.points[-1])
        if (hj + 1) % len(workset.hulls[li]) == hi:
            allowed.add(second.points[-1])
    if _separated_profile_hulls(first, second, frozenset(allowed)):
        return None
    first_size = sum(
        (a - b) ** 2 for a, b in zip(first.points[0], first.points[-1], strict=True)
    )
    second_size = sum(
        (a - b) ** 2 for a, b in zip(second.points[0], second.points[-1], strict=True)
    )
    return (
        _ProfileSubdivision(li, hi)
        if len(first.controls) > 2
        and (len(second.controls) == 2 or first_size >= second_size)
        else _ProfileSubdivision(lj, hj)
    )


def _profile_contact_split(
    workset: _ProfileHullWorkset,
) -> _ProfileSubdivision | None:
    entries = [
        (li, hi, hull)
        for li, loop in enumerate(workset.hulls)
        for hi, hull in enumerate(loop)
    ]
    for index, (li, hi, first) in enumerate(entries):
        for lj, hj, second in entries[index + 1 :]:
            split = _profile_pair_split(workset, li, hi, first, lj, hj, second)
            if split is not None:
                return split
    return None


def _profile_polygons(
    edges: tuple[tuple[_ProfileEdge, ...], ...], /
) -> tuple[tuple[Point, ...], ...]:
    """Return control polygons after proving the source curves' isotopy.

    Strict monotone projections establish regularity, and separating source
    hull planes exclude unintended contacts before polygon containment is used.
    """
    workset = _prepare_profile_hulls(edges)
    hulls = workset.hulls
    for _ in range(256):
        split = _profile_regularity_split(workset)
        if split is None:
            split = _profile_contact_split(workset)
        if split is None:
            polygons = tuple(
                tuple(p for hull in loop for p in hull.points[:-1]) for loop in hulls
            )
            _require_profile_containment(polygons)
            return polygons
        li, hi = split.loop, split.hull
        hull = hulls[li][hi]
        if len(hull.controls) == 2:
            raise ValueError(
                "PlanarProfile.loops must not touch, cross or overlap; got intersecting source hulls"
            )
        hulls[li][hi : hi + 1] = hull.split()
    raise ValueError(
        "PlanarProfile.topology must resolve within its source-hull subdivision budget; got unresolved singularity or separation"
    )


def _require_profile_containment(polygons: tuple[tuple[Point, ...], ...], /) -> None:
    if any(area(polygon) == 0 for polygon in polygons):
        raise ValueError(
            "PlanarProfile.loops must enclose positive area; got a zero-area loop"
        )
    for index, hole in enumerate(polygons[1:]):
        if not inside_loop(hole[0], polygons[0]):
            raise ValueError(
                "PlanarProfile.holes must lie inside the outer loop; got an exterior hole"
            )
        for other in polygons[index + 2 :]:
            if inside_loop(hole[0], other) or inside_loop(other[0], hole):
                raise ValueError(
                    "PlanarProfile.holes must not be nested; got nested holes"
                )


def _source_trim_inside(
    loop: CurveTrimLoop,
    point: tuple[float, float],
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> bool:
    classification = TrimDomain(loop).classify(
        np.asarray((point,), dtype=np.float64), preparations=preparations, budget=budget
    )
    if not bool(classification.resolved[0]) or bool(classification.boundary[0]):
        raise ValueError(
            "PlanarProfile.containment must resolve away from the source boundary; got an unresolved boundary certificate"
        )
    return bool(classification.inside[0])


def _require_mixed_profile_relations(
    loops: tuple[CurveTrimLoop, ...],
    edges: tuple[tuple[_ProfileEdge, ...], ...],
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> None:
    for first in range(len(edges)):
        for second in range(first + 1, len(edges)):
            for a in edges[first]:
                for b in edges[second]:
                    result = intersect_curve_ranges(
                        CurveRange(a.pcurve, a.first, a.last),
                        CurveRange(b.pcurve, b.first, b.last),
                        preparations=preparations,
                        coordinate_budget=budget,
                    )
                    if not result.complete:
                        raise ValueError(
                            "PlanarProfile.cross_loop_intersections must resolve; got unresolved source intersections"
                        )
                    if result.points or result.coincident:
                        raise ValueError(
                            "PlanarProfile.loops must not touch or cross; got cross-loop contact"
                        )
    for index in range(1, len(loops)):
        point = edges[index][0].start
        if not _source_trim_inside(
            loops[0], point, preparations=preparations, budget=budget
        ):
            raise ValueError(
                "PlanarProfile.holes must lie inside the outer loop; got an exterior hole"
            )
        for other in range(index + 1, len(loops)):
            if _source_trim_inside(
                loops[other], point, preparations=preparations, budget=budget
            ) or _source_trim_inside(
                loops[index],
                edges[other][0].start,
                preparations=preparations,
                budget=budget,
            ):
                raise ValueError(
                    "PlanarProfile.holes must not be nested; got nested holes"
                )


def _profile_endpoint_gap_row(
    edge: _ProfileEdge,
    row: tuple[float, float],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:
    """Certify one source endpoint pair against its two declared limits."""
    if len(row) != 2:
        raise ValueError(
            "PlanarProfile.endpoint_gap_bounds must contain endpoint pairs; "
            f"got {len(row)} values"
        )
    limits = tuple(
        nonnegative_real_scalar(value, "PlanarProfile.endpoint_gap_bounds")
        for value in row
    )
    bounds = endpoint_gap_bounds(
        edge.pcurve,
        (edge.first, edge.last),
        np.asarray((edge.start, edge.end), dtype=np.float64),
        budget=budget,
        endpoint_roots=_profile_period_endpoints(edge, budget=budget),
    )
    if any(bound > limit for bound, limit in zip(bounds, limits, strict=True)):
        raise ValueError(
            "PlanarProfile.endpoint_gap_bounds must cover certified source gaps; "
            f"got bounds {bounds} above limits {limits}"
        )
    return bounds


def _profile_endpoint_gaps(
    edges: tuple[tuple[_ProfileEdge, ...], ...],
    declared: tuple[tuple[tuple[float, float], ...], ...] | None,
    /,
    *,
    conic_use: bool,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[float, float], ...], ...] | None:
    """Replay each caller bound against immutable carrier parameters and vertices."""
    if declared is None and not conic_use:
        return None
    if declared is not None and (
        len(declared) != len(edges)
        or any(len(rows) != len(loop) for rows, loop in zip(declared, edges, strict=True))
    ):
        raise ValueError(
            "PlanarProfile.endpoint_gap_bounds must align with every source edge; "
            "got mismatched rows"
        )
    certified = tuple(
        tuple(
            _profile_endpoint_gap_row(
                edge,
                (0.0, 0.0) if declared is None else declared[index][edge_index],
                budget=budget,
            )
            for edge_index, edge in enumerate(loop)
        )
        for index, loop in enumerate(edges)
    )
    has_gap = any(bound for rows in certified for row in rows for bound in row)
    return certified if has_gap else None


def _quadrant_profile_endpoint(
    curve: CircleCurve,
    point: tuple[float, float],
    parameter: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> NativePeriodEndpoint | None:
    """Prove a legacy authored vertex's rational quarter-turn carrier image."""
    quarter = round(parameter / (0.5 * pi))
    if parameter != quarter * (0.5 * pi):
        return None
    cosine, sine = ((1, 0), (0, 1), (-1, 0), (0, -1))[quarter % 4]
    center = rational_vector(np.asarray(curve.center))
    first = rational_vector(np.asarray(curve.first_axis))
    second = rational_vector(np.asarray(curve.second_axis))
    radius = Fraction(float(curve.radius))
    expected = tuple(
        origin + radius * (cosine * a + sine * b)
        for origin, a, b in zip(center, first, second, strict=True)
    )
    if expected != rational_vector(point):
        return None
    return NativePeriodEndpoint(
        curve, turns=Fraction(quarter, 4), budget=budget, preparations=preparations
    )


def _profile_period_endpoints(
    edge: _ProfileEdge,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[NativePeriodEndpoint | None, NativePeriodEndpoint | None]:
    """Retain mathematical phase atoms only for the legacy vertex-authored arcs."""
    if edge.authored_range or not isinstance(edge.pcurve, CircleCurve):
        return None, None
    if edge.start == edge.end:
        phase = Fraction(edge.first)
        return (
            NativePeriodEndpoint(
                edge.pcurve,
                rational=phase,
                turns=0,
                budget=budget,
                preparations=preparations,
            ),
            NativePeriodEndpoint(
                edge.pcurve,
                rational=phase,
                turns=1,
                budget=budget,
                preparations=preparations,
            ),
        )
    return (
        _quadrant_profile_endpoint(
            edge.pcurve,
            edge.start,
            edge.first,
            budget=budget,
            preparations=preparations,
        ),
        _quadrant_profile_endpoint(
            edge.pcurve,
            edge.end,
            edge.last,
            budget=budget,
            preparations=preparations,
        ),
    )


def _certificate_segment(
    edge: _ProfileEdge,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> CurveTrimSegment:
    """Bind source trims to proved legacy periodic atoms without changing carriers."""
    first, last = _profile_period_endpoints(
        edge, budget=budget, preparations=preparations
    )
    return CurveTrimSegment(
        edge.pcurve,
        edge.first,
        edge.last,
        first_root=first,
        last_root=last,
        budget=budget,
        preparations=preparations,
    )


def _bounded_profile_trim_loop(
    curves: tuple[CurveTrimSegment, ...],
    edges: tuple[_ProfileEdge, ...],
    partition: np.ndarray,
    gaps: tuple[tuple[float, float], ...] | None,
    /,
    *,
    tolerance: float,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> CurveTrimLoop:
    """Build the E-039 homotopy cover, retaining authored junction vertices."""
    relative_closure = 0.0
    chords = None
    if gaps is not None:
        points = tuple(
            np.asarray(
                curve.evaluate(
                    jnp.asarray(curve.parameter_interval),
                    budget=budget,
                    preparations=preparations,
                ),
                dtype=np.float64,
            )
            for curve in curves
        )
        scale = max(1.0, float(np.max(np.abs(np.asarray(points)))))
        closure = max(
            float(np.nextafter(row[1] + gaps[(index + 1) % len(gaps)][0], np.inf))
            for index, row in enumerate(gaps)
        )
        relative_closure = float(np.nextafter(closure / scale, np.inf))
        chord_rows = []
        for curve, edge in zip(curves, edges, strict=True):
            values = np.array(
                curve.evaluate(
                    jnp.asarray(partition[:-1]),
                    budget=budget,
                    preparations=preparations,
                ),
                dtype=np.float64,
                copy=True,
            )
            values[0] = edge.start
            chord_rows.extend(values)
        chords = np.asarray(chord_rows, dtype=np.float64)
    loop = CurveTrimLoop(
        curves,
        tolerance=tolerance,
        maximum_arcs=8192,
        arc_parameters=(partition,) * len(curves),
        chord_vertices=chords,
        relative_closure_tolerance=relative_closure,
        budget=budget,
        preparations=preparations,
    )
    if gaps is None:
        return loop
    # Junction bridges include the authored shared vertex as well as both
    # independently enclosed source endpoint images.
    vertices = np.asarray([edge.end for edge in edges], dtype=np.float64)
    return eqx.tree_at(
        lambda source: (source.junction_lower, source.junction_upper),
        loop,
        (
            np.minimum(loop.junction_lower, vertices),
            np.maximum(loop.junction_upper, vertices),
        ),
    )


def _profile_covers_disjoint(
    loops: tuple[CurveTrimLoop, ...], /, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Separate inter-loop homotopy enclosures, not sampled polylines."""
    for index, first in enumerate(loops):
        for second in loops[index + 1 :]:
            for lower, upper in zip(first.arc_lower, first.arc_upper, strict=True):
                budget.reserve(second.arc_lower.shape[0])
                if np.any(
                    np.all(
                        (upper >= second.arc_lower) & (second.arc_upper >= lower), axis=1
                    )
                ):
                    return False
    return True


def _profile_cover_inputs(
    edges: tuple[tuple[_ProfileEdge, ...], ...],
    /,
    *,
    require_exact_joins: bool,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[CurveTrimSegment, ...], ...], float]:
    """Prepare source trims and a fixed chart-cover scale for E-039."""
    curves = tuple(
        tuple(
            _certificate_segment(edge, budget=budget, preparations=preparations)
            for edge in loop
        )
        for loop in edges
    )
    if require_exact_joins:
        for loop in curves:
            if any(
                not first.shares_endpoint(
                    second, preparations=preparations, budget=budget
                )
                for first, second in zip(loop, (*loop[1:], loop[0]), strict=True)
            ):
                raise ValueError(
                    "PlanarProfile.junctions must retain source endpoint identity; got incompatible source endpoints"
                )
    boxes = np.stack(
        [
            edge.pcurve.bounding_box(
                edge.first, edge.last, preparations=preparations, budget=budget
            )
            for loop in edges
            for edge in loop
        ]
    )
    scale = float(
        np.linalg.norm(np.max(boxes[:, 1], axis=0) - np.min(boxes[:, 0], axis=0))
    )
    if scale == 0.0:
        raise ValueError(
            "PlanarProfile.loops must enclose positive area; got a zero-area loop"
        )
    return curves, scale


def _affine_profile_homotopy_hull(
    loop: CurveTrimLoop, arc: int, /, *, budget: CoordinateEnclosureBudget
) -> _ProfileHull | None:
    """Enclose an affine source/chord homotopy over one prescribed arc."""
    segment = loop.curves[int(loop.arc_curves[arc])]
    if not isinstance(segment, CurveTrimSegment) or not isinstance(
        segment.curve, AbstractCurve
    ):
        return None
    if segment.first_root is not None or segment.last_root is not None:
        return None
    coefficients = _admitted_affine_chord_coefficients(segment.curve, budget=budget)
    if coefficients is None:
        return None
    origin, direction = coefficients
    scalars = rational_points(
        np.asarray(
            (
                (segment.first, segment.last),
                (loop.arc_first[arc], loop.arc_last[arc]),
            ),
            dtype=np.float64,
        ),
        budget=budget,
    )
    reserve_fraction_work((*coefficients, *scalars), 24, 20, 8, budget=budget)
    points = []
    for local in scalars[1]:
        fraction = 1 - local if segment.reversed else local
        parameter = scalars[0][0] + fraction * (scalars[0][1] - scalars[0][0])
        points.append(
            (
                origin[0] + parameter * direction[0],
                origin[1] + parameter * direction[1],
            )
        )
    points.extend(
        rational_points(
            np.stack((loop.chords[arc], loop.chords[(arc + 1) % loop.chords.shape[0]])),
            budget=budget,
        )
    )
    source_index = int(loop.arc_curves[arc])
    junctions = []
    if loop.arc_first[arc] == 0.0:
        junctions.append((source_index - 1) % len(loop.curves))
    if loop.arc_last[arc] == 1.0:
        junctions.append(source_index)
    for junction in junctions:
        lower, upper = loop.junction_lower[junction], loop.junction_upper[junction]
        points.extend(
            rational_points(
                np.asarray(
                    (
                        (lower[0], lower[1]),
                        (lower[0], upper[1]),
                        (upper[0], lower[1]),
                        (upper[0], upper[1]),
                    ),
                    dtype=np.float64,
                ),
                budget=budget,
            )
        )
    budget.reserve(len(points), 64 * len(points))
    reserve_fraction_work(tuple(points), len(points), 1, 1, budget=budget)
    weight = Fraction(1)
    return _ProfileHull(
        tuple(
            (
                rational_coefficient(x, role="brep.profile_homotopy_coordinate"),
                rational_coefficient(y, role="brep.profile_homotopy_coordinate"),
                weight,
            )
            for x, y in points
        )
    )


def _affine_profile_pairs_separated(
    loop: CurveTrimLoop,
    pairs: tuple[tuple[int, int], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove E-039 nonadjacent separation with rational convex-hull planes.

    Axis-aligned boxes of separated affine rays can overlap at every uniform
    refinement near a sharp corner. Their source/chord convex hulls retain a
    separating rational plane. Regularity and adjacent-pair failures are never
    cleared by this proof.
    """
    count = loop.chords.shape[0]
    hulls: dict[int, _ProfileHull | None] = {}
    for first, second in pairs:
        if (first - second) % count in (0, 1, count - 1):
            return False
        for arc in (first, second):
            if arc not in hulls:
                hulls[arc] = _affine_profile_homotopy_hull(loop, arc, budget=budget)
        left, right = hulls[first], hulls[second]
        if left is None or right is None:
            return False
        with budget.temporary_scope():
            points = (*left.points, *right.points)
            work = 32 * len(points) ** 3
            reserve_fraction_work(points, work, work, 32, budget=budget)
            if not _separated_profile_hulls(left, right, frozenset()):
                return False
    return True


def _profile_cover_ready(
    loops: tuple[CurveTrimLoop, ...],
    /,
    *,
    separate_homotopies: bool,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Require regular loop isotopies and, for gap topology, inter-loop separation."""
    certificates = tuple(
        loop.certify_topology(preparations=preparations, budget=budget) for loop in loops
    )
    if any(certificate.budget_exhausted for certificate in certificates):
        raise ValueError(
            "PlanarProfile.topology must resolve within the source-pair budget; got an exhausted pair budget"
        )
    for loop, certificate in zip(loops, certificates, strict=True):
        if certificate.certified:
            continue
        with budget.temporary_scope():
            if not _affine_profile_pairs_separated(
                loop,
                certificate.unresolved_pairs,
                budget=budget,
            ):
                return False
    return not separate_homotopies or _profile_covers_disjoint(loops, budget=budget)


def _accepted_profile_polygons(
    loops: tuple[CurveTrimLoop, ...],
    edges: tuple[tuple[_ProfileEdge, ...], ...],
    /,
    *,
    gap_topology: bool,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Point, ...], ...]:
    """Apply orientation and containment only after source isotopy certification."""
    if not gap_topology:
        _require_mixed_profile_relations(
            loops, edges, preparations=preparations, budget=budget
        )
    polygons = tuple(
        tuple(
            (Fraction(float(point[0])), Fraction(float(point[1])))
            for point in loop.chords
        )
        for loop in loops
    )
    if any(area(polygon) == 0 for polygon in polygons):
        raise ValueError(
            "PlanarProfile.loops must enclose positive area; got a zero-area loop"
        )
    if gap_topology:
        _require_profile_containment(polygons)
    return polygons


def _mixed_profile_polygons(
    edges: tuple[tuple[_ProfileEdge, ...], ...],
    /,
    *,
    endpoint_gaps: tuple[tuple[tuple[float, float], ...], ...] | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Point, ...], ...]:
    """Return polygons after certifying mixed native curve loops."""
    curves, tolerance = _profile_cover_inputs(
        edges,
        require_exact_joins=endpoint_gaps is None,
        preparations=preparations,
        budget=budget,
    )
    segments = 4
    while segments * sum(len(loop) for loop in curves) <= 8192:
        partition = np.linspace(0.0, 1.0, segments + 1, dtype=np.float64)
        loops = tuple(
            _bounded_profile_trim_loop(
                loop,
                source,
                partition,
                None if endpoint_gaps is None else endpoint_gaps[index],
                tolerance=tolerance,
                preparations=preparations,
                budget=budget,
            )
            for index, (loop, source) in enumerate(zip(curves, edges, strict=True))
        )
        if _profile_cover_ready(
            loops,
            separate_homotopies=endpoint_gaps is not None,
            preparations=preparations,
            budget=budget,
        ):
            return _accepted_profile_polygons(
                loops,
                edges,
                gap_topology=endpoint_gaps is not None,
                preparations=preparations,
                budget=budget,
            )
        segments *= 2
    raise ValueError(
        "PlanarProfile.topology must resolve within its arc budget; got unresolved source/chord topology"
    )


@checked
def reversed_bspline(curve: BSplineCurve, /) -> BSplineCurve:
    """The same rational spline traversed backwards: ``t -> -t`` on mirrored knots."""
    return eqx.tree_at(
        lambda source: (source.control_points, source.weights, source.knots),
        curve,
        (curve.control_points[::-1], curve.weights[::-1], -curve.knots[::-1]),
    )


def _reflected_conic(curve: CircleCurve | EllipseCurve, /) -> CircleCurve | EllipseCurve:
    """Reflect parameters algebraically: the image at ``t`` is the source at ``-t``."""
    return eqx.tree_at(lambda source: source.second_axis, curve, -curve.second_axis)


def _reverse_loop(loop: ProfileLoop, /) -> ProfileLoop:
    count = len(loop.vertices)
    vertices = (loop.vertices[0],) + tuple(reversed(loop.vertices[1:]))
    segments: list[ProfileSegment] = []
    for index in range(count):
        # Reversed segment joins vertex -index to -(index + 1) in the source loop.
        original = loop.segments[(count - 1 - index) % count]
        match original:
            case ProfileLine():
                segments.append(original)
            case ProfileArc(center=center, counterclockwise=counterclockwise):
                segments.append(ProfileArc(center, not counterclockwise))
            case BSplineCurve():
                segments.append(reversed_bspline(original))
            case ProfileCurveUse(curve=curve, first=first, last=last):
                segments.append(ProfileCurveUse(curve, last, first))
            case _:
                raise TypeError(
                    f"ProfileLoop.segment must be a supported sketch segment; got {type(original).__name__}"
                )
    return ProfileLoop(vertices, tuple(segments))


@dataclass(frozen=True, slots=True)
class PlanarProfile:
    """Planar region bounded by an outer loop and disjoint interior holes.

    Source Bernstein hull separation and strict monotone projections certify
    spline topology before orientation and containment use their isotopic
    control polygons. Mixed families use the source/chord isotopy
    certificate, source intersections and source trim classification.
    Analytic arc loops use polynomial and quadratic radical exact predicates.
    Unresolved singularities and exhausted certificates refuse construction;
    sampled chords never decide profile acceptance.
    ``endpoint_gap_bounds`` optionally declares one pair of maximum Euclidean
    endpoint distances per segment, grouped by outer loop then holes. Every row
    is independently recomputed; absent rows require zero conic endpoint gaps.
    Certified junction bridges retain the authored vertices without snapping. Gap-bearing
    profiles use E-039 homotopy covers before exact orientation and containment.
    ``coordinate_budget`` borrows the caller's preparation ticket without storing
    it in the profile or its identity; omission retains the default source request.
    """

    plane: ProfilePlane
    outer: ProfileLoop
    holes: tuple[ProfileLoop, ...] = ()
    residual_bound: float = field(default=0.0, kw_only=True)
    endpoint_gap_bounds: tuple[tuple[tuple[float, float], ...], ...] | None = field(
        default=None, kw_only=True
    )
    coordinate_budget: InitVar[CoordinateEnclosureBudget | None] = field(
        default=None, kw_only=True
    )
    edges: tuple[tuple[_ProfileEdge, ...], ...] = field(init=False, repr=False)

    def __post_init__(self, coordinate_budget: CoordinateEnclosureBudget | None) -> None:
        """Normalize the loops and retain weighted sketch residual evidence.

        ``residual_bound`` is solver residual evidence, not a spatial length or
        a source-geometry tolerance. Zero represents an exactly authored profile.
        """
        object.__setattr__(
            self,
            "residual_bound",
            nonnegative_real_scalar(self.residual_bound, "PlanarProfile.residual_bound"),
        )
        preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (self.plane, self.outer, self.holes), budget=coordinate_budget
        ) as budget:
            if not isinstance(self.plane, ProfilePlane):
                raise TypeError(
                    f"PlanarProfile.plane must be a ProfilePlane; got {type(self.plane).__name__}"
                )
            loops = (self.outer, *tuple(self.holes))
            if not all(isinstance(loop, ProfileLoop) for loop in loops):
                raise TypeError(
                    "PlanarProfile.loops must be ProfileLoop values; got an unsupported loop"
                )
            original_edges = tuple(_loop_edges(loop) for loop in loops)
            conic_use = any(
                isinstance(segment, ProfileCurveUse)
                for loop in loops
                for segment in loop.segments
            )
            gap_rows = _profile_endpoint_gaps(
                original_edges,
                self.endpoint_gap_bounds,
                conic_use=conic_use,
                budget=budget,
            )
            object.__setattr__(self, "endpoint_gap_bounds", gap_rows)
            spline = any(
                isinstance(edge.pcurve, BSplineCurve)
                for loop in original_edges
                for edge in loop
            )
            arc = any(
                isinstance(edge.pcurve, CircleCurve)
                for loop in original_edges
                for edge in loop
            )
            areas: tuple[Fraction | int, ...]
            if conic_use or gap_rows is not None:
                polygons = _mixed_profile_polygons(
                    original_edges,
                    endpoint_gaps=gap_rows,
                    preparations=preparations,
                    budget=budget,
                )
                areas = tuple(area(polygon) for polygon in polygons)
            elif spline:
                polygons = (
                    _mixed_profile_polygons(
                        original_edges, preparations=preparations, budget=budget
                    )
                    if arc
                    else _profile_polygons(original_edges)
                )
                areas = tuple(area(polygon) for polygon in polygons)
            else:
                _require_analytic_profile_topology(original_edges)
                areas = tuple(
                    _analytic_profile_orientation(loop) for loop in original_edges
                )
            canonical = []
            for index, (loop, area_bank) in enumerate(zip(loops, areas, strict=True)):
                if area_bank == 0:
                    raise ValueError(
                        "PlanarProfile.loops must enclose positive area; got a zero-area loop"
                    )
                wanted = 1 if index == 0 else -1
                canonical.append(loop if area_bank * wanted > 0 else _reverse_loop(loop))
            edges = tuple(_loop_edges(loop) for loop in canonical)
            if gap_rows is not None:
                gap_rows = tuple(
                    rows
                    if area_bank * (1 if index == 0 else -1) > 0
                    else tuple((b, a) for a, b in reversed(rows))
                    for index, (rows, area_bank) in enumerate(
                        zip(gap_rows, areas, strict=True)
                    )
                )
                object.__setattr__(self, "endpoint_gap_bounds", gap_rows)
            object.__setattr__(self, "outer", canonical[0])
            object.__setattr__(self, "holes", tuple(canonical[1:]))
            object.__setattr__(self, "edges", edges)

    @property
    def loops(self) -> tuple[ProfileLoop, ...]:
        return (self.outer, *self.holes)


def _crossing_parity(points: np.ndarray, polyline: np.ndarray, /) -> np.ndarray:
    start = polyline
    end = np.roll(polyline, -1, axis=0)
    x, y = points[:, :1], points[:, 1:2]
    crossing = (start[:, 1] > y) != (end[:, 1] > y)
    denominator = np.where(end[:, 1] != start[:, 1], end[:, 1] - start[:, 1], 1.0)
    intersection = (end[:, 0] - start[:, 0]) * (y - start[:, 1]) / denominator + start[
        :, 0
    ]
    return np.sum(crossing & (x < intersection), axis=1) % 2 == 1


type _RadicalPoint = tuple[Point, Point]


def _radical_sign(a: Fraction, b: Fraction, radicand: Fraction, /) -> int:
    """Return the sign of ``a + b sqrt(radicand)`` without evaluating a root."""
    if radicand < 0:
        raise ValueError(f"PlanarProfile.radicand must be nonnegative; got {radicand}")
    if b == 0 or radicand == 0:
        return (a > 0) - (a < 0)
    if a == 0 or (a > 0) == (b > 0):
        return (b > 0) - (b < 0)
    comparison = a * a - b * b * radicand
    return ((a > 0) - (a < 0)) * ((comparison > 0) - (comparison < 0))


def _arc_circle(edge: _ProfileEdge, /) -> tuple[Point, Fraction]:
    if not isinstance(edge.pcurve, CircleCurve):
        raise TypeError(
            f"ProfileArc.carrier must be a CircleCurve; got {type(edge.pcurve).__name__}"
        )
    center = tuple(float(v) for v in np.asarray(edge.pcurve.center))
    if len(center) != 2:
        raise ValueError(f"ProfileArc.center must be planar; got dimension {len(center)}")
    center_ = Fraction(center[0]), Fraction(center[1])
    delta = sub(rational_vector(edge.start), center_)
    return center_, delta[0] ** 2 + delta[1] ** 2


def _arc_contains_radical(
    edge: _ProfileEdge, point: _RadicalPoint, radicand: Fraction, /
) -> bool:
    if edge.start == edge.end:
        return True
    center, _ = _arc_circle(edge)
    start = sub(rational_vector(edge.start), center)
    end = sub(rational_vector(edge.end), center)
    base, radical = point
    relative = sub(base, center)
    turn = 1 if edge.turn > 0 else -1
    before = turn * _radical_sign(
        rational_cross_2d(start, relative), rational_cross_2d(start, radical), radicand
    )
    after = turn * _radical_sign(
        rational_cross_2d(relative, end), rational_cross_2d(radical, end), radicand
    )
    minor = turn * rational_cross_2d(start, end) >= 0
    return before >= 0 and after >= 0 if minor else before >= 0 or after >= 0


def _radical_is_vertex(
    point: _RadicalPoint, radicand: Fraction, vertices: frozenset[Point], /
) -> bool:
    base, radical = point
    return any(
        _radical_sign(base[0] - vertex[0], radical[0], radicand) == 0
        and _radical_sign(base[1] - vertex[1], radical[1], radicand) == 0
        for vertex in vertices
    )


def _line_circle_roots(
    start: Point, direction: Point, arc: _ProfileEdge, /
) -> tuple[Fraction, tuple[tuple[Fraction, Fraction], ...]]:
    center, radius_squared = _arc_circle(arc)
    relative = sub(start, center)
    a = direction[0] ** 2 + direction[1] ** 2
    b = 2 * (relative[0] * direction[0] + relative[1] * direction[1])
    c = relative[0] ** 2 + relative[1] ** 2 - radius_squared
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return discriminant, ()
    roots = ((-b / (2 * a), -1 / (2 * a)), (-b / (2 * a), 1 / (2 * a)))
    return discriminant, roots[:1] if discriminant == 0 else roots


def _line_arc_contact(
    line: _ProfileEdge, arc: _ProfileEdge, allowed: frozenset[Point], /
) -> bool:
    start = rational_vector(line.start)
    direction = sub(rational_vector(line.end), start)
    discriminant, roots = _line_circle_roots(start, direction, arc)
    for a, b in roots:
        if (
            _radical_sign(a, b, discriminant) < 0
            or _radical_sign(a - 1, b, discriminant) > 0
        ):
            continue
        point = (
            (start[0] + a * direction[0], start[1] + a * direction[1]),
            (b * direction[0], b * direction[1]),
        )
        if _arc_contains_radical(arc, point, discriminant) and not _radical_is_vertex(
            point, discriminant, allowed
        ):
            return True
    return False


def _arc_arc_contact(
    first: _ProfileEdge, second: _ProfileEdge, allowed: frozenset[Point], /
) -> bool:
    a, ra = _arc_circle(first)
    b, rb = _arc_circle(second)
    delta = sub(b, a)
    distance_squared = delta[0] ** 2 + delta[1] ** 2
    if distance_squared == 0:
        if ra != rb:
            return False
        if first.start == first.end or second.start == second.end:
            return True
        for edge, other in ((first, second), (second, first)):
            for vertex in (edge.start, edge.end):
                point = rational_vector(vertex)
                if point not in allowed and _arc_contains_radical(
                    other, (point, (Fraction(), Fraction())), Fraction()
                ):
                    return True
            start = sub(rational_vector(edge.start), a)
            end = sub(rational_vector(edge.end), a)
            middle = (start[0] + end[0], start[1] + end[1])
            if middle == (0, 0):
                turn = 1 if edge.turn > 0 else -1
                middle = (-turn * start[1], turn * start[0])
            elif (1 if edge.turn > 0 else -1) * rational_cross_2d(start, end) < 0:
                middle = (-middle[0], -middle[1])
            candidate = (a[0] + middle[0], a[1] + middle[1])
            if _arc_contains_radical(
                other, (candidate, (Fraction(), Fraction())), Fraction()
            ):
                return True
        return False
    along = (ra - rb + distance_squared) / (2 * distance_squared)
    radicand = ra / distance_squared - along * along
    if radicand < 0:
        return False
    base = (a[0] + along * delta[0], a[1] + along * delta[1])
    for sign in (1,) if radicand == 0 else (-1, 1):
        point = (base, (-sign * delta[1], sign * delta[0]))
        if (
            _arc_contains_radical(first, point, radicand)
            and _arc_contains_radical(second, point, radicand)
            and not _radical_is_vertex(point, radicand, allowed)
        ):
            return True
    return False


def _analytic_loop_contains(point: Point, loop: tuple[_ProfileEdge, ...], /) -> bool:
    # A generic rational ray avoids all boundary vertices and tangencies.
    direction: Point | None = None
    for slope in range(4 * len(loop) + 1):
        candidate = (Fraction(1), Fraction(slope))
        if any(
            rational_cross_2d(sub(rational_vector(edge.start), point), candidate) == 0
            for edge in loop
        ):
            continue
        if any(
            _line_circle_roots(point, candidate, edge)[0] == 0
            for edge in loop
            if isinstance(edge.pcurve, CircleCurve)
        ):
            continue
        direction = candidate
        break
    if direction is None:
        raise ValueError(
            "PlanarProfile.containment_ray must avoid degeneracy; got unresolved ray degeneracy"
        )
    crossings = 0
    for edge in loop:
        match edge.pcurve:
            case LineCurve():
                start = rational_vector(edge.start)
                delta = sub(rational_vector(edge.end), start)
                determinant = rational_cross_2d(direction, delta)
                if determinant == 0:
                    continue
                relative = sub(start, point)
                ray = rational_cross_2d(relative, delta) / determinant
                segment = rational_cross_2d(relative, direction) / determinant
                crossings += ray > 0 and 0 < segment < 1
            case CircleCurve():
                discriminant, roots = _line_circle_roots(point, direction, edge)
                for a, b in roots:
                    if _radical_sign(a, b, discriminant) <= 0:
                        continue
                    root = (
                        (point[0] + a * direction[0], point[1] + a * direction[1]),
                        (b * direction[0], b * direction[1]),
                    )
                    crossings += _arc_contains_radical(edge, root, discriminant)
            case _:
                raise TypeError(
                    "PlanarProfile.analytic_carrier must be a line or circle; got an unsupported carrier"
                )
    return crossings % 2 == 1


def _analytic_segment_clear(
    start: Point, end: Point, loop: tuple[_ProfileEdge, ...], /
) -> bool:
    """Exclude all source-boundary contacts except at the starting vertex."""
    direction = sub(end, start)
    for edge in loop:
        if isinstance(edge.pcurve, LineCurve):
            contacts = intersections(
                (start, end), (rational_vector(edge.start), rational_vector(edge.end))
            )
            if any(point != start for point in contacts):
                return False
        else:
            discriminant, roots = _line_circle_roots(start, direction, edge)
            for a, b in roots:
                if (
                    _radical_sign(a, b, discriminant) <= 0
                    or _radical_sign(a - 1, b, discriminant) > 0
                ):
                    continue
                point = (
                    (start[0] + a * direction[0], start[1] + a * direction[1]),
                    (b * direction[0], b * direction[1]),
                )
                if _arc_contains_radical(edge, point, discriminant):
                    return False
    return True


def _analytic_profile_orientation(loop: tuple[_ProfileEdge, ...], /) -> int:
    """Determine the traversal's interior side using rational queries."""
    if len(loop) == 1 and isinstance(loop[0].pcurve, CircleCurve):
        return 1 if loop[0].turn > 0 else -1
    for edge in loop:
        if isinstance(edge.pcurve, LineCurve):
            start, end = rational_vector(edge.start), rational_vector(edge.end)
            point = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            tangent = sub(end, start)
            normal = (-tangent[1], tangent[0])
            break
    else:
        point = rational_vector(loop[0].start)
        normals = []
        for edge in (loop[-1], loop[0]):
            center, _ = _arc_circle(edge)
            radial = sub(point, center)
            turn = 1 if edge.turn > 0 else -1
            normals.append((-turn * radial[0], -turn * radial[1]))
        first, last = normals
        dot = first[0] * last[0] + first[1] * last[1]
        if dot < 0:
            first_squared = first[0] ** 2 + first[1] ** 2
            last_squared = last[0] ** 2 + last[1] ** 2
            lower, upper = -dot / last_squared, first_squared / -dot
            if lower >= upper:
                raise ValueError(
                    "PlanarProfile.orientation must resolve a regular local side; got an unresolved cusp"
                )
            weight = (lower + upper) / 2
        else:
            weight = Fraction(1)
        normal = (first[0] + weight * last[0], first[1] + weight * last[1])
    offset = Fraction(1)
    for _ in range(256):
        left = (point[0] + offset * normal[0], point[1] + offset * normal[1])
        right = (point[0] - offset * normal[0], point[1] - offset * normal[1])
        if _analytic_segment_clear(point, left, loop) and _analytic_segment_clear(
            point, right, loop
        ):
            left_inside = _analytic_loop_contains(left, loop)
            right_inside = _analytic_loop_contains(right, loop)
            if left_inside != right_inside:
                return 1 if left_inside else -1
        offset /= 2
    raise ValueError(
        "PlanarProfile.orientation must resolve a regular local side; got an unresolved local-side certificate"
    )


def _require_analytic_profile_topology(
    edges: tuple[tuple[_ProfileEdge, ...], ...], /
) -> None:
    entries = [
        (li, ei, edge) for li, loop in enumerate(edges) for ei, edge in enumerate(loop)
    ]
    for index, (li, ei, first) in enumerate(entries):
        for lj, ej, second in entries[index + 1 :]:
            allowed: set[Point] = set()
            if li == lj:
                if (ei + 1) % len(edges[li]) == ej:
                    allowed.add(rational_vector(first.end))
                if (ej + 1) % len(edges[li]) == ei:
                    allowed.add(rational_vector(second.end))
            vertices = frozenset(allowed)
            match first.pcurve, second.pcurve:
                case LineCurve(), LineCurve():
                    contacts = intersections(
                        (rational_vector(first.start), rational_vector(first.end)),
                        (rational_vector(second.start), rational_vector(second.end)),
                    )
                    contact = len(contacts) > 1 or any(
                        p not in vertices for p in contacts
                    )
                case LineCurve(), CircleCurve():
                    contact = _line_arc_contact(first, second, vertices)
                case CircleCurve(), LineCurve():
                    contact = _line_arc_contact(second, first, vertices)
                case CircleCurve(), CircleCurve():
                    contact = _arc_arc_contact(first, second, vertices)
                case _:
                    raise TypeError(
                        "PlanarProfile.analytic_carrier must be a line or circle; got an unsupported carrier"
                    )
            if contact:
                raise ValueError(
                    "PlanarProfile.loops must be simple and must not touch or cross; got source-boundary contact"
                )
    for index, hole in enumerate(edges[1:]):
        if not _analytic_loop_contains(rational_vector(hole[0].start), edges[0]):
            raise ValueError(
                "PlanarProfile.holes must lie inside the outer loop; got an exterior hole"
            )
        for other in edges[index + 2 :]:
            if _analytic_loop_contains(
                rational_vector(hole[0].start), other
            ) or _analytic_loop_contains(rational_vector(other[0].start), hole):
                raise ValueError(
                    "PlanarProfile.holes must not be nested; got nested holes"
                )


# ------------------------------------------------------------------ builder


def _rotation(axis: np.ndarray, angle: float, /) -> np.ndarray:
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    return np.eye(3) + sin(angle) * cross + (1.0 - cos(angle)) * (cross @ cross)


def _space_curve(edge: _ProfileEdge, plane: ProfilePlane, /) -> AbstractCurve:
    pcurve = edge.pcurve
    match pcurve:
        case LineCurve():
            origin = plane.point(np.asarray(pcurve.origin))
            return _endpoint_line_curve(
                origin,
                plane.vector(np.asarray(pcurve.direction)),
                edge.first,
                edge.last,
                plane.point(edge.start),
                plane.point(edge.end),
            )
        case CircleCurve():
            return CircleCurve(
                plane.point(np.asarray(pcurve.center)),
                plane.vector(np.asarray(pcurve.first_axis)),
                plane.vector(np.asarray(pcurve.second_axis)),
                pcurve.radius,
            )
        case EllipseCurve():
            return EllipseCurve(
                plane.point(np.asarray(pcurve.center)),
                plane.vector(np.asarray(pcurve.first_axis)),
                plane.vector(np.asarray(pcurve.second_axis)),
                pcurve.first_radius,
                pcurve.second_radius,
            )
        case BSplineCurve():
            return BSplineCurve(
                plane.point(np.asarray(pcurve.control_points)),
                pcurve.weights,
                pcurve.knots,
                pcurve.degree,
            )
        case _:
            raise TypeError(
                f"ProfileLoop.carrier must be a supported sketch carrier; got {type(pcurve).__name__}"
            )


def _moved_curve(
    curve: AbstractCurve, rotation: np.ndarray, pivot: np.ndarray, offset: np.ndarray, /
) -> AbstractCurve:
    """Rigid image ``pivot + rotation (x - pivot) + offset`` of a carrier."""

    def point(value: Array) -> np.ndarray:
        return pivot + rotation @ (np.asarray(value) - pivot) + offset

    match curve:
        case LineCurve():
            return LineCurve(point(curve.origin), rotation @ np.asarray(curve.direction))
        case CircleCurve():
            return CircleCurve(
                point(curve.center),
                rotation @ np.asarray(curve.first_axis),
                rotation @ np.asarray(curve.second_axis),
                curve.radius,
            )
        case EllipseCurve():
            return EllipseCurve(
                point(curve.center),
                rotation @ np.asarray(curve.first_axis),
                rotation @ np.asarray(curve.second_axis),
                curve.first_radius,
                curve.second_radius,
            )
        case BSplineCurve():
            controls = np.asarray(curve.control_points)
            return BSplineCurve(
                pivot + (controls - pivot) @ rotation.T + offset,
                curve.weights,
                curve.knots,
                curve.degree,
            )
        case _:
            raise TypeError(
                f"Builder.carrier must be a supported sweep carrier; got {type(curve).__name__}"
            )


@dataclass(slots=True)
class _Face:
    patch: AbstractSurfacePatch
    box: np.ndarray
    loops: list[list[int]]
    orientation: int
    tag: str


@dataclass(slots=True)
class Builder:
    """Stage one construction in mutable host storage."""

    vertices: list[np.ndarray] = field(default_factory=list)
    curves: list[BRepCurve] = field(default_factory=list)
    edge_curves: list[int] = field(default_factory=list)
    edge_ranges: list[tuple[float, float]] = field(default_factory=list)
    edge_vertices: list[tuple[int, int]] = field(default_factory=list)
    edge_vertex_gap_bounds: tuple[tuple[float, float], ...] | None = None
    recompute_endpoint_gaps: bool = False
    edge_endpoint_roots: list[
        tuple[NativePeriodEndpoint | None, NativePeriodEndpoint | None]
    ] = field(default_factory=list)
    pcurves: list[BRepPCurve] = field(default_factory=list)
    coedge_edges: list[int] = field(default_factory=list)
    coedge_senses: list[int] = field(default_factory=list)
    faces: list[_Face] = field(default_factory=list)
    coedge_endpoint_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = field(
        default_factory=list
    )

    def vertex(self, point: np.ndarray, /) -> int:
        self.vertices.append(np.asarray(point, dtype=np.float64))
        return len(self.vertices) - 1

    def edge(
        self, curve: BRepCurve, first: float, last: float, start: int, end: int, /
    ) -> int:
        self.curves.append(curve)
        self.edge_curves.append(len(self.curves) - 1)
        self.edge_ranges.append((float(first), float(last)))
        self.edge_vertices.append((start, end))
        self.edge_endpoint_roots.append((None, None))
        return len(self.edge_curves) - 1

    def degenerate(self, vertex: int, first: float, last: float, /) -> int:
        self.edge_curves.append(-1)
        self.edge_ranges.append((float(first), float(last)))
        self.edge_vertices.append((vertex, vertex))
        self.edge_endpoint_roots.append((None, None))
        return len(self.edge_curves) - 1

    def coedge(
        self,
        edge: int,
        sense: int,
        pcurve: BRepPCurve,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> int:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    edge,
                    sense,
                    pcurve,
                )
            ) as budget:
                return self.coedge(
                    edge, sense, pcurve, preparations=preparations, budget=budget
                )
        self.coedge_edges.append(edge)
        self.coedge_senses.append(sense)
        self.pcurves.append(pcurve)
        self.coedge_endpoint_roots.append(
            self._coedge_endpoints(edge, pcurve, preparations=preparations, budget=budget)
        )
        return len(self.coedge_edges) - 1

    def _coedge_endpoints(
        self,
        edge: int,
        pcurve: BRepPCurve,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[RootEndpoint | None, RootEndpoint | None]:
        first, last = self.edge_endpoint_roots[edge]

        def bind(
            endpoint: NativePeriodEndpoint | None,
            /,
            *,
            budget: CoordinateEnclosureBudget = budget,
        ) -> NativePeriodEndpoint | None:
            return (
                None
                if endpoint is None
                else NativePeriodEndpoint(
                    pcurve,
                    endpoint.patch,
                    endpoint.axis,
                    rational=endpoint.rational,
                    turns=endpoint.turns,
                )
            )

        return bind(first), bind(last)

    def native_phase(
        self,
        edge: int,
        patch: AbstractSurfacePatch | None,
        axis: Literal[0, 1] | None,
        /,
        *,
        rational: Fraction = Fraction(),
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        """Author a full native phase on an edge and every source parameter use."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    edge,
                    patch,
                    axis,
                )
            ) as budget:
                return self.native_phase(
                    edge,
                    patch,
                    axis,
                    rational=rational,
                    preparations=preparations,
                    budget=budget,
                )
        curve_index = self.edge_curves[edge]
        curve = (
            self.pcurves[
                next(
                    index
                    for index, source_edge in enumerate(self.coedge_edges)
                    if source_edge == edge
                )
            ]
            if curve_index < 0
            else self.curves[curve_index]
        )
        self.edge_endpoint_roots[edge] = (
            NativePeriodEndpoint(
                curve,
                patch,
                axis,
                rational=rational,
                turns=0,
                budget=budget,
                preparations=preparations,
            ),
            NativePeriodEndpoint(
                curve,
                patch,
                axis,
                rational=rational,
                turns=1,
                budget=budget,
                preparations=preparations,
            ),
        )
        for coedge, source_edge in enumerate(self.coedge_edges):
            if source_edge == edge:
                self.coedge_endpoint_roots[coedge] = self._coedge_endpoints(
                    edge, self.pcurves[coedge], preparations=preparations, budget=budget
                )

    def face(
        self,
        patch: AbstractSurfacePatch,
        box: np.ndarray,
        loops: list[list[int]],
        tag: str,
        outward: tuple[int, float, np.ndarray] | None,
        /,
        orientation: int = 1,
    ) -> int:
        """Stage a face; ``outward = (coedge, parameter, direction)`` fixes its sense.

        The orientation is the sign of the parametric normal against a known
        outward direction at one coedge sample.
        """
        if outward is not None:
            coedge, parameter, direction = outward
            uv = self.pcurves[coedge].evaluate(jnp.asarray(parameter))
            differential = np.asarray(_surface_jacobian(patch, uv))
            normal = np.cross(differential[:, 0], differential[:, 1])
            value = float(normal @ direction)
            if abs(value) <= 1.0e-12 * max(1.0, float(np.linalg.norm(normal))):
                raise ValueError(
                    "Builder.face must be transverse to its sweep direction; got a tangent swept face"
                )
            orientation = 1 if value > 0.0 else -1
        self.faces.append(
            _Face(patch, np.asarray(box, dtype=np.float64), loops, orientation, tag)
        )
        return len(self.faces) - 1


def _bind_profile_period_endpoints(
    builder: Builder,
    index: int,
    source: _ProfileEdge,
    /,
    *,
    patch: AbstractSurfacePatch | None = None,
    axis: Literal[0, 1] | None = None,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Transport legacy phase witnesses to spatial carriers and their p-curve uses."""
    roots = _profile_period_endpoints(source, budget=budget, preparations=preparations)
    if roots == (None, None):
        return
    curve = builder.curves[builder.edge_curves[index]]

    def bind(root: NativePeriodEndpoint | None) -> NativePeriodEndpoint | None:
        if root is None:
            return None
        return NativePeriodEndpoint(
            curve,
            patch,
            axis,
            rational=root.rational,
            turns=root.turns,
            budget=budget,
            preparations=preparations,
        )

    builder.edge_endpoint_roots[index] = (bind(roots[0]), bind(roots[1]))
    for coedge, edge in enumerate(builder.coedge_edges):
        if edge == index:
            builder.coedge_endpoint_roots[coedge] = builder._coedge_endpoints(
                index, builder.pcurves[coedge], budget=budget, preparations=preparations
            )


def _builder_endpoint_gap_rows(
    builder: Builder, /, *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[float, float], ...]:
    """Certify lifted coefficients anew against the published vertices."""
    rows = []
    for edge, (curve_index, parameters, (first, last)) in enumerate(
        zip(builder.edge_curves, builder.edge_ranges, builder.edge_vertices, strict=True)
    ):
        curve = builder.curves[curve_index]
        if not isinstance(curve, AbstractCurve):
            raise TypeError(
                f"Builder.gap_carrier must be a native curve; got {type(curve).__name__}"
            )
        rows.append(
            endpoint_gap_bounds(
                curve,
                parameters,
                np.stack((builder.vertices[first], builder.vertices[last])),
                budget=budget,
                endpoint_roots=builder.edge_endpoint_roots[edge],
            )
        )
    return tuple(rows)


@eqx.filter_jit
def _surface_jacobian(patch: AbstractSurfacePatch, uv: Array, /) -> Array:
    """One compiled chart Jacobian per carrier structure, coefficients dynamic."""
    return jax.jacfwd(patch.evaluate)(uv)


@eqx.filter_jit
def _curve_velocity(curve: AbstractCurve, parameter: Array, /) -> Array:
    """One compiled carrier tangent per curve structure, coefficients dynamic."""
    return jax.jacfwd(curve.evaluate)(parameter)


def _curve_tangent(curve: AbstractCurve, parameter: float, /) -> np.ndarray:
    return np.asarray(_curve_velocity(curve, jnp.asarray(parameter, dtype=jnp.float64)))


def _profile_box(
    profile: PlanarProfile,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    boxes = np.stack(
        [
            edge.pcurve.bounding_box(
                edge.first, edge.last, preparations=preparations, budget=budget
            )
            for edge in profile.edges[0]
        ]
    )
    return np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))


def _cap(
    builder: Builder,
    profile: PlanarProfile,
    patch: PlanePatch,
    loop_edges: list[list[int]],
    orientation: int,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> int:
    for edges, profile_edges in zip(loop_edges, profile.edges, strict=True):
        for edge, source in zip(edges, profile_edges, strict=True):
            if profile.endpoint_gap_bounds is not None:
                _bind_profile_period_endpoints(
                    builder, edge, source, budget=budget, preparations=preparations
                )
            if (
                isinstance(source.pcurve, CircleCurve)
                and source.start == source.end
                and not source.authored_range
                and builder.edge_endpoint_roots[edge] == (None, None)
            ):
                # ProfileLoop's closed arc authors one mathematical circle
                # turn; its binary64 range is only the numerical realization.
                builder.native_phase(
                    edge,
                    None,
                    None,
                    rational=Fraction(source.first),
                    preparations=preparations,
                    budget=budget,
                )
    loops = [
        [
            builder.coedge(
                edge, 1, profile_edge.pcurve, preparations=preparations, budget=budget
            )
            for edge, profile_edge in zip(edges, profile_edges, strict=True)
        ]
        for edges, profile_edges in zip(loop_edges, profile.edges, strict=True)
    ]
    return builder.face(
        patch,
        _profile_box(profile, preparations=preparations, budget=budget),
        loops,
        "plane",
        None,
        orientation,
    )


def _line2(origin: tuple[float, float], direction: tuple[float, float], /) -> LineCurve:
    """Canonicalize signed zeros that carry no geometric chart distinction."""
    return LineCurve(
        tuple(0.0 if value == 0.0 else value for value in origin),
        tuple(0.0 if value == 0.0 else value for value in direction),
    )


# ------------------------------------------------------------------ extrusion


@checked
def stage_extrusion(
    builder: Builder,
    profile: PlanarProfile,
    vector: np.ndarray,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> None:
    plane = profile.plane
    normal = plane.normal
    height = float(np.linalg.norm(vector))
    direction = vector / height
    if abs(float(direction @ normal)) <= 1.0e-9:
        raise ValueError(
            "brep_extrusion.direction must leave the sketch plane; got an in-plane sweep direction"
        )
    identity = np.eye(3)
    zero = np.zeros(3)
    bottom_edges: list[list[int]] = []
    top_edges: list[list[int]] = []
    side_faces: list[tuple[int, int, int, int, _ProfileEdge, AbstractCurve]] = []
    for loop in profile.edges:
        count = len(loop)
        bottom_vertices = [builder.vertex(plane.point(edge.start)) for edge in loop]
        top_vertices = [builder.vertex(plane.point(edge.start) + vector) for edge in loop]
        verticals = [
            builder.edge(
                LineCurve(plane.point(edge.start), direction),
                0.0,
                height,
                bottom_vertices[index],
                top_vertices[index],
            )
            for index, edge in enumerate(loop)
        ]
        bottom, top = [], []
        for index, edge in enumerate(loop):
            curve = _space_curve(edge, plane)
            following = (index + 1) % count
            bottom.append(
                builder.edge(
                    curve,
                    edge.first,
                    edge.last,
                    bottom_vertices[index],
                    bottom_vertices[following],
                )
            )
            top.append(
                builder.edge(
                    _moved_curve(curve, identity, zero, vector),
                    edge.first,
                    edge.last,
                    top_vertices[index],
                    top_vertices[following],
                )
            )
            side_faces.append(
                (bottom[-1], top[-1], verticals[index], verticals[following], edge, curve)
            )
        bottom_edges.append(bottom)
        top_edges.append(top)
    if profile.endpoint_gap_bounds is not None:
        builder.recompute_endpoint_gaps = True
    sweep_sign = 1 if float(direction @ normal) > 0.0 else -1
    _cap(
        builder,
        profile,
        PlanePatch(plane.origin, plane.x_axis, plane.y_axis),
        bottom_edges,
        -sweep_sign,
        preparations=preparations,
        budget=budget,
    )
    _cap(
        builder,
        profile,
        PlanePatch(np.asarray(plane.origin) + vector, plane.x_axis, plane.y_axis),
        top_edges,
        sweep_sign,
        preparations=preparations,
        budget=budget,
    )
    for bottom, top, left, right, edge, curve in side_faces:
        _stage_extruded_side(
            builder,
            bottom,
            top,
            left,
            right,
            edge,
            curve,
            direction,
            height,
            normal,
            preparations=preparations,
            budget=budget,
        )


def _stage_extruded_side(
    builder: Builder,
    bottom: int,
    top: int,
    left: int,
    right: int,
    edge: _ProfileEdge,
    curve: AbstractCurve,
    direction: np.ndarray,
    height: float,
    normal: np.ndarray,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> None:
    first, last = edge.first, edge.last
    patch: AbstractSurfacePatch
    match curve:
        case LineCurve():
            patch = PlanePatch(curve.origin, curve.direction, direction)
            tag = "plane"
        case CircleCurve() if abs(abs(float(direction @ normal)) - 1.0) <= 1.0e-12:
            patch = CylinderPatch(
                curve.center, curve.first_axis, curve.second_axis, direction, curve.radius
            )
            tag = "cylinder"
        case CircleCurve() | EllipseCurve() | BSplineCurve():
            patch = ExtrusionSurface(curve, direction)
            tag = "extrusion"
        case _:
            raise TypeError(
                f"brep_extrusion.carrier must be a supported extrusion carrier; got {type(curve).__name__}"
            )
    if builder.recompute_endpoint_gaps:
        for index in (bottom, top):
            _bind_profile_period_endpoints(
                builder,
                index,
                edge,
                patch=patch,
                axis=0,
                budget=budget,
                preparations=preparations,
            )
    box = np.asarray(((first, 0.0), (last, height)))
    lower_seam: BRepPCurve = _line2((first, 0.0), (0.0, 1.0))
    upper_seam: BRepPCurve
    if isinstance(curve, CircleCurve) and left == right and not edge.authored_range:
        builder.native_phase(
            bottom,
            patch,
            0,
            rational=Fraction(first),
            preparations=preparations,
            budget=budget,
        )
        builder.native_phase(
            top,
            patch,
            0,
            rational=Fraction(first),
            preparations=preparations,
            budget=budget,
        )
        # The closed source circle constructs two sheets of one vertical edge.
        source_seam = lower_seam
        lower_seam = PeriodicPCurve(source_seam, patch, (0, 0))
        upper_seam = PeriodicPCurve(source_seam, patch, (1, 0))
    else:
        upper_seam = _line2((last, 0.0), (0.0, 1.0))
    loop = [
        builder.coedge(
            bottom,
            1,
            _line2((0.0, 0.0), (1.0, 0.0)),
            preparations=preparations,
            budget=budget,
        ),
        builder.coedge(right, 1, upper_seam, preparations=preparations, budget=budget),
        builder.coedge(
            top,
            -1,
            _line2((0.0, height), (1.0, 0.0)),
            preparations=preparations,
            budget=budget,
        ),
        builder.coedge(left, -1, lower_seam, preparations=preparations, budget=budget),
    ]
    middle = 0.5 * (first + last)
    outward = np.cross(_curve_tangent(curve, middle), normal)
    builder.face(patch, box, [loop], tag, (loop[0], middle, outward))


# ------------------------------------------------------------------ revolution


@dataclass(frozen=True, slots=True)
class _Axis:
    origin: np.ndarray
    direction: np.ndarray
    radial: np.ndarray
    tangential: np.ndarray
    scale: float

    def axial(self, point: np.ndarray, /) -> float:
        return float((point - self.origin) @ self.direction)

    def radius(self, point: np.ndarray, /) -> float:
        relative = point - self.origin
        return float(
            np.linalg.norm(relative - (relative @ self.direction) * self.direction)
        )

    def on_axis(self, point: np.ndarray, /) -> bool:
        return self.radius(point) <= _GEOMETRIC_TOLERANCE * self.scale


type _ProjectedValue = tuple[Fraction, Fraction, Fraction]


def _profile_projection_bounds(
    edge: _ProfileEdge, origin: Point, direction: Point, /
) -> tuple[_ProjectedValue, _ProjectedValue]:
    def projection(point: Point) -> Fraction:
        relative = sub(point, origin)
        return relative[0] * direction[0] + relative[1] * direction[1]

    if isinstance(edge.pcurve, (LineCurve, BSplineCurve)):
        values = [
            projection(point)
            for hull in _source_profile_hulls(edge)
            for point in hull.points
        ]
        return (min(values), Fraction(), Fraction()), (
            max(values),
            Fraction(),
            Fraction(),
        )
    if not isinstance(edge.pcurve, CircleCurve):
        raise TypeError(
            f"PlanarProfile.projection_carrier must be a line, spline or circle; got {type(edge.pcurve).__name__}"
        )
    center, radius_squared = _arc_circle(edge)
    radicand = radius_squared * (direction[0] ** 2 + direction[1] ** 2)
    center_projection = projection(center)
    values = [
        (projection(rational_vector(point)), Fraction(), radicand)
        for point in (edge.start, edge.end)
    ]
    for sign in (-1, 1):
        radial = (center[0] + sign * direction[0], center[1] + sign * direction[1])
        if _arc_contains_radical(edge, (radial, (Fraction(), Fraction())), Fraction()):
            values.append((center_projection, Fraction(sign), radicand))
    low = high = values[0]
    for value in values[1:]:
        if _radical_sign(value[0] - low[0], value[1] - low[1], radicand) < 0:
            low = value
        if _radical_sign(value[0] - high[0], value[1] - high[1], radicand) > 0:
            high = value
    return low, high


def _revolution_axis(
    profile: PlanarProfile,
    origin: tuple[float, float],
    direction: tuple[float, float],
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> _Axis:
    if any(edge.authored_range for loop in profile.edges for edge in loop):
        raise ValueError(
            "brep_revolution.profile must use legacy conic topology; "
            "got an authored angular conic use supported only by planar faces and extrusion"
        )
    plane = profile.plane
    axis_direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(axis_direction))
    if not norm > 0.0:
        raise ValueError(
            "brep_revolution.axis_direction must be nonzero; got zero direction"
        )
    axis_direction = axis_direction / norm
    perpendicular = np.asarray((-axis_direction[1], axis_direction[0]))
    exact_origin = rational_vector(origin)
    exact_perpendicular = (-Fraction(direction[1]), Fraction(direction[0]))
    bounds = [
        (edge, _profile_projection_bounds(edge, exact_origin, exact_perpendicular))
        for loop in profile.edges
        for edge in loop
    ]
    if all(_radical_sign(*low) >= 0 for _, (low, _) in bounds):
        side = 1.0
    elif all(_radical_sign(*high) <= 0 for _, (_, high) in bounds):
        side = -1.0
    else:
        raise ValueError(
            "brep_revolution.profile must lie on one side of its revolution axis; got an axis-crossing source profile"
        )
    for edge, (low, high) in bounds:
        nearest = low if side > 0 else high
        if (
            isinstance(edge.pcurve, BSplineCurve)
            and _radical_sign(*nearest) == 0
            and affine_curve_coefficients(
                edge.pcurve,
                np.zeros((edge.pcurve.ambient_dimension,), dtype=np.float64),
            )
            is None
        ):
            raise ValueError(
                "brep_revolution.axis_contact must have a regularity certificate; got an unresolved spline singularity"
            )
    scale = max(
        1.0,
        float(
            np.max(
                np.abs(_profile_box(profile, preparations=preparations, budget=budget))
            )
        ),
    )
    axis3 = plane.vector(axis_direction)
    radial3 = plane.vector(side * perpendicular)
    return _Axis(plane.point(origin), axis3, radial3, np.cross(axis3, radial3), scale)


@dataclass(frozen=True, slots=True)
class _SweptChart:
    """Analytic chart of a revolved edge: ``v = slope * t + intercept``."""

    patch: AbstractSurfacePatch
    tag: str
    slope: float
    intercept: float
    planar: bool


def _revolved_chart(curve: AbstractCurve, edge: _ProfileEdge, axis: _Axis) -> _SweptChart:
    origin, k, e1, e2 = axis.origin, axis.direction, axis.radial, axis.tangential

    def affine_chart(start: np.ndarray, direction: np.ndarray, /) -> _SweptChart:
        along = float(direction @ k)
        if abs(along) <= 1.0e-12:
            center = origin + axis.axial(start) * k
            return _SweptChart(PlanePatch(center, e1, e2), "plane", 0.0, 0.0, True)
        intercept = axis.axial(start)
        if abs(abs(along) - 1.0) <= 1.0e-12:
            patch = CylinderPatch(origin, e1, e2, k, axis.radius(start))
            return _SweptChart(patch, "cylinder", along, intercept, False)
        slope = float(direction @ e1) / along
        reference = axis.radius(start) - intercept * slope
        patch = ConePatch(origin, e1, e2, k, reference, float(np.arctan(slope)))
        return _SweptChart(patch, "cone", along, intercept, False)

    match curve:
        case LineCurve():
            return affine_chart(np.asarray(curve.origin), np.asarray(curve.direction))
        case CircleCurve():
            center = np.asarray(curve.center)
            first = np.asarray(curve.first_axis)
            second = np.asarray(curve.second_axis)
            # The arc angle maps affinely onto the chart angle measured from
            # ``e1`` toward ``k``: ``v = +-t + alpha`` (sign by frame handedness).
            slope = (
                1.0 if float(np.cross(first, second) @ np.cross(e1, k)) > 0.0 else -1.0
            )
            intercept = atan2(float(first @ k), float(first @ e1))
            middle = slope * 0.5 * (edge.first + edge.last) + intercept
            intercept -= _TWO_PI * round(middle / _TWO_PI)
            radius = float(curve.radius)
            if axis.on_axis(center):
                patch = SpherePatch(center, e1, e2, k, radius)
                return _SweptChart(patch, "sphere", slope, intercept, False)
            ring_center = origin + axis.axial(center) * k
            patch = TorusPatch(ring_center, e1, e2, k, axis.radius(center), radius)
            return _SweptChart(patch, "torus", slope, intercept, False)
        case BSplineCurve():
            coefficients = affine_curve_coefficients(
                curve,
                np.zeros((curve.ambient_dimension,), dtype=np.float64),
            )
            if coefficients is not None:
                return affine_chart(
                    np.asarray(tuple(float(value) for value in coefficients[0])),
                    np.asarray(tuple(float(value) for value in coefficients[1])),
                )
            if axis.on_axis(np.asarray(curve.control_points[0])) or axis.on_axis(
                np.asarray(curve.control_points[-1])
            ):
                raise ValueError(
                    "brep_revolution.poles must have a regularity certificate; got unresolved spline singularity"
                )
            return _SweptChart(
                RevolutionSurface(curve, origin, k), "revolution", 1.0, 0.0, False
            )
        case _:
            raise TypeError(
                f"brep_revolution.carrier must be supported; got {type(curve).__name__}"
            )


class _Revolution:
    """Stage the topology of a profile revolved about an in-plane axis.

    Off-axis profile vertices sweep circle (or arc) edges; on-axis vertices
    stay single vertices and become degenerate pole edges of curved swept
    faces. Every off-axis profile edge sweeps one face; on-axis segments sweep
    nothing (a partial revolution shares them between its two caps). In a
    full revolution curved swept faces use their profile edge as the periodic
    seam, while planar annuli need no seam.
    """

    def __init__(
        self,
        builder: Builder,
        profile: PlanarProfile,
        axis: _Axis,
        angle: float,
        *,
        native_phase: bool,
    ) -> None:
        self.builder = builder
        self.profile = profile
        self.axis = axis
        self.angle = angle
        self.full = native_phase
        self.rotation = _rotation(axis.direction, angle)
        self.start_vertices: dict[tuple[float, float], int] = {}
        self.end_vertices: dict[tuple[float, float], int] = {}
        self.circles: dict[tuple[float, float], int] = {}

    def _on_axis(self, point2: tuple[float, float], /) -> bool:
        return self.axis.on_axis(self.profile.plane.point(point2))

    def vertex(self, point2: tuple[float, float], rotated: bool, /) -> int:
        point = self.profile.plane.point(point2)
        moved = rotated and not self.full and not self.axis.on_axis(point)
        table = self.end_vertices if moved else self.start_vertices
        if point2 not in table:
            position = (
                self.axis.origin + self.rotation @ (point - self.axis.origin)
                if moved
                else point
            )
            table[point2] = self.builder.vertex(position)
        return table[point2]

    def circle(self, point2: tuple[float, float], /) -> int:
        if point2 not in self.circles:
            point = self.profile.plane.point(point2)
            axis = self.axis
            carrier = CircleCurve(
                axis.origin + axis.axial(point) * axis.direction,
                axis.radial,
                axis.tangential,
                axis.radius(point),
            )
            self.circles[point2] = self.builder.edge(
                carrier,
                0.0,
                self.angle,
                self.vertex(point2, False),
                self.vertex(point2, True),
            )
        return self.circles[point2]

    def stage(
        self,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.stage(preparations=preparations, budget=budget)
        plane = self.profile.plane
        start_edges: list[list[int]] = []
        end_edges: list[list[int]] = []
        for loop in self.profile.edges:
            starts, ends = [], []
            for edge in loop:
                first, last = self._stage_edge(
                    edge, preparations=preparations, budget=budget
                )
                starts.append(first)
                ends.append(last)
            start_edges.append(starts)
            end_edges.append(ends)
        if self.full:
            return
        sweep_sign = 1 if float(self.axis.tangential @ plane.normal) > 0.0 else -1
        _cap(
            self.builder,
            self.profile,
            PlanePatch(plane.origin, plane.x_axis, plane.y_axis),
            start_edges,
            -sweep_sign,
            preparations=preparations,
            budget=budget,
        )
        origin = self.axis.origin + self.rotation @ (
            np.asarray(plane.origin) - self.axis.origin
        )
        _cap(
            self.builder,
            self.profile,
            PlanePatch(
                origin,
                self.rotation @ np.asarray(plane.x_axis),
                self.rotation @ np.asarray(plane.y_axis),
            ),
            end_edges,
            sweep_sign,
            preparations=preparations,
            budget=budget,
        )

    def _stage_edge(
        self,
        edge: _ProfileEdge,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[int, int]:
        """Stage one profile segment; returns its edges at angle zero and the end."""
        plane = self.profile.plane
        curve = _space_curve(edge, plane)
        axial = (
            isinstance(curve, LineCurve)
            and self._on_axis(edge.start)
            and self._on_axis(edge.end)
        )
        if axial and self.full:
            return -1, -1
        chart = None if axial else _revolved_chart(curve, edge, self.axis)
        if chart is not None and chart.planar and self.full:
            self._stage_annulus(
                edge, curve, chart, preparations=preparations, budget=budget
            )
            return -1, -1
        if chart is not None and chart.planar:
            # A partial sector's Cartesian end ray would independently round
            # sin(angle) and cos(angle), losing its source circular trim join.
            # Retain the revolved profile and rectangular UV chain.
            chart = _SweptChart(
                RevolutionSurface(curve, self.axis.origin, self.axis.direction),
                chart.tag,
                1.0,
                0.0,
                False,
            )
        first = self.builder.edge(
            curve,
            edge.first,
            edge.last,
            self.vertex(edge.start, False),
            self.vertex(edge.end, False),
        )
        last = first
        if not axial and not self.full:
            last = self.builder.edge(
                _moved_curve(curve, self.rotation, self.axis.origin, np.zeros(3)),
                edge.first,
                edge.last,
                self.vertex(edge.start, True),
                self.vertex(edge.end, True),
            )
        if chart is not None:
            self._stage_curved(
                edge, curve, chart, first, last, preparations=preparations, budget=budget
            )
        return first, last

    def _outward(self, curve: AbstractCurve, parameter: float, /) -> np.ndarray:
        # In the profile plane the solid lies left of the traversal direction.
        return np.cross(_curve_tangent(curve, parameter), self.profile.plane.normal)

    def _side(
        self,
        point2: tuple[float, float],
        pcurve: BRepPCurve,
        forward: bool,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> int:
        """Coedge swept by a profile vertex: its circle or a degenerate pole."""
        sense = 1 if forward else -1
        if self._on_axis(point2):
            pole = self.builder.degenerate(self.vertex(point2, False), 0.0, self.angle)
            return self.builder.coedge(
                pole, sense, pcurve, preparations=preparations, budget=budget
            )
        return self.builder.coedge(
            self.circle(point2), sense, pcurve, preparations=preparations, budget=budget
        )

    def _stage_curved(
        self,
        edge: _ProfileEdge,
        curve: AbstractCurve,
        chart: _SweptChart,
        first: int,
        last: int,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        values = (
            chart.slope * edge.first + chart.intercept,
            chart.slope * edge.last + chart.intercept,
        )
        exact_values = (
            Fraction(chart.slope) * Fraction(edge.first) + Fraction(chart.intercept),
            Fraction(chart.slope) * Fraction(edge.last) + Fraction(chart.intercept),
        )
        increasing = values[0] < values[1]
        low, high = (edge.start, edge.end) if increasing else (edge.end, edge.start)
        sense = 1 if increasing else -1
        lower_u: BRepPCurve = _line2((0.0, chart.intercept), (0.0, chart.slope))
        upper_u: BRepPCurve
        if self.full:
            # A full source revolution constructs both sheets from the same
            # source edge parameterization; sense only changes traversal.
            source_u = lower_u
            lower_u = PeriodicPCurve(source_u, chart.patch, (0, 0))
            upper_u = PeriodicPCurve(source_u, chart.patch, (1, 0))
        else:
            upper_u = _line2((self.angle, chart.intercept), (0.0, chart.slope))

        # The corner coordinate belongs to the source affine profile map,
        # not a separately rounded endpoint constant on the adjacent coedge.
        def radial_pcurve(offset: Fraction) -> BRepPCurve:
            return exact_line_pcurve((Fraction(0), offset), (1.0, 0.0))

        lower_v = radial_pcurve(min(exact_values))
        upper_v: BRepPCurve
        if (
            isinstance(curve, CircleCurve)
            and isinstance(chart.patch, TorusPatch)
            and edge.start == edge.end
        ):
            # Retain a nonzero affine source endpoint, not its rounded sum.
            # Zero has the unwrapped line representation.
            source_offset = Fraction(chart.slope) * Fraction(edge.first) + Fraction(
                chart.intercept
            )
            source_v = radial_pcurve(source_offset)
            if chart.slope < 0.0:
                source_v = PeriodicPCurve(source_v, chart.patch, (0, -1))
            lower_v = PeriodicPCurve(source_v, chart.patch, (0, 0))
            upper_v = PeriodicPCurve(source_v, chart.patch, (0, 1))
        else:
            upper_v = radial_pcurve(max(exact_values))
        loop = [
            self._side(low, lower_v, True, preparations=preparations, budget=budget),
            self.builder.coedge(
                last, sense, upper_u, preparations=preparations, budget=budget
            ),
            self._side(high, upper_v, False, preparations=preparations, budget=budget),
        ]
        reference = self.builder.coedge(
            first, -sense, lower_u, preparations=preparations, budget=budget
        )
        loop.append(reference)
        if self.full:
            for coedge in (loop[0], loop[2]):
                self.builder.native_phase(
                    self.builder.coedge_edges[coedge],
                    chart.patch,
                    0,
                    preparations=preparations,
                    budget=budget,
                )
        if isinstance(curve, CircleCurve) and edge.start == edge.end:
            self.builder.native_phase(
                first,
                chart.patch,
                1,
                rational=Fraction(edge.first),
                preparations=preparations,
                budget=budget,
            )
        middle = 0.5 * (edge.first + edge.last)
        self.builder.face(
            chart.patch,
            np.asarray(((0.0, min(values)), (self.angle, max(values)))),
            [loop],
            chart.tag,
            (reference, middle, self._outward(curve, middle)),
        )

    def _radii(self, edge: _ProfileEdge, /) -> tuple[float, float]:
        plane = self.profile.plane
        return (
            self.axis.radius(plane.point(edge.start)),
            self.axis.radius(plane.point(edge.end)),
        )

    def _plane_box(self, inner: float, outer: float, /) -> np.ndarray:
        if self.full:
            return np.asarray(((-outer, -outer), (outer, outer)))
        quarter = 0.5 * pi
        angles = [0.0, self.angle] + [
            index * quarter for index in range(1, int(self.angle / quarter) + 1)
        ]
        x = [r * cos(a) for r in (inner, outer) for a in angles]
        y = [r * sin(a) for r in (inner, outer) for a in angles]
        return np.asarray(((min(x), min(y)), (max(x), max(y))))

    @staticmethod
    def _circle_pcurve(radius: float, /) -> CircleCurve:
        return CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), radius)

    def _stage_annulus(
        self,
        edge: _ProfileEdge,
        curve: AbstractCurve,
        chart: _SweptChart,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        """Full disk or annulus swept by a segment perpendicular to the axis."""
        radii = self._radii(edge)
        outer_index = 0 if radii[0] > radii[1] else 1
        points = (edge.start, edge.end)
        outer_point, inner_point = points[outer_index], points[1 - outer_index]
        outer = self.builder.coedge(
            self.circle(outer_point),
            1,
            self._circle_pcurve(radii[outer_index]),
            preparations=preparations,
            budget=budget,
        )
        loops = [[outer]]
        inner_radius = radii[1 - outer_index]
        if not self._on_axis(inner_point):
            loops.append(
                [
                    self.builder.coedge(
                        self.circle(inner_point),
                        -1,
                        self._circle_pcurve(inner_radius),
                        preparations=preparations,
                        budget=budget,
                    )
                ]
            )
        middle = 0.5 * (edge.first + edge.last)
        # The annulus normal is constant (+-axis): compare at any circle sample.
        self.builder.face(
            chart.patch,
            self._plane_box(0.0, radii[outer_index]),
            loops,
            chart.tag,
            (outer, 0.0, self._outward(curve, middle)),
        )


# ------------------------------------------------------- derived tessellation


def _admitted_affine_chord_coefficients(
    curve: AbstractCurve, *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    """Admit source coefficient work before proving the affine family."""
    match curve:
        case LineCurve():
            origin, direction = jax.device_get((curve.origin, curve.direction))
            if np.asarray(origin).shape != (2,) or np.asarray(direction).shape != (2,):
                return None
            coefficients = rational_points(np.stack((origin, direction)), budget=budget)
            reserve_fraction_work(coefficients, 0, 4, 1, budget=budget)
        case BSplineCurve(degree=1):
            controls, weights, knots = jax.device_get(
                (curve.control_points, curve.weights, curve.knots)
            )
            if (
                np.asarray(controls).shape != (2, 2)
                or np.asarray(weights).shape != (2,)
                or np.asarray(knots).shape != (4,)
            ):
                return None
            coefficients = rational_points(
                np.concatenate((controls, weights[None], knots.reshape((2, 2)))),
                budget=budget,
            )
            reserve_fraction_work(coefficients, 5, 0, 2, budget=budget)
            if (
                coefficients[2][0] <= 0
                or coefficients[2][0] != coefficients[2][1]
                or coefficients[3][0] != coefficients[3][1]
                or coefficients[4][0] != coefficients[4][1]
                or coefficients[3][1] >= coefficients[4][0]
            ):
                return None
            # The coefficient predicate visits five guards and constructs twelve
            # lifted scalars plus nine span/coefficient arithmetic outputs.
            reserve_fraction_work(coefficients, 14, 21, 7, budget=budget)
        case _:
            return None
    return affine_curve_coefficients(curve, np.zeros((2,), dtype=np.float64))


class NormalizedTrimCurve(AbstractTrimCurve):
    """Normalize an affine chart while preserving carrier/enclosure agreement."""

    curve: AbstractTrimCurve
    lower: Array
    extent: Array
    patch: AbstractSurfacePatch
    start_root: BRepVertexRoot | None
    end_root: BRepVertexRoot | None
    start_endpoint: RootEndpoint | None
    end_endpoint: RootEndpoint | None

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return self.curve.parameter_interval

    @property
    def source_definition(self) -> dict[str, object]:
        return {
            "kind": "normalized-source-trim",
            "curve": encode_geometry(self.curve),
            "lower": np.asarray(self.lower),
            "extent": np.asarray(self.extent),
        }

    @property
    def source_id(self) -> str:
        return canonical_fingerprint(self.source_definition)

    def _source_segment(self) -> CurveTrimSegment:
        match self.curve:
            case CurveTrimSegment() as segment:
                return segment
            case _:
                raise TypeError(
                    "NormalizedTrimCurve.curve must be a ranged trim carrier; "
                    f"got {type(self.curve).__name__}"
                )

    def source_parameters(self, parameters: Array, /) -> Array:
        return self._source_segment().source_parameters(parameters)

    def source_parameter_enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.source_parameter_enclosure(
                    first, last, preparations=preparations, budget=budget
                )
        return self._source_segment().carrier_parameter_enclosure(
            first, last, preparations=preparations, budget=budget
        )

    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.is_c1_on(
                    first, last, preparations=preparations, budget=budget
                )
        return self._source_segment().is_c1_on(
            first, last, preparations=preparations, budget=budget
        )

    @checked
    def affine_chord_evidence(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> AffineTrimChordEvidence | None:
        """Certify normalized affine endpoints with their source-root parameter bounds."""
        match self.curve:
            case (
                CurveTrimSegment(curve=(LineCurve() | BSplineCurve()) as basis) as segment
            ):
                pass
            case _:
                return None
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self, first, last)) as budget:
                return self.affine_chord_evidence(
                    first, last, preparations=preparations, budget=budget
                )
        source_coefficients = _admitted_affine_chord_coefficients(basis, budget=budget)
        if source_coefficients is None:
            return None
        origin, direction = source_coefficients
        lower, extent = jax.device_get((self.lower, self.extent))
        lower_values = np.asarray(lower, dtype=np.float64)
        extent_values = np.asarray(extent, dtype=np.float64)
        if (
            lower_values.shape != (2,)
            or extent_values.shape != (2,)
            or np.any(extent_values <= 0.0)
            or not np.all(np.isfinite(lower_values))
            or not np.all(np.isfinite(extent_values))
        ):
            raise ValueError(
                "NormalizedTrimCurve.normalization must be finite with positive two-coordinate extents; "
                f"got lower={lower_values!r}, extent={extent_values!r}."
            )
        coefficients = rational_points(
            np.stack(
                (
                    lower_values,
                    extent_values,
                    np.asarray((segment.first, segment.last), dtype=np.float64),
                    np.asarray((first, last), dtype=np.float64),
                )
            ),
            budget=budget,
        )
        coefficients = tuple(
            tuple(
                rational_coefficient(value, role="NormalizedTrimCurve.chord_coefficient")
                for value in row
            )
            for row in coefficients
        )
        chart_lower, chart_extent = coefficients[:2]

        def point(parameter: Fraction) -> tuple[Fraction, Fraction]:
            # Two multiply/add/subtract/divide coordinate expressions.
            reserve_fraction_work(
                (origin, direction, chart_lower, chart_extent, (parameter,)),
                8,
                2,
                5,
                budget=budget,
            )
            return (
                rational_coefficient(
                    (origin[0] + direction[0] * parameter - chart_lower[0])
                    / chart_extent[0],
                    role="NormalizedTrimCurve.chord_coordinate",
                ),
                rational_coefficient(
                    (origin[1] + direction[1] * parameter - chart_lower[1])
                    / chart_extent[1],
                    role="NormalizedTrimCurve.chord_coordinate",
                ),
            )

        nominal = []
        boxes = []
        for endpoint, local in enumerate((first, last)):
            fraction = coefficients[3][endpoint]
            # Reversal, range subtraction, scale, and translation.
            reserve_fraction_work((coefficients[2], (fraction,)), 4, 3, 6, budget=budget)
            fraction = 1 - fraction if segment.reversed else fraction
            parameter = coefficients[2][0] + fraction * (
                coefficients[2][1] - coefficients[2][0]
            )
            nominal.append(point(parameter))
            interval = segment.carrier_parameter_enclosure(
                local, local, preparations=preparations, budget=budget
            )
            root_parameters = rational_points(
                np.asarray(((float(interval[0]), float(interval[1])),), dtype=np.float64),
                budget=budget,
            )[0]
            images = (point(root_parameters[0]), point(root_parameters[1]))
            # Four endpoint comparisons and four directed binary64 conversions.
            reserve_fraction_work(images, 8, 4, 2, budget=budget)
            boxes.append(
                (
                    np.asarray(
                        [fraction_down(min(a, b)) for a, b in zip(*images, strict=True)],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [fraction_up(max(a, b)) for a, b in zip(*images, strict=True)],
                        dtype=np.float64,
                    ),
                )
            )
        # Admit the record's rational-versus-binary64 containment validation.
        recorded_bounds = rational_points(
            np.stack((boxes[0][0], boxes[0][1], boxes[1][0], boxes[1][1])),
            budget=budget,
        )
        reserve_fraction_work((*tuple(nominal), *recorded_bounds), 8, 8, 2, budget=budget)
        return AffineTrimChordEvidence(
            nominal_start=nominal[0],
            nominal_end=nominal[1],
            start_lower=boxes[0][0],
            start_upper=boxes[0][1],
            end_lower=boxes[1][0],
            end_upper=boxes[1][1],
            source_id=canonical_fingerprint(carrier_payload(self)),
            endpoint_root_ids=tuple(
                root.root_id
                for root in (segment.first_root, segment.last_root)
                if root is not None
            ),
        )

    def evaluate(
        self,
        parameters: Array,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> Array:
        """Normalize point values while borrowing the supplied source request."""
        return (
            self.curve.evaluate(parameters, budget=budget, preparations=preparations)
            - self.lower
        ) / self.extent

    def host_points(
        self,
        parameters: ArrayLike,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Normalize concrete host points without per-operation device dispatch."""
        points = type(self.curve).host_points(
            self.curve, parameters, preparations=preparations, budget=budget
        )
        return (points - np.asarray(self.lower)) / np.asarray(self.extent)

    def enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.enclosure(
                    first, last, preparations=preparations, budget=budget
                )
        box = type(self.curve).enclosure(
            self.curve, first, last, preparations=preparations, budget=budget
        )
        lower = np.nextafter(box[0] - np.asarray(self.lower), -np.inf)
        upper = np.nextafter(box[1] - np.asarray(self.lower), np.inf)
        return np.stack(
            (
                np.nextafter(lower / np.asarray(self.extent), -np.inf),
                np.nextafter(upper / np.asarray(self.extent), np.inf),
            )
        )

    def enclosures(
        self,
        intervals: npt.ArrayLike,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Normalize each independently enclosed row with the scalar rounding order."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request((self, intervals)) as budget:
                return self.enclosures(
                    intervals, preparations=preparations, budget=budget
                )
        boxes = type(self.curve).enclosures(
            self.curve, intervals, preparations=preparations, budget=budget
        )
        lower = np.nextafter(boxes[:, 0] - np.asarray(self.lower), -np.inf)
        upper = np.nextafter(boxes[:, 1] - np.asarray(self.lower), np.inf)
        return np.stack(
            (
                np.nextafter(lower / np.asarray(self.extent), -np.inf),
                np.nextafter(upper / np.asarray(self.extent), np.inf),
            ),
            axis=1,
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.derivative_bounds(
                    first,
                    last,
                    order=order,
                    preparations=preparations,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                    budget=budget,
                )
        lower, upper = type(self.curve).derivative_bounds(
            self.curve, first,
            last,
            order=order,
            preparations=preparations,
            budget=budget,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        return np.nextafter(lower / np.asarray(self.extent), -np.inf), np.nextafter(
            upper / np.asarray(self.extent), np.inf
        )

    def shares_endpoint(
        self,
        other: AbstractTrimCurve,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    other,
                )
            ) as budget:
                return self.shares_endpoint(
                    other, preparations=preparations, budget=budget
                )
        if (
            not isinstance(other, NormalizedTrimCurve)
            or not np.array_equal(np.asarray(self.lower), np.asarray(other.lower))
            or not np.array_equal(np.asarray(self.extent), np.asarray(other.extent))
        ):
            return False
        if self.curve.shares_endpoint(
            other.curve, preparations=preparations, budget=budget
        ):
            return True
        if (
            self.end_root is None
            or other.start_root is None
            or self.end_root.root_id != other.start_root.root_id
            or self.end_endpoint is None
            or other.start_endpoint is None
        ):
            return False
        if not isinstance(self.curve, CurveTrimSegment) or not isinstance(
            other.curve, CurveTrimSegment
        ):
            return False
        first, second = self.curve.curve, other.curve.curve
        if not isinstance(first, (AbstractCurve, IntersectionPCurve)) or not isinstance(
            second, (AbstractCurve, IntersectionPCurve)
        ):
            return False
        return self.end_root.same_uv_endpoint(
            self.patch,
            first,
            self.end_endpoint,
            second,
            other.start_endpoint,
            preparations=preparations,
            budget=budget,
        )


@checked
def brep_trim_curve(
    geometry: BRepGeometry,
    coedge: int,
    patch: AbstractSurfacePatch,
    /,
    *,
    parameter_bounds: np.ndarray | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    preparations: TrimRequestPreparations | None = None,
) -> AbstractTrimCurve:
    """One source-rooted coedge carrier; no chord-cover allocation or scheduling."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                geometry,
                coedge,
                patch,
            )
        ) as budget:
            return brep_trim_curve(
                geometry,
                coedge,
                patch,
                parameter_bounds=parameter_bounds,
                budget=budget,
                preparations=preparations,
            )
    if not 0 <= coedge < len(geometry.coedge_edges):
        raise ValueError(
            f"brep_trim_curve.coedge must index an oriented edge use; got {coedge}"
        )
    if parameter_bounds is None:
        lower, extent = np.zeros((2,), dtype=np.float64), np.ones((2,), dtype=np.float64)
    else:
        bounds = patch.validate_parameter_box(parameter_bounds, budget=budget)
        lower, extent = bounds[0], bounds[1] - bounds[0]
    edge = geometry.coedge_edges[coedge]
    first, last = np.asarray(geometry.edge_ranges)[edge]
    first_root, last_root = geometry.coedge_endpoint_roots[coedge]
    start, end = (0, 1) if geometry.coedge_senses[coedge] > 0 else (1, 0)
    with original_curve_surface_root_preparation(
        (first_root, last_root), preparations=preparations
    ):
        segment = CurveTrimSegment(
            geometry.pcurves[coedge],
            float(first),
            float(last),
            reversed=geometry.coedge_senses[coedge] < 0,
            first_root=first_root,
            last_root=last_root,
            budget=budget,
            preparations=preparations,
        )
    return NormalizedTrimCurve(
        segment,
        jnp.asarray(lower, dtype=jnp.float64),
        jnp.asarray(extent, dtype=jnp.float64),
        patch,
        geometry.vertex_roots[geometry.edge_vertices[edge][start]],
        geometry.vertex_roots[geometry.edge_vertices[edge][end]],
        geometry.coedge_endpoint_roots[coedge][start],
        geometry.coedge_endpoint_roots[coedge][end],
    )


def _prepared_brep_trim_loop(
    curves: tuple[AbstractTrimCurve, ...],
    *,
    tolerance: float,
    maximum_arcs: int,
    relative_closure_tolerance: float,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> CurveTrimLoop:
    """Reuse only a complete loop bound to its full source data and exact policy.

    The certified cover is a deterministic function of the curves and both
    tolerances; ``maximum_arcs`` only refuses covers that exceed it. The cache
    key therefore omits the cap, and a retained cover is reused only when its
    arc count fits this request's cap (a larger one refuses identically on rebuild).
    """
    budget.ticket.checkpoint()

    def build() -> CurveTrimLoop:
        return CurveTrimLoop(
            curves,
            tolerance=tolerance,
            maximum_arcs=maximum_arcs,
            relative_closure_tolerance=relative_closure_tolerance,
            preparations=preparations,
            budget=budget,
        )

    # Invalid policies keep the constructor's existing validation order.
    if (
        isinstance(tolerance, bool)
        or not np.isfinite(tolerance)
        or tolerance <= 0
        or isinstance(relative_closure_tolerance, bool)
        or not np.isfinite(relative_closure_tolerance)
        or relative_closure_tolerance < 0
        or not isinstance(maximum_arcs, int)
        or isinstance(maximum_arcs, bool)
        or maximum_arcs < len(curves)
    ):
        return build()
    with budget.temporary_scope():
        budget.reserve(0, 256 + 128 * len(curves))
        sources = []
        for curve in curves:
            with budget.temporary_scope():
                budget.reserve(0, 65536)
                admit_carrier_fingerprint(curve, budget)
                sources.append(canonical_fingerprint(carrier_payload(curve)))
        key = canonical_fingerprint(
            {
                "curves": sources,
                "tolerance": tolerance,
                "relative_closure_tolerance": relative_closure_tolerance,
            }
        )
        cached = budget.brep_trim_loop_cache.get(key)
        if (
            cached is not None
            and cached.tolerance == tolerance
            and cached.relative_closure_tolerance == relative_closure_tolerance
            and cached.chords.shape[0] <= maximum_arcs
            and tree_content_equal(curves, cached.curves)
        ):
            return cached
        result = build()
        budget.retain_basis((key, result))
        budget.brep_trim_loop_cache[key] = result
        return result


@checked
def brep_trim_domain(
    geometry: BRepGeometry,
    face: int,
    patch: AbstractSurfacePatch,
    /,
    *,
    tolerance: float,
    maximum_arcs: int = 65536,
    parameter_bounds: np.ndarray | None = None,
    relative_closure_tolerance: float = 1.0e-9,
    budget: CoordinateEnclosureBudget | None = None,
    preparations: TrimRequestPreparations | None = None,
) -> TrimDomain:
    """Build rooted loops in raw UV, or one explicitly normalized chart."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                geometry,
                face,
                patch,
            )
        ) as budget:
            return brep_trim_domain(
                geometry,
                face,
                patch,
                tolerance=tolerance,
                maximum_arcs=maximum_arcs,
                parameter_bounds=parameter_bounds,
                relative_closure_tolerance=relative_closure_tolerance,
                budget=budget,
                preparations=preparations,
            )
    if not 0 <= face < len(geometry.face_loops):
        raise ValueError(f"brep_trim_domain.face must index a face loop; got {face}")
    loops = []
    for loop in geometry.face_loops[face]:
        curves = tuple(
            brep_trim_curve(
                geometry,
                coedge,
                patch,
                parameter_bounds=parameter_bounds,
                budget=budget,
                preparations=preparations,
            )
            for coedge in loop
        )
        loops.append(
            _prepared_brep_trim_loop(
                curves,
                tolerance=tolerance,
                maximum_arcs=maximum_arcs,
                relative_closure_tolerance=relative_closure_tolerance,
                preparations=preparations,
                budget=budget,
            )
        )
    return TrimDomain(loops[0], loops[1:])


# ------------------------------------------------------------------ assembly


_NATIVE_CONSTRUCTION_POLICY_ID = canonical_fingerprint(
    {"kind": "native-brep-construction"}
)


def _check_publication_inputs(
    source_id: str,
    tessellation: BRepTessellationPolicy | None,
    /,
) -> BRepTessellationPolicy:
    # Nominal input contracts are enforced at public construction boundaries.
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(
            f"assemble_brep_model.source_id must be a non-empty string; got {source_id!r}"
        )
    policy = BRepTessellationPolicy() if tessellation is None else tessellation
    return policy


def _solid_edge_balance(
    geometry: BRepGeometry, orientation: np.ndarray, /
) -> tuple[int, ...]:
    """Per-solid closure witnesses must not cancel across region interfaces."""
    return tuple(
        value
        for solid in range(len(geometry.solid_shells))
        for value, degenerate in zip(
            geometry.edge_use_balance(orientation, solid=solid),
            geometry.degenerate_edges,
            strict=True,
        )
        if not degenerate
    )


def _publish(
    builder: Builder,
    geometry: BRepGeometry,
    policy: BRepTessellationPolicy,
    *,
    coordinate_contract: SpatialCoordinateContract,
    source_id: str,
    source_format: str,
    source_digest: str,
    import_policy_id: str,
    converted_surface_count: int,
    curve_surface_tolerance: float | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> BRepModel:
    """Publish geometry and trim charts, realizing a query mesh only on request."""
    face_count = len(builder.faces)
    orientation = np.asarray(
        [face.orientation for face in builder.faces], dtype=np.float64
    )
    patches = tuple(face.patch for face in builder.faces)
    bounds = np.asarray([face.box for face in builder.faces], dtype=np.float64).reshape(
        (-1, 2, 2)
    )
    tags = tuple(face.tag for face in builder.faces)
    physical_scale = brep_physical_scale(
        geometry, preparations=preparations, budget=budget
    )
    correspondence_tolerance = (
        1.0e-8 * physical_scale
        if curve_surface_tolerance is None
        else curve_surface_tolerance
    )
    if (
        isinstance(correspondence_tolerance, bool)
        or not np.isfinite(correspondence_tolerance)
        or correspondence_tolerance < 0.0
    ):
        raise ValueError(
            f"assemble_brep_model.curve_surface_tolerance must be a finite nonnegative physical length; got {correspondence_tolerance!r}"
        )

    source_revision = canonical_fingerprint(
        {
            "kind": "brep-source-revision",
            "source_digest": source_digest,
            "spatial_id": coordinate_contract.spatial_id,
        }
    )
    # The scheduler imports meshing domains, which consume these constructors.
    from . import _tessellation as tessellation_owner

    staged = tessellation_owner.tessellate_brep(
        geometry,
        patches,
        orientation,
        policy,
        source_id=source_id,
        source_revision=source_revision,
    )
    triangles = staged.triangles
    topology = geometry.topology()
    report = BRepImportReport(
        source_id=source_id,
        source_digest=source_digest,
        source_format=source_format,
        coordinate_contract=coordinate_contract,
        import_policy_id=import_policy_id,
        num_solids=topology.num_solids,
        num_faces=face_count,
        num_edges=topology.num_edges,
        num_vertices=topology.num_vertices,
        num_triangles=triangles.shape[0],
        linear_deflection=policy.linear_deflection,
        angular_deflection=policy.angular_deflection,
        trim_samples_per_edge=policy.trim_samples_per_edge,
        converted_surface_count=converted_surface_count,
        curve_surface_tolerance=correspondence_tolerance,
        curve_surface_scale=physical_scale,
    )
    return BRepModel(
        patches=patches,
        parameter_bounds=bounds,
        orientation=orientation,
        trim_domains=tuple(
            brep_trim_domain(
                geometry,
                index,
                face.patch,
                tolerance=1.0 / policy.trim_samples_per_edge,
                maximum_arcs=policy.maximum_triangles * 3,
                parameter_bounds=face.box,
                budget=budget,
                preparations=preparations,
            )
            for index, face in enumerate(builder.faces)
        ),
        topology=topology,
        coordinate_contract=coordinate_contract,
        mesh_vertices=staged.vertices,
        mesh_faces=triangles,
        triangle_face_ids=staged.face_ids,
        triangle_parameters=staged.parameters,
        tessellation_deviation_bounds=staged.deviation_bounds,
        tessellation_normal_bounds=staged.normal_bounds,
        mesh_vertex_source_dimensions=staged.vertex_source_dimensions,
        mesh_vertex_source_indices=staged.vertex_source_indices,
        mesh_vertex_parameters=staged.vertex_parameters,
        mesh_chart_restriction_vertices=staged.chart_restriction_vertices,
        mesh_chart_restriction_edges=staged.chart_restriction_edges,
        mesh_chart_restriction_endpoint_parameters=(
            staged.chart_restriction_endpoint_parameters
        ),
        mesh_chart_restriction_parameters=staged.chart_restriction_parameters,
        triangle_occurrence_ids=staged.triangle_occurrence_ids,
        vertex_occurrence_ids=staged.vertex_occurrence_ids,
        physical_tags=tags,
        report=report,
        geometry=geometry,
        budget=budget,
        preparations=preparations,
    )


type _ShellStructure = tuple[
    tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]
]


@checked
def finish(
    builder: Builder,
    *,
    solid: bool,
    coordinate_contract: SpatialCoordinateContract,
    source_id: str,
    tessellation: BRepTessellationPolicy | None,
    occurrences: tuple[BRepOccurrence, ...] | None = None,
    shells: _ShellStructure | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> BRepModel:
    """Publish staged topology; ``shells`` keeps a source's shell/solid structure."""
    policy = _check_publication_inputs(source_id, tessellation)
    face_count = len(builder.faces)
    shell_faces, shell_orientations, solid_shells = (
        (
            ((tuple(range(face_count)),), ((1,) * face_count,), ((0,),))
            if solid
            else ((), (), ())
        )
        if shells is None
        else shells
    )
    edge_ranges = np.asarray(builder.edge_ranges).reshape(-1, 2)
    if builder.recompute_endpoint_gaps:
        rows = _builder_endpoint_gap_rows(builder, budget=budget)
        builder.edge_vertex_gap_bounds = (
            rows if any(bound for row in rows for bound in row) else None
        )
    face_gaps = None
    if builder.edge_vertex_gap_bounds is not None or builder.recompute_endpoint_gaps:
        budget.reserve(len(builder.coedge_edges), 16 * len(builder.coedge_edges))
        owners = [0] * len(builder.coedge_edges)
        for face, staged_face in enumerate(builder.faces):
            for loop in staged_face.loops:
                for coedge in loop:
                    owners[coedge] = face
        face_gaps = _certify_edge_face_gap_bounds(
            tuple(builder.curves),
            tuple(builder.edge_curves),
            edge_ranges,
            tuple(builder.pcurves),
            tuple(builder.coedge_edges),
            tuple(owners),
            tuple(face.patch for face in builder.faces),
            budget=budget,
            exact_unsupported=builder.recompute_endpoint_gaps,
            edge_endpoint_roots=(
                tuple(builder.edge_endpoint_roots)
                if builder.recompute_endpoint_gaps
                else None
            ),
            coedge_endpoint_roots=(
                tuple(builder.coedge_endpoint_roots)
                if builder.recompute_endpoint_gaps
                else None
            ),
        )
    geometry = BRepGeometry(
        vertex_points=np.asarray(builder.vertices).reshape(-1, 3),
        curves=tuple(builder.curves),
        edge_curves=tuple(builder.edge_curves),
        edge_ranges=edge_ranges,
        edge_vertices=tuple(builder.edge_vertices),
        edge_vertex_gap_bounds=builder.edge_vertex_gap_bounds,
        edge_face_gap_bounds=face_gaps,
        pcurves=tuple(builder.pcurves),
        coedge_edges=tuple(builder.coedge_edges),
        coedge_senses=tuple(builder.coedge_senses),
        face_loops=tuple(
            tuple(tuple(loop) for loop in face.loops) for face in builder.faces
        ),
        shell_faces=shell_faces,
        shell_orientations=shell_orientations,
        solid_shells=solid_shells,
        occurrences=occurrences,
        edge_endpoint_roots=tuple(builder.edge_endpoint_roots),
        coedge_endpoint_roots=tuple(builder.coedge_endpoint_roots),
        budget=budget,
        preparations=preparations,
    )
    orientation = np.asarray(
        [face.orientation for face in builder.faces], dtype=np.float64
    )
    if any(_solid_edge_balance(geometry, orientation)):
        raise RuntimeError(
            "finish.shell must be consistently oriented; got a nonzero solid edge-use balance"
        )
    digest = canonical_fingerprint(
        {
            "kind": "native-brep-source",
            "geometry": geometry.geometry_id,
            "patches": [carrier_payload(face.patch) for face in builder.faces],
            "parameter_bounds": np.stack([face.box for face in builder.faces]),
            "orientation": orientation,
            "physical_tags": [face.tag for face in builder.faces],
        }
    )
    return _publish(
        builder,
        geometry,
        policy,
        coordinate_contract=coordinate_contract,
        source_id=source_id,
        source_format="native",
        source_digest=digest,
        import_policy_id=_NATIVE_CONSTRUCTION_POLICY_ID,
        converted_surface_count=0,
        preparations=preparations,
        budget=budget,
    )


@checked
def assemble_brep_model(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    parameter_bounds: ConvertibleToArray,
    orientation: ConvertibleToArray,
    physical_tags: tuple[str, ...],
    /,
    *,
    coordinate_contract: SpatialCoordinateContract,
    source_id: str,
    source_format: str,
    source_digest: str,
    import_policy_id: str,
    converted_surface_count: int = 0,
    tessellation: BRepTessellationPolicy | None = None,
    curve_surface_tolerance: float | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepModel:
    """Publish decoded source geometry with an optional derived tessellation.

    ``patches``, ``parameter_bounds`` (one ``(2, 2)`` box per face),
    ``orientation`` (per face, the sign of the parametric normal against its
    shell's outward side) and ``physical_tags`` align with
    ``geometry.face_loops``. Shells, solids and occurrences come from
    ``geometry``. Every non-degenerate edge of a solid's shells must be used
    once in each direction; an inconsistently oriented or open solid is
    refused rather than repaired.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                geometry,
                patches,
                parameter_bounds,
                orientation,
                physical_tags,
            )
        ) as budget:
            return assemble_brep_model(
                geometry,
                patches,
                parameter_bounds,
                orientation,
                physical_tags,
                coordinate_contract=coordinate_contract,
                source_id=source_id,
                source_format=source_format,
                source_digest=source_digest,
                import_policy_id=import_policy_id,
                converted_surface_count=converted_surface_count,
                tessellation=tessellation,
                curve_surface_tolerance=curve_surface_tolerance,
                preparations=preparations,
                budget=budget,
            )
    policy = _check_publication_inputs(source_id, tessellation)
    face_count = len(geometry.face_loops)
    bounds = np.asarray(parameter_bounds, dtype=np.float64)
    signs = np.asarray(orientation, dtype=np.float64).reshape(-1)
    if (
        len(patches) != face_count
        or bounds.shape != (face_count, 2, 2)
        or signs.shape != (face_count,)
        or len(physical_tags) != face_count
    ):
        raise ValueError(
            f"assemble_brep_model.faces must align patches, parameter_bounds, orientation and physical_tags; got {len(patches)} patches, bounds {bounds.shape}, signs {signs.shape}, {len(physical_tags)} tags for {face_count} faces"
        )
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError(
            "assemble_brep_model.orientation must contain -1 or 1; got invalid signs"
        )
    if any(not isinstance(tag, str) or not tag for tag in physical_tags):
        raise ValueError(
            "assemble_brep_model.physical_tags must be non-empty strings; got an invalid tag"
        )
    if any(_solid_edge_balance(geometry, signs)):
        raise ValueError(
            "assemble_brep_model.shell must have each edge used once in each direction; got an open or inconsistently oriented solid shell"
        )
    builder = Builder(
        vertices=list(np.asarray(geometry.vertex_points)),
        curves=list(geometry.curves),
        edge_curves=list(geometry.edge_curves),
        edge_ranges=[
            (float(first), float(last))
            for first, last in np.asarray(geometry.edge_ranges)
        ],
        edge_vertices=list(geometry.edge_vertices),
        pcurves=list(geometry.pcurves),
        coedge_edges=list(geometry.coedge_edges),
        coedge_senses=list(geometry.coedge_senses),
        faces=[
            _Face(
                patches[index],
                bounds[index, :, :],
                [list(loop) for loop in geometry.face_loops[index]],
                int(signs[index]),
                physical_tags[index],
            )
            for index in range(face_count)
        ],
    )
    return _publish(
        builder,
        geometry,
        policy,
        coordinate_contract=coordinate_contract,
        source_id=source_id,
        source_format=source_format,
        source_digest=source_digest,
        import_policy_id=import_policy_id,
        converted_surface_count=converted_surface_count,
        curve_surface_tolerance=curve_surface_tolerance,
        preparations=preparations,
        budget=budget,
    )


# ------------------------------------------------------------------ public API


@checked
def brep_planar_face(
    profile: PlanarProfile,
    /,
    *,
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-planar-face",
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepModel:
    """Return a trimmed planar face bounded by the profile loops."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request((profile,)) as budget:
            return brep_planar_face(
                profile,
                coordinate_contract=coordinate_contract,
                tessellation=tessellation,
                source_id=source_id,
                preparations=preparations,
                budget=budget,
            )
    profile_ = profile
    builder = Builder()
    plane = profile_.plane
    loop_edges = []
    for loop in profile_.edges:
        vertices = [builder.vertex(plane.point(edge.start)) for edge in loop]
        loop_edges.append(
            [
                builder.edge(
                    _space_curve(edge, plane),
                    edge.first,
                    edge.last,
                    vertices[index],
                    vertices[(index + 1) % len(loop)],
                )
                for index, edge in enumerate(loop)
            ]
        )
    if profile_.endpoint_gap_bounds is not None:
        builder.recompute_endpoint_gaps = True
    _cap(
        builder,
        profile_,
        PlanePatch(plane.origin, plane.x_axis, plane.y_axis),
        loop_edges,
        1,
        preparations=preparations,
        budget=budget,
    )
    return finish(
        builder,
        solid=False,
        coordinate_contract=coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        preparations=preparations,
        budget=budget,
    )


@checked
def brep_extrusion(
    profile: PlanarProfile,
    vector: ConvertibleToArray,
    /,
    *,
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-extrusion",
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepModel:
    """Return the solid swept by translating a planar profile along ``vector``.

    Line segments sweep planes, arcs sweep cylinders (perpendicular sweeps) or
    generating extrusion surfaces (oblique sweeps); caps are the profile and its
    translate. ``vector`` must leave the sketch plane.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                profile,
                vector,
            )
        ) as budget:
            return brep_extrusion(
                profile,
                vector,
                coordinate_contract=coordinate_contract,
                tessellation=tessellation,
                source_id=source_id,
                preparations=preparations,
                budget=budget,
            )
    profile_ = profile
    vector_ = _point3(vector, "vector")
    if not np.linalg.norm(vector_) > 0.0:
        raise ValueError("brep_extrusion.vector must be nonzero; got zero vector")
    builder = Builder()
    stage_extrusion(builder, profile_, vector_, preparations=preparations, budget=budget)
    return finish(
        builder,
        solid=True,
        coordinate_contract=coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        preparations=preparations,
        budget=budget,
    )


@checked
def brep_revolution(
    profile: PlanarProfile,
    axis_origin: ConvertibleToArray,
    axis_direction: ConvertibleToArray,
    /,
    *,
    angle: float | None = None,
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-revolution",
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepModel:
    """Return the solid swept by revolving a profile about an in-plane axis.

    ``axis_origin`` and ``axis_direction`` are sketch coordinates; the profile
    must lie in one closed half-plane of the axis. The positive rotation sense
    is right-handed about the axis. ``angle=None`` authors one mathematical
    native turn with identified endpoints; explicit numeric angles in
    ``(0, 2 pi]`` retain their literal binary phase and distinct capped ends.
    Swept edges are recognized as plane, cylinder, cone, sphere or torus faces.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                profile,
                axis_origin,
                axis_direction,
            )
        ) as budget:
            return brep_revolution(
                profile,
                axis_origin,
                axis_direction,
                angle=angle,
                coordinate_contract=coordinate_contract,
                tessellation=tessellation,
                source_id=source_id,
                preparations=preparations,
                budget=budget,
            )
    profile_ = profile
    angle_ = _TWO_PI if angle is None else positive_finite_float(angle, "angle")
    if angle_ > _TWO_PI:
        raise ValueError(f"brep_revolution.angle must lie in (0, 2 pi]; got {angle_}")
    axis = _revolution_axis(
        profile_,
        _point2(axis_origin, "axis_origin"),
        _point2(axis_direction, "axis_direction"),
        preparations=preparations,
        budget=budget,
    )
    builder = Builder()
    _Revolution(builder, profile_, axis, angle_, native_phase=angle is None).stage(
        preparations=preparations, budget=budget
    )
    return finish(
        builder,
        solid=True,
        coordinate_contract=coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        preparations=preparations,
        budget=budget,
    )


# ------------------------------------------------------------------ offset


type _LineMap = tuple[
    tuple[Fraction, Fraction], tuple[Fraction, Fraction], tuple[int, int]
]


_CROSS_FACE_OFFSET = (
    "brep_offset.edges must have a G1 certificate or blend faces between distinct "
    "source faces; got sharp or cross-face edges"
)


def _line_pcurve_map(pcurve: AbstractCurve | AbstractTrimCurve, /) -> _LineMap | None:
    """Return ``(origin, direction, sheet shifts)`` of an affine/periodic line p-curve.

    Recognition is by the rational parameter map, not the curve class: a
    polynomial B-spline is the affine map ``origin + direction t`` iff every
    control equals that map at its Greville abscissa (linear precision and
    uniqueness of B-spline coefficients).
    """
    match pcurve:
        case LineCurve():
            origin = tuple(Fraction(float(value)) for value in np.asarray(pcurve.origin))
            direction = tuple(
                Fraction(float(value)) for value in np.asarray(pcurve.direction)
            )
            return (origin[0], origin[1]), (direction[0], direction[1]), (0, 0)
        case BSplineCurve() if pcurve.ambient_dimension == 2:
            weights = {Fraction(float(value)) for value in np.asarray(pcurve.weights)}
            if len(weights) != 1:
                return None
            knots = [Fraction(float(value)) for value in np.asarray(pcurve.knots)]
            controls = [
                (Fraction(float(row[0])), Fraction(float(row[1])))
                for row in np.asarray(pcurve.control_points)
            ]
            degree = pcurve.degree
            greville = [
                sum(knots[index + 1 : index + degree + 1], Fraction()) / degree
                for index in range(len(controls))
            ]
            span = greville[-1] - greville[0]
            if span == 0:
                return None
            slope = tuple(
                (controls[-1][axis] - controls[0][axis]) / span for axis in (0, 1)
            )
            intercept = tuple(
                controls[0][axis] - slope[axis] * greville[0] for axis in (0, 1)
            )
            if any(
                control[axis] != intercept[axis] + slope[axis] * abscissa
                for control, abscissa in zip(controls, greville, strict=True)
                for axis in (0, 1)
            ):
                return None
            return (intercept[0], intercept[1]), (slope[0], slope[1]), (0, 0)
        case PeriodicPCurve():
            inner = _line_pcurve_map(pcurve.source_curve)
            if inner is None:
                return None
            shifts = pcurve.period_shifts
            return inner[0], inner[1], (inner[2][0] + shifts[0], inner[2][1] + shifts[1])
        case AffinePCurve():
            inner = _line_pcurve_map(pcurve.curve)
            if inner is None or inner[2] != (0, 0):
                return None
            (a, b), (c, d) = pcurve.matrix
            (x, y), (dx, dy) = inner[0], inner[1]
            return (
                (a * x + b * y + pcurve.offset[0], c * x + d * y + pcurve.offset[1]),
                (a * dx + b * dy, c * dx + d * dy),
                (0, 0),
            )
        case _:
            return None


class _OffsetIsoline(StrictModule):
    axis: int
    value: Fraction
    shifts: tuple[int, int]


def _offset_isoline(pcurve: BRepPCurve, /) -> _OffsetIsoline:
    """Return the fixed axis, rational source value and sheet shifts of an isoline."""
    mapped = _line_pcurve_map(pcurve) if isinstance(pcurve, AbstractCurve) else None
    if mapped is not None:
        origin, direction, shifts = mapped
        for axis in (0, 1):
            free = 1 - axis
            if direction[axis] == 0 and direction[free] == 1 and origin[free] == 0:
                return _OffsetIsoline(axis, origin[axis], shifts)
    raise ValueError(
        "brep_offset.coedge must be an identity-parameter isoline of its source chart; got a general trim requiring a curve-on-surface offset carrier"
    )


def _rebound_pcurve(pcurve: BRepPCurve, patch: AbstractSurfacePatch, /) -> BRepPCurve:
    """The same p-curve operation tree with its period proofs on ``patch``."""
    match pcurve:
        case PeriodicPCurve():
            source = pcurve.source_curve
            return PeriodicPCurve(
                _rebound_pcurve(source, patch)
                if isinstance(source, AbstractCurve)
                else source,
                patch,
                pcurve.period_shifts,
            )
        case AffinePCurve():
            source = pcurve.curve
            return AffinePCurve(
                _rebound_pcurve(source, patch)
                if isinstance(source, AbstractCurve)
                else source,
                pcurve.matrix,
                pcurve.offset,
            )
        case _:
            return pcurve


def _offset_face_gluing(
    geometry: BRepGeometry, face: int, patch: AbstractSurfacePatch, box: np.ndarray, /
) -> tuple[tuple[bool, bool], bool]:
    """Glued chart axes of one closed rectangular source face, and its poles.

    Every side must be either half of a self-glued smooth seam of this face
    or a degenerate pole. Sharp or cross-face edges would need new blend
    faces (pipe/rolling-ball offsets) and are refused, never bridged.
    """
    loops = geometry.face_loops[face]
    if len(loops) != 1:
        raise ValueError(
            "brep_offset.chart must be a closed rectangle without holes; got multiple source loops"
        )
    sides: dict[tuple[int, int], set[int]] = {}
    poles: set[tuple[int, int]] = set()
    for coedge in loops[0]:
        isoline = _offset_isoline(geometry.pcurves[coedge])
        axis = isoline.axis
        chart = float(isoline.value) + isoline.shifts[axis] * _TWO_PI
        if chart not in (box[0, axis], box[1, axis]):
            raise ValueError(
                "brep_offset.edge must be a side of its full source chart rectangle; got an interior isoline"
            )
        side = (axis, 0 if chart == box[0, axis] else 1)
        edge = geometry.coedge_edges[coedge]
        if geometry.edge_curves[edge] < 0:
            poles.add(side)
        else:
            sides.setdefault(side, set()).add(edge)
    glued = []
    for axis in (0, 1):
        low, high = sides.get((axis, 0), set()), sides.get((axis, 1), set())
        ends = {(axis, 0), (axis, 1)}
        if low and low == high and not ends & poles:
            if not closure_tangent_continuous(
                patch, axis, float(box[0, axis]), float(box[1, axis])
            ):
                raise ValueError(
                    "brep_offset.seam must have a tangent-continuity certificate; got a sharp seam requiring blend faces"
                )
            glued.append(True)
        elif not low and not high and ends <= poles:
            glued.append(False)
        else:
            raise ValueError(_CROSS_FACE_OFFSET)
    return (glued[0], glued[1]), bool(poles)


class _OffsetChart(StrictModule):
    base: AbstractSurfacePatch
    offset: OffsetSurface
    parameter_box: np.ndarray
    glued: tuple[bool, bool]


# Fixed equal pieces of the isotopy parameter per certificate cell level. The
# hull of DX and DF_1 alone widens with |delta| * curvature at every cell size.
_ISOTOPY_PIECES = 4


def _projected_full_rank(lower: np.ndarray, upper: np.ndarray, /) -> bool:
    """Full column rank of every matrix in a ``(3, 2)`` hull, all three rows.

    For the point pseudo-inverse ``C`` of the midpoint, ``|I - C A|_inf < 1``
    for every ``A`` in the hull makes ``C A`` invertible, so ``A`` has full
    column rank; by the mean-value form the map is then injective on any
    convex chart whose derivatives lie in the hull. Coordinate minors lose
    this when both columns lean diagonally against every coordinate pair.
    """

    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return False
    preconditioner = np.linalg.pinv(0.5 * (lower + upper))
    if not np.all(np.isfinite(preconditioner)):
        return False
    product_lower, product_upper = point_times_interval(
        preconditioner[None], lower[None], upper[None]
    )
    residual = np.maximum(
        np.abs(np.nextafter(np.eye(2)[None] - product_upper, -np.inf)),
        np.abs(np.nextafter(np.eye(2)[None] - product_lower, np.inf)),
    )
    row_sum = np.nextafter(residual[..., 0] + residual[..., 1], np.inf)
    return bool(np.max(row_sum) < 1.0)


def _offset_cells_certified(
    charts: tuple[_OffsetChart, ...],
    convex: tuple[bool, ...],
    count: int,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """One uniform ``count x count`` cell level of the offset isotopy certificate.

    Same-chart pairs of a chart with a proved convex-meridian embedding skip
    the box separation; cross-chart pairs always keep it.
    """
    lows, highs, owners, rows, columns = [], [], [], [], []
    grid = np.arange(count)
    for face, chart in enumerate(charts):
        base, offset = chart.base, chart.offset
        box, glued = chart.parameter_box, chart.glued
        knots = [np.linspace(box[0, axis], box[1, axis], count + 1) for axis in (0, 1)]
        index_u, index_v = (
            value.reshape(-1) for value in np.meshgrid(grid, grid, indexing="ij")
        )
        boxes = np.stack(
            (
                np.stack((knots[0][index_u], knots[1][index_v]), axis=-1),
                np.stack((knots[0][index_u + 1], knots[1][index_v + 1]), axis=-1),
            ),
            axis=1,
        )
        # DF_t is affine in t, so on each t piece it is a convex combination of
        # the two endpoint offsets' Jacobians.
        distance = float(offset.distance)
        jacobians = []
        for piece in range(_ISOTOPY_PIECES + 1):
            stage: AbstractSurfacePatch = (
                base
                if piece == 0
                else offset
                if piece == _ISOTOPY_PIECES
                else OffsetSurface(
                    base,
                    distance * piece / _ISOTOPY_PIECES,
                    budget=budget,
                    preparations=preparations,
                )
            )
            low, high = stage.derivative_bounds_batch(
                boxes, order=1, preparations=preparations, budget=budget
            )
            jacobians.append(
                (low.reshape((count, count, 3, 2)), high.reshape((count, count, 3, 2)))
            )
        for i in range(count if glued[0] else count - 1):
            for j in range(count if glued[1] else count - 1):
                block = np.ix_((i, (i + 1) % count), (j, (j + 1) % count))
                for (start_low, start_high), (end_low, end_high) in zip(
                    jacobians[:-1], jacobians[1:], strict=True
                ):
                    hull = (
                        np.minimum(
                            np.min(start_low[block], axis=(0, 1)),
                            np.min(end_low[block], axis=(0, 1)),
                        ),
                        np.maximum(
                            np.max(start_high[block], axis=(0, 1)),
                            np.max(end_high[block], axis=(0, 1)),
                        ),
                    )
                    if not (
                        interval_jacobian_full_rank(*hull) or _projected_full_rank(*hull)
                    ):
                        return False
        source_low, source_high = surface_value_bounds(base, boxes, budget=budget)
        image_low, image_high = surface_value_bounds(offset, boxes, budget=budget)
        lows.append(np.minimum(source_low, image_low))
        highs.append(np.maximum(source_high, image_high))
        owners.append(np.full(boxes.shape[0], face))
        rows.append(index_u)
        columns.append(index_v)
    low, high = np.concatenate(lows), np.concatenate(highs)
    owner, row, column = (
        np.concatenate(owners),
        np.concatenate(rows),
        np.concatenate(columns),
    )
    for cell in range(low.shape[0] - 1):
        others = slice(cell + 1, None)
        separated = np.any(
            (low[others] > high[cell]) | (high[others] < low[cell]), axis=1
        )
        glued = charts[owner[cell].item()].glued
        same = owner[others] == owner[cell]
        if convex[owner[cell].item()]:
            near = same
            if np.any(~separated & ~near):
                return False
            continue
        near = same
        for axis, offsets in (
            (0, row[others] - row[cell]),
            (1, column[others] - column[cell]),
        ):
            distance = np.abs(offsets)
            if glued[axis]:
                distance = np.minimum(distance, count - distance)
            near &= distance <= 1
        if np.any(~separated & ~near):
            return False
    return True


def _certify_offset_embedding(
    charts: tuple[_OffsetChart, ...],
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prove every intermediate normal offset embeds the closed source shell.

    With ``F_t = X + t delta n`` for ``t`` in ``[0, 1]``, ``DF_t`` is affine in
    ``t``: on each of ``_ISOTOPY_PIECES`` equal pieces it is a convex
    combination of the endpoint offsets' Jacobians, and ``F_t(p)`` lies on the
    segment ``[X(p), F_1(p)]``. A full-rank interval hull of each piece's
    endpoint Jacobians over every glued 2x2 cell block therefore proves
    regularity and local injectivity of all ``F_t`` there (mean-value form
    across the glued seam chart), and disjoint hulls of both value enclosures
    prove that non-neighboring cells never collide. A chart with proved
    ``ConvexOffsetEvidence`` embeds each layer globally,
    replacing its same-chart box separation, which cannot separate cells
    nearer than ``|delta|``. The isotopy carries the source embedding and its
    outward orientation to the offset. Unresolved cells refuse after the budget.
    """
    convex = tuple(
        convex_meridian_offset(chart.offset, chart.glued, budget=budget).proved
        for chart in charts
    )
    for count in (8, 16, 32, 64):
        if _offset_cells_certified(
            charts, convex, count, preparations=preparations, budget=budget
        ):
            return
    raise ValueError(
        "brep_offset.embedding must prove regularity and global injectivity of the isotopy within its cell budget; got an unresolved embedding certificate"
    )


def _payload_id(carrier: StrictModule, /) -> str:
    return canonical_fingerprint(carrier_payload(carrier))


def _rebound_root(
    root: RootEndpoint | None,
    carriers: Mapping[str, BRepCurve | BRepPCurve],
    patches: dict[str, OffsetSurface],
    /,
) -> NativePeriodEndpoint | None:
    """Re-author a native-period endpoint on its offset carriers."""
    if root is None:
        return None
    if not isinstance(root, NativePeriodEndpoint):
        raise ValueError(
            "brep_offset.endpoint must be an authored native-period endpoint; got an unsupported endpoint root"
        )
    curve = carriers.get(_payload_id(root.curve))
    patch = None if root.patch is None else patches.get(_payload_id(root.patch))
    if (
        curve is None
        or (root.patch is not None and patch is None)
        or isinstance(curve, IntersectionPCurve)
    ):
        raise ValueError(
            "brep_offset.endpoint must reference an offset source carrier; got a carrier outside the offset source"
        )
    return NativePeriodEndpoint(
        curve, patch, root.axis, rational=root.rational, turns=root.turns
    )


def _offset_topology(
    geometry: BRepGeometry, offsets: tuple[OffsetSurface, ...], /
) -> Builder:
    """Rebind topology onto offset faces with the same charts and isoline edges."""
    coedge_faces = {
        coedge: face
        for face, loops in enumerate(geometry.face_loops)
        for loop in loops
        for coedge in loop
    }
    pcurves = [
        _rebound_pcurve(pcurve, offsets[coedge_faces[coedge]])
        for coedge, pcurve in enumerate(geometry.pcurves)
    ]
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    curves: list[BRepCurve] = []
    edge_curves: list[int] = []
    vertices: dict[int, np.ndarray] = {}
    edge_carriers: list[dict[str, BRepCurve | BRepPCurve]] = []
    for edge, source_index in enumerate(geometry.edge_curves):
        coedges = [c for c, used in enumerate(geometry.coedge_edges) if used == edge]
        face = coedge_faces[coedges[0]]
        mapping: dict[str, BRepCurve | BRepPCurve] = {
            _payload_id(geometry.pcurves[c]): pcurves[c] for c in coedges
        }
        if source_index < 0:
            edge_curves.append(-1)
            uv = np.asarray(pcurves[coedges[0]].evaluate(jnp.asarray(ranges[edge, 0])))
            vertices.setdefault(
                geometry.edge_vertices[edge][0],
                np.asarray(offsets[face].evaluate(jnp.asarray(uv))),
            )
        else:
            isoline = _offset_isoline(geometry.pcurves[coedges[0]])
            value = isoline.value
            if Fraction(float(value)) != value:
                raise ValueError(
                    f"brep_offset.isoline_coordinate must be representable without binary rounding; got {value}"
                )
            curve = SurfaceIsoparametricCurve(offsets[face], isoline.axis, float(value))
            mapping = {_payload_id(geometry.curves[source_index]): curve}
            curves.append(curve)
            edge_curves.append(len(curves) - 1)
            ends = np.asarray(curve.evaluate(jnp.asarray(ranges[edge])))
            for vertex, point in zip(geometry.edge_vertices[edge], ends, strict=True):
                vertices.setdefault(vertex, point)
        edge_carriers.append(mapping)
    if sorted(vertices) != list(range(np.asarray(geometry.vertex_points).shape[0])):
        raise ValueError(
            "brep_offset.vertices must lie on offset edges; got an unbound vertex"
        )
    patches = {_payload_id(patch.base): patch for patch in offsets}
    return Builder(
        vertices=[vertices[index] for index in range(len(vertices))],
        curves=curves,
        edge_curves=edge_curves,
        edge_ranges=[(float(first), float(last)) for first, last in ranges],
        edge_vertices=list(geometry.edge_vertices),
        edge_endpoint_roots=[
            (
                _rebound_root(first, carriers, patches),
                _rebound_root(last, carriers, patches),
            )
            for (first, last), carriers in zip(
                geometry.edge_endpoint_roots, edge_carriers, strict=True
            )
        ],
        pcurves=pcurves,
        coedge_edges=list(geometry.coedge_edges),
        coedge_senses=list(geometry.coedge_senses),
        coedge_endpoint_roots=[
            (
                _rebound_root(first, carriers, patches),
                _rebound_root(last, carriers, patches),
            )
            for (first, last), carriers in zip(
                geometry.coedge_endpoint_roots,
                (
                    {_payload_id(source): pcurve}
                    for source, pcurve in zip(geometry.pcurves, pcurves, strict=True)
                ),
                strict=True,
            )
        ],
    )


@checked
def brep_offset(
    model: BRepModel,
    distance: float,
    /,
    *,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-offset",
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    include_parameter_range: bool = False,
) -> BRepOffsetResult:
    """Return a published signed offset and its operation-intent evidence.

    Planar solids retain their face lattice by rational translated-plane
    intersection, local edge-direction and global separation certificates.
    A vanishing face or collision refuses. Smooth single-chart solids use native
    ``OffsetSurface`` carriers with a certified regular, injective normal isotopy.
    Curved sharp edges requiring blend faces and placed occurrences remain unsupported.

    ``include_parameter_range`` retains the open certified distance interval on
    the result. Refusals retain typed range evidence regardless of this opt-in;
    unsupported interval proofs explicitly identify unknown endpoints.
    """
    try:
        return _brep_offset(
            model,
            distance,
            tessellation=tessellation,
            source_id=source_id,
            preparations=preparations,
            budget=budget,
            include_parameter_range=include_parameter_range,
        )
    except CoordinateEnclosureResourceError:
        raise
    except ValueError as error:
        if "parameter_range" not in error.__dict__:
            attach_parameter_range(error, unknown_parameter_range("distance", str(error)))
        raise


def _brep_offset(
    model: BRepModel,
    distance: float,
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget | None,
    include_parameter_range: bool,
) -> BRepOffsetResult:
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                model,
                distance,
            )
        ) as budget:
            return brep_offset(
                model,
                distance,
                tessellation=tessellation,
                source_id=source_id,
                preparations=preparations,
                budget=budget,
                include_parameter_range=include_parameter_range,
            )
    distance_ = finite_real_scalar(distance, "distance")
    if distance_ == 0.0:
        raise ValueError("brep_offset.distance must be nonzero; got zero distance")
    geometry = model.require_geometry()
    if len(geometry.solid_shells) != 1 or geometry.occurrences != (
        BRepOccurrence(("solid0",), 0),
    ):
        raise ValueError(
            "brep_offset.model must be one unplaced solid; got multiple solids or placed occurrences requiring materialization"
        )
    from ._planar_offset import planar_carrier, planar_offset

    if all(planar_carrier(patch) is not None for patch in model.patches):
        return planar_offset(
            model,
            distance_,
            tessellation=tessellation,
            source_id=source_id,
            preparations=preparations,
            budget=budget,
            include_parameter_range=include_parameter_range,
        )
    if any(root is not None for root in geometry.vertex_roots):
        raise ValueError(
            "brep_offset.vertices must have an isoline offset representation; got rooted source vertices"
        )
    # A nondegenerate edge joining two faces is a sharp or cross-face edge
    # whatever either face's trim representation; it needs blend faces.
    edge_faces: dict[int, set[int]] = {}
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                if geometry.edge_curves[edge] >= 0:
                    edge_faces.setdefault(edge, set()).add(face)
    if any(len(faces) > 1 for faces in edge_faces.values()):
        raise ValueError(_CROSS_FACE_OFFSET)
    orientation = np.asarray(model.orientation, dtype=np.float64)
    bounds = np.asarray(model.parameter_bounds, dtype=np.float64)
    offsets = tuple(
        OffsetSurface(
            patch,
            float(orientation[face]) * distance_,
            budget=budget,
            preparations=preparations,
        )
        for face, patch in enumerate(model.patches)
    )
    gluing = tuple(
        _offset_face_gluing(geometry, face, patch, bounds[face])
        for face, patch in enumerate(model.patches)
    )
    analytic = len(offsets) == 1 and (
        (sphere := sphere_source_equivalence(offsets[0], budget=budget)) is not None
        and sphere[1] > 0
        or offsets[0].analytic_equivalent(budget=budget) is not None
    )
    if not analytic:
        if any(poles for _, poles in gluing):
            raise ValueError(
                "brep_offset.pole_chart must have an analytic equivalence; got a singular pole chart"
            )
        _certify_offset_embedding(
            tuple(
                _OffsetChart(patch, offset, bounds[face], glued)
                for face, (patch, offset, (glued, _)) in enumerate(
                    zip(model.patches, offsets, gluing, strict=True)
                )
            ),
            preparations=preparations,
            budget=budget,
        )
    builder = _offset_topology(geometry, offsets)
    builder.faces = [
        _Face(
            offset,
            bounds[face],
            [list(loop) for loop in geometry.face_loops[face]],
            int(orientation[face]),
            "offset",
        )
        for face, offset in enumerate(offsets)
    ]
    published = finish(
        builder,
        solid=True,
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        shells=(geometry.shell_faces, geometry.shell_orientations, geometry.solid_shells),
        preparations=preparations,
        budget=budget,
    )
    return exact_operation_offset_result(
        published, include_parameter_range=include_parameter_range
    )


def _triple(vector: np.ndarray, /) -> tuple[float, float, float]:
    return float(vector[0]), float(vector[1]), float(vector[2])


def _axial_plane(origin: ConvertibleToArray, axis: ConvertibleToArray, /) -> ProfilePlane:
    """Profile frame with ``x`` radial and ``y`` along ``axis``."""
    direction = _unit3(axis, "axis")
    radial, _ = axis_frame(direction)
    return ProfilePlane(
        _triple(_point3(origin, "origin")), _triple(radial), _triple(direction)
    )


@checked
def brep_box(
    lower: ConvertibleToArray,
    upper: ConvertibleToArray,
    /,
    *,
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-box",
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepModel:
    """Axis-aligned box ``[lower, upper]`` (an extruded rectangle)."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                lower,
                upper,
            )
        ) as budget:
            return brep_box(
                lower,
                upper,
                coordinate_contract=coordinate_contract,
                tessellation=tessellation,
                source_id=source_id,
                preparations=preparations,
                budget=budget,
            )
    lower_, upper_ = _point3(lower, "lower"), _point3(upper, "upper")
    size = upper_ - lower_
    if np.any(size <= 0.0):
        raise ValueError(
            "brep_box.upper must exceed its lower corner; got nonpositive box extent"
        )
    profile = PlanarProfile(
        ProfilePlane(_triple(lower_)),
        ProfileLoop.polygon(
            ((0.0, 0.0), (size[0], 0.0), (size[0], size[1]), (0.0, size[1]))
        ),
    )
    return brep_extrusion(
        profile,
        (0.0, 0.0, size[2]),
        coordinate_contract=coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
        preparations=preparations,
        budget=budget,
    )


@checked
def brep_cylinder(
    radius: float,
    height: float,
    /,
    *,
    base_center: ConvertibleToArray = (0.0, 0.0, 0.0),
    axis: ConvertibleToArray = (0.0, 0.0, 1.0),
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-cylinder",
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepModel:
    """Right circular cylinder: lateral face with one seam and two disk caps."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                radius,
                height,
            )
        ) as budget:
            return brep_cylinder(
                radius,
                height,
                base_center=base_center,
                axis=axis,
                coordinate_contract=coordinate_contract,
                tessellation=tessellation,
                source_id=source_id,
                preparations=preparations,
                budget=budget,
            )
    return brep_cone(
        radius,
        radius,
        height,
        base_center=base_center,
        axis=axis,
        coordinate_contract=coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
        preparations=preparations,
        budget=budget,
    )


@checked
def brep_cone(
    bottom_radius: float,
    top_radius: float,
    height: float,
    /,
    *,
    base_center: ConvertibleToArray = (0.0, 0.0, 0.0),
    axis: ConvertibleToArray = (0.0, 0.0, 1.0),
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-cone",
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepModel:
    """Right circular cone or frustum; a zero radius is an apex (pole)."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                bottom_radius,
                top_radius,
                height,
            )
        ) as budget:
            return brep_cone(
                bottom_radius,
                top_radius,
                height,
                base_center=base_center,
                axis=axis,
                coordinate_contract=coordinate_contract,
                tessellation=tessellation,
                source_id=source_id,
                preparations=preparations,
                budget=budget,
            )
    radii = (
        float(np.asarray(bottom_radius, dtype=np.float64)),
        float(np.asarray(top_radius, dtype=np.float64)),
    )
    height_ = positive_finite_float(height, "height")
    if min(radii) < 0.0 or max(radii) <= 0.0 or not all(np.isfinite(radii)):
        raise ValueError(
            "brep_cone.radii must be finite, nonnegative and not both zero; got invalid cone radii"
        )
    corners = [(0.0, 0.0), (radii[0], 0.0), (radii[1], height_), (0.0, height_)]
    corners = [
        corner for index, corner in enumerate(corners) if corner not in corners[:index]
    ]
    profile = PlanarProfile(_axial_plane(base_center, axis), ProfileLoop.polygon(corners))
    return brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
        preparations=preparations,
        budget=budget,
    )


@checked
def brep_sphere(
    radius: float,
    /,
    *,
    center: ConvertibleToArray = (0.0, 0.0, 0.0),
    axis: ConvertibleToArray = (0.0, 0.0, 1.0),
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-sphere",
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepModel:
    """Sphere: one face with a meridian seam and two pole (degenerate) edges."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request((radius,)) as budget:
            return brep_sphere(
                radius,
                center=center,
                axis=axis,
                coordinate_contract=coordinate_contract,
                tessellation=tessellation,
                source_id=source_id,
                preparations=preparations,
                budget=budget,
            )
    radius_ = positive_finite_float(radius, "radius")
    profile = PlanarProfile(
        _axial_plane(center, axis),
        ProfileLoop(
            ((0.0, -radius_), (0.0, radius_)),
            (ProfileArc((0.0, 0.0)), ProfileLine()),
        ),
    )
    return brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
        preparations=preparations,
        budget=budget,
    )


@checked
def brep_torus(
    major_radius: float,
    minor_radius: float,
    /,
    *,
    center: ConvertibleToArray = (0.0, 0.0, 0.0),
    axis: ConvertibleToArray = (0.0, 0.0, 1.0),
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-torus",
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepModel:
    """Ring torus: one doubly periodic face with two seam circles."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                major_radius,
                minor_radius,
            )
        ) as budget:
            return brep_torus(
                major_radius,
                minor_radius,
                center=center,
                axis=axis,
                coordinate_contract=coordinate_contract,
                tessellation=tessellation,
                source_id=source_id,
                preparations=preparations,
                budget=budget,
            )
    major = positive_finite_float(major_radius, "major_radius")
    minor = positive_finite_float(minor_radius, "minor_radius")
    if minor >= major:
        raise ValueError(
            f"brep_torus.minor_radius must be below major_radius; got {minor} >= {major}"
        )
    profile = PlanarProfile(
        _axial_plane(center, axis), ProfileLoop.circle((major, 0.0), minor)
    )
    return brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
        preparations=preparations,
        budget=budget,
    )


__all__ = [
    "BRepTessellationPolicy",
    "PlanarProfile",
    "ProfileArc",
    "ProfileLine",
    "ProfileLoop",
    "ProfilePlane",
    "brep_box",
    "brep_cone",
    "brep_cylinder",
    "brep_extrusion",
    "brep_offset",
    "brep_planar_face",
    "brep_revolution",
    "brep_sphere",
    "brep_torus",
]
