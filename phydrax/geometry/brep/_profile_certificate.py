#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact hull-subdivision certificates for curved planar B-spline profiles.

A positive-weight rational Bézier piece lies in the convex hull of its
Euclidean control points, and its derivative is a nonnegative combination of
the control differences ``P_i - P_j`` with ``i > j`` (``w^2 c' =
sum_{i>j} w_i w_j (B_i' B_j - B_i B_j') (P_i - P_j)`` with every coefficient
nonnegative on ``[0, 1]``). When those differences span a pointed cone the
piece is regular, injective and its tangent turns by exactly the principal
angle between its end tangents. Every decision below uses those two
properties in exact rational arithmetic; undecided pieces are subdivided by
exact de Casteljau bisection until they are decided or reach binary64
resolution, which is a typed refusal rather than a cap.

Circular arcs keep the profile's existing exact model: the circle through
the rational center with the squared binary64 radius, restricted to the wedge
between the rational rays toward the authored endpoints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import atan2, pi, sqrt

import numpy as np

from ..._interpolation._bspline_extraction import rational_bspline_curve_spans
from ._patches import BSplineCurve


type _Point = tuple[Fraction, Fraction]

_EPSILON = float(np.finfo(np.float64).eps)


def _sub(first: _Point, second: _Point, /) -> _Point:
    return first[0] - second[0], first[1] - second[1]


def _cross(first: _Point, second: _Point, /) -> Fraction:
    return first[0] * second[1] - first[1] * second[0]


def _dot(first: _Point, second: _Point, /) -> Fraction:
    return first[0] * second[0] + first[1] * second[1]


def _exact_point(point: Sequence[float], /) -> _Point:
    return Fraction.from_float(float(point[0])), Fraction.from_float(float(point[1]))


@dataclass(frozen=True, slots=True)
class _LinePiece:
    start: _Point
    end: _Point

    @property
    def points(self) -> tuple[_Point, ...]:
        return self.start, self.end


@dataclass(frozen=True, slots=True)
class _BezierPiece:
    """Exact homogeneous Bernstein rows ``(w x, w y, w)`` of one restriction."""

    rows: tuple[tuple[Fraction, Fraction, Fraction], ...]

    @property
    def points(self) -> tuple[_Point, ...]:
        return tuple((x / w, y / w) for x, y, w in self.rows)

    @property
    def start(self) -> _Point:
        x, y, w = self.rows[0]
        return x / w, y / w

    @property
    def end(self) -> _Point:
        x, y, w = self.rows[-1]
        return x / w, y / w

    def split(self) -> tuple[_BezierPiece, _BezierPiece]:
        rows = self.rows
        left, right = [rows[0]], [rows[-1]]
        while len(rows) > 1:
            rows = tuple(
                (
                    (a[0] + b[0]) / 2,
                    (a[1] + b[1]) / 2,
                    (a[2] + b[2]) / 2,
                )
                for a, b in zip(rows[:-1], rows[1:], strict=True)
            )
            left.append(rows[0])
            right.append(rows[-1])
        return _BezierPiece(tuple(left)), _BezierPiece(tuple(reversed(right)))


@dataclass(frozen=True, slots=True)
class _ArcPiece:
    """Exact circle restricted to the wedge from ray ``start`` to ray ``end``.

    ``start``/``end`` are directions from ``center``; the sweep never exceeds
    one half turn, so the wedge lies in the closed half-plane on the turn side
    of each boundary ray.
    """

    center: _Point
    radius_squared: Fraction
    start_ray: _Point
    end_ray: _Point
    turn: int

    def split(self) -> tuple[_ArcPiece, _ArcPiece]:
        start, end = self.start_ray, self.end_ray
        if self.turn * _cross(start, end) > 0:
            middle = (start[0] + end[0], start[1] + end[1])
        else:
            # A half turn: the quarter-turn ray in the sweep direction.
            middle = (-self.turn * start[1], self.turn * start[0])
        return (
            _ArcPiece(self.center, self.radius_squared, start, middle, self.turn),
            _ArcPiece(self.center, self.radius_squared, middle, end, self.turn),
        )

    def on_boundary_ray(self, point: _Point, ray: _Point, /) -> bool:
        relative = _sub(point, self.center)
        return (
            _dot(relative, relative) == self.radius_squared
            and _cross(relative, ray) == 0
            and _dot(relative, ray) > 0
        )


type _Piece = _LinePiece | _BezierPiece | _ArcPiece


def _resolved(points: Sequence[_Point], /) -> bool:
    """Whether control points are indistinguishable at binary64 resolution."""
    values = np.asarray([[float(x), float(y)] for x, y in points], dtype=np.float64)
    spread = float(np.max(np.ptp(values, axis=0)))
    magnitude = float(np.max(np.abs(values)))
    return spread <= 8.0 * _EPSILON * max(magnitude, np.finfo(np.float64).tiny)


def _arc_resolved(arc: _ArcPiece, /) -> bool:
    start, end = arc.start_ray, arc.end_ray
    scale = sqrt(float(_dot(start, start)) * float(_dot(end, end)))
    return (
        abs(float(_cross(start, end))) <= 8.0 * _EPSILON * scale and _dot(start, end) > 0
    )


def _pointed_cone(vectors: Sequence[_Point], /) -> tuple[_Point, _Point] | None:
    """Return the extreme generators of a cone narrower than a half turn."""
    nonzero = tuple(vector for vector in vectors if vector != (0, 0))
    if not nonzero:
        return None

    def bounds_all(candidate: _Point, sign: int) -> bool:
        # Every generator lies strictly on the ``sign`` side of the candidate
        # ray, or along it in the same direction.
        return all(
            sign * _cross(candidate, other) > 0
            or (_cross(candidate, other) == 0 and _dot(candidate, other) > 0)
            for other in nonzero
        )

    lower = next((vector for vector in nonzero if bounds_all(vector, 1)), None)
    upper = next((vector for vector in nonzero if bounds_all(vector, -1)), None)
    if lower is None or upper is None:
        return None
    return lower, upper


def _in_cone(direction: _Point, lower: _Point, upper: _Point, /) -> bool:
    after = _cross(lower, direction)
    before = _cross(direction, upper)
    return (
        after >= 0
        and before >= 0
        and not (after == 0 and _dot(lower, direction) < 0)
        and not (before == 0 and _dot(direction, upper) < 0)
    )


def _cones_disjoint(
    first: tuple[_Point, _Point], second: tuple[_Point, _Point], /
) -> bool:
    return not (
        _in_cone(second[0], *first)
        or _in_cone(second[1], *first)
        or _in_cone(first[0], *second)
    )


def _hodograph_cone(piece: _BezierPiece, /) -> tuple[_Point, _Point] | None:
    points = piece.points
    if points[1] == points[0] or points[-1] == points[-2]:
        return None
    return _pointed_cone(
        tuple(
            _sub(points[later], points[earlier])
            for earlier in range(len(points))
            for later in range(earlier + 1, len(points))
        )
    )


def _bezier_pieces(curve: BSplineCurve, /) -> tuple[_BezierPiece, ...]:
    spans = rational_bspline_curve_spans(
        np.asarray(curve.control_points).tolist(),
        np.asarray(curve.weights).tolist(),
        np.asarray(curve.knots).tolist(),
        curve.degree,
        exact_control_points=curve.exact_control_points,
        exact_weights=curve.exact_weights,
        exact_knots=curve.exact_knots,
    )
    pieces = []
    for span in spans:
        rows: list[tuple[Fraction, Fraction, Fraction]] = []
        for row in span.homogeneous_bernstein_coefficients:
            x, y, w = row if len(row) == 3 else (None, None, None)
            if not (
                isinstance(x, Fraction)
                and isinstance(y, Fraction)
                and isinstance(w, Fraction)
            ):
                raise TypeError(
                    "Planar profile B-spline certificates require rational "
                    "two-dimensional source coefficients."
                )
            rows.append((x, y, w))
        pieces.append(_BezierPiece(tuple(rows)))
    return tuple(pieces)


def _regular_bezier_pieces(curve: BSplineCurve, /) -> tuple[_BezierPiece, ...]:
    """Subdivide source spans until each has a pointed hodograph cone."""
    pending = list(reversed(_bezier_pieces(curve)))
    regular: list[_BezierPiece] = []
    while pending:
        piece = pending.pop()
        if _hodograph_cone(piece) is not None:
            regular.append(piece)
            continue
        if _resolved(piece.points):
            raise ValueError(
                "Profile B-spline tangents must stay regular; got an unresolved "
                "zero derivative or cusp at binary64 resolution."
            )
        left, right = piece.split()
        pending.extend((right, left))
    return tuple(regular)


def _angle(first: _Point, second: _Point, /) -> float:
    return atan2(float(_cross(first, second)), float(_dot(first, second)))


def bspline_turn_terms(curve: BSplineCurve, /) -> tuple[float, ...]:
    """Signed tangent turns of a profile B-spline from its exact hodograph.

    Each regular piece contributes the principal angle between its end
    tangents, which its pointed hodograph cone makes exact; consecutive pieces
    contribute their junction corner. A reversing junction is a cusp.
    """
    terms: list[float] = []
    previous: _Point | None = None
    for piece in _regular_bezier_pieces(curve):
        points = piece.points
        initial = _sub(points[1], points[0])
        final = _sub(points[-1], points[-2])
        if previous is not None:
            if _cross(previous, initial) == 0 and _dot(previous, initial) < 0:
                raise ValueError(
                    "Profile B-spline tangents must stay regular; got a cusp at "
                    "an internal knot."
                )
            terms.append(_angle(previous, initial))
        terms.append(_angle(initial, final))
        previous = final
    return tuple(terms)


def line_piece(start: Sequence[float], end: Sequence[float], /) -> tuple[_Piece, ...]:
    return (_LinePiece(_exact_point(start), _exact_point(end)),)


def arc_pieces(
    center: Sequence[float],
    radius: float,
    start: Sequence[float],
    end: Sequence[float],
    counterclockwise: bool,
    sweep: float,
    /,
) -> tuple[_Piece, ...]:
    """Exact wedge pieces of one circular arc, each at most a half turn."""
    center_ = _exact_point(center)
    radius_ = Fraction.from_float(float(radius))
    radius_squared = radius_ * radius_
    turn = 1 if counterclockwise else -1
    start_ray = _sub(_exact_point(start), center_)
    end_ray = _sub(_exact_point(end), center_)
    opposite = (-start_ray[0], -start_ray[1])
    if start_ray == end_ray:
        # One full turn is two exact half turns.
        rays = (start_ray, opposite, end_ray)
    elif abs(sweep) > pi:
        # A major arc passes the quarter and half turns from its start ray.
        quarter = (-turn * start_ray[1], turn * start_ray[0])
        rays = (start_ray, quarter, opposite, end_ray)
    else:
        rays = (start_ray, end_ray)
    return tuple(
        _ArcPiece(center_, radius_squared, first, last, turn)
        for first, last in zip(rays[:-1], rays[1:], strict=True)
    )


def bspline_pieces(curve: BSplineCurve, /) -> tuple[_Piece, ...]:
    return _regular_bezier_pieces(curve)


def _point_set(piece: _LinePiece | _BezierPiece, /) -> tuple[_Point, ...]:
    return piece.points


def _hulls_separated(first: tuple[_Point, ...], second: tuple[_Point, ...], /) -> bool:
    normals = {(Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))}
    for points in (first, second):
        for index, point in enumerate(points):
            for other in points[index + 1 :]:
                normal = (point[1] - other[1], other[0] - point[0])
                if normal != (0, 0):
                    normals.add(normal)
    for normal in normals:
        left = tuple(_dot(point, normal) for point in first)
        right = tuple(_dot(point, normal) for point in second)
        if max(left) < min(right) or max(right) < min(left):
            return True
    return False


def _arc_hull_separated(arc: _ArcPiece, points: tuple[_Point, ...], /) -> bool:
    relative = tuple(_sub(point, arc.center) for point in points)
    radius_squared = arc.radius_squared
    if all(_dot(vector, vector) < radius_squared for vector in relative):
        return True
    for axis in relative:
        if axis == (0, 0):
            continue
        scale = _dot(axis, axis)
        if all(
            _dot(vector, axis) > 0
            and _dot(vector, axis) * _dot(vector, axis) > radius_squared * scale
            for vector in relative
        ):
            return True
    if all(arc.turn * _cross(arc.start_ray, vector) < 0 for vector in relative):
        return True
    return all(arc.turn * _cross(vector, arc.end_ray) < 0 for vector in relative)


def _separated(first: _Piece, second: _Piece, /) -> bool:
    if isinstance(first, _ArcPiece):
        if isinstance(second, _ArcPiece):
            return False
        return _arc_hull_separated(first, _point_set(second))
    if isinstance(second, _ArcPiece):
        return _arc_hull_separated(second, _point_set(first))
    return _hulls_separated(_point_set(first), _point_set(second))


def _inner_radius(arc: _ArcPiece, ray: _Point, /) -> Fraction:
    """A rational ``l > 0`` with ``l^2 |ray|^2 <= r^2``."""
    scale = _dot(ray, ray)
    value = Fraction.from_float(
        sqrt(float(arc.radius_squared) / float(scale)) * (1.0 - 2.0**-40)
    )
    while value * value * scale > arc.radius_squared:
        value /= 2
    return value


def _junction_cone(piece: _Piece, vertex: _Point, /) -> tuple[_Point, _Point] | None:
    """Cone at ``vertex`` containing ``piece`` minus ``vertex``."""
    match piece:
        case _LinePiece():
            other = piece.end if piece.start == vertex else piece.start
            return _pointed_cone((_sub(other, vertex),))
        case _BezierPiece():
            points = piece.points
            rest = points[1:] if points[0] == vertex else points[:-1]
            return _pointed_cone(tuple(_sub(point, vertex) for point in rest))
        case _ArcPiece():
            # Chords from an arc endpoint lie between its tangent and the
            # direction toward any point of the opposite radius inside the
            # circle; the sweep of at most a half turn keeps that cone pointed.
            if piece.on_boundary_ray(vertex, piece.start_ray):
                ray, far = piece.start_ray, piece.end_ray
                tangent = (-piece.turn * ray[1], piece.turn * ray[0])
            else:
                ray, far = piece.end_ray, piece.start_ray
                tangent = (piece.turn * ray[1], -piece.turn * ray[0])
            inner = _inner_radius(piece, far)
            anchor = (
                piece.center[0] + inner * far[0],
                piece.center[1] + inner * far[1],
            )
            return _pointed_cone((tangent, _sub(anchor, vertex)))
        case _:
            raise TypeError("Unsupported profile certificate piece.")


def _touches(piece: _Piece, vertex: _Point, /) -> bool:
    if isinstance(piece, _ArcPiece):
        return piece.on_boundary_ray(vertex, piece.start_ray) or piece.on_boundary_ray(
            vertex, piece.end_ray
        )
    return vertex in (piece.start, piece.end)


def _junction_separated(first: _Piece, second: _Piece, vertex: _Point, /) -> bool:
    first_cone = _junction_cone(first, vertex)
    second_cone = _junction_cone(second, vertex)
    return (
        first_cone is not None
        and second_cone is not None
        and _cones_disjoint(first_cone, second_cone)
    )


def _splittable(piece: _Piece, /) -> bool:
    if isinstance(piece, _BezierPiece):
        return not _resolved(piece.points)
    if isinstance(piece, _ArcPiece):
        return not _arc_resolved(piece)
    return False


def _extent(piece: _Piece, /) -> float:
    if isinstance(piece, _ArcPiece):
        start, end = piece.start_ray, piece.end_ray
        cosine = float(_dot(start, end)) / sqrt(
            float(_dot(start, start)) * float(_dot(end, end))
        )
        return sqrt(float(piece.radius_squared)) * sqrt(max(0.0, 2.0 - 2.0 * cosine))
    values = np.asarray(
        [[float(x), float(y)] for x, y in _point_set(piece)], dtype=np.float64
    )
    return float(np.max(np.ptp(values, axis=0)))


def _certify_pair(first: _Piece, second: _Piece, vertices: frozenset[_Point]) -> None:
    pending: list[tuple[_Piece, _Piece, frozenset[_Point]]] = [(first, second, vertices)]
    while pending:
        left, right, shared = pending.pop()
        shared = frozenset(
            vertex
            for vertex in shared
            if _touches(left, vertex) and _touches(right, vertex)
        )
        if not shared and _separated(left, right):
            continue
        if len(shared) == 1 and _junction_separated(left, right, next(iter(shared))):
            continue
        candidates = [
            piece
            for piece in (left, right)
            if isinstance(piece, (_BezierPiece, _ArcPiece)) and _splittable(piece)
        ]
        if not candidates:
            raise ValueError(
                "Profile loops must not touch, cross or overlap; got an unresolved "
                "B-spline source contact at binary64 resolution."
            )
        target = max(candidates, key=_extent)
        other = right if target is left else left
        for child in target.split():
            pending.append((child, other, shared))


def certify_profile_contacts(edges: Sequence[tuple[_Piece, ...]], /) -> None:
    """Prove a closed loop meets itself only at its junctions.

    ``edges`` lists each loop segment's pieces in loop order. Pairs without a
    B-spline piece are left to the caller's exact line/circle decisions.
    """
    pieces = tuple(piece for edge in edges for piece in edge)
    count = len(pieces)
    for first_index in range(count):
        for second_index in range(first_index + 1, count):
            first, second = pieces[first_index], pieces[second_index]
            if not isinstance(first, _BezierPiece) and not isinstance(
                second, _BezierPiece
            ):
                continue
            shared: set[_Point] = set()
            if second_index == first_index + 1:
                shared.add(_end(first))
            if first_index == 0 and second_index == count - 1:
                shared.add(_end(second))
            _certify_pair(first, second, frozenset(shared))


def _end(piece: _Piece, /) -> _Point:
    if isinstance(piece, _ArcPiece):
        ray = piece.end_ray
        return piece.center[0] + ray[0], piece.center[1] + ray[1]
    return piece.end


__all__ = [
    "arc_pieces",
    "bspline_pieces",
    "bspline_turn_terms",
    "certify_profile_contacts",
    "line_piece",
]
