#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Planar clipping with field coordinates and bounded host preparation."""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from typing import cast, final, overload

import numpy as np
from numpy.typing import NDArray

from .._host_storage import declared_integer_storage_bytes
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateLiveStorage,
    reserve_polynomial,
)
from ..discretization._exact_algebraic_geometry import (
    coefficient_bits,
    RadicalElement,
    reserve_algebraic_work,
)
from ..typing import checked


type Point = tuple[Fraction | RadicalElement, ...]
type Polygon = tuple[Point, ...]
type ScalarPoint[Scalar] = tuple[Scalar, ...]
type ScalarPolygon[Scalar] = tuple[ScalarPoint[Scalar], ...]


def _scalar_bits(value: Fraction | RadicalElement, /) -> int:
    if isinstance(value, Fraction):
        return abs(value.numerator).bit_length() + value.denominator.bit_length()
    return 2 * max(1, len(value.terms)) * coefficient_bits(value)


@dataclass(frozen=True, slots=True)
class PolygonIntersectionPreparation:
    """Request-owned rational bounds and supporting lines bound to one polygon."""

    polygon: Polygon
    lower: Point
    upper: Point
    halfplanes: tuple[Point, ...]
    bits: int
    point_membership: tuple[tuple[Point, int, int], ...] = ()


@final
class PolygonSidePreparation:
    """Retain Fraction supporting-line decisions across request-owned polygon pairs."""

    @checked
    def __init__(
        self,
        budget: CoordinateEnclosureBudget,
        storage: CoordinateLiveStorage,
        /,
    ) -> None:
        self._budget = budget
        self._storage = storage
        self._decisions: dict[tuple[Point, Point], bool] = {}
        self._lines: dict[Point, tuple[Point, int]] = {}
        self._points: dict[Point, Point] = {}
        self._bytes_upper = 1536
        storage.set_bound(self._bytes_upper)

    def _line(self, plane: Point, bits: int, /) -> tuple[Point, int]:
        prepared = self._lines.get(plane)
        if prepared is None:
            divisor = abs(plane[0] if plane[0] else plane[1])
            if not divisor:
                raise ValueError(
                    "PolygonSidePreparation.plane must have a nonzero normal; got a zero supporting line"
                )
            normalized = plane
            normalized_bits = bits
            if divisor != 1:
                normalized_bits = 2 * bits + 1
                reserve_polynomial(3, 3, 0, normalized_bits, budget=self._budget)
                normalized = tuple(value / divisor for value in plane)
            prepared = normalized, normalized_bits
            # Positive scaling preserves the closed halfplane and gives shared
            # collinear edges one coefficient key, independent of edge length.
            self._bytes_upper += 4096 + 13 * declared_integer_storage_bytes(
                normalized_bits
            )
            self._storage.set_bound(self._bytes_upper, work=1)
            self._lines[plane] = prepared
        return prepared

    def _point(self, point: Point, bits: int, /) -> Point:
        owned = self._points.get(point)
        if owned is None:
            self._bytes_upper += (
                768
                + 384 * len(point)
                + 2 * len(point) * declared_integer_storage_bytes(bits)
            )
            self._storage.set_bound(self._bytes_upper)
            self._points[point] = point
            return point
        return owned

    def outside(self, plane: Point, point: Point, bits: int, /) -> bool:
        with self._budget.temporary_scope():
            self._budget.reserve(1, 96)
            plane, plane_bits = self._line(plane, bits)
            bits = max(bits, plane_bits)
            point = self._point(point, bits)
            key = plane, point
            result = self._decisions.get(key)
            if result is None:
                result = _normalized_line_outside(plane, point, bits, budget=self._budget)
                # Lines and shared point payloads have one retained owner.
                # Each decision adds only its pair key, Boolean and index slots.
                self._bytes_upper += 1024
                self._storage.set_bound(self._bytes_upper, work=1)
                self._decisions[key] = result
            return result


def _normalized_line_outside(
    plane: Point,
    point: Point,
    bits: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Evaluate a supporting line with its first nonzero normal coefficient ±1."""
    if plane[1] == 0:
        reserve_polynomial(1 if plane[0] == 1 else 2, 1, 0, bits + 1, budget=budget)
        return point[0] >= plane[2] if plane[0] == 1 else -point[0] >= plane[2]
    if plane[0] == 0:
        reserve_polynomial(1 if plane[1] == 1 else 2, 1, 0, bits + 1, budget=budget)
        return point[1] >= plane[2] if plane[1] == 1 else -point[1] >= plane[2]
    reserve_polynomial(5, 4, 3, 4 * bits + 64, budget=budget)
    return plane[0] * point[0] + plane[1] * point[1] >= plane[2]


def prepare_polygon_intersection(
    polygon: Polygon,
    orientation: int,
    *,
    budget: CoordinateEnclosureBudget,
    point_membership: bool = False,
) -> PolygonIntersectionPreparation:
    """Retain clipping coefficients using the caller's earned orientation.

    For an interval that orientation is the sign of its exact signed measure,
    so it already orders the two endpoints without another comparison.
    """
    if len(polygon) == 2 and len(polygon[0]) == 1:
        lower, upper = polygon if orientation > 0 else polygon[::-1]
    else:
        lower, upper = polygon_bounds(polygon, budget=budget)
    halfplanes, bits = _prepared_polygon_halfplanes(polygon, orientation, budget=budget)
    membership = (
        _prepare_point_membership(halfplanes, bits, budget)
        if point_membership and halfplanes else ()
    )
    return PolygonIntersectionPreparation(polygon, lower, upper, halfplanes, bits, membership)


def _prepared_polygon_halfplanes(
    polygon: Polygon, orientation: int, *, budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Point, ...], int]:
    """Construct supporting operands after the caller has prepared exact bounds."""
    halfplanes: tuple[Point, ...] = ()
    if len(polygon[0]) == 2:
        reserve_fraction_work(
            polygon, 8 * len(polygon), 3 * len(polygon), 8, budget=budget
        )
        rows = []
        for a, b in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
            dx, dy = b[0] - a[0], b[1] - a[1]
            rows.append(
                (
                    orientation * dy,
                    -orientation * dx,
                    orientation * (dy * a[0] - dx * a[1]),
                )
            )
        halfplanes = tuple(rows)
    budget.reserve(sum(len(row) for row in (*polygon, *halfplanes)))
    bits = max(_scalar_bits(value) for row in (*polygon, *halfplanes) for value in row)
    return halfplanes, bits


def _prepare_point_membership(
    planes: tuple[Point, ...], bits_upper: int, budget: CoordinateEnclosureBudget, /
) -> tuple[tuple[Point, int, int], ...]:
    """Normalize actual rational supporting lines once, not membership verdicts."""
    budget.reserve(0, 128 + 128 * len(planes))
    result = []
    for plane in planes:
        budget.reserve(len(plane))
        if any(type(value) is not Fraction for value in plane):
            return ()
        reserve_polynomial(4, 1, 0, 2 * bits_upper + 2, budget=budget)
        divisor = abs(plane[0] if plane[0] else plane[1])
        if not divisor:
            return ()
        normalized = plane
        if divisor != 1:
            reserve_polynomial(3, 3, 0, 4 * bits_upper + 4, budget=budget)
            normalized = tuple(value / divisor for value in plane)
        budget.reserve(len(normalized))
        bits = max(_scalar_bits(value) for value in normalized)
        budget.reserve(1)
        if normalized[1] == 0:
            budget.reserve(1)
            kind = 1 if normalized[0] > 0 else -1
        else:
            budget.reserve(1)
            if normalized[0] == 0:
                budget.reserve(1)
                kind = 2 if normalized[1] > 0 else -2
            else:
                budget.reserve(1)
                kind = 3 if normalized[0] > 0 else -3
        result.append((normalized, bits, kind))
    return tuple(result)


def _prepared_separated(
    first: PolygonIntersectionPreparation,
    second: PolygonIntersectionPreparation,
    *,
    budget: CoordinateEnclosureBudget,
    sides: PolygonSidePreparation | None = None,
) -> bool:
    """Prove zero measure, including shared edge and vertex contacts."""
    if sides is not None and sides._budget is not budget:
        raise ValueError(
            "PolygonSidePreparation.budget must match the current request; got a different coefficient ledger"
        )
    budget.reserve(2 * len(first.lower))
    if any(
        first.upper[axis] <= second.lower[axis] or second.upper[axis] <= first.lower[axis]
        for axis in range(len(first.lower))
    ):
        return True
    return _prepared_supports_separated(first, second, budget=budget, sides=sides)


def _prepared_supports_separated(
    first: PolygonIntersectionPreparation,
    second: PolygonIntersectionPreparation,
    *,
    budget: CoordinateEnclosureBudget,
    sides: PolygonSidePreparation | None = None,
) -> bool:
    """Inspect exact supporting lines after the caller has tested exact boxes."""
    if sides is not None and sides._budget is not budget:
        raise ValueError(
            "PolygonSidePreparation.budget must match the current request; got a different coefficient ledger"
        )
    for prepared, other in ((first, second), (second, first)):
        for plane in prepared.halfplanes:
            if sides is None:
                with budget.temporary_scope():
                    budget.admit_work_bound(5 * len(other.polygon))
                    reserve_polynomial(
                        0, 4, 3, 4 * max(prepared.bits, other.bits) + 64, budget=budget
                    )
                    actual = 0
                    separated = True
                    for point in other.polygon:
                        actual += 5
                        if plane[0] * point[0] + plane[1] * point[1] < plane[2]:
                            separated = False
                            break
                    budget.reserve(actual)
            else:
                separated = all(
                    sides.outside(plane, point, max(prepared.bits, other.bits))
                    for point in other.polygon
                )
            # The prepared polygon lies in the closed opposite halfplane.
            # Equality leaves only a supporting line, with exactly zero area.
            if separated:
                return True
    return False


def prepared_point_inside(
    container: PolygonIntersectionPreparation,
    point: Point,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Decide closed membership with the clipping halfplane side values.

    Each halfplane charges and evaluates the side value one Sutherland-Hodgman
    clipping pass evaluates for this point. A polygon whose every vertex is
    inside is returned unchanged by that clipping, so its intersection measure
    is its own measure. The first negative side ends the decision.
    """
    if len(container.lower) != 2:
        raise ValueError(
            "prepared_point_inside.container must prepare a planar polygon; got another dimension"
        )
    if container.point_membership:
        budget.reserve(len(point))
        if all(type(value) is Fraction for value in point):
            budget.reserve(len(point))
            point_bits = max(_scalar_bits(value) for value in point)
            for plane, plane_bits, kind in container.point_membership:
                bits = max(point_bits, plane_bits)
                work = 1 if kind in (1, 2) else 2 if kind in (-1, -2) else 4
                reserve_polynomial(work, 4, 0, 4 * bits + 4, budget=budget)
                if kind in (1, -1):
                    inside = (point[0] if kind == 1 else -point[0]) <= plane[2]
                elif kind in (2, -2):
                    inside = (point[1] if kind == 2 else -point[1]) <= plane[2]
                else:
                    first = plane[2] - point[0] if kind == 3 else plane[2] + point[0]
                    inside = first - plane[1] * point[1] >= 0
                if not inside:
                    return False
            return True
    for plane in container.halfplanes:
        reserve_fraction_work((point, plane), 5, 1, 6, budget=budget)
        if plane[2] - plane[0] * point[0] - plane[1] * point[1] < 0:
            return False
    return True


def reserve_fraction_work(
    points: Polygon,
    visits: int,
    outputs: int,
    operands: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Charge rational term visits and an expression-derived storage bound.

    ``operands`` bounds input occurrences in each expanded rational result,
    including denominators. Multiplying their numerator/denominator bit lengths
    bounds denominator products; the visit count bounds additive carry bits.
    These are allocation bounds, never independent scientific admission limits.
    """

    if budget is None:
        return
    if any(isinstance(value, RadicalElement) for row in points for value in row):
        reserve_algebraic_work(points, visits, outputs, operands, budget=budget)
        return
    budget.reserve(sum(len(point) for point in points))
    bits = max(
        (_scalar_bits(value) for point in points for value in point),
        default=1,
    )
    reserve_polynomial(
        visits, outputs, 0, operands * bits + max(visits, 1).bit_length(), budget=budget
    )


def binary64_fraction_bits(value: float, /) -> int:
    """Return a numerator/denominator bit bound for a finite binary64 value."""
    absolute = abs(float(value))
    if absolute == 0.0:
        return 1
    if absolute < sys.float_info.min:
        return 1075
    _, exponent = math.frexp(absolute)
    return max(53, exponent, 54 - exponent)


@overload
def fraction_points(
    points: NDArray[np.float64] | tuple[tuple[float, ...], ...], /
) -> ScalarPolygon[Fraction]: ...


@overload
def fraction_points(points: np.ndarray, /) -> Polygon: ...


def fraction_points(points: np.ndarray | tuple[tuple[float, ...], ...], /) -> Polygon:
    """Convert finite scalar rows after the caller admits the operation."""
    return tuple(
        tuple(
            value
            if isinstance(value, (Fraction, RadicalElement))
            else Fraction(float(value))
            for value in row
        )
        for row in points
    )


@overload
def rational_points(
    points: NDArray[np.float64], *, budget: CoordinateEnclosureBudget
) -> ScalarPolygon[Fraction]: ...


@overload
def rational_points(
    points: np.ndarray, *, budget: CoordinateEnclosureBudget
) -> Polygon: ...


def rational_points(points: np.ndarray, *, budget: CoordinateEnclosureBudget) -> Polygon:

    budget.reserve(2 * points.size, 256)
    bits = (
        max(
            (
                _scalar_bits(value)
                if isinstance(value, (Fraction, RadicalElement))
                else binary64_fraction_bits(float(value))
            )
            for row in points
            for value in row
        )
        if points.size
        else 1
    )
    reserve_polynomial(0, points.size, 0, bits, budget=budget)
    return fraction_points(points)


@overload
def prepared_plane_key(
    points: ScalarPolygon[Fraction], /
) -> tuple[ScalarPoint[Fraction], tuple[int, ...], int, int] | None: ...


@overload
def prepared_plane_key(
    points: Polygon, /
) -> tuple[Point, tuple[int, ...], int, int] | None: ...


def prepared_plane_key(
    points: Polygon, /
) -> tuple[Point, tuple[int, ...], int, int] | None:
    """Return a rational plane normalized to a unit first nonzero component."""
    first = tuple(b - a for a, b in zip(points[0], points[1], strict=True))
    if len(first) == 2:
        normal = (first[1], -first[0])
    else:
        second = tuple(b - a for a, b in zip(points[0], points[2], strict=True))
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
    pivot = next((index for index, value in enumerate(normal) if value), None)
    if pivot is None:
        return None
    divisor = normal[pivot]
    normalized = tuple(value / divisor for value in normal)
    offset = -sum(
        (
            value * coordinate
            for value, coordinate in zip(normalized, points[0], strict=True)
        ),
        Fraction(0),
    )
    axes = ((1,), (0,))[pivot] if len(first) == 2 else ((1, 2), (2, 0), (0, 1))[pivot]
    orientation = -1 if len(first) == 2 and pivot == 1 else 1
    return (*normalized, offset), axes, orientation, 1 if divisor > 0 else -1


@overload
def plane_key(
    points: ScalarPolygon[Fraction], *, budget: CoordinateEnclosureBudget
) -> tuple[ScalarPoint[Fraction], tuple[int, ...], int, int] | None: ...


@overload
def plane_key(
    points: Polygon, *, budget: CoordinateEnclosureBudget
) -> tuple[Point, tuple[int, ...], int, int] | None: ...


def plane_key(
    points: Polygon, *, budget: CoordinateEnclosureBudget
) -> tuple[Point, tuple[int, ...], int, int] | None:
    dimension = len(points[0])
    if (
        dimension == 3 and len(points) >= 3
        and len(points[1]) == 3 and len(points[2]) == 3
    ):
        # The normal input traversal still admits every actual coefficient,
        # including later polygon vertices; this is not a coplanarity proof.
        budget.reserve(sum(len(point) for point in points))
        if all(type(value) is Fraction for point in points for value in point):
            rational = cast("ScalarPolygon[Fraction]", points)
            bits = max(_scalar_bits(value) for point in rational for value in point)
            for pivot, projection in enumerate(((1, 2), (2, 0), (0, 1))):
                budget.reserve(1)
                if rational[0][pivot] != rational[1][pivot]:
                    continue
                budget.reserve(1)
                if rational[0][pivot] != rational[2][pivot]:
                    continue
                # Four differences, two products, one subtraction. The
                # unreduced determinant integers fit 8 input bit bounds + 3.
                reserve_polynomial(7, 9, 0, 8 * bits + 3, budget=budget)
                a, b = projection
                first_a = rational[1][a] - rational[0][a]
                first_b = rational[1][b] - rational[0][b]
                second_a = rational[2][a] - rational[0][a]
                second_b = rational[2][b] - rational[0][b]
                determinant = first_a * second_b - first_b * second_a
                budget.reserve(1)
                if not determinant:
                    return None
                # Two exact constants and the offset negation, then a fresh
                # determinant sign. No zero cross terms or normalization
                # divisions are evaluated on this branch.
                reserve_polynomial(3, 4, 0, bits, budget=budget)
                zero, one = Fraction(0), Fraction(1)
                normal = ((one, zero, zero), (zero, one, zero), (zero, zero, one))[pivot]
                offset = -rational[0][pivot]
                budget.reserve(1)
                sign = 1 if determinant.numerator > 0 else -1
                return (*normal, offset), projection, 1, sign
            # The coefficient scan above already paid the ordinary Fraction
            # profile; retain the unchanged generic expression circuit.
            reserve_polynomial(21, 13, 0, 51 * bits + 5, budget=budget)
            return prepared_plane_key(points)
    # Differences, cross products, normalization and the plane offset use at
    # most ((normal input occurrences * 2) + 1) * dimension operands.
    reserve_fraction_work(
        points,
        dimension * (2 if dimension == 2 else 5) + 2 * dimension,
        4 * dimension + 1,
        ((2 if dimension == 2 else 8) * 2 + 1) * dimension,
        budget=budget,
    )
    return prepared_plane_key(points)


def prepared_project[Scalar](
    points: ScalarPolygon[Scalar], axes: tuple[int, ...], /
) -> ScalarPolygon[Scalar]:
    """Select Fraction coordinates after the caller admits visits."""
    return tuple(tuple(point[axis] for axis in axes) for point in points)


def project[Scalar: Fraction | RadicalElement](
    points: ScalarPolygon[Scalar],
    axes: tuple[int, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> ScalarPolygon[Scalar]:
    reserve_fraction_work(
        points, len(points) * len(axes), len(points) * len(axes), 1, budget=budget
    )
    return prepared_project(points, axes)


@overload
def turn(
    a: ScalarPoint[Fraction],
    b: ScalarPoint[Fraction],
    p: ScalarPoint[Fraction],
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction: ...


@overload
def turn(
    a: Point, b: Point, p: Point, *, budget: CoordinateEnclosureBudget
) -> Fraction | RadicalElement: ...


def turn(
    a: Point,
    b: Point,
    p: Point,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction | RadicalElement:
    reserve_fraction_work((a, b, p), 6, 1, 8, budget=budget)
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


@overload
def prepared_signed_measure(polygon: ScalarPolygon[Fraction], /) -> Fraction: ...


@overload
def prepared_signed_measure(polygon: Polygon, /) -> Fraction | RadicalElement: ...


def prepared_signed_measure(polygon: Polygon, /) -> Fraction | RadicalElement:
    """Return a rational signed polygon measure after arithmetic admission."""
    if not polygon:
        return Fraction(0)
    if len(polygon[0]) == 1:
        return polygon[-1][0] - polygon[0][0]
    return (
        sum(
            (
                a[0] * b[1] - a[1] * b[0]
                for a, b in zip(polygon, (*polygon[1:], polygon[0]), strict=True)
            ),
            Fraction(0),
        )
        / 2
    )


@overload
def signed_measure(
    polygon: ScalarPolygon[Fraction], *, budget: CoordinateEnclosureBudget
) -> Fraction: ...


@overload
def signed_measure(
    polygon: Polygon, *, budget: CoordinateEnclosureBudget
) -> Fraction | RadicalElement: ...


def signed_measure(
    polygon: Polygon, *, budget: CoordinateEnclosureBudget
) -> Fraction | RadicalElement:
    if not polygon:
        return Fraction(0)
    reserve_fraction_work(polygon, 2 * len(polygon), 1, 4 * len(polygon), budget=budget)
    return prepared_signed_measure(polygon)


def simplex_key[Scalar: Fraction | RadicalElement](
    points: ScalarPolygon[Scalar], /, *, budget: CoordinateEnclosureBudget
) -> ScalarPolygon[Scalar]:
    """Return sorted Fraction vertices with bounded comparison and hashing work."""

    coordinates = sum(len(point) for point in points)
    budget.reserve(coordinates * (len(points).bit_length() + 1))
    return tuple(sorted(points))


def polygon_bounds[Scalar: Fraction | RadicalElement](
    points: ScalarPolygon[Scalar], /, *, budget: CoordinateEnclosureBudget
) -> tuple[ScalarPoint[Scalar], ScalarPoint[Scalar]]:
    """Return Fraction coordinate bounds, charging each scalar comparison once."""
    dimension = len(points[0])

    budget.reserve(4 * dimension * (len(points) - 1))
    return (
        tuple(min(point[axis] for point in points) for axis in range(dimension)),
        tuple(max(point[axis] for point in points) for axis in range(dimension)),
    )


def directed_binary64(value: Fraction | RadicalElement, toward: float, /) -> float:
    try:
        rounded = float(value)
    except OverflowError:
        rounded = math.inf if value > 0 else -math.inf
    return math.nextafter(rounded, toward)


def binary64_polygon_bounds(
    points: Polygon,
    lower_toward: float,
    upper_toward: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    lower, upper = polygon_bounds(points, budget=budget)

    budget.reserve(len(lower) + len(upper))
    return (
        tuple(directed_binary64(value, lower_toward) for value in lower),
        tuple(directed_binary64(value, upper_toward) for value in upper),
    )


@overload
def intersection(
    subject: ScalarPolygon[Fraction],
    target: ScalarPolygon[Fraction],
    *,
    budget: CoordinateEnclosureBudget,
    subject_preparation: PolygonIntersectionPreparation | None = None,
    target_preparation: PolygonIntersectionPreparation | None = None,
) -> Fraction: ...


@overload
def intersection(
    subject: Polygon,
    target: Polygon,
    *,
    budget: CoordinateEnclosureBudget,
    subject_preparation: PolygonIntersectionPreparation | None = None,
    target_preparation: PolygonIntersectionPreparation | None = None,
) -> Fraction | RadicalElement: ...


def intersection(
    subject: Polygon,
    target: Polygon,
    *,
    budget: CoordinateEnclosureBudget,
    subject_preparation: PolygonIntersectionPreparation | None = None,
    target_preparation: PolygonIntersectionPreparation | None = None,
) -> Fraction | RadicalElement:
    """Return the field-valued intersection measure of convex polygons or intervals."""

    if subject_preparation is not None and subject_preparation.polygon is not subject:
        raise ValueError(
            "intersection.subject_preparation must bind the subject polygon; got a different polygon object."
        )
    if target_preparation is not None and target_preparation.polygon is not target:
        raise ValueError(
            "intersection.target_preparation must bind the target polygon; got a different polygon object."
        )
    with nullcontext() if budget is None else budget.temporary_scope():
        if (
            subject_preparation is not None
            and target_preparation is not None
            and _prepared_separated(
                subject_preparation, target_preparation, budget=budget
            )
        ):
            measure = Fraction(0)
        elif (
            subject_preparation is not None
            and target_preparation is not None
            and len(subject_preparation.lower) == 1
            and len(target_preparation.lower) == 1
            and all(
                isinstance(value, Fraction)
                for value in (
                    *subject_preparation.lower, *subject_preparation.upper,
                    *target_preparation.lower, *target_preparation.upper,
                )
            )
        ):
            # The two admitted interval boxes are the intervals themselves.
            # Nonseparation above proves their intersection is nonempty, so
            # reuse its endpoints instead of constructing/reprofiling a polygon.
            bits = max(subject_preparation.bits, target_preparation.bits)
            reserve_polynomial(3, 1, 0, 2 * bits + 2, budget=budget)
            low = max(subject_preparation.lower[0], target_preparation.lower[0])
            high = min(subject_preparation.upper[0], target_preparation.upper[0])
            measure = high - low
        else:
            measure = abs(
                signed_measure(
                    intersection_polygon(
                        subject,
                        target,
                        budget=budget,
                        target_preparation=target_preparation,
                    ),
                    budget=budget,
                )
            )
    # Only the returned scalar outlives this clipping workspace.
    reserve_fraction_work(((measure,),), 0, 1, 1, budget=budget)
    return measure


@overload
def intersection_polygon(
    subject: ScalarPolygon[Fraction],
    target: ScalarPolygon[Fraction],
    *,
    budget: CoordinateEnclosureBudget,
    target_preparation: PolygonIntersectionPreparation | None = None,
) -> ScalarPolygon[Fraction]: ...


@overload
def intersection_polygon(
    subject: Polygon,
    target: Polygon,
    *,
    budget: CoordinateEnclosureBudget,
    target_preparation: PolygonIntersectionPreparation | None = None,
) -> Polygon: ...


def intersection_polygon(
    subject: Polygon,
    target: Polygon,
    *,
    budget: CoordinateEnclosureBudget,
    target_preparation: PolygonIntersectionPreparation | None = None,
) -> Polygon:
    """Return the convex intersection, retaining the subject's oriented loop.

    The caller owns the returned polygon's live storage scope. Scalar measure
    consumers use ``intersection`` to release the clipping workspace promptly.
    """
    if target_preparation is not None and target_preparation.polygon is not target:
        raise ValueError(
            "intersection_polygon.target_preparation must bind the target polygon; got a different polygon object."
        )
    if not subject or not target:
        return ()
    if len(subject[0]) == 1:
        reserve_fraction_work(
            (*subject, *target), 2 * (len(subject) + len(target)), 2, 2, budget=budget
        )
        low = max(min(p[0] for p in subject), min(p[0] for p in target))
        high = min(max(p[0] for p in subject), max(p[0] for p in target))
        return ((low,), (high,)) if high >= low else ()
    polygon = subject
    sign = (
        (1 if signed_measure(target, budget=budget) > 0 else -1)
        if target_preparation is None
        else 0
    )
    for edge, (a, b) in enumerate(zip(target, (*target[1:], target[0]), strict=True)):
        clipped = []
        if not polygon:
            break
        sides: tuple[Fraction | RadicalElement, ...] | None = None
        if target_preparation is not None:
            plane = target_preparation.halfplanes[edge]
            reserve_fraction_work(
                (*polygon, plane), 5 * len(polygon), len(polygon), 6, budget=budget
            )
            sides = tuple(
                plane[2] - plane[0] * point[0] - plane[1] * point[1] for point in polygon
            )
        previous = polygon[-1]
        previous_side = (
            sign * turn(a, b, previous, budget=budget) if sides is None else sides[-1]
        )
        for index, point in enumerate(polygon):
            side = (
                sign * turn(a, b, point, budget=budget) if sides is None else sides[index]
            )
            if (side >= 0) != (previous_side >= 0):
                reserve_fraction_work(
                    ((previous_side, side), previous, point),
                    2 + len(point),
                    len(point),
                    8,
                    budget=budget,
                )
                ratio = previous_side / (previous_side - side)
                clipped.append(
                    tuple(
                        x + ratio * (y - x) for x, y in zip(previous, point, strict=True)
                    )
                )
            if side >= 0:
                clipped.append(point)
            previous, previous_side = point, side
        polygon = tuple(clipped)
    return polygon


def convex_hull[Scalar: Fraction | RadicalElement](
    points: ScalarPolygon[Scalar], *, budget: CoordinateEnclosureBudget
) -> ScalarPolygon[Scalar]:

    budget.reserve(len(points), 128 * len(points))
    points = tuple(sorted(set(points)))
    if not points or len(points[0]) == 1:
        return points[:1] if len(points) < 2 else (points[0], points[-1])
    lower: list[ScalarPoint[Scalar]] = []
    upper: list[ScalarPoint[Scalar]] = []
    for point in points:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point, budget=budget) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point, budget=budget) <= 0:
            upper.pop()
        upper.append(point)
    return tuple((*lower[:-1], *upper[:-1]))
