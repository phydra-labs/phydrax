#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact clipping and Bernstein image containment in planar source unions."""

from __future__ import annotations

import math
import sys
from bisect import bisect_left, bisect_right
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..discretization import _coordinate_enclosure as algebra
from ._mapped_coverage import (
    _children,
    _node_count,
    nonnegative,
    projected_chart_measure,
    projected_jacobian,
    SubdivisionLedger,
)


type Point = tuple[Fraction, ...]
type Polygon = tuple[Point, ...]
type _AffineSourceSimplex = tuple[Point, tuple[int, ...], Polygon, int]


@dataclass(frozen=True, slots=True)
class SourceGroup:
    plane: Point
    axes: tuple[int, ...]
    orientation: int
    regions: tuple[int, int]
    members: tuple[int, ...]
    simplices: tuple[Polygon, ...]
    orientations: tuple[int, ...]
    measures: tuple[Fraction, ...]
    box_lower: np.ndarray
    box_upper: np.ndarray
    simplex_keys: tuple[Polygon, ...]
    simplex_key_owners: tuple[int, ...]


def _reserve_fraction_work(
    points: Polygon,
    visits: int,
    outputs: int,
    operands: int,
    /,
) -> None:
    """Charge rational term visits and an expression-derived storage bound.

    ``operands`` bounds input occurrences in each expanded rational result,
    including denominators. Multiplying their numerator/denominator bit lengths
    bounds denominator products; the visit count bounds additive carry bits.
    These are allocation bounds, never independent scientific admission limits.
    """
    budget = algebra._COORDINATE_BUDGET.get()
    if budget is None:
        return
    budget.reserve(sum(len(point) for point in points))
    bits = max(
        (
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            for point in points
            for value in point
        ),
        default=1,
    )
    algebra._reserve_polynomial(
        visits, outputs, 0, operands * bits + max(visits, 1).bit_length()
    )


def _binary64_fraction_bits(value: float, /) -> int:
    """Exact-ratio bit bound from one finite binary64 exponent."""
    absolute = abs(float(value))
    if absolute == 0.0:
        return 1
    if absolute < sys.float_info.min:
        return 1075
    _, exponent = math.frexp(absolute)
    return max(53, exponent, 54 - exponent)


def _fraction_points(points: np.ndarray, /) -> Polygon:
    """Convert finite scalar rows after the owning caller admits the operation."""
    return tuple(
        tuple(
            value if isinstance(value, Fraction) else Fraction(float(value))
            for value in row
        )
        for row in points
    )


def rational_points(points: np.ndarray) -> Polygon:
    budget = algebra._COORDINATE_BUDGET.get()
    if budget is not None:
        # Read binary64 exponents before constructing Python integers. This
        # retains the exact-ratio storage bound without charging every ordinary
        # coordinate as if it were the smallest subnormal.
        budget.reserve(2 * points.size, 256)
        bits = (
            max(
                (
                    max(abs(value.numerator).bit_length(), value.denominator.bit_length())
                    if isinstance(value, Fraction)
                    else _binary64_fraction_bits(float(value))
                )
                for row in points
                for value in row
            )
            if points.size
            else 1
        )
        algebra._reserve_polynomial(0, points.size, 0, bits)
    return _fraction_points(points)


def _plane_key(points: Polygon, /) -> tuple[Point, tuple[int, ...], int, int] | None:
    """Canonical exact plane after the owning caller admits arithmetic."""
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


def plane_key(points: Polygon) -> tuple[Point, tuple[int, ...], int, int] | None:
    dimension = len(points[0])
    # Differences, cross products, normalization and the plane offset use at
    # most ((normal input occurrences * 2) + 1) * dimension operands.
    _reserve_fraction_work(
        points,
        dimension * (2 if dimension == 2 else 5) + 2 * dimension,
        4 * dimension + 1,
        ((2 if dimension == 2 else 8) * 2 + 1) * dimension,
    )
    return _plane_key(points)


def _project(points: Polygon, axes: tuple[int, ...], /) -> Polygon:
    """Select exact coordinates after the owning caller admits visits."""
    return tuple(tuple(point[axis] for axis in axes) for point in points)


def project(points: Polygon, axes: tuple[int, ...]) -> Polygon:
    _reserve_fraction_work(points, len(points) * len(axes), len(points) * len(axes), 1)
    return _project(points, axes)


def turn(a: Point, b: Point, p: Point) -> Fraction:
    _reserve_fraction_work((a, b, p), 6, 1, 8)
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _signed_measure(polygon: Polygon, /) -> Fraction:
    """Exact polygon measure after the owning caller admits arithmetic."""
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


def signed_measure(polygon: Polygon) -> Fraction:
    if not polygon:
        return Fraction(0)
    _reserve_fraction_work(polygon, 2 * len(polygon), 1, 4 * len(polygon))
    return _signed_measure(polygon)


def _simplex_key(points: Polygon, /) -> Polygon:
    """Canonical exact vertex key with bounded comparison and hashing work."""
    budget = algebra._COORDINATE_BUDGET.get()
    if budget is not None:
        coordinates = sum(len(point) for point in points)
        budget.reserve(coordinates * (len(points).bit_length() + 1))
    return tuple(sorted(points))


def _polygon_bounds(points: Polygon, /) -> tuple[Point, Point]:
    """Exact coordinate bounds with work charged once per scalar comparison."""
    dimension = len(points[0])
    budget = algebra._COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(4 * dimension * (len(points) - 1))
    return (
        tuple(min(point[axis] for point in points) for axis in range(dimension)),
        tuple(max(point[axis] for point in points) for axis in range(dimension)),
    )


def _directed_binary64(value: Fraction, toward: float, /) -> float:
    try:
        rounded = float(value)
    except OverflowError:
        rounded = math.inf if value > 0 else -math.inf
    return math.nextafter(rounded, toward)


def _binary64_polygon_bounds(
    points: Polygon,
    lower_toward: float,
    upper_toward: float,
    /,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    lower, upper = _polygon_bounds(points)
    budget = algebra._COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(len(lower) + len(upper))
    return (
        tuple(_directed_binary64(value, lower_toward) for value in lower),
        tuple(_directed_binary64(value, upper_toward) for value in upper),
    )


def intersection(subject: Polygon, target: Polygon) -> Fraction:
    """Exact intersection measure of convex projected polygons or intervals."""
    budget = algebra._COORDINATE_BUDGET.get()
    with nullcontext() if budget is None else budget.temporary_scope():
        measure = abs(signed_measure(intersection_polygon(subject, target)))
    # Only the returned scalar outlives this clipping workspace.
    _reserve_fraction_work(((measure,),), 0, 1, 1)
    return measure


def intersection_polygon(subject: Polygon, target: Polygon) -> Polygon:
    """Exact convex intersection, retaining the subject's oriented loop.

    The caller owns the returned polygon's live storage scope. Scalar measure
    consumers use ``intersection`` to release the clipping workspace promptly.
    """
    if not subject or not target:
        return ()
    if len(subject[0]) == 1:
        _reserve_fraction_work(
            (*subject, *target), 2 * (len(subject) + len(target)), 2, 2
        )
        low = max(min(p[0] for p in subject), min(p[0] for p in target))
        high = min(max(p[0] for p in subject), max(p[0] for p in target))
        return ((low,), (high,)) if high >= low else ()
    polygon = subject
    sign = 1 if signed_measure(target) > 0 else -1
    for a, b in zip(target, (*target[1:], target[0]), strict=True):
        clipped = []
        if not polygon:
            break
        previous = polygon[-1]
        previous_side = sign * turn(a, b, previous)
        for point in polygon:
            side = sign * turn(a, b, point)
            if (side >= 0) != (previous_side >= 0):
                _reserve_fraction_work(
                    ((previous_side, side), previous, point),
                    2 + len(point),
                    len(point),
                    8,
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


def source_groups(
    vertices: np.ndarray, facets: np.ndarray, regions: np.ndarray, maximum_pairs: int
) -> tuple[tuple[SourceGroup, ...], tuple[tuple[int, int], ...], int, bool]:
    from ._mesh_certificates import _boxes, _candidate_pairs

    budget = algebra._COORDINATE_BUDGET.get()

    collected: dict[
        tuple[Point, tuple[int, ...], int, tuple[int, int]], list[tuple[int, Polygon]]
    ] = {}
    plane_members: dict[Point, list[tuple[int, Polygon, tuple[int, ...]]]] = {}
    for index, row in enumerate(facets):
        with nullcontext() if budget is None else budget.temporary_scope():
            points = rational_points(vertices[row])
            key = plane_key(points)
            if key is None:
                raise ValueError("Source facets must define nondegenerate exact planes.")
            plane, axes, orientation, sign = key
            pair = (int(regions[index, 0]), int(regions[index, 1]))
            if sign < 0:
                pair = pair[::-1]
            projected = project(points, axes)
            if budget is not None:
                budget.retain_basis((plane, pair, index, projected, axes))
        if budget is not None:
            # Both source indexes keep this row beyond its algebra workspace.
            budget.reserve(2, 256)
        collected.setdefault((plane, axes, orientation, pair), []).append(
            (index, projected)
        )
        plane_members.setdefault(plane, []).append((index, projected, axes))
    overlaps = []
    used = 0
    for members in plane_members.values():
        if len(members) < 2:
            continue
        rows = facets[np.asarray(tuple(member[0] for member in members), dtype=np.int64)]
        firsts, seconds, exceeded = _candidate_pairs(
            *_boxes(vertices, rows),
            maximum_pairs - used,
        )
        used += firsts.size
        for first, second in zip(firsts.tolist(), seconds.tolist(), strict=True):
            with nullcontext() if budget is None else budget.temporary_scope():
                a, b = members[first], members[second]
                overlapping = intersection(a[1], b[1]) > 0
            if overlapping:
                if budget is not None:
                    budget.reserve(1, 128)
                overlaps.append((a[0], b[0]))
        if exceeded:
            return (), tuple(overlaps), used, True
    groups = []
    for (plane, axes, orientation, pair), members in sorted(collected.items()):
        with nullcontext() if budget is None else budget.temporary_scope():
            # The original facet proof already established this normalized plane
            # and projection. Reuse its exact images, not the discarded 3D rows.
            simplices = tuple(points for _, points in members)
            identifiers = tuple(index for index, _ in members)
            signed = tuple(signed_measure(points) for points in simplices)
            orientations = tuple(1 if value > 0 else -1 for value in signed)
            measures = tuple(abs(value) for value in signed)
            if len(simplices) == 1:
                box_lower = np.empty((0, len(axes)), dtype=np.float64)
                box_upper = np.empty((0, len(axes)), dtype=np.float64)
                simplex_keys: tuple[Polygon, ...] = ()
                simplex_key_owners: tuple[int, ...] = ()
            else:
                boxes = tuple(
                    _binary64_polygon_bounds(points, -math.inf, math.inf)
                    for points in simplices
                )
                box_lower = np.asarray(
                    tuple(lower for lower, _ in boxes),
                    dtype=np.float64,
                )
                box_upper = np.asarray(
                    tuple(upper for _, upper in boxes),
                    dtype=np.float64,
                )
                keyed = sorted(
                    (_simplex_key(points), index)
                    for index, points in enumerate(simplices)
                )
                simplex_keys = tuple(key for key, _ in keyed)
                simplex_key_owners = tuple(index for _, index in keyed)
            box_lower.setflags(write=False)
            box_upper.setflags(write=False)
            if budget is not None:
                # retain_basis traverses containers, not arbitrary dataclasses.
                budget.retain_basis(
                    (
                        plane,
                        axes,
                        pair,
                        identifiers,
                        simplices,
                        orientations,
                        measures,
                        box_lower,
                        box_upper,
                        simplex_keys,
                        simplex_key_owners,
                    )
                )
            group = SourceGroup(
                plane,
                axes,
                orientation,
                pair,
                identifiers,
                simplices,
                orientations,
                measures,
                box_lower,
                box_upper,
                simplex_keys,
                simplex_key_owners,
            )
        if budget is not None:
            # Fixed slots eliminate class-lifetime-dependent shared-key dict
            # allocations while retaining the actual complete record size.
            budget.reserve(1, sys.getsizeof(group))
        groups.append(group)
    return tuple(groups), tuple(overlaps), used, False


def convex_hull(points: Polygon) -> Polygon:
    budget = algebra._COORDINATE_BUDGET.get()
    if budget is not None:
        # Ordering and deduplication visit each exact point; hull turns below
        # own their arithmetic and output bounds.
        budget.reserve(len(points), 128 * len(points))
    points = tuple(sorted(set(points)))
    if not points or len(points[0]) == 1:
        return points[:1] if len(points) < 2 else (points[0], points[-1])
    lower: list[Point] = []
    upper: list[Point] = []
    for point in points:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple((*lower[:-1], *upper[:-1]))


def image_hull(
    coordinates: tuple[algebra.Expression, ...],
    domain: str,
    axes: tuple[int, ...],
    maximum_nodes: int,
) -> Polygon | None:
    dimension = len(axes)
    projected = tuple(coordinates[axis] for axis in axes)
    if any(isinstance(value, algebra.RationalPolynomial) for value in projected):
        controls = algebra.expression_control_points(
            projected, domain, dimension, maximum_nodes
        )
        return None if controls is None else convex_hull(controls)
    polynomial_projected = tuple(
        value for value in projected if not isinstance(value, algebra.RationalPolynomial)
    )
    if domain == "simplex":
        degree = max(
            (sum(index) for value in polynomial_projected for index in value), default=0
        )
        markers = ((degree, *(0 for _ in range(dimension - 1))),)
    else:
        degrees = tuple(
            max(
                (index[axis] for value in polynomial_projected for index in value),
                default=0,
            )
            for axis in range(dimension)
        )
        markers = tuple(
            tuple(degrees[axis] if i == axis else 0 for i in range(dimension))
            for axis in range(dimension)
        )
    coefficients = []
    for polynomial in polynomial_projected:
        elevated = dict(polynomial)
        for marker in markers:
            elevated.setdefault(marker, Fraction(0))
        if _node_count(elevated, domain, dimension) > maximum_nodes:
            return None
        coefficients.append(algebra.bernstein_coefficients(elevated, domain, dimension))
    return convex_hull(tuple(tuple(row) for row in zip(*coefficients, strict=True)))


def _simplex_contains(
    polygon: Polygon,
    simplex: Polygon,
    orientation: int,
    /,
) -> bool:
    edge_count = len(simplex)
    _reserve_fraction_work(
        (*simplex, *polygon),
        2 * edge_count + 5 * edge_count * len(polygon),
        edge_count * len(polygon),
        8,
    )
    for a, b in zip(simplex, (*simplex[1:], simplex[0]), strict=True):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if any(
            orientation * (dx * (point[1] - a[1]) - dy * (point[0] - a[0])) < 0
            for point in polygon
        ):
            return False
    return True


def _affine_triangle_corners(
    coordinates: tuple[algebra.Expression, ...],
    domain: str,
    /,
) -> tuple[Point, Point, Point] | None:
    if domain != "simplex" or any(
        isinstance(value, algebra.RationalPolynomial)
        or any(sum(index) > 1 for index in value)
        for value in coordinates
    ):
        return None
    reference = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    corners = tuple(
        tuple(algebra.expression_evaluate(value, point) for value in coordinates)
        for point in reference
    )
    return corners[0], corners[1], corners[2]


def _corners_on_plane(corners: tuple[Point, ...], plane: Point, /) -> bool:
    dimension = len(corners[0])
    _reserve_fraction_work(
        (*corners, plane),
        2 * len(corners) * dimension,
        len(corners),
        (2 * dimension + 1) * len(corners),
    )
    return not any(
        sum(
            (
                weight * value
                for weight, value in zip(
                    plane[:-1],
                    corner,
                    strict=True,
                )
            ),
            plane[-1],
        )
        for corner in corners
    )


def _prepare_affine_exact_source_simplex(
    exact: Polygon,
    /,
) -> _AffineSourceSimplex:
    key = plane_key(exact)
    if key is None or len(key[1]) != 2:
        raise ValueError("Authoritative PLC source triangle is exactly degenerate.")
    plane, axes, _, _ = key
    polygon = project(exact, axes)
    measure = signed_measure(polygon)
    if measure == 0:
        raise ValueError("Authoritative PLC source triangle is exactly degenerate.")
    return plane, axes, polygon, 1 if measure > 0 else -1


def _prepare_affine_source_simplex(points: np.ndarray, /) -> _AffineSourceSimplex:
    return _prepare_affine_exact_source_simplex(rational_points(points))


def _affine_simplex_corner_containment(
    corners: tuple[Point, Point, Point],
    source: _AffineSourceSimplex,
    /,
) -> tuple[Fraction, int] | None:
    plane, axes, simplex, source_orientation = source
    if not _corners_on_plane(corners, plane):
        return None
    polygon = tuple(tuple(corner[axis] for axis in axes) for corner in corners)
    measure = signed_measure(polygon)
    if measure == 0 or not _simplex_contains(polygon, simplex, source_orientation):
        return None
    orientation = source_orientation if measure > 0 else -source_orientation
    return measure, orientation


def _affine_segment_containment(
    endpoints: tuple[Point, Point],
    source: _AffineSourceSimplex,
    /,
) -> bool:
    plane, axes, simplex, source_orientation = source
    if not _corners_on_plane(endpoints, plane):
        return False
    segment = tuple(tuple(point[axis] for axis in axes) for point in endpoints)
    return _simplex_contains(segment, simplex, source_orientation)


def _affine_simplex_containment(
    coordinates: tuple[algebra.Expression, ...],
    domain: str,
    source: _AffineSourceSimplex,
    /,
) -> tuple[Fraction, int] | None:
    corners = _affine_triangle_corners(coordinates, domain)
    return (
        None if corners is None else _affine_simplex_corner_containment(corners, source)
    )


def _unique_containing_simplex(polygon: Polygon, group: SourceGroup, /) -> int | None:
    """Return the sole convex source simplex containing every target corner."""
    if len(group.simplices) == 1:
        return (
            0
            if _simplex_contains(
                polygon,
                group.simplices[0],
                group.orientations[0],
            )
            else None
        )
    key = _simplex_key(polygon)
    first = bisect_left(group.simplex_keys, key)
    after = bisect_right(group.simplex_keys, key, lo=first)
    if after == first + 1:
        owner = group.simplex_key_owners[first]
        if _simplex_contains(
            polygon,
            group.simplices[owner],
            group.orientations[owner],
        ):
            return owner
    target_lower_, target_upper_ = _binary64_polygon_bounds(
        polygon,
        math.inf,
        -math.inf,
    )
    target_lower = np.asarray(target_lower_, dtype=np.float64)
    target_upper = np.asarray(target_upper_, dtype=np.float64)
    budget = algebra._COORDINATE_BUDGET.get()
    if budget is not None:
        count, dimension = group.box_lower.shape
        budget.reserve(
            0,
            256
            + target_lower.nbytes
            + target_upper.nbytes
            + (2 * dimension + 10) * count,
        )
    # The caller has charged every broad-phase pair. These directed binary64
    # boxes only discard impossible owners; exact oriented edges prove the one
    # surviving owner below.
    candidates = np.flatnonzero(
        np.all(group.box_lower <= target_lower, axis=1)
        & np.all(group.box_upper >= target_upper, axis=1)
    )
    owner: int | None = None
    for index in candidates.tolist():
        if not _simplex_contains(
            polygon,
            group.simplices[index],
            group.orientations[index],
        ):
            continue
        if owner is not None:
            return None
        owner = index
    return owner


def mapped_containment(
    coordinates: tuple[algebra.Expression, ...],
    domain: str,
    group: SourceGroup,
    own: int,
    other: int,
    maximum_nodes: int,
    maximum_depth: int,
    maximum_pieces: int,
    maximum_pairs: int,
    work: SubdivisionLedger,
) -> tuple[str, Fraction, tuple[int, ...]]:
    if not ((own, other) == group.regions or (other, own) == group.regions):
        return "absent", Fraction(0), ()
    dimension = len(group.axes)
    corners = _affine_triangle_corners(coordinates, domain) if dimension == 2 else None
    if corners is not None:
        if not _corners_on_plane(corners, group.plane):
            return "absent", Fraction(0), ()
        polygon = tuple(tuple(corner[axis] for axis in group.axes) for corner in corners)
        raw = signed_measure(polygon)
        orientation = group.orientation * (1 if own == group.regions[0] else -1)
        measure = orientation * raw
        if measure <= 0:
            return "absent", Fraction(0), ()
        work.candidate_pairs += len(group.simplices)
        if work.candidate_pairs > maximum_pairs:
            return "unresolved", Fraction(0), ()
        owner = _unique_containing_simplex(polygon, group)
        if owner is not None:
            return "proven", measure, (owner,)
        intersections = tuple(intersection(polygon, target) for target in group.simplices)
        _reserve_fraction_work(
            (intersections,), len(intersections), 1, len(intersections)
        )
        if sum(intersections, Fraction(0)) != measure:
            return "absent", Fraction(0), ()
        candidates = tuple(
            index for index, value in enumerate(intersections) if value > 0
        )
        return "proven", measure, candidates
    plane = algebra.expression_add(
        algebra.expression_sum(
            tuple(
                algebra.expression_scale(value, weight)
                for value, weight in zip(coordinates, group.plane[:-1], strict=True)
            )
        ),
        algebra.constant(group.plane[-1], dimension),
    )
    if plane:
        return "absent", Fraction(0), ()
    orientation = group.orientation * (1 if own == group.regions[0] else -1)
    jacobian = projected_jacobian(coordinates, group.axes)
    signed = algebra.expression_scale(jacobian, orientation)
    raw_measure = projected_chart_measure(
        coordinates, domain, group.axes, jacobian=jacobian
    )
    if raw_measure is None:
        return "unresolved", Fraction(0), ()
    measure = orientation * raw_measure
    if measure <= 0:
        return "absent", Fraction(0), ()
    if not nonnegative(
        signed, domain, dimension, maximum_nodes, maximum_depth, maximum_pieces, work
    ):
        return "unresolved", Fraction(0), ()
    pending = [(coordinates, 0)]
    children = None
    candidates: tuple[int, ...] = ()
    while pending:
        current, depth = pending.pop()
        work.charge(depth)
        if work.pieces > maximum_pieces:
            return "unresolved", Fraction(0), ()
        hull = image_hull(current, domain, group.axes, maximum_nodes)
        if hull is None:
            return "unresolved", Fraction(0), ()
        work.candidate_pairs += len(group.simplices)
        if work.candidate_pairs > maximum_pairs:
            return "unresolved", Fraction(0), ()
        area = abs(signed_measure(hull))
        intersections = tuple(intersection(hull, target) for target in group.simplices)
        _reserve_fraction_work(
            (intersections,), len(intersections), 1, len(intersections)
        )
        if depth == 0:
            candidates = tuple(
                index for index, value in enumerate(intersections) if value > 0
            )
        if area > 0 and sum(intersections, Fraction(0)) == area:
            continue
        if depth >= maximum_depth:
            return "unresolved", Fraction(0), ()
        if children is None:
            children = _children(domain, dimension)
        pending.extend(
            (
                tuple(algebra.expression_compose(value, child) for value in current),
                depth + 1,
            )
            for child in children
        )
    return "proven", measure, candidates
