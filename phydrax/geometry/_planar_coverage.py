#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rational clipping and Bernstein image containment in planar source unions."""

from __future__ import annotations

import math
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

import phydrax.geometry._embedding_state as embedding_state_owner
from phydrax._strict import StrictModule
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
)
from phydrax.typing import checked, parse

from ..discretization import _coordinate_enclosure as algebra
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ._mapped_coverage import (
    nonnegative,
    projected_chart_measure,
    projected_jacobian,
    subdivision_children,
    SubdivisionLedger,
)
from ._planar_primitives import (
    binary64_polygon_bounds,
    convex_hull,
    intersection,
    plane_key,
    Point,
    Polygon,
    project,
    rational_points,
    reserve_fraction_work,
    signed_measure,
    simplex_key,
)


type PlanarContainmentStatus = Literal["absent", "unresolved", "proven"]


class AffineSourceSimplex(StrictModule):
    """A source simplex's plane, projection, and oriented projected polygon."""

    plane: Point
    axes: tuple[int, ...]
    polygon: Polygon
    orientation: int


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


class _SourceCollection(StrictModule):
    collected: dict[
        tuple[Point, tuple[int, ...], int, tuple[int, int]], list[tuple[int, Polygon]]
    ]
    plane_members: dict[Point, list[tuple[int, Polygon, tuple[int, ...]]]]


class _SourceOverlaps(StrictModule):
    overlaps: tuple[tuple[int, int], ...]
    used: int
    exceeded: bool


def _collect_source_facets(
    vertices: np.ndarray,
    facets: np.ndarray,
    regions: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> _SourceCollection:
    collected: dict[
        tuple[Point, tuple[int, ...], int, tuple[int, int]], list[tuple[int, Polygon]]
    ] = {}
    plane_members: dict[Point, list[tuple[int, Polygon, tuple[int, ...]]]] = {}
    for index, row in enumerate(facets):
        with budget.temporary_scope():
            points = rational_points(vertices[row], budget=budget)
            key = plane_key(points, budget=budget)
            if key is None:
                raise ValueError(
                    "source_groups.facets must define nondegenerate planes; got degenerate facet"
                )
            plane, axes, orientation, sign = key
            pair = (int(regions[index, 0]), int(regions[index, 1]))
            if sign < 0:
                pair = pair[::-1]
            projected = project(points, axes, budget=budget)
            budget.retain_basis((plane, pair, index, projected, axes))
        budget.reserve(2, 256)
        collected.setdefault((plane, axes, orientation, pair), []).append(
            (index, projected)
        )
        plane_members.setdefault(plane, []).append((index, projected, axes))
    return _SourceCollection(collected, plane_members)


def _source_overlaps(
    vertices: np.ndarray,
    facets: np.ndarray,
    plane_members: dict[Point, list[tuple[int, Polygon, tuple[int, ...]]]],
    maximum_pairs: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _SourceOverlaps:
    overlaps = []
    used = 0
    for members in plane_members.values():
        if len(members) < 2:
            continue
        rows = facets[np.asarray(tuple(member[0] for member in members), dtype=np.int64)]
        firsts, seconds, exceeded = embedding_state_owner.candidate_pairs(
            *embedding_state_owner.boxes(vertices, rows),
            maximum_pairs - used,
        )
        used += firsts.size
        for first, second in zip(firsts.tolist(), seconds.tolist(), strict=True):
            with budget.temporary_scope():
                a, b = members[first], members[second]
                overlapping = intersection(a[1], b[1], budget=budget) > 0
            if overlapping:
                budget.reserve(1, 128)
                overlaps.append((a[0], b[0]))
        if exceeded:
            return _SourceOverlaps(tuple(overlaps), used, True)
    return _SourceOverlaps(tuple(overlaps), used, False)


def _prepare_source_group(
    plane: Point,
    axes: tuple[int, ...],
    orientation: int,
    pair: tuple[int, int],
    members: list[tuple[int, Polygon]],
    *,
    budget: CoordinateEnclosureBudget,
) -> SourceGroup:
    with budget.temporary_scope():
        # Reuse the normalized plane and projected facet images.
        simplices = tuple(points for _, points in members)
        identifiers = tuple(index for index, _ in members)
        signed = tuple(
            rational_coefficient(
                signed_measure(points, budget=budget),
                role="planar_coverage.source_measure",
            )
            for points in simplices
        )
        orientations = tuple(1 if value > 0 else -1 for value in signed)
        measures = tuple(abs(value) for value in signed)
        if len(simplices) == 1:
            box_lower = np.empty((0, len(axes)), dtype=np.float64)
            box_upper = np.empty((0, len(axes)), dtype=np.float64)
            simplex_keys: tuple[Polygon, ...] = ()
            simplex_key_owners: tuple[int, ...] = ()
        else:
            boxes = tuple(
                binary64_polygon_bounds(points, -math.inf, math.inf, budget=budget)
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
                (simplex_key(points, budget=budget), index)
                for index, points in enumerate(simplices)
            )
            simplex_keys = tuple(key for key, _ in keyed)
            simplex_key_owners = tuple(index for _, index in keyed)
        box_lower.setflags(write=False)
        box_upper.setflags(write=False)
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
    budget.reserve(1, sys.getsizeof(group))
    return group


@checked
def source_groups(
    vertices: np.ndarray,
    facets: np.ndarray,
    regions: np.ndarray,
    maximum_pairs: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[SourceGroup, ...], tuple[tuple[int, int], ...], int, bool]:
    """Group coplanar labeled source facets and report positive-area overlaps."""
    collected = _collect_source_facets(vertices, facets, regions, budget=budget)
    overlaps = _source_overlaps(
        vertices, facets, collected.plane_members, maximum_pairs, budget=budget
    )
    if overlaps.exceeded:
        return (), overlaps.overlaps, overlaps.used, True
    groups = tuple(
        _prepare_source_group(plane, axes, orientation, pair, members, budget=budget)
        for (plane, axes, orientation, pair), members in sorted(
            collected.collected.items()
        )
    )
    return groups, overlaps.overlaps, overlaps.used, False


@checked
def image_hull(
    coordinates: tuple[algebra.Expression, ...],
    domain: str,
    axes: tuple[int, ...],
    maximum_nodes: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Polygon | None:
    domain = parse(domain, algebra.BernsteinDomain, "image_hull.domain")
    dimension = len(axes)
    projected = tuple(coordinates[axis] for axis in axes)
    if any(isinstance(value, algebra.RationalPolynomial) for value in projected):
        controls = algebra.expression_control_points(
            projected, domain, dimension, maximum_nodes, budget=budget
        )
        return None if controls is None else convex_hull(controls, budget=budget)
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
        if algebra.expression_node_count(elevated, domain, dimension) > maximum_nodes:
            return None
        coefficients.append(
            algebra.bernstein_coefficients(elevated, domain, dimension, budget=budget)
        )
    return convex_hull(
        tuple(tuple(row) for row in zip(*coefficients, strict=True)), budget=budget
    )


def _simplex_contains(
    polygon: Polygon,
    simplex: Polygon,
    orientation: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    edge_count = len(simplex)
    reserve_fraction_work(
        (*simplex, *polygon),
        2 * edge_count + 5 * edge_count * len(polygon),
        edge_count * len(polygon),
        8,
        budget=budget,
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
    *,
    budget: CoordinateEnclosureBudget,
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
        tuple(
            algebra.expression_evaluate(value, point, budget=budget)
            for value in coordinates
        )
        for point in reference
    )
    return corners[0], corners[1], corners[2]


def _corners_on_plane(
    corners: tuple[Point, ...],
    plane: Point,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    dimension = len(corners[0])
    reserve_fraction_work(
        (*corners, plane),
        2 * len(corners) * dimension,
        len(corners),
        (2 * dimension + 1) * len(corners),
        budget=budget,
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


@checked
def prepare_affine_exact_source_simplex(
    points: Polygon, /, *, budget: CoordinateEnclosureBudget
) -> AffineSourceSimplex:
    key = plane_key(points, budget=budget)
    if key is None or len(key[1]) != 2:
        raise ValueError(
            "prepare_affine_exact_source_simplex.points must define a nondegenerate source triangle; got degenerate plane"
        )
    plane, axes, _, _ = key
    polygon = project(points, axes, budget=budget)
    measure = signed_measure(polygon, budget=budget)
    if measure == 0:
        raise ValueError(
            "prepare_affine_exact_source_simplex.points must define a nondegenerate source triangle; got zero projected measure"
        )
    return AffineSourceSimplex(plane, axes, polygon, 1 if measure > 0 else -1)


@checked
def affine_simplex_corner_containment(
    corners: tuple[Point, Point, Point],
    source: AffineSourceSimplex,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, int] | None:
    plane, axes, simplex, source_orientation = (
        source.plane,
        source.axes,
        source.polygon,
        source.orientation,
    )
    if not _corners_on_plane(corners, plane, budget=budget):
        return None
    polygon = tuple(tuple(corner[axis] for axis in axes) for corner in corners)
    measure = rational_coefficient(
        signed_measure(polygon, budget=budget), role="planar_coverage.simplex_measure"
    )
    if measure == 0 or not _simplex_contains(
        polygon, simplex, source_orientation, budget=budget
    ):
        return None
    orientation = source_orientation if measure > 0 else -source_orientation
    return measure, orientation


@checked
def affine_segment_containment(
    endpoints: tuple[Point, Point],
    source: AffineSourceSimplex,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    plane, axes, simplex, source_orientation = (
        source.plane,
        source.axes,
        source.polygon,
        source.orientation,
    )
    if not _corners_on_plane(endpoints, plane, budget=budget):
        return False
    segment = tuple(tuple(point[axis] for axis in axes) for point in endpoints)
    return _simplex_contains(segment, simplex, source_orientation, budget=budget)


def _unique_containing_simplex(
    polygon: Polygon, group: SourceGroup, /, *, budget: CoordinateEnclosureBudget
) -> int | None:
    """Return the sole convex source simplex containing every target corner."""
    if len(group.simplices) == 1:
        return (
            0
            if _simplex_contains(
                polygon, group.simplices[0], group.orientations[0], budget=budget
            )
            else None
        )
    key = simplex_key(polygon, budget=budget)
    first = bisect_left(group.simplex_keys, key)
    after = bisect_right(group.simplex_keys, key, lo=first)
    if after == first + 1:
        owner = group.simplex_key_owners[first]
        if _simplex_contains(
            polygon, group.simplices[owner], group.orientations[owner], budget=budget
        ):
            return owner
    target_lower_, target_upper_ = binary64_polygon_bounds(
        polygon, math.inf, -math.inf, budget=budget
    )
    target_lower = np.asarray(target_lower_, dtype=np.float64)
    target_upper = np.asarray(target_upper_, dtype=np.float64)

    count, dimension = group.box_lower.shape
    budget.reserve(
        0,
        256 + target_lower.nbytes + target_upper.nbytes + (2 * dimension + 10) * count,
    )
    # The caller has charged every broad-phase pair. These directed binary64
    # boxes discard impossible owners; rational oriented edges prove containment
    # for the surviving owner below.
    candidates = np.flatnonzero(
        np.all(group.box_lower <= target_lower, axis=1)
        & np.all(group.box_upper >= target_upper, axis=1)
    )
    owner: int | None = None
    for index in candidates.tolist():
        if not _simplex_contains(
            polygon, group.simplices[index], group.orientations[index], budget=budget
        ):
            continue
        if owner is not None:
            return None
        owner = index
    return owner


class _Containment(StrictModule):
    outcome: PlanarContainmentStatus
    measure: Fraction
    candidates: tuple[int, ...] = ()


def _group_intersections(
    polygon: Polygon, group: SourceGroup, /, *, budget: CoordinateEnclosureBudget
) -> tuple[Fraction, ...]:
    """Expand only intersections not excluded by the prepared outward boxes."""
    if len(group.simplices) == 1:
        return (
            rational_coefficient(
                intersection(polygon, group.simplices[0], budget=budget),
                role="planar_coverage.intersection_measure",
            ),
        )
    lower, upper = binary64_polygon_bounds(polygon, -math.inf, math.inf, budget=budget)
    count, dimension = group.box_lower.shape
    budget.reserve(
        2 * dimension * count,
        256 + 16 * dimension + (2 * dimension + 10) * count,
    )
    candidates = np.all(
        group.box_lower <= np.asarray(upper, dtype=np.float64), axis=1
    ) & np.all(group.box_upper >= np.asarray(lower, dtype=np.float64), axis=1)
    # Strict box separation excludes the entire compact image. Keep zeros in
    # source order, including all touching boxes, so overlap witnesses and
    # rational accumulation retain their declared meanings.
    return tuple(
        rational_coefficient(
            intersection(polygon, target, budget=budget),
            role="planar_coverage.intersection_measure",
        )
        if candidate
        else Fraction(0)
        for target, candidate in zip(group.simplices, candidates, strict=True)
    )


def _affine_group_containment(
    corners: tuple[Point, Point, Point],
    group: SourceGroup,
    own: int,
    maximum_pairs: int,
    work: SubdivisionLedger,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Containment:
    if not _corners_on_plane(corners, group.plane, budget=budget):
        return _Containment("absent", Fraction(0), ())
    polygon = tuple(tuple(corner[axis] for axis in group.axes) for corner in corners)
    raw = rational_coefficient(
        signed_measure(polygon, budget=budget), role="planar_coverage.group_measure"
    )
    orientation = group.orientation * (1 if own == group.regions[0] else -1)
    measure = orientation * raw
    if measure <= 0:
        return _Containment("absent", Fraction(0), ())
    work.candidate_pairs += len(group.simplices)
    if work.candidate_pairs > maximum_pairs:
        return _Containment("unresolved", Fraction(0), ())
    owner = _unique_containing_simplex(polygon, group, budget=budget)
    if owner is not None:
        return _Containment("proven", measure, (owner,))
    intersections = _group_intersections(polygon, group, budget=budget)
    reserve_fraction_work(
        (intersections,), len(intersections), 1, len(intersections), budget=budget
    )
    if sum(intersections, Fraction(0)) != measure:
        return _Containment("absent", Fraction(0), ())
    candidates = tuple(index for index, value in enumerate(intersections) if value > 0)
    return _Containment("proven", measure, candidates)


def _mapped_group_measure(
    coordinates: tuple[algebra.Expression, ...],
    domain: algebra.BernsteinDomain,
    group: SourceGroup,
    own: int,
    maximum_nodes: int,
    maximum_depth: int,
    maximum_pieces: int,
    work: SubdivisionLedger,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Containment:
    dimension = len(group.axes)
    plane = algebra.expression_add(
        algebra.expression_sum(
            tuple(
                algebra.expression_scale(value, weight, budget=budget)
                for value, weight in zip(coordinates, group.plane[:-1], strict=True)
            ),
            budget=budget,
        ),
        algebra.constant(group.plane[-1], dimension, budget=budget),
        budget=budget,
    )
    if plane:
        return _Containment("absent", Fraction(0), ())
    orientation = group.orientation * (1 if own == group.regions[0] else -1)
    jacobian = projected_jacobian(coordinates, group.axes, budget=budget)
    signed = algebra.expression_scale(jacobian, orientation, budget=budget)
    raw_measure = projected_chart_measure(
        coordinates, domain, group.axes, jacobian=jacobian, budget=budget
    )
    if raw_measure is None:
        return _Containment("unresolved", Fraction(0), ())
    measure = orientation * raw_measure
    if measure <= 0:
        return _Containment("absent", Fraction(0), ())
    if not nonnegative(
        signed,
        domain,
        dimension,
        maximum_nodes,
        maximum_depth,
        maximum_pieces,
        work,
        budget=budget,
    ):
        return _Containment("unresolved", Fraction(0), ())
    return _Containment("proven", measure)


def _subdivision_group_containment(
    coordinates: tuple[algebra.Expression, ...],
    domain: algebra.BernsteinDomain,
    group: SourceGroup,
    measure: Fraction,
    maximum_nodes: int,
    maximum_depth: int,
    maximum_pieces: int,
    maximum_pairs: int,
    work: SubdivisionLedger,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Containment:
    dimension = len(group.axes)
    pending = [(coordinates, 0)]
    children = None
    candidates: tuple[int, ...] = ()
    while pending:
        current, depth = pending.pop()
        work.charge(depth)
        if work.pieces > maximum_pieces:
            return _Containment("unresolved", Fraction(0), ())
        hull = image_hull(current, domain, group.axes, maximum_nodes, budget=budget)
        if hull is None:
            return _Containment("unresolved", Fraction(0), ())
        work.candidate_pairs += len(group.simplices)
        if work.candidate_pairs > maximum_pairs:
            return _Containment("unresolved", Fraction(0), ())
        area = abs(signed_measure(hull, budget=budget))
        intersections = _group_intersections(hull, group, budget=budget)
        reserve_fraction_work(
            (intersections,), len(intersections), 1, len(intersections), budget=budget
        )
        if depth == 0:
            candidates = tuple(
                index for index, value in enumerate(intersections) if value > 0
            )
        if area > 0 and sum(intersections, Fraction(0)) == area:
            continue
        if depth >= maximum_depth:
            return _Containment("unresolved", Fraction(0), ())
        if children is None:
            children = subdivision_children(domain, dimension, budget=budget)
        pending.extend(
            (
                tuple(
                    algebra.expression_compose(value, child, budget=budget)
                    for value in current
                ),
                depth + 1,
            )
            for child in children
        )
    return _Containment("proven", measure, candidates)


@checked
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
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[PlanarContainmentStatus, Fraction, tuple[int, ...]]:
    """Prove an oriented chart image is contained in its labeled source union."""
    if not ((own, other) == group.regions or (other, own) == group.regions):
        return "absent", Fraction(0), ()
    domain = parse(domain, algebra.BernsteinDomain, "mapped_containment.domain")
    corners = (
        _affine_triangle_corners(coordinates, domain, budget=budget)
        if len(group.axes) == 2
        else None
    )
    if corners is not None:
        result = _affine_group_containment(
            corners, group, own, maximum_pairs, work, budget=budget
        )
    else:
        result = _mapped_group_measure(
            coordinates,
            domain,
            group,
            own,
            maximum_nodes,
            maximum_depth,
            maximum_pieces,
            work,
            budget=budget,
        )
        if result.outcome == "proven":
            result = _subdivision_group_containment(
                coordinates,
                domain,
                group,
                result.measure,
                maximum_nodes,
                maximum_depth,
                maximum_pieces,
                maximum_pairs,
                work,
                budget=budget,
            )
    return result.outcome, result.measure, result.candidates
