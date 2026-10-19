"""Certified section atlases and boundary clearance for represented B-Reps."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final, Literal, TypeAlias

import equinox as eqx
import numpy as np

from ..._strict import StrictModule
from ..._validation import nonnegative_integer
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    RationalEnclosureError,
)
from ...linalg._exact import solve_exact
from ...typing import parse
from .._interval_enclosure import (
    interval_add,
    interval_divide,
    interval_multiply,
    IntervalExtensionError,
)
from .._rational import (
    fraction_down,
    fraction_up,
    rational_cross,
    rational_dot,
    rational_vector,
)
from ._clearance_bounds import (
    branch_clearance,
    ClearanceBounds,
    ClearanceFace,
    prepare_clearance_faces,
)
from ._coordinate_request import coordinate_source_request
from ._exact import square_root_interval
from ._intersection import (
    intersect_surface_regions,
    ParametricIntersectionPolicy,
    SurfaceIntersectionResult,
)
from ._intersection_curve import SurfaceRegion
from ._model import BRepModel
from ._patches import CylinderPatch, PlanePatch
from ._query_membership import full_rectangle
from ._sphere_membership import prepare_full_sphere, SphereSource


BRepCertifiedQueryStatus: TypeAlias = Literal["certified", "unresolved", "unsupported"]
SectionOperand: TypeAlias = BRepModel | SurfaceRegion


@final
class BRepSectionResult(StrictModule):
    """Per-face-pair curve certificates retaining all unresolved parameter regions."""

    status: BRepCertifiedQueryStatus = eqx.field(static=True)
    intersections: tuple[SurfaceIntersectionResult, ...]
    face_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    def __init__(
        self,
        status: BRepCertifiedQueryStatus,
        intersections: tuple[SurfaceIntersectionResult, ...],
        face_pairs: tuple[tuple[int, int], ...],
        reason: str = "",
    ) -> None:
        if len(intersections) != len(face_pairs):
            raise ValueError(
                f"BRepSectionResult.intersections must match face_pairs; got {len(intersections)} certificates and {len(face_pairs)} pairs"
            )
        self.status = parse(status, BRepCertifiedQueryStatus, "status")
        self.intersections, self.face_pairs, self.reason = (
            intersections,
            face_pairs,
            reason,
        )


@final
class BRepClearanceResult(StrictModule):
    """Minimum boundary-distance enclosure with points attaining its upper bound.

    Point-enclosure rows correspond to the first and second operands.
    Unsupported families have [0, infinity] distance and no points.
    """

    status: BRepCertifiedQueryStatus = eqx.field(static=True)
    distance_lower: float = eqx.field(static=True)
    distance_upper: float = eqx.field(static=True)
    witness_lower: np.ndarray
    witness_upper: np.ndarray
    reason: str = eqx.field(static=True)
    unresolved_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    boxes_processed: int = eqx.field(static=True)

    def __init__(
        self,
        status: BRepCertifiedQueryStatus,
        distance_lower: float,
        distance_upper: float,
        witness_lower: np.ndarray,
        witness_upper: np.ndarray,
        reason: str = "",
        *,
        unresolved_pairs: tuple[tuple[int, int], ...] = (),
        boxes_processed: int = 0,
    ) -> None:
        _clearance_metadata(unresolved_pairs, boxes_processed)
        if not 0 <= distance_lower <= distance_upper:
            raise ValueError(
                f"BRepClearanceResult.distance_enclosure must be nonnegative and ordered; got [{distance_lower}, {distance_upper}]"
            )
        if witness_lower.shape != witness_upper.shape or witness_lower.shape not in (
            (0, 3),
            (2, 3),
        ):
            raise ValueError(
                f"BRepClearanceResult.witness_lower must match witness_upper with shape (2, 3) or (0, 3); got {witness_lower.shape} and {witness_upper.shape}"
            )
        if np.any(witness_lower > witness_upper):
            raise ValueError(
                "BRepClearanceResult.witness_lower must not exceed witness_upper; got reversed endpoints"
            )
        self.status = parse(status, BRepCertifiedQueryStatus, "status")
        self.distance_lower, self.distance_upper = distance_lower, distance_upper
        self.witness_lower, self.witness_upper = witness_lower, witness_upper
        self.reason = reason
        self.unresolved_pairs = unresolved_pairs
        self.boxes_processed = boxes_processed


def _clearance_metadata(pairs: tuple[tuple[int, int], ...], boxes_processed: int) -> None:
    nonnegative_integer(boxes_processed, "BRepClearanceResult.boxes_processed")
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(
                f"BRepClearanceResult.unresolved_pairs must contain face-index pairs; got {pair!r}"
            )
        for index in pair:
            nonnegative_integer(index, "BRepClearanceResult.unresolved_pairs.index")
    if pairs != tuple(sorted(set(pairs))):
        raise ValueError(
            f"BRepClearanceResult.unresolved_pairs must be unique and canonically ordered; got {pairs!r}"
        )


def _regions(
    operand: SectionOperand, budget: CoordinateEnclosureBudget
) -> tuple[SurfaceRegion, ...] | None:
    if isinstance(operand, SurfaceRegion):
        return (operand,)
    geometry = operand.geometry
    if geometry is None:
        return None
    boxes = np.asarray(operand.parameter_bounds)
    if not all(
        full_rectangle(geometry, face, patch, boxes[face])
        for face, patch in enumerate(operand.patches)
    ):
        return None
    return tuple(
        SurfaceRegion(patch, boxes[face], budget=budget)
        for face, patch in enumerate(operand.patches)
    )


def _project_box_corners(
    box: np.ndarray,
    origin: tuple[Fraction, ...],
    axes: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    gram: tuple[tuple[Fraction, ...], ...],
) -> list[tuple[Fraction, ...]]:
    parameters = []
    for mask in range(8):
        point = tuple(
            Fraction(float(box[(mask >> k) & 1, k])) - origin[k] for k in range(3)
        )
        solution = solve_exact(
            [list(row) for row in gram], [[rational_dot(a, point)] for a in axes]
        )
        parameters.append(tuple(row[0] for row in solution))
    return parameters


def _plane_region(
    plane: PlanePatch,
    regions: tuple[SurfaceRegion, ...],
    budget: CoordinateEnclosureBudget,
) -> SurfaceRegion:
    axes = (
        rational_vector(np.asarray(plane.first_axis)),
        rational_vector(np.asarray(plane.second_axis)),
    )
    origin = rational_vector(np.asarray(plane.origin))
    gram = tuple(tuple(rational_dot(a, b) for b in axes) for a in axes)
    parameters: list[tuple[Fraction, ...]] = []
    for region in regions:
        box = np.asarray(region.patch.bounding_box(region.parameter_box, budget=budget))
        parameters.extend(_project_box_corners(box, origin, axes, gram))
    lower = [fraction_down(min(point[k] for point in parameters)) for k in range(2)]
    upper = [fraction_up(max(point[k] for point in parameters)) for k in range(2)]
    # An open margin ensures an intersection on an extremal source box is not
    # accidentally placed on an artificial plane-domain boundary.
    return SurfaceRegion(
        plane,
        np.stack((np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf))),
        budget=budget,
    )


def _require_spatial_agreement(
    first: SectionOperand, second: SectionOperand | PlanePatch
) -> None:
    if isinstance(first, BRepModel) and isinstance(second, BRepModel):
        if first.coordinate_contract.spatial_id != second.coordinate_contract.spatial_id:
            raise ValueError(
                "BRepCertifiedQuery.operands.coordinate_contract must use the same spatial identity; got different spatial IDs"
            )


def brep_section(
    first: SectionOperand,
    second: SectionOperand | PlanePatch,
    *,
    maximum_boxes: int = 20000,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> BRepSectionResult:
    """Intersect source boundaries or an infinite plane, retaining unresolved regions.

    Rectangular authored face domains are supported. Nonrectangular trims
    explicitly refuse rather than returning intersections of untrimmed carriers.
    """
    _require_spatial_agreement(first, second)
    with coordinate_source_request(
        (first, second, maximum_boxes), budget=coordinate_budget
    ) as budget:
        first_regions = _regions(first, budget)
        if first_regions is None or not first_regions:
            return BRepSectionResult(
                "unsupported",
                (),
                (),
                "first operand lacks certified rectangular face domains",
            )
        second_regions = (
            (_plane_region(second, first_regions, budget),)
            if isinstance(second, PlanePatch)
            else _regions(second, budget)
        )
        if second_regions is None or not second_regions:
            return BRepSectionResult(
                "unsupported",
                (),
                (),
                "second operand lacks certified rectangular face domains",
            )
        policy = ParametricIntersectionPolicy(maximum_boxes=maximum_boxes)
        results: list[SurfaceIntersectionResult] = []
        pairs: list[tuple[int, int]] = []
        remaining = maximum_boxes
        for i, region in enumerate(first_regions):
            for j, other in enumerate(second_regions):
                if remaining <= 0:
                    return BRepSectionResult(
                        "unresolved",
                        tuple(results),
                        tuple(pairs),
                        "section box budget exhausted before all face pairs",
                    )
                policy = ParametricIntersectionPolicy(maximum_boxes=remaining)
                result = intersect_surface_regions(
                    region, other, policy=policy, coordinate_budget=budget
                )
                remaining -= result.work.boxes_processed
                results.append(result)
                pairs.append((i, j))
        status: BRepCertifiedQueryStatus = (
            "certified" if all(r.complete for r in results) else "unresolved"
        )
        return BRepSectionResult(status, tuple(results), tuple(pairs))


def _sphere(
    operand: SectionOperand, budget: CoordinateEnclosureBudget
) -> SphereSource | None:
    regions = _regions(operand, budget)
    if regions is None or len(regions) != 1:
        return None
    region = regions[0]
    box = np.asarray(region.parameter_box)
    if (
        box[1, 0] - box[0, 0] < math.tau
        or box[0, 1] > -math.pi / 2
        or box[1, 1] < math.pi / 2
    ):
        return None
    source = prepare_full_sphere(region.patch, budget=budget)
    if source is None:
        return None
    if any(
        sum((source.basis[k][i] * source.basis[k][j] for k in range(3)), Fraction(0))
        != (1 if i == j else 0)
        for i in range(3)
        for j in range(3)
    ):
        return None
    return source


def _rational_bounds(value: Fraction) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(fraction_down(value)), np.asarray(fraction_up(value))


def _concentric_sphere_points(
    first: SphereSource, second: SphereSource
) -> tuple[np.ndarray, np.ndarray]:
    points = [
        tuple(c + (source.radius if k == 0 else 0) for k, c in enumerate(source.center))
        for source in (first, second)
    ]
    return (
        np.asarray([[fraction_down(x) for x in point] for point in points]),
        np.asarray([[fraction_up(x) for x in point] for point in points]),
    )


def _separated_sphere_points(
    first: SphereSource,
    second: SphereSource,
    delta: tuple[Fraction, ...],
    distance: tuple[float, float],
    outside: bool,
) -> tuple[np.ndarray, np.ndarray]:
    direction = [
        interval_divide(
            _rational_bounds(x), (np.asarray(distance[0]), np.asarray(distance[1]))
        )
        for x in delta
    ]
    signs = (1, -1) if outside else ((1, 1) if first.radius > second.radius else (-1, -1))
    points_bounds = []
    for source, sign in ((first, signs[0]), (second, signs[1])):
        points_bounds.append(
            [
                interval_add(
                    _rational_bounds(c),
                    interval_multiply(_rational_bounds(sign * source.radius), v),
                )
                for c, v in zip(source.center, direction, strict=True)
            ]
        )
    return (
        np.asarray([[v[0] for v in point] for point in points_bounds]),
        np.asarray([[v[1] for v in point] for point in points_bounds]),
    )


def _intersecting_sphere_points(
    first: SphereSource,
    second: SphereSource,
    delta: tuple[Fraction, ...],
    squared: Fraction,
    perpendicular_direction: tuple[Fraction, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    factor = (squared + first.radius**2 - second.radius**2) / (2 * squared)
    height_squared = first.radius**2 - factor * factor * squared
    if perpendicular_direction is None:
        pivot = next(k for k, x in enumerate(delta) if x != 0)
        other = (pivot + 1) % 3
        perpendicular_list = [Fraction(0)] * 3
        perpendicular_list[pivot], perpendicular_list[other] = -delta[other], delta[pivot]
        perpendicular = tuple(perpendicular_list)
    else:
        perpendicular = perpendicular_direction
    scale = square_root_interval(
        height_squared / rational_dot(perpendicular, perpendicular)
    )
    point = [
        interval_add(
            _rational_bounds(c + factor * d),
            interval_multiply(
                _rational_bounds(p), (np.asarray(scale[0]), np.asarray(scale[1]))
            ),
        )
        for c, d, p in zip(first.center, delta, perpendicular, strict=True)
    ]
    return np.asarray([[v[0] for v in point]] * 2), np.asarray(
        [[v[1] for v in point]] * 2
    )


def _sphere_clearance(first: SphereSource, second: SphereSource) -> BRepClearanceResult:
    delta = tuple(b - a for a, b in zip(first.center, second.center, strict=True))
    squared = rational_dot(delta, delta)
    r, s = first.radius, second.radius
    dlo, dhi = square_root_interval(squared)
    # Rational squared comparisons decide separation, containment, or intersection.
    outside = squared > (r + s) ** 2
    inside = squared < (r - s) ** 2
    if outside:
        lower = max(0.0, fraction_down(Fraction(dlo) - r - s))
        upper = fraction_up(Fraction(dhi) - r - s)
    elif inside:
        lower = max(0.0, fraction_down(abs(r - s) - Fraction(dhi)))
        upper = fraction_up(abs(r - s) - Fraction(dlo))
    else:
        lower = upper = 0.0
    if squared == 0:
        lo, hi = _concentric_sphere_points(first, second)
    elif outside or inside:
        lo, hi = _separated_sphere_points(first, second, delta, (dlo, dhi), outside)
    else:
        lo, hi = _intersecting_sphere_points(first, second, delta, squared)
    return BRepClearanceResult("certified", lower, upper, lo, hi)


def _unavailable_clearance(
    status: BRepCertifiedQueryStatus, counts: tuple[int, ...], reason: str
) -> BRepClearanceResult:
    empty = np.empty((0, 3), dtype=np.float64)
    return BRepClearanceResult(
        status,
        0.0,
        np.inf,
        empty,
        empty,
        reason,
        unresolved_pairs=tuple(
            (i, j) for i in range(counts[0]) for j in range(counts[1])
        ),
    )


def _cylinder_source(
    operand: SectionOperand, budget: CoordinateEnclosureBudget
) -> tuple[CylinderPatch, SurfaceRegion] | None:
    regions = _regions(operand, budget)
    if regions is None or len(regions) != 1:
        return None
    region = regions[0]
    patch = region.patch
    if (
        not isinstance(patch, CylinderPatch)
        or region.upper[0] - region.lower[0] < math.tau
    ):
        return None
    axes = tuple(
        rational_vector(np.asarray(a))
        for a in (patch.first_axis, patch.second_axis, patch.axis)
    )
    if any(
        rational_dot(a, b) != (1 if i == j else 0)
        for i, a in enumerate(axes)
        for j, b in enumerate(axes)
    ):
        return None
    return patch, region


def _circle_source(center: tuple[Fraction, ...], radius: Fraction) -> SphereSource:
    identity = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    return SphereSource(
        center,
        identity,
        Fraction(1),
        radius,
        (fraction_down(radius), fraction_up(radius)),
        (1.0, 1.0),
    )


def _circle_witnesses(
    first: SphereSource,
    second: SphereSource,
    axis: tuple[Fraction, ...],
    radial_axis: tuple[Fraction, ...],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    delta = tuple(b - a for a, b in zip(first.center, second.center, strict=True))
    squared = rational_dot(delta, delta)
    result = _sphere_clearance(first, second)
    if squared == 0:
        points = [
            tuple(
                c + source.radius * a
                for c, a in zip(source.center, radial_axis, strict=True)
            )
            for source in (first, second)
        ]
        lo = np.asarray([[fraction_down(x) for x in p] for p in points])
        hi = np.asarray([[fraction_up(x) for x in p] for p in points])
    elif (
        (first.radius - second.radius) ** 2
        <= squared
        <= (first.radius + second.radius) ** 2
    ):
        lo, hi = _intersecting_sphere_points(
            first, second, delta, squared, rational_cross(axis, delta)
        )
    else:
        lo, hi = result.witness_lower, result.witness_upper
    return lo, hi, result.distance_lower, result.distance_upper


def _parallel_cylinders(
    first: SectionOperand, second: SectionOperand, budget: CoordinateEnclosureBudget
) -> BRepClearanceResult | None:
    a, b = _cylinder_source(first, budget), _cylinder_source(second, budget)
    if a is None or b is None:
        return None
    pa, ra = a
    pb, rb = b
    axis, other = (
        rational_vector(np.asarray(pa.axis)),
        rational_vector(np.asarray(pb.axis)),
    )
    sign = rational_dot(axis, other)
    if abs(sign) != 1:
        return None
    ca, cb = (
        rational_vector(np.asarray(pa.origin)),
        rational_vector(np.asarray(pb.origin)),
    )
    offset = rational_dot(tuple(y - x for x, y in zip(ca, cb, strict=True)), axis)
    heights_a = tuple(Fraction(x) for x in (ra.lower[1], ra.upper[1]))
    heights_b = tuple(
        sorted(offset + sign * Fraction(x) for x in (rb.lower[1], rb.upper[1]))
    )
    if heights_a[1] < heights_b[0]:
        ha, hb = heights_a[1], heights_b[0]
    elif heights_b[1] < heights_a[0]:
        ha, hb = heights_a[0], heights_b[1]
    else:
        ha = hb = max(heights_a[0], heights_b[0])
    projected = tuple(c - offset * v for c, v in zip(cb, axis, strict=True))
    lo, hi, lower, upper = _circle_witnesses(
        _circle_source(ca, Fraction(float(pa.radius))),
        _circle_source(projected, Fraction(float(pb.radius))),
        axis,
        rational_vector(np.asarray(pa.first_axis)),
    )
    for row, height in enumerate((ha, hb)):
        for k, direction in enumerate(axis):
            lo[row, k] = fraction_down(Fraction(float(lo[row, k])) + height * direction)
            hi[row, k] = fraction_up(Fraction(float(hi[row, k])) + height * direction)
    lower = square_root_interval(Fraction(lower) ** 2 + (ha - hb) ** 2)[0]
    upper = square_root_interval(Fraction(upper) ** 2 + (ha - hb) ** 2)[1]
    return BRepClearanceResult("certified", lower, upper, lo, hi)


def brep_clearance(
    first: SectionOperand,
    second: SectionOperand,
    *,
    maximum_boxes: int = 256,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> BRepClearanceResult:
    """Enclose minimum boundary distance using certified source charts.

    Full spheres and full parallel-axis cylinder walls have closed-form
    decisions. Other faces use best-first interval subdivision, retaining
    potentially minimizing face pairs when the explicit box budget ends.
    The reported point intervals enclose points on the authored trimmed faces.
    """
    if (
        isinstance(maximum_boxes, bool)
        or not isinstance(maximum_boxes, int)
        or maximum_boxes < 0
    ):
        raise ValueError(
            f"brep_clearance.maximum_boxes must be a nonnegative integer; got {maximum_boxes!r}"
        )
    _require_spatial_agreement(first, second)
    counts = tuple(
        len(value.patches) if isinstance(value, BRepModel) else 1
        for value in (first, second)
    )
    with coordinate_source_request(
        (first, second), budget=coordinate_budget, rows=8 * maximum_boxes + sum(counts)
    ) as budget:
        try:
            a, b = _sphere(first, budget), _sphere(second, budget)
            if a is not None and b is not None:
                return _sphere_clearance(a, b)
            cylinders = _parallel_cylinders(first, second, budget)
            if cylinders is not None:
                return cylinders
            faces = (
                prepare_clearance_faces(first, budget),
                prepare_clearance_faces(second, budget),
            )
            if faces[0] is None or faces[1] is None or not faces[0] or not faces[1]:
                return _unavailable_clearance(
                    "unsupported", counts, "operand lacks represented source face domains"
                )

            def closed_pair(a: ClearanceFace, b: ClearanceFace) -> ClearanceBounds | None:
                if a.trim is not None or b.trim is not None:
                    return None
                result = _parallel_cylinders(a.region, b.region, budget)
                if result is None:
                    return None
                return ClearanceBounds(
                    result.distance_lower,
                    result.distance_upper,
                    result.witness_lower,
                    result.witness_upper,
                    (),
                    0,
                )

            result = branch_clearance(
                faces[0], faces[1], maximum_boxes, budget, closed_pair
            )
        except (CoordinateEnclosureResourceError, RationalEnclosureError) as error:
            return _unavailable_clearance("unresolved", counts, str(error))
        except (NotImplementedError, IntervalExtensionError) as error:
            return _unavailable_clearance("unsupported", counts, str(error))
        return BRepClearanceResult(
            "unresolved" if result.unresolved_pairs else "certified",
            result.lower,
            result.upper,
            result.witness_lower,
            result.witness_upper,
            "clearance box budget or source trim proof unresolved"
            if result.unresolved_pairs
            else "",
            unresolved_pairs=result.unresolved_pairs,
            boxes_processed=result.boxes_processed,
        )
