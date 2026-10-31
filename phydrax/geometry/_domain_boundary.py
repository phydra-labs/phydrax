#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Source-bound boundary queries and continuous chart coverage for MeshingDomain."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from fractions import Fraction
from typing import final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar, positive_integer
from ..discretization._coordinate_enclosure import Polynomial
from ..discretization._exact_algebraic_geometry import (
    coefficient_payload,
    coefficient_rounding_error,
)
from ..linalg._exact import determinant_exact
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ..typing import checked
from ._embedding_state import MeshCertificateFinding
from ._interval_enclosure import prepare_interval_function, PreparedIntervalFunction
from ._meshing_domain import MeshingDomain, PatchCurveUse, PatchPoleUse
from ._rational import fraction_down, fraction_endpoint, fraction_up, rational_vector
from ._source_boundary_records import (
    SourceBoundaryChartCover,
    SourceBoundaryDistance,
    SourceBoundarySamples,
)
from ._triangulation import ConstrainedDelaunayTriangulation
from .brep._patches import AbstractSurfacePatch, CircleCurve, ExactLineCurve, LineCurve


_Point2 = tuple[Fraction, Fraction]
_Triangle2 = tuple[_Point2, _Point2, _Point2]
_Point3 = tuple[Fraction, Fraction, Fraction]
_Triangle3 = tuple[_Point3, _Point3, _Point3]


def _point2(values: tuple[object, ...], /) -> _Point2:
    if len(values) != 2:
        raise ValueError("An exact p-curve point must have two coordinates.")
    first, second = values
    if not isinstance(first, Fraction) or not isinstance(second, Fraction):
        raise ValueError("An exact p-curve endpoint must be rational.")
    return first, second


def _point3(values: tuple[object, ...], /) -> _Point3:
    if len(values) != 3:
        raise ValueError("An exact boundary point must have three coordinates.")
    first, second, third = values
    if (
        not isinstance(first, Fraction)
        or not isinstance(second, Fraction)
        or not isinstance(third, Fraction)
    ):
        raise ValueError("An exact boundary point must be rational.")
    return first, second, third


def _triangle3(values: tuple[tuple[object, ...], ...], /) -> _Triangle3:
    if len(values) != 3:
        raise ValueError("A source facet must be triangular.")
    return (_point3(values[0]), _point3(values[1]), _point3(values[2]))


def _has_exact_rational_surface_span_source(surface: AbstractSurfacePatch, /) -> bool:
    from ._meshing_domain import _exact_rational_surface_spans

    return _exact_rational_surface_spans(surface) is not None


def _exact_span_point_gap_upper(
    surface: AbstractSurfacePatch,
    parameters: _Point2,
    realized: np.ndarray,
    /,
) -> Fraction:
    """Bound a realized corner's distance to its exact rational span source point."""
    from ._source_interpolation import _exact_rational_surface_point_bounds

    source = _exact_rational_surface_point_bounds(surface, parameters)
    target = np.asarray(realized, dtype=np.float64)
    if (
        source is None
        or len(source) != 3
        or target.shape != (3,)
        or not np.all(np.isfinite(target))
    ):
        raise ValueError(
            "An exact surface source point has no exact rational span enclosure."
        )
    squared = Fraction(0)
    for (lower, upper), value in zip(source, target, strict=True):
        represented = Fraction.from_float(float(value))
        coordinate_error = max(abs(lower - represented), abs(upper - represented))
        squared += coordinate_error * coordinate_error
    return fraction_sqrt_interval(squared)[1]


def _cross2(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    third: tuple[Fraction, Fraction],
    /,
) -> Fraction:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def _signed_area2(points: tuple[tuple[Fraction, Fraction], ...], /) -> Fraction:
    return sum(
        (
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(points, (*points[1:], points[0]), strict=True)
        ),
        Fraction(0),
    )


def _inside_polygon(
    point: tuple[Fraction, Fraction],
    polygon: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> bool:
    inside = False
    x, y = point
    for first, second in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
        if (first[1] > y) != (second[1] > y):
            crossing = first[0] + (y - first[1]) * (second[0] - first[0]) / (
                second[1] - first[1]
            )
            if x < crossing:
                inside = not inside
    return inside


def _inside_triangle(
    point: tuple[Fraction, Fraction],
    triangle: tuple[tuple[Fraction, Fraction], ...],
    orientation: int,
    /,
) -> bool:
    return all(
        orientation * _cross2(first, second, point) >= 0
        for first, second in zip(triangle, (*triangle[1:], triangle[0]), strict=True)
    )


def _hole_seed(
    polygon: tuple[tuple[Fraction, Fraction], ...], /
) -> tuple[Fraction, Fraction]:
    signed = _signed_area2(polygon)
    if signed == 0:
        raise ValueError("A source trim hole has zero exact chart area.")
    orientation = 1 if signed > 0 else -1
    count = len(polygon)
    for index in range(count):
        triangle = (
            polygon[(index - 1) % count],
            polygon[index],
            polygon[(index + 1) % count],
        )
        if orientation * _cross2(*triangle) <= 0:
            continue
        if any(
            _inside_triangle(point, triangle, orientation)
            for candidate, point in enumerate(polygon)
            if candidate not in ((index - 1) % count, index, (index + 1) % count)
        ):
            continue
        center = (
            sum((point[0] for point in triangle), Fraction(0)) / 3,
            sum((point[1] for point in triangle), Fraction(0)) / 3,
        )
        if _inside_polygon(center, polygon):
            return center
    raise ValueError("No exact interior point was found for a source trim hole.")


def _line_endpoints(use: PatchCurveUse | PatchPoleUse, /) -> tuple[_Point2, _Point2]:
    if isinstance(use.pcurve, (ExactLineCurve, LineCurve)):
        first_exact = _point2(
            tuple(use.pcurve.evaluate_exact(Fraction.from_float(float(use.first))))
        )
        last_exact = _point2(
            tuple(use.pcurve.evaluate_exact(Fraction.from_float(float(use.last))))
        )
        points: tuple[_Point2, _Point2] = (
            (
                Fraction.from_float(float(first_exact[0])),
                Fraction.from_float(float(first_exact[1])),
            ),
            (
                Fraction.from_float(float(last_exact[0])),
                Fraction.from_float(float(last_exact[1])),
            ),
        )
        if any(
            represented != exact
            for point, exact_point in zip(points, (first_exact, last_exact), strict=True)
            for represented, exact in zip(point, exact_point, strict=True)
        ):
            raise ValueError(
                "Source p-curve endpoints are not exactly represented in binary64."
            )
    else:
        # One whole native turn returns to its start chart point exactly.
        last = (
            use.first
            if isinstance(use, PatchCurveUse) and use.closes_native_turn()
            else use.last
        )
        values = np.asarray(
            use.pcurve.evaluate(jnp.asarray((use.first, last), dtype=jnp.float64)),
            dtype=np.float64,
        )
        if values.shape != (2, 2) or np.any(~np.isfinite(values)):
            raise ValueError("Source p-curve endpoints must be finite UV points.")
        points = (
            (
                Fraction.from_float(float(values[0, 0])),
                Fraction.from_float(float(values[0, 1])),
            ),
            (
                Fraction.from_float(float(values[1, 0])),
                Fraction.from_float(float(values[1, 1])),
            ),
        )
    if use.sense < 0:
        points = (points[1], points[0])
    return points[0], points[1]


def _patch_chart_polygon(
    domain: MeshingDomain,
    patch: int,
    /,
    curved_chain: Callable[
        [PatchCurveUse | PatchPoleUse, _Point2, _Point2], tuple[_Point2, ...]
    ]
    | None = None,
    *,
    drop_enclosed_curved_loops: bool = False,
    trim_loops: tuple[tuple[PatchCurveUse | PatchPoleUse, ...], ...] | None = None,
) -> (
    tuple[
        tuple[_Point2, ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[_Point2, ...], ...],
    ]
    | None
):
    """Return the chord polygon of one patch trim in its exact chart.

    With ``drop_enclosed_curved_loops`` a curved loop whose chords enclose no
    chart area (a single full circle, two half arcs) is omitted: every point it
    bounds lies in the convex hull of its arcs, which the curved-trim enclosure
    boxes cover, so only cover construction may request it. ``trim_loops``
    replaces the authored trim by the same trim in another exact patch chart.
    """
    points: list[tuple[Fraction, Fraction]] = []
    segments: list[tuple[int, int]] = []
    loops: list[tuple[tuple[Fraction, Fraction], ...]] = []
    dropped = False
    for loop in domain.patches[patch].loops if trim_loops is None else trim_loops:
        if not all(isinstance(use, (PatchCurveUse, PatchPoleUse)) for use in loop):
            return None
        loop_points: list[tuple[Fraction, Fraction]] = []
        loop_start: tuple[Fraction, Fraction] | None = None
        previous_end: tuple[Fraction, Fraction] | None = None
        for use in loop:
            first, last = _line_endpoints(use)
            if loop_start is None:
                loop_start = first
            if previous_end is not None and previous_end != first:
                raise ValueError(
                    "Source trim loop is not an exact connected chart chain."
                )
            loop_points.append(first)
            if curved_chain is not None and not isinstance(
                use.pcurve, (ExactLineCurve, LineCurve)
            ):
                loop_points.extend(curved_chain(use, first, last))
            previous_end = last
        if not loop_points or loop_start is None or previous_end != loop_start:
            raise ValueError("Source trim loop does not close in its exact chart.")
        if (
            drop_enclosed_curved_loops
            and _signed_area2(tuple(loop_points)) == 0
            and any(
                not isinstance(use.pcurve, (ExactLineCurve, LineCurve)) for use in loop
            )
        ):
            dropped = True
            continue
        offset = len(points)
        segments.extend(
            (offset + index, offset + (index + 1) % len(loop_points))
            for index in range(len(loop_points))
        )
        points.extend(loop_points)
        loops.append(tuple(loop_points))
    if not loops and not dropped:
        return None
    return tuple(points), tuple(segments), tuple(loops)


def _exact_trim_triangles(
    uv_points: tuple[_Point2, ...],
    segments: tuple[tuple[int, int], ...],
    loops: tuple[tuple[_Point2, ...], ...],
    /,
) -> tuple[_Triangle2, ...]:
    """Triangulate one exact oriented chart trim polygon without new vertices."""
    if _signed_area2(loops[0]) <= 0 or any(
        _signed_area2(loop) >= 0 for loop in loops[1:]
    ):
        raise ValueError("Source trim loops do not have the required orientation.")
    hole_seeds = tuple(_hole_seed(loop) for loop in loops[1:])
    points = np.asarray(
        [[float(value) for value in point] for point in uv_points],
        dtype=np.float64,
    )
    segments_array = np.asarray(segments, dtype=np.int64)
    holes = (
        None
        if not hole_seeds
        else np.asarray(
            [[float(value) for value in point] for point in hole_seeds],
            dtype=np.float64,
        )
    )
    triangulation = ConstrainedDelaunayTriangulation(
        points,
        segments_array,
        holes=holes,
        keep_convex_hull=False,
    )
    triangulation_points = np.asarray(triangulation.points, dtype=np.float64)
    if triangulation_points.shape != points.shape or not np.array_equal(
        triangulation_points, points
    ):
        raise ValueError(
            "Exact source trim triangulation changed its parameter vertices."
        )
    triangle_rows = np.asarray(triangulation.triangles, dtype=np.int64)
    if triangle_rows.ndim != 2 or triangle_rows.shape[1] != 3:
        raise ValueError("Source trim triangulation must return triangle rows.")
    edge_counts: Counter[tuple[int, int]] = Counter()
    triangle_area = Fraction(0)
    exact_triangles: list[_Triangle2] = []
    for triangle_row in range(triangle_rows.shape[0]):
        indices = (
            int(triangle_rows[triangle_row, 0]),
            int(triangle_rows[triangle_row, 1]),
            int(triangle_rows[triangle_row, 2]),
        )
        values = (uv_points[indices[0]], uv_points[indices[1]], uv_points[indices[2]])
        area = _cross2(values[0], values[1], values[2])
        if area <= 0:
            raise ValueError("Source parameter triangles must be positively oriented.")
        triangle_area += area
        exact_triangles.append(values)
        for first, second in (
            (indices[0], indices[1]),
            (indices[1], indices[2]),
            (indices[2], indices[0]),
        ):
            edge_counts[(min(first, second), max(first, second))] += 1
    boundary = {edge for edge, count in edge_counts.items() if count == 1}
    expected = {
        (min(int(first), int(second)), max(int(first), int(second)))
        for first, second in segments_array
    }
    source_area = sum((_signed_area2(loop) for loop in loops), Fraction(0))
    if boundary != expected or triangle_area != source_area:
        raise ValueError(
            "Source parameter triangles do not cover the oriented trim chain."
        )
    return tuple(exact_triangles)


_TURN = 2.0 * math.pi


class ExactPlaneChart(NamedTuple):
    """Exact affine chart of a planar patch whose authored chart is not affine.

    The patch point of plane chart ``(s, t)`` is ``origin + s * first_axis +
    t * second_axis``. The chart has the authored chart's orientation, and its
    trim loops are the authored trims mapped exactly into it: a revolution of a
    profile line perpendicular to the axis whose radial offset is collinear
    with the line is the disk or annulus of its rim circles, and the authored
    seam and pole are interior to it.
    """

    chart_id: str
    origin: _Point3
    first_axis: _Point3
    second_axis: _Point3
    loops: tuple[tuple[PatchCurveUse, ...], ...]
    radius_offset: Fraction
    radius_sign: int

    def chart_point(self, parameters: _Point2, /) -> _Point2:
        """Binary64 plane chart point of one authored ``(angle, profile)`` point.

        The plane point is ``r (cos angle, -sign sin angle)`` with radius
        ``r = sign (offset + profile)``; the authored seam's turn end is the
        angle-zero side, and an angle-zero point is exact, as is a trim circle's
        start point.
        """
        angle, profile = parameters
        radius = self.radius_sign * (self.radius_offset + profile)
        if radius < 0:
            raise ValueError("Authored point lies outside the planar revolution chart.")
        turn_angle = 0.0 if float(angle) == _TURN else float(angle)
        scale = float(radius)
        return (
            Fraction.from_float(scale * math.cos(turn_angle)),
            Fraction.from_float(-self.radius_sign * scale * math.sin(turn_angle)),
        )

    def coordinate_expressions(self, /) -> tuple[Polynomial, ...]:
        """Exact affine coordinate polynomials of the plane chart."""
        expressions: list[Polynomial] = []
        for axis in range(3):
            expression: Polynomial = {
                (0, 0): self.origin[axis],
                (1, 0): self.first_axis[axis],
                (0, 1): self.second_axis[axis],
            }
            expressions.append(expression)
        return tuple(expressions)


def _exact_line_profile(
    curve: object, /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    """Rational base point and direction of an exact affine profile line."""
    if isinstance(curve, LineCurve):
        base, direction = tuple(curve.origin_exact), tuple(curve.direction_exact)
    elif isinstance(curve, ExactLineCurve):
        base = tuple(curve.first_exact)
        direction = tuple(
            last - first
            for first, last in zip(curve.first_exact, curve.last_exact, strict=True)
        )
    else:
        return None
    exact_base = tuple(value for value in base if isinstance(value, Fraction))
    exact_direction = tuple(value for value in direction if isinstance(value, Fraction))
    if len(base) != 3 or len(exact_base) != 3 or len(exact_direction) != 3:
        return None
    return exact_base, exact_direction


def _constant_line_coordinate(
    use: PatchCurveUse | PatchPoleUse, axis: int, /
) -> Fraction | None:
    """Exact chart coordinate ``axis`` of a straight authored use constant in it."""
    pcurve = use.pcurve
    if not isinstance(pcurve, (ExactLineCurve, LineCurve)):
        from .brep._boolean_core import PeriodicPCurve

        if not isinstance(pcurve, PeriodicPCurve) or not isinstance(
            pcurve.source_curve, (ExactLineCurve, LineCurve)
        ):
            return None
        values = np.asarray(use.traversal_endpoints(), dtype=np.float64)
        source_values = np.asarray(
            pcurve.source_curve.evaluate(jnp.asarray((use.first, use.last))),
            dtype=np.float64,
        )
        if source_values[0, axis] != source_values[1, axis]:
            return None
        if values[0, axis] != values[1, axis]:
            return None
        return Fraction.from_float(float(values[0, axis]))
    first, last = _line_endpoints(use)
    if first[axis] != last[axis]:
        return None
    return first[axis]


def exact_plane_chart(domain: MeshingDomain, patch: int, /) -> ExactPlaneChart | None:
    """Exact plane chart of a patch proved planar by its carrier, if any.

    A surface of revolution of an exact profile line ``P(v) = P0 + v d`` about
    the axis ``(O, a)`` evaluates ``C + rho(v) (d cos u + (a x d) sin u)`` with
    ``C = O + a ((P0 - O) . a)`` whenever ``d . a == 0`` and the radial offset
    ``(P0 - O) - a ((P0 - O) . a) == rho0 d`` hold exactly, where
    ``rho(v) = rho0 + v``. Its full-turn trim is then the exact disk or annulus
    of its rim circles in the chart ``first_axis = sign d``,
    ``second_axis = -(a x d)``, which preserves the authored orientation. Every
    other carrier, a partial turn, or any inexact premise returns ``None``.
    """
    from ._meshing_domain import _PlacedSurfacePatch
    from ._rational import rational_cross
    from .brep._patches import RevolutionSurface

    surface = domain.patches[patch].surface
    placement = None
    if isinstance(surface, _PlacedSurfacePatch):
        surface, placement = surface.source, surface.placement
    if not isinstance(surface, RevolutionSurface):
        return None
    profile = _exact_line_profile(surface.curve)
    if profile is None:
        return None
    base, direction = profile
    axis_origin = rational_vector(np.asarray(surface.axis_origin, dtype=np.float64))
    axis = rational_vector(np.asarray(surface.axis_direction, dtype=np.float64))

    def dot(first: tuple[Fraction, ...], second: tuple[Fraction, ...], /) -> Fraction:
        return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))

    if dot(direction, axis) != 0:
        return None
    relative = tuple(
        point - origin for point, origin in zip(base, axis_origin, strict=True)
    )
    along = dot(relative, axis)
    radial = tuple(
        value - along * unit for value, unit in zip(relative, axis, strict=True)
    )
    if any(value != 0 for value in rational_cross(radial, direction)):
        return None
    radius_offset = dot(radial, direction) / dot(direction, direction)
    loops = domain.patches[patch].loops
    if len(loops) != 1 or len(loops[0]) != 4:
        return None
    rims: list[tuple[PatchCurveUse, Fraction]] = []
    seams: dict[int, list[Fraction]] = {}
    for use in loops[0]:
        if isinstance(use, PatchPoleUse):
            profile_value = _constant_line_coordinate(use, 1)
            if profile_value is None or radius_offset + profile_value != 0:
                return None
            continue
        angle = _constant_line_coordinate(use, 0)
        if angle is not None:
            seams.setdefault(use.curve, []).append(angle)
            continue
        profile_value = _constant_line_coordinate(use, 1)
        if profile_value is None:
            return None
        first, last = _line_endpoints(use)
        if {first[0], last[0]} != {Fraction(0), Fraction.from_float(_TURN)}:
            return None
        rims.append((use, profile_value))
    if (
        not rims
        or len(seams) != 1
        or sorted(next(iter(seams.values()))) != [Fraction(0), Fraction.from_float(_TURN)]
    ):
        return None
    offsets = {radius_offset + value for _, value in rims}
    if 0 in offsets or len({value > 0 for value in offsets}) != 1:
        return None
    sign = 1 if next(iter(offsets)) > 0 else -1
    ordered = sorted(rims, key=lambda item: -sign * (radius_offset + item[1]))
    plane_loops: list[tuple[PatchCurveUse, ...]] = []
    for row, (use, profile_value) in enumerate(ordered):
        radius = sign * (radius_offset + profile_value)
        circle = CircleCurve(
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), float(radius), radius_exact=radius
        )
        plane_loops.append(
            (
                PatchCurveUse(
                    use.curve, circle, 0.0, circle.period, sense=1 if row == 0 else -1
                ),
            )
        )
    first_axis = tuple(sign * value for value in direction)
    second_axis = tuple(-value for value in rational_cross(axis, direction))
    origin = tuple(
        point + along * unit for point, unit in zip(axis_origin, axis, strict=True)
    )
    if placement is not None:
        matrix = placement.matrix

        def place(
            vector: tuple[Fraction, ...], translate: bool, /
        ) -> tuple[Fraction, ...]:
            return tuple(
                sum(
                    (matrix[row][column] * vector[column] for column in range(3)),
                    Fraction(0),
                )
                + (matrix[row][3] if translate else Fraction(0))
                for row in range(3)
            )

        origin = place(origin, True)
        first_axis = place(first_axis, False)
        second_axis = place(second_axis, False)
    return ExactPlaneChart(
        f"{domain.entity_id(2, patch)}#plane",
        _point3(origin),
        _point3(first_axis),
        _point3(second_axis),
        tuple(plane_loops),
        radius_offset,
        sign,
    )


class CurvedTrimLens(NamedTuple):
    """Chart region between one target chord of a curved trim and its source arc.

    Every chart point enclosed by the chord, the arc, and the connections from the
    chord ends to their located arc points lies in the parameter box
    ``lower``–``upper`` and within ``parameter_bound`` of both the chord and the
    arc. Each straight segment realizing those distances stays inside the box.
    """

    chord: tuple[_Point2, _Point2]
    lower: _Point2
    upper: _Point2
    parameter_bound: Fraction


_CHAIN_PARAMETER_REFINEMENTS = 14
_POLE_SLIVER_PIECE_LIMIT = 256


def _target_boundary_successors(
    triangles: tuple[_Triangle2, ...], /
) -> dict[_Point2, list[_Point2]]:
    """Directed target boundary edges with the target interior on their left."""
    counts: Counter[frozenset[_Point2]] = Counter()
    oriented: list[_Triangle2] = []
    for triangle in triangles:
        area = _cross2(*triangle)
        if area == 0:
            raise ValueError("Target chart triangles must have nonzero chart area.")
        oriented.append(triangle if area > 0 else (triangle[0], triangle[2], triangle[1]))
        for index in range(3):
            counts[frozenset((triangle[index], triangle[(index + 1) % 3]))] += 1
    successors: dict[_Point2, list[_Point2]] = {}
    for triangle in oriented:
        for index in range(3):
            start, stop = triangle[index], triangle[(index + 1) % 3]
            if counts[frozenset((start, stop))] == 1:
                successors.setdefault(start, []).append(stop)
    return successors


def _target_boundary_chain(
    successors: dict[_Point2, list[_Point2]],
    first: _Point2,
    last: _Point2,
    /,
) -> tuple[_Point2, ...]:
    """Walk directed target boundary edges from ``first`` to ``last``.

    A self-closing trim (one full turn, ``first == last``) walks its whole loop.
    """
    chain = [first]
    while len(chain) == 1 or chain[-1] != last:
        following = successors.get(chain[-1], ())
        if len(following) != 1 or len(chain) > len(successors):
            raise ValueError(
                "Target charts have no unique boundary chain along a curved source trim."
            )
        chain.append(following[0])
    return tuple(chain)


def _use_parameter_range(use: PatchCurveUse | PatchPoleUse, /) -> tuple[float, float]:
    """Return the p-curve parameters at the chart start and end of one loop use."""
    if use.sense < 0:
        return float(use.last), float(use.first)
    return float(use.first), float(use.last)


def _chain_parameters(
    use: PatchCurveUse | PatchPoleUse, chain: tuple[_Point2, ...], /
) -> tuple[float, ...]:
    """Locate interior chain points on the p-curve by nested nearest-sample search.

    The located parameters only select which arc piece each chord is compared
    with; every claim drawn from them is certified by p-curve enclosures.
    """
    start, stop = _use_parameter_range(use)
    if len(chain) == 2:
        return start, stop
    targets = np.asarray(
        [[float(value) for value in point] for point in chain[1:-1]],
        dtype=np.float64,
    )
    low, high = min(start, stop), max(start, stop)

    def evaluate(parameters: np.ndarray, /) -> np.ndarray:
        values = np.asarray(
            use.pcurve.evaluate(jnp.asarray(parameters.reshape(-1), dtype=jnp.float64)),
            dtype=np.float64,
        )
        if values.shape != (parameters.size, 2) or np.any(~np.isfinite(values)):
            raise ValueError("Curved trim p-curve evaluation must remain finite.")
        return values.reshape((*parameters.shape, 2))

    count = 8 * len(chain)
    grid = np.linspace(start, stop, count + 1, dtype=np.float64)
    distance = np.sum((evaluate(grid)[None] - targets[:, None]) ** 2, axis=-1)
    parameters = grid[np.argmin(distance, axis=1)]
    width = (high - low) / count
    offsets = np.linspace(-1.0, 1.0, 17, dtype=np.float64)
    rows = np.arange(targets.shape[0])
    for _ in range(_CHAIN_PARAMETER_REFINEMENTS):
        candidates = np.clip(parameters[:, None] + width * offsets[None], low, high)
        distance = np.sum((evaluate(candidates) - targets[:, None]) ** 2, axis=-1)
        parameters = candidates[rows, np.argmin(distance, axis=1)]
        width /= 8.0
    return start, *(float(value) for value in parameters), stop


def _chart_box_corners(bounds: ArrayLike, /) -> tuple[_Point2, ...]:
    box = np.asarray(bounds, dtype=np.float64)
    if box.shape != (2, 2) or np.any(~np.isfinite(box)) or np.any(box[0] > box[1]):
        raise ValueError("Curved trim p-curve has no finite ordered chart enclosure.")
    lower = (Fraction.from_float(float(box[0, 0])), Fraction.from_float(float(box[0, 1])))
    upper = (Fraction.from_float(float(box[1, 0])), Fraction.from_float(float(box[1, 1])))
    return (
        lower,
        (upper[0], lower[1]),
        upper,
        (lower[0], upper[1]),
    )


def _curved_trim_lens(
    use: PatchCurveUse | PatchPoleUse,
    first: _Point2,
    second: _Point2,
    first_parameter: float,
    second_parameter: float,
    /,
) -> CurvedTrimLens:
    """Certify the lens between one target chord and the arc between its ends.

    With ``n`` normal and ``d`` parallel to the chord, every arc point has a
    certified normal offset (from the arc box, or from the end boxes plus the
    derivative range of ``n·c'``) and a certified overshoot past the chord ends.
    The closed curve formed by chord, connections and arc therefore stays in the
    convex tube of radius ``r`` around the chord, and so does every point it
    encloses. A line normal to the chord through an enclosed point meets the
    chord at most once, so it meets the arc or a connection within ``2 r``.
    """
    low, high = (
        min(first_parameter, second_parameter),
        max(first_parameter, second_parameter),
    )
    if not low < high:
        raise ValueError(
            "Target chart points do not advance along their curved source trim."
        )
    pcurve = use.pcurve
    arc = _chart_box_corners(pcurve.bounding_box(low, high))
    ends = (
        (
            first,
            _chart_box_corners(pcurve.bounding_box(first_parameter, first_parameter)),
        ),
        (
            second,
            _chart_box_corners(pcurve.bounding_box(second_parameter, second_parameter)),
        ),
    )
    direction = (second[0] - first[0], second[1] - first[1])
    length_squared = direction[0] ** 2 + direction[1] ** 2
    if length_squared == 0:
        raise ValueError("A target chord of a curved source trim is degenerate.")

    def across(point: _Point2, /) -> Fraction:
        return (point[1] - first[1]) * direction[0] - (point[0] - first[0]) * direction[1]

    def along(point: _Point2, /) -> Fraction:
        return (point[0] - first[0]) * direction[0] + (point[1] - first[1]) * direction[1]

    across_bound = max(abs(across(corner)) for corner in arc)
    try:
        rate_lower, rate_upper = pcurve.derivative_bounds(low, high)
    except (ArithmeticError, NotImplementedError, TypeError, ValueError):
        pass
    else:
        rates = np.stack(
            (
                np.asarray(rate_lower, dtype=np.float64),
                np.asarray(rate_upper, dtype=np.float64),
            )
        )
        if rates.shape == (2, 2) and np.all(np.isfinite(rates)):
            normal = (-direction[1], direction[0])
            rate_terms = tuple(
                sorted(
                    normal[axis] * Fraction.from_float(float(rates[row, axis]))
                    for row in range(2)
                )
                for axis in range(2)
            )
            rate_width = sum((terms[1] - terms[0] for terms in rate_terms), Fraction(0))
            end_across = max(
                abs(across(corner)) for _, corners in ends for corner in corners
            )
            span = Fraction.from_float(high) - Fraction.from_float(low)
            across_bound = min(across_bound, end_across + span * rate_width / 4)
    overshoot = max(
        Fraction(0),
        *(-along(corner) for corner in arc),
        *(along(corner) - length_squared for corner in arc),
    )
    length_lower = fraction_sqrt_interval(length_squared)[0]
    radius = fraction_sqrt_interval(across_bound**2 + overshoot**2)[1] / length_lower
    gap = max(
        fraction_sqrt_interval((corner[0] - point[0]) ** 2 + (corner[1] - point[1]) ** 2)[
            1
        ]
        for point, corners in ends
        for corner in corners
    )
    return CurvedTrimLens(
        (first, second),
        (min(first[0], second[0]) - radius, min(first[1], second[1]) - radius),
        (max(first[0], second[0]) + radius, max(first[1], second[1]) + radius),
        2 * radius + gap,
    )


def _trim_enclosure_triangles(
    domain: MeshingDomain,
    patch: int,
    /,
) -> tuple[_Triangle2, ...]:
    """Triangulate continuous chart boxes enclosing curved trim-to-chord gaps."""
    triangles: list[_Triangle2] = []
    for loop in domain.patches[patch].loops:
        for use in loop:
            if isinstance(use.pcurve, (ExactLineCurve, LineCurve)):
                continue
            bounds = np.asarray(
                use.pcurve.bounding_box(
                    min(use.first, use.last), max(use.first, use.last)
                ),
                dtype=np.float64,
            )
            if (
                bounds.shape != (2, 2)
                or np.any(~np.isfinite(bounds))
                or np.any(bounds[0] > bounds[1])
            ):
                raise ValueError(
                    "Curved trim p-curve has no finite ordered interval enclosure."
                )
            lower = tuple(Fraction.from_float(float(value)) for value in bounds[0])
            upper = tuple(Fraction.from_float(float(value)) for value in bounds[1])
            if lower[0] == upper[0] or lower[1] == upper[1]:
                continue
            first = (lower[0], lower[1])
            second = (upper[0], lower[1])
            third = (upper[0], upper[1])
            fourth = (lower[0], upper[1])
            triangles.extend(((first, second, third), (first, third, fourth)))
    return tuple(triangles)


def _point_segment_squared(
    point: tuple[Fraction, ...],
    first: tuple[Fraction, ...],
    second: tuple[Fraction, ...],
    /,
) -> Fraction:
    direction = tuple(b - a for a, b in zip(first, second, strict=True))
    length_squared = sum((value * value for value in direction), Fraction(0))
    if not length_squared:
        return sum(
            ((value - a) ** 2 for value, a in zip(point, first, strict=True)), Fraction(0)
        )
    parameter = (
        sum(
            (
                (value - a) * delta
                for value, a, delta in zip(point, first, direction, strict=True)
            ),
            Fraction(0),
        )
        / length_squared
    )
    parameter = min(max(parameter, Fraction(0)), Fraction(1))
    return sum(
        (
            (value - a - parameter * delta) ** 2
            for value, a, delta in zip(point, first, direction, strict=True)
        ),
        Fraction(0),
    )


def _point_triangle_squared(
    point: _Point3,
    triangle: _Triangle3,
    /,
) -> Fraction:
    origin = triangle[0]
    first = (
        triangle[1][0] - origin[0],
        triangle[1][1] - origin[1],
        triangle[1][2] - origin[2],
    )
    second = (
        triangle[2][0] - origin[0],
        triangle[2][1] - origin[1],
        triangle[2][2] - origin[2],
    )
    relative = (
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    )
    gram = (
        (
            sum((a * a for a in first), Fraction(0)),
            sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0)),
        ),
        (
            sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0)),
            sum((a * a for a in second), Fraction(0)),
        ),
    )
    right = (
        (sum((a * b for a, b in zip(first, relative, strict=True)), Fraction(0)),),
        (sum((a * b for a, b in zip(second, relative, strict=True)), Fraction(0)),),
    )
    determinant = determinant_exact(gram)
    if determinant:
        from ..linalg._exact import solve_exact

        solution = solve_exact(gram, right)
        u, v = solution[0][0], solution[1][0]
        if u >= 0 and v >= 0 and u + v <= 1:
            residual = tuple(
                relative[axis] - u * first[axis] - v * second[axis] for axis in range(3)
            )
            return sum((value * value for value in residual), Fraction(0))
    return min(
        _point_segment_squared(point, triangle[index], triangle[(index + 1) % 3])
        for index in range(3)
    )


def _cover_chord_triangles(
    patch: int,
    uv_points: tuple[_Point2, ...],
    segments: tuple[tuple[int, int], ...],
    loops: tuple[tuple[_Point2, ...], ...],
    /,
) -> tuple[_Triangle2, ...] | MeshCertificateFinding:
    """Triangulate the chord polygon of one cover patch or report why it cannot.

    A trim whose curved loops all enclose no chord area has no chord polygon; its
    curved-trim enclosure boxes are then the whole cover.
    """
    if not loops:
        return ()
    outer_sign = 1 if _signed_area2(loops[0]) > 0 else -1
    if outer_sign < 0 or any(_signed_area2(loop) >= 0 for loop in loops[1:]):
        return MeshCertificateFinding(
            "source_trim_orientation",
            "violated",
            "source_facet",
            (patch,),
            detail="Source trim loops must be outer counterclockwise and holes clockwise.",
        )
    hole_seeds = tuple(_hole_seed(loop) for loop in loops[1:])
    points = np.asarray(
        [[float(value) for value in point] for point in uv_points],
        dtype=np.float64,
    )
    segments_array = np.asarray(segments, dtype=np.int64)
    holes = (
        None
        if not hole_seeds
        else np.asarray(
            [[float(value) for value in point] for point in hole_seeds],
            dtype=np.float64,
        )
    )
    triangulation = ConstrainedDelaunayTriangulation(
        points,
        segments_array,
        holes=holes,
        keep_convex_hull=False,
    )
    triangulation_points = np.asarray(triangulation.points, dtype=np.float64)
    if triangulation_points.shape != points.shape or not np.array_equal(
        triangulation_points, points
    ):
        return MeshCertificateFinding(
            "source_trim_triangulation",
            "unresolved",
            "source_facet",
            (patch,),
            detail="Triangulation introduced or moved a chart vertex outside the exact trim vertex table.",
        )
    triangle_rows = np.asarray(triangulation.triangles, dtype=np.int64)
    if triangle_rows.ndim != 2 or triangle_rows.shape[1] != 3:
        return MeshCertificateFinding(
            "source_trim_triangulation",
            "unresolved",
            "source_facet",
            (patch,),
            detail="Constrained source triangulation did not return triangular rows.",
        )
    edge_counts: Counter[tuple[int, int]] = Counter()
    for triangle_row in range(triangle_rows.shape[0]):
        indices = (
            int(triangle_rows[triangle_row, 0]),
            int(triangle_rows[triangle_row, 1]),
            int(triangle_rows[triangle_row, 2]),
        )
        for first, second in (
            (indices[0], indices[1]),
            (indices[1], indices[2]),
            (indices[2], indices[0]),
        ):
            edge_counts[(min(first, second), max(first, second))] += 1
    boundary = {edge for edge, count in edge_counts.items() if count == 1}
    expected = {
        (min(int(first), int(second)), max(int(first), int(second)))
        for first, second in segments_array
    }
    triangle_area = Fraction(0)
    for triangle_row in range(triangle_rows.shape[0]):
        values = (
            uv_points[int(triangle_rows[triangle_row, 0])],
            uv_points[int(triangle_rows[triangle_row, 1])],
            uv_points[int(triangle_rows[triangle_row, 2])],
        )
        area = _cross2(values[0], values[1], values[2])
        if area <= 0:
            raise ValueError(
                "Constrained source chart triangles must have positive orientation."
            )
        triangle_area += area
    source_area = sum((_signed_area2(loop) for loop in loops), Fraction(0))
    if boundary != expected or triangle_area != source_area:
        return MeshCertificateFinding(
            "source_trim_coverage",
            "unresolved",
            "source_facet",
            (patch,),
            detail="Constrained triangulation did not reproduce the complete oriented trim chain and measure.",
        )
    return tuple(
        (
            uv_points[int(triangle_rows[row, 0])],
            uv_points[int(triangle_rows[row, 1])],
            uv_points[int(triangle_rows[row, 2])],
        )
        for row in range(triangle_rows.shape[0])
    )


@final
class MeshingDomainBoundarySource(StrictModule, NonTrainableState):
    """Bounded boundary queries over selected source patches and trims."""

    domain: MeshingDomain
    patches: tuple[int, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_scope_id: str = eqx.field(static=True)
    resolution: int = eqx.field(static=True)
    boundary_cell_ids: np.ndarray | None
    boundary_patch_ids: np.ndarray | None
    boundary_occurrences: np.ndarray | None

    @checked
    def __init__(
        self,
        domain: MeshingDomain,
        patches: tuple[int, ...],
        /,
        *,
        resolution: int = 48,
        boundary_occurrences: ArrayLike | None = None,
        boundary_cell_ids: ArrayLike | None = None,
        boundary_patch_ids: ArrayLike | None = None,
    ) -> None:
        selected = tuple(sorted(set(int(value) for value in patches)))
        if not selected or any(
            value < 0 or value >= len(domain.patches) for value in selected
        ):
            raise ValueError(
                "MeshingDomainBoundarySource.patches must select existing source patches."
            )
        resolution_ = positive_integer(resolution, "resolution")
        supplied = (
            boundary_occurrences is not None,
            boundary_cell_ids is not None,
            boundary_patch_ids is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError(
                "Boundary occurrence ownership requires all three aligned arrays."
            )
        occurrences = cells = owners = None
        if all(supplied):
            cells = np.asarray(boundary_cell_ids, dtype=np.int64)
            owners = np.asarray(boundary_patch_ids, dtype=np.int32)
            occurrences = np.asarray(boundary_occurrences, dtype=np.int32)
            if (
                cells.ndim != 1
                or owners.shape != cells.shape
                or occurrences.shape != (cells.size, 3, 3)
                or np.any(cells < 0)
                or np.unique(cells).size != cells.size
                or not np.all(np.isin(owners, selected))
                or np.any(occurrences < 0)
            ):
                raise ValueError(
                    "Boundary occurrence arrays must be aligned and source-owned."
                )
            cells = np.array(cells, copy=True)
            owners = np.array(owners, copy=True)
            occurrences = np.array(occurrences, copy=True)
            for array in (cells, owners, occurrences):
                array.setflags(write=False)
        domain.require_current()
        self.domain = domain
        self.patches = selected
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        selected_entities = []
        for entity_id in domain.scope_entity_ids(2):
            stable_id = int(entity_id)
            entity_key, occurrence_path = domain.source_entity_key_for_scope(2, stable_id)
            row = domain.resolve_entity_id(2, entity_key, occurrence_path=occurrence_path)
            if row in selected:
                selected_entities.append((stable_id, entity_key, occurrence_path))
        self.source_scope_id = canonical_fingerprint(
            {
                "kind": "meshing-domain-boundary-scope",
                "domain": domain.domain_id,
                "entity_set": domain.scope_entity_set_id(2),
                "selected_entities": tuple(sorted(selected_entities)),
            }
        )
        self.resolution = resolution_
        self.boundary_occurrences = occurrences
        self.boundary_cell_ids = cells
        self.boundary_patch_ids = owners

    @property
    def ambient_dimension(self) -> int:
        return 3

    def require_current(self) -> None:
        self.domain.require_current(self.source_id, self.source_revision)

    def boundary_parameter_triangles(
        self, /
    ) -> tuple[tuple[int, tuple[_Triangle2, ...]], ...]:
        """Return the exact parameter triangles of the complete source trim cover."""
        self.require_current()
        result: list[tuple[int, tuple[_Triangle2, ...]]] = []
        for patch in self.patches:
            chart = _patch_chart_polygon(self.domain, patch)
            if chart is None:
                raise ValueError(
                    "Source trim has no supported exact parameter triangulation."
                )
            result.append((patch, _exact_trim_triangles(*chart)))
        return tuple(result)

    def trim_partition_reference(
        self, patch: int, triangles: tuple[_Triangle2, ...], /, *, plane: bool = False
    ) -> tuple[tuple[_Triangle2, ...], tuple[CurvedTrimLens, ...]]:
        """Return the exact reference partition of one trim for target chart triangles.

        Straight trims are exact chart segments. Along a curved trim the reference
        follows the target's boundary chain between the trim's chart endpoints once
        every chain point is located in order on the source arc; each chord then
        carries the certified lens separating it from the exact curved trim. With
        ``plane`` the triangles and trim are in the patch's `exact_plane_chart`.
        """
        self.require_current()
        return self._trim_partition_reference_unchecked(patch, triangles, plane=plane)

    def _trim_partition_reference_unchecked(
        self, patch: int, triangles: tuple[_Triangle2, ...], /, *, plane: bool = False
    ) -> tuple[tuple[_Triangle2, ...], tuple[CurvedTrimLens, ...]]:
        # Per-patch callers validate currentness once before their loop: the
        # check fingerprints the whole domain, so it must not run per patch.
        successors = _target_boundary_successors(triangles)
        lenses: list[CurvedTrimLens] = []

        def curved_chain(
            use: PatchCurveUse | PatchPoleUse, first: _Point2, last: _Point2, /
        ) -> tuple[_Point2, ...]:
            chain = _target_boundary_chain(successors, first, last)
            parameters = _chain_parameters(use, chain)
            lenses.extend(
                _curved_trim_lens(
                    use, chain[row], chain[row + 1], parameters[row], parameters[row + 1]
                )
                for row in range(len(chain) - 1)
            )
            return chain[1:-1]

        trim_loops = None
        if plane:
            plane_chart = exact_plane_chart(self.domain, patch)
            if plane_chart is None:
                raise ValueError("Source patch has no exact plane chart.")
            trim_loops = plane_chart.loops
        chart = _patch_chart_polygon(
            self.domain, patch, curved_chain, trim_loops=trim_loops
        )
        if chart is None:
            raise ValueError(
                "Source trim has no supported exact parameter triangulation."
            )
        return _exact_trim_triangles(*chart), tuple(lenses)

    def collapsed_pole_slivers(
        self, patch: int, triangles: tuple[_Triangle2, ...], /
    ) -> tuple[tuple[int, _Triangle2], ...]:
        """Chart slivers that a target pole fan collapses onto one source corner.

        A collapsed pole trim side maps to one physical corner. Between consecutive
        chart points ``p, q`` of that side that share no target triangle, a pole fan
        leaves the chart triangle ``(p, q, w)`` whose image degenerates to the mesh
        edge from the pole to the fan apex ``w``. Its two other sides are target
        boundary edges or source trim sides. Each sliver is returned with the target
        row holding a pole-side edge to ``w``: that cell's chart preimage is its
        affine triangle together with the sliver.
        """
        chart = _patch_chart_polygon(self.domain, patch)
        if chart is None:
            return ()
        _, _, loops = chart
        trim_sides = tuple(
            (first, last)
            for loop in loops
            for first, last in zip(loop, (*loop[1:], loop[0]), strict=True)
        )
        edge_rows: dict[frozenset[_Point2], list[int]] = {}
        neighbors: dict[_Point2, set[_Point2]] = {}
        for row, triangle in enumerate(triangles):
            for first, second in ((0, 1), (1, 2), (2, 0)):
                start, stop = triangle[first], triangle[second]
                edge_rows.setdefault(frozenset((start, stop)), []).append(row)
                neighbors.setdefault(start, set()).add(stop)
                neighbors.setdefault(stop, set()).add(start)

        def on_segment(point: _Point2, start: _Point2, stop: _Point2, /) -> bool:
            direction = (stop[0] - start[0], stop[1] - start[1])
            offset = (point[0] - start[0], point[1] - start[1])
            along = offset[0] * direction[0] + offset[1] * direction[1]
            return (
                offset[0] * direction[1] == offset[1] * direction[0]
                and 0 <= along <= direction[0] ** 2 + direction[1] ** 2
            )

        def sliver_side(start: _Point2, stop: _Point2, /) -> bool:
            rows = edge_rows.get(frozenset((start, stop)))
            if rows is not None:
                return len(rows) == 1
            return any(
                on_segment(start, first, last) and on_segment(stop, first, last)
                for first, last in trim_sides
            )

        orientation = int(self.domain.patches[patch].orientation)
        slivers: list[tuple[int, _Triangle2]] = []
        for loop in self.domain.patches[patch].loops:
            for use in loop:
                if not isinstance(use, PatchPoleUse) or not isinstance(
                    use.pcurve, (ExactLineCurve, LineCurve)
                ):
                    continue
                start, stop = _line_endpoints(use)
                direction = (stop[0] - start[0], stop[1] - start[1])
                side = sorted(
                    {
                        point
                        for point in (start, stop, *neighbors)
                        if on_segment(point, start, stop)
                    },
                    key=lambda point: (
                        (point[0] - start[0]) * direction[0]
                        + (point[1] - start[1]) * direction[1]
                    ),
                )
                side_points = set(side)
                for first, second in zip(side[:-1], side[1:], strict=True):
                    if frozenset((first, second)) in edge_rows:
                        continue
                    apexes = [
                        apex
                        for apex in neighbors.get(first, set())
                        | neighbors.get(second, set())
                        if apex not in side_points
                        and sliver_side(first, apex)
                        and sliver_side(second, apex)
                    ]
                    if len(apexes) != 1:
                        continue
                    apex = apexes[0]
                    owners = edge_rows.get(frozenset((first, apex))) or edge_rows.get(
                        frozenset((second, apex))
                    )
                    area = _signed_area2((first, second, apex))
                    if not owners or not area:
                        continue
                    sliver = (
                        (first, second, apex)
                        if orientation * area > 0
                        else (second, first, apex)
                    )
                    slivers.append((owners[0], sliver))
        return tuple(slivers)

    def pole_sliver_cap_bound(self, patch: int, sliver: _Triangle2, /) -> Fraction:
        """Upper bound on the source-image diameter of one collapsed pole sliver.

        The sliver contains its owner's pole chart corner, so every source point of
        the sliver lies within this bound of that corner's exact source point. The
        chart box is enclosed by pieces no wider than its short side, because a
        source box enclosure widens with its longest chart extent.
        """
        from ._meshing_domain import _outward_surface_parameter_box
        from ._source_interpolation import _fraction_float_interval

        surface = self.domain.patches[patch].surface
        lower = tuple(
            _fraction_float_interval(min(point[axis] for point in sliver))[0]
            for axis in range(2)
        )
        upper = tuple(
            _fraction_float_interval(max(point[axis] for point in sliver))[1]
            for axis in range(2)
        )
        extents = (upper[0] - lower[0], upper[1] - lower[1])
        long_axis = 0 if extents[0] >= extents[1] else 1
        short = extents[1 - long_axis]
        pieces = (
            _POLE_SLIVER_PIECE_LIMIT
            if short <= 0.0
            else min(
                _POLE_SLIVER_PIECE_LIMIT,
                max(1, math.ceil(extents[long_axis] / short)),
            )
        )
        cuts = np.linspace(lower[long_axis], upper[long_axis], pieces + 1)
        cuts[0], cuts[-1] = lower[long_axis], upper[long_axis]
        bounds = np.empty((2, 3), dtype=np.float64)
        bounds[0], bounds[1] = np.inf, -np.inf
        for first, last in zip(cuts[:-1], cuts[1:], strict=True):
            piece_lower, piece_upper = list(lower), list(upper)
            piece_lower[long_axis], piece_upper[long_axis] = float(first), float(last)
            piece = np.asarray(
                surface.bounding_box(
                    _outward_surface_parameter_box(surface, piece_lower, piece_upper)
                ),
                dtype=np.float64,
            )
            if (
                piece.shape != (2, 3)
                or np.any(~np.isfinite(piece))
                or np.any(piece[0] > piece[1])
            ):
                raise ValueError(
                    "Collapsed pole sliver has no finite ordered source box."
                )
            bounds[0] = np.minimum(bounds[0], piece[0])
            bounds[1] = np.maximum(bounds[1], piece[1])
        widths = tuple(
            Fraction(float(bounds[1, axis])) - Fraction(float(bounds[0, axis]))
            for axis in range(3)
        )
        return fraction_sqrt_interval(
            sum((width * width for width in widths), Fraction(0))
        )[1]

    def boundary_chart_cover(self, /) -> SourceBoundaryChartCover:
        images: list[np.ndarray] = []
        deviations: list[np.ndarray] = []
        findings: list[MeshCertificateFinding] = []
        restrictions: list[str] = []
        complete = True
        from ._source_interpolation import (
            _exact_rational_span_interpolation_bounds,
            _exact_surface_interpolation_bound,
            _has_unexpanded_exact_surface_carrier,
            exact_surface_coordinate_expressions,
        )
        from .brep._root_bindings import _source_coefficients_are_binary

        for patch in self.patches:
            surface = self.domain.patches[patch].surface
            try:
                exact_surface = exact_surface_coordinate_expressions(surface)
            except (ArithmeticError, FloatingPointError, TypeError, ValueError):
                exact_surface = None
                exact_surface_failed = True
            else:
                exact_surface_failed = False

            exact_span_source = False
            if exact_surface is None and not exact_surface_failed:
                try:
                    exact_span_source = _has_exact_rational_surface_span_source(surface)
                except (ArithmeticError, FloatingPointError, TypeError, ValueError):
                    exact_span_source = False
            if exact_surface is None and (
                exact_surface_failed
                or (
                    not exact_span_source
                    and (
                        not _source_coefficients_are_binary(surface)
                        or _has_unexpanded_exact_surface_carrier(surface)
                    )
                )
            ):
                complete = False
                findings.append(
                    MeshCertificateFinding(
                        "source_surface_coefficients",
                        "unresolved",
                        "source_facet",
                        (patch,),
                        detail="Exact surface coefficients require a registered source-bound continuous map.",
                    )
                )
                continue
            chart = _patch_chart_polygon(
                self.domain, patch, drop_enclosed_curved_loops=True
            )
            if chart is None:
                complete = False
                findings.append(
                    MeshCertificateFinding(
                        "source_trim_coverage",
                        "unresolved",
                        "source_facet",
                        (patch,),
                        detail="Nonlinear or collapsed trim chart requires a source-specific continuous cover.",
                    )
                )
                continue
            uv_points = chart[0]
            chord_charts = _cover_chord_triangles(patch, *chart)
            if isinstance(chord_charts, MeshCertificateFinding):
                complete = False
                findings.append(chord_charts)
                continue
            exact_charts = chord_charts + _trim_enclosure_triangles(self.domain, patch)
            uv_triangles = np.asarray(
                [
                    [[float(value) for value in point] for point in triangle]
                    for triangle in exact_charts
                ],
                dtype=np.float64,
            )
            patch_rows = np.full((len(exact_charts) * 3,), patch, dtype=np.int64)
            physical = self.domain.evaluate(
                patch_rows, uv_triangles.reshape((-1, 2))
            ).reshape((-1, 3, 3))
            if exact_surface is None:
                try:
                    if exact_span_source:
                        interpolation = _exact_rational_span_interpolation_bounds(
                            surface, exact_charts
                        )
                    else:
                        interpolation = self.domain.interpolation_bounds(
                            patch, uv_triangles
                        )
                    if exact_span_source:
                        if interpolation.shape != (len(exact_charts),) or not np.all(
                            np.isfinite(interpolation)
                        ):
                            raise NotImplementedError(
                                "Exact rational spans did not provide finite interpolation bounds."
                            )
                        for row, exact_chart in enumerate(exact_charts):
                            endpoint_error = max(
                                _exact_span_point_gap_upper(
                                    surface, exact_point, physical[row, corner]
                                )
                                for corner, exact_point in enumerate(exact_chart)
                            )
                            interpolation[row] = fraction_endpoint(
                                Fraction.from_float(float(interpolation[row]))
                                + endpoint_error,
                                upper=True,
                                saturate=True,
                            )
                except (
                    ArithmeticError,
                    FloatingPointError,
                    NotImplementedError,
                    TypeError,
                    ValueError,
                ) as error:
                    complete = False
                    findings.append(
                        MeshCertificateFinding(
                            "source_surface_interpolation",
                            "unresolved",
                            "source_facet",
                            (patch,),
                            detail=f"Exact source chart interpolation is unavailable: {error}",
                        )
                    )
                    continue
            else:
                from ..discretization._coordinate_enclosure import evaluate_expression

                interpolation = np.empty((len(exact_charts),), dtype=np.float64)
                try:
                    for row, exact_chart in enumerate(exact_charts):
                        exact_corners = tuple(
                            tuple(
                                evaluate_expression(expression, point)
                                for expression in exact_surface
                            )
                            for point in exact_chart
                        )
                        endpoint_error = Fraction(0)
                        for exact_corner, realized_corner in zip(
                            exact_corners, physical[row], strict=True
                        ):
                            component_errors = tuple(
                                coefficient_rounding_error(
                                    exact_value, float(realized_value)
                                )
                                for exact_value, realized_value in zip(
                                    exact_corner, realized_corner, strict=True
                                )
                            )
                            endpoint_error = max(
                                endpoint_error,
                                fraction_sqrt_interval(
                                    sum(
                                        (value * value for value in component_errors),
                                        Fraction(0),
                                    )
                                )[1],
                            )
                        interpolation_bound = _exact_surface_interpolation_bound(
                            exact_surface, exact_chart
                        )
                        interpolation[row] = (
                            math.inf
                            if not math.isfinite(interpolation_bound)
                            else fraction_endpoint(
                                Fraction.from_float(interpolation_bound) + endpoint_error,
                                upper=True,
                                saturate=True,
                            )
                        )
                except (
                    ArithmeticError,
                    FloatingPointError,
                    NotImplementedError,
                    TypeError,
                    ValueError,
                ) as error:
                    complete = False
                    findings.append(
                        MeshCertificateFinding(
                            "source_surface_interpolation",
                            "unresolved",
                            "source_facet",
                            (patch,),
                            detail=f"Exact source chart interpolation is unavailable: {error}",
                        )
                    )
                    continue
            images.append(physical)
            deviations.append(interpolation)
            restrictions.append(
                canonical_fingerprint(
                    {
                        "source_revision": self.source_revision,
                        "patch": self.domain.entity_id(2, patch),
                        "uv_vertices": tuple(
                            tuple(coefficient_payload(value) for value in point)
                            for point in uv_points
                        ),
                        "uv_triangles": array_tree_fingerprint(uv_triangles),
                    }
                )
            )
        simplices = (
            np.concatenate(images) if images else np.empty((0, 3, 3), dtype=np.float64)
        )
        bounds = (
            np.concatenate(deviations) if deviations else np.empty((0,), dtype=np.float64)
        )
        semantics = (
            "certified"
            if complete
            and self.domain.accuracy in ("exact_represented", "certified_enclosure")
            else "sampled"
        )
        return SourceBoundaryChartCover(
            simplices,
            bounds,
            semantics,
            complete,
            self.source_id,
            self.source_revision,
            findings=tuple(findings),
            restriction_ids=tuple(restrictions),
        )

    def boundary_cover(self, maximum_radius: float, /) -> SourceBoundarySamples:
        """Return a complete physical-radius cover of the selected source patches."""
        if isinstance(maximum_radius, (bool, np.bool_)):
            raise TypeError(
                "maximum_radius must be a finite nonnegative physical distance."
            )
        requested = finite_real_scalar(maximum_radius, "maximum_radius")
        if requested < 0.0:
            raise ValueError("maximum_radius must be nonnegative.")
        requested_exact = Fraction.from_float(requested)
        self.require_current()
        try:
            parameter_patches = self.boundary_parameter_triangles()
        except (ArithmeticError, NotImplementedError, TypeError, ValueError):
            return SourceBoundarySamples(
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                "sampled",
                complete=False,
            )

        points: list[np.ndarray] = []
        covering_radii: list[float] = []
        point_errors: list[float] = []
        complete = True
        semantics = (
            "certified"
            if self.domain.accuracy in ("exact_represented", "certified_enclosure")
            else "sampled"
        )

        def sample_point(
            patch: int,
            evaluator: PreparedIntervalFunction,
            parameter: _Point2,
            /,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Fraction]:
            chart = np.asarray([[float(value) for value in parameter]], dtype=np.float64)
            sample = np.asarray(
                self.domain.evaluate(np.asarray([patch], dtype=np.int64), chart)[0],
                dtype=np.float64,
            )
            lower_chart = np.asarray(
                [[fraction_down(value) for value in parameter]], dtype=np.float64
            )
            upper_chart = np.asarray(
                [[fraction_up(value) for value in parameter]], dtype=np.float64
            )
            lower, upper = evaluator.evaluate(lower_chart, upper_chart)
            low = np.asarray(lower[0], dtype=np.float64)
            high = np.asarray(upper[0], dtype=np.float64)
            if (
                sample.shape != (3,)
                or low.shape != (3,)
                or high.shape != (3,)
                or not np.all(np.isfinite(sample))
                or not np.all(np.isfinite(low))
                or not np.all(np.isfinite(high))
                or np.any(low > sample)
                or np.any(sample > high)
            ):
                raise ArithmeticError(
                    "Source point has no finite enclosing representative."
                )
            squared = sum(
                (
                    max(
                        abs(
                            Fraction.from_float(float(sample[axis]))
                            - Fraction.from_float(float(low[axis]))
                        ),
                        abs(
                            Fraction.from_float(float(high[axis]))
                            - Fraction.from_float(float(sample[axis]))
                        ),
                    )
                    ** 2
                    for axis in range(3)
                ),
                Fraction(0),
            )
            error = fraction_sqrt_interval(squared)[1]
            return sample, low, high, error

        def split_triangle(
            triangle: _Triangle2, /
        ) -> tuple[_Triangle2, _Triangle2, _Triangle2, _Triangle2]:
            first, second, third = triangle
            middle_first_second = (
                (first[0] + second[0]) / 2,
                (first[1] + second[1]) / 2,
            )
            middle_second_third = (
                (second[0] + third[0]) / 2,
                (second[1] + third[1]) / 2,
            )
            middle_third_first = (
                (third[0] + first[0]) / 2,
                (third[1] + first[1]) / 2,
            )
            return (
                (first, middle_first_second, middle_third_first),
                (middle_first_second, second, middle_second_third),
                (middle_third_first, middle_second_third, third),
                (middle_first_second, middle_second_third, middle_third_first),
            )

        from .brep._root_bindings import _source_coefficients_are_binary

        for patch, source_triangles in parameter_patches:
            surface = self.domain.patches[patch].surface
            if not _source_coefficients_are_binary(surface):
                complete = False
                continue
            try:
                evaluator = prepare_interval_function(surface.evaluate, 2)
            except (ArithmeticError, NotImplementedError, TypeError, ValueError):
                complete = False
                continue
            pending = list(reversed(source_triangles))
            while pending:
                triangle = pending.pop()
                try:
                    vertex_data = tuple(
                        sample_point(patch, evaluator, parameter)
                        for parameter in triangle
                    )
                    center_parameter = (
                        sum((point[0] for point in triangle), Fraction(0)) / 3,
                        sum((point[1] for point in triangle), Fraction(0)) / 3,
                    )
                    center, center_lower, center_upper, point_error = sample_point(
                        patch, evaluator, center_parameter
                    )
                    chart_lower = tuple(
                        fraction_down(min(point[axis] for point in triangle))
                        for axis in range(2)
                    )
                    chart_upper = tuple(
                        fraction_up(max(point[axis] for point in triangle))
                        for axis in range(2)
                    )
                    interval_cell = np.asarray(
                        [
                            [
                                chart_lower,
                                (chart_upper[0], chart_lower[1]),
                                chart_upper,
                                (chart_lower[0], chart_upper[1]),
                            ]
                        ],
                        dtype=np.float64,
                    )
                    deviation_value = float(
                        self.domain.interpolation_bounds(patch, interval_cell)[0]
                    )
                except (ArithmeticError, NotImplementedError, TypeError, ValueError):
                    complete = False
                    continue
                if not math.isfinite(deviation_value) or deviation_value < 0.0:
                    complete = False
                    continue
                deviation = Fraction.from_float(deviation_value)
                maximum_corner_error = max(data[3] for data in vertex_data)
                corner_distances = []
                for sample, _lower, _upper, _error in vertex_data:
                    squared = Fraction(0)
                    for axis in range(3):
                        value = Fraction.from_float(float(sample[axis]))
                        lower = Fraction.from_float(float(center_lower[axis]))
                        upper = Fraction.from_float(float(center_upper[axis]))
                        gap = max(lower - value, Fraction(0), value - upper)
                        squared += gap * gap
                    corner_distances.append(fraction_sqrt_interval(squared)[1])
                radius = max(corner_distances) + maximum_corner_error + deviation
                if radius <= requested_exact:
                    points.append(center)
                    covering_radii.append(fraction_up(radius))
                    point_errors.append(fraction_up(point_error))
                    continue
                children = split_triangle(triangle)
                midpoints = tuple(
                    tuple(float(value) for value in point)
                    for point in (
                        tuple(
                            (triangle[0][axis] + triangle[1][axis]) / 2
                            for axis in range(2)
                        ),
                        tuple(
                            (triangle[1][axis] + triangle[2][axis]) / 2
                            for axis in range(2)
                        ),
                        tuple(
                            (triangle[2][axis] + triangle[0][axis]) / 2
                            for axis in range(2)
                        ),
                    )
                )
                vertices = tuple(
                    tuple(float(value) for value in point) for point in triangle
                )
                if (
                    any(
                        midpoint == vertex
                        for midpoint in midpoints
                        for vertex in vertices
                    )
                    or len(set(midpoints)) != 3
                ):
                    complete = False
                    continue
                pending.extend(reversed(children))

        return SourceBoundarySamples(
            np.asarray(points, dtype=np.float64).reshape((-1, 3)),
            np.asarray(covering_radii, dtype=np.float64),
            np.asarray(point_errors, dtype=np.float64),
            semantics if complete else "sampled",
            complete=complete,
        )

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        queries = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        if not np.all(np.isfinite(queries)):
            raise ValueError("boundary_distance points must be finite.")
        cover = self.boundary_chart_cover()
        if not cover.complete or cover.simplices.shape[0] == 0:
            return SourceBoundaryDistance(
                np.zeros((queries.shape[0],), dtype=np.float64),
                np.full((queries.shape[0],), np.inf, dtype=np.float64),
                "sampled",
            )
        lower = np.empty((queries.shape[0],), dtype=np.float64)
        upper = np.empty_like(lower)
        exact_triangles: tuple[_Triangle3, ...] = tuple(
            _triangle3(tuple(rational_vector(row) for row in triangle))
            for triangle in cover.simplices
        )
        deviations = tuple(Fraction(float(value)) for value in cover.deviation_bounds)
        for row, query in enumerate(queries):
            exact_query = _point3(rational_vector(query))
            low_bounds: list[Fraction] = []
            high_bounds = []
            for triangle, deviation in zip(exact_triangles, deviations, strict=True):
                squared = _point_triangle_squared(exact_query, triangle)
                root = fraction_sqrt_interval(squared)
                low_bounds.append(max(Fraction(0), root[0] - deviation))
                high_bounds.append(root[1] + deviation)
            lower[row] = fraction_endpoint(min(low_bounds), upper=False, saturate=True)
            upper[row] = fraction_endpoint(min(high_bounds), upper=True, saturate=True)
        return SourceBoundaryDistance(lower, upper, cover.semantics)


__all__ = ["MeshingDomainBoundarySource"]
