#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rational rounding bounds for planar offset publication."""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations

import numpy as np

from .._rational import fraction_down, fraction_up, rational_vector
from ._constructors import _Builder as Builder
from ._exact import square_root_interval
from ._model import BRepGeometry
from ._offset_result import BRepOffsetEvidence
from ._patches import ExactLineCurve, LineCurve, PlanePatch


def rounding_norm(values: tuple[Fraction, ...], /) -> float:
    squared = sum((value * value for value in values), Fraction())
    if squared == 0:
        return 0.0
    return square_root_interval(squared)[1]


def vertex_rounding_bounds(
    vertices: tuple[tuple[Fraction, ...], ...], /
) -> tuple[float, ...]:
    return tuple(
        rounding_norm(tuple(Fraction(float(value)) - value for value in vertex))
        for vertex in vertices
    )


def _vertex_exact(builder: Builder, index: int, /) -> tuple[Fraction, ...]:
    exact = builder.vertex_exact_points[index]
    if exact is not None:
        return exact
    return rational_vector(builder.vertices[index])


def _line_coefficients(
    curve: ExactLineCurve | LineCurve, /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    if isinstance(curve, ExactLineCurve):
        origin = curve.first_exact
        direction = tuple(
            last - first
            for first, last in zip(curve.first_exact, curve.last_exact, strict=True)
        )
        return origin, direction
    return rational_vector(np.asarray(curve.origin)), rational_vector(
        np.asarray(curve.direction)
    )


def publish_endpoint_rounding(
    builder: Builder, /
) -> tuple[tuple[float, float], ...] | None:
    """Record line endpoint-to-vertex gaps, not a displacement estimate."""
    rows = []
    for edge, (first, last) in enumerate(builder.edge_vertices):
        curve_index = builder.edge_curves[edge]
        if curve_index < 0:
            rows.append((0.0, 0.0))
            continue
        curve = builder.curves[curve_index]
        if not isinstance(curve, (ExactLineCurve, LineCurve)):
            raise ValueError(
                "brep_offset.endpoint_gaps must use affine line carriers; got another curve family"
            )
        origin, direction = _line_coefficients(curve)
        bounds = []
        for parameter, vertex in zip(
            builder.edge_ranges[edge], (first, last), strict=True
        ):
            point = _vertex_exact(builder, vertex)
            t = Fraction.from_float(float(parameter))
            difference = tuple(
                a + t * b - c for a, b, c in zip(origin, direction, point, strict=True)
            )
            bounds.append(rounding_norm(difference))
        rows.append((bounds[0], bounds[1]))
    return tuple(rows) if any(a or b for a, b in rows) else None


def _multiply(
    a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    products = tuple(x * y for x in a for y in b)
    return min(products), max(products)


def _determinant_enclosure(
    points: tuple[tuple[tuple[Fraction, Fraction], ...], ...], /
) -> tuple[Fraction, Fraction]:
    lower = upper = Fraction()
    for order in permutations(range(3)):
        term = _multiply(
            _multiply(points[0][order[0]], points[1][order[1]]), points[2][order[2]]
        )
        inversions = sum(
            order[first] > order[last]
            for first in range(3)
            for last in range(first + 1, 3)
        )
        if inversions % 2:
            lower -= term[1]
            upper -= term[0]
        else:
            lower += term[0]
            upper += term[1]
    return lower, upper


def offset_rounding_evidence(
    builder: Builder,
    geometry: BRepGeometry,
    vertices: tuple[tuple[Fraction, ...], ...],
    normals: tuple[tuple[Fraction, ...], ...],
    orientations: tuple[int, ...],
    /,
) -> tuple[BRepOffsetEvidence, float, tuple[float, ...] | None]:
    """Enclose the coefficient-rounding homotopy in each oriented face integral."""
    face_bounds = []
    edge_face_bounds = [0.0] * len(builder.coedge_edges)
    lower = upper = Fraction()
    vertex_bounds = vertex_rounding_bounds(vertices)
    rounding_margin = max(vertex_bounds, default=0.0)
    for face, staged in enumerate(builder.faces):
        patch = staged.patch
        if not isinstance(patch, PlanePatch):
            raise ValueError(
                "brep_offset.face_rounding must use affine planes; got another patch family"
            )
        source_edge = geometry.coedge_edges[geometry.face_loops[face][0][0]]
        anchor = geometry.edge_vertices[source_edge][0]
        origin, first_axis, second_axis = (
            rational_vector(np.asarray(value))
            for value in (patch.origin, patch.first_axis, patch.second_axis)
        )
        error = tuple(a - b for a, b in zip(origin, vertices[anchor], strict=True))
        plane_distance = abs(
            sum((a * b for a, b in zip(normals[face], error, strict=True)), Fraction())
        )
        face_bounds.append(fraction_up(plane_distance))
        for loop in staged.loops:
            points = []
            for coedge in loop:
                edge = builder.coedge_edges[coedge]
                pcurve = builder.pcurves[coedge]
                if not isinstance(pcurve, LineCurve):
                    raise ValueError(
                        "brep_offset.trim_rounding must use affine lines; got another trim family"
                    )
                uv_origin, uv_direction = (
                    rational_vector(np.asarray(pcurve.origin)),
                    rational_vector(np.asarray(pcurve.direction)),
                )
                represented_points = []
                # Both affine images agree up to an affine error along the edge, so
                # its supremum is attained at one of the two endpoints.
                for endpoint in (0, 1):
                    vertex = builder.edge_vertices[edge][endpoint]
                    t = Fraction(builder.edge_ranges[edge][endpoint])
                    u, v = tuple(
                        a + t * b for a, b in zip(uv_origin, uv_direction, strict=True)
                    )
                    represented = tuple(
                        a + u * b + v * c
                        for a, b, c in zip(origin, first_axis, second_axis, strict=True)
                    )
                    gap = rounding_norm(
                        tuple(
                            a - b
                            for a, b in zip(represented, vertices[vertex], strict=True)
                        )
                    )
                    edge_face_bounds[coedge] = max(edge_face_bounds[coedge], gap)
                    rounding_margin = max(rounding_margin, gap)
                    represented_points.append((vertex, represented))
                vertex, represented = represented_points[
                    0 if builder.coedge_senses[coedge] == 1 else 1
                ]
                points.append(
                    tuple(
                        (min(a, b), max(a, b))
                        for a, b in zip(vertices[vertex], represented, strict=True)
                    )
                )
            for index in range(1, len(points) - 1):
                lo, hi = _determinant_enclosure(
                    (points[0], points[index], points[index + 1])
                )
                if orientations[face] < 0:
                    lo, hi = -hi, -lo
                lower += lo / 6
                upper += hi / 6
    edge_face_gap_bounds = tuple(edge_face_bounds) if any(edge_face_bounds) else None
    evidence = BRepOffsetEvidence(
        face_rounding_bounds=np.asarray(face_bounds, dtype=np.float64),
        vertex_rounding_bounds=np.asarray(vertex_bounds, dtype=np.float64),
        volume_enclosure=np.asarray(
            (fraction_down(lower), fraction_up(upper)), dtype=np.float64
        ),
    )
    return evidence, rounding_margin, edge_face_gap_bounds
