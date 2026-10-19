#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rational rounding and volume enclosures for planar offset publication."""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations

import numpy as np

from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from .._planar_primitives import reserve_fraction_work
from .._rational import fraction_down, fraction_up, rational_vector
from ._constructors import Builder
from ._exact import square_root_interval
from ._model import BRepGeometry
from ._offset_result import BRepOffsetEvidence
from ._patches import LineCurve, PlanePatch


def rounding_norm(
    values: tuple[Fraction, ...], /, *, budget: CoordinateEnclosureBudget
) -> float:
    reserve_fraction_work((values,), 3 * len(values), 1, 6, budget=budget)
    squared = sum((value * value for value in values), Fraction())
    if squared == 0:
        return 0.0
    return square_root_interval(squared)[1]


def vertex_rounding_bounds(
    vertices: tuple[tuple[Fraction, ...], ...], /, *, budget: CoordinateEnclosureBudget
) -> tuple[float, ...]:
    return tuple(
        rounding_norm(
            tuple(Fraction(float(value)) - value for value in vertex), budget=budget
        )
        for vertex in vertices
    )


def publish_endpoint_rounding(
    builder: Builder, /, *, budget: CoordinateEnclosureBudget
) -> None:
    """Record published line-endpoint gaps, never a displacement estimate."""
    rows = []
    for edge, (first, last) in enumerate(builder.edge_vertices):
        curve = builder.curves[builder.edge_curves[edge]]
        if not isinstance(curve, LineCurve):
            raise ValueError(
                "brep_offset.endpoint_gaps must use affine line carriers; got another curve family"
            )
        origin, direction = (
            rational_vector(np.asarray(curve.origin)),
            rational_vector(np.asarray(curve.direction)),
        )
        bounds = []
        for parameter, vertex in zip(
            builder.edge_ranges[edge], (first, last), strict=True
        ):
            point = rational_vector(builder.vertices[vertex])
            difference = tuple(
                a + Fraction(parameter) * b - c
                for a, b, c in zip(origin, direction, point, strict=True)
            )
            bounds.append(rounding_norm(difference, budget=budget))
        rows.append((bounds[0], bounds[1]))
    builder.edge_vertex_gap_bounds = tuple(rows) if any(a or b for a, b in rows) else None


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
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[BRepOffsetEvidence, float]:
    """Bound both intended and represented face integrals by multilinearity.

    Each triangle vertex lies in the rational component hull of its intended
    incident-plane solution and its published p-curve lift. Directed rational
    determinant arithmetic encloses the whole coefficient-rounding homotopy.
    """
    reserve_fraction_work(
        vertices,
        100 * len(builder.coedge_edges),
        48 * len(builder.coedge_edges),
        32,
        budget=budget,
    )
    face_bounds = []
    lower = upper = Fraction()
    vertex_bounds = vertex_rounding_bounds(vertices, budget=budget)
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
                endpoint = 0 if builder.coedge_senses[coedge] == 1 else 1
                vertex = builder.edge_vertices[edge][endpoint]
                pcurve = builder.pcurves[coedge]
                if not isinstance(pcurve, LineCurve):
                    raise ValueError(
                        "brep_offset.trim_rounding must use affine lines; got another trim family"
                    )
                t = Fraction(builder.edge_ranges[edge][endpoint])
                uv_origin, uv_direction = (
                    rational_vector(np.asarray(pcurve.origin)),
                    rational_vector(np.asarray(pcurve.direction)),
                )
                u, v = tuple(
                    a + t * b for a, b in zip(uv_origin, uv_direction, strict=True)
                )
                represented = tuple(
                    a + u * b + v * c
                    for a, b, c in zip(origin, first_axis, second_axis, strict=True)
                )
                rounding_margin = max(
                    rounding_margin,
                    rounding_norm(
                        tuple(
                            a - b
                            for a, b in zip(represented, vertices[vertex], strict=True)
                        ),
                        budget=budget,
                    ),
                )
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
    evidence = BRepOffsetEvidence(
        face_rounding_bounds=np.asarray(face_bounds, dtype=np.float64),
        vertex_rounding_bounds=np.asarray(vertex_bounds, dtype=np.float64),
        volume_enclosure=np.asarray(
            (fraction_down(lower), fraction_up(upper)), dtype=np.float64
        ),
    )
    return evidence, rounding_margin
