#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact source/target coplanar triangle partition mathematics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import numpy as np

from ..discretization._exact_algebraic_geometry import ExactScalar
from ._planar_primitives import (
    box_overlap_pairs,
    intersection,
    plane_key,
    polygon_boxes,
    project_polygon,
    projection_axes,
    signed_area2,
)


type _Point2 = tuple[Fraction, Fraction]
type _TriangleInput = Sequence[Sequence[ExactScalar | int | np.integer]]


@dataclass(frozen=True, slots=True)
class CoplanarTrianglePiece:
    """One intersection simplex in both input triangles' exact unit charts."""

    source_row: int
    target_row: int
    source_chart: tuple[tuple[Fraction, Fraction], ...]
    target_chart: tuple[tuple[Fraction, Fraction], ...]


def _unit_chart(point: _Point2, triangle: Sequence[_Point2]) -> _Point2:
    a, b, c = triangle
    x, y = point[0] - a[0], point[1] - a[1]
    ux, uy = b[0] - a[0], b[1] - a[1]
    vx, vy = c[0] - a[0], c[1] - a[1]
    determinant = ux * vy - uy * vx
    if not determinant:
        raise ValueError("Source and target overlap triangles must be nondegenerate.")
    return ((x * vy - y * vx) / determinant, (ux * y - uy * x) / determinant)


def _exact_triangle(triangle: _TriangleInput) -> tuple[tuple[Fraction, ...], ...]:
    if len(triangle) != 3 or any(len(point) != 3 for point in triangle):
        raise ValueError("Overlap triangles require three exact 3D points.")
    if any(
        not isinstance(value, (Fraction, int, np.integer))
        or isinstance(value, (bool, np.bool_))
        for point in triangle
        for value in point
    ):
        raise TypeError("Overlap coordinates must be exact rational values, not floats.")
    return tuple(
        tuple(
            Fraction(int(value)) if isinstance(value, np.integer) else Fraction(value)
            for value in point
        )
        for point in cast(Sequence[Sequence[Fraction | int | np.integer]], triangle)
    )


def exact_coplanar_triangle_partitions(
    source_triangles: Sequence[_TriangleInput],
    target_triangles: Sequence[_TriangleInput],
    /,
    *,
    require_source: bool = True,
    require_target: bool = True,
) -> tuple[CoplanarTrianglePiece, ...]:
    """Intersect coplanar triangle banks and prove their requested partitions.

    Input coordinates are exact rational physical points. Each result retains
    both input row identities and both unit-chart images. Exact plane grouping
    and the canonical spatial box index bound the candidate workset. The caller
    must select matching occurrence/material strata before invoking this routine.
    """
    from ._source_fidelity import (
        _require_reference_chart_partition,
        _screened_intersection_areas,
    )

    sources = tuple(_exact_triangle(triangle) for triangle in source_triangles)
    targets = tuple(_exact_triangle(triangle) for triangle in target_triangles)
    source_groups = {}
    target_groups = {}
    for row, triangle in enumerate(sources):
        source_groups.setdefault(plane_key(triangle), []).append(row)
    for row, triangle in enumerate(targets):
        target_groups.setdefault(plane_key(triangle), []).append(row)
    by_source = [[] for _ in sources]
    by_target = [[] for _ in targets]
    pieces = []
    for plane in source_groups.keys() & target_groups.keys():
        source_rows = source_groups[plane]
        target_rows = target_groups[plane]
        axes = projection_axes(plane)
        source_polygons = tuple(
            project_polygon(sources[row], axes) for row in source_rows
        )
        target_polygons = tuple(
            project_polygon(targets[row], axes) for row in target_rows
        )
        source_boxes = polygon_boxes(source_polygons)
        target_boxes = polygon_boxes(target_polygons)
        if source_boxes is None or target_boxes is None:
            raise ValueError(
                "Exact overlap coordinates exceed the bounded spatial index."
            )
        candidates = box_overlap_pairs(
            len(source_rows),
            len(target_rows),
            source_boxes,
            target_boxes,
            self_pairs=False,
        )
        for source_index, target_index in candidates:
            source_index, target_index = int(source_index), int(target_index)
            source_polygon = source_polygons[source_index]
            target_polygon = target_polygons[target_index]
            polygon = intersection(target_polygon, source_polygon)
            if len(polygon) < 3 or signed_area2(polygon) == 0:
                continue
            # Preserve the target face's oriented vertex order.
            if signed_area2(target_polygon) < 0:
                polygon = tuple(reversed(polygon))
            for corner in range(1, len(polygon) - 1):
                simplex = (polygon[0], polygon[corner], polygon[corner + 1])
                if signed_area2(simplex) == 0:
                    continue
                source_chart = (
                    _unit_chart(simplex[0], source_polygon),
                    _unit_chart(simplex[1], source_polygon),
                    _unit_chart(simplex[2], source_polygon),
                )
                target_chart = (
                    _unit_chart(simplex[0], target_polygon),
                    _unit_chart(simplex[1], target_polygon),
                    _unit_chart(simplex[2], target_polygon),
                )
                source_row, target_row = (
                    source_rows[source_index],
                    target_rows[target_index],
                )
                pieces.append(
                    CoplanarTrianglePiece(
                        source_row, target_row, source_chart, target_chart
                    )
                )
                by_source[source_row].append(source_chart)
                by_target[target_row].append(target_chart)
    standard = (
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
    )
    for banks, complete in ((by_source, require_source), (by_target, require_target)):
        for bank in banks:
            triangles = tuple(bank)
            if complete:
                _require_reference_chart_partition(triangles, standard)
            elif len(triangles) > 1 and any(
                area > 0
                for _, _, area in _screened_intersection_areas(
                    triangles, triangles, self_pairs=True
                )
            ):
                raise ValueError(
                    "Source or target overlap pieces have positive-area overlap."
                )
    return tuple(
        sorted(
            pieces,
            key=lambda piece: (piece.target_row, piece.source_row, piece.target_chart),
        )
    )
