#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact original CAD face regions remaining outside an accepted layer."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..geometry._coplanar_overlap import exact_coplanar_triangle_partitions
from ..geometry._domain_boundary import _hole_seed
from ..geometry._planar_chain_embedding import triangulate_planar_polygon
from ..geometry._planar_primitives import plane_key, projection_axes, signed_area2
from ..geometry._triangulation import ConstrainedDelaunayTriangulation
from ..geometry.brep._planar_arrangement import (
    arrange,
    inside_loop,
    on_segment,
    rational_loops,
    selected_cell_boundaries,
)


if TYPE_CHECKING:
    from ._layer_core_cad_footprints import LayerPlanarCadFootprints
    from ._layer_core_edge_policy import _PlanarCadStrata


type _Point2 = tuple[Fraction, Fraction]
type _Point3 = tuple[Fraction, Fraction, Fraction]
type _Triangle3 = tuple[_Point3, _Point3, _Point3]
type _Loop2 = tuple[_Point2, ...]


def _patch_loops(
    strata: _PlanarCadStrata, patch: int, axes: tuple[int, int], /
) -> tuple[_Loop2, ...]:
    """Reconstruct each declared coedge loop from its proved exact segment chain."""
    grouped: dict[int, set[tuple[_Point2, _Point2]]] = {}
    for segment in strata.segments:
        first = (segment.first[axes[0]], segment.first[axes[1]])
        last = (segment.last[axes[0]], segment.last[axes[1]])
        edge = (min(first, last), max(first, last))
        for use in segment.uses:
            if use.patch_row == patch:
                grouped.setdefault(use.loop_index, set()).add(edge)
    if not grouped or tuple(sorted(grouped)) != tuple(range(len(grouped))):
        raise ValueError("A source face lacks complete ordered CAD coedge loops.")
    output: list[_Loop2] = []
    for loop_index in sorted(grouped):
        adjacency: dict[_Point2, set[_Point2]] = {}
        for first, last in grouped[loop_index]:
            adjacency.setdefault(first, set()).add(last)
            adjacency.setdefault(last, set()).add(first)
        if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
            raise ValueError("Source coedge segments do not form a simple closed loop.")
        start = min(adjacency)
        previous, current = start, min(adjacency[start])
        loop = [start]
        while current != start:
            if current in loop:
                raise ValueError("Source coedge loop repeats a vertex.")
            loop.append(current)
            next_values = adjacency[current] - {previous}
            if len(next_values) != 1:
                raise ValueError("Source coedge loop has ambiguous incidence.")
            previous, current = current, next(iter(next_values))
        if len(loop) != len(adjacency):
            raise ValueError("A declared source coedge loop has disconnected pieces.")
        output.append(tuple(loop))
    return rational_loops(tuple(output))


def _lift(
    point: _Point2, plane: tuple[int, int, int, int], axes: tuple[int, int], /
) -> _Point3:
    dropped = next(axis for axis in range(3) if axis not in axes)
    values = [Fraction(0)] * 3
    values[axes[0]], values[axes[1]] = point
    values[dropped] = (
        -(Fraction(plane[3]) + plane[axes[0]] * point[0] + plane[axes[1]] * point[1])
        / plane[dropped]
    )
    return values[0], values[1], values[2]


def _triangulate_region(
    loops: tuple[_Loop2, ...], /
) -> tuple[tuple[_Point2, _Point2, _Point2], ...]:
    """Triangulate without changing any represented source or cap boundary vertex."""
    if len(loops) == 1:
        outer = loops[0]
        triangles = triangulate_planar_polygon(
            tuple((x, y, Fraction(0)) for x, y in outer)
        )
        return tuple((outer[a], outer[b], outer[c]) for a, b, c in triangles)
    points = tuple(point for loop in loops for point in loop)
    if len(set(points)) != len(points):
        raise ValueError("Distinct source boundary loops share a point.")
    indices = {point: row for row, point in enumerate(points)}
    segments = np.asarray(
        [
            (indices[a], indices[b])
            for loop in loops
            for a, b in zip(loop, (*loop[1:], loop[0]), strict=True)
        ],
        dtype=np.int64,
    )
    numeric = np.asarray(
        [[float(value) for value in point] for point in points], dtype=np.float64
    )
    if any(
        Fraction.from_float(float(numeric[row, axis])) != point[axis]
        for row, point in enumerate(points)
        for axis in range(2)
    ):
        raise ValueError(
            "Exact source boundary vertices cannot be represented by the native planar carrier."
        )
    holes = []
    for loop in loops[1:]:
        seed = _hole_seed(loop)
        rounded = tuple(Fraction.from_float(float(value)) for value in seed)
        if not inside_loop((rounded[0], rounded[1]), loop):
            raise ValueError(
                "A native hole seed cannot be represented inside its exact source hole."
            )
        holes.append(tuple(float(value) for value in seed))
    triangulation = ConstrainedDelaunayTriangulation(
        numeric,
        segments,
        holes=np.asarray(holes, dtype=np.float64),
        min_angle=0.0,
        max_area=float("inf"),
    )
    if triangulation.points.shape != numeric.shape or not np.array_equal(
        triangulation.points, numeric
    ):
        raise ValueError(
            "Planar source triangulation changed an exact protected boundary vertex."
        )
    return tuple(
        (points[int(row[0])], points[int(row[1])], points[int(row[2])])
        for row in triangulation.triangles
    )


def remaining_brep_face_triangles(
    footprints: LayerPlanarCadFootprints,
    cap_edges: tuple[tuple[_Point3, _Point3], ...],
    /,
) -> tuple[tuple[int, _Triangle3], ...]:
    """Subtract actual layer images from each original exact CAD face.

    The original CAD trim and actual layer boundary are arranged directly;
    source surface tessellation diagonals are never promoted to constraints.
    """
    strata = footprints.strata
    output: list[tuple[int, _Triangle3]] = []
    for patch_record, layer_rows in zip(
        strata.patches, footprints.patch_triangles, strict=True
    ):
        patch = patch_record.patch_row
        source = tuple(
            triangle
            for triangle, (row, _) in zip(strata.triangles, strata.charts, strict=True)
            if row == patch
        )
        if not source:
            raise ValueError("An original CAD face has no exact source atlas triangles.")
        plane = plane_key(source[0])
        if any(plane_key(triangle) != plane for triangle in source):
            raise ValueError("Source face spans more than one exact plane.")
        axes = projection_axes(plane)
        layer = tuple(row.vertices for row in layer_rows)
        pieces = (
            exact_coplanar_triangle_partitions(
                source, layer, require_source=False, require_target=True
            )
            if layer
            else ()
        )
        coverage = [Fraction(0)] * len(source)
        for piece in pieces:
            coverage[piece.source_row] += abs(signed_area2(piece.source_chart))
        if all(value == 1 for value in coverage):
            exact_coplanar_triangle_partitions(source, layer)
            continue
        loops = _patch_loops(strata, patch, axes)
        regions: tuple[tuple[object, ...], ...] = (tuple(loops),) + tuple(
            (tuple((point[axes[0]], point[axes[1]]) for point in triangle),)
            for triangle in layer
        )
        cells = arrange(regions)
        remaining = tuple(
            cell for cell in cells if 0 in cell.members and len(cell.members) == 1
        )
        boundaries = selected_cell_boundaries(remaining)
        projected_caps = tuple(
            ((first[axes[0]], first[axes[1]]), (last[axes[0]], last[axes[1]]))
            for first, last in cap_edges
            if all(
                sum(plane[axis] * point[axis] for axis in range(3)) + plane[3] == 0
                for point in (first, last)
            )
        )
        cap_vertices = {point for edge in projected_caps for point in edge}
        triangles: list[_Triangle3] = []
        for region in boundaries:
            for loop in region:
                for point in loop:
                    if point not in cap_vertices and any(
                        on_segment(point, edge) for edge in projected_caps
                    ):
                        raise ValueError(
                            "A source arrangement would split an unchanged cap edge."
                        )
            for triangle in _triangulate_region(region):
                physical = tuple(_lift(point, plane, axes) for point in triangle)
                triangles.append((physical[0], physical[1], physical[2]))
        exact_coplanar_triangle_partitions(source, tuple(triangles) + layer)
        output.extend((patch, triangle) for triangle in triangles)
    return tuple(output)
