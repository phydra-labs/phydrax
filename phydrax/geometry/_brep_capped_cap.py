#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact trimmed planar-cap cells with native circular boundary arcs."""

from __future__ import annotations

from fractions import Fraction
from typing import cast, NamedTuple

import numpy as np

from ..discretization._exact_algebraic_geometry import ExactScalar
from ._brep_circle_rim_proof import (
    CircleRimEdge,
    prove_circle_cap_disk,
    require_circle_disk_point,
)
from ._circle_fan_material import _coefficient, CircleFanMaterialMap
from ._domain_boundary import ExactPlaneChart
from ._interval_enclosure import (
    Interval,
    interval_add,
    interval_cos,
    interval_multiply,
    interval_sin,
    interval_subtract,
)
from ._surface_source_support import _cad_root_edges, SurfaceSourceCharts
from .brep._boolean_core import ChartScalar
from .brep._carrier_pieces import carrier_content_id
from .brep._patches import CircleCurve
from .brep._source_embedding import _same
from .brep._trim_helpers import period_scalar_bounds, PeriodOffset


class CirclePoint(NamedTuple):
    curve: CircleCurve
    phase: ChartScalar


type ExactPoint = tuple[ExactScalar, ExactScalar, ExactScalar]
type CapPoint = ExactPoint | CirclePoint


class PreparedCapCells(NamedTuple):
    corners: dict[int, tuple[CapPoint, CapPoint, CapPoint]]
    fans: dict[int, tuple[CircleFanMaterialMap, tuple[int, int, int]]]


def _dot(a: tuple[ExactScalar, ...], b: tuple[ExactScalar, ...], /) -> ExactScalar:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _project(plane: ExactPlaneChart, coordinate: np.ndarray, /) -> ExactPoint:
    """Project the exact carrier shadow into the exact source plane frame."""
    offset = tuple(
        Fraction.from_float(float(coordinate[i])) - plane.origin[i] for i in range(3)
    )
    a, b = plane.first_axis, plane.second_axis
    aa, ab, bb = _dot(a, a), _dot(a, b), _dot(b, b)
    determinant = aa * bb - ab * ab
    if determinant == 0:
        raise ValueError("The cap source plane frame is singular.")
    u = (bb * _dot(a, offset) - ab * _dot(b, offset)) / determinant
    v = (aa * _dot(b, offset) - ab * _dot(a, offset)) / determinant
    values = tuple(plane.origin[i] + u * a[i] + v * b[i] for i in range(3))
    return values[0], values[1], values[2]


def _point_interval(point: CapPoint, /) -> tuple[Interval, ...]:
    if not isinstance(point, CirclePoint):
        return tuple(_coefficient(value) for value in point)
    curve = point.curve
    angle = period_scalar_bounds(point.phase)
    cosine, sine = interval_cos(angle), interval_sin(angle)
    radius = _coefficient(curve.radius_exact)
    return tuple(
        interval_add(
            _coefficient(curve.center_exact[i]),
            interval_multiply(
                radius,
                interval_add(
                    interval_multiply(cosine, _coefficient(curve.first_axis_exact[i])),
                    interval_multiply(sine, _coefficient(curve.second_axis_exact[i])),
                ),
            ),
        )
        for i in range(3)
    )


def _orientation(
    points: tuple[CapPoint, CapPoint, CapPoint], plane: ExactPlaneChart, /
) -> int:
    """Strict outward interval determinant in the retained exact plane frame."""
    a, b, c = (_point_interval(point) for point in points)
    u = tuple(interval_subtract(b[i], a[i]) for i in range(3))
    v = tuple(interval_subtract(c[i], a[i]) for i in range(3))
    normal = (
        plane.first_axis[1] * plane.second_axis[2]
        - plane.first_axis[2] * plane.second_axis[1],
        plane.first_axis[2] * plane.second_axis[0]
        - plane.first_axis[0] * plane.second_axis[2],
        plane.first_axis[0] * plane.second_axis[1]
        - plane.first_axis[1] * plane.second_axis[0],
    )
    cross = tuple(
        interval_subtract(interval_multiply(u[j], v[k]), interval_multiply(u[k], v[j]))
        for j, k in ((1, 2), (2, 0), (0, 1))
    )
    value = interval_add(
        interval_add(
            interval_multiply(cross[0], _coefficient(normal[0])),
            interval_multiply(cross[1], _coefficient(normal[1])),
        ),
        interval_multiply(cross[2], _coefficient(normal[2])),
    )
    if float(value[0]) > 0:
        return 1
    if float(value[1]) < 0:
        return -1
    raise ValueError(
        "A cap material triangle has no strict whole-cell orientation proof."
    )


def prepare_circular_cap_cells(
    charts: SurfaceSourceCharts, rims: dict[int, CircleRimEdge], side_cells: set[int], /
) -> PreparedCapCells:
    """Prove every cap cell is inside its actual disk with conforming edges."""
    charts.require_current()
    edge_vertices, adjacency = _cad_root_edges(charts)
    row_of = {cell: row for row, cell in enumerate(charts.cell_global_ids)}
    vertices: dict[int, tuple[int, int, int]] = {}
    for block in charts.mesh.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Cap material charts require triangles.")
        for cell, row in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        ):
            vertices[int(cell)] = (int(row[0]), int(row[1]), int(row[2]))
    planes: dict[int, ExactPlaneChart] = {}
    cap_curves: dict[int, CircleCurve] = {}
    rim_phases: dict[tuple[int, int], CirclePoint] = {}
    cell_rims: dict[int, int] = {}
    tau = PeriodOffset(Fraction(0), Fraction(1))
    for edge, rim in rims.items():
        patch = int(charts.root_patches[row_of[rim.cap_cell]])
        if patch not in planes:
            planes[patch] = prove_circle_cap_disk(
                charts.domain, patch, rim.curve_row, rim.curve
            )
            cap_curves[patch] = rim.curve
        elif carrier_content_id(cap_curves[patch]) != carrier_content_id(rim.curve):
            raise ValueError("A circular cap changes its exact rim carrier.")
        if rim.cap_cell in cell_rims:
            raise ValueError(
                "A cap material triangle cannot have two circular rim edges."
            )
        cell_rims[rim.cap_cell] = edge
        for vertex, phase in zip(edge_vertices[edge], (rim.first, rim.last), strict=True):
            key = patch, vertex
            previous = rim_phases.get(key)
            if previous is not None and not (
                _same(previous.phase, phase)
                or (_same(previous.phase, Fraction(0)) and _same(phase, tau))
                or (_same(phase, Fraction(0)) and _same(previous.phase, tau))
            ):
                raise ValueError("Cap rim edges disagree on a source vertex phase.")
            rim_phases[key] = CirclePoint(rim.curve, phase)
    coordinate = np.asarray(charts.mesh.coordinates, dtype=np.float64)
    projected: dict[tuple[int, int], ExactPoint] = {}
    corners: dict[int, tuple[CapPoint, CapPoint, CapPoint]] = {}
    fans: dict[int, tuple[CircleFanMaterialMap, tuple[int, int, int]]] = {}
    orientations: dict[int, int] = {}
    slots = ((1, 2), (2, 0), (0, 1))
    for cell, row in row_of.items():
        if cell in side_cells:
            continue
        patch = int(charts.root_patches[row])
        if patch not in planes:
            raise ValueError(
                "A material cell is neither a proved cap nor a cylindrical side."
            )
        values: list[CapPoint] = []
        for vertex in vertices[cell]:
            key = patch, vertex
            point: CapPoint | None = rim_phases.get(key)
            if point is None:
                if key not in projected:
                    projected[key] = _project(planes[patch], coordinate[vertex])
                    require_circle_disk_point(cap_curves[patch], projected[key])
                point = projected[key]
            values.append(point)
        triangle = cast(tuple[CapPoint, CapPoint, CapPoint], tuple(values))
        sign = _orientation(triangle, planes[patch])
        if patch in orientations and orientations[patch] != sign:
            raise ValueError("Cap material cells have inconsistent exact orientation.")
        orientations[patch] = sign
        corners[cell] = triangle
        edge = cell_rims.get(cell)
        if edge is None:
            continue
        rim = rims[edge]
        first_slot, last_slot = slots[rim.cap_slot]
        apex_slot = (0, 1, 2)[rim.cap_slot]
        apex = triangle[apex_slot]
        if isinstance(apex, CirclePoint):
            raise ValueError("A rim fan apex must have an exact interior plane point.")
        local = (vertices[cell][first_slot], vertices[cell][last_slot])
        canonical = edge_vertices[edge]
        if local == canonical:
            first, last = rim.first, rim.last
        elif local == canonical[::-1]:
            first, last = rim.last, rim.first
        else:
            raise ValueError("A cap fan changes its canonical rim edge.")
        fan = CircleFanMaterialMap(rim.curve, apex, first, last)
        a, b = rim.curve.first_axis_exact, rim.curve.second_axis_exact
        p, q = planes[patch].first_axis, planes[patch].second_axis
        n = (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
        m = (
            p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0],
        )
        alignment = _coefficient(_dot(n, m))
        direction = 1 if float(alignment[0]) > 0 else -1 if float(alignment[1]) < 0 else 0
        if not direction or fan.orientation * direction != sign:
            raise ValueError("The curved fan and its cap have inconsistent orientation.")
        fans[cell] = fan, (apex_slot, first_slot, last_slot)
    for edge, uses in adjacency.items():
        cap = [cell for cell, _, _, _ in uses if cell in corners]
        if not cap:
            continue
        if len(uses) != 2:
            raise ValueError("A cap material edge lacks two manifold uses.")
        if len(cap) == 1:
            if (
                edge not in rims
                or rims[edge].cap_cell != cap[0]
                or rims[edge].side_cell not in side_cells
            ):
                raise ValueError("A cap boundary has no independently proved side trace.")
        elif len(cap) == 2:
            if edge in rims or int(charts.root_patches[row_of[cap[0]]]) != int(
                charts.root_patches[row_of[cap[1]]]
            ):
                raise ValueError("A shared cap edge changes its source patch.")
            # Both endpoints are keyed by the same (patch, canonical vertex).
            # Each non-rim edge of a fan is exactly the segment to its apex.
        else:
            raise ValueError("A cap edge has invalid material incidence.")
    return PreparedCapCells(corners, fans)
