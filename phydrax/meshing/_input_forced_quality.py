#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prove final tetrahedral quality rows forced by authored PLC input angles.

A tetrahedron whose edge lies on a source ridge sits inside the ridge's
interior wedge, so its dihedral at that edge cannot exceed the input angle. If
that exact angle is below the requested dihedral aim, no mesh can meet the aim
there. Canonical acceptance may exempt such a row only when every other
dihedral of the same cell provably meets the aim.

Likewise every boundary triangle in the corner of an authored planar facet at
a source vertex has an angle of at most the corner's interior angle alpha, so
its radius-edge ratio, and that of every tetrahedron on it, is at least
1/(2 sin alpha). When that forced ratio exceeds the requested bound, the cell's
radius-edge row is exempt.

Every decision below is exact rational arithmetic, against an outward
enclosure of the aim's cosine where trigonometry appears. An unresolved
comparison leaves the row hard.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import combinations

import numpy as np

from ..geometry._domain_coverage import PiecewiseLinearDomain
from ._plc_source import PiecewiseLinearComplex
from ._volume_generation import VolumeConstruction


_Vector = tuple[Fraction, Fraction, Fraction]
_QualityRow = tuple[str, str, int, float]

# PHX_MC_TET_UNMET_RADIUS_EDGE and PHX_MC_TET_UNMET_DIHEDRAL.
_RADIUS_EDGE_CRITERION = "0"
_DIHEDRAL_CRITERION = "2"
_INTERVAL_DIGITS = 80


@dataclass(frozen=True, slots=True)
class ForcingRidge:
    """Authored PLC edge whose exact interior dihedral is below the aim."""

    entity_id: int
    ridge_id: str
    input_angle_degrees: float


@dataclass(frozen=True, slots=True)
class InputForcedDihedral:
    """Final native dihedral row forced by authored ridges of its cell.

    ``row`` indexes the evaluated quality rows; ``cell_id`` is the published
    mesh cell ID and ``achieved_degrees`` the native row value.
    """

    row: int
    cell_id: int
    achieved_degrees: float
    ridges: tuple[ForcingRidge, ...]


@dataclass(frozen=True, slots=True)
class ForcingCorner:
    """Authored facet corner whose interior angle forces a bad boundary face.

    The corner is the source vertex between two authored PLC edges of one
    planar polygon. ``forced_ratio`` is ``1 / (2 sin alpha)`` and
    ``face_ratio`` the radius-edge ratio of the cell face in that corner.
    """

    vertex_entity_id: int
    vertex_id: str
    edge_entity_ids: tuple[int, int]
    input_angle_degrees: float
    forced_ratio: float
    face_ratio: float


@dataclass(frozen=True, slots=True)
class InputForcedRadiusEdge:
    """Final native radius-edge row forced by authored facet corners.

    ``row`` indexes the evaluated quality rows; ``cell_id`` is the published
    mesh cell ID and ``achieved_ratio`` the native row value.
    """

    row: int
    cell_id: int
    achieved_ratio: float
    corners: tuple[ForcingCorner, ...]


@dataclass(frozen=True, slots=True)
class _Corner:
    facet: int
    normal: _Vector
    next_direction: _Vector
    previous_direction: _Vector
    edge_rows: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _Ridge:
    region: int
    cosine_numerator: Fraction
    cosine_denominator_squared: Fraction


def _sub(first: _Vector, second: _Vector, /) -> _Vector:
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def _cross(first: _Vector, second: _Vector, /) -> _Vector:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first: _Vector, second: _Vector, /) -> Fraction:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _negate(vector: _Vector, /) -> _Vector:
    return (-vector[0], -vector[1], -vector[2])


def _point(row: np.ndarray, /) -> _Vector:
    return (Fraction(float(row[0])), Fraction(float(row[1])), Fraction(float(row[2])))


def _is_zero(vector: _Vector, /) -> bool:
    return vector[0] == 0 and vector[1] == 0 and vector[2] == 0


@cache
def _target_cosine_bounds(target_degrees: float, /) -> tuple[Fraction, Fraction]:
    """Outward rational enclosure of ``cos(target_degrees)``."""

    from mpmath.libmp import to_rational

    from ..geometry.brep._intersection_curve import interval_context

    context = interval_context(_INTERVAL_DIGITS)
    target = Fraction(target_degrees)
    radians = (
        context.pi
        * (context.mpf(target.numerator) / context.mpf(target.denominator))
        / 180
    )
    cosine = context.cos(radians)
    return (
        Fraction(*to_rational(cosine._mpi_[0])),
        Fraction(*to_rational(cosine._mpi_[1])),
    )


def _cosine_exceeds(
    numerator: Fraction, denominator_squared: Fraction, bound: Fraction, /
) -> bool:
    """Prove ``numerator / sqrt(denominator_squared) > bound``."""

    if bound < 0:
        return (
            numerator >= 0 or numerator * numerator < bound * bound * denominator_squared
        )
    return numerator > 0 and numerator * numerator > bound * bound * denominator_squared


def _cosine_at_most(
    numerator: Fraction, denominator_squared: Fraction, bound: Fraction, /
) -> bool:
    """Prove ``numerator / sqrt(denominator_squared) <= bound``."""

    if bound >= 0:
        return (
            numerator <= 0 or numerator * numerator <= bound * bound * denominator_squared
        )
    return numerator < 0 and numerator * numerator >= bound * bound * denominator_squared


def _angle_degrees(numerator: Fraction, denominator_squared: Fraction, /) -> float:
    """Binary64 value of ``acos(numerator / sqrt(denominator_squared))`` in degrees."""

    from mpmath.libmp import to_rational

    from ..geometry.brep._intersection_curve import interval_context

    context = interval_context(_INTERVAL_DIGITS)

    def scalar(value: Fraction) -> object:
        return context.mpf(value.numerator) / context.mpf(value.denominator)

    # |u x v|^2 = |u|^2 |v|^2 - (u . v)^2 exactly; the context lacks acos.
    sine = context.sqrt(scalar(denominator_squared - numerator * numerator))
    angle = context.atan2(sine, scalar(numerator)) * 180 / context.pi
    lower = Fraction(*to_rational(angle._mpi_[0]))
    upper = Fraction(*to_rational(angle._mpi_[1]))
    return float((lower + upper) / 2)


def _forced_ratio(lengths_squared: Fraction, cross_squared: Fraction, /) -> float:
    """Binary64 value of ``1 / (2 sin alpha)`` from ``|u|^2 |v|^2`` and ``|u x v|^2``."""

    from mpmath.libmp import to_rational

    from ..geometry.brep._intersection_curve import interval_context

    context = interval_context(_INTERVAL_DIGITS)
    quotient = lengths_squared / (4 * cross_squared)
    ratio = context.sqrt(
        context.mpf(quotient.numerator) / context.mpf(quotient.denominator)
    )
    lower = Fraction(*to_rational(ratio._mpi_[0]))
    upper = Fraction(*to_rational(ratio._mpi_[1]))
    return float((lower + upper) / 2)


def _triangle_ratio(corners: np.ndarray, /) -> float:
    """Circumradius over shortest edge of one boundary triangle."""

    lengths = np.linalg.norm(corners[[1, 2, 0]] - corners, axis=1)
    doubled_area = np.linalg.norm(
        np.cross(corners[1] - corners[0], corners[2] - corners[0])
    )
    return float(np.prod(lengths) / (2.0 * doubled_area) / np.min(lengths))


class _SourceGeometry:
    """Exact source-plane ridge and corner angles of one authored PLC."""

    def __init__(self, complex_: PiecewiseLinearComplex, /) -> None:
        self._complex = complex_
        self._points = tuple(_point(row) for row in np.asarray(complex_.vertices))
        self._uses: dict[tuple[int, int], list[tuple[int, int, int]]] | None = None
        self._ridges: dict[int, _Ridge | None] = {}

    def point(self, vertex: int, /) -> _Vector:
        return self._points[vertex]

    def corners(
        self, constrained: dict[tuple[int, int], int], /
    ) -> dict[int, list[_Corner]]:
        """Convex acute planar polygon corners between two constrained PLC edges.

        ``constrained`` maps sorted source endpoints to recovered PLC edge rows.
        """

        complex_ = self._complex
        offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
        loops = np.asarray(complex_.polygon_vertices, dtype=np.int64)
        result: dict[int, list[_Corner]] = {}
        for polygon in range(offsets.size - 1):
            loop = loops[offsets[polygon] : offsets[polygon + 1]].tolist()
            normal: _Vector | None = None
            for offset, vertex in enumerate(loop):
                previous = loop[offset - 1]
                following = loop[(offset + 1) % len(loop)]
                edges = (
                    constrained.get((min(vertex, following), max(vertex, following))),
                    constrained.get((min(vertex, previous), max(vertex, previous))),
                )
                if edges[0] is None or edges[1] is None:
                    continue
                if normal is None:
                    normal = self._polygon_normal(polygon)
                    if normal is None:
                        break
                point = self._points[vertex]
                next_direction = _sub(self._points[following], point)
                previous_direction = _sub(self._points[previous], point)
                if (
                    _dot(_cross(next_direction, previous_direction), normal) <= 0
                    or _dot(next_direction, previous_direction) <= 0
                ):
                    continue
                result.setdefault(vertex, []).append(
                    _Corner(
                        int(complex_.polygon_facets[polygon]),
                        normal,
                        next_direction,
                        previous_direction,
                        (edges[0], edges[1]),
                    )
                )
        return result

    def _edge_uses(self) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
        if self._uses is None:
            offsets = np.asarray(self._complex.polygon_offsets, dtype=np.int64)
            loops = np.asarray(self._complex.polygon_vertices, dtype=np.int64)
            uses: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
            for polygon in range(offsets.size - 1):
                loop = loops[offsets[polygon] : offsets[polygon + 1]].tolist()
                for offset, first in enumerate(loop):
                    second = loop[(offset + 1) % len(loop)]
                    key = (min(first, second), max(first, second))
                    uses.setdefault(key, []).append((polygon, first, second))
            self._uses = uses
        return self._uses

    def _polygon_normal(self, polygon: int, /) -> _Vector | None:
        """Right-hand normal of the polygon's exact source plane."""

        complex_ = self._complex
        offsets = complex_.polygon_offsets
        loop = tuple(
            self._points[int(vertex)]
            for vertex in complex_.polygon_vertices[
                int(offsets[polygon]) : int(offsets[polygon + 1])
            ]
        )
        newell: _Vector = (Fraction(0), Fraction(0), Fraction(0))
        for offset, first in enumerate(loop):
            term = _cross(first, loop[(offset + 1) % len(loop)])
            newell = (newell[0] + term[0], newell[1] + term[1], newell[2] + term[2])
        if _is_zero(newell):
            return None
        facet = int(complex_.polygon_facets[polygon])
        planes = complex_.facet_planes
        if planes is not None and not np.all(np.isnan(planes[facet])):
            declared = _cross(_point(planes[facet, 1]), _point(planes[facet, 2]))
            orientation = _dot(declared, newell)
            if orientation == 0:
                return None
            return declared if orientation > 0 else _negate(declared)
        if any(_dot(newell, _sub(point, loop[0])) != 0 for point in loop[1:]):
            return None
        return newell

    def ridge(self, edge_row: int, /) -> _Ridge | None:
        """Exact interior wedge of a PLC edge with exactly two planar polygon uses."""

        if edge_row in self._ridges:
            return self._ridges[edge_row]
        result = self._compute(edge_row)
        self._ridges[edge_row] = result
        return result

    def _compute(self, edge_row: int, /) -> _Ridge | None:
        complex_ = self._complex
        first, second = complex_.plc_edge_vertices[edge_row]
        entries = self._edge_uses().get((min(first, second), max(first, second)), [])
        if len(entries) != 2 or entries[0][0] == entries[1][0]:
            return None
        normals: list[_Vector] = []
        directions: list[_Vector] = []
        for polygon, start, stop in entries:
            normal = self._polygon_normal(polygon)
            if normal is None:
                return None
            normals.append(normal)
            directions.append(_sub(self._points[stop], self._points[start]))
        line = _cross(normals[0], normals[1])
        if _is_zero(line):
            return None
        # Each polygon lies left of its directed loop edge: normal x direction.
        inward: list[_Vector] = []
        for normal, direction in zip(normals, directions, strict=True):
            orientation = _dot(line, direction)
            if orientation == 0:
                return None
            inward.append(_cross(normal, line if orientation > 0 else _negate(line)))
        regions: list[int] = []
        reflex: list[int] = []
        for (polygon, _, _), normal, other in zip(
            entries, normals, reversed(inward), strict=True
        ):
            side = _dot(normal, other)
            if side == 0:
                return None
            positive, negative = complex_.facet_regions[
                int(complex_.polygon_facets[polygon])
            ]
            regions.append(int(positive if side > 0 else negative))
            reflex.append(int(negative if side > 0 else positive))
        # The convex wedge spanned by both inward directions holds this region;
        # cell regions identify that wedge only when the reflex side differs.
        if regions[0] != regions[1] or regions[0] < 0 or regions[0] in reflex:
            return None
        return _Ridge(
            regions[0],
            _dot(inward[0], inward[1]),
            _dot(inward[0], inward[0]) * _dot(inward[1], inward[1]),
        )


def _eligible(
    complex_: PiecewiseLinearComplex, construction: VolumeConstruction, /
) -> bool:
    """Only an unmapped authored PLC's source planes are the represented input."""

    return (
        isinstance(construction.domain, PiecewiseLinearDomain)
        and complex_.geometry is None
        and complex_.cad_source_charts is None
        and complex_.complex_id == construction.complex_id
    )


def _native_cell_rows(construction: VolumeConstruction, /) -> dict[int, int]:
    return {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(construction.native_cell_ids).tolist()
        )
    }


def _segment_chain(construction: VolumeConstruction, /) -> dict[tuple[int, int], int]:
    """Recovered mesh segments keyed by sorted mesh vertices, to PLC edge rows."""

    complex_rows = np.asarray(construction.source_segment_complex_rows, dtype=np.int64)[
        np.asarray(construction.segment_source_edges, dtype=np.int64)
    ]
    return {
        (min(segment), max(segment)): int(edge_row)
        for segment, edge_row in zip(
            np.asarray(construction.segments, dtype=np.int64).tolist(),
            complex_rows.tolist(),
            strict=True,
        )
    }


def input_forced_dihedral_exemptions(
    complex_: PiecewiseLinearComplex,
    construction: VolumeConstruction,
    points: np.ndarray,
    target_degrees: float,
    rows: tuple[_QualityRow, ...],
    /,
) -> tuple[InputForcedDihedral, ...]:
    """Return the dihedral rows in ``rows`` proven forced by authored ridges.

    Only unmapped authored PLCs without a source coordinate map or CAD charts
    qualify: their source planes are the represented input. A row qualifies when
    its cell has an edge on the recovered chain of a PLC edge with exactly two
    planar polygon uses, the cell lies in that edge's convex interior wedge, the
    exact wedge angle is below ``target_degrees``, and every other cell edge has
    a dihedral proven at least ``target_degrees``.
    """

    if target_degrees <= 0.0 or not _eligible(complex_, construction):
        return ()
    candidates = [
        (index, row) for index, row in enumerate(rows) if row[0] == _DIHEDRAL_CRITERION
    ]
    if not candidates:
        return ()
    lower, upper = _target_cosine_bounds(float(target_degrees))
    geometry = _SourceGeometry(complex_)
    block = construction.mesh.blocks[0]
    cells = np.asarray(block.vertices, dtype=np.int64)
    cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    native_rows = _native_cell_rows(construction)
    chain = _segment_chain(construction)
    angles: dict[int, float] = {}
    exemptions: list[InputForcedDihedral] = []
    for index, (_, _, native_cell, value) in candidates:
        cell_row = native_rows.get(int(native_cell))
        if cell_row is None:
            continue
        cell = cells[cell_row].tolist()
        region = int(construction.cell_regions[cell_row])
        corners = tuple(_point(points[vertex]) for vertex in cell)
        forcing: list[int] = []
        proven = True
        for first, second in combinations(range(4), 2):
            edge_row = chain.get(
                (min(cell[first], cell[second]), max(cell[first], cell[second]))
            )
            ridge = None if edge_row is None else geometry.ridge(edge_row)
            if (
                edge_row is not None
                and ridge is not None
                and ridge.region == region
                and _cosine_exceeds(
                    ridge.cosine_numerator, ridge.cosine_denominator_squared, upper
                )
            ):
                forcing.append(edge_row)
                if edge_row not in angles:
                    angles[edge_row] = _angle_degrees(
                        ridge.cosine_numerator, ridge.cosine_denominator_squared
                    )
                continue
            third, fourth = (row for row in range(4) if row not in (first, second))
            axis = _sub(corners[second], corners[first])
            left = _cross(axis, _sub(corners[third], corners[first]))
            right = _cross(axis, _sub(corners[fourth], corners[first]))
            squared = _dot(left, left) * _dot(right, right)
            if squared == 0 or not _cosine_at_most(_dot(left, right), squared, lower):
                proven = False
                break
        if not proven or not forcing:
            continue
        exemptions.append(
            InputForcedDihedral(
                index,
                int(cell_ids[cell_row]),
                float(value),
                tuple(
                    ForcingRidge(
                        int(complex_.plc_edge_entity_ids[edge_row]),
                        complex_.plc_edge_ids[edge_row],
                        angles[edge_row],
                    )
                    for edge_row in forcing
                ),
            )
        )
    return tuple(exemptions)


def _in_corner(corner: _Corner, offset: _Vector, /) -> bool:
    """Exactly decide that the corner vertex plus ``offset`` lies in its closed cone."""

    normal = corner.normal
    return (
        not _is_zero(offset)
        and _dot(normal, offset) == 0
        and _dot(_cross(corner.next_direction, offset), normal) >= 0
        and _dot(_cross(offset, corner.previous_direction), normal) >= 0
    )


def _corner_forcing(corner: _Corner, bound: Fraction, /) -> tuple[float, float] | None:
    """Angle and ``1 / (2 sin alpha)`` when that ratio provably exceeds ``bound``."""

    cross = _cross(corner.next_direction, corner.previous_direction)
    cross_squared = _dot(cross, cross)
    lengths_squared = _dot(corner.next_direction, corner.next_direction) * _dot(
        corner.previous_direction, corner.previous_direction
    )
    # 1 / (2 sin alpha) > bound  <=>  4 bound^2 |u x v|^2 < |u|^2 |v|^2.
    if not 4 * bound * bound * cross_squared < lengths_squared:
        return None
    return (
        _angle_degrees(
            _dot(corner.next_direction, corner.previous_direction), lengths_squared
        ),
        _forced_ratio(lengths_squared, cross_squared),
    )


def input_forced_radius_edge_exemptions(
    complex_: PiecewiseLinearComplex,
    construction: VolumeConstruction,
    points: np.ndarray,
    radius_edge_bound: float | None,
    rows: tuple[_QualityRow, ...],
    /,
) -> tuple[InputForcedRadiusEdge, ...]:
    """Return the radius-edge rows in ``rows`` proven forced by authored facet corners.

    A row qualifies when its cell has a boundary face of a facet incident to an
    unmoved source vertex, the face lies exactly in the closed corner of one of
    that facet's planar polygons between two recovered PLC edges, and the
    corner's interior angle alpha is acute with ``1 / (2 sin alpha)`` exactly
    above ``radius_edge_bound``.
    """

    if radius_edge_bound is None or not _eligible(complex_, construction):
        return ()
    candidates = [
        (index, row) for index, row in enumerate(rows) if row[0] == _RADIUS_EDGE_CRITERION
    ]
    if not candidates:
        return ()
    bound = Fraction(float(radius_edge_bound))
    geometry = _SourceGeometry(complex_)
    constrained: dict[tuple[int, int], int] = {}
    for edge_row in np.unique(
        np.asarray(construction.source_segment_complex_rows, dtype=np.int64)
    ).tolist():
        first, second = complex_.plc_edge_vertices[edge_row]
        constrained[(min(first, second), max(first, second))] = int(edge_row)
    corners = geometry.corners(constrained)
    block = construction.mesh.blocks[0]
    cells = np.asarray(block.vertices, dtype=np.int64)
    cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    native_rows = _native_cell_rows(construction)
    faces = {
        tuple(sorted(face)): row
        for row, face in enumerate(
            np.asarray(construction.boundary_faces, dtype=np.int64).tolist()
        )
    }
    face_facets = np.asarray(construction.face_source_facets, dtype=np.int64)
    source_vertices = np.asarray(construction.input_vertices, dtype=np.int64)
    forcing: dict[tuple[int, int, int], tuple[float, float] | None] = {}
    exemptions: list[InputForcedRadiusEdge] = []
    for index, (_, _, native_cell, value) in candidates:
        cell_row = native_rows.get(int(native_cell))
        if cell_row is None:
            continue
        cell = cells[cell_row].tolist()
        records: list[ForcingCorner] = []
        for omitted in range(4):
            face = tuple(sorted(v for k, v in enumerate(cell) if k != omitted))
            face_row = faces.get(face)
            if face_row is None:
                continue
            facet = int(face_facets[face_row])
            for vertex in face:
                source_vertex = int(source_vertices[vertex])
                if source_vertex < 0:
                    continue
                apex = geometry.point(source_vertex)
                if _point(points[vertex]) != apex:
                    continue
                offsets = tuple(
                    _sub(_point(points[other]), apex) for other in face if other != vertex
                )
                for corner in corners.get(source_vertex, ()):
                    if corner.facet != facet or not all(
                        _in_corner(corner, offset) for offset in offsets
                    ):
                        continue
                    key = (source_vertex, *corner.edge_rows)
                    if key not in forcing:
                        forcing[key] = _corner_forcing(corner, bound)
                    evidence = forcing[key]
                    if evidence is None:
                        continue
                    edge_entities = sorted(
                        int(complex_.plc_edge_entity_ids[edge])
                        for edge in corner.edge_rows
                    )
                    records.append(
                        ForcingCorner(
                            int(complex_.vertex_entity_ids[source_vertex]),
                            complex_.vertex_ids[source_vertex],
                            (edge_entities[0], edge_entities[1]),
                            evidence[0],
                            evidence[1],
                            _triangle_ratio(np.asarray(points)[list(face)]),
                        )
                    )
        if records:
            exemptions.append(
                InputForcedRadiusEdge(
                    index, int(cell_ids[cell_row]), float(value), tuple(records)
                )
            )
    return tuple(exemptions)


__all__ = [
    "ForcingCorner",
    "ForcingRidge",
    "InputForcedDihedral",
    "InputForcedRadiusEdge",
    "input_forced_dihedral_exemptions",
    "input_forced_radius_edge_exemptions",
]
