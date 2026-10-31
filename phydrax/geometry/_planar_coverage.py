#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact overlap proofs between oriented source and candidate facet partitions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import final, Literal

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..typing import as_host_array, Dim, HostFloat64, HostInt64, Identifier, Scope
from ._domain_coverage import PiecewiseLinearDomain
from ._planar_primitives import (
    area2,
    intersection,
    plane_key,
    plane_orientation,
    project_polygon,
    projection_axes,
    segment_overlap,
)


type OverlapSemantics = Literal["exact", "candidate"]
type CoordinateSpace = Literal["physical", "reference"]
type RegionPair = tuple[int, int]
type ExactPoint = tuple[Fraction, ...]
# Cell reference family, global vertex IDs and exact carrier corners.
type CellCorners = tuple[str, tuple[int, ...], tuple[ExactPoint, ...]]


@dataclass(frozen=True, slots=True)
class OrientedFacet:
    facet_id: int
    region_pair: RegionPair
    vertices: tuple[tuple[Fraction, ...], ...]


@dataclass(frozen=True, slots=True)
class FacetOverlap:
    source_facet_id: int
    target_facet_id: int
    region_pair: RegionPair
    measure: Fraction
    semantics: OverlapSemantics
    coordinate_space: CoordinateSpace


def source_facets(domain: PiecewiseLinearDomain, /) -> tuple[OrientedFacet, ...]:
    points = tuple(
        tuple(Fraction(float(value)) for value in row) for row in domain.vertices
    )
    return tuple(
        OrientedFacet(
            facet_id,
            (
                int(domain.facet_regions[facet_id, 0]),
                int(domain.facet_regions[facet_id, 1]),
            ),
            tuple(points[int(index)] for index in domain.facets[facet_id, :]),
        )
        for facet_id in range(domain.facets.shape[0])
    )


class SourceFacetIndex:
    """Source facets bucketed by their exact canonical supporting plane.

    Two 3-D facets overlap with positive area only when their plane keys are
    equal, and a facet without a plane key overlaps nothing, so a 3-D query
    needs only the facets of its own plane. Segment (2-D) queries scan them all.
    """

    __slots__ = ("_by_plane", "facets")

    def __init__(self, facets: tuple[OrientedFacet, ...], /) -> None:
        by_plane: dict[tuple[int, int, int, int], list[int]] = {}
        for row, facet in enumerate(facets):
            try:
                key = plane_key(facet.vertices)
            except ValueError:
                continue
            by_plane.setdefault(key, []).append(row)
        self.facets = facets
        self._by_plane = {key: tuple(rows) for key, rows in by_plane.items()}

    def candidates(
        self, vertices: tuple[tuple[Fraction, ...], ...], dimension: int, /
    ) -> tuple[int, ...]:
        """Rows, in source order, of every facet that can overlap ``vertices``."""
        if dimension == 2:
            return tuple(range(len(self.facets)))
        try:
            key = plane_key(vertices)
        except ValueError:
            return ()
        return self._by_plane.get(key, ())

    def coplanar_rows(self) -> tuple[tuple[int, ...], ...]:
        """Source-ordered row groups that share one supporting plane."""
        return tuple(self._by_plane.values())


def _facet_measure(facet: OrientedFacet, dimension: int, /) -> Fraction:
    if dimension == 2:
        first, second = facet.vertices
        delta = (second[0] - first[0], second[1] - first[1])
        axis = max(range(2), key=lambda value: abs(delta[value]))
        return abs(delta[axis])
    plane = plane_key(facet.vertices)
    return area2(project_polygon(facet.vertices, projection_axes(plane))) / 2


def _orientation_matches(
    first: OrientedFacet, second: OrientedFacet, dimension: int, /
) -> bool:
    if dimension == 2:
        first_direction = tuple(
            second_value - first_value
            for first_value, second_value in zip(
                first.vertices[0], first.vertices[1], strict=True
            )
        )
        second_direction = tuple(
            second_value - first_value
            for first_value, second_value in zip(
                second.vertices[0], second.vertices[1], strict=True
            )
        )
        return (
            sum(
                left * right
                for left, right in zip(first_direction, second_direction, strict=True)
            )
            > 0
        )
    return plane_orientation(first.vertices) == plane_orientation(second.vertices)


def _overlap(first: OrientedFacet, second: OrientedFacet, dimension: int, /) -> Fraction:
    if dimension == 2:
        first_start, first_end = ((x, y) for x, y in first.vertices)
        second_start, second_end = ((x, y) for x, y in second.vertices)
        return segment_overlap((first_start, first_end), (second_start, second_end))
    try:
        first_plane = plane_key(first.vertices)
        if first_plane != plane_key(second.vertices):
            return Fraction(0)
    except ValueError:
        return Fraction(0)
    axes = projection_axes(first_plane)
    first_polygon = project_polygon(first.vertices, axes)
    second_polygon = project_polygon(second.vertices, axes)
    return area2(intersection(first_polygon, second_polygon)) / 2


def partition_overlaps(
    domain: PiecewiseLinearDomain,
    targets: tuple[OrientedFacet, ...],
    /,
    *,
    semantics: OverlapSemantics = "exact",
    coordinate_space: CoordinateSpace = "physical",
    sources: tuple[OrientedFacet, ...] | None = None,
) -> tuple[tuple[FacetOverlap, ...], tuple[tuple[str, int, int], ...]]:
    """Match both complete oriented facet chains; return exact witnesses/findings.

    Each accepted overlap has exact rational measure. Containment of each target
    facet and equality of every source-facet overlap sum, together with the
    positive embedding premise, proves source coverage and excludes missing or
    repeated pieces. No total-volume-only inference is made. ``sources`` is the
    domain's facet chain realized in declared-plane charts
    (``PlcSourcePoints.sources``); by default the domain's own facets.
    """
    if not isinstance(domain, PiecewiseLinearDomain):
        raise TypeError("domain must be PiecewiseLinearDomain.")
    dimension = domain.ambient_dimension
    if sources is None:
        sources = source_facets(domain)
    elif len(sources) != domain.facets.shape[0] or any(
        facet.facet_id != row for row, facet in enumerate(sources)
    ):
        raise ValueError("sources must realize every domain facet in row order.")
    index = SourceFacetIndex(sources)
    overlaps: list[FacetOverlap] = []
    findings: list[tuple[str, int, int]] = []
    source_totals = [Fraction(0) for _ in sources]

    groups = (tuple(range(len(sources))),) if dimension == 2 else index.coplanar_rows()
    overlapping_pairs = sorted(
        (rows[first], rows[second])
        for rows in groups
        for first in range(len(rows))
        for second in range(first + 1, len(rows))
        if _overlap(sources[rows[first]], sources[rows[second]], dimension)
    )
    findings.extend(
        ("overlapping_source_facets", sources[first].facet_id, sources[second].facet_id)
        for first, second in overlapping_pairs
    )

    for target in targets:
        target_measure = _facet_measure(target, dimension)
        if target_measure <= 0:
            findings.append(("degenerate_target_facet", target.facet_id, -1))
            continue
        matched = Fraction(0)
        reverse = Fraction(0)
        for source_index in index.candidates(target.vertices, dimension):
            source = sources[source_index]
            overlap = _overlap(source, target, dimension)
            if overlap <= 0:
                continue
            if source.region_pair == target.region_pair and _orientation_matches(
                source, target, dimension
            ):
                matched += overlap
                source_totals[source_index] += overlap
                overlaps.append(
                    FacetOverlap(
                        source.facet_id,
                        target.facet_id,
                        target.region_pair,
                        overlap,
                        semantics,
                        coordinate_space,
                    )
                )
            elif source.region_pair == target.region_pair[::-1] or (
                source.region_pair == target.region_pair
                and not _orientation_matches(source, target, dimension)
            ):
                reverse += overlap
        if reverse:
            check = (
                "boundary_orientation"
                if -1 in target.region_pair
                else "interface_orientation"
            )
            findings.append((check, target.facet_id, -1))
        if matched != target_measure:
            findings.append(("unmatched_boundary_facet", target.facet_id, -1))

    for source_index, source in enumerate(sources):
        expected = _facet_measure(source, dimension)
        if source_totals[source_index] == 0:
            findings.append(("uncovered_boundary", source.facet_id, -1))
        elif source_totals[source_index] != expected:
            findings.append(("double_or_partial_source_coverage", source.facet_id, -1))

    return tuple(overlaps), tuple(findings)


class _WitnessVertexDim(Dim, minimum=1):
    """Mesh vertices addressed by one source-point witness."""


class _WitnessSegmentDim(Dim):
    """Source segments addressed by edge source points."""


class _WitnessFaceDim(Dim, minimum=1):
    """Source triangles addressed by facet and plane source points."""


class _WitnessDomainFacetDim(Dim, minimum=1):
    """Coverage-domain facets realized by source triangles."""


@final
class PlcSourcePointWitness(StrictModule, NonTrainableState):
    """Exact source points of a PLC mesh's vertices for exact chain coverage.

    Row ``r`` addresses mesh vertex ``vertex_global_ids[r]``. Its kind is
    ``-1`` (no source point), ``0`` (domain vertex ``entity``), ``1``
    (``p + t (q - p)`` on ``segments[entity]``), ``2`` (``(1 - u - v) a + u b
    + v c`` on ``faces[entity]``) or ``3`` (``o + s e + t f`` on the declared
    plane ``face_planes[entity]`` of rows origin, first and second axis; an
    all-NaN row is undeclared), with ``(t, -)``, ``(u, v)`` or ``(s, t)`` in
    ``vertex_parameters``. Source triangles sharing a ``face_groups`` value
    form one source facet; ``domain_face_rows`` names the source triangle
    realizing each coverage-domain facet. The exact source point is
    authoritative and the mesh carrier must be its correctly rounded binary64
    realization; certification recomputes both from these parameters.
    """

    __strict_contract__ = True

    vertex_global_ids: HostInt64[_WitnessVertexDim]
    vertex_kinds: HostInt64[_WitnessVertexDim]
    vertex_entities: HostInt64[_WitnessVertexDim]
    vertex_parameters: HostFloat64[_WitnessVertexDim, Literal[2]]
    segments: HostInt64[_WitnessSegmentDim, Literal[2]]
    faces: HostInt64[_WitnessFaceDim, Literal[3]]
    face_groups: HostInt64[_WitnessFaceDim]
    face_planes: HostFloat64[_WitnessFaceDim, Literal[3], Literal[3]]
    face_plane_tolerances: HostFloat64[_WitnessFaceDim]
    domain_face_rows: HostInt64[_WitnessDomainFacetDim]
    domain_id: Identifier = eqx.field(static=True)
    witness_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        vertex_global_ids: ArrayLike,
        vertex_kinds: ArrayLike,
        vertex_entities: ArrayLike,
        vertex_parameters: ArrayLike,
        /,
        *,
        segments: ArrayLike,
        faces: ArrayLike,
        face_groups: ArrayLike,
        face_planes: ArrayLike,
        face_plane_tolerances: ArrayLike,
        domain_face_rows: ArrayLike,
        domain_id: str,
    ) -> None:
        scope = Scope()
        ids = as_host_array(
            vertex_global_ids,
            HostInt64[_WitnessVertexDim],
            "vertex_global_ids",
            scope=scope,
        )
        kinds = as_host_array(
            vertex_kinds, HostInt64[_WitnessVertexDim], "vertex_kinds", scope=scope
        )
        entities = as_host_array(
            vertex_entities,
            HostInt64[_WitnessVertexDim],
            "vertex_entities",
            scope=scope,
        )
        parameters = as_host_array(
            vertex_parameters,
            HostFloat64[_WitnessVertexDim, Literal[2]],
            "vertex_parameters",
            scope=scope,
        )
        segment_rows = as_host_array(
            segments, HostInt64[_WitnessSegmentDim, Literal[2]], "segments", scope=scope
        )
        face_rows = as_host_array(
            faces, HostInt64[_WitnessFaceDim, Literal[3]], "faces", scope=scope
        )
        groups = as_host_array(
            face_groups, HostInt64[_WitnessFaceDim], "face_groups", scope=scope
        )
        planes = as_host_array(
            face_planes,
            HostFloat64[_WitnessFaceDim, Literal[3], Literal[3]],
            "face_planes",
            scope=scope,
        )
        tolerances = as_host_array(
            face_plane_tolerances,
            HostFloat64[_WitnessFaceDim],
            "face_plane_tolerances",
            scope=scope,
        )
        domain_rows = as_host_array(
            domain_face_rows,
            HostInt64[_WitnessDomainFacetDim],
            "domain_face_rows",
            scope=scope,
        )
        arrays = tuple(
            np.array(value, copy=True)
            for value in (
                ids,
                kinds,
                entities,
                parameters,
                segment_rows,
                face_rows,
                groups,
                planes,
                tolerances,
                domain_rows,
            )
        )
        _require_witness_tables(*arrays)
        identity = canonical_identifier(domain_id, "domain_id")
        for array in arrays:
            array.setflags(write=False)
        (
            self.vertex_global_ids,
            self.vertex_kinds,
            self.vertex_entities,
            self.vertex_parameters,
            self.segments,
            self.faces,
            self.face_groups,
            self.face_planes,
            self.face_plane_tolerances,
            self.domain_face_rows,
        ) = arrays
        self.domain_id = identity
        self.witness_id = canonical_fingerprint(
            {
                "kind": "plc-source-point-witness",
                "domain": identity,
                "tables": tuple(array_tree_fingerprint(array) for array in arrays),
            }
        )


def _require_witness_tables(
    ids: np.ndarray,
    kinds: np.ndarray,
    entities: np.ndarray,
    parameters: np.ndarray,
    segments: np.ndarray,
    faces: np.ndarray,
    groups: np.ndarray,
    planes: np.ndarray,
    tolerances: np.ndarray,
    domain_rows: np.ndarray,
    /,
) -> None:
    if np.any(ids < 0) or np.unique(ids).size != ids.size:
        raise ValueError("vertex_global_ids must be distinct nonnegative IDs.")
    if not np.all(np.isin(kinds, (-1, 0, 1, 2, 3))):
        raise ValueError("vertex_kinds must be -1, 0, 1, 2 or 3.")
    if not np.all(np.isfinite(parameters)):
        raise ValueError("vertex_parameters must be finite.")
    if np.any(segments < 0) or np.any(faces < 0) or np.any(groups < 0):
        raise ValueError("Source segment, triangle and facet rows must be nonnegative.")
    declared = ~np.any(np.isnan(planes), axis=(1, 2))
    if not np.all(np.isnan(planes[~declared])) or not np.all(
        np.isfinite(planes[declared])
    ):
        raise ValueError("face_planes rows must be finite or entirely NaN.")
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0.0):
        raise ValueError("face_plane_tolerances must be finite and nonnegative.")
    if (
        np.any(domain_rows < 0)
        or np.any(domain_rows >= faces.shape[0])
        or np.unique(domain_rows).size != domain_rows.size
    ):
        raise ValueError("domain_face_rows must name distinct source triangles.")
    limits = {0: None, 1: segments.shape[0], 2: faces.shape[0], 3: faces.shape[0]}
    for kind, limit in limits.items():
        selected = entities[kinds == kind]
        if np.any(selected < 0) or (limit is not None and np.any(selected >= limit)):
            raise ValueError(f"Kind-{kind} source points name an absent source row.")
    if np.any(entities[kinds == -1] != -1):
        raise ValueError("Vertices without a source point must name entity -1.")
    if not np.all(declared[entities[kinds == 3]]):
        raise ValueError("Plane source points must name a declared plane row.")


class SourcePointRefusal(ValueError):
    """A retained source-point witness does not establish an exact mesh point."""

    def __init__(
        self,
        check: str,
        detail: str,
        /,
        *,
        entity_ids: tuple[int, ...] = (),
        status: Literal["violated", "unresolved"] = "violated",
    ) -> None:
        super().__init__(detail)
        self.check = check
        self.entity_ids = entity_ids
        self.status = status


@dataclass(frozen=True, slots=True)
class PlcSourcePoints:
    """Validated exact source points and declared-plane charts of one mesh.

    ``points`` maps each vertex whose exact source point differs from its
    carrier to that point. ``charts`` maps each declared source facet whose
    triangles leave its plane to the plane origin and normal; the facet's
    source triangles (``sources``) and target facets are compared after exact
    orthogonal projection onto it. ``cell_measures`` are exact measures of the
    target cells with substituted vertices.
    """

    points: Mapping[int, ExactPoint]
    incidence: Mapping[int, frozenset[int]]
    charts: Mapping[int, tuple[ExactPoint, ExactPoint]]
    sources: tuple[OrientedFacet, ...]
    cell_measures: Mapping[int, Fraction]
    carrier_deviation_upper: float
    chart_deviation_upper: float

    def substitute(
        self, vertices: tuple[int, ...], points: tuple[ExactPoint, ...], /
    ) -> tuple[ExactPoint, ...]:
        """Replace carriers by the exact source points they round."""
        return tuple(
            self.points.get(vertex, point)
            for vertex, point in zip(vertices, points, strict=True)
        )

    def chart(
        self, vertices: tuple[int, ...], points: tuple[ExactPoint, ...], /
    ) -> tuple[ExactPoint, ...]:
        """Project a target facet onto the chart of the one source facet holding it."""
        groups: frozenset[int] | None = None
        for vertex in vertices:
            incident = self.incidence.get(vertex, frozenset())
            groups = incident if groups is None else groups & incident
        held = frozenset() if groups is None else groups
        charted = tuple(group for group in held if group in self.charts)
        if not charted:
            return points
        if len(held) != 1:
            raise SourcePointRefusal(
                "source_point_facet_ambiguous",
                "A target facet's source points lie on more than one source facet.",
                status="unresolved",
            )
        origin, normal = self.charts[charted[0]]
        return tuple(_project(point, origin, normal) for point in points)


def _dot(first: ExactPoint, second: ExactPoint, /) -> Fraction:
    return sum(
        (left * right for left, right in zip(first, second, strict=True)), Fraction(0)
    )


def _project(point: ExactPoint, origin: ExactPoint, normal: ExactPoint, /) -> ExactPoint:
    height = _dot(normal, tuple(a - b for a, b in zip(point, origin, strict=True)))
    if height == 0:
        return point
    scale = height / _dot(normal, normal)
    return tuple(value - scale * axis for value, axis in zip(point, normal, strict=True))


def _sqrt_upper(value: Fraction, /) -> float:
    """Smallest binary64 value whose square is at least ``value``, within one step."""
    result = math.sqrt(float(value))
    while Fraction(result) ** 2 < value:
        result = math.nextafter(result, math.inf)
    return result


def _carriers(cells: tuple[CellCorners, ...], /) -> dict[int, ExactPoint]:
    carriers: dict[int, ExactPoint] = {}
    for _, vertices, corners in cells:
        for vertex, corner in zip(vertices, corners, strict=True):
            if carriers.setdefault(vertex, corner) != corner:
                raise SourcePointRefusal(
                    "source_point_carrier",
                    "Cells disagree on the exact carrier of a shared vertex.",
                    entity_ids=(vertex,),
                )
    return carriers


def _source_point(
    witness: PlcSourcePointWitness,
    row: int,
    vertices: tuple[ExactPoint, ...],
    /,
) -> ExactPoint:
    kind = int(witness.vertex_kinds[row])
    entity = int(witness.vertex_entities[row])
    first, second = (Fraction(float(value)) for value in witness.vertex_parameters[row])
    vertex_id = (int(witness.vertex_global_ids[row]),)
    match kind:
        case 0:
            if entity >= len(vertices):
                raise SourcePointRefusal(
                    "source_point_entity",
                    "A vertex source point names an absent domain vertex.",
                    entity_ids=vertex_id,
                )
            return vertices[entity]
        case 1:
            if not 0 < first < 1:
                raise SourcePointRefusal(
                    "source_point_parameter",
                    "A segment source point parameter leaves the open source segment.",
                    entity_ids=vertex_id,
                )
            start, stop = (vertices[int(index)] for index in witness.segments[entity])
            return tuple(a + first * (b - a) for a, b in zip(start, stop, strict=True))
        case 2:
            if first < 0 or second < 0 or first + second > 1:
                raise SourcePointRefusal(
                    "source_point_parameter",
                    "A facet source point's barycentric parameters leave its triangle.",
                    entity_ids=vertex_id,
                )
            a, b, c = (vertices[int(index)] for index in witness.faces[entity])
            return tuple(
                (1 - first - second) * x + first * y + second * z
                for x, y, z in zip(a, b, c, strict=True)
            )
        case 3:
            origin, axis, other = (
                tuple(Fraction(float(value)) for value in plane_row)
                for plane_row in witness.face_planes[entity]
            )
            return tuple(
                o + first * u + second * v
                for o, u, v in zip(origin, axis, other, strict=True)
            )
        case _:
            raise ValueError("Source-point kinds were validated at construction.")


def _rounding_box_meets_plane(
    origin: ExactPoint, normal: ExactPoint, carrier: tuple[float, ...], /
) -> bool:
    """Whether the plane meets the set of reals rounding to ``carrier``."""
    offset = _dot(
        normal,
        tuple(Fraction(value) - o for value, o in zip(carrier, origin, strict=True)),
    )
    low = high = offset
    for axis, value in zip(normal, carrier, strict=True):
        down = (Fraction(value) - Fraction(math.nextafter(value, -math.inf))) / 2
        up = (Fraction(math.nextafter(value, math.inf)) - Fraction(value)) / 2
        if axis > 0:
            low, high = low - axis * down, high + axis * up
        elif axis < 0:
            low, high = low + axis * up, high - axis * down
    return low <= 0 <= high


def _declared_charts(
    witness: PlcSourcePointWitness,
    domain: PiecewiseLinearDomain,
    vertices: tuple[ExactPoint, ...],
    facet_groups: tuple[int, ...],
    /,
) -> tuple[dict[int, tuple[ExactPoint, ExactPoint]], Fraction]:
    """Charts of declared source facets whose triangles leave the plane.

    Every corner must meet its facet's declared plane within the declared
    tolerance or inside its own binary64 rounding box; the largest squared
    corner distance bounds the chart identification.
    """
    rows_by_group: dict[int, list[int]] = {}
    for facet, group in enumerate(facet_groups):
        rows_by_group.setdefault(group, []).append(facet)
    charts: dict[int, tuple[ExactPoint, ExactPoint]] = {}
    largest = Fraction(0)
    for group, facets in rows_by_group.items():
        face_rows = tuple(int(witness.domain_face_rows[facet]) for facet in facets)
        plane = witness.face_planes[face_rows[0]]
        if any(
            not np.array_equal(witness.face_planes[row], plane, equal_nan=True)
            for row in face_rows
        ):
            raise SourcePointRefusal(
                "source_point_plane",
                "Source triangles of one facet declare different planes.",
                entity_ids=tuple(facets),
            )
        if np.isnan(plane[0, 0]):
            continue
        origin, axis, other = (
            tuple(Fraction(float(value)) for value in plane_row) for plane_row in plane
        )
        normal = (
            axis[1] * other[2] - axis[2] * other[1],
            axis[2] * other[0] - axis[0] * other[2],
            axis[0] * other[1] - axis[1] * other[0],
        )
        norm = _dot(normal, normal)
        if norm == 0:
            raise SourcePointRefusal(
                "source_point_plane",
                "A declared source plane has dependent axes.",
                entity_ids=tuple(facets),
            )
        tolerance = Fraction(float(witness.face_plane_tolerances[face_rows[0]]))
        corners = sorted(
            {int(index) for facet in facets for index in domain.facets[facet]}
        )
        group_largest = Fraction(0)
        for corner in corners:
            point = vertices[corner]
            height = _dot(
                normal, tuple(a - b for a, b in zip(point, origin, strict=True))
            )
            squared = height * height / norm
            if squared > tolerance * tolerance and not _rounding_box_meets_plane(
                origin, normal, tuple(float(value) for value in domain.vertices[corner])
            ):
                raise SourcePointRefusal(
                    "source_point_plane_admission",
                    "A source facet corner leaves its declared plane beyond the "
                    "declared tolerance and its rounding box.",
                    entity_ids=(corner,),
                )
            group_largest = max(group_largest, squared)
        if group_largest > 0:
            charts[group] = (origin, normal)
            largest = max(largest, group_largest)
    return charts, largest


def _substituted_cell_measures(
    cells: tuple[CellCorners, ...], points: Mapping[int, ExactPoint], /
) -> dict[int, Fraction]:
    measures: dict[int, Fraction] = {}
    for row, (kind, vertices, corners) in enumerate(cells):
        if not any(vertex in points for vertex in vertices):
            continue
        placed = tuple(
            points.get(vertex, corner)
            for vertex, corner in zip(vertices, corners, strict=True)
        )
        base = placed[0]
        edges = tuple(
            tuple(value - origin for value, origin in zip(point, base, strict=True))
            for point in placed[1:]
        )
        if kind == "triangle" and len(placed) == 3:
            (a, b), (c, d) = edges
            measures[row] = (a * d - b * c) / 2
        elif kind == "tetrahedron" and len(placed) == 4:
            (a, b, c), (d, e, f), (g, h, i) = edges
            measures[row] = (
                a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
            ) / 6
        else:
            raise SourcePointRefusal(
                "source_point_cell_kind",
                "Exact source points substitute only simplex corners.",
                entity_ids=(row,),
                status="unresolved",
            )
    return measures


def resolve_plc_source_points(
    witness: PlcSourcePointWitness,
    domain: PiecewiseLinearDomain,
    vertex_global_ids: np.ndarray,
    cells: tuple[CellCorners, ...],
    /,
) -> PlcSourcePoints:
    """Recompute every exact source point and require carriers to round it.

    ``cells`` are the target cells' exact carrier corners in cell order. The
    returned carrier deviation bounds the distance between each carrier and
    its exact source point; the chart deviation bounds the normal distance
    between a declared-plane chart and its source triangles, twice the
    largest corner distance to the declared plane.
    """
    if not isinstance(witness, PlcSourcePointWitness):
        raise TypeError("witness must be PlcSourcePointWitness.")
    if witness.domain_id != domain.domain_id or not np.array_equal(
        witness.vertex_global_ids, vertex_global_ids
    ):
        raise SourcePointRefusal(
            "source_point_witness_binding",
            "The source-point witness binds another domain or mesh vertex bank.",
        )
    if domain.ambient_dimension != 3 or not np.array_equal(
        witness.faces[witness.domain_face_rows], domain.facets
    ):
        raise SourcePointRefusal(
            "source_point_witness_binding",
            "Witness source triangles do not realize the coverage-domain facets.",
        )
    vertex_count = domain.vertices.shape[0]
    if np.any(witness.segments >= vertex_count) or np.any(witness.faces >= vertex_count):
        raise SourcePointRefusal(
            "source_point_witness_binding",
            "Witness source rows name absent domain vertices.",
        )
    vertices = tuple(
        tuple(Fraction(float(value)) for value in row) for row in domain.vertices
    )
    facet_groups = tuple(
        int(witness.face_groups[row]) for row in witness.domain_face_rows.tolist()
    )
    domain_groups = frozenset(facet_groups)
    vertex_groups: dict[int, set[int]] = {}
    edge_groups: dict[tuple[int, int], set[int]] = {}
    for facet, group in enumerate(facet_groups):
        corners = tuple(int(value) for value in domain.facets[facet])
        for corner in corners:
            vertex_groups.setdefault(corner, set()).add(group)
        for index, corner in enumerate(corners):
            following = corners[(index + 1) % 3]
            edge = (min(corner, following), max(corner, following))
            edge_groups.setdefault(edge, set()).add(group)
    carriers = _carriers(cells)
    points: dict[int, ExactPoint] = {}
    incidence: dict[int, frozenset[int]] = {}
    deviation = Fraction(0)
    for row, kind in enumerate(witness.vertex_kinds.tolist()):
        if kind < 0:
            continue
        vertex = int(witness.vertex_global_ids[row])
        exact = _source_point(witness, row, vertices)
        carrier = carriers.get(vertex)
        if carrier is None or carrier != tuple(Fraction(float(value)) for value in exact):
            raise SourcePointRefusal(
                "source_point_rounding",
                "A mesh carrier is not the correctly rounded exact source point.",
                entity_ids=(vertex,),
            )
        if exact != carrier:
            points[vertex] = exact
            deviation = max(
                deviation,
                sum(
                    ((a - b) ** 2 for a, b in zip(carrier, exact, strict=True)),
                    Fraction(0),
                ),
            )
        entity = int(witness.vertex_entities[row])
        if kind == 0:
            groups = vertex_groups.get(entity, set())
        elif kind == 1:
            start, stop = (int(value) for value in witness.segments[entity])
            groups = edge_groups.get((min(start, stop), max(start, stop)), set())
        else:
            groups = {int(witness.face_groups[entity])} & domain_groups
        incidence[vertex] = frozenset(groups)
    charts, chart_squared = _declared_charts(witness, domain, vertices, facet_groups)
    sources = tuple(
        facet
        if facet_groups[facet.facet_id] not in charts
        else OrientedFacet(
            facet.facet_id,
            facet.region_pair,
            tuple(
                _project(point, *charts[facet_groups[facet.facet_id]])
                for point in facet.vertices
            ),
        )
        for facet in source_facets(domain)
    )
    return PlcSourcePoints(
        points,
        incidence,
        charts,
        sources,
        _substituted_cell_measures(cells, points),
        _sqrt_upper(deviation),
        _sqrt_upper(4 * chart_squared),
    )


__all__ = [
    "CellCorners",
    "CoordinateSpace",
    "ExactPoint",
    "FacetOverlap",
    "OrientedFacet",
    "OverlapSemantics",
    "PlcSourcePoints",
    "PlcSourcePointWitness",
    "RegionPair",
    "SourceFacetIndex",
    "SourcePointRefusal",
    "partition_overlaps",
    "resolve_plc_source_points",
    "source_facets",
]
