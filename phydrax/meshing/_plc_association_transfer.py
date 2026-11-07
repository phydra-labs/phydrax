#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Carry authored piecewise-linear-complex strata through tetrahedral topology edits."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._meshcore import exact_orient3d
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..geometry._affine_plc_domain import AffinePlcDomain
from ..geometry._domain_coverage import PiecewiseLinearDomain
from ..geometry._planar_coverage import _sqrt_upper, PlcSourcePointWitness
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryFaceImageWitness,
)
from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
from ..geometry.surface import SurfaceMetadata, SurfaceModel
from ..typing import checked
from ._association import (
    AssociationEntityClasses,
    AssociationPropagationError,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
)
from ._association_routes import (
    associations_for_namespace,
    entity_rows,
    incidence_pairs,
    ordered_vertices,
    require_transfer_lineage,
    target_dimension,
)
from ._certification_inputs import ScopedFidelityInput
from ._lineage import MeshLineage, VertexInterpolationStencil
from ._result import CellMeshingResult
from ._topology_edit import key_rows
from .providers._native_plc_geometry import simplex_volume_entities
from .providers._native_sources import NativePlcSource


_ROLES = (
    GeometrySourceEntityRole.VERTEX,
    GeometrySourceEntityRole.EDGE,
    GeometrySourceEntityRole.FACET,
    GeometrySourceEntityRole.REGION,
)
_ROLE_DIMENSIONS = {role: dimension for dimension, role in enumerate(_ROLES)}
_TETRAHEDRA = ("tetrahedron", "simplex:3")
# Local faces opposite vertices 0..3; row ``k`` omits vertex ``k``.
_OPPOSITE_TETRAHEDRON_FACES = np.asarray(
    ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)), dtype=np.int64
)

type _ExactPoint = tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class _Classified:
    """Source stratum, PLC row, residual and orientation of every mesh entity row."""

    dimensions: np.ndarray
    indices: np.ndarray
    residuals: np.ndarray
    orientations: np.ndarray


@dataclass(frozen=True, slots=True)
class _PlcStrata:
    """Physical geometry and identity banks of one bound PLC source.

    ``vertices`` are the binary64 physical vertices and ``exact_vertices``
    the exact points they realize (equal for an untransformed PLC).
    """

    rows: tuple[dict[tuple[str, tuple[str, ...]], int], ...]
    names: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    vertices: np.ndarray
    exact_vertices: tuple[_ExactPoint, ...]
    edges: np.ndarray
    edge_by_pair: dict[tuple[int, int], int]
    facet_normals: np.ndarray
    facet_loops: tuple[tuple[np.ndarray, ...], ...]
    facet_vertices: tuple[frozenset[int], ...]
    facet_sides: tuple[tuple[str | None, str | None], ...]
    edge_regions: tuple[frozenset[int], ...]
    vertex_regions: tuple[frozenset[int], ...]
    facet_regions: np.ndarray

    @property
    def offsets(self) -> np.ndarray:
        counts = np.asarray([len(bank) for bank in self.names], dtype=np.int64)
        return np.concatenate(([0], np.cumsum(counts)[:-1]))


def _plc_strata(
    source: NativePlcSource, domain: PiecewiseLinearDomain | AffinePlcDomain, /
) -> _PlcStrata:
    complex_ = source.complex
    names = (
        tuple(zip(complex_.vertex_ids, complex_.vertex_occurrence_paths, strict=True)),
        tuple(
            zip(complex_.plc_edge_ids, complex_.plc_edge_occurrence_paths, strict=True)
        ),
        tuple(zip(complex_.facet_ids, complex_.facet_occurrence_paths, strict=True)),
        tuple(zip(complex_.region_ids, complex_.region_occurrence_paths, strict=True)),
    )
    rows = tuple({key: row for row, key in enumerate(bank)} for bank in names)
    if any(len(lookup) != len(bank) for lookup, bank in zip(rows, names, strict=True)):
        raise ValueError("PLC identity banks must name each source entity once.")
    vertices = np.asarray(domain.vertices, dtype=np.float64)
    exact_vertices = (
        tuple(domain.exact_vertices)
        if isinstance(domain, AffinePlcDomain)
        else tuple(tuple(Fraction(value) for value in row) for row in vertices.tolist())
    )
    edges = np.asarray(complex_.plc_edge_vertices, dtype=np.int64).reshape((-1, 2))
    edge_by_pair = {
        (min(first, second), max(first, second)): row
        for row, (first, second) in enumerate(edges.tolist())
    }
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    loops = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    polygon_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
    facet_count = complex_.facet_count
    by_facet: list[list[np.ndarray]] = [[] for _ in range(facet_count)]
    for polygon, facet in enumerate(polygon_facets.tolist()):
        by_facet[facet].append(loops[offsets[polygon] : offsets[polygon + 1]])
    normals = np.zeros((facet_count, 3), dtype=np.float64)
    for facet, facet_loops in enumerate(by_facet):
        if not facet_loops:
            raise ValueError("Every PLC facet must own at least one polygon loop.")
        # The Newell sum over all loops is the right-hand facet normal; hole
        # loops are oppositely wound and subtract their own area. It orients
        # rows only; stratum membership is decided exactly.
        normal = np.zeros(3, dtype=np.float64)
        for loop in facet_loops:
            points = vertices[loop] - vertices[facet_loops[0][0]]
            following = np.roll(points, -1, axis=0)
            normal += np.sum(np.cross(points, following), axis=0)
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length == 0.0:
            raise ValueError("A PLC facet has no oriented area.")
        normals[facet] = normal / length
    facet_regions = np.asarray(complex_.facet_regions, dtype=np.int64)
    region_names = names[3]
    facet_sides = tuple(
        tuple(None if region < 0 else region_names[region][0] for region in pair)
        for pair in facet_regions.tolist()
    )

    def regions_of(facets: Iterable[int]) -> frozenset[int]:
        return frozenset(
            int(region)
            for facet in facets
            for region in facet_regions[int(facet)].tolist()
            if region >= 0
        )

    edge_regions = tuple(regions_of(facets) for facets in complex_.plc_edge_facets)
    vertex_facets: list[set[int]] = [set() for _ in range(vertices.shape[0])]
    for facet, facet_loops in enumerate(by_facet):
        for loop in facet_loops:
            for vertex in loop.tolist():
                vertex_facets[vertex].add(facet)
    return _PlcStrata(
        rows,
        names,
        vertices,
        exact_vertices,
        edges,
        edge_by_pair,
        normals,
        tuple(tuple(value) for value in by_facet),
        tuple(
            frozenset(int(vertex) for loop in facet_loops for vertex in loop.tolist())
            for facet_loops in by_facet
        ),
        tuple((first, second) for first, second in facet_sides),
        edge_regions,
        tuple(regions_of(facets) for facets in vertex_facets),
        facet_regions,
    )


def _subtract(first: _ExactPoint, second: _ExactPoint, /) -> _ExactPoint:
    return tuple(a - b for a, b in zip(first, second, strict=True))


def _dot(first: _ExactPoint, second: _ExactPoint, /) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _cross(first: _ExactPoint, second: _ExactPoint, /) -> _ExactPoint:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _on_closed_segment(
    point: _ExactPoint, start: _ExactPoint, stop: _ExactPoint, /
) -> bool:
    direction = _subtract(stop, start)
    relative = _subtract(point, start)
    if any(_cross(direction, relative)):
        return False
    projection = _dot(relative, direction)
    return 0 <= projection <= _dot(direction, direction)


@dataclass(frozen=True, slots=True)
class _ExactFacet:
    """Exact plane and dominant-axis projection of one planar PLC facet."""

    origin: _ExactPoint
    normal: _ExactPoint
    axes: tuple[int, int]
    loops: tuple[tuple[tuple[Fraction, Fraction], ...], ...]


def _exact_facet(strata: _PlcStrata, facet: int, /) -> _ExactFacet | None:
    """Exact facet plane, or ``None`` when the authored loops are not exactly planar."""
    loops = tuple(
        tuple(strata.exact_vertices[int(vertex)] for vertex in loop.tolist())
        for loop in strata.facet_loops[facet]
    )
    origin = loops[0][0]
    normal = (Fraction(0), Fraction(0), Fraction(0))
    for loop in loops:
        for index, corner in enumerate(loop):
            following = loop[(index + 1) % len(loop)]
            term = _cross(_subtract(corner, origin), _subtract(following, origin))
            normal = tuple(a + b for a, b in zip(normal, term, strict=True))
    if not any(normal) or any(
        _dot(normal, _subtract(corner, origin)) for loop in loops for corner in loop
    ):
        return None
    dropped = max(range(3), key=lambda axis: abs(normal[axis]))
    kept = tuple(axis for axis in range(3) if axis != dropped)
    first, second = kept
    return _ExactFacet(
        origin,
        normal,
        (first, second),
        tuple(
            tuple((corner[first], corner[second]) for corner in loop) for loop in loops
        ),
    )


def _in_closed_facet(facet: _ExactFacet, point: _ExactPoint, /) -> bool:
    """Exact closed-polygon membership: on the facet plane and inside or on a loop."""
    if _dot(facet.normal, _subtract(point, facet.origin)):
        return False
    x, y = point[facet.axes[0]], point[facet.axes[1]]
    inside = False
    for loop in facet.loops:
        for index, (ax, ay) in enumerate(loop):
            bx, by = loop[(index + 1) % len(loop)]
            if (
                (bx - ax) * (y - ay) == (by - ay) * (x - ax)
                and min(ax, bx) <= x <= max(ax, bx)
                and min(ay, by) <= y <= max(ay, by)
            ):
                return True
            if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                inside = not inside
    return inside


@dataclass(frozen=True, slots=True)
class _SourcePoints:
    """Exact source chart of every mesh vertex row whose carrier realizes one.

    ``kinds`` follow :class:`PlcSourcePointWitness` (``-1`` none, ``0`` PLC
    vertex, ``1`` witness segment, ``2`` witness triangle, ``3`` declared
    plane of a witness triangle). ``residuals`` are the exact carrier
    distances to those source points, rounded once.
    """

    kinds: np.ndarray
    entities: np.ndarray
    residuals: np.ndarray
    segment_pairs: tuple[tuple[int, int], ...]
    face_groups: np.ndarray
    facet_pairs: dict[int, frozenset[tuple[int, int]]]


def _source_points(
    strata: _PlcStrata,
    domain: PiecewiseLinearDomain | AffinePlcDomain,
    mesh: CellMesh,
    witness: PlcSourcePointWitness | None,
    /,
) -> _SourcePoints:
    """Bind exact source points to mesh vertices from the construction witness.

    Without a witness only carriers equal to a PLC vertex carry a source
    point. A witnessed point whose exact value the carrier does not round is
    discarded; such a vertex then needs exact membership of its carrier.
    """
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    count = coordinates.shape[0]
    kinds = np.full(count, -1, dtype=np.int64)
    entities = np.full(count, -1, dtype=np.int64)
    residuals = np.zeros(count, dtype=np.float64)
    if witness is None:
        lookup = {
            tuple(row): vertex for vertex, row in enumerate(strata.vertices.tolist())
        }
        for row, carrier in enumerate(coordinates.tolist()):
            vertex = lookup.get(tuple(carrier))
            if vertex is not None:
                kinds[row], entities[row] = 0, vertex
        segment_pairs: tuple[tuple[int, int], ...] = ()
        face_groups = np.zeros((0,), dtype=np.int64)
        facet_pairs: dict[int, frozenset[tuple[int, int]]] = {}
        exact = isinstance(domain, PiecewiseLinearDomain)
        if not exact:
            for row in np.flatnonzero(kinds == 0).tolist():
                residuals[row] = _carrier_distance(
                    coordinates[row], strata.exact_vertices[int(entities[row])]
                )
        return _SourcePoints(
            kinds, entities, residuals, segment_pairs, face_groups, facet_pairs
        )
    if witness.domain_id != domain.domain_id or not np.array_equal(
        np.asarray(witness.vertex_global_ids, dtype=np.int64),
        np.asarray(mesh.vertex_global_ids, dtype=np.int64),
    ):
        raise ValueError(
            "The PLC source-point witness binds another domain or mesh vertex bank."
        )
    segments = np.asarray(witness.segments, dtype=np.int64).reshape((-1, 2))
    faces = np.asarray(witness.faces, dtype=np.int64).reshape((-1, 3))
    face_groups = np.asarray(witness.face_groups, dtype=np.int64)
    planes = np.asarray(witness.face_planes, dtype=np.float64)
    pairs: dict[int, set[tuple[int, int]]] = {}
    for corners, group in zip(faces.tolist(), face_groups.tolist(), strict=True):
        for index, corner in enumerate(corners):
            following = corners[(index + 1) % 3]
            pairs.setdefault(int(group), set()).add(
                (min(corner, following), max(corner, following))
            )
    witness_kinds = np.asarray(witness.vertex_kinds, dtype=np.int64)
    witness_entities = np.asarray(witness.vertex_entities, dtype=np.int64)
    parameters = np.asarray(witness.vertex_parameters, dtype=np.float64)
    vertices = strata.exact_vertices
    untransformed = isinstance(domain, PiecewiseLinearDomain)
    deviations = witness.vertex_deviations
    for row in np.flatnonzero(witness_kinds >= 0).tolist():
        kind, entity = int(witness_kinds[row]), int(witness_entities[row])
        carrier = coordinates[row]
        if kind == 0 and untransformed:
            if np.array_equal(carrier, strata.vertices[entity]):
                kinds[row], entities[row] = 0, entity
            continue
        first, second = (Fraction(float(value)) for value in parameters[row])
        if kind == 0:
            point = vertices[entity]
        elif kind == 1:
            start, stop = (vertices[int(value)] for value in segments[entity])
            point = tuple(a + first * (b - a) for a, b in zip(start, stop, strict=True))
        elif kind == 2:
            a, b, c = (vertices[int(value)] for value in faces[entity])
            point = tuple(
                (1 - first - second) * x + first * y + second * z
                for x, y, z in zip(a, b, c, strict=True)
            )
        elif untransformed:
            origin, axis, other = (
                tuple(Fraction(float(value)) for value in plane_row)
                for plane_row in planes[entity]
            )
            point = tuple(
                o + first * u + second * v
                for o, u, v in zip(origin, axis, other, strict=True)
            )
        else:
            continue
        if deviations is None:
            if tuple(float(value) for value in point) != tuple(carrier.tolist()):
                continue
        else:
            offset = _subtract(tuple(Fraction(float(v)) for v in carrier), point)
            if _dot(offset, offset) > Fraction(float(deviations[row])) ** 2:
                continue
        kinds[row], entities[row] = kind, entity
        residuals[row] = _carrier_distance(carrier, point)
    return _SourcePoints(
        kinds,
        entities,
        residuals,
        tuple((min(pair), max(pair)) for pair in segments.tolist()),
        face_groups,
        {group: frozenset(value) for group, value in pairs.items()},
    )


def _rounding_box(carrier: tuple[float, ...], /) -> tuple[tuple[Fraction, Fraction], ...]:
    """Closed box of reals that round to a binary64 carrier (ties included)."""
    return tuple(
        (
            (Fraction(value) + Fraction(math.nextafter(value, -math.inf))) / 2,
            (Fraction(value) + Fraction(math.nextafter(value, math.inf))) / 2,
        )
        for value in carrier
    )


def _clip_polygon_to_box(
    corners: tuple[_ExactPoint, ...],
    box: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> bool:
    """Whether a convex polygon (or a segment, as two corners) meets a closed box."""
    polygon = list(corners)
    for axis, (low, high) in enumerate(box):
        for sign, limit in ((1, low), (-1, high)):
            clipped: list[_ExactPoint] = []
            for index, current in enumerate(polygon):
                following = polygon[(index + 1) % len(polygon)]
                inside = sign * (current[axis] - limit) >= 0
                ahead = sign * (following[axis] - limit) >= 0
                if inside:
                    clipped.append(current)
                if inside != ahead:
                    t = (limit - current[axis]) / (following[axis] - current[axis])
                    clipped.append(
                        tuple(a + t * (b - a) for a, b in zip(current, following))
                    )
            polygon = clipped
            if not polygon:
                return False
    return True


def _convex_loop(facet: _ExactFacet, /) -> bool:
    """Whether the single projected facet loop turns one way at every corner."""
    (loop,) = facet.loops
    turns = set()
    for index, (ax, ay) in enumerate(loop):
        bx, by = loop[(index + 1) % len(loop)]
        cx, cy = loop[(index + 2) % len(loop)]
        turn = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if turn:
            turns.add(turn > 0)
    return len(turns) == 1


def _carrier_distance(carrier: np.ndarray, point: _ExactPoint, /) -> float:
    offset = _subtract(tuple(Fraction(float(value)) for value in carrier), point)
    return math.sqrt(float(_dot(offset, offset)))


def _chart_in_stratum(
    strata: _PlcStrata, points: _SourcePoints, row: int, stratum: int, index: int, /
) -> bool:
    """Whether a vertex's exact source chart lies in one closed PLC stratum."""
    kind = int(points.kinds[row])
    entity = int(points.entities[row])
    if kind == 0:
        if stratum == 0:
            return entity == index
        if stratum == 1:
            return entity in strata.edges[index].tolist()
        return entity in strata.facet_vertices[index]
    if kind == 1:
        pair = points.segment_pairs[entity]
        if stratum == 1:
            return strata.edge_by_pair.get(pair) == index
        return stratum == 2 and pair in points.facet_pairs.get(index, frozenset())
    if kind in (2, 3):
        return stratum == 2 and int(points.face_groups[entity]) == index
    return False


@final
class PlcAssociationTransfer(StrictModule, NonTrainableState):
    """Carry an authored PLC namespace through tetrahedral refinement and coarsening.

    Every target entity inherits the vertex, edge, facet or region class of the
    lowest-dimensional source mesh entity that contains it in the exact
    topology lineage; coordinates never select a class. Each class is then
    checked against the source stratum it names: every vertex of the entity
    lies exactly in that closed vertex, segment or facet polygon, either as
    the correctly rounded carrier of its native construction source point
    (whose chart is that stratum or one of its boundary strata) or by exact
    rational membership of its carrier; facet sides and incident material
    regions agree, and oriented edge/face rows follow the authored segment
    and right-hand facet directions. Recorded residuals are the exact
    carrier distances to those source points, rounded once.
    """

    source: NativePlcSource
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(self, source: NativePlcSource, /) -> None:
        if source.mapped_domain is not None or source.complex.geometry is not None:
            raise ValueError(
                "Mapped PLC sources carry associations through their mapped-reference transfer."
            )
        self.source = source
        self.transfer_id = canonical_fingerprint(
            {"kind": "plc-association-transfer", "source_binding": source.binding_id}
        )

    @property
    def source_namespaces(self) -> tuple[tuple[str, str], ...]:
        return ((self.source.source_id, self.source.source_revision),)

    def _domain(
        self, result: CellMeshingResult, /
    ) -> PiecewiseLinearDomain | AffinePlcDomain:
        certification = result.certification
        domain = None if certification is None else certification.inputs.domain
        if not isinstance(domain, (PiecewiseLinearDomain, AffinePlcDomain)):
            raise ValueError(
                "PLC association transfer requires the accepted declared PLC domain."
            )
        source = self.source
        if (
            domain.source_id != source.source_id
            or tuple(domain.region_ids) != tuple(source.complex.region_ids)
            or np.asarray(domain.vertices).shape
            != np.asarray(source.complex.vertices).shape
            or (
                isinstance(domain, PiecewiseLinearDomain)
                and not np.array_equal(
                    np.asarray(domain.vertices), np.asarray(source.complex.vertices)
                )
            )
            or (
                isinstance(domain, AffinePlcDomain)
                and not np.array_equal(
                    np.asarray(domain.reference_domain.vertices),
                    np.asarray(source.complex.vertices),
                )
            )
        ):
            raise ValueError(
                "Accepted PLC domain does not declare this transfer's source complex."
            )
        return domain

    @staticmethod
    def _witness(result: CellMeshingResult, /) -> PlcSourcePointWitness | None:
        certification = result.certification
        witness = (
            None
            if certification is None
            else certification.inputs.source_coverage_witness
        )
        if witness is not None and not isinstance(witness, PlcSourcePointWitness):
            raise ValueError(
                "PLC association transfer requires PLC source-point coverage evidence."
            )
        return witness

    def _tables(self, source: CellMeshingResult, /) -> dict[int, GeometryAssociation]:
        mesh = source.mesh
        if mesh.ambient_dimension != 3 or mesh.topological_dimension != 3:
            raise ValueError(
                "PLC association transfer requires a tetrahedral volume carrier."
            )
        if not mesh.blocks or any(
            block.cell_kind not in _TETRAHEDRA for block in mesh.blocks
        ):
            raise ValueError(
                "PLC association transfer requires tetrahedral simplex blocks."
            )
        namespace = self.source_namespaces[0]
        if any(
            (value.source_id, value.source_revision) == namespace
            and value.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
            for value in source.associations
        ):
            raise ValueError("A PLC namespace cannot mix association kinds.")
        tables = associations_for_namespace(
            source, namespace, kinds=(GeometryAssociationKind.PIECEWISE_LINEAR,)
        )
        by_dimension = {target_dimension(mesh, value): value for value in tables}
        if set(by_dimension) != {0, 1, 2, 3}:
            raise ValueError(
                "PLC associations must classify every vertex, edge, face and cell."
            )
        return by_dimension

    def _classify(
        self,
        mesh: CellMesh,
        tables: dict[int, GeometryAssociation],
        strata: _PlcStrata,
        points: _SourcePoints,
        /,
    ) -> tuple[_Classified, ...]:
        """Resolve and verify every row against its named source stratum."""
        coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
        classified: list[_Classified] = []
        members: dict[tuple[int, int, int], float | None] = {}
        facets: dict[int, _ExactFacet | None] = {}
        for dimension in range(4):
            table = tables[dimension]
            count = mesh.entity_set(dimension).count
            rows = entity_rows(
                mesh, dimension, np.asarray(table.target_global_ids, dtype=np.int64)
            )
            if rows.size != count or np.any(rows < 0) or np.unique(rows).size != count:
                raise ValueError(
                    "PLC association rows must name each carrier entity once."
                )
            dimensions = np.empty(count, dtype=np.int64)
            indices = np.empty(count, dtype=np.int64)
            sides: list[tuple[str | None, str | None]] = [(None, None)] * count
            for association_row, mesh_row in enumerate(rows.tolist()):
                role = table.source_roles[association_row]
                stratum = _ROLE_DIMENSIONS.get(role) if role is not None else None
                if stratum is None or stratum < dimension:
                    raise ValueError(
                        "PLC association rows must name a source stratum containing their entity."
                    )
                key = (
                    table.source_entity_ids[association_row],
                    table.source_occurrence_paths[association_row],
                )
                index = strata.rows[stratum].get(key)
                if index is None:
                    raise ValueError(
                        f"PLC association names an entity outside its source complex: {key!r}."
                    )
                dimensions[mesh_row] = stratum
                indices[mesh_row] = index
                sides[mesh_row] = table.source_region_sides[association_row]
            for mesh_row in range(count):
                expected = (
                    strata.facet_sides[int(indices[mesh_row])]
                    if dimensions[mesh_row] == 2
                    else (None, None)
                )
                if tuple(sides[mesh_row]) != expected:
                    raise ValueError(
                        "PLC association region sides differ from the authored facet incidence."
                    )
            if dimension == 0:
                vertex_rows = np.arange(count, dtype=np.int64)[:, None]
            else:
                pairs = incidence_pairs(mesh, 0, dimension)
                order = np.lexsort((pairs[:, 0], pairs[:, 1]))
                vertex_rows = pairs[order, 0].reshape((count, dimension + 1))
            residuals = np.zeros(count, dtype=np.float64)
            violated = np.zeros(count, dtype=np.bool_)
            for mesh_row in np.flatnonzero(dimensions < 3).tolist():
                stratum, index = int(dimensions[mesh_row]), int(indices[mesh_row])
                for vertex in vertex_rows[mesh_row].tolist():
                    key = (vertex, stratum, index)
                    if key not in members:
                        members[key] = (
                            float(points.residuals[vertex])
                            if _chart_in_stratum(strata, points, vertex, stratum, index)
                            else self._exact_member(
                                strata, facets, coordinates[vertex], stratum, index
                            )
                        )
                    member = members[key]
                    if member is None:
                        violated[mesh_row] = True
                        break
                    residuals[mesh_row] = max(residuals[mesh_row], member)
            if np.any(violated):
                raise AssociationPropagationError(
                    "PLC association rows lie off the source strata they name.",
                    np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)[
                        violated
                    ],
                )
            orientations = self._orientations(
                mesh, coordinates, strata, dimension, dimensions, indices
            )
            classified.append(_Classified(dimensions, indices, residuals, orientations))
        self._check_regions(mesh, strata, classified)
        return tuple(classified)

    @staticmethod
    def _exact_member(
        strata: _PlcStrata,
        facets: dict[int, _ExactFacet | None],
        carrier: np.ndarray,
        stratum: int,
        index: int,
        /,
    ) -> float | None:
        """Whether an unwitnessed carrier rounds a point of the closed stratum.

        Decided exactly: the carrier itself lies in the stratum (residual
        zero), or the stratum meets the carrier's closed rounding box, whose
        outward-rounded half diagonal is then the residual. ``None`` refuses.
        """
        values = tuple(float(value) for value in carrier)
        point = tuple(Fraction(value) for value in values)
        box = _rounding_box(values)
        if stratum == 0:
            vertex = strata.exact_vertices[index]
            if point == vertex:
                return 0.0
            inside = all(low <= x <= high for x, (low, high) in zip(vertex, box))
            return _rounding_bound(values) if inside else None
        if stratum == 1:
            start, stop = (strata.exact_vertices[int(v)] for v in strata.edges[index])
            if _on_closed_segment(point, start, stop):
                return 0.0
            meets = _clip_polygon_to_box((start, stop), box)
            return _rounding_bound(values) if meets else None
        if index not in facets:
            facets[index] = _exact_facet(strata, index)
        facet = facets[index]
        if facet is None:
            return None
        if _in_closed_facet(facet, point):
            return 0.0
        loops = strata.facet_loops[index]
        if len(loops) != 1 or not _convex_loop(facet):
            return None
        corners = tuple(strata.exact_vertices[int(v)] for v in loops[0].tolist())
        return _rounding_bound(values) if _clip_polygon_to_box(corners, box) else None

    def _orientations(
        self,
        mesh: CellMesh,
        coordinates: np.ndarray,
        strata: _PlcStrata,
        dimension: int,
        dimensions: np.ndarray,
        indices: np.ndarray,
        /,
    ) -> np.ndarray:
        orientations = np.zeros(dimensions.shape, dtype=np.int8)
        if dimension not in (1, 2):
            return orientations
        ordered = ordered_vertices(mesh, dimension)
        if ordered is None:
            raise ValueError(
                "PLC association transfer requires canonical simplex entity order."
            )
        oriented = np.flatnonzero(dimensions == dimension)
        if not oriented.size:
            return orientations
        corners = coordinates[ordered[oriented]]
        if dimension == 1:
            endpoints = strata.edges[indices[oriented]]
            reference = (
                strata.vertices[endpoints[:, 1]] - strata.vertices[endpoints[:, 0]]
            )
            measure = np.einsum("ij,ij->i", corners[:, 1] - corners[:, 0], reference)
        else:
            normal = np.cross(
                corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
            )
            measure = np.einsum(
                "ij,ij->i", normal, strata.facet_normals[indices[oriented]]
            )
        if np.any(measure == 0.0) or not np.all(np.isfinite(measure)):
            raise AssociationPropagationError(
                "Oriented PLC rows must not degenerate against their source direction.",
                np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)[
                    oriented[measure == 0.0]
                ],
            )
        orientations[oriented] = np.where(measure > 0.0, 1, -1)
        return orientations

    def _check_regions(
        self, mesh: CellMesh, strata: _PlcStrata, classified: list[_Classified], /
    ) -> None:
        if np.any(classified[3].dimensions != 3):
            raise ValueError("Every PLC tetrahedron must retain one material region.")
        cell_regions = classified[3].indices
        for dimension in range(3):
            pairs = incidence_pairs(mesh, dimension, 3)
            level = classified[dimension]
            rows = pairs[:, 0]
            regions = cell_regions[pairs[:, 1]]
            strata_dimensions = level.dimensions[rows]
            strata_indices = level.indices[rows]
            allowed = np.empty(rows.size, dtype=np.bool_)
            interior = strata_dimensions == 3
            allowed[interior] = regions[interior] == strata_indices[interior]
            on_facet = strata_dimensions == 2
            facet_regions = strata.facet_regions[strata_indices[on_facet]]
            allowed[on_facet] = np.any(facet_regions == regions[on_facet, None], axis=1)
            for stratum, bank in ((1, strata.edge_regions), (0, strata.vertex_regions)):
                selected = np.flatnonzero(strata_dimensions == stratum)
                allowed[selected] = [
                    int(region) in bank[int(index)]
                    for region, index in zip(
                        regions[selected].tolist(),
                        strata_indices[selected].tolist(),
                        strict=True,
                    )
                ]
            if not np.all(allowed):
                raise AssociationPropagationError(
                    "PLC association classes disagree with incident material regions.",
                    np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)[
                        np.unique(rows[~allowed])
                    ],
                )

    def _source_state(
        self, source: CellMeshingResult, /
    ) -> tuple[dict[int, GeometryAssociation], _PlcStrata, tuple[_Classified, ...]]:
        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        domain = self._domain(source)
        strata = _plc_strata(self.source, domain)
        tables = self._tables(source)
        points = _source_points(strata, domain, source.mesh, self._witness(source))
        classified = self._classify(source.mesh, tables, strata, points)
        for dimension, level in enumerate(classified):
            table = tables[dimension]
            rows = entity_rows(
                source.mesh,
                dimension,
                np.asarray(table.target_global_ids, dtype=np.int64),
            )
            if not np.array_equal(
                np.asarray(table.orientations, dtype=np.int8), level.orientations[rows]
            ):
                raise ValueError(
                    "PLC association orientations differ from the authored source directions."
                )
        return tables, strata, classified

    def validate_source(self, source: CellMeshingResult, /) -> None:
        self._source_state(source)

    def classes(
        self, source: CellMeshingResult, /
    ) -> tuple[AssociationEntityClasses, ...]:
        _, strata, classified = self._source_state(source)
        offsets = strata.offsets
        return tuple(
            AssociationEntityClasses(
                level.dimensions.copy(),
                level.indices.copy(),
                np.ones(level.dimensions.shape, dtype=np.bool_),
                offsets[level.dimensions] + level.indices,
            )
            for level in classified
        )

    def remap_boundary(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
    ) -> SurfaceModel | None:
        """Carry the exterior boundary model of a PLC volume through its lineage.

        The source model must be exactly the exterior of the source carrier.
        Each target exterior face inherits the unique tag of its source face
        lineage parents and faces away from its one incident tetrahedron by
        the exact orientation predicate.
        """
        boundary = source.boundary
        if boundary is None:
            return None
        self.validate_source(source)
        if not isinstance(lineage, MeshLineage) or not isinstance(target, CellMesh):
            raise TypeError("lineage and target must be MeshLineage and CellMesh values.")
        metadata = boundary.metadata
        if (metadata.source_id, metadata.source_revision) != self.source_namespaces[0]:
            raise ValueError(
                "PLC boundary metadata must match its association namespace."
            )
        if (
            metadata.coordinate_contract.spatial_id
            != source.coordinate_contract.spatial_id
        ):
            raise ValueError("PLC boundary must preserve the source coordinate contract.")
        if (
            boundary.selections
            or boundary.interfaces
            or boundary.orientation_repair is not None
        ):
            raise ValueError(
                "PLC boundary remapping refuses selections, interfaces, or orientation repairs."
            )
        require_transfer_lineage(source.mesh, lineage, target, dimensions=(2,))
        source_block_ids = np.concatenate(
            [
                np.asarray(block.global_ids, dtype=np.int64)
                for block in boundary.mesh.blocks
            ]
        )
        source_exterior = np.asarray(
            source.mesh.entity_set(2).entity_ids, dtype=np.int64
        )[np.asarray(source.mesh.boundary_masks[2], dtype=np.bool_)]
        if not np.array_equal(np.sort(source_block_ids), np.sort(source_exterior)):
            raise ValueError(
                "PLC boundary model must be exactly the exterior of the source carrier."
            )
        source_tags = metadata.cell_tags
        if source_tags and len(source_tags) != source_block_ids.size:
            raise ValueError("PLC boundary tags must align with source boundary cells.")
        tag_by_id = (
            dict(zip(source_block_ids.tolist(), source_tags, strict=True))
            if source_tags
            else {}
        )
        _, all_faces = simplex_volume_entities(target)
        mask = np.asarray(target.boundary_masks[2], dtype=np.bool_)
        faces = np.asarray(all_faces[mask], dtype=np.int64)
        face_ids = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)[mask]
        if faces.size == 0:
            raise ValueError("PLC target volume must retain an exterior boundary.")
        cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in target.blocks]
        )
        local_faces = cells[:, _OPPOSITE_TETRAHEDRON_FACES].reshape((-1, 3))
        owners = key_rows(np.sort(local_faces, axis=1), np.sort(faces, axis=1))
        if np.any(owners < 0):
            raise ValueError("PLC target exterior faces must bound one cell.")
        coordinates = np.asarray(target.coordinates, dtype=np.float64)
        signs = exact_orient3d(
            coordinates[faces[:, 0]],
            coordinates[faces[:, 1]],
            coordinates[faces[:, 2]],
            coordinates[cells.reshape(-1)[owners]],
        )
        if np.any(signs == 0):
            raise ValueError("A PLC exterior face has a degenerate incident cell.")
        inward = signs > 0
        faces[inward] = faces[inward][:, (0, 2, 1)]
        record = lineage.entity_lineage(2)
        parents: dict[int, list[int]] = {}
        for parent, child in zip(
            np.asarray(record.source_global_ids, dtype=np.int64).tolist(),
            np.asarray(record.target_global_ids, dtype=np.int64).tolist(),
            strict=True,
        ):
            parents.setdefault(child, []).append(parent)
        target_tags: list[str] = []
        if source_tags:
            for identifier in face_ids.tolist():
                inherited = {
                    tag_by_id[parent]
                    for parent in parents.get(identifier, ())
                    if parent in tag_by_id
                }
                if not inherited and identifier in tag_by_id:
                    inherited = {tag_by_id[identifier]}
                if len(inherited) != 1:
                    raise ValueError(
                        "Each PLC target exterior face must inherit one source boundary tag."
                    )
                target_tags.append(next(iter(inherited)))
        used, inverse = np.unique(faces, return_inverse=True)
        return SurfaceModel.from_triangles(
            coordinates[used],
            inverse.reshape(faces.shape),
            SurfaceMetadata(
                source_id=metadata.source_id,
                source_revision=metadata.source_revision,
                coordinate_contract=metadata.coordinate_contract,
                provenance=(*metadata.provenance, lineage.lineage_id),
                cell_tags=tuple(target_tags),
            ),
            vertex_global_ids=np.asarray(target.vertex_global_ids, dtype=np.int64)[used],
            cell_global_ids=face_ids,
            numeric_version=target.numeric_version,
        )

    @checked
    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        vertex_stencil: VertexInterpolationStencil | None,
        geometry_transition: CellGeometryTransition | None,
        target_fidelity_witness: SourceBoundaryChartWitness
        | SourceBoundaryFaceImageWitness
        | None,
        target_scoped_fidelity: tuple[ScopedFidelityInput, ...],
        validated_fidelity: tuple[ValidatedSourceFidelityWitness, ...],
    ) -> tuple[GeometryAssociation, ...]:
        del geometry_transition, target_fidelity_witness
        del target_scoped_fidelity, validated_fidelity
        tables, strata, classified = self._source_state(source)
        if not isinstance(lineage, MeshLineage) or not isinstance(target, CellMesh):
            raise TypeError("lineage and target must be MeshLineage and CellMesh values.")
        if (
            target.ambient_dimension != 3
            or target.topological_dimension != 3
            or any(block.cell_kind not in _TETRAHEDRA for block in target.blocks)
        ):
            raise ValueError(
                "PLC association transfer requires a tetrahedral target carrier."
            )
        require_transfer_lineage(source.mesh, lineage, target, dimensions=(0, 1, 2, 3))
        source_rows = {
            dimension: {
                int(identifier): row
                for row, identifier in enumerate(
                    np.asarray(
                        source.mesh.entity_set(dimension).entity_ids, dtype=np.int64
                    ).tolist()
                )
            }
            for dimension in range(4)
        }
        parents = self._parents(lineage, target, source_rows, classified)
        namespace = self.source_namespaces[0]
        output: list[GeometryAssociation] = []
        target_tables: dict[int, GeometryAssociation] = {}
        for dimension in range(4):
            parent_dimensions, parent_ids = parents[dimension]
            identifiers = np.asarray(
                target.entity_set(dimension).entity_ids, dtype=np.int64
            )
            source_mesh_rows = np.asarray(
                [
                    source_rows[int(d)][int(p)]
                    for d, p in zip(
                        parent_dimensions.tolist(), parent_ids.tolist(), strict=True
                    )
                ],
                dtype=np.int64,
            )
            stratum = np.empty(identifiers.size, dtype=np.int64)
            index = np.empty(identifiers.size, dtype=np.int64)
            for d in range(4):
                selected = np.flatnonzero(parent_dimensions == d)
                if not selected.size:
                    continue
                level = classified[d]
                stratum[selected] = level.dimensions[source_mesh_rows[selected]]
                index[selected] = level.indices[source_mesh_rows[selected]]
            if np.any(stratum < dimension):
                raise AssociationPropagationError(
                    "A target entity inherits a source stratum of lower dimension.",
                    identifiers[stratum < dimension],
                )
            keys = tuple(
                strata.names[int(s)][int(i)]
                for s, i in zip(stratum.tolist(), index.tolist(), strict=True)
            )
            table = tables[dimension]
            association = GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                namespace[0],
                namespace[1],
                target.entity_set(dimension).entity_set_id,
                identifiers,
                tuple(key[0] for key in keys),
                np.zeros(identifiers.size, dtype=np.float64),
                exact=False,
                source_roles=tuple(_ROLES[int(value)] for value in stratum.tolist()),
                source_occurrence_paths=tuple(key[1] for key in keys),
                source_region_sides=tuple(
                    strata.facet_sides[int(i)] if s == 2 else (None, None)
                    for s, i in zip(stratum.tolist(), index.tolist(), strict=True)
                ),
                parent_dimensions=parent_dimensions.astype(np.int8),
                parent_ids=parent_ids,
                provenance=GeometryAssociationProvenance.LINEAGE,
                parent_association_id=table.association_id,
            )
            target_tables[dimension] = association
        # Target carriers are checked against the canonical renewal of the
        # construction's exact source points, the same evidence the target
        # certification recomputes.
        domain = self._domain(source)
        witness = self._witness(source)
        target_points = _source_points(
            strata,
            domain,
            target,
            None
            if witness is None
            else renew_plc_source_point_witness(witness, domain, target, vertex_stencil),
        )
        verified = self._classify(target, target_tables, strata, target_points)
        for dimension in range(4):
            draft = target_tables[dimension]
            level = verified[dimension]
            output.append(
                GeometryAssociation(
                    draft.association_kind,
                    draft.source_id,
                    draft.source_revision,
                    draft.target_entity_set_id,
                    draft.target_global_ids,
                    draft.source_entity_ids,
                    level.residuals,
                    exact=bool(np.all(level.residuals == 0.0)),
                    source_roles=draft.source_roles,
                    source_occurrence_paths=draft.source_occurrence_paths,
                    source_region_sides=draft.source_region_sides,
                    orientations=level.orientations,
                    parent_dimensions=draft.parent_dimensions,
                    parent_ids=draft.parent_ids,
                    provenance=GeometryAssociationProvenance.LINEAGE,
                    parent_association_id=draft.parent_association_id,
                )
            )
        return tuple(output)

    def _parents(
        self,
        lineage: MeshLineage,
        target: CellMesh,
        source_rows: dict[int, dict[int, int]],
        classified: tuple[_Classified, ...],
        /,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Lowest-dimensional source mesh entity containing every target entity.

        Retained and same-dimension lineage parents come first; an entity created
        inside a split entity takes the parents of the lowest-dimensional target
        entities containing it that have them. Several parents must carry one
        source class; the least parent ID is recorded.
        """
        same: dict[int, list[set[int]]] = {}
        for dimension in range(4):
            identifiers = np.asarray(
                target.entity_set(dimension).entity_ids, dtype=np.int64
            )
            record = lineage.entity_lineage(dimension)
            found: list[set[int]] = [set() for _ in range(identifiers.size)]
            target_rows = entity_rows(
                target, dimension, np.asarray(record.target_global_ids, dtype=np.int64)
            )
            for row, parent in zip(
                target_rows.tolist(),
                np.asarray(record.source_global_ids, dtype=np.int64).tolist(),
                strict=True,
            ):
                if row >= 0 and parent in source_rows[dimension]:
                    found[row].add(parent)
            for row, identifier in enumerate(identifiers.tolist()):
                if not found[row] and identifier in source_rows[dimension]:
                    found[row].add(identifier)
            same[dimension] = found
        output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for dimension in range(4):
            identifiers = np.asarray(
                target.entity_set(dimension).entity_ids, dtype=np.int64
            )
            parent_dimensions = np.full(identifiers.size, -1, dtype=np.int64)
            candidates: list[set[int]] = [set() for _ in range(identifiers.size)]
            for row, values in enumerate(same[dimension]):
                if values:
                    parent_dimensions[row] = dimension
                    candidates[row] = set(values)
            for higher in range(dimension + 1, 4):
                open_rows = parent_dimensions < 0
                if not np.any(open_rows):
                    break
                for low, high in incidence_pairs(target, dimension, higher).tolist():
                    if open_rows[low] and same[higher][high]:
                        candidates[low] |= same[higher][high]
                newly = open_rows & np.asarray(
                    [bool(values) for values in candidates], dtype=np.bool_
                )
                parent_dimensions[newly] = higher
            if np.any(parent_dimensions < 0):
                raise AssociationPropagationError(
                    "Target entities have no source ancestor in the topology lineage.",
                    identifiers[parent_dimensions < 0],
                )
            disagreeing = [
                row
                for row, values in enumerate(candidates)
                if len(
                    {
                        (
                            int(classified[level].dimensions[source_rows[level][value]]),
                            int(classified[level].indices[source_rows[level][value]]),
                        )
                        for level in (int(parent_dimensions[row]),)
                        for value in values
                    }
                )
                != 1
            ]
            if disagreeing:
                raise AssociationPropagationError(
                    "Target entities merge parents of distinct PLC source strata.",
                    identifiers[np.asarray(disagreeing, dtype=np.int64)],
                )
            output[dimension] = (
                parent_dimensions,
                np.asarray([min(values) for values in candidates], dtype=np.int64),
            )
        return output


type _Parameters = tuple[Fraction, Fraction]
type _Space = tuple[int, int]
_CORNER_PARAMETERS: tuple[_Parameters, ...] = (
    (Fraction(0), Fraction(0)),
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
)


def _float_neighbours(value: Fraction, /) -> tuple[float, ...]:
    centre = float(value)
    output = [centre]
    up = down = centre
    for _ in range(3):
        up = math.nextafter(up, math.inf)
        down = math.nextafter(down, -math.inf)
        output.extend((up, down))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class _WitnessTables:
    vertices: tuple[tuple[Fraction, ...], ...]
    segments: np.ndarray
    faces: np.ndarray
    planes: np.ndarray
    vertex_segments: dict[int, tuple[int, ...]]
    vertex_faces: dict[int, tuple[int, ...]]

    def point(self, space: _Space, parameters: _Parameters, /) -> tuple[Fraction, ...]:
        kind, entity = space
        first, second = parameters
        if kind == 1:
            start, stop = (self.vertices[int(value)] for value in self.segments[entity])
            return tuple(a + first * (b - a) for a, b in zip(start, stop, strict=True))
        if kind == 2:
            a, b, c = (self.vertices[int(value)] for value in self.faces[entity])
            return tuple(
                (1 - first - second) * x + first * y + second * z
                for x, y, z in zip(a, b, c, strict=True)
            )
        origin, axis, other = (
            tuple(Fraction(float(value)) for value in row) for row in self.planes[entity]
        )
        return tuple(
            o + first * e + second * f
            for o, e, f in zip(origin, axis, other, strict=True)
        )

    def admissible(self, space: _Space, parameters: _Parameters, /) -> bool:
        first, second = parameters
        if space[0] == 1:
            return 0 < first < 1
        if space[0] == 2:
            return first >= 0 and second >= 0 and first + second <= 1
        return True

    def spaces(
        self, kind: int, entity: int, parameters: _Parameters, /
    ) -> dict[_Space, _Parameters]:
        """Exact parameters of one witnessed point in every source chart containing it."""
        if kind == 0:
            output: dict[_Space, _Parameters] = {}
            for segment in self.vertex_segments.get(entity, ()):
                start = int(self.segments[segment][0])
                output[(1, segment)] = (
                    Fraction(0 if start == entity else 1),
                    Fraction(0),
                )
            for face in self.vertex_faces.get(entity, ()):
                corner = [int(value) for value in self.faces[face]].index(entity)
                output[(2, face)] = _CORNER_PARAMETERS[corner]
            return output
        if kind == 1:
            output = {(1, entity): parameters}
            start, stop = (int(value) for value in self.segments[entity])
            shared = set(self.vertex_faces.get(start, ())) & set(
                self.vertex_faces.get(stop, ())
            )
            t = parameters[0]
            for face in sorted(shared):
                corners = [int(value) for value in self.faces[face]]
                first = _CORNER_PARAMETERS[corners.index(start)]
                second = _CORNER_PARAMETERS[corners.index(stop)]
                output[(2, face)] = (
                    (1 - t) * first[0] + t * second[0],
                    (1 - t) * first[1] + t * second[1],
                )
            return output
        return {(kind, entity): parameters}

    def projection(
        self, space: _Space, carrier: tuple[Fraction, ...], /
    ) -> _Parameters | None:
        """Exact least-squares parameters of a carrier on one affine source chart."""
        base = self.point(space, (Fraction(0), Fraction(0)))
        first = tuple(
            a - b
            for a, b in zip(
                self.point(space, (Fraction(1), Fraction(0))), base, strict=True
            )
        )
        relative = tuple(a - b for a, b in zip(carrier, base, strict=True))
        if space[0] == 1:
            norm = sum(value * value for value in first)
            if norm == 0:
                return None
            return (
                sum(a * b for a, b in zip(relative, first, strict=True)) / norm,
                Fraction(0),
            )
        second = tuple(
            a - b
            for a, b in zip(
                self.point(space, (Fraction(0), Fraction(1))), base, strict=True
            )
        )
        g11 = sum(a * a for a in first)
        g12 = sum(a * b for a, b in zip(first, second, strict=True))
        g22 = sum(b * b for b in second)
        r1 = sum(a * b for a, b in zip(relative, first, strict=True))
        r2 = sum(a * b for a, b in zip(relative, second, strict=True))
        determinant = g11 * g22 - g12 * g12
        if determinant == 0:
            return None
        return ((r1 * g22 - r2 * g12) / determinant, (r2 * g11 - r1 * g12) / determinant)


def renew_plc_source_point_witness(
    witness: PlcSourcePointWitness,
    domain: PiecewiseLinearDomain | AffinePlcDomain,
    target: CellMesh,
    stencil: VertexInterpolationStencil | None,
    /,
) -> PlcSourcePointWitness:
    """Exact source points of a refined PLC target's vertices.

    Retained vertices keep their rows. A created vertex on the target boundary
    lies in the lowest source chart (segment, then triangle, then declared
    plane) shared by all of its exact stencil parents. Its parameters are
    binary64 values whose exact source point the actual carrier is, or rounds
    to; the exact stencil combination and the exact carrier projection are
    tried with their nearest binary64 neighbours.

    Nested refinement places a created carrier at the rounded combination of
    its parents' carriers; between rounded source points no chart point need
    round to it. Such a vertex takes the exact stencil combination of its
    parents' source points, which lies in their common chart, at the nearest
    admissible binary64 parameters, and records an outward-rounded deviation
    bound: the weighted parent bounds, its own carrier rounding and the
    parameter rounding. The witness then bounds every row (see
    :class:`PlcSourcePointWitness`). Certification recomputes every point and
    bound. A boundary vertex with neither is refused.
    """
    if not isinstance(witness, PlcSourcePointWitness):
        raise TypeError("witness must be PlcSourcePointWitness.")
    if isinstance(domain, AffinePlcDomain):
        raise ValueError(
            "Refined affine PLC source points are not admitted; the affine map is authored per source vertex."
        )
    segments = np.asarray(witness.segments, dtype=np.int64).reshape((-1, 2))
    faces = np.asarray(witness.faces, dtype=np.int64).reshape((-1, 3))
    groups = np.asarray(witness.face_groups, dtype=np.int64)
    vertex_segments: dict[int, list[int]] = {}
    for row, pair in enumerate(segments.tolist()):
        for vertex in pair:
            vertex_segments.setdefault(int(vertex), []).append(row)
    vertex_faces: dict[int, list[int]] = {}
    for row, corners in enumerate(faces.tolist()):
        for vertex in corners:
            vertex_faces.setdefault(int(vertex), []).append(row)
    tables = _WitnessTables(
        tuple(tuple(Fraction(float(value)) for value in row) for row in domain.vertices),
        segments,
        faces,
        np.asarray(witness.face_planes, dtype=np.float64),
        {key: tuple(value) for key, value in vertex_segments.items()},
        {key: tuple(value) for key, value in vertex_faces.items()},
    )
    source_deviations = witness.vertex_deviations
    source_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(witness.vertex_global_ids).tolist())
    }
    parents: dict[int, list[tuple[int, Fraction]]] = {}
    if stencil is not None:
        valid = np.asarray(stencil.valid, dtype=np.bool_)
        for row, identifier in enumerate(np.asarray(stencil.target_global_ids).tolist()):
            terms = []
            for column in np.flatnonzero(valid[row]).tolist():
                weight = stencil.exact_weights[row][column]
                if not isinstance(weight, Fraction):
                    raise ValueError(
                        "PLC source-point renewal requires rational stencils."
                    )
                terms.append((int(stencil.source_global_ids[row, column]), weight))
            parents[int(identifier)] = terms
    from ._association_routes import boundary_mask

    on_boundary = boundary_mask(target, 0)
    coordinates = np.asarray(target.coordinates, dtype=np.float64)
    identifiers = np.asarray(target.vertex_global_ids, dtype=np.int64)
    kinds = np.full(identifiers.size, -1, dtype=np.int64)
    entities = np.full(identifiers.size, -1, dtype=np.int64)
    parameters = np.zeros((identifiers.size, 2), dtype=np.float64)
    refused: list[int] = []
    bounded: dict[int, float] = {}

    def parent_bound(parent_row: int, point: tuple[Fraction, ...], /) -> Fraction:
        if source_deviations is not None:
            return Fraction(float(source_deviations[parent_row]))
        return Fraction(_rounding_bound(tuple(float(value) for value in point)))

    for row, identifier in enumerate(identifiers.tolist()):
        source_row = source_rows.get(identifier)
        if source_row is not None:
            kinds[row] = witness.vertex_kinds[source_row]
            entities[row] = witness.vertex_entities[source_row]
            parameters[row] = witness.vertex_parameters[source_row]
            continue
        if not on_boundary[row]:
            continue
        terms = parents.get(identifier, [])
        if not terms or sum(weight for _, weight in terms) != 1:
            refused.append(identifier)
            continue
        charts: list[dict[_Space, _Parameters]] = []
        points: list[tuple[Fraction, ...]] = []
        for parent, _ in terms:
            parent_row = source_rows.get(parent)
            if parent_row is None or int(witness.vertex_kinds[parent_row]) < 0:
                charts = []
                break
            kind = int(witness.vertex_kinds[parent_row])
            entity = int(witness.vertex_entities[parent_row])
            first, second = (
                Fraction(float(value)) for value in witness.vertex_parameters[parent_row]
            )
            charts.append(tables.spaces(kind, entity, (first, second)))
            points.append(
                tables.vertices[entity]
                if kind == 0
                else tables.point((kind, entity), (first, second))
            )
        if not charts:
            refused.append(identifier)
            continue
        # One source facet may be several coplanar source triangles: a point
        # combined from parents on one facet lies in one of its triangles.
        common_groups = set.intersection(
            *(
                {int(groups[face]) for kind, face in chart if kind == 2}
                for chart in charts
            )
        )
        candidates = set.intersection(*(set(chart) for chart in charts)) | {
            (2, face)
            for face in range(faces.shape[0])
            if int(groups[face]) in common_groups
        }
        combined = tuple(
            sum(
                (
                    weight * point[axis]
                    for (_, weight), point in zip(terms, points, strict=True)
                ),
                Fraction(0),
            )
            for axis in range(len(points[0]))
        )
        carrier = tuple(Fraction(float(value)) for value in coordinates[row])
        chosen: tuple[_Space, tuple[float, float]] | None = None
        rounded = False
        for space in sorted(candidates):
            bases = [
                value
                for value in (
                    tables.projection(space, combined),
                    tables.projection(space, carrier),
                )
                if value is not None
            ]
            for base in bases:
                for first in _float_neighbours(base[0]):
                    seconds = (0.0,) if space[0] == 1 else _float_neighbours(base[1])
                    for second in seconds:
                        candidate = (Fraction(first), Fraction(second))
                        if not tables.admissible(space, candidate):
                            continue
                        exact = tables.point(space, candidate)
                        if (
                            exact == carrier
                            or tuple(Fraction(float(value)) for value in exact) == carrier
                        ):
                            chosen = (space, (first, second))
                            break
                    if chosen is not None:
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                rounded = True
                break
            # The lowest chart holding the exact parent combination wins over
            # an exactly rounded point of a higher chart.
            nested = _nested_source_point(tables, (space,), combined)
            if nested is not None:
                chosen = (nested[0], nested[1])
                break
        if chosen is None:
            refused.append(identifier)
            continue
        point = tables.point(chosen[0], (Fraction(chosen[1][0]), Fraction(chosen[1][1])))
        # The exact nested geometry corner of a created vertex is the stencil
        # combination of its parents' exact corners. Parents of a correctly
        # rounded witness have binary64 corners, the rounded source points.
        corner = (
            None
            if source_deviations is not None
            else tuple(
                sum(
                    (
                        weight * Fraction(float(parent_point[axis]))
                        for (_, weight), parent_point in zip(terms, points, strict=True)
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            )
        )
        if not rounded or corner != carrier:
            total = sum(
                (
                    abs(weight) * parent_bound(source_rows[parent], parent_point)
                    for (parent, weight), parent_point in zip(terms, points, strict=True)
                ),
                Fraction(0),
            ) + Fraction(_rounding_bound(tuple(coordinates[row].tolist())))
            offset = _subtract(combined, point)
            total += Fraction(_sqrt_upper(_dot(offset, offset)))
            bound = _float_upper(total)
            carrier_offset = _subtract(carrier, point)
            if _dot(carrier_offset, carrier_offset) > Fraction(bound) ** 2:
                refused.append(identifier)
                continue
            bounded[row] = bound
        (kinds[row], entities[row]), parameters[row] = chosen[0], chosen[1]
    if refused:
        raise AssociationPropagationError(
            "Refined PLC boundary vertices have no exactly rounded or bounded source point.",
            np.asarray(refused, dtype=np.int64),
        )
    deviations = None
    if bounded or source_deviations is not None:
        deviations = np.zeros(identifiers.size, dtype=np.float64)
        for row in np.flatnonzero(kinds >= 0).tolist():
            source_row = source_rows.get(int(identifiers[row]))
            if row in bounded:
                deviations[row] = bounded[row]
            elif source_row is not None and source_deviations is not None:
                deviations[row] = source_deviations[source_row]
            else:
                deviations[row] = _rounding_bound(tuple(coordinates[row].tolist()))
    return PlcSourcePointWitness(
        identifiers,
        kinds,
        entities,
        parameters,
        segments=witness.segments,
        faces=witness.faces,
        face_groups=witness.face_groups,
        face_planes=witness.face_planes,
        face_plane_tolerances=witness.face_plane_tolerances,
        domain_face_rows=witness.domain_face_rows,
        domain_id=witness.domain_id,
        vertex_deviations=deviations,
    )


def _rounding_bound(carrier: tuple[float, ...], /) -> float:
    """Outward bound on the distance from a carrier to any point rounding to it."""
    squared = Fraction(0)
    for value in carrier:
        gap = max(
            Fraction(math.nextafter(value, math.inf)) - Fraction(value),
            Fraction(value) - Fraction(math.nextafter(value, -math.inf)),
        )
        squared += (gap / 2) ** 2
    return _sqrt_upper(squared)


def _float_upper(value: Fraction, /) -> float:
    result = float(value)
    return result if Fraction(result) >= value else math.nextafter(result, math.inf)


def _nested_source_point(
    tables: _WitnessTables,
    candidates: Iterable[_Space],
    combined: tuple[Fraction, ...],
    /,
) -> tuple[_Space, tuple[float, float], tuple[Fraction, ...]] | None:
    """Nearest admissible binary64 chart point to an exact combination in that chart."""
    for space in sorted(candidates):
        base = tables.projection(space, combined)
        if (
            base is None
            or not tables.admissible(space, base)
            or tables.point(space, base) != combined
        ):
            continue
        for first in _float_neighbours(base[0]):
            seconds = (0.0,) if space[0] == 1 else _float_neighbours(base[1])
            for second in seconds:
                candidate = (Fraction(first), Fraction(second))
                if tables.admissible(space, candidate):
                    return space, (first, second), tables.point(space, candidate)
    return None
