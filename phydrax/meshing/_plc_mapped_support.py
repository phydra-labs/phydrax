#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact source support of Q1 and restricted mixed coordinate-map images.

Association identities are inputs, never outputs of geometric searches. Positive
maps, global embedding and complete oriented PLC coverage establish whole-cell
region support; source-polynomial restrictions establish edge and face support.
All uncertain enclosures and exhausted resources fail closed.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike

from .._meshcore import charge_native_geometry_queries
from ..discretization import _coordinate_enclosure as algebra, CellMesh
from ..discretization._cell_complex import PolyhedralConnectivity, TetrahedralConnectivity
from ..discretization._cell_geometry import (
    _require_scalar_coordinate_element,
    CellGeometryElement,
    CellGeometrySpec,
    RestrictedCellGeometryElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from ..discretization._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._mapped_coverage import (
    _children,
    face_charts,
    nonnegative,
    SubdivisionLedger,
)
from ..geometry._mesh_certificates import (
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
    MeshCertificateLimits,
    PiecewiseLinearDomain,
)
from ..geometry._planar_coverage import (
    _affine_segment_containment,
    _affine_simplex_corner_containment,
    _AffineSourceSimplex,
    _prepare_affine_exact_source_simplex,
    _reserve_fraction_work,
    convex_hull,
    intersection,
    mapped_containment,
    project,
    rational_points,
    signed_measure,
    source_groups,
    SourceGroup,
    turn,
)


if TYPE_CHECKING:
    from ._association import GeometryAssociation, PlcAssociationTransfer
    from ._certification import MeshCertificationPreparedEvidence
    from ._certification_inputs import MeshCertificationInputs
    from ._organization import MeshLabel, MeshPatch, MeshZone
    from ._result import CellMeshingResult


@dataclass(frozen=True)
class _CellMap:
    kind: str
    vertices: tuple[int, ...]
    coordinates: tuple[algebra.Expression, ...] | None
    element: CellGeometryElement
    controls: algebra.CoordinateSourceBank
    root_cell_id: int
    root_vertex_ids: tuple[int, ...]


type _Point = tuple[Fraction, ...]


def _cell_maps(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    *,
    prepare_expressions: bool = True,
) -> tuple[tuple[_CellMap, ...], tuple[_Point, ...]]:
    """Exact cell maps and the exact image of every carrier vertex, in row order.

    The coordinate maps are the geometry. Each published binary64 carrier corner
    must equal, bit for bit, the correctly rounded image of its exact map corner,
    and every incident cell must map the vertex to the same exact point.
    """
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError("Mapped PLC support requires a three-dimensional volume mesh.")
    from contextlib import nullcontext

    ledger = algebra._COORDINATE_BUDGET.get()
    elements, routes, _ = geometry.resolve(mesh)
    controls = algebra.prepared_coordinate_source_bank(geometry)
    carrier = np.ascontiguousarray(mesh.coordinates, dtype=np.float64).view(np.uint64)
    identity_geometry = geometry
    record = identity_geometry.restriction_source
    while record is None and identity_geometry.periodic_source is not None:
        parent_geometry = identity_geometry.periodic_source.source_geometry
        if not isinstance(parent_geometry, CellGeometrySpec):
            raise TypeError(
                "Periodic PLC root ownership requires the canonical CellGeometrySpec."
            )
        identity_geometry = parent_geometry
        record = identity_geometry.restriction_source
    global_vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    images: list[_Point | None] = [None] * carrier.shape[0]
    cells: list[_CellMap] = []
    local_controls: dict[tuple[int, ...], algebra.CoordinateSourceBank] = {}
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        element = _require_scalar_coordinate_element(element, "Mapped PLC support")
        if (
            block.cell_kind not in ("tetrahedron", "hexahedron", "prism", "pyramid")
            or element.degree != 1
        ):
            raise ValueError(
                "Mapped PLC support requires canonical Q1 or restricted degree-one mixed maps."
            )
        root = element
        while isinstance(root, RestrictedCellGeometryElement):
            root = root.source_element
        if root.degree != 1:
            raise ValueError(
                "Restricted PLC maps require an actual degree-one root expression."
            )
        local_values: list[algebra.CoordinateSourceBank] = []
        for row in np.asarray(route, dtype=np.int64):
            if ledger is not None:
                # Tuple slots, Python int64-sized integers and the route index.
                ledger.reserve(row.size, 128 + 40 * row.size)
            key = tuple(row.tolist())
            local = local_controls.get(key)
            if local is None:
                local = tuple(controls[index] for index in key)
                local_controls[key] = local
            local_values.append(local)
        parent_ids = (
            np.asarray(block.global_ids, dtype=np.int64)
            if record is None
            else np.asarray(record.block_parent_cell_ids[block.name], dtype=np.int64)
        )
        parent_vertices = (
            None
            if record is None
            else np.asarray(record.block_parent_vertex_ids[block.name], dtype=np.int64)
        )
        for cell_row, (local, vertices) in enumerate(
            zip(local_values, np.asarray(block.vertices, dtype=np.int64), strict=True)
        ):
            vertices = np.asarray(vertices, dtype=np.int64).reshape(-1)
            # Sparse source-basis corner weights evaluate the complete actual
            # map chain, including rational/pyramid charts, without forming
            # unused interior physical polynomials.
            polynomial = None
            with nullcontext() if ledger is None else ledger.temporary_scope():
                if prepare_expressions:
                    polynomial = algebra.coordinate_expressions(element, local)
                    if polynomial is None:
                        raise ValueError(
                            "Mapped PLC coordinate source expression is unresolved."
                        )
                corners = algebra.coordinate_corner_images(element, local)
            if corners is None:
                raise ValueError("Mapped PLC coordinate corner expression is unresolved.")
            if ledger is not None:
                ledger.reserve(sum(len(image) for image in corners))
            for vertex, image in zip(vertices.tolist(), corners, strict=True):
                known = images[vertex]
                if known is None:
                    if not np.array_equal(
                        algebra.rounded_point(image).view(np.uint64), carrier[vertex]
                    ):
                        raise ValueError(
                            "PLC carrier corners are not the correctly rounded images of their exact coordinate maps."
                        )
                    images[vertex] = image
                elif known != image:
                    raise ValueError(
                        "Incident PLC coordinate maps disagree on an exact shared corner."
                    )
            roots = (
                tuple(int(identifier) for identifier in global_vertices[vertices])
                if parent_vertices is None
                else tuple(
                    int(identifier)
                    for identifier in parent_vertices[cell_row]
                    if identifier >= 0
                )
            )
            cells.append(
                _CellMap(
                    block.cell_kind,
                    tuple(vertices.tolist()),
                    polynomial,
                    element,
                    local,
                    int(parent_ids[cell_row]),
                    roots,
                )
            )
    exact: list[_Point] = []
    for image in images:
        if image is None:
            raise ValueError(
                "A PLC carrier vertex is not a corner of any coordinate-mapped cell."
            )
        exact.append(image)
    return tuple(cells), tuple(exact)


def _exact_affine_plc_validity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cells: tuple[_CellMap, ...],
    vertex_images: tuple[_Point, ...],
    /,
) -> CellValidityCertificate | None:
    """Certify constant exact tetrahedron Jacobians without polynomial materialization."""
    if not isinstance(
        geometry.exact_source,
        (ExactPlcCellGeometrySource, ExactPlcCellGeometryConvexSource),
    ) or any(cell.kind != "tetrahedron" for cell in cells):
        return None
    ledger = algebra._COORDINATE_BUDGET.get()
    if ledger is None:
        raise RuntimeError("Exact affine PLC validity lost its coefficient ledger.")
    policy = CellValidityPolicy()
    floor = Fraction(policy.relative_determinant_floor)
    status = np.empty((len(cells),), dtype=np.int32)
    lower = np.empty((len(cells),), dtype=np.float64)
    upper = np.empty((len(cells),), dtype=np.float64)
    prepared: dict[tuple[tuple[Fraction, ...], ...], tuple[Fraction, Fraction]] = {}
    for row, cell in enumerate(cells):
        corners = tuple(vertex_images[vertex] for vertex in cell.vertices)
        ledger.reserve(sum(len(point) for point in corners) + 9)
        columns = tuple(
            tuple(corners[column + 1][axis] - corners[0][axis] for column in range(3))
            for axis in range(3)
        )
        values = prepared.get(columns)
        if values is None:
            bits = max(
                abs(value.numerator).bit_length() + value.denominator.bit_length()
                for column in columns
                for value in column
            )
            algebra._reserve_polynomial(34, 16, 0, 3 * bits + 4)
            determinant = (
                columns[0][0]
                * (columns[1][1] * columns[2][2] - columns[1][2] * columns[2][1])
                - columns[0][1]
                * (columns[1][0] * columns[2][2] - columns[1][2] * columns[2][0])
                + columns[0][2]
                * (columns[1][0] * columns[2][1] - columns[1][1] * columns[2][0])
            )
            squared_scale = Fraction(
                math.prod(
                    sum(
                        (columns[axis][column] ** 2 for axis in range(3)),
                        Fraction(0),
                    )
                    for column in range(3)
                )
            )
            values = determinant, squared_scale
            prepared[columns] = values
        determinant, squared_scale = values
        status[row] = (
            CellValidityStatus.CERTIFIED_VALID
            if determinant > 0
            and determinant * determinant > floor * floor * squared_scale
            else CellValidityStatus.INVALID
        )
        lower[row] = algebra.outward(determinant, -math.inf)
        upper[row] = algebra.outward(determinant, math.inf)
    offsets = [0]
    for block in mesh.blocks:
        offsets.append(offsets[-1] + block.vertices.shape[0])
    return CellValidityCertificate(
        status,
        lower,
        upper,
        np.zeros((len(cells),), dtype=np.int32),
        block_names=tuple(block.name for block in mesh.blocks),
        block_offsets=tuple(offsets),
        unsupported_block_names=(),
        unresolved_reasons=(),
        geometry_id=cell_geometry_id(geometry),
        geometry_layout_id=geometry.geometry_layout_id,
        topology_id=mesh.topology_id,
        policy_id=policy.policy_id,
    )


def _cell_coordinates(cell: _CellMap, /) -> tuple[algebra.Expression, ...]:
    if cell.coordinates is not None:
        return cell.coordinates
    if algebra._COORDINATE_BUDGET.get() is None:
        raise ValueError(
            "Deferred mapped PLC expressions require their owning support ledger."
        )
    coordinates = algebra.coordinate_expressions(cell.element, cell.controls)
    if coordinates is None:
        raise ValueError("Mapped PLC coordinate source expression is unresolved.")
    return coordinates


def plc_mapped_carrier(mesh: CellMesh, geometry: CellGeometrySpec, /) -> bool:
    """Validate actual Q1/restricted expressions and their rounded carrier corners."""
    _cell_maps(mesh, geometry)
    return True


def plc_mapped_vertices(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Canonical vertex rows in entity-set order, padded by -1 for mixed faces."""
    if dimension == 0:
        return np.arange(mesh.entity_set(0).count, dtype=np.int64)[:, None]
    connectivity = mesh.connectivity
    if dimension == 1 and isinstance(
        connectivity,
        (TetrahedralConnectivity, HexahedralConnectivity, PolyhedralConnectivity),
    ):
        return np.asarray(connectivity.edges, dtype=np.int64)
    if dimension == 2 and isinstance(
        connectivity, (TetrahedralConnectivity, HexahedralConnectivity)
    ):
        return np.asarray(connectivity.faces, dtype=np.int64)
    if dimension == 2 and isinstance(connectivity, PolyhedralConnectivity):
        offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        rows = np.full(
            (mesh.entity_set(2).count, connectivity.maximum_face_arity),
            -1,
            dtype=np.int64,
        )
        for row, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            rows[row, : stop - start] = values[start:stop]
        return rows
    if dimension == 3:
        ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
        positions = {identifier: row for row, identifier in enumerate(ids.tolist())}
        rows = np.full(
            (ids.size, max(block.vertices.shape[1] for block in mesh.blocks)),
            -1,
            dtype=np.int64,
        )
        for block in mesh.blocks:
            for identifier, vertices in zip(
                np.asarray(block.global_ids).tolist(),
                np.asarray(block.vertices, dtype=np.int64),
                strict=True,
            ):
                rows[positions[identifier], : vertices.size] = vertices
        return rows
    raise ValueError("PLC source support requires canonical fixed mixed-cell incidence.")


def _charts(
    cells: tuple[_CellMap, ...], rows: np.ndarray, dimension: int
) -> tuple[tuple[tuple[algebra.Expression, ...], str], ...]:
    vertices = tuple(rows[rows >= 0].tolist())
    result: list[tuple[tuple[algebra.Expression, ...], str]] = []
    for cell in cells:
        topology = reference_cell_topology(cell.kind)
        entities = topology.entities[dimension]
        for entity in entities:
            if {cell.vertices[index] for index in entity} != set(vertices):
                continue
            ordered = tuple(cell.vertices.index(vertex) for vertex in vertices)
            coordinates = _cell_coordinates(cell)
            if dimension == 2:
                if cell.kind == "pyramid":
                    patches = face_charts(coordinates, cell.kind, entity)
                    position = entity.index(ordered[0])
                    forward = entity[(position + 1) % len(entity)] == ordered[1]
                    if not forward:
                        arguments = tuple(reversed(algebra.axes(2)))
                        patches = tuple(
                            (
                                tuple(
                                    algebra.expression_compose(value, arguments)
                                    for value in chart
                                ),
                                domain,
                            )
                            for chart, domain in patches
                        )
                    result.extend(patches)
                else:
                    result.extend(face_charts(coordinates, cell.kind, ordered))
            else:
                corners = np.asarray(topology.vertices, dtype=np.float64)[
                    np.asarray(ordered, dtype=np.int64)
                ]
                if cell.kind == "pyramid" and 4 in ordered:
                    apex = ordered.index(4)
                    corners[apex, :2] = corners[1 - apex, :2]
                arguments = algebra.affine_arguments(
                    corners[0], (corners[1] - corners[0])[:, None]
                )
                result.append(
                    (
                        tuple(
                            algebra.expression_compose(value, arguments)
                            for value in coordinates
                        ),
                        "box",
                    )
                )
    if not result:
        raise ValueError("PLC entity has no actual coordinate-map incidence chart.")
    # Equal canonical coefficients prove equal entire charts. Neighbor incidence
    # remains checked; only mathematically identical image proofs are reused.
    unique: list[tuple[tuple[algebra.Expression, ...], str]] = []
    for chart in result:
        if chart not in unique:
            unique.append(chart)
    return tuple(unique)


def _spend(work: list[int], amount: int) -> None:
    work[0] -= amount
    if work[0] < 0:
        raise ValueError("Mapped PLC source-support query budget exhausted.")


def _nonnegative(
    value: algebra.Expression,
    dimension: int,
    limits: MeshCertificateLimits,
    work: list[int],
) -> bool:
    used = SubdivisionLedger()
    result = nonnegative(
        value,
        "box",
        dimension,
        limits.maximum_bernstein_nodes,
        limits.maximum_subdivision_depth,
        min(limits.maximum_subdivision_pieces, work[0]),
        used,
    )
    _spend(work, used.pieces)
    return result


def _edge_support(
    chart: tuple[algebra.Expression, ...],
    edge: np.ndarray,
    limits: MeshCertificateLimits,
    work: list[int],
) -> tuple[bool, int]:
    endpoints = rational_points(edge)
    _reserve_fraction_work(endpoints, len(endpoints[0]), len(endpoints[0]), 2)
    direction = tuple(b - a for a, b in zip(*endpoints, strict=True))
    axis = next((index for index, value in enumerate(direction) if value), None)
    if axis is None:
        raise ValueError("Authoritative PLC source edge is collapsed.")
    parameter = algebra.expression_scale(
        algebra.expression_add(chart[axis], algebra.constant(-endpoints[0][axis], 1)),
        1 / direction[axis],
    )
    for coordinate, origin, delta in zip(chart, endpoints[0], direction, strict=True):
        if algebra.expression_add(
            coordinate,
            algebra.expression_scale(
                algebra.expression_add(
                    algebra.constant(origin, 1),
                    algebra.expression_scale(parameter, delta),
                ),
                -1,
            ),
        ):
            return False, 0
    if not _nonnegative(parameter, 1, limits, work) or not _nonnegative(
        algebra.expression_add(
            algebra.constant(1, 1), algebra.expression_scale(parameter, -1)
        ),
        1,
        limits,
        work,
    ):
        return False, 0
    derivative = algebra.expression_derivative(parameter, 0)
    start = algebra.expression_evaluate(parameter, (Fraction(0),))
    end = algebra.expression_evaluate(parameter, (Fraction(1),))
    _reserve_fraction_work(((start, end),), 1, 1, 2)
    displacement = end - start
    orientation = 1 if displacement > 0 else -1 if displacement < 0 else 0
    return orientation != 0 and _nonnegative(
        algebra.expression_scale(derivative, orientation), 1, limits, work
    ), orientation


def _exact_segment_support(
    segment: tuple[_Point, _Point],
    endpoints: tuple[_Point, _Point],
    /,
) -> tuple[bool, int]:
    _reserve_fraction_work(
        (*segment, *endpoints),
        4 * len(segment[0]),
        2 + 2 * len(segment[0]),
        8,
    )
    direction = tuple(b - a for a, b in zip(*endpoints, strict=True))
    axis = next((index for index, value in enumerate(direction) if value), None)
    if axis is None:
        raise ValueError("Authoritative PLC source edge is collapsed.")
    parameters = tuple(
        (point[axis] - endpoints[0][axis]) / direction[axis] for point in segment
    )
    supported = all(
        0 <= parameter <= 1
        and all(
            value == origin + parameter * delta
            for value, origin, delta in zip(
                point,
                endpoints[0],
                direction,
                strict=True,
            )
        )
        for point, parameter in zip(segment, parameters, strict=True)
    )
    displacement = parameters[1] - parameters[0]
    orientation = 1 if displacement > 0 else -1 if displacement < 0 else 0
    return supported and orientation != 0, orientation


def _segment_in_union(
    segment: tuple[tuple[Fraction, ...], ...],
    triangles: tuple[tuple[tuple[Fraction, ...], ...], ...],
) -> bool:
    if len(segment) == 1:
        return any(
            all(
                (1 if signed_measure(triangle) > 0 else -1) * turn(a, b, segment[0]) >= 0
                for a, b in zip(triangle, (*triangle[1:], triangle[0]), strict=True)
            )
            for triangle in triangles
        )
    intervals: list[tuple[Fraction, Fraction]] = []
    for triangle in triangles:
        lower, upper = Fraction(0), Fraction(1)
        sign = 1 if signed_measure(triangle) > 0 else -1
        for a, b in zip(triangle, (*triangle[1:], triangle[0]), strict=True):
            start = sign * turn(a, b, segment[0])
            end = sign * turn(a, b, segment[-1])
            _reserve_fraction_work(((start, end),), 1, 1, 2)
            slope = end - start
            if slope > 0:
                _reserve_fraction_work(((start, slope),), 2, 1, 2)
                lower = max(lower, -start / slope)
            elif slope < 0:
                _reserve_fraction_work(((start, slope),), 2, 1, 2)
                upper = min(upper, -start / slope)
            elif start < 0:
                lower, upper = Fraction(1), Fraction(0)
        if lower <= upper:
            intervals.append((lower, upper))
    covered = Fraction(0)
    for lower, upper in sorted(intervals):
        if lower > covered:
            return False
        covered = max(covered, upper)
    return covered >= 1


def _edge_on_facet(
    chart: tuple[algebra.Expression, ...],
    group: SourceGroup,
    limits: MeshCertificateLimits,
    work: list[int],
) -> bool:
    plane = algebra.expression_add(
        algebra.expression_sum(
            tuple(
                algebra.expression_scale(value, weight)
                for value, weight in zip(chart, group.plane[:-1], strict=True)
            )
        ),
        algebra.constant(group.plane[-1], 1),
    )
    if plane:
        return False
    pending = [(chart, 0)]
    while pending:
        current, depth = pending.pop()
        _spend(work, 1 + len(group.simplices))
        controls = algebra.expression_control_points(
            tuple(current[axis] for axis in group.axes),
            "box",
            1,
            limits.maximum_bernstein_nodes,
        )
        if controls is None:
            return False
        hull = convex_hull(controls)
        area = abs(signed_measure(hull))
        contained = (
            sum(
                (intersection(hull, triangle) for triangle in group.simplices),
                Fraction(0),
            )
            == area
            if area > 0
            else _segment_in_union(hull, group.simplices)
        )
        if contained:
            continue
        if depth >= limits.maximum_subdivision_depth:
            return False
        pending.extend(
            (
                tuple(algebra.expression_compose(value, child) for value in current),
                depth + 1,
            )
            for child in _children("box", 1)
        )
    return True


@dataclass(frozen=True)
class MappedPlcSupport:
    """Current geometry-owner evidence and its exact source-expression charts."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    domain: PiecewiseLinearDomain
    cells: tuple[_CellMap, ...]
    cell_regions: np.ndarray
    embedding: GlobalEmbeddingCertificate
    coverage: DomainCoverageCertificate
    limits: MeshCertificateLimits
    entity_vertices: tuple[np.ndarray, ...]
    # Per dimension, immutable CSR (offsets, cell rows) of incident cells.
    entity_parents: tuple[tuple[np.ndarray, np.ndarray], ...]
    boundary_entities: tuple[np.ndarray, ...]
    prepared: MeshCertificationPreparedEvidence
    vertex_images: tuple[_Point, ...]
    # One exact-expression ledger under the original request limits owns every
    # support query and retained source group of this preparation; it is never
    # renewed per query.
    ledger: algebra.CoordinateEnclosureBudget = field(repr=False, compare=False)
    work_start: int = field(repr=False, compare=False)
    _facet_groups: dict[tuple[int, int, str, bytes, bytes], tuple[SourceGroup, ...]] = (
        field(default_factory=dict, repr=False, compare=False)
    )
    _source_edges: dict[tuple[int, int, str, bytes], tuple[_Point, _Point]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _source_vertex_banks: dict[int, tuple[np.ndarray, tuple[_Point, ...]]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def parents(self, dimension: int, row: int, /) -> tuple[int, ...]:
        """Incident cell rows of one entity, in incidence order."""
        if dimension == 3:
            return (row,)
        offsets, values = self.entity_parents[dimension]
        return tuple(values[offsets[row] : offsets[row + 1]].tolist())

    def _source_points(self, vertices: np.ndarray, /) -> tuple[_Point, ...]:
        if vertices.flags.writeable:
            return rational_points(vertices)
        identifier = id(vertices)
        cached = self._source_vertex_banks.get(identifier)
        if cached is not None and cached[0] is vertices:
            return cached[1]
        exact = rational_points(vertices)
        self._source_vertex_banks[identifier] = (vertices, exact)
        self.ledger.retain_basis(exact)
        return exact

    def _source_edge(
        self,
        source_index: int,
        edge: np.ndarray,
        vertices: np.ndarray,
        /,
    ) -> tuple[_Point, _Point]:
        exact = self._source_points(vertices)
        self.ledger.reserve(edge.size, 256 + edge.nbytes)
        key = (
            source_index,
            id(exact),
            edge.dtype.str,
            edge.tobytes(),
        )
        cached = self._source_edges.get(key)
        if cached is not None:
            return cached
        result = exact[int(edge[0])], exact[int(edge[1])]
        self._source_edges[key] = result
        self.ledger.retain_basis(key)
        self.ledger.retain_basis(result)
        return result

    def _source_simplex(
        self,
        triangle: np.ndarray,
        vertices: np.ndarray,
        /,
    ) -> _AffineSourceSimplex:
        exact = self._source_points(vertices)
        self.ledger.reserve(triangle.size)
        points = tuple(exact[int(vertex)] for vertex in triangle)
        return _prepare_affine_exact_source_simplex(points)

    def _source_facet_groups(
        self,
        source_index: int,
        triangles: np.ndarray,
        vertices: np.ndarray,
        work: list[int],
    ) -> tuple[SourceGroup, ...]:
        # The explicit source index selects identity. Full authority-row and
        # coordinate bytes bind only reuse of its already-proved planar union;
        # mutation of either bank cannot reuse a previous source theorem.
        points = vertices[triangles]
        # The key's byte copies are this query's workspace, charged before use.
        self.ledger.reserve(
            triangles.size + points.size, 256 + triangles.nbytes + points.nbytes
        )
        key = (
            source_index,
            triangles.shape[0],
            triangles.dtype.str,
            triangles.tobytes(),
            points.tobytes(),
        )
        cached = self._facet_groups.get(key)
        if cached is not None:
            return cached
        _spend(work, 1)
        if len(self._facet_groups) >= self.limits.maximum_candidate_pairs:
            raise ValueError("Mapped PLC prepared facet-group capacity exhausted.")
        groups, overlaps, used, exceeded = source_groups(
            vertices,
            triangles,
            np.tile(np.asarray(((0, -1),), dtype=np.int64), (triangles.shape[0], 1)),
            min(self.limits.maximum_candidate_pairs, work[0]),
        )
        _spend(work, used)
        if overlaps or exceeded:
            raise ValueError(
                "Authoritative PLC facet union is overlapping or its resource bound is unresolved."
            )
        self._facet_groups[key] = groups
        self.ledger.retain_basis(key)
        self.ledger.retain_basis(groups)
        return groups

    def entity_support(
        self,
        dimension: int,
        row: int,
        source_dimension: int,
        source_index: int,
        /,
        *,
        edge_vertices: np.ndarray,
        triangle_vertices: np.ndarray,
        triangle_facets: np.ndarray,
        work: list[int],
        source_vertices: np.ndarray | None = None,
    ) -> tuple[bool, int]:
        """Prove support of an explicitly identified stratum; return orientation.

        Exact chart algebra is charged to this preparation's ledger; a refusal
        raises ``CoordinateEnclosureResourceError`` with its requested and
        completed work.
        """
        with (
            self.ledger.activate(),
            self.ledger.bound_stage(
                self.prepared.request.limits.maximum_work_units,
                self.prepared.request.limits.maximum_scratch_bytes,
                starting_work_units=self.work_start,
            ),
            self.ledger.temporary_scope(),
        ):
            return self._entity_support(
                dimension,
                row,
                source_dimension,
                source_index,
                edge_vertices=edge_vertices,
                triangle_vertices=triangle_vertices,
                triangle_facets=triangle_facets,
                work=work,
                source_vertices=source_vertices,
            )

    def _entity_support(
        self,
        dimension: int,
        row: int,
        source_dimension: int,
        source_index: int,
        /,
        *,
        edge_vertices: np.ndarray,
        triangle_vertices: np.ndarray,
        triangle_facets: np.ndarray,
        work: list[int],
        source_vertices: np.ndarray | None = None,
    ) -> tuple[bool, int]:
        if (
            dimension not in range(4)
            or row < 0
            or row >= self.mesh.entity_set(dimension).count
        ):
            raise ValueError("Mapped PLC support names an undeclared mesh entity.")
        if source_dimension not in range(4) or source_index < 0:
            raise ValueError(
                "Mapped PLC support requires explicit valid source-stratum indices."
            )
        authority = self.domain.vertices if source_vertices is None else source_vertices
        if source_dimension == 1 and source_index >= edge_vertices.shape[0]:
            raise ValueError("Mapped PLC support names an undeclared source edge.")
        parents = self.parents(dimension, row)
        if source_dimension == 3:
            if dimension < 3 and self.boundary_entities[dimension][row]:
                return False, 0
            _spend(work, len(parents))
            charge_native_geometry_queries(len(parents))
            return bool(parents) and bool(
                np.all(
                    self.cell_regions[np.asarray(parents, dtype=np.int64)] == source_index
                )
            ), 0
        if dimension not in (1, 2) or source_dimension < dimension:
            return False, 0
        entity_vertices = self.entity_vertices[dimension][row]
        exact_points = (
            tuple(
                self.vertex_images[int(vertex)]
                for vertex in entity_vertices
                if vertex >= 0
            )
            if all(self.cells[parent].kind == "tetrahedron" for parent in parents)
            else ()
        )
        if source_dimension == 1:
            if dimension == 1 and len(exact_points) == 2:
                _spend(work, 1)
                charge_native_geometry_queries(1)
                return _exact_segment_support(
                    (exact_points[0], exact_points[1]),
                    self._source_edge(
                        source_index,
                        edge_vertices[source_index],
                        authority,
                    ),
                )
            charts = _charts(
                tuple(self.cells[parent] for parent in parents),
                entity_vertices,
                dimension,
            )
            charge_native_geometry_queries(len(charts))
            outcomes = tuple(
                _edge_support(
                    chart,
                    authority[edge_vertices[source_index]],
                    self.limits,
                    work,
                )
                for chart, _ in charts
            )
            signs = {orientation for supported, orientation in outcomes if supported}
            return (
                all(supported for supported, _ in outcomes) and len(signs) == 1,
                next(iter(signs)) if len(signs) == 1 else 0,
            )
        if source_dimension != 2:
            return False, 0
        triangles = triangle_vertices[triangle_facets == source_index]
        if triangles.shape[0] == 0:
            return False, 0
        # One authoritative simplex has one unambiguous owner. Exact affine
        # endpoints/corners prove support directly; multi-simplex ownership
        # keeps the full union and clipping path below.
        if triangles.shape[0] == 1 and len(exact_points) == dimension + 1:
            _spend(work, 2)
            charge_native_geometry_queries(1)
            source_simplex = self._source_simplex(
                triangles[0],
                authority,
            )
            if dimension == 1:
                return _affine_segment_containment(
                    (exact_points[0], exact_points[1]),
                    source_simplex,
                ), 0
            direct = _affine_simplex_corner_containment(
                (exact_points[0], exact_points[1], exact_points[2]),
                source_simplex,
            )
            return (direct is not None, 0 if direct is None else direct[1])
        charts = _charts(
            tuple(self.cells[parent] for parent in parents),
            entity_vertices,
            dimension,
        )
        groups = self._source_facet_groups(
            source_index,
            triangles,
            authority,
            work,
        )
        orientations: set[int] = set()
        for chart, chart_domain in charts:
            supported = False
            for group in groups:
                if dimension == 1:
                    charge_native_geometry_queries(1)
                    supported = _edge_on_facet(chart, group, self.limits, work)
                else:
                    for own, other in (group.regions, group.regions[::-1]):
                        charge_native_geometry_queries(1)
                        used_work = SubdivisionLedger()
                        outcome, _, _ = mapped_containment(
                            chart,
                            chart_domain,
                            group,
                            own,
                            other,
                            self.limits.maximum_bernstein_nodes,
                            self.limits.maximum_subdivision_depth,
                            min(self.limits.maximum_subdivision_pieces, work[0]),
                            min(self.limits.maximum_candidate_pairs, work[0]),
                            used_work,
                        )
                        _spend(work, used_work.pieces + used_work.candidate_pairs)
                        if outcome != "proven":
                            continue
                        # Reuse the exact target and source orientation already
                        # established by containment and source preparation.
                        target_orientation = group.orientation * (
                            1 if own == group.regions[0] else -1
                        )
                        orientations.add(target_orientation * group.orientations[0])
                        supported = True
                        break
                if supported:
                    break
            if not supported:
                return False, 0
        return len(orientations) <= 1, next(iter(orientations)) if orientations else 0

    def vertex_support(
        self,
        row: int,
        source_dimension: int,
        source_index: int,
        /,
        *,
        edge_vertices: np.ndarray,
        triangle_vertices: np.ndarray,
        triangle_facets: np.ndarray,
        work: list[int],
        source_vertices: np.ndarray | None = None,
    ) -> bool:
        """Prove stratum support of a vertex on its exact map corner, not its rounding.

        ``source_index`` is the local domain vertex/edge row for strata 0/1 and
        the declared facet/region identifier for strata 2/3. Exact algebra is
        charged to this preparation's ledger.
        """
        with (
            self.ledger.activate(),
            self.ledger.bound_stage(
                self.prepared.request.limits.maximum_work_units,
                self.prepared.request.limits.maximum_scratch_bytes,
                starting_work_units=self.work_start,
            ),
            self.ledger.temporary_scope(),
        ):
            return self._vertex_support(
                row,
                source_dimension,
                source_index,
                edge_vertices=edge_vertices,
                triangle_vertices=triangle_vertices,
                triangle_facets=triangle_facets,
                work=work,
                source_vertices=source_vertices,
            )

    def _vertex_support(
        self,
        row: int,
        source_dimension: int,
        source_index: int,
        /,
        *,
        edge_vertices: np.ndarray,
        triangle_vertices: np.ndarray,
        triangle_facets: np.ndarray,
        work: list[int],
        source_vertices: np.ndarray | None = None,
    ) -> bool:
        if row < 0 or row >= self.mesh.entity_set(0).count:
            raise ValueError("Mapped PLC support names an undeclared mesh vertex.")
        if source_dimension == 3:
            return self._entity_support(
                0,
                row,
                3,
                source_index,
                edge_vertices=edge_vertices,
                triangle_vertices=triangle_vertices,
                triangle_facets=triangle_facets,
                work=work,
                source_vertices=source_vertices,
            )[0]
        point = self.vertex_images[int(self.entity_vertices[0][row][0])]
        _spend(work, 1)
        charge_native_geometry_queries(1)
        authority = self.domain.vertices if source_vertices is None else source_vertices
        if source_dimension == 0:
            return point == rational_points(authority[source_index : source_index + 1])[0]
        if source_dimension == 1:
            start, end = rational_points(authority[edge_vertices[source_index]])
            _reserve_fraction_work(
                (point, start, end), 2 + 3 * len(point), 1 + 2 * len(point), 8
            )
            direction = tuple(b - a for a, b in zip(start, end, strict=True))
            axis = max(range(len(direction)), key=lambda value: abs(direction[value]))
            if not direction[axis]:
                raise ValueError("Authoritative PLC source edge is collapsed.")
            parameter = (point[axis] - start[axis]) / direction[axis]
            return 0 <= parameter <= 1 and all(
                value == origin + parameter * delta
                for value, origin, delta in zip(point, start, direction, strict=True)
            )
        if source_dimension != 2:
            raise ValueError("Unknown PLC source stratum dimension.")
        triangles = triangle_vertices[triangle_facets == source_index]
        if triangles.shape[0] == 0:
            return False
        for group in self._source_facet_groups(source_index, triangles, authority, work):
            _spend(work, len(group.simplices))
            charge_native_geometry_queries(len(group.simplices))
            _reserve_fraction_work(
                (group.plane, point), 2 * len(point), 1, 2 * len(point) + 1
            )
            if sum(
                (
                    weight * value
                    for weight, value in zip(group.plane[:-1], point, strict=True)
                ),
                group.plane[-1],
            ) == 0 and _segment_in_union(project((point,), group.axes), group.simplices):
                return True
        return False


def _require_support_work_bounds(
    prepared: MeshCertificationPreparedEvidence,
    limits: MeshCertificateLimits,
    vertices: tuple[np.ndarray, ...],
    boundary: tuple[np.ndarray, ...],
    embedding_was_supplied: bool,
) -> None:
    """Preserve the support owner's stricter work caps on canonical premises."""
    charged = [
        (
            prepared.coverage.candidate_pair_count,
            limits.maximum_candidate_pairs,
            "coverage candidates",
        ),
        (
            prepared.coverage.subdivision_piece_count,
            limits.maximum_subdivision_pieces,
            "coverage subdivision pieces",
        ),
        (
            prepared.coverage.maximum_subdivision_depth_reached,
            limits.maximum_subdivision_depth,
            "coverage subdivision depth",
        ),
    ]
    if not embedding_was_supplied:
        embedding = prepared.embedding
        charged.extend(
            (
                (
                    embedding.candidate_pair_count,
                    limits.maximum_candidate_pairs,
                    "embedding candidates",
                ),
                (
                    embedding.ray_test_count,
                    limits.maximum_ray_tests,
                    "embedding ray tests",
                ),
                (
                    embedding.subdivision_piece_count,
                    limits.maximum_subdivision_pieces,
                    "embedding subdivision pieces",
                ),
            )
        )
        if "exterior_degree" in embedding.evaluated_checks:
            # The owning affine degree proof preflights every shell against all
            # triangulated boundary pieces, not just the rays eventually used.
            pieces = sum(
                int(np.count_nonzero(row >= 0)) - 2 for row in vertices[2][boundary[2]]
            )
            charged.append(
                (
                    embedding.shell_count * pieces,
                    limits.maximum_ray_tests,
                    "exterior degree preflight",
                )
            )
    for actual, maximum, name in charged:
        if actual > maximum:
            raise ValueError(
                f"Mapped PLC source support exhausts its {name} bound: {actual} > {maximum}."
            )
    # Neither premise reports a Bernstein-node ledger and the embedding reports
    # no depth ledger. A no-larger canonical policy is itself a rigorous bound;
    # do not guess usage beyond that policy.
    if prepared.request.limits.maximum_bernstein_nodes > limits.maximum_bernstein_nodes:
        raise ValueError(
            "Prepared coverage does not establish the source support coefficient bound."
        )
    if (
        not embedding_was_supplied
        and prepared.request.limits.maximum_subdivision_depth
        > limits.maximum_subdivision_depth
    ):
        raise ValueError(
            "Prepared embedding does not establish the source support depth bound."
        )


def prepare_mapped_plc_support(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain,
    cell_regions: np.ndarray,
    region_indices: np.ndarray,
    /,
    *,
    maximum_support_queries: int,
    embedding: GlobalEmbeddingCertificate | None = None,
    validity: CellValidityCertificate | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
    certification_request: MeshCertificationInputs | None = None,
) -> MappedPlcSupport:
    """Validate source evidence or independently certify the complete target image.

    Labels are explicit region identifiers in entity-set order. Owning positive
    validity and embedding must bind the same actual map. The canonical prepared
    acceptance owner computes coverage once with explicit domain-row labels in
    block order. Its canonical request policy is distinct from the row budget,
    while actual charged work must still meet the original stricter support caps.
    Publication supplies its exact limits; existing embedding requires validity.
    """
    if certification_request is not None:
        from ._certification_inputs import MeshCertificationInputs

        if not isinstance(certification_request, MeshCertificationInputs):
            raise TypeError(
                "certification_request must be MeshCertificationInputs or None."
            )
        certification_request.validate_source_integrity()
        if (
            certificate_limits is not None
            and certificate_limits.limits_id != certification_request.limits.limits_id
        ):
            raise ValueError(
                "Mapped source support and its target request use different limits."
            )
        certificate_limits_ = certification_request.limits
    else:
        certificate_limits_ = (
            MeshCertificateLimits() if certificate_limits is None else certificate_limits
        )
    ledger = algebra.coordinate_enclosure_budget(
        certificate_limits_.maximum_work_units,
        certificate_limits_.maximum_scratch_bytes,
    )
    work_start = ledger.work_units
    with (
        ledger.activate(),
        ledger.bound_stage(
            certificate_limits_.maximum_work_units,
            certificate_limits_.maximum_scratch_bytes,
            starting_work_units=work_start,
        ),
    ):
        return _prepare_mapped_plc_support(
            mesh,
            geometry,
            domain,
            cell_regions,
            region_indices,
            maximum_support_queries=maximum_support_queries,
            embedding=embedding,
            validity=validity,
            certificate_limits=certificate_limits_,
            certification_request=certification_request,
            ledger=ledger,
            work_start=work_start,
        )


def _prepare_mapped_plc_support(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain,
    cell_regions: np.ndarray,
    region_indices: np.ndarray,
    /,
    *,
    maximum_support_queries: int,
    embedding: GlobalEmbeddingCertificate | None,
    validity: CellValidityCertificate | None,
    certificate_limits: MeshCertificateLimits,
    certification_request: MeshCertificationInputs | None,
    ledger: algebra.CoordinateEnclosureBudget,
    work_start: int,
) -> MappedPlcSupport:
    from ._association import _boundary_mask, _entity_rows, _incidence_pairs
    from ._certification import (
        MeshCertificationPreparedEvidence,
        MeshCertificationSchedule,
    )

    embedding_was_supplied = embedding is not None
    with ledger.activate():
        with ledger.temporary_scope():
            cells, vertex_images = _cell_maps(mesh, geometry, prepare_expressions=False)
        # Exact source controls remain shared by their actual source-dof route;
        # physical expressions are formed only for requested edge/face charts.
        retained_controls: set[int] = set()
        for cell in cells:
            if id(cell.controls) not in retained_controls:
                ledger.retain_basis(cell.controls)
                retained_controls.add(id(cell.controls))
        ledger.retain_basis(vertex_images)
        if geometry.exact_source is not None and all(
            cell.kind == "tetrahedron" for cell in cells
        ):
            ledger.reserve(sum(len(point) for point in vertex_images))
            if all(
                Fraction(float(value)) == value
                for point in vertex_images
                for value in point
            ):
                scope_key = algebra.coordinate_scope_key(mesh, geometry)
                affine_cells = np.empty((0,), dtype=np.int64)
                affine_cells.setflags(write=False)
                prepared_scope = (False, affine_cells, vertex_images)
                ledger.retain_basis((scope_key, prepared_scope))
                ledger.scope_cache[scope_key] = prepared_scope
        # The record dictionaries and integer vertex tuples remain live after
        # preparation; coefficient/point data are retained separately above.
        ledger.reserve(
            sum(2 + len(cell.vertices) + len(cell.root_vertex_ids) for cell in cells),
            sys.getsizeof(cells)
            + sum(
                sys.getsizeof(cell)
                + sys.getsizeof(cell.__dict__)
                + sys.getsizeof(cell.root_cell_id)
                + sys.getsizeof(cell.vertices)
                + sys.getsizeof(cell.root_vertex_ids)
                + sum(sys.getsizeof(vertex) for vertex in cell.vertices)
                + sum(sys.getsizeof(vertex) for vertex in cell.root_vertex_ids)
                for cell in cells
            ),
        )
    labels = np.asarray(cell_regions, dtype=np.int64)
    declared = np.asarray(region_indices, dtype=np.int64)
    if labels.shape != (mesh.entity_set(3).count,) or not np.all(
        np.isin(labels, declared)
    ):
        raise ValueError(
            "Mapped PLC cells require complete explicit source-region identifiers."
        )
    if (
        declared.shape != (len(domain.region_ids),)
        or np.unique(declared).size != declared.size
    ):
        raise ValueError(
            "Mapped PLC region identifiers must bijectively bind domain regions."
        )
    limits = MeshCertificateLimits(
        maximum_candidate_pairs=maximum_support_queries,
        maximum_ray_tests=maximum_support_queries,
        maximum_subdivision_pieces=maximum_support_queries,
    )
    block_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    order = _entity_rows(mesh, 3, block_ids)
    indices = {label: index for index, label in enumerate(declared.tolist())}
    regions = np.asarray(
        [indices[label] for label in labels[order].tolist()], dtype=np.int64
    )
    if embedding is not None and validity is None:
        raise ValueError(
            "Mapped PLC embedding reuse requires its actual bound validity premise."
        )
    with ledger.activate():
        if validity is None:
            validity = _exact_affine_plc_validity(
                mesh,
                geometry,
                cells,
                vertex_images,
            )
            if validity is None:
                validity = certify_cell_geometry_validity(geometry, mesh=mesh)
        elif not isinstance(validity, CellValidityCertificate):
            raise TypeError("validity must be an owning CellValidityCertificate.")
        validity.require_bound(geometry, mesh=mesh)
        if not validity.all_certified:
            raise ValueError(
                "Mapped PLC coordinate-map positivity is invalid or unresolved."
            )

    def prepare_evidence() -> MeshCertificationPreparedEvidence:
        return MeshCertificationPreparedEvidence(
            mesh,
            geometry,
            validity,
            schedule=MeshCertificationSchedule("volume_plc")
            if certification_request is None
            else certification_request.schedule,
            domain=domain,
            cell_regions=regions,
            embedding=embedding,
            limits=certificate_limits,
            source=None
            if certification_request is None
            else certification_request.source,
            fidelity_tolerance=None
            if certification_request is None
            else certification_request.fidelity_tolerance,
            fidelity_sample_order=4
            if certification_request is None
            else certification_request.fidelity_sample_order,
            scoped_fidelity=()
            if certification_request is None
            else certification_request.scoped_fidelity,
        )

    with ledger.activate():
        prepared = prepare_evidence()
    if certification_request is not None:
        prepared.require(mesh, geometry, certification_request)
    embedding, coverage = prepared.embedding, prepared.coverage
    frozen = np.array(labels, dtype=np.int64, copy=True)
    frozen.setflags(write=False)
    inverse_order = np.argsort(order, kind="stable")
    ordered_cells = tuple(cells[index] for index in inverse_order.tolist())
    unused_top_vertices = np.empty(
        (mesh.entity_set(3).count, 0),
        dtype=np.int64,
    )
    unused_top_vertices.setflags(write=False)
    vertices = (
        *(plc_mapped_vertices(mesh, dimension) for dimension in range(3)),
        unused_top_vertices,
    )
    # Parent cells of every lower-dimensional entity, in incidence order.
    parents: list[tuple[np.ndarray, np.ndarray]] = []
    for dimension in range(3):
        incidence = _incidence_pairs(mesh, dimension, 3)
        rows = np.argsort(incidence[:, 0], kind="stable")
        offsets = np.concatenate(
            (
                np.zeros((1,), dtype=np.int64),
                np.cumsum(
                    np.bincount(
                        incidence[:, 0], minlength=mesh.entity_set(dimension).count
                    ),
                    dtype=np.int64,
                ),
            )
        )
        values = np.ascontiguousarray(incidence[rows, 1], dtype=np.int64)
        offsets.setflags(write=False)
        values.setflags(write=False)
        parents.append((offsets, values))
    unused_top_parents = (
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
    )
    unused_top_parents[0].setflags(write=False)
    unused_top_parents[1].setflags(write=False)
    parents.append(unused_top_parents)
    boundary = (
        *(_boundary_mask(mesh, dimension) for dimension in range(3)),
        np.empty((0,), dtype=np.bool_),
    )
    boundary[3].setflags(write=False)
    _require_support_work_bounds(
        prepared, limits, vertices, boundary, embedding_was_supplied
    )
    retained = (
        frozen,
        *vertices,
        *(array for pair in parents for array in pair),
        *boundary,
    )
    ledger.reserve(
        sum(array.size for array in retained),
        sys.getsizeof(ordered_cells)
        + sys.getsizeof(vertices)
        + sys.getsizeof(parents)
        + sys.getsizeof(boundary)
        + sum(
            sys.getsizeof(array)
            if array.flags.owndata
            else array.nbytes + sys.getsizeof(array)
            for array in retained
        ),
    )
    # The published preparation owns its retained support tables too. Rebuild
    # after their charge so required/charged expression evidence is complete.
    with ledger.activate():
        prepared = prepare_evidence()
    if certification_request is not None:
        prepared.require(mesh, geometry, certification_request)
    embedding, coverage = prepared.embedding, prepared.coverage
    return MappedPlcSupport(
        mesh,
        geometry,
        domain,
        ordered_cells,
        frozen,
        embedding,
        coverage,
        limits,
        vertices,
        tuple(parents),
        boundary,
        prepared,
        vertex_images,
        ledger,
        work_start,
    )


def _mapped_association_orientations(
    transfer: PlcAssociationTransfer,
    mesh: CellMesh,
    proof: MappedPlcSupport,
    association: GeometryAssociation,
    dimension: int,
    work: list[int],
) -> np.ndarray:
    from ._association import _entity_rows, AssociationPropagationError

    rows = _entity_rows(
        mesh, dimension, np.asarray(association.target_global_ids, dtype=np.int64)
    )
    if np.any(rows < 0):
        raise ValueError("Mapped PLC row proof names an undeclared target entity.")
    orientations = np.zeros(rows.shape, dtype=np.int8)
    kinds = np.asarray(association.source_dimensions, dtype=np.int64)
    indices = np.asarray(association.source_indices, dtype=np.int64)
    for position, row, kind, index in zip(
        range(rows.size), rows.tolist(), kinds.tolist(), indices.tolist(), strict=True
    ):
        local_index = index
        if kind in (0, 1):
            matches = np.flatnonzero(
                (transfer.vertex_indices if kind == 0 else transfer.edge_indices) == index
            )
            if matches.size != 1:
                raise ValueError(
                    "Mapped PLC source vertex/edge lacks its explicit authority row."
                )
            local_index = int(matches[0])
        if dimension == 0:
            supported = proof.vertex_support(
                row,
                kind,
                local_index,
                edge_vertices=transfer.edge_vertices,
                triangle_vertices=transfer.triangle_vertices,
                triangle_facets=transfer.triangle_facets,
                work=work,
            )
            orientation = 0
        else:
            supported, orientation = proof.entity_support(
                dimension,
                row,
                kind,
                local_index,
                edge_vertices=transfer.edge_vertices,
                triangle_vertices=transfer.triangle_vertices,
                triangle_facets=transfer.triangle_facets,
                work=work,
            )
        if not supported:
            raise AssociationPropagationError(
                "Actual Q1 source image does not have its declared exact PLC stratum support.",
                np.asarray(association.target_global_ids)[position : position + 1],
            )
        orientations[position] = orientation
    return orientations


@dataclass(frozen=True)
class MappedPlcAssociationProof:
    """Exact source rows together with their canonical labelled acceptance premises."""

    associations: tuple[GeometryAssociation, ...]
    support: MappedPlcSupport


def certify_mapped_plc_associations(
    transfer: PlcAssociationTransfer,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    associations: tuple[GeometryAssociation, ...],
    /,
    *,
    embedding: GlobalEmbeddingCertificate | None = None,
    validity: CellValidityCertificate | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
) -> MappedPlcAssociationProof:
    """Publish exact represented-source rows only after proving actual Q1 support.

    This is a source-publication theorem, not relaxation of a transfer policy.
    The input rows supply authoritative stratum identities and lineage. Rounded
    simplex-parent containment does not decide membership in a different PLC
    stratum. Every point and complete coordinate-map image is instead checked
    against the explicitly named authoritative PLC source before residual and
    orientation evidence is issued. Input rows and their source remain unchanged.
    """
    from ._association import (
        _target_dimension,
        GeometryAssociation,
        GeometryAssociationKind,
        GeometrySourceEntityRole,
    )
    from ._volume_generation import _entity

    if transfer.domain.ambient_dimension != 3:
        raise ValueError("Mapped PLC source publication requires volume authority.")
    roles = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.FACET,
        GeometrySourceEntityRole.REGION,
    )
    dimensions: list[int] = []
    regions: np.ndarray | None = None
    for association in associations:
        if (
            association.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
            or association.source_id != transfer.domain.source_id
            or association.source_revision != transfer.source_revision
        ):
            raise ValueError(
                "Mapped PLC publication requires its actual represented source namespace."
            )
        dimension = _target_dimension(mesh, association)
        association.validate_target(mesh.entity_set(dimension))
        if dimension in dimensions or not association.complete:
            raise ValueError(
                "Mapped PLC publication requires complete unambiguous explicit source rows."
            )
        kinds = np.asarray(association.source_dimensions, dtype=np.int64)
        indices = np.asarray(association.source_indices, dtype=np.int64)
        if np.any(kinds < 0) or np.any(kinds > 3) or np.any(indices < 0):
            raise ValueError(
                "Mapped PLC publication requires declared source-stratum identities."
            )
        expected_roles = tuple(roles[kind] for kind in kinds.tolist())
        expected_ids = tuple(
            _entity(transfer.source_revision, role.value, index)
            for role, index in zip(expected_roles, indices.tolist(), strict=True)
        )
        if (
            association.source_entity_roles != expected_roles
            or association.source_entity_ids != expected_ids
        ):
            raise ValueError(
                "Mapped PLC source roles and namespace IDs disagree with their explicit indices."
            )
        if (
            np.any(~np.isin(indices[kinds == 0], transfer.vertex_indices))
            or np.any(~np.isin(indices[kinds == 1], transfer.edge_indices))
            or np.any(indices[kinds == 2] >= transfer.facet_regions.shape[0])
            or np.any(~np.isin(indices[kinds == 3], transfer.region_indices))
        ):
            raise ValueError(
                "Mapped PLC publication references undeclared source authority."
            )
        if dimension == 3:
            if np.any(kinds != 3):
                raise ValueError("Mapped PLC cells must name explicit source regions.")
            regions = indices[
                association.target_rows(
                    np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
                )
            ]
        dimensions.append(dimension)
    if 0 not in dimensions or regions is None:
        raise ValueError(
            "Mapped PLC publication requires complete vertex and cell source rows."
        )
    proof = prepare_mapped_plc_support(
        mesh,
        geometry,
        transfer.domain,
        regions,
        transfer.region_indices,
        maximum_support_queries=transfer.maximum_support_queries,
        embedding=embedding,
        validity=validity,
        certificate_limits=certificate_limits,
    )
    work = [transfer.maximum_support_queries]
    output: list[GeometryAssociation] = []
    for association in associations:
        dimension = _target_dimension(mesh, association)
        orientations = _mapped_association_orientations(
            transfer, mesh, proof, association, dimension, work
        )
        kinds = np.asarray(association.source_dimensions, dtype=np.int64)
        indices = np.asarray(association.source_indices, dtype=np.int64)
        output.append(
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                transfer.domain.source_id,
                transfer.source_revision,
                association.target_entity_set_id,
                association.target_global_ids,
                association.source_entity_ids,
                np.zeros(association.target_global_ids.shape, dtype=np.float64),
                exact=True,
                source_dimensions=kinds,
                source_indices=indices,
                source_entity_roles=association.source_entity_roles,
                parameters=association.parameters,
                orientations=orientations,
                parent_dimensions=association.parent_dimensions,
                parent_ids=association.parent_ids,
                parent_association_id=association.parent_association_id,
                provenance=association.provenance,
            )
        )
    return MappedPlcAssociationProof(tuple(output), proof)


def _require_mapped_source_translation(
    transfer: PlcAssociationTransfer,
    source: CellMeshingResult,
    predecessor: PlcAssociationTransfer,
    target: CellMesh,
    geometry: CellGeometrySpec,
    shift: np.ndarray,
) -> None:
    if (
        transfer.domain.ambient_dimension != 3
        or predecessor.domain.ambient_dimension != 3
    ):
        raise ValueError("Mapped source translation requires volume PLC authority.")
    if source.mesh.storage is not None or target.storage is not None:
        raise ValueError(
            "Mapped source translation requires complete host-owned carriers."
        )
    if (
        transfer.domain.source_id != predecessor.domain.source_id
        or transfer.coordinate_contract.spatial_id
        != predecessor.coordinate_contract.spatial_id
        or source.coordinate_contract.spatial_id
        != predecessor.coordinate_contract.spatial_id
    ):
        raise ValueError(
            "Mapped source translation cannot change source identity, coordinate frame or units."
        )
    if transfer.source_revision == predecessor.source_revision:
        raise ValueError(
            "Mapped source translation requires an explicit successor revision."
        )
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        raise ValueError(
            "Mapped source translation requires a finite three-dimensional displacement."
        )
    if (
        target.topology_id != source.mesh.topology_id
        or geometry.geometry_layout_id != source.geometry.geometry_layout_id
    ):
        raise ValueError(
            "Mapped source translation must retain topology and actual coordinate-map layout."
        )
    for dimension in range(4):
        if not np.array_equal(
            target.entity_set(dimension).entity_ids,
            source.mesh.entity_set(dimension).entity_ids,
        ):
            raise ValueError(
                "Mapped source translation must preserve every scientific mesh entity identity."
            )
    tables = (
        (transfer.edge_vertices, predecessor.edge_vertices),
        (transfer.vertex_indices, predecessor.vertex_indices),
        (transfer.edge_indices, predecessor.edge_indices),
        (transfer.triangle_vertices, predecessor.triangle_vertices),
        (transfer.triangle_facets, predecessor.triangle_facets),
        (transfer.facet_regions, predecessor.facet_regions),
        (transfer.region_indices, predecessor.region_indices),
        (transfer.domain.facets, predecessor.domain.facets),
        (transfer.domain.facet_regions, predecessor.domain.facet_regions),
    )
    if any(not np.array_equal(actual, original) for actual, original in tables):
        raise ValueError(
            "Mapped source translation cannot change stratum incidence or authority indices."
        )
    if transfer.domain.region_ids != predecessor.domain.region_ids:
        raise ValueError(
            "Mapped source translation cannot change declared domain-region identities."
        )
    for actual, original in (
        (transfer.source_vertices, predecessor.source_vertices),
        (transfer.domain.vertices, predecessor.domain.vertices),
        (np.asarray(target.coordinates), np.asarray(source.mesh.coordinates)),
        (np.asarray(geometry.coordinates), np.asarray(source.geometry.coordinates)),
    ):
        if actual.shape != original.shape or not np.array_equal(actual, original + shift):
            raise ValueError(
                "Mapped source transition differs from its declared translation."
            )
    original_maps, _ = _cell_maps(source.mesh, source.geometry)
    target_maps, _ = _cell_maps(target, geometry)
    for original, actual in zip(original_maps, target_maps, strict=True):
        expected_map = tuple(
            algebra.expression_add(
                value, algebra.constant(Fraction(float(displacement)), 3)
            )
            for value, displacement in zip(
                _cell_coordinates(original), shift, strict=True
            )
        )
        if (
            original.kind != actual.kind
            or original.vertices != actual.vertices
            or _cell_coordinates(actual) != expected_map
        ):
            raise ValueError(
                "Actual mapped coordinate polynomials do not realize the declared exact source translation."
            )


def transition_mapped_source_associations(
    transfer: PlcAssociationTransfer,
    source: CellMeshingResult,
    predecessor: PlcAssociationTransfer,
    target: CellMesh,
    /,
    *,
    geometry: CellGeometrySpec,
    translation: ArrayLike,
) -> tuple[
    tuple[MeshPatch, ...],
    tuple[MeshZone, ...],
    tuple[MeshLabel, ...],
    tuple[GeometryAssociation, ...],
]:
    """Carry explicit represented volume strata through a proven physical translation."""
    from ._association import (
        _target_dimension,
        GeometryAssociation,
        GeometryAssociationKind,
        GeometryAssociationProvenance,
    )
    from ._lineage import identity_lineage, inherit_mesh_organization
    from ._volume_generation import _entity

    shift = np.asarray(translation, dtype=np.float64)
    _require_mapped_source_translation(
        transfer, source, predecessor, target, geometry, shift
    )
    predecessor.classes(source)
    cell = next(
        association
        for association in source.associations
        if _target_dimension(source.mesh, association) == 3
    )
    regions = np.asarray(cell.source_indices, dtype=np.int64)[
        cell.target_rows(np.asarray(target.entity_set(3).entity_ids, dtype=np.int64))
    ]
    proof = prepare_mapped_plc_support(
        target,
        geometry,
        transfer.domain,
        regions,
        transfer.region_indices,
        maximum_support_queries=transfer.maximum_support_queries,
    )
    work = [transfer.maximum_support_queries]
    output: list[GeometryAssociation] = []
    for previous in source.associations:
        dimension = _target_dimension(source.mesh, previous)
        orientations = _mapped_association_orientations(
            transfer, target, proof, previous, dimension, work
        )
        roles = previous.source_entity_roles
        if roles is None or any(role is None for role in roles):
            raise ValueError(
                "Mapped source translation requires explicit volume source entity roles."
            )
        names = tuple(
            _entity(transfer.source_revision, role.value, int(index))
            for role, index in zip(
                roles, np.asarray(previous.source_indices), strict=True
            )
            if role is not None
        )
        output.append(
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                transfer.domain.source_id,
                transfer.source_revision,
                target.entity_set(dimension).entity_set_id,
                previous.target_global_ids,
                names,
                np.zeros(previous.target_global_ids.shape, dtype=np.float64),
                exact=True,
                source_dimensions=previous.source_dimensions,
                source_indices=previous.source_indices,
                source_entity_roles=roles,
                parameters=previous.parameters,
                orientations=orientations,
                parent_dimensions=np.full(
                    previous.target_global_ids.shape, dimension, dtype=np.int64
                ),
                parent_ids=previous.target_global_ids,
                parent_association_id=previous.association_id,
                provenance=GeometryAssociationProvenance.LINEAGE,
            )
        )
    patches, zones, labels = inherit_mesh_organization(
        source, target, identity_lineage(source.mesh, target)
    )
    return patches, zones, labels, tuple(output)
