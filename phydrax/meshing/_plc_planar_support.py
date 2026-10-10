#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lineage-owned support of represented planar regions and source segments.

Source dimensions are geometric (vertex zero, edge one, region two). Explicit
source entity roles distinguish planar regions from three-dimensional PLC facets.
Identity comes from authority tables and lineage; rational geometry proves support.
Triangle, bilinear Q1, and restricted Q1 coordinate polynomials stay authoritative:
native positivity, global embedding, and exact domain coverage prove their whole
images; polynomial edge restrictions prove their declared source-segment support.
"""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple, Protocol

import numpy as np

from .._physical import SpatialCoordinateContract
from ..discretization import _coordinate_enclosure as algebra, CellMesh
from ..discretization._cell_geometry import (
    _require_scalar_coordinate_element,
    CellGeometrySpec,
    RestrictedCellGeometryElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._mesh_certificates import (
    certify_domain_coverage,
    certify_global_embedding,
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
    MeshCertificateLimits,
    PiecewiseLinearDomain,
)
from ..geometry._planar_coverage import Point, Polygon, rational_points, turn
from ._association import (
    _child_sources,
    _entity_rows,
    _incidence_pairs,
    _ordered_vertices,
    _target_dimension,
    AssociationPropagationError,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
    PlcEntityClasses,
)
from ._lineage import EntityLineageKind, MeshLineage
from ._plc_mapped_support import _edge_support
from ._result import CellMeshingResult, require_original_meshing_source


class PlanarPlcTransfer(Protocol):
    """Immutable tables supplied by the represented source's transfer owner."""

    domain: PiecewiseLinearDomain
    coordinate_contract: SpatialCoordinateContract
    source_vertices: np.ndarray
    edge_vertices: np.ndarray
    vertex_indices: np.ndarray
    edge_indices: np.ndarray
    region_indices: np.ndarray
    source_revision: str
    maximum_support_queries: int


def _spend(work: list[int], count: int, /) -> None:
    work[0] -= count
    if work[0] < 0:
        raise ValueError("Planar PLC exact support query budget exhausted.")


def _position(indices: np.ndarray, identifier: int, /) -> int:
    rows = np.flatnonzero(indices == identifier)
    if rows.size != 1:
        raise ValueError("Planar PLC support names an undeclared authority index.")
    return int(rows[0])


def _vertices(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    if dimension == 0:
        return np.arange(mesh.entity_set(0).count, dtype=np.int64)[:, None]
    if dimension == 2:
        width = max(block.vertices.shape[1] for block in mesh.blocks)
        rows = np.full((mesh.entity_set(2).count, width), -1, dtype=np.int64)
        for block in mesh.blocks:
            positions = _entity_rows(
                mesh, 2, np.asarray(block.global_ids, dtype=np.int64)
            )
            rows[positions, : block.vertices.shape[1]] = np.asarray(
                block.vertices, dtype=np.int64
            )
        return rows
    rows = _ordered_vertices(mesh, dimension)
    if rows is None:
        raise ValueError(
            "Planar PLC transfer requires native triangle or quadrilateral incidence."
        )
    return rows


class _PlanarMap(NamedTuple):
    kind: str
    vertices: tuple[int, ...]
    coordinates: tuple[algebra.Polynomial, ...]


def _cell_maps(mesh: CellMesh, geometry: CellGeometrySpec, /) -> tuple[_PlanarMap, ...]:
    if mesh.topological_dimension != 2 or mesh.ambient_dimension != 2:
        raise ValueError("Planar PLC transfer requires actual two-dimensional cells.")
    elements, routes, values = geometry.resolve(mesh)
    controls = np.asarray(values, dtype=np.float64)
    carrier = rational_points(np.asarray(mesh.coordinates, dtype=np.float64))
    block_cells: list[_PlanarMap] = []
    block_ids: list[np.ndarray] = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        element = _require_scalar_coordinate_element(element, "Planar PLC transfer")
        if block.cell_kind not in ("triangle", "quadrilateral") or element.degree != 1:
            raise ValueError(
                "Planar PLC transfer requires actual degree-one native maps."
            )
        root = element
        while isinstance(root, RestrictedCellGeometryElement):
            root = root.source_element
        if root.degree != 1:
            raise ValueError(
                "Planar PLC restrictions require a degree-one source expression."
            )
        corners = rational_points(
            np.asarray(
                reference_cell_topology(block.cell_kind).vertices, dtype=np.float64
            )
        )
        for local, vertices in zip(
            controls[np.asarray(route, dtype=np.int64)],
            np.asarray(block.vertices, dtype=np.int64),
            strict=True,
        ):
            local = np.asarray(local, dtype=np.float64)
            vertices = np.asarray(vertices, dtype=np.int64)
            polynomial = algebra.coordinate_polynomials(element, local)
            if polynomial is None:
                raise ValueError("Planar PLC coordinate source expression is unresolved.")
            for corner, vertex in zip(corners, vertices.tolist(), strict=True):
                actual = tuple(algebra.evaluate(value, corner) for value in polynomial)
                if actual != carrier[vertex]:
                    raise ValueError(
                        "Planar carrier corners disagree with actual coordinate-map expressions."
                    )
            block_cells.append(
                _PlanarMap(block.cell_kind, tuple(vertices.tolist()), polynomial)
            )
        block_ids.append(np.asarray(block.global_ids, dtype=np.int64))
    order = _entity_rows(mesh, 2, np.concatenate(block_ids))
    inverse = np.argsort(order, kind="stable")
    return tuple(block_cells[index] for index in inverse.tolist())


def _prepare_planar_support(
    transfer: PlanarPlcTransfer,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    regions: np.ndarray,
    /,
    *,
    embedding: GlobalEmbeddingCertificate | None = None,
    coverage: DomainCoverageCertificate | None = None,
) -> tuple[tuple[_PlanarMap, ...], MeshCertificateLimits]:
    """Prove actual image coverage, including bilinear and restricted Q1 maps."""
    cells = _cell_maps(mesh, geometry)
    limits = MeshCertificateLimits(
        maximum_candidate_pairs=transfer.maximum_support_queries,
        maximum_ray_tests=transfer.maximum_support_queries,
        maximum_subdivision_pieces=transfer.maximum_support_queries,
    )
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    rows = _entity_rows(mesh, 2, ids)
    labels = np.asarray(
        [_position(transfer.region_indices, label) for label in regions[rows].tolist()],
        dtype=np.int64,
    )
    if mesh.storage is not None and (embedding is None or coverage is None):
        raise ValueError(
            "Owner-local PLC support requires current collective embedding and domain-coverage premises."
        )
    if embedding is None:
        policy = CellValidityPolicy(
            maximum_subdivision_depth=limits.maximum_subdivision_depth,
            maximum_piece_count=limits.maximum_subdivision_pieces,
            maximum_bernstein_nodes=limits.maximum_bernstein_nodes,
        )
        validity = certify_cell_geometry_validity(geometry, mesh=mesh, policy=policy)
        if not validity.all_certified:
            raise ValueError(
                "Planar PLC coordinate-map positivity is invalid or unresolved."
            )
        embedding = certify_global_embedding(mesh, geometry, validity, limits=limits)
    embedding.binding.require(mesh, geometry)
    if coverage is None:
        coverage = certify_domain_coverage(
            mesh, geometry, transfer.domain, labels, embedding=embedding, limits=limits
        )
    coverage.binding.require(mesh, geometry)
    if (
        embedding.status != "certified"
        or coverage.status != "certified"
        or coverage.domain_id != transfer.domain.domain_id
        or coverage.embedding_certificate_id != embedding.certificate_id
    ):
        raise ValueError(
            "Planar PLC source support requires current global embedding and complete exact domain coverage."
        )
    if coverage.region_ids != transfer.domain.region_ids:
        raise ValueError(
            "PLC coverage does not bind the authoritative source-region identities."
        )
    if (
        mesh.storage is not None
        and mesh.storage.evidence_id not in coverage.premise_certificate_ids
    ):
        raise ValueError(
            "Owner-local PLC coverage lacks the actual collective partition premise."
        )
    return cells, limits


def _edge_charts(
    cells: tuple[_PlanarMap, ...], vertices: np.ndarray, parents: np.ndarray, /
) -> tuple[tuple[algebra.Polynomial, ...], ...]:
    charts: list[tuple[algebra.Polynomial, ...]] = []
    for parent in parents.tolist():
        cell = cells[parent]
        local = np.asarray(
            [cell.vertices.index(vertex) for vertex in vertices.tolist()], dtype=np.int64
        )
        topology = reference_cell_topology(cell.kind)
        if not any(set(local.tolist()) == set(edge) for edge in topology.entities[1]):
            raise ValueError("Planar edge does not bind an actual reference-map edge.")
        corners = np.asarray(topology.vertices, dtype=np.float64)[local]
        arguments = algebra.affine_arguments(
            corners[0], (corners[1] - corners[0])[:, None]
        )
        charts.append(
            tuple(algebra.compose(value, arguments) for value in cell.coordinates)
        )
    if not charts:
        raise ValueError("Planar edge lacks actual coordinate-map incidence.")
    return tuple(charts)


def _segments(transfer: PlanarPlcTransfer, region: int, /) -> tuple[Polygon, ...]:
    local = _position(transfer.region_indices, region)
    pairs = transfer.domain.facet_regions
    rows = np.array(
        transfer.domain.facets[(pairs[:, 0] == local) | (pairs[:, 1] == local)],
        dtype=np.int64,
        copy=True,
    )
    selected = pairs[(pairs[:, 0] == local) | (pairs[:, 1] == local)]
    reverse = selected[:, 1] == local
    rows[reverse] = rows[reverse, ::-1]
    return tuple(rational_points(transfer.domain.vertices[row]) for row in rows)


def _on_segment(point: Point, segment: Polygon, /) -> bool:
    return turn(segment[0], segment[1], point) == 0 and all(
        min(a, b) <= p <= max(a, b)
        for a, b, p in zip(segment[0], segment[1], point, strict=True)
    )


def _in_region(point: Point, segments: tuple[Polygon, ...], /, *, closed: bool) -> bool:
    winding = 0
    for segment in segments:
        a, b = segment
        # A segment strictly above or below this exact ordinate can neither
        # contain the point nor cross its winding ray. Keep the full original
        # query charge; only the unnecessary determinant arithmetic is skipped.
        if (point[1] < a[1] and point[1] < b[1]) or (point[1] > a[1] and point[1] > b[1]):
            continue
        side = turn(a, b, point)
        if side == 0 and all(
            min(first, last) <= coordinate <= max(first, last)
            for first, last, coordinate in zip(a, b, point, strict=True)
        ):
            return closed
        if a[1] <= point[1] < b[1] and side > 0:
            winding += 1
        elif b[1] <= point[1] < a[1] and side < 0:
            winding -= 1
    return winding == 1


def _open_triangle_contact(triangle: Polygon, segment: Polygon, /) -> bool:
    """Exact strict half-plane feasibility along the complete source segment."""
    low, high = Fraction(0), Fraction(1)
    for a, b in zip(triangle, (*triangle[1:], triangle[0]), strict=True):
        first, last = turn(a, b, segment[0]), turn(a, b, segment[1])
        slope = last - first
        if slope == 0:
            if first <= 0:
                return False
        elif slope > 0:
            low = max(low, -first / slope)
        else:
            high = min(high, -first / slope)
    return low < high


def planar_point_support(
    transfer: PlanarPlcTransfer,
    dimension: int,
    index: int,
    point: np.ndarray,
    work: list[int],
    /,
    *,
    region_segments: tuple[Polygon, ...] | None = None,
) -> bool:
    p = rational_points(point[None, :])[0]
    if dimension == 0:
        _spend(work, 1)
        row = _position(transfer.vertex_indices, index)
        return p == rational_points(transfer.source_vertices[row : row + 1])[0]
    if dimension == 1:
        _spend(work, 1)
        edge = transfer.edge_vertices[_position(transfer.edge_indices, index)]
        return _on_segment(p, rational_points(transfer.source_vertices[edge]))
    if dimension == 2:
        segments = (
            _segments(transfer, index) if region_segments is None else region_segments
        )
        _spend(work, len(segments))
        return _in_region(p, segments, closed=False)
    raise ValueError("Planar PLC source strata are vertices, edges, and regions.")


def planar_cell_support(
    transfer: PlanarPlcTransfer,
    points: np.ndarray,
    region: int,
    work: list[int],
    /,
) -> bool:
    """Whole-cell containment: positive triangle, degree one, no interior cut.

    The center is rational, not a rounded float witness. A cavity, disconnected
    component, or interface inside the cell is caught by the complete source
    segment/open-triangle intersection, even when every cell corner is inside.
    """
    triangle = rational_points(points)
    if len(triangle) != 3 or turn(*triangle) <= 0:
        return False
    segments = _segments(transfer, region)
    _spend(work, 2 * len(segments))
    center = tuple(sum((p[axis] for p in triangle), Fraction(0)) / 3 for axis in range(2))
    return _in_region(center, segments, closed=False) and not any(
        _open_triangle_contact(triangle, segment) for segment in segments
    )


def _consensus(mesh: CellMesh, dimension: int, regions: np.ndarray, /) -> np.ndarray:
    if dimension == 2:
        return regions.copy()
    links = _incidence_pairs(mesh, dimension, 2)
    pairs = np.unique(np.stack((links[:, 0], regions[links[:, 1]]), axis=1), axis=0)
    rows, counts = np.unique(pairs[:, 0], return_counts=True)
    result = np.full((mesh.entity_set(dimension).count,), -1, dtype=np.int64)
    single = rows[counts == 1]
    result[single] = pairs[np.searchsorted(pairs[:, 0], single), 1]
    return result


def _roles(dimensions: np.ndarray, /) -> tuple[GeometrySourceEntityRole, ...]:
    """The planar source owner's explicit vertex/edge/region stratum schema."""
    schema = {
        0: GeometrySourceEntityRole.VERTEX,
        1: GeometrySourceEntityRole.EDGE,
        2: GeometrySourceEntityRole.REGION,
    }
    if any(dimension not in schema for dimension in dimensions.tolist()):
        raise ValueError("Planar sources require geometric dimensions zero, one, or two.")
    return tuple(schema[dimension] for dimension in dimensions.tolist())


def planar_source_associations(
    transfer: PlanarPlcTransfer,
    source: CellMeshingResult,
    /,
) -> tuple[GeometryAssociation, tuple[int, ...]]:
    if not isinstance(source, CellMeshingResult):
        raise TypeError("source must be CellMeshingResult.")
    if (
        transfer.domain.ambient_dimension != 2
        or source.coordinate_contract.spatial_id
        != transfer.coordinate_contract.spatial_id
    ):
        raise ValueError(
            "Planar PLC transfer requires its actual two-dimensional source frame."
        )
    _cell_maps(source.mesh, source.geometry)
    if source.mesh.storage is not None:
        # Owner-local epochs carry no copied serial certificate: the canonical
        # theorem chain reaches the original source, and this exact target
        # retains its own collectively recertified embedding and coverage.
        collective = source.collective_evidence
        if collective is None or source.collective_certificates is None:
            raise ValueError(
                "Owner-local planar sources require their collective theorem and target certificates."
            )
        collective.require_passed()
        storage = source.mesh.storage
        if (
            collective.mesh_id != source.mesh.mesh_id
            or collective.evidence_id != storage.evidence_id
            or collective.coordinate_geometry_id != cell_geometry_id(source.geometry)
        ):
            raise ValueError(
                "Owner-local planar theorem is not bound to this target publication."
            )
        if not isinstance(require_original_meshing_source(source), CellMeshingResult):
            raise ValueError(
                "Planar PLC transfer requires an original serial certified planar source."
            )
        embedding, coverage = source.collective_certificates
        embedding.binding.require(source.mesh, source.geometry)
        coverage.binding.require(source.mesh, source.geometry)
        if (
            embedding.status != "certified"
            or coverage.status != "certified"
            or coverage.domain_id != transfer.domain.domain_id
            or storage.evidence_id not in coverage.premise_certificate_ids
        ):
            raise ValueError(
                "Owner-local planar target lacks its current certified embedding and coverage."
            )
    else:
        certificate = source.certification
        if (
            certificate is None
            or not certificate.passed
            or certificate.mesh_id != source.mesh.mesh_id
        ):
            raise ValueError(
                "Planar PLC transfer requires the source's current passed certification."
            )
        embedding = certificate.embedding
        if embedding is None:
            raise ValueError(
                "Planar source requires current independent global embedding."
            )
        embedding.binding.require(source.mesh, source.geometry)
        if certificate.coverage is not None:
            certificate.coverage.binding.require(source.mesh, source.geometry)
    dimensions: list[int] = []
    vertex: GeometryAssociation | None = None
    namespaces = {
        0: transfer.vertex_indices,
        1: transfer.edge_indices,
        2: transfer.region_indices,
    }
    for association in source.associations:
        if (
            association.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
            or association.source_id != transfer.domain.source_id
            or association.source_revision != transfer.source_revision
        ):
            raise ValueError(
                "Planar PLC transfer binds another represented source revision."
            )
        dimension = _target_dimension(source.mesh, association)
        if dimension in dimensions:
            raise ValueError("Planar source requires one association per mesh dimension.")
        association.validate_target(source.mesh.entity_set(dimension))
        if not association.complete:
            raise ValueError(
                "Planar source associations must be resolved and unambiguous."
            )
        dims = np.asarray(association.source_dimensions, dtype=np.int64)
        indices = np.asarray(association.source_indices, dtype=np.int64)
        for kind in np.unique(dims).tolist():
            if kind not in namespaces or not np.all(
                np.isin(indices[dims == kind], namespaces[kind])
            ):
                raise ValueError(
                    "Planar source association names undeclared authority indices."
                )
        if association.source_entity_roles != _roles(dims):
            raise ValueError(
                "Planar source entity roles disagree with its vertex/edge/region topology."
            )
        if dimension == 2:
            association.target_rows(
                np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)
            )
            if np.any(dims != 2):
                raise ValueError("Planar cells must name authoritative source regions.")
        if dimension == 0:
            vertex = association
            association.target_rows(
                np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
            )
        dimensions.append(dimension)
    if vertex is None or 1 not in dimensions or 2 not in dimensions:
        raise ValueError(
            "Planar PLC transfer requires complete vertex and cell associations and source constraints."
        )
    return vertex, tuple(sorted(d for d in dimensions if d != 0))


def _vertex_closure(
    transfer: PlanarPlcTransfer,
    mesh: CellMesh,
    levels: list[PlcEntityClasses],
    eligible: np.ndarray,
    /,
) -> None:
    links = _incidence_pairs(mesh, 0, 1)
    vertices, edges = levels[0], levels[1]
    for row in np.flatnonzero(eligible).tolist():
        incident = links[links[:, 0] == row, 1]
        identifiers = np.unique(edges.indices[incident][edges.dimensions[incident] == 1])
        if identifiers.size == 0:
            continue
        if identifiers.size == 1:
            vertices.dimensions[row], vertices.indices[row] = 1, identifiers[0]
            continue
        common: set[int] | None = None
        for identifier in identifiers.tolist():
            endpoints = set(
                transfer.edge_vertices[
                    _position(transfer.edge_indices, identifier)
                ].tolist()
            )
            common = endpoints if common is None else common & endpoints
        if common is None or len(common) != 1:
            raise AssociationPropagationError(
                "Planar incident source edges lack a unique authoritative corner.",
                np.asarray(mesh.vertex_global_ids)[row : row + 1],
            )
        vertices.dimensions[row], vertices.indices[row] = (
            0,
            transfer.vertex_indices[next(iter(common))],
        )


def _prove_levels(
    transfer: PlanarPlcTransfer,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    levels: tuple[PlcEntityClasses, ...] | list[PlcEntityClasses],
    work: list[int],
    /,
    *,
    embedding: GlobalEmbeddingCertificate | None = None,
    coverage: DomainCoverageCertificate | None = None,
    judged: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, ...]:
    """Exact support of every class; ``judged`` limits owner-local proofs.

    An owner-local entity outside ``judged`` has a truncated resident star; its
    edge incidence and charts are incomplete, so it receives no local verdict.
    """
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    signs: list[np.ndarray] = []
    edge_links = _incidence_pairs(mesh, 1, 2)
    edge_cells = np.bincount(edge_links[:, 0], minlength=mesh.entity_set(1).count)
    edge_regions = _consensus(mesh, 1, levels[2].indices)
    cells, limits = _prepare_planar_support(
        transfer,
        mesh,
        geometry,
        levels[2].indices,
        embedding=embedding,
        coverage=coverage,
    )
    # Exact authored endpoints are immutable within this proof. Retain one
    # bank per used region, not one reconstruction per mesh vertex; every
    # predicate visit still consumes the original support-query allowance.
    region_segments: dict[int, tuple[Polygon, ...]] = {}
    for dimension, level in enumerate(levels):
        identifiers = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        orientations = np.zeros(identifiers.shape, dtype=np.int8)
        for row, vertices in enumerate(_vertices(mesh, dimension)):
            if judged is not None and not judged[dimension][row]:
                continue
            kind, index = int(level.dimensions[row]), int(level.indices[row])
            points = coordinates[vertices[vertices >= 0]]
            if dimension == 2:
                supported = kind == 2
            elif dimension == 1 and kind == 2:
                # An interior edge may end on a lower stratum. Whole-map coverage
                # proves its region support without misclassifying its endpoints.
                supported = edge_regions[row] == index and edge_cells[row] == 2
            else:
                if kind == 2 and index not in region_segments:
                    region_segments[index] = _segments(transfer, index)
                supported = all(
                    planar_point_support(
                        transfer,
                        kind,
                        index,
                        point,
                        work,
                        region_segments=region_segments.get(index) if kind == 2 else None,
                    )
                    for point in points
                )
                if dimension == 1 and kind == 1 and supported:
                    edge = transfer.edge_vertices[_position(transfer.edge_indices, index)]
                    charts = _edge_charts(
                        cells, vertices, edge_links[edge_links[:, 0] == row, 1]
                    )
                    outcomes = tuple(
                        _edge_support(chart, transfer.source_vertices[edge], limits, work)
                        for chart in charts
                    )
                    realized = {
                        orientation for supported_, orientation in outcomes if supported_
                    }
                    supported = (
                        all(supported_ for supported_, _ in outcomes)
                        and len(realized) == 1
                    )
                    if supported:
                        orientations[row] = next(iter(realized))
            if not supported:
                raise AssociationPropagationError(
                    "Planar coordinates lack their inherited exact source support.",
                    identifiers[row : row + 1],
                )
        signs.append(orientations)
    return tuple(signs)


def _finish_classes(
    transfer: PlanarPlcTransfer, levels: list[PlcEntityClasses], /
) -> tuple[PlcEntityClasses, ...]:
    codes = {
        kind: {label: offset_ + row for row, label in enumerate(indices.tolist())}
        for kind, indices, offset_ in (
            (0, transfer.vertex_indices, 0),
            (1, transfer.edge_indices, transfer.vertex_indices.size),
            (
                2,
                transfer.region_indices,
                transfer.vertex_indices.size + transfer.edge_indices.size,
            ),
        )
    }
    for level in levels:
        level.resolved[:] = (level.dimensions >= 0) & (level.indices >= 0)
        if not np.all(level.resolved):
            raise ValueError(
                "Planar source strata lack unambiguous authoritative incidence."
            )
        for row, (kind, index) in enumerate(
            zip(level.dimensions.tolist(), level.indices.tolist(), strict=True)
        ):
            level.codes[row] = codes[kind][index]
        for value in level:
            value.setflags(write=False)
    return tuple(levels)


def planar_classes(
    transfer: PlanarPlcTransfer, source: CellMeshingResult, /
) -> tuple[PlcEntityClasses, ...]:
    planar_source_associations(transfer, source)
    cell = next(
        value
        for value in source.associations
        if _target_dimension(source.mesh, value) == 2
    )
    regions = np.asarray(cell.source_indices, dtype=np.int64)[
        cell.target_rows(np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64))
    ]
    levels: list[PlcEntityClasses] = []
    for dimension in range(3):
        indices = _consensus(source.mesh, dimension, regions)
        dims = np.where(indices >= 0, 2, -1).astype(np.int64)
        for association in source.associations:
            if _target_dimension(source.mesh, association) == dimension:
                rows = _entity_rows(
                    source.mesh,
                    dimension,
                    np.asarray(association.target_global_ids, dtype=np.int64),
                )
                dims[rows], indices[rows] = (
                    np.asarray(association.source_dimensions),
                    np.asarray(association.source_indices),
                )
        levels.append(
            PlcEntityClasses(
                dims,
                indices,
                np.zeros(dims.shape, dtype=np.bool_),
                np.empty(dims.shape, dtype=np.int64),
            )
        )
    # planar_source_associations admitted exactly one of these premise owners.
    if source.mesh.storage is not None and source.collective_certificates is not None:
        embedding, coverage = source.collective_certificates
    elif source.certification is not None:
        embedding, coverage = (
            source.certification.embedding,
            source.certification.coverage,
        )
    else:
        raise ValueError("Planar source support requires current certification.")
    orientations = _prove_levels(
        transfer,
        source.mesh,
        source.geometry,
        levels,
        [transfer.maximum_support_queries],
        embedding=embedding,
        coverage=coverage,
    )
    for association in source.associations:
        if _target_dimension(source.mesh, association) == 1:
            rows = _entity_rows(
                source.mesh, 1, np.asarray(association.target_global_ids, dtype=np.int64)
            )
            declared = np.asarray(association.orientations, dtype=np.int8)
            if np.any((declared != 0) & (declared != orientations[1][rows])):
                raise ValueError(
                    "Planar source edge orientation contradicts its actual authoritative support."
                )
    return _finish_classes(transfer, levels)


def planar_protected_edges(
    transfer: PlanarPlcTransfer,
    source: CellMeshingResult,
    /,
    *,
    midpoint_required: bool = True,
) -> np.ndarray:
    levels = planar_classes(transfer, source)
    endpoints = np.asarray(source.mesh.coordinates, dtype=np.float64)[
        _vertices(source.mesh, 1)
    ]
    midpoints = np.sum(endpoints * np.float64(0.5), axis=1)
    protected = np.all(midpoints == endpoints[:, 0], axis=1) | np.all(
        midpoints == endpoints[:, 1], axis=1
    )
    if midpoint_required:
        work = [transfer.maximum_support_queries]
        for row in np.flatnonzero(levels[1].dimensions == 1).tolist():
            protected[row] |= not planar_point_support(
                transfer, 1, int(levels[1].indices[row]), midpoints[row], work
            )
    return protected


def _complete_stars(mesh: CellMesh, /) -> tuple[np.ndarray, ...]:
    """Owner-local entities whose every incident cell is resident.

    A resident cell is complete when each of its facets is matched or physical
    boundary. An entity whose resident incident cells are all complete cannot
    reach a missing cell across a facet, so its resident star is its whole
    star. A star truncated at the artificial halo boundary yields no verdict.
    """
    storage = mesh.storage
    if storage is None or storage.local_neighborhood_complete is None:
        raise ValueError(
            "Owner-local planar transfer requires resident-star completeness receipts."
        )
    complete = np.asarray(storage.local_neighborhood_complete, dtype=np.bool_)
    rows = _entity_rows(mesh, 2, np.asarray(storage.entity_global_ids[2], dtype=np.int64))
    count = mesh.entity_set(2).count
    if (
        complete.shape != rows.shape
        or rows.size != count
        or np.any(rows < 0)
        or np.unique(rows).size != count
    ):
        raise ValueError(
            "Owner-local neighborhood completeness must label every resident cell once."
        )
    cells = np.zeros((count,), dtype=np.bool_)
    cells[rows] = complete
    stars: list[np.ndarray] = []
    for dimension in (0, 1):
        links = _incidence_pairs(mesh, dimension, 2)
        truncated = np.zeros((mesh.entity_set(dimension).count,), dtype=np.bool_)
        np.logical_or.at(truncated, links[:, 0], ~cells[links[:, 1]])
        stars.append(~truncated)
    stars.append(cells)
    return tuple(stars)


_VERDICT_COLUMNS = 8
_LevelParents = list[tuple[np.ndarray, np.ndarray]]
_LocalVerdicts = tuple[
    list[PlcEntityClasses], _LevelParents, tuple[np.ndarray, ...], tuple[np.ndarray, ...]
]


def _agreed_owner_verdicts(
    target: CellMesh,
    local: _LocalVerdicts | None,
    failure: ValueError | None,
    /,
) -> tuple[list[PlcEntityClasses], _LevelParents, tuple[np.ndarray, ...]]:
    """Collective: every global entity has one agreed complete-star verdict.

    Every process enters both gathers, including one whose local proof failed,
    so no owner waits on a peer that already refused. Owners judging the same
    global entity must agree exactly on its stratum and orientation; an entity
    judged by no owner is refused, never inferred from a truncated star. Each
    resident row adopts the verdict and parent of its lowest judging process.
    """
    from jax.experimental import multihost_utils

    storage = target.storage
    if storage is None:
        raise ValueError("Owner-local planar verdicts require canonical mesh storage.")
    ids = tuple(
        np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
        for dimension in range(3)
    )
    packets: list[np.ndarray] = []
    for dimension in range(3):
        if local is None:
            packets.append(np.empty((0, _VERDICT_COLUMNS), dtype=np.int64))
            continue
        levels, parents, orientations, judged = local
        rows = np.flatnonzero(judged[dimension])
        packets.append(
            np.stack(
                (
                    np.ones(rows.shape, dtype=np.int64),
                    np.full(rows.shape, dimension, dtype=np.int64),
                    ids[dimension][rows],
                    levels[dimension].dimensions[rows],
                    levels[dimension].indices[rows],
                    orientations[dimension][rows].astype(np.int64),
                    parents[dimension][0][rows].astype(np.int64),
                    parents[dimension][1][rows],
                ),
                axis=1,
            )
        )
    sizes = np.asarray(
        (*(packet.shape[0] for packet in packets), int(failure is not None)),
        dtype=np.int64,
    )
    gathered_sizes = np.asarray(
        multihost_utils.process_allgather(sizes, tiled=False), dtype=np.int64
    ).reshape((-1, 4))
    if np.any(gathered_sizes[:, 3]):
        if failure is not None:
            raise failure
        raise ValueError(
            "An owning process refused its complete-star planar source verdicts."
        )
    if local is None:
        raise ValueError("Owner-local planar verdicts lost their local proof.")
    packed = np.zeros(
        (int(np.sum(np.max(gathered_sizes[:, :3], axis=0))), _VERDICT_COLUMNS),
        dtype=np.int64,
    )
    flat = np.concatenate(packets, axis=0)
    packed[: flat.shape[0]] = flat
    table = np.asarray(
        multihost_utils.process_allgather(packed, tiled=False), dtype=np.int64
    ).reshape((-1, _VERDICT_COLUMNS))
    table = table[table[:, 0] == 1]
    levels, parents, orientations, _ = local
    adopted = []
    for dimension in range(3):
        rows = table[table[:, 1] == dimension]
        rows = rows[np.argsort(rows[:, 2], kind="stable")]
        unique, first = np.unique(rows[:, 2], return_index=True)
        if unique.size != storage.global_entity_counts[dimension]:
            raise AssociationPropagationError(
                "A global planar entity has no owner holding its complete star.",
                np.setdiff1d(ids[dimension], unique),
            )
        group = np.searchsorted(unique, rows[:, 2])
        conflicting = np.any(rows[:, 3:6] != rows[first[group], 3:6], axis=1)
        if np.any(conflicting):
            raise AssociationPropagationError(
                "Owners disagree on a complete-star planar source verdict.",
                np.unique(rows[conflicting, 2]),
            )
        positions = np.searchsorted(unique, ids[dimension])
        if np.any(positions >= unique.size) or np.any(
            unique[np.minimum(positions, unique.size - 1)] != ids[dimension]
        ):
            raise AssociationPropagationError(
                "A resident planar entity lies outside the agreed global verdicts.",
                ids[dimension],
            )
        verdicts = rows[first[positions]]
        levels[dimension].dimensions[:] = verdicts[:, 3]
        levels[dimension].indices[:] = verdicts[:, 4]
        parents[dimension][0][:] = verdicts[:, 6]
        parents[dimension][1][:] = verdicts[:, 7]
        adopted.append(verdicts[:, 5].astype(np.int8))
    return levels, parents, tuple(adopted)


def _propagated_levels(
    transfer: PlanarPlcTransfer,
    source: CellMeshingResult,
    lineage: MeshLineage,
    target: CellMesh,
    geometry: CellGeometrySpec,
    judged: tuple[np.ndarray, ...] | None,
    /,
    *,
    embedding: GlobalEmbeddingCertificate | None,
    coverage: DomainCoverageCertificate | None,
) -> tuple[
    list[PlcEntityClasses], list[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, ...]
]:
    """Lineage, consensus and closure classes with their exact support proof.

    ``judged`` restricts locally inferred classes and proofs to owner-local
    complete stars; ``None`` decides every entity of a serial mesh.
    """
    _cell_maps(target, geometry)
    old = planar_classes(transfer, source)
    levels, parents = _lineage_levels(source.mesh, target, lineage, old)
    cell_ids = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)
    if np.any(levels[2].dimensions != 2):
        raise AssociationPropagationError(
            "Planar target cells lack unambiguous source region lineage.",
            cell_ids[levels[2].dimensions != 2],
        )
    for dimension in (1, 0):
        level = levels[dimension]
        decided = (
            np.ones(level.dimensions.shape, dtype=np.bool_)
            if judged is None
            else judged[dimension]
        )
        if dimension == 0:
            missing = np.flatnonzero(level.dimensions < 0)
            if missing.size:
                ids = np.asarray(target.vertex_global_ids, dtype=np.int64)[missing]
                order = np.argsort(ids, kind="stable")
                dims, rows = _child_sources(lineage, source.mesh, target, ids[order])
                for row, kind, parent in zip(
                    missing[order].tolist(), dims.tolist(), rows.tolist(), strict=True
                ):
                    if kind >= 0 and parent >= 0:
                        level.dimensions[row], level.indices[row] = (
                            old[kind].dimensions[parent],
                            old[kind].indices[parent],
                        )
                        parents[0][0][row] = kind
                        parents[0][1][row] = np.asarray(
                            source.mesh.entity_set(kind).entity_ids
                        )[parent]
        regions = _consensus(target, dimension, levels[2].indices)
        missing = (level.dimensions < 0) & (regions >= 0) & decided
        links = _incidence_pairs(target, dimension, 2)
        for row in np.flatnonzero(missing).tolist():
            cells = links[links[:, 0] == row, 1]
            if cells.size == 0:
                raise ValueError(
                    "Planar created stratum has no authoritative cell lineage."
                )
            cell = int(cells[0])
            parents[dimension][0][row] = 2
            parents[dimension][1][row] = parents[2][1][cell]
        if dimension == 0:
            _vertex_closure(transfer, target, levels, (level.dimensions < 0) & decided)
        missing = (level.dimensions < 0) & (regions >= 0) & decided
        level.dimensions[missing], level.indices[missing] = 2, regions[missing]
    orientations = _prove_levels(
        transfer,
        target,
        geometry,
        levels,
        [transfer.maximum_support_queries],
        embedding=embedding,
        coverage=coverage,
        judged=judged,
    )
    return levels, parents, orientations


def _lineage_levels(
    source: CellMesh,
    target: CellMesh,
    lineage: MeshLineage,
    old: tuple[PlcEntityClasses, ...],
    /,
) -> tuple[list[PlcEntityClasses], list[tuple[np.ndarray, np.ndarray]]]:
    levels: list[PlcEntityClasses] = []
    parents: list[tuple[np.ndarray, np.ndarray]] = []
    for dimension in range(3):
        ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
        dims, indices = (
            np.full(ids.shape, -1, dtype=np.int64),
            np.full(ids.shape, -1, dtype=np.int64),
        )
        parent_dims, parent_ids = (
            np.full(ids.shape, -1, dtype=np.int8),
            np.full(ids.shape, -1, dtype=np.int64),
        )
        record = lineage.entity_lineage(dimension)
        if (
            record.source_entity_set_id != source.entity_set(dimension).entity_set_id
            or record.target_entity_set_id != target.entity_set(dimension).entity_set_id
        ):
            raise ValueError(
                "Planar lineage binds different source or target entity sets."
            )
        if np.any(np.asarray(record.relation_kinds) == EntityLineageKind.UNKNOWN):
            raise AssociationPropagationError(
                "Unknown lineage cannot identify planar source strata.", ids
            )
        source_ids = np.asarray(record.source_global_ids, dtype=np.int64)
        source_rows = _entity_rows(source, dimension, source_ids)
        target_rows = _entity_rows(
            target, dimension, np.asarray(record.target_global_ids, dtype=np.int64)
        )
        if np.any(source_rows < 0) or np.any(target_rows < 0):
            raise ValueError("Planar lineage names undeclared mesh entities.")
        for row in np.unique(target_rows).tolist():
            selected = target_rows == row
            rows = source_rows[selected]
            candidates = np.unique(
                np.stack(
                    (old[dimension].dimensions[rows], old[dimension].indices[rows]),
                    axis=1,
                ),
                axis=0,
            )
            if candidates.shape[0] == 1:
                dims[row], indices[row] = candidates[0]
                parent_dims[row], parent_ids[row] = (
                    dimension,
                    np.min(source_ids[selected]),
                )
        levels.append(
            PlcEntityClasses(
                dims,
                indices,
                np.zeros(ids.shape, dtype=np.bool_),
                np.empty(ids.shape, dtype=np.int64),
            )
        )
        parents.append((parent_dims, parent_ids))
    return levels, parents


def _preserved_parameters(
    source: CellMesh,
    target: CellMesh,
    dimension: int,
    previous: GeometryAssociation,
    level: PlcEntityClasses,
    /,
) -> np.ndarray:
    """Retain provided parameters only on the same unchanged geometric entity."""
    ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
    parameters = np.zeros((ids.size, 2), dtype=np.float64)
    records = {
        identifier: row
        for row, identifier in enumerate(
            np.asarray(previous.target_global_ids, dtype=np.int64).tolist()
        )
    }
    source_rows = _entity_rows(source, dimension, ids)
    old_dimensions = np.asarray(previous.source_dimensions, dtype=np.int64)
    old_indices = np.asarray(previous.source_indices, dtype=np.int64)
    old_parameters = np.asarray(previous.parameters, dtype=np.float64)
    source_vertices, target_vertices = (
        _vertices(source, dimension),
        _vertices(target, dimension),
    )
    source_ids, target_ids = (
        np.asarray(source.vertex_global_ids, dtype=np.int64),
        np.asarray(target.vertex_global_ids, dtype=np.int64),
    )
    source_points, target_points = (
        np.asarray(source.coordinates, dtype=np.float64),
        np.asarray(target.coordinates, dtype=np.float64),
    )
    for row, identifier in enumerate(ids.tolist()):
        original = records.get(identifier)
        if original is None or source_rows[row] < 0:
            continue
        if (
            old_dimensions[original] != level.dimensions[row]
            or old_indices[original] != level.indices[row]
        ):
            continue
        old = source_vertices[source_rows[row]]
        new = target_vertices[row]
        old, new = old[old >= 0], new[new >= 0]
        old = old[np.argsort(source_ids[old], kind="stable")]
        new = new[np.argsort(target_ids[new], kind="stable")]
        if np.array_equal(source_ids[old], target_ids[new]) and np.array_equal(
            source_points[old], target_points[new]
        ):
            parameters[row] = old_parameters[original]
    return parameters


def _require_lineage(
    source: CellMeshingResult, lineage: MeshLineage, target: CellMesh, /
) -> None:
    if (
        not isinstance(lineage, MeshLineage)
        or lineage.source_topology_id != source.mesh.topology_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError(
            "Planar PLC transfer requires the actual source-to-target lineage."
        )


def planar_propagate(
    transfer: PlanarPlcTransfer,
    source: CellMeshingResult,
    lineage: MeshLineage,
    target: CellMesh,
    /,
    *,
    geometry: CellGeometrySpec,
    embedding: GlobalEmbeddingCertificate | None = None,
    coverage: DomainCoverageCertificate | None = None,
) -> tuple[GeometryAssociation, ...]:
    # Source checks bind only the shared serial source, so every owner reaches
    # the same verdict; target-dependent proofs stay inside the collective.
    vertex, dimensions = planar_source_associations(transfer, source)
    if target.storage is None:
        _require_lineage(source, lineage, target)
        levels, parents, orientations = _propagated_levels(
            transfer,
            source,
            lineage,
            target,
            geometry,
            None,
            embedding=embedding,
            coverage=coverage,
        )
    else:
        # Owner-local: decide only complete resident stars, then adopt the one
        # collectively agreed verdict of every global entity.
        local: _LocalVerdicts | None = None
        failure: ValueError | None = None
        try:
            _require_lineage(source, lineage, target)
            judged = _complete_stars(target)
            levels, parents, orientations = _propagated_levels(
                transfer,
                source,
                lineage,
                target,
                geometry,
                judged,
                embedding=embedding,
                coverage=coverage,
            )
            local = (levels, parents, orientations, judged)
        except ValueError as error:
            failure = error
        levels, parents, orientations = _agreed_owner_verdicts(target, local, failure)
    final = _finish_classes(transfer, levels)
    output: list[GeometryAssociation] = []
    for dimension in (0, *dimensions):
        previous = (
            vertex
            if dimension == 0
            else next(
                value
                for value in source.associations
                if _target_dimension(source.mesh, value) == dimension
            )
        )
        level = final[dimension]
        ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
        roles = _roles(level.dimensions)
        names = tuple(
            f"{transfer.source_revision}:{role.value}:{index}"
            for role, index in zip(roles, level.indices.tolist(), strict=True)
        )
        output.append(
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                transfer.domain.source_id,
                transfer.source_revision,
                target.entity_set(dimension).entity_set_id,
                ids,
                names,
                np.zeros(ids.shape, dtype=np.float64),
                exact=True,
                source_dimensions=level.dimensions,
                source_indices=level.indices,
                source_entity_roles=roles,
                parameters=_preserved_parameters(
                    source.mesh, target, dimension, previous, level
                ),
                orientations=orientations[dimension],
                parent_dimensions=parents[dimension][0],
                parent_ids=parents[dimension][1],
                parent_association_id=previous.association_id,
                provenance=GeometryAssociationProvenance.LINEAGE,
            )
        )
    return tuple(output)
