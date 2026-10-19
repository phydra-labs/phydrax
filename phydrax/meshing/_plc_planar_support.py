#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lineage-owned support of represented planar regions and source segments.

Source dimensions are geometric (vertex zero, edge one, region two). Explicit
source entity roles distinguish planar regions from three-dimensional PLC facets.
Identity comes from source ownership tables and lineage; rational geometry proves support.
Triangle, bilinear Q1, and restricted Q1 coordinate polynomials stay declared:
native positivity, global embedding, and rational domain coverage prove their whole
images; polynomial edge restrictions prove their declared source-segment support.
"""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple, Protocol

import numpy as np
from jax.experimental import multihost_utils

import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.geometry._planar_primitives as planar_primitives_owner
import phydrax.meshing._association_routes as association_routes
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._lineage_records as _dependency_phydrax_meshing__lineage
import phydrax.meshing._plc_mapped_support as _dependency_phydrax_meshing__plc_mapped_support
import phydrax.meshing._result as _dependency_phydrax_meshing__result
from phydrax.discretization._coordinate_identity import coordinate_scope_key

from .._physical import SpatialCoordinateContract
from ..discretization import _coordinate_enclosure as algebra, CellMesh
from ..discretization._reference_cell import reference_cell_topology
from ..typing import checked
from ._association_routes import PlcEntityClasses
from ._support_work import spend_support_work


class PlanarPlcTransfer(Protocol):
    """Immutable tables supplied by the represented source's transfer owner."""

    domain: domain_coverage_owner.PiecewiseLinearDomain
    coordinate_contract: SpatialCoordinateContract
    source_vertices: np.ndarray
    edge_vertices: np.ndarray
    vertex_indices: np.ndarray
    edge_indices: np.ndarray
    region_indices: np.ndarray
    source_revision: str
    maximum_support_queries: int


def _position(indices: np.ndarray, identifier: int, /) -> int:
    rows = np.flatnonzero(indices == identifier)
    if rows.size != 1:
        raise ValueError(
            "PlanarPlcSupport.source_index must name one declared source row; got an undeclared or duplicate index"
        )
    return int(rows[0])


def _vertices(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    if dimension == 0:
        return np.arange(mesh.entity_set(0).count, dtype=np.int64)[:, None]
    if dimension == 2:
        width = max(block.vertices.shape[1] for block in mesh.blocks)
        rows = np.full((mesh.entity_set(2).count, width), -1, dtype=np.int64)
        for block in mesh.blocks:
            positions = association_state_owner.entity_rows(
                mesh, 2, np.asarray(block.global_ids, dtype=np.int64)
            )
            rows[positions, : block.vertices.shape[1]] = np.asarray(
                block.vertices, dtype=np.int64
            )
        return rows
    rows = association_routes.ordered_vertices(mesh, dimension)
    if rows is None:
        raise ValueError(
            "PlanarPlcSupport.mesh must expose triangle or quadrilateral incidence; got unsupported incidence"
        )
    return rows


class PlanarMap(NamedTuple):
    kind: str
    vertices: tuple[int, ...]
    coordinates: tuple[algebra.Polynomial, ...]


@checked
def cell_maps(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[PlanarMap, ...]:
    with budget.activate():
        if mesh.topological_dimension != 2 or mesh.ambient_dimension != 2:
            raise ValueError(
                "PlanarPlcSupport.mesh must have topology and ambient dimension 2; got different dimensions"
            )
        key = coordinate_scope_key(mesh, geometry)
        cached = budget.planar_cell_map_cache.get(key)
        if cached is not None:
            budget.reserve(0)
            return cached
        elements, routes, _ = geometry.resolve(mesh, budget=budget)
        controls = algebra.prepared_coordinate_source_bank(geometry, budget=budget)
        carrier = planar_primitives_owner.rational_points(
            np.asarray(mesh.coordinates, dtype=np.float64), budget=budget
        )
        expected_corners: dict[int, tuple[algebra.ExactScalar, ...]] = (
            dict(enumerate(carrier)) if geometry.exact_source is None else {}
        )
        block_cells: list[PlanarMap] = []
        block_ids: list[np.ndarray] = []
        for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
            element = _dependency_phydrax_discretization__cell_geometry.require_scalar_coordinate_element(
                element, "Planar PLC transfer"
            )
            if (
                block.cell_kind not in ("triangle", "quadrilateral")
                or element.degree != 1
            ):
                raise ValueError(
                    "PlanarPlcSupport.elements must be degree-one triangle or quadrilateral maps; got unsupported elements"
                )
            root = element
            while isinstance(
                root,
                _dependency_phydrax_discretization__cell_geometry.RestrictedCellGeometryElement,
            ):
                root = root.source_element
            if root.degree != 1:
                raise ValueError(
                    "PlanarPlcSupport.root_element must have degree 1; got a different source degree"
                )
            for route_row, vertices in zip(
                np.asarray(route, dtype=np.int64),
                np.asarray(block.vertices, dtype=np.int64),
                strict=True,
            ):
                local = tuple(
                    controls[int(index)]
                    for index in np.asarray(route_row, dtype=np.int64).reshape((-1,))
                )
                vertices = np.asarray(vertices, dtype=np.int64)
                corners = algebra.coordinate_corner_images(element, local, budget=budget)
                if corners is None:
                    raise ValueError(
                        "PlanarPlcSupport.coordinates must resolve source expressions; got unresolved expressions"
                    )
                polynomial = algebra.coordinate_polynomials(element, local, budget=budget)
                if polynomial is None:
                    raise ValueError(
                        "PlanarPlcSupport.coordinates must resolve source expressions; got unresolved expressions"
                    )
                for actual, vertex in zip(corners, vertices.tolist(), strict=True):
                    prior = expected_corners.setdefault(vertex, actual)
                    if actual != prior:
                        raise ValueError(
                            "PlanarPlcSupport.carrier must agree with coordinate-map corner images; got different corners"
                        )
                block_cells.append(
                    PlanarMap(block.cell_kind, tuple(vertices.tolist()), polynomial)
                )
            block_ids.append(np.asarray(block.global_ids, dtype=np.int64))
        order = association_state_owner.entity_rows(mesh, 2, np.concatenate(block_ids))
        inverse = np.argsort(order, kind="stable")
        maps = tuple(block_cells[index] for index in inverse.tolist())
        budget.retain_basis((key, maps))
        budget.planar_cell_map_cache[key] = maps
        return maps


def _prepare_planar_support(
    transfer: PlanarPlcTransfer,
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    regions: np.ndarray,
    /,
    *,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None = None,
    coverage: domain_coverage_owner.DomainCoverageCertificate | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    tuple[PlanarMap, ...],
    _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
]:
    """Prove represented image coverage, including bilinear and restricted Q1 maps."""
    cells = cell_maps(mesh, geometry, budget=budget)
    limits = _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits(
        maximum_candidate_pairs=transfer.maximum_support_queries,
        maximum_ray_tests=transfer.maximum_support_queries,
        maximum_subdivision_pieces=transfer.maximum_support_queries,
    )
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    rows = association_state_owner.entity_rows(mesh, 2, ids)
    labels = np.asarray(
        [_position(transfer.region_indices, label) for label in regions[rows].tolist()],
        dtype=np.int64,
    )
    if mesh.storage is not None and (embedding is None or coverage is None):
        raise ValueError(
            "PlanarPlcSupport.storage must retain collective embedding and domain coverage; got missing certificates"
        )
    if embedding is None:
        policy = (
            _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy(
                maximum_subdivision_depth=limits.maximum_subdivision_depth,
                maximum_piece_count=limits.maximum_subdivision_pieces,
                maximum_bernstein_nodes=limits.maximum_bernstein_nodes,
            )
        )
        validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
            geometry, mesh=mesh, policy=policy, budget=budget
        )
        if not validity.all_certified:
            raise ValueError(
                "PlanarPlcSupport.validity must certify coordinate-map positivity; got invalid or unresolved cells"
            )
        embedding = global_embedding_owner.certify_global_embedding(
            mesh, geometry, validity, limits=limits, budget=budget
        )
    embedding.binding.require(mesh, geometry)
    if coverage is None:
        coverage = domain_coverage_owner.certify_domain_coverage(
            mesh,
            geometry,
            transfer.domain,
            labels,
            embedding=embedding,
            limits=limits,
            budget=budget,
        )
    coverage.binding.require(mesh, geometry)
    if (
        embedding.status != "certified"
        or coverage.status != "certified"
        or coverage.domain_id != transfer.domain.domain_id
        or coverage.embedding_certificate_id != embedding.certificate_id
    ):
        raise ValueError(
            "PlanarPlcSupport.coverage must bind certified embedding and whole-domain coverage; got failed or different certificates"
        )
    if coverage.region_ids != transfer.domain.region_ids:
        raise ValueError(
            "PlanarPlcSupport.coverage must bind declared region identities; got different region IDs"
        )
    if (
        mesh.storage is not None
        and mesh.storage.evidence_id not in coverage.premise_certificate_ids
    ):
        raise ValueError(
            "PlanarPlcSupport.coverage must retain the collective partition evidence; got no storage evidence ID"
        )
    return cells, limits


def _edge_charts(
    cells: tuple[PlanarMap, ...],
    vertices: np.ndarray,
    parents: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[tuple[algebra.Polynomial, ...], ...]:
    charts: list[tuple[algebra.Polynomial, ...]] = []
    for parent in parents.tolist():
        cell = cells[parent]
        local = np.asarray(
            [cell.vertices.index(vertex) for vertex in vertices.tolist()], dtype=np.int64
        )
        topology = reference_cell_topology(cell.kind)
        if not any(set(local.tolist()) == set(edge) for edge in topology.entities[1]):
            raise ValueError(
                "PlanarPlcSupport.edge must bind a reference-map edge; got no incident reference edge"
            )
        corners = np.asarray(topology.vertices, dtype=np.float64)[local]
        arguments = algebra.affine_arguments(
            corners[0],
            (corners[1] - corners[0])[:, None],
            budget=budget,
        )
        charts.append(
            tuple(
                algebra.compose(value, arguments, budget=budget)
                for value in cell.coordinates
            )
        )
    if not charts:
        raise ValueError(
            "PlanarPlcSupport.edge must have coordinate-map incidence; got no chart"
        )
    return tuple(charts)


def _segments(
    transfer: PlanarPlcTransfer,
    region: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[planar_primitives_owner.Polygon, ...]:
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
    return tuple(
        planar_primitives_owner.rational_points(
            transfer.domain.vertices[row], budget=budget
        )
        for row in rows
    )


def _on_segment(
    point: planar_primitives_owner.Point,
    segment: planar_primitives_owner.Polygon,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    return planar_primitives_owner.turn(
        segment[0], segment[1], point, budget=budget
    ) == 0 and all(
        min(a, b) <= p <= max(a, b)
        for a, b, p in zip(segment[0], segment[1], point, strict=True)
    )


def _in_region(
    point: planar_primitives_owner.Point,
    segments: tuple[planar_primitives_owner.Polygon, ...],
    /,
    *,
    closed: bool,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    winding = 0
    for segment in segments:
        a, b = segment
        # A segment strictly above or below this rational ordinate can neither
        # contain the point nor cross its winding ray. Keep the full source
        # query charge; only the unnecessary determinant arithmetic is skipped.
        if (point[1] < a[1] and point[1] < b[1]) or (point[1] > a[1] and point[1] > b[1]):
            continue
        side = planar_primitives_owner.turn(a, b, point, budget=budget)
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


def _open_triangle_contact(
    triangle: planar_primitives_owner.Polygon,
    segment: planar_primitives_owner.Polygon,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    """rational strict half-plane feasibility along the whole source segment."""
    low, high = Fraction(0), Fraction(1)
    for a, b in zip(triangle, (*triangle[1:], triangle[0]), strict=True):
        first, last = (
            planar_primitives_owner.turn(a, b, segment[0], budget=budget),
            planar_primitives_owner.turn(a, b, segment[1], budget=budget),
        )
        slope = last - first
        if slope == 0:
            if first <= 0:
                return False
        elif slope > 0:
            low = max(low, -first / slope)
        else:
            high = min(high, -first / slope)
    return low < high


@checked
def planar_point_support(
    transfer: PlanarPlcTransfer,
    dimension: int,
    index: int,
    point: np.ndarray,
    work: list[int],
    /,
    *,
    region_segments: tuple[planar_primitives_owner.Polygon, ...] | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    p = tuple(Fraction(value) for value in point.tolist())
    if dimension == 0:
        work[0] = spend_support_work(
            work[0],
            1,
            "PlanarPlcSupport.work must respect the support query allowance; got resource exhaustion",
        )
        row = _position(transfer.vertex_indices, index)
        return (
            p
            == planar_primitives_owner.rational_points(
                transfer.source_vertices[row : row + 1], budget=budget
            )[0]
        )
    if dimension == 1:
        work[0] = spend_support_work(
            work[0],
            1,
            "PlanarPlcSupport.work must respect the support query allowance; got resource exhaustion",
        )
        edge = transfer.edge_vertices[_position(transfer.edge_indices, index)]
        return _on_segment(
            p,
            planar_primitives_owner.rational_points(
                transfer.source_vertices[edge], budget=budget
            ),
            budget=budget,
        )
    if dimension == 2:
        segments = (
            _segments(transfer, index, budget=budget)
            if region_segments is None
            else region_segments
        )
        work[0] = spend_support_work(
            work[0],
            len(segments),
            "PlanarPlcSupport.work must respect the support query allowance; got resource exhaustion",
        )
        return _in_region(p, segments, closed=False, budget=budget)
    raise ValueError(
        "PlanarPlcSupport.source_dimension must be 0, 1, or 2; got another dimension"
    )


@checked
def planar_cell_support(
    transfer: PlanarPlcTransfer,
    points: np.ndarray,
    region: int,
    work: list[int],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    """Whole-cell containment: positive triangle, degree one, no interior cut.

    The center is rational, not a rounded float ancestry. A cavity, disconnected
    component, or interface inside the cell is caught by the whole source
    segment/open-triangle intersection, even when every cell corner is inside.
    """
    triangle = planar_primitives_owner.rational_points(points, budget=budget)
    if len(triangle) != 3 or planar_primitives_owner.turn(*triangle, budget=budget) <= 0:
        return False
    segments = _segments(transfer, region, budget=budget)
    work[0] = spend_support_work(
        work[0],
        2 * len(segments),
        "PlanarPlcSupport.work must respect the support query allowance; got resource exhaustion",
    )
    center = tuple(sum((p[axis] for p in triangle), Fraction(0)) / 3 for axis in range(2))
    return _in_region(center, segments, closed=False, budget=budget) and not any(
        _open_triangle_contact(triangle, segment, budget=budget) for segment in segments
    )


def source_entity_roles(
    dimensions: np.ndarray, /
) -> tuple[association_state_owner.GeometrySourceEntityRole, ...]:
    """The planar source owner's explicit vertex/edge/region stratum schema."""
    schema = {
        0: association_state_owner.GeometrySourceEntityRole.VERTEX,
        1: association_state_owner.GeometrySourceEntityRole.EDGE,
        2: association_state_owner.GeometrySourceEntityRole.REGION,
    }
    if any(dimension not in schema for dimension in dimensions.tolist()):
        raise ValueError(
            "PlanarPlcSupport.source_dimension must be 0, 1, or 2; got another dimension"
        )
    return tuple(schema[dimension] for dimension in dimensions.tolist())


@checked
def planar_source_associations(
    transfer: PlanarPlcTransfer,
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[association_state_owner.GeometryAssociation, tuple[int, ...]]:
    if (
        transfer.domain.ambient_dimension != 2
        or source.coordinate_contract.spatial_id
        != transfer.coordinate_contract.spatial_id
    ):
        raise ValueError(
            "PlanarPlcSupport.source must retain its two-dimensional coordinate frame; got another source binding"
        )
    cell_maps(source.mesh, source.geometry, budget=budget)
    if source.mesh.storage is not None:
        # Owner-local epochs carry no copied serial certificate: the normalized
        # proof chain reaches the source source, and this rational target
        # retains its own collectively recertified embedding and coverage.
        collective = source.collective_evidence
        if collective is None or source.collective_certificates is None:
            raise ValueError(
                "PlanarPlcSupport.source must retain collective proof and target certificates; got missing evidence"
            )
        collective.require_passed()
        storage = source.mesh.storage
        if (
            collective.mesh_id != source.mesh.mesh_id
            or collective.evidence_id != storage.evidence_id
            or collective.coordinate_geometry_id
            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                source.geometry
            )
        ):
            raise ValueError(
                "PlanarPlcSupport.collective_evidence must bind the published target; got another publication"
            )
        if not isinstance(
            _dependency_phydrax_meshing__result.require_original_meshing_source(source),
            _dependency_phydrax_meshing__result.CellMeshingResult,
        ):
            raise ValueError(
                "PlanarPlcSupport.source must have a serial certified planar predecessor; got another source kind"
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
                "PlanarPlcSupport.target must retain certified embedding and coverage; got missing or failed certificates"
            )
    else:
        certificate = source.certification
        if (
            certificate is None
            or not certificate.passed
            or certificate.mesh_id != source.mesh.mesh_id
        ):
            raise ValueError(
                "PlanarPlcSupport.source must retain passed current certification; got missing or failed certification"
            )
        embedding = certificate.embedding
        if embedding is None:
            raise ValueError(
                "PlanarPlcSupport.source must retain bound certified global embedding; got failed or different evidence"
            )
        embedding.binding.require(source.mesh, source.geometry)
        if certificate.coverage is not None:
            certificate.coverage.binding.require(source.mesh, source.geometry)
    dimensions: list[int] = []
    vertex: association_state_owner.GeometryAssociation | None = None
    namespaces = {
        0: transfer.vertex_indices,
        1: transfer.edge_indices,
        2: transfer.region_indices,
    }
    for association in source.associations:
        if (
            association.association_kind
            is not association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR
            or association.source_id != transfer.domain.source_id
            or association.source_revision != transfer.source_revision
        ):
            raise ValueError(
                "PlanarPlcSupport.source_revision must match the represented revision; got a different revision"
            )
        dimension = association_state_owner.target_dimension(source.mesh, association)
        if dimension in dimensions:
            raise ValueError(
                "PlanarPlcSupport.associations must have one row per mesh dimension; got repeated dimensions"
            )
        association.validate_target(source.mesh.entity_set(dimension))
        if not association.complete:
            raise ValueError(
                "PlanarPlcSupport.association must be resolved and unambiguous; got unresolved or ambiguous rows"
            )
        dims = np.asarray(association.source_dimensions, dtype=np.int64)
        indices = np.asarray(association.source_indices, dtype=np.int64)
        for kind in np.unique(dims).tolist():
            if kind not in namespaces or not np.all(
                np.isin(indices[dims == kind], namespaces[kind])
            ):
                raise ValueError(
                    "PlanarPlcSupport.association must index declared source rows; got undeclared source IDs"
                )
        if association.source_entities.roles() != source_entity_roles(dims):
            raise ValueError(
                "PlanarPlcSupport.association must align roles with vertex, edge, or region topology; got different roles"
            )
        if dimension == 2:
            association.target_rows(
                np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)
            )
            if np.any(dims != 2):
                raise ValueError(
                    "PlanarPlcSupport.cells must name declared source regions; got nonregion cell rows"
                )
        if dimension == 0:
            vertex = association
            association.target_rows(
                np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
            )
        dimensions.append(dimension)
    if vertex is None or 1 not in dimensions or 2 not in dimensions:
        raise ValueError(
            "PlanarPlcSupport.associations must cover vertices and cells with source constraints; got missing rows"
        )
    return vertex, tuple(sorted(d for d in dimensions if d != 0))


def _vertex_closure(
    transfer: PlanarPlcTransfer,
    mesh: CellMesh,
    levels: list[PlcEntityClasses],
    eligible: np.ndarray,
    /,
) -> None:
    links = association_routes.incidence_pairs(mesh, 0, 1)
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
            raise association_routes.AssociationPropagationError(
                "PlanarPlcSupport.vertex must identify one shared source corner; got conflicting incident edges",
                np.asarray(mesh.vertex_global_ids)[row : row + 1],
            )
        vertices.dimensions[row], vertices.indices[row] = (
            0,
            transfer.vertex_indices[next(iter(common))],
        )


def _prove_planar_edge(
    transfer: PlanarPlcTransfer,
    index: int,
    cells: tuple[PlanarMap, ...],
    vertices: np.ndarray,
    parents: np.ndarray,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    work: list[int],
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[bool, int]:
    """Require every incident chart to realize the same source-edge orientation."""
    edge = transfer.edge_vertices[_position(transfer.edge_indices, index)]
    charts = _edge_charts(cells, vertices, parents, budget=budget)
    outcomes = tuple(
        _dependency_phydrax_meshing__plc_mapped_support.edge_support(
            chart,
            transfer.source_vertices[edge],
            limits,
            work,
            budget=budget,
        )
        for chart in charts
    )
    realized = {orientation for supported_, orientation in outcomes if supported_}
    supported = all(supported_ for supported_, _ in outcomes) and len(realized) == 1
    return supported, next(iter(realized)) if supported else 0


# Entity traversal keeps judged-star, endpoint, then whole-edge verdicts and support-query charges in order.
@checked
def prove_levels(
    transfer: PlanarPlcTransfer,
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    levels: tuple[PlcEntityClasses, ...] | list[PlcEntityClasses],
    work: list[int],
    /,
    *,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None = None,
    coverage: domain_coverage_owner.DomainCoverageCertificate | None = None,
    judged: tuple[np.ndarray, ...] | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, ...]:
    """rational support of every class; ``judged`` limits owner-local proofs.

    An owner-local entity outside ``judged`` has a truncated resident star; its
    edge incidence and charts are incomplete, so it receives no local verdict.
    """
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    signs: list[np.ndarray] = []
    edge_links = association_routes.incidence_pairs(mesh, 1, 2)
    edge_cells = np.bincount(edge_links[:, 0], minlength=mesh.entity_set(1).count)
    edge_regions = association_routes.stratum_region_consensus(
        mesh, 1, levels[2].indices, 2
    )
    cells, limits = _prepare_planar_support(
        transfer,
        mesh,
        geometry,
        levels[2].indices,
        embedding=embedding,
        coverage=coverage,
        budget=budget,
    )
    if geometry.exact_source is not None:
        coordinates = np.asarray(
            planar_primitives_owner.rational_points(coordinates, budget=budget),
            dtype=object,
        )
        for cell in cells:
            corners = planar_primitives_owner.rational_points(
                np.asarray(reference_cell_topology(cell.kind).vertices, dtype=np.float64),
                budget=budget,
            )
            for vertex, corner in zip(cell.vertices, corners, strict=True):
                coordinates[vertex] = tuple(
                    algebra.evaluate(value, corner, budget=budget)
                    for value in cell.coordinates
                )
    # rational authored endpoints are immutable within this proof. Retain one
    # bank per used region, not one reconstruction per mesh vertex; every
    # predicate visit still consumes the source support-query allowance.
    region_segments: dict[int, tuple[planar_primitives_owner.Polygon, ...]] = {}
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
                    region_segments[index] = _segments(transfer, index, budget=budget)
                supported = all(
                    planar_point_support(
                        transfer,
                        kind,
                        index,
                        point,
                        work,
                        region_segments=region_segments.get(index) if kind == 2 else None,
                        budget=budget,
                    )
                    for point in points
                )
                if dimension == 1 and kind == 1 and supported:
                    supported, orientations[row] = _prove_planar_edge(
                        transfer,
                        index,
                        cells,
                        vertices,
                        edge_links[edge_links[:, 0] == row, 1],
                        limits,
                        work,
                        budget=budget,
                    )
            if not supported:
                raise association_routes.AssociationPropagationError(
                    "PlanarPlcSupport.coordinates must retain inherited source support; got unsupported coordinates",
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
                "PlanarPlcSupport.strata must have unambiguous source incidence; got incompatible incidence"
            )
        for row, (kind, index) in enumerate(
            zip(level.dimensions.tolist(), level.indices.tolist(), strict=True)
        ):
            level.codes[row] = codes[kind][index]
        for value in level:
            value.setflags(write=False)
    return tuple(levels)


@checked
def planar_classes(
    transfer: PlanarPlcTransfer,
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[PlcEntityClasses, ...]:
    with budget.activate():
        planar_source_associations(transfer, source, budget=budget)
        cell = association_routes.association_for_dimension(source, 2)
        regions = np.asarray(cell.source_indices, dtype=np.int64)[
            cell.target_rows(
                np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)
            )
        ]
        levels: list[PlcEntityClasses] = []
        for dimension in range(3):
            indices = association_routes.stratum_region_consensus(
                source.mesh, dimension, regions, 2
            )
            dims = np.where(indices >= 0, 2, -1).astype(np.int64)
            for association in source.associations:
                if (
                    association_state_owner.target_dimension(source.mesh, association)
                    == dimension
                ):
                    rows = association_state_owner.entity_rows(
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
        # planar_source_associations admitted exactly one of these evidence owners.
        if source.mesh.storage is not None and source.collective_certificates is not None:
            embedding, coverage = source.collective_certificates
        elif source.certification is not None:
            embedding, coverage = (
                source.certification.embedding,
                source.certification.coverage,
            )
        else:
            raise ValueError(
                "PlanarPlcSupport.source must retain current certification; got no bound evidence"
            )
        orientations = prove_levels(
            transfer,
            source.mesh,
            source.geometry,
            levels,
            [transfer.maximum_support_queries],
            embedding=embedding,
            coverage=coverage,
            budget=budget,
        )
        for association in source.associations:
            if association_state_owner.target_dimension(source.mesh, association) == 1:
                rows = association_state_owner.entity_rows(
                    source.mesh,
                    1,
                    np.asarray(association.target_global_ids, dtype=np.int64),
                )
                declared = np.asarray(association.orientations, dtype=np.int8)
                if np.any((declared != 0) & (declared != orientations[1][rows])):
                    raise ValueError(
                        "PlanarPlcSupport.edge_orientation must agree with source support; got opposite orientation"
                    )
        return _finish_classes(transfer, levels)


@checked
def planar_protected_edges(
    transfer: PlanarPlcTransfer,
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    midpoint_required: bool = True,
    _classes: tuple[PlcEntityClasses, ...] | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    levels = (
        planar_classes(transfer, source, budget=budget) if _classes is None else _classes
    )
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
                transfer,
                1,
                int(levels[1].indices[row]),
                midpoints[row],
                work,
                budget=budget,
            )
    return protected


def _complete_stars(mesh: CellMesh, /) -> tuple[np.ndarray, ...]:
    """Owner-local entities whose every incident cell is resident.

    A resident cell is whole when each of its facets is matched or physical
    boundary. An entity whose resident incident cells are all whole cannot
    reach a missing cell across a facet, so its resident star is its whole
    star. A star truncated at the artificial halo boundary yields no verdict.
    """
    storage = mesh.storage
    if storage is None or storage.local_neighborhood_complete is None:
        raise ValueError(
            "PlanarPlcSupport.storage must retain resident-star completeness evidence; got missing evidence"
        )
    complete = np.asarray(storage.local_neighborhood_complete, dtype=np.bool_)
    rows = association_state_owner.entity_rows(
        mesh, 2, np.asarray(storage.entity_global_ids[2], dtype=np.int64)
    )
    count = mesh.entity_set(2).count
    if (
        complete.shape != rows.shape
        or rows.size != count
        or np.any(rows < 0)
        or np.unique(rows).size != count
    ):
        raise ValueError(
            "PlanarPlcSupport.neighborhood must label every resident cell once; got invalid resident IDs"
        )
    cells = np.zeros((count,), dtype=np.bool_)
    cells[rows] = complete
    stars: list[np.ndarray] = []
    for dimension in (0, 1):
        links = association_routes.incidence_pairs(mesh, dimension, 2)
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
    """Collective: every global entity has one agreed whole-star verdict.

    Every process enters both gathers, including one whose local proof failed,
    so no owner waits on a peer that already refused. Owners judging the same
    global entity must agree exactly on its stratum and orientation; an entity
    judged by no owner is refused, never inferred from a truncated star. Each
    resident row adopts the verdict and parent of its lowest judging process.
    """

    storage = target.storage
    if storage is None:
        raise ValueError(
            "PlanarPlcSupport.verdicts must bind mesh storage; got no storage"
        )
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
        multihost_utils.process_allgather(sizes, tiled=False),
        dtype=np.int64,
    ).reshape((-1, 4))
    if np.any(gathered_sizes[:, 3]):
        if failure is not None:
            raise failure
        raise ValueError(
            "PlanarPlcSupport.verdicts must be proved by every responsible process; got a process refusal"
        )
    if local is None:
        raise ValueError(
            "PlanarPlcSupport.verdicts must retain local proof results; got no local proof"
        )
    packed = np.zeros(
        (int(np.sum(np.max(gathered_sizes[:, :3], axis=0))), _VERDICT_COLUMNS),
        dtype=np.int64,
    )
    flat = np.concatenate(packets, axis=0)
    packed[: flat.shape[0]] = flat
    table = np.asarray(
        multihost_utils.process_allgather(packed, tiled=False),
        dtype=np.int64,
    ).reshape((-1, _VERDICT_COLUMNS))
    table = table[table[:, 0] == 1]
    levels, parents, orientations, _ = local
    adopted = []
    for dimension in range(3):
        rows = table[table[:, 1] == dimension]
        rows = rows[np.argsort(rows[:, 2], kind="stable")]
        unique, first = np.unique(rows[:, 2], return_index=True)
        if unique.size != storage.global_entity_counts[dimension]:
            raise association_routes.AssociationPropagationError(
                "PlanarPlcSupport.entity must have a process holding its whole incident star; got no qualifying process",
                np.setdiff1d(ids[dimension], unique),
            )
        group = np.searchsorted(unique, rows[:, 2])
        conflicting = np.any(rows[:, 3:6] != rows[first[group], 3:6], axis=1)
        if np.any(conflicting):
            raise association_routes.AssociationPropagationError(
                "PlanarPlcSupport.verdicts must agree across processes holding the incident star; got conflicting verdicts",
                np.unique(rows[conflicting, 2]),
            )
        positions = np.searchsorted(unique, ids[dimension])
        if np.any(positions >= unique.size) or np.any(
            unique[np.minimum(positions, unique.size - 1)] != ids[dimension]
        ):
            raise association_routes.AssociationPropagationError(
                "PlanarPlcSupport.entity must occur in agreed global verdicts; got an absent resident entity",
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
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    lineage: _dependency_phydrax_meshing__lineage.MeshLineage,
    target: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    judged: tuple[np.ndarray, ...] | None,
    /,
    *,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None,
    coverage: domain_coverage_owner.DomainCoverageCertificate | None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    list[PlcEntityClasses], list[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, ...]
]:
    """Lineage, consensus and closure classes with their rational support proof.

    ``judged`` restricts locally inferred classes and proofs to owner-local
    whole stars; ``None`` decides every entity of a serial mesh.
    """
    cell_maps(target, geometry, budget=budget)
    old = planar_classes(transfer, source, budget=budget)
    levels, parents = _lineage_levels(source.mesh, target, lineage, old)
    cell_ids = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)
    if np.any(levels[2].dimensions != 2):
        raise association_routes.AssociationPropagationError(
            "PlanarPlcSupport.cells must have unambiguous source-region lineage; got incompatible region ancestry",
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
                dims, rows = association_routes.child_sources(
                    lineage, source.mesh, target, ids[order]
                )
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
        regions = association_routes.stratum_region_consensus(
            target, dimension, levels[2].indices, 2
        )
        missing = (level.dimensions < 0) & (regions >= 0) & decided
        links = association_routes.incidence_pairs(target, dimension, 2)
        for row in np.flatnonzero(missing).tolist():
            cells = links[links[:, 0] == row, 1]
            if cells.size == 0:
                raise ValueError(
                    "PlanarPlcSupport.created_stratum must have source cell lineage; got no matching cell parent"
                )
            cell = int(cells[0])
            parents[dimension][0][row] = 2
            parents[dimension][1][row] = parents[2][1][cell]
        if dimension == 0:
            _vertex_closure(transfer, target, levels, (level.dimensions < 0) & decided)
        missing = (level.dimensions < 0) & (regions >= 0) & decided
        level.dimensions[missing], level.indices[missing] = 2, regions[missing]
    orientations = prove_levels(
        transfer,
        target,
        geometry,
        levels,
        [transfer.maximum_support_queries],
        embedding=embedding,
        coverage=coverage,
        judged=judged,
        budget=budget,
    )
    return levels, parents, orientations


def _lineage_levels(
    source: CellMesh,
    target: CellMesh,
    lineage: _dependency_phydrax_meshing__lineage.MeshLineage,
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
                "PlanarPlcSupport.lineage must bind source and target entity sets; got different set identities"
            )
        if np.any(
            np.asarray(record.relation_kinds)
            == _dependency_phydrax_meshing__lineage.EntityLineageKind.UNKNOWN
        ):
            raise association_routes.AssociationPropagationError(
                "PlanarPlcSupport.lineage must identify source strata; got an unknown relation",
                ids,
            )
        source_ids = np.asarray(record.source_global_ids, dtype=np.int64)
        source_rows = association_state_owner.entity_rows(source, dimension, source_ids)
        target_rows = association_state_owner.entity_rows(
            target, dimension, np.asarray(record.target_global_ids, dtype=np.int64)
        )
        if np.any(source_rows < 0) or np.any(target_rows < 0):
            raise ValueError(
                "PlanarPlcSupport.lineage must index declared mesh entities; got undeclared entity IDs"
            )
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


@checked
def preserved_parameters(
    source: CellMesh,
    target: CellMesh,
    dimension: int,
    previous: association_state_owner.GeometryAssociation,
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
    source_rows = association_state_owner.entity_rows(source, dimension, ids)
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


@checked
def planar_propagate(
    transfer: PlanarPlcTransfer,
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    lineage: _dependency_phydrax_meshing__lineage.MeshLineage,
    target: CellMesh,
    /,
    *,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None = None,
    coverage: domain_coverage_owner.DomainCoverageCertificate | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[association_state_owner.GeometryAssociation, ...]:
    with budget.activate():
        # Source checks bind the shared serial source; target proofs stay collective.
        vertex, dimensions = planar_source_associations(transfer, source, budget=budget)
        if target.storage is None:
            association_routes.require_transfer_lineage(
                source.mesh,
                lineage,
                target,
                message="PlanarPlcSupport.lineage must bind this source and target; got different mesh identities",
            )
            levels, parents, orientations = _propagated_levels(
                transfer,
                source,
                lineage,
                target,
                geometry,
                None,
                embedding=embedding,
                coverage=coverage,
                budget=budget,
            )
        else:
            # Only whole resident stars contribute an owner-local verdict.
            local: _LocalVerdicts | None = None
            failure: ValueError | None = None
            try:
                association_routes.require_transfer_lineage(
                    source.mesh,
                    lineage,
                    target,
                    message="PlanarPlcSupport.lineage must bind this source and target; got different mesh identities",
                )
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
                    budget=budget,
                )
                local = (levels, parents, orientations, judged)
            except ValueError as error:
                failure = error
            levels, parents, orientations = _agreed_owner_verdicts(target, local, failure)
    final = _finish_classes(transfer, levels)
    output: list[association_state_owner.GeometryAssociation] = []
    for dimension in (0, *dimensions):
        previous = (
            vertex
            if dimension == 0
            else association_routes.association_for_dimension(source, dimension)
        )
        level = final[dimension]
        ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
        roles = source_entity_roles(level.dimensions)
        names = tuple(
            f"{transfer.source_revision}:{role.value}:{index}"
            for role, index in zip(roles, level.indices.tolist(), strict=True)
        )
        output.append(
            association_state_owner.GeometryAssociation(
                association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
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
                parameters=preserved_parameters(
                    source.mesh, target, dimension, previous, level
                ),
                orientations=orientations[dimension],
                parent_dimensions=parents[dimension][0],
                parent_ids=parents[dimension][1],
                parent_association_id=previous.association_id,
                provenance=association_state_owner.GeometryAssociationProvenance.LINEAGE,
            )
        )
    return tuple(output)
