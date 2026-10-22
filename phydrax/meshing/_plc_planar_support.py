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
Native initial planar publications use their authenticated whole-domain theorem
as the original premise. Adapted publications retain target-bound certificates.
Truncated source closures use theorem-bound global edge incidence routed by
stable IDs; they never infer a whole star from a one-cell resident edge.
Exact corner images share their request-retained map bank. Construction scratch
and per-entity predicate preparations end at their actual last consumer; retained
proofs, support-query charges, and source/trace predicates remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import NoReturn, Protocol
from weakref import WeakKeyDictionary

import jax
import jax.numpy as jnp
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
from phydrax.discretization._coordinate_identity import (
    coordinate_scope_key,
    coordinate_source_bank_key,
)
from phydrax.discretization._coordinate_operands import seal_coordinate_operand_array
from phydrax.discretization._distributed_field import (
    resolve_owner_local_field_values,
    resolve_owner_local_record_values,
)

from .._execution_runtime import process_owner_devices
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from ..discretization import _coordinate_enclosure as algebra, CellMesh
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._reference_cell_types import CellGeometryElement
from ..typing import checked
from ._association_routes import PlcEntityClasses
from ._collective_failure import agreed_local_step
from ._scope import local_logical_lookup
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
    transfer_id: str


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


@dataclass(frozen=True, slots=True, eq=False)
class PlanarMaps:
    """Request-retained exact coordinate maps of one planar mesh and geometry.

    Rows follow the mesh's cell entity order. ``images`` binds every mesh
    vertex to its one exact corner image. A proven affine triangle map is its
    corner interpolant and keeps no polynomial graph; every other map keeps its
    full coordinate polynomials in ``polynomials``.
    """

    kinds: tuple[str, ...]
    vertices: np.ndarray
    images: tuple[tuple[algebra.ExactScalar, ...] | None, ...]
    polynomials: dict[int, tuple[algebra.Polynomial, ...]]

    def __len__(self) -> int:
        return len(self.kinds)

    def cell_vertices(self, row: int, /) -> tuple[int, ...]:
        vertices = self.vertices[row]
        return tuple(vertices[vertices >= 0].tolist())

    def corners(self, row: int, /) -> tuple[tuple[algebra.ExactScalar, ...], ...]:
        corners = []
        for vertex in self.cell_vertices(row):
            image = self.images[vertex]
            if image is None:
                raise ValueError(
                    "PlanarPlcSupport.images must bind every cell corner; got an unbound vertex"
                )
            corners.append(image)
        return tuple(corners)


@checked
def cell_maps(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PlanarMaps:
    # The returned map bank is retained below; carrier rationalization and
    # coordinate-algebra construction intermediates die with this preparation.
    with budget.activate(), budget.temporary_scope():
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
        # Coordinate maps own the geometry. A carrier-defined source's float
        # carrier must be each exact corner image correctly rounded; adapted
        # barycentric midpoints are rational, not necessarily binary64 points.
        carrier = (
            np.asarray(mesh.coordinates, dtype=np.float64).view(np.uint64)
            if geometry.exact_source is None
            else None
        )
        blocks: list[tuple[CellGeometryElement, np.ndarray, np.ndarray]] = []
        affine: list[bool] = []
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
            blocks.append(
                (
                    element,
                    np.asarray(route, dtype=np.int64),
                    np.asarray(block.vertices, dtype=np.int64),
                )
            )
            # An affine simplex source map is exactly its corner interpolant.
            affine.append(
                block.cell_kind == "triangle"
                and algebra.affine_simplex_source(element, budget=budget)
            )
            block_ids.append(np.asarray(block.global_ids, dtype=np.int64))
        images = algebra.prepared_coordinate_vertex_images(
            mesh,
            geometry,
            mismatch="PlanarPlcSupport.coordinates must agree on shared corner images; got unequal rational points",
            budget=budget,
        )
        if images is None:
            raise ValueError(
                "PlanarPlcSupport.coordinates must resolve source expressions; got unresolved expressions"
            )
        bound = [
            (vertex, image) for vertex, image in enumerate(images) if image is not None
        ]
        budget.reserve(sum(len(image) for _, image in bound))
        if carrier is not None and bound:
            rounded = np.asarray(
                [algebra.rounded_point(image) for _, image in bound],
                dtype=np.float64,
            ).view(np.uint64)
            if not np.array_equal(rounded, carrier[[vertex for vertex, _ in bound]]):
                raise ValueError(
                    "PlanarPlcSupport.carrier must equal correctly rounded coordinate-map images; got different corner bits"
                )
        order = association_state_owner.entity_rows(mesh, 2, np.concatenate(block_ids))
        width = max(vertices.shape[1] for _, _, vertices in blocks)
        budget.reserve(0, 256 + 8 * order.size * width)
        vertices = np.full((order.size, width), -1, dtype=np.int64)
        kinds: list[str] = [""] * order.size
        polynomials: dict[int, tuple[algebra.Polynomial, ...]] = {}
        cursor = 0
        for block, (element, route, block_vertices), proven in zip(
            mesh.blocks, blocks, affine, strict=True
        ):
            rows = order[cursor : cursor + block_vertices.shape[0]]
            cursor += block_vertices.shape[0]
            vertices[rows, : block_vertices.shape[1]] = block_vertices
            for row in rows.tolist():
                kinds[row] = block.cell_kind
            if proven:
                continue
            for row, route_row in zip(rows.tolist(), route, strict=True):
                polynomial = algebra.coordinate_polynomials(
                    element,
                    tuple(controls[int(index)] for index in route_row.tolist()),
                    budget=budget,
                )
                if polynomial is None:
                    raise ValueError(
                        "PlanarPlcSupport.coordinates must resolve source expressions; got unresolved expressions"
                    )
                polynomials[row] = polynomial
        vertices = seal_coordinate_operand_array(vertices, budget=budget)
        maps = PlanarMaps(tuple(kinds), vertices, images, polynomials)
        budget.retain_basis((key, maps, maps.kinds, images, polynomials))
        budget.retain_owner(vertices)
        budget.planar_cell_map_cache[key] = maps
        return maps


def _require_initial_planar_domain(
    transfer: PlanarPlcTransfer,
    evidence: _dependency_phydrax_meshing__result.AbstractCollectiveMeshTheorem,
    /,
) -> None:
    from ._initial_certification import (
        InitialCollectiveMeshEvidence,
        InitialPlanarPremise,
    )

    if not isinstance(evidence, InitialCollectiveMeshEvidence):
        raise ValueError(
            "PlanarPlcSupport.initial_premise must be an initial collective theorem; got another evidence kind"
        )

    evidence.require_current()
    premise = evidence.premise
    if not isinstance(premise, InitialPlanarPremise):
        raise ValueError(
            "PlanarPlcSupport.original must retain an initial planar premise; got another source kind"
        )
    domain = premise.require_current_domain()
    if (
        premise.source_id != transfer.domain.source_id
        or premise.source_revision != transfer.source_revision
        or domain.domain_id != transfer.domain.domain_id
    ):
        raise ValueError(
            "PlanarPlcSupport.initial_premise must bind the represented source, revision and domain; got another source binding"
        )


def _prepare_planar_support(
    transfer: PlanarPlcTransfer,
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    regions: np.ndarray,
    /,
    *,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None = None,
    coverage: domain_coverage_owner.DomainCoverageCertificate | None = None,
    initial_evidence: _dependency_phydrax_meshing__result.AbstractCollectiveMeshTheorem
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    PlanarMaps,
    _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
]:
    """Prove represented image coverage, including bilinear and restricted Q1 maps."""
    cells = cell_maps(mesh, geometry, budget=budget)
    limits = _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits(
        maximum_candidate_pairs=transfer.maximum_support_queries,
        maximum_ray_tests=transfer.maximum_support_queries,
        maximum_subdivision_pieces=transfer.maximum_support_queries,
    )

    def labels() -> np.ndarray:
        # Owner-local lineage-incomplete cells carry no region; consumers below
        # are the initial premise and serial coverage, which declare every cell.
        ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        rows = association_state_owner.entity_rows(mesh, 2, ids)
        return np.asarray(
            [
                _position(transfer.region_indices, label)
                for label in regions[rows].tolist()
            ],
            dtype=np.int64,
        )

    if initial_evidence is not None:
        _require_initial_planar_domain(transfer, initial_evidence)
        storage = mesh.storage
        if (
            storage is None
            or embedding is not None
            or coverage is not None
            or storage.evidence_id != initial_evidence.evidence_id
            or mesh.mesh_id != initial_evidence.mesh_id
            or mesh.topology_id != initial_evidence.topology_id
            or mesh.geometry_id != initial_evidence.geometry_id
            or _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                geometry
            )
            != initial_evidence.coordinate_geometry_id
            or np.any(labels() != 0)
        ):
            raise ValueError(
                "PlanarPlcSupport.initial_premise must bind the original collective mesh and coordinate maps; got another publication"
            )
        # The authenticated initial theorem proves this native tile complex's
        # embedding and whole-region coverage. It is not a target certificate:
        # only its original publication may consume this premise directly.
        # prove_levels still proves every declared vertex/edge support below.
        return cells, limits
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
            labels(),
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


def _edge_locals(
    cells: PlanarMaps,
    vertices: np.ndarray,
    parent: int,
    /,
) -> np.ndarray:
    """Local corners of one mesh edge in an incident cell's reference edge."""
    cell_vertices = cells.cell_vertices(parent)
    if any(vertex not in cell_vertices for vertex in vertices.tolist()):
        raise ValueError(
            "PlanarPlcSupport.edge must bind a reference-map edge; got no incident reference edge"
        )
    local = np.asarray(
        [cell_vertices.index(vertex) for vertex in vertices.tolist()], dtype=np.int64
    )
    topology = reference_cell_topology(cells.kinds[parent])
    if not any(set(local.tolist()) == set(edge) for edge in topology.entities[1]):
        raise ValueError(
            "PlanarPlcSupport.edge must bind a reference-map edge; got no incident reference edge"
        )
    return local


def _edge_chart(
    cells: PlanarMaps,
    vertices: np.ndarray,
    parent: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[algebra.Polynomial, ...]:
    """Restrict a retained non-affine cell polynomial map to one mesh edge."""
    local = _edge_locals(cells, vertices, parent)
    corners = np.asarray(
        reference_cell_topology(cells.kinds[parent]).vertices, dtype=np.float64
    )[local]
    arguments = algebra.affine_arguments(
        corners[0],
        (corners[1] - corners[0])[:, None],
        budget=budget,
    )
    return tuple(
        algebra.compose(value, arguments, budget=budget)
        for value in cells.polynomials[parent]
    )


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
        # A straddling segment meets this ordinate at an abscissa between its
        # endpoint abscissae. A point strictly right of both cannot lie on it
        # or have it cross its rightward ray; a point strictly left of both has
        # the side sign of the segment's ordinate direction. The determinant
        # then decides nothing, so the winding verdict is unchanged.
        if point[0] > a[0] and point[0] > b[0]:
            continue
        if point[0] < a[0] and point[0] < b[0]:
            if a[1] <= point[1] < b[1]:
                winding += 1
            elif b[1] <= point[1] < a[1]:
                winding -= 1
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
        # Normalize every collective epoch to its original scientific premise.
        # Initial native roots own their proof; adapted targets must retain
        # their independently bound collective embedding and coverage.
        from ._initial_certification import InitialCollectiveMeshEvidence as initial_type

        collective = source.collective_evidence
        if collective is None:
            raise ValueError(
                "PlanarPlcSupport.source must retain collective proof; got missing evidence"
            )
        # Concrete dispatch avoids constructing Eqx BoundMethods over retained
        # source DAGs; every scientific reauthentication still runs below.
        type(collective).require_passed(collective)
        storage = source.mesh.storage
        if (
            collective.mesh_id != source.mesh.mesh_id
            or collective.topology_id != source.mesh.topology_id
            or collective.geometry_id != source.mesh.geometry_id
            or collective.partition_count != storage.partition_count
            or collective.global_entity_counts != storage.global_entity_counts
            or collective.evidence_id != storage.evidence_id
            or collective.coordinate_geometry_id
            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                source.geometry
            )
        ):
            raise ValueError(
                "PlanarPlcSupport.collective_evidence must bind the published target; got another publication"
            )
        original = _dependency_phydrax_meshing__result.require_original_meshing_source(
            source
        )
        if isinstance(original, initial_type):
            _require_initial_planar_domain(transfer, original)
        if not isinstance(collective, initial_type):
            if source.collective_certificates is None:
                raise ValueError(
                    "PlanarPlcSupport.source must retain collective proof and target certificates; got missing evidence"
                )
            embedding, coverage = source.collective_certificates
            if not isinstance(
                coverage, domain_coverage_owner.DomainCoverageCertificate
            ):
                raise ValueError(
                    "PlanarPlcSupport.coverage must certify the exact represented PLC domain; got approximate source coverage"
                )
            type(embedding.binding).require(
                embedding.binding, source.mesh, source.geometry
            )
            type(coverage.binding).require(
                coverage.binding, source.mesh, source.geometry
            )
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
        type(embedding.binding).require(
            embedding.binding, source.mesh, source.geometry
        )
        if certificate.coverage is not None:
            if not isinstance(
                certificate.coverage, domain_coverage_owner.DomainCoverageCertificate
            ):
                raise ValueError(
                    "PlanarPlcSupport.coverage must certify the exact represented PLC domain; got approximate source coverage"
                )
            type(certificate.coverage.binding).require(
                certificate.coverage.binding, source.mesh, source.geometry
            )
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
        type(association).validate_target(
            association, type(source.mesh).entity_set(source.mesh, dimension)
        )
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
        if type(association.source_entities).roles(
            association.source_entities
        ) != source_entity_roles(dims):
            raise ValueError(
                "PlanarPlcSupport.association must align roles with vertex, edge, or region topology; got different roles"
            )
        if dimension == 2:
            type(association).target_rows(
                association,
                np.asarray(
                    type(source.mesh).entity_set(source.mesh, 2).entity_ids,
                    dtype=np.int64,
                ),
            )
            if np.any(dims != 2):
                raise ValueError(
                    "PlanarPlcSupport.cells must name declared source regions; got nonregion cell rows"
                )
        if dimension == 0:
            vertex = association
            type(association).target_rows(
                association, np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
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
    cells: PlanarMaps,
    vertices: np.ndarray,
    parents: np.ndarray,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    work: list[int],
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[bool, int]:
    """Require every incident chart to realize the same source-edge orientation.

    A proven affine triangle restricts to the segment between its exact corner
    images, so the rational segment predicate decides exactly what the affine
    edge chart's Bernstein support proof decides, consuming the same three
    single-piece support queries. Other maps keep their polynomial edge charts.
    """
    if not parents.size:
        raise ValueError(
            "PlanarPlcSupport.edge must have coordinate-map incidence; got no chart"
        )
    edge = transfer.edge_vertices[_position(transfer.edge_indices, index)]
    endpoints: (
        tuple[tuple[algebra.ExactScalar, ...], tuple[algebra.ExactScalar, ...]] | None
    ) = None
    outcomes: list[tuple[bool, int]] = []
    for parent in parents.tolist():
        if parent in cells.polynomials:
            outcomes.append(
                _dependency_phydrax_meshing__plc_mapped_support.edge_support(
                    _edge_chart(cells, vertices, parent, budget=budget),
                    transfer.source_vertices[edge],
                    limits,
                    work,
                    budget=budget,
                )
            )
            continue
        _edge_locals(cells, vertices, parent)
        if endpoints is None:
            first, second = planar_primitives_owner.rational_points(
                transfer.source_vertices[edge], budget=budget
            )
            endpoints = first, second
        work[0] = spend_support_work(
            work[0],
            3,
            "MappedPlcSupport.work must respect the support query allowance; got resource exhaustion",
        )
        start, end = cells.images[int(vertices[0])], cells.images[int(vertices[1])]
        if start is None or end is None:
            raise ValueError(
                "PlanarPlcSupport.images must bind every cell corner; got an unbound vertex"
            )
        outcomes.append(
            _dependency_phydrax_meshing__plc_mapped_support._exact_segment_support(
                (start, end), endpoints, budget=budget
            )
        )
    realized = {orientation for supported_, orientation in outcomes if supported_}
    supported = all(supported_ for supported_, _ in outcomes) and len(realized) == 1
    return supported, next(iter(realized)) if supported else 0


def _source_edge_incidence(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
) -> np.ndarray:
    """Route authenticated global incidence to the resident stable edge IDs.

    The retained publication supplies partitioned ID queries, not verdicts.
    Counts and corner keys come from the accepted sharded theorem banks; only
    this owner's requested rows become host arrays.
    """
    from ..geometry._collective_domain_coverage import _facet_incidence, _lookup
    from ._initial_certification import InitialCollectiveMeshEvidence
    from ._organization import retained_mesh_organization_projections
    from ._publication_lowering import addressable

    mesh, evidence = source.mesh, source.collective_evidence
    storage = mesh.storage
    if storage is None or evidence is None:
        raise ValueError(
            "PlanarPlcSupport.incidence must retain collective source storage; got missing evidence"
        )
    if storage.partition_count == 1 and all(
        mesh.entity_set(degree).count == evidence.global_entity_counts[degree]
        for degree in range(3)
    ):
        links = association_routes.incidence_pairs(mesh, 1, 2)
        return np.bincount(links[:, 0], minlength=mesh.entity_set(1).count)
    scopes, _ = retained_mesh_organization_projections(source)
    if not scopes:
        raise ValueError(
            "PlanarPlcSupport.incidence must retain its owner-local publication projection; got missing source projection"
        )
    publication = scopes[0][1].publication
    publication.require_source(evidence.logical_arrays)
    if (
        publication.topology_id != mesh.topology_id
        or publication.coordinate_geometry_id != evidence.coordinate_geometry_id
        or publication.partition_count != storage.partition_count
    ):
        raise ValueError(
            "PlanarPlcSupport.incidence must bind this collective publication; got another source projection"
        )
    arrays = dict(evidence.logical_arrays)
    count = evidence.global_entity_counts[1]
    identifiers = arrays["entity_global_ids_1"][:count]
    keys = arrays["entity_vertices_1"][:count]
    if isinstance(evidence, InitialCollectiveMeshEvidence):
        counts = jnp.where(arrays["initial/edge_physical"][:count], 1, 2)
    else:
        _, counts = _facet_incidence(mesh, arrays, evidence.global_entity_counts[2])
    projected = dict(publication.projected_arrays)
    queries, valid = projected["entity/1/ids"], projected["entity/1/valid"]
    positions, present = _lookup(identifiers, queries)
    if not bool(jax.device_get(jnp.all(~valid | present))):
        raise ValueError(
            "PlanarPlcSupport.incidence must identify every resident edge in the source theorem; got an absent stable ID"
        )
    routed_counts = jax.device_put(jnp.where(valid, counts[positions], 0), queries.sharding)
    routed_keys = jax.device_put(keys[positions], projected["entity/1/keys"].sharding)
    mask = np.asarray(addressable(valid, storage.partition_index), dtype=np.bool_)
    local_ids = np.asarray(addressable(queries, storage.partition_index))[mask]
    rows = association_state_owner.entity_rows(mesh, 1, local_ids)
    local_keys = np.asarray(addressable(routed_keys, storage.partition_index))[mask]
    expected = np.sort(
        np.asarray(mesh.vertex_global_ids)[_vertices(mesh, 1)], axis=1
    )
    if (
        rows.size != mesh.entity_set(1).count
        or np.any(rows < 0)
        or np.unique(rows).size != rows.size
        or not np.array_equal(local_keys, expected[rows])
    ):
        raise ValueError(
            "PlanarPlcSupport.incidence must bind stable edge IDs to current corner keys; got a different resident closure"
        )
    result = np.empty((rows.size,), dtype=np.int64)
    result[rows] = np.asarray(addressable(routed_counts, storage.partition_index))[mask]
    return result


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
    initial_evidence: _dependency_phydrax_meshing__result.AbstractCollectiveMeshTheorem
    | None = None,
    collective_source: _dependency_phydrax_meshing__result.CellMeshingResult | None = None,
    judged: tuple[np.ndarray, ...] | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, ...]:
    """rational support of every class; ``judged`` limits owner-local proofs.

    An owner-local entity outside ``judged`` has a truncated resident star; its
    edge incidence and charts are incomplete, so it receives no local verdict.
    Source classification can instead supply ``collective_source``: authenticated
    global incidence then completes truncated stars by stable edge identity while
    the available coordinate charts still undergo the same support predicates.
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
        initial_evidence=initial_evidence,
        budget=budget,
    )
    if collective_source is not None:
        if collective_source.mesh is not mesh or collective_source.geometry is not geometry:
            raise ValueError(
                "PlanarPlcSupport.incidence must use the proved source carrier; got another mesh or coordinate map"
            )
        global_incidence = _source_edge_incidence(collective_source)
        complete = _complete_stars(mesh)[1]
        if np.any(edge_cells > global_incidence) or np.any(
            complete & (edge_cells != global_incidence)
        ):
            raise ValueError(
                "PlanarPlcSupport.incidence must agree with the source theorem on every whole resident star; got inconsistent incidence"
            )
        edge_cells = global_incidence
    # Support predicates judge the coordinate-map geometry, not its rounded
    # carrier: adapted barycentric midpoints are exact rational images. The
    # retained map bank already binds and agreement-checks every cell corner;
    # only a vertex no cell maps is rationalized from its carrier.
    exact = np.empty(coordinates.shape, dtype=object)
    unbound = [vertex for vertex, image in enumerate(cells.images) if image is None]
    if unbound:
        exact[unbound] = np.asarray(
            planar_primitives_owner.rational_points(coordinates[unbound], budget=budget),
            dtype=object,
        )
    for vertex, image in enumerate(cells.images):
        if image is not None:
            exact[vertex] = image
    coordinates = exact
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
                # Only the boolean support verdict and edge sign escape this
                # entity proof; its predicate arithmetic and Bernstein memos
                # are dead before the next entity is visited.
                with budget.polynomial_preparation_scope():
                    supported = all(
                        planar_point_support(
                            transfer,
                            kind,
                            index,
                            point,
                            work,
                            region_segments=region_segments.get(index)
                            if kind == 2
                            else None,
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
        # One request classifies one source under one transfer once: the
        # frozen verdict arrays are the complete result of the proof below.
        cache_key = (
            coordinate_scope_key(source.mesh, source.geometry),
            canonical_fingerprint((transfer.transfer_id, array_tree_fingerprint(source.associations))),
        )
        cached = budget.planar_class_cache.get(cache_key)
        if cached is not None:
            budget.reserve(0)
            return cached
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
        from ._initial_certification import InitialCollectiveMeshEvidence

        initial_evidence = None
        if source.mesh.storage is not None and source.collective_certificates is not None:
            embedding, coverage = source.collective_certificates
            if not isinstance(
                coverage, domain_coverage_owner.DomainCoverageCertificate
            ):
                raise ValueError(
                    "PlanarPlcSupport.coverage must certify the exact represented PLC domain; got approximate source coverage"
                )
        elif isinstance(
            source.collective_evidence,
            InitialCollectiveMeshEvidence,
        ):
            embedding, coverage = None, None
            initial_evidence = source.collective_evidence
        elif source.certification is not None:
            embedding, coverage = (
                source.certification.embedding,
                source.certification.coverage,
            )
            if coverage is not None and not isinstance(
                coverage, domain_coverage_owner.DomainCoverageCertificate
            ):
                raise ValueError(
                    "PlanarPlcSupport.coverage must certify the exact represented PLC domain; got approximate source coverage"
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
            initial_evidence=initial_evidence,
            collective_source=source if source.mesh.storage is not None else None,
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
        classes = _finish_classes(transfer, levels)
        budget.retain_owner(classes)
        budget.planar_class_cache[cache_key] = classes
        return classes


def restart_planar_lineage(
    historical: _dependency_phydrax_meshing__result.CellMeshingResult,
    archived: _dependency_phydrax_meshing__lineage.MeshLineage,
    source: CellMesh,
    target: CellMesh,
    /,
) -> _dependency_phydrax_meshing__lineage.MeshLineage:
    """Reconstruct placement-only support lineage; retain archived provenance.

    The archived transition supplies the stable-ID relation, not the current
    entity-set identity. Every row must be a preserved, one-to-one relation
    covering both current closures before their owning constructors bind it.
    """
    owner = _dependency_phydrax_meshing__lineage
    with agreed_local_step():
        if (
            archived.source_topology_id != historical.mesh.topology_id
            or archived.target_topology_id != target.topology_id
            or source.topology_id != target.topology_id
            or len(archived.entities) != 3
        ):
            raise ValueError("restart.PLC lineage must bind the authenticated placement-only transition")
        records = []
        for dimension in range(3):
            record = archived.entity_lineage(dimension)
            authenticated = owner.EntityLineage(
                dimension, record.source_entity_set_id, record.target_entity_set_id,
                record.source_global_ids, record.target_global_ids, record.relation_kinds,
                created_target_ids=record.created_target_ids,
                deleted_source_ids=record.deleted_source_ids,
            )
            before = np.asarray(record.source_global_ids, dtype=np.int64)
            after = np.asarray(record.target_global_ids, dtype=np.int64)
            current_source = np.asarray(source.entity_set(dimension).entity_ids, dtype=np.int64)
            current_target = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
            if (
                authenticated.lineage_id != record.lineage_id
                or record.source_entity_set_id != historical.mesh.entity_set(dimension).entity_set_id
                or record.target_entity_set_id != target.entity_set(dimension).entity_set_id
                or np.asarray(record.created_target_ids).size
                or np.asarray(record.deleted_source_ids).size
                or np.any(np.asarray(record.relation_kinds) != owner.EntityLineageKind.PRESERVED)
                or not np.array_equal(before, after)
                or np.unique(before).size != before.size
                or np.unique(current_source).size != current_source.size
                or np.unique(current_target).size != current_target.size
                or not np.array_equal(np.sort(before), np.sort(current_source))
                or not np.array_equal(np.sort(after), np.sort(current_target))
            ):
                raise ValueError("restart.PLC lineage must preserve every current stable ID exactly once")
            records.append(owner.EntityLineage(
                dimension, source.entity_set(dimension).entity_set_id,
                target.entity_set(dimension).entity_set_id,
                before, after, record.relation_kinds,
            ))
        authenticated_lineage = owner.MeshLineage(
            archived.source_topology_id, archived.target_topology_id, archived.entities,
        )
        if authenticated_lineage.lineage_id != archived.lineage_id:
            raise ValueError("restart.PLC lineage must retain its authenticated content")
        return owner.MeshLineage(source.topology_id, target.topology_id, tuple(records))


def restart_planar_associations(
    transfer: PlanarPlcTransfer,
    historical: tuple[_dependency_phydrax_meshing__result.CellMeshingResult, ...],
    predecessor: _dependency_phydrax_meshing__result.CellMeshingResult,
    current: CellMesh,
    after: _dependency_phydrax_meshing__result.AbstractCollectiveMeshTheorem,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[association_state_owner.GeometryAssociation, ...]:
    """Reconstruct current PLC science from disjoint authenticated archive owners.

    Archived storage remains historical. Its owned class records travel by stable
    entity ID from the process assigned that archive to the actual current mesh.
    No resident ghost copy supplies authority, and no historical owner is relabeled.
    The publication owner must certify these rows and run ``planar_classes`` on
    the resulting current result before it may answer any field query.
    """
    with agreed_local_step():
        before = predecessor.collective_evidence
        storage = current.storage
        if before is None or after is None or storage is None:
            raise ValueError("restart.PLC must retain historical and current collective evidence")
        if (
            predecessor.mesh.topology_id != current.topology_id
            or before.global_entity_counts != after.global_entity_counts
        ):
            raise ValueError("restart.PLC must preserve topology and bind current certificates")
        assignment = np.zeros(before.partition_count, dtype=np.int32)
        last_owner = -1
        banks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        owned_ids: list[list[np.ndarray]] = [[], [], []]
        owned_classes: list[list[np.ndarray]] = [[], [], []]
        owned_parameters: list[list[np.ndarray]] = [[], [], []]
        for source in historical:
            saved = source.mesh.storage
            evidence = source.collective_evidence
            if (
                saved is None
                or evidence is None
                or evidence.evidence_id != before.evidence_id
                or source.mesh.topology_id != predecessor.mesh.topology_id
                or saved.partition_count != before.partition_count
                or not 0 <= saved.partition_index < before.partition_count
                or saved.partition_index <= last_owner
                or assignment[saved.partition_index]
            ):
                raise ValueError("restart.PLC archive owners must be distinct members of the accepted epoch")
            assignment[saved.partition_index] = 1
            last_owner = saved.partition_index
        if isinstance(before, _dependency_phydrax_meshing__result.CollectiveMeshEvidence):
            expected = np.zeros(before.partition_count, dtype=np.int32)
            for shard in before.compiled_states.mesh.cell_ids.addressable_shards:
                selection = shard.index[0]
                expected[selection] = 1
            if not np.array_equal(assignment, expected):
                raise ValueError(
                    "restart.PLC archive owners must match the authenticated current-to-historical placement"
                )
    # Reject missing, duplicate, and foreign archive assignments before any
    # source classification can enter its theorem-backed numerical routes.
    assignments = np.asarray(
        multihost_utils.process_allgather(assignment, tiled=False), dtype=np.int32
    ).reshape((-1, before.partition_count))
    with agreed_local_step():
        if np.any(np.sum(assignments, axis=0) != 1):
            raise ValueError("restart.PLC archives must cover every historical owner exactly once")
    for source in historical:
        with agreed_local_step():
            planar_source_associations(transfer, source, budget=budget)
        with agreed_local_step():
            classes = planar_classes(transfer, source, budget=budget)
            saved = source.mesh.storage
            if saved is None:
                raise ValueError("restart.PLC archive must retain its historical storage")
        for dimension in range(3):
            with agreed_local_step():
                owned = np.asarray(saved.entity_owned[dimension], dtype=np.bool_)
                ids = np.asarray(saved.entity_global_ids[dimension], dtype=np.int64)[owned]
                count = before.global_entity_counts[dimension]
                table = jnp.asarray(before.entity_ids[dimension][:count], dtype=jnp.int64)
                order = jnp.argsort(table, stable=True)
            owners, present = local_logical_lookup(
                table[order], before.entity_owners[dimension][:count][order], ids
            )
            with agreed_local_step():
                if not np.all(present) or np.any(np.asarray(owners) != saved.partition_index):
                    raise ValueError("restart.PLC archive rows must match accepted historical ownership")
                rows = _storage_rows(source.mesh, dimension)[owned]
                association = association_routes.association_for_dimension(source, dimension)
                association_rows = association_state_owner.entity_rows(
                    source.mesh, dimension, np.asarray(association.target_global_ids, dtype=np.int64)
                )
                # Initial edge associations constrain only represented boundary
                # strata. Interior region classes are independently proved above;
                # they carry no edge orientation or source-chart parameter.
                signs = np.zeros(source.mesh.entity_set(dimension).count, dtype=np.int64)
                source_parameters = np.full((signs.size, 2), np.nan, dtype=np.float64)
                signs[association_rows] = np.asarray(association.orientations, dtype=np.int64)
                source_parameters[association_rows] = np.asarray(association.parameters)
                owned_ids[dimension].append(ids)
                owned_classes[dimension].append(
                    np.column_stack((
                        classes[dimension].dimensions[rows],
                        classes[dimension].indices[rows],
                        signs[rows],
                    ))
                )
                owned_parameters[dimension].append(source_parameters[rows])
    with agreed_local_step():
        for dimension in range(3):
            banks.append((
                np.concatenate(owned_ids[dimension]) if owned_ids[dimension] else np.empty(0, dtype=np.int64),
                np.concatenate(owned_classes[dimension]) if owned_classes[dimension] else np.empty((0, 3), dtype=np.int64),
                np.concatenate(owned_parameters[dimension]) if owned_parameters[dimension] else np.empty((0, 2), dtype=np.float64),
            ))
    counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray([bank[0].size for bank in banks], dtype=np.int64), tiled=False
        ), dtype=np.int64
    ).reshape((-1, 3))
    with agreed_local_step():
        if (
            np.any(np.sum(assignments, axis=0) != 1)
            or not np.array_equal(np.sum(counts, axis=0), before.global_entity_counts)
        ):
            raise ValueError("restart.PLC archives must cover every historical owner and entity exactly once")
        devices = process_owner_devices()
        if len(devices) != assignments.shape[0] or storage.partition_index != jax.process_index():
            raise ValueError("restart.PLC archive routes must bind the current process-owner devices")
        archive_owners = np.argmax(assignments, axis=0).astype(np.int32)
    levels: list[PlcEntityClasses] = []
    parameters: list[np.ndarray] = []
    orientations: list[np.ndarray] = []
    for dimension, (ids, classes, params) in enumerate(banks):
        with agreed_local_step():
            queries = np.asarray(current.entity_set(dimension).entity_ids, dtype=np.int64)
            if np.unique(queries).size != queries.size:
                raise ValueError("restart.PLC current entities must have unique stable IDs")
            current_count = after.global_entity_counts[dimension]
            current_table = jnp.asarray(after.entity_ids[dimension][:current_count], dtype=jnp.int64)
            current_order = jnp.argsort(current_table, stable=True)
            current_owners, current_present = local_logical_lookup(
                current_table[current_order],
                after.entity_owners[dimension][:current_count][current_order], queries,
            )
            stored_rows = _storage_rows(current, dimension)
            if (
                not np.all(current_present)
                or not np.array_equal(
                    np.asarray(current_owners)[stored_rows],
                    np.asarray(storage.entity_owner[dimension]),
                )
                or not np.array_equal(
                    np.asarray(storage.entity_owned[dimension], dtype=np.bool_),
                    np.asarray(current_owners)[stored_rows] == storage.partition_index,
                )
            ):
                raise ValueError("restart.PLC current entities must bind accepted owner placement")
            count = before.global_entity_counts[dimension]
            table = jnp.asarray(before.entity_ids[dimension][:count], dtype=jnp.int64)
            order = jnp.argsort(table, stable=True)
            owners, present = local_logical_lookup(
                table[order], before.entity_owners[dimension][:count][order], queries
            )
            if not np.all(present):
                raise ValueError("restart.PLC current entities must exist in the authenticated historical bank")
            destinations = archive_owners[np.asarray(owners, dtype=np.int32)]
        routed = []
        for label, values in (("classes", classes), ("parameters", params)):
            (answer,) = resolve_owner_local_record_values(
                (ids,), (values,), (queries,), (destinations,),
                ranks=(storage.partition_index,), devices=devices, axis_name="parts",
                message_capacity=max(count, 1),
                identity=f"restart-plc:{before.evidence_id}:{after.evidence_id}:{dimension}:{label}",
            )
            routed.append(np.asarray(jax.device_get(answer)))
        dims, indices, signs = routed[0].T
        levels.append(PlcEntityClasses(
            dims, indices, np.zeros(dims.shape, dtype=np.bool_), np.empty(dims.shape, dtype=np.int64)
        ))
        parameters.append(routed[1])
        orientations.append(signs)
    with agreed_local_step():
        output = []
        for dimension, level in enumerate(levels):
            ids = np.asarray(current.entity_set(dimension).entity_ids, dtype=np.int64)
            roles = source_entity_roles(level.dimensions)
            output.append(association_state_owner.GeometryAssociation(
                association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                transfer.domain.source_id, transfer.source_revision,
                current.entity_set(dimension).entity_set_id, ids,
                tuple(f"{transfer.source_revision}:{role.value}:{index}" for role, index in zip(roles, level.indices.tolist(), strict=True)),
                np.zeros(ids.shape, dtype=np.float64), exact=True,
                source_dimensions=level.dimensions, source_indices=level.indices,
                source_entity_roles=roles, parameters=parameters[dimension],
                orientations=orientations[dimension],
                provenance=association_state_owner.GeometryAssociationProvenance.CLASSIFICATION,
            ))
        return tuple(output)


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class PlanarPlcSourceProof:
    """Planar source maps and support classes earned by one preparation.

    Process-local and ledger-free. Only an execution of the same request may
    bind it, after that request's charged preparation receipt already holds
    the source work that earned it. Binding still charges its actual retained
    object-profile visits and bytes, but never repeats the source predicates.
    """

    mesh: CellMesh
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    transfer_id: str
    maps: PlanarMaps
    classes: tuple[PlcEntityClasses, ...]
    source_bank: algebra.CoordinateSourceBank
    plc_carrier_bound: bool
    published_carrier_bound: bool

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "PlanarPlcSourceProof must remain process-local; got serialization"
        )


@dataclass(frozen=True, slots=True)
class _PlanarSourceProofSeal:
    fields: tuple[object, ...]
    arrays: tuple[tuple[object, ...], ...]
    polynomials: tuple[object, ...]
    basis: tuple[object, ...]


# Immutable numeric payloads, every nested field identity, array interpretation
# metadata and exact mutable-polynomial content are all part of admission.
_issued_planar_proofs: WeakKeyDictionary[
    PlanarPlcSourceProof, _PlanarSourceProofSeal
] = WeakKeyDictionary()


def _planar_proof_fields(proof: PlanarPlcSourceProof, /) -> tuple[object, ...]:
    return (
        proof.mesh,
        proof.geometry,
        proof.transfer_id,
        proof.maps,
        proof.maps.kinds,
        proof.maps.vertices,
        proof.maps.images,
        proof.maps.polynomials,
        proof.classes,
        proof.source_bank,
        proof.plc_carrier_bound,
        proof.published_carrier_bound,
        *(value for level in proof.classes for value in level),
    )


def _planar_array_layouts(
    proof: PlanarPlcSourceProof, /
) -> tuple[tuple[object, ...], ...]:
    arrays = (proof.maps.vertices, *(value for level in proof.classes for value in level))
    if any(type(value) is not np.ndarray for value in arrays):
        raise ValueError(
            "PlanarPlcSupport.source_proof must retain its issued numeric banks; got altered arrays"
        )
    return tuple(
        (
            value.dtype.str,
            value.shape,
            value.strides,
            value.__array_interface__["data"][0],
            value.nbytes,
        )
        for value in arrays
    )


def _planar_polynomial_content(
    maps: PlanarMaps,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[object, ...]:
    # Reuse the predecessor owner's exact rational/radical scalar encoding;
    # mutable polynomial dictionaries cannot borrow an object-identity seal.
    from ..discretization._cell_geometry_predecessor import _coefficient_content_key

    terms = sum(
        len(polynomial)
        for polynomials in maps.polynomials.values()
        for polynomial in polynomials
    )
    with budget.temporary_scope():
        budget.reserve(len(maps.polynomials) + terms, 256 + 256 * terms)
        return tuple(
            (
                row,
                tuple(
                    tuple(
                        (index, _coefficient_content_key(coefficient))
                        for index, coefficient in polynomial.items()
                    )
                    for polynomial in polynomials
                ),
            )
            for row, polynomials in sorted(maps.polynomials.items())
        )


def issue_planar_source_proof(
    transfer: PlanarPlcTransfer,
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PlanarPlcSourceProof | None:
    """Issue this ledger's completed source maps and classes, if both exist."""
    key = coordinate_scope_key(source.mesh, source.geometry)
    maps = budget.planar_cell_map_cache.get(key)
    classes = budget.planar_class_cache.get((key, transfer.transfer_id))
    if maps is None or classes is None:
        return None
    bank = budget.source_bank_cache.get(coordinate_source_bank_key(source.geometry))
    if bank is None:
        return None
    classes = tuple(
        PlcEntityClasses(
            seal_coordinate_operand_array(level.dimensions, budget=budget),
            seal_coordinate_operand_array(level.indices, budget=budget),
            seal_coordinate_operand_array(level.resolved, budget=budget),
            seal_coordinate_operand_array(level.codes, budget=budget),
        )
        for level in classes
    )
    budget.retain_owner(classes)
    budget.planar_class_cache[(key, transfer.transfer_id)] = classes
    proof = PlanarPlcSourceProof(
        source.mesh,
        source.geometry,
        transfer.transfer_id,
        maps,
        classes,
        bank,
        key in budget.plc_carrier_binding_cache,
        key in budget.published_affine_carrier_cache,
    )
    fields = _planar_proof_fields(proof)
    budget.reserve(len(fields), 256 + 16 * len(fields))
    layouts = _planar_array_layouts(proof)
    polynomials = _planar_polynomial_content(maps, budget=budget)
    basis = (layouts, polynomials)
    budget.retain_basis(basis)
    _issued_planar_proofs[proof] = _PlanarSourceProofSeal(
        fields, layouts, polynomials, basis
    )
    return proof


def bind_planar_source_proof(
    transfer: PlanarPlcTransfer,
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    proof: PlanarPlcSourceProof,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Admit an issued proof of this source and transfer into a request ledger.

    The caller binds only proofs whose earning preparation receipt this
    request already charges. A foreign, copied or field-replaced proof refuses.
    """
    seal = (
        _issued_planar_proofs.get(proof)
        if type(proof) is PlanarPlcSourceProof
        else None
    )
    if (
        seal is None
        or type(proof.maps) is not PlanarMaps
        or len(_planar_proof_fields(proof)) != len(seal.fields)
        or any(
            actual is not sealed
            for actual, sealed in zip(
                _planar_proof_fields(proof), seal.fields, strict=True
            )
        )
        or proof.mesh is not source.mesh
        or proof.geometry is not source.geometry
        or proof.transfer_id != transfer.transfer_id
    ):
        raise ValueError(
            "PlanarPlcSupport.source_proof must be an issued proof of this source and transfer; got a foreign or altered proof"
        )
    budget.reserve(
        len(seal.fields), 256 + 16 * len(seal.fields) + 512 * len(seal.arrays)
    )
    if (
        _planar_array_layouts(proof) != seal.arrays
        or _planar_polynomial_content(proof.maps, budget=budget) != seal.polynomials
    ):
        raise ValueError(
            "PlanarPlcSupport.source_proof must retain its issued bank layouts and polynomial content; got a foreign or altered proof"
        )
    budget.retain_basis(seal.basis)
    key = coordinate_scope_key(source.mesh, source.geometry)
    maps = proof.maps
    bank_key = coordinate_source_bank_key(source.geometry)
    if (
        budget.planar_cell_map_cache.get(key) is maps
        and budget.planar_class_cache.get((key, proof.transfer_id)) is proof.classes
        and budget.source_bank_cache.get(bank_key) is proof.source_bank
        and budget.coordinate_vertex_image_cache.get(key) is maps.images
        and (not proof.plc_carrier_bound or key in budget.plc_carrier_binding_cache)
        and (
            not proof.published_carrier_bound
            or key in budget.published_affine_carrier_cache
        )
    ):
        return
    budget.retain_basis((bank_key, proof.source_bank), operand=True)
    budget.retain_basis((key, maps, maps.kinds, maps.images, maps.polynomials))
    budget.retain_owner(maps.vertices)
    budget.retain_owner(proof.classes)
    budget.planar_cell_map_cache.setdefault(key, maps)
    budget.planar_class_cache.setdefault((key, proof.transfer_id), proof.classes)
    budget.coordinate_vertex_image_cache.setdefault(key, maps.images)
    budget.source_bank_cache.setdefault(bank_key, proof.source_bank)
    if proof.plc_carrier_bound:
        budget.retain_basis((key,))
        budget.plc_carrier_binding_cache.add(key)
    if proof.published_carrier_bound:
        budget.retain_basis((key,))
        budget.published_affine_carrier_cache.add(key)


@checked
def planar_protected_edges(
    transfer: PlanarPlcTransfer,
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    midpoint_required: bool = True,
    exact_chart_midpoints: bool = False,
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
    exact_affine_midpoints = False
    if midpoint_required and exact_chart_midpoints:
        elements, _, _ = source.geometry.resolve(source.mesh, budget=budget)
        exact_affine_midpoints = all(
            block.cell_kind == "triangle"
            and algebra.affine_simplex_source(element, budget=budget)
            for block, element in zip(source.mesh.blocks, elements, strict=True)
        )
    if midpoint_required:
        work = [transfer.maximum_support_queries]
        for row in np.flatnonzero(levels[1].dimensions == 1).tolist():
            if exact_affine_midpoints and levels[1].resolved[row]:
                # The issued source class proved the whole affine edge image
                # in this closed segment. Its exact chart midpoint is a convex
                # combination of those endpoints; the responsible producer
                # keeps that map and separately validates its carrier law.
                continue
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

    A resident edge's star is whole when both incident cells are resident, or
    its one resident cell meets the physical boundary there. In a conforming
    planar triangulation a vertex star is the fan connected across its incident
    edges, so it is whole exactly when every incident edge is whole. A resident
    cell is judged when each of its facets is matched or physical boundary. A
    star truncated at the artificial halo boundary yields no verdict.
    """
    storage = mesh.storage
    if (
        storage is None
        or storage.local_neighborhood_complete is None
        or storage.local_physical_boundary_facets is None
    ):
        raise ValueError(
            "PlanarPlcSupport.storage must retain resident-star completeness evidence; got missing evidence"
        )
    marks: list[np.ndarray] = []
    for dimension, local in (
        (1, storage.local_physical_boundary_facets),
        (2, storage.local_neighborhood_complete),
    ):
        flags = np.asarray(local, dtype=np.bool_)
        rows = association_state_owner.entity_rows(
            mesh, dimension, np.asarray(storage.entity_global_ids[dimension], dtype=np.int64)
        )
        count = mesh.entity_set(dimension).count
        if (
            flags.shape != rows.shape
            or rows.size != count
            or np.any(rows < 0)
            or np.unique(rows).size != count
        ):
            raise ValueError(
                "PlanarPlcSupport.neighborhood must label every resident entity once; got invalid resident IDs"
            )
        marks.append(np.zeros((count,), dtype=np.bool_))
        marks[-1][rows] = flags
    physical, cells = marks
    incidence = np.bincount(
        association_routes.incidence_pairs(mesh, 1, 2)[:, 0], minlength=physical.size
    )
    edges = (incidence == 2) | ((incidence == 1) & physical)
    links = association_routes.incidence_pairs(mesh, 0, 1)
    vertex_count = mesh.entity_set(0).count
    truncated = np.bincount(links[:, 0], minlength=vertex_count) == 0
    np.logical_or.at(truncated, links[:, 0], ~edges[links[:, 1]])
    return ~truncated, edges, cells


_VERDICT_COLUMNS = 5
_LevelParents = list[tuple[np.ndarray, np.ndarray]]
_LocalVerdicts = tuple[
    list[PlcEntityClasses], _LevelParents, tuple[np.ndarray, ...], tuple[np.ndarray, ...]
]
# Per dimension: sorted accepted source IDs, their source dimensions and indices.
_RemoteClasses = tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]


def _storage_rows(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Resident mesh rows in owner-local storage order."""
    storage = mesh.storage
    if storage is None:
        raise ValueError(
            "PlanarPlcSupport.storage must bind owner-local entity routes; got no storage"
        )
    ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    stored = np.asarray(storage.entity_global_ids[dimension], dtype=np.int64)
    if (
        stored.size != ids.size
        or np.unique(ids).size != ids.size
        or np.unique(stored).size != stored.size
    ):
        raise ValueError(
            "PlanarPlcSupport.storage must order every resident entity exactly once; got missing or duplicate IDs"
        )
    order = np.argsort(ids, kind="stable")
    rows = order[np.minimum(np.searchsorted(ids[order], stored), max(ids.size - 1, 0))]
    if stored.size != ids.size or np.any(ids[rows] != stored):
        raise ValueError(
            "PlanarPlcSupport.storage must order exactly the resident entities; got another entity inventory"
        )
    return rows


def _owner_answers(
    mesh: CellMesh,
    dimension: int,
    values: np.ndarray,
    queries: np.ndarray,
    owners: np.ndarray,
    /,
) -> np.ndarray:
    """Collective: rows of ``values`` (storage order) answered by each query's owner.

    Only the accepted owner of an entity answers for it, through bounded peer
    packets; every process enters, including one without queries.
    """
    storage = mesh.storage
    if storage is None:
        raise ValueError(
            "PlanarPlcSupport.storage must bind owner-local entity routes; got no storage"
        )
    (answer,) = resolve_owner_local_field_values(
        (mesh,),
        (jnp.asarray(values),),
        (np.asarray(queries, dtype=np.int64),),
        (np.asarray(owners, dtype=np.int32),),
        dimension,
        source_topology_id=storage.logical_topology_id,
        devices=process_owner_devices(),
        axis_name="parts",
        message_capacity=max(storage.global_entity_counts[dimension], 1),
    )
    return np.asarray(jax.device_get(answer))


def _owner_certified_classes(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    old: tuple[PlcEntityClasses, ...] | None,
    queries: tuple[np.ndarray, ...] | None,
    failure: Exception | None,
    /,
) -> _RemoteClasses:
    """Collective: accepted classes of lineage sources beyond this closure.

    Every process enters, including one whose local proof already failed: one
    fixed-size gather first agrees on refusal and on which dimensions carry
    queries. Each accepted source entity is answered only by the process that
    owns it in the accepted source publication; no class table is replicated.
    An owner that cannot be located leaves its query unresolved, so the
    lineage proof refuses it after the next agreement.
    """
    counts = (
        (0, 0, 0)
        if queries is None or failure is not None
        else tuple(int(ids.size) for ids in queries)
    )
    gathered = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(failure is not None), *counts), dtype=np.int64),
            tiled=False,
        ),
        dtype=np.int64,
    ).reshape((-1, 4))
    if np.any(gathered[:, 0]):
        if failure is not None:
            raise failure
        raise ValueError(
            "PlanarPlcSupport.lineage must be classified by every responsible process; got a process refusal"
        )
    if old is None or queries is None:
        raise ValueError(
            "PlanarPlcSupport.lineage must retain local source classes; got no local classification"
        )
    mesh, evidence = source.mesh, source.collective_evidence
    if mesh.storage is None or evidence is None:
        raise ValueError(
            "PlanarPlcSupport.source must retain its accepted collective publication; got no storage or evidence"
        )
    resolved = []
    for dimension, ids in enumerate(queries):
        if not np.any(gathered[:, 1 + dimension]):
            empty = np.zeros((0,), dtype=np.int64)
            resolved.append((empty, empty, empty))
            continue
        count = evidence.global_entity_counts[dimension]
        table = jnp.asarray(evidence.entity_ids[dimension][:count], dtype=jnp.int64)
        order = jnp.argsort(table, stable=True)
        owners, present = local_logical_lookup(
            table[order],
            jnp.asarray(evidence.entity_owners[dimension][:count], dtype=jnp.int32)[
                order
            ],
            ids,
        )
        known = ids[np.asarray(present, dtype=np.bool_)]
        classes = np.column_stack(
            (old[dimension].dimensions, old[dimension].indices)
        ).astype(np.int64)[_storage_rows(mesh, dimension)]
        answers = _owner_answers(
            mesh,
            dimension,
            classes,
            known,
            np.asarray(owners, dtype=np.int32)[np.asarray(present, dtype=np.bool_)],
        )
        resolved.append((known, answers[:, 0], answers[:, 1]))
    return tuple(resolved)


def _agreed_owner_verdicts(
    target: CellMesh,
    local: _LocalVerdicts | None,
    failure: Exception | None,
    /,
) -> tuple[list[PlcEntityClasses], _LevelParents, tuple[np.ndarray, ...]]:
    """Collective: every resident entity carries its owner's whole-star verdict.

    Every process enters, including one whose local proof failed; one
    fixed-size gather first agrees on refusal and on owner coverage. The owner
    of an entity holds its whole incident star and must judge it; every other
    resident copy receives the owner's verdict through bounded peer packets,
    and a copy that judged the entity itself must agree exactly on stratum and
    orientation. No verdict table is replicated across processes.
    """

    storage = target.storage
    if storage is None:
        raise ValueError(
            "PlanarPlcSupport.verdicts must bind mesh storage; got no storage"
        )
    staged = []
    unjudged: list[np.ndarray] = []
    if failure is None:
        try:
            if local is None:
                raise ValueError(
                    "PlanarPlcSupport.verdicts must retain local proof results; got no local proof"
                )
            levels, parents, orientations, judged = local
            for dimension in range(3):
                rows = _storage_rows(target, dimension)
                owned = np.asarray(storage.entity_owned[dimension], dtype=np.bool_)
                verdicts = np.column_stack(
                    (
                        levels[dimension].dimensions,
                        levels[dimension].indices,
                        orientations[dimension].astype(np.int64),
                        parents[dimension][0].astype(np.int64),
                        parents[dimension][1],
                    )
                ).astype(np.int64)[rows]
                decided = np.asarray(judged[dimension], dtype=np.bool_)[rows]
                unjudged.append(
                    np.asarray(storage.entity_global_ids[dimension], dtype=np.int64)[
                        owned & ~decided
                    ]
                )
                staged.append((rows, owned, verdicts, decided))
        except Exception as error:
            failure = error
    status = np.asarray(
        (int(failure is not None), int(any(ids.size for ids in unjudged))),
        dtype=np.int32,
    )
    gathered = np.asarray(
        multihost_utils.process_allgather(status, tiled=False), dtype=np.int32
    ).reshape((-1, 2))
    if np.any(gathered[:, 0]):
        if failure is not None:
            raise failure
        raise ValueError(
            "PlanarPlcSupport.verdicts must be proved by every responsible process; got a process refusal"
        )
    if np.any(gathered[:, 1]):
        raise association_routes.AssociationPropagationError(
            "PlanarPlcSupport.entity must be judged by its owner holding its whole incident star; got an unjudged owned entity",
            np.concatenate(unjudged) if unjudged else np.zeros((0,), dtype=np.int64),
        )
    if local is None:
        raise ValueError(
            "PlanarPlcSupport.verdicts must retain local proof results; got no local proof"
        )
    levels, parents, _, _ = local
    adopted = []
    conflicts: list[np.ndarray] = []
    for dimension, (rows, owned, verdicts, decided) in enumerate(staged):
        ids = np.asarray(storage.entity_global_ids[dimension], dtype=np.int64)
        ghosts = np.flatnonzero(~owned)
        answers = _owner_answers(
            target,
            dimension,
            verdicts,
            ids[ghosts],
            np.asarray(storage.entity_owner[dimension], dtype=np.int32)[ghosts],
        )
        differs = np.any(answers[:, :3] != verdicts[ghosts, :3], axis=1)
        conflicts.append(ids[ghosts[differs & decided[ghosts]]])
        merged = verdicts.copy()
        merged[ghosts] = answers
        ordered = np.empty_like(merged)
        ordered[rows] = merged
        levels[dimension].dimensions[:] = ordered[:, 0]
        levels[dimension].indices[:] = ordered[:, 1]
        parents[dimension][0][:] = ordered[:, 3]
        parents[dimension][1][:] = ordered[:, 4]
        adopted.append(ordered[:, 2].astype(np.int8))
    conflicting = np.concatenate(conflicts)
    if np.any(
        np.asarray(
            multihost_utils.process_allgather(
                np.int32(conflicting.size > 0), tiled=False
            )
        )
    ):
        raise association_routes.AssociationPropagationError(
            "PlanarPlcSupport.verdicts must agree across processes holding the incident star; got conflicting verdicts",
            conflicting,
        )
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
    remote: _RemoteClasses | None,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None,
    coverage: domain_coverage_owner.DomainCoverageCertificate | None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    list[PlcEntityClasses],
    list[tuple[np.ndarray, np.ndarray]],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...] | None,
]:
    """Lineage, consensus and closure classes with their rational support proof.

    ``judged`` restricts locally inferred classes and proofs to owner-local
    whole stars; ``None`` decides every entity of a serial mesh. ``remote``
    holds the owner-certified classes of accepted source entities beyond this
    closure. A vertex whose class and parent come wholly from authenticated
    lineage reads no incident cell or edge: its verdict is the inherited source
    class plus exact support of its own coordinate-map image, so every resident
    copy judges it. Stratum consensus and vertex closure still require a whole
    resident star. Returns the entities actually judged.
    """
    cell_maps(target, geometry, budget=budget)
    # Source support produces integer class arrays, not its rational predicate
    # intermediates. Retained coordinate banks remain charged to the request.
    with budget.temporary_scope():
        old = planar_classes(transfer, source, budget=budget)
    levels, parents = _lineage_levels(source.mesh, target, lineage, old, remote)
    cell_ids = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)
    ambiguous = levels[2].dimensions != 2
    if np.any(ambiguous):
        raise association_routes.AssociationPropagationError(
            "PlanarPlcSupport.cells must have unambiguous source-region lineage; got incompatible region ancestry",
            cell_ids[ambiguous],
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
            if judged is not None:
                judged = (judged[0] | (level.dimensions >= 0), *judged[1:])
                decided = judged[0]
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
    return levels, parents, orientations, judged


def _lineage_sources(
    source: CellMesh,
    lineage: _dependency_phydrax_meshing__lineage.MeshLineage,
    /,
) -> tuple[np.ndarray, ...]:
    """Sorted lineage source IDs of each dimension beyond the accepted closure."""
    absent = []
    for dimension in range(3):
        ids = np.asarray(
            lineage.entity_lineage(dimension).source_global_ids, dtype=np.int64
        )
        rows = association_state_owner.entity_rows(source, dimension, ids)
        absent.append(np.unique(ids[rows < 0]))
    return tuple(absent)


def _lineage_levels(
    source: CellMesh,
    target: CellMesh,
    lineage: _dependency_phydrax_meshing__lineage.MeshLineage,
    old: tuple[PlcEntityClasses, ...],
    remote: _RemoteClasses | None,
    /,
) -> tuple[list[PlcEntityClasses], list[tuple[np.ndarray, np.ndarray]]]:
    """Inherited classes and parents of every target entity.

    An owner-local target closure may descend from accepted source entities
    beyond this process's source closure; ``remote`` holds their accepted
    classes, answered by their owners. A serial lineage must be wholly declared.
    """
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
        absent = source_rows < 0
        if np.any(target_rows < 0) or (np.any(absent) and remote is None):
            raise ValueError(
                "PlanarPlcSupport.lineage must index declared mesh entities; got undeclared entity IDs"
            )
        classes = np.full((source_ids.size, 2), -1, dtype=np.int64)
        present = np.flatnonzero(~absent)
        classes[present, 0] = old[dimension].dimensions[source_rows[present]]
        classes[present, 1] = old[dimension].indices[source_rows[present]]
        if remote is not None and np.any(absent):
            known, known_dimensions, known_indices = remote[dimension]
            queried = source_ids[absent]
            positions = np.minimum(
                np.searchsorted(known, queried), max(known.size - 1, 0)
            )
            if known.size == 0 or np.any(known[positions] != queried):
                raise ValueError(
                    "PlanarPlcSupport.lineage must bind owner-certified source classes; got an unresolved accepted source entity"
                )
            classes[absent, 0] = known_dimensions[positions]
            classes[absent, 1] = known_indices[positions]
        # A target entity inherits a class only from one unanimous source class.
        pairs = np.unique(np.column_stack((target_rows, classes)), axis=0)
        rows, first, counts = np.unique(
            pairs[:, 0], return_index=True, return_counts=True
        )
        chosen = rows[counts == 1]
        dims[chosen] = pairs[first[counts == 1], 1]
        indices[chosen] = pairs[first[counts == 1], 2]
        lowest = np.full(ids.shape, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(lowest, target_rows, source_ids)
        parent_dims[chosen] = dimension
        parent_ids[chosen] = lowest[chosen]
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
        if target.storage is None:
            # Source checks bind the shared serial source.
            vertex, dimensions = planar_source_associations(
                transfer, source, budget=budget
            )
            association_routes.require_transfer_lineage(
                source.mesh,
                lineage,
                target,
                message="PlanarPlcSupport.lineage must bind this source and target; got different mesh identities",
            )
            levels, parents, orientations, _ = _propagated_levels(
                transfer,
                source,
                lineage,
                target,
                geometry,
                None,
                remote=None,
                embedding=embedding,
                coverage=coverage,
                budget=budget,
            )
        else:
            # Only verdicts whose inputs are all resident contribute. Every
            # owner-local failure, including one in the collective source
            # classification, is carried into the next agreement so no
            # process waits in a gather that a refusing peer never enters.
            # Lineage beyond the accepted closure inherits the classes its
            # accepted owners answer for, never a truncated local guess.
            local: _LocalVerdicts | None = None
            failure: Exception | None = None
            classified: tuple[association_state_owner.GeometryAssociation, tuple[int, ...]] | None = None
            old: tuple[PlcEntityClasses, ...] | None = None
            queries: tuple[np.ndarray, ...] | None = None
            try:
                classified = planar_source_associations(transfer, source, budget=budget)
                association_routes.require_transfer_lineage(
                    source.mesh,
                    lineage,
                    target,
                    message="PlanarPlcSupport.lineage must bind this source and target; got different mesh identities",
                )
                with budget.temporary_scope():
                    old = planar_classes(transfer, source, budget=budget)
                queries = _lineage_sources(source.mesh, lineage)
            except Exception as error:
                failure = error
            remote = _owner_certified_classes(source, old, queries, failure)
            try:
                levels, parents, orientations, judged = _propagated_levels(
                    transfer,
                    source,
                    lineage,
                    target,
                    geometry,
                    _complete_stars(target),
                    remote=remote,
                    embedding=embedding,
                    coverage=coverage,
                    budget=budget,
                )
                if judged is None:
                    raise ValueError(
                        "PlanarPlcSupport.verdicts must retain owner-local judged entities; got no judged entities"
                    )
                local = (levels, parents, orientations, judged)
            except Exception as error:
                failure = error
            levels, parents, orientations = _agreed_owner_verdicts(target, local, failure)
            if classified is None:
                raise ValueError(
                    "PlanarPlcSupport.source must retain agreed source classes; got no source classification"
                )
            vertex, dimensions = classified
    final = _finish_classes(transfer, levels)
    output: list[association_state_owner.GeometryAssociation] = []
    for dimension in (0, *dimensions):
        previous = (
            vertex
            if dimension == 0
            else association_routes.association_for_dimension(source, dimension)
        )
        level = final[dimension]
        ids = np.asarray(
            type(target).entity_set(target, dimension).entity_ids, dtype=np.int64
        )
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
                type(target).entity_set(target, dimension).entity_set_id,
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
