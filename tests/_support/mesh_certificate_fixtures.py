#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent mesh/source builders and explicit coordinate-ledger fixtures."""

import math
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from fractions import Fraction
from itertools import product
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
    require_scalar_coordinate_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.meshing._association import PlcAssociationTransfer
from phydrax.meshing._host_banks import entity_key
from phydrax.meshing._plc_mapped_support import certify_mapped_plc_associations
from phydrax.meshing.providers._native_publication import NativeCertificationRequest
from tests._support.domain_coverage import split_square_domain


@pytest.fixture
def coordinate_budget() -> Iterator[Callable[[int, int], CoordinateEnclosureBudget]]:
    """Settle explicit test-owned tickets without changing declared domain maxima."""
    with ExitStack() as resources:

        def create(work: int, memory: int) -> CoordinateEnclosureBudget:
            ticket = resources.enter_context(
                HostBatchTicket(
                    work,
                    queries_upper=work,
                    bytes_upper=memory,
                )
            )
            ledger = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=work,
                maximum_memory_bytes=memory,
            )
            resources.enter_context(ledger.activate())
            return ledger

        yield create


@pytest.fixture
def reference_budget(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> CoordinateEnclosureBudget:
    limits = phx.geometry.MeshCertificateLimits()
    return coordinate_budget(limits.maximum_work_units, limits.maximum_scratch_bytes)


TETRAHEDRON = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)


def embedding(mesh: Any) -> Any:
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    return (
        geometry,
        validity,
        phx.geometry.certify_global_embedding(mesh, geometry, validity),
    )


def two_tetrahedra(second: np.ndarray) -> Any:
    return phx.discretization.CellMesh.from_tetrahedra(
        np.concatenate((TETRAHEDRON, second)),
        np.asarray(((0, 1, 2, 3), (4, 5, 6, 7))),
    )


def annulus_with_island(origin: tuple[float, float]) -> Any:
    points: list[tuple[float, float]] = [
        (0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2),
        (3, 3), (2, 3), (1, 3), (0, 3), (0, 2), (0, 1),
        (1, 1), (2, 1), (2, 2), (1, 2),
    ]  # fmt: skip
    quads = (
        (0, 1, 12, 11), (1, 2, 13, 12), (2, 3, 4, 13), (13, 4, 5, 14),
        (14, 5, 6, 7), (15, 14, 7, 8), (10, 15, 8, 9), (11, 12, 15, 10),
    )  # fmt: skip
    triangles = [row for a, b, c, d in quads for row in ((a, b, c), (a, c, d))]
    x, y = origin
    points += [(x, y), (x + 0.5, y), (x + 0.5, y + 0.5), (x, y + 0.5)]
    triangles += [(16, 17, 18), (16, 18, 19)]
    return phx.discretization.CellMesh.from_triangles(
        np.asarray(points, dtype=np.float64), np.asarray(triangles)
    )


SPLIT_SQUARE_CELLS = np.asarray(((0, 1, 4), (0, 4, 5), (1, 2, 3), (1, 3, 4)))


def coverage(mesh: Any, regions: Any) -> Any:
    geometry, _, embedding_certificate = embedding(mesh)
    return phx.geometry.certify_domain_coverage(
        mesh, geometry, split_square_domain(), regions, embedding=embedding_certificate
    )


def mapped_reference_domain(kind: str, curved: bool = False) -> tuple[Any, Any, Any]:

    topology = reference_cell_topology(kind)
    vertices = np.asarray(topology.vertices, dtype=np.float64)
    if topology.dimension == 2:
        mesh = phx.discretization.CellMesh.from_polygons(
            vertices, (np.arange(vertices.shape[0], dtype=np.int32),)
        )
    else:
        mesh = phx.discretization.CellMesh.from_mixed_3d(
            vertices,
            (
                phx.discretization.CellBlock(
                    "mapped", kind, np.arange(vertices.shape[0], dtype=np.int32)[None]
                ),
            ),
            polyhedra={},
        )
    facets = [
        piece
        for face in topology.entities[topology.dimension - 1]
        for piece in (
            (face,)
            if len(face) <= 3
            else ((face[0], face[1], face[2]), (face[0], face[2], face[3]))
        )
    ]
    domain = phx.geometry.PiecewiseLinearDomain(
        vertices,
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray(((0, -1),), dtype=np.int64), (len(facets), 1)),
        ("material",),
        source_id=f"reference-{kind}",
    )
    element = coordinate_lagrange_element(kind, 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    if curved:
        nodes[:, 0] += 0.125 * np.prod(nodes * (1.0 - nodes), axis=1)
    name = mesh.blocks[0].name
    geometry = phx.discretization.CellGeometrySpec(
        {name: element}, {name: np.arange(nodes.shape[0], dtype=np.int64)[None]}, nodes
    )
    return mesh, geometry, domain


def cube_with_independent_boundary_triangulation(
    mapped: bool,
) -> tuple[Any, Any, Any]:

    _, _, domain = mapped_reference_domain("hexahedron")
    points = np.asarray(
        tuple(product((0.0, 0.5, 1.0), (0.0, 1.0), (0.0, 1.0))), dtype=np.float64
    )
    index = {tuple(point): row for row, point in enumerate(points)}
    reference_corners = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    scale = np.asarray((0.5, 1.0, 1.0), dtype=np.float64)
    cells = np.asarray(
        tuple(
            tuple(
                index[tuple(point)]
                for point in reference_corners * scale
                + np.asarray((origin, 0.0, 0.0), dtype=np.float64)
            )
            for origin in (0.0, 0.5)
        ),
        dtype=np.int32,
    )
    mesh = phx.discretization.CellMesh.from_mixed_3d(
        points,
        (phx.discretization.CellBlock("mapped", "hexahedron", cells),),
        polyhedra={},
    )
    if not mapped:
        return mesh, phx.discretization.CellGeometrySpec.affine(mesh), domain
    element = coordinate_lagrange_element("hexahedron", 2)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    local = np.stack(
        tuple(
            reference * scale + np.asarray((origin, 0.0, 0.0), dtype=np.float64)
            for origin in (0.0, 0.5)
        )
    )
    local[:, :, 0] += 0.125 * np.prod(reference * (1.0 - reference), axis=1)
    geometry = phx.discretization.CellGeometrySpec(
        {"mapped": element},
        {
            "mapped": np.arange(local.shape[0] * local.shape[1], dtype=np.int64).reshape(
                local.shape[:2]
            )
        },
        local.reshape(-1, 3),
    )
    return mesh, geometry, domain


def independent_mapped_shear_domain() -> tuple[Any, Any, Any]:

    source_mesh, _, reference_domain = mapped_reference_domain("hexahedron")
    source_element = coordinate_lagrange_element("hexahedron", 1)
    controls = np.asarray(source_element.reference_nodes, dtype=np.float64).copy()
    controls[:, 2] += 0.125 * controls[:, 0] * controls[:, 1]
    source_geometry = phx.discretization.CellGeometrySpec(
        {"mapped": source_element},
        {"mapped": np.arange(8, dtype=np.int64)[None]},
        controls,
    )
    domain = MappedReferenceDomain(
        reference_domain,
        source_mesh,
        source_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id="independent-dyadic-shear",
        source_revision="exact-source-controls",
    )
    reference_mesh, _, _ = cube_with_independent_boundary_triangulation(False)
    points = np.asarray(reference_mesh.coordinates, dtype=np.float64).copy()
    points[:, 2] += 0.125 * points[:, 0] * points[:, 1]
    cells = np.asarray(reference_mesh.blocks[0].vertices)
    blocks = tuple(
        phx.discretization.CellBlock(
            name,
            "hexahedron",
            cells[index : index + 1],
            global_ids=np.asarray((index,), dtype=np.int64),
        )
        for index, name in enumerate(("left", "right"))
    )
    mesh = phx.discretization.CellMesh(points, blocks)
    matrix = np.diag(np.asarray((0.5, 1.0, 1.0), dtype=np.float64))
    elements = {
        name: RestrictedCellGeometryElement(
            source_element,
            "hexahedron",
            matrix,
            np.asarray((0.5 * index, 0.0, 0.0), dtype=np.float64),
        )
        for index, name in enumerate(("left", "right"))
    }
    record = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        source_mesh.topology_id,
        {name: np.asarray((0,), dtype=np.int64) for name in elements},
        {name: np.arange(8, dtype=np.int64)[None] for name in elements},
    )
    geometry = phx.discretization.CellGeometrySpec(
        elements,
        {name: np.arange(8, dtype=np.int64)[None] for name in elements},
        controls,
        restriction_source=record,
    )
    return mesh, geometry, domain


class SampledSphere:
    """Protocol double reporting only sampled (uncertified) distance bounds."""

    source_id = "sampled-sphere"
    source_revision = "0"
    ambient_dimension = 3

    def boundary_distance(self, points: np.ndarray, /) -> Any:
        distance = np.abs(np.linalg.norm(points, axis=-1) - 1.0)
        return phx.geometry.SourceBoundaryDistance(distance, distance, "sampled")

    def boundary_samples(self, maximum_samples: int, /) -> Any:
        del maximum_samples
        points = np.eye(3)
        zero = np.zeros((3,))
        return phx.geometry.SourceBoundarySamples(
            points, zero, zero, "sampled", complete=True
        )


def octahedron() -> Any:
    points = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1.0))
    )
    faces = np.asarray(
        (
            (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
            (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
        )
    )  # fmt: skip
    return phx.discretization.CellMesh.from_triangles(points, faces)


def plc_audit(mesh: Any, **policy: Any) -> Any:
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    return geometry, phx.meshing.audit_cell_mesh(
        mesh, geometry, policy=phx.meshing.CellMeshAuditPolicy(**policy)
    )


class ParabolicTriangleSource:
    """Independent graph source with a proven Lipschitz source-point cover."""

    source_id = "parabolic-triangle"
    ambient_dimension = 3

    def __init__(self, curvature: float = 0.125, *, certified: bool = True) -> None:
        self.curvature = curvature
        self.certified = certified
        self.source_revision = f"parabolic-triangle-{curvature!r}"

    def graph(self, parameters: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (parameters, self.curvature * np.sum(parameters * parameters, axis=1))
        )

    def boundary_distance(self, points: np.ndarray, /) -> Any:
        # A valid source-point candidate gives an upper bound. |grad h| <= .25
        # on this chart gives the independent lower bound, including queries
        # whose rounded planar coordinates are slightly outside the triangle.
        parameters = np.clip(points[:, :2], 0.0, 1.0)
        excess = np.maximum(np.sum(parameters, axis=1) - 1.0, 0.0)
        parameters -= excess[:, None] / 2
        candidates = self.graph(parameters)
        planar_distance = np.linalg.norm(points[:, :2] - parameters, axis=1)
        residual = np.abs(points[:, 2] - candidates[:, 2])
        slack = 64 * np.finfo(np.float64).eps
        return phx.geometry.SourceBoundaryDistance(
            np.maximum(residual / 2 - planar_distance - slack, 0),
            np.linalg.norm(points - candidates, axis=1) + slack,
            "certified" if self.certified else "sampled",
        )

    def boundary_samples(self, maximum_samples: int, /) -> Any:
        parameters = np.asarray(
            [(i / 16, j / 16) for i in range(17) for j in range(17 - i)],
            dtype=np.float64,
        )
        count = parameters.shape[0]
        complete = count <= maximum_samples
        points = self.graph(parameters[:maximum_samples])
        # The planar lattice diameter is sqrt(2)/16, and graph Lipschitz
        # norm is at most sqrt(1 + 4*curvature**2) < 1.031 for these fixtures.
        return phx.geometry.SourceBoundarySamples(
            points,
            np.full((points.shape[0],), 0.1, dtype=np.float64),
            np.full((points.shape[0],), 64 * np.finfo(np.float64).eps, dtype=np.float64),
            "certified" if self.certified else "sampled",
            complete=complete,
        )


class ParabolicChartSource(ParabolicTriangleSource):
    def boundary_chart_cover(
        self, maximum_patches: int, /, *, budget: CoordinateEnclosureBudget
    ) -> Any:
        corners = self.graph(np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))))
        # Independent polynomial calculus: h-L = c*(x*x+y*y-x-y), whose
        # absolute maximum over this entire closed chart triangle is c/2.
        return phx.geometry.SourceBoundaryChartCover(
            corners[None],
            np.asarray((self.curvature / 2 + 1e-14,)),
            "certified",
            maximum_patches >= 1,
            self.source_id,
            self.source_revision,
        )


def parabolic_coordinate_mesh(
    source: ParabolicTriangleSource,
    *,
    bias: float = 0.0,
    bow: float = 0.0,
    degree: int = 2,
) -> tuple[Any, Any]:
    element = (
        phx.discretization.fem.lagrange_element("triangle", 2)
        if degree == 2
        else phx.discretization.fem.SimplexNodalFamily(
            "triangle", degree
        ).finite_element()
    )
    parameters = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinates = source.graph(parameters)
    coordinates[:, 2] += bias + bow * parameters[:, 0] * parameters[:, 1]
    mesh = phx.discretization.CellMesh.from_triangles(
        source.graph(np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"triangles": element},
        {"triangles": np.arange(parameters.shape[0], dtype=np.int64)[None]},
        coordinates,
    )
    return mesh, geometry


def curved_tetrahedral_components(
    scale: float, offset: float, degree: int
) -> tuple[Any, Any, Any]:

    element = coordinate_lagrange_element("tetrahedron", degree)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)

    def shear(points: np.ndarray) -> np.ndarray:
        result = points.copy()
        result[:, 0] += 0.0625 * points[:, 1] ** 2
        return result

    corners = np.concatenate((shear(TETRAHEDRON), shear(scale * TETRAHEDRON + offset)))
    points = np.concatenate((shear(reference), shear(scale * reference + offset)))
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        corners, np.asarray(((0, 1, 2, 3), (4, 5, 6, 7)), dtype=np.int32)
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"tetrahedra": element},
        {
            "tetrahedra": np.arange(points.shape[0], dtype=np.int32).reshape(
                (2, reference.shape[0])
            )
        },
        points,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    return mesh, geometry, validity


def prepared_tetrahedral_acceptance() -> tuple[Any, Any, Any, Any]:
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        TETRAHEDRON,
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
    )
    geometry, audit = plc_audit(
        mesh,
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
    )
    facets = np.asarray(
        phx.discretization.reference_cell_topology("tetrahedron").entities[2],
        dtype=np.int64,
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        TETRAHEDRON,
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (4, 1)),
        ("body",),
        source_id="prepared-independent-source",
    )
    prepared = phx.meshing.MeshCertificationPreparedEvidence(
        mesh,
        geometry,
        audit.validity,
        schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
    )
    return mesh, geometry, audit, prepared


def prepared_publication_authority(
    reference_budget: CoordinateEnclosureBudget,
) -> tuple[Any, Any, Any, Any]:

    mesh, geometry, _, initial = prepared_tetrahedral_acceptance()
    domain = initial.request.domain
    revision = "authored-proof-source"
    request = NativeCertificationRequest(
        initial.request.schedule,
        domain.source_id,
        revision,
        phx.meshing.MeshingLimits(),
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
    )
    embedding = phx.geometry.certify_global_embedding(
        mesh,
        geometry,
        initial.validity,
        limits=request.certificate_limits,
    )
    transfer = PlcAssociationTransfer(
        domain,
        phx.SpatialCoordinateContract.si(),
        revision,
        edge_vertices=np.asarray(
            phx.discretization.reference_cell_topology("tetrahedron").entities[1],
            dtype=np.int64,
        ),
        triangle_vertices=domain.facets,
        triangle_facets=np.arange(domain.facets.shape[0], dtype=np.int64),
        facet_regions=domain.facet_regions,
        maximum_support_queries=2_000_000,
    )
    declarations = []
    for dimension, role, indices in (
        (0, phx.meshing.GeometrySourceEntityRole.VERTEX, np.arange(4, dtype=np.int64)),
        (
            3,
            phx.meshing.GeometrySourceEntityRole.REGION,
            np.asarray((0,), dtype=np.int64),
        ),
    ):
        entities = mesh.entity_set(dimension)
        declarations.append(
            phx.meshing.GeometryAssociation(
                phx.meshing.GeometryAssociationKind.PIECEWISE_LINEAR,
                domain.source_id,
                revision,
                entities.entity_set_id,
                entities.entity_ids,
                tuple(
                    entity_key(revision, role.value, index) for index in indices.tolist()
                ),
                np.zeros(indices.shape, dtype=np.float64),
                source_dimensions=np.full(indices.shape, dimension, dtype=np.int64),
                source_indices=indices,
                source_entity_roles=(role,) * indices.size,
            )
        )
    proof = certify_mapped_plc_associations(
        transfer,
        mesh,
        geometry,
        tuple(declarations),
        embedding=embedding,
        validity=initial.validity,
        certificate_limits=request.certificate_limits,
        budget=reference_budget,
    )
    return mesh, geometry, request, proof


def affine_tetrahedral_root_restrictions(
    second: np.ndarray,
) -> tuple[phx.discretization.CellMesh, phx.discretization.CellGeometrySpec]:

    source = phx.discretization.CellMesh.from_tetrahedra(
        np.concatenate((TETRAHEDRON, second)),
        np.asarray(((0, 1, 2, 3), (4, 5, 6, 7)), dtype=np.int64),
        cell_global_ids=np.asarray((101, 205), dtype=np.int64),
    )
    source_geometry = phx.discretization.CellGeometrySpec.affine(source)
    elements, routes, controls = source_geometry.resolve(source)
    cells = np.asarray(source.blocks[0].vertices, dtype=np.int64)
    points = np.asarray(source.coordinates, dtype=np.float64)[cells]
    points = points[:, :1] + (points - points[:, :1]) * 0.125
    target = phx.discretization.CellMesh.from_tetrahedra(
        points.reshape((-1, 3)),
        np.arange(8, dtype=np.int64).reshape((2, 4)),
        cell_global_ids=np.asarray((701, 907), dtype=np.int64),
    )
    name = target.blocks[0].name
    record = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        source.topology_id,
        {name: np.asarray(source.blocks[0].global_ids, dtype=np.int64)},
        {name: np.asarray(source.vertex_global_ids, dtype=np.int64)[cells]},
    )
    geometry = phx.discretization.CellGeometrySpec(
        {
            name: RestrictedCellGeometryElement(
                require_scalar_coordinate_element(elements[0], "Affine root regression"),
                "tetrahedron",
                np.eye(3, dtype=np.float64) * 0.125,
                np.zeros((3,), dtype=np.float64),
            )
        },
        {name: routes[0]},
        controls,
        restriction_source=record,
    )
    return target, geometry


def bumped_square(first: Any, second: Any) -> Any:
    """Cubic triangles ``(0, 1, 2)`` and ``(2, 3, 0)`` traversing their diagonal oppositely.

    ``first``/``second`` give the physical bump amplitude along the diagonal
    normal as a function of the diagonal parameter ``t`` from vertex 0 to 2.
    """

    corners = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    cells = np.asarray(((0, 1, 2), (2, 3, 0)), dtype=np.int32)
    # The rational-lattice coordinate element: its k/3 reference nodes give both
    # cells the same diagonal chart, so only authored bumps separate traces.
    element = coordinate_lagrange_element("triangle", 3)
    reference = [
        tuple(Fraction(float(value)).limit_denominator(3) for value in node)
        for node in np.asarray(element.reference_nodes, dtype=np.float64)
    ]
    normal = np.asarray((1.0, -1.0)) / math.sqrt(2.0)
    nodes = []
    for row, ((a, b, c), bump) in enumerate(zip(cells, (first, second), strict=True)):
        for u, v in reference:
            exact = tuple(
                Fraction(corners[a][axis])
                + u * Fraction(corners[b][axis] - corners[a][axis])
                + v * Fraction(corners[c][axis] - corners[a][axis])
                for axis in range(2)
            )
            # The diagonal is the local 0 -> 2 edge of the first cell and 2 -> 0 of the second.
            parameter = v if row == 0 else 1 - v
            displacement = bump(np.asarray([float(parameter)]))[0] * normal
            nodes.append(np.asarray([float(value) for value in exact]) + displacement)
    mesh = phx.discretization.CellMesh.from_triangles(corners, cells)
    geometry = phx.discretization.CellGeometrySpec(
        {"triangles": element},
        {
            "triangles": np.arange(2 * len(reference), dtype=np.int32).reshape(
                (2, len(reference))
            )
        },
        np.asarray(nodes),
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    return phx.geometry.certify_global_embedding(mesh, geometry, validity)


def straight(t: np.ndarray) -> np.ndarray:
    return np.zeros_like(t)


def antisymmetric(t: np.ndarray) -> np.ndarray:
    # Vanishes at both endpoints and the midpoint: three samples cannot see it.
    return 0.1 * t * (1.0 - t) * (1.0 - 2.0 * t)


def symmetric(t: np.ndarray) -> np.ndarray:
    return 0.05 * t * (1.0 - t)


def tapered_hexes(top: float) -> Any:
    """Two planar-faced trilinear hexes whose top face narrows by ``top`` in y."""

    logical = np.asarray(tuple(np.ndindex((3, 2, 2))), dtype=np.float64)
    points = logical.copy()
    points[:, 1] = 0.5 + (1.0 + (top - 1.0) * logical[:, 2]) * (logical[:, 1] - 0.5)
    index = {tuple(point): row for row, point in enumerate(logical.tolist())}
    corners = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    cells = np.asarray(
        [
            [
                index[tuple((corner + np.asarray((origin, 0.0, 0.0))).tolist())]
                for corner in corners
            ]
            for origin in (0.0, 1.0)
        ],
        dtype=np.int32,
    )
    return phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("hex", "hexahedron", cells),)
    )


def rotated_neighbor_hexes() -> Any:
    """Tapered trilinear hexes sharing ``x = 1``; the second sees that face through a quarter turn."""

    logical = np.asarray(tuple(np.ndindex((3, 2, 2))), dtype=np.float64)
    index = {tuple(point): row for row, point in enumerate(logical.tolist())}
    corners = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    # A proper quarter turn about the x axis: (u, v, w) -> (1 + u, 1 - w, v).
    turned = np.stack((corners[:, 0] + 1.0, 1.0 - corners[:, 2], corners[:, 1]), axis=1)
    cells = np.asarray(
        [
            [index[tuple(corner.tolist())] for corner in placed]
            for placed in (corners, turned)
        ],
        dtype=np.int32,
    )
    # Planar faces with a non-affine map, so the mapped proof owns both cells.
    points = logical.copy()
    points[:, 1] = 0.5 + (1.0 - 0.5 * logical[:, 2]) * (logical[:, 1] - 0.5)
    return phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("hex", "hexahedron", cells),)
    )


def mixed_degree_hexes(bulge: float) -> tuple[Any, Any]:
    """A trilinear hex beside a triquadratic hex whose shared face center moves by ``bulge``."""

    logical = np.asarray(tuple(np.ndindex((3, 2, 2))), dtype=np.float64)
    index = {tuple(point): row for row, point in enumerate(logical.tolist())}
    corners = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    cells = np.asarray(
        [
            [
                index[tuple((corner + np.asarray((origin, 0.0, 0.0))).tolist())]
                for corner in corners
            ]
            for origin in (0.0, 1.0)
        ],
        dtype=np.int32,
    )
    names = ("linear", "quadratic")
    blocks = tuple(
        phx.discretization.CellBlock(
            name,
            "hexahedron",
            cells[row : row + 1],
            global_ids=np.asarray((row,), dtype=np.int64),
        )
        for row, name in enumerate(names)
    )
    mesh = phx.discretization.CellMesh(logical, blocks)
    elements = (
        coordinate_lagrange_element("hexahedron", 1),
        coordinate_lagrange_element("hexahedron", 2),
    )
    first = np.asarray(elements[0].reference_nodes, dtype=np.float64)
    second = np.asarray(elements[1].reference_nodes, dtype=np.float64) + np.asarray(
        (1.0, 0.0, 0.0)
    )
    second[np.all(np.isclose(second, (1.0, 0.5, 0.5)), axis=1), 0] += bulge
    geometry = phx.discretization.CellGeometrySpec(
        dict(zip(names, elements, strict=True)),
        {
            "linear": np.arange(first.shape[0], dtype=np.int32)[None],
            "quadratic": first.shape[0]
            + np.arange(second.shape[0], dtype=np.int32)[None],
        },
        np.concatenate((first, second)),
    )
    return mesh, geometry


def halved_tapered_restriction() -> tuple[Any, Any]:
    """Two tapered trilinear roots, each restricted to two rational half-x charts."""

    def taper(local: np.ndarray) -> np.ndarray:
        tapered = local.copy()
        tapered[:, 1] = 0.5 + (1.0 - 0.5 * local[:, 2]) * (local[:, 1] - 0.5)
        return tapered

    roots = tapered_hexes(0.5)
    element = coordinate_lagrange_element("hexahedron", 1)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    routes = np.arange(2 * nodes.shape[0], dtype=np.int64).reshape((2, nodes.shape[0]))
    source = phx.discretization.CellGeometrySpec(
        {"hex": element},
        {"hex": routes},
        np.concatenate(
            [taper(nodes + np.asarray((origin, 0.0, 0.0))) for origin in (0.0, 1.0)]
        ),
    )
    # Each root is halved in x; the children are affine source restrictions.
    logical = np.asarray(tuple(np.ndindex((5, 2, 2))), dtype=np.float64) * np.asarray(
        (0.5, 1.0, 1.0)
    )
    points = taper(logical)
    index = {tuple(point): row for row, point in enumerate(logical.tolist())}
    corners = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    root_vertices = np.asarray(roots.vertex_global_ids)[
        np.asarray(roots.blocks[0].vertices)
    ]
    blocks, elements, dofs, parents, parent_vertices = [], {}, {}, {}, {}
    for half, name in enumerate(("lower", "upper")):
        cells = np.asarray(
            [
                [
                    index[
                        tuple(
                            (
                                corner * np.asarray((0.5, 1.0, 1.0))
                                + np.asarray((origin + 0.5 * half, 0.0, 0.0))
                            ).tolist()
                        )
                    ]
                    for corner in corners
                ]
                for origin in (0.0, 1.0)
            ],
            dtype=np.int32,
        )
        blocks.append(
            phx.discretization.CellBlock(
                name,
                "hexahedron",
                cells,
                global_ids=np.asarray((half, 2 + half), dtype=np.int64),
            )
        )
        elements[name] = RestrictedCellGeometryElement(
            element,
            "hexahedron",
            np.diag((0.5, 1.0, 1.0)),
            np.asarray((0.5 * half, 0.0, 0.0)),
        )
        dofs[name], parents[name], parent_vertices[name] = (
            routes,
            np.asarray(roots.blocks[0].global_ids),
            root_vertices,
        )
    mesh = phx.discretization.CellMesh(points, tuple(blocks))
    record = CellGeometryRestrictionSource(
        cell_geometry_id(source), roots.topology_id, parents, parent_vertices
    )
    return mesh, phx.discretization.CellGeometrySpec(
        elements, dofs, source.coordinates, restriction_source=record
    )


def split_square_variant(vertices: Any, *, reversed_facets: bool, source_id: str) -> Any:
    """The split-square facets over ``vertices``, optionally with every facet reoriented."""
    declared = split_square_domain()
    facets, regions = np.asarray(declared.facets), np.asarray(declared.facet_regions)
    if reversed_facets:
        # Reversing a facet flips its normal; swapping its regions keeps the same oriented interface.
        facets, regions = facets[:, ::-1].copy(), regions[:, ::-1].copy()
    return phx.geometry.PiecewiseLinearDomain(
        vertices, facets, regions, declared.region_ids, source_id=source_id
    )


def curved_shared_hex_interface() -> tuple[Any, Any]:

    mesh = tapered_hexes(1.0)
    element = coordinate_lagrange_element("hexahedron", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    bubble = nodes[:, 1] * (1.0 - nodes[:, 1]) * nodes[:, 2] * (1.0 - nodes[:, 2])
    first, second = nodes.copy(), nodes + np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    first[:, 0] += 0.125 * nodes[:, 0] * bubble
    second[:, 0] += 0.125 * (1.0 - nodes[:, 0]) * bubble
    name = mesh.blocks[0].name
    geometry = phx.discretization.CellGeometrySpec(
        {name: element},
        {name: np.arange(2 * nodes.shape[0], dtype=np.int64).reshape((2, -1))},
        np.concatenate((first, second)),
    )
    return mesh, geometry
