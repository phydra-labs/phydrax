#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import phydrax as phx


_TETRAHEDRON = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)


def _embedding(mesh: Any) -> Any:
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    return (
        geometry,
        validity,
        phx.geometry.certify_global_embedding(mesh, geometry, validity),
    )


def _two_tetrahedra(second: np.ndarray) -> Any:
    return phx.discretization.CellMesh.from_tetrahedra(
        np.concatenate((_TETRAHEDRON, second)),
        np.asarray(((0, 1, 2, 3), (4, 5, 6, 7))),
    )


def _checks(certificate: Any, status: str) -> set[str]:
    return {value.check for value in certificate.findings if value.status == status}


@pytest.mark.parametrize(
    ("points", "cells", "status"),
    (
        pytest.param(
            ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 0.5), (1.0, -1.0, 0.5)),
            ((0, 1), (2, 3)),
            "certified",
            id="skew-overlapping-boxes",
        ),
        pytest.param(
            ((-1.0, -1.0, -2.0), (1.0, 1.0, 2.0), (-1.0, 1.0, 0.0), (1.0, -1.0, 0.0)),
            ((0, 1), (2, 3)),
            "violated",
            id="tilted-plane-crossing",
        ),
        pytest.param(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            ((0, 1), (1, 2)),
            "certified",
            id="shared-axis-endpoint",
        ),
        pytest.param(
            ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0)),
            ((0, 1), (1, 2)),
            "certified",
            id="shared-bent-endpoint",
        ),
        pytest.param(
            ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0, 1), (1, 2)),
            "violated",
            id="shared-axis-overlap",
        ),
        pytest.param(
            ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
            ((0, 1), (2, 3)),
            "violated",
            id="disconnected-axis-overlap",
        ),
        pytest.param(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ((0, 1), (2, 3)),
            "violated",
            id="unshared-endpoint",
        ),
    ),
)
def test_spatial_interval_embedding_uses_exact_contacts(
    points: tuple[tuple[float, ...], ...],
    cells: tuple[tuple[int, int], ...],
    status: str,
) -> None:
    mesh = phx.discretization.CellMesh(
        np.asarray(points, dtype=np.float64),
        (
            phx.discretization.CellBlock(
                "segments", "interval", np.asarray(cells, dtype=np.int32)
            ),
        ),
    )
    _, validity, certificate = _embedding(mesh)

    assert validity.all_certified
    assert certificate.status == status
    assert "cell_contact" in certificate.evaluated_checks
    if status == "violated":
        assert "cell_contact" in _checks(certificate, "violated")
    else:
        assert certificate.findings == ()


def test_spatial_interval_embedding_preserves_candidate_capacity_refusal() -> None:
    mesh = phx.discretization.CellMesh(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0), (3.0, 1.0, 0.0)),
            dtype=np.float64,
        ),
        (
            phx.discretization.CellBlock(
                "segments",
                "interval",
                np.asarray(((0, 1), (1, 2), (2, 3)), dtype=np.int32),
            ),
        ),
    )
    geometry, validity, _ = _embedding(mesh)
    certificate = phx.geometry.certify_global_embedding(
        mesh,
        geometry,
        validity,
        limits=phx.geometry.MeshCertificateLimits(maximum_candidate_pairs=1),
    )

    assert certificate.status == "unresolved"
    assert "cell_contact_capacity" in _checks(certificate, "unresolved")


@pytest.mark.parametrize(
    "mesh",
    [
        pytest.param(
            phx.discretization.CellMesh.from_tetrahedra(
                np.concatenate((_TETRAHEDRON, ((1.0, 1.0, 1.0),))),
                np.asarray(((0, 1, 2, 3), (1, 2, 3, 4))),
            ),
            id="shared-face",
        ),
        pytest.param(_two_tetrahedra(_TETRAHEDRON + 5.0), id="disjoint-components"),
    ],
)
def test_embedded_tetrahedral_meshes_are_certified(mesh: Any) -> None:
    _, validity, certificate = _embedding(mesh)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert certificate.findings == ()
    assert "exterior_degree" in certificate.evaluated_checks
    assert certificate.binding.topology_id == mesh.topology_id


def test_positive_jacobian_overlapping_components_are_not_embedded() -> None:
    mesh = _two_tetrahedra(_TETRAHEDRON + 0.25)
    _, validity, certificate = _embedding(mesh)

    # Every cell is positively oriented; only the global check sees the overlap.
    assert validity.all_certified
    assert certificate.status == "violated"
    assert "boundary_contact" in _checks(certificate, "violated")
    contact = next(v for v in certificate.findings if v.check == "boundary_contact")
    assert contact.entity_kind == "facet"
    assert contact.entity_ids


def test_contained_component_without_boundary_contact_is_detected() -> None:
    mesh = _two_tetrahedra(0.1 * _TETRAHEDRON + 0.2)
    _, validity, certificate = _embedding(mesh)
    audit = phx.meshing.audit_cell_mesh(
        mesh, phx.discretization.CellGeometrySpec.affine(mesh)
    )

    # The corner audit (boundary self-intersection) cannot see a contained shell.
    assert audit.passed
    assert validity.all_certified
    assert certificate.status == "violated"
    assert _checks(certificate, "violated") == {"nested_component"}


def _annulus_with_island(origin: tuple[float, float]) -> Any:
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


@pytest.mark.parametrize(
    ("origin", "status"),
    [
        pytest.param((1.25, 1.25), "certified", id="island-in-cavity"),
        pytest.param((0.25, 0.25), "violated", id="island-inside-material"),
    ],
)
def test_nested_planar_components_follow_the_exterior_degree(
    origin: tuple[float, float], status: str
) -> None:
    _, _, certificate = _embedding(_annulus_with_island(origin))

    assert certificate.status == status
    assert certificate.shell_count == 3
    if status == "violated":
        assert _checks(certificate, "violated") == {"nested_component"}


def test_continuous_curved_triangle_embedding_is_certified() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))), np.asarray(((0, 1, 2),))
    )
    element = phx.discretization.fem.lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    nodes[:, 0] += 0.0625 * nodes[:, 0] * nodes[:, 1]
    geometry = phx.discretization.CellGeometrySpec(
        {"triangles": element}, {"triangles": np.arange(nodes.shape[0])[None]}, nodes
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert "mapped_local_injectivity" in certificate.evaluated_checks
    assert certificate.findings == ()
    assert certificate.source_expression_work_units > 1
    assert certificate.source_expression_peak_bytes > 1
    for limits in (
        phx.geometry.MeshCertificateLimits(maximum_work_units=1),
        phx.geometry.MeshCertificateLimits(maximum_scratch_bytes=1),
    ):
        capped = phx.geometry.certify_global_embedding(
            mesh, geometry, validity, limits=limits
        )
        assert capped.status == "unresolved"
        assert "mapped_source_expression_resource_budget" in _checks(capped, "unresolved")
        assert capped.source_expression_work_units <= limits.maximum_work_units
        assert capped.source_expression_peak_bytes <= limits.maximum_scratch_bytes


def test_embedding_refuses_a_validity_certificate_of_other_coordinates() -> None:
    mesh = _two_tetrahedra(_TETRAHEDRON + 5.0)
    moved = mesh.with_coordinates(mesh.coordinates + 1.0, numeric_version="moved")
    validity = phx.discretization.certify_cell_geometry_validity(mesh)

    with pytest.raises(ValueError, match="not bound"):
        phx.geometry.certify_global_embedding(
            moved, phx.discretization.CellGeometrySpec.affine(moved), validity
        )


_SQUARE = np.asarray(((0, 0), (0.5, 0), (1, 0), (1, 1), (0.5, 1), (0, 1.0)))
_CELLS = np.asarray(((0, 1, 4), (0, 4, 5), (1, 2, 3), (1, 3, 4)))


def _split_square_domain() -> Any:
    return phx.geometry.PiecewiseLinearDomain(
        _SQUARE,
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4))),
        np.asarray(((0, -1), (1, -1), (1, -1), (1, -1), (0, -1), (0, -1), (0, 1))),
        ("left", "right"),
        source_id="split-square",
    )


def _coverage(mesh: Any, regions: Any) -> Any:
    geometry, _, embedding = _embedding(mesh)
    return phx.geometry.certify_domain_coverage(
        mesh, geometry, _split_square_domain(), regions, embedding=embedding
    )


def test_conforming_two_region_mesh_covers_domain_and_interface_exactly() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(_SQUARE, _CELLS)
    certificate = _coverage(mesh, np.asarray((0, 0, 1, 1)))

    assert certificate.status == "certified"
    assert certificate.requested_region_measures == (0.5, 0.5)
    assert certificate.achieved_region_measures == (0.5, 0.5)
    assert certificate.covered_source_facet_count == certificate.source_facet_count


def test_omitted_material_interface_is_a_coverage_violation() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(_SQUARE, _CELLS)
    certificate = _coverage(mesh, np.asarray((0, 0, 0, 0)))

    violated = _checks(certificate, "violated")
    assert certificate.status == "violated"
    assert {"omitted_interface", "region_measure"} <= violated
    interface = next(v for v in certificate.findings if v.check == "omitted_interface")
    assert interface.entity_kind == "source_facet"
    assert interface.entity_ids == (6,)
    assert certificate.achieved_region_measures == (1.0, 0.0)


def test_gap_leaves_unmatched_facets_and_a_measure_deficit() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(_SQUARE, _CELLS[:3])
    certificate = _coverage(mesh, np.asarray((0, 0, 1)))

    assert certificate.status == "violated"
    assert {"unmatched_boundary_facet", "uncovered_boundary", "region_measure"} <= (
        _checks(certificate, "violated")
    )
    assert certificate.achieved_region_measures == (0.5, 0.25)


def test_double_coverage_is_detected_by_embedding_and_measure() -> None:
    points = np.concatenate((_SQUARE, ((0.1, 0.1), (0.4, 0.1), (0.1, 0.4))))
    cells = np.concatenate((_CELLS, ((6, 7, 8),)))
    mesh = phx.discretization.CellMesh.from_triangles(points, cells)
    geometry, _, embedding = _embedding(mesh)
    certificate = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        _split_square_domain(),
        np.asarray((0, 0, 1, 1, 0)),
        embedding=embedding,
    )

    assert embedding.status == "violated"
    assert certificate.status == "violated"
    assert "embedding_premise" in _checks(certificate, "unresolved")
    assert certificate.achieved_region_measures[0] is not None
    assert (
        certificate.achieved_region_measures[0] > certificate.requested_region_measures[0]
    )


def _mapped_reference_domain(kind: str, curved: bool = False) -> tuple[Any, Any, Any]:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._reference_cell import reference_cell_topology

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


@pytest.mark.parametrize("kind", ("tetrahedron", "prism", "hexahedron"))
@pytest.mark.parametrize("bowed_face", (False, True))
def test_mapped_mixed_root_associations_classify_full_polynomial_strata(
    kind: str,
    bowed_face: bool,
) -> None:
    from phydrax.discretization._cell_geometry import (
        CellGeometryRestrictionSource,
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain

    mesh, source_geometry, reference_domain = _mapped_reference_domain(kind)
    domain = MappedReferenceDomain(
        reference_domain,
        mesh,
        source_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id=f"mapped-{kind}",
        source_revision="independent-mixed-root",
    )
    root = source_geometry.elements[0]
    chart = coordinate_lagrange_element(kind, 2)
    nodes = np.asarray(chart.reference_nodes, dtype=np.float64)
    controls = nodes.copy()
    if bowed_face:
        remaining = 1.0 - (
            np.sum(nodes, axis=1)
            if kind == "tetrahedron"
            else np.sum(nodes[:, :2], axis=1)
        )
        controls[:, 1] += (
            nodes[:, 0] * (1.0 - nodes[:, 0]) * (1.0 - nodes[:, 1])
            if kind == "hexahedron"
            else nodes[:, 0] * remaining
        ) / 8
    elif kind == "hexahedron":
        controls[:, 0] += nodes[:, 0] * (1.0 - nodes[:, 0]) / 8
    else:
        displacement = nodes[:, 0] * nodes[:, 1] / 8
        controls[:, 0] += displacement
        controls[:, 1] -= displacement
    exact = tuple(tuple(Fraction(float(value)) for value in row) for row in controls)
    element = PolynomialComposedCellGeometryElement(
        root,
        chart,
        np.asarray([[value.numerator for value in row] for row in exact], dtype=np.int64),
        np.asarray(
            [[value.denominator for value in row] for row in exact], dtype=np.uint64
        ),
    )
    block = mesh.blocks[0]
    record = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        mesh.topology_id,
        {block.name: np.asarray(block.global_ids, dtype=np.int64)},
        {block.name: np.asarray(mesh.vertex_global_ids, dtype=np.int64)[block.vertices]},
    )
    geometry = phx.discretization.CellGeometrySpec(
        {block.name: element},
        {block.name: source_geometry.geometry_dofs[0]},
        source_geometry.coordinates,
        restriction_source=record,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    transfer = phx.meshing.MappedReferenceAssociationTransfer(domain)
    associations = transfer.associations(mesh, geometry)

    assert validity.all_certified
    assert len(associations) == 4
    for degree, association in enumerate(associations):
        assert association.exact and association.complete
        assert association.source_id == domain.source_id
        assert association.source_revision == domain.source_revision
        assert association.target_entity_set_id == mesh.entity_set(degree).entity_set_id
        if not bowed_face or degree in (0, 3):
            assert np.all(np.asarray(association.parent_dimensions) == degree)
            np.testing.assert_array_equal(
                association.parent_ids, mesh.entity_set(degree).entity_ids
            )
    if bowed_face:
        # Every carrier corner stays on its original stratum, but a whole
        # polynomial face bows into the root interior and must name the cell.
        assert np.any(np.asarray(associations[2].parent_dimensions) == 3)
    topology = phx.discretization.reference_cell_topology(kind)
    capped = phx.meshing.MappedReferenceAssociationTransfer(
        domain,
        maximum_support_queries=sum(len(rows) for rows in topology.entities) - 1,
    )
    with pytest.raises(phx.meshing.MeshingFailure, match="support budget"):
        capped.associations(mesh, geometry)


@pytest.mark.parametrize(
    ("kind", "measure"),
    [
        pytest.param("triangle", 0.5, id="quadratic-triangle"),
        pytest.param("quadrilateral", 1.0, id="quadratic-quadrilateral"),
        pytest.param("tetrahedron", 1.0 / 6.0, id="quadratic-tetrahedron"),
        pytest.param("hexahedron", 1.0, id="quadratic-hexahedron"),
        pytest.param("prism", 0.5, id="quadratic-prism"),
        pytest.param("pyramid", 1.0 / 3.0, id="rational-quadratic-pyramid"),
    ],
)
def test_mapped_standard_cells_cover_their_declared_domain(
    kind: str, measure: float
) -> None:
    from fractions import Fraction

    mesh, geometry, domain = _mapped_reference_domain(kind)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0,), dtype=np.int64), embedding=embedding
    )

    assert certificate.status == "certified"
    assert certificate.achieved_region_measures == (measure,)
    assert certificate.requested_region_measures == (measure,)
    assert certificate.integration_error_bounds == (0.0,)
    assert certificate.covered_source_facet_count == len(domain.facets)
    bounds = certificate.achieved_region_measure_bounds[0]
    assert bounds is not None
    lower, upper = bounds
    exact = {"tetrahedron": Fraction(1, 6), "pyramid": Fraction(1, 3)}.get(
        kind, Fraction(measure)
    )
    assert Fraction(lower) <= exact <= Fraction(upper)
    assert certificate.binding.source_revision == domain.source_revision


def test_curved_volume_with_fixed_planar_boundary_has_exact_coverage() -> None:
    mesh, geometry, domain = _mapped_reference_domain("hexahedron", curved=True)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0,), dtype=np.int64), embedding=embedding
    )

    assert validity.all_certified
    assert embedding.status == "certified"
    assert certificate.status == "certified"
    assert certificate.achieved_region_measures == (1.0,)
    assert certificate.achieved_region_measure_bounds == ((1.0, 1.0),)
    assert certificate.integration_error_bounds == (0.0,)


def _mapped_split_square(
    cells: int = 2, right_bulge: float = 0.0
) -> tuple[Any, Any, Any]:
    """Q2 split square; ``right_bulge`` moves the outer right edge midpoint outward."""
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    loops = (
        np.asarray((0, 1, 4, 5), dtype=np.int32),
        np.asarray((1, 2, 3, 4), dtype=np.int32),
    )[:cells]
    mesh = phx.discretization.CellMesh.from_polygons(_SQUARE, loops)
    element = coordinate_lagrange_element("quadrilateral", 2)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    local = np.stack(
        tuple(
            reference * np.asarray((0.5, 1.0), dtype=np.float64)
            + np.asarray((0.5 * index, 0.0), dtype=np.float64)
            for index in range(cells)
        )
    )
    local[:, :, 0] += 0.125 * np.prod(reference * (1.0 - reference), axis=1)
    local[-1, np.all(reference == (1.0, 0.5), axis=1), 0] += right_bulge
    name = mesh.blocks[0].name
    geometry = phx.discretization.CellGeometrySpec(
        {name: element},
        {
            name: np.arange(local.shape[0] * local.shape[1], dtype=np.int64).reshape(
                local.shape[:2]
            )
        },
        local.reshape(-1, 2),
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    return mesh, geometry, embedding


def test_curved_two_region_interiors_cover_planar_material_interface() -> None:
    mesh, geometry, embedding = _mapped_split_square()
    certificate = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        _split_square_domain(),
        np.asarray((0, 1), dtype=np.int64),
        embedding=embedding,
    )

    assert certificate.status == "certified"
    assert certificate.achieved_region_measures == (0.5, 0.5)
    assert certificate.integration_error_bounds == (0.0, 0.0)
    assert certificate.covered_source_facet_count == 7


def test_mapped_coverage_reports_and_caps_its_actual_subdivision_work() -> None:
    mesh, geometry, embedding = _mapped_split_square()
    domain = _split_square_domain()
    regions = np.asarray((0, 1), dtype=np.int64)
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding
    )

    assert certificate.status == "certified"
    assert certificate.subdivision_piece_count > 0
    assert (
        0
        <= certificate.maximum_subdivision_depth_reached
        <= phx.geometry.MeshCertificateLimits().maximum_subdivision_depth
    )

    limits = phx.geometry.MeshCertificateLimits(
        maximum_subdivision_pieces=certificate.subdivision_piece_count - 1
    )
    capped = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding, limits=limits
    )

    assert capped.status == "unresolved"
    assert "mapped_facet_containment" in _checks(capped, "unresolved")
    # The refused over-budget piece is charged, not hidden under the cap.
    assert capped.subdivision_piece_count > limits.maximum_subdivision_pieces


@pytest.mark.parametrize("positive", (True, False))
def test_complete_tetrahedral_polynomial_support_subdivides_all_reference_axes(
    positive: bool,
) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.geometry._mapped_coverage import nonnegative, SubdivisionLedger

    # The z dependence cannot be projected onto a triangle or replaced by
    # corner values: both signs have nonnegative original corner samples.
    margin = Fraction(1 if positive else -1, 64)
    work = SubdivisionLedger()
    ledger = algebra.CoordinateEnclosureBudget(100_000, 8 * 1024**2)
    with ledger.activate():
        z = algebra.axes(3)[2]
        centered = algebra.add(z, algebra.constant(Fraction(-1, 2), 3))
        polynomial = algebra.add(
            algebra.multiply(centered, centered),
            algebra.constant(margin, 3),
        )
        assert nonnegative(polynomial, "simplex", 3, 64, 4, 1_000, work) is positive
    assert work.pieces > 1
    assert work.maximum_depth > 0
    assert ledger.work_units > 0


def test_mapped_coverage_charges_its_expression_algebra_to_the_request_ledger() -> None:
    mesh, geometry, embedding = _mapped_split_square()
    domain = _split_square_domain()
    regions = np.asarray((0, 1), dtype=np.int64)
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding
    )
    required = certificate.source_expression_work_units

    assert certificate.status == "certified"
    assert required is not None and required > 0
    assert (
        certificate.source_expression_peak_bytes is not None
        and certificate.source_expression_peak_bytes > 0
    )
    for _ in range(3):
        renewed = phx.geometry.certify_domain_coverage(
            mesh, geometry, domain, regions, embedding=embedding
        )
        assert renewed.status == "certified"
        assert renewed.source_expression_work_units == required
        assert (
            renewed.source_expression_peak_bytes
            == certificate.source_expression_peak_bytes
        )
        assert renewed.certificate_id == certificate.certificate_id
    limits = phx.geometry.MeshCertificateLimits(maximum_work_units=required - 1)
    capped = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding, limits=limits
    )

    assert capped.status == "unresolved"
    assert "mapped_coverage_resource_budget" in _checks(capped, "unresolved")
    assert capped.source_expression_work_units is not None
    assert capped.source_expression_work_units <= limits.maximum_work_units
    assert capped.achieved_region_measures == (None, None)


@pytest.mark.parametrize("reversed_facets", (False, True))
def test_source_facet_partition_consumes_its_coverage_budget(
    reversed_facets: bool,
) -> None:
    """Independent source refinement preserves coverage but is not free proof work."""
    mesh, geometry, domain = _mapped_reference_domain("hexahedron")
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    regions = np.asarray((0,), dtype=np.int64)
    baseline = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding
    )
    points = list(np.asarray(domain.vertices, dtype=np.float64))
    midpoints: dict[tuple[int, int], int] = {}
    facets: list[tuple[int, int, int]] = []
    for a, b, c in np.asarray(domain.facets, dtype=np.int64).tolist():
        middle: list[int] = []
        for first, second in ((a, b), (b, c), (c, a)):
            key = (min(first, second), max(first, second))
            if key not in midpoints:
                midpoints[key] = len(points)
                points.append((points[first] + points[second]) / 2)
            middle.append(midpoints[key])
        ab, bc, ca = middle
        facets.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    partition_facets = np.asarray(facets, dtype=np.int64)
    partition_regions = np.repeat(domain.facet_regions, 4, axis=0)
    if reversed_facets:
        partition_facets = partition_facets[:, ::-1].copy()
        partition_regions = partition_regions[:, ::-1].copy()
    partition = phx.geometry.PiecewiseLinearDomain(
        np.asarray(points, dtype=np.float64),
        partition_facets,
        partition_regions,
        domain.region_ids,
        source_id=domain.source_id,
    )
    assert partition.source_revision != domain.source_revision
    complete = phx.geometry.certify_domain_coverage(
        mesh, geometry, partition, regions, embedding=embedding
    )
    assert baseline.status == complete.status == "certified"
    assert complete.achieved_region_measures == baseline.achieved_region_measures
    assert complete.covered_source_facet_count == len(facets)
    original = baseline.source_expression_work_units
    required = complete.source_expression_work_units
    assert original is not None and required is not None and required > original
    limits = phx.geometry.MeshCertificateLimits(maximum_work_units=original)
    capped = phx.geometry.certify_domain_coverage(
        mesh, geometry, partition, regions, embedding=embedding, limits=limits
    )
    assert capped.status == "unresolved"
    assert "mapped_coverage_resource_budget" in _checks(capped, "unresolved")
    assert capped.binding.source_revision == partition.source_revision


def test_absent_region_integrals_are_unbounded_not_certified() -> None:
    mesh, geometry, embedding = _mapped_split_square()
    domain = _split_square_domain()
    certified = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        domain,
        np.asarray((0, 1), dtype=np.int64),
        embedding=embedding,
    )
    finding = phx.geometry.MeshCertificateFinding(
        "mapped_measure_source", "unresolved", "cell", (0,)
    )

    def rebuild(findings: tuple[Any, ...], bounds: Any, errors: Any) -> Any:
        return phx.geometry.DomainCoverageCertificate(
            certified.binding,
            certified.embedding_certificate_id,
            domain,
            findings,
            requested_region_measures=certified.requested_region_measures,
            achieved_region_measures=(None, certified.achieved_region_measures[1]),
            requested_region_measure_bounds=certified.requested_region_measure_bounds,
            achieved_region_measure_bounds=bounds,
            integration_error_bounds=errors,
            covered_source_facet_count=certified.covered_source_facet_count,
            facet_source_overlaps=certified.facet_source_overlaps,
            premise_certificate_ids=certified.premise_certificate_ids,
            candidate_pair_count=certified.candidate_pair_count,
            subdivision_piece_count=certified.subdivision_piece_count,
            maximum_subdivision_depth_reached=certified.maximum_subdivision_depth_reached,
            source_expression_work_units=certified.source_expression_work_units,
            source_expression_required_work_units=(
                certified.source_expression_required_work_units
            ),
            source_expression_peak_bytes=certified.source_expression_peak_bytes,
        )

    present = certified.achieved_region_measure_bounds[1]
    absent = rebuild((finding,), (None, present), (None, 0.0))
    assert absent.status == "unresolved"
    assert absent.achieved_region_measure_bounds[0] is None
    assert absent.integration_error_bounds[0] is None
    with pytest.raises(ValueError, match="absent region integral"):
        rebuild((finding,), ((-math.inf, math.inf), present), (math.inf, 0.0))
    with pytest.raises(ValueError, match="every achieved region integral"):
        rebuild((), (None, present), (None, 0.0))


def test_mapped_coverage_refuses_missing_material_interface() -> None:
    mesh, geometry, embedding = _mapped_split_square()
    certificate = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        _split_square_domain(),
        np.asarray((0, 0), dtype=np.int64),
        embedding=embedding,
    )

    assert certificate.status == "violated"
    assert {"omitted_interface", "region_measure"} <= _checks(certificate, "violated")
    assert certificate.achieved_region_measures == (1.0, 0.0)


def test_mapped_coverage_refuses_missing_region() -> None:
    mesh, geometry, embedding = _mapped_split_square(cells=1)
    certificate = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        _split_square_domain(),
        np.asarray((0,), dtype=np.int64),
        embedding=embedding,
    )

    assert certificate.status == "violated"
    assert {"uncovered_boundary", "omitted_interface", "region_measure"} <= _checks(
        certificate, "violated"
    )
    assert certificate.achieved_region_measures == (0.5, 0.0)


def test_planar_corners_do_not_certify_a_nonplanar_mapped_boundary() -> None:
    mesh, geometry, domain = _mapped_reference_domain("hexahedron")
    coordinates = np.asarray(geometry.coordinates, dtype=np.float64).copy()
    face_center = np.all(
        coordinates == np.asarray((0.5, 0.5, 0.0), dtype=np.float64), axis=1
    )
    coordinates[face_center, 2] = -0.0625
    geometry = phx.discretization.CellGeometrySpec(
        {"mapped": geometry.elements[0]},
        {"mapped": np.arange(coordinates.shape[0], dtype=np.int64)[None]},
        coordinates,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0,), dtype=np.int64), embedding=embedding
    )

    assert certificate.status != "certified"
    assert "unmatched_boundary_facet" in _checks(certificate, "violated")


def test_mapped_containment_capacity_cannot_be_reported_as_coverage() -> None:
    mesh, geometry, domain = _mapped_reference_domain("hexahedron")
    coordinates = np.asarray(geometry.coordinates, dtype=np.float64).copy()
    face_center = np.all(
        coordinates == np.asarray((0.5, 0.5, 0.0), dtype=np.float64), axis=1
    )
    # Keep the facet in its declared plane but make its parameterization
    # genuinely quadratic, so one Bernstein node cannot prove containment.
    coordinates[face_center, 0] += 0.0625
    geometry = phx.discretization.CellGeometrySpec(
        {"mapped": geometry.elements[0]},
        {"mapped": np.arange(coordinates.shape[0], dtype=np.int64)[None]},
        coordinates,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    certificate = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        domain,
        np.asarray((0,), dtype=np.int64),
        embedding=embedding,
        limits=phx.geometry.MeshCertificateLimits(maximum_bernstein_nodes=1),
    )

    assert certificate.status == "unresolved"
    assert "mapped_facet_containment" in _checks(certificate, "unresolved")
    assert certificate.covered_source_facet_count < domain.facets.shape[0]
    assert certificate.achieved_region_measures == (1.0,)
    assert certificate.integration_error_bounds == (0.0,)


def _cube_with_independent_boundary_triangulation(
    mapped: bool,
) -> tuple[Any, Any, Any]:
    from itertools import product

    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._reference_cell import reference_cell_topology

    _, _, domain = _mapped_reference_domain("hexahedron")
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


@pytest.mark.parametrize("mapped", [False, True], ids=["affine-clipping", "mapped-union"])
def test_independent_source_diagonals_do_not_constrain_target_facets(
    mapped: bool,
) -> None:
    mesh, geometry, domain = _cube_with_independent_boundary_triangulation(mapped)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0, 0), dtype=np.int64), embedding=embedding
    )

    assert certificate.status == "certified"
    assert certificate.covered_source_facet_count == 12
    assert certificate.achieved_region_measures == (1.0,)
    relations = certificate.facet_source_overlaps
    assert {row[1] for row in relations} == set(range(12))
    assert {row[5] for row in relations} == {"candidate" if mapped else "exact"}
    target_fragments: dict[int, set[int]] = {}
    for facet, source, axes, lower, upper, _, space in relations:
        target_fragments.setdefault(facet, set()).add(source)
        assert len(axes) == 2
        assert 0.0 <= lower <= upper
        assert space == "physical"
    assert any(len(rows) > 1 for rows in target_fragments.values())
    if not mapped:
        for source in range(12):
            assert sum(row[3] for row in relations if row[1] == source) == 0.5
            assert sum(row[4] for row in relations if row[1] == source) == 0.5


@pytest.mark.parametrize("mapped", [False, True], ids=["affine", "mapped"])
def test_overlapping_authoritative_source_fragments_are_not_a_union_proof(
    mapped: bool,
) -> None:
    mesh, geometry, source = _cube_with_independent_boundary_triangulation(mapped)
    domain = phx.geometry.PiecewiseLinearDomain(
        source.vertices,
        np.concatenate((source.facets, source.facets[:1])),
        np.concatenate((source.facet_regions, source.facet_regions[:1])),
        source.region_ids,
        source_id="overlapping-cube-fragments",
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0, 0), dtype=np.int64), embedding=embedding
    )

    assert certificate.status == "violated"
    assert "overlapping_source_facets" in _checks(certificate, "violated")


def _independent_mapped_shear_domain() -> tuple[Any, Any, Any]:
    from phydrax.discretization._cell_geometry import (
        CellGeometryRestrictionSource,
        coordinate_lagrange_element,
        RestrictedCellGeometryElement,
    )
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain

    source_mesh, _, reference_domain = _mapped_reference_domain("hexahedron")
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
    reference_mesh, _, _ = _cube_with_independent_boundary_triangulation(False)
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


def test_exact_root_restrictions_cover_an_independent_nonplanar_mapped_source() -> None:
    mesh, geometry, domain = _independent_mapped_shear_domain()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0, 0), dtype=np.int64), embedding=embedding
    )

    assert certificate.status == "certified"
    assert certificate.binding.source_id == domain.source_id
    assert certificate.binding.source_revision == domain.source_revision
    assert certificate.requested_region_measures == (1.0,)
    assert certificate.achieved_region_measures == (1.0,)
    assert certificate.integration_error_bounds == (0.0,)
    assert certificate.covered_source_facet_count == 12
    assert {row[1] for row in certificate.facet_source_overlaps} == set(range(12))
    assert {row[5:] for row in certificate.facet_source_overlaps} == {
        ("candidate", "reference")
    }


def test_mapped_source_identity_does_not_substitute_for_actual_root_coefficients() -> (
    None
):
    mesh, geometry, domain = _independent_mapped_shear_domain()
    changed = np.asarray(geometry.coordinates, dtype=np.float64).copy()
    changed[0, 2] += 0.0625
    geometry = phx.discretization.CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        changed,
        restriction_source=geometry.restriction_source,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0, 0), dtype=np.int64), embedding=embedding
    )

    assert certificate.status == "violated"
    assert "mapped_domain_root_expression" in _checks(certificate, "violated")


def test_matching_mapped_coefficients_without_scientific_root_identity_remain_unresolved() -> (
    None
):
    mesh, geometry, domain = _independent_mapped_shear_domain()
    geometry = phx.discretization.CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        geometry.coordinates,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0, 0), dtype=np.int64), embedding=embedding
    )

    assert certificate.status == "unresolved"
    assert "mapped_domain_restriction_source" in _checks(certificate, "unresolved")


def test_mapped_root_partition_gap_is_not_filled_by_copying_source_measure() -> None:
    mesh, geometry, domain = _independent_mapped_shear_domain()
    mesh = phx.discretization.CellMesh(
        mesh.coordinates, (mesh.blocks[0],), vertex_global_ids=mesh.vertex_global_ids
    )
    record = geometry.restriction_source
    if record is None:
        raise RuntimeError(
            "The independently declared restriction fixture lost its source."
        )
    from phydrax.discretization._cell_geometry import CellGeometryRestrictionSource

    record = CellGeometryRestrictionSource(
        record.source_geometry_id,
        record.source_topology_id,
        {"left": record.block_parent_cell_ids["left"]},
        {"left": record.block_parent_vertex_ids["left"]},
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"left": geometry.elements[0]},
        {"left": geometry.geometry_dofs[0]},
        geometry.coordinates,
        restriction_source=record,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0,), dtype=np.int64), embedding=embedding
    )

    assert certificate.status == "violated"
    assert "mapped_domain_root_coverage_premise" in _checks(certificate, "violated")
    assert certificate.requested_region_measures == (1.0,)
    assert certificate.achieved_region_measures == (0.5,)
    assert certificate.achieved_region_measure_bounds == ((0.5, 0.5),)
    assert certificate.integration_error_bounds == (0.0,)


def test_mapped_image_entity_identity_binds_actual_source_map_not_reference_coincidence() -> (
    None
):
    from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain

    _, _, domain = _independent_mapped_shear_domain()
    geometry = domain.source_geometry
    changed = phx.discretization.CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        geometry.coordinates + np.asarray((0.0, 0.0, 0.25), dtype=np.float64),
    )
    other = MappedReferenceDomain(
        domain.reference_domain,
        domain.reference_mesh,
        changed,
        domain.cell_regions,
        source_id=domain.source_id,
        source_revision=domain.source_revision,
    )

    assert domain.entity_set_id(3) != other.entity_set_id(3)
    assert domain.image_entity_id(3, 0) != other.image_entity_id(3, 0)
    with pytest.raises(ValueError, match="actual reference entity"):
        domain.image_entity_id(3, 1)


@pytest.mark.parametrize("mapped", [False, True], ids=["affine", "curved-interior"])
def test_selected_planar_source_scope_proves_complete_fragment_union(
    mapped: bool,
) -> None:
    from phydrax.geometry._planar_facet_scope import certify_planar_facet_source_scope

    mesh, geometry, domain = _cube_with_independent_boundary_triangulation(mapped)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    coverage = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0, 0), dtype=np.int64), embedding=embedding
    )
    facets = tuple(
        sorted({row[0] for row in coverage.facet_source_overlaps if row[1] in (0, 1)})
    )
    certificate = certify_planar_facet_source_scope(
        mesh,
        geometry,
        domain,
        np.asarray(facets, dtype=np.int64),
        {42: (0, 1)},
        source_face_entity_set_id="declared-source-face-labels",
        coverage=coverage,
    )

    assert certificate.status == "certified"
    assert certificate.distance_upper == 0.0
    assert tuple(
        (row[2], row[3]) for row in certificate.source_fragment_projected_measures
    ) == ((1, 2), (1, 2))


@pytest.mark.parametrize("mapped", [False, True], ids=["affine", "curved-interior"])
def test_whole_domain_coverage_does_not_fill_a_selected_facet_scope_gap(
    mapped: bool,
) -> None:
    from phydrax.geometry._planar_facet_scope import certify_planar_facet_source_scope

    mesh, geometry, domain = _cube_with_independent_boundary_triangulation(mapped)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    coverage = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0, 0), dtype=np.int64), embedding=embedding
    )
    facets = tuple(
        sorted({row[0] for row in coverage.facet_source_overlaps if row[1] in (0, 1)})
    )
    certificate = certify_planar_facet_source_scope(
        mesh,
        geometry,
        domain,
        np.asarray(facets[:1], dtype=np.int64),
        {42: (0, 1)},
        source_face_entity_set_id="declared-source-face-labels",
        coverage=coverage,
    )

    assert coverage.status == "certified"
    assert certificate.status != "certified"
    assert certificate.distance_upper is None


def test_generated_caps_cannot_be_declared_as_original_source_faces() -> None:
    from phydrax.geometry._planar_facet_scope import certify_planar_facet_source_scope

    mesh, geometry, domain = _cube_with_independent_boundary_triangulation(False)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    coverage = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0, 0), dtype=np.int64), embedding=embedding
    )
    with pytest.raises(ValueError, match="Generated caps"):
        certify_planar_facet_source_scope(
            mesh,
            geometry,
            domain,
            np.asarray((coverage.facet_source_overlaps[0][0],), dtype=np.int64),
            {-1: (0,)},
            source_face_entity_set_id="original-source-faces",
            coverage=coverage,
        )


class _SampledSphere:
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


def _octahedron() -> Any:
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


@pytest.mark.parametrize(
    ("tolerance", "status"),
    [
        pytest.param(0.8, "certified", id="loose"),
        pytest.param(0.1, "violated", id="tight"),
    ],
)
def test_exact_sdf_source_fidelity_is_two_sided_and_certified(
    tolerance: float, status: str
) -> None:
    surface = _octahedron()
    sphere = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="sphere").compile()
    source = phx.geometry.ImplicitBoundarySource(
        sphere, source_id="unit-sphere", spacing=0.2
    )
    certificate = phx.geometry.certify_source_fidelity(
        surface,
        phx.discretization.CellGeometrySpec.affine(surface),
        source,
        tolerance=tolerance,
    )
    # Independent reference: the octahedron face centers lie 1 - 1/sqrt(3) from
    # the sphere, the largest deviation in both directions.
    hausdorff = 1.0 - 1.0 / math.sqrt(3.0)

    assert certificate.status == status
    assert certificate.mesh_to_source_semantics == "certified"
    assert certificate.source_to_mesh_semantics == "certified"
    assert (
        certificate.source_to_mesh_lower <= hausdorff <= certificate.source_to_mesh_upper
    )
    assert (
        certificate.mesh_to_source_lower <= hausdorff <= certificate.mesh_to_source_upper
    )
    assert certificate.binding.source_revision == source.source_revision


def test_sampled_source_bounds_never_certify_fidelity() -> None:
    surface = _octahedron()
    certificate = phx.geometry.certify_source_fidelity(
        surface,
        phx.discretization.CellGeometrySpec.affine(surface),
        _SampledSphere(),
        tolerance=10.0,
    )

    assert certificate.mesh_to_source_semantics == "sampled"
    assert certificate.status == "unresolved"


def _plc_audit(mesh: Any, **policy: Any) -> Any:
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    return geometry, phx.meshing.audit_cell_mesh(
        mesh, geometry, policy=phx.meshing.CellMeshAuditPolicy(**policy)
    )


def test_volume_route_refuses_an_audit_that_skipped_closure() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(_SQUARE, _CELLS)
    geometry, audit = _plc_audit(mesh)

    with pytest.raises(ValueError, match="open_boundary"):
        phx.meshing.certify_meshing_acceptance(
            mesh,
            geometry,
            audit,
            schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
            domain=_split_square_domain(),
            cell_regions=np.asarray((0, 0, 1, 1)),
        )


@pytest.mark.parametrize(
    ("regions", "stage_status"),
    [
        pytest.param(
            (0, 0, 1, 1), phx.meshing.MeshingStageStatus.PASSED, id="conforming"
        ),
        pytest.param((0, 0, 0, 0), phx.meshing.MeshingStageStatus.FAILED, id="omitted"),
    ],
)
def test_plc_acceptance_schedule_composes_certificates(
    regions: tuple[int, ...], stage_status: Any
) -> None:
    mesh = phx.discretization.CellMesh.from_triangles(_SQUARE, _CELLS)
    geometry, audit = _plc_audit(
        mesh, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=_split_square_domain(),
        cell_regions=np.asarray(regions),
    )
    stage = phx.meshing.acceptance_stage_report(report)

    assert stage.status == stage_status
    assert {value.check for value in report.outcomes} == set(
        report.schedule.required_checks
    )
    assert ("region_measure[left]", 0.5) in report.requested
    if report.passed:
        report.require_passed()
        return
    failing = {value.check: value for value in report.failing_outcomes}
    assert set(failing) == {"domain_coverage"}
    with pytest.raises(phx.meshing.MeshingFailure) as raised:
        report.require_passed()
    assert raised.value.stage == "certification"
    assert ("region_measure[left]", 1.0) in raised.value.evidence.achieved


def test_certificate_trace_keeps_independent_resource_quantities() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(_SQUARE, _CELLS)
    geometry, audit = _plc_audit(
        mesh,
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
    )
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=_split_square_domain(),
        cell_regions=np.asarray((0, 0, 1, 1), dtype=np.int64),
        limits=phx.geometry.MeshCertificateLimits(maximum_work_units=1),
    )
    stage = phx.meshing.acceptance_stage_report(report)
    assert stage.status is phx.meshing.MeshingStageStatus.UNRESOLVED
    assert report.embedding is not None
    refusal = next(
        finding
        for finding in report.embedding.findings
        if finding.resource == "coefficient_work"
    )
    assert dict(refusal.requested)["limit"] == 1
    assert dict(refusal.requested)["requested"] > 1
    assert dict(refusal.achieved)["completed"] <= 1
    with pytest.raises(phx.meshing.MeshingFailure) as raised:
        report.require_passed()
    assert raised.value.category is phx.meshing.MeshingFailureCategory.AUDIT_FAILED
    assert any(
        name.endswith(":coefficient_work:limit") and value == 1
        for name, value in raised.value.evidence.requested
    )
    assert any(
        name.endswith(":coefficient_work:completed") and value <= 1
        for name, value in raised.value.evidence.achieved
    )


def test_route_refuses_inputs_its_checks_do_not_use() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(_SQUARE, _CELLS)
    geometry, audit = _plc_audit(mesh)

    with pytest.raises(ValueError, match="exactly the inputs"):
        phx.meshing.certify_meshing_acceptance(
            mesh,
            geometry,
            audit,
            schedule=phx.meshing.MeshCertificationSchedule("curve"),
            domain=_split_square_domain(),
            cell_regions=np.asarray((0, 0, 1, 1)),
        )


class _ParabolicTriangleSource:
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


class _ParabolicChartSource(_ParabolicTriangleSource):
    def boundary_chart_cover(self, maximum_patches: int, /) -> Any:
        corners = self.graph(np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))))
        # Independent exact calculus: h-L = c*(x*x+y*y-x-y), whose
        # absolute maximum over this entire closed chart triangle is c/2.
        return phx.geometry.SourceBoundaryChartCover(
            corners[None],
            np.asarray((self.curvature / 2 + 1e-14,)),
            "certified",
            maximum_patches >= 1,
            self.source_id,
            self.source_revision,
        )


def _parabolic_coordinate_mesh(
    source: _ParabolicTriangleSource,
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


@pytest.mark.parametrize("degree", [2, 3], ids=["quadratic", "canonical-source-cubic"])
def test_curved_source_fidelity_uses_continuous_coordinate_and_source_covers(
    degree: int,
) -> None:
    source = _ParabolicTriangleSource()
    mesh, geometry = _parabolic_coordinate_mesh(source, degree=degree)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.25,
    )

    assert certificate.status == "certified"
    assert certificate.binding.coordinate_scope == "mapped"
    assert certificate.mesh_to_source_semantics == "certified"
    assert certificate.source_to_mesh_semantics == "certified"
    assert certificate.mesh_to_source_upper <= 0.25
    assert certificate.source_to_mesh_upper <= 0.25
    assert certificate.binding.source_revision == source.source_revision


def test_curved_chart_taylor_evidence_certifies_requested_hard_accuracy() -> None:
    source = _ParabolicChartSource(curvature=1e-4)
    mesh, geometry = _parabolic_coordinate_mesh(source)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=2e-4,
        limits=phx.geometry.MeshCertificateLimits(
            maximum_source_samples=1,
            maximum_subdivision_depth=0,
        ),
    )

    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper <= 2e-4
    assert certificate.source_to_mesh_upper <= 2e-4
    assert certificate.binding.source_revision == source.source_revision


@pytest.mark.parametrize(
    ("bias", "bow"),
    [pytest.param(0.3, 0.0, id="biased"), pytest.param(0.0, 1.2, id="bowed-interior")],
)
def test_curved_offsource_coordinates_are_witnessed_violations(
    bias: float, bow: float
) -> None:
    source = _ParabolicChartSource(curvature=1e-4)
    mesh, geometry = _parabolic_coordinate_mesh(source, bias=bias, bow=bow)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.05,
    )

    assert certificate.status == "violated"
    assert "boundary_deviation" in _checks(certificate, "violated")
    assert certificate.mesh_to_source_lower > certificate.tolerance


def test_curved_uncertified_source_never_promotes_coordinate_bounds() -> None:
    source = _ParabolicTriangleSource(certified=False)
    mesh, geometry = _parabolic_coordinate_mesh(source)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.25,
    )

    assert certificate.status == "unresolved"
    assert certificate.mesh_to_source_semantics == "sampled"
    assert certificate.source_to_mesh_semantics == "sampled"


def test_curved_source_cover_budget_refusal_reports_unbounded_direction() -> None:
    source = _ParabolicTriangleSource()
    mesh, geometry = _parabolic_coordinate_mesh(source)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.01,
        limits=phx.geometry.MeshCertificateLimits(maximum_source_samples=1),
    )

    assert certificate.status == "unresolved"
    assert "mesh_sample_capacity" in _checks(certificate, "unresolved")
    assert "source_sample_capacity" in _checks(certificate, "unresolved")
    assert math.isinf(certificate.mesh_to_source_upper)
    assert math.isinf(certificate.source_to_mesh_upper)


@pytest.mark.parametrize(
    ("budget", "finding"),
    [
        pytest.param({"maximum_subdivision_depth": 0}, "subdivision_depth", id="depth"),
        pytest.param(
            {"maximum_subdivision_pieces": 1}, "subdivision_capacity", id="pieces"
        ),
        pytest.param(
            {"maximum_bernstein_nodes": 1}, "bernstein_capacity", id="bernstein"
        ),
        pytest.param(
            {"maximum_distance_evaluations": 1}, "distance_capacity", id="queries"
        ),
    ],
)
def test_curved_enclosure_resource_limits_remain_explicitly_unresolved(
    budget: dict[str, int],
    finding: str,
) -> None:
    source = _ParabolicTriangleSource()
    mesh, geometry = _parabolic_coordinate_mesh(source)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.01,
        limits=phx.geometry.MeshCertificateLimits(**budget),
    )

    assert certificate.status == "unresolved"
    assert finding in _checks(certificate, "unresolved")
    assert certificate.mesh_to_source_upper > certificate.tolerance


def test_curved_source_fidelity_refines_an_affine_face_cover_without_relaxing_tolerance() -> (
    None
):
    source = _ParabolicTriangleSource()
    mesh, _ = _parabolic_coordinate_mesh(source)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        source,
        tolerance=0.25,
        sample_order=4,
    )

    assert certificate.status == "certified"
    assert certificate.binding.coordinate_scope == "affine"
    assert certificate.mesh_to_source_upper <= certificate.tolerance
    assert certificate.source_to_mesh_upper <= certificate.tolerance


def test_curved_analytic_source_has_independent_degree_one_projection_coverage() -> None:
    from phydrax.geometry._mesh_certificates import ImplicitProjectionBoundarySource
    from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile

    radius = 0.371
    implicit = phx.geometry.Sphere((0.0, 0.0, 0.0), radius, feature_id="sphere").compile()
    profile = AnalyticImplicitProfile(
        implicit,
        phx.SpatialCoordinateContract.si(),
        tube_radius=0.49 * radius,
        source_id="independent-sphere",
    )
    mesh = _octahedron().with_coordinates(
        radius * np.asarray(_octahedron().coordinates),
        numeric_version="source-scale",
    )
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        ImplicitProjectionBoundarySource(profile),
        tolerance=0.49 * radius,
    )

    assert certificate.status == "certified"
    assert certificate.source_to_mesh_upper <= certificate.tolerance
    assert certificate.projection_coverage is not None
    assert certificate.projection_coverage.status == "certified"
    assert certificate.projection_coverage.source_state_id == profile.state_id
    assert certificate.projection_coverage.source_reach_lower == profile.reach_lower
    assert certificate.projection_coverage.fiber_crossing_classes.count("interior") == 1


def test_curved_source_projection_refuses_a_missing_closed_mesh_premise() -> None:
    from phydrax.geometry._mesh_certificates import ImplicitProjectionBoundarySource
    from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile

    radius = 0.371
    implicit = phx.geometry.Sphere((0.0, 0.0, 0.0), radius, feature_id="sphere").compile()
    profile = AnalyticImplicitProfile(
        implicit,
        phx.SpatialCoordinateContract.si(),
        tube_radius=0.49 * radius,
    )
    closed = _octahedron()
    mesh = phx.discretization.CellMesh.from_triangles(
        radius * np.asarray(closed.coordinates),
        np.asarray(closed.blocks[0].vertices, dtype=np.int64)[:-1],
    )
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        ImplicitProjectionBoundarySource(profile),
        tolerance=0.49 * radius,
    )

    assert certificate.status == "unresolved"
    assert math.isinf(certificate.source_to_mesh_upper)
    assert certificate.projection_coverage is not None
    assert certificate.projection_coverage.status == "unresolved"


def test_curved_source_projection_reports_a_proved_distance_counterexample() -> None:
    from phydrax.geometry._mesh_certificates import ImplicitProjectionBoundarySource
    from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile

    radius = 0.371
    implicit = phx.geometry.Sphere((0.0, 0.0, 0.0), radius, feature_id="sphere").compile()
    profile = AnalyticImplicitProfile(
        implicit,
        phx.SpatialCoordinateContract.si(),
        tube_radius=0.49 * radius,
    )
    closed = _octahedron()
    mesh = phx.discretization.CellMesh.from_triangles(
        3.0 * radius * np.asarray(closed.coordinates),
        np.asarray(closed.blocks[0].vertices, dtype=np.int64),
    )
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        ImplicitProjectionBoundarySource(profile),
        tolerance=0.1 * radius,
    )

    assert certificate.status == "violated"
    assert certificate.projection_coverage is not None
    assert certificate.projection_coverage.status == "violated"
    assert any(
        finding.check == "projection_source_orientation_or_distance"
        and finding.status == "violated"
        for finding in certificate.findings
    )


def test_true_bilinear_mapped_source_boundary_equality_certifies_exact_zero() -> None:
    from phydrax.geometry._mesh_certificates import MappedDomainBoundarySource

    mesh, geometry, domain = _independent_mapped_shear_domain()
    source = MappedDomainBoundarySource(domain, np.asarray((0, 0), dtype=np.int64))
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.0,
    )

    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper == 0.0
    assert certificate.source_to_mesh_upper == 0.0
    assert certificate.domain_coverage is not None
    assert certificate.domain_coverage.status == "certified"
    assert certificate.binding.source_revision == domain.source_revision


def _curved_tetrahedral_components(
    scale: float, offset: float, degree: int
) -> tuple[Any, Any, Any]:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    element = coordinate_lagrange_element("tetrahedron", degree)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)

    def shear(points: np.ndarray) -> np.ndarray:
        result = points.copy()
        result[:, 0] += 0.0625 * points[:, 1] ** 2
        return result

    corners = np.concatenate((shear(_TETRAHEDRON), shear(scale * _TETRAHEDRON + offset)))
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


@pytest.mark.parametrize("degree", (2, 3, 4))
def test_curved_disconnected_volume_components_have_a_continuous_embedding(
    degree: int,
) -> None:
    mesh, geometry, validity = _curved_tetrahedral_components(1.0, 3.0, degree)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert certificate.findings == ()


@pytest.mark.parametrize(
    ("scale", "offset"),
    ((1.0, 0.25), (0.1, 0.2)),
    ids=("intersecting-components", "contained-component"),
)
def test_positive_jacobian_curved_volume_overlap_is_witnessed(
    scale: float, offset: float
) -> None:
    mesh, geometry, validity = _curved_tetrahedral_components(scale, offset, 2)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert "mapped_cell_overlap" in _checks(certificate, "violated")
    assert (
        certificate.subdivision_piece_count
        <= phx.geometry.MeshCertificateLimits().maximum_subdivision_pieces
    )


@pytest.mark.parametrize("kind", ("prism", "pyramid", "hexahedron"))
@pytest.mark.parametrize("degree", (2, 3))
def test_standard_hybrid_curved_maps_require_and_satisfy_local_injectivity(
    kind: str, degree: int
) -> None:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    element = coordinate_lagrange_element(kind, degree)
    points = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    corners = np.asarray(
        phx.discretization.reference_cell_topology(kind).vertices, dtype=np.float64
    ).copy()
    points[:, 0] += 0.03125 * points[:, 2] ** 2
    corners[:, 0] += 0.03125 * corners[:, 2] ** 2
    mesh = phx.discretization.CellMesh(
        corners,
        (
            phx.discretization.CellBlock(
                "cells", kind, np.arange(corners.shape[0], dtype=np.int32)[None]
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(points.shape[0], dtype=np.int32)[None]},
        points,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert "mapped_local_injectivity" in certificate.evaluated_checks


def test_positive_gram_determinant_does_not_certify_a_self_crossing_curve() -> None:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    element = coordinate_lagrange_element("interval", 3)
    t = np.asarray(element.reference_nodes, dtype=np.float64)[:, 0]
    points = np.stack((t * (1.0 - t), (t - 0.5) * (t * (1.0 - t) - 0.125)), axis=-1)
    mesh = phx.discretization.CellMesh(
        points[[0, -1]],
        (
            phx.discretization.CellBlock(
                "cells", "interval", np.asarray(((0, 1),), dtype=np.int32)
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"cells": element}, {"cells": np.arange(4, dtype=np.int32)[None]}, points
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert "mapped_self_contact" in _checks(certificate, "violated")


def test_nonplanar_trilinear_hex_faces_are_not_silently_triangulated() -> None:
    logical = np.asarray(tuple(np.ndindex((3, 3, 3))), dtype=np.float64) / 2.0
    points = logical.copy()
    points[:, 2] += 0.125 * logical[:, 0] * logical[:, 1]
    index = np.arange(27, dtype=np.int32).reshape((3, 3, 3))
    cells = np.asarray(
        [
            (
                index[i, j, k],
                index[i + 1, j, k],
                index[i + 1, j + 1, k],
                index[i, j + 1, k],
                index[i, j, k + 1],
                index[i + 1, j, k + 1],
                index[i + 1, j + 1, k + 1],
                index[i, j + 1, k + 1],
            )
            for i, j, k in np.ndindex((2, 2, 2))
        ],
        dtype=np.int32,
    )
    mesh = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("cells", "hexahedron", cells),)
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.binding.coordinate_scope == "mapped"
    assert certificate.status == "certified"
    # Bilinear boundary faces are decided as exact curved patches by the
    # boundary-degree theorem, never triangulated.
    assert "mapped_boundary_degree" in certificate.evaluated_checks
    assert certificate.boundary_degree is not None
    assert certificate.boundary_degree.status == "embedded"


def test_thin_gap_is_unresolved_with_the_exact_requested_subdivision_budget() -> None:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    element = coordinate_lagrange_element("triangle", 2)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    graph = np.column_stack((reference, reference[:, 0] ** 2))
    points = np.concatenate(
        (graph, graph + np.asarray((0.0, 0.0, 0.001), dtype=np.float64))
    )
    corners = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 0.0)), dtype=np.float64
    )
    mesh = phx.discretization.CellMesh.from_triangles(
        np.concatenate(
            (corners, corners + np.asarray((0.0, 0.0, 0.001), dtype=np.float64))
        ),
        np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"triangles": element},
        {
            "triangles": np.arange(points.shape[0], dtype=np.int32).reshape(
                (2, reference.shape[0])
            )
        },
        points,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    limits = phx.geometry.MeshCertificateLimits(maximum_subdivision_depth=0)
    certificate = phx.geometry.certify_global_embedding(
        mesh, geometry, validity, limits=limits
    )

    assert validity.all_certified
    assert certificate.status == "unresolved"
    assert "mapped_intersection_subdivision_depth" in _checks(certificate, "unresolved")
    assert "mapped_cell_overlap" not in _checks(certificate, "violated")
    assert certificate.binding.limits_id == limits.limits_id


def test_retained_certification_renews_source_coverage_on_actual_refinement() -> None:
    from phydrax.meshing._lineage import EntityLineage, EntityLineageKind, MeshLineage

    source = phx.discretization.CellMesh.from_tetrahedra(
        _TETRAHEDRON, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    geometry, audit = _plc_audit(
        source, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    facets = np.asarray(
        phx.discretization.reference_cell_topology("tetrahedron").entities[2],
        dtype=np.int64,
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        _TETRAHEDRON,
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (4, 1)),
        ("body",),
        source_id="independent-refinement-source",
    )
    original = phx.meshing.certify_meshing_acceptance(
        source,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
    )
    target = phx.discretization.CellMesh.from_tetrahedra(
        np.concatenate((_TETRAHEDRON, np.asarray(((0.5, 0.0, 0.0),), dtype=np.float64))),
        np.asarray(((0, 4, 2, 3), (4, 1, 2, 3)), dtype=np.int32),
    )
    target_geometry, target_audit = _plc_audit(
        target, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    lineage = MeshLineage(
        source.topology_id,
        target.topology_id,
        (
            EntityLineage(
                3,
                source.entity_set(3).entity_set_id,
                target.entity_set(3).entity_set_id,
                np.asarray((0, 0), dtype=np.int64),
                np.asarray((0, 1), dtype=np.int64),
                np.asarray((int(EntityLineageKind.REFINED_FROM),) * 2, dtype=np.int32),
            ),
        ),
    )
    renewed = original.request.recertify_transition(
        target, target_geometry, target_audit, lineage=lineage
    )

    assert original.passed and renewed.passed
    assert renewed.mesh_id == target.mesh_id
    assert renewed.geometry_id != original.geometry_id
    assert renewed.coverage is not None
    assert renewed.coverage.domain_id == domain.domain_id
    assert renewed.coverage.achieved_region_measures == pytest.approx((1.0 / 6.0,))
    assert np.array_equal(
        np.asarray(renewed.request.cell_regions), np.asarray((0, 0), dtype=np.int64)
    )


def test_retained_source_integrity_rejects_changed_region_assignments() -> None:
    import equinox as eqx
    import jax.numpy as jnp

    from phydrax.meshing._certification_inputs import MeshCertificationInputs

    mesh = phx.discretization.CellMesh.from_triangles(_SQUARE, _CELLS)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    request = MeshCertificationInputs(
        mesh,
        geometry,
        phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=_split_square_domain(),
        cell_regions=np.asarray((0, 0, 1, 1), dtype=np.int64),
    )
    changed = eqx.tree_at(
        lambda value: value.cell_regions, request, jnp.zeros((4,), dtype=jnp.int64)
    )

    request.validate_source_integrity()
    with pytest.raises(ValueError, match="assignments"):
        changed.validate_source_integrity()


@pytest.mark.parametrize(
    "extent", (2, 3), ids=("nonconvex-seven-hexes", "cavity-twenty-six-hexes")
)
@pytest.mark.parametrize(
    "restricted", (False, True), ids=("owning-root-maps", "retained-root-restrictions")
)
def test_common_polynomial_extension_certifies_curved_nonconvex_and_cavity_atlases(
    extent: int, restricted: bool
) -> None:
    from phydrax.discretization._cell_geometry import (
        CellGeometryRestrictionSource,
        coordinate_lagrange_element,
        RestrictedCellGeometryElement,
    )
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id

    element = coordinate_lagrange_element("hexahedron", 2)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    points = np.asarray(tuple(np.ndindex((extent + 1,) * 3)), dtype=np.float64)
    index = np.arange(points.shape[0], dtype=np.int32).reshape((extent + 1,) * 3)
    rows = []
    controls = []
    for i, j, k in np.ndindex((extent,) * 3):
        if (i, j, k) == (1, 1, 1):
            continue
        rows.append(
            (
                index[i, j, k],
                index[i + 1, j, k],
                index[i + 1, j + 1, k],
                index[i, j + 1, k],
                index[i, j, k + 1],
                index[i + 1, j, k + 1],
                index[i + 1, j + 1, k + 1],
                index[i, j + 1, k + 1],
            )
        )
        local = reference + np.asarray((i, j, k), dtype=np.float64)
        local[:, 2] += local[:, 0] ** 2 * local[:, 1] / 64.0
        controls.append(local)
    points[:, 2] += points[:, 0] ** 2 * points[:, 1] / 64.0
    cells = np.asarray(rows, dtype=np.int32)
    mesh = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("hex", "hexahedron", cells),)
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"hex": element},
        {
            "hex": np.arange(cells.shape[0] * reference.shape[0], dtype=np.int32).reshape(
                (cells.shape[0], reference.shape[0])
            )
        },
        np.concatenate(controls),
    )
    if restricted:
        record = CellGeometryRestrictionSource(
            cell_geometry_id(geometry),
            mesh.topology_id,
            {"hex": mesh.blocks[0].global_ids},
            {"hex": np.asarray(mesh.vertex_global_ids)[cells]},
        )
        geometry = phx.discretization.CellGeometrySpec(
            {
                "hex": RestrictedCellGeometryElement(
                    element,
                    "hexahedron",
                    np.eye(3, dtype=np.float64),
                    np.zeros((3,), dtype=np.float64),
                )
            },
            {"hex": geometry.geometry_dofs[0]},
            geometry.coordinates,
            restriction_source=record,
        )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert certificate.findings == ()


def test_positive_root_jacobians_do_not_admit_a_cracked_restricted_atlas() -> None:
    from phydrax.discretization._cell_geometry import (
        CellGeometryRestrictionSource,
        coordinate_lagrange_element,
        RestrictedCellGeometryElement,
    )
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.discretization._reference_cell import reference_cell_topology

    points = np.asarray(tuple(np.ndindex((3, 2, 2))), dtype=np.float64)
    lookup = {tuple(point): row for row, point in enumerate(points)}
    corners = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    cells = np.asarray(
        [
            [
                lookup[tuple(corner + np.asarray((origin, 0, 0), dtype=np.float64))]
                for corner in corners
            ]
            for origin in range(2)
        ],
        dtype=np.int64,
    )
    mesh = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("hex", "hexahedron", cells),)
    )
    element = coordinate_lagrange_element("hexahedron", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    controls = np.stack(
        (nodes.copy(), nodes + np.asarray((1.0, 0.0, 0.0), dtype=np.float64))
    )
    controls[0, :, 2] += 0.125 * nodes[:, 1] * (1.0 - nodes[:, 1])
    routes = np.arange(controls.size // 3, dtype=np.int64).reshape((2, nodes.shape[0]))
    source = phx.discretization.CellGeometrySpec(
        {"hex": element}, {"hex": routes}, controls.reshape((-1, 3))
    )
    record = CellGeometryRestrictionSource(
        cell_geometry_id(source),
        mesh.topology_id,
        {"hex": mesh.blocks[0].global_ids},
        {"hex": np.asarray(mesh.vertex_global_ids)[cells]},
    )
    geometry = phx.discretization.CellGeometrySpec(
        {
            "hex": RestrictedCellGeometryElement(
                element,
                "hexahedron",
                np.eye(3, dtype=np.float64),
                np.zeros((3,), dtype=np.float64),
            )
        },
        {"hex": routes},
        source.coordinates,
        restriction_source=record,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert validity.all_certified
    assert certificate.status == "violated"


@pytest.mark.parametrize(
    "triangles",
    (((0, 2, 1), (0, 3, 2)), ((0, 1, 3), (1, 2, 3))),
    ids=("original-diagonal", "changed-diagonal"),
)
def test_curved_adjacent_sphere_triangles_have_checked_physical_embedding(
    triangles: tuple[tuple[int, ...], ...],
) -> None:
    import jax.numpy as jnp

    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.geometry.brep._patches import SpherePatch

    surface = SpherePatch(
        np.zeros((3,), dtype=np.float64),
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        1.0,
    )
    charts = np.asarray(
        ((0.125, 0.125), (0.25, 0.125), (0.25, 0.25), (0.125, 0.25)), dtype=np.float64
    )
    element = coordinate_lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinates = []
    for row in triangles:
        corners = charts[np.asarray(row, dtype=np.int32)]
        reference = corners[0] + nodes @ (corners[1:] - corners[0])
        coordinates.append(
            np.asarray(
                surface.evaluate(jnp.asarray(reference, dtype=jnp.float64)),
                dtype=np.float64,
            )
        )
    mesh = phx.discretization.CellMesh(
        np.asarray(
            surface.evaluate(jnp.asarray(charts, dtype=jnp.float64)), dtype=np.float64
        ),
        (
            phx.discretization.CellBlock(
                "tri", "triangle", np.asarray(triangles, dtype=np.int32)
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"tri": element},
        {"tri": np.arange(12, dtype=np.int32).reshape((2, 6))},
        np.concatenate(coordinates),
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert certificate.findings == ()


def _prepared_tetrahedral_acceptance() -> tuple[Any, Any, Any, Any]:
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        _TETRAHEDRON,
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
    )
    geometry, audit = _plc_audit(
        mesh,
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
    )
    facets = np.asarray(
        phx.discretization.reference_cell_topology("tetrahedron").entities[2],
        dtype=np.int64,
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        _TETRAHEDRON,
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


def test_prepared_acceptance_preserves_scientific_report_and_work() -> None:
    mesh, geometry, audit, prepared = _prepared_tetrahedral_acceptance()
    request = prepared.request
    cold = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=request.schedule,
        domain=request.domain,
        cell_regions=request.cell_regions,
    )
    reused = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=request.schedule,
        domain=request.domain,
        cell_regions=request.cell_regions,
        prepared=prepared,
    )
    assert cold.passed and reused.passed
    assert reused.report_id == cold.report_id
    assert reused.embedding is not None and cold.embedding is not None
    assert reused.coverage is not None and cold.coverage is not None
    assert reused.embedding.candidate_pair_count == cold.embedding.candidate_pair_count
    assert reused.coverage.candidate_pair_count == cold.coverage.candidate_pair_count
    assert (
        reused.coverage.achieved_region_measure_bounds
        == cold.coverage.achieved_region_measure_bounds
    )


@pytest.mark.parametrize("mismatch", ("labels", "source", "limits", "coordinates"))
def test_prepared_acceptance_refuses_changed_scientific_bindings(mismatch: str) -> None:
    mesh, geometry, audit, prepared = _prepared_tetrahedral_acceptance()
    request = prepared.request
    domain, regions, limits = request.domain, request.cell_regions, request.limits
    if mismatch == "labels":
        regions = np.asarray((-1,), dtype=np.int64)
    elif mismatch == "source":
        domain = phx.geometry.PiecewiseLinearDomain(
            domain.vertices,
            domain.facets,
            domain.facet_regions,
            domain.region_ids,
            source_id="foreign-source",
        )
    elif mismatch == "limits":
        limits = phx.geometry.MeshCertificateLimits(maximum_candidate_pairs=1)
    else:
        mesh = mesh.with_coordinates(2.0 * _TETRAHEDRON, numeric_version="changed")
        geometry, audit = _plc_audit(
            mesh,
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
        )
    with pytest.raises(ValueError, match="actual acceptance request"):
        phx.meshing.certify_meshing_acceptance(
            mesh,
            geometry,
            audit,
            schedule=request.schedule,
            domain=domain,
            cell_regions=regions,
            limits=limits,
            prepared=prepared,
        )


def test_mapped_prepared_acceptance_keeps_original_limits_and_region_measure() -> None:
    from phydrax.meshing._plc_mapped_support import prepare_mapped_plc_support

    mesh, geometry, audit, initial = _prepared_tetrahedral_acceptance()
    domain = initial.request.domain
    assert isinstance(domain, phx.geometry.PiecewiseLinearDomain)
    limits = phx.geometry.MeshCertificateLimits(
        maximum_candidate_pairs=4096,
        maximum_ray_tests=8192,
        maximum_subdivision_pieces=16384,
    )
    request = phx.meshing.MeshCertificationInputs(
        mesh,
        geometry,
        initial.request.schedule,
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
        limits=limits,
        fidelity_sample_order=3,
    )
    original_coordinates = geometry.source_coordinates()
    support = prepare_mapped_plc_support(
        mesh,
        geometry,
        domain,
        np.asarray((17,), dtype=np.int64),
        np.asarray((17,), dtype=np.int64),
        maximum_support_queries=32768,
        validity=audit.validity,
        certification_request=request,
    )
    reused = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=request.schedule,
        domain=domain,
        cell_regions=request.cell_regions,
        limits=limits,
        fidelity_sample_order=3,
        prepared=support.prepared,
    )
    assert reused.passed
    assert reused.embedding is not None and reused.coverage is not None
    for certificate in (reused.embedding, reused.coverage):
        certificate.binding.require(mesh, geometry)
        assert certificate.binding.limits_id == limits.limits_id
        assert certificate.status == "certified"
        assert certificate.candidate_pair_count <= limits.maximum_candidate_pairs
        assert certificate.source_expression_work_units is not None
        assert certificate.source_expression_work_units <= limits.maximum_work_units
        assert certificate.source_expression_peak_bytes is not None
        assert certificate.source_expression_peak_bytes <= limits.maximum_scratch_bytes
    assert reused.coverage.binding.source_id == domain.source_id
    assert reused.coverage.binding.source_revision == domain.source_revision
    assert reused.coverage.region_ids == domain.region_ids
    assert geometry.source_coordinates() == original_coordinates
    bounds = reused.coverage.achieved_region_measure_bounds[0]
    assert bounds is not None
    lower, upper = bounds
    assert lower <= 1.0 / 6.0 <= upper


def test_mapped_support_preserves_its_stricter_original_quota_across_queries() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.meshing._plc_mapped_support import prepare_mapped_plc_support

    mesh, geometry, audit, initial = _prepared_tetrahedral_acceptance()
    domain = initial.request.domain
    assert isinstance(domain, phx.geometry.PiecewiseLinearDomain)
    labels = np.asarray((0,), dtype=np.int64)
    measured = prepare_mapped_plc_support(
        mesh,
        geometry,
        domain,
        labels,
        labels,
        maximum_support_queries=32768,
        validity=audit.validity,
    )
    limits = phx.geometry.MeshCertificateLimits(
        maximum_work_units=measured.ledger.work_units + 96,
    )
    ambient = algebra.CoordinateEnclosureBudget(
        4 * limits.maximum_work_units + 1024,
        limits.maximum_scratch_bytes,
    )
    empty_edges = np.empty((0, 2), dtype=np.int64)
    empty_triangles = np.empty((0, 3), dtype=np.int64)
    empty_facets = np.empty((0,), dtype=np.int64)
    query_work = [limits.maximum_work_units]
    with ambient.activate():
        algebra.expression_bernstein_coefficients(
            {(2, 0, 0): Fraction(1)},
            "simplex",
            3,
        )
        original_start = ambient.work_units
        support = prepare_mapped_plc_support(
            mesh,
            geometry,
            domain,
            labels,
            labels,
            maximum_support_queries=32768,
            validity=audit.validity,
            certificate_limits=limits,
        )
        assert ambient.work_units > original_start
        with pytest.raises(algebra.CoordinateEnclosureResourceError) as refused:
            # Every genuine point-authority query visits at least three exact
            # coordinates. The owning quota must refuse before the larger
            # ambient request is exhausted, even across separate activations.
            for _ in range(limits.maximum_work_units // 3 + 1):
                assert support.vertex_support(
                    0,
                    0,
                    0,
                    edge_vertices=empty_edges,
                    triangle_vertices=empty_triangles,
                    triangle_facets=empty_facets,
                    work=query_work,
                )
        assert refused.value.resource == "coefficient_work"
        assert refused.value.limit == original_start + limits.maximum_work_units
        assert refused.value.completed == ambient.work_units
        assert ambient.work_units < ambient.maximum_work_units


@pytest.mark.parametrize("mismatch", ("facets", "tolerance"))
def test_mapped_prepared_acceptance_refuses_changed_original_source_scope(
    mismatch: str,
) -> None:
    from examples._native_surface_sources import sphere
    from phydrax.meshing._plc_mapped_support import prepare_mapped_plc_support

    mesh, geometry, audit, initial = _prepared_tetrahedral_acceptance()
    domain = initial.request.domain
    assert isinstance(domain, phx.geometry.PiecewiseLinearDomain)
    source = phx.geometry.MeshingDomainBoundarySource(sphere().domain, (0,), resolution=4)
    facets = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    scoped = ((source, facets[:1], 2.0),)
    request = phx.meshing.MeshCertificationInputs(
        mesh,
        geometry,
        phx.meshing.MeshCertificationSchedule("volume_implicit"),
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
        source=source,
        fidelity_tolerance=2.0,
        scoped_fidelity=scoped,
    )
    # Preparation owns positive volume premises, not a source-fidelity verdict.
    support = prepare_mapped_plc_support(
        mesh,
        geometry,
        domain,
        np.asarray((0,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        maximum_support_queries=32768,
        validity=audit.validity,
        certification_request=request,
    )
    changed = (
        ((source, facets[1:2], 2.0),)
        if mismatch == "facets"
        else ((source, facets[:1], 1.5),)
    )
    with pytest.raises(ValueError, match="actual acceptance request"):
        phx.meshing.certify_meshing_acceptance(
            mesh,
            geometry,
            audit,
            schedule=request.schedule,
            domain=domain,
            cell_regions=request.cell_regions,
            source=source,
            fidelity_tolerance=2.0,
            scoped_fidelity=changed,
            prepared=support.prepared,
        )


def test_prepared_audit_refuses_a_different_validity_policy() -> None:
    mesh, geometry, _, prepared = _prepared_tetrahedral_acceptance()
    policy = phx.meshing.CellMeshAuditPolicy(
        validity_policy=phx.discretization.CellValidityPolicy(
            maximum_subdivision_depth=1
        ),
    )
    with pytest.raises(ValueError, match="exact audit policy"):
        phx.meshing.audit_cell_mesh(
            mesh,
            geometry,
            policy=policy,
            prepared_validity=prepared.validity,
        )


def _prepared_publication_authority() -> tuple[Any, Any, Any, Any]:
    from phydrax.meshing._association import PlcAssociationTransfer
    from phydrax.meshing._plc_mapped_support import certify_mapped_plc_associations
    from phydrax.meshing._volume_generation import _entity
    from phydrax.meshing.providers._native_publication import NativeCertificationRequest

    mesh, geometry, _, initial = _prepared_tetrahedral_acceptance()
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
                tuple(_entity(revision, role.value, index) for index in indices.tolist()),
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
    )
    return mesh, geometry, request, proof


def test_prepared_native_publication_refuses_authored_revision_drift() -> None:
    from dataclasses import replace

    from phydrax.meshing.providers._native_publication import publish_native_result

    mesh, geometry, request, proof = _prepared_publication_authority()
    request = replace(request, prepared=proof.support.prepared)
    assert request.source_revision != request.domain.source_revision
    policy = phx.meshing.CellMeshAuditPolicy(
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
    )
    published = publish_native_result(
        mesh,
        phx.SpatialCoordinateContract.si(),
        phx.meshing.MeshingComplianceReport("authoritative-publication"),
        (),
        phx.meshing.NativeMeshingProvider.info(),
        {"kind": "authoritative-publication"},
        request,
        audit_policy=policy,
        derivative_mode=phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(),
        unenforced_limits=(),
        geometry=geometry,
        associations=proof.associations,
    )
    assert published.certification is not None and published.certification.passed
    with pytest.raises(ValueError, match="actual source association authority"):
        publish_native_result(
            mesh,
            phx.SpatialCoordinateContract.si(),
            phx.meshing.MeshingComplianceReport("authoritative-publication"),
            (),
            phx.meshing.NativeMeshingProvider.info(),
            {"kind": "authoritative-publication"},
            replace(request, source_revision="foreign-authored-revision"),
            audit_policy=policy,
            derivative_mode=phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE,
            enforced_limits=(),
            unenforced_limits=(),
            geometry=geometry,
            associations=proof.associations,
        )


def _affine_tetrahedral_root_restrictions(
    second: np.ndarray,
) -> tuple[phx.discretization.CellMesh, phx.discretization.CellGeometrySpec]:
    from phydrax.discretization._cell_geometry import (
        _require_scalar_coordinate_element,
        CellGeometryRestrictionSource,
        RestrictedCellGeometryElement,
    )
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id

    source = phx.discretization.CellMesh.from_tetrahedra(
        np.concatenate((_TETRAHEDRON, second)),
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
                _require_scalar_coordinate_element(elements[0], "Affine root regression"),
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


def test_affine_restricted_root_overlap_is_a_source_premise_not_child_overlap() -> None:
    mesh, geometry = _affine_tetrahedral_root_restrictions(_TETRAHEDRON + 0.25)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    # These small children are disjoint, but their declared complete roots cross.
    assert validity.all_certified
    assert certificate.status == "unresolved"
    assert all(finding.status != "violated" for finding in certificate.findings)
    affected = {
        identifier
        for finding in certificate.findings
        if finding.entity_kind == "source_cell"
        for identifier in finding.entity_ids
    }
    assert affected == {101, 205}


def test_affine_restricted_roots_preserve_candidate_pair_exhaustion() -> None:
    mesh, geometry = _affine_tetrahedral_root_restrictions(_TETRAHEDRON + 5.0)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(
        mesh,
        geometry,
        validity,
        limits=phx.geometry.MeshCertificateLimits(maximum_candidate_pairs=1),
    )
    assert certificate.status == "unresolved"
    assert certificate.candidate_pair_count <= 1
    assert any(finding.entity_kind == "source_cell" for finding in certificate.findings)


def _bumped_square(first: Any, second: Any) -> Any:
    """Cubic triangles ``(0, 1, 2)`` and ``(2, 3, 0)`` traversing their diagonal oppositely.

    ``first``/``second`` give the physical bump amplitude along the diagonal
    normal as a function of the diagonal parameter ``t`` from vertex 0 to 2.
    """
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    corners = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    cells = np.asarray(((0, 1, 2), (2, 3, 0)), dtype=np.int32)
    # The exact-lattice coordinate element: its k/3 reference nodes give both
    # cells the same exact diagonal chart, so only authored bumps separate traces.
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


def _straight(t: np.ndarray) -> np.ndarray:
    return np.zeros_like(t)


def _antisymmetric(t: np.ndarray) -> np.ndarray:
    # Vanishes at both endpoints and the midpoint: three samples cannot see it.
    return 0.1 * t * (1.0 - t) * (1.0 - 2.0 * t)


def _symmetric(t: np.ndarray) -> np.ndarray:
    return 0.05 * t * (1.0 - t)


def test_shared_curved_edge_with_opposite_local_orientation_is_one_trace() -> None:
    certificate = _bumped_square(_antisymmetric, _antisymmetric)

    assert "mapped_trace_continuity" in certificate.evaluated_checks
    assert "mapped_trace_mismatch" not in _checks(certificate, "violated")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(_straight, _symmetric, id="interior-only-quadratic-gap"),
        pytest.param(_straight, _antisymmetric, id="cubic-gap-vanishing-at-three-points"),
        pytest.param(_symmetric, _antisymmetric, id="unequal-nonlinear-degrees"),
    ],
)
def test_shared_edge_interior_trace_gap_is_violated(first: Any, second: Any) -> None:
    certificate = _bumped_square(first, second)

    assert certificate.status == "violated"
    # The shared diagonal is the single interior facet; the pair decision may
    # separately report the same cells, which is not a trace proof.
    facets = [
        value
        for value in certificate.findings
        if value.check == "mapped_trace_mismatch" and value.entity_kind == "facet"
    ]
    assert len(facets) == 1
    assert facets[0].status == "violated"


@pytest.mark.parametrize("kind", ("quadrilateral", "hexahedron"))
def test_multi_affine_corner_coordinates_equal_the_source_basis_combination(
    kind: str,
) -> None:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_corner_images,
        coordinate_expressions,
        CoordinateEnclosureBudget,
        multi_affine_coordinates,
    )

    rng = np.random.default_rng(29)

    def bank(element: Any) -> tuple[tuple[Fraction, ...], ...]:
        rows = rng.uniform(-1.0, 1.0, (np.asarray(element.reference_nodes).shape[0], 3))
        return tuple(tuple(Fraction(float(value)) for value in row) for row in rows)

    linear, quadratic = (
        coordinate_lagrange_element(kind, 1),
        coordinate_lagrange_element(kind, 2),
    )
    local = bank(linear)
    ledger = CoordinateEnclosureBudget(1_000_000, 64 * 1024**2)
    with ledger.activate():
        prepared = multi_affine_coordinates(linear, local)
        assert multi_affine_coordinates(quadratic, bank(quadratic)) is None
    assert prepared is not None
    polynomials, corners = prepared
    # The unbudgeted owners form the source-basis combination and evaluate it.
    assert polynomials == coordinate_expressions(linear, local)
    assert corners == coordinate_corner_images(linear, local)
    assert ledger.work_units > 0


@pytest.mark.parametrize("kind", ("quadrilateral", "hexahedron"))
@pytest.mark.parametrize("exact_bank", (False, True))
def test_exact_coordinate_consumers_preserve_complete_signed_source_bank(
    kind: str,
    exact_bank: bool,
) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    element = coordinate_lagrange_element(kind, 1)
    controls = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    controls[-1, 0] += 0.125
    # Nonbinary signed source rows exercise the immutable exact-bank path too;
    # conversion to a float carrier must never become the source authority.
    local = (
        tuple(
            tuple(Fraction(float(value)) - Fraction(2, 3) for value in row)
            for row in controls
        )
        if exact_bank
        else controls
    )
    expected = algebra.coordinate_expressions(element, local)
    expected_corners = algebra.coordinate_corner_images(element, local)
    ledger = algebra.CoordinateEnclosureBudget(1_000_000, 64 * 1024**2)
    with ledger.activate():
        prepared = algebra.multi_affine_coordinates(element, local)
        assert prepared == (expected, expected_corners)
        assert algebra.coordinate_expressions(element, local) == expected
        assert algebra.coordinate_polynomials(element, local) == expected
        assert algebra.coordinate_corner_images(element, local) == expected_corners


@pytest.mark.parametrize("kind", ("quadrilateral", "hexahedron"))
def test_coordinate_preparation_rebinds_changed_complete_source_and_reference_map(
    kind: str,
) -> None:
    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._cell_geometry import (
        coordinate_lagrange_element,
        RestrictedCellGeometryElement,
    )

    element = coordinate_lagrange_element(kind, 1)
    controls = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    dimension = controls.shape[1]
    first = RestrictedCellGeometryElement(
        element,
        kind,
        np.eye(dimension, dtype=np.float64) * 0.5,
        np.zeros(dimension, dtype=np.float64),
    )
    second = RestrictedCellGeometryElement(
        element,
        kind,
        np.eye(dimension, dtype=np.float64) * 0.5,
        np.full(dimension, 0.25, dtype=np.float64),
    )
    expected_first = algebra.coordinate_expressions(first, controls)
    expected_second = algebra.coordinate_expressions(second, controls)
    ledger = algebra.CoordinateEnclosureBudget(1_000_000, 64 * 1024**2)
    with ledger.activate():
        assert algebra.coordinate_expressions(first, controls) == expected_first
        assert algebra.coordinate_expressions(second, controls) == expected_second
        before = algebra.coordinate_polynomials(element, controls)
        controls[-1, 0] += 0.125
        after = algebra.coordinate_polynomials(element, controls)
        changed_restriction = algebra.coordinate_expressions(first, controls)
        with pytest.raises(ValueError, match="every source degree of freedom"):
            algebra.coordinate_corner_images(element, controls[:-1])
    assert before != after
    assert after == algebra.coordinate_polynomials(element, controls)
    assert changed_restriction == algebra.coordinate_expressions(first, controls)
    assert expected_first != expected_second


def test_resumed_coordinate_stage_keeps_its_original_work_deadline() -> None:
    from phydrax.discretization._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    ledger = CoordinateEnclosureBudget(10, 1_000_000)
    ledger.reserve(5)
    start = ledger.work_units
    with ledger.bound_stage(2, 1_000_000, starting_work_units=start):
        ledger.reserve(2)
    ledger.reserve(1)
    completed = ledger.work_units
    with pytest.raises(CoordinateEnclosureResourceError) as refusal:
        with ledger.bound_stage(2, 1_000_000, starting_work_units=start):
            pytest.fail("A completed owner's work deadline was renewed.")
    assert refusal.value.resource == "coefficient_work"
    assert refusal.value.limit == start + 2
    assert refusal.value.requested == refusal.value.completed == completed
    assert ledger.work_units == completed
    with ledger.bound_stage(2, 1_000_000):
        ledger.reserve(2)
    assert ledger.work_units == ledger.maximum_work_units


@pytest.mark.parametrize("proof", ("embedding", "coverage"))
def test_public_coordinate_proof_borrows_the_exhausted_original_allowance(
    proof: str,
) -> None:
    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget

    mesh, geometry, embedding = _mapped_split_square()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    ledger = CoordinateEnclosureBudget(1, 64 * 1024**2)
    ledger.reserve(ledger.maximum_work_units)
    with ledger.activate():
        if proof == "embedding":
            certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)
        else:
            certificate = phx.geometry.certify_domain_coverage(
                mesh,
                geometry,
                _split_square_domain(),
                np.asarray((0, 1), dtype=np.int64),
                embedding=embedding,
            )
    assert certificate.status == "unresolved"
    refusal = next(
        finding
        for finding in certificate.findings
        if finding.resource == "coefficient_work"
    )
    requested, achieved = dict(refusal.requested), dict(refusal.achieved)
    assert requested["limit"] == ledger.maximum_work_units
    assert requested["requested"] > requested["limit"]
    assert (
        achieved["completed"]
        == achieved["source_expression_work_units"]
        == ledger.work_units
    )
    assert certificate.source_expression_work_units == ledger.work_units


def _tapered_hexes(top: float) -> Any:
    """Two planar-faced trilinear hexes whose top face narrows by ``top`` in y."""
    from phydrax.discretization._reference_cell import reference_cell_topology

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


@pytest.mark.parametrize("change", ("carrier", "source-bank", "source-route"))
def test_prepared_coordinate_scope_rechecks_every_current_carrier_corner(
    change: str,
) -> None:
    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget
    from phydrax.geometry._mesh_certificates import _coordinate_scope

    mesh = _tapered_hexes(0.5)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    ledger = CoordinateEnclosureBudget(1_000_000, 64 * 1024**2)
    scope, mapped, source = _coordinate_scope(mesh, geometry, ledger)
    assert scope == "mapped"
    np.testing.assert_array_equal(mapped, np.asarray((0, 1), dtype=np.int64))
    changed = np.asarray(mesh.coordinates, dtype=np.float64).copy()
    changed[-1, 0] += 0.125
    if change == "carrier":
        foreign = phx.discretization.CellMesh(changed, mesh.blocks)
        with pytest.raises(ValueError, match="correctly rounded source image"):
            _coordinate_scope(foreign, geometry, ledger)
    elif change == "source-bank":
        with pytest.raises(ValueError, match="correctly rounded source image"):
            _coordinate_scope(mesh, geometry.with_coordinates(changed), ledger)
    else:
        routes = {
            name: np.asarray(route, dtype=np.int64).copy()
            for name, route in zip(
                geometry.block_names, geometry.geometry_dofs, strict=True
            )
        }
        routes[geometry.block_names[0]][0, :2] = routes[geometry.block_names[0]][0, 1::-1]
        changed_route = phx.discretization.CellGeometrySpec(
            dict(zip(geometry.block_names, geometry.elements, strict=True)),
            routes,
            geometry.coordinates,
        )
        with pytest.raises(ValueError, match="correctly rounded source image"):
            _coordinate_scope(mesh, changed_route, ledger)
    repeated_scope, repeated_mapped, repeated_source = _coordinate_scope(
        mesh, geometry, ledger
    )
    assert repeated_scope == scope and repeated_source == source
    np.testing.assert_array_equal(repeated_mapped, mapped)


@pytest.mark.parametrize(
    ("top", "status"),
    ((0.5, "certified"), (0.05, "unresolved")),
    ids=("corner-controls-at-root", "generic-evidence-after-corner-refusal"),
)
def test_planar_trilinear_hexes_decide_through_exact_corner_proofs(
    top: float, status: str
) -> None:
    # A non-SPD corner control is the projected Jacobian's actual value there, so
    # no subdivision can prove the strong taper: refusal keeps the generic evidence.
    mesh = _tapered_hexes(top)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.binding.coordinate_scope == "mapped"
    assert certificate.status == status
    if status == "certified":
        assert {"mapped_trace_continuity", "mapped_straight_boundary_degree"} <= set(
            certificate.evaluated_checks
        )
        assert certificate.subdivision_piece_count == 2
    else:
        assert {value.check for value in certificate.findings} == {
            "local_injectivity_subdivision_depth"
        }
        assert certificate.subdivision_piece_count > 2


def test_trilinear_corner_proofs_refuse_one_unit_below_their_charged_work() -> None:
    mesh = _tapered_hexes(0.5)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    required = certificate.source_expression_work_units
    capped = phx.geometry.certify_global_embedding(
        mesh,
        geometry,
        validity,
        limits=phx.geometry.MeshCertificateLimits(maximum_work_units=required - 1),
    )

    assert certificate.status == "certified" and required > 0
    assert capped.status != "certified"
    assert "mapped_source_expression_resource_budget" in _checks(capped, "unresolved")


def _rotated_neighbor_hexes() -> Any:
    """Tapered trilinear hexes sharing ``x = 1``; the second sees that face through a quarter turn."""
    from phydrax.discretization._reference_cell import reference_cell_topology

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


def test_trilinear_face_trace_identity_holds_through_a_rotated_local_chart() -> None:
    mesh = _rotated_neighbor_hexes()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert "mapped_trace_mismatch" not in _checks(certificate, "violated")
    assert {"mapped_trace_continuity", "mapped_straight_boundary_degree"} <= set(
        certificate.evaluated_checks
    )


def _mixed_degree_hexes(bulge: float) -> tuple[Any, Any]:
    """A trilinear hex beside a triquadratic hex whose shared face center moves by ``bulge``."""
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._reference_cell import reference_cell_topology

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


@pytest.mark.parametrize(
    ("bulge", "status"),
    ((0.0, "certified"), (0.0625, "violated")),
    ids=("continuous-mixed-degree-face", "bulged-quadratic-face"),
)
def test_mixed_degree_hex_faces_use_the_exact_restriction_owner(
    bulge: float, status: str
) -> None:
    mesh, geometry = _mixed_degree_hexes(bulge)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert "mapped_trace_continuity" in certificate.evaluated_checks
    assert certificate.status == status
    assert ("mapped_trace_mismatch" in _checks(certificate, "violated")) is bool(bulge)


def _halved_tapered_restriction() -> tuple[Any, Any]:
    """Two tapered trilinear roots, each restricted to two exact half-x charts."""
    from phydrax.discretization._cell_geometry import (
        CellGeometryRestrictionSource,
        coordinate_lagrange_element,
        RestrictedCellGeometryElement,
    )
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.discretization._reference_cell import reference_cell_topology

    def taper(local: np.ndarray) -> np.ndarray:
        tapered = local.copy()
        tapered[:, 1] = 0.5 + (1.0 - 0.5 * local[:, 2]) * (local[:, 1] - 0.5)
        return tapered

    roots = _tapered_hexes(0.5)
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
    # Each root is halved in x; the children are exact affine source charts.
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


def test_restricted_children_of_planar_trilinear_roots_reuse_the_straight_root_theorem() -> (
    None
):
    mesh, geometry = _halved_tapered_restriction()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert (
        "restriction_multi_affine_root:mapped_straight_boundary_degree"
        in certificate.evaluated_checks
    )


def test_restricted_chart_measures_reuse_the_exact_root_determinant() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.geometry._mapped_coverage import integrate
    from phydrax.geometry._mesh_certificates import (
        _EmbeddingState,
        _mapped_coverage_measures,
    )

    mesh, geometry = _halved_tapered_restriction()
    regions = np.asarray((0, 1, 0, 1), dtype=np.int64)
    ledger = algebra.CoordinateEnclosureBudget(1_000_000, 64 * 1024**2)
    state = _EmbeddingState([], [])
    with ledger.activate():
        totals, _ = _mapped_coverage_measures(state, mesh, geometry, regions)
    elements, routes, _ = geometry.resolve(mesh)
    controls = geometry.source_coordinates()
    expected = [Fraction(0), Fraction(0)]
    cursor = 0
    for element, route in zip(elements, routes, strict=True):
        for row in np.asarray(route):
            coordinates = algebra.coordinate_polynomials(
                element, tuple(controls[index] for index in row)
            )
            assert coordinates is not None
            jacobian = tuple(
                tuple(algebra.derivative(value, axis) for axis in range(3))
                for value in coordinates
            )
            expected[int(regions[cursor])] += integrate(
                algebra.determinant(jacobian), "box", 3
            )
            cursor += 1

    assert state.findings == []
    assert totals == tuple(expected)
    assert ledger.work_units > 0


def test_affine_tetrahedral_restriction_measures_use_constant_jacobians() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._cell_geometry import (
        coordinate_lagrange_element,
        RestrictedCellGeometryElement,
    )
    from phydrax.geometry._mapped_coverage import integrate
    from phydrax.geometry._mesh_certificates import (
        _EmbeddingState,
        _mapped_coverage_measures,
    )

    element = coordinate_lagrange_element("tetrahedron", 1)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    shear = np.asarray(((1.0, 0.25, 0.0), (0.0, 1.5, 0.125), (0.0, 0.0, 0.75)))
    controls = nodes @ shear.T + np.asarray((0.5, -0.25, 1.0))
    # Two exact affine charts: the corner subtetrahedron and a sheared interior one.
    charts = (
        (np.diag((0.5, 0.5, 0.5)), np.zeros((3,))),
        (
            np.asarray(((0.25, 0.0, 0.0), (0.125, 0.25, 0.0), (0.0, 0.0, 0.25))),
            np.asarray((0.25, 0.125, 0.125)),
        ),
    )
    blocks, elements, dofs = [], {}, {}
    for row, (matrix, offset) in enumerate(charts):
        name = f"child{row}"
        vertices = np.arange(4, dtype=np.int32) + 4 * row
        blocks.append(
            phx.discretization.CellBlock(
                name,
                "tetrahedron",
                vertices[None],
                global_ids=np.asarray((row,), dtype=np.int64),
            )
        )
        elements[name] = RestrictedCellGeometryElement(
            element, "tetrahedron", matrix, offset
        )
        dofs[name] = np.arange(nodes.shape[0], dtype=np.int64)[None]
    mesh = phx.discretization.CellMesh(
        np.concatenate((controls, controls)), tuple(blocks)
    )
    geometry = phx.discretization.CellGeometrySpec(elements, dofs, controls)
    regions = np.asarray((0, 1), dtype=np.int64)
    ledger = algebra.CoordinateEnclosureBudget(1_000_000, 64 * 1024**2)
    state = _EmbeddingState([], [])
    with ledger.activate():
        totals, _ = _mapped_coverage_measures(state, mesh, geometry, regions)
    bank = geometry.source_coordinates()
    expected = []
    for name in elements:
        coordinates = algebra.coordinate_polynomials(
            elements[name], tuple(bank[index] for index in dofs[name][0])
        )
        assert coordinates is not None
        jacobian = tuple(
            tuple(algebra.derivative(value, axis) for axis in range(3))
            for value in coordinates
        )
        expected.append(integrate(algebra.determinant(jacobian), "simplex", 3))

    assert state.findings == []
    assert totals == tuple(expected)
    assert all(value > Fraction(0) for value in expected)
    assert ledger.work_units > 0


def _split_square_variant(vertices: Any, *, reversed_facets: bool, source_id: str) -> Any:
    """The split-square facets over ``vertices``, optionally with every facet reoriented."""
    declared = _split_square_domain()
    facets, regions = np.asarray(declared.facets), np.asarray(declared.facet_regions)
    if reversed_facets:
        # Reversing a facet flips its normal; swapping its regions keeps the same oriented interface.
        facets, regions = facets[:, ::-1].copy(), regions[:, ::-1].copy()
    return phx.geometry.PiecewiseLinearDomain(
        vertices, facets, regions, declared.region_ids, source_id=source_id
    )


def test_mapped_coverage_is_invariant_under_equivalent_source_facet_orientation() -> None:
    mesh, geometry, embedding = _mapped_split_square()
    regions = np.asarray((0, 1), dtype=np.int64)
    # y = 0 and y = 1 each carry two same-plane groups of different region pairs;
    # x = 0, 0.5 and 1 are distinct parallel planes.
    declared, reversed_ = (
        phx.geometry.certify_domain_coverage(
            mesh,
            geometry,
            _split_square_variant(
                _SQUARE, reversed_facets=flag, source_id="split-square"
            ),
            regions,
            embedding=embedding,
        )
        for flag in (False, True)
    )

    for certificate in (declared, reversed_):
        assert certificate.status == "certified"
        # Analytic areas: the interior bulge moves the interface, not the outer boundary.
        assert certificate.achieved_region_measures == (0.5, 0.5)
        assert certificate.covered_source_facet_count == 7
        assert {row[1] for row in certificate.facet_source_overlaps} == set(range(7))
        work = certificate.source_expression_work_units
        assert (
            work is not None
            and 0 < work <= phx.geometry.MeshCertificateLimits().maximum_work_units
        )
    assert reversed_.facet_source_overlaps == declared.facet_source_overlaps


def test_mapped_coverage_never_assigns_a_facet_to_a_parallel_source_plane() -> None:
    mesh, geometry, embedding = _mapped_split_square()
    vertices = _SQUARE.copy()
    vertices[[2, 3], 0] = 1.25
    domain = _split_square_variant(
        vertices, reversed_facets=False, source_id="stretched-split-square"
    )
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0, 1), dtype=np.int64), embedding=embedding
    )

    assert certificate.status == "violated"
    assert {"unmatched_boundary_facet", "uncovered_boundary"} <= _checks(
        certificate, "violated"
    )
    # The mesh edge x = 1 is parallel to the declared x = 1.25 facet, never covering it.
    assert 2 not in {row[1] for row in certificate.facet_source_overlaps}
    assert certificate.achieved_region_measures == (0.5, 0.5)
    assert certificate.requested_region_measures == (0.5, 0.75)


def test_curved_boundary_charts_match_no_declared_plane_and_keep_their_exact_measure() -> (
    None
):
    from fractions import Fraction

    bulge = Fraction(1, 8)
    mesh, geometry, embedding = _mapped_split_square(right_bulge=float(bulge))
    certificate = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        _split_square_domain(),
        np.asarray((0, 1), dtype=np.int64),
        embedding=embedding,
    )

    assert embedding.status == "certified"
    assert certificate.status == "violated"
    assert "unmatched_boundary_facet" in _checks(certificate, "violated")
    assert 2 not in {row[1] for row in certificate.facet_source_overlaps}
    # A quadratic edge with midpoint offset d encloses 2d/3 beyond its chord.
    assert certificate.achieved_region_measures == (
        0.5,
        float(Fraction(1, 2) + 2 * bulge / 3),
    )


@pytest.mark.parametrize(
    ("kind", "order"),
    (("quadrilateral", 1), ("quadrilateral", 2), ("hexahedron", 1), ("hexahedron", 2)),
)
def test_integer_box_jacobian_measure_equals_the_exact_determinant_integral(
    kind: str, order: int
) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.geometry._mapped_coverage import integrate
    from phydrax.geometry._mesh_certificates import _box_jacobian_measure

    element = coordinate_lagrange_element(kind, order)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    rng = np.random.default_rng(31)
    controls = nodes + rng.uniform(-0.1, 0.1, nodes.shape)
    local = tuple(tuple(Fraction(float(value)) for value in row) for row in controls)
    coordinates = algebra.coordinate_polynomials(element, local)
    assert coordinates is not None
    dimension = nodes.shape[1]
    jacobian = tuple(
        tuple(algebra.derivative(value, axis) for axis in range(dimension))
        for value in coordinates
    )

    assert _box_jacobian_measure(coordinates) == integrate(
        algebra.determinant(jacobian), "box", dimension
    )


@pytest.mark.parametrize("degrees", ((1, 1), (2, 3)), ids=("bilinear", "higher-degree"))
@pytest.mark.parametrize("reverse", (False, True), ids=("outward", "reversed"))
def test_full_nonplanar_polynomial_chart_flux_has_exact_signed_integral(
    degrees: tuple[int, int],
    reverse: bool,
) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.geometry._mapped_coverage import chart_flux

    first, second = algebra.axes(2)
    p, q = degrees
    height = algebra.add(
        algebra.constant(1, 2),
        algebra.multiply(algebra.power(first, p, 2), algebra.power(second, q, 2)),
    )
    coordinates = (second, first, height) if reverse else (first, second, height)
    # X=(u,v,1+u^p v^q): X dot (Xu cross Xv)=1+(1-p-q)u^p v^q.
    expected = (Fraction(1) + Fraction(1 - p - q, (p + 1) * (q + 1))) / 3
    assert chart_flux(coordinates, "box") == (-expected if reverse else expected)


@pytest.mark.parametrize("domain", ("box", "simplex"))
@pytest.mark.parametrize("height", (0, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_polynomial_flux_preserves_zero_and_translated_planar_source(
    domain: str,
    height: int,
    reverse: bool,
) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.geometry._mapped_coverage import chart_flux

    u, v = algebra.axes(2)
    first, second = (v, u) if reverse else (u, v)
    coordinates = (
        first,
        algebra.constant(height, 2),
        algebra.add(second, algebra.multiply(first, second)),
    )
    # X=(u,h,v+uv): X dot (Xu cross Xv)=-h(1+u).
    # Integrating on the unit box/triangle is independent of flux preparation.
    expected = -Fraction(height) * (Fraction(1, 2) if domain == "box" else Fraction(2, 9))
    if reverse:
        expected = -expected
    for shift in range(3):
        rotated = coordinates[shift:] + coordinates[:shift]
        assert chart_flux(rotated, domain) == expected


def test_exact_polynomial_zero_annihilates_without_visiting_other_support() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra

    polynomial: algebra.Polynomial = {(7, 3): Fraction(-2, 3), (0, 0): Fraction(5, 7)}
    original = dict(polynomial)
    ledger = algebra.CoordinateEnclosureBudget(0, 0)
    with ledger.activate():
        assert algebra.multiply({}, polynomial) == {}
        assert algebra.multiply(polynomial, {}) == {}
        assert algebra.scale(polynomial, 0) == {}
        assert algebra.math_product_polynomials((polynomial, polynomial, {}), 2) == {}
        assert algebra.determinant(((polynomial, polynomial), ({}, {}))) == {}
    assert polynomial == original


def test_exact_reference_late_zero_removes_unexecuted_prefix_powers() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra

    u, v = algebra.axes(2)
    arguments = (algebra.add(u, v), u, {})
    polynomial: algebra.Polynomial = {(12, 8, 1): Fraction(5, 7)}
    ledger = algebra.CoordinateEnclosureBudget(1_000, 1_000_000)
    with ledger.activate():
        composition = algebra.ArgumentComposition(arguments)
        prepared_work = ledger.work_units
        assert composition(polynomial) == {}
        assert ledger.work_units == prepared_work
        assert not ledger.reference_monomial_cache


@pytest.mark.parametrize("domain", ("box", "simplex"))
@pytest.mark.parametrize("axis", (0, 1, 2))
def test_polynomial_flux_exact_zero_coordinate_needs_no_tangent_work(
    domain: str,
    axis: int,
) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.geometry._mapped_coverage import chart_flux

    u, v = algebra.axes(2)
    high_order = algebra.add(algebra.power(u, 7, 2), algebra.power(v, 9, 2))
    coordinates = (high_order, algebra.multiply(u, v))
    with_zero = coordinates[:axis] + ({},) + coordinates[axis:]
    ledger = algebra.CoordinateEnclosureBudget(0, 0)
    with ledger.activate():
        assert chart_flux(with_zero, domain) == Fraction(0)
        assert chart_flux(tuple(reversed(with_zero)), domain) == Fraction(0)


@pytest.mark.parametrize("reverse", (False, True))
def test_sparse_exact_determinant_preserves_nonzero_signed_terms(reverse: bool) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra

    sign = -1 if reverse else 1
    ledger = algebra.CoordinateEnclosureBudget(100_000, 10_000_000)
    with ledger.activate():
        for variables in (2, 3):
            u, v = algebra.axes(variables)[:2]
            matrix: tuple[tuple[algebra.Polynomial, ...], ...] = (
                (u, u, {}),
                ({}, v, v),
                ({}, {}, algebra.add(u, v)),
            )
            if reverse:
                matrix = tuple(reversed(matrix))
            suffix = (0,) * (variables - 2)
            expected: algebra.Polynomial = {
                (2, 1) + suffix: Fraction(sign),
                (1, 2) + suffix: Fraction(sign),
            }
            # Polynomial Jacobian/Gram determinants seed n formal variables
            # for an n-by-n matrix. The expression API explicitly owns a
            # separate variable_dimension and also supports two-variable data.
            if variables == len(matrix):
                assert algebra.determinant(matrix) == expected
            assert (
                algebra.expression_determinant(matrix, variable_dimension=variables)
                == expected
            )


def test_immutable_coordinate_preparation_reuses_only_complete_current_source() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    element = coordinate_lagrange_element("hexahedron", 1)
    bank = tuple(
        tuple(Fraction(float(value)) for value in row) for row in element.reference_nodes
    )
    ledger = algebra.CoordinateEnclosureBudget(100_000, 10_000_000)
    with ledger.activate():
        first = algebra._coordinate_preparation_key(element, bank)
        first_work = ledger.work_units
        retained = ledger.retained_basis_bytes
        same = tuple(tuple(value for value in row) for row in bank)
        assert algebra._coordinate_preparation_key(element, same) is first
        assert ledger.work_units == first_work
        assert ledger.retained_basis_bytes == retained > 0
        changed = ((bank[0][0] + 1, *bank[0][1:]), *bank[1:])
        assert algebra._coordinate_preparation_key(element, changed)[0] == changed
        assert ledger.work_units > first_work
        mutable = np.asarray(element.reference_nodes, dtype=np.float64).copy()
        original = algebra._coordinate_preparation_key(element, mutable)
        before = ledger.work_units
        mutable[0, 0] += 1
        current = algebra._coordinate_preparation_key(element, mutable)
        assert current[0] != original[0]
        assert ledger.work_units > before


def test_exact_reference_monomials_reuse_immutable_ordered_preparation() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra

    first: algebra.Polynomial = {(1, 0): Fraction(1), (0, 1): Fraction(1)}
    second: algebra.Polynomial = {(1, 0): Fraction(1)}
    polynomial: algebra.Polynomial = {(3, 2, 0): Fraction(-2, 3)}
    arguments = (first, second, {})
    expected: algebra.Polynomial = {
        (5, 0): Fraction(-2, 3),
        (4, 1): Fraction(-2),
        (3, 2): Fraction(-2),
        (2, 3): Fraction(-2, 3),
    }
    ledger = algebra.CoordinateEnclosureBudget(100_000, 10_000_000)
    with ledger.activate():
        original = algebra.ArgumentComposition(arguments)(polynomial)
        first_work = ledger.work_units
        retained = ledger.retained_basis_bytes
        repeated = algebra.ArgumentComposition((dict(first), dict(second), {}))(
            polynomial
        )
        repeated_work = ledger.work_units - first_work
        assert tuple(original.items()) == tuple(expected.items())
        assert tuple(repeated.items()) == tuple(expected.items())
        assert 0 < repeated_work < first_work
        assert ledger.retained_basis_bytes == retained > 0
        assert len(ledger.reference_monomial_cache) == 1

        # Neither a mutable result nor subsequent source edits may poison the
        # immutable exact owner. The complete ordered support is scientific.
        original[(5, 0)] = Fraction(99)
        pristine: algebra.Polynomial = {(1, 0): Fraction(1), (0, 1): Fraction(1)}
        assert algebra.ArgumentComposition((pristine, second, {}))(polynomial) == expected
        first[(0, 1)] = Fraction(-1)
        changed = algebra.ArgumentComposition(arguments)(polynomial)
        assert changed[(4, 1)] == Fraction(2)
        assert len(ledger.reference_monomial_cache) == 2
        reversed_support = dict(reversed(tuple(pristine.items())))
        reordered = algebra.ArgumentComposition((reversed_support, second, {}))(
            polynomial
        )
        assert reordered == expected
        assert tuple(reordered) == tuple(reversed(tuple(expected)))
        assert len(ledger.reference_monomial_cache) == 3
        lower_power: algebra.Polynomial = {(2, 2, 0): Fraction(1)}
        lower_expected: algebra.Polynomial = {
            (4, 0): Fraction(1),
            (3, 1): Fraction(2),
            (2, 2): Fraction(1),
        }
        assert (
            algebra.ArgumentComposition((pristine, second, {}))(lower_power)
            == lower_expected
        )
        assert len(ledger.reference_monomial_cache) == 4

    # A separate owner earns its own first preparation; no process-global
    # source cache or second/reset ledger erases the genuine first work.
    independent = algebra.CoordinateEnclosureBudget(100_000, 10_000_000)
    with independent.activate():
        assert algebra.ArgumentComposition((pristine, second, {}))(polynomial) == expected
    assert independent.work_units > repeated_work
    assert independent.retained_basis_bytes > 0


def test_exact_reference_preparation_preserves_oriented_nonplanar_flux() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.geometry._mapped_coverage import chart_flux, face_charts

    x, y, z = algebra.axes(3)
    coordinates = (x, y, algebra.add(z, algebra.multiply(x, y)))
    faces = ((0, 3, 2, 1), (0, 1, 2, 3))
    expected = tuple(
        tuple(
            chart_flux(chart, domain)
            for chart, domain in face_charts(coordinates, "hexahedron", face)
        )
        for face in faces
    )
    assert expected[0] == tuple(-value for value in expected[1] if value is not None)
    assert all(
        value is not None and value != Fraction(0)
        for values in expected
        for value in values
    )
    ledger = algebra.CoordinateEnclosureBudget(100_000, 10_000_000)
    with ledger.activate():
        for _ in range(2):
            actual = tuple(
                tuple(
                    chart_flux(chart, domain)
                    for chart, domain in face_charts(coordinates, "hexahedron", face)
                )
                for face in faces
            )
            assert actual == expected
        assert ledger.reference_monomial_cache


def _curved_shared_hex_interface() -> tuple[Any, Any]:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    mesh = _tapered_hexes(1.0)
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


@pytest.mark.parametrize("split", (False, True), ids=("same-region", "two-materials"))
def test_full_curved_interface_region_flux_preserves_exact_physical_volumes(
    split: bool,
) -> None:
    from fractions import Fraction

    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget
    from phydrax.geometry._mesh_certificates import (
        _EmbeddingState,
        _mapped_coverage_measures,
    )

    mesh, geometry = _curved_shared_hex_interface()
    before = np.asarray(geometry.coordinates).copy()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    regions = np.asarray((0, 1) if split else (0, 0), dtype=np.int64)
    ledger = CoordinateEnclosureBudget(2_000_000, 81_920_000)
    state = _EmbeddingState([], [])
    with ledger.activate():
        embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
        assert validity.all_certified and embedding.status == "certified"
        totals, cells = _mapped_coverage_measures(
            state, mesh, geometry, regions, embedding=embedding
        )
    # The complete Q2 interface is x=1+y(1-y)z(1-z)/8, not its corner plane x=1.
    expected = (Fraction(289, 288), Fraction(287, 288)) if split else (Fraction(2),)
    assert totals == expected and len(cells) == 2
    assert not state.findings
    np.testing.assert_array_equal(geometry.coordinates, before)


def test_mapped_embedding_releases_local_root_workspace_but_keeps_complete_maps() -> None:
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_scope_key,
        CoordinateEnclosureBudget,
    )
    from phydrax.geometry._mapped_embedding import certify_mapped_embedding
    from phydrax.geometry._mesh_certificates import (
        _cell_global_ids,
        _EmbeddingState,
        MeshCertificateLimits,
    )

    mesh, geometry = _curved_shared_hex_interface()
    ledger = CoordinateEnclosureBudget(2_000_000, 81_920_000)
    state = _EmbeddingState([], [])
    with ledger.activate():
        ledger.reserve(1, 128)
        before = ledger.temporary_bytes_upper
        certify_mapped_embedding(
            state, mesh, geometry, _cell_global_ids(mesh), MeshCertificateLimits()
        )
        assert ledger.temporary_bytes_upper == before
        prepared = ledger.prepared_cell_cache[coordinate_scope_key(mesh, geometry)]
        assert len(prepared.cell_ids) == 2 and len(prepared.coordinates) == 2
        assert ledger.retained_basis_bytes > 0
        assert ledger.work_units > 1
    assert not state.findings


@pytest.mark.parametrize("dimension", (2, 3))
def test_full_barycentric_offsum_measure_uses_actual_physical_map(dimension: int) -> None:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        coordinate_lagrange_element,
    )
    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget
    from phydrax.geometry._mesh_certificates import (
        _EmbeddingState,
        _mapped_coverage_measures,
    )

    kind = "triangle" if dimension == 2 else "tetrahedron"
    original = coordinate_lagrange_element(kind, 1)
    origin = np.asarray((3.0, 5.0, 7.0), dtype=np.float64)[:dimension]
    controls = np.concatenate(
        (origin[None, :], origin[None, :] + np.eye(dimension, dtype=np.float64))
    )
    weights = np.eye(dimension + 1, dtype=np.float64)
    delta = Fraction(1, 2**52)
    weights[0, 1] = float(delta)
    action = BarycentricCellGeometryElement(original, weights)
    carrier = weights @ controls
    vertices = np.arange(dimension + 1, dtype=np.int64)[None, :]
    mesh = (
        phx.discretization.CellMesh.from_triangles(carrier, vertices)
        if dimension == 2
        else phx.discretization.CellMesh.from_tetrahedra(carrier, vertices)
    )
    name = mesh.blocks[0].name
    geometry = phx.discretization.CellGeometrySpec(
        {name: action}, {name: vertices}, controls
    )
    state = _EmbeddingState([], [])
    ledger = CoordinateEnclosureBudget(100_000, 1 << 24)
    with ledger.activate():
        totals, cells = _mapped_coverage_measures(
            state, mesh, geometry, np.asarray((0,), dtype=np.int64)
        )
    # J = I - delta * original_vertex_1 @ ones.T. Translation participates
    # because the complete action is not a partition-of-unity restriction.
    expected = (
        1 - delta * sum(Fraction(float(value)) for value in controls[1])
    ) / math.factorial(dimension)
    assert totals == (expected,) and len(cells) == 1
    assert totals != ((1 - delta) / math.factorial(dimension),)
    assert not state.findings
    np.testing.assert_array_equal(geometry.coordinates, controls)


def test_full_barycentric_incident_cells_keep_exact_shared_trace_and_measure() -> None:
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        coordinate_lagrange_element,
    )
    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget
    from phydrax.geometry._mesh_certificates import (
        _EmbeddingState,
        _mapped_coverage_measures,
    )

    original = coordinate_lagrange_element("triangle", 1)
    controls = np.asarray(((3.0, 5.0), (4.0, 5.0), (3.0, 6.0)), dtype=np.float64)
    delta = Fraction(1, 2**52)
    midpoint = np.asarray((0.5, 0.5 + float(delta), 0.0), dtype=np.float64)
    first = np.stack(
        (
            np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
            midpoint,
            np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        )
    )
    second = np.stack(
        (
            midpoint,
            np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        )
    )
    midpoint_image = midpoint @ controls
    points = np.stack((controls[0], midpoint_image, controls[1], controls[2]))
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "left",
                "triangle",
                np.asarray(((0, 1, 3),), dtype=np.int64),
                global_ids=np.asarray((101,), dtype=np.int64),
            ),
            phx.discretization.CellBlock(
                "right",
                "triangle",
                np.asarray(((1, 2, 3),), dtype=np.int64),
                global_ids=np.asarray((205,), dtype=np.int64),
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {
            "left": BarycentricCellGeometryElement(original, first),
            "right": BarycentricCellGeometryElement(original, second),
        },
        {
            "left": np.asarray(((0, 1, 2),), dtype=np.int64),
            "right": np.asarray(((0, 1, 2),), dtype=np.int64),
        },
        controls,
    )
    expected = Fraction(0)
    for bank in (first, second):
        images = tuple(
            tuple(
                sum(
                    (
                        Fraction(float(row[source]))
                        * Fraction(float(controls[source, axis]))
                        for source in range(3)
                    ),
                    Fraction(0),
                )
                for axis in range(2)
            )
            for row in bank
        )
        a = tuple(
            value - start for value, start in zip(images[1], images[0], strict=True)
        )
        b = tuple(
            value - start for value, start in zip(images[2], images[0], strict=True)
        )
        expected += (a[0] * b[1] - a[1] * b[0]) / 2
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    ledger = CoordinateEnclosureBudget(100_000, 1 << 24)
    state = _EmbeddingState([], [])
    with ledger.activate():
        embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
        assert validity.all_certified and embedding.status == "certified"
        totals, cells = _mapped_coverage_measures(
            state, mesh, geometry, np.asarray((0, 0), dtype=np.int64), embedding=embedding
        )
    assert "mapped_affine_source_maps" in embedding.evaluated_checks
    assert "mapped_exact_shared_corner_incidence" in embedding.evaluated_checks
    assert totals == (expected,) and len(cells) == 2
    assert expected != Fraction(1, 2)
    assert not state.findings
    np.testing.assert_array_equal(geometry.coordinates, controls)


def test_unproved_interface_trace_retains_actual_signed_cell_integrals() -> None:
    from fractions import Fraction

    from phydrax.geometry._mesh_certificates import (
        _EmbeddingState,
        _mapped_coverage_measures,
    )

    mesh, geometry = _mixed_degree_hexes(0.0625)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "violated"
    totals, _ = _mapped_coverage_measures(
        _EmbeddingState([], []),
        mesh,
        geometry,
        np.asarray((0, 0), dtype=np.int64),
        embedding=embedding,
    )
    # A missing shared-face trace cannot cancel the unmatched Q2 face bubble.
    assert totals == (Fraction(2) - Fraction(4, 9) * Fraction(1, 16),)


def test_boundary_region_integral_rejects_changed_complete_source_binding() -> None:
    from phydrax.geometry._mesh_certificates import (
        _EmbeddingState,
        _mapped_coverage_measures,
    )

    mesh, geometry = _curved_shared_hex_interface()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    changed = np.asarray(geometry.coordinates).copy()
    changed[0, 0] += 0.125
    with pytest.raises(ValueError, match="not bound"):
        _mapped_coverage_measures(
            _EmbeddingState([], []),
            mesh,
            geometry.with_coordinates(changed),
            np.asarray((0, 0), dtype=np.int64),
            embedding=embedding,
        )


def test_exact_boundary_region_integral_refuses_below_actual_work_without_renewal() -> (
    None
):
    from phydrax.discretization._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.geometry._mesh_certificates import (
        _EmbeddingState,
        _mapped_coverage_measures,
    )

    mesh, geometry = _curved_shared_hex_interface()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    regions = np.asarray((0, 1), dtype=np.int64)
    measured = CoordinateEnclosureBudget(2_000_000, 81_920_000)
    with measured.activate():
        _mapped_coverage_measures(
            _EmbeddingState([], []), mesh, geometry, regions, embedding=embedding
        )
    ledger = CoordinateEnclosureBudget(measured.work_units - 1, 81_920_000)
    before = np.asarray(geometry.coordinates).copy()
    with ledger.activate(), pytest.raises(CoordinateEnclosureResourceError) as refusal:
        _mapped_coverage_measures(
            _EmbeddingState([], []), mesh, geometry, regions, embedding=embedding
        )
    assert refusal.value.resource == "coefficient_work"
    assert refusal.value.requested > refusal.value.limit
    assert ledger.work_units <= ledger.maximum_work_units
    np.testing.assert_array_equal(geometry.coordinates, before)


@pytest.mark.parametrize(
    "kind,face",
    (
        ("tetrahedron", (1, 2, 3)),
        ("hexahedron", (0, 1, 5, 4)),
        ("pyramid", (0, 1, 4)),
    ),
    ids=("triangle-face", "full-quad-triangles", "pyramid-cube-side"),
)
@pytest.mark.parametrize(
    "rational", (False, True), ids=("complete-polynomial", "complete-rational")
)
def test_shared_face_composition_preserves_every_physical_source_component(
    kind: str,
    face: tuple[int, ...],
    rational: bool,
) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.geometry._mapped_coverage import face_charts

    x, y, z = algebra.axes(3)
    numerators = (
        algebra.add(x, algebra.multiply(algebra.multiply(x, y), z)),
        algebra.add(y, algebra.multiply(algebra.power(x, 2, 3), z)),
        algebra.add(z, algebra.multiply(x, algebra.power(y, 3, 3))),
    )
    denominator = algebra.add(
        algebra.constant(1, 3), algebra.scale(algebra.multiply(x, z), Fraction(1, 5))
    )
    coordinates: tuple[algebra.Expression, ...] = (
        tuple(algebra.rational_expression(value, denominator) for value in numerators)
        if rational
        else numerators
    )
    ledger = algebra.CoordinateEnclosureBudget(2_000_000, 81_920_000)
    with ledger.activate(), ledger.temporary_scope():
        charts = face_charts(coordinates, kind, face)
    corners = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)[
        np.asarray(face)
    ]
    pieces = (corners,) if len(face) == 3 else (corners[[0, 1, 2]], corners[[0, 2, 3]])
    # Seven exact nodes per axis determine the cross-multiplied degree-six
    # identity; this checks the complete functions, not a few corner samples.
    for chart_index, (chart, domain) in enumerate(charts):
        assert domain == ("box" if kind == "pyramid" else "simplex")
        for first in range(7):
            for second in range(7):
                u, v = Fraction(first, 12), Fraction(second, 12)
                if kind == "pyramid":
                    point = (u, Fraction(0), v)
                else:
                    piece = pieces[chart_index]
                    point = tuple(
                        Fraction(float(start))
                        + u * Fraction(float(a - start))
                        + v * Fraction(float(b - start))
                        for start, a, b in zip(piece[0], piece[1], piece[2], strict=True)
                    )
                a, b, c = point
                expected = (a + a * b * c, b + a * a * c, c + a * b * b * b)
                if rational:
                    weight = 1 + a * c / 5
                    expected = tuple(value / weight for value in expected)
                assert (
                    tuple(algebra.expression_evaluate(value, (u, v)) for value in chart)
                    == expected
                )
    assert 0 < ledger.work_units <= ledger.maximum_work_units


@pytest.mark.parametrize("rational", (False, True))
def test_oriented_face_flux_preserves_complete_source_and_triangle_integral(
    rational: bool,
) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.geometry._mapped_coverage import chart_flux, face_charts, face_flux

    x, y, z = algebra.axes(3)
    coordinates = (
        algebra.add(x, algebra.multiply(y, z)),
        algebra.add(y, algebra.scale(algebra.multiply(x, z), Fraction(1, 3))),
        algebra.add(z, algebra.scale(algebra.multiply(x, y), Fraction(1, 5))),
    )
    if rational:
        # A complete rational map with planar faces, not endpoint-authored data.
        denominator = algebra.add(
            algebra.constant(1, 3), algebra.scale(x, Fraction(1, 5))
        )
        coordinates = tuple(
            algebra.rational_expression(value, denominator) for value in (x, y, z)
        )
    for face in reference_cell_topology("hexahedron").entities[2]:
        contributions = tuple(
            chart_flux(chart, domain)
            for chart, domain in face_charts(coordinates, "hexahedron", face)
        )
        expected: Fraction | None = Fraction(0)
        for contribution in contributions:
            if contribution is None:
                expected = None
                break
            expected += contribution
        actual = face_flux(coordinates, "hexahedron", face)
        assert actual == expected
        if actual is not None:
            assert face_flux(coordinates, "hexahedron", tuple(reversed(face))) == -actual


def test_prepared_reference_action_preserves_current_source_and_independent_ledger() -> (
    None
):
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._cell_geometry import (
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )

    source = coordinate_lagrange_element("hexahedron", 1)
    chart = coordinate_lagrange_element("hexahedron", 2)
    nodes = np.asarray(chart.reference_nodes, dtype=np.float64)
    values = nodes.copy()
    values[:, 0] += nodes[:, 0] * (1 - nodes[:, 0]) * nodes[:, 1] / 8

    def action(current: np.ndarray) -> Any:
        exact = tuple(tuple(Fraction(float(value)) for value in row) for row in current)
        return PolynomialComposedCellGeometryElement(
            source,
            chart,
            np.asarray(
                [[value.numerator for value in row] for row in exact], dtype=np.int64
            ),
            np.asarray(
                [[value.denominator for value in row] for row in exact], dtype=np.uint64
            ),
        )

    element = action(values)
    original = tuple(
        tuple(Fraction(float(value)) for value in row) for row in source.reference_nodes
    )
    u, v, w = algebra.axes(3)
    expected = (
        algebra.add(
            u,
            algebra.scale(
                algebra.multiply(
                    algebra.multiply(
                        u, algebra.add(algebra.constant(1, 3), algebra.scale(u, -1))
                    ),
                    v,
                ),
                Fraction(1, 8),
            ),
        ),
        v,
        w,
    )
    ledger = algebra.CoordinateEnclosureBudget(2_000_000, 81_920_000)
    with ledger.activate():
        prepared = algebra.prepared_reference_arguments(element)
        first_work = ledger.work_units
        first_retained = ledger.retained_basis_bytes
        repeated = algebra.prepared_reference_arguments(element)
        assert ledger.work_units == first_work > 0
        assert ledger.retained_basis_bytes == first_retained > 0
        assert (
            tuple(algebra.ExpressionComposition(repeated)(axis) for axis in (u, v, w))
            == expected
        )
        assert algebra.coordinate_expressions(element, original) == expected

        # A current physical coefficient bank remains a fresh binding even
        # while it borrows the same immutable reference preparation.
        moved = tuple((row[0] * 2 + 1, *row[1:]) for row in original)
        assert algebra.coordinate_expressions(element, moved) == (
            algebra.add(algebra.scale(expected[0], 2), algebra.constant(1, 3)),
            v,
            w,
        )

        # A changed actual chart earns its own preparation and full map.
        changed_values = values.copy()
        center = np.flatnonzero(np.all(nodes == 0.5, axis=1))[0]
        changed_values[center, 0] += 1 / 16
        changed = action(changed_values)
        before = ledger.work_units
        changed_preparation = algebra.prepared_reference_arguments(changed)
        assert ledger.work_units > before
        changed_map = tuple(
            algebra.ExpressionComposition(changed_preparation)(axis) for axis in (u, v, w)
        )
        assert changed_map != expected
        cold = algebra.reference_composition_arguments(changed)
        assert changed_map == tuple(
            algebra.ExpressionComposition(cold)(axis) for axis in (u, v, w)
        )
        assert (
            tuple(algebra.ExpressionComposition(prepared)(axis) for axis in (u, v, w))
            == expected
        )

    independent = algebra.CoordinateEnclosureBudget(2_000_000, 81_920_000)
    with independent.activate():
        preparation = algebra.prepared_reference_arguments(element)
        assert independent.work_units > 0 and independent.retained_basis_bytes > 0
        assert (
            tuple(algebra.ExpressionComposition(preparation)(axis) for axis in (u, v, w))
            == expected
        )


def test_immutable_support_profiles_preserve_actual_denominators_and_source_mutation() -> (
    None
):
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra

    source: algebra.Polynomial = {
        (0, 0): Fraction(17, 6),
        (1, 0): Fraction(-19, 10),
        (0, 1): Fraction(23, 15),
    }
    original = dict(source)
    current: algebra.Polynomial = {(0, 0): Fraction(1, 17)}
    ledger = algebra.CoordinateEnclosureBudget(100_000, 10_000_000)
    with ledger.activate():
        prepared = algebra.PreparedPolynomialSupport(source)
        assert algebra._coefficient_profile((prepared,), exact_denominator=True) == (
            5,
            30,
        )
        for exact in (False, True):
            assert algebra._coefficient_profile(
                (current, prepared), exact_denominator=exact
            ) == (
                algebra._coefficient_profile((current, original), exact_denominator=exact)
            )
        before = ledger.work_units
        algebra._coefficient_profile((prepared,))
        reused_work = ledger.work_units - before
        before = ledger.work_units
        algebra._coefficient_profile((original,))
        assert 0 < reused_work < ledger.work_units - before
        source[(0, 0)] = Fraction(1, 2**70)
        assert dict(prepared) == original
        assert algebra._coefficient_profile((prepared,), exact_denominator=True) == (
            5,
            30,
        )
        assert algebra._coefficient_profile((source,), exact_denominator=True) != (5, 30)

    independent = algebra.CoordinateEnclosureBudget(100_000, 10_000_000)
    with independent.activate():
        prepared = algebra.PreparedPolynomialSupport(original)
        assert independent.work_units > 0 and independent.retained_basis_bytes > 0
        assert algebra._coefficient_profile((prepared,), exact_denominator=True) == (
            5,
            30,
        )


def test_composition_accumulator_profiles_preserve_coprime_exact_reduction_order() -> (
    None
):
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra

    u, v = algebra.axes(2)
    arguments = (algebra.add(u, v), algebra.add(u, algebra.scale(v, -1)))
    polynomial: algebra.Polynomial = {
        (0, 0): Fraction(1, 17),
        (1, 0): Fraction(-1, 19),
        (0, 1): Fraction(1, 23),
        (2, 0): Fraction(1, 29),
        (1, 1): Fraction(-1, 31),
        (0, 2): Fraction(1, 37),
    }
    expected = algebra.compose(polynomial, arguments)
    ledger = algebra.CoordinateEnclosureBudget(100_000, 10_000_000)
    with ledger.activate():
        actual = algebra.compose(polynomial, arguments)
        assert tuple(actual.items()) == tuple(expected.items())
        assert algebra._coefficient_profile((actual,), exact_denominator=True) == (
            algebra._coefficient_profile((expected,), exact_denominator=True)
        )
        actual[(0, 0)] = Fraction(99)
        assert tuple(algebra.compose(polynomial, arguments).items()) == tuple(
            expected.items()
        )
    independent = algebra.CoordinateEnclosureBudget(100_000, 10_000_000)
    with independent.activate():
        assert tuple(algebra.compose(polynomial, arguments).items()) == tuple(
            expected.items()
        )
        assert independent.work_units > 0


def test_zero_reference_products_do_not_reprepare_a_nonempty_accumulator() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra

    u, v = algebra.axes(2)
    arguments = (algebra.add(u, v), u, {})
    nonzero: algebra.Polynomial = {(0, 0, 0): Fraction(1, 17), (1, 0, 0): Fraction(2, 19)}
    with_zero = {**nonzero, (12, 8, 1): Fraction(5, 7), (9, 7, 3): Fraction(-11, 13)}
    ledgers = tuple(
        algebra.CoordinateEnclosureBudget(100_000, 10_000_000) for _ in range(2)
    )
    results = []
    for ledger, polynomial in zip(ledgers, (nonzero, with_zero), strict=True):
        with ledger.activate():
            results.append(algebra.ArgumentComposition(arguments)(polynomial))
    assert tuple(results[0].items()) == tuple(results[1].items())
    assert ledgers[0].work_units == ledgers[1].work_units > 0
    assert ledgers[0].retained_basis_bytes == ledgers[1].retained_basis_bytes > 0


def test_sibling_corner_operations_reuse_only_identical_complete_source_inputs() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._cell_geometry import (
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )

    source = coordinate_lagrange_element("hexahedron", 1)
    nodes = np.asarray(source.reference_nodes, dtype=np.float64)

    def half(origin: float) -> Any:
        points = nodes / 2 + np.asarray((origin, 0.0, 0.0))
        exact = tuple(tuple(Fraction(float(value)) for value in row) for row in points)
        return PolynomialComposedCellGeometryElement(
            source,
            source,
            np.asarray(
                [[value.numerator for value in row] for row in exact], dtype=np.int64
            ),
            np.asarray(
                [[value.denominator for value in row] for row in exact], dtype=np.uint64
            ),
        )

    first, second = half(0.0), half(0.5)
    bank = tuple(tuple(Fraction(float(value)) for value in row) for row in nodes)
    expected_first = algebra.coordinate_corner_images(first, bank)
    expected_second = algebra.coordinate_corner_images(second, bank)
    assert expected_first is not None and expected_second is not None
    ledger = algebra.CoordinateEnclosureBudget(1_000_000, 64 * 1024**2)
    with ledger.activate():
        assert algebra.coordinate_corner_images(first, bank) == expected_first
        before = ledger.work_units
        assert algebra.coordinate_corner_images(second, bank) == expected_second
        reused_work = ledger.work_units - before
        moved = tuple((row[0] + Fraction(1, 17), *row[1:]) for row in bank)
        expected_moved = tuple(
            (row[0] + Fraction(1, 17), *row[1:]) for row in expected_second
        )
        assert algebra.coordinate_corner_images(second, moved) == expected_moved
        assert algebra.coordinate_corner_images(second, bank) == expected_second

    independent = algebra.CoordinateEnclosureBudget(1_000_000, 64 * 1024**2)
    with independent.activate():
        assert algebra.coordinate_corner_images(second, bank) == expected_second
    assert independent.work_units > reused_work > 0
    assert independent.retained_basis_bytes > 0


def test_prepared_coverage_refusal_transports_the_actual_negative_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.meshing import _certification
    from phydrax.meshing._certification import _MeshCertificationPremiseFailure

    mesh, geometry, audit, positive = _prepared_tetrahedral_acceptance()
    original = positive.request.domain
    assert original is not None
    domain = phx.geometry.PiecewiseLinearDomain(
        _TETRAHEDRON + np.asarray((2.0, 0.0, 0.0)),
        original.facets,
        original.facet_regions,
        original.region_ids,
        source_id="independent-shifted-coverage-source",
    )
    actual_calls = []
    theorem = _certification.certify_domain_coverage

    def coverage(*args: Any, **kwargs: Any) -> Any:
        result = theorem(*args, **kwargs)
        actual_calls.append(result)
        return result

    monkeypatch.setattr(_certification, "certify_domain_coverage", coverage)
    ledger = algebra.CoordinateEnclosureBudget(2_000_000, 81_920_000)
    with ledger.activate(), pytest.raises(_MeshCertificationPremiseFailure) as caught:
        phx.meshing.MeshCertificationPreparedEvidence(
            mesh,
            geometry,
            audit.validity,
            schedule=positive.request.schedule,
            domain=domain,
            cell_regions=np.asarray((0,), dtype=np.int64),
            embedding=positive.embedding,
            limits=positive.request.limits,
        )
    assert len(actual_calls) == 1  # No second theorem for failure reporting.
    failure = caught.value
    certificate = failure.coverage
    assert certificate.certificate_id == actual_calls[0].certificate_id
    assert certificate.status == "violated" and certificate.findings
    assert certificate.binding.source_id == domain.source_id
    failure_domain = failure.request.domain
    if failure_domain is None or failure.ledger is None:
        raise RuntimeError("Prepared coverage failure lost its domain or active ledger.")
    assert failure_domain.domain_id == domain.domain_id
    assert failure.geometry.geometry_layout_id == geometry.geometry_layout_id
    assert failure.mesh.topology_id == mesh.topology_id
    assert failure.ledger.work_units == ledger.work_units
    assert failure.evidence.provider_code == certificate.certificate_id
    assert failure.evidence.category is phx.meshing.MeshingFailureCategory.AUDIT_FAILED
    assert failure.evidence.stage == "certification"
    assert all(
        finding.check in failure.evidence.message
        for finding in certificate.findings
        if finding.status == certificate.status
    )
    for finding in certificate.findings:
        prefix = f"certificate:{certificate.certificate_id}:finding:{finding.finding_id}:{finding.check}"
        if finding.resource is not None:
            prefix += f":resource:{finding.resource}"
        assert all(
            dict(failure.evidence.requested)[f"{prefix}:{key}"] == value
            for key, value in finding.requested
        )
        assert all(
            dict(failure.evidence.achieved)[f"{prefix}:{key}"] == value
            for key, value in finding.achieved
        )


def test_uncertain_prepared_refusal_keeps_unbounded_values_out_of_finite_evidence() -> (
    None
):
    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.meshing._certification import _MeshCertificationPremiseFailure

    mesh, geometry, audit, positive = _prepared_tetrahedral_acceptance()
    request = positive.request
    assert request.domain is not None and request.cell_regions is not None
    ledger = algebra.CoordinateEnclosureBudget(0, 81_920_000)
    with ledger.activate():
        coverage = phx.geometry.certify_domain_coverage(
            mesh,
            geometry,
            request.domain,
            request.cell_regions,
            embedding=positive.embedding,
            limits=request.limits,
        )
        assert coverage.status == "unresolved"
        assert any(
            value is None or not np.isfinite(value)
            for value in coverage.achieved_region_measures
        )
        with pytest.raises(_MeshCertificationPremiseFailure) as caught:
            raise _MeshCertificationPremiseFailure(
                mesh, geometry, request, audit.validity, positive.embedding, coverage
            )
    failure = caught.value
    assert failure.coverage.certificate_id == coverage.certificate_id
    assert failure.coverage.achieved_region_measures == coverage.achieved_region_measures
    assert failure.unbounded_region_measures == tuple(
        (region, "achieved", measured)
        for region, measured in zip(
            coverage.region_ids, coverage.achieved_region_measures, strict=True
        )
        if measured is None or not np.isfinite(measured)
    )
    assert all(
        np.isfinite(value)
        for _, value in (*failure.evidence.requested, *failure.evidence.achieved)
    )
    assert all(
        f"region_measure[{region}]" not in dict(failure.evidence.achieved)
        for region, measured in zip(
            coverage.region_ids, coverage.achieved_region_measures, strict=True
        )
        if measured is None or not np.isfinite(measured)
    )
    assert failure.evidence.evidence_id
    for finding in coverage.findings:
        prefix = f"certificate:{coverage.certificate_id}:finding:{finding.finding_id}:{finding.check}"
        if finding.resource is not None:
            prefix += f":resource:{finding.resource}"
        assert all(
            dict(failure.evidence.requested)[f"{prefix}:{key}"] == value
            for key, value in finding.requested
            if np.isfinite(value)
        )
        assert all(
            dict(failure.evidence.achieved)[f"{prefix}:{key}"] == value
            for key, value in finding.achieved
            if np.isfinite(value)
        )


@pytest.mark.parametrize("resource", ("coefficient_work", "polynomial_storage"))
def test_resumed_coordinate_admission_preserves_already_completed_quantities(
    resource: str,
) -> None:
    from phydrax.discretization._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    ledger = CoordinateEnclosureBudget(100, 1024)
    ledger.reserve(12, 64)
    with pytest.raises(CoordinateEnclosureResourceError) as caught:
        with ledger.bound_stage(
            5 if resource == "coefficient_work" else 100,
            32 if resource == "polynomial_storage" else 1024,
            starting_work_units=0,
        ):
            raise AssertionError(
                "An inadmissible completed prefix cannot enter the stage."
            )
    failure = caught.value
    assert failure.resource == resource and failure.admission
    assert (
        failure.requested
        == failure.completed
        == (12 if resource == "coefficient_work" else 64)
    )
    assert failure.limit == (5 if resource == "coefficient_work" else 32)
    assert ledger.work_units == 12 and ledger.temporary_bytes_upper == 64


def test_coordinate_refusal_metadata_rejects_false_preoperation_quantities() -> None:
    from phydrax.discretization._coordinate_enclosure import (
        CoordinateEnclosureResourceError,
    )

    for limit, requested, completed in (
        (10, 10, 9),
        (10, 11, 12),
        (10, 11, 11),
        (-1, 2, 0),
    ):
        with pytest.raises(ValueError):
            CoordinateEnclosureResourceError(
                "coefficient_work", limit, requested, completed
            )
    with pytest.raises(TypeError):
        CoordinateEnclosureResourceError("coefficient_work", 10, 11, True)
    failure = CoordinateEnclosureResourceError("coefficient_work", 10, 11, 9)
    assert (failure.limit, failure.requested, failure.completed) == (10, 11, 9)
    assert not failure.admission


def test_native_work_refusal_is_not_relabelled_as_coordinate_storage() -> None:
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionBudget
    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget

    ledger = CoordinateEnclosureBudget(100, 8192)
    native = NativeExecutionBudget(
        max_work=1,
        max_geometry_queries=1,
        max_cavity_cells=1,
        max_scratch_bytes=8192,
        max_wall_seconds=120,
    )
    with pytest.raises(MeshcoreError) as caught:
        with native, ledger.activate():
            ledger.reserve(0, 128)
            with pytest.raises(MeshcoreError):
                native.charge(work=2)
            ledger.reserve(0, 1)
    assert caught.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert "execution_admit_work_bound" in str(caught.value)
    assert (
        native.evidence is not None
        and native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    )
    assert (
        caught.value.work_evidence is not None
        and caught.value.memory_evidence is not None
    )
    assert ledger.work_units == 0 and ledger.temporary_bytes_upper == 128


def test_public_failure_preserves_actual_fidelity_resource_quantities() -> None:
    from phydrax.meshing._certification import _MeshCertificationReportFailure

    mesh = _octahedron()
    geometry, audit = _plc_audit(
        mesh, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    sphere = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="sphere").compile()
    source = phx.geometry.ImplicitBoundarySource(
        sphere, source_id="limited-fidelity-sphere", spacing=0.2
    )
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("surface"),
        source=source,
        fidelity_tolerance=0.8,
        limits=phx.geometry.MeshCertificateLimits(maximum_distance_evaluations=1),
    )
    assert report.fidelity is not None and report.fidelity.status == "unresolved"
    assert "distance_capacity" in _checks(report.fidelity, "unresolved")
    with pytest.raises(_MeshCertificationReportFailure) as caught:
        report.require_passed()
    failure = caught.value
    assert failure.report.report_id == report.report_id
    assert failure.report.request.request_id == report.request.request_id
    failed_fidelity = failure.report.fidelity
    if failed_fidelity is None:
        raise RuntimeError("Fidelity failure lost its actual source certificate.")
    assert failed_fidelity.binding.source_id == source.source_id
    assert all(
        np.isfinite(value)
        for _, value in (*failure.evidence.requested, *failure.evidence.achieved)
    )
    for certificate in (
        report.embedding,
        report.coverage,
        report.fidelity,
        *report.scoped_fidelity,
    ):
        if certificate is None:
            continue
        for finding in certificate.findings:
            prefix = f"certificate:{certificate.certificate_id}:finding:{finding.finding_id}:{finding.check}"
            if finding.resource is not None:
                prefix += f":resource:{finding.resource}"
            assert all(
                dict(failure.evidence.requested)[f"{prefix}:{key}"] == value
                for key, value in finding.requested
                if np.isfinite(value)
            )
            assert all(
                dict(failure.evidence.achieved)[f"{prefix}:{key}"] == value
                for key, value in finding.achieved
                if np.isfinite(value)
            )
            assert all(
                dict(failure.evidence.achieved)[f"{prefix}:observation:{key}"] == value
                for key, value in finding.observations
                if np.isfinite(value)
            )
    assert len(dict(failure.evidence.requested)) == len(failure.evidence.requested)
    assert len(dict(failure.evidence.achieved)) == len(failure.evidence.achieved)


def test_coordinate_reserve_admits_actual_predecessor_and_unpaid_native_debt() -> None:
    from fractions import Fraction

    from phydrax._meshcore import (
        exact_orient2d,
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionBudget,
    )
    from phydrax.discretization._coordinate_enclosure import (
        _coefficient_profile,
        CoordinateEnclosureBudget,
    )

    host_terms: dict[tuple[int, ...], Fraction] = {
        (index,): Fraction(index + 1, 17) for index in range(15)
    }

    first = np.zeros((60, 2), dtype=np.float64)
    second = np.broadcast_to(np.asarray((1.0, 0.0)), first.shape)
    third = np.broadcast_to(np.asarray((0.0, 1.0)), first.shape)
    predecessor = NativeExecutionBudget(
        max_work=1000,
        max_geometry_queries=1000,
        max_cavity_cells=100,
        max_scratch_bytes=1024**2,
        max_wall_seconds=120,
    )
    with predecessor:
        assert np.all(exact_orient2d(first[:20], second[:20], third[:20]) > 0)
    prior = predecessor.evidence
    assert prior is not None and prior.status == MeshcoreStatus.OK
    ledger = CoordinateEnclosureBudget(1000, 1024**2)
    native = NativeExecutionBudget(
        max_work=100,
        max_geometry_queries=1000,
        max_cavity_cells=100,
        max_scratch_bytes=1024**2,
        max_wall_seconds=120,
    )
    with pytest.raises(MeshcoreError) as caught:
        with native:
            native.import_preparation(
                work=int(prior.work_evidence[0]),
                geometry_queries=int(prior.work_evidence[1]),
                elapsed_seconds=prior.elapsed_seconds,
            )
            with ledger.activate():
                assert _coefficient_profile((host_terms,)) == (4, 5)
                assert np.all(exact_orient2d(first, second, third) > 0)
                remaining = native.remaining().remaining_work_units
                assert ledger.work_units - ledger.native_charged_work_units <= remaining
                ledger.reserve(
                    remaining - ledger.work_units + ledger.native_charged_work_units + 1
                )
    assert caught.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert "execution_admit_work_bound" in str(caught.value)
    assert ledger.work_units == 15 and ledger.native_charged_work_units == 0
    assert (
        native.evidence is not None
        and native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    )
    assert caught.value.work_evidence is not None
    assert int(caught.value.work_evidence[0]) == 100 - remaining


def test_coordinate_live_release_does_not_mask_a_primary_callback_error() -> None:
    from phydrax._meshcore import MeshcoreError, NativeExecutionBudget
    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget

    ledger = CoordinateEnclosureBudget(100, 8192)
    native = NativeExecutionBudget(
        max_work=1,
        max_geometry_queries=1,
        max_cavity_cells=1,
        max_scratch_bytes=8192,
        max_wall_seconds=120,
    )
    with pytest.raises(ValueError, match="original callback provenance") as caught:
        with native, ledger.activate(), ledger.live_storage() as owner:
            owner.set_bound(128)
            with pytest.raises(MeshcoreError):
                native.charge(work=2)
            raise ValueError("original callback provenance")
    assert any(
        "Coordinate live-storage release also refused" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert native.evidence is not None
    assert ledger.work_units == 0
