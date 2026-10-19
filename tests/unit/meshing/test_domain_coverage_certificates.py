#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer-visible domain coverage certificates contracts."""

import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import _coordinate_enclosure as algebra
from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._domain_coverage import (
    box_jacobian_measure,
    mapped_coverage_measures,
)
from phydrax.geometry._embedding_state import EmbeddingState
from phydrax.geometry._mapped_coverage import (
    chart_flux,
    face_charts,
    face_flux,
    integrate,
)
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._planar_coverage import _group_intersections, source_groups
from phydrax.geometry._planar_facet_scope import certify_planar_facet_source_scope
from phydrax.geometry._planar_primitives import intersection
from tests._support.domain_coverage import (
    mapped_split_square,
    split_square_domain,
    SPLIT_SQUARE_POINTS,
)
from tests._support.mesh_certificate_fixtures import (
    coordinate_budget as coordinate_budget,
    coverage,
    cube_with_independent_boundary_triangulation,
    curved_shared_hex_interface,
    embedding,
    halved_tapered_restriction,
    independent_mapped_shear_domain,
    mapped_reference_domain,
    mixed_degree_hexes,
    reference_budget as reference_budget,
    SPLIT_SQUARE_CELLS,
    split_square_variant,
)
from tests._support.meshing_fixtures import finding_checks


def test_conforming_two_region_mesh_covers_domain_and_interface_exactly() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    certificate = coverage(mesh, np.asarray((0, 0, 1, 1)))

    assert certificate.status == "certified"
    assert certificate.requested_region_measures == (0.5, 0.5)
    assert certificate.achieved_region_measures == (0.5, 0.5)
    assert certificate.covered_source_facet_count == certificate.source_facet_count


def test_omitted_material_interface_is_a_coverage_violation() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    certificate = coverage(mesh, np.asarray((0, 0, 0, 0)))

    violated = finding_checks(certificate, "violated")
    assert certificate.status == "violated"
    assert {"omitted_interface", "region_measure"} <= violated
    interface = next(v for v in certificate.findings if v.check == "omitted_interface")
    assert interface.entity_kind == "source_facet"
    assert interface.entity_ids == (6,)
    assert certificate.achieved_region_measures == (1.0, 0.0)


def test_gap_leaves_unmatched_facets_and_a_measure_deficit() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS[:3]
    )
    certificate = coverage(mesh, np.asarray((0, 0, 1)))

    assert certificate.status == "violated"
    assert {"unmatched_boundary_facet", "uncovered_boundary", "region_measure"} <= (
        finding_checks(certificate, "violated")
    )
    assert certificate.achieved_region_measures == (0.5, 0.25)


def test_double_coverage_is_detected_by_embedding_and_measure() -> None:
    points = np.concatenate((SPLIT_SQUARE_POINTS, ((0.1, 0.1), (0.4, 0.1), (0.1, 0.4))))
    cells = np.concatenate((SPLIT_SQUARE_CELLS, ((6, 7, 8),)))
    mesh = phx.discretization.CellMesh.from_triangles(points, cells)
    geometry, _, embedding_certificate = embedding(mesh)
    certificate = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        split_square_domain(),
        np.asarray((0, 0, 1, 1, 0)),
        embedding=embedding_certificate,
    )

    assert embedding_certificate.status == "violated"
    assert certificate.status == "violated"
    assert "embedding_premise" in finding_checks(certificate, "unresolved")
    assert certificate.achieved_region_measures[0] is not None
    assert (
        certificate.achieved_region_measures[0] > certificate.requested_region_measures[0]
    )


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

    mesh, geometry, domain = mapped_reference_domain(kind)
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
    mesh, geometry, domain = mapped_reference_domain("hexahedron", curved=True)
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


def test_curved_two_region_interiors_cover_planar_material_interface() -> None:
    mesh, geometry, embedding = mapped_split_square()
    certificate = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        split_square_domain(),
        np.asarray((0, 1), dtype=np.int64),
        embedding=embedding,
    )

    assert certificate.status == "certified"
    assert certificate.achieved_region_measures == (0.5, 0.5)
    assert certificate.integration_error_bounds == (0.0, 0.0)
    assert certificate.covered_source_facet_count == 7


def test_mapped_coverage_reports_and_caps_its_actual_subdivision_work() -> None:
    mesh, geometry, embedding = mapped_split_square()
    domain = split_square_domain()
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
    assert "mapped_facet_containment" in finding_checks(capped, "unresolved")
    # The refused over-budget piece is charged, not hidden under the cap.
    assert capped.subdivision_piece_count > limits.maximum_subdivision_pieces


def test_mapped_coverage_charges_its_expression_algebra_to_the_request_ledger() -> None:
    mesh, geometry, embedding = mapped_split_square()
    domain = split_square_domain()
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
    assert "mapped_coverage_resource_budget" in finding_checks(capped, "unresolved")
    assert capped.source_expression_work_units is not None
    assert capped.source_expression_work_units <= limits.maximum_work_units
    assert capped.achieved_region_measures == (None, None)


@pytest.mark.parametrize("reversed_facets", (False, True))
def test_source_facet_partition_consumes_its_coverage_budget(
    reversed_facets: bool,
) -> None:
    """Independent source refinement preserves coverage but is not free proof work."""
    mesh, geometry, domain = mapped_reference_domain("hexahedron")
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
    assert "mapped_coverage_resource_budget" in finding_checks(capped, "unresolved")
    assert capped.binding.source_revision == partition.source_revision


def test_absent_region_integrals_are_unbounded_not_certified() -> None:
    mesh, geometry, embedding = mapped_split_square()
    domain = split_square_domain()
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
            measures=phx.geometry.CoverageMeasures(
                requested=certified.requested_region_measures,
                achieved=(None, certified.achieved_region_measures[1]),
                requested_bounds=certified.requested_region_measure_bounds,
                achieved_bounds=bounds,
                integration_errors=errors,
            ),
            evidence=phx.geometry.CoverageEvidence(
                covered_source_facets=certified.covered_source_facet_count,
                facet_source_overlaps=certified.facet_source_overlaps,
                premise_certificate_ids=certified.premise_certificate_ids,
                candidate_pairs=certified.candidate_pair_count,
                subdivision_pieces=certified.subdivision_piece_count,
                maximum_subdivision_depth=certified.maximum_subdivision_depth_reached,
            ),
            expression_work=phx.geometry.CoverageExpressionWork(
                charged=certified.source_expression_work_units,
                required=certified.source_expression_required_work_units,
                peak_bytes=certified.source_expression_peak_bytes,
            ),
        )

    present = certified.achieved_region_measure_bounds[1]
    absent = rebuild((finding,), (None, present), (None, 0.0))
    assert absent.status == "unresolved"
    assert absent.achieved_region_measure_bounds[0] is None
    assert absent.integration_error_bounds[0] is None
    with pytest.raises(ValueError, match="DomainCoverageCertificate.measures.achieved"):
        rebuild((finding,), ((-math.inf, math.inf), present), (math.inf, 0.0))
    with pytest.raises(ValueError, match="DomainCoverageCertificate.measures.achieved"):
        rebuild((), (None, present), (None, 0.0))


def test_mapped_coverage_refuses_missing_material_interface() -> None:
    mesh, geometry, embedding = mapped_split_square()
    certificate = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        split_square_domain(),
        np.asarray((0, 0), dtype=np.int64),
        embedding=embedding,
    )

    assert certificate.status == "violated"
    assert {"omitted_interface", "region_measure"} <= finding_checks(
        certificate, "violated"
    )
    assert certificate.achieved_region_measures == (1.0, 0.0)


def test_mapped_coverage_refuses_missing_region() -> None:
    mesh, geometry, embedding = mapped_split_square(cells=1)
    certificate = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        split_square_domain(),
        np.asarray((0,), dtype=np.int64),
        embedding=embedding,
    )

    assert certificate.status == "violated"
    assert {
        "uncovered_boundary",
        "omitted_interface",
        "region_measure",
    } <= finding_checks(certificate, "violated")
    assert certificate.achieved_region_measures == (0.5, 0.0)


def test_planar_corners_do_not_certify_a_nonplanar_mapped_boundary() -> None:
    mesh, geometry, domain = mapped_reference_domain("hexahedron")
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
    assert "unmatched_boundary_facet" in finding_checks(certificate, "violated")


def test_mapped_containment_capacity_cannot_be_reported_as_coverage() -> None:
    mesh, geometry, domain = mapped_reference_domain("hexahedron")
    coordinates = np.asarray(geometry.coordinates, dtype=np.float64).copy()
    face_center = np.all(
        coordinates == np.asarray((0.5, 0.5, 0.0), dtype=np.float64), axis=1
    )
    # Keep the facet in its declared plane but make its parameterization
    # quadratic, so one Bernstein node cannot prove containment.
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
    assert "mapped_facet_containment" in finding_checks(certificate, "unresolved")
    assert certificate.covered_source_facet_count < domain.facets.shape[0]
    assert certificate.achieved_region_measures == (1.0,)
    assert certificate.integration_error_bounds == (0.0,)


@pytest.mark.parametrize("mapped", [False, True], ids=["affine-clipping", "mapped-union"])
def test_independent_source_diagonals_do_not_constrain_target_facets(
    mapped: bool,
) -> None:
    mesh, geometry, domain = cube_with_independent_boundary_triangulation(mapped)
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
def test_overlapping_declared_source_fragments_are_not_a_union_proof(
    mapped: bool,
) -> None:
    mesh, geometry, source = cube_with_independent_boundary_triangulation(mapped)
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
    assert "overlapping_source_facets" in finding_checks(certificate, "violated")


def test_exact_root_restrictions_cover_an_independent_nonplanar_mapped_source() -> None:
    mesh, geometry, domain = independent_mapped_shear_domain()
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
    mesh, geometry, domain = independent_mapped_shear_domain()
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
    assert "mapped_domain_root_expression" in finding_checks(certificate, "violated")


def test_matching_mapped_coefficients_without_scientific_root_identity_remain_unresolved() -> (
    None
):
    mesh, geometry, domain = independent_mapped_shear_domain()
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
    assert "mapped_domain_restriction_source" in finding_checks(certificate, "unresolved")


def test_mapped_root_partition_gap_is_not_filled_by_copying_source_measure() -> None:
    mesh, geometry, domain = independent_mapped_shear_domain()
    mesh = phx.discretization.CellMesh(
        mesh.coordinates, (mesh.blocks[0],), vertex_global_ids=mesh.vertex_global_ids
    )
    record = geometry.restriction_source
    if record is None:
        raise RuntimeError("fixture.source_record must exist for restriction; got None")

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
    assert "mapped_domain_root_coverage_premise" in finding_checks(
        certificate, "violated"
    )
    assert certificate.requested_region_measures == (1.0,)
    assert certificate.achieved_region_measures == (0.5,)
    assert certificate.achieved_region_measure_bounds == ((0.5, 0.5),)
    assert certificate.integration_error_bounds == (0.0,)


def test_mapped_image_entity_identity_binds_actual_source_map_not_reference_coincidence() -> (
    None
):

    _, _, domain = independent_mapped_shear_domain()
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
    with pytest.raises(ValueError, match="identify a reference entity"):
        domain.image_entity_id(3, 1)


@pytest.mark.parametrize("mapped", [False, True], ids=["affine", "curved-interior"])
def test_selected_planar_source_scope_proves_complete_fragment_union(
    mapped: bool,
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, domain = cube_with_independent_boundary_triangulation(mapped)
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
        budget=reference_budget,
    )

    assert certificate.status == "certified"
    assert certificate.distance_upper == 0.0
    assert tuple(
        (row[2], row[3]) for row in certificate.source_fragment_projected_measures
    ) == ((1, 2), (1, 2))


@pytest.mark.parametrize("mapped", [False, True], ids=["affine", "curved-interior"])
def test_whole_domain_coverage_does_not_fill_a_selected_facet_scope_gap(
    mapped: bool,
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, domain = cube_with_independent_boundary_triangulation(mapped)
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
        budget=reference_budget,
    )

    assert coverage.status == "certified"
    assert certificate.status != "certified"
    assert certificate.distance_upper is None


def test_generated_caps_reject_negative_source_face_tags(
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, domain = cube_with_independent_boundary_triangulation(False)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    coverage = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0, 0), dtype=np.int64), embedding=embedding
    )
    with pytest.raises(ValueError, match="source_face_fragments"):
        certify_planar_facet_source_scope(
            mesh,
            geometry,
            domain,
            np.asarray((coverage.facet_source_overlaps[0][0],), dtype=np.int64),
            {-1: (0,)},
            source_face_entity_set_id="original-source-faces",
            coverage=coverage,
            budget=reference_budget,
        )


def test_restricted_chart_measures_reuse_the_exact_root_determinant(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    mesh, geometry = halved_tapered_restriction()
    regions = np.asarray((0, 1, 0, 1), dtype=np.int64)
    ledger = coordinate_budget(1_000_000, 64 * 1024**2)
    state = EmbeddingState([], [])
    with ledger.activate():
        totals, _ = mapped_coverage_measures(
            state, mesh, geometry, regions, budget=ledger
        )
    # Each half-x chart has volume (1/2) * integral_0^1 (1-z/2) dz = 3/8; each region owns two charts.
    expected = (Fraction(3, 4), Fraction(3, 4))

    assert state.findings == []
    assert totals == tuple(expected)
    assert ledger.work_units > 0


def test_affine_tetrahedral_restriction_measures_use_constant_jacobians(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    element = coordinate_lagrange_element("tetrahedron", 1)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    shear = np.asarray(((1.0, 0.25, 0.0), (0.0, 1.5, 0.125), (0.0, 0.0, 0.75)))
    controls = nodes @ shear.T + np.asarray((0.5, -0.25, 1.0))
    # Two affine charts: the corner subtetrahedron and a sheared interior one.
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
    ledger = coordinate_budget(1_000_000, 64 * 1024**2)
    state = EmbeddingState([], [])
    with ledger.activate():
        totals, _ = mapped_coverage_measures(
            state, mesh, geometry, regions, budget=ledger
        )
    # The triangular physical shear has determinant 9/8; chart determinants are 1/8 and 1/64, and the reference tetrahedron has volume 1/6.
    expected = (Fraction(3, 128), Fraction(3, 1024))

    assert state.findings == []
    assert totals == tuple(expected)
    assert all(value > Fraction(0) for value in expected)
    assert ledger.work_units > 0


def test_mapped_coverage_is_invariant_under_equivalent_source_facet_orientation() -> None:
    mesh, geometry, embedding = mapped_split_square()
    regions = np.asarray((0, 1), dtype=np.int64)
    # y = 0 and y = 1 each carry two same-plane groups of different region pairs;
    # x = 0, 0.5 and 1 are distinct parallel planes.
    declared, reversed_ = (
        phx.geometry.certify_domain_coverage(
            mesh,
            geometry,
            split_square_variant(
                SPLIT_SQUARE_POINTS, reversed_facets=flag, source_id="split-square"
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
    mesh, geometry, embedding = mapped_split_square()
    vertices = SPLIT_SQUARE_POINTS.copy()
    vertices[[2, 3], 0] = 1.25
    domain = split_square_variant(
        vertices, reversed_facets=False, source_id="stretched-split-square"
    )
    certificate = phx.geometry.certify_domain_coverage(
        mesh, geometry, domain, np.asarray((0, 1), dtype=np.int64), embedding=embedding
    )

    assert certificate.status == "violated"
    assert {"unmatched_boundary_facet", "uncovered_boundary"} <= finding_checks(
        certificate, "violated"
    )
    # The mesh edge x = 1 is parallel to the declared x = 1.25 facet, never covering it.
    assert 2 not in {row[1] for row in certificate.facet_source_overlaps}
    assert certificate.achieved_region_measures == (0.5, 0.5)
    assert certificate.requested_region_measures == (0.5, 0.75)


def test_curved_boundary_charts_match_no_declared_plane_and_keep_their_exact_measure() -> (
    None
):

    bulge = Fraction(1, 8)
    mesh, geometry, embedding = mapped_split_square(right_bulge=float(bulge))
    certificate = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        split_square_domain(),
        np.asarray((0, 1), dtype=np.int64),
        embedding=embedding,
    )

    assert embedding.status == "certified"
    assert certificate.status == "violated"
    assert "unmatched_boundary_facet" in finding_checks(certificate, "violated")
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
    kind: str,
    order: int,
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    element = coordinate_lagrange_element(kind, order)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    rng = np.random.default_rng(31)
    controls = nodes + rng.uniform(-0.1, 0.1, nodes.shape)
    local = tuple(tuple(Fraction(float(value)) for value in row) for row in controls)
    coordinates = algebra.coordinate_polynomials(element, local, budget=reference_budget)
    assert coordinates is not None
    dimension = nodes.shape[1]
    jacobian = tuple(
        tuple(
            algebra.derivative(value, axis, budget=reference_budget)
            for axis in range(dimension)
        )
        for value in coordinates
    )

    assert box_jacobian_measure(coordinates, budget=reference_budget) == integrate(
        algebra.determinant(jacobian, budget=reference_budget),
        "box",
        dimension,
        budget=reference_budget,
    )


@pytest.mark.parametrize("degrees", ((1, 1), (2, 3)), ids=("bilinear", "higher-degree"))
@pytest.mark.parametrize("reverse", (False, True), ids=("outward", "reversed"))
def test_full_nonplanar_polynomial_chart_flux_has_exact_signed_integral(
    degrees: tuple[int, int],
    reverse: bool,
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    first, second = algebra.axes(2, budget=reference_budget)
    p, q = degrees
    height = algebra.add(
        algebra.constant(1, 2, budget=reference_budget),
        algebra.multiply(
            algebra.power(first, p, 2, budget=reference_budget),
            algebra.power(second, q, 2, budget=reference_budget),
            budget=reference_budget,
        ),
        budget=reference_budget,
    )
    coordinates = (second, first, height) if reverse else (first, second, height)
    # X=(u,v,1+u^p v^q): X dot (Xu cross Xv)=1+(1-p-q)u^p v^q.
    expected = (Fraction(1) + Fraction(1 - p - q, (p + 1) * (q + 1))) / 3
    assert chart_flux(coordinates, "box", budget=reference_budget) == (
        -expected if reverse else expected
    )


@pytest.mark.parametrize("domain", ("box", "simplex"))
@pytest.mark.parametrize("height", (0, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_polynomial_flux_preserves_zero_and_translated_planar_source(
    domain: algebra.BernsteinDomain,
    height: int,
    reverse: bool,
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    u, v = algebra.axes(2, budget=reference_budget)
    first, second = (v, u) if reverse else (u, v)
    coordinates = (
        first,
        algebra.constant(height, 2, budget=reference_budget),
        algebra.add(
            second,
            algebra.multiply(first, second, budget=reference_budget),
            budget=reference_budget,
        ),
    )
    # X=(u,h,v+uv): X dot (Xu cross Xv)=-h(1+u).
    # Integrating on the unit box/triangle is independent of flux preparation.
    expected = -Fraction(height) * (Fraction(1, 2) if domain == "box" else Fraction(2, 9))
    if reverse:
        expected = -expected
    for shift in range(3):
        rotated = coordinates[shift:] + coordinates[:shift]
        assert chart_flux(rotated, domain, budget=reference_budget) == expected


@pytest.mark.parametrize("split", (False, True), ids=("same-region", "two-materials"))
def test_full_curved_interface_region_flux_preserves_exact_physical_volumes(
    split: bool,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    mesh, geometry = curved_shared_hex_interface()
    before = np.asarray(geometry.coordinates).copy()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    regions = np.asarray((0, 1) if split else (0, 0), dtype=np.int64)
    ledger = coordinate_budget(2_000_000, 81_920_000)
    state = EmbeddingState([], [])
    with ledger.activate():
        embedding = phx.geometry.certify_global_embedding(
            mesh, geometry, validity, budget=ledger
        )
        assert validity.all_certified and embedding.status == "certified"
        totals, cells = mapped_coverage_measures(
            state,
            mesh,
            geometry,
            regions,
            embedding=embedding,
            budget=ledger,
        )
    # The Q2 interface is x=1+y(1-y)z(1-z)/8, not its corner plane x=1.
    expected = (Fraction(289, 288), Fraction(287, 288)) if split else (Fraction(2),)
    assert totals == expected and len(cells) == 2
    assert not state.findings
    np.testing.assert_array_equal(geometry.coordinates, before)


@pytest.mark.parametrize("dimension", (2, 3))
def test_full_barycentric_offsum_measure_uses_actual_physical_map(
    dimension: int,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

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
    state = EmbeddingState([], [])
    ledger = coordinate_budget(100_000, 1 << 24)
    with ledger.activate():
        totals, cells = mapped_coverage_measures(
            state,
            mesh,
            geometry,
            np.asarray((0,), dtype=np.int64),
            budget=ledger,
        )
    # J = I - delta * original_vertex_1 @ ones.T. Translation participates
    # because the coefficient action is not a partition-of-unity restriction.
    expected = (
        1 - delta * sum(Fraction(float(value)) for value in controls[1])
    ) / math.factorial(dimension)
    assert totals == (expected,) and len(cells) == 1
    assert totals != ((1 - delta) / math.factorial(dimension),)
    assert not state.findings
    np.testing.assert_array_equal(geometry.coordinates, controls)


def test_full_barycentric_incident_cells_keep_exact_shared_trace_and_measure(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

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
    ledger = coordinate_budget(100_000, 1 << 24)
    state = EmbeddingState([], [])
    with ledger.activate():
        embedding = phx.geometry.certify_global_embedding(
            mesh, geometry, validity, budget=ledger
        )
        assert validity.all_certified and embedding.status == "certified"
        totals, cells = mapped_coverage_measures(
            state,
            mesh,
            geometry,
            np.asarray((0, 0), dtype=np.int64),
            embedding=embedding,
            budget=ledger,
        )
    assert "mapped_affine_source_maps" in embedding.evaluated_checks
    assert "mapped_exact_shared_corner_incidence" in embedding.evaluated_checks
    assert totals == (expected,) and len(cells) == 2
    assert expected != Fraction(1, 2)
    assert not state.findings
    np.testing.assert_array_equal(geometry.coordinates, controls)


def test_unproved_interface_trace_retains_actual_signed_cell_integrals(
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry = mixed_degree_hexes(0.0625)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "violated"
    totals, _ = mapped_coverage_measures(
        EmbeddingState([], []),
        mesh,
        geometry,
        np.asarray((0, 0), dtype=np.int64),
        embedding=embedding,
        budget=reference_budget,
    )
    # A missing shared-face trace cannot cancel the unmatched Q2 face bubble.
    assert totals == (Fraction(2) - Fraction(4, 9) * Fraction(1, 16),)


def test_boundary_region_integral_rejects_changed_complete_source_binding(
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry = curved_shared_hex_interface()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    changed = np.asarray(geometry.coordinates).copy()
    changed[0, 0] += 0.125
    with pytest.raises(ValueError, match="MeshCertificateBinding.geometry"):
        mapped_coverage_measures(
            EmbeddingState([], []),
            mesh,
            geometry.with_coordinates(changed),
            np.asarray((0, 0), dtype=np.int64),
            embedding=embedding,
            budget=reference_budget,
        )


def test_exact_boundary_region_integral_refuses_below_actual_work_without_renewal(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    mesh, geometry = curved_shared_hex_interface()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    regions = np.asarray((0, 1), dtype=np.int64)
    measured = coordinate_budget(2_000_000, 81_920_000)
    with measured.activate():
        mapped_coverage_measures(
            EmbeddingState([], []),
            mesh,
            geometry,
            regions,
            embedding=embedding,
            budget=measured,
        )
    ledger = coordinate_budget(measured.work_units - 1, 81_920_000)
    before = np.asarray(geometry.coordinates).copy()
    with ledger.activate(), pytest.raises(CoordinateEnclosureResourceError) as refusal:
        mapped_coverage_measures(
            EmbeddingState([], []),
            mesh,
            geometry,
            regions,
            embedding=embedding,
            budget=ledger,
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
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    x, y, z = algebra.axes(3, budget=reference_budget)
    numerators = (
        algebra.add(
            x,
            algebra.multiply(
                algebra.multiply(x, y, budget=reference_budget),
                z,
                budget=reference_budget,
            ),
            budget=reference_budget,
        ),
        algebra.add(
            y,
            algebra.multiply(
                algebra.power(x, 2, 3, budget=reference_budget),
                z,
                budget=reference_budget,
            ),
            budget=reference_budget,
        ),
        algebra.add(
            z,
            algebra.multiply(
                x,
                algebra.power(y, 3, 3, budget=reference_budget),
                budget=reference_budget,
            ),
            budget=reference_budget,
        ),
    )
    denominator = algebra.add(
        algebra.constant(1, 3, budget=reference_budget),
        algebra.scale(
            algebra.multiply(x, z, budget=reference_budget),
            Fraction(1, 5),
            budget=reference_budget,
        ),
        budget=reference_budget,
    )
    coordinates: tuple[algebra.Expression, ...] = (
        tuple(
            algebra.rational_expression(value, denominator, budget=reference_budget)
            for value in numerators
        )
        if rational
        else numerators
    )
    ledger = coordinate_budget(2_000_000, 81_920_000)
    with ledger.activate(), ledger.temporary_scope():
        charts = face_charts(coordinates, kind, face, budget=ledger)
    corners = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)[
        np.asarray(face)
    ]
    pieces = (corners,) if len(face) == 3 else (corners[[0, 1, 2]], corners[[0, 2, 3]])
    # Seven rational nodes per axis determine the cross-multiplied degree-six
    # identity; this checks the polynomial functions, not a few corner samples.
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
                    tuple(
                        algebra.expression_evaluate(
                            value, (u, v), budget=reference_budget
                        )
                        for value in chart
                    )
                    == expected
                )
    assert 0 < ledger.work_units <= ledger.maximum_work_units


@pytest.mark.parametrize("rational", (False, True))
def test_oriented_face_flux_preserves_complete_source_and_triangle_integral(
    rational: bool,
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    x, y, z = algebra.axes(3, budget=reference_budget)
    coordinates = (
        algebra.add(
            x, algebra.multiply(y, z, budget=reference_budget), budget=reference_budget
        ),
        algebra.add(
            y,
            algebra.scale(
                algebra.multiply(x, z, budget=reference_budget),
                Fraction(1, 3),
                budget=reference_budget,
            ),
            budget=reference_budget,
        ),
        algebra.add(
            z,
            algebra.scale(
                algebra.multiply(x, y, budget=reference_budget),
                Fraction(1, 5),
                budget=reference_budget,
            ),
            budget=reference_budget,
        ),
    )
    if rational:
        # A rational map with planar faces, not endpoint-authored data.
        denominator = algebra.add(
            algebra.constant(1, 3, budget=reference_budget),
            algebra.scale(x, Fraction(1, 5), budget=reference_budget),
            budget=reference_budget,
        )
        coordinates = tuple(
            algebra.rational_expression(value, denominator, budget=reference_budget)
            for value in (x, y, z)
        )
    for face in reference_cell_topology("hexahedron").entities[2]:
        contributions = tuple(
            chart_flux(chart, domain, budget=reference_budget)
            for chart, domain in face_charts(
                coordinates, "hexahedron", face, budget=reference_budget
            )
        )
        expected: Fraction | None = Fraction(0)
        for contribution in contributions:
            if contribution is None:
                expected = None
                break
            expected += contribution
        actual = face_flux(coordinates, "hexahedron", face, budget=reference_budget)
        assert actual == expected
        if actual is not None:
            assert (
                face_flux(
                    coordinates,
                    "hexahedron",
                    tuple(reversed(face)),
                    budget=reference_budget,
                )
                == -actual
            )


@pytest.mark.parametrize("shift", (0, 1, 2, 4), ids=("overlap", "touch", "gap", "second"))
def test_source_union_intersections_expand_only_relevant_exact_images(
    shift: int,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    points = np.asarray(
        [
            (4 * cell + x, y, 0)
            for cell in range(8)
            for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))
        ],
        dtype=np.float64,
    )
    facets = np.asarray(
        [
            tuple(4 * cell + corner for corner in triangle)
            for cell in range(8)
            for triangle in ((0, 1, 2), (0, 2, 3))
        ],
        dtype=np.int64,
    )
    regions = np.tile(np.asarray((0, -1), dtype=np.int64), (facets.shape[0], 1))
    preparation = coordinate_budget(2_000_000, 81_920_000)
    groups, overlaps, _, exceeded = source_groups(
        points, facets, regions, 1024, budget=preparation
    )
    assert not exceeded and not overlaps and len(groups) == 1
    group = groups[0]
    polygon = tuple(
        (Fraction(shift) + x, y)
        for x, y in (
            (Fraction(0), Fraction(1, 4)),
            (Fraction(3, 4), Fraction(1, 4)),
            (Fraction(3, 4), Fraction(3, 4)),
            (Fraction(0), Fraction(3, 4)),
        )
    )
    limits = phx.geometry.MeshCertificateLimits()
    reduced = coordinate_budget(limits.maximum_work_units, limits.maximum_scratch_bytes)
    actual = _group_intersections(polygon, group, budget=reduced)
    expanded = coordinate_budget(limits.maximum_work_units, limits.maximum_scratch_bytes)
    expected = tuple(
        intersection(polygon, target, budget=expanded) for target in group.simplices
    )
    assert actual == expected
    assert sum(actual, Fraction(0)) == (
        Fraction(3, 8) if shift in (0, 4) else Fraction(0)
    )
    assert reduced.work_units < expanded.work_units


def test_constant_coordinate_flux_integrates_before_scalar_density_expansion(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    coordinates: tuple[algebra.Polynomial, ...] = (
        {(degree, 0): Fraction(1) for degree in range(1, 4)},
        {(0, degree): Fraction(1) for degree in range(1, 5)},
        {(0, 0): Fraction(7, 11)},
    )
    limits = phx.geometry.MeshCertificateLimits()
    reduced = coordinate_budget(limits.maximum_work_units, limits.maximum_scratch_bytes)
    actual = chart_flux(coordinates, "box", budget=reduced)
    expanded = coordinate_budget(limits.maximum_work_units, limits.maximum_scratch_bytes)
    tangents = tuple(
        tuple(algebra.derivative(value, axis, budget=expanded) for value in coordinates)
        for axis in range(2)
    )
    # Independent divergence formula: integrate X dot (Xu cross Xv) / 3.
    terms = []
    for axis, value in enumerate(coordinates):
        normal = algebra.add(
            algebra.multiply(
                tangents[0][(axis + 1) % 3],
                tangents[1][(axis + 2) % 3],
                budget=expanded,
            ),
            algebra.scale(
                algebra.multiply(
                    tangents[0][(axis + 2) % 3],
                    tangents[1][(axis + 1) % 3],
                    budget=expanded,
                ),
                -1,
                budget=expanded,
            ),
            budget=expanded,
        )
        terms.append(algebra.multiply(value, normal, budget=expanded))
    density = algebra.sum_polynomials(tuple(terms), budget=expanded)
    expected = integrate(density, "box", 2, budget=expanded) / 3
    # The parameter rectangle maps to [0, 3] x [0, 4] at height 7/11.
    assert actual == expected == Fraction(28, 11)
    assert reduced.work_units < expanded.work_units
