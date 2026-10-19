#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Directed source bounds, grouped restoration and refusal phase evidence."""

from fractions import Fraction

import equinox as eqx
import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._coordinate_enclosure import evaluate, Polynomial
from phydrax.discretization._coordinate_identity import coordinate_scope_key
from phydrax.geometry import ParameterId
from phydrax.geometry._embedding_state import (
    EmbeddingState,
    MeshCertificateFinding,
    MeshCertificateLimits,
)
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._source_boundary_records import SourceBoundaryChartCover
from phydrax.geometry._source_fidelity import (
    _fidelity_boundary_scope,
    _fidelity_maps,
    _mapped_domain_source_samples,
    certify_source_fidelity,
    FidelityMap,
    ImplicitBoundarySource,
    ImplicitProjectionBoundarySource,
    MappedDomainBoundarySource,
    ParametricCurveBoundarySource,
    source_to_mesh,
    SourceFidelityBounds,
    SourceFidelityCertificate,
    SourceFidelityCoverage,
    split_fidelity_map,
    split_fidelity_polynomial,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.mesh_certificate_fixtures import curved_tetrahedral_components
from tests._support.source_fidelity_cases import (
    fidelity_triangle,
    grouped_fidelity_certificate,
    implicit_boundary_source,
)


def test_grouped_fidelity_bounds_preserve_certificate_identity_payload() -> None:
    certificate = grouped_fidelity_certificate()
    expected = canonical_fingerprint(
        {
            "kind": "source-fidelity-certificate",
            "binding": certificate.binding.binding_id,
            "status": "certified",
            "findings": [],
            "tolerance": 0.5,
            "semantics": ("certified", "certified"),
            "mesh_to_source": ("0.25", "0.125"),
            "source_to_mesh": ("0.375", "0.0625"),
            "sample_order": 4,
            "sample_counts": (4, 5),
            "projection_coverage": None,
            "domain_coverage": None,
            "chart_coverage": None,
            "boundary_degree": "triangle-boundary-degree",
        }
    )
    assert certificate.certificate_id == expected
    assert (certificate.mesh_to_source_upper, certificate.mesh_to_source_lower) == (
        0.25,
        0.125,
    )
    assert (certificate.source_to_mesh_upper, certificate.source_to_mesh_lower) == (
        0.375,
        0.0625,
    )
    assert (certificate.mesh_sample_count, certificate.source_sample_count) == (4, 5)


@pytest.mark.parametrize("facets", (None, (7, 9)), ids=("whole-boundary", "facet-scope"))
def test_stored_fidelity_reconstruction_recomputes_the_same_bound_identity(
    facets: tuple[int, ...] | None,
) -> None:
    certificate = grouped_fidelity_certificate(facets)
    restored = SourceFidelityCertificate.from_stored_fields(certificate)
    assert restored is not certificate
    assert restored.certificate_id == certificate.certificate_id
    assert restored.status == certificate.status == "certified"
    assert restored.target_facet_ids == facets
    assert restored.source_scope_id == certificate.source_scope_id
    assert restored.boundary_degree_id == certificate.boundary_degree_id
    assert (restored.mesh_sample_count, restored.source_sample_count) == (4, 5)


def test_stored_fidelity_reconstruction_refuses_stale_finding_identity() -> None:
    certificate = grouped_fidelity_certificate()
    changed = eqx.tree_at(
        lambda value: value.findings,
        certificate,
        (MeshCertificateFinding("source_uncovered", "unresolved", "mesh"),),
        is_leaf=lambda value: isinstance(value, tuple),
    )
    with pytest.raises(ValueError, match="SourceFidelityCertificate.certificate_id"):
        SourceFidelityCertificate.from_stored_fields(changed)


def test_grouped_fidelity_coverage_refuses_a_different_source_revision() -> None:
    certificate = grouped_fidelity_certificate()
    source_id = certificate.binding.source_id
    assert source_id is not None
    cover = SourceBoundaryChartCover(
        np.asarray((((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),), dtype=np.float64),
        np.zeros((1,), dtype=np.float64),
        "certified",
        True,
        source_id,
        "different-source-revision",
    )
    with pytest.raises(ValueError, match="SourceFidelityCertificate.chart_coverage"):
        SourceFidelityCertificate(
            certificate.binding,
            (),
            tolerance=0.5,
            bounds=SourceFidelityBounds(
                semantics=("certified", "certified"),
                mesh_to_source=(0.25, 0.125),
                source_to_mesh=(0.375, 0.0625),
                sample_counts=(4, 5),
            ),
            sample_order=4,
            coverage=SourceFidelityCoverage(chart=cover),
        )


def test_polynomial_midpoint_children_preserve_order_and_expression_charges() -> None:
    # x**2 + y has rational dyadic values at all four child centroids.
    polynomial: Polynomial = {(2, 0): Fraction(1), (0, 1): Fraction(1)}
    limits = MeshCertificateLimits()
    with coordinate_reference_budget(limits) as budget:
        polynomial_children = split_fidelity_polynomial(polynomial, budget=budget)
        polynomial_work = budget.work_units
        polynomial_required_work = budget.required_work_units
        polynomial_peak_bytes = budget.peak_bytes_upper
    with coordinate_reference_budget(limits) as budget:
        map_children = split_fidelity_map(FidelityMap((polynomial,), 2, 0), budget=budget)
        assert budget.work_units == polynomial_work
        assert budget.required_work_units == polynomial_required_work
        assert budget.peak_bytes_upper == polynomial_peak_bytes
    assert tuple(child.coordinates[0] for child in map_children) == polynomial_children
    with coordinate_reference_budget(limits) as budget:
        centroid = (Fraction(1, 3), Fraction(1, 3))
        values = tuple(
            evaluate(child, centroid, budget=budget) for child in polynomial_children
        )
    # Child centroid images are (1/6,1/6), (2/3,1/6), (1/6,2/3), (1/3,1/3).
    assert values == (Fraction(7, 36), Fraction(11, 18), Fraction(25, 36), Fraction(4, 9))


def test_mapped_boundary_sampling_releases_completed_polynomial_preparations() -> None:
    curve = FidelityMap(
        (
            {(0,): Fraction(1), (1,): Fraction(1)},
            {(0,): Fraction(1), (1,): Fraction(1), (2,): Fraction(1)},
        ),
        1,
        0,
    )
    radius = 0.02
    limits = MeshCertificateLimits()
    with coordinate_reference_budget(limits) as budget:
        _mapped_domain_source_samples([curve], 2, 4.0, 1024, limits, budget=budget)
        retained_basis_bytes = budget.retained_basis_bytes
        samples = _mapped_domain_source_samples(
            [curve], 2, radius, 1024, limits, budget=budget
        )
        # The fixed-degree reference basis stays reusable, but completed sample
        # preparations must not add request-retained storage as the cover grows.
        assert budget.retained_basis_bytes == retained_basis_bytes
        assert budget.temporary_bytes_upper == 0
        assert budget.work_units > 0
    assert samples.complete
    assert samples.semantics == "certified"
    assert samples.points.shape[0] > 1
    assert np.all(samples.covering_radius <= radius)
    x = samples.points[:, 0]
    np.testing.assert_array_equal(samples.points[:, 1], x**2 - x + 1)
    np.testing.assert_array_equal(samples.point_error, np.zeros_like(samples.point_error))


def test_affine_boundary_fidelity_certifies_both_independent_line_covers() -> None:
    mesh, geometry, source = fidelity_triangle()
    certificate = certify_source_fidelity(mesh, geometry, source, tolerance=0.3)
    assert certificate.status == "certified"
    assert certificate.findings == ()
    assert certificate.mesh_to_source_semantics == "certified"
    assert certificate.source_to_mesh_semantics == "certified"
    assert (
        0.0 <= certificate.mesh_to_source_lower <= certificate.mesh_to_source_upper <= 0.3
    )
    assert (
        0.0 <= certificate.source_to_mesh_lower <= certificate.source_to_mesh_upper <= 0.3
    )
    assert certificate.mesh_sample_count > 0
    assert certificate.source_sample_count == source.samples.points.shape[0]
    restored = SourceFidelityCertificate.from_stored_fields(certificate)
    assert restored.certificate_id == certificate.certificate_id


def test_affine_source_refusal_reports_only_its_earned_coordinate_phase() -> None:
    mesh, _, source = fidelity_triangle()
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    items = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    limits = MeshCertificateLimits(maximum_work_units=32)
    state = EmbeddingState([], [])
    with coordinate_reference_budget(limits) as budget:
        # Five earlier request units are settled by the next source-cover charge.
        budget.reserve(5)
        result = source_to_mesh(state, source, points, items, 0.3, limits, budget=budget)
        assert budget.work_units == 17
    assert result[0] == "certified" and np.isinf(result[1]) and result[2] == 0.0
    assert result[3] == source.samples.points.shape[0]
    assert len(state.findings) == 1
    finding = state.findings[0]
    assert finding.check == "source_cover_expression_budget"
    assert finding.status == "unresolved" and finding.resource == "coefficient_work"
    assert dict(finding.requested) == {"limit": 32, "requested": 41}
    achieved = dict(finding.achieved)
    assert achieved["completed"] == 17
    assert achieved["source_expression_work_units"] == 12


def test_projection_boundary_source_refuses_a_missing_analytic_owner() -> None:
    with pytest.raises(TypeError, match="profile"):
        ImplicitProjectionBoundarySource(None)  # ty: ignore[invalid-argument-type]


def test_mapped_boundary_source_refuses_a_missing_independent_domain() -> None:
    with pytest.raises(TypeError, match="domain"):
        MappedDomainBoundarySource(
            None,  # ty: ignore[invalid-argument-type]
            np.asarray((0,), dtype=np.int64),
        )


def test_chart_resource_counts_preserve_numpy_scalar_fingerprint_identity() -> None:
    simplices = np.asarray((((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),), dtype=np.float64)
    bounds = np.zeros((1,), dtype=np.float64)
    numpy_counts = (("phase_work", np.int64(7), np.int64(9)),)
    numpy_cover = SourceBoundaryChartCover(
        simplices,
        bounds,
        "sampled",
        True,
        "resource-count-source",
        "resource-count-revision",
        resource_counts=numpy_counts,
    )
    scalar_cover = SourceBoundaryChartCover(
        simplices,
        bounds,
        "sampled",
        True,
        "resource-count-source",
        "resource-count-revision",
        resource_counts=(("phase_work", 7, 9),),
    )
    assert numpy_cover.resource_counts == numpy_counts
    assert isinstance(numpy_cover.resource_counts[0][1], np.integer)
    assert isinstance(numpy_cover.resource_counts[0][2], np.integer)
    assert numpy_cover.cover_id != scalar_cover.cover_id


def test_parametric_source_reconstruction_replays_enclosures_and_samples() -> None:
    _, _, source = fidelity_triangle()
    restored = ParametricCurveBoundarySource.from_stored_fields(source)
    assert restored is not source
    assert restored.source_id == source.source_id
    assert restored.source_revision == source.source_revision
    assert restored.parameter_ranges == source.parameter_ranges
    assert restored.samples.complete == source.samples.complete
    np.testing.assert_array_equal(restored.boxes, source.boxes)
    np.testing.assert_array_equal(restored.samples.points, source.samples.points)
    np.testing.assert_array_equal(
        restored.samples.covering_radius, source.samples.covering_radius
    )
    np.testing.assert_array_equal(
        restored.samples.point_error, source.samples.point_error
    )
    assert array_tree_fingerprint(restored) == array_tree_fingerprint(source)


def test_parametric_source_reconstruction_refuses_corrupted_carrier_inputs() -> None:
    _, _, source = fidelity_triangle()
    changed = eqx.tree_at(
        lambda value: value.curves,
        source,
        (None, *source.curves[1:]),
        is_leaf=lambda value: isinstance(value, tuple),
    )
    with pytest.raises(TypeError, match="curves"):
        ParametricCurveBoundarySource.from_stored_fields(changed)


def test_implicit_source_reconstruction_recomputes_its_state_revision() -> None:
    source = implicit_boundary_source()
    restored = ImplicitBoundarySource.from_stored_fields(source)
    assert restored is not source
    assert restored.source_id == source.source_id
    assert restored.source_revision == source.source_revision
    assert restored.spacing == source.spacing
    assert restored.geometry.equivalent(source.geometry)
    assert array_tree_fingerprint(restored) == array_tree_fingerprint(source)


def test_implicit_source_reconstruction_refuses_an_invalid_geometry_state() -> None:
    source = implicit_boundary_source()
    geometry = source.geometry.with_parameters(
        {ParameterId("source-reconstruction-sphere", "radius"): -1.0}
    )
    changed = eqx.tree_at(lambda value: value.geometry, source, geometry)
    assert not bool(geometry.validity().accepted)
    with pytest.raises(ValueError, match="ImplicitBoundarySource.geometry"):
        ImplicitBoundarySource.from_stored_fields(changed)


@pytest.mark.parametrize("scoped", (False, True), ids=("whole-boundary", "one-facet"))
def test_fidelity_expands_only_cells_incident_to_requested_hex_faces(
    scoped: bool,
) -> None:
    # The LF13 4x4x4 sweep has eight strictly interior cells.
    points = np.asarray(
        tuple(
            (i / 4, j / 4, k / 4) for i in range(5) for j in range(5) for k in range(5)
        ),
        dtype=np.float64,
    )
    cells = np.asarray(
        tuple(
            tuple(
                (i + u) * 25 + (j + v) * 5 + k + w
                for u, v, w in (
                    (0, 0, 0),
                    (1, 0, 0),
                    (1, 1, 0),
                    (0, 1, 0),
                    (0, 0, 1),
                    (1, 0, 1),
                    (1, 1, 1),
                    (0, 1, 1),
                )
            )
            for i in range(4)
            for j in range(4)
            for k in range(4)
        ),
        dtype=np.int32,
    )
    mesh = CellMesh(points, (CellBlock("sweep", "hexahedron", cells),))
    geometry = CellGeometrySpec.affine(mesh)
    state = EmbeddingState([], [])
    with coordinate_reference_budget() as budget:
        boundary = _fidelity_boundary_scope(mesh, True, None, budget=budget)
        selected = (boundary.facet_entities[min(boundary.boundary)],) if scoped else None
        maps = _fidelity_maps(
            state, mesh, geometry, target_facet_ids=selected, budget=budget
        )
        assert len(maps) == (2 if scoped else 192)
        assert len(budget.coefficient_preparation_cache) == (1 if scoped else 56)
    assert state.findings == []
    assert all(item.dimension == 2 for item in maps)
    if selected is not None:
        assert all(
            item.entity == selected[0] and item.entity_kind == "facet" for item in maps
        )
    else:
        assert all(item.entity_kind == "cell" for item in maps)


def test_fidelity_reuses_earned_current_embedding_maps_without_reexpansion() -> None:
    mesh, geometry, validity = curved_tetrahedral_components(1.0, 2.0, 2)
    with coordinate_reference_budget() as budget:
        state = EmbeddingState([], [])
        expected = _fidelity_maps(state, mesh, geometry, budget=budget)
        standalone_work = budget.work_units
        assert state.findings == []
    with coordinate_reference_budget() as budget:
        embedding = certify_global_embedding(mesh, geometry, validity, budget=budget)
        assert embedding.status == "certified"
        assert coordinate_scope_key(mesh, geometry) in budget.prepared_cell_cache
        before = budget.work_units
        state = EmbeddingState([], [])
        actual = _fidelity_maps(state, mesh, geometry, budget=budget)
        assert budget.work_units - before < standalone_work
        assert actual == expected
        assert state.findings == []


def test_fidelity_does_not_reuse_embedding_maps_after_live_geometry_changes() -> None:
    mesh, geometry, validity = curved_tetrahedral_components(1.0, 2.0, 2)
    with coordinate_reference_budget() as budget:
        embedding = certify_global_embedding(mesh, geometry, validity, budget=budget)
        assert embedding.status == "certified"
        original = _fidelity_maps(EmbeddingState([], []), mesh, geometry, budget=budget)
        changed = eqx.tree_at(
            lambda value: value.coordinates, geometry, geometry.coordinates + 0.125
        )
        assert coordinate_scope_key(mesh, changed) not in budget.prepared_cell_cache
        state = EmbeddingState([], [])
        actual = _fidelity_maps(state, mesh, changed, budget=budget)
        assert actual != original
        assert state.findings == []
    with coordinate_reference_budget() as budget:
        independent = _fidelity_maps(EmbeddingState([], []), mesh, changed, budget=budget)
        assert actual == independent
