"""Consumer contracts for region coverage, phase work, and stored evidence."""

from __future__ import annotations

import io
from collections.abc import Callable
from fractions import Fraction

import equinox as eqx
import numpy as np
import pytest

from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import (
    certify_domain_coverage,
    CoverageEvidence,
    CoverageExpressionWork,
    CoverageMeasures,
    DomainCoverageCertificate,
    MeshCertificateFinding,
    MeshCertificateLimits,
    PiecewiseLinearDomain,
)
from phydrax.geometry._domain_coverage import (
    _CoveringPieces,
    _partition_covering_pieces,
)
from phydrax.geometry._embedding_state import EmbeddingState
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.domain_coverage import (
    split_square_coverage_carrier,
    split_square_domain,
)
from tests._support.mesh_certificate_fixtures import (
    coordinate_budget as coordinate_budget,
)


@pytest.fixture(scope="module", params=(False, True), ids=("affine", "mapped"))
def coverage(
    request: pytest.FixtureRequest,
) -> tuple[PiecewiseLinearDomain, DomainCoverageCertificate]:
    domain = split_square_domain()
    mesh, geometry, embedding, regions = split_square_coverage_carrier(
        mapped=request.param
    )
    certificate = certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding
    )
    return domain, certificate


def test_region_integrals_and_source_rows_follow_declared_order(
    coverage: tuple[PiecewiseLinearDomain, DomainCoverageCertificate],
) -> None:
    domain, certificate = coverage
    # Dyadic half-unit regions have exactly representable area; no tolerance is needed.
    assert certificate.status == "certified"
    assert certificate.region_ids == domain.region_ids == ("left", "right")
    assert (
        certificate.requested_region_measures
        == certificate.achieved_region_measures
        == (0.5, 0.5)
    )
    assert certificate.integration_error_bounds == (0.0, 0.0)
    assert certificate.covered_source_facet_count == certificate.source_facet_count == 7
    assert {row[1] for row in certificate.facet_source_overlaps} == set(range(7))
    assert certificate.facet_source_overlaps == tuple(
        sorted(certificate.facet_source_overlaps)
    )


def test_stored_field_reconstruction_preserves_certificate_identity(
    coverage: tuple[PiecewiseLinearDomain, DomainCoverageCertificate],
) -> None:
    domain, certificate = coverage
    rebuilt = DomainCoverageCertificate.from_stored_fields(certificate, domain)
    assert rebuilt.certificate_id == certificate.certificate_id
    assert rebuilt.binding.binding_id == certificate.binding.binding_id
    assert rebuilt.domain_id == domain.domain_id
    assert rebuilt.facet_source_overlaps == certificate.facet_source_overlaps
    assert (
        rebuilt.source_expression_work_units == certificate.source_expression_work_units
    )
    assert (
        rebuilt.source_expression_required_work_units
        == certificate.source_expression_required_work_units
    )
    assert (
        rebuilt.source_expression_peak_bytes == certificate.source_expression_peak_bytes
    )


def test_deserialized_coverage_revalidates_against_restored_domain(
    coverage: tuple[PiecewiseLinearDomain, DomainCoverageCertificate],
) -> None:
    domain, certificate = coverage
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(stream, (domain, certificate))
    stream.seek(0)
    restored_domain, restored_certificate = eqx.tree_deserialise_leaves(
        stream, (domain, certificate)
    )
    rebuilt = DomainCoverageCertificate.from_stored_fields(
        restored_certificate, restored_domain
    )
    assert rebuilt.certificate_id == certificate.certificate_id
    assert rebuilt.domain_id == restored_domain.domain_id
    np.testing.assert_array_equal(restored_domain.facets, domain.facets)
    np.testing.assert_array_equal(restored_domain.facet_regions, domain.facet_regions)


def test_changed_stored_findings_cannot_retain_certified_identity(
    coverage: tuple[PiecewiseLinearDomain, DomainCoverageCertificate],
) -> None:
    domain, certificate = coverage
    finding = MeshCertificateFinding("embedding_premise", "unresolved", "mesh")
    changed = eqx.tree_at(lambda value: value.findings, certificate, replace=(finding,))
    rebuilt = DomainCoverageCertificate.from_stored_fields(changed, domain)
    assert rebuilt.status == "unresolved"
    assert rebuilt.findings == (finding,)
    assert rebuilt.certificate_id != certificate.certificate_id


def test_grouped_constructor_refuses_partial_absent_integral(
    coverage: tuple[PiecewiseLinearDomain, DomainCoverageCertificate],
) -> None:
    domain, certificate = coverage
    measures = CoverageMeasures(
        certificate.requested_region_measures,
        (None, certificate.achieved_region_measures[1]),
        certificate.requested_region_measure_bounds,
        certificate.achieved_region_measure_bounds,
        certificate.integration_error_bounds,
    )
    with pytest.raises(ValueError, match=r"DomainCoverageCertificate.measures.achieved"):
        DomainCoverageCertificate(
            certificate.binding,
            certificate.embedding_certificate_id,
            domain,
            (
                MeshCertificateFinding(
                    "mapped_measure_source", "unresolved", "cell", (0,)
                ),
            ),
            measures=measures,
            evidence=CoverageEvidence(
                certificate.covered_source_facet_count,
                certificate.facet_source_overlaps,
                certificate.premise_certificate_ids,
                certificate.candidate_pair_count,
                certificate.subdivision_piece_count,
                certificate.maximum_subdivision_depth_reached,
            ),
            expression_work=CoverageExpressionWork(
                certificate.source_expression_work_units,
                certificate.source_expression_required_work_units,
                certificate.source_expression_peak_bytes,
            ),
        )


@pytest.mark.parametrize("mapped", (False, True), ids=("affine", "mapped"))
def test_omitted_material_interface_retains_region_finding_order(mapped: bool) -> None:
    mesh, geometry, embedding, regions = split_square_coverage_carrier(mapped=mapped)
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        split_square_domain(),
        np.zeros_like(regions),
        embedding=embedding,
    )
    assert certificate.status == "violated"
    assert certificate.achieved_region_measures == (1.0, 0.0)
    interface = next(
        finding
        for finding in certificate.findings
        if finding.check == "omitted_interface"
    )
    assert interface.entity_kind == "source_facet" and interface.entity_ids == (6,)
    assert certificate.findings[-1].check == "region_measure"
    assert certificate.findings[-1].entity_ids == (0, 1)


def test_mapped_resource_refusal_keeps_unknown_integrals_and_phase_receipt() -> None:
    mesh, geometry, embedding, regions = split_square_coverage_carrier(mapped=True)
    domain = split_square_domain()
    baseline = certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding
    )
    required = baseline.source_expression_required_work_units
    assert baseline.status == "certified" and required is not None and required > 0
    limits = MeshCertificateLimits(maximum_work_units=required - 1)
    refused = certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding, limits=limits
    )
    assert refused.status == "unresolved"
    assert refused.achieved_region_measures == (None, None)
    assert refused.achieved_region_measure_bounds == (None, None)
    assert refused.integration_error_bounds == (None, None)
    assert refused.source_expression_work_units is not None
    assert refused.source_expression_work_units <= limits.maximum_work_units
    assert refused.source_expression_required_work_units is not None
    assert refused.source_expression_required_work_units > limits.maximum_work_units
    finding = next(
        finding
        for finding in refused.findings
        if finding.check == "mapped_coverage_resource_budget"
    )
    assert finding.resource is not None and finding.requested and finding.achieved


def test_coverage_phase_requirement_excludes_shared_request_predecessors(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    mesh, geometry, embedding, regions = split_square_coverage_carrier(mapped=True)
    domain = split_square_domain()
    limits = MeshCertificateLimits()
    invocation = coordinate_budget(
        limits.maximum_work_units, limits.maximum_scratch_bytes
    )
    baseline = certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding, budget=invocation
    )
    required = baseline.source_expression_required_work_units
    charged = baseline.source_expression_work_units
    assert baseline.status == "certified" and required is not None and charged is not None
    invocation_work = invocation.work_units
    invocation_requirement = invocation.required_work_units
    # The guarded coverage phase starts after coordinate-scope preparation.
    assert invocation_work > charged
    assert invocation_requirement > required
    predecessor_work = 5
    shared = coordinate_budget(
        predecessor_work + invocation_requirement, limits.maximum_scratch_bytes
    )
    shared.reserve(predecessor_work)
    certificate = certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding, budget=shared
    )
    assert certificate.status == "certified"
    assert certificate.source_expression_required_work_units == required
    assert certificate.source_expression_work_units == charged
    assert shared.work_units == predecessor_work + invocation_work
    assert certificate.certificate_id == baseline.certificate_id

    phase_only_limit = predecessor_work + required
    phase_only = coordinate_budget(phase_only_limit, limits.maximum_scratch_bytes)
    phase_only.reserve(predecessor_work)
    refused = certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding, budget=phase_only
    )
    assert refused.status == "unresolved"
    finding = next(
        finding
        for finding in refused.findings
        if finding.check == "mapped_coverage_resource_budget"
    )
    assert finding.resource == "coefficient_work"
    assert dict(finding.requested)["limit"] == phase_only_limit
    assert dict(finding.requested)["requested"] > phase_only_limit


@pytest.mark.parametrize("tilted", (False, True), ids=("same-axes", "independent-axes"))
@pytest.mark.parametrize(
    ("rows", "expected"),
    (
        (((0, 0), (1, 0), (0, 1)), Fraction(1, 2)),
        (((0.5, 0), (1.5, 0), (0.5, 1)), Fraction(1, 8)),
        (((1, 0), (1, 1), (0, 1)), Fraction(0)),
        (((1, 0), (2, 0), (1, 1)), Fraction(0)),
        (((1.125, 0), (1.125, 1), (0.125, 1)), Fraction(0)),
    ),
    ids=("whole", "positive-overlap", "edge-contact", "vertex-contact", "oblique-gap"),
)
def test_affine_facet_partition_retains_exact_contact_and_overlap_evidence(
    rows: tuple[tuple[float, float], ...],
    expected: Fraction,
    tilted: bool,
) -> None:
    source = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float64)
    target = np.asarray(rows, dtype=np.float64)
    source_height = source[:, 0] + 2 * source[:, 1] if tilted else np.zeros((3,))
    target_height = target[:, 0] + 2 * target[:, 1] if tilted else np.zeros((3,))
    vertices = np.column_stack((source, source_height))
    points = np.column_stack((target, target_height))
    domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(((0, 1, 2),), dtype=np.int64),
        np.asarray(((0, -1),), dtype=np.int64),
        ("material",),
        source_id="exact-facet-partition",
    )
    covering = _CoveringPieces(
        np.asarray(((0, 1, 2),), dtype=np.int64),
        np.asarray((42,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        np.asarray((-1,), dtype=np.int64),
    )
    target_axes = (0, 2) if tilted else (0, 1)
    state = EmbeddingState([], ["domain_coverage"])
    with coordinate_reference_budget() as budget:
        covered, unmatched, incomplete, relations = _partition_covering_pieces(
            state,
            covering,
            points,
            domain,
            np.asarray((target_axes,), dtype=np.int64),
            MeshCertificateLimits(),
            budget=budget,
        )
    projected = expected * (2 if tilted else 1)
    assert covered == [projected]
    assert unmatched.tolist() == [expected != Fraction(1, 2)]
    assert not incomplete
    assert state.candidate_pairs == 1
    assert state.findings == []
    assert relations == (
        {(42, 0, target_axes, "exact", "physical"): projected} if expected > 0 else {}
    )


def test_affine_facet_partition_accumulates_independent_projection_fragments() -> None:
    points = np.asarray(
        ((0, 0, 0), (1, 0, 1), (0, 1, 2), (0.5, 0.5, 1.5)), dtype=np.float64
    )
    domain = PiecewiseLinearDomain(
        points[:3],
        np.asarray(((0, 1, 2),), dtype=np.int64),
        np.asarray(((0, -1),), dtype=np.int64),
        ("material",),
        source_id="partitioned-tilted-facet",
    )
    covering = _CoveringPieces(
        np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int64),
        np.asarray((42, 43), dtype=np.int64),
        np.asarray((0, 0), dtype=np.int64),
        np.asarray((-1, -1), dtype=np.int64),
    )
    state = EmbeddingState([], ["domain_coverage"])
    with coordinate_reference_budget() as budget:
        covered, unmatched, incomplete, relations = _partition_covering_pieces(
            state,
            covering,
            points,
            domain,
            np.asarray(((0, 2),), dtype=np.int64),
            MeshCertificateLimits(),
            budget=budget,
        )
    assert covered == [Fraction(1)]
    assert unmatched.tolist() == [False, False]
    assert not incomplete and state.candidate_pairs == 2 and state.findings == []
    assert relations == {
        (42, 0, (0, 2), "exact", "physical"): Fraction(1, 2),
        (43, 0, (0, 2), "exact", "physical"): Fraction(1, 2),
    }
