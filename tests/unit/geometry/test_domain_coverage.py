"""Consumer contracts for region coverage, phase work, and stored evidence."""

from __future__ import annotations

import gc
import io
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Never

import equinox as eqx
import numpy as np
import pytest

from phydrax._fingerprint import tree_content_equal
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
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
    CoverageOverlapBank,
    CoverageOverlapKey,
    DomainCoveragePredecessor,
)
from phydrax.geometry._embedding_state import EmbeddingState
from phydrax.geometry._global_embedding import GlobalEmbeddingCertificate
from phydrax.geometry._planar_primitives import Polygon, PolygonIntersectionPreparation
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
    overlaps = certificate.facet_source_overlaps
    assert set(overlaps.sources.tolist()) == set(range(7))
    assert tree_content_equal(CoverageOverlapBank.from_stored_fields(overlaps), overlaps)


def test_overlap_bank_refuses_noncanonical_rows(
    coverage: tuple[PiecewiseLinearDomain, DomainCoverageCertificate],
) -> None:
    domain, certificate = coverage
    overlaps = certificate.facet_source_overlaps
    facets = np.asarray(overlaps.facets)
    assert facets[0] != facets[-1]
    swapped = facets[::-1]
    with pytest.raises(ValueError, match=r"CoverageOverlapBank.rows"):
        CoverageOverlapBank.from_stored_fields(
            eqx.tree_at(lambda value: value.facets, overlaps, swapped)
        )
    with pytest.raises(ValueError, match=r"CoverageOverlapBank.semantic_values"):
        CoverageOverlapBank.from_stored_fields(
            eqx.tree_at(
                lambda value: value.semantic_codes,
                overlaps,
                np.asarray(overlaps.semantic_codes) + 1,
            )
        )
    with pytest.raises(TypeError, match=r"CoverageOverlapBank.sources"):
        CoverageOverlapBank.from_stored_fields(
            eqx.tree_at(
                lambda value: value.sources,
                overlaps,
                np.asarray(overlaps.sources, dtype=np.int32),
            )
        )
    widened = np.asarray(overlaps.upper) * 2.0
    changed = eqx.tree_at(
        lambda value: value.facet_source_overlaps.upper, certificate, widened
    )
    rebuilt = DomainCoverageCertificate.from_stored_fields(changed, domain)
    assert rebuilt.certificate_id != certificate.certificate_id


def test_stored_field_reconstruction_preserves_certificate_identity(
    coverage: tuple[PiecewiseLinearDomain, DomainCoverageCertificate],
) -> None:
    domain, certificate = coverage
    rebuilt = DomainCoverageCertificate.from_stored_fields(certificate, domain)
    assert rebuilt.certificate_id == certificate.certificate_id
    assert rebuilt.binding.binding_id == certificate.binding.binding_id
    assert rebuilt.domain_id == domain.domain_id
    assert tree_content_equal(
        rebuilt.facet_source_overlaps, certificate.facet_source_overlaps
    )
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
    identity = baseline.certificate_id
    assert baseline.status == "certified" and required is not None and charged is not None
    invocation_work = invocation.work_units
    invocation_requirement = invocation.required_work_units
    # The guarded coverage phase starts after coordinate-scope preparation.
    assert invocation_work > charged
    assert invocation_requirement > required
    # Release the issued proof so each later request proves coverage afresh.
    del baseline
    gc.collect()
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
    assert certificate.certificate_id == identity
    del certificate
    gc.collect()

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


@pytest.mark.parametrize("mapped", (False, True), ids=("affine", "mapped"))
def test_issued_coverage_is_reused_only_while_owned_for_identical_inputs(
    mapped: bool,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    mesh, geometry, embedding, regions = split_square_coverage_carrier(mapped=mapped)
    domain = split_square_domain()
    limits = MeshCertificateLimits()

    def ledger() -> CoordinateEnclosureBudget:
        return coordinate_budget(limits.maximum_work_units, limits.maximum_scratch_bytes)

    issuing = ledger()
    issued = certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding, budget=issuing
    )
    assert issued.status == "certified"
    issuing_work, identity = issuing.work_units, issued.certificate_id
    assert issuing.retained_owner_objects.get(id(issued)) is None

    with HostBatchTicket(
        limits.maximum_work_units,
        queries_upper=limits.maximum_work_units,
        bytes_upper=limits.maximum_scratch_bytes,
    ) as ticket:
        reusing = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with reusing.activate():
            assert (
                certify_domain_coverage(
                    mesh, geometry, domain, regions, embedding=embedding, budget=reusing
                )
                is issued
            )
        # The reusing request admits and retains the issued bytes, not the proof.
        assert reusing.retained_owner_objects[id(issued)] is issued
        assert reusing.retained_basis_bytes > 0
        assert 0 < reusing.work_units < issuing_work

    # Another policy, assignment or mesh object needs its own proof.
    changed_limits = MeshCertificateLimits(
        maximum_candidate_pairs=limits.maximum_candidate_pairs + 1
    )
    assert (
        certify_domain_coverage(
            mesh, geometry, domain, regions, embedding=embedding, limits=changed_limits
        )
        is not issued
    )
    unassigned = np.full_like(regions, -1)
    assert (
        certify_domain_coverage(mesh, geometry, domain, unassigned, embedding=embedding)
        is not issued
    )
    other_mesh, other_geometry, other_embedding, _ = split_square_coverage_carrier(
        mapped=mapped
    )
    other = certify_domain_coverage(
        other_mesh, other_geometry, domain, regions, embedding=other_embedding
    )
    assert other is not issued and other.certificate_id == identity

    # The registry retains neither proof nor carrier: once every owner releases
    # the issued proof, an identical request proves coverage again.
    del issued, reusing
    gc.collect()
    fresh_ledger = ledger()
    fresh = certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding, budget=fresh_ledger
    )
    assert fresh.certificate_id == identity
    assert fresh_ledger.work_units == issuing_work
    assert fresh_ledger.retained_owner_objects.get(id(fresh)) is None


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


@pytest.mark.parametrize("rotation", (0, 1, 2))
@pytest.mark.parametrize("changed", (False, True), ids=("identical", "different-vertex"))
def test_identical_triangle_coverage_needs_no_polygon_preparation(
    monkeypatch: pytest.MonkeyPatch,
    rotation: int,
    changed: bool,
) -> None:
    from phydrax.geometry import _domain_coverage as owner

    source = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    points = source.copy()
    if changed:
        points[1, 0] = 0.75
    domain = PiecewiseLinearDomain(
        source,
        np.asarray(((0, 1, 2),)),
        np.asarray(((0, -1),)),
        ("material",),
        source_id="exact-triangle-identity",
    )
    covering = _CoveringPieces(
        np.asarray((np.roll(np.arange(3), rotation),)),
        np.asarray((42,)),
        np.asarray((0,)),
        np.asarray((-1,)),
    )
    original = owner.planar_primitives_owner.prepare_polygon_intersection
    preparations = 0

    def prepare(
        polygon: Polygon,
        orientation: int,
        *,
        budget: CoordinateEnclosureBudget,
        point_membership: bool = False,
    ) -> PolygonIntersectionPreparation:
        nonlocal preparations
        preparations += 1
        return original(
            polygon, orientation, budget=budget, point_membership=point_membership
        )

    monkeypatch.setattr(
        owner.planar_primitives_owner, "prepare_polygon_intersection", prepare
    )
    with coordinate_reference_budget() as budget:
        covered, unmatched, incomplete, relations = _partition_covering_pieces(
            EmbeddingState([], ["domain_coverage"]),
            covering,
            points,
            domain,
            np.asarray(((0, 1),)),
            MeshCertificateLimits(),
            budget=budget,
        )
    expected = Fraction(3, 8) if changed else Fraction(1, 2)
    assert covered == [expected]
    assert not unmatched.any() and not incomplete
    assert relations == {(42, 0, (0, 1), "exact", "physical"): expected}
    if changed:
        assert preparations > 0
    else:
        assert preparations == 0


def test_exact_triangle_coverage_has_measured_work_boundary(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    domain = PiecewiseLinearDomain(
        points,
        np.asarray(((0, 1, 2),)),
        np.asarray(((0, -1),)),
        ("material",),
        source_id="bounded-exact-triangle",
    )
    covering = _CoveringPieces(
        np.asarray(((0, 1, 2),)),
        np.asarray((42,)),
        np.asarray((0,)),
        np.asarray((-1,)),
    )

    def measure(budget: CoordinateEnclosureBudget) -> list[Fraction]:
        covered, unmatched, incomplete, _ = _partition_covering_pieces(
            EmbeddingState([], ["domain_coverage"]),
            covering,
            points,
            domain,
            np.asarray(((0, 1),)),
            MeshCertificateLimits(),
            budget=budget,
        )
        assert not unmatched.any() and not incomplete
        return covered

    ample = coordinate_budget(1_000_000, 1 << 20)
    assert measure(ample) == [Fraction(1, 2)]
    required = ample.work_units
    assert measure(coordinate_budget(required, 1 << 20)) == [Fraction(1, 2)]
    with pytest.raises(CoordinateEnclosureResourceError) as refusal:
        measure(coordinate_budget(required - 1, 1 << 20))
    assert refusal.value.resource == "coefficient_work"


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


def test_affine_intersection_reuse_keeps_current_incidence_source_and_limits() -> None:
    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    domain = PiecewiseLinearDomain(
        points,
        np.asarray(((0, 1, 2),)),
        np.asarray(((0, -1),)),
        ("material",),
        source_id="cached-exact-facet",
    )
    limits = MeshCertificateLimits(maximum_candidate_pairs=32)
    # The piece leaves the source facet, so its measure needs exact clipping.
    shifted = points + (0.5, 0, 0)

    def partition(
        bank: np.ndarray,
        owner: int,
        own_region: int,
        current_limits: MeshCertificateLimits,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[list[Fraction], np.ndarray, bool, dict[CoverageOverlapKey, Fraction]]:
        covering = _CoveringPieces(
            np.asarray(((0, 1, 2),)),
            np.asarray((owner,)),
            np.asarray((own_region,)),
            np.asarray((-1,)),
        )
        return _partition_covering_pieces(
            EmbeddingState([], ["domain_coverage"]),
            covering,
            bank,
            domain,
            np.asarray(((0, 1),)),
            current_limits,
            budget=budget,
        )

    with coordinate_reference_budget() as budget:
        start = budget.work_units
        first = partition(shifted, 42, 0, limits, budget)
        first_work = budget.work_units - start
        entries = len(budget.affine_coverage_intersection_cache)
        start = budget.work_units
        second = partition(shifted.copy(), 43, 0, limits, budget)
        assert budget.work_units - start < first_work
        assert len(budget.affine_coverage_intersection_cache) == entries
        assert first[3] == {(42, 0, (0, 1), "exact", "physical"): Fraction(1, 8)}
        assert second[3] == {(43, 0, (0, 1), "exact", "physical"): Fraction(1, 8)}
        changed_region = partition(shifted, 44, 1, limits, budget)
        assert changed_region[0] == [0]
        assert changed_region[1].tolist() == [True]
        assert changed_region[3] == {}
        changed_points = partition(points + (0.25, 0, 0), 45, 0, limits, budget)
        assert changed_points[0] == [Fraction(9, 32)]
        assert len(budget.affine_coverage_intersection_cache) == entries + 1
        partition(
            shifted, 46, 0, MeshCertificateLimits(maximum_candidate_pairs=64), budget
        )
        assert len(budget.affine_coverage_intersection_cache) == entries + 2


def test_affine_partition_contained_piece_skips_coplanar_neighbor_clipping() -> None:
    # One square facet triangulated by its diagonal; the source triangles meet
    # only along that diagonal.
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=np.float64)
    domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        np.asarray(((0, -1), (0, -1)), dtype=np.int64),
        ("material",),
        source_id="diagonal-square",
    )
    points = np.asarray(
        (
            (0.5, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.5, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    # Piece 7 lies inside source facet 0 although its box meets facet 1;
    # piece 8 crosses the diagonal and is clipped against both facets.
    covering = _CoveringPieces(
        np.asarray(((0, 1, 2), (3, 0, 4)), dtype=np.int64),
        np.asarray((7, 8), dtype=np.int64),
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
            np.asarray(((0, 1), (0, 1)), dtype=np.int64),
            MeshCertificateLimits(),
            budget=budget,
        )
        cached = len(budget.affine_coverage_intersection_cache)
    assert not incomplete and state.findings == []
    assert unmatched.tolist() == [False, False]
    assert covered == [Fraction(1, 8) + Fraction(1, 12), Fraction(1, 6)]
    assert relations == {
        (7, 0, (0, 1), "exact", "physical"): Fraction(1, 8),
        (8, 0, (0, 1), "exact", "physical"): Fraction(1, 12),
        (8, 1, (0, 1), "exact", "physical"): Fraction(1, 6),
    }
    # One coplanar source pair plus two candidate facets for each piece.
    assert state.candidate_pairs == 5
    # Only the straddling piece's two pairs needed clipping.
    assert cached == 2


@dataclass(frozen=True)
class _NestedCoverageCase:
    domain: PiecewiseLinearDomain
    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    source_embedding: GlobalEmbeddingCertificate
    source_coverage: DomainCoverageCertificate
    mesh: CellMesh
    geometry: CellGeometrySpec
    embedding: GlobalEmbeddingCertificate
    regions: np.ndarray
    predecessor: DomainCoveragePredecessor
    limits: MeshCertificateLimits


def _nested_coverage_case(
    dimension: int,
    budget: CoordinateEnclosureBudget,
    *,
    fault: str | None = None,
    connected: bool = False,
    noncontiguous_facets: bool = False,
) -> _NestedCoverageCase:
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        CellGeometryRestrictionSource,
        require_p1_cardinal_source,
    )
    from phydrax.discretization._cell_geometry_predecessor import (
        prepare_cell_geometry_predecessor,
    )
    from phydrax.discretization._cell_geometry_validity import (
        cell_geometry_id,
        certify_cell_geometry_validity,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.geometry._global_embedding import certify_global_embedding

    width = dimension + 1
    kind = "triangle" if dimension == 2 else "tetrahedron"
    points = np.vstack((np.zeros(dimension), np.eye(dimension)))
    if connected:
        assert dimension == 3
        points = np.vstack((points, (-1, 0, 0)))
        source_rows = np.asarray(((0, 1, 2, 3), (0, 2, 4, 3)))
    else:
        translation = np.zeros(dimension)
        translation[0] = 3
        points = np.vstack((points, points + translation))
        source_rows = np.arange(2 * width).reshape(2, width)
    source_ids = np.asarray((10, 20), dtype=np.int64)
    source_mesh = CellMesh(
        points,
        (CellBlock("parent", kind, source_rows, global_ids=source_ids),),
        entity_global_ids=(
            {dimension - 1: 101 + 17 * np.arange(2 * width - int(connected))[::-1]}
            if noncontiguous_facets
            else None
        ),
    )
    source_geometry = CellGeometrySpec.affine(source_mesh)
    root = require_p1_cardinal_source(source_geometry.elements[0])
    faces = reference_cell_topology(kind).entities[dimension - 1]
    domain_faces: dict[tuple[int, ...], tuple[tuple[int, ...], tuple[int, int]]] = {}
    for region, row in enumerate(source_rows):
        for face in faces:
            oriented = tuple(int(row[vertex]) for vertex in face)
            key = tuple(sorted(oriented))
            if key in domain_faces:
                original, pair = domain_faces[key]
                domain_faces[key] = original, (pair[0], region)
            else:
                domain_faces[key] = oriented, (region, -1)
    domain = PiecewiseLinearDomain(
        points,
        np.asarray([entry[0] for entry in domain_faces.values()]),
        np.asarray([entry[1] for entry in domain_faces.values()]),
        ("first", "second"),
        source_id="exact-nested-coverage-source",
    )
    limits = MeshCertificateLimits()
    source_validity = certify_cell_geometry_validity(
        source_geometry, mesh=source_mesh, budget=budget
    )
    source_embedding = certify_global_embedding(
        source_mesh, source_geometry, source_validity, limits=limits, budget=budget
    )
    source_coverage = certify_domain_coverage(
        source_mesh,
        source_geometry,
        domain,
        np.asarray((0, 1), dtype=np.int64),
        embedding=source_embedding,
        limits=limits,
        budget=budget,
    )
    assert source_coverage.status == "certified"
    weights = np.eye(width)
    weights[-1] = 0
    weights[-1, (0, dimension)] = 0.5
    second = np.eye(width)
    second[0] = weights[-1]
    if fault == "fold":
        second[[0, 1]] = second[[1, 0]]
    midpoint = 0.5 * (points[source_rows[:, 0]] + points[source_rows[:, -1]])
    unique_midpoints, inverse = np.unique(midpoint, axis=0, return_inverse=True)
    target_points = np.vstack((points, unique_midpoints))
    first_rows, second_rows = source_rows.copy(), source_rows.copy()
    first_rows[:, -1] = points.shape[0] + inverse
    second_rows[:, 0] = points.shape[0] + inverse
    if fault == "fold":
        # Keep shared vertex images coherent while reversing the actual child
        # map and its cell orientation; positivity, not a rounding mismatch, fails.
        second_rows[:, [0, 1]] = second_rows[:, [1, 0]]
    keep = 1 if fault == "missing-child" else 2
    mesh = CellMesh(
        target_points,
        (
            CellBlock("first", kind, first_rows, global_ids=np.asarray((100, 101))),
            CellBlock(
                "second",
                kind,
                second_rows[:keep],
                global_ids=np.asarray((102, 103))[:keep],
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {
            "first": BarycentricCellGeometryElement(root, weights),
            "second": BarycentricCellGeometryElement(root, second),
        },
        {"first": source_rows, "second": source_rows[:keep]},
        source_geometry.coordinates,
        restriction_source=CellGeometryRestrictionSource(
            cell_geometry_id(source_geometry),
            source_mesh.topology_id,
            {"first": source_ids, "second": source_ids[:keep]},
            {"first": source_rows, "second": source_rows[:keep]},
            block_source_blocks={"first": "parent", "second": "parent"},
        ),
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
    predecessor = prepare_cell_geometry_predecessor(
        source_mesh,
        source_geometry,
        source_validity,
        source_embedding,
        mesh,
        geometry,
        source_limits=limits,
        budget=budget,
    )
    embedding = certify_global_embedding(
        mesh, geometry, validity, limits=limits, predecessor=predecessor, budget=budget
    )
    regions = np.concatenate((np.asarray((0, 1)), np.asarray((0, 1))[:keep]))
    if fault == "swapped-regions":
        regions = 1 - regions
    return _NestedCoverageCase(
        domain,
        source_mesh,
        source_geometry,
        source_embedding,
        source_coverage,
        mesh,
        geometry,
        embedding,
        regions,
        DomainCoveragePredecessor(predecessor, source_coverage, (0, 1), limits),
        limits,
    )


@pytest.mark.parametrize(
    ("dimension", "connected", "noncontiguous_facets"),
    ((2, False, False), (3, False, False), (3, True, False), (3, True, True)),
)
def test_exact_nested_coverage_preserves_regions_and_candidate_relations(
    dimension: int,
    connected: bool,
    noncontiguous_facets: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(
            dimension,
            budget,
            connected=connected,
            noncontiguous_facets=noncontiguous_facets,
        )
        assert "restriction_parent_partition" in case.embedding.evaluated_checks
        with budget.measurement():
            baseline = owner._execute_affine_coverage(
                EmbeddingState([], ["domain_coverage"]),
                case.mesh,
                case.geometry,
                case.domain,
                case.regions,
                case.limits,
                case.embedding.binding,
                case.embedding,
                (),
                budget=budget,
            ).certificate

        def forbid_repartition(*args: object, **kwargs: object) -> None:
            raise AssertionError("An earned exact regional partition was repartitioned.")

        monkeypatch.setattr(owner, "_execute_affine_coverage", forbid_repartition)
        inherited = certify_domain_coverage(
            case.mesh,
            case.geometry,
            case.domain,
            case.regions,
            embedding=case.embedding,
            limits=case.limits,
            budget=budget,
            predecessor=case.predecessor,
        )
        assert inherited.status == baseline.status == "certified"
        assert inherited.achieved_region_measures == baseline.achieved_region_measures
        assert (
            inherited.achieved_region_measure_bounds
            == baseline.achieved_region_measure_bounds
        )
        assert inherited.covered_source_facet_count == baseline.covered_source_facet_count
        assert case.source_coverage.certificate_id in inherited.premise_certificate_ids
        assert inherited.facet_source_overlaps.semantic_values == ("candidate",)
        assert np.all(inherited.facet_source_overlaps.lower == 0)
        bounds = {
            (int(facet), int(source), tuple(axes)): float(upper)
            for facet, source, axes, upper in zip(
                inherited.facet_source_overlaps.facets,
                inherited.facet_source_overlaps.sources,
                inherited.facet_source_overlaps.axes,
                inherited.facet_source_overlaps.upper,
                strict=True,
            )
        }
        old = baseline.facet_source_overlaps
        for facet, source, axes, upper in zip(
            old.facets, old.sources, old.axes, old.upper, strict=True
        ):
            assert bounds[int(facet), int(source), tuple(axes)] >= upper


@pytest.mark.parametrize("dimension", (2, 3))
@pytest.mark.parametrize("fault", ("missing-child", "swapped-regions", "fold"))
def test_exact_nested_coverage_does_not_inherit_broken_partition_or_regions(
    dimension: int,
    fault: str,
) -> None:
    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        try:
            case = _nested_coverage_case(dimension, budget, fault=fault)
        except ValueError as error:
            # The real manifold constructor can reject the folded child even
            # before a geometry or coverage certificate exists.
            assert fault == "fold"
            assert "opposite orientation" in str(error)
            return
        inherited = owner._execute_nested_coverage(
            case.mesh,
            case.geometry,
            case.domain,
            case.regions,
            case.limits,
            case.embedding.binding,
            case.embedding,
            case.predecessor,
            budget=budget,
        )
        assert inherited is None
        if fault != "fold":
            actual = certify_domain_coverage(
                case.mesh,
                case.geometry,
                case.domain,
                case.regions,
                embedding=case.embedding,
                limits=case.limits,
                budget=budget,
            )
            assert actual.status != "certified"


@pytest.mark.parametrize("dimension", (2, 3))
def test_nested_coverage_refuses_rebuilt_or_mutated_source_report(dimension: int) -> None:
    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(dimension, budget)
        reconstructed = DomainCoverageCertificate.from_stored_fields(
            case.source_coverage, case.domain
        )
        detached = DomainCoveragePredecessor(
            case.predecessor.geometry, reconstructed, (0, 1), case.limits
        )
        assert (
            owner._execute_nested_coverage(
                case.mesh,
                case.geometry,
                case.domain,
                case.regions,
                case.limits,
                case.embedding.binding,
                case.embedding,
                detached,
                budget=budget,
            )
            is None
        )
        upper = case.source_coverage.facet_source_overlaps.upper
        original = upper.copy()
        upper.setflags(write=True)
        upper[0] *= 2
        try:
            assert (
                owner._execute_nested_coverage(
                    case.mesh,
                    case.geometry,
                    case.domain,
                    case.regions,
                    case.limits,
                    case.embedding.binding,
                    case.embedding,
                    case.predecessor,
                    budget=budget,
                )
                is None
            )
        finally:
            upper[:] = original
            upper.setflags(write=False)


@pytest.mark.parametrize("dimension", (2, 3))
def test_nested_coverage_cold_reports_require_current_budget_reproof(
    dimension: int,
) -> None:
    from phydrax.discretization._cell_geometry_predecessor import (
        prepare_cell_geometry_predecessor,
    )
    from phydrax.discretization._cell_geometry_validity import (
        certify_cell_geometry_validity,
    )
    from phydrax.geometry import _domain_coverage as owner
    from phydrax.geometry._global_embedding import certify_global_embedding

    with coordinate_reference_budget() as original_budget:
        case = _nested_coverage_case(dimension, original_budget)
    values = (
        case.source_mesh,
        case.source_geometry,
        case.source_coverage,
        case.mesh,
        case.geometry,
        case.embedding,
    )
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(stream, values)
    stream.seek(0)
    source_mesh, source_geometry, source_report, mesh, geometry, report = (
        eqx.tree_deserialise_leaves(stream, values)
    )
    with coordinate_reference_budget() as budget:
        source_validity = certify_cell_geometry_validity(
            source_geometry, mesh=source_mesh, budget=budget
        )
        source_embedding = certify_global_embedding(
            source_mesh,
            source_geometry,
            source_validity,
            limits=case.limits,
            budget=budget,
        )
        parent = prepare_cell_geometry_predecessor(
            source_mesh,
            source_geometry,
            source_validity,
            source_embedding,
            mesh,
            geometry,
            source_limits=case.limits,
            budget=budget,
        )
        detached = DomainCoveragePredecessor(parent, source_report, (0, 1), case.limits)
        assert (
            owner._execute_nested_coverage(
                mesh,
                geometry,
                case.domain,
                case.regions,
                case.limits,
                report.binding,
                report,
                detached,
                budget=budget,
            )
            is None
        )
        # Neither restored certificate has portable authority. Reprove the source
        # cover and the target closed partition on this actual new ledger.
        source_coverage = certify_domain_coverage(
            source_mesh,
            source_geometry,
            case.domain,
            np.asarray((0, 1), dtype=np.int64),
            embedding=source_embedding,
            limits=case.limits,
            budget=budget,
        )
        validity = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
        embedding = certify_global_embedding(
            mesh,
            geometry,
            validity,
            limits=case.limits,
            predecessor=parent,
            budget=budget,
        )
        renewed = certify_domain_coverage(
            mesh,
            geometry,
            case.domain,
            case.regions,
            embedding=embedding,
            limits=case.limits,
            budget=budget,
            predecessor=DomainCoveragePredecessor(
                parent, source_coverage, (0, 1), case.limits
            ),
        )
        assert renewed.status == "certified"
        assert (
            renewed.achieved_region_measures
            == case.source_coverage.achieved_region_measures
        )
        assert source_coverage.certificate_id in renewed.premise_certificate_ids
        assert renewed.source_expression_work_units is not None
        assert renewed.source_expression_work_units > 0


def test_held_issued_coverage_remains_authoritative_after_independent_reproof() -> None:
    from dataclasses import replace

    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(2, budget)
        original = case.source_coverage
        regions = np.asarray((0, 1), dtype=np.int64)
        with budget.measurement():
            independent = owner._execute_affine_coverage(
                EmbeddingState([], ["domain_coverage"]),
                case.source_mesh,
                case.source_geometry,
                case.domain,
                regions,
                case.limits,
                original.binding,
                case.source_embedding,
                (),
                budget=budget,
            ).certificate
        assert independent is not original
        assert independent.status == "certified"
        assert independent.achieved_region_measures == original.achieved_region_measures
        request_key = owner._coverage_request_key(
            case.source_mesh,
            case.source_geometry,
            case.domain,
            regions,
            case.source_embedding,
            case.limits,
        )
        owner._issue_domain_coverage(
            case.source_mesh, request_key, independent, budget=budget
        )
        for certificate in (original, independent):
            assert owner.is_issued_domain_coverage(
                case.source_mesh,
                case.source_geometry,
                case.domain,
                regions,
                case.source_embedding,
                case.limits,
                certificate,
                budget=budget,
            )
        assert not owner.is_issued_domain_coverage(
            case.source_mesh,
            case.source_geometry,
            case.domain,
            regions[::-1],
            case.source_embedding,
            case.limits,
            original,
            budget=budget,
        )
        changed_limits = replace(
            case.limits, maximum_work_units=case.limits.maximum_work_units + 1
        )
        assert not owner.is_issued_domain_coverage(
            case.source_mesh,
            case.source_geometry,
            case.domain,
            regions,
            case.source_embedding,
            changed_limits,
            original,
            budget=budget,
        )
        clone = DomainCoverageCertificate.from_stored_fields(original, case.domain)
        assert not owner.is_issued_domain_coverage(
            case.source_mesh,
            case.source_geometry,
            case.domain,
            regions,
            case.source_embedding,
            case.limits,
            clone,
            budget=budget,
        )
        upper = original.facet_source_overlaps.upper
        prior = upper.copy()
        upper.setflags(write=True)
        upper[0] *= 2
        try:
            assert not owner.is_issued_domain_coverage(
                case.source_mesh,
                case.source_geometry,
                case.domain,
                regions,
                case.source_embedding,
                case.limits,
                original,
                budget=budget,
            )
        finally:
            upper[:] = prior
            upper.setflags(write=False)


@pytest.mark.parametrize("permuted", (False, True))
def test_selected_tetrahedral_facet_ids_match_canonical_global_mapper(
    permuted: bool,
) -> None:
    from phydrax.geometry import _domain_coverage as owner
    from phydrax.geometry._embedding_state import facet_entity_ids

    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0)))
    cells = np.asarray(((0, 1, 2, 3), (0, 2, 4, 3)), dtype=np.int64)
    if permuted:
        cells = cells[::-1]
    blocks = (CellBlock("parents", "tetrahedron", cells),)
    original = CellMesh(points, blocks)
    face_count = original.entity_set(2).entity_ids.size
    face_ids = 103 + 13 * np.arange(face_count, dtype=np.int64)
    if permuted:
        face_ids = face_ids[::-1]
    mesh = CellMesh(
        points,
        blocks,
        vertex_global_ids=np.asarray((101, 73, 7, 211, 29)),
        entity_global_ids={2: face_ids},
    )
    owners = np.asarray((1, 0, 1, 0, 0, 1), dtype=np.int64)
    omitted = np.asarray((2, 3, 1, 0, 2, 3), dtype=np.int64)
    rows = np.asarray(
        [
            [cells[cell, corner] for corner in range(4) if corner != axis]
            for cell, axis in zip(owners, omitted, strict=True)
        ]
    )
    if permuted:
        rows = rows[:, ::-1]
    expected = facet_entity_ids(mesh, rows)
    with coordinate_reference_budget() as budget:
        start = budget.work_units
        actual = owner._indexed_tetrahedral_facet_ids(
            mesh, owners, omitted, budget=budget
        )
        assert budget.work_units > start
    np.testing.assert_array_equal(actual, expected)
    assert set(actual).issubset(set(face_ids))


@pytest.mark.parametrize("changed", ("predecessor-rows", "target-array"))
def test_nested_embedding_authority_authenticates_current_inputs(
    changed: str,
) -> None:
    from phydrax.geometry._global_embedding import is_issued_current_nested_partition

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(3, budget, connected=True)
        parent = case.predecessor.geometry
        assert is_issued_current_nested_partition(
            case.mesh, case.geometry, parent, case.embedding, case.limits, budget=budget
        )
        original: object
        replacement: object
        if changed == "predecessor-rows":
            owner, name = parent, "source_rows"
            original = parent.source_rows
            assert parent.source_rows
            replacement = (parent.source_rows[0] + 1,) + parent.source_rows[1:]
        else:
            owner, name = case.mesh, "coordinates"
            coordinates = case.mesh.coordinates
            original = coordinates
            replacement = coordinates.at[0, 0].add(0.25)
        object.__setattr__(owner, name, replacement)
        try:
            with pytest.raises(ValueError, match="changed content|must bind"):
                is_issued_current_nested_partition(
                    case.mesh,
                    case.geometry,
                    parent,
                    case.embedding,
                    case.limits,
                    budget=budget,
                )
        finally:
            object.__setattr__(owner, name, original)


def test_authenticate_issued_coverage_keeps_real_owner_and_historical_costs() -> None:
    import gc
    import weakref

    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(2, budget)
        regions = np.asarray((0, 1), dtype=np.int64)
        historical = DomainCoverageCertificate.from_stored_fields(
            case.source_coverage, case.domain
        )
        charged = historical.source_expression_work_units
        peak = historical.source_expression_peak_bytes
        assert charged is not None and peak is not None
        object.__setattr__(historical, "source_expression_work_units", charged + 13)
        object.__setattr__(historical, "source_expression_required_work_units", 123456)
        object.__setattr__(historical, "source_expression_peak_bytes", peak + 256)
        before = budget.work_units
        resolved = owner.authenticate_issued_domain_coverage(
            case.source_mesh,
            case.source_geometry,
            case.domain,
            regions,
            case.source_embedding,
            case.limits,
            historical,
            case.source_coverage,
            budget=budget,
        )
        assert resolved is case.source_coverage
        assert budget.work_units > before
        assert historical.source_expression_work_units == charged + 13
        assert historical.source_expression_required_work_units == 123456
        assert historical.source_expression_peak_bytes == peak + 256
        assert not owner.is_issued_domain_coverage(
            case.source_mesh,
            case.source_geometry,
            case.domain,
            regions,
            case.source_embedding,
            case.limits,
            historical,
            budget=budget,
        )
        assert (
            owner.authenticate_issued_domain_coverage(
                case.source_mesh,
                case.source_geometry,
                case.domain,
                regions,
                case.source_embedding,
                case.limits,
                historical,
                historical,
                budget=budget,
            )
            is None
        )
        reference = weakref.ref(historical)
        del historical
        gc.collect()
        assert reference() is None


@pytest.mark.parametrize("changed", ("science", "regions", "limits", "current-bank"))
def test_authenticate_issued_coverage_denies_changed_science_or_current_inputs(
    changed: str,
) -> None:
    from dataclasses import replace

    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(2, budget)
        historical = DomainCoverageCertificate.from_stored_fields(
            case.source_coverage, case.domain
        )
        regions = np.asarray((0, 1), dtype=np.int64)
        limits = case.limits
        bank = case.source_coverage.facet_source_overlaps.upper
        original = bank.copy()
        if changed == "science":
            measure = case.source_coverage.achieved_region_measures[0]
            assert measure is not None
            object.__setattr__(
                historical,
                "achieved_region_measures",
                (measure + 0.25, 0.5),
            )
        elif changed == "regions":
            regions = regions[::-1]
        elif changed == "limits":
            limits = replace(limits, maximum_ray_tests=limits.maximum_ray_tests + 1)
        else:
            bank.setflags(write=True)
            bank[0] *= 2
        try:
            assert (
                owner.authenticate_issued_domain_coverage(
                    case.source_mesh,
                    case.source_geometry,
                    case.domain,
                    regions,
                    case.source_embedding,
                    limits,
                    historical,
                    case.source_coverage,
                    budget=budget,
                )
                is None
            )
        finally:
            if changed == "current-bank":
                bank[:] = original
                bank.setflags(write=False)


def test_authenticate_issued_coverage_does_not_invent_a_cold_owner() -> None:
    from jax import tree_util

    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(2, budget)
        restored_mesh = tree_util.tree_map(lambda leaf: leaf, case.source_mesh)
        assert restored_mesh is not case.source_mesh
        assert (
            owner.authenticate_issued_domain_coverage(
                restored_mesh,
                case.source_geometry,
                case.domain,
                np.asarray((0, 1), dtype=np.int64),
                case.source_embedding,
                case.limits,
                case.source_coverage,
                case.source_coverage,
                budget=budget,
            )
            is None
        )


@pytest.mark.parametrize("boundary", ("issued", "handoff"))
@pytest.mark.parametrize("changed", ("limits", "embedding"))
def test_coverage_authority_denies_current_payload_with_stale_cached_id(
    changed: str,
    boundary: str,
) -> None:
    from dataclasses import replace

    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(2, budget)
        limits = replace(case.limits)
        embedding = GlobalEmbeddingCertificate.from_stored_fields(
            case.source_embedding, budget=budget
        )

        def authenticate() -> bool | DomainCoverageCertificate | None:
            arguments = (
                case.source_mesh,
                case.source_geometry,
                case.domain,
                np.asarray((0, 1), dtype=np.int64),
                embedding,
                limits,
                case.source_coverage,
            )
            if boundary == "issued":
                return owner.is_issued_domain_coverage(*arguments, budget=budget)
            return owner.authenticate_issued_domain_coverage(
                *arguments, case.source_coverage, budget=budget
            )

        if changed == "limits":
            identity = limits.limits_id
            object.__setattr__(limits, "maximum_ray_tests", limits.maximum_ray_tests + 1)
            assert limits.limits_id == identity
            assert not authenticate()
        else:
            identity = embedding.certificate_id
            object.__setattr__(
                embedding,
                "evaluated_checks",
                (*embedding.evaluated_checks, "forged_check"),
            )
            assert embedding.certificate_id == identity
            with pytest.raises(ValueError, match="regenerated proof identity"):
                authenticate()


@pytest.mark.parametrize("boundary", ("issued", "handoff"))
def test_coverage_authority_propagates_current_validation_resource_refusal(
    boundary: str,
) -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as source_budget:
        case = _nested_coverage_case(2, source_budget)
    ticket = HostBatchTicket(1, bytes_upper=1 << 20, queries_upper=1)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=1, maximum_memory_bytes=1 << 20
    )
    with pytest.raises(CoordinateEnclosureResourceError):
        arguments = (
            case.source_mesh,
            case.source_geometry,
            case.domain,
            np.asarray((0, 1), dtype=np.int64),
            case.source_embedding,
            case.limits,
            case.source_coverage,
        )
        if boundary == "issued":
            owner.is_issued_domain_coverage(*arguments, budget=budget)
        else:
            owner.authenticate_issued_domain_coverage(
                *arguments, case.source_coverage, budget=budget
            )
    assert budget.work_units == 1


def test_issued_coverage_public_cache_keeps_independent_coverage_limits() -> None:
    from dataclasses import replace

    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(2, budget)
        regions = np.asarray((0, 1), dtype=np.int64)
        before = budget.work_units
        cached = certify_domain_coverage(
            case.source_mesh,
            case.source_geometry,
            case.domain,
            regions,
            embedding=case.source_embedding,
            limits=case.limits,
            budget=budget,
        )
        assert cached is case.source_coverage
        assert budget.work_units > before
        limits = replace(
            case.limits, maximum_work_units=case.limits.maximum_work_units + 1
        )
        renewed = certify_domain_coverage(
            case.source_mesh,
            case.source_geometry,
            case.domain,
            regions,
            embedding=case.source_embedding,
            limits=limits,
            budget=budget,
        )
        assert renewed.status == "certified"
        assert renewed.binding.limits_id != case.source_embedding.binding.limits_id
        assert owner.is_issued_domain_coverage(
            case.source_mesh,
            case.source_geometry,
            case.domain,
            regions,
            case.source_embedding,
            limits,
            renewed,
            budget=budget,
        )
        assert (
            certify_domain_coverage(
                case.source_mesh,
                case.source_geometry,
                case.domain,
                regions,
                embedding=case.source_embedding,
                limits=limits,
                budget=budget,
            )
            is renewed
        )


def test_issued_coverage_without_budget_does_not_open_a_helper_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(2, budget)

    def no_new_budget(*args: object, **kwargs: object) -> Never:
        raise AssertionError("Standalone authority authentication opened a new quota.")

    monkeypatch.setattr(CoordinateEnclosureBudget, "__init__", no_new_budget)
    assert owner.is_issued_domain_coverage(
        case.source_mesh,
        case.source_geometry,
        case.domain,
        np.asarray((0, 1), dtype=np.int64),
        case.source_embedding,
        case.limits,
        case.source_coverage,
    )


def test_binding_matches_and_require_share_the_exact_binding_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _mesh_certificates as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(2, budget)
        binding = case.source_embedding.binding
        assert binding.matches(case.source_mesh, case.source_geometry)
        binding.require(case.source_mesh, case.source_geometry)
        assert not binding.matches(case.mesh, case.geometry)
        with pytest.raises(ValueError, match="must match its mesh, topology"):
            binding.require(case.mesh, case.geometry)

        def malformed_geometry(*args: object, **kwargs: object) -> Never:
            raise ValueError("malformed geometry payload")

        monkeypatch.setattr(owner, "cell_geometry_id", malformed_geometry)
        with pytest.raises(ValueError, match="malformed geometry payload"):
            binding.matches(case.source_mesh, case.source_geometry)
        with pytest.raises(ValueError, match="malformed geometry payload"):
            binding.require(case.source_mesh, case.source_geometry)


def test_issued_coverage_foreign_binding_is_a_bool_miss() -> None:
    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(2, budget)
        assert not owner.is_issued_domain_coverage(
            case.mesh,
            case.geometry,
            case.domain,
            np.asarray((0, 1), dtype=np.int64),
            case.source_embedding,
            case.limits,
            case.source_coverage,
            budget=budget,
        )


@pytest.mark.parametrize("changed", ("limits", "embedding-after-limits-miss"))
def test_issued_coverage_propagates_malformed_current_inputs(changed: str) -> None:
    from dataclasses import replace

    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as budget:
        case = _nested_coverage_case(2, budget)
        limits = replace(case.limits)
        embedding = GlobalEmbeddingCertificate.from_stored_fields(case.source_embedding)
        if changed == "limits":
            object.__setattr__(limits, "maximum_ray_tests", -1)
        else:
            object.__setattr__(limits, "maximum_ray_tests", limits.maximum_ray_tests + 1)
            object.__setattr__(
                embedding, "evaluated_checks", (*embedding.evaluated_checks, "forged")
            )
        with pytest.raises(ValueError):
            owner.is_issued_domain_coverage(
                case.source_mesh,
                case.source_geometry,
                case.domain,
                np.asarray((0, 1), dtype=np.int64),
                embedding,
                limits,
                case.source_coverage,
                budget=budget,
            )


def test_issued_coverage_authentication_work_threshold_is_exact() -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.geometry import _domain_coverage as owner

    with coordinate_reference_budget() as source_budget:
        case = _nested_coverage_case(2, source_budget)

    def authenticate(work: int) -> int:
        ticket = HostBatchTicket(work, bytes_upper=1 << 28)
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=1 << 28
        )
        with ticket, budget.activate():
            assert owner.is_issued_domain_coverage(
                case.source_mesh,
                case.source_geometry,
                case.domain,
                np.asarray((0, 1), dtype=np.int64),
                case.source_embedding,
                case.limits,
                case.source_coverage,
                budget=budget,
            )
        return budget.work_units

    required = authenticate(1_000_000)
    assert required > 1
    assert authenticate(required) == required
    with pytest.raises(CoordinateEnclosureResourceError) as refused:
        authenticate(required - 1)
    assert refused.value.resource == "coefficient_work"
