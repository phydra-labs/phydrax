#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer-visible global certification contracts."""

from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax.discretization import _coordinate_enclosure as algebra
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._embedding_state import EmbeddingState, MeshCertificateFinding
from phydrax.geometry._source_fidelity import (
    SourceFidelityBounds,
    SourceFidelityCertificate,
)
from phydrax.meshing._certification import (
    MeshCertificationOutcome,
    MeshCertificationPremiseFailure,
    MeshCertificationReport,
    MeshCertificationReportFailure,
)
from phydrax.meshing._certification_inputs import MeshCertificationInputs
from phydrax.meshing._lineage_records import EntityLineage, EntityLineageKind, MeshLineage
from phydrax.meshing._plc_mapped_support import prepare_mapped_plc_support
from phydrax.meshing.providers._native_publication import (
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from phydrax.meshing.providers._native_route_policy import NativePublicationPolicy
from phydrax.meshing.qualification import sphere_surface
from phydrax.units import METER
from tests._support.domain_coverage import split_square_domain, SPLIT_SQUARE_POINTS
from tests._support.mesh_certificate_fixtures import (
    coordinate_budget as coordinate_budget,
    octahedron,
    plc_audit,
    prepared_publication_authority,
    prepared_tetrahedral_acceptance,
    reference_budget as reference_budget,
    SPLIT_SQUARE_CELLS,
    TETRAHEDRON,
)
from tests._support.meshing_fixtures import finding_checks


def test_volume_route_refuses_an_audit_that_skipped_closure() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    geometry, audit = plc_audit(mesh)

    with pytest.raises(ValueError, match="open_boundary"):
        phx.meshing.certify_meshing_acceptance(
            mesh,
            geometry,
            audit,
            schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
            domain=split_square_domain(),
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
    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    geometry, audit = plc_audit(
        mesh, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=split_square_domain(),
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
    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    geometry, audit = plc_audit(
        mesh,
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
    )
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=split_square_domain(),
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
    assert raised.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert any(
        name.endswith(":coefficient_work:limit") and value == 1
        for name, value in raised.value.evidence.requested
    )
    assert any(
        name.endswith(":coefficient_work:completed") and value <= 1
        for name, value in raised.value.evidence.achieved
    )


def test_route_refuses_inputs_its_checks_do_not_use() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    geometry, audit = plc_audit(mesh)

    with pytest.raises(ValueError, match="exactly the inputs"):
        phx.meshing.certify_meshing_acceptance(
            mesh,
            geometry,
            audit,
            schedule=phx.meshing.MeshCertificationSchedule("curve"),
            domain=split_square_domain(),
            cell_regions=np.asarray((0, 0, 1, 1)),
        )


def test_retained_certification_renews_source_coverage_on_actual_refinement() -> None:

    source = phx.discretization.CellMesh.from_tetrahedra(
        TETRAHEDRON, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    geometry, audit = plc_audit(
        source, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
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
        np.concatenate((TETRAHEDRON, np.asarray(((0.5, 0.0, 0.0),), dtype=np.float64))),
        np.asarray(((0, 4, 2, 3), (4, 1, 2, 3)), dtype=np.int32),
    )
    target_geometry, target_audit = plc_audit(
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
    assert renewed.coverage.achieved_region_measures == (float(Fraction(1, 6)),)
    assert np.array_equal(
        np.asarray(renewed.request.cell_regions), np.asarray((0, 0), dtype=np.int64)
    )


def test_retained_source_integrity_rejects_changed_region_assignments() -> None:

    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    request = MeshCertificationInputs(
        mesh,
        geometry,
        phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=split_square_domain(),
        cell_regions=np.asarray((0, 0, 1, 1), dtype=np.int64),
    )
    changed = eqx.tree_at(
        lambda value: value.cell_regions, request, jnp.zeros((4,), dtype=jnp.int64)
    )

    request.validate_source_integrity()
    with pytest.raises(ValueError, match="assignments"):
        changed.validate_source_integrity()


def test_prepared_acceptance_preserves_scientific_report_and_work() -> None:
    mesh, geometry, audit, prepared = prepared_tetrahedral_acceptance()
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
        prepared=phx.meshing.PreparedVolumeAcceptance(prepared),
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
    mesh, geometry, audit, prepared = prepared_tetrahedral_acceptance()
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
        mesh = mesh.with_coordinates(2.0 * TETRAHEDRON, numeric_version="changed")
        geometry, audit = plc_audit(
            mesh,
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
        )
    with pytest.raises(ValueError, match=r"MeshCertificationPreparedEvidence\.request"):
        phx.meshing.certify_meshing_acceptance(
            mesh,
            geometry,
            audit,
            schedule=request.schedule,
            domain=domain,
            cell_regions=regions,
            limits=limits,
            prepared=phx.meshing.PreparedVolumeAcceptance(prepared),
        )


def test_mapped_prepared_acceptance_keeps_source_limits_and_region_measure(
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, audit, initial = prepared_tetrahedral_acceptance()
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
    original_coordinates = geometry.source_coordinates(budget=reference_budget)
    support = prepare_mapped_plc_support(
        mesh,
        geometry,
        domain,
        np.asarray((17,), dtype=np.int64),
        np.asarray((17,), dtype=np.int64),
        maximum_support_queries=32768,
        validity=audit.validity,
        certification_request=request,
        budget=reference_budget,
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
        prepared=phx.meshing.PreparedVolumeAcceptance(support.prepared),
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
    assert geometry.source_coordinates(budget=reference_budget) == original_coordinates
    bounds = reused.coverage.achieved_region_measure_bounds[0]
    assert bounds is not None
    lower, upper = bounds
    assert lower <= 1.0 / 6.0 <= upper


def test_mapped_support_preserves_its_stricter_source_quota_across_queries(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, audit, initial = prepared_tetrahedral_acceptance()
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
        budget=reference_budget,
    )
    limits = phx.geometry.MeshCertificateLimits(
        maximum_work_units=measured.ledger.work_units + 96,
    )
    ambient = coordinate_budget(
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
            budget=ambient,
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
            budget=ambient,
        )
        assert ambient.work_units > original_start
        with pytest.raises(CoordinateEnclosureResourceError) as refused:
            # Every source-point query visits at least three rational
            # coordinates. The source quota must refuse before the larger
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
def test_mapped_prepared_acceptance_refuses_changed_declared_source_scope(
    mismatch: str,
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, audit, initial = prepared_tetrahedral_acceptance()
    domain = initial.request.domain
    assert isinstance(domain, phx.geometry.PiecewiseLinearDomain)
    source = phx.geometry.MeshingDomainBoundarySource(
        sphere_surface().domain, (0,), resolution=4
    )
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
        budget=reference_budget,
    )
    changed = (
        ((source, facets[1:2], 2.0),)
        if mismatch == "facets"
        else ((source, facets[:1], 1.5),)
    )
    with pytest.raises(ValueError, match=r"MeshCertificationPreparedEvidence\.request"):
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
            prepared=phx.meshing.PreparedVolumeAcceptance(support.prepared),
        )


def test_prepared_audit_refuses_a_different_validity_policy() -> None:
    mesh, geometry, _, prepared = prepared_tetrahedral_acceptance()
    policy = phx.meshing.CellMeshAuditPolicy(
        validity_policy=phx.discretization.CellValidityPolicy(
            maximum_subdivision_depth=1
        ),
    )
    with pytest.raises(ValueError, match="prepared_validity"):
        phx.meshing.audit_cell_mesh(
            mesh,
            geometry,
            policy=policy,
            prepared_validity=prepared.validity,
        )


def test_prepared_native_publication_refuses_authored_revision_drift(
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, request, proof = prepared_publication_authority(
        reference_budget=reference_budget
    )
    request = replace(request, prepared=proof.support.prepared)
    assert request.source_revision != request.domain.source_revision
    policy = phx.meshing.CellMeshAuditPolicy(
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
    )
    with (
        NativeExecutionScope(request.limits.native_limits()) as execution_scope,
        execution_scope.host_workspace() as workspace,
    ):
        published = publish_native_result(
            mesh,
            phx.SpatialCoordinateContract.si(),
            phx.meshing.MeshingComplianceReport("authoritative-publication"),
            (),
            phx.meshing.NativeMeshingProvider.info(),
            {"kind": "authoritative-publication"},
            request,
            policy=NativePublicationPolicy(
                policy, phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE, (), ()
            ),
            organization=NativePublicationOrganization(associations=proof.associations),
            execution=NativePublicationExecution(execution_scope, workspace, None, None),
            geometry=geometry,
        )
    assert published.certification is not None and published.certification.passed
    with pytest.raises(ValueError, match="publication.associations"):
        with (
            NativeExecutionScope(request.limits.native_limits()) as execution_scope,
            execution_scope.host_workspace() as workspace,
        ):
            publish_native_result(
                mesh,
                phx.SpatialCoordinateContract.si(),
                phx.meshing.MeshingComplianceReport("authoritative-publication"),
                (),
                phx.meshing.NativeMeshingProvider.info(),
                {"kind": "authoritative-publication"},
                replace(request, source_revision="foreign-authored-revision"),
                policy=NativePublicationPolicy(
                    policy, phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE, (), ()
                ),
                organization=NativePublicationOrganization(
                    associations=proof.associations
                ),
                execution=NativePublicationExecution(
                    execution_scope, workspace, None, None
                ),
                geometry=geometry,
            )


def test_prepared_coverage_refusal_transports_the_negative_scientific_certificate(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    mesh, geometry, audit, positive = prepared_tetrahedral_acceptance()
    original = positive.request.domain
    assert original is not None
    domain = phx.geometry.PiecewiseLinearDomain(
        TETRAHEDRON + np.asarray((2.0, 0.0, 0.0)),
        original.facets,
        original.facet_regions,
        original.region_ids,
        source_id="independent-shifted-coverage-source",
    )
    negative = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        domain,
        np.asarray((0,), dtype=np.int64),
        embedding=positive.embedding,
        limits=positive.request.limits,
    )
    ledger = coordinate_budget(2_000_000, 81_920_000)
    with ledger.activate(), pytest.raises(MeshCertificationPremiseFailure) as caught:
        phx.meshing.MeshCertificationPreparedEvidence(
            mesh,
            geometry,
            audit.validity,
            schedule=positive.request.schedule,
            domain=domain,
            cell_regions=np.asarray((0,), dtype=np.int64),
            embedding=positive.embedding,
            limits=positive.request.limits,
            budget=ledger,
        )
    failure = caught.value
    certificate = failure.coverage
    assert certificate.certificate_id == negative.certificate_id
    assert tuple(finding.finding_id for finding in certificate.findings) == tuple(
        finding.finding_id for finding in negative.findings
    )
    assert certificate.status == "violated" and certificate.findings
    assert certificate.binding.source_id == domain.source_id
    failure_domain = failure.request.domain
    if failure_domain is None or failure.ledger is None:
        raise RuntimeError(
            "fixture.coverage_failure must retain domain and active ledger; got missing evidence"
        )
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


def test_uncertain_prepared_refusal_keeps_unbounded_values_out_of_finite_evidence(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    mesh, geometry, audit, positive = prepared_tetrahedral_acceptance()
    request = positive.request
    assert request.domain is not None and request.cell_regions is not None
    ledger = coordinate_budget(0, 81_920_000)
    with ledger.activate():
        coverage = phx.geometry.certify_domain_coverage(
            mesh,
            geometry,
            request.domain,
            request.cell_regions,
            embedding=positive.embedding,
            limits=request.limits,
            budget=ledger,
        )
        assert coverage.status == "unresolved"
        assert any(
            value is None or not np.isfinite(value)
            for value in coverage.achieved_region_measures
        )
        with pytest.raises(MeshCertificationPremiseFailure) as caught:
            raise MeshCertificationPremiseFailure(
                mesh,
                geometry,
                request,
                audit.validity,
                positive.embedding,
                coverage,
                budget=ledger,
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


def test_public_failure_preserves_actual_fidelity_resource_quantities() -> None:

    mesh = octahedron()
    geometry, audit = plc_audit(
        mesh, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    sphere_surface = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 1.0, feature_id="sphere"
    ).compile(coordinate_unit=METER)
    source = phx.geometry.ImplicitBoundarySource(
        sphere_surface, source_id="limited-fidelity-sphere", spacing=0.2
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
    assert "distance_capacity" in finding_checks(report.fidelity, "unresolved")
    with pytest.raises(MeshCertificationReportFailure) as caught:
        report.require_passed()
    failure = caught.value
    assert failure.report.report_id == report.report_id
    assert failure.report.request.request_id == report.request.request_id
    failed_fidelity = failure.report.fidelity
    if failed_fidelity is None:
        raise RuntimeError(
            "fixture.fidelity_failure must retain source certificate; got None"
        )
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


def test_native_refusal_survives_fidelity_certificate_and_report_consumer() -> None:
    mesh = octahedron()
    geometry, audit = plc_audit(
        mesh, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    compiled = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    source = phx.geometry.ImplicitBoundarySource(
        compiled, source_id="native-refusal-fidelity-sphere", spacing=0.2
    )
    scientific = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("surface"),
        source=source,
        fidelity_tolerance=0.1,
    )
    original = scientific.fidelity
    assert original is not None and original.status == "violated"
    with pytest.raises(MeshCertificationReportFailure) as violated:
        scientific.require_passed()
    assert violated.value.category is phx.meshing.MeshingFailureCategory.AUDIT_FAILED
    assert violated.value.report.fidelity is original
    assert all(finding.native_refusal is None for finding in original.findings)

    scope = NativeExecutionScope(NativeExecutionLimits(20, 10, 8, 65536, 120))
    with pytest.raises(MeshcoreError) as caught:
        with scope:
            with scope.batch(
                12, queries_upper=6, cavity_upper=4, bytes_upper=4096
            ) as ticket:
                ticket.consume(3, geometry_queries=2, cavity_cells=2, managed_bytes=512)
                ticket.refuse(18, queries_upper=9, cavity_upper=9, bytes_upper=65537)
    error = caught.value
    assert error.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert scope.evidence is not None
    state = EmbeddingState([], [])
    state.add("distance_capacity", "unresolved", "mesh", resource_error=error)
    finding = state.findings[0]
    refusal = finding.native_refusal
    assert refusal is not None
    assert refusal.status is error.status
    assert refusal.admitted_upper == (12, 6, 4, 4096)
    assert refusal.requested_upper == (18, 9, 9, 65537)
    assert error.batch_refusal_evidence is not None
    assert refusal.root_receipt == tuple(
        int(value) for value in error.batch_refusal_evidence
    )
    assert refusal.root_receipt is not None
    assert refusal.root_receipt[4:6] == (3, 2)
    assert finding.resource is None
    assert finding.requested == finding.achieved == ()
    repeated = EmbeddingState([], [])
    repeated.add("distance_capacity", "unresolved", "mesh", resource_error=error)
    assert repeated.findings[0].finding_id == finding.finding_id
    assert (
        finding.finding_id
        != MeshCertificateFinding("distance_capacity", "unresolved", "mesh").finding_id
    )

    certificate = SourceFidelityCertificate(
        original.binding,
        tuple(state.findings),
        tolerance=original.tolerance,
        bounds=SourceFidelityBounds(
            semantics=("sampled", "sampled"),
            mesh_to_source=(np.inf, 0.0),
            source_to_mesh=(np.inf, 0.0),
            sample_counts=(0, 0),
        ),
        sample_order=original.sample_order,
    )
    outcomes = tuple(
        MeshCertificationOutcome(
            "source_fidelity",
            certificate.status,
            certificate.certificate_id,
            reasons=tuple(value.check for value in certificate.findings),
        )
        if outcome.check == "source_fidelity"
        else outcome
        for outcome in scientific.outcomes
    )
    report = MeshCertificationReport(
        scientific.schedule,
        mesh,
        geometry,
        audit,
        outcomes,
        embedding=scientific.embedding,
        coverage=scientific.coverage,
        fidelity=certificate,
        request=scientific.request,
    )
    assert certificate.status == "unresolved"
    assert certificate.findings[0].native_refusal is refusal
    assert report.failing_outcomes[0].evidence_id == certificate.certificate_id
    with pytest.raises(MeshCertificationReportFailure) as failed:
        report.require_passed()
    failure = failed.value
    assert failure.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert failure.report is report
    assert failure.report.fidelity is certificate
    assert failure.report.fidelity.findings[0].finding_id == finding.finding_id
    prefix = (
        f"certificate:{certificate.certificate_id}:finding:{finding.finding_id}"
        f":{finding.check}:native_batch"
    )
    requested = dict(failure.evidence.requested)
    achieved = dict(failure.evidence.achieved)
    assert achieved[f"{prefix}:status"] == int(error.status)
    dimensions = ("work_units", "geometry_queries", "cavity_cells", "managed_bytes")
    for name, admitted, wanted in zip(
        dimensions, refusal.admitted_upper, refusal.requested_upper, strict=True
    ):
        assert achieved[f"{prefix}:admitted_{name}"] == admitted
        assert requested[f"{prefix}:requested_{name}"] == wanted
    receipt_names = (
        *(f"remaining_{name}" for name in dimensions),
        "consumed_work_units",
        "consumed_geometry_queries",
        "peak_cavity_cells",
        "peak_managed_bytes",
    )
    for name, value in zip(receipt_names, refusal.root_receipt, strict=True):
        assert achieved[f"{prefix}:{name}"] == value
    with pytest.raises(MeshCertificationReportFailure) as again:
        report.require_passed()
    assert again.value.evidence.evidence_id == failure.evidence.evidence_id
    with pytest.raises(MeshCertificationReportFailure) as scientific_again:
        scientific.require_passed()
    assert (
        scientific_again.value.category is phx.meshing.MeshingFailureCategory.AUDIT_FAILED
    )
    retained_scientific = scientific_again.value.report.fidelity
    assert retained_scientific is not None
    assert original.certificate_id == retained_scientific.certificate_id
    assert tuple(value.finding_id for value in original.findings) == tuple(
        value.finding_id for value in retained_scientific.findings
    )


type _CanonicalAcceptance = tuple[
    phx.discretization.CellMesh,
    phx.discretization.CellGeometrySpec,
    phx.meshing.MeshCertificationPreparedEvidence,
    phx.meshing.MeshLineage,
    phx.meshing.CellMeshAuditPolicy,
]


@pytest.fixture(scope="module")
def canonical_tetrahedral_acceptance() -> _CanonicalAcceptance:
    mesh, geometry, _, prepared = prepared_tetrahedral_acceptance()
    lineage = phx.meshing.MeshLineage(
        mesh.topology_id,
        mesh.topology_id,
        tuple(
            phx.meshing.EntityLineage(
                entities.intrinsic_dimension,
                entities.entity_set_id,
                entities.entity_set_id,
                entities.entity_ids,
                entities.entity_ids,
                np.full(
                    entities.entity_ids.shape,
                    phx.meshing.EntityLineageKind.PRESERVED,
                    dtype=np.int32,
                ),
            )
            for entities in mesh.topology.entity_sets
        ),
    )
    policy = phx.meshing.CellMeshAuditPolicy(
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    return mesh, geometry, prepared, lineage, policy


@pytest.mark.parametrize("kind", ("volume", "boundary"))
def test_canonical_certification_prepared_variants_preserve_acceptance_identity(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance, kind: str
) -> None:
    mesh, geometry, evidence, lineage, policy = canonical_tetrahedral_acceptance
    prepared = (
        phx.meshing.PreparedVolumeAcceptance(evidence)
        if kind == "volume"
        else phx.meshing.PreparedBoundaryAcceptance(embedding=evidence.embedding)
    )
    cold = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
        audit_policy=policy,
        certification_inputs=evidence.request,
        lineage=lineage,
    )
    reused = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
        audit_policy=policy,
        certification_inputs=evidence.request,
        lineage=lineage,
        certification_prepared=prepared,
    )
    assert reused.certification is not None and cold.certification is not None
    assert reused.certification.report_id == cold.certification.report_id
    assert reused.audit.report_id == cold.audit.report_id
    assert reused.trace.trace_id == cold.trace.trace_id
    assert reused.result_id == cold.result_id


def test_canonical_volume_preparation_owns_audit_validity(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance,
) -> None:
    mesh, geometry, evidence, lineage, policy = canonical_tetrahedral_acceptance
    result = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
        audit_policy=policy,
        certification_inputs=evidence.request,
        lineage=lineage,
        certification_prepared=phx.meshing.PreparedVolumeAcceptance(evidence),
        audit_prepared_validity=evidence.validity,
    )
    assert result.audit.validity is evidence.validity
    assert result.certification is not None
    assert result.certification.embedding is evidence.embedding
    assert result.certification.coverage is evidence.coverage


@pytest.mark.parametrize("policy", (None, phx.meshing.CellMeshAuditPolicy()))
def test_canonical_publication_executes_retained_volume_closure(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance,
    policy: phx.meshing.CellMeshAuditPolicy | None,
) -> None:
    mesh, geometry, evidence, lineage, required_policy = canonical_tetrahedral_acceptance
    result = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
        audit_policy=policy,
        certification_inputs=evidence.request,
        lineage=lineage,
        certification_prepared=phx.meshing.PreparedVolumeAcceptance(evidence),
    )
    assert result.certification is not None and result.certification.passed
    assert result.audit.policy_id == required_policy.policy_id
    assert "open_boundary" in result.audit.evaluated_checks


def test_canonical_retained_volume_still_refuses_a_tampered_skipped_audit(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance,
) -> None:
    mesh, geometry, evidence, lineage, _ = canonical_tetrahedral_acceptance
    skipped = phx.meshing.audit_cell_mesh(mesh, geometry)
    with pytest.raises(ValueError, match="open_boundary"):
        evidence.request.recertify_transition(
            mesh,
            geometry,
            skipped,
            lineage=lineage,
            prepared=phx.meshing.PreparedVolumeAcceptance(evidence),
        )


@pytest.mark.parametrize("kind", ("volume", "boundary"))
def test_canonical_prepared_evidence_requires_its_transition_request(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance, kind: str
) -> None:
    mesh, geometry, evidence, _, policy = canonical_tetrahedral_acceptance
    prepared = (
        phx.meshing.PreparedVolumeAcceptance(evidence)
        if kind == "volume"
        else phx.meshing.PreparedBoundaryAcceptance(embedding=evidence.embedding)
    )
    with pytest.raises(ValueError, match="certification_prepared"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
            audit_policy=policy,
            certification_prepared=prepared,
        )


def test_canonical_prepared_validity_must_agree_with_volume_evidence(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance,
) -> None:
    mesh, geometry, evidence, lineage, policy = canonical_tetrahedral_acceptance
    different = phx.discretization.certify_cell_geometry_validity(
        geometry,
        mesh=mesh,
        policy=phx.discretization.CellValidityPolicy(maximum_subdivision_depth=0),
    )
    assert different.certificate_id != evidence.validity.certificate_id
    with pytest.raises(ValueError, match="audit_prepared_validity"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
            audit_policy=policy,
            certification_inputs=evidence.request,
            lineage=lineage,
            certification_prepared=phx.meshing.PreparedVolumeAcceptance(evidence),
            audit_prepared_validity=different,
        )


def test_canonical_checked_preparation_refuses_the_unwrapped_volume_evidence(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance,
) -> None:
    mesh, geometry, evidence, lineage, policy = canonical_tetrahedral_acceptance
    with pytest.raises(TypeError, match="certification_prepared"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
            audit_policy=policy,
            certification_inputs=evidence.request,
            lineage=lineage,
            # ty: ignore[invalid-argument-type]
            certification_prepared=evidence,
        )
