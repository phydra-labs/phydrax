#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Route-specific acceptance schedules over separately owned mesh evidence.

This module orchestrates; it decides nothing geometric itself. Mapped-cell
validity comes from the Bernstein certificate, topology and sampled quality
from the cell-mesh audit, and global embedding, domain/interface coverage and
source fidelity from the geometry-owned mesh certificates. A schedule fixes
which of these a route must certify; every required check must be
``certified`` for the report to pass. Unresolved and violated checks stay
distinct and carry their failing entities and requested/achieved quantities.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import final

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.geometry._mapped_reference_domain as _dependency_phydrax_geometry__mapped_reference_domain
import phydrax.geometry._mesh_certificates as _dependency_phydrax_geometry__mesh_certificates
import phydrax.geometry._source_boundary_records as source_boundary_records_owner
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.meshing._audit as _dependency_phydrax_meshing__audit
import phydrax.meshing._certification_inputs as _dependency_phydrax_meshing__certification_inputs
import phydrax.meshing._certification_schedule as _leaf_phydrax_meshing__certification_schedule
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._trace as _dependency_phydrax_meshing__trace
from phydrax.meshing._failures import MeshingFailure

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..typing import checked, parse


_QUALITY_ISSUES = ("minimum_measure", "minimum_mean_ratio", "maximum_aspect_ratio")
_VALIDITY_NAMES = ("invalid_geometry", "unresolved_geometry_validity")
_LARGEST = float(np.finfo(np.float64).max)


@final
class MeshCertificationOutcome(StrictModule, NonTrainableState):
    """Status of one scheduled check with its evidence and failing entities."""

    check: _leaf_phydrax_meshing__certification_schedule.MeshCertificationCheck = (
        eqx.field(static=True)
    )
    status: _dependency_phydrax_geometry__mesh_certificates.MeshCertificateStatus = (
        eqx.field(static=True)
    )
    evidence_id: str = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    entity_kind: _dependency_phydrax_geometry__embedding_state.MeshCertificateEntityKind = eqx.field(
        static=True
    )
    entity_ids: tuple[int, ...] = eqx.field(static=True)
    outcome_id: str = eqx.field(static=True)

    def __init__(
        self,
        check: _leaf_phydrax_meshing__certification_schedule.MeshCertificationCheck,
        status: _dependency_phydrax_geometry__mesh_certificates.MeshCertificateStatus,
        evidence_id: str,
        /,
        *,
        reasons: tuple[str, ...] = (),
        entity_kind: _dependency_phydrax_geometry__embedding_state.MeshCertificateEntityKind = "mesh",
        entity_ids: tuple[int, ...] = (),
    ) -> None:
        self.check = parse(
            check,
            _leaf_phydrax_meshing__certification_schedule.MeshCertificationCheck,
            "check",
        )
        self.status = parse(
            status,
            _dependency_phydrax_geometry__mesh_certificates.MeshCertificateStatus,
            "status",
        )
        self.evidence_id = str(evidence_id)
        self.reasons = tuple(dict.fromkeys(str(value) for value in reasons))
        self.entity_kind = parse(
            entity_kind,
            _dependency_phydrax_geometry__embedding_state.MeshCertificateEntityKind,
            "entity_kind",
        )
        self.entity_ids = tuple(sorted({int(value) for value in entity_ids}))
        if (self.status == "certified") != (not self.reasons):
            raise ValueError(
                f"MeshCertificationOutcome.reasons must be empty exactly when status is certified; got status={self.status!r}, reasons={self.reasons!r}"
            )
        self.outcome_id = canonical_fingerprint(
            {
                "kind": "mesh-certification-outcome",
                "check": self.check,
                "status": self.status,
                "evidence": self.evidence_id,
                "reasons": self.reasons,
                "entity_kind": self.entity_kind,
                "entity_ids": self.entity_ids,
            }
        )


@final
class MeshCertificationReport(StrictModule, NonTrainableState):
    """Route acceptance verdict bound to one mesh, geometry and audit.

    ``passed`` holds only when every required check is certified.
    ``requested``/``achieved`` follow the ``MeshingComplianceReport``
    convention for coverage measures and fidelity deviations.

    ``scoped_fidelity`` retains each source-stratum/target-trace proof
    independently of whole-boundary fidelity and domain coverage. The retained
    request binds source query scope, global facet IDs, and physical tolerance;
    every scoped verdict participates in acceptance and report identity.
    Renewal maps facet lineage rather than reusing old bounds.
    """

    schedule: _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule
    request: _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs
    mesh_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    audit_report_id: str = eqx.field(static=True)
    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None
    coverage: domain_coverage_owner.DomainCoverageCertificate | None
    fidelity: source_fidelity_owner.SourceFidelityCertificate | None
    scoped_fidelity: tuple[source_fidelity_owner.SourceFidelityCertificate, ...]
    outcomes: tuple[MeshCertificationOutcome, ...]
    requested: tuple[tuple[str, float], ...] = eqx.field(static=True)
    achieved: tuple[tuple[str, float], ...] = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        schedule: _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport,
        outcomes: tuple[MeshCertificationOutcome, ...],
        /,
        *,
        embedding: global_embedding_owner.GlobalEmbeddingCertificate | None,
        coverage: domain_coverage_owner.DomainCoverageCertificate | None,
        fidelity: source_fidelity_owner.SourceFidelityCertificate | None,
        request: _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs,
        scoped_fidelity: tuple[source_fidelity_owner.SourceFidelityCertificate, ...] = (),
    ) -> None:
        if (
            request.mesh_id != mesh.mesh_id
            or request.geometry_id
            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                geometry
            )
            or request.schedule.schedule_id != schedule.schedule_id
        ):
            raise ValueError(
                f"MeshCertificationReport.request must bind mesh, geometry and schedule; got request_id={request.request_id!r}"
            )
        if {outcome.check for outcome in outcomes} != set(schedule.required_checks):
            raise ValueError(
                f"MeshCertificationReport.outcomes must cover the required checks; got {tuple(outcome.check for outcome in outcomes)!r}"
            )
        if len(scoped_fidelity) != len(request.scoped_fidelity):
            raise ValueError(
                f"MeshCertificationReport.scoped_fidelity must cover every retained source/target request; got {len(scoped_fidelity)} certificates for {len(request.scoped_fidelity)} requests"
            )
        for certificate, (query, identifiers, tolerance) in zip(
            scoped_fidelity, request.scoped_fidelity, strict=True
        ):
            if not isinstance(
                certificate,
                source_fidelity_owner.SourceFidelityCertificate,
            ):
                raise TypeError(
                    f"MeshCertificationReport.scoped_fidelity must contain SourceFidelityCertificate records; got {type(certificate).__name__}"
                )
            certificate.binding.require(mesh, geometry)
            if (
                certificate.source_scope_id != query.source_scope_id
                or certificate.target_facet_ids
                != tuple(np.asarray(identifiers, dtype=np.int64).tolist())
                or certificate.tolerance != tolerance
                or (certificate.binding.source_id, certificate.binding.source_revision)
                != (query.source_id, query.source_revision)
            ):
                raise ValueError(
                    f"MeshCertificationReport.scoped_fidelity must bind its source stratum, target trace and physical tolerance; got certificate_id={certificate.certificate_id!r}"
                )
            expected = _certificate_outcome("source_fidelity", certificate)
            if not any(value.outcome_id == expected.outcome_id for value in outcomes):
                raise ValueError(
                    f"MeshCertificationReport.outcomes must retain scoped fidelity status; got certificate status={certificate.status!r}"
                )
        requested: list[tuple[str, float]] = []
        achieved: list[tuple[str, float]] = []
        if coverage is not None:
            for region, wanted, measured in zip(
                coverage.region_ids,
                coverage.requested_region_measures,
                coverage.achieved_region_measures,
                strict=True,
            ):
                requested.append((f"region_measure[{region}]", wanted))
                achieved.append(
                    (
                        f"region_measure[{region}]",
                        np.inf if measured is None else measured,
                    )
                )
        if fidelity is not None:
            for name, measured in (
                ("mesh_to_source_deviation", fidelity.mesh_to_source_upper),
                ("source_to_mesh_deviation", fidelity.source_to_mesh_upper),
            ):
                requested.append((name, fidelity.tolerance))
                achieved.append((name, measured))
        for certificate in scoped_fidelity:
            for name, measured in (
                ("mesh_to_source_deviation", certificate.mesh_to_source_upper),
                ("source_to_mesh_deviation", certificate.source_to_mesh_upper),
            ):
                key = f"scope:{certificate.source_scope_id}:{name}"
                requested.append((key, certificate.tolerance))
                achieved.append((key, measured))
        resource_requested, resource_achieved = _certificate_finding_quantities(
            (embedding, coverage, fidelity, *scoped_fidelity)
        )
        requested.extend(resource_requested)
        achieved.extend(resource_achieved)
        self.schedule = schedule
        self.request = request
        self.mesh_id = mesh.mesh_id
        self.topology_id = mesh.topology_id
        self.geometry_id = (
            _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                geometry
            )
        )
        self.geometry_layout_id = geometry.geometry_layout_id
        self.audit_report_id = audit.report_id
        self.embedding = embedding
        self.coverage = coverage
        self.fidelity = fidelity
        self.scoped_fidelity = tuple(scoped_fidelity)
        self.outcomes = tuple(outcomes)
        self.requested = tuple(requested)
        self.achieved = tuple(achieved)
        self.passed = all(outcome.status == "certified" for outcome in outcomes)
        self.report_id = canonical_fingerprint(
            {
                "kind": "mesh-certification-report",
                "schedule": schedule.schedule_id,
                "request": request.request_id,
                "mesh": self.mesh_id,
                "geometry": self.geometry_id,
                "audit": self.audit_report_id,
                "outcomes": [outcome.outcome_id for outcome in outcomes],
                "scoped_fidelity": tuple(
                    certificate.certificate_id for certificate in scoped_fidelity
                ),
                # repr preserves infinite fidelity bounds in the fingerprint payload.
                "requested": [(name, repr(value)) for name, value in self.requested],
                "achieved": [(name, repr(value)) for name, value in self.achieved],
            }
        )

    @property
    def failing_outcomes(self) -> tuple[MeshCertificationOutcome, ...]:
        return tuple(value for value in self.outcomes if value.status != "certified")

    def require_passed(self, /) -> None:
        """Raise ``MeshingFailure`` with the failing entities and quantities."""

        failing = self.failing_outcomes
        if not failing:
            return
        raise MeshCertificationReportFailure(self)


def _require_audit(
    schedule: _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule,
    audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport,
    /,
) -> None:
    missing = tuple(
        name
        for name in schedule.required_audit_checks
        if name not in audit.evaluated_checks
    )
    if missing:
        raise ValueError(
            f"certify_meshing_acceptance.audit must evaluate required route checks; got missing checks={missing!r}"
        )


def _validity_outcome(
    audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport, /
) -> MeshCertificationOutcome:
    validity = audit.validity
    status = np.asarray(validity.status)
    failing = dict(audit.failing_cells)
    if np.any(
        status
        == _dependency_phydrax_discretization__cell_geometry_validity.CellValidityStatus.INVALID
    ):
        return MeshCertificationOutcome(
            "cell_validity",
            "violated",
            validity.certificate_id,
            reasons=("invalid_geometry",),
            entity_kind="cell",
            entity_ids=failing["invalid_geometry"],
        )
    if np.any(
        status
        == _dependency_phydrax_discretization__cell_geometry_validity.CellValidityStatus.UNRESOLVED
    ):
        return MeshCertificationOutcome(
            "cell_validity",
            "unresolved",
            validity.certificate_id,
            reasons=tuple(reason for _, reason in validity.unresolved_reasons)
            or ("unresolved_geometry_validity",),
            entity_kind="cell",
            entity_ids=failing["unresolved_geometry_validity"],
        )
    return MeshCertificationOutcome("cell_validity", "certified", validity.certificate_id)


def _topology_outcome(
    audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport,
    excused: tuple[str, ...],
    /,
) -> MeshCertificationOutcome:
    counted = tuple(
        name
        for name, count in audit.check_counts
        if count
        and name not in _VALIDITY_NAMES
        and name not in excused
        and not name.startswith("unresolved_")
    )
    issues = tuple(
        name
        for name in audit.issues
        if name not in _QUALITY_ISSUES
        and name not in _VALIDITY_NAMES
        and not name.startswith("unresolved_")
    )
    if counted or issues:
        return MeshCertificationOutcome(
            "topology", "violated", audit.report_id, reasons=(*issues, *counted)
        )
    unresolved = tuple(name for name in audit.unresolved if name != "geometry_validity")
    if unresolved:
        return MeshCertificationOutcome(
            "topology", "unresolved", audit.report_id, reasons=unresolved
        )
    return MeshCertificationOutcome("topology", "certified", audit.report_id)


def _quality_outcome(
    audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport, /
) -> MeshCertificationOutcome:
    failing = tuple(name for name in audit.issues if name in _QUALITY_ISSUES)
    if failing:
        return MeshCertificationOutcome(
            "quality",
            "violated",
            audit.quality.report_id,
            reasons=failing,
            entity_kind="cell",
            entity_ids=audit.quality.worst_cell_global_ids,
        )
    return MeshCertificationOutcome("quality", "certified", audit.quality.report_id)


def _certificate_outcome(
    check: _leaf_phydrax_meshing__certification_schedule.MeshCertificationCheck,
    certificate: global_embedding_owner.GlobalEmbeddingCertificate
    | domain_coverage_owner.DomainCoverageCertificate
    | source_fidelity_owner.SourceFidelityCertificate,
    /,
) -> MeshCertificationOutcome:
    if certificate.status == "certified":
        return MeshCertificationOutcome(check, "certified", certificate.certificate_id)
    relevant = tuple(
        value for value in certificate.findings if value.status == certificate.status
    )
    kinds = {value.entity_kind for value in relevant}
    kind: _dependency_phydrax_geometry__embedding_state.MeshCertificateEntityKind = (
        kinds.pop() if len(kinds) == 1 else "mesh"
    )
    return MeshCertificationOutcome(
        check,
        certificate.status,
        certificate.certificate_id,
        reasons=tuple(value.check for value in relevant),
        entity_kind=kind,
        entity_ids=tuple(
            identifier
            for value in relevant
            if value.entity_kind == kind
            for identifier in value.entity_ids
        ),
    )


def _certificate_finding_quantities(
    certificates: tuple[
        global_embedding_owner.GlobalEmbeddingCertificate
        | domain_coverage_owner.DomainCoverageCertificate
        | source_fidelity_owner.SourceFidelityCertificate
        | None,
        ...,
    ],
    /,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Return stored finding quantities and named native batch counter values."""
    requested: dict[str, int] = {}
    achieved: dict[str, int] = {}
    for certificate in certificates:
        if certificate is None:
            continue
        for finding in certificate.findings:
            if (
                not finding.requested
                and not finding.achieved
                and not finding.observations
                and finding.native_refusal is None
            ):
                continue
            prefix = (
                f"certificate:{certificate.certificate_id}:finding:{finding.finding_id}"
                f":{finding.check}"
            )
            if finding.resource is not None:
                prefix += f":resource:{finding.resource}"
            requested.update(
                (f"{prefix}:{key}", value)
                for key, value in finding.requested
                if np.isfinite(value)
            )
            achieved.update(
                (f"{prefix}:{key}", value)
                for key, value in finding.achieved
                if np.isfinite(value)
            )
            achieved.update(
                (f"{prefix}:observation:{key}", value)
                for key, value in finding.observations
                if np.isfinite(value)
            )
            refusal = finding.native_refusal
            if refusal is None:
                continue
            native_prefix = f"{prefix}:native_batch"
            achieved[f"{native_prefix}:status"] = int(refusal.status)
            dimensions = (
                "work_units",
                "geometry_queries",
                "cavity_cells",
                "managed_bytes",
            )
            if refusal.requested_upper is not None:
                requested.update(
                    (f"{native_prefix}:requested_{name}", value)
                    for name, value in zip(
                        dimensions, refusal.requested_upper, strict=True
                    )
                )
            if refusal.admitted_upper is not None:
                achieved.update(
                    (f"{native_prefix}:admitted_{name}", value)
                    for name, value in zip(
                        dimensions, refusal.admitted_upper, strict=True
                    )
                )
            if refusal.root_receipt is not None:
                receipt_names = (
                    *(f"remaining_{name}" for name in dimensions),
                    "consumed_work_units",
                    "consumed_geometry_queries",
                    "peak_cavity_cells",
                    "peak_managed_bytes",
                )
                achieved.update(
                    (f"{native_prefix}:{name}", value)
                    for name, value in zip(
                        receipt_names, refusal.root_receipt, strict=True
                    )
                )
    return tuple(sorted(requested.items())), tuple(sorted(achieved.items()))


class MeshCertificationReportFailure(MeshingFailure):
    """Report failure quantities from the failing certification report."""

    @checked
    def __init__(self, report: MeshCertificationReport, /) -> None:
        self.report = report
        self.unbounded_quantities = tuple(
            (bank, name, value)
            for bank, values in (
                ("requested", report.requested),
                ("achieved", report.achieved),
            )
            for name, value in values
            if not np.isfinite(value)
        )
        requested = {
            name: value for name, value in report.requested if np.isfinite(value)
        }
        achieved = {name: value for name, value in report.achieved if np.isfinite(value)}
        finding_requested, finding_achieved = _certificate_finding_quantities(
            (report.embedding, report.coverage, report.fidelity, *report.scoped_fidelity)
        )
        requested.update(finding_requested)
        achieved.update(finding_achieved)
        failing = report.failing_outcomes
        blocking_unresolved = {
            outcome.evidence_id for outcome in failing if outcome.status == "unresolved"
        }
        resource_unresolved = any(
            finding.status == "unresolved"
            and (finding.native_refusal is not None or finding.resource is not None)
            for certificate in (
                report.embedding,
                report.coverage,
                report.fidelity,
                *report.scoped_fidelity,
            )
            if certificate is not None
            and certificate.certificate_id in blocking_unresolved
            for finding in certificate.findings
        )
        category = _leaf_phydrax_meshing__failures.MeshingFailureCategory.AUDIT_FAILED
        if resource_unresolved and not any(
            outcome.status == "violated" for outcome in failing
        ):
            category = (
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
            )
        super().__init__(
            category,
            "MeshCertificationReport.outcomes must all be certified; got "
            + "; ".join(
                f"{value.check} {value.status}: {', '.join(value.reasons)}"
                for value in failing
            ),
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.CERTIFICATION.value,
            entity_ids=tuple(
                identifier
                for value in failing
                if value.entity_kind in ("cell", "facet")
                for identifier in value.entity_ids
            ),
            requested=tuple(sorted(requested.items())),
            achieved=tuple(sorted(achieved.items())),
        )


class MeshCertificationPremiseFailure(MeshingFailure):
    """Describe failed volume-preparation checks and their quantities."""

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        request: _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs,
        validity: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate,
        embedding: global_embedding_owner.GlobalEmbeddingCertificate,
        coverage: domain_coverage_owner.DomainCoverageCertificate,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> None:

        self.mesh = mesh
        self.geometry = geometry
        self.request = request
        self.validity = validity
        self.embedding = embedding
        self.coverage = coverage
        self.ledger = budget
        requested, achieved = _certificate_finding_quantities((embedding, coverage))
        requested = (
            *requested,
            *(
                (f"region_measure[{region}]", wanted)
                for region, wanted in zip(
                    coverage.region_ids, coverage.requested_region_measures, strict=True
                )
                if np.isfinite(wanted)
            ),
        )
        achieved = (
            *achieved,
            *(
                (f"region_measure[{region}]", measured)
                for region, measured in zip(
                    coverage.region_ids, coverage.achieved_region_measures, strict=True
                )
                if measured is not None and np.isfinite(measured)
            ),
        )
        self.unbounded_region_measures = (
            *(
                (region, "requested", wanted)
                for region, wanted in zip(
                    coverage.region_ids, coverage.requested_region_measures, strict=True
                )
                if not np.isfinite(wanted)
            ),
            *(
                (region, "achieved", measured)
                for region, measured in zip(
                    coverage.region_ids, coverage.achieved_region_measures, strict=True
                )
                if measured is None or not np.isfinite(measured)
            ),
        )
        # Optional measures, bounds, SCI and reference source data stay in
        # coverage and geometry rather than this failure summary.
        outcome = _certificate_outcome("domain_coverage", coverage)
        super().__init__(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.AUDIT_FAILED,
            f"MeshCertificationPreparedEvidence.coverage must be certified; got domain_coverage {coverage.status}: {', '.join(outcome.reasons)}",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.CERTIFICATION.value,
            provider_code=coverage.certificate_id,
            entity_ids=tuple(
                identifier
                for finding in coverage.findings
                if finding.entity_kind in ("cell", "facet")
                for identifier in finding.entity_ids
            ),
            requested=requested,
            achieved=achieved,
        )


@contextmanager
def _certificate_budget(
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits | None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None,
    /,
) -> Iterator[
    _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
]:
    """Reuse explicit request accounting or settle one standalone host request."""
    if budget is not None:
        with budget.activate():
            yield budget
        return
    limits_ = (
        _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits()
        if limits is None
        else limits
    )
    with HostBatchTicket(
        limits_.maximum_work_units,
        queries_upper=limits_.maximum_candidate_pairs + limits_.maximum_ray_tests,
        bytes_upper=limits_.maximum_scratch_bytes,
    ) as ticket:
        ledger = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits_.maximum_work_units,
            maximum_memory_bytes=limits_.maximum_scratch_bytes,
        )
        with ledger.activate():
            yield ledger


@final
class MeshCertificationPreparedEvidence(StrictModule, NonTrainableState):
    """Prepare volume coverage and embedding under one source request.

    Coverage binds the cell-region assignments as well as the declared domain.
    Publication reuses these certificates and their measured work counters.
    Fidelity, topology and quality remain independently evaluated by acceptance.
    """

    request: _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs
    validity: (
        _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
    )
    embedding: global_embedding_owner.GlobalEmbeddingCertificate
    coverage: domain_coverage_owner.DomainCoverageCertificate
    mesh_arrays_id: str = eqx.field(static=True)
    validity_arrays_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        validity: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate,
        /,
        *,
        schedule: _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule,
        domain: domain_coverage_owner.PiecewiseLinearDomain
        | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
        cell_regions: ArrayLike,
        embedding: global_embedding_owner.GlobalEmbeddingCertificate | None = None,
        limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
        | None = None,
        source: source_boundary_records_owner.SourceBoundaryQuery | None = None,
        fidelity_tolerance: float | None = None,
        fidelity_sample_order: int = 4,
        scoped_fidelity: tuple[
            tuple[
                source_boundary_records_owner.SourceBoundaryQuery,
                ArrayLike,
                float,
            ],
            ...,
        ] = (),
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
        | None = None,
    ) -> None:
        if "domain_coverage" not in schedule.required_checks:
            raise ValueError(
                f"MeshCertificationPreparedEvidence.schedule must require domain coverage; got route={schedule.route!r}"
            )
        if domain is None or cell_regions is None:
            raise ValueError(
                "MeshCertificationPreparedEvidence.coverage must supply both domain and cell_regions; got missing coverage inputs"
            )
        if (source is None) != (fidelity_tolerance is None) or (source is not None) != (
            "source_fidelity" in schedule.required_checks
        ):
            raise ValueError(
                f"MeshCertificationPreparedEvidence.source must supply exactly the schedule's source inputs; got route={schedule.route!r}, source_present={source is not None}, tolerance={fidelity_tolerance!r}"
            )
        request = (
            _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs(
                mesh,
                geometry,
                schedule,
                domain=domain,
                cell_regions=cell_regions,
                source=source,
                fidelity_tolerance=fidelity_tolerance,
                fidelity_sample_order=fidelity_sample_order,
                limits=limits,
                scoped_fidelity=scoped_fidelity,
                budget=budget,
            )
        )
        with _certificate_budget(request.limits, budget) as budget:
            request.validate_source_integrity(budget=budget)
            validity.require_bound(geometry, mesh=mesh)
            if (
                not validity.all_certified
                or not np.all(
                    np.asarray(validity.status)
                    == _dependency_phydrax_discretization__cell_geometry_validity.CellValidityStatus.CERTIFIED_VALID
                )
                or validity.status.shape[0]
                != sum(block.vertices.shape[0] for block in mesh.blocks)
            ):
                raise ValueError(
                    f"MeshCertificationPreparedEvidence.validity must certify all mesh cells; got all_certified={validity.all_certified!r}, status_count={validity.status.shape[0]}"
                )
            embedding_ = (
                global_embedding_owner.certify_global_embedding(
                    mesh,
                    geometry,
                    validity,
                    limits=request.limits,
                    budget=budget,
                )
                if embedding is None
                else embedding
            )
            require_prepared_embedding(
                mesh, geometry, validity, embedding_, request.limits
            )
            if request.domain is None or request.cell_regions is None:
                raise ValueError(
                    "MeshCertificationPreparedEvidence.request must retain domain and cell_regions; got missing coverage inputs"
                )
            coverage = domain_coverage_owner.certify_domain_coverage(
                mesh,
                geometry,
                request.domain,
                request.cell_regions,
                embedding=embedding_,
                limits=request.limits,
                budget=budget,
            )
            if coverage.status != "certified" or coverage.findings:
                raise MeshCertificationPremiseFailure(
                    mesh, geometry, request, validity, embedding_, coverage, budget=budget
                )
            _require_prepared_coverage(coverage, request.limits)
            self.request = request
            self.validity = validity
            self.embedding = embedding_
            self.coverage = coverage
            self.mesh_arrays_id = canonical_fingerprint(array_tree_fingerprint(mesh))
            self.validity_arrays_id = canonical_fingerprint(
                array_tree_fingerprint(validity)
            )
            self.evidence_id = canonical_fingerprint(
                {
                    "kind": "mesh-certification-prepared-evidence",
                    "request": request.request_id,
                    "validity": validity.certificate_id,
                    "embedding": embedding_.certificate_id,
                    "coverage": coverage.certificate_id,
                    "mesh_arrays": self.mesh_arrays_id,
                    "validity_arrays": self.validity_arrays_id,
                }
            )

    @checked
    def require(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        request: _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
        | None = None,
    ) -> None:
        """Validate retained volume evidence against the request and target bindings."""
        self.request.validate_source_integrity(budget=budget)
        if request.request_id != self.request.request_id or not tree_content_equal(
            request, self.request
        ):
            raise ValueError(
                f"MeshCertificationPreparedEvidence.request must match the acceptance request; got prepared_id={self.request.request_id!r}, acceptance_id={request.request_id!r}"
            )
        self.validity.require_bound(geometry, mesh=mesh)
        if (
            not self.validity.all_certified
            or not np.all(
                np.asarray(self.validity.status)
                == _dependency_phydrax_discretization__cell_geometry_validity.CellValidityStatus.CERTIFIED_VALID
            )
            or canonical_fingerprint(array_tree_fingerprint(mesh)) != self.mesh_arrays_id
            or canonical_fingerprint(array_tree_fingerprint(self.validity))
            != self.validity_arrays_id
        ):
            raise ValueError(
                "MeshCertificationPreparedEvidence.bindings must retain mesh, geometry and validity values; got changed numerical evidence"
            )
        require_prepared_embedding(
            mesh,
            geometry,
            self.validity,
            self.embedding,
            request.limits,
        )
        coverage = self.coverage
        if not isinstance(
            coverage,
            domain_coverage_owner.DomainCoverageCertificate,
        ):
            raise TypeError(
                f"MeshCertificationPreparedEvidence.coverage must be DomainCoverageCertificate; got {type(coverage).__name__}"
            )
        coverage.binding.require(mesh, geometry)
        _require_prepared_coverage(coverage, request.limits)
        domain = request.domain
        if (
            domain is None
            or coverage.status != "certified"
            or coverage.findings
            or coverage.domain_id != domain.domain_id
            or coverage.region_ids != domain.region_ids
            or coverage.embedding_certificate_id != self.embedding.certificate_id
            or coverage.binding.limits_id != request.limits.limits_id
            or coverage.binding.source_id != domain.source_id
            or coverage.binding.source_revision != domain.source_revision
            or coverage.source_facet_count != domain.facets.shape[0]
            or coverage.covered_source_facet_count != coverage.source_facet_count
        ):
            raise ValueError(
                f"MeshCertificationPreparedEvidence.coverage must bind the governed domain and region assignments; got certificate_id={coverage.certificate_id!r}"
            )
        rebuilt = domain_coverage_owner.DomainCoverageCertificate(
            coverage.binding,
            coverage.embedding_certificate_id,
            domain,
            coverage.findings,
            measures=domain_coverage_owner.CoverageMeasures(
                coverage.requested_region_measures,
                coverage.achieved_region_measures,
                coverage.requested_region_measure_bounds,
                coverage.achieved_region_measure_bounds,
                coverage.integration_error_bounds,
            ),
            evidence=domain_coverage_owner.CoverageEvidence(
                coverage.covered_source_facet_count,
                coverage.facet_source_overlaps,
                coverage.premise_certificate_ids,
                coverage.candidate_pair_count,
                coverage.subdivision_piece_count,
                coverage.maximum_subdivision_depth_reached,
            ),
            expression_work=domain_coverage_owner.CoverageExpressionWork(
                coverage.source_expression_work_units,
                coverage.source_expression_required_work_units,
                coverage.source_expression_peak_bytes,
            ),
        )
        identity = canonical_fingerprint(
            {
                "kind": "mesh-certification-prepared-evidence",
                "request": self.request.request_id,
                "validity": self.validity.certificate_id,
                "embedding": self.embedding.certificate_id,
                "coverage": rebuilt.certificate_id,
                "mesh_arrays": self.mesh_arrays_id,
                "validity_arrays": self.validity_arrays_id,
            }
        )
        if not tree_content_equal(coverage, rebuilt) or identity != self.evidence_id:
            raise ValueError(
                f"MeshCertificationPreparedEvidence.evidence_id must match its numerical proof records; got {self.evidence_id!r}"
            )


def _require_prepared_coverage(
    coverage: domain_coverage_owner.DomainCoverageCertificate,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    /,
) -> None:
    if coverage.status != "certified" or coverage.findings:
        raise ValueError(
            f"MeshCertificationPreparedEvidence.coverage must be certified without findings; got status={coverage.status!r}, finding_count={len(coverage.findings)}"
        )
    for measured, bounds, error in zip(
        coverage.achieved_region_measures,
        coverage.achieved_region_measure_bounds,
        coverage.integration_error_bounds,
        strict=True,
    ):
        if (
            measured is None
            or bounds is None
            or error is None
            or not np.isfinite(measured)
            or not all(np.isfinite(value) for value in bounds)
            or not np.isfinite(error)
        ):
            raise ValueError(
                f"MeshCertificationPreparedEvidence.coverage.measures must contain finite measurements, bounds and errors; got measurement={measured!r}, bounds={bounds!r}, error={error!r}"
            )
    if (
        min(
            coverage.candidate_pair_count,
            coverage.subdivision_piece_count,
            coverage.maximum_subdivision_depth_reached,
        )
        < 0
        or coverage.candidate_pair_count > limits.maximum_candidate_pairs
        or coverage.subdivision_piece_count > limits.maximum_subdivision_pieces
        or coverage.maximum_subdivision_depth_reached > limits.maximum_subdivision_depth
        or (
            coverage.source_expression_work_units is not None
            and coverage.source_expression_work_units > limits.maximum_work_units
        )
        or (
            coverage.source_expression_peak_bytes is not None
            and coverage.source_expression_peak_bytes > limits.maximum_scratch_bytes
        )
    ):
        raise ValueError(
            f"MeshCertificationPreparedEvidence.coverage.work must fit requested limits; got candidate_pairs={coverage.candidate_pair_count}, subdivision_pieces={coverage.subdivision_piece_count}, subdivision_depth={coverage.maximum_subdivision_depth_reached}"
        )


@checked
def require_prepared_embedding(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    /,
) -> None:
    rebuilt_validity = _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate(
        validity.status,
        validity.determinant_lower,
        validity.determinant_upper,
        validity.depth,
        block_names=validity.block_names,
        block_offsets=validity.block_offsets,
        unsupported_block_names=validity.unsupported_block_names,
        unresolved_reasons=validity.unresolved_reasons,
        geometry_id=validity.geometry_id,
        geometry_layout_id=validity.geometry_layout_id,
        topology_id=validity.topology_id,
        policy_id=validity.policy_id,
    )
    expected_offsets = tuple(
        np.cumsum(
            (0, *(block.vertices.shape[0] for block in mesh.blocks)),
            dtype=np.int64,
        ).tolist()
    )
    if (
        not tree_content_equal(validity, rebuilt_validity)
        or validity.block_names != tuple(block.name for block in mesh.blocks)
        or validity.block_offsets != expected_offsets
        or validity.unsupported_block_names
        or validity.unresolved_reasons
    ):
        raise ValueError(
            f"MeshCertificationPreparedEvidence.validity must match its numerical proof records; got certificate_id={validity.certificate_id!r}"
        )
    embedding.binding.require(mesh, geometry)
    if (
        embedding.binding.source_id is not None
        or embedding.binding.source_revision is not None
    ):
        raise ValueError(
            f"MeshCertificationPreparedEvidence.embedding must bind mesh geometry; got source_id={embedding.binding.source_id!r}"
        )
    checks = set(embedding.evaluated_checks)
    complete = {"cell_validity"}
    if mesh.periodic_topology is not None:
        complete.update(
            (
                "periodic_authored_group_identity",
                "periodic_full_source_hulls",
                "periodic_image_sufficiency",
                "periodic_mapped_trace_equivariance",
                "periodic_mapped_self_image_embedding",
            )
        )
    elif embedding.binding.coordinate_scope == "mapped":
        complete.add("facet_pairing")
        if "mapped_affine_source_maps" in checks:
            complete.update(
                (
                    "mapped_affine_source_maps",
                    "mapped_exact_shared_corner_incidence",
                    "mapped_cell_contact",
                )
            )
        else:
            complete.update(("mapped_local_injectivity", "mapped_cell_contact"))
            if "mapped_restriction_embedding" in checks:
                complete.update(
                    (
                        "mapped_restriction_source",
                        "mapped_restriction_reference_domain",
                        "mapped_restriction_embedding",
                    )
                )
            else:
                complete.add("mapped_trace_continuity")
    else:
        complete.update(("boundary_contact", "exterior_degree"))
        complete.add("facet_pairing")
    if (
        embedding.status != "certified"
        or embedding.findings
        or not complete <= checks
        or embedding.validity_certificate_id != validity.certificate_id
        or embedding.binding.limits_id != limits.limits_id
        or embedding.binding.junction_vertices
        or embedding.cell_count != sum(block.vertices.shape[0] for block in mesh.blocks)
        or min(
            embedding.candidate_pair_count,
            embedding.ray_test_count,
            embedding.subdivision_piece_count,
        )
        < 0
        or embedding.candidate_pair_count > limits.maximum_candidate_pairs
        or embedding.ray_test_count > limits.maximum_ray_tests
        or embedding.subdivision_piece_count > limits.maximum_subdivision_pieces
    ):
        raise ValueError(
            f"MeshCertificationPreparedEvidence.embedding must contain sufficient governed evidence; got certificate_id={embedding.certificate_id!r}"
        )
    rebuilt = global_embedding_owner.GlobalEmbeddingCertificate.from_stored_fields(
        embedding
    )
    if not tree_content_equal(embedding, rebuilt):
        raise ValueError(
            f"MeshCertificationPreparedEvidence.embedding must match its numerical proof records; got certificate_id={embedding.certificate_id!r}"
        )


def _junction_excuses(mesh: CellMesh, junction_vertices: ArrayLike, /) -> tuple[str, ...]:
    """Nonmanifold audit counts explained by declared junctions alone.

    The audit counts are recounted independently: every vertex joining more
    than two intervals must be a declared junction, otherwise nothing is excused.
    """

    declared = np.asarray(junction_vertices, dtype=np.int64).reshape(-1)
    rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64).reshape(-1) for block in mesh.blocks]
    )
    valence = np.bincount(rows, minlength=mesh.coordinates.shape[0])
    identifiers = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if np.any((valence > 2) & ~np.isin(identifiers, declared)):
        return ()
    return ("nonmanifold_facets", "nonmanifold_vertices")


@final
class _AcceptanceInputs(StrictModule, NonTrainableState):
    """Retain invocation inputs across spatial and fidelity certification phases."""

    schedule: _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule
    domain: (
        domain_coverage_owner.PiecewiseLinearDomain
        | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain
        | None
    )
    cell_regions: ArrayLike | None
    source: source_boundary_records_owner.SourceBoundaryQuery | None
    fidelity_tolerance: float | None
    fidelity_sample_order: int
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits | None
    junction_vertices: ArrayLike | None
    scoped_fidelity: tuple[
        tuple[
            source_boundary_records_owner.SourceBoundaryQuery,
            ArrayLike,
            float,
        ],
        ...,
    ]
    budget: (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
        | None
    )


@final
class _SpatialAcceptanceEvidence(StrictModule, NonTrainableState):
    embedding: global_embedding_owner.GlobalEmbeddingCertificate
    coverage: domain_coverage_owner.DomainCoverageCertificate | None
    request: (
        _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs | None
    )


def _certify_spatial_acceptance(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport,
    inputs: _AcceptanceInputs,
    volume: MeshCertificationPreparedEvidence | None,
    prepared_embedding: global_embedding_owner.GlobalEmbeddingCertificate | None,
) -> _SpatialAcceptanceEvidence:
    schedule = inputs.schedule
    domain = inputs.domain
    cell_regions = inputs.cell_regions
    source = inputs.source
    fidelity_tolerance = inputs.fidelity_tolerance
    fidelity_sample_order = inputs.fidelity_sample_order
    limits = inputs.limits
    junction_vertices = inputs.junction_vertices
    scoped_fidelity = inputs.scoped_fidelity
    budget = inputs.budget
    coverage_inputs = domain is not None or cell_regions is not None
    request = None
    if volume is not None:
        request = (
            _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs(
                mesh,
                geometry,
                schedule,
                domain=domain,
                cell_regions=cell_regions,
                source=source,
                fidelity_tolerance=fidelity_tolerance,
                fidelity_sample_order=fidelity_sample_order,
                limits=limits,
                junction_vertices=junction_vertices,
                scoped_fidelity=scoped_fidelity,
                budget=budget,
            )
        )
        volume.require(mesh, geometry, request, budget=budget)
        if not tree_content_equal(audit.validity, volume.validity):
            raise ValueError(
                f"certify_meshing_acceptance.audit.validity must use the prepared validity certificate; got audit_id={audit.validity.certificate_id!r}, prepared_id={volume.validity.certificate_id!r}"
            )
        embedding = volume.embedding
        coverage = volume.coverage
    else:
        if prepared_embedding is None:
            embedding = global_embedding_owner.certify_global_embedding(
                mesh,
                geometry,
                audit.validity,
                limits=limits,
                junction_vertices=junction_vertices,
                budget=budget,
            )
        else:
            prepared_embedding.binding.require(mesh, geometry)
            expected_limits = (
                _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits()
                if limits is None
                else limits
            )
            if (
                prepared_embedding.status != "certified"
                or prepared_embedding.validity_certificate_id
                != audit.validity.certificate_id
                or prepared_embedding.binding.limits_id != expected_limits.limits_id
            ):
                raise ValueError(
                    f"certify_meshing_acceptance.prepared.embedding must be current, certified and governed by the requested limits; got status={prepared_embedding.status!r}, limits_id={prepared_embedding.binding.limits_id!r}"
                )
            embedding = prepared_embedding
        coverage = None
        if domain is not None and cell_regions is not None:
            coverage = domain_coverage_owner.certify_domain_coverage(
                mesh,
                geometry,
                domain,
                cell_regions,
                embedding=embedding,
                limits=limits,
                budget=budget,
            )
        elif coverage_inputs:
            raise ValueError(
                "certify_meshing_acceptance.domain_coverage must supply both domain and cell_regions; got missing coverage inputs"
            )
    return _SpatialAcceptanceEvidence(embedding, coverage, request)


@final
class _FidelityAcceptanceEvidence(StrictModule, NonTrainableState):
    fidelity: source_fidelity_owner.SourceFidelityCertificate | None
    request: _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs
    scoped_certificates: tuple[source_fidelity_owner.SourceFidelityCertificate, ...]


def _certify_fidelity_acceptance(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    inputs: _AcceptanceInputs,
    request: _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs
    | None,
    prepared_fidelity: source_fidelity_owner.SourceFidelityCertificate | None,
) -> _FidelityAcceptanceEvidence:
    schedule = inputs.schedule
    domain = inputs.domain
    cell_regions = inputs.cell_regions
    source = inputs.source
    fidelity_tolerance = inputs.fidelity_tolerance
    fidelity_sample_order = inputs.fidelity_sample_order
    limits = inputs.limits
    junction_vertices = inputs.junction_vertices
    scoped_fidelity = inputs.scoped_fidelity
    budget = inputs.budget
    required = schedule.required_checks
    fidelity_inputs = source is not None or fidelity_tolerance is not None
    fidelity = None
    if source is not None and fidelity_tolerance is not None:
        if prepared_fidelity is None:
            fidelity = source_fidelity_owner.certify_source_fidelity(
                mesh,
                geometry,
                source,
                tolerance=fidelity_tolerance,
                sample_order=fidelity_sample_order,
                limits=limits,
                budget=budget,
            )
        else:
            prepared_fidelity.binding.require(mesh, geometry)
            if (
                prepared_fidelity.status != "certified"
                or prepared_fidelity.tolerance != fidelity_tolerance
                or prepared_fidelity.binding.source_id != source.source_id
                or prepared_fidelity.binding.source_revision != source.source_revision
                or prepared_fidelity.mesh_to_source_semantics != "certified"
                or prepared_fidelity.source_to_mesh_semantics != "certified"
            ):
                raise ValueError(
                    f"certify_meshing_acceptance.prepared.fidelity must be current, certified and governed by the requested source and tolerance; got status={prepared_fidelity.status!r}, tolerance={prepared_fidelity.tolerance!r}"
                )
            fidelity = prepared_fidelity
    elif fidelity_inputs:
        raise ValueError(
            "certify_meshing_acceptance.source_fidelity must supply both source and fidelity_tolerance; got missing fidelity inputs"
        )
    if scoped_fidelity and "source_fidelity" not in required:
        raise ValueError(
            f"certify_meshing_acceptance.scoped_fidelity must also schedule whole-source fidelity acceptance; got route={schedule.route!r}"
        )
    if request is None:
        request = (
            _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs(
                mesh,
                geometry,
                schedule,
                domain=domain,
                cell_regions=cell_regions,
                source=source,
                fidelity_tolerance=fidelity_tolerance,
                fidelity_sample_order=fidelity_sample_order,
                limits=limits,
                junction_vertices=junction_vertices,
                scoped_fidelity=scoped_fidelity,
                budget=budget,
            )
        )
    scoped_certificates = tuple(
        source_fidelity_owner.certify_source_fidelity(
            mesh,
            geometry,
            query,
            tolerance=tolerance,
            sample_order=fidelity_sample_order,
            limits=limits,
            target_facet_ids=identifiers,
            budget=budget,
        )
        for query, identifiers, tolerance in request.scoped_fidelity
    )
    return _FidelityAcceptanceEvidence(fidelity, request, scoped_certificates)


@final
class PreparedVolumeAcceptance(StrictModule, NonTrainableState):
    """Retain prepared volume evidence and optional source fidelity."""

    volume: MeshCertificationPreparedEvidence
    fidelity: source_fidelity_owner.SourceFidelityCertificate | None = None

    @checked
    def __init__(
        self,
        volume: MeshCertificationPreparedEvidence,
        fidelity: source_fidelity_owner.SourceFidelityCertificate | None = None,
    ) -> None:
        self.volume = volume
        self.fidelity = fidelity


@final
class PreparedBoundaryAcceptance(StrictModule, NonTrainableState):
    """Retain independently prepared embedding and source fidelity."""

    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None = None
    fidelity: source_fidelity_owner.SourceFidelityCertificate | None = None

    @checked
    def __init__(
        self,
        embedding: global_embedding_owner.GlobalEmbeddingCertificate | None = None,
        fidelity: source_fidelity_owner.SourceFidelityCertificate | None = None,
    ) -> None:
        self.embedding = embedding
        self.fidelity = fidelity


type PreparedMeshingAcceptance = PreparedVolumeAcceptance | PreparedBoundaryAcceptance


@checked
def certify_meshing_acceptance(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport,
    /,
    *,
    schedule: _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule,
    domain: domain_coverage_owner.PiecewiseLinearDomain
    | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain
    | None = None,
    cell_regions: ArrayLike | None = None,
    source: source_boundary_records_owner.SourceBoundaryQuery | None = None,
    fidelity_tolerance: float | None = None,
    fidelity_sample_order: int = 4,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    junction_vertices: ArrayLike | None = None,
    prepared: PreparedMeshingAcceptance | None = None,
    scoped_fidelity: tuple[
        tuple[
            source_boundary_records_owner.SourceBoundaryQuery,
            ArrayLike,
            float,
        ],
        ...,
    ] = (),
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None = None,
) -> MeshCertificationReport:
    """Run the route's acceptance schedule and return its bound report.

    ``audit`` must be the audit of ``mesh`` with ``geometry`` and must have
    evaluated every closure/topology check the route may not skip. PLC volume
    routes need ``domain`` and ``cell_regions``; implicit volume and surface
    routes need ``source`` and ``fidelity_tolerance``. Inputs a route does not
    use are refused rather than ignored. ``junction_vertices`` (curve route
    only) declares the vertex global ids of curve-network junctions, the only
    vertices where more than two intervals may meet.
    ``prepared`` selects ``PreparedVolumeAcceptance`` for volume evidence or
    ``PreparedBoundaryAcceptance`` for independently prepared embedding. Both
    variants may retain source fidelity; volume evidence cannot select a second
    embedding. Acceptance validates each retained binding before reusing it.
    """

    with _certificate_budget(limits, budget) as budget:
        match prepared:
            case PreparedVolumeAcceptance(volume=volume, fidelity=prepared_fidelity):
                prepared_embedding = None
            case PreparedBoundaryAcceptance(
                embedding=prepared_embedding, fidelity=prepared_fidelity
            ):
                volume = None
            case None:
                volume = None
                prepared_embedding = None
                prepared_fidelity = None
            case _:
                raise TypeError(
                    "certify_meshing_acceptance.prepared must contain prepared acceptance evidence; got unsupported variant"
                )
        if (
            audit.mesh_id != mesh.mesh_id
            or audit.geometry_id
            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                geometry
            )
            or audit.geometry_layout_id != geometry.geometry_layout_id
        ):
            raise ValueError(
                f"certify_meshing_acceptance.audit must bind the mesh and geometry; got audit_mesh_id={audit.mesh_id!r}, audit_geometry_id={audit.geometry_id!r}"
            )
        _require_audit(schedule, audit)
        required = schedule.required_checks
        coverage_inputs = domain is not None or cell_regions is not None
        fidelity_inputs = source is not None or fidelity_tolerance is not None
        if coverage_inputs != ("domain_coverage" in required) or fidelity_inputs != (
            "source_fidelity" in required
        ):
            raise ValueError(
                f"certify_meshing_acceptance.inputs must supply exactly the inputs of its checks; got route={schedule.route!r}, coverage_inputs={coverage_inputs!r}, fidelity_inputs={fidelity_inputs!r}"
            )
        excused: tuple[str, ...] = ()
        if junction_vertices is not None:
            if schedule.route != "curve":
                raise ValueError(
                    f"certify_meshing_acceptance.junction_vertices must be declared only on the curve route; got route={schedule.route!r}"
                )
            excused = _junction_excuses(mesh, junction_vertices)
        request = None
        inputs = _AcceptanceInputs(
            schedule,
            domain,
            cell_regions,
            source,
            fidelity_tolerance,
            fidelity_sample_order,
            limits,
            junction_vertices,
            scoped_fidelity,
            budget,
        )
        spatial = _certify_spatial_acceptance(
            mesh, geometry, audit, inputs, volume, prepared_embedding
        )
        embedding, coverage, request = (
            spatial.embedding,
            spatial.coverage,
            spatial.request,
        )
        fidelity_phase = _certify_fidelity_acceptance(
            mesh, geometry, inputs, request, prepared_fidelity
        )
        fidelity = fidelity_phase.fidelity
        request = fidelity_phase.request
        scoped_certificates = fidelity_phase.scoped_certificates
        outcomes = []
        for check in required:
            match check:
                case "cell_validity":
                    outcomes.append(_validity_outcome(audit))
                case "topology":
                    outcomes.append(_topology_outcome(audit, excused))
                case "quality":
                    outcomes.append(_quality_outcome(audit))
                case "global_embedding":
                    outcomes.append(_certificate_outcome(check, embedding))
                case "domain_coverage" if coverage is not None:
                    outcomes.append(_certificate_outcome(check, coverage))
                case "source_fidelity" if fidelity is not None:
                    outcomes.append(_certificate_outcome(check, fidelity))
                case _:
                    raise RuntimeError(
                        f"certify_meshing_acceptance.evidence must cover the scheduled check; got check={check!r} without evidence"
                    )
        outcomes.extend(
            _certificate_outcome("source_fidelity", certificate)
            for certificate in scoped_certificates
        )
        return MeshCertificationReport(
            schedule,
            mesh,
            geometry,
            audit,
            tuple(outcomes),
            embedding=embedding,
            coverage=coverage,
            fidelity=fidelity,
            request=request,
            scoped_fidelity=scoped_certificates,
        )


@checked
def acceptance_stage_report(
    report: MeshCertificationReport, /
) -> _dependency_phydrax_meshing__trace.MeshingStageReport:
    """Trace stage of one certification report with failing-entity diagnostics."""

    achieved = dict(report.achieved)
    diagnostics = tuple(
        _dependency_phydrax_meshing__trace.MeshingDiagnostic(
            _dependency_phydrax_meshing__trace.MeshingDiagnosticSeverity.ERROR,
            f"{outcome.check} {outcome.status}: {', '.join(outcome.reasons)}",
            failure_category=_leaf_phydrax_meshing__failures.MeshingFailureCategory.AUDIT_FAILED,
            entity_ids=outcome.entity_ids,
            # Diagnostics saturate an infinite bound at the largest finite
            # float; the retained report preserves the infinite bound.
            quantities=tuple(
                (name, wanted, min(achieved[name], _LARGEST))
                for name, wanted in report.requested
                if name in achieved
            ),
        )
        for outcome in report.failing_outcomes
    )
    statuses = {outcome.status for outcome in report.failing_outcomes}
    status = (
        _dependency_phydrax_meshing__trace.MeshingStageStatus.FAILED
        if "violated" in statuses
        else _dependency_phydrax_meshing__trace.MeshingStageStatus.UNRESOLVED
        if statuses
        else _dependency_phydrax_meshing__trace.MeshingStageStatus.PASSED
    )
    certificates = tuple(
        value.certificate_id
        for value in (report.embedding, report.coverage, report.fidelity)
        if value is not None
    )
    return _dependency_phydrax_meshing__trace.MeshingStageReport(
        _dependency_phydrax_meshing__trace.MeshingStageKind.CERTIFICATION,
        status,
        input_ids=(report.mesh_id, report.geometry_id, report.audit_report_id),
        output_ids=(report.report_id, *certificates),
        diagnostics=diagnostics,
    )


__all__ = [
    "MeshCertificationOutcome",
    "MeshCertificationPreparedEvidence",
    "MeshCertificationReport",
    "PreparedBoundaryAcceptance",
    "PreparedMeshingAcceptance",
    "PreparedVolumeAcceptance",
    "acceptance_stage_report",
    "certify_meshing_acceptance",
]
