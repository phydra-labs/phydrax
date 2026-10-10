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

from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityStatus,
)
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..geometry._mesh_certificates import (
    certify_domain_coverage,
    certify_global_embedding,
    certify_source_fidelity,
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
    MeshCertificateEntityKind,
    MeshCertificateLimits,
    MeshCertificateStatus,
    PiecewiseLinearDomain,
    SourceBoundaryQuery,
    SourceFidelityCertificate,
)
from ..typing import parse
from ._audit import CellMeshAuditReport
from ._certification_inputs import MeshCertificationInputs
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._trace import (
    MeshingDiagnostic,
    MeshingDiagnosticSeverity,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
)


MeshCertificationCheck: TypeAlias = Literal[
    "cell_validity",
    "topology",
    "quality",
    "global_embedding",
    "domain_coverage",
    "source_fidelity",
]
MeshCertificationRoute: TypeAlias = Literal[
    "volume_plc",
    "volume_implicit",
    "volume_layer",
    "mapped_volume",
    "surface",
    "curve",
    "periodic",
]

_QUALITY_ISSUES = ("minimum_measure", "minimum_mean_ratio", "maximum_aspect_ratio")
_VALIDITY_NAMES = ("invalid_geometry", "unresolved_geometry_validity")
_LARGEST = float(np.finfo(np.float64).max)


def _route_checks(
    route: MeshCertificationRoute, /
) -> tuple[tuple[MeshCertificationCheck, ...], tuple[str, ...]]:
    """Required certificates and audit checks that may not be skipped."""

    common: tuple[MeshCertificationCheck, ...] = (
        "cell_validity",
        "topology",
        "quality",
        "global_embedding",
    )
    welded = ("coincident_vertices", "collapsed_cells", "duplicate_cells")
    manifold = (
        "nonmanifold_facets",
        "nonmanifold_edges",
        "nonmanifold_vertices",
        "inconsistent_orientation",
        "self_intersection",
    )
    match route:
        case "volume_plc":
            return (*common, "domain_coverage"), (*welded, *manifold, "open_boundary")
        case "volume_implicit" | "volume_layer" | "mapped_volume":
            return (
                (*common, "domain_coverage", "source_fidelity"),
                (*welded, *manifold, "open_boundary"),
            )
        case "surface":
            # Surfaces may be intentionally open; closure is not applicable.
            return (*common, "source_fidelity"), (*welded, *manifold)
        case "curve":
            return common, (*welded, "inconsistent_orientation")
        case "periodic":
            return common, (*welded, *manifold)
        case _:
            assert_never(route)


@final
class MeshCertificationSchedule(StrictModule, NonTrainableState):
    """Checks a route must certify and audit checks it may not skip."""

    route: MeshCertificationRoute = eqx.field(static=True)
    required_checks: tuple[MeshCertificationCheck, ...] = eqx.field(static=True)
    required_audit_checks: tuple[str, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, route: MeshCertificationRoute, /) -> None:
        route_ = parse(route, MeshCertificationRoute, "route")
        required, audit_checks = _route_checks(route_)
        self.route = route_
        self.required_checks = required
        self.required_audit_checks = audit_checks
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "mesh-certification-schedule",
                "route": route_,
                "required": required,
                "audit_checks": audit_checks,
            }
        )


@final
class MeshCertificationOutcome(StrictModule, NonTrainableState):
    """Status of one scheduled check with its evidence and failing entities."""

    check: MeshCertificationCheck = eqx.field(static=True)
    status: MeshCertificateStatus = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    entity_kind: MeshCertificateEntityKind = eqx.field(static=True)
    entity_ids: tuple[int, ...] = eqx.field(static=True)
    outcome_id: str = eqx.field(static=True)

    def __init__(
        self,
        check: MeshCertificationCheck,
        status: MeshCertificateStatus,
        evidence_id: str,
        /,
        *,
        reasons: tuple[str, ...] = (),
        entity_kind: MeshCertificateEntityKind = "mesh",
        entity_ids: tuple[int, ...] = (),
    ) -> None:
        self.check = parse(check, MeshCertificationCheck, "check")
        self.status = parse(status, MeshCertificateStatus, "status")
        self.evidence_id = str(evidence_id)
        self.reasons = tuple(dict.fromkeys(str(value) for value in reasons))
        self.entity_kind = parse(entity_kind, MeshCertificateEntityKind, "entity_kind")
        self.entity_ids = tuple(sorted({int(value) for value in entity_ids}))
        if (self.status == "certified") != (not self.reasons):
            raise ValueError("Only non-certified outcomes carry reasons.")
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

    ``scoped_fidelity`` retains each full original-stratum/target-trace proof
    independently of whole-boundary fidelity and domain coverage. The retained
    request binds source query scope, actual global facet IDs, and physical
    tolerance; every scoped verdict participates in acceptance and report
    identity. Renewal uses exact facet lineage rather than reusing old bounds.
    """

    schedule: MeshCertificationSchedule
    request: MeshCertificationInputs
    mesh_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    audit_report_id: str = eqx.field(static=True)
    embedding: GlobalEmbeddingCertificate | None
    coverage: DomainCoverageCertificate | None
    fidelity: SourceFidelityCertificate | None
    scoped_fidelity: tuple[SourceFidelityCertificate, ...]
    outcomes: tuple[MeshCertificationOutcome, ...]
    requested: tuple[tuple[str, float], ...] = eqx.field(static=True)
    achieved: tuple[tuple[str, float], ...] = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule: MeshCertificationSchedule,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        audit: CellMeshAuditReport,
        outcomes: tuple[MeshCertificationOutcome, ...],
        /,
        *,
        embedding: GlobalEmbeddingCertificate | None,
        coverage: DomainCoverageCertificate | None,
        fidelity: SourceFidelityCertificate | None,
        request: MeshCertificationInputs,
        scoped_fidelity: tuple[SourceFidelityCertificate, ...] = (),
    ) -> None:
        if not isinstance(request, MeshCertificationInputs):
            raise TypeError("request must be MeshCertificationInputs.")
        if (
            request.mesh_id != mesh.mesh_id
            or request.geometry_id != cell_geometry_id(geometry)
            or request.schedule.schedule_id != schedule.schedule_id
        ):
            raise ValueError(
                "Certification request must bind actual mesh, geometry and schedule."
            )
        if {outcome.check for outcome in outcomes} != set(schedule.required_checks):
            raise ValueError("Certification outcomes must cover the required checks.")
        if len(scoped_fidelity) != len(request.scoped_fidelity):
            raise ValueError(
                "Scoped fidelity evidence must cover every exact retained source/target request."
            )
        for certificate, (query, identifiers, tolerance) in zip(
            scoped_fidelity, request.scoped_fidelity, strict=True
        ):
            if not isinstance(certificate, SourceFidelityCertificate):
                raise TypeError(
                    "Scoped fidelity must retain owning numerical proof records."
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
                    "Scoped fidelity evidence binds another source stratum, target trace, or physical error request."
                )
            expected = _certificate_outcome("source_fidelity", certificate)
            if not any(value.outcome_id == expected.outcome_id for value in outcomes):
                raise ValueError(
                    "Scoped numerical proof status must reach an actual certification outcome."
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
        self.geometry_id = cell_geometry_id(geometry)
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
                # Unresolved fidelity bounds are inf; exact float reprs carry them.
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
        raise _MeshCertificationReportFailure(self)


def _require_audit(
    schedule: MeshCertificationSchedule, audit: CellMeshAuditReport, /
) -> None:
    missing = tuple(
        name
        for name in schedule.required_audit_checks
        if name not in audit.evaluated_checks
    )
    if missing:
        raise ValueError(
            f"Route {schedule.route!r} may not skip audit checks: " + ", ".join(missing)
        )


def _validity_outcome(audit: CellMeshAuditReport, /) -> MeshCertificationOutcome:
    validity = audit.validity
    status = np.asarray(validity.status)
    failing = dict(audit.failing_cells)
    if np.any(status == CellValidityStatus.INVALID):
        return MeshCertificationOutcome(
            "cell_validity",
            "violated",
            validity.certificate_id,
            reasons=("invalid_geometry",),
            entity_kind="cell",
            entity_ids=failing["invalid_geometry"],
        )
    if np.any(status == CellValidityStatus.UNRESOLVED):
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
    audit: CellMeshAuditReport, excused: tuple[str, ...], /
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


def _quality_outcome(audit: CellMeshAuditReport, /) -> MeshCertificationOutcome:
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
    check: MeshCertificationCheck,
    certificate: GlobalEmbeddingCertificate
    | DomainCoverageCertificate
    | SourceFidelityCertificate,
    /,
) -> MeshCertificationOutcome:
    if certificate.status == "certified":
        return MeshCertificationOutcome(check, "certified", certificate.certificate_id)
    relevant = tuple(
        value for value in certificate.findings if value.status == certificate.status
    )
    kinds = {value.entity_kind for value in relevant}
    kind: MeshCertificateEntityKind = kinds.pop() if len(kinds) == 1 else "mesh"
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
        GlobalEmbeddingCertificate
        | DomainCoverageCertificate
        | SourceFidelityCertificate
        | None,
        ...,
    ],
    /,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Preserve the owning certificate/finding/resource names without reinterpretation."""
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
    return tuple(sorted(requested.items())), tuple(sorted(achieved.items()))


class _MeshCertificationReportFailure(MeshingFailure):
    """Public failure quantities backed by the complete actual failing report."""

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
        super().__init__(
            MeshingFailureCategory.AUDIT_FAILED,
            "; ".join(
                f"{value.check} {value.status}: {', '.join(value.reasons)}"
                for value in failing
            ),
            stage=MeshingStageKind.CERTIFICATION.value,
            entity_ids=tuple(
                identifier
                for value in failing
                if value.entity_kind in ("cell", "facet")
                for identifier in value.entity_ids
            ),
            requested=tuple(sorted(requested.items())),
            achieved=tuple(sorted(achieved.items())),
        )


class _MeshCertificationPremiseFailure(MeshingFailure):
    """Negative computed premises, never a partial positive prepared artifact."""

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        request: MeshCertificationInputs,
        validity: CellValidityCertificate,
        embedding: GlobalEmbeddingCertificate,
        coverage: DomainCoverageCertificate,
        /,
    ) -> None:
        from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

        self.mesh = mesh
        self.geometry = geometry
        self.request = request
        self.validity = validity
        self.embedding = embedding
        self.coverage = coverage
        self.ledger = _COORDINATE_BUDGET.get()
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
        # The exact optional measures, bounds, SCI and reference source data
        # remain in the owning coverage/geometry above, not reconstructed here.
        outcome = _certificate_outcome("domain_coverage", coverage)
        super().__init__(
            MeshingFailureCategory.AUDIT_FAILED,
            f"domain_coverage {coverage.status}: {', '.join(outcome.reasons)}",
            stage=MeshingStageKind.CERTIFICATION.value,
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


@final
class MeshCertificationPreparedEvidence(StrictModule, NonTrainableState):
    """Positive volume premises computed once under an exact owning request.

    Construction performs actual labelled coverage, not adoption of an external
    coverage flag: that certificate alone does not bind the cell-region vector.
    Publication reuses these same certificates and their actual work counters.
    Fidelity, topology and quality remain independently evaluated by acceptance.
    """

    request: MeshCertificationInputs
    validity: CellValidityCertificate
    embedding: GlobalEmbeddingCertificate
    coverage: DomainCoverageCertificate
    mesh_arrays_id: str = eqx.field(static=True)
    validity_arrays_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        validity: CellValidityCertificate,
        /,
        *,
        schedule: MeshCertificationSchedule,
        domain: PiecewiseLinearDomain | MappedReferenceDomain,
        cell_regions: ArrayLike,
        embedding: GlobalEmbeddingCertificate | None = None,
        limits: MeshCertificateLimits | None = None,
        source: SourceBoundaryQuery | None = None,
        fidelity_tolerance: float | None = None,
        fidelity_sample_order: int = 4,
        scoped_fidelity: tuple[tuple[SourceBoundaryQuery, ArrayLike, float], ...] = (),
    ) -> None:
        if not isinstance(schedule, MeshCertificationSchedule):
            raise TypeError("schedule must be MeshCertificationSchedule.")
        if "domain_coverage" not in schedule.required_checks:
            raise ValueError(
                "Prepared volume premises require scheduled domain coverage."
            )
        if domain is None or cell_regions is None:
            raise ValueError("Prepared coverage requires both domain and cell_regions.")
        if (source is None) != (fidelity_tolerance is None) or (source is not None) != (
            "source_fidelity" in schedule.required_checks
        ):
            raise ValueError(
                "Prepared evidence requires exactly the schedule's source inputs."
            )
        request = MeshCertificationInputs(
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
        )
        request.validate_source_integrity()
        if not isinstance(validity, CellValidityCertificate):
            raise TypeError("validity must be CellValidityCertificate.")
        validity.require_bound(geometry, mesh=mesh)
        if (
            not validity.all_certified
            or not np.all(
                np.asarray(validity.status) == CellValidityStatus.CERTIFIED_VALID
            )
            or validity.status.shape[0]
            != sum(block.vertices.shape[0] for block in mesh.blocks)
        ):
            raise ValueError("Prepared validity must be wholly positive and complete.")
        embedding_ = (
            certify_global_embedding(
                mesh,
                geometry,
                validity,
                limits=request.limits,
            )
            if embedding is None
            else embedding
        )
        _require_prepared_embedding(mesh, geometry, validity, embedding_, request.limits)
        assert request.domain is not None and request.cell_regions is not None
        coverage = certify_domain_coverage(
            mesh,
            geometry,
            request.domain,
            request.cell_regions,
            embedding=embedding_,
            limits=request.limits,
        )
        if coverage.status != "certified" or coverage.findings:
            raise _MeshCertificationPremiseFailure(
                mesh, geometry, request, validity, embedding_, coverage
            )
        _require_prepared_coverage(coverage, request.limits)
        self.request = request
        self.validity = validity
        self.embedding = embedding_
        self.coverage = coverage
        self.mesh_arrays_id = canonical_fingerprint(array_tree_fingerprint(mesh))
        self.validity_arrays_id = canonical_fingerprint(array_tree_fingerprint(validity))
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

    def require(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        request: MeshCertificationInputs,
        /,
    ) -> None:
        """Refuse stale, foreign, incomplete or differently governed premises."""
        self.request.validate_source_integrity()
        if request.request_id != self.request.request_id or not bool(
            eqx.tree_equal(request, self.request, typematch=True)
        ):
            raise ValueError(
                "Prepared evidence differs from the actual acceptance request."
            )
        self.validity.require_bound(geometry, mesh=mesh)
        if (
            not self.validity.all_certified
            or not np.all(
                np.asarray(self.validity.status) == CellValidityStatus.CERTIFIED_VALID
            )
            or canonical_fingerprint(array_tree_fingerprint(mesh)) != self.mesh_arrays_id
            or canonical_fingerprint(array_tree_fingerprint(self.validity))
            != self.validity_arrays_id
        ):
            raise ValueError("Prepared mesh or validity numerical evidence changed.")
        _require_prepared_embedding(
            mesh,
            geometry,
            self.validity,
            self.embedding,
            request.limits,
        )
        coverage = self.coverage
        if not isinstance(coverage, DomainCoverageCertificate):
            raise TypeError("Prepared coverage must be DomainCoverageCertificate.")
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
            raise ValueError("Prepared coverage has incomplete or foreign premises.")
        rebuilt = DomainCoverageCertificate(
            coverage.binding,
            coverage.embedding_certificate_id,
            domain,
            coverage.findings,
            requested_region_measures=coverage.requested_region_measures,
            achieved_region_measures=coverage.achieved_region_measures,
            requested_region_measure_bounds=coverage.requested_region_measure_bounds,
            achieved_region_measure_bounds=coverage.achieved_region_measure_bounds,
            integration_error_bounds=coverage.integration_error_bounds,
            covered_source_facet_count=coverage.covered_source_facet_count,
            facet_source_overlaps=coverage.facet_source_overlaps,
            premise_certificate_ids=coverage.premise_certificate_ids,
            candidate_pair_count=coverage.candidate_pair_count,
            subdivision_piece_count=coverage.subdivision_piece_count,
            maximum_subdivision_depth_reached=coverage.maximum_subdivision_depth_reached,
            source_expression_work_units=coverage.source_expression_work_units,
            source_expression_required_work_units=(
                coverage.source_expression_required_work_units
            ),
            source_expression_peak_bytes=coverage.source_expression_peak_bytes,
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
        if (
            not bool(eqx.tree_equal(coverage, rebuilt, typematch=True))
            or identity != self.evidence_id
        ):
            raise ValueError(
                "Prepared evidence differs from its numerical proof records."
            )


def _require_prepared_coverage(
    coverage: DomainCoverageCertificate,
    limits: MeshCertificateLimits,
    /,
) -> None:
    if coverage.status != "certified" or coverage.findings:
        raise ValueError("Prepared domain coverage must be wholly positive.")
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
                "Prepared coverage requires actual bounded region-measure evidence."
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
        raise ValueError("Prepared coverage exceeds its actual requested work limits.")


def _require_prepared_embedding(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    embedding: GlobalEmbeddingCertificate,
    limits: MeshCertificateLimits,
    /,
) -> None:
    rebuilt_validity = CellValidityCertificate(
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
        not bool(eqx.tree_equal(validity, rebuilt_validity, typematch=True))
        or validity.block_names != tuple(block.name for block in mesh.blocks)
        or validity.block_offsets != expected_offsets
        or validity.unsupported_block_names
        or validity.unresolved_reasons
    ):
        raise ValueError(
            "Prepared validity differs from its complete numerical proof records."
        )
    if not isinstance(embedding, GlobalEmbeddingCertificate):
        raise TypeError("embedding must be GlobalEmbeddingCertificate.")
    embedding.binding.require(mesh, geometry)
    if (
        embedding.binding.source_id is not None
        or embedding.binding.source_revision is not None
    ):
        raise ValueError(
            "Prepared global embedding must bind mesh geometry, not a foreign source."
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
            "Prepared embedding has incomplete, foreign or insufficient premises."
        )
    rebuilt = GlobalEmbeddingCertificate(
        embedding.binding,
        embedding.validity_certificate_id,
        embedding.findings,
        embedding.evaluated_checks,
        cell_count=embedding.cell_count,
        boundary_facet_count=embedding.boundary_facet_count,
        shell_count=embedding.shell_count,
        candidate_pair_count=embedding.candidate_pair_count,
        ray_test_count=embedding.ray_test_count,
        subdivision_piece_count=embedding.subdivision_piece_count,
        periodic_image_count=embedding.periodic_image_count,
        source_expression_work_units=embedding.source_expression_work_units,
        source_expression_peak_bytes=embedding.source_expression_peak_bytes,
        boundary_degree=embedding.boundary_degree,
    )
    if not bool(eqx.tree_equal(embedding, rebuilt, typematch=True)):
        raise ValueError("Prepared embedding differs from its numerical proof records.")


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


def certify_meshing_acceptance(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    audit: CellMeshAuditReport,
    /,
    *,
    schedule: MeshCertificationSchedule,
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None = None,
    cell_regions: ArrayLike | None = None,
    source: SourceBoundaryQuery | None = None,
    fidelity_tolerance: float | None = None,
    fidelity_sample_order: int = 4,
    limits: MeshCertificateLimits | None = None,
    junction_vertices: ArrayLike | None = None,
    prepared: MeshCertificationPreparedEvidence | None = None,
    prepared_embedding: GlobalEmbeddingCertificate | None = None,
    prepared_fidelity: SourceFidelityCertificate | None = None,
    scoped_fidelity: tuple[tuple[SourceBoundaryQuery, ArrayLike, float], ...] = (),
) -> MeshCertificationReport:
    """Run the route's acceptance schedule and return its bound report.

    ``audit`` must be the audit of ``mesh`` with ``geometry`` and must have
    evaluated every closure/topology check the route may not skip. PLC volume
    routes need ``domain`` and ``cell_regions``; implicit volume and surface
    routes need ``source`` and ``fidelity_tolerance``. Inputs a route does not
    use are refused rather than ignored. ``junction_vertices`` (curve route
    only) declares the vertex global ids of curve-network junctions, the only
    vertices where more than two intervals may meet.
    ``prepared`` is positive volume evidence from the owning preparation API;
    mismatched bindings are refused, never silently recertified.
    """

    if not isinstance(schedule, MeshCertificationSchedule):
        raise TypeError("schedule must be MeshCertificationSchedule.")
    if not isinstance(audit, CellMeshAuditReport):
        raise TypeError("audit must be CellMeshAuditReport.")
    if (
        audit.mesh_id != mesh.mesh_id
        or audit.geometry_id != cell_geometry_id(geometry)
        or audit.geometry_layout_id != geometry.geometry_layout_id
    ):
        raise ValueError("The audit must be bound to this mesh and geometry.")
    _require_audit(schedule, audit)
    required = schedule.required_checks
    coverage_inputs = domain is not None or cell_regions is not None
    fidelity_inputs = source is not None or fidelity_tolerance is not None
    if coverage_inputs != ("domain_coverage" in required) or fidelity_inputs != (
        "source_fidelity" in required
    ):
        raise ValueError(
            f"Route {schedule.route!r} requires exactly the inputs of its checks."
        )
    excused: tuple[str, ...] = ()
    if junction_vertices is not None:
        if schedule.route != "curve":
            raise ValueError("junction_vertices are declared on the curve route only.")
        excused = _junction_excuses(mesh, junction_vertices)
    request = None
    if prepared is not None and prepared_embedding is not None:
        raise ValueError(
            "Whole prepared evidence and a separate embedding are mutually exclusive."
        )
    if prepared is not None:
        if not isinstance(prepared, MeshCertificationPreparedEvidence):
            raise TypeError("prepared must be MeshCertificationPreparedEvidence or None.")
        request = MeshCertificationInputs(
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
        )
        prepared.require(mesh, geometry, request)
        if not bool(eqx.tree_equal(audit.validity, prepared.validity, typematch=True)):
            raise ValueError("The audit must use the exact prepared validity premise.")
        embedding = prepared.embedding
        coverage = prepared.coverage
    else:
        if prepared_embedding is None:
            embedding = certify_global_embedding(
                mesh,
                geometry,
                audit.validity,
                limits=limits,
                junction_vertices=junction_vertices,
            )
        else:
            prepared_embedding.binding.require(mesh, geometry)
            expected_limits = MeshCertificateLimits() if limits is None else limits
            if (
                prepared_embedding.status != "certified"
                or prepared_embedding.validity_certificate_id
                != audit.validity.certificate_id
                or prepared_embedding.binding.limits_id != expected_limits.limits_id
            ):
                raise ValueError(
                    "Prepared embedding is stale, unresolved, or differently governed."
                )
            embedding = prepared_embedding
        coverage = None
        if domain is not None and cell_regions is not None:
            coverage = certify_domain_coverage(
                mesh,
                geometry,
                domain,
                cell_regions,
                embedding=embedding,
                limits=limits,
            )
        elif coverage_inputs:
            raise ValueError("Domain coverage needs both domain and cell_regions.")
    fidelity = None
    if source is not None and fidelity_tolerance is not None:
        if prepared_fidelity is None:
            fidelity = certify_source_fidelity(
                mesh,
                geometry,
                source,
                tolerance=fidelity_tolerance,
                sample_order=fidelity_sample_order,
                limits=limits,
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
                    "Prepared fidelity is stale, unresolved, or differently governed."
                )
            fidelity = prepared_fidelity
    elif fidelity_inputs:
        raise ValueError("Source fidelity needs both source and fidelity_tolerance.")
    if scoped_fidelity and "source_fidelity" not in required:
        raise ValueError(
            "Scoped original-source checks require independent whole-source fidelity acceptance."
        )
    if request is None:
        request = MeshCertificationInputs(
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
        )
    scoped_certificates = tuple(
        certify_source_fidelity(
            mesh,
            geometry,
            query,
            tolerance=tolerance,
            sample_order=fidelity_sample_order,
            limits=limits,
            target_facet_ids=identifiers,
        )
        for query, identifiers, tolerance in request.scoped_fidelity
    )
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
                raise RuntimeError(f"Scheduled check {check!r} has no evidence.")
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


def acceptance_stage_report(report: MeshCertificationReport, /) -> MeshingStageReport:
    """Trace stage of one certification report with failing-entity diagnostics."""

    if not isinstance(report, MeshCertificationReport):
        raise TypeError("report must be MeshCertificationReport.")
    achieved = dict(report.achieved)
    diagnostics = tuple(
        MeshingDiagnostic(
            MeshingDiagnosticSeverity.ERROR,
            f"{outcome.check} {outcome.status}: {', '.join(outcome.reasons)}",
            failure_category=MeshingFailureCategory.AUDIT_FAILED,
            entity_ids=outcome.entity_ids,
            # An unresolved bound is unbounded (inf); diagnostics carry it as
            # the largest finite float, the report keeps the exact value.
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
        MeshingStageStatus.FAILED
        if "violated" in statuses
        else MeshingStageStatus.UNRESOLVED
        if statuses
        else MeshingStageStatus.PASSED
    )
    certificates = tuple(
        value.certificate_id
        for value in (report.embedding, report.coverage, report.fidelity)
        if value is not None
    )
    return MeshingStageReport(
        MeshingStageKind.CERTIFICATION,
        status,
        input_ids=(report.mesh_id, report.geometry_id, report.audit_report_id),
        output_ids=(report.report_id, *certificates),
        diagnostics=diagnostics,
    )


__all__ = [
    "MeshCertificationCheck",
    "MeshCertificationInputs",
    "MeshCertificationOutcome",
    "MeshCertificationPreparedEvidence",
    "MeshCertificationReport",
    "MeshCertificationRoute",
    "MeshCertificationSchedule",
    "acceptance_stage_report",
    "certify_meshing_acceptance",
]
