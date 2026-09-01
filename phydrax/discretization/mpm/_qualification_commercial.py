#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._commercial import (
    MPMClaimTuple,
    MPMIntendedUse,
    MPMReleaseEvidenceBundle,
    MPMReleaseGate,
    MPMReleaseGateEvidence,
    MPMSupportMatrix,
)


class MPMCommercialProfileKind(IntEnum):
    CODE_SOLUTION_VERIFIED = 0
    COMMERCIAL_RUNTIME = 1
    ADVANCED_MECHANICS = 2
    LARGE_SCALE = 3
    DIFFERENTIABLE_ENGINEERING = 4
    APPLICATION_VALIDATED = 5
    REGULATED_OVERLAY = 6


class MPMIndependentReview(StrictModule, NonTrainableState):
    author_id: str = eqx.field(static=True)
    technical_reviewer_id: str = eqx.field(static=True)
    release_approver_id: str = eqx.field(static=True)
    validation_data_owner_id: str | None = eqx.field(static=True)
    review_record_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        author_id: str,
        technical_reviewer_id: str,
        release_approver_id: str,
        validation_data_owner_id: str | None = None,
    ):
        values = (
            str(author_id),
            str(technical_reviewer_id),
            str(release_approver_id),
        )
        data_owner = (
            None if validation_data_owner_id is None else str(validation_data_owner_id)
        )
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError("Commercial review roles must be non-empty and independent.")
        self.author_id, self.technical_reviewer_id, self.release_approver_id = values
        self.validation_data_owner_id = data_owner
        self.review_record_id = canonical_fingerprint(
            {
                "kind": "mpm-independent-review",
                "author": values[0],
                "technical_reviewer": values[1],
                "release_approver": values[2],
                "validation_data_owner": data_owner,
            }
        )


class MPMStandardsTrace(StrictModule, NonTrainableState):
    standard: str = eqx.field(static=True)
    edition: str = eqx.field(static=True)
    applicability: str = eqx.field(static=True)
    requirement: str = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    satisfied: bool = eqx.field(static=True)
    trace_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        standard: str,
        edition: str,
        applicability: str,
        requirement: str,
        evidence_ids: Sequence[str],
        satisfied: bool,
    ):
        standard_ = str(standard)
        edition_ = str(edition)
        applicability_ = str(applicability)
        requirement_ = str(requirement)
        evidence = tuple(str(value) for value in evidence_ids)
        if (
            any(
                not value for value in (standard_, edition_, applicability_, requirement_)
            )
            or not evidence
        ):
            raise ValueError("Standards trace record is incomplete.")
        self.standard = standard_
        self.edition = edition_
        self.applicability = applicability_
        self.requirement = requirement_
        self.evidence_ids = evidence
        self.satisfied = bool(satisfied)
        self.trace_id = canonical_fingerprint(
            {
                "kind": "mpm-standards-trace",
                "standard": standard_,
                "edition": edition_,
                "applicability": applicability_,
                "requirement": requirement_,
                "evidence": evidence,
                "satisfied": bool(satisfied),
            }
        )


class MPMStandardsTraceabilityMatrix(StrictModule, NonTrainableState):
    traces: tuple[MPMStandardsTrace, ...]
    matrix_id: str = eqx.field(static=True)

    def __init__(self, traces: Sequence[MPMStandardsTrace], /):
        traces_ = tuple(traces)
        if not traces_ or any(
            not isinstance(value, MPMStandardsTrace) for value in traces_
        ):
            raise TypeError("Standards matrix requires typed trace records.")
        self.traces = traces_
        self.matrix_id = canonical_fingerprint(
            {"kind": "mpm-standards-matrix", "traces": [v.trace_id for v in traces_]}
        )

    @property
    def satisfied(self):
        return all(value.satisfied for value in self.traces)


class MPMCommercialProfile(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    kind: MPMCommercialProfileKind = eqx.field(static=True)
    required_gates: tuple[MPMReleaseGate, ...] = eqx.field(static=True)
    support_matrix: MPMSupportMatrix
    standards: MPMStandardsTraceabilityMatrix
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        kind: MPMCommercialProfileKind,
        support_matrix: MPMSupportMatrix,
        standards: MPMStandardsTraceabilityMatrix,
        /,
    ):
        name_ = str(name)
        kind_ = MPMCommercialProfileKind(kind)
        if (
            not name_
            or not isinstance(support_matrix, MPMSupportMatrix)
            or not isinstance(standards, MPMStandardsTraceabilityMatrix)
        ):
            raise ValueError("Commercial MPM profile is incomplete.")
        if kind_ == MPMCommercialProfileKind.CODE_SOLUTION_VERIFIED:
            required = (
                MPMReleaseGate.INTENDED_USE,
                MPMReleaseGate.CODE_VERIFICATION,
                MPMReleaseGate.SOLUTION_VERIFICATION,
                MPMReleaseGate.PROVENANCE_SBOM,
                MPMReleaseGate.QUALITY_RELIABILITY,
                MPMReleaseGate.RELEASE_DECISION,
            )
        else:
            required = tuple(MPMReleaseGate)
        self.name = name_
        self.kind = kind_
        self.required_gates = required
        self.support_matrix = support_matrix
        self.standards = standards
        self.profile_id = canonical_fingerprint(
            {
                "kind": "mpm-commercial-profile",
                "name": name_,
                "profile_kind": int(kind_),
                "required_gates": [int(value) for value in required],
                "support_matrix": support_matrix.matrix_id,
                "standards": standards.matrix_id,
            }
        )


class MPMReleaseAssessment(StrictModule, NonTrainableState):
    profile_id: str = eqx.field(static=True)
    claim_id: str = eqx.field(static=True)
    intended_use_id: str = eqx.field(static=True)
    gate_ids: tuple[str, ...] = eqx.field(static=True)
    review_record_id: str = eqx.field(static=True)
    standards_matrix_id: str = eqx.field(static=True)
    releasable: bool = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    assessment_id: str = eqx.field(static=True)


def assess_release(
    profile: MPMCommercialProfile,
    claim: MPMClaimTuple,
    intended_use: MPMIntendedUse,
    gate_evidence: Mapping[MPMReleaseGate, MPMReleaseGateEvidence],
    review: MPMIndependentReview,
    /,
) -> MPMReleaseAssessment:
    if not isinstance(profile, MPMCommercialProfile):
        raise TypeError("profile must be MPMCommercialProfile.")
    if not isinstance(claim, MPMClaimTuple) or not isinstance(
        intended_use, MPMIntendedUse
    ):
        raise TypeError("Release assessment needs claim and intended use.")
    if not isinstance(review, MPMIndependentReview):
        raise TypeError("review must be MPMIndependentReview.")
    decision = profile.support_matrix.decision(claim.claim_id)
    reasons = []
    if decision.outcome.name != "SUPPORTED":
        reasons.append(f"support:{decision.outcome.name}:{decision.reason}")
    missing = [gate for gate in profile.required_gates if gate not in gate_evidence]
    if missing:
        reasons.append("missing-gates:" + ",".join(value.name for value in missing))
    failed = [
        gate.name
        for gate in profile.required_gates
        if gate in gate_evidence and not gate_evidence[gate].passed
    ]
    if failed:
        reasons.append("failed-gates:" + ",".join(failed))
    deviations = [
        gate.name
        for gate in profile.required_gates
        if gate in gate_evidence and gate_evidence[gate].deviation_ids
    ]
    if deviations:
        reasons.append("unapproved-deviations:" + ",".join(deviations))
    if not profile.standards.satisfied:
        reasons.append("standards-traceability-incomplete")
    releasable = not reasons
    evidence = tuple(gate_evidence[gate] for gate in MPMReleaseGate)
    bundle = MPMReleaseEvidenceBundle(
        claim,
        intended_use,
        evidence,
        independent_approver_id=review.release_approver_id,
    )
    return MPMReleaseAssessment(
        profile.profile_id,
        claim.claim_id,
        intended_use.intended_use_id,
        tuple(value.gate_id for value in evidence),
        review.review_record_id,
        profile.standards.matrix_id,
        releasable & bundle.releasable,
        tuple(reasons),
        canonical_fingerprint(
            {
                "kind": "mpm-release-assessment",
                "profile": profile.profile_id,
                "bundle": bundle.bundle_id,
                "review": review.review_record_id,
                "standards": profile.standards.matrix_id,
                "reasons": reasons,
            }
        ),
    )


__all__ = [
    "MPMCommercialProfile",
    "MPMCommercialProfileKind",
    "MPMIndependentReview",
    "MPMReleaseAssessment",
    "MPMStandardsTrace",
    "MPMStandardsTraceabilityMatrix",
    "assess_release",
]
