#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Two-reference, uncertainty-aware DFN entry; no DFN implementation or authority.

The numerical assessment is deliberately distinct from authenticated admission.
Only evaluate_dfn_entry_gate consumes release trust and purpose-signed records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint
from ...qualification import ReferenceArtifactManifest
from ...qualification._criterion import (
    CampaignObservationRecord,
    CampaignStartRecord,
    QualificationCriterion,
    validate_qualification_causality,
)
from ...qualification._evidence import QualificationEvidence
from ...qualification._trust import SignedQualificationRecord
from ._rights import ReferenceRightsAttestation


_MATCH_AXES = ("parameters", "geometry", "initialization", "hold", "output")


def _identifier(value):
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("Entry identities must be nonempty canonical strings.")
    return value


def _numbers(values, name, *, nonnegative=False):
    if (
        type(values) is not tuple
        or not values
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or (nonnegative and v < 0)
            for v in values
        )
    ):
        raise ValueError(f"{name} must be a nonempty finite numeric tuple.")


@dataclass(frozen=True, slots=True)
class DfnEntryPolicy:
    """Precommitted finite case/observable/time matrix and separate audit limits.

    Values are normalized by explicit SI scales, so unlike an average, the
    maximum cannot dilute a discrepancy by adding easy cases or observables.
    """

    support_tuple_id: str
    source_build_id: str
    samples: tuple[tuple[str, str, float], ...]
    scales: tuple[float, ...]
    discrepancy_threshold: float
    self_convergence_limit: float
    cross_reference_limit: float
    contraction_limit: float
    issued_at: int
    expires_at: int

    def __post_init__(self):
        _identifier(self.support_tuple_id)
        _identifier(self.source_build_id)
        _numbers(self.scales, "Observable SI scales")
        if (
            type(self.samples) is not tuple
            or not self.samples
            or len(self.samples) != len(self.scales)
            or len(set(self.samples)) != len(self.samples)
        ):
            raise ValueError(
                "Entry requires a unique finite sample matrix and exact scales."
            )
        for case, observable, time in self.samples:
            _identifier(case)
            _identifier(observable)
            if type(time) not in (int, float) or not math.isfinite(time) or time < 0:
                raise ValueError("Sample times must be finite and nonnegative.")
        if min(self.scales) <= 0:
            raise ValueError("Observable scales must be positive.")
        _numbers(
            (
                self.discrepancy_threshold,
                self.self_convergence_limit,
                self.cross_reference_limit,
                self.contraction_limit,
            ),
            "Audit limits",
        )
        if (
            min(
                self.discrepancy_threshold,
                self.self_convergence_limit,
                self.cross_reference_limit,
            )
            <= 0
            or not 0 < self.contraction_limit < 1
        ):
            raise ValueError(
                "Audit limits must be positive; contraction must be in (0, 1)."
            )
        if (
            type(self.issued_at) is not int
            or type(self.expires_at) is not int
            or not 0 <= self.issued_at < self.expires_at
        ):
            raise ValueError("Policy validity interval is invalid.")

    def to_record(self):
        return {
            "kind": "battery-dfn-two-reference-entry-policy",
            "support_tuple_id": self.support_tuple_id,
            "source_build_id": self.source_build_id,
            "samples": [list(s) for s in self.samples],
            "scales": list(self.scales),
            "discrepancy_threshold": self.discrepancy_threshold,
            "self_convergence_limit": self.self_convergence_limit,
            "cross_reference_limit": self.cross_reference_limit,
            "contraction_limit": self.contraction_limit,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    @property
    def policy_id(self):
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True, slots=True)
class DfnReferenceComparison:
    """Immutable reference sample values with actual three-level convergence data."""

    engine_id: str
    runtime_family: str
    manifest: ReferenceArtifactManifest
    mapping_ids: tuple[tuple[str, str], ...]
    coarse_values: tuple[float, ...]
    medium_values: tuple[float, ...]
    fine_values: tuple[float, ...]
    uncertainty: tuple[float, ...]
    discretization_ids: tuple[str, str, str]
    rights: ReferenceRightsAttestation
    rights_signature: SignedQualificationRecord

    def __post_init__(self):
        _identifier(self.engine_id)
        _identifier(self.runtime_family)
        if not isinstance(self.manifest, ReferenceArtifactManifest):
            raise TypeError("Reference comparison requires a real artifact manifest.")
        if (
            type(self.mapping_ids) is not tuple
            or tuple(k for k, _ in self.mapping_ids) != _MATCH_AXES
        ):
            raise ValueError(
                "Mapping audit must bind parameters, geometry, initialization, hold, output in order."
            )
        for _, value in self.mapping_ids:
            _identifier(value)
        _numbers(self.coarse_values, "coarse_values")
        _numbers(self.medium_values, "medium_values")
        _numbers(self.fine_values, "fine_values")
        _numbers(self.uncertainty, "uncertainty", nonnegative=True)
        if (
            len(
                {
                    len(self.coarse_values),
                    len(self.medium_values),
                    len(self.fine_values),
                    len(self.uncertainty),
                }
            )
            != 1
        ):
            raise ValueError(
                "Reference levels and uncertainty must have identical coverage."
            )
        if (
            type(self.discretization_ids) is not tuple
            or len(self.discretization_ids) != 3
            or len(set(self.discretization_ids)) != 3
        ):
            raise ValueError(
                "Three distinct independently refined discretizations are required."
            )
        for value in self.discretization_ids:
            _identifier(value)

    def to_record(self):
        return {
            "engine_id": self.engine_id,
            "runtime_family": self.runtime_family,
            "manifest": self.manifest.to_record(),
            "mapping_ids": dict(self.mapping_ids),
            "coarse_values": list(self.coarse_values),
            "medium_values": list(self.medium_values),
            "fine_values": list(self.fine_values),
            "uncertainty": list(self.uncertainty),
            "discretization_ids": list(self.discretization_ids),
            "rights_id": self.rights.rights_id,
        }


@dataclass(frozen=True, slots=True)
class DfnEntryAudit:
    """One precommitted criterion, durable start, observation and signed outcome."""

    name: str
    criterion: QualificationCriterion
    start: CampaignStartRecord
    observation: CampaignObservationRecord
    evidence: QualificationEvidence
    criterion_signature: SignedQualificationRecord
    start_signature: SignedQualificationRecord
    observation_signature: SignedQualificationRecord
    evidence_signature: SignedQualificationRecord

    def to_record(self):
        return {
            "name": self.name,
            "criterion": self.criterion.to_record(),
            "start": self.start.to_record(),
            "observation": self.observation.to_record(),
            "evidence": self.evidence.to_record(),
            "criterion_signature": self.criterion_signature.to_record(),
            "start_signature": self.start_signature.to_record(),
            "observation_signature": self.observation_signature.to_record(),
            "evidence_signature": self.evidence_signature.to_record(),
        }

    @classmethod
    def from_record(cls, record):
        return cls(
            record["name"],
            QualificationCriterion.from_record(record["criterion"]),
            CampaignStartRecord.from_record(record["start"]),
            CampaignObservationRecord.from_record(record["observation"]),
            QualificationEvidence.from_record(record["evidence"]),
            SignedQualificationRecord.from_record(record["criterion_signature"]),
            SignedQualificationRecord.from_record(record["start_signature"]),
            SignedQualificationRecord.from_record(record["observation_signature"]),
            SignedQualificationRecord.from_record(record["evidence_signature"]),
        )


@dataclass(frozen=True, slots=True)
class DfnEntryAssessment:
    policy: DfnEntryPolicy
    references: tuple[DfnReferenceComparison, DfnReferenceComparison]
    candidate_values: tuple[float, ...]
    candidate_uncertainty: tuple[float, ...]
    candidate_mapping_ids: tuple[tuple[str, str], ...]
    started_at: int
    observed_at: int
    raw_artifact_ids: tuple[str, ...]
    criterion_approval: SignedQualificationRecord | None = None
    executor_signature: SignedQualificationRecord | None = None
    reviewer_signature: SignedQualificationRecord | None = None
    decision_signature: SignedQualificationRecord | None = None
    audits: tuple[DfnEntryAudit, ...] = ()

    def __post_init__(self):
        if not isinstance(self.policy, DfnEntryPolicy):
            raise TypeError("Entry policy must be typed and precommitted.")
        if (
            type(self.references) is not tuple
            or len(self.references) != 2
            or not all(isinstance(r, DfnReferenceComparison) for r in self.references)
        ):
            raise ValueError("Exactly two reference comparisons are required.")
        _numbers(self.candidate_values, "Candidate values")
        _numbers(self.candidate_uncertainty, "Candidate uncertainty", nonnegative=True)
        if (
            not len(self.candidate_values)
            == len(self.candidate_uncertainty)
            == len(self.policy.samples)
        ):
            raise ValueError(
                "Candidate data must cover every precommitted sample exactly."
            )
        if (
            type(self.candidate_mapping_ids) is not tuple
            or tuple(k for k, _ in self.candidate_mapping_ids) != _MATCH_AXES
        ):
            raise ValueError("Candidate mapping audit axes are incomplete.")
        for _, value in self.candidate_mapping_ids:
            _identifier(value)
        if (
            type(self.started_at) is not int
            or type(self.observed_at) is not int
            or not self.policy.issued_at
            < self.started_at
            <= self.observed_at
            < self.policy.expires_at
        ):
            raise ValueError(
                "Policy must predate start, with an observation inside its validity interval."
            )
        if (
            type(self.raw_artifact_ids) is not tuple
            or len(self.raw_artifact_ids) < 3
            or len(set(self.raw_artifact_ids)) != len(self.raw_artifact_ids)
        ):
            raise ValueError(
                "Candidate and both independent raw artifacts must be cited."
            )
        for raw in self.raw_artifact_ids:
            _identifier(raw)

    def observation_record(self):
        return {
            "kind": "battery-dfn-two-reference-assessment",
            "policy_id": self.policy.policy_id,
            "references": [r.to_record() for r in self.references],
            "candidate_values": list(self.candidate_values),
            "candidate_uncertainty": list(self.candidate_uncertainty),
            "candidate_mapping_ids": dict(self.candidate_mapping_ids),
            "started_at": self.started_at,
            "observed_at": self.observed_at,
            "raw_artifact_ids": list(self.raw_artifact_ids),
        }

    @property
    def observation_id(self):
        return canonical_fingerprint(self.observation_record())

    def to_record(self):
        return {
            **self.observation_record(),
            "audits": [audit.to_record() for audit in self.audits],
        }

    @property
    def assessment_id(self):
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True, slots=True)
class DfnReferenceConsensus:
    """Numerical result only; eligible here means a proposed scientific decision."""

    assessment_id: str
    eligible: bool
    conclusive: bool
    reason: str
    reference_intervals: tuple[tuple[float, float], ...]
    self_convergence_errors: tuple[float, ...]
    cross_reference_error: float | None
    robust_lower_bound: float | None

    def to_record(self):
        return {
            "kind": "battery-dfn-reference-consensus",
            "assessment_id": self.assessment_id,
            "eligible": self.eligible,
            "conclusive": self.conclusive,
            "reason": self.reason,
            "reference_intervals": [list(x) for x in self.reference_intervals],
            "self_convergence_errors": list(self.self_convergence_errors),
            "cross_reference_error": self.cross_reference_error,
            "robust_lower_bound": self.robust_lower_bound,
        }


def evaluate_reference_consensus(
    assessment: DfnEntryAssessment, /
) -> DfnReferenceConsensus:
    """Min of per-reference maximum-discrepancy lower bounds, never pooled max."""
    if not isinstance(assessment, DfnEntryAssessment):
        raise TypeError("assessment must be DfnEntryAssessment.")
    policy, refs = assessment.policy, assessment.references
    intervals, self_errors = [], []

    def result(reason, eligible=False, conclusive=False, cross=None):
        return DfnReferenceConsensus(
            assessment.assessment_id,
            eligible,
            conclusive,
            reason,
            tuple(intervals),
            tuple(self_errors),
            cross,
            min(x[0] for x in intervals) if len(intervals) == 2 else None,
        )

    if (
        refs[0].engine_id == refs[1].engine_id
        or refs[0].runtime_family == refs[1].runtime_family
        or refs[0].manifest.manifest_id == refs[1].manifest.manifest_id
        or refs[0].manifest.checksum == refs[1].manifest.checksum
    ):
        return result("references-not-independent")
    for reference in refs:
        if reference.mapping_ids != assessment.candidate_mapping_ids or len(
            reference.fine_values
        ) != len(policy.samples):
            return result("reference-mapping-or-coverage-mismatch")
        coarse_gap = max(
            abs(m - c) / scale
            for c, m, scale in zip(
                reference.coarse_values,
                reference.medium_values,
                policy.scales,
                strict=True,
            )
        )
        fine_gap = max(
            abs(f - m) / scale
            for m, f, scale in zip(
                reference.medium_values, reference.fine_values, policy.scales, strict=True
            )
        )
        self_errors.append(fine_gap)
        if (
            fine_gap > policy.self_convergence_limit
            or fine_gap > policy.contraction_limit * coarse_gap
        ):
            return result("reference-self-convergence-inconclusive")
        declared = dict(reference.manifest.uncertainty)
        if any(observable not in declared for _, observable, _ in policy.samples):
            return result("reference-uncertainty-coverage-incomplete")
        for (_, observable, _), mid, fine, uncertainty in zip(
            policy.samples,
            reference.medium_values,
            reference.fine_values,
            reference.uncertainty,
            strict=True,
        ):
            if uncertainty < max(
                declared[observable], abs(fine - mid) / (1 - policy.contraction_limit)
            ):
                return result("reference-uncertainty-understates-convergence")
        lower, upper = [], []
        for candidate, fine, own_u, ref_u, scale in zip(
            assessment.candidate_values,
            reference.fine_values,
            assessment.candidate_uncertainty,
            reference.uncertainty,
            policy.scales,
            strict=True,
        ):
            discrepancy, uncertainty = abs(candidate - fine), own_u + ref_u
            lower.append(max(0.0, discrepancy - uncertainty) / scale)
            upper.append((discrepancy + uncertainty) / scale)
        intervals.append((max(lower), max(upper)))
    cross = max(
        abs(a - b) / scale
        for a, b, scale in zip(
            refs[0].fine_values, refs[1].fine_values, policy.scales, strict=True
        )
    )
    if cross > policy.cross_reference_limit:
        return result("cross-reference-disagreement", cross=cross)
    robust = min(interval[0] for interval in intervals)
    if robust > policy.discrepancy_threshold:
        return result("dfn-entry-criteria-satisfied", True, True, cross)
    if all(interval[1] < policy.discrepancy_threshold for interval in intervals):
        return result("discrepancy-below-precommitted-threshold", False, True, cross)
    return result("discrepancy-uncertainty-overlap", cross=cross)


def evaluate_dfn_entry_gate(
    *,
    release_index,
    trust_policy,
    marquis_profile_id,
    marquis_support,
    assessment: DfnEntryAssessment | None,
    at_time: int,
):
    """Authenticate policy, execution, review, rights and decision against live SPMe.

    No existing DFN-entry profile is required: this decision is its prerequisite,
    not its dependent. The release builder signs an eligible-only entry profile.
    """
    from ...qualification import require_profile
    from ...qualification._trust import AsymmetricReleaseTrustPolicy
    from ._qualification import BatteryExpansionGateDecision, MARQUIS_2019_SPME_SUPPORT
    from ._release import BatteryReleaseRecord
    from ._validity import MARQUIS_2019_SPME_ENVELOPE

    def refusal(reason):
        return BatteryExpansionGateDecision(False, False, reason, ())

    if type(at_time) is not int or not 0 <= at_time <= 2**63 - 1:
        return refusal("invalid-decision-time")
    if (
        not isinstance(trust_policy, AsymmetricReleaseTrustPolicy)
        or marquis_support != MARQUIS_2019_SPME_SUPPORT
    ):
        return refusal("dfn-release-not-admitted")
    if not isinstance(assessment, DfnEntryAssessment):
        return refusal("two-reference-assessment-missing")
    try:
        profile = require_profile(
            release_index,
            marquis_profile_id,
            MARQUIS_2019_SPME_SUPPORT,
            trust_policy,
            at_time=at_time,
        )
        if (
            tuple(s.support_tuple_id for s in profile.support_tuples)
            != (MARQUIS_2019_SPME_SUPPORT.support_tuple_id,)
            or assessment.policy.support_tuple_id
            != MARQUIS_2019_SPME_SUPPORT.support_tuple_id
            or not assessment.observed_at <= at_time < assessment.policy.expires_at
        ):
            return refusal("dfn-reference-scope-mismatch")
        proofs = tuple(
            proof
            for proof in trust_policy.proofs
            if isinstance(proof, BatteryReleaseRecord)
            and proof.profile.profile_id == marquis_profile_id
        )
        if (
            len(proofs) != 1
            or proofs[0].distribution_id != assessment.policy.source_build_id
            or proofs[0].envelope_id != MARQUIS_2019_SPME_ENVELOPE.envelope_id
        ):
            return refusal("dfn-source-build-or-envelope-mismatch")
        proofs[0].verify(trust_policy, at_time=at_time)
        roles = trust_policy.roles
        roles.verify(
            assessment.policy,
            assessment.criterion_approval,
            role="criterion-approver",
            at_time=at_time,
        )
        if assessment.criterion_approval.issued_at >= assessment.started_at:
            return refusal("dfn-policy-not-precommitted")
        roles.verify(
            assessment, assessment.executor_signature, role="executor", at_time=at_time
        )
        roles.verify(
            assessment,
            assessment.reviewer_signature,
            role="scientific-reviewer",
            at_time=at_time,
        )
        if (
            min(
                assessment.executor_signature.issued_at,
                assessment.reviewer_signature.issued_at,
            )
            < assessment.observed_at
        ):
            return refusal("dfn-signature-precedes-observation")
        for reference in assessment.references:
            reference.rights.verify(
                reference.manifest, reference.rights_signature, roles, at_time=at_time
            )
        consensus = evaluate_reference_consensus(assessment)
        if not consensus.conclusive:
            return refusal(consensus.reason)
        _verify_audits(assessment, consensus, roles, marquis_profile_id, at_time)
        roles.verify(
            consensus,
            assessment.decision_signature,
            role="entry-decision-authority",
            at_time=at_time,
        )
        if (
            assessment.decision_signature.issued_at
            < assessment.reviewer_signature.issued_at
        ):
            return refusal("dfn-decision-precedes-review")
    except (AttributeError, KeyError, TypeError, ValueError):
        return refusal("dfn-authentication-or-rights-inconclusive")
    citations = (
        assessment.assessment_id,
        assessment.policy.policy_id,
        canonical_fingerprint(consensus.to_record()),
        *(reference.manifest.manifest_id for reference in assessment.references),
        *(reference.rights.rights_id for reference in assessment.references),
        *assessment.raw_artifact_ids,
        *(audit.evidence.evidence_id for audit in assessment.audits),
        *(item.evidence_id for item in profile.release_evidence),
    )
    return BatteryExpansionGateDecision(
        consensus.eligible,
        consensus.conclusive,
        consensus.reason,
        tuple(sorted(set(citations))),
    )


def _verify_audits(assessment, consensus, roles, source_profile_id, at_time):
    expected = {f"mapping:{axis}": (0.0, 0.0, "equal") for axis in _MATCH_AXES}
    expected.update(
        {
            "self-convergence:first": (
                consensus.self_convergence_errors[0],
                assessment.policy.self_convergence_limit,
                "less-than-or-equal",
            ),
            "self-convergence:second": (
                consensus.self_convergence_errors[1],
                assessment.policy.self_convergence_limit,
                "less-than-or-equal",
            ),
            "cross-reference-agreement": (
                consensus.cross_reference_error,
                assessment.policy.cross_reference_limit,
                "less-than-or-equal",
            ),
            "robust-discrepancy": (
                consensus.robust_lower_bound,
                assessment.policy.discrepancy_threshold,
                "greater-than-or-equal",
            ),
        }
    )
    audits = {audit.name: audit for audit in assessment.audits}
    if len(audits) != len(assessment.audits) or set(audits) != set(expected):
        raise ValueError(
            "DFN entry requires exactly one retained outcome for every mapping, "
            "convergence, consensus and discrepancy criterion."
        )
    for name, (value, target, comparison) in expected.items():
        audit = audits[name]
        criterion, start, observation, evidence = (
            audit.criterion,
            audit.start,
            audit.observation,
            audit.evidence,
        )
        validate_qualification_causality(criterion, start, observation, evidence)
        if (
            criterion.support_tuple_id != assessment.policy.support_tuple_id
            or criterion.applicability != assessment.policy.policy_id
            or criterion.metric != name
            or criterion.unit != "1"
            or criterion.aggregation != "maximum"
            or criterion.uncertainty != "two-reference-interval"
            or criterion.comparison != comparison
            or criterion.target != target
            or not criterion.is_valid(at_time)
            or start.started_at != assessment.started_at
            or observation.observed_at != assessment.observed_at
            or evidence.subject_ids
            != tuple(sorted((source_profile_id, assessment.policy.support_tuple_id)))
            or evidence.build_id != assessment.policy.source_build_id
            or not evidence.is_current(at_time)
            or evidence.criteria_ids != (criterion.criterion_id,)
            or evidence.campaign_start_record_ids != (start.start_record_id,)
            or evidence.campaign_observation_record_ids
            != (observation.observation_record_id,)
            or assessment.observation_id not in evidence.raw_artifact_ids
            or not set(assessment.raw_artifact_ids).issubset(evidence.raw_artifact_ids)
            or not {r.manifest.manifest_id for r in assessment.references}.issubset(
                evidence.raw_artifact_ids
            )
        ):
            raise ValueError(
                "DFN entry audit scope, timing or exact observation citations differ."
            )
        passed = (
            value == target
            if comparison == "equal"
            else value <= target
            if comparison == "less-than-or-equal"
            else value >= target
        )
        if evidence.inconclusive or evidence.passed != passed:
            raise ValueError(
                "DFN audit outcome disagrees with the actual observed statistic."
            )
        roles.verify(
            criterion,
            audit.criterion_signature,
            role="criterion-approver",
            at_time=at_time,
        )
        roles.verify(start, audit.start_signature, role="executor", at_time=at_time)
        roles.verify(
            observation, audit.observation_signature, role="executor", at_time=at_time
        )
        roles.verify(
            evidence,
            audit.evidence_signature,
            role="scientific-reviewer",
            at_time=at_time,
        )
        if (
            audit.criterion_signature.issued_at >= start.started_at
            or audit.start_signature.issued_at != start.started_at
            or audit.observation_signature.issued_at < observation.observed_at
            or audit.evidence_signature.issued_at < evidence.issued_at
            or audit.evidence_signature.issued_at
            > assessment.reviewer_signature.issued_at
        ):
            raise ValueError(
                "DFN entry audit signatures violate approval/start/observation/review causality."
            )


def _assessment_archive(assessment):
    return {
        "assessment": assessment.to_record(),
        "policy": assessment.policy.to_record(),
        "rights": [reference.rights.to_record() for reference in assessment.references],
        "rights_signatures": [
            reference.rights_signature.to_record() for reference in assessment.references
        ],
        "criterion_approval": assessment.criterion_approval.to_record(),
        "executor_signature": assessment.executor_signature.to_record(),
        "reviewer_signature": assessment.reviewer_signature.to_record(),
        "decision_signature": assessment.decision_signature.to_record(),
    }


def _assessment_from_archive(record):
    raw_policy = dict(record["policy"])
    if raw_policy.pop("kind") != "battery-dfn-two-reference-entry-policy":
        raise ValueError("Wrong entry policy record kind.")
    raw_policy["samples"] = tuple(tuple(sample) for sample in raw_policy["samples"])
    raw_policy["scales"] = tuple(raw_policy["scales"])
    policy = DfnEntryPolicy(**raw_policy)
    raw = record["assessment"]
    if (
        raw["kind"] != "battery-dfn-two-reference-assessment"
        or raw["policy_id"] != policy.policy_id
    ):
        raise ValueError("Archived entry assessment does not bind the exact policy.")
    references = []
    for reference, rights, signature in zip(
        raw["references"], record["rights"], record["rights_signatures"], strict=True
    ):
        claim = ReferenceRightsAttestation.from_record(rights)
        if claim.rights_id != reference["rights_id"]:
            raise ValueError("Archived reference rights identity differs.")
        references.append(
            DfnReferenceComparison(
                reference["engine_id"],
                reference["runtime_family"],
                ReferenceArtifactManifest.from_record(reference["manifest"]),
                tuple((axis, reference["mapping_ids"][axis]) for axis in _MATCH_AXES),
                tuple(reference["coarse_values"]),
                tuple(reference["medium_values"]),
                tuple(reference["fine_values"]),
                tuple(reference["uncertainty"]),
                tuple(reference["discretization_ids"]),
                claim,
                SignedQualificationRecord.from_record(signature),
            )
        )
    assessment = DfnEntryAssessment(
        policy,
        tuple(references),
        tuple(raw["candidate_values"]),
        tuple(raw["candidate_uncertainty"]),
        tuple((axis, raw["candidate_mapping_ids"][axis]) for axis in _MATCH_AXES),
        raw["started_at"],
        raw["observed_at"],
        tuple(raw["raw_artifact_ids"]),
        SignedQualificationRecord.from_record(record["criterion_approval"]),
        SignedQualificationRecord.from_record(record["executor_signature"]),
        SignedQualificationRecord.from_record(record["reviewer_signature"]),
        SignedQualificationRecord.from_record(record["decision_signature"]),
        tuple(DfnEntryAudit.from_record(audit) for audit in raw["audits"]),
    )
    if canonical_fingerprint(raw) != canonical_fingerprint(assessment.to_record()):
        raise ValueError("Archived assessment has unrecognized or substituted content.")
    return assessment


def _entry_profile(assessment, source_profile_id, issued_at, expires_at):
    from ...qualification import CapabilityProfile, ReleaseGateEvidence, SupportDependency
    from ._qualification import DFN_ENTRY_SUPPORT

    consensus = evaluate_reference_consensus(assessment)
    if not consensus.eligible:
        raise ValueError("Only an eligible consensus can construct a DFN-entry profile.")
    gate = ReleaseGateEvidence(
        "battery.dfn-entry.scientific",
        passed=True,
        evidence_ids=(
            assessment.assessment_id,
            assessment.policy.policy_id,
            canonical_fingerprint(consensus.to_record()),
        ),
        reviewer_id=assessment.reviewer_signature.signature.key_id,
        issued_at=issued_at,
        expires_at=expires_at,
    )
    return CapabilityProfile(
        "battery.dfn-entry",
        "phydrax",
        "release",
        (DFN_ENTRY_SUPPORT,),
        dependencies=(
            SupportDependency(source_profile_id, assessment.policy.support_tuple_id),
        ),
        required_gates=(gate.gate,),
        release_evidence=(gate,),
        released=True,
    )


def _entry_expiry(assessment, source_proof, roles):
    limits = [
        assessment.policy.expires_at,
        source_proof.expires_at,
        *(reference.rights.expires_at for reference in assessment.references),
    ]
    signatures = [
        assessment.criterion_approval,
        assessment.executor_signature,
        assessment.reviewer_signature,
        assessment.decision_signature,
        *(reference.rights_signature for reference in assessment.references),
    ]
    for audit in assessment.audits:
        if audit.criterion.valid_until is not None:
            limits.append(audit.criterion.valid_until)
        limits.append(audit.evidence.expires_at)
        signatures.extend(
            (
                audit.criterion_signature,
                audit.start_signature,
                audit.observation_signature,
                audit.evidence_signature,
            )
        )
    return min((*limits, *(roles.expiry(signature) for signature in signatures)))


@dataclass(frozen=True, slots=True)
class DfnEntryReleaseRecord:
    """Retained eligible-only proof for a signed staging ReleaseIndex."""

    assessment: DfnEntryAssessment
    source_profile_id: str
    source_index: object
    profile: object
    issued_at: int
    expires_at: int

    @property
    def gate_id(self):
        return self.profile.release_evidence[0].evidence_id

    @property
    def distribution_id(self):
        return self.assessment.policy.source_build_id

    @property
    def envelope_id(self):
        from ._validity import MARQUIS_2019_SPME_ENVELOPE

        return MARQUIS_2019_SPME_ENVELOPE.envelope_id

    def verify(self, policy, /, *, at_time):
        from ._qualification import MARQUIS_2019_SPME_SUPPORT
        from ._release import BatteryReleaseRecord

        if not self.issued_at <= at_time < self.expires_at:
            raise ValueError("DFN entry release is stale or inactive.")
        decision = evaluate_dfn_entry_gate(
            release_index=self.source_index,
            trust_policy=policy,
            marquis_profile_id=self.source_profile_id,
            marquis_support=MARQUIS_2019_SPME_SUPPORT,
            assessment=self.assessment,
            at_time=at_time,
        )
        if not decision.eligible:
            raise ValueError(
                f"DFN entry prerequisite is no longer eligible: {decision.reason}"
            )
        proofs = tuple(
            proof
            for proof in policy.proofs
            if isinstance(proof, BatteryReleaseRecord)
            and proof.profile.profile_id == self.source_profile_id
        )
        if len(proofs) != 1 or self.expires_at > min(
            _entry_expiry(self.assessment, proofs[0], policy.roles),
            self.source_index.issued_at + policy.max_index_age,
        ):
            raise ValueError("DFN entry outlives a retained prerequisite.")
        expected = _entry_profile(
            self.assessment, self.source_profile_id, self.issued_at, self.expires_at
        )
        if self.profile.to_record() != expected.to_record():
            raise ValueError(
                "DFN entry profile, dependency or citations were substituted."
            )

    def to_record(self):
        return {
            "kind": "battery-dfn-entry-release",
            "assessment_archive": _assessment_archive(self.assessment),
            "source_profile_id": self.source_profile_id,
            "source_index": self.source_index.to_record(),
            "profile": self.profile.to_record(),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_record(cls, record):
        from ...qualification import CapabilityProfile, ReleaseIndex

        if record["kind"] != "battery-dfn-entry-release":
            raise ValueError("Wrong DFN-entry proof kind.")
        return cls(
            _assessment_from_archive(record["assessment_archive"]),
            record["source_profile_id"],
            ReleaseIndex.from_record(record["source_index"]),
            CapabilityProfile.from_record(record["profile"]),
            record["issued_at"],
            record["expires_at"],
        )


def build_dfn_entry_release(
    assessment, /, *, source_index, source_profile_id, trust_policy, at_time, expires_at
):
    from ._qualification import MARQUIS_2019_SPME_SUPPORT
    from ._release import BatteryReleaseRecord

    decision = evaluate_dfn_entry_gate(
        release_index=source_index,
        trust_policy=trust_policy,
        marquis_profile_id=source_profile_id,
        marquis_support=MARQUIS_2019_SPME_SUPPORT,
        assessment=assessment,
        at_time=at_time,
    )
    if not decision.eligible:
        raise ValueError(f"DFN entry cannot be released: {decision.reason}")
    if type(expires_at) is not int or expires_at <= at_time:
        raise ValueError("DFN entry expiry must follow issuance.")
    proofs = tuple(
        proof
        for proof in trust_policy.proofs
        if isinstance(proof, BatteryReleaseRecord)
        and proof.profile.profile_id == source_profile_id
    )
    if len(proofs) != 1:
        raise ValueError("DFN entry needs one exact retained SPMe proof.")
    expiry = min(
        expires_at,
        _entry_expiry(assessment, proofs[0], trust_policy.roles),
        source_index.issued_at + trust_policy.max_index_age,
    )
    if expiry <= at_time:
        raise ValueError("DFN entry prerequisite expires before issuance.")
    return DfnEntryReleaseRecord(
        assessment,
        source_profile_id,
        source_index,
        _entry_profile(assessment, source_profile_id, at_time, expiry),
        at_time,
        expiry,
    )
