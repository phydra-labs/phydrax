#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import replace
from pathlib import Path

import pytest

from phydrax.applications.cardiovascular._case import CardiovascularCaseManifest
from phydrax.applications.cardiovascular._commercial import (
    assess_cardiovascular_release,
    CardiovascularArtifactKind,
    CardiovascularArtifactReference,
    CardiovascularArtifactSet,
    CardiovascularClaimDecision,
    CardiovascularClaimsMatrix,
    CardiovascularClaimStatus,
    CardiovascularCommercialSupportProfile,
    CardiovascularGateEvidence,
    CardiovascularPrivacyPolicy,
    CardiovascularQualificationBundle,
    CardiovascularReleaseDecision,
    CardiovascularReleaseGate,
    CardiovascularResourcePolicy,
    CardiovascularReviewRoles,
    CardiovascularSecurityPolicy,
    CardiovascularSignedNonClaim,
    CardiovascularUsePolicy,
    evaluate_cardiovascular_release_candidate,
    make_cardiovascular_release_decision,
)
from phydrax.applications.cardiovascular._execution import (
    CardiovascularCapacityManifest,
    CardiovascularExecutionManifest,
    CardiovascularSerialExecution,
)
from phydrax.artifacts import ArtifactManifest
from phydrax.lifecycle import RunRecord
from phydrax.qualification import (
    CapabilityProfile,
    HMACSHA256ReleaseSigner,
    HMACSHA256TrustPolicy,
    ReleaseGateEvidence,
    SupportTuple,
)
from tools.cardiovascular_release_qualification import (
    build_cardiovascular_release_artifacts,
    cardiovascular_release_preflight,
)


_REQUIRED_EXCLUSIONS = (
    "clinical-decision-support",
    "diagnosis",
    "regulated-medical-device",
    "treatment",
)


def _support(**overrides: str | bool) -> SupportTuple:
    attributes: dict[str, str | int | bool] = {
        "data_classification": "non-phi",
        "deployment": "local",
        "fidelity_route": "monodomain-lumped-circulation",
        "precision": "float64",
        "regulated_device": False,
    }
    attributes.update(overrides)
    return SupportTuple("cardiovascular.workflow", attributes)


def _manifest(kind: CardiovascularArtifactKind, ordinal: int) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=f"cardiovascular-{kind.value}",
        producer="qualification-fixture",
        version="1",
        sha256=f"{ordinal:x}" * 64,
        byte_size=ordinal,
        source_uri=f"file:///qualification/{kind.value}.json",
        license_id="LicenseRef-Commercial-Evaluation",
        model="cardiovascular-release",
        coverage="exact-support-tuple",
    )


def _artifacts(*, expires_at: int = 100) -> CardiovascularArtifactSet:
    references: list[CardiovascularArtifactReference] = []
    for ordinal, kind in enumerate(CardiovascularArtifactKind, start=1):
        dependencies: tuple[str, ...] = ()
        if kind == CardiovascularArtifactKind.SUPPLY_CHAIN_ATTESTATION:
            dependencies = (
                next(
                    reference.reference_id
                    for reference in references
                    if reference.kind == CardiovascularArtifactKind.SBOM
                ),
                next(
                    reference.reference_id
                    for reference in references
                    if reference.kind == CardiovascularArtifactKind.BUILD_PROVENANCE
                ),
            )
        references.append(
            CardiovascularArtifactReference(
                kind,
                _manifest(kind, ordinal),
                issued_at=10,
                expires_at=expires_at,
                dependency_reference_ids=dependencies,
            )
        )
    return CardiovascularArtifactSet(tuple(references))


class _HMACVerifier:
    def __init__(self, signer_id: str, secret: bytes):
        self._signer_id = signer_id
        self._secret = secret

    @property
    def signer_id(self) -> str:
        return self._signer_id

    @property
    def signature_algorithm(self) -> str:
        return "hmac-sha256"

    def verify(self, payload: bytes, signature: bytes, /) -> bool:
        expected = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)


def _profile_parts(support: SupportTuple, dependency_profile_ids: tuple[str, ...] = ()):
    claims = CardiovascularClaimsMatrix(
        (
            CardiovascularClaimDecision(
                support,
                CardiovascularClaimStatus.TECHNICAL_SUPPORT_CANDIDATE,
                ("research-grade-forward-simulation",),
                _REQUIRED_EXCLUSIONS,
                "Scope is limited to qualified engineering simulation.",
            ),
        )
    )
    resources = CardiovascularResourcePolicy(
        maximum_wall_time_seconds=3600,
        maximum_resident_bytes=2**30,
        maximum_artifact_bytes=2**28,
        maximum_concurrent_runs=1,
    )
    privacy = CardiovascularPrivacyPolicy(maximum_retention_days=7)
    roles = CardiovascularReviewRoles(
        "author",
        "technical-reviewer",
        "validation-reviewer",
        "security-reviewer",
        "release-approver",
    )
    security = CardiovascularSecurityPolicy(
        authorized_reviewer_ids=(
            roles.technical_reviewer_id,
            roles.validation_reviewer_id,
            roles.security_reviewer_id,
        ),
        trusted_signer_ids=(
            roles.technical_reviewer_id,
            roles.validation_reviewer_id,
            roles.security_reviewer_id,
            roles.release_approver_id,
        ),
    )
    use = CardiovascularUsePolicy("Local, non-PHI engineering research and evaluation.")
    profile = CardiovascularCommercialSupportProfile(
        "cardiovascular.local-non-phi",
        "qualification-1",
        support,
        claims,
        resources,
        privacy,
        security,
        use,
        dependency_profile_ids=dependency_profile_ids,
    )
    return profile, roles


def _complete_case(
    *,
    artifact_expires_at: int = 100,
    dependency_profile_ids: tuple[str, ...] = (),
):
    support = _support()
    profile, roles = _profile_parts(support, dependency_profile_ids)
    secrets = {
        roles.technical_reviewer_id: b"technical-key",
        roles.validation_reviewer_id: b"validation-key",
        roles.security_reviewer_id: b"security-key",
        roles.release_approver_id: b"approver-key",
    }
    signers = {
        signer_id: HMACSHA256ReleaseSigner(signer_id, secret)
        for signer_id, secret in secrets.items()
    }
    verifiers = {
        signer_id: _HMACVerifier(signer_id, secret)
        for signer_id, secret in secrets.items()
    }
    trust = HMACSHA256TrustPolicy(
        {"release-index-signer": b"index-key"},
        maximum_index_age=100,
        maximum_evidence_age=100,
    )
    non_claims = tuple(
        CardiovascularSignedNonClaim.issue(
            support,
            excluded_use,
            f"PhydraX cardiovascular does not claim {excluded_use} use.",
            author_id=roles.author_id,
            issued_at=10,
            expires_at=100,
            signer=signers[roles.technical_reviewer_id],
        )
        for excluded_use in _REQUIRED_EXCLUSIONS
    )
    artifacts = _artifacts(expires_at=artifact_expires_at)
    artifact_by_kind = {reference.kind: reference for reference in artifacts.references}
    case = CardiovascularCaseManifest(
        "qualification-case",
        "anatomy-definition",
        "coupled-model",
        "pacing-protocol",
        profile.profile_id,
        "upstream-release",
        artifact_by_kind[
            CardiovascularArtifactKind.BUILD_PROVENANCE
        ].manifest.artifact_id,
        artifact_by_kind[CardiovascularArtifactKind.SBOM].manifest.artifact_id,
        observation_ids=("synthetic-observation",),
        license_ids=(
            artifact_by_kind[
                CardiovascularArtifactKind.COMMERCIAL_LICENSE
            ].manifest.artifact_id,
        ),
        data_rights_ids=(
            artifact_by_kind[CardiovascularArtifactKind.DATA_RIGHTS].manifest.artifact_id,
        ),
        metadata={
            "data_classification": "non-phi",
            "intended_use": "engineering-evaluation",
        },
    )
    capacity = CardiovascularCapacityManifest(
        maximum_cohort_cases=1,
        maximum_state_values=1024,
        maximum_checkpoint_arrays=8,
        maximum_checkpoint_bytes=2**20,
        maximum_macro_steps=100,
        maximum_scheduled_steps=1000,
        maximum_events=8,
        maximum_partitions=1,
    )
    execution = CardiovascularExecutionManifest(
        case_manifest_id=case.manifest_id,
        analysis_plan_id="analysis-plan",
        numeric_revision_id="numeric-revision",
        topology_id="fixed-topology",
        solver_policy_id="solver-policy",
        precision_policy_id="float64-policy",
        backend="jax",
        capacity=capacity,
        route=CardiovascularSerialExecution(0),
    )
    run = RunRecord(
        "qualification-run",
        "analysis-plan",
        "numeric-revision",
        execution.manifest_id,
        "completed",
        result_ids=("qualification-result",),
        diagnostic_ids=("qualification-diagnostics",),
    )
    evidence_ids = {
        CardiovascularReleaseGate.G0_INTENDED_USE: (
            profile.claims_matrix.matrix_id,
            profile.use_policy.policy_id,
            case.manifest_id,
            *(record.non_claim_id for record in non_claims),
        ),
        CardiovascularReleaseGate.G1_CODE_VERIFICATION: (
            run.record_id,
            execution.manifest_id,
        ),
        CardiovascularReleaseGate.G2_SOLUTION_VERIFICATION: (
            run.record_id,
            execution.manifest_id,
        ),
        CardiovascularReleaseGate.G3_VALIDATION_UQ: (
            run.record_id,
            execution.manifest_id,
        ),
        CardiovascularReleaseGate.G4_DERIVATIVE_VALIDITY: (
            run.record_id,
            execution.manifest_id,
        ),
        CardiovascularReleaseGate.G5_PROVENANCE_SUPPLY_CHAIN: tuple(
            reference.reference_id for reference in artifacts.references
        ),
        CardiovascularReleaseGate.G6_QUALITY_OPERATIONS: (
            profile.resource_policy.policy_id,
            profile.privacy_policy.policy_id,
            profile.security_policy.policy_id,
        ),
        CardiovascularReleaseGate.G7_INDEPENDENT_RELEASE_REVIEW: (roles.roles_id,),
    }
    gates = tuple(
        CardiovascularGateEvidence.issue(
            gate,
            passed=True,
            evidence_ids=evidence_ids[gate],
            reviewer_id=roles.expected_reviewer(gate),
            dossier_id="cardiovascular-release-dossier",
            issued_at=10,
            expires_at=100,
            signer=signers[roles.expected_reviewer(gate)],
        )
        for gate in CardiovascularReleaseGate
    )
    bundle = CardiovascularQualificationBundle(
        support,
        gates,
        artifacts,
        (run,),
        non_claims,
        roles,
        case,
        (execution,),
    )
    return profile, bundle, trust, signers, verifiers


def test_complete_exact_tuple_requires_separate_release_decision() -> None:
    profile, bundle, trust, signers, verifiers = _complete_case()
    candidate = evaluate_cardiovascular_release_candidate(
        profile,
        bundle,
        trust,
        verifiers,
        at_time=20,
    )

    assert candidate.qualified
    assert not candidate.commercial_ready
    assert candidate.blockers == ()
    assert len(candidate.capability_profile.support_tuples) == 1
    assert not candidate.capability_profile.released

    decision = make_cardiovascular_release_decision(
        candidate,
        bundle.roles,
        approved=True,
        signer=signers[bundle.roles.release_approver_id],
        decided_at=21,
        rationale="All independent technical gates passed.",
    )
    assessment = assess_cardiovascular_release(candidate, decision, trust, verifiers)

    assert assessment.commercial_ready
    assert assessment.blockers == ()
    assert assessment.capability_profile.released
    assert assessment.require_commercial_ready() == assessment.assessment_id
    assert not profile.use_policy.grants_commercial_license


def test_claims_matrix_is_exact_and_profile_rejects_nonlocal_or_regulated_tuple() -> None:
    support = _support()
    decision = CardiovascularClaimDecision(
        support,
        CardiovascularClaimStatus.TECHNICAL_SUPPORT_CANDIDATE,
        ("forward-simulation",),
        _REQUIRED_EXCLUSIONS,
        "Exact route only.",
    )
    with pytest.raises(ValueError, match="exactly one decision"):
        CardiovascularClaimsMatrix((decision, decision))

    profile, _ = _profile_parts(support)
    almost_same = _support(fidelity_route="bidomain-lumped-circulation")
    assert profile.claims_matrix.decision_for(support) is decision or (
        profile.claims_matrix.decision_for(support).support_tuple.support_tuple_id
        == support.support_tuple_id
    )
    with pytest.raises(KeyError, match=almost_same.support_tuple_id):
        profile.claims_matrix.decision_for(almost_same)

    for unsupported in (
        _support(deployment="managed-service"),
        _support(data_classification="phi"),
        _support(regulated_device=True),
    ):
        unsupported_claims = CardiovascularClaimsMatrix(
            (
                CardiovascularClaimDecision(
                    unsupported,
                    CardiovascularClaimStatus.TECHNICAL_SUPPORT_CANDIDATE,
                    ("forward-simulation",),
                    _REQUIRED_EXCLUSIONS,
                    "Unsupported profile boundary.",
                ),
            )
        )
        with pytest.raises(ValueError, match="local, non-PHI, non-regulated"):
            CardiovascularCommercialSupportProfile(
                "cardiovascular.unsupported",
                "1",
                unsupported,
                unsupported_claims,
                profile.resource_policy,
                profile.privacy_policy,
                profile.security_policy,
                profile.use_policy,
            )


def test_profile_refuses_permissive_privacy_security_and_medical_use() -> None:
    support = _support()
    profile, _ = _profile_parts(support)
    permissive_privacy = CardiovascularPrivacyPolicy(phi_allowed=True)
    with pytest.raises(ValueError, match="local and non-PHI"):
        replace(profile, privacy_policy=permissive_privacy)

    permissive_security = replace(profile.security_policy, network_access_allowed=True)
    with pytest.raises(ValueError, match="fail-closed local security"):
        replace(profile, security_policy=permissive_security)

    regulated_use = CardiovascularUsePolicy(
        "Clinical decision support.", regulated_device_use_allowed=True
    )
    with pytest.raises(ValueError, match="Regulated, diagnostic, treatment"):
        replace(profile, use_policy=regulated_use)


def test_absent_release_artifacts_produce_deterministic_current_blockers() -> None:
    profile, bundle, trust, signers, verifiers = _complete_case()
    g5 = CardiovascularGateEvidence.issue(
        CardiovascularReleaseGate.G5_PROVENANCE_SUPPLY_CHAIN,
        passed=True,
        evidence_ids=(profile.security_policy.policy_id,),
        reviewer_id=bundle.roles.security_reviewer_id,
        dossier_id="cardiovascular-release-dossier",
        issued_at=10,
        expires_at=100,
        signer=signers[bundle.roles.security_reviewer_id],
    )
    gates = tuple(
        g5 if gate.gate == CardiovascularReleaseGate.G5_PROVENANCE_SUPPLY_CHAIN else gate
        for gate in bundle.gates
    )
    incomplete = CardiovascularQualificationBundle(
        bundle.support_tuple,
        gates,
        CardiovascularArtifactSet(),
        bundle.lifecycle_records,
        bundle.non_claims,
        bundle.roles,
        bundle.case_manifest,
        bundle.execution_manifests,
    )
    candidate = evaluate_cardiovascular_release_candidate(
        profile,
        incomplete,
        trust,
        verifiers,
        at_time=20,
    )

    assert not candidate.qualified
    assert candidate.blockers[:6] == tuple(
        f"missing-artifact:{kind.value}" for kind in profile.required_artifact_kinds
    )
    with pytest.raises(ValueError, match="blocked cardiovascular candidate"):
        make_cardiovascular_release_decision(
            candidate,
            incomplete.roles,
            approved=True,
            signer=signers[incomplete.roles.release_approver_id],
            decided_at=21,
            rationale="Cannot override missing evidence.",
        )


def test_artifact_dependencies_and_freshness_fail_closed() -> None:
    profile, bundle, trust, signers, verifiers = _complete_case(artifact_expires_at=15)
    candidate = evaluate_cardiovascular_release_candidate(
        profile,
        bundle,
        trust,
        verifiers,
        at_time=20,
    )
    assert not candidate.qualified
    assert any(reason.startswith("stale-artifact:") for reason in candidate.blockers)
    assert any(
        reason.startswith("stale-artifact-dependency:") for reason in candidate.blockers
    )

    detached_supply_chain = CardiovascularArtifactReference(
        CardiovascularArtifactKind.SUPPLY_CHAIN_ATTESTATION,
        _manifest(CardiovascularArtifactKind.SUPPLY_CHAIN_ATTESTATION, 9),
        issued_at=10,
        expires_at=100,
        dependency_reference_ids=("absent-build-reference",),
    )
    detached = CardiovascularArtifactSet(
        tuple(
            reference
            for reference in bundle.artifacts.references
            if reference.kind != CardiovascularArtifactKind.SUPPLY_CHAIN_ATTESTATION
        )
        + (detached_supply_chain,)
    )
    blockers = detached.blockers(profile.required_artifact_kinds, 20)
    assert (
        "missing-artifact-dependency:supply-chain-attestation:absent-build-reference"
    ) in blockers


def test_failed_stale_and_unapproved_gate_evidence_are_distinct() -> None:
    profile, bundle, trust, signers, verifiers = _complete_case()
    replacements = {
        CardiovascularReleaseGate.G1_CODE_VERIFICATION: dict(
            passed=False, issued_at=10, expires_at=100, deviation_ids=()
        ),
        CardiovascularReleaseGate.G2_SOLUTION_VERIFICATION: dict(
            passed=True, issued_at=0, expires_at=15, deviation_ids=()
        ),
        CardiovascularReleaseGate.G3_VALIDATION_UQ: dict(
            passed=True,
            issued_at=10,
            expires_at=100,
            deviation_ids=("open-deviation",),
        ),
    }
    gates = []
    for gate_record in bundle.gates:
        gate = gate_record.gate
        if gate not in replacements:
            gates.append(gate_record)
            continue
        values = replacements[gate]
        gates.append(
            CardiovascularGateEvidence.issue(
                gate,
                passed=values["passed"],
                evidence_ids=gate_record.release_evidence.evidence_ids,
                reviewer_id=bundle.roles.expected_reviewer(gate),
                dossier_id=gate_record.dossier_id,
                issued_at=values["issued_at"],
                expires_at=values["expires_at"],
                signer=signers[bundle.roles.expected_reviewer(gate)],
                deviation_ids=values["deviation_ids"],
            )
        )
    changed = CardiovascularQualificationBundle(
        bundle.support_tuple,
        tuple(gates),
        bundle.artifacts,
        bundle.lifecycle_records,
        bundle.non_claims,
        bundle.roles,
        bundle.case_manifest,
        bundle.execution_manifests,
    )
    candidate = evaluate_cardiovascular_release_candidate(
        profile,
        changed,
        trust,
        verifiers,
        at_time=20,
    )

    assert (
        "failed-gate:g1-code-verification",
        "stale-gate:g2-solution-verification",
        "unapproved-deviation:g3-validation-uq",
    ) == tuple(
        reason
        for reason in candidate.blockers
        if reason.startswith(("failed-gate", "stale-gate", "unapproved-deviation"))
    )


def test_non_claim_signatures_scope_and_freshness_are_enforced() -> None:
    profile, bundle, trust, signers, verifiers = _complete_case()
    original = bundle.non_claims[0]
    tampered = CardiovascularSignedNonClaim(
        original.support_tuple_id,
        original.excluded_use,
        original.statement + " Altered.",
        original.author_id,
        original.issued_at,
        original.expires_at,
        original.signer_id,
        original.signature_algorithm,
        original.signature,
    )
    non_claims = (tampered, *bundle.non_claims[1:])
    g0 = CardiovascularGateEvidence.issue(
        CardiovascularReleaseGate.G0_INTENDED_USE,
        passed=True,
        evidence_ids=(
            profile.claims_matrix.matrix_id,
            profile.use_policy.policy_id,
            bundle.case_manifest.manifest_id,
            *(record.non_claim_id for record in non_claims),
        ),
        reviewer_id=bundle.roles.technical_reviewer_id,
        dossier_id="cardiovascular-release-dossier",
        issued_at=10,
        expires_at=100,
        signer=signers[bundle.roles.technical_reviewer_id],
    )
    changed = CardiovascularQualificationBundle(
        bundle.support_tuple,
        tuple(
            g0 if gate.gate == CardiovascularReleaseGate.G0_INTENDED_USE else gate
            for gate in bundle.gates
        ),
        bundle.artifacts,
        bundle.lifecycle_records,
        non_claims,
        bundle.roles,
        bundle.case_manifest,
        bundle.execution_manifests,
    )
    candidate = evaluate_cardiovascular_release_candidate(
        profile,
        changed,
        trust,
        verifiers,
        at_time=20,
    )
    assert f"invalid-non-claim-signature:{tampered.excluded_use}" in candidate.blockers

    restored = CardiovascularSignedNonClaim.from_record(original.to_record())
    assert restored.non_claim_id == original.non_claim_id
    assert restored.verify(verifiers[restored.signer_id])
    damaged_record = original.to_record()
    damaged_record["statement"] = "Different exclusion."
    with pytest.raises(ValueError, match="invalid content address"):
        CardiovascularSignedNonClaim.from_record(damaged_record)


def test_gate_and_release_decision_signatures_are_verified_not_resigned() -> None:
    profile, bundle, trust, signers, verifiers = _complete_case()
    tampered_gate = replace(bundle.gates[0], signature="00")
    tampered_bundle = replace(
        bundle,
        gates=(tampered_gate, *bundle.gates[1:]),
    )
    candidate = evaluate_cardiovascular_release_candidate(
        profile,
        tampered_bundle,
        trust,
        verifiers,
        at_time=20,
    )
    assert "invalid-gate-signature:g0-intended-use" in candidate.blockers

    valid_candidate = evaluate_cardiovascular_release_candidate(
        profile,
        bundle,
        trust,
        verifiers,
        at_time=20,
    )
    decision = make_cardiovascular_release_decision(
        valid_candidate,
        bundle.roles,
        approved=True,
        signer=signers[bundle.roles.release_approver_id],
        decided_at=21,
        rationale="Authenticated approval.",
    )
    tampered_decision = replace(decision, signature="00")
    assessment = assess_cardiovascular_release(
        valid_candidate,
        tampered_decision,
        trust,
        verifiers,
    )
    assert not assessment.commercial_ready
    assert "release-decision-signature:invalid" in assessment.blockers


def test_lifecycle_completion_and_independent_reviewer_are_required() -> None:
    profile, bundle, trust, signers, verifiers = _complete_case()
    running = RunRecord(
        "qualification-run",
        "different-analysis-plan",
        "different-numeric-revision",
        bundle.execution_manifests[0].manifest_id,
        "running",
    )
    changed_gates = []
    for gate_record in bundle.gates:
        evidence_ids = tuple(
            running.record_id if value == bundle.lifecycle_records[0].record_id else value
            for value in gate_record.release_evidence.evidence_ids
        )
        reviewer = (
            bundle.roles.security_reviewer_id
            if gate_record.gate == CardiovascularReleaseGate.G1_CODE_VERIFICATION
            else gate_record.release_evidence.reviewer_id
        )
        changed_gates.append(
            CardiovascularGateEvidence.issue(
                gate_record.gate,
                passed=True,
                evidence_ids=evidence_ids,
                reviewer_id=reviewer,
                dossier_id=gate_record.dossier_id,
                issued_at=10,
                expires_at=100,
                signer=signers[reviewer],
            )
        )
    changed = CardiovascularQualificationBundle(
        bundle.support_tuple,
        tuple(changed_gates),
        bundle.artifacts,
        (running,),
        bundle.non_claims,
        bundle.roles,
        bundle.case_manifest,
        bundle.execution_manifests,
    )
    candidate = evaluate_cardiovascular_release_candidate(
        profile,
        changed,
        trust,
        verifiers,
        at_time=20,
    )

    assert "independent-reviewer-mismatch:g1-code-verification" in candidate.blockers
    assert f"lifecycle-analysis-plan-mismatch:{running.record_id}" in candidate.blockers
    assert (
        f"lifecycle-numeric-revision-mismatch:{running.record_id}" in candidate.blockers
    )
    for gate in (
        CardiovascularReleaseGate.G1_CODE_VERIFICATION,
        CardiovascularReleaseGate.G2_SOLUTION_VERIFICATION,
        CardiovascularReleaseGate.G3_VALIDATION_UQ,
        CardiovascularReleaseGate.G4_DERIVATIVE_VALIDITY,
    ):
        assert (
            f"missing-exact-run-execution-evidence:{gate.gate_key}" in candidate.blockers
        )


def test_rejected_or_mismatched_release_decision_never_releases() -> None:
    profile, bundle, trust, signers, verifiers = _complete_case()
    candidate = evaluate_cardiovascular_release_candidate(
        profile,
        bundle,
        trust,
        verifiers,
        at_time=20,
    )
    rejected = make_cardiovascular_release_decision(
        candidate,
        bundle.roles,
        approved=False,
        signer=signers[bundle.roles.release_approver_id],
        decided_at=21,
        rationale="Release authorization withheld.",
    )
    assessment = assess_cardiovascular_release(candidate, rejected, trust, verifiers)
    assert not assessment.commercial_ready
    assert assessment.blockers == ("release-decision:not-approved",)
    assert not assessment.capability_profile.released
    with pytest.raises(ValueError, match="not commercial-ready"):
        assessment.require_commercial_ready()

    mismatched_unsigned = CardiovascularReleaseDecision(
        "different-candidate",
        True,
        bundle.roles.release_approver_id,
        21,
        "Not applicable to this candidate.",
        signers[bundle.roles.release_approver_id].signature_algorithm,
        "00",
    )
    mismatched = CardiovascularReleaseDecision(
        mismatched_unsigned.candidate_id,
        mismatched_unsigned.approved,
        mismatched_unsigned.approver_id,
        mismatched_unsigned.decided_at,
        mismatched_unsigned.rationale,
        mismatched_unsigned.signature_algorithm,
        signers[bundle.roles.release_approver_id]
        .sign(mismatched_unsigned.signed_payload)
        .hex(),
    )
    mismatch_assessment = assess_cardiovascular_release(
        candidate, mismatched, trust, verifiers
    )
    assert not mismatch_assessment.commercial_ready
    assert mismatch_assessment.blockers == (
        "release-decision-candidate-mismatch:different-candidate",
    )


def test_release_rechecks_trust_freshness_at_decision_time() -> None:
    profile, bundle, _, signers, verifiers = _complete_case()
    short_trust = HMACSHA256TrustPolicy(
        {"release-index-signer": b"index-key"},
        maximum_index_age=100,
        maximum_evidence_age=5,
    )
    candidate = evaluate_cardiovascular_release_candidate(
        profile,
        bundle,
        short_trust,
        verifiers,
        at_time=14,
    )
    assert candidate.qualified
    decision = make_cardiovascular_release_decision(
        candidate,
        bundle.roles,
        approved=True,
        signer=signers[bundle.roles.release_approver_id],
        decided_at=16,
        rationale="Approval remains subject to release-time trust evaluation.",
    )
    assessment = assess_cardiovascular_release(
        candidate, decision, short_trust, verifiers
    )
    assert not assessment.commercial_ready
    assert len(
        [
            blocker
            for blocker in assessment.blockers
            if blocker.startswith("release-time-rejected-evidence:")
        ]
    ) == len(CardiovascularReleaseGate)


def test_dependency_profiles_must_be_complete_released_and_fresh() -> None:
    dependency_support = SupportTuple(
        "cardiovascular.solver", {"route": "native", "precision": "float64"}
    )
    dependency_evidence = ReleaseGateEvidence(
        "solver-qualified",
        passed=True,
        evidence_ids=("solver-verification-artifact",),
        reviewer_id="dependency-reviewer",
        issued_at=10,
        expires_at=100,
    )
    dependency = CapabilityProfile(
        "cardiovascular.solver-native",
        "phydrax",
        "1",
        (dependency_support,),
        required_gates=("solver-qualified",),
        release_evidence=(dependency_evidence,),
        released=True,
    )
    profile, bundle, trust, signers, verifiers = _complete_case(
        dependency_profile_ids=(dependency.profile_id,)
    )

    missing = evaluate_cardiovascular_release_candidate(
        profile,
        bundle,
        trust,
        verifiers,
        at_time=20,
    )
    assert f"missing-dependency-profile:{dependency.profile_id}" in missing.blockers
    assert not any(
        blocker.startswith("case-support-profile-mismatch")
        for blocker in missing.blockers
    )

    complete = evaluate_cardiovascular_release_candidate(
        profile,
        bundle,
        trust,
        verifiers,
        at_time=20,
        dependency_profiles=(dependency,),
    )
    assert complete.qualified


def _write_supply_chain_record(path: Path, record: dict[str, object]) -> Path:
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_supply_chain_dossier(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    license_lock_sha256: str | None = None,
    license_source: str | None = None,
    omit_license_hash: bool = False,
    attested_wheel_sha256: str | None = None,
) -> dict[str, object]:
    commit = "0123456789abcdef0123456789abcdef01234567"

    def clean_git(command, **_kwargs):
        stdout = f"{commit}\n" if command[1:3] == ["rev-parse", "HEAD"] else ""
        return type(
            "CompletedGitCommand",
            (),
            {"returncode": 0, "stdout": stdout},
        )()

    monkeypatch.setattr(
        "tools.cardiovascular_release_qualification.subprocess.run", clean_git
    )
    alpha_hash = "11" * 32
    beta_hash = "22" * 32
    (root / "uv.lock").write_text(
        "version = 1\n"
        "[[package]]\n"
        'name = "alpha"\n'
        'version = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        'dependencies = [{ name = "beta" }]\n'
        'sdist = { url = "https://example.test/alpha-1.0.tar.gz", '
        f'hash = "sha256:{alpha_hash}" }}\n'
        "[[package]]\n"
        'name = "beta"\n'
        'version = "2.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        'sdist = { url = "https://example.test/beta-2.0.tar.gz", '
        f'hash = "sha256:{beta_hash}" }}\n'
        "[[package]]\n"
        'name = "phydrax"\n'
        'version = "0.2.2"\n'
        'source = { editable = "." }\n'
        'dependencies = [{ name = "alpha" }]\n',
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "phydrax"\nversion = "0.2.2"\n',
        encoding="utf-8",
    )
    (root / "LICENSE").write_text(
        "Synthetic test-only commercial terms.\n", encoding="utf-8"
    )
    (root / "NOTICE").write_text(
        "SING\nSee LICENSES/SING-MIT.txt.\nASDEX\nSee LICENSES/ASDEX-MIT.txt.\n",
        encoding="utf-8",
    )
    licenses = root / "LICENSES"
    licenses.mkdir()
    (licenses / "SING-MIT.txt").write_text("Synthetic SING notice.\n", encoding="utf-8")
    (licenses / "ASDEX-MIT.txt").write_text("Synthetic ASDEX notice.\n", encoding="utf-8")
    wheel = root / "phydrax-0.2.2-py3-none-any.whl"
    sdist = root / "phydrax-0.2.2.tar.gz"
    container = root / "phydrax-0.2.2.oci.tar"
    wheel.write_bytes(b"synthetic wheel")
    sdist.write_bytes(b"synthetic sdist")
    container.write_bytes(b"synthetic OCI input")
    distributions = {"wheel": wheel, "sdist": sdist, "container": container}
    distribution_hashes = {
        kind: _file_sha256(path) for kind, path in distributions.items()
    }
    lock_sha256 = _file_sha256(root / "uv.lock")
    signature = "ab" * 32

    def bound(kind: str, **fields: object) -> dict[str, object]:
        return {
            "kind": kind,
            "source_commit": commit,
            "lock_sha256": lock_sha256,
            "signer_id": "synthetic-external-signer",
            "signature_algorithm": "synthetic-test-signature",
            "signature": signature,
            **fields,
        }

    commercial_license = _write_supply_chain_record(
        root / "commercial-license.json",
        bound(
            "cardiovascular-commercial-license-authorization",
            authorization_status="authorized",
        ),
    )
    data_rights = _write_supply_chain_record(
        root / "data-rights.json",
        bound(
            "cardiovascular-data-rights-determination",
            rights_status="authorized",
        ),
    )
    signer = _write_supply_chain_record(
        root / "signer.json",
        bound("cardiovascular-release-signer", signer_status="active"),
    )
    root_hashes = sorted(distribution_hashes.values())
    license_packages: list[dict[str, object]] = [
        {
            "name": "alpha",
            "version": "1.0",
            "hashes": [] if omit_license_hash else [f"sha256:{alpha_hash}"],
            "license_concluded": "Apache-2.0",
            "license_declared": "Apache-2.0",
            "copyright_text": "Copyright Synthetic Alpha",
        },
        {
            "name": "beta",
            "version": "2.0",
            "hashes": [f"sha256:{beta_hash}"],
            "license_concluded": "MIT",
            "license_declared": "MIT",
            "copyright_text": "Copyright Synthetic Beta",
        },
        {
            "name": "phydrax",
            "version": "0.2.2",
            "hashes": [f"sha256:{digest}" for digest in root_hashes],
            "license_concluded": "LicenseRef-Synthetic-Commercial-Test",
            "license_declared": "LicenseRef-Synthetic-Commercial-Test",
            "copyright_text": "Copyright Synthetic PhydraX",
        },
    ]
    if license_source is not None:
        license_packages[0]["source"] = license_source
    license_scan = bound(
        "cardiovascular-license-scan",
        scan_status="passed",
        scanner={"name": "synthetic-license-scanner", "version": "1"},
        packages=license_packages,
    )
    if license_lock_sha256 is not None:
        license_scan["lock_sha256"] = license_lock_sha256
    license_report = _write_supply_chain_record(root / "license-scan.json", license_scan)
    vulnerability_report = _write_supply_chain_record(
        root / "vulnerability-scan.json",
        bound(
            "cardiovascular-vulnerability-scan",
            scan_status="passed",
            scanner={"name": "synthetic-vulnerability-scanner", "version": "1"},
            packages=[
                {
                    "name": name,
                    "version": version,
                    "hashes": [f"sha256:{digest}" for digest in hashes],
                    "status": "passed",
                    "vulnerabilities": [],
                }
                for name, version, hashes in (
                    ("alpha", "1.0", [alpha_hash]),
                    ("beta", "2.0", [beta_hash]),
                    ("phydrax", "0.2.2", root_hashes),
                )
            ],
        ),
    )
    attestation_subject_hashes = {
        "dependency-lock": lock_sha256,
        "commercial-license": _file_sha256(commercial_license),
        "data-rights": _file_sha256(data_rights),
        "signer": _file_sha256(signer),
        "license-report": _file_sha256(license_report),
        "vulnerability-report": _file_sha256(vulnerability_report),
        **{
            f"distribution:{kind}": digest for kind, digest in distribution_hashes.items()
        },
    }
    if attested_wheel_sha256 is not None:
        attestation_subject_hashes["distribution:wheel"] = attested_wheel_sha256
    attestation = _write_supply_chain_record(
        root / "supply-chain-attestation.json",
        bound(
            "cardiovascular-supply-chain-attestation",
            attestation_status="verified",
            subjects=[
                {"name": name, "sha256": digest}
                for name, digest in sorted(attestation_subject_hashes.items())
            ],
        ),
    )
    verifier_subject_paths = {
        "commercial-license": commercial_license,
        "data-rights": data_rights,
        "signer": signer,
        "license-report": license_report,
        "vulnerability-report": vulnerability_report,
        "supply-chain-attestation": attestation,
    }
    verifier = _write_supply_chain_record(
        root / "verifier.json",
        bound(
            "cardiovascular-signature-verification",
            verification_status="verified",
            subjects=[
                {"name": name, "sha256": _file_sha256(path)}
                for name, path in sorted(verifier_subject_paths.items())
            ],
        ),
    )
    return {
        "commercial_license_record": commercial_license,
        "data_rights_record": data_rights,
        "signer_record": signer,
        "verifier_record": verifier,
        "vulnerability_report": vulnerability_report,
        "license_report": license_report,
        "supply_chain_attestation": attestation,
        "distribution_artifacts": distributions,
    }


def test_release_artifact_builder_emits_dependency_complete_g5_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_supply_chain_dossier(tmp_path, monkeypatch)

    report = build_cardiovascular_release_artifacts(
        tmp_path,
        tmp_path / "derived-release",
        **inputs,
    )

    assert report["blockers"] == []
    assert report["g5_evidence_ready"]
    assert not report["commercial_ready"]
    assert not report["grants_commercial_license"]
    assert set(report["distribution_artifacts"]) == {"wheel", "sdist", "container"}
    for record in report["distribution_artifacts"].values():
        assert record["sha256"] == _file_sha256(Path(record["path"]))

    spdx_path = Path(report["generated"]["sbom-spdx"]["path"])
    spdx = json.loads(spdx_path.read_text(encoding="utf-8"))
    assert "NOASSERTION" not in json.dumps(spdx)
    assert {package["name"] for package in spdx["packages"]} == {
        "alpha",
        "beta",
        "phydrax",
    }
    package_ids = {package["name"]: package["SPDXID"] for package in spdx["packages"]}
    relationships = {
        (
            relationship["spdxElementId"],
            relationship["relationshipType"],
            relationship["relatedSpdxElement"],
        )
        for relationship in spdx["relationships"]
    }
    assert (
        package_ids["phydrax"],
        "DEPENDS_ON",
        package_ids["alpha"],
    ) in relationships
    assert (
        package_ids["alpha"],
        "DEPENDS_ON",
        package_ids["beta"],
    ) in relationships
    assert (
        "SPDXRef-DOCUMENT",
        "DESCRIBES",
        package_ids["phydrax"],
    ) in relationships
    assert all(package["checksums"] for package in spdx["packages"])

    cyclonedx_path = Path(report["generated"]["sbom-cyclonedx"]["path"])
    cyclonedx = json.loads(cyclonedx_path.read_text(encoding="utf-8"))
    assert cyclonedx["metadata"]["component"]["name"] == "phydrax"
    component_refs = {
        component["name"]: component["bom-ref"]
        for component in [
            cyclonedx["metadata"]["component"],
            *cyclonedx["components"],
        ]
    }
    dependency_graph = {
        dependency["ref"]: set(dependency["dependsOn"])
        for dependency in cyclonedx["dependencies"]
    }
    assert component_refs["alpha"] in dependency_graph[component_refs["phydrax"]]
    assert component_refs["beta"] in dependency_graph[component_refs["alpha"]]
    assert all(
        component["hashes"] and component["licenses"]
        for component in [
            cyclonedx["metadata"]["component"],
            *cyclonedx["components"],
        ]
    )

    manifest_path = Path(report["generated"]["supply-chain-evidence-manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["g5_evidence_ready"]
    assert manifest["dependency_graph_complete"]
    assert set(manifest["external_records"]) == {
        "commercial-license",
        "data-rights",
        "signer",
        "verifier",
        "vulnerability-report",
        "license-report",
        "supply-chain-attestation",
    }


def test_release_artifact_builder_rejects_mismatched_bound_supply_chain_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_supply_chain_dossier(
        tmp_path,
        monkeypatch,
        license_lock_sha256="ff" * 32,
        license_source="https://unrelated.example/simple",
        omit_license_hash=True,
        attested_wheel_sha256="ee" * 32,
    )

    report = build_cardiovascular_release_artifacts(
        tmp_path,
        tmp_path / "derived-release",
        **inputs,
    )

    assert not report["g5_evidence_ready"]
    assert "external-license-report-record:lock-sha256-mismatch" in report["blockers"]
    assert (
        "external-license-report-record:package-source-mismatch:alpha@1.0"
        in report["blockers"]
    )
    assert (
        f"external-license-report-record:package-hash-missing:alpha@1.0:{'11' * 32}"
        in report["blockers"]
    )
    assert (
        "external-supply-chain-attestation-record:"
        "subject-hash-mismatch:distribution:wheel" in report["blockers"]
    )


def test_repository_preflight_refuses_nonproduction_license_and_missing_dossier(
    tmp_path,
) -> None:
    (tmp_path / "LICENSE").write_text(
        "PHYDRA NON-PRODUCTION LICENSE\nCommercial Use requires a separate license\n",
        encoding="utf-8",
    )
    (tmp_path / "NOTICE").write_text("Release notice.\\n", encoding="utf-8")

    report = cardiovascular_release_preflight(tmp_path)

    assert not report["preflight_passed"]
    assert not report["commercial_ready"]
    assert not report["grants_commercial_license"]
    assert not report["regulated_device_claim"]
    assert report["release_decision"]["separate_from_qualification"]
    assert report["preflight_blockers"][:3] == [
        "commercial-license-grant-absent:repository-license-is-non-production-only",
        "artifact-sbom:missing",
        "artifact-build-provenance:missing",
    ]
    assert tuple(report["gates"]) == tuple(
        gate.gate_key for gate in CardiovascularReleaseGate
    )


def test_release_artifact_builder_derives_unsigned_records_and_keeps_authority_external(
    tmp_path,
) -> None:
    (tmp_path / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "asdex"\nversion = "0.5.1"\n',
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "phydrax"\nversion = "0.2.2"\n',
        encoding="utf-8",
    )
    (tmp_path / "LICENSE").write_text("PHYDRA NON-PRODUCTION LICENSE\n", encoding="utf-8")
    (tmp_path / "NOTICE").write_text(
        "SING\nSee LICENSES/SING-MIT.txt.\nASDEX\nSee LICENSES/ASDEX-MIT.txt.\n",
        encoding="utf-8",
    )
    (tmp_path / "LICENSES").mkdir()

    report = build_cardiovascular_release_artifacts(
        tmp_path, tmp_path / "derived-release"
    )

    assert not report["commercial_ready"]
    assert not report["grants_commercial_license"]
    assert not report["g5_evidence_ready"]
    assert {
        "sbom-spdx",
        "sbom-cyclonedx",
        "build-provenance",
        "notice-audit",
        "supply-chain-evidence-manifest",
        "unsigned-gate-dossier",
        "artifact-hashes",
    } == set(report["generated"])
    for record in report["generated"].values():
        assert Path(record["path"]).is_file()
        assert len(record["sha256"]) == 64
    assert "notice-license-text-missing:SING-MIT.txt" in report["blockers"]
    assert "notice-license-text-missing:ASDEX-MIT.txt" in report["blockers"]
    assert (
        "commercial-license-grant-absent:repository-license-is-pnpl" in report["blockers"]
    )
    assert "external-commercial-license-record:missing" in report["blockers"]
    assert "external-data-rights-record:missing" in report["blockers"]
    assert "external-signer-record:missing" in report["blockers"]
    assert "external-verifier-record:missing" in report["blockers"]
    assert "external-vulnerability-report-record:missing" in report["blockers"]
    assert "external-license-report-record:missing" in report["blockers"]
    assert "external-supply-chain-attestation-record:missing" in report["blockers"]
    assert "distribution-artifact:missing" in report["blockers"]
    assert "dependency-metadata:asdex@0.5.1:source-unresolved" in report["blockers"]
    assert "dependency-metadata:asdex@0.5.1:hash-unresolved" in report["blockers"]
    assert (
        "dependency-metadata:asdex@0.5.1:license-concluded-unresolved"
        in report["blockers"]
    )
    spdx = json.loads(
        Path(report["generated"]["sbom-spdx"]["path"]).read_text(encoding="utf-8")
    )
    assert "NOASSERTION" in json.dumps(spdx)


def test_review_roles_must_be_independent() -> None:
    with pytest.raises(ValueError, match="roles must be independent"):
        CardiovascularReviewRoles(
            "author",
            "same-reviewer",
            "same-reviewer",
            "security",
            "approver",
        )
