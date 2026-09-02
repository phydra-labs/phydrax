#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import pytest

from phydrax.qualification._evidence import (
    ForecastResourceRecord,
    ObservedResourceRecord,
    QualificationEvidence,
    QualificationMatrix,
    SupportDependency,
)
from phydrax.qualification._reference import ReferenceArtifactManifest
from phydrax.qualification._registry import (
    CapabilityProfile,
    HMACSHA256ReleaseSigner,
    HMACSHA256TrustPolicy,
    ReleaseIndex,
    require_profile,
    SupportTuple,
)


def _evidence(
    evidence_kind: str,
    outcome: str = "passed",
    *,
    expires_at: int = 100,
    reason: str = "criteria-evaluated",
    supersedes_evidence_ids: tuple[str, ...] = (),
) -> QualificationEvidence:
    return QualificationEvidence(
        evidence_kind,
        outcome,
        ("subject",),
        build_id="build",
        environment_id="environment",
        backend="cpu",
        topology="single-rank",
        precision="float64",
        reduction="pairwise",
        replay_id="replay",
        criteria_ids=("criterion",),
        raw_artifact_ids=("raw",),
        reviewer_id="reviewer",
        issued_at=1,
        expires_at=expires_at,
        reason=reason,
        supersedes_evidence_ids=supersedes_evidence_ids,
        requalification_triggers=("build-change",),
    )


def _matrix() -> QualificationMatrix:
    return QualificationMatrix(
        {
            "scientific-gate": {
                "evidence_kind": "scientific",
                "subject_id": "subject",
                "build_id": "build",
                "environment_id": "environment",
                "backend": "cpu",
                "topology": "single-rank",
                "precision": "float64",
                "reduction": "pairwise",
                "replay_id": "replay",
                "criterion_id": "criterion",
            }
        }
    )


def _trust() -> tuple[HMACSHA256ReleaseSigner, HMACSHA256TrustPolicy]:
    signer = HMACSHA256ReleaseSigner("qualification-signer", b"secret")
    policy = HMACSHA256TrustPolicy(
        {"qualification-signer": b"secret"},
        maximum_index_age=100,
        maximum_evidence_age=100,
    )
    return signer, policy


def test_exact_support_dependency_mismatch_is_not_admitted():
    dependency_support = SupportTuple("dependency.core", {"backend": "cpu"})
    wrong_support = SupportTuple("dependency.core", {"backend": "gpu"})
    dependency_profile = CapabilityProfile(
        "dependency.core",
        "phydrax",
        "release",
        (dependency_support,),
        released=True,
    )
    requested_support = SupportTuple("application.solver", {"mode": "dns"})
    dependent_profile = CapabilityProfile(
        "application.solver",
        "phydrax",
        "release",
        (requested_support,),
        dependencies=(
            SupportDependency(
                dependency_profile.profile_id, wrong_support.support_tuple_id
            ),
        ),
        released=True,
    )
    signer, policy = _trust()
    index = ReleaseIndex.sign(
        (dependency_profile, dependent_profile), signer, issued_at=1
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"dependency:{dependency_profile.profile_id}:"
            rf"unsupported-tuple:{wrong_support.support_tuple_id}"
        ),
    ):
        require_profile(
            index,
            dependent_profile.profile_id,
            requested_support,
            policy,
            at_time=2,
        )

    admitted_profile = CapabilityProfile(
        "application.solver",
        "phydrax",
        "release-exact",
        (requested_support,),
        dependencies=(
            SupportDependency(
                dependency_profile.profile_id,
                dependency_support.support_tuple_id,
            ),
        ),
        released=True,
    )
    admitted_index = ReleaseIndex.sign(
        (dependency_profile, admitted_profile), signer, issued_at=1
    )
    assert (
        require_profile(
            admitted_index,
            admitted_profile.profile_id,
            requested_support,
            policy,
            at_time=2,
        ).profile_id
        == admitted_profile.profile_id
    )


def test_performance_evidence_cannot_satisfy_scientific_predicate():
    report = _matrix().evaluate((_evidence("performance"),), at_time=2)

    assert report.outcome == "inconclusive"
    assert report.inconclusive_predicate_ids == ("scientific-gate",)
    assert report.gaps == (("scientific-gate", "inconclusive", ("missing-evidence",)),)


def test_all_evidence_kinds_have_isolated_identities():
    kinds = (
        "unit",
        "smoke",
        "performance",
        "scientific",
        "reference",
        "operational",
        "security",
    )
    evidence = tuple(_evidence(kind) for kind in kinds)

    assert tuple(item.evidence_kind for item in evidence) == kinds
    assert len({item.evidence_id for item in evidence}) == len(kinds)


def test_expiry_and_supersession_do_not_resurrect_passing_evidence():
    matrix = _matrix()
    expired = _evidence("scientific", expires_at=2)
    expired_report = matrix.evaluate((expired,), at_time=3)
    replacement = _evidence(
        "scientific",
        "failed",
        reason="reference-disagreement",
        supersedes_evidence_ids=(expired.evidence_id,),
    )
    replaced_report = matrix.evaluate((expired, replacement), at_time=3)

    assert expired_report.outcome == "inconclusive"
    assert expired_report.gaps[0][2] == (f"expired-evidence:{expired.evidence_id}",)
    assert replaced_report.outcome == "failed"
    assert replaced_report.failed_predicate_ids == ("scientific-gate",)
    assert replaced_report.gaps[0][2] == (
        f"failed:{replacement.evidence_id}:reference-disagreement",
    )


def test_evidence_matrix_and_dependency_identities_are_order_independent():
    first = QualificationEvidence(
        "scientific",
        "passed",
        ("subject-b", "subject-a"),
        build_id="build",
        environment_id="environment",
        backend="cpu",
        topology="single-rank",
        precision="float64",
        reduction="pairwise",
        replay_id="replay",
        criteria_ids=("criterion-b", "criterion-a"),
        raw_artifact_ids=("raw-b", "raw-a"),
        reviewer_id="reviewer",
        issued_at=1,
        expires_at=100,
        reason="criteria-evaluated",
        requalification_triggers=("environment-change", "build-change"),
    )
    second = QualificationEvidence(
        "scientific",
        "passed",
        ("subject-a", "subject-b"),
        build_id="build",
        environment_id="environment",
        backend="cpu",
        topology="single-rank",
        precision="float64",
        reduction="pairwise",
        replay_id="replay",
        criteria_ids=("criterion-a", "criterion-b"),
        raw_artifact_ids=("raw-a", "raw-b"),
        reviewer_id="reviewer",
        issued_at=1,
        expires_at=100,
        reason="criteria-evaluated",
        requalification_triggers=("build-change", "environment-change"),
    )
    matrix_a = QualificationMatrix(
        {
            "b": {"subject_id": "subject-b", "evidence_kind": "scientific"},
            "a": {"criterion_id": "criterion-a", "evidence_kind": "scientific"},
        }
    )
    matrix_b = QualificationMatrix(
        {
            "a": {"evidence_kind": "scientific", "criterion_id": "criterion-a"},
            "b": {"evidence_kind": "scientific", "subject_id": "subject-b"},
        }
    )

    dependency_a = SupportDependency("profile", "tuple")
    dependency_b = SupportDependency("profile", "tuple")

    assert first.evidence_id == second.evidence_id
    assert matrix_a.matrix_id == matrix_b.matrix_id
    assert dependency_a.dependency_id == dependency_b.dependency_id
    assert (
        matrix_a.evaluate((first,), at_time=2).report_id
        == matrix_b.evaluate((second,), at_time=2).report_id
    )


def test_resource_observations_and_forecasts_are_exact_and_content_addressed():
    observed_a = ObservedResourceRecord(
        "subject",
        "build",
        "environment",
        backend="cpu",
        topology="two-rank",
        measurements={"wall-seconds": 2.0, "peak-memory-bytes": 4096},
        observed_at=5,
        raw_artifact_ids=("stderr", "scheduler-record"),
    )
    observed_b = ObservedResourceRecord(
        "subject",
        "build",
        "environment",
        backend="cpu",
        topology="two-rank",
        measurements={"peak-memory-bytes": 4096, "wall-seconds": 2.0},
        observed_at=5,
        raw_artifact_ids=("scheduler-record", "stderr"),
    )
    forecast = ForecastResourceRecord(
        "subject",
        "build",
        "environment",
        backend="cpu",
        topology="four-rank",
        estimates={"wall-seconds": 1.2, "peak-memory-bytes": 8192},
        uncertainty_bounds={
            "wall-seconds": (1.0, 1.5),
            "peak-memory-bytes": (8000, 8400),
        },
        forecast_model_id="strong-scaling-fit",
        source_record_ids=(observed_a.record_id,),
        issued_at=6,
        expires_at=20,
    )

    assert observed_a.record_id == observed_b.record_id
    assert forecast.is_current(10)
    assert ForecastResourceRecord.from_record(forecast.to_record()).record_id == (
        forecast.record_id
    )


def test_reference_manifest_refuses_unlicensed_requested_rights():
    manifest = ReferenceArtifactManifest(
        "reference-case",
        checksum_algorithm="sha256",
        checksum="ab" * 32,
        size_bytes=4096,
        license_id="research-only",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unclassified",
        nondimensionalization={"length": 1.0, "velocity": 2.0},
        uncertainty={"pressure-l2": 0.01},
        lineage_ids=("source-publication", "digitization-run"),
    )
    assert ReferenceArtifactManifest.from_record(manifest.to_record()).manifest_id == (
        manifest.manifest_id
    )

    with pytest.raises(
        PermissionError,
        match="commercial-use-not-permitted; redistribution-not-permitted",
    ):
        manifest.require_rights(commercial_use=True, redistribution=True)


def test_matrix_preserves_failed_and_inconclusive_gaps_deterministically():
    matrix = QualificationMatrix(
        {
            "reference-gate": {
                "evidence_kind": "reference",
                "criterion_id": "criterion",
            },
            "scientific-gate": {
                "evidence_kind": "scientific",
                "criterion_id": "criterion",
            },
            "security-gate": {
                "evidence_kind": "security",
                "criterion_id": "criterion",
            },
        }
    )
    failed = _evidence("scientific", "failed", reason="residual-too-large")
    inconclusive = _evidence("security", "inconclusive", reason="scan-interrupted")

    first = matrix.evaluate((inconclusive, failed), at_time=2)
    second = matrix.evaluate((failed, inconclusive), at_time=2)

    assert first.report_id == second.report_id
    assert first.outcome == "failed"
    assert first.failed_predicate_ids == ("scientific-gate",)
    assert first.inconclusive_predicate_ids == (
        "reference-gate",
        "security-gate",
    )
    assert tuple(gap[0] for gap in first.gaps) == (
        "reference-gate",
        "scientific-gate",
        "security-gate",
    )


def test_candidate_profiles_keep_legacy_profile_dependencies_and_round_trip():
    support = SupportTuple(
        "candidate.profile",
        {"backend": "cpu", "precision": "float64"},
    )
    candidate = CapabilityProfile(
        "candidate.profile",
        "phydrax",
        "pr-246-candidate",
        (support,),
        dependencies=("legacy-profile-id",),
        released=False,
    )

    restored = CapabilityProfile.from_record(candidate.to_record())

    assert restored.profile_id == candidate.profile_id
    assert restored.dependencies == ("legacy-profile-id",)
    assert not restored.released
