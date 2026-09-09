#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import pytest

from phydrax.qualification import (
    QualificationEvidence,
    ScientificClaimProfile,
    ScientificMetricCriterion,
    SupportTuple,
)


_CAMPAIGN_ID = "campaign-held-out-families"
_TRIGGERS = (
    "model-parameters-change",
    "observation-law-change",
    "operating-domain-change",
    "preprocessing-change",
    "source-artifact-change",
)


def _criterion(maximum: float = 1.0) -> ScientificMetricCriterion:
    return ScientificMetricCriterion(
        "family-macro-mae",
        "at_most",
        None,
        maximum,
        "kcal/mol",
        "independent_unit_macro",
    )


def _profile(
    *,
    frozen_criteria_ids: tuple[str, ...] | None = None,
) -> ScientificClaimProfile:
    criterion = _criterion()
    return ScientificClaimProfile(
        "protein.stability",
        SupportTuple(
            "protein.stability",
            {"population": "small-natural-monomers", "temperature-k": 298},
        ),
        ("mutation-delta-g",),
        ("assay-buffer-298K",),
        _CAMPAIGN_ID,
        ("measurement-calibration", "locked-prediction"),
        (criterion,),
        "abstain-outside-admitted-family-and-condition",
        _TRIGGERS,
        frozen_criteria_ids=(
            (criterion.criterion_id,)
            if frozen_criteria_ids is None
            else frozen_criteria_ids
        ),
    )


def _stage(
    stage_id: str,
    outcome: str = "passed",
    *,
    raw_artifact_id: str | None = None,
    issued_at: int = 1,
    expires_at: int = 100,
) -> QualificationEvidence:
    return QualificationEvidence(
        "scientific",
        outcome,
        (_CAMPAIGN_ID,),
        build_id="build",
        environment_id="environment",
        backend="cpu",
        topology="single-rank",
        precision="float64",
        reduction="pairwise",
        replay_id=f"replay-{stage_id}",
        criteria_ids=(stage_id,),
        raw_artifact_ids=(
            f"artifact-{stage_id}" if raw_artifact_id is None else raw_artifact_id,
        ),
        reviewer_id="stage-reviewer",
        issued_at=issued_at,
        expires_at=expires_at,
        reason=f"{stage_id}-{outcome}",
    )


def _evaluate(
    profile: ScientificClaimProfile,
    stages: tuple[QualificationEvidence, ...],
    *,
    value: float = 0.8,
    unit: str = "kcal/mol",
    aggregation: str = "independent_unit_macro",
    issued_at: int = 2,
    expires_at: int = 50,
) -> QualificationEvidence:
    return profile.evaluate(
        {"family-macro-mae": value},
        stages,
        metric_units={"family-macro-mae": unit},
        metric_aggregations={"family-macro-mae": aggregation},
        build_id="claim-build",
        environment_id="claim-environment",
        backend="cpu",
        topology="single-rank",
        precision="float64",
        reduction="pairwise",
        replay_id="claim-replay",
        raw_artifact_ids=("locked-metrics",),
        reviewer_id="claim-reviewer",
        issued_at=issued_at,
        expires_at=expires_at,
    )


def test_metric_pass_cannot_replace_missing_measurement_calibration():
    evidence = _evaluate(_profile(), (_stage("locked-prediction"),))

    assert evidence.inconclusive
    assert "measurement-calibration" in evidence.reason


def test_failed_required_stage_fails_but_absent_stage_is_inconclusive():
    profile = _profile()
    failed = _evaluate(
        profile,
        (
            _stage("measurement-calibration", "failed"),
            _stage("locked-prediction"),
        ),
    )
    absent = _evaluate(profile, (_stage("measurement-calibration"),))

    assert failed.failed
    assert "measurement-calibration" in failed.reason
    assert absent.inconclusive
    assert "locked-prediction" in absent.reason


def test_metric_threshold_uses_declared_unit_and_aggregation():
    profile = _profile()
    stages = (
        _stage("measurement-calibration"),
        _stage("locked-prediction"),
    )

    assert _evaluate(profile, stages, value=1.0).passed
    assert _evaluate(profile, stages, value=1.01).failed
    assert _evaluate(profile, stages, unit="kJ/mol").inconclusive


def test_family_macro_criterion_rejects_pooled_value_with_same_metric_id():
    evidence = _evaluate(
        _profile(),
        (
            _stage("measurement-calibration"),
            _stage("locked-prediction"),
        ),
        aggregation="pooled",
    )

    assert evidence.inconclusive
    assert "family-macro-mae" in evidence.reason


def test_claim_evidence_preserves_all_requalification_triggers():
    profile = _profile()
    evidence = _evaluate(
        profile,
        (
            _stage("measurement-calibration"),
            _stage("locked-prediction"),
        ),
    )

    assert evidence.passed
    assert evidence.requalification_triggers == tuple(sorted(_TRIGGERS))
    assert profile.claim_id in evidence.subject_ids
    assert profile.support.support_tuple_id in evidence.subject_ids


def test_claim_evidence_identity_changes_when_prerequisite_is_replaced():
    profile = _profile()
    calibration_v1 = _stage(
        "measurement-calibration",
        raw_artifact_id="calibration-v1",
    )
    calibration_v2 = _stage(
        "measurement-calibration",
        raw_artifact_id="calibration-v2",
    )
    locked = _stage("locked-prediction")

    first = _evaluate(profile, (calibration_v1, locked))
    replaced = _evaluate(profile, (calibration_v2, locked))

    assert calibration_v1.evidence_id in first.subject_ids
    assert calibration_v2.evidence_id in replaced.subject_ids
    assert first.evidence_id != replaced.evidence_id


def test_claim_evidence_identity_changes_with_passing_metric_value():
    profile = _profile()
    stages = (
        _stage("measurement-calibration"),
        _stage("locked-prediction"),
    )

    first = _evaluate(profile, stages, value=0.7)
    substituted = _evaluate(profile, stages, value=0.8)

    assert first.passed
    assert substituted.passed
    assert first.evidence_id != substituted.evidence_id


def test_claim_expiry_is_capped_by_earliest_matched_prerequisite():
    evidence = _evaluate(
        _profile(),
        (
            _stage("measurement-calibration", expires_at=30),
            _stage("locked-prediction", expires_at=40),
        ),
        expires_at=50,
    )

    assert evidence.passed
    assert evidence.expires_at == 30


def test_claim_refuses_prerequisite_expiring_at_derived_issuance():
    with pytest.raises(ValueError, match="remain current after derived issuance"):
        _evaluate(
            _profile(),
            (
                _stage("measurement-calibration", expires_at=2),
                _stage("locked-prediction"),
            ),
            issued_at=2,
        )


def test_claim_identity_is_order_independent_and_content_verified():
    profile = _profile()
    reconstructed = ScientificClaimProfile.from_record(profile.to_record())

    assert reconstructed.claim_id == profile.claim_id
    assert reconstructed.to_record() == profile.to_record()

    corrupted = profile.to_record()
    corrupted["claim_id"] = "not-the-content-address"
    with pytest.raises(ValueError, match="invalid content address"):
        ScientificClaimProfile.from_record(corrupted)


def test_metric_criterion_identity_binds_exact_threshold_and_is_verified():
    criterion = _criterion()
    reconstructed = ScientificMetricCriterion.from_record(criterion.to_record())
    changed_threshold = _criterion(0.9)

    assert reconstructed == criterion
    assert changed_threshold.criterion_id != criterion.criterion_id

    corrupted = criterion.to_record()
    corrupted["criterion_id"] = "not-the-content-address"
    with pytest.raises(ValueError, match="invalid content address"):
        ScientificMetricCriterion.from_record(corrupted)


def test_claim_rejects_criterion_not_frozen_by_campaign():
    with pytest.raises(ValueError, match="frozen in campaign criteria_ids"):
        _profile(frozen_criteria_ids=("different-criterion",))


def test_metric_criterion_rejects_ambiguous_bounds():
    with pytest.raises(ValueError, match="requires only an upper"):
        ScientificMetricCriterion("mae", "at_most", 0.0, 1.0, "kcal/mol", "pooled")
