#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import pytest

from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    QualificationCriterion,
    QualificationEvidence,
    QualificationMatrix,
    SupportTuple,
    validate_qualification_causality,
)


def _support_tuple_id(*, chemistry: str = "nmc") -> str:
    return SupportTuple(
        "battery.cycling",
        {
            "chemistry": chemistry,
            "protocol": "constant-current",
            "cells": 4,
        },
    ).support_tuple_id


def _criterion(
    *,
    issued_at: int = 1,
    valid_until: int | None = 10,
) -> QualificationCriterion:
    return QualificationCriterion(
        support_tuple_id=_support_tuple_id(),
        metric="terminal-voltage-rmse",
        unit="V",
        comparison="less-than-or-equal",
        target=0.02,
        aggregation="root-mean-square",
        uncertainty="upper-95-percent-confidence-bound",
        applicability="held-out-constant-current-cycle",
        approval_id="approval:test-lab:2026-01",
        issued_at=issued_at,
        valid_until=valid_until,
    )


def _start(
    criterion: QualificationCriterion,
    *,
    campaign_spec_id: str = "campaign-spec",
    criterion_id: str | None = None,
    resolved_run_spec_id: str = "resolved-run-spec",
    support_tuple_id: str | None = None,
    started_at: int = 2,
) -> CampaignStartRecord:
    return CampaignStartRecord(
        campaign_spec_id=campaign_spec_id,
        criterion_id=(criterion.criterion_id if criterion_id is None else criterion_id),
        resolved_run_spec_id=resolved_run_spec_id,
        support_tuple_id=(
            criterion.support_tuple_id if support_tuple_id is None else support_tuple_id
        ),
        started_at=started_at,
    )


def _observation(
    criterion: QualificationCriterion,
    start: CampaignStartRecord,
    *,
    start_record_id: str | None = None,
    campaign_spec_id: str | None = None,
    criterion_id: str | None = None,
    resolved_run_spec_id: str | None = None,
    support_tuple_id: str | None = None,
    raw_artifact_ids: tuple[str, ...] = ("raw-current", "raw-voltage"),
    observed_at: int = 3,
) -> CampaignObservationRecord:
    return CampaignObservationRecord(
        start_record_id=(
            start.start_record_id if start_record_id is None else start_record_id
        ),
        campaign_spec_id=(
            start.campaign_spec_id if campaign_spec_id is None else campaign_spec_id
        ),
        criterion_id=(criterion.criterion_id if criterion_id is None else criterion_id),
        resolved_run_spec_id=(
            start.resolved_run_spec_id
            if resolved_run_spec_id is None
            else resolved_run_spec_id
        ),
        support_tuple_id=(
            start.support_tuple_id if support_tuple_id is None else support_tuple_id
        ),
        raw_artifact_ids=raw_artifact_ids,
        observed_at=observed_at,
    )


def _evidence(
    criterion: QualificationCriterion,
    start: CampaignStartRecord,
    observation: CampaignObservationRecord,
    *,
    outcome: str = "passed",
    criteria_ids: tuple[str, ...] | None = None,
    raw_artifact_ids: tuple[str, ...] | None = None,
    campaign_start_record_ids: tuple[str, ...] | None = None,
    campaign_observation_record_ids: tuple[str, ...] | None = None,
    issued_at: int = 4,
    expires_at: int = 20,
    supersedes_evidence_ids: tuple[str, ...] = (),
) -> QualificationEvidence:
    return QualificationEvidence(
        "scientific",
        outcome,
        ("cell-specimen", "solver-output"),
        build_id="build",
        environment_id="laboratory",
        backend="cpu",
        topology="single-process",
        precision="float64",
        reduction="pairwise",
        replay_id="replay",
        criteria_ids=(
            (criterion.criterion_id, "secondary-criterion")
            if criteria_ids is None
            else criteria_ids
        ),
        raw_artifact_ids=(
            observation.raw_artifact_ids if raw_artifact_ids is None else raw_artifact_ids
        ),
        campaign_start_record_ids=(
            (start.start_record_id,)
            if campaign_start_record_ids is None
            else campaign_start_record_ids
        ),
        campaign_observation_record_ids=(
            (observation.observation_record_id,)
            if campaign_observation_record_ids is None
            else campaign_observation_record_ids
        ),
        reviewer_id="reviewer",
        issued_at=issued_at,
        expires_at=expires_at,
        reason="synthetic campaign evidence",
        supersedes_evidence_ids=supersedes_evidence_ids,
    )


def _linked_records():
    criterion = _criterion()
    start = _start(criterion)
    observation = _observation(criterion, start)
    evidence = _evidence(criterion, start, observation)
    return criterion, start, observation, evidence


def test_content_addresses_are_deterministic_and_versionless():
    criterion_a = _criterion()
    criterion_b = _criterion()
    start_a = _start(criterion_a)
    start_b = _start(criterion_b)
    observation_a = _observation(
        criterion_a,
        start_a,
        raw_artifact_ids=("raw-voltage", "raw-current"),
    )
    observation_b = _observation(criterion_b, start_b)

    assert criterion_a.criterion_id == criterion_b.criterion_id
    assert start_a.start_record_id == start_b.start_record_id
    assert observation_a.observation_record_id == observation_b.observation_record_id
    assert (
        QualificationCriterion.from_record(criterion_a.to_record()).criterion_id
        == criterion_a.criterion_id
    )
    assert (
        CampaignStartRecord.from_record(start_a.to_record()).start_record_id
        == start_a.start_record_id
    )
    assert (
        CampaignObservationRecord.from_record(
            observation_a.to_record()
        ).observation_record_id
        == observation_a.observation_record_id
    )
    assert all(
        "schema_version" not in record
        for record in (
            criterion_a.to_record(),
            start_a.to_record(),
            observation_a.to_record(),
        )
    )


def test_criterion_may_have_no_validity_deadline():
    criterion = _criterion(valid_until=None)

    assert criterion.valid_until is None
    assert criterion.is_valid(1_000_000)
    assert QualificationCriterion.from_record(criterion.to_record()).valid_until is None


def test_valid_causality_accepts_boundaries_and_multi_criterion_evidence():
    criterion = _criterion(issued_at=1, valid_until=2)
    start = _start(criterion, started_at=2)
    observation = _observation(criterion, start, observed_at=2)
    evidence = _evidence(criterion, start, observation, issued_at=2)

    assert (
        validate_qualification_causality(criterion, start, observation, evidence)
        == evidence.evidence_id
    )


def test_postdated_and_expired_criteria_are_refused():
    postdated = _criterion(issued_at=2, valid_until=4)
    postdated_start = _start(postdated, started_at=2)
    postdated_observation = _observation(postdated, postdated_start, observed_at=3)
    postdated_evidence = _evidence(
        postdated, postdated_start, postdated_observation, issued_at=4
    )
    with pytest.raises(ValueError, match="issued before campaign start"):
        validate_qualification_causality(
            postdated,
            postdated_start,
            postdated_observation,
            postdated_evidence,
        )

    expired = _criterion(issued_at=1, valid_until=2)
    expired_start = _start(expired, started_at=3)
    expired_observation = _observation(expired, expired_start, observed_at=3)
    expired_evidence = _evidence(expired, expired_start, expired_observation, issued_at=4)
    with pytest.raises(ValueError, match="expired before campaign start"):
        validate_qualification_causality(
            expired,
            expired_start,
            expired_observation,
            expired_evidence,
        )


def test_forged_serialized_records_are_refused():
    criterion, start, observation, evidence = _linked_records()
    cases = (
        (criterion, QualificationCriterion, "metric", "forged-metric"),
        (start, CampaignStartRecord, "campaign_spec_id", "forged-spec"),
        (
            observation,
            CampaignObservationRecord,
            "raw_artifact_ids",
            ["forged-artifact"],
        ),
        (evidence, QualificationEvidence, "reason", "forged-reason"),
    )
    for value, record_type, field, forged_value in cases:
        record = value.to_record()
        record[field] = forged_value
        with pytest.raises(ValueError, match="invalid content address"):
            record_type.from_record(record)


def test_superseded_passing_evidence_cannot_satisfy_campaign_predicates():
    criterion, start, observation, old_evidence = _linked_records()
    replacement = _evidence(
        criterion,
        start,
        observation,
        outcome="failed",
        issued_at=5,
        expires_at=20,
        supersedes_evidence_ids=(old_evidence.evidence_id,),
    )
    matrix = QualificationMatrix(
        {
            "campaign-result": {
                "evidence_kind": "scientific",
                "criterion_id": criterion.criterion_id,
                "campaign_start_record_id": start.start_record_id,
                "campaign_observation_record_id": (observation.observation_record_id),
            }
        }
    )

    report = matrix.evaluate((old_evidence, replacement), at_time=6)

    assert report.outcome == "failed"
    assert report.matched_evidence_ids == (replacement.evidence_id,)


def test_borrowed_observation_is_refused():
    criterion = _criterion()
    start = _start(criterion)
    other_start = _start(criterion, started_at=3)
    observation = _observation(
        criterion,
        start,
        start_record_id=other_start.start_record_id,
    )
    evidence = _evidence(criterion, start, observation)

    with pytest.raises(ValueError, match="borrowed from another start"):
        validate_qualification_causality(criterion, start, observation, evidence)


def test_mutated_live_records_are_refused_before_linkage():
    mutation_cases = (
        ("criterion", "metric", "mutated-metric"),
        ("start", "started_at", 9),
        ("observation", "observed_at", 9),
        ("evidence", "reason", "mutated-reason"),
    )
    for record_name, field, value in mutation_cases:
        criterion, start, observation, evidence = _linked_records()
        records = {
            "criterion": criterion,
            "start": start,
            "observation": observation,
            "evidence": evidence,
        }
        object.__setattr__(records[record_name], field, value)
        with pytest.raises(ValueError, match="invalid content address"):
            validate_qualification_causality(criterion, start, observation, evidence)


def test_mismatched_start_and_observation_records_are_refused():
    criterion = _criterion()
    mismatched_start = _start(criterion, criterion_id="other-criterion")
    observation = _observation(criterion, mismatched_start)
    evidence = _evidence(criterion, mismatched_start, observation)
    with pytest.raises(ValueError, match="exact criterion"):
        validate_qualification_causality(
            criterion, mismatched_start, observation, evidence
        )

    mismatched_start = _start(
        criterion, support_tuple_id=_support_tuple_id(chemistry="lfp")
    )
    observation = _observation(criterion, mismatched_start)
    evidence = _evidence(criterion, mismatched_start, observation)
    with pytest.raises(ValueError, match="criterion support tuple"):
        validate_qualification_causality(
            criterion, mismatched_start, observation, evidence
        )

    start = _start(criterion)
    observation_cases = (
        (
            _observation(criterion, start, campaign_spec_id="other-spec"),
            "campaign_spec",
        ),
        (
            _observation(criterion, start, criterion_id="other-criterion"),
            "criterion",
        ),
        (
            _observation(criterion, start, resolved_run_spec_id="other-resolved-run"),
            "resolved_run_spec",
        ),
        (
            _observation(
                criterion,
                start,
                support_tuple_id=_support_tuple_id(chemistry="lfp"),
            ),
            "support_tuple",
        ),
    )
    for mismatched_observation, field in observation_cases:
        mismatched_evidence = _evidence(criterion, start, mismatched_observation)
        with pytest.raises(ValueError, match=field):
            validate_qualification_causality(
                criterion, start, mismatched_observation, mismatched_evidence
            )


def test_mismatched_evidence_linkage_and_artifacts_are_refused():
    criterion, start, observation, _ = _linked_records()
    evidence_cases = (
        (
            _evidence(
                criterion,
                start,
                observation,
                criteria_ids=("other-criterion",),
            ),
            "exact criterion",
        ),
        (
            _evidence(
                criterion,
                start,
                observation,
                campaign_start_record_ids=("other-start",),
            ),
            "campaign start",
        ),
        (
            _evidence(
                criterion,
                start,
                observation,
                campaign_observation_record_ids=("other-observation",),
            ),
            "campaign observation",
        ),
        (
            _evidence(
                criterion,
                start,
                observation,
                raw_artifact_ids=("other-artifact",),
            ),
            "exact campaign raw artifacts",
        ),
    )
    for evidence, message in evidence_cases:
        with pytest.raises(ValueError, match=message):
            validate_qualification_causality(criterion, start, observation, evidence)


@pytest.mark.parametrize(
    ("observed_at", "evidence_issued_at"),
    ((1, 4), (5, 4)),
)
def test_observation_must_follow_start_and_precede_evidence(
    observed_at: int,
    evidence_issued_at: int,
):
    criterion = _criterion()
    start = _start(criterion, started_at=2)
    observation = _observation(criterion, start, observed_at=observed_at)
    evidence = _evidence(
        criterion,
        start,
        observation,
        issued_at=evidence_issued_at,
    )

    with pytest.raises(ValueError, match="started_at <= observed_at <= evidence"):
        validate_qualification_causality(criterion, start, observation, evidence)
