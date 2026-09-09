#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.finance.qualification import (
    advanced_finance_support,
    build_finance_qualification_matrix,
    evaluate_finance_campaign,
    valuation_support,
)
from phydrax.qualification import QualificationEvidence


def _evidence(support_id: str, kind: str) -> QualificationEvidence:
    return QualificationEvidence(
        kind,
        "passed",
        (support_id,),
        build_id="synthetic-build",
        environment_id="synthetic-cpu",
        backend="jax",
        topology="single-process",
        precision="float64",
        reduction="deterministic",
        replay_id="synthetic-replay",
        criteria_ids=(f"{kind}-criterion",),
        raw_artifact_ids=(f"{kind}-artifact",),
        campaign_start_record_ids=(),
        campaign_observation_record_ids=(),
        reviewer_id="synthetic-reviewer",
        issued_at=10,
        expires_at=30,
        reason="criterion-satisfied",
    )


def test_finance_campaign_keeps_evidence_planes_distinct_and_fails_closed():
    support = valuation_support(
        "analytic",
        product="european-option",
        model="black-scholes",
        pricing_law="usd-risk-neutral",
    )
    matrix = build_finance_qualification_matrix((support,))
    incomplete = evaluate_finance_campaign(
        matrix,
        tuple(
            _evidence(support.support_tuple_id, kind)
            for kind in ("reference", "scientific", "unit")
        ),
        at_time=20,
    )
    assert not incomplete.passed
    assert incomplete.coverage.outcome == "inconclusive"
    assert len(incomplete.coverage.inconclusive_predicate_ids) == 1

    complete = evaluate_finance_campaign(
        matrix,
        tuple(
            _evidence(support.support_tuple_id, kind)
            for kind in ("reference", "scientific", "unit", "operational")
        ),
        at_time=20,
    )
    assert complete.passed
    assert complete.coverage.coverage_fraction == 1.0
    restored = type(complete).from_record(complete.to_record())
    assert complete.campaign_id == restored.campaign_id


def test_advanced_support_cannot_claim_more_than_candidate_maturity():
    with pytest.raises(ValueError, match="candidate ceiling"):
        advanced_finance_support(
            "rough",
            representation="rough-volatility-kernel",
            law="physical",
            candidate_ceiling="production",
        )
