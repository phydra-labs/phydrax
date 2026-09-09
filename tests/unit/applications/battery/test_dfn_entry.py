#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace

import pytest

from phydrax.applications.battery._dfn_entry import (
    DfnEntryAssessment,
    DfnEntryPolicy,
    DfnReferenceComparison,
    evaluate_dfn_entry_gate,
    evaluate_reference_consensus,
)
from phydrax.applications.battery._qualification import MARQUIS_2019_SPME_SUPPORT
from phydrax.applications.battery._rights import ReferenceRightsAttestation
from phydrax.qualification import ReferenceArtifactManifest


MAPPING = tuple(
    (axis, f"test:{axis}")
    for axis in ("parameters", "geometry", "initialization", "hold", "output")
)


def _assessment(candidate=0.10, *, reference_values=(0.0, 0.001), uncertainty=0.002):
    policy = DfnEntryPolicy(
        MARQUIS_2019_SPME_SUPPORT.support_tuple_id,
        "test:distribution",
        (("pulse", "voltage_v", 0.0), ("rest", "voltage_v", 1.0)),
        (1.0, 1.0),
        0.05,
        0.001,
        0.01,
        0.5,
        1,
        100,
    )
    references = []
    for index, value in enumerate(reference_values):
        manifest = ReferenceArtifactManifest(
            f"test-reference-{index}",
            checksum_algorithm="sha256",
            checksum=str(index + 1) * 64,
            size_bytes=10,
            license_id="test-only",
            commercial_use_permitted=True,
            redistribution_permitted=True,
            training_use_permitted=False,
            export_permitted=True,
            export_classification="test-only",
            nondimensionalization={"voltage": 1.0},
            uncertainty={"voltage_v": 0.001},
            lineage_ids=(f"test:engine-{index}",),
        )
        rights = ReferenceRightsAttestation(
            manifest.manifest_id,
            f"test:source-{index}",
            "test-only",
            "test:notice",
            True,
            True,
            True,
            True,
            1,
            100,
        )
        references.append(
            DfnReferenceComparison(
                f"test:engine-{index}",
                f"test:runtime-{index}",
                manifest,
                MAPPING,
                (value + 0.0008,) * 2,
                (value + 0.0002,) * 2,
                (value,) * 2,
                (uncertainty,) * 2,
                (f"test:{index}-coarse", f"test:{index}-medium", f"test:{index}-fine"),
                rights,
                None,
            )
        )
    return DfnEntryAssessment(
        policy,
        tuple(references),
        (candidate,) * 2,
        (0.001,) * 2,
        MAPPING,
        2,
        3,
        ("raw:candidate", "raw:first", "raw:second"),
    )


def test_authorization_uses_the_minimum_of_two_lower_bounds():
    assessment = _assessment()
    decision = evaluate_reference_consensus(assessment)
    assert decision.eligible and decision.conclusive
    assert decision.robust_lower_bound == pytest.approx(0.096)
    # One apparently decisive reference cannot override the other reference.
    second = replace(
        assessment.references[1],
        coarse_values=(0.0478,) * 2,
        medium_values=(0.0472,) * 2,
        fine_values=(0.047,) * 2,
    )
    widened_consensus = replace(assessment.policy, cross_reference_limit=0.1)
    decision = evaluate_reference_consensus(
        replace(
            assessment,
            policy=widened_consensus,
            references=(assessment.references[0], second),
        )
    )
    assert not decision.eligible
    assert not decision.conclusive


def test_no_build_requires_both_uncertainty_upper_bounds_strictly_below():
    decision = evaluate_reference_consensus(_assessment(candidate=0.01))
    assert not decision.eligible and decision.conclusive
    at_boundary = evaluate_reference_consensus(
        _assessment(candidate=0.047, reference_values=(0.0, 0.0))
    )
    assert not at_boundary.eligible and not at_boundary.conclusive


def test_cross_reference_disagreement_cannot_authorize_even_when_each_error_is_large():
    assessment = _assessment(candidate=0.2, reference_values=(0.0, 0.1))
    decision = evaluate_reference_consensus(assessment)
    assert all(
        lower > assessment.policy.discrepancy_threshold
        for lower, _ in decision.reference_intervals
    )
    assert not decision.eligible and not decision.conclusive


def test_two_distinct_runtimes_and_manifests_are_mandatory():
    assessment = _assessment()
    duplicate_runtime = replace(
        assessment.references[1], runtime_family=assessment.references[0].runtime_family
    )
    assert not evaluate_reference_consensus(
        replace(assessment, references=(assessment.references[0], duplicate_runtime))
    ).conclusive
    with pytest.raises(ValueError):
        replace(assessment, references=(assessment.references[0],))
    with pytest.raises(ValueError):
        replace(assessment, references=(*assessment.references, assessment.references[0]))


def test_convergence_and_matched_geometry_are_independent_required_audits():
    assessment = _assessment()
    unresolved = replace(assessment.references[0], medium_values=(0.1,) * 2)
    assert not evaluate_reference_consensus(
        replace(assessment, references=(unresolved, assessment.references[1]))
    ).conclusive
    mismatched = replace(
        assessment.references[0],
        mapping_ids=tuple(
            (axis, "wrong:geometry" if axis == "geometry" else value)
            for axis, value in MAPPING
        ),
    )
    assert not evaluate_reference_consensus(
        replace(assessment, references=(mismatched, assessment.references[1]))
    ).conclusive
    underestimated = replace(assessment.references[0], uncertainty=(0.00001,) * 2)
    assert not evaluate_reference_consensus(
        replace(assessment, references=(underestimated, assessment.references[1]))
    ).conclusive


def test_numeric_eligibility_without_authority_never_admits_implementation():
    assessment = _assessment()
    assert evaluate_reference_consensus(assessment).eligible
    decision = evaluate_dfn_entry_gate(
        release_index=None,
        trust_policy=None,
        marquis_profile_id="unsigned:profile",
        marquis_support=MARQUIS_2019_SPME_SUPPORT,
        assessment=assessment,
        at_time=10,
    )
    assert not decision.eligible and not decision.conclusive


def test_policy_precedes_execution_and_nonfinite_discrepancies_are_rejected():
    assessment = _assessment()
    with pytest.raises(ValueError):
        replace(assessment, started_at=assessment.policy.issued_at)
    with pytest.raises(ValueError):
        replace(assessment, candidate_values=(float("nan"), 1.0))
