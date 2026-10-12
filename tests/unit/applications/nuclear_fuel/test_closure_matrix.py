#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Numerical closure cannot authorize a rights-blocked scientific release."""

from __future__ import annotations

import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.nuclear_fuel import nuclear_fuel_profiles
from phydrax.qualification._closure_taxonomy import ClosureDisposition
from phydrax.qualification._nuclear_fuel_release_matrix import (
    FUEL_VERIFICATION_CASES,
    nuclear_fuel_closure_matrices,
    nuclear_fuel_release_matrix_candidate_profiles,
    nuclear_fuel_source_references,
)


def test_numerical_closure_does_not_authorize_scientific_release() -> None:
    # These are isolated contract-test artifact identities, not physical evidence.
    artifacts = {
        row.case_id: (canonical_fingerprint({"contract-fixture": row.case_id}),)
        for row in FUEL_VERIFICATION_CASES
    }
    faults = {
        row.case_id: (canonical_fingerprint({"fault-contract-fixture": row.case_id}),)
        for row in FUEL_VERIFICATION_CASES
    }
    numerical, scientific = nuclear_fuel_closure_matrices(
        artifacts, fault_adequacy_evidence=faults
    )
    assert numerical.implementation_closed
    assert not numerical.release_closed
    assert not scientific.implementation_closed
    assert not scientific.release_closed
    assert all(
        row.disposition is ClosureDisposition.MISSING for row in scientific.resolutions
    )


def test_partial_campaign_keeps_implementation_open() -> None:
    numerical, _ = nuclear_fuel_closure_matrices(
        {"V1": (canonical_fingerprint({"case": "V1"}),)},
        fault_adequacy_evidence={"V1": (canonical_fingerprint({"fault": "V1"}),)},
    )
    assert not numerical.implementation_closed
    assert not numerical.release_closed
    missing = {
        row.requirement_id
        for row in numerical.resolutions
        if row.disposition is ClosureDisposition.MISSING
    }
    assert {
        row.requirement for row in numerical.requirements if row.requirement_id in missing
    } == {f"nuclear-fuel-V{index}" for index in range(2, 16)}


@pytest.mark.parametrize("dimension", ["numerical", "fault-adequacy"])
def test_one_evidence_dimension_cannot_close_implementation(dimension: str) -> None:
    artifact = canonical_fingerprint({"dimension": dimension, "case": "V1"})
    observed = {"V1": (artifact,)}
    numerical, _ = nuclear_fuel_closure_matrices(
        observed if dimension == "numerical" else None,
        fault_adequacy_evidence=observed if dimension == "fault-adequacy" else None,
    )
    requirement = next(
        row for row in numerical.requirements if row.requirement == "nuclear-fuel-V1"
    )
    resolution = next(
        row
        for row in numerical.resolutions
        if row.requirement_id == requirement.requirement_id
    )
    assert resolution.disposition is ClosureDisposition.MISSING
    assert resolution.evidence_ids == (artifact,)
    assert not numerical.implementation_closed


def test_numerical_artifact_cannot_be_reused_as_fault_witness() -> None:
    artifacts = {"V1": (canonical_fingerprint({"numerical-only-fixture": "V1"}),)}
    with pytest.raises(ValueError, match="distinct observed artifacts"):
        nuclear_fuel_closure_matrices(artifacts, fault_adequacy_evidence=artifacts)


@pytest.mark.parametrize(
    "evidence",
    [{"V16": ("unknown",)}, {"V1": ()}],
    ids=["unknown-case", "empty-artifact"],
)
def test_invalid_campaign_evidence_is_refused(
    evidence: dict[str, tuple[str, ...]],
) -> None:
    with pytest.raises(ValueError, match="nonempty known"):
        nuclear_fuel_closure_matrices(evidence)


def test_family_and_matrix_have_no_release_evidence() -> None:
    family = nuclear_fuel_profiles()[0]
    matrix = nuclear_fuel_release_matrix_candidate_profiles()[0]
    assert not family.released and not matrix.released
    assert family.release_evidence == matrix.release_evidence == ()
    assert "nuclear-fuel.ifpe-data-rights" in family.required_gates
    assert "nuclear-fuel.independent-locked-evaluation" in matrix.required_gates
    assert (
        dict(family.support_tuples[0].attributes)["support_kind"]
        == "source-bound-family-declaration"
    )


def test_public_report_does_not_grant_experimental_data_rights() -> None:
    sources = {source.source_id: source for source in nuclear_fuel_source_references()}
    for source in sources.values():
        assert not source.copying_permitted
        assert not source.code_inspected and not source.behavior_inspected
        assert source.legal_reviewer == "unreviewed"
