#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import pytest

from phydrax.qualification import (
    CapabilityCatalog,
    CapabilityDeclaration,
    CapabilityDisposition,
    CapabilityProfile,
    declarations_from_profiles,
    EvidenceAssessment,
    EvidenceDimension,
    EvidenceState,
    ReleaseGateEvidence,
    SupportTuple,
)


def _candidate(capability: str = "core.linear-solve") -> CapabilityProfile:
    return CapabilityProfile(
        f"{capability}.profile",
        "phydrax",
        "candidate",
        (SupportTuple(capability, {"precision": "float64"}),),
        required_gates=("numerical",),
    )


def test_candidate_catalog_roundtrip_and_lookup() -> None:
    profile = _candidate()
    declaration = CapabilityDeclaration(
        profile.capability,
        "phydrax.linalg",
        CapabilityDisposition.CANDIDATE,
        domain_maturity="candidate",
        public_symbols=("phydrax.linalg.solve",),
        profiles=(profile,),
        documentation=("docs/api/linalg.md",),
        examples=("examples/linear_solve.py",),
        intended_uses=("bounded-linear-solve",),
        nonclaims=("not-released",),
    )
    catalog = CapabilityCatalog((declaration,))

    restored = CapabilityCatalog.from_record(catalog.to_record())

    assert restored.catalog_id == catalog.catalog_id
    assert restored.declaration(profile.capability).declaration_id == (
        declaration.declaration_id
    )
    assert restored.symbol_owner("phydrax.linalg.solve").capability == (
        profile.capability
    )
    assert restored.by_disposition("candidate") == restored.declarations


def test_candidate_factory_deduplicates_identical_profiles() -> None:
    profile = _candidate()
    declarations = declarations_from_profiles(
        (profile, profile),
        owners={profile.capability: "phydrax.linalg"},
    )

    assert len(declarations) == 1
    assert declarations[0].profiles == (profile,)


def test_catalog_refuses_duplicate_public_symbol_ownership() -> None:
    first = _candidate("core.first")
    second = _candidate("core.second")
    declarations = tuple(
        CapabilityDeclaration(
            profile.capability,
            f"phydrax.{profile.capability.split('.')[-1]}",
            "candidate",
            public_symbols=("phydrax.shared.symbol",),
            profiles=(profile,),
            nonclaims=("not-released",),
        )
        for profile in (first, second)
    )

    with pytest.raises(ValueError, match="owned by both"):
        CapabilityCatalog(declarations)


def test_catalog_refuses_unknown_dependency_and_cycle() -> None:
    profile = _candidate("core.first")
    with pytest.raises(ValueError, match="unknown dependencies"):
        CapabilityCatalog(
            (
                CapabilityDeclaration(
                    profile.capability,
                    "phydrax.first",
                    "candidate",
                    profiles=(profile,),
                    dependencies=("core.missing",),
                    nonclaims=("not-released",),
                ),
            )
        )

    second = _candidate("core.second")
    with pytest.raises(ValueError, match="dependency cycle"):
        CapabilityCatalog(
            (
                CapabilityDeclaration(
                    profile.capability,
                    "phydrax.first",
                    "candidate",
                    profiles=(profile,),
                    dependencies=(second.capability,),
                    nonclaims=("not-released",),
                ),
                CapabilityDeclaration(
                    second.capability,
                    "phydrax.second",
                    "candidate",
                    profiles=(second,),
                    dependencies=(profile.capability,),
                    nonclaims=("not-released",),
                ),
            )
        )


def test_released_declaration_requires_complete_evidence_and_public_surface() -> None:
    support = SupportTuple("core.linear-solve", {"precision": "float64"})
    gates = tuple(dimension.value for dimension in EvidenceDimension)
    release_evidence = tuple(
        ReleaseGateEvidence(
            gate,
            passed=True,
            evidence_ids=(f"artifact:{gate}",),
            reviewer_id=f"reviewer:{gate}",
            issued_at=1,
            expires_at=2,
        )
        for gate in gates
    )
    profile = CapabilityProfile(
        "core.linear-solve.profile",
        "phydrax",
        "1",
        (support,),
        required_gates=gates,
        release_evidence=release_evidence,
        released=True,
    )
    assessments = tuple(
        EvidenceAssessment(
            dimension,
            EvidenceState.PASSED,
            evidence_ids=(f"artifact:{dimension.value}",),
            reason="accepted",
        )
        for dimension in EvidenceDimension
    )

    declaration = CapabilityDeclaration(
        support.capability,
        "phydrax.linalg",
        "released",
        public_symbols=("phydrax.linalg.solve",),
        profiles=(profile,),
        evidence=assessments,
        documentation=("docs/api/linalg.md",),
        examples=("examples/linear_solve.py",),
        intended_uses=("bounded-linear-solve",),
    )

    assert declaration.disposition is CapabilityDisposition.RELEASED


def test_nonreleased_declaration_refuses_released_profile() -> None:
    support = SupportTuple("core.linear-solve", {"precision": "float64"})
    gate = ReleaseGateEvidence(
        "numerical",
        passed=True,
        evidence_ids=("artifact:numerical",),
        reviewer_id="reviewer:numerical",
        issued_at=1,
        expires_at=2,
    )
    profile = CapabilityProfile(
        "core.linear-solve.profile",
        "phydrax",
        "1",
        (support,),
        required_gates=("numerical",),
        release_evidence=(gate,),
        released=True,
    )

    with pytest.raises(ValueError, match="non-released declaration"):
        CapabilityDeclaration(
            support.capability,
            "phydrax.linalg",
            "candidate",
            profiles=(profile,),
            nonclaims=("not-released",),
        )
