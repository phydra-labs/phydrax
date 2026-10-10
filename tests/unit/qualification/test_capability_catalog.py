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
        (SupportTuple(capability, {"precision": "float64"}),),
        required_gates=("numerical",),
    )


def test_capability_catalog_scenario_1() -> None:
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
    profile = _candidate()
    declarations = declarations_from_profiles(
        (profile, profile),
        owners={profile.capability: "phydrax.linalg"},
    )

    assert len(declarations) == 1
    assert declarations[0].profiles == (profile,)
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


def test_capability_catalog_scenario_2() -> None:
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
    from phydrax.qualification import builtin_capability_catalog
    from phydrax.rom import rom_capability_catalog

    expected = {
        entry.profile(provider="phydrax").capability: (
            entry.maturity.name.lower().replace("_", "-")
        )
        for entry in rom_capability_catalog()
    }
    declared = {
        declaration.capability: declaration.domain_maturity
        for declaration in builtin_capability_catalog().declarations
        if declaration.owner == "phydrax.rom"
    }

    assert declared == expected


def test_native_meshing_catalog_keeps_qualification_and_leadership_open() -> None:
    from phydrax.qualification import builtin_capability_catalog

    declarations = tuple(
        declaration
        for declaration in builtin_capability_catalog().declarations
        if declaration.capability.startswith("meshing.native.")
    )
    assert declarations
    for declaration in declarations:
        assert declaration.disposition in (
            CapabilityDisposition.CANDIDATE,
            CapabilityDisposition.RESEARCH,
        )
        assert all(not profile.released for profile in declaration.profiles)
        assert "docs/guides_meshing.md" in declaration.documentation
        assert (
            "focused-test-and-tooling-passes-are-not-final-qualification-artifacts"
            in declaration.nonclaims
        )
        assert "final-like-for-like-leadership-campaign-not-run" in declaration.nonclaims

    for capability in (
        "meshing.native.mandatory-periodic-combination",
        "meshing.native.mandatory-hybrid-layers",
    ):
        declaration = next(item for item in declarations if item.capability == capability)
        assert declaration.disposition is CapabilityDisposition.CANDIDATE
        assert (
            "historical-curved-periodic-narrow-gap-source-remains-an-immutable-exact-negative"
            in declaration.nonclaims
        )
