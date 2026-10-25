from phydrax.atomistic._mace_profiles import mace_candidate_profiles
from phydrax.atomistic._profiles import (
    atomistic_candidate_declarations,
    CLOSURE_EXCEPTION,
)
from phydrax.qualification import (
    builtin_candidate_profiles,
    builtin_capability_catalog,
    builtin_source_absorption_ledger,
    CapabilityDeclaration,
    CapabilityDisposition,
    SourceReuseClass,
    validate_source_coverage,
)
from phydrax.qualification._registry import SupportValue


_OWNERS = {
    "atomistic.aimnet2-inference": "phydrax.nn.atomistic",
    "atomistic.crystal-structure-prediction": "phydrax.atomistic.crystal",
    "atomistic.enhanced-sampling": "phydrax.atomistic.sampling",
    "atomistic.mace-extended-inference": "phydrax.nn.atomistic",
    "atomistic.maceles-inference": "phydrax.nn.atomistic",
    "atomistic.model-artifact-interchange": "phydrax.atomistic.interchange",
    "atomistic.particle-grid-spline": "phydrax.discretization.splatting",
    "atomistic.reaction-path": "phydrax.chemistry.reaction",
    "atomistic.segmented-route-reduction": "phydrax.sparse",
    "atomistic.uma-inference": "phydrax.nn.atomistic",
}


def _attributes(
    declaration: CapabilityDeclaration,
) -> tuple[dict[str, SupportValue], ...]:
    return tuple(
        dict(support.attributes)
        for profile in declaration.profiles
        for support in profile.support_tuples
    )


def test_atomistic_candidates_load_unreleased_through_public_catalog() -> None:
    catalog = builtin_capability_catalog()
    discovered = {profile.profile_id for profile in builtin_candidate_profiles()}
    declarations = atomistic_candidate_declarations()
    assert len({value.capability for value in declarations}) == len(declarations)
    for expected in declarations:
        declaration = catalog.declaration(expected.capability)
        assert declaration.declaration_id == expected.declaration_id
        assert declaration.disposition is CapabilityDisposition.CANDIDATE
        assert declaration.owner == _OWNERS.get(
            declaration.capability, "phydrax.atomistic"
        )
        assert "not-release-authorized" in declaration.nonclaims
        assert declaration.profiles
        for profile in declaration.profiles:
            assert profile.profile_id in discovered
            assert profile.released is False
            assert not profile.release_evidence
            assert {"implementation-complete", "independent-review"} <= set(
                profile.required_gates
            )
        for attributes in _attributes(declaration):
            assert attributes["closure-exception"] == CLOSURE_EXCEPTION
            assert attributes["release-claim"] is False
            assert attributes["provider"] == "phydrax-native"


def test_standard_mace_profiles_keep_their_owner_and_identity() -> None:
    catalog = builtin_capability_catalog()
    standard = catalog.declaration("atomistic.mace-inference")
    assert standard.owner == "phydrax.atomistic"
    assert {profile.profile_id for profile in standard.profiles} == {
        profile.profile_id for profile in mace_candidate_profiles()
    }
    for capability in (
        "atomistic.mace-extended-inference",
        "atomistic.maceles-inference",
    ):
        declaration = catalog.declaration(capability)
        assert "atomistic.mace-inference" in declaration.dependencies
        assert all(
            attributes.get("architecture") != "standard-mace"
            for attributes in _attributes(declaration)
        )


def test_crystal_domain_is_its_own_closure_exception() -> None:
    declaration = builtin_capability_catalog().declaration(
        "atomistic.crystal-structure-prediction"
    )
    assert declaration.owner == "phydrax.atomistic.crystal"
    assert declaration.public_symbols
    assert all(
        symbol.startswith("phydrax.atomistic.crystal.")
        for symbol in declaration.public_symbols
    )
    assert {"atomistic.generation", "atomistic.relaxation"} <= set(
        declaration.dependencies
    )
    assert {
        "no-authoritative-wyckoff-letters",
        "radial-screening-is-not-equivalence-proof",
    } <= set(declaration.nonclaims)
    for profile in declaration.profiles:
        assert "space-group-table-rights-admission" in profile.required_gates
    stages = {attributes["stage"] for attributes in _attributes(declaration)}
    assert {
        "space-group-setting",
        "packing",
        "physical-relaxation",
        "equivalence-confirmation",
        "radial-screening",
        "ranking",
    } <= stages


def test_envelopes_keep_geometry_derivative_and_rights_boundaries() -> None:
    catalog = builtin_capability_catalog()
    for declaration in atomistic_candidate_declarations():
        for attributes in _attributes(declaration):
            derivative = str(attributes.get("derivative", ""))
            if attributes.get("geometry") == "finite":
                assert "stress" not in derivative
            if attributes.get("method") == "fourier-d3-modified-cn":
                assert attributes["geometry"] in ("orthorhombic", "triclinic")
            if attributes.get("cell") == "mttk-tuckerman-2006-exponential":
                assert attributes["periodic-axes"] == 3
    for capability, gate in (
        ("atomistic.aimnet2-inference", "weights-rights-admission"),
        ("atomistic.uma-inference", "weights-rights-admission"),
        ("atomistic.maceles-inference", "les-source-rights-admission"),
        ("atomistic.latent-electrostatics", "les-source-rights-admission"),
        ("atomistic.dispersion-d3", "d3-table-rights-admission"),
    ):
        for profile in catalog.declaration(capability).profiles:
            assert gate in profile.required_gates
    uma = catalog.declaration("atomistic.uma-inference")
    for attributes in _attributes(uma):
        if attributes["model-kind"] == "property-only":
            assert attributes["derivative"] == "none"
            assert attributes["invented-energy"] is False


def test_inspected_atomistic_sources_keep_rights_and_inspection_distinct() -> None:
    ledger = builtin_source_absorption_ledger()
    assert validate_source_coverage(ledger) == ()
    sources = {source.source_id: source for source in ledger.sources}
    for source_id in (
        "aimnetcentral",
        "fairchem",
        "grimme-dftd3",
        "les",
        "mace-code-0.3.15",
        "nvalchemi-toolkit",
        "nvalchemi-toolkit-ops",
        "spglib",
    ):
        source = sources[source_id]
        assert source.code_inspected is True
        assert source.behavior_inspected is False
    les = sources["les"]
    assert les.reuse_class is SourceReuseClass.SOURCE_AVAILABLE
    assert les.copying_permitted is False
    assert les.provider_only is True
    dftd3 = sources["grimme-dftd3"]
    assert dftd3.reuse_class is SourceReuseClass.STRONG_COPYLEFT
    assert dftd3.copying_permitted is False
