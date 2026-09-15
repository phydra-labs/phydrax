#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import tools.cm_production_closure as closure_orchestration
from phydrax.applications._condensed_matter_evidence import (
    condensed_matter_candidate_campaigns,
    condensed_matter_candidate_profiles,
    condensed_matter_frontier_candidate_campaigns,
    condensed_matter_frontier_candidate_profiles,
    CondensedMatterCampaignAggregation,
    CondensedMatterCampaignReference,
    CondensedMatterClosureLedger,
    require_condensed_matter_closure,
)
from phydrax.applications.semiconductor._detector_lifecycle import (
    read_detector_artifact_archive,
    write_detector_artifact_archive,
)
from phydrax.applications.semiconductor._detector_response import (
    ShockleyRamoResourceEvidence,
    ShockleyRamoResponseResult,
)
from phydrax.chemistry.periodic._lifecycle import (
    read_periodic_artifact_archive,
    write_periodic_artifact_archive,
)
from phydrax.chemistry.periodic._qualification import periodic_candidate_profiles
from phydrax.lifecycle._array_artifact import ArrayArtifactProvenance
from phydrax.operators.periodic import (
    PeriodicTranslationFamilyPlan,
    PeriodicTranslationFamilyState,
)
from phydrax.operators.quantum import FermionModeOrder
from phydrax.operators.quantum.lattice import (
    FixedCardinalityFermionBasis,
    SectorBasisResourcePolicy,
)
from phydrax.operators.quantum.lattice._lifecycle import (
    read_quantum_lattice_artifact_archive,
    write_quantum_lattice_artifact_archive,
)
from phydrax.qualification import (
    CampaignRole,
    CapabilityProfile,
    ReleaseIndex,
    ScientificCampaign,
    ScientificCase,
    SupportTuple,
)
from phydrax.sparse import EdgeRelation


def _provenance(profile_id="profile:periodic"):
    return ArrayArtifactProvenance(
        "phydrax-test-producer",
        ("source:independent-fixture",),
        (profile_id,),
        ("unit:explicit",),
    )


def test_periodic_family_archive_round_trip_requires_matching_structure(tmp_path):
    relation = EdgeRelation([0], [0], source_size=1, target_size=1)
    plan = PeriodicTranslationFamilyPlan(relation, [[0]], [0])
    state = PeriodicTranslationFamilyState(plan, [[[2.5]]])
    path = tmp_path / "periodic-family.pxa"

    written = write_periodic_artifact_archive(path, state, _provenance())
    restored, reopened = read_periodic_artifact_archive(path, state, _provenance())

    assert reopened.artifact_id == written.artifact_id
    assert restored.numeric_id == state.numeric_id
    np.testing.assert_array_equal(restored.values, state.values)

    other = PeriodicTranslationFamilyState(plan, [[[3.0]]])
    with pytest.raises(ValueError, match="caller-prepared structure"):
        read_periodic_artifact_archive(path, other, _provenance())


def test_quantum_sector_archive_preserves_direct_basis_tables(tmp_path):
    policy = SectorBasisResourcePolicy(
        maximum_dimension=16,
        maximum_table_bytes=4096,
    )
    basis = FixedCardinalityFermionBasis(
        FermionModeOrder(("left-up", "left-down", "right-up", "right-down")),
        2,
        resources=policy,
    )
    path = tmp_path / "sector.pxa"

    write_quantum_lattice_artifact_archive(path, basis, _provenance("profile:sector"))
    restored, _ = read_quantum_lattice_artifact_archive(
        path,
        basis,
        _provenance("profile:sector"),
    )

    assert restored.basis_id == basis.basis_id
    np.testing.assert_array_equal(restored.state_charges, basis.state_charges)
    np.testing.assert_array_equal(restored.suffix_counts, basis.suffix_counts)


def test_detector_result_archive_preserves_masks_residuals_and_sign(tmp_path):
    result = ShockleyRamoResponseResult(
        jnp.asarray([[[0.0, -1.0, -2.0], [0.0, 1.0, 2.0]]]),
        jnp.asarray([[[-1.0, -1.0], [1.0, 1.0]]]),
        jnp.asarray([[-2.0, 2.0]]),
        jnp.asarray([[-2.0, 2.0]]),
        jnp.asarray([[0.0, 0.0]]),
        jnp.asarray([[True, True, False]]),
        jnp.asarray([[True, True, False]]),
        jnp.asarray([[True, False]]),
        jnp.asarray([False]),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(False),
        ShockleyRamoResourceEvidence(1, 3, 3, 2),
        "trajectory:fixture",
        "weighting:fixture",
        "route:fixture",
        "Q_induced=-q*phi_w;i_into_electrode=dQ_induced/dt",
    )
    path = tmp_path / "detector-result.pxa"

    write_detector_artifact_archive(path, result, _provenance("profile:detector"))
    restored, _ = read_detector_artifact_archive(
        path,
        result,
        _provenance("profile:detector"),
    )

    np.testing.assert_array_equal(
        restored.interpolation_support, result.interpolation_support
    )
    np.testing.assert_array_equal(
        restored.current_integral_closure_defect,
        result.current_integral_closure_defect,
    )
    assert restored.sign_convention == result.sign_convention
    assert not bool(restored.successful)


def test_periodic_profiles_are_exact_unreleased_single_tuple_candidates():
    profiles = periodic_candidate_profiles()

    assert profiles
    assert all(not profile.released for profile in profiles)
    assert all(len(profile.support_tuples) == 1 for profile in profiles)
    assert all(
        not profile.capability.startswith("chemistry.candidate.") for profile in profiles
    )
    assert len({profile.profile_id for profile in profiles}) == len(profiles)


def test_candidate_ledger_covers_every_implemented_baseline_slice():
    profiles = condensed_matter_candidate_profiles()
    capabilities = {profile.capability for profile in profiles}
    assert {
        "chemistry.periodic.lattice.ifc2",
        "chemistry.periodic.lattice.harmonic",
        "chemistry.periodic.lattice.qha",
        "chemistry.periodic.lattice.three-phonon-rta",
        "solver.impurity.all-sector-ed",
        "chemistry.periodic.embedding.single-site-dmft",
        "quantum.green.maximum-entropy-continuation",
        "chemistry.spectroscopy.optical-dielectric",
        "magnetic-resonance.nmr.exact-single-crystal",
        "soft-matter.passive-phase-field",
    } <= capabilities
    assert {
        "chemistry.lattice.ifc3.native",
        "functional-rg.fermionic-fermi-surface-patch",
        "diagrammatic-field.lattice-parquet",
        "nonequilibrium-field.fermionic-second-born",
        "sign-problem.controlled-sign-study",
    }.isdisjoint(capabilities)
    assert all(
        "candidate_only" not in dict(support.attributes)
        for profile in profiles
        for support in profile.support_tuples
    )


def test_frontier_inventory_is_separate_and_maturity_neutral():
    profiles = condensed_matter_frontier_candidate_profiles()
    capabilities = {profile.capability for profile in profiles}
    assert {
        "chemistry.lattice.ifc3.native",
        "functional-rg.fermionic-fermi-surface-patch",
        "diagrammatic-field.lattice-parquet",
        "nonequilibrium-field.fermionic-second-born",
        "sign-problem.controlled-sign-study",
    } <= capabilities
    assert all(
        "candidate_only" not in dict(support.attributes)
        for profile in profiles
        for support in profile.support_tuples
    )


class _StructuralTrustPolicy:
    def verify_index(self, index, at_time):
        return True

    def accepts_evidence(self, evidence, at_time):
        return True


def _candidate(name):
    support = SupportTuple(name, {"scope": "exact"})
    return CapabilityProfile(
        f"{name}.profile",
        "phydrax",
        "candidate",
        (support,),
        required_gates=("scientific-validation",),
        released=False,
    )


def test_closure_ledger_refuses_missing_and_unreleased_dependencies():
    required = _candidate("test.required")
    unrelated = _candidate("test.unrelated")
    ledger = CondensedMatterClosureLedger.from_profiles((required,))
    assert (
        CondensedMatterClosureLedger.from_record(ledger.to_record()).ledger_id
        == ledger.ledger_id
    )
    policy = _StructuralTrustPolicy()

    missing = ReleaseIndex(
        (unrelated,),
        issued_at=0,
        signer_id="structural-test",
        signature_algorithm="structural-test",
        signature="not-release-evidence",
    )
    with pytest.raises(ValueError, match="missing dependency"):
        require_condensed_matter_closure(missing, ledger, policy, at_time=0)

    unreleased = ReleaseIndex(
        (required,),
        issued_at=0,
        signer_id="structural-test",
        signature_algorithm="structural-test",
        signature="not-release-evidence",
    )
    with pytest.raises(ValueError, match="profile-unreleased"):
        require_condensed_matter_closure(unreleased, ledger, policy, at_time=0)


def _campaign(prefix, calibration_unit, locked_unit):
    calibration = ScientificCase(
        f"{prefix}-calibration",
        calibration_unit,
        f"{prefix}-construct-calibration",
        f"{prefix}-condition-calibration",
        f"{prefix}-preparation-calibration",
        f"{prefix}-batch-calibration",
        (f"{prefix}-source-calibration",),
    )
    locked = ScientificCase(
        f"{prefix}-locked",
        locked_unit,
        f"{prefix}-construct-locked",
        f"{prefix}-condition-locked",
        f"{prefix}-preparation-locked",
        f"{prefix}-batch-locked",
        (f"{prefix}-source-locked",),
    )
    return ScientificCampaign(
        (calibration, locked),
        (
            CampaignRole("calibration", (calibration.case_id,)),
            CampaignRole("locked_evaluation", (locked.case_id,)),
        ),
    )


def test_campaign_aggregation_retains_roles_and_refuses_cross_owner_leakage():
    aggregation = condensed_matter_candidate_campaigns()
    assert (
        CondensedMatterCampaignAggregation.from_record(
            aggregation.to_record()
        ).aggregation_id
        == aggregation.aggregation_id
    )
    assert {
        "chemistry.periodic.lattice",
        "chemistry.periodic.embedding",
        "chemistry.spectroscopy",
        "operators.quantum.lattice",
        "applications.magnetism",
        "applications.magnetic-resonance",
        "applications.semiconductor.detector",
        "applications.soft-matter",
    } <= {reference.owner_id for reference in aggregation.campaigns}
    assert {
        "applications.functional-rg",
        "applications.diagrammatic-field",
        "applications.nonequilibrium-field",
        "applications.sign-problem",
    }.isdisjoint({reference.owner_id for reference in aggregation.campaigns})
    frontier = condensed_matter_frontier_candidate_campaigns()
    assert (
        CondensedMatterCampaignAggregation.from_record(
            frontier.to_record()
        ).aggregation_id
        == frontier.aggregation_id
    )
    assert {
        "chemistry.periodic.lattice-frontier",
        "applications.functional-rg",
        "applications.diagrammatic-field",
        "applications.nonequilibrium-field",
        "applications.sign-problem",
    } <= {reference.owner_id for reference in frontier.campaigns}
    for reference in aggregation.campaigns:
        role_membership = {
            role.name: set(role.case_ids) for role in reference.campaign.roles
        }
        assert role_membership["calibration"]
        assert role_membership["locked_evaluation"]
        assert role_membership["calibration"].isdisjoint(
            role_membership["locked_evaluation"]
        )

    with pytest.raises(ValueError, match="crosses roles"):
        CondensedMatterCampaignAggregation(
            (
                CondensedMatterCampaignReference(
                    "owner.one", _campaign("one", "shared-unit", "one-locked-unit")
                ),
                CondensedMatterCampaignReference(
                    "owner.two", _campaign("two", "two-calibration-unit", "shared-unit")
                ),
            )
        )


def test_orchestrator_retains_every_component_failure(monkeypatch):
    returncodes = iter((0, 7, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0))
    invoked = []

    def execute(command, **kwargs):
        invoked.append(command)
        return SimpleNamespace(returncode=next(returncodes))

    monkeypatch.setattr(closure_orchestration.subprocess, "run", execute)
    result = closure_orchestration.run("qualification")

    assert len(invoked) == 12
    assert not result["passed"]
    assert [component["returncode"] for component in result["components"]] == [
        0,
        7,
        0,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
