#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology import _full_dark_sector_claims as claims
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)
from phydrax.qualification import (
    CampaignRole,
    PromotionState,
    QualificationRoleTrust,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
    SignedQualificationRecord,
)
from phydrax.service._security import (
    Ed25519Signer,
    Ed25519Verifier,
    SigningKeyTrustRecord,
    SigningTrustStore,
)


_FACTORIES = (
    ("relativistic-stress-energy-pm", claims.relativistic_stress_energy_pm_claim_profile),
    ("einstein-vlasov-z4c", claims.einstein_vlasov_z4c_claim_profile),
    ("dynamic-epoch-runtime", claims.dynamic_epoch_runtime_claim_profile),
    ("fixed-multiplicity", claims.fixed_multiplicity_claim_profile),
    ("parton-shower", claims.parton_shower_claim_profile),
    ("hadronization", claims.hadronization_claim_profile),
    ("quantum-uu", claims.quantum_uu_claim_profile),
    ("thermal-qft", claims.thermal_qft_claim_profile),
    ("coherent-qke", claims.coherent_qke_claim_profile),
    ("off-shell-kb", claims.off_shell_kb_claim_profile),
    ("radiation-packet", claims.radiation_packet_claim_profile),
    ("radiation-m1", claims.radiation_m1_claim_profile),
    ("radiation-vet", claims.radiation_vet_claim_profile),
    ("radiation-hierarchy", claims.radiation_hierarchy_claim_profile),
    ("fully-coupled-closure", claims.fully_coupled_closure_claim_profile),
)


def _units(*, spin="average-initial-sum-final"):
    return RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 3, 2, 1),
        RelativityConvention(metric_signature="mostly_minus"),
        spin_normalization=spin,
    )


def _frame(units, *, snapshot_token=4):
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(snapshot_token),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="one-cell",
        geometry_lineage_id="flat-lineage",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        source_id="test-observer",
    )
    return LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="test-observer",
        orientation_id="future-right-handed",
    )


def _reference(*, commercial=True, artifact_name="full-dark-sector-reference"):
    return ReferenceArtifactManifest(
        artifact_name,
        checksum_algorithm="sha256",
        checksum="0" * 64,
        size_bytes=1,
        license_id="test-license",
        commercial_use_permitted=commercial,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=("locked-reference",),
    )


def _campaign(criteria):
    calibration = ScientificCase(
        "calibration-case",
        "calibration-unit",
        "dark-sector",
        "calibration-condition",
        "calibration-preparation",
        "calibration-batch",
        ("calibration-source",),
    )
    locked = ScientificCase(
        "locked-case",
        "locked-unit",
        "dark-sector",
        "locked-condition",
        "locked-preparation",
        "locked-batch",
        ("locked-source",),
    )
    return ScientificCampaign(
        (calibration, locked),
        (
            CampaignRole("calibration", (calibration.case_id,)),
            CampaignRole("locked_evaluation", (locked.case_id,)),
        ),
        criteria_ids=tuple(value.criterion_id for value in criteria),
    )


def _profile(profile_name, factory, *, units=None, frame=None, runtime_support=None):
    units_ = _units() if units is None else units
    frame_ = _frame(units_) if frame is None else frame
    metric_ids = claims.full_dark_sector_claim_metric_ids(profile_name)
    criteria = claims.full_dark_sector_claim_criteria(
        profile_name,
        {metric_id: 1.0 for metric_id in metric_ids},
    )
    return factory(
        _campaign(criteria),
        criteria,
        (f"{profile_name}-locked-domain",),
        {
            "backend": "cpu",
            "precision": "float64",
            "maximum-resident-values": 4096,
            "provider-revision": "native-test-revision",
            **({} if runtime_support is None else runtime_support),
        },
        units_,
        frame_,
        reference_artifacts=(_reference(),),
        requested_use=claims.FullDarkSectorReferenceUse(),
    )


@pytest.mark.parametrize(("profile_name", "factory"), _FACTORIES)
def test_all_full_closure_profiles_bind_independent_complete_support(
    profile_name,
    factory,
):
    claim = _profile(profile_name, factory)
    support = dict(claim.support.attributes)
    differentiation = claims.full_dark_sector_differentiation_contract(profile_name)

    assert support["profile"] == profile_name
    assert support["unit_contract_id"]
    assert support["frame_id"]
    assert support["frame_realization_id"]
    assert support["frame_snapshot_token"] == 4
    assert support["differentiation_contract_id"] == differentiation.contract_id
    assert support["fixed_capacity"] is True
    assert support["semantic_unboundedness"] == "durable-finite-epoch-chain"
    assert support["production_inheritance"] is False
    assert support["checkpoint_product"] not in support["analysis_output_products"].split(
        ","
    )
    assert "changed-physics-unit-frame-or-support-identity" in claim.invalidation_triggers
    assert "missing-or-inadmissible-source-rights" in claim.invalidation_triggers
    assert set(value.metric_id for value in claim.criteria) == set(
        claims.full_dark_sector_claim_metric_ids(profile_name)
    )


def test_claim_content_identity_changes_with_runtime_units_and_frame_snapshot():
    name, factory = _FACTORIES[0]
    baseline_units = _units()
    baseline_frame = _frame(baseline_units)
    baseline = _profile(name, factory, units=baseline_units, frame=baseline_frame)
    repeated = _profile(name, factory, units=baseline_units, frame=baseline_frame)
    changed_runtime = _profile(
        name,
        factory,
        units=baseline_units,
        frame=baseline_frame,
        runtime_support={"provider-revision": "second-revision"},
    )
    changed_frame = _profile(
        name,
        factory,
        units=baseline_units,
        frame=_frame(baseline_units, snapshot_token=5),
    )
    changed_units = _units(spin="sum-all")
    changed_normalization = _profile(
        name,
        factory,
        units=changed_units,
        frame=_frame(changed_units),
    )

    assert baseline.claim_id == repeated.claim_id
    assert baseline.claim_id != changed_runtime.claim_id
    assert baseline.claim_id != changed_frame.claim_id
    assert baseline.claim_id != changed_normalization.claim_id
    assert claims.full_dark_sector_promotion_channel(baseline).endswith(baseline.claim_id)


def test_exact_criteria_reserved_support_and_source_rights_are_enforced():
    name, factory = _FACTORIES[3]
    units = _units()
    frame = _frame(units)
    metric_ids = claims.full_dark_sector_claim_metric_ids(name)
    with pytest.raises(ValueError, match="thresholds changed"):
        claims.full_dark_sector_claim_criteria(
            name,
            {metric_id: 1.0 for metric_id in metric_ids[:-1]},
        )

    criteria = claims.full_dark_sector_claim_criteria(
        name,
        {metric_id: 1.0 for metric_id in metric_ids},
    )
    campaign = _campaign(criteria)
    with pytest.raises(ValueError, match="cannot replace governed"):
        factory(
            campaign,
            criteria,
            ("locked-domain",),
            {"backend": "cpu", "unit_contract_id": "forged"},
            units,
            frame,
            reference_artifacts=(_reference(),),
            requested_use=claims.FullDarkSectorReferenceUse(),
        )
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        factory(
            campaign,
            criteria,
            ("locked-domain",),
            {"backend": "cpu"},
            units,
            frame,
            reference_artifacts=(_reference(commercial=False),),
            requested_use=claims.FullDarkSectorReferenceUse(commercial_use=True),
        )


def _promotion_trust():
    pytest.importorskip("cryptography")
    role_keys = {
        "criterion-approver": ("criterion-key",),
        "executor": ("executor-key",),
        "scientific-reviewer": ("reviewer-key",),
        "entry-decision-authority": ("entry-key",),
        "release-authority": ("release-key",),
        "channel-promoter": ("promoter-key",),
    }
    signers = {
        key: Ed25519Signer(key, bytes((index + 1,)) * 32)
        for index, key in enumerate(value[0] for value in role_keys.values())
    }
    store = SigningTrustStore()
    for key, signer in signers.items():
        store.trust(
            SigningKeyTrustRecord(key, "Ed25519", 0, 100),
            Ed25519Verifier(key, signer.public_key_bytes),
        )
    return QualificationRoleTrust(store, role_keys), signers["promoter-key"]


def _signed_promotion(claim, *, channel=None):
    trust, signer = _promotion_trust()
    content = {
        "kind": "promotion-state",
        "channel": claims.full_dark_sector_promotion_channel(claim)
        if channel is None
        else channel,
        "generation": 1,
        "index_id": "release-index",
        "previous_state_id": None,
        "action": "promote",
        "reason": "qualified",
        "approver": signer.key_id,
        "effective_at": 10,
        "expires_at": 90,
        "distribution_id": "distribution",
    }
    signature = SignedQualificationRecord.sign(
        content,
        signer,
        role="channel-promoter",
        issued_at=10,
        expires_at=90,
    )
    promotion = PromotionState(
        **{name: value for name, value in content.items() if name != "kind"},
        signature=signature,
    )
    return promotion, trust


def test_signed_promotion_binds_exact_claim_derivative_gate_and_source_rights():
    name, factory = _FACTORIES[6]
    claim = _profile(name, factory)
    promotion, trust = _signed_promotion(claim)
    differentiation = claims.full_dark_sector_differentiation_contract(name)

    decision = claims.bind_full_dark_sector_promotion(
        claim,
        promotion,
        trust,
        differentiation,
        at_time=20,
        qualification_level="production",
        expected_index_id="release-index",
        reference_artifacts=(_reference(),),
        requested_use=claims.FullDarkSectorReferenceUse(),
    )

    assert decision.admitted
    assert decision.claim.claim_id == claim.claim_id
    assert decision.promotion.state_id == promotion.state_id
    assert decision.differentiation.contract_id == differentiation.contract_id

    with pytest.raises(ValueError, match="derivative gate"):
        claims.bind_full_dark_sector_promotion(
            claim,
            promotion,
            trust,
            phx.DerivativeContract(route=phx.DerivativeRoute.DIRECT),
            at_time=20,
            qualification_level="production",
            expected_index_id="release-index",
            reference_artifacts=(_reference(),),
            requested_use=claims.FullDarkSectorReferenceUse(),
        )

    with pytest.raises(ValueError, match="source manifests"):
        claims.bind_full_dark_sector_promotion(
            claim,
            promotion,
            trust,
            differentiation,
            at_time=20,
            qualification_level="production",
            expected_index_id="release-index",
            reference_artifacts=(_reference(artifact_name="substituted-reference"),),
            requested_use=claims.FullDarkSectorReferenceUse(),
        )

    wrong_promotion, wrong_trust = _signed_promotion(claim, channel="wrong.channel")
    with pytest.raises(ValueError, match="does not bind this exact claim"):
        claims.bind_full_dark_sector_promotion(
            claim,
            wrong_promotion,
            wrong_trust,
            differentiation,
            at_time=20,
            qualification_level="production",
            expected_index_id="release-index",
            reference_artifacts=(_reference(),),
            requested_use=claims.FullDarkSectorReferenceUse(),
        )
