#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from phydrax._fingerprint import canonical_fingerprint, canonical_json
from phydrax.applications.battery._admission import (
    battery_release_deployment_record,
    issue_battery_execution_admission,
    load_battery_execution_admission,
    validate_battery_execution_admission,
)
from phydrax.applications.battery._qualification import (
    THERMAL_ECM_CANDIDATE,
    THERMAL_ECM_SUPPORT,
    validate_battery_candidate_profile,
)
from phydrax.applications.battery._release import (
    BatteryReleaseBundle,
    BatteryReleaseRecord,
    build_battery_release,
    CleanReplayComparison,
    REQUIRED_RESOURCE_MEASUREMENTS,
)
from phydrax.applications.battery._release_contracts import battery_release_contract
from phydrax.applications.battery._validity import THERMAL_ECM_ENVELOPE
from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    CapabilityProfile,
    ObservedResourceRecord,
    QualificationCriterion,
    QualificationEvidence,
    QualificationMatrix,
    ReleaseIndex,
    SupportDependency,
)
from phydrax.qualification._promotion import (
    advance_channel,
    PromotionConflictError,
    PromotionRepository,
)
from phydrax.qualification._runtime_distribution import RuntimeDistributionAttestation
from phydrax.qualification._trust import (
    AsymmetricReleaseSigner,
    AsymmetricReleaseTrustPolicy,
    QualificationRoleTrust,
    SignedQualificationRecord,
)
from phydrax.service._security import (
    Ed25519Signer,
    Ed25519Verifier,
    SigningKeyTrustRecord,
    SigningTrustStore,
)


TEST_SOURCE_BUILD_ID = canonical_fingerprint({"test-only-source-closure": 1})
TEST_DISTRIBUTION_CONTENT = {
    "kind": "battery-unpromoted-distribution",
    "source_build_id": TEST_SOURCE_BUILD_ID,
    "freeze_id": canonical_fingerprint({"test-only-freeze": 1}),
    "sbom_id": canonical_fingerprint({"test-only-sbom": 1}),
    **{
        name: {
            "filename": f"test-only.{name}",
            "size_bytes": 1,
            "sha256": canonical_fingerprint({"test-only-artifact": name}),
        }
        for name in ("wheel", "sdist", "sbom")
    },
}
TEST_DISTRIBUTION_ID = canonical_fingerprint(TEST_DISTRIBUTION_CONTENT)
TEST_DISTRIBUTION_MANIFEST_JSON = canonical_json(
    {**TEST_DISTRIBUTION_CONTENT, "distribution_id": TEST_DISTRIBUTION_ID}
)


@pytest.fixture
def release_fixture(monkeypatch):
    pytest.importorskip("cryptography")
    role_keys = {
        "criterion-approver": ("approve",),
        "executor": ("execute", "replay-execute"),
        "scientific-reviewer": ("review", "replay-review"),
        "entry-decision-authority": ("entry",),
        "release-authority": ("release",),
        "channel-promoter": ("promote",),
    }
    keys = [key for values in role_keys.values() for key in values]
    signers = {
        key: Ed25519Signer(key, bytes([position + 1]) * 32)
        for position, key in enumerate(keys)
    }
    store = SigningTrustStore()
    for key, signer in signers.items():
        store.trust(
            SigningKeyTrustRecord(key, "Ed25519", 0, 1000),
            Ed25519Verifier(key, signer.public_key_bytes),
        )
    roles = QualificationRoleTrust(store, role_keys)
    policy = AsymmetricReleaseTrustPolicy(roles, max_index_age=100)
    attestations = []

    def sign(record, key, role, issued):
        value = SignedQualificationRecord.sign(
            record, signers[key], role=role, issued_at=issued, expires_at=300
        )
        attestations.append(value)
        return value

    support = THERMAL_ECM_SUPPORT.support_tuple_id
    model = THERMAL_ECM_ENVELOPE.model_id
    # Synthetic finite bounds are registered only in this isolated governance test.
    # The shipped candidate envelopes deliberately lack approved production limits.
    bounded_envelope = replace(
        THERMAL_ECM_ENVELOPE,
        production_parameter_bounds=(("['capacity']", 1.0, 10.0),),
        production_initial_bounds=(("['charge']", 1.0, 5.0),),
        production_protocol_bounds=(("['current']", -1.0, 1.0),),
        production_layout_bounds=(
            ("plan.branch_count", 1, 1),
            ("protocol.duration_s", 0.0, 100.0),
        ),
        resource_limits=tuple(
            (name, 1000.0) for name in sorted(REQUIRED_RESOURCE_MEASUREMENTS)
        ),
    )
    monkeypatch.setattr(
        "phydrax.applications.battery._release.BATTERY_NUMERICAL_ENVELOPES",
        (bounded_envelope,),
    )
    contract = battery_release_contract(model)
    replay_limits = {key: 0.0 for key in contract.replay_keys}
    requirements = contract.requirements(bounded_envelope, replay_limits)
    criteria = tuple(
        QualificationCriterion(
            support_tuple_id=support,
            metric=metric.name,
            unit=metric.unit,
            comparison=metric.comparison,
            target=metric.target_limit,
            aggregation=metric.aggregation,
            uncertainty=metric.uncertainty,
            applicability=case,
            approval_id="approve",
            issued_at=1,
            valid_until=250,
        )
        for (case, _), (metric, _) in requirements.items()
    )
    for criterion in criteria:
        sign(criterion, "approve", "criterion-approver", 1)
    raw_records = []
    resources = []

    def execution(replay):
        offset = 30 if replay else 0
        executor, reviewer = (
            ("replay-execute", "replay-review") if replay else ("execute", "review")
        )
        raw = {
            "kind": "battery-qualification-raw-output",
            "execution": executor,
            "metrics": {
                f"{criterion.applicability}/{criterion.metric}": {
                    "value": criterion.target if criterion.comparison == "equal" else 0.0
                }
                for criterion in criteria
            },
            "infrastructure_failures": [],
        }
        raw_id = canonical_fingerprint(raw)
        raw_records.append(canonical_json({**raw, "raw_artifact_id": raw_id}))
        resource = ObservedResourceRecord(
            support,
            TEST_SOURCE_BUILD_ID,
            "environment:cpu",
            backend="cpu",
            topology=model,
            measurements={name: 0 for name in REQUIRED_RESOURCE_MEASUREMENTS},
            observed_at=20 + offset,
            raw_artifact_ids=(raw_id,),
        )
        resources.append(resource)
        sign(resource, executor, "executor", 20 + offset)
        starts, observations, evidence = [], [], []
        for criterion in criteria:
            kind = requirements[(criterion.applicability, criterion.metric)][1]
            start = CampaignStartRecord(
                campaign_spec_id=f"campaign:{executor}",
                criterion_id=criterion.criterion_id,
                resolved_run_spec_id=f"run:{executor}",
                support_tuple_id=support,
                started_at=10 + offset,
            )
            observation = CampaignObservationRecord(
                start_record_id=start.start_record_id,
                campaign_spec_id=start.campaign_spec_id,
                criterion_id=criterion.criterion_id,
                resolved_run_spec_id=start.resolved_run_spec_id,
                support_tuple_id=support,
                raw_artifact_ids=(raw_id,),
                observed_at=20 + offset,
            )
            outcome = QualificationEvidence(
                kind,
                "passed",
                (THERMAL_ECM_CANDIDATE.profile_id, support),
                build_id=TEST_SOURCE_BUILD_ID,
                environment_id="environment:cpu",
                backend="cpu",
                topology=model,
                precision="float64",
                reduction="deterministic",
                replay_id="replay:precommitted",
                criteria_ids=(criterion.criterion_id,),
                raw_artifact_ids=(raw_id,),
                campaign_start_record_ids=(start.start_record_id,),
                campaign_observation_record_ids=(observation.observation_record_id,),
                observed_resource_record_ids=(resource.record_id,),
                reviewer_id=reviewer,
                issued_at=30 + offset,
                expires_at=240,
                reason="measured-residual-within-approved-bound",
            )
            starts.append(start)
            observations.append(observation)
            evidence.append(outcome)
            sign(start, executor, "executor", start.started_at)
            sign(observation, executor, "executor", observation.observed_at)
            sign(outcome, executor, "executor", outcome.issued_at)
            sign(outcome, reviewer, "scientific-reviewer", outcome.issued_at)
        return tuple(starts), tuple(observations), tuple(evidence)

    starts, observations, evidence = execution(False)
    replay_starts, replay_observations, replay_evidence = execution(True)
    replay = CleanReplayComparison(
        contract.replay_case_ids,
        (),
        tuple(replay_limits.items()),
        tuple(replay_limits.items()),
        replay_starts,
        replay_observations,
        replay_evidence,
        True,
        False,
        False,
        False,
    )
    sign(replay, "replay-execute", "executor", 60)
    sign(replay, "replay-review", "scientific-reviewer", 60)
    matrix = QualificationMatrix(
        {
            f"{criterion.applicability}/{criterion.metric}": {
                "evidence_kind": item.evidence_kind,
                "criterion_id": criterion.criterion_id,
                "subject_id": support,
                "build_id": TEST_SOURCE_BUILD_ID,
                "topology": model,
            }
            for criterion, item in zip(criteria, evidence, strict=True)
        }
    )
    coverage = matrix.evaluate(evidence, at_time=70)
    sign(coverage, "review", "scientific-reviewer", 70)
    bundle = BatteryReleaseBundle(
        bounded_envelope,
        TEST_DISTRIBUTION_ID,
        TEST_DISTRIBUTION_MANIFEST_JSON,
        criteria,
        starts,
        observations,
        evidence,
        (),
        (),
        (),
        tuple(resources),
        replay,
        matrix,
        coverage,
        tuple(raw_records),
        (),
    )
    sign(bundle.approval_record(support), "approve", "criterion-approver", 2)
    bundle = replace(bundle, attestations=tuple(attestations))
    proof = build_battery_release(
        THERMAL_ECM_CANDIDATE, bundle, trust_policy=policy, at_time=80, expires_at=220
    )
    policy.proofs = (proof,)
    index = ReleaseIndex.sign(
        (proof.profile,),
        AsymmetricReleaseSigner(signers["release"], issued_at=90),
        issued_at=90,
    )
    return bundle, proof, policy, index, signers


def _admit(fixture):
    _, proof, policy, index, signers = fixture
    runtime = RuntimeDistributionAttestation.attest_verified_install(
        TEST_DISTRIBUTION_ID, signers["execute"], issued_at=95, expires_at=200
    )
    return issue_battery_execution_admission(
        index,
        proof.profile.profile_id,
        THERMAL_ECM_SUPPORT,
        policy,
        envelope=proof.bundle.envelope,
        distribution_id=TEST_DISTRIBUTION_ID,
        at_time=100,
        expires_at=180,
        runtime_attestation=runtime,
    )


def test_retained_proof_roundtrip_and_equation_only_admission(release_fixture):
    _, proof, policy, _, _ = release_fixture
    recovered = BatteryReleaseRecord.from_record(proof.to_record())
    recovered.verify(policy, at_time=100)
    assert recovered.bundle.source_build_id == TEST_SOURCE_BUILD_ID
    assert recovered.distribution_id == TEST_DISTRIBUTION_ID
    assert recovered.bundle.source_build_id != recovered.distribution_id
    token = _admit(release_fixture)
    assert (
        validate_battery_execution_admission(
            proof.profile,
            THERMAL_ECM_SUPPORT,
            token,
            distribution_id=TEST_DISTRIBUTION_ID,
            at_time=100,
        )
        is token
    )
    assert token.predictive_admission_id is None
    with pytest.raises(ValueError):
        token.validate_adapter(
            SimpleNamespace(model_id=THERMAL_ECM_ENVELOPE.model_id, equation_form="ode")
        )
    with pytest.raises(ValueError):
        validate_battery_execution_admission(
            THERMAL_ECM_CANDIDATE,
            THERMAL_ECM_SUPPORT,
            token,
            distribution_id=None,
            at_time=100,
        )


@pytest.mark.parametrize("mutation", ("manifest-content", "different-source"))
def test_distribution_mapping_cannot_relabel_scientific_evidence(
    release_fixture, mutation
):
    bundle, _, policy, _, signers = release_fixture
    manifest = json.loads(bundle.distribution_manifest_json)
    manifest["source_build_id"] = canonical_fingerprint({"test-only-different-source": 1})
    if mutation == "different-source":
        manifest["distribution_id"] = canonical_fingerprint(
            {key: value for key, value in manifest.items() if key != "distribution_id"}
        )
    changed = replace(
        bundle,
        distribution_id=manifest["distribution_id"],
        distribution_manifest_json=canonical_json(manifest),
    )
    if mutation == "different-source":
        approval = SignedQualificationRecord.sign(
            changed.approval_record(THERMAL_ECM_SUPPORT.support_tuple_id),
            signers["approve"],
            role="criterion-approver",
            issued_at=2,
            expires_at=300,
        )
        changed = replace(changed, attestations=(*changed.attestations, approval))
    with pytest.raises(ValueError):
        build_battery_release(
            THERMAL_ECM_CANDIDATE,
            changed,
            trust_policy=policy,
            at_time=80,
            expires_at=200,
        )


@pytest.mark.parametrize(
    "mutation",
    ("tamper", "stale", "revoked", "distribution", "dependency", "replay", "opaque"),
)
def test_production_admission_fails_closed(release_fixture, mutation):
    bundle, proof, policy, _, _ = release_fixture
    token = _admit(release_fixture)
    if mutation == "tamper":
        signature = bundle.attestations[0]
        changed = replace(signature, content_json="{}")
        policy.proofs = (
            replace(
                proof,
                bundle=replace(bundle, attestations=(changed, *bundle.attestations[1:])),
            ),
        )
    elif mutation == "revoked":
        policy.roles.store.revoke("execute", 110)
    elif mutation == "dependency":
        policy.proofs = (
            replace(
                proof,
                bundle=replace(
                    bundle,
                    dependencies=(
                        SupportDependency(
                            "missing", THERMAL_ECM_SUPPORT.support_tuple_id
                        ),
                    ),
                ),
            ),
        )
    elif mutation == "replay":
        policy.proofs = (
            replace(
                proof,
                bundle=replace(bundle, replay=replace(bundle.replay, reused_caches=True)),
            ),
        )
    elif mutation == "opaque":
        policy.proofs = ()
    with pytest.raises((ValueError, RuntimeError)):
        validate_battery_execution_admission(
            proof.profile,
            THERMAL_ECM_SUPPORT,
            token,
            distribution_id="distribution:other"
            if mutation == "distribution"
            else TEST_DISTRIBUTION_ID,
            at_time=180 if mutation == "stale" else 100,
        )


def test_expiry_is_capped_by_typed_evidence_and_unknown_candidates_refused(
    release_fixture,
):
    bundle, _, policy, _, _ = release_fixture
    proof = build_battery_release(
        THERMAL_ECM_CANDIDATE, bundle, trust_policy=policy, at_time=80, expires_at=900
    )
    assert proof.expires_at == 240
    forged = CapabilityProfile(
        "battery.thermal-ecm.other", "phydrax", "candidate", (THERMAL_ECM_SUPPORT,)
    )
    with pytest.raises(ValueError):
        validate_battery_candidate_profile(forged, THERMAL_ECM_SUPPORT)
    with pytest.raises(TypeError):
        build_battery_release(
            THERMAL_ECM_CANDIDATE,
            "opaque-evidence-id",
            trust_policy=policy,
            at_time=80,
            expires_at=90,
        )


def test_unbounded_candidate_domain_cannot_be_promoted(release_fixture):
    bundle, _, policy, _, _ = release_fixture
    unbounded = replace(bundle, envelope=THERMAL_ECM_ENVELOPE)
    with pytest.raises(ValueError):
        build_battery_release(
            THERMAL_ECM_CANDIDATE,
            unbounded,
            trust_policy=policy,
            at_time=80,
            expires_at=200,
        )


def test_signed_self_consistent_subset_cannot_omit_physical_criteria(release_fixture):
    bundle, _, policy, _, signers = release_fixture
    omitted = next(
        item.criterion_id
        for item in bundle.criteria
        if item.metric == "maximum-normalized-analytic-residual"
    )
    criteria = tuple(item for item in bundle.criteria if item.criterion_id != omitted)
    evidence = tuple(item for item in bundle.evidence if item.criteria_ids != (omitted,))
    predicates = {
        name: dict(predicate)
        for name, predicate in bundle.matrix.predicates
        if dict(predicate)["criterion_id"] != omitted
    }
    matrix = QualificationMatrix(predicates)
    coverage = matrix.evaluate(evidence, at_time=70)
    assert coverage.passed
    assert {item.evidence_kind for item in evidence} == {
        "scientific",
        "performance",
        "operational",
    }
    replay = replace(
        bundle.replay,
        starts=tuple(
            item for item in bundle.replay.starts if item.criterion_id != omitted
        ),
        observations=tuple(
            item for item in bundle.replay.observations if item.criterion_id != omitted
        ),
        evidence=tuple(
            item for item in bundle.replay.evidence if item.criteria_ids != (omitted,)
        ),
    )
    subset = replace(
        bundle,
        criteria=criteria,
        evidence=evidence,
        matrix=matrix,
        coverage=coverage,
        replay=replay,
        starts=tuple(item for item in bundle.starts if item.criterion_id != omitted),
        observations=tuple(
            item for item in bundle.observations if item.criterion_id != omitted
        ),
    )
    # Re-approve the internally consistent subset with legitimate test authorities:
    # signatures and caller-relative coverage are not the omitted physical proof.
    signatures = (
        SignedQualificationRecord.sign(
            subset.approval_record(THERMAL_ECM_SUPPORT.support_tuple_id),
            signers["approve"],
            role="criterion-approver",
            issued_at=2,
            expires_at=300,
        ),
        SignedQualificationRecord.sign(
            coverage,
            signers["review"],
            role="scientific-reviewer",
            issued_at=70,
            expires_at=300,
        ),
        SignedQualificationRecord.sign(
            replay,
            signers["replay-execute"],
            role="executor",
            issued_at=60,
            expires_at=300,
        ),
        SignedQualificationRecord.sign(
            replay,
            signers["replay-review"],
            role="scientific-reviewer",
            issued_at=60,
            expires_at=300,
        ),
    )
    subset = replace(subset, attestations=(*bundle.attestations, *signatures))
    with pytest.raises(ValueError):
        build_battery_release(
            THERMAL_ECM_CANDIDATE, subset, trust_policy=policy, at_time=80, expires_at=200
        )


def test_whole_run_jit_cannot_cache_time_scoped_production_authorization(release_fixture):
    import jax
    import jax.numpy as jnp

    _, proof, _, _, _ = release_fixture
    token = _admit(release_fixture)

    def escaped_host_dispatch():
        validate_battery_execution_admission(
            proof.profile,
            THERMAL_ECM_SUPPORT,
            token,
            distribution_id=TEST_DISTRIBUTION_ID,
            at_time=100,
        )
        return jnp.asarray(1.0)

    with pytest.raises(ValueError):
        jax.jit(escaped_host_dispatch).lower()


def test_changed_imported_runtime_cannot_reuse_old_admission(
    release_fixture, monkeypatch
):
    _, proof, _, _, _ = release_fixture
    token = _admit(release_fixture)
    changed = list(token.runtime_attestation.package_files)
    path, digest, size = changed[0]
    changed[0] = (path, "0" * 64 if digest != "0" * 64 else "1" * 64, size)
    monkeypatch.setattr(
        "phydrax.qualification._runtime_distribution._runtime_package_files",
        lambda: tuple(changed),
    )
    with pytest.raises(ValueError):
        validate_battery_execution_admission(
            proof.profile,
            THERMAL_ECM_SUPPORT,
            token,
            distribution_id=TEST_DISTRIBUTION_ID,
            at_time=100,
        )


def test_offline_deployment_requires_retained_proofs_and_separate_public_roots(
    release_fixture,
):
    _, proof, policy, index, signers = release_fixture
    token = _admit(release_fixture)
    roots = {
        "kind": "qualification-public-trust",
        "roles": {role: sorted(keys) for role, keys in policy.roles.roles.items()},
        "keys": [
            {
                "record": asdict(record),
                "public_key_hex": signers[record.key_id].public_key_bytes.hex(),
            }
            for record in policy.roles.store.records()
        ],
    }
    deployment = battery_release_deployment_record(index, (proof,))
    loaded = load_battery_execution_admission(
        deployment,
        roots,
        profile_id=proof.profile.profile_id,
        support_tuple_id=THERMAL_ECM_SUPPORT.support_tuple_id,
        distribution_id=TEST_DISTRIBUTION_ID,
        at_time=100,
        expires_at=180,
        max_index_age=100,
        runtime_attestation=token.runtime_attestation,
    )
    assert loaded.numerical_admission_id == token.numerical_admission_id
    deployment["proofs"] = []
    with pytest.raises(ValueError):
        load_battery_execution_admission(
            deployment,
            roots,
            profile_id=proof.profile.profile_id,
            support_tuple_id=THERMAL_ECM_SUPPORT.support_tuple_id,
            distribution_id=TEST_DISTRIBUTION_ID,
            at_time=100,
            expires_at=180,
            max_index_age=100,
            runtime_attestation=token.runtime_attestation,
        )


def test_channel_withdrawal_rollback_cas_and_signed_empty_refusal(
    release_fixture, tmp_path
):
    _, _, policy, index, signers = release_fixture
    repository = PromotionRepository(tmp_path / "promotions.sqlite")
    promote = advance_channel(
        repository,
        signers["promote"],
        policy,
        channel="production",
        expected_state_id=None,
        action="promote",
        reason="reviewed",
        at_time=100,
        expires_at=180,
        index=index,
        distribution_id=TEST_DISTRIBUTION_ID,
    )
    withdraw = advance_channel(
        repository,
        signers["promote"],
        policy,
        channel="production",
        expected_state_id=promote.state_id,
        action="withdraw",
        reason="maintenance",
        at_time=101,
        expires_at=180,
    )
    assert repository.current("production", policy.roles, at_time=101).index_id is None
    rollback = advance_channel(
        repository,
        signers["promote"],
        policy,
        channel="production",
        expected_state_id=withdraw.state_id,
        action="rollback",
        reason="restore",
        at_time=102,
        expires_at=180,
        index=index,
        distribution_id=TEST_DISTRIBUTION_ID,
    )
    assert rollback.index_id == promote.index_id
    assert rollback.generation == 3
    with pytest.raises(PromotionConflictError):
        repository.compare_and_swap(
            promote, expected_state_id=None, trust_policy=policy, at_time=102, index=index
        )
    reopened = PromotionRepository(tmp_path / "promotions.sqlite")
    with pytest.raises(PromotionConflictError):
        reopened.current("production", policy.roles, at_time=102, minimum_generation=4)
    policy.roles.store.revoke("execute", 103)
    refusal = advance_channel(
        reopened,
        signers["promote"],
        policy,
        channel="production",
        expected_state_id=rollback.state_id,
        action="rollback",
        reason="revoked-proof",
        at_time=103,
        expires_at=180,
        index=index,
        distribution_id=TEST_DISTRIBUTION_ID,
    )
    assert (refusal.action, refusal.index_id, refusal.generation) == ("refuse", None, 4)
    assert (
        reopened.current("production", policy.roles, at_time=103).state_id
        == refusal.state_id
    )
    assert tuple(state.generation for state in reopened.history("production")) == (
        1,
        2,
        3,
        4,
    )
