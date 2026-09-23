#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import copy
from dataclasses import replace

import jax.random as jr
import pytest

import phydrax as phx
from phydrax.privacy._provider import _prepare_private_gradient


pytest.importorskip("dp_accounting")
pytest.importorskip("jax_privacy")


def _prepared(*, epsilon: float = 3.0, iterations: int = 3):
    definition = phx.privacy.PrivacyDefinition(
        phx.privacy.PrivacyUnit("operator-case"),
        phx.privacy.NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    scope = phx.privacy.PrivateDataScope("private-study", definition)
    budget = phx.privacy.PrivacyBudget(epsilon, 1e-6)
    training = phx.privacy.PrivateTrainingPlan(
        phx.privacy.DPSGDPlan(
            scope,
            budget,
            sampling_probability=0.1,
            iterations=iterations,
            clipping_norm=1.0,
            normalize_by=2.0,
            dtype="float32",
        )
    )
    return _prepare_private_gradient(
        training,
        noise_key=jr.key(2),
        sampler_seed=3,
    )


def test_accounting_event_certificate_and_release_ledger_are_content_verified():
    prepared = _prepared()
    trace = prepared.trace(3)
    event = phx.privacy.dp_event_from_record(trace.event_record)
    invalid_event = dict(trace.event_record)
    invalid_event["unexpected"] = True
    with pytest.raises(ValueError, match="canonical contract"):
        phx.privacy.dp_event_from_record(invalid_event)
    assert phx.privacy.dp_event_to_record(event) == trace.event_record

    certificate = prepared.certificate(3)
    assert certificate.guarantee.epsilon <= prepared.training_plan.budget.epsilon
    assert certificate.guarantee.trace_id == trace.trace_id
    assert certificate.mechanism_plan_id == prepared.prepared_id
    assert certificate.query_l2_sensitivity == 0.5
    assert certificate.planned_iterations == 3
    assert certificate.qualification_profile_id is not None
    assert not certificate.public_release_allowed
    with pytest.raises(PermissionError, match="not qualified"):
        certificate.require_public_release()
    with pytest.raises(ValueError, match="public-release authorized"):
        replace(
            certificate,
            qualification_profile_id="self-attested-profile",
            qualification_released=True,
        )

    restored = phx.privacy.PrivacyCertificate.from_record(certificate.to_record())
    assert restored == certificate
    tampered = copy.deepcopy(certificate.to_record())
    tampered["provider_id"] = "tampered-provider"
    with pytest.raises(ValueError, match="content address"):
        phx.privacy.PrivacyCertificate.from_record(tampered)
    unknown = copy.deepcopy(certificate.to_record())
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="canonical contract"):
        phx.privacy.PrivacyCertificate.from_record(unknown)

    ledger = phx.privacy.PrivacyReleaseLedger(
        prepared.training_plan.scope,
        prepared.training_plan.budget,
    )
    charged = ledger.register("model-root", certificate)
    assert charged.register("model-root", certificate) is charged
    assert charged.spent is not None
    assert charged.spent.epsilon == certificate.guarantee.epsilon
    assert (
        phx.privacy.PrivacyReleaseLedger.from_record(charged.to_record(), previous=ledger)
        == charged
    )
    with pytest.raises(ValueError, match="exact previous head"):
        phx.privacy.PrivacyReleaseLedger.from_record(charged.to_record())
    fork_record = copy.deepcopy(charged.to_record())
    fork_record.pop("ledger_id")
    fork_record["receipts"][0].pop("receipt_id")
    fork_record["receipts"][0]["release_root_id"] = "forked-root"
    with pytest.raises(ValueError, match="monotonically extend"):
        phx.privacy.PrivacyReleaseLedger.from_record(fork_record, previous=charged)

    with pytest.raises(ValueError, match="exceeds budget"):
        charged.register("independent-root", certificate)
    assert len(charged.receipts) == 1
    charged.require_successor(ledger)
    with pytest.raises(ValueError, match="monotonically extend"):
        ledger.require_successor(charged)
    with pytest.raises(ValueError, match="monotonically extend"):
        phx.privacy.PrivacyReleaseLedger.from_record(ledger.to_record(), previous=charged)

    forged_guarantee = replace(
        certificate.guarantee,
        epsilon=certificate.guarantee.epsilon / 2.0,
    )
    with pytest.raises(ValueError, match="exact native accounting"):
        replace(certificate, guarantee=forged_guarantee)


def test_privacy_records_reject_coerced_counts_and_flags():
    prepared = _prepared()
    definition_record = prepared.training_plan.scope.definition.to_record()
    definition_record.pop("definition_id")
    definition_record["population_size_public"] = "false"
    with pytest.raises(TypeError, match="must be a boolean"):
        phx.privacy.PrivacyDefinition.from_record(definition_record)

    with pytest.raises(TypeError, match="repetitions must be an integer"):
        phx.privacy.MechanismTrace(prepared.per_step_trace.event_json, 1.5)
    with pytest.raises(TypeError, match="iterations must be an integer"):
        replace(prepared.training_plan.mechanism, iterations=1.5)


def test_private_plan_rejects_mismatched_definition_and_unsafe_policies():
    definition = phx.privacy.PrivacyDefinition(
        phx.privacy.PrivacyUnit("subject"),
        phx.privacy.NeighboringRelation.REPLACE_ONE,
    )
    scope = phx.privacy.PrivateDataScope("subjects", definition)
    with pytest.raises(ValueError, match="add/remove-one"):
        phx.privacy.DPSGDPlan(
            scope,
            phx.privacy.PrivacyBudget(2.0, 1e-6),
            sampling_probability=0.1,
            iterations=2,
            clipping_norm=1.0,
        )

    valid = _prepared().training_plan.mechanism
    rdp = _prepare_private_gradient(
        phx.privacy.PrivateTrainingPlan(
            replace(valid, accounting_method=phx.privacy.AccountingMethod.RDP)
        ),
        noise_key=jr.key(5),
        sampler_seed=6,
    )
    assert "rdp-default-orders" in rdp.planned_guarantee.accountant_id
    assert rdp.planned_guarantee.epsilon <= rdp.training_plan.budget.epsilon
    with pytest.raises(ValueError, match="microbatch_size=1"):
        replace(valid, microbatch_size=2)
    with pytest.raises(ValueError, match="normalization"):
        phx.privacy.PrivateTrainingPlan(valid, normalization_is_public=False)
    with pytest.raises(ValueError, match="metric release"):
        phx.privacy.PrivateTrainingPlan(valid, release_private_metrics=True)
