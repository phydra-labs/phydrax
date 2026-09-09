#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.applications.battery import _qualification as qualification
from phydrax.applications.battery._pack_entry import evaluate_series_pack_entry_gate
from phydrax.qualification import (
    CapabilityProfile,
    HMACSHA256ReleaseSigner,
    HMACSHA256TrustPolicy,
    ReleaseGateEvidence,
    ReleaseIndex,
    SupportTuple,
)


def test_candidate_registry_rejects_a_prefix_compatible_circuit_substitution():
    attributes = dict(qualification.CIRCUIT_ECM_SUPPORT.attributes)
    attributes["model_id"] = "battery:ecm:circuit-connected-unqualified"
    substituted = SupportTuple("battery.simulation", attributes)
    profile = CapabilityProfile(
        "battery.circuit-ecm.candidate", "phydrax", "candidate", (substituted,)
    )
    with pytest.raises(ValueError):
        qualification.validate_battery_candidate_profile(profile, substituted)


def test_gate_names_and_signed_opaque_ids_cannot_authorize_pack_entry():
    gates = tuple(
        ReleaseGateEvidence(
            name,
            passed=True,
            evidence_ids=(f"opaque:{name}",),
            reviewer_id="test-reviewer",
            issued_at=10,
            expires_at=90,
        )
        for name in qualification.SERIES_PACK_REQUIRED_GATES
    )
    profile = CapabilityProfile(
        "battery.circuit-ecm",
        "phydrax",
        "numerical-qualified",
        (qualification.CIRCUIT_ECM_SUPPORT,),
        required_gates=qualification.SERIES_PACK_REQUIRED_GATES,
        release_evidence=gates,
        released=True,
    )
    signer = HMACSHA256ReleaseSigner("test-only", b"test-only-secret")
    trust = HMACSHA256TrustPolicy(
        {"test-only": b"test-only-secret"},
        maximum_index_age=100,
        maximum_evidence_age=100,
    )
    index = ReleaseIndex.sign((profile,), signer, issued_at=20)
    decision = evaluate_series_pack_entry_gate(
        release_index=index,
        profile_id=profile.profile_id,
        trust_policy=trust,
        assessment=None,
        at_time=30,
    )
    assert not decision.eligible
    assert not decision.conclusive
