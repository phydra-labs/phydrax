#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import pytest

from phydrax.chemistry._production_qualification import (
    periodic_chemistry_support_tuples,
    production_chemistry_support_tuples,
)
from phydrax.qualification import (
    CapabilityProfile,
    discover_profiles,
    HMACSHA256ReleaseSigner,
    HMACSHA256TrustPolicy,
    ReleaseGateEvidence,
    ReleaseIndex,
    require_profile,
    SupportDependency,
    SupportTuple,
)


def _gate(gate: str) -> ReleaseGateEvidence:
    return ReleaseGateEvidence(
        gate,
        passed=True,
        evidence_ids=(f"test:artifact:{gate}",),
        reviewer_id="test:independent-reviewer",
        issued_at=10,
        expires_at=100,
    )


def _trust() -> tuple[HMACSHA256ReleaseSigner, HMACSHA256TrustPolicy]:
    signer = HMACSHA256ReleaseSigner("test:chemistry-release-signer", b"secret")
    policy = HMACSHA256TrustPolicy(
        {"test:chemistry-release-signer": b"secret"},
        maximum_index_age=100,
        maximum_evidence_age=100,
    )
    return signer, policy


def _support_bytes(profile: CapabilityProfile) -> bytes:
    return json.dumps(
        profile.support_tuples[0].to_record(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_periodic_support_requires_signed_profile_evidence_and_exact_dependencies():
    periodic_support = next(
        item
        for item in periodic_chemistry_support_tuples()
        if item.capability == "chemistry.periodic.pencil.generalized"
    )
    assert all(
        item.support_tuple_id != periodic_support.support_tuple_id
        for item in production_chemistry_support_tuples()
    )

    candidate = CapabilityProfile(
        "chemistry.periodic.pencil.generalized",
        "test-fixture",
        "candidate",
        (periodic_support,),
        released=False,
    )
    signer, policy = _trust()
    candidate_index = ReleaseIndex.sign((candidate,), signer, issued_at=10)
    assert discover_profiles(candidate_index, periodic_support, policy, at_time=20) == ()
    with pytest.raises(ValueError, match="profile-unreleased"):
        require_profile(
            candidate_index,
            candidate.profile_id,
            periodic_support,
            policy,
            at_time=20,
        )

    with pytest.raises(ValueError, match="evidence for every required gate"):
        CapabilityProfile(
            "chemistry.periodic.pencil.generalized",
            "test-fixture",
            "missing-evidence",
            (periodic_support,),
            required_gates=("scientific", "runtime"),
            released=True,
        )

    dependency_support = SupportTuple(
        "operators.periodic.translation-family",
        {
            "representation": "edge-relation-integer-translation-values",
            "runtime": "jax-cpu-single-device-float64",
        },
    )
    dependency = CapabilityProfile(
        "operators.periodic.translation-family",
        "test-fixture",
        "qualified",
        (dependency_support,),
        required_gates=("scientific", "runtime"),
        release_evidence=(_gate("scientific"), _gate("runtime")),
        released=True,
    )
    released = CapabilityProfile(
        "chemistry.periodic.pencil.generalized",
        "test-fixture",
        "qualified",
        (periodic_support,),
        dependencies=(
            SupportDependency(
                dependency.profile_id,
                dependency_support.support_tuple_id,
            ),
        ),
        required_gates=("scientific", "runtime"),
        release_evidence=(_gate("scientific"), _gate("runtime")),
        released=True,
    )

    missing_dependency_index = ReleaseIndex.sign((released,), signer, issued_at=10)
    assert (
        discover_profiles(
            missing_dependency_index,
            periodic_support,
            policy,
            at_time=20,
        )
        == ()
    )
    with pytest.raises(ValueError, match="missing-dependency"):
        require_profile(
            missing_dependency_index,
            released.profile_id,
            periodic_support,
            policy,
            at_time=20,
        )

    released_index = ReleaseIndex.sign(
        (dependency, released),
        signer,
        issued_at=10,
    )
    forged_index = ReleaseIndex(
        released_index.profiles,
        issued_at=released_index.issued_at,
        signer_id=released_index.signer_id,
        signature_algorithm=released_index.signature_algorithm,
        signature="00",
    )
    with pytest.raises(ValueError, match="signature, signer trust, or freshness"):
        discover_profiles(forged_index, periodic_support, policy, at_time=20)

    assert tuple(
        profile.profile_id
        for profile in discover_profiles(
            released_index,
            periodic_support,
            policy,
            at_time=20,
        )
    ) == (released.profile_id,)
    assert (
        require_profile(
            released_index,
            released.profile_id,
            periodic_support,
            policy,
            at_time=20,
        ).profile_id
        == released.profile_id
    )


def test_profile_maturity_does_not_change_periodic_support_tuple_bytes():
    support = periodic_chemistry_support_tuples()[0]
    candidate = CapabilityProfile(
        support.capability,
        "test-fixture",
        "candidate",
        (support,),
        released=False,
    )
    released = CapabilityProfile(
        support.capability,
        "test-fixture",
        "qualified",
        (support,),
        required_gates=("scientific",),
        release_evidence=(_gate("scientific"),),
        released=True,
    )

    assert _support_bytes(candidate) == _support_bytes(released)
    assert (
        not {
            "candidate",
            "maturity",
            "qualification_state",
            "released",
        }
        & dict(support.attributes).keys()
    )
