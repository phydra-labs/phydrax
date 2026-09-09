#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.applications.battery import _qualification as battery_qualification
from phydrax.qualification import (
    CapabilityProfile,
    ReferenceArtifactManifest,
    SupportTuple,
)


def test_candidate_profile_binding_rejects_unrecognized_releases():
    support = battery_qualification.THERMAL_ECM_SUPPORT
    candidate = CapabilityProfile(
        "battery.test-candidate",
        "phydrax-tests",
        "candidate",
        (support,),
        released=False,
    )
    with pytest.raises(ValueError):
        battery_qualification.validate_battery_candidate_profile(candidate, support)

    other = SupportTuple(
        "battery.simulation",
        {"model": "other", "control": "prescribed-current"},
    )
    with pytest.raises(ValueError):
        battery_qualification.validate_battery_candidate_profile(candidate, other)
    released = CapabilityProfile(
        "battery.test-release",
        "phydrax-tests",
        "release",
        (support,),
        released=True,
    )
    with pytest.raises(ValueError):
        battery_qualification.validate_battery_candidate_profile(released, support)


def test_reference_artifact_rights_and_identity_stay_generic():
    manifest = ReferenceArtifactManifest(
        "test-reference",
        checksum_algorithm="sha256",
        checksum="a" * 64,
        size_bytes=128,
        license_id="test-license",
        commercial_use_permitted=True,
        redistribution_permitted=False,
        training_use_permitted=True,
        export_permitted=False,
        export_classification="restricted",
        nondimensionalization={"voltage": 1.0},
        uncertainty={"voltage": 0.01},
        lineage_ids=("source:test",),
    )
    assert battery_qualification.artifact_identity(manifest) == manifest.manifest_id
    assert (
        manifest.require_rights(commercial_use=True, training_use=True)
        == manifest.manifest_id
    )
    with pytest.raises(PermissionError, match="redistribution-not-permitted"):
        manifest.require_rights(redistribution=True)
