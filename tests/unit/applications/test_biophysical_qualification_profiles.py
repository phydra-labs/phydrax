#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.applications._biophysical_qualification import (
    biophysical_candidate_profile,
    biophysical_candidate_profiles,
)


def test_biophysical_profiles_are_narrow_unreleased_and_evidence_free():
    profiles = biophysical_candidate_profiles()

    assert len(profiles) == 4
    assert [profile.name for profile in profiles] == sorted(
        profile.name for profile in profiles
    )
    assert all(not profile.released for profile in profiles)
    assert all(not profile.release_evidence for profile in profiles)
    assert all(profile.required_gates for profile in profiles)
    assert all(len(profile.support_tuples) == 1 for profile in profiles)


def test_coordinate_profile_refuses_an_equilibrium_claim():
    profile = biophysical_candidate_profile(
        "protein.coordinate-proposal.fixed-construct-standard-chemistry.v1"
    )
    support = dict(profile.support_tuples[0].attributes)

    assert support["equilibrium"] == "not-claimed"
    assert support["generalization"] == "fixed-construct-only"
    assert set(profile.required_gates) == {
        "source-admission",
        "numerical-validity",
        "chemical-validity",
    }


def test_unknown_biophysical_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown biophysical capability profile"):
        biophysical_candidate_profile("protein-folding-qualified")
