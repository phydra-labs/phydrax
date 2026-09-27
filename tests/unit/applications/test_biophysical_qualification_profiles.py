#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.applications._biophysical_qualification import (
    biophysical_candidate_profile,
    biophysical_candidate_profiles,
)


def test_biophysical_qualification_profiles_scenario_1() -> None:
    profiles = biophysical_candidate_profiles()

    assert {
        "protein.stability.megascale-natural-small-domain.canonical",
        "nucleic.strand-displacement.rna-to-dna.declared-condition.canonical",
        "protein.coordinate-proposal.fixed-construct-standard-chemistry.canonical",
        "rna.ensemble.adenine-riboswitch.declared-protocol.canonical",
    } <= {profile.name for profile in profiles}
    assert [profile.name for profile in profiles] == sorted(
        profile.name for profile in profiles
    )
    assert all(not profile.released for profile in profiles)
    assert all(not profile.release_evidence for profile in profiles)
    assert all(profile.required_gates for profile in profiles)
    assert all(len(profile.support_tuples) == 1 for profile in profiles)
    profile = biophysical_candidate_profile(
        "protein.coordinate-proposal.fixed-construct-standard-chemistry.canonical"
    )
    support = dict(profile.support_tuples[0].attributes)

    assert support["equilibrium"] == "not-claimed"
    assert support["generalization"] == "fixed-construct-only"
    assert set(profile.required_gates) == {
        "source-admission",
        "numerical-validity",
        "chemical-validity",
    }
    with pytest.raises(ValueError, match="Unknown biophysical capability profile"):
        biophysical_candidate_profile("protein-folding-qualified")
