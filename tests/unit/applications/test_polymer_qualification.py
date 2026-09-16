import phydrax as phx


def test_polymer_capabilities_are_separate_unreleased_profiles_and_campaigns():
    supports = phx.applications.soft_matter_support_tuples()
    profiles = phx.applications.soft_matter_candidate_profiles()
    campaigns = phx.applications.soft_matter_candidate_campaigns()
    capabilities = {support.capability for support in supports}
    expected = {
        "soft-matter.polymer-particle-kremer-grest",
        "soft-matter.polymer-prism-hnc",
        "soft-matter.polymer-scft-fixed-cell",
        "soft-matter.polymer-fts-partial-saddle",
        "soft-matter.polymer-construction",
        "soft-matter.polymer-reaction-epochs",
        "soft-matter.chromatin-atomistic-coupling",
        "soft-matter.overdamped-atomistic",
        "soft-matter.generalized-langevin-atomistic",
        "soft-matter.polymer-equilibrium-rheology",
    }
    assert expected <= capabilities
    assert len(profiles) == len(supports) == len(campaigns)
    assert all(not profile.released for profile in profiles)
    for campaign in campaigns:
        roles = {role.name: role for role in campaign.roles}
        calibration = roles["calibration"]
        locked = roles["locked_evaluation"]
        assert set(calibration.case_ids).isdisjoint(locked.case_ids)
