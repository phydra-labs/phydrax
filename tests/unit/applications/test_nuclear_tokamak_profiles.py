import phydrax as phx


def test_nuclear_tokamak_and_reactor_profiles_remain_unreleased_and_scoped():
    profiles = (
        *phx.nuclear.nuclear_candidate_profiles(),
        *phx.applications.tokamak.tokamak_candidate_profiles(),
        *phx.applications.reactor_physics.reactor_candidate_profiles(),
    )

    assert len({profile.profile_id for profile in profiles}) == len(profiles)
    assert all(not profile.released for profile in profiles)
    assert all(profile.version == "candidate" for profile in profiles)
    assert all(profile.required_gates for profile in profiles)
    assert {profile.capability for profile in profiles} >= {
        "nuclear.activation",
        "nuclear.fusion-reaction",
        "tokamak.core-transport",
        "tokamak.grad-shafranov",
        "reactor.neutron-diffusion",
        "reactor.point-kinetics",
    }
