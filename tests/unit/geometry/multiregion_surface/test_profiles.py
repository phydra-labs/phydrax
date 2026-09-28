from phydrax.applications.foams import foam_candidate_profiles
from phydrax.geometry.multiregion_surface import (
    multiregion_surface_candidate_profiles,
)
from phydrax.qualification import CapabilityProfile


_GEOMETRY_CAPABILITIES = (
    "multiregion-surface.geometry",
    "multiregion-surface.label-field-extraction",
    "multiregion-surface.remeshing",
    "multiregion-surface.topology-events",
)
_FOAM_CAPABILITIES = (
    "foams.quasistatic-equilibrium",
    "foams.thickness-driven-rupture",
    "foams.constrained-dynamics",
    "foams.vortex-sheet-air",
    "foams.plateau-border-drainage",
)
_GEOMETRY_GATES = (
    "analytic-control",
    "documentation-nonclaims",
    "public-workflow",
    "qualification-campaign",
    "topology-evidence",
)
_FOAM_GATES = (
    "analytic-control",
    "documentation-nonclaims",
    "kkt-evidence",
    "public-workflow",
    "qualification-campaign",
)


def test_candidate_profile_ownership_and_gates() -> None:
    geometry = multiregion_surface_candidate_profiles()
    foams = foam_candidate_profiles()

    assert tuple(profile.capability for profile in geometry) == _GEOMETRY_CAPABILITIES
    assert tuple(profile.capability for profile in foams) == _FOAM_CAPABILITIES
    assert all(profile.required_gates == _GEOMETRY_GATES for profile in geometry)
    assert all(profile.required_gates == _FOAM_GATES for profile in foams)
    assert {profile.capability for profile in geometry}.isdisjoint(
        profile.capability for profile in foams
    )


def test_candidate_profile_ids_are_deterministic_and_content_verified() -> None:
    profiles = (
        *multiregion_surface_candidate_profiles(),
        *foam_candidate_profiles(),
    )
    repeated = (
        *multiregion_surface_candidate_profiles(),
        *foam_candidate_profiles(),
    )

    profile_ids = tuple(profile.profile_id for profile in profiles)
    assert profile_ids == tuple(profile.profile_id for profile in repeated)
    assert len(set(profile_ids)) == len(profile_ids)
    assert (
        tuple(
            CapabilityProfile.from_record(profile.to_record()).profile_id
            for profile in profiles
        )
        == profile_ids
    )
