#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cross-route consistency release matrix of the charged-particle radiation stack.

Each row pairs independently implemented routes that compute one physical
observable and names the tests that assert their agreement at a measured,
convergence-justified tolerance. The matrix is one candidate profile: its
required gates are exactly those test node IDs and its dependencies are the
member route profiles, so release discovery of the matrix requires every member
profile to be released and every row to carry current evidence.
"""

from __future__ import annotations

from typing import NamedTuple

from ._registry import CapabilityProfile, SupportTuple


_CROSS_ROUTE = "tests/integration/test_radiation_cross_route.py"


class _MatrixRow(NamedTuple):
    pair: str
    observable: str
    members: tuple[str, ...]
    tests: tuple[str, ...]


_MATRIX = (
    _MatrixRow(
        "a1-b1",
        "vacuum-circular-orbit-far-field-complex-spectrum",
        (
            "electromagnetics.vacuum-trajectory-radiation",
            "electromagnetics.maxwell-far-field",
            "electromagnetics.moving-charge-transition",
        ),
        (
            "tests/unit/solver/test_prescribed_charge_vacuum_orbit.py"
            "::test_finest_grid_far_field_matches_trajectory_radiation",
            "tests/unit/solver/test_prescribed_charge_vacuum_orbit.py"
            "::test_far_field_error_converges_at_second_order",
        ),
    ),
    _MatrixRow(
        "a1-p",
        "pic-tracked-cyclotron-spectrum-equals-exact-helix-trajectory-radiation",
        (
            "electromagnetics.vacuum-trajectory-radiation",
            "pic.dispersive-self-consistent",
        ),
        (
            f"{_CROSS_ROUTE}"
            "::test_pic_tracked_cyclotron_spectrum_converges_to_a1_on_exact_helix",
        ),
    ),
    _MatrixRow(
        "b2-b4",
        "cherenkov-and-smith-purcell-time-domain-equals-frequency-domain",
        (
            "electromagnetics.frequency-moving-charge",
            "electromagnetics.moving-charge-cherenkov",
            "electromagnetics.moving-charge-smith-purcell",
        ),
        (
            "tests/unit/solver/test_prescribed_charge_frequency_domain.py"
            "::test_cherenkov_time_domain_equals_frequency_domain",
            "tests/unit/solver/test_prescribed_charge_frequency_domain.py"
            "::test_smith_purcell_time_domain_equals_frequency_domain",
        ),
    ),
    _MatrixRow(
        "p1-p2-p3",
        "axisymmetric-tm-pulse-cochain-cartesian-psatd-quasi-cylindrical",
        (
            "pic.dispersive-self-consistent",
            "pic.psatd",
            "pic.quasi-cylindrical",
        ),
        (
            f"{_CROSS_ROUTE}"
            "::test_axisymmetric_tm_pulse_agrees_across_cochain_and_psatd_solvers",
        ),
    ),
    _MatrixRow(
        "q1-q2",
        "small-chi-compton-energy-loss-equals-classical-landau-lifshitz",
        ("pic.radiation-reaction", "pic.qed-cascade"),
        (
            f"{_CROSS_ROUTE}"
            "::test_compton_energy_loss_approaches_classical_landau_lifshitz_as_chi",
        ),
    ),
    _MatrixRow(
        "c2-a1",
        "thermal-cyclotron-harmonic-emissivity-equals-averaged-helix-spectra",
        (
            "electromagnetics.magnetobremsstrahlung",
            "electromagnetics.vacuum-trajectory-radiation",
        ),
        (
            f"{_CROSS_ROUTE}"
            "::test_thermal_cyclotron_harmonics_match_trajectory_radiation_of_helices",
        ),
    ),
    _MatrixRow(
        "m2-b2",
        "cherenkov-photon-yield-equals-poynting-flux-over-photon-energy",
        (
            "optics.transport.charged-step-optical-sources",
            "electromagnetics.frequency-moving-charge",
        ),
        (
            f"{_CROSS_ROUTE}::test_source_cherenkov_yield_equals_field_poynting_photon_flux",
        ),
    ),
    _MatrixRow(
        "x4",
        "csr-3d-steady-igf-equals-retarded-mesh",
        ("accelerator.csr-3d",),
        (
            "tests/unit/applications/test_accelerator_csr.py"
            "::test_retarded_mesh_and_steady_igf_agree_on_longitudinal_and_horizontal_forces",
            "tests/unit/applications/test_accelerator_csr.py"
            "::test_retarded_mesh_kick_converges_in_particle_count_and_kernel_width",
        ),
    ),
    _MatrixRow(
        "x5-x6",
        "seeded-small-signal-gain-full-wave-equals-averaged-fel",
        ("accelerator.fel-averaged", "accelerator.fel-full-wave"),
        (
            "tests/unit/applications/test_accelerator_fel_full_wave.py"
            "::test_seeded_small_signal_gain_matches_the_averaged_fel",
        ),
    ),
)


def _member_profiles() -> dict[str, CapabilityProfile]:
    # Member owners import this package, so they are resolved on first use.
    from ..applications.accelerator._qualification import (
        accelerator_candidate_profiles,
    )
    from ..electromagnetics._qualification import (
        electromagnetic_radiation_candidate_profiles,
    )
    from ..optics.transport._qualification import optical_transport_candidate_profiles
    from ..solver._maxwell_qualification import (
        maxwell_far_field_candidate_profiles,
        maxwell_frequency_moving_charge_candidate_profiles,
        maxwell_moving_charge_candidate_profiles,
    )
    from ..solver._pic_qualification import (
        pic_dispersive_self_consistent_candidate_profiles,
        pic_qed_cascade_candidate_profiles,
        pic_radiation_reaction_candidate_profiles,
        pic_spectral_candidate_profiles,
    )

    providers = (
        accelerator_candidate_profiles,
        electromagnetic_radiation_candidate_profiles,
        optical_transport_candidate_profiles,
        maxwell_far_field_candidate_profiles,
        maxwell_frequency_moving_charge_candidate_profiles,
        maxwell_moving_charge_candidate_profiles,
        pic_dispersive_self_consistent_candidate_profiles,
        pic_qed_cascade_candidate_profiles,
        pic_radiation_reaction_candidate_profiles,
        pic_spectral_candidate_profiles,
    )
    return {
        profile.capability: profile for provider in providers for profile in provider()
    }


def radiation_release_matrix_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return the unreleased cross-route release-matrix profile."""
    profiles = _member_profiles()
    members = sorted({member for row in _MATRIX for member in row.members})
    missing = [member for member in members if member not in profiles]
    if missing:
        raise ValueError(
            "Release-matrix members have no registered candidate profile: "
            + ", ".join(missing)
        )
    return (
        CapabilityProfile(
            "radiation.cross-route-release-matrix.profile",
            "phydrax",
            "candidate",
            (
                SupportTuple(
                    "radiation.cross-route-release-matrix",
                    {row.pair: row.observable for row in _MATRIX},
                ),
            ),
            dependencies=tuple(profiles[member].profile_id for member in members),
            required_gates=tuple(test for row in _MATRIX for test in row.tests),
            released=False,
        ),
    )


__all__ = ["radiation_release_matrix_candidate_profiles"]
