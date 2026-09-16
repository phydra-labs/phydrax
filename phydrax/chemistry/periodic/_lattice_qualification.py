#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Maturity-neutral profiles and campaigns for implemented lattice materials."""

from __future__ import annotations

from ...qualification import (
    CampaignRole,
    CapabilityProfile,
    ScientificCampaign,
    ScientificCase,
    SupportTuple,
)


_SPECS = (
    (
        "chemistry.periodic.lattice.ifc2",
        {
            "representation": "canonical-real-space-edge-relation",
            "source": "native-finite-displacement-or-normalized-provider",
            "constraints": "raw-and-corrected-pair-acoustic-rotational",
            "units": "explicit-energy-per-length-squared",
        },
        ("ifc2-symmetry", "sum-rules", "displacement-refinement"),
    ),
    (
        "chemistry.periodic.lattice.harmonic",
        {
            "input": "canonical-ifc2",
            "dynamical_matrix": "mass-weighted-periodic-family",
            "eigensystem": "bounded-hermitian",
            "evidence": "hermiticity-acoustic-modes-q-reversal",
        },
        ("dynamical-hermiticity", "acoustic-modes", "q-refinement"),
    ),
    (
        "chemistry.periodic.lattice.qha",
        {
            "input": "phase-matched-harmonic-volume-series",
            "thermodynamics": "discrete-volume-quasiharmonic",
            "volume": "inside-supplied-support",
            "instability": "explicit-refusal",
        },
        ("phase-identity", "thermodynamic-identities", "volume-support"),
    ),
    (
        "chemistry.periodic.lattice.three-phonon-rta",
        {
            "input": "canonical-ifc3-or-provider-mode-vertices",
            "approximation": "single-mode-relaxation-time",
            "selection": "explicit-momentum-and-energy",
            "evidence": "permutation-linewidth-mesh-broadening",
        },
        ("ifc3-permutation", "linewidth-positivity", "mesh-broadening-refinement"),
    ),
)
_REQUIRED_GATES = (
    "scientific-validation",
    "resource-envelope",
    "lifecycle-restore",
    "runtime-distribution",
    "documentation-nonclaims",
)


def lattice_material_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(
        SupportTuple(capability, attributes) for capability, attributes, _ in _SPECS
    )


def lattice_material_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_REQUIRED_GATES,
            released=False,
        )
        for support in lattice_material_support_tuples()
    )


def _campaign(capability: str, criteria: tuple[str, ...], /) -> ScientificCampaign:
    slug = capability.replace(".", "-")
    calibration = ScientificCase(
        f"{slug}-calibration",
        f"{slug}-calibration-unit",
        capability,
        "analytic-control",
        f"{slug}-calibration-preparation",
        f"{slug}-calibration-batch",
        (f"source:{slug}:analytic",),
    )
    locked = ScientificCase(
        f"{slug}-locked",
        f"{slug}-locked-unit",
        capability,
        "independent-locked-control",
        f"{slug}-locked-preparation",
        f"{slug}-locked-batch",
        (f"source:{slug}:independent",),
    )
    return ScientificCampaign(
        (calibration, locked),
        (
            CampaignRole("calibration", (calibration.case_id,)),
            CampaignRole("locked_evaluation", (locked.case_id,)),
        ),
        criteria_ids=criteria,
    )


def lattice_material_candidate_campaigns() -> tuple[ScientificCampaign, ...]:
    return tuple(_campaign(capability, criteria) for capability, _, criteria in _SPECS)


__all__ = [
    "lattice_material_candidate_campaigns",
    "lattice_material_candidate_profiles",
    "lattice_material_support_tuples",
]
