#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Maturity-neutral profiles and campaigns for implemented magnetic resonance."""

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
        "magnetic-resonance.nmr.exact-single-crystal",
        {
            "sites": "finite-nuclear-spin",
            "interactions": "shielding-j-dipolar-quadrupolar",
            "execution": "exact-small-hilbert-space",
        },
    ),
    (
        "magnetic-resonance.epr.exact-single-crystal",
        {
            "sites": "exactly-one-electron-plus-finite-nuclei",
            "interactions": "g-tensor-hyperfine-dipolar",
            "execution": "exact-small-hilbert-space",
        },
    ),
    (
        "magnetic-resonance.musr.exact-static-site",
        {
            "sites": "exactly-one-positive-muon-static",
            "interactions": "zeeman-and-declared-spin-couplings",
            "execution": "exact-small-hilbert-space",
        },
    ),
)
_GATES = (
    "scientific-validation",
    "resource-envelope",
    "lifecycle-restore",
    "runtime-distribution",
    "documentation-nonclaims",
)


def magnetic_resonance_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(
        SupportTuple(capability, attributes) for capability, attributes in _SPECS
    )


def magnetic_resonance_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_GATES,
            released=False,
        )
        for support in magnetic_resonance_support_tuples()
    )


def _campaign(capability: str, /) -> ScientificCampaign:
    slug = capability.replace(".", "-")
    calibration = ScientificCase(
        f"{slug}-calibration",
        f"{slug}-calibration-unit",
        capability,
        "analytic-pulse-control",
        f"{slug}-calibration-preparation",
        f"{slug}-calibration-batch",
        (f"source:{slug}:analytic",),
    )
    locked = ScientificCase(
        f"{slug}-locked",
        f"{slug}-locked-unit",
        capability,
        "independent-echo-control",
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
        criteria_ids=(
            "unit-and-sign-conventions",
            "propagator-unitarity",
            "analytic-pulse-echo",
        ),
    )


def magnetic_resonance_candidate_campaigns() -> tuple[ScientificCampaign, ...]:
    return tuple(_campaign(capability) for capability, _ in _SPECS)


__all__ = [
    "magnetic_resonance_candidate_campaigns",
    "magnetic_resonance_candidate_profiles",
    "magnetic_resonance_support_tuples",
]
