#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Application-leaf profiles for implemented soft-matter compositions."""

from __future__ import annotations

from ..qualification import (
    CampaignRole,
    CapabilityProfile,
    ScientificCampaign,
    ScientificCase,
    SupportTuple,
)


_SPECS = (
    (
        "soft-matter.passive-phase-field",
        {
            "models": "allen-cahn-and-cahn-hilliard",
            "discretization": "finite-element",
            "transaction": "energy-mass-evidence-atomic-rollback",
        },
    ),
    (
        "soft-matter.binary-free-energy-lbm",
        {
            "model": "bounded-model-h",
            "state": "hydrodynamic-and-conservative-phase-populations",
            "transaction": "conservation-energy-population-atomic",
        },
    ),
    (
        "soft-matter.passive-nematic-mac",
        {
            "model": "passive-beris-edwards",
            "coupling": "periodic-mac-work-adjoint",
            "activity": False,
        },
    ),
    (
        "soft-matter.atomistic-observables-protocols",
        {
            "observables": "structure-factor-msd-vacf-diffusion",
            "protocols": "lj-binary-glass-wca-colloid-linear-polymer",
            "kinetic_coarse_fidelity": False,
        },
    ),
    (
        "soft-matter.discrete-path-thermodynamics",
        {
            "input": "canonical-path-buffer",
            "weights": "separately-normalized-forward-reverse",
            "evidence": "first-law-detailed-balance-fluctuation-reversal-ess",
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


def soft_matter_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(
        SupportTuple(capability, attributes) for capability, attributes in _SPECS
    )


def soft_matter_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_GATES,
            released=False,
        )
        for support in soft_matter_support_tuples()
    )


def _campaign(capability: str, /) -> ScientificCampaign:
    slug = capability.replace(".", "-")
    calibration = ScientificCase(
        f"{slug}-calibration",
        f"{slug}-calibration-unit",
        capability,
        "analytic-thermodynamic-control",
        f"{slug}-calibration-preparation",
        f"{slug}-calibration-batch",
        (f"source:{slug}:analytic",),
    )
    locked = ScientificCase(
        f"{slug}-locked",
        f"{slug}-locked-unit",
        capability,
        "independent-transaction-control",
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
            "thermodynamic-identity",
            "transactional-refusal",
            "resource-admission",
        ),
    )


def soft_matter_candidate_campaigns() -> tuple[ScientificCampaign, ...]:
    return tuple(_campaign(capability) for capability, _ in _SPECS)


__all__ = [
    "soft_matter_candidate_campaigns",
    "soft_matter_candidate_profiles",
    "soft_matter_support_tuples",
]
