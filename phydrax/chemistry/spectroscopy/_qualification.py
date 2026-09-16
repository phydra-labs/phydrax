#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Maturity-neutral profiles and campaigns for implemented material spectroscopy."""

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
        "chemistry.spectroscopy.optical-dielectric",
        {
            "source": "positive-frequency-kubo",
            "response": "regular-interband-plus-separate-drude",
            "instrument": "separate-normalized-transform",
        },
    ),
    (
        "chemistry.spectroscopy.periodic-vibrational",
        {
            "modalities": "infrared-and-nonresonant-raman",
            "source": "gamma-phonons-and-provider-tensors",
            "evidence": "charge-neutrality-raman-symmetry-stokes-balance",
        },
    ),
    (
        "chemistry.spectroscopy.arpes",
        {
            "approximation": "sudden-fixed-support",
            "matrix_elements": "required-provider-artifact",
            "evidence": "forbidden-zero-and-spectral-moments",
        },
    ),
    (
        "chemistry.spectroscopy.tersoff-hamann",
        {
            "approximation": "weak-tunneling-s-wave-constant-tip-dos",
            "input": "provider-vacuum-ldos",
            "output": "signed-current-and-nonnegative-conductance",
        },
    ),
    (
        "chemistry.spectroscopy.eels",
        {
            "scope": "macroscopic-longitudinal-valence-single-scattering",
            "input": "nonzero-q-dielectric",
            "evidence": "positive-transfer-and-thermal-factor",
        },
    ),
    (
        "chemistry.spectroscopy.elastic-scattering",
        {
            "probes": "nonresonant-xray-and-coherent-neutron",
            "inputs": "provider-form-factors-or-provenanced-lengths",
            "evidence": "friedel-passivity",
        },
    ),
    (
        "chemistry.spectroscopy.dynamic-structure",
        {
            "system": "bounded-exact-finite-state",
            "output": "unbroadened-transition-bank",
            "evidence": "adjoint-equal-time-detailed-balance",
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


def material_spectroscopy_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(
        SupportTuple(capability, attributes) for capability, attributes in _SPECS
    )


def material_spectroscopy_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_GATES,
            released=False,
        )
        for support in material_spectroscopy_support_tuples()
    )


def _campaign(capability: str, /) -> ScientificCampaign:
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
        criteria_ids=(
            "normalization-and-sum-rule",
            "units-and-convention",
            "instrument-separation",
        ),
    )


def material_spectroscopy_candidate_campaigns() -> tuple[ScientificCampaign, ...]:
    return tuple(_campaign(capability) for capability, _ in _SPECS)


__all__ = [
    "material_spectroscopy_candidate_campaigns",
    "material_spectroscopy_candidate_profiles",
    "material_spectroscopy_support_tuples",
]
