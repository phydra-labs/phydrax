#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased, evidence-free nuclear capability profiles."""

from __future__ import annotations

from ..qualification import CapabilityProfile, SupportTuple


_SPECS = {
    "nuclear.data.multigroup": (
        "nuclear.data",
        {
            "energy-order": "ascending-joule",
            "provenance": "evaluated-and-processed-explicit",
            "uncertainty": "optional-never-implied-exact",
        },
        ("source-admission", "semantic-round-trip", "unit-validity"),
    ),
    "nuclear.fusion.thermal-maxwellian": (
        "nuclear.fusion-reaction",
        {
            "distribution": "maxwellian-tabulated-reactivity",
            "kinematics": "two-body-nonrelativistic",
            "angular-source": "isotropic",
        },
        ("source-admission", "conservation", "numerical-validity", "differentiation"),
    ),
    "nuclear.activation.fixed-network": (
        "nuclear.activation",
        {
            "network": "fixed-topology",
            "flux": "piecewise-constant-multigroup",
            "integration": "matrix-exponential-action",
        },
        ("source-admission", "conservation", "numerical-validity", "locked-reference"),
    ),
}


def nuclear_candidate_profile(name: str, /) -> CapabilityProfile:
    if name not in _SPECS:
        raise ValueError("Unknown nuclear candidate profile.")
    capability, attributes, gates = _SPECS[name]
    return CapabilityProfile(
        name,
        "phydrax",
        "candidate",
        (SupportTuple(capability, attributes),),
        required_gates=gates,
        released=False,
    )


def nuclear_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(nuclear_candidate_profile(name) for name in sorted(_SPECS))


__all__ = ["nuclear_candidate_profile", "nuclear_candidate_profiles"]
