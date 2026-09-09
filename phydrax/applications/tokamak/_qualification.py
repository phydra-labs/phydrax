#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased, evidence-free axisymmetric tokamak capability profiles."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


_SPECS = {
    "tokamak.equilibrium.imported-nested": (
        "tokamak.equilibrium-import",
        {
            "geometry": "axisymmetric-r-phi-z",
            "topology": "nested-star-shaped-inside-lcfs",
            "convention": "explicit-full-sign-and-flux-normalization",
        },
        ("source-admission", "coordinate-round-trip", "geometry-validity"),
    ),
    "tokamak.transport.prescribed-conductance": (
        "tokamak.core-transport",
        {
            "equations": "electron-particle-electron-energy-ion-energy",
            "geometry": "fixed-flux-surface",
            "closure": "prescribed-integrated-conductance",
            "integration": "implicit-backward-euler",
        },
        (
            "conservation",
            "numerical-validity",
            "differentiation",
            "manufactured-solution",
        ),
    ),
    "tokamak.equilibrium.fixed-boundary-current": (
        "tokamak.grad-shafranov",
        {
            "boundary": "fixed-dirichlet",
            "source": "prescribed-toroidal-current-density",
            "grid": "uniform-r-z",
        },
        (
            "conservation",
            "numerical-validity",
            "differentiation",
            "manufactured-solution",
        ),
    ),
    "tokamak.equilibrium.free-boundary-circuit": (
        "tokamak.free-boundary",
        {
            "plasma-current": "prescribed",
            "circuits": "active-and-passive-linear",
            "coupling": "quasi-static-coil-boundary-response",
        },
        ("reciprocity", "energy-balance", "numerical-validity", "locked-reference"),
    ),
    "tokamak.engineering.fusion-activation": (
        "tokamak.fusion-engineering",
        {
            "coupling": "staggered-transport-fusion-response-activation",
            "neutron-response": "explicit-native-or-constant-artifact",
            "external-gradient": "not-claimed",
        },
        ("source-admission", "conservation", "numerical-validity", "locked-reference"),
    ),
}


def tokamak_candidate_profile(name: str, /) -> CapabilityProfile:
    if name not in _SPECS:
        raise ValueError("Unknown tokamak candidate profile.")
    capability, attributes, gates = _SPECS[name]
    return CapabilityProfile(
        name,
        "phydrax",
        "candidate",
        (SupportTuple(capability, attributes),),
        required_gates=gates,
        released=False,
    )


def tokamak_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(tokamak_candidate_profile(name) for name in sorted(_SPECS))


__all__ = ["tokamak_candidate_profile", "tokamak_candidate_profiles"]
