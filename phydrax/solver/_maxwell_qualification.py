#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased Maxwell far-field, antenna, and dispersion-audit capability declarations."""

from __future__ import annotations

from ..qualification import CapabilityProfile, SupportTuple


_REQUIRED_GATES = (
    "hertzian-dipole-pattern-and-energy",
    "grid-convergence",
    "nested-surface-agreement",
    "surface-poynting-energy-balance",
    "admissibility-refusals",
    "documentation-nonclaims",
)

_ANTENNA_REQUIRED_GATES = (
    "one-way-forward-backward-ratio",
    "paraxial-gaussian-waist-and-gouy-phase",
    "antenna-work-equals-injected-energy",
    "moving-antenna-doppler",
    "laser-envelope-adapter-round-trip",
    "admissibility-refusals",
    "documentation-nonclaims",
)

_DISPERSION_REQUIRED_GATES = (
    "vacuum-yee-analytic-dispersion",
    "audit-equals-runtime-phase-advance",
    "ade-continuum-limit",
    "courant-boundary",
    "appleton-hartree-continuum-and-cutoffs",
    "negative-index-band",
    "cherenkov-regime-masks",
    "passivity-and-power-balance",
    "admissibility-refusals",
    "documentation-nonclaims",
)


def maxwell_far_field_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "electromagnetics.maxwell-far-field",
            {
                "acquisition": "time-integral-trapezoid-positive-exponent",
                "surface": "structured-huygens-box-or-tetrahedral-face-set",
                "exterior": "declared-lossless-homogeneous",
                "admissibility": "no-cpml-overlap-no-surface-current",
                "outputs": "field-spectrum-coherency-stokes-spectral-energy",
            },
        ),
    )


def maxwell_far_field_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_REQUIRED_GATES,
            released=False,
        )
        for support in maxwell_far_field_support_tuples()
    )


def maxwell_antenna_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "electromagnetics.one-way-antenna",
            {
                "source": "sampled-plane-electric-and-magnetic-sheet",
                "placement": "total-field-scattered-field-staggered-sheets",
                "medium": "declared-lossless-homogeneous-on-sheet-support",
                "motion": "normal-boost-in-scale-vacuum",
                "adapters": "pulse-envelope-focused-gaussian-openpmd-laser-envelope",
                "outputs": "cochain-currents-work-ledger-support-evidence",
            },
        ),
    )


def maxwell_antenna_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_ANTENNA_REQUIRED_GATES,
            released=False,
        )
        for support in maxwell_antenna_support_tuples()
    )


def maxwell_dispersion_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "electromagnetics.discrete-dispersion-audit",
            {
                "update": "executed-compatible-leapfrog-one-step-bloch-map",
                "grid": "uniform-structured-axes-declared-homogeneous-region",
                "media": "diagonal-lorentz-drude-magnetic-poles-magnetized-cold-plasma",
                "admissibility": "local-update-no-magnetic-projection-verified-stencil",
                "outputs": "multipliers-numerical-frequencies-cherenkov-regime-masks",
            },
        ),
    )


def maxwell_dispersion_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_DISPERSION_REQUIRED_GATES,
            released=False,
        )
        for support in maxwell_dispersion_support_tuples()
    )


__all__ = [
    "maxwell_antenna_candidate_profiles",
    "maxwell_antenna_support_tuples",
    "maxwell_dispersion_candidate_profiles",
    "maxwell_dispersion_support_tuples",
    "maxwell_far_field_candidate_profiles",
    "maxwell_far_field_support_tuples",
]
