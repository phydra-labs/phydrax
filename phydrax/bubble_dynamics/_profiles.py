#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Candidate capability profiles for radial bubble dynamics."""

from __future__ import annotations

from ..qualification import CapabilityProfile, SupportTuple


def bubble_dynamics_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return the unreleased radial-bubble candidates; each graduates only via its campaign."""
    specifications = (
        (
            "bubble-dynamics.clean-radial",
            {
                "equations": "rayleigh-plesset,radiation,gas-radiation,keller-miksis,gilmore",
                "gas": "polytropic,hard-core",
                "interface": "clean",
                "integration": "native-diffrax-events",
            },
            (
                "rayleigh-collapse",
                "energy-identity",
                "compressible-limit",
                "minnaert-frequency",
                "derivative-duality",
                "public-workflow",
            ),
        ),
        (
            "bubble-dynamics.coated-microbubble",
            {
                "shells": "marmottant,gompertz,hoff,church-equal-density,sarkar,doinikov,maxwell",
                "regimes": "native-event-localized",
                "inference": "dynamic-shell-parameters",
            },
            (
                "regime-radii",
                "event-time-convergence",
                "coated-resonance",
                "compression-only-reference",
                "shell-gradient",
                "public-workflow",
            ),
        ),
        (
            "bubble-dynamics.thermal-compressible",
            {
                "gas": "boundary-layer,reduced-transfer,spectral-chebyshev,material",
                "liquid": "newtonian,power-law,kelvin-voigt,zener,oldroyd-b",
                "equation": "gilmore-tait",
            },
            (
                "prosperetti-linear-theory",
                "gilmore-keller-miksis-limit",
                "viscoelastic-limits",
                "spectral-refinement",
                "public-workflow",
            ),
        ),
        (
            "bubble-dynamics.surface-nanobubble",
            {
                "dissolution": "epstein-plesset-quasi-static,epstein-plesset-full-history",
                "surface": "pinned-spherical-cap,unpinned-spherical-cap",
                "bulk_nanobubbles": False,
            },
            (
                "closed-form-lifetime",
                "independent-history-reference",
                "lohse-zhang-equilibrium",
                "stability-sign",
                "continuum-evidence",
                "public-workflow",
            ),
        ),
    )
    return tuple(
        CapabilityProfile(
            f"{capability}.profile",
            "phydrax",
            (SupportTuple(capability, attributes),),
            required_gates=gates,
        )
        for capability, attributes, gates in specifications
    )


__all__ = ["bubble_dynamics_candidate_profiles"]
