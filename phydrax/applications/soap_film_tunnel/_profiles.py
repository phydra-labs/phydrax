#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Candidate capability profile of the planar soap-film tunnel."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


def soap_film_tunnel_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Unreleased candidate profile of quasi-2D gravity-driven soap-film flow."""
    support = SupportTuple(
        "soap-film-tunnel.planar-plug-flow",
        {
            "film": "symmetric-plug-flow-marangoni-elasticity",
            "geometry": "planar-channel-with-optional-circular-hole",
            "mesh": "meshcore-constrained-delaunay-with-conductance-audit",
            "boundaries": "inflow-outflow-wire-and-rim",
            "drive": "gravity-balanced-by-linear-air-drag",
            "evidence": "flux-ledgers-obstacle-force-film-mach-strouhal",
            "time-integration": "first-order-imex",
            "nonclaims": "no-3d-foam-no-nonlinear-air-boundary-layer-drag",
        },
    )
    return (
        CapabilityProfile(
            support.capability,
            "phydrax",
            "candidate",
            (support,),
            required_gates=(
                "terminal-velocity-uniform-flow",
                "flux-balance-ledgers",
                "no-slip-enforcement",
                "film-mach-evidence",
                "cylinder-wake-strouhal-campaign",
            ),
            released=False,
        ),
    )


__all__ = ["soap_film_tunnel_candidate_profiles"]
