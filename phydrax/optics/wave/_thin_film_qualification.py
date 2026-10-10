#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


def thin_film_interference_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    support = SupportTuple(
        "optics.thin-film-interference",
        {
            "stack": "lossless-ambient-lossless-film-passive-substrate",
            "model": "exact-airy-summation-exp-minus-i-omega-t",
            "polarization": "s-and-p-amplitudes-and-flux",
            "evidence": "energy-residual-and-spectral-fringe-sampling",
            "nonclaims": "no-absorbing-film-or-multilayer-stack",
        },
    )
    return (
        CapabilityProfile(
            support.capability,
            "phydrax",
            (support,),
            required_gates=(
                "single-interface-fresnel-limit",
                "lossless-energy-balance",
                "symmetric-film-extrema",
                "spectral-fringe-sampling",
                "characteristic-matrix-cross-check",
            ),
            released=False,
        ),
    )


__all__ = ["thin_film_interference_candidate_profiles"]
