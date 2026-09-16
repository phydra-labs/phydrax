#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ..qualification import CapabilityProfile, SupportTuple


def particle_spectrum_candidate_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "particle-spectrum.interchange-provider",
            {
                "interchange": "strict-bounded-slha-with-unknown-block-preservation",
                "provider": "pinned-host-executable",
                "status": "separate-numerical-physical-approximation-provider",
                "claim": "finite-calculator-specific-spectrum",
            },
        ),
        SupportTuple(
            "particle-spectrum.native-scale-bvp",
            {
                "flow": "fixed-step-log-scale-rk4",
                "constraints": "explicit-low-and-high-square-residual",
                "corrector": "native-jacobian-newton-backtracking",
                "evidence": "full-trajectory-residual-history-status",
            },
        ),
    )


def particle_spectrum_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    gates = {
        "particle-spectrum.interchange-provider": (
            "slha-semantic-roundtrip",
            "unknown-block-retention",
            "pinned-provider",
            "typed-failure-status",
            "approximation-profile",
        ),
        "particle-spectrum.native-scale-bvp": (
            "beta-source-identity",
            "scale-ordering",
            "boundary-residual-shape",
            "analytic-flow-control",
            "full-trajectory-evidence",
        ),
    }
    return tuple(
        CapabilityProfile(
            support.capability,
            "phydrax",
            "candidate",
            (support,),
            required_gates=gates[support.capability],
            released=False,
        )
        for support in particle_spectrum_candidate_support_tuples()
    )


__all__ = [
    "particle_spectrum_candidate_profiles",
    "particle_spectrum_candidate_support_tuples",
]
