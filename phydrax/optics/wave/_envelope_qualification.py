#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


def envelope_propagation_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    support = SupportTuple(
        "optics.scalar-envelope-gnlse",
        {
            "representation": "scalar-complex-envelope-exp-minus-i-omega-t",
            "linear": "declared-beta-orders-and-loss",
            "nonlinear": "kerr-optional-delayed-raman-and-shock",
            "integration": "fixed-rk4ip-and-adaptive-step-doubling",
            "evidence": "energy-band-edge-refinement-realized-work",
        },
    )
    return (
        CapabilityProfile(
            support.capability,
            "phydrax",
            "candidate",
            (support,),
            required_gates=(
                "linear-spectral-phase",
                "fundamental-soliton",
                "raman-normalization",
                "shock-convention",
                "fixed-step-refinement",
                "adaptive-realized-work",
            ),
            released=False,
        ),
    )


__all__ = ["envelope_propagation_candidate_profiles"]
