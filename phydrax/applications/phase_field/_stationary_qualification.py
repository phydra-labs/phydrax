#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


def stationary_soliton_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    support = SupportTuple(
        "phase-field.stationary-double-well-kink",
        {
            "model": "one-dimensional-symmetric-double-well",
            "sector": "fixed-minus-one-to-plus-one-dirichlet",
            "solver": "bounded-newton-backtracking",
            "evidence": "euler-energy-boundary-sector-stability-refinement",
        },
    )
    return (
        CapabilityProfile(
            support.capability,
            "phydrax",
            "candidate",
            (support,),
            required_gates=(
                "analytic-tanh-reference",
                "euler-lagrange-residual",
                "topological-sector",
                "energy-and-stability",
                "mesh-refinement",
            ),
            released=False,
        ),
    )


__all__ = ["stationary_soliton_candidate_profiles"]
