#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


def spin_network_candidate_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "spin-network.fixed-graph-su2",
            {
                "graph": "finite-oriented-fixed",
                "edges": "doubled-su2-spins",
                "vertices": "left-associated-total-zero-intertwiners",
                "observable": "edge-area-spectrum",
                "claim": "finite-gauss-invariant-basis-no-dynamics",
            },
        ),
    )


def spin_network_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    support = spin_network_candidate_support_tuples()[0]
    return (
        CapabilityProfile(
            support.capability,
            "phydrax",
            "research-only",
            (support,),
            required_gates=(
                "edge-orientation-and-spin-convention",
                "vertex-coupling-order",
                "gauss-invariant-intertwiners",
                "area-reference",
                "fixed-graph-nonclaim",
            ),
            released=False,
        ),
    )


__all__ = [
    "spin_network_candidate_profiles",
    "spin_network_candidate_support_tuples",
]
