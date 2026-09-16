#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


def spin_foam_candidate_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "spin-foam.su2-bf-identities",
            {
                "algebra": "condon-shortley-cg-6j-f-moves",
                "evidence": "orthogonality-tetrahedral-pentagon",
                "claim": "finite-su2-bf-control",
            },
        ),
        SupportTuple(
            "spin-foam.eprl-semantic-provider",
            {
                "model": "finite-cutoff-lorentzian-eprl-4-simplex",
                "boundary": "ten-spins-five-intertwiners",
                "provider": "pinned-process-protocol-only",
                "native": "zero-spin-b4-reference",
                "claim": "research-only-no-quantum-gravity-validation",
            },
        ),
    )


def spin_foam_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    gates = {
        "spin-foam.su2-bf-identities": (
            "cg-orthogonality",
            "recoupling-unitarity",
            "6j-tetrahedral-symmetry",
            "pentagon-identity",
        ),
        "spin-foam.eprl-semantic-provider": (
            "boundary-admissibility",
            "cutoff-and-resource-admission",
            "complete-convention-ledger",
            "pinned-process-provider",
            "zero-spin-booster-reference",
        ),
    }
    return tuple(
        CapabilityProfile(
            support.capability,
            "phydrax",
            "research-only",
            (support,),
            required_gates=gates[support.capability],
            released=False,
        )
        for support in spin_foam_candidate_support_tuples()
    )


__all__ = [
    "spin_foam_candidate_profiles",
    "spin_foam_candidate_support_tuples",
]
