#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


def fuzzy_space_candidate_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "fuzzy-space.two-particle-sphere",
            {
                "geometry": "lowest-landau-level-fuzzy-sphere",
                "statistics": "explicit-boson-or-fermion-exchange-sector",
                "interaction": "complete-pair-spin-pseudopotentials",
                "symmetry": "exact-su2-condon-shortley-coupling",
                "claim": "finite-cutoff-two-particle-reference",
            },
        ),
    )


def fuzzy_space_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    support = fuzzy_space_candidate_support_tuples()[0]
    return (
        CapabilityProfile(
            support.capability,
            "phydrax",
            "candidate",
            (support,),
            required_gates=(
                "su2-transform-orthonormality",
                "statistics-sector-completeness",
                "rotational-commutator",
                "locked-small-spectrum",
                "cutoff-nonclaim",
            ),
            released=False,
        ),
    )


__all__ = [
    "fuzzy_space_candidate_profiles",
    "fuzzy_space_candidate_support_tuples",
]
