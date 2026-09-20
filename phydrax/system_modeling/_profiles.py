#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from ..qualification import CapabilityProfile, SupportTuple


def system_modeling_candidate_profiles():
    specs = (
        (
            "system-modeling.acausal-connectors",
            {"variables": "across-through", "connections": "equality-conservation"},
        ),
        ("system-modeling.structural-incidence", {"analysis": "rank-matching"}),
        (
            "system-modeling.fmi3-contract",
            {"modes": "model-exchange-co-simulation-scheduled"},
        ),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attrs),),
            required_gates=(
                "connection-conservation",
                "structural-analysis",
                "public-workflow",
            ),
        )
        for name, attrs in specs
    )


__all__ = ["system_modeling_candidate_profiles"]
