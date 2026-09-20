#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from ..qualification import CapabilityProfile, SupportTuple


def materials_candidate_profiles():
    specs = (
        ("materials.identity-history", {"state": "temperature-pressure-phase-history"}),
        ("materials.homogenization", {"methods": "voigt-reuss-hill"}),
        ("materials.phase-transformation", {"methods": "jmak-koistinen-marburger"}),
        ("materials.linear-calibration", {"method": "weighted-least-squares"}),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attrs),),
            required_gates=("analytic-control", "conservation", "public-workflow"),
        )
        for name, attrs in specs
    )


__all__ = ["materials_candidate_profiles"]
