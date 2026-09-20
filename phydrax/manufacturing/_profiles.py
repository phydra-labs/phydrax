#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from ..qualification import CapabilityProfile, SupportTuple


def manufacturing_candidate_profiles():
    specs = (
        (
            "manufacturing.process-schedule",
            {"events": "move-deposit-remove-dwell-heat-fixture-transfer"},
        ),
        (
            "manufacturing.material-activation",
            {"topology": "fixed-capacity-activation-removal"},
        ),
        ("manufacturing.moving-gaussian-source", {"source": "surface-gaussian"}),
        ("manufacturing.moving-goldak-source", {"source": "double-ellipsoid"}),
        ("manufacturing.process-history", {"ledgers": "mass-energy-active-measure"}),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attrs),),
            required_gates=("identity", "conservation", "restart", "public-workflow"),
        )
        for name, attrs in specs
    )


__all__ = ["manufacturing_candidate_profiles"]
