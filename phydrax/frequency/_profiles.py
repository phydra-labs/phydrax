#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from ..qualification import CapabilityProfile, SupportTuple


def frequency_candidate_profiles():
    specs = (
        ("frequency.axis-convention", {"axis": "strict-hz", "phasor": "explicit"}),
        ("frequency.scattering-network", {"matrix": "frequency-port-port"}),
        ("frequency.fixed-pole-fit", {"method": "least-squares"}),
        ("frequency.harmonic-balance", {"method": "fourier-collocation"}),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attrs),),
            required_gates=("convention", "analytic-control", "public-workflow"),
        )
        for name, attrs in specs
    )


__all__ = ["frequency_candidate_profiles"]
