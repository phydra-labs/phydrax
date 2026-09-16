#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased fermionic nonequilibrium profile."""

from ...qualification import CapabilityProfile, SupportTuple


FERMIONIC_SECOND_BORN_SUPPORT = SupportTuple(
    "nonequilibrium-field.fermionic-second-born",
    {
        "statistics": "fermionic",
        "contour": "real-time-keldysh",
        "closure": "self-consistent-second-born",
        "system": "closed-spin-degenerate-local-hubbard-control",
    },
)
FERMIONIC_SECOND_BORN_CANDIDATE = CapabilityProfile(
    "nonequilibrium-field.fermionic-second-born.candidate",
    "phydrax",
    "candidate",
    (FERMIONIC_SECOND_BORN_SUPPORT,),
    required_gates=(
        "car",
        "causality",
        "schwinger-dyson",
        "conservation",
        "resource-envelope",
    ),
    released=False,
)


__all__ = ["FERMIONIC_SECOND_BORN_CANDIDATE", "FERMIONIC_SECOND_BORN_SUPPORT"]
