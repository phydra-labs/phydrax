#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact support coordinates for bounded periodic electronic transport."""

from __future__ import annotations

from ...qualification import SupportTuple


KUBO_TRANSPORT_CAPABILITY = "chemistry.periodic.transport.kubo-independent-particle"
BOLTZMANN_TRANSPORT_CAPABILITY = "chemistry.periodic.transport.boltzmann-constant-tau"


def periodic_transport_support_tuples() -> tuple[SupportTuple, ...]:
    """Return candidate coordinates unchanged by any later maturity promotion."""

    return (
        SupportTuple(
            KUBO_TRANSPORT_CAPABILITY,
            {
                "statistics": "fermionic-independent-particle",
                "basis": "periodic-orbital-pencil",
                "velocity": "physical-cartesian-m-per-s",
                "response": "raw-interband-plus-separate-drude",
                "finite_frequency": "strictly-positive-regular-charge-conductivity",
                "linewidth": "named-physical-interband-energy-width",
                "sum_rule": "independent-diamagnetic-f-sum",
                "spin": "conserved-collinear-closure-only",
                "units": "si",
            },
        ),
        SupportTuple(
            BOLTZMANN_TRANSPORT_CAPABILITY,
            {
                "statistics": "fermionic-independent-particle",
                "basis": "periodic-orbital-pencil",
                "velocity": "physical-cartesian-m-per-s",
                "collision_model": "supplied-positive-constant-scalar-relaxation-time",
                "response": "electrical-seebeck-peltier-electronic-thermal",
                "magnetic_field": "zero",
                "rank_policy": "full-cartesian-electrical-moment-required",
                "reciprocity": "kelvin-onsager",
                "units": "si",
            },
        ),
    )


__all__ = [
    "BOLTZMANN_TRANSPORT_CAPABILITY",
    "KUBO_TRANSPORT_CAPABILITY",
    "periodic_transport_support_tuples",
]
