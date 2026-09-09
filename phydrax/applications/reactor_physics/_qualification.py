#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased, evidence-free reactor-physics capability profiles."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


def reactor_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    diffusion = CapabilityProfile(
        "reactor.diffusion.one-dimensional-multigroup",
        "phydrax",
        "candidate",
        (
            SupportTuple(
                "reactor.neutron-diffusion",
                {
                    "geometry": "one-dimensional-metric-line",
                    "energy": "multigroup",
                    "boundary": "zero-dirichlet-vacuum",
                    "criticality": "power-iteration",
                },
            ),
        ),
        required_gates=(
            "source-admission",
            "neutron-balance",
            "numerical-validity",
            "locked-reference",
        ),
        released=False,
    )
    kinetics = CapabilityProfile(
        "reactor.kinetics.point-delayed-neutron",
        "phydrax",
        "candidate",
        (
            SupportTuple(
                "reactor.point-kinetics",
                {
                    "precursors": "fixed-family-axis",
                    "integration": "implicit-euler",
                    "feedback": "prescribed-reactivity",
                },
            ),
        ),
        required_gates=(
            "neutron-balance",
            "numerical-validity",
            "differentiation",
            "analytic-reference",
        ),
        released=False,
    )
    return diffusion, kinetics


__all__ = ["reactor_candidate_profiles"]
