#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Candidate capability profiles owned by foam mechanics."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


_SPECS = (
    (
        "foams.quasistatic-equilibrium",
        {
            "energy": "effective-pair-tension-area",
            "constraints": "independent-region-volumes",
            "solver": "native-sqp-augmented-lagrangian",
            "derivatives": "fixed-topology-implicit-kkt",
            "topology-events": False,
        },
    ),
    (
        "foams.thickness-driven-rupture",
        {
            "trigger": "accepted-deterministic-sheet-slot-thickness",
            "transaction": "whole-sheet-delete-region-merge",
            "liquid-ledger": "resolved-rim-border-when-supported-otherwise-unresolved",
            "random-rupture": False,
            "derivatives-across-event": False,
        },
    ),
    (
        "foams.constrained-dynamics",
        {
            "air": "incompressible-targets-or-compartment-gas",
            "routes": "overdamped-gradient,film-inertia-shake-rattle",
            "constraints": "independent-region-volume-basis",
            "linear-solve": "native-rank-condition-evidence",
            "derivatives": "fixed-topology-only",
        },
    ),
    (
        "foams.vortex-sheet-air",
        {
            "state": "circulation-per-vertex-region-pair",
            "gauge": "pairwise-mean-zero",
            "velocity": "regularized-biot-savart-vortex-fmm",
            "constraints": "e6-shake-rattle-volumes",
            "topology": "transactional-conservative-circulation-transfer",
            "status": "candidate-until-kornek-refinement",
        },
    ),
    (
        "foams.plateau-border-drainage",
        {
            "network": "fixed-capacity-physical-triple-edges",
            "junctions": "quad-point-mass-pressure-balance",
            "flow": "gravity-viscous-triangular-channel",
            "films": "per-region-pair-prepared-manifold-operators",
            "ledgers": "liquid,surfactant,declared-evaporation",
            "derivatives": "fixed-topology-only",
        },
    ),
)

_GATES = (
    "analytic-control",
    "kkt-evidence",
    "qualification-campaign",
    "public-workflow",
    "documentation-nonclaims",
)


def foam_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return the unreleased profiles owned by foam mechanics."""
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            (SupportTuple(name, attributes),),
            required_gates=_GATES,
        )
        for name, attributes in _SPECS
    )


__all__ = ["foam_candidate_profiles"]
