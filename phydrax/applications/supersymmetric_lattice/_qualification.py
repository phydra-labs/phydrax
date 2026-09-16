#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased finite supersymmetric-lattice capability declarations."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


def supersymmetric_lattice_candidate_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "supersymmetric-lattice.bosonic-reference",
            {
                "theory": "finite-complexified-twisted-sym-and-bfss",
                "links": "independent-forward-reverse",
                "evidence": "gauge-ward-small-pfaffian",
                "claim": "finite-regulated-reference",
            },
        ),
        SupportTuple(
            "supersymmetric-lattice.twisted-n2-rhmc",
            {
                "theory": "two-dimensional-twisted-n2-u-n",
                "fermions": "eta-psi0-psi1-chi01-kahler-dirac",
                "measure": "regulated-phase-quenched-pfaffian-magnitude",
                "coordinates": "bounded-real-independent-complex-links",
                "evidence": "antisymmetry-force-rational-hmc-ward-phase-ess",
            },
        ),
    )


def supersymmetric_lattice_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    gates = {
        "supersymmetric-lattice.bosonic-reference": (
            "field-placement",
            "complexified-gauge-invariance",
            "locked-action-reference",
        ),
        "supersymmetric-lattice.twisted-n2-rhmc": (
            "real-coordinate-roundtrip",
            "kahler-dirac-antisymmetry-and-adjoint",
            "structural-normal-interval",
            "rational-approximation-evidence",
            "force-directional-derivative",
            "reversible-accept-reject",
            "pfaffian-phase-and-overlap",
        ),
    }
    return tuple(
        CapabilityProfile(
            support.capability,
            "phydrax",
            "candidate",
            (support,),
            required_gates=gates[support.capability],
            released=False,
        )
        for support in supersymmetric_lattice_candidate_support_tuples()
    )


__all__ = [
    "supersymmetric_lattice_candidate_profiles",
    "supersymmetric_lattice_candidate_support_tuples",
]
