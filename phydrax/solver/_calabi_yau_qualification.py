#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased Calabi–Yau metric and observable capability declarations."""

from __future__ import annotations

from ..qualification import CapabilityProfile, SupportTuple


def calabi_yau_candidate_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "calabi-yau.metric-evidence",
            {
                "geometry": "projective-anticanonical-hypersurface",
                "metric": "kahler-potential-candidate",
                "sampling": "independent-fixed-ancestry-heldout",
                "evidence": "monge-ampere-positivity-volume-ess-batches-optional-ricci",
                "claim": "approximate-metric-no-exact-ricci-flat-claim",
            },
        ),
        SupportTuple(
            "calabi-yau.moduli-observables",
            {
                "family": "fixed-support-pivot-and-pgl-slice",
                "metric": "sampled-representative-gram",
                "yukawa": "sampled-symmetric-density",
                "topology": "sampled-chern-weil-control",
                "authority": "harmonic-representatives-only",
            },
        ),
    )


def calabi_yau_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    gates = {
        "calabi-yau.metric-evidence": (
            "disjoint-training-heldout-ancestry",
            "measure-and-effective-samples",
            "positivity-and-validity",
            "batch-uncertainty",
            "artifact-provenance",
        ),
        "calabi-yau.moduli-observables": (
            "transverse-family",
            "representative-provenance",
            "weil-petersson-hermiticity-positivity",
            "yukawa-permutation-symmetry",
            "chern-weil-convention",
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
        for support in calabi_yau_candidate_support_tuples()
    )


__all__ = [
    "calabi_yau_candidate_profiles",
    "calabi_yau_candidate_support_tuples",
]
