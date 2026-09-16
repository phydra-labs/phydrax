#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased conformal Einstein–AdS capability declarations."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


def ads_conformal_candidate_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "numerical-relativity.conformal-einstein-ads",
            {
                "dimension": "four",
                "formulation": "vacuum-metric-conformal-zero-quantities",
                "boundary": "cartesian-timelike-conformal",
                "gauge": "explicit-generalized-wave-and-conformal-scalar-curvature",
                "evidence": "full-zero-quantity-ledger-exact-ads-control",
            },
        ),
        SupportTuple(
            "numerical-relativity.ads-scalar-reference",
            {
                "background": "fixed-einstein-cylinder-conformal-ads",
                "matter": "linear-reflecting-scalar",
                "integrator": "fixed-step-rk4",
                "evidence": "normal-mode-energy-boundary",
            },
        ),
        SupportTuple(
            "numerical-relativity.holographic-observables",
            {
                "scalar": "declared-two-exponent-source-response-fit",
                "stress": "declared-fefferman-graham-coefficient-counterterms",
                "evidence": "fit-condition-residual-trace-conservation-symmetry",
                "claim": "no-automatic-holographic-renormalization",
            },
        ),
    )


def ads_conformal_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    gates = {
        "numerical-relativity.conformal-einstein-ads": (
            "equation-and-sign-ledger",
            "exact-ads-zero-quantities",
            "generalized-wave-gauge",
            "timelike-boundary-and-corner",
        ),
        "numerical-relativity.ads-scalar-reference": (
            "normal-mode-frequency",
            "reflecting-boundary",
            "energy-convergence",
            "fixed-background-nonclaim",
        ),
        "numerical-relativity.holographic-observables": (
            "declared-asymptotic-exponents",
            "fit-conditioning",
            "counterterm-source-id",
            "trace-and-conservation-audit",
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
        for support in ads_conformal_candidate_support_tuples()
    )


__all__ = [
    "ads_conformal_candidate_profiles",
    "ads_conformal_candidate_support_tuples",
]
