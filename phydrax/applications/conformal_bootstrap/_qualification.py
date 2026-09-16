#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased finite conformal-bootstrap capability declarations."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


def conformal_bootstrap_candidate_support_tuples() -> tuple[SupportTuple, ...]:
    return (
        SupportTuple(
            "conformal-bootstrap.global-block",
            {
                "external": "four-scalars",
                "dimension": "real-d-greater-than-one",
                "representation": "radial-meromorphic-or-d2-d4-factorized",
                "evidence": "truncation-casimir-finiteness",
            },
        ),
        SupportTuple(
            "conformal-bootstrap.crossing-pmp",
            {
                "crossing": "explicit-basis-gauge-matrices",
                "positivity": "exact-decimal-polynomial-matrix-program",
                "native-audit": "finite-sampled-psd-only",
                "claim": "finite-frontend-no-continuum-exclusion",
            },
        ),
        SupportTuple(
            "conformal-bootstrap.sdpb-provider",
            {
                "execution": "pinned-host-process",
                "wire": "exact-decimal-json-pmp",
                "result": "exact-summary-and-functional-reconstruction",
                "authority": "external-result-plus-independent-sampled-audit",
            },
        ),
        SupportTuple(
            "conformal-bootstrap.virasoro-reference",
            {
                "blocks": "caller-derived-bpz-hypergeometric-and-exact-ising-sigma",
                "coordinates": "principal-real-z-in-zero-one",
                "crossing": "exact-ising-four-sigma-channel-sum",
                "evidence": "series-tail-nome-branch-crossing",
                "claim": "bounded-degenerate-reference-not-general-2d-bootstrap",
            },
        ),
    )


def conformal_bootstrap_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    gates = {
        "conformal-bootstrap.global-block": (
            "resource-admission",
            "closed-reference",
            "casimir-residual",
            "truncation-evidence",
        ),
        "conformal-bootstrap.crossing-pmp": (
            "crossing-algebra",
            "exact-decimal-round-trip",
            "finite-psd-audit",
            "approximation-ledger",
        ),
        "conformal-bootstrap.sdpb-provider": (
            "pinned-executables",
            "bounded-host-execution",
            "solver-summary",
            "functional-reconstruction",
            "independent-audit",
        ),
        "conformal-bootstrap.virasoro-reference": (
            "explicit-branch-and-derivation-source",
            "hypergeometric-closed-reference",
            "elliptic-nome",
            "exact-ising-channel-crossing",
            "bounded-family-nonclaim",
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
        for support in conformal_bootstrap_candidate_support_tuples()
    )


__all__ = [
    "conformal_bootstrap_candidate_profiles",
    "conformal_bootstrap_candidate_support_tuples",
]
