#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased exact-scope quantum-lattice candidate declarations."""

from __future__ import annotations

from ....qualification import CapabilityProfile, SupportTuple


def quantum_lattice_candidate_support_tuples() -> tuple[SupportTuple, ...]:
    """Return exact candidate coordinates; these records are not evidence."""
    return (
        SupportTuple(
            "quantum-lattice.fixed-sector",
            {
                "basis": "direct-rank-unrank",
                "operator": "matrix-free-no-ambient-enumeration",
                "statistics": "fermion-spin-boson",
                "execution": "single-device-complex128",
            },
        ),
        SupportTuple(
            "quantum-lattice.orbit-sector",
            {
                "basis": "finite-monomial-group-character-projection",
                "operator": "prepared-coalesced-reduced-routes",
                "representation": "one-dimensional-unitary-character",
                "evidence": "closure-invariance-hermiticity-full-sector-parity",
            },
        ),
        SupportTuple(
            "quantum-lattice.tpq",
            {
                "ensemble": "canonical-fixed-sector",
                "probes": "random-phase-ratio-of-sums",
                "propagation": "native-krylov-exponential-action",
                "execution": "single-device-complex128",
            },
        ),
        SupportTuple(
            "quantum-lattice.response",
            {
                "temperature": "zero",
                "channel": "explicit-source-target-sector",
                "method": "shifted-lanczos-retarded",
                "evidence": "moments-positivity-residuals",
            },
        ),
        SupportTuple(
            "quantum-lattice.response",
            {
                "temperature": "finite",
                "channel": "explicit-source-target-sector",
                "method": "tpq-time-correlation",
                "evidence": "raw-probes-moments-positivity-kms",
            },
        ),
        SupportTuple(
            "quantum-lattice.stochastic-sign-free",
            {
                "scope": "method-specific-bounded-control",
                "input": "bounded-raw-chain-order-sign-records",
                "evidence": "order-autocorrelation-ess-covariance",
                "claim": "method-specific-control-not-sign-cure",
            },
        ),
        SupportTuple(
            "quantum-lattice.configuration-columns",
            {
                "address": "rank-free-exact-packed-uint32-words",
                "domain": "explicit-species-product-or-exact-abelian-charges",
                "operator": "sparse-outgoing-target-source",
                "binding": "separate-topology-and-occurrence-numeric-slots",
                "sampling": "physical-raw-route-probability-with-null-attempts",
                "statistics": "finite-fermion-spin-boson",
                "execution": "single-device-complex128-float64",
                "resources": "explicit-transition-route-coordinate-workspace-limits",
            },
        ),
        *(
            SupportTuple(
                "quantum-lattice.positive-guide",
                {
                    "provider": "frozen-strictmodule-log-amplitude-magnitude",
                    "mapping": "explicit-packed-address-provider-domain-binding",
                    "node-policy": node_policy,
                    "similarity": "positive-invertible-original-self-adjoint",
                    "metric": "same-actual-guide-inverse-square",
                    "observables": "original-physical-real-or-complex",
                    "global-support": "provider-declaration-not-encountered-proof",
                    "execution": "single-device-complex128-float64",
                },
            )
            for node_policy in ("reject", "positive-log-floor")
        ),
        *(
            SupportTuple(
                "quantum-lattice.projector-monte-carlo",
                {
                    "target": "self-adjoint-ground-state",
                    "address": "rank-free-exact-packed-uint32-words",
                    "spawn": spawn,
                    "compression": compression,
                    "controller": controller,
                    "guide": guide,
                    "annihilation": "complete-seeded-compensated-before-late-compression",
                    "estimator": "physical-projected-and-per-time-aggregated-replica-ratio",
                    "history": "complete-incoming-shift-finite-window-reweighting",
                    "uncertainty": "joint-common-block-covariance-denominator-gated",
                    "restart": "atomic-commit-explicit-same-draw-resource-replay",
                    "execution": "single-device-complex128-float64",
                    "resources": "explicit-support-group-event-attempt-source-history-byte-limits",
                    "approximation": "no-initiator-finite-euler-finite-population",
                    "claim": "candidate-not-sign-cure-or-unbiased-stationary-certificate",
                },
            )
            for spawn in ("exact", "sampled", "semistochastic")
            for compression in ("none", "threshold")
            for controller in ("fixed-shift", "double-log")
            for guide in (
                "unguided",
                "frozen-positive-reject",
                "frozen-positive-log-floor",
            )
        ),
    )


def quantum_lattice_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    supports = quantum_lattice_candidate_support_tuples()
    gates = {
        "quantum-lattice.fixed-sector": (
            "resource-admission",
            "car-and-charge-conservation",
            "cross-target-parity",
            "locked-reference",
        ),
        "quantum-lattice.orbit-sector": (
            "resource-admission",
            "group-closure-and-character-order",
            "operator-invariance",
            "full-versus-quotient-spectrum",
            "locked-reference",
        ),
        "quantum-lattice.tpq": (
            "resource-admission",
            "statistical-coverage",
            "krylov-residual",
            "locked-reference",
        ),
        "quantum-lattice.response": (
            "resource-admission",
            "source-target-charge-map",
            "spectral-positivity-and-moments",
            "locked-reference",
        ),
        "quantum-lattice.stochastic-sign-free": (
            "resource-admission",
            "raw-chain-retention",
            "sign-and-effective-sample-evidence",
            "locked-control",
        ),
        "quantum-lattice.configuration-columns": (
            "resource-admission-and-overflow-refusal",
            "fermionic-sign-and-complex-column-orientation",
            "raw-route-probability-and-null-attempt-expectation",
            "numeric-binding-and-same-support-refresh",
            "locked-independent-reference",
        ),
        "quantum-lattice.positive-guide": (
            "resource-and-numerical-range-refusal",
            "original-self-adjoint-and-frozen-guide-binding",
            "same-actual-inverse-guide-metric",
            "real-and-complex-physical-observable-recovery",
            "locked-independent-reference",
        ),
        "quantum-lattice.projector-monte-carlo": (
            "resource-admission-and-atomic-overflow-refusal",
            "fermionic-sign-and-complex-phase",
            "physical-raw-route-probability-and-compression-expectation",
            "complete-annihilation-and-raw-history-retention",
            "same-actual-guide-physical-metric",
            "joint-covariance-and-denominator-safety",
            "checkpoint-and-same-draw-resource-replay",
            "population-history-timestep-and-sign-resolution",
            "locked-independent-reference",
        ),
    }
    grouped: dict[str, list[SupportTuple]] = {}
    for support in supports:
        grouped.setdefault(support.capability, []).append(support)
    return tuple(
        CapabilityProfile(
            capability,
            "phydrax",
            "candidate",
            tuple(grouped[capability]),
            required_gates=gates[capability],
            released=False,
        )
        for capability in sorted(grouped)
    )


__all__ = [
    "quantum_lattice_candidate_profiles",
    "quantum_lattice_candidate_support_tuples",
]
