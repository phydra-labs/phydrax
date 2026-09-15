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
