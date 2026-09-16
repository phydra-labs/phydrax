#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Maturity-neutral support coordinates for magnetic and BdG candidate profiles."""

from __future__ import annotations

from ...qualification import CapabilityProfile, SupportTuple


_SPECS = (
    (
        "condensed-matter.magnetism.classical-spin-energy",
        {
            "state": "classical-unit-vector-s2",
            "bond-counting": "once-stable-particle-id-orientation",
            "channels": "exchange-gamma-dmi-anisotropy-zeeman",
            "units": "explicit-energy-field-moment",
            "capacity": "caller-explicit-hard-policy",
        },
    ),
    (
        "condensed-matter.magnetism.deterministic-llg",
        {
            "equation": "gilbert",
            "integrator": "rkmk-fixed-step",
            "geometry": "product-s2",
            "capacity": "caller-explicit-hard-policy",
        },
    ),
    (
        "condensed-matter.magnetism.thermal-stratonovich-llg",
        {
            "equation": "gilbert-stratonovich-fdt",
            "integrator": "srkmk-fixed-step",
            "prng": "caller-wiener-realization",
            "capacity": "caller-explicit-hard-policy",
        },
    ),
    (
        "condensed-matter.magnetism.quantum-spin-lattice",
        {
            "spin-normalization": "dimensionless-s-in-hbar-units",
            "compiler": "operators-quantum-lattice",
            "models": "heisenberg-xxz-tfim-oriented-dmi-spin-one",
            "capacity": "compiler-resource-policy",
        },
    ),
    (
        "condensed-matter.magnetism.collinear-lswt",
        {
            "reference": "caller-torque-stationary-collinear",
            "metric": "bosonic-krein",
            "periodic-input": "canonical-translation-family-evaluation",
            "capacity": "caller-explicit-hard-policy",
        },
    ),
    (
        "condensed-matter.magnetism.supplied-soc-spin-observables",
        {
            "soc": "supplied-l-dot-s",
            "spin-order": "orbital-major-up-down",
            "degeneracy": "projected-cluster-matrices",
            "capacity": "caller-explicit-hard-policy",
        },
    ),
    (
        "condensed-matter.magnetism.caller-supplied-symmetry",
        {
            "group": "caller-finite-metric-isometry",
            "antiunitary": "z2-homomorphism",
            "constraints": "rank-certified-real-linear",
            "database": False,
            "capacity": "caller-explicit-hard-policy",
        },
    ),
)

_REQUIRED_GATES = (
    "scientific-validation",
    "resource-envelope",
    "lifecycle-restore",
    "runtime-distribution",
    "documentation-nonclaims",
)


def magnetism_support_tuples() -> tuple[SupportTuple, ...]:
    """Return exact coordinates; maturity is intentionally absent from content."""

    return tuple(
        SupportTuple(capability, attributes) for capability, attributes in _SPECS
    )


def magnetism_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return unreleased profiles over the same tuples future release must retain."""

    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_REQUIRED_GATES,
            released=False,
        )
        for support in magnetism_support_tuples()
    )


__all__ = ["magnetism_candidate_profiles", "magnetism_support_tuples"]
