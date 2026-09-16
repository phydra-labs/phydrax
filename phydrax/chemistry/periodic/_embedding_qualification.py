#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Maturity-neutral profiles and campaigns for implemented Green/embedding paths."""

from __future__ import annotations

from ...qualification import (
    CampaignRole,
    CapabilityProfile,
    ScientificCampaign,
    ScientificCase,
    SupportTuple,
)


_SPECS = (
    (
        "quantum.green.matsubara-dlr",
        {
            "statistics": "fermionic-or-bosonic",
            "representations": "matsubara-imaginary-time-dlr-lehmann",
            "evidence": "moments-causality-transform-residual",
        },
        ("transform-roundtrip", "moment-closure", "statistics-boundary"),
    ),
    (
        "solver.impurity.causal-anderson-bath-fit",
        {
            "bath": "finite-normal-anderson",
            "fit": "nonnegative-strength-projected-gradient",
            "evidence": "causality-moments-fit-finite-bath",
        },
        ("causality", "moment-residual", "fit-refinement"),
    ),
    (
        "solver.impurity.all-sector-ed",
        {
            "model": "normal-single-orbital-finite-bath",
            "sectors": "all-particle-number-direct-bases",
            "output": "green-self-energy-density-double-occupancy",
        },
        ("lehmann-parity", "dyson", "density-moments"),
    ),
    (
        "chemistry.periodic.embedding.single-site-dmft",
        {
            "lattice": "normal-orthonormal-one-orbital",
            "impurity": "finite-bath-all-sector-ed",
            "closure": "self-energy-and-particle-number",
            "derivative": "smooth-branch-implicit-root-only",
        },
        ("fixed-point", "particle-number", "causality", "implicit-jacobian"),
    ),
    (
        "quantum.green.maximum-entropy-continuation",
        {
            "input": "imaginary-axis-covariance-aware",
            "spectrum": "nonnegative-normalized-grid",
            "selection": "explicit-alpha-policy",
            "evidence": "forward-residual-sum-rule-stability",
        },
        (
            "forward-reconstruction",
            "positivity",
            "normalization",
            "regularization-stability",
        ),
    ),
)
_GATES = (
    "scientific-validation",
    "resource-envelope",
    "lifecycle-restore",
    "runtime-distribution",
    "documentation-nonclaims",
)


def green_embedding_support_tuples() -> tuple[SupportTuple, ...]:
    return tuple(
        SupportTuple(capability, attributes) for capability, attributes, _ in _SPECS
    )


def green_embedding_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{support.capability}.profile",
            "phydrax",
            "candidate",
            (support,),
            required_gates=_GATES,
            released=False,
        )
        for support in green_embedding_support_tuples()
    )


def _campaign(capability: str, criteria: tuple[str, ...], /) -> ScientificCampaign:
    slug = capability.replace(".", "-")
    calibration = ScientificCase(
        f"{slug}-calibration",
        f"{slug}-calibration-unit",
        capability,
        "analytic-control",
        f"{slug}-calibration-preparation",
        f"{slug}-calibration-batch",
        (f"source:{slug}:analytic",),
    )
    locked = ScientificCase(
        f"{slug}-locked",
        f"{slug}-locked-unit",
        capability,
        "independent-locked-control",
        f"{slug}-locked-preparation",
        f"{slug}-locked-batch",
        (f"source:{slug}:independent",),
    )
    return ScientificCampaign(
        (calibration, locked),
        (
            CampaignRole("calibration", (calibration.case_id,)),
            CampaignRole("locked_evaluation", (locked.case_id,)),
        ),
        criteria_ids=criteria,
    )


def green_embedding_candidate_campaigns() -> tuple[ScientificCampaign, ...]:
    return tuple(_campaign(capability, criteria) for capability, _, criteria in _SPECS)


__all__ = [
    "green_embedding_candidate_campaigns",
    "green_embedding_candidate_profiles",
    "green_embedding_support_tuples",
]
