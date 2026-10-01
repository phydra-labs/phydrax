# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Unreleased, evidence-gated meshfree capability declarations."""

from __future__ import annotations

from ...qualification._registry import CapabilityProfile, SupportTuple


def meshfree_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Declare candidate support, without treating campaign availability as release."""
    specifications: tuple[
        tuple[str, dict[str, str | int | bool], tuple[str, ...]], ...
    ] = (
        (
            "strong-form",
            {
                "dimension": "2,3",
                "methods": "phs-rbf-fd,gmls",
                "polynomials": "bounded-degree",
                "neighbors": "bounded-morton",
                "row-refusal": True,
            },
            ("analytic", "convergence", "row-refusal"),
        ),
        (
            "elliptic-solve",
            {
                "dimension": "2,3",
                "boundary": "dirichlet,neumann,robin",
                "operator": "prepared-sparse",
                "stabilization": "configured-not-universal",
            },
            ("analytic", "convergence", "boundary", "true-residual"),
        ),
        (
            "multilevel",
            {
                "dimension": "2,3",
                "hierarchy": "native-point-cloud",
                "coarse-space": "declared-affine-degree",
                "fine-system-comparison": "native-ilu,native-smoothed-aggregation",
            },
            ("analytic", "true-residual", "fine-system-cost", "transfer-reproduction"),
        ),
        (
            "conservative-exterior",
            {
                "dimension": "2,3",
                "conservation": "incidence-ledger",
                "moments": "feasibility-certified",
                "nonnegative": "may-refuse",
                "coordinate-derivative": "fixed-topology",
            },
            (
                "analytic",
                "conservation",
                "coercivity",
                "feasibility",
                "coordinate-derivative",
            ),
        ),
        (
            "surface-operators",
            {
                "embedding-dimension": 3,
                "intrinsic-dimension": 2,
                "geometry": "supplied-or-sampled-normals",
                "reach": "estimate-not-certificate",
            },
            ("analytic", "convergence", "normal-comparison", "geometry-refusal"),
        ),
        (
            "moving-surface",
            {
                "embedding-dimension": 3,
                "evolution": "native-material-ale",
                "remap": "explicit-mass-ledger",
                "topology": "explicit-epoch",
            },
            (
                "analytic",
                "conservation",
                "dilution",
                "repeated-remap",
                "epoch-transition",
            ),
        ),
        (
            "bulk-surface-exchange",
            {
                "kinetics": "langmuir",
                "coupling": "native-bulk-surface",
                "balance": "closed-exchange-ledger",
                "window": "explicit-buffer",
            },
            ("analytic", "convergence", "conservation", "window-lag"),
        ),
        (
            "learned-constitutive-flux",
            {
                "law": "bounded-covered-features",
                "solve": "native-nonlinear",
                "derivative": "implicit",
                "coverage-refusal": True,
            },
            (
                "analytic",
                "law-recovery",
                "implicit-derivative",
                "coverage",
                "newton-refusal",
            ),
        ),
    )
    common = ("resource", "public-workflow", "scaling")
    return tuple(
        CapabilityProfile(
            f"meshfree.{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(f"meshfree.{name}", attributes),),
            required_gates=(*gates, *common),
            released=False,
        )
        for name, attributes, gates in specifications
    )


__all__ = ["meshfree_candidate_profiles"]
