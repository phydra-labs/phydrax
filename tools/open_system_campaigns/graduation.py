#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

from phydrax.operators.quantum import OpenSystemPromotionPolicy

from .contracts import OpenSystemGraduationResult, VerifiedOpenSystemCampaign


CAMPAIGN_IDS = (
    "gaussian-affine-v1",
    "dense-trajectories-v1",
    "mps-trajectories-v1",
    "lpdo-xxz-v1",
    "heom-spin-boson-v1",
    "constructive-memory-v1",
    "process-recovery-v1",
    "causal-distillation-v1",
    "enumerable-neural-v1",
)


def _policy(campaign_id: str) -> OpenSystemPromotionPolicy:
    requirements = {
        "gaussian-affine-v1": (
            ("time-step",),
            ("analytic-covariance-error",),
            ("representation-closure",),
        ),
        "dense-trajectories-v1": (
            ("relative-tolerance", "trajectory-count"),
            ("coupled-observable-difference", "dense-reference-difference"),
            ("trace", "hermiticity", "positivity"),
        ),
        "mps-trajectories-v1": (
            ("time-step", "bond-dimension"),
            (
                "maximum-discarded-weight",
                "event-time-reference-error",
                "maximum-root-residual",
            ),
            ("trace", "representation-closure"),
        ),
        "lpdo-xxz-v1": (
            ("time-step", "physical-bond", "purification-rank"),
            (
                "time-refinement-error",
                "maximum-trace-residual",
                "maximum-bond-discarded-weight",
                "maximum-kraus-discarded-weight",
                "maximum-canonical-residual",
            ),
            ("trace", "positivity", "representation-closure"),
        ),
        "heom-spin-boson-v1": (
            ("hierarchy-depth", "bath-pole-order", "relative-tolerance"),
            (
                "depth-difference",
                "bath-difference",
                "adaptive-tolerance-difference",
                "maximum-local-error-ratio",
                "maximum-top-tier-norm",
            ),
            ("trace", "hermiticity", "positivity"),
        ),
        "constructive-memory-v1": (
            ("memory-step", "memory-horizon"),
            (
                "time-refinement-error",
                "maximum-trace-preservation-residual",
                "maximum-complete-positivity-violation",
            ),
            (
                "trace",
                "hermiticity",
                "positivity",
                "complete-positivity",
                "trace-preservation",
            ),
        ),
        "process-recovery-v1": (
            ("memory-dimension", "intervention-settings"),
            ("held-out-probability-error", "post-fit-to-pre-fit-error-ratio"),
            ("trace", "positivity", "complete-positivity", "trace-preservation"),
        ),
        "causal-distillation-v1": (
            ("memory-dimension", "slot-count"),
            ("held-out-probability-error", "post-fit-to-pre-fit-error-ratio"),
            ("trace", "positivity", "complete-positivity", "trace-preservation"),
        ),
        "enumerable-neural-v1": (
            ("sample-count", "time-step", "parameter-dimension"),
            (
                "rate-standard-error",
                "initial-rate-reference-error",
                "jump-projection-residual",
            ),
            ("trace", "representation-closure"),
        ),
    }
    axes, quantities, physicality = requirements[campaign_id]
    return OpenSystemPromotionPolicy(
        axes,
        quantities,
        physicality,
        require_precision=True,
        policy_id=f"{campaign_id}:policy",
    )


def run_open_system_graduation(
    campaigns: Sequence[VerifiedOpenSystemCampaign],
    /,
) -> OpenSystemGraduationResult:
    """Graduate exact verified campaign artifacts without rerunning solvers."""
    campaigns_ = tuple(campaigns)
    ids = tuple(value.record.campaign_id for value in campaigns_)
    if ids != CAMPAIGN_IDS:
        raise ValueError(
            "Graduation requires one ordered verified artifact per campaign ID."
        )
    policies = tuple(_policy(campaign_id) for campaign_id in CAMPAIGN_IDS)
    return OpenSystemGraduationResult(campaigns_, policies)


__all__ = ["CAMPAIGN_IDS", "run_open_system_graduation"]
