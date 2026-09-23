#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

from phydrax.operators.quantum import OpenSystemPromotionPolicy

from .contracts import OpenSystemGraduationResult, VerifiedOpenSystemCampaign


CAMPAIGN_IDS = (
    "gaussian-affine",
    "dense-trajectories",
    "mps-trajectories",
    "lpdo-xxz",
    "heom-spin-boson",
    "constructive-memory",
    "process-recovery",
    "causal-distillation",
    "enumerable-neural",
)

QUANTITY_THRESHOLDS = {
    "gaussian-affine": {"analytic-covariance-error": 1.0e-6},
    "dense-trajectories": {
        "coupled-observable-difference": 0.15,
        "dense-reference-difference": 0.15,
    },
    "mps-trajectories": {
        "maximum-discarded-weight": 1.0e-6,
        "event-time-reference-error": 1.0e-6,
        "maximum-root-residual": 1.0e-8,
    },
    "lpdo-xxz": {
        "time-refinement-error": 1.0e-3,
        "maximum-trace-residual": 1.0e-6,
        "maximum-bond-discarded-weight": 1.0e-6,
        "maximum-kraus-discarded-weight": 1.0e-6,
        "maximum-canonical-residual": 1.0e-6,
    },
    "heom-spin-boson": {
        "depth-difference": 0.1,
        "bath-difference": 0.1,
        "adaptive-tolerance-difference": 1.0e-3,
        "maximum-local-error-ratio": 1.0,
        "maximum-top-tier-norm": 0.1,
    },
    "constructive-memory": {
        "time-refinement-error": 1.0e-3,
        "maximum-trace-preservation-residual": 1.0e-8,
        "maximum-complete-positivity-violation": 1.0e-8,
    },
    "process-recovery": {
        "held-out-probability-error": 1.0e-2,
        "post-fit-to-pre-fit-error-ratio": 0.5,
    },
    "causal-distillation": {
        "held-out-probability-error": 5.0e-2,
        "post-fit-to-pre-fit-error-ratio": 0.75,
    },
    "enumerable-neural": {
        "rate-standard-error": 1.0,
        "initial-rate-reference-error": 0.25,
        "jump-projection-residual": 1.0e-12,
    },
}


def _validate_policy_evidence(campaign: VerifiedOpenSystemCampaign) -> None:
    record = campaign.record
    expected = QUANTITY_THRESHOLDS[record.campaign_id]
    observed = {
        quantity.name: float(quantity.threshold)
        for quantity in record.approximation.quantities
    }
    if observed.keys() != expected.keys() or any(
        not math.isclose(observed[name], threshold, rel_tol=0.0, abs_tol=0.0)
        for name, threshold in expected.items()
    ):
        raise ValueError(
            f"Campaign {record.campaign_id!r} thresholds do not match policy."
        )
    if record.approximation.precision_policy_ids != (record.precision.policy_id,):
        raise ValueError(
            f"Campaign {record.campaign_id!r} precision policy is not bound to evidence."
        )
    replay = record.replay
    expected_replay = (1.0e-6, 0.05, 1.0e-5)
    observed_replay = (
        float(replay.event_time_tolerance),
        float(replay.disagreement_tolerance),
        float(replay.observable_tolerance),
    )
    if observed_replay != expected_replay:
        raise ValueError(
            f"Campaign {record.campaign_id!r} replay tolerances do not match policy."
        )


def _policy(campaign_id: str) -> OpenSystemPromotionPolicy:
    requirements = {
        "gaussian-affine": (
            ("time-step",),
            ("analytic-covariance-error",),
            ("representation-closure",),
        ),
        "dense-trajectories": (
            ("relative-tolerance", "trajectory-count"),
            ("coupled-observable-difference", "dense-reference-difference"),
            ("trace", "hermiticity", "positivity"),
        ),
        "mps-trajectories": (
            ("time-step", "bond-dimension"),
            (
                "maximum-discarded-weight",
                "event-time-reference-error",
                "maximum-root-residual",
            ),
            ("trace", "representation-closure"),
        ),
        "lpdo-xxz": (
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
        "heom-spin-boson": (
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
        "constructive-memory": (
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
        "process-recovery": (
            ("memory-dimension", "intervention-settings"),
            ("held-out-probability-error", "post-fit-to-pre-fit-error-ratio"),
            ("trace", "positivity", "complete-positivity", "trace-preservation"),
        ),
        "causal-distillation": (
            ("memory-dimension", "slot-count"),
            ("held-out-probability-error", "post-fit-to-pre-fit-error-ratio"),
            ("trace", "positivity", "complete-positivity", "trace-preservation"),
        ),
        "enumerable-neural": (
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
    for campaign in campaigns_:
        _validate_policy_evidence(campaign)
    policies = tuple(_policy(campaign_id) for campaign_id in CAMPAIGN_IDS)
    return OpenSystemGraduationResult(campaigns_, policies)


__all__ = ["CAMPAIGN_IDS", "run_open_system_graduation"]
