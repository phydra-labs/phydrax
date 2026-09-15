#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unreleased controlled sign-study profile."""

from ...qualification import CapabilityProfile, SupportTuple


CONTROLLED_SIGN_STUDY_SUPPORT = SupportTuple(
    "sign-problem.controlled-sign-study",
    {
        "input_measure": "phase-quenched-chain",
        "evidence": "raw-phase-ess-covariance-abstention",
        "claim": "diagnostic-not-cure",
    },
)
CONTROLLED_SIGN_STUDY_CANDIDATE = CapabilityProfile(
    "sign-problem.controlled-sign-study.candidate",
    "phydrax",
    "candidate",
    (CONTROLLED_SIGN_STUDY_SUPPORT,),
    required_gates=("raw-chain", "autocorrelation", "abstention", "locked-small-control"),
    released=False,
)


__all__ = ["CONTROLLED_SIGN_STUDY_CANDIDATE", "CONTROLLED_SIGN_STUDY_SUPPORT"]
