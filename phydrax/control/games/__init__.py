#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-horizon dynamic-game solvers and evidence."""

from ._layout import PlayerControlPartition
from ._linear_quadratic import (
    finite_horizon_lq_feedback_nash,
    FiniteHorizonLQFeedbackNashDiagnostics,
    FiniteHorizonLQFeedbackNashResult,
    LQFeedbackNashStatus,
)


__all__ = [
    "FiniteHorizonLQFeedbackNashDiagnostics",
    "FiniteHorizonLQFeedbackNashResult",
    "LQFeedbackNashStatus",
    "PlayerControlPartition",
    "finite_horizon_lq_feedback_nash",
]
