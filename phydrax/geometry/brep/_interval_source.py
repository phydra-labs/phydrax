#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Budget-free callback contracts for outward interval source evaluation."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class IntervalEvaluator(Protocol):
    """Evaluate a batch of closed boxes with outward-rounded bounds."""

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]: ...


def evaluate_interval(
    evaluator: IntervalEvaluator,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = evaluator.evaluate(lower, upper)
    return (
        np.asarray(bounds[0], dtype=np.float64),
        np.asarray(bounds[1], dtype=np.float64),
    )


__all__ = ["IntervalEvaluator", "evaluate_interval"]
