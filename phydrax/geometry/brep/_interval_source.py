"""Interval callback contracts for pure programs and scoped source bounds."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from .._interval_enclosure import Interval
from ._request_preparations import TrimRequestPreparations


class IntervalEvaluator(Protocol):
    def evaluate(self, lower: np.ndarray, upper: np.ndarray, /) -> Interval: ...


@runtime_checkable
class ScopedIntervalEvaluator(Protocol):
    """Host bounds whose preparation consumes the current request view."""

    def evaluate_scope(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> Interval: ...


def evaluate_interval(
    evaluator: IntervalEvaluator | ScopedIntervalEvaluator,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    match evaluator:
        case ScopedIntervalEvaluator():
            bounds = evaluator.evaluate_scope(
                lower, upper, budget=budget, preparations=preparations
            )
        case _:
            bounds = evaluator.evaluate(lower, upper)
    # B-Rep residuals and derivatives have binary64 numerical outputs. Existing
    # float64 arrays are borrowed; scalar outputs gain their explicit rank-zero view.
    return (
        np.asarray(bounds[0], dtype=np.float64),
        np.asarray(bounds[1], dtype=np.float64),
    )
