#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from jaxtyping import PyTree

from ._iterative._base import AbstractMinimizationMethod
from ._iterative._types import (
    Bounds,
    MinimizationProblem,
    MinimizationResult,
    NonlinearConstraint,
    OptimizationTermination,
)


def minimize(
    problem_or_objective: MinimizationProblem | Callable[[PyTree[Any], Any], Any],
    initial_parameters: PyTree[Any],
    /,
    *,
    method: AbstractMinimizationMethod,
    termination: OptimizationTermination | None = None,
    args: Any = None,
    has_aux: bool = False,
    bounds: Bounds | None = None,
    constraints: Sequence[NonlinearConstraint] = (),
) -> MinimizationResult:
    """Minimize a scalar problem through one explicit method adapter."""

    if isinstance(problem_or_objective, MinimizationProblem):
        if bounds is not None or constraints:
            raise ValueError(
                "bounds and constraints must be declared on an existing "
                "MinimizationProblem, not passed twice."
            )
        problem = problem_or_objective
    else:
        problem = MinimizationProblem(
            problem_or_objective,
            has_aux=has_aux,
            bounds=bounds,
            constraints=constraints,
        )
    if not isinstance(method, AbstractMinimizationMethod):
        raise TypeError("method must be an AbstractMinimizationMethod.")
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination or None.")
    return method.solve(
        problem,
        initial_parameters,
        termination=termination_,
        args=args,
    )


__all__ = ["minimize"]
