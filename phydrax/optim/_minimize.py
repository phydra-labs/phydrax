#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from jaxtyping import PyTree

from .._iteration import IterationPlan
from ._iteration import attach_terminal_optimization_iteration
from ._iterative._base import AbstractMinimizationMethod, AbstractScalarIterativeMethod
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
    iteration: IterationPlan | None = None,
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
    if (
        problem.derivative_execution == "explicit-host"
        and not method.capabilities.explicit_host_gradient
    ):
        raise ValueError(
            f"Optimization method {method.method_id!r} does not support explicit host gradients."
        )
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination or None.")
    if iteration is not None and not isinstance(iteration, IterationPlan):
        raise TypeError("iteration must be IterationPlan or None.")
    if isinstance(method, AbstractScalarIterativeMethod):
        result = method.solve(
            problem,
            initial_parameters,
            termination=termination_,
            args=args,
            iteration=iteration,
        )
    else:
        if iteration is not None and iteration.granularity != "terminal":
            raise ValueError(
                "This optimization method supports terminal iteration evidence only."
            )
        result = method.solve(
            problem,
            initial_parameters,
            termination=termination_,
            args=args,
        )
    return attach_terminal_optimization_iteration(
        result,
        iteration,
        method.method_id,
    )


__all__ = ["minimize"]
