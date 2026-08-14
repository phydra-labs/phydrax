#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx

from .._strict import StrictModule
from ._plans import LinearSolvePlan
from ._problems import AbstractLinearProblem


class PreparedLinearSolve(StrictModule):
    """Reusable numerical state bound to exactly one problem and execution plan."""

    problem: AbstractLinearProblem
    plan: LinearSolvePlan
    state: Any
    preconditioning_state: Any
    transformed_state: Any
    numeric_version: int = eqx.field(static=True)

    def __init__(
        self,
        problem: AbstractLinearProblem,
        plan: LinearSolvePlan,
        state: Any,
        /,
        *,
        preconditioning_state: Any = None,
        transformed_state: Any = None,
        numeric_version: int = 0,
    ):
        if not isinstance(problem, AbstractLinearProblem):
            raise TypeError("problem must be an AbstractLinearProblem.")
        if not isinstance(plan, LinearSolvePlan):
            raise TypeError("plan must be a LinearSolvePlan.")
        if plan.problem_id != problem.problem_id:
            raise ValueError("Prepared plan and problem IDs must match.")
        version = int(numeric_version)
        if version < 0:
            raise ValueError("numeric_version must be non-negative.")
        self.problem = problem
        self.plan = plan
        self.state = state
        self.preconditioning_state = preconditioning_state
        self.transformed_state = transformed_state
        self.numeric_version = version


__all__ = ["PreparedLinearSolve"]
