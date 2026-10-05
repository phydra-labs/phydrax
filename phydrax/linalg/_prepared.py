#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from jax import Array

from .._strict import StrictModule
from ..typing import checked
from ._binding import LinearSolveTemplate
from ._preconditioning import PreparedPreconditioner
from ._problems import AbstractLinearProblem
from ._results import _numeric_versions


if TYPE_CHECKING:
    from ._spaces import RHSLayout


class PreparedLinearSolve(StrictModule):
    """Numerical state bound to one problem and reusable symbolic template."""

    problem: AbstractLinearProblem
    template: LinearSolveTemplate
    state: Any
    preconditioning_state: PreparedPreconditioner | None
    numeric_version: Array

    @checked
    def __init__(
        self,
        problem: AbstractLinearProblem,
        template: LinearSolveTemplate,
        state: Any,
        /,
        *,
        preconditioning_state: PreparedPreconditioner | None = None,
        numeric_version: Any = 0,
    ) -> None:
        if template.plan.problem_id != problem.problem_id:
            raise ValueError("Prepared template and problem IDs must match.")
        if preconditioning_state is not None and not isinstance(
            preconditioning_state, PreparedPreconditioner
        ):
            raise TypeError(
                "preconditioning_state must be a PreparedPreconditioner or None."
            )
        (version,) = _numeric_versions(
            (numeric_version,),
            lambda value: value < 0,
            "numeric_version must be scalar.",
            "numeric_version must be non-negative.",
        )
        self.problem = problem
        self.template = template
        self.state = state
        self.preconditioning_state = preconditioning_state
        self.numeric_version = version

    @property
    def plan(self) -> Any:
        return self.template.plan

    @property
    def rhs_layout(self) -> RHSLayout | None:
        return self.plan.rhs_layout

    @property
    def recycling_capacity(self) -> int:
        return self.plan.recycling_capacity

    @property
    def recycling_state_bytes(self) -> int:
        return self.plan.recycling_state_bytes


__all__ = ["PreparedLinearSolve"]
