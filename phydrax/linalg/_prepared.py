#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ._binding import LinearSolveTemplate
from ._preconditioning import PreparedPreconditioner
from ._problems import AbstractLinearProblem


if TYPE_CHECKING:
    from ._spaces import RHSLayout


class PreparedLinearSolve(StrictModule):
    """Numerical state bound to one problem and reusable symbolic template."""

    problem: AbstractLinearProblem
    template: LinearSolveTemplate
    state: Any
    preconditioning_state: PreparedPreconditioner | None
    numeric_version: Array

    def __init__(
        self,
        problem: AbstractLinearProblem,
        template: LinearSolveTemplate,
        state: Any,
        /,
        *,
        preconditioning_state: PreparedPreconditioner | None = None,
        numeric_version: Any = 0,
    ):
        if not isinstance(problem, AbstractLinearProblem):
            raise TypeError("problem must be an AbstractLinearProblem.")
        if not isinstance(template, LinearSolveTemplate):
            raise TypeError("template must be a LinearSolveTemplate.")
        if template.plan.problem_id != problem.problem_id:
            raise ValueError("Prepared template and problem IDs must match.")
        if preconditioning_state is not None and not isinstance(
            preconditioning_state, PreparedPreconditioner
        ):
            raise TypeError(
                "preconditioning_state must be a PreparedPreconditioner or None."
            )
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.ndim != 0:
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version,
            version < 0,
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
