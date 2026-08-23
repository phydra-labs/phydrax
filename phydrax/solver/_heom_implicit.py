#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import ArraySpace, FunctionLinearOperator, LinearSystem, solve
from ._heom import HEOMProblem, HEOMSolution


class HEOMImplicitEvidence(StrictModule):
    linear_residuals: Array
    successful_steps: Array
    valid: Array

    def __init__(self, linear_residuals: ArrayLike, successful_steps: ArrayLike, /):
        self.linear_residuals = jnp.asarray(linear_residuals)
        self.successful_steps = jnp.asarray(successful_steps, dtype=bool)
        self.valid = jnp.all(jnp.isfinite(self.linear_residuals)) & jnp.all(
            self.successful_steps
        )


class HEOMImplicitResult(StrictModule):
    solution: HEOMSolution
    evidence: HEOMImplicitEvidence
    valid: Array

    def __init__(self, solution: HEOMSolution, evidence: HEOMImplicitEvidence, /):
        self.solution = solution
        self.evidence = evidence
        self.valid = solution.valid & evidence.valid


def solve_heom_backward_euler(
    problem: HEOMProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
) -> HEOMImplicitResult:
    """Matrix-free backward Euler for the linear HEOM generator."""
    step = jnp.asarray(step_size, dtype=float).reshape(())
    state = problem.initial_state
    shape = state.shape
    space = ArraySpace(shape, dtype=state.dtype, space_id=f"{problem.problem_id}:ado")
    residuals = []
    successful = []
    roots = [state[0]]
    for index in range(int(steps)):
        operator = FunctionLinearOperator(
            lambda value: value - step * problem.rhs(value),
            source=space,
            target=space,
            operator_id=f"{problem.problem_id}:be:{index}",
        )
        result = solve(
            LinearSystem(operator, problem_id=f"{problem.problem_id}:be-system"),
            state,
        )
        state = result.value
        residuals.append(result.diagnostics.residual_norm)
        successful.append(result.successful)
        roots.append(state[0])
    solution = HEOMSolution(
        problem,
        jnp.stack(roots),
        state,
        step * jnp.arange(int(steps) + 1),
    )
    evidence = HEOMImplicitEvidence(jnp.stack(residuals), jnp.stack(successful))
    return HEOMImplicitResult(solution, evidence)


__all__ = [
    "HEOMImplicitEvidence",
    "HEOMImplicitResult",
    "solve_heom_backward_euler",
]
