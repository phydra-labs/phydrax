#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import ArraySpace, FunctionLinearOperator, LinearSystem, solve
from ._bdf_method import bdf_shift_offset
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


class HEOMTierBlockPreconditioner(StrictModule):
    diagonal: Array

    def __init__(
        self,
        problem: HEOMProblem,
        shift: ArrayLike,
        /,
    ):
        decay = jnp.real(problem.hierarchy.multi_indices @ problem.expansion.exponents)
        self.diagonal = jnp.asarray(shift) + decay

    def apply(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        return values / self.diagonal[:, None, None]


class HEOMBDFEvidence(StrictModule):
    linear_residuals: Array
    successful_steps: Array
    orders: Array
    preconditioned_rhs_norms: Array
    valid: Array

    def __init__(
        self,
        linear_residuals: ArrayLike,
        successful_steps: ArrayLike,
        orders: ArrayLike,
        preconditioned_rhs_norms: ArrayLike,
        /,
    ):
        self.linear_residuals = jnp.asarray(linear_residuals)
        self.successful_steps = jnp.asarray(successful_steps, dtype=bool)
        self.orders = jnp.asarray(orders, dtype=jnp.int32)
        self.preconditioned_rhs_norms = jnp.asarray(preconditioned_rhs_norms)
        self.valid = (
            jnp.all(jnp.isfinite(self.linear_residuals))
            & jnp.all(self.successful_steps)
            & jnp.all(jnp.isfinite(self.preconditioned_rhs_norms))
        )


class HEOMBDFResult(StrictModule):
    solution: HEOMSolution
    evidence: HEOMBDFEvidence
    valid: Array

    def __init__(self, solution: HEOMSolution, evidence: HEOMBDFEvidence, /):
        self.solution = solution
        self.evidence = evidence
        self.valid = solution.valid & evidence.valid


def solve_heom_bdf(
    problem: HEOMProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    maximum_order: int = 5,
) -> HEOMBDFResult:
    """Fixed-step variable-order BDF1–5 with exact linear HEOM action."""
    order_limit = int(maximum_order)
    if not 1 <= order_limit <= 5:
        raise ValueError("maximum_order must lie in [1,5].")
    step = jnp.asarray(step_size, dtype=float).reshape(())
    if int(steps) < 0 or float(step) <= 0.0:
        raise ValueError("HEOM BDF steps and step_size must be positive.")
    state = problem.initial_state
    space = ArraySpace(
        state.shape,
        dtype=state.dtype,
        space_id=f"{problem.problem_id}:bdf-ado",
    )
    history = jnp.broadcast_to(state, (5,) + state.shape)
    history_times = -step * jnp.arange(5)
    roots = [state[0]]
    residuals = []
    successful = []
    orders = []
    preconditioned = []
    for index in range(int(steps)):
        target_time = (index + 1) * step
        order = min(index + 1, order_limit)
        shift, offset = bdf_shift_offset(
            history,
            history_times,
            target_time,
            jnp.asarray(order, dtype=jnp.int32),
        )
        operator = FunctionLinearOperator(
            lambda value: shift * value - problem.rhs(value),
            source=space,
            target=space,
            operator_id=f"{problem.problem_id}:bdf:{index}:order-{order}",
        )
        right_hand_side = -offset
        result = solve(
            LinearSystem(
                operator,
                problem_id=f"{problem.problem_id}:bdf-system",
            ),
            right_hand_side,
        )
        state = result.value
        preconditioner = HEOMTierBlockPreconditioner(problem, shift)
        preconditioned.append(jnp.linalg.norm(preconditioner.apply(right_hand_side)))
        residuals.append(result.diagnostics.residual_norm)
        successful.append(result.successful)
        orders.append(order)
        roots.append(state[0])
        history = jnp.concatenate((state[None, ...], history[:-1]), axis=0)
        history_times = jnp.concatenate((target_time[None], history_times[:-1]), axis=0)
    solution = HEOMSolution(
        problem,
        jnp.stack(roots),
        state,
        step * jnp.arange(int(steps) + 1),
    )
    evidence = HEOMBDFEvidence(
        jnp.stack(residuals) if residuals else jnp.zeros((0,)),
        jnp.stack(successful) if successful else jnp.zeros((0,), dtype=bool),
        jnp.asarray(orders),
        jnp.stack(preconditioned) if preconditioned else jnp.zeros((0,)),
    )
    return HEOMBDFResult(solution, evidence)


__all__ = [
    "HEOMBDFEvidence",
    "HEOMBDFResult",
    "HEOMImplicitEvidence",
    "HEOMImplicitResult",
    "HEOMTierBlockPreconditioner",
    "solve_heom_backward_euler",
    "solve_heom_bdf",
]
