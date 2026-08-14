#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from matfree import lstsq as matfree_lstsq

from ..._strict import StrictModule
from .._plans import _certified_rank, LinearSolvePlan
from .._policies import LSMR
from .._results import LinearSolveStatus


class MatfreeState(StrictModule):
    problem: Any


class MatfreeBackendOutput(StrictModule):
    value: Array
    status: Array
    iterations: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    rank: Array
    condition_estimate: Array
    singular_values: Array | None


def prepare_matfree(problem: Any, plan: LinearSolvePlan, /) -> MatfreeState:
    if plan.backend != "matfree" or plan.method != LSMR().name:
        raise ValueError("Matfree preparation currently supports only LSMR plans.")
    return MatfreeState(problem)


def solve_matfree(
    state: MatfreeState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
    *,
    initial_guess: Array | None = None,
) -> MatfreeBackendOutput:
    if rhs.ndim != 2:
        raise ValueError("Matfree canonical right-hand sides must have shape (m, k).")
    problem = state.problem
    if plan.policy.differentiation.mode == "rhs-only":
        problem = jax.tree.map(
            lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
            problem,
        )
    operator = problem.operator
    method = plan.policy.method
    selected = method if isinstance(method, LSMR) else LSMR()
    max_steps = plan.policy.tolerance.max_steps or max(
        operator.source.size, operator.target.size
    )
    guesses = (
        jnp.zeros((operator.source.size, rhs.shape[1]), dtype=rhs.dtype)
        if initial_guess is None
        else initial_guess
    )
    if guesses.shape != (operator.source.size, rhs.shape[1]):
        raise ValueError("initial_guess must match canonical solution and RHS axes.")

    def adjoint_action(target):
        return operator.source.flatten(
            operator.adjoint_mv(operator.target.unflatten(target))
        )

    def forward_action(source):
        return operator.target.flatten(operator.mv(operator.source.unflatten(source)))

    certified_rank = _certified_rank(operator)

    def solve_column(target, guess):
        target_norm = jnp.linalg.norm(target)
        effective_tolerance = plan.policy.tolerance.relative + jnp.where(
            target_norm > 0.0,
            plan.policy.tolerance.absolute / target_norm,
            0.0,
        )
        full_rank = certified_rank == min(
            operator.source.size,
            operator.target.size,
        )
        runner = matfree_lstsq.lsmr(
            atol=effective_tolerance,
            btol=effective_tolerance,
            ctol=1.0 / selected.condition_limit,
            maxiter=max_steps,
            custom_vjp=False,
            is_full_rank=full_rank,
        )
        value, stats = runner(
            adjoint_action,
            target,
            x0=guess,
            damp=selected.damping,
        )
        return value, (
            stats["iteration_count"],
            stats["norm_residual"],
            stats["norm_At_residual"],
            stats["cond_A"],
            stats["istop"],
        )

    value, auxiliary = jax.vmap(solve_column, in_axes=(1, 1), out_axes=(1, 0))(
        rhs, guesses
    )
    iterations, _, _, condition, stop = auxiliary
    status = jnp.full(stop.shape, int(LinearSolveStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        stop == 3,
        int(LinearSolveStatus.CONDITION_LIMIT_REACHED),
        status,
    )
    status = jnp.where(
        stop == 7,
        int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
        status,
    )
    status = jnp.where(
        stop < 0,
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        status,
    )
    if plan.policy.differentiation.mode == "none":
        value = jax.lax.stop_gradient(value)
    rank = certified_rank
    return MatfreeBackendOutput(
        value=value,
        status=status,
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        matvec_count=jnp.asarray(iterations + 1, dtype=jnp.int32),
        adjoint_matvec_count=jnp.asarray(iterations + 1, dtype=jnp.int32),
        rank=jnp.asarray(-1 if rank is None else rank, dtype=jnp.int32),
        condition_estimate=condition,
        singular_values=None,
    )


__all__ = [
    "MatfreeBackendOutput",
    "MatfreeState",
    "prepare_matfree",
    "solve_matfree",
]
