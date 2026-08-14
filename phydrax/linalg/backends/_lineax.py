#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array

from ..._strict import StrictModule
from .._plans import _certified_rank, LinearSolvePlan
from .._policies import BiCGStab, ConjugateGradient
from .._problems import LeastSquaresProblem, LinearSystem, MinimumNormProblem
from .._results import LinearSolveStatus
from ._jax_dense import _metric_diagonal


class LineaxState(StrictModule):
    operator: Any
    adjoint_operator: Any
    solver: Any
    solver_state: Any
    options: dict[str, Any]
    row_scale: Array | None
    source_inverse_square_root: Array | None
    rank: Array


class LineaxBackendOutput(StrictModule):
    value: Array
    status: Array
    iterations: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    rank: Array
    condition_estimate: Array
    singular_values: Array | None


def prepare_lineax(problem, plan: LinearSolvePlan, /) -> LineaxState:
    """Adapt one unbatched Phydrax operator to a private Lineax backend."""
    operator = problem.operator
    if operator.batch_shape:
        raise ValueError("The Lineax backend does not yet accept operator batches.")

    dtype = jnp.result_type(
        *[spec.dtype for spec in jax.tree.leaves(operator.source.structure())]
    )
    if isinstance(problem, LeastSquaresProblem):
        row_scale = jnp.sqrt(_metric_diagonal(operator.target))
        source_inverse_square_root = None
    elif isinstance(problem, MinimumNormProblem):
        row_scale = None
        source_inverse_square_root = jax.lax.rsqrt(_metric_diagonal(operator.source))
    else:
        metric = _metric_diagonal(operator.source)
        row_scale = jnp.sqrt(metric)
        source_inverse_square_root = jax.lax.rsqrt(metric)

    def action(coordinates):
        if source_inverse_square_root is not None:
            coordinates = source_inverse_square_root * coordinates
        vector = operator.source.unflatten(coordinates)
        image = operator.target.flatten(operator.mv(vector))
        return image if row_scale is None else row_scale * image

    structure = jax.ShapeDtypeStruct((operator.source.size,), dtype)
    tags: tuple[object, ...] = ()
    if isinstance(problem, LinearSystem):
        is_real = not jnp.issubdtype(dtype, jnp.complexfloating)
        if operator.properties.certifies("self_adjoint") and is_real:
            tags += (lx.symmetric_tag,)
        if operator.properties.certifies("positive_definite"):
            tags += (lx.positive_semidefinite_tag,)
    lineax_operator = lx.FunctionLinearOperator(
        action,
        structure,
        tags=tags,
        closure_convert=True,
    )
    solver = _solver(plan)
    options: dict[str, Any] = {}
    preconditioner = plan.policy.preconditioner
    if preconditioner is not None:

        def precondition(coordinates):
            primal_coordinates = (
                coordinates
                if source_inverse_square_root is None
                else source_inverse_square_root * coordinates
            )
            residual = operator.source.unflatten(primal_coordinates)
            image = operator.source.flatten(preconditioner.apply(residual))
            return image if row_scale is None else row_scale * image

        preconditioner_tags = (
            (lx.positive_semidefinite_tag,) if preconditioner.positive_definite else ()
        )
        options["preconditioner"] = lx.FunctionLinearOperator(
            precondition,
            structure,
            tags=preconditioner_tags,
            closure_convert=True,
        )
    solver_state = solver.init(lineax_operator, options)
    certified_rank = _certified_rank(operator)
    return LineaxState(
        operator=lineax_operator,
        adjoint_operator=lx.conj(lineax_operator).transpose(),
        solver=solver,
        solver_state=solver_state,
        options=options,
        row_scale=row_scale,
        source_inverse_square_root=source_inverse_square_root,
        rank=jnp.asarray(
            -1 if certified_rank is None else certified_rank,
            dtype=jnp.int32,
        ),
    )


def solve_lineax(
    state: LineaxState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
    *,
    initial_guess: Array | None = None,
) -> LineaxBackendOutput:
    if rhs.ndim != 2:
        raise ValueError("Lineax canonical right-hand sides must have shape (m, k).")
    if state.row_scale is not None:
        rhs = state.row_scale[:, None] * rhs

    mathematical = plan.policy.differentiation.mode in ("mathematical", "rhs-only")

    def run(vector, guess):
        options = dict(state.options)
        if guess is not None:
            options["y0"] = guess
        if mathematical:
            return lx.linear_solve(
                state.operator,
                vector,
                solver=state.solver,
                options=options,
                throw=False,
            )
        return lx.linear_solve(
            state.operator,
            vector,
            solver=state.solver,
            options=options,
            state=state.solver_state,
            throw=False,
        )

    def solve_one(vector, guess):
        solution = run(vector, guess)
        iterations = solution.stats.get("num_steps", jnp.asarray(0, dtype=jnp.int32))
        condition = solution.stats.get("cond_A", jnp.asarray(jnp.nan))
        if plan.method == ConjugateGradient().name:
            matvec_count = 1 + iterations + iterations // 10
            adjoint_matvec_count = jnp.asarray(0, dtype=jnp.int32)
        elif plan.method == BiCGStab().name:
            matvec_count = 1 + 2 * iterations
            adjoint_matvec_count = jnp.asarray(0, dtype=jnp.int32)
        else:
            raise ValueError(f"Unsupported Lineax method {plan.method!r}.")
        return (
            solution.value,
            _status(solution.result),
            iterations,
            condition,
            matvec_count,
            adjoint_matvec_count,
        )

    if initial_guess is None:
        guesses = jnp.zeros(
            (state.operator.in_structure().shape[0], rhs.shape[1]),
            dtype=rhs.dtype,
        )
    elif state.source_inverse_square_root is None:
        guesses = initial_guess
    else:
        guesses = initial_guess / state.source_inverse_square_root[:, None]
    value, status, iterations, condition, matvec_count, adjoint_matvec_count = jax.vmap(
        solve_one
    )(
        jnp.swapaxes(rhs, 0, 1),
        jnp.swapaxes(guesses, 0, 1),
    )
    value = jnp.swapaxes(value, 0, 1)
    if state.source_inverse_square_root is not None:
        value = state.source_inverse_square_root[:, None] * value
    return LineaxBackendOutput(
        value=value,
        status=status,
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        matvec_count=jnp.asarray(matvec_count, dtype=jnp.int32),
        adjoint_matvec_count=jnp.asarray(adjoint_matvec_count, dtype=jnp.int32),
        rank=state.rank,
        condition_estimate=condition,
        singular_values=None,
    )


def _solver(plan: LinearSolvePlan, /):
    tolerance = plan.policy.tolerance
    method = plan.policy.method
    method_name = plan.method if method.name == "auto" else method.name
    if method_name == ConjugateGradient().name:
        return lx.CG(
            rtol=tolerance.relative,
            atol=tolerance.absolute,
            max_steps=tolerance.max_steps,
        )
    if method_name == BiCGStab().name:
        return lx.BiCGStab(
            rtol=tolerance.relative,
            atol=tolerance.absolute,
            max_steps=tolerance.max_steps,
        )
    raise ValueError(f"Unsupported Lineax method {method_name!r}.")


def _status(result, /) -> Array:
    successful = result == lx.RESULTS.successful
    status = jnp.full(
        jnp.shape(successful),
        int(LinearSolveStatus.BREAKDOWN),
        dtype=jnp.int32,
    )
    status = jnp.where(
        successful,
        int(LinearSolveStatus.SUCCESS),
        status,
    )
    status = jnp.where(
        result == lx.RESULTS.max_steps_reached,
        int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
        status,
    )
    status = jnp.where(
        result == lx.RESULTS.stagnation,
        int(LinearSolveStatus.STAGNATION),
        status,
    )
    status = jnp.where(
        result == lx.RESULTS.singular,
        int(LinearSolveStatus.SINGULAR),
        status,
    )
    status = jnp.where(
        result == lx.RESULTS.conlim,
        int(LinearSolveStatus.CONDITION_LIMIT_REACHED),
        status,
    )
    status = jnp.where(
        result == lx.RESULTS.nonfinite_input,
        int(LinearSolveStatus.NONFINITE_INPUT),
        status,
    )
    return status


__all__ = ["LineaxBackendOutput", "prepare_lineax", "solve_lineax"]
