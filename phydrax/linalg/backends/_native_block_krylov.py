#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from .._plans import _certified_rank, LinearSolvePlan
from .._policies import BlockCG, BlockGMRES
from .._preconditioners import AbstractPreconditioner
from .._results import LinearSolveStatus
from ._native_krylov import (
    _action_coordinates,
    _preconditioner_action,
    _space_inner,
)


class NativeBlockKrylovState(StrictModule):
    problem: Any
    preconditioner: AbstractPreconditioner | None


class NativeBlockKrylovBackendOutput(StrictModule):
    value: Array
    status: Array
    iterations: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    rank: Array
    condition_estimate: Array
    singular_values: Array | None
    effective_block_rank: Array
    deflated_rhs_count: Array


def prepare_native_block_krylov(
    problem: Any,
    plan: LinearSolvePlan,
    /,
    *,
    preconditioner: AbstractPreconditioner | None = None,
) -> NativeBlockKrylovState:
    if plan.backend != "native-block-krylov":
        raise ValueError(
            "Native block Krylov preparation requires a native-block-krylov plan."
        )
    if plan.rhs_layout is None or plan.rhs_layout.size <= 1:
        raise ValueError("Native block Krylov preparation requires a true block layout.")
    if problem.operator.batch_shape:
        raise ValueError(
            "Native block Krylov preparation requires an unbatched operator."
        )
    if (plan.preconditioner_plan is None) != (preconditioner is None):
        raise ValueError("Prepared preconditioning must match the symbolic solve plan.")
    return NativeBlockKrylovState(problem, preconditioner)


def solve_native_block_krylov(
    state: NativeBlockKrylovState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
    *,
    initial_guess: Array | None = None,
) -> NativeBlockKrylovBackendOutput:
    if rhs.ndim != 2:
        raise ValueError("Native block Krylov right-hand sides must have shape (n, k).")
    if plan.rhs_layout is None or rhs.shape[1] != plan.rhs_layout.size:
        raise ValueError("Canonical RHS width does not match the planned block layout.")
    problem = state.problem
    if plan.policy.differentiation.mode == "rhs-only":
        problem = jax.tree.map(
            lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
            problem,
        )
    source_size = problem.operator.source.size
    if rhs.shape[0] != problem.operator.target.size:
        raise ValueError("Canonical RHS dimension does not match the operator target.")
    guess = (
        jnp.zeros((source_size, rhs.shape[1]), dtype=rhs.dtype)
        if initial_guess is None
        else jnp.asarray(initial_guess)
    )
    if guess.shape != (source_size, rhs.shape[1]):
        raise ValueError("initial_guess must match canonical solution and RHS axes.")

    column_action = lambda vector: _action_coordinates(problem.operator, vector)
    action = lambda block: jax.vmap(column_action, in_axes=1, out_axes=1)(block)
    column_precondition = _preconditioner_action(
        state.preconditioner,
        problem.operator.source,
    )
    precondition = lambda block, iteration: jax.vmap(
        lambda vector: column_precondition(vector, iteration),
        in_axes=1,
        out_axes=1,
    )(block)
    block_gram = lambda left, right: _block_gram(
        problem.operator.source,
        left,
        right,
    )
    tolerance = plan.policy.tolerance
    max_steps = tolerance.max_steps or max(1, source_size)
    effective_relative = max(
        tolerance.relative,
        10.0 * float(jnp.finfo(rhs.real.dtype).eps) * float(source_size),
    )
    thresholds = tolerance.absolute + effective_relative * _column_norms(rhs, block_gram)
    if source_size == 0 and rhs.shape[0] == 0:
        if not isinstance(plan.policy.method, (BlockGMRES, BlockCG)):
            raise ValueError(f"Unsupported true block method {plan.method!r}.")
        if plan.policy.differentiation.mode == "none":
            guess = jax.lax.stop_gradient(guess)
        operator_rank = _certified_rank(problem.operator)
        return NativeBlockKrylovBackendOutput(
            value=guess,
            status=jnp.full(
                (rhs.shape[1],),
                int(LinearSolveStatus.SUCCESS),
                dtype=jnp.int32,
            ),
            iterations=jnp.zeros((rhs.shape[1],), dtype=jnp.int32),
            matvec_count=jnp.asarray(0, dtype=jnp.int32),
            adjoint_matvec_count=jnp.asarray(0, dtype=jnp.int32),
            rank=jnp.asarray(
                -1 if operator_rank is None else operator_rank,
                dtype=jnp.int32,
            ),
            condition_estimate=jnp.asarray(jnp.nan, dtype=rhs.real.dtype),
            singular_values=None,
            effective_block_rank=jnp.asarray(0, dtype=jnp.int32),
            deflated_rhs_count=jnp.asarray(rhs.shape[1], dtype=jnp.int32),
        )

    if isinstance(plan.policy.method, BlockGMRES):
        (
            value,
            iterations,
            matvec_count,
            effective_rank,
            breakdown,
            last_executed_iteration,
        ) = _block_gmres_raw(
            action,
            precondition,
            block_gram,
            rhs,
            guess,
            max_steps=max_steps,
            restart=min(plan.policy.method.restart, max_steps),
            thresholds=thresholds,
        )
    elif isinstance(plan.policy.method, BlockCG):
        (
            value,
            iterations,
            matvec_count,
            effective_rank,
            breakdown,
            last_executed_iteration,
        ) = _block_cg_raw(
            action,
            precondition,
            block_gram,
            rhs,
            guess,
            max_steps=max_steps,
            thresholds=thresholds,
        )
    else:
        raise ValueError(f"Unsupported true block method {plan.method!r}.")

    true_residual = rhs - action(value)
    matvec_count = matvec_count + jnp.asarray(1, dtype=jnp.int32)
    residual_norms = _column_norms(true_residual, block_gram)
    converged = residual_norms <= thresholds
    rhs_finite = jnp.all(jnp.isfinite(rhs), axis=0)
    output_finite = jnp.all(jnp.isfinite(value), axis=0) & jnp.isfinite(residual_norms)
    status = jnp.where(
        converged,
        int(LinearSolveStatus.SUCCESS),
        int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
    ).astype(jnp.int32)
    status = jnp.where(
        breakdown & ~converged,
        int(LinearSolveStatus.BREAKDOWN),
        status,
    )
    status = jnp.where(
        ~rhs_finite,
        int(LinearSolveStatus.NONFINITE_INPUT),
        status,
    )
    status = jnp.where(
        rhs_finite & ~output_finite,
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        status,
    )
    iterations = jnp.where(converged, iterations, last_executed_iteration)
    if plan.policy.differentiation.mode == "none":
        value = jax.lax.stop_gradient(value)
    operator_rank = _certified_rank(problem.operator)
    return NativeBlockKrylovBackendOutput(
        value=value,
        status=status,
        iterations=iterations,
        matvec_count=matvec_count,
        adjoint_matvec_count=jnp.asarray(0, dtype=jnp.int32),
        rank=jnp.asarray(-1 if operator_rank is None else operator_rank, dtype=jnp.int32),
        condition_estimate=jnp.asarray(jnp.nan, dtype=rhs.real.dtype),
        singular_values=None,
        effective_block_rank=effective_rank,
        deflated_rhs_count=jnp.asarray(rhs.shape[1], dtype=jnp.int32) - effective_rank,
    )


def _block_gmres_raw(
    action,
    precondition,
    block_gram,
    rhs: Array,
    initial: Array,
    *,
    max_steps: int,
    restart: int,
    thresholds: Array,
):
    dimension, rhs_count = rhs.shape
    block_width = min(dimension, rhs_count)
    x = initial
    residual = rhs - action(x)
    matvec_count = jnp.asarray(1, dtype=jnp.int32)
    residual_norms = _column_norms(residual, block_gram)
    converged = residual_norms <= thresholds
    iterations = jnp.zeros((rhs_count,), dtype=jnp.int32)
    _, _, _, initial_rank = _rank_revealing_factor(residual, block_gram)
    breakdown = jnp.zeros((rhs_count,), dtype=bool)
    last_executed_iteration = jnp.asarray(0, dtype=jnp.int32)
    cycles = (max_steps + restart - 1) // restart

    for cycle in range(cycles):
        cycle_base = x
        basis = jnp.zeros(
            (dimension, (restart + 1) * block_width),
            dtype=rhs.dtype,
        )
        preconditioned_basis = jnp.zeros(
            (dimension, restart * block_width),
            dtype=rhs.dtype,
        )
        hessenberg = jnp.zeros(
            ((restart + 1) * block_width, restart * block_width),
            dtype=rhs.dtype,
        )
        right = jnp.zeros(
            ((restart + 1) * block_width, rhs_count),
            dtype=rhs.dtype,
        )
        first_basis, beta, active_block, _ = _rank_revealing_factor(
            residual,
            block_gram,
        )
        basis = basis.at[:, :block_width].set(first_basis)
        right = right.at[:block_width].set(beta)

        for local_index in range(restart):
            global_index = cycle * restart + local_index
            if global_index >= max_steps:
                continue

            def execute(operand):
                (
                    basis_,
                    preconditioned_basis_,
                    hessenberg_,
                    x_,
                    residual_,
                    converged_,
                    iterations_,
                    matvec_count_,
                    breakdown_,
                    _,
                    last_executed_iteration_,
                ) = operand
                column_start = local_index * block_width
                column_stop = column_start + block_width
                current_basis = basis_[:, column_start:column_stop]
                z = precondition(
                    current_basis,
                    jnp.asarray(global_index, dtype=jnp.int32),
                )
                candidate_block = action(z)
                hessenberg_column = jnp.zeros(
                    ((restart + 1) * block_width, block_width),
                    dtype=rhs.dtype,
                )
                for basis_index in range(local_index + 1):
                    row_start = basis_index * block_width
                    row_stop = row_start + block_width
                    previous_basis = basis_[:, row_start:row_stop]
                    coefficient = block_gram(previous_basis, candidate_block)
                    candidate_block = candidate_block - previous_basis @ coefficient
                    hessenberg_column = hessenberg_column.at[row_start:row_stop].set(
                        coefficient
                    )
                for basis_index in range(local_index + 1):
                    row_start = basis_index * block_width
                    row_stop = row_start + block_width
                    previous_basis = basis_[:, row_start:row_stop]
                    correction = block_gram(previous_basis, candidate_block)
                    candidate_block = candidate_block - previous_basis @ correction
                    hessenberg_column = hessenberg_column.at[row_start:row_stop].add(
                        correction
                    )
                next_basis, next_factor, next_active, next_rank = _rank_revealing_factor(
                    candidate_block, block_gram
                )
                next_start = (local_index + 1) * block_width
                next_stop = next_start + block_width
                basis_ = basis_.at[:, next_start:next_stop].set(next_basis)
                preconditioned_basis_ = preconditioned_basis_.at[
                    :, column_start:column_stop
                ].set(z)
                hessenberg_column = hessenberg_column.at[next_start:next_stop].set(
                    next_factor
                )
                hessenberg_ = hessenberg_.at[:, column_start:column_stop].set(
                    hessenberg_column
                )
                reduced_rows = (local_index + 2) * block_width
                reduced_columns = (local_index + 1) * block_width
                reduced = hessenberg_[:reduced_rows, :reduced_columns]
                coefficients = _pinv(reduced) @ right[:reduced_rows]
                candidate_x = (
                    cycle_base + preconditioned_basis_[:, :reduced_columns] @ coefficients
                )
                true_residual = rhs - action(candidate_x)
                next_converged = _column_norms(true_residual, block_gram) <= thresholds
                newly_converged = ~converged_ & next_converged
                iterations_ = jnp.where(
                    newly_converged,
                    jnp.asarray(global_index + 1, dtype=jnp.int32),
                    iterations_,
                )
                exhausted = next_rank == 0
                breakdown_ = breakdown_ | (exhausted & ~next_converged)
                return (
                    basis_,
                    preconditioned_basis_,
                    hessenberg_,
                    candidate_x,
                    true_residual,
                    next_converged,
                    iterations_,
                    matvec_count_ + jnp.asarray(2, dtype=jnp.int32),
                    breakdown_,
                    next_active,
                    last_executed_iteration_ + jnp.asarray(1, dtype=jnp.int32),
                )

            operand = (
                basis,
                preconditioned_basis,
                hessenberg,
                x,
                residual,
                converged,
                iterations,
                matvec_count,
                breakdown,
                active_block,
                last_executed_iteration,
            )
            should_execute = (~jnp.all(converged | breakdown)) & jnp.any(active_block)
            (
                basis,
                preconditioned_basis,
                hessenberg,
                x,
                residual,
                converged,
                iterations,
                matvec_count,
                breakdown,
                active_block,
                last_executed_iteration,
            ) = jax.lax.cond(should_execute, execute, lambda value: value, operand)
    return (
        x,
        iterations,
        matvec_count,
        initial_rank,
        breakdown,
        last_executed_iteration,
    )


def _block_cg_raw(
    action,
    precondition,
    block_gram,
    rhs: Array,
    initial: Array,
    *,
    max_steps: int,
    thresholds: Array,
):
    rhs_count = rhs.shape[1]
    initial_residual = rhs - action(initial)
    matvec_count = jnp.asarray(1, dtype=jnp.int32)
    reduced_rhs, reconstruction, active, effective_rank = _rank_revealing_factor(
        initial_residual,
        block_gram,
    )
    correction = jnp.zeros_like(reduced_rhs)
    residual = reduced_rhs
    transformed = precondition(residual, jnp.asarray(0, dtype=jnp.int32))
    direction = transformed
    gram = _hermitian_gram(block_gram, residual, transformed)
    value = initial
    true_residual = initial_residual
    converged = _column_norms(true_residual, block_gram) <= thresholds
    iterations = jnp.zeros((rhs_count,), dtype=jnp.int32)
    breakdown = jnp.zeros((rhs_count,), dtype=bool)
    last_executed_iteration = jnp.asarray(0, dtype=jnp.int32)

    for index in range(max_steps):

        def execute(operand):
            (
                correction_,
                residual_,
                direction_,
                gram_,
                value_,
                converged_,
                iterations_,
                matvec_count_,
                breakdown_,
                last_executed_iteration_,
            ) = operand
            action_direction = action(direction_)
            curvature = _hermitian_gram(
                block_gram,
                direction_,
                action_direction,
            )
            curvature_rank = _matrix_rank(curvature)
            alpha = _pinv_hermitian(curvature) @ gram_
            next_correction = correction_ + direction_ @ alpha
            next_residual = residual_ - action_direction @ alpha
            next_value = initial + next_correction @ reconstruction
            next_true_residual = next_residual @ reconstruction
            next_converged = (
                _column_norms(
                    next_true_residual,
                    block_gram,
                )
                <= thresholds
            )
            newly_converged = ~converged_ & next_converged
            iterations_ = jnp.where(
                newly_converged,
                jnp.asarray(index + 1, dtype=jnp.int32),
                iterations_,
            )
            next_transformed = precondition(
                next_residual,
                jnp.asarray(index + 1, dtype=jnp.int32),
            )
            next_gram = _hermitian_gram(
                block_gram,
                next_residual,
                next_transformed,
            )
            beta = _pinv_hermitian(gram_) @ next_gram
            next_direction = next_transformed + direction_ @ beta
            exhausted = curvature_rank == 0
            breakdown_ = breakdown_ | (exhausted & ~next_converged)
            return (
                next_correction,
                next_residual,
                next_direction,
                next_gram,
                next_value,
                next_converged,
                iterations_,
                matvec_count_ + jnp.asarray(1, dtype=jnp.int32),
                breakdown_,
                last_executed_iteration_ + jnp.asarray(1, dtype=jnp.int32),
            )

        operand = (
            correction,
            residual,
            direction,
            gram,
            value,
            converged,
            iterations,
            matvec_count,
            breakdown,
            last_executed_iteration,
        )
        should_execute = (~jnp.all(converged | breakdown)) & jnp.any(active)
        (
            correction,
            residual,
            direction,
            gram,
            value,
            converged,
            iterations,
            matvec_count,
            breakdown,
            last_executed_iteration,
        ) = jax.lax.cond(should_execute, execute, lambda selected: selected, operand)
    return (
        value,
        iterations,
        matvec_count,
        effective_rank,
        breakdown,
        last_executed_iteration,
    )


def _rank_revealing_factor(value: Array, block_gram):
    reduced_size = min(value.shape)
    if reduced_size == 0:
        return (
            jnp.zeros((value.shape[0], 0), dtype=value.dtype),
            jnp.zeros((0, value.shape[1]), dtype=value.dtype),
            jnp.zeros((0,), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
        )
    column_norms = _column_norms(value, block_gram)
    nonzero = column_norms > 0.0
    scaled = value / jnp.where(nonzero, column_norms, 1.0)[None, :]
    gram = _hermitian_gram(block_gram, scaled, scaled)
    squared_singular_values, vectors = jnp.linalg.eigh(gram)
    squared_singular_values = squared_singular_values[::-1][:reduced_size]
    vectors = vectors[:, ::-1][:, :reduced_size]
    singular_values = jnp.sqrt(jnp.maximum(squared_singular_values, 0.0))
    tolerance = (
        jnp.asarray(max(value.shape), dtype=singular_values.dtype)
        * jnp.finfo(value.real.dtype).eps
        * singular_values[0]
    )
    active = singular_values > tolerance
    inverse = jnp.where(active, 1.0 / singular_values, 0.0)
    basis = scaled @ (vectors * inverse[None, :])
    factor = (
        (singular_values * active)[:, None] * jnp.conj(vectors.T) * column_norms[None, :]
    )
    rank = jnp.sum(active, dtype=jnp.int32)
    return basis, factor, active, rank


def _block_gram(space, left: Array, right: Array) -> Array:
    def row(left_column):
        return jax.vmap(
            lambda right_column: _space_inner(space, left_column, right_column),
            in_axes=1,
        )(right)

    return jax.vmap(row, in_axes=1)(left)


def _hermitian_gram(block_gram, left: Array, right: Array) -> Array:
    gram = block_gram(left, right)
    return (gram + jnp.conj(gram.T)) / 2


def _matrix_rank(value: Array) -> Array:
    if min(value.shape) == 0:
        return jnp.asarray(0, dtype=jnp.int32)
    singular_values = jnp.linalg.svd(value, compute_uv=False)
    tolerance = (
        jnp.asarray(max(value.shape), dtype=singular_values.dtype)
        * jnp.finfo(value.real.dtype).eps
        * singular_values[0]
    )
    return jnp.sum(singular_values > tolerance, dtype=jnp.int32)


def _pinv(value: Array) -> Array:
    return jnp.linalg.pinv(
        value,
        rtol=float(jnp.finfo(value.real.dtype).eps) * float(max(value.shape)),
    )


def _pinv_hermitian(value: Array) -> Array:
    hermitian = (value + jnp.conj(value.T)) / 2
    return jnp.linalg.pinv(
        hermitian,
        rtol=float(jnp.finfo(value.real.dtype).eps) * float(max(value.shape)),
        hermitian=True,
    )


def _column_norms(value: Array, block_gram) -> Array:
    gram = block_gram(value, value)
    return jnp.sqrt(jnp.maximum(jnp.real(jnp.diag(gram)), 0.0))


__all__ = [
    "NativeBlockKrylovBackendOutput",
    "NativeBlockKrylovState",
    "prepare_native_block_krylov",
    "solve_native_block_krylov",
]
