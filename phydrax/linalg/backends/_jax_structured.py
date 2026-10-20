#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, final, TypeAlias

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

import phydrax.ein as ein

from ..._strict import StrictModule
from .._bordered import _bordered_base, BorderedBlockLinearOperator
from .._local_blocks import (
    LocalBlockFactorization,
    prepare_local_block_factorization,
    solve_local_blocks,
)
from .._operators import (
    AbstractLinearOperator,
    AdjointLinearOperator,
    DenseLinearOperator,
    DiagonalLinearOperator,
    IdentityLinearOperator,
    TransposeLinearOperator,
)
from .._plans import _certified_rank, _structured_candidate_fits, LinearSolvePlan
from .._results import LinearSolveStatus
from .._spaces import _coordinate_pairing_weights, _has_diagonal_pairing
from .._structured_operators import (
    _is_structured_exact,
    BandedLinearOperator,
    BlockDiagonalLinearOperator,
    DiagonalPlusLowRankLinearOperator,
    KroneckerLinearOperator,
    KroneckerSumLinearOperator,
    LocalBlockDiagonalLinearOperator,
    PermutationLinearOperator,
    TriangularLinearOperator,
    TridiagonalLinearOperator,
)
from .._transform_operators import TransformDiagonalLinearOperator
from .._tree import _prepare_tree, _solve_tree, TreeLinearOperator


_TridiagonalCarry: TypeAlias = tuple[Array, Array, Array, Array, Array]


class _LUState(StrictModule):
    factor: Array
    pivots: Array
    singular: Array


class _BandedLUState(StrictModule):
    factor: Array
    pivots: Array
    singular: Array
    pivot_margin: Array
    diagonal_index: int = eqx.field(static=True)


class _TridiagonalState(StrictModule):
    diagonal: Array
    upper: Array
    second_upper: Array
    cosines: Array
    sines: Array
    singular_pivots: Array


class _WoodburyState(StrictModule):
    singular_diagonal: Array
    inverse_diagonal: Array
    inverse_left: Array
    core: _LUState | None
    dense: _LUState | None


class _KroneckerSumState(StrictModule):
    eigenvalues: tuple[Array, ...]
    eigenvectors: tuple[Array, ...]
    metric_sqrts: tuple[Array, ...]
    summed_eigenvalues: Array
    singular_entries: Array
    preparation_failed: Array


class _TransformDiagonalState(StrictModule):
    inverse_spectrum: Array
    singular: Array


@final
class _BorderedState(StrictModule):
    locals: LocalBlockFactorization | _BandedLUState
    upper_solved: Array
    schur: _LUState
    failed_blocks: Array


@final
class _StructuredSymbolicState(StrictModule):
    bordered_structure: str | None = eqx.field(static=True)


class StructuredState(StrictModule):
    operator: AbstractLinearOperator
    prepared: Any


class StructuredBackendOutput(StrictModule):
    value: Array
    status: Array
    iterations: Array
    rank: Array
    condition_estimate: Array
    singular_values: Array | None


def is_structured_exact(operator: Any, /) -> bool:
    """Return whether the native backend has a structure-preserving exact solve."""
    return _is_structured_exact(operator)


def prepare_structured(problem: Any, plan: LinearSolvePlan, /) -> StructuredState:
    fits, explanation = _structured_candidate_fits(problem, plan.policy)
    if not fits:
        raise ValueError(f"Structured preparation exceeds policy: {explanation}.")
    if not is_structured_exact(problem.operator):
        raise ValueError("Operator has no native exact structured solve.")
    if problem.operator.batch_shape and not isinstance(
        problem.operator, BandedLinearOperator
    ):
        raise ValueError(
            "Only canonical banded structured solves accept operator batches."
        )
    return StructuredState(problem.operator, _prepare_operator(problem.operator))


def solve_structured(
    state: StructuredState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
) -> StructuredBackendOutput:
    expected_rank = len(state.operator.batch_shape) + 2
    if rhs.ndim != expected_rank:
        raise ValueError(
            "Structured canonical right-hand sides must have batch_shape + (n, k)."
        )
    value, failed = _solve_operator(state.operator, state.prepared, rhs)
    count = rhs.shape[-1]
    status_shape = state.operator.batch_shape + (count,)
    status = jnp.full(status_shape, int(LinearSolveStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(failed, int(LinearSolveStatus.SINGULAR), status)
    if isinstance(state.prepared, _BorderedState):
        rank_array = _bordered_rank(state.operator, state.prepared)
    else:
        rank = _certified_rank(state.operator)
        rank_value = -1 if rank is None else rank
        rank_array = jnp.full(
            state.operator.batch_shape,
            rank_value,
            dtype=jnp.int32,
        )
    return StructuredBackendOutput(
        value=value,
        status=status,
        iterations=jnp.zeros(status_shape, dtype=jnp.int32),
        rank=rank_array,
        condition_estimate=jnp.full(
            state.operator.batch_shape, jnp.nan, dtype=rhs.real.dtype
        ),
        singular_values=None,
    )


def _prepare_lu(matrix: Array, /) -> _LUState:
    factor, pivots = jsp.linalg.lu_factor(matrix)
    diagonal = jnp.diagonal(factor, axis1=-2, axis2=-1)
    scale = jnp.maximum(jnp.max(jnp.abs(matrix), axis=(-2, -1)), 1.0)
    threshold = jnp.finfo(matrix.real.dtype).eps * float(matrix.shape[-1]) * scale
    singular = (
        jnp.any(jnp.abs(diagonal) <= threshold[..., None], axis=-1)
        | jnp.any(~jnp.isfinite(matrix), axis=(-2, -1))
        | jnp.any(~jnp.isfinite(factor), axis=(-2, -1))
    )
    return _LUState(factor, pivots, singular)


def _solve_lu(
    state: _LUState, rhs: Array, /, *, transpose: bool = False
) -> tuple[Array, Array]:
    value = jsp.linalg.lu_solve(
        (state.factor, state.pivots), rhs, trans=1 if transpose else 0
    )
    failed = state.singular[..., None] | jnp.any(~jnp.isfinite(value), axis=-2)
    return value, failed


def _prepare_banded_lu(operator: BandedLinearOperator, /) -> _BandedLUState:
    lower = operator.lower_bandwidth
    upper = operator.upper_bandwidth
    size = operator.source.size
    diagonal = upper + lower
    factor_width = 2 * lower + upper + 1
    flattened_bands = operator.bands.reshape((-1, lower + upper + 1, size))

    def factor_one(bands: Array) -> tuple[Array, Array, Array, Array]:
        factor = jnp.zeros((factor_width, size), dtype=bands.dtype)
        factor = factor.at[lower : lower + lower + upper + 1].set(bands)
        scale = jnp.maximum(jnp.max(jnp.abs(bands)), 1.0)
        threshold = jnp.finfo(bands.real.dtype).eps * float(size) * scale
        singular = jnp.any(~jnp.isfinite(bands))
        minimum_margin = jnp.asarray(jnp.inf, dtype=scale.dtype)
        pivots = jnp.arange(size, dtype=jnp.int32)

        def column_step(
            column: int, carry: tuple[Array, Array, Array, Array]
        ) -> tuple[Array, Array, Array, Array]:
            factor, pivots, singular, minimum_margin = carry
            offsets = jnp.arange(lower + 1, dtype=jnp.int32)
            candidates = jnp.abs(factor[diagonal : diagonal + lower + 1, column])
            candidates = jnp.where(offsets < size - column, candidates, -jnp.inf)
            pivot_row = column + jnp.argmax(candidates).astype(jnp.int32)
            pivots = pivots.at[column].set(pivot_row)
            # Historical multipliers stay in their elimination columns. Compact
            # band LU stores the pivot sequence, not a globally permuted dense L.
            for target_offset in range(upper + lower + 1):
                target_column = column + target_offset

                def swap(factor: Array, *, target_column: int = target_column) -> Array:
                    first_index = diagonal + column - target_column
                    second_index = diagonal + pivot_row - target_column
                    first_value = factor[first_index, target_column]
                    second_value = factor[second_index, target_column]
                    factor = factor.at[first_index, target_column].set(second_value)
                    return factor.at[second_index, target_column].set(first_value)

                factor = jax.lax.cond(target_column < size, swap, lambda f: f, factor)
            pivot = factor[diagonal, column]
            minimum_margin = jnp.minimum(minimum_margin, jnp.abs(pivot) / scale)
            singular = singular | (jnp.abs(pivot) <= threshold) | ~jnp.isfinite(pivot)
            safe_pivot = jnp.where(jnp.abs(pivot) <= threshold, 1.0, pivot)
            for row_offset in range(1, lower + 1):

                def eliminate(factor: Array, *, row_offset: int = row_offset) -> Array:
                    lower_index = diagonal + row_offset
                    multiplier = factor[lower_index, column] / safe_pivot
                    factor = factor.at[lower_index, column].set(multiplier)
                    for column_offset in range(1, upper + lower + 1):
                        target = column + column_offset
                        band_index = diagonal + row_offset - column_offset

                        def update(
                            factor: Array,
                            *,
                            target: int = target,
                            band_index: int = band_index,
                            column_offset: int = column_offset,
                        ) -> Array:
                            return factor.at[band_index, target].add(
                                -multiplier * factor[diagonal - column_offset, target]
                            )

                        factor = jax.lax.cond(target < size, update, lambda f: f, factor)
                    return factor

                factor = jax.lax.cond(
                    column + row_offset < size, eliminate, lambda f: f, factor
                )
            return factor, pivots, singular, minimum_margin

        factor, pivots, singular, minimum_margin = jax.lax.fori_loop(
            0, size, column_step, (factor, pivots, singular, minimum_margin)
        )
        singular = singular | jnp.any(~jnp.isfinite(factor))
        return factor, pivots, singular, minimum_margin

    factors, pivots, singular, margins = jax.vmap(factor_one)(flattened_bands)
    return _BandedLUState(
        factor=factors.reshape(operator.batch_shape + (factor_width, size)),
        pivots=pivots.reshape(operator.batch_shape + (size,)),
        singular=singular.reshape(operator.batch_shape),
        pivot_margin=margins.reshape(operator.batch_shape),
        diagonal_index=diagonal,
    )


def _solve_banded_lu(
    operator: BandedLinearOperator,
    state: _BandedLUState,
    rhs: Array,
    /,
    *,
    transpose: bool = False,
) -> tuple[Array, Array]:
    lower = operator.lower_bandwidth
    upper_factor = operator.upper_bandwidth + lower
    size = operator.source.size
    diagonal = state.diagonal_index
    factors = state.factor.reshape((-1, state.factor.shape[-2], size))
    pivots = state.pivots.reshape((-1, size))
    flattened_rhs = rhs.reshape((-1, size, rhs.shape[-1]))

    def solve_one(factor: Array, pivot_rows: Array, value: Array) -> Array:
        if transpose:

            def upper_transpose(row: int, value: Array) -> Array:
                for offset in range(1, upper_factor + 1):

                    def update(value: Array, *, offset: int = offset) -> Array:
                        return value.at[row].add(
                            -factor[diagonal - offset, row] * value[row - offset]
                        )

                    value = jax.lax.cond(offset <= row, update, lambda v: v, value)
                return value.at[row].set(value[row] / factor[diagonal, row])

            value = jax.lax.fori_loop(0, size, upper_transpose, value)

            def lower_transpose(index: int, value: Array) -> Array:
                row = size - 1 - index
                for offset in range(1, lower + 1):

                    def update(value: Array, *, offset: int = offset) -> Array:
                        return value.at[row].add(
                            -factor[diagonal + offset, row] * value[row + offset]
                        )

                    value = jax.lax.cond(offset < size - row, update, lambda v: v, value)
                pivot_row = pivot_rows[row]
                first = value[row]
                second = value[pivot_row]
                value = value.at[row].set(second)
                return value.at[pivot_row].set(first)

            return jax.lax.fori_loop(0, size, lower_transpose, value)

        def lower_forward(row: int, value: Array) -> Array:
            pivot_row = pivot_rows[row]
            first = value[row]
            second = value[pivot_row]
            value = value.at[row].set(second)
            value = value.at[pivot_row].set(first)
            for offset in range(1, lower + 1):

                def update(value: Array, *, offset: int = offset) -> Array:
                    return value.at[row + offset].add(
                        -factor[diagonal + offset, row] * value[row]
                    )

                value = jax.lax.cond(offset < size - row, update, lambda v: v, value)
            return value

        value = jax.lax.fori_loop(0, size, lower_forward, value)

        def upper_backward(index: int, value: Array) -> Array:
            row = size - 1 - index
            for offset in range(1, upper_factor + 1):

                def update(value: Array, *, offset: int = offset) -> Array:
                    return value.at[row].add(
                        -factor[diagonal - offset, row + offset] * value[row + offset]
                    )

                value = jax.lax.cond(offset < size - row, update, lambda v: v, value)
            pivot = factor[diagonal, row]
            return value.at[row].set(value[row] / jnp.where(pivot == 0, 1.0, pivot))

        return jax.lax.fori_loop(0, size, upper_backward, value)

    value = jax.vmap(solve_one)(factors, pivots, flattened_rhs).reshape(rhs.shape)
    failed = state.singular[..., None] | jnp.any(~jnp.isfinite(value), axis=-2)
    return value, failed


def _prepare_woodbury(operator: DiagonalPlusLowRankLinearOperator, /) -> _WoodburyState:
    singular_diagonal = (
        jnp.asarray(False)
        if operator.nonsingular_diagonal
        else jnp.any(operator.diagonal == 0)
    )
    safe_diagonal = (
        operator.diagonal
        if operator.nonsingular_diagonal
        else jnp.where(operator.diagonal == 0, 1, operator.diagonal)
    )
    inverse_diagonal = jnp.reciprocal(safe_diagonal)
    inverse_left = inverse_diagonal[:, None] * operator.left_factor
    rank = operator.left_factor.shape[1]
    core = (
        None
        if rank == 0
        else _prepare_lu(
            jnp.eye(rank, dtype=operator.left_factor.dtype)
            + jnp.conj(operator.right_factor.T) @ inverse_left
        )
    )
    dense = None
    if not operator.nonsingular_diagonal and (
        isinstance(singular_diagonal, jax_core.Tracer) or bool(singular_diagonal)
    ):
        dense = _prepare_lu(operator._materialize())
    return _WoodburyState(
        singular_diagonal,
        inverse_diagonal,
        inverse_left,
        core,
        dense,
    )


def _prepare_kronecker_sum(
    operator: KroneckerSumLinearOperator,
    /,
) -> _KroneckerSumState:
    eigenvalues = []
    eigenvectors = []
    metric_sqrts = []
    failed = jnp.asarray(False)
    for factor in operator.factors:
        matrix = factor._materialize()
        weights = _coordinate_pairing_weights(factor.source)
        real_weights = jnp.real(weights)
        valid_weights = jnp.all(
            jnp.isfinite(real_weights) & (real_weights > 0.0) & (jnp.imag(weights) == 0.0)
        )
        metric_sqrt = jnp.sqrt(real_weights)
        transformed = metric_sqrt[:, None] * matrix / metric_sqrt[None, :]
        scale = jnp.maximum(jnp.max(jnp.abs(transformed)), 1.0)
        tolerance = (
            jnp.finfo(transformed.real.dtype).eps * float(factor.source.size) * scale
        )
        symmetry_error = jnp.max(jnp.abs(transformed - jnp.conj(transformed.T)))
        values, vectors = jnp.linalg.eigh(transformed)
        failed = (
            failed
            | ~valid_weights
            | (symmetry_error > tolerance)
            | jnp.any(~jnp.isfinite(matrix))
            | jnp.any(~jnp.isfinite(values))
            | jnp.any(~jnp.isfinite(vectors))
        )
        eigenvalues.append(values)
        eigenvectors.append(vectors)
        metric_sqrts.append(metric_sqrt)

    sizes = tuple(factor.source.size for factor in operator.factors)
    summed = jnp.zeros(sizes, dtype=eigenvalues[0].dtype)
    for axis, values in enumerate(eigenvalues):
        shape = [1] * len(sizes)
        shape[axis] = sizes[axis]
        summed = summed + values.reshape(tuple(shape))
    summed_scale = jnp.maximum(jnp.max(jnp.abs(summed)), 1.0)
    threshold = jnp.finfo(summed.dtype).eps * float(operator.source.size) * summed_scale
    singular_entries = jnp.abs(summed) <= threshold
    return _KroneckerSumState(
        tuple(eigenvalues),
        tuple(eigenvectors),
        tuple(metric_sqrts),
        summed,
        singular_entries,
        failed,
    )


def _apply_axis_matrix(value: Array, matrix: Array, axis: int, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    shape = moved.shape
    applied = matrix @ moved.reshape((shape[0], -1))
    return jnp.moveaxis(applied.reshape((matrix.shape[0],) + shape[1:]), 0, axis)


def _solve_kronecker_sum(
    operator: KroneckerSumLinearOperator,
    state: _KroneckerSumState,
    rhs: Array,
    /,
) -> tuple[Array, Array]:
    sizes = tuple(factor.source.size for factor in operator.factors)
    value = rhs.reshape(sizes + (rhs.shape[1],))
    for axis, metric_sqrt in enumerate(state.metric_sqrts):
        shape = [1] * value.ndim
        shape[axis] = sizes[axis]
        value = value * metric_sqrt.reshape(tuple(shape))
    for axis, vectors in enumerate(state.eigenvectors):
        value = _apply_axis_matrix(value, jnp.conj(vectors.T), axis)
    safe_eigenvalues = jnp.where(
        state.singular_entries,
        jnp.ones((), dtype=state.summed_eigenvalues.dtype),
        state.summed_eigenvalues,
    )
    value = value / safe_eigenvalues[..., None]
    for axis, vectors in enumerate(state.eigenvectors):
        value = _apply_axis_matrix(value, vectors, axis)
    for axis, metric_sqrt in enumerate(state.metric_sqrts):
        shape = [1] * value.ndim
        shape[axis] = sizes[axis]
        value = value / metric_sqrt.reshape(tuple(shape))
    value = value.reshape((operator.source.size, rhs.shape[1]))
    failed = (
        state.preparation_failed
        | jnp.any(state.singular_entries)
        | jnp.any(~jnp.isfinite(value))
    )
    value = jnp.where(
        state.preparation_failed,
        jnp.zeros((), dtype=value.dtype),
        value,
    )
    return value, failed


def _solve_bordered_locals(
    operator: BorderedBlockLinearOperator,
    factors: LocalBlockFactorization | _BandedLUState,
    rhs: Array,
    /,
    *,
    transpose: bool,
) -> tuple[Array, Array]:
    if isinstance(factors, LocalBlockFactorization):
        value = jsp.linalg.lu_solve(
            (factors.factors, factors.pivots),
            rhs,
            trans=1 if transpose else 0,
        )
        failed = factors.failed_blocks[:, None] | jnp.any(~jnp.isfinite(value), axis=-2)
        return value, failed
    if not isinstance(operator.local_blocks, BandedLinearOperator):
        raise ValueError("Banded bordered factors require their local operator.")
    return _solve_banded_lu(operator.local_blocks, factors, rhs, transpose=transpose)


def _prepare_bordered(operator: BorderedBlockLinearOperator, /) -> _BorderedState:
    if isinstance(operator.local_blocks, BandedLinearOperator):
        locals_: LocalBlockFactorization | _BandedLUState = _prepare_banded_lu(
            operator.local_blocks
        )
        failed_blocks = locals_.singular
    else:
        locals_ = prepare_local_block_factorization(operator.local_blocks)
        failed_blocks = locals_.failed_blocks
    # Condense the shared border, not a duplicated border per local matrix.
    # The existing local-elimination plan owns a different retained-DOF layout.
    upper_solved, upper_failed = _solve_bordered_locals(
        operator, locals_, operator.upper, transpose=False
    )
    failed_blocks = failed_blocks | jnp.any(upper_failed, axis=-1)
    schur = operator.border - ein.contract("lbi,lic->bc", operator.lower, upper_solved)
    return _BorderedState(locals_, upper_solved, _prepare_lu(schur), failed_blocks)


def _solve_bordered(
    operator: BorderedBlockLinearOperator,
    state: _BorderedState,
    rhs: Array,
    /,
    *,
    transpose: bool,
) -> tuple[Array, Array]:
    rhs = operator._pack_columns(rhs)
    split = operator.num_blocks * operator.local_size
    local_rhs = rhs[:split].reshape(
        (operator.num_blocks, operator.local_size, rhs.shape[-1])
    )
    local_load, local_failed = _solve_bordered_locals(
        operator, state.locals, local_rhs, transpose=transpose
    )
    border_rhs = rhs[split:]
    if transpose:
        border_rhs = border_rhs - ein.contract("lib,lik->bk", operator.upper, local_load)
    else:
        border_rhs = border_rhs - ein.contract("lbi,lik->bk", operator.lower, local_load)
    border_value, border_failed = _solve_lu(state.schur, border_rhs, transpose=transpose)
    if transpose:
        local_value, reconstruction_failed = _solve_bordered_locals(
            operator,
            state.locals,
            local_rhs - jnp.swapaxes(operator.lower, -1, -2) @ border_value,
            transpose=True,
        )
        local_failed = local_failed | reconstruction_failed
    else:
        local_value = local_load - state.upper_solved @ border_value
    value = jnp.concatenate(
        (local_value.reshape((split, rhs.shape[-1])), border_value), axis=0
    )
    value = operator._unpack_columns(value)
    failed = (
        jnp.any(state.failed_blocks)
        | jnp.any(local_failed, axis=0)
        | border_failed
        | jnp.any(~jnp.isfinite(value), axis=0)
    )
    return value, failed


def _solve_bordered_adjoint(
    operator: BorderedBlockLinearOperator,
    state: _BorderedState,
    rhs: Array,
    /,
    *,
    transpose: bool = True,
) -> tuple[Array, Array]:
    weighted = jax.vmap(operator.source.riesz, in_axes=1, out_axes=1)(rhs)
    value, failed = _solve_bordered(
        operator, state, jnp.conj(weighted), transpose=transpose
    )
    value = jax.vmap(operator.target.inverse_riesz, in_axes=1, out_axes=1)(
        jnp.conj(value)
    )
    return value, failed | jnp.any(~jnp.isfinite(value), axis=0)


def _bordered_rank(operator: AbstractLinearOperator, state: _BorderedState, /) -> Array:
    # A failed local pivot is not proof of global rank deficiency: another
    # elimination ordering might succeed. Only successful factors certify rank.
    failed = jnp.any(state.failed_blocks) | state.schur.singular
    return jnp.where(failed, -1, operator.source.size).astype(jnp.int32)


def _solve_banded_transformed(
    state: StructuredState, rhs: Array, /, *, adjoint: bool
) -> StructuredBackendOutput:
    operator = state.operator
    factors = state.prepared
    if not isinstance(operator, BandedLinearOperator) or not isinstance(
        factors, _BandedLUState
    ):
        raise TypeError("Prepared banded transformed solves require banded LU factors.")
    if operator.batch_shape or rhs.ndim != 2 or rhs.shape[0] != operator.source.size:
        raise ValueError("Transformed banded right-hand sides must have shape (n, k).")
    if adjoint:
        weighted = jax.vmap(operator.source.riesz, in_axes=1, out_axes=1)(rhs)
        value, failed = _solve_banded_lu(
            operator, factors, jnp.conj(weighted), transpose=True
        )
        value = jax.vmap(operator.target.inverse_riesz, in_axes=1, out_axes=1)(
            jnp.conj(value)
        )
    else:
        value, failed = _solve_banded_lu(operator, factors, rhs, transpose=True)
    failed = failed | jnp.any(~jnp.isfinite(value), axis=0)
    return StructuredBackendOutput(
        value=value,
        status=jnp.where(
            failed, int(LinearSolveStatus.SINGULAR), int(LinearSolveStatus.SUCCESS)
        ).astype(jnp.int32),
        iterations=jnp.zeros((rhs.shape[-1],), dtype=jnp.int32),
        rank=jnp.where(factors.singular, -1, operator.source.size).astype(jnp.int32),
        condition_estimate=jnp.asarray(jnp.nan, dtype=rhs.real.dtype),
        singular_values=None,
    )


def solve_structured_transformed(
    state: StructuredState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
    *,
    adjoint: bool,
) -> StructuredBackendOutput:
    del plan
    if isinstance(state.operator, BandedLinearOperator):
        return _solve_banded_transformed(state, rhs, adjoint=adjoint)
    operator = _bordered_base(state.operator)
    if operator is None or not isinstance(state.prepared, _BorderedState):
        raise ValueError(
            "Prepared transformed structured reuse requires bordered factors."
        )
    if rhs.ndim != 2 or rhs.shape[0] != state.operator.source.size:
        raise ValueError("Transformed bordered right-hand sides must have shape (n, k).")
    if isinstance(state.operator, TransposeLinearOperator):
        value, failed = (
            _solve_bordered_adjoint(operator, state.prepared, rhs, transpose=False)
            if adjoint
            else _solve_bordered(operator, state.prepared, rhs, transpose=False)
        )
    elif isinstance(state.operator, AdjointLinearOperator):
        if adjoint:
            value, failed = _solve_bordered(
                operator, state.prepared, rhs, transpose=False
            )
        else:
            # For an endomorphism, (G^-1 A^H G)^-T =
            # G^T conjugate(A^-1) G^-T; Riesz maps own both metric actions.
            weighted = jax.vmap(operator.source.inverse_riesz, in_axes=1, out_axes=1)(
                jnp.conj(rhs)
            )
            value, failed = _solve_bordered(
                operator, state.prepared, weighted, transpose=False
            )
            value = jnp.conj(
                jax.vmap(operator.target.riesz, in_axes=1, out_axes=1)(value)
            )
    else:
        value, failed = (
            _solve_bordered_adjoint(operator, state.prepared, rhs)
            if adjoint
            else _solve_bordered(operator, state.prepared, rhs, transpose=True)
        )
    failed = failed | jnp.any(~jnp.isfinite(value), axis=0)
    return StructuredBackendOutput(
        value=value,
        status=jnp.where(
            failed, int(LinearSolveStatus.SINGULAR), int(LinearSolveStatus.SUCCESS)
        ).astype(jnp.int32),
        iterations=jnp.zeros((rhs.shape[-1],), dtype=jnp.int32),
        rank=_bordered_rank(state.operator, state.prepared),
        condition_estimate=jnp.asarray(jnp.nan, dtype=rhs.real.dtype),
        singular_values=None,
    )


def _prepare_operator(operator: Any, /) -> Any:
    if isinstance(operator, BorderedBlockLinearOperator):
        return _prepare_bordered(operator)
    if isinstance(
        operator, (TransposeLinearOperator, AdjointLinearOperator)
    ) and isinstance(operator.operator, BorderedBlockLinearOperator):
        return _prepare_bordered(operator.operator)
    if isinstance(operator, TreeLinearOperator):
        return _prepare_tree(operator)
    if isinstance(operator, DenseLinearOperator):
        return _prepare_lu(operator.matrix)
    if isinstance(operator, BandedLinearOperator):
        return _prepare_banded_lu(operator)
    if isinstance(operator, TridiagonalLinearOperator):
        return _prepare_tridiagonal(operator)
    if isinstance(operator, DiagonalPlusLowRankLinearOperator):
        return _prepare_woodbury(operator)
    if isinstance(operator, BlockDiagonalLinearOperator):
        return tuple(_prepare_operator(block) for block in operator.blocks)
    if isinstance(operator, KroneckerLinearOperator):
        return tuple(_prepare_operator(factor) for factor in operator.factors)
    if isinstance(operator, KroneckerSumLinearOperator):
        return _prepare_kronecker_sum(operator)
    if isinstance(operator, TransformDiagonalLinearOperator):
        singular = (
            jnp.asarray(False)
            if operator.nonsingular
            else jnp.any(operator.spectrum == 0)
        )
        safe = (
            operator.spectrum
            if operator.nonsingular
            else jnp.where(operator.spectrum == 0, 1, operator.spectrum)
        )
        return _TransformDiagonalState(jnp.reciprocal(safe), singular)
    if isinstance(operator, LocalBlockDiagonalLinearOperator):
        positive = operator.properties.certifies(
            "positive_definite"
        ) and _has_diagonal_pairing(operator.source)
        weights = (
            _coordinate_pairing_weights(operator.source).reshape(
                (operator.num_blocks, operator.input_block_size)
            )
            if positive
            else None
        )
        return prepare_local_block_factorization(
            operator.blocks,
            positive_definite=positive,
            metric_weights=weights,
        )
    return None


def _solve_operator(
    operator: Any,
    prepared: Any,
    rhs: Array,
    /,
) -> tuple[Array, Array]:
    if isinstance(operator, BorderedBlockLinearOperator):
        return _solve_bordered(operator, prepared, rhs, transpose=False)
    if isinstance(operator, TransposeLinearOperator) and isinstance(
        operator.operator, BorderedBlockLinearOperator
    ):
        return _solve_bordered(operator.operator, prepared, rhs, transpose=True)
    if isinstance(operator, AdjointLinearOperator) and isinstance(
        operator.operator, BorderedBlockLinearOperator
    ):
        return _solve_bordered_adjoint(operator.operator, prepared, rhs)
    if isinstance(operator, TreeLinearOperator):
        return _solve_tree(operator, prepared, rhs)
    if isinstance(operator, IdentityLinearOperator):
        return rhs, jnp.asarray(False)
    if isinstance(operator, DenseLinearOperator):
        return _solve_lu(prepared, rhs)
    if isinstance(operator, DiagonalLinearOperator):
        singular = jnp.any(operator.diagonal == 0)
        safe = jnp.where(operator.diagonal == 0, 1, operator.diagonal)
        return rhs / safe[:, None], singular
    if isinstance(operator, PermutationLinearOperator):
        return rhs[operator.inverse_permutation], jnp.asarray(False)
    if isinstance(operator, TriangularLinearOperator):
        diagonal = jnp.diag(operator.matrix)
        singular = (
            jnp.any(diagonal == 0) if not operator.unit_diagonal else jnp.asarray(False)
        )
        value = jsp.linalg.solve_triangular(
            operator.matrix,
            rhs,
            lower=operator.lower,
            unit_diagonal=operator.unit_diagonal,
        )
        return value, singular | jnp.any(~jnp.isfinite(value))
    if isinstance(operator, TridiagonalLinearOperator):
        return _solve_tridiagonal(prepared, rhs)
    if isinstance(operator, BandedLinearOperator):
        return _solve_banded_lu(operator, prepared, rhs)
    if isinstance(operator, LocalBlockDiagonalLinearOperator):
        grouped_rhs = rhs.reshape(
            (operator.num_blocks, operator.output_block_size, rhs.shape[1])
        )
        grouped_value, failed = solve_local_blocks(prepared, grouped_rhs)
        return grouped_value.reshape((operator.source.size, rhs.shape[1])), failed
    if isinstance(operator, BlockDiagonalLinearOperator):
        values = []
        failed = jnp.asarray(False)
        target_offset = 0
        for block, block_state in zip(operator.blocks, prepared, strict=True):
            block_rhs = rhs[target_offset : target_offset + block.target.size]
            block_value, block_failed = _solve_operator(
                block,
                block_state,
                block_rhs,
            )
            values.append(block_value)
            failed = failed | block_failed
            target_offset += block.target.size
        return jnp.concatenate(tuple(values), axis=0), failed
    if isinstance(operator, DiagonalPlusLowRankLinearOperator):
        state: _WoodburyState = prepared

        def woodbury_solve(_: None) -> tuple[Array, Array]:
            inverse_rhs = state.inverse_diagonal[:, None] * rhs
            if state.core is None:
                value = inverse_rhs
                failed = jnp.asarray(False)
            else:
                correction, failed = _solve_lu(
                    state.core,
                    jnp.conj(operator.right_factor.T) @ inverse_rhs,
                )
                value = inverse_rhs - state.inverse_left @ correction
            return value, failed | jnp.any(~jnp.isfinite(value))

        dense = state.dense
        if dense is None:
            return woodbury_solve(None)
        return jax.lax.cond(
            state.singular_diagonal,
            lambda _: _solve_lu(dense, rhs),
            woodbury_solve,
            operand=None,
        )
    if isinstance(operator, KroneckerSumLinearOperator):
        return _solve_kronecker_sum(operator, prepared, rhs)
    if isinstance(operator, TransformDiagonalLinearOperator):
        value = operator._solve_flat_columns(rhs, prepared.inverse_spectrum)
        return value, prepared.singular | jnp.any(~jnp.isfinite(value))
    if isinstance(operator, KroneckerLinearOperator):
        value = rhs.reshape(
            tuple(factor.target.size for factor in operator.factors) + (rhs.shape[1],)
        )
        failed = jnp.zeros((rhs.shape[1],), dtype=jnp.bool_)
        for axis, (factor, factor_state) in enumerate(
            zip(operator.factors, prepared, strict=True)
        ):
            moved = jnp.moveaxis(value, axis, 0)
            moved_shape = moved.shape
            solved, factor_failed = _solve_operator(
                factor,
                factor_state,
                moved.reshape((factor.target.size, -1)),
            )
            factor_failed = factor_failed.reshape(moved_shape[1:])
            if factor_failed.ndim > 1:
                factor_failed = jnp.any(
                    factor_failed,
                    axis=tuple(range(factor_failed.ndim - 1)),
                )
            failed = failed | factor_failed
            value = jnp.moveaxis(
                solved.reshape((factor.source.size,) + moved_shape[1:]),
                0,
                axis,
            )
        return value.reshape((operator.source.size, rhs.shape[1])), failed
    raise TypeError(f"Unsupported structured operator {type(operator).__name__}.")


def _prepare_tridiagonal(
    operator: TridiagonalLinearOperator,
    /,
) -> _TridiagonalState:
    n = operator.source.size
    diagonal = operator.diagonal
    upper = operator.upper
    second_upper = jnp.zeros((max(n - 2, 0),), dtype=diagonal.dtype)
    cosines = jnp.zeros((max(n - 1, 0),), dtype=diagonal.real.dtype)
    sines = jnp.zeros((max(n - 1, 0),), dtype=diagonal.dtype)

    def eliminate(index: Array, state: _TridiagonalCarry) -> _TridiagonalCarry:
        diagonal_state, upper_state, second_state, cosine_state, sine_state = state
        pivot = diagonal_state[index]
        subdiagonal = operator.lower[index]
        magnitude = jnp.hypot(jnp.abs(pivot), jnp.abs(subdiagonal))
        pivot_abs = jnp.abs(pivot)
        phase = jnp.where(pivot_abs > 0, pivot / pivot_abs, 1)
        cosine = jnp.where(magnitude > 0, pivot_abs / magnitude, 1)
        sine = jnp.where(
            magnitude > 0,
            phase * jnp.conj(subdiagonal) / magnitude,
            0,
        )

        current_upper = upper_state[index]
        next_diagonal = diagonal_state[index + 1]
        diagonal_state = diagonal_state.at[index].set(phase * magnitude)
        upper_state = upper_state.at[index].set(
            cosine * current_upper + sine * next_diagonal
        )
        diagonal_state = diagonal_state.at[index + 1].set(
            -jnp.conj(sine) * current_upper + cosine * next_diagonal
        )

        def update_fill(values: tuple[Array, Array]) -> tuple[Array, Array]:
            upper_value, second_value = values
            next_upper = upper_value[index + 1]
            second_value = second_value.at[index].set(sine * next_upper)
            upper_value = upper_value.at[index + 1].set(cosine * next_upper)
            return upper_value, second_value

        if n > 2:
            upper_state, second_state = jax.lax.cond(
                index < n - 2,
                update_fill,
                lambda values: values,
                (upper_state, second_state),
            )
        cosine_state = cosine_state.at[index].set(cosine)
        sine_state = sine_state.at[index].set(sine)
        return (
            diagonal_state,
            upper_state,
            second_state,
            cosine_state,
            sine_state,
        )

    if n > 1:
        diagonal, upper, second_upper, cosines, sines = jax.lax.fori_loop(
            0,
            n - 1,
            eliminate,
            (diagonal, upper, second_upper, cosines, sines),
        )
    zero = jnp.zeros((), dtype=diagonal.real.dtype)
    scale = jnp.maximum(
        jnp.max(jnp.abs(operator.diagonal)),
        jnp.maximum(
            jnp.max(jnp.abs(operator.upper), initial=zero),
            jnp.max(jnp.abs(operator.lower), initial=zero),
        ),
    )
    threshold = jnp.finfo(diagonal.real.dtype).eps * jnp.maximum(scale, 1)
    singular_pivots = jnp.abs(diagonal) <= threshold
    return _TridiagonalState(
        diagonal,
        upper,
        second_upper,
        cosines,
        sines,
        singular_pivots,
    )


def _solve_tridiagonal(
    state: _TridiagonalState,
    rhs: Array,
    /,
) -> tuple[Array, Array]:
    n = state.diagonal.size

    def rotate_rhs(index: Array, rhs_state: Array) -> Array:
        current_rhs = rhs_state[index]
        next_rhs = rhs_state[index + 1]
        cosine = state.cosines[index]
        sine = state.sines[index]
        rhs_state = rhs_state.at[index].set(cosine * current_rhs + sine * next_rhs)
        return rhs_state.at[index + 1].set(
            -jnp.conj(sine) * current_rhs + cosine * next_rhs
        )

    transformed_rhs = jax.lax.fori_loop(0, n - 1, rotate_rhs, rhs) if n > 1 else rhs
    safe_diagonal = jnp.where(state.singular_pivots, 1, state.diagonal)
    value = jnp.zeros_like(rhs)
    value = value.at[-1].set(transformed_rhs[-1] / safe_diagonal[-1])
    if n > 1:
        value = value.at[-2].set(
            (transformed_rhs[-2] - state.upper[-1] * value[-1]) / safe_diagonal[-2]
        )

    def substitute(reverse_index: Array, result: Array) -> Array:
        index = n - 3 - reverse_index
        numerator = (
            transformed_rhs[index]
            - state.upper[index] * result[index + 1]
            - state.second_upper[index] * result[index + 2]
        )
        return result.at[index].set(numerator / safe_diagonal[index])

    if n > 2:
        value = jax.lax.fori_loop(0, n - 2, substitute, value)
    failed = jnp.any(state.singular_pivots) | jnp.any(~jnp.isfinite(value))
    return value, failed


__all__ = [
    "StructuredBackendOutput",
    "StructuredState",
    "is_structured_exact",
    "prepare_structured",
    "solve_structured",
]
