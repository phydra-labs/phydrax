#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array

from ..._strict import StrictModule
from .._local_blocks import (
    prepare_local_block_factorization,
    solve_local_blocks,
)
from .._operators import (
    DenseLinearOperator,
    DiagonalLinearOperator,
    IdentityLinearOperator,
)
from .._plans import _certified_rank, LinearSolvePlan
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


class StructuredState(StrictModule):
    operator: Any
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
    del plan
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
    diagonal = jnp.diag(factor)
    scale = jnp.maximum(jnp.max(jnp.abs(matrix)), 1.0)
    threshold = jnp.finfo(matrix.real.dtype).eps * float(matrix.shape[0]) * scale
    singular = (
        jnp.any(jnp.abs(diagonal) <= threshold)
        | jnp.any(~jnp.isfinite(matrix))
        | jnp.any(~jnp.isfinite(factor))
    )
    return _LUState(factor, pivots, singular)


def _solve_lu(state: _LUState, rhs: Array, /) -> tuple[Array, Array]:
    value = jsp.linalg.lu_solve((state.factor, state.pivots), rhs)
    return value, state.singular | jnp.any(~jnp.isfinite(value))


def _prepare_banded_lu(operator: BandedLinearOperator, /) -> _BandedLUState:
    lower = operator.lower_bandwidth
    upper = operator.upper_bandwidth
    size = operator.source.size
    diagonal = upper + lower
    factor_width = 2 * lower + upper + 1
    flattened_bands = operator.bands.reshape((-1, lower + upper + 1, size))

    def factor_one(bands):
        factor = jnp.zeros((factor_width, size), dtype=bands.dtype)
        factor = factor.at[lower : lower + lower + upper + 1].set(bands)
        scale = jnp.maximum(jnp.max(jnp.abs(bands)), 1.0)
        threshold = jnp.finfo(bands.real.dtype).eps * float(size) * scale
        singular = jnp.any(~jnp.isfinite(bands))
        minimum_margin = jnp.asarray(jnp.inf, dtype=scale.dtype)
        pivots = jnp.arange(size, dtype=jnp.int32)
        for column in range(size):
            candidate_count = min(lower + 1, size - column)
            candidates = jnp.abs(
                factor[
                    diagonal : diagonal + candidate_count,
                    column,
                ]
            )
            pivot_row = column + jnp.argmax(candidates).astype(jnp.int32)
            pivots = pivots.at[column].set(pivot_row)
            for target_column in range(size):
                first_index = diagonal + column - target_column
                second_index = diagonal + pivot_row - target_column
                first_valid = (first_index >= 0) & (first_index < factor_width)
                second_valid = (second_index >= 0) & (second_index < factor_width)
                first_clipped = jnp.clip(first_index, 0, factor_width - 1)
                second_clipped = jnp.clip(second_index, 0, factor_width - 1)
                first_value = jnp.where(
                    first_valid, factor[first_clipped, target_column], 0
                )
                second_value = jnp.where(
                    second_valid, factor[second_clipped, target_column], 0
                )
                factor = jax.lax.cond(
                    first_valid,
                    lambda current: current.at[first_clipped, target_column].set(
                        second_value
                    ),
                    lambda current: current,
                    factor,
                )
                factor = jax.lax.cond(
                    second_valid,
                    lambda current: current.at[second_clipped, target_column].set(
                        first_value
                    ),
                    lambda current: current,
                    factor,
                )
            pivot = factor[diagonal, column]
            minimum_margin = jnp.minimum(minimum_margin, jnp.abs(pivot) / scale)
            singular = singular | (jnp.abs(pivot) <= threshold)
            safe_pivot = jnp.where(jnp.abs(pivot) <= threshold, 1.0, pivot)
            for row_offset in range(1, min(lower, size - column - 1) + 1):
                lower_index = diagonal + row_offset
                multiplier = factor[lower_index, column] / safe_pivot
                factor = factor.at[lower_index, column].set(multiplier)
                for column_offset in range(1, min(upper + lower, size - column - 1) + 1):
                    band_index = diagonal + row_offset - column_offset
                    if 0 <= band_index < factor_width:
                        target = column + column_offset
                        factor = factor.at[band_index, target].add(
                            -multiplier * factor[diagonal - column_offset, target]
                        )
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
) -> tuple[Array, Array]:
    lower = operator.lower_bandwidth
    upper_factor = operator.upper_bandwidth + lower
    size = operator.source.size
    diagonal = state.diagonal_index
    factors = state.factor.reshape((-1, state.factor.shape[-2], size))
    pivots = state.pivots.reshape((-1, size))
    flattened_rhs = rhs.reshape((-1, size, rhs.shape[-1]))

    def solve_one(factor, pivot_rows, value):
        for row in range(size):
            pivot_row = pivot_rows[row]
            first = value[row]
            second = value[pivot_row]
            value = value.at[row].set(second)
            value = value.at[pivot_row].set(first)
        for row in range(size):
            for offset in range(1, min(lower, row) + 1):
                value = value.at[row].add(
                    -factor[diagonal + offset, row - offset] * value[row - offset]
                )
        for row in range(size - 1, -1, -1):
            for offset in range(1, min(upper_factor, size - row - 1) + 1):
                value = value.at[row].add(
                    -factor[diagonal - offset, row + offset] * value[row + offset]
                )
            pivot = factor[diagonal, row]
            value = value.at[row].set(value[row] / jnp.where(pivot == 0, 1.0, pivot))
        return value

    value = jax.vmap(solve_one)(factors, pivots, flattened_rhs).reshape(rhs.shape)
    failed = state.singular[..., None] | jnp.any(~jnp.isfinite(value), axis=-2)
    return value, failed


def _prepare_woodbury(operator: DiagonalPlusLowRankLinearOperator, /):
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


def _prepare_operator(operator: Any, /) -> Any:
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
        singular = jnp.any(diagonal == 0) & ~operator.unit_diagonal
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

        def woodbury_solve(_):
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
        failed = jnp.asarray(False)
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

    def eliminate(index, state):
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

        def update_fill(values):
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
    scale = jnp.maximum(
        jnp.max(jnp.abs(operator.diagonal)),
        jnp.maximum(
            jnp.max(jnp.abs(operator.upper), initial=0.0),
            jnp.max(jnp.abs(operator.lower), initial=0.0),
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

    def rotate_rhs(index, rhs_state):
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

    def substitute(reverse_index, result):
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
