#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import functools
import operator
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


_LOCAL_EXTENT = 4


class SmallLinearSolvePlan(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    singular_tolerance: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    refinement_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        singular_tolerance: float = 1e-12,
        maximum_condition: float = 1e12,
        refinement_iterations: int = 1,
    ) -> None:
        dimension_ = int(dimension)
        if dimension_ not in (1, 2, 3, 4):
            raise ValueError("SmallLinearSolvePlan supports dimensions 1 through 4.")
        if singular_tolerance <= 0.0 or maximum_condition <= 1.0:
            raise ValueError("Small linear solve tolerances are invalid.")
        if refinement_iterations < 0:
            raise ValueError("refinement_iterations must be non-negative.")
        self.dimension = dimension_
        self.singular_tolerance = float(singular_tolerance)
        self.maximum_condition = float(maximum_condition)
        self.refinement_iterations = int(refinement_iterations)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "small-linear-solve-plan",
                "dimension": dimension_,
                "singular_tolerance": singular_tolerance,
                "maximum_condition": maximum_condition,
                "refinement_iterations": refinement_iterations,
            }
        )


class SmallLinearSolveResult(StrictModule):
    value: Array
    determinant: Array
    rank: Array
    condition_estimate: Array
    residual_norm: Array
    refinement_iterations: Array
    successful: Array
    status: Array


def _local_fold(
    value: Array,
    combine: Callable[[Array, Array], Array],
    reduce: Callable[..., Array],
    /,
) -> Array:
    """Fold the trailing axis left to right when its static extent is local.

    Matrix axes are bounded by the plan dimension, so their folds unroll into
    elementwise algebra that XLA fuses with the surrounding solve and its
    derivatives. Only caller-sized right-hand-side column axes wider than the
    local bound use the generic reduction.
    """
    extent = value.shape[-1]
    if not 0 < extent <= _LOCAL_EXTENT:
        return reduce(value, axis=-1)
    return functools.reduce(combine, [value[..., index] for index in range(extent)])


def _local_sum(value: Array, /) -> Array:
    return _local_fold(value, operator.add, jnp.sum)


def _local_all(value: Array, /) -> Array:
    return _local_fold(value, jnp.logical_and, jnp.all)


def _local_max(value: Array, /) -> Array:
    return _local_fold(value, jnp.maximum, jnp.max)


def _local_product(left: Array, right: Array, /) -> Array:
    """Multiply batched bounded matrices by an unrolled sum over the inner index."""
    return functools.reduce(
        operator.add,
        [
            left[..., :, inner, None] * right[..., None, inner, :]
            for inner in range(left.shape[-1])
        ],
    )


def _matrix_scale(matrix: Array, /) -> Array:
    """Return the safe max-entry scale with its original differentiation path."""
    scale = _local_max(_local_max(jnp.abs(matrix)))
    return jnp.where(scale > 0.0, scale, 1.0).astype(matrix.dtype)


def _lu_factor_four(matrix: Array, /) -> tuple[Array, Array, Array]:
    """Factor scaled 4-by-4 batches with deterministic partial pivoting."""
    factors = matrix
    batch_shape = matrix.shape[:-2]
    identity = jnp.broadcast_to(jnp.eye(4, dtype=matrix.dtype), matrix.shape)
    permutation = identity
    parity = jnp.ones(batch_shape, dtype=matrix.real.dtype)
    row_indices = jnp.arange(4, dtype=jnp.int32)
    for column in range(3):
        # First maximal magnitude wins and a NaN candidate is maximal, as in argmax.
        pivot = jnp.full(batch_shape, column, dtype=jnp.int32)
        largest = jnp.abs(factors[..., column, column])
        for row in range(column + 1, 4):
            candidate = jnp.abs(factors[..., row, column])
            take = (candidate > largest) | (jnp.isnan(candidate) & ~jnp.isnan(largest))
            pivot = jnp.where(take, row, pivot)
            largest = jnp.where(take, candidate, largest)
        column_row = jnp.broadcast_to(
            jnp.eye(4, dtype=matrix.dtype)[column], batch_shape + (4,)
        )
        pivot_row = jax.nn.one_hot(pivot, 4, dtype=matrix.dtype)
        swap = (
            identity
            - column_row[..., :, None] * column_row[..., None, :]
            - pivot_row[..., :, None] * pivot_row[..., None, :]
            + column_row[..., :, None] * pivot_row[..., None, :]
            + pivot_row[..., :, None] * column_row[..., None, :]
        )
        factors = _local_product(swap, factors)
        permutation = _local_product(swap, permutation)
        parity = parity * jnp.where(pivot == column, 1.0, -1.0)
        pivot_value = factors[..., column, column]
        safe_pivot = jnp.where(jnp.abs(pivot_value) > 0.0, pivot_value, 1.0)
        multipliers = factors[..., :, column] / safe_pivot[..., None]
        active_rows = jnp.broadcast_to(row_indices > column, batch_shape + (4,))
        multipliers = jnp.where(active_rows, multipliers, 0.0)
        update = multipliers[..., :, None] * factors[..., None, column, :]
        active_trailing = active_rows[..., :, None] & active_rows[..., None, :]
        factors = factors - jnp.where(active_trailing, update, 0.0)
        column_entries = jnp.broadcast_to(row_indices == column, batch_shape + (4,))
        multiplier_entries = active_rows[..., :, None] & column_entries[..., None, :]
        factors = jnp.where(multiplier_entries, multipliers[..., :, None], factors)
    determinant = parity.astype(matrix.dtype) * functools.reduce(
        operator.mul,
        [factors[..., index, index] for index in range(4)],
    )
    return factors, permutation, determinant


def _lu_solve_four(factors: Array, permutation: Array, right: Array, /) -> Array:
    """Solve a factored 4-by-4 batch for one or more right-hand sides.

    Each substitution row accumulates all four factor columns in index order,
    with unsolved rows held at zero, so non-finite factors propagate exactly as
    in a dense row product.
    """
    permuted = _local_product(permutation, right)
    zero = jnp.zeros_like(permuted[..., 0, :])
    lower_solution = [zero] * 4
    for row in range(4):
        contribution = functools.reduce(
            operator.add,
            [
                factors[..., row, column, None] * lower_solution[column]
                for column in range(4)
            ],
        )
        lower_solution[row] = permuted[..., row, :] - contribution
    solution = [zero] * 4
    for row in range(3, -1, -1):
        contribution = functools.reduce(
            operator.add,
            [factors[..., row, column, None] * solution[column] for column in range(4)],
        )
        pivot = factors[..., row, row]
        safe_pivot = jnp.where(jnp.abs(pivot) > 0.0, pivot, 1.0)
        solution[row] = (lower_solution[row] - contribution) / safe_pivot[..., None]
    return jnp.stack(solution, axis=-2)


def _cross(left: Array, right: Array, /) -> Array:
    """Batched 3-vector cross product with the ``jnp.cross`` term order."""
    l0, l1, l2 = (left[..., index] for index in range(3))
    r0, r1, r2 = (right[..., index] for index in range(3))
    return jnp.stack((l1 * r2 - l2 * r1, l2 * r0 - l0 * r2, l0 * r1 - l1 * r0), axis=-1)


def _cofactor_rows(matrix: Array, /) -> tuple[Array, Array, Array]:
    first = matrix[..., 0, :]
    second = matrix[..., 1, :]
    third = matrix[..., 2, :]
    return (
        _cross(second, third),
        _cross(third, first),
        _cross(first, second),
    )


def _determinant_three(matrix: Array, cofactor: Array, /) -> Array:
    return _local_sum(matrix[..., 0, :] * cofactor)


def _determinant_two(matrix: Array, /) -> Array:
    return matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]


def _inverse(matrix: Array, dimension: int, /) -> tuple[Array, Array]:
    if dimension == 1:
        determinant = matrix[..., 0, 0]
        inverse = (1.0 / jnp.where(determinant != 0.0, determinant, 1.0))[..., None, None]
        return inverse, determinant
    if dimension == 2:
        a = matrix[..., 0, 0]
        b = matrix[..., 0, 1]
        c = matrix[..., 1, 0]
        d = matrix[..., 1, 1]
        determinant = _determinant_two(matrix)
        adjugate = jnp.stack((d, -b, -c, a), axis=-1).reshape(matrix.shape)
        return adjugate / jnp.where(determinant != 0.0, determinant, 1.0)[
            ..., None, None
        ], determinant
    cofactor_rows = _cofactor_rows(matrix)
    determinant = _determinant_three(matrix, cofactor_rows[0])
    # Stacking cofactor rows as columns forms the adjugate without a transpose.
    inverse = (
        jnp.stack(cofactor_rows, axis=-1)
        / jnp.where(determinant != 0.0, determinant, 1.0)[..., None, None]
    )
    return inverse, determinant


def determinant_small_linear(
    plan: SmallLinearSolvePlan,
    matrix: ArrayLike,
    /,
) -> Array:
    """Evaluate a scaled batched determinant for one-to-four dimensional matrices."""
    if not isinstance(plan, SmallLinearSolvePlan):
        raise TypeError("plan must be a SmallLinearSolvePlan.")
    value = jnp.asarray(matrix)
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype("float64")
    dimension = plan.dimension
    if value.shape[-2:] != (dimension, dimension):
        raise ValueError("Small matrix shape does not match the plan dimension.")
    safe_scale = _matrix_scale(value)
    scaled = value / safe_scale[..., None, None]
    if dimension == 1:
        determinant = scaled[..., 0, 0]
    elif dimension == 2:
        determinant = _determinant_two(scaled)
    elif dimension == 3:
        determinant = _determinant_three(scaled, _cofactor_rows(scaled)[0])
    else:
        _, _, determinant = _lu_factor_four(scaled)
    return determinant * safe_scale**dimension


def solve_small_linear(
    plan: SmallLinearSolvePlan,
    matrix: ArrayLike,
    right_hand_side: ArrayLike,
    /,
) -> SmallLinearSolveResult:
    matrix_ = jnp.asarray(matrix)
    if not jnp.issubdtype(matrix_.dtype, jnp.inexact):
        matrix_ = matrix_.astype("float64")
    right = jnp.asarray(right_hand_side)
    dtype = np.result_type(matrix_.dtype, right.dtype)
    matrix_, right = matrix_.astype(dtype), right.astype(dtype)
    dimension = plan.dimension
    if matrix_.shape[-2:] != (dimension, dimension):
        raise ValueError("Small matrix shape does not match the plan dimension.")
    vector_rhs = right.shape == matrix_.shape[:-1]
    if vector_rhs:
        right = right[..., :, None]
    if right.shape[:-2] != matrix_.shape[:-2] or right.shape[-2] != dimension:
        raise ValueError("Small linear right-hand side shape is incompatible.")
    field_scale = _matrix_scale(matrix_)
    scaled_matrix = matrix_ / field_scale[..., None, None]
    if dimension == 4:
        factors, permutation, scaled_determinant = _lu_factor_four(scaled_matrix)
        identity = jnp.broadcast_to(
            jnp.eye(4, dtype=matrix_.dtype),
            matrix_.shape,
        )
        scaled_inverse = _lu_solve_four(factors, permutation, identity)
        inverse = scaled_inverse / field_scale[..., None, None]
        value = _lu_solve_four(
            factors,
            permutation,
            right / field_scale[..., None, None],
        )
    else:
        scaled_inverse, scaled_determinant = _inverse(scaled_matrix, dimension)
        inverse = scaled_inverse / field_scale[..., None, None]
        value = _local_product(inverse, right)
    determinant = scaled_determinant * field_scale**dimension
    dtype_tolerance = float(dimension) * jnp.finfo(matrix_.real.dtype).eps
    singular_tolerance = jnp.maximum(
        jnp.asarray(plan.singular_tolerance, dtype=matrix_.real.dtype),
        jnp.asarray(dtype_tolerance, dtype=matrix_.real.dtype),
    )
    nonsingular = jnp.abs(scaled_determinant) > singular_tolerance
    refinement_count = jnp.zeros(determinant.shape, dtype=jnp.int32)
    for _ in range(plan.refinement_iterations):
        residual = right - _local_product(matrix_, value)
        correction = (
            _lu_solve_four(
                factors,
                permutation,
                residual / field_scale[..., None, None],
            )
            if dimension == 4
            else _local_product(inverse, residual)
        )
        candidate = value + correction
        candidate_residual = right - _local_product(matrix_, candidate)
        residual_size = _local_sum(_local_sum(jnp.abs(residual) ** 2))
        candidate_size = _local_sum(_local_sum(jnp.abs(candidate_residual) ** 2))
        apply = (
            nonsingular
            & _local_all(_local_all(jnp.isfinite(correction)))
            & (candidate_size < residual_size)
        )
        value = jnp.where(apply[..., None, None], candidate, value)
        refinement_count = refinement_count + apply.astype(jnp.int32)
    residual = right - _local_product(matrix_, value)
    residual_norm = jnp.sqrt(_local_sum(_local_sum(jnp.abs(residual) ** 2)))
    matrix_norm = _local_max(_local_sum(jnp.abs(matrix_)))
    inverse_norm = _local_max(_local_sum(jnp.abs(inverse)))
    condition = matrix_norm * inverse_norm
    finite = (
        _local_all(_local_all(jnp.isfinite(matrix_)))
        & _local_all(_local_all(jnp.isfinite(right)))
        & _local_all(_local_all(jnp.isfinite(value)))
    )
    successful = nonsingular & finite & (condition <= plan.maximum_condition)
    rank = jnp.where(successful, dimension, 0).astype(jnp.int32)
    status = jnp.where(successful, 0, jnp.where(nonsingular, 2, 1)).astype(jnp.int32)
    value = jnp.where(successful[..., None, None], value, 0.0)
    if vector_rhs:
        value = value[..., 0]
    return SmallLinearSolveResult(
        value,
        determinant,
        rank,
        condition,
        residual_norm,
        refinement_count,
        successful,
        status,
    )


def inverse_small_linear(
    plan: SmallLinearSolvePlan,
    matrix: ArrayLike,
    /,
) -> SmallLinearSolveResult:
    """Materialize a batched 1x1 through 4x4 inverse with solve evidence."""
    value = jnp.asarray(matrix)
    dimension = plan.dimension
    if value.shape[-2:] != (dimension, dimension):
        raise ValueError("Small matrix shape does not match the plan dimension.")
    identity = jnp.broadcast_to(jnp.eye(dimension, dtype=value.dtype), value.shape)
    return solve_small_linear(plan, value, identity)


__all__ = [
    "SmallLinearSolvePlan",
    "SmallLinearSolveResult",
    "determinant_small_linear",
    "inverse_small_linear",
    "solve_small_linear",
]
