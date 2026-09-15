#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


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
    ):
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


def _lu_factor_four(matrix: Array, /) -> tuple[Array, Array, Array]:
    """Factor scaled 4-by-4 batches with deterministic partial pivoting."""
    factors = matrix
    batch_shape = matrix.shape[:-2]
    permutation = jnp.broadcast_to(
        jnp.eye(4, dtype=matrix.dtype),
        batch_shape + (4, 4),
    )
    parity = jnp.ones(batch_shape, dtype=matrix.real.dtype)
    row_indices = jnp.arange(4)
    for column in range(3):
        pivot = column + jnp.argmax(
            jnp.abs(factors[..., column:, column]),
            axis=-1,
        )
        column_row = jax.nn.one_hot(
            jnp.full(batch_shape, column, dtype=jnp.int32),
            4,
            dtype=matrix.dtype,
        )
        pivot_row = jax.nn.one_hot(pivot, 4, dtype=matrix.dtype)
        swap = (
            jnp.eye(4, dtype=matrix.dtype)
            - column_row[..., :, None] * column_row[..., None, :]
            - pivot_row[..., :, None] * pivot_row[..., None, :]
            + column_row[..., :, None] * pivot_row[..., None, :]
            + pivot_row[..., :, None] * column_row[..., None, :]
        )
        factors = contract("...ij,...jk->...ik", swap, factors)
        permutation = contract("...ij,...jk->...ik", swap, permutation)
        parity = parity * jnp.where(pivot == column, 1.0, -1.0)
        pivot_value = factors[..., column, column]
        safe_pivot = jnp.where(jnp.abs(pivot_value) > 0.0, pivot_value, 1.0)
        multipliers = factors[..., :, column] / safe_pivot[..., None]
        active_rows = row_indices > column
        multipliers = jnp.where(active_rows, multipliers, 0.0)
        update = multipliers[..., :, None] * factors[..., None, column, :]
        active_trailing = active_rows[:, None] & (row_indices[None, :] > column)
        factors = factors - jnp.where(active_trailing, update, 0.0)
        factors = factors.at[..., :, column].set(
            jnp.where(active_rows, multipliers, factors[..., :, column])
        )
    determinant = parity.astype(matrix.dtype) * jnp.prod(
        jnp.diagonal(factors, axis1=-2, axis2=-1),
        axis=-1,
    )
    return factors, permutation, determinant


def _lu_solve_four(factors: Array, permutation: Array, right: Array, /) -> Array:
    """Solve a factored 4-by-4 batch for one or more right-hand sides."""
    permuted = contract("...ij,...jk->...ik", permutation, right)
    lower_solution = jnp.zeros_like(permuted)
    for row in range(4):
        contribution = jnp.sum(
            factors[..., row, :, None] * lower_solution,
            axis=-2,
        )
        value = permuted[..., row, :] - contribution
        lower_solution = lower_solution.at[..., row, :].set(value)
    solution = jnp.zeros_like(permuted)
    for row in range(3, -1, -1):
        contribution = jnp.sum(
            factors[..., row, :, None] * solution,
            axis=-2,
        )
        pivot = factors[..., row, row]
        safe_pivot = jnp.where(jnp.abs(pivot) > 0.0, pivot, 1.0)
        value = (lower_solution[..., row, :] - contribution) / safe_pivot[..., None]
        solution = solution.at[..., row, :].set(value)
    return solution


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
        determinant = a * d - b * c
        adjugate = jnp.stack((d, -b, -c, a), axis=-1).reshape(matrix.shape)
        return adjugate / jnp.where(determinant != 0.0, determinant, 1.0)[
            ..., None, None
        ], determinant
    if dimension == 3:
        first = matrix[..., 0, :]
        second = matrix[..., 1, :]
        third = matrix[..., 2, :]
        cofactor_rows = jnp.stack(
            (
                jnp.cross(second, third),
                jnp.cross(third, first),
                jnp.cross(first, second),
            ),
            axis=-2,
        )
        determinant = jnp.sum(first * cofactor_rows[..., 0, :], axis=-1)
        inverse = (
            jnp.swapaxes(cofactor_rows, -1, -2)
            / jnp.where(determinant != 0.0, determinant, 1.0)[..., None, None]
        )
        return inverse, determinant
    factors, permutation, determinant = _lu_factor_four(matrix)
    identity = jnp.broadcast_to(
        jnp.eye(4, dtype=matrix.dtype),
        matrix.shape,
    )
    return _lu_solve_four(factors, permutation, identity), determinant


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
        value = value.astype(float)
    dimension = plan.dimension
    if value.shape[-2:] != (dimension, dimension):
        raise ValueError("Small matrix shape does not match the plan dimension.")
    scale = jnp.max(jnp.abs(value), axis=(-2, -1))
    safe_scale = jnp.where(scale > 0.0, scale, 1.0)
    scaled = value / safe_scale[..., None, None]
    if dimension == 1:
        determinant = scaled[..., 0, 0]
    elif dimension == 2:
        determinant = (
            scaled[..., 0, 0] * scaled[..., 1, 1] - scaled[..., 0, 1] * scaled[..., 1, 0]
        )
    elif dimension == 3:
        determinant = jnp.sum(
            scaled[..., 0, :] * jnp.cross(scaled[..., 1, :], scaled[..., 2, :]),
            axis=-1,
        )
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
        matrix_ = matrix_.astype(float)
    right = jnp.asarray(right_hand_side)
    dimension = plan.dimension
    if matrix_.shape[-2:] != (dimension, dimension):
        raise ValueError("Small matrix shape does not match the plan dimension.")
    vector_rhs = right.shape == matrix_.shape[:-1]
    if vector_rhs:
        right = right[..., :, None]
    if right.shape[:-2] != matrix_.shape[:-2] or right.shape[-2] != dimension:
        raise ValueError("Small linear right-hand side shape is incompatible.")
    scale = jnp.max(jnp.abs(matrix_), axis=(-2, -1))
    safe_scale = jnp.where(scale > 0.0, scale, 1.0)
    scaled_matrix = matrix_ / safe_scale[..., None, None]
    if dimension == 4:
        factors, permutation, scaled_determinant = _lu_factor_four(scaled_matrix)
        identity = jnp.broadcast_to(
            jnp.eye(4, dtype=matrix_.dtype),
            matrix_.shape,
        )
        scaled_inverse = _lu_solve_four(factors, permutation, identity)
        value = _lu_solve_four(
            factors,
            permutation,
            right / safe_scale[..., None, None],
        )
    else:
        scaled_inverse, scaled_determinant = _inverse(scaled_matrix, dimension)
        value = contract(
            "...ij,...jk->...ik",
            scaled_inverse / safe_scale[..., None, None],
            right,
        )
    inverse = scaled_inverse / safe_scale[..., None, None]
    determinant = scaled_determinant * safe_scale**dimension
    dtype_tolerance = float(dimension) * jnp.finfo(matrix_.real.dtype).eps
    singular_tolerance = jnp.maximum(
        jnp.asarray(plan.singular_tolerance, dtype=matrix_.real.dtype),
        jnp.asarray(dtype_tolerance, dtype=matrix_.real.dtype),
    )
    nonsingular = jnp.abs(scaled_determinant) > singular_tolerance
    refinement_count = jnp.zeros(determinant.shape, dtype=jnp.int32)
    for _ in range(plan.refinement_iterations):
        residual = right - contract("...ij,...jk->...ik", matrix_, value)
        correction = (
            _lu_solve_four(
                factors,
                permutation,
                residual / safe_scale[..., None, None],
            )
            if dimension == 4
            else contract("...ij,...jk->...ik", inverse, residual)
        )
        candidate = value + correction
        candidate_residual = right - contract(
            "...ij,...jk->...ik",
            matrix_,
            candidate,
        )
        residual_size = jnp.sum(jnp.abs(residual) ** 2, axis=(-2, -1))
        candidate_size = jnp.sum(
            jnp.abs(candidate_residual) ** 2,
            axis=(-2, -1),
        )
        apply = (
            nonsingular
            & jnp.all(jnp.isfinite(correction), axis=(-2, -1))
            & (candidate_size < residual_size)
        )
        value = jnp.where(apply[..., None, None], candidate, value)
        refinement_count = refinement_count + apply.astype(jnp.int32)
    residual = right - contract("...ij,...jk->...ik", matrix_, value)
    residual_norm = jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2, axis=(-2, -1)))
    matrix_norm = jnp.max(jnp.sum(jnp.abs(matrix_), axis=-1), axis=-1)
    inverse_norm = jnp.max(jnp.sum(jnp.abs(inverse), axis=-1), axis=-1)
    condition = matrix_norm * inverse_norm
    finite = (
        jnp.all(jnp.isfinite(matrix_), axis=(-2, -1))
        & jnp.all(jnp.isfinite(right), axis=(-2, -1))
        & jnp.all(jnp.isfinite(value), axis=(-2, -1))
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
