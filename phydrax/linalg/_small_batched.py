#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

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
        if dimension_ not in (1, 2, 3):
            raise ValueError("SmallLinearSolvePlan supports dimensions 1, 2, and 3.")
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
    first = matrix[..., 0, :]
    second = matrix[..., 1, :]
    third = matrix[..., 2, :]
    cofactor_rows = jnp.stack(
        (jnp.cross(second, third), jnp.cross(third, first), jnp.cross(first, second)),
        axis=-2,
    )
    determinant = jnp.sum(first * cofactor_rows[..., 0, :], axis=-1)
    inverse = (
        jnp.swapaxes(cofactor_rows, -1, -2)
        / jnp.where(determinant != 0.0, determinant, 1.0)[..., None, None]
    )
    return inverse, determinant


def solve_small_linear(
    plan: SmallLinearSolvePlan,
    matrix: ArrayLike,
    right_hand_side: ArrayLike,
    /,
) -> SmallLinearSolveResult:
    matrix_ = jnp.asarray(matrix)
    right = jnp.asarray(right_hand_side)
    dimension = plan.dimension
    if matrix_.shape[-2:] != (dimension, dimension):
        raise ValueError("Small matrix shape does not match the plan dimension.")
    vector_rhs = right.shape == matrix_.shape[:-1]
    if vector_rhs:
        right = right[..., :, None]
    if right.shape[:-2] != matrix_.shape[:-2] or right.shape[-2] != dimension:
        raise ValueError("Small linear right-hand side shape is incompatible.")
    inverse, determinant = _inverse(matrix_, dimension)
    scale = jnp.max(jnp.abs(matrix_), axis=(-2, -1))
    determinant_scale = jnp.maximum(scale**dimension, jnp.finfo(matrix_.dtype).tiny)
    nonsingular = jnp.abs(determinant) > plan.singular_tolerance * determinant_scale
    value = contract("...ij,...jk->...ik", inverse, right)
    refinement_count = jnp.zeros(determinant.shape, dtype=jnp.int32)
    for _ in range(plan.refinement_iterations):
        residual = right - contract("...ij,...jk->...ik", matrix_, value)
        correction = contract("...ij,...jk->...ik", inverse, residual)
        apply = nonsingular[..., None, None] & jnp.all(
            jnp.isfinite(correction), axis=(-2, -1), keepdims=True
        )
        value = value + jnp.where(apply, correction, 0.0)
        refinement_count = refinement_count + apply[..., 0, 0].astype(jnp.int32)
    residual = right - contract("...ij,...jk->...ik", matrix_, value)
    residual_norm = jnp.sqrt(jnp.sum(residual * residual, axis=(-2, -1)))
    matrix_norm = jnp.max(jnp.sum(jnp.abs(matrix_), axis=-1), axis=-1)
    inverse_norm = jnp.max(jnp.sum(jnp.abs(inverse), axis=-1), axis=-1)
    condition = matrix_norm * inverse_norm
    finite = jnp.all(jnp.isfinite(value), axis=(-2, -1))
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


__all__ = ["SmallLinearSolvePlan", "SmallLinearSolveResult", "solve_small_linear"]
