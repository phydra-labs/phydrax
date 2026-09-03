#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...optim import AbstractRobustLoss, IdentityLoss


class TriangulationStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    INSUFFICIENT_RAYS = 2
    RANK_DEFICIENT = 3
    ILL_CONDITIONED = 4
    NONCONVERGENCE = 5


class TriangulationResult(StrictModule, NonTrainableState):
    """Triangulated point with all-ray residual, conditioning, and UQ evidence."""

    point: Array
    covariance: Array
    residuals: Array
    ray_weights: Array
    valid: Array
    status: Array
    rank: Array
    condition_number: Array
    residual_norm: Array
    effective_ray_count: Array
    iterations: Array


def _rank_three(matrix: Array, tolerance: float) -> Array:
    scale = jnp.max(jnp.abs(matrix), axis=(-2, -1))
    threshold = tolerance * jnp.maximum(scale, jnp.finfo(matrix.dtype).tiny)
    rank_one = jnp.any(jnp.abs(matrix) > threshold[..., None, None], axis=(-2, -1))
    minors = []
    for row_a, row_b in ((0, 1), (0, 2), (1, 2)):
        for column_a, column_b in ((0, 1), (0, 2), (1, 2)):
            minors.append(
                matrix[..., row_a, column_a] * matrix[..., row_b, column_b]
                - matrix[..., row_a, column_b] * matrix[..., row_b, column_a]
            )
    minors_ = jnp.stack(minors, axis=-1)
    rank_two = jnp.any(jnp.abs(minors_) > (threshold * scale)[..., None], axis=-1)
    determinant = (
        matrix[..., 0, 0]
        * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1]
        * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2]
        * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )
    rank_three = jnp.abs(determinant) > tolerance * jnp.maximum(
        scale**3,
        jnp.finfo(matrix.dtype).tiny,
    )
    return jnp.where(
        rank_three,
        3,
        jnp.where(rank_two, 2, jnp.where(rank_one, 1, 0)),
    ).astype(jnp.int32)


def triangulate_weighted_rays(
    origins: ArrayLike,
    directions: ArrayLike,
    valid: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    robust_loss: AbstractRobustLoss | None = None,
    maximum_iterations: int = 8,
    convergence_tolerance: float = 1e-7,
    small_solve_plan: SmallLinearSolvePlan | None = None,
) -> TriangulationResult:
    """Triangulate fixed-capacity ray sets by robust all-ray normal equations."""

    origins_ = jnp.asarray(origins)
    directions_ = jnp.asarray(directions)
    valid_ = jnp.asarray(valid, dtype=bool)
    weights_ = jnp.asarray(weights)
    if origins_.shape != directions_.shape or origins_.shape[-1:] != (3,):
        raise ValueError("origins and directions must have the same shape (..., R, 3).")
    if origins_.ndim < 2:
        raise ValueError("Ray arrays must include a fixed-capacity ray axis.")
    ray_shape = origins_.shape[:-1]
    if valid_.shape != ray_shape or weights_.shape != ray_shape:
        raise ValueError("valid and weights must have shape (..., R).")
    if origins_.shape[-2] < 2:
        raise ValueError("Ray capacity must be at least two.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")
    if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be finite and positive.")
    if (
        jnp.issubdtype(origins_.dtype, jnp.complexfloating)
        or jnp.issubdtype(directions_.dtype, jnp.complexfloating)
        or jnp.issubdtype(weights_.dtype, jnp.complexfloating)
    ):
        raise TypeError("Ray geometry and weights must be real-valued.")
    if not jnp.issubdtype(origins_.dtype, jnp.inexact):
        origins_ = origins_.astype(float)
    if not jnp.issubdtype(directions_.dtype, jnp.inexact):
        directions_ = directions_.astype(float)
    dtype = jnp.result_type(origins_, directions_, weights_, 0.0)
    origins_ = origins_.astype(dtype)
    directions_ = directions_.astype(dtype)
    weights_ = weights_.astype(dtype)
    loss = IdentityLoss() if robust_loss is None else robust_loss
    if not isinstance(loss, AbstractRobustLoss):
        raise TypeError("robust_loss must be an AbstractRobustLoss or None.")
    plan = (
        SmallLinearSolvePlan(
            3,
            singular_tolerance=1e-12,
            maximum_condition=1e12,
            refinement_iterations=1,
        )
        if small_solve_plan is None
        else small_solve_plan
    )
    if not isinstance(plan, SmallLinearSolvePlan) or plan.dimension != 3:
        raise TypeError("small_solve_plan must be a three-dimensional plan or None.")

    finite_origin = jnp.all(jnp.isfinite(origins_), axis=-1)
    finite_direction = jnp.all(jnp.isfinite(directions_), axis=-1)
    finite_weight = jnp.isfinite(weights_) & (weights_ >= 0.0)
    direction_norm = jnp.sqrt(jnp.sum(directions_ * directions_, axis=-1))
    direction_ok = direction_norm > jnp.finfo(dtype).eps
    requested_invalid = valid_ & ~(
        finite_origin & finite_direction & finite_weight & direction_ok
    )
    ray_valid = valid_ & finite_origin & finite_direction & finite_weight & direction_ok
    safe_norm = jnp.where(direction_ok, direction_norm, 1.0)
    unit_directions = directions_ / safe_norm[..., None]
    base_weights = jnp.where(ray_valid, weights_, 0.0)
    identity = jnp.eye(3, dtype=dtype)
    projectors = identity - contract(
        "...ri,...rj->...rij",
        unit_directions,
        unit_directions,
    )
    projected_origins = contract("...rij,...rj->...ri", projectors, origins_)
    batch_shape = origins_.shape[:-2]
    identity_rhs = jnp.broadcast_to(identity, batch_shape + (3, 3))

    def solve_with(ray_weights):
        normal = contract("...r,...rij->...ij", ray_weights, projectors)
        right = contract("...r,...ri->...i", ray_weights, projected_origins)
        right_with_inverse = jnp.concatenate((right[..., None], identity_rhs), axis=-1)
        solved = solve_small_linear(plan, normal, right_with_inverse)
        return solved, normal, solved.value[..., 0], solved.value[..., 1:]

    linear, normal, point, inverse_normal = solve_with(base_weights)
    converged = jnp.zeros(batch_shape, dtype=bool)
    iterations = jnp.zeros(batch_shape, dtype=jnp.int32)
    ray_weights = base_weights
    for _ in range(int(maximum_iterations)):
        displacement = point[..., None, :] - origins_
        perpendicular = contract("...rij,...rj->...ri", projectors, displacement)
        squared_residual = jnp.sum(perpendicular * perpendicular, axis=-1)
        robust_weight = jnp.maximum(loss.evaluate(squared_residual).first, 0.0)
        candidate_weights = base_weights * robust_weight
        candidate_linear, candidate_normal, candidate_point, candidate_inverse = (
            solve_with(candidate_weights)
        )
        step_norm = jnp.sqrt(jnp.sum((candidate_point - point) ** 2, axis=-1))
        threshold = convergence_tolerance * (
            1.0 + jnp.sqrt(jnp.sum(point * point, axis=-1))
        )
        step_converged = candidate_linear.successful & (step_norm <= threshold)
        active = linear.successful & ~converged
        point = jnp.where(active[..., None], candidate_point, point)
        normal = jnp.where(active[..., None, None], candidate_normal, normal)
        inverse_normal = jnp.where(
            active[..., None, None], candidate_inverse, inverse_normal
        )
        ray_weights = jnp.where(active[..., None], candidate_weights, ray_weights)
        linear = candidate_linear
        converged = converged | (active & step_converged)
        iterations = iterations + active.astype(jnp.int32)

    displacement = point[..., None, :] - origins_
    perpendicular = contract("...rij,...rj->...ri", projectors, displacement)
    residuals = jnp.sqrt(jnp.sum(perpendicular * perpendicular, axis=-1))
    maximum_weight = jnp.max(ray_weights, axis=-1, initial=0.0)
    effective = ray_valid & (
        ray_weights
        > jnp.maximum(maximum_weight * 1e-12, jnp.finfo(dtype).tiny)[..., None]
    )
    effective_count = jnp.sum(effective, axis=-1).astype(jnp.int32)
    rank = _rank_three(normal, plan.singular_tolerance)
    condition = jnp.where(rank == 3, linear.condition_estimate, jnp.inf)
    input_valid = ~jnp.any(requested_invalid, axis=-1)
    enough = effective_count >= 2
    solve_valid = linear.successful & (rank == 3)
    successful = input_valid & enough & solve_valid & converged
    status = jnp.where(
        ~input_valid,
        int(TriangulationStatus.NONFINITE_INPUT),
        jnp.where(
            ~enough,
            int(TriangulationStatus.INSUFFICIENT_RAYS),
            jnp.where(
                rank < 3,
                int(TriangulationStatus.RANK_DEFICIENT),
                jnp.where(
                    ~linear.successful,
                    int(TriangulationStatus.ILL_CONDITIONED),
                    jnp.where(
                        ~converged,
                        int(TriangulationStatus.NONCONVERGENCE),
                        int(TriangulationStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    weighted_squared_residual = jnp.sum(ray_weights * residuals**2, axis=-1)
    degrees_of_freedom = jnp.maximum(2 * effective_count - 3, 1)
    residual_variance = weighted_squared_residual / degrees_of_freedom.astype(dtype)
    covariance = residual_variance[..., None, None] * inverse_normal
    residual_norm = jnp.sqrt(weighted_squared_residual)
    point = jnp.where(successful[..., None], point, jnp.nan)
    covariance = jnp.where(successful[..., None, None], covariance, jnp.nan)
    return TriangulationResult(
        point,
        covariance,
        residuals,
        ray_weights,
        successful,
        status,
        rank,
        condition,
        residual_norm,
        effective_count,
        iterations,
    )


__all__ = [
    "TriangulationResult",
    "TriangulationStatus",
    "triangulate_weighted_rays",
]
