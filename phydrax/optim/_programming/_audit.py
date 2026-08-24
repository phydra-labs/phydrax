#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ._quadratic import _max_abs, _min_value, QuadraticProgram


class PrimalRayAudit(NamedTuple):
    ray: Array
    residual_norm: Array
    objective: Array
    valid: Array


class DualRayAudit(NamedTuple):
    equality_ray: Array
    inequality_ray: Array
    lower_bound_ray: Array
    upper_bound_ray: Array
    residual_norm: Array
    objective: Array
    valid: Array


def _normalize_vector(value: Array, /) -> tuple[Array, Array]:
    scale = jnp.maximum(1.0, _max_abs(value))
    return value / scale[..., None], scale


def audit_primal_recession_ray(
    problem: QuadraticProgram,
    candidate: Array,
    /,
    *,
    tolerance: float,
) -> PrimalRayAudit:
    """Audit a primal recession direction proving dual infeasibility."""

    ray, _ = _normalize_vector(jnp.asarray(candidate, dtype=problem.linear.dtype))
    quadratic_residual = _max_abs(oe.contract("...ij,...j->...i", problem.quadratic, ray))
    equality_residual = _max_abs(
        oe.contract(
            "...ij,...j->...i",
            problem.equality_matrix[..., : problem.num_user_equalities, :],
            ray,
        )
    )
    inequality_direction = oe.contract(
        "...ij,...j->...i",
        problem.inequality_matrix[..., : problem.num_user_inequalities, :],
        ray,
    )
    inequality_residual = _max_abs(jnp.maximum(inequality_direction, 0.0))
    bound_residual = jnp.zeros(problem.batch_shape, dtype=ray.dtype)
    lower_finite = jnp.isfinite(problem.lower_bounds)
    upper_finite = jnp.isfinite(problem.upper_bounds)
    fixed = lower_finite & upper_finite & (problem.lower_bounds == problem.upper_bounds)
    lower_only = lower_finite & ~fixed
    upper_only = upper_finite & ~fixed
    bound_violation = jnp.where(
        fixed,
        jnp.abs(ray),
        jnp.where(
            lower_only & upper_only,
            jnp.abs(ray),
            jnp.where(
                lower_only,
                jnp.maximum(-ray, 0.0),
                jnp.where(upper_only, jnp.maximum(ray, 0.0), 0.0),
            ),
        ),
    )
    bound_residual = _max_abs(bound_violation)
    objective = jnp.sum(problem.linear * ray, axis=-1)
    residual = jnp.maximum(
        jnp.maximum(quadratic_residual, equality_residual),
        jnp.maximum(inequality_residual, bound_residual),
    )
    threshold = jnp.asarray(tolerance, dtype=ray.dtype)
    valid = (
        jnp.all(jnp.isfinite(ray), axis=-1)
        & (residual <= threshold)
        & (objective < -threshold)
    )
    return PrimalRayAudit(ray, residual, objective, valid)


def audit_dual_infeasibility_ray(
    problem: QuadraticProgram,
    equality_candidate: Array,
    inequality_candidate: Array,
    lower_bound_candidate: Array,
    upper_bound_candidate: Array,
    /,
    *,
    tolerance: float,
) -> DualRayAudit:
    """Audit a Farkas ray proving that the primal constraint set is empty."""

    equality = jnp.asarray(equality_candidate, dtype=problem.linear.dtype)
    inequality = jnp.asarray(inequality_candidate, dtype=problem.linear.dtype)
    lower = jnp.asarray(lower_bound_candidate, dtype=problem.linear.dtype)
    upper = jnp.asarray(upper_bound_candidate, dtype=problem.linear.dtype)
    scale = jnp.maximum(
        1.0,
        jnp.maximum(
            jnp.maximum(_max_abs(equality), _max_abs(inequality)),
            jnp.maximum(_max_abs(lower), _max_abs(upper)),
        ),
    )
    equality = equality / scale[..., None]
    inequality = inequality / scale[..., None]
    lower = lower / scale[..., None]
    upper = upper / scale[..., None]
    stationarity = (
        oe.contract(
            "...ji,...j->...i",
            problem.equality_matrix[..., : problem.num_user_equalities, :],
            equality,
        )
        + oe.contract(
            "...ji,...j->...i",
            problem.inequality_matrix[..., : problem.num_user_inequalities, :],
            inequality,
        )
        - lower
        + upper
    )
    lower_term = jnp.where(
        jnp.isfinite(problem.lower_bounds), problem.lower_bounds * lower, 0.0
    )
    upper_term = jnp.where(
        jnp.isfinite(problem.upper_bounds), problem.upper_bounds * upper, 0.0
    )
    objective = (
        jnp.sum(
            problem.equality_rhs[..., : problem.num_user_equalities] * equality,
            axis=-1,
        )
        + jnp.sum(
            problem.inequality_rhs[..., : problem.num_user_inequalities] * inequality,
            axis=-1,
        )
        - jnp.sum(lower_term, axis=-1)
        + jnp.sum(upper_term, axis=-1)
    )
    residual = _max_abs(stationarity)
    threshold = jnp.asarray(tolerance, dtype=problem.linear.dtype)
    valid = (
        (residual <= threshold)
        & (objective < -threshold)
        & (_min_value(inequality) >= -threshold)
        & (_min_value(lower) >= -threshold)
        & (_min_value(upper) >= -threshold)
        & jnp.isfinite(objective)
    )
    return DualRayAudit(
        equality,
        inequality,
        lower,
        upper,
        residual,
        objective,
        valid,
    )


__all__ = [
    "DualRayAudit",
    "PrimalRayAudit",
    "audit_dual_infeasibility_ray",
    "audit_primal_recession_ray",
]
