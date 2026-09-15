#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._operators import AbstractLinearOperator
from ..krylov._results import KrylovBreakdownStatus


class StreamingShiftedOutput(NamedTuple):
    positive_solutions: Array
    recurrence_residual_norm: Array
    iterations: Array
    unfinished: Array
    curvature_failure: Array
    finite: Array
    breakdown_status: Array
    matvec_count: Array
    reference_shift: Array


class _StreamingState(NamedTuple):
    iteration: Array
    previous: Array
    current: Array
    previous_beta: Array
    solutions: Array
    directions: Array
    residual_factors: Array
    inverse_pivots: Array
    recurrence_weights: Array
    active: Array
    iterations: Array
    curvature_failure: Array
    finite: Array
    breakdown_status: Array
    matvec_count: Array


def streaming_shifted_lanczos(
    operator: AbstractLinearOperator,
    rhs: Array,
    shifts: Array,
    admissible: Array,
    /,
    *,
    max_steps: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    breakdown_tolerance: float | None,
) -> StreamingShiftedOutput:
    """Solve ``(A-z_j I)y_j=b`` through one fixed-shape Lanczos recurrence."""
    if rhs.ndim != 1:
        raise ValueError("Streaming shifted right-hand side must be one vector.")
    if shifts.ndim != 1 or admissible.shape != shifts.shape:
        raise ValueError("Streaming shifts and admissibility must be matching vectors.")
    if rhs.shape[0] != operator.source.size:
        raise ValueError(
            "Streaming right-hand side width must match the operator source."
        )
    dimension = int(max_steps)
    if dimension < 1:
        raise ValueError("Streaming shifted max_steps must be positive.")

    real_dtype = rhs.real.dtype
    tolerance = jnp.asarray(
        jnp.sqrt(jnp.finfo(real_dtype).eps)
        if breakdown_tolerance is None
        else breakdown_tolerance,
        dtype=real_dtype,
    )
    rhs_norm = _norm(operator, rhs)
    threshold = (
        jnp.asarray(absolute_tolerance, dtype=real_dtype)
        + jnp.asarray(relative_tolerance, dtype=real_dtype) * rhs_norm
    )
    zero_rhs = rhs_norm == 0.0
    safe_rhs_norm = jnp.where(zero_rhs, 1.0, rhs_norm)
    current = rhs / safe_rhs_norm
    count = shifts.shape[0]
    real_shifts = jnp.real(shifts).astype(real_dtype)
    reference_shift = jnp.max(jnp.where(admissible, real_shifts, -jnp.inf))
    reference_shift = jnp.where(jnp.any(admissible), reference_shift, 0.0)
    relative_shifts = reference_shift - real_shifts
    active = admissible & ~zero_rhs
    state = _StreamingState(
        iteration=jnp.asarray(0, dtype=jnp.int32),
        previous=jnp.zeros_like(rhs),
        current=current,
        previous_beta=jnp.asarray(0.0, dtype=real_dtype),
        solutions=jnp.zeros((count, rhs.size), dtype=rhs.dtype),
        directions=jnp.broadcast_to(rhs, (count, rhs.size)),
        residual_factors=jnp.full((count,), rhs_norm, dtype=real_dtype),
        inverse_pivots=jnp.ones((count,), dtype=real_dtype),
        recurrence_weights=jnp.zeros((count,), dtype=real_dtype),
        active=active,
        iterations=jnp.zeros((count,), dtype=jnp.int32),
        curvature_failure=jnp.zeros((count,), dtype=bool),
        finite=jnp.full((count,), jnp.all(jnp.isfinite(rhs)), dtype=bool),
        breakdown_status=jnp.asarray(int(KrylovBreakdownStatus.NONE), dtype=jnp.int32),
        matvec_count=jnp.asarray(0, dtype=jnp.int32),
    )

    def continue_iteration(value: _StreamingState) -> Array:
        return (value.iteration < dimension) & jnp.any(value.active)

    def iterate(value: _StreamingState) -> _StreamingState:
        image = _action_coordinates(operator, value.current)
        reference_image = image - reference_shift.astype(rhs.dtype) * value.current
        candidate = (
            reference_image - value.previous_beta.astype(rhs.dtype) * value.previous
        )
        alpha = jnp.real(_inner_coordinates(operator, value.current, candidate))
        residual = candidate - alpha.astype(rhs.dtype) * value.current
        beta_next = _norm(operator, residual)
        candidate_norm = _norm(operator, candidate)
        scale = candidate_norm + jnp.abs(alpha) + jnp.abs(value.previous_beta)
        near_breakdown = beta_next <= tolerance * scale
        finite_shared = (
            jnp.all(jnp.isfinite(image)) & jnp.isfinite(alpha) & jnp.isfinite(beta_next)
        )
        safe_beta = jnp.where(near_breakdown | ~finite_shared, 1.0, beta_next)
        next_vector = residual / safe_beta.astype(rhs.dtype)
        next_vector = jnp.where(
            near_breakdown | ~finite_shared,
            jnp.zeros_like(next_vector),
            next_vector,
        )

        safe_previous_pivot = jnp.where(
            value.inverse_pivots != 0.0,
            value.inverse_pivots,
            1.0,
        )
        previous_term = value.recurrence_weights / safe_previous_pivot
        diagonal = alpha + relative_shifts
        pivots = diagonal - previous_term
        pivot_scale = jnp.abs(diagonal) + jnp.abs(previous_term)
        bad_pivot = value.active & (
            ~jnp.isfinite(pivots) | (pivots <= 0.0) | (pivots <= tolerance * pivot_scale)
        )
        update = value.active & finite_shared & ~bad_pivot
        safe_pivots = jnp.where(update, pivots, 1.0)
        inverse_pivots = 1.0 / safe_pivots
        candidate_solutions = (
            value.solutions + inverse_pivots[:, None].astype(rhs.dtype) * value.directions
        )
        theta = beta_next * inverse_pivots
        residual_factors = -theta * value.residual_factors
        recurrence_weights = theta * theta
        candidate_directions = (
            residual_factors[:, None].astype(rhs.dtype) * next_vector[None, :]
            + recurrence_weights[:, None].astype(rhs.dtype) * value.directions
        )
        solutions = jnp.where(update[:, None], candidate_solutions, value.solutions)
        directions = jnp.where(update[:, None], candidate_directions, value.directions)
        residual_values = jnp.where(
            update,
            residual_factors,
            value.residual_factors,
        )
        pivot_values = jnp.where(update, inverse_pivots, value.inverse_pivots)
        weight_values = jnp.where(
            update,
            recurrence_weights,
            value.recurrence_weights,
        )
        recurrence_converged = jnp.abs(residual_values) <= threshold
        terminate = value.active & (
            recurrence_converged | bad_pivot | ~finite_shared | near_breakdown
        )
        iterations = jnp.where(
            terminate,
            value.iteration + jnp.asarray(1, dtype=jnp.int32),
            value.iterations,
        )
        active_next = value.active & ~terminate
        finite = value.finite & jnp.where(
            value.active,
            finite_shared
            & jnp.isfinite(residual_values)
            & jnp.isfinite(pivot_values)
            & jnp.all(jnp.isfinite(solutions), axis=1)
            & jnp.all(jnp.isfinite(directions), axis=1),
            True,
        )
        breakdown_status = jnp.where(
            ~finite_shared,
            int(KrylovBreakdownStatus.NONFINITE_ACTION),
            jnp.where(
                near_breakdown,
                int(KrylovBreakdownStatus.HAPPY),
                value.breakdown_status,
            ),
        ).astype(jnp.int32)
        return _StreamingState(
            iteration=value.iteration + jnp.asarray(1, dtype=jnp.int32),
            previous=value.current,
            current=next_vector,
            previous_beta=beta_next,
            solutions=solutions,
            directions=directions,
            residual_factors=residual_values,
            inverse_pivots=pivot_values,
            recurrence_weights=weight_values,
            active=active_next,
            iterations=iterations,
            curvature_failure=value.curvature_failure | bad_pivot,
            finite=finite,
            breakdown_status=breakdown_status,
            matvec_count=value.matvec_count + jnp.asarray(1, dtype=jnp.int32),
        )

    final = jax.lax.while_loop(continue_iteration, iterate, state)
    iterations = jnp.where(final.active, final.iteration, final.iterations)
    return StreamingShiftedOutput(
        positive_solutions=final.solutions,
        recurrence_residual_norm=jnp.abs(final.residual_factors),
        iterations=iterations,
        unfinished=final.active,
        curvature_failure=final.curvature_failure,
        finite=final.finite,
        breakdown_status=final.breakdown_status,
        matvec_count=final.matvec_count,
        reference_shift=reference_shift,
    )


def _action_coordinates(operator: AbstractLinearOperator, vector: Array, /) -> Array:
    value = operator.source.unflatten(vector)
    return operator.target.flatten(operator.mv(value))


def _inner_coordinates(
    operator: AbstractLinearOperator,
    left: Array,
    right: Array,
    /,
) -> Array:
    return operator.source.inner(
        operator.source.unflatten(left),
        operator.source.unflatten(right),
    )


def _norm(operator: AbstractLinearOperator, vector: Array, /) -> Array:
    squared = jnp.real(_inner_coordinates(operator, vector, vector))
    return jnp.sqrt(jnp.maximum(squared, 0.0))


__all__ = ["StreamingShiftedOutput", "streaming_shifted_lanczos"]
