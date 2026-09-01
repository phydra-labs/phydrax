#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


class TridiagonalLineSolveResult(StrictModule):
    """Batched tridiagonal line solution with residual and pivot evidence."""

    value: Array
    residual_norm: Array
    minimum_pivot: Array
    finite: Array
    nonsingular: Array
    successful: Array


def solve_tridiagonal_lines(
    lower: ArrayLike,
    diagonal: ArrayLike,
    upper: ArrayLike,
    rhs: ArrayLike,
    axis: int = -1,
    /,
    *,
    pivot_tolerance: float = 1.0e-14,
) -> TridiagonalLineSolveResult:
    """Solve independent tridiagonal systems along ``axis`` without host loops."""
    lower_ = jnp.asarray(lower)
    diagonal_ = jnp.asarray(diagonal, dtype=lower_.dtype)
    upper_ = jnp.asarray(upper, dtype=lower_.dtype)
    rhs_ = jnp.asarray(rhs, dtype=lower_.dtype)
    if not (lower_.shape == diagonal_.shape == upper_.shape == rhs_.shape):
        raise ValueError("Tridiagonal line coefficients and RHS must share one shape.")
    if lower_.ndim == 0:
        raise ValueError("Tridiagonal line systems require at least one axis.")
    axis_ = int(axis) % lower_.ndim
    a = jnp.moveaxis(lower_, axis_, 0)
    b = jnp.moveaxis(diagonal_, axis_, 0)
    c = jnp.moveaxis(upper_, axis_, 0)
    d = jnp.moveaxis(rhs_, axis_, 0)
    tolerance = jnp.asarray(pivot_tolerance, dtype=b.dtype)

    first_pivot = b[0]
    first_safe = jnp.where(jnp.abs(first_pivot) > tolerance, first_pivot, 1.0)
    first_c = c[0] / first_safe
    first_d = d[0] / first_safe

    def forward(carry, row):
        previous_c, previous_d, minimum = carry
        a_row, b_row, c_row, d_row = row
        pivot = b_row - a_row * previous_c
        safe = jnp.where(jnp.abs(pivot) > tolerance, pivot, 1.0)
        modified_c = c_row / safe
        modified_d = (d_row - a_row * previous_d) / safe
        minimum = jnp.minimum(minimum, jnp.min(jnp.abs(pivot)))
        return (modified_c, modified_d, minimum), (modified_c, modified_d)

    initial_minimum = jnp.min(jnp.abs(first_pivot))
    (_, _, minimum_pivot), (tail_c, tail_d) = jax.lax.scan(
        forward,
        (first_c, first_d, initial_minimum),
        (a[1:], b[1:], c[1:], d[1:]),
    )
    modified_c = jnp.concatenate((first_c[None], tail_c), axis=0)
    modified_d = jnp.concatenate((first_d[None], tail_d), axis=0)

    def backward(next_value, row):
        c_row, d_row = row
        value = d_row - c_row * next_value
        return value, value

    last = modified_d[-1]
    _, reversed_values = jax.lax.scan(
        backward,
        last,
        (modified_c[:-1][::-1], modified_d[:-1][::-1]),
    )
    moved_value = jnp.concatenate((reversed_values[::-1], last[None]), axis=0)
    value = jnp.moveaxis(moved_value, 0, axis_)

    rolled_lower = jnp.roll(value, 1, axis=axis_)
    rolled_upper = jnp.roll(value, -1, axis=axis_)
    lower_term = lower_ * rolled_lower
    upper_term = upper_ * rolled_upper
    lower_location = [slice(None)] * value.ndim
    upper_location = [slice(None)] * value.ndim
    lower_location[axis_] = 0
    upper_location[axis_] = value.shape[axis_] - 1
    lower_term = lower_term.at[tuple(lower_location)].set(0.0)
    upper_term = upper_term.at[tuple(upper_location)].set(0.0)
    residual = lower_term + diagonal_ * value + upper_term - rhs_
    residual_norm = jnp.sqrt(jnp.real(jnp.vdot(residual, residual)))
    finite = (
        jnp.all(jnp.isfinite(value))
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(minimum_pivot)
    )
    nonsingular = minimum_pivot > tolerance
    return TridiagonalLineSolveResult(
        value=value,
        residual_norm=residual_norm,
        minimum_pivot=minimum_pivot,
        finite=finite,
        nonsingular=nonsingular,
        successful=finite & nonsingular,
    )


__all__ = ["TridiagonalLineSolveResult", "solve_tridiagonal_lines"]
