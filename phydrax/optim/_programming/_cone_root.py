#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from operator import index

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule


class SafeguardedRootResult(StrictModule):
    root: Array
    residual: Array
    bracket_width: Array
    iterations: Array
    finite: Array
    converged: Array


def safeguarded_newton_bisection(
    function: Callable[[Array], tuple[Array, Array]],
    lower: Array,
    upper: Array,
    /,
    *,
    absolute_tolerance: Array,
    relative_tolerance: Array,
    maximum_steps: int = 64,
) -> SafeguardedRootResult:
    """Solve one bracketed scalar root with static Newton-bisection control flow."""

    if isinstance(maximum_steps, bool):
        raise TypeError("maximum_steps must be an integer.")
    steps = index(maximum_steps)
    if steps < 1:
        raise ValueError("maximum_steps must be positive.")
    lower_ = jnp.asarray(lower)
    upper_ = jnp.asarray(upper)
    absolute = jnp.asarray(absolute_tolerance)
    relative = jnp.asarray(relative_tolerance)
    if (
        not jnp.issubdtype(lower_.dtype, jnp.floating)
        or upper_.dtype != lower_.dtype
        or absolute.dtype != lower_.dtype
        or relative.dtype != lower_.dtype
    ):
        raise TypeError("Root brackets and tolerances must share a real floating dtype.")
    if lower_.shape != upper_.shape:
        raise ValueError("Root lower and upper brackets must share shape.")
    if absolute.shape not in ((), lower_.shape) or relative.shape not in (
        (),
        lower_.shape,
    ):
        raise ValueError("Root tolerances must be scalar or match the bracket shape.")
    broadcast_shape = jnp.broadcast_shapes(
        lower_.shape,
        absolute.shape,
        relative.shape,
    )
    lower_ = jnp.broadcast_to(lower_, broadcast_shape)
    upper_ = jnp.broadcast_to(upper_, broadcast_shape)
    absolute = jnp.broadcast_to(absolute, broadcast_shape)
    relative = jnp.broadcast_to(relative, broadcast_shape)
    invalid = (
        ~jnp.isfinite(lower_)
        | ~jnp.isfinite(upper_)
        | ~jnp.isfinite(absolute)
        | ~jnp.isfinite(relative)
        | (lower_ > upper_)
        | (absolute < 0.0)
        | (relative < 0.0)
    )
    lower_ = jnp.where(invalid, jnp.nan, lower_)
    f_lower, _ = function(lower_)
    f_upper, _ = function(upper_)
    endpoint_finite = (
        jnp.isfinite(lower_)
        & jnp.isfinite(upper_)
        & jnp.isfinite(f_lower)
        & jnp.isfinite(f_upper)
        & (lower_ <= upper_)
    )
    bracketed = endpoint_finite & (
        (f_lower == 0.0)
        | (f_upper == 0.0)
        | (jnp.signbit(f_lower) != jnp.signbit(f_upper))
    )
    midpoint = 0.5 * lower_ + 0.5 * upper_
    endpoint_exact = (f_lower == 0.0) | (f_upper == 0.0)
    endpoint_root = jnp.where(jnp.abs(f_lower) <= jnp.abs(f_upper), lower_, upper_)
    root = jnp.where(endpoint_exact, endpoint_root, midpoint)
    lower_ = jnp.where(endpoint_exact, endpoint_root, lower_)
    upper_ = jnp.where(endpoint_exact, endpoint_root, upper_)
    f_lower = jnp.where(endpoint_exact, 0.0, f_lower)
    f_upper = jnp.where(endpoint_exact, 0.0, f_upper)
    state = (
        lower_,
        upper_,
        f_lower,
        f_upper,
        root,
        jnp.zeros_like(lower_, dtype=jnp.int32),
        bracketed,
    )

    def body(_, current):
        lo, hi, flo, fhi, point, iterations, active = current
        value, derivative = function(point)
        exact = value == 0.0
        lo = jnp.where(exact, point, lo)
        hi = jnp.where(exact, point, hi)
        flo = jnp.where(exact, 0.0, flo)
        fhi = jnp.where(exact, 0.0, fhi)
        value_scale = jnp.maximum(jnp.maximum(jnp.abs(flo), jnp.abs(fhi)), 1.0)
        residual_tolerance = absolute + relative * value_scale
        width_tolerance = absolute + relative * jnp.maximum(jnp.abs(point), 1.0)
        converged = (jnp.abs(value) <= residual_tolerance) & (hi - lo <= width_tolerance)
        derivative_floor = jnp.sqrt(jnp.finfo(point.dtype).eps) * jnp.maximum(
            jnp.abs(value), 1.0
        )
        newton = point - value / jnp.where(derivative != 0.0, derivative, 1.0)
        newton_valid = (
            jnp.isfinite(newton)
            & jnp.isfinite(derivative)
            & (jnp.abs(derivative) > derivative_floor)
            & (newton > lo)
            & (newton < hi)
        )
        newton_value, _ = function(newton)
        improving = jnp.isfinite(newton_value) & (jnp.abs(newton_value) < jnp.abs(value))
        bisected = 0.5 * lo + 0.5 * hi
        candidate = jnp.where(newton_valid & improving, newton, bisected)
        candidate_value, _ = function(candidate)
        lower_side = (flo == 0.0) | (jnp.signbit(flo) != jnp.signbit(candidate_value))
        next_lo = jnp.where(lower_side, lo, candidate)
        next_flo = jnp.where(lower_side, flo, candidate_value)
        next_hi = jnp.where(lower_side, candidate, hi)
        next_fhi = jnp.where(lower_side, candidate_value, fhi)
        update = active & ~converged & jnp.isfinite(value) & jnp.isfinite(candidate_value)
        return (
            jnp.where(update, next_lo, lo),
            jnp.where(update, next_hi, hi),
            jnp.where(update, next_flo, flo),
            jnp.where(update, next_fhi, fhi),
            jnp.where(update, candidate, point),
            iterations + update.astype(jnp.int32),
            active & ~converged,
        )

    lower_, upper_, _, _, root, iterations, _ = jax.lax.fori_loop(
        0,
        steps,
        body,
        state,
    )
    residual, _ = function(root)
    bracket_width = upper_ - lower_
    value_scale = jnp.maximum(jnp.maximum(jnp.abs(f_lower), jnp.abs(f_upper)), 1.0)
    residual_tolerance = absolute + relative * value_scale
    width_tolerance = absolute + relative * jnp.maximum(jnp.abs(root), 1.0)
    finite = (
        bracketed
        & jnp.isfinite(root)
        & jnp.isfinite(residual)
        & jnp.isfinite(bracket_width)
    )
    converged = (
        finite
        & (jnp.abs(residual) <= residual_tolerance)
        & (bracket_width <= width_tolerance)
    )
    return SafeguardedRootResult(
        root,
        jnp.abs(residual),
        bracket_width,
        iterations,
        finite,
        converged,
    )


__all__ = ["SafeguardedRootResult", "safeguarded_newton_bisection"]
