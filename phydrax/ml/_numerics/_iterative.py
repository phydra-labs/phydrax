#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule


class IterationResult(StrictModule):
    """Fixed-capacity iterative state with convergence history."""

    value: Any
    objective_history: Array
    residual_history: Array
    iterations: Array
    converged: Array
    finite: Array
    method: str = eqx.field(static=True)


def run_fixed_iterations(
    initial: Any,
    step: Callable[[Any, Array], tuple[Any, Array, Array]],
    /,
    *,
    max_iterations: int,
    tolerance: float,
    method: str,
) -> IterationResult:
    """Run a convergence-masked fixed scan without changing output structure."""
    count = int(max_iterations)
    if count <= 0:
        raise ValueError("max_iterations must be positive.")
    if float(tolerance) < 0.0:
        raise ValueError("tolerance must be non-negative.")

    def body(carry, iteration):
        value, done, used = carry
        candidate, objective, residual = step(value, iteration)
        finite = jnp.isfinite(objective) & jnp.isfinite(residual)
        newly_done = finite & (residual <= float(tolerance))
        value = jax.tree_util.tree_map(
            lambda old, new: jnp.where(done, old, new), value, candidate
        )
        used = jnp.where(done, used, iteration + 1)
        return (value, done | newly_done, used), (objective, residual, finite)

    (value, converged, iterations), history = jax.lax.scan(
        body,
        (initial, jnp.asarray(False), jnp.asarray(0, dtype=jnp.int32)),
        jnp.arange(count, dtype=jnp.int32),
    )
    objectives, residuals, finite_steps = history
    return IterationResult(
        value=value,
        objective_history=objectives,
        residual_history=residuals,
        iterations=iterations,
        converged=converged,
        finite=jnp.all(finite_steps),
        method=str(method),
    )


def soft_threshold(value: Array, threshold: Array, /) -> Array:
    """Elementwise proximal map of the L1 norm."""
    magnitude = jnp.abs(value)
    scale = jnp.maximum(magnitude - threshold, 0.0) / jnp.maximum(
        magnitude, jnp.finfo(value.real.dtype).tiny
    )
    return value * scale


def group_soft_threshold(value: Array, threshold: Array, /, *, axis: int = -1) -> Array:
    """Group-lasso proximal map over one declared axis."""
    norm = jnp.linalg.norm(value, axis=axis, keepdims=True)
    scale = jnp.maximum(1.0 - threshold / jnp.maximum(norm, jnp.finfo(float).tiny), 0.0)
    return value * scale


def project_simplex(value: Array, /) -> Array:
    """Euclidean projection onto the probability simplex along the final axis."""
    ordered = jnp.sort(value, axis=-1)[..., ::-1]
    cumulative = jnp.cumsum(ordered, axis=-1) - 1.0
    denominator = jnp.arange(1, value.shape[-1] + 1, dtype=value.dtype)
    active = ordered - cumulative / denominator > 0.0
    count = jnp.maximum(jnp.sum(active, axis=-1), 1)
    threshold = (
        jnp.take_along_axis(
            cumulative,
            (count - 1)[..., None],
            axis=-1,
        )[..., 0]
        / count
    )
    return jnp.maximum(value - threshold[..., None], 0.0)


__all__ = [
    "IterationResult",
    "group_soft_threshold",
    "project_simplex",
    "run_fixed_iterations",
    "soft_threshold",
]
