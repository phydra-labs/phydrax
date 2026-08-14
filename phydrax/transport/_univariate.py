#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def wasserstein_distance_1d(
    source: ArrayLike,
    target: ArrayLike,
    /,
    *,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    p: float = 2.0,
) -> Array:
    """Return exact weighted empirical one-dimensional Wasserstein distance."""
    source_values = _values_1d(source, name="source")
    target_values = _values_1d(target, name="target")
    source_probabilities = _probabilities(
        source_weights,
        source_values.shape[0],
        name="source_weights",
        dtype=source_values.dtype,
    )
    target_probabilities = _probabilities(
        target_weights,
        target_values.shape[0],
        name="target_weights",
        dtype=target_values.dtype,
    )
    exponent = _validate_p(p)
    cost = _wasserstein_cost_1d(
        source_values,
        target_values,
        source_probabilities,
        target_probabilities,
        p=exponent,
    )
    return cost ** (1.0 / exponent)


def _wasserstein_cost_1d(
    source: Array,
    target: Array,
    source_probabilities: Array,
    target_probabilities: Array,
    /,
    *,
    p: float,
) -> Array:
    source_order = jnp.argsort(source)
    target_order = jnp.argsort(target)
    source_sorted = source[source_order]
    target_sorted = target[target_order]
    source_weights = source_probabilities[source_order]
    target_weights = target_probabilities[target_order]
    source_cumulative = jnp.cumsum(source_weights).at[-1].set(1.0)
    target_cumulative = jnp.cumsum(target_weights).at[-1].set(1.0)
    breakpoints = jnp.sort(
        jnp.concatenate(
            [
                jnp.zeros((1,), dtype=source.dtype),
                source_cumulative,
                target_cumulative,
            ]
        )
    )
    widths = jnp.diff(breakpoints)
    midpoints = 0.5 * (breakpoints[:-1] + breakpoints[1:])
    source_indices = jnp.clip(
        jnp.searchsorted(source_cumulative, midpoints, side="left"),
        0,
        source.shape[0] - 1,
    )
    target_indices = jnp.clip(
        jnp.searchsorted(target_cumulative, midpoints, side="left"),
        0,
        target.shape[0] - 1,
    )
    displacement = jnp.abs(source_sorted[source_indices] - target_sorted[target_indices])
    return jnp.sum(widths * displacement**p)


def _values_1d(values: ArrayLike, /, *, name: str) -> Array:
    result = jnp.asarray(values, dtype=float)
    if result.ndim != 1 or result.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty rank-one array.")
    return eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result)),
        f"{name} must contain only finite values.",
    )


def _probabilities(
    weights: ArrayLike | None,
    count: int,
    /,
    *,
    name: str,
    dtype,
) -> Array:
    if weights is None:
        return jnp.full((count,), 1.0 / float(count), dtype=dtype)
    values = jnp.asarray(weights, dtype=dtype)
    if values.shape != (count,):
        raise ValueError(f"{name} must have shape {(count,)}.")
    values = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)) | jnp.any(values < 0.0),
        f"{name} must contain only finite nonnegative values.",
    )
    total = jnp.sum(values)
    total = eqx.error_if(
        total,
        ~jnp.isfinite(total) | (total <= 0.0),
        f"{name} must contain positive mass.",
    )
    return values / total


def _validate_p(p: float, /) -> float:
    exponent = float(p)
    if not math.isfinite(exponent) or exponent < 1.0:
        raise ValueError("p must be finite and at least one.")
    return exponent


__all__ = ["wasserstein_distance_1d"]
