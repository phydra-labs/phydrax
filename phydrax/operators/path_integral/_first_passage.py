#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._action import _paths_array
from ._discretization import PathDiscretization
from ._estimate import PathIntegralEstimate


def _inside_mask(paths: ArrayLike, inside: Callable[[Array], ArrayLike], /) -> Array:
    if not callable(inside):
        raise TypeError("inside must be callable as inside(x) -> bool.")
    q = jnp.asarray(paths, dtype=float)
    if q.ndim < 3:
        raise ValueError("paths must have shape (..., num_paths, num_nodes, state_dim).")
    if int(q.shape[-3]) < 1 or int(q.shape[-2]) < 1 or int(q.shape[-1]) < 1:
        raise ValueError("path, node, and state dimensions must be non-empty.")
    q = eqx.error_if(q, ~jnp.all(jnp.isfinite(q)), "paths must be finite.")
    flat = jnp.reshape(q, (-1, int(q.shape[-1])))
    mask = jnp.asarray(jax.vmap(inside)(flat))
    if mask.shape != (flat.shape[0],):
        raise ValueError(
            "inside must return one boolean per state; "
            f"got {mask.shape}, expected {(flat.shape[0],)}."
        )
    if mask.dtype != jnp.bool_:
        raise TypeError("inside must return boolean values.")
    return jnp.reshape(mask, q.shape[:-1])


def first_exit_index(
    paths: ArrayLike,
    inside: Callable[[Array], ArrayLike],
    /,
) -> Array:
    """Return each path's first outside-node index, or ``-1`` if it survives."""
    outside = ~_inside_mask(paths, inside)
    exited = jnp.any(outside, axis=-1)
    first = jnp.argmax(outside, axis=-1)
    return jnp.where(exited, first, -1)


def first_exit_time(
    paths: ArrayLike,
    inside: Callable[[Array], ArrayLike],
    /,
    *,
    slicing: PathDiscretization,
) -> Array:
    """Return discrete first-crossing times, or ``inf`` for surviving paths.

    Crossings are detected only at stored path nodes; no continuous-time crossing or
    interpolation is implied.
    """
    q = _paths_array(paths, slicing)
    index = first_exit_index(q, inside)
    clipped = jnp.maximum(index, 0)
    times = slicing.times[clipped]
    return jnp.where(index >= 0, times, jnp.inf)


def survival_probability(
    paths: ArrayLike,
    inside: Callable[[Array], ArrayLike],
    /,
) -> PathIntegralEstimate:
    """Estimate discrete-time survival probability and Bernoulli standard error."""
    index = first_exit_index(paths, inside)
    count = int(index.shape[-1])
    survived = index < 0
    probability = jnp.mean(survived, axis=-1)
    if count == 1:
        standard_error = jnp.full_like(probability, jnp.nan)
    else:
        standard_error = jnp.sqrt(probability * (1.0 - probability) / float(count - 1))
    return PathIntegralEstimate(
        value=probability,
        standard_error=standard_error,
        effective_sample_size=jnp.full_like(probability, float(count)),
        log_mean_weight=jnp.zeros_like(probability),
        num_paths=count,
    )


__all__ = [
    "first_exit_index",
    "first_exit_time",
    "survival_probability",
]
