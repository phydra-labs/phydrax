#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def stable_masked_order(values: Array, valid: Array, /) -> Array:
    """Return a stable ascending order with invalid entries placed last."""

    safe = jnp.where(valid, values, jnp.inf)
    return jnp.argsort(safe, axis=-1, stable=True)


def stable_masked_argmin(values: Array, valid: Array, /) -> tuple[Array, Array, Array]:
    """Return stable index, value, and availability along the final axis."""

    order = stable_masked_order(values, valid)
    index = order[..., 0]
    available = jnp.any(valid, axis=-1)
    selected = jnp.take_along_axis(values, index[..., None], axis=-1)[..., 0]
    return (
        jnp.where(available, index, -1),
        jnp.where(available, selected, jnp.nan),
        available,
    )


def stable_first_second(
    values: Array,
    valid: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return stable best index/value, second value, availability, and tie margin."""

    order = stable_masked_order(values, valid)
    count = values.shape[-1]
    best_index = order[..., 0]
    num_valid = jnp.sum(valid, axis=-1)
    available = num_valid > 0
    best = jnp.take_along_axis(values, best_index[..., None], axis=-1)[..., 0]
    if count > 1:
        second_index = order[..., 1]
        second = jnp.take_along_axis(
            values,
            second_index[..., None],
            axis=-1,
        )[..., 0]
    else:
        second = jnp.full_like(best, jnp.inf)
    second_available = num_valid > 1
    margin = jnp.where(second_available, second - best, jnp.inf)
    return (
        jnp.where(available, best_index, -1),
        jnp.where(available, best, jnp.nan),
        jnp.where(second_available, second, jnp.nan),
        available,
        jnp.where(available, margin, jnp.nan),
    )


def relative_gap(absolute_gap: Array, primal: Array, dual: Array, /) -> Array:
    """Return a scale-aware non-negative relative objective gap."""

    scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(primal), jnp.abs(dual)))
    return absolute_gap / scale


__all__: list[str] = []
