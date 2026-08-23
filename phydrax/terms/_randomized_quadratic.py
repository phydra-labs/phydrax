#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal, TypeAlias

import jax.numpy as jnp
from jaxtyping import Array

from ..integration import IntegrationPrecisionPolicy


RandomizedQuadraticMode: TypeAlias = Literal[
    "u_statistic",
    "independent_product",
    "plug_in",
]


def event_inner(
    values: Array,
    event_shape: tuple[int, ...],
    /,
    *,
    precision: IntegrationPrecisionPolicy | None = None,
) -> Array:
    """Return the real squared norm over declared trailing event dimensions."""
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy or None.")
    accumulated = precision_.accumulation(values)
    if not event_shape:
        return jnp.real(jnp.conj(accumulated) * accumulated)
    event_size = prod(event_shape)
    flattened = accumulated.reshape(
        accumulated.shape[: -len(event_shape)] + (event_size,)
    )
    return jnp.sum(
        precision_.accumulation(jnp.real(jnp.conj(flattened) * flattened)),
        axis=-1,
    )


def cross_inner(
    left: Array,
    right: Array,
    event_shape: tuple[int, ...],
    /,
    *,
    precision: IntegrationPrecisionPolicy | None = None,
) -> Array:
    """Return the real cross inner product over trailing event dimensions."""
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy or None.")
    left_ = precision_.accumulation(left)
    right_ = precision_.accumulation(right)
    if not event_shape:
        return jnp.real(jnp.conj(left_) * right_)
    event_size = prod(event_shape)
    left_flat = left_.reshape(left_.shape[: -len(event_shape)] + (event_size,))
    right_flat = right_.reshape(right_.shape[: -len(event_shape)] + (event_size,))
    return jnp.sum(
        precision_.accumulation(jnp.real(jnp.conj(left_flat) * right_flat)),
        axis=-1,
    )


def randomized_squared_mean(
    left: Array,
    event_shape: tuple[int, ...],
    mode: RandomizedQuadraticMode,
    /,
    *,
    right: Array | None = None,
    precision: IntegrationPrecisionPolicy | None = None,
) -> Array:
    """Estimate the squared norm of a mean from independent realizations."""
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy or None.")
    left = precision_.accumulation(left)
    if left.ndim < 1 + len(event_shape):
        raise ValueError(
            "left must have shape (num_realizations,) + sample_shape + event_shape."
        )
    if event_shape and left.shape[-len(event_shape) :] != event_shape:
        raise ValueError("left trailing dimensions do not match event_shape.")
    count = int(left.shape[0])
    if count < 2:
        raise ValueError("At least two realizations are required.")
    if mode == "plug_in":
        return precision_.decision(
            event_inner(
                jnp.mean(left, axis=0),
                event_shape,
                precision=precision_,
            )
        )
    if mode == "independent_product":
        if right is None:
            raise RuntimeError("Independent-product realizations are unavailable.")
        if right.shape != left.shape:
            raise ValueError("Independent realization groups must have equal shapes.")
        return precision_.decision(
            cross_inner(
                jnp.mean(left, axis=0),
                jnp.mean(precision_.accumulation(right), axis=0),
                event_shape,
                precision=precision_,
            )
        )
    if mode != "u_statistic":
        raise ValueError(f"Unknown randomized quadratic mode {mode!r}.")
    summed = jnp.sum(left, axis=0)
    total_cross = event_inner(
        summed,
        event_shape,
        precision=precision_,
    ) - jnp.sum(
        event_inner(left, event_shape, precision=precision_),
        axis=0,
    )
    return precision_.decision(total_cross / float(count * (count - 1)))


__all__ = [
    "RandomizedQuadraticMode",
    "cross_inner",
    "event_inner",
    "randomized_squared_mean",
]
