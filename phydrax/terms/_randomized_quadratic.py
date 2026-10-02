#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import assert_never

import jax.numpy as jnp
from jax import Array

from .._randomized_residual_modes import (
    RandomizedResidualLossMode,
    RealizationSamplingDesign,
)
from ..integration import IntegrationPrecisionPolicy
from ..typing import parse


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
    mode: RandomizedResidualLossMode,
    /,
    *,
    right: Array | None = None,
    precision: IntegrationPrecisionPolicy | None = None,
    sampling_design: RealizationSamplingDesign = "unknown",
    right_sampling_design: RealizationSamplingDesign = "unknown",
) -> Array:
    """Estimate a squared mean only under the declared admissible sampling law."""
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy or None.")
    mode = parse(mode, RandomizedResidualLossMode, "mode")
    sampling_design = parse(sampling_design, RealizationSamplingDesign, "sampling_design")
    right_sampling_design = parse(
        right_sampling_design, RealizationSamplingDesign, "right_sampling_design"
    )
    left = precision_.accumulation(left)
    if left.ndim < 1 + len(event_shape):
        raise ValueError(
            "left must have shape (num_realizations,) + sample_shape + event_shape."
        )
    if event_shape and left.shape[-len(event_shape) :] != event_shape:
        raise ValueError("left trailing dimensions do not match event_shape.")
    count = left.shape[0]
    if count < 1:
        raise ValueError("At least one realization is required.")
    match mode:
        case "plug_in":
            return precision_.decision(
                event_inner(jnp.mean(left, axis=0), event_shape, precision=precision_)
            )
        case "independent_product":
            if right is None:
                raise RuntimeError("Independent-product realizations are unavailable.")
            if right.ndim != left.ndim or right.shape[1:] != left.shape[1:]:
                raise ValueError("Independent groups must share sample and event shapes.")
            if right.shape[0] < 1:
                raise ValueError("At least one realization is required in each group.")
            if sampling_design == "unknown" or right_sampling_design == "unknown":
                raise ValueError(
                    "independent_product requires known-unbiased realization groups; "
                    "declare their sampling_design or use plug_in."
                )
            return precision_.decision(
                cross_inner(
                    jnp.mean(left, axis=0),
                    jnp.mean(precision_.accumulation(right), axis=0),
                    event_shape,
                    precision=precision_,
                )
            )
        case "u_statistic":
            if sampling_design == "exact":
                return precision_.decision(
                    event_inner(jnp.mean(left, axis=0), event_shape, precision=precision_)
                )
            if sampling_design != "iid":
                raise ValueError(
                    "u_statistic requires iid realizations, not finite_population or "
                    "unknown samples; use independent_product with two independently "
                    "keyed known-unbiased groups, or plug_in."
                )
            if count < 2:
                raise ValueError("u_statistic requires at least two iid realizations.")
        case _:
            assert_never(mode)
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
    "cross_inner",
    "event_inner",
    "randomized_squared_mean",
]
