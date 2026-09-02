#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from math import isfinite
from typing import TypeAlias

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._fast_order import (
    _data_axis,
    _restore,
    fast_soft_rank,
    fast_soft_sort,
    fast_weighted_soft_rank,
    fast_weighted_soft_sort,
)
from ._soft import soft_rank, soft_sort


Value = ArrayLike | cx.Field


class HardOrdering(StrictModule):
    """Stable hard ordering with no ordinary derivative at ties."""


class PAVOrdering(StrictModule):
    temperature: float = eqx.field(static=True)

    def __init__(self, temperature: float = 0.5, /):
        self.temperature = _positive_temperature(temperature)


class WeightedPAVOrdering(StrictModule):
    temperature: float = eqx.field(static=True)

    def __init__(self, temperature: float = 0.5, /):
        self.temperature = _positive_temperature(temperature)


class SinkhornOrdering(StrictModule):
    epsilon: float = eqx.field(static=True)

    def __init__(self, epsilon: float = 0.05, /):
        self.epsilon = _positive_temperature(epsilon)


OrderingSurrogate: TypeAlias = PAVOrdering | WeightedPAVOrdering | SinkhornOrdering


class StraightThroughOrdering(StrictModule):
    """Hard forward values with an explicitly identified soft backward estimator."""

    surrogate: OrderingSurrogate

    def __init__(self, surrogate: OrderingSurrogate | None = None, /):
        surrogate_ = PAVOrdering() if surrogate is None else surrogate
        if not isinstance(
            surrogate_, (PAVOrdering, WeightedPAVOrdering, SinkhornOrdering)
        ):
            raise TypeError("Straight-through surrogate must be a soft ordering mode.")
        self.surrogate = surrogate_


OrderingMethod: TypeAlias = (
    HardOrdering
    | PAVOrdering
    | WeightedPAVOrdering
    | SinkhornOrdering
    | StraightThroughOrdering
)


def _positive_temperature(value: float, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError("Ordering temperature/epsilon must be finite and positive.")
    return resolved


def _hard_values(
    values: Value, /, *, axis: int | str, descending: bool
) -> Array | cx.Field:
    data, position, dims = _data_axis(values, axis=axis)
    output = jnp.sort(data, axis=position, descending=descending, stable=True)
    return _restore(jax.lax.stop_gradient(output), dims)


def _hard_ranks(
    values: Value, /, *, axis: int | str, descending: bool
) -> Array | cx.Field:
    data, position, dims = _data_axis(values, axis=axis)
    moved = jnp.moveaxis(data, position, -1)
    permutation = jnp.argsort(moved, axis=-1, stable=True)
    ranks = jnp.argsort(permutation, axis=-1, stable=True)
    if descending:
        ranks = moved.shape[-1] - 1 - ranks
    output = jnp.moveaxis(ranks.astype(data.dtype), -1, position)
    return _restore(jax.lax.stop_gradient(output), dims)


def ordered_values(
    values: Value,
    method: OrderingMethod,
    /,
    *,
    weights: Value | None = None,
    axis: int | str = -1,
    descending: bool = False,
) -> Array | cx.Field:
    """Dispatch value ordering without collapsing hard, PAV, and OT semantics."""
    if isinstance(method, HardOrdering):
        if weights is not None:
            raise ValueError("HardOrdering values do not consume weights.")
        return _hard_values(values, axis=axis, descending=descending)
    if isinstance(method, PAVOrdering):
        if weights is not None:
            raise ValueError("PAVOrdering is unweighted; use WeightedPAVOrdering.")
        return fast_soft_sort(
            values, temperature=method.temperature, axis=axis, descending=descending
        )
    if isinstance(method, WeightedPAVOrdering):
        if weights is None:
            raise ValueError("WeightedPAVOrdering requires strictly positive weights.")
        return fast_weighted_soft_sort(
            values,
            weights,
            temperature=method.temperature,
            axis=axis,
            descending=descending,
        )
    if isinstance(method, SinkhornOrdering):
        return soft_sort(
            values,
            weights=weights,
            epsilon=method.epsilon,
            axis=axis,
            descending=descending,
        )
    if isinstance(method, StraightThroughOrdering):
        return straight_through_sort(
            values,
            method.surrogate,
            weights=weights,
            axis=axis,
            descending=descending,
        )
    raise TypeError("method must be an OrderingMethod.")


def ordered_ranks(
    values: Value,
    method: OrderingMethod,
    /,
    *,
    weights: Value | None = None,
    axis: int | str = -1,
    descending: bool = False,
) -> Array | cx.Field:
    """Dispatch zero-based or mass-rank ordering under one explicit method value."""
    if isinstance(method, HardOrdering):
        if weights is not None:
            raise ValueError("HardOrdering ranks do not consume weights.")
        return _hard_ranks(values, axis=axis, descending=descending)
    if isinstance(method, PAVOrdering):
        if weights is not None:
            raise ValueError("PAVOrdering is unweighted; use WeightedPAVOrdering.")
        return fast_soft_rank(
            values, temperature=method.temperature, axis=axis, descending=descending
        )
    if isinstance(method, WeightedPAVOrdering):
        if weights is None:
            raise ValueError("WeightedPAVOrdering requires strictly positive weights.")
        return fast_weighted_soft_rank(
            values,
            weights,
            temperature=method.temperature,
            axis=axis,
            descending=descending,
        )
    if isinstance(method, SinkhornOrdering):
        return soft_rank(
            values,
            weights=weights,
            epsilon=method.epsilon,
            axis=axis,
            descending=descending,
        )
    if isinstance(method, StraightThroughOrdering):
        return straight_through_rank(
            values,
            method.surrogate,
            weights=weights,
            axis=axis,
            descending=descending,
        )
    raise TypeError("method must be an OrderingMethod.")


def _straight_through(hard, soft):
    if isinstance(hard, cx.Field):
        if not isinstance(soft, cx.Field) or hard.dims != soft.dims:
            raise ValueError("Hard and soft ordering fields must share dimensions.")
        return cx.Field(
            soft.data + jax.lax.stop_gradient(hard.data - soft.data), dims=hard.dims
        )
    if isinstance(soft, cx.Field):
        raise TypeError("Hard and soft ordering representations differ.")
    return soft + jax.lax.stop_gradient(hard - soft)


def straight_through_sort(
    values: Value,
    surrogate: OrderingSurrogate,
    /,
    *,
    weights: Value | None = None,
    axis: int | str = -1,
    descending: bool = False,
):
    """Stable hard-sort forward with a declared PAV or Sinkhorn gradient estimator."""
    hard = _hard_values(values, axis=axis, descending=descending)
    soft = ordered_values(
        values,
        surrogate,
        weights=weights,
        axis=axis,
        descending=descending,
    )
    return _straight_through(hard, soft)


def straight_through_rank(
    values: Value,
    surrogate: OrderingSurrogate,
    /,
    *,
    weights: Value | None = None,
    axis: int | str = -1,
    descending: bool = False,
):
    """Stable hard-rank forward with a declared PAV or Sinkhorn gradient estimator."""
    hard = _hard_ranks(values, axis=axis, descending=descending)
    soft = ordered_ranks(
        values,
        surrogate,
        weights=weights,
        axis=axis,
        descending=descending,
    )
    return _straight_through(hard, soft)


__all__ = [
    "HardOrdering",
    "OrderingMethod",
    "OrderingSurrogate",
    "PAVOrdering",
    "SinkhornOrdering",
    "StraightThroughOrdering",
    "WeightedPAVOrdering",
    "ordered_ranks",
    "ordered_values",
    "straight_through_rank",
    "straight_through_sort",
]
