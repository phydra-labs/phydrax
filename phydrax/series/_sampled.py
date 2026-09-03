#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._support import _checked_series_index, _identifier, SeriesSupport
from ._types import SeriesAlignment


def _numeric_array(value: Any, /) -> Array:
    array = jnp.asarray(value)
    if not (
        jnp.issubdtype(array.dtype, jnp.bool_)
        or jnp.issubdtype(array.dtype, jnp.integer)
        or jnp.issubdtype(array.dtype, jnp.inexact)
    ):
        raise TypeError("SampledSeries values must be numerical arrays.")
    return array


def _expanded_sample_mask(mask: Array, value: Array, prefix_rank: int, /) -> Array:
    return mask.reshape(mask.shape + (1,) * (value.ndim - prefix_rank))


def _component_masks(
    value_valid: Any | None,
    values: Any,
    sample_shape: tuple[int, ...],
    /,
) -> Any | None:
    if value_valid is None:
        return None
    if jax.tree_util.tree_structure(value_valid) != jax.tree_util.tree_structure(values):
        raise ValueError("SampledSeries value_valid must match the values PyTree.")

    def prepare(mask_value: ArrayLike, value: Array) -> Array:
        mask = jnp.asarray(mask_value, dtype=bool)
        if mask.shape == sample_shape:
            return jnp.broadcast_to(
                mask.reshape(mask.shape + (1,) * (value.ndim - len(sample_shape))),
                value.shape,
            )
        if mask.shape != value.shape:
            raise ValueError(
                "SampledSeries value-valid leaves must match their value leaf or "
                f"sample shape {sample_shape}; got {mask.shape} for {value.shape}."
            )
        return mask

    return jax.tree_util.tree_map(prepare, value_valid, values)


class SampledSeries(StrictModule):
    """A numerical PyTree aligned with nodes or edges of a `SeriesSupport`."""

    support: SeriesSupport
    values: Any
    value_valid: Any | None
    alignment: SeriesAlignment = eqx.field(static=True)
    series_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: SeriesSupport,
        values: Any,
        /,
        *,
        alignment: SeriesAlignment = "node",
        value_valid: Any | None = None,
        series_id: str,
    ):
        if not isinstance(support, SeriesSupport):
            raise TypeError("support must be a SeriesSupport.")
        if alignment not in ("node", "edge"):
            raise ValueError("alignment must be 'node' or 'edge'.")
        values_ = jax.tree_util.tree_map(_numeric_array, values)
        leaves = jax.tree_util.tree_leaves(values_)
        if not leaves:
            raise ValueError("SampledSeries values must contain at least one array leaf.")

        count = support.capacity if alignment == "node" else support.capacity - 1
        sample_shape = support.series_shape + (count,)
        for leaf in leaves:
            if (
                leaf.ndim < len(sample_shape)
                or leaf.shape[: len(sample_shape)] != sample_shape
            ):
                raise ValueError(
                    "SampledSeries value leaves must begin with sample shape "
                    f"{sample_shape}; got {leaf.shape}."
                )
        masks = _component_masks(value_valid, values_, sample_shape)
        base_valid = support.node_valid if alignment == "node" else support.edge_valid
        bad = jnp.asarray(False)
        value_masks = (
            jax.tree_util.tree_map(
                lambda value: jnp.ones(value.shape, dtype=bool), values_
            )
            if masks is None
            else masks
        )
        for leaf, mask in zip(
            leaves,
            jax.tree_util.tree_leaves(value_masks),
            strict=True,
        ):
            expanded = _expanded_sample_mask(base_valid, leaf, len(sample_shape))
            bad = bad | jnp.any(expanded & mask & ~jnp.isfinite(leaf))
        first = leaves[0]
        first = eqx.error_if(
            first,
            bad,
            "SampledSeries active values must be finite.",
        )
        values_ = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(values_),
            (first, *leaves[1:]),
        )

        self.support = support
        self.values = values_
        self.value_valid = masks
        self.alignment = alignment
        self.series_id = _identifier(series_id, "series_id")

    @property
    def sample_shape(self) -> tuple[int, ...]:
        count = (
            self.support.capacity
            if self.alignment == "node"
            else self.support.capacity - 1
        )
        return self.support.series_shape + (count,)

    @property
    def sample_valid(self) -> Array:
        """Return samples whose complete numerical payload is valid."""
        base = (
            self.support.node_valid
            if self.alignment == "node"
            else self.support.edge_valid
        )
        if self.value_valid is None:
            return base
        complete = base
        prefix_rank = len(self.sample_shape)
        for mask in jax.tree_util.tree_leaves(self.value_valid):
            event_axes = tuple(range(prefix_rank, mask.ndim))
            leaf_complete = jnp.all(mask, axis=event_axes) if event_axes else mask
            complete = complete & leaf_complete
        return complete

    def values_for(self, series_index: ArrayLike = 0, /) -> Any:
        """Return one physical series selected by its flat leading-series index."""
        index = _checked_series_index(series_index, self.support.num_series)
        count = self.sample_shape[-1]

        def select(value: Array) -> Array:
            event_shape = value.shape[len(self.sample_shape) :]
            rows = value.reshape((self.support.num_series, count) + event_shape)
            return rows[index]

        return jax.tree_util.tree_map(select, self.values)

    def valid_for(self, series_index: ArrayLike = 0, /) -> Array:
        """Return complete sample validity for one flat physical-series index."""
        index = _checked_series_index(series_index, self.support.num_series)
        count = self.sample_shape[-1]
        return self.sample_valid.reshape((self.support.num_series, count))[index]


__all__ = ["SampledSeries"]
