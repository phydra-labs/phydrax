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
from ._sampled import SampledSeries


class SeriesPairView(StrictModule):
    """Lazy source/target pairs from one node-aligned sampled series."""

    series: SampledSeries
    source_indices: Array
    target_indices: Array

    def __init__(
        self,
        series: SampledSeries,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        /,
    ):
        if not isinstance(series, SampledSeries):
            raise TypeError("series must be a SampledSeries.")
        if series.alignment != "node":
            raise ValueError("SeriesPairView requires a node-aligned series.")
        source = jnp.asarray(source_indices)
        target = jnp.asarray(target_indices)
        if source.ndim != 1 or target.shape != source.shape:
            raise ValueError("Series pair indices must be equal-shape rank-one arrays.")
        if not jnp.issubdtype(source.dtype, jnp.integer) or not jnp.issubdtype(
            target.dtype, jnp.integer
        ):
            raise TypeError("Series pair indices must be integers.")
        source = source.astype(jnp.int32)
        target = target.astype(jnp.int32)
        invalid = (
            (source < 0)
            | (target < 0)
            | (source >= series.support.capacity)
            | (target >= series.support.capacity)
            | (source >= target)
        )
        source = eqx.error_if(
            source,
            jnp.any(invalid),
            "Series pair indices must satisfy 0 <= source < target < capacity.",
        )
        self.series = series
        self.source_indices = source
        self.target_indices = target

    @classmethod
    def from_lag(cls, series: SampledSeries, lag: int = 1, /) -> SeriesPairView:
        """Construct every index pair separated by one positive fixed lag."""
        lag_ = int(lag)
        capacity = series.support.capacity
        if lag_ <= 0 or lag_ >= capacity:
            raise ValueError("lag must satisfy 1 <= lag < series capacity.")
        source = jnp.arange(capacity - lag_, dtype=jnp.int32)
        return cls(series, source, source + lag_)

    @property
    def pair_count(self) -> int:
        return int(self.source_indices.shape[0])

    @property
    def source_coordinates(self) -> Array:
        coordinates = self.series.support.broadcast_coordinates()
        return jnp.take(coordinates, self.source_indices, axis=-1)

    @property
    def target_coordinates(self) -> Array:
        coordinates = self.series.support.broadcast_coordinates()
        return jnp.take(coordinates, self.target_indices, axis=-1)

    @property
    def coordinate_delta(self) -> Array:
        return self.target_coordinates - self.source_coordinates

    @property
    def source_values(self) -> Any:
        axis = len(self.series.support.series_shape)
        return jax.tree_util.tree_map(
            lambda value: jnp.take(value, self.source_indices, axis=axis),
            self.series.values,
        )

    @property
    def target_values(self) -> Any:
        axis = len(self.series.support.series_shape)
        return jax.tree_util.tree_map(
            lambda value: jnp.take(value, self.target_indices, axis=axis),
            self.series.values,
        )

    @property
    def valid(self) -> Array:
        sample_valid = self.series.sample_valid
        source_valid = jnp.take(sample_valid, self.source_indices, axis=-1)
        target_valid = jnp.take(sample_valid, self.target_indices, axis=-1)
        effective_edges = (
            self.series.support.edge_valid
            & sample_valid[..., :-1]
            & sample_valid[..., 1:]
        )
        invalid_edges = (~effective_edges).astype(jnp.int32)
        prefix = jnp.concatenate(
            (
                jnp.zeros(invalid_edges.shape[:-1] + (1,), dtype=jnp.int32),
                jnp.cumsum(invalid_edges, axis=-1),
            ),
            axis=-1,
        )
        source_prefix = jnp.take(prefix, self.source_indices, axis=-1)
        target_prefix = jnp.take(prefix, self.target_indices, axis=-1)
        return source_valid & target_valid & (source_prefix == target_prefix)


__all__ = ["SeriesPairView"]
