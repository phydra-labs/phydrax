#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._types import CoordinateKind


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    identifier = value.strip()
    return identifier


def _series_shape(
    value: Sequence[int] | None, inferred: tuple[int, ...], /
) -> tuple[int, ...]:
    shape = inferred if value is None else tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("SeriesSupport series_shape entries must be positive.")
    return shape


def _series_axes(
    value: Sequence[str] | None, shape: tuple[int, ...], /
) -> tuple[str, ...]:
    axes = (
        tuple(f"series_{index}" for index in range(len(shape)))
        if value is None
        else tuple(str(axis).strip() for axis in value)
    )
    if len(axes) != len(shape):
        raise ValueError("SeriesSupport series_axes must match series_shape rank.")
    if any(not axis for axis in axes) or len(set(axes)) != len(axes):
        raise ValueError("SeriesSupport series_axes must be unique non-empty strings.")
    return axes


def _boolean_mask(
    value: ArrayLike | None,
    *,
    default: Array,
    full_shape: tuple[int, ...],
    shared_shape: tuple[int, ...],
    name: str,
) -> Array:
    if value is None:
        return default
    mask = jnp.asarray(value, dtype=bool)
    if mask.shape == shared_shape:
        return jnp.broadcast_to(mask, full_shape)
    if mask.shape != full_shape:
        raise ValueError(
            f"SeriesSupport {name} must have shape {shared_shape} or {full_shape}; "
            f"got {mask.shape}."
        )
    return mask


class SeriesSupport(StrictModule):
    """Masked ordered scalar coordinates shared by one or more sampled series.

    Active edges define connectivity. Coordinates must increase only across active
    edges, so disconnected episodes may restart their coordinate without losing a
    static padded representation.
    """

    coordinates: Array
    node_valid: Array
    edge_valid: Array
    series_shape: tuple[int, ...] = eqx.field(static=True)
    series_axes: tuple[str, ...] = eqx.field(static=True)
    coordinate_name: str = eqx.field(static=True)
    coordinate_kind: CoordinateKind = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)
    shared_coordinates: bool = eqx.field(static=True)
    capacity: int = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        /,
        *,
        node_valid: ArrayLike | None = None,
        edge_valid: ArrayLike | None = None,
        series_shape: Sequence[int] | None = None,
        series_axes: Sequence[str] | None = None,
        coordinate_name: str = "coordinate",
        coordinate_kind: CoordinateKind = "continuous",
        coordinate_id: str = "coordinate",
    ):
        coordinates_ = jnp.asarray(coordinates)
        if coordinates_.ndim < 1 or int(coordinates_.shape[-1]) < 1:
            raise ValueError(
                "SeriesSupport coordinates must have a non-empty trailing coordinate axis."
            )
        if jnp.issubdtype(coordinates_.dtype, jnp.bool_) or jnp.issubdtype(
            coordinates_.dtype, jnp.complexfloating
        ):
            raise TypeError("SeriesSupport coordinates must be real numerical values.")
        if not (
            jnp.issubdtype(coordinates_.dtype, jnp.integer)
            or jnp.issubdtype(coordinates_.dtype, jnp.floating)
        ):
            raise TypeError("SeriesSupport coordinates must be real numerical values.")
        if coordinate_kind not in ("continuous", "discrete"):
            raise ValueError("coordinate_kind must be 'continuous' or 'discrete'.")

        capacity = int(coordinates_.shape[-1])
        inferred_shape = () if coordinates_.ndim == 1 else tuple(coordinates_.shape[:-1])
        shape = _series_shape(series_shape, inferred_shape)
        full_node_shape = shape + (capacity,)
        shared = coordinates_.ndim == 1
        if not shared and coordinates_.shape != full_node_shape:
            raise ValueError(
                "SeriesSupport per-series coordinates must have shape "
                f"{full_node_shape}; got {coordinates_.shape}."
            )

        nodes = _boolean_mask(
            node_valid,
            default=jnp.ones(full_node_shape, dtype=bool),
            full_shape=full_node_shape,
            shared_shape=(capacity,),
            name="node_valid",
        )
        full_edge_shape = shape + (capacity - 1,)
        endpoint_valid = nodes[..., :-1] & nodes[..., 1:]
        edges = _boolean_mask(
            edge_valid,
            default=endpoint_valid,
            full_shape=full_edge_shape,
            shared_shape=(capacity - 1,),
            name="edge_valid",
        )
        coordinates_full = (
            jnp.broadcast_to(coordinates_, full_node_shape) if shared else coordinates_
        )
        coordinates_ = eqx.error_if(
            coordinates_,
            jnp.any(nodes & ~jnp.isfinite(coordinates_full)),
            "SeriesSupport valid coordinates must be finite.",
        )
        edges = eqx.error_if(
            edges,
            jnp.any(edges & ~endpoint_valid),
            "SeriesSupport active edges must connect valid nodes.",
        )
        increasing = coordinates_full[..., 1:] > coordinates_full[..., :-1]
        edges = eqx.error_if(
            edges,
            jnp.any(edges & ~increasing),
            "SeriesSupport coordinates must increase strictly across active edges.",
        )

        self.coordinates = coordinates_
        self.node_valid = nodes
        self.edge_valid = edges
        self.series_shape = shape
        self.series_axes = _series_axes(series_axes, shape)
        self.coordinate_name = _identifier(coordinate_name, "coordinate_name")
        self.coordinate_kind = coordinate_kind
        self.coordinate_id = _identifier(coordinate_id, "coordinate_id")
        self.shared_coordinates = shared
        self.capacity = capacity

    @property
    def num_series(self) -> int:
        count = 1
        for size in self.series_shape:
            count *= size
        return count

    def broadcast_coordinates(self) -> Array:
        """Return coordinates with explicit leading series axes."""
        if self.shared_coordinates:
            return jnp.broadcast_to(
                self.coordinates, self.series_shape + (self.capacity,)
            )
        return self.coordinates

    def coordinates_for(self, series_index: ArrayLike = 0, /) -> Array:
        """Return one coordinate row selected by a flat physical-series index."""
        index = _checked_series_index(series_index, self.num_series)
        rows = self.broadcast_coordinates().reshape((self.num_series, self.capacity))
        return rows[index]

    def node_valid_for(self, series_index: ArrayLike = 0, /) -> Array:
        """Return one node-validity row selected by a flat physical-series index."""
        index = _checked_series_index(series_index, self.num_series)
        return self.node_valid.reshape((self.num_series, self.capacity))[index]

    def edge_valid_for(self, series_index: ArrayLike = 0, /) -> Array:
        """Return one edge-validity row selected by a flat physical-series index."""
        index = _checked_series_index(series_index, self.num_series)
        return self.edge_valid.reshape((self.num_series, self.capacity - 1))[index]

    def connected_prefix_valid(self) -> Array:
        """Return whether every series is one nonempty connected valid prefix."""
        lengths = jnp.sum(self.node_valid, axis=-1, dtype=jnp.int32)
        node_expected = jnp.arange(self.capacity) < lengths[..., None]
        edge_expected = (
            jnp.arange(max(self.capacity - 1, 0)) < jnp.maximum(lengths - 1, 0)[..., None]
        )
        return (
            jnp.all(lengths > 0)
            & jnp.all(self.node_valid == node_expected)
            & jnp.all(self.edge_valid == edge_expected)
        )


def _checked_series_index(value: ArrayLike, count: int, /) -> Array:
    raw = jnp.asarray(value)
    if raw.shape != ():
        raise ValueError("Series index must be scalar.")
    if not jnp.issubdtype(raw.dtype, jnp.integer):
        raise TypeError("Series index must be an integer.")
    index = raw.astype(jnp.int32)
    index = eqx.error_if(
        index,
        (index < 0) | (index >= count),
        "Series index is out of bounds.",
    )
    return jnp.clip(index, 0, max(count - 1, 0))


__all__ = ["SeriesSupport"]
