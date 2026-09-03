#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._interpolation import (
    BoundsMode,
    cubic_hermite_segment,
    linear_segment,
    local_cubic_slope,
    NearestTiePolicy,
)
from .._strict import StrictModule
from ._sampled import SampledSeries
from ._types import (
    SeriesEvaluation,
    SeriesInterpolation,
    SeriesReconstructionCapabilities,
)


NodeSide = Literal["left", "right"]


def _capabilities(method: SeriesInterpolation, /) -> SeriesReconstructionCapabilities:
    if method == "nearest":
        return SeriesReconstructionCapabilities("node", 0, False, False)
    if method == "previous":
        return SeriesReconstructionCapabilities("node", 0, True, False)
    if method == "linear":
        return SeriesReconstructionCapabilities("node", 1, False, True)
    if method == "cubic_hermite":
        return SeriesReconstructionCapabilities("node", 2, False, True)
    return SeriesReconstructionCapabilities("edge", 0, True, False)


def _expand(value: Array, reference: Array, /) -> Array:
    return value.reshape(value.shape + (1,) * (reference.ndim - value.ndim))


def _broadcast_series_indices(
    value: ArrayLike,
    query_shape: tuple[int, ...],
    count: int,
    /,
) -> Array:
    raw = jnp.asarray(value)
    if not jnp.issubdtype(raw.dtype, jnp.integer):
        raise TypeError("Series indices must be integers.")
    if raw.ndim > len(query_shape):
        raise ValueError("Series indices must broadcast to the query shape.")
    padded_shape = (1,) * (len(query_shape) - raw.ndim) + raw.shape
    if any(
        source not in (1, target)
        for source, target in zip(padded_shape, query_shape, strict=True)
    ):
        raise ValueError("Series indices must broadcast to the query shape.")
    indices = jnp.broadcast_to(raw, query_shape).astype(jnp.int32)
    indices = eqx.error_if(
        indices,
        jnp.any((indices < 0) | (indices >= count)),
        "Series index is out of bounds.",
    )
    return jnp.clip(indices, 0, max(count - 1, 0))


class SampledSeriesReconstruction(StrictModule):
    """Explicit reconstruction of one connected sampled numerical series."""

    series: SampledSeries
    interpolation: SeriesInterpolation = eqx.field(static=True)
    bounds: BoundsMode = eqx.field(static=True)
    nearest_tie_policy: NearestTiePolicy = eqx.field(static=True)
    node_side: NodeSide = eqx.field(static=True)
    snap_tolerance: float = eqx.field(static=True)
    fill_value: Array | None

    def __init__(
        self,
        series: SampledSeries,
        /,
        *,
        interpolation: SeriesInterpolation = "linear",
        bounds: BoundsMode = "error",
        nearest_tie_policy: NearestTiePolicy = "lower",
        node_side: NodeSide = "right",
        snap_tolerance: float = 0.0,
        fill_value: ArrayLike | None = None,
    ):
        if not isinstance(series, SampledSeries):
            raise TypeError("series must be a SampledSeries.")
        if interpolation not in (
            "nearest",
            "previous",
            "linear",
            "cubic_hermite",
            "interval_hold",
        ):
            raise ValueError("Unsupported sampled-series interpolation method.")
        if bounds not in ("clip", "error", "extrapolate", "fill"):
            raise ValueError("bounds must be 'clip', 'error', 'extrapolate', or 'fill'.")
        if nearest_tie_policy not in ("lower", "round_even", "upper"):
            raise ValueError(
                "nearest_tie_policy must be 'lower', 'round_even', or 'upper'."
            )
        if node_side not in ("left", "right"):
            raise ValueError("node_side must be 'left' or 'right'.")
        tolerance = float(snap_tolerance)
        if tolerance < 0.0:
            raise ValueError("snap_tolerance must be non-negative.")

        capabilities = _capabilities(interpolation)
        if series.alignment != capabilities.alignment:
            raise ValueError(
                f"interpolation={interpolation!r} requires a "
                f"{capabilities.alignment}-aligned series."
            )
        if series.support.coordinate_kind == "discrete" and interpolation in (
            "linear",
            "cubic_hermite",
        ):
            raise ValueError("Continuous reconstruction requires continuous coordinates.")
        if bounds == "extrapolate" and interpolation in (
            "nearest",
            "previous",
            "interval_hold",
        ):
            raise ValueError(
                f"interpolation={interpolation!r} does not support extrapolation."
            )

        node_valid = series.support.node_valid
        lengths = jnp.sum(node_valid, axis=-1, dtype=jnp.int32)
        node_expected = jnp.arange(series.support.capacity) < lengths[..., None]
        edge_expected = (
            jnp.arange(max(series.support.capacity - 1, 0))
            < jnp.maximum(lengths - 1, 0)[..., None]
        )
        structure_valid = jnp.all(lengths > 0)
        structure_valid = structure_valid & jnp.all(node_valid == node_expected)
        structure_valid = structure_valid & jnp.all(
            series.support.edge_valid == edge_expected
        )
        if series.alignment == "node":
            structure_valid = structure_valid & jnp.all(
                series.sample_valid == node_expected
            )
        else:
            structure_valid = structure_valid & jnp.all(lengths >= 2)
            structure_valid = structure_valid & jnp.all(
                series.sample_valid == edge_expected
            )
        coordinates = eqx.error_if(
            series.support.coordinates,
            ~structure_valid,
            "Sampled-series reconstruction requires one connected valid prefix per series.",
        )
        series_ = eqx.tree_at(
            lambda candidate: candidate.support.coordinates,
            series,
            coordinates,
        )

        self.series = series_
        self.interpolation = interpolation
        self.bounds = bounds
        self.nearest_tie_policy = nearest_tie_policy
        self.node_side = node_side
        self.snap_tolerance = tolerance
        self.fill_value = None if fill_value is None else jnp.asarray(fill_value)

    @property
    def capabilities(self) -> SeriesReconstructionCapabilities:
        return _capabilities(self.interpolation)

    def _geometry(
        self,
        query: ArrayLike,
        series_indices: ArrayLike,
        /,
    ) -> tuple[Any, ...]:
        coordinates = self.series.support.broadcast_coordinates()
        dtype = jnp.result_type(coordinates.dtype, jnp.asarray(query).dtype, float)
        query_ = jnp.asarray(query, dtype=dtype)
        query_ = eqx.error_if(
            query_, jnp.any(~jnp.isfinite(query_)), "Series queries must be finite."
        )
        query_shape = query_.shape
        indices = _broadcast_series_indices(
            series_indices,
            query_shape,
            self.series.support.num_series,
        )
        capacity = self.series.support.capacity
        coordinate_rows = coordinates.reshape(
            (self.series.support.num_series, capacity)
        ).astype(dtype)
        valid_rows = self.series.support.node_valid.reshape(
            (self.series.support.num_series, capacity)
        )
        flat_indices = indices.reshape((-1,))
        selected_coordinates = coordinate_rows[flat_indices]
        selected_valid = valid_rows[flat_indices]
        lengths = jnp.sum(selected_valid, axis=-1, dtype=jnp.int32)
        safe_coordinates = jnp.where(selected_valid, selected_coordinates, jnp.inf)
        query_flat = query_.reshape((-1,))
        first = selected_coordinates[:, 0]
        last = selected_coordinates[
            jnp.arange(query_flat.size), jnp.maximum(lengths - 1, 0)
        ]
        outside = (query_flat < first) | (query_flat > last)
        if self.bounds == "error":
            query_flat = eqx.error_if(
                query_flat,
                jnp.any(outside),
                "Series query is outside the connected coordinate support.",
            )
        query_eval = (
            jnp.clip(query_flat, first, last)
            if self.bounds in ("clip", "fill")
            else query_flat
        )
        support = ~outside if self.bounds == "fill" else jnp.ones_like(outside)
        upper_raw = jax.vmap(
            lambda row, value: jnp.searchsorted(row, value, side="right")
        )(safe_coordinates, query_eval)
        lower = jnp.clip(upper_raw - 1, 0, jnp.maximum(lengths - 2, 0)).astype(jnp.int32)
        upper = jnp.minimum(lower + 1, lengths - 1).astype(jnp.int32)
        x0 = selected_coordinates[jnp.arange(query_flat.size), lower]
        x1 = selected_coordinates[jnp.arange(query_flat.size), upper]
        width = jnp.where(lengths > 1, x1 - x0, 1.0)
        fraction = jnp.where(lengths > 1, (query_eval - x0) / width, 0.0)
        return (
            indices,
            selected_coordinates,
            lengths,
            lower,
            upper,
            fraction,
            query_eval,
            width,
            support,
            outside,
            query_shape,
        )

    def _selected_value_rows(
        self, indices: Array, query_shape: tuple[int, ...], /
    ) -> Any:
        count = self.series.sample_shape[-1]
        flat_indices = indices.reshape((-1,))

        def select(value: Array) -> Array:
            event_shape = value.shape[len(self.series.sample_shape) :]
            rows = value.reshape((self.series.support.num_series, count) + event_shape)
            selected = rows[flat_indices]
            return selected.reshape(query_shape + (count,) + event_shape)

        return jax.tree_util.tree_map(select, self.series.values)

    def evaluate(
        self,
        query: ArrayLike,
        series_indices: ArrayLike = 0,
        /,
        *,
        derivative_order: int = 0,
    ) -> SeriesEvaluation:
        """Evaluate values or an explicit coordinate derivative at query points."""
        order = int(derivative_order)
        if order < 0 or order > self.capabilities.maximum_explicit_derivative_order:
            raise ValueError(
                f"interpolation={self.interpolation!r} supports derivative orders "
                f"0 through {self.capabilities.maximum_explicit_derivative_order}."
            )
        (
            indices,
            coordinates,
            lengths,
            lower,
            upper,
            fraction,
            query_eval,
            width,
            support,
            outside,
            query_shape,
        ) = self._geometry(query, series_indices)
        rows_tree = self._selected_value_rows(indices, query_shape)
        query_count = int(query_eval.size)
        lower_flat = lower.reshape((-1,))
        upper_flat = upper.reshape((-1,))
        fraction_flat = fraction.reshape((-1,))
        width_flat = width.reshape((-1,))

        def node_value(rows: Array) -> Array:
            flat_rows = rows.reshape((query_count,) + rows.shape[len(query_shape) :])
            event_shape = flat_rows.shape[2:]
            y0 = flat_rows[jnp.arange(query_count), lower_flat]
            y1 = flat_rows[jnp.arange(query_count), upper_flat]
            fraction_value = _expand(fraction_flat, y0)
            width_value = _expand(width_flat, y0)
            if self.interpolation == "linear":
                result = linear_segment(
                    y0,
                    y1,
                    fraction_value,
                    width_value,
                    derivative_order=order,
                )
            else:
                previous = jnp.maximum(lower_flat - 1, 0)
                following = jnp.minimum(upper_flat + 1, lengths - 1)
                y_previous = flat_rows[jnp.arange(query_count), previous]
                y_following = flat_rows[jnp.arange(query_count), following]
                selected_coordinates = coordinates
                x_previous = selected_coordinates[jnp.arange(query_count), previous]
                x0 = selected_coordinates[jnp.arange(query_count), lower_flat]
                x1 = selected_coordinates[jnp.arange(query_count), upper_flat]
                x_following = selected_coordinates[jnp.arange(query_count), following]
                m0 = local_cubic_slope(
                    y_previous,
                    y0,
                    y1,
                    previous_width=_expand(x0 - x_previous, y0),
                    next_width=width_value,
                    has_previous=_expand(lower_flat > 0, y0),
                    has_next=_expand(lengths > 1, y0),
                )
                m1 = local_cubic_slope(
                    y0,
                    y1,
                    y_following,
                    previous_width=width_value,
                    next_width=_expand(x_following - x1, y0),
                    has_previous=_expand(lengths > 1, y0),
                    has_next=_expand(upper_flat < lengths - 1, y0),
                )
                result = cubic_hermite_segment(
                    y0,
                    y1,
                    m0,
                    m1,
                    fraction_value,
                    width_value,
                    derivative_order=order,
                )
            if order == 0 and self.snap_tolerance > 0.0:
                valid_nodes = (
                    jnp.arange(self.series.support.capacity)[None, :] < lengths[:, None]
                )
                node_distance = jnp.where(
                    valid_nodes,
                    jnp.abs(coordinates - query_eval[:, None]),
                    jnp.inf,
                )
                nearest = jnp.argmin(node_distance, axis=-1).astype(jnp.int32)
                snapped = flat_rows[jnp.arange(query_count), nearest]
                on_node = jnp.min(node_distance, axis=-1) <= self.snap_tolerance
                result = jnp.where(_expand(on_node, result), snapped, result)
            return result.reshape(query_shape + event_shape)

        def discrete_value(rows: Array) -> Array:
            flat_rows = rows.reshape((query_count,) + rows.shape[len(query_shape) :])
            event_shape = flat_rows.shape[2:]
            if self.interpolation == "nearest":
                lower_node = jnp.clip(
                    jax.vmap(
                        lambda row, value: jnp.searchsorted(row, value, side="right")
                    )(
                        jnp.where(
                            jnp.arange(self.series.support.capacity)[None, :]
                            < lengths[:, None],
                            coordinates,
                            jnp.inf,
                        ),
                        query_eval,
                    )
                    - 1,
                    0,
                    lengths - 1,
                ).astype(jnp.int32)
                upper_node = jnp.minimum(lower_node + 1, lengths - 1)
                lower_coordinate = coordinates[jnp.arange(query_count), lower_node]
                upper_coordinate = coordinates[jnp.arange(query_count), upper_node]
                lower_distance = jnp.abs(query_eval - lower_coordinate)
                upper_distance = jnp.abs(upper_coordinate - query_eval)
                if self.nearest_tie_policy == "lower":
                    use_upper = upper_distance < lower_distance
                elif self.nearest_tie_policy == "upper":
                    use_upper = upper_distance <= lower_distance
                else:
                    tied = upper_distance == lower_distance
                    even_upper = (upper_node % 2) == 0
                    use_upper = (upper_distance < lower_distance) | (tied & even_upper)
                selected = jnp.where(use_upper, upper_node, lower_node)
            elif self.interpolation == "previous":
                selected = jnp.clip(
                    jax.vmap(
                        lambda row, value: jnp.searchsorted(row, value, side="right")
                    )(
                        jnp.where(
                            jnp.arange(self.series.support.capacity)[None, :]
                            < lengths[:, None],
                            coordinates,
                            jnp.inf,
                        ),
                        query_eval,
                    )
                    - 1,
                    0,
                    lengths - 1,
                ).astype(jnp.int32)
            else:
                side = "left" if self.node_side == "left" else "right"
                selected = jnp.clip(
                    jax.vmap(lambda row, value: jnp.searchsorted(row, value, side=side))(
                        jnp.where(
                            jnp.arange(self.series.support.capacity)[None, :]
                            < lengths[:, None],
                            coordinates,
                            jnp.inf,
                        ),
                        query_eval,
                    )
                    - 1,
                    0,
                    lengths - 2,
                ).astype(jnp.int32)
            result = flat_rows[jnp.arange(query_count), selected]
            return result.reshape(query_shape + event_shape)

        if self.interpolation in ("linear", "cubic_hermite"):
            values = jax.tree_util.tree_map(node_value, rows_tree)
        else:
            values = jax.tree_util.tree_map(discrete_value, rows_tree)
        if self.bounds == "fill" and self.fill_value is not None:
            values = jax.tree_util.tree_map(
                lambda value: jnp.where(
                    _expand(outside.reshape(query_shape), value),
                    self.fill_value.astype(value.dtype),
                    value,
                ),
                values,
            )
        return SeriesEvaluation(values, support.reshape(query_shape))

    def breakpoints(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        series_index: ArrayLike = 0,
        /,
    ) -> tuple[Array, Array]:
        """Return fixed-capacity interior coordinates and their validity mask."""
        coordinates = self.series.support.coordinates_for(series_index)
        dtype = jnp.result_type(
            coordinates.dtype, jnp.asarray(lower).dtype, jnp.asarray(upper).dtype, float
        )
        lower_ = jnp.asarray(lower, dtype=dtype)
        upper_ = jnp.asarray(upper, dtype=dtype)
        if lower_.shape != () or upper_.shape != ():
            raise ValueError("Breakpoint bounds must be scalar.")
        bounds = jnp.stack((lower_, upper_))
        bounds = eqx.error_if(
            bounds,
            jnp.any(~jnp.isfinite(bounds)) | (bounds[1] < bounds[0]),
            "Breakpoint bounds must be finite and ordered.",
        )
        valid = self.series.support.node_valid_for(series_index)
        coordinates_eval = coordinates.astype(dtype)
        mask = valid & (coordinates_eval > bounds[0]) & (coordinates_eval < bounds[1])
        return coordinates, mask


__all__ = ["SampledSeriesReconstruction"]
