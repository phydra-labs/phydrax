#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._stencil import apply_gather_stencil, GatherStencil
from ._types import (
    BoundsMode,
    InterpolationCapabilities,
    InterpolationResult,
    MaskMode,
    NearestTiePolicy,
)


NEAREST_CAPABILITIES = InterpolationCapabilities(
    partition_of_unity=True,
    nonnegative_value_weights=True,
    local_support=True,
    mask_renormalizable=True,
    tensor_product_composable=True,
    maximum_explicit_derivative_order=0,
)
LINEAR_CAPABILITIES = InterpolationCapabilities(
    partition_of_unity=True,
    nonnegative_value_weights=True,
    local_support=True,
    mask_renormalizable=True,
    tensor_product_composable=True,
    maximum_explicit_derivative_order=1,
)
CUBIC_HERMITE_CAPABILITIES = InterpolationCapabilities(
    partition_of_unity=True,
    nonnegative_value_weights=False,
    local_support=True,
    mask_renormalizable=False,
    tensor_product_composable=True,
    maximum_explicit_derivative_order=2,
)


def _nodes_and_query(nodes: ArrayLike, query: ArrayLike, /) -> tuple[Array, Array]:
    nodes_raw = jnp.asarray(nodes)
    query_raw = jnp.asarray(query)
    if jnp.issubdtype(nodes_raw.dtype, jnp.complexfloating) or jnp.issubdtype(
        query_raw.dtype,
        jnp.complexfloating,
    ):
        raise TypeError("Piecewise interpolation coordinates must be real-valued.")
    dtype = jnp.result_type(nodes_raw, query_raw, float)
    nodes_ = nodes_raw.astype(dtype)
    query_ = query_raw.astype(dtype)
    if nodes_.ndim != 1 or int(nodes_.shape[0]) <= 0:
        raise ValueError(
            "Piecewise interpolation nodes must be a non-empty rank-one array."
        )
    spacing = jnp.diff(nodes_)
    nodes_ = eqx.error_if(
        nodes_,
        jnp.any(~jnp.isfinite(nodes_)) | jnp.any(spacing <= 0.0),
        "Piecewise interpolation nodes must be finite and strictly increasing.",
    )
    query_ = eqx.error_if(
        query_,
        jnp.any(~jnp.isfinite(query_)),
        "Piecewise interpolation queries must be finite.",
    )
    return nodes_, query_


def _piecewise_geometry(
    nodes: ArrayLike,
    query: ArrayLike,
    /,
    *,
    bounds: BoundsMode,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    if bounds not in ("clip", "error", "extrapolate", "fill"):
        raise ValueError("bounds must be 'clip', 'error', 'extrapolate', or 'fill'.")
    nodes_, query_ = _nodes_and_query(nodes, query)
    outside = (query_ < nodes_[0]) | (query_ > nodes_[-1])
    if bounds == "error":
        query_ = eqx.error_if(
            query_,
            jnp.any(outside),
            "Piecewise interpolation query is outside the node interval.",
        )
    query_eval = (
        jnp.clip(query_, nodes_[0], nodes_[-1]) if bounds in ("clip", "fill") else query_
    )
    support = ~outside if bounds == "fill" else jnp.ones(query_.shape, dtype=bool)

    count = int(nodes_.shape[0])
    if count == 1:
        index = jnp.zeros(query_.shape, dtype=jnp.int32)
        return nodes_, query_, index, index, jnp.zeros_like(query_), support

    upper_raw = jnp.searchsorted(nodes_, query_eval, side="right")
    lower = jnp.clip(upper_raw - 1, 0, count - 2).astype(jnp.int32)
    upper = lower + 1
    x0 = nodes_[lower]
    x1 = nodes_[upper]
    fraction = (query_eval - x0) / (x1 - x0)
    return nodes_, query_, lower, upper, fraction, support


def nearest_stencil_from_indices(
    indices: ArrayLike,
    /,
    *,
    source_size: int,
    support: ArrayLike | None = None,
    valid: ArrayLike | None = None,
) -> GatherStencil:
    index = jnp.asarray(indices)
    return GatherStencil(
        indices=index[..., None],
        weights=jnp.ones(index.shape + (1,), dtype=float),
        source_size=source_size,
        valid=None if valid is None else jnp.asarray(valid)[..., None],
        support=support,
    )


def linear_stencil_from_indices(
    lower: ArrayLike,
    upper: ArrayLike,
    fraction: ArrayLike,
    /,
    *,
    source_size: int,
    derivative_order: int = 0,
    interval_width: ArrayLike | None = None,
    support: ArrayLike | None = None,
    valid: ArrayLike | None = None,
) -> GatherStencil:
    lower_ = jnp.asarray(lower)
    upper_ = jnp.asarray(upper)
    fraction_ = jnp.asarray(fraction, dtype=float)
    if lower_.shape != upper_.shape or fraction_.shape != lower_.shape:
        raise ValueError("Linear indices and fractions must have matching shapes.")
    order = int(derivative_order)
    if order == 0:
        weights = jnp.stack((1.0 - fraction_, fraction_), axis=-1)
    elif order == 1:
        if interval_width is None:
            raise ValueError("Linear derivative stencils require interval_width.")
        width = jnp.asarray(interval_width, dtype=float)
        if width.shape != lower_.shape:
            width = jnp.broadcast_to(width, lower_.shape)
        width = eqx.error_if(
            width,
            jnp.any(width <= 0.0) | jnp.any(~jnp.isfinite(width)),
            "Linear interpolation interval widths must be finite and positive.",
        )
        weights = jnp.stack((-1.0 / width, 1.0 / width), axis=-1)
    else:
        raise ValueError(
            "Linear interpolation supports derivatives only through order 1."
        )
    return GatherStencil(
        indices=jnp.stack((lower_, upper_), axis=-1),
        weights=weights,
        source_size=source_size,
        valid=valid,
        support=support,
    )


def nearest_stencil(
    nodes: ArrayLike,
    query: ArrayLike,
    /,
    *,
    bounds: BoundsMode = "clip",
    tie_policy: NearestTiePolicy = "lower",
) -> GatherStencil:
    if tie_policy not in ("lower", "round_even", "upper"):
        raise ValueError("tie_policy must be 'lower', 'round_even', or 'upper'.")
    nodes_, query_, lower, upper, _fraction, support = _piecewise_geometry(
        nodes, query, bounds=bounds
    )
    if int(nodes_.shape[0]) == 1:
        selected = lower
    else:
        lower_distance = jnp.abs(query_ - nodes_[lower])
        upper_distance = jnp.abs(nodes_[upper] - query_)
        scale = jnp.maximum(1.0, jnp.maximum(lower_distance, upper_distance))
        tied = jnp.abs(lower_distance - upper_distance) <= (
            4.0 * jnp.finfo(nodes_.dtype).eps * scale
        )
        use_upper = upper_distance < lower_distance
        if tie_policy == "upper":
            use_upper = use_upper | tied
        elif tie_policy == "round_even":
            use_upper = use_upper | (tied & ((upper % 2) == 0))
        selected = jnp.where(use_upper, upper, lower)
    return nearest_stencil_from_indices(
        selected,
        source_size=int(nodes_.shape[0]),
        support=support,
    )


def linear_stencil(
    nodes: ArrayLike,
    query: ArrayLike,
    /,
    *,
    derivative_order: int = 0,
    bounds: BoundsMode = "clip",
) -> GatherStencil:
    nodes_, _query, lower, upper, fraction, support = _piecewise_geometry(
        nodes, query, bounds=bounds
    )
    width = jnp.where(
        lower == upper,
        1.0,
        nodes_[upper] - nodes_[lower],
    )
    return linear_stencil_from_indices(
        lower,
        upper,
        fraction,
        source_size=int(nodes_.shape[0]),
        derivative_order=derivative_order,
        interval_width=width,
        support=support,
    )


def _source_axis(values: ArrayLike, nodes: Array, axis: int, /) -> tuple[Array, int]:
    array = jnp.asarray(values)
    if array.ndim < 1:
        raise ValueError("Piecewise values must contain a source-node axis.")
    axis_ = int(axis) % array.ndim
    if int(array.shape[axis_]) != int(nodes.shape[0]):
        raise ValueError("Piecewise values source axis must match the node count.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return jnp.moveaxis(array, axis_, 0), axis_


def _fill_result(
    result: InterpolationResult,
    fill_value: Any,
    /,
) -> InterpolationResult:
    payload_ndim = result.values.ndim - result.support.ndim
    support = result.support.reshape(result.support.shape + (1,) * payload_ndim)
    values = jnp.where(
        support,
        result.values,
        jnp.asarray(fill_value, dtype=result.values.dtype),
    )
    return InterpolationResult(values, result.support)


def nearest_interpolate(
    nodes: ArrayLike,
    values: ArrayLike,
    query: ArrayLike,
    /,
    *,
    axis: int = 0,
    bounds: BoundsMode = "clip",
    tie_policy: NearestTiePolicy = "lower",
    source_mask: ArrayLike | None = None,
    mask_mode: MaskMode = "strict",
    fill_value: Any = 0.0,
) -> InterpolationResult:
    nodes_, _ = _nodes_and_query(nodes, query)
    source, _axis = _source_axis(values, nodes_, axis)
    stencil = nearest_stencil(nodes_, query, bounds=bounds, tie_policy=tie_policy)
    return _fill_result(
        apply_gather_stencil(
            source,
            stencil,
            source_mask=source_mask,
            mask_mode=mask_mode,
        ),
        fill_value,
    )


def linear_interpolate(
    nodes: ArrayLike,
    values: ArrayLike,
    query: ArrayLike,
    /,
    *,
    axis: int = 0,
    derivative_order: int = 0,
    bounds: BoundsMode = "clip",
    source_mask: ArrayLike | None = None,
    mask_mode: MaskMode = "strict",
    fill_value: Any = 0.0,
) -> InterpolationResult:
    nodes_, _ = _nodes_and_query(nodes, query)
    source, _axis = _source_axis(values, nodes_, axis)
    stencil = linear_stencil(
        nodes_, query, derivative_order=derivative_order, bounds=bounds
    )
    return _fill_result(
        apply_gather_stencil(
            source,
            stencil,
            source_mask=source_mask,
            mask_mode=mask_mode,
        ),
        fill_value,
    )


def local_cubic_slopes(
    nodes: ArrayLike,
    values: ArrayLike,
    /,
    *,
    axis: int = 0,
) -> Array:
    """Return local cubic slopes using endpoint and secant-average rules."""
    nodes_, _ = _nodes_and_query(nodes, jnp.asarray(0.0))
    source, axis_ = _source_axis(values, nodes_, axis)
    count = int(nodes_.shape[0])
    if count == 1:
        slopes = jnp.zeros_like(source)
    else:
        widths = jnp.diff(nodes_).reshape((count - 1,) + (1,) * (source.ndim - 1))
        secants = jnp.diff(source, axis=0) / widths
        slopes = jnp.concatenate(
            (
                secants[:1],
                0.5 * (secants[:-1] + secants[1:]),
                secants[-1:],
            ),
            axis=0,
        )
    return jnp.moveaxis(slopes, 0, axis_)


def _expand_for_payload(value: ArrayLike, reference: Array, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim > reference.ndim:
        raise ValueError("Interpolation coefficient rank exceeds payload rank.")
    return array.reshape(array.shape + (1,) * (reference.ndim - array.ndim))


def local_cubic_slope(
    previous: ArrayLike,
    current: ArrayLike,
    following: ArrayLike,
    /,
    *,
    previous_width: ArrayLike,
    next_width: ArrayLike,
    has_previous: ArrayLike,
    has_next: ArrayLike,
) -> Array:
    """Evaluate the local secant-average slope with one-sided endpoints."""
    previous_ = jnp.asarray(previous)
    current_ = jnp.asarray(current)
    following_ = jnp.asarray(following)
    h0 = _expand_for_payload(previous_width, current_)
    h1 = _expand_for_payload(next_width, current_)
    has0 = _expand_for_payload(has_previous, current_)
    has1 = _expand_for_payload(has_next, current_)
    left = (current_ - previous_) / jnp.where(has0, h0, 1.0)
    right = (following_ - current_) / jnp.where(has1, h1, 1.0)
    return jnp.where(
        has0 & has1,
        0.5 * (left + right),
        jnp.where(has0, left, jnp.where(has1, right, jnp.zeros_like(current_))),
    )


def linear_segment(
    y0: ArrayLike,
    y1: ArrayLike,
    fraction: ArrayLike,
    interval_width: ArrayLike,
    /,
    *,
    derivative_order: int = 0,
) -> Array:
    y0_ = jnp.asarray(y0)
    y1_ = jnp.asarray(y1)
    order = int(derivative_order)
    if order == 0:
        fraction_ = _expand_for_payload(fraction, y0_)
        return (1.0 - fraction_) * y0_ + fraction_ * y1_
    if order == 1:
        width = _expand_for_payload(interval_width, y0_)
        return (y1_ - y0_) / width
    raise ValueError("Linear interpolation supports derivatives only through order 1.")


def cubic_hermite_segment(
    y0: ArrayLike,
    y1: ArrayLike,
    slope0: ArrayLike,
    slope1: ArrayLike,
    fraction: ArrayLike,
    interval_width: ArrayLike,
    /,
    *,
    derivative_order: int = 0,
) -> Array:
    """Evaluate one cubic Hermite segment or its first two derivatives."""
    y0_ = jnp.asarray(y0)
    y1_ = jnp.asarray(y1)
    slope0_ = jnp.asarray(slope0)
    slope1_ = jnp.asarray(slope1)
    s = _expand_for_payload(fraction, y0_)
    width = _expand_for_payload(interval_width, y0_)
    order = int(derivative_order)
    s2 = s * s

    if order == 0:
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2
        return h00 * y0_ + h10 * width * slope0_ + h01 * y1_ + h11 * width * slope1_
    if order == 1:
        h00 = 6.0 * s2 - 6.0 * s
        h10 = 3.0 * s2 - 4.0 * s + 1.0
        h01 = -6.0 * s2 + 6.0 * s
        h11 = 3.0 * s2 - 2.0 * s
        return (
            h00 * y0_ + h10 * width * slope0_ + h01 * y1_ + h11 * width * slope1_
        ) / width
    if order == 2:
        h00 = 12.0 * s - 6.0
        h10 = 6.0 * s - 4.0
        h01 = -12.0 * s + 6.0
        h11 = 6.0 * s - 2.0
        return (h00 * y0_ + h10 * width * slope0_ + h01 * y1_ + h11 * width * slope1_) / (
            width * width
        )
    raise ValueError(
        "Cubic Hermite interpolation supports derivatives only through order 2."
    )


def cubic_hermite_interpolate(
    nodes: ArrayLike,
    values: ArrayLike,
    query: ArrayLike,
    /,
    *,
    slopes: ArrayLike | None = None,
    axis: int = 0,
    derivative_order: int = 0,
    bounds: BoundsMode = "extrapolate",
    snap_tolerance: float = 0.0,
    fill_value: Any = 0.0,
) -> InterpolationResult:
    nodes_, query_, lower, upper, fraction, support = _piecewise_geometry(
        nodes, query, bounds=bounds
    )
    source, axis_ = _source_axis(values, nodes_, axis)
    slope_values = (
        local_cubic_slopes(nodes_, values, axis=axis_)
        if slopes is None
        else jnp.asarray(slopes)
    )
    slopes_source, _ = _source_axis(slope_values, nodes_, axis_)

    if int(nodes_.shape[0]) == 1:
        output = jnp.broadcast_to(source[0], query_.shape + source.shape[1:])
        if int(derivative_order) > 0:
            output = jnp.zeros_like(output)
    else:
        width = nodes_[upper] - nodes_[lower]
        output = cubic_hermite_segment(
            source[lower],
            source[upper],
            slopes_source[lower],
            slopes_source[upper],
            fraction,
            width,
            derivative_order=derivative_order,
        )
        snap = float(snap_tolerance)
        if snap < 0.0:
            raise ValueError("snap_tolerance must be non-negative.")
        if snap > 0.0 and int(derivative_order) == 0:
            lower_distance = jnp.abs(query_ - nodes_[lower])
            upper_distance = jnp.abs(nodes_[upper] - query_)
            use_upper = upper_distance < lower_distance
            nearest = jnp.where(use_upper, upper, lower)
            on_node = jnp.minimum(lower_distance, upper_distance) <= snap
            output = jnp.where(
                _expand_for_payload(on_node, output),
                source[nearest],
                output,
            )

    return _fill_result(InterpolationResult(output, support), fill_value)


__all__ = [
    "CUBIC_HERMITE_CAPABILITIES",
    "LINEAR_CAPABILITIES",
    "NEAREST_CAPABILITIES",
    "cubic_hermite_interpolate",
    "cubic_hermite_segment",
    "linear_interpolate",
    "linear_segment",
    "linear_stencil",
    "linear_stencil_from_indices",
    "local_cubic_slope",
    "local_cubic_slopes",
    "nearest_interpolate",
    "nearest_stencil",
    "nearest_stencil_from_indices",
]
