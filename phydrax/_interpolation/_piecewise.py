#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, assert_never, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._dtype_names import inexact_result_type
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    ArraySpace,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    prepare as prepare_linear_solve,
    PreparedLinearSolve,
    RHSLayout,
    solve as solve_linear,
    StructuredDirect,
    TridiagonalLinearOperator,
)
from ..typing import parse
from ._stencil import apply_gather_stencil, GatherStencil
from ._types import (
    BoundsMode,
    InterpolationCapabilities,
    InterpolationResult,
    MaskMode,
    NearestTiePolicy,
)


CubicSplineEndCondition: TypeAlias = Literal["clamped", "natural", "not-a-knot"]


def _derivative_order(value: int, maximum: int, family: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("derivative_order must be an integer.")
    order = int(value)
    if order < 0 or order > maximum:
        raise ValueError(
            f"{family} interpolation supports derivatives only through order {maximum}."
        )
    return order


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
    dtype = inexact_result_type(nodes_raw, query_raw)
    nodes_ = nodes_raw.astype(dtype)
    query_ = query_raw.astype(dtype)
    if nodes_.ndim != 1 or nodes_.shape[0] <= 0:
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
    bounds = parse(bounds, BoundsMode, "bounds")
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
    support = ~outside if bounds == "fill" else jnp.ones(query_.shape, dtype=jnp.bool_)

    count = nodes_.shape[0]
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
        weights=jnp.ones(index.shape + (1,), dtype=jnp.float64),
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
    fraction_ = jnp.asarray(fraction, dtype=jnp.float64)
    if lower_.shape != upper_.shape or fraction_.shape != lower_.shape:
        raise ValueError("Linear indices and fractions must have matching shapes.")
    order = _derivative_order(derivative_order, 1, "Linear")
    if order == 0:
        weights = jnp.stack((1.0 - fraction_, fraction_), axis=-1)
    else:
        if interval_width is None:
            raise ValueError("Linear derivative stencils require interval_width.")
        width = jnp.asarray(interval_width, dtype=jnp.float64)
        if width.shape != lower_.shape:
            width = jnp.broadcast_to(width, lower_.shape)
        width = eqx.error_if(
            width,
            jnp.any(width <= 0.0) | jnp.any(~jnp.isfinite(width)),
            "Linear interpolation interval widths must be finite and positive.",
        )
        weights = jnp.stack((-1.0 / width, 1.0 / width), axis=-1)
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
    tie_policy = parse(tie_policy, NearestTiePolicy, "tie_policy")
    nodes_, query_, lower, upper, _fraction, support = _piecewise_geometry(
        nodes, query, bounds=bounds
    )
    if nodes_.shape[0] == 1:
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
        source_size=nodes_.shape[0],
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
        source_size=nodes_.shape[0],
        derivative_order=derivative_order,
        interval_width=width,
        support=support,
    )


def _source_axis(values: ArrayLike, nodes: Array, axis: int, /) -> tuple[Array, int]:
    array = jnp.asarray(values)
    if array.ndim < 1:
        raise ValueError("Piecewise values must contain a source-node axis.")
    if isinstance(axis, bool) or not isinstance(axis, Integral):
        raise TypeError("axis must be an integer.")
    axis_value = int(axis)
    if axis_value < -array.ndim or axis_value >= array.ndim:
        raise ValueError(
            f"Piecewise source axis {axis_value} is out of bounds for rank {array.ndim}."
        )
    axis_ = axis_value % array.ndim
    if array.shape[axis_] != nodes.shape[0]:
        raise ValueError("Piecewise values source axis must match the node count.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype("float64")
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
    left_fill_value: Any | None = None,
    right_fill_value: Any | None = None,
) -> InterpolationResult:
    nodes_, query_ = _nodes_and_query(nodes, query)
    source, _axis = _source_axis(values, nodes_, axis)
    if bounds != "fill" and (left_fill_value is not None or right_fill_value is not None):
        raise ValueError("Side-specific fill values require bounds='fill'.")
    stencil = linear_stencil(
        nodes_, query_, derivative_order=derivative_order, bounds=bounds
    )
    result = _fill_result(
        apply_gather_stencil(
            source,
            stencil,
            source_mask=source_mask,
            mask_mode=mask_mode,
        ),
        fill_value,
    )
    if bounds != "fill" or (left_fill_value is None and right_fill_value is None):
        return result
    payload_ndim = result.values.ndim - query_.ndim
    mask_shape = query_.shape + (1,) * payload_ndim
    left = fill_value if left_fill_value is None else left_fill_value
    right = fill_value if right_fill_value is None else right_fill_value
    filled = jnp.where(
        (query_ < nodes_[0]).reshape(mask_shape),
        jnp.asarray(left, dtype=result.values.dtype),
        result.values,
    )
    filled = jnp.where(
        (query_ > nodes_[-1]).reshape(mask_shape),
        jnp.asarray(right, dtype=result.values.dtype),
        filled,
    )
    return InterpolationResult(filled, result.support)


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
    count = nodes_.shape[0]
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
    order = _derivative_order(derivative_order, 1, "Linear")
    if order == 0:
        fraction_ = _expand_for_payload(fraction, y0_)
        return (1.0 - fraction_) * y0_ + fraction_ * y1_
    width = _expand_for_payload(interval_width, y0_)
    return (y1_ - y0_) / width


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
    order = _derivative_order(derivative_order, 2, "Cubic Hermite")
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
    h00 = 12.0 * s - 6.0
    h10 = 6.0 * s - 4.0
    h01 = -12.0 * s + 6.0
    h11 = 6.0 * s - 2.0
    return (h00 * y0_ + h10 * width * slope0_ + h01 * y1_ + h11 * width * slope1_) / (
        width * width
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
    order = _derivative_order(derivative_order, 2, "Cubic Hermite")
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

    if nodes_.shape[0] == 1:
        output = jnp.broadcast_to(source[0], query_.shape + source.shape[1:])
        if order > 0:
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
            derivative_order=order,
        )
        snap = float(snap_tolerance)
        if snap < 0.0:
            raise ValueError("snap_tolerance must be non-negative.")
        if snap > 0.0 and order == 0:
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


class UniformSpanLocation(StrictModule, NonTrainableState):
    """Constant-time span location of queries on a uniform node grid."""

    lower: Array
    fraction: Array
    support: Array

    def __init__(self, lower: Array, fraction: Array, support: Array, /) -> None:
        lower_ = jnp.asarray(lower, dtype=jnp.int32)
        fraction_ = jnp.asarray(fraction)
        support_ = jnp.asarray(support, dtype=jnp.bool_)
        if lower_.shape != fraction_.shape or support_.shape != fraction_.shape:
            raise ValueError("Uniform span location fields must share the query shape.")
        self.lower = lower_
        self.fraction = fraction_
        self.support = support_


class UniformNodeGrid(StrictModule, NonTrainableState):
    """Uniform node grid with constant-time span lookup."""

    start: Array
    spacing: Array
    node_count: int = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        start: float,
        stop: float,
        node_count: int,
        /,
        *,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        if isinstance(start, bool) or not isinstance(start, Real):
            raise TypeError("Uniform grid start must be a real scalar.")
        if isinstance(stop, bool) or not isinstance(stop, Real):
            raise TypeError("Uniform grid stop must be a real scalar.")
        if isinstance(node_count, bool) or not isinstance(node_count, Integral):
            raise TypeError("Uniform grid node_count must be an integer.")
        dtype_ = np.dtype(dtype)
        if not np.issubdtype(dtype_, np.floating):
            raise TypeError("Uniform grid dtype must be a real floating dtype.")
        start_ = float(start)
        stop_ = float(stop)
        count = int(node_count)
        if not (math.isfinite(start_) and math.isfinite(stop_)):
            raise ValueError("Uniform grid bounds must be finite.")
        if not stop_ > start_:
            raise ValueError("Uniform grid stop must exceed start.")
        if count < 2:
            raise ValueError("Uniform grid requires at least two nodes.")
        start_host = np.asarray(start_, dtype=dtype_)
        spacing_host = (np.asarray(stop_, dtype=dtype_) - start_host) / np.asarray(
            count - 1, dtype=dtype_
        )
        if not (np.isfinite(spacing_host) and spacing_host > 0):
            raise ValueError("Uniform grid spacing must be finite and positive.")
        self.start = jnp.asarray(start_host)
        self.spacing = jnp.asarray(spacing_host)
        self.node_count = count
        self.grid_id = canonical_fingerprint(
            {
                "kind": "uniform-node-grid",
                "start": start_,
                "stop": stop_,
                "node_count": count,
                "dtype": dtype_.name,
            }
        )

    @property
    def nodes(self) -> Array:
        return self.start + self.spacing * jnp.arange(
            self.node_count, dtype=self.spacing.dtype
        )

    @property
    def stop(self) -> Array:
        return self.start + self.spacing * jnp.asarray(
            self.node_count - 1, dtype=self.spacing.dtype
        )

    def locate(
        self,
        query: ArrayLike,
        /,
        *,
        bounds: BoundsMode = "error",
    ) -> UniformSpanLocation:
        bounds = parse(bounds, BoundsMode, "bounds")
        query_raw = jnp.asarray(query)
        if jnp.issubdtype(query_raw.dtype, jnp.complexfloating):
            raise TypeError("Uniform grid queries must be real-valued.")
        dtype = inexact_result_type(query_raw, self.start)
        query_ = eqx.error_if(
            query_raw.astype(dtype),
            jnp.any(~jnp.isfinite(query_raw)),
            "Piecewise interpolation queries must be finite.",
        )
        start = self.start.astype(dtype)
        stop = self.stop.astype(dtype)
        outside = (query_ < start) | (query_ > stop)
        if bounds == "error":
            query_ = eqx.error_if(
                query_,
                jnp.any(outside),
                "Piecewise interpolation query is outside the node interval.",
            )
        query_eval = (
            jnp.clip(query_, start, stop) if bounds in ("clip", "fill") else query_
        )
        support = (
            ~outside if bounds == "fill" else jnp.ones(query_.shape, dtype=jnp.bool_)
        )
        position = (query_eval - start) / self.spacing.astype(dtype)
        lower = jnp.clip(jnp.floor(position), 0, self.node_count - 2).astype(jnp.int32)
        fraction = position - lower.astype(dtype)
        return UniformSpanLocation(lower, fraction, support)


def _uniform_node_payload(
    grid: UniformNodeGrid, values: ArrayLike, name: str, /
) -> Array:
    array = jnp.asarray(values)
    if array.ndim < 1 or array.shape[0] != grid.node_count:
        raise ValueError(f"{name} must lead with the uniform grid node axis.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(grid.spacing.dtype)
    return array


def cubic_hermite_uniform_interpolate(
    grid: UniformNodeGrid,
    values: ArrayLike,
    slopes: ArrayLike,
    query: ArrayLike,
    /,
    *,
    derivative_order: int = 0,
    bounds: BoundsMode = "error",
    fill_value: Any = 0.0,
) -> InterpolationResult:
    """Evaluate a cubic Hermite interpolant on a uniform grid in constant time."""
    order = _derivative_order(derivative_order, 2, "Cubic Hermite")
    source = _uniform_node_payload(grid, values, "Uniform Hermite values")
    slopes_ = _uniform_node_payload(grid, slopes, "Uniform Hermite slopes")
    if slopes_.shape != source.shape:
        raise ValueError("Uniform Hermite slopes must match the values shape.")
    location = grid.locate(query, bounds=bounds)
    upper = location.lower + 1
    output = cubic_hermite_segment(
        source[location.lower],
        source[upper],
        slopes_[location.lower],
        slopes_[upper],
        location.fraction,
        grid.spacing,
        derivative_order=order,
    )
    return _fill_result(InterpolationResult(output, location.support), fill_value)


class CubicSplineSlopes(StrictModule, NonTrainableState):
    """Global C2 cubic-spline node slopes plus linear-solve evidence."""

    slopes: Array
    status: Array
    successful: Array

    def __init__(self, slopes: Array, status: Array, /) -> None:
        slopes_ = jnp.asarray(slopes)
        status_ = jnp.asarray(status, dtype=jnp.int32)
        self.slopes = slopes_
        self.status = status_
        self.successful = jnp.all(status_ == int(LinearSolveStatus.SUCCESS)) & jnp.all(
            jnp.isfinite(slopes_)
        )


def _end_rows(
    condition: CubicSplineEndCondition,
    /,
) -> tuple[float, float]:
    match condition:
        case "clamped":
            return 1.0, 0.0
        case "natural":
            return 2.0, 1.0
        case "not-a-knot":
            return 2.0, 4.0
        case _:
            assert_never(condition)


class CubicSplineSlopePlan(StrictModule, NonTrainableState):
    """Prepared global cubic-spline slope solve on a uniform grid."""

    grid: UniformNodeGrid
    prepared: PreparedLinearSolve
    left: CubicSplineEndCondition = eqx.field(static=True)
    right: CubicSplineEndCondition = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: UniformNodeGrid,
        /,
        *,
        left: CubicSplineEndCondition,
        right: CubicSplineEndCondition,
    ) -> None:
        if not isinstance(grid, UniformNodeGrid):
            raise TypeError("CubicSplineSlopePlan requires a UniformNodeGrid.")
        left_ = parse(left, CubicSplineEndCondition, "left")
        right_ = parse(right, CubicSplineEndCondition, "right")
        count = grid.node_count
        minimum = 4 if "not-a-knot" in (left_, right_) else 2
        if count < minimum:
            raise ValueError(
                f"Cubic spline end conditions ({left_}, {right_}) require at least "
                f"{minimum} nodes."
            )
        dtype = np.dtype(grid.spacing.dtype)
        diagonal = np.full((count,), 4.0, dtype=dtype)
        lower = np.ones((count - 1,), dtype=dtype)
        upper = np.ones((count - 1,), dtype=dtype)
        diagonal[0], upper[0] = _end_rows(left_)
        diagonal[-1], lower[-1] = _end_rows(right_)
        operator = TridiagonalLinearOperator(
            lower,
            diagonal,
            upper,
            space=ArraySpace((count,), dtype=dtype),
        )
        policy = LinearSolvePolicy(StructuredDirect(), failure=FailurePolicy("status"))
        prepared = prepare_linear_solve(LinearSystem(operator), policy)
        self.grid = grid
        self.prepared = prepared
        self.left = left_
        self.right = right_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "uniform-cubic-spline-slope-plan",
                "grid": grid.grid_id,
                "left": left_,
                "right": right_,
            }
        )

    def slopes(
        self,
        values: ArrayLike,
        /,
        *,
        left_slope: ArrayLike | None = None,
        right_slope: ArrayLike | None = None,
    ) -> CubicSplineSlopes:
        if (left_slope is None) != (self.left != "clamped"):
            raise ValueError(
                "left_slope is required exactly when the left end is clamped."
            )
        if (right_slope is None) != (self.right != "clamped"):
            raise ValueError(
                "right_slope is required exactly when the right end is clamped."
            )
        dtype = self.grid.spacing.dtype
        source = _uniform_node_payload(self.grid, values, "Cubic spline values")
        if jnp.issubdtype(source.dtype, jnp.complexfloating):
            raise TypeError("Cubic spline values must be real-valued.")
        source = source.astype(dtype)
        payload_shape = source.shape[1:]
        columns = math.prod(payload_shape)
        y = source.reshape((self.grid.node_count, columns))
        h = self.grid.spacing
        interior = 3.0 * (y[2:] - y[:-2]) / h
        first = self._end_rhs(
            self.left, 1.0, y[0], y[1], y[2:3], left_slope, payload_shape
        )
        last = self._end_rhs(
            self.right, -1.0, y[-1], y[-2], y[-3:-2], right_slope, payload_shape
        )
        rhs = jnp.concatenate((first[None], interior, last[None]), axis=0)
        result = solve_linear(self.prepared, rhs, rhs_layout=RHSLayout((columns,)))
        slopes = result.value.reshape(source.shape)
        return CubicSplineSlopes(slopes, result.status.reshape(payload_shape))

    def _end_rhs(
        self,
        condition: CubicSplineEndCondition,
        orientation: float,
        edge: Array,
        neighbor: Array,
        beyond: Array,
        slope: ArrayLike | None,
        payload_shape: tuple[int, ...],
        /,
    ) -> Array:
        """Right-hand side of one end row.

        `orientation` is `+1` at the left end and `-1` at the right end, where
        `edge`, `neighbor`, and `beyond` walk inward from that end.
        """
        h = self.grid.spacing
        match condition:
            case "clamped":
                if slope is None:
                    raise ValueError("A clamped spline end requires its slope.")
                slope_ = jnp.broadcast_to(
                    jnp.asarray(slope, dtype=h.dtype), payload_shape
                )
                return slope_.reshape(edge.shape)
            case "natural":
                return orientation * 3.0 * (neighbor - edge) / h
            case "not-a-knot":
                far = beyond[0]
                return (
                    orientation
                    * (2.0 * (-edge + 2.0 * neighbor - far) + 3.0 * (far - edge))
                    / h
                )
            case _:
                assert_never(condition)


class CubicHermiteKnotJets(StrictModule, NonTrainableState):
    """One-sided first and second derivatives at both ends of every segment."""

    segment_start: Array
    segment_end: Array

    def __init__(self, segment_start: Array, segment_end: Array, /) -> None:
        start = jnp.asarray(segment_start)
        end = jnp.asarray(segment_end)
        if start.shape != end.shape or start.ndim < 2 or start.shape[0] != 2:
            raise ValueError("Knot jets must have matching shape (2, segments, ...).")
        self.segment_start = start
        self.segment_end = end


def cubic_hermite_knot_jets(
    grid: UniformNodeGrid,
    values: ArrayLike,
    slopes: ArrayLike,
    /,
) -> CubicHermiteKnotJets:
    """Return one-sided knot derivatives of a uniform cubic Hermite interpolant."""
    source = _uniform_node_payload(grid, values, "Uniform Hermite values")
    slopes_ = _uniform_node_payload(grid, slopes, "Uniform Hermite slopes")
    if slopes_.shape != source.shape:
        raise ValueError("Uniform Hermite slopes must match the values shape.")
    segments = grid.node_count - 1
    fraction_dtype = inexact_result_type(source, grid.spacing)

    def jet(fraction: float, /) -> Array:
        s = jnp.full((segments,), fraction, dtype=fraction_dtype)
        return jnp.stack(
            [
                cubic_hermite_segment(
                    source[:-1],
                    source[1:],
                    slopes_[:-1],
                    slopes_[1:],
                    s,
                    grid.spacing,
                    derivative_order=order,
                )
                for order in (1, 2)
            ]
        )

    return CubicHermiteKnotJets(jet(0.0), jet(1.0))


__all__ = [
    "CUBIC_HERMITE_CAPABILITIES",
    "LINEAR_CAPABILITIES",
    "NEAREST_CAPABILITIES",
    "CubicHermiteKnotJets",
    "CubicSplineEndCondition",
    "CubicSplineSlopePlan",
    "CubicSplineSlopes",
    "UniformNodeGrid",
    "UniformSpanLocation",
    "cubic_hermite_knot_jets",
    "cubic_hermite_interpolate",
    "cubic_hermite_uniform_interpolate",
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
