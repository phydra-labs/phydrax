#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


class PhaseGeometryMetrics(StrictModule):
    """Quadrature-based phase measure and centroid evidence."""

    measure: Array
    centroid: Array
    centroid_defined: Array

    def __init__(
        self,
        *,
        measure: ArrayLike,
        centroid: ArrayLike,
        centroid_defined: ArrayLike,
    ):
        self.measure = jnp.asarray(measure)
        self.centroid = jnp.asarray(centroid)
        self.centroid_defined = jnp.asarray(centroid_defined, dtype=bool)


class InterfaceDistanceMetrics(StrictModule):
    """Symmetric point-set distances between predicted and reference interfaces."""

    symmetric_mean_distance: Array
    hausdorff_distance: Array
    percentile_hausdorff_distance: Array

    def __init__(
        self,
        *,
        symmetric_mean_distance: ArrayLike,
        hausdorff_distance: ArrayLike,
        percentile_hausdorff_distance: ArrayLike,
    ):
        self.symmetric_mean_distance = jnp.asarray(symmetric_mean_distance)
        self.hausdorff_distance = jnp.asarray(hausdorff_distance)
        self.percentile_hausdorff_distance = jnp.asarray(percentile_hausdorff_distance)


def _real_array(value: ArrayLike, name: str, /) -> Array:
    values = jnp.asarray(value)
    if jnp.iscomplexobj(values):
        raise TypeError(f"{name} must be real-valued.")
    return values.astype(float)


def _positive_width(width: float, /) -> float:
    value = float(width)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("width must be finite and positive.")
    return value


def regularized_heaviside_values(
    level_set_values: ArrayLike,
    /,
    *,
    width: float,
) -> Array:
    """Evaluate the compact cosine Heaviside regularization on real values."""

    values = _real_array(level_set_values, "level_set_values")
    width_ = _positive_width(width)
    scaled = values / width_
    interior = 0.5 * (1.0 + scaled + jnp.sin(jnp.pi * scaled) / jnp.pi)
    return jnp.where(
        values <= -width_,
        0.0,
        jnp.where(values >= width_, 1.0, interior),
    )


def regularized_delta_values(
    level_set_values: ArrayLike,
    /,
    *,
    width: float,
) -> Array:
    """Evaluate the compact derivative of the cosine Heaviside regularization."""

    values = _real_array(level_set_values, "level_set_values")
    width_ = _positive_width(width)
    scaled = values / width_
    interior = (1.0 + jnp.cos(jnp.pi * scaled)) / (2.0 * width_)
    return jnp.where(jnp.abs(values) < width_, interior, 0.0)


def _point_cloud(value: ArrayLike, name: str, /) -> Array:
    points = _real_array(value, name)
    if points.ndim < 2 or int(points.shape[-2]) <= 0 or int(points.shape[-1]) <= 0:
        raise ValueError(f"{name} must end in non-empty (point, coordinate) axes.")
    return points


def _point_mask_shape(mask: ArrayLike | None, count: int, name: str, /):
    if mask is None:
        return (), None
    values = jnp.asarray(mask, dtype=bool)
    if values.ndim < 1 or int(values.shape[-1]) != count:
        raise ValueError(f"{name} must end in a point axis of length {count}.")
    return tuple(int(size) for size in values.shape[:-1]), values


def _broadcast_point_mask(
    mask: Array | None,
    case_shape: tuple[int, ...],
    count: int,
    /,
) -> Array:
    values = jnp.ones((count,), dtype=bool) if mask is None else mask
    return jnp.broadcast_to(values, case_shape + (count,))


def _checked_active_points(points: Array, mask: Array, name: str, /) -> Array:
    active_finite = (~mask) | jnp.all(jnp.isfinite(points), axis=-1)
    points = eqx.error_if(
        points,
        jnp.any(~active_finite),
        f"Active {name} must be finite.",
    )
    points = eqx.error_if(
        points,
        jnp.any(jnp.sum(mask, axis=-1) == 0),
        f"Every {name} case must contain at least one active point.",
    )
    return jnp.where(mask[..., None], points, 0.0)


def _directed_nearest_distances(
    source: Array,
    target: Array,
    target_mask: Array,
    /,
    *,
    chunk_size: int,
) -> Array:
    target_square = jnp.sum(target * target, axis=-1)
    blocks = []
    for start in range(0, int(source.shape[-2]), chunk_size):
        stop = min(start + chunk_size, int(source.shape[-2]))
        block = source[..., start:stop, :]
        block_square = jnp.sum(block * block, axis=-1, keepdims=True)
        cross = oe.contract("...id,...jd->...ij", block, target)
        squared = jnp.maximum(
            block_square + target_square[..., None, :] - 2.0 * cross,
            0.0,
        )
        squared = jnp.where(target_mask[..., None, :], squared, jnp.inf)
        blocks.append(jnp.sqrt(jnp.min(squared, axis=-1)))
    return jnp.concatenate(blocks, axis=-1)


def _masked_mean(values: Array, mask: Array, /) -> Array:
    count = jnp.sum(mask, axis=-1)
    return jnp.sum(jnp.where(mask, values, 0.0), axis=-1) / count


def _masked_max(values: Array, mask: Array, /) -> Array:
    return jnp.max(jnp.where(mask, values, -jnp.inf), axis=-1)


def _masked_quantile(values: Array, mask: Array, quantile: float, /) -> Array:
    ordered = jnp.sort(jnp.where(mask, values, jnp.inf), axis=-1)
    count = jnp.sum(mask, axis=-1).astype(jnp.int32)
    position = quantile * (count.astype(values.dtype) - 1.0)
    lower = jnp.floor(position).astype(jnp.int32)
    upper = jnp.ceil(position).astype(jnp.int32)
    lower_value = jnp.take_along_axis(ordered, lower[..., None], axis=-1)[..., 0]
    upper_value = jnp.take_along_axis(ordered, upper[..., None], axis=-1)[..., 0]
    fraction = position - lower.astype(position.dtype)
    return lower_value + fraction * (upper_value - lower_value)


def phase_geometry_metrics(
    phase_fraction: ArrayLike,
    coordinates: ArrayLike,
    quadrature_weights: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
) -> PhaseGeometryMetrics:
    """Measure one phase and its centroid from flattened physical quadrature.

    ``phase_fraction`` and ``quadrature_weights`` end in one point axis;
    ``coordinates`` ends in ``(point, coordinate)``. Shared arrays broadcast over
    leading cases. Active phase fractions must lie in ``[0, 1]`` and quadrature
    weights must be finite and nonnegative. A zero-measure phase has a NaN
    centroid and ``centroid_defined=False``.
    """

    fraction = _real_array(phase_fraction, "phase_fraction")
    points = _point_cloud(coordinates, "coordinates")
    weights = _real_array(quadrature_weights, "quadrature_weights")
    if fraction.ndim < 1:
        raise ValueError("phase_fraction must end in a point axis.")
    count = int(fraction.shape[-1])
    if int(points.shape[-2]) != count:
        raise ValueError("coordinates and phase_fraction point counts must match.")
    if weights.ndim < 1 or int(weights.shape[-1]) != count:
        raise ValueError("quadrature_weights must match the phase point count.")
    mask_case, mask_ = _point_mask_shape(mask, count, "mask")
    case_shape = jnp.broadcast_shapes(
        tuple(int(size) for size in fraction.shape[:-1]),
        tuple(int(size) for size in points.shape[:-2]),
        tuple(int(size) for size in weights.shape[:-1]),
        mask_case,
    )
    dimension = int(points.shape[-1])
    fraction = jnp.broadcast_to(fraction, case_shape + (count,))
    points = jnp.broadcast_to(points, case_shape + (count, dimension))
    weights = jnp.broadcast_to(weights, case_shape + (count,))
    active = _broadcast_point_mask(mask_, case_shape, count)

    valid_fraction = (~active) | (
        jnp.isfinite(fraction) & (fraction >= 0.0) & (fraction <= 1.0)
    )
    valid_weights = (~active) | (jnp.isfinite(weights) & (weights >= 0.0))
    fraction = eqx.error_if(
        fraction,
        jnp.any(~valid_fraction),
        "Active phase fractions must be finite and lie in [0, 1].",
    )
    weights = eqx.error_if(
        weights,
        jnp.any(~valid_weights),
        "Active phase quadrature weights must be finite and nonnegative.",
    )
    safe_points = _checked_active_points(points, active, "phase coordinates")
    effective = jnp.where(active, fraction * weights, 0.0)
    measure = jnp.sum(effective, axis=-1)
    numerator = oe.contract("...n,...nd->...d", effective, safe_points)
    defined = measure > 0.0
    safe_measure = jnp.where(defined, measure, 1.0)
    centroid = numerator / safe_measure[..., None]
    centroid = jnp.where(defined[..., None], centroid, jnp.nan)
    return PhaseGeometryMetrics(
        measure=measure,
        centroid=centroid,
        centroid_defined=defined,
    )


def interface_distance_metrics(
    predicted_points: ArrayLike,
    reference_points: ArrayLike,
    /,
    *,
    predicted_mask: ArrayLike | None = None,
    reference_mask: ArrayLike | None = None,
    percentile: float = 0.95,
    chunk_size: int = 1024,
) -> InterfaceDistanceMetrics:
    """Return symmetric mean, Hausdorff, and percentile-Hausdorff distances.

    Point sets end in ``(point, coordinate)`` and may carry broadcast-compatible
    leading case axes. Masks support fixed-capacity padded point sets. The
    symmetric mean averages the two directed nearest-point means; Hausdorff
    quantities take the maximum of their two directed values. Percentiles use
    linear interpolation over active directed distances.
    """

    predicted = _point_cloud(predicted_points, "predicted_points")
    reference = _point_cloud(reference_points, "reference_points")
    if int(predicted.shape[-1]) != int(reference.shape[-1]):
        raise ValueError("Predicted and reference coordinate dimensions must match.")
    percentile_ = float(percentile)
    if not math.isfinite(percentile_) or not 0.0 <= percentile_ <= 1.0:
        raise ValueError("percentile must be finite and lie in [0, 1].")
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be positive.")

    predicted_count = int(predicted.shape[-2])
    reference_count = int(reference.shape[-2])
    predicted_mask_case, predicted_mask_ = _point_mask_shape(
        predicted_mask,
        predicted_count,
        "predicted_mask",
    )
    reference_mask_case, reference_mask_ = _point_mask_shape(
        reference_mask,
        reference_count,
        "reference_mask",
    )
    case_shape = jnp.broadcast_shapes(
        tuple(int(size) for size in predicted.shape[:-2]),
        tuple(int(size) for size in reference.shape[:-2]),
        predicted_mask_case,
        reference_mask_case,
    )
    dimension = int(predicted.shape[-1])
    predicted = jnp.broadcast_to(
        predicted,
        case_shape + (predicted_count, dimension),
    )
    reference = jnp.broadcast_to(
        reference,
        case_shape + (reference_count, dimension),
    )
    predicted_active = _broadcast_point_mask(
        predicted_mask_,
        case_shape,
        predicted_count,
    )
    reference_active = _broadcast_point_mask(
        reference_mask_,
        case_shape,
        reference_count,
    )
    predicted = _checked_active_points(
        predicted,
        predicted_active,
        "predicted interface",
    )
    reference = _checked_active_points(
        reference,
        reference_active,
        "reference interface",
    )

    predicted_to_reference = _directed_nearest_distances(
        predicted,
        reference,
        reference_active,
        chunk_size=chunk,
    )
    reference_to_predicted = _directed_nearest_distances(
        reference,
        predicted,
        predicted_active,
        chunk_size=chunk,
    )
    mean_distance = 0.5 * (
        _masked_mean(predicted_to_reference, predicted_active)
        + _masked_mean(reference_to_predicted, reference_active)
    )
    hausdorff = jnp.maximum(
        _masked_max(predicted_to_reference, predicted_active),
        _masked_max(reference_to_predicted, reference_active),
    )
    percentile_hausdorff = jnp.maximum(
        _masked_quantile(predicted_to_reference, predicted_active, percentile_),
        _masked_quantile(reference_to_predicted, reference_active, percentile_),
    )
    return InterfaceDistanceMetrics(
        symmetric_mean_distance=mean_distance,
        hausdorff_distance=hausdorff,
        percentile_hausdorff_distance=percentile_hausdorff,
    )


__all__ = [
    "InterfaceDistanceMetrics",
    "PhaseGeometryMetrics",
    "interface_distance_metrics",
    "phase_geometry_metrics",
    "regularized_delta_values",
    "regularized_heaviside_values",
]
