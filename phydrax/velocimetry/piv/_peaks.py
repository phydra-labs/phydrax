#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ._types import CorrelationBatch, PeakBatch


def _axis_fit(
    negative: Array,
    center: Array,
    positive: Array,
    interior: Array,
    /,
    *,
    method: str,
) -> tuple[Array, Array, Array]:
    if method == "gaussian":
        positive_values = (negative > 0.0) & (center > 0.0) & (positive > 0.0)
        negative_ = jnp.log(jnp.where(positive_values, negative, 1.0))
        center_ = jnp.log(jnp.where(positive_values, center, 1.0))
        positive_ = jnp.log(jnp.where(positive_values, positive, 1.0))
        fit_support = interior & positive_values
    else:
        negative_, center_, positive_ = negative, center, positive
        fit_support = interior
    curvature = negative_ - 2.0 * center_ + positive_
    concave = curvature < -jnp.finfo(center.dtype).eps
    fit_valid = fit_support & concave & jnp.isfinite(curvature)
    delta = 0.5 * (negative_ - positive_) / jnp.where(fit_valid, curvature, 1.0)
    delta = jnp.where(fit_valid, jnp.clip(delta, -1.0, 1.0), 0.0)
    return delta, jnp.where(fit_valid, curvature, 0.0), fit_valid


def find_top_peaks(
    correlation: CorrelationBatch,
    /,
    *,
    top_k: int,
    method: str,
) -> PeakBatch:
    """Select deterministic row-major ties and fit separable subpixel peaks."""
    if not isinstance(correlation, CorrelationBatch):
        raise TypeError("correlation must be a CorrelationBatch.")
    count = int(top_k)
    if count < 1 or count > correlation.values.shape[-2] * correlation.values.shape[-1]:
        raise ValueError("top_k is outside the correlation surface capacity.")
    method_ = str(method)
    if method_ not in ("parabolic", "gaussian"):
        raise ValueError("method must be parabolic or gaussian.")
    shape = correlation.values.shape
    flattened = correlation.values.reshape((shape[0], -1))
    valid_flattened = correlation.valid.reshape((shape[0], -1))
    safe = jnp.where(valid_flattened & jnp.isfinite(flattened), flattened, -jnp.inf)
    order = jnp.argsort(-safe, axis=-1, stable=True)
    indices = order[:, :count]
    values = jnp.take_along_axis(safe, indices, axis=-1)
    peak_valid = jnp.isfinite(values)
    columns = shape[-1]
    row = indices // columns
    column = indices % columns
    batch = jnp.arange(shape[0], dtype=jnp.int32)[:, None]

    def gather(row_index: Array, column_index: Array) -> Array:
        return correlation.values[
            batch,
            jnp.clip(row_index, 0, shape[-2] - 1),
            jnp.clip(column_index, 0, shape[-1] - 1),
        ]

    center = gather(row, column)
    row_delta, row_curvature, row_fit = _axis_fit(
        gather(row - 1, column),
        center,
        gather(row + 1, column),
        (row > 0) & (row < shape[-2] - 1),
        method=method_,
    )
    column_delta, column_curvature, column_fit = _axis_fit(
        gather(row, column - 1),
        center,
        gather(row, column + 1),
        (column > 0) & (column < shape[-1] - 1),
        method=method_,
    )
    integer_lags = correlation.lags_rc[row, column].astype(correlation.values.dtype)
    offsets = integer_lags + jnp.stack((row_delta, column_delta), axis=-1)
    curvature = jnp.stack((row_curvature, column_curvature), axis=-1)
    epsilon = jnp.finfo(correlation.values.dtype).eps
    variances = jnp.stack(
        (
            jnp.where(row_fit, 1.0 / jnp.maximum(-row_curvature, epsilon), jnp.inf),
            jnp.where(column_fit, 1.0 / jnp.maximum(-column_curvature, epsilon), jnp.inf),
        ),
        axis=-1,
    )
    covariance = jnp.zeros(offsets.shape + (2,), dtype=offsets.dtype)
    covariance = covariance.at[..., 0, 0].set(variances[..., 0])
    covariance = covariance.at[..., 1, 1].set(variances[..., 1])
    offsets = jnp.where(peak_valid[..., None], offsets, 0.0)
    curvature = jnp.where(peak_valid[..., None], curvature, 0.0)
    return PeakBatch(offsets, values, peak_valid, curvature, covariance, method_)


__all__ = ["find_top_peaks"]
