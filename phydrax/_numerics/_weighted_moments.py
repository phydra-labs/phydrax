#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule


def _abs_square(values: Array) -> Array:
    return jnp.real(values * jnp.conj(values))


def _resolved_axes(
    axes: int | tuple[int, ...],
    ndim: int,
    /,
) -> tuple[int, ...]:
    raw = (axes,) if isinstance(axes, int) else tuple(axes)
    if not raw:
        raise ValueError("sample_axes must contain at least one axis.")
    resolved = tuple(axis + ndim if axis < 0 else axis for axis in raw)
    if any(axis < 0 or axis >= ndim for axis in resolved):
        raise ValueError("sample_axes contains an out-of-range axis.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("sample_axes must not contain duplicates.")
    return resolved


def _canonical_weights(
    log_weights: Array,
    sample_axes: int | tuple[int, ...],
    mask: Array | None,
    /,
) -> tuple[Array, Array, tuple[int, ...], tuple[int, ...]]:
    weights = jnp.asarray(log_weights, dtype=float)
    axes = _resolved_axes(sample_axes, weights.ndim)
    batch_axes = tuple(axis for axis in range(weights.ndim) if axis not in axes)
    permutation = axes + batch_axes
    transposed = jnp.transpose(weights, permutation)
    included = (
        jnp.ones(weights.shape, dtype=bool)
        if mask is None
        else jnp.broadcast_to(jnp.asarray(mask, dtype=bool), weights.shape)
    )
    included = jnp.transpose(included, permutation)
    sample_shape = tuple(weights.shape[axis] for axis in axes)
    batch_shape = tuple(weights.shape[axis] for axis in batch_axes)
    sample_count = prod(sample_shape)
    return (
        transposed.reshape((sample_count,) + batch_shape),
        included.reshape((sample_count,) + batch_shape),
        axes,
        batch_axes,
    )


def _canonical_values(
    values: Array,
    log_weights: Array,
    sample_axes: int | tuple[int, ...],
    mask: Array | None,
    /,
) -> tuple[Array, Array, Array]:
    values_ = jnp.asarray(values)
    weights_ = jnp.asarray(log_weights, dtype=float)
    if weights_.ndim > values_.ndim:
        raise ValueError("log_weights rank cannot exceed values rank.")
    if values_.shape[: weights_.ndim] != weights_.shape:
        raise ValueError(
            "Canonical values must begin with the complete log_weights shape."
        )
    weights, included, axes, batch_axes = _canonical_weights(weights_, sample_axes, mask)
    output_axes = tuple(range(weights_.ndim, values_.ndim))
    permutation = axes + batch_axes + output_axes
    transposed = jnp.transpose(values_, permutation)
    sample_count = weights.shape[0]
    batch_shape = weights.shape[1:]
    output_shape = values_.shape[weights_.ndim :]
    return (
        transposed.reshape((sample_count,) + batch_shape + output_shape),
        weights,
        included,
    )


def _expand_batch(values: Array, output_ndim: int, /) -> Array:
    return jnp.reshape(values, values.shape + (1,) * output_ndim)


class LogWeightedAccumulator(StrictModule):
    """Mergeable masked log-weighted moments with retained batch dimensions."""

    log_scale: Array
    weight_sum: Array
    squared_weight_sum: Array
    weighted_value_sum: Array
    weighted_abs_square_sum: Array
    squared_weight_value_sum: Array
    squared_weight_abs_square_sum: Array
    count: Array

    def __init__(
        self,
        *,
        log_scale: Array,
        weight_sum: Array,
        squared_weight_sum: Array,
        weighted_value_sum: Array,
        weighted_abs_square_sum: Array,
        squared_weight_value_sum: Array,
        squared_weight_abs_square_sum: Array,
        count: Array,
    ):
        self.log_scale = jnp.asarray(log_scale)
        self.weight_sum = jnp.asarray(weight_sum)
        self.squared_weight_sum = jnp.asarray(squared_weight_sum)
        self.weighted_value_sum = jnp.asarray(weighted_value_sum)
        self.weighted_abs_square_sum = jnp.asarray(weighted_abs_square_sum)
        self.squared_weight_value_sum = jnp.asarray(squared_weight_value_sum)
        self.squared_weight_abs_square_sum = jnp.asarray(squared_weight_abs_square_sum)
        self.count = jnp.asarray(count, dtype=jnp.int32)

    @classmethod
    def from_values(
        cls,
        values: Array,
        log_weights: Array,
        /,
        *,
        sample_axes: int | tuple[int, ...] = 0,
        mask: Array | None = None,
    ) -> "LogWeightedAccumulator":
        values_, log_weights_, included = _canonical_values(
            values, log_weights, sample_axes, mask
        )
        finite = jnp.isfinite(log_weights_)
        active = included & finite
        scale = jnp.max(
            jnp.where(active, log_weights_, -jnp.inf),
            axis=0,
            initial=-jnp.inf,
        )
        safe_scale = jnp.where(jnp.isfinite(scale), scale, 0.0)
        weights = jnp.where(active, jnp.exp(log_weights_ - safe_scale[None, ...]), 0.0)
        output_ndim = values_.ndim - weights.ndim
        weights_expanded = _expand_batch(weights, output_ndim)
        squared_weights = weights * weights
        squared_weights_expanded = _expand_batch(squared_weights, output_ndim)
        active_expanded = _expand_batch(active, output_ndim)
        safe_values = jnp.where(active_expanded, values_, 0)
        return cls(
            log_scale=scale,
            weight_sum=jnp.sum(weights, axis=0),
            squared_weight_sum=jnp.sum(squared_weights, axis=0),
            weighted_value_sum=jnp.sum(weights_expanded * safe_values, axis=0),
            weighted_abs_square_sum=jnp.sum(
                weights_expanded * _abs_square(safe_values), axis=0
            ),
            squared_weight_value_sum=jnp.sum(
                squared_weights_expanded * safe_values, axis=0
            ),
            squared_weight_abs_square_sum=jnp.sum(
                squared_weights_expanded * _abs_square(safe_values), axis=0
            ),
            count=jnp.sum(included, axis=0, dtype=jnp.int32),
        )

    @property
    def output_ndim(self) -> int:
        return self.weighted_value_sum.ndim - self.log_scale.ndim

    def merge(self, other: "LogWeightedAccumulator", /) -> "LogWeightedAccumulator":
        """Merge aligned independent chunks without losing relative weight scale."""
        if self.weighted_value_sum.shape != other.weighted_value_sum.shape:
            raise ValueError("Weighted accumulator value shapes must match.")
        if self.log_scale.shape != other.log_scale.shape:
            raise ValueError("Weighted accumulator batch shapes must match.")
        scale = jnp.maximum(self.log_scale, other.log_scale)
        left = jnp.where(
            jnp.isfinite(self.log_scale), jnp.exp(self.log_scale - scale), 0.0
        )
        right = jnp.where(
            jnp.isfinite(other.log_scale), jnp.exp(other.log_scale - scale), 0.0
        )
        left_values = _expand_batch(left, self.output_ndim)
        right_values = _expand_batch(right, self.output_ndim)
        return LogWeightedAccumulator(
            log_scale=scale,
            weight_sum=left * self.weight_sum + right * other.weight_sum,
            squared_weight_sum=(left * left) * self.squared_weight_sum
            + (right * right) * other.squared_weight_sum,
            weighted_value_sum=left_values * self.weighted_value_sum
            + right_values * other.weighted_value_sum,
            weighted_abs_square_sum=left_values * self.weighted_abs_square_sum
            + right_values * other.weighted_abs_square_sum,
            squared_weight_value_sum=(left_values * left_values)
            * self.squared_weight_value_sum
            + (right_values * right_values) * other.squared_weight_value_sum,
            squared_weight_abs_square_sum=(left_values * left_values)
            * self.squared_weight_abs_square_sum
            + (right_values * right_values) * other.squared_weight_abs_square_sum,
            count=self.count + other.count,
        )

    @property
    def normalized_mean(self) -> Array:
        denominator = jnp.maximum(self.weight_sum, jnp.finfo(float).tiny)
        return self.weighted_value_sum / _expand_batch(denominator, self.output_ndim)

    @property
    def raw_mean(self) -> Array:
        count = jnp.maximum(self.count, 1)
        scale = _expand_batch(jnp.exp(self.log_scale), self.output_ndim)
        return scale * self.weighted_value_sum / _expand_batch(count, self.output_ndim)

    @property
    def raw_normalizer(self) -> Array:
        """Sample mean of the unnormalized nonnegative weights."""
        return jnp.exp(self.log_scale) * self.weight_sum / jnp.maximum(self.count, 1)

    @property
    def raw_normalizer_standard_error(self) -> Array:
        """IID standard error of the sample-mean normalization estimate."""
        count = jnp.asarray(self.count)
        mean = self.raw_normalizer
        sum_square = jnp.exp(2.0 * self.log_scale) * self.squared_weight_sum
        sample_variance = jnp.where(
            count > 1,
            jnp.maximum(sum_square - count * mean * mean, 0.0) / (count - 1),
            jnp.inf,
        )
        return jnp.sqrt(sample_variance / jnp.maximum(count, 1))

    @property
    def weight_ess(self) -> Array:
        denominator = jnp.maximum(self.squared_weight_sum, jnp.finfo(float).tiny)
        return self.weight_sum * self.weight_sum / denominator

    @property
    def relative_weight_ess(self) -> Array:
        return self.weight_ess / jnp.maximum(self.count, 1)

    @property
    def maximum_normalized_weight(self) -> Array:
        return jnp.where(
            self.weight_sum > 0.0,
            1.0 / jnp.maximum(self.weight_sum, jnp.finfo(float).tiny),
            0.0,
        )

    @property
    def normalized_standard_error(self) -> Array:
        mean = self.normalized_mean
        squared_weight_sum = _expand_batch(self.squared_weight_sum, self.output_ndim)
        numerator = (
            self.squared_weight_abs_square_sum
            - 2.0 * jnp.real(jnp.conj(mean) * self.squared_weight_value_sum)
            + _abs_square(mean) * squared_weight_sum
        )
        denominator = self.weight_sum * self.weight_sum - self.squared_weight_sum
        denominator_expanded = _expand_batch(denominator, self.output_ndim)
        variance = jnp.where(
            denominator_expanded > 0.0,
            jnp.maximum(numerator, 0.0) / denominator_expanded,
            jnp.inf,
        )
        return jnp.sqrt(variance)

    @property
    def raw_standard_error(self) -> Array:
        count = jnp.asarray(self.count)
        count_expanded = _expand_batch(count, self.output_ndim)
        raw_mean = self.raw_mean
        scale = _expand_batch(jnp.exp(2.0 * self.log_scale), self.output_ndim)
        sum_square = scale * self.squared_weight_abs_square_sum
        sample_variance = jnp.where(
            count_expanded > 1,
            jnp.maximum(sum_square - count_expanded * _abs_square(raw_mean), 0.0)
            / (count_expanded - 1),
            jnp.inf,
        )
        return jnp.sqrt(sample_variance / jnp.maximum(count_expanded, 1))


class WeightedMomentsDiagnostics(StrictModule):
    """Stable weighted diagnostics retained over every non-sample dimension."""

    weight_ess: Array
    relative_weight_ess: Array
    coefficient_of_variation: Array
    maximum_normalized_weight: Array
    entropy: Array
    log_weight_range: Array
    finite_count: Array

    def __init__(
        self,
        *,
        weight_ess: Array,
        relative_weight_ess: Array,
        coefficient_of_variation: Array,
        maximum_normalized_weight: Array,
        entropy: Array,
        log_weight_range: Array,
        finite_count: Array,
    ):
        self.weight_ess = jnp.asarray(weight_ess)
        self.relative_weight_ess = jnp.asarray(relative_weight_ess)
        self.coefficient_of_variation = jnp.asarray(coefficient_of_variation)
        self.maximum_normalized_weight = jnp.asarray(maximum_normalized_weight)
        self.entropy = jnp.asarray(entropy)
        self.log_weight_range = jnp.asarray(log_weight_range)
        self.finite_count = jnp.asarray(finite_count, dtype=jnp.int32)


def weighted_diagnostics(
    accumulator: LogWeightedAccumulator,
    log_weights: Array,
    /,
    *,
    sample_axes: int | tuple[int, ...] = 0,
    mask: Array | None = None,
) -> WeightedMomentsDiagnostics:
    log_weights_, included, _, _ = _canonical_weights(log_weights, sample_axes, mask)
    finite = jnp.isfinite(log_weights_)
    active = included & finite
    finite_count = jnp.sum(active, axis=0, dtype=jnp.int32)
    safe_scale = jnp.where(
        jnp.isfinite(accumulator.log_scale), accumulator.log_scale, 0.0
    )
    normalized = jnp.where(
        active,
        jnp.exp(log_weights_ - safe_scale[None, ...])
        / jnp.maximum(accumulator.weight_sum[None, ...], jnp.finfo(float).tiny),
        0.0,
    )
    entropy = -jnp.sum(
        jnp.where(normalized > 0.0, normalized * jnp.log(normalized), 0.0),
        axis=0,
    )
    finite_max = jnp.max(
        jnp.where(active, log_weights_, -jnp.inf), axis=0, initial=-jnp.inf
    )
    finite_min = jnp.min(
        jnp.where(active, log_weights_, jnp.inf), axis=0, initial=jnp.inf
    )
    weight_ess = accumulator.weight_ess
    coefficient = jnp.sqrt(
        jnp.maximum(
            jnp.asarray(accumulator.count)
            / jnp.maximum(weight_ess, jnp.finfo(float).tiny)
            - 1.0,
            0.0,
        )
    )
    return WeightedMomentsDiagnostics(
        weight_ess=weight_ess,
        relative_weight_ess=accumulator.relative_weight_ess,
        coefficient_of_variation=coefficient,
        maximum_normalized_weight=jnp.max(normalized, axis=0),
        entropy=entropy,
        log_weight_range=jnp.where(finite_count > 0, finite_max - finite_min, jnp.inf),
        finite_count=finite_count,
    )


__all__ = [
    "LogWeightedAccumulator",
    "WeightedMomentsDiagnostics",
    "weighted_diagnostics",
]
