#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._base import (
    _nan_where_invalid,
    _prepare_pair,
    _result,
    METRIC_EMPTY,
    METRIC_INVALID_INPUT,
    METRIC_SUCCESS,
    MetricResult,
)
from ._classification import _probability_inputs, _require_integer_labels


CalibrationNorm = Literal["l1", "l2"]


class CalibrationResult(StrictModule):
    """Calibration error plus fixed-shape bin diagnostics."""

    value: Array
    bin_weight: Array
    mean_probability: Array
    empirical_frequency: Array
    valid: Array
    status: Array
    effective_weight: Array
    hard_binning: bool = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        /,
        *,
        bin_weight: ArrayLike,
        mean_probability: ArrayLike,
        empirical_frequency: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        effective_weight: ArrayLike,
        hard_binning: bool,
    ):
        self.value = jnp.asarray(value)
        self.bin_weight = jnp.asarray(bin_weight)
        self.mean_probability = jnp.asarray(mean_probability)
        self.empirical_frequency = jnp.asarray(empirical_frequency)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.effective_weight = jnp.asarray(effective_weight)
        self.hard_binning = bool(hard_binning)


def _binary_probability_inputs(
    y_true: ArrayLike,
    probability: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    metric: str,
    from_logits: bool,
):
    labels_raw = jnp.asarray(y_true)
    probability_raw = jnp.asarray(probability)
    _require_integer_labels(labels_raw, metric)
    labels, values, weights, active, invalid, axis = _prepare_pair(
        labels_raw,
        probability_raw,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric=metric,
        allow_complex=False,
    )
    if axis != labels.ndim - 1:
        raise ValueError("Calibration inputs must have case_shape + (sample,) axes.")
    if from_logits:
        values = jax.nn.sigmoid(values)
        probability_valid = jnp.isfinite(probability_raw)
    else:
        probability_valid = jnp.isfinite(values) & (values >= 0.0) & (values <= 1.0)
    label_valid = (labels == 0) | (labels == 1)
    invalid = invalid | jnp.any(active & ~(label_valid & probability_valid), axis=-1)
    active = active & label_valid & probability_valid
    weights = jnp.where(active, weights, 0.0)
    values = jnp.where(active, values, 0.0)
    labels = jnp.where(active, labels, 0).astype(values.dtype)
    mass = jnp.sum(weights, axis=-1)
    return labels, values, weights, active, invalid, mass


def _calibration_result(
    *,
    score: Array,
    bin_weight: Array,
    mean_probability: Array,
    empirical_frequency: Array,
    invalid: Array,
    mass: Array,
    hard_binning: bool,
) -> CalibrationResult:
    valid = ~(invalid | (mass <= 0.0))
    status = jnp.where(
        invalid,
        METRIC_INVALID_INPUT,
        jnp.where(mass <= 0.0, METRIC_EMPTY, METRIC_SUCCESS),
    )
    return CalibrationResult(
        _nan_where_invalid(score, valid),
        bin_weight=bin_weight,
        mean_probability=mean_probability,
        empirical_frequency=empirical_frequency,
        valid=valid,
        status=status,
        effective_weight=mass,
        hard_binning=hard_binning,
    )


def _hard_bins(
    labels: Array,
    probability: Array,
    weights: Array,
    *,
    num_bins: int,
) -> tuple[Array, Array, Array]:
    indices = jnp.minimum((probability * num_bins).astype(jnp.int32), num_bins - 1)
    membership = jax.nn.one_hot(indices, num_bins, dtype=weights.dtype)
    weighted_membership = weights[..., :, None] * membership
    bin_weight = jnp.sum(weighted_membership, axis=-2)
    mean_probability = jnp.sum(
        weighted_membership * probability[..., :, None], axis=-2
    ) / jnp.where(bin_weight > 0.0, bin_weight, 1.0)
    empirical_frequency = jnp.sum(
        weighted_membership * labels[..., :, None], axis=-2
    ) / jnp.where(bin_weight > 0.0, bin_weight, 1.0)
    return bin_weight, mean_probability, empirical_frequency


def expected_calibration_error(
    y_true: ArrayLike,
    probability: ArrayLike,
    /,
    *,
    num_bins: int = 10,
    norm: CalibrationNorm = "l1",
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> CalibrationResult:
    """Exact binary ECE under hard equal-width bins."""
    bins = int(num_bins)
    if bins <= 0:
        raise ValueError("num_bins must be positive.")
    if norm not in {"l1", "l2"}:
        raise ValueError("norm must be 'l1' or 'l2'.")
    labels, values, weights, _, invalid, mass = _binary_probability_inputs(
        y_true,
        probability,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="expected_calibration_error",
        from_logits=from_logits,
    )
    bin_weight, mean_probability, frequency = _hard_bins(
        labels, values, weights, num_bins=bins
    )
    gap = jnp.abs(mean_probability - frequency)
    if norm == "l1":
        score = jnp.sum(bin_weight * gap, axis=-1) / jnp.where(mass > 0.0, mass, 1.0)
    else:
        score = jnp.sqrt(
            jnp.sum(bin_weight * gap**2, axis=-1) / jnp.where(mass > 0.0, mass, 1.0)
        )
    return _calibration_result(
        score=score,
        bin_weight=bin_weight,
        mean_probability=mean_probability,
        empirical_frequency=frequency,
        invalid=invalid,
        mass=mass,
        hard_binning=True,
    )


def maximum_calibration_error(
    y_true: ArrayLike,
    probability: ArrayLike,
    /,
    *,
    num_bins: int = 10,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> CalibrationResult:
    """Maximum occupied-bin calibration gap under exact hard bins."""
    bins = int(num_bins)
    if bins <= 0:
        raise ValueError("num_bins must be positive.")
    labels, values, weights, _, invalid, mass = _binary_probability_inputs(
        y_true,
        probability,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="maximum_calibration_error",
        from_logits=from_logits,
    )
    bin_weight, mean_probability, frequency = _hard_bins(
        labels, values, weights, num_bins=bins
    )
    gap = jnp.abs(mean_probability - frequency)
    score = jnp.max(jnp.where(bin_weight > 0.0, gap, 0.0), axis=-1)
    return _calibration_result(
        score=score,
        bin_weight=bin_weight,
        mean_probability=mean_probability,
        empirical_frequency=frequency,
        invalid=invalid,
        mass=mass,
        hard_binning=True,
    )


def smooth_expected_calibration_error(
    y_true: ArrayLike,
    probability: ArrayLike,
    /,
    *,
    num_bins: int = 10,
    bin_temperature: float = 0.1,
    gap_smoothing: float = 1e-4,
    norm: CalibrationNorm = "l1",
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> CalibrationResult:
    """Gaussian soft-bin calibration surrogate, smooth in probabilities."""
    bins = int(num_bins)
    if bins <= 0:
        raise ValueError("num_bins must be positive.")
    if bin_temperature <= 0.0 or gap_smoothing <= 0.0:
        raise ValueError("bin_temperature and gap_smoothing must be positive.")
    if norm not in {"l1", "l2"}:
        raise ValueError("norm must be 'l1' or 'l2'.")
    labels, values, weights, _, invalid, mass = _binary_probability_inputs(
        y_true,
        probability,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_expected_calibration_error",
        from_logits=from_logits,
    )
    centers = (jnp.arange(bins, dtype=values.dtype) + 0.5) / float(bins)
    logits = -0.5 * ((values[..., :, None] - centers) / float(bin_temperature)) ** 2
    membership = jax.nn.softmax(logits, axis=-1)
    weighted_membership = weights[..., :, None] * membership
    bin_weight = jnp.sum(weighted_membership, axis=-2)
    mean_probability = jnp.sum(
        weighted_membership * values[..., :, None], axis=-2
    ) / jnp.where(bin_weight > 0.0, bin_weight, 1.0)
    frequency = jnp.sum(weighted_membership * labels[..., :, None], axis=-2) / jnp.where(
        bin_weight > 0.0, bin_weight, 1.0
    )
    difference = mean_probability - frequency
    if norm == "l1":
        gap = jnp.sqrt(difference**2 + float(gap_smoothing) ** 2) - float(gap_smoothing)
        score = jnp.sum(bin_weight * gap, axis=-1) / jnp.where(mass > 0.0, mass, 1.0)
    else:
        score = jnp.sqrt(
            jnp.sum(bin_weight * difference**2, axis=-1)
            / jnp.where(mass > 0.0, mass, 1.0)
            + float(gap_smoothing) ** 2
        ) - float(gap_smoothing)
    return _calibration_result(
        score=score,
        bin_weight=bin_weight,
        mean_probability=mean_probability,
        empirical_frequency=frequency,
        invalid=invalid,
        mass=mass,
        hard_binning=False,
    )


def smooth_maximum_calibration_error(
    y_true: ArrayLike,
    probability: ArrayLike,
    /,
    *,
    num_bins: int = 10,
    bin_temperature: float = 0.1,
    maximum_temperature: float = 0.05,
    gap_smoothing: float = 1e-4,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> CalibrationResult:
    """Soft-bin, log-mean-exp maximum-calibration-error surrogate."""
    if maximum_temperature <= 0.0:
        raise ValueError("maximum_temperature must be positive.")
    base = smooth_expected_calibration_error(
        y_true,
        probability,
        num_bins=num_bins,
        bin_temperature=bin_temperature,
        gap_smoothing=gap_smoothing,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        from_logits=from_logits,
    )
    difference = base.mean_probability - base.empirical_frequency
    gap = jnp.sqrt(difference**2 + float(gap_smoothing) ** 2) - float(gap_smoothing)
    normalized_weight = base.bin_weight / jnp.where(
        base.effective_weight[..., None] > 0.0,
        base.effective_weight[..., None],
        1.0,
    )
    fallback_weight = jax.nn.one_hot(
        jnp.zeros(base.valid.shape, dtype=jnp.int32),
        base.bin_weight.shape[-1],
        dtype=base.bin_weight.dtype,
    )
    normalized_weight = jnp.where(
        base.valid[..., None], normalized_weight, fallback_weight
    )
    score = float(maximum_temperature) * logsumexp(
        gap / float(maximum_temperature), axis=-1, b=normalized_weight
    )
    return CalibrationResult(
        _nan_where_invalid(score, base.valid),
        bin_weight=base.bin_weight,
        mean_probability=base.mean_probability,
        empirical_frequency=base.empirical_frequency,
        valid=base.valid,
        status=base.status,
        effective_weight=base.effective_weight,
        hard_binning=False,
    )


def classwise_expected_calibration_error(
    y_true: ArrayLike,
    probability: ArrayLike,
    /,
    *,
    num_bins: int = 10,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> MetricResult:
    """Mean one-vs-rest hard-bin ECE across classes."""
    bins = int(num_bins)
    if bins <= 0:
        raise ValueError("num_bins must be positive.")
    labels, probabilities, weights, active, invalid, classes = _probability_inputs(
        y_true,
        probability,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="classwise_expected_calibration_error",
        from_logits=from_logits,
    )
    targets = jax.nn.one_hot(labels, classes, dtype=probabilities.dtype)
    indices = jnp.minimum((probabilities * bins).astype(jnp.int32), bins - 1)
    membership = jax.nn.one_hot(indices, bins, dtype=weights.dtype)
    weighted_membership = weights[..., :, None, None] * membership
    bin_weight = jnp.sum(weighted_membership, axis=-3)
    mean_probability = jnp.sum(
        weighted_membership * probabilities[..., :, :, None], axis=-3
    ) / jnp.where(bin_weight > 0.0, bin_weight, 1.0)
    frequency = jnp.sum(
        weighted_membership * targets[..., :, :, None], axis=-3
    ) / jnp.where(bin_weight > 0.0, bin_weight, 1.0)
    per_class = jnp.sum(
        bin_weight * jnp.abs(mean_probability - frequency), axis=-1
    ) / jnp.where(jnp.sum(bin_weight, axis=-1) > 0.0, jnp.sum(bin_weight, axis=-1), 1.0)
    value = jnp.mean(per_class, axis=-1)
    mass = jnp.sum(jnp.where(active, weights, 0.0), axis=-1)
    return _result(value, invalid=invalid, effective_weight=mass)


__all__ = [
    "CalibrationNorm",
    "CalibrationResult",
    "classwise_expected_calibration_error",
    "expected_calibration_error",
    "maximum_calibration_error",
    "smooth_expected_calibration_error",
    "smooth_maximum_calibration_error",
]
