#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..nn.operator.data import OperatorPrediction
from ._metrics import energy_distance, energy_score, ensemble_crps
from ._operator import (
    _output_mask,
    _queries_equal,
    _select_prediction_field,
    OperatorPredictionInterval,
    OperatorPredictiveField,
)
from ._operator_event import (
    case_count as _case_count,
    event_weights as _energy_weights,
    Measure,
    measure_weights as _measure_weights,
    operator_target_values as _operator_target_values,
    OperatorReduction,
    reduce_cases as _reduce_cases,
    require_predictive as _require_predictive,
    sample_case_event as _sample_case_event,
    weighted_case_mean as _weighted_case_mean,
)


def operator_ensemble_crps(
    prediction: OperatorPredictiveField,
    target: ArrayLike | OperatorPrediction,
    /,
    *,
    measure: Measure = "quadrature",
    reduction: OperatorReduction = "mean",
) -> Array:
    """Marginal empirical CRPS reduced over each physical operator case."""
    _require_predictive(prediction)
    samples = _sample_case_event(prediction)
    target_values = _operator_target_values(
        target,
        query=prediction.query,
        output_spec=prediction.output_spec,
        case_axes=prediction.case_axes,
        case_shape=prediction.case_shape,
        field_name=prediction.field_name,
    ).reshape((_case_count(prediction.case_shape), -1))
    pointwise = ensemble_crps(samples, target_values, sample_axis=0)
    mask = prediction.output_mask().reshape(target_values.shape)
    weights = _measure_weights(
        prediction.query,
        prediction.output_spec,
        prediction.case_shape,
        measure=measure,
    ).reshape(target_values.shape)
    per_case = _weighted_case_mean(pointwise, mask, weights)
    return _reduce_cases(per_case, prediction.case_shape, reduction=reduction)


def operator_energy_score(
    prediction: OperatorPredictiveField,
    target: ArrayLike | OperatorPrediction,
    /,
    *,
    measure: Measure = "quadrature",
    beta: float = 1.0,
    chunk_size: int | None = None,
    reduction: OperatorReduction = "mean",
) -> Array:
    """Whole-field energy score with one multivariate event per physical case."""
    _require_predictive(prediction)
    samples = _sample_case_event(prediction)
    count = _case_count(prediction.case_shape)
    target_values = _operator_target_values(
        target,
        query=prediction.query,
        output_spec=prediction.output_spec,
        case_axes=prediction.case_axes,
        case_shape=prediction.case_shape,
        field_name=prediction.field_name,
    ).reshape((count, -1))
    mask = prediction.output_mask().reshape(target_values.shape)
    weights = _energy_weights(
        prediction.query,
        prediction.output_spec,
        prediction.case_shape,
        measure=measure,
    ).reshape(target_values.shape)
    scale = jnp.sqrt(jnp.where(mask, weights, 0.0))
    scaled_samples = samples * scale[None, ...]
    scaled_target = target_values * scale
    per_case = jnp.stack(
        tuple(
            energy_score(
                scaled_samples[:, index, :],
                scaled_target[index],
                sample_axis=0,
                beta=beta,
                chunk_size=chunk_size,
            )
            for index in range(count)
        )
    )
    return _reduce_cases(per_case, prediction.case_shape, reduction=reduction)


def operator_ensemble_energy_distance(
    left: OperatorPredictiveField,
    right: OperatorPredictiveField,
    /,
    *,
    measure: Measure = "quadrature",
    beta: float = 1.0,
    chunk_size: int | None = None,
    reduction: OperatorReduction = "mean",
) -> Array:
    """Whole-field energy distance between two operator ensembles."""
    _require_predictive(left)
    _require_predictive(right)
    if (
        left.case_axes != right.case_axes
        or left.case_shape != right.case_shape
        or left.output_spec.channels != right.output_spec.channels
        or left.output_spec.component_names != right.output_spec.component_names
        or not _queries_equal(left.query, right.query)
    ):
        raise ValueError("Operator ensembles must share one physical output contract.")
    left_samples = _sample_case_event(left)
    right_samples = _sample_case_event(right)
    count = _case_count(left.case_shape)
    mask = left.output_mask().reshape((count, -1))
    weights = _energy_weights(
        left.query,
        left.output_spec,
        left.case_shape,
        measure=measure,
    ).reshape((count, -1))
    scale = jnp.sqrt(jnp.where(mask, weights, 0.0))
    scaled_left = left_samples * scale[None, ...]
    scaled_right = right_samples * scale[None, ...]
    per_case = jnp.stack(
        tuple(
            energy_distance(
                scaled_left[:, index, :],
                scaled_right[:, index, :],
                sample_axis=0,
                beta=beta,
                chunk_size=chunk_size,
            )
            for index in range(count)
        )
    )
    return _reduce_cases(per_case, left.case_shape, reduction=reduction)


def operator_interval_coverage(
    interval: OperatorPredictionInterval,
    target: ArrayLike | OperatorPrediction,
    /,
    *,
    field_name: str,
    mode: Literal["pointwise", "simultaneous"] = "pointwise",
    measure: Measure = "quadrature",
    reduction: OperatorReduction = "mean",
) -> Array:
    """Pointwise fraction or whole-case simultaneous interval coverage."""
    if not isinstance(interval, OperatorPredictionInterval):
        raise TypeError("interval must be an OperatorPredictionInterval.")
    selected_name, lower_field, query = _select_prediction_field(
        interval.lower,
        field_name,
    )
    _, upper_field, upper_query = _select_prediction_field(
        interval.upper,
        selected_name,
    )
    output_spec = lower_field.spec
    if not _queries_equal(query, upper_query):
        raise ValueError("Operator interval query contracts differ.")
    target_values = _operator_target_values(
        target,
        query=query,
        output_spec=output_spec,
        case_axes=interval.lower.case_axes,
        case_shape=interval.lower.case_shape,
        field_name=selected_name,
    )
    mask = _output_mask(query, output_spec, interval.lower.case_shape)
    covered = (
        (target_values >= jnp.asarray(lower_field.values))
        & (target_values <= jnp.asarray(upper_field.values))
        & mask
    )
    count = _case_count(interval.lower.case_shape)
    covered_flat = covered.reshape((count, -1))
    mask_flat = mask.reshape((count, -1))
    if mode == "simultaneous":
        per_case = jnp.all(covered_flat | ~mask_flat, axis=-1).astype(float)
    elif mode == "pointwise":
        weights = _measure_weights(
            query,
            output_spec,
            interval.lower.case_shape,
            measure=measure,
        ).reshape((count, -1))
        per_case = _weighted_case_mean(covered_flat.astype(float), mask_flat, weights)
    else:
        raise ValueError("mode must be 'pointwise' or 'simultaneous'.")
    return _reduce_cases(per_case, interval.lower.case_shape, reduction=reduction)


def operator_interval_width(
    interval: OperatorPredictionInterval,
    /,
    *,
    field_name: str,
    measure: Measure = "quadrature",
    reduction: OperatorReduction = "mean",
) -> Array:
    """Mean physical interval width for each operator case."""
    if not isinstance(interval, OperatorPredictionInterval):
        raise TypeError("interval must be an OperatorPredictionInterval.")
    selected_name, lower_field, query = _select_prediction_field(
        interval.lower,
        field_name,
    )
    _, upper_field, upper_query = _select_prediction_field(
        interval.upper,
        selected_name,
    )
    if not _queries_equal(query, upper_query):
        raise ValueError("Operator interval query contracts differ.")
    output_spec = lower_field.spec
    width = jnp.asarray(upper_field.values) - jnp.asarray(lower_field.values)
    mask = _output_mask(query, output_spec, interval.lower.case_shape)
    count = _case_count(interval.lower.case_shape)
    weights = _measure_weights(
        query,
        output_spec,
        interval.lower.case_shape,
        measure=measure,
    )
    per_case = _weighted_case_mean(
        width.reshape((count, -1)),
        mask.reshape((count, -1)),
        weights.reshape((count, -1)),
    )
    return _reduce_cases(per_case, interval.lower.case_shape, reduction=reduction)


__all__ = [
    "operator_energy_score",
    "operator_ensemble_energy_distance",
    "operator_ensemble_crps",
    "operator_interval_coverage",
    "operator_interval_width",
]
