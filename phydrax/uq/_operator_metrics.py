#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..nn.models.core._operator import OperatorOutputSpec, OperatorPrediction
from ._metrics import energy_score, ensemble_crps
from ._operator import (
    _expected_output_shape,
    _output_mask,
    _output_weights,
    _physical_dims,
    _queries_equal,
    _select_prediction_field,
    OperatorPredictionInterval,
    OperatorPredictiveField,
)


Measure = Literal["quadrature", "uniform"]
OperatorReduction = Literal["none", "mean", "sum"]


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


def _require_predictive(prediction: OperatorPredictiveField, /) -> None:
    if not isinstance(prediction, OperatorPredictiveField):
        raise TypeError("prediction must be an OperatorPredictiveField.")
    valid = prediction.predictive.valid
    if valid is not None and not bool(jnp.all(jnp.asarray(valid.data))):
        raise ValueError(
            "Operator proper scores require every requested predictive draw to be valid."
        )


def _sample_case_event(prediction: OperatorPredictiveField, /) -> Array:
    sample_dims = tuple(axis.dim for axis in prediction.predictive.sample_axes)
    physical_dims = _physical_dims(
        prediction.query,
        prediction.output_spec,
        prediction.case_axes,
    )
    case_dims = prediction.case_axes
    event_dims = physical_dims[len(case_dims) :]
    dims = prediction.predictive.samples.dims
    ordered_dims = sample_dims + case_dims + event_dims
    permutation = tuple(dims.index(dim) for dim in ordered_dims)
    data = jnp.asarray(prediction.predictive.samples.data)
    if permutation != tuple(range(data.ndim)):
        data = jnp.transpose(data, permutation)
    sample_count = 1
    for dim in sample_dims:
        sample_count *= int(prediction.predictive.samples.data.shape[dims.index(dim)])
    return data.reshape((sample_count, _case_count(prediction.case_shape), -1))


def _operator_target_values(
    target: ArrayLike | OperatorPrediction,
    /,
    *,
    query,
    output_spec: OperatorOutputSpec,
    case_axes: tuple[str, ...],
    case_shape: tuple[int, ...],
    field_name: str,
) -> Array:
    if isinstance(target, OperatorPrediction):
        _, field, target_query = _select_prediction_field(target, field_name)
        if (
            target.case_axes != case_axes
            or target.case_shape != case_shape
            or field.spec.channels != output_spec.channels
            or field.spec.component_names != output_spec.component_names
            or not _queries_equal(target_query, query)
        ):
            raise ValueError("Operator target does not match the prediction contract.")
        values = jnp.asarray(field.values, dtype=float)
    else:
        values = jnp.asarray(target, dtype=float)
    expected = _expected_output_shape(query, output_spec, case_shape)
    if values.shape != expected:
        raise ValueError(
            f"Operator target must have shape {expected}; got {values.shape}."
        )
    mask = _output_mask(query, output_spec, case_shape)
    if bool(jnp.any(~jnp.isfinite(values) & mask)):
        raise ValueError("Operator target must be finite at every valid query location.")
    return jnp.where(mask, values, 0.0)


def _measure_weights(
    query,
    output_spec: OperatorOutputSpec,
    case_shape: tuple[int, ...],
    /,
    *,
    measure: Measure,
) -> Array:
    _validate_measure(measure)
    if measure == "uniform":
        return jnp.ones(
            _expected_output_shape(query, output_spec, case_shape),
            dtype=float,
        )
    return _output_weights(query, output_spec, case_shape, normalized=False)


def _energy_weights(
    query,
    output_spec: OperatorOutputSpec,
    case_shape: tuple[int, ...],
    /,
    *,
    measure: Measure,
) -> Array:
    _validate_measure(measure)
    query_mask = query.mask_array(case_shape=case_shape)
    if measure == "quadrature":
        weights = query.weights(case_shape=case_shape, normalized=True)
    else:
        query_rank = len(query.sample_shape)
        axes = tuple(range(len(case_shape), len(case_shape) + query_rank))
        denominator = jnp.sum(query_mask, axis=axes, keepdims=True)
        weights = jnp.where(
            query_mask,
            1.0 / jnp.maximum(denominator, 1.0),
            0.0,
        )
    weights = jnp.where(query_mask, weights, 0.0)
    if output_spec.channels != "scalar":
        weights = jnp.broadcast_to(
            weights[..., None],
            weights.shape + output_spec.channel_shape,
        )
    return weights


def _weighted_case_mean(values: Array, mask: Array, weights: Array, /) -> Array:
    effective = jnp.where(mask, weights, 0.0)
    denominator = jnp.sum(effective, axis=-1)
    if bool(jnp.any(denominator <= 0.0)):
        raise ValueError("Every operator case must contain positive physical measure.")
    return jnp.sum(jnp.where(mask, values * effective, 0.0), axis=-1) / denominator


def _reduce_cases(
    values: Array,
    case_shape: tuple[int, ...],
    /,
    *,
    reduction: OperatorReduction,
) -> Array:
    shaped = jnp.asarray(values).reshape(case_shape)
    if reduction == "none":
        return shaped
    if reduction == "mean":
        return jnp.mean(shaped)
    if reduction == "sum":
        return jnp.sum(shaped)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def _case_count(case_shape: tuple[int, ...], /) -> int:
    count = 1
    for size in case_shape:
        count *= int(size)
    return count


def _validate_measure(measure: str, /) -> None:
    if measure not in ("quadrature", "uniform"):
        raise ValueError("measure must be 'quadrature' or 'uniform'.")


__all__ = [
    "operator_energy_score",
    "operator_ensemble_crps",
    "operator_interval_coverage",
    "operator_interval_width",
]
