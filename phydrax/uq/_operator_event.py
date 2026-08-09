#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from ..nn.operator.data import OperatorOutputSpec, OperatorPrediction
from ._operator import (
    _expected_output_shape,
    _output_mask,
    _output_weights,
    _physical_dims,
    _queries_equal,
    _select_prediction_field,
    OperatorPredictiveField,
)


Measure = Literal["quadrature", "uniform"]
OperatorReduction = Literal["none", "mean", "sum"]


def require_predictive(prediction: OperatorPredictiveField, /) -> None:
    if not isinstance(prediction, OperatorPredictiveField):
        raise TypeError("prediction must be an OperatorPredictiveField.")


def sample_case_event(prediction: OperatorPredictiveField, /) -> Array:
    """Return predictive draws with shape ``(sample, case, event)``."""
    require_predictive(prediction)
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
    valid = prediction.predictive.valid
    if valid is not None:
        invalid = ~jnp.all(jnp.asarray(valid.data))
        message = (
            "Operator proper scores require every requested predictive draw to be valid."
        )
        if isinstance(invalid, jax_core.Tracer):
            data = eqx.error_if(data, invalid, message)
        elif bool(invalid):
            raise ValueError(message)
    if permutation != tuple(range(data.ndim)):
        data = jnp.transpose(data, permutation)
    sample_count = 1
    for dim in sample_dims:
        sample_count *= int(prediction.predictive.samples.data.shape[dims.index(dim)])
    return data.reshape((sample_count, case_count(prediction.case_shape), -1))


def operator_target_values(
    target: ArrayLike | OperatorPrediction,
    /,
    *,
    query,
    output_spec: OperatorOutputSpec,
    case_axes: tuple[str, ...],
    case_shape: tuple[int, ...],
    field_name: str,
) -> Array:
    """Validate and flatten one operator target contract."""
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
    invalid = jnp.any(~jnp.isfinite(values) & mask)
    message = "Operator target must be finite at every valid query location."
    if isinstance(invalid, jax_core.Tracer):
        values = eqx.error_if(values, invalid, message)
    elif bool(invalid):
        raise ValueError(message)
    return jnp.where(mask, values, 0.0)


def measure_weights(
    query,
    output_spec: OperatorOutputSpec,
    case_shape: tuple[int, ...],
    /,
    *,
    measure: Measure,
) -> Array:
    """Return physical pointwise reduction weights."""
    validate_measure(measure)
    if measure == "uniform":
        return jnp.ones(
            _expected_output_shape(query, output_spec, case_shape),
            dtype=float,
        )
    return _output_weights(query, output_spec, case_shape, normalized=False)


def event_weights(
    query,
    output_spec: OperatorOutputSpec,
    case_shape: tuple[int, ...],
    /,
    *,
    measure: Measure,
) -> Array:
    """Return normalized physical weights for whole-event Euclidean geometry."""
    validate_measure(measure)
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


def weighted_case_mean(values: Array, mask: Array, weights: Array, /) -> Array:
    effective = jnp.where(mask, weights, 0.0)
    denominator = jnp.sum(effective, axis=-1)
    denominator = eqx.error_if(
        denominator,
        jnp.any(denominator <= 0.0),
        "Every operator case must contain positive physical measure.",
    )
    return jnp.sum(jnp.where(mask, values * effective, 0.0), axis=-1) / denominator


def reduce_cases(
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


def case_count(case_shape: tuple[int, ...], /) -> int:
    count = 1
    for size in case_shape:
        count *= int(size)
    return count


def validate_measure(measure: str, /) -> None:
    if measure not in ("quadrature", "uniform"):
        raise ValueError("measure must be 'quadrature' or 'uniform'.")


__all__ = [
    "Measure",
    "OperatorReduction",
    "case_count",
    "event_weights",
    "measure_weights",
    "operator_target_values",
    "reduce_cases",
    "require_predictive",
    "sample_case_event",
    "validate_measure",
    "weighted_case_mean",
]
