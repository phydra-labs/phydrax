#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ..nn.models.core._operator import (
    FunctionSamples,
    OperatorBatch,
    OperatorOutputSpec,
    OperatorPrediction,
)
from ._likelihoods import AbstractLikelihood, GaussianLikelihood
from ._operator import (
    _broadcast_named,
    _output_mask,
    _physical_dims,
    _queries_equal,
    _select_prediction_field,
)
from ._posterior_terms import AbstractPosteriorTerm


class FixedOperatorObservationLikelihood(AbstractPosteriorTerm):
    """Normalized finite-observation likelihood for one fixed operator batch."""

    batch: OperatorBatch
    target: Array
    likelihood: AbstractLikelihood
    output_spec: OperatorOutputSpec
    observation_mask: Array
    predict_fn: Callable[[PyTree[Any]], OperatorPrediction] = eqx.field(static=True)
    parameters_fn: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]] | None = (
        eqx.field(static=True)
    )
    case_count: int = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    query_name: str = eqx.field(static=True)

    def __init__(
        self,
        predict: Callable[[PyTree[Any]], OperatorPrediction],
        batch: OperatorBatch,
        target: ArrayLike | cx.Field | OperatorPrediction,
        likelihood: AbstractLikelihood,
        /,
        *,
        output_spec: OperatorOutputSpec,
        field_name: str,
        query_name: str,
        observation_mask: ArrayLike | cx.Field | None = None,
        parameters: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]]
        | None = None,
        label: str = "operator_observation",
    ):
        if not callable(predict):
            raise TypeError("predict must be callable.")
        if not isinstance(batch, OperatorBatch):
            raise TypeError("batch must be an OperatorBatch.")
        if not isinstance(output_spec, OperatorOutputSpec):
            raise TypeError("output_spec must be an OperatorOutputSpec.")
        if not isinstance(likelihood, AbstractLikelihood):
            raise TypeError("likelihood must implement AbstractLikelihood.")
        if parameters is not None and not callable(parameters):
            raise TypeError("parameters must be callable or None.")

        selected_query_name = str(query_name)
        selected_field_name = str(field_name)
        if not selected_query_name or not selected_field_name:
            raise ValueError("field_name and query_name must be non-empty.")
        query = batch.query(selected_query_name)
        expected_shape = (
            batch.case_shape + query.sample_shape + output_spec.channel_shape
        )
        physical_dims = _physical_dims(query, output_spec, batch.case_axes)
        target_array = _target_array(
            target,
            batch=batch,
            query=query,
            field_name=selected_field_name,
            output_spec=output_spec,
            expected_shape=expected_shape,
            physical_dims=physical_dims,
        )
        query_mask = _output_mask(query, output_spec, batch.case_shape)
        user_mask = _observation_mask(
            observation_mask,
            expected_shape=expected_shape,
            physical_dims=physical_dims,
            has_channels=output_spec.channels != "scalar",
        )
        combined_mask = query_mask & user_mask
        count = _case_count(batch.case_shape)
        per_case_mask = combined_mask.reshape((count, -1))
        if bool(jnp.any(~jnp.any(per_case_mask, axis=-1))):
            raise ValueError(
                "Every physical operator case must contain at least one observation."
            )
        if bool(jnp.any(~jnp.isfinite(target_array) & combined_mask)):
            raise ValueError("Observed operator targets must be finite.")

        self.batch = batch
        self.target = jnp.where(combined_mask, target_array, 0.0)
        self.likelihood = likelihood
        self.output_spec = output_spec
        self.observation_mask = combined_mask
        self.predict_fn = predict
        self.parameters_fn = parameters
        self.case_count = count
        self.label = _label(label)
        self.field_name = selected_field_name
        self.query_name = selected_query_name

    def _prediction(self, parameters: PyTree[Any], /) -> Array:
        prediction = self.predict_fn(parameters)
        if not isinstance(prediction, OperatorPrediction):
            raise TypeError("Operator likelihood prediction must be OperatorPrediction.")
        _, field, query = _select_prediction_field(
            prediction,
            self.field_name,
        )
        if (
            prediction.case_axes != self.batch.case_axes
            or prediction.case_shape != self.batch.case_shape
            or field.spec.channels != self.output_spec.channels
            or field.spec.component_names != self.output_spec.component_names
        ):
            raise ValueError(
                "Operator likelihood prediction does not match the fixed batch contract."
            )
        values = jnp.asarray(field.values, dtype=float)
        if values.shape != self.target.shape:
            raise ValueError(
                f"Operator likelihood prediction must have shape {self.target.shape}; "
                f"got {values.shape}."
            )
        return _checked_query_values(
            values,
            query,
            self.batch.query(self.query_name),
        )

    def _likelihood_parameters(
        self,
        parameters: PyTree[Any],
        /,
    ) -> dict[str, Array]:
        if self.parameters_fn is None:
            return {}
        values = self.parameters_fn(parameters)
        if not isinstance(values, Mapping):
            raise TypeError("Likelihood parameters callback must return a mapping.")
        return {
            str(name): jnp.asarray(value.data if isinstance(value, cx.Field) else value)
            for name, value in values.items()
        }

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        prediction = self._prediction(parameters)
        safe_prediction = jnp.where(self.observation_mask, prediction, 0.0)
        values = jnp.asarray(
            self.likelihood.log_prob(
                safe_prediction,
                self.target,
                **self._likelihood_parameters(parameters),
            ),
            dtype=float,
        )
        values = jnp.broadcast_to(values, self.target.shape)
        invalid_prediction = self.observation_mask & ~jnp.isfinite(prediction)
        invalid_density = self.observation_mask & (
            jnp.isnan(values) | jnp.isposinf(values)
        )
        elements = jnp.where(self.observation_mask, values, 0.0)
        elements = jnp.where(
            invalid_prediction | invalid_density,
            -jnp.inf,
            elements,
        )
        return jnp.sum(elements.reshape((self.case_count, -1)), axis=-1)

    def standardized_residual(self, parameters: PyTree[Any], /) -> Array:
        """Return fixed-Gaussian residuals with masked components set to zero."""
        if not isinstance(self.likelihood, GaussianLikelihood):
            raise TypeError("standardized_residual requires a fixed GaussianLikelihood.")
        prediction = self._prediction(parameters)
        scale = jnp.broadcast_to(self.likelihood.scale, self.target.shape)
        residual = (self.target - prediction) / scale
        return jnp.where(self.observation_mask, residual, 0.0)


def _target_array(
    target: ArrayLike | cx.Field | OperatorPrediction,
    /,
    *,
    batch: OperatorBatch,
    query: FunctionSamples,
    field_name: str,
    output_spec: OperatorOutputSpec,
    expected_shape: tuple[int, ...],
    physical_dims: tuple[str, ...],
) -> Array:
    if isinstance(target, OperatorPrediction):
        _, field, target_query = _select_prediction_field(target, field_name)
        if (
            target.case_axes != batch.case_axes
            or target.case_shape != batch.case_shape
            or field.spec.channels != output_spec.channels
            or field.spec.component_names != output_spec.component_names
            or not _queries_equal(target_query, query)
        ):
            raise ValueError("Operator target does not match the fixed batch contract.")
        target_array = jnp.asarray(field.values, dtype=float)
    elif isinstance(target, cx.Field):
        template = cx.Field(jnp.empty(expected_shape), dims=physical_dims)
        target_array = jnp.asarray(_broadcast_named(target, template), dtype=float)
    else:
        target_array = jnp.asarray(target, dtype=float)
    if target_array.shape != expected_shape:
        raise ValueError(
            f"Operator target must have shape {expected_shape}; got {target_array.shape}."
        )
    return target_array


def _checked_query_values(
    values: Array,
    left: FunctionSamples,
    right: FunctionSamples,
    /,
) -> Array:
    if len(left.axes) != len(right.axes):
        raise ValueError(
            "Operator likelihood prediction does not match the fixed batch contract."
        )
    equal = jnp.asarray(True)
    for left_axis, right_axis in zip(left.axes, right.axes, strict=True):
        if (
            left_axis.name != right_axis.name
            or left_axis.basis != right_axis.basis
            or left_axis.periodic != right_axis.periodic
            or left_axis.nodes.shape != right_axis.nodes.shape
            or not _same_optional_structure(
                left_axis.quadrature_weights,
                right_axis.quadrature_weights,
            )
        ):
            raise ValueError(
                "Operator likelihood prediction does not match the fixed batch contract."
            )
        equal = equal & jnp.array_equal(left_axis.nodes, right_axis.nodes)
        left_weights = left_axis.quadrature_weights
        right_weights = right_axis.quadrature_weights
        if left_weights is not None and right_weights is not None:
            equal = equal & jnp.array_equal(left_weights, right_weights)
    for left_value, right_value in (
        (left.coordinates, right.coordinates),
        (left.quadrature_weights, right.quadrature_weights),
        (left.mask, right.mask),
    ):
        if not _same_optional_structure(left_value, right_value):
            raise ValueError(
                "Operator likelihood prediction does not match the fixed batch contract."
            )
        if left_value is not None and right_value is not None:
            equal = equal & jnp.array_equal(left_value, right_value)
    return eqx.error_if(
        values,
        ~equal,
        "Operator likelihood prediction does not match the fixed batch contract.",
    )


def _same_optional_structure(
    left: Array | None,
    right: Array | None,
    /,
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return left.shape == right.shape


def _observation_mask(
    mask: ArrayLike | cx.Field | None,
    /,
    *,
    expected_shape: tuple[int, ...],
    physical_dims: tuple[str, ...],
    has_channels: bool,
) -> Array:
    if mask is None:
        return jnp.ones(expected_shape, dtype=bool)
    if isinstance(mask, cx.Field):
        template = cx.Field(jnp.empty(expected_shape), dims=physical_dims)
        return jnp.asarray(_broadcast_named(mask, template), dtype=bool)
    value = jnp.asarray(mask, dtype=bool)
    if has_channels and value.shape == expected_shape[:-1]:
        value = value[..., None]
    return jnp.broadcast_to(value, expected_shape)


def _case_count(case_shape: tuple[int, ...], /) -> int:
    count = 1
    for size in case_shape:
        count *= int(size)
    return count


def _label(value: str, /) -> str:
    label = str(value)
    if not label:
        raise ValueError("label must be non-empty.")
    return label


__all__ = ["FixedOperatorObservationLikelihood"]
