#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

import phydrax.ein as ein

from ...._doc import DOC_KEY0
from ....linalg import (
    DenseCholesky,
    DenseLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    RHSLayout,
    solve,
)
from ..data import (
    FunctionSamples,
    OperatorBatch,
    OperatorOutputSpec,
    OperatorPrediction,
)
from ..protocols import OperatorModel
from ._execution import samples_with_values
from ._physics import operator_hilbert_inner_product
from ._trained_operator import TrainedOperator


def _batch_with_source(
    batch: OperatorBatch,
    source_name: str,
    values: Array,
    /,
) -> OperatorBatch:
    inputs = dict(batch.inputs)
    inputs[source_name] = samples_with_values(inputs[source_name], values)
    return OperatorBatch(
        inputs=inputs,
        queries=batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def _predict(
    operator: Any,
    batch: OperatorBatch,
    key: Key[Array, ""],
    /,
) -> OperatorPrediction:
    if isinstance(operator, TrainedOperator):
        prepared = operator.prepare_prevalidated(batch)
        return operator.predict_prepared(prepared, key=key)
    if isinstance(operator, OperatorModel):
        return operator.predict_prevalidated(batch, key=key)
    prediction = operator(batch, key=key)
    if not isinstance(prediction, OperatorPrediction):
        raise TypeError(
            "Physical linearization requires named OperatorPrediction output."
        )
    return prediction


def _field_name(prediction: OperatorPrediction, requested: str | None) -> str:
    if requested is not None:
        if requested not in prediction.fields:
            raise KeyError(
                f"Unknown linearization field {requested!r}; "
                f"expected one of {tuple(prediction.fields)!r}."
            )
        return str(requested)
    if len(prediction.fields) != 1:
        raise ValueError(
            "field_name is required for multi-output operator linearization."
        )
    return next(iter(prediction.fields))


def _apply_channel_metric(values: Array, metric: Array | None, /) -> Array:
    if metric is None:
        return values
    matrix = jnp.asarray(metric)
    if values.ndim == 0 or matrix.shape != (values.shape[-1], values.shape[-1]):
        raise ValueError(
            "Channel metric must be square and match the trailing channel dimension."
        )
    return ein.contract("ij,...j->...i", matrix, values)


def _riesz_map(
    values: Array,
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    metric: Array | None,
    /,
) -> Array:
    mapped = _apply_channel_metric(jnp.asarray(values), metric)
    weights = samples.weights(case_shape=case_shape)
    if mapped.ndim == weights.ndim + 1:
        weights = weights[..., None]
    return mapped * weights


def _inverse_riesz_map(
    covector: Array,
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    metric: Array | None,
    /,
) -> Array:
    weights = samples.weights(case_shape=case_shape)
    if covector.ndim == weights.ndim + 1:
        weights = weights[..., None]
    unweighted = jnp.where(weights > 0.0, covector / weights, 0.0)
    if metric is None:
        return unweighted
    matrix = jnp.asarray(metric)
    if unweighted.ndim == 0 or matrix.shape != (
        unweighted.shape[-1],
        unweighted.shape[-1],
    ):
        raise ValueError(
            "Channel metric must be square and match the trailing channel dimension."
        )
    channel_count = matrix.shape[-1]
    right_hand_side = jnp.moveaxis(unweighted, -1, 0).reshape((channel_count, -1))
    operator = DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "asserted",
                "positive_definite": "asserted",
            },
        ),
    )
    result = solve(
        LinearSystem(operator),
        right_hand_side,
        policy=LinearSolvePolicy(DenseCholesky()),
        rhs_layout=RHSLayout((right_hand_side.shape[-1],)),
    )
    solved = eqx.error_if(
        result.value,
        jnp.any(~result.successful),
        "Channel metric solve failed.",
    )
    return jnp.moveaxis(
        solved.reshape((channel_count,) + unweighted.shape[:-1]),
        0,
        -1,
    )


@dataclass(frozen=True)
class OperatorLinearization:
    """Matrix-free Fréchet linearization of one source-to-field operator map.

    A :class:`TrainedOperator` is differentiated from physical source values to
    dimensionalized physical output values. A raw model is differentiated in its
    execution coordinates. No Jacobian is materialized.
    """

    operator: Any
    batch: OperatorBatch
    source_name: str
    field_name: str
    key: Key[Array, ""]
    base_input: Array
    base_output: Array
    output_query: FunctionSamples
    output_spec: OperatorOutputSpec

    @property
    def source_samples(self) -> FunctionSamples:
        return self.batch.input(self.source_name)

    @property
    def output_samples(self) -> FunctionSamples:
        return self.output_query

    def _evaluate(self, source_values: Array, /) -> Array:
        perturbed = _batch_with_source(
            self.batch,
            self.source_name,
            source_values,
        )
        prediction = _predict(self.operator, perturbed, self.key)
        return prediction.field(self.field_name).values

    def pushforward(self, tangent: Any, /) -> Array:
        """Apply the Fréchet derivative to one source perturbation via JVP."""
        tangent_array = jnp.asarray(tangent, dtype=self.base_input.dtype)
        if tangent_array.shape != self.base_input.shape:
            raise ValueError(
                f"Source tangent must have shape {self.base_input.shape}; "
                f"got {tangent_array.shape}."
            )
        _, output_tangent = jax.jvp(
            self._evaluate,
            (self.base_input,),
            (tangent_array,),
        )
        return output_tangent

    def pullback(self, cotangent: Any, /, *, hermitian: bool = True) -> Array:
        """Apply the Euclidean transpose, or Hermitian transpose for complex arrays."""
        cotangent_array = jnp.asarray(cotangent, dtype=self.base_output.dtype)
        if cotangent_array.shape != self.base_output.shape:
            raise ValueError(
                f"Output cotangent must have shape {self.base_output.shape}; "
                f"got {cotangent_array.shape}."
            )
        _, pullback = jax.vjp(self._evaluate, self.base_input)
        if hermitian and (
            jnp.issubdtype(self.base_input.dtype, jnp.complexfloating)
            or jnp.issubdtype(self.base_output.dtype, jnp.complexfloating)
        ):
            return jnp.conj(pullback(jnp.conj(cotangent_array))[0])
        return pullback(cotangent_array)[0]

    def adjoint(
        self,
        cotangent: Any,
        /,
        *,
        source_channel_metric: Array | None = None,
        output_channel_metric: Array | None = None,
    ) -> Array:
        """Apply the quadrature-aware Hilbert adjoint of the linearization."""
        output_covector = _riesz_map(
            jnp.asarray(cotangent, dtype=self.base_output.dtype),
            self.output_samples,
            self.batch.case_shape,
            output_channel_metric,
        )
        source_covector = self.pullback(output_covector, hermitian=True)
        return _inverse_riesz_map(
            source_covector,
            self.source_samples,
            self.batch.case_shape,
            source_channel_metric,
        )

    def adjoint_identity_error(
        self,
        tangent: Any,
        cotangent: Any,
        /,
        *,
        source_channel_metric: Array | None = None,
        output_channel_metric: Array | None = None,
        epsilon: float = 1e-12,
    ) -> Array:
        """Return the relative defect in ``<J h,v> = <h,J* v>`` per case."""
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        tangent_array = jnp.asarray(tangent, dtype=self.base_input.dtype)
        cotangent_array = jnp.asarray(cotangent, dtype=self.base_output.dtype)
        left = operator_hilbert_inner_product(
            self.pushforward(tangent_array),
            cotangent_array,
            self.output_samples,
            case_shape=self.batch.case_shape,
            channel_metric=output_channel_metric,
        )
        right = operator_hilbert_inner_product(
            tangent_array,
            self.adjoint(
                cotangent_array,
                source_channel_metric=source_channel_metric,
                output_channel_metric=output_channel_metric,
            ),
            self.source_samples,
            case_shape=self.batch.case_shape,
            channel_metric=source_channel_metric,
        )
        scale = jnp.maximum(jnp.maximum(jnp.abs(left), jnp.abs(right)), float(epsilon))
        return jnp.abs(left - right) / scale


def linearize_operator(
    operator: Any,
    batch: OperatorBatch,
    source_name: str,
    /,
    *,
    field_name: str | None = None,
    key: Key[Array, ""] = DOC_KEY0,
) -> OperatorLinearization:
    """Construct a matrix-free physical/execution operator linearization."""
    if not isinstance(batch, OperatorBatch):
        raise TypeError("linearize_operator requires an OperatorBatch.")
    if source_name not in batch.inputs:
        raise KeyError(
            f"Unknown linearization source {source_name!r}; "
            f"expected one of {tuple(batch.inputs)!r}."
        )
    source = batch.input(source_name)
    if source.values is None:
        raise ValueError(f"Linearization source {source_name!r} has no values.")
    prediction = _predict(operator, batch, key)
    resolved_field = _field_name(prediction, field_name)
    return OperatorLinearization(
        operator=operator,
        batch=batch,
        source_name=str(source_name),
        field_name=resolved_field,
        key=key,
        base_input=jnp.asarray(source.values),
        base_output=jnp.asarray(prediction.field(resolved_field).values),
        output_query=prediction.query_geometry(
            prediction.field(resolved_field).query_name
        ),
        output_spec=prediction.field(resolved_field).spec,
    )


__all__ = ["OperatorLinearization", "linearize_operator"]
