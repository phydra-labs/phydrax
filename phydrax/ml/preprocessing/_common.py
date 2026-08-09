#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    FitResult,
    GradientContract,
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import effective_sample_size, weighted_mean
from .._schema import FeatureSchema
from .._sparse_features import SparseFeatures


class PreprocessingDiagnostics(StrictModule):
    """Auditable shape, schema, and observation diagnostics for a transform fit."""

    valid: Array
    status: Array
    observed_weight: Array
    effective_samples: Array
    constant_features: Array
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)
    input_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    details: tuple[tuple[str, Any], ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        observed_weight: Any,
        effective_samples: Any,
        constant_features: Any,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        method: str,
        details: tuple[tuple[str, Any], ...] = (),
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.observed_weight = jnp.asarray(observed_weight)
        self.effective_samples = jnp.asarray(effective_samples)
        self.constant_features = jnp.asarray(constant_features, dtype=bool)
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.input_shape = tuple(int(size) for size in input_shape)
        self.output_shape = tuple(int(size) for size in output_shape)
        self.method = str(method)
        self.details = tuple(details)


def _dense_batch(batch: MLBatch, /) -> Array:
    if isinstance(batch.features, SparseFeatures):
        raise TypeError(
            "Preprocessing fits require dense features; sparse semantics must remain explicit."
        )
    return jnp.asarray(batch.features)


def _feature_observations(
    batch: MLBatch,
    /,
    *,
    weight_policy: WeightPolicy,
    extra_mask: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return safe dense values and per-feature weights without invalid arithmetic."""
    values = _dense_batch(batch)
    sample_weight = batch.effective_weight(weight_policy)
    weight_ok = jnp.isfinite(sample_weight) & (sample_weight >= 0.0)
    finite = jnp.isfinite(values)
    included = batch.feature_mask & batch.sample_mask[..., None] & weight_ok[..., None]
    included = included & finite
    if extra_mask is not None:
        included = included & jnp.broadcast_to(
            jnp.asarray(extra_mask, dtype=bool), values.shape
        )
    weights = jnp.where(included, sample_weight[..., None], 0.0)
    safe_values = jnp.where(included, values, 0)
    mass = jnp.sum(weights, axis=-2)
    effective = effective_sample_size(weights, axis=-2)
    finite_weight = jnp.all(jnp.isfinite(sample_weight) | ~batch.sample_mask, axis=-1)
    nonnegative_weight = jnp.all((sample_weight >= 0.0) | ~batch.sample_mask, axis=-1)
    policy_valid = finite_weight & nonnegative_weight
    observed = mass > 0.0
    valid = policy_valid & jnp.all(observed, axis=-1)
    status = jnp.where(
        ~finite_weight,
        ML_NONFINITE,
        jnp.where(
            ~nonnegative_weight,
            ML_INFEASIBLE,
            jnp.where(jnp.all(observed, axis=-1), ML_SUCCESS, ML_INSUFFICIENT_DATA),
        ),
    )
    return safe_values, weights, mass, effective, valid, status


def _weighted_mean(values: Array, weights: Array, /) -> Array:
    return weighted_mean(values, weights, axis=-2)


def _weighted_quantiles(
    values: Array,
    weights: Array,
    probabilities: Array,
    /,
) -> Array:
    """Exact inverse-CDF weighted quantiles over the sample axis."""
    probs = jnp.asarray(probabilities, dtype=weights.dtype)
    if probs.ndim != 1 or probs.shape[0] == 0:
        raise ValueError("probabilities must be a nonempty rank-one array.")
    probs = eqx.error_if(
        probs,
        jnp.any((probs < 0.0) | (probs > 1.0)),
        "probabilities must lie in [0, 1].",
    )
    ordered_values = jnp.moveaxis(values, -2, -1)
    ordered_weights = jnp.moveaxis(weights, -2, -1)
    filler = jnp.asarray(jnp.inf, dtype=ordered_values.dtype)
    sortable = jnp.where(ordered_weights > 0.0, ordered_values, filler)
    order = jnp.argsort(sortable, axis=-1)
    sorted_values = jnp.take_along_axis(sortable, order, axis=-1)
    sorted_weights = jnp.take_along_axis(ordered_weights, order, axis=-1)
    cumulative = jnp.cumsum(sorted_weights, axis=-1)
    total = cumulative[..., -1]
    target = total[..., None] * probs
    reached = cumulative[..., None, :] >= target[..., :, None]
    indices = jnp.argmax(reached, axis=-1)
    quantiles = jnp.take_along_axis(sorted_values, indices, axis=-1)
    return jnp.where(total[..., None] > 0.0, quantiles, jnp.zeros_like(quantiles))


def _align_parameter(
    parameter: Array,
    x: Array,
    case_shape: tuple[int, ...],
    /,
    *,
    trailing_rank: int = 1,
) -> Array:
    extra = x.ndim - 1 - len(case_shape)
    if extra < 0:
        return parameter
    trailing = parameter.shape[-trailing_rank:]
    return parameter.reshape(case_shape + (1,) * extra + trailing)


def _check_features(x: Any, in_size: int, /) -> Array:
    values = jnp.asarray(x)
    if values.ndim < 1 or int(values.shape[-1]) != int(in_size):
        raise ValueError(
            f"Expected a final feature axis of length {in_size}; got {values.shape}."
        )
    return values


def _diagnostics(
    batch: MLBatch,
    output_schema: FeatureSchema,
    mass: Array,
    effective: Array,
    valid: Array,
    status: Array,
    /,
    *,
    method: str,
    constant: Array | None = None,
    details: tuple[tuple[str, Any], ...] = (),
) -> PreprocessingDiagnostics:
    if constant is None:
        constant = jnp.zeros_like(mass, dtype=bool)
    return PreprocessingDiagnostics(
        valid=valid,
        status=status,
        observed_weight=mass,
        effective_samples=effective,
        constant_features=constant,
        input_schema=batch.feature_schema,
        output_schema=output_schema,
        input_shape=batch.case_shape + (batch.sample_count, batch.feature_count),
        output_shape=batch.case_shape + (batch.sample_count, len(output_schema.names)),
        method=method,
        details=details,
    )


def _fit_result(
    model,
    diagnostics: PreprocessingDiagnostics,
    contract: GradientContract,
    /,
) -> FitResult:
    return FitResult(
        model,
        diagnostics,
        valid=diagnostics.valid,
        status=diagnostics.status,
        method=diagnostics.method,
        gradient_contract=contract,
    )


__all__ = ["PreprocessingDiagnostics"]
