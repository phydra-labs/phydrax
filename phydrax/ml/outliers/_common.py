#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._model import ModelBinding
from ..._strict import StrictModule
from .._batch import MLBatch
from .._contracts import ML_INSUFFICIENT_DATA, ML_NONFINITE, ML_SUCCESS


_BLOCKWISE_BINDING = ModelBinding.blockwise("flat", pass_key=True)


class OutlierDiagnostics(StrictModule):
    """Immutable numerical and calibration diagnostics for anomaly fits."""

    valid: Array
    status: Array
    objective: Array
    iterations: Array
    effective_samples: Array
    threshold: Array
    score_minimum: Array
    score_maximum: Array
    rank: Array
    condition: Array
    converged: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        objective: Any = jnp.nan,
        iterations: Any = 0,
        effective_samples: Any = 0,
        threshold: Any = jnp.nan,
        score_minimum: Any = jnp.nan,
        score_maximum: Any = jnp.nan,
        rank: Any = -1,
        condition: Any = jnp.nan,
        converged: Any = True,
        method: str,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.objective = jnp.asarray(objective)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.effective_samples = jnp.asarray(effective_samples)
        self.threshold = jnp.asarray(threshold)
        self.score_minimum = jnp.asarray(score_minimum)
        self.score_maximum = jnp.asarray(score_maximum)
        self.rank = jnp.asarray(rank, dtype=jnp.int32)
        self.condition = jnp.asarray(condition)
        self.converged = jnp.asarray(converged, dtype=bool)
        self.method = str(method)


def _case_count(case_shape: tuple[int, ...]) -> int:
    return prod(case_shape) if case_shape else 1


def _fit_arrays(batch: MLBatch) -> tuple[Array, Array, Array]:
    x = batch.dense_features()
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.float32)
    weights = batch.effective_weight("statistical").astype(x.real.dtype)
    weights = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights) | (weights < 0.0)),
        "Outlier sample weights must be finite and nonnegative.",
    )
    finite = jnp.all(jnp.isfinite(jnp.real(x)) & jnp.isfinite(jnp.imag(x)), axis=-1)
    feature_valid = jnp.all(batch.feature_mask, axis=-1)
    active = (
        batch.sample_mask
        & feature_valid
        & finite
        & jnp.isfinite(weights)
        & (weights > 0.0)
    )
    return (
        jnp.where(active[..., None], x, 0),
        jnp.where(active, weights, 0.0),
        active,
    )


def _prepare_queries(
    value: ArrayLike,
    *,
    case_shape: tuple[int, ...],
    feature_count: int,
) -> tuple[Array, tuple[int, ...]]:
    x = jnp.asarray(value)
    if x.ndim == 0 or int(x.shape[-1]) != feature_count:
        raise ValueError(f"Input must end in feature axis of size {feature_count}.")
    if (
        case_shape
        and x.ndim > len(case_shape)
        and tuple(int(s) for s in x.shape[: len(case_shape)]) == case_shape
    ):
        query_shape = tuple(int(s) for s in x.shape[len(case_shape) : -1])
        shaped = x.reshape(
            (
                _case_count(case_shape),
                prod(query_shape) if query_shape else 1,
                feature_count,
            )
        )
    else:
        query_shape = tuple(int(s) for s in x.shape[:-1])
        shaped = jnp.broadcast_to(x, case_shape + x.shape).reshape(
            (
                _case_count(case_shape),
                prod(query_shape) if query_shape else 1,
                feature_count,
            )
        )
    return shaped, query_shape


def _restore_scores(
    values: Array,
    *,
    case_shape: tuple[int, ...],
    query_shape: tuple[int, ...],
) -> Array:
    return values.reshape(case_shape + query_shape)


def _weighted_quantile_one(values: Array, weights: Array, quantile: float) -> Array:
    order = jax.lax.stop_gradient(jnp.argsort(values))
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = jnp.cumsum(sorted_weights)
    target = float(quantile) * jnp.sum(sorted_weights)
    index = jax.lax.stop_gradient(jnp.argmax(cumulative >= target))
    return sorted_values[index]


def _weighted_threshold(values: Array, weights: Array, contamination: float) -> Array:
    case_shape = tuple(int(s) for s in values.shape[:-1])
    cases = _case_count(case_shape)
    result = jax.vmap(
        lambda score_, weight_: _weighted_quantile_one(
            score_, weight_, 1.0 - float(contamination)
        )
    )(
        values.reshape((cases, values.shape[-1])),
        weights.reshape((cases, weights.shape[-1])),
    )
    return result.reshape(case_shape)


def _score_bounds(scores: Array, active: Array) -> tuple[Array, Array]:
    minimum = jnp.min(jnp.where(active, scores, jnp.inf), axis=-1)
    maximum = jnp.max(jnp.where(active, scores, -jnp.inf), axis=-1)
    return minimum, maximum


def _fit_status(finite: Array, enough: Array) -> Array:
    return jnp.where(
        ~finite,
        ML_NONFINITE,
        jnp.where(enough, ML_SUCCESS, ML_INSUFFICIENT_DATA),
    ).astype(jnp.int32)


__all__ = ["OutlierDiagnostics"]
