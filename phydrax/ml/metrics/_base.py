#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


METRIC_SUCCESS = 0
METRIC_EMPTY = 1
METRIC_INVALID_INPUT = 2
METRIC_ZERO_DENOMINATOR = 3
METRIC_SINGLE_CLASS = 4
METRIC_UNDEFINED = 5

OutputReduction: TypeAlias = Literal["raw_values", "uniform_average", "variance_weighted"]
Average: TypeAlias = Literal["binary", "micro", "macro", "weighted", "none"]


class MetricResult(StrictModule):
    """A metric value accompanied by JAX-compatible edge-state diagnostics.

    ``valid`` and ``status`` describe the independent case/output units represented
    by ``value``. ``effective_weight`` is the total included sample mass. Shapes may
    differ for structured values such as confusion matrices.
    """

    value: Array
    valid: Array
    status: Array
    effective_weight: Array

    def __init__(
        self,
        value: ArrayLike,
        /,
        *,
        valid: ArrayLike,
        status: ArrayLike,
        effective_weight: ArrayLike,
    ):
        self.value = jnp.asarray(value)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.effective_weight = jnp.asarray(effective_weight)


def _normalize_axis(axis: int, ndim: int, /) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"sample_axis {axis} is invalid for rank-{ndim} input.")
    return normalized


def _real_dtype(*arrays: Array):
    dtype = jnp.result_type(*(array.dtype for array in arrays), jnp.float32)
    return jnp.empty((), dtype=dtype).real.dtype


def _reject_complex(*arrays: Array, metric: str) -> None:
    if any(jnp.issubdtype(array.dtype, jnp.complexfloating) for array in arrays):
        raise TypeError(f"{metric} is not defined for complex-valued inputs.")


def _broadcast_layout(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    /,
) -> str | None:
    direct = len(source_shape) <= len(target_shape) and all(
        source == 1 or source == target
        for source, target in zip(reversed(source_shape), reversed(target_shape))
    )
    if direct:
        return "direct"
    if len(source_shape) > len(target_shape):
        return None
    expanded_shape = source_shape + (1,) * (len(target_shape) - len(source_shape))
    trailing = all(
        source == 1 or source == target
        for source, target in zip(expanded_shape, target_shape)
    )
    return "trailing" if trailing else None


def _broadcast_prefix(
    value: ArrayLike | None,
    prefix_shape: tuple[int, ...],
    /,
    *,
    dtype,
    fill: float,
    name: str,
) -> Array:
    if value is None:
        return jnp.full(prefix_shape, fill, dtype=dtype)
    array = jnp.asarray(value, dtype=dtype)
    layout = _broadcast_layout(array.shape, prefix_shape)
    if layout == "direct":
        return jnp.broadcast_to(array, prefix_shape)
    if layout == "trailing":
        expanded = array.reshape(array.shape + (1,) * (len(prefix_shape) - array.ndim))
        return jnp.broadcast_to(expanded, prefix_shape)
    raise ValueError(f"{name} must broadcast to case/sample shape {prefix_shape}.")


def _broadcast_full(
    value: ArrayLike | None,
    shape: tuple[int, ...],
    /,
    *,
    dtype,
    fill: bool,
    name: str,
) -> Array:
    if value is None:
        return jnp.full(shape, fill, dtype=dtype)
    array = jnp.asarray(value, dtype=dtype)
    layout = _broadcast_layout(array.shape, shape)
    if layout == "direct":
        return jnp.broadcast_to(array, shape)
    if layout == "trailing":
        expanded = array.reshape(array.shape + (1,) * (len(shape) - array.ndim))
        return jnp.broadcast_to(expanded, shape)
    raise ValueError(
        f"{name} must broadcast to value shape {shape}, either directly "
        "or across trailing value axes."
    )


def _broadcast_metric_mask(
    mask: ArrayLike | None,
    shape: tuple[int, ...],
    prefix_shape: tuple[int, ...],
    /,
) -> Array:
    if mask is None:
        return jnp.ones(shape, dtype=bool)
    array = jnp.asarray(mask, dtype=bool)
    if _broadcast_layout(array.shape, prefix_shape) is None:
        return _broadcast_full(array, shape, dtype=bool, fill=True, name="mask")
    prefix = _broadcast_prefix(array, prefix_shape, dtype=bool, fill=1.0, name="mask")
    return jnp.broadcast_to(
        prefix.reshape(prefix.shape + (1,) * (len(shape) - len(prefix.shape))),
        shape,
    )


def _prepare_pair(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    metric: str,
    allow_complex: bool,
) -> tuple[Array, Array, Array, Array, Array, int]:
    true = jnp.asarray(y_true)
    pred = jnp.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"{metric} requires y_true and y_pred to have identical shapes.")
    if true.ndim == 0:
        raise ValueError(f"{metric} requires a sample axis.")
    if not allow_complex:
        _reject_complex(true, pred, metric=metric)
    axis = _normalize_axis(sample_axis, true.ndim)
    prefix_shape = tuple(int(size) for size in true.shape[: axis + 1])
    dtype = _real_dtype(true, pred)
    weights = _broadcast_prefix(
        sample_weight,
        prefix_shape,
        dtype=dtype,
        fill=1.0,
        name="sample_weight",
    )
    expanded_weights = weights.reshape(weights.shape + (1,) * (true.ndim - weights.ndim))
    included = _broadcast_metric_mask(
        mask, tuple(int(size) for size in true.shape), prefix_shape
    )
    finite_values = jnp.isfinite(true) & jnp.isfinite(pred)
    valid_weights = jnp.isfinite(expanded_weights) & (expanded_weights >= 0.0)
    invalid = jnp.any(included & ~(finite_values & valid_weights), axis=axis)
    active = included & finite_values & valid_weights
    safe_true = jnp.where(active, true, 0)
    safe_pred = jnp.where(active, pred, 0)
    return safe_true, safe_pred, expanded_weights, active, invalid, axis


def _prepare_values(
    values: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    metric: str,
    allow_complex: bool,
) -> tuple[Array, Array, Array, Array, int]:
    value = jnp.asarray(values)
    if value.ndim == 0:
        raise ValueError(f"{metric} requires a sample axis.")
    if not allow_complex:
        _reject_complex(value, metric=metric)
    axis = _normalize_axis(sample_axis, value.ndim)
    prefix_shape = tuple(int(size) for size in value.shape[: axis + 1])
    dtype = _real_dtype(value)
    weights = _broadcast_prefix(
        sample_weight,
        prefix_shape,
        dtype=dtype,
        fill=1.0,
        name="sample_weight",
    )
    expanded_weights = weights.reshape(weights.shape + (1,) * (value.ndim - weights.ndim))
    included = _broadcast_metric_mask(
        mask, tuple(int(size) for size in value.shape), prefix_shape
    )
    valid_weights = jnp.isfinite(expanded_weights) & (expanded_weights >= 0.0)
    finite_values = jnp.isfinite(value)
    invalid = jnp.any(included & ~(finite_values & valid_weights), axis=axis)
    active = included & finite_values & valid_weights
    safe_value = jnp.where(active, value, 0)
    return safe_value, expanded_weights, active, invalid, axis


def _weighted_sum(values: Array, weights: Array, active: Array, axis: int) -> Array:
    return jnp.sum(jnp.where(active, weights * values, 0), axis=axis)


def _weighted_mean(
    values: Array, weights: Array, active: Array, axis: int
) -> tuple[Array, Array]:
    total = _weighted_sum(values, weights, active, axis)
    mass = jnp.sum(jnp.where(active, weights, 0.0), axis=axis)
    return total / jnp.where(mass > 0.0, mass, 1.0), mass


def _status(
    *,
    invalid: Array,
    empty: Array,
    undefined: Array | None = None,
    undefined_status: int = METRIC_UNDEFINED,
) -> tuple[Array, Array]:
    if undefined is None:
        undefined = jnp.zeros_like(empty, dtype=bool)
    status = jnp.where(
        invalid,
        METRIC_INVALID_INPUT,
        jnp.where(
            empty, METRIC_EMPTY, jnp.where(undefined, undefined_status, METRIC_SUCCESS)
        ),
    ).astype(jnp.int32)
    return status == METRIC_SUCCESS, status


def _nan_where_invalid(value: Array, valid: Array) -> Array:
    valid_ = valid.reshape(valid.shape + (1,) * (value.ndim - valid.ndim))
    dtype = jnp.result_type(value.dtype, jnp.float32)
    return jnp.where(valid_, value.astype(dtype), jnp.asarray(jnp.nan, dtype=dtype))


def _reduce_outputs(
    result: MetricResult,
    /,
    *,
    output_ndim: int,
    reduction: OutputReduction,
    variance: Array | None = None,
) -> MetricResult:
    if reduction == "raw_values" or output_ndim == 0:
        return result
    if reduction not in {"uniform_average", "variance_weighted"}:
        raise ValueError(f"Unsupported output reduction {reduction!r}.")
    axes = tuple(range(result.value.ndim - output_ndim, result.value.ndim))
    all_valid = jnp.all(result.valid, axis=axes)
    status = jnp.max(result.status, axis=axes)
    if reduction == "uniform_average":
        value = jnp.mean(result.value, axis=axes)
    else:
        if variance is None:
            raise ValueError("variance_weighted reduction requires output variances.")
        variance_ = jnp.where(result.valid, jnp.asarray(variance), 0.0)
        denominator = jnp.sum(variance_, axis=axes)
        value = jnp.sum(variance_ * result.value, axis=axes) / jnp.where(
            denominator > 0.0, denominator, 1.0
        )
        undefined = denominator <= 0.0
        all_valid = all_valid & ~undefined
        status = jnp.where(undefined, METRIC_ZERO_DENOMINATOR, status)
    effective = jnp.min(result.effective_weight, axis=axes)
    return MetricResult(
        _nan_where_invalid(value, all_valid),
        valid=all_valid,
        status=status,
        effective_weight=effective,
    )


def _result(
    value: Array,
    /,
    *,
    invalid: Array,
    effective_weight: Array,
    undefined: Array | None = None,
    undefined_status: int = METRIC_UNDEFINED,
) -> MetricResult:
    valid, status = _status(
        invalid=invalid,
        empty=effective_weight <= 0.0,
        undefined=undefined,
        undefined_status=undefined_status,
    )
    return MetricResult(
        _nan_where_invalid(jnp.asarray(value), valid),
        valid=valid,
        status=status,
        effective_weight=effective_weight,
    )


__all__ = [
    "Average",
    "METRIC_EMPTY",
    "METRIC_INVALID_INPUT",
    "METRIC_SINGLE_CLASS",
    "METRIC_SUCCESS",
    "METRIC_UNDEFINED",
    "METRIC_ZERO_DENOMINATOR",
    "MetricResult",
    "OutputReduction",
]
