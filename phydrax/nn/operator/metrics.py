#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array

from .data import FunctionSamples


OperatorReduction = Literal["none", "mean", "sum"]


def _sample_layout(values: Array, query: FunctionSamples, /) -> tuple[int, bool]:
    shape = tuple(int(size) for size in query.sample_shape)
    if not shape:
        raise ValueError("Operator metrics require a non-empty query sample shape.")
    values_shape = tuple(int(size) for size in values.shape)
    rank = len(shape)
    if values.ndim >= rank and values_shape[-rank:] == shape:
        return values.ndim - rank, False
    if values.ndim > rank and values_shape[-rank - 1 : -1] == shape:
        return values.ndim - rank - 1, True
    raise ValueError(
        "Operator values must contain the query sample shape at the trailing axes, "
        "optionally followed by one channel axis; "
        f"got values shape {values.shape} and query shape {shape}."
    )


def _validate_pair(
    prediction: Array,
    target: Array,
    query: FunctionSamples,
    /,
) -> tuple[Array, Array, int, bool]:
    pred = jnp.asarray(prediction)
    truth = jnp.asarray(target)
    if pred.shape != truth.shape:
        raise ValueError(
            f"Prediction and target shapes must match; got {pred.shape} and {truth.shape}."
        )
    start, has_channels = _sample_layout(pred, query)
    return pred, truth, start, has_channels


def _reduce_cases(values: Array, reduction: OperatorReduction, /) -> Array:
    if reduction == "none":
        return values
    if reduction == "mean":
        return jnp.mean(values)
    if reduction == "sum":
        return jnp.sum(values)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def _weighted_energy(
    values: Array,
    query: FunctionSamples,
    /,
    *,
    sample_start: int,
    has_channels: bool,
) -> Array:
    energy = jnp.abs(values) ** 2
    if has_channels:
        energy = jnp.sum(energy, axis=-1)
    shape = query.sample_shape
    case_shape = tuple(int(size) for size in energy.shape[:sample_start])
    weights = query.weights(case_shape=case_shape)
    weighted = energy * weights
    axes = tuple(range(sample_start, sample_start + len(shape)))
    return jnp.sum(weighted, axis=axes)


def operator_l2_loss(
    prediction: Array,
    target: Array,
    query: FunctionSamples,
    /,
    *,
    relative: bool = False,
    squared: bool = False,
    eps: float = 1e-12,
    reduction: OperatorReduction = "mean",
) -> Array:
    """Quadrature-aware per-case L2 error on an operator output."""
    pred, truth, start, has_channels = _validate_pair(prediction, target, query)
    error_energy = _weighted_energy(
        pred - truth,
        query,
        sample_start=start,
        has_channels=has_channels,
    )
    if relative:
        target_energy = _weighted_energy(
            truth,
            query,
            sample_start=start,
            has_channels=has_channels,
        )
        error_energy = error_energy / jnp.maximum(target_energy, float(eps))
    values = error_energy if squared else jnp.sqrt(jnp.maximum(error_energy, 0.0))
    return _reduce_cases(values, reduction)


def _first_derivative(values: Array, nodes: Array, axis: int, /) -> Array:
    nodes_ = jnp.asarray(nodes, dtype=float)
    n = int(nodes_.shape[0])
    if n <= 1:
        return jnp.zeros_like(values)

    moved = jnp.moveaxis(values, axis, 0)
    left = (moved[1] - moved[0]) / (nodes_[1] - nodes_[0])
    right = (moved[-1] - moved[-2]) / (nodes_[-1] - nodes_[-2])
    if n == 2:
        derivative = jnp.stack((left, right), axis=0)
    else:
        denominator = nodes_[2:] - nodes_[:-2]
        denominator = denominator.reshape((n - 2,) + (1,) * (moved.ndim - 1))
        interior = (moved[2:] - moved[:-2]) / denominator
        derivative = jnp.concatenate(
            (left[None, ...], interior, right[None, ...]), axis=0
        )
    return jnp.moveaxis(derivative, 0, axis)


def operator_sobolev_loss(
    prediction: Array,
    target: Array,
    query: FunctionSamples,
    /,
    *,
    order: int = 1,
    derivative_weights: Sequence[float] | None = None,
    relative: bool = False,
    squared: bool = False,
    eps: float = 1e-12,
    reduction: OperatorReduction = "mean",
) -> Array:
    """Quadrature-aware isotropic Sobolev error on tensor-product coordinates."""
    if int(order) < 0:
        raise ValueError("Sobolev order must be non-negative.")
    if not query.axes:
        raise ValueError("Sobolev loss requires tensor-product query axes.")
    pred, truth, start, has_channels = _validate_pair(prediction, target, query)
    if derivative_weights is None:
        weights = (1.0,) * (int(order) + 1)
    else:
        weights = tuple(float(weight) for weight in derivative_weights)
        if len(weights) != int(order) + 1:
            raise ValueError("derivative_weights must contain order + 1 values.")

    error_total = weights[0] * _weighted_energy(
        pred - truth,
        query,
        sample_start=start,
        has_channels=has_channels,
    )
    target_total = weights[0] * _weighted_energy(
        truth,
        query,
        sample_start=start,
        has_channels=has_channels,
    )

    pred_derivatives = [pred]
    truth_derivatives = [truth]
    for derivative_order in range(1, int(order) + 1):
        next_pred: list[Array] = []
        next_truth: list[Array] = []
        for pred_value, truth_value in zip(
            pred_derivatives, truth_derivatives, strict=True
        ):
            for axis_index, axis in enumerate(query.axes):
                array_axis = start + axis_index
                next_pred.append(_first_derivative(pred_value, axis.nodes, array_axis))
                next_truth.append(_first_derivative(truth_value, axis.nodes, array_axis))
        pred_derivatives = next_pred
        truth_derivatives = next_truth
        for pred_value, truth_value in zip(
            pred_derivatives, truth_derivatives, strict=True
        ):
            coefficient = weights[derivative_order]
            error_total = error_total + coefficient * _weighted_energy(
                pred_value - truth_value,
                query,
                sample_start=start,
                has_channels=has_channels,
            )
            target_total = target_total + coefficient * _weighted_energy(
                truth_value,
                query,
                sample_start=start,
                has_channels=has_channels,
            )

    if relative:
        error_total = error_total / jnp.maximum(target_total, float(eps))
    values = error_total if squared else jnp.sqrt(jnp.maximum(error_total, 0.0))
    return _reduce_cases(values, reduction)


def operator_h1_loss(
    prediction: Array,
    target: Array,
    query: FunctionSamples,
    /,
    **kwargs,
) -> Array:
    """Convenience wrapper for first-order Sobolev error."""
    return operator_sobolev_loss(
        prediction,
        target,
        query,
        order=1,
        **kwargs,
    )


def operator_spectral_loss(
    prediction: Array,
    target: Array,
    query: FunctionSamples,
    /,
    *,
    frequency_power: float = 0.0,
    relative: bool = False,
    eps: float = 1e-12,
    reduction: OperatorReduction = "mean",
) -> Array:
    """Per-case spectral error with optional high-frequency weighting."""
    pred, truth, start, has_channels = _validate_pair(prediction, target, query)
    axes = tuple(range(start, start + len(query.sample_shape)))
    pred_ft = jnp.fft.fftn(pred, axes=axes, norm="ortho")
    truth_ft = jnp.fft.fftn(truth, axes=axes, norm="ortho")

    frequency_factors = []
    for size in query.sample_shape:
        frequency_factors.append(jnp.fft.fftfreq(size) * float(size))
    grids = jnp.meshgrid(*frequency_factors, indexing="ij")
    frequency_squared = jnp.zeros(query.sample_shape, dtype=float)
    for grid in grids:
        frequency_squared = frequency_squared + grid**2
    spectral_weight = (1.0 + frequency_squared) ** float(frequency_power)
    weight_shape = (1,) * start + query.sample_shape
    if has_channels:
        weight_shape = weight_shape + (1,)
    spectral_weight = spectral_weight.reshape(weight_shape)

    error = jnp.abs(pred_ft - truth_ft) ** 2 * spectral_weight
    target_energy = jnp.abs(truth_ft) ** 2 * spectral_weight
    reduction_axes = axes + ((pred.ndim - 1,) if has_channels else ())
    per_case = jnp.sum(error, axis=reduction_axes)
    if relative:
        denominator = jnp.sum(target_energy, axis=reduction_axes)
        per_case = per_case / jnp.maximum(denominator, float(eps))
    values = jnp.sqrt(jnp.maximum(per_case, 0.0))
    return _reduce_cases(values, reduction)


def operator_conservation_error(
    prediction: Array,
    target: Array,
    query: FunctionSamples,
    /,
    *,
    relative: bool = False,
    eps: float = 1e-12,
    reduction: OperatorReduction = "mean",
) -> Array:
    """Error between predicted and target spatial integrals, per physical case."""
    pred, truth, start, has_channels = _validate_pair(prediction, target, query)
    case_shape = tuple(int(size) for size in pred.shape[:start])
    weights = query.weights(case_shape=case_shape)
    if has_channels:
        weights = weights[..., None]
    axes = tuple(range(start, start + len(query.sample_shape)))
    pred_integral = jnp.sum(pred * weights, axis=axes)
    truth_integral = jnp.sum(truth * weights, axis=axes)
    error = jnp.abs(pred_integral - truth_integral)
    if has_channels:
        error = jnp.sqrt(jnp.sum(error**2, axis=-1))
        truth_norm = jnp.sqrt(jnp.sum(jnp.abs(truth_integral) ** 2, axis=-1))
    else:
        truth_norm = jnp.abs(truth_integral)
    if relative:
        error = error / jnp.maximum(truth_norm, float(eps))
    return _reduce_cases(error, reduction)


__all__ = [
    "OperatorReduction",
    "operator_conservation_error",
    "operator_h1_loss",
    "operator_l2_loss",
    "operator_sobolev_loss",
    "operator_spectral_loss",
]
