#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._model import AbstractArrayModel
from .._batch import MLBatch, WeightPolicy
from .._contracts import AbstractRecipe, FitResult, GradientContract
from .._schema import FeatureSchema
from ._common import (
    _align_parameter,
    _check_features,
    _diagnostics,
    _feature_observations,
    _fit_result,
    _weighted_mean,
    _weighted_quantiles,
)


class _AbstractAffineTransform(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    center: Array
    scale: Array
    output_offset: Array
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    clip_bounds: tuple[float, float] | None = eqx.field(static=True)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        center = _align_parameter(self.center, values, self.case_shape)
        scale = _align_parameter(self.scale, values, self.case_shape)
        offset = _align_parameter(self.output_offset, values, self.case_shape)
        transformed = (values - center) / scale + offset
        if self.clip_bounds is not None:
            transformed = jnp.clip(transformed, self.clip_bounds[0], self.clip_bounds[1])
        return transformed

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        if self.clip_bounds is not None:
            raise NotImplementedError(
                "A clipping affine transform is not bijective and has no inverse transform."
            )
        values = _check_features(x, self.out_size)
        center = _align_parameter(self.center, values, self.case_shape)
        scale = _align_parameter(self.scale, values, self.case_shape)
        offset = _align_parameter(self.output_offset, values, self.case_shape)
        return (values - offset) * scale + center


class FittedStandardScaler(_AbstractAffineTransform):
    """Immutable affine standardization learned from weighted observations."""

    def __init__(
        self,
        center: Array,
        scale: Array,
        /,
        *,
        schema: FeatureSchema,
        case_shape: tuple[int, ...],
    ):
        self.in_size = len(schema.names)
        self.out_size = len(schema.names)
        self.center = jnp.asarray(center)
        self.scale = jnp.asarray(scale)
        self.output_offset = jnp.zeros_like(self.scale)
        self.input_schema = schema
        self.output_schema = schema
        self.case_shape = tuple(case_shape)
        self.clip_bounds = None


class StandardScaler(AbstractRecipe):
    """Weighted, mask-aware population standardization recipe."""

    with_mean: bool = eqx.field(static=True)
    with_std: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        weight_policy: WeightPolicy = "statistical",
    ):
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        mean = _weighted_mean(x, weights)
        center = mean if self.with_mean else jnp.zeros_like(mean)
        residual = jnp.where(weights > 0.0, x - mean[..., None, :], 0)
        variance = _weighted_mean(jnp.real(residual * jnp.conj(residual)), weights)
        raw_scale = jnp.sqrt(jnp.maximum(variance, 0.0))
        constant = raw_scale == 0.0
        scale = (
            jnp.where(constant, jnp.ones_like(raw_scale), raw_scale)
            if self.with_std
            else jnp.ones_like(raw_scale)
        )
        model = FittedStandardScaler(
            center,
            scale,
            schema=batch.feature_schema,
            case_shape=batch.case_shape,
        )
        diagnostics = _diagnostics(
            batch,
            model.output_schema,
            mass,
            effective,
            valid,
            status,
            method="standard_scaler",
            constant=constant,
            details=(("with_mean", self.with_mean), ("with_std", self.with_std)),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_features="conditional",
                fit_targets="none",
                fit_weights="conditional",
                fit_hyperparameters="none",
                fit_mode="direct",
                conditions=("Feature masks and positive-weight support are held fixed.",),
            ),
        )


class FittedMinMaxScaler(_AbstractAffineTransform):
    clip: bool = eqx.field(static=True)
    feature_range: tuple[float, float] = eqx.field(static=True)

    def __init__(
        self,
        center: Array,
        scale: Array,
        output_offset: Array,
        /,
        *,
        schema: FeatureSchema,
        case_shape: tuple[int, ...],
        feature_range: tuple[float, float],
        clip: bool,
    ):
        self.in_size = len(schema.names)
        self.out_size = len(schema.names)
        self.center = jnp.asarray(center)
        self.scale = jnp.asarray(scale)
        self.output_offset = jnp.asarray(output_offset)
        self.input_schema = schema
        self.output_schema = schema
        self.case_shape = tuple(case_shape)
        self.feature_range = feature_range
        self.clip = bool(clip)
        self.clip_bounds = feature_range if clip else None


class MinMaxScaler(AbstractRecipe):
    """Weighted/masked extrema scaling to an explicit finite interval."""

    feature_range: tuple[float, float] = eqx.field(static=True)
    clip: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        feature_range: tuple[float, float] = (0.0, 1.0),
        *,
        clip: bool = False,
        weight_policy: WeightPolicy = "statistical",
    ):
        lower, upper = (float(feature_range[0]), float(feature_range[1]))
        if not jnp.isfinite(lower) or not jnp.isfinite(upper) or not upper > lower:
            raise ValueError("feature_range must contain finite increasing bounds.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.feature_range = (lower, upper)
        self.clip = bool(clip)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError("MinMaxScaler requires real-valued features.")
        minimum = jnp.min(jnp.where(weights > 0.0, x, jnp.inf), axis=-2)
        maximum = jnp.max(jnp.where(weights > 0.0, x, -jnp.inf), axis=-2)
        minimum = jnp.where(mass > 0.0, minimum, jnp.zeros_like(minimum))
        maximum = jnp.where(mass > 0.0, maximum, jnp.zeros_like(maximum))
        span = maximum - minimum
        constant = span == 0.0
        safe_span = jnp.where(constant, jnp.ones_like(span), span)
        output_span = self.feature_range[1] - self.feature_range[0]
        divisor = safe_span / output_span
        offset = jnp.full_like(divisor, self.feature_range[0])
        model = FittedMinMaxScaler(
            minimum,
            divisor,
            offset,
            schema=batch.feature_schema,
            case_shape=batch.case_shape,
            feature_range=self.feature_range,
            clip=self.clip,
        )
        diagnostics = _diagnostics(
            batch,
            model.output_schema,
            mass,
            effective,
            valid,
            status,
            method="minmax_scaler",
            constant=constant,
            details=(("feature_range", self.feature_range), ("clip", self.clip)),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="almost-everywhere" if self.clip else "smooth",
                prediction_parameters="smooth",
                fit_features="almost-everywhere",
                fit_targets="none",
                fit_weights="none",
                fit_hyperparameters="none",
                fit_mode="direct",
                conditions=(
                    "Extremum identities and positive-weight support are held fixed.",
                ),
            ),
        )


class FittedMaxAbsScaler(_AbstractAffineTransform):
    def __init__(
        self,
        scale: Array,
        /,
        *,
        schema: FeatureSchema,
        case_shape: tuple[int, ...],
    ):
        self.in_size = len(schema.names)
        self.out_size = len(schema.names)
        self.center = jnp.zeros_like(scale)
        self.scale = jnp.asarray(scale)
        self.output_offset = jnp.zeros_like(scale)
        self.input_schema = schema
        self.output_schema = schema
        self.case_shape = tuple(case_shape)
        self.clip_bounds = None


class MaxAbsScaler(AbstractRecipe):
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(self, *, weight_policy: WeightPolicy = "statistical"):
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        raw_scale = jnp.max(jnp.where(weights > 0.0, jnp.abs(x), 0.0), axis=-2)
        constant = raw_scale == 0.0
        scale = jnp.where(constant, jnp.ones_like(raw_scale), raw_scale)
        model = FittedMaxAbsScaler(
            scale, schema=batch.feature_schema, case_shape=batch.case_shape
        )
        diagnostics = _diagnostics(
            batch,
            model.output_schema,
            mass,
            effective,
            valid,
            status,
            method="maxabs_scaler",
            constant=constant,
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_features="almost-everywhere",
                fit_targets="none",
                fit_weights="none",
                fit_hyperparameters="none",
                fit_mode="direct",
                conditions=("Maximum-absolute-value identities are held fixed.",),
            ),
        )


class FittedRobustScaler(_AbstractAffineTransform):
    quantile_range: tuple[float, float] = eqx.field(static=True)

    def __init__(
        self,
        center: Array,
        scale: Array,
        /,
        *,
        schema: FeatureSchema,
        case_shape: tuple[int, ...],
        quantile_range: tuple[float, float],
    ):
        self.in_size = len(schema.names)
        self.out_size = len(schema.names)
        self.center = jnp.asarray(center)
        self.scale = jnp.asarray(scale)
        self.output_offset = jnp.zeros_like(scale)
        self.input_schema = schema
        self.output_schema = schema
        self.case_shape = tuple(case_shape)
        self.clip_bounds = None
        self.quantile_range = quantile_range


class RobustScaler(AbstractRecipe):
    """Hard weighted median/IQR fit with a smooth affine apply path."""

    with_centering: bool = eqx.field(static=True)
    with_scaling: bool = eqx.field(static=True)
    quantile_range: tuple[float, float] = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple[float, float] = (25.0, 75.0),
        weight_policy: WeightPolicy = "statistical",
    ):
        low, high = float(quantile_range[0]), float(quantile_range[1])
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("quantile_range must be an increasing interval in [0, 100].")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.with_centering = bool(with_centering)
        self.with_scaling = bool(with_scaling)
        self.quantile_range = (low, high)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError("RobustScaler requires real-valued features.")
        probabilities = jnp.asarray(
            [self.quantile_range[0] / 100.0, 0.5, self.quantile_range[1] / 100.0],
            dtype=weights.dtype,
        )
        quantiles = _weighted_quantiles(x, weights, probabilities)
        median = quantiles[..., 1]
        iqr = quantiles[..., 2] - quantiles[..., 0]
        constant = iqr == 0.0
        center = median if self.with_centering else jnp.zeros_like(median)
        scale = (
            jnp.where(constant, jnp.ones_like(iqr), iqr)
            if self.with_scaling
            else jnp.ones_like(iqr)
        )
        model = FittedRobustScaler(
            center,
            scale,
            schema=batch.feature_schema,
            case_shape=batch.case_shape,
            quantile_range=self.quantile_range,
        )
        diagnostics = _diagnostics(
            batch,
            model.output_schema,
            mass,
            effective,
            valid,
            status,
            method="robust_scaler",
            constant=constant,
            details=(("quantile_range", self.quantile_range),),
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="none",
            fit_targets="none",
            fit_weights="none",
            fit_hyperparameters="none",
            fit_mode="stopped",
            nondifferentiable_outputs=("median", "interquartile_range"),
            conditions=(
                "The fitted weighted order statistics are held fixed during apply.",
            ),
        )
        return _fit_result(model, diagnostics, contract)


class FittedNormScaler(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    norm: Literal["l1", "l2", "max"] = eqx.field(static=True)
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)

    def __init__(
        self, size: int, /, *, norm: Literal["l1", "l2", "max"], schema: FeatureSchema
    ):
        self.in_size = int(size)
        self.out_size = int(size)
        self.norm = norm
        self.input_schema = schema
        self.output_schema = schema

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        absolute = jnp.abs(values)
        if self.norm == "l1":
            magnitude = jnp.sum(absolute, axis=-1, keepdims=True)
        elif self.norm == "l2":
            magnitude = jnp.sqrt(jnp.sum(absolute * absolute, axis=-1, keepdims=True))
        else:
            magnitude = jnp.max(absolute, axis=-1, keepdims=True)
        safe = jnp.where(magnitude > 0.0, magnitude, jnp.ones_like(magnitude))
        return values / safe

    def transform(
        self,
        x: Any,
        /,
        *,
        mask: Any | None = None,
        key: Any = None,
    ) -> Array:
        values = _check_features(x, self.in_size)
        if mask is not None:
            values = jnp.where(
                jnp.broadcast_to(jnp.asarray(mask, dtype=bool), values.shape), values, 0
            )
        return self(values, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        raise NotImplementedError(
            "Norm scaling discards vector magnitude and is not bijective."
        )


class NormScaler(AbstractRecipe):
    """Fit-free per-vector L1, L2, or max normalization with schema binding."""

    norm: Literal["l1", "l2", "max"] = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        norm: Literal["l1", "l2", "max"] = "l2",
        *,
        weight_policy: WeightPolicy = "statistical",
    ):
        if norm not in ("l1", "l2", "max"):
            raise ValueError("norm must be 'l1', 'l2', or 'max'.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.norm = norm
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        _x, _weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        model = FittedNormScaler(
            batch.feature_count, norm=self.norm, schema=batch.feature_schema
        )
        diagnostics = _diagnostics(
            batch,
            model.output_schema,
            mass,
            effective,
            valid,
            status,
            method="norm_scaler",
            details=(("norm", self.norm),),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="almost-everywhere",
                prediction_parameters="none",
                fit_mode="direct",
                conditions=("The zero vector maps to itself.",),
            ),
        )


__all__ = [
    "FittedMaxAbsScaler",
    "FittedMinMaxScaler",
    "FittedNormScaler",
    "FittedRobustScaler",
    "FittedStandardScaler",
    "MaxAbsScaler",
    "MinMaxScaler",
    "NormScaler",
    "RobustScaler",
    "StandardScaler",
]
