#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from ..._model import AbstractArrayModel, ModelBinding
from .._batch import MLBatch
from .._contracts import AbstractRecipe, FitResult, GradientContract
from .._schema import FeatureSchema, TargetSchema
from .._sparse_features import SparseFeatures
from ._common import (
    _canonical_feature_output,
    _combine_results,
    _composition_binding,
    _feature_width,
    _predict_values,
    _prepare_input,
    _split_key,
    _transform_batch,
    _validate_model_input,
    CompositionDiagnostics,
    CompositionProvenance,
    ReversibleTransformModel,
)


def _target_feature_schema(batch: MLBatch, width: int, /) -> FeatureSchema:
    names = batch.target_schema.names
    if len(names) != width:
        names = tuple(f"target_{index}" for index in range(width))
    return FeatureSchema(names, kinds=("continuous",) * width, layout_id="target")


def _combine_target_results(
    transform_result: FitResult,
    regressor_result: FitResult,
    /,
) -> tuple[jax.Array, jax.Array, GradientContract]:
    valid, status, base = _combine_results((transform_result, regressor_result))
    transform = transform_result.gradient_contract
    regressor = regressor_result.gradient_contract
    level_order = {
        "none": 0,
        "conditional": 1,
        "almost-everywhere": 2,
        "smooth": 3,
    }

    def minimum(*levels: Any) -> Any:
        return min(levels, key=level_order.__getitem__)

    inverse_condition = (
        "inverse_transform obeys the target transform prediction-gradient contract."
    )
    conditions = (
        base.conditions
        if inverse_condition in base.conditions
        else base.conditions + (inverse_condition,)
    )
    contract = GradientContract(
        prediction_inputs=minimum(
            regressor.prediction_inputs, transform.prediction_inputs
        ),
        prediction_parameters=minimum(
            regressor.prediction_parameters,
            transform.prediction_parameters,
        ),
        fit_features=regressor.fit_features,
        fit_targets=minimum(
            transform.fit_features,
            transform.prediction_inputs,
            regressor.fit_targets,
        ),
        fit_weights=minimum(transform.fit_weights, regressor.fit_weights),
        fit_hyperparameters=minimum(
            transform.fit_hyperparameters,
            regressor.fit_hyperparameters,
        ),
        fit_mode=base.fit_mode,
        nondifferentiable_outputs=base.nondifferentiable_outputs,
        conditions=conditions,
    )
    return valid, status, contract


def _targets_as_feature_batch(batch: MLBatch, /) -> tuple[MLBatch, tuple[int, ...]]:
    targets = batch.require_targets()
    target_shape = batch.target_shape
    if target_shape is None:
        raise ValueError("TransformedTargetRegressor requires targets.")
    width = math.prod(target_shape) if target_shape else 1
    leading = batch.case_shape + (batch.sample_count,)
    values = jnp.asarray(targets).reshape(leading + (width,))
    mask = (
        jnp.ones(values.shape, dtype=bool)
        if batch.target_mask is None
        else jnp.asarray(batch.target_mask, dtype=bool).reshape(values.shape)
    )
    target_batch = MLBatch(
        values,
        feature_mask=mask,
        sample_mask=batch.sample_mask,
        sample_weight=batch.sample_weight,
        measure_weight=batch.measure_weight,
        groups=batch.groups,
        feature_schema=_target_feature_schema(batch, width),
    )
    return target_batch, target_shape


def _regression_batch(
    source: MLBatch,
    transformed_targets: MLBatch,
    /,
    *,
    scalar_target: bool,
) -> MLBatch:
    if isinstance(transformed_targets.features, SparseFeatures):
        raise TypeError(
            "Sparse transformed targets are unsupported; no implicit densification "
            "is performed."
        )
    targets = transformed_targets.features
    target_mask = transformed_targets.feature_mask
    if scalar_target and transformed_targets.feature_count == 1:
        targets = jnp.squeeze(targets, axis=-1)
        target_mask = jnp.squeeze(target_mask, axis=-1)
    return MLBatch(
        source.features,
        targets,
        feature_mask=(
            None if isinstance(source.features, SparseFeatures) else source.feature_mask
        ),
        target_mask=target_mask,
        sample_mask=source.sample_mask,
        sample_weight=source.sample_weight,
        measure_weight=source.measure_weight,
        groups=source.groups,
        feature_schema=source.feature_schema,
        target_schema=source.target_schema,
    )


def _prediction_features(
    prediction: Any,
    /,
    *,
    out_size: Any,
) -> tuple[jax.Array, tuple[int, ...]]:
    array = jnp.asarray(prediction)
    width = _feature_width(out_size, role="Regressor output")
    if out_size == "scalar":
        leading = tuple(int(size) for size in array.shape)
        return jnp.expand_dims(array, axis=-1), leading
    if array.ndim < 1 or int(array.shape[-1]) != width:
        raise ValueError(
            f"Regressor output must end in width {width}; got {array.shape}."
        )
    return array, tuple(int(size) for size in array.shape[:-1])


def _inverse_pointwise(
    transformer: ReversibleTransformModel,
    values: jax.Array,
    transform_out_size: Any,
    /,
    *,
    key: Any,
) -> jax.Array:
    width = int(values.shape[-1])
    flat = values.reshape((-1, width))
    count = int(flat.shape[0])
    if key is None:
        mapped = jax.vmap(
            lambda row: transformer.inverse_transform(
                _prepare_input(row, transform_out_size), key=None
            )
        )(flat)
    else:
        keys = jax.random.split(key, count)
        mapped = jax.vmap(
            lambda row, point_key: transformer.inverse_transform(
                _prepare_input(row, transform_out_size), key=point_key
            )
        )(flat, keys)
    return jnp.asarray(mapped).reshape(values.shape[:-1] + tuple(mapped.shape[1:]))


def _inverse_targets(
    transformer: AbstractArrayModel,
    prediction: Any,
    target_shape: tuple[int, ...],
    /,
    *,
    prediction_out_size: Any,
    key: Any,
    composed_blockwise: bool,
) -> jax.Array:
    if not isinstance(transformer, ReversibleTransformModel):
        raise TypeError(
            "The fitted target transform does not implement ReversibleTransformModel."
        )
    values, leading = _prediction_features(prediction, out_size=prediction_out_size)
    binding = transformer.input_binding()
    if binding.input_mode != "flat" or binding.batch_mode == "axis":
        raise TypeError(
            "Reversible target transforms require flat pointwise or blockwise bindings."
        )
    if composed_blockwise and binding.batch_mode == "pointwise":
        inverted = _inverse_pointwise(transformer, values, transformer.out_size, key=key)
    else:
        inverted = transformer.inverse_transform(
            _prepare_input(values, transformer.out_size), key=key
        )
    restored = _canonical_feature_output(
        inverted,
        leading_shape=leading,
        out_size=transformer.in_size,
    )
    if isinstance(restored, SparseFeatures):
        raise TypeError("A reversible target transform must return dense targets.")
    if not target_shape:
        return jnp.squeeze(restored, axis=-1)
    return jnp.asarray(restored).reshape(leading + target_shape)


class FittedTransformedTargetRegressor(AbstractArrayModel):
    """Fitted regressor paired with the exact fitted inverse target transform."""

    regressor: AbstractArrayModel
    transformer: AbstractArrayModel
    provenance: CompositionProvenance
    feature_schema: FeatureSchema
    transform_input_schema: FeatureSchema
    transform_output_schema: FeatureSchema
    target_schema: TargetSchema
    target_shape: tuple[int, ...] = eqx.field(static=True)
    gradient_contract: GradientContract
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding: ModelBinding = eqx.field(static=True)  # ty: ignore[invalid-attribute-override]

    def __init__(
        self,
        regressor: AbstractArrayModel,
        transformer: AbstractArrayModel,
        fit_results: tuple[FitResult, FitResult],
        /,
        *,
        feature_schema: FeatureSchema,
        transform_input_schema: FeatureSchema,
        transform_output_schema: FeatureSchema,
        target_schema: TargetSchema,
        target_shape: tuple[int, ...],
        gradient_contract: GradientContract,
    ):
        if not isinstance(regressor, AbstractArrayModel):
            raise TypeError("regressor must be an AbstractArrayModel.")
        if not isinstance(transformer, AbstractArrayModel):
            raise TypeError("transformer must be an AbstractArrayModel.")
        if not isinstance(transformer, ReversibleTransformModel):
            raise TypeError(
                "The fitted target transformer must implement inverse_transform."
            )
        if len(fit_results) != 2:
            raise ValueError("Target transformer and regressor fit results are required.")
        _validate_model_input(regressor, len(feature_schema.names), schema=feature_schema)
        _validate_model_input(
            transformer,
            len(transform_input_schema.names),
            schema=transform_input_schema,
        )
        transformed_width = _feature_width(
            transformer.out_size, role="Target transform output"
        )
        regressor_width = _feature_width(regressor.out_size, role="Regressor output")
        if regressor_width != transformed_width:
            raise ValueError(
                "Regressor output width must equal the transformed target width."
            )
        if len(transform_output_schema.names) != transformed_width:
            raise ValueError(
                "Target transform output schema does not match its out_size."
            )
        self.regressor = regressor
        self.transformer = transformer
        self.provenance = CompositionProvenance(("transformer", "regressor"), fit_results)
        self.feature_schema = feature_schema
        self.transform_input_schema = transform_input_schema
        self.transform_output_schema = transform_output_schema
        self.target_schema = target_schema
        self.target_shape = tuple(target_shape)
        self.gradient_contract = gradient_contract
        self.in_size = regressor.in_size
        if not target_shape:
            self.out_size = "scalar"
        elif len(target_shape) == 1:
            self.out_size = int(target_shape[0])
        else:
            self.out_size = tuple(target_shape)
        self._input_binding = _composition_binding((regressor, transformer))

    @property
    def fit_results(self) -> tuple[FitResult, ...]:
        return self.provenance.results

    @property
    def transformer_result(self) -> FitResult:
        return self.fit_results[0]

    @property
    def regressor_result(self) -> FitResult:
        return self.fit_results[1]

    def __call__(self, x: Any, /, *, key: Any = None):
        regressor_key, inverse_key = _split_key(key, 2)
        blockwise = self._input_binding.batch_mode == "blockwise"
        prediction = _predict_values(
            self.regressor,
            x,
            key=regressor_key,
            composed_blockwise=blockwise,
        )
        return _inverse_targets(
            self.transformer,
            prediction,
            self.target_shape,
            prediction_out_size=self.regressor.out_size,
            key=inverse_key,
            composed_blockwise=blockwise,
        )


class TransformedTargetRegressor(AbstractRecipe):
    """Fit a reversible target transform and a regressor without feature leakage."""

    regressor: AbstractRecipe
    transformer: AbstractRecipe

    def __init__(
        self,
        regressor: AbstractRecipe,
        transformer: AbstractRecipe,
        /,
    ):
        if not isinstance(regressor, AbstractRecipe):
            raise TypeError("regressor must be an AbstractRecipe.")
        if not isinstance(transformer, AbstractRecipe):
            raise TypeError("transformer must be an AbstractRecipe.")
        self.regressor = regressor
        self.transformer = transformer

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if not isinstance(batch, MLBatch):
            raise TypeError(
                "TransformedTargetRegressor.fit_batch requires an already-selected "
                "MLBatch."
            )
        if batch.target_schema.kind not in ("continuous", "count"):
            raise ValueError(
                "TransformedTargetRegressor supports only continuous or count targets."
            )
        target_batch, target_shape = _targets_as_feature_batch(batch)
        transform_fit_key, transform_key, regressor_key = _split_key(key, 3)
        transform_result = self.transformer.fit_batch(target_batch, key=transform_fit_key)
        if not isinstance(transform_result, FitResult):
            raise TypeError("Target transformer did not return a FitResult.")
        transform_model = transform_result.as_trainable()
        if not isinstance(transform_model, ReversibleTransformModel):
            raise TypeError(
                "TransformedTargetRegressor requires a fitted transformer with "
                "inverse_transform(values, *, key=None)."
            )
        transformed = _transform_batch(transform_model, target_batch, key=transform_key)
        regression_batch = _regression_batch(
            batch,
            transformed,
            scalar_target=(target_shape == ()),
        )
        regressor_result = self.regressor.fit_batch(regression_batch, key=regressor_key)
        if not isinstance(regressor_result, FitResult):
            raise TypeError("Regressor did not return a FitResult.")
        regressor_model = regressor_result.as_trainable()
        results = (transform_result, regressor_result)
        valid, status, contract = _combine_target_results(
            transform_result, regressor_result
        )
        model = FittedTransformedTargetRegressor(
            regressor_model,
            transform_model,
            results,
            feature_schema=batch.feature_schema,
            transform_input_schema=target_batch.feature_schema,
            transform_output_schema=transformed.feature_schema,
            target_schema=batch.target_schema,
            target_shape=target_shape,
            gradient_contract=contract,
        )
        diagnostics = CompositionDiagnostics(
            ("transformer", "regressor"),
            results,
            valid=valid,
            status=status,
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="transformed_target_regressor",
            gradient_contract=contract,
        )


__all__ = [
    "FittedTransformedTargetRegressor",
    "TransformedTargetRegressor",
]
