#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from numbers import Number
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._schema import FeatureSchema
from ._common import (
    _align_parameter,
    _check_features,
    _dense_batch,
    _diagnostics,
    _feature_observations,
    _fit_result,
    _weighted_mean,
    _weighted_quantiles,
)


UnknownPolicy = Literal["fail", "indicator"]
ImputationStrategy = Literal["mean", "median", "most_frequent", "constant"]


class CategoricalSchema(StrictModule):
    """Explicit, fixed-capacity numeric vocabularies for categorical features."""

    names: tuple[str, ...] = eqx.field(static=True)
    categories: tuple[tuple[Number, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        categories: Sequence[Sequence[Number]],
        /,
        *,
        names: Sequence[str] | None = None,
    ):
        categories_ = tuple(tuple(values) for values in categories)
        if not categories_ or any(not values for values in categories_):
            raise ValueError("Every categorical feature requires a nonempty vocabulary.")
        for values in categories_:
            if any(not isinstance(value, Number) for value in values):
                raise TypeError(
                    "JAX categorical vocabularies contain numeric scalars only."
                )
            if len(set(values)) != len(values):
                raise ValueError("Category values must be unique within each feature.")
            array = jnp.asarray(values)
            if bool(jnp.any(~jnp.isfinite(array))):
                raise ValueError("Category values must be finite.")
        names_ = (
            tuple(f"feature_{index}" for index in range(len(categories_)))
            if names is None
            else tuple(str(name) for name in names)
        )
        if len(names_) != len(categories_) or any(not name for name in names_):
            raise ValueError("Categorical names must align with vocabularies.")
        if len(set(names_)) != len(names_):
            raise ValueError("Categorical names must be unique.")
        self.names = names_
        self.categories = categories_

    @property
    def feature_count(self) -> int:
        return len(self.categories)

    @property
    def category_counts(self) -> tuple[int, ...]:
        return tuple(len(values) for values in self.categories)


class CategoricalDiagnostics(StrictModule):
    """Exact known/unknown category mass and schema expansion diagnostics."""

    valid: Array
    status: Array
    category_weight: Array
    unknown_weight: Array
    unknown_count: Array
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)
    category_counts: tuple[int, ...] = eqx.field(static=True)
    unknown_policy: UnknownPolicy = eqx.field(static=True)
    method: str = eqx.field(static=True)
    input_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        category_weight: Any,
        unknown_weight: Any,
        unknown_count: Any,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
        category_counts: tuple[int, ...],
        unknown_policy: UnknownPolicy,
        method: str,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.category_weight = jnp.asarray(category_weight)
        self.unknown_weight = jnp.asarray(unknown_weight)
        self.unknown_count = jnp.asarray(unknown_count, dtype=jnp.int32)
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.category_counts = tuple(category_counts)
        self.unknown_policy = unknown_policy
        self.method = str(method)
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)


def _category_bank(schema: CategoricalSchema, dtype) -> tuple[Array, Array]:
    capacity = max(schema.category_counts)
    values = []
    valid = []
    for categories in schema.categories:
        pad = capacity - len(categories)
        values.append(tuple(categories) + (0,) * pad)
        valid.append((True,) * len(categories) + (False,) * pad)
    flat_categories = tuple(
        value for categories in schema.categories for value in categories
    )
    category_dtype = jnp.result_type(dtype, *flat_categories)
    return jnp.asarray(values, dtype=category_dtype), jnp.asarray(valid, dtype=bool)


def _category_matches(
    x: Array, categories: Array, category_valid: Array
) -> tuple[Array, Array]:
    matches = (x[..., :, None] == categories) & category_valid
    known = jnp.any(matches, axis=-1)
    return matches, known


def _categorical_fit_diagnostics(
    batch: MLBatch,
    schema: CategoricalSchema,
    output_schema: FeatureSchema,
    unknown_policy: UnknownPolicy,
    /,
    *,
    weight_policy: WeightPolicy,
    method: str,
) -> CategoricalDiagnostics:
    x = _dense_batch(batch)
    categories, category_valid = _category_bank(schema, x.dtype)
    matches, known = _category_matches(x, categories, category_valid)
    raw_weight = batch.effective_weight(weight_policy)
    weight_ok = jnp.isfinite(raw_weight) & (raw_weight >= 0.0)
    included = batch.sample_mask & weight_ok
    feature_included = included[..., None] & batch.feature_mask & jnp.isfinite(x)
    weights = jnp.where(feature_included, raw_weight[..., None], 0.0)
    category_weight = jnp.sum(weights[..., None] * matches, axis=-3)
    unknown_weight = jnp.sum(jnp.where(~known, weights, 0.0), axis=-2)
    unknown_active = batch.sample_mask[..., None] & batch.feature_mask & ~known
    unknown_count = jnp.sum(unknown_active, axis=-2, dtype=jnp.int32)
    finite_weight = jnp.all(jnp.isfinite(raw_weight) | ~batch.sample_mask, axis=-1)
    nonnegative_weight = jnp.all((raw_weight >= 0.0) | ~batch.sample_mask, axis=-1)
    weights_valid = finite_weight & nonnegative_weight
    fail_valid = (
        jnp.all(unknown_count == 0, axis=-1)
        if unknown_policy == "fail"
        else jnp.asarray(True)
    )
    valid = weights_valid & fail_valid
    status = jnp.where(
        ~finite_weight,
        ML_NONFINITE,
        jnp.where(~nonnegative_weight | ~fail_valid, ML_INFEASIBLE, ML_SUCCESS),
    )
    return CategoricalDiagnostics(
        valid=valid,
        status=status,
        category_weight=category_weight,
        unknown_weight=unknown_weight,
        unknown_count=unknown_count,
        input_schema=batch.feature_schema,
        output_schema=output_schema,
        category_counts=schema.category_counts,
        unknown_policy=unknown_policy,
        method=method,
        input_shape=batch.case_shape + (batch.sample_count, batch.feature_count),
        output_shape=batch.case_shape + (batch.sample_count, len(output_schema.names)),
    )


class FittedSimpleImputer(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    fill_values: Array
    missing_values: Number = eqx.field(static=True)
    missing_is_nan: bool = eqx.field(static=True)
    add_indicator: bool = eqx.field(static=True)
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        fill_values: Array,
        /,
        *,
        missing_values: Number,
        missing_is_nan: bool,
        add_indicator: bool,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
        case_shape: tuple[int, ...],
    ):
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self.fill_values = jnp.asarray(fill_values)
        self.missing_values = missing_values
        self.missing_is_nan = bool(missing_is_nan)
        self.add_indicator = bool(add_indicator)
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.case_shape = tuple(case_shape)

    def _missing(self, values: Array, mask: Any | None) -> Array:
        missing = (
            jnp.isnan(values) if self.missing_is_nan else values == self.missing_values
        )
        if mask is not None:
            missing = missing | ~jnp.broadcast_to(
                jnp.asarray(mask, dtype=bool), values.shape
            )
        return missing

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        return self.transform(x, key=key)

    def transform(
        self,
        x: Any,
        /,
        *,
        mask: Any | None = None,
        key: Any = None,
    ) -> Array:
        del key
        values = _check_features(x, self.in_size)
        missing = self._missing(values, mask)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values) & ~missing),
            "SimpleImputer encountered a non-finite value that is not its missing sentinel.",
        )
        missing = self._missing(values, mask)
        fill = _align_parameter(self.fill_values, values, self.case_shape)
        imputed = jnp.where(missing, fill, values)
        if self.add_indicator:
            imputed = jnp.concatenate(
                (imputed, missing.astype(imputed.real.dtype)), axis=-1
            )
        return imputed

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        raise NotImplementedError(
            "Imputation loses missing-value identity and is not bijective."
        )


class SimpleImputer(AbstractRecipe):
    """Mask-aware weighted scalar imputation with an optional missingness channel."""

    strategy: ImputationStrategy = eqx.field(static=True)
    fill_value: Number = eqx.field(static=True)
    missing_values: Number = eqx.field(static=True)
    missing_is_nan: bool = eqx.field(static=True)
    add_indicator: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        strategy: ImputationStrategy = "mean",
        fill_value: Number = 0.0,
        missing_values: Number = float("nan"),
        add_indicator: bool = False,
        weight_policy: WeightPolicy = "statistical",
    ):
        if strategy not in ("mean", "median", "most_frequent", "constant"):
            raise ValueError("Unsupported imputation strategy.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        if not isinstance(fill_value, Number) or not isinstance(missing_values, Number):
            raise TypeError(
                "JAX-native imputation sentinels and fill values must be numeric."
            )
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        self.missing_is_nan = bool(jnp.isnan(jnp.asarray(missing_values)))
        self.add_indicator = bool(add_indicator)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        raw = _dense_batch(batch)
        available = ~jnp.isnan(raw) if self.missing_is_nan else raw != self.missing_values
        x, weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy, extra_mask=available
        )
        if self.strategy == "mean":
            fill = _weighted_mean(x, weights)
        elif self.strategy == "median":
            if jnp.issubdtype(x.dtype, jnp.complexfloating):
                raise TypeError("Median imputation requires real-valued features.")
            fill = _weighted_quantiles(
                x, weights, jnp.asarray([0.5], dtype=weights.dtype)
            )[..., 0]
        elif self.strategy == "most_frequent":
            candidates = x[..., :, None, :]
            observations = x[..., None, :, :]
            equality = candidates == observations
            candidate_mass = jnp.sum(equality * weights[..., None, :, :], axis=-2)
            indices = jnp.argmax(candidate_mass, axis=-2)
            fill = jnp.take_along_axis(x, indices[..., None, :], axis=-2)[..., 0, :]
        else:
            fill = jnp.broadcast_to(
                jnp.asarray(self.fill_value, dtype=jnp.result_type(x, self.fill_value)),
                mass.shape,
            )
        if self.strategy == "constant":
            raw_weight = batch.effective_weight(self.weight_policy)
            finite_weight = jnp.all(
                jnp.isfinite(raw_weight) | ~batch.sample_mask, axis=-1
            )
            nonnegative_weight = jnp.all(
                (raw_weight >= 0.0) | ~batch.sample_mask, axis=-1
            )
            valid = finite_weight & nonnegative_weight
            status = jnp.where(
                ~finite_weight,
                ML_NONFINITE,
                jnp.where(nonnegative_weight, ML_SUCCESS, ML_INFEASIBLE),
            )
        if self.add_indicator:
            names = batch.feature_schema.names + tuple(
                f"{name}_missing" for name in batch.feature_schema.names
            )
            kinds = batch.feature_schema.kinds + ("boolean",) * batch.feature_count
            output_schema = FeatureSchema(
                names, kinds=kinds, layout_id=batch.feature_schema.layout_id
            )
        else:
            output_schema = batch.feature_schema
        model = FittedSimpleImputer(
            fill,
            missing_values=self.missing_values,
            missing_is_nan=self.missing_is_nan,
            add_indicator=self.add_indicator,
            input_schema=batch.feature_schema,
            output_schema=output_schema,
            case_shape=batch.case_shape,
        )
        diagnostics = _diagnostics(
            batch,
            output_schema,
            mass,
            effective,
            valid,
            status,
            method="simple_imputer",
            details=(("strategy", self.strategy), ("add_indicator", self.add_indicator)),
        )
        contract = (
            GradientContract(
                prediction_inputs="conditional",
                prediction_parameters="smooth",
                fit_features="conditional",
                fit_targets="none",
                fit_weights="conditional",
                fit_hyperparameters="none",
                fit_mode="direct",
                conditions=(
                    "Missingness, masks, and positive-weight support are held fixed.",
                ),
            )
            if self.strategy == "mean"
            else GradientContract(
                prediction_inputs="conditional",
                prediction_parameters="smooth",
                fit_features="none",
                fit_targets="none",
                fit_weights="none",
                fit_hyperparameters="none",
                fit_mode="stopped",
                nondifferentiable_outputs=("imputation_choice",),
                conditions=("The hard fitted imputation choice is fixed during apply.",),
            )
        )
        return _fit_result(model, diagnostics, contract)


class FittedOrdinalEncoder(AbstractArrayModel, NonTrainableState):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    categories: Array
    category_valid: Array
    unknown_policy: UnknownPolicy = eqx.field(static=True)
    unknown_value: int = eqx.field(static=True)
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)

    def __init__(
        self,
        categories: Array,
        category_valid: Array,
        /,
        *,
        unknown_policy: UnknownPolicy,
        unknown_value: int,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
    ):
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self.categories = jnp.asarray(categories)
        self.category_valid = jnp.asarray(category_valid, dtype=bool)
        self.unknown_policy = unknown_policy
        self.unknown_value = int(unknown_value)
        self.input_schema = input_schema
        self.output_schema = output_schema

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        matches, known = _category_matches(values, self.categories, self.category_valid)
        if self.unknown_policy == "fail":
            values = eqx.error_if(
                values, jnp.any(~known), "OrdinalEncoder encountered an unknown category."
            )
            matches, known = _category_matches(
                values, self.categories, self.category_valid
            )
        codes = jnp.where(known, jnp.argmax(matches, axis=-1), self.unknown_value).astype(
            jnp.int32
        )
        if self.unknown_policy == "indicator":
            codes = jnp.concatenate((codes, (~known).astype(jnp.int32)), axis=-1)
        return codes

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        encoded = _check_features(x, self.out_size)
        codes = encoded[..., : self.in_size]
        rounded = jnp.rint(codes).astype(jnp.int32)
        invalid = (codes != rounded) | (rounded < 0)
        counts = jnp.sum(self.category_valid, axis=-1)
        invalid = invalid | (rounded >= counts)
        if self.unknown_policy == "indicator":
            invalid = invalid | (encoded[..., self.in_size :] != 0)
        encoded = eqx.error_if(
            encoded,
            jnp.any(invalid),
            "Ordinal codes are not invertible known categories.",
        )
        rounded = jnp.rint(encoded[..., : self.in_size]).astype(jnp.int32)
        categories = jnp.broadcast_to(
            self.categories, rounded.shape + (self.categories.shape[-1],)
        )
        return jnp.take_along_axis(categories, rounded[..., None], axis=-1)[..., 0]


class OrdinalEncoder(AbstractRecipe):
    schema: CategoricalSchema = eqx.field(static=True)
    unknown_policy: UnknownPolicy = eqx.field(static=True)
    unknown_value: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        schema: CategoricalSchema,
        /,
        *,
        unknown_policy: UnknownPolicy = "fail",
        unknown_value: int = -1,
        weight_policy: WeightPolicy = "statistical",
    ):
        if unknown_policy not in ("fail", "indicator"):
            raise ValueError("unknown_policy must be 'fail' or 'indicator'.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.schema = schema
        self.unknown_policy = unknown_policy
        self.unknown_value = int(unknown_value)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if batch.feature_count != self.schema.feature_count:
            raise ValueError("Categorical schema does not match the input feature count.")
        x = _dense_batch(batch)
        categories, category_valid = _category_bank(self.schema, x.dtype)
        names = self.schema.names
        kinds = ("ordinal",) * batch.feature_count
        if self.unknown_policy == "indicator":
            names = names + tuple(f"{name}_unknown" for name in names)
            kinds = kinds + ("boolean",) * batch.feature_count
        output_schema = FeatureSchema(
            names, kinds=kinds, layout_id=batch.feature_schema.layout_id
        )
        model = FittedOrdinalEncoder(
            categories,
            category_valid,
            unknown_policy=self.unknown_policy,
            unknown_value=self.unknown_value,
            input_schema=batch.feature_schema,
            output_schema=output_schema,
        )
        diagnostics = _categorical_fit_diagnostics(
            batch,
            self.schema,
            output_schema,
            self.unknown_policy,
            weight_policy=self.weight_policy,
            method="ordinal_encoder",
        )
        return FitResult(
            model,
            diagnostics,
            valid=diagnostics.valid,
            status=diagnostics.status,
            method=diagnostics.method,
            gradient_contract=GradientContract(
                prediction_inputs="none",
                prediction_parameters="none",
                fit_mode="stopped",
                nondifferentiable_outputs=("ordinal_codes", "unknown_indicators"),
            ),
        )


class FittedOneHotEncoder(AbstractArrayModel, NonTrainableState):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    categories: Array
    category_valid: Array
    offsets: tuple[int, ...] = eqx.field(static=True)
    unknown_policy: UnknownPolicy = eqx.field(static=True)
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)

    def __init__(
        self,
        categories: Array,
        category_valid: Array,
        /,
        *,
        offsets: tuple[int, ...],
        unknown_policy: UnknownPolicy,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
    ):
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self.categories = jnp.asarray(categories)
        self.category_valid = jnp.asarray(category_valid, dtype=bool)
        self.offsets = tuple(offsets)
        self.unknown_policy = unknown_policy
        self.input_schema = input_schema
        self.output_schema = output_schema

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        matches, known = _category_matches(values, self.categories, self.category_valid)
        if self.unknown_policy == "fail":
            values = eqx.error_if(
                values, jnp.any(~known), "OneHotEncoder encountered an unknown category."
            )
            matches, known = _category_matches(
                values, self.categories, self.category_valid
            )
        pieces = [
            matches[..., index, : self.offsets[index + 1] - self.offsets[index]]
            for index in range(self.in_size)
        ]
        if self.unknown_policy == "indicator":
            pieces.extend(
                (~known[..., index])[..., None] for index in range(self.in_size)
            )
        dtype = (
            values.real.dtype
            if jnp.issubdtype(values.dtype, jnp.inexact)
            else jnp.dtype(float)
        )
        return jnp.concatenate(pieces, axis=-1).astype(dtype)

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        encoded = _check_features(x, self.out_size)
        invalid = jnp.zeros(encoded.shape[:-1], dtype=bool)
        for feature in range(self.in_size):
            start, stop = self.offsets[feature], self.offsets[feature + 1]
            block = encoded[..., start:stop]
            valid = jnp.sum(block == 1, axis=-1) == 1
            valid = valid & jnp.all((block == 0) | (block == 1), axis=-1)
            if self.unknown_policy == "indicator":
                valid = valid & (encoded[..., self.offsets[-1] + feature] == 0)
            invalid = invalid | ~valid
        checked = eqx.error_if(
            encoded,
            jnp.any(invalid),
            "One-hot vectors are not invertible known categories.",
        )
        decoded = []
        for feature in range(self.in_size):
            start, stop = self.offsets[feature], self.offsets[feature + 1]
            index = jnp.argmax(checked[..., start:stop], axis=-1)
            decoded.append(self.categories[feature, index])
        return jnp.stack(decoded, axis=-1)


class OneHotEncoder(AbstractRecipe):
    schema: CategoricalSchema = eqx.field(static=True)
    unknown_policy: UnknownPolicy = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        schema: CategoricalSchema,
        /,
        *,
        unknown_policy: UnknownPolicy = "fail",
        weight_policy: WeightPolicy = "statistical",
    ):
        if unknown_policy not in ("fail", "indicator"):
            raise ValueError("unknown_policy must be 'fail' or 'indicator'.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.schema = schema
        self.unknown_policy = unknown_policy
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if batch.feature_count != self.schema.feature_count:
            raise ValueError("Categorical schema does not match the input feature count.")
        x = _dense_batch(batch)
        categories, category_valid = _category_bank(self.schema, x.dtype)
        offsets = [0]
        names = []
        for name, values in zip(self.schema.names, self.schema.categories, strict=True):
            names.extend(f"{name}={value}" for value in values)
            offsets.append(offsets[-1] + len(values))
        if self.unknown_policy == "indicator":
            names.extend(f"{name}_unknown" for name in self.schema.names)
        output_schema = FeatureSchema(
            tuple(names),
            kinds=("boolean",) * len(names),
            layout_id=batch.feature_schema.layout_id,
        )
        model = FittedOneHotEncoder(
            categories,
            category_valid,
            offsets=tuple(offsets),
            unknown_policy=self.unknown_policy,
            input_schema=batch.feature_schema,
            output_schema=output_schema,
        )
        diagnostics = _categorical_fit_diagnostics(
            batch,
            self.schema,
            output_schema,
            self.unknown_policy,
            weight_policy=self.weight_policy,
            method="onehot_encoder",
        )
        return FitResult(
            model,
            diagnostics,
            valid=diagnostics.valid,
            status=diagnostics.status,
            method=diagnostics.method,
            gradient_contract=GradientContract(
                prediction_inputs="none",
                prediction_parameters="none",
                fit_mode="stopped",
                nondifferentiable_outputs=("one_hot_codes", "unknown_indicators"),
            ),
        )


class FittedTargetEncoder(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    category_schema: CategoricalSchema = eqx.field(static=True)
    encodings: Array
    global_mean: Array
    unknown_policy: UnknownPolicy = eqx.field(static=True)
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        encodings: Array,
        global_mean: Array,
        /,
        *,
        unknown_policy: UnknownPolicy,
        category_schema: CategoricalSchema,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
        case_shape: tuple[int, ...],
    ):
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self.category_schema = category_schema
        self.encodings = jnp.asarray(encodings)
        self.global_mean = jnp.asarray(global_mean)
        self.unknown_policy = unknown_policy
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.case_shape = tuple(case_shape)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        categories, category_valid = _category_bank(self.category_schema, values.dtype)
        matches, known = _category_matches(values, categories, category_valid)
        if self.unknown_policy == "fail":
            values = eqx.error_if(
                values, jnp.any(~known), "TargetEncoder encountered an unknown category."
            )
            matches, known = _category_matches(values, categories, category_valid)
        indices = jnp.argmax(matches, axis=-1)
        encodings = _align_parameter(
            self.encodings, values, self.case_shape, trailing_rank=2
        )
        encodings = jnp.broadcast_to(
            encodings, values.shape + (self.encodings.shape[-1],)
        )
        transformed = jnp.take_along_axis(encodings, indices[..., None], axis=-1)[..., 0]
        global_mean = _align_parameter(
            self.global_mean[..., None], values, self.case_shape
        )
        transformed = jnp.where(known, transformed, global_mean)
        if self.unknown_policy == "indicator":
            transformed = jnp.concatenate(
                (transformed, (~known).astype(transformed.real.dtype)), axis=-1
            )
        return transformed

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        raise NotImplementedError("Target encoding is many-to-one and is not invertible.")


class TargetEncoder(AbstractRecipe):
    """Weighted smoothed category means for a scalar supervised target."""

    schema: CategoricalSchema = eqx.field(static=True)
    smoothing: Array
    unknown_policy: UnknownPolicy = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        schema: CategoricalSchema,
        /,
        *,
        smoothing: ArrayLike = 1.0,
        unknown_policy: UnknownPolicy = "fail",
        weight_policy: WeightPolicy = "statistical",
    ):
        smoothing_ = jnp.asarray(smoothing, dtype=float)
        if smoothing_.ndim != 0:
            raise ValueError("smoothing must be scalar.")
        smoothing_ = eqx.error_if(
            smoothing_,
            ~jnp.isfinite(smoothing_) | (smoothing_ < 0.0),
            "smoothing must be finite and nonnegative.",
        )
        if unknown_policy not in ("fail", "indicator"):
            raise ValueError("unknown_policy must be 'fail' or 'indicator'.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.schema = schema
        self.smoothing = smoothing_
        self.unknown_policy = unknown_policy
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if batch.feature_count != self.schema.feature_count:
            raise ValueError("Categorical schema does not match the input feature count.")
        targets = batch.require_targets()
        if batch.target_shape == (1,):
            targets = targets[..., 0]
            target_mask = batch.target_mask[..., 0]
        elif batch.target_shape == ():
            target_mask = batch.target_mask
        else:
            raise ValueError("TargetEncoder requires one scalar target per sample.")
        x = _dense_batch(batch)
        categories, category_valid = _category_bank(self.schema, x.dtype)
        matches, known = _category_matches(x, categories, category_valid)
        raw_weight = batch.effective_weight(self.weight_policy)
        weight_ok = jnp.isfinite(raw_weight) & (raw_weight >= 0.0)
        included = batch.sample_mask & target_mask & weight_ok & jnp.isfinite(targets)
        target_weight = jnp.where(included, raw_weight, 0.0)
        safe_targets = jnp.where(included, targets, 0)
        total_mass = jnp.sum(target_weight, axis=-1)
        global_mean = jnp.sum(target_weight * safe_targets, axis=-1) / jnp.maximum(
            total_mass, jnp.finfo(target_weight.dtype).tiny
        )
        membership_weight = target_weight[..., :, None, None] * matches
        category_mass = jnp.sum(membership_weight, axis=-3)
        category_sum = jnp.sum(
            membership_weight * safe_targets[..., :, None, None], axis=-3
        )
        denominator = category_mass + self.smoothing
        encodings = jnp.where(
            denominator > 0.0,
            (category_sum + self.smoothing * global_mean[..., None, None])
            / jnp.maximum(denominator, jnp.finfo(target_weight.dtype).tiny),
            global_mean[..., None, None],
        )
        names = tuple(f"{name}_target" for name in self.schema.names)
        kinds = ("continuous",) * batch.feature_count
        if self.unknown_policy == "indicator":
            names = names + tuple(f"{name}_unknown" for name in self.schema.names)
            kinds = kinds + ("boolean",) * batch.feature_count
        output_schema = FeatureSchema(
            names, kinds=kinds, layout_id=batch.feature_schema.layout_id
        )
        model = FittedTargetEncoder(
            encodings,
            global_mean,
            unknown_policy=self.unknown_policy,
            category_schema=self.schema,
            input_schema=batch.feature_schema,
            output_schema=output_schema,
            case_shape=batch.case_shape,
        )
        diagnostics = _categorical_fit_diagnostics(
            batch,
            self.schema,
            output_schema,
            self.unknown_policy,
            weight_policy=self.weight_policy,
            method="target_encoder",
        )
        target_valid = total_mass > 0.0
        valid = diagnostics.valid & target_valid
        status = jnp.where(
            diagnostics.valid,
            jnp.where(target_valid, ML_SUCCESS, ML_INSUFFICIENT_DATA),
            diagnostics.status,
        )
        diagnostics = CategoricalDiagnostics(
            valid=valid,
            status=status,
            category_weight=category_mass,
            unknown_weight=diagnostics.unknown_weight,
            unknown_count=diagnostics.unknown_count,
            input_schema=batch.feature_schema,
            output_schema=output_schema,
            category_counts=self.schema.category_counts,
            unknown_policy=self.unknown_policy,
            method="target_encoder",
            input_shape=batch.case_shape + (batch.sample_count, batch.feature_count),
            output_shape=batch.case_shape
            + (batch.sample_count, len(output_schema.names)),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="target_encoder",
            gradient_contract=GradientContract(
                prediction_inputs="none",
                prediction_parameters="smooth",
                fit_features="none",
                fit_targets="conditional",
                fit_weights="conditional",
                fit_hyperparameters="conditional",
                fit_mode="direct",
                nondifferentiable_outputs=("category_membership", "unknown_indicators"),
                conditions=("Category membership and target masks are held fixed.",),
            ),
        )


__all__ = [
    "CategoricalDiagnostics",
    "CategoricalSchema",
    "FittedOneHotEncoder",
    "FittedOrdinalEncoder",
    "FittedSimpleImputer",
    "FittedTargetEncoder",
    "ImputationStrategy",
    "OneHotEncoder",
    "OrdinalEncoder",
    "SimpleImputer",
    "TargetEncoder",
    "UnknownPolicy",
]
