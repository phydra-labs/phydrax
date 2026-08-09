#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp

from ..._model import AbstractArrayModel, ModelBinding
from .._batch import MLBatch
from .._contracts import AbstractRecipe, FitResult, GradientContract
from .._schema import FeatureSchema
from .._sparse_features import FeatureArray, SparseFeatures
from ._common import (
    _combine_results,
    _composition_binding,
    _join_feature_batches,
    _join_feature_values,
    _split_key,
    _transform_batch,
    _transform_values,
    _validate_model_input,
    CompositionDiagnostics,
    CompositionProvenance,
)


ColumnSelector: TypeAlias = int | str | slice | Sequence[int | str]
ResolvedColumnTransformer: TypeAlias = tuple[str, AbstractArrayModel, tuple[int, ...]]


def _normalize_transformers(
    transformers: Sequence[tuple[str, AbstractRecipe, ColumnSelector]],
    /,
) -> tuple[tuple[str, AbstractRecipe, ColumnSelector], ...]:
    entries = tuple(transformers)
    if not entries:
        raise ValueError("ColumnTransformer requires at least one named transform.")
    normalized: list[tuple[str, AbstractRecipe, ColumnSelector]] = []
    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise TypeError(
                "Each ColumnTransformer entry must be a (name, recipe, columns) tuple."
            )
        name, recipe, selector = entry
        name_ = str(name)
        if not name_:
            raise ValueError("ColumnTransformer names must be non-empty.")
        if not isinstance(recipe, AbstractRecipe):
            raise TypeError(
                f"ColumnTransformer entry {name_!r} is not an AbstractRecipe."
            )
        if isinstance(selector, list):
            selector = tuple(selector)
        normalized.append((name_, recipe, selector))
        names.append(name_)
    if len(set(names)) != len(names):
        raise ValueError("ColumnTransformer names must be unique.")
    if "remainder" in names:
        raise ValueError("'remainder' is reserved for passthrough columns.")
    return tuple(normalized)


def _resolve_columns(
    selector: ColumnSelector,
    schema: FeatureSchema,
    /,
) -> tuple[int, ...]:
    count = len(schema.names)
    if isinstance(selector, bool):
        raise TypeError("Boolean values are not column selectors.")
    if isinstance(selector, int):
        items: tuple[int | str, ...] = (selector,)
    elif isinstance(selector, str):
        items = (selector,)
    elif isinstance(selector, slice):
        indices = tuple(range(*selector.indices(count)))
        if not indices:
            raise ValueError("Column selector resolves to no features.")
        return indices
    elif isinstance(selector, Sequence):
        items = tuple(selector)
    else:
        raise TypeError(
            "Columns must be indices, names, a slice, or a sequence of indices/names."
        )
    if not items:
        raise ValueError("Column selector resolves to no features.")
    resolved: list[int] = []
    for item in items:
        if isinstance(item, bool):
            raise TypeError("Boolean values are not column selectors.")
        if isinstance(item, int):
            index = int(item)
            if index < 0:
                index += count
            if index < 0 or index >= count:
                raise IndexError(f"Column index {item} is out of range.")
        elif isinstance(item, str):
            if item not in schema.names:
                raise KeyError(f"Unknown feature name {item!r}.")
            index = schema.names.index(item)
        else:
            raise TypeError("Column selector entries must be integer indices or names.")
        resolved.append(index)
    if len(set(resolved)) != len(resolved):
        raise ValueError("A single column transform cannot select a feature twice.")
    return tuple(resolved)


def _select_sparse_columns(
    features: SparseFeatures,
    indices: tuple[int, ...],
    /,
) -> SparseFeatures:
    if not indices:
        raise ValueError("Sparse column selection requires at least one feature.")
    lookup = jnp.zeros((features.feature_count,), dtype=jnp.int32)
    selected = jnp.zeros((features.feature_count,), dtype=bool)
    source = jnp.asarray(indices, dtype=jnp.int32)
    lookup = lookup.at[source].set(jnp.arange(len(indices), dtype=jnp.int32))
    selected = selected.at[source].set(True)
    old_indices = features.columns.source_indices
    valid = features.columns.valid & selected[old_indices]
    return SparseFeatures(
        features.values,
        lookup[old_indices],
        feature_count=len(indices),
        valid=valid,
        case_shape=features.case_shape,
    )


def _select_values(
    values: FeatureArray,
    indices: tuple[int, ...],
    /,
) -> FeatureArray:
    if isinstance(values, SparseFeatures):
        return _select_sparse_columns(values, indices)
    return jnp.take(jnp.asarray(values), jnp.asarray(indices, dtype=jnp.int32), axis=-1)


def _select_batch(batch: MLBatch, indices: tuple[int, ...], /) -> MLBatch:
    features = _select_values(batch.features, indices)
    return batch.with_features(
        features,
        feature_schema=batch.feature_schema.select(indices),
        feature_mask=(
            None
            if isinstance(features, SparseFeatures)
            else jnp.take(
                batch.feature_mask,
                jnp.asarray(indices, dtype=jnp.int32),
                axis=-1,
            )
        ),
    )


class FittedColumnTransformer(AbstractArrayModel):
    """Schema-resolved immutable fitted column branches."""

    transformers: tuple[ResolvedColumnTransformer, ...]
    provenance: CompositionProvenance
    branch_input_schemas: tuple[FeatureSchema, ...]
    branch_output_schemas: tuple[FeatureSchema, ...]
    remainder_indices: tuple[int, ...] = eqx.field(static=True)
    input_schema: FeatureSchema
    output_schema: FeatureSchema
    gradient_contract: GradientContract
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding: ModelBinding = eqx.field(static=True)

    def __init__(
        self,
        transformers: Sequence[ResolvedColumnTransformer],
        fit_results: Sequence[FitResult],
        branch_input_schemas: Sequence[FeatureSchema],
        branch_output_schemas: Sequence[FeatureSchema],
        /,
        *,
        remainder_indices: Sequence[int],
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
        gradient_contract: GradientContract,
    ):
        transformers_ = tuple(transformers)
        results = tuple(fit_results)
        input_schemas = tuple(branch_input_schemas)
        output_schemas = tuple(branch_output_schemas)
        if not transformers_:
            raise ValueError("FittedColumnTransformer requires fitted branches.")
        if not (
            len(transformers_)
            == len(results)
            == len(input_schemas)
            == len(output_schemas)
        ):
            raise ValueError("Column branches, schemas, and fit results must align.")
        models = tuple(model for _, model, _ in transformers_)
        if any(not isinstance(model, AbstractArrayModel) for model in models):
            raise TypeError("Column branches must be AbstractArrayModel instances.")
        for model, schema in zip(models, input_schemas, strict=True):
            _validate_model_input(model, len(schema.names), schema=schema)
        self.transformers = transformers_
        self.provenance = CompositionProvenance(
            tuple(name for name, _, _ in transformers_), results
        )
        self.branch_input_schemas = input_schemas
        self.branch_output_schemas = output_schemas
        self.remainder_indices = tuple(int(index) for index in remainder_indices)
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.gradient_contract = gradient_contract
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self._input_binding = _composition_binding(models)

    @property
    def fit_results(self) -> tuple[FitResult, ...]:
        return self.provenance.results

    def __call__(self, x: Any, /, *, key: Any = None):
        keys = _split_key(key, len(self.transformers))
        blockwise = self._input_binding.batch_mode == "blockwise"
        outputs: list[tuple[str, FeatureArray]] = []
        for index, ((name, model, columns), branch_key) in enumerate(
            zip(self.transformers, keys, strict=True)
        ):
            selected = _select_values(x, columns)
            outputs.append(
                (
                    name,
                    _transform_values(
                        model,
                        selected,
                        self.branch_input_schemas[index],
                        key=branch_key,
                        composed_blockwise=blockwise,
                    ),
                )
            )
        if self.remainder_indices:
            outputs.append(("remainder", _select_values(x, self.remainder_indices)))
        return _join_feature_values(outputs)

    def transform_batch(self, batch: MLBatch, /, *, key: Any = None) -> MLBatch:
        if not isinstance(batch, MLBatch):
            raise TypeError("transform_batch requires an MLBatch.")
        keys = _split_key(key, len(self.transformers))
        outputs: list[tuple[str, MLBatch]] = []
        for (name, model, columns), branch_key in zip(
            self.transformers, keys, strict=True
        ):
            selected = _select_batch(batch, columns)
            outputs.append((name, _transform_batch(model, selected, key=branch_key)))
        if self.remainder_indices:
            outputs.append(("remainder", _select_batch(batch, self.remainder_indices)))
        return _join_feature_batches(batch, outputs)


class ColumnTransformer(AbstractRecipe):
    """Resolve named columns once, fit only on the selected batch, and join outputs."""

    transformers: tuple[tuple[str, AbstractRecipe, ColumnSelector], ...]
    remainder: Literal["drop", "passthrough"] = eqx.field(static=True)

    def __init__(
        self,
        transformers: Sequence[tuple[str, AbstractRecipe, ColumnSelector]],
        /,
        *,
        remainder: Literal["drop", "passthrough"] = "drop",
    ):
        if remainder not in ("drop", "passthrough"):
            raise ValueError("remainder must be 'drop' or 'passthrough'.")
        self.transformers = _normalize_transformers(transformers)
        self.remainder = remainder

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if not isinstance(batch, MLBatch):
            raise TypeError(
                "ColumnTransformer.fit_batch requires an already-selected MLBatch."
            )
        resolved = tuple(
            (name, recipe, _resolve_columns(selector, batch.feature_schema))
            for name, recipe, selector in self.transformers
        )
        keys = _split_key(key, 2 * len(resolved))
        fitted: list[ResolvedColumnTransformer] = []
        results: list[FitResult] = []
        input_schemas: list[FeatureSchema] = []
        output_schemas: list[FeatureSchema] = []
        outputs: list[tuple[str, MLBatch]] = []
        used: set[int] = set()
        for index, (name, recipe, columns) in enumerate(resolved):
            selected = _select_batch(batch, columns)
            result = recipe.fit_batch(selected, key=keys[2 * index])
            if not isinstance(result, FitResult):
                raise TypeError(f"Column branch {name!r} did not return a FitResult.")
            model = result.as_trainable()
            transformed = _transform_batch(model, selected, key=keys[2 * index + 1])
            fitted.append((name, model, columns))
            results.append(result)
            input_schemas.append(selected.feature_schema)
            output_schemas.append(transformed.feature_schema)
            outputs.append((name, transformed))
            used.update(columns)

        remainder = tuple(
            index for index in range(batch.feature_count) if index not in used
        )
        remainder_indices = remainder if self.remainder == "passthrough" else ()
        if remainder_indices:
            outputs.append(("remainder", _select_batch(batch, remainder_indices)))
        joined = _join_feature_batches(batch, outputs)
        valid, status, contract = _combine_results(results)
        model = FittedColumnTransformer(
            fitted,
            results,
            input_schemas,
            output_schemas,
            remainder_indices=remainder_indices,
            input_schema=batch.feature_schema,
            output_schema=joined.feature_schema,
            gradient_contract=contract,
        )
        diagnostics = CompositionDiagnostics(
            tuple(name for name, _, _ in self.transformers),
            results,
            valid=valid,
            status=status,
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="column_transformer",
            gradient_contract=contract,
        )


__all__ = [
    "ColumnSelector",
    "ColumnTransformer",
    "FittedColumnTransformer",
]
