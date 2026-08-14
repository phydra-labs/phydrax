#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx

from ..._model import AbstractArrayModel, ModelBinding
from .._batch import MLBatch
from .._contracts import AbstractRecipe, FitResult, GradientContract
from .._schema import FeatureSchema
from ._common import (
    _combine_results,
    _composition_binding,
    _join_feature_batches,
    _join_feature_values,
    _normalize_recipe_specs,
    _split_key,
    _transform_batch,
    _transform_values,
    _validate_model_input,
    CompositionDiagnostics,
    CompositionProvenance,
)


class FittedFeatureUnion(AbstractArrayModel):
    """Immutable parallel fitted transforms with ordered, prefixed outputs."""

    transformer_list: tuple[tuple[str, AbstractArrayModel], ...]
    provenance: CompositionProvenance
    branch_output_schemas: tuple[FeatureSchema, ...]
    input_schema: FeatureSchema
    output_schema: FeatureSchema
    gradient_contract: GradientContract
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding: ModelBinding = eqx.field(static=True)  # ty: ignore[invalid-attribute-override]

    def __init__(
        self,
        transformer_list: Sequence[tuple[str, AbstractArrayModel]],
        fit_results: Sequence[FitResult],
        branch_output_schemas: Sequence[FeatureSchema],
        /,
        *,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
        gradient_contract: GradientContract,
    ):
        transformers = tuple(transformer_list)
        results = tuple(fit_results)
        schemas = tuple(branch_output_schemas)
        if not transformers:
            raise ValueError("FittedFeatureUnion requires fitted branches.")
        if len(transformers) != len(results) or len(schemas) != len(transformers):
            raise ValueError("Feature union branches, schemas, and results must align.")
        models = tuple(model for _, model in transformers)
        if any(not isinstance(model, AbstractArrayModel) for model in models):
            raise TypeError(
                "Feature union branches must be AbstractArrayModel instances."
            )
        for model in models:
            _validate_model_input(model, len(input_schema.names), schema=input_schema)
        self.transformer_list = transformers
        self.provenance = CompositionProvenance(
            tuple(name for name, _ in transformers), results
        )
        self.branch_output_schemas = schemas
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.gradient_contract = gradient_contract
        self.in_size = models[0].in_size
        self.out_size = len(output_schema.names)
        self._input_binding = _composition_binding(models)

    @property
    def fit_results(self) -> tuple[FitResult, ...]:
        return self.provenance.results

    def __call__(self, x: Any, /, *, key: Any = None):
        keys = _split_key(key, len(self.transformer_list))
        blockwise = self._input_binding.batch_mode == "blockwise"
        outputs = tuple(
            (
                name,
                _transform_values(
                    model,
                    x,
                    self.input_schema,
                    key=branch_key,
                    composed_blockwise=blockwise,
                ),
            )
            for (name, model), branch_key in zip(self.transformer_list, keys, strict=True)
        )
        return _join_feature_values(outputs)

    def transform_batch(self, batch: MLBatch, /, *, key: Any = None) -> MLBatch:
        if not isinstance(batch, MLBatch):
            raise TypeError("transform_batch requires an MLBatch.")
        keys = _split_key(key, len(self.transformer_list))
        outputs = tuple(
            (
                name,
                _transform_batch(model, batch, key=branch_key),
            )
            for (name, model), branch_key in zip(self.transformer_list, keys, strict=True)
        )
        return _join_feature_batches(batch, outputs)


class FeatureUnion(AbstractRecipe):
    """Fit independent native transforms on the same selected batch and join them."""

    transformer_list: tuple[tuple[str, AbstractRecipe], ...]

    def __init__(self, transformer_list: Sequence[tuple[str, AbstractRecipe]], /):
        self.transformer_list = _normalize_recipe_specs(
            transformer_list, kind="FeatureUnion", recipe_type=AbstractRecipe
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if not isinstance(batch, MLBatch):
            raise TypeError(
                "FeatureUnion.fit_batch requires an already-selected MLBatch."
            )
        keys = _split_key(key, 2 * len(self.transformer_list))
        fitted: list[tuple[str, AbstractArrayModel]] = []
        results: list[FitResult] = []
        outputs: list[tuple[str, MLBatch]] = []
        branch_schemas: list[FeatureSchema] = []
        for index, (name, recipe) in enumerate(self.transformer_list):
            result = recipe.fit_batch(batch, key=keys[2 * index])
            if not isinstance(result, FitResult):
                raise TypeError(
                    f"Feature union branch {name!r} did not return FitResult."
                )
            model = result.as_trainable()
            transformed = _transform_batch(model, batch, key=keys[2 * index + 1])
            fitted.append((name, model))
            results.append(result)
            outputs.append((name, transformed))
            branch_schemas.append(transformed.feature_schema)

        joined = _join_feature_batches(batch, outputs)
        valid, status, contract = _combine_results(results)
        model = FittedFeatureUnion(
            fitted,
            results,
            branch_schemas,
            input_schema=batch.feature_schema,
            output_schema=joined.feature_schema,
            gradient_contract=contract,
        )
        diagnostics = CompositionDiagnostics(
            tuple(name for name, _ in self.transformer_list),
            results,
            valid=valid,
            status=status,
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="feature_union",
            gradient_contract=contract,
        )


__all__ = ["FeatureUnion", "FittedFeatureUnion"]
