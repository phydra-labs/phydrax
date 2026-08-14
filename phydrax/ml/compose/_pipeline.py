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
    _normalize_recipe_specs,
    _predict_values,
    _split_key,
    _transform_batch,
    _transform_values,
    _validate_model_input,
    CompositionDiagnostics,
    CompositionProvenance,
)


class FittedPipeline(AbstractArrayModel):
    """Immutable ordered fitted stages with their complete fit provenance."""

    steps: tuple[tuple[str, AbstractArrayModel], ...]
    provenance: CompositionProvenance
    stage_input_schemas: tuple[FeatureSchema, ...]
    stage_output_schemas: tuple[FeatureSchema, ...]
    feature_schema: FeatureSchema
    final_feature_schema: FeatureSchema
    gradient_contract: GradientContract
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding: ModelBinding = eqx.field(static=True)  # ty: ignore[invalid-attribute-override]

    def __init__(
        self,
        steps: Sequence[tuple[str, AbstractArrayModel]],
        fit_results: Sequence[FitResult],
        stage_input_schemas: Sequence[FeatureSchema],
        stage_output_schemas: Sequence[FeatureSchema],
        /,
        *,
        feature_schema: FeatureSchema,
        final_feature_schema: FeatureSchema,
        gradient_contract: GradientContract,
    ):
        steps_ = tuple(steps)
        results_ = tuple(fit_results)
        input_schemas_ = tuple(stage_input_schemas)
        output_schemas_ = tuple(stage_output_schemas)
        if not steps_:
            raise ValueError("FittedPipeline requires at least one fitted stage.")
        if len(steps_) != len(results_) or len(input_schemas_) != len(steps_):
            raise ValueError("Fitted pipeline stages, schemas, and results must align.")
        if len(output_schemas_) != len(steps_) - 1:
            raise ValueError("Pipeline output schemas describe intermediate stages only.")
        models = tuple(model for _, model in steps_)
        if any(not isinstance(model, AbstractArrayModel) for model in models):
            raise TypeError(
                "Fitted pipeline stages must be AbstractArrayModel instances."
            )
        for model, schema in zip(models, input_schemas_, strict=True):
            _validate_model_input(model, len(schema.names), schema=schema)
        self.steps = steps_
        self.provenance = CompositionProvenance(
            tuple(name for name, _ in steps_), results_
        )
        self.stage_input_schemas = input_schemas_
        self.stage_output_schemas = output_schemas_
        self.feature_schema = feature_schema
        self.final_feature_schema = final_feature_schema
        self.gradient_contract = gradient_contract
        self.in_size = models[0].in_size
        self.out_size = models[-1].out_size
        self._input_binding = _composition_binding(models)

    @property
    def fit_results(self) -> tuple[FitResult, ...]:
        return self.provenance.results

    def __call__(self, x: Any, /, *, key: Any = None):
        keys = _split_key(key, len(self.steps))
        blockwise = self._input_binding.batch_mode == "blockwise"
        values = x
        for index, (_, model) in enumerate(self.steps[:-1]):
            values = _transform_values(
                model,
                values,
                self.stage_input_schemas[index],
                key=keys[index],
                composed_blockwise=blockwise,
            )
        return _predict_values(
            self.steps[-1][1],
            values,
            key=keys[-1],
            composed_blockwise=blockwise,
        )

    def transform_batch(self, batch: MLBatch, /, *, key: Any = None) -> MLBatch:
        if not isinstance(batch, MLBatch):
            raise TypeError("transform_batch requires an MLBatch.")
        keys = _split_key(key, len(self.steps))
        current = batch
        for (_, model), stage_key in zip(self.steps, keys, strict=True):
            current = _transform_batch(model, current, key=stage_key)
        return current


class Pipeline(AbstractRecipe):
    """Pure sequential fitting over one already-selected training batch."""

    steps: tuple[tuple[str, AbstractRecipe], ...]

    def __init__(self, steps: Sequence[tuple[str, AbstractRecipe]], /):
        self.steps = _normalize_recipe_specs(
            steps, kind="Pipeline", recipe_type=AbstractRecipe
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if not isinstance(batch, MLBatch):
            raise TypeError("Pipeline.fit_batch requires an already-selected MLBatch.")
        keys = _split_key(key, 2 * len(self.steps) - 1)
        current = batch
        fitted_steps: list[tuple[str, AbstractArrayModel]] = []
        results: list[FitResult] = []
        input_schemas: list[FeatureSchema] = []
        output_schemas: list[FeatureSchema] = []
        for index, (name, recipe) in enumerate(self.steps):
            input_schemas.append(current.feature_schema)
            result = recipe.fit_batch(current, key=keys[2 * index])
            if not isinstance(result, FitResult):
                raise TypeError(f"Pipeline stage {name!r} did not return a FitResult.")
            model = result.as_trainable()
            fitted_steps.append((name, model))
            results.append(result)
            if index < len(self.steps) - 1:
                current = _transform_batch(model, current, key=keys[2 * index + 1])
                output_schemas.append(current.feature_schema)

        valid, status, contract = _combine_results(results)
        fitted = FittedPipeline(
            fitted_steps,
            results,
            input_schemas,
            output_schemas,
            feature_schema=batch.feature_schema,
            final_feature_schema=current.feature_schema,
            gradient_contract=contract,
        )
        diagnostics = CompositionDiagnostics(
            tuple(name for name, _ in self.steps),
            results,
            valid=valid,
            status=status,
        )
        return FitResult(
            fitted,
            diagnostics,
            valid=valid,
            status=status,
            method="pipeline",
            gradient_contract=contract,
        )


__all__ = ["FittedPipeline", "Pipeline"]
