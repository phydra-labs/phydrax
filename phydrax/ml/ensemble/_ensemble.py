#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from ..._model import AbstractArrayModel, ModelBinding
from ..._strict import StrictModule
from ...uq import HeterogeneousFunctionEnsemble, HomogeneousFunctionEnsemble
from .._batch import MLBatch
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_SUCCESS,
)
from .._schema import FeatureSchema, TargetSchema
from .._sparse_features import SparseFeatures


def _require_key(key: Any, owner: str) -> Array:
    if key is None:
        raise ValueError(f"{owner} requires an explicit JAX key.")
    return jnp.asarray(key)


def _key(key: Array, stream: int) -> Array:
    return jr.fold_in(key, int(stream))


def _call_members(members: Sequence[AbstractArrayModel], x: Any, key: Any) -> Array:
    count = len(members)
    keys = (None,) * count if key is None else tuple(jr.split(key, count))
    return jnp.stack(
        tuple(
            member(x, key=member_key)
            for member, member_key in zip(members, keys, strict=True)
        ),
        axis=0,
    )


def _homogeneous_predictions(
    ensemble: HomogeneousFunctionEnsemble, x: Any, key: Any
) -> Array:
    if key is None:
        return eqx.filter_vmap(
            lambda member: member(x, key=None), in_axes=eqx.if_array(0)
        )(ensemble.model)
    keys = jr.split(key, ensemble.num_members)
    return eqx.filter_vmap(
        lambda member, member_key: member(x, key=member_key),
        in_axes=(eqx.if_array(0), 0),
    )(ensemble.model, keys)


def _normalised_weights(weights: Array, count: int) -> Array:
    value = jnp.asarray(weights)
    if value.shape != (count,):
        raise ValueError(f"member_weights must have shape ({count},).")
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise TypeError("member_weights must be real-valued.")
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype(float)
    value = eqx.error_if(
        value,
        jnp.any(~jnp.isfinite(value)) | jnp.any(value < 0.0) | (jnp.sum(value) <= 0.0),
        "member_weights must be finite, nonnegative, and have positive sum.",
    )
    return value / jnp.sum(value)


def _weighted_mean(predictions: Array, weights: Array) -> Array:
    shape = (weights.shape[0],) + (1,) * (predictions.ndim - 1)
    return jnp.sum(predictions * weights.reshape(shape), axis=0)


def _model_sizes(members: Sequence[AbstractArrayModel]) -> tuple[Any, Any]:
    first = members[0]
    for member in members[1:]:
        if member.in_size != first.in_size or member.out_size != first.out_size:
            raise ValueError(
                "Ensemble members must have identical input and output sizes."
            )
    return first.in_size, first.out_size


class EnsembleFitDiagnostics(StrictModule):
    member_valid: Array
    member_status: Array
    auxiliary_status: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        member_valid: Any,
        member_status: Any,
        /,
        *,
        method: str,
        auxiliary_status: Any = (),
    ):
        self.member_valid = jnp.asarray(member_valid, dtype=bool)
        self.member_status = jnp.asarray(member_status, dtype=jnp.int32)
        self.auxiliary_status = jnp.asarray(auxiliary_status, dtype=jnp.int32)
        self.method = str(method)


class HomogeneousEnsembleModel(AbstractArrayModel):
    """Differentiable mean of a member-axis-stacked UQ ensemble."""

    ensemble: HomogeneousFunctionEnsemble
    member_weights: Array
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(
        self,
        members: Sequence[AbstractArrayModel],
        /,
        *,
        member_weights: Any = None,
    ):
        values = tuple(members)
        if not values:
            raise ValueError("members must be non-empty.")
        in_size, out_size = _model_sizes(values)
        self.ensemble = HomogeneousFunctionEnsemble.from_members(values)
        weights = jnp.ones((len(values),)) if member_weights is None else member_weights
        self.member_weights = _normalised_weights(jnp.asarray(weights), len(values))
        self.in_size = in_size
        self.out_size = out_size

    def member_predictions(self, x: Any, /, *, key: Any = None) -> Array:
        return _homogeneous_predictions(self.ensemble, x, key)

    def predictive(self, x: Any, /, *, key: Any):
        """Return the shared UQ PredictiveField over raw member samples."""
        return self.ensemble.predict(x, key=_require_key(key, "predictive"))

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        return _weighted_mean(self.member_predictions(x, key=key), self.member_weights)


class HeterogeneousEnsembleModel(AbstractArrayModel):
    """Differentiable mean of an explicitly heterogeneous UQ ensemble."""

    ensemble: HeterogeneousFunctionEnsemble
    member_weights: Array
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(
        self,
        members: Sequence[AbstractArrayModel],
        /,
        *,
        member_weights: Any = None,
    ):
        values = tuple(members)
        if not values:
            raise ValueError("members must be non-empty.")
        in_size, out_size = _model_sizes(values)
        self.ensemble = HeterogeneousFunctionEnsemble(values)
        weights = jnp.ones((len(values),)) if member_weights is None else member_weights
        self.member_weights = _normalised_weights(jnp.asarray(weights), len(values))
        self.in_size = in_size
        self.out_size = out_size

    def member_predictions(self, x: Any, /, *, key: Any = None) -> Array:
        return _call_members(self.ensemble.members, x, key)

    def predictive(self, x: Any, /, *, key: Any):
        """Return the shared UQ PredictiveField over raw member samples."""
        return self.ensemble.predict(x, key=_require_key(key, "predictive"))

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        return _weighted_mean(self.member_predictions(x, key=key), self.member_weights)


class SoftVotingModel(AbstractArrayModel):
    """Smooth weighted voting over aligned scores or probabilities."""

    ensemble: HeterogeneousFunctionEnsemble
    member_weights: Array
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(
        self, members: Sequence[AbstractArrayModel], /, *, member_weights: Any = None
    ):
        values = tuple(members)
        if not values:
            raise ValueError("members must be non-empty.")
        self.in_size, self.out_size = _model_sizes(values)
        self.ensemble = HeterogeneousFunctionEnsemble(values)
        raw = jnp.ones((len(values),)) if member_weights is None else member_weights
        self.member_weights = _normalised_weights(jnp.asarray(raw), len(values))

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        return _weighted_mean(
            _call_members(self.ensemble.members, x, key), self.member_weights
        )


class HardVotingModel(AbstractArrayModel):
    """Exact elementwise majority vote; its output is nondifferentiable."""

    ensemble: HeterogeneousFunctionEnsemble
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(self, members: Sequence[AbstractArrayModel], /):
        values = tuple(members)
        if not values:
            raise ValueError("members must be non-empty.")
        self.in_size, self.out_size = _model_sizes(values)
        self.ensemble = HeterogeneousFunctionEnsemble(values)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        predictions = _call_members(self.ensemble.members, x, key)
        counts = jnp.sum(predictions[:, None] == predictions[None, :], axis=1)
        winners = jnp.argmax(counts, axis=0)
        voted = jnp.take_along_axis(predictions, winners[None, ...], axis=0)[0]
        return jax.lax.stop_gradient(voted)


class FeatureSubsetModel(AbstractArrayModel):
    model: AbstractArrayModel
    indices: Array
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(self, model: AbstractArrayModel, indices: Any, /, *, input_size: int):
        selected = jnp.asarray(indices, dtype=jnp.int32)
        if selected.ndim != 1 or selected.shape[0] == 0:
            raise ValueError("indices must be a nonempty vector.")
        self.model = model
        self.indices = selected
        self.in_size = int(input_size)
        self.out_size = model.out_size

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        return self.model(jnp.take(jnp.asarray(x), self.indices, axis=-1), key=key)


class StackingModel(AbstractArrayModel):
    """Leakage-safe stacking predictor fitted from out-of-fold meta-features."""

    base_ensemble: HeterogeneousFunctionEnsemble
    meta_model: AbstractArrayModel
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(
        self, bases: Sequence[AbstractArrayModel], meta_model: AbstractArrayModel, /
    ):
        values = tuple(bases)
        if not values:
            raise ValueError("bases must be non-empty.")
        first_in = values[0].in_size
        if any(model.in_size != first_in for model in values[1:]):
            raise ValueError("Stacking base models must share an input size.")
        self.base_ensemble = HeterogeneousFunctionEnsemble(values)
        self.meta_model = meta_model
        self.in_size = first_in
        self.out_size = meta_model.out_size

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        count = self.base_ensemble.num_members
        keys = (None,) * (count + 1) if key is None else tuple(jr.split(key, count + 1))
        features = tuple(
            _flatten_prediction(model(x, key=keys[index]), x)
            for index, model in enumerate(self.base_ensemble.members)
        )
        return self.meta_model(jnp.concatenate(features, axis=-1), key=keys[-1])


class MixtureOfExpertsModel(AbstractArrayModel):
    """Smooth mixture with a learned softmax gating model."""

    experts: HeterogeneousFunctionEnsemble
    gate: AbstractArrayModel
    temperature: Array
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    output_rank: int = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(
        self,
        experts: Sequence[AbstractArrayModel],
        gate: AbstractArrayModel,
        /,
        *,
        temperature: Any = 1.0,
    ):
        values = tuple(experts)
        if not values:
            raise ValueError("experts must be non-empty.")
        self.in_size, self.out_size = _model_sizes(values)
        if gate.in_size != self.in_size:
            raise ValueError("The gating model and experts must share an input size.")
        self.experts = HeterogeneousFunctionEnsemble(values)
        self.gate = gate
        self.temperature = eqx.error_if(
            jnp.asarray(temperature),
            jnp.any(jnp.asarray(temperature) <= 0.0),
            "temperature must be positive.",
        )
        self.output_rank = (
            0
            if self.out_size == "scalar"
            else (1 if isinstance(self.out_size, int) else len(self.out_size))
        )

    def gating_weights(self, x: Any, /, *, key: Any = None) -> Array:
        logits = jnp.asarray(self.gate(x, key=key))
        if logits.shape[-1] != self.experts.num_members:
            raise ValueError("The gating model output must have one score per expert.")
        return jax.nn.softmax(logits / self.temperature, axis=-1)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        count = self.experts.num_members
        keys = (None,) * (count + 1) if key is None else tuple(jr.split(key, count + 1))
        predictions = _call_members(
            self.experts.members, x, None if key is None else jnp.asarray(keys[0])
        )
        weights = self.gating_weights(x, key=keys[-1])
        member_weights = jnp.moveaxis(weights, -1, 0)
        member_weights = member_weights.reshape(
            member_weights.shape + (1,) * self.output_rank
        )
        return jnp.sum(predictions * member_weights, axis=0)


def _flatten_prediction(prediction: Any, features: Any) -> Array:
    value = jnp.asarray(prediction)
    x = jnp.asarray(features)
    leading = x.shape[:-1]
    if value.shape[: len(leading)] != leading:
        raise ValueError("Model prediction does not preserve the input leading axes.")
    return (
        value.reshape(leading + (-1,)) if value.ndim > len(leading) else value[..., None]
    )


def _fit_result(
    model: AbstractArrayModel,
    diagnostics: EnsembleFitDiagnostics,
    contract: GradientContract,
    method: str,
) -> FitResult:
    valid = jnp.all(diagnostics.member_valid, axis=0)
    status = jnp.where(valid, ML_SUCCESS, jnp.max(diagnostics.member_status, axis=0))
    return FitResult(
        model,
        diagnostics,
        valid=valid,
        status=status,
        method=method,
        gradient_contract=contract,
    )


def _fit_members(
    recipes: Sequence[AbstractRecipe], batch: MLBatch, key: Array, stream: int
):
    results = tuple(
        recipe.fit_batch(batch, key=_key(key, stream + index))
        for index, recipe in enumerate(recipes)
    )
    models = tuple(result.as_trainable() for result in results)
    valid = jnp.stack(tuple(result.valid for result in results))
    status = jnp.stack(tuple(result.status for result in results))
    return results, models, valid, status


class BaggingRecipe(AbstractRecipe):
    recipe: AbstractRecipe
    num_members: int = eqx.field(static=True)
    sample_fraction: float = eqx.field(static=True)

    def __init__(
        self,
        recipe: AbstractRecipe,
        /,
        *,
        num_members: int = 16,
        sample_fraction: float = 1.0,
    ):
        if not isinstance(recipe, AbstractRecipe):
            raise TypeError("recipe must be an AbstractRecipe.")
        if int(num_members) <= 0 or not (0.0 < float(sample_fraction) <= 1.0):
            raise ValueError(
                "num_members must be positive and sample_fraction in (0, 1]."
            )
        self.recipe = recipe
        self.num_members = int(num_members)
        self.sample_fraction = float(sample_fraction)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        root = _require_key(key, "BaggingRecipe")
        size = max(1, int(self.sample_fraction * batch.sample_count))
        results = []
        for index in range(self.num_members):
            member_key = _key(root, index)
            indices = jr.randint(_key(member_key, 1), (size,), 0, batch.sample_count)
            results.append(
                self.recipe.fit_batch(
                    batch.take_samples(indices), key=_key(member_key, 2)
                )
            )
        models = tuple(result.as_trainable() for result in results)
        valid = jnp.stack(tuple(result.valid for result in results))
        status = jnp.stack(tuple(result.status for result in results))
        diagnostics = EnsembleFitDiagnostics(valid, status, method="bootstrap-bagging")
        return _fit_result(
            HomogeneousEnsembleModel(models),
            diagnostics,
            GradientContract(
                fit_features="none",
                fit_targets="none",
                fit_weights="none",
                fit_mode="stopped",
                nondifferentiable_outputs=("bootstrap_indices",),
                conditions=(
                    "Predictions are smooth when every child prediction is smooth.",
                ),
            ),
            "bagging",
        )


class RandomSubspaceRecipe(AbstractRecipe):
    recipe: AbstractRecipe
    num_members: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)

    def __init__(
        self, recipe: AbstractRecipe, /, *, num_members: int = 16, feature_count: int
    ):
        if not isinstance(recipe, AbstractRecipe):
            raise TypeError("recipe must be an AbstractRecipe.")
        if int(num_members) <= 0 or int(feature_count) <= 0:
            raise ValueError("num_members and feature_count must be positive.")
        self.recipe = recipe
        self.num_members = int(num_members)
        self.feature_count = int(feature_count)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        root = _require_key(key, "RandomSubspaceRecipe")
        if self.feature_count > batch.feature_count:
            raise ValueError("feature_count cannot exceed the batch feature count.")
        dense = batch.dense_features()
        results = []
        models = []
        for index in range(self.num_members):
            member_key = _key(root, index)
            indices = jr.choice(
                _key(member_key, 1),
                batch.feature_count,
                (self.feature_count,),
                replace=False,
            )
            selected_batch = batch.with_features(
                jnp.take(dense, indices, axis=-1),
                feature_schema=FeatureSchema.anonymous(self.feature_count),
                feature_mask=jnp.take(batch.feature_mask, indices, axis=-1),
            )
            result = self.recipe.fit_batch(selected_batch, key=_key(member_key, 2))
            results.append(result)
            models.append(
                FeatureSubsetModel(
                    result.as_trainable(), indices, input_size=batch.feature_count
                )
            )
        valid = jnp.stack(tuple(result.valid for result in results))
        status = jnp.stack(tuple(result.status for result in results))
        diagnostics = EnsembleFitDiagnostics(valid, status, method="random-subspace")
        return _fit_result(
            HeterogeneousEnsembleModel(tuple(models)),
            diagnostics,
            GradientContract(
                fit_mode="stopped",
                nondifferentiable_outputs=("feature_subspaces",),
                conditions=("Predictions are smooth conditional on sampled subspaces.",),
            ),
            "random-subspace",
        )


class SoftVotingRecipe(AbstractRecipe):
    recipes: tuple[AbstractRecipe, ...]
    member_weights: Array

    def __init__(
        self, recipes: Sequence[AbstractRecipe], /, *, member_weights: Any = None
    ):
        values = tuple(recipes)
        if not values or any(not isinstance(value, AbstractRecipe) for value in values):
            raise TypeError(
                "recipes must be a nonempty sequence of AbstractRecipe values."
            )
        self.recipes = values
        raw = jnp.ones((len(values),)) if member_weights is None else member_weights
        self.member_weights = _normalised_weights(jnp.asarray(raw), len(values))

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        root = _require_key(key, "SoftVotingRecipe")
        _, models, valid, status = _fit_members(self.recipes, batch, root, 10)
        diagnostics = EnsembleFitDiagnostics(valid, status, method="soft-voting")
        return _fit_result(
            SoftVotingModel(models, member_weights=self.member_weights),
            diagnostics,
            GradientContract(conditions=("All member scores must be aligned.",)),
            "soft-voting",
        )


class HardVotingRecipe(AbstractRecipe):
    recipes: tuple[AbstractRecipe, ...]

    def __init__(self, recipes: Sequence[AbstractRecipe], /):
        values = tuple(recipes)
        if not values or any(not isinstance(value, AbstractRecipe) for value in values):
            raise TypeError(
                "recipes must be a nonempty sequence of AbstractRecipe values."
            )
        self.recipes = values

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        root = _require_key(key, "HardVotingRecipe")
        _, models, valid, status = _fit_members(self.recipes, batch, root, 20)
        diagnostics = EnsembleFitDiagnostics(valid, status, method="hard-voting")
        return _fit_result(
            HardVotingModel(models),
            diagnostics,
            GradientContract(
                prediction_inputs="none",
                prediction_parameters="none",
                fit_mode="stopped",
                nondifferentiable_outputs=("majority_vote",),
            ),
            "hard-voting",
        )


class StackingRecipe(AbstractRecipe):
    base_recipes: tuple[AbstractRecipe, ...]
    meta_recipe: AbstractRecipe
    num_folds: int = eqx.field(static=True)

    def __init__(
        self,
        base_recipes: Sequence[AbstractRecipe],
        meta_recipe: AbstractRecipe,
        /,
        *,
        num_folds: int = 5,
    ):
        bases = tuple(base_recipes)
        if not bases or any(not isinstance(value, AbstractRecipe) for value in bases):
            raise TypeError("base_recipes must be a nonempty sequence of recipes.")
        if not isinstance(meta_recipe, AbstractRecipe):
            raise TypeError("meta_recipe must be an AbstractRecipe.")
        if int(num_folds) < 2:
            raise ValueError("num_folds must be at least two.")
        self.base_recipes = bases
        self.meta_recipe = meta_recipe
        self.num_folds = int(num_folds)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        root = _require_key(key, "StackingRecipe")
        if self.num_folds > batch.sample_count:
            raise ValueError("num_folds cannot exceed the sample count.")
        permutation = jr.permutation(_key(root, 1), batch.sample_count)
        bounds = tuple(
            (fold * batch.sample_count) // self.num_folds
            for fold in range(self.num_folds + 1)
        )
        oof_blocks: list[Array] = []
        fold_valid: list[Array] = []
        fold_status: list[Array] = []
        dense = batch.dense_features()
        sample_axis = len(batch.case_shape)
        for base_index, recipe in enumerate(self.base_recipes):
            oof = None
            for fold in range(self.num_folds):
                start, stop = bounds[fold], bounds[fold + 1]
                validation_indices = permutation[start:stop]
                training_indices = jnp.concatenate(
                    (permutation[:start], permutation[stop:])
                )
                result = recipe.fit_batch(
                    batch.take_samples(training_indices),
                    key=_key(root, 1000 + base_index * self.num_folds + fold),
                )
                prediction = result.model(
                    jnp.take(dense, validation_indices, axis=sample_axis),
                    key=_key(root, 2000 + base_index * self.num_folds + fold),
                )
                flat = _flatten_prediction(
                    prediction, jnp.take(dense, validation_indices, axis=sample_axis)
                )
                if oof is None:
                    oof = jnp.zeros(
                        batch.case_shape + (batch.sample_count, flat.shape[-1]),
                        dtype=flat.dtype,
                    )
                index = (slice(None),) * len(batch.case_shape) + (
                    validation_indices,
                    slice(None),
                )
                oof = oof.at[index].set(flat)
                fold_valid.append(result.valid)
                fold_status.append(result.status)
            if oof is None:
                raise RuntimeError("Stacking produced no out-of-fold predictions.")
            oof_blocks.append(oof)
        meta_features = jnp.concatenate(tuple(oof_blocks), axis=-1)
        meta_batch = batch.with_features(
            meta_features,
            feature_schema=FeatureSchema.anonymous(meta_features.shape[-1]),
            feature_mask=jnp.isfinite(jnp.real(meta_features))
            & jnp.isfinite(jnp.imag(meta_features)),
        )
        meta_result = self.meta_recipe.fit_batch(meta_batch, key=_key(root, 3000))
        final_results = tuple(
            recipe.fit_batch(batch, key=_key(root, 4000 + index))
            for index, recipe in enumerate(self.base_recipes)
        )
        fold_valid_array = jnp.stack(tuple(fold_valid))
        fold_status_array = jnp.stack(tuple(fold_status))
        final_valid = jnp.stack(tuple(result.valid for result in final_results))
        final_status = jnp.stack(tuple(result.status for result in final_results))
        all_valid = jnp.concatenate(
            (fold_valid_array, final_valid, meta_result.valid[None, ...]), axis=0
        )
        all_status = jnp.concatenate(
            (fold_status_array, final_status, meta_result.status[None, ...]), axis=0
        )
        diagnostics = EnsembleFitDiagnostics(
            all_valid,
            all_status,
            auxiliary_status=fold_status_array.reshape(
                (len(self.base_recipes), self.num_folds) + fold_status_array.shape[1:]
            ),
            method="out-of-fold-stacking",
        )
        model = StackingModel(
            tuple(result.as_trainable() for result in final_results),
            meta_result.as_trainable(),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                fit_mode="stopped",
                nondifferentiable_outputs=("fold_assignment",),
                conditions=(
                    "Meta-model training uses predictions from models excluding each held-out fold.",
                ),
            ),
            "stacking",
        )


class MixtureOfExpertsRecipe(AbstractRecipe):
    expert_recipes: tuple[AbstractRecipe, ...]
    gate_recipe: AbstractRecipe
    temperature: float = eqx.field(static=True)

    def __init__(
        self,
        expert_recipes: Sequence[AbstractRecipe],
        gate_recipe: AbstractRecipe,
        /,
        *,
        temperature: float = 1.0,
    ):
        experts = tuple(expert_recipes)
        if not experts or any(not isinstance(value, AbstractRecipe) for value in experts):
            raise TypeError("expert_recipes must be a nonempty sequence of recipes.")
        if not isinstance(gate_recipe, AbstractRecipe):
            raise TypeError("gate_recipe must be an AbstractRecipe.")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive.")
        self.expert_recipes = experts
        self.gate_recipe = gate_recipe
        self.temperature = float(temperature)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        root = _require_key(key, "MixtureOfExpertsRecipe")
        targets = batch.require_targets()
        expert_results = tuple(
            recipe.fit_batch(batch, key=_key(root, 5000 + index))
            for index, recipe in enumerate(self.expert_recipes)
        )
        predictions = tuple(
            _flatten_prediction(
                result.model(batch.dense_features(), key=_key(root, 6000 + index)),
                batch.dense_features(),
            )
            for index, result in enumerate(expert_results)
        )
        target_flat = targets.reshape(targets.shape[: len(batch.case_shape) + 1] + (-1,))
        if batch.target_mask is None:
            raise ValueError("Mixture-of-experts gating requires a target mask.")
        target_mask = batch.target_mask.reshape(target_flat.shape)
        target_flat = jnp.where(target_mask, target_flat, 0)
        output_count = jnp.sum(target_mask, axis=-1)

        def expert_loss(prediction: Array) -> Array:
            if prediction.shape != target_flat.shape:
                raise ValueError(
                    "Every expert prediction must match the flattened target shape."
                )
            difference = prediction - target_flat
            squared = jnp.real(difference * jnp.conj(difference))
            return jnp.sum(jnp.where(target_mask, squared, 0.0), axis=-1) / jnp.maximum(
                output_count, 1
            )

        losses = jnp.stack(
            tuple(expert_loss(prediction) for prediction in predictions), axis=-1
        )
        soft_assignments = jax.nn.softmax(-losses / self.temperature, axis=-1)
        gate_sample_mask = batch.sample_mask & (output_count > 0)
        gate_batch = MLBatch(
            batch.features,
            soft_assignments,
            feature_mask=None
            if isinstance(batch.features, SparseFeatures)
            else batch.feature_mask,
            target_mask=jnp.broadcast_to(
                gate_sample_mask[..., None], soft_assignments.shape
            ),
            sample_mask=gate_sample_mask,
            sample_weight=batch.sample_weight,
            measure_weight=batch.measure_weight,
            groups=batch.groups,
            feature_schema=batch.feature_schema,
            target_schema=TargetSchema("continuous"),
        )
        gate_result = self.gate_recipe.fit_batch(gate_batch, key=_key(root, 7000))
        valid = jnp.concatenate(
            (
                jnp.stack(tuple(result.valid for result in expert_results)),
                gate_result.valid[None, ...],
            ),
            axis=0,
        )
        status = jnp.concatenate(
            (
                jnp.stack(tuple(result.status for result in expert_results)),
                gate_result.status[None, ...],
            ),
            axis=0,
        )
        diagnostics = EnsembleFitDiagnostics(valid, status, method="mixture-of-experts")
        model = MixtureOfExpertsModel(
            tuple(result.as_trainable() for result in expert_results),
            gate_result.as_trainable(),
            temperature=self.temperature,
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                fit_features="conditional",
                fit_targets="conditional",
                fit_weights="conditional",
                fit_mode="unrolled",
                conditions=(
                    "Expert and gate recipes must expose the corresponding fit gradients.",
                ),
            ),
            "mixture-of-experts",
        )


__all__ = [
    "BaggingRecipe",
    "EnsembleFitDiagnostics",
    "FeatureSubsetModel",
    "HardVotingModel",
    "HardVotingRecipe",
    "HeterogeneousEnsembleModel",
    "HomogeneousEnsembleModel",
    "MixtureOfExpertsModel",
    "MixtureOfExpertsRecipe",
    "RandomSubspaceRecipe",
    "SoftVotingModel",
    "SoftVotingRecipe",
    "StackingModel",
    "StackingRecipe",
]
