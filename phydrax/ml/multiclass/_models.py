#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from .._batch import MLBatch
from .._contracts import (
    AbstractRecipe,
    DecisionFunctionModel,
    FitResult,
    GradientContract,
    LogProbabilityModel,
    ML_INFEASIBLE,
    ML_SUCCESS,
)
from .._schema import FeatureSchema, TargetSchema
from .._sparse_features import SparseFeatures
from ..discriminant._models import _labels_for


class CompositionDiagnostics(StrictModule):
    """Per-subproblem validity and status for a composed classifier."""

    valid: Array
    status: Array
    component_valid: Array
    component_status: Array
    component_count: int = eqx.field(static=True)
    method: str = eqx.field(static=True)

    def __init__(self, component_valid: Any, component_status: Any, /, *, method: str):
        validity = jnp.asarray(component_valid, dtype=bool)
        statuses = jnp.asarray(component_status, dtype=jnp.int32)
        self.component_valid = validity
        self.component_status = statuses
        self.valid = jnp.all(validity, axis=-1)
        self.status = jnp.max(statuses, axis=-1).astype(jnp.int32)
        self.component_count = int(validity.shape[-1])
        self.method = str(method)


def _child_keys(key: Any, count: int) -> tuple[Any, ...]:
    if key is None:
        return (None,) * count
    return tuple(jax.random.split(key, count))


def _features_and_mask(batch: MLBatch) -> tuple[Any, Any]:
    if isinstance(batch.features, SparseFeatures):
        return batch.features, None
    return batch.features, batch.feature_mask


def _subbatch(
    batch: MLBatch,
    targets: Array,
    *,
    sample_mask: Array | None = None,
    target_mask: Array | None = None,
    feature_schema: FeatureSchema | None = None,
) -> MLBatch:
    features, feature_mask = _features_and_mask(batch)
    mask = batch.sample_mask if sample_mask is None else (batch.sample_mask & sample_mask)
    return MLBatch(
        features,
        targets,
        feature_mask=feature_mask,
        target_mask=target_mask,
        sample_mask=mask,
        sample_weight=batch.sample_weight,
        measure_weight=batch.measure_weight,
        groups=batch.groups,
        feature_schema=batch.feature_schema if feature_schema is None else feature_schema,
        target_schema=TargetSchema("binary", class_labels=(0, 1)),
    )


def _binary_score(model: AbstractArrayModel, x: Any) -> Array:
    if isinstance(model, DecisionFunctionModel):
        score = model.decision_function(x)
        if model.out_size == 2:
            return score[..., 1] - score[..., 0]
        if score.ndim > 0 and score.shape[-1] == 1:
            return score[..., 0]
        return score
    if isinstance(model, LogProbabilityModel):
        log_probability = model.predict_log_proba(x)
        return log_probability[..., 1] - log_probability[..., 0]
    probability = model(x)
    if model.out_size == 2:
        positive = probability[..., 1]
    elif probability.ndim > 0 and probability.shape[-1] == 1:
        positive = probability[..., 0]
    else:
        positive = probability
    tiny = jnp.finfo(positive.dtype).tiny
    clipped = jnp.clip(positive, tiny, 1.0 - jnp.finfo(positive.dtype).eps)
    return jnp.log(clipped) - jnp.log1p(-clipped)


def _scalar_vocabulary_valid(batch: MLBatch, targets: Array, labels: Array) -> Array:
    target_valid = (
        batch.target_mask
        if batch.target_mask is not None
        else jnp.ones_like(targets, dtype=bool)
    )
    known = jnp.any(targets[..., None] == labels, axis=-1)
    return jnp.all(~(batch.sample_mask & target_valid) | known, axis=-1)


def _multilabel_domain_valid(batch: MLBatch, targets: Array) -> Array:
    target_valid = (
        batch.target_mask
        if batch.target_mask is not None
        else jnp.ones_like(targets, dtype=bool)
    )
    required = batch.sample_mask[..., None] & target_valid
    binary = (targets == 0) | (targets == 1)
    return jnp.all(~required | binary, axis=(-2, -1))


def _composition_result(
    model: AbstractArrayModel,
    results: tuple[FitResult, ...],
    *,
    method: str,
    prediction_inputs: str = "smooth",
    semantic_valid: Any = None,
) -> FitResult:
    component_valid = jnp.stack(tuple(result.valid for result in results), axis=-1)
    component_status = jnp.stack(tuple(result.status for result in results), axis=-1)
    if semantic_valid is not None:
        semantic = jnp.asarray(semantic_valid, dtype=bool)
        component_valid = jnp.concatenate((component_valid, semantic[..., None]), axis=-1)
        semantic_status = jnp.where(semantic, ML_SUCCESS, ML_INFEASIBLE).astype(jnp.int32)
        component_status = jnp.concatenate(
            (component_status, semantic_status[..., None]), axis=-1
        )
    diagnostics = CompositionDiagnostics(component_valid, component_status, method=method)
    contract = GradientContract(
        prediction_inputs=prediction_inputs,
        prediction_parameters="conditional" if prediction_inputs == "none" else "smooth",
        fit_features="conditional",
        fit_targets="none",
        fit_weights="conditional",
        fit_hyperparameters="conditional",
        fit_mode="direct",
        nondifferentiable_outputs=("predict", "predict_indices"),
        conditions=("all binary component fits valid", "fixed class vocabulary"),
    )
    return FitResult(
        model,
        diagnostics,
        valid=diagnostics.valid,
        status=diagnostics.status,
        method=method,
        gradient_contract=contract,
    )


def _validate_binary_models(
    models: tuple[AbstractArrayModel, ...], *, same_input: bool
) -> None:
    if not models:
        raise ValueError("Classifier composition requires at least one binary component.")
    if any(model.out_size not in {"scalar", 1, 2} for model in models):
        raise ValueError(
            "Every component recipe must produce a scalar or two-class model."
        )
    if same_input and any(model.in_size != models[0].in_size for model in models[1:]):
        raise ValueError("All parallel binary components must share one input shape.")


class OneVsRestModel(AbstractArrayModel):
    models: tuple[AbstractArrayModel, ...]
    labels: Array
    target_schema: TargetSchema
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        models: tuple[AbstractArrayModel, ...],
        labels: Array,
        target_schema: TargetSchema,
    ):
        _validate_binary_models(models, same_input=True)
        if len(models) < 2 or len(models) != int(jnp.asarray(labels).shape[0]):
            raise ValueError("One-vs-rest components must align with class labels.")
        self.models = tuple(models)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.in_size = int(models[0].in_size)
        self.out_size = len(models)

    def decision_function(self, x: Any, /) -> Array:
        return jnp.stack(tuple(_binary_score(model, x) for model in self.models), axis=-1)

    def predict_log_proba(self, x: Any, /) -> Array:
        return jax.nn.log_softmax(self.decision_function(x), axis=-1)

    def predict_proba(self, x: Any, /) -> Array:
        return jax.nn.softmax(self.decision_function(x), axis=-1)

    def predict_indices(self, x: Any, /) -> Array:
        return jnp.argmax(self.decision_function(x), axis=-1)

    def predict(self, x: Any, /) -> Array:
        return jnp.take(self.labels, self.predict_indices(x), axis=0)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


class OneVsRestRecipe(AbstractRecipe):
    base_recipe: AbstractRecipe
    num_classes: int | None = eqx.field(static=True)

    def __init__(self, base_recipe: AbstractRecipe, /, *, num_classes: int | None = None):
        if not isinstance(base_recipe, AbstractRecipe):
            raise TypeError("base_recipe must be an AbstractRecipe.")
        self.base_recipe = base_recipe
        self.num_classes = None if num_classes is None else int(num_classes)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        labels, schema = _labels_for(batch, self.num_classes)
        y = batch.require_targets()
        if batch.target_shape != ():
            raise ValueError("One-vs-rest requires scalar multiclass targets.")
        keys = _child_keys(key, int(labels.shape[0]))
        results = tuple(
            self.base_recipe.fit_batch(
                _subbatch(
                    batch, (y == label).astype(jnp.int32), target_mask=batch.target_mask
                ),
                key=child_key,
            )
            for label, child_key in zip(labels, keys, strict=True)
        )
        model = OneVsRestModel(
            tuple(result.as_trainable() for result in results), labels, schema
        )
        return _composition_result(
            model,
            results,
            method="one-vs-rest",
            semantic_valid=_scalar_vocabulary_valid(batch, y, labels),
        )


class OneVsOneModel(AbstractArrayModel):
    models: tuple[AbstractArrayModel, ...]
    pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    labels: Array
    target_schema: TargetSchema
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        models: tuple[AbstractArrayModel, ...],
        pairs: tuple[tuple[int, int], ...],
        labels: Array,
        target_schema: TargetSchema,
    ):
        if len(models) != len(pairs) or not models:
            raise ValueError("One-vs-one models must align with class pairs.")
        _validate_binary_models(models, same_input=True)
        self.models = tuple(models)
        self.pairs = tuple(pairs)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.in_size = int(models[0].in_size)
        self.out_size = int(self.labels.shape[0])

    def pairwise_decision_function(self, x: Any, /) -> Array:
        return jnp.stack(tuple(_binary_score(model, x) for model in self.models), axis=-1)

    def vote_counts(self, x: Any, /) -> Array:
        pair_scores = self.pairwise_decision_function(x)
        votes = jnp.zeros(pair_scores.shape[:-1] + (self.out_size,), dtype=jnp.int32)
        for pair_index, (negative, positive) in enumerate(self.pairs):
            choose_positive = pair_scores[..., pair_index] >= 0.0
            votes = votes.at[..., negative].add((~choose_positive).astype(jnp.int32))
            votes = votes.at[..., positive].add(choose_positive.astype(jnp.int32))
        return votes

    def decision_function(self, x: Any, /) -> Array:
        pair_scores = self.pairwise_decision_function(x)
        evidence = jnp.zeros(
            pair_scores.shape[:-1] + (self.out_size,), dtype=pair_scores.dtype
        )
        for pair_index, (negative, positive) in enumerate(self.pairs):
            score = pair_scores[..., pair_index]
            evidence = evidence.at[..., negative].add(-jax.nn.softplus(score))
            evidence = evidence.at[..., positive].add(-jax.nn.softplus(-score))
        return evidence

    def predict_log_proba(self, x: Any, /) -> Array:
        return jax.nn.log_softmax(self.decision_function(x), axis=-1)

    def predict_proba(self, x: Any, /) -> Array:
        return jax.nn.softmax(self.decision_function(x), axis=-1)

    def predict_indices(self, x: Any, /) -> Array:
        return jnp.argmax(self.vote_counts(x), axis=-1)

    def predict(self, x: Any, /) -> Array:
        return jnp.take(self.labels, self.predict_indices(x), axis=0)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


class OneVsOneRecipe(AbstractRecipe):
    base_recipe: AbstractRecipe
    num_classes: int | None = eqx.field(static=True)

    def __init__(self, base_recipe: AbstractRecipe, /, *, num_classes: int | None = None):
        if not isinstance(base_recipe, AbstractRecipe):
            raise TypeError("base_recipe must be an AbstractRecipe.")
        self.base_recipe = base_recipe
        self.num_classes = None if num_classes is None else int(num_classes)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        labels, schema = _labels_for(batch, self.num_classes)
        y = batch.require_targets()
        if batch.target_shape != ():
            raise ValueError("One-vs-one requires scalar multiclass targets.")
        pairs = tuple(
            (i, j)
            for i in range(int(labels.shape[0]))
            for j in range(i + 1, int(labels.shape[0]))
        )
        keys = _child_keys(key, len(pairs))
        results = []
        for (negative, positive), child_key in zip(pairs, keys, strict=True):
            selected = (y == labels[negative]) | (y == labels[positive])
            targets = (y == labels[positive]).astype(jnp.int32)
            results.append(
                self.base_recipe.fit_batch(
                    _subbatch(
                        batch,
                        targets,
                        sample_mask=selected,
                        target_mask=batch.target_mask,
                    ),
                    key=child_key,
                )
            )
        result_tuple = tuple(results)
        model = OneVsOneModel(
            tuple(result.as_trainable() for result in result_tuple), pairs, labels, schema
        )
        return _composition_result(
            model,
            result_tuple,
            method="one-vs-one",
            semantic_valid=_scalar_vocabulary_valid(batch, y, labels),
        )


class OutputCodeModel(AbstractArrayModel):
    models: tuple[AbstractArrayModel, ...]
    codebook: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    labels: Array
    target_schema: TargetSchema
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        models: tuple[AbstractArrayModel, ...],
        codebook: tuple[tuple[int, ...], ...],
        labels: Array,
        target_schema: TargetSchema,
    ):
        _validate_binary_models(models, same_input=True)
        if (
            not codebook
            or len(models) != len(codebook[0])
            or len(codebook) != int(jnp.asarray(labels).shape[0])
        ):
            raise ValueError(
                "Output-code models must align with codebook rows and columns."
            )
        self.models = tuple(models)
        self.codebook = tuple(tuple(int(bit) for bit in row) for row in codebook)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.in_size = int(models[0].in_size)
        self.out_size = len(self.codebook)

    def code_decision_function(self, x: Any, /) -> Array:
        return jnp.stack(tuple(_binary_score(model, x) for model in self.models), axis=-1)

    def decision_function(self, x: Any, /) -> Array:
        bit_scores = self.code_decision_function(x)
        signs = 2.0 * jnp.asarray(self.codebook, dtype=bit_scores.dtype) - 1.0
        return jnp.sum(-jax.nn.softplus(-bit_scores[..., None, :] * signs), axis=-1)

    def predict_log_proba(self, x: Any, /) -> Array:
        return jax.nn.log_softmax(self.decision_function(x), axis=-1)

    def predict_proba(self, x: Any, /) -> Array:
        return jax.nn.softmax(self.decision_function(x), axis=-1)

    def predict_indices(self, x: Any, /) -> Array:
        hard_bits = self.code_decision_function(x) >= 0.0
        code = jnp.asarray(self.codebook, dtype=bool)
        distance = jnp.sum(hard_bits[..., None, :] != code, axis=-1)
        return jnp.argmin(distance, axis=-1)

    def predict(self, x: Any, /) -> Array:
        return jnp.take(self.labels, self.predict_indices(x), axis=0)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


class OutputCodeRecipe(AbstractRecipe):
    base_recipe: AbstractRecipe
    codebook: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    num_classes: int | None = eqx.field(static=True)

    def __init__(
        self,
        base_recipe: AbstractRecipe,
        codebook: tuple[tuple[int, ...], ...] = (),
        /,
        *,
        num_classes: int | None = None,
    ):
        if not isinstance(base_recipe, AbstractRecipe):
            raise TypeError("base_recipe must be an AbstractRecipe.")
        self.base_recipe = base_recipe
        self.codebook = tuple(tuple(int(bit) for bit in row) for row in codebook)
        self.num_classes = None if num_classes is None else int(num_classes)
        if self.codebook:
            width = len(self.codebook[0])
            if (
                len(self.codebook) < 2
                or width < 3
                or any(len(row) != width for row in self.codebook)
                or any(bit not in {0, 1} for row in self.codebook for bit in row)
                or len(set(self.codebook)) != len(self.codebook)
            ):
                raise ValueError(
                    "codebook rows must be distinct equal-width binary codes with at least two rows and width >= 3."
                )
            if any(len(set(column)) < 2 for column in zip(*self.codebook, strict=True)):
                raise ValueError(
                    "Every output-code column must separate at least two classes."
                )
            minimum_distance = min(
                sum(
                    left_bit != right_bit
                    for left_bit, right_bit in zip(left, right, strict=True)
                )
                for row, left in enumerate(self.codebook)
                for right in self.codebook[row + 1 :]
            )
            if minimum_distance < 3:
                raise ValueError(
                    "An error-correcting codebook requires minimum Hamming distance >= 3."
                )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        labels, schema = _labels_for(batch, self.num_classes)
        classes = int(labels.shape[0])
        y = batch.require_targets()
        if batch.target_shape != ():
            raise ValueError(
                "Output-code classification requires scalar multiclass targets."
            )
        codebook = self.codebook or tuple(
            tuple(int(i == bit) for bit in range(classes))
            + tuple(int(i != bit) for bit in range(classes))
            for i in range(classes)
        )
        if len(codebook) != classes:
            raise ValueError(
                "codebook rows must align with the external class vocabulary."
            )
        matched = y[..., None] == labels
        encoded = jnp.argmax(matched, axis=-1)
        code = jnp.asarray(codebook, dtype=jnp.int32)
        targets = code[encoded]
        keys = _child_keys(key, len(codebook[0]))
        results = tuple(
            self.base_recipe.fit_batch(
                _subbatch(batch, targets[..., bit], target_mask=batch.target_mask),
                key=child_key,
            )
            for bit, child_key in enumerate(keys)
        )
        model = OutputCodeModel(
            tuple(result.as_trainable() for result in results), codebook, labels, schema
        )
        return _composition_result(
            model,
            results,
            method="error-correcting-output-code",
            semantic_valid=_scalar_vocabulary_valid(batch, y, labels),
        )


class MultilabelModel(AbstractArrayModel):
    models: tuple[AbstractArrayModel, ...]
    target_schema: TargetSchema
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self, models: tuple[AbstractArrayModel, ...], target_schema: TargetSchema
    ):
        _validate_binary_models(models, same_input=True)
        self.models = tuple(models)
        self.target_schema = target_schema
        self.in_size = int(models[0].in_size)
        self.out_size = len(models)

    def decision_function(self, x: Any, /) -> Array:
        return jnp.stack(tuple(_binary_score(model, x) for model in self.models), axis=-1)

    def predict_proba(self, x: Any, /) -> Array:
        return jax.nn.sigmoid(self.decision_function(x))

    def predict_log_proba(self, x: Any, /) -> Array:
        return jax.nn.log_sigmoid(self.decision_function(x))

    def predict(self, x: Any, /) -> Array:
        return (self.decision_function(x) >= 0.0).astype(jnp.int32)

    def predict_indices(self, x: Any, /) -> Array:
        return self.predict(x)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


class MultilabelRecipe(AbstractRecipe):
    base_recipe: AbstractRecipe

    def __init__(self, base_recipe: AbstractRecipe, /):
        if not isinstance(base_recipe, AbstractRecipe):
            raise TypeError("base_recipe must be an AbstractRecipe.")
        self.base_recipe = base_recipe

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        targets = batch.require_targets()
        if len(batch.target_shape or ()) != 1:
            raise ValueError("Multilabel targets must end in one label axis.")
        labels = int(targets.shape[-1])
        keys = _child_keys(key, labels)
        results = tuple(
            self.base_recipe.fit_batch(
                _subbatch(
                    batch,
                    targets[..., index],
                    target_mask=None
                    if batch.target_mask is None
                    else batch.target_mask[..., index],
                ),
                key=child_key,
            )
            for index, child_key in enumerate(keys)
        )
        schema = (
            batch.target_schema
            if batch.target_schema.kind == "multilabel"
            else TargetSchema(
                "multilabel", names=tuple(f"label_{index}" for index in range(labels))
            )
        )
        model = MultilabelModel(
            tuple(result.as_trainable() for result in results), schema
        )
        return _composition_result(
            model,
            results,
            method="multilabel-binary-relevance",
            semantic_valid=_multilabel_domain_valid(batch, targets),
        )


def _chain_schema(schema: FeatureSchema, count: int) -> FeatureSchema:
    names = schema.names + tuple(f"chain_label_{index}" for index in range(count))
    kinds = schema.kinds + ("boolean",) * count
    return FeatureSchema(names, kinds=kinds, layout_id=schema.layout_id)


def _append_chain_batch(batch: MLBatch, appended: Array, appended_mask: Array) -> MLBatch:
    features = jnp.concatenate((batch.dense_features(), appended), axis=-1)
    feature_mask = jnp.concatenate((batch.feature_mask, appended_mask), axis=-1)
    return MLBatch(
        features,
        batch.targets,
        feature_mask=feature_mask,
        target_mask=batch.target_mask,
        sample_mask=batch.sample_mask,
        sample_weight=batch.sample_weight,
        measure_weight=batch.measure_weight,
        groups=batch.groups,
        feature_schema=_chain_schema(batch.feature_schema, appended.shape[-1]),
        target_schema=batch.target_schema,
    )


class ClassifierChainModel(AbstractArrayModel):
    models: tuple[AbstractArrayModel, ...]
    target_schema: TargetSchema
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        models: tuple[AbstractArrayModel, ...],
        target_schema: TargetSchema,
        *,
        in_size: int,
    ):
        _validate_binary_models(models, same_input=False)
        if any(
            model.in_size != int(in_size) + index for index, model in enumerate(models)
        ):
            raise ValueError(
                "Classifier-chain component inputs must grow by one feature per link."
            )
        self.models = tuple(models)
        self.target_schema = target_schema
        self.in_size = int(in_size)
        self.out_size = len(models)

    def decision_function(self, x: Any, /) -> Array:
        augmented = jnp.asarray(x)
        scores = []
        for model in self.models:
            score = _binary_score(model, augmented)
            scores.append(score)
            augmented = jnp.concatenate(
                (augmented, (score >= 0.0)[..., None].astype(augmented.dtype)), axis=-1
            )
        return jnp.stack(scores, axis=-1)

    def predict_proba(self, x: Any, /) -> Array:
        return jax.nn.sigmoid(self.decision_function(x))

    def predict_log_proba(self, x: Any, /) -> Array:
        return jax.nn.log_sigmoid(self.decision_function(x))

    def predict(self, x: Any, /) -> Array:
        return (self.decision_function(x) >= 0.0).astype(jnp.int32)

    def predict_indices(self, x: Any, /) -> Array:
        return self.predict(x)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


class SmoothClassifierChainModel(AbstractArrayModel):
    models: tuple[AbstractArrayModel, ...]
    target_schema: TargetSchema
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        models: tuple[AbstractArrayModel, ...],
        target_schema: TargetSchema,
        *,
        in_size: int,
    ):
        _validate_binary_models(models, same_input=False)
        if any(
            model.in_size != int(in_size) + index for index, model in enumerate(models)
        ):
            raise ValueError(
                "Smooth classifier-chain component inputs must grow by one feature per link."
            )
        self.models = tuple(models)
        self.target_schema = target_schema
        self.in_size = int(in_size)
        self.out_size = len(models)

    def decision_function(self, x: Any, /) -> Array:
        augmented = jnp.asarray(x)
        scores = []
        for model in self.models:
            score = _binary_score(model, augmented)
            scores.append(score)
            augmented = jnp.concatenate(
                (augmented, jax.nn.sigmoid(score)[..., None].astype(augmented.dtype)),
                axis=-1,
            )
        return jnp.stack(scores, axis=-1)

    def predict_proba(self, x: Any, /) -> Array:
        return jax.nn.sigmoid(self.decision_function(x))

    def predict_log_proba(self, x: Any, /) -> Array:
        return jax.nn.log_sigmoid(self.decision_function(x))

    def predict(self, x: Any, /) -> Array:
        return (self.decision_function(x) >= 0.0).astype(jnp.int32)

    def predict_indices(self, x: Any, /) -> Array:
        return self.predict(x)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


def _fit_chain(
    base_recipe: AbstractRecipe, batch: MLBatch, key: Any, *, smooth: bool
) -> FitResult:
    targets = batch.require_targets()
    if len(batch.target_shape or ()) != 1:
        raise ValueError("Classifier-chain targets must end in one label axis.")
    count = int(targets.shape[-1])
    keys = _child_keys(key, count)
    results = []
    for index, child_key in enumerate(keys):
        if index == 0:
            current = batch
        else:
            target_mask = (
                jnp.ones_like(targets[..., :index], dtype=bool)
                if batch.target_mask is None
                else batch.target_mask[..., :index]
            )
            current = _append_chain_batch(batch, targets[..., :index], target_mask)
        binary = _subbatch(
            current,
            targets[..., index],
            target_mask=None
            if batch.target_mask is None
            else batch.target_mask[..., index],
            feature_schema=current.feature_schema,
        )
        results.append(base_recipe.fit_batch(binary, key=child_key))
    result_tuple = tuple(results)
    schema = (
        batch.target_schema
        if batch.target_schema.kind == "multilabel"
        else TargetSchema(
            "multilabel", names=tuple(f"label_{index}" for index in range(count))
        )
    )
    models = tuple(result.as_trainable() for result in result_tuple)
    model: AbstractArrayModel
    if smooth:
        model = SmoothClassifierChainModel(models, schema, in_size=batch.feature_count)
    else:
        model = ClassifierChainModel(models, schema, in_size=batch.feature_count)
    return _composition_result(
        model,
        result_tuple,
        method="smooth-classifier-chain" if smooth else "classifier-chain",
        prediction_inputs="smooth" if smooth else "none",
        semantic_valid=_multilabel_domain_valid(batch, targets),
    )


class ClassifierChainRecipe(AbstractRecipe):
    base_recipe: AbstractRecipe

    def __init__(self, base_recipe: AbstractRecipe, /):
        if not isinstance(base_recipe, AbstractRecipe):
            raise TypeError("base_recipe must be an AbstractRecipe.")
        self.base_recipe = base_recipe

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_chain(self.base_recipe, batch, key, smooth=False)


class SmoothClassifierChainRecipe(AbstractRecipe):
    base_recipe: AbstractRecipe

    def __init__(self, base_recipe: AbstractRecipe, /):
        if not isinstance(base_recipe, AbstractRecipe):
            raise TypeError("base_recipe must be an AbstractRecipe.")
        self.base_recipe = base_recipe

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_chain(self.base_recipe, batch, key, smooth=True)


__all__ = [
    "ClassifierChainModel",
    "ClassifierChainRecipe",
    "CompositionDiagnostics",
    "MultilabelModel",
    "MultilabelRecipe",
    "OneVsOneModel",
    "OneVsOneRecipe",
    "OneVsRestModel",
    "OneVsRestRecipe",
    "OutputCodeModel",
    "OutputCodeRecipe",
    "SmoothClassifierChainModel",
    "SmoothClassifierChainRecipe",
]
