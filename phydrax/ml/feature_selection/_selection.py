#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from ..._model import AbstractArrayModel, ModelBinding
from ..._strict import StrictModule
from .._batch import MLBatch
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import assign_bins, quantile_bin_edges
from .._schema import FeatureSchema


class ExactSelection(StrictModule):
    """Fixed-capacity exact selection; inactive padded entries are explicit."""

    indices: Array
    selected: Array
    scores: Array

    def __init__(self, indices: Any, selected: Any, scores: Any, /):
        indices_ = jnp.asarray(indices, dtype=jnp.int32)
        selected_ = jnp.asarray(selected, dtype=bool)
        if indices_.ndim != 1 or selected_.shape != indices_.shape:
            raise ValueError(
                "indices and selected must be aligned one-dimensional arrays."
            )
        self.indices = indices_
        self.selected = selected_
        self.scores = jnp.asarray(scores)


class FeatureSelectionDiagnostics(StrictModule):
    selection: ExactSelection | None
    relaxed_gates: Array | None
    valid: Array
    status: Array
    iterations: Array
    estimator_status: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        selection: ExactSelection | None = None,
        relaxed_gates: Any = None,
        valid: Any,
        status: Any,
        iterations: Any = 0,
        estimator_status: Any = (),
        method: str,
    ):
        if (selection is None) == (relaxed_gates is None):
            raise ValueError(
                "Diagnostics must contain exactly one of exact selection or relaxed gates."
            )
        self.selection = selection
        self.relaxed_gates = None if relaxed_gates is None else jnp.asarray(relaxed_gates)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.estimator_status = jnp.asarray(estimator_status, dtype=jnp.int32)
        self.method = str(method)


class ExactFeatureSelectorModel(AbstractArrayModel):
    """Exact fixed-capacity gather, smooth in values conditional on fitted indices."""

    selection: ExactSelection
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(self, selection: ExactSelection, /, *, input_size: int):
        if not isinstance(selection, ExactSelection):
            raise TypeError("selection must be an ExactSelection.")
        self.selection = selection
        self.in_size = int(input_size)
        self.out_size = int(selection.indices.shape[0])

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        selected = jnp.take(jnp.asarray(x), self.selection.indices, axis=-1)
        mask = self.selection.selected.reshape((1,) * (selected.ndim - 1) + (-1,))
        return jnp.where(mask, selected, 0)


class ContinuousFeatureGateModel(AbstractArrayModel):
    """Smooth, shape-preserving sparse feature gate."""

    gates: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(self, gates: Any, /):
        gates_ = jnp.asarray(gates)
        if gates_.ndim != 1 or gates_.shape[0] == 0:
            raise ValueError("gates must be a nonempty vector.")
        self.gates = eqx.error_if(
            gates_,
            jnp.any(~jnp.isfinite(gates_))
            | jnp.any(gates_ < 0.0)
            | jnp.any(gates_ > 1.0),
            "gates must be finite values in [0, 1].",
        )
        self.in_size = int(gates_.shape[0])
        self.out_size = int(gates_.shape[0])

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return jnp.asarray(x) * self.gates


def _flatten(batch: MLBatch) -> tuple[Array, Array, Array]:
    features = batch.dense_features()
    mask = batch.feature_mask & batch.sample_mask[..., None]
    weights = batch.effective_weight("statistical")
    weights = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
        "Sample weights must be finite and nonnegative.",
    )
    return (
        features.reshape((-1, batch.feature_count)),
        mask.reshape((-1, batch.feature_count)),
        weights.reshape((-1,)),
    )


def _weighted_variance(batch: MLBatch) -> Array:
    x, mask, sample_weight = _flatten(batch)
    weights = jnp.where(mask, sample_weight[:, None], 0.0)
    total = jnp.sum(weights, axis=0)
    safe_total = jnp.maximum(total, jnp.finfo(weights.dtype).tiny)
    mean = jnp.sum(weights * x, axis=0) / safe_total
    variance = (
        jnp.sum(weights * jnp.real((x - mean) * jnp.conj(x - mean)), axis=0) / safe_total
    )
    return jnp.where(total > 0.0, variance, -jnp.inf)


def _target_vector(batch: MLBatch) -> tuple[Array, Array]:
    targets = batch.require_targets()
    sample_ndim = len(batch.case_shape) + 1
    if targets.ndim != sample_ndim:
        raise ValueError("This selector requires scalar targets per sample.")
    target_mask = batch.target_mask
    if target_mask is None:
        target_mask = jnp.ones_like(targets, dtype=bool)
    valid = batch.sample_mask & target_mask
    safe_targets = jnp.where(valid, targets, 0)
    return safe_targets.reshape((-1,)), valid.reshape((-1,))


def _correlation_scores(batch: MLBatch) -> Array:
    x, feature_mask, sample_weight = _flatten(batch)
    y, target_valid = _target_vector(batch)
    weights = jnp.where(feature_mask & target_valid[:, None], sample_weight[:, None], 0.0)
    total = jnp.sum(weights, axis=0)
    safe = jnp.maximum(total, jnp.finfo(weights.dtype).tiny)
    x_mean = jnp.sum(weights * x, axis=0) / safe
    y_mean = jnp.sum(weights * y[:, None], axis=0) / safe
    centered_x = x - x_mean
    centered_y = y[:, None] - y_mean
    covariance = jnp.sum(weights * centered_x * jnp.conj(centered_y), axis=0) / safe
    x_var = jnp.sum(weights * jnp.real(centered_x * jnp.conj(centered_x)), axis=0) / safe
    y_var = jnp.sum(weights * jnp.real(centered_y * jnp.conj(centered_y)), axis=0) / safe
    scale = jnp.sqrt(jnp.maximum(x_var * y_var, jnp.finfo(weights.dtype).tiny))
    return jnp.where(total > 0.0, jnp.abs(covariance) / scale, -jnp.inf)


def _selection(scores: Array, eligible: Array, capacity: int) -> ExactSelection:
    if capacity <= 0 or capacity > scores.shape[0]:
        raise ValueError("Selection capacity must be in [1, feature_count].")
    if jnp.issubdtype(scores.dtype, jnp.complexfloating):
        raise TypeError("Exact feature ranking requires real-valued scores.")
    ranked = jnp.where(eligible & jnp.isfinite(scores), scores, -jnp.inf)
    top_scores, indices = jax.lax.top_k(ranked, capacity)
    selected = jnp.isfinite(top_scores)
    safe_indices = jnp.where(selected, indices, 0)
    return ExactSelection(safe_indices, selected, scores)


def _exact_result(
    selection: ExactSelection,
    feature_count: int,
    /,
    *,
    method: str,
    iterations: Any = 0,
    estimator_status: Any = (),
) -> FitResult:
    valid = jnp.any(selection.selected) & jnp.all(
        jnp.isfinite(jnp.where(jnp.isfinite(selection.scores), selection.scores, 0.0))
    )
    status = jnp.where(valid, ML_SUCCESS, ML_INSUFFICIENT_DATA)
    diagnostics = FeatureSelectionDiagnostics(
        selection=selection,
        valid=valid,
        status=status,
        iterations=iterations,
        estimator_status=estimator_status,
        method=method,
    )
    return FitResult(
        ExactFeatureSelectorModel(selection, input_size=feature_count),
        diagnostics,
        valid=valid,
        status=status,
        method=method,
        gradient_contract=GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="none",
            fit_mode="stopped",
            nondifferentiable_outputs=("selected_indices", "selected_mask"),
            conditions=(
                "Selection capacity is static; inactive padded entries evaluate to zero.",
            ),
        ),
    )


class VarianceFilterRecipe(AbstractRecipe):
    threshold: float = eqx.field(static=True)
    max_features: int | None = eqx.field(static=True)

    def __init__(self, threshold: float = 0.0, /, *, max_features: int | None = None):
        if float(threshold) < 0.0 or (
            max_features is not None and int(max_features) <= 0
        ):
            raise ValueError("threshold must be nonnegative and max_features positive.")
        self.threshold = float(threshold)
        self.max_features = None if max_features is None else int(max_features)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        scores = _weighted_variance(batch)
        capacity = (
            batch.feature_count
            if self.max_features is None
            else min(self.max_features, batch.feature_count)
        )
        return _exact_result(
            _selection(scores, scores > self.threshold, capacity),
            batch.feature_count,
            method="variance-filter",
        )


class ScoreFilterRecipe(AbstractRecipe):
    scorer: Callable[[MLBatch], Array] = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    max_features: int | None = eqx.field(static=True)

    def __init__(
        self,
        scorer: Callable[[MLBatch], Array] | None = None,
        /,
        *,
        threshold: float = 0.0,
        max_features: int | None = None,
    ):
        if scorer is not None and not callable(scorer):
            raise TypeError("scorer must be callable.")
        if max_features is not None and int(max_features) <= 0:
            raise ValueError("max_features must be positive.")
        self.scorer = _correlation_scores if scorer is None else scorer
        self.threshold = float(threshold)
        self.max_features = None if max_features is None else int(max_features)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        scores = jnp.asarray(self.scorer(batch))
        if scores.shape != (batch.feature_count,):
            raise ValueError("scorer must return one score per feature.")
        capacity = (
            batch.feature_count
            if self.max_features is None
            else min(self.max_features, batch.feature_count)
        )
        return _exact_result(
            _selection(scores, scores > self.threshold, capacity),
            batch.feature_count,
            method="score-filter",
        )


class MutualInformationFilterRecipe(AbstractRecipe):
    num_bins: int = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    max_features: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_bins: int = 16,
        threshold: float = 0.0,
        max_features: int | None = None,
    ):
        if int(num_bins) < 2 or (max_features is not None and int(max_features) <= 0):
            raise ValueError("num_bins must be at least two and max_features positive.")
        self.num_bins = int(num_bins)
        self.threshold = float(threshold)
        self.max_features = None if max_features is None else int(max_features)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, feature_mask, sample_weight = _flatten(batch)
        y, target_valid = _target_vector(batch)
        if jnp.issubdtype(x.dtype, jnp.complexfloating) or jnp.issubdtype(
            y.dtype, jnp.complexfloating
        ):
            raise TypeError("Mutual information binning is undefined for complex values.")
        valid_weight = sample_weight * target_valid.astype(sample_weight.dtype)
        feature_total = jnp.sum(feature_mask * valid_weight[:, None], axis=0)
        feature_mean = jnp.sum(
            jnp.where(feature_mask, x, 0.0) * valid_weight[:, None], axis=0
        ) / jnp.maximum(feature_total, 1.0)
        safe_x = jnp.where(feature_mask & target_valid[:, None], x, feature_mean)
        y_total = jnp.sum(valid_weight)
        y_mean = jnp.sum(valid_weight * y) / jnp.maximum(y_total, 1.0)
        safe_y = jnp.where(target_valid, y, y_mean)
        x_edges = quantile_bin_edges(safe_x, num_bins=self.num_bins)
        y_edges = quantile_bin_edges(safe_y[:, None], num_bins=self.num_bins)
        x_bins = assign_bins(safe_x, x_edges)
        y_bins = assign_bins(safe_y[:, None], y_edges)[:, 0]
        y_one_hot = jax.nn.one_hot(y_bins, self.num_bins)

        def one_feature(feature_bins: Array, feature_valid: Array) -> Array:
            weight = valid_weight * feature_valid.astype(valid_weight.dtype)
            x_one_hot = jax.nn.one_hot(feature_bins, self.num_bins)
            joint = jnp.einsum("n,ni,nj->ij", weight, x_one_hot, y_one_hot)
            joint = joint / jnp.maximum(jnp.sum(joint), jnp.finfo(weight.dtype).tiny)
            px = jnp.sum(joint, axis=1, keepdims=True)
            py = jnp.sum(joint, axis=0, keepdims=True)
            independent = px * py
            ratio = jnp.where(
                joint > 0.0,
                joint / jnp.maximum(independent, jnp.finfo(weight.dtype).tiny),
                1.0,
            )
            return jnp.sum(jnp.where(joint > 0.0, joint * jnp.log(ratio), 0.0))

        scores = jax.vmap(one_feature, in_axes=(1, 1))(x_bins, feature_mask)
        capacity = (
            batch.feature_count
            if self.max_features is None
            else min(self.max_features, batch.feature_count)
        )
        return _exact_result(
            _selection(scores, scores > self.threshold, capacity),
            batch.feature_count,
            method="mutual-information-filter",
        )


def _active_indices(active: Array, capacity: int) -> Array:
    return jnp.nonzero(active, size=capacity, fill_value=0)[0]


def _take_features(batch: MLBatch, indices: Array) -> MLBatch:
    return batch.with_features(
        jnp.take(batch.dense_features(), indices, axis=-1),
        feature_schema=FeatureSchema.anonymous(indices.shape[0]),
        feature_mask=jnp.take(batch.feature_mask, indices, axis=-1),
    )


def _importance(
    model: AbstractArrayModel,
    feature_count: int,
    getter: Callable[[AbstractArrayModel], Array],
    /,
) -> Array:
    value = jnp.asarray(getter(model))
    axes = tuple(index for index, size in enumerate(value.shape) if size == feature_count)
    if not axes:
        raise ValueError("Importance array has no axis matching the feature count.")
    if value.ndim >= 2 and value.shape[-2] == feature_count:
        feature_axis = value.ndim - 2
    elif value.shape[-1] == feature_count:
        feature_axis = value.ndim - 1
    elif len(axes) == 1:
        feature_axis = axes[0]
    else:
        raise ValueError(
            "Importance array has ambiguous feature axes; supply importance_getter "
            "returning a feature vector."
        )
    reduce_axes = tuple(index for index in range(value.ndim) if index != feature_axis)
    return jnp.mean(jnp.abs(value), axis=reduce_axes) if reduce_axes else jnp.abs(value)


class RecursiveFeatureEliminationRecipe(AbstractRecipe):
    estimator: AbstractRecipe
    num_features: int = eqx.field(static=True)
    importance_getter: Callable[[AbstractArrayModel], Array] = eqx.field(static=True)

    def __init__(
        self,
        estimator: AbstractRecipe,
        /,
        *,
        num_features: int,
        importance_getter: Callable[[AbstractArrayModel], Array],
    ):
        if not isinstance(estimator, AbstractRecipe):
            raise TypeError("estimator must be an AbstractRecipe.")
        if int(num_features) <= 0:
            raise ValueError("num_features must be positive.")
        if not callable(importance_getter):
            raise TypeError("importance_getter must be callable.")
        self.estimator = estimator
        self.num_features = int(num_features)
        self.importance_getter = importance_getter

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if self.num_features > batch.feature_count:
            raise ValueError("num_features cannot exceed the feature count.")
        active = jnp.ones((batch.feature_count,), dtype=bool)
        statuses = []
        scores = jnp.zeros((batch.feature_count,))
        for step in range(batch.feature_count - self.num_features):
            member_key = None if key is None else jr.fold_in(key, step)
            capacity = batch.feature_count - step
            indices = _active_indices(active, capacity)
            result = self.estimator.fit_batch(
                _take_features(batch, indices), key=member_key
            )
            statuses.append(result.status)
            local_scores = _importance(
                result.as_trainable(), capacity, self.importance_getter
            )
            scores = (
                jnp.full((batch.feature_count,), jnp.inf).at[indices].set(local_scores)
            )
            removed = jnp.argmin(jnp.where(active, scores, jnp.inf))
            active = active & ~jax.nn.one_hot(removed, batch.feature_count, dtype=bool)
        indices = _active_indices(active, self.num_features)
        final = self.estimator.fit_batch(
            _take_features(batch, indices),
            key=None if key is None else jr.fold_in(key, batch.feature_count),
        )
        statuses.append(final.status)
        local_scores = _importance(
            final.as_trainable(), self.num_features, self.importance_getter
        )
        scores = jnp.zeros((batch.feature_count,)).at[indices].set(local_scores)
        return _exact_result(
            ExactSelection(indices, jnp.ones((self.num_features,), dtype=bool), scores),
            batch.feature_count,
            method="recursive-feature-elimination",
            iterations=batch.feature_count - self.num_features,
            estimator_status=jnp.stack(statuses),
        )


def _default_estimator_score(model: AbstractArrayModel, batch: MLBatch) -> Array:
    targets = batch.require_targets()
    prediction = jnp.asarray(model(batch.dense_features(), key=None))
    residual = prediction - targets
    sample_ndim = len(batch.case_shape) + 1
    if residual.ndim > sample_ndim:
        loss = jnp.mean(
            jnp.real(residual * jnp.conj(residual)),
            axis=tuple(range(sample_ndim, residual.ndim)),
        )
    else:
        loss = jnp.real(residual * jnp.conj(residual))
    weights = batch.effective_weight("statistical")
    return -jnp.sum(weights * loss) / jnp.maximum(
        jnp.sum(weights), jnp.finfo(weights.dtype).tiny
    )


class SequentialFeatureSelectionRecipe(AbstractRecipe):
    estimator: AbstractRecipe
    num_features: int = eqx.field(static=True)
    direction: Literal["forward", "backward"] = eqx.field(static=True)
    validation_fraction: float = eqx.field(static=True)
    scorer: Callable[[AbstractArrayModel, MLBatch], Array] = eqx.field(static=True)

    def __init__(
        self,
        estimator: AbstractRecipe,
        /,
        *,
        num_features: int,
        direction: Literal["forward", "backward"] = "forward",
        validation_fraction: float = 0.2,
        scorer: Callable[[AbstractArrayModel, MLBatch], Array] | None = None,
    ):
        if not isinstance(estimator, AbstractRecipe):
            raise TypeError("estimator must be an AbstractRecipe.")
        if int(num_features) <= 0 or direction not in ("forward", "backward"):
            raise ValueError("num_features must be positive and direction supported.")
        if not 0.0 < float(validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be in (0, 1).")
        self.estimator = estimator
        self.num_features = int(num_features)
        self.direction = direction
        self.validation_fraction = float(validation_fraction)
        self.scorer = _default_estimator_score if scorer is None else scorer

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError(
                "SequentialFeatureSelectionRecipe requires an explicit JAX key."
            )
        if self.num_features > batch.feature_count:
            raise ValueError("num_features cannot exceed the feature count.")
        validation_size = max(1, int(batch.sample_count * self.validation_fraction))
        if validation_size >= batch.sample_count:
            raise ValueError("The validation split leaves no training samples.")
        permutation = jr.permutation(jr.fold_in(key, 1), batch.sample_count)
        validation = batch.take_samples(permutation[:validation_size])
        training = batch.take_samples(permutation[validation_size:])
        active = (
            jnp.zeros((batch.feature_count,), dtype=bool)
            if self.direction == "forward"
            else jnp.ones((batch.feature_count,), dtype=bool)
        )
        steps = (
            self.num_features
            if self.direction == "forward"
            else batch.feature_count - self.num_features
        )
        statuses = []
        last_scores = jnp.zeros((batch.feature_count,))
        for step in range(steps):
            candidate_scores = []
            candidate_status = []
            capacity = (
                step + 1
                if self.direction == "forward"
                else batch.feature_count - step - 1
            )
            for feature in range(batch.feature_count):
                bit = jax.nn.one_hot(feature, batch.feature_count, dtype=bool)
                candidate = active | bit if self.direction == "forward" else active & ~bit
                indices = _active_indices(candidate, capacity)
                training_candidate = _take_features(training, indices)
                validation_candidate = _take_features(validation, indices)
                result = self.estimator.fit_batch(
                    training_candidate,
                    key=jr.fold_in(key, 1000 + step * batch.feature_count + feature),
                )
                score = jnp.asarray(
                    self.scorer(result.as_trainable(), validation_candidate)
                )
                eligible = (
                    ~active[feature] if self.direction == "forward" else active[feature]
                )
                candidate_scores.append(jnp.where(eligible, score, -jnp.inf))
                candidate_status.append(result.status)
            last_scores = jnp.stack(candidate_scores)
            statuses.extend(candidate_status)
            chosen = jnp.argmax(last_scores)
            chosen_bit = jax.nn.one_hot(chosen, batch.feature_count, dtype=bool)
            active = (
                active | chosen_bit
                if self.direction == "forward"
                else active & ~chosen_bit
            )
        indices = jnp.nonzero(active, size=self.num_features, fill_value=0)[0]
        return _exact_result(
            ExactSelection(
                indices, jnp.ones((self.num_features,), dtype=bool), last_scores
            ),
            batch.feature_count,
            method=f"sequential-{self.direction}",
            iterations=steps,
            estimator_status=jnp.stack(statuses)
            if statuses
            else jnp.empty((0,), dtype=jnp.int32),
        )


class ModelBasedSelectionRecipe(AbstractRecipe):
    estimator: AbstractRecipe
    threshold: float = eqx.field(static=True)
    max_features: int | None = eqx.field(static=True)
    importance_getter: Callable[[AbstractArrayModel], Array] = eqx.field(static=True)

    def __init__(
        self,
        estimator: AbstractRecipe,
        /,
        *,
        threshold: float = 0.0,
        max_features: int | None = None,
        importance_getter: Callable[[AbstractArrayModel], Array],
    ):
        if not isinstance(estimator, AbstractRecipe):
            raise TypeError("estimator must be an AbstractRecipe.")
        if max_features is not None and int(max_features) <= 0:
            raise ValueError("max_features must be positive.")
        if not callable(importance_getter):
            raise TypeError("importance_getter must be callable.")
        self.estimator = estimator
        self.threshold = float(threshold)
        self.max_features = None if max_features is None else int(max_features)
        self.importance_getter = importance_getter

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        result = self.estimator.fit_batch(batch, key=key)
        scores = _importance(
            result.as_trainable(), batch.feature_count, self.importance_getter
        )
        capacity = (
            batch.feature_count
            if self.max_features is None
            else min(self.max_features, batch.feature_count)
        )
        return _exact_result(
            _selection(scores, scores > self.threshold, capacity),
            batch.feature_count,
            method="model-based-selection",
            iterations=1,
            estimator_status=result.status[None, ...],
        )


class ContinuousSparseGateRecipe(AbstractRecipe):
    temperature: float = eqx.field(static=True)
    sparsity: float = eqx.field(static=True)
    scorer: Callable[[MLBatch], Array] = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        temperature: float = 0.1,
        sparsity: float = 0.5,
        scorer: Callable[[MLBatch], Array] | None = None,
    ):
        if float(temperature) <= 0.0 or not 0.0 <= float(sparsity) <= 1.0:
            raise ValueError("temperature must be positive and sparsity in [0, 1].")
        self.temperature = float(temperature)
        self.sparsity = float(sparsity)
        self.scorer = _correlation_scores if scorer is None else scorer

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        scores = jnp.asarray(self.scorer(batch))
        if scores.shape != (batch.feature_count,):
            raise ValueError("scorer must return one score per feature.")
        if jnp.issubdtype(scores.dtype, jnp.complexfloating):
            raise TypeError("Continuous feature gates require real-valued scores.")
        finite_scores = jnp.where(jnp.isfinite(scores), scores, 0.0)
        minimum = jnp.min(finite_scores)
        maximum = jnp.max(finite_scores)
        normalised = (finite_scores - minimum) / jnp.maximum(
            maximum - minimum, jnp.finfo(finite_scores.dtype).eps
        )
        gates = jax.nn.sigmoid((normalised - self.sparsity) / self.temperature)
        valid = jnp.all(jnp.isfinite(gates))
        status = jnp.where(valid, ML_SUCCESS, ML_NONFINITE)
        diagnostics = FeatureSelectionDiagnostics(
            relaxed_gates=gates,
            valid=valid,
            status=status,
            method="continuous-sparse-gates",
        )
        return FitResult(
            ContinuousFeatureGateModel(gates),
            diagnostics,
            valid=valid,
            status=status,
            method="continuous-sparse-gates",
            gradient_contract=GradientContract(
                fit_features="smooth",
                fit_targets="smooth",
                fit_hyperparameters="smooth",
                fit_mode="relaxed",
                conditions=(
                    "The configured score function must itself be differentiable.",
                ),
            ),
        )


__all__ = [
    "ContinuousFeatureGateModel",
    "ContinuousSparseGateRecipe",
    "ExactFeatureSelectorModel",
    "ExactSelection",
    "FeatureSelectionDiagnostics",
    "ModelBasedSelectionRecipe",
    "MutualInformationFilterRecipe",
    "RecursiveFeatureEliminationRecipe",
    "ScoreFilterRecipe",
    "SequentialFeatureSelectionRecipe",
    "VarianceFilterRecipe",
]
