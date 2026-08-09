#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..._strict import StrictModule
from .._batch import MLBatch
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_NONFINITE,
    ML_SUCCESS,
)
from ._splits import (
    _require_key,
    _validate_split_result_for_batch,
    AbstractSplitPlan,
    FoldRecord,
    SplitPlanResult,
)


MetricPath = str | tuple[str, ...] | None


@runtime_checkable
class _ScoreResult(Protocol):
    value: Any
    valid: Any
    status: Any


@runtime_checkable
class _WeightedScoreResult(_ScoreResult, Protocol):
    effective_weight: Any


@runtime_checkable
class _Scorer(Protocol):
    def score(
        self,
        predictions: Any,
        targets: Any,
        *,
        sample_weight: Any,
        mask: Any,
    ) -> Any: ...


class ScoreRecord(StrictModule):
    """Normalized scalar or PyTree-valued output from a callable scorer."""

    value: Any
    valid: Any
    status: Any
    effective_weight: Any
    raw: Any

    def __init__(
        self,
        value: Any,
        /,
        *,
        valid: Any,
        status: Any,
        effective_weight: Any,
        raw: Any,
    ):
        self.value = value
        self.valid = valid
        self.status = status
        self.effective_weight = effective_weight
        self.raw = raw


def _broadcast_score_field(field: Any, value: Any, /, *, dtype: Any) -> Any:
    array = jnp.asarray(field, dtype=dtype)
    target = jnp.asarray(value)
    if array.ndim > target.ndim:
        raise ValueError("Scorer diagnostics cannot have more axes than the score value.")
    if array.ndim < target.ndim:
        array = jnp.reshape(array, array.shape + (1,) * (target.ndim - array.ndim))
    return jnp.broadcast_to(array, target.shape)


def _score_leaf(value: Any, raw: Any, /) -> ScoreRecord:
    array = jnp.asarray(value)
    if not (
        jnp.issubdtype(array.dtype, jnp.number) or jnp.issubdtype(array.dtype, jnp.bool_)
    ):
        raise TypeError("Scorer values must have numeric or boolean JAX dtypes.")
    valid = jnp.isfinite(array)
    return ScoreRecord(
        array,
        valid=valid,
        status=jnp.where(valid, ML_SUCCESS, ML_NONFINITE).astype(jnp.int32),
        effective_weight=jnp.ones_like(array, dtype=jnp.result_type(array.real, float)),
        raw=raw,
    )


def _normalize_score(result: Any, /) -> ScoreRecord:
    if isinstance(result, _ScoreResult):
        value = jnp.asarray(result.value)
        valid = _broadcast_score_field(result.valid, value, dtype=bool)
        status = _broadcast_score_field(result.status, value, dtype=jnp.int32)
        weight = (
            result.effective_weight
            if isinstance(result, _WeightedScoreResult)
            else jnp.ones_like(value, dtype=float)
        )
        effective_weight = _broadcast_score_field(
            weight,
            value,
            dtype=None,
        )
        if not jnp.issubdtype(effective_weight.dtype, jnp.number) or jnp.issubdtype(
            effective_weight.dtype, jnp.complexfloating
        ):
            raise TypeError("Scorer effective_weight must be real and numeric.")
        valid = (
            valid
            & jnp.isfinite(value)
            & jnp.isfinite(effective_weight)
            & (effective_weight >= 0)
        )
        return ScoreRecord(
            value,
            valid=valid,
            status=jnp.where(valid, status, jnp.maximum(status, ML_NONFINITE)),
            effective_weight=effective_weight,
            raw=result,
        )
    if isinstance(result, Mapping):
        if not result:
            raise ValueError("A structured scorer cannot return an empty mapping.")
        normalized = {
            str(name): _normalize_score(value) for name, value in result.items()
        }
        return ScoreRecord(
            {name: item.value for name, item in normalized.items()},
            valid={name: item.valid for name, item in normalized.items()},
            status={name: item.status for name, item in normalized.items()},
            effective_weight={
                name: item.effective_weight for name, item in normalized.items()
            },
            raw=result,
        )
    if isinstance(result, (tuple, list)):
        if not result:
            raise ValueError("A structured scorer cannot return an empty sequence.")
        normalized = tuple(_normalize_score(value) for value in result)
        return ScoreRecord(
            tuple(item.value for item in normalized),
            valid=tuple(item.valid for item in normalized),
            status=tuple(item.status for item in normalized),
            effective_weight=tuple(item.effective_weight for item in normalized),
            raw=tuple(result),
        )
    return _score_leaf(result, result)


def _validation_mask(batch: MLBatch, /) -> Any:
    if batch.target_mask is None:
        return batch.sample_mask
    extra = batch.target_mask.ndim - batch.sample_mask.ndim
    sample_mask = jnp.reshape(
        batch.sample_mask, batch.sample_mask.shape + (1,) * max(0, extra)
    )
    return batch.target_mask & sample_mask


def _call_scorer(scorer: Any, predictions: Any, batch: MLBatch, /) -> ScoreRecord:
    targets = batch.require_targets()
    if isinstance(scorer, _Scorer):
        result = scorer.score(
            predictions,
            targets,
            sample_weight=batch.sample_weight,
            mask=_validation_mask(batch),
        )
    elif callable(scorer):
        result = scorer(
            predictions,
            targets,
            sample_weight=batch.sample_weight,
            mask=_validation_mask(batch),
        )
    else:
        raise TypeError("scorer must be callable or expose a callable score method.")
    return _normalize_score(result)


def _predict(fit_result: FitResult, batch: MLBatch, /, *, key: Any) -> Any:
    model = fit_result.as_trainable()
    binding = model.input_binding()
    if binding.batch_mode == "axis":
        raise NotImplementedError(
            "Axis-bound models cannot be scored from an MLBatch without a named-axis domain."
        )
    return binding.call(
        model,
        batch.features,
        key=key,
        iter_=None,
        kwargs={},
    )


class FoldEvaluation(StrictModule):
    """One independently fitted fold, including its complete fit diagnostics."""

    fold: FoldRecord
    fit_result: FitResult
    scorer_result: ScoreRecord
    predictions: Any
    fit_key: Any
    prediction_key: Any

    def __init__(
        self,
        fold: FoldRecord,
        fit_result: FitResult,
        scorer_result: ScoreRecord,
        predictions: Any,
        /,
        *,
        fit_key: Any,
        prediction_key: Any,
    ):
        self.fold = fold
        self.fit_result = fit_result
        self.scorer_result = scorer_result
        self.predictions = predictions
        self.fit_key = fit_key
        self.prediction_key = prediction_key


def _stack_tree(values: tuple[Any, ...], /) -> Any:
    return jax.tree_util.tree_map(lambda *items: jnp.stack(items, axis=0), *values)


def _aggregate_scores(folds: tuple[FoldEvaluation, ...], /) -> ScoreRecord:
    values = _stack_tree(tuple(fold.scorer_result.value for fold in folds))
    valid = _stack_tree(tuple(fold.scorer_result.valid for fold in folds))
    statuses = _stack_tree(tuple(fold.scorer_result.status for fold in folds))
    effective_weight = _stack_tree(
        tuple(fold.scorer_result.effective_weight for fold in folds)
    )

    def aggregate(value, valid_, weight):
        weight_ = jnp.asarray(weight)
        valid_ = jnp.asarray(valid_, dtype=bool)
        usable = valid_ & jnp.isfinite(value) & jnp.isfinite(weight_) & (weight_ >= 0)
        selected_weight = jnp.where(usable, weight_, 0)
        denominator = jnp.sum(selected_weight, axis=0)
        numerator = jnp.sum(jnp.where(usable, value * selected_weight, 0), axis=0)
        mean = jnp.where(denominator > 0, numerator / denominator, jnp.nan)
        return mean

    mean = jax.tree_util.tree_map(aggregate, values, valid, effective_weight)
    aggregate_valid = jax.tree_util.tree_map(
        lambda mean_, valid_: (
            jnp.isfinite(mean_) & jnp.all(jnp.asarray(valid_, dtype=bool), axis=0)
        ),
        mean,
        valid,
    )
    aggregate_status = jax.tree_util.tree_map(
        lambda status, valid_: jnp.where(
            valid_,
            jnp.max(jnp.asarray(status), axis=0),
            jnp.maximum(jnp.max(jnp.asarray(status), axis=0), ML_NONFINITE),
        ),
        statuses,
        aggregate_valid,
    )
    total_weight = jax.tree_util.tree_map(
        lambda valid_, weight: jnp.sum(
            jnp.where(
                jnp.asarray(valid_, dtype=bool),
                weight,
                0,
            ),
            axis=0,
        ),
        valid,
        effective_weight,
    )
    return ScoreRecord(
        mean,
        valid=aggregate_valid,
        status=aggregate_status,
        effective_weight=total_weight,
        raw=tuple(fold.scorer_result.raw for fold in folds),
    )


_LEVEL_ORDER = {"none": 0, "conditional": 1, "almost-everywhere": 2, "smooth": 3}


def _weakest_level(
    contracts: tuple[GradientContract, ...],
    select: Callable[[GradientContract], str],
    /,
) -> str:
    return min(
        (select(contract) for contract in contracts),
        key=lambda level: _LEVEL_ORDER[level],
    )


def _cross_validation_contract(folds: tuple[FoldEvaluation, ...], /) -> GradientContract:
    contracts = tuple(fold.fit_result.gradient_contract for fold in folds)
    modes = {contract.fit_mode for contract in contracts}
    return GradientContract(
        prediction_inputs=_weakest_level(
            contracts, lambda contract: contract.prediction_inputs
        ),
        prediction_parameters=_weakest_level(
            contracts, lambda contract: contract.prediction_parameters
        ),
        fit_features=_weakest_level(contracts, lambda contract: contract.fit_features),
        fit_targets=_weakest_level(contracts, lambda contract: contract.fit_targets),
        fit_weights=_weakest_level(contracts, lambda contract: contract.fit_weights),
        fit_hyperparameters=_weakest_level(
            contracts, lambda contract: contract.fit_hyperparameters
        ),
        fit_mode=next(iter(modes)) if len(modes) == 1 else "stopped",
        nondifferentiable_outputs=(
            "fold_indices",
            "valid",
            "status",
        ),
        conditions=(
            "Fold indices are fixed and gradients never pass through split membership.",
            "Score gradients additionally require the supplied scorer to be differentiable.",
            "Every batch-dependent transform must be an unfitted recipe component and is refit per fold.",
        ),
    )


class CrossValidationResult(StrictModule):
    """Structured results from leakage-safe, independently refitted folds."""

    folds: tuple[FoldEvaluation, ...]
    split_result: SplitPlanResult
    aggregate_score: ScoreRecord
    valid: Any
    status: Any
    key: Any
    gradient_contract: GradientContract
    method: str = eqx.field(static=True)

    def __init__(
        self,
        folds: tuple[FoldEvaluation, ...],
        split_result: SplitPlanResult,
        aggregate_score: ScoreRecord,
        /,
        *,
        key: Any,
        gradient_contract: GradientContract,
    ):
        fit_valid = jnp.all(
            jnp.stack([jnp.all(jnp.asarray(fold.fit_result.valid)) for fold in folds])
        )
        score_valid = jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.asarray(leaf))
                    for leaf in jax.tree_util.tree_leaves(aggregate_score.valid)
                ]
            )
        )
        statuses = [
            jnp.max(jnp.asarray(fold.fit_result.status, dtype=jnp.int32))
            for fold in folds
        ]
        statuses.extend(
            jnp.max(jnp.asarray(leaf, dtype=jnp.int32))
            for leaf in jax.tree_util.tree_leaves(aggregate_score.status)
        )
        self.folds = tuple(folds)
        self.split_result = split_result
        self.aggregate_score = aggregate_score
        self.valid = fit_valid & score_valid
        self.status = jnp.max(jnp.stack(statuses))
        self.key = _require_key(key)
        self.gradient_contract = gradient_contract
        self.method = "cross_validation"


def _cross_validate_materialized(
    recipe: AbstractRecipe,
    batch: MLBatch,
    split_result: SplitPlanResult,
    scorer: Any,
    /,
    *,
    key: Any,
) -> CrossValidationResult:
    if not isinstance(recipe, AbstractRecipe):
        raise TypeError(
            "cross-validation requires an unfitted AbstractRecipe; fitted models and "
            "preprocessed estimators are rejected to prevent leakage."
        )
    if not isinstance(batch, MLBatch):
        raise TypeError("batch must be an MLBatch.")
    if not isinstance(split_result, SplitPlanResult):
        raise TypeError("split_result must be a SplitPlanResult.")
    key = _require_key(key)
    evaluations: list[FoldEvaluation] = []
    for position, fold in enumerate(split_result.folds):
        fold_key = jr.fold_in(key, position)
        fit_key, prediction_key = jr.split(fold_key)
        train_batch = batch.take_samples(fold.train_indices)
        validation_batch = batch.take_samples(fold.validation_indices)
        fit_result = recipe.fit_batch(train_batch, key=fit_key)
        if not isinstance(fit_result, FitResult):
            raise TypeError("Recipe.fit_batch must return a FitResult.")
        predictions = _predict(fit_result, validation_batch, key=prediction_key)
        score = _call_scorer(scorer, predictions, validation_batch)
        evaluations.append(
            FoldEvaluation(
                fold,
                fit_result,
                score,
                predictions,
                fit_key=fit_key,
                prediction_key=prediction_key,
            )
        )
    folds = tuple(evaluations)
    aggregate = _aggregate_scores(folds)
    return CrossValidationResult(
        folds,
        split_result,
        aggregate,
        key=key,
        gradient_contract=_cross_validation_contract(folds),
    )


def cross_validate(
    recipe: AbstractRecipe,
    batch: MLBatch,
    split_plan: AbstractSplitPlan | SplitPlanResult,
    scorer: Any,
    /,
    *,
    key: Any,
) -> CrossValidationResult:
    """Refit ``recipe`` from scratch on every training fold and score holdouts."""
    key = _require_key(key)
    if isinstance(split_plan, AbstractSplitPlan):
        split_result = split_plan.split(batch, key=jr.fold_in(key, 0))
        fit_key = jr.fold_in(key, 1)
    elif isinstance(split_plan, SplitPlanResult):
        split_result = split_plan
        fit_key = key
    else:
        raise TypeError("split_plan must be an AbstractSplitPlan or SplitPlanResult.")
    _validate_split_result_for_batch(split_result, batch)
    return _cross_validate_materialized(recipe, batch, split_result, scorer, key=fit_key)


class CrossValidator(StrictModule):
    """Immutable functional cross-validation configuration."""

    split_plan: AbstractSplitPlan
    scorer: Any

    def __init__(self, split_plan: AbstractSplitPlan, scorer: Any, /):
        if not isinstance(split_plan, AbstractSplitPlan):
            raise TypeError("split_plan must be an AbstractSplitPlan.")
        if not callable(scorer) and not isinstance(scorer, _Scorer):
            raise TypeError("scorer must be callable or expose a score method.")
        self.split_plan = split_plan
        self.scorer = scorer

    def evaluate(
        self, recipe: AbstractRecipe, batch: MLBatch, /, *, key: Any
    ) -> CrossValidationResult:
        return cross_validate(recipe, batch, self.split_plan, self.scorer, key=key)


def select_metric(value: Any, path: MetricPath, /) -> Any:
    """Select one objective leaf from a structured scorer value."""
    if path is None:
        current = value
        while isinstance(current, Mapping):
            if len(current) != 1:
                raise ValueError(
                    "A multi-metric scorer requires an explicit primary_metric path."
                )
            current = next(iter(current.values()))
        if isinstance(current, tuple):
            if len(current) != 1:
                raise ValueError(
                    "A multi-output scorer requires an explicit primary_metric path."
                )
            current = current[0]
        return current
    names = (path,) if isinstance(path, str) else tuple(path)
    current = value
    for name in names:
        if not isinstance(current, Mapping) or name not in current:
            raise KeyError(f"Unknown scorer metric path component {name!r}.")
        current = current[name]
    return current


__all__ = [
    "CrossValidationResult",
    "CrossValidator",
    "FoldEvaluation",
    "MetricPath",
    "ScoreRecord",
    "cross_validate",
    "select_metric",
]
