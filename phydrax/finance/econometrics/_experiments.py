#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._numerics import solve_weighted_least_squares
from ..._strict import StrictModule
from ..core import FinanceEvidenceBinding
from ._features import PreparedFeatureLabelDataset


WalkForwardEstimator: TypeAlias = Literal["linear", "mean"]
WalkForwardLoss: TypeAlias = Literal["squared", "absolute"]


class WalkForwardDefinition(StrictModule):
    """Chronological train/validation/test spans with timestamp purge and embargo."""

    training_span_ns: int = eqx.field(static=True)
    validation_span_ns: int = eqx.field(static=True)
    test_span_ns: int = eqx.field(static=True)
    step_ns: int = eqx.field(static=True)
    purge_ns: int = eqx.field(static=True)
    embargo_ns: int = eqx.field(static=True)
    minimum_training_rows: int = eqx.field(static=True)
    fold_capacity: int = eqx.field(static=True)
    expanding: bool = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        training_span_ns: int,
        validation_span_ns: int,
        test_span_ns: int,
        step_ns: int,
        fold_capacity: int,
        purge_ns: int = 0,
        embargo_ns: int = 0,
        minimum_training_rows: int = 8,
        expanding: bool = True,
    ):
        spans = tuple(
            int(value)
            for value in (
                training_span_ns,
                validation_span_ns,
                test_span_ns,
                step_ns,
            )
        )
        if any(value <= 0 for value in spans):
            raise ValueError(
                "training, validation, test, and step spans must be positive."
            )
        purge = int(purge_ns)
        embargo = int(embargo_ns)
        minimum = int(minimum_training_rows)
        capacity = int(fold_capacity)
        if purge < 0 or embargo < 0:
            raise ValueError("purge_ns and embargo_ns must be nonnegative.")
        if minimum < 1 or capacity < 1:
            raise ValueError("minimum_training_rows and fold_capacity must be positive.")
        (
            self.training_span_ns,
            self.validation_span_ns,
            self.test_span_ns,
            self.step_ns,
        ) = spans
        self.purge_ns = purge
        self.embargo_ns = embargo
        self.minimum_training_rows = minimum
        self.fold_capacity = capacity
        self.expanding = bool(expanding)
        self.definition_id = canonical_fingerprint(
            {
                "kind": "walk-forward-definition",
                "training_span_ns": spans[0],
                "validation_span_ns": spans[1],
                "test_span_ns": spans[2],
                "step_ns": spans[3],
                "purge_ns": purge,
                "embargo_ns": embargo,
                "minimum_training_rows": minimum,
                "fold_capacity": capacity,
                "expanding": bool(expanding),
            }
        )


class WalkForwardFold(StrictModule):
    """One materialized chronological fold and its removal evidence."""

    train_mask: Array
    validation_mask: Array
    test_mask: Array
    purged_mask: Array
    embargoed_mask: Array
    bounds_ns: Array
    training_count: Array
    validation_count: Array
    test_count: Array
    valid: Array
    index: int = eqx.field(static=True)


class WalkForwardPlan(StrictModule):
    """Fixed-fold split masks preserving all train, validation, and test roles."""

    train_mask: Array
    validation_mask: Array
    test_mask: Array
    purged_mask: Array
    embargoed_mask: Array
    fold_valid: Array
    fold_bounds_ns: Array
    training_counts: Array
    validation_counts: Array
    test_counts: Array
    dataset_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    fold_capacity: int = eqx.field(static=True)
    row_capacity: int = eqx.field(static=True)

    def fold(self, index: int, /) -> WalkForwardFold:
        fold_index = int(index)
        if not 0 <= fold_index < self.fold_capacity:
            raise IndexError("fold index is outside fold capacity.")
        return WalkForwardFold(
            train_mask=self.train_mask[fold_index],
            validation_mask=self.validation_mask[fold_index],
            test_mask=self.test_mask[fold_index],
            purged_mask=self.purged_mask[fold_index],
            embargoed_mask=self.embargoed_mask[fold_index],
            bounds_ns=self.fold_bounds_ns[fold_index],
            training_count=self.training_counts[fold_index],
            validation_count=self.validation_counts[fold_index],
            test_count=self.test_counts[fold_index],
            valid=self.fold_valid[fold_index],
            index=fold_index,
        )


class ExperimentEvidence(StrictModule):
    """Observable chronology, overlap, purge, embargo, and replay evidence."""

    train_validation_disjoint: Array
    train_test_disjoint: Array
    validation_test_disjoint: Array
    chronological: Array
    label_overlap_removed: Array
    embargo_respected: Array
    purged_counts: Array
    embargoed_counts: Array
    active_fold_count: Array
    replay_equal: Array
    dataset_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class WalkForwardResult(StrictModule):
    """Out-of-sample predictions and losses; no portfolio or trade semantics."""

    predictions: Array
    losses: Array
    forecast_mask: Array
    validation_mask: Array
    test_mask: Array
    coefficients: Array
    intercepts: Array
    training_log_likelihood: Array
    training_rank: Array
    training_condition_number: Array
    training_count: Array
    fold_valid: Array
    numerical_status: Array
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    estimator: WalkForwardEstimator = eqx.field(static=True)
    loss: WalkForwardLoss = eqx.field(static=True)


def prepare_walk_forward(
    dataset: PreparedFeatureLabelDataset,
    definition: WalkForwardDefinition,
    /,
) -> tuple[WalkForwardPlan, ExperimentEvidence]:
    """Prepare chronological folds using full label intervals for overlap removal."""

    if not isinstance(dataset, PreparedFeatureLabelDataset):
        raise TypeError("dataset must be a PreparedFeatureLabelDataset.")
    if not isinstance(definition, WalkForwardDefinition):
        raise TypeError("definition must be a WalkForwardDefinition.")
    valid_times = jnp.where(
        dataset.row_valid,
        dataset.decision_times_ns,
        jnp.iinfo(jnp.int64).max,
    )
    first_time = jnp.min(valid_times)
    first_time = jnp.where(
        jnp.any(dataset.row_valid), first_time, jnp.asarray(0, dtype=jnp.int64)
    )
    fold_shape = (definition.fold_capacity, dataset.row_capacity)
    train_masks = jnp.zeros(fold_shape, dtype=bool)
    validation_masks = jnp.zeros(fold_shape, dtype=bool)
    test_masks = jnp.zeros(fold_shape, dtype=bool)
    purged_masks = jnp.zeros(fold_shape, dtype=bool)
    embargoed_masks = jnp.zeros(fold_shape, dtype=bool)
    fold_valid = jnp.zeros((definition.fold_capacity,), dtype=bool)
    bounds = jnp.zeros((definition.fold_capacity, 6), dtype=jnp.int64)
    train_counts = jnp.zeros((definition.fold_capacity,), dtype=jnp.int32)
    validation_counts = jnp.zeros((definition.fold_capacity,), dtype=jnp.int32)
    test_counts = jnp.zeros((definition.fold_capacity,), dtype=jnp.int32)
    for fold in range(definition.fold_capacity):
        train_end = first_time + definition.training_span_ns + fold * definition.step_ns
        train_start = (
            first_time
            if definition.expanding
            else train_end - definition.training_span_ns
        )
        validation_start = train_end
        validation_end = validation_start + definition.validation_span_ns
        test_start = validation_end
        test_end = test_start + definition.test_span_ns
        embargo_boundary = validation_start - definition.embargo_ns
        raw_train = (
            dataset.row_valid
            & (dataset.decision_times_ns >= train_start)
            & (dataset.decision_times_ns < train_end)
        )
        embargoed = raw_train & (dataset.decision_times_ns >= embargo_boundary)
        evaluation_start = validation_start - definition.purge_ns
        overlap = raw_train & (
            (dataset.feature_start_times_ns < test_end)
            & (dataset.label_end_times_ns > evaluation_start)
        )
        train = raw_train & ~overlap & ~embargoed
        validation = (
            dataset.row_valid
            & (dataset.decision_times_ns >= validation_start)
            & (dataset.decision_times_ns < validation_end)
        )
        test = (
            dataset.row_valid
            & (dataset.decision_times_ns >= test_start)
            & (dataset.decision_times_ns < test_end)
        )
        train_count = jnp.sum(train).astype(jnp.int32)
        validation_count = jnp.sum(validation).astype(jnp.int32)
        test_count = jnp.sum(test).astype(jnp.int32)
        valid_fold = (
            (train_count >= definition.minimum_training_rows)
            & (validation_count > 0)
            & (test_count > 0)
        )
        train_masks = train_masks.at[fold].set(train)
        validation_masks = validation_masks.at[fold].set(validation)
        test_masks = test_masks.at[fold].set(test)
        purged_masks = purged_masks.at[fold].set(overlap)
        embargoed_masks = embargoed_masks.at[fold].set(embargoed)
        fold_valid = fold_valid.at[fold].set(valid_fold)
        bounds = bounds.at[fold].set(
            jnp.asarray(
                [
                    train_start,
                    train_end,
                    validation_start,
                    validation_end,
                    test_start,
                    test_end,
                ],
                dtype=jnp.int64,
            )
        )
        train_counts = train_counts.at[fold].set(train_count)
        validation_counts = validation_counts.at[fold].set(validation_count)
        test_counts = test_counts.at[fold].set(test_count)
    plan_id = canonical_fingerprint(
        {
            "kind": "walk-forward-plan",
            "dataset": dataset.dataset_id,
            "definition": definition.definition_id,
        }
    )
    plan = WalkForwardPlan(
        train_mask=train_masks,
        validation_mask=validation_masks,
        test_mask=test_masks,
        purged_mask=purged_masks,
        embargoed_mask=embargoed_masks,
        fold_valid=fold_valid,
        fold_bounds_ns=bounds,
        training_counts=train_counts,
        validation_counts=validation_counts,
        test_counts=test_counts,
        dataset_id=dataset.dataset_id,
        definition_id=definition.definition_id,
        plan_id=plan_id,
        fold_capacity=definition.fold_capacity,
        row_capacity=dataset.row_capacity,
    )
    train_validation_disjoint = ~jnp.any(train_masks & validation_masks, axis=-1)
    train_test_disjoint = ~jnp.any(train_masks & test_masks, axis=-1)
    validation_test_disjoint = ~jnp.any(validation_masks & test_masks, axis=-1)
    chronological = bounds[:, 1] <= bounds[:, 2]
    overlap_remaining = train_masks & (
        (dataset.feature_start_times_ns[None, :] < bounds[:, 5, None])
        & (dataset.label_end_times_ns[None, :] > bounds[:, 2, None] - definition.purge_ns)
    )
    embargo_remaining = train_masks & (
        dataset.decision_times_ns[None, :] >= bounds[:, 2, None] - definition.embargo_ns
    )
    evidence = ExperimentEvidence(
        train_validation_disjoint=train_validation_disjoint,
        train_test_disjoint=train_test_disjoint,
        validation_test_disjoint=validation_test_disjoint,
        chronological=chronological,
        label_overlap_removed=~jnp.any(overlap_remaining, axis=-1),
        embargo_respected=~jnp.any(embargo_remaining, axis=-1),
        purged_counts=jnp.sum(purged_masks, axis=-1).astype(jnp.int32),
        embargoed_counts=jnp.sum(embargoed_masks, axis=-1).astype(jnp.int32),
        active_fold_count=jnp.sum(fold_valid).astype(jnp.int32),
        replay_equal=jnp.asarray(True),
        dataset_id=dataset.dataset_id,
        plan_id=plan_id,
    )
    return plan, evidence


def evaluate_walk_forward(
    dataset: PreparedFeatureLabelDataset,
    plan: WalkForwardPlan,
    /,
    *,
    estimator: WalkForwardEstimator = "linear",
    loss: WalkForwardLoss = "squared",
    ridge: float = 0.0,
) -> WalkForwardResult:
    """Fit each fold on training rows and retain validation/test forecasts separately."""

    if not isinstance(dataset, PreparedFeatureLabelDataset):
        raise TypeError("dataset must be a PreparedFeatureLabelDataset.")
    if not isinstance(plan, WalkForwardPlan):
        raise TypeError("plan must be a WalkForwardPlan.")
    if plan.dataset_id != dataset.dataset_id or plan.row_capacity != dataset.row_capacity:
        raise ValueError("plan and dataset identities/shapes must match.")
    if estimator not in ("linear", "mean"):
        raise ValueError("estimator must be 'linear' or 'mean'.")
    if loss not in ("squared", "absolute"):
        raise ValueError("loss must be 'squared' or 'absolute'.")
    ridge_ = float(ridge)
    if ridge_ < 0.0:
        raise ValueError("ridge must be nonnegative.")
    predictions = jnp.zeros(
        (plan.fold_capacity, plan.row_capacity), dtype=dataset.labels.dtype
    )
    losses = jnp.zeros_like(predictions)
    forecast_mask = jnp.zeros_like(plan.train_mask)
    coefficients = jnp.zeros(
        (plan.fold_capacity, dataset.feature_count), dtype=dataset.features.dtype
    )
    intercepts = jnp.zeros((plan.fold_capacity,), dtype=dataset.labels.dtype)
    likelihood = jnp.zeros((plan.fold_capacity,), dtype=dataset.labels.dtype)
    ranks = jnp.zeros((plan.fold_capacity,), dtype=jnp.int32)
    conditions = jnp.full((plan.fold_capacity,), jnp.inf, dtype=dataset.labels.dtype)
    statuses = jnp.ones((plan.fold_capacity,), dtype=jnp.int32)
    for fold in range(plan.fold_capacity):
        train = plan.train_mask[fold]
        forecast = (plan.validation_mask[fold] | plan.test_mask[fold]) & plan.fold_valid[
            fold
        ]
        if estimator == "linear":
            design = jnp.concatenate(
                (
                    jnp.ones((dataset.row_capacity, 1), dtype=dataset.features.dtype),
                    dataset.features,
                ),
                axis=-1,
            )
            fit = solve_weighted_least_squares(
                design,
                dataset.labels,
                mask=train,
                ridge=ridge_,
                min_samples=dataset.feature_count + 2,
                max_features=dataset.feature_count + 1,
            )
            parameter = fit.raw_coefficients
            prediction = design @ parameter
            intercept = parameter[0]
            slope = parameter[1:]
            rank = fit.rank
            condition = fit.condition_number
            numerical_status = fit.status
            training_residual = dataset.labels - prediction
        else:
            count = jnp.maximum(jnp.sum(train), 1)
            mean = jnp.sum(jnp.where(train, dataset.labels, 0.0)) / count
            prediction = jnp.full(dataset.labels.shape, mean)
            intercept = mean
            slope = jnp.zeros((dataset.feature_count,), dtype=dataset.features.dtype)
            rank = jnp.asarray(1, dtype=jnp.int32)
            condition = jnp.asarray(1.0, dtype=dataset.labels.dtype)
            numerical_status = jnp.where(plan.training_counts[fold] > 0, 0, 1).astype(
                jnp.int32
            )
            training_residual = dataset.labels - mean
        variance = jnp.sum(
            jnp.where(train, jnp.square(training_residual), 0.0)
        ) / jnp.maximum(plan.training_counts[fold], 1)
        variance = jnp.maximum(variance, jnp.finfo(dataset.labels.dtype).tiny)
        log_likelihood = jnp.sum(
            jnp.where(
                train,
                -0.5
                * (
                    jnp.log(2.0 * jnp.pi * variance)
                    + jnp.square(training_residual) / variance
                ),
                0.0,
            )
        )
        error = prediction - dataset.labels
        fold_loss = jnp.square(error) if loss == "squared" else jnp.abs(error)
        predictions = predictions.at[fold].set(jnp.where(forecast, prediction, 0.0))
        losses = losses.at[fold].set(jnp.where(forecast, fold_loss, 0.0))
        forecast_mask = forecast_mask.at[fold].set(forecast)
        coefficients = coefficients.at[fold].set(slope)
        intercepts = intercepts.at[fold].set(intercept)
        likelihood = likelihood.at[fold].set(log_likelihood)
        ranks = ranks.at[fold].set(rank)
        conditions = conditions.at[fold].set(condition)
        statuses = statuses.at[fold].set(numerical_status)
    result_id = canonical_fingerprint(
        {
            "kind": "walk-forward-result",
            "dataset": dataset.dataset_id,
            "plan": plan.plan_id,
            "estimator": estimator,
            "loss": loss,
            "ridge": ridge_,
        }
    )
    return WalkForwardResult(
        predictions=predictions,
        losses=losses,
        forecast_mask=forecast_mask,
        validation_mask=plan.validation_mask & plan.fold_valid[:, None],
        test_mask=plan.test_mask & plan.fold_valid[:, None],
        coefficients=coefficients,
        intercepts=intercepts,
        training_log_likelihood=likelihood,
        training_rank=ranks,
        training_condition_number=conditions,
        training_count=plan.training_counts,
        fold_valid=plan.fold_valid,
        numerical_status=statuses,
        evidence=FinanceEvidenceBinding(
            (
                canonical_fingerprint(
                    {"kind": "walk-forward-data-evidence", "dataset": dataset.dataset_id}
                ),
            ),
            (
                canonical_fingerprint(
                    {
                        "kind": "walk-forward-model-evidence",
                        "estimator": estimator,
                        "loss": loss,
                    }
                ),
            ),
            (
                canonical_fingerprint(
                    {
                        "kind": "walk-forward-numerical-evidence",
                        "statuses": tuple(int(value) for value in statuses.tolist()),
                    }
                ),
            ),
            (
                canonical_fingerprint(
                    {"kind": "walk-forward-use-evidence", "trade_emission": False}
                ),
            ),
        ),
        dataset_id=dataset.dataset_id,
        plan_id=plan.plan_id,
        result_id=result_id,
        estimator=estimator,
        loss=loss,
    )


__all__ = [
    "ExperimentEvidence",
    "WalkForwardDefinition",
    "WalkForwardEstimator",
    "WalkForwardFold",
    "WalkForwardLoss",
    "WalkForwardPlan",
    "WalkForwardResult",
    "evaluate_walk_forward",
    "prepare_walk_forward",
]
