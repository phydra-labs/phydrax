#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ._base import (
    _broadcast_full,
    _nan_where_invalid,
    _normalize_axis,
    _prepare_pair,
    _prepare_values,
    _real_dtype,
    _reject_complex,
    _result,
    _weighted_mean,
    Average,
    METRIC_EMPTY,
    METRIC_INVALID_INPUT,
    METRIC_SINGLE_CLASS,
    METRIC_SUCCESS,
    METRIC_ZERO_DENOMINATOR,
    MetricResult,
)


class PrecisionRecallFScoreResult(StrictModule):
    """Precision, recall, F-score, support, and their shared edge state."""

    precision: Array
    recall: Array
    fscore: Array
    support: Array
    valid: Array
    status: Array
    effective_weight: Array
    average: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        precision: ArrayLike,
        recall: ArrayLike,
        fscore: ArrayLike,
        support: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        effective_weight: ArrayLike,
        average: Average,
    ):
        self.precision = jnp.asarray(precision)
        self.recall = jnp.asarray(recall)
        self.fscore = jnp.asarray(fscore)
        self.support = jnp.asarray(support)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.effective_weight = jnp.asarray(effective_weight)
        self.average = average


def _require_integer_labels(labels: Array, metric: str) -> None:
    _reject_complex(labels, metric=metric)
    if not jnp.issubdtype(labels.dtype, jnp.integer) and labels.dtype != jnp.bool_:
        raise TypeError(f"{metric} requires integer class labels.")


def _probability_inputs(
    y_true: ArrayLike,
    values: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    metric: str,
    from_logits: bool,
) -> tuple[Array, Array, Array, Array, Array, int]:
    labels = jnp.asarray(y_true)
    raw = jnp.asarray(values)
    _require_integer_labels(labels, metric)
    _reject_complex(raw, metric=metric)
    if labels.ndim == 0:
        raise ValueError(f"{metric} requires a sample axis.")
    axis = _normalize_axis(sample_axis, labels.ndim)
    if axis != labels.ndim - 1:
        raise ValueError("Classification labels must have case_shape + (sample,) axes.")
    if raw.ndim != labels.ndim + 1 or raw.shape[:-1] != labels.shape:
        raise ValueError(f"{metric} requires values with shape y_true.shape + (class,).")
    classes = int(raw.shape[-1])
    if classes < 2:
        raise ValueError(f"{metric} requires at least two classes.")
    safe_labels, weights, active, invalid, _ = _prepare_values(
        labels,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=axis,
        metric=metric,
        allow_complex=False,
    )
    included = _broadcast_full(
        mask,
        tuple(int(size) for size in labels.shape),
        dtype=bool,
        fill=True,
        name="mask",
    )
    label_in_range = (safe_labels >= 0) & (safe_labels < classes)
    raw_finite = jnp.all(jnp.isfinite(raw), axis=-1)
    if from_logits:
        probabilities = jax.nn.softmax(raw, axis=-1)
        values_valid = raw_finite
    else:
        dtype = _real_dtype(raw)
        tolerance = jnp.asarray(32 * classes, dtype=dtype) * jnp.finfo(dtype).eps
        total = jnp.sum(raw, axis=-1)
        values_valid = (
            raw_finite
            & jnp.all(raw >= 0.0, axis=-1)
            & jnp.all(raw <= 1.0, axis=-1)
            & (jnp.abs(total - 1.0) <= tolerance)
        )
        probabilities = raw
    invalid = invalid | jnp.any(included & ~(label_in_range & values_valid), axis=axis)
    active = active & label_in_range & values_valid
    probabilities = jnp.where(active[..., None], probabilities, 0.0)
    return safe_labels.astype(jnp.int32), probabilities, weights, active, invalid, classes


def _hard_confusion_components(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    num_classes: int,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    metric: str,
) -> tuple[Array, Array, Array, Array]:
    classes = int(num_classes)
    if classes < 2:
        raise ValueError("num_classes must be at least two.")
    true_raw = jnp.asarray(y_true)
    pred_raw = jnp.asarray(y_pred)
    _require_integer_labels(true_raw, metric)
    _require_integer_labels(pred_raw, metric)
    true, pred, weights, active, invalid, axis = _prepare_pair(
        true_raw,
        pred_raw,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric=metric,
        allow_complex=False,
    )
    if axis != true.ndim - 1:
        raise ValueError("Classification labels must have case_shape + (sample,) axes.")
    in_range = (true >= 0) & (true < classes) & (pred >= 0) & (pred < classes)
    invalid = invalid | jnp.any(active & ~in_range, axis=axis)
    active = active & in_range
    mass = jnp.sum(jnp.where(active, weights, 0.0), axis=axis)
    true_hot = jax.nn.one_hot(true.astype(jnp.int32), classes, dtype=weights.dtype)
    pred_hot = jax.nn.one_hot(pred.astype(jnp.int32), classes, dtype=weights.dtype)
    confusion = ein.contract(
        "...ni,...n,...nj->...ij",
        true_hot,
        jnp.where(active, weights, 0.0),
        pred_hot,
    )
    return confusion, mass, invalid, active


def accuracy_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Exact weighted hard-label accuracy (nondifferentiable in labels)."""
    true_raw = jnp.asarray(y_true)
    pred_raw = jnp.asarray(y_pred)
    _require_integer_labels(true_raw, "accuracy_score")
    _require_integer_labels(pred_raw, "accuracy_score")
    true, pred, weights, active, invalid, axis = _prepare_pair(
        true_raw,
        pred_raw,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="accuracy_score",
        allow_complex=False,
    )
    if axis != true.ndim - 1:
        raise ValueError("Classification labels must have case_shape + (sample,) axes.")
    value, mass = _weighted_mean(
        (true == pred).astype(weights.dtype), weights, active, axis
    )
    return _result(value, invalid=invalid, effective_weight=mass)


def smooth_accuracy_score(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> MetricResult:
    """Expected accuracy under predicted class probabilities; smooth in predictions."""
    labels, probability, weights, active, invalid, classes = _probability_inputs(
        y_true,
        y_probability,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_accuracy_score",
        from_logits=from_logits,
    )
    selected = jnp.sum(
        probability * jax.nn.one_hot(labels, classes, dtype=probability.dtype), axis=-1
    )
    value, mass = _weighted_mean(selected, weights, active, labels.ndim - 1)
    return _result(value, invalid=invalid, effective_weight=mass)


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    num_classes: int,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    normalize: str | None = None,
) -> MetricResult:
    """Exact hard confusion matrix with rows=true and columns=predicted."""
    confusion, mass, invalid, _ = _hard_confusion_components(
        y_true,
        y_pred,
        num_classes=num_classes,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="confusion_matrix",
    )
    if normalize is None:
        value = confusion
        undefined = jnp.zeros_like(mass, dtype=bool)
    elif normalize == "true":
        denominator = jnp.sum(confusion, axis=-1, keepdims=True)
        undefined = jnp.any(denominator <= 0.0, axis=(-2, -1))
        value = confusion / jnp.where(denominator > 0.0, denominator, 1.0)
    elif normalize == "pred":
        denominator = jnp.sum(confusion, axis=-2, keepdims=True)
        undefined = jnp.any(denominator <= 0.0, axis=(-2, -1))
        value = confusion / jnp.where(denominator > 0.0, denominator, 1.0)
    elif normalize == "all":
        denominator = mass[..., None, None]
        undefined = mass <= 0.0
        value = confusion / jnp.where(denominator > 0.0, denominator, 1.0)
    else:
        raise ValueError("normalize must be None, 'true', 'pred', or 'all'.")
    valid = ~(invalid | (mass <= 0.0) | undefined)
    status = jnp.where(
        invalid,
        METRIC_INVALID_INPUT,
        jnp.where(
            mass <= 0.0,
            METRIC_EMPTY,
            jnp.where(undefined, METRIC_ZERO_DENOMINATOR, METRIC_SUCCESS),
        ),
    )
    return MetricResult(
        _nan_where_invalid(value, valid),
        valid=valid,
        status=status,
        effective_weight=mass,
    )


def smooth_confusion_matrix(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
    normalize: str | None = None,
) -> MetricResult:
    """Expected confusion matrix, smooth in predicted probabilities."""
    labels, probability, weights, active, invalid, classes = _probability_inputs(
        y_true,
        y_probability,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_confusion_matrix",
        from_logits=from_logits,
    )
    true_hot = jax.nn.one_hot(labels, classes, dtype=probability.dtype)
    confusion = ein.contract(
        "...ni,...n,...nj->...ij",
        true_hot,
        jnp.where(active, weights, 0.0),
        probability,
    )
    mass = jnp.sum(jnp.where(active, weights, 0.0), axis=-1)
    if normalize is None:
        value = confusion
        undefined = jnp.zeros_like(mass, dtype=bool)
    elif normalize == "true":
        denominator = jnp.sum(confusion, axis=-1, keepdims=True)
        undefined = jnp.any(denominator <= 0.0, axis=(-2, -1))
        value = confusion / jnp.where(denominator > 0.0, denominator, 1.0)
    elif normalize == "pred":
        denominator = jnp.sum(confusion, axis=-2, keepdims=True)
        undefined = jnp.any(denominator <= 0.0, axis=(-2, -1))
        value = confusion / jnp.where(denominator > 0.0, denominator, 1.0)
    elif normalize == "all":
        denominator = mass[..., None, None]
        undefined = mass <= 0.0
        value = confusion / jnp.where(denominator > 0.0, denominator, 1.0)
    else:
        raise ValueError("normalize must be None, 'true', 'pred', or 'all'.")
    valid = ~(invalid | (mass <= 0.0) | undefined)
    status = jnp.where(
        invalid,
        METRIC_INVALID_INPUT,
        jnp.where(
            mass <= 0.0,
            METRIC_EMPTY,
            jnp.where(undefined, METRIC_ZERO_DENOMINATOR, METRIC_SUCCESS),
        ),
    )
    return MetricResult(
        _nan_where_invalid(value, valid),
        valid=valid,
        status=status,
        effective_weight=mass,
    )


def _balanced_from_confusion(
    confusion: Array, mass: Array, invalid: Array
) -> MetricResult:
    support = jnp.sum(confusion, axis=-1)
    recalls = jnp.diagonal(confusion, axis1=-2, axis2=-1) / jnp.where(
        support > 0.0, support, 1.0
    )
    represented = support > 0.0
    represented_count = jnp.sum(represented, axis=-1)
    value = jnp.sum(jnp.where(represented, recalls, 0.0), axis=-1) / jnp.maximum(
        represented_count, 1
    )
    single = represented_count < 2
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=single,
        undefined_status=METRIC_SINGLE_CLASS,
    )


def balanced_accuracy_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    num_classes: int,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Macro recall over represented classes using exact hard predictions."""
    confusion, mass, invalid, _ = _hard_confusion_components(
        y_true,
        y_pred,
        num_classes=num_classes,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="balanced_accuracy_score",
    )
    return _balanced_from_confusion(confusion, mass, invalid)


def smooth_balanced_accuracy_score(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> MetricResult:
    """Balanced expected accuracy, smooth in predicted probabilities."""
    labels, probability, weights, active, invalid, classes = _probability_inputs(
        y_true,
        y_probability,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_balanced_accuracy_score",
        from_logits=from_logits,
    )
    true_hot = jax.nn.one_hot(labels, classes, dtype=probability.dtype)
    confusion = ein.contract(
        "...ni,...n,...nj->...ij",
        true_hot,
        jnp.where(active, weights, 0.0),
        probability,
    )
    mass = jnp.sum(jnp.where(active, weights, 0.0), axis=-1)
    return _balanced_from_confusion(confusion, mass, invalid)


def precision_recall_fscore(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    num_classes: int,
    beta: float = 1.0,
    average: Average = "binary",
    positive_class: int = 1,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> PrecisionRecallFScoreResult:
    """Exact hard-label precision/recall/F-beta with explicit zero-denominator states."""
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    classes = int(num_classes)
    if average not in {"binary", "micro", "macro", "weighted", "none"}:
        raise ValueError(f"Unsupported average {average!r}.")
    if average == "binary" and not 0 <= int(positive_class) < classes:
        raise ValueError("positive_class is outside the class range.")
    confusion, mass, invalid, _ = _hard_confusion_components(
        y_true,
        y_pred,
        num_classes=classes,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="precision_recall_fscore",
    )
    tp = jnp.diagonal(confusion, axis1=-2, axis2=-1)
    support = jnp.sum(confusion, axis=-1)
    predicted = jnp.sum(confusion, axis=-2)
    fp = predicted - tp
    fn = support - tp
    beta2 = float(beta) ** 2

    if average == "micro":
        tp = jnp.sum(tp, axis=-1)
        fp = jnp.sum(fp, axis=-1)
        fn = jnp.sum(fn, axis=-1)
        support_out = jnp.sum(support, axis=-1)
    else:
        precision_by_class = tp / jnp.where(tp + fp > 0.0, tp + fp, 1.0)
        recall_by_class = tp / jnp.where(tp + fn > 0.0, tp + fn, 1.0)
        f_by_class = (
            (1.0 + beta2)
            * tp
            / jnp.where(
                (1.0 + beta2) * tp + beta2 * fn + fp > 0.0,
                (1.0 + beta2) * tp + beta2 * fn + fp,
                1.0,
            )
        )
        defined_by_class = (
            (tp + fp > 0.0)
            & (tp + fn > 0.0)
            & ((1.0 + beta2) * tp + beta2 * fn + fp > 0.0)
        )
        if average == "none":
            invalid_out = invalid[..., None]
            empty = mass[..., None] <= 0.0
            valid = ~invalid_out & ~empty & defined_by_class
            status = jnp.where(
                invalid_out,
                METRIC_INVALID_INPUT,
                jnp.where(
                    empty,
                    METRIC_EMPTY,
                    jnp.where(defined_by_class, METRIC_SUCCESS, METRIC_ZERO_DENOMINATOR),
                ),
            )
            effective = jnp.broadcast_to(mass[..., None], support.shape)
            return PrecisionRecallFScoreResult(
                precision=_nan_where_invalid(precision_by_class, valid),
                recall=_nan_where_invalid(recall_by_class, valid),
                fscore=_nan_where_invalid(f_by_class, valid),
                support=support,
                valid=valid,
                status=status,
                effective_weight=effective,
                average=average,
            )
        if average == "binary":
            index = int(positive_class)
            tp, fp, fn = tp[..., index], fp[..., index], fn[..., index]
            support_out = support[..., index]
        else:
            represented = support > 0.0
            if average == "macro":
                averaging_weight = represented.astype(support.dtype)
            else:
                averaging_weight = support
            averaging_weight = jnp.where(defined_by_class, averaging_weight, 0.0)
            denominator = jnp.sum(averaging_weight, axis=-1)
            precision = jnp.sum(
                averaging_weight * precision_by_class, axis=-1
            ) / jnp.where(denominator > 0.0, denominator, 1.0)
            recall = jnp.sum(averaging_weight * recall_by_class, axis=-1) / jnp.where(
                denominator > 0.0, denominator, 1.0
            )
            fscore = jnp.sum(averaging_weight * f_by_class, axis=-1) / jnp.where(
                denominator > 0.0, denominator, 1.0
            )
            support_out = jnp.sum(support, axis=-1)
            undefined = (denominator <= 0.0) | jnp.any(
                represented & ~defined_by_class, axis=-1
            )
            valid = ~(invalid | (mass <= 0.0) | undefined)
            status = jnp.where(
                invalid,
                METRIC_INVALID_INPUT,
                jnp.where(
                    mass <= 0.0,
                    METRIC_EMPTY,
                    jnp.where(undefined, METRIC_ZERO_DENOMINATOR, METRIC_SUCCESS),
                ),
            )
            return PrecisionRecallFScoreResult(
                precision=_nan_where_invalid(precision, valid),
                recall=_nan_where_invalid(recall, valid),
                fscore=_nan_where_invalid(fscore, valid),
                support=support_out,
                valid=valid,
                status=status,
                effective_weight=mass,
                average=average,
            )

    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f_denominator = (1.0 + beta2) * tp + beta2 * fn + fp
    precision = tp / jnp.where(precision_denominator > 0.0, precision_denominator, 1.0)
    recall = tp / jnp.where(recall_denominator > 0.0, recall_denominator, 1.0)
    fscore = (1.0 + beta2) * tp / jnp.where(f_denominator > 0.0, f_denominator, 1.0)
    undefined = (
        (precision_denominator <= 0.0)
        | (recall_denominator <= 0.0)
        | (f_denominator <= 0.0)
    )
    valid = ~(invalid | (mass <= 0.0) | undefined)
    status = jnp.where(
        invalid,
        METRIC_INVALID_INPUT,
        jnp.where(
            mass <= 0.0,
            METRIC_EMPTY,
            jnp.where(undefined, METRIC_ZERO_DENOMINATOR, METRIC_SUCCESS),
        ),
    )
    return PrecisionRecallFScoreResult(
        precision=_nan_where_invalid(precision, valid),
        recall=_nan_where_invalid(recall, valid),
        fscore=_nan_where_invalid(fscore, valid),
        support=support_out,
        valid=valid,
        status=status,
        effective_weight=mass,
        average=average,
    )


def _prf_metric(
    result: PrecisionRecallFScoreResult,
    select: Callable[[PrecisionRecallFScoreResult], Array],
    /,
) -> MetricResult:
    return MetricResult(
        select(result),
        valid=result.valid,
        status=result.status,
        effective_weight=result.effective_weight,
    )


def precision_score(
    y_true: ArrayLike, y_pred: ArrayLike, /, **kwargs: Any
) -> MetricResult:
    return _prf_metric(
        precision_recall_fscore(y_true, y_pred, **kwargs),
        lambda result: result.precision,
    )


def recall_score(y_true: ArrayLike, y_pred: ArrayLike, /, **kwargs: Any) -> MetricResult:
    return _prf_metric(
        precision_recall_fscore(y_true, y_pred, **kwargs),
        lambda result: result.recall,
    )


def fbeta_score(y_true: ArrayLike, y_pred: ArrayLike, /, **kwargs: Any) -> MetricResult:
    return _prf_metric(
        precision_recall_fscore(y_true, y_pred, **kwargs),
        lambda result: result.fscore,
    )


def f1_score(y_true: ArrayLike, y_pred: ArrayLike, /, **kwargs: Any) -> MetricResult:
    if "beta" in kwargs:
        raise TypeError("f1_score fixes beta=1; use fbeta_score for other beta values.")
    return _prf_metric(
        precision_recall_fscore(y_true, y_pred, beta=1.0, **kwargs),
        lambda result: result.fscore,
    )


def log_loss(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> MetricResult:
    """Categorical negative log score; zero assigned probability gives +infinity."""
    labels, probability, weights, active, invalid, classes = _probability_inputs(
        y_true,
        y_probability,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="log_loss",
        from_logits=from_logits,
    )
    target = jax.nn.one_hot(labels, classes, dtype=probability.dtype)
    if from_logits:
        logits = jnp.where(active[..., None], jnp.asarray(y_probability), 0.0)
        losses = -jnp.sum(jax.nn.log_softmax(logits, axis=-1) * target, axis=-1)
    else:
        selected = jnp.sum(probability * target, axis=-1)
        losses = jnp.where(selected > 0.0, -jnp.log(selected), jnp.inf)
    value, mass = _weighted_mean(losses, weights, active, labels.ndim - 1)
    return _result(value, invalid=invalid, effective_weight=mass)


def brier_score(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> MetricResult:
    """Binary or multiclass Brier score, smooth in probabilities."""
    labels_raw = jnp.asarray(y_true)
    probability_raw = jnp.asarray(y_probability)
    if probability_raw.shape == labels_raw.shape:
        _require_integer_labels(labels_raw, "brier_score")
        _reject_complex(probability_raw, metric="brier_score")
        true, probability, weights, active, invalid, axis = _prepare_pair(
            labels_raw,
            probability_raw,
            sample_weight=sample_weight,
            mask=mask,
            sample_axis=sample_axis,
            metric="brier_score",
            allow_complex=False,
        )
        if axis != true.ndim - 1:
            raise ValueError(
                "Classification labels must have case_shape + (sample,) axes."
            )
        if from_logits:
            probability = jax.nn.sigmoid(probability)
            probability_valid = jnp.isfinite(probability_raw)
        else:
            probability_valid = (
                jnp.isfinite(probability) & (probability >= 0.0) & (probability <= 1.0)
            )
        label_valid = (true == 0) | (true == 1)
        invalid = invalid | jnp.any(
            active & ~(label_valid & probability_valid), axis=axis
        )
        active = active & label_valid & probability_valid
        value, mass = _weighted_mean(
            (probability - true.astype(probability.dtype)) ** 2, weights, active, axis
        )
        return _result(value, invalid=invalid, effective_weight=mass)
    labels, probability, weights, active, invalid, classes = _probability_inputs(
        labels_raw,
        probability_raw,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="brier_score",
        from_logits=from_logits,
    )
    target = jax.nn.one_hot(labels, classes, dtype=probability.dtype)
    per_sample = jnp.sum((probability - target) ** 2, axis=-1)
    value, mass = _weighted_mean(per_sample, weights, active, labels.ndim - 1)
    return _result(value, invalid=invalid, effective_weight=mass)


def _binary_score_inputs(
    y_true: ArrayLike,
    y_score: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    metric: str,
) -> tuple[Array, Array, Array, Array, Array]:
    true_raw = jnp.asarray(y_true)
    score_raw = jnp.asarray(y_score)
    _require_integer_labels(true_raw, metric)
    _reject_complex(score_raw, metric=metric)
    true, score, weights, active, invalid, axis = _prepare_pair(
        true_raw,
        score_raw,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric=metric,
        allow_complex=False,
    )
    if axis != true.ndim - 1:
        raise ValueError(
            "Binary labels and scores must have case_shape + (sample,) axes."
        )
    binary = (true == 0) | (true == 1)
    invalid = invalid | jnp.any(active & ~binary, axis=-1)
    active = active & binary
    weights = jnp.where(active, weights, 0.0)
    mass = jnp.sum(weights, axis=-1)
    return true.astype(jnp.int32), score, weights, invalid, mass


def _map_binary_cases(true: Array, score: Array, weights: Array, function):
    case_shape = true.shape[:-1]
    sample_count = true.shape[-1]
    count = prod(case_shape)
    values = jax.vmap(function)(
        true.reshape((count, sample_count)),
        score.reshape((count, sample_count)),
        weights.reshape((count, sample_count)),
    )
    return values.reshape(case_shape)


def roc_auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Exact weighted binary ROC-AUC using a hard stable sort and half-credit ties."""
    true, score, weights, invalid, mass = _binary_score_inputs(
        y_true,
        y_score,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="roc_auc_score",
    )

    def one_case(labels, scores, weight):
        order = jnp.argsort(scores, stable=True)
        labels = labels[order]
        scores = scores[order]
        weight = weight[order]
        positive = weight * (labels == 1)
        negative = weight * (labels == 0)
        comparison = scores[:, None] - scores[None, :]
        credit = (comparison > 0.0).astype(weight.dtype) + 0.5 * (comparison == 0.0)
        numerator = jnp.sum(positive[:, None] * negative[None, :] * credit)
        positive_mass = jnp.sum(positive)
        negative_mass = jnp.sum(negative)
        return numerator / jnp.where(
            positive_mass * negative_mass > 0.0, positive_mass * negative_mass, 1.0
        )

    value = _map_binary_cases(true, score, weights, one_case)
    positive_mass = jnp.sum(weights * (true == 1), axis=-1)
    negative_mass = jnp.sum(weights * (true == 0), axis=-1)
    single = (positive_mass <= 0.0) | (negative_mass <= 0.0)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=single,
        undefined_status=METRIC_SINGLE_CLASS,
    )


def pr_auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Exact trapezoidal precision-recall AUC at hard, tie-grouped thresholds."""
    true, score, weights, invalid, mass = _binary_score_inputs(
        y_true,
        y_score,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="pr_auc_score",
    )

    def one_case(labels, scores, weight):
        order = jnp.argsort(-scores, stable=True)
        labels = labels[order]
        scores = scores[order]
        weight = weight[order]
        positive = weight * (labels == 1)
        negative = weight * (labels == 0)
        cumulative_positive = jnp.cumsum(positive)
        cumulative_negative = jnp.cumsum(negative)
        group = scores[:, None] == scores[None, :]
        group_positive = jnp.sum(group * positive[None, :], axis=-1)
        group_negative = jnp.sum(group * negative[None, :], axis=-1)
        previous_positive = cumulative_positive - group_positive
        previous_negative = cumulative_negative - group_negative
        positive_mass = jnp.sum(positive)
        recall = cumulative_positive / jnp.where(positive_mass > 0.0, positive_mass, 1.0)
        previous_recall = previous_positive / jnp.where(
            positive_mass > 0.0, positive_mass, 1.0
        )
        precision = cumulative_positive / jnp.where(
            cumulative_positive + cumulative_negative > 0.0,
            cumulative_positive + cumulative_negative,
            1.0,
        )
        previous_total = previous_positive + previous_negative
        previous_precision = jnp.where(
            previous_total > 0.0,
            previous_positive / jnp.where(previous_total > 0.0, previous_total, 1.0),
            1.0,
        )
        endpoint = jnp.concatenate((scores[1:] != scores[:-1], jnp.asarray([True])))
        increments = 0.5 * (precision + previous_precision) * (recall - previous_recall)
        return jnp.sum(jnp.where(endpoint, increments, 0.0))

    value = _map_binary_cases(true, score, weights, one_case)
    positive_mass = jnp.sum(weights * (true == 1), axis=-1)
    negative_mass = jnp.sum(weights * (true == 0), axis=-1)
    single = (positive_mass <= 0.0) | (negative_mass <= 0.0)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=single,
        undefined_status=METRIC_SINGLE_CLASS,
    )


def smooth_roc_auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    /,
    *,
    temperature: float = 1.0,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Pairwise logistic ROC-AUC surrogate, smooth in scores (not labels)."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    true, score, weights, invalid, mass = _binary_score_inputs(
        y_true,
        y_score,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_roc_auc_score",
    )
    positive_weight = weights * (true == 1)
    negative_weight = weights * (true == 0)
    pair_weight = positive_weight[..., :, None] * negative_weight[..., None, :]
    credit = jax.nn.sigmoid(
        (score[..., :, None] - score[..., None, :]) / float(temperature)
    )
    denominator = jnp.sum(pair_weight, axis=(-2, -1))
    value = jnp.sum(pair_weight * credit, axis=(-2, -1)) / jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=denominator <= 0.0,
        undefined_status=METRIC_SINGLE_CLASS,
    )


def smooth_pr_auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    /,
    *,
    temperature: float = 1.0,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Smooth soft-threshold average-precision surrogate for PR-AUC."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    true, score, weights, invalid, mass = _binary_score_inputs(
        y_true,
        y_score,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_pr_auc_score",
    )
    positive_weight = weights * (true == 1)
    soft_above = jax.nn.sigmoid(
        (score[..., None, :] - score[..., :, None]) / float(temperature)
    )
    predicted_mass = jnp.sum(soft_above * weights[..., None, :], axis=-1)
    true_positive_mass = jnp.sum(soft_above * positive_weight[..., None, :], axis=-1)
    soft_precision = true_positive_mass / jnp.where(
        predicted_mass > 0.0, predicted_mass, 1.0
    )
    positive_mass = jnp.sum(positive_weight, axis=-1)
    negative_mass = jnp.sum(weights * (true == 0), axis=-1)
    value = jnp.sum(positive_weight * soft_precision, axis=-1) / jnp.where(
        positive_mass > 0.0, positive_mass, 1.0
    )
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=(positive_mass <= 0.0) | (negative_mass <= 0.0),
        undefined_status=METRIC_SINGLE_CLASS,
    )


def smooth_precision_recall_fscore(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    /,
    *,
    beta: float = 1.0,
    average: Average = "binary",
    positive_class: int = 1,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> PrecisionRecallFScoreResult:
    """Expected-count precision/recall/F-beta, smooth in class probabilities."""
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if average not in {"binary", "micro", "macro", "weighted", "none"}:
        raise ValueError(f"Unsupported average {average!r}.")
    matrix_result = smooth_confusion_matrix(
        y_true,
        y_probability,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        from_logits=from_logits,
    )
    confusion = jnp.where(matrix_result.valid[..., None, None], matrix_result.value, 0.0)
    classes = confusion.shape[-1]
    if average == "binary" and not 0 <= int(positive_class) < classes:
        raise ValueError("positive_class is outside the class range.")
    tp = jnp.diagonal(confusion, axis1=-2, axis2=-1)
    support = jnp.sum(confusion, axis=-1)
    predicted = jnp.sum(confusion, axis=-2)
    fp = predicted - tp
    fn = support - tp
    beta2 = float(beta) ** 2
    precision_by_class = tp / jnp.where(tp + fp > 0.0, tp + fp, 1.0)
    recall_by_class = tp / jnp.where(tp + fn > 0.0, tp + fn, 1.0)
    f_denominator_by_class = (1.0 + beta2) * tp + beta2 * fn + fp
    f_by_class = (
        (1.0 + beta2)
        * tp
        / jnp.where(f_denominator_by_class > 0.0, f_denominator_by_class, 1.0)
    )
    defined_by_class = (tp + fp > 0.0) & (tp + fn > 0.0) & (f_denominator_by_class > 0.0)
    invalid = matrix_result.status == METRIC_INVALID_INPUT
    empty = matrix_result.effective_weight <= 0.0
    mass = matrix_result.effective_weight
    if average == "none":
        valid = matrix_result.valid[..., None] & defined_by_class
        status = jnp.where(
            invalid[..., None],
            METRIC_INVALID_INPUT,
            jnp.where(
                empty[..., None],
                METRIC_EMPTY,
                jnp.where(defined_by_class, METRIC_SUCCESS, METRIC_ZERO_DENOMINATOR),
            ),
        )
        return PrecisionRecallFScoreResult(
            precision=_nan_where_invalid(precision_by_class, valid),
            recall=_nan_where_invalid(recall_by_class, valid),
            fscore=_nan_where_invalid(f_by_class, valid),
            support=support,
            valid=valid,
            status=status,
            effective_weight=jnp.broadcast_to(mass[..., None], support.shape),
            average=average,
        )
    if average == "binary":
        index = int(positive_class)
        precision = precision_by_class[..., index]
        recall = recall_by_class[..., index]
        fscore = f_by_class[..., index]
        support_out = support[..., index]
        undefined = ~defined_by_class[..., index]
    elif average == "micro":
        tp_total = jnp.sum(tp, axis=-1)
        fp_total = jnp.sum(fp, axis=-1)
        fn_total = jnp.sum(fn, axis=-1)
        precision_denominator = tp_total + fp_total
        recall_denominator = tp_total + fn_total
        f_denominator = (1.0 + beta2) * tp_total + beta2 * fn_total + fp_total
        precision = tp_total / jnp.where(
            precision_denominator > 0.0, precision_denominator, 1.0
        )
        recall = tp_total / jnp.where(recall_denominator > 0.0, recall_denominator, 1.0)
        fscore = (
            (1.0 + beta2) * tp_total / jnp.where(f_denominator > 0.0, f_denominator, 1.0)
        )
        support_out = jnp.sum(support, axis=-1)
        undefined = (
            (precision_denominator <= 0.0)
            | (recall_denominator <= 0.0)
            | (f_denominator <= 0.0)
        )
    else:
        represented = support > 0.0
        averaging_weight = (
            represented.astype(support.dtype) if average == "macro" else support
        )
        averaging_weight = jnp.where(defined_by_class, averaging_weight, 0.0)
        denominator = jnp.sum(averaging_weight, axis=-1)
        precision = jnp.sum(averaging_weight * precision_by_class, axis=-1) / jnp.where(
            denominator > 0.0, denominator, 1.0
        )
        recall = jnp.sum(averaging_weight * recall_by_class, axis=-1) / jnp.where(
            denominator > 0.0, denominator, 1.0
        )
        fscore = jnp.sum(averaging_weight * f_by_class, axis=-1) / jnp.where(
            denominator > 0.0, denominator, 1.0
        )
        support_out = jnp.sum(support, axis=-1)
        undefined = (denominator <= 0.0) | jnp.any(
            represented & ~defined_by_class, axis=-1
        )
    valid = ~(invalid | empty | undefined)
    status = jnp.where(
        invalid,
        METRIC_INVALID_INPUT,
        jnp.where(
            empty,
            METRIC_EMPTY,
            jnp.where(undefined, METRIC_ZERO_DENOMINATOR, METRIC_SUCCESS),
        ),
    )
    return PrecisionRecallFScoreResult(
        precision=_nan_where_invalid(precision, valid),
        recall=_nan_where_invalid(recall, valid),
        fscore=_nan_where_invalid(fscore, valid),
        support=support_out,
        valid=valid,
        status=status,
        effective_weight=mass,
        average=average,
    )


def smooth_precision_score(
    y_true: ArrayLike, y_probability: ArrayLike, /, **kwargs: Any
) -> MetricResult:
    return _prf_metric(
        smooth_precision_recall_fscore(y_true, y_probability, **kwargs),
        lambda result: result.precision,
    )


def smooth_recall_score(
    y_true: ArrayLike, y_probability: ArrayLike, /, **kwargs: Any
) -> MetricResult:
    return _prf_metric(
        smooth_precision_recall_fscore(y_true, y_probability, **kwargs),
        lambda result: result.recall,
    )


def smooth_fbeta_score(
    y_true: ArrayLike, y_probability: ArrayLike, /, **kwargs: Any
) -> MetricResult:
    return _prf_metric(
        smooth_precision_recall_fscore(y_true, y_probability, **kwargs),
        lambda result: result.fscore,
    )


def smooth_f1_score(
    y_true: ArrayLike, y_probability: ArrayLike, /, **kwargs: Any
) -> MetricResult:
    if "beta" in kwargs:
        raise TypeError(
            "smooth_f1_score fixes beta=1; use smooth_fbeta_score for other beta values."
        )
    return _prf_metric(
        smooth_precision_recall_fscore(y_true, y_probability, beta=1.0, **kwargs),
        lambda result: result.fscore,
    )


__all__ = [
    "PrecisionRecallFScoreResult",
    "accuracy_score",
    "balanced_accuracy_score",
    "brier_score",
    "confusion_matrix",
    "f1_score",
    "fbeta_score",
    "log_loss",
    "pr_auc_score",
    "precision_recall_fscore",
    "precision_score",
    "recall_score",
    "roc_auc_score",
    "smooth_accuracy_score",
    "smooth_balanced_accuracy_score",
    "smooth_confusion_matrix",
    "smooth_f1_score",
    "smooth_fbeta_score",
    "smooth_precision_recall_fscore",
    "smooth_precision_score",
    "smooth_pr_auc_score",
    "smooth_recall_score",
    "smooth_roc_auc_score",
]
