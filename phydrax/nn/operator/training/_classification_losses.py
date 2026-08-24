#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from ...._classification import (
    classification_probabilities,
    pointwise_classification_loss,
)
from ..data import (
    FunctionSamples,
    OperatorBatch,
    OperatorClassificationSpec,
    OperatorPrediction,
    OperatorTargetBatch,
)
from ._losses import AbstractOperatorLossTerm, OperatorLossContext


OperatorSupportReduction = Literal["mean", "integral"]
OperatorCaseReduction = Literal["mean", "sum"]
OperatorZeroMeasure = Literal["error", "zero"]
OperatorOverlapKind = Literal["dice", "jaccard", "tversky"]
OperatorOverlapClassReduction = Literal["micro", "macro", "weighted"]
OperatorEmptyOverlap = Literal["error", "zero", "one"]


def _validate_common(
    *,
    name: str,
    weight: float,
    classification: OperatorClassificationSpec,
    prediction_field: str | None,
    target_field: str | None,
    support_reduction: OperatorSupportReduction,
    case_reduction: OperatorCaseReduction,
    zero_measure: OperatorZeroMeasure,
) -> None:
    if not name:
        raise ValueError("Operator classification loss names must be non-empty.")
    if not jnp.isfinite(weight) or weight < 0.0:
        raise ValueError(
            "Operator classification loss weights must be finite and nonnegative."
        )
    if not isinstance(classification, OperatorClassificationSpec):
        raise TypeError("classification must be an OperatorClassificationSpec.")
    if prediction_field is not None and not prediction_field:
        raise ValueError("prediction_field must be non-empty or None.")
    if target_field is not None and not target_field:
        raise ValueError("target_field must be non-empty or None.")
    if support_reduction not in ("mean", "integral"):
        raise ValueError("support_reduction must be 'mean' or 'integral'.")
    if case_reduction not in ("mean", "sum"):
        raise ValueError("case_reduction must be 'mean' or 'sum'.")
    if zero_measure not in ("error", "zero"):
        raise ValueError("zero_measure must be 'error' or 'zero'.")


def _resolve_fields(
    prediction: OperatorPrediction,
    targets: OperatorTargetBatch,
    classification: OperatorClassificationSpec,
    prediction_field: str | None,
    target_field: str | None,
    /,
) -> tuple[Array, Array, FunctionSamples]:
    prediction_name = prediction_field
    if prediction_name is None:
        if len(prediction.fields) != 1:
            raise ValueError("prediction_field is required for multi-output predictions.")
        prediction_name = next(iter(prediction.fields))
    target_name = target_field
    if target_name is None:
        if prediction_name in targets.fields:
            target_name = prediction_name
        elif len(targets.fields) == 1:
            target_name = next(iter(targets.fields))
        else:
            raise ValueError("target_field is required for multi-target batches.")
    predicted = prediction.field(prediction_name)
    truth = targets.field(target_name)
    if predicted.query_name != truth.query_name:
        raise ValueError(
            f"Prediction {prediction_name!r} and target {target_name!r} must use "
            "the same query."
        )
    for label, field in (("prediction", predicted), ("target", truth)):
        actual = field.spec.classification
        if actual is None or actual.to_dict() != classification.to_dict():
            raise ValueError(
                f"Operator classification {label} field does not match the loss spec."
            )
    return predicted.values, truth.values, prediction.query_geometry(predicted.query_name)


def _query_measure(
    query: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
) -> tuple[Array, Array, tuple[int, ...]]:
    weights = query.weights(case_shape=case_shape)
    axes = tuple(range(len(case_shape), len(case_shape) + len(query.sample_shape)))
    measure = jnp.sum(weights, axis=axes)
    return weights, measure, axes


def _handle_zero_measure(
    values: Array,
    measure: Array,
    zero_measure: OperatorZeroMeasure,
    /,
) -> Array:
    if zero_measure == "error":
        return eqx.error_if(
            values,
            jnp.any(measure <= 0.0),
            "Operator classification loss query has zero measure.",
        )
    return jnp.where(measure > 0.0, values, jnp.zeros_like(values))


def _reduce_cases(values: Array, reduction: OperatorCaseReduction, /) -> Array:
    return jnp.mean(values) if reduction == "mean" else jnp.sum(values)


def _reduce_pointwise(
    pointwise: Array,
    query: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
    *,
    support_reduction: OperatorSupportReduction,
    case_reduction: OperatorCaseReduction,
    zero_measure: OperatorZeroMeasure,
) -> Array:
    scores = jnp.asarray(pointwise)
    expected = case_shape + query.sample_shape
    if tuple(int(size) for size in scores.shape) != expected:
        raise ValueError(
            "Pointwise classification loss must have one scalar per query sample; "
            f"expected {expected}, got {scores.shape}."
        )
    weights, measure, axes = _query_measure(query, case_shape)
    per_case = jnp.sum(weights * scores, axis=axes)
    if support_reduction == "mean":
        per_case = jnp.where(
            measure > 0.0,
            per_case / jnp.where(measure > 0.0, measure, 1.0),
            jnp.zeros_like(per_case),
        )
    per_case = _handle_zero_measure(per_case, measure, zero_measure)
    return _reduce_cases(per_case, case_reduction)


def _fingerprint(kind: str, values: dict[str, Any], /) -> str:
    payload = json.dumps(
        {"kind": kind, **values},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _common_fingerprint(term: Any, /) -> dict[str, Any]:
    return {
        "name": term.name,
        "weight": term.weight,
        "classification": term.classification.to_dict(),
        "prediction_field": term.prediction_field,
        "target_field": term.target_field,
        "support_reduction": term.support_reduction,
        "case_reduction": term.case_reduction,
        "zero_measure": term.zero_measure,
    }


def _pointwise_term(
    term: Any,
    prediction: OperatorPrediction,
    targets: OperatorTargetBatch,
    /,
    *,
    objective: Literal["nll", "soft_cross_entropy", "focal"],
    gamma: float = 2.0,
    alpha: float | tuple[float, ...] | None = None,
) -> Array:
    if term.weight == 0.0:
        return jnp.zeros((), dtype=float)
    logits, target, query = _resolve_fields(
        prediction,
        targets,
        term.classification,
        term.prediction_field,
        term.target_field,
    )
    mask = query.mask_array(case_shape=prediction.case_shape)
    pointwise_mask = (
        jnp.broadcast_to(mask[..., None], logits.shape)
        if term.classification.kind == "multilabel"
        else mask
    )
    score = pointwise_classification_loss(
        logits,
        target,
        kind=term.classification.kind,
        objective=objective,
        class_count=term.classification.class_count,
        target_mask=pointwise_mask,
        gamma=gamma,
        alpha=alpha,
        thresholds=(
            term.classification.thresholds
            if term.classification.kind == "ordinal"
            else None
        ),
    )
    value = _reduce_pointwise(
        score,
        query,
        prediction.case_shape,
        support_reduction=term.support_reduction,
        case_reduction=term.case_reduction,
        zero_measure=term.zero_measure,
    )
    return jnp.asarray(term.weight, dtype=value.dtype) * value


@dataclass(frozen=True)
class OperatorClassificationNLL(AbstractOperatorLossTerm):
    """Geometry-aware hard-label negative log likelihood."""

    classification: OperatorClassificationSpec
    name: str = "classification_nll"
    weight: float = 1.0
    prediction_field: str | None = None
    target_field: str | None = None
    support_reduction: OperatorSupportReduction = "mean"
    case_reduction: OperatorCaseReduction = "mean"
    zero_measure: OperatorZeroMeasure = "error"

    def __post_init__(self):
        _validate_common(**self.__dict__)
        if self.classification.target != "hard":
            raise ValueError("OperatorClassificationNLL requires hard targets.")

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        del model, batch, key, step, training, context
        return _pointwise_term(self, prediction, targets, objective="nll")

    @property
    def fingerprint(self) -> str:
        return _fingerprint("operator_classification_nll", _common_fingerprint(self))


@dataclass(frozen=True)
class OperatorSoftClassificationLoss(AbstractOperatorLossTerm):
    """Geometry-aware cross entropy for explicit soft class targets."""

    classification: OperatorClassificationSpec
    name: str = "soft_classification"
    weight: float = 1.0
    prediction_field: str | None = None
    target_field: str | None = None
    support_reduction: OperatorSupportReduction = "mean"
    case_reduction: OperatorCaseReduction = "mean"
    zero_measure: OperatorZeroMeasure = "error"

    def __post_init__(self):
        _validate_common(**self.__dict__)
        if self.classification.target != "soft":
            raise ValueError("OperatorSoftClassificationLoss requires soft targets.")

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        del model, batch, key, step, training, context
        return _pointwise_term(
            self,
            prediction,
            targets,
            objective="soft_cross_entropy",
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            "operator_soft_classification",
            _common_fingerprint(self),
        )


@dataclass(frozen=True)
class OperatorFocalClassificationLoss(AbstractOperatorLossTerm):
    """Geometry-aware focal loss for hard operator labels."""

    classification: OperatorClassificationSpec
    name: str = "focal_classification"
    weight: float = 1.0
    prediction_field: str | None = None
    target_field: str | None = None
    support_reduction: OperatorSupportReduction = "mean"
    case_reduction: OperatorCaseReduction = "mean"
    zero_measure: OperatorZeroMeasure = "error"
    gamma: float = 2.0
    alpha: float | tuple[float, ...] | None = None

    def __post_init__(self):
        _validate_common(
            name=self.name,
            weight=self.weight,
            classification=self.classification,
            prediction_field=self.prediction_field,
            target_field=self.target_field,
            support_reduction=self.support_reduction,
            case_reduction=self.case_reduction,
            zero_measure=self.zero_measure,
        )
        if self.classification.target != "hard":
            raise ValueError("OperatorFocalClassificationLoss requires hard targets.")
        if self.classification.kind == "ordinal":
            raise ValueError(
                "Ordinal classification currently supports NLL, not focal loss."
            )
        if not jnp.isfinite(self.gamma) or self.gamma < 0.0:
            raise ValueError("Focal gamma must be finite and nonnegative.")
        alpha = self.alpha
        if alpha is not None:
            resolved = (
                (float(alpha),)
                if isinstance(alpha, (int, float))
                else tuple(float(value) for value in alpha)
            )
            if self.classification.kind == "multiclass":
                if len(resolved) == 1:
                    resolved = resolved * self.classification.class_count
                if len(resolved) != self.classification.class_count or any(
                    not jnp.isfinite(value) or value <= 0.0 for value in resolved
                ):
                    raise ValueError(
                        "Multiclass focal alpha must provide one finite positive "
                        "value per class."
                    )
                alpha_value: float | tuple[float, ...] = resolved
            else:
                if len(resolved) != 1 or not (
                    jnp.isfinite(resolved[0]) and 0.0 < resolved[0] < 1.0
                ):
                    raise ValueError(
                        "Binary and multilabel focal alpha must be a scalar inside "
                        "(0, 1)."
                    )
                alpha_value = resolved[0]
            object.__setattr__(self, "alpha", alpha_value)
        object.__setattr__(self, "gamma", float(self.gamma))

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        del model, batch, key, step, training, context
        return _pointwise_term(
            self,
            prediction,
            targets,
            objective="focal",
            gamma=self.gamma,
            alpha=self.alpha,
        )

    @property
    def fingerprint(self) -> str:
        values = _common_fingerprint(self)
        values.update(
            {
                "gamma": self.gamma,
                "alpha": list(self.alpha)
                if isinstance(self.alpha, tuple)
                else self.alpha,
            }
        )
        return _fingerprint("operator_focal_classification", values)


def _overlap_score(
    intersection: Array,
    predicted_mass: Array,
    target_mass: Array,
    /,
    *,
    kind: OperatorOverlapKind,
    alpha: float,
    beta: float,
    empty: OperatorEmptyOverlap,
) -> Array:
    if kind == "dice":
        denominator = predicted_mass + target_mass
        numerator = 2.0 * intersection
    elif kind == "jaccard":
        denominator = predicted_mass + target_mass - intersection
        numerator = intersection
    else:
        denominator = (
            intersection
            + alpha * (predicted_mass - intersection)
            + beta * (target_mass - intersection)
        )
        numerator = intersection
    empty_mass = denominator <= 0.0
    if empty == "error":
        numerator = eqx.error_if(
            numerator,
            jnp.any(empty_mass),
            "Operator overlap is undefined for empty prediction and target mass.",
        )
        empty_value = 0.0
    else:
        empty_value = 1.0 if empty == "one" else 0.0
    return jnp.where(
        empty_mass,
        jnp.asarray(empty_value, dtype=numerator.dtype),
        numerator / jnp.where(empty_mass, 1.0, denominator),
    )


@dataclass(frozen=True)
class OperatorOverlapLoss(AbstractOperatorLossTerm):
    """Dice, Jaccard, or Tversky loss reduced once over physical support."""

    classification: OperatorClassificationSpec
    name: str = "classification_overlap"
    weight: float = 1.0
    prediction_field: str | None = None
    target_field: str | None = None
    support_reduction: OperatorSupportReduction = "integral"
    case_reduction: OperatorCaseReduction = "mean"
    zero_measure: OperatorZeroMeasure = "error"
    overlap: OperatorOverlapKind = "dice"
    class_reduction: OperatorOverlapClassReduction = "micro"
    empty: OperatorEmptyOverlap = "zero"
    alpha: float = 0.5
    beta: float = 0.5

    def __post_init__(self):
        _validate_common(
            name=self.name,
            weight=self.weight,
            classification=self.classification,
            prediction_field=self.prediction_field,
            target_field=self.target_field,
            support_reduction=self.support_reduction,
            case_reduction=self.case_reduction,
            zero_measure=self.zero_measure,
        )
        if self.overlap not in ("dice", "jaccard", "tversky"):
            raise ValueError("overlap must be 'dice', 'jaccard', or 'tversky'.")
        if self.class_reduction not in ("micro", "macro", "weighted"):
            raise ValueError("class_reduction must be 'micro', 'macro', or 'weighted'.")
        if self.empty not in ("error", "zero", "one"):
            raise ValueError("empty must be 'error', 'zero', or 'one'.")
        if not jnp.isfinite(self.alpha) or not jnp.isfinite(self.beta):
            raise ValueError("Tversky alpha and beta must be finite.")
        if self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError("Tversky alpha and beta must be nonnegative.")

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        if self.weight == 0.0:
            return jnp.zeros((), dtype=float)
        del model, batch, key, step, training, context
        logits, target, query = _resolve_fields(
            prediction,
            targets,
            self.classification,
            self.prediction_field,
            self.target_field,
        )
        probabilities = classification_probabilities(
            logits,
            kind=self.classification.kind,
            class_count=self.classification.class_count,
            thresholds=(
                self.classification.thresholds
                if self.classification.kind == "ordinal"
                else None
            ),
        )
        mask = query.mask_array(case_shape=prediction.case_shape)
        raw_weights, measure, axes = _query_measure(query, prediction.case_shape)
        if self.support_reduction == "mean":
            broadcast_measure = measure.reshape(
                measure.shape + (1,) * len(query.sample_shape)
            )
            weights = jnp.where(
                broadcast_measure > 0.0,
                raw_weights / jnp.where(broadcast_measure > 0.0, broadcast_measure, 1.0),
                jnp.zeros_like(raw_weights),
            )
        else:
            weights = raw_weights

        kind = self.classification.kind
        if kind in ("multiclass", "ordinal"):
            probability_channels = probabilities
            if self.classification.target == "soft":
                finite = jnp.all(jnp.isfinite(target), axis=-1)
                bounded = jnp.all((target >= 0.0) & (target <= 1.0), axis=-1)
                simplex = jnp.abs(jnp.sum(target, axis=-1) - 1.0) <= 1e-6
                target_channels = eqx.error_if(
                    target,
                    jnp.any(mask & ~(finite & bounded & simplex)),
                    "Active soft multiclass targets must be probability simplexes.",
                )
            else:
                valid = (target >= 0) & (target < self.classification.class_count)
                target = eqx.error_if(
                    target,
                    jnp.any(mask & ~valid),
                    "Active hard classification target is outside the class range.",
                )
                target_channels = None
        elif kind == "multilabel":
            probability_channels = probabilities
            target_channels = target
            if self.classification.target == "hard":
                valid = (target == 0) | (target == 1)
                target_channels = eqx.error_if(
                    target,
                    jnp.any(mask[..., None] & ~valid),
                    "Active hard multilabel targets must be zero or one.",
                )
            else:
                valid = jnp.isfinite(target) & (target >= 0.0) & (target <= 1.0)
                target_channels = eqx.error_if(
                    target,
                    jnp.any(mask[..., None] & ~valid),
                    "Active soft multilabel targets must be probabilities.",
                )
        else:
            probability_channels = probabilities[..., None]
            target_channels = target[..., None]
            if self.classification.target == "hard":
                valid = (target == 0) | (target == 1)
                target_channels = eqx.error_if(
                    target_channels,
                    jnp.any(mask & ~valid),
                    "Active hard binary targets must be zero or one.",
                )
            else:
                valid = jnp.isfinite(target) & (target >= 0.0) & (target <= 1.0)
                target_channels = eqx.error_if(
                    target_channels,
                    jnp.any(mask & ~valid),
                    "Active soft binary targets must be probabilities.",
                )

        intersections = []
        predicted_masses = []
        target_masses = []
        channel_count = int(probability_channels.shape[-1])
        for index in range(channel_count):
            probability = jnp.where(
                mask,
                probability_channels[..., index],
                jnp.zeros((), dtype=probability_channels.dtype),
            )
            if target_channels is None:
                indicator = jnp.where(mask, target == index, False).astype(
                    probability.dtype
                )
            else:
                indicator = jnp.where(
                    mask,
                    target_channels[..., index],
                    jnp.zeros((), dtype=target_channels.dtype),
                ).astype(probability.dtype)
            intersections.append(jnp.sum(weights * probability * indicator, axis=axes))
            predicted_masses.append(jnp.sum(weights * probability, axis=axes))
            target_masses.append(jnp.sum(weights * indicator, axis=axes))
        intersection = jnp.stack(intersections, axis=-1)
        predicted_mass = jnp.stack(predicted_masses, axis=-1)
        target_mass = jnp.stack(target_masses, axis=-1)

        if self.class_reduction == "micro":
            score = _overlap_score(
                jnp.sum(intersection, axis=-1),
                jnp.sum(predicted_mass, axis=-1),
                jnp.sum(target_mass, axis=-1),
                kind=self.overlap,
                alpha=self.alpha,
                beta=self.beta,
                empty=self.empty,
            )
        else:
            class_scores = _overlap_score(
                intersection,
                predicted_mass,
                target_mass,
                kind=self.overlap,
                alpha=self.alpha,
                beta=self.beta,
                empty=self.empty,
            )
            if self.class_reduction == "macro":
                score = jnp.mean(class_scores, axis=-1)
            else:
                total_target = jnp.sum(target_mass, axis=-1)
                weighted = jnp.sum(class_scores * target_mass, axis=-1)
                if self.empty == "error":
                    weighted = eqx.error_if(
                        weighted,
                        jnp.any(total_target <= 0.0),
                        "Target-weighted overlap is undefined with zero target mass.",
                    )
                fallback = 1.0 if self.empty == "one" else 0.0
                score = jnp.where(
                    total_target > 0.0,
                    weighted / jnp.where(total_target > 0.0, total_target, 1.0),
                    jnp.asarray(fallback, dtype=weighted.dtype),
                )
        per_case = _handle_zero_measure(1.0 - score, measure, self.zero_measure)
        value = _reduce_cases(per_case, self.case_reduction)
        return jnp.asarray(self.weight, dtype=value.dtype) * value

    @property
    def fingerprint(self) -> str:
        values = _common_fingerprint(self)
        values.update(
            {
                "overlap": self.overlap,
                "class_reduction": self.class_reduction,
                "empty": self.empty,
                "alpha": self.alpha,
                "beta": self.beta,
            }
        )
        return _fingerprint("operator_classification_overlap", values)


__all__ = [
    "OperatorCaseReduction",
    "OperatorClassificationNLL",
    "OperatorEmptyOverlap",
    "OperatorFocalClassificationLoss",
    "OperatorOverlapClassReduction",
    "OperatorOverlapKind",
    "OperatorOverlapLoss",
    "OperatorSoftClassificationLoss",
    "OperatorSupportReduction",
    "OperatorZeroMeasure",
]
