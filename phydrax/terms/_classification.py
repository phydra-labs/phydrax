#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key
from opt_einsum import contract

from phydrax.domain import DomainComponent, DomainFunction, PointSampling

from .._classification import (
    binary_probabilities_from_logits,
    categorical_probabilities_from_logits,
    classification_probabilities,
    independent_bernoulli_probabilities_from_logits,
    pointwise_classification_loss,
)
from .._doc import DOC_KEY0
from .._exponential_family import BernoulliFamily, CategoricalFamily
from .._likelihoods import (
    CategoricalExponentialFamilyLikelihood,
    IndependentBernoulliLikelihood,
    OrdinalCumulativeLinkLikelihood,
    ScalarNaturalExponentialFamilyLikelihood,
)
from ..ml._classification import ClassificationObjective
from ..ml._schema import TargetSchema
from ..ml.metrics._base import METRIC_INVALID_INPUT, METRIC_SUCCESS
from ._likelihood import (
    _AbstractSupervisedDatasetObservationTerm,
    _AbstractSupervisedLikelihoodTerm,
)
from ._supervised_dataset import SupervisedDatasetBatch


ClassificationKind = Literal["binary", "multiclass", "multilabel"]


def _canonical_hard_targets(
    targets: ArrayLike,
    /,
    *,
    kind: ClassificationKind | Literal["ordinal"],
    width: int,
) -> Array:
    encoded = jnp.asarray(targets)
    if kind != "multilabel" and encoded.ndim == 2 and int(encoded.shape[1]) == 1:
        encoded = encoded[:, 0]
    expected_ndim = 2 if kind == "multilabel" else 1
    if encoded.ndim != expected_ndim:
        expected = "(N, L)" if kind == "multilabel" else "(N,) or (N, 1)"
        raise ValueError(f"{kind} classification targets must have shape {expected}.")
    if kind == "multilabel" and int(encoded.shape[-1]) != width:
        raise ValueError(f"Multilabel targets must end in label count {width}.")
    if encoded.dtype != jnp.bool_ and not jnp.issubdtype(encoded.dtype, jnp.integer):
        raise TypeError("Hard classification targets must be integer or Boolean labels.")
    return encoded


def _fold_case_target_mask(
    target: Array,
    target_mask: ArrayLike | None,
    sample_mask: ArrayLike | None,
    /,
    *,
    event_mask: bool,
) -> tuple[ArrayLike | None, ArrayLike | None]:
    if target_mask is None or event_mask:
        return sample_mask, target_mask
    mask = jnp.asarray(target_mask)
    if mask.dtype != jnp.bool_ or mask.shape != target.shape:
        raise ValueError("target_mask must be Boolean and match classification targets.")
    if sample_mask is None:
        return mask, None
    cases = jnp.asarray(sample_mask)
    if cases.dtype != jnp.bool_ or cases.shape != (int(target.shape[0]),):
        raise ValueError("sample_mask must be Boolean with shape (N,).")
    return cases & mask, None


def _validate_active_hard_targets(
    values: Array,
    configured: Array,
    /,
    *,
    upper: int,
    target_mask: Array | None,
) -> None:
    selected = values[configured]
    mask = (
        jnp.ones(selected.shape, dtype=bool)
        if target_mask is None
        else target_mask[configured]
    )
    invalid = mask & ((selected < 0) | (selected >= int(upper)))
    if bool(jnp.any(invalid)):
        raise ValueError(
            f"Configured classification targets must lie within [0, {int(upper)})."
        )


def _configured(term: _AbstractSupervisedDatasetObservationTerm, /) -> Array:
    return (
        jnp.arange(term.domain.size, dtype=jnp.int32)
        if term.indices is None
        else term.indices
    )


def _weights(batch: SupervisedDatasetBatch, /) -> Array:
    return (
        jnp.ones((int(batch.target.shape[0]),), dtype=float)
        if batch.sample_weight is None
        else jnp.asarray(batch.sample_weight, dtype=float)
    )


def _weighted_mean(values: Array, weights: Array, /) -> Array:
    return jnp.sum(weights * values) / jnp.sum(weights)


def _metric_state(*values: Array) -> tuple[Array, Array]:
    valid = jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in values)))
    status = jnp.where(valid, METRIC_SUCCESS, METRIC_INVALID_INPUT).astype(jnp.int32)
    return valid, status


def _hard_metrics(
    logits: Array,
    target: Array,
    /,
    *,
    kind: ClassificationKind,
    class_count: int,
    target_mask: Array | None,
    sample_weight: Array,
) -> dict[str, Array]:
    per_case_nll = pointwise_classification_loss(
        logits,
        target,
        kind=kind,
        objective="nll",
        class_count=class_count,
        target_mask=target_mask,
    )
    nll = _weighted_mean(per_case_nll, sample_weight)
    if kind == "binary":
        probabilities = binary_probabilities_from_logits(logits)
        prediction = probabilities >= 0.5
        accuracy = _weighted_mean(
            (prediction == jnp.asarray(target, dtype=bool)).astype(float), sample_weight
        )
        brier = _weighted_mean(
            (probabilities - jnp.asarray(target, dtype=float)) ** 2,
            sample_weight,
        )
        metric_mass = jnp.sum(sample_weight)
        accuracy_key = "data_accuracy"
    elif kind == "multiclass":
        probabilities = categorical_probabilities_from_logits(
            logits, class_count=class_count
        )
        labels = jnp.asarray(target, dtype=jnp.int32)
        selected = jnp.take_along_axis(probabilities, labels[..., None], axis=-1)[..., 0]
        prediction = jnp.argmax(probabilities, axis=-1)
        accuracy = _weighted_mean((prediction == labels).astype(float), sample_weight)
        per_case_brier = jnp.sum(probabilities**2, axis=-1) - 2.0 * selected + 1.0
        brier = _weighted_mean(per_case_brier, sample_weight)
        metric_mass = jnp.sum(sample_weight)
        accuracy_key = "data_accuracy"
    else:
        raw_logits = jnp.asarray(logits)
        mask = (
            jnp.ones(raw_logits.shape, dtype=bool)
            if target_mask is None
            else jnp.asarray(target_mask)
        )
        safe_logits = jnp.where(mask, raw_logits, 0.0)
        safe_target = jnp.where(mask, jnp.asarray(target), 0)
        probabilities = independent_bernoulli_probabilities_from_logits(safe_logits)
        label_weight = sample_weight[..., None] * mask.astype(float)
        metric_mass = jnp.sum(label_weight)
        prediction = probabilities >= 0.5
        target_bool = jnp.asarray(safe_target, dtype=bool)
        accuracy = (
            jnp.sum(label_weight * (prediction == target_bool).astype(float))
            / metric_mass
        )
        brier = (
            jnp.sum(
                label_weight
                * (probabilities - jnp.asarray(safe_target, dtype=float)) ** 2
            )
            / metric_mass
        )
        accuracy_key = "data_binary_accuracy"
    valid, status = _metric_state(per_case_nll, accuracy, brier)
    result = {
        "data_negative_log_likelihood": jnp.asarray(nll, dtype=float).reshape(()),
        accuracy_key: jnp.asarray(accuracy, dtype=float).reshape(()),
        "data_brier_score": jnp.asarray(brier, dtype=float).reshape(()),
        "data_effective_weight": jnp.sum(sample_weight).reshape(()),
        "data_valid": jnp.asarray(valid, dtype=bool).reshape(()),
        "data_status": status.reshape(()),
    }
    if kind == "multilabel":
        result["data_effective_label_weight"] = jnp.asarray(
            metric_mass, dtype=float
        ).reshape(())
        result["data_observed_label_count"] = jnp.sum(
            jnp.asarray(target_mask, dtype=bool)
            if target_mask is not None
            else jnp.ones(jnp.asarray(target).shape, dtype=bool)
        ).astype(float)
    return result


class SupervisedClassificationTerm(_AbstractSupervisedLikelihoodTerm):
    """Train hard binary, multiclass, or independent multilabel logits."""

    target_schema: TargetSchema
    classification_kind: ClassificationKind = eqx.field(static=True)
    class_count: int = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        targets: ArrayLike,
        target_schema: TargetSchema,
        /,
        *,
        sampling: PointSampling,
        observation_operator: Callable[[DomainFunction], DomainFunction] | None = None,
        sample_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        target_mask: ArrayLike | None = None,
        weight: ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        label: str | None = None,
    ):
        if not isinstance(target_schema, TargetSchema):
            raise TypeError("target_schema must be a TargetSchema.")
        kind = target_schema.kind
        if kind == "binary":
            class_count = 2
            likelihood = ScalarNaturalExponentialFamilyLikelihood(BernoulliFamily())
        elif kind == "multiclass":
            class_count = target_schema.num_classes
            if class_count < 2:
                raise ValueError(
                    "Multiclass classification requires class_labels for every class."
                )
            likelihood = CategoricalExponentialFamilyLikelihood(
                CategoricalFamily(class_count),
                prediction_coordinates="full_logits",
            )
        elif kind == "multilabel":
            class_count = target_schema.num_labels
            likelihood = IndependentBernoulliLikelihood(class_count)
        else:
            raise ValueError(
                "SupervisedClassificationTerm supports binary, multiclass, and "
                "multilabel TargetSchema kinds."
            )
        encoded = _canonical_hard_targets(
            targets,
            kind=kind,
            width=class_count,
        )
        sample_mask, target_mask = _fold_case_target_mask(
            encoded,
            target_mask,
            sample_mask,
            event_mask=kind == "multilabel",
        )
        super().__init__(
            str(field),
            component,
            encoded,
            likelihood,
            sampling=sampling,
            observation_operator=observation_operator,
            sample_mask=sample_mask,
            sample_weight=sample_weight,
            target_mask=target_mask,
            weight=weight,
            reduction=reduction,
            indices=indices,
            label=label,
        )
        _validate_active_hard_targets(
            self.values,
            _configured(self),
            upper=2 if kind in ("binary", "multilabel") else class_count,
            target_mask=self.target_mask,
        )
        self.target_schema = target_schema
        self.classification_kind = kind
        self.class_count = class_count

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: SupervisedDatasetBatch | None = None,
        **kwargs: Any,
    ) -> dict[str, Array]:
        batch_value = self.sample(key=key) if batch is None else batch
        location, target = self.likelihood.align_observations(
            self._location(functions, batch_value, key=key, **kwargs),
            batch_value.target,
        )
        return _hard_metrics(
            location,
            target,
            kind=self.classification_kind,
            class_count=self.class_count,
            target_mask=batch_value.target_mask,
            sample_weight=_weights(batch_value),
        )


class SupervisedSoftClassificationTerm(_AbstractSupervisedDatasetObservationTerm):
    """Train binary or categorical logits against probability-valued targets."""

    target_schema: TargetSchema
    classification_kind: Literal["binary", "multiclass"] = eqx.field(static=True)
    class_count: int = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        targets: ArrayLike,
        target_schema: TargetSchema,
        /,
        *,
        sampling: PointSampling,
        observation_operator: Callable[[DomainFunction], DomainFunction] | None = None,
        sample_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        weight: ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        label: str | None = None,
    ):
        if not isinstance(target_schema, TargetSchema) or target_schema.kind not in (
            "binary",
            "multiclass",
        ):
            raise ValueError("Soft classification requires binary or multiclass schema.")
        kind = target_schema.kind
        class_count = 2 if kind == "binary" else target_schema.num_classes
        values = jnp.asarray(targets)
        if kind == "binary" and values.ndim == 2 and int(values.shape[-1]) == 1:
            values = values[:, 0]
        expected_ndim = 1 if kind == "binary" else 2
        if values.ndim != expected_ndim or (
            kind == "multiclass" and int(values.shape[-1]) != class_count
        ):
            raise ValueError(
                "Soft classification target shape is incompatible with schema."
            )
        if not jnp.issubdtype(values.dtype, jnp.inexact) or jnp.issubdtype(
            values.dtype, jnp.complexfloating
        ):
            raise TypeError("Soft classification targets must be real inexact arrays.")
        super().__init__(
            str(field),
            component,
            values,
            sampling=sampling,
            observation_operator=observation_operator,
            sample_mask=sample_mask,
            sample_weight=sample_weight,
            weight=weight,
            reduction=reduction,
            indices=indices,
            label=label,
        )
        configured = _configured(self)
        probe_logits = (
            jnp.zeros_like(self.values[configured])
            if kind == "binary"
            else jnp.zeros(self.values[configured].shape, dtype=float)
        )
        probe = pointwise_classification_loss(
            probe_logits,
            self.values[configured],
            kind=kind,
            objective="soft_cross_entropy",
            class_count=class_count,
        )
        if bool(jnp.any(~jnp.isfinite(probe))):
            raise ValueError("Configured soft classification targets are invalid.")
        self.target_schema = target_schema
        self.classification_kind = kind
        self.class_count = class_count

    def per_case_loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""],
        batch: SupervisedDatasetBatch,
        **kwargs: Any,
    ) -> Array:
        return pointwise_classification_loss(
            self._location(functions, batch, key=key, **kwargs),
            batch.target,
            kind=self.classification_kind,
            objective="soft_cross_entropy",
            class_count=self.class_count,
        )

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: SupervisedDatasetBatch | None = None,
        **kwargs: Any,
    ) -> dict[str, Array]:
        batch_value = self.sample(key=key) if batch is None else batch
        logits = self._location(functions, batch_value, key=key, **kwargs)
        target = jnp.asarray(batch_value.target, dtype=float)
        per_case = pointwise_classification_loss(
            logits,
            target,
            kind=self.classification_kind,
            objective="soft_cross_entropy",
            class_count=self.class_count,
        )
        probabilities = classification_probabilities(
            logits,
            kind=self.classification_kind,
            class_count=self.class_count,
        )
        if self.classification_kind == "binary":
            per_case_brier = (probabilities - target) ** 2
            entropy = -(
                jnp.where(target > 0.0, target * jnp.log(target), 0.0)
                + jnp.where(
                    target < 1.0,
                    (1.0 - target) * jnp.log1p(-target),
                    0.0,
                )
            )
        else:
            per_case_brier = jnp.sum((probabilities - target) ** 2, axis=-1)
            entropy = -contract(
                "...k,...k->...",
                target,
                jnp.where(target > 0.0, jnp.log(target), 0.0),
            )
        weights = _weights(batch_value)
        cross_entropy = _weighted_mean(per_case, weights)
        brier = _weighted_mean(per_case_brier, weights)
        target_entropy = _weighted_mean(entropy, weights)
        valid, status = _metric_state(per_case, brier, target_entropy)
        return {
            "data_cross_entropy": cross_entropy.reshape(()),
            "data_brier_score": brier.reshape(()),
            "data_target_entropy": target_entropy.reshape(()),
            "data_effective_weight": jnp.sum(weights).reshape(()),
            "data_valid": valid.reshape(()),
            "data_status": status.reshape(()),
        }


class SupervisedFocalClassificationTerm(_AbstractSupervisedDatasetObservationTerm):
    """Train hard binary, multiclass, or multilabel logits with focal risk."""

    target_schema: TargetSchema
    classification_kind: ClassificationKind = eqx.field(static=True)
    class_count: int = eqx.field(static=True)
    objective: ClassificationObjective

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        targets: ArrayLike,
        target_schema: TargetSchema,
        /,
        *,
        gamma: float = 2.0,
        alpha: float | tuple[float, ...] | None = None,
        sampling: PointSampling,
        observation_operator: Callable[[DomainFunction], DomainFunction] | None = None,
        sample_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        target_mask: ArrayLike | None = None,
        weight: ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        label: str | None = None,
    ):
        if not isinstance(target_schema, TargetSchema) or target_schema.kind not in (
            "binary",
            "multiclass",
            "multilabel",
        ):
            raise ValueError(
                "Focal classification requires a hard classification schema."
            )
        kind = target_schema.kind
        class_count = (
            2
            if kind == "binary"
            else target_schema.num_classes
            if kind == "multiclass"
            else target_schema.num_labels
        )
        encoded = _canonical_hard_targets(
            targets,
            kind=kind,
            width=class_count,
        )
        objective = ClassificationObjective.focal(gamma=gamma, alpha=alpha)
        sample_mask, target_mask = _fold_case_target_mask(
            encoded,
            target_mask,
            sample_mask,
            event_mask=kind == "multilabel",
        )
        super().__init__(
            str(field),
            component,
            encoded,
            sampling=sampling,
            observation_operator=observation_operator,
            sample_mask=sample_mask,
            sample_weight=sample_weight,
            target_mask=target_mask,
            weight=weight,
            reduction=reduction,
            indices=indices,
            label=label,
        )
        _validate_active_hard_targets(
            self.values,
            _configured(self),
            upper=2 if kind in ("binary", "multilabel") else class_count,
            target_mask=self.target_mask,
        )
        self.target_schema = target_schema
        self.classification_kind = kind
        self.class_count = class_count
        self.objective = objective

    def per_case_loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""],
        batch: SupervisedDatasetBatch,
        **kwargs: Any,
    ) -> Array:
        return pointwise_classification_loss(
            self._location(functions, batch, key=key, **kwargs),
            batch.target,
            kind=self.classification_kind,
            objective="focal",
            class_count=self.class_count,
            target_mask=batch.target_mask,
            gamma=self.objective.gamma,
            alpha=self.objective.alpha,
        )

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: SupervisedDatasetBatch | None = None,
        **kwargs: Any,
    ) -> dict[str, Array]:
        batch_value = self.sample(key=key) if batch is None else batch
        logits = self._location(functions, batch_value, key=key, **kwargs)
        metrics = _hard_metrics(
            logits,
            batch_value.target,
            kind=self.classification_kind,
            class_count=self.class_count,
            target_mask=batch_value.target_mask,
            sample_weight=_weights(batch_value),
        )
        focal = self.per_case_loss(
            functions,
            key=key,
            batch=batch_value,
            **kwargs,
        )
        metrics["data_focal_risk"] = _weighted_mean(focal, _weights(batch_value)).reshape(
            ()
        )
        return metrics


class SupervisedOrdinalClassificationTerm(_AbstractSupervisedLikelihoodTerm):
    """Train fixed- or learned-cutpoint ordinal models on hard or soft targets."""

    target_schema: TargetSchema
    class_count: int = eqx.field(static=True)
    target_encoding: Literal["hard", "soft"] = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        targets: ArrayLike,
        target_schema: TargetSchema,
        /,
        *,
        thresholds: ArrayLike | None = None,
        prediction_mode: Literal["location", "cumulative_logits"] = "location",
        target_encoding: Literal["hard", "soft"] = "hard",
        sampling: PointSampling,
        observation_operator: Callable[[DomainFunction], DomainFunction] | None = None,
        sample_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        target_mask: ArrayLike | None = None,
        weight: ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        label: str | None = None,
    ):
        if not isinstance(target_schema, TargetSchema) or target_schema.kind != "ordinal":
            raise ValueError("Ordinal classification requires an ordinal TargetSchema.")
        if target_encoding not in ("hard", "soft"):
            raise ValueError("target_encoding must be 'hard' or 'soft'.")
        class_count = target_schema.num_classes
        likelihood = OrdinalCumulativeLinkLikelihood(
            thresholds,
            class_count=class_count,
            prediction_mode=prediction_mode,
        )
        if target_encoding == "hard":
            encoded = _canonical_hard_targets(
                targets,
                kind="ordinal",
                width=class_count,
            )
        else:
            encoded = jnp.asarray(targets)
            if (
                encoded.ndim != 2
                or int(encoded.shape[-1]) != class_count
                or not jnp.issubdtype(encoded.dtype, jnp.inexact)
                or jnp.issubdtype(encoded.dtype, jnp.complexfloating)
            ):
                raise ValueError("Soft ordinal targets must be real class-mass arrays.")
        sample_mask, target_mask = _fold_case_target_mask(
            encoded,
            target_mask,
            sample_mask,
            event_mask=False,
        )
        super().__init__(
            str(field),
            component,
            encoded,
            likelihood,
            sampling=sampling,
            observation_operator=observation_operator,
            sample_mask=sample_mask,
            sample_weight=sample_weight,
            target_mask=target_mask,
            weight=weight,
            reduction=reduction,
            indices=indices,
            label=label,
        )
        configured = _configured(self)
        if target_encoding == "hard":
            _validate_active_hard_targets(
                self.values,
                configured,
                upper=class_count,
                target_mask=self.target_mask,
            )
        else:
            probe = jnp.zeros(
                self.values[configured].shape[:-1] + (class_count - 1,),
                dtype=float,
            )
            losses = pointwise_classification_loss(
                probe,
                self.values[configured],
                kind="ordinal",
                objective="soft_cross_entropy",
                thresholds=None,
            )
            if bool(jnp.any(~jnp.isfinite(losses))):
                raise ValueError("Configured soft ordinal targets are invalid.")
        self.target_schema = target_schema
        self.class_count = class_count
        self.target_encoding = target_encoding

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: SupervisedDatasetBatch | None = None,
        **kwargs: Any,
    ) -> dict[str, Array]:
        batch_value = self.sample(key=key) if batch is None else batch
        prediction_values, target = self.likelihood.align_observations(
            self._location(functions, batch_value, key=key, **kwargs),
            batch_value.target,
        )
        probabilities = self.likelihood.class_probabilities(prediction_values)
        if self.target_encoding == "hard":
            target_masses = jax.nn.one_hot(
                jnp.asarray(target, dtype=jnp.int32),
                self.class_count,
                dtype=probabilities.dtype,
            )
        else:
            target_masses = jnp.asarray(target, dtype=probabilities.dtype)
        labels = jnp.argmax(target_masses, axis=-1)
        per_case_nll = -self.likelihood.log_prob(prediction_values, target)
        prediction = jnp.argmax(probabilities, axis=-1)
        levels = jnp.arange(self.class_count, dtype=probabilities.dtype)
        expected_rank = contract("...k,k->...", probabilities, levels)
        target_rank = contract("...k,k->...", target_masses, levels)
        per_case_brier = jnp.sum(
            (probabilities - target_masses) ** 2,
            axis=-1,
        )
        weights = _weights(batch_value)
        nll = _weighted_mean(per_case_nll, weights)
        accuracy = _weighted_mean((prediction == labels).astype(float), weights)
        brier = _weighted_mean(per_case_brier, weights)
        mean_expected_rank = _weighted_mean(expected_rank, weights)
        rank_mae = _weighted_mean(jnp.abs(expected_rank - target_rank), weights)
        valid, status = _metric_state(per_case_nll, accuracy, brier, rank_mae)
        return {
            "data_negative_log_likelihood": nll.reshape(()),
            "data_accuracy": accuracy.reshape(()),
            "data_brier_score": brier.reshape(()),
            "data_expected_rank": mean_expected_rank.reshape(()),
            "data_rank_mean_absolute_error": rank_mae.reshape(()),
            "data_effective_weight": jnp.sum(weights).reshape(()),
            "data_valid": valid.reshape(()),
            "data_status": status.reshape(()),
        }


__all__ = [
    "SupervisedClassificationTerm",
    "SupervisedFocalClassificationTerm",
    "SupervisedOrdinalClassificationTerm",
    "SupervisedSoftClassificationTerm",
]
