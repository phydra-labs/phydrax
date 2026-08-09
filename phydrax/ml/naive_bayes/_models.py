#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ..._strict import AbstractAttribute, StrictModule
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import effective_sample_size
from .._schema import TargetSchema
from ..discriminant._models import _labels_for, _reshape_for_samples


class NaiveBayesDiagnostics(StrictModule):
    """Weighted sufficient-statistic diagnostics shared by Naive Bayes fits."""

    valid: Array
    status: Array
    effective_samples: Array
    class_mass: Array
    absent_classes: Array
    feature_mass: Array
    domain_valid: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        effective_samples: Any,
        class_mass: Any,
        feature_mass: Any,
        domain_valid: Any,
        method: str,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.effective_samples = jnp.asarray(effective_samples)
        self.class_mass = jnp.asarray(class_mass)
        self.absent_classes = self.class_mass <= 0.0
        self.feature_mass = jnp.asarray(feature_mass)
        self.domain_valid = jnp.asarray(domain_valid, dtype=bool)
        self.method = str(method)


class AbstractNaiveBayesModel(AbstractArrayModel):
    """Common normalized classification API for native Naive Bayes models."""

    labels: AbstractAttribute[Array]
    target_schema: AbstractAttribute[TargetSchema]
    case_shape: AbstractAttribute[tuple[int, ...]]

    @abstractmethod
    def joint_log_likelihood(self, x: Any, /) -> Array:
        raise NotImplementedError

    def decision_function(self, x: Any, /) -> Array:
        return self.joint_log_likelihood(x)

    def predict_log_proba(self, x: Any, /) -> Array:
        return jax.nn.log_softmax(self.joint_log_likelihood(x), axis=-1)

    def predict_proba(self, x: Any, /) -> Array:
        return jax.nn.softmax(self.joint_log_likelihood(x), axis=-1)

    def predict_indices(self, x: Any, /) -> Array:
        return jnp.argmax(self.joint_log_likelihood(x), axis=-1)

    def predict(self, x: Any, /) -> Array:
        return jnp.take(self.labels, self.predict_indices(x), axis=0)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


class GaussianNaiveBayesModel(AbstractNaiveBayesModel):
    means: Array
    variances: Array
    log_priors: Array
    labels: Array
    target_schema: TargetSchema
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        means: Array,
        variances: Array,
        log_priors: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        case_shape: tuple[int, ...],
    ):
        self.means = jnp.asarray(means)
        self.variances = jnp.asarray(variances)
        self.log_priors = jnp.asarray(log_priors)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.case_shape = tuple(case_shape)
        self.in_size = int(self.means.shape[-1])
        self.out_size = int(self.means.shape[-2])

    def joint_log_likelihood(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        if values.shape[-1] != self.in_size:
            raise ValueError(f"Expected {self.in_size} features; got {values.shape[-1]}.")
        extra = values.ndim - len(self.case_shape) - 1
        means = _reshape_for_samples(self.means, self.case_shape, extra)
        variances = _reshape_for_samples(self.variances, self.case_shape, extra)
        log_priors = _reshape_for_samples(self.log_priors, self.case_shape, extra)
        difference = values[..., None, :] - means
        terms = (
            jnp.log(2.0 * jnp.pi * variances)
            + jnp.real(difference * jnp.conj(difference)) / variances
        )
        return log_priors - 0.5 * jnp.sum(terms, axis=-1)


class BernoulliNaiveBayesModel(AbstractNaiveBayesModel):
    feature_log_prob: Array
    feature_log_neg_prob: Array
    log_priors: Array
    labels: Array
    target_schema: TargetSchema
    threshold: float = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        feature_log_prob: Array,
        feature_log_neg_prob: Array,
        log_priors: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        threshold: float,
        case_shape: tuple[int, ...],
    ):
        self.feature_log_prob = jnp.asarray(feature_log_prob)
        self.feature_log_neg_prob = jnp.asarray(feature_log_neg_prob)
        self.log_priors = jnp.asarray(log_priors)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.threshold = float(threshold)
        self.case_shape = tuple(case_shape)
        self.in_size = int(self.feature_log_prob.shape[-1])
        self.out_size = int(self.feature_log_prob.shape[-2])

    def joint_log_likelihood(self, x: Any, /) -> Array:
        raw = jnp.asarray(x)
        if jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise TypeError("Bernoulli Naive Bayes requires real-valued features.")
        values = (raw > self.threshold).astype(self.feature_log_prob.dtype)
        if values.shape[-1] != self.in_size:
            raise ValueError(f"Expected {self.in_size} features; got {values.shape[-1]}.")
        extra = values.ndim - len(self.case_shape) - 1
        positive = _reshape_for_samples(self.feature_log_prob, self.case_shape, extra)
        negative = _reshape_for_samples(self.feature_log_neg_prob, self.case_shape, extra)
        priors = _reshape_for_samples(self.log_priors, self.case_shape, extra)
        return priors + jnp.sum(
            values[..., None, :] * positive + (1.0 - values[..., None, :]) * negative,
            axis=-1,
        )


class MultinomialNaiveBayesModel(AbstractNaiveBayesModel):
    feature_log_prob: Array
    log_priors: Array
    labels: Array
    target_schema: TargetSchema
    complement: bool = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        feature_log_prob: Array,
        log_priors: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        complement: bool,
        case_shape: tuple[int, ...],
    ):
        self.feature_log_prob = jnp.asarray(feature_log_prob)
        self.log_priors = jnp.asarray(log_priors)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.complement = bool(complement)
        self.case_shape = tuple(case_shape)
        self.in_size = int(self.feature_log_prob.shape[-1])
        self.out_size = int(self.feature_log_prob.shape[-2])

    def joint_log_likelihood(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        if not (
            jnp.issubdtype(values.dtype, jnp.integer)
            or jnp.issubdtype(values.dtype, jnp.floating)
        ):
            raise TypeError(
                "Multinomial and complement Naive Bayes require real count features."
            )
        if values.shape[-1] != self.in_size:
            raise ValueError(f"Expected {self.in_size} features; got {values.shape[-1]}.")
        extra = values.ndim - len(self.case_shape) - 1
        log_prob = _reshape_for_samples(self.feature_log_prob, self.case_shape, extra)
        priors = _reshape_for_samples(self.log_priors, self.case_shape, extra)
        score = jnp.einsum("...f,...cf->...c", values, log_prob)
        result = priors - score if self.complement else priors + score
        domain_valid = jnp.all(jnp.isfinite(values) & (values >= 0.0), axis=-1)
        return jnp.where(domain_valid[..., None], result, jnp.nan)


class CategoricalNaiveBayesModel(AbstractNaiveBayesModel):
    feature_log_prob: Array
    log_priors: Array
    labels: Array
    target_schema: TargetSchema
    category_counts: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        feature_log_prob: Array,
        log_priors: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        category_counts: tuple[int, ...],
        case_shape: tuple[int, ...],
    ):
        self.feature_log_prob = jnp.asarray(feature_log_prob)
        self.log_priors = jnp.asarray(log_priors)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.category_counts = tuple(category_counts)
        self.case_shape = tuple(case_shape)
        self.in_size = len(self.category_counts)
        self.out_size = int(self.feature_log_prob.shape[-3])

    def joint_log_likelihood(self, x: Any, /) -> Array:
        raw_values = jnp.asarray(x)
        if jnp.issubdtype(raw_values.dtype, jnp.complexfloating):
            raise TypeError("Categorical Naive Bayes requires real category codes.")
        values = raw_values.astype(jnp.int32)
        if values.shape[-1] != self.in_size:
            raise ValueError(f"Expected {self.in_size} features; got {values.shape[-1]}.")
        extra = values.ndim - len(self.case_shape) - 1
        table = _reshape_for_samples(self.feature_log_prob, self.case_shape, extra)
        score = jnp.zeros(values.shape[:-1] + (self.out_size,), dtype=table.dtype)
        domain_valid = jnp.ones(values.shape[:-1], dtype=bool)
        for feature, categories in enumerate(self.category_counts):
            raw = raw_values[..., feature]
            valid = (
                jnp.isfinite(raw)
                & (raw >= 0)
                & (raw < categories)
                & (raw == jnp.floor(raw))
            )
            domain_valid = domain_valid & valid
            index = jnp.clip(values[..., feature], 0, categories - 1)
            one_hot = jax.nn.one_hot(index, table.shape[-1], dtype=table.dtype)
            score = score + jnp.sum(
                table[..., :, feature, :] * one_hot[..., None, :], axis=-1
            )
        result = score + _reshape_for_samples(self.log_priors, self.case_shape, extra)
        return jnp.where(domain_valid[..., None], result, jnp.nan)


def _prepare(
    batch: MLBatch, num_classes: int | None, policy: WeightPolicy
) -> tuple[Array, Array, Array, Array, Array, TargetSchema]:
    labels, schema = _labels_for(batch, num_classes)
    y = batch.require_targets()
    if batch.target_shape != ():
        raise ValueError("Naive Bayes requires one scalar class label per sample.")
    x = batch.dense_features()
    matched = y[..., None] == labels
    encoded = jnp.argmax(matched, axis=-1).astype(jnp.int32)
    known = jnp.any(matched, axis=-1)
    target_ok = (
        batch.target_mask if batch.target_mask is not None else jnp.ones_like(known)
    )
    raw_weight = batch.effective_weight(policy)
    weight_ok = jnp.isfinite(raw_weight) & (raw_weight >= 0.0)
    active = batch.sample_mask & known & target_ok & weight_ok
    weight = jnp.where(active, raw_weight, 0.0)
    vocabulary_valid = jnp.all(~(batch.sample_mask & target_ok) | known, axis=-1)
    case_weight_valid = jnp.all(weight_ok, axis=-1) & vocabulary_valid
    weight = jnp.where(case_weight_valid[..., None], weight, jnp.nan)
    return x, encoded, weight, active & case_weight_valid[..., None], labels, schema


def _membership(y: Array, weight: Array, classes: int) -> tuple[Array, Array]:
    weighted = weight[..., :, None] * jax.nn.one_hot(y, classes, dtype=weight.dtype)
    return weighted, jnp.sum(weighted, axis=-2)


def _priors(mass: Array, specified: tuple[float, ...]) -> Array:
    tiny = jnp.finfo(mass.dtype).tiny
    if specified:
        return jnp.broadcast_to(jnp.asarray(specified, dtype=mass.dtype), mass.shape)
    return mass / jnp.maximum(jnp.sum(mass, axis=-1, keepdims=True), tiny)


def _result(
    model: AbstractArrayModel,
    *,
    batch: MLBatch,
    weight: Array,
    mass: Array,
    feature_mass: Array,
    domain_valid: Array,
    method: str,
) -> FitResult:
    finite = (
        jnp.all(
            jnp.isfinite(feature_mass),
            axis=tuple(range(len(batch.case_shape), feature_mass.ndim)),
        )
        if feature_mass.ndim > len(batch.case_shape)
        else jnp.isfinite(feature_mass)
    )
    absent = jnp.any(mass <= 0.0, axis=-1)
    valid = domain_valid & finite & ~absent
    status = jnp.where(
        ~finite,
        ML_NONFINITE,
        jnp.where(
            absent,
            ML_INSUFFICIENT_DATA,
            jnp.where(~domain_valid, ML_INFEASIBLE, ML_SUCCESS),
        ),
    ).astype(jnp.int32)
    diagnostics = NaiveBayesDiagnostics(
        valid=valid,
        status=status,
        effective_samples=effective_sample_size(weight),
        class_mass=mass,
        feature_mass=feature_mass,
        domain_valid=domain_valid,
        method=method,
    )
    contract = GradientContract(
        prediction_inputs="almost-everywhere"
        if method in {"bernoulli-nb", "categorical-nb"}
        else "smooth",
        prediction_parameters="smooth",
        fit_features="conditional",
        fit_targets="none",
        fit_weights="conditional",
        fit_hyperparameters="conditional",
        fit_mode="direct",
        nondifferentiable_outputs=("predict", "predict_indices"),
        conditions=(
            "fixed class vocabulary",
            "positive class mass",
            "valid feature domain",
        ),
    )
    return FitResult(
        model,
        diagnostics,
        valid=valid,
        status=status,
        method=method,
        gradient_contract=contract,
    )


def _validate_common(
    num_classes: int | None, priors: tuple[float, ...], policy: WeightPolicy
) -> tuple[int | None, tuple[float, ...], WeightPolicy]:
    classes = None if num_classes is None else int(num_classes)
    if classes is not None and classes < 2:
        raise ValueError("num_classes must be at least two.")
    values = tuple(float(value) for value in priors)
    if values and (
        any(value <= 0.0 for value in values) or abs(sum(values) - 1.0) > 1e-6
    ):
        raise ValueError("class_prior must be positive and sum to one.")
    if classes is not None and values and len(values) != classes:
        raise ValueError("class_prior must align with num_classes.")
    if policy not in {"none", "statistical", "measure", "product"}:
        raise ValueError("Unsupported weight policy.")
    return classes, values, policy


class GaussianNaiveBayesRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    class_prior: tuple[float, ...] = eqx.field(static=True)
    var_smoothing: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        class_prior: tuple[float, ...] = (),
        var_smoothing: float = 1e-9,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.num_classes, self.class_prior, self.weight_policy = _validate_common(
            num_classes, class_prior, weight_policy
        )
        self.var_smoothing = jnp.asarray(var_smoothing, dtype=float)
        if self.var_smoothing.ndim != 0 or float(self.var_smoothing) <= 0.0:
            raise ValueError("var_smoothing must be positive.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, y, weight, active, labels, schema = _prepare(
            batch, self.num_classes, self.weight_policy
        )
        if not (
            jnp.issubdtype(x.dtype, jnp.floating)
            or jnp.issubdtype(x.dtype, jnp.complexfloating)
        ):
            x = x.astype(jnp.float32)
        raw_finite = jnp.isfinite(jnp.real(x)) & jnp.isfinite(jnp.imag(x))
        finite = raw_finite & batch.feature_mask
        classes = int(labels.shape[0])
        class_weight, mass = _membership(y, weight, classes)
        feature_weight = class_weight[..., :, :, None] * finite[..., :, None, :]
        feature_mass = jnp.sum(feature_weight, axis=-3)
        safe_x = jnp.where(finite, x, 0)
        means = jnp.einsum("...ncf,...nf->...cf", feature_weight, safe_x) / jnp.maximum(
            feature_mass, jnp.finfo(weight.dtype).tiny
        )
        difference = safe_x[..., :, None, :] - means[..., None, :, :]
        variances = jnp.sum(
            feature_weight * jnp.real(difference * jnp.conj(difference)), axis=-3
        ) / jnp.maximum(feature_mass, jnp.finfo(weight.dtype).tiny)
        global_scale = jnp.max(variances, axis=(-2, -1), keepdims=True)
        variances = variances + self.var_smoothing * jnp.maximum(global_scale, 1.0)
        priors = _priors(mass, self.class_prior)
        model = GaussianNaiveBayesModel(
            means,
            variances,
            jnp.log(jnp.maximum(priors, jnp.finfo(weight.dtype).tiny)),
            labels,
            schema,
            case_shape=batch.case_shape,
        )
        data_valid = jnp.all(
            ~(active[..., None] & batch.feature_mask) | raw_finite, axis=(-2, -1)
        )
        domain_valid = (
            data_valid
            & jnp.all(jnp.isfinite(weight), axis=-1)
            & jnp.all(feature_mass > 0.0, axis=(-2, -1))
        )
        return _result(
            model,
            batch=batch,
            weight=weight,
            mass=mass,
            feature_mass=feature_mass,
            domain_valid=domain_valid,
            method="gaussian-nb",
        )


class BernoulliNaiveBayesRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    class_prior: tuple[float, ...] = eqx.field(static=True)
    alpha: Array
    threshold: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        class_prior: tuple[float, ...] = (),
        alpha: float = 1.0,
        threshold: float = 0.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.num_classes, self.class_prior, self.weight_policy = _validate_common(
            num_classes, class_prior, weight_policy
        )
        self.alpha = jnp.asarray(alpha, dtype=float)
        self.threshold = float(threshold)
        if self.alpha.ndim != 0 or float(self.alpha) <= 0.0:
            raise ValueError("alpha must be positive.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, y, weight, active, labels, schema = _prepare(
            batch, self.num_classes, self.weight_policy
        )
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError("Bernoulli Naive Bayes requires real-valued features.")
        raw_finite = jnp.isfinite(x)
        finite = raw_finite & batch.feature_mask
        binary = (jnp.where(finite, x, 0) > self.threshold).astype(weight.dtype)
        class_weight, mass = _membership(y, weight, int(labels.shape[0]))
        feature_weight = class_weight[..., :, :, None] * finite[..., :, None, :]
        feature_mass = jnp.sum(feature_weight, axis=-3)
        positive = jnp.einsum("...ncf,...nf->...cf", feature_weight, binary)
        probability = (positive + self.alpha) / (feature_mass + 2.0 * self.alpha)
        priors = _priors(mass, self.class_prior)
        model = BernoulliNaiveBayesModel(
            jnp.log(probability),
            jnp.log1p(-probability),
            jnp.log(jnp.maximum(priors, jnp.finfo(weight.dtype).tiny)),
            labels,
            schema,
            threshold=self.threshold,
            case_shape=batch.case_shape,
        )
        data_valid = jnp.all(
            ~(active[..., None] & batch.feature_mask) | raw_finite, axis=(-2, -1)
        )
        domain_valid = data_valid & jnp.all(feature_mass > 0.0, axis=(-2, -1))
        return _result(
            model,
            batch=batch,
            weight=weight,
            mass=mass,
            feature_mass=feature_mass,
            domain_valid=domain_valid,
            method="bernoulli-nb",
        )


def _fit_multinomial(recipe: Any, batch: MLBatch, *, complement: bool) -> FitResult:
    x, y, weight, active, labels, schema = _prepare(
        batch, recipe.num_classes, recipe.weight_policy
    )
    values = jnp.asarray(x)
    if not (
        jnp.issubdtype(values.dtype, jnp.integer)
        or jnp.issubdtype(values.dtype, jnp.floating)
    ):
        raise TypeError(
            "Multinomial and complement Naive Bayes require real count features."
        )
    entry_active = active[..., :, None] & batch.feature_mask
    domain_entry = (~entry_active) | (jnp.isfinite(values) & (values >= 0.0))
    safe_values = jnp.where(
        entry_active & jnp.isfinite(values) & (values >= 0.0), values, 0.0
    )
    class_weight, mass = _membership(y, weight, int(labels.shape[0]))
    counts = jnp.einsum("...nc,...nf->...cf", class_weight, safe_values)
    feature_mass = counts
    if complement:
        counts = jnp.sum(counts, axis=-2, keepdims=True) - counts
    smoothed = counts + recipe.alpha
    probability = smoothed / jnp.sum(smoothed, axis=-1, keepdims=True)
    priors = _priors(mass, recipe.class_prior)
    model = MultinomialNaiveBayesModel(
        jnp.log(probability),
        jnp.log(jnp.maximum(priors, jnp.finfo(weight.dtype).tiny)),
        labels,
        schema,
        complement=complement,
        case_shape=batch.case_shape,
    )
    domain_valid = jnp.all(domain_entry, axis=(-2, -1))
    return _result(
        model,
        batch=batch,
        weight=weight,
        mass=mass,
        feature_mass=feature_mass,
        domain_valid=domain_valid,
        method="complement-nb" if complement else "multinomial-nb",
    )


class MultinomialNaiveBayesRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    class_prior: tuple[float, ...] = eqx.field(static=True)
    alpha: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        class_prior: tuple[float, ...] = (),
        alpha: float = 1.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.num_classes, self.class_prior, self.weight_policy = _validate_common(
            num_classes, class_prior, weight_policy
        )
        self.alpha = jnp.asarray(alpha, dtype=float)
        if self.alpha.ndim != 0 or float(self.alpha) <= 0.0:
            raise ValueError("alpha must be positive.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_multinomial(self, batch, complement=False)


class ComplementNaiveBayesRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    class_prior: tuple[float, ...] = eqx.field(static=True)
    alpha: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        class_prior: tuple[float, ...] = (),
        alpha: float = 1.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.num_classes, self.class_prior, self.weight_policy = _validate_common(
            num_classes, class_prior, weight_policy
        )
        self.alpha = jnp.asarray(alpha, dtype=float)
        if self.alpha.ndim != 0 or float(self.alpha) <= 0.0:
            raise ValueError("alpha must be positive.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_multinomial(self, batch, complement=True)


class CategoricalNaiveBayesRecipe(AbstractRecipe):
    category_counts: tuple[int, ...] = eqx.field(static=True)
    num_classes: int | None = eqx.field(static=True)
    class_prior: tuple[float, ...] = eqx.field(static=True)
    alpha: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        category_counts: tuple[int, ...],
        /,
        *,
        num_classes: int | None = None,
        class_prior: tuple[float, ...] = (),
        alpha: float = 1.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.category_counts = tuple(int(count) for count in category_counts)
        if not self.category_counts or any(count < 2 for count in self.category_counts):
            raise ValueError(
                "Each categorical feature must declare at least two categories."
            )
        self.num_classes, self.class_prior, self.weight_policy = _validate_common(
            num_classes, class_prior, weight_policy
        )
        self.alpha = jnp.asarray(alpha, dtype=float)
        if self.alpha.ndim != 0 or float(self.alpha) <= 0.0:
            raise ValueError("alpha must be positive.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if len(self.category_counts) != batch.feature_count:
            raise ValueError("category_counts must align with the feature axis.")
        x, y, weight, active, labels, schema = _prepare(
            batch, self.num_classes, self.weight_policy
        )
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError("Categorical Naive Bayes requires real category codes.")
        values = jnp.asarray(x, dtype=jnp.int32)
        classes = int(labels.shape[0])
        max_categories = max(self.category_counts)
        class_weight, mass = _membership(y, weight, classes)
        tables = []
        masses = []
        domain = jnp.ones(batch.case_shape, dtype=bool)
        for feature, categories in enumerate(self.category_counts):
            raw = x[..., feature]
            valid = (
                batch.feature_mask[..., feature]
                & jnp.isfinite(raw)
                & (raw >= 0)
                & (raw < categories)
                & (raw == jnp.floor(raw))
            )
            domain = domain & jnp.all((~active) | valid, axis=-1)
            membership = jax.nn.one_hot(
                jnp.clip(values[..., feature], 0, categories - 1),
                max_categories,
                dtype=weight.dtype,
            )
            feature_weight = class_weight * valid[..., :, None]
            counts = jnp.einsum("...nc,...nk->...ck", feature_weight, membership)
            category_mask = jnp.arange(max_categories) < categories
            smoothed = jnp.where(category_mask, counts + self.alpha, 0.0)
            probability = smoothed / jnp.sum(smoothed, axis=-1, keepdims=True)
            tables.append(
                jnp.where(
                    category_mask,
                    jnp.log(jnp.maximum(probability, jnp.finfo(weight.dtype).tiny)),
                    0.0,
                )
            )
            masses.append(jnp.sum(feature_weight, axis=-2))
        table = jnp.stack(tables, axis=-2)
        feature_mass = jnp.stack(masses, axis=-1)
        priors = _priors(mass, self.class_prior)
        model = CategoricalNaiveBayesModel(
            table,
            jnp.log(jnp.maximum(priors, jnp.finfo(weight.dtype).tiny)),
            labels,
            schema,
            category_counts=self.category_counts,
            case_shape=batch.case_shape,
        )
        domain_valid = domain & jnp.all(feature_mass > 0.0, axis=(-2, -1))
        return _result(
            model,
            batch=batch,
            weight=weight,
            mass=mass,
            feature_mass=feature_mass,
            domain_valid=domain_valid,
            method="categorical-nb",
        )


__all__ = [
    "AbstractNaiveBayesModel",
    "BernoulliNaiveBayesModel",
    "BernoulliNaiveBayesRecipe",
    "CategoricalNaiveBayesModel",
    "CategoricalNaiveBayesRecipe",
    "ComplementNaiveBayesRecipe",
    "GaussianNaiveBayesModel",
    "GaussianNaiveBayesRecipe",
    "MultinomialNaiveBayesModel",
    "MultinomialNaiveBayesRecipe",
    "NaiveBayesDiagnostics",
]
