#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Number
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_RANK_DEFICIENT,
    ML_SUCCESS,
)
from .._numerics import effective_sample_size
from .._schema import TargetSchema


class DiscriminantDiagnostics(StrictModule):
    """Auditable class-mass and covariance diagnostics for a discriminant fit."""

    valid: Array
    status: Array
    effective_samples: Array
    class_mass: Array
    absent_classes: Array
    covariance_rank: Array
    covariance_condition: Array
    minimum_eigenvalue: Array
    maximum_eigenvalue: Array
    log_determinant: Array
    raw_singular: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        effective_samples: Any,
        class_mass: Any,
        covariance_rank: Any,
        covariance_condition: Any,
        minimum_eigenvalue: Any,
        maximum_eigenvalue: Any,
        log_determinant: Any,
        raw_singular: Any,
        method: str,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.effective_samples = jnp.asarray(effective_samples)
        self.class_mass = jnp.asarray(class_mass)
        self.absent_classes = self.class_mass <= 0.0
        self.covariance_rank = jnp.asarray(covariance_rank, dtype=jnp.int32)
        self.covariance_condition = jnp.asarray(covariance_condition)
        self.minimum_eigenvalue = jnp.asarray(minimum_eigenvalue)
        self.maximum_eigenvalue = jnp.asarray(maximum_eigenvalue)
        self.log_determinant = jnp.asarray(log_determinant)
        self.raw_singular = jnp.asarray(raw_singular, dtype=bool)
        self.method = str(method)


def _labels_for(batch: MLBatch, num_classes: int | None) -> tuple[Array, TargetSchema]:
    labels = batch.target_schema.class_labels
    if labels:
        if num_classes is not None and len(labels) != num_classes:
            raise ValueError("num_classes conflicts with target_schema.class_labels.")
        count = len(labels)
        schema = batch.target_schema
    else:
        if num_classes is None or int(num_classes) < 2:
            raise ValueError(
                "Classification requires target_schema.class_labels or num_classes >= 2."
            )
        count = int(num_classes)
        labels = tuple(range(count))
        kind: Literal["binary", "multiclass"] = "binary" if count == 2 else "multiclass"
        schema = TargetSchema(kind, names=batch.target_schema.names, class_labels=labels)
    encoded_labels = (
        jnp.asarray(labels)
        if all(isinstance(label, Number) for label in labels)
        else jnp.arange(count, dtype=jnp.int32)
    )
    if encoded_labels.ndim != 1 or encoded_labels.shape[0] < 2:
        raise ValueError("Class labels must form a one-dimensional vocabulary.")
    return encoded_labels, schema


def _training_arrays(
    batch: MLBatch, labels: Array, policy: WeightPolicy
) -> tuple[Array, Array, Array, Array]:
    x = batch.dense_features()
    y = batch.require_targets()
    if batch.target_shape != ():
        raise ValueError(
            "Discriminant analysis requires one scalar class label per sample."
        )
    if not (
        jnp.issubdtype(x.dtype, jnp.floating)
        or jnp.issubdtype(x.dtype, jnp.complexfloating)
    ):
        x = x.astype(jnp.float32)
    matched = y[..., None] == labels
    encoded = jnp.argmax(matched, axis=-1).astype(jnp.int32)
    known = jnp.any(matched, axis=-1)
    target_ok = (
        batch.target_mask if batch.target_mask is not None else jnp.ones_like(known)
    )
    feature_ok = jnp.all(batch.feature_mask, axis=-1)
    finite_x = jnp.all(jnp.isfinite(jnp.real(x)) & jnp.isfinite(jnp.imag(x)), axis=-1)
    raw_weight = batch.effective_weight(policy)
    weight_valid = jnp.isfinite(raw_weight) & (raw_weight >= 0.0)
    vocabulary_valid = jnp.all(~(batch.sample_mask & target_ok) | known, axis=-1)
    data_valid = jnp.all(
        ~(batch.sample_mask & target_ok & feature_ok) | finite_x, axis=-1
    )
    case_finite = jnp.all(weight_valid, axis=-1) & vocabulary_valid & data_valid
    active = batch.sample_mask & known & target_ok & feature_ok & finite_x & weight_valid
    weight = jnp.where(active, raw_weight, 0.0)
    safe_x = jnp.where(active[..., None], x, 0)
    return safe_x, encoded, weight, case_finite


def _class_statistics(
    x: Array, y: Array, weight: Array, classes: int
) -> tuple[Array, Array, Array, Array]:
    membership = jax.nn.one_hot(y, classes, dtype=weight.dtype)
    class_weight = weight[..., :, None] * membership
    mass = jnp.sum(class_weight, axis=-2)
    tiny = jnp.finfo(weight.dtype).tiny
    means = oe.contract("...nc,...nf->...cf", class_weight, x)
    means = means / jnp.maximum(mass[..., :, None], tiny)
    centered = x[..., :, None, :] - means[..., None, :, :]
    scatter = oe.contract(
        "...nc,...ncf,...ncg->...cfg",
        class_weight,
        jnp.conj(centered),
        centered,
    )
    total = jnp.sum(mass, axis=-1, keepdims=True)
    priors = mass / jnp.maximum(total, tiny)
    return mass, means, scatter, priors


def _regularize(
    covariance: Array, shrinkage: float, regularization: float
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    feature_count = covariance.shape[-1]
    hermitian = (covariance + jnp.swapaxes(jnp.conj(covariance), -1, -2)) * 0.5
    raw_eigenvalues = jnp.linalg.eigvalsh(hermitian)
    scale = jnp.real(jnp.trace(hermitian, axis1=-2, axis2=-1)) / feature_count
    eye = jnp.eye(feature_count, dtype=hermitian.dtype)
    adjusted = (
        (1.0 - shrinkage) * hermitian
        + shrinkage * scale[..., None, None] * eye
        + regularization * eye
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(adjusted)
    largest = jnp.max(jnp.abs(eigenvalues), axis=-1)
    tolerance = (
        jnp.maximum(largest, 1.0) * jnp.finfo(eigenvalues.dtype).eps * feature_count
    )
    positive = eigenvalues > tolerance[..., None]
    rank = jnp.sum(positive, axis=-1, dtype=jnp.int32)
    safe_eigenvalues = jnp.maximum(eigenvalues, tolerance[..., None])
    inverse = oe.contract(
        "...ik,...k,...jk->...ij",
        eigenvectors,
        1.0 / safe_eigenvalues,
        jnp.conj(eigenvectors),
    )
    condition = jnp.max(safe_eigenvalues, axis=-1) / jnp.min(safe_eigenvalues, axis=-1)
    logdet = jnp.sum(jnp.log(safe_eigenvalues), axis=-1)
    raw_tolerance = jnp.maximum(jnp.max(jnp.abs(raw_eigenvalues), axis=-1), 1.0)
    raw_tolerance = raw_tolerance * jnp.finfo(raw_eigenvalues.dtype).eps * feature_count
    raw_singular = jnp.min(raw_eigenvalues, axis=-1) <= raw_tolerance
    return adjusted, inverse, eigenvalues, rank, condition, logdet, raw_singular


def _reshape_for_samples(
    parameter: Array, case_shape: tuple[int, ...], extra: int
) -> Array:
    if extra <= 0:
        return parameter
    return parameter.reshape(
        case_shape + (1,) * extra + parameter.shape[len(case_shape) :]
    )


class LinearDiscriminantModel(AbstractArrayModel):
    """Fitted shared-covariance Gaussian discriminant classifier."""

    coefficients: Array
    intercepts: Array
    labels: Array
    target_schema: TargetSchema
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        coefficients: Array,
        intercepts: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        case_shape: tuple[int, ...],
    ):
        self.coefficients = jnp.asarray(coefficients)
        self.intercepts = jnp.asarray(intercepts)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.case_shape = tuple(case_shape)
        self.in_size = int(self.coefficients.shape[-1])
        self.out_size = int(self.coefficients.shape[-2])

    def decision_function(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        if values.shape[-1] != self.in_size:
            raise ValueError(
                f"Expected {self.in_size} input features; got {values.shape[-1]}."
            )
        extra = values.ndim - len(self.case_shape) - 1
        coefficients = _reshape_for_samples(self.coefficients, self.case_shape, extra)
        intercepts = _reshape_for_samples(self.intercepts, self.case_shape, extra)
        return (
            jnp.real(oe.contract("...f,...cf->...c", jnp.conj(values), coefficients))
            + intercepts
        )

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


class QuadraticDiscriminantModel(AbstractArrayModel):
    """Fitted class-specific covariance Gaussian discriminant classifier."""

    means: Array
    precisions: Array
    log_priors: Array
    log_determinants: Array
    labels: Array
    target_schema: TargetSchema
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        means: Array,
        precisions: Array,
        log_priors: Array,
        log_determinants: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        case_shape: tuple[int, ...],
    ):
        self.means = jnp.asarray(means)
        self.precisions = jnp.asarray(precisions)
        self.log_priors = jnp.asarray(log_priors)
        self.log_determinants = jnp.asarray(log_determinants)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.case_shape = tuple(case_shape)
        self.in_size = int(self.means.shape[-1])
        self.out_size = int(self.means.shape[-2])

    def decision_function(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        if values.shape[-1] != self.in_size:
            raise ValueError(
                f"Expected {self.in_size} input features; got {values.shape[-1]}."
            )
        extra = values.ndim - len(self.case_shape) - 1
        means = _reshape_for_samples(self.means, self.case_shape, extra)
        precisions = _reshape_for_samples(self.precisions, self.case_shape, extra)
        log_priors = _reshape_for_samples(self.log_priors, self.case_shape, extra)
        logdet = _reshape_for_samples(self.log_determinants, self.case_shape, extra)
        difference = values[..., None, :] - means
        quadratic = jnp.real(
            oe.contract(
                "...cf,...cfg,...cg->...c", jnp.conj(difference), precisions, difference
            )
        )
        return log_priors - 0.5 * (logdet + quadratic)

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


def _fit_discriminant(
    recipe: Any,
    batch: MLBatch,
    *,
    quadratic: bool,
    method: str,
) -> FitResult:
    labels, schema = _labels_for(batch, recipe.num_classes)
    classes = int(labels.shape[0])
    x, y, weight, case_finite = _training_arrays(batch, labels, recipe.weight_policy)
    mass, means, scatter, empirical_priors = _class_statistics(x, y, weight, classes)
    tiny = jnp.finfo(weight.dtype).tiny
    if recipe.priors:
        priors = jnp.asarray(recipe.priors, dtype=weight.dtype)
        priors = jnp.broadcast_to(priors, mass.shape)
    else:
        priors = empirical_priors
    absent = mass <= 0.0
    finite = (
        case_finite
        & jnp.all(
            jnp.isfinite(jnp.real(means)) & jnp.isfinite(jnp.imag(means)), axis=(-2, -1)
        )
        & jnp.all(jnp.isfinite(priors), axis=-1)
    )
    priors_valid = jnp.all(priors > 0.0, axis=-1) & jnp.isclose(
        jnp.sum(priors, axis=-1), 1.0, rtol=1e-5, atol=1e-7
    )
    if quadratic:
        covariance = scatter / jnp.maximum(mass[..., :, None, None], tiny)
    else:
        pooled = jnp.sum(scatter, axis=-3)
        covariance = pooled / jnp.maximum(jnp.sum(mass, axis=-1)[..., None, None], tiny)
    adjusted, inverse, eigenvalues, rank, condition, logdet, raw_singular = _regularize(
        covariance, recipe.shrinkage, recipe.regularization
    )
    del adjusted
    resolved = rank == batch.feature_count
    enough = ~jnp.any(absent, axis=-1)
    if quadratic:
        covariance_finite = jnp.all(jnp.isfinite(eigenvalues), axis=(-2, -1))
        resolved_case = jnp.all(resolved, axis=-1)
    else:
        covariance_finite = jnp.all(jnp.isfinite(eigenvalues), axis=-1)
        resolved_case = resolved
    valid = finite & priors_valid & covariance_finite & enough & resolved_case
    status = jnp.where(
        ~enough,
        ML_INSUFFICIENT_DATA,
        jnp.where(
            ~finite | ~covariance_finite | ~priors_valid,
            ML_NONFINITE,
            jnp.where(~resolved_case, ML_RANK_DEFICIENT, ML_SUCCESS),
        ),
    ).astype(jnp.int32)
    log_priors = jnp.log(jnp.maximum(priors, tiny))
    if quadratic:
        model = QuadraticDiscriminantModel(
            means,
            inverse,
            log_priors,
            logdet,
            labels,
            schema,
            case_shape=batch.case_shape,
        )
    else:
        coefficients = oe.contract("...fg,...cg->...cf", inverse, means)
        norm = jnp.real(oe.contract("...cf,...cf->...c", jnp.conj(means), coefficients))
        intercepts = log_priors - 0.5 * norm
        model = LinearDiscriminantModel(
            coefficients, intercepts, labels, schema, case_shape=batch.case_shape
        )
    diagnostics = DiscriminantDiagnostics(
        valid=valid,
        status=status,
        effective_samples=effective_sample_size(weight),
        class_mass=mass,
        covariance_rank=rank,
        covariance_condition=condition,
        minimum_eigenvalue=jnp.min(eigenvalues, axis=-1),
        maximum_eigenvalue=jnp.max(eigenvalues, axis=-1),
        log_determinant=logdet,
        raw_singular=raw_singular,
        method=method,
    )
    contract = GradientContract(
        prediction_inputs="smooth",
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
            "nonsingular regularized covariance",
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


def _validate_recipe(
    num_classes: int | None,
    priors: tuple[float, ...],
    shrinkage: Any,
    regularization: Any,
    weight_policy: WeightPolicy,
) -> tuple[int | None, tuple[float, ...], Array, Array, WeightPolicy]:
    classes = None if num_classes is None else int(num_classes)
    if classes is not None and classes < 2:
        raise ValueError("num_classes must be at least two.")
    prior_values = tuple(float(value) for value in priors)
    if prior_values:
        if classes is not None and len(prior_values) != classes:
            raise ValueError("priors must align with num_classes.")
        if (
            any(value <= 0.0 for value in prior_values)
            or abs(sum(prior_values) - 1.0) > 1e-6
        ):
            raise ValueError("priors must be positive and sum to one.")
    shrink = jnp.asarray(shrinkage, dtype=float)
    ridge = jnp.asarray(regularization, dtype=float)
    if shrink.ndim != 0 or not 0.0 <= float(shrink) <= 1.0:
        raise ValueError("shrinkage must be a scalar in [0, 1].")
    if ridge.ndim != 0 or float(ridge) < 0.0:
        raise ValueError("regularization must be a nonnegative scalar.")
    if weight_policy not in {"none", "statistical", "measure", "product"}:
        raise ValueError("Unsupported weight policy.")
    return classes, prior_values, shrink, ridge, weight_policy


class LinearDiscriminantRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    priors: tuple[float, ...] = eqx.field(static=True)
    shrinkage: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        priors: tuple[float, ...] = (),
        shrinkage: float = 0.0,
        regularization: float = 0.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        (
            self.num_classes,
            self.priors,
            self.shrinkage,
            self.regularization,
            self.weight_policy,
        ) = _validate_recipe(
            num_classes, priors, shrinkage, regularization, weight_policy
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_discriminant(self, batch, quadratic=False, method="weighted-lda")


class QuadraticDiscriminantRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    priors: tuple[float, ...] = eqx.field(static=True)
    shrinkage: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        priors: tuple[float, ...] = (),
        shrinkage: float = 0.0,
        regularization: float = 0.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        (
            self.num_classes,
            self.priors,
            self.shrinkage,
            self.regularization,
            self.weight_policy,
        ) = _validate_recipe(
            num_classes, priors, shrinkage, regularization, weight_policy
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_discriminant(self, batch, quadratic=True, method="weighted-qda")


class ShrinkageDiscriminantRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    priors: tuple[float, ...] = eqx.field(static=True)
    shrinkage: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        shrinkage: float = 0.1,
        num_classes: int | None = None,
        priors: tuple[float, ...] = (),
        regularization: float = 0.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        (
            self.num_classes,
            self.priors,
            self.shrinkage,
            self.regularization,
            self.weight_policy,
        ) = _validate_recipe(
            num_classes, priors, shrinkage, regularization, weight_policy
        )
        if float(self.shrinkage) <= 0.0:
            raise ValueError("ShrinkageDiscriminantRecipe requires positive shrinkage.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_discriminant(self, batch, quadratic=False, method="shrinkage-lda")


class RegularizedDiscriminantRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    priors: tuple[float, ...] = eqx.field(static=True)
    shrinkage: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        regularization: float = 1e-6,
        shrinkage: float = 0.0,
        num_classes: int | None = None,
        priors: tuple[float, ...] = (),
        weight_policy: WeightPolicy = "statistical",
    ):
        (
            self.num_classes,
            self.priors,
            self.shrinkage,
            self.regularization,
            self.weight_policy,
        ) = _validate_recipe(
            num_classes, priors, shrinkage, regularization, weight_policy
        )
        if float(self.regularization) <= 0.0:
            raise ValueError(
                "RegularizedDiscriminantRecipe requires positive regularization."
            )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_discriminant(self, batch, quadratic=True, method="regularized-qda")


__all__ = [
    "DiscriminantDiagnostics",
    "LinearDiscriminantModel",
    "LinearDiscriminantRecipe",
    "QuadraticDiscriminantModel",
    "QuadraticDiscriminantRecipe",
    "RegularizedDiscriminantRecipe",
    "ShrinkageDiscriminantRecipe",
]
