#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._batch import MLBatch, WeightPolicy
from .._contracts import AbstractRecipe, FitResult
from .._numerics import group_soft_threshold, soft_threshold
from ._base import (
    AbstractLinearRegressorModel,
    design_matmul,
    design_row_norm_bound,
    design_transpose_matmul,
    iterative_fit,
    parameter_dtype,
    prepare_supervised,
    unrolled_contract,
)


class LassoModel(AbstractLinearRegressorModel):
    """Fitted elementwise L1-regularized linear model."""


class ElasticNetModel(AbstractLinearRegressorModel):
    """Fitted combined L1/L2-regularized linear model."""


class GroupLassoModel(AbstractLinearRegressorModel):
    """Fitted non-overlapping feature-group lasso model."""


class SparseGroupLassoModel(AbstractLinearRegressorModel):
    """Fitted sparse-group lasso model."""


def _positive_scalar(value: ArrayLike, name: str, /, *, allow_zero: bool = True) -> Array:
    result = jnp.asarray(value)
    if result.weak_type:
        result = result.astype(jnp.float32)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    invalid = ~jnp.isfinite(result) | ((result < 0.0) if allow_zero else (result <= 0.0))
    relation = "non-negative" if allow_zero else "positive"
    return eqx.error_if(result, invalid, f"{name} must be {relation}.")


def _group_prox(
    coefficients: Array,
    threshold: Array,
    groups: tuple[int, ...],
    /,
) -> Array:
    result = coefficients
    for group in sorted(set(groups)):
        indices = tuple(index for index, value in enumerate(groups) if value == group)
        block = result[:, indices, :]
        shrunk = group_soft_threshold(
            block.reshape((block.shape[0], -1)), threshold, axis=-1
        ).reshape(block.shape)
        result = result.at[:, indices, :].set(shrunk)
    return result


def _group_penalty(coefficients: Array, groups: tuple[int, ...], /) -> Array:
    value = jnp.zeros((coefficients.shape[0],), dtype=jnp.real(coefficients).dtype)
    for group in sorted(set(groups)):
        indices = tuple(index for index, item in enumerate(groups) if item == group)
        value = value + jnp.sqrt(
            jnp.sum(jnp.abs(coefficients[:, indices, :]) ** 2, axis=(1, 2))
        )
    return value


def _fit_penalized(
    batch: MLBatch,
    /,
    *,
    l1: Array,
    l2: Array,
    group_strength: Array,
    feature_groups: tuple[int, ...] | None,
    fit_intercept: bool,
    regularize_intercept: bool,
    learning_rate: Array | None,
    max_iterations: int,
    tolerance: float,
    weight_policy: WeightPolicy,
    method: str,
    model_type: type[AbstractLinearRegressorModel],
) -> FitResult:
    prepared = prepare_supervised(batch, weight_policy=weight_policy)
    features = prepared.design.features
    if feature_groups is not None and len(feature_groups) != features:
        raise ValueError("feature_groups must contain one group id per feature.")
    dtype = jnp.result_type(parameter_dtype(prepared), l1, l2, group_strength)
    if learning_rate is not None:
        dtype = jnp.result_type(dtype, learning_rate)
    coefficients = jnp.zeros(
        (prepared.targets.shape[0], features, prepared.outputs), dtype=dtype
    )
    if fit_intercept:
        mass = jnp.sum(prepared.weights, axis=1)
        intercept = jnp.where(
            mass > 0.0,
            jnp.sum(prepared.weights * prepared.targets, axis=1)
            / jnp.maximum(mass, jnp.finfo(mass.dtype).tiny),
            0,
        )
    else:
        intercept = jnp.zeros((prepared.targets.shape[0], prepared.outputs), dtype=dtype)
    intercept = intercept.astype(dtype)

    if learning_rate is None:
        augmented_norm = design_row_norm_bound(prepared.design) + float(fit_intercept)
        lipschitz = jnp.max(
            jnp.sum(prepared.weights * augmented_norm[..., None], axis=1)
        ) + l2 * (1.0 + float(regularize_intercept))
        step_size = 1.0 / jnp.maximum(lipschitz, jnp.finfo(lipschitz.dtype).tiny)
    else:
        step_size = learning_rate

    def objective(beta, bias):
        residual = (
            design_matmul(prepared.design, beta) + bias[:, None, :] - prepared.targets
        )
        value = 0.5 * jnp.sum(
            prepared.weights * jnp.real(residual * jnp.conj(residual)), axis=(1, 2)
        )
        value = value + 0.5 * l2 * jnp.sum(jnp.abs(beta) ** 2, axis=(1, 2))
        if regularize_intercept:
            value = value + 0.5 * l2 * jnp.sum(jnp.abs(bias) ** 2, axis=1)
        value = value + l1 * jnp.sum(jnp.abs(beta), axis=(1, 2))
        if feature_groups is not None:
            value = value + group_strength * _group_penalty(beta, feature_groups)
        return value

    def step(state, iteration):
        del iteration
        beta, bias = state
        residual = (
            design_matmul(prepared.design, beta) + bias[:, None, :] - prepared.targets
        )
        weighted = prepared.weights * residual
        beta_candidate = beta - step_size * (
            design_transpose_matmul(prepared.design, weighted) + l2 * beta
        )
        bias_gradient = jnp.sum(weighted, axis=1)
        if regularize_intercept:
            bias_gradient = bias_gradient + l2 * bias
        bias_candidate = bias - step_size * bias_gradient if fit_intercept else bias
        beta_candidate = soft_threshold(beta_candidate, step_size * l1)
        if feature_groups is not None:
            beta_candidate = _group_prox(
                beta_candidate, step_size * group_strength, feature_groups
            )
        residual_norm = jnp.maximum(
            jnp.max(jnp.abs(beta_candidate - beta)),
            jnp.max(jnp.abs(bias_candidate - bias)),
        )
        value = jnp.sum(objective(beta_candidate, bias_candidate))
        return (beta_candidate, bias_candidate), value, residual_norm

    return iterative_fit(
        prepared,
        step=step,
        initial=(coefficients, intercept),
        max_iterations=max_iterations,
        tolerance=tolerance,
        method=method,
        objective=objective,
        model_factory=lambda beta, bias: model_type(
            beta,
            bias,
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
        ),
        gradient_contract=unrolled_contract(nonsmooth=True),
    )


class LassoRecipe(AbstractRecipe):
    """Weighted multi-output lasso solved by fixed proximal-gradient iterations."""

    alpha: Array
    learning_rate: Array | None
    fit_intercept: bool = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        alpha: ArrayLike = 1.0,
        *,
        fit_intercept: bool = True,
        learning_rate: ArrayLike | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.alpha = _positive_scalar(alpha, "alpha")
        self.learning_rate = (
            None
            if learning_rate is None
            else _positive_scalar(learning_rate, "learning_rate", allow_zero=False)
        )
        self.fit_intercept = bool(fit_intercept)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_penalized(
            batch,
            l1=self.alpha,
            l2=jnp.asarray(0.0),
            group_strength=jnp.asarray(0.0),
            feature_groups=None,
            fit_intercept=self.fit_intercept,
            regularize_intercept=False,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            weight_policy=self.weight_policy,
            method="weighted-lasso-fixed-proximal-gradient",
            model_type=LassoModel,
        )


class ElasticNetRecipe(AbstractRecipe):
    """Weighted elastic net with independent L1/L2 strengths."""

    l1_strength: Array
    l2_strength: Array
    learning_rate: Array | None
    fit_intercept: bool = eqx.field(static=True)
    regularize_intercept: bool = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        alpha: ArrayLike = 1.0,
        *,
        l1_ratio: ArrayLike = 0.5,
        fit_intercept: bool = True,
        regularize_intercept: bool = False,
        learning_rate: ArrayLike | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        alpha_ = _positive_scalar(alpha, "alpha")
        ratio = jnp.asarray(l1_ratio)
        if ratio.weak_type:
            ratio = ratio.astype(jnp.float32)
        if ratio.ndim != 0:
            raise ValueError("l1_ratio must be a scalar in [0, 1].")
        ratio = eqx.error_if(
            ratio,
            ~jnp.isfinite(ratio) | (ratio < 0.0) | (ratio > 1.0),
            "l1_ratio must be a scalar in [0, 1].",
        )
        self.l1_strength = alpha_ * ratio
        self.l2_strength = alpha_ * (1.0 - ratio)
        self.learning_rate = (
            None
            if learning_rate is None
            else _positive_scalar(learning_rate, "learning_rate", allow_zero=False)
        )
        self.fit_intercept = bool(fit_intercept)
        self.regularize_intercept = bool(regularize_intercept)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_penalized(
            batch,
            l1=self.l1_strength,
            l2=self.l2_strength,
            group_strength=jnp.asarray(0.0),
            feature_groups=None,
            fit_intercept=self.fit_intercept,
            regularize_intercept=self.regularize_intercept,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            weight_policy=self.weight_policy,
            method="weighted-elastic-net-fixed-proximal-gradient",
            model_type=ElasticNetModel,
        )


class GroupLassoRecipe(AbstractRecipe):
    """Weighted non-overlapping group lasso over declared feature groups."""

    alpha: Array
    feature_groups: tuple[int, ...] = eqx.field(static=True)
    learning_rate: Array | None
    fit_intercept: bool = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        feature_groups: tuple[int, ...],
        *,
        alpha: ArrayLike = 1.0,
        fit_intercept: bool = True,
        learning_rate: ArrayLike | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        groups = tuple(int(group) for group in feature_groups)
        if not groups:
            raise ValueError("feature_groups cannot be empty.")
        self.alpha = _positive_scalar(alpha, "alpha")
        self.feature_groups = groups
        self.learning_rate = (
            None
            if learning_rate is None
            else _positive_scalar(learning_rate, "learning_rate", allow_zero=False)
        )
        self.fit_intercept = bool(fit_intercept)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_penalized(
            batch,
            l1=jnp.asarray(0.0),
            l2=jnp.asarray(0.0),
            group_strength=self.alpha,
            feature_groups=self.feature_groups,
            fit_intercept=self.fit_intercept,
            regularize_intercept=False,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            weight_policy=self.weight_policy,
            method="weighted-group-lasso-fixed-proximal-gradient",
            model_type=GroupLassoModel,
        )


class SparseGroupLassoRecipe(AbstractRecipe):
    """Weighted sparse-group lasso with separable L1 and group penalties."""

    l1_strength: Array
    group_strength: Array
    feature_groups: tuple[int, ...] = eqx.field(static=True)
    learning_rate: Array | None
    fit_intercept: bool = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        feature_groups: tuple[int, ...],
        *,
        alpha: ArrayLike = 1.0,
        l1_ratio: ArrayLike = 0.5,
        fit_intercept: bool = True,
        learning_rate: ArrayLike | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        groups = tuple(int(group) for group in feature_groups)
        if not groups:
            raise ValueError("feature_groups cannot be empty.")
        alpha_ = _positive_scalar(alpha, "alpha")
        ratio = jnp.asarray(l1_ratio)
        if ratio.weak_type:
            ratio = ratio.astype(jnp.float32)
        if ratio.ndim != 0:
            raise ValueError("l1_ratio must be a scalar in [0, 1].")
        ratio = eqx.error_if(
            ratio,
            ~jnp.isfinite(ratio) | (ratio < 0.0) | (ratio > 1.0),
            "l1_ratio must be a scalar in [0, 1].",
        )
        self.l1_strength = alpha_ * ratio
        self.group_strength = alpha_ * (1.0 - ratio)
        self.feature_groups = groups
        self.learning_rate = (
            None
            if learning_rate is None
            else _positive_scalar(learning_rate, "learning_rate", allow_zero=False)
        )
        self.fit_intercept = bool(fit_intercept)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_penalized(
            batch,
            l1=self.l1_strength,
            l2=jnp.asarray(0.0),
            group_strength=self.group_strength,
            feature_groups=self.feature_groups,
            fit_intercept=self.fit_intercept,
            regularize_intercept=False,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            weight_policy=self.weight_policy,
            method="weighted-sparse-group-lasso-fixed-proximal-gradient",
            model_type=SparseGroupLassoModel,
        )


__all__ = [
    "ElasticNetRecipe",
    "ElasticNetModel",
    "GroupLassoRecipe",
    "GroupLassoModel",
    "LassoRecipe",
    "LassoModel",
    "SparseGroupLassoRecipe",
    "SparseGroupLassoModel",
]
