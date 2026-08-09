#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._exponential_family import BernoulliFamily, PoissonFamily
from .._batch import MLBatch, WeightPolicy
from .._contracts import AbstractRecipe, FitResult
from ._base import (
    AbstractGeneralizedLinearModel,
    binary_targets,
    design_matmul,
    design_row_norm_bound,
    design_transpose_matmul,
    iterative_fit,
    LogisticClassifierModel,
    multinomial_targets,
    MultinomialLogisticModel,
    parameter_dtype,
    prepare_supervised,
    unrolled_contract,
)


_BERNOULLI_FAMILY = BernoulliFamily()
_POISSON_FAMILY = PoissonFamily()


def _scalar(value: ArrayLike, name: str, /, *, positive: bool = False) -> Array:
    result = jnp.asarray(value)
    if result.weak_type:
        result = result.astype(jnp.float32)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    invalid = ~jnp.isfinite(result) | ((result <= 0.0) if positive else (result < 0.0))
    qualifier = "positive" if positive else "non-negative"
    return eqx.error_if(result, invalid, f"{name} must be {qualifier}.")


class PoissonModel(AbstractGeneralizedLinearModel):
    """Fitted log-link Poisson mean model."""


class GammaModel(AbstractGeneralizedLinearModel):
    """Fitted log-link Gamma mean model."""


class TweedieModel(AbstractGeneralizedLinearModel):
    """Fitted log-link Tweedie mean model."""

    power: float = eqx.field(static=True)

    def __init__(self, *args, power: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.power = float(power)


def _fit_binary_logistic(
    recipe,
    batch: MLBatch,
    /,
) -> FitResult:
    prepared = prepare_supervised(
        batch, weight_policy=recipe.weight_policy, require_real=True
    )
    targets, labels, label_valid = binary_targets(prepared, batch.target_schema)
    cases = prepared.targets.shape[0]
    features = prepared.design.features
    outputs = prepared.outputs
    dtype = jnp.result_type(parameter_dtype(prepared), recipe.l2_strength)
    if recipe.learning_rate is not None:
        dtype = jnp.result_type(dtype, recipe.learning_rate)
    coefficients = jnp.zeros((cases, features, outputs), dtype=dtype)
    mass = jnp.sum(prepared.weights, axis=1)
    proportion = jnp.where(
        mass > 0.0,
        jnp.sum(prepared.weights * targets, axis=1)
        / jnp.maximum(mass, jnp.finfo(mass.dtype).tiny),
        0.5,
    )
    eps = jnp.finfo(dtype).eps
    intercept = (
        jnp.log(jnp.clip(proportion, eps, 1.0 - eps))
        - jnp.log1p(-jnp.clip(proportion, eps, 1.0 - eps))
        if recipe.fit_intercept
        else jnp.zeros((cases, outputs), dtype=dtype)
    )
    intercept = intercept.astype(dtype)
    if recipe.learning_rate is None:
        bound = design_row_norm_bound(prepared.design) + float(recipe.fit_intercept)
        lipschitz = (
            0.25 * jnp.max(jnp.sum(prepared.weights * bound[..., None], axis=1))
            + recipe.l2_strength
        )
        learning_rate = 1.0 / jnp.maximum(lipschitz, jnp.finfo(dtype).tiny)
    else:
        learning_rate = recipe.learning_rate

    def objective(beta, bias):
        scores = design_matmul(prepared.design, beta) + bias[:, None, :]
        natural = _BERNOULLI_FAMILY.natural(scores[..., None])
        loss = _BERNOULLI_FAMILY.canonical_loss(natural, targets)
        return jnp.sum(
            prepared.weights * loss, axis=(1, 2)
        ) + 0.5 * recipe.l2_strength * jnp.sum(beta * beta, axis=(1, 2))

    def step(state, iteration):
        del iteration
        beta, bias = state
        scores = design_matmul(prepared.design, beta) + bias[:, None, :]
        natural = _BERNOULLI_FAMILY.natural(scores[..., None])
        derivative = (
            prepared.weights * _BERNOULLI_FAMILY.canonical_score(natural, targets)[..., 0]
        )
        beta_candidate = beta - learning_rate * (
            design_transpose_matmul(prepared.design, derivative)
            + recipe.l2_strength * beta
        )
        bias_candidate = (
            bias - learning_rate * jnp.sum(derivative, axis=1)
            if recipe.fit_intercept
            else bias
        )
        residual = jnp.maximum(
            jnp.max(jnp.abs(beta_candidate - beta)),
            jnp.max(jnp.abs(bias_candidate - bias)),
        )
        return (
            (
                beta_candidate,
                bias_candidate,
            ),
            jnp.sum(objective(beta_candidate, bias_candidate)),
            residual,
        )

    return iterative_fit(
        prepared,
        step=step,
        initial=(coefficients, intercept),
        max_iterations=recipe.max_iterations,
        tolerance=recipe.tolerance,
        method="weighted-binary-logistic-fixed-gradient",
        objective=objective,
        model_factory=lambda beta, bias: LogisticClassifierModel(
            beta,
            bias,
            labels,
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
        ),
        gradient_contract=unrolled_contract(
            fit_targets="none", hard_outputs=("predict", "predict_indices")
        ),
        extra_valid=label_valid,
    )


class LogisticRegressionRecipe(AbstractRecipe):
    """Weighted binary or multilabel logistic regression with smooth probabilities."""

    l2_strength: Array
    learning_rate: Array | None
    fit_intercept: bool = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        l2_strength: ArrayLike = 0.0,
        fit_intercept: bool = True,
        learning_rate: ArrayLike | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.l2_strength = _scalar(l2_strength, "l2_strength")
        self.learning_rate = (
            None
            if learning_rate is None
            else _scalar(learning_rate, "learning_rate", positive=True)
        )
        self.fit_intercept = bool(fit_intercept)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_binary_logistic(self, batch)


class MultinomialLogisticRegressionRecipe(AbstractRecipe):
    """Weighted multiclass softmax regression with an identified zero-mean parameterization."""

    l2_strength: Array
    learning_rate: Array | None
    num_classes: int | None = eqx.field(static=True)
    fit_intercept: bool = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        num_classes: int | None = None,
        *,
        l2_strength: ArrayLike = 0.0,
        fit_intercept: bool = True,
        learning_rate: ArrayLike | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.l2_strength = _scalar(l2_strength, "l2_strength")
        self.learning_rate = (
            None
            if learning_rate is None
            else _scalar(learning_rate, "learning_rate", positive=True)
        )
        self.num_classes = None if num_classes is None else int(num_classes)
        self.fit_intercept = bool(fit_intercept)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        prepared = prepare_supervised(
            batch, weight_policy=self.weight_policy, require_real=True
        )
        targets, labels, label_valid = multinomial_targets(
            prepared, batch.target_schema, num_classes=self.num_classes
        )
        classes = int(labels.shape[0])
        cases = prepared.targets.shape[0]
        features = prepared.design.features
        dtype = jnp.result_type(parameter_dtype(prepared), self.l2_strength)
        if self.learning_rate is not None:
            dtype = jnp.result_type(dtype, self.learning_rate)
        coefficients = jnp.zeros((cases, features, classes), dtype=dtype)
        sample_weight = prepared.weights[..., 0]
        one_hot = jax.nn.one_hot(targets, classes, dtype=dtype)
        class_mass = jnp.sum(sample_weight[..., None] * one_hot, axis=1)
        class_valid = jnp.all(class_mass > 0.0, axis=-1)
        if self.fit_intercept:
            intercept = jnp.log(jnp.maximum(class_mass, jnp.finfo(dtype).tiny))
            intercept = intercept - jnp.mean(intercept, axis=-1, keepdims=True)
        else:
            intercept = jnp.zeros((cases, classes), dtype=dtype)
        intercept = intercept.astype(dtype)
        if self.learning_rate is None:
            bound = design_row_norm_bound(prepared.design) + float(self.fit_intercept)
            lipschitz = jnp.max(jnp.sum(sample_weight * bound, axis=1)) + self.l2_strength
            learning_rate = 1.0 / jnp.maximum(lipschitz, jnp.finfo(dtype).tiny)
        else:
            learning_rate = self.learning_rate

        def objective(beta, bias):
            scores = design_matmul(prepared.design, beta) + bias[:, None, :]
            loss = (
                jax.nn.logsumexp(scores, axis=-1)
                - jnp.take_along_axis(scores, targets[..., None], axis=-1)[..., 0]
            )
            return jnp.sum(
                sample_weight * loss, axis=1
            ) + 0.5 * self.l2_strength * jnp.sum(beta * beta, axis=(1, 2))

        def step(state, iteration):
            del iteration
            beta, bias = state
            scores = design_matmul(prepared.design, beta) + bias[:, None, :]
            derivative = sample_weight[..., None] * (
                jax.nn.softmax(scores, axis=-1) - one_hot
            )
            beta_candidate = beta - learning_rate * (
                design_transpose_matmul(prepared.design, derivative)
                + self.l2_strength * beta
            )
            bias_candidate = (
                bias - learning_rate * jnp.sum(derivative, axis=1)
                if self.fit_intercept
                else bias
            )
            beta_candidate = beta_candidate - jnp.mean(
                beta_candidate, axis=-1, keepdims=True
            )
            bias_candidate = bias_candidate - jnp.mean(
                bias_candidate, axis=-1, keepdims=True
            )
            residual = jnp.maximum(
                jnp.max(jnp.abs(beta_candidate - beta)),
                jnp.max(jnp.abs(bias_candidate - bias)),
            )
            return (
                (
                    beta_candidate,
                    bias_candidate,
                ),
                jnp.sum(objective(beta_candidate, bias_candidate)),
                residual,
            )

        return iterative_fit(
            prepared,
            step=step,
            initial=(coefficients, intercept),
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            method="weighted-multinomial-logistic-fixed-gradient",
            objective=objective,
            model_factory=lambda beta, bias: MultinomialLogisticModel(
                beta,
                bias,
                labels,
                case_shape=prepared.case_shape,
            ),
            gradient_contract=unrolled_contract(
                fit_targets="none", hard_outputs=("predict", "predict_indices")
            ),
            extra_valid=label_valid & class_valid,
        )


def _fit_log_glm(recipe, batch: MLBatch, /, *, family: str, power: float) -> FitResult:
    prepared = prepare_supervised(
        batch, weight_policy=recipe.weight_policy, require_real=True
    )
    cases = prepared.targets.shape[0]
    features = prepared.design.features
    outputs = prepared.outputs
    dtype = jnp.result_type(
        parameter_dtype(prepared), recipe.l2_strength, recipe.learning_rate
    )
    coefficients = jnp.zeros((cases, features, outputs), dtype=dtype)
    mass = jnp.sum(prepared.weights, axis=1)
    mean = jnp.where(
        mass > 0.0,
        jnp.sum(prepared.weights * prepared.targets, axis=1)
        / jnp.maximum(mass, jnp.finfo(mass.dtype).tiny),
        1.0,
    )
    intercept = (
        jnp.log(jnp.maximum(mean, jnp.finfo(dtype).tiny))
        if recipe.fit_intercept
        else jnp.zeros((cases, outputs), dtype=dtype)
    )
    intercept = intercept.astype(dtype)
    learning_rate = recipe.learning_rate
    active = prepared.weights > 0.0
    if family == "poisson":
        domain = _POISSON_FAMILY.sufficient_statistics(prepared.targets).valid
        family_targets = jnp.where(domain, prepared.targets, 0.0)
    elif family == "gamma":
        domain = prepared.targets > 0.0
        family_targets = prepared.targets
    else:
        domain = prepared.targets > 0.0 if power == 2.0 else prepared.targets >= 0.0
        family_targets = prepared.targets
    domain_valid = jnp.all(domain | (~active), axis=(1, 2))

    def loss_and_derivative(eta):
        if family == "poisson":
            natural = _POISSON_FAMILY.natural(eta[..., None])
            loss = _POISSON_FAMILY.canonical_loss(natural, family_targets)
            derivative = _POISSON_FAMILY.canonical_score(natural, family_targets)[..., 0]
            return loss, derivative
        if family == "gamma":
            inverse_mean = jnp.exp(-eta)
            return (
                prepared.targets * inverse_mean + eta,
                1.0 - prepared.targets * inverse_mean,
            )
        if power == 1.0:
            mean_value = jnp.exp(eta)
            return mean_value - prepared.targets * eta, mean_value - prepared.targets
        if power == 2.0:
            inverse_mean = jnp.exp(-eta)
            return (
                prepared.targets * inverse_mean + eta,
                1.0 - prepared.targets * inverse_mean,
            )
        first = jnp.exp((2.0 - power) * eta)
        second = jnp.exp((1.0 - power) * eta)
        loss = first / (2.0 - power) - prepared.targets * second / (1.0 - power)
        derivative = first - prepared.targets * second
        return loss, derivative

    def objective(beta, bias):
        eta = design_matmul(prepared.design, beta) + bias[:, None, :]
        loss, _ = loss_and_derivative(eta)
        return jnp.sum(
            prepared.weights * loss, axis=(1, 2)
        ) + 0.5 * recipe.l2_strength * jnp.sum(beta * beta, axis=(1, 2))

    def step(state, iteration):
        del iteration
        beta, bias = state
        eta = design_matmul(prepared.design, beta) + bias[:, None, :]
        _, derivative = loss_and_derivative(eta)
        weighted = prepared.weights * derivative
        beta_candidate = beta - learning_rate * (
            design_transpose_matmul(prepared.design, weighted) + recipe.l2_strength * beta
        )
        bias_candidate = (
            bias - learning_rate * jnp.sum(weighted, axis=1)
            if recipe.fit_intercept
            else bias
        )
        residual = jnp.maximum(
            jnp.max(jnp.abs(beta_candidate - beta)),
            jnp.max(jnp.abs(bias_candidate - bias)),
        )
        return (
            (
                beta_candidate,
                bias_candidate,
            ),
            jnp.sum(objective(beta_candidate, bias_candidate)),
            residual,
        )

    model_type = {"poisson": PoissonModel, "gamma": GammaModel}.get(family, TweedieModel)

    def make_model(beta, bias):
        kwargs = dict(
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
            inverse_link="exp",
        )
        if family == "tweedie":
            return model_type(beta, bias, power=power, **kwargs)
        return model_type(beta, bias, **kwargs)

    return iterative_fit(
        prepared,
        step=step,
        initial=(coefficients, intercept),
        max_iterations=recipe.max_iterations,
        tolerance=recipe.tolerance,
        method=f"weighted-{family}-log-link-fixed-gradient",
        objective=objective,
        model_factory=make_model,
        gradient_contract=unrolled_contract(),
        extra_valid=domain_valid,
    )


class _AbstractLogGLMRecipe(AbstractRecipe):
    l2_strength: Array
    learning_rate: Array
    fit_intercept: bool = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        l2_strength: ArrayLike,
        fit_intercept: bool,
        learning_rate: ArrayLike,
        max_iterations: int,
        tolerance: float,
        weight_policy: WeightPolicy,
    ):
        self.l2_strength = _scalar(l2_strength, "l2_strength")
        self.learning_rate = _scalar(learning_rate, "learning_rate", positive=True)
        self.fit_intercept = bool(fit_intercept)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weight_policy = weight_policy


class PoissonRegressorRecipe(_AbstractLogGLMRecipe):
    """Weighted Poisson GLM with canonical log link."""

    def __init__(
        self,
        *,
        l2_strength: ArrayLike = 0.0,
        fit_intercept: bool = True,
        learning_rate: ArrayLike = 1e-3,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        super().__init__(
            l2_strength=l2_strength,
            fit_intercept=fit_intercept,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            weight_policy=weight_policy,
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_log_glm(self, batch, family="poisson", power=1.0)


class GammaRegressorRecipe(_AbstractLogGLMRecipe):
    """Weighted Gamma GLM with a log mean link."""

    def __init__(
        self,
        *,
        l2_strength: ArrayLike = 0.0,
        fit_intercept: bool = True,
        learning_rate: ArrayLike = 1e-3,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        super().__init__(
            l2_strength=l2_strength,
            fit_intercept=fit_intercept,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            weight_policy=weight_policy,
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_log_glm(self, batch, family="gamma", power=2.0)


class TweedieRegressorRecipe(_AbstractLogGLMRecipe):
    """Weighted log-link Tweedie GLM for powers in the compound Poisson-Gamma range."""

    power: float = eqx.field(static=True)

    def __init__(
        self,
        power: float = 1.5,
        *,
        l2_strength: ArrayLike = 0.0,
        fit_intercept: bool = True,
        learning_rate: ArrayLike = 1e-3,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        power_ = float(power)
        if not 1.0 <= power_ <= 2.0:
            raise ValueError("This Tweedie implementation supports powers in [1, 2].")
        super().__init__(
            l2_strength=l2_strength,
            fit_intercept=fit_intercept,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            weight_policy=weight_policy,
        )
        self.power = power_

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_log_glm(self, batch, family="tweedie", power=self.power)


__all__ = [
    "GammaModel",
    "GammaRegressorRecipe",
    "LogisticClassifierModel",
    "LogisticRegressionRecipe",
    "MultinomialLogisticModel",
    "MultinomialLogisticRegressionRecipe",
    "PoissonModel",
    "PoissonRegressorRecipe",
    "TweedieModel",
    "TweedieRegressorRecipe",
]
