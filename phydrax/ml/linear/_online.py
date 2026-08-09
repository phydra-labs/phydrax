#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    ML_INFEASIBLE,
    ML_NONFINITE,
    ML_SUCCESS,
)
from ._base import (
    AbstractLinearRegressorModel,
    AbstractLinearScoreClassifierModel,
    binary_targets,
    design_matmul,
    parameter_dtype,
    prepare_supervised,
    PreparedBatch,
    restore_case_shape,
    unrolled_contract,
    weighted_rank_condition,
)


class SGDRegressorModel(AbstractLinearRegressorModel):
    """Fitted sequential stochastic-gradient regressor."""


class PassiveAggressiveRegressorModel(AbstractLinearRegressorModel):
    """Fitted passive-aggressive online regressor."""


class AbstractOnlineClassifierModel(AbstractLinearScoreClassifierModel):
    """Online binary classifier with scores and optional logistic probabilities."""

    probabilistic: bool = eqx.field(static=True)

    def __init__(self, *args, probabilistic: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.probabilistic = bool(probabilistic)

    def positive_probability(self, x: Any, /) -> Array:
        if not self.probabilistic:
            raise ValueError(
                "This margin classifier does not define calibrated probabilities."
            )
        return jax.nn.sigmoid(self.decision_function(x))

    def predict_proba(self, x: Any, /) -> Array:
        positive = self.positive_probability(x)
        return jnp.stack((1.0 - positive, positive), axis=-1)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return (
            self.positive_probability(x)
            if self.probabilistic
            else self.decision_function(x)
        )


class OnlineClassifierModel(AbstractOnlineClassifierModel):
    """Generic online binary classifier."""


class SGDClassifierModel(AbstractOnlineClassifierModel):
    """Fitted logistic or hinge SGD classifier."""


class PerceptronModel(AbstractOnlineClassifierModel):
    """Fitted hard-mistake perceptron score model."""


class PassiveAggressiveClassifierModel(AbstractOnlineClassifierModel):
    """Fitted passive-aggressive margin classifier."""


def _scalar(value: ArrayLike, name: str, /, *, positive: bool = False) -> Array:
    result = jnp.asarray(value)
    if result.weak_type:
        result = result.astype(jnp.float32)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    invalid = ~jnp.isfinite(result) | ((result <= 0.0) if positive else (result < 0.0))
    qualifier = "positive" if positive else "non-negative"
    return eqx.error_if(result, invalid, f"{name} must be {qualifier}.")


def _orders(
    prepared: PreparedBatch,
    /,
    *,
    passes: int,
    shuffle: bool,
    key: Any,
) -> Array:
    cases = prepared.targets.shape[0]
    samples = prepared.design.samples
    if not shuffle:
        return jnp.broadcast_to(
            jnp.arange(samples, dtype=jnp.int32), (passes, cases, samples)
        )
    if key is None:
        raise ValueError("Shuffled online fitting requires an explicit JAX key.")
    keys = jax.random.split(key, passes)
    return jax.vmap(
        lambda pass_key: jnp.argsort(
            jax.random.uniform(pass_key, (cases, samples)), axis=-1
        )
    )(keys)


def _online_score(prepared: PreparedBatch, beta: Array, indices: Array) -> Array:
    cases = beta.shape[0]
    row = jnp.arange(cases)
    design = prepared.design
    if not design.sparse:
        assert design.dense is not None
        values = design.dense[row, indices]
        return jnp.einsum("cf,cfo->co", values, beta)
    assert design.values is not None
    assert design.indices is not None
    assert design.entry_valid is not None
    values = design.values[row, indices]
    columns = design.indices[row, indices]
    valid = design.entry_valid[row, indices]
    gathered = beta[row[:, None], columns]
    return jnp.sum(jnp.where(valid[..., None], values[..., None] * gathered, 0), axis=1)


def _online_update(
    prepared: PreparedBatch,
    beta: Array,
    indices: Array,
    direction: Array,
) -> Array:
    cases = beta.shape[0]
    row = jnp.arange(cases)
    design = prepared.design
    if not design.sparse:
        assert design.dense is not None
        values = design.dense[row, indices]
        return beta + values[..., None] * direction[:, None, :]
    assert design.values is not None
    assert design.indices is not None
    assert design.entry_valid is not None
    entries = design.values[row, indices]
    columns = design.indices[row, indices]
    valid = design.entry_valid[row, indices]

    def one(current, values, positions, keep, update):
        increments = jnp.where(keep[..., None], values[..., None] * update[None, :], 0)
        return current.at[positions].add(increments)

    return jax.vmap(one)(beta, entries, columns, valid, direction)


def _finish_online(
    prepared: PreparedBatch,
    beta: Array,
    bias: Array,
    /,
    *,
    objective: Array,
    passes: int,
    method: str,
    model,
    extra_valid: Array | bool = True,
    nonsmooth: bool = False,
    fit_targets: str | None = None,
    hard_outputs: tuple[str, ...] = (),
) -> FitResult:
    finite = (
        jnp.all(jnp.isfinite(beta), axis=(1, 2))
        & jnp.all(jnp.isfinite(bias), axis=1)
        & jnp.isfinite(objective)
    )
    extra = jnp.broadcast_to(jnp.asarray(extra_valid, dtype=bool), finite.shape)
    valid = prepared.data_valid & extra & finite
    status = jnp.where(
        ~prepared.data_valid,
        prepared.data_status,
        jnp.where(
            ~extra,
            ML_INFEASIBLE,
            jnp.where(finite, ML_SUCCESS, ML_NONFINITE),
        ),
    ).astype(jnp.int32)
    rank, condition = weighted_rank_condition(prepared.design, prepared.weights)
    valid_cases = restore_case_shape(prepared, valid)
    status_cases = restore_case_shape(prepared, status)
    diagnostics = FitDiagnostics(
        valid=valid_cases,
        status=status_cases,
        objective=restore_case_shape(prepared, objective),
        iterations=restore_case_shape(
            prepared, jnp.asarray(passes * prepared.design.samples)
        ),
        effective_samples=restore_case_shape(prepared, prepared.effective_samples),
        rank=restore_case_shape(prepared, rank),
        condition=restore_case_shape(prepared, condition),
        method=method,
    )
    return FitResult(
        model,
        diagnostics,
        valid=valid_cases,
        status=status_cases,
        method=method,
        gradient_contract=unrolled_contract(
            nonsmooth=nonsmooth,
            fit_targets=fit_targets,
            hard_outputs=hard_outputs,
        ),
    )


class SGDRegressorRecipe(AbstractRecipe):
    """Sequential weighted squared-loss SGD with optional elastic regularization."""

    learning_rate: Array
    l1_strength: Array
    l2_strength: Array
    fit_intercept: bool = eqx.field(static=True)
    passes: int = eqx.field(static=True)
    shuffle: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        learning_rate: ArrayLike = 1e-2,
        l1_strength: ArrayLike = 0.0,
        l2_strength: ArrayLike = 0.0,
        fit_intercept: bool = True,
        passes: int = 10,
        shuffle: bool = True,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(passes) <= 0:
            raise ValueError("passes must be positive.")
        self.learning_rate = _scalar(learning_rate, "learning_rate", positive=True)
        self.l1_strength = _scalar(l1_strength, "l1_strength")
        self.l2_strength = _scalar(l2_strength, "l2_strength")
        self.fit_intercept = bool(fit_intercept)
        self.passes = int(passes)
        self.shuffle = bool(shuffle)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        prepared = prepare_supervised(batch, weight_policy=self.weight_policy)
        cases = prepared.targets.shape[0]
        dtype = jnp.result_type(
            parameter_dtype(prepared),
            self.learning_rate,
            self.l1_strength,
            self.l2_strength,
        )
        beta = jnp.zeros((cases, prepared.design.features, prepared.outputs), dtype=dtype)
        bias = jnp.zeros((cases, prepared.outputs), dtype=dtype)
        order = _orders(
            prepared, passes=self.passes, shuffle=self.shuffle, key=key
        ).reshape((-1, cases))
        row = jnp.arange(cases)

        def transition(state, indices):
            coefficients, intercept = state
            prediction = _online_score(prepared, coefficients, indices) + intercept
            target = prepared.targets[row, indices]
            weight = prepared.weights[row, indices]
            gradient = weight * (prediction - target)
            active_step = jnp.any(weight > 0.0, axis=-1)
            coefficients = coefficients * (
                1.0 - self.learning_rate * self.l2_strength * active_step[:, None, None]
            )
            coefficients = _online_update(
                prepared, coefficients, indices, -self.learning_rate * gradient
            )

            def shrink(current):
                magnitude = jnp.abs(current)
                threshold = (
                    self.learning_rate * self.l1_strength * active_step[:, None, None]
                )
                return current * jnp.maximum(
                    1.0
                    - threshold
                    / jnp.maximum(magnitude, jnp.finfo(jnp.real(current).dtype).tiny),
                    0.0,
                )

            coefficients = jax.lax.cond(
                self.l1_strength == 0.0,
                lambda current: current,
                shrink,
                coefficients,
            )
            intercept = (
                intercept - self.learning_rate * gradient
                if self.fit_intercept
                else intercept
            )
            return (coefficients, intercept), None

        (beta, bias), _ = jax.lax.scan(transition, (beta, bias), order)
        residual = (
            design_matmul(prepared.design, beta) + bias[:, None, :] - prepared.targets
        )
        objective = (
            0.5
            * jnp.sum(
                prepared.weights * jnp.real(residual * jnp.conj(residual)), axis=(1, 2)
            )
            + 0.5 * self.l2_strength * jnp.sum(jnp.abs(beta) ** 2, axis=(1, 2))
            + self.l1_strength * jnp.sum(jnp.abs(beta), axis=(1, 2))
        )
        model = SGDRegressorModel(
            beta,
            bias,
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
        )
        return _finish_online(
            prepared,
            beta,
            bias,
            objective=objective,
            passes=self.passes,
            method="weighted-online-sgd-regression",
            model=model,
            nonsmooth=True,
        )


class SGDClassifierRecipe(AbstractRecipe):
    """Sequential weighted binary/multilabel SGD with logistic or hinge loss."""

    learning_rate: Array
    l2_strength: Array
    loss: Literal["logistic", "hinge"] = eqx.field(static=True)
    fit_intercept: bool = eqx.field(static=True)
    passes: int = eqx.field(static=True)
    shuffle: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        loss: Literal["logistic", "hinge"] = "logistic",
        learning_rate: ArrayLike = 1e-2,
        l2_strength: ArrayLike = 0.0,
        fit_intercept: bool = True,
        passes: int = 10,
        shuffle: bool = True,
        weight_policy: WeightPolicy = "statistical",
    ):
        if loss not in {"logistic", "hinge"}:
            raise ValueError("loss must be 'logistic' or 'hinge'.")
        if int(passes) <= 0:
            raise ValueError("passes must be positive.")
        self.learning_rate = _scalar(learning_rate, "learning_rate", positive=True)
        self.l2_strength = _scalar(l2_strength, "l2_strength")
        self.loss = loss
        self.fit_intercept = bool(fit_intercept)
        self.passes = int(passes)
        self.shuffle = bool(shuffle)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        prepared = prepare_supervised(
            batch, weight_policy=self.weight_policy, require_real=True
        )
        encoded, labels, label_valid = binary_targets(prepared, batch.target_schema)
        signed = 2.0 * encoded - 1.0
        cases = prepared.targets.shape[0]
        dtype = jnp.result_type(
            parameter_dtype(prepared), self.learning_rate, self.l2_strength
        )
        beta = jnp.zeros((cases, prepared.design.features, prepared.outputs), dtype=dtype)
        bias = jnp.zeros((cases, prepared.outputs), dtype=dtype)
        order = _orders(
            prepared, passes=self.passes, shuffle=self.shuffle, key=key
        ).reshape((-1, cases))
        row = jnp.arange(cases)

        def transition(state, indices):
            coefficients, intercept = state
            score = _online_score(prepared, coefficients, indices) + intercept
            target01 = encoded[row, indices]
            target_sign = signed[row, indices]
            weight = prepared.weights[row, indices]
            if self.loss == "logistic":
                gradient = weight * (jax.nn.sigmoid(score) - target01)
            else:
                gradient = jnp.where(
                    target_sign * score < 1.0, -weight * target_sign, 0.0
                )
            active_step = jnp.any(weight > 0.0, axis=-1)
            coefficients = coefficients * (
                1.0 - self.learning_rate * self.l2_strength * active_step[:, None, None]
            )
            coefficients = _online_update(
                prepared, coefficients, indices, -self.learning_rate * gradient
            )
            intercept = (
                intercept - self.learning_rate * gradient
                if self.fit_intercept
                else intercept
            )
            return (coefficients, intercept), None

        (beta, bias), _ = jax.lax.scan(transition, (beta, bias), order)
        scores = design_matmul(prepared.design, beta) + bias[:, None, :]
        if self.loss == "logistic":
            loss = jax.nn.softplus(scores) - encoded * scores
        else:
            loss = jnp.maximum(0.0, 1.0 - signed * scores)
        objective = jnp.sum(
            prepared.weights * loss, axis=(1, 2)
        ) + 0.5 * self.l2_strength * jnp.sum(beta * beta, axis=(1, 2))
        model = SGDClassifierModel(
            beta,
            bias,
            labels,
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
            probabilistic=self.loss == "logistic",
        )
        return _finish_online(
            prepared,
            beta,
            bias,
            objective=objective,
            passes=self.passes,
            method=f"weighted-online-sgd-{self.loss}-classification",
            fit_targets="none",
            model=model,
            extra_valid=label_valid,
            nonsmooth=self.loss == "hinge",
            hard_outputs=("predict", "predict_indices"),
        )


class PerceptronRecipe(AbstractRecipe):
    """Weighted online binary/multilabel perceptron with exact hard mistake updates."""

    learning_rate: Array
    fit_intercept: bool = eqx.field(static=True)
    passes: int = eqx.field(static=True)
    shuffle: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        learning_rate: ArrayLike = 1.0,
        fit_intercept: bool = True,
        passes: int = 10,
        shuffle: bool = True,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(passes) <= 0:
            raise ValueError("passes must be positive.")
        self.learning_rate = _scalar(learning_rate, "learning_rate", positive=True)
        self.fit_intercept = bool(fit_intercept)
        self.passes = int(passes)
        self.shuffle = bool(shuffle)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        prepared = prepare_supervised(
            batch, weight_policy=self.weight_policy, require_real=True
        )
        encoded, labels, label_valid = binary_targets(prepared, batch.target_schema)
        signed = 2.0 * encoded - 1.0
        cases = prepared.targets.shape[0]
        dtype = jnp.result_type(parameter_dtype(prepared), self.learning_rate)
        beta = jnp.zeros((cases, prepared.design.features, prepared.outputs), dtype=dtype)
        bias = jnp.zeros((cases, prepared.outputs), dtype=dtype)
        order = _orders(
            prepared, passes=self.passes, shuffle=self.shuffle, key=key
        ).reshape((-1, cases))
        row = jnp.arange(cases)

        def transition(state, indices):
            coefficients, intercept = state
            score = _online_score(prepared, coefficients, indices) + intercept
            target = signed[row, indices]
            weight = prepared.weights[row, indices]
            update = jnp.where(
                target * score <= 0.0, self.learning_rate * weight * target, 0.0
            )
            coefficients = _online_update(prepared, coefficients, indices, update)
            intercept = intercept + update if self.fit_intercept else intercept
            return (coefficients, intercept), None

        (beta, bias), _ = jax.lax.scan(transition, (beta, bias), order)
        scores = design_matmul(prepared.design, beta) + bias[:, None, :]
        objective = jnp.sum(
            prepared.weights * jnp.maximum(0.0, -signed * scores), axis=(1, 2)
        )
        model = PerceptronModel(
            beta,
            bias,
            labels,
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
        )
        return _finish_online(
            prepared,
            beta,
            bias,
            objective=objective,
            passes=self.passes,
            method="weighted-online-perceptron",
            fit_targets="none",
            model=model,
            extra_valid=label_valid,
            nonsmooth=True,
            hard_outputs=("predict", "predict_indices", "mistake_updates"),
        )


class _AbstractPassiveAggressiveRecipe(AbstractRecipe):
    aggressiveness: Array
    variant: Literal["pa1", "pa2"] = eqx.field(static=True)
    fit_intercept: bool = eqx.field(static=True)
    passes: int = eqx.field(static=True)
    shuffle: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        aggressiveness: ArrayLike,
        variant: Literal["pa1", "pa2"],
        fit_intercept: bool,
        passes: int,
        shuffle: bool,
        weight_policy: WeightPolicy,
    ):
        if variant not in {"pa1", "pa2"}:
            raise ValueError("variant must be 'pa1' or 'pa2'.")
        if int(passes) <= 0:
            raise ValueError("passes must be positive.")
        self.aggressiveness = _scalar(aggressiveness, "aggressiveness", positive=True)
        self.variant = variant
        self.fit_intercept = bool(fit_intercept)
        self.passes = int(passes)
        self.shuffle = bool(shuffle)
        self.weight_policy = weight_policy

    def _tau(self, loss: Array, norm: Array) -> Array:
        if self.variant == "pa1":
            return jnp.minimum(
                self.aggressiveness,
                loss / jnp.maximum(norm, jnp.finfo(norm.dtype).tiny),
            )
        denominator = norm + 1.0 / (2.0 * self.aggressiveness)
        return loss / jnp.maximum(denominator, jnp.finfo(denominator.dtype).tiny)


class PassiveAggressiveRegressorRecipe(_AbstractPassiveAggressiveRecipe):
    """Weighted passive-aggressive epsilon-insensitive online regression."""

    epsilon: Array

    def __init__(
        self,
        *,
        aggressiveness: ArrayLike = 1.0,
        epsilon: ArrayLike = 0.1,
        variant: Literal["pa1", "pa2"] = "pa1",
        fit_intercept: bool = True,
        passes: int = 10,
        shuffle: bool = True,
        weight_policy: WeightPolicy = "statistical",
    ):
        super().__init__(
            aggressiveness=aggressiveness,
            variant=variant,
            fit_intercept=fit_intercept,
            passes=passes,
            shuffle=shuffle,
            weight_policy=weight_policy,
        )
        self.epsilon = _scalar(epsilon, "epsilon")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        prepared = prepare_supervised(
            batch, weight_policy=self.weight_policy, require_real=True
        )
        cases = prepared.targets.shape[0]
        dtype = jnp.result_type(
            parameter_dtype(prepared),
            self.aggressiveness,
            self.epsilon,
        )
        beta = jnp.zeros((cases, prepared.design.features, prepared.outputs), dtype=dtype)
        bias = jnp.zeros((cases, prepared.outputs), dtype=dtype)
        order = _orders(
            prepared, passes=self.passes, shuffle=self.shuffle, key=key
        ).reshape((-1, cases))
        row = jnp.arange(cases)

        def transition(state, indices):
            coefficients, intercept = state
            prediction = _online_score(prepared, coefficients, indices) + intercept
            target = prepared.targets[row, indices]
            weight = prepared.weights[row, indices]
            error = target - prediction
            loss = weight * jnp.maximum(0.0, jnp.abs(error) - self.epsilon)
            row_norm = _row_norm(prepared, indices) + float(self.fit_intercept)
            tau = self._tau(loss, row_norm[:, None])
            update = tau * jnp.sign(error)
            coefficients = _online_update(prepared, coefficients, indices, update)
            intercept = intercept + update if self.fit_intercept else intercept
            return (coefficients, intercept), None

        (beta, bias), _ = jax.lax.scan(transition, (beta, bias), order)
        residual = (
            design_matmul(prepared.design, beta) + bias[:, None, :] - prepared.targets
        )
        objective = jnp.sum(
            prepared.weights * jnp.maximum(0.0, jnp.abs(residual) - self.epsilon),
            axis=(1, 2),
        )
        model = PassiveAggressiveRegressorModel(
            beta,
            bias,
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
        )
        return _finish_online(
            prepared,
            beta,
            bias,
            objective=objective,
            passes=self.passes,
            method=f"weighted-online-passive-aggressive-regression-{self.variant}",
            model=model,
            nonsmooth=True,
        )


class PassiveAggressiveClassifierRecipe(_AbstractPassiveAggressiveRecipe):
    """Weighted passive-aggressive binary/multilabel margin classification."""

    def __init__(
        self,
        *,
        aggressiveness: ArrayLike = 1.0,
        variant: Literal["pa1", "pa2"] = "pa1",
        fit_intercept: bool = True,
        passes: int = 10,
        shuffle: bool = True,
        weight_policy: WeightPolicy = "statistical",
    ):
        super().__init__(
            aggressiveness=aggressiveness,
            variant=variant,
            fit_intercept=fit_intercept,
            passes=passes,
            shuffle=shuffle,
            weight_policy=weight_policy,
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        prepared = prepare_supervised(
            batch, weight_policy=self.weight_policy, require_real=True
        )
        encoded, labels, label_valid = binary_targets(prepared, batch.target_schema)
        signed = 2.0 * encoded - 1.0
        cases = prepared.targets.shape[0]
        dtype = jnp.result_type(parameter_dtype(prepared), self.aggressiveness)
        beta = jnp.zeros((cases, prepared.design.features, prepared.outputs), dtype=dtype)
        bias = jnp.zeros((cases, prepared.outputs), dtype=dtype)
        order = _orders(
            prepared, passes=self.passes, shuffle=self.shuffle, key=key
        ).reshape((-1, cases))
        row = jnp.arange(cases)

        def transition(state, indices):
            coefficients, intercept = state
            score = _online_score(prepared, coefficients, indices) + intercept
            target = signed[row, indices]
            weight = prepared.weights[row, indices]
            loss = weight * jnp.maximum(0.0, 1.0 - target * score)
            row_norm = _row_norm(prepared, indices) + float(self.fit_intercept)
            tau = self._tau(loss, row_norm[:, None])
            update = tau * target
            coefficients = _online_update(prepared, coefficients, indices, update)
            intercept = intercept + update if self.fit_intercept else intercept
            return (coefficients, intercept), None

        (beta, bias), _ = jax.lax.scan(transition, (beta, bias), order)
        scores = design_matmul(prepared.design, beta) + bias[:, None, :]
        objective = jnp.sum(
            prepared.weights * jnp.maximum(0.0, 1.0 - signed * scores), axis=(1, 2)
        )
        model = PassiveAggressiveClassifierModel(
            beta,
            bias,
            labels,
            case_shape=prepared.case_shape,
            target_shape=prepared.target_shape,
        )
        return _finish_online(
            prepared,
            beta,
            bias,
            objective=objective,
            passes=self.passes,
            method=f"weighted-online-passive-aggressive-classification-{self.variant}",
            fit_targets="none",
            model=model,
            extra_valid=label_valid,
            nonsmooth=True,
            hard_outputs=("predict", "predict_indices", "margin_updates"),
        )


def _row_norm(prepared: PreparedBatch, indices: Array) -> Array:
    row = jnp.arange(prepared.targets.shape[0])
    design = prepared.design
    if not design.sparse:
        assert design.dense is not None
        values = design.dense[row, indices]
        return jnp.sum(jnp.abs(values) ** 2, axis=-1)
    assert design.values is not None
    assert design.indices is not None
    assert design.entry_valid is not None
    values = design.values[row, indices]
    columns = design.indices[row, indices]
    valid = design.entry_valid[row, indices]
    same_feature = columns[:, :, None] == columns[:, None, :]
    keep = valid[:, :, None] & valid[:, None, :] & same_feature
    products = jnp.conj(values)[:, :, None] * values[:, None, :]
    return jnp.real(jnp.sum(jnp.where(keep, products, 0), axis=(1, 2)))


__all__ = [
    "OnlineClassifierModel",
    "PassiveAggressiveClassifierRecipe",
    "PassiveAggressiveClassifierModel",
    "PassiveAggressiveRegressorRecipe",
    "PassiveAggressiveRegressorModel",
    "PerceptronRecipe",
    "PerceptronModel",
    "SGDClassifierRecipe",
    "SGDClassifierModel",
    "SGDRegressorRecipe",
    "SGDRegressorModel",
]
