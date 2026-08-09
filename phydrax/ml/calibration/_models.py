#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    DecisionFunctionModel,
    FitResult,
    GradientContract,
    LogProbabilityModel,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import effective_sample_size, run_fixed_iterations
from .._schema import FeatureSchema, TargetSchema
from ..discriminant._models import _labels_for, _reshape_for_samples


class CalibrationDiagnostics(StrictModule):
    """Optimization, class-support, and convergence diagnostics for calibration."""

    valid: Array
    status: Array
    objective: Array
    iterations: Array
    converged: Array
    finite: Array
    effective_samples: Array
    class_mass: Array
    absent_classes: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        objective: Any,
        iterations: Any,
        converged: Any,
        finite: Any,
        effective_samples: Any,
        class_mass: Any,
        method: str,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.objective = jnp.asarray(objective)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.converged = jnp.asarray(converged, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.effective_samples = jnp.asarray(effective_samples)
        self.class_mass = jnp.asarray(class_mass)
        self.absent_classes = self.class_mass <= 0.0
        self.method = str(method)


def _validate_optimization(
    learning_rate: Any,
    max_iterations: int,
    tolerance: float,
    l2: Any,
    policy: WeightPolicy,
) -> tuple[Array, int, float, Array, WeightPolicy]:
    rate = jnp.asarray(learning_rate, dtype=float)
    iterations = int(max_iterations)
    tolerance_ = float(tolerance)
    penalty = jnp.asarray(l2, dtype=float)
    if (
        rate.ndim != 0
        or penalty.ndim != 0
        or float(rate) <= 0.0
        or iterations <= 0
        or tolerance_ < 0.0
        or float(penalty) < 0.0
    ):
        raise ValueError(
            "Optimization requires scalar positive learning_rate/max_iterations and nonnegative tolerance/l2."
        )
    if policy not in {"none", "statistical", "measure", "product"}:
        raise ValueError("Unsupported weight policy.")
    return rate, iterations, tolerance_, penalty, policy


def _prepare(
    batch: MLBatch, num_classes: int | None, policy: WeightPolicy
) -> tuple[Array, Array, Array, Array, Array, TargetSchema]:
    logits = batch.dense_features()
    if jnp.issubdtype(logits.dtype, jnp.complexfloating):
        raise TypeError("Calibration scores must be real-valued logits.")
    if not jnp.issubdtype(logits.dtype, jnp.floating):
        logits = logits.astype(jnp.float32)
    labels, schema = _labels_for(batch, num_classes)
    targets = batch.require_targets()
    if batch.target_shape != ():
        raise ValueError("Calibration requires one scalar class label per sample.")
    matched = targets[..., None] == labels
    encoded = jnp.argmax(matched, axis=-1).astype(jnp.int32)
    known = jnp.any(matched, axis=-1)
    target_valid = (
        batch.target_mask if batch.target_mask is not None else jnp.ones_like(known)
    )
    feature_valid = jnp.all(batch.feature_mask & jnp.isfinite(logits), axis=-1)
    raw_weight = batch.effective_weight(policy)
    weight_valid = jnp.isfinite(raw_weight) & (raw_weight >= 0.0)
    vocabulary_valid = jnp.all(~(batch.sample_mask & target_valid) | known, axis=-1)
    active = batch.sample_mask & known & target_valid & feature_valid & weight_valid
    weight = jnp.where(active, raw_weight, 0.0)
    complete_scores = jnp.all(batch.feature_mask, axis=-1)
    finite_scores = jnp.all(jnp.isfinite(logits), axis=-1)
    data_valid = jnp.all(
        ~(batch.sample_mask & target_valid & complete_scores) | finite_scores, axis=-1
    )
    case_valid = jnp.all(weight_valid, axis=-1) & vocabulary_valid & data_valid
    weight = jnp.where(case_valid[..., None], weight, jnp.nan)
    membership = weight[..., :, None] * jax.nn.one_hot(
        encoded, int(labels.shape[0]), dtype=weight.dtype
    )
    mass = jnp.sum(membership, axis=-2)
    return logits, encoded, weight, mass, labels, schema


def _optimize(
    initial: Any,
    loss,
    *,
    learning_rate: Array,
    max_iterations: int,
    tolerance: float,
    method: str,
):
    value_and_grad = jax.value_and_grad(loss)

    def step(parameters, iteration):
        del iteration
        objective, gradient = value_and_grad(parameters)
        residual = jnp.max(
            jnp.stack(
                tuple(
                    jnp.max(jnp.abs(leaf)) for leaf in jax.tree_util.tree_leaves(gradient)
                )
            )
        )
        candidate = jax.tree_util.tree_map(
            lambda value, derivative: value - learning_rate * derivative,
            parameters,
            gradient,
        )
        return candidate, objective, residual

    return run_fixed_iterations(
        initial, step, max_iterations=max_iterations, tolerance=tolerance, method=method
    )


def _diagnostics(
    optimization, weight: Array, mass: Array, *, method: str
) -> CalibrationDiagnostics:
    absent = jnp.any(mass <= 0.0, axis=-1)
    finite = optimization.finite & jnp.all(jnp.isfinite(mass), axis=-1)
    converged = optimization.converged
    valid = finite & converged & ~absent
    status = jnp.where(
        ~finite,
        ML_NONFINITE,
        jnp.where(
            absent,
            ML_INSUFFICIENT_DATA,
            jnp.where(~converged, ML_NONCONVERGED, ML_SUCCESS),
        ),
    ).astype(jnp.int32)
    return CalibrationDiagnostics(
        valid=valid,
        status=status,
        objective=optimization.objective_history[-1],
        iterations=optimization.iterations,
        converged=converged,
        finite=finite,
        effective_samples=effective_sample_size(weight),
        class_mass=mass,
        method=method,
    )


def _contract(
    *, smooth_inputs: bool = True, fit_mode: str = "unrolled"
) -> GradientContract:
    return GradientContract(
        prediction_inputs="smooth" if smooth_inputs else "none",
        prediction_parameters="smooth" if smooth_inputs else "almost-everywhere",
        fit_features="conditional" if fit_mode != "stopped" else "none",
        fit_targets="none",
        fit_weights="conditional" if fit_mode != "stopped" else "none",
        fit_hyperparameters="conditional" if fit_mode != "stopped" else "none",
        fit_mode=fit_mode,
        nondifferentiable_outputs=("predict", "predict_indices"),
        conditions=(
            "fixed class vocabulary",
            "positive class support",
            "finite calibration scores",
        ),
    )


class PlattCalibrationModel(AbstractArrayModel):
    slope: Array
    intercept: Array
    labels: Array
    target_schema: TargetSchema
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        slope: Array,
        intercept: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        case_shape: tuple[int, ...],
    ):
        self.slope = jnp.asarray(slope)
        self.intercept = jnp.asarray(intercept)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.case_shape = tuple(case_shape)
        self.in_size = 1
        self.out_size = 2

    def decision_function(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        score = values[..., 0] if values.ndim > 0 and values.shape[-1] == 1 else values
        extra = score.ndim - len(self.case_shape)
        slope = _reshape_for_samples(self.slope, self.case_shape, extra)
        intercept = _reshape_for_samples(self.intercept, self.case_shape, extra)
        calibrated = slope * score + intercept
        return jnp.stack((jnp.zeros_like(calibrated), calibrated), axis=-1)

    def predict_proba(self, x: Any, /) -> Array:
        positive = jax.nn.sigmoid(self.decision_function(x)[..., 1])
        return jnp.stack((1.0 - positive, positive), axis=-1)

    def predict_log_proba(self, x: Any, /) -> Array:
        decision = self.decision_function(x)[..., 1]
        return jnp.stack(
            (jax.nn.log_sigmoid(-decision), jax.nn.log_sigmoid(decision)), axis=-1
        )

    def predict_indices(self, x: Any, /) -> Array:
        return (self.decision_function(x)[..., 1] >= 0.0).astype(jnp.int32)

    def predict(self, x: Any, /) -> Array:
        return jnp.take(self.labels, self.predict_indices(x), axis=0)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


class TemperatureCalibrationModel(AbstractArrayModel):
    temperature: Array
    labels: Array
    target_schema: TargetSchema
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        temperature: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        case_shape: tuple[int, ...],
    ):
        self.temperature = jnp.asarray(temperature)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.case_shape = tuple(case_shape)
        self.in_size = int(self.labels.shape[0])
        self.out_size = self.in_size

    def decision_function(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        if values.shape[-1] != self.in_size:
            raise ValueError(
                "Temperature calibration input must align with the class vocabulary."
            )
        extra = values.ndim - len(self.case_shape) - 1
        temperature = _reshape_for_samples(self.temperature, self.case_shape, extra)
        return values / temperature[..., None]

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


class VectorCalibrationModel(AbstractArrayModel):
    scale: Array
    bias: Array
    labels: Array
    target_schema: TargetSchema
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        scale: Array,
        bias: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        case_shape: tuple[int, ...],
    ):
        self.scale = jnp.asarray(scale)
        self.bias = jnp.asarray(bias)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.case_shape = tuple(case_shape)
        self.in_size = int(self.labels.shape[0])
        self.out_size = self.in_size

    def decision_function(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        if values.shape[-1] != self.in_size:
            raise ValueError(
                "Vector calibration input must align with the class vocabulary."
            )
        extra = values.ndim - len(self.case_shape) - 1
        return values * _reshape_for_samples(
            self.scale, self.case_shape, extra
        ) + _reshape_for_samples(self.bias, self.case_shape, extra)

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


class MatrixCalibrationModel(AbstractArrayModel):
    matrix: Array
    bias: Array
    labels: Array
    target_schema: TargetSchema
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        matrix: Array,
        bias: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        case_shape: tuple[int, ...],
    ):
        self.matrix = jnp.asarray(matrix)
        self.bias = jnp.asarray(bias)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.case_shape = tuple(case_shape)
        self.in_size = int(self.labels.shape[0])
        self.out_size = self.in_size

    def decision_function(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        if values.shape[-1] != self.in_size:
            raise ValueError(
                "Matrix calibration input must align with the class vocabulary."
            )
        extra = values.ndim - len(self.case_shape) - 1
        matrix = _reshape_for_samples(self.matrix, self.case_shape, extra)
        bias = _reshape_for_samples(self.bias, self.case_shape, extra)
        return jnp.einsum("...f,...cf->...c", values, matrix) + bias

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


class MulticlassCalibrationModel(AbstractArrayModel):
    slope: Array
    intercept: Array
    labels: Array
    target_schema: TargetSchema
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        slope: Array,
        intercept: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        case_shape: tuple[int, ...],
    ):
        self.slope = jnp.asarray(slope)
        self.intercept = jnp.asarray(intercept)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.case_shape = tuple(case_shape)
        self.in_size = int(labels.shape[0])
        self.out_size = self.in_size

    def decision_function(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        if values.shape[-1] != self.in_size:
            raise ValueError(
                "Multiclass calibration input must align with the class vocabulary."
            )
        extra = values.ndim - len(self.case_shape) - 1
        return values * _reshape_for_samples(
            self.slope, self.case_shape, extra
        ) + _reshape_for_samples(self.intercept, self.case_shape, extra)

    def predict_proba(self, x: Any, /) -> Array:
        positive = jax.nn.sigmoid(self.decision_function(x))
        return positive / jnp.maximum(
            jnp.sum(positive, axis=-1, keepdims=True), jnp.finfo(positive.dtype).tiny
        )

    def predict_log_proba(self, x: Any, /) -> Array:
        return jnp.log(self.predict_proba(x))

    def predict_indices(self, x: Any, /) -> Array:
        return jnp.argmax(self.decision_function(x), axis=-1)

    def predict(self, x: Any, /) -> Array:
        return jnp.take(self.labels, self.predict_indices(x), axis=0)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


def _fit_smooth(recipe: Any, batch: MLBatch, *, kind: str) -> FitResult:
    expected_classes = 2 if kind == "platt" else recipe.num_classes
    logits, encoded, weight, mass, labels, schema = _prepare(
        batch, expected_classes, recipe.weight_policy
    )
    classes = int(labels.shape[0])
    tiny = jnp.finfo(weight.dtype).tiny
    denominator = jnp.maximum(jnp.sum(weight), tiny)
    one_hot = jax.nn.one_hot(encoded, classes, dtype=logits.dtype)
    case_shape = batch.case_shape
    if kind == "platt":
        if batch.feature_count != 1:
            raise ValueError("Platt calibration requires exactly one score feature.")
        scores = logits[..., 0]
        initial = (
            jnp.ones(case_shape, dtype=logits.dtype),
            jnp.zeros(case_shape, dtype=logits.dtype),
        )

        def loss(parameters):
            slope, intercept = parameters
            calibrated = slope[..., None] * scores + intercept[..., None]
            objective = (
                jnp.sum(
                    weight * (jax.nn.softplus(calibrated) - one_hot[..., 1] * calibrated)
                )
                / denominator
            )
            return objective + recipe.l2 * (
                jnp.sum((slope - 1.0) ** 2) + jnp.sum(intercept**2)
            )

    elif kind == "temperature":
        if batch.feature_count != classes:
            raise ValueError("Temperature calibration needs one logit per class.")
        initial_temperature = jnp.maximum(
            1.0 - recipe.minimum_temperature, jnp.finfo(logits.dtype).eps
        )
        raw_initial = jnp.log(jnp.expm1(initial_temperature))
        initial = (jnp.broadcast_to(raw_initial, case_shape),)

        def loss(parameters):
            (raw_temperature,) = parameters
            temperature = jax.nn.softplus(raw_temperature) + recipe.minimum_temperature
            calibrated = logits / temperature[..., None, None]
            return (
                -jnp.sum(
                    weight
                    * jnp.sum(one_hot * jax.nn.log_softmax(calibrated, axis=-1), axis=-1)
                )
                / denominator
            )

    elif kind in {"vector", "multiclass"}:
        if batch.feature_count != classes:
            raise ValueError("Vector/multiclass calibration needs one score per class.")
        initial = (
            jnp.ones(case_shape + (classes,), dtype=logits.dtype),
            jnp.zeros(case_shape + (classes,), dtype=logits.dtype),
        )

        def loss(parameters):
            scale, bias = parameters
            calibrated = logits * scale[..., None, :] + bias[..., None, :]
            if kind == "multiclass":
                binary_loss = jax.nn.softplus(calibrated) - one_hot * calibrated
                objective = jnp.sum(weight[..., None] * binary_loss) / denominator
            else:
                objective = (
                    -jnp.sum(
                        weight
                        * jnp.sum(
                            one_hot * jax.nn.log_softmax(calibrated, axis=-1), axis=-1
                        )
                    )
                    / denominator
                )
            return objective + recipe.l2 * (
                jnp.sum((scale - 1.0) ** 2) + jnp.sum(bias**2)
            )

    elif kind == "matrix":
        if batch.feature_count != classes:
            raise ValueError("Matrix calibration needs one score per class.")
        identity = jnp.broadcast_to(
            jnp.eye(classes, dtype=logits.dtype), case_shape + (classes, classes)
        )
        initial = (identity, jnp.zeros(case_shape + (classes,), dtype=logits.dtype))

        def loss(parameters):
            matrix, bias = parameters
            calibrated = (
                jnp.einsum("...nf,...cf->...nc", logits, matrix) + bias[..., None, :]
            )
            objective = (
                -jnp.sum(
                    weight
                    * jnp.sum(one_hot * jax.nn.log_softmax(calibrated, axis=-1), axis=-1)
                )
                / denominator
            )
            return objective + recipe.l2 * (
                jnp.sum((matrix - identity) ** 2) + jnp.sum(bias**2)
            )

    else:
        raise ValueError(f"Unknown smooth calibration kind {kind!r}.")
    optimization = _optimize(
        initial,
        loss,
        learning_rate=recipe.learning_rate,
        max_iterations=recipe.max_iterations,
        tolerance=recipe.tolerance,
        method=kind,
    )
    if kind == "platt":
        model: AbstractArrayModel = PlattCalibrationModel(
            *optimization.value, labels, schema, case_shape=case_shape
        )
    elif kind == "temperature":
        temperature = jax.nn.softplus(optimization.value[0]) + recipe.minimum_temperature
        model = TemperatureCalibrationModel(
            temperature, labels, schema, case_shape=case_shape
        )
    elif kind == "vector":
        model = VectorCalibrationModel(
            *optimization.value, labels, schema, case_shape=case_shape
        )
    elif kind == "matrix":
        model = MatrixCalibrationModel(
            *optimization.value, labels, schema, case_shape=case_shape
        )
    else:
        model = MulticlassCalibrationModel(
            *optimization.value, labels, schema, case_shape=case_shape
        )
    diagnostics = _diagnostics(optimization, weight, mass, method=kind)
    return FitResult(
        model,
        diagnostics,
        valid=diagnostics.valid,
        status=diagnostics.status,
        method=kind,
        gradient_contract=_contract(),
    )


class PlattCalibrationRecipe(AbstractRecipe):
    learning_rate: Array
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    l2: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        max_iterations: int = 256,
        tolerance: float = 1e-6,
        l2: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        (
            self.learning_rate,
            self.max_iterations,
            self.tolerance,
            self.l2,
            self.weight_policy,
        ) = _validate_optimization(
            learning_rate, max_iterations, tolerance, l2, weight_policy
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_smooth(self, batch, kind="platt")


class TemperatureCalibrationRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    minimum_temperature: Array
    learning_rate: Array
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    l2: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        minimum_temperature: float = 1e-4,
        learning_rate: float = 0.05,
        max_iterations: int = 256,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.num_classes = None if num_classes is None else int(num_classes)
        self.minimum_temperature = jnp.asarray(minimum_temperature, dtype=float)
        (
            self.learning_rate,
            self.max_iterations,
            self.tolerance,
            self.l2,
            self.weight_policy,
        ) = _validate_optimization(
            learning_rate, max_iterations, tolerance, 0.0, weight_policy
        )
        if self.minimum_temperature.ndim != 0 or float(self.minimum_temperature) <= 0.0:
            raise ValueError("minimum_temperature must be positive.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_smooth(self, batch, kind="temperature")


class VectorCalibrationRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    learning_rate: Array
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    l2: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        learning_rate: float = 0.02,
        max_iterations: int = 512,
        tolerance: float = 1e-6,
        l2: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.num_classes = None if num_classes is None else int(num_classes)
        (
            self.learning_rate,
            self.max_iterations,
            self.tolerance,
            self.l2,
            self.weight_policy,
        ) = _validate_optimization(
            learning_rate, max_iterations, tolerance, l2, weight_policy
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_smooth(self, batch, kind="vector")


class MatrixCalibrationRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    learning_rate: Array
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    l2: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        learning_rate: float = 0.01,
        max_iterations: int = 768,
        tolerance: float = 1e-6,
        l2: float = 1e-5,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.num_classes = None if num_classes is None else int(num_classes)
        (
            self.learning_rate,
            self.max_iterations,
            self.tolerance,
            self.l2,
            self.weight_policy,
        ) = _validate_optimization(
            learning_rate, max_iterations, tolerance, l2, weight_policy
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_smooth(self, batch, kind="matrix")


class MulticlassCalibrationRecipe(AbstractRecipe):
    num_classes: int | None = eqx.field(static=True)
    learning_rate: Array
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    l2: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        learning_rate: float = 0.02,
        max_iterations: int = 512,
        tolerance: float = 1e-6,
        l2: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.num_classes = None if num_classes is None else int(num_classes)
        (
            self.learning_rate,
            self.max_iterations,
            self.tolerance,
            self.l2,
            self.weight_policy,
        ) = _validate_optimization(
            learning_rate, max_iterations, tolerance, l2, weight_policy
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_smooth(self, batch, kind="multiclass")


def _pav_one(scores: Array, targets: Array, weights: Array) -> tuple[Array, Array, Array]:
    order = jnp.argsort(scores)
    ordered_scores = scores[order]
    ordered_targets = targets[order]
    ordered_weights = weights[order]
    count = scores.shape[0]
    initial = (
        jnp.zeros((count,), dtype=targets.dtype),
        jnp.zeros((count,), dtype=weights.dtype),
        jnp.full((count,), jnp.inf, dtype=scores.dtype),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def push(index, state):
        levels, masses, uppers, top = state

        def add(current):
            levels_, masses_, uppers_, top_ = current
            levels_ = levels_.at[top_].set(ordered_targets[index])
            masses_ = masses_.at[top_].set(ordered_weights[index])
            uppers_ = uppers_.at[top_].set(ordered_scores[index])
            return levels_, masses_, uppers_, top_ + 1

        levels, masses, uppers, top = jax.lax.cond(
            ordered_weights[index] > 0.0,
            add,
            lambda current: current,
            (levels, masses, uppers, top),
        )

        def condition(current):
            levels_, masses_, uppers_, top_ = current
            left = jnp.maximum(top_ - 2, 0)
            right = jnp.maximum(top_ - 1, 0)
            violation = (levels_[left] > levels_[right]) | (
                uppers_[left] == uppers_[right]
            )
            return (top_ >= 2) & violation

        def merge(current):
            levels_, masses_, uppers_, top_ = current
            left = top_ - 2
            right = top_ - 1
            merged_mass = masses_[left] + masses_[right]
            merged_level = (
                masses_[left] * levels_[left] + masses_[right] * levels_[right]
            ) / jnp.maximum(merged_mass, jnp.finfo(weights.dtype).tiny)
            levels_ = levels_.at[left].set(merged_level)
            masses_ = masses_.at[left].set(merged_mass)
            uppers_ = uppers_.at[left].set(uppers_[right])
            return levels_, masses_, uppers_, top_ - 1

        return jax.lax.while_loop(condition, merge, (levels, masses, uppers, top))

    levels, masses, thresholds, block_count = jax.lax.fori_loop(0, count, push, initial)
    last_index = jnp.maximum(block_count - 1, 0)
    last_value = levels[last_index]
    active = jnp.arange(count) < block_count
    levels = jnp.where(active, levels, last_value)
    thresholds = jnp.where(active, thresholds, jnp.inf)
    return thresholds, levels, block_count


class IsotonicCalibrationModel(AbstractArrayModel):
    thresholds: Array
    values: Array
    block_count: Array
    labels: Array
    target_schema: TargetSchema
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        thresholds: Array,
        values: Array,
        block_count: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        case_shape: tuple[int, ...],
    ):
        self.thresholds = jnp.asarray(thresholds)
        self.values = jnp.asarray(values)
        self.block_count = jnp.asarray(block_count, dtype=jnp.int32)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.case_shape = tuple(case_shape)
        self.in_size = 1
        self.out_size = 2

    def positive_probability(self, x: Any, /) -> Array:
        raw = jnp.asarray(x)
        score = raw[..., 0] if raw.ndim > 0 and raw.shape[-1] == 1 else raw
        extra = score.ndim - len(self.case_shape)
        thresholds = _reshape_for_samples(self.thresholds, self.case_shape, extra)
        values = _reshape_for_samples(self.values, self.case_shape, extra)
        index = jnp.sum(score[..., None] > thresholds, axis=-1)
        index = jnp.minimum(
            index, _reshape_for_samples(self.block_count, self.case_shape, extra) - 1
        )
        return jnp.clip(
            jnp.take_along_axis(values, index[..., None], axis=-1)[..., 0], 0.0, 1.0
        )

    def decision_function(self, x: Any, /) -> Array:
        probability = self.positive_probability(x)
        tiny = jnp.finfo(probability.dtype).tiny
        score = jnp.log(jnp.maximum(probability, tiny)) - jnp.log(
            jnp.maximum(1.0 - probability, tiny)
        )
        return jnp.stack((jnp.zeros_like(score), score), axis=-1)

    def predict_proba(self, x: Any, /) -> Array:
        positive = self.positive_probability(x)
        return jnp.stack((1.0 - positive, positive), axis=-1)

    def predict_log_proba(self, x: Any, /) -> Array:
        return jnp.log(
            jnp.maximum(self.predict_proba(x), jnp.finfo(self.values.dtype).tiny)
        )

    def predict_indices(self, x: Any, /) -> Array:
        return (self.positive_probability(x) >= 0.5).astype(jnp.int32)

    def predict(self, x: Any, /) -> Array:
        return jnp.take(self.labels, self.predict_indices(x), axis=0)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


class SmoothIsotonicCalibrationModel(AbstractArrayModel):
    thresholds: Array
    values: Array
    block_count: Array
    labels: Array
    target_schema: TargetSchema
    bandwidth: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        thresholds: Array,
        values: Array,
        block_count: Array,
        labels: Array,
        target_schema: TargetSchema,
        *,
        bandwidth: float,
        case_shape: tuple[int, ...],
    ):
        self.thresholds = jnp.asarray(thresholds)
        self.values = jnp.asarray(values)
        self.block_count = jnp.asarray(block_count, dtype=jnp.int32)
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.bandwidth = jnp.asarray(bandwidth)
        self.case_shape = tuple(case_shape)
        self.in_size = 1
        self.out_size = 2

    def positive_probability(self, x: Any, /) -> Array:
        raw = jnp.asarray(x)
        score = raw[..., 0] if raw.ndim > 0 and raw.shape[-1] == 1 else raw
        extra = score.ndim - len(self.case_shape)
        thresholds = _reshape_for_samples(self.thresholds, self.case_shape, extra)
        values = _reshape_for_samples(self.values, self.case_shape, extra)
        count = _reshape_for_samples(self.block_count, self.case_shape, extra)
        transitions = thresholds[..., :-1]
        deltas = values[..., 1:] - values[..., :-1]
        active = jnp.arange(deltas.shape[-1]) < (count[..., None] - 1)
        smooth_steps = jax.nn.sigmoid((score[..., None] - transitions) / self.bandwidth)
        probability = values[..., 0] + jnp.sum(
            jnp.where(active, deltas * smooth_steps, 0.0), axis=-1
        )
        return jnp.clip(probability, 0.0, 1.0)

    def decision_function(self, x: Any, /) -> Array:
        probability = self.positive_probability(x)
        eps = jnp.finfo(probability.dtype).eps
        clipped = jnp.clip(probability, eps, 1.0 - eps)
        score = jnp.log(clipped) - jnp.log1p(-clipped)
        return jnp.stack((jnp.zeros_like(score), score), axis=-1)

    def predict_proba(self, x: Any, /) -> Array:
        positive = self.positive_probability(x)
        return jnp.stack((1.0 - positive, positive), axis=-1)

    def predict_log_proba(self, x: Any, /) -> Array:
        return jnp.log(
            jnp.maximum(self.predict_proba(x), jnp.finfo(self.values.dtype).tiny)
        )

    def predict_indices(self, x: Any, /) -> Array:
        return (self.positive_probability(x) >= 0.5).astype(jnp.int32)

    def predict(self, x: Any, /) -> Array:
        return jnp.take(self.labels, self.predict_indices(x), axis=0)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


def _fit_isotonic(recipe: Any, batch: MLBatch, *, smooth: bool) -> FitResult:
    scores, encoded, weight, mass, labels, schema = _prepare(
        batch, 2, recipe.weight_policy
    )
    if batch.feature_count != 1:
        raise ValueError("Isotonic calibration requires exactly one score feature.")
    score = scores[..., 0]
    targets = (encoded == 1).astype(score.dtype)
    cases = 1
    for size in batch.case_shape:
        cases *= size
    outputs = jax.vmap(_pav_one)(
        score.reshape((cases, batch.sample_count)),
        targets.reshape((cases, batch.sample_count)),
        weight.reshape((cases, batch.sample_count)),
    )
    thresholds, values, block_count = outputs
    thresholds = thresholds.reshape(batch.case_shape + (batch.sample_count,))
    values = values.reshape(batch.case_shape + (batch.sample_count,))
    block_count = block_count.reshape(batch.case_shape)
    if smooth:
        model: AbstractArrayModel = SmoothIsotonicCalibrationModel(
            thresholds,
            values,
            block_count,
            labels,
            schema,
            bandwidth=recipe.bandwidth,
            case_shape=batch.case_shape,
        )
        method = "smooth-isotonic"
    else:
        model = IsotonicCalibrationModel(
            thresholds, values, block_count, labels, schema, case_shape=batch.case_shape
        )
        method = "exact-isotonic"
    pav_finite = (block_count <= 0) | (
        jnp.all(jnp.isfinite(thresholds[..., :1]), axis=-1)
        & jnp.all(jnp.isfinite(values), axis=-1)
    )
    finite = jnp.all(jnp.isfinite(mass), axis=-1) & pav_finite
    absent = jnp.any(mass <= 0.0, axis=-1)
    valid = finite & ~absent & (block_count > 0)
    status = jnp.where(
        ~finite,
        ML_NONFINITE,
        jnp.where(absent | (block_count <= 0), ML_INSUFFICIENT_DATA, ML_SUCCESS),
    ).astype(jnp.int32)
    diagnostics = CalibrationDiagnostics(
        valid=valid,
        status=status,
        objective=jnp.asarray(jnp.nan),
        iterations=block_count,
        converged=True,
        finite=finite,
        effective_samples=effective_sample_size(weight),
        class_mass=mass,
        method=method,
    )
    return FitResult(
        model,
        diagnostics,
        valid=valid,
        status=status,
        method=method,
        gradient_contract=_contract(smooth_inputs=smooth, fit_mode="stopped"),
    )


class IsotonicCalibrationRecipe(AbstractRecipe):
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(self, *, weight_policy: WeightPolicy = "statistical"):
        if weight_policy not in {"none", "statistical", "measure", "product"}:
            raise ValueError("Unsupported weight policy.")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_isotonic(self, batch, smooth=False)


class SmoothIsotonicCalibrationRecipe(AbstractRecipe):
    bandwidth: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self, *, bandwidth: float = 0.1, weight_policy: WeightPolicy = "statistical"
    ):
        bandwidth_ = jnp.asarray(bandwidth, dtype=float)
        if (
            bandwidth_.ndim != 0
            or not bool(jnp.isfinite(bandwidth_))
            or float(bandwidth_) <= 0.0
        ):
            raise ValueError("bandwidth must be a finite positive scalar.")
        self.bandwidth = bandwidth_
        if weight_policy not in {"none", "statistical", "measure", "product"}:
            raise ValueError("Unsupported weight policy.")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_isotonic(self, batch, smooth=True)


def _base_logits(model: AbstractArrayModel, x: Any) -> Array:
    if isinstance(model, DecisionFunctionModel):
        return model.decision_function(x)
    if isinstance(model, LogProbabilityModel):
        return model.predict_log_proba(x)
    probability = model(x)
    return jnp.log(jnp.maximum(probability, jnp.finfo(probability.dtype).tiny))


def _calibration_input(model: AbstractArrayModel, x: Any, in_size: int) -> Array:
    logits = _base_logits(model, x)
    if in_size != 1:
        if logits.shape[-1] != in_size:
            raise ValueError(
                "Base classifier scores do not align with the calibrator input."
            )
        return logits
    if model.out_size == 2:
        score = logits[..., 1] - logits[..., 0]
    elif logits.ndim > 0 and logits.shape[-1] == 1:
        score = logits[..., 0]
    else:
        score = logits
    return score[..., None]


class CalibratedClassifierModel(AbstractArrayModel):
    base_model: AbstractArrayModel
    calibration_model: AbstractArrayModel
    labels: Array
    target_schema: TargetSchema
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        base_model: AbstractArrayModel,
        calibration_model: AbstractArrayModel,
        labels: Array,
        target_schema: TargetSchema,
    ):
        self.base_model = base_model
        self.calibration_model = calibration_model
        self.labels = jnp.asarray(labels)
        self.target_schema = target_schema
        self.in_size = int(base_model.in_size)
        self.out_size = int(self.labels.shape[0])
        if calibration_model.out_size != self.out_size:
            raise ValueError(
                "Calibrator output must align with the external class vocabulary."
            )

    def decision_function(self, x: Any, /) -> Array:
        logits = _calibration_input(
            self.base_model, x, int(self.calibration_model.in_size)
        )
        if isinstance(self.calibration_model, DecisionFunctionModel):
            return self.calibration_model.decision_function(logits)
        return jnp.log(
            jnp.maximum(self.calibration_model(logits), jnp.finfo(logits.dtype).tiny)
        )

    def predict_proba(self, x: Any, /) -> Array:
        probability = self.calibration_model(
            _calibration_input(self.base_model, x, int(self.calibration_model.in_size))
        )
        return probability / jnp.maximum(
            jnp.sum(probability, axis=-1, keepdims=True),
            jnp.finfo(probability.dtype).tiny,
        )

    def predict_log_proba(self, x: Any, /) -> Array:
        return jnp.log(jnp.maximum(self.predict_proba(x), jnp.finfo(float).tiny))

    def predict_indices(self, x: Any, /) -> Array:
        return jnp.argmax(self.predict_proba(x), axis=-1)

    def predict(self, x: Any, /) -> Array:
        return jnp.take(self.labels, self.predict_indices(x), axis=0)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.predict_proba(x)


class CalibratedClassifierRecipe(AbstractRecipe):
    base_recipe: AbstractRecipe
    calibration_recipe: AbstractRecipe
    num_classes: int | None = eqx.field(static=True)

    def __init__(
        self,
        base_recipe: AbstractRecipe,
        calibration_recipe: AbstractRecipe,
        /,
        *,
        num_classes: int | None = None,
    ):
        if not isinstance(base_recipe, AbstractRecipe) or not isinstance(
            calibration_recipe, AbstractRecipe
        ):
            raise TypeError(
                "base_recipe and calibration_recipe must be AbstractRecipe instances."
            )
        self.base_recipe = base_recipe
        self.calibration_recipe = calibration_recipe
        self.num_classes = None if num_classes is None else int(num_classes)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        labels, schema = _labels_for(batch, self.num_classes)
        base_key, calibration_key = (
            (None, None) if key is None else tuple(jax.random.split(key, 2))
        )
        base_result = self.base_recipe.fit_batch(batch, key=base_key)
        base_model = base_result.as_trainable()
        scalar_calibrator = isinstance(
            self.calibration_recipe,
            (
                PlattCalibrationRecipe,
                IsotonicCalibrationRecipe,
                SmoothIsotonicCalibrationRecipe,
            ),
        )
        calibration_in_size = 1 if scalar_calibrator else int(labels.shape[0])
        logits = _calibration_input(
            base_model, batch.dense_features(), calibration_in_size
        )
        calibration_sample_mask = batch.sample_mask & jnp.all(batch.feature_mask, axis=-1)
        calibration_batch = MLBatch(
            logits,
            batch.require_targets(),
            feature_mask=jnp.broadcast_to(
                calibration_sample_mask[..., None], logits.shape
            ),
            target_mask=batch.target_mask,
            sample_mask=calibration_sample_mask,
            sample_weight=batch.sample_weight,
            measure_weight=batch.measure_weight,
            groups=batch.groups,
            feature_schema=FeatureSchema.anonymous(logits.shape[-1]),
            target_schema=schema,
        )
        calibration_result = self.calibration_recipe.fit_batch(
            calibration_batch, key=calibration_key
        )
        calibration_model = calibration_result.as_trainable()
        model = CalibratedClassifierModel(base_model, calibration_model, labels, schema)
        valid = base_result.valid & calibration_result.valid
        status = jnp.maximum(base_result.status, calibration_result.status)
        diagnostics = StrictCalibrationCompositionDiagnostics(
            base_result.valid,
            base_result.status,
            calibration_result.valid,
            calibration_result.status,
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="calibrated-classifier",
            gradient_contract=_contract(),
        )


class StrictCalibrationCompositionDiagnostics(StrictModule):
    base_valid: Array
    base_status: Array
    calibration_valid: Array
    calibration_status: Array
    valid: Array
    status: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        base_valid: Any,
        base_status: Any,
        calibration_valid: Any,
        calibration_status: Any,
    ):
        self.base_valid = jnp.asarray(base_valid, dtype=bool)
        self.base_status = jnp.asarray(base_status, dtype=jnp.int32)
        self.calibration_valid = jnp.asarray(calibration_valid, dtype=bool)
        self.calibration_status = jnp.asarray(calibration_status, dtype=jnp.int32)
        self.valid = self.base_valid & self.calibration_valid
        self.status = jnp.maximum(self.base_status, self.calibration_status)
        self.method = "calibrated-classifier"


__all__ = [
    "CalibratedClassifierModel",
    "CalibratedClassifierRecipe",
    "CalibrationDiagnostics",
    "IsotonicCalibrationModel",
    "IsotonicCalibrationRecipe",
    "MatrixCalibrationModel",
    "MatrixCalibrationRecipe",
    "MulticlassCalibrationModel",
    "MulticlassCalibrationRecipe",
    "PlattCalibrationModel",
    "PlattCalibrationRecipe",
    "SmoothIsotonicCalibrationModel",
    "SmoothIsotonicCalibrationRecipe",
    "StrictCalibrationCompositionDiagnostics",
    "TemperatureCalibrationModel",
    "TemperatureCalibrationRecipe",
    "VectorCalibrationModel",
    "VectorCalibrationRecipe",
]
