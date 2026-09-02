#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from .._batch import MLBatch
from .._contracts import FitResult


class PartialDependenceResult(StrictModule):
    grid: Array
    feature_indices: Array
    ice: Array
    average: Array
    sample_weight: Array

    def __init__(
        self,
        grid: Any,
        feature_indices: Any,
        ice: Any,
        average: Any,
        sample_weight: Any,
        /,
    ):
        self.grid = jnp.asarray(grid)
        self.feature_indices = jnp.asarray(feature_indices, dtype=jnp.int32)
        self.ice = jnp.asarray(ice)
        self.average = jnp.asarray(average)
        self.sample_weight = jnp.asarray(sample_weight)


class PermutationImportanceResult(StrictModule):
    baseline_score: Array
    permuted_scores: Array
    importances: Array
    mean_importance: Array
    standard_error: Array

    def __init__(
        self,
        baseline_score: Any,
        permuted_scores: Any,
        importances: Any,
        /,
    ):
        permuted = jnp.asarray(permuted_scores)
        importance = jnp.asarray(importances)
        if permuted.shape != importance.shape or permuted.ndim < 2:
            raise ValueError(
                "permuted_scores and importances must align as repeat/feature arrays."
            )
        self.baseline_score = jnp.asarray(baseline_score)
        self.permuted_scores = permuted
        self.importances = importance
        self.mean_importance = jnp.mean(importance, axis=0)
        self.standard_error = jnp.std(importance, axis=0, ddof=1) / jnp.sqrt(
            jnp.asarray(max(importance.shape[0], 1), dtype=importance.real.dtype)
        )


class SensitivityResult(StrictModule):
    values: Array
    derivative: Array
    order: int = eqx.field(static=True)
    holomorphic: bool = eqx.field(static=True)

    def __init__(self, values: Any, derivative: Any, /, *, order: int, holomorphic: bool):
        self.values = jnp.asarray(values)
        self.derivative = jnp.asarray(derivative)
        self.order = int(order)
        self.holomorphic = bool(holomorphic)


class RegressionInfluenceDiagnostics(StrictModule):
    prediction: Array
    residual: Array
    leverage: Array
    cooks_distance: Array
    mean_squared_error: Array
    effective_parameters: Array
    valid: Array

    def __init__(
        self,
        prediction: Any,
        residual: Any,
        leverage: Any,
        cooks_distance: Any,
        mean_squared_error: Any,
        effective_parameters: Any,
        valid: Any,
        /,
    ):
        self.prediction = jnp.asarray(prediction)
        self.residual = jnp.asarray(residual)
        self.leverage = jnp.asarray(leverage)
        self.cooks_distance = jnp.asarray(cooks_distance)
        self.mean_squared_error = jnp.asarray(mean_squared_error)
        self.effective_parameters = jnp.asarray(effective_parameters, dtype=jnp.int32)
        self.valid = jnp.asarray(valid, dtype=bool)


class InfluenceFunctionResult(StrictModule):
    parameter_influence: Array
    loss_influence: Array
    hessian: Array
    sample_gradients: Array
    evaluation_gradients: Array
    valid: Array

    def __init__(
        self,
        parameter_influence: Any,
        loss_influence: Any,
        hessian: Any,
        sample_gradients: Any,
        evaluation_gradients: Any,
        valid: Any,
        /,
    ):
        self.parameter_influence = jnp.asarray(parameter_influence)
        self.loss_influence = jnp.asarray(loss_influence)
        self.hessian = jnp.asarray(hessian)
        self.sample_gradients = jnp.asarray(sample_gradients)
        self.evaluation_gradients = jnp.asarray(evaluation_gradients)
        self.valid = jnp.asarray(valid, dtype=bool)


def _validated_weights(batch: MLBatch) -> Array:
    weights = jnp.asarray(batch.sample_weight)
    weights = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
        "Sample weights must be finite and nonnegative.",
    )
    return jnp.where(batch.sample_mask, weights, 0.0)


def _batch_inputs(features: MLBatch | Any) -> tuple[Array, Array, tuple[int, ...]]:
    if isinstance(features, MLBatch):
        return (
            features.dense_features(),
            _validated_weights(features),
            features.case_shape,
        )
    array = jnp.asarray(features)
    if array.ndim < 2:
        raise ValueError("features must have shape case_shape + (sample, feature).")
    case_shape = tuple(int(size) for size in array.shape[:-2])
    weight_dtype = jnp.result_type(array.real.dtype, jnp.float32)
    return array, jnp.ones(array.shape[:-1], dtype=weight_dtype), case_shape


def _feature_grid(grid: Any, feature_count: int) -> Array:
    value = jnp.asarray(grid)
    if feature_count == 1 and value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[1] != feature_count or value.shape[0] == 0:
        raise ValueError("grid must have shape (grid_point, selected_feature).")
    return value


def individual_conditional_expectation(
    model: AbstractArrayModel,
    features: MLBatch | Any,
    feature_indices: Sequence[int],
    grid: Any,
    /,
    *,
    key: Any = None,
) -> PartialDependenceResult:
    """Evaluate all individual conditional-expectation curves as JAX arrays."""
    if not isinstance(model, AbstractArrayModel):
        raise TypeError("model must be an AbstractArrayModel.")
    x, weights, case_shape = _batch_inputs(features)
    indices_tuple = tuple(int(index) for index in feature_indices)
    if not indices_tuple or len(set(indices_tuple)) != len(indices_tuple):
        raise ValueError("feature_indices must be nonempty and unique.")
    if any(index < 0 or index >= x.shape[-1] for index in indices_tuple):
        raise IndexError("feature_indices contains an out-of-range index.")
    values = _feature_grid(grid, len(indices_tuple))
    common_dtype = jnp.result_type(x.dtype, values.dtype)
    x = x.astype(common_dtype)
    values = values.astype(common_dtype)
    predictions = []
    for grid_index in range(values.shape[0]):
        modified = x.at[..., jnp.asarray(indices_tuple)].set(values[grid_index])
        member_key = None if key is None else jr.fold_in(key, grid_index)
        predictions.append(jnp.asarray(model(modified, key=member_key)))
    ice = jnp.stack(tuple(predictions), axis=0)
    sample_axis = 1 + len(case_shape)
    output_rank = ice.ndim - (sample_axis + 1)
    expanded_weight = weights[None, ...].reshape(
        (1,) + weights.shape + (1,) * output_rank
    )
    numerator = jnp.sum(expanded_weight * ice, axis=sample_axis)
    denominator = jnp.sum(expanded_weight, axis=sample_axis)
    average = numerator / jnp.maximum(denominator, jnp.finfo(weights.dtype).tiny)
    return PartialDependenceResult(
        values, jnp.asarray(indices_tuple), ice, average, weights
    )


def partial_dependence(
    model: AbstractArrayModel,
    features: MLBatch | Any,
    feature_indices: Sequence[int],
    grid: Any,
    /,
    *,
    key: Any = None,
) -> PartialDependenceResult:
    """Return both weighted partial dependence and the underlying ICE tensor."""
    return individual_conditional_expectation(
        model, features, feature_indices, grid, key=key
    )


def _weighted_mean_squared_error(batch: MLBatch, prediction: Array) -> Array:
    targets = batch.require_targets()
    if prediction.shape != targets.shape:
        raise ValueError(
            "Prediction and target shapes must match for the default metric."
        )
    residual_loss = jnp.real((prediction - targets) * jnp.conj(prediction - targets))
    sample_ndim = len(batch.case_shape) + 1
    if batch.target_mask is not None:
        residual_loss = jnp.where(batch.target_mask, residual_loss, 0.0)
        output_count = (
            jnp.sum(batch.target_mask, axis=tuple(range(sample_ndim, residual_loss.ndim)))
            if residual_loss.ndim > sample_ndim
            else batch.target_mask.astype(int)
        )
    else:
        output_count = jnp.asarray(1 if residual_loss.ndim == sample_ndim else 1)
    if residual_loss.ndim > sample_ndim:
        loss = jnp.sum(residual_loss, axis=tuple(range(sample_ndim, residual_loss.ndim)))
        if batch.target_mask is None:
            count = 1
            for size in residual_loss.shape[sample_ndim:]:
                count *= int(size)
            output_count = jnp.full(loss.shape, count)
    else:
        loss = residual_loss
    weights = _validated_weights(batch)
    effective_weight = weights * (output_count > 0)
    per_sample = loss / jnp.maximum(output_count, 1)
    return jnp.sum(effective_weight * per_sample, axis=-1) / jnp.maximum(
        jnp.sum(effective_weight, axis=-1), jnp.finfo(weights.dtype).tiny
    )


def permutation_importance(
    model: AbstractArrayModel,
    batch: MLBatch,
    /,
    *,
    key: Any,
    repeats: int = 8,
    metric: Callable[[MLBatch, Array], Array] | None = None,
) -> PermutationImportanceResult:
    """Estimate loss increase under deterministic keyed feature permutations."""
    if not isinstance(model, AbstractArrayModel) or not isinstance(batch, MLBatch):
        raise TypeError("model and batch must use native Phydrax ML types.")
    if key is None:
        raise ValueError("permutation_importance requires an explicit JAX key.")
    if int(repeats) <= 1:
        raise ValueError("repeats must be greater than one.")
    scorer = _weighted_mean_squared_error if metric is None else metric
    x = batch.dense_features()
    sample_axis = len(batch.case_shape)
    baseline_prediction = jnp.asarray(model(x, key=jr.fold_in(key, 0)))
    baseline = jnp.asarray(scorer(batch, baseline_prediction))
    repeat_scores = []
    for repeat in range(int(repeats)):
        feature_scores = []
        for feature in range(batch.feature_count):
            stream = 1 + repeat * batch.feature_count + feature
            permutation = jr.permutation(jr.fold_in(key, stream), batch.sample_count)
            column = jnp.take(x[..., feature], permutation, axis=sample_axis)
            permuted = x.at[..., feature].set(column)
            prediction = jnp.asarray(
                model(permuted, key=jr.fold_in(key, 100000 + stream))
            )
            feature_scores.append(jnp.asarray(scorer(batch, prediction)))
        repeat_scores.append(jnp.stack(tuple(feature_scores), axis=0))
    permuted_scores = jnp.stack(tuple(repeat_scores), axis=0)
    baseline_expanded = baseline.reshape((1, 1) + baseline.shape)
    return PermutationImportanceResult(
        baseline,
        permuted_scores,
        permuted_scores - baseline_expanded,
    )


def _require_pointwise(model: AbstractArrayModel) -> None:
    if not isinstance(model, AbstractArrayModel):
        raise TypeError("model must be an AbstractArrayModel.")
    if model.input_binding().batch_mode != "pointwise":
        raise ValueError("Sensitivity inspection requires a pointwise model contract.")


def _full_derivative(
    function: Callable[[Array], Array],
    argument: Array,
    /,
    *,
    order: int,
    holomorphic: bool,
) -> Array:
    value = function(argument)
    input_complex = jnp.issubdtype(argument.dtype, jnp.complexfloating)
    output_complex = jnp.issubdtype(value.dtype, jnp.complexfloating)
    if holomorphic:
        if not input_complex or not output_complex:
            raise TypeError(
                "holomorphic=True requires complex inputs and complex outputs."
            )
        derivative = jax.jacrev(function, holomorphic=True)
        if order == 2:
            derivative = jax.jacfwd(derivative, holomorphic=True)
        return derivative(argument)
    if input_complex:
        raise TypeError(
            "Complex inputs require holomorphic=True; Wirtinger conventions are "
            "not inferred."
        )
    if output_complex:

        def real_function(value: Array) -> Array:
            return jnp.real(function(value))

        def imag_function(value: Array) -> Array:
            return jnp.imag(function(value))

        real_derivative = jax.jacrev(real_function)
        imag_derivative = jax.jacrev(imag_function)
        if order == 2:
            real_derivative = jax.jacfwd(real_derivative)
            imag_derivative = jax.jacfwd(imag_derivative)
        return real_derivative(argument) + 1j * imag_derivative(argument)
    derivative = jax.jacrev(function)
    if order == 2:
        derivative = jax.jacfwd(derivative)
    return derivative(argument)


def _batch_sensitivity(
    model: AbstractArrayModel,
    points: Any,
    /,
    *,
    key: Any,
    holomorphic: bool,
    order: int,
) -> tuple[Array, Array]:
    _require_pointwise(model)
    x = jnp.asarray(points)
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("points must end in a nonempty feature axis.")
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        raise TypeError("Sensitivity derivatives require inexact-valued inputs.")
    leading_shape = x.shape[:-1]
    point_count = 1
    for size in leading_shape:
        point_count *= int(size)
    flat = x.reshape((point_count, x.shape[-1]))
    values = jnp.asarray(model(x, key=key))
    if values.shape[: len(leading_shape)] != leading_shape:
        raise ValueError(
            "Pointwise model predictions must preserve all input leading axes."
        )
    output_shape = values.shape[len(leading_shape) :]

    def evaluate(argument: Array) -> Array:
        output = jnp.asarray(model(argument.reshape(x.shape), key=key))
        return output.reshape((point_count, -1))

    derivative = _full_derivative(evaluate, flat, order=order, holomorphic=holomorphic)
    diagonal = jnp.arange(point_count)
    if order == 1:
        diagonal_derivative = derivative[diagonal, :, diagonal, :]
        trailing = (x.shape[-1],)
    else:
        diagonal_derivative = derivative[diagonal, :, diagonal, :, diagonal, :]
        trailing = (x.shape[-1], x.shape[-1])
    if output_shape:
        shaped = diagonal_derivative.reshape(leading_shape + output_shape + trailing)
    else:
        shaped = diagonal_derivative[:, 0].reshape(leading_shape + trailing)
    return values, shaped


def jacobian_sensitivity(
    model: AbstractArrayModel,
    points: Any,
    /,
    *,
    key: Any = None,
    holomorphic: bool = False,
) -> SensitivityResult:
    """Pointwise output Jacobians without cross-sample or cross-case terms."""
    values, derivative = _batch_sensitivity(
        model, points, key=key, holomorphic=holomorphic, order=1
    )
    return SensitivityResult(
        values,
        derivative,
        order=1,
        holomorphic=holomorphic,
    )


def gradient_sensitivity(
    model: AbstractArrayModel,
    points: Any,
    /,
    *,
    key: Any = None,
    holomorphic: bool = False,
) -> SensitivityResult:
    """Gradient of a scalar-output pointwise model."""
    result = jacobian_sensitivity(model, points, key=key, holomorphic=holomorphic)
    value = result.values
    if value.shape == jnp.asarray(points).shape[:-1]:
        derivative = result.derivative
    elif value.shape == jnp.asarray(points).shape[:-1] + (1,):
        value = value[..., 0]
        derivative = result.derivative[..., 0, :]
    else:
        raise ValueError("gradient_sensitivity requires a scalar-output model.")
    return SensitivityResult(value, derivative, order=1, holomorphic=holomorphic)


def hessian_sensitivity(
    model: AbstractArrayModel,
    points: Any,
    /,
    *,
    key: Any = None,
    holomorphic: bool = False,
) -> SensitivityResult:
    """Input Hessian of a scalar-output pointwise model."""
    values, derivative = _batch_sensitivity(
        model, points, key=key, holomorphic=holomorphic, order=2
    )
    leading_shape = jnp.asarray(points).shape[:-1]
    if values.shape == leading_shape:
        scalar_values = values
        scalar_derivative = derivative
    elif values.shape == leading_shape + (1,):
        scalar_values = values[..., 0]
        scalar_derivative = derivative[..., 0, :, :]
    else:
        raise ValueError("hessian_sensitivity requires a scalar-output model.")
    return SensitivityResult(
        scalar_values,
        scalar_derivative,
        order=2,
        holomorphic=holomorphic,
    )


def leverage_and_cooks_distance(
    model: AbstractArrayModel,
    batch: MLBatch,
    /,
    *,
    fit_intercept: bool = True,
    rcond: float | None = None,
    key: Any = None,
) -> RegressionInfluenceDiagnostics:
    """Weighted leverage and Cook-style diagnostics for a fitted regression model."""
    if not isinstance(model, AbstractArrayModel) or not isinstance(batch, MLBatch):
        raise TypeError("model and batch must use native Phydrax ML types.")
    x = batch.dense_features()
    targets = batch.require_targets()
    prediction = jnp.asarray(model(x, key=key))
    if prediction.shape != targets.shape:
        raise ValueError(
            "Regression diagnostics require prediction and target shapes to match."
        )
    sample_prefix = batch.case_shape + (batch.sample_count,)
    target_shape = targets.shape[len(sample_prefix) :]
    target_flat = (
        targets.reshape(sample_prefix + (-1,)) if target_shape else targets[..., None]
    )
    prediction_flat = (
        prediction.reshape(sample_prefix + (-1,))
        if target_shape
        else prediction[..., None]
    )
    residual = prediction_flat - target_flat
    weights = _validated_weights(batch)
    valid_sample = batch.sample_mask & jnp.all(batch.feature_mask, axis=-1)
    if batch.target_mask is not None:
        target_valid = batch.target_mask
        if target_valid.ndim > len(sample_prefix):
            target_valid = jnp.all(
                target_valid, axis=tuple(range(len(sample_prefix), target_valid.ndim))
            )
        valid_sample = valid_sample & target_valid
    weights = jnp.where(valid_sample, weights, 0.0)
    residual = jnp.where(valid_sample[..., None], residual, 0)
    design = (
        jnp.concatenate((x, jnp.ones(x.shape[:-1] + (1,), dtype=x.dtype)), axis=-1)
        if fit_intercept
        else x
    )
    design = design.astype(jnp.result_type(design.dtype, jnp.float32))
    cases = 1
    for size in batch.case_shape:
        cases *= int(size)
    design_cases = design.reshape((cases, batch.sample_count, design.shape[-1]))
    weight_cases = weights.reshape((cases, batch.sample_count))
    residual_cases = residual.reshape((cases, batch.sample_count, residual.shape[-1]))
    cutoff_dtype = jnp.result_type(design.real.dtype, jnp.float32)
    cutoff = (
        max(design.shape[-2:]) * jnp.finfo(cutoff_dtype).eps
        if rcond is None
        else float(rcond)
    )

    def one_case(case_design: Array, case_weight: Array, case_residual: Array):
        sqrt_weight = jnp.sqrt(case_weight)
        weighted_design = sqrt_weight[:, None] * case_design
        left, singular, _ = jnp.linalg.svd(weighted_design, full_matrices=False)
        threshold = jnp.max(singular, initial=0.0) * cutoff
        retained = singular > threshold
        leverage = jnp.sum(
            jnp.abs(left) ** 2 * retained[None, :],
            axis=-1,
        )
        parameters = jnp.sum(retained, dtype=jnp.int32)
        degrees = jnp.maximum(jnp.sum(case_weight > 0.0) - parameters, 1)
        squared = jnp.real(case_residual * jnp.conj(case_residual))
        rss = jnp.sum(case_weight[:, None] * squared, axis=0)
        mse = rss / degrees
        scaled_residual = case_weight[:, None] * squared
        cooks = (
            scaled_residual
            / jnp.maximum(parameters * mse[None, :], jnp.finfo(mse.dtype).tiny)
            * leverage[:, None]
            / jnp.maximum((1.0 - leverage[:, None]) ** 2, jnp.finfo(mse.dtype).tiny)
        )
        valid = (
            jnp.all(jnp.isfinite(leverage))
            & jnp.all(jnp.isfinite(cooks))
            & (parameters > 0)
        )
        return leverage, cooks, mse, parameters, valid

    leverage, cooks, mse, parameters, valid = jax.vmap(one_case)(
        design_cases, weight_cases, residual_cases
    )
    leverage = leverage.reshape(batch.case_shape + (batch.sample_count,))
    cooks = cooks.reshape(batch.case_shape + (batch.sample_count, residual.shape[-1]))
    mse = mse.reshape(batch.case_shape + (residual.shape[-1],))
    parameters = parameters.reshape(batch.case_shape)
    valid = valid.reshape(batch.case_shape)
    output_residual = residual.reshape(targets.shape if target_shape else sample_prefix)
    output_cooks = cooks.reshape(
        batch.case_shape + (batch.sample_count,) + (target_shape or ())
    )
    output_mse = mse.reshape(batch.case_shape + (target_shape or ()))
    return RegressionInfluenceDiagnostics(
        prediction,
        output_residual,
        leverage,
        output_cooks,
        output_mse,
        parameters,
        valid,
    )


def _default_loss(prediction: Array, target: Array) -> Array:
    residual = prediction - target
    return jnp.sum(jnp.real(residual * jnp.conj(residual)))


def _influence_arrays(
    batch: MLBatch,
) -> tuple[Array, Array, Array, tuple[int, ...]]:
    x = batch.dense_features()
    targets = batch.require_targets()
    sample_prefix = batch.case_shape + (batch.sample_count,)
    valid = batch.sample_mask & jnp.all(batch.feature_mask, axis=-1)
    if batch.target_mask is not None:
        target_valid = batch.target_mask
        if target_valid.ndim > len(sample_prefix):
            target_valid = jnp.all(
                target_valid,
                axis=tuple(range(len(sample_prefix), target_valid.ndim)),
            )
        valid = valid & target_valid
    target_mask = valid.reshape(valid.shape + (1,) * (targets.ndim - len(sample_prefix)))
    targets = jnp.where(target_mask, targets, 0)
    weight = (_validated_weights(batch) * valid).reshape((-1,))
    return x, targets, weight, sample_prefix


def influence_functions(
    result: FitResult,
    batch: MLBatch,
    /,
    *,
    loss: Callable[[Array, Array], Array] | None = None,
    evaluation_batch: MLBatch | None = None,
    damping: float = 1e-6,
    key: Any = None,
) -> InfluenceFunctionResult:
    """Compute damped empirical influence functions when fit gradients are certified."""
    if not isinstance(result, FitResult) or not isinstance(batch, MLBatch):
        raise TypeError("result and batch must use native Phydrax ML types.")
    contract = result.gradient_contract
    if contract.fit_mode == "stopped" or contract.fit_targets == "none":
        raise ValueError(
            "The fit result's GradientContract does not permit influence functions."
        )
    if float(damping) < 0.0:
        raise ValueError("damping must be nonnegative.")
    model = result.as_trainable()
    dynamic, static = eqx.partition(model, eqx.is_inexact_array)
    parameters, unravel = ravel_pytree(dynamic)
    if parameters.shape[0] == 0:
        raise ValueError("The fitted model has no inexact array parameters.")
    if jnp.issubdtype(parameters.dtype, jnp.complexfloating):
        raise TypeError(
            "Influence functions require a real parameterization; no implicit "
            "Wirtinger convention is selected."
        )
    loss_function = _default_loss if loss is None else loss
    x, y, weights, sample_prefix = _influence_arrays(batch)
    evaluation = batch if evaluation_batch is None else evaluation_batch
    eval_x, eval_y, eval_weights, eval_prefix = _influence_arrays(evaluation)
    normalizer = jnp.maximum(jnp.sum(weights), jnp.finfo(weights.dtype).tiny)

    def rebuild(theta: Array) -> AbstractArrayModel:
        return eqx.combine(unravel(theta), static)

    def loss_vector(
        theta: Array,
        features: Array,
        targets: Array,
        weight: Array,
        prefix: tuple[int, ...],
    ) -> Array:
        prediction = jnp.asarray(rebuild(theta)(features, key=key))
        if prediction.shape != targets.shape:
            raise ValueError(
                "Influence loss requires prediction and target shapes to match."
            )
        output_shape = targets.shape[len(prefix) :]
        prediction_flat = prediction.reshape((-1,) + output_shape)
        target_flat = targets.reshape((-1,) + output_shape)

        def weighted_loss(predicted: Array, target: Array, sample_weight: Array):
            value = jnp.asarray(loss_function(predicted, target))
            if value.ndim != 0 or jnp.issubdtype(value.dtype, jnp.complexfloating):
                raise ValueError("loss must return one real scalar per sample.")
            return sample_weight * value

        return jax.vmap(weighted_loss)(prediction_flat, target_flat, weight)

    def training_losses(theta: Array) -> Array:
        return loss_vector(theta, x, y, weights, sample_prefix)

    def evaluation_losses(theta: Array) -> Array:
        return loss_vector(theta, eval_x, eval_y, eval_weights, eval_prefix)

    sample_gradients = jax.jacrev(training_losses)(parameters)
    evaluation_gradients = jax.jacrev(evaluation_losses)(parameters)

    def objective(theta: Array) -> Array:
        return jnp.sum(training_losses(theta)) / normalizer

    hessian = jax.jacfwd(jax.jacrev(objective))(parameters)
    regularized = hessian + float(damping) * jnp.eye(
        parameters.shape[0], dtype=hessian.dtype
    )
    solved = jnp.linalg.solve(regularized, sample_gradients.T).T
    parameter_influence = -solved / normalizer
    loss_influence = -(evaluation_gradients @ solved.T) / normalizer
    valid = (
        jnp.all(result.valid)
        & jnp.all(jnp.isfinite(hessian))
        & jnp.all(jnp.isfinite(parameter_influence))
        & jnp.all(jnp.isfinite(loss_influence))
    )
    return InfluenceFunctionResult(
        parameter_influence,
        loss_influence,
        hessian,
        sample_gradients,
        evaluation_gradients,
        valid,
    )


__all__ = [
    "InfluenceFunctionResult",
    "PartialDependenceResult",
    "PermutationImportanceResult",
    "RegressionInfluenceDiagnostics",
    "SensitivityResult",
    "gradient_sensitivity",
    "hessian_sensitivity",
    "individual_conditional_expectation",
    "influence_functions",
    "jacobian_sensitivity",
    "leverage_and_cooks_distance",
    "partial_dependence",
    "permutation_importance",
]
