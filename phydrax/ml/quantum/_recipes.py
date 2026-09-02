#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from ..._trainable import combine_trainable, partition_trainable
from ..._tree_math import tree_allfinite, tree_inner, tree_norm
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import run_fixed_iterations
from .._schema import FeatureSchema
from ._models import (
    BinaryVariationalCircuitClassifier,
    DenseCircuitExpectationModel,
)


class FittedCircuitFeatureTransform(AbstractArrayModel):
    """Schema-bound exact dense quantum expectation feature transform."""

    model: DenseCircuitExpectationModel
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        model: DenseCircuitExpectationModel,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
        /,
    ):
        if not isinstance(model, DenseCircuitExpectationModel):
            raise TypeError("model must be DenseCircuitExpectationModel.")
        if len(input_schema.names) != model.in_size:
            raise ValueError("Input feature schema does not match the circuit model.")
        if len(output_schema.names) != model.out_size:
            raise ValueError("Output feature schema does not match the circuit model.")
        self.model = model
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.in_size = model.in_size
        self.out_size = model.out_size

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        return self.model(x, key=key)

    def transform(
        self,
        x: Any,
        /,
        *,
        mask: Any | None = None,
        key: Any = None,
    ) -> Array:
        values = jnp.asarray(x)
        if values.ndim < 1 or values.shape[-1] != self.in_size:
            raise ValueError(
                "Circuit transform input must end in the fitted feature axis."
            )
        if mask is not None:
            values = jnp.where(
                jnp.broadcast_to(jnp.asarray(mask, dtype=bool), values.shape),
                values,
                jnp.zeros((), dtype=values.dtype),
            )
        leading_shape = values.shape[:-1]
        flat = values.reshape((-1, self.in_size))
        if key is None:
            transformed = jax.vmap(lambda row: self.model(row))(flat)
        else:
            keys = jax.random.split(key, flat.shape[0])
            transformed = jax.vmap(lambda row, row_key: self.model(row, key=row_key))(
                flat, keys
            )
        return transformed.reshape(leading_shape + (self.out_size,))


class CircuitFeatureTransformRecipe(AbstractRecipe):
    """Fit-free schema binding for an exact circuit expectation feature model."""

    model: DenseCircuitExpectationModel
    output_names: tuple[str, ...] = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        model: DenseCircuitExpectationModel,
        /,
        *,
        output_names: Sequence[str] = (),
        weight_policy: WeightPolicy = "statistical",
    ):
        if not isinstance(model, DenseCircuitExpectationModel):
            raise TypeError("model must be DenseCircuitExpectationModel.")
        names = tuple(str(name) for name in output_names)
        if names and len(names) != model.out_size:
            raise ValueError("output_names must match the circuit observable count.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.model = model
        self.output_names = names
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if not isinstance(batch, MLBatch):
            raise TypeError("batch must be an MLBatch.")
        if batch.feature_count != self.model.in_size:
            raise ValueError("Batch feature count does not match the circuit model.")
        features = batch.dense_features()
        weights = batch.effective_weight(self.weight_policy)
        active = weights > 0.0
        entries_valid = batch.feature_mask & jnp.isfinite(features)
        data_valid = jnp.all(
            jnp.where(active[..., None], entries_valid, True),
            axis=(-2, -1),
        )
        weights_valid = batch.weights_valid(self.weight_policy)
        valid = data_valid & weights_valid
        status = jnp.where(valid, ML_SUCCESS, ML_NONFINITE)
        names = self.output_names or tuple(
            f"quantum_feature_{index}" for index in range(self.model.out_size)
        )
        output_schema = FeatureSchema(
            names,
            layout_id=self.model.model_id,
        )
        fitted = FittedCircuitFeatureTransform(
            self.model,
            batch.feature_schema,
            output_schema,
        )
        effective = jnp.sum(active, axis=-1)
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            objective=jnp.full(batch.case_shape, jnp.nan),
            iterations=jnp.zeros(batch.case_shape, dtype=jnp.int32),
            effective_samples=effective,
            method="circuit_feature_transform",
        )
        contract = GradientContract(
            prediction_inputs="conditional",
            prediction_parameters="conditional",
            fit_mode="direct",
            conditions=(
                "The dense program and local observables remain valid.",
                "The circuit feature model is frozen by this fit-free recipe.",
            ),
        )
        return FitResult(
            fitted,
            diagnostics,
            valid=valid,
            status=status,
            method="circuit_feature_transform",
            gradient_contract=contract,
        )


class CircuitFitDiagnostics(StrictModule):
    common: FitDiagnostics
    final_gradient_norm: Array
    converged: Array
    logical_program_evaluations: Array
    parameterized_occurrences: int = eqx.field(static=True)
    gradient_method: str = eqx.field(static=True)


class VariationalCircuitClassifierRecipe(AbstractRecipe):
    """Full-batch exact binary fit over one initialized circuit feature model."""

    feature_model: DenseCircuitExpectationModel
    negative_label: float = eqx.field(static=True)
    positive_label: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    l2_strength: float = eqx.field(static=True)

    def __init__(
        self,
        feature_model: DenseCircuitExpectationModel,
        /,
        *,
        class_labels: tuple[float, float] = (0.0, 1.0),
        weight_policy: WeightPolicy = "statistical",
        learning_rate: float = 1e-2,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        l2_strength: float = 0.0,
    ):
        if not isinstance(feature_model, DenseCircuitExpectationModel):
            raise TypeError("feature_model must be DenseCircuitExpectationModel.")
        labels = (float(class_labels[0]), float(class_labels[1]))
        values = (
            float(learning_rate),
            float(tolerance),
            float(l2_strength),
        )
        if not all(isfinite(label) for label in labels) or labels[0] == labels[1]:
            raise ValueError("class_labels must be distinct finite scalars.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        if not all(isfinite(value) for value in values):
            raise ValueError("Classifier numerical controls must be finite.")
        if values[0] <= 0.0 or values[1] < 0.0 or values[2] < 0.0:
            raise ValueError(
                "Learning rate must be positive; tolerances must be nonnegative."
            )
        if int(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive.")
        self.feature_model = feature_model
        self.negative_label = labels[0]
        self.positive_label = labels[1]
        self.weight_policy = weight_policy
        self.learning_rate = values[0]
        self.max_iterations = int(max_iterations)
        self.tolerance = values[1]
        self.l2_strength = values[2]

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if not isinstance(batch, MLBatch):
            raise TypeError("batch must be an MLBatch.")
        if key is None:
            raise ValueError("Variational circuit fitting requires an explicit JAX key.")
        if batch.case_shape:
            raise ValueError(
                "Variational circuit fitting currently requires one ML case."
            )
        if batch.feature_count != self.feature_model.in_size:
            raise ValueError("Batch feature count does not match the circuit model.")
        if batch.target_shape != ():
            raise ValueError("Variational binary classification requires scalar targets.")
        if batch.target_schema.class_labels and tuple(
            batch.target_schema.class_labels
        ) != (self.negative_label, self.positive_label):
            raise ValueError("Target-schema labels do not match recipe class_labels.")
        features = batch.dense_features()
        targets = batch.require_targets()
        target_mask = (
            jnp.ones_like(targets, dtype=bool)
            if batch.target_mask is None
            else batch.target_mask
        )
        weights = batch.effective_weight(self.weight_policy)
        active = (weights > 0.0) & target_mask
        entries_valid = batch.feature_mask & jnp.isfinite(features)
        features_valid = jnp.all(jnp.where(active[:, None], entries_valid, True))
        targets_finite = jnp.isfinite(targets)
        labels_valid = (targets == self.negative_label) | (targets == self.positive_label)
        targets_valid = jnp.all(jnp.where(active, targets_finite & labels_valid, True))
        weights_valid = jnp.all(jnp.isfinite(weights) & (weights >= 0.0)) & (
            jnp.sum(jnp.where(active, weights, 0.0)) > 0.0
        )
        encoded = (targets == self.positive_label).astype(features.dtype)
        negative_mass = jnp.sum(
            jnp.where(active & (targets == self.negative_label), weights, 0.0)
        )
        positive_mass = jnp.sum(
            jnp.where(active & (targets == self.positive_label), weights, 0.0)
        )
        enough = (negative_mass > 0.0) & (positive_mass > 0.0)
        input_valid = features_valid & targets_valid & weights_valid & enough
        safe_features = jnp.where(
            batch.feature_mask & jnp.isfinite(features),
            features,
            0.0,
        )
        safe_weights = jnp.where(active & jnp.isfinite(weights), weights, 0.0)
        mass = jnp.maximum(jnp.sum(safe_weights), jnp.finfo(features.dtype).tiny)
        head_weight = 0.1 * jax.random.normal(
            key,
            (self.feature_model.out_size,),
            dtype=features.dtype,
        )
        classifier = BinaryVariationalCircuitClassifier(
            self.feature_model,
            head_weight,
            jnp.asarray(0.0, dtype=features.dtype),
            self.negative_label,
            self.positive_label,
        )
        trainable, fixed = partition_trainable(classifier)
        optimizer = optax.adam(self.learning_rate)
        optimizer_state = optimizer.init(trainable)

        def objective(parameters):
            candidate = combine_trainable(parameters, fixed)
            logits = jax.vmap(candidate.decision_function)(safe_features)
            losses = jax.nn.softplus(logits) - encoded * logits
            data_loss = jnp.sum(safe_weights * losses) / mass
            return data_loss + self.l2_strength * tree_inner(parameters, parameters)

        value_and_grad = eqx.filter_value_and_grad(objective)

        def step(value, iteration):
            del iteration
            parameters, state = value
            loss, gradient = value_and_grad(parameters)
            updates, next_state = optimizer.update(gradient, state, parameters)
            next_parameters = eqx.apply_updates(parameters, updates)
            return (next_parameters, next_state), loss, tree_norm(gradient)

        iteration = run_fixed_iterations(
            (trainable, optimizer_state),
            step,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            method="variational_circuit_classifier",
        )
        fitted_trainable, _ = iteration.value
        fitted = combine_trainable(fitted_trainable, fixed)
        final_index = jnp.maximum(iteration.iterations - 1, 0)
        objective_value = iteration.objective_history[final_index]
        gradient_norm = iteration.residual_history[final_index]
        finite = iteration.finite & tree_allfinite(fitted_trainable)
        valid = input_valid & finite
        status = jnp.where(
            ~enough,
            ML_INSUFFICIENT_DATA,
            jnp.where(valid, ML_SUCCESS, ML_NONFINITE),
        )
        common = FitDiagnostics(
            valid=valid,
            status=status,
            objective=objective_value,
            iterations=iteration.iterations,
            effective_samples=jnp.sum(active),
            method="variational_circuit_classifier",
        )
        shift_evaluations = (
            1 + self.feature_model.execution.shift_plan.evaluation_count
            if self.feature_model.gradient_method == "parameter-shift"
            else 1
        )
        logical_evaluations = jnp.asarray(
            self.max_iterations * batch.sample_count * shift_evaluations,
            dtype=jnp.int64,
        )
        diagnostics = CircuitFitDiagnostics(
            common,
            gradient_norm,
            iteration.converged,
            logical_evaluations,
            self.feature_model.execution.shift_plan.occurrence_count,
            self.feature_model.gradient_method,
        )
        contract = GradientContract(
            prediction_inputs="conditional",
            prediction_parameters="conditional",
            fit_mode="stopped",
            nondifferentiable_outputs=("predict",),
            conditions=(
                "The dense quantum program and local observables remain valid.",
                "Parameter-shift mode certifies first-order Pauli-angle derivatives only.",
            ),
        )
        return FitResult(
            fitted,
            diagnostics,
            valid=valid,
            status=status,
            method="variational_circuit_classifier",
            gradient_contract=contract,
        )


__all__ = [
    "CircuitFeatureTransformRecipe",
    "CircuitFitDiagnostics",
    "FittedCircuitFeatureTransform",
    "VariationalCircuitClassifierRecipe",
]
