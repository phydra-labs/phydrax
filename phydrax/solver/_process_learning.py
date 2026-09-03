#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded Stinespring process learning on the complex Stiefel manifold."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix import ComplexStiefelManifold


class ProcessExperimentPlan(StrictModule):
    """Fixed experiment batch with complete POVMs and observed outcome counts."""

    input_densities: Array
    effects: Array
    observed_counts: Array
    input_trace_residuals: Array
    input_hermiticity_residuals: Array
    input_minimum_eigenvalues: Array
    effect_completeness_residuals: Array
    finite: Array
    valid: Array
    experiment_count: int = eqx.field(static=True)
    outcome_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        input_densities: ArrayLike,
        effects: ArrayLike,
        observed_counts: ArrayLike,
        /,
        *,
        tolerance: float = 1e-8,
    ):
        inputs = jnp.asarray(input_densities)
        measurements = jnp.asarray(effects)
        counts = jnp.asarray(observed_counts)
        tolerance_ = float(tolerance)
        if inputs.ndim != 3 or inputs.shape[0] < 1 or inputs.shape[1] != inputs.shape[2]:
            raise ValueError("input_densities require shape (experiments,d,d).")
        if measurements.ndim != 4 or measurements.shape[0] != inputs.shape[0]:
            raise ValueError("effects require shape (experiments,outcomes,d,d).")
        if measurements.shape[2:] != inputs.shape[1:] or measurements.shape[1] < 1:
            raise ValueError("Process effects have incompatible dimensions.")
        if counts.shape != measurements.shape[:2]:
            raise ValueError("observed_counts require shape (experiments,outcomes).")
        if not jnp.issubdtype(inputs.dtype, jnp.complexfloating) or not jnp.issubdtype(
            measurements.dtype, jnp.complexfloating
        ):
            raise TypeError("Process inputs/effects must use complex coordinates.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError(
                "Process experiment tolerance must be finite and nonnegative."
            )
        if not bool(jax.device_get(jnp.all(counts >= 0))) or not bool(
            jax.device_get(jnp.sum(counts) > 0)
        ):
            raise ValueError(
                "Observed counts must be nonnegative with positive total mass."
            )
        input_adjoint = jnp.swapaxes(jnp.conj(inputs), -1, -2)
        input_hermiticity = jnp.max(jnp.abs(inputs - input_adjoint), axis=(-2, -1))
        input_minimum = jnp.min(
            jnp.linalg.eigvalsh(0.5 * (inputs + input_adjoint)), axis=-1
        )
        trace_residual = jnp.abs(jnp.real(jnp.trace(inputs, axis1=-2, axis2=-1)) - 1.0)
        identity = jnp.eye(inputs.shape[-1], dtype=measurements.dtype)
        completeness = jnp.max(
            jnp.abs(jnp.sum(measurements, axis=1) - identity), axis=(-2, -1)
        )
        effect_adjoint = jnp.swapaxes(jnp.conj(measurements), -1, -2)
        effect_hermiticity = jnp.max(
            jnp.abs(measurements - effect_adjoint), axis=(-2, -1)
        )
        effect_minimum = jnp.min(
            jnp.linalg.eigvalsh(0.5 * (measurements + effect_adjoint)), axis=-1
        )
        finite = (
            jnp.all(jnp.isfinite(inputs))
            & jnp.all(jnp.isfinite(measurements))
            & jnp.all(jnp.isfinite(counts))
        )
        valid = (
            finite
            & jnp.all(trace_residual <= tolerance_)
            & jnp.all(input_hermiticity <= tolerance_)
            & jnp.all(input_minimum >= -tolerance_)
            & jnp.all(completeness <= tolerance_)
            & jnp.all(effect_hermiticity <= tolerance_)
            & jnp.all(effect_minimum >= -tolerance_)
        )
        self.input_densities = inputs
        self.effects = measurements
        self.observed_counts = counts
        self.input_trace_residuals = trace_residual
        self.input_hermiticity_residuals = input_hermiticity
        self.input_minimum_eigenvalues = input_minimum
        self.effect_completeness_residuals = completeness
        self.finite = finite
        self.valid = valid
        self.experiment_count = int(inputs.shape[0])
        self.outcome_count = int(measurements.shape[1])
        self.dimension = int(inputs.shape[-1])
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "process-experiment-plan",
                "input_shape": inputs.shape,
                "effect_shape": measurements.shape,
                "count_shape": counts.shape,
                "input_dtype": str(inputs.dtype),
                "effect_dtype": str(measurements.dtype),
                "tolerance": tolerance_,
            }
        )


class StinespringProcessModel(StrictModule):
    """A channel isometry V: input -> environment x output."""

    isometry: Array
    stiefel_residual: Array
    valid: Array
    dimension: int = eqx.field(static=True)
    environment_dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        isometry: ArrayLike,
        /,
        *,
        dimension: int,
        environment_dimension: int,
        tolerance: float = 1e-8,
    ):
        matrix = jnp.asarray(isometry)
        dimension_ = int(dimension)
        environment = int(environment_dimension)
        tolerance_ = float(tolerance)
        if dimension_ < 1 or environment < 1:
            raise ValueError("Process dimensions must be positive.")
        if matrix.shape != (environment * dimension_, dimension_):
            raise ValueError("Stinespring isometry has incompatible shape.")
        if not jnp.issubdtype(matrix.dtype, jnp.complexfloating):
            raise TypeError("Stinespring isometry must use complex coordinates.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Stinespring tolerance must be finite and nonnegative.")
        manifold = ComplexStiefelManifold(
            environment * dimension_, dimension_, tolerance=tolerance_
        )
        residual = manifold.constraint_residual(matrix)
        self.isometry = matrix
        self.stiefel_residual = residual
        self.valid = manifold.contains(matrix)
        self.dimension = dimension_
        self.environment_dimension = environment
        self.tolerance = tolerance_
        self.model_id = canonical_fingerprint(
            {
                "kind": "stinespring-process-model",
                "shape": matrix.shape,
                "dtype": str(matrix.dtype),
                "dimension": dimension_,
                "environment_dimension": environment,
                "tolerance": tolerance_,
            }
        )

    def kraus(self) -> Array:
        return self.isometry.reshape(
            (self.environment_dimension, self.dimension, self.dimension)
        )


def process_output_densities(
    model: StinespringProcessModel,
    input_densities: ArrayLike,
    /,
) -> Array:
    if not isinstance(model, StinespringProcessModel):
        raise TypeError("model must be StinespringProcessModel.")
    inputs = jnp.asarray(input_densities)
    if inputs.ndim != 3 or inputs.shape[1:] != (model.dimension, model.dimension):
        raise ValueError("input_densities have incompatible shape.")
    kraus = model.kraus()
    return ein.contract("kai,eij,kbj->eab", kraus, inputs, jnp.conj(kraus))


def process_experiment_probabilities(
    model: StinespringProcessModel,
    experiments: ProcessExperimentPlan,
    /,
) -> Array:
    if not isinstance(experiments, ProcessExperimentPlan):
        raise TypeError("experiments must be ProcessExperimentPlan.")
    if experiments.dimension != model.dimension:
        raise ValueError("Process model and experiment dimensions differ.")
    outputs = process_output_densities(model, experiments.input_densities)
    return jnp.real(ein.contract("eoij,eji->eo", experiments.effects, outputs))


def _negative_log_likelihood(
    matrix: Array,
    template: StinespringProcessModel,
    experiments: ProcessExperimentPlan,
    probability_floor: float,
) -> Array:
    model = StinespringProcessModel(
        matrix,
        dimension=template.dimension,
        environment_dimension=template.environment_dimension,
        tolerance=template.tolerance,
    )
    probabilities = process_experiment_probabilities(model, experiments)
    log_probabilities = jnp.log(
        jnp.where(probabilities > probability_floor, probabilities, probability_floor)
    )
    return -jnp.sum(experiments.observed_counts * log_probabilities) / jnp.sum(
        experiments.observed_counts
    )


class ProcessFitResult(StrictModule):
    model: StinespringProcessModel
    training_loss_history: Array
    tangent_norm_history: Array
    retraction_residual_history: Array
    training_probabilities: Array
    held_out_negative_log_likelihood: Array
    held_out_probabilities: Array
    training_probability_floor_activations: Array
    held_out_probability_floor_activations: Array
    stiefel_residual: Array
    finite: Array
    converged: Array
    valid: Array
    iterations: int = eqx.field(static=True)
    estimator: str = eqx.field(static=True)
    fit_id: str = eqx.field(static=True)


def fit_stinespring_process_model(
    initial_model: StinespringProcessModel,
    training: ProcessExperimentPlan,
    held_out: ProcessExperimentPlan,
    /,
    *,
    iterations: int,
    learning_rate: float,
    probability_floor: float = 1e-12,
    gradient_tolerance: float = 1e-7,
) -> ProcessFitResult:
    """Execute a fixed number of explicit Stiefel gradient/retraction steps."""
    if not isinstance(initial_model, StinespringProcessModel):
        raise TypeError("initial_model must be StinespringProcessModel.")
    if not isinstance(training, ProcessExperimentPlan) or not isinstance(
        held_out, ProcessExperimentPlan
    ):
        raise TypeError("training/held_out must be ProcessExperimentPlan values.")
    count = int(iterations)
    rate = float(learning_rate)
    floor = float(probability_floor)
    gradient_tolerance_ = float(gradient_tolerance)
    if count < 1:
        raise ValueError("Process fitting iterations must be positive.")
    if any(
        not isfinite(value) or value <= 0.0
        for value in (rate, floor, gradient_tolerance_)
    ):
        raise ValueError("Process fitting rates/tolerances must be finite and positive.")
    if (
        training.dimension != initial_model.dimension
        or held_out.dimension != initial_model.dimension
    ):
        raise ValueError("Process model and experiment dimensions differ.")
    manifold = ComplexStiefelManifold(
        initial_model.environment_dimension * initial_model.dimension,
        initial_model.dimension,
        tolerance=initial_model.tolerance,
    )
    objective = lambda matrix: _negative_log_likelihood(
        matrix, initial_model, training, floor
    )
    value_and_gradient = jax.value_and_grad(objective)
    matrix = initial_model.isometry
    losses: list[Array] = []
    tangent_norms: list[Array] = []
    retraction_residuals: list[Array] = []
    for _ in range(count):
        loss, ambient_gradient = value_and_gradient(matrix)
        tangent = manifold.project_tangent(matrix, ambient_gradient)
        matrix = manifold.retract(matrix, -rate * tangent)
        losses.append(loss)
        tangent_norms.append(jnp.linalg.norm(tangent))
        retraction_residuals.append(manifold.constraint_residual(matrix))
    losses.append(objective(matrix))
    model = StinespringProcessModel(
        matrix,
        dimension=initial_model.dimension,
        environment_dimension=initial_model.environment_dimension,
        tolerance=initial_model.tolerance,
    )
    training_probabilities = process_experiment_probabilities(model, training)
    held_out_probabilities = process_experiment_probabilities(model, held_out)
    held_out_log = jnp.log(
        jnp.where(held_out_probabilities > floor, held_out_probabilities, floor)
    )
    held_out_nll = -jnp.sum(held_out.observed_counts * held_out_log) / jnp.sum(
        held_out.observed_counts
    )
    training_floor_activations = jnp.sum(
        (training_probabilities <= floor) & (training.observed_counts > 0),
        dtype=jnp.int32,
    )
    held_out_floor_activations = jnp.sum(
        (held_out_probabilities <= floor) & (held_out.observed_counts > 0),
        dtype=jnp.int32,
    )
    losses_ = jnp.stack(losses)
    tangent_norms_ = jnp.stack(tangent_norms)
    retraction_residuals_ = jnp.stack(retraction_residuals)
    finite = (
        jnp.all(jnp.isfinite(losses_))
        & jnp.all(jnp.isfinite(tangent_norms_))
        & jnp.all(jnp.isfinite(retraction_residuals_))
        & jnp.isfinite(held_out_nll)
        & jnp.all(jnp.isfinite(held_out_probabilities))
        & jnp.all(jnp.isfinite(training_probabilities))
    )
    converged = finite & (tangent_norms_[-1] <= gradient_tolerance_)
    identifier = canonical_fingerprint(
        {
            "kind": "stinespring-process-fit",
            "model": initial_model.model_id,
            "training": training.plan_id,
            "held_out": held_out.plan_id,
            "iterations": count,
            "learning_rate": rate,
            "probability_floor": floor,
        }
    )
    return ProcessFitResult(
        model,
        losses_,
        tangent_norms_,
        retraction_residuals_,
        training_probabilities,
        held_out_nll,
        held_out_probabilities,
        training_floor_activations,
        held_out_floor_activations,
        model.stiefel_residual,
        finite,
        converged,
        finite & training.valid & held_out.valid & model.valid,
        count,
        "full-batch-Stiefel-gradient-with-explicit-QR-retraction",
        identifier,
    )


class QuantumDigitalTwinState(StrictModule, NonTrainableState):
    """Immutable checkpoint leaf set for deterministic process-fit continuation."""

    model: StinespringProcessModel
    training_loss_history: Array
    tangent_norm_history: Array
    held_out_negative_log_likelihood: Array
    completed_iterations: Array
    fit_valid: Array
    checkpoint_id: str = eqx.field(static=True)

    def __init__(self, result: ProcessFitResult, /, *, prior_iterations: int = 0):
        if not isinstance(result, ProcessFitResult):
            raise TypeError("result must be ProcessFitResult.")
        completed = int(prior_iterations) + result.iterations
        self.model = result.model
        self.training_loss_history = result.training_loss_history
        self.tangent_norm_history = result.tangent_norm_history
        self.held_out_negative_log_likelihood = result.held_out_negative_log_likelihood
        self.completed_iterations = jnp.asarray(completed, dtype=jnp.int32)
        self.fit_valid = result.valid
        self.checkpoint_id = canonical_fingerprint(
            {
                "arrays": array_tree_fingerprint(
                    (
                        result.model.isometry,
                        result.training_loss_history,
                        result.tangent_norm_history,
                        result.held_out_negative_log_likelihood,
                    )
                ),
                "kind": "quantum-digital-twin-checkpoint",
                "model": result.model.model_id,
                "completed_iterations": completed,
                "history_shape": result.training_loss_history.shape,
                "history_dtype": str(result.training_loss_history.dtype),
            }
        )


__all__ = [
    "ProcessExperimentPlan",
    "ProcessFitResult",
    "QuantumDigitalTwinState",
    "StinespringProcessModel",
    "fit_stinespring_process_model",
    "process_experiment_probabilities",
    "process_output_densities",
]
