#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, ArrayLike, Key
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._training import TrainingController, TrainingProgress
from .._dynamics import PreparedAtomisticDynamics
from ._bias import (
    AbstractAtomisticBiasPlan,
    AbstractAtomisticBiasState,
    AbstractPreparedAtomisticBias,
    AtomisticBiasEvaluation,
)
from ._collective_variable import (
    AbstractCollectiveVariableProgram,
    CollectiveVariableMetric,
)


class MeanForceData(StrictModule, NonTrainableState):
    centers: Array
    free_energy_gradients: Array
    gradient_standard_error: Array
    weights: Array
    valid: Array
    metrics: tuple[CollectiveVariableMetric, ...]
    source_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        centers: ArrayLike,
        free_energy_gradients: ArrayLike,
        /,
        *,
        gradient_standard_error: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        valid: ArrayLike | None = None,
        metrics: tuple[CollectiveVariableMetric, ...] | None = None,
        source_id: str,
    ):
        center = np.asarray(centers, dtype=float)
        gradient = np.asarray(free_energy_gradients, dtype=float)
        if center.ndim != 2 or center.shape[0] == 0 or gradient.shape != center.shape:
            raise ValueError(
                "Mean-force centers and gradients require shape (sample, cv)."
            )
        error = (
            np.ones_like(center)
            if gradient_standard_error is None
            else np.asarray(gradient_standard_error, dtype=float)
        )
        mass = (
            np.ones((center.shape[0],), dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        support = (
            np.ones((center.shape[0],), dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if (
            error.shape != center.shape
            or mass.shape != support.shape
            or mass.shape != center.shape[:1]
        ):
            raise ValueError("Mean-force errors, weights, and validity do not align.")
        if np.any(~np.isfinite(center[support])) or np.any(
            ~np.isfinite(gradient[support])
        ):
            raise ValueError("Valid mean-force centers and gradients must be finite.")
        if np.any(~np.isfinite(error[support])) or np.any(error[support] <= 0.0):
            raise ValueError(
                "Valid mean-force standard errors must be finite and positive."
            )
        if (
            np.any(~np.isfinite(mass))
            or np.any(mass < 0.0)
            or np.sum(mass[support]) <= 0.0
        ):
            raise ValueError("Mean-force weights must contain positive finite support.")
        resolved_metrics = (
            tuple(CollectiveVariableMetric() for _ in range(center.shape[1]))
            if metrics is None
            else tuple(metrics)
        )
        if len(resolved_metrics) != center.shape[1] or any(
            not isinstance(metric, CollectiveVariableMetric)
            for metric in resolved_metrics
        ):
            raise TypeError("metrics must contain one CV metric per coordinate.")
        identifier = str(source_id).strip()
        if not identifier:
            raise ValueError("source_id must be non-empty.")
        self.centers = jnp.asarray(center)
        self.free_energy_gradients = jnp.asarray(gradient)
        self.gradient_standard_error = jnp.asarray(error)
        self.weights = jnp.asarray(mass)
        self.valid = jnp.asarray(support)
        self.metrics = resolved_metrics
        self.source_id = identifier
        self.dataset_id = canonical_fingerprint(
            {
                "kind": "mean-force-data",
                "source": identifier,
                "metrics": [metric.metric_id for metric in resolved_metrics],
                "arrays": array_tree_fingerprint(
                    {
                        "centers": center,
                        "gradients": gradient,
                        "standard_error": error,
                        "weights": mass,
                        "valid": support,
                    }
                ),
            }
        )


class RestrainedMeanForcePlan(StrictModule, NonTrainableState):
    centers: Array
    stiffness: Array
    metrics: tuple[CollectiveVariableMetric, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        centers: ArrayLike,
        stiffness: ArrayLike,
        /,
        *,
        metrics: tuple[CollectiveVariableMetric, ...] | None = None,
    ):
        center = np.asarray(centers, dtype=float)
        if center.ndim != 2 or center.shape[0] == 0:
            raise ValueError("Restrained centers require shape (window, cv).")
        stiffness_ = np.asarray(stiffness, dtype=float)
        if stiffness_.size == 1:
            stiffness_ = np.full((center.shape[1],), float(stiffness_.reshape(())))
        else:
            stiffness_ = stiffness_.reshape((-1,))
        if (
            stiffness_.shape != center.shape[1:]
            or np.any(~np.isfinite(stiffness_))
            or np.any(stiffness_ <= 0.0)
        ):
            raise ValueError("Restrained stiffness must be positive and align with CVs.")
        resolved_metrics = (
            tuple(CollectiveVariableMetric() for _ in range(center.shape[1]))
            if metrics is None
            else tuple(metrics)
        )
        if len(resolved_metrics) != center.shape[1]:
            raise ValueError("Restrained metrics must align with CVs.")
        self.centers = jnp.asarray(center)
        self.stiffness = jnp.asarray(stiffness_)
        self.metrics = resolved_metrics
        self.plan_id = canonical_fingerprint(
            {
                "kind": "restrained-mean-force",
                "metrics": [metric.metric_id for metric in resolved_metrics],
                "arrays": array_tree_fingerprint(
                    {"centers": center, "stiffness": stiffness_}
                ),
                "approximation": "finite-stiffness-restrained-gradient",
            }
        )


def estimate_restrained_free_energy_gradient(
    plan: RestrainedMeanForcePlan,
    samples: ArrayLike,
    /,
    *,
    sample_weights: ArrayLike | None = None,
    sample_valid: ArrayLike | None = None,
    source_id: str,
) -> MeanForceData:
    """Estimate free-energy gradients from finite-stiffness restrained windows."""

    if not isinstance(plan, RestrainedMeanForcePlan):
        raise TypeError("plan must be RestrainedMeanForcePlan.")
    values = jnp.asarray(samples, dtype=plan.centers.dtype)
    expected = (
        (plan.centers.shape[0], values.shape[1], plan.centers.shape[1])
        if values.ndim == 3
        else ()
    )
    if values.ndim != 3 or values.shape != expected:
        raise ValueError("samples must have shape (window, sample, cv).")
    weights = (
        jnp.ones(values.shape[:2], dtype=values.dtype)
        if sample_weights is None
        else jnp.asarray(sample_weights, dtype=values.dtype)
    )
    valid = (
        jnp.ones(values.shape[:2], dtype=bool)
        if sample_valid is None
        else jnp.asarray(sample_valid, dtype=bool)
    )
    if weights.shape != values.shape[:2] or valid.shape != values.shape[:2]:
        raise ValueError(
            "Restrained sample weights and validity must align with windows."
        )
    finite = (
        jnp.all(jnp.isfinite(values), axis=-1) & jnp.isfinite(weights) & (weights >= 0.0)
    )
    active = valid & finite
    weight = jnp.where(active, weights, 0.0)
    delta_components = tuple(
        metric.difference(values[..., index], plan.centers[:, None, index])
        for index, metric in enumerate(plan.metrics)
    )
    delta = jnp.stack(delta_components, axis=-1)
    total = jnp.sum(weight, axis=1)
    mean_delta = contract("ws,wsc->wc", weight, delta) / jnp.maximum(total[:, None], 1.0)
    centered = jnp.where(active[..., None], delta - mean_delta[:, None, :], 0.0)
    variance = contract("ws,wsc,wsc->wc", weight, centered, centered) / jnp.maximum(
        total[:, None] - 1.0, 1.0
    )
    weight_square = jnp.sum(weight * weight, axis=1)
    effective = jnp.where(weight_square > 0.0, total * total / weight_square, 0.0)
    gradient = -plan.stiffness[None, :] * mean_delta
    standard_error = plan.stiffness[None, :] * jnp.sqrt(
        variance / jnp.maximum(effective[:, None], 1.0)
    )
    tolerance = jnp.sqrt(jnp.finfo(values.dtype).eps)
    standard_error = jnp.maximum(standard_error, tolerance)
    window_valid = (total > 0.0) & jnp.all(jnp.isfinite(gradient), axis=-1)
    return MeanForceData(
        plan.centers,
        gradient,
        gradient_standard_error=standard_error,
        weights=effective,
        valid=window_valid,
        metrics=plan.metrics,
        source_id=f"{source_id}:{plan.plan_id}",
    )


class FreeEnergyTrainingPolicy(StrictModule, NonTrainableState):
    maximum_steps: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    validation_interval: int = eqx.field(static=True)
    patience: int | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int = 1000,
        learning_rate: float = 1.0e-3,
        validation_interval: int = 10,
        patience: int | None = None,
    ):
        steps = int(maximum_steps)
        rate = float(learning_rate)
        interval = int(validation_interval)
        patience_ = None if patience is None else int(patience)
        if steps < 0 or interval <= 0 or not isfinite(rate) or rate <= 0.0:
            raise ValueError("Free-energy training controls are invalid.")
        if patience_ is not None and patience_ <= 0:
            raise ValueError("patience must be positive when provided.")
        self.maximum_steps = steps
        self.learning_rate = rate
        self.validation_interval = interval
        self.patience = patience_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "free-energy-training-policy",
                "maximum_steps": steps,
                "learning_rate": rate.hex(),
                "validation_interval": interval,
                "patience": patience_,
            }
        )


class FreeEnergyFitResult(StrictModule):
    model: AbstractArrayModel
    training_loss: Array
    validation_loss: Array
    progress: TrainingProgress
    valid: Array
    model_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _free_energy_loss(
    model: AbstractArrayModel, data: MeanForceData, /
) -> tuple[Array, Array]:
    def model_gradient(point):
        return jax.grad(lambda value: jnp.asarray(model(value, key=None)).reshape(()))(
            point
        )

    predicted = jax.vmap(model_gradient)(data.centers)
    residual = predicted - data.free_energy_gradients
    precision = data.weights[:, None] / (data.gradient_standard_error**2)
    active = data.valid[:, None] & jnp.isfinite(residual)
    weighted = jnp.where(active, precision * residual * residual, 0.0)
    normalizer = jnp.sum(jnp.where(active, precision, 0.0))
    loss = jnp.sum(weighted) / jnp.maximum(normalizer, 1.0)
    valid = (
        (normalizer > 0.0)
        & jnp.isfinite(loss)
        & jnp.all(jnp.isfinite(jnp.where(active, predicted, 0.0)))
    )
    return loss, valid


def fit_free_energy_model(
    model: AbstractArrayModel,
    data: MeanForceData,
    key: Key[Array, ""],
    /,
    *,
    model_id: str,
    policy: FreeEnergyTrainingPolicy | None = None,
    validation_data: MeanForceData | None = None,
    optimizer: optax.GradientTransformation | None = None,
) -> FreeEnergyFitResult:
    if not isinstance(model, AbstractArrayModel) or not isinstance(data, MeanForceData):
        raise TypeError("model and data must satisfy free-energy training contracts.")
    if model.in_size != data.centers.shape[1] or model.out_size != 1:
        raise ValueError("Free-energy model must map CV coordinates to one scalar.")
    identifier = str(model_id).strip()
    if not identifier:
        raise ValueError("model_id must be non-empty.")
    policy_ = FreeEnergyTrainingPolicy() if policy is None else policy
    if not isinstance(policy_, FreeEnergyTrainingPolicy):
        raise TypeError("policy must be FreeEnergyTrainingPolicy or None.")
    validation = data if validation_data is None else validation_data
    if validation.centers.shape[1] != data.centers.shape[1]:
        raise ValueError("Training and validation CV dimensions differ.")
    optimizer_ = optax.adam(policy_.learning_rate) if optimizer is None else optimizer
    state = optimizer_.init(eqx.filter(model, eqx.is_inexact_array))
    controller = TrainingController(total_steps=policy_.maximum_steps, key=key)

    @eqx.filter_jit
    def update(current, optimizer_state):
        def objective(candidate):
            return _free_energy_loss(candidate, data)

        (loss, valid), gradient = eqx.filter_value_and_grad(objective, has_aux=True)(
            current
        )
        updates, next_state = optimizer_.update(gradient, optimizer_state, current)
        return eqx.apply_updates(current, updates), next_state, loss, valid

    current = model
    training_history: list[Array] = []
    validation_history: list[Array] = []
    initial_validation, initial_valid = _free_energy_loss(current, validation)
    controller.select(
        float(initial_validation), current, step=0, mode="min", patience=policy_.patience
    )
    valid = initial_valid
    for step in range(1, policy_.maximum_steps + 1):
        current, state, training_loss, step_valid = update(current, state)
        controller.complete_update(step)
        if step % policy_.validation_interval == 0 or step == policy_.maximum_steps:
            validation_loss, validation_valid = _free_energy_loss(current, validation)
            valid = step_valid & validation_valid
            training_history.append(training_loss)
            validation_history.append(validation_loss)
            if not bool(valid):
                break
            controller.select(
                float(validation_loss),
                current,
                step=step,
                mode="min",
                patience=policy_.patience,
            )
            if controller.stop_requested:
                break
    selected = controller.selected(current)
    training_values = jnp.asarray(training_history)
    validation_values = jnp.asarray(validation_history)
    valid = (
        valid
        & jnp.all(jnp.isfinite(training_values))
        & jnp.all(jnp.isfinite(validation_values))
    )
    result_id = canonical_fingerprint(
        {
            "kind": "free-energy-fit",
            "model": identifier,
            "dataset": data.dataset_id,
            "validation": validation.dataset_id,
            "policy": policy_.policy_id,
        }
    )
    return FreeEnergyFitResult(
        model=selected,
        training_loss=training_values,
        validation_loss=validation_values,
        progress=controller.progress,
        valid=valid,
        model_id=identifier,
        dataset_id=data.dataset_id,
        result_id=result_id,
    )


class LearnedFreeEnergyBiasState(AbstractAtomisticBiasState):
    successful: Array
    bias_id: str = eqx.field(static=True)


class LearnedFreeEnergyBiasPlan(AbstractAtomisticBiasPlan):
    variables: AbstractCollectiveVariableProgram
    models: tuple[AbstractArrayModel, ...]
    model_ids: tuple[str, ...] = eqx.field(static=True)
    reference: Array
    offsets: Array
    bias_fraction: float = eqx.field(static=True)
    trusted_uncertainty: float = eqx.field(static=True)
    rejected_uncertainty: float = eqx.field(static=True)
    bias_id: str = eqx.field(static=True)

    def __init__(
        self,
        variables: AbstractCollectiveVariableProgram,
        models: tuple[AbstractArrayModel, ...],
        /,
        *,
        model_ids: tuple[str, ...],
        reference: ArrayLike,
        bias_fraction: float = 1.0,
        trusted_uncertainty: float = 0.0,
        rejected_uncertainty: float = 1.0,
    ):
        if not isinstance(variables, AbstractCollectiveVariableProgram):
            raise TypeError("variables must implement AbstractCollectiveVariableProgram.")
        members = tuple(models)
        identifiers = tuple(str(value).strip() for value in model_ids)
        if (
            not members
            or len(members) != len(identifiers)
            or any(not value for value in identifiers)
        ):
            raise ValueError("A learned bias requires aligned models and model IDs.")
        if any(
            not isinstance(model, AbstractArrayModel)
            or model.in_size != variables.output_size
            or model.out_size != 1
            for model in members
        ):
            raise ValueError("Bias models must map the CV program to one scalar.")
        reference_ = jnp.asarray(reference, dtype=float).reshape((-1,))
        if reference_.shape != (variables.output_size,) or not bool(
            jnp.all(jnp.isfinite(reference_))
        ):
            raise ValueError("Bias reference must be a finite CV vector.")
        fraction = float(bias_fraction)
        trusted = float(trusted_uncertainty)
        rejected = float(rejected_uncertainty)
        if (
            not all(isfinite(value) for value in (fraction, trusted, rejected))
            or fraction < 0.0
            or fraction > 1.0
            or trusted < 0.0
            or rejected <= trusted
        ):
            raise ValueError("Bias fraction or uncertainty thresholds are invalid.")
        offsets = jnp.stack(
            tuple(
                jnp.asarray(model(reference_, key=None)).reshape(()) for model in members
            )
        )
        self.variables = variables
        self.models = members
        self.model_ids = identifiers
        self.reference = reference_
        self.offsets = offsets
        self.bias_fraction = fraction
        self.trusted_uncertainty = trusted
        self.rejected_uncertainty = rejected
        self.bias_id = canonical_fingerprint(
            {
                "kind": "learned-free-energy-bias",
                "variables": variables.program_id,
                "models": list(identifiers),
                "reference": np.asarray(reference_).tolist(),
                "bias_fraction": fraction.hex(),
                "trusted_uncertainty": trusted.hex(),
                "rejected_uncertainty": rejected.hex(),
            }
        )

    def initialize(self, dtype=jnp.float64) -> LearnedFreeEnergyBiasState:
        return LearnedFreeEnergyBiasState(
            successful=jnp.asarray(True, dtype=bool), bias_id=self.bias_id
        )

    def prepare(
        self, dynamics: PreparedAtomisticDynamics, /
    ) -> "PreparedLearnedFreeEnergyBias":
        return PreparedLearnedFreeEnergyBias(self, dynamics)


class PreparedLearnedFreeEnergyBias(AbstractPreparedAtomisticBias):
    plan: LearnedFreeEnergyBiasPlan
    dynamics: PreparedAtomisticDynamics
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: LearnedFreeEnergyBiasPlan,
        dynamics: PreparedAtomisticDynamics,
        /,
    ):
        if not isinstance(plan, LearnedFreeEnergyBiasPlan):
            raise TypeError("plan must be LearnedFreeEnergyBiasPlan.")
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        self.plan = plan
        self.dynamics = dynamics
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-learned-free-energy-bias",
                "plan": plan.bias_id,
                "dynamics": dynamics.prepared_id,
            }
        )

    def energy(
        self,
        positions: Array,
        state: LearnedFreeEnergyBiasState,
        time: Array,
        /,
    ):
        del time
        values, valid = self.plan.variables.evaluate(
            positions, cell=self.dynamics.system.cell
        )
        predictions = jnp.stack(
            tuple(
                jnp.asarray(model(values, key=None)).reshape(()) - offset
                for model, offset in zip(self.plan.models, self.plan.offsets, strict=True)
            )
        )
        mean = jnp.mean(predictions)
        variance = jnp.mean((predictions - mean) ** 2)
        uncertainty = jnp.sqrt(variance + jnp.asarray(jnp.finfo(predictions.dtype).tiny))
        coordinate = jnp.clip(
            (uncertainty - self.plan.trusted_uncertainty)
            / (self.plan.rejected_uncertainty - self.plan.trusted_uncertainty),
            0.0,
            1.0,
        )
        trust = 1.0 - coordinate * coordinate * (3.0 - 2.0 * coordinate)
        energy = -self.plan.bias_fraction * trust * mean
        successful = (
            valid
            & state.successful
            & jnp.all(jnp.isfinite(predictions))
            & jnp.isfinite(energy)
        )
        return energy, (values, successful, uncertainty, trust)

    def evaluate(
        self,
        positions: Array,
        state: LearnedFreeEnergyBiasState,
        time: Array,
        /,
    ) -> AtomisticBiasEvaluation:
        (energy, auxiliary), gradient = jax.value_and_grad(
            lambda value: self.energy(value, state, time), has_aux=True
        )(positions)
        values, successful, uncertainty, trust = auxiliary
        successful = successful & jnp.all(jnp.isfinite(gradient))
        return AtomisticBiasEvaluation(
            energy,
            -gradient,
            values,
            successful,
            jnp.zeros((values.shape[0],) + positions.shape, dtype=positions.dtype),
            uncertainty,
            trust,
            state,
            self.prepared_id,
        )

    def update(
        self,
        state: LearnedFreeEnergyBiasState,
        evaluation: AtomisticBiasEvaluation,
        physical_forces: Array,
        /,
    ) -> LearnedFreeEnergyBiasState:
        del physical_forces
        return LearnedFreeEnergyBiasState(
            successful=state.successful & evaluation.successful,
            bias_id=state.bias_id,
        )


__all__ = [
    "FreeEnergyFitResult",
    "FreeEnergyTrainingPolicy",
    "LearnedFreeEnergyBiasPlan",
    "LearnedFreeEnergyBiasState",
    "MeanForceData",
    "PreparedLearnedFreeEnergyBias",
    "RestrainedMeanForcePlan",
    "estimate_restrained_free_energy_gradient",
    "fit_free_energy_model",
]
