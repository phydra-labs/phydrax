#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import combine_trainable, NonTrainableState, partition_trainable
from .._training import TrainingCallback, TrainingController, TrainingProgress
from ..nn.atomistic._painn import PaiNNPotential
from ._types import AtomisticBatch, AtomisticStatus


class AtomisticTrainingProblem(StrictModule, NonTrainableState):
    """Typed energy/force supervision for one train and optional validation split."""

    training_batch: AtomisticBatch
    training_energy: Array | None
    training_forces: Array | None
    training_energy_mask: Array | None
    training_force_mask: Array | None
    validation_batch: AtomisticBatch | None
    validation_energy: Array | None
    validation_forces: Array | None
    validation_energy_mask: Array | None
    validation_force_mask: Array | None
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        training_batch: AtomisticBatch,
        /,
        *,
        training_energy: ArrayLike | None = None,
        training_forces: ArrayLike | None = None,
        training_energy_mask: ArrayLike | None = None,
        training_force_mask: ArrayLike | None = None,
        validation_batch: AtomisticBatch | None = None,
        validation_energy: ArrayLike | None = None,
        validation_forces: ArrayLike | None = None,
        validation_energy_mask: ArrayLike | None = None,
        validation_force_mask: ArrayLike | None = None,
    ):
        if not isinstance(training_batch, AtomisticBatch):
            raise TypeError("training_batch must be an AtomisticBatch.")
        if training_energy is None and training_forces is None:
            raise ValueError("Training requires energy labels, force labels, or both.")
        train_energy, train_energy_mask = _energy_supervision(
            training_batch,
            training_energy,
            training_energy_mask,
            prefix="training",
        )
        train_forces, train_force_mask = _force_supervision(
            training_batch,
            training_forces,
            training_force_mask,
            prefix="training",
        )
        if validation_batch is None:
            if any(
                value is not None
                for value in (
                    validation_energy,
                    validation_forces,
                    validation_energy_mask,
                    validation_force_mask,
                )
            ):
                raise ValueError("Validation labels and masks require validation_batch.")
            valid_energy = None
            valid_forces = None
            valid_energy_mask = None
            valid_force_mask = None
        else:
            if not isinstance(validation_batch, AtomisticBatch):
                raise TypeError("validation_batch must be an AtomisticBatch or None.")
            if validation_batch.scale.scale_id != training_batch.scale.scale_id:
                raise ValueError("Training and validation batches must share one scale.")
            if (validation_energy is None) != (training_energy is None) or (
                validation_forces is None
            ) != (training_forces is None):
                raise ValueError(
                    "Validation must provide the same energy/force target kinds as training."
                )
            valid_energy, valid_energy_mask = _energy_supervision(
                validation_batch,
                validation_energy,
                validation_energy_mask,
                prefix="validation",
            )
            valid_forces, valid_force_mask = _force_supervision(
                validation_batch,
                validation_forces,
                validation_force_mask,
                prefix="validation",
            )
        self.training_batch = training_batch
        self.training_energy = train_energy
        self.training_forces = train_forces
        self.training_energy_mask = train_energy_mask
        self.training_force_mask = train_force_mask
        self.validation_batch = validation_batch
        self.validation_energy = valid_energy
        self.validation_forces = valid_forces
        self.validation_energy_mask = valid_energy_mask
        self.validation_force_mask = valid_force_mask
        self.problem_id = canonical_fingerprint(
            {
                "kind": "atomistic-training-problem",
                "training_batch": training_batch.batch_id,
                "validation_batch": (
                    None if validation_batch is None else validation_batch.batch_id
                ),
                "targets": array_tree_fingerprint(
                    {
                        "training_energy": None
                        if train_energy is None
                        else np.asarray(train_energy),
                        "training_forces": None
                        if train_forces is None
                        else np.asarray(train_forces),
                        "training_energy_mask": None
                        if train_energy_mask is None
                        else np.asarray(train_energy_mask),
                        "training_force_mask": None
                        if train_force_mask is None
                        else np.asarray(train_force_mask),
                        "validation_energy": None
                        if valid_energy is None
                        else np.asarray(valid_energy),
                        "validation_forces": None
                        if valid_forces is None
                        else np.asarray(valid_forces),
                    }
                ),
            }
        )


class AtomisticTrainingPolicy(StrictModule, NonTrainableState):
    """Domain-specific full-batch Adam, scaling, selection, and stopping policy."""

    maximum_steps: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    energy_weight: float = eqx.field(static=True)
    force_weight: float = eqx.field(static=True)
    energy_scale: float | None = eqx.field(static=True)
    force_scale: float | None = eqx.field(static=True)
    normalization_floor: float = eqx.field(static=True)
    validation_every: int = eqx.field(static=True)
    patience: int | None = eqx.field(static=True)
    min_delta: float = eqx.field(static=True)
    select_best: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    continuation_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int,
        learning_rate: float = 1e-3,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        energy_scale: float | None = None,
        force_scale: float | None = None,
        normalization_floor: float = 1e-12,
        validation_every: int = 1,
        patience: int | None = None,
        min_delta: float = 0.0,
        select_best: bool = True,
    ):
        steps = int(maximum_steps)
        rate = float(learning_rate)
        energy_w = float(energy_weight)
        force_w = float(force_weight)
        energy_s = None if energy_scale is None else float(energy_scale)
        force_s = None if force_scale is None else float(force_scale)
        floor = float(normalization_floor)
        every = int(validation_every)
        patience_ = None if patience is None else int(patience)
        delta = float(min_delta)
        if steps < 0:
            raise ValueError("maximum_steps must be non-negative.")
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if (
            not math.isfinite(energy_w)
            or energy_w < 0.0
            or not math.isfinite(force_w)
            or force_w < 0.0
            or energy_w + force_w <= 0.0
        ):
            raise ValueError("Loss weights must be finite, non-negative, and not both zero.")
        for name, value in (("energy_scale", energy_s), ("force_scale", force_s)):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when provided.")
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("normalization_floor must be finite and positive.")
        if every <= 0:
            raise ValueError("validation_every must be positive.")
        if patience_ is not None and patience_ <= 0:
            raise ValueError("patience must be positive when provided.")
        if not math.isfinite(delta) or delta < 0.0:
            raise ValueError("min_delta must be finite and non-negative.")
        continuation_data = {
            "kind": "atomistic-training-continuation-policy",
            "optimizer": "adam",
            "learning_rate": rate,
            "energy_weight": energy_w,
            "force_weight": force_w,
            "energy_scale": energy_s,
            "force_scale": force_s,
            "normalization_floor": floor,
            "validation_every": every,
            "patience": patience_,
            "min_delta": delta,
            "select_best": bool(select_best),
        }
        self.maximum_steps = steps
        self.learning_rate = rate
        self.energy_weight = energy_w
        self.force_weight = force_w
        self.energy_scale = energy_s
        self.force_scale = force_s
        self.normalization_floor = floor
        self.validation_every = every
        self.patience = patience_
        self.min_delta = delta
        self.select_best = bool(select_best)
        self.continuation_id = canonical_fingerprint(continuation_data)
        self.policy_id = canonical_fingerprint(
            {**continuation_data, "maximum_steps": steps}
        )


class AtomisticTrainingNormalization(StrictModule, NonTrainableState):
    """Loss normalization fitted exclusively from the training split."""

    energy_per_atom_mean: Array
    energy_per_atom_scale: Array
    force_component_scale: Array
    fitted_from_problem_id: str = eqx.field(static=True)
    normalization_id: str = eqx.field(static=True)


class AtomisticTrainingResult(StrictModule, NonTrainableState):
    """Complete continuation, best/final model, histories, and terminal status."""

    potential: PaiNNPotential
    best_potential: PaiNNPotential
    optimizer_state: Any
    key: Array
    normalization: AtomisticTrainingNormalization
    training_loss_history: Array
    energy_loss_history: Array
    force_loss_history: Array
    validation_loss_history: Array
    validation_steps: Array
    final_loss: Array
    best_loss: Array
    status: Array
    progress: TrainingProgress = eqx.field(static=True)
    termination: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    continuation_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def final_potential(self) -> PaiNNPotential:
        return self.potential

    @property
    def successful(self) -> Array:
        return (self.status == int(AtomisticStatus.SUCCESS)) | (
            self.status == int(AtomisticStatus.STOPPED_EARLY)
        )


def _energy_supervision(
    batch: AtomisticBatch,
    values: ArrayLike | None,
    mask: ArrayLike | None,
    /,
    *,
    prefix: str,
) -> tuple[Array | None, Array | None]:
    if values is None:
        if mask is not None:
            raise ValueError(f"{prefix}_energy_mask requires energy labels.")
        return None, None
    energy = jnp.asarray(values, dtype=batch.positions.dtype)
    if energy.shape != (batch.case_count,):
        raise ValueError(f"{prefix}_energy must have shape (case,).")
    energy_mask = (
        jnp.ones((batch.case_count,), dtype=bool)
        if mask is None
        else jnp.asarray(mask, dtype=bool)
    )
    if energy_mask.shape != energy.shape:
        raise ValueError(f"{prefix}_energy_mask must have shape (case,).")
    if not np.any(np.asarray(energy_mask)):
        raise ValueError(f"{prefix}_energy_mask must select at least one label.")
    return energy, energy_mask


def _force_supervision(
    batch: AtomisticBatch,
    values: ArrayLike | None,
    mask: ArrayLike | None,
    /,
    *,
    prefix: str,
) -> tuple[Array | None, Array | None]:
    if values is None:
        if mask is not None:
            raise ValueError(f"{prefix}_force_mask requires force labels.")
        return None, None
    forces = jnp.asarray(values, dtype=batch.positions.dtype)
    if forces.shape != batch.positions.shape:
        raise ValueError(f"{prefix}_forces must have shape (case, atom, 3).")
    active = jnp.broadcast_to(batch.atom_mask[:, :, None], forces.shape)
    force_mask = active if mask is None else jnp.asarray(mask, dtype=bool) & active
    if force_mask.shape != forces.shape:
        raise ValueError(f"{prefix}_force_mask must have shape (case, atom, 3).")
    if not np.any(np.asarray(force_mask)):
        raise ValueError(f"{prefix}_force_mask must select at least one component.")
    return forces, force_mask


def _normalization(
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
) -> AtomisticTrainingNormalization:
    if problem.training_energy is None:
        energy_mean = 0.0
        fitted_energy_scale = 1.0
    else:
        values = np.asarray(problem.training_energy) / np.asarray(
            problem.training_batch.atom_counts
        )
        mask = np.asarray(problem.training_energy_mask, dtype=bool)
        selected = values[mask]
        if np.all(np.isfinite(selected)):
            energy_mean = float(np.mean(selected))
            fitted_energy_scale = float(np.std(selected))
        else:
            energy_mean = 0.0
            fitted_energy_scale = 1.0
    if problem.training_forces is None:
        fitted_force_scale = 1.0
    else:
        values = np.asarray(problem.training_forces)
        mask = np.asarray(problem.training_force_mask, dtype=bool)
        selected = values[mask]
        fitted_force_scale = (
            float(np.sqrt(np.mean(selected * selected)))
            if np.all(np.isfinite(selected))
            else 1.0
        )
    energy_scale = (
        max(fitted_energy_scale, policy.normalization_floor)
        if policy.energy_scale is None
        else policy.energy_scale
    )
    force_scale = (
        max(fitted_force_scale, policy.normalization_floor)
        if policy.force_scale is None
        else policy.force_scale
    )
    normalization_id = canonical_fingerprint(
        {
            "kind": "atomistic-training-normalization",
            "problem": problem.problem_id,
            "energy_per_atom_mean": energy_mean,
            "energy_per_atom_scale": energy_scale,
            "force_component_scale": force_scale,
        }
    )
    return AtomisticTrainingNormalization(
        energy_per_atom_mean=jnp.asarray(
            energy_mean, dtype=problem.training_batch.positions.dtype
        ),
        energy_per_atom_scale=jnp.asarray(
            energy_scale, dtype=problem.training_batch.positions.dtype
        ),
        force_component_scale=jnp.asarray(
            force_scale, dtype=problem.training_batch.positions.dtype
        ),
        fitted_from_problem_id=problem.problem_id,
        normalization_id=normalization_id,
    )


def _loss(
    potential: PaiNNPotential,
    batch: AtomisticBatch,
    energy_target: Array | None,
    force_target: Array | None,
    energy_mask: Array | None,
    force_mask: Array | None,
    normalization: AtomisticTrainingNormalization,
    policy: AtomisticTrainingPolicy,
    /,
) -> tuple[Array, tuple[Array, Array]]:
    need_forces = force_target is not None and policy.force_weight > 0.0
    if need_forces:
        def energy_closure(position: Array) -> tuple[Array, tuple[Array, Array]]:
            energy, _, graph = potential._energy_unchecked(batch, position)
            return jnp.sum(energy), (energy, graph.overflow)

        (_, auxiliary), position_gradient = jax.value_and_grad(
            energy_closure, has_aux=True
        )(batch.positions)
        predicted_energy, overflow = auxiliary
        predicted_forces = -position_gradient
    else:
        predicted_energy, _, graph = potential._energy_unchecked(
            batch, batch.positions
        )
        overflow = graph.overflow
        predicted_forces = None
    zero = jnp.asarray(0.0, dtype=predicted_energy.dtype)
    energy_loss = zero
    if energy_target is not None and policy.energy_weight > 0.0:
        count = batch.atom_counts.astype(predicted_energy.dtype)
        residual = (predicted_energy - energy_target) / count
        residual = residual / normalization.energy_per_atom_scale
        mask = jnp.asarray(energy_mask, dtype=bool)
        energy_loss = jnp.sum(jnp.where(mask, residual * residual, 0.0)) / jnp.sum(
            mask
        )
    force_loss = zero
    if force_target is not None and policy.force_weight > 0.0:
        if predicted_forces is None:
            raise RuntimeError("Force loss requested without a conservative force evaluation.")
        residual = (predicted_forces - force_target) / normalization.force_component_scale
        mask = jnp.asarray(force_mask, dtype=bool)
        force_loss = jnp.sum(jnp.where(mask, residual * residual, 0.0)) / jnp.sum(
            mask
        )
    total = policy.energy_weight * energy_loss + policy.force_weight * force_loss
    total = jnp.where(jnp.any(overflow), jnp.asarray(jnp.nan, total.dtype), total)
    return total, (energy_loss, force_loss)


def _training_loss(
    potential: PaiNNPotential,
    problem: AtomisticTrainingProblem,
    normalization: AtomisticTrainingNormalization,
    policy: AtomisticTrainingPolicy,
    /,
) -> tuple[Array, tuple[Array, Array]]:
    return _loss(
        potential,
        problem.training_batch,
        problem.training_energy,
        problem.training_forces,
        problem.training_energy_mask,
        problem.training_force_mask,
        normalization,
        policy,
    )


def _validation_loss(
    potential: PaiNNPotential,
    problem: AtomisticTrainingProblem,
    normalization: AtomisticTrainingNormalization,
    policy: AtomisticTrainingPolicy,
    /,
) -> Array:
    if problem.validation_batch is None:
        return _training_loss(potential, problem, normalization, policy)[0]
    return _loss(
        potential,
        problem.validation_batch,
        problem.validation_energy,
        problem.validation_forces,
        problem.validation_energy_mask,
        problem.validation_force_mask,
        normalization,
        policy,
    )[0]


def _tree_finite(tree: Any, /) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return all(
        bool(np.asarray(jnp.all(jnp.isfinite(leaf))))
        for leaf in leaves
        if eqx.is_inexact_array(leaf)
    )


def fit_atomistic_potential(
    potential: PaiNNPotential,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    callbacks: Sequence[TrainingCallback] = (),
    continuation: AtomisticTrainingResult | None = None,
) -> AtomisticTrainingResult:
    """Fit one PaiNN energy potential with typed energy/force supervision."""

    if not isinstance(potential, PaiNNPotential):
        raise TypeError("potential must be a PaiNNPotential.")
    if not isinstance(problem, AtomisticTrainingProblem):
        raise TypeError("problem must be an AtomisticTrainingProblem.")
    if not isinstance(policy, AtomisticTrainingPolicy):
        raise TypeError("policy must be an AtomisticTrainingPolicy.")
    if potential.scale.scale_id != problem.training_batch.scale.scale_id:
        raise ValueError("Potential and training problem must share one scale contract.")
    if problem.training_energy is None and policy.force_weight <= 0.0:
        raise ValueError("The policy disables the only available force supervision.")
    if problem.training_forces is None and policy.energy_weight <= 0.0:
        raise ValueError("The policy disables the only available energy supervision.")
    potential._validate_batch(problem.training_batch)
    if problem.validation_batch is not None:
        potential._validate_batch(problem.validation_batch)
    normalization = _normalization(problem, policy)
    optimizer = optax.adam(policy.learning_rate)
    if continuation is None:
        current = potential
        trainable, _ = partition_trainable(current)
        optimizer_state = optimizer.init(trainable)
        progress = TrainingProgress()
        master_key = jnp.asarray(key)
        training_history: list[float] = []
        energy_history: list[float] = []
        force_history: list[float] = []
        validation_history: list[float] = []
        validation_steps: list[int] = []
    else:
        if not isinstance(continuation, AtomisticTrainingResult):
            raise TypeError("continuation must be an AtomisticTrainingResult or None.")
        if continuation.problem_id != problem.problem_id:
            raise ValueError("Continuation result belongs to a different training problem.")
        if continuation.continuation_id != policy.continuation_id:
            raise ValueError("Continuation policy changed optimizer, loss, or selection semantics.")
        if continuation.normalization.normalization_id != normalization.normalization_id:
            raise ValueError("Continuation normalization no longer matches the training split.")
        if continuation.progress.update_step > policy.maximum_steps:
            raise ValueError("Continuation step exceeds the requested training ceiling.")
        current = continuation.potential
        optimizer_state = continuation.optimizer_state
        progress = continuation.progress
        master_key = continuation.key
        training_history = np.asarray(continuation.training_loss_history).tolist()
        energy_history = np.asarray(continuation.energy_loss_history).tolist()
        force_history = np.asarray(continuation.force_loss_history).tolist()
        validation_history = np.asarray(continuation.validation_loss_history).tolist()
        validation_steps = np.asarray(continuation.validation_steps).tolist()
    control = TrainingController(
        total_steps=policy.maximum_steps,
        key=master_key,
        progress=progress,
        callbacks=callbacks,
    )
    if continuation is not None:
        control.best_payload = continuation.best_potential
    control.emit("start")
    terminal_status = AtomisticStatus.SUCCESS
    termination = "maximum_steps"
    if control.stop_requested:
        terminal_status = AtomisticStatus.STOPPED_EARLY
        termination = "callback_stop_before_first_update"
    for step in range(progress.update_step + 1, policy.maximum_steps + 1):
        if control.stop_requested:
            break
        trainable, fixed = partition_trainable(current)

        def parameter_loss(parameters: Any) -> tuple[Array, tuple[Array, Array]]:
            candidate = combine_trainable(parameters, fixed)
            return _training_loss(candidate, problem, normalization, policy)

        (loss_value, _), gradients = jax.value_and_grad(
            parameter_loss, has_aux=True
        )(trainable)
        if not bool(np.asarray(jnp.isfinite(loss_value))) or not _tree_finite(gradients):
            training_history.append(float("nan"))
            energy_history.append(float("nan"))
            force_history.append(float("nan"))
            terminal_status = AtomisticStatus.NONFINITE
            termination = "nonfinite_training_loss_or_gradient"
            break
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, params=trainable
        )
        trainable = optax.apply_updates(trainable, updates)
        current = combine_trainable(trainable, fixed)
        post_loss, post_components = _training_loss(
            current, problem, normalization, policy
        )
        post_energy, post_force = post_components
        training_history.append(float(np.asarray(post_loss)))
        energy_history.append(float(np.asarray(post_energy)))
        force_history.append(float(np.asarray(post_force)))
        control.complete_update(step)
        control.emit(
            "update",
            metrics={
                "loss": post_loss,
                "energy_loss": post_energy,
                "force_loss": post_force,
            },
        )
        if not bool(np.asarray(jnp.isfinite(post_loss))):
            terminal_status = AtomisticStatus.NONFINITE
            termination = "nonfinite_updated_loss"
            break
        validate = step % policy.validation_every == 0 or step == policy.maximum_steps
        if validate:
            selected_loss = _validation_loss(
                current, problem, normalization, policy
            )
            selected_value = float(np.asarray(selected_loss))
            validation_history.append(selected_value)
            validation_steps.append(step)
            if not math.isfinite(selected_value):
                terminal_status = AtomisticStatus.NONFINITE
                termination = "nonfinite_validation_loss"
                break
            control.select(
                selected_value,
                current,
                step=step,
                min_delta=policy.min_delta,
                patience=policy.patience,
            )
            control.emit("validation", metrics={"loss": selected_loss})
        if control.stop_requested:
            terminal_status = AtomisticStatus.STOPPED_EARLY
            termination = "selection_or_callback_stop"
            break
    if not validation_history:
        selected_loss = _validation_loss(current, problem, normalization, policy)
        selected_value = float(np.asarray(selected_loss))
        validation_history.append(selected_value)
        validation_steps.append(control.progress.update_step)
        if math.isfinite(selected_value):
            control.select(selected_value, current, step=control.progress.update_step)
        else:
            terminal_status = AtomisticStatus.NONFINITE
            termination = "nonfinite_initial_loss"
    final_loss = (
        training_history[-1]
        if training_history
        else validation_history[-1]
    )
    best_loss = (
        float(control.progress.best_value)
        if control.progress.best_value is not None
        else float("nan")
    )
    best_potential = control.selected(current) if policy.select_best else current
    control.emit("stop", metrics={"final_loss": final_loss, "best_loss": best_loss})
    result_id = canonical_fingerprint(
        {
            "kind": "atomistic-training-result",
            "problem": problem.problem_id,
            "policy": policy.policy_id,
            "potential": current.potential_id,
            "normalization": normalization.normalization_id,
            "updates": control.progress.update_step,
            "status": int(terminal_status),
            "training_history": training_history,
            "validation_history": validation_history,
        }
    )
    dtype = problem.training_batch.positions.dtype
    return AtomisticTrainingResult(
        potential=current,
        best_potential=best_potential,
        optimizer_state=optimizer_state,
        key=master_key,
        normalization=normalization,
        training_loss_history=jnp.asarray(training_history, dtype=dtype),
        energy_loss_history=jnp.asarray(energy_history, dtype=dtype),
        force_loss_history=jnp.asarray(force_history, dtype=dtype),
        validation_loss_history=jnp.asarray(validation_history, dtype=dtype),
        validation_steps=jnp.asarray(validation_steps, dtype=jnp.int32),
        final_loss=jnp.asarray(final_loss, dtype=dtype),
        best_loss=jnp.asarray(best_loss, dtype=dtype),
        status=jnp.asarray(int(terminal_status), dtype=jnp.int32),
        progress=control.progress,
        termination=termination,
        problem_id=problem.problem_id,
        policy_id=policy.policy_id,
        continuation_id=policy.continuation_id,
        result_id=result_id,
    )


__all__ = [
    "AtomisticTrainingNormalization",
    "AtomisticTrainingPolicy",
    "AtomisticTrainingProblem",
    "AtomisticTrainingResult",
    "fit_atomistic_potential",
]
