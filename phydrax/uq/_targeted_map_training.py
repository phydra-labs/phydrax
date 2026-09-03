#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._training import TrainingController, TrainingProgress
from ._posterior import AbstractBijector
from ._targeted_free_energy import (
    _evaluate_forward,
    _evaluate_reverse,
    TargetedFreeEnergyProblem,
    TargetedMapPlan,
)


class TargetedMapTrainingPolicy(StrictModule, NonTrainableState):
    maximum_steps: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    forward_weight: float = eqx.field(static=True)
    reverse_weight: float = eqx.field(static=True)
    displacement_weight: float = eqx.field(static=True)
    validation_interval: int = eqx.field(static=True)
    patience: int | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int = 1000,
        learning_rate: float = 1.0e-3,
        forward_weight: float = 1.0,
        reverse_weight: float = 1.0,
        displacement_weight: float = 0.0,
        validation_interval: int = 10,
        patience: int | None = None,
    ):
        steps = int(maximum_steps)
        interval = int(validation_interval)
        rate = float(learning_rate)
        weights = float(forward_weight), float(reverse_weight), float(displacement_weight)
        patience_ = None if patience is None else int(patience)
        if steps < 0 or interval <= 0 or not isfinite(rate) or rate <= 0.0:
            raise ValueError("Targeted-map training controls are invalid.")
        if (
            any(not isfinite(value) or value < 0.0 for value in weights)
            or sum(weights[:2]) <= 0.0
        ):
            raise ValueError("Targeted-map objective weights are invalid.")
        if patience_ is not None and patience_ <= 0:
            raise ValueError("patience must be positive when provided.")
        self.maximum_steps = steps
        self.learning_rate = rate
        self.forward_weight, self.reverse_weight, self.displacement_weight = weights
        self.validation_interval = interval
        self.patience = patience_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "targeted-map-training-policy",
                "maximum_steps": steps,
                "learning_rate": rate.hex(),
                "forward_weight": weights[0].hex(),
                "reverse_weight": weights[1].hex(),
                "displacement_weight": weights[2].hex(),
                "validation_interval": interval,
                "patience": patience_,
            }
        )


class TargetedMapFitResult(StrictModule):
    mapping: TargetedMapPlan
    training_loss: Array
    validation_loss: Array
    forward_effective_samples: Array
    reverse_effective_samples: Array
    progress: TrainingProgress
    valid: Array
    policy_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _objective(
    bijector: AbstractBijector,
    problem: TargetedFreeEnergyProblem,
    source_samples: Array,
    target_samples: Array | None,
    policy: TargetedMapTrainingPolicy,
    /,
) -> tuple[Array, Array, Array, Array]:
    mapping = eqx.tree_at(lambda value: value.bijector, problem.mapping, bijector)
    current = eqx.tree_at(lambda value: value.mapping, problem, mapping)
    mapped, forward_work, _, forward_residual, forward_valid = jax.vmap(
        lambda value: _evaluate_forward(current, value)
    )(source_samples)
    forward_mass = jnp.maximum(jnp.sum(forward_valid), 1)
    forward_mean = jnp.sum(jnp.where(forward_valid, forward_work, 0.0)) / forward_mass
    displacement = (
        jnp.sum(
            jnp.where(
                forward_valid.reshape((-1,) + (1,) * len(problem.mapping.event_shape)),
                (mapped - source_samples) ** 2,
                0.0,
            )
        )
        / forward_mass
    )
    loss = (
        policy.forward_weight * forward_mean + policy.displacement_weight * displacement
    )
    reverse_mean = jnp.asarray(0.0, dtype=forward_mean.dtype)
    reverse_valid = jnp.asarray(True)
    if target_samples is not None and policy.reverse_weight > 0.0:
        _, reverse_work, _, reverse_residual, reverse_support = jax.vmap(
            lambda value: _evaluate_reverse(current, value)
        )(target_samples)
        reverse_mass = jnp.maximum(jnp.sum(reverse_support), 1)
        reverse_mean = (
            jnp.sum(jnp.where(reverse_support, reverse_work, 0.0)) / reverse_mass
        )
        loss = loss + policy.reverse_weight * reverse_mean
        reverse_valid = jnp.all(reverse_support) & jnp.all(jnp.isfinite(reverse_residual))
    valid = (
        jnp.all(forward_valid)
        & reverse_valid
        & jnp.all(jnp.isfinite(forward_residual))
        & jnp.isfinite(loss)
    )
    return loss, valid, forward_mean, reverse_mean


def _effective_samples(work: Array, /) -> Array:
    shifted = -work - jnp.max(-work)
    weights = jnp.exp(shifted)
    return jnp.sum(weights) ** 2 / jnp.maximum(jnp.sum(weights * weights), 1.0e-30)


def fit_targeted_free_energy_map(
    problem: TargetedFreeEnergyProblem,
    source_samples: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    target_samples: ArrayLike | None = None,
    validation_source: ArrayLike | None = None,
    validation_target: ArrayLike | None = None,
    policy: TargetedMapTrainingPolicy | None = None,
    optimizer: optax.GradientTransformation | None = None,
) -> TargetedMapFitResult:
    if not isinstance(problem, TargetedFreeEnergyProblem):
        raise TypeError("problem must be TargetedFreeEnergyProblem.")
    policy_ = TargetedMapTrainingPolicy() if policy is None else policy
    if not isinstance(policy_, TargetedMapTrainingPolicy):
        raise TypeError("policy must be TargetedMapTrainingPolicy or None.")
    source = jnp.asarray(source_samples)
    target = None if target_samples is None else jnp.asarray(target_samples)
    validation_source_ = (
        source if validation_source is None else jnp.asarray(validation_source)
    )
    validation_target_ = (
        target if validation_target is None else jnp.asarray(validation_target)
    )
    expected = problem.mapping.event_shape
    for name, values in (
        ("source_samples", source),
        ("validation_source", validation_source_),
    ):
        if values.ndim < 1 or tuple(values.shape[1:]) != expected:
            raise ValueError(f"{name} must have shape (sample,) + event_shape.")
    for name, values in (
        ("target_samples", target),
        ("validation_target", validation_target_),
    ):
        if values is not None and (
            values.ndim < 1 or tuple(values.shape[1:]) != expected
        ):
            raise ValueError(f"{name} must have shape (sample,) + event_shape.")
    if policy_.reverse_weight > 0.0 and target is None:
        raise ValueError("Positive reverse_weight requires target_samples.")
    current = problem.mapping.bijector
    optimizer_ = optax.adam(policy_.learning_rate) if optimizer is None else optimizer
    state = optimizer_.init(eqx.filter(current, eqx.is_inexact_array))
    controller = TrainingController(total_steps=policy_.maximum_steps, key=key)

    @eqx.filter_jit
    def update(bijector, optimizer_state):
        def loss_fn(candidate):
            loss, valid, _, _ = _objective(candidate, problem, source, target, policy_)
            return loss, valid

        (loss, valid), gradient = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            bijector
        )
        updates, next_state = optimizer_.update(gradient, optimizer_state, bijector)
        return eqx.apply_updates(bijector, updates), next_state, loss, valid

    initial_loss, valid, _, _ = _objective(
        current, problem, validation_source_, validation_target_, policy_
    )
    controller.select(
        float(initial_loss), current, step=0, mode="min", patience=policy_.patience
    )
    training_history: list[Array] = []
    validation_history: list[Array] = []
    for step in range(1, policy_.maximum_steps + 1):
        current, state, training_loss, step_valid = update(current, state)
        controller.complete_update(step)
        if step % policy_.validation_interval == 0 or step == policy_.maximum_steps:
            validation_loss, validation_valid, _, _ = _objective(
                current, problem, validation_source_, validation_target_, policy_
            )
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
    mapping = problem.mapping.with_bijector(selected)
    selected_problem = TargetedFreeEnergyProblem(problem.source, problem.target, mapping)
    from ._targeted_free_energy import evaluate_targeted_work

    evaluation = evaluate_targeted_work(
        selected_problem,
        validation_source_,
        target_samples=validation_target_,
    )
    forward_ess = _effective_samples(evaluation.forward_work)
    reverse_ess = (
        jnp.asarray(jnp.nan)
        if evaluation.reverse_work is None
        else _effective_samples(evaluation.reverse_work)
    )
    training_values = jnp.asarray(training_history)
    validation_values = jnp.asarray(validation_history)
    valid = (
        evaluation.valid
        & valid
        & jnp.all(jnp.isfinite(training_values))
        & jnp.all(jnp.isfinite(validation_values))
    )
    result_id = canonical_fingerprint(
        {
            "kind": "targeted-map-fit",
            "problem": problem.problem_id,
            "mapping": mapping.map_id,
            "policy": policy_.policy_id,
        }
    )
    return TargetedMapFitResult(
        mapping=mapping,
        training_loss=training_values,
        validation_loss=validation_values,
        forward_effective_samples=forward_ess,
        reverse_effective_samples=reverse_ess,
        progress=controller.progress,
        valid=valid,
        policy_id=policy_.policy_id,
        result_id=result_id,
    )


__all__ = [
    "TargetedMapFitResult",
    "TargetedMapTrainingPolicy",
    "fit_targeted_free_energy_map",
]
