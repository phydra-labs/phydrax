#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array

from ..._fingerprint import (
    array_tree_fingerprint,
    array_tree_signature,
    canonical_fingerprint,
)
from ..._frozendict import frozendict
from ..._model import AbstractArrayModel
from ..._trainable import combine_trainable, partition_trainable
from ..._training import (
    DelayedTargetPolicy,
    EvaluationParametersFn,
    ExponentialMovingAverageTargetPolicy,
    resolve_evaluation_parameters,
    TargetParameterState,
    TensorBoardLogger,
    TrainingCallback,
    TrainingController,
    TrainingProgress,
    TrainingSignalGuard,
)
from ..._training_objective import (
    _GradientAccumulationState,
    _ObjectiveAccumulator,
    _ObjectiveContribution,
)
from ...metrix import EuclideanStateGeometry
from .._layout import InputLayout, StateLayout
from .._model_system import DiscreteModelTransition
from .._system import DiscreteStepContext, DiscreteSystem
from .._trajectory import TrajectoryData
from ._neural_checkpoint import (
    _load_neural_training_checkpoint,
    _read_neural_training_manifest,
    _save_neural_training_checkpoint,
)
from ._neural_windows import (
    _active_window_evidence,
    _KEY_POLICY_ID,
    _NeuralWindowBatch,
    _NeuralWindowSource,
    _semantic_window_keys,
)


@dataclass(frozen=True, slots=True)
class DiscreteModelRolloutPolicy:
    """Static rollout capacity, traced curriculum, BPTT, and rematerialization."""

    max_horizon: int
    min_horizon: int | None = None
    transition_steps: int = 0
    schedule: Literal["constant", "linear"] = "constant"
    truncate_every: int | None = None
    rematerialize: bool = False

    def __post_init__(self) -> None:
        maximum = int(self.max_horizon)
        minimum = maximum if self.min_horizon is None else int(self.min_horizon)
        if maximum < 1 or minimum < 1 or minimum > maximum:
            raise ValueError(
                "Rollout horizons must satisfy 1 <= min_horizon <= max_horizon."
            )
        if int(self.transition_steps) < 0:
            raise ValueError("transition_steps must be nonnegative.")
        if self.schedule not in ("constant", "linear"):
            raise ValueError("schedule must be 'constant' or 'linear'.")
        if self.schedule == "constant" and minimum != maximum:
            raise ValueError("A constant schedule requires equal rollout horizons.")
        if (
            self.schedule == "linear"
            and minimum < maximum
            and int(self.transition_steps) < 1
        ):
            raise ValueError(
                "A nonconstant linear schedule requires transition_steps > 0."
            )
        if self.truncate_every is not None and int(self.truncate_every) < 1:
            raise ValueError("truncate_every must be positive or None.")
        object.__setattr__(self, "max_horizon", maximum)
        object.__setattr__(self, "min_horizon", minimum)
        object.__setattr__(self, "transition_steps", int(self.transition_steps))
        object.__setattr__(
            self,
            "truncate_every",
            None if self.truncate_every is None else int(self.truncate_every),
        )

    def active_horizon(self, step: Array, /) -> Array:
        """Return a scalar traced horizon for one optimizer update index."""

        assert self.min_horizon is not None
        minimum = int(self.min_horizon)
        if self.schedule == "constant" or minimum == self.max_horizon:
            return jnp.asarray(self.max_horizon, dtype=jnp.int32)
        progress = jnp.clip(
            jnp.asarray(step, dtype=jnp.float32) / float(self.transition_steps),
            0.0,
            1.0,
        )
        span = self.max_horizon - minimum
        return minimum + jnp.floor(progress * span).astype(jnp.int32)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                **asdict(self),
                "key_policy_id": _KEY_POLICY_ID,
            }
        )


@dataclass(frozen=True, slots=True)
class SupervisedDiscreteModelObjective:
    """Weighted Euclidean comparison with stored rollout states."""

    name: str = "supervised"
    weight: float = 1.0
    time_weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        _validate_objective_header(self.name, self.weight)
        object.__setattr__(self, "time_weights", _coefficient_tuple(self.time_weights))

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "supervised",
                "name": self.name,
                "weight": float(self.weight),
                "time_weights": self.time_weights,
            }
        )


@dataclass(frozen=True, slots=True)
class TargetDiscreteModelObjective:
    """Rollout consistency against the stopped delayed/EMA target model."""

    name: str = "target_consistency"
    weight: float = 1.0
    time_weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        _validate_objective_header(self.name, self.weight)
        object.__setattr__(self, "time_weights", _coefficient_tuple(self.time_weights))

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "target_consistency",
                "name": self.name,
                "weight": float(self.weight),
                "time_weights": self.time_weights,
            }
        )


@dataclass(frozen=True, slots=True)
class ReferenceBranchDiscreteModelObjective:
    """Diverted branches from the learned chain through a deterministic system."""

    reference: DiscreteSystem
    branch_length: int
    name: str = "reference_branch"
    weight: float = 1.0
    origin_weights: Sequence[float] | None = None
    branch_weights: Sequence[float] | None = None
    reference_gradient: Literal["coupled", "stopped"] = "coupled"

    def __post_init__(self) -> None:
        if not isinstance(self.reference, DiscreteSystem):
            raise TypeError("reference must be a DiscreteSystem.")
        if int(self.branch_length) < 1:
            raise ValueError("branch_length must be positive.")
        if self.reference_gradient not in ("coupled", "stopped"):
            raise ValueError("reference_gradient must be 'coupled' or 'stopped'.")
        _validate_objective_header(self.name, self.weight)
        object.__setattr__(self, "branch_length", int(self.branch_length))
        object.__setattr__(
            self, "origin_weights", _coefficient_tuple(self.origin_weights)
        )
        object.__setattr__(
            self, "branch_weights", _coefficient_tuple(self.branch_weights)
        )

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "reference_branch",
                "name": self.name,
                "weight": float(self.weight),
                "branch_length": self.branch_length,
                "origin_weights": self.origin_weights,
                "branch_weights": self.branch_weights,
                "reference_gradient": self.reference_gradient,
                "reference_system_id": self.reference.system_id,
                "reference_step": {
                    "size": self.reference.step_size,
                    "rtol": self.reference.step_rtol,
                    "atol": self.reference.step_atol,
                },
                "reference_state": array_tree_fingerprint(self.reference),
            }
        )


@dataclass(frozen=True, slots=True)
class ResidualDiscreteModelObjective:
    """Weighted Euclidean residual on each learned transition."""

    residual: Callable[[Array, Array, Array, Array | None], Array]
    residual_id: str
    name: str = "residual"
    weight: float = 1.0
    time_weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if not callable(self.residual):
            raise TypeError("residual must be callable.")
        if not isinstance(self.residual_id, str) or not self.residual_id:
            raise ValueError("residual_id must be a nonempty string.")
        _validate_objective_header(self.name, self.weight)
        object.__setattr__(self, "time_weights", _coefficient_tuple(self.time_weights))

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "residual",
                "name": self.name,
                "weight": float(self.weight),
                "time_weights": self.time_weights,
                "residual_id": self.residual_id,
            }
        )


DiscreteModelObjective = (
    SupervisedDiscreteModelObjective
    | TargetDiscreteModelObjective
    | ReferenceBranchDiscreteModelObjective
    | ResidualDiscreteModelObjective
)


@dataclass(frozen=True, slots=True)
class DiscreteModelValidationPolicy:
    """Validation cadence, early stopping, and selected-model semantics."""

    every: int = 1
    monitor: str = "loss"
    mode: Literal["min", "max"] = "min"
    patience: int | None = None
    minimum_delta: float = 0.0
    relative_minimum_delta: float = 0.0
    select_best: bool = True

    def __post_init__(self) -> None:
        if int(self.every) < 1:
            raise ValueError("Validation cadence must be positive.")
        if not isinstance(self.monitor, str) or not self.monitor:
            raise ValueError("Validation monitor must be nonempty.")
        if self.mode not in ("min", "max"):
            raise ValueError("Validation mode must be 'min' or 'max'.")
        if self.patience is not None and int(self.patience) < 1:
            raise ValueError("Validation patience must be positive or None.")
        values = (float(self.minimum_delta), float(self.relative_minimum_delta))
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Validation improvement deltas must be finite and nonnegative."
            )


@dataclass(frozen=True, slots=True)
class DiscreteModelFitHistory:
    """Immutable learning curves from one discrete-model fit."""

    initial_metrics: frozendict[str, float]
    train_steps: tuple[int, ...]
    train_metrics: tuple[frozendict[str, float], ...]
    validation_steps: tuple[int, ...]
    validation_metrics: tuple[frozendict[str, float], ...]
    final_metrics: frozendict[str, float]

    @property
    def losses(self) -> tuple[float, ...]:
        return tuple(metrics["loss"] for metrics in self.train_metrics)

    @property
    def validation_losses(self) -> tuple[float, ...]:
        return tuple(metrics["loss"] for metrics in self.validation_metrics)


@dataclass(frozen=True, slots=True)
class DiscreteModelFitResult:
    """Selected model, bound system, lifecycle state, and learning curves."""

    model: AbstractArrayModel
    last_model: AbstractArrayModel
    system: DiscreteSystem
    history: DiscreteModelFitHistory
    progress: TrainingProgress
    resumed_from_step: int
    training_seconds: float
    checkpoint_path: Path | None
    stopped_by_signal: bool = False
    stopped_by_callback: bool = False

    @property
    def initial_loss(self) -> float:
        return self.history.initial_metrics["loss"]

    @property
    def final_loss(self) -> float:
        return self.history.final_metrics["loss"]

    @property
    def completed_steps(self) -> int:
        return self.progress.update_step


class _RolloutCarry(NamedTuple):
    state: Array
    runtime_valid: Array


def _validate_objective_header(name: str, weight: float, /) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Objective names must be nonempty strings.")
    resolved = float(weight)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError("Objective weights must be finite and nonnegative.")


def _coefficient_tuple(values: Sequence[float] | None, /) -> tuple[float, ...] | None:
    if values is None:
        return None
    coefficients = tuple(float(value) for value in values)
    if not coefficients or any(
        not np.isfinite(value) or value < 0.0 for value in coefficients
    ):
        raise ValueError(
            "Objective coefficients must be finite, nonnegative, and nonempty."
        )
    if not any(value > 0.0 for value in coefficients):
        raise ValueError("Objective coefficients must contain positive support.")
    return coefficients


def _normalized_coefficients(
    values: Sequence[float] | None,
    length: int,
    active: Array,
    /,
) -> Array:
    coefficients = (
        jnp.ones((length,), dtype=jnp.float32) if values is None else jnp.asarray(values)
    )
    if coefficients.shape != (length,):
        raise ValueError(f"Objective coefficients must have length {length}.")
    mask = jnp.arange(length, dtype=jnp.int32) < active
    masked = jnp.where(mask, coefficients, jnp.zeros_like(coefficients))
    denominator = jnp.sum(masked)
    safe = jnp.where(denominator > 0.0, denominator, jnp.ones_like(denominator))
    return jnp.where(denominator > 0.0, masked / safe, jnp.zeros_like(masked))


def _event_mask(mask: Array, event_rank: int, /) -> Array:
    result = mask
    for _ in range(event_rank):
        result = result[..., None]
    return result


def _mean_square(values: Array, event_rank: int, /) -> Array:
    squared = jnp.square(values)
    if event_rank:
        axes = tuple(range(squared.ndim - event_rank, squared.ndim))
        return jnp.mean(squared, axis=axes)
    return squared


def _model_point(
    model: AbstractArrayModel,
    state: Array,
    inputs: Array | None,
    key: Array,
    iteration: Array,
    /,
) -> Array:
    binding = model.input_binding()
    point = (
        binding.pack_point((state,))
        if inputs is None
        else binding.pack_point((state, inputs))
    )
    return jnp.asarray(
        binding.call(model, point, key=key, iter_=iteration, kwargs={}),
        dtype=state.dtype,
    )


def _rollout_scan_step(
    carry: _RolloutCarry,
    depth: Array,
    /,
    *,
    model: AbstractArrayModel,
    batch: _NeuralWindowBatch,
    eligible: Array,
    active_horizon: Array,
    state_layout: StateLayout,
    truncate_every: int | None,
    root_key: Array,
    iteration: Array,
) -> tuple[_RolloutCarry, tuple[Array, Array]]:
    active = depth < active_horizon
    run = eligible & active
    keys = _semantic_window_keys(
        root_key,
        batch.parent_index,
        batch.start_index,
        depth,
        0,
    )
    controls = None if batch.inputs is None else batch.inputs[:, depth]

    if controls is None:

        def one(state, key, enabled):
            return jax.lax.cond(
                enabled,
                lambda _: _model_point(model, state, None, key, iteration),
                lambda _: state,
                operand=None,
            )

        candidate = jax.vmap(one)(carry.state, keys, run)
    else:

        def one(state, inputs, key, enabled):
            return jax.lax.cond(
                enabled,
                lambda _: _model_point(model, state, inputs, key, iteration),
                lambda _: state,
                operand=None,
            )

        candidate = jax.vmap(one)(carry.state, controls, keys, run)
    finite = jnp.all(
        jnp.isfinite(candidate),
        axis=tuple(range(1, candidate.ndim)),
    )
    member = jax.vmap(state_layout.geometry.contains)(candidate)
    step_valid = ~run | (finite & member)
    safe_candidate = jnp.where(
        _event_mask(step_valid, len(state_layout.shape)),
        candidate,
        jnp.zeros_like(candidate),
    )
    next_state = jnp.where(
        _event_mask(run, len(state_layout.shape)),
        safe_candidate,
        carry.state,
    )
    output = next_state
    if truncate_every is not None:
        cut = active & (((depth + 1) % int(truncate_every)) == 0)
        next_state = jnp.where(cut, jax.lax.stop_gradient(next_state), next_state)
    return (
        _RolloutCarry(next_state, carry.runtime_valid & step_valid),
        (output, next_state),
    )


def _rollout_states(
    model: AbstractArrayModel,
    batch: _NeuralWindowBatch,
    active_horizon: Array,
    policy: DiscreteModelRolloutPolicy,
    state_layout: StateLayout,
    root_key: Array,
    iteration: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    eligible, evidence = _active_window_evidence(batch, active_horizon)

    def step(carry, depth):
        return _rollout_scan_step(
            carry,
            depth,
            model=model,
            batch=batch,
            eligible=eligible,
            active_horizon=active_horizon,
            state_layout=state_layout,
            truncate_every=policy.truncate_every,
            root_key=root_key,
            iteration=iteration,
        )

    scan_step = jax.checkpoint(step) if policy.rematerialize else step
    initial = _RolloutCarry(batch.states[:, 0], jnp.ones_like(eligible))
    final, (raw_predictions, recurrent_predictions) = jax.lax.scan(
        scan_step,
        initial,
        jnp.arange(policy.max_horizon, dtype=jnp.int32),
    )
    endpoint_states = jnp.concatenate(
        (batch.states[:, :1], jnp.swapaxes(raw_predictions, 0, 1)),
        axis=1,
    )
    origin_states = jnp.concatenate(
        (batch.states[:, :1], jnp.swapaxes(recurrent_predictions, 0, 1)),
        axis=1,
    )
    return endpoint_states, origin_states, evidence, jnp.all(final.runtime_valid)


def _supervised_window_values(
    objective: SupervisedDiscreteModelObjective,
    states: Array,
    batch: _NeuralWindowBatch,
    active_horizon: Array,
    state_rank: int,
    /,
) -> tuple[Array, Array]:
    coefficients = _normalized_coefficients(
        objective.time_weights,
        batch.max_horizon,
        active_horizon,
    )
    residual = states[:, 1:] - batch.states[:, 1:]
    values = _mean_square(residual, state_rank)
    return jnp.sum(values * coefficients[None, :], axis=1), jnp.asarray(True)


def _target_window_values(
    objective: TargetDiscreteModelObjective,
    states: Array,
    target_states: Array,
    batch: _NeuralWindowBatch,
    active_horizon: Array,
    state_rank: int,
    /,
) -> tuple[Array, Array]:
    coefficients = _normalized_coefficients(
        objective.time_weights,
        batch.max_horizon,
        active_horizon,
    )
    residual = states[:, 1:] - jax.lax.stop_gradient(target_states[:, 1:])
    values = _mean_square(residual, state_rank)
    return jnp.sum(values * coefficients[None, :], axis=1), jnp.asarray(True)


def _residual_window_values(
    objective: ResidualDiscreteModelObjective,
    endpoint_states: Array,
    origin_states: Array,
    batch: _NeuralWindowBatch,
    active_horizon: Array,
    state_layout: StateLayout,
    /,
) -> tuple[Array, Array]:
    coefficients = _normalized_coefficients(
        objective.time_weights,
        batch.max_horizon,
        active_horizon,
    )
    eligible, _ = _active_window_evidence(batch, active_horizon)

    def at_depth(depth):
        previous = origin_states[:, depth]
        following = endpoint_states[:, depth + 1]
        coordinate = batch.coordinates[:, depth]
        controls = None if batch.inputs is None else batch.inputs[:, depth]
        enabled = eligible & (depth < active_horizon)
        if controls is None:

            def one(next_state, prior_state, source, active):
                return jax.lax.cond(
                    active,
                    lambda _: objective.residual(
                        next_state,
                        prior_state,
                        source,
                        None,
                    ),
                    lambda _: jnp.zeros_like(next_state),
                    operand=None,
                )

            values = jax.vmap(one)(
                following,
                previous,
                coordinate,
                enabled,
            )
        else:

            def one(next_state, prior_state, source, inputs, active):
                return jax.lax.cond(
                    active,
                    lambda _: objective.residual(
                        next_state,
                        prior_state,
                        source,
                        inputs,
                    ),
                    lambda _: jnp.zeros_like(next_state),
                    operand=None,
                )

            values = jax.vmap(one)(
                following,
                previous,
                coordinate,
                controls,
                enabled,
            )
        values = jnp.asarray(values)
        if values.shape != following.shape:
            raise ValueError("Residual output must have the state layout shape.")
        finite = jnp.all(
            jnp.isfinite(values),
            axis=tuple(range(1, values.ndim)),
        )
        safe = jnp.where(
            _event_mask(finite, len(state_layout.shape)),
            values,
            jnp.zeros_like(values),
        )
        return _mean_square(safe, len(state_layout.shape)), finite

    values, finite = jax.vmap(at_depth)(jnp.arange(batch.max_horizon, dtype=jnp.int32))
    values = jnp.swapaxes(values, 0, 1)
    finite = jnp.swapaxes(finite, 0, 1)
    active = eligible[:, None] & (jnp.arange(batch.max_horizon)[None, :] < active_horizon)
    runtime_valid = jnp.all(~active | finite)
    return jnp.sum(values * coefficients[None, :], axis=1), runtime_valid


def _reference_window_values(
    objective: ReferenceBranchDiscreteModelObjective,
    endpoint_states: Array,
    origin_states: Array,
    batch: _NeuralWindowBatch,
    active_horizon: Array,
    state_layout: StateLayout,
    /,
) -> tuple[Array, Array]:
    horizon = batch.max_horizon
    active_origins = jnp.maximum(
        active_horizon - int(objective.branch_length) + 1,
        0,
    )
    origin_coefficients = _normalized_coefficients(
        objective.origin_weights,
        horizon,
        active_origins,
    )
    branch_coefficients = _normalized_coefficients(
        objective.branch_weights,
        objective.branch_length,
        jnp.asarray(objective.branch_length, dtype=jnp.int32),
    )
    total = jnp.zeros((batch.size,), dtype=endpoint_states.dtype)
    coefficient_total = jnp.asarray(0.0, dtype=endpoint_states.dtype)
    runtime_valid = jnp.asarray(True)
    reference = objective.reference
    assert reference.step_size is not None
    eligible, _ = _active_window_evidence(batch, active_horizon)

    for origin in range(horizon):
        reference_state = origin_states[:, origin]
        if objective.reference_gradient == "stopped":
            reference_state = jax.lax.stop_gradient(reference_state)
        origin_active = origin < active_origins
        for branch in range(objective.branch_length):
            target_index = origin + branch + 1
            in_capacity = target_index <= horizon
            if in_capacity:
                coordinate = batch.coordinates[:, origin + branch]
                controls = (
                    None if batch.inputs is None else batch.inputs[:, origin + branch]
                )

                enabled = eligible & origin_active
                if controls is None:

                    def one(source, state, active):
                        return jax.lax.cond(
                            active,
                            lambda _: reference.evaluate(
                                DiscreteStepContext(
                                    source,
                                    source + reference.step_size,
                                    jnp.asarray(origin + branch, dtype=jnp.int32),
                                ),
                                state,
                                None,
                            ),
                            lambda _: state,
                            operand=None,
                        )

                    candidate = jax.vmap(one)(
                        coordinate,
                        reference_state,
                        enabled,
                    )
                else:

                    def one(source, state, inputs, active):
                        return jax.lax.cond(
                            active,
                            lambda _: reference.evaluate(
                                DiscreteStepContext(
                                    source,
                                    source + reference.step_size,
                                    jnp.asarray(origin + branch, dtype=jnp.int32),
                                ),
                                state,
                                None,
                                inputs=inputs,
                            ),
                            lambda _: state,
                            operand=None,
                        )

                    candidate = jax.vmap(one)(
                        coordinate,
                        reference_state,
                        controls,
                        enabled,
                    )
                finite = jnp.all(
                    jnp.isfinite(candidate),
                    axis=tuple(range(1, candidate.ndim)),
                )
                member = jax.vmap(reference.state_layout.geometry.contains)(candidate)
                valid = finite & member
                reference_state = jnp.where(
                    _event_mask(valid, len(state_layout.shape)),
                    candidate,
                    jnp.zeros_like(candidate),
                )
                runtime_valid = runtime_valid & jnp.all(~enabled | valid)
                difference = endpoint_states[:, target_index] - reference_state
                value = _mean_square(difference, len(state_layout.shape))
                coefficient = origin_coefficients[origin] * branch_coefficients[branch]
                total = total + jnp.where(enabled, coefficient * value, 0.0)
                coefficient_total = coefficient_total + jnp.where(
                    jnp.any(enabled),
                    coefficient,
                    0.0,
                )
    safe = jnp.where(
        coefficient_total > 0.0,
        coefficient_total,
        jnp.ones_like(coefficient_total),
    )
    return jnp.where(coefficient_total > 0.0, total / safe, 0.0), runtime_valid


def _objective_contributions(
    model: AbstractArrayModel,
    batch: _NeuralWindowBatch,
    active_horizon: Array,
    policy: DiscreteModelRolloutPolicy,
    objectives: tuple[DiscreteModelObjective, ...],
    state_layout: StateLayout,
    root_key: Array,
    iteration: Array | None = None,
    /,
    *,
    target_model: AbstractArrayModel | None = None,
) -> tuple[_ObjectiveContribution, tuple[_ObjectiveContribution, ...], Array]:
    resolved_iteration = (
        jnp.asarray(0, dtype=jnp.int32) if iteration is None else jnp.asarray(iteration)
    )
    endpoint_states, origin_states, evidence, rollout_valid = _rollout_states(
        model,
        batch,
        active_horizon,
        policy,
        state_layout,
        root_key,
        resolved_iteration,
    )
    runtime_valid = rollout_valid
    target_endpoint_states = None
    if any(isinstance(value, TargetDiscreteModelObjective) for value in objectives):
        if target_model is None:
            raise ValueError("Target objective requires a target model.")
        target_endpoint_states, _, _, target_valid = _rollout_states(
            target_model,
            batch,
            active_horizon,
            policy,
            state_layout,
            jr.fold_in(root_key, 707),
            resolved_iteration,
        )
        runtime_valid = rollout_valid & target_valid
    term_contributions: list[_ObjectiveContribution] = []
    combined_window = jnp.zeros_like(evidence)
    for objective in objectives:
        if isinstance(objective, TargetDiscreteModelObjective):
            assert target_endpoint_states is not None
            values, valid = _target_window_values(
                objective,
                endpoint_states,
                target_endpoint_states,
                batch,
                active_horizon,
                len(state_layout.shape),
            )
        elif isinstance(objective, SupervisedDiscreteModelObjective):
            values, valid = _supervised_window_values(
                objective,
                endpoint_states,
                batch,
                active_horizon,
                len(state_layout.shape),
            )
        elif isinstance(objective, ResidualDiscreteModelObjective):
            values, valid = _residual_window_values(
                objective,
                endpoint_states,
                origin_states,
                batch,
                active_horizon,
                state_layout,
            )
        else:
            values, valid = _reference_window_values(
                objective,
                endpoint_states,
                origin_states,
                batch,
                active_horizon,
                state_layout,
            )
        runtime_valid = runtime_valid & valid
        numerator = jnp.sum(evidence * values)
        support = jnp.sum(evidence)
        term_contributions.append(_ObjectiveContribution(numerator, support))
        combined_window = combined_window + float(objective.weight) * values
    total = _ObjectiveContribution(
        jnp.sum(evidence * combined_window),
        jnp.sum(evidence),
    )
    return total, tuple(term_contributions), runtime_valid


def _tree_real_result_dtype(tree: Any, /):
    dtypes = tuple(
        leaf.dtype for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_array(leaf)
    )
    if not dtypes:
        return jnp.dtype(jnp.float32)
    return jnp.result_type(*dtypes)


def _tree_finite(tree: Any, /) -> Array:
    checks = [
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves(tree)
        if eqx.is_array(leaf)
    ]
    return jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True)


def _validate_precision(model: AbstractArrayModel, *datasets: TrajectoryData) -> None:
    allowed = (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64))
    parameters, _ = partition_trainable(model)
    for leaf in jax.tree_util.tree_leaves(parameters):
        if eqx.is_array(leaf) and (
            jnp.issubdtype(leaf.dtype, jnp.complexfloating) or leaf.dtype not in allowed
        ):
            raise TypeError(
                "Discrete-model fitting requires real float32/float64 parameters."
            )
    for data in datasets:
        arrays = (data.coordinates, data.states, data.weights, data.inputs)
        for value in arrays:
            if value is not None and (
                jnp.issubdtype(value.dtype, jnp.complexfloating)
                or value.dtype not in allowed
            ):
                raise TypeError(
                    "Discrete-model fitting requires real float32/float64 data."
                )


def _validate_data_contract(
    data: TrajectoryData,
    state_layout: StateLayout,
    input_layout: InputLayout | None,
    /,
) -> None:
    if not isinstance(data, TrajectoryData):
        raise TypeError("Training and validation data must be TrajectoryData.")
    if data.state_layout.layout_id != state_layout.layout_id:
        raise ValueError("Trajectory and fit state layouts must match exactly.")
    if (data.input_layout is None) != (input_layout is None):
        raise ValueError(
            "Trajectory and fit input layouts must both be present or absent."
        )
    if input_layout is not None:
        assert data.input_layout is not None
        if data.input_layout.layout_id != input_layout.layout_id:
            raise ValueError("Trajectory and fit input layouts must match exactly.")


def _validate_reference_contracts(
    objectives: tuple[DiscreteModelObjective, ...],
    rollout_policy: DiscreteModelRolloutPolicy,
    state_layout: StateLayout,
    input_layout: InputLayout | None,
    step_size: float,
    step_rtol: float,
    step_atol: float,
    /,
) -> None:
    for objective in objectives:
        if not isinstance(objective, ReferenceBranchDiscreteModelObjective):
            continue
        assert rollout_policy.min_horizon is not None
        if objective.branch_length > int(rollout_policy.min_horizon):
            raise ValueError(
                "Reference branch length cannot exceed the minimum rollout horizon."
            )
        reference = objective.reference
        if reference.state_layout.layout_id != state_layout.layout_id:
            raise ValueError("Reference and learned state layouts must match exactly.")
        if (reference.input_layout is None) != (input_layout is None):
            raise ValueError("Reference and learned input layouts must match exactly.")
        if input_layout is not None:
            assert reference.input_layout is not None
            if reference.input_layout.layout_id != input_layout.layout_id:
                raise ValueError(
                    "Reference and learned input layouts must match exactly."
                )
        if (
            reference.step_size != step_size
            or reference.step_rtol != step_rtol
            or reference.step_atol != step_atol
        ):
            raise ValueError(
                "Reference and learned fixed-step contracts must be identical."
            )


def fit_discrete_model(
    model: AbstractArrayModel,
    train: TrajectoryData,
    /,
    *,
    validation: TrajectoryData | None = None,
    state_layout: StateLayout,
    input_layout: InputLayout | None = None,
    model_id: str | None = None,
    system_id: str,
    step_size: float,
    step_rtol: float = 1e-7,
    step_atol: float = 1e-12,
    rollout_policy: DiscreteModelRolloutPolicy,
    objectives: Sequence[DiscreteModelObjective] | None = None,
    optimizer: optax.GradientTransformation
    | optax.GradientTransformationExtraArgs
    | None = None,
    optimizer_id: str | None = None,
    evaluation_parameters: EvaluationParametersFn | None = None,
    evaluation_parameters_id: str | None = None,
    target_policy: DelayedTargetPolicy
    | ExponentialMovingAverageTargetPolicy
    | None = None,
    learning_rate: float = 1e-3,
    epochs: int = 1,
    steps: int | None = None,
    batch_size: int | None = None,
    validation_batch_size: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    key: Any | None = None,
    gradient_accumulation: int = 1,
    validation_policy: DiscreteModelValidationPolicy | None = None,
    jit: bool = True,
    callbacks: Sequence[TrainingCallback] = (),
    tensorboard_log_dir: str | Path | None = None,
    tensorboard_every: int = 1,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 1,
    resume: bool = False,
) -> DiscreteModelFitResult:
    """Fit a deterministic pointwise next-state model from fixed-step trajectories."""

    if not isinstance(model, AbstractArrayModel):
        raise TypeError("fit_discrete_model requires an AbstractArrayModel.")
    if not isinstance(rollout_policy, DiscreteModelRolloutPolicy):
        raise TypeError("rollout_policy must be a DiscreteModelRolloutPolicy.")
    if not isinstance(state_layout, StateLayout):
        raise TypeError("state_layout must be a StateLayout.")
    if not isinstance(state_layout.geometry, EuclideanStateGeometry):
        raise TypeError(
            "Discrete-model fitting currently requires EuclideanStateGeometry."
        )
    resolved_model_id = None if model_id is None else str(model_id).strip()
    if checkpoint_path is not None and not resolved_model_id:
        raise ValueError("Checkpointed fits require a stable model_id.")
    if resolved_model_id is None:
        resolved_model_id = f"{type(model).__module__}.{type(model).__qualname__}"
    if input_layout is not None and not isinstance(input_layout, InputLayout):
        raise TypeError("input_layout must be an InputLayout or None.")
    if not isinstance(system_id, str) or not system_id:
        raise ValueError("system_id must be a nonempty string.")
    DiscreteModelTransition(
        model,
        state_layout=state_layout,
        input_layout=input_layout,
        step_size=step_size,
        step_rtol=step_rtol,
        step_atol=step_atol,
    )
    _validate_data_contract(train, state_layout, input_layout)
    if validation is not None:
        _validate_data_contract(validation, state_layout, input_layout)
    _validate_precision(model, train, *((validation,) if validation is not None else ()))
    if int(epochs) < 0 or (steps is not None and int(steps) < 0):
        raise ValueError("epochs and steps must be nonnegative.")
    if int(gradient_accumulation) < 1:
        raise ValueError("gradient_accumulation must be positive.")
    if int(checkpoint_every) < 1 or int(tensorboard_every) < 1:
        raise ValueError("Checkpoint and TensorBoard cadences must be positive.")
    if evaluation_parameters is None:
        if evaluation_parameters_id is not None:
            raise ValueError("evaluation_parameters_id requires evaluation_parameters.")
        resolved_evaluation_id = None
    else:
        if not callable(evaluation_parameters):
            raise TypeError("evaluation_parameters must be callable.")
        resolved_evaluation_id = (
            None
            if evaluation_parameters_id is None
            else str(evaluation_parameters_id).strip()
        )
        if checkpoint_path is not None and not resolved_evaluation_id:
            raise ValueError(
                "Checkpointed fits with evaluation_parameters require a stable identity."
            )
    if optimizer is None:
        rate = float(learning_rate)
        if not np.isfinite(rate) or rate < 0.0:
            raise ValueError("learning_rate must be finite and nonnegative.")
        optimizer = optax.adam(rate)
        resolved_optimizer_id = f"optax.adam:{rate:.17g}"
    else:
        if not isinstance(optimizer_id, str) or not optimizer_id:
            raise ValueError("Custom optimizers require a stable optimizer_id.")
        resolved_optimizer_id = optimizer_id

    terms = (
        (SupervisedDiscreteModelObjective(),) if objectives is None else tuple(objectives)
    )
    if not terms or any(
        not isinstance(
            term,
            (
                SupervisedDiscreteModelObjective,
                TargetDiscreteModelObjective,
                ReferenceBranchDiscreteModelObjective,
                ResidualDiscreteModelObjective,
            ),
        )
        for term in terms
    ):
        raise TypeError("objectives must contain supported discrete-model objectives.")
    if len({term.name for term in terms}) != len(terms):
        raise ValueError("Objective names must be unique.")
    if not any(float(term.weight) > 0.0 for term in terms):
        raise ValueError("At least one objective must have positive weight.")
    if any(isinstance(term, TargetDiscreteModelObjective) for term in terms) and (
        target_policy is None
    ):
        raise ValueError("Target discrete objective requires target_policy.")
    assert rollout_policy.min_horizon is not None
    reachable_horizons = range(
        int(rollout_policy.min_horizon),
        int(rollout_policy.max_horizon) + 1,
    )
    for term in terms:
        if isinstance(
            term,
            (
                SupervisedDiscreteModelObjective,
                TargetDiscreteModelObjective,
                ResidualDiscreteModelObjective,
            ),
        ):
            if (
                term.time_weights is not None
                and len(term.time_weights) != rollout_policy.max_horizon
            ):
                raise ValueError("time_weights must match rollout_policy.max_horizon.")
            if term.time_weights is not None and any(
                sum(term.time_weights[:horizon]) <= 0.0 for horizon in reachable_horizons
            ):
                raise ValueError(
                    "time_weights must have positive mass at every reachable horizon."
                )
        elif (
            term.origin_weights is not None
            and len(term.origin_weights) != rollout_policy.max_horizon
        ):
            raise ValueError("origin_weights must match rollout_policy.max_horizon.")
        if isinstance(term, ReferenceBranchDiscreteModelObjective):
            if term.branch_length > rollout_policy.max_horizon:
                raise ValueError(
                    "branch_length cannot exceed rollout_policy.max_horizon."
                )
            if (
                term.branch_weights is not None
                and len(term.branch_weights) != term.branch_length
            ):
                raise ValueError("branch_weights must match branch_length.")
            if term.branch_weights is not None and sum(term.branch_weights) <= 0.0:
                raise ValueError("branch_weights must have positive mass.")
            if term.origin_weights is not None and any(
                sum(term.origin_weights[: horizon - int(term.branch_length) + 1]) <= 0.0
                for horizon in reachable_horizons
                if horizon >= int(term.branch_length)
            ):
                raise ValueError(
                    "origin_weights must have positive mass at every reachable horizon."
                )
    _validate_reference_contracts(
        terms,
        rollout_policy,
        state_layout,
        input_layout,
        float(step_size),
        float(step_rtol),
        float(step_atol),
    )

    train_source = _NeuralWindowSource(
        train,
        max_horizon=rollout_policy.max_horizon,
        step_size=step_size,
        step_rtol=step_rtol,
        step_atol=step_atol,
    )
    validation_source = (
        None
        if validation is None
        else _NeuralWindowSource(
            validation,
            max_horizon=rollout_policy.max_horizon,
            step_size=step_size,
            step_rtol=step_rtol,
            step_atol=step_atol,
        )
    )
    resolved_batch_size = train_source.size if batch_size is None else int(batch_size)
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be positive.")
    resolved_validation_batch = (
        None
        if validation_source is None
        else validation_source.size
        if validation_batch_size is None
        else int(validation_batch_size)
    )
    if resolved_validation_batch is not None and resolved_validation_batch < 1:
        raise ValueError("validation_batch_size must be positive.")
    batches_per_epoch = ceil(train_source.size / resolved_batch_size)
    maximum_steps = (
        int(steps)
        if steps is not None
        else int(epochs) * ceil(batches_per_epoch / int(gradient_accumulation))
    )
    validation_config = (
        DiscreteModelValidationPolicy()
        if validation is not None and validation_policy is None
        else validation_policy
    )
    if validation_config is not None and not isinstance(
        validation_config,
        DiscreteModelValidationPolicy,
    ):
        raise TypeError(
            "validation_policy must be a DiscreteModelValidationPolicy or None."
        )
    if validation_config is not None and validation_source is None:
        raise ValueError("validation_policy requires validation data.")

    parameters, fixed = partition_trainable(model)
    accumulation_dtype = _tree_real_result_dtype(parameters)
    optimizer_state = optimizer.init(parameters)
    evaluated_parameters = resolve_evaluation_parameters(
        evaluation_parameters,
        optimizer_state,
        parameters,
    )
    initial_target_parameters = (
        evaluated_parameters
        if isinstance(target_policy, ExponentialMovingAverageTargetPolicy)
        and target_policy.source == "evaluation"
        else parameters
    )
    target_state = (
        None
        if target_policy is None
        else TargetParameterState.initialize(initial_target_parameters, target_policy)
    )
    evaluation_model = eqx.nn.inference_mode(
        combine_trainable(evaluated_parameters, fixed)
    )
    best_model = evaluation_model
    master_key = jr.key(seed) if key is None else key
    progress = TrainingProgress()
    metric_names = ("loss",) + tuple(term.name for term in terms)
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    fit_contract = {
        "model_id": resolved_model_id,
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_signature": array_tree_signature(model),
        "train_fingerprint": train_source.fingerprint,
        "validation_fingerprint": (
            None if validation_source is None else validation_source.fingerprint
        ),
        "state_layout": state_layout.layout_id,
        "input_layout": None if input_layout is None else input_layout.layout_id,
        "system_id": system_id,
        "step_size": float(step_size),
        "step_rtol": float(step_rtol),
        "step_atol": float(step_atol),
        "rollout": rollout_policy.fingerprint,
        "objectives": [term.fingerprint for term in terms],
        "optimizer_id": resolved_optimizer_id,
        "evaluation_parameters_id": resolved_evaluation_id,
        "target_policy": None if target_policy is None else asdict(target_policy),
        "batch_size": resolved_batch_size,
        "validation_batch_size": resolved_validation_batch,
        "shuffle": bool(shuffle),
        "seed": int(seed),
        "gradient_accumulation": int(gradient_accumulation),
        "validation_policy": (
            None if validation_config is None else asdict(validation_config)
        ),
        "jit": bool(jit),
        "key_policy_id": _KEY_POLICY_ID,
        "root_key": array_tree_fingerprint(master_key),
    }
    fit_fingerprint = canonical_fingerprint(fit_contract)

    control = TrainingController(
        total_steps=maximum_steps,
        key=master_key,
        progress=progress,
        callbacks=callbacks,
    )
    train_steps: list[int] = []
    train_history: list[dict[str, float]] = []
    validation_steps: list[int] = []
    validation_history: list[dict[str, float]] = []
    resumed_from_step = 0
    prior_training_seconds = 0.0

    def loss_components(current_model, target_model, batch, root_key, step):
        active_horizon = rollout_policy.active_horizon(step)
        return _objective_contributions(
            current_model,
            batch,
            active_horizon,
            rollout_policy,
            terms,
            state_layout,
            root_key,
            step,
            target_model=target_model,
        )

    def gradient_fn(current_parameters, target_parameters, batch, root_key, step):
        def objective(candidate):
            current_model = combine_trainable(candidate, fixed)
            target_model = (
                combine_trainable(target_parameters, fixed)
                if target_state is not None
                else None
            )
            contribution, components, valid = loss_components(
                current_model,
                target_model,
                batch,
                root_key,
                step,
            )
            return contribution.numerator, (
                contribution.numerator,
                contribution.support,
                contribution.log_scale,
                tuple(
                    (component.numerator, component.support, component.log_scale)
                    for component in components
                ),
                valid,
            )

        (_, auxiliary), gradient = eqx.filter_value_and_grad(
            objective,
            has_aux=True,
        )(current_parameters)
        numerator, support, log_scale, component_arrays, valid = auxiliary
        finite = valid & _tree_finite(
            (numerator, support, log_scale, component_arrays, gradient)
        )
        return (
            (numerator, support, log_scale),
            component_arrays,
            gradient,
            finite,
        )

    def update_fn(current_parameters, current_state, gradient):
        updates, next_state = optimizer.update(
            gradient,
            current_state,
            current_parameters,
        )
        next_parameters = eqx.apply_updates(current_parameters, updates)
        return next_parameters, next_state, _tree_finite((next_parameters, next_state))

    run_gradient = eqx.filter_jit(gradient_fn) if jit else gradient_fn
    run_update = eqx.filter_jit(update_fn) if jit else update_fn

    def batches(source, epoch, size, *, shuffle_data):
        indices = source.ordered_indices(epoch, shuffle=shuffle_data, seed=seed)
        for batch_index, start in enumerate(range(0, source.size, size)):
            yield batch_index, source.prepare(indices[start : start + size])

    def evaluate(current_model, source, size, step):
        metric_accumulators = [_ObjectiveAccumulator() for _ in metric_names]
        evaluation_key = control.key_for(int(step), site=1000)
        target_model = (
            combine_trainable(target_state.target, fixed)
            if target_state is not None
            else None
        )
        for _, batch in batches(source, 0, size, shuffle_data=False):
            total, components, valid_array = loss_components(
                current_model,
                target_model,
                batch,
                evaluation_key,
                jnp.asarray(step, dtype=jnp.int32),
            )
            if not bool(jax.device_get(valid_array)):
                raise FloatingPointError(
                    "Model, reference, target, or residual failed during evaluation."
                )
            metric_accumulators = [
                accumulator.add(contribution)
                for accumulator, contribution in zip(
                    metric_accumulators,
                    (total,) + components,
                    strict=True,
                )
            ]
        return {
            name: float(jax.device_get(accumulator.value))
            for name, accumulator in zip(
                metric_names,
                metric_accumulators,
                strict=True,
            )
        }

    initial_metrics: dict[str, float]
    resume_manifest = None
    if checkpoint is not None and resume and (checkpoint / "manifest.json").is_file():
        resume_manifest, _ = _read_neural_training_manifest(checkpoint)
    if resume_manifest is not None:
        assert checkpoint is not None
        if resume_manifest["metadata"]["fit_fingerprint"] != fit_fingerprint:
            raise ValueError("Discrete-model checkpoint fit contract mismatch.")
        restored = _load_neural_training_checkpoint(
            checkpoint,
            (model, best_model),
            (optimizer_state, target_state),
        )
        model, best_model = restored.model
        optimizer_state, target_state = restored.optimizer_state
        metadata = restored.metadata
        progress = TrainingProgress(**metadata["progress"])
        if progress.update_step != restored.step or progress.update_step > maximum_steps:
            raise ValueError("Discrete-model checkpoint progress is incompatible.")
        master_key = restored.key
        control = TrainingController(
            total_steps=maximum_steps,
            key=master_key,
            progress=progress,
            callbacks=callbacks,
        )
        control.best_payload = best_model
        resumed_from_step = progress.update_step
        prior_training_seconds = float(metadata["training_seconds"])
        initial_metrics = dict(metadata["initial_metrics"])
        train_steps = [int(value) for value in metadata["train_steps"]]
        train_history = [dict(value) for value in metadata["train_metrics"]]
        validation_steps = [int(value) for value in metadata["validation_steps"]]
        validation_history = [dict(value) for value in metadata["validation_metrics"]]
        parameters, fixed = partition_trainable(model)
    else:
        initial_metrics = evaluate(
            evaluation_model,
            train_source,
            resolved_batch_size,
            0,
        )
        if validation_source is not None:
            assert resolved_validation_batch is not None
            assert validation_config is not None
            validation_metrics = evaluate(
                evaluation_model,
                validation_source,
                int(resolved_validation_batch),
                0,
            )
            if validation_config.monitor not in validation_metrics:
                raise KeyError(
                    f"Unknown validation monitor {validation_config.monitor!r}."
                )
            validation_steps.append(0)
            validation_history.append(validation_metrics)
            control.best_payload = evaluation_model
            best_model = evaluation_model
            control.progress = replace(
                control.progress,
                best_value=validation_metrics[validation_config.monitor],
                best_step=0,
            )

    def save_progress(training_seconds):
        if checkpoint is None or not gradient_accumulator.is_empty:
            return
        _save_neural_training_checkpoint(
            checkpoint,
            (model, best_model),
            (optimizer_state, target_state),
            step=control.progress.update_step,
            key=master_key,
            metadata={
                "fit_fingerprint": fit_fingerprint,
                "fit_contract": fit_contract,
                "progress": asdict(control.progress),
                "initial_metrics": initial_metrics,
                "train_steps": train_steps,
                "train_metrics": train_history,
                "validation_steps": validation_steps,
                "validation_metrics": validation_history,
                "training_seconds": float(training_seconds),
            },
        )
        control.emit("checkpoint", metrics={"step": control.progress.update_step})

    def consider_validation(metrics, current_model):
        if validation_config is None:
            raise RuntimeError("Validation configuration is unavailable.")
        nonlocal best_model
        score = float(metrics[validation_config.monitor])
        previous = control.progress.best_value
        strict = previous is None or (
            score < previous if validation_config.mode == "min" else score > previous
        )
        required = (
            float(validation_config.minimum_delta)
            if previous is None
            else max(
                float(validation_config.minimum_delta),
                float(validation_config.relative_minimum_delta)
                * max(abs(previous), 1e-12),
            )
        )
        meaningful = previous is None or (
            score < previous - required
            if validation_config.mode == "min"
            else score > previous + required
        )
        if strict:
            best_model = current_model
            control.best_payload = current_model
        stale = 0 if meaningful else control.progress.stale_validations + 1
        stopped = validation_config.patience is not None and stale >= int(
            validation_config.patience
        )
        control.progress = replace(
            control.progress,
            best_value=score if strict else previous,
            best_step=control.progress.update_step
            if strict
            else control.progress.best_step,
            stale_validations=stale,
            stopped_early=stopped,
        )
        if stopped:
            control.stop_requested = True

    logger_context = (
        nullcontext(None)
        if tensorboard_log_dir is None
        else TensorBoardLogger(tensorboard_log_dir)
    )
    started = time.perf_counter()
    stopped_by_signal = False
    control.emit("train_begin", metrics=initial_metrics)
    with logger_context as tensorboard, TrainingSignalGuard() as signal_guard:
        gradient_accumulator = _GradientAccumulationState.empty(
            parameters,
            accumulation_dtype=accumulation_dtype,
        )
        accumulated_metrics = [_ObjectiveAccumulator() for _ in metric_names]
        has_trainable = any(
            eqx.is_array(leaf) for leaf in jax.tree_util.tree_leaves(parameters)
        )
        if has_trainable and not control.progress.stopped_early:
            for epoch in range(control.progress.epoch, int(epochs)):
                if control.stop_requested or signal_guard.stop_requested:
                    break
                control.emit("epoch_begin", metrics={"epoch": epoch})
                for batch_index, batch in batches(
                    train_source,
                    epoch,
                    resolved_batch_size,
                    shuffle_data=shuffle,
                ):
                    if batch_index < control.progress.next_batch_index:
                        continue
                    if control.progress.update_step >= maximum_steps:
                        break
                    root_key = control.key_for(control.progress.microstep, site=0)
                    total_arrays, component_arrays, gradient, finite_array = run_gradient(
                        parameters,
                        (target_state.target if target_state is not None else parameters),
                        batch,
                        root_key,
                        jnp.asarray(
                            control.progress.update_step,
                            dtype=jnp.int32,
                        ),
                    )
                    if not bool(jax.device_get(finite_array)):
                        raise FloatingPointError(
                            "Nonfinite model, reference, residual, loss, or gradient encountered."
                        )
                    total_contribution = _ObjectiveContribution(*total_arrays)
                    component_contributions = tuple(
                        _ObjectiveContribution(*values) for values in component_arrays
                    )
                    gradient_accumulator = gradient_accumulator.add(
                        gradient,
                        total_contribution,
                    )
                    accumulated_metrics = [
                        accumulator.add(contribution)
                        for accumulator, contribution in zip(
                            accumulated_metrics,
                            (total_contribution,) + component_contributions,
                            strict=True,
                        )
                    ]
                    control.progress = replace(
                        control.progress,
                        microstep=control.progress.microstep + 1,
                        next_batch_index=batch_index + 1,
                    )
                    end_of_epoch = batch_index + 1 >= batches_per_epoch
                    if (
                        gradient_accumulator.microsteps < int(gradient_accumulation)
                        and not end_of_epoch
                    ):
                        continue
                    if not bool(
                        jax.device_get(gradient_accumulator.has_positive_support)
                    ):
                        control.emit("zero_support")
                        gradient_accumulator = _GradientAccumulationState.empty(
                            parameters,
                            accumulation_dtype=accumulation_dtype,
                        )
                        accumulated_metrics = [
                            _ObjectiveAccumulator() for _ in metric_names
                        ]
                        if control.stop_requested or signal_guard.stop_requested:
                            break
                        continue

                    averaged_gradient = gradient_accumulator.normalized_gradient(
                        parameters
                    )
                    candidate_parameters, candidate_state, candidate_finite = run_update(
                        parameters,
                        optimizer_state,
                        averaged_gradient,
                    )
                    if not bool(jax.device_get(candidate_finite)):
                        raise FloatingPointError("Optimizer produced nonfinite state.")
                    parameters = candidate_parameters
                    optimizer_state = candidate_state
                    model = combine_trainable(parameters, fixed)
                    update_step = control.progress.update_step + 1
                    control.complete_update(update_step)
                    if target_state is not None:
                        target_state = target_state.update(
                            parameters,
                            accepted=True,
                            evaluation_parameters=resolve_evaluation_parameters(
                                evaluation_parameters,
                                optimizer_state,
                                parameters,
                            ),
                        )
                    metrics = {
                        name: float(jax.device_get(accumulator.value))
                        for name, accumulator in zip(
                            metric_names,
                            accumulated_metrics,
                            strict=True,
                        )
                    }
                    train_steps.append(update_step)
                    train_history.append(metrics)
                    gradient_accumulator = _GradientAccumulationState.empty(
                        parameters,
                        accumulation_dtype=accumulation_dtype,
                    )
                    accumulated_metrics = [_ObjectiveAccumulator() for _ in metric_names]
                    control.emit("batch_end", metrics=metrics)
                    if (
                        tensorboard is not None
                        and update_step % int(tensorboard_every) == 0
                    ):
                        for name, value in metrics.items():
                            tensorboard.scalar(f"train/{name}", value, update_step)
                    if (
                        validation_source is not None
                        and validation_config is not None
                        and update_step % int(validation_config.every) == 0
                    ):
                        assert resolved_validation_batch is not None
                        evaluated_parameters = resolve_evaluation_parameters(
                            evaluation_parameters,
                            optimizer_state,
                            parameters,
                        )
                        evaluation_model = eqx.nn.inference_mode(
                            combine_trainable(evaluated_parameters, fixed)
                        )
                        validation_metrics = evaluate(
                            evaluation_model,
                            validation_source,
                            int(resolved_validation_batch),
                            update_step,
                        )
                        if validation_config.monitor not in validation_metrics:
                            raise KeyError(
                                f"Unknown validation monitor {validation_config.monitor!r}."
                            )
                        validation_steps.append(update_step)
                        validation_history.append(validation_metrics)
                        consider_validation(validation_metrics, evaluation_model)
                        control.emit("validation_end", metrics=validation_metrics)
                        if tensorboard is not None:
                            for name, value in validation_metrics.items():
                                tensorboard.scalar(
                                    f"validation/{name}",
                                    value,
                                    update_step,
                                )
                    elapsed = prior_training_seconds + time.perf_counter() - started
                    if (
                        checkpoint is not None
                        and update_step % int(checkpoint_every) == 0
                    ):
                        save_progress(elapsed)
                    if control.stop_requested or signal_guard.stop_requested:
                        break
                if control.progress.next_batch_index >= batches_per_epoch:
                    control.progress = replace(
                        control.progress,
                        epoch=epoch + 1,
                        next_batch_index=0,
                    )
                if control.progress.update_step >= maximum_steps:
                    break
        stopped_by_signal = signal_guard.stop_requested

    training_seconds = prior_training_seconds + time.perf_counter() - started
    evaluated_parameters = resolve_evaluation_parameters(
        evaluation_parameters,
        optimizer_state,
        parameters,
    )
    evaluation_model = eqx.nn.inference_mode(
        combine_trainable(evaluated_parameters, fixed)
    )
    if (
        validation_source is not None
        and validation_config is not None
        and (not validation_steps or validation_steps[-1] != control.progress.update_step)
    ):
        assert resolved_validation_batch is not None
        validation_metrics = evaluate(
            evaluation_model,
            validation_source,
            int(resolved_validation_batch),
            control.progress.update_step,
        )
        validation_steps.append(control.progress.update_step)
        validation_history.append(validation_metrics)
        consider_validation(validation_metrics, evaluation_model)
    selected_model = (
        best_model
        if validation_config is not None and validation_config.select_best
        else evaluation_model
    )
    final_metrics = evaluate(
        selected_model,
        train_source,
        resolved_batch_size,
        control.progress.update_step,
    )
    if checkpoint is not None and (
        control.progress.update_step == 0
        or not train_steps
        or train_steps[-1] == control.progress.update_step
    ):
        save_progress(training_seconds)
    control.emit("train_end", metrics=final_metrics)
    history = DiscreteModelFitHistory(
        initial_metrics=frozendict(initial_metrics),
        train_steps=tuple(train_steps),
        train_metrics=tuple(frozendict(value) for value in train_history),
        validation_steps=tuple(validation_steps),
        validation_metrics=tuple(frozendict(value) for value in validation_history),
        final_metrics=frozendict(final_metrics),
    )
    transition = DiscreteModelTransition(
        selected_model,
        state_layout=state_layout,
        input_layout=input_layout,
        step_size=step_size,
        step_rtol=step_rtol,
        step_atol=step_atol,
    )
    sample_state = train.states.reshape((-1,) + state_layout.shape)[0]
    sample_coordinate = jnp.asarray(0.0, dtype=train.coordinates.dtype)
    sample_context = DiscreteStepContext(
        sample_coordinate,
        sample_coordinate + step_size,
        jnp.asarray(0, dtype=jnp.int32),
    )
    if input_layout is None:
        jax.eval_shape(
            lambda state: transition(sample_context, state, None),
            sample_state,
        )
    else:
        assert train.inputs is not None
        sample_input = train.inputs.reshape((-1,) + input_layout.shape)[0]
        jax.eval_shape(
            lambda state, inputs: transition(
                sample_context,
                state,
                inputs,
                None,
            ),
            sample_state,
            sample_input,
        )
    system = DiscreteSystem(
        transition,
        state_layout=state_layout,
        input_layout=input_layout,
        system_id=system_id,
        step_size=step_size,
        step_rtol=step_rtol,
        step_atol=step_atol,
    )
    return DiscreteModelFitResult(
        model=selected_model,
        last_model=evaluation_model,
        system=system,
        history=history,
        progress=control.progress,
        resumed_from_step=resumed_from_step,
        training_seconds=training_seconds,
        checkpoint_path=checkpoint,
        stopped_by_signal=stopped_by_signal,
        stopped_by_callback=control.stop_requested and not control.progress.stopped_early,
    )


__all__ = [
    "DiscreteModelFitHistory",
    "DiscreteModelFitResult",
    "DiscreteModelRolloutPolicy",
    "DiscreteModelValidationPolicy",
    "ReferenceBranchDiscreteModelObjective",
    "ResidualDiscreteModelObjective",
    "TargetDiscreteModelObjective",
    "SupervisedDiscreteModelObjective",
    "fit_discrete_model",
]
