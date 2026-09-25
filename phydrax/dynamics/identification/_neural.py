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
from typing import Any, final, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array

from ..._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from ..._fingerprint import (
    array_tree_fingerprint,
    array_tree_signature,
    canonical_fingerprint,
)
from ..._frozendict import frozendict
from ..._iteration import IterationSession
from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from ..._trainable import (
    combine_parameters,
    partition_parameters,
    require_parameter_roles,
)
from ..._training import (
    _update_validation_selection,
    DelayedTargetPolicy,
    EvaluationParametersFn,
    ExponentialMovingAverageTargetPolicy,
    TensorBoardLogger,
    TrainingController,
    TrainingIterationKind,
    TrainingProgress,
    TrainingSignalGuard,
)
from ..._training_checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_metadata,
    save_training_checkpoint,
)
from ..._training_kernel import (
    build_training_checkpoint,
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    run_training_attempt,
    training_accepted_site_key,
    TrainingAttemptOutcome,
    TrainingKernelSpec,
    TrainingRejectionBudgetError,
)
from ..._training_objective import (
    _ObjectiveAccumulator,
    _ObjectiveContribution,
)
from ...metrix import EuclideanStateGeometry
from .._layout import InputLayout, StateLayout
from .._system import DiscreteStepContext, DiscreteSystem
from .._trajectory import TrajectoryData
from ._linear_refinement import (
    ProgressiveLinearRefinementPolicy,
    ProgressiveLinearRefinementRecord,
    ProgressiveLinearRefinementState,
)
from ._neural_transition import (
    AbstractDiscreteModelRolloutTransition,
    DirectDiscreteModelRolloutTransition,
)
from ._neural_windows import (
    _active_window_evidence,
    _KEY_POLICY_ID,
    _NeuralWindowBatch,
    _NeuralWindowSource,
    _semantic_window_keys,
)


_OBJECTIVE_ID = "discrete-model-rollout"
_CHECKPOINT_FORMAT = "phydrax-discrete-model-training-checkpoint"


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
    fit_fingerprint: str
    history: DiscreteModelFitHistory
    progress: TrainingProgress
    resumed_from_step: int
    linear_refinement_state: ProgressiveLinearRefinementState | None
    linear_refinement_records: tuple[ProgressiveLinearRefinementRecord, ...]
    training_seconds: float
    checkpoint_path: Path | None
    stopped_by_signal: bool = False
    stopped_by_host_control: bool = False

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
    transition: AbstractDiscreteModelRolloutTransition,
    truncate_every: int | None,
    root_key: Array,
    iteration: Array,
    execution_control: Any,
) -> tuple[_RolloutCarry, tuple[Array, Array]]:
    active = depth < active_horizon
    run = eligible & active
    keys = _semantic_window_keys(
        root_key,
        batch.parent_index,
        batch.start_index,
        depth,
    )
    controls = None if batch.inputs is None else batch.inputs[:, depth]

    sources = batch.coordinates[:, depth]
    if controls is None:

        def one(state, source, key, enabled):
            def advance(_):
                result = transition.evaluate(
                    model,
                    DiscreteStepContext(
                        source,
                        source + transition.step_size,
                        depth,
                    ),
                    state,
                    None,
                    key=key,
                    iteration=iteration,
                    control=execution_control,
                )
                return result.accepted_state, result.training_usable

            return jax.lax.cond(
                enabled,
                advance,
                lambda _: (state, jnp.asarray(True)),
                operand=None,
            )

        candidate, transition_valid = jax.vmap(one)(
            carry.state,
            sources,
            keys,
            run,
        )
    else:

        def one(state, inputs, source, key, enabled):
            def advance(_):
                result = transition.evaluate(
                    model,
                    DiscreteStepContext(
                        source,
                        source + transition.step_size,
                        depth,
                    ),
                    state,
                    inputs,
                    key=key,
                    iteration=iteration,
                    control=execution_control,
                )
                return result.accepted_state, result.training_usable

            return jax.lax.cond(
                enabled,
                advance,
                lambda _: (state, jnp.asarray(True)),
                operand=None,
            )

        candidate, transition_valid = jax.vmap(one)(
            carry.state,
            controls,
            sources,
            keys,
            run,
        )
    finite = jnp.all(
        jnp.isfinite(candidate),
        axis=tuple(range(1, candidate.ndim)),
    )
    member = jax.vmap(state_layout.geometry.contains)(candidate)
    step_valid = ~run | (transition_valid & finite & member)
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
    transition: AbstractDiscreteModelRolloutTransition,
    execution_control: Any,
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
            execution_control=execution_control,
            transition=transition,
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
    coefficient_total = jnp.zeros((batch.size,), dtype=endpoint_states.dtype)
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
                    enabled,
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
    transition: AbstractDiscreteModelRolloutTransition,
    root_key: Array,
    iteration: Array | None = None,
    /,
    *,
    target_model: AbstractArrayModel | None = None,
    target_key: Array | None = None,
    execution_control: Any = None,
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
        transition,
        execution_control,
        root_key,
        resolved_iteration,
    )
    runtime_valid = rollout_valid
    target_endpoint_states = None
    if any(isinstance(value, TargetDiscreteModelObjective) for value in objectives):
        if target_model is None or target_key is None:
            raise ValueError("Target objective requires a target model and key.")
        target_endpoint_states, _, _, target_valid = _rollout_states(
            target_model,
            batch,
            active_horizon,
            policy,
            state_layout,
            transition,
            execution_control,
            target_key,
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


@final
class _DiscreteRolloutObjective(StrictModule):
    """Kernel objective of one discrete-model fit.

    The rollout transition is a visible dynamic child: its arrays (for example a
    discretization, operators, and projection solver) are PyTree leaves instead
    of state captured by the objective. `evaluate` is a stateless operation over
    the array-free fit configuration and receives the transition explicitly.
    """

    transition: AbstractDiscreteModelRolloutTransition
    evaluate: Callable[..., Any] = eqx.field(static=True)

    def __call__(
        self, parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any, /
    ) -> tuple[_ObjectiveContribution, Any, Any]:
        return self.evaluate(
            self.transition, parameters, model_state, fixed, payload, keys
        )


def _tree_real_result_dtype(tree: Any, /):
    dtypes = tuple(
        leaf.dtype for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_array(leaf)
    )
    if not dtypes:
        return jnp.dtype(jnp.float32)
    return jnp.result_type(*dtypes)


def _accumulate(kernel, state, payload):
    return kernel.accumulate_with_diagnostics(state, payload)


_compiled_accumulate = eqx.filter_jit(_accumulate)


def _add_metric_diagnostics(
    accumulators: list[_ObjectiveAccumulator], diagnostics: tuple[Any, ...], /
) -> list[_ObjectiveAccumulator]:
    """Merge one microbatch's total and per-term contributions into the metrics."""
    (terms,) = diagnostics
    return [
        accumulator.add(_ObjectiveContribution(*values))
        for accumulator, values in zip(accumulators, terms, strict=True)
    ]


def _validate_precision(model: AbstractArrayModel, *datasets: TrajectoryData) -> None:
    allowed = (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64))
    parameters, _, _ = partition_parameters(model)
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


def _resolve_discrete_fit_request(
    model: AbstractArrayModel,
    rollout_policy: DiscreteModelRolloutPolicy,
    state_layout: StateLayout,
    input_layout: InputLayout | None,
    train: TrajectoryData,
    validation: TrajectoryData | None,
    /,
    *,
    model_id: str | None,
    checkpoint_path: str | Path | None,
    system_id: str,
    transition: AbstractDiscreteModelRolloutTransition | None,
    step_size: float,
    step_rtol: float,
    step_atol: float,
    linear_refinement: ProgressiveLinearRefinementPolicy | None,
    gradient_accumulation: int,
    epochs: int,
    steps: int | None,
    checkpoint_every: int,
    tensorboard_every: int,
    evaluation_parameters: EvaluationParametersFn | None,
    evaluation_parameters_id: str | None,
    optimizer: Any,
    optimizer_id: str | None,
    learning_rate: float,
):
    if not isinstance(model, AbstractArrayModel):
        raise TypeError("fit_discrete_model requires an AbstractArrayModel.")
    require_parameter_roles(model, context="fit_discrete_model")
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
    resolved_transition = (
        DirectDiscreteModelRolloutTransition(
            state_layout,
            input_layout=input_layout,
            step_size=step_size,
            step_rtol=step_rtol,
            step_atol=step_atol,
        )
        if transition is None
        else transition
    )
    if not isinstance(
        resolved_transition,
        AbstractDiscreteModelRolloutTransition,
    ):
        raise TypeError(
            "transition must be an AbstractDiscreteModelRolloutTransition or None."
        )
    if (
        resolved_transition.state_layout.layout_id != state_layout.layout_id
        or (
            None
            if resolved_transition.input_layout is None
            else resolved_transition.input_layout.layout_id
        )
        != (None if input_layout is None else input_layout.layout_id)
        or resolved_transition.step_size != float(step_size)
        or resolved_transition.step_rtol != float(step_rtol)
        or resolved_transition.step_atol != float(step_atol)
    ):
        raise ValueError(
            "Discrete rollout transition and fit contracts must match exactly."
        )
    resolved_transition.validate_model(model)
    if linear_refinement is not None:
        if not isinstance(linear_refinement, ProgressiveLinearRefinementPolicy):
            raise TypeError(
                "linear_refinement must be ProgressiveLinearRefinementPolicy or None."
            )
        if not resolved_transition.supports_linear_refinement:
            raise ValueError(
                "The selected discrete rollout transition does not support linear refinement."
            )
        if validation is None:
            raise ValueError("Linear refinement requires fixed validation data.")
        if int(gradient_accumulation) != 1:
            raise ValueError("Linear refinement does not support gradient accumulation.")
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
    return (
        resolved_model_id,
        resolved_transition,
        resolved_evaluation_id,
        optimizer,
        resolved_optimizer_id,
    )


def _validate_discrete_fit_objectives(
    objectives: Sequence[Any] | None,
    rollout_policy: DiscreteModelRolloutPolicy,
    target_policy: DelayedTargetPolicy | ExponentialMovingAverageTargetPolicy | None,
    resolved_transition: AbstractDiscreteModelRolloutTransition,
    state_layout: StateLayout,
    input_layout: InputLayout | None,
    step_size: float,
    step_rtol: float,
    step_atol: float,
    /,
) -> tuple[Any, ...]:
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
    return terms


def _prepare_discrete_fit_sources(
    train: TrajectoryData,
    validation: TrajectoryData | None,
    rollout_policy: DiscreteModelRolloutPolicy,
    /,
    *,
    step_size: float,
    step_rtol: float,
    step_atol: float,
    batch_size: int | None,
    validation_batch_size: int | None,
    steps: int | None,
    epochs: int,
    gradient_accumulation: int,
    validation_policy: DiscreteModelValidationPolicy | None,
    linear_refinement: ProgressiveLinearRefinementPolicy | None,
):
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
    if linear_refinement is not None:
        assert validation_config is not None
        if validation_config.mode != "min":
            raise ValueError("Linear refinement requires a minimized validation metric.")
    return (
        train_source,
        validation_source,
        resolved_batch_size,
        resolved_validation_batch,
        batches_per_epoch,
        maximum_steps,
        validation_config,
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
    transition: AbstractDiscreteModelRolloutTransition | None = None,
    rollout_policy: DiscreteModelRolloutPolicy,
    linear_refinement: ProgressiveLinearRefinementPolicy | None = None,
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
    session: IterationSession | None = None,
    tensorboard_log_dir: str | Path | None = None,
    tensorboard_every: int = 1,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 1,
    resume: bool = False,
) -> DiscreteModelFitResult:
    """Fit a deterministic pointwise next-state model from fixed-step trajectories."""

    (
        resolved_model_id,
        resolved_transition,
        resolved_evaluation_id,
        optimizer,
        resolved_optimizer_id,
    ) = _resolve_discrete_fit_request(
        model,
        rollout_policy,
        state_layout,
        input_layout,
        train,
        validation,
        model_id=model_id,
        checkpoint_path=checkpoint_path,
        system_id=system_id,
        transition=transition,
        step_size=step_size,
        step_rtol=step_rtol,
        step_atol=step_atol,
        linear_refinement=linear_refinement,
        gradient_accumulation=gradient_accumulation,
        epochs=epochs,
        steps=steps,
        checkpoint_every=checkpoint_every,
        tensorboard_every=tensorboard_every,
        evaluation_parameters=evaluation_parameters,
        evaluation_parameters_id=evaluation_parameters_id,
        optimizer=optimizer,
        optimizer_id=optimizer_id,
        learning_rate=learning_rate,
    )
    terms = _validate_discrete_fit_objectives(
        objectives,
        rollout_policy,
        target_policy,
        resolved_transition,
        state_layout,
        input_layout,
        step_size,
        step_rtol,
        step_atol,
    )
    (
        train_source,
        validation_source,
        resolved_batch_size,
        resolved_validation_batch,
        batches_per_epoch,
        maximum_steps,
        validation_config,
    ) = _prepare_discrete_fit_sources(
        train,
        validation,
        rollout_policy,
        step_size=step_size,
        step_rtol=step_rtol,
        step_atol=step_atol,
        batch_size=batch_size,
        validation_batch_size=validation_batch_size,
        steps=steps,
        epochs=epochs,
        gradient_accumulation=gradient_accumulation,
        validation_policy=validation_policy,
        linear_refinement=linear_refinement,
    )

    parameters, _, _ = partition_parameters(model)
    accumulation_dtype = _tree_real_result_dtype(parameters)
    master_key = jr.key(seed) if key is None else jnp.asarray(key)
    if not jax.dtypes.issubdtype(master_key.dtype, jax.dtypes.prng_key):
        master_key = jr.wrap_key_data(master_key)
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
        "transition": resolved_transition.transition_id,
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
        "linear_refinement": (
            None if linear_refinement is None else linear_refinement.policy_id
        ),
        "jit": bool(jit),
        "key_policy_id": _KEY_POLICY_ID,
        "iteration_session": None if session is None else session.session_id,
        "iteration_control": None if session is None else session.control_id,
        "root_key": array_tree_fingerprint(master_key),
    }
    fit_fingerprint = canonical_fingerprint(fit_contract)

    def training_objective(transition, parameters, model_state, fixed, payload, keys):
        batch, target_parameters, execution_control = payload
        step = keys.accepted_cursor
        total, components, valid = _objective_contributions(
            combine_parameters(parameters, model_state, fixed),
            batch,
            rollout_policy.active_horizon(step),
            rollout_policy,
            terms,
            state_layout,
            transition,
            keys.attempt_key("rollout"),
            step,
            target_model=(
                None
                if target_parameters is None
                else combine_parameters(target_parameters, model_state, fixed)
            ),
            target_key=keys.attempt_key("target-rollout"),
            execution_control=execution_control,
        )
        # A failed model, reference, target, or residual rollout is a nonfinite
        # evaluation, so the kernel rolls the attempt back.
        numerator = jnp.where(valid, total.numerator, jnp.nan)
        diagnostics = tuple(
            (value.numerator, value.support, value.log_scale)
            for value in (total, *components)
        )
        return (
            _ObjectiveContribution(numerator, total.support, total.log_scale),
            model_state,
            diagnostics,
        )

    kernel = prepare_training_kernel(
        model,
        (
            KernelObjective(
                objective_id=_OBJECTIVE_ID,
                kind=ObjectiveKind.ROLLOUT,
                route=DerivativeRoute.UNROLLED,
                fn=_DiscreteRolloutObjective(resolved_transition, training_objective),
            ),
        ),
        TrainingKernelSpec(
            OptaxUpdateRule(
                optimizer,
                rule_id=canonical_fingerprint(
                    {
                        "kind": "optax",
                        "optimizer_id": resolved_optimizer_id,
                        "evaluation_parameters_id": resolved_evaluation_id,
                    }
                ),
                evaluation_parameters=evaluation_parameters,
            ),
            context="fit_discrete_model",
            rejection_budget=0,
            target_policy=target_policy,
            accumulation_dtype=accumulation_dtype,
        ),
        root_authority=ComponentAuthority.MODEL,
    )
    state = kernel.init(model, master_key)

    def evaluation_view():
        evaluated = kernel.rule.evaluation_parameters(state.rule_state, state.parameters)
        return eqx.nn.inference_mode(
            combine_parameters(evaluated, state.model_state, kernel.fixed)
        )

    evaluation_model = evaluation_view()
    best_model = evaluation_model

    control = TrainingController(
        total_steps=maximum_steps,
        algorithm_id="discrete-model-training",
        progress=progress,
        session=session,
    )
    train_steps: list[int] = []
    train_history: list[dict[str, float]] = []
    validation_steps: list[int] = []
    validation_history: list[dict[str, float]] = []
    resumed_from_step = 0
    prior_training_seconds = 0.0
    refinement_state = (
        None if linear_refinement is None else linear_refinement.initialize()
    )
    refinement_records: list[ProgressiveLinearRefinementRecord] = []

    def training_execution_control():
        if linear_refinement is None:
            return None
        assert refinement_state is not None
        return linear_refinement.training_control(refinement_state)

    evaluation_execution_control = (
        None if linear_refinement is None else linear_refinement.evaluation_control()
    )

    def batches(source, epoch, size, *, shuffle_data):
        indices = source.ordered_indices(epoch, shuffle=shuffle_data, seed=seed)
        for batch_index, start in enumerate(range(0, source.size, size)):
            yield batch_index, source.prepare(indices[start : start + size])

    def evaluate(current_model, source, size, step, execution_control):
        metric_accumulators = [_ObjectiveAccumulator() for _ in metric_names]
        evaluation_key, target_key = (
            training_accepted_site_key(
                master_key,
                objective_id=_OBJECTIVE_ID,
                site=site,
                accepted=int(step),
                microstep=0,
            )
            for site in ("evaluation-rollout", "evaluation-target-rollout")
        )
        target_model = None if state.targets is None else kernel.target_tree(state)
        for _, batch in batches(source, 0, size, shuffle_data=False):
            total, components, valid_array = _objective_contributions(
                current_model,
                batch,
                rollout_policy.active_horizon(jnp.asarray(step, dtype=jnp.int32)),
                rollout_policy,
                terms,
                state_layout,
                resolved_transition,
                evaluation_key,
                jnp.asarray(step, dtype=jnp.int32),
                target_model=target_model,
                target_key=target_key,
                execution_control=execution_control,
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
    resume_metadata = None
    if checkpoint is not None and resume and (checkpoint / "manifest.json").is_file():
        resume_metadata = read_training_checkpoint_metadata(
            checkpoint, format=_CHECKPOINT_FORMAT
        )
    if resume_metadata is not None:
        assert checkpoint is not None
        if resume_metadata.get("fit_fingerprint") != fit_fingerprint:
            raise ValueError("Discrete-model checkpoint fit contract mismatch.")
        loaded = load_training_checkpoint(
            checkpoint,
            kernel,
            state,
            best_model,
            format=_CHECKPOINT_FORMAT,
        )
        state = loaded.restored.state
        best_model = loaded.extra
        metadata = loaded.metadata
        if loaded.restored.selection is None:
            raise ValueError("Discrete-model checkpoint progress is missing.")
        progress = loaded.restored.selection
        if (
            progress.update_step != int(jax.device_get(state.accepted_cursor))
            or progress.update_step > maximum_steps
            or int(jax.device_get(state.microstep)) != 0
        ):
            raise ValueError("Discrete-model checkpoint progress is incompatible.")
        control = TrainingController(
            total_steps=maximum_steps,
            algorithm_id="discrete-model-training",
            progress=progress,
            session=session,
        )
        control.best_payload = best_model
        resumed_from_step = progress.update_step
        prior_training_seconds = float(metadata["training_seconds"])
        initial_metrics = dict(metadata["initial_metrics"])
        train_steps = [int(value) for value in metadata["train_steps"]]
        train_history = [dict(value) for value in metadata["train_metrics"]]
        validation_steps = [int(value) for value in metadata["validation_steps"]]
        validation_history = [dict(value) for value in metadata["validation_metrics"]]
        if linear_refinement is not None:
            saved_refinement = dict(metadata["linear_refinement_state"])
            saved_refinement["history"] = tuple(saved_refinement["history"])
            refinement_state = ProgressiveLinearRefinementState(**saved_refinement)
            refinement_records = [
                ProgressiveLinearRefinementRecord(**dict(value))
                for value in metadata["linear_refinement_records"]
            ]
    else:
        initial_metrics = evaluate(
            evaluation_model,
            train_source,
            resolved_batch_size,
            0,
            evaluation_execution_control,
        )
        if validation_source is not None:
            assert resolved_validation_batch is not None
            assert validation_config is not None
            validation_metrics = evaluate(
                evaluation_model,
                validation_source,
                int(resolved_validation_batch),
                0,
                evaluation_execution_control,
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
            if linear_refinement is not None:
                assert refinement_state is not None
                refinement_state, record = linear_refinement.observe(
                    refinement_state,
                    validation_metrics[validation_config.monitor],
                    validation_step=0,
                )
                refinement_records.append(record)

    window_microbatches = 0

    def save_progress(training_seconds, *, emit_event=True):
        if checkpoint is None or window_microbatches:
            return
        if emit_event:
            control.emit(
                TrainingIterationKind.CHECKPOINT,
                metrics={"step": control.progress.update_step},
            )
        save_training_checkpoint(
            checkpoint,
            # A closed window after a zero-support skip is a committed state
            # whose only change is the kernel's attempt bookkeeping.
            build_training_checkpoint(
                kernel, state, selection=control.progress, allow_intermediate=True
            ),
            best_model,
            format=_CHECKPOINT_FORMAT,
            metadata={
                "fit_fingerprint": fit_fingerprint,
                "fit_contract": fit_contract,
                "initial_metrics": initial_metrics,
                "train_steps": train_steps,
                "train_metrics": train_history,
                "validation_steps": validation_steps,
                "validation_metrics": validation_history,
                "linear_refinement_state": (
                    None if refinement_state is None else asdict(refinement_state)
                ),
                "linear_refinement_records": [
                    asdict(value) for value in refinement_records
                ],
                "training_seconds": float(training_seconds),
            },
        )

    def consider_validation(metrics, current_model):
        if validation_config is None:
            raise RuntimeError("Validation configuration is unavailable.")
        nonlocal best_model
        score = float(metrics[validation_config.monitor])
        control.progress, strict_better = _update_validation_selection(
            control.progress,
            score,
            step=control.progress.update_step,
            mode=validation_config.mode,
            minimum_delta=validation_config.minimum_delta,
            relative_minimum_delta=validation_config.relative_minimum_delta,
            patience=validation_config.patience,
        )
        if strict_better:
            best_model = current_model
            control.best_payload = current_model
        if control.progress.stopped_early:
            control.stop_requested = True

    accumulate = _compiled_accumulate if jit else _accumulate
    logger_context = (
        nullcontext(None)
        if tensorboard_log_dir is None
        else TensorBoardLogger(tensorboard_log_dir)
    )
    started = time.perf_counter()
    stopped_by_signal = False
    control.emit(TrainingIterationKind.RUN_START, metrics=initial_metrics)
    with logger_context as tensorboard, TrainingSignalGuard() as signal_guard:
        accumulated_metrics = [_ObjectiveAccumulator() for _ in metric_names]
        if not control.progress.stopped_early:
            for epoch in range(control.progress.epoch, int(epochs)):
                if control.stop_requested or signal_guard.stop_requested:
                    break
                control.emit(TrainingIterationKind.EPOCH_START, metrics={"epoch": epoch})
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
                    payload = (
                        batch,
                        None if state.targets is None else state.targets.target,
                        training_execution_control(),
                    )
                    control.progress = replace(
                        control.progress,
                        microstep=control.progress.microstep + 1,
                        next_batch_index=batch_index + 1,
                    )
                    window_microbatches += 1
                    end_of_epoch = batch_index + 1 >= batches_per_epoch
                    if (
                        window_microbatches < int(gradient_accumulation)
                        and not end_of_epoch
                    ):
                        state, diagnostics = accumulate(kernel, state, payload)
                        accumulated_metrics = _add_metric_diagnostics(
                            accumulated_metrics, diagnostics
                        )
                        continue
                    window_microbatches = 0
                    try:
                        state, evidence = run_training_attempt(
                            kernel, state, payload, jit=jit
                        )
                    except TrainingRejectionBudgetError as error:
                        raise FloatingPointError(
                            "Nonfinite model, reference, residual, loss, gradient, or "
                            "optimizer state encountered; the update was rolled back."
                        ) from error
                    accumulated_metrics = _add_metric_diagnostics(
                        accumulated_metrics, evidence.diagnostics
                    )
                    # Budget 0 raises on every supported rejection, so a returned
                    # rejection is a zero-support window.
                    if int(evidence.outcome) != TrainingAttemptOutcome.ACCEPTED:
                        control.emit(TrainingIterationKind.SKIP)
                        accumulated_metrics = [
                            _ObjectiveAccumulator() for _ in metric_names
                        ]
                        if control.stop_requested or signal_guard.stop_requested:
                            break
                        continue

                    update_step = control.progress.update_step + 1
                    control.complete_update(update_step)
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
                    accumulated_metrics = [_ObjectiveAccumulator() for _ in metric_names]
                    control.emit(TrainingIterationKind.UPDATE, metrics=metrics)
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
                        evaluation_model = evaluation_view()
                        validation_metrics = evaluate(
                            evaluation_model,
                            validation_source,
                            int(resolved_validation_batch),
                            update_step,
                            evaluation_execution_control,
                        )
                        if validation_config.monitor not in validation_metrics:
                            raise KeyError(
                                f"Unknown validation monitor {validation_config.monitor!r}."
                            )
                        validation_steps.append(update_step)
                        validation_history.append(validation_metrics)
                        consider_validation(validation_metrics, evaluation_model)
                        if linear_refinement is not None:
                            assert refinement_state is not None
                            refinement_state, record = linear_refinement.observe(
                                refinement_state,
                                validation_metrics[validation_config.monitor],
                                validation_step=update_step,
                            )
                            refinement_records.append(record)
                        control.emit(
                            TrainingIterationKind.VALIDATION,
                            metrics=validation_metrics,
                        )
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
    evaluation_model = evaluation_view()
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
            evaluation_execution_control,
        )
        validation_steps.append(control.progress.update_step)
        validation_history.append(validation_metrics)
        consider_validation(validation_metrics, evaluation_model)
        if linear_refinement is not None:
            assert refinement_state is not None
            refinement_state, record = linear_refinement.observe(
                refinement_state,
                validation_metrics[validation_config.monitor],
                validation_step=control.progress.update_step,
            )
            refinement_records.append(record)
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
        evaluation_execution_control,
    )
    control.emit(TrainingIterationKind.RUN_TERMINAL, metrics=final_metrics)
    if checkpoint is not None and (
        control.progress.update_step == 0
        or not train_steps
        or train_steps[-1] == control.progress.update_step
    ):
        save_progress(training_seconds, emit_event=False)
    history = DiscreteModelFitHistory(
        initial_metrics=frozendict(initial_metrics),
        train_steps=tuple(train_steps),
        train_metrics=tuple(frozendict(value) for value in train_history),
        validation_steps=tuple(validation_steps),
        validation_metrics=tuple(frozendict(value) for value in validation_history),
        final_metrics=frozendict(final_metrics),
    )
    system = resolved_transition.bind(selected_model, system_id=system_id)
    sample_state = train.states.reshape((-1,) + state_layout.shape)[0]
    sample_coordinate = jnp.asarray(0.0, dtype=train.coordinates.dtype)
    sample_context = DiscreteStepContext(
        sample_coordinate,
        sample_coordinate + step_size,
        jnp.asarray(0, dtype=jnp.int32),
    )
    if input_layout is None:
        jax.eval_shape(
            lambda state: system.evaluate(sample_context, state, None),
            sample_state,
        )
    else:
        assert train.inputs is not None
        sample_input = train.inputs.reshape((-1,) + input_layout.shape)[0]
        jax.eval_shape(
            lambda state, inputs: system.evaluate(
                sample_context,
                state,
                None,
                inputs=inputs,
            ),
            sample_state,
            sample_input,
        )
    return DiscreteModelFitResult(
        model=selected_model,
        last_model=evaluation_model,
        system=system,
        fit_fingerprint=fit_fingerprint,
        history=history,
        progress=control.progress,
        resumed_from_step=resumed_from_step,
        linear_refinement_state=refinement_state,
        linear_refinement_records=tuple(refinement_records),
        training_seconds=training_seconds,
        checkpoint_path=checkpoint,
        stopped_by_signal=stopped_by_signal,
        stopped_by_host_control=control.stop_requested
        and not control.progress.stopped_early,
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
