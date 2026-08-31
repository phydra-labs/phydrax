#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...._doc import DOC_KEY0
from ..._keys import EvalKey
from ..data import OperatorBatch, OperatorPrediction
from ..engine import AbstractOperatorModel
from ..sharding import OperatorShardingPolicy, shard_operator_batch
from ..task import OperatorTask
from ._dtype import OperatorDTypePolicy
from ._execution import (
    _evaluate_operator_step,
    nondimensionalize_batch,
    samples_with_values,
)
from ._normalization import OperatorNormalizationPolicy
from ._physics import OperatorOutputPipeline
from ._trained_operator import TrainedOperator


_ROLLOUT_MODEL_KEY_DOMAIN = 100


@dataclass(frozen=True)
class OperatorRolloutRoute:
    """One coincident task state routed from a physical output to its source."""

    source_name: str
    prediction_name: str
    task_field: str

    def __post_init__(self):
        for name, value in (
            ("source_name", self.source_name),
            ("prediction_name", self.prediction_name),
            ("task_field", self.task_field),
        ):
            if not str(value):
                raise ValueError(f"Operator rollout {name} must be non-empty.")
        object.__setattr__(self, "source_name", str(self.source_name))
        object.__setattr__(self, "prediction_name", str(self.prediction_name))
        object.__setattr__(self, "task_field", str(self.task_field))


@dataclass(frozen=True)
class OperatorRolloutPolicy:
    """Static-maximum recurrent horizon, truncation, and rematerialization policy."""

    maximum_horizon: int
    initial_horizon: int = 1
    transition_steps: int = 1
    truncate_every: int | None = None
    rematerialize: bool = False

    def __post_init__(self):
        if int(self.maximum_horizon) <= 0:
            raise ValueError("maximum_horizon must be positive.")
        if int(self.initial_horizon) <= 0:
            raise ValueError("initial_horizon must be positive.")
        if int(self.initial_horizon) > int(self.maximum_horizon):
            raise ValueError("initial_horizon cannot exceed maximum_horizon.")
        if int(self.transition_steps) <= 0:
            raise ValueError("transition_steps must be positive.")
        if self.truncate_every is not None and int(self.truncate_every) <= 0:
            raise ValueError("truncate_every must be positive when provided.")
        object.__setattr__(self, "maximum_horizon", int(self.maximum_horizon))
        object.__setattr__(self, "initial_horizon", int(self.initial_horizon))
        object.__setattr__(self, "transition_steps", int(self.transition_steps))
        object.__setattr__(
            self,
            "truncate_every",
            None if self.truncate_every is None else int(self.truncate_every),
        )
        object.__setattr__(self, "rematerialize", bool(self.rematerialize))

    def active_horizon(self, step: Any, /) -> Array:
        """Return the traced linearly-clamped integer horizon at an update step."""
        maximum = jnp.asarray(int(self.maximum_horizon), dtype=jnp.int32)
        initial = jnp.asarray(int(self.initial_horizon), dtype=jnp.int32)
        if int(self.maximum_horizon) == int(self.initial_horizon):
            return maximum
        progress = jnp.clip(
            jnp.asarray(step, dtype=jnp.float32) / float(self.transition_steps),
            0.0,
            1.0,
        )
        horizon = initial + jnp.floor(
            progress * float(int(self.maximum_horizon) - int(self.initial_horizon))
        ).astype(jnp.int32)
        return jnp.clip(horizon, initial, maximum)


@dataclass(frozen=True)
class OperatorRollout:
    """Physical predictions and continuation state from one deployed rollout."""

    predictions: tuple[OperatorPrediction, ...]
    final_batch: OperatorBatch
    next_step: int


class _OperatorRolloutCarry(NamedTuple):
    physical_batch: OperatorBatch
    execution_batch: OperatorBatch
    next_step: Array


def _validate_rollout_route(
    route: OperatorRolloutRoute,
    task: OperatorTask,
    output_field_map: Mapping[str, str],
    batch: OperatorBatch,
    /,
) -> None:
    if not isinstance(route, OperatorRolloutRoute):
        raise TypeError("route must be an OperatorRolloutRoute.")
    if task.problem.source_query_relation != "coincident":
        raise ValueError("Operator rollout requires a coincident source/query task.")
    if route.task_field not in task.field_by_name:
        raise KeyError(f"Unknown rollout task field {route.task_field!r}.")
    field = task.field_by_name[route.task_field]
    if not field.is_source or not field.is_target:
        raise ValueError("The routed task field must be both a source and a target.")
    if field.is_classification:
        raise ValueError("Classification fields cannot be recurrent rollout state.")
    if field.source_name != route.source_name:
        raise ValueError("Rollout source_name disagrees with the task field binding.")
    if output_field_map.get(route.prediction_name) != route.task_field:
        raise ValueError("Rollout prediction_name disagrees with the model output map.")
    assert field.query_name is not None
    source = batch.input(route.source_name)
    query = batch.query(field.query_name)
    if source.support_id != query.support_id or source.sample_shape != query.sample_shape:
        raise ValueError(
            "Operator rollout source and target query supports must coincide."
        )
    if source.values is None:
        raise ValueError("Operator rollout state source has no values.")
    assert field.output_spec is not None
    expected = batch.case_shape + source.sample_shape + field.output_spec.channel_shape
    if tuple(int(size) for size in source.values.shape) != expected:
        raise ValueError("Operator rollout source values do not match the target spec.")


def _prepare_rollout_execution_batch(
    physical_batch: OperatorBatch,
    task: OperatorTask,
    normalization: OperatorNormalizationPolicy | None,
    dtype_policy: OperatorDTypePolicy,
    sharding_policy: OperatorShardingPolicy | None,
    /,
) -> OperatorBatch:
    execution_batch = nondimensionalize_batch(physical_batch, task)
    if normalization is not None:
        execution_batch = normalization.normalize_batch(execution_batch)
    execution_batch = dtype_policy.cast_batch(execution_batch)
    if sharding_policy is not None:
        execution_batch = shard_operator_batch(execution_batch, sharding_policy)
    return execution_batch


def _feedback_physical_batch(
    batch: OperatorBatch,
    prediction: OperatorPrediction,
    route: OperatorRolloutRoute,
    /,
) -> OperatorBatch:
    inputs = dict(batch.inputs)
    source = batch.input(route.source_name)
    values = prediction.field(route.task_field).values
    assert source.values is not None
    values = values.astype(source.values.dtype)
    inputs[route.source_name] = samples_with_values(source, values)
    return OperatorBatch(
        inputs=inputs,
        queries=batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def _operator_rollout_step(
    predictor: Callable,
    model: AbstractOperatorModel,
    carry: _OperatorRolloutCarry,
    route: OperatorRolloutRoute,
    task: OperatorTask,
    output_field_map: Mapping[str, str],
    output_pipeline: OperatorOutputPipeline | None,
    normalization: OperatorNormalizationPolicy | None,
    dtype_policy: OperatorDTypePolicy,
    sharding_policy: OperatorShardingPolicy | None,
    key: EvalKey,
    /,
) -> tuple[_OperatorRolloutCarry, tuple[Any, ...]]:
    step_key = (
        None
        if key is None
        else jax.random.fold_in(
            jax.random.fold_in(key, _ROLLOUT_MODEL_KEY_DOMAIN),
            carry.next_step,
        )
    )
    execution_prediction, physical_prediction = _evaluate_operator_step(
        model,
        carry.execution_batch,
        carry.physical_batch,
        task,
        output_field_map,
        output_pipeline,
        normalization,
        dtype_policy,
        step_key,
        predictor=predictor,
    )
    next_physical_batch = _feedback_physical_batch(
        carry.physical_batch,
        physical_prediction,
        route,
    )
    next_execution_batch = _prepare_rollout_execution_batch(
        next_physical_batch,
        task,
        normalization,
        dtype_policy,
        sharding_policy,
    )
    next_carry = _OperatorRolloutCarry(
        next_physical_batch,
        next_execution_batch,
        carry.next_step + jnp.asarray(1, dtype=carry.next_step.dtype),
    )
    return next_carry, (
        execution_prediction,
        physical_prediction,
        carry.execution_batch,
        carry.physical_batch,
    )


def _stop_rollout_feedback(carry: _OperatorRolloutCarry, /) -> _OperatorRolloutCarry:
    def stop(value):
        return jax.lax.stop_gradient(value) if eqx.is_array(value) else value

    return _OperatorRolloutCarry(
        jax.tree_util.tree_map(stop, carry.physical_batch),
        jax.tree_util.tree_map(stop, carry.execution_batch),
        carry.next_step,
    )


def _operator_rollout_scan(
    predictor: Callable,
    model: AbstractOperatorModel,
    physical_batch: OperatorBatch,
    execution_batch: OperatorBatch,
    route: OperatorRolloutRoute,
    policy: OperatorRolloutPolicy,
    task: OperatorTask,
    output_field_map: Mapping[str, str],
    output_pipeline: OperatorOutputPipeline | None,
    normalization: OperatorNormalizationPolicy | None,
    dtype_policy: OperatorDTypePolicy,
    sharding_policy: OperatorShardingPolicy | None,
    key: EvalKey,
    /,
    *,
    step_offset: int = 0,
    active_horizon: int | None = None,
) -> tuple[_OperatorRolloutCarry, tuple[Any, ...]]:
    carry = _OperatorRolloutCarry(
        physical_batch,
        execution_batch,
        jnp.asarray(step_offset, dtype=jnp.int32),
    )

    def scan_step(current_carry, index):
        return _operator_rollout_step(
            predictor,
            model,
            current_carry,
            route,
            task,
            output_field_map,
            output_pipeline,
            normalization,
            dtype_policy,
            sharding_policy,
            key,
        )

    authored_step = (
        eqx.filter_checkpoint(scan_step) if policy.rematerialize else scan_step
    )
    outputs = []
    horizon = (
        int(policy.maximum_horizon) if active_horizon is None else int(active_horizon)
    )
    if horizon < 1 or horizon > int(policy.maximum_horizon):
        raise ValueError("active_horizon must lie within the rollout policy.")
    for index in range(horizon):
        carry, predictions = authored_step(carry, index)
        if (
            policy.truncate_every is not None
            and (index + 1) % int(policy.truncate_every) == 0
        ):
            carry = _stop_rollout_feedback(carry)
        outputs.append(predictions)
    return carry, tuple(zip(*outputs, strict=True))


def autoregressive_operator_rollout(
    trained_operator: TrainedOperator,
    initial_batch: OperatorBatch,
    steps: int,
    route: OperatorRolloutRoute,
    /,
    *,
    key: EvalKey = DOC_KEY0,
    step_offset: int = 0,
) -> OperatorRollout:
    """Deploy one task-bound physical-state recurrence with semantic step keys."""
    if not isinstance(trained_operator, TrainedOperator):
        raise TypeError("autoregressive_operator_rollout requires a TrainedOperator.")
    if int(steps) < 0:
        raise ValueError("steps must be non-negative.")
    if int(step_offset) < 0:
        raise ValueError("step_offset must be non-negative.")
    plan = trained_operator.execution_plan
    if not plan.contract.capabilities.autoregressive_rollout:
        raise ValueError("The trained operator architecture does not support rollout.")
    if int(step_offset) + int(steps) > int(plan.task.problem.rollout_steps):
        raise ValueError("Requested rollout exceeds the task rollout contract.")
    _validate_rollout_route(
        route,
        plan.task,
        plan.output_field_map,
        initial_batch,
    )
    prepared = plan.prepare(initial_batch)
    carry = _OperatorRolloutCarry(
        prepared.physical_batch,
        prepared.execution_batch,
        jnp.asarray(step_offset, dtype=jnp.int32),
    )
    predictions: list[OperatorPrediction] = []
    for _ in range(int(steps)):
        carry, (_, physical_prediction, _, _) = _operator_rollout_step(
            plan.lowered_callable,
            plan.execution_model,
            carry,
            route,
            plan.task,
            plan.output_field_map,
            plan.output_pipeline,
            plan.normalization,
            plan.dtype_policy,
            plan.sharding_policy,
            key,
        )
        predictions.append(physical_prediction)
    return OperatorRollout(
        predictions=tuple(predictions),
        final_batch=carry.physical_batch,
        next_step=int(step_offset) + int(steps),
    )


__all__ = [
    "OperatorRollout",
    "OperatorRolloutPolicy",
    "OperatorRolloutRoute",
    "autoregressive_operator_rollout",
]
