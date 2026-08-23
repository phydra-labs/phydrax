#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import prod
from typing import Any, cast, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from .._doc import DOC_KEY0
from .._fingerprint import canonical_fingerprint
from .._flow_matching_metric import (
    AbstractFlowMatchingMetric,
    EuclideanFlowMatchingMetric,
    ManifoldFlowMatchingMetric,
    RiemannianFlowMatchingMetric,
)
from .._frozendict import frozendict
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ..transport.continuous._coupling import EndpointCouplingSample
from ..transport.continuous._interpolant import AbstractEndpointInterpolant
from ._sample_statistics import effective_sample_size, normalized_log_weights


FlowMatchingSamplingMode: TypeAlias = Literal["fixed", "resample"]
FlowEndpointProvider: TypeAlias = Callable[[Key[Array, ""]], EndpointCouplingSample]


class FlowMatchingPolicy(StrictModule):
    """Time-sampling policy for endpoint flow matching."""

    minimum_time: Array
    maximum_time: Array
    distribution: Literal["uniform"] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        minimum_time: ArrayLike = 0.0,
        maximum_time: ArrayLike = 1.0,
        /,
        *,
        distribution: Literal["uniform"] = "uniform",
        policy_id: str | None = None,
    ):
        lower = jnp.asarray(minimum_time, dtype=float).reshape(())
        upper = jnp.asarray(maximum_time, dtype=float).reshape(())
        if not bool(jnp.isfinite(lower) & jnp.isfinite(upper)):
            raise ValueError("Flow-matching time bounds must be finite.")
        if not bool(upper > lower):
            raise ValueError("Flow-matching maximum_time must exceed minimum_time.")
        if distribution != "uniform":
            raise ValueError(
                "The initial flow-matching policy supports only uniform time."
            )
        resolved_id = (
            canonical_fingerprint(
                {
                    "kind": "flow-matching-time-policy-v1",
                    "minimum_time": float(lower),
                    "maximum_time": float(upper),
                    "distribution": distribution,
                }
            )
            if policy_id is None
            else str(policy_id)
        )
        if not resolved_id:
            raise ValueError("policy_id must be non-empty.")
        self.minimum_time = lower
        self.maximum_time = upper
        self.distribution = distribution
        self.policy_id = resolved_id


class FlowMatchingBatch(StrictModule):
    """Materialized endpoint-interpolation batch for one objective evaluation."""

    state: Array
    time: Array
    target_velocity: Array
    valid: Array
    log_weights: Array
    context: frozendict[str, Array]
    evaluation_key: Array
    source_indices: Array
    target_indices: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)
    num_pairs: int = eqx.field(static=True)
    interpolant_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state: ArrayLike,
        time: ArrayLike,
        target_velocity: ArrayLike,
        valid: ArrayLike,
        log_weights: ArrayLike,
        context: Mapping[str, ArrayLike] | None,
        evaluation_key: Key[Array, ""],
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        interpolant_id: str,
        coupling_id: str,
        policy_id: str,
        batch_id: str,
    ):
        state_array = jnp.asarray(state)
        velocity = jnp.asarray(target_velocity, dtype=state_array.dtype)
        if state_array.shape != velocity.shape or state_array.ndim < 1:
            raise ValueError(
                "Flow-matching state and target velocity require matching pair-first shapes."
            )
        count = int(state_array.shape[0])
        expected = (count,)
        time_array = jnp.asarray(time, dtype=state_array.real.dtype)
        validity = jnp.asarray(valid, dtype=bool)
        weights = jnp.asarray(log_weights, dtype=float)
        source_index = jnp.asarray(source_indices, dtype=jnp.int32)
        target_index = jnp.asarray(target_indices, dtype=jnp.int32)
        if not (
            time_array.shape
            == validity.shape
            == weights.shape
            == source_index.shape
            == target_index.shape
            == expected
        ):
            raise ValueError("Flow-matching pair metadata must have shape (num_pairs,).")
        resolved_context = frozendict(
            {}
            if context is None
            else {str(name): jnp.asarray(value) for name, value in context.items()}
        )
        for name, value in resolved_context.items():
            if not name or name in ("x", "t", "source", "target"):
                raise ValueError(f"Flow-matching context label {name!r} is invalid.")
            if value.ndim < 1 or int(value.shape[0]) != count:
                raise ValueError(
                    f"Flow-matching context {name!r} must begin with {count}; "
                    f"got {value.shape}."
                )
        for name, identifier in (
            ("interpolant_id", interpolant_id),
            ("coupling_id", coupling_id),
            ("policy_id", policy_id),
            ("batch_id", batch_id),
        ):
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{name} must be a non-empty string.")
        self.state = state_array
        self.time = time_array
        self.target_velocity = velocity
        self.valid = validity
        self.log_weights = weights
        self.context = resolved_context
        self.evaluation_key = jnp.asarray(evaluation_key)
        self.source_indices = source_index
        self.target_indices = target_index
        self.event_shape = tuple(state_array.shape[1:])
        self.num_pairs = count
        self.interpolant_id = interpolant_id
        self.coupling_id = coupling_id
        self.policy_id = policy_id
        self.batch_id = batch_id


class FlowMatchingDiagnostics(StrictModule):
    objective: Array
    root_mean_squared_component_error: Array
    mean_predicted_velocity_norm: Array
    mean_target_velocity_norm: Array
    valid_fraction: Array
    effective_sample_size: Array
    minimum_sampled_time: Array
    maximum_sampled_time: Array
    mean_sampled_time: Array
    finite: Array
    num_pairs: int = eqx.field(static=True)
    event_size: int = eqx.field(static=True)
    interpolant_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return bool(self.finite) and bool(self.valid_fraction > 0.0)


class _FlowMatchingNodeEvaluation(StrictModule):
    loss: Array
    squared_component_error: Array
    predicted_norm: Array
    target_norm: Array
    weights: Array
    valid: Array
    time: Array


class FlowMatchingTerm(AbstractSamplingTerm):
    """Endpoint-coupled conditional velocity regression objective."""

    fixed_endpoints: EndpointCouplingSample | None
    endpoint_provider: FlowEndpointProvider | None
    interpolant: AbstractEndpointInterpolant
    policy: FlowMatchingPolicy
    metric: AbstractFlowMatchingMetric
    scalar_weight: Array
    velocity_name: str = eqx.field(static=True)
    state_label: str = eqx.field(static=True)
    time_label: str = eqx.field(static=True)
    sampling_mode: FlowMatchingSamplingMode = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        velocity_name: str,
        endpoints: EndpointCouplingSample | FlowEndpointProvider,
        interpolant: AbstractEndpointInterpolant,
        /,
        *,
        policy: FlowMatchingPolicy | None = None,
        metric: AbstractFlowMatchingMetric | None = None,
        sampling_mode: FlowMatchingSamplingMode = "fixed",
        scalar_weight: ArrayLike = 1.0,
        state_label: str = "x",
        time_label: str = "t",
        label: str | None = None,
    ):
        if not isinstance(velocity_name, str) or not velocity_name:
            raise ValueError("velocity_name must be a non-empty string.")
        if not isinstance(interpolant, AbstractEndpointInterpolant):
            raise TypeError("interpolant must implement AbstractEndpointInterpolant.")
        if sampling_mode not in ("fixed", "resample"):
            raise ValueError("sampling_mode must be 'fixed' or 'resample'.")
        if not state_label or not time_label or state_label == time_label:
            raise ValueError("state_label and time_label must be distinct and non-empty.")
        resolved_policy = FlowMatchingPolicy() if policy is None else policy
        resolved_metric = EuclideanFlowMatchingMetric() if metric is None else metric
        if not isinstance(resolved_policy, FlowMatchingPolicy):
            raise TypeError("policy must be FlowMatchingPolicy or None.")
        if not isinstance(resolved_metric, AbstractFlowMatchingMetric):
            raise TypeError("metric must implement AbstractFlowMatchingMetric.")
        if bool(resolved_policy.minimum_time < interpolant.source_coordinate) or bool(
            resolved_policy.maximum_time > interpolant.target_coordinate
        ):
            raise ValueError(
                "Flow-matching time policy exceeds the interpolant interval."
            )
        if sampling_mode == "fixed":
            if not isinstance(endpoints, EndpointCouplingSample):
                raise TypeError("Fixed flow matching requires EndpointCouplingSample.")
            fixed = endpoints
            provider = None
        else:
            if not callable(endpoints):
                raise TypeError("Resampled flow matching requires an endpoint provider.")
            fixed = None
            provider = cast(FlowEndpointProvider, endpoints)
        if fixed is not None and fixed.event_shape != interpolant.event_shape:
            raise ValueError("Endpoint and interpolant event shapes must match.")
        weight = jnp.asarray(scalar_weight, dtype=float).reshape(())
        if not bool(jnp.isfinite(weight)) or float(weight) < 0.0:
            raise ValueError("scalar_weight must be finite and nonnegative.")
        self.fixed_endpoints = fixed
        self.endpoint_provider = provider
        self.interpolant = interpolant
        self.policy = resolved_policy
        self.metric = resolved_metric
        self.scalar_weight = weight
        self.velocity_name = velocity_name
        self.state_label = str(state_label)
        self.time_label = str(time_label)
        self.sampling_mode = sampling_mode
        self.label = None if label is None else str(label)

    def sample(self, *, key: Key[Array, ""] = DOC_KEY0) -> FlowMatchingBatch:
        endpoint_key, time_key, evaluation_key = jr.split(key, 3)
        if self.sampling_mode == "fixed":
            if self.fixed_endpoints is None:
                raise RuntimeError("Fixed endpoint coupling is unavailable.")
            endpoints = self.fixed_endpoints
        else:
            if self.endpoint_provider is None:
                raise RuntimeError("Endpoint provider is unavailable.")
            endpoints = self.endpoint_provider(endpoint_key)
        if endpoints.event_shape != self.interpolant.event_shape:
            raise ValueError("Endpoint provider returned an incompatible event shape.")
        time = jr.uniform(
            time_key,
            (endpoints.num_pairs,),
            minval=self.policy.minimum_time,
            maxval=self.policy.maximum_time,
            dtype=endpoints.source.real.dtype,
        )
        evaluation = self.interpolant.evaluate(
            time,
            endpoints.source,
            endpoints.target,
        )
        return FlowMatchingBatch(
            state=evaluation.state,
            time=evaluation.time,
            target_velocity=evaluation.conditional_velocity,
            valid=endpoints.valid & evaluation.valid,
            log_weights=endpoints.log_weights,
            context=endpoints.context,
            evaluation_key=evaluation_key,
            source_indices=endpoints.source_indices,
            target_indices=endpoints.target_indices,
            interpolant_id=evaluation.interpolant_id,
            coupling_id=endpoints.coupling_id,
            policy_id=self.policy.policy_id,
            batch_id=self.label
            or canonical_fingerprint(
                {
                    "interpolant": evaluation.interpolant_id,
                    "coupling": endpoints.coupling_id,
                    "policy": self.policy.policy_id,
                    "metric": self.metric.metric_id,
                }
            ),
        )

    def _velocity_function(
        self,
        functions: Mapping[str, DomainFunction],
        batch: FlowMatchingBatch,
        /,
    ) -> DomainFunction:
        if self.velocity_name not in functions:
            raise KeyError(f"Missing velocity field {self.velocity_name!r}.")
        velocity = functions[self.velocity_name]
        if not isinstance(velocity, DomainFunction):
            raise TypeError("velocity field must be a DomainFunction.")
        allowed = {self.state_label, self.time_label, *batch.context}
        unknown = tuple(label for label in velocity.deps if label not in allowed)
        if unknown or self.state_label not in velocity.deps:
            raise ValueError(
                "velocity field must depend on the state and only declared time/context labels."
            )
        return velocity

    def _evaluate_nodes(
        self,
        functions: Mapping[str, DomainFunction],
        batch: FlowMatchingBatch,
        /,
    ) -> _FlowMatchingNodeEvaluation:
        if not isinstance(batch, FlowMatchingBatch):
            raise TypeError("batch must be FlowMatchingBatch.")
        velocity = self._velocity_function(functions, batch)
        count = batch.num_pairs
        state_shape = batch.event_shape
        state_rank = len(state_shape)
        event_axes = tuple(range(1, 1 + state_rank))
        valid = batch.valid
        expanded_valid = valid.reshape((count,) + (1,) * state_rank)
        safe_state = jnp.where(expanded_valid, batch.state, jnp.zeros_like(batch.state))
        safe_target = jnp.where(
            expanded_valid,
            batch.target_velocity,
            jnp.zeros_like(batch.target_velocity),
        )
        safe_time = jnp.where(valid, batch.time, 0.0)
        safe_context = {
            name: jnp.where(
                valid.reshape((count,) + (1,) * (value.ndim - 1)),
                value,
                jnp.zeros_like(value),
            )
            for name, value in batch.context.items()
        }
        arguments: list[Array] = []
        for dependency in velocity.deps:
            if dependency == self.state_label:
                arguments.append(safe_state)
            elif dependency == self.time_label:
                arguments.append(safe_time)
            else:
                arguments.append(safe_context[dependency])
        node_keys = jr.split(batch.evaluation_key, count)

        def velocity_at(key, *values):
            return jnp.asarray(velocity.func(*values, key=key))

        predicted = jax.vmap(velocity_at)(node_keys, *arguments)
        if predicted.shape != batch.state.shape:
            raise ValueError(
                "velocity field must preserve the complete interpolant event shape."
            )
        predicted = jnp.where(expanded_valid, predicted, jnp.zeros_like(predicted))
        loss = jax.vmap(self.metric)(safe_state, predicted, safe_target)
        residual_squared = jnp.abs(predicted - safe_target) ** 2
        predicted_squared = jnp.abs(predicted) ** 2
        target_squared = jnp.abs(safe_target) ** 2
        if event_axes:
            residual_squared = jnp.sum(residual_squared, axis=event_axes)
            predicted_squared = jnp.sum(predicted_squared, axis=event_axes)
            target_squared = jnp.sum(target_squared, axis=event_axes)
        weights = normalized_log_weights(batch.log_weights, valid)
        return _FlowMatchingNodeEvaluation(
            loss=loss,
            squared_component_error=residual_squared / float(max(prod(state_shape), 1)),
            predicted_norm=predicted_squared,
            target_norm=target_squared,
            weights=weights,
            valid=valid,
            time=batch.time,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        batch: FlowMatchingBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        materialized = self.sample(key=key) if batch is None else batch
        evaluation = self._evaluate_nodes(functions, materialized)
        return self.scalar_weight * jnp.sum(evaluation.weights * evaluation.loss)

    def diagnostics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: FlowMatchingBatch | None = None,
    ) -> FlowMatchingDiagnostics:
        materialized = self.sample(key=key) if batch is None else batch
        evaluation = self._evaluate_nodes(functions, materialized)
        objective = self.scalar_weight * jnp.sum(evaluation.weights * evaluation.loss)
        rms = jnp.sqrt(jnp.sum(evaluation.weights * evaluation.squared_component_error))
        predicted_norm = jnp.sqrt(jnp.sum(evaluation.weights * evaluation.predicted_norm))
        target_norm = jnp.sqrt(jnp.sum(evaluation.weights * evaluation.target_norm))
        minimum_time = jnp.min(jnp.where(evaluation.valid, evaluation.time, jnp.inf))
        maximum_time = jnp.max(jnp.where(evaluation.valid, evaluation.time, -jnp.inf))
        mean_time = jnp.sum(evaluation.weights * evaluation.time)
        finite = (
            jnp.isfinite(objective)
            & jnp.isfinite(rms)
            & jnp.isfinite(predicted_norm)
            & jnp.isfinite(target_norm)
            & jnp.isfinite(minimum_time)
            & jnp.isfinite(maximum_time)
        )
        return FlowMatchingDiagnostics(
            objective=objective,
            root_mean_squared_component_error=rms,
            mean_predicted_velocity_norm=predicted_norm,
            mean_target_velocity_norm=target_norm,
            valid_fraction=jnp.mean(evaluation.valid),
            effective_sample_size=effective_sample_size(evaluation.weights),
            minimum_sampled_time=minimum_time,
            maximum_sampled_time=maximum_time,
            mean_sampled_time=mean_time,
            finite=finite,
            num_pairs=materialized.num_pairs,
            event_size=max(prod(materialized.event_shape), 1),
            interpolant_id=materialized.interpolant_id,
            coupling_id=materialized.coupling_id,
            policy_id=materialized.policy_id,
            metric_id=self.metric.metric_id,
        )


__all__ = [
    "AbstractFlowMatchingMetric",
    "EuclideanFlowMatchingMetric",
    "RiemannianFlowMatchingMetric",
    "ManifoldFlowMatchingMetric",
    "FlowEndpointProvider",
    "FlowMatchingBatch",
    "FlowMatchingDiagnostics",
    "FlowMatchingPolicy",
    "FlowMatchingSamplingMode",
    "FlowMatchingTerm",
]
