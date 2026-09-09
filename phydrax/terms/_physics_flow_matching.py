#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from .._trainable import NonTrainableState
from ..domain import DomainFunction
from ..transport.continuous._coupling import EndpointCouplingSample
from ..transport.continuous._interpolant import AbstractEndpointInterpolant
from ._flow_matching import (
    AbstractFlowMatchingMetric,
    FlowEndpointProvider,
    FlowMatchingBatch,
    FlowMatchingDiagnostics,
    FlowMatchingSamplingMode,
    FlowMatchingTerm,
)
from ._sample_statistics import normalized_log_weights
from ._time_sampling import AbstractTimeSamplingPolicy


class AbstractFlowEndpointFunctional(StrictModule, NonTrainableState):
    """Measure-aware scalar physical energy evaluated on one endpoint state."""

    event_shape: tuple[int, ...] = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)

    @abstractmethod
    def __call__(self, state: Array, context: Mapping[str, Array], /) -> Array:
        raise NotImplementedError


class CallableFlowEndpointFunctional(AbstractFlowEndpointFunctional):
    """Explicitly identified endpoint functional backed by a static callable."""

    function: Callable = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Array, Mapping[str, Array]], ArrayLike],
        /,
        *,
        event_shape: tuple[int, ...],
        functional_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        shape = tuple(int(size) for size in event_shape)
        identifier = str(functional_id).strip()
        if not shape or any(size <= 0 for size in shape) or not identifier:
            raise ValueError("Endpoint functional shape and identity are invalid.")
        self.function = function
        self.event_shape = shape
        self.functional_id = identifier

    def __call__(self, state: Array, context: Mapping[str, Array], /) -> Array:
        return jnp.asarray(self.function(state, context))


class FlowEndpointRolloutPolicy(StrictModule, NonTrainableState):
    """Static-capacity Euler terminal reconstruction with a traced curriculum."""

    maximum_steps: int = eqx.field(static=True)
    initial_steps: int = eqx.field(static=True)
    transition_steps: int = eqx.field(static=True)
    schedule: Literal["constant", "linear"] = eqx.field(static=True)
    rematerialize: bool = eqx.field(static=True)
    time_power: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_steps: int,
        /,
        *,
        initial_steps: int | None = None,
        transition_steps: int = 0,
        schedule: Literal["constant", "linear"] = "constant",
        rematerialize: bool = False,
        time_power: float = 0.0,
    ):
        maximum = int(maximum_steps)
        initial = maximum if initial_steps is None else int(initial_steps)
        transition = int(transition_steps)
        power = float(time_power)
        if maximum < 1 or initial < 1 or initial > maximum or transition < 0:
            raise ValueError("Endpoint rollout capacities are invalid.")
        if schedule not in ("constant", "linear"):
            raise ValueError("schedule must be 'constant' or 'linear'.")
        if schedule == "constant" and initial != maximum:
            raise ValueError("A constant endpoint schedule requires equal step counts.")
        if schedule == "linear" and initial < maximum and transition < 1:
            raise ValueError("A nonconstant endpoint schedule requires transition_steps.")
        if not np.isfinite(power) or power < 0.0:
            raise ValueError("time_power must be finite and nonnegative.")
        self.maximum_steps = maximum
        self.initial_steps = initial
        self.transition_steps = transition
        self.schedule = schedule
        self.rematerialize = bool(rematerialize)
        self.time_power = power
        self.policy_id = canonical_fingerprint(
            {
                "kind": "flow-endpoint-rollout-policy",
                "maximum_steps": maximum,
                "initial_steps": initial,
                "transition_steps": transition,
                "schedule": schedule,
                "rematerialize": bool(rematerialize),
                "time_power": power,
                "integrator": "explicit-euler",
            }
        )

    def active_steps(self, iteration: Any, /) -> Array:
        if self.schedule == "constant" or self.initial_steps == self.maximum_steps:
            return jnp.asarray(self.maximum_steps, dtype=jnp.int32)
        step = jnp.asarray(0 if iteration is None else iteration, dtype=jnp.float32)
        progress = jnp.clip(step / float(self.transition_steps), 0.0, 1.0)
        span = self.maximum_steps - self.initial_steps
        return self.initial_steps + jnp.floor(progress * span).astype(jnp.int32)


class PhysicsFlowMatchingDiagnostics(StrictModule):
    flow: FlowMatchingDiagnostics
    flow_objective: Array
    physics_objective: Array
    mean_endpoint_energy: Array
    maximum_endpoint_energy: Array
    terminal_valid_fraction: Array
    active_steps: Array
    finite: Array
    endpoint_functional_id: str = eqx.field(static=True)
    rollout_policy_id: str = eqx.field(static=True)


class PhysicsFlowMatchingTerm(AbstractSamplingTerm):
    """Flow matching plus a physical energy on one unrolled terminal estimate."""

    flow: FlowMatchingTerm
    label: str | None = eqx.field(static=True)
    endpoint_functional: AbstractFlowEndpointFunctional
    endpoint_rollout: FlowEndpointRolloutPolicy
    physics_weight: Array

    def __init__(
        self,
        velocity_name: str,
        endpoints: EndpointCouplingSample | FlowEndpointProvider,
        interpolant: AbstractEndpointInterpolant,
        endpoint_functional: AbstractFlowEndpointFunctional,
        endpoint_rollout: FlowEndpointRolloutPolicy,
        /,
        *,
        policy: AbstractTimeSamplingPolicy | None = None,
        metric: AbstractFlowMatchingMetric | None = None,
        sampling_mode: FlowMatchingSamplingMode = "fixed",
        scalar_weight: ArrayLike = 1.0,
        physics_weight: ArrayLike = 1.0,
        state_label: str = "x",
        time_label: str = "t",
        label: str | None = None,
    ):
        if not isinstance(endpoint_functional, AbstractFlowEndpointFunctional):
            raise TypeError(
                "endpoint_functional must implement AbstractFlowEndpointFunctional."
            )
        if not isinstance(endpoint_rollout, FlowEndpointRolloutPolicy):
            raise TypeError("endpoint_rollout must be FlowEndpointRolloutPolicy.")
        if endpoint_functional.event_shape != interpolant.event_shape:
            raise ValueError(
                "Endpoint functional and interpolant event shapes must match."
            )
        physics = jnp.asarray(physics_weight, dtype=float).reshape(())
        if not bool(jnp.isfinite(physics)) or float(physics) < 0.0:
            raise ValueError("physics_weight must be finite and nonnegative.")
        self.flow = FlowMatchingTerm(
            velocity_name,
            endpoints,
            interpolant,
            policy=policy,
            metric=metric,
            sampling_mode=sampling_mode,
            scalar_weight=scalar_weight,
            state_label=state_label,
            time_label=time_label,
            label=label,
        )
        self.endpoint_functional = endpoint_functional
        self.endpoint_rollout = endpoint_rollout
        self.physics_weight = physics
        self.label = self.flow.label

    def sample(self, *, key: Key[Array, ""] = DOC_KEY0) -> FlowMatchingBatch:
        return self.flow.sample(key=key)

    def _terminal_endpoint(
        self,
        functions: Mapping[str, DomainFunction],
        batch: FlowMatchingBatch,
        iteration: Any,
        /,
    ) -> tuple[Array, Array, Array]:
        velocity = self.flow._velocity_function(functions, batch)
        count = batch.num_pairs
        state_rank = len(batch.event_shape)
        valid = batch.valid
        expanded = valid.reshape((count,) + (1,) * state_rank)
        initial_state = jnp.where(expanded, batch.state, jnp.zeros_like(batch.state))
        initial_time = jnp.where(valid, batch.time, 0.0)
        active_steps = self.endpoint_rollout.active_steps(iteration)
        target_time = jnp.asarray(
            self.flow.interpolant.target_coordinate,
            dtype=batch.time.dtype,
        )
        step_size = (target_time - initial_time) / active_steps.astype(batch.time.dtype)
        context_names = tuple(batch.context)
        context_values = tuple(batch.context[name] for name in context_names)
        node_keys = jr.split(batch.evaluation_key, count)

        def step(carry, depth):
            state, time = carry
            enabled = depth < active_steps
            keys = jax.vmap(lambda value: jr.fold_in(value, depth))(node_keys)

            def velocity_at(key, point, coordinate, *contexts):
                values = {
                    name: context
                    for name, context in zip(context_names, contexts, strict=True)
                }
                arguments = []
                for dependency in velocity.deps:
                    if dependency == self.flow.state_label:
                        arguments.append(point)
                    elif dependency == self.flow.time_label:
                        arguments.append(coordinate)
                    else:
                        arguments.append(values[dependency])
                return jnp.asarray(velocity.func(*arguments, key=key))

            predicted = jax.vmap(velocity_at)(
                keys,
                state,
                time,
                *context_values,
            )
            if predicted.shape != state.shape:
                raise ValueError(
                    "Flow velocity must preserve the endpoint event shape during unrolling."
                )
            next_state = (
                state + step_size.reshape((count,) + (1,) * state_rank) * predicted
            )
            next_time = time + step_size
            state = jnp.where(expanded & enabled, next_state, state)
            time = jnp.where(valid & enabled, next_time, time)
            return (state, time), None

        scan_step = jax.checkpoint(step) if self.endpoint_rollout.rematerialize else step
        (endpoint, final_time), _ = jax.lax.scan(
            scan_step,
            (initial_state, initial_time),
            jnp.arange(self.endpoint_rollout.maximum_steps, dtype=jnp.int32),
        )
        event_axes = tuple(range(1, endpoint.ndim))
        finite = jnp.isfinite(final_time)
        if event_axes:
            finite = finite & jnp.all(jnp.isfinite(endpoint), axis=event_axes)
        terminal_valid = valid & finite & jnp.isclose(final_time, target_time)
        endpoint = jnp.where(
            terminal_valid.reshape((count,) + (1,) * state_rank),
            endpoint,
            jnp.zeros_like(endpoint),
        )
        return endpoint, terminal_valid, active_steps

    def _physics_component(
        self,
        functions: Mapping[str, DomainFunction],
        batch: FlowMatchingBatch,
        iteration: Any,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        endpoint, valid, active_steps = self._terminal_endpoint(
            functions,
            batch,
            iteration,
        )
        context_names = tuple(batch.context)
        context_values = tuple(batch.context[name] for name in context_names)

        def evaluate_endpoint(state, *contexts):
            values = {
                name: context
                for name, context in zip(context_names, contexts, strict=True)
            }
            return jnp.asarray(self.endpoint_functional(state, values)).reshape(())

        energies = jax.vmap(evaluate_endpoint)(endpoint, *context_values)
        if energies.shape != (batch.num_pairs,):
            raise ValueError("Endpoint functional must return one scalar per flow pair.")
        energies = eqx.error_if(
            energies,
            jnp.any(valid & (~jnp.isfinite(energies) | (energies < 0.0))),
            "Active endpoint physical energies must be finite and nonnegative.",
        )
        weights = normalized_log_weights(batch.log_weights, valid)
        source = jnp.asarray(
            self.flow.interpolant.source_coordinate, dtype=batch.time.dtype
        )
        target = jnp.asarray(
            self.flow.interpolant.target_coordinate, dtype=batch.time.dtype
        )
        progress = jnp.clip((batch.time - source) / (target - source), 0.0, 1.0)
        time_weight = progress**self.endpoint_rollout.time_power
        value = self.physics_weight * jnp.sum(weights * time_weight * energies)
        return value, energies, valid, active_steps

    def objective_components(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: Any = None,
        batch: FlowMatchingBatch | None = None,
        **kwargs: Any,
    ) -> tuple[Array, Array]:
        del kwargs
        materialized = self.sample(key=key) if batch is None else batch
        evaluation = self.flow._evaluate_nodes(functions, materialized)
        precision = self.flow.metric.precision
        flow = precision.decision(
            precision.decision(self.flow.scalar_weight)
            * precision.sum(evaluation.weights * precision.accumulation(evaluation.loss))
        )
        physics, _, _, _ = self._physics_component(
            functions,
            materialized,
            iter_,
        )
        return flow, physics

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: Any = None,
        batch: FlowMatchingBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        flow, physics = self.objective_components(
            functions,
            key=key,
            iter_=iter_,
            batch=batch,
            **kwargs,
        )
        return flow + physics

    def diagnostics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: FlowMatchingBatch | None = None,
        iter_: Any = None,
    ) -> PhysicsFlowMatchingDiagnostics:
        materialized = self.sample(key=key) if batch is None else batch
        flow = self.flow.diagnostics(functions, key=key, batch=materialized)
        physics, energies, valid, active_steps = self._physics_component(
            functions,
            materialized,
            iter_,
        )
        weights = normalized_log_weights(materialized.log_weights, valid)
        safe = jnp.where(valid, energies, 0.0)
        return PhysicsFlowMatchingDiagnostics(
            flow=flow,
            flow_objective=flow.objective,
            physics_objective=physics,
            mean_endpoint_energy=jnp.sum(weights * safe),
            maximum_endpoint_energy=jnp.max(safe, initial=0.0),
            terminal_valid_fraction=jnp.mean(valid.astype(float)),
            active_steps=active_steps,
            finite=flow.finite & jnp.all(~valid | jnp.isfinite(energies)),
            endpoint_functional_id=self.endpoint_functional.functional_id,
            rollout_policy_id=self.endpoint_rollout.policy_id,
        )


__all__ = [
    "AbstractFlowEndpointFunctional",
    "CallableFlowEndpointFunctional",
    "FlowEndpointRolloutPolicy",
    "PhysicsFlowMatchingDiagnostics",
    "PhysicsFlowMatchingTerm",
]
