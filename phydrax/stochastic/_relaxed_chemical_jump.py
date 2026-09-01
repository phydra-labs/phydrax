#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_jump import ChemicalJumpProcess, ChemicalJumpRuntime


class RelaxedChemicalJumpParameters(StrictModule, NonTrainableState):
    selector_sharpness: float = eqx.field(static=True)
    jump_sharpness: float = eqx.field(static=True)
    maximum_events: int = eqx.field(static=True)
    final_time: float = eqx.field(static=True)

    def __init__(
        self,
        selector_sharpness: float,
        jump_sharpness: float,
        maximum_events: int,
        final_time: float,
        /,
    ):
        selector = float(selector_sharpness)
        jump = float(jump_sharpness)
        capacity = int(maximum_events)
        final = float(final_time)
        if (
            not np.isfinite(selector)
            or not np.isfinite(jump)
            or selector <= 0.0
            or jump <= 0.0
            or capacity <= 0
            or not np.isfinite(final)
            or final <= 0.0
        ):
            raise ValueError("Relaxed jump controls must be finite and positive.")
        self.selector_sharpness = selector
        self.jump_sharpness = jump
        self.maximum_events = capacity
        self.final_time = final


class RelaxedChemicalJumpEvidence(StrictModule):
    nonintegral_residual: Array
    minimum_state: Array
    event_count: Array
    capacity_exhausted: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class RelaxedChemicalJumpResult(StrictModule):
    final_state: Array
    final_time: Array
    states: Array
    times: Array
    selector_weights: Array
    event_mask: Array
    evidence: RelaxedChemicalJumpEvidence


class RelaxedChemicalJumpPlan(StrictModule, NonTrainableState):
    process: ChemicalJumpProcess
    parameters: RelaxedChemicalJumpParameters
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        process: ChemicalJumpProcess,
        parameters: RelaxedChemicalJumpParameters,
        /,
    ):
        if not isinstance(process, ChemicalJumpProcess):
            raise TypeError("process must be ChemicalJumpProcess.")
        if not isinstance(parameters, RelaxedChemicalJumpParameters):
            raise TypeError("parameters must be RelaxedChemicalJumpParameters.")
        self.process = process
        self.parameters = parameters
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relaxed-chemical-jump",
                "process": process.process_id,
                "selector_sharpness": parameters.selector_sharpness,
                "jump_sharpness": parameters.jump_sharpness,
                "maximum_events": parameters.maximum_events,
                "final_time": parameters.final_time,
            }
        )

    def simulate(
        self,
        initial_state: ArrayLike,
        runtime: ChemicalJumpRuntime,
        key: Array,
        /,
    ) -> RelaxedChemicalJumpResult:
        state = jnp.asarray(initial_state)
        if state.shape != self.process.state_shape:
            raise ValueError("initial_state must match chemical jump state shape.")
        if not isinstance(runtime, ChemicalJumpRuntime):
            raise TypeError("runtime must be ChemicalJumpRuntime.")
        keys = jax.random.split(key, self.parameters.maximum_events * 2).reshape(
            self.parameters.maximum_events, 2
        )
        channel_indices = jnp.arange(self.process.num_channels, dtype=state.dtype)

        def step(carry, event_keys):
            time, current = carry
            intensity = self.process.intensities(time, current, runtime)
            total = jnp.sum(intensity)
            active = (
                (time < self.parameters.final_time) & jnp.isfinite(total) & (total > 0.0)
            )
            waiting_uniform = jax.random.uniform(
                event_keys[0], (), minval=jnp.finfo(state.dtype).tiny, maxval=1.0
            )
            waiting = -jnp.log(waiting_uniform) / jnp.where(total > 0.0, total, 1.0)
            next_time = time + waiting
            active = active & (next_time <= self.parameters.final_time)
            selector = jax.random.uniform(event_keys[1], ())
            cumulative = jnp.cumsum(intensity / jnp.where(total > 0.0, total, 1.0))
            lower = jnp.concatenate((jnp.zeros((1,), dtype=state.dtype), cumulative[:-1]))
            selector_weights = jax.nn.sigmoid(
                self.parameters.selector_sharpness * (selector - lower)
            ) - jax.nn.sigmoid(
                self.parameters.selector_sharpness * (selector - cumulative)
            )
            selector_weights = selector_weights / jnp.maximum(
                jnp.sum(selector_weights), jnp.finfo(state.dtype).tiny
            )
            center = jnp.sum(selector_weights * channel_indices)
            jump_weights = jnp.exp(
                -self.parameters.jump_sharpness * (channel_indices - center) ** 2
            )
            jump_weights = jump_weights / jnp.maximum(
                jnp.sum(jump_weights), jnp.finfo(state.dtype).tiny
            )
            increment = jump_weights @ self.process.channel_stoichiometry
            candidate = current + increment
            accepted_state = jnp.where(active, candidate, current)
            accepted_time = jnp.where(active, next_time, time)
            recorded_weights = jnp.where(active, jump_weights, 0.0)
            return (accepted_time, accepted_state), (
                accepted_time,
                accepted_state,
                recorded_weights,
                active,
            )

        (final_time, final_state), history = jax.lax.scan(
            step,
            (jnp.asarray(0.0, dtype=state.dtype), state),
            keys,
        )
        times, states, weights, event_mask = history
        event_count = jnp.sum(event_mask.astype(jnp.int32))
        capacity_exhausted = (event_count == self.parameters.maximum_events) & (
            final_time < self.parameters.final_time
        )
        nonintegral = jnp.max(jnp.abs(final_state - jnp.round(final_state)))
        minimum = jnp.min(final_state)
        successful = (
            jnp.all(jnp.isfinite(final_state))
            & jnp.isfinite(final_time)
            & ~capacity_exhausted
        )
        evidence = RelaxedChemicalJumpEvidence(
            nonintegral,
            minimum,
            event_count,
            capacity_exhausted,
            successful,
            self.plan_id,
        )
        return RelaxedChemicalJumpResult(
            final_state,
            final_time,
            states,
            times,
            weights,
            event_mask,
            evidence,
        )


def relaxed_exact_moment_discrepancy(
    relaxed_samples: ArrayLike,
    exact_samples: ArrayLike,
    /,
) -> tuple[Array, Array]:
    relaxed = jnp.asarray(relaxed_samples)
    exact = jnp.asarray(exact_samples, dtype=relaxed.dtype)
    if (
        relaxed.ndim < 2
        or exact.ndim != relaxed.ndim
        or exact.shape[1:] != relaxed.shape[1:]
    ):
        raise ValueError("Relaxed and exact samples must share non-sample axes.")
    mean_error = jnp.max(jnp.abs(jnp.mean(relaxed, axis=0) - jnp.mean(exact, axis=0)))
    variance_error = jnp.max(jnp.abs(jnp.var(relaxed, axis=0) - jnp.var(exact, axis=0)))
    return mean_error, variance_error


__all__ = [
    "RelaxedChemicalJumpEvidence",
    "RelaxedChemicalJumpParameters",
    "RelaxedChemicalJumpPlan",
    "RelaxedChemicalJumpResult",
    "relaxed_exact_moment_discrepancy",
]
