#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ..stochastic import StateSpaceProblem


class StateSpacePathLogDensity(StrictModule):
    """Normalized initial, transition, and observation path-density terms."""

    problem: StateSpaceProblem
    states: Array
    prior: Array
    transition: Array
    observation: Array
    case_log_density: Array
    log_density: Array
    valid: Array
    approximation_id: str = eqx.field(static=True)


def _event_all(value: Array, event_shape: tuple[int, ...], /) -> Array:
    if not event_shape:
        return value
    axes = tuple(range(value.ndim - len(event_shape), value.ndim))
    return jnp.all(value, axis=axes)


def state_space_path_log_density(
    problem: StateSpaceProblem,
    states: Array,
    /,
) -> StateSpacePathLogDensity:
    """Evaluate one complete latent path under a bound state-space problem.

    The path contains the initial state followed by one state per observation
    step. Inactive padded steps must preserve their predecessor exactly.
    """

    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    if not problem.model.prior.has_log_density:
        raise ValueError("State-space path density requires a normalized state prior.")
    if not problem.model.transition.has_log_density:
        raise ValueError("State-space path density requires normalized transitions.")
    values = jnp.asarray(states)
    case_shape = problem.observations.case_shape
    state_shape = problem.model.state_shape
    num_steps = problem.observations.num_steps
    expected = case_shape + (num_steps + 1,) + state_shape
    if values.shape != expected:
        raise ValueError(f"states must have shape {expected}; got {values.shape}.")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("State-space latent paths must be real floating arrays.")

    case_count = prod(case_shape) if case_shape else 1
    state_size_shape = state_shape
    flat_states = values.reshape((case_count, num_steps + 1) + state_size_shape)
    times = problem.observations.times.reshape((case_count, num_steps))
    active = problem.observations.step_valid.reshape((case_count, num_steps))
    observation_values = problem.observations.values.reshape(
        (case_count, num_steps) + problem.model.observation_shape
    )
    observation_masks = problem.observations.observation_mask.reshape(
        (case_count, num_steps) + problem.model.observation_shape
    )
    initial_times = problem.initial_time.reshape((case_count,))

    sequence_axis = len(case_shape)
    initial_states = jnp.take(values, 0, axis=sequence_axis)
    prior_values = jnp.asarray(problem.model.prior.log_prob(initial_states))
    prior_terms = prior_values.reshape((case_count,))
    transition_cases = []
    observation_cases = []
    for case_index in range(case_count):
        transition_terms = []
        observation_terms = []
        for step_index in range(num_steps):
            step_active = active[case_index, step_index]
            previous_state = flat_states[case_index, step_index]
            next_state = flat_states[case_index, step_index + 1]
            start_time = (
                initial_times[case_index]
                if step_index == 0
                else times[case_index, step_index - 1]
            )
            end_time = times[case_index, step_index]
            context = problem.step_context(case_index, step_index)

            def transition_term(_):
                return jnp.asarray(
                    problem.model.transition.log_prob(
                        next_state,
                        previous_state,
                        start_time,
                        end_time,
                        context,
                    )
                ).reshape(())

            def observation_term(_):
                return jnp.asarray(
                    problem.model.observation.log_prob(
                        observation_values[case_index, step_index],
                        next_state,
                        end_time,
                        observation_masks[case_index, step_index],
                        context,
                    )
                ).reshape(())

            transition_terms.append(
                jax.lax.cond(
                    step_active,
                    transition_term,
                    lambda _: jnp.zeros((), dtype=values.dtype),
                    operand=None,
                )
            )
            observation_terms.append(
                jax.lax.cond(
                    step_active,
                    observation_term,
                    lambda _: jnp.zeros((), dtype=values.dtype),
                    operand=None,
                )
            )
        transition_cases.append(jnp.stack(transition_terms))
        observation_cases.append(jnp.stack(observation_terms))

    transition = jnp.stack(transition_cases).reshape(case_shape + (num_steps,))
    observation = jnp.stack(observation_cases).reshape(case_shape + (num_steps,))
    prior = prior_terms.reshape(case_shape)
    finite_states = _event_all(jnp.isfinite(values), state_shape)
    active_with_initial = jnp.concatenate(
        (jnp.ones(case_shape + (1,), dtype=bool), problem.observations.step_valid),
        axis=-1,
    )
    path_finite = jnp.all(finite_states | ~active_with_initial, axis=-1)
    later_states = jnp.take(values, jnp.arange(1, num_steps + 1), axis=sequence_axis)
    earlier_states = jnp.take(values, jnp.arange(num_steps), axis=sequence_axis)
    frozen = _event_all(later_states == earlier_states, state_shape)
    padding_frozen = jnp.all(problem.observations.step_valid | frozen, axis=-1)
    component_finite = (
        jnp.isfinite(prior)
        & jnp.all(jnp.isfinite(transition), axis=-1)
        & jnp.all(jnp.isfinite(observation), axis=-1)
    )
    valid = path_finite & padding_frozen & component_finite
    raw_case_log_density = prior + jnp.sum(transition + observation, axis=-1)
    case_log_density = jnp.where(valid, raw_case_log_density, -jnp.inf)
    return StateSpacePathLogDensity(
        problem=problem,
        states=values,
        prior=prior,
        transition=transition,
        observation=observation,
        case_log_density=case_log_density,
        log_density=jnp.sum(case_log_density),
        valid=valid,
        approximation_id="normalized-state-space-path-density",
    )


__all__ = ["state_space_path_log_density", "StateSpacePathLogDensity"]
