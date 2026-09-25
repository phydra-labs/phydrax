#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array

from .._strict import StrictModule
from ._particle import _parameter_lane, ParticleFilterResult


class StateSpaceModelScore(StrictModule):
    """Prior, transition, and observation scores over their PARAMETER leaves.

    Each score keeps its component's structure with `None` in place of FIXED and
    MODEL_STATE leaves.
    """

    prior: Any
    transition: Any
    observation: Any


class ParticleGenealogicalScoreResult(StrictModule):
    """Complete-model `O(TN)` Fisher score propagated through realized ancestry.

    Only PARAMETER leaves are scored; `parameter_paths` lists them in
    `flat_score` order (prior, then transition, then observation).
    """

    score: StateSpaceModelScore
    flat_score: Array
    case_scores: Array
    valid: Array
    filter_result: ParticleFilterResult
    parameter_paths: tuple[str, ...] = eqx.field(static=True)
    parameter_size: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    ancestry_gradient: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    input_id: str | None = eqx.field(static=True)


def _gradient(constants: Any, objective: Any, parameters: Any, /) -> Array:
    """Flat gradient of `objective(component)` over the component's PARAMETER lane."""
    gradient = jax.grad(
        lambda values: jnp.asarray(objective(eqx.combine(values, constants))).reshape(())
    )(parameters)
    return ravel_pytree(gradient)[0]


def particle_genealogical_score(
    result: ParticleFilterResult,
    /,
) -> ParticleGenealogicalScoreResult:
    """Estimate the complete stored-model score with linear particle complexity."""

    if not isinstance(result, ParticleFilterResult):
        raise TypeError("result must be a ParticleFilterResult.")
    problem = result.problem
    if not problem.model.prior.has_log_density:
        raise ValueError("A genealogical score requires a normalized state prior.")
    if not problem.model.transition.has_log_density:
        raise ValueError("A genealogical score requires normalized transitions.")
    case_count = prod(result.case_shape) if result.case_shape else 1
    num_steps = problem.observations.num_steps
    count = result.num_particles
    state_shape = result.state_shape
    observation_shape = result.observation_shape
    initial_particles = jax.lax.stop_gradient(
        result.initial_particles.reshape((case_count, count) + state_shape)
    )
    initial_log_weights = jax.lax.stop_gradient(
        result.initial_log_weights.reshape((case_count, count))
    )
    particles = jax.lax.stop_gradient(
        result.particles.reshape((case_count, num_steps, count) + state_shape)
    )
    predicted = jax.lax.stop_gradient(
        result.predicted_particles.reshape((case_count, num_steps, count) + state_shape)
    )
    posterior_log_weights = jax.lax.stop_gradient(
        result.posterior_log_weights.reshape((case_count, num_steps, count))
    )
    ancestors = jax.lax.stop_gradient(
        result.ancestor_indices.reshape((case_count, num_steps, count))
    )
    active = jax.lax.stop_gradient(result.step_valid.reshape((case_count, num_steps)))
    times = problem.observations.times.reshape((case_count, num_steps))
    initial_times = problem.initial_time.reshape((case_count,))
    observations = problem.observations.values.reshape(
        (case_count, num_steps) + observation_shape
    )
    observation_masks = problem.observations.observation_mask.reshape(
        (case_count, num_steps) + observation_shape
    )

    prior_parameters, prior_constants, _ = _parameter_lane(
        problem.model.prior, "particle_genealogical_score prior"
    )
    transition_parameters, transition_constants, _ = _parameter_lane(
        problem.model.transition, "particle_genealogical_score transition"
    )
    observation_parameters, observation_constants, _ = _parameter_lane(
        problem.model.observation, "particle_genealogical_score observation"
    )
    prior_reference, unravel_prior = ravel_pytree(prior_parameters)
    transition_reference, unravel_transition = ravel_pytree(transition_parameters)
    observation_reference, unravel_observation = ravel_pytree(observation_parameters)
    prior_size = prior_reference.size
    transition_size = transition_reference.size
    observation_size = observation_reference.size
    parameter_size = prior_size + transition_size + observation_size
    if parameter_size < 1:
        raise ValueError(
            "particle_genealogical_score: the stored state-space model has no "
            "PARAMETER leaves."
        )

    def prior_vector(particle):
        return _gradient(
            prior_constants,
            lambda prior: jnp.sum(prior.log_prob(particle)),
            prior_parameters,
        )

    initial_scores = []
    for case_index in range(case_count):
        case_scores = []
        for particle_index in range(count):
            prior_score = prior_vector(initial_particles[case_index, particle_index])
            case_scores.append(
                jnp.concatenate(
                    (
                        prior_score,
                        jnp.zeros(
                            (transition_size + observation_size,),
                            dtype=prior_score.dtype,
                        ),
                    )
                )
            )
        initial_scores.append(jnp.stack(case_scores))
    cumulative = jnp.stack(initial_scores)
    terminal_scores = cumulative
    terminal_log_weights = initial_log_weights

    for step_index in range(num_steps):
        pre_resampling_cases = []
        post_resampling_cases = []
        for case_index in range(case_count):
            context = problem.step_context(case_index, step_index)
            start_time = (
                initial_times[case_index]
                if step_index == 0
                else times[case_index, step_index - 1]
            )
            end_time = times[case_index, step_index]
            previous_particles = (
                initial_particles[case_index]
                if step_index == 0
                else particles[case_index, step_index - 1]
            )
            local_scores = []
            for particle_index in range(count):
                next_state = predicted[case_index, step_index, particle_index]
                previous_state = previous_particles[particle_index]

                def active_score(_):
                    transition_vector = _gradient(
                        transition_constants,
                        lambda transition: transition.log_prob(
                            next_state,
                            previous_state,
                            start_time,
                            end_time,
                            context,
                        ),
                        transition_parameters,
                    )
                    observation_vector = _gradient(
                        observation_constants,
                        lambda observation: observation.log_prob(
                            observations[case_index, step_index],
                            next_state,
                            end_time,
                            observation_masks[case_index, step_index],
                            context,
                        ),
                        observation_parameters,
                    )
                    return jnp.concatenate(
                        (
                            jnp.zeros((prior_size,), dtype=cumulative.dtype),
                            transition_vector.astype(cumulative.dtype),
                            observation_vector.astype(cumulative.dtype),
                        )
                    )

                local_scores.append(
                    jax.lax.cond(
                        active[case_index, step_index],
                        active_score,
                        lambda _: jnp.zeros((parameter_size,), dtype=cumulative.dtype),
                        operand=None,
                    )
                )
            pre_resampling = cumulative[case_index] + jnp.stack(local_scores)
            pre_resampling_cases.append(pre_resampling)
            post_resampling_cases.append(
                pre_resampling[ancestors[case_index, step_index]]
            )
        pre_resampling_scores = jnp.stack(pre_resampling_cases)
        post_resampling_scores = jnp.stack(post_resampling_cases)
        step_active = active[:, step_index]
        terminal_scores = jnp.where(
            step_active[:, None, None],
            pre_resampling_scores,
            terminal_scores,
        )
        terminal_log_weights = jnp.where(
            step_active[:, None],
            posterior_log_weights[:, step_index],
            terminal_log_weights,
        )
        cumulative = jnp.where(
            step_active[:, None, None],
            post_resampling_scores,
            cumulative,
        )

    normalized_weights = jnp.exp(terminal_log_weights)
    case_scores = jnp.sum(normalized_weights[..., None] * terminal_scores, axis=1)
    valid = result.successful.reshape((case_count,)) & result.initial_valid.reshape(
        (case_count,)
    )
    case_scores = jnp.where(valid[:, None], case_scores, 0.0)
    flat_score = jnp.sum(case_scores, axis=0)
    prior_score = unravel_prior(flat_score[:prior_size])
    transition_score = unravel_transition(
        flat_score[prior_size : prior_size + transition_size]
    )
    observation_score = unravel_observation(flat_score[prior_size + transition_size :])
    score = StateSpaceModelScore(
        prior=prior_score,
        transition=transition_score,
        observation=observation_score,
    )
    path_leaves = jax.tree_util.tree_flatten_with_path(score)[0]
    parameter_paths = tuple(
        jax.tree_util.keystr(path) or "<root>"
        for path, leaf in path_leaves
        if eqx.is_inexact_array(leaf)
    )
    return ParticleGenealogicalScoreResult(
        score=score,
        flat_score=flat_score,
        case_scores=case_scores.reshape(result.case_shape + (parameter_size,)),
        valid=valid.reshape(result.case_shape),
        filter_result=result,
        parameter_paths=parameter_paths,
        parameter_size=parameter_size,
        method_id="particle-complete-model-genealogical-score",
        ancestry_gradient="stopped-realized-ancestry",
        model_id=result.model_id,
        problem_id=result.problem_id,
        sequence_id=result.sequence_id,
        input_id=result.input_id,
    )


__all__ = [
    "particle_genealogical_score",
    "ParticleGenealogicalScoreResult",
    "StateSpaceModelScore",
]
