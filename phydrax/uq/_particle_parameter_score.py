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
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._parameterized_state_space import ParameterizedStateSpaceProblem
from ._particle import ParticleFilterResult


class ParameterizedParticleGenealogicalScoreResult(StrictModule):
    """`O(TN)` likelihood score in unconstrained global-parameter coordinates."""

    gradient: PyTree[Array]
    flat_score: Array
    case_scores: Array
    valid: Array
    filter_result: ParticleFilterResult
    parameter_paths: tuple[str, ...] = eqx.field(static=True)
    parameter_size: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    ancestry_gradient: str = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)


def parameterized_particle_genealogical_score(
    parameterized: ParameterizedStateSpaceProblem,
    position: PyTree[Any],
    result: ParticleFilterResult,
    /,
) -> ParameterizedParticleGenealogicalScoreResult:
    """Propagate complete-model local scores in global parameter coordinates."""

    if not isinstance(parameterized, ParameterizedStateSpaceProblem):
        raise TypeError("parameterized must be a ParameterizedStateSpaceProblem.")
    if not isinstance(result, ParticleFilterResult):
        raise TypeError("result must be a ParticleFilterResult.")
    if result.problem_id != parameterized.problem.problem_id:
        raise ValueError("Particle result and parameterized problem IDs do not match.")
    flat_position, unravel = ravel_pytree(position)
    parameter_size = int(flat_position.size)
    if parameter_size < 1:
        raise ValueError("Parameterized particle scores require nonempty coordinates.")
    case_count = prod(result.case_shape) if result.case_shape else 1
    num_steps = result.problem.observations.num_steps
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
    times = result.problem.observations.times.reshape((case_count, num_steps))
    initial_times = result.problem.initial_time.reshape((case_count,))
    observations = result.problem.observations.values.reshape(
        (case_count, num_steps) + observation_shape
    )
    observation_masks = result.problem.observations.observation_mask.reshape(
        (case_count, num_steps) + observation_shape
    )
    initial_log_prob_function = parameterized.initial_log_prob_function

    def initial_vector(particle):
        if initial_log_prob_function is None:
            return jnp.zeros_like(flat_position)

        def initial_log_density(current_position):
            physical = parameterized.parameter_space.constrain(current_position)
            return jnp.sum(initial_log_prob_function(physical, particle))

        gradient = jax.grad(initial_log_density)(position)
        return ravel_pytree(gradient)[0]

    initial_scores = []
    for case_index in range(case_count):
        initial_scores.append(
            jnp.stack(
                [
                    initial_vector(initial_particles[case_index, particle_index])
                    for particle_index in range(count)
                ]
            )
        )
    cumulative = jnp.stack(initial_scores)
    terminal_scores = cumulative
    terminal_log_weights = initial_log_weights

    for step_index in range(num_steps):
        pre_resampling_cases = []
        post_resampling_cases = []
        for case_index in range(case_count):
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
                    def local_log_density(current_position):
                        physical = parameterized.parameter_space.constrain(
                            current_position
                        )
                        bound = parameterized.bind_physical(physical)
                        context = bound.step_context(case_index, step_index)
                        transition = bound.model.transition.log_prob(
                            next_state,
                            previous_state,
                            start_time,
                            end_time,
                            context,
                        )
                        observation = bound.model.observation.log_prob(
                            observations[case_index, step_index],
                            next_state,
                            end_time,
                            observation_masks[case_index, step_index],
                            context,
                        )
                        return jnp.asarray(transition + observation).reshape(())

                    gradient = jax.grad(local_log_density)(position)
                    return ravel_pytree(gradient)[0]

                local_scores.append(
                    jax.lax.cond(
                        active[case_index, step_index],
                        active_score,
                        lambda _: jnp.zeros_like(flat_position),
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

    weights = jnp.exp(terminal_log_weights)
    case_scores = jnp.sum(weights[..., None] * terminal_scores, axis=1)
    valid = result.successful.reshape((case_count,)) & result.initial_valid.reshape(
        (case_count,)
    )
    case_scores = jnp.where(valid[:, None], case_scores, 0.0)
    flat_score = jnp.sum(case_scores, axis=0)
    path_leaves = jax.tree_util.tree_flatten_with_path(position)[0]
    parameter_paths = tuple(
        jax.tree_util.keystr(path) or "<root>" for path, _ in path_leaves
    )
    return ParameterizedParticleGenealogicalScoreResult(
        gradient=unravel(flat_score),
        flat_score=flat_score,
        case_scores=case_scores.reshape(result.case_shape + (parameter_size,)),
        valid=valid.reshape(result.case_shape),
        filter_result=result,
        parameter_paths=parameter_paths,
        parameter_size=parameter_size,
        method_id="particle-parameter-genealogical-score",
        ancestry_gradient="stopped-realized-ancestry",
        parameterization_id=parameterized.parameterization_id,
    )


__all__ = [
    "parameterized_particle_genealogical_score",
    "ParameterizedParticleGenealogicalScoreResult",
]
