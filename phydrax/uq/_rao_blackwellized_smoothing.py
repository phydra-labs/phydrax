#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from .._strict import StrictModule
from ..stochastic._state_space import state_space_key
from ._covariance import _solve_covariance_system
from ._particle import normalize_log_weights
from ._rao_blackwellized import (
    _condition_linear_state,
    RaoBlackwellizedFilterResult,
)


RAO_BLACKWELLIZED_SMOOTHER_SUCCESS = 0
RAO_BLACKWELLIZED_SMOOTHER_NONFINITE = 1


class RaoBlackwellizedBackwardSimulationResult(StrictModule):
    """Full-interval nonlinear paths with their particle-index provenance."""

    initial_nonlinear_states: Array
    nonlinear_paths: Array
    particle_indices: Array
    step_valid: Array
    valid: Array
    filter_result: RaoBlackwellizedFilterResult
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    ancestry_gradient: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    input_id: str | None = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


class RaoBlackwellizedSmootherResult(StrictModule):
    """Conditional RTS moments along full-interval nonlinear FFBSi paths."""

    linear_means: Array
    linear_covariances: Array
    gains: Array
    lag_one_covariances: Array
    valid: Array
    status: Array
    backward_simulation: RaoBlackwellizedBackwardSimulationResult
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    ancestry_gradient: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    input_id: str | None = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        step_valid = self.backward_simulation.step_valid
        prefix = (1,) * len(self.sample_shape)
        schedule = step_valid.reshape(prefix + step_valid.shape)
        return jnp.all(self.valid | ~schedule, axis=-1)


def _sample_shape(value: tuple[int, ...], /) -> tuple[tuple[int, ...], int]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("sample_shape dimensions must be positive.")
    return shape, prod(shape) if shape else 1


def rao_blackwellized_backward_simulation(
    key: Key[Array, ""],
    result: RaoBlackwellizedFilterResult,
    /,
    *,
    sample_shape: tuple[int, ...] = (),
) -> RaoBlackwellizedBackwardSimulationResult:
    """Draw nonlinear FFBSi paths using the complete normalized transition density."""
    if not isinstance(result, RaoBlackwellizedFilterResult):
        raise TypeError("result must be a RaoBlackwellizedFilterResult.")
    transition = result.problem.model.nonlinear_transition
    if not transition.has_log_density:
        raise ValueError(
            "Rao-Blackwellized backward simulation requires a normalized "
            "nonlinear transition density."
        )
    samples, sample_count = _sample_shape(sample_shape)
    case_shape = result.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = result.step_valid.shape[-1]
    count = result.num_particles
    nonlinear_shape = result.nonlinear_state_shape
    nonlinear_size = prod(nonlinear_shape) if nonlinear_shape else 1
    particles = result.nonlinear_particles.reshape(
        (case_count, num_steps, count, nonlinear_size)
    )
    log_weights = result.log_weights.reshape((case_count, num_steps, count))
    initial_particles = result.initial_nonlinear_particles.reshape(
        (case_count, count, nonlinear_size)
    )
    initial_weights = result.initial_log_weights.reshape((case_count, count))
    times = result.times.reshape((case_count, num_steps))
    initial_times = result.problem.initial_time.reshape((case_count,))
    active = result.step_valid.reshape((case_count, num_steps))
    filter_valid = result.valid.reshape((case_count, num_steps))
    paths = np.full((sample_count, case_count, num_steps, nonlinear_size), np.nan)
    initial_states = np.full((sample_count, case_count, nonlinear_size), np.nan)
    indices = np.full((sample_count, case_count, num_steps + 1), -1, dtype=np.int32)
    valid = np.zeros((sample_count, case_count), dtype=bool)
    members = jnp.arange(sample_count, dtype=jnp.uint32)
    steps = jnp.arange(num_steps, dtype=jnp.uint32)
    for case_index, case_id in enumerate(result.case_ids):
        active_count = int(np.sum(np.asarray(jax.device_get(active[case_index]))))
        if active_count == 0 or not bool(
            jnp.all(filter_valid[case_index, :active_count])
        ):
            continue
        terminal = active_count - 1
        terminal_weights, _, terminal_valid = normalize_log_weights(
            log_weights[case_index, terminal]
        )
        if not bool(terminal_valid):
            continue
        terminal_keys = jax.vmap(
            lambda member: state_space_key(
                key,
                "rao-blackwellized-backward-simulation",
                case_id,
                terminal,
                member=member,
            )
        )(members)
        draw_keys = jax.vmap(
            lambda member: jax.vmap(
                lambda step: state_space_key(
                    key,
                    "rao-blackwellized-backward-simulation",
                    case_id,
                    step,
                    member=member,
                )
            )(steps)
        )(members)
        initial_keys = jax.vmap(
            lambda member: state_space_key(
                key,
                "rao-blackwellized-backward-simulation-initial",
                case_id,
                0,
                member=member,
            )
        )(members)
        case_particles = particles[case_index]
        case_log_weights = log_weights[case_index]
        case_initial_particles = initial_particles[case_index]
        case_initial_weights = initial_weights[case_index]
        case_times = times[case_index]
        case_initial_time = initial_times[case_index]

        def draw_path(terminal_key, path_keys, initial_key):
            particle_index = jr.categorical(terminal_key, terminal_weights).astype(
                jnp.int32
            )
            path = jnp.full((num_steps, nonlinear_size), jnp.nan, dtype=float)
            path = path.at[terminal].set(case_particles[terminal, particle_index])
            path_indices = jnp.full((num_steps + 1,), -1, dtype=jnp.int32)
            path_indices = path_indices.at[terminal + 1].set(particle_index)
            path_is_valid = jnp.asarray(True)
            for step in range(terminal - 1, -1, -1):
                next_state = path[step + 1]
                context = result.problem.step_context(case_index, step + 1)
                density = jax.vmap(
                    lambda previous_particle: transition.log_prob(
                        next_state.reshape(nonlinear_shape),
                        previous_particle.reshape(nonlinear_shape),
                        case_times[step],
                        case_times[step + 1],
                        context,
                    ).reshape(())
                )(case_particles[step])
                backward_weights, _, weights_valid = normalize_log_weights(
                    case_log_weights[step] + density
                )
                particle_index = jr.categorical(path_keys[step], backward_weights).astype(
                    jnp.int32
                )
                path = path.at[step].set(case_particles[step, particle_index])
                path_indices = path_indices.at[step + 1].set(particle_index)
                path_is_valid = path_is_valid & weights_valid
            first_state = path[0]
            initial_context = result.problem.step_context(case_index, 0)
            initial_density = jax.vmap(
                lambda initial_particle: transition.log_prob(
                    first_state.reshape(nonlinear_shape),
                    initial_particle.reshape(nonlinear_shape),
                    case_initial_time,
                    case_times[0],
                    initial_context,
                ).reshape(())
            )(case_initial_particles)
            backward_initial, _, initial_valid = normalize_log_weights(
                case_initial_weights + initial_density
            )
            initial_index = jr.categorical(initial_key, backward_initial).astype(
                jnp.int32
            )
            path_indices = path_indices.at[0].set(initial_index)
            initial_state = case_initial_particles[initial_index]
            path_is_valid = path_is_valid & initial_valid
            return (
                jnp.where(path_is_valid, path, jnp.nan),
                jnp.where(path_is_valid, initial_state, jnp.nan),
                jnp.where(path_is_valid, path_indices, -1),
                path_is_valid,
            )

        case_paths, case_initial_states, case_indices, case_valid = jax.jit(
            jax.vmap(draw_path)
        )(terminal_keys, draw_keys, initial_keys)
        paths[:, case_index] = np.asarray(case_paths)
        initial_states[:, case_index] = np.asarray(case_initial_states)
        indices[:, case_index] = np.asarray(case_indices)
        valid[:, case_index] = np.asarray(case_valid)
    output_paths = jnp.asarray(paths).reshape(
        samples + case_shape + (num_steps,) + nonlinear_shape
    )
    output_initial = jnp.asarray(initial_states).reshape(
        samples + case_shape + nonlinear_shape
    )
    output_indices = jax.lax.stop_gradient(
        jnp.asarray(indices).reshape(samples + case_shape + (num_steps + 1,))
    )
    output_valid = jnp.asarray(valid).reshape(samples + case_shape)
    if not samples:
        output_paths = output_paths.reshape(case_shape + (num_steps,) + nonlinear_shape)
        output_initial = output_initial.reshape(case_shape + nonlinear_shape)
        output_indices = output_indices.reshape(case_shape + (num_steps + 1,))
        output_valid = output_valid.reshape(case_shape)
    problem = result.problem
    return RaoBlackwellizedBackwardSimulationResult(
        initial_nonlinear_states=output_initial,
        nonlinear_paths=output_paths,
        particle_indices=output_indices,
        step_valid=result.step_valid,
        valid=output_valid,
        filter_result=result,
        sample_shape=samples,
        method_id="rao-blackwellized-ffbsi",
        ancestry_gradient="stop",
        model_id=problem.model.model_id,
        problem_id=problem.problem_id,
        sequence_id=problem.observations.sequence_id,
        input_id=None if problem.input_signal is None else problem.input_signal.input_id,
        process_id=problem.model.nonlinear_transition.process_id,
        approximation_id=problem.model.nonlinear_transition.approximation_id,
    )


def sample_rao_blackwellized_backward_paths(
    key: Key[Array, ""],
    result: RaoBlackwellizedFilterResult,
    /,
    *,
    sample_shape: tuple[int, ...] = (),
) -> Array:
    """Draw full-interval nonlinear paths and return only observation-time states."""
    return rao_blackwellized_backward_simulation(
        key, result, sample_shape=sample_shape
    ).nonlinear_paths


def rao_blackwellized_particle_smoother(
    key: Key[Array, ""],
    result: RaoBlackwellizedFilterResult,
    /,
    *,
    sample_shape: tuple[int, ...] = (),
) -> RaoBlackwellizedSmootherResult:
    """Conditionally smooth the linear state along nonlinear FFBSi trajectories."""
    backward = rao_blackwellized_backward_simulation(
        key, result, sample_shape=sample_shape
    )
    samples = backward.sample_shape
    sample_count = prod(samples) if samples else 1
    case_shape = result.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = result.step_valid.shape[-1]
    nonlinear_shape = result.nonlinear_state_shape
    nonlinear_size = prod(nonlinear_shape) if nonlinear_shape else 1
    linear_shape = result.linear_state_shape
    linear_size = prod(linear_shape) if linear_shape else 1
    model = result.problem.model
    paths = backward.nonlinear_paths.reshape(
        (sample_count, case_count, num_steps, nonlinear_size)
    )
    initial_states = backward.initial_nonlinear_states.reshape(
        (sample_count, case_count, nonlinear_size)
    )
    path_valid = backward.valid.reshape((sample_count, case_count))
    active = result.step_valid.reshape((case_count, num_steps))
    times = result.times.reshape((case_count, num_steps))
    initial_times = result.problem.initial_time.reshape((case_count,))
    values = result.problem.observations.values.reshape(
        (case_count, num_steps) + model.observation_shape
    )
    masks = result.problem.observations.observation_mask.reshape(
        (case_count, num_steps) + model.observation_shape
    )
    smoothed_means = np.full((sample_count, case_count, num_steps, linear_size), np.nan)
    smoothed_covariances = np.full(
        (sample_count, case_count, num_steps, linear_size, linear_size), np.nan
    )
    gains = np.zeros(
        (sample_count, case_count, max(num_steps - 1, 0), linear_size, linear_size)
    )
    lag_one = np.zeros_like(gains)
    output_valid = np.zeros((sample_count, case_count, num_steps), dtype=bool)
    status = np.full(
        (sample_count, case_count, num_steps),
        RAO_BLACKWELLIZED_SMOOTHER_NONFINITE,
        dtype=np.int32,
    )
    for case_index in range(case_count):
        active_count = int(np.sum(np.asarray(jax.device_get(active[case_index]))))
        if active_count == 0:
            continue
        case_times = times[case_index]
        case_initial_time = initial_times[case_index]
        case_values = values[case_index]
        case_masks = masks[case_index]

        def smooth_path(nonlinear_path, initial_nonlinear, supplied_valid):
            mean, covariance = model.initial_linear_gaussian(
                initial_nonlinear.reshape(nonlinear_shape), result.problem.args
            )
            forecast_means = []
            forecast_covariances = []
            filtered_means = []
            filtered_covariances = []
            transitions = []
            step_validity = []
            previous_nonlinear = initial_nonlinear
            previous_time = case_initial_time
            for step in range(active_count):
                nonlinear = nonlinear_path[step]
                context = result.problem.step_context(case_index, step)
                transition_matrix, _, _ = model.linear_transition_parameters(
                    previous_nonlinear.reshape(nonlinear_shape),
                    nonlinear.reshape(nonlinear_shape),
                    previous_time,
                    case_times[step],
                    context,
                )
                conditioned = _condition_linear_state(
                    model,
                    previous_nonlinear.reshape(nonlinear_shape),
                    nonlinear.reshape(nonlinear_shape),
                    mean,
                    covariance,
                    previous_time,
                    case_times[step],
                    case_values[step],
                    case_masks[step],
                    context,
                )
                forecast_mean, forecast_covariance = conditioned[:2]
                filtered_mean, filtered_covariance, _, linear_valid = conditioned[2:]
                forecast_means.append(forecast_mean.reshape((linear_size,)))
                forecast_covariances.append(forecast_covariance)
                filtered_means.append(filtered_mean.reshape((linear_size,)))
                filtered_covariances.append(filtered_covariance)
                transitions.append(transition_matrix)
                step_validity.append(linear_valid)
                mean = filtered_mean
                covariance = filtered_covariance
                previous_nonlinear = nonlinear
                previous_time = case_times[step]
            forecast_means_array = jnp.stack(forecast_means)
            forecast_covariances_array = jnp.stack(forecast_covariances)
            filtered_means_array = jnp.stack(filtered_means)
            filtered_covariances_array = jnp.stack(filtered_covariances)
            transitions_array = jnp.stack(transitions)
            sample_is_valid = supplied_valid & jnp.all(jnp.stack(step_validity))
            means = filtered_means_array
            covariances = filtered_covariances_array
            path_gains = jnp.zeros(
                (max(active_count - 1, 0), linear_size, linear_size),
                dtype=means.dtype,
            )
            path_lag_one = jnp.zeros_like(path_gains)
            for step in range(active_count - 2, -1, -1):
                cross = filtered_covariances_array[step] @ transitions_array[step + 1].T
                solve_result = _solve_covariance_system(
                    forecast_covariances_array[step + 1],
                    cross.T,
                )
                gain = solve_result.value.T
                proposed_mean = filtered_means_array[step] + gain @ (
                    means[step + 1] - forecast_means_array[step + 1]
                )
                proposed_covariance = (
                    filtered_covariances_array[step]
                    + gain
                    @ (covariances[step + 1] - forecast_covariances_array[step + 1])
                    @ gain.T
                )
                proposed_covariance = 0.5 * (proposed_covariance + proposed_covariance.T)
                sample_is_valid = (
                    sample_is_valid
                    & jnp.all(solve_result.successful)
                    & jnp.all(jnp.isfinite(gain))
                    & jnp.all(jnp.isfinite(proposed_mean))
                    & jnp.all(jnp.isfinite(proposed_covariance))
                )
                means = means.at[step].set(proposed_mean)
                covariances = covariances.at[step].set(proposed_covariance)
                path_gains = path_gains.at[step].set(gain)
                path_lag_one = path_lag_one.at[step].set(covariances[step + 1] @ gain.T)
            full_means = (
                jnp.full((num_steps, linear_size), jnp.nan, dtype=means.dtype)
                .at[:active_count]
                .set(means)
            )
            full_covariances = (
                jnp.full(
                    (num_steps, linear_size, linear_size),
                    jnp.nan,
                    dtype=covariances.dtype,
                )
                .at[:active_count]
                .set(covariances)
            )
            full_gains = jnp.zeros(
                (max(num_steps - 1, 0), linear_size, linear_size),
                dtype=path_gains.dtype,
            )
            full_lag_one = jnp.zeros_like(full_gains)
            if active_count > 1:
                full_gains = full_gains.at[: active_count - 1].set(path_gains)
                full_lag_one = full_lag_one.at[: active_count - 1].set(path_lag_one)
            active_steps = jnp.arange(num_steps) < active_count
            valid_steps = active_steps & sample_is_valid
            path_status = jnp.where(
                valid_steps,
                RAO_BLACKWELLIZED_SMOOTHER_SUCCESS,
                RAO_BLACKWELLIZED_SMOOTHER_NONFINITE,
            )
            return (
                jnp.where(sample_is_valid, full_means, jnp.nan),
                jnp.where(sample_is_valid, full_covariances, jnp.nan),
                jnp.where(sample_is_valid, full_gains, 0.0),
                jnp.where(sample_is_valid, full_lag_one, 0.0),
                valid_steps,
                path_status,
            )

        (
            case_means,
            case_covariances,
            case_gains,
            case_lag_one,
            case_valid,
            case_status,
        ) = jax.jit(jax.vmap(smooth_path))(
            paths[:, case_index],
            initial_states[:, case_index],
            path_valid[:, case_index],
        )
        smoothed_means[:, case_index] = np.asarray(case_means)
        smoothed_covariances[:, case_index] = np.asarray(case_covariances)
        gains[:, case_index] = np.asarray(case_gains)
        lag_one[:, case_index] = np.asarray(case_lag_one)
        output_valid[:, case_index] = np.asarray(case_valid)
        status[:, case_index] = np.asarray(case_status)
    output_means = jnp.asarray(smoothed_means).reshape(
        samples + case_shape + (num_steps,) + linear_shape
    )
    output_covariances = jnp.asarray(smoothed_covariances).reshape(
        samples + case_shape + (num_steps, linear_size, linear_size)
    )
    output_gains = jnp.asarray(gains).reshape(
        samples + case_shape + (max(num_steps - 1, 0), linear_size, linear_size)
    )
    output_lag_one = jnp.asarray(lag_one).reshape(output_gains.shape)
    output_valid_array = jnp.asarray(output_valid).reshape(
        samples + case_shape + (num_steps,)
    )
    output_status = jnp.asarray(status).reshape(samples + case_shape + (num_steps,))
    if not samples:
        output_means = output_means.reshape(case_shape + (num_steps,) + linear_shape)
        output_covariances = output_covariances.reshape(
            case_shape + (num_steps, linear_size, linear_size)
        )
        output_gains = output_gains.reshape(
            case_shape + (max(num_steps - 1, 0), linear_size, linear_size)
        )
        output_lag_one = output_lag_one.reshape(output_gains.shape)
        output_valid_array = output_valid_array.reshape(case_shape + (num_steps,))
        output_status = output_status.reshape(case_shape + (num_steps,))
    return RaoBlackwellizedSmootherResult(
        linear_means=output_means,
        linear_covariances=output_covariances,
        gains=output_gains,
        lag_one_covariances=output_lag_one,
        valid=output_valid_array,
        status=output_status,
        backward_simulation=backward,
        sample_shape=samples,
        method_id="rao-blackwellized-ffbsi-rts",
        ancestry_gradient="stop",
        model_id=backward.model_id,
        problem_id=backward.problem_id,
        sequence_id=backward.sequence_id,
        input_id=backward.input_id,
        process_id=backward.process_id,
        approximation_id=backward.approximation_id,
    )


__all__ = [
    "RAO_BLACKWELLIZED_SMOOTHER_NONFINITE",
    "RAO_BLACKWELLIZED_SMOOTHER_SUCCESS",
    "RaoBlackwellizedBackwardSimulationResult",
    "RaoBlackwellizedSmootherResult",
    "rao_blackwellized_backward_simulation",
    "rao_blackwellized_particle_smoother",
    "sample_rao_blackwellized_backward_paths",
]
