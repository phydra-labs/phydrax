#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key

from .._strict import StrictModule
from ..stochastic._state_space import (
    GaussianStatePrior,
    LinearGaussianObservationModel,
    LinearGaussianTransitionKernel,
    state_space_key,
    StateSpaceProblem,
)
from ._covariance import _factor_and_solve_covariance_system
from ._gaussian_chain import (
    associative_affine_values,
    associative_freeze,
    associative_gaussian_filter,
    associative_gaussian_smoother,
)


KalmanExecutionMethod: TypeAlias = Literal["sequential", "parallel", "auto"]
KalmanCovarianceForm: TypeAlias = Literal["covariance", "square_root"]
KalmanStatus: TypeAlias = Literal["success", "innovation_covariance_failure", "nonfinite"]
KALMAN_SUCCESS = 0
KALMAN_INNOVATION_COVARIANCE_FAILURE = 1
KALMAN_NONFINITE = 2


def kalman_status_name(value: int, /) -> KalmanStatus:
    code = int(value)
    if code == KALMAN_SUCCESS:
        return "success"
    if code == KALMAN_INNOVATION_COVARIANCE_FAILURE:
        return "innovation_covariance_failure"
    if code == KALMAN_NONFINITE:
        return "nonfinite"
    raise ValueError(f"Unknown Kalman status code {code}.")


def _linear_problem(
    problem: StateSpaceProblem,
) -> tuple[
    GaussianStatePrior, LinearGaussianTransitionKernel, LinearGaussianObservationModel
]:
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    prior = problem.model.prior
    transition = problem.model.transition
    observation = problem.model.observation
    if not isinstance(prior, GaussianStatePrior):
        raise TypeError("Kalman filtering requires GaussianStatePrior.")
    if not isinstance(transition, LinearGaussianTransitionKernel):
        raise TypeError("Kalman filtering requires LinearGaussianTransitionKernel.")
    if not isinstance(observation, LinearGaussianObservationModel):
        raise TypeError("Kalman filtering requires LinearGaussianObservationModel.")
    return prior, transition, observation


def _sizes(problem: StateSpaceProblem) -> tuple[int, int, int, tuple[int, ...]]:
    state_shape = problem.model.state_shape
    observation_shape = problem.model.observation_shape
    state_size = prod(state_shape) if state_shape else 1
    observation_size = prod(observation_shape) if observation_shape else 1
    case_shape = problem.observations.case_shape
    case_count = prod(case_shape) if case_shape else 1
    return state_size, observation_size, case_count, case_shape


def _transition_parameters(
    problem: StateSpaceProblem,
    starts: Array,
    ends: Array,
    step_indices: Array,
    /,
) -> tuple[Array, Array, Array]:
    _, kernel, _ = _linear_problem(problem)
    _, _, case_count, _ = _sizes(problem)
    flat_start = starts.reshape((-1,))
    flat_end = ends.reshape((-1,))
    flat_steps = jnp.broadcast_to(step_indices, starts.shape).reshape((-1,))
    repetitions = flat_start.shape[0] // case_count
    case_indices = jnp.tile(jnp.arange(case_count, dtype=jnp.int32), repetitions)
    return jax.vmap(
        lambda start, end, case_index, step_index: kernel.parameters(
            start,
            end,
            problem.step_context(case_index, step_index),
        )
    )(flat_start, flat_end, case_indices, flat_steps)


def _observation_parameters(
    problem: StateSpaceProblem,
    times: Array,
    step_indices: Array,
    /,
) -> tuple[Array, Array, Array]:
    _, _, model = _linear_problem(problem)
    _, _, case_count, _ = _sizes(problem)
    flat_times = times.reshape((-1,))
    flat_steps = jnp.broadcast_to(step_indices, times.shape).reshape((-1,))
    repetitions = flat_times.shape[0] // case_count
    case_indices = jnp.tile(jnp.arange(case_count, dtype=jnp.int32), repetitions)
    return jax.vmap(
        lambda time, case_index, step_index: model.parameters(
            time,
            problem.step_context(case_index, step_index),
        )
    )(flat_times, case_indices, flat_steps)


class KalmanFilterState(StrictModule):
    """Streaming Kalman state after the most recently processed observation step."""

    mean: Array
    covariance: Array
    time: Array
    log_likelihood: Array
    valid: Array
    status: Array
    step_index: Array
    problem_id: str = eqx.field(static=True)
    covariance_regularization: float = eqx.field(static=True)


class KalmanFilterStep(StrictModule):
    """One forecast/update record shared by streaming and batch execution."""

    predicted_mean: Array
    predicted_covariance: Array
    filtered_mean: Array
    filtered_covariance: Array
    transition_matrix: Array
    innovation: Array
    innovation_covariance: Array
    normalized_innovation_squared: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    observed_count: Array
    active: Array
    valid: Array
    status: Array


class KalmanFilterResult(StrictModule):
    """Complete masked Kalman history with exact model and schedule provenance."""

    predicted_means: Array
    predicted_covariances: Array
    filtered_means: Array
    filtered_covariances: Array
    transition_matrices: Array
    innovations: Array
    innovation_covariances: Array
    normalized_innovation_squared: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    observed_counts: Array
    step_valid: Array
    valid: Array
    status: Array
    final_state: KalmanFilterState
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    input_id: str | None = eqx.field(static=True)
    covariance_regularization: float = eqx.field(static=True)
    execution_method: str = eqx.field(static=True)
    covariance_form: KalmanCovarianceForm = eqx.field(static=True, default="covariance")

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


def initialize_kalman_filter(
    problem: StateSpaceProblem,
    /,
    *,
    covariance_regularization: float = 0.0,
) -> KalmanFilterState:
    prior, _, _ = _linear_problem(problem)
    regularization = float(covariance_regularization)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("covariance_regularization must be finite and nonnegative.")
    state_size, _, _, case_shape = _sizes(problem)
    mean = prior.mean
    covariance = jnp.broadcast_to(prior.covariance, case_shape + (state_size, state_size))
    valid = jnp.ones(case_shape, dtype=bool)
    return KalmanFilterState(
        mean=mean,
        covariance=covariance,
        time=problem.initial_time,
        log_likelihood=jnp.zeros(case_shape, dtype=mean.dtype),
        valid=valid,
        status=jnp.zeros(case_shape, dtype=jnp.int32),
        step_index=jnp.asarray(0, dtype=jnp.int32),
        problem_id=problem.problem_id,
        covariance_regularization=regularization,
    )


def kalman_filter_step(
    problem: StateSpaceProblem,
    state: KalmanFilterState,
    /,
) -> tuple[KalmanFilterState, KalmanFilterStep]:
    """Process exactly one observation step using fixed-size masked linear algebra."""
    _, transition, observation = _linear_problem(problem)
    if not isinstance(state, KalmanFilterState):
        raise TypeError("state must be a KalmanFilterState.")
    if state.problem_id != problem.problem_id:
        raise ValueError("Kalman state and problem IDs do not match.")
    sequence = problem.observations
    index = eqx.error_if(
        state.step_index,
        state.step_index >= sequence.num_steps,
        "The Kalman state has already consumed every observation step.",
    )
    state_size, observation_size, case_count, case_shape = _sizes(problem)
    active = sequence.step_valid[..., index]
    target_time = sequence.times[..., index]
    safe_time = jnp.where(active, target_time, state.time)
    matrices, offsets, process_covariances = _transition_parameters(
        problem,
        state.time,
        safe_time,
        jnp.broadcast_to(index, state.time.shape),
    )
    matrices = jnp.broadcast_to(matrices, (case_count, state_size, state_size))
    offsets = jnp.broadcast_to(offsets, (case_count, state_size))
    process_covariances = jnp.broadcast_to(
        process_covariances, (case_count, state_size, state_size)
    )
    previous_mean = state.mean.reshape((case_count, state_size))
    previous_covariance = state.covariance.reshape((case_count, state_size, state_size))
    forecast_mean = jnp.einsum("cij,cj->ci", matrices, previous_mean) + offsets
    forecast_covariance = (
        jnp.einsum("cij,cjk,clk->cil", matrices, previous_covariance, matrices)
        + process_covariances
    )
    forecast_covariance = 0.5 * (
        forecast_covariance + jnp.swapaxes(forecast_covariance, -1, -2)
    )
    active_flat = active.reshape((case_count,))
    forecast_mean = jnp.where(active_flat[:, None], forecast_mean, previous_mean)
    forecast_covariance = jnp.where(
        active_flat[:, None, None], forecast_covariance, previous_covariance
    )
    matrices = jnp.where(
        active_flat[:, None, None],
        matrices,
        jnp.eye(state_size, dtype=matrices.dtype)[None, ...],
    )

    observation_matrices, observation_offsets, observation_covariances = (
        _observation_parameters(
            problem,
            safe_time,
            jnp.broadcast_to(index, safe_time.shape),
        )
    )
    observation_matrices = jnp.broadcast_to(
        observation_matrices, (case_count, observation_size, state_size)
    )
    observation_offsets = jnp.broadcast_to(
        observation_offsets, (case_count, observation_size)
    )
    observation_covariances = jnp.broadcast_to(
        observation_covariances, (case_count, observation_size, observation_size)
    )
    step_axis = len(case_shape)
    values = jnp.take(sequence.values, index, axis=step_axis).reshape(
        (case_count, observation_size)
    )
    mask = jnp.take(sequence.observation_mask, index, axis=step_axis).reshape(
        (case_count, observation_size)
    ) & active.reshape((case_count, 1))
    active_float = mask.astype(forecast_mean.dtype)
    effective_matrix = observation_matrices * active_float[..., :, None]
    eye_observation = jnp.eye(observation_size, dtype=forecast_mean.dtype)
    effective_covariance = (
        observation_covariances * active_float[..., :, None] * active_float[..., None, :]
        + eye_observation[None, ...] * (1.0 - active_float[..., :, None])
        + state.covariance_regularization
        * eye_observation[None, ...]
        * active_float[..., :, None]
    )
    predicted_observation = (
        jnp.einsum("cij,cj->ci", observation_matrices, forecast_mean)
        + observation_offsets
    )
    innovation = jnp.where(mask, values - predicted_observation, 0.0)
    innovation_covariance = (
        jnp.einsum(
            "cij,cjk,clk->cil",
            effective_matrix,
            forecast_covariance,
            effective_matrix,
        )
        + effective_covariance
    )
    cross_covariance = jnp.einsum("cij,ckj->cik", forecast_covariance, effective_matrix)
    gain_result, scale = _factor_and_solve_covariance_system(
        innovation_covariance,
        jnp.swapaxes(cross_covariance, -1, -2),
    )
    diagonal = jnp.diagonal(scale, axis1=-2, axis2=-1)
    covariance_valid = (
        jnp.all(jnp.isfinite(scale), axis=(-1, -2))
        & jnp.all(diagonal > 0.0, axis=-1)
        & jnp.all(gain_result.successful, axis=-1)
    )
    gain = jnp.swapaxes(gain_result.value, -1, -2)
    updated_mean = forecast_mean + jnp.einsum("cij,cj->ci", gain, innovation)
    identity = jnp.eye(state_size, dtype=forecast_mean.dtype)
    update_operator = identity[None, ...] - jnp.einsum(
        "cij,cjk->cik", gain, effective_matrix
    )
    updated_covariance = jnp.einsum(
        "cij,cjk,clk->cil",
        update_operator,
        forecast_covariance,
        update_operator,
    ) + jnp.einsum("cij,cjk,clk->cil", gain, effective_covariance, gain)
    updated_covariance = 0.5 * (
        updated_covariance + jnp.swapaxes(updated_covariance, -1, -2)
    )
    solved_innovation = jax.scipy.linalg.solve_triangular(
        scale, innovation[..., None], lower=True
    )[..., 0]
    nis = jnp.sum(solved_innovation**2, axis=-1)
    observed_count = jnp.sum(mask, axis=-1)
    logdet = 2.0 * jnp.sum(jnp.log(diagonal), axis=-1)
    log_likelihood = -0.5 * (nis + logdet + observed_count * jnp.log(2.0 * jnp.pi))
    finite = (
        jnp.all(jnp.isfinite(updated_mean), axis=-1)
        & jnp.all(jnp.isfinite(updated_covariance), axis=(-1, -2))
        & jnp.isfinite(log_likelihood)
    )
    step_valid = state.valid.reshape((case_count,)) & covariance_valid & finite
    active_flat = active.reshape((case_count,))
    accepted = active_flat & step_valid
    next_mean = jnp.where(accepted[:, None], updated_mean, previous_mean)
    next_covariance = jnp.where(
        accepted[:, None, None], updated_covariance, previous_covariance
    )
    next_time = jnp.where(active, target_time, state.time)
    next_valid = state.valid & jnp.where(active, step_valid.reshape(case_shape), True)
    status = jnp.where(
        ~active_flat,
        KALMAN_SUCCESS,
        jnp.where(
            ~covariance_valid,
            KALMAN_INNOVATION_COVARIANCE_FAILURE,
            jnp.where(~finite, KALMAN_NONFINITE, KALMAN_SUCCESS),
        ),
    ).astype(jnp.int32)
    accepted_log_likelihood = jnp.where(accepted, log_likelihood, 0.0)
    cumulative = state.log_likelihood.reshape((case_count,)) + accepted_log_likelihood
    next_state = KalmanFilterState(
        mean=next_mean.reshape(case_shape + problem.model.state_shape),
        covariance=next_covariance.reshape(case_shape + (state_size, state_size)),
        time=next_time,
        log_likelihood=cumulative.reshape(case_shape),
        valid=next_valid,
        status=status.reshape(case_shape),
        step_index=index + 1,
        problem_id=problem.problem_id,
        covariance_regularization=state.covariance_regularization,
    )
    record = KalmanFilterStep(
        predicted_mean=forecast_mean.reshape(case_shape + problem.model.state_shape),
        predicted_covariance=forecast_covariance.reshape(
            case_shape + (state_size, state_size)
        ),
        filtered_mean=next_state.mean,
        filtered_covariance=next_state.covariance,
        transition_matrix=matrices.reshape(case_shape + (state_size, state_size)),
        innovation=innovation.reshape(case_shape + problem.model.observation_shape),
        innovation_covariance=innovation_covariance.reshape(
            case_shape + (observation_size, observation_size)
        ),
        normalized_innovation_squared=jnp.where(active_flat, nis, 0.0).reshape(
            case_shape
        ),
        incremental_log_likelihood=accepted_log_likelihood.reshape(case_shape),
        cumulative_log_likelihood=cumulative.reshape(case_shape),
        observed_count=observed_count.reshape(case_shape),
        active=active,
        valid=step_valid.reshape(case_shape),
        status=status.reshape(case_shape),
    )
    return next_state, record


def _time_to_result_axis(value: Array, case_rank: int, /) -> Array:
    return jnp.moveaxis(value, 0, case_rank)


def _sequential_kalman_filter(
    problem: StateSpaceProblem,
    /,
    *,
    covariance_regularization: float = 0.0,
    raise_on_failure: bool = False,
) -> KalmanFilterResult:
    """Run the exact linear-Gaussian filter through one fused temporal scan."""
    initial_state = initialize_kalman_filter(
        problem, covariance_regularization=covariance_regularization
    )

    def step(state: KalmanFilterState, _):
        return kalman_filter_step(problem, state)

    state, records = jax.lax.scan(
        step,
        initial_state,
        xs=None,
        length=problem.observations.num_steps,
    )
    rank = len(problem.observations.case_shape)
    result = KalmanFilterResult(
        predicted_means=_time_to_result_axis(records.predicted_mean, rank),
        predicted_covariances=_time_to_result_axis(records.predicted_covariance, rank),
        filtered_means=_time_to_result_axis(records.filtered_mean, rank),
        filtered_covariances=_time_to_result_axis(records.filtered_covariance, rank),
        transition_matrices=_time_to_result_axis(records.transition_matrix, rank),
        innovations=_time_to_result_axis(records.innovation, rank),
        innovation_covariances=_time_to_result_axis(records.innovation_covariance, rank),
        normalized_innovation_squared=_time_to_result_axis(
            records.normalized_innovation_squared, rank
        ),
        incremental_log_likelihood=_time_to_result_axis(
            records.incremental_log_likelihood, rank
        ),
        cumulative_log_likelihood=_time_to_result_axis(
            records.cumulative_log_likelihood, rank
        ),
        observed_counts=_time_to_result_axis(records.observed_count, rank),
        step_valid=problem.observations.step_valid,
        valid=_time_to_result_axis(records.valid, rank),
        status=_time_to_result_axis(records.status, rank),
        final_state=state,
        state_shape=problem.model.state_shape,
        observation_shape=problem.model.observation_shape,
        case_shape=problem.observations.case_shape,
        case_ids=problem.observations.case_ids,
        model_id=problem.model.model_id,
        problem_id=problem.problem_id,
        sequence_id=problem.observations.sequence_id,
        input_id=(
            None if problem.input_signal is None else problem.input_signal.input_id
        ),
        covariance_regularization=float(covariance_regularization),
        execution_method="sequential",
        covariance_form="covariance",
    )
    if raise_on_failure and not bool(jnp.all(result.successful)):
        raise RuntimeError("Kalman filtering failed for at least one physical case.")
    return result


def _resolve_execution_method(
    problem: StateSpaceProblem,
    method: KalmanExecutionMethod,
    /,
) -> str:
    if method not in ("sequential", "parallel", "auto"):
        raise ValueError("method must be 'sequential', 'parallel', or 'auto'.")
    if method != "auto":
        return method
    state_size, observation_size, _, _ = _sizes(problem)
    long_enough = problem.observations.num_steps >= 64
    modest_factors = state_size <= 32 and observation_size <= 32
    return "parallel" if long_enough and modest_factors else "sequential"


def _parallel_kalman_filter(
    problem: StateSpaceProblem,
    /,
    *,
    covariance_regularization: float,
    raise_on_failure: bool,
) -> KalmanFilterResult:
    prior, transition, observation = _linear_problem(problem)
    sequence = problem.observations
    state_size, observation_size, case_count, case_shape = _sizes(problem)
    num_steps = sequence.num_steps
    initial_times = problem.initial_time.reshape((case_count,))
    flat_times = sequence.times.reshape((case_count, num_steps))
    flat_active = sequence.step_valid.reshape((case_count, num_steps))
    active_time_major = jnp.swapaxes(flat_active, 0, 1)
    target_times = jnp.swapaxes(flat_times, 0, 1)
    safe_times = associative_freeze(initial_times, target_times, active_time_major)
    start_times = jnp.concatenate((initial_times[None, ...], safe_times[:-1]), axis=0)
    step_indices = jnp.broadcast_to(
        jnp.arange(num_steps, dtype=jnp.int32)[:, None],
        start_times.shape,
    )
    transition_array, offset_array, process_covariance_array = _transition_parameters(
        problem,
        start_times,
        safe_times,
        step_indices,
    )
    transition_array = transition_array.reshape(
        (num_steps, case_count, state_size, state_size)
    )
    offset_array = offset_array.reshape((num_steps, case_count, state_size))
    process_covariance_array = process_covariance_array.reshape(
        (num_steps, case_count, state_size, state_size)
    )
    transition_array = jnp.where(
        active_time_major[..., None, None],
        transition_array,
        jnp.eye(state_size, dtype=transition_array.dtype),
    )
    offset_array = jnp.where(active_time_major[..., None], offset_array, 0.0)
    process_covariance_array = jnp.where(
        active_time_major[..., None, None],
        process_covariance_array,
        0.0,
    )
    (
        observation_matrix_array,
        observation_offset_array,
        observation_covariance_array,
    ) = _observation_parameters(
        problem,
        safe_times,
        step_indices,
    )
    observation_matrix_array = observation_matrix_array.reshape(
        (num_steps, case_count, observation_size, state_size)
    )
    observation_offset_array = observation_offset_array.reshape(
        (num_steps, case_count, observation_size)
    )
    observation_covariance_array = observation_covariance_array.reshape(
        (num_steps, case_count, observation_size, observation_size)
    )
    current_times = safe_times[-1]
    observations = jnp.swapaxes(
        sequence.values.reshape((case_count, num_steps, observation_size)), 0, 1
    )
    masks = jnp.swapaxes(
        sequence.observation_mask.reshape((case_count, num_steps, observation_size)),
        0,
        1,
    )
    initial_mean = prior.mean.reshape((case_count, state_size))
    initial_covariance = prior.covariance.reshape((case_count, state_size, state_size))
    filtered_mean, filtered_covariance = associative_gaussian_filter(
        initial_mean,
        initial_covariance,
        transition_array,
        offset_array,
        process_covariance_array,
        observation_matrix_array,
        observation_offset_array,
        observation_covariance_array,
        observations,
        masks,
        covariance_regularization=covariance_regularization,
    )
    previous_mean = jnp.concatenate((initial_mean[None, ...], filtered_mean[:-1]))
    previous_covariance = jnp.concatenate(
        (initial_covariance[None, ...], filtered_covariance[:-1])
    )
    predicted_mean = (
        jnp.einsum("tcij,tcj->tci", transition_array, previous_mean) + offset_array
    )
    predicted_covariance = (
        jnp.einsum(
            "tcij,tcjk,tclk->tcil",
            transition_array,
            previous_covariance,
            transition_array,
        )
        + process_covariance_array
    )
    predicted_covariance = 0.5 * (
        predicted_covariance + jnp.swapaxes(predicted_covariance, -1, -2)
    )
    active_float = masks.astype(predicted_mean.dtype)
    effective_matrix = observation_matrix_array * active_float[..., :, None]
    observation_identity = jnp.eye(observation_size, dtype=predicted_mean.dtype)
    effective_covariance = (
        observation_covariance_array
        * active_float[..., :, None]
        * active_float[..., None, :]
        + observation_identity * (1.0 - active_float[..., :, None])
        + covariance_regularization * observation_identity * active_float[..., :, None]
    )
    predicted_observation = (
        jnp.einsum("tcij,tcj->tci", observation_matrix_array, predicted_mean)
        + observation_offset_array
    )
    innovation = jnp.where(masks, observations - predicted_observation, 0.0)
    innovation_covariance = (
        jnp.einsum(
            "tcij,tcjk,tclk->tcil",
            effective_matrix,
            predicted_covariance,
            effective_matrix,
        )
        + effective_covariance
    )
    scale = jnp.linalg.cholesky(innovation_covariance)
    diagonal = jnp.diagonal(scale, axis1=-2, axis2=-1)
    solved_innovation = jax.scipy.linalg.solve_triangular(
        scale, innovation[..., None], lower=True
    )[..., 0]
    raw_nis = jnp.sum(solved_innovation**2, axis=-1)
    observed_counts = jnp.sum(masks, axis=-1)
    raw_logdet = 2.0 * jnp.sum(jnp.log(diagonal), axis=-1)
    raw_increment = -0.5 * (
        raw_nis + raw_logdet + observed_counts * jnp.log(2.0 * jnp.pi)
    )
    raw_covariance_valid = jnp.all(jnp.isfinite(scale), axis=(-1, -2)) & jnp.all(
        diagonal > 0.0, axis=-1
    )
    raw_finite = (
        jnp.all(jnp.isfinite(filtered_mean), axis=-1)
        & jnp.all(jnp.isfinite(filtered_covariance), axis=(-1, -2))
        & jnp.isfinite(raw_increment)
    )
    raw_step_success = ~active_time_major | (raw_covariance_valid & raw_finite)
    raw_alive_after = jax.lax.associative_scan(jnp.logical_and, raw_step_success, axis=0)
    raw_alive_before = jnp.concatenate(
        (jnp.ones_like(raw_alive_after[:1]), raw_alive_after[:-1]), axis=0
    )
    raw_accepted = (
        active_time_major & raw_alive_before & raw_covariance_valid & raw_finite
    )
    frozen_mean = associative_freeze(initial_mean, filtered_mean, raw_accepted)
    frozen_covariance = associative_freeze(
        initial_covariance, filtered_covariance, raw_accepted
    )

    previous_mean = jnp.concatenate((initial_mean[None, ...], frozen_mean[:-1]))
    previous_covariance = jnp.concatenate(
        (initial_covariance[None, ...], frozen_covariance[:-1])
    )
    predicted_mean = (
        jnp.einsum("tcij,tcj->tci", transition_array, previous_mean) + offset_array
    )
    predicted_covariance = (
        jnp.einsum(
            "tcij,tcjk,tclk->tcil",
            transition_array,
            previous_covariance,
            transition_array,
        )
        + process_covariance_array
    )
    predicted_covariance = 0.5 * (
        predicted_covariance + jnp.swapaxes(predicted_covariance, -1, -2)
    )
    predicted_observation = (
        jnp.einsum("tcij,tcj->tci", observation_matrix_array, predicted_mean)
        + observation_offset_array
    )
    innovation = jnp.where(masks, observations - predicted_observation, 0.0)
    innovation_covariance = (
        jnp.einsum(
            "tcij,tcjk,tclk->tcil",
            effective_matrix,
            predicted_covariance,
            effective_matrix,
        )
        + effective_covariance
    )
    cross_covariance = predicted_covariance @ jnp.swapaxes(effective_matrix, -1, -2)
    gain_result, scale = _factor_and_solve_covariance_system(
        innovation_covariance,
        jnp.swapaxes(cross_covariance, -1, -2),
    )
    diagonal = jnp.diagonal(scale, axis1=-2, axis2=-1)
    covariance_valid = (
        jnp.all(jnp.isfinite(scale), axis=(-1, -2))
        & jnp.all(diagonal > 0.0, axis=-1)
        & jnp.all(gain_result.successful, axis=-1)
    )
    gain = jnp.swapaxes(gain_result.value, -1, -2)
    updated_mean = predicted_mean + jnp.einsum("tcij,tcj->tci", gain, innovation)
    identity = jnp.eye(state_size, dtype=predicted_mean.dtype)
    update_operator = identity - gain @ effective_matrix
    updated_covariance = update_operator @ predicted_covariance @ jnp.swapaxes(
        update_operator, -1, -2
    ) + gain @ effective_covariance @ jnp.swapaxes(gain, -1, -2)
    updated_covariance = 0.5 * (
        updated_covariance + jnp.swapaxes(updated_covariance, -1, -2)
    )
    solved_innovation = jax.scipy.linalg.solve_triangular(
        scale, innovation[..., None], lower=True
    )[..., 0]
    normalized_innovation_squared = jnp.sum(solved_innovation**2, axis=-1)
    logdet = 2.0 * jnp.sum(jnp.log(diagonal), axis=-1)
    log_likelihood = -0.5 * (
        normalized_innovation_squared + logdet + observed_counts * jnp.log(2.0 * jnp.pi)
    )
    finite = (
        jnp.all(jnp.isfinite(updated_mean), axis=-1)
        & jnp.all(jnp.isfinite(updated_covariance), axis=(-1, -2))
        & jnp.isfinite(log_likelihood)
    )
    step_success = ~active_time_major | (covariance_valid & finite)
    alive_after = jax.lax.associative_scan(jnp.logical_and, step_success, axis=0)
    alive_before = jnp.concatenate(
        (jnp.ones_like(alive_after[:1]), alive_after[:-1]), axis=0
    )
    valid = alive_before & covariance_valid & finite
    accepted = active_time_major & valid
    filtered_mean = associative_freeze(initial_mean, updated_mean, accepted)
    filtered_covariance = associative_freeze(
        initial_covariance, updated_covariance, accepted
    )
    increments = jnp.where(accepted, log_likelihood, 0.0)
    cumulative = jnp.cumsum(increments, axis=0)
    status = jnp.where(
        ~active_time_major,
        KALMAN_SUCCESS,
        jnp.where(
            ~covariance_valid,
            KALMAN_INNOVATION_COVARIANCE_FAILURE,
            jnp.where(~finite, KALMAN_NONFINITE, KALMAN_SUCCESS),
        ),
    ).astype(jnp.int32)

    def restore(values: Array, trailing_shape: tuple[int, ...] = ()) -> Array:
        return jnp.swapaxes(values, 0, 1).reshape(
            case_shape + (num_steps,) + trailing_shape
        )

    result_filtered_mean = restore(filtered_mean, problem.model.state_shape)
    result_filtered_covariance = restore(filtered_covariance, (state_size, state_size))
    final_state = KalmanFilterState(
        mean=jnp.take(result_filtered_mean, -1, axis=len(case_shape)),
        covariance=result_filtered_covariance[..., -1, :, :],
        time=current_times.reshape(case_shape),
        log_likelihood=restore(cumulative)[..., -1],
        valid=alive_after[-1].reshape(case_shape),
        status=status[-1].reshape(case_shape),
        step_index=jnp.asarray(num_steps, dtype=jnp.int32),
        problem_id=problem.problem_id,
        covariance_regularization=covariance_regularization,
    )
    result = KalmanFilterResult(
        predicted_means=restore(predicted_mean, problem.model.state_shape),
        predicted_covariances=restore(predicted_covariance, (state_size, state_size)),
        filtered_means=result_filtered_mean,
        filtered_covariances=result_filtered_covariance,
        transition_matrices=restore(transition_array, (state_size, state_size)),
        innovations=restore(innovation, problem.model.observation_shape),
        innovation_covariances=restore(
            innovation_covariance, (observation_size, observation_size)
        ),
        normalized_innovation_squared=restore(
            jnp.where(active_time_major, normalized_innovation_squared, 0.0)
        ),
        incremental_log_likelihood=restore(increments),
        cumulative_log_likelihood=restore(cumulative),
        observed_counts=restore(observed_counts),
        step_valid=sequence.step_valid,
        valid=restore(valid),
        status=restore(status),
        final_state=final_state,
        state_shape=problem.model.state_shape,
        observation_shape=problem.model.observation_shape,
        case_shape=case_shape,
        case_ids=sequence.case_ids,
        model_id=problem.model.model_id,
        problem_id=problem.problem_id,
        sequence_id=sequence.sequence_id,
        input_id=(
            None if problem.input_signal is None else problem.input_signal.input_id
        ),
        covariance_regularization=covariance_regularization,
        execution_method="parallel",
        covariance_form="covariance",
    )
    if raise_on_failure and not bool(jnp.all(result.successful)):
        raise RuntimeError("Kalman filtering failed for at least one physical case.")
    return result


def kalman_filter(
    problem: StateSpaceProblem,
    /,
    *,
    covariance_regularization: float = 0.0,
    raise_on_failure: bool = False,
    method: KalmanExecutionMethod = "auto",
    covariance_form: KalmanCovarianceForm = "covariance",
) -> KalmanFilterResult:
    """Run an exact linear-Gaussian filter with explicit covariance storage."""
    regularization = float(covariance_regularization)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("covariance_regularization must be finite and nonnegative.")
    if covariance_form not in ("covariance", "square_root"):
        raise ValueError("covariance_form must be 'covariance' or 'square_root'.")
    if covariance_form == "square_root":
        if method not in ("sequential", "parallel", "auto"):
            raise ValueError("method must be 'sequential', 'parallel', or 'auto'.")
        if method == "parallel":
            raise ValueError(
                "Square-root Kalman filtering does not support method='parallel'."
            )
        from ._square_root import _square_root_kalman_filter

        return _square_root_kalman_filter(
            problem,
            covariance_regularization=regularization,
            raise_on_failure=raise_on_failure,
        )
    resolved = _resolve_execution_method(problem, method)
    if resolved == "sequential":
        return _sequential_kalman_filter(
            problem,
            covariance_regularization=regularization,
            raise_on_failure=raise_on_failure,
        )
    return _parallel_kalman_filter(
        problem,
        covariance_regularization=regularization,
        raise_on_failure=raise_on_failure,
    )


class KalmanSmootherResult(StrictModule):
    """Rauch--Tung--Striebel smoothed marginals and backward gains."""

    means: Array
    covariances: Array
    gains: Array
    valid: Array
    filter_result: KalmanFilterResult
    execution_method: str = eqx.field(static=True)
    covariance_form: KalmanCovarianceForm = eqx.field(static=True, default="covariance")


def _psd_pseudoinverse(covariance: Array, /) -> Array:
    hermitian = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
    values, vectors = jnp.linalg.eigh(hermitian)
    inverse = jnp.where(values > 0.0, 1.0 / values, 0.0)
    return jnp.einsum("...ij,...j,...kj->...ik", vectors, inverse, vectors)


def _sequential_rts_smoother(result: KalmanFilterResult, /) -> KalmanSmootherResult:
    """Apply the exact RTS backward recursion to a Kalman filter history."""
    if not isinstance(result, KalmanFilterResult):
        raise TypeError("result must be a KalmanFilterResult.")
    case_shape = result.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = int(result.filtered_means.shape[len(case_shape)])
    state_size = prod(result.state_shape) if result.state_shape else 1
    filtered_mean = result.filtered_means.reshape((case_count, num_steps, state_size))
    filtered_covariance = result.filtered_covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    predicted_mean = result.predicted_means.reshape((case_count, num_steps, state_size))
    predicted_covariance = result.predicted_covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    transitions = result.transition_matrices.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    active = result.step_valid.reshape((case_count, num_steps))
    valid = result.valid.reshape((case_count, num_steps)) & active
    means = filtered_mean
    covariances = filtered_covariance
    gains = jnp.zeros((case_count, max(num_steps - 1, 0), state_size, state_size))
    for index in range(num_steps - 2, -1, -1):
        cross = jnp.einsum(
            "cij,ckj->cik", filtered_covariance[:, index], transitions[:, index + 1]
        )
        gain = jnp.einsum(
            "cij,cjk->cik",
            cross,
            _psd_pseudoinverse(predicted_covariance[:, index + 1]),
        )
        pair_valid = valid[:, index] & valid[:, index + 1]
        proposed_mean = filtered_mean[:, index] + jnp.einsum(
            "cij,cj->ci", gain, means[:, index + 1] - predicted_mean[:, index + 1]
        )
        proposed_covariance = filtered_covariance[:, index] + jnp.einsum(
            "cij,cjk,clk->cil",
            gain,
            covariances[:, index + 1] - predicted_covariance[:, index + 1],
            gain,
        )
        proposed_covariance = 0.5 * (
            proposed_covariance + jnp.swapaxes(proposed_covariance, -1, -2)
        )
        means = means.at[:, index].set(
            jnp.where(pair_valid[:, None], proposed_mean, means[:, index])
        )
        covariances = covariances.at[:, index].set(
            jnp.where(
                pair_valid[:, None, None], proposed_covariance, covariances[:, index]
            )
        )
        gains = gains.at[:, index].set(jnp.where(pair_valid[:, None, None], gain, 0.0))
    return KalmanSmootherResult(
        means=means.reshape(case_shape + (num_steps,) + result.state_shape),
        covariances=covariances.reshape(case_shape + (num_steps, state_size, state_size)),
        gains=gains.reshape(case_shape + (max(num_steps - 1, 0), state_size, state_size)),
        valid=valid.reshape(case_shape + (num_steps,)),
        filter_result=result,
        execution_method="sequential",
        covariance_form="covariance",
    )


def _parallel_rts_smoother(result: KalmanFilterResult, /) -> KalmanSmootherResult:
    if not isinstance(result, KalmanFilterResult):
        raise TypeError("result must be a KalmanFilterResult.")
    case_shape = result.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = int(result.filtered_means.shape[len(case_shape)])
    state_size = prod(result.state_shape) if result.state_shape else 1
    filtered_means = jnp.swapaxes(
        result.filtered_means.reshape((case_count, num_steps, state_size)), 0, 1
    )
    filtered_covariances = jnp.swapaxes(
        result.filtered_covariances.reshape(
            (case_count, num_steps, state_size, state_size)
        ),
        0,
        1,
    )
    predicted_means = jnp.swapaxes(
        result.predicted_means.reshape((case_count, num_steps, state_size)), 0, 1
    )
    predicted_covariances = jnp.swapaxes(
        result.predicted_covariances.reshape(
            (case_count, num_steps, state_size, state_size)
        ),
        0,
        1,
    )
    transitions = jnp.swapaxes(
        result.transition_matrices.reshape(
            (case_count, num_steps, state_size, state_size)
        ),
        0,
        1,
    )
    valid = jnp.swapaxes(
        (result.valid & result.step_valid).reshape((case_count, num_steps)), 0, 1
    )
    means, covariances, gains = associative_gaussian_smoother(
        filtered_means,
        filtered_covariances,
        predicted_means,
        predicted_covariances,
        transitions,
        valid,
    )
    return KalmanSmootherResult(
        means=jnp.swapaxes(means, 0, 1).reshape(
            case_shape + (num_steps,) + result.state_shape
        ),
        covariances=jnp.swapaxes(covariances, 0, 1).reshape(
            case_shape + (num_steps, state_size, state_size)
        ),
        gains=jnp.swapaxes(gains, 0, 1).reshape(
            case_shape + (max(num_steps - 1, 0), state_size, state_size)
        ),
        valid=jnp.swapaxes(valid, 0, 1).reshape(case_shape + (num_steps,)),
        filter_result=result,
        execution_method="parallel",
        covariance_form="covariance",
    )


def rts_smoother(
    result: KalmanFilterResult,
    /,
    *,
    method: KalmanExecutionMethod = "auto",
    covariance_form: KalmanCovarianceForm = "covariance",
) -> KalmanSmootherResult:
    """Apply the RTS recursion with explicit execution and covariance provenance."""
    if not isinstance(result, KalmanFilterResult):
        raise TypeError("result must be a KalmanFilterResult.")
    if method not in ("sequential", "parallel", "auto"):
        raise ValueError("method must be 'sequential', 'parallel', or 'auto'.")
    if covariance_form not in ("covariance", "square_root"):
        raise ValueError("covariance_form must be 'covariance' or 'square_root'.")
    if covariance_form == "square_root":
        if method == "parallel":
            raise ValueError(
                "Square-root RTS smoothing does not support method='parallel'."
            )
        from ._square_root import _square_root_rts_smoother

        return _square_root_rts_smoother(result)
    if method == "auto":
        resolved = (
            "parallel"
            if result.execution_method == "parallel" or result.step_valid.shape[-1] >= 64
            else "sequential"
        )
    else:
        resolved = method
    if resolved == "sequential":
        return _sequential_rts_smoother(result)
    return _parallel_rts_smoother(result)


def _gaussian_draw(key: Array, mean: Array, covariance: Array, /) -> Array:
    values, vectors = jnp.linalg.eigh(covariance)
    factor = vectors * jnp.sqrt(jnp.maximum(values, 0.0))[None, :]
    return mean + factor @ jax.random.normal(key, mean.shape, dtype=mean.dtype)


def _sequential_sample_kalman_smoother_paths(
    key: Key[Array, ""],
    smoother: KalmanSmootherResult,
    /,
    *,
    sample_shape: tuple[int, ...] = (),
) -> Array:
    """Sample coherent backward conditional paths, never independent time marginals."""
    if not isinstance(smoother, KalmanSmootherResult):
        raise TypeError("smoother must be a KalmanSmootherResult.")
    samples = tuple(int(size) for size in sample_shape)
    if any(size <= 0 for size in samples):
        raise ValueError("sample_shape dimensions must be positive.")
    sample_count = prod(samples) if samples else 1
    result = smoother.filter_result
    case_count = prod(result.case_shape) if result.case_shape else 1
    num_steps = result.filtered_means.shape[len(result.case_shape)]
    state_size = prod(result.state_shape) if result.state_shape else 1
    filtered_mean = result.filtered_means.reshape((case_count, num_steps, state_size))
    filtered_covariance = result.filtered_covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    predicted_mean = result.predicted_means.reshape((case_count, num_steps, state_size))
    predicted_covariance = result.predicted_covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    smooth_mean = smoother.means.reshape((case_count, num_steps, state_size))
    smooth_covariance = smoother.covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    gains = smoother.gains.reshape(
        (case_count, max(num_steps - 1, 0), state_size, state_size)
    )
    active = result.step_valid.reshape((case_count, num_steps))
    paths = np.zeros((sample_count, case_count, num_steps, state_size), dtype=np.float64)
    for sample_index in range(sample_count):
        for case_index in range(case_count):
            valid_count = int(np.sum(np.asarray(active[case_index])))
            if valid_count == 0:
                continue
            terminal = valid_count - 1
            case_id = result.case_ids[case_index]
            terminal_key = state_space_key(
                key, "kalman-smoother", case_id, terminal, member=sample_index
            )
            terminal_value = _gaussian_draw(
                terminal_key,
                smooth_mean[case_index, terminal],
                smooth_covariance[case_index, terminal],
            )
            path = jnp.zeros((num_steps, state_size), dtype=smooth_mean.dtype)
            path = path.at[terminal].set(terminal_value)
            for index in range(terminal - 1, -1, -1):
                gain = gains[case_index, index]
                conditional_mean = filtered_mean[case_index, index] + gain @ (
                    path[index + 1] - predicted_mean[case_index, index + 1]
                )
                conditional_covariance = filtered_covariance[case_index, index] - (
                    gain @ predicted_covariance[case_index, index + 1] @ gain.T
                )
                draw_key = state_space_key(
                    key, "kalman-smoother", case_id, index, member=sample_index
                )
                path = path.at[index].set(
                    _gaussian_draw(draw_key, conditional_mean, conditional_covariance)
                )
            if valid_count < num_steps:
                path = path.at[valid_count:].set(path[terminal])
            paths[sample_index, case_index] = np.asarray(path)
    output = jnp.asarray(paths).reshape(
        samples + result.case_shape + (num_steps,) + result.state_shape
    )
    if samples:
        return output
    return output.reshape(result.case_shape + (num_steps,) + result.state_shape)


def _parallel_sample_kalman_smoother_paths(
    key: Key[Array, ""],
    smoother: KalmanSmootherResult,
    /,
    *,
    sample_shape: tuple[int, ...],
) -> Array:
    if not isinstance(smoother, KalmanSmootherResult):
        raise TypeError("smoother must be a KalmanSmootherResult.")
    samples = tuple(int(size) for size in sample_shape)
    if any(size <= 0 for size in samples):
        raise ValueError("sample_shape dimensions must be positive.")
    sample_count = prod(samples) if samples else 1
    result = smoother.filter_result
    case_count = prod(result.case_shape) if result.case_shape else 1
    num_steps = result.filtered_means.shape[len(result.case_shape)]
    state_size = prod(result.state_shape) if result.state_shape else 1
    filtered_mean = result.filtered_means.reshape((case_count, num_steps, state_size))
    filtered_covariance = result.filtered_covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    predicted_mean = result.predicted_means.reshape((case_count, num_steps, state_size))
    predicted_covariance = result.predicted_covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    smooth_mean = smoother.means.reshape((case_count, num_steps, state_size))
    smooth_covariance = smoother.covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    gains = smoother.gains.reshape(
        (case_count, max(num_steps - 1, 0), state_size, state_size)
    )
    active = result.step_valid.reshape((case_count, num_steps))
    terminal = jnp.sum(active, axis=-1, dtype=jnp.int32) - 1
    steps = jnp.arange(num_steps, dtype=jnp.uint32)
    members = jnp.arange(sample_count, dtype=jnp.uint32)

    def case_keys(case_id: str) -> Array:
        return jax.vmap(
            lambda member: jax.vmap(
                lambda step: state_space_key(
                    key,
                    "kalman-smoother",
                    case_id,
                    step,
                    member=member,
                )
            )(steps)
        )(members)

    keys = jnp.stack(tuple(case_keys(case_id) for case_id in result.case_ids), axis=1)
    terminal_mean = jax.vmap(lambda values, index: values[index])(smooth_mean, terminal)
    terminal_covariance = jax.vmap(lambda values, index: values[index])(
        smooth_covariance, terminal
    )
    terminal_keys = jax.vmap(
        lambda sample_keys: jax.vmap(lambda case_key, index: case_key[index])(
            sample_keys, terminal
        )
    )(keys)
    terminal_values = jax.vmap(
        lambda sample_keys: jax.vmap(_gaussian_draw)(
            sample_keys, terminal_mean, terminal_covariance
        )
    )(terminal_keys)

    conditional_covariance = filtered_covariance[:, :-1] - (
        gains @ predicted_covariance[:, 1:] @ jnp.swapaxes(gains, -1, -2)
    )
    conditional_covariance = 0.5 * (
        conditional_covariance + jnp.swapaxes(conditional_covariance, -1, -2)
    )
    conditional_offset = filtered_mean[:, :-1] - jnp.einsum(
        "ctij,ctj->cti", gains, predicted_mean[:, 1:]
    )
    gains = jnp.concatenate(
        (
            gains,
            jnp.zeros(
                (case_count, 1, state_size, state_size),
                dtype=filtered_mean.dtype,
            ),
        ),
        axis=1,
    )
    conditional_covariance = jnp.concatenate(
        (
            conditional_covariance,
            jnp.zeros(
                (case_count, 1, state_size, state_size),
                dtype=filtered_mean.dtype,
            ),
        ),
        axis=1,
    )
    conditional_offset = jnp.concatenate(
        (
            conditional_offset,
            jnp.zeros((case_count, 1, state_size), dtype=filtered_mean.dtype),
        ),
        axis=1,
    )
    zero_mean = jnp.zeros((state_size,), dtype=filtered_mean.dtype)
    noise = jax.vmap(
        lambda sample_keys: jax.vmap(
            lambda case_key, covariance: jax.vmap(
                lambda draw_key, draw_covariance: _gaussian_draw(
                    draw_key, zero_mean, draw_covariance
                )
            )(case_key, covariance)
        )(sample_keys, conditional_covariance)
    )(keys)
    before_terminal = jnp.arange(num_steps)[None, :] < terminal[:, None]
    transitions = jnp.where(
        before_terminal[None, ..., None, None],
        gains[None, ...],
        0.0,
    )
    transitions = jnp.broadcast_to(
        transitions,
        (sample_count, case_count, num_steps, state_size, state_size),
    )
    offsets = jnp.where(
        before_terminal[None, ..., None],
        conditional_offset[None, ...] + noise,
        terminal_values[:, :, None, :],
    )
    reverse_paths = associative_affine_values(
        jnp.moveaxis(transitions[:, :, ::-1], 2, 0),
        jnp.moveaxis(offsets[:, :, ::-1], 2, 0),
    )
    paths = jnp.moveaxis(reverse_paths[::-1], 0, 2)
    output = paths.reshape(
        samples + result.case_shape + (num_steps,) + result.state_shape
    )
    if samples:
        return output
    return output.reshape(result.case_shape + (num_steps,) + result.state_shape)


def sample_kalman_smoother_paths(
    key: Key[Array, ""],
    smoother: KalmanSmootherResult,
    /,
    *,
    sample_shape: tuple[int, ...] = (),
    method: KalmanExecutionMethod = "auto",
) -> Array:
    """Sample coherent keyed paths with explicit temporal execution."""
    if method not in ("sequential", "parallel", "auto"):
        raise ValueError("method must be 'sequential', 'parallel', or 'auto'.")
    resolved = smoother.execution_method if method == "auto" else method
    if resolved == "sequential":
        return _sequential_sample_kalman_smoother_paths(
            key, smoother, sample_shape=sample_shape
        )
    return _parallel_sample_kalman_smoother_paths(
        key, smoother, sample_shape=sample_shape
    )


class KalmanInnovationDiagnostics(StrictModule):
    """Finite-value, PSD, NIS, and lag-one innovation diagnostics."""

    mean_normalized_innovation_squared: Array
    innovation_lag_one_correlation: Array
    minimum_filtered_covariance_eigenvalue: Array
    valid_steps: Array
    finite: Array

    @property
    def passed(self) -> bool:
        return bool(jnp.all(self.finite)) and bool(
            jnp.all(self.minimum_filtered_covariance_eigenvalue >= -1e-10)
        )


def kalman_innovation_diagnostics(
    result: KalmanFilterResult,
    /,
) -> KalmanInnovationDiagnostics:
    if not isinstance(result, KalmanFilterResult):
        raise TypeError("result must be a KalmanFilterResult.")
    case_shape = result.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = result.step_valid.shape[-1]
    observation_size = prod(result.observation_shape) if result.observation_shape else 1
    valid = (result.step_valid & result.valid).reshape((case_count, num_steps))
    counts = jnp.sum(valid, axis=-1)
    nis = result.normalized_innovation_squared.reshape((case_count, num_steps))
    mean_nis = jnp.sum(jnp.where(valid, nis, 0.0), axis=-1) / jnp.maximum(counts, 1)
    innovations = result.innovations.reshape((case_count, num_steps, observation_size))
    left_valid = valid[:, :-1] & valid[:, 1:]
    numerator = jnp.sum(
        jnp.where(left_valid[..., None], innovations[:, :-1] * innovations[:, 1:], 0.0),
        axis=1,
    )
    denominator = jnp.sqrt(
        jnp.sum(jnp.where(left_valid[..., None], innovations[:, :-1] ** 2, 0.0), axis=1)
        * jnp.sum(jnp.where(left_valid[..., None], innovations[:, 1:] ** 2, 0.0), axis=1)
    )
    correlation = jnp.where(denominator > 0.0, numerator / denominator, 0.0)
    eigenvalues = jnp.linalg.eigvalsh(result.filtered_covariances)
    state_axes = tuple(range(len(case_shape) + 1, eigenvalues.ndim))
    minimum = jnp.min(eigenvalues, axis=state_axes)
    mean_axes = tuple(range(len(case_shape) + 1, result.filtered_means.ndim))
    covariance_axes = tuple(range(len(case_shape) + 1, result.filtered_covariances.ndim))
    finite = jnp.all(jnp.isfinite(result.filtered_means), axis=mean_axes) & jnp.all(
        jnp.isfinite(result.filtered_covariances), axis=covariance_axes
    )
    return KalmanInnovationDiagnostics(
        mean_normalized_innovation_squared=mean_nis.reshape(case_shape),
        innovation_lag_one_correlation=correlation.reshape(
            case_shape + result.observation_shape
        ),
        minimum_filtered_covariance_eigenvalue=minimum,
        valid_steps=counts.reshape(case_shape),
        finite=finite,
    )


__all__ = [
    "KalmanCovarianceForm",
    "initialize_kalman_filter",
    "KALMAN_INNOVATION_COVARIANCE_FAILURE",
    "KALMAN_NONFINITE",
    "KALMAN_SUCCESS",
    "kalman_filter",
    "KalmanExecutionMethod",
    "KalmanFilterResult",
    "KalmanFilterState",
    "KalmanFilterStep",
    "kalman_filter_step",
    "kalman_innovation_diagnostics",
    "KalmanInnovationDiagnostics",
    "kalman_status_name",
    "KalmanStatus",
    "KalmanSmootherResult",
    "rts_smoother",
    "sample_kalman_smoother_paths",
]
