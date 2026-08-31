#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ._gaussian_factor import gaussian_factor_from_covariance, GaussianFactor
from ._kalman import (
    _linear_problem,
    _observation_parameters,
    _sizes,
    _time_to_result_axis,
    _transition_parameters,
    KALMAN_INNOVATION_COVARIANCE_FAILURE,
    KALMAN_NONFINITE,
    KALMAN_SUCCESS,
    KalmanFilterResult,
    KalmanFilterState,
    KalmanSmootherResult,
)


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def _covariance(factor: Array, /) -> Array:
    return factor @ _adjoint(factor)


def _covariance_assembly_roundoff(covariance: Array, /) -> Array:
    """Bound backward error from assembling an algebraically PSD block matrix."""
    real_dtype = jnp.real(covariance).dtype
    precision = jnp.finfo(real_dtype)
    operation_bound = jnp.asarray(
        4 * covariance.shape[-1],
        dtype=real_dtype,
    )
    scale = jnp.maximum(
        jnp.max(jnp.abs(covariance)),
        jnp.asarray(precision.tiny, dtype=real_dtype),
    )
    return operation_bound * jnp.asarray(precision.eps, dtype=real_dtype) * scale


def _qr_lower_factor(columns: Array, /) -> Array:
    """Compress covariance columns without forming their Gram matrix."""
    _, upper = jnp.linalg.qr(_adjoint(columns), mode="reduced")
    return _adjoint(upper)


def _qr_gaussian_factor(columns: Array, /, *, factor_id: str) -> GaussianFactor:
    return GaussianFactor(
        _qr_lower_factor(columns),
        factor_id=factor_id,
        resolved_method="qr-square-root",
    )


def _forecast_factor(
    transition: Array,
    filtered: GaussianFactor,
    process: GaussianFactor,
    /,
) -> GaussianFactor:
    columns = jnp.concatenate((transition @ filtered.factor, process.factor), axis=-1)
    return _qr_gaussian_factor(columns, factor_id="kalman-forecast-factor")


def _update_factors(
    predicted: GaussianFactor,
    observation_matrix: Array,
    observation_noise: GaussianFactor,
    /,
) -> tuple[GaussianFactor, GaussianFactor, Array]:
    observation_size = observation_matrix.shape[-2]
    state_size = observation_matrix.shape[-1]
    batch_shape = observation_matrix.shape[:-2]
    dtype = jnp.result_type(
        predicted.factor, observation_matrix, observation_noise.factor
    )
    zero = jnp.zeros(batch_shape + (state_size, observation_size), dtype=dtype)
    top = jnp.concatenate(
        (observation_noise.factor, observation_matrix @ predicted.factor), axis=-1
    )
    bottom = jnp.concatenate((zero, predicted.factor), axis=-1)
    joint = _qr_lower_factor(jnp.concatenate((top, bottom), axis=-2))
    innovation = GaussianFactor(
        joint[..., :observation_size, :observation_size],
        factor_id="kalman-innovation-factor",
        resolved_method="qr-square-root-update",
    )
    filtered = GaussianFactor(
        joint[..., observation_size:, observation_size:],
        factor_id="kalman-filtered-factor",
        resolved_method="qr-square-root-update",
    )
    cross_factor = joint[..., observation_size:, :observation_size]
    gain = jnp.swapaxes(
        jax.scipy.linalg.solve_triangular(
            jnp.swapaxes(innovation.factor, -1, -2),
            jnp.swapaxes(cross_factor, -1, -2),
            lower=False,
        ),
        -1,
        -2,
    )
    return innovation, filtered, gain


class _SquareRootFilterState(StrictModule):
    mean: Array
    factor: GaussianFactor
    time: Array
    log_likelihood: Array
    valid: Array
    status: Array
    step_index: Array


class _SquareRootFilterStep(StrictModule):
    predicted_mean: Array
    predicted_factor: Array
    filtered_mean: Array
    filtered_factor: Array
    transition_matrix: Array
    innovation: Array
    innovation_factor: Array
    normalized_innovation_squared: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    observed_count: Array
    active: Array
    valid: Array
    status: Array


def _square_root_kalman_filter(
    problem,
    /,
    *,
    covariance_regularization: float,
    raise_on_failure: bool,
) -> KalmanFilterResult:
    prior, _, _ = _linear_problem(problem)
    state_size, observation_size, case_count, case_shape = _sizes(problem)
    prior_mean = prior.mean.reshape((case_count, state_size))
    prior_root = jnp.broadcast_to(
        prior.factor, case_shape + (state_size, state_size)
    ).reshape((case_count, state_size, state_size))
    initial_factor = GaussianFactor(
        prior_root,
        factor_id="kalman-state-factor",
        resolved_method="qr-square-root-state",
    )
    initial = _SquareRootFilterState(
        mean=prior_mean,
        factor=initial_factor,
        time=problem.initial_time.reshape((case_count,)),
        log_likelihood=jnp.zeros((case_count,), dtype=jnp.real(prior_mean).dtype),
        valid=jnp.ones((case_count,), dtype=bool),
        status=jnp.zeros((case_count,), dtype=jnp.int32),
        step_index=jnp.asarray(0, dtype=jnp.int32),
    )
    sequence = problem.observations
    case_rank = len(case_shape)

    def step(state: _SquareRootFilterState, _):
        index = state.step_index
        active = sequence.step_valid[..., index].reshape((case_count,))
        target_time = sequence.times[..., index].reshape((case_count,))
        safe_time = jnp.where(active, target_time, state.time)
        matrices, offsets, process_covariances = _transition_parameters(
            problem,
            state.time.reshape(case_shape),
            safe_time.reshape(case_shape),
            jnp.broadcast_to(index, case_shape),
        )
        matrices = jnp.broadcast_to(matrices, (case_count, state_size, state_size))
        offsets = jnp.broadcast_to(offsets, (case_count, state_size))
        process_covariances = jnp.broadcast_to(
            process_covariances, (case_count, state_size, state_size)
        )
        process_factor = gaussian_factor_from_covariance(
            process_covariances,
            factor_id="kalman-process-factor",
        )
        forecast_mean = matrices @ state.mean[..., None]
        forecast_mean = forecast_mean[..., 0] + offsets
        forecast_factor = _forecast_factor(matrices, state.factor, process_factor)
        identity_state = jnp.eye(state_size, dtype=matrices.dtype)
        forecast_mean = jnp.where(active[:, None], forecast_mean, state.mean)
        forecast_root = jnp.where(
            active[:, None, None], forecast_factor.factor, state.factor.factor
        )
        forecast_factor = GaussianFactor(
            forecast_root,
            factor_id="kalman-forecast-factor",
            resolved_method="qr-square-root",
        )
        matrices = jnp.where(active[:, None, None], matrices, identity_state[None, ...])

        observation_matrices, observation_offsets, observation_covariances = (
            _observation_parameters(
                problem,
                safe_time.reshape(case_shape),
                jnp.broadcast_to(index, case_shape),
            )
        )
        observation_matrices = jnp.broadcast_to(
            observation_matrices,
            (case_count, observation_size, state_size),
        )
        observation_offsets = jnp.broadcast_to(
            observation_offsets, (case_count, observation_size)
        )
        observation_covariances = jnp.broadcast_to(
            observation_covariances,
            (case_count, observation_size, observation_size),
        )
        values = jnp.take(sequence.values, index, axis=case_rank).reshape(
            (case_count, observation_size)
        )
        mask = (
            jnp.take(sequence.observation_mask, index, axis=case_rank).reshape(
                (case_count, observation_size)
            )
            & active[:, None]
        )
        mask_value = mask.astype(forecast_mean.dtype)
        effective_matrix = observation_matrices * mask_value[..., :, None]
        eye_observation = jnp.eye(observation_size, dtype=observation_covariances.dtype)
        effective_covariance = (
            observation_covariances * mask_value[..., :, None] * mask_value[..., None, :]
            + eye_observation[None, ...] * (1.0 - mask_value[..., :, None])
            + covariance_regularization
            * eye_observation[None, ...]
            * mask_value[..., :, None]
        )
        observation_factor = gaussian_factor_from_covariance(
            effective_covariance,
            factor_id="kalman-observation-factor",
        )
        predicted_observation = (observation_matrices @ forecast_mean[..., None])[
            ..., 0
        ] + observation_offsets
        innovation = jnp.where(mask, values - predicted_observation, 0.0)
        innovation_factor, updated_factor, gain = _update_factors(
            forecast_factor, effective_matrix, observation_factor
        )
        updated_mean = forecast_mean + (gain @ innovation[..., None])[..., 0]
        diagonal = jnp.diagonal(innovation_factor.factor, axis1=-2, axis2=-1)
        covariance_valid = (
            process_factor.valid
            & observation_factor.valid
            & forecast_factor.valid
            & innovation_factor.valid
            & updated_factor.valid
            & jnp.all(jnp.abs(diagonal) > 0.0, axis=-1)
        )
        solved_innovation = jax.scipy.linalg.solve_triangular(
            innovation_factor.factor, innovation[..., None], lower=True
        )[..., 0]
        nis = jnp.real(jnp.sum(jnp.conj(solved_innovation) * solved_innovation, axis=-1))
        observed_count = jnp.sum(mask, axis=-1)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(diagonal)), axis=-1)
        incremental = -0.5 * (
            nis + logdet + observed_count * jnp.log(jnp.asarray(2.0 * jnp.pi))
        )
        finite = (
            jnp.all(jnp.isfinite(updated_mean), axis=-1)
            & jnp.all(jnp.isfinite(updated_factor.factor), axis=(-1, -2))
            & jnp.isfinite(incremental)
        )
        step_valid = state.valid & covariance_valid & finite
        accepted = active & step_valid
        next_mean = jnp.where(accepted[:, None], updated_mean, state.mean)
        next_root = jnp.where(
            accepted[:, None, None], updated_factor.factor, state.factor.factor
        )
        next_factor = GaussianFactor(
            next_root,
            factor_id="kalman-state-factor",
            resolved_method="qr-square-root-state",
        )
        next_time = jnp.where(active, target_time, state.time)
        next_valid = state.valid & jnp.where(active, step_valid, True)
        status = jnp.where(
            ~active,
            KALMAN_SUCCESS,
            jnp.where(
                ~covariance_valid,
                KALMAN_INNOVATION_COVARIANCE_FAILURE,
                jnp.where(~finite, KALMAN_NONFINITE, KALMAN_SUCCESS),
            ),
        ).astype(jnp.int32)
        accepted_increment = jnp.where(accepted, incremental, 0.0)
        cumulative = state.log_likelihood + accepted_increment
        next_state = _SquareRootFilterState(
            mean=next_mean,
            factor=next_factor,
            time=next_time,
            log_likelihood=cumulative,
            valid=next_valid,
            status=status,
            step_index=index + 1,
        )
        record = _SquareRootFilterStep(
            predicted_mean=forecast_mean,
            predicted_factor=forecast_factor.factor,
            filtered_mean=next_mean,
            filtered_factor=next_factor.factor,
            transition_matrix=matrices,
            innovation=innovation,
            innovation_factor=innovation_factor.factor,
            normalized_innovation_squared=jnp.where(active, nis, 0.0),
            incremental_log_likelihood=accepted_increment,
            cumulative_log_likelihood=cumulative,
            observed_count=observed_count,
            active=active,
            valid=step_valid,
            status=status,
        )
        return next_state, record

    state, records = jax.lax.scan(step, initial, xs=None, length=sequence.num_steps)
    predicted_covariance = _covariance(records.predicted_factor)
    filtered_covariance = _covariance(records.filtered_factor)
    innovation_covariance = _covariance(records.innovation_factor)
    final_covariance = _covariance(state.factor.factor)
    final_state = KalmanFilterState(
        mean=state.mean.reshape(case_shape + problem.model.state_shape),
        covariance=final_covariance.reshape(case_shape + (state_size, state_size)),
        time=state.time.reshape(case_shape),
        log_likelihood=state.log_likelihood.reshape(case_shape),
        valid=state.valid.reshape(case_shape),
        status=state.status.reshape(case_shape),
        step_index=state.step_index,
        problem_id=problem.problem_id,
        covariance_regularization=covariance_regularization,
    )
    result = KalmanFilterResult(
        predicted_means=_time_to_result_axis(
            records.predicted_mean.reshape(
                (sequence.num_steps,) + case_shape + problem.model.state_shape
            ),
            case_rank,
        ),
        predicted_covariances=_time_to_result_axis(
            predicted_covariance.reshape(
                (sequence.num_steps,) + case_shape + (state_size, state_size)
            ),
            case_rank,
        ),
        filtered_means=_time_to_result_axis(
            records.filtered_mean.reshape(
                (sequence.num_steps,) + case_shape + problem.model.state_shape
            ),
            case_rank,
        ),
        filtered_covariances=_time_to_result_axis(
            filtered_covariance.reshape(
                (sequence.num_steps,) + case_shape + (state_size, state_size)
            ),
            case_rank,
        ),
        transition_matrices=_time_to_result_axis(
            records.transition_matrix.reshape(
                (sequence.num_steps,) + case_shape + (state_size, state_size)
            ),
            case_rank,
        ),
        innovations=_time_to_result_axis(
            records.innovation.reshape(
                (sequence.num_steps,) + case_shape + problem.model.observation_shape
            ),
            case_rank,
        ),
        innovation_covariances=_time_to_result_axis(
            innovation_covariance.reshape(
                (sequence.num_steps,) + case_shape + (observation_size, observation_size)
            ),
            case_rank,
        ),
        normalized_innovation_squared=_time_to_result_axis(
            records.normalized_innovation_squared.reshape(
                (sequence.num_steps,) + case_shape
            ),
            case_rank,
        ),
        incremental_log_likelihood=_time_to_result_axis(
            records.incremental_log_likelihood.reshape(
                (sequence.num_steps,) + case_shape
            ),
            case_rank,
        ),
        cumulative_log_likelihood=_time_to_result_axis(
            records.cumulative_log_likelihood.reshape((sequence.num_steps,) + case_shape),
            case_rank,
        ),
        observed_counts=_time_to_result_axis(
            records.observed_count.reshape((sequence.num_steps,) + case_shape),
            case_rank,
        ),
        step_valid=sequence.step_valid,
        valid=_time_to_result_axis(
            records.valid.reshape((sequence.num_steps,) + case_shape),
            case_rank,
        ),
        status=_time_to_result_axis(
            records.status.reshape((sequence.num_steps,) + case_shape),
            case_rank,
        ),
        final_state=final_state,
        state_shape=problem.model.state_shape,
        observation_shape=problem.model.observation_shape,
        case_shape=case_shape,
        case_ids=sequence.case_ids,
        model_id=problem.model.model_id,
        problem_id=problem.problem_id,
        sequence_id=sequence.sequence_id,
        input_id=None if problem.input_signal is None else problem.input_signal.input_id,
        covariance_regularization=covariance_regularization,
        execution_method="sequential",
        covariance_form="square_root",
    )
    if raise_on_failure and not bool(jnp.all(result.successful)):
        raise RuntimeError(
            "Square-root Kalman filtering failed for at least one physical case."
        )
    return result


def _psd_pseudoinverse(covariance: Array, /) -> Array:
    values, vectors = jnp.linalg.eigh(0.5 * (covariance + _adjoint(covariance)))
    inverse = jnp.where(values > 0.0, 1.0 / values, 0.0)
    return (vectors * inverse[..., None, :]) @ _adjoint(vectors)


def _smoothing_factor(
    filtered: GaussianFactor,
    predicted: GaussianFactor,
    transition: Array,
    next_smoothed: GaussianFactor,
    /,
) -> tuple[GaussianFactor, Array]:
    state_size = filtered.event_size
    next_current = transition @ filtered.covariance
    current_next = _adjoint(next_current)
    joint_covariance = jnp.concatenate(
        (
            jnp.concatenate((predicted.covariance, next_current), axis=-1),
            jnp.concatenate((current_next, filtered.covariance), axis=-1),
        ),
        axis=-2,
    )

    def factor_one(covariance: Array, /) -> Array:
        return gaussian_factor_from_covariance(
            covariance,
            rank_tolerance=_covariance_assembly_roundoff(covariance),
            factor_id="rts-joint-factor",
        ).factor

    if joint_covariance.ndim == 2:
        joint_factor = factor_one(joint_covariance)
    else:
        event_size = joint_covariance.shape[-1]
        flattened = joint_covariance.reshape((-1, event_size, event_size))
        joint_factor = jax.vmap(factor_one)(flattened).reshape(joint_covariance.shape)
    joint_root = _qr_lower_factor(joint_factor)
    conditional_root = joint_root[..., state_size:, state_size:]
    gain = current_next @ _psd_pseudoinverse(predicted.covariance)
    transported_root = gain @ next_smoothed.factor
    smoothed = _qr_gaussian_factor(
        jnp.concatenate((conditional_root, transported_root), axis=-1),
        factor_id="rts-smoothed-factor",
    )
    return smoothed, gain


def _square_root_rts_smoother(result: KalmanFilterResult, /) -> KalmanSmootherResult:
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
    filtered_factors = gaussian_factor_from_covariance(
        filtered_covariance,
        factor_id="rts-filtered-factor",
    )
    base_valid = (
        result.valid.reshape((case_count, num_steps)) & active & filtered_factors.valid
    )

    def step(carry, inputs):
        next_mean, next_root, next_valid = carry
        (
            current_mean,
            current_root,
            current_factor_valid,
            next_predicted_mean,
            next_predicted_covariance,
            next_transition,
            current_valid,
        ) = inputs
        filtered_factor = GaussianFactor(
            current_root,
            factor_id="rts-filtered-factor",
            resolved_method=filtered_factors.resolved_method,
        )
        predicted_factor = gaussian_factor_from_covariance(
            next_predicted_covariance,
            factor_id="rts-predicted-factor",
        )
        next_factor = GaussianFactor(
            next_root,
            factor_id="rts-next-smoothed-factor",
            resolved_method="qr-square-root-smoothing",
        )
        proposed_factor, gain = _smoothing_factor(
            filtered_factor,
            predicted_factor,
            next_transition,
            next_factor,
        )
        proposed_mean = (
            current_mean + (gain @ (next_mean - next_predicted_mean)[..., None])[..., 0]
        )
        factor_valid = (
            current_factor_valid
            & filtered_factor.valid
            & predicted_factor.valid
            & next_factor.valid
            & proposed_factor.valid
        )
        proposed_finite = jnp.all(jnp.isfinite(proposed_mean), axis=-1) & jnp.all(
            jnp.isfinite(proposed_factor.covariance),
            axis=(-2, -1),
        )
        pair_valid = current_valid & next_valid & factor_valid & proposed_finite
        mean = jnp.where(pair_valid[:, None], proposed_mean, current_mean)
        root = jnp.where(
            pair_valid[:, None, None],
            proposed_factor.factor,
            current_root,
        )
        accepted_gain = jnp.where(pair_valid[:, None, None], gain, 0.0)
        next_carry = (mean, root, pair_valid)
        return next_carry, (mean, root, accepted_gain, pair_valid)

    scan_inputs = (
        jnp.swapaxes(filtered_mean[:, :-1], 0, 1),
        jnp.swapaxes(filtered_factors.factor[:, :-1], 0, 1),
        jnp.swapaxes(filtered_factors.valid[:, :-1], 0, 1),
        jnp.swapaxes(predicted_mean[:, 1:], 0, 1),
        jnp.swapaxes(predicted_covariance[:, 1:], 0, 1),
        jnp.swapaxes(transitions[:, 1:], 0, 1),
        jnp.swapaxes(base_valid[:, :-1], 0, 1),
    )
    initial = (
        filtered_mean[:, -1],
        filtered_factors.factor[:, -1],
        base_valid[:, -1],
    )
    _, history = jax.lax.scan(step, initial, scan_inputs, reverse=True)
    history_mean, history_root, gains, history_valid = history
    means = jnp.swapaxes(
        jnp.concatenate((history_mean, initial[0][None, ...]), axis=0),
        0,
        1,
    )
    roots = jnp.swapaxes(
        jnp.concatenate((history_root, initial[1][None, ...]), axis=0),
        0,
        1,
    )
    valid = jnp.swapaxes(
        jnp.concatenate((history_valid, initial[2][None, ...]), axis=0),
        0,
        1,
    )
    covariances = _covariance(roots)
    gains = jnp.swapaxes(gains, 0, 1)
    return KalmanSmootherResult(
        means=means.reshape(case_shape + (num_steps,) + result.state_shape),
        covariances=covariances.reshape(case_shape + (num_steps, state_size, state_size)),
        gains=gains.reshape(case_shape + (max(num_steps - 1, 0), state_size, state_size)),
        valid=valid.reshape(case_shape + (num_steps,)),
        filter_result=result,
        execution_method="sequential",
        covariance_form="square_root",
    )


__all__: list[str] = []
