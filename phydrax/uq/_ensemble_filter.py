#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal, TypeAlias

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key

from .._strict import StrictModule
from ..stochastic._state_space import (
    GaussianObservationModel,
    LinearGaussianObservationModel,
    state_space_key,
    StateSpaceProblem,
)
from ._predictive import PredictiveField, SampleAxis


EnsembleFilterStatus: TypeAlias = Literal[
    "success", "transition_failure", "transform_failure", "nonfinite"
]
ENSEMBLE_FILTER_SUCCESS = 0
ENSEMBLE_FILTER_TRANSITION_FAILURE = 1
ENSEMBLE_FILTER_TRANSFORM_FAILURE = 2
ENSEMBLE_FILTER_NONFINITE = 3


def ensemble_filter_status_name(value: int, /) -> EnsembleFilterStatus:
    code = int(value)
    if code == ENSEMBLE_FILTER_SUCCESS:
        return "success"
    if code == ENSEMBLE_FILTER_TRANSITION_FAILURE:
        return "transition_failure"
    if code == ENSEMBLE_FILTER_TRANSFORM_FAILURE:
        return "transform_failure"
    if code == ENSEMBLE_FILTER_NONFINITE:
        return "nonfinite"
    raise ValueError(f"Unknown ensemble-filter status code {code}.")


def _validate_problem(problem: StateSpaceProblem) -> None:
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    if not isinstance(
        problem.model.observation,
        (GaussianObservationModel, LinearGaussianObservationModel),
    ):
        raise TypeError(
            "The ensemble transform filter requires a Gaussian observation model."
        )


def _configuration(
    ensemble_size: int,
    inflation: float,
    covariance_regularization: float,
) -> tuple[int, float, float]:
    count = int(ensemble_size)
    if count < 2:
        raise ValueError("ensemble_size must be at least two.")
    inflation_value = float(inflation)
    if not np.isfinite(inflation_value) or inflation_value <= 0.0:
        raise ValueError("inflation must be finite and positive.")
    regularization = float(covariance_regularization)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("covariance_regularization must be finite and nonnegative.")
    return count, inflation_value, regularization


def _case_count(problem: StateSpaceProblem) -> int:
    shape = problem.observations.case_shape
    return prod(shape) if shape else 1


def _case_value(value: Array, case_index: int, case_shape: tuple[int, ...], /) -> Array:
    array = jnp.asarray(value)
    if not case_shape:
        return array
    return array.reshape((prod(case_shape),) + array.shape[len(case_shape) :])[case_index]


def _observation_covariance(model, time: Array, /) -> Array:
    if isinstance(model, GaussianObservationModel):
        return model.covariance_at(time)
    if isinstance(model, LinearGaussianObservationModel):
        return model.parameters(time)[2]
    raise TypeError("The ensemble transform filter requires Gaussian observations.")


class EnsembleFilterState(StrictModule):
    """Streaming deterministic square-root filter state."""

    ensemble: Array
    time: Array
    log_likelihood: Array
    valid: Array
    status: Array
    root_key: Array
    step_index: int = eqx.field(static=True)
    ensemble_size: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    inflation: float = eqx.field(static=True)
    covariance_regularization: float = eqx.field(static=True)


class EnsembleFilterStep(StrictModule):
    """One forecast and ensemble-transform analysis record."""

    forecast_ensemble: Array
    analysis_ensemble: Array
    forecast_observations: Array
    innovation: Array
    normalized_innovation_squared: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    observed_count: Array
    active: Array
    valid: Array
    status: Array


class EnsembleFilterResult(StrictModule):
    """Complete ETKF history using only member- and observation-space solves."""

    forecast_ensembles: Array
    analysis_ensembles: Array
    forecast_observations: Array
    innovations: Array
    normalized_innovation_squared: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    observed_counts: Array
    step_valid: Array
    valid: Array
    status: Array
    times: Array
    final_state: EnsembleFilterState
    problem: StateSpaceProblem
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    ensemble_size: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    inflation: float = eqx.field(static=True)
    covariance_regularization: float = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


def initialize_ensemble_filter(
    key: Key[Array, ""],
    problem: StateSpaceProblem,
    /,
    *,
    ensemble_size: int,
    inflation: float = 1.0,
    covariance_regularization: float = 0.0,
) -> EnsembleFilterState:
    _validate_problem(problem)
    count, inflation_value, regularization = _configuration(
        ensemble_size, inflation, covariance_regularization
    )
    case_shape = problem.observations.case_shape
    draws = []
    for case_index, case_id in enumerate(problem.observations.case_ids):
        case_draws = []
        for member in range(count):
            member_key = state_space_key(
                key, "ensemble-filter-prior", case_id, 0, member=member
            )
            complete_draw = problem.model.prior.sample(member_key)
            case_draws.append(_case_value(complete_draw, case_index, case_shape))
        draws.append(jnp.stack(case_draws, axis=0))
    ensemble = jnp.stack(draws, axis=0).reshape(
        case_shape + (count,) + problem.model.state_shape
    )
    finite = jnp.all(
        jnp.isfinite(ensemble),
        axis=tuple(range(len(case_shape), ensemble.ndim)),
    )
    return EnsembleFilterState(
        ensemble=ensemble,
        time=problem.initial_time,
        log_likelihood=jnp.zeros(case_shape, dtype=ensemble.dtype),
        valid=finite,
        status=jnp.where(
            finite, ENSEMBLE_FILTER_SUCCESS, ENSEMBLE_FILTER_NONFINITE
        ).astype(jnp.int32),
        root_key=jnp.asarray(key),
        step_index=0,
        ensemble_size=count,
        problem_id=problem.problem_id,
        inflation=inflation_value,
        covariance_regularization=regularization,
    )


def _forecast(
    problem: StateSpaceProblem,
    state: EnsembleFilterState,
    active: Array,
    target_time: Array,
) -> tuple[Array, Array]:
    case_shape = problem.observations.case_shape
    case_count = _case_count(problem)
    count = state.ensemble_size
    previous = state.ensemble.reshape((case_count, count) + problem.model.state_shape)
    starts = state.time.reshape((case_count,))
    ends = target_time.reshape((case_count,))
    active_flat = active.reshape((case_count,))
    cases = []
    valid_cases = []
    for case_index, case_id in enumerate(problem.observations.case_ids):
        members = []
        member_validity = []
        for member in range(count):
            if bool(active_flat[case_index]) and bool(
                state.valid.reshape((-1,))[case_index]
            ):
                member_key = state_space_key(
                    state.root_key,
                    "ensemble-filter-transition",
                    case_id,
                    state.step_index,
                    member=member,
                )
                sample = problem.model.transition.sample(
                    member_key,
                    previous[case_index, member],
                    starts[case_index],
                    ends[case_index],
                )
                sample_valid = jnp.all(sample.valid) & jnp.all(sample.status == 0)
                members.append(
                    jnp.where(
                        sample_valid,
                        sample.values,
                        previous[case_index, member],
                    )
                )
                member_validity.append(sample_valid)
            else:
                members.append(previous[case_index, member])
                member_validity.append(jnp.asarray(True))
        cases.append(jnp.stack(members, axis=0))
        valid_cases.append(jnp.stack(member_validity, axis=0))
    return (
        jnp.stack(cases, axis=0).reshape(
            case_shape + (count,) + problem.model.state_shape
        ),
        jnp.stack(valid_cases, axis=0).reshape(case_shape + (count,)),
    )


def _etkf_case(
    problem: StateSpaceProblem,
    forecast: Array,
    value: Array,
    mask: Array,
    time: Array,
    /,
    *,
    inflation: float,
    covariance_regularization: float,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    count = int(forecast.shape[0])
    state_size = prod(problem.model.state_shape) if problem.model.state_shape else 1
    observation_size = (
        prod(problem.model.observation_shape) if problem.model.observation_shape else 1
    )
    forecast_flat = forecast.reshape((count, state_size))
    forecast_mean = jnp.mean(forecast_flat, axis=0)
    state_anomalies = (forecast_flat - forecast_mean[None, :]) * inflation
    inflated_forecast = forecast_mean[None, :] + state_anomalies
    observations = []
    for member in range(count):
        observations.append(
            problem.model.observation.location(
                inflated_forecast[member].reshape(problem.model.state_shape), time
            )
        )
    forecast_observations = jnp.stack(observations, axis=0).reshape(
        (count, observation_size)
    )
    observation_mean = jnp.mean(forecast_observations, axis=0)
    observation_anomalies = forecast_observations - observation_mean[None, :]
    mask_flat = jnp.asarray(mask, dtype=bool).reshape((observation_size,))
    active_float = mask_flat.astype(forecast.dtype)
    observation_anomalies = observation_anomalies * active_float[None, :]
    innovation = jnp.where(
        mask_flat,
        jnp.asarray(value).reshape((observation_size,)) - observation_mean,
        0.0,
    )
    covariance = jnp.asarray(
        _observation_covariance(problem.model.observation, time)
    ).reshape((observation_size, observation_size))
    identity_observation = jnp.eye(observation_size, dtype=forecast.dtype)
    effective_covariance = (
        covariance * active_float[:, None] * active_float[None, :]
        + identity_observation * (1.0 - active_float[:, None])
        + covariance_regularization * identity_observation * active_float[:, None]
    )
    covariance_scale = jnp.linalg.cholesky(effective_covariance)
    inverse_observation_anomalies = jax.scipy.linalg.cho_solve(
        (covariance_scale, True), observation_anomalies.T
    ).T
    inverse_innovation = jax.scipy.linalg.cho_solve((covariance_scale, True), innovation)
    transform_precision = (count - 1) * jnp.eye(
        count, dtype=forecast.dtype
    ) + observation_anomalies @ inverse_observation_anomalies.T
    eigenvalues, eigenvectors = jnp.linalg.eigh(transform_precision)
    transform_valid = (
        jnp.all(jnp.isfinite(eigenvalues))
        & jnp.all(eigenvalues > 0.0)
        & jnp.all(jnp.isfinite(covariance_scale))
    )
    mean_weights = jnp.linalg.solve(
        transform_precision,
        observation_anomalies @ inverse_innovation,
    )
    analysis_mean = forecast_mean + state_anomalies.T @ mean_weights
    transform = (
        eigenvectors
        * jnp.sqrt((count - 1) / jnp.maximum(eigenvalues, jnp.finfo(float).tiny))[None, :]
    ) @ eigenvectors.T
    analysis_anomalies = transform @ state_anomalies
    analysis = analysis_mean[None, :] + analysis_anomalies
    predictive_covariance = (
        observation_anomalies.T @ observation_anomalies / (count - 1)
        + effective_covariance
    )
    predictive_scale = jnp.linalg.cholesky(predictive_covariance)
    solved_innovation = jax.scipy.linalg.solve_triangular(
        predictive_scale, innovation, lower=True
    )
    nis = jnp.sum(solved_innovation**2)
    observed_count = jnp.sum(mask_flat)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(predictive_scale)))
    log_likelihood = -0.5 * (nis + logdet + observed_count * jnp.log(2.0 * jnp.pi))
    finite = (
        jnp.all(jnp.isfinite(analysis))
        & jnp.all(jnp.isfinite(forecast_observations))
        & jnp.isfinite(log_likelihood)
    )
    return (
        inflated_forecast.reshape((count,) + problem.model.state_shape),
        analysis.reshape((count,) + problem.model.state_shape),
        forecast_observations.reshape((count,) + problem.model.observation_shape),
        innovation.reshape(problem.model.observation_shape),
        nis,
        log_likelihood,
        transform_valid & finite,
    )


def ensemble_filter_step(
    problem: StateSpaceProblem,
    state: EnsembleFilterState,
    /,
) -> tuple[EnsembleFilterState, EnsembleFilterStep]:
    """Process one ETKF step with no state-space covariance construction."""
    _validate_problem(problem)
    if not isinstance(state, EnsembleFilterState):
        raise TypeError("state must be an EnsembleFilterState.")
    if state.problem_id != problem.problem_id:
        raise ValueError("Ensemble-filter state and problem IDs do not match.")
    index = state.step_index
    sequence = problem.observations
    if index >= sequence.num_steps:
        raise ValueError("The ensemble-filter state consumed every observation step.")
    case_shape = sequence.case_shape
    case_count = _case_count(problem)
    count = state.ensemble_size
    active = sequence.step_valid[..., index]
    target_time = sequence.times[..., index]
    step_axis = len(case_shape)
    values = jnp.take(sequence.values, index, axis=step_axis)
    masks = jnp.take(sequence.observation_mask, index, axis=step_axis)
    raw_forecast, transition_valid = _forecast(problem, state, active, target_time)
    flat_forecast = raw_forecast.reshape((case_count, count) + problem.model.state_shape)
    flat_values = values.reshape((case_count,) + problem.model.observation_shape)
    flat_masks = masks.reshape((case_count,) + problem.model.observation_shape)
    flat_times = target_time.reshape((case_count,))
    flat_active = active.reshape((case_count,))
    previous = state.ensemble.reshape((case_count, count) + problem.model.state_shape)
    forecasts = []
    analyses = []
    forecast_observations = []
    innovations = []
    nis_values = []
    likelihood_values = []
    transform_validity = []
    for case_index in range(case_count):
        if bool(flat_active[case_index]) and bool(state.valid.reshape((-1,))[case_index]):
            values_case = _etkf_case(
                problem,
                flat_forecast[case_index],
                flat_values[case_index],
                flat_masks[case_index],
                flat_times[case_index],
                inflation=state.inflation,
                covariance_regularization=state.covariance_regularization,
            )
            (
                forecast_case,
                analysis_case,
                observation_case,
                innovation_case,
                nis,
                likelihood,
                valid,
            ) = values_case
        else:
            forecast_case = previous[case_index]
            analysis_case = previous[case_index]
            observation_case = jnp.stack(
                [
                    problem.model.observation.location(
                        previous[case_index, member],
                        state.time.reshape((-1,))[case_index],
                    )
                    for member in range(count)
                ],
                axis=0,
            )
            innovation_case = jnp.zeros(problem.model.observation_shape)
            nis = jnp.asarray(0.0)
            likelihood = jnp.asarray(0.0)
            valid = jnp.asarray(True)
        forecasts.append(forecast_case)
        analyses.append(analysis_case)
        forecast_observations.append(observation_case)
        innovations.append(innovation_case)
        nis_values.append(nis)
        likelihood_values.append(likelihood)
        transform_validity.append(valid)
    forecast = jnp.stack(forecasts, axis=0).reshape(
        case_shape + (count,) + problem.model.state_shape
    )
    analysis = jnp.stack(analyses, axis=0).reshape(
        case_shape + (count,) + problem.model.state_shape
    )
    predicted_observations = jnp.stack(forecast_observations, axis=0).reshape(
        case_shape + (count,) + problem.model.observation_shape
    )
    innovation = jnp.stack(innovations, axis=0).reshape(
        case_shape + problem.model.observation_shape
    )
    nis = jnp.stack(nis_values, axis=0).reshape(case_shape)
    likelihood = jnp.stack(likelihood_values, axis=0).reshape(case_shape)
    transform_valid = jnp.stack(transform_validity, axis=0).reshape(case_shape)
    transition_case_valid = jnp.all(transition_valid, axis=-1)
    accepted = active & state.valid & transition_case_valid & transform_valid
    next_ensemble = jnp.where(
        accepted[..., None, *([None] * len(problem.model.state_shape))],
        analysis,
        state.ensemble,
    )
    status = jnp.where(
        ~active,
        ENSEMBLE_FILTER_SUCCESS,
        jnp.where(
            ~transition_case_valid,
            ENSEMBLE_FILTER_TRANSITION_FAILURE,
            jnp.where(
                ~transform_valid,
                ENSEMBLE_FILTER_TRANSFORM_FAILURE,
                ENSEMBLE_FILTER_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    next_valid = state.valid & jnp.where(
        active, transition_case_valid & transform_valid, True
    )
    increment = jnp.where(active, likelihood, 0.0)
    cumulative = state.log_likelihood + increment
    next_state = EnsembleFilterState(
        ensemble=next_ensemble,
        time=jnp.where(active, target_time, state.time),
        log_likelihood=cumulative,
        valid=next_valid,
        status=status,
        root_key=state.root_key,
        step_index=index + 1,
        ensemble_size=count,
        problem_id=problem.problem_id,
        inflation=state.inflation,
        covariance_regularization=state.covariance_regularization,
    )
    record = EnsembleFilterStep(
        forecast_ensemble=forecast,
        analysis_ensemble=next_ensemble,
        forecast_observations=predicted_observations,
        innovation=innovation,
        normalized_innovation_squared=nis,
        incremental_log_likelihood=increment,
        cumulative_log_likelihood=cumulative,
        observed_count=jnp.sum(masks.reshape(case_shape + (-1,)), axis=-1),
        active=active,
        valid=transition_case_valid & transform_valid,
        status=status,
    )
    return next_state, record


def _stack(values: list[Array], case_rank: int, /) -> Array:
    return jnp.stack(values, axis=case_rank)


def ensemble_transform_kalman_filter(
    key: Key[Array, ""],
    problem: StateSpaceProblem,
    /,
    *,
    ensemble_size: int,
    inflation: float = 1.0,
    covariance_regularization: float = 0.0,
    raise_on_failure: bool = False,
) -> EnsembleFilterResult:
    """Run a deterministic ensemble transform Kalman filter."""
    state = initialize_ensemble_filter(
        key,
        problem,
        ensemble_size=ensemble_size,
        inflation=inflation,
        covariance_regularization=covariance_regularization,
    )
    records: list[EnsembleFilterStep] = []
    for _ in range(problem.observations.num_steps):
        state, record = ensemble_filter_step(problem, state)
        records.append(record)
    rank = len(problem.observations.case_shape)
    result = EnsembleFilterResult(
        forecast_ensembles=_stack([record.forecast_ensemble for record in records], rank),
        analysis_ensembles=_stack([record.analysis_ensemble for record in records], rank),
        forecast_observations=_stack(
            [record.forecast_observations for record in records], rank
        ),
        innovations=_stack([record.innovation for record in records], rank),
        normalized_innovation_squared=_stack(
            [record.normalized_innovation_squared for record in records], rank
        ),
        incremental_log_likelihood=_stack(
            [record.incremental_log_likelihood for record in records], rank
        ),
        cumulative_log_likelihood=_stack(
            [record.cumulative_log_likelihood for record in records], rank
        ),
        observed_counts=_stack([record.observed_count for record in records], rank),
        step_valid=problem.observations.step_valid,
        valid=_stack([record.valid for record in records], rank),
        status=_stack([record.status for record in records], rank),
        times=problem.observations.times,
        final_state=state,
        problem=problem,
        state_shape=problem.model.state_shape,
        observation_shape=problem.model.observation_shape,
        case_shape=problem.observations.case_shape,
        case_axes=problem.observations.case_axes,
        case_ids=problem.observations.case_ids,
        ensemble_size=state.ensemble_size,
        model_id=problem.model.model_id,
        problem_id=problem.problem_id,
        sequence_id=problem.observations.sequence_id,
        inflation=state.inflation,
        covariance_regularization=state.covariance_regularization,
    )
    if raise_on_failure and not bool(jnp.all(result.successful)):
        raise RuntimeError("Ensemble filtering failed for at least one physical case.")
    return result


class EnsembleSmootherResult(StrictModule):
    """Fixed-interval ensemble smoother evaluated through member-space regressions."""

    ensembles: Array
    valid: Array
    filter_result: EnsembleFilterResult
    pseudoinverse_tolerance: float = eqx.field(static=True)


def ensemble_kalman_smoother(
    result: EnsembleFilterResult,
    /,
    *,
    pseudoinverse_tolerance: float = 1e-10,
) -> EnsembleSmootherResult:
    """Smooth an ETKF ensemble without constructing a state covariance matrix."""
    if not isinstance(result, EnsembleFilterResult):
        raise TypeError("result must be an EnsembleFilterResult.")
    tolerance = float(pseudoinverse_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("pseudoinverse_tolerance must be finite and positive.")
    case_count = prod(result.case_shape) if result.case_shape else 1
    num_steps = result.step_valid.shape[-1]
    state_size = prod(result.state_shape) if result.state_shape else 1
    count = result.ensemble_size
    analysis = result.analysis_ensembles.reshape(
        (case_count, num_steps, count, state_size)
    )
    forecast = result.forecast_ensembles.reshape(
        (case_count, num_steps, count, state_size)
    )
    valid = result.valid.reshape((case_count, num_steps)) & result.step_valid.reshape(
        (case_count, num_steps)
    )
    smoothed = analysis
    for step in range(num_steps - 2, -1, -1):
        filtered_anomalies = analysis[:, step] - jnp.mean(
            analysis[:, step], axis=1, keepdims=True
        )
        forecast_anomalies = forecast[:, step + 1] - jnp.mean(
            forecast[:, step + 1], axis=1, keepdims=True
        )
        gram = jnp.einsum("cid,cjd->cij", forecast_anomalies, forecast_anomalies)
        eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
        cutoff = tolerance * jnp.maximum(eigenvalues[:, -1:], 1.0)
        inverse_values = jnp.where(eigenvalues > cutoff, 1.0 / eigenvalues, 0.0)
        gram_pseudoinverse = jnp.einsum(
            "cik,ck,cjk->cij", eigenvectors, inverse_values, eigenvectors
        )
        delta = smoothed[:, step + 1] - forecast[:, step + 1]
        coefficients = jnp.einsum(
            "cid,cjd,cjk->cik", delta, forecast_anomalies, gram_pseudoinverse
        )
        correction = jnp.einsum("cij,cjd->cid", coefficients, filtered_anomalies)
        proposed = analysis[:, step] + correction
        pair_valid = valid[:, step] & valid[:, step + 1]
        smoothed = smoothed.at[:, step].set(
            jnp.where(pair_valid[:, None, None], proposed, smoothed[:, step])
        )
    return EnsembleSmootherResult(
        ensembles=smoothed.reshape(
            result.case_shape + (num_steps, count) + result.state_shape
        ),
        valid=valid.reshape(result.case_shape + (num_steps,)),
        filter_result=result,
        pseudoinverse_tolerance=tolerance,
    )


def ensemble_filter_predictive(
    result: EnsembleFilterResult | EnsembleSmootherResult,
    /,
    *,
    member_dim: str = "ensemble",
    time_dim: str = "time",
) -> PredictiveField:
    """Expose filter or smoother ensembles through the shared predictive contract."""
    if isinstance(result, EnsembleSmootherResult):
        values = result.ensembles
        filter_result = result.filter_result
        valid = result.valid
    elif isinstance(result, EnsembleFilterResult):
        values = result.analysis_ensembles
        filter_result = result
        valid = result.valid & result.step_valid
    else:
        raise TypeError(
            "result must be an EnsembleFilterResult or EnsembleSmootherResult."
        )
    if not member_dim or not time_dim or member_dim == time_dim:
        raise ValueError("member_dim and time_dim must be distinct non-empty names.")
    if member_dim in filter_result.case_axes or time_dim in filter_result.case_axes:
        raise ValueError(
            "Predictive member/time dimensions must not collide with case axes."
        )
    mask = valid[..., None, *([None] * len(filter_result.state_shape))]
    samples = jnp.where(mask, values, jnp.nan)
    dims = (
        filter_result.case_axes
        + (time_dim, member_dim)
        + (None,) * len(filter_result.state_shape)
    )
    return PredictiveField(
        cx.Field(samples, dims=dims),
        (SampleAxis(member_dim, "process"),),
    )


class EnsembleFilterDiagnostics(StrictModule):
    """Spread, rank, innovation, and finite-value diagnostics by case and time."""

    ensemble_spread: Array
    effective_rank: Array
    normalized_innovation_squared: Array
    valid_steps: Array
    finite: Array

    @property
    def passed(self) -> bool:
        return bool(jnp.all(self.finite))


def ensemble_filter_diagnostics(
    result: EnsembleFilterResult,
    /,
    *,
    rank_tolerance: float = 1e-10,
) -> EnsembleFilterDiagnostics:
    if not isinstance(result, EnsembleFilterResult):
        raise TypeError("result must be an EnsembleFilterResult.")
    tolerance = float(rank_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank_tolerance must be finite and positive.")
    state_size = prod(result.state_shape) if result.state_shape else 1
    ensemble = result.analysis_ensembles.reshape(
        result.case_shape
        + (result.step_valid.shape[-1], result.ensemble_size, state_size)
    )
    anomalies = ensemble - jnp.mean(ensemble, axis=-2, keepdims=True)
    spread = jnp.sqrt(jnp.sum(jnp.var(ensemble, axis=-2, ddof=1), axis=-1))
    singular_values = jnp.linalg.svd(anomalies, compute_uv=False)
    cutoff = tolerance * jnp.maximum(singular_values[..., :1], 1.0)
    rank = jnp.sum(singular_values > cutoff, axis=-1)
    active = result.step_valid
    finite = (
        jnp.all(result.valid | ~active, axis=-1)
        & jnp.all(jnp.isfinite(spread) | ~active, axis=-1)
        & jnp.isfinite(result.final_state.log_likelihood)
    )
    return EnsembleFilterDiagnostics(
        ensemble_spread=spread,
        effective_rank=rank,
        normalized_innovation_squared=result.normalized_innovation_squared,
        valid_steps=jnp.sum(active, axis=-1),
        finite=finite,
    )


__all__ = [
    "ENSEMBLE_FILTER_NONFINITE",
    "ENSEMBLE_FILTER_SUCCESS",
    "ENSEMBLE_FILTER_TRANSFORM_FAILURE",
    "ENSEMBLE_FILTER_TRANSITION_FAILURE",
    "ensemble_filter_diagnostics",
    "EnsembleFilterDiagnostics",
    "ensemble_filter_predictive",
    "EnsembleFilterResult",
    "EnsembleFilterState",
    "ensemble_filter_status_name",
    "EnsembleFilterStatus",
    "EnsembleFilterStep",
    "ensemble_filter_step",
    "ensemble_kalman_smoother",
    "EnsembleSmootherResult",
    "ensemble_transform_kalman_filter",
    "initialize_ensemble_filter",
]
