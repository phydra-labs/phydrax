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
import optimistix as optx
from jaxtyping import Array

import phydrax.ein as ein

from .._strict import StrictModule
from ..stochastic._state_space import (
    GaussianStatePrior,
    LinearGaussianObservationModel,
    LinearGaussianTransitionKernel,
    StateSpaceProblem,
)
from ._kalman import initialize_kalman_filter, kalman_filter_step, KalmanFilterState


BellmanExecutionMethod: TypeAlias = Literal["auto", "analytic", "optimization"]
BellmanCurvatureMethod: TypeAlias = Literal["observed", "score-outer-product"]
BellmanStatus: TypeAlias = Literal[
    "success",
    "initialization_optimizer_failure",
    "initialization_curvature_failure",
    "prediction_optimizer_failure",
    "prediction_curvature_failure",
    "update_optimizer_failure",
    "update_curvature_failure",
    "pseudo_likelihood_failure",
]

BELLMAN_SUCCESS = 0
BELLMAN_INITIALIZATION_OPTIMIZER_FAILURE = 1
BELLMAN_INITIALIZATION_CURVATURE_FAILURE = 2
BELLMAN_PREDICTION_OPTIMIZER_FAILURE = 3
BELLMAN_PREDICTION_CURVATURE_FAILURE = 4
BELLMAN_UPDATE_OPTIMIZER_FAILURE = 5
BELLMAN_UPDATE_CURVATURE_FAILURE = 6
BELLMAN_PSEUDO_LIKELIHOOD_FAILURE = 7


def bellman_filter_status_name(value: int, /) -> BellmanStatus:
    code = int(value)
    names: tuple[BellmanStatus, ...] = (
        "success",
        "initialization_optimizer_failure",
        "initialization_curvature_failure",
        "prediction_optimizer_failure",
        "prediction_curvature_failure",
        "update_optimizer_failure",
        "update_curvature_failure",
        "pseudo_likelihood_failure",
    )
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown Bellman status code {code}.")
    return names[code]


def _state_size(problem: StateSpaceProblem, /) -> int:
    return prod(problem.model.state_shape) if problem.model.state_shape else 1


def _case_count(problem: StateSpaceProblem, /) -> int:
    shape = problem.observations.case_shape
    return prod(shape) if shape else 1


def _validated_configuration(
    *,
    method: BellmanExecutionMethod,
    curvature: BellmanCurvatureMethod,
    curvature_damping: float,
    optimizer_rtol: float,
    optimizer_atol: float,
    optimizer_max_steps: int,
    max_dimension: int,
) -> tuple[
    BellmanExecutionMethod,
    BellmanCurvatureMethod,
    float,
    float,
    float,
    int,
    int,
]:
    if method not in ("auto", "analytic", "optimization"):
        raise ValueError("method must be 'auto', 'analytic', or 'optimization'.")
    if curvature not in ("observed", "score-outer-product"):
        raise ValueError("curvature must be 'observed' or 'score-outer-product'.")
    damping = float(curvature_damping)
    rtol = float(optimizer_rtol)
    atol = float(optimizer_atol)
    steps = int(optimizer_max_steps)
    dimension = int(max_dimension)
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError("curvature_damping must be finite and nonnegative.")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("optimizer_rtol must be finite and positive.")
    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("optimizer_atol must be finite and positive.")
    if steps <= 0:
        raise ValueError("optimizer_max_steps must be positive.")
    if dimension <= 0:
        raise ValueError("max_dimension must be positive.")
    return method, curvature, damping, rtol, atol, steps, dimension


def _configuration(
    problem: StateSpaceProblem,
    /,
    *,
    method: BellmanExecutionMethod,
    curvature: BellmanCurvatureMethod,
    curvature_damping: float,
    optimizer_rtol: float,
    optimizer_atol: float,
    optimizer_max_steps: int,
    max_dimension: int,
) -> tuple[str, str, float, float, float, int, int]:
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    (
        method,
        curvature,
        damping,
        rtol,
        atol,
        steps,
        dimension,
    ) = _validated_configuration(
        method=method,
        curvature=curvature,
        curvature_damping=curvature_damping,
        optimizer_rtol=optimizer_rtol,
        optimizer_atol=optimizer_atol,
        optimizer_max_steps=optimizer_max_steps,
        max_dimension=max_dimension,
    )
    state_dimension = _state_size(problem)
    if state_dimension > dimension:
        raise ValueError(
            f"Bellman filtering requires dense {state_dimension} by {state_dimension} "
            f"curvature, exceeding max_dimension={dimension}."
        )
    if not problem.model.prior.has_log_density:
        raise ValueError("Bellman filtering requires a normalized prior log density.")
    if not problem.model.transition.has_log_density:
        raise ValueError(
            "Bellman filtering requires a normalized transition log density."
        )
    if not jnp.issubdtype(problem.model.prior.location.dtype, jnp.inexact):
        raise TypeError("Bellman filtering requires an inexact continuous state dtype.")
    analytic = (
        isinstance(problem.model.prior, GaussianStatePrior)
        and isinstance(problem.model.transition, LinearGaussianTransitionKernel)
        and isinstance(problem.model.observation, LinearGaussianObservationModel)
        and curvature == "observed"
        and damping == 0.0
    )
    if method == "analytic" and not analytic:
        raise ValueError(
            "method='analytic' requires a nonsingular linear-Gaussian model, "
            "observed curvature, and zero curvature_damping."
        )
    resolved = (
        "analytic"
        if method == "analytic" or (method == "auto" and analytic)
        else "optimization"
    )
    return resolved, curvature, damping, rtol, atol, steps, dimension


def _symmetrize(matrix: Array, /) -> Array:
    return 0.5 * (matrix + matrix.T)


def _positive_information(
    raw_information: Array,
    damping: float,
    /,
) -> tuple[Array, Array, Array, Array]:
    raw = _symmetrize(jnp.asarray(raw_information))
    size = raw.shape[0]
    identity = jnp.eye(size, dtype=raw.dtype)
    raw_eigenvalues = jnp.linalg.eigvalsh(raw)
    raw_minimum = jnp.min(raw_eigenvalues)
    implemented = raw + jnp.asarray(damping, dtype=raw.dtype) * identity
    scale = jnp.linalg.cholesky(implemented)
    diagonal = jnp.diag(scale)
    valid = (
        jnp.all(jnp.isfinite(raw))
        & jnp.isfinite(raw_minimum)
        & jnp.all(jnp.isfinite(scale))
        & jnp.all(diagonal > 0.0)
    )
    safe_information = jnp.where(valid, implemented, identity)
    safe_scale = jnp.linalg.cholesky(safe_information)
    covariance = jax.scipy.linalg.cho_solve((safe_scale, True), identity)
    return safe_information, _symmetrize(covariance), raw_minimum, valid


def _covariance_to_information(
    covariance: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    covariance = _symmetrize(jnp.asarray(covariance))
    size = covariance.shape[0]
    identity = jnp.eye(size, dtype=covariance.dtype)
    scale = jnp.linalg.cholesky(covariance)
    diagonal = jnp.diag(scale)
    valid = (
        jnp.all(jnp.isfinite(covariance))
        & jnp.all(jnp.isfinite(scale))
        & jnp.all(diagonal > 0.0)
    )
    safe_covariance = jnp.where(valid, covariance, identity)
    safe_scale = jnp.linalg.cholesky(safe_covariance)
    information = jax.scipy.linalg.cho_solve((safe_scale, True), identity)
    minimum = jnp.min(jnp.linalg.eigvalsh(_symmetrize(information)))
    return _symmetrize(information), safe_covariance, minimum, valid


def _minimize(
    objective,
    initial: Array,
    /,
    *,
    rtol: float,
    atol: float,
    max_steps: int,
) -> tuple[Array, Array, Array, Array, Array]:
    solver = optx.BFGS(rtol=rtol, atol=atol)
    solution = optx.minimise(
        lambda value, _: objective(value),
        solver,
        initial,
        max_steps=max_steps,
        throw=False,
    )
    value = jnp.asarray(solution.value)
    objective_value = jnp.asarray(objective(value)).reshape(())
    gradient = jax.grad(objective)(value)
    gradient_norm = jnp.linalg.norm(gradient)
    converged = solution.result == optx.RESULTS.successful
    iterations = jnp.asarray(solution.stats["num_steps"], dtype=jnp.int32)
    finite = (
        jnp.all(jnp.isfinite(value))
        & jnp.isfinite(objective_value)
        & jnp.isfinite(gradient_norm)
    )
    return value, objective_value, gradient_norm, iterations, converged & finite


class BellmanFilterState(StrictModule):
    """Streaming posterior-mode state and its implemented local curvature."""

    mode: Array
    information: Array
    covariance: Array
    time: Array
    pseudo_log_likelihood: Array
    mode_valid: Array
    pseudo_likelihood_valid: Array
    status: Array
    step_index: Array
    problem_id: str = eqx.field(static=True)
    execution_method: str = eqx.field(static=True)
    curvature_method: str = eqx.field(static=True)
    curvature_damping: float = eqx.field(static=True)
    optimizer_rtol: float = eqx.field(static=True)
    optimizer_atol: float = eqx.field(static=True)
    optimizer_max_steps: int = eqx.field(static=True)
    max_dimension: int = eqx.field(static=True)


class BellmanFilterStep(StrictModule):
    """One prediction/update record with separate mode and likelihood validity."""

    revised_previous_mode: Array
    predicted_mode: Array
    predicted_information: Array
    predicted_covariance: Array
    filtered_mode: Array
    filtered_information: Array
    filtered_covariance: Array
    transition_matrix: Array
    prediction_objective: Array
    update_objective: Array
    prediction_gradient_norm: Array
    update_gradient_norm: Array
    prediction_iterations: Array
    update_iterations: Array
    prediction_converged: Array
    update_converged: Array
    predicted_raw_minimum_eigenvalue: Array
    filtered_raw_minimum_eigenvalue: Array
    information_gain_minimum_eigenvalue: Array
    observation_log_prob: Array
    realized_kl_penalty: Array
    incremental_pseudo_log_likelihood: Array
    cumulative_pseudo_log_likelihood: Array
    observed_count: Array
    active: Array
    mode_valid: Array
    pseudo_likelihood_valid: Array
    valid: Array
    status: Array


class BellmanFilterResult(StrictModule):
    """Complete Bellman history with optimizer, curvature, and model provenance."""

    revised_previous_modes: Array
    predicted_modes: Array
    predicted_information: Array
    predicted_covariances: Array
    filtered_modes: Array
    filtered_information: Array
    filtered_covariances: Array
    transition_matrices: Array
    prediction_objectives: Array
    update_objectives: Array
    prediction_gradient_norms: Array
    update_gradient_norms: Array
    prediction_iterations: Array
    update_iterations: Array
    prediction_converged: Array
    update_converged: Array
    predicted_raw_minimum_eigenvalues: Array
    filtered_raw_minimum_eigenvalues: Array
    information_gain_minimum_eigenvalues: Array
    observation_log_prob: Array
    realized_kl_penalties: Array
    incremental_pseudo_log_likelihood: Array
    cumulative_pseudo_log_likelihood: Array
    observed_counts: Array
    step_valid: Array
    mode_valid: Array
    pseudo_likelihood_valid: Array
    valid: Array
    status: Array
    times: Array
    final_state: BellmanFilterState
    problem: StateSpaceProblem
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    input_id: str | None = eqx.field(static=True)
    execution_method: str = eqx.field(static=True)
    curvature_method: str = eqx.field(static=True)
    curvature_damping: float = eqx.field(static=True)
    optimizer_rtol: float = eqx.field(static=True)
    optimizer_atol: float = eqx.field(static=True)
    optimizer_max_steps: int = eqx.field(static=True)
    max_dimension: int = eqx.field(static=True)

    @property
    def filter_successful(self) -> Array:
        return jnp.all(self.mode_valid | ~self.step_valid, axis=-1)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


def _initial_analytic_state(
    problem: StateSpaceProblem,
    /,
    *,
    curvature: str,
    damping: float,
    rtol: float,
    atol: float,
    max_steps: int,
    max_dimension: int,
) -> BellmanFilterState:
    kalman = initialize_kalman_filter(problem)
    case_shape = problem.observations.case_shape
    count = _case_count(problem)
    size = _state_size(problem)
    converted = [
        _covariance_to_information(kalman.covariance.reshape((count, size, size))[index])
        for index in range(count)
    ]
    information = jnp.stack([item[0] for item in converted]).reshape(
        case_shape + (size, size)
    )
    covariance = jnp.stack([item[1] for item in converted]).reshape(
        case_shape + (size, size)
    )
    valid = jnp.stack([item[3] for item in converted]).reshape(case_shape)
    status = jnp.where(
        valid, BELLMAN_SUCCESS, BELLMAN_INITIALIZATION_CURVATURE_FAILURE
    ).astype(jnp.int32)
    return BellmanFilterState(
        mode=kalman.mean,
        information=information,
        covariance=covariance,
        time=kalman.time,
        pseudo_log_likelihood=jnp.zeros(case_shape, dtype=kalman.mean.dtype),
        mode_valid=valid,
        pseudo_likelihood_valid=valid,
        status=status,
        step_index=kalman.step_index,
        problem_id=problem.problem_id,
        execution_method="analytic",
        curvature_method=curvature,
        curvature_damping=damping,
        optimizer_rtol=rtol,
        optimizer_atol=atol,
        optimizer_max_steps=max_steps,
        max_dimension=max_dimension,
    )


def _initial_optimization_state(
    problem: StateSpaceProblem,
    /,
    *,
    curvature: str,
    damping: float,
    rtol: float,
    atol: float,
    max_steps: int,
    max_dimension: int,
) -> BellmanFilterState:
    prior = problem.model.prior
    case_shape = problem.observations.case_shape
    count = _case_count(problem)
    size = _state_size(problem)
    state_shape = problem.model.state_shape
    locations = jnp.asarray(prior.location).reshape((count, size))
    modes = []
    information = []
    covariances = []
    validities = []
    statuses = []
    for case_index in range(count):

        def objective(flat_state):
            complete = locations.at[case_index].set(flat_state)
            values = complete.reshape(case_shape + state_shape)
            return -jnp.asarray(prior.log_prob(values)).reshape((count,))[case_index]

        value, _, _, _, converged = _minimize(
            objective,
            locations[case_index],
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
        )
        raw = jax.hessian(objective)(value)
        info, covariance, _, curvature_valid = _positive_information(raw, damping)
        mode_valid = converged & curvature_valid
        modes.append(jnp.where(mode_valid, value, locations[case_index]))
        information.append(info)
        covariances.append(covariance)
        validities.append(mode_valid)
        statuses.append(
            jnp.where(
                ~converged,
                BELLMAN_INITIALIZATION_OPTIMIZER_FAILURE,
                jnp.where(
                    ~curvature_valid,
                    BELLMAN_INITIALIZATION_CURVATURE_FAILURE,
                    BELLMAN_SUCCESS,
                ),
            ).astype(jnp.int32)
        )
    valid = jnp.stack(validities).reshape(case_shape)
    return BellmanFilterState(
        mode=jnp.stack(modes).reshape(case_shape + state_shape),
        information=jnp.stack(information).reshape(case_shape + (size, size)),
        covariance=jnp.stack(covariances).reshape(case_shape + (size, size)),
        time=problem.initial_time,
        pseudo_log_likelihood=jnp.zeros(case_shape, dtype=locations.dtype),
        mode_valid=valid,
        pseudo_likelihood_valid=valid,
        status=jnp.stack(statuses).reshape(case_shape),
        step_index=jnp.asarray(0, dtype=jnp.int32),
        problem_id=problem.problem_id,
        execution_method="optimization",
        curvature_method=curvature,
        curvature_damping=damping,
        optimizer_rtol=rtol,
        optimizer_atol=atol,
        optimizer_max_steps=max_steps,
        max_dimension=max_dimension,
    )


def initialize_bellman_filter(
    problem: StateSpaceProblem,
    /,
    *,
    method: BellmanExecutionMethod = "auto",
    curvature: BellmanCurvatureMethod = "observed",
    curvature_damping: float = 0.0,
    optimizer_rtol: float = 1e-7,
    optimizer_atol: float = 1e-9,
    optimizer_max_steps: int = 128,
    max_dimension: int = 64,
) -> BellmanFilterState:
    """Initialize a deterministic posterior-mode filter from the normalized prior."""
    resolved, curvature_, damping, rtol, atol, steps, dimension = _configuration(
        problem,
        method=method,
        curvature=curvature,
        curvature_damping=curvature_damping,
        optimizer_rtol=optimizer_rtol,
        optimizer_atol=optimizer_atol,
        optimizer_max_steps=optimizer_max_steps,
        max_dimension=max_dimension,
    )
    if resolved == "analytic":
        return _initial_analytic_state(
            problem,
            curvature=curvature_,
            damping=damping,
            rtol=rtol,
            atol=atol,
            max_steps=steps,
            max_dimension=dimension,
        )
    return _initial_optimization_state(
        problem,
        curvature=curvature_,
        damping=damping,
        rtol=rtol,
        atol=atol,
        max_steps=steps,
        max_dimension=dimension,
    )


def _pseudo_likelihood(
    predicted_mode: Array,
    predicted_information: Array,
    filtered_mode: Array,
    filtered_information: Array,
    observation_log_prob: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    displacement = filtered_mode - predicted_mode
    _, predicted_logdet = jnp.linalg.slogdet(predicted_information)
    _, filtered_logdet = jnp.linalg.slogdet(filtered_information)
    quadratic = displacement @ predicted_information @ displacement
    penalty = 0.5 * (filtered_logdet - predicted_logdet + quadratic)
    increment = observation_log_prob - penalty
    gain_minimum = jnp.min(
        jnp.linalg.eigvalsh(_symmetrize(filtered_information - predicted_information))
    )
    dtype = jnp.asarray(increment).dtype
    tolerance = (
        100.0 * jnp.finfo(dtype).eps * jnp.maximum(1.0, jnp.abs(observation_log_prob))
    )
    valid = (
        jnp.isfinite(observation_log_prob)
        & jnp.isfinite(penalty)
        & jnp.isfinite(increment)
        & (penalty >= -tolerance)
    )
    return penalty, increment, gain_minimum, valid


def _analytic_bellman_filter_step(
    problem: StateSpaceProblem,
    state: BellmanFilterState,
    /,
) -> tuple[BellmanFilterState, BellmanFilterStep]:
    case_shape = problem.observations.case_shape
    count = _case_count(problem)
    size = _state_size(problem)
    kalman_state = KalmanFilterState(
        mean=state.mode,
        covariance=state.covariance,
        time=state.time,
        log_likelihood=state.pseudo_log_likelihood,
        valid=state.mode_valid,
        status=jnp.zeros(case_shape, dtype=jnp.int32),
        step_index=state.step_index,
        problem_id=problem.problem_id,
        covariance_regularization=0.0,
    )
    kalman_next, record = kalman_filter_step(problem, kalman_state)
    predicted_converted = [
        _covariance_to_information(
            record.predicted_covariance.reshape((count, size, size))[index]
        )
        for index in range(count)
    ]
    filtered_converted = [
        _covariance_to_information(
            record.filtered_covariance.reshape((count, size, size))[index]
        )
        for index in range(count)
    ]
    predicted_information = jnp.stack([item[0] for item in predicted_converted])
    predicted_covariance = jnp.stack([item[1] for item in predicted_converted])
    predicted_minimum = jnp.stack([item[2] for item in predicted_converted])
    predicted_valid = jnp.stack([item[3] for item in predicted_converted])
    filtered_information = jnp.stack([item[0] for item in filtered_converted])
    filtered_covariance = jnp.stack([item[1] for item in filtered_converted])
    filtered_minimum = jnp.stack([item[2] for item in filtered_converted])
    filtered_valid = jnp.stack([item[3] for item in filtered_converted])
    index = state.step_index
    sequence = problem.observations
    active = sequence.step_valid[..., index]
    target_time = sequence.times[..., index]
    step_axis = len(case_shape)
    values = jnp.take(sequence.values, index, axis=step_axis).reshape(
        (count,) + problem.model.observation_shape
    )
    masks = jnp.take(sequence.observation_mask, index, axis=step_axis).reshape(
        (count,) + problem.model.observation_shape
    )
    filtered_modes = record.filtered_mean.reshape((count, size))
    predicted_modes = record.predicted_mean.reshape((count, size))
    observations = []
    penalties = []
    increments = []
    gain_minima = []
    pseudo_validities = []
    observed_counts = []
    for case_index in range(count):
        context = problem.step_context(case_index, index)
        mask = masks[case_index] & active.reshape((count,))[case_index]
        observation_value = problem.model.observation.log_prob(
            values[case_index],
            filtered_modes[case_index].reshape(problem.model.state_shape),
            target_time.reshape((count,))[case_index],
            mask,
            context,
        ).reshape(())
        penalty, increment, gain_minimum, pseudo_valid = _pseudo_likelihood(
            predicted_modes[case_index],
            predicted_information[case_index],
            filtered_modes[case_index],
            filtered_information[case_index],
            observation_value,
        )
        observations.append(
            jnp.where(active.reshape((count,))[case_index], observation_value, 0.0)
        )
        penalties.append(jnp.where(active.reshape((count,))[case_index], penalty, 0.0))
        increments.append(jnp.where(active.reshape((count,))[case_index], increment, 0.0))
        gain_minima.append(gain_minimum)
        pseudo_validities.append(
            jnp.where(active.reshape((count,))[case_index], pseudo_valid, True)
        )
        observed_counts.append(jnp.sum(mask))
    observation_values = jnp.stack(observations)
    penalty_values = jnp.stack(penalties)
    increment_values = jnp.stack(increments)
    gain_minimum_values = jnp.stack(gain_minima)
    local_pseudo_valid = jnp.stack(pseudo_validities).reshape(case_shape)
    mode_valid = (
        kalman_next.valid
        & predicted_valid.reshape(case_shape)
        & filtered_valid.reshape(case_shape)
    )
    cumulative_pseudo_valid = (
        state.pseudo_likelihood_valid & local_pseudo_valid & mode_valid
    )
    accepted_increment = jnp.where(
        active.reshape((count,)) & cumulative_pseudo_valid.reshape((count,)),
        increment_values,
        0.0,
    )
    cumulative = jnp.where(
        cumulative_pseudo_valid.reshape((count,)),
        state.pseudo_log_likelihood.reshape((count,)) + accepted_increment,
        -jnp.inf,
    )
    status = jnp.where(
        ~mode_valid,
        BELLMAN_UPDATE_CURVATURE_FAILURE,
        jnp.where(
            ~cumulative_pseudo_valid,
            BELLMAN_PSEUDO_LIKELIHOOD_FAILURE,
            BELLMAN_SUCCESS,
        ),
    ).astype(jnp.int32)
    next_state = BellmanFilterState(
        mode=kalman_next.mean,
        information=filtered_information.reshape(case_shape + (size, size)),
        covariance=filtered_covariance.reshape(case_shape + (size, size)),
        time=kalman_next.time,
        pseudo_log_likelihood=cumulative.reshape(case_shape),
        mode_valid=mode_valid,
        pseudo_likelihood_valid=cumulative_pseudo_valid,
        status=status,
        step_index=kalman_next.step_index,
        problem_id=state.problem_id,
        execution_method=state.execution_method,
        curvature_method=state.curvature_method,
        curvature_damping=state.curvature_damping,
        optimizer_rtol=state.optimizer_rtol,
        optimizer_atol=state.optimizer_atol,
        optimizer_max_steps=state.optimizer_max_steps,
        max_dimension=state.max_dimension,
    )
    record_ = BellmanFilterStep(
        revised_previous_mode=state.mode,
        predicted_mode=record.predicted_mean,
        predicted_information=predicted_information.reshape(case_shape + (size, size)),
        predicted_covariance=predicted_covariance.reshape(case_shape + (size, size)),
        filtered_mode=next_state.mode,
        filtered_information=next_state.information,
        filtered_covariance=next_state.covariance,
        transition_matrix=record.transition_matrix,
        prediction_objective=jnp.zeros(case_shape, dtype=state.mode.dtype),
        update_objective=jnp.zeros(case_shape, dtype=state.mode.dtype),
        prediction_gradient_norm=jnp.zeros(case_shape, dtype=state.mode.dtype),
        update_gradient_norm=jnp.zeros(case_shape, dtype=state.mode.dtype),
        prediction_iterations=jnp.zeros(case_shape, dtype=jnp.int32),
        update_iterations=jnp.zeros(case_shape, dtype=jnp.int32),
        prediction_converged=mode_valid,
        update_converged=mode_valid,
        predicted_raw_minimum_eigenvalue=predicted_minimum.reshape(case_shape),
        filtered_raw_minimum_eigenvalue=filtered_minimum.reshape(case_shape),
        information_gain_minimum_eigenvalue=gain_minimum_values.reshape(case_shape),
        observation_log_prob=observation_values.reshape(case_shape),
        realized_kl_penalty=penalty_values.reshape(case_shape),
        incremental_pseudo_log_likelihood=accepted_increment.reshape(case_shape),
        cumulative_pseudo_log_likelihood=cumulative.reshape(case_shape),
        observed_count=jnp.stack(observed_counts).reshape(case_shape),
        active=active,
        mode_valid=mode_valid,
        pseudo_likelihood_valid=cumulative_pseudo_valid,
        valid=mode_valid & cumulative_pseudo_valid,
        status=status,
    )
    return next_state, record_


def _optimization_case_step(
    problem: StateSpaceProblem,
    state: BellmanFilterState,
    case_index: int,
    target_time: Array,
    observation_value: Array,
    observation_mask: Array,
    active: Array,
    /,
):
    size = _state_size(problem)
    state_shape = problem.model.state_shape
    count = _case_count(problem)
    previous_mode = state.mode.reshape((count, size))[case_index]
    previous_information = state.information.reshape((count, size, size))[case_index]
    previous_covariance = state.covariance.reshape((count, size, size))[case_index]
    previous_time = state.time.reshape((count,))[case_index]
    previous_mode_valid = state.mode_valid.reshape((count,))[case_index]
    previous_pseudo_valid = state.pseudo_likelihood_valid.reshape((count,))[case_index]
    previous_pseudo = state.pseudo_log_likelihood.reshape((count,))[case_index]
    previous_status = state.status.reshape((count,))[case_index]
    context = problem.step_context(case_index, state.step_index)
    transition = problem.model.transition
    observation = problem.model.observation
    identity = jnp.eye(size, dtype=previous_mode.dtype)

    def inactive(_):
        return (
            previous_mode,
            previous_mode,
            previous_information,
            previous_covariance,
            previous_mode,
            previous_information,
            previous_covariance,
            identity,
            jnp.asarray(0.0, dtype=previous_mode.dtype),
            jnp.asarray(0.0, dtype=previous_mode.dtype),
            jnp.asarray(0.0, dtype=previous_mode.dtype),
            jnp.asarray(0.0, dtype=previous_mode.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            previous_mode_valid,
            previous_mode_valid,
            jnp.min(jnp.linalg.eigvalsh(previous_information)),
            jnp.min(jnp.linalg.eigvalsh(previous_information)),
            jnp.asarray(0.0, dtype=previous_mode.dtype),
            jnp.asarray(0.0, dtype=previous_mode.dtype),
            jnp.asarray(0.0, dtype=previous_mode.dtype),
            jnp.asarray(0.0, dtype=previous_mode.dtype),
            previous_pseudo,
            jnp.asarray(0, dtype=jnp.int32),
            previous_mode_valid,
            previous_pseudo_valid,
            previous_mode_valid & previous_pseudo_valid,
            jnp.where(
                previous_mode_valid & previous_pseudo_valid,
                BELLMAN_SUCCESS,
                previous_status,
            ).astype(jnp.int32),
            previous_time,
        )

    def active_step(_):
        def prediction_objective(joint):
            prior_state = joint[:size]
            next_state = joint[size:]
            displacement = prior_state - previous_mode
            prior_value = 0.5 * displacement @ previous_information @ displacement
            transition_value = transition.log_prob(
                next_state.reshape(state_shape),
                prior_state.reshape(state_shape),
                previous_time,
                target_time,
                context,
            ).reshape(())
            return prior_value - transition_value

        joint_initial = jnp.concatenate([previous_mode, previous_mode])
        (
            joint_mode,
            prediction_value,
            prediction_gradient,
            prediction_iterations,
            prediction_converged,
        ) = _minimize(
            prediction_objective,
            joint_initial,
            rtol=state.optimizer_rtol,
            atol=state.optimizer_atol,
            max_steps=state.optimizer_max_steps,
        )
        joint_hessian = _symmetrize(jax.hessian(prediction_objective)(joint_mode))
        previous_block = _symmetrize(joint_hessian[:size, :size])
        previous_scale = jnp.linalg.cholesky(previous_block)
        previous_diagonal = jnp.diag(previous_scale)
        profile_valid = jnp.all(jnp.isfinite(previous_scale)) & jnp.all(
            previous_diagonal > 0.0
        )
        safe_previous = jnp.where(profile_valid, previous_block, identity)
        cross = joint_hessian[:size, size:]
        profiled = joint_hessian[size:, size:] - cross.T @ jnp.linalg.solve(
            safe_previous, cross
        )
        (
            predicted_information,
            predicted_covariance,
            predicted_minimum,
            predicted_curvature_valid,
        ) = _positive_information(profiled, state.curvature_damping)
        predicted_mode = joint_mode[size:]
        revised_previous = joint_mode[:size]
        prediction_valid = (
            prediction_converged
            & profile_valid
            & predicted_curvature_valid
            & jnp.all(jnp.isfinite(predicted_mode))
        )

        observed_count = jnp.sum(observation_mask)

        def missing_update(_):
            return (
                predicted_mode,
                predicted_information,
                predicted_covariance,
                jnp.asarray(0.0, dtype=predicted_mode.dtype),
                jnp.asarray(0.0, dtype=predicted_mode.dtype),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(True),
                prediction_valid,
                predicted_minimum,
                jnp.asarray(0.0, dtype=predicted_mode.dtype),
            )

        def observed_update(_):
            def update_objective(flat_state):
                displacement = flat_state - predicted_mode
                prediction_value_ = (
                    0.5 * displacement @ predicted_information @ displacement
                )
                observation_value_ = observation.log_prob(
                    observation_value,
                    flat_state.reshape(state_shape),
                    target_time,
                    observation_mask,
                    context,
                ).reshape(())
                return prediction_value_ - observation_value_

            (
                filtered_mode,
                update_value,
                update_gradient,
                update_iterations,
                update_converged,
            ) = _minimize(
                update_objective,
                predicted_mode,
                rtol=state.optimizer_rtol,
                atol=state.optimizer_atol,
                max_steps=state.optimizer_max_steps,
            )
            if state.curvature_method == "observed":
                raw_filtered = jax.hessian(update_objective)(filtered_mode)
            else:
                score = jax.grad(
                    lambda flat_state: observation.log_prob(
                        observation_value,
                        flat_state.reshape(state_shape),
                        target_time,
                        observation_mask,
                        context,
                    ).reshape(())
                )(filtered_mode)
                raw_filtered = predicted_information + jnp.outer(score, score)
            (
                filtered_information,
                filtered_covariance,
                filtered_minimum,
                filtered_curvature_valid,
            ) = _positive_information(raw_filtered, state.curvature_damping)
            update_valid = (
                prediction_valid
                & update_converged
                & filtered_curvature_valid
                & jnp.all(jnp.isfinite(filtered_mode))
            )
            return (
                filtered_mode,
                filtered_information,
                filtered_covariance,
                update_value,
                update_gradient,
                update_iterations,
                update_converged,
                update_valid,
                filtered_minimum,
                observation.log_prob(
                    observation_value,
                    filtered_mode.reshape(state_shape),
                    target_time,
                    observation_mask,
                    context,
                ).reshape(()),
            )

        (
            filtered_mode,
            filtered_information,
            filtered_covariance,
            update_value,
            update_gradient,
            update_iterations,
            update_converged,
            update_valid,
            filtered_minimum,
            observation_log_prob,
        ) = jax.lax.cond(
            observed_count == 0, missing_update, observed_update, operand=None
        )
        penalty, increment, information_gain_minimum, local_pseudo_valid = (
            _pseudo_likelihood(
                predicted_mode,
                predicted_information,
                filtered_mode,
                filtered_information,
                observation_log_prob,
            )
        )
        mode_valid = prediction_valid & update_valid
        pseudo_valid = previous_pseudo_valid & mode_valid & local_pseudo_valid
        cumulative = jnp.where(pseudo_valid, previous_pseudo + increment, -jnp.inf)
        accepted_mode = jnp.where(mode_valid, filtered_mode, previous_mode)
        accepted_information = jnp.where(
            mode_valid, filtered_information, previous_information
        )
        accepted_covariance = jnp.where(
            mode_valid, filtered_covariance, previous_covariance
        )
        if isinstance(transition, LinearGaussianTransitionKernel):
            transition_matrix = transition.parameters(
                previous_time, target_time, context
            ).transition
        else:
            transition_matrix = jnp.zeros((size, size), dtype=previous_mode.dtype)
        status = jnp.where(
            ~prediction_converged,
            BELLMAN_PREDICTION_OPTIMIZER_FAILURE,
            jnp.where(
                ~prediction_valid,
                BELLMAN_PREDICTION_CURVATURE_FAILURE,
                jnp.where(
                    ~update_converged,
                    BELLMAN_UPDATE_OPTIMIZER_FAILURE,
                    jnp.where(
                        ~update_valid,
                        BELLMAN_UPDATE_CURVATURE_FAILURE,
                        jnp.where(
                            ~pseudo_valid,
                            BELLMAN_PSEUDO_LIKELIHOOD_FAILURE,
                            BELLMAN_SUCCESS,
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return (
            revised_previous,
            predicted_mode,
            predicted_information,
            predicted_covariance,
            accepted_mode,
            accepted_information,
            accepted_covariance,
            transition_matrix,
            prediction_value,
            update_value,
            prediction_gradient,
            update_gradient,
            prediction_iterations,
            update_iterations,
            prediction_converged,
            update_converged,
            predicted_minimum,
            filtered_minimum,
            information_gain_minimum,
            observation_log_prob,
            penalty,
            jnp.where(pseudo_valid, increment, 0.0),
            cumulative,
            observed_count.astype(jnp.int32),
            mode_valid,
            pseudo_valid,
            mode_valid & pseudo_valid,
            status,
            target_time,
        )

    return jax.lax.cond(active & previous_mode_valid, active_step, inactive, operand=None)


def _optimization_bellman_filter_step(
    problem: StateSpaceProblem,
    state: BellmanFilterState,
    /,
) -> tuple[BellmanFilterState, BellmanFilterStep]:
    case_shape = problem.observations.case_shape
    count = _case_count(problem)
    size = _state_size(problem)
    sequence = problem.observations
    index = state.step_index
    active = sequence.step_valid[..., index]
    target_times = sequence.times[..., index]
    step_axis = len(case_shape)
    values = jnp.take(sequence.values, index, axis=step_axis).reshape(
        (count,) + problem.model.observation_shape
    )
    masks = jnp.take(sequence.observation_mask, index, axis=step_axis).reshape(
        (count,) + problem.model.observation_shape
    )
    outputs = [
        _optimization_case_step(
            problem,
            state,
            case_index,
            target_times.reshape((count,))[case_index],
            values[case_index],
            masks[case_index],
            active.reshape((count,))[case_index],
        )
        for case_index in range(count)
    ]
    stacked = tuple(
        jnp.stack([output[field] for output in outputs]) for field in range(29)
    )
    next_state = BellmanFilterState(
        mode=stacked[4].reshape(case_shape + problem.model.state_shape),
        information=stacked[5].reshape(case_shape + (size, size)),
        covariance=stacked[6].reshape(case_shape + (size, size)),
        time=stacked[28].reshape(case_shape),
        pseudo_log_likelihood=stacked[22].reshape(case_shape),
        mode_valid=stacked[24].reshape(case_shape),
        pseudo_likelihood_valid=stacked[25].reshape(case_shape),
        status=stacked[27].reshape(case_shape),
        step_index=index + 1,
        problem_id=state.problem_id,
        execution_method=state.execution_method,
        curvature_method=state.curvature_method,
        curvature_damping=state.curvature_damping,
        optimizer_rtol=state.optimizer_rtol,
        optimizer_atol=state.optimizer_atol,
        optimizer_max_steps=state.optimizer_max_steps,
        max_dimension=state.max_dimension,
    )
    record = BellmanFilterStep(
        revised_previous_mode=stacked[0].reshape(case_shape + problem.model.state_shape),
        predicted_mode=stacked[1].reshape(case_shape + problem.model.state_shape),
        predicted_information=stacked[2].reshape(case_shape + (size, size)),
        predicted_covariance=stacked[3].reshape(case_shape + (size, size)),
        filtered_mode=next_state.mode,
        filtered_information=next_state.information,
        filtered_covariance=next_state.covariance,
        transition_matrix=stacked[7].reshape(case_shape + (size, size)),
        prediction_objective=stacked[8].reshape(case_shape),
        update_objective=stacked[9].reshape(case_shape),
        prediction_gradient_norm=stacked[10].reshape(case_shape),
        update_gradient_norm=stacked[11].reshape(case_shape),
        prediction_iterations=stacked[12].reshape(case_shape),
        update_iterations=stacked[13].reshape(case_shape),
        prediction_converged=stacked[14].reshape(case_shape),
        update_converged=stacked[15].reshape(case_shape),
        predicted_raw_minimum_eigenvalue=stacked[16].reshape(case_shape),
        filtered_raw_minimum_eigenvalue=stacked[17].reshape(case_shape),
        information_gain_minimum_eigenvalue=stacked[18].reshape(case_shape),
        observation_log_prob=stacked[19].reshape(case_shape),
        realized_kl_penalty=stacked[20].reshape(case_shape),
        incremental_pseudo_log_likelihood=stacked[21].reshape(case_shape),
        cumulative_pseudo_log_likelihood=stacked[22].reshape(case_shape),
        observed_count=stacked[23].reshape(case_shape),
        active=active,
        mode_valid=stacked[24].reshape(case_shape),
        pseudo_likelihood_valid=stacked[25].reshape(case_shape),
        valid=stacked[26].reshape(case_shape),
        status=stacked[27].reshape(case_shape),
    )
    return next_state, record


def bellman_filter_step(
    problem: StateSpaceProblem,
    state: BellmanFilterState,
    /,
) -> tuple[BellmanFilterState, BellmanFilterStep]:
    """Process one canonical observation step without changing case semantics."""
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    if not isinstance(state, BellmanFilterState):
        raise TypeError("state must be a BellmanFilterState.")
    if state.problem_id != problem.problem_id:
        raise ValueError("Bellman state and problem IDs do not match.")
    index = eqx.error_if(
        state.step_index,
        state.step_index >= problem.observations.num_steps,
        "The Bellman state has already consumed every observation step.",
    )
    state = eqx.tree_at(lambda item: item.step_index, state, index)
    if state.execution_method == "analytic":
        return _analytic_bellman_filter_step(problem, state)
    return _optimization_bellman_filter_step(problem, state)


def _stack_steps(
    records: list[BellmanFilterStep], case_rank: int, /
) -> BellmanFilterStep:
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *records)
    return jax.tree_util.tree_map(
        lambda value: jnp.moveaxis(value, 0, case_rank), stacked
    )


def bellman_filter(
    problem: StateSpaceProblem,
    /,
    *,
    method: BellmanExecutionMethod = "auto",
    curvature: BellmanCurvatureMethod = "observed",
    curvature_damping: float = 0.0,
    optimizer_rtol: float = 1e-7,
    optimizer_atol: float = 1e-9,
    optimizer_max_steps: int = 128,
    max_dimension: int = 64,
    raise_on_failure: bool = False,
) -> BellmanFilterResult:
    """Run dense posterior-mode filtering and an explicitly named pseudo-likelihood."""
    state = initialize_bellman_filter(
        problem,
        method=method,
        curvature=curvature,
        curvature_damping=curvature_damping,
        optimizer_rtol=optimizer_rtol,
        optimizer_atol=optimizer_atol,
        optimizer_max_steps=optimizer_max_steps,
        max_dimension=max_dimension,
    )
    records = []
    for _ in range(problem.observations.num_steps):
        state, record = bellman_filter_step(problem, state)
        records.append(record)
    stacked = _stack_steps(records, len(problem.observations.case_shape))
    result = BellmanFilterResult(
        revised_previous_modes=stacked.revised_previous_mode,
        predicted_modes=stacked.predicted_mode,
        predicted_information=stacked.predicted_information,
        predicted_covariances=stacked.predicted_covariance,
        filtered_modes=stacked.filtered_mode,
        filtered_information=stacked.filtered_information,
        filtered_covariances=stacked.filtered_covariance,
        transition_matrices=stacked.transition_matrix,
        prediction_objectives=stacked.prediction_objective,
        update_objectives=stacked.update_objective,
        prediction_gradient_norms=stacked.prediction_gradient_norm,
        update_gradient_norms=stacked.update_gradient_norm,
        prediction_iterations=stacked.prediction_iterations,
        update_iterations=stacked.update_iterations,
        prediction_converged=stacked.prediction_converged,
        update_converged=stacked.update_converged,
        predicted_raw_minimum_eigenvalues=stacked.predicted_raw_minimum_eigenvalue,
        filtered_raw_minimum_eigenvalues=stacked.filtered_raw_minimum_eigenvalue,
        information_gain_minimum_eigenvalues=stacked.information_gain_minimum_eigenvalue,
        observation_log_prob=stacked.observation_log_prob,
        realized_kl_penalties=stacked.realized_kl_penalty,
        incremental_pseudo_log_likelihood=stacked.incremental_pseudo_log_likelihood,
        cumulative_pseudo_log_likelihood=stacked.cumulative_pseudo_log_likelihood,
        observed_counts=stacked.observed_count,
        step_valid=problem.observations.step_valid,
        mode_valid=stacked.mode_valid,
        pseudo_likelihood_valid=stacked.pseudo_likelihood_valid,
        valid=stacked.valid,
        status=stacked.status,
        times=problem.observations.times,
        final_state=state,
        problem=problem,
        state_shape=problem.model.state_shape,
        observation_shape=problem.model.observation_shape,
        case_shape=problem.observations.case_shape,
        case_axes=problem.observations.case_axes,
        case_ids=problem.observations.case_ids,
        model_id=problem.model.model_id,
        problem_id=problem.problem_id,
        sequence_id=problem.observations.sequence_id,
        input_id=None if problem.input_signal is None else problem.input_signal.input_id,
        execution_method=state.execution_method,
        curvature_method=state.curvature_method,
        curvature_damping=state.curvature_damping,
        optimizer_rtol=state.optimizer_rtol,
        optimizer_atol=state.optimizer_atol,
        optimizer_max_steps=state.optimizer_max_steps,
        max_dimension=state.max_dimension,
    )
    if raise_on_failure and not bool(jnp.all(result.successful)):
        raise RuntimeError("Bellman filtering failed for at least one physical case.")
    return result


class StateSpaceLaplaceLikelihood(StrictModule):
    """Configured Bellman pseudo-likelihood backend for state-space experiments."""

    method: BellmanExecutionMethod = eqx.field(static=True)
    curvature: BellmanCurvatureMethod = eqx.field(static=True)
    curvature_damping: float = eqx.field(static=True)
    optimizer_rtol: float = eqx.field(static=True)
    optimizer_atol: float = eqx.field(static=True)
    optimizer_max_steps: int = eqx.field(static=True)
    max_dimension: int = eqx.field(static=True)
    raise_on_failure: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        method: BellmanExecutionMethod = "auto",
        curvature: BellmanCurvatureMethod = "observed",
        curvature_damping: float = 0.0,
        optimizer_rtol: float = 1e-9,
        optimizer_atol: float = 1e-11,
        optimizer_max_steps: int = 128,
        max_dimension: int = 128,
        raise_on_failure: bool = False,
    ):
        (
            method,
            curvature,
            damping,
            rtol,
            atol,
            steps,
            dimension,
        ) = _validated_configuration(
            method=method,
            curvature=curvature,
            curvature_damping=curvature_damping,
            optimizer_rtol=optimizer_rtol,
            optimizer_atol=optimizer_atol,
            optimizer_max_steps=optimizer_max_steps,
            max_dimension=max_dimension,
        )
        if not isinstance(raise_on_failure, bool):
            raise TypeError("raise_on_failure must be a bool.")
        self.method = method
        self.curvature = curvature
        self.curvature_damping = damping
        self.optimizer_rtol = rtol
        self.optimizer_atol = atol
        self.optimizer_max_steps = steps
        self.max_dimension = dimension
        self.raise_on_failure = raise_on_failure

    def __call__(self, problem: StateSpaceProblem, /) -> BellmanFilterResult:
        """Evaluate the configured deterministic local-Laplace filter."""
        return bellman_filter(
            problem,
            method=self.method,
            curvature=self.curvature,
            curvature_damping=self.curvature_damping,
            optimizer_rtol=self.optimizer_rtol,
            optimizer_atol=self.optimizer_atol,
            optimizer_max_steps=self.optimizer_max_steps,
            max_dimension=self.max_dimension,
            raise_on_failure=self.raise_on_failure,
        )


class BellmanSmootherResult(StrictModule):
    """RTS-form local Gaussian smoothing for affine Gaussian transitions."""

    modes: Array
    covariances: Array
    gains: Array
    lag_one_covariances: Array
    valid: Array
    filter_result: BellmanFilterResult
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.filter_result.step_valid, axis=-1)


def bellman_smoother(result: BellmanFilterResult, /) -> BellmanSmootherResult:
    """Apply RTS recursion to Bellman local moments for a linear Gaussian transition."""
    if not isinstance(result, BellmanFilterResult):
        raise TypeError("result must be a BellmanFilterResult.")
    if not isinstance(result.problem.model.transition, LinearGaussianTransitionKernel):
        raise ValueError(
            "Bellman smoothing requires a state-independent affine Gaussian transition."
        )
    case_shape = result.case_shape
    count = prod(case_shape) if case_shape else 1
    steps = result.step_valid.shape[-1]
    size = prod(result.state_shape) if result.state_shape else 1
    filtered_modes = result.filtered_modes.reshape((count, steps, size))
    filtered_covariances = result.filtered_covariances.reshape((count, steps, size, size))
    predicted_modes = result.predicted_modes.reshape((count, steps, size))
    predicted_covariances = result.predicted_covariances.reshape(
        (count, steps, size, size)
    )
    transitions = result.transition_matrices.reshape((count, steps, size, size))
    valid = (result.mode_valid & result.step_valid).reshape((count, steps))
    modes = filtered_modes
    covariances = filtered_covariances
    gains = jnp.zeros((count, max(steps - 1, 0), size, size), dtype=filtered_modes.dtype)
    lag_one = jnp.zeros_like(gains)
    for index in range(steps - 2, -1, -1):
        cross = filtered_covariances[:, index] @ jnp.swapaxes(
            transitions[:, index + 1], -1, -2
        )
        gain = jnp.swapaxes(
            jnp.linalg.solve(
                predicted_covariances[:, index + 1],
                jnp.swapaxes(cross, -1, -2),
            ),
            -1,
            -2,
        )
        pair_valid = valid[:, index] & valid[:, index + 1]
        proposed_mode = filtered_modes[:, index] + ein.contract(
            "cij,cj->ci", gain, modes[:, index + 1] - predicted_modes[:, index + 1]
        )
        proposed_covariance = filtered_covariances[:, index] + gain @ (
            covariances[:, index + 1] - predicted_covariances[:, index + 1]
        ) @ jnp.swapaxes(gain, -1, -2)
        proposed_covariance = 0.5 * (
            proposed_covariance + jnp.swapaxes(proposed_covariance, -1, -2)
        )
        modes = modes.at[:, index].set(
            jnp.where(pair_valid[:, None], proposed_mode, modes[:, index])
        )
        covariances = covariances.at[:, index].set(
            jnp.where(
                pair_valid[:, None, None], proposed_covariance, covariances[:, index]
            )
        )
        gains = gains.at[:, index].set(jnp.where(pair_valid[:, None, None], gain, 0.0))
        next_previous = covariances[:, index + 1] @ jnp.swapaxes(gain, -1, -2)
        lag_one = lag_one.at[:, index].set(
            jnp.where(pair_valid[:, None, None], next_previous, 0.0)
        )
    return BellmanSmootherResult(
        modes=modes.reshape(case_shape + (steps,) + result.state_shape),
        covariances=covariances.reshape(case_shape + (steps, size, size)),
        gains=gains.reshape(case_shape + (max(steps - 1, 0), size, size)),
        lag_one_covariances=lag_one.reshape(case_shape + (max(steps - 1, 0), size, size)),
        valid=valid.reshape(case_shape + (steps,)),
        filter_result=result,
        method_id="bellman-rts-local-gaussian",
    )


__all__ = [
    "BELLMAN_INITIALIZATION_CURVATURE_FAILURE",
    "BELLMAN_INITIALIZATION_OPTIMIZER_FAILURE",
    "BELLMAN_PREDICTION_CURVATURE_FAILURE",
    "BELLMAN_PREDICTION_OPTIMIZER_FAILURE",
    "BELLMAN_PSEUDO_LIKELIHOOD_FAILURE",
    "BELLMAN_SUCCESS",
    "BELLMAN_UPDATE_CURVATURE_FAILURE",
    "BELLMAN_UPDATE_OPTIMIZER_FAILURE",
    "BellmanCurvatureMethod",
    "BellmanExecutionMethod",
    "BellmanFilterResult",
    "BellmanFilterState",
    "BellmanFilterStep",
    "BellmanSmootherResult",
    "BellmanStatus",
    "StateSpaceLaplaceLikelihood",
    "bellman_filter",
    "bellman_filter_status_name",
    "bellman_filter_step",
    "bellman_smoother",
    "initialize_bellman_filter",
]
