#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._frozendict import frozendict
from .._strict import StrictModule
from ..stochastic._state_space import (
    _state_space_input_validity,
    _state_space_step_context,
    AbstractStatePrior,
    AbstractTransitionKernel,
    ObservationSequence,
    state_space_key,
    StateSpaceStepContext,
)
from ..stochastic._state_space_input import AbstractStateSpaceInput
from ._conditional_moments import _condition_affine_gaussian_diagonal
from ._covariance import DiagonalCovariance
from ._particle import (
    effective_sample_size,
    normalize_log_weights,
    PARTICLE_FILTER_NONFINITE,
    PARTICLE_FILTER_SUCCESS,
    PARTICLE_FILTER_TRANSITION_FAILURE,
    PARTICLE_FILTER_WEIGHT_DEGENERACY,
    resample_indices,
    ResamplingMethod,
    ResamplingPolicy,
)


InitialLinearGaussian: TypeAlias = Callable[[Array, Any], tuple[ArrayLike, ArrayLike]]
ConditionalLinearTransition: TypeAlias = Callable[
    [Array, Array, Array, Array, StateSpaceStepContext],
    tuple[ArrayLike, ArrayLike, ArrayLike],
]
ConditionalLinearObservation: TypeAlias = Callable[
    [Array, Array, StateSpaceStepContext],
    tuple[ArrayLike, ArrayLike, ArrayLike | DiagonalCovariance],
]


class RaoBlackwellizedStateSpaceModel(StrictModule):
    """Nonlinear Markov state coupled to a conditionally linear Gaussian state.

    The callbacks return normalized Gaussian parameters:

    - ``initial_linear_gaussian(mode, args) -> (mean, covariance)``;
    - ``linear_transition(previous_mode, mode, t0, t1, context) -> (A, b, Q)``;
    - ``observation(mode, time, context) -> (H, d, R)``, where ``R`` may be a
      dense array or :class:`DiagonalCovariance`.
    """

    nonlinear_prior: AbstractStatePrior
    nonlinear_transition: AbstractTransitionKernel
    initial_linear_gaussian_fn: InitialLinearGaussian = eqx.field(static=True)
    linear_transition_fn: ConditionalLinearTransition = eqx.field(static=True)
    observation_fn: ConditionalLinearObservation = eqx.field(static=True)
    metadata: frozendict[str, Any]
    nonlinear_state_shape: tuple[int, ...] = eqx.field(static=True)
    linear_state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        nonlinear_prior: AbstractStatePrior,
        nonlinear_transition: AbstractTransitionKernel,
        initial_linear_gaussian: InitialLinearGaussian,
        linear_transition: ConditionalLinearTransition,
        observation: ConditionalLinearObservation,
        /,
        *,
        linear_state_shape: Sequence[int],
        observation_shape: Sequence[int],
        model_id: str,
        metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(nonlinear_prior, AbstractStatePrior):
            raise TypeError("nonlinear_prior must implement AbstractStatePrior.")
        if not isinstance(nonlinear_transition, AbstractTransitionKernel):
            raise TypeError(
                "nonlinear_transition must implement AbstractTransitionKernel."
            )
        if nonlinear_prior.state_shape != nonlinear_transition.state_shape:
            raise ValueError("Nonlinear prior and transition state shapes must agree.")
        for name, function in (
            ("initial_linear_gaussian", initial_linear_gaussian),
            ("linear_transition", linear_transition),
            ("observation", observation),
        ):
            if not callable(function):
                raise TypeError(f"{name} must be callable.")
        linear_shape = tuple(int(size) for size in linear_state_shape)
        observed_shape = tuple(int(size) for size in observation_shape)
        if any(size <= 0 for size in linear_shape):
            raise ValueError("linear_state_shape dimensions must be positive.")
        if any(size <= 0 for size in observed_shape):
            raise ValueError("observation_shape dimensions must be positive.")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string.")
        self.nonlinear_prior = nonlinear_prior
        self.nonlinear_transition = nonlinear_transition
        self.initial_linear_gaussian_fn = initial_linear_gaussian
        self.linear_transition_fn = linear_transition
        self.observation_fn = observation
        self.metadata = frozendict({} if metadata is None else metadata)
        self.nonlinear_state_shape = nonlinear_prior.state_shape
        self.linear_state_shape = linear_shape
        self.observation_shape = observed_shape
        self.model_id = model_id

    def initial_linear_gaussian(
        self, nonlinear_state: ArrayLike, args: Any = None, /
    ) -> tuple[Array, Array]:
        mean, covariance = self.initial_linear_gaussian_fn(
            jnp.asarray(nonlinear_state), args
        )
        size = prod(self.linear_state_shape) if self.linear_state_shape else 1
        mean_array = jnp.asarray(mean, dtype=float)
        covariance_array = jnp.asarray(covariance, dtype=float)
        if mean_array.shape != self.linear_state_shape:
            raise ValueError("Initial linear mean must have shape linear_state_shape.")
        if covariance_array.shape != (size, size):
            raise ValueError(
                "Initial linear covariance must have shape "
                "(linear_state_size, linear_state_size)."
            )
        return mean_array, covariance_array

    def linear_transition_parameters(
        self,
        previous_nonlinear_state: ArrayLike,
        nonlinear_state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> tuple[Array, Array, Array]:
        matrix, offset, covariance = self.linear_transition_fn(
            jnp.asarray(previous_nonlinear_state),
            jnp.asarray(nonlinear_state),
            jnp.asarray(t0),
            jnp.asarray(t1),
            context,
        )
        size = prod(self.linear_state_shape) if self.linear_state_shape else 1
        matrix_array = jnp.asarray(matrix, dtype=float)
        covariance_array = jnp.asarray(covariance, dtype=float)
        offset_array = jnp.broadcast_to(jnp.asarray(offset, dtype=float), (size,))
        if matrix_array.shape != (size, size):
            raise ValueError("Conditional transition matrix has incompatible shape.")
        if covariance_array.shape != (size, size):
            raise ValueError("Conditional process covariance has incompatible shape.")
        return matrix_array, offset_array, covariance_array

    def observation_parameters(
        self,
        nonlinear_state: ArrayLike,
        time: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> tuple[Array, Array, Array | DiagonalCovariance]:
        matrix, offset, covariance = self.observation_fn(
            jnp.asarray(nonlinear_state), jnp.asarray(time), context
        )
        linear_size = prod(self.linear_state_shape) if self.linear_state_shape else 1
        observation_size = prod(self.observation_shape) if self.observation_shape else 1
        matrix_array = jnp.asarray(matrix, dtype=float)
        offset_array = jnp.broadcast_to(
            jnp.asarray(offset, dtype=float), (observation_size,)
        )
        if matrix_array.shape != (observation_size, linear_size):
            raise ValueError("Conditional observation matrix has incompatible shape.")
        if isinstance(covariance, DiagonalCovariance):
            variance = jnp.asarray(covariance.variance, dtype=float)
            if variance.shape != self.observation_shape:
                raise ValueError(
                    "Conditional diagonal observation variance has incompatible shape."
                )
            covariance_value = covariance
        else:
            covariance_value = jnp.asarray(covariance, dtype=float)
            if covariance_value.shape != (observation_size, observation_size):
                raise ValueError(
                    "Conditional observation covariance has incompatible shape."
                )
        return matrix_array, offset_array, covariance_value


class RaoBlackwellizedStateSpaceProblem(StrictModule):
    """Conditionally linear model bound to a canonical masked observation schedule."""

    model: RaoBlackwellizedStateSpaceModel
    observations: ObservationSequence
    initial_time: Array
    input_signal: AbstractStateSpaceInput | None
    input_valid: Array
    args: Any
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: RaoBlackwellizedStateSpaceModel,
        observations: ObservationSequence,
        /,
        *,
        initial_time: ArrayLike,
        problem_id: str,
        args: Any = None,
        input_signal: AbstractStateSpaceInput | None = None,
    ):
        if not isinstance(model, RaoBlackwellizedStateSpaceModel):
            raise TypeError("model must be a RaoBlackwellizedStateSpaceModel.")
        if not isinstance(observations, ObservationSequence):
            raise TypeError("observations must be an ObservationSequence.")
        if model.nonlinear_prior.batch_shape != observations.case_shape:
            raise ValueError(
                "Nonlinear prior batch_shape must equal the observation case_shape."
            )
        if model.observation_shape != observations.observation_shape:
            raise ValueError("Model and sequence observation shapes must agree.")
        if input_signal is not None:
            if not isinstance(input_signal, AbstractStateSpaceInput):
                raise TypeError("input_signal must implement AbstractStateSpaceInput.")
            if input_signal.case_shape != observations.case_shape:
                raise ValueError(
                    "Input signal case_shape must equal the observation case_shape."
                )
        initial = jnp.broadcast_to(
            jnp.asarray(initial_time, dtype=float), observations.case_shape
        )
        if bool(jnp.any(~jnp.isfinite(initial))):
            raise ValueError("initial_time must be finite.")
        if bool(jnp.any(initial > observations.times[..., 0])):
            raise ValueError("initial_time cannot exceed the first observation time.")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError("problem_id must be a non-empty string.")
        input_valid = _state_space_input_validity(
            observations,
            initial,
            input_signal,
        )
        self.model = model
        self.observations = observations
        self.initial_time = initial
        self.input_signal = input_signal
        self.input_valid = input_valid
        self.args = args
        self.problem_id = problem_id

    def step_context(
        self, case_index: ArrayLike, step_index: ArrayLike, /
    ) -> StateSpaceStepContext:
        return _state_space_step_context(
            self.observations,
            self.initial_time,
            self.input_signal,
            self.input_valid,
            self.args,
            case_index,
            step_index,
        )


class RaoBlackwellizedFilterState(StrictModule):
    nonlinear_particles: Array
    linear_means: Array
    linear_covariances: Array
    log_weights: Array
    time: Array
    log_likelihood: Array
    valid: Array
    status: Array
    root_key: Array
    num_particles: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)


class RaoBlackwellizedFilterResult(StrictModule):
    """Particle modes and analytic conditional Gaussian filtering histories."""
    initial_nonlinear_particles: Array
    initial_linear_means: Array
    initial_linear_covariances: Array
    initial_log_weights: Array

    predicted_nonlinear_particles: Array
    predicted_linear_means: Array
    predicted_linear_covariances: Array
    posterior_linear_means: Array
    posterior_linear_covariances: Array
    posterior_log_weights: Array
    nonlinear_particles: Array
    linear_means: Array
    linear_covariances: Array
    log_weights: Array
    ancestor_indices: Array
    transition_valid: Array
    effective_sample_sizes: Array
    resampled: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    step_valid: Array
    valid: Array
    status: Array
    times: Array
    final_state: RaoBlackwellizedFilterState
    problem: RaoBlackwellizedStateSpaceProblem
    nonlinear_state_shape: tuple[int, ...] = eqx.field(static=True)
    linear_state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    num_particles: int = eqx.field(static=True)
    resampling_method: ResamplingMethod = eqx.field(static=True)
    resampling_policy: ResamplingPolicy = eqx.field(static=True)
    resampling_threshold: float = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


def _configuration(
    num_particles: int,
    method: ResamplingMethod,
    policy: ResamplingPolicy,
    threshold: float,
) -> tuple[int, ResamplingMethod, ResamplingPolicy, float]:
    count = int(num_particles)
    if count < 1:
        raise ValueError("num_particles must be positive.")
    if method not in ("systematic", "stratified", "multinomial", "residual"):
        raise ValueError("Unknown resampling_method.")
    if policy not in ("ess", "always", "never"):
        raise ValueError("Unknown resampling_policy.")
    level = float(threshold)
    if not np.isfinite(level) or not 0.0 < level <= 1.0:
        raise ValueError("resampling_threshold must lie in (0, 1].")
    return count, method, policy, level


def _condition_linear_state(
    model: RaoBlackwellizedStateSpaceModel,
    previous_nonlinear: Array,
    nonlinear: Array,
    previous_mean: Array,
    previous_covariance: Array,
    t0: Array,
    t1: Array,
    value: Array,
    mask: Array,
    context: StateSpaceStepContext,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    linear_size = prod(model.linear_state_shape) if model.linear_state_shape else 1
    observation_size = prod(model.observation_shape) if model.observation_shape else 1
    transition, transition_offset, process_covariance = (
        model.linear_transition_parameters(previous_nonlinear, nonlinear, t0, t1, context)
    )
    observation, observation_offset, observation_covariance = (
        model.observation_parameters(nonlinear, t1, context)
    )
    flat_mean = previous_mean.reshape((linear_size,))
    forecast_mean = transition @ flat_mean + transition_offset
    forecast_covariance = (
        transition @ previous_covariance @ transition.T + process_covariance
    )
    forecast_covariance = 0.5 * (forecast_covariance + forecast_covariance.T)
    flat_value = value.reshape((observation_size,))
    flat_mask = mask.reshape((observation_size,))
    if isinstance(observation_covariance, DiagonalCovariance):
        (
            filtered_mean,
            filtered_covariance,
            log_likelihood,
            valid,
        ) = _condition_affine_gaussian_diagonal(
            forecast_mean,
            forecast_covariance,
            observation,
            observation_offset,
            jnp.asarray(observation_covariance.variance).reshape(
                (observation_size,)
            ),
            flat_value,
            flat_mask,
        )
    else:
        active = flat_mask.astype(forecast_mean.dtype)
        effective_observation = observation * active[:, None]
        effective_covariance = observation_covariance * active[:, None] * active[
            None, :
        ] + jnp.diag(1.0 - active)
        innovation = jnp.where(
            flat_mask,
            flat_value - observation @ forecast_mean - observation_offset,
            0.0,
        )
        innovation_covariance = (
            effective_observation @ forecast_covariance @ effective_observation.T
            + effective_covariance
        )
        scale = jnp.linalg.cholesky(innovation_covariance)
        cross = forecast_covariance @ effective_observation.T
        gain = jnp.linalg.solve(innovation_covariance, cross.T).T
        filtered_mean = forecast_mean + gain @ innovation
        identity = jnp.eye(linear_size, dtype=forecast_mean.dtype)
        update = identity - gain @ effective_observation
        filtered_covariance = (
            update @ forecast_covariance @ update.T
            + gain @ effective_covariance @ gain.T
        )
        filtered_covariance = 0.5 * (
            filtered_covariance + filtered_covariance.T
        )
        diagonal = jnp.diagonal(scale)
        solved = jax.scipy.linalg.solve_triangular(
            scale, innovation[:, None], lower=True
        )[:, 0]
        log_likelihood = -0.5 * (
            jnp.sum(solved**2)
            + 2.0 * jnp.sum(jnp.log(diagonal))
            + jnp.sum(flat_mask) * jnp.log(2.0 * jnp.pi)
        )
        valid = (
            jnp.all(jnp.isfinite(forecast_mean))
            & jnp.all(jnp.isfinite(forecast_covariance))
            & jnp.all(jnp.isfinite(filtered_mean))
            & jnp.all(jnp.isfinite(filtered_covariance))
            & jnp.all(jnp.isfinite(scale))
            & jnp.all(diagonal > 0.0)
            & jnp.isfinite(log_likelihood)
        )
    return (
        forecast_mean.reshape(model.linear_state_shape),
        forecast_covariance,
        filtered_mean.reshape(model.linear_state_shape),
        filtered_covariance,
        log_likelihood,
        valid,
    )


def rao_blackwellized_particle_filter(
    key: Key[Array, ""],
    problem: RaoBlackwellizedStateSpaceProblem,
    /,
    *,
    num_particles: int,
    resampling_method: ResamplingMethod = "systematic",
    resampling_policy: ResamplingPolicy = "ess",
    resampling_threshold: float = 0.5,
    raise_on_failure: bool = False,
) -> RaoBlackwellizedFilterResult:
    """Integrate the linear Gaussian state analytically within each mode particle."""
    if not isinstance(problem, RaoBlackwellizedStateSpaceProblem):
        raise TypeError("problem must be a RaoBlackwellizedStateSpaceProblem.")
    count, method, policy, threshold = _configuration(
        num_particles, resampling_method, resampling_policy, resampling_threshold
    )
    model = problem.model
    sequence = problem.observations
    case_shape = sequence.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = sequence.num_steps
    nonlinear_shape = model.nonlinear_state_shape
    linear_shape = model.linear_state_shape
    linear_size = prod(linear_shape) if linear_shape else 1
    identity = jnp.arange(count, dtype=jnp.int32)

    nonlinear_cases = []
    mean_cases = []
    covariance_cases = []
    initial_valid = []
    for case_index, case_id in enumerate(sequence.case_ids):
        modes = []
        means = []
        covariances = []
        particle_validity = []
        for particle_index in range(count):
            draw_key = state_space_key(
                key, "rao-blackwellized-prior", case_id, 0, member=particle_index
            )
            complete = model.nonlinear_prior.sample(draw_key)
            mode = (
                complete
                if not case_shape
                else complete.reshape((case_count,) + nonlinear_shape)[case_index]
            )
            mean, covariance = model.initial_linear_gaussian(mode, problem.args)
            valid = (
                jnp.all(jnp.isfinite(mode))
                & jnp.all(jnp.isfinite(mean))
                & jnp.all(jnp.isfinite(covariance))
                & jnp.all(jnp.linalg.eigvalsh(covariance) >= -1e-10)
            )
            modes.append(mode)
            means.append(mean)
            covariances.append(covariance)
            particle_validity.append(valid)
        nonlinear_cases.append(jnp.stack(modes))
        mean_cases.append(jnp.stack(means))
        covariance_cases.append(jnp.stack(covariances))
        initial_valid.append(jnp.all(jnp.stack(particle_validity)))
    nonlinear_particles = jnp.stack(nonlinear_cases)
    linear_means = jnp.stack(mean_cases)
    linear_covariances = jnp.stack(covariance_cases)
    log_weights = jnp.full(
        (case_count, count), -jnp.log(float(count)), dtype=linear_means.dtype
    )
    initial_nonlinear_particles = nonlinear_particles
    initial_linear_means = linear_means
    initial_linear_covariances = linear_covariances
    initial_log_weights = log_weights
    times = problem.initial_time.reshape((case_count,))
    cumulative = jnp.zeros((case_count,), dtype=linear_means.dtype)
    alive = jnp.stack(initial_valid)
    final_status = jnp.where(
        alive, PARTICLE_FILTER_SUCCESS, PARTICLE_FILTER_NONFINITE
    ).astype(jnp.int32)

    predicted_nonlinear_history: list[Array] = []
    predicted_mean_history: list[Array] = []
    predicted_covariance_history: list[Array] = []
    posterior_mean_history: list[Array] = []
    posterior_covariance_history: list[Array] = []
    posterior_weight_history: list[Array] = []
    nonlinear_history: list[Array] = []
    mean_history: list[Array] = []
    covariance_history: list[Array] = []
    weight_history: list[Array] = []
    ancestor_history: list[Array] = []
    transition_valid_history: list[Array] = []
    ess_history: list[Array] = []
    resampled_history: list[Array] = []
    increment_history: list[Array] = []
    cumulative_history: list[Array] = []
    valid_history: list[Array] = []
    status_history: list[Array] = []

    flat_times = sequence.times.reshape((case_count, num_steps))
    flat_active = sequence.step_valid.reshape((case_count, num_steps))
    flat_values = sequence.values.reshape(
        (case_count, num_steps) + model.observation_shape
    )
    flat_masks = sequence.observation_mask.reshape(
        (case_count, num_steps) + model.observation_shape
    )

    for step in range(num_steps):
        records = [[] for _ in range(17)]
        for case_index, case_id in enumerate(sequence.case_ids):
            active = bool(flat_active[case_index, step])
            if not active or not bool(alive[case_index]):
                records[0].append(nonlinear_particles[case_index])
                records[1].append(linear_means[case_index])
                records[2].append(linear_covariances[case_index])
                records[3].append(linear_means[case_index])
                records[4].append(linear_covariances[case_index])
                records[5].append(log_weights[case_index])
                records[6].append(nonlinear_particles[case_index])
                records[7].append(linear_means[case_index])
                records[8].append(linear_covariances[case_index])
                records[9].append(log_weights[case_index])
                records[10].append(identity)
                records[11].append(jnp.full((count,), alive[case_index], dtype=bool))
                records[12].append(effective_sample_size(log_weights[case_index]))
                records[13].append(jnp.asarray(False))
                records[14].append(jnp.asarray(0.0, dtype=linear_means.dtype))
                records[15].append(cumulative[case_index])
                records[16].append(alive[case_index])
                continue

            start = times[case_index]
            end = flat_times[case_index, step]
            value = flat_values[case_index, step]
            mask = flat_masks[case_index, step]
            context = problem.step_context(case_index, step)
            proposed_modes = []
            forecast_means = []
            forecast_covariances = []
            filtered_means = []
            filtered_covariances = []
            likelihoods = []
            particle_validity = []
            for particle_index in range(count):
                transition_key = state_space_key(
                    key,
                    "rao-blackwellized-transition",
                    case_id,
                    step,
                    member=particle_index,
                )
                transition_sample = model.nonlinear_transition.sample(
                    transition_key,
                    nonlinear_particles[case_index, particle_index],
                    start,
                    end,
                    context,
                )
                mode_valid = jnp.all(transition_sample.valid) & jnp.all(
                    transition_sample.status == 0
                )
                mode = jnp.where(
                    mode_valid,
                    transition_sample.values,
                    nonlinear_particles[case_index, particle_index],
                )
                conditioned = _condition_linear_state(
                    model,
                    nonlinear_particles[case_index, particle_index],
                    mode,
                    linear_means[case_index, particle_index],
                    linear_covariances[case_index, particle_index],
                    start,
                    end,
                    value,
                    mask,
                    context,
                )
                forecast_mean, forecast_covariance = conditioned[:2]
                filtered_mean, filtered_covariance, likelihood, linear_valid = (
                    conditioned[2:]
                )
                valid = mode_valid & linear_valid
                proposed_modes.append(mode)
                forecast_means.append(forecast_mean)
                forecast_covariances.append(forecast_covariance)
                filtered_means.append(filtered_mean)
                filtered_covariances.append(filtered_covariance)
                likelihoods.append(jnp.where(valid, likelihood, -jnp.inf))
                particle_validity.append(valid)
            proposed_modes = jnp.stack(proposed_modes)
            forecast_means = jnp.stack(forecast_means)
            forecast_covariances = jnp.stack(forecast_covariances)
            candidate_means = jnp.stack(filtered_means)
            candidate_covariances = jnp.stack(filtered_covariances)
            particle_validity = jnp.stack(particle_validity)
            candidates = log_weights[case_index] + jnp.stack(likelihoods)
            posterior_weights, log_increment, weights_valid = normalize_log_weights(
                candidates
            )
            accepted = bool(
                jnp.any(particle_validity) & weights_valid & jnp.isfinite(log_increment)
            )
            ess = effective_sample_size(posterior_weights)
            should_resample = policy == "always" or (
                policy == "ess" and float(ess) < threshold * count
            )
            do_resample = accepted and should_resample
            if do_resample:
                resampling_key = state_space_key(
                    key, "rao-blackwellized-resampling", case_id, step
                )
                ancestors = resample_indices(
                    resampling_key, posterior_weights, method=method
                )
                output_modes = proposed_modes[ancestors]
                output_means = candidate_means[ancestors]
                output_covariances = candidate_covariances[ancestors]
                output_weights = jnp.full_like(posterior_weights, -jnp.log(float(count)))
            elif accepted:
                ancestors = identity
                output_modes = proposed_modes
                output_means = candidate_means
                output_covariances = candidate_covariances
                output_weights = posterior_weights
            else:
                ancestors = identity
                output_modes = nonlinear_particles[case_index]
                output_means = linear_means[case_index]
                output_covariances = linear_covariances[case_index]
                output_weights = log_weights[case_index]
            if accepted:
                status = PARTICLE_FILTER_SUCCESS
            elif not bool(jnp.any(particle_validity)):
                status = PARTICLE_FILTER_TRANSITION_FAILURE
            elif not bool(weights_valid):
                status = PARTICLE_FILTER_WEIGHT_DEGENERACY
            else:
                status = PARTICLE_FILTER_NONFINITE
            next_cumulative = cumulative[case_index] + jnp.where(
                accepted, log_increment, 0.0
            )
            values_to_append = (
                proposed_modes,
                forecast_means,
                forecast_covariances,
                candidate_means,
                candidate_covariances,
                posterior_weights,
                output_modes,
                output_means,
                output_covariances,
                output_weights,
                ancestors,
                particle_validity,
                ess,
                jnp.asarray(do_resample),
                jnp.where(accepted, log_increment, 0.0),
                next_cumulative,
                jnp.asarray(accepted),
            )
            for record, record_value in zip(records, values_to_append, strict=True):
                record.append(record_value)
            final_status = final_status.at[case_index].set(status)

        stacked = [jnp.stack(record) for record in records]
        nonlinear_particles = stacked[6]
        linear_means = stacked[7]
        linear_covariances = stacked[8]
        log_weights = stacked[9]
        cumulative = stacked[15]
        step_validity = stacked[16]
        alive = alive & step_validity
        times = jnp.where(flat_active[:, step], flat_times[:, step], times)
        predicted_nonlinear_history.append(stacked[0])
        predicted_mean_history.append(stacked[1])
        predicted_covariance_history.append(stacked[2])
        posterior_mean_history.append(stacked[3])
        posterior_covariance_history.append(stacked[4])
        posterior_weight_history.append(stacked[5])
        nonlinear_history.append(stacked[6])
        mean_history.append(stacked[7])
        covariance_history.append(stacked[8])
        weight_history.append(stacked[9])
        ancestor_history.append(stacked[10])
        transition_valid_history.append(stacked[11])
        ess_history.append(stacked[12])
        resampled_history.append(stacked[13])
        increment_history.append(stacked[14])
        cumulative_history.append(stacked[15])
        valid_history.append(stacked[16])
        status_history.append(final_status)

    def restore(history: list[Array], trailing_shape: tuple[int, ...]) -> Array:
        return jnp.stack(history, axis=1).reshape(
            case_shape + (num_steps,) + trailing_shape
        )

    final_state = RaoBlackwellizedFilterState(
        nonlinear_particles=nonlinear_particles.reshape(
            case_shape + (count,) + nonlinear_shape
        ),
        linear_means=linear_means.reshape(case_shape + (count,) + linear_shape),
        linear_covariances=linear_covariances.reshape(
            case_shape + (count, linear_size, linear_size)
        ),
        log_weights=log_weights.reshape(case_shape + (count,)),
        time=times.reshape(case_shape),
        log_likelihood=cumulative.reshape(case_shape),
        valid=alive.reshape(case_shape),
        status=final_status.reshape(case_shape),
        root_key=jnp.asarray(key),
        num_particles=count,
        problem_id=problem.problem_id,
    )
    result = RaoBlackwellizedFilterResult(
        initial_nonlinear_particles=initial_nonlinear_particles.reshape(
            case_shape + (count,) + nonlinear_shape
        ),
        initial_linear_means=initial_linear_means.reshape(
            case_shape + (count,) + linear_shape
        ),
        initial_linear_covariances=initial_linear_covariances.reshape(
            case_shape + (count, linear_size, linear_size)
        ),
        initial_log_weights=initial_log_weights.reshape(case_shape + (count,)),
        predicted_nonlinear_particles=restore(
            predicted_nonlinear_history, (count,) + nonlinear_shape
        ),
        predicted_linear_means=restore(predicted_mean_history, (count,) + linear_shape),
        predicted_linear_covariances=restore(
            predicted_covariance_history, (count, linear_size, linear_size)
        ),
        posterior_linear_means=restore(posterior_mean_history, (count,) + linear_shape),
        posterior_linear_covariances=restore(
            posterior_covariance_history, (count, linear_size, linear_size)
        ),
        posterior_log_weights=restore(posterior_weight_history, (count,)),
        nonlinear_particles=restore(nonlinear_history, (count,) + nonlinear_shape),
        linear_means=restore(mean_history, (count,) + linear_shape),
        linear_covariances=restore(covariance_history, (count, linear_size, linear_size)),
        log_weights=restore(weight_history, (count,)),
        ancestor_indices=restore(ancestor_history, (count,)),
        transition_valid=restore(transition_valid_history, (count,)),
        effective_sample_sizes=restore(ess_history, ()),
        resampled=restore(resampled_history, ()),
        incremental_log_likelihood=restore(increment_history, ()),
        cumulative_log_likelihood=restore(cumulative_history, ()),
        step_valid=sequence.step_valid,
        valid=restore(valid_history, ()),
        status=restore(status_history, ()),
        times=sequence.times,
        final_state=final_state,
        problem=problem,
        nonlinear_state_shape=nonlinear_shape,
        linear_state_shape=linear_shape,
        observation_shape=model.observation_shape,
        case_shape=case_shape,
        case_axes=sequence.case_axes,
        case_ids=sequence.case_ids,
        num_particles=count,
        resampling_method=method,
        resampling_policy=policy,
        resampling_threshold=threshold,
    )
    if raise_on_failure and not bool(jnp.all(result.successful)):
        raise RuntimeError(
            "Rao-Blackwellized filtering failed for at least one physical case."
        )
    return result


class RaoBlackwellizedFilterLikelihood(StrictModule):
    """Configured conditionally linear particle likelihood for experiments."""

    key_data: tuple[int, ...] = eqx.field(static=True)
    key_implementation: str = eqx.field(static=True)
    num_particles: int = eqx.field(static=True)
    resampling_method: ResamplingMethod = eqx.field(static=True)
    resampling_policy: ResamplingPolicy = eqx.field(static=True)
    resampling_threshold: float = eqx.field(static=True)
    raise_on_failure: bool = eqx.field(static=True)

    def __init__(
        self,
        key: Key[Array, ""],
        /,
        *,
        num_particles: int,
        resampling_method: ResamplingMethod = "systematic",
        resampling_policy: ResamplingPolicy = "ess",
        resampling_threshold: float = 0.5,
        raise_on_failure: bool = False,
    ):
        count, method, policy, threshold = _configuration(
            num_particles,
            resampling_method,
            resampling_policy,
            resampling_threshold,
        )
        if not isinstance(raise_on_failure, bool):
            raise TypeError("raise_on_failure must be a bool.")
        key_data = np.asarray(jax.device_get(jax.random.key_data(key)))
        if key_data.ndim != 1:
            raise ValueError("key must be one unbatched JAX random key.")
        self.key_data = tuple(int(value) for value in key_data)
        self.key_implementation = str(jax.random.key_impl(key))
        self.num_particles = count
        self.resampling_method = method
        self.resampling_policy = policy
        self.resampling_threshold = threshold
        self.raise_on_failure = raise_on_failure

    @property
    def key(self) -> Array:
        """Reconstruct the configured typed JAX key without static array state."""
        return jax.random.wrap_key_data(
            jnp.asarray(self.key_data, dtype=jnp.uint32),
            impl=self.key_implementation,
        )

    def __call__(
        self, problem: RaoBlackwellizedStateSpaceProblem, /
    ) -> RaoBlackwellizedFilterResult:
        """Evaluate the configured deterministic-key particle likelihood."""
        return rao_blackwellized_particle_filter(
            self.key,
            problem,
            num_particles=self.num_particles,
            resampling_method=self.resampling_method,
            resampling_policy=self.resampling_policy,
            resampling_threshold=self.resampling_threshold,
            raise_on_failure=self.raise_on_failure,
        )


__all__ = [
    "ConditionalLinearObservation",
    "ConditionalLinearTransition",
    "InitialLinearGaussian",
    "rao_blackwellized_particle_filter",
    "RaoBlackwellizedFilterResult",
    "RaoBlackwellizedFilterState",
    "RaoBlackwellizedFilterLikelihood",
    "RaoBlackwellizedStateSpaceModel",
    "RaoBlackwellizedStateSpaceProblem",
]
