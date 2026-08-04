#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from .._strict import StrictModule
from ..stochastic._state_space import state_space_key, StateSpaceProblem
from ._checkpoint import (
    array_tree_fingerprint,
    read_checkpoint_archive,
    write_checkpoint_archive,
)
from ._predictive import PredictiveField, SampleAxis


ResamplingMethod: TypeAlias = Literal[
    "systematic", "stratified", "multinomial", "residual"
]
ResamplingPolicy: TypeAlias = Literal["ess", "always", "never"]
ParticleFilterStatus: TypeAlias = Literal[
    "success", "transition_failure", "weight_degeneracy", "nonfinite"
]
PARTICLE_FILTER_SUCCESS = 0
PARTICLE_FILTER_TRANSITION_FAILURE = 1
PARTICLE_FILTER_WEIGHT_DEGENERACY = 2
PARTICLE_FILTER_NONFINITE = 3


def particle_filter_status_name(value: int, /) -> ParticleFilterStatus:
    code = int(value)
    if code == PARTICLE_FILTER_SUCCESS:
        return "success"
    if code == PARTICLE_FILTER_TRANSITION_FAILURE:
        return "transition_failure"
    if code == PARTICLE_FILTER_WEIGHT_DEGENERACY:
        return "weight_degeneracy"
    if code == PARTICLE_FILTER_NONFINITE:
        return "nonfinite"
    raise ValueError(f"Unknown particle-filter status code {code}.")


def normalize_log_weights(log_weights: Array, /) -> tuple[Array, Array, Array]:
    """Normalize the final axis and explicitly report all-invalid weight sets."""
    values = jnp.asarray(log_weights, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("log_weights must have a non-empty particle axis.")
    log_normalizer = jax.scipy.special.logsumexp(values, axis=-1)
    valid = jnp.isfinite(log_normalizer) & jnp.all(~jnp.isnan(values), axis=-1)
    count = int(values.shape[-1])
    uniform = jnp.full_like(values, -jnp.log(float(count)))
    normalized = jnp.where(valid[..., None], values - log_normalizer[..., None], uniform)
    return normalized, log_normalizer, valid


def effective_sample_size(log_weights: Array, /) -> Array:
    """Effective sample size of normalized or unnormalized log weights."""
    normalized, _, valid = normalize_log_weights(log_weights)
    value = 1.0 / jnp.sum(jnp.exp(2.0 * normalized), axis=-1)
    return jnp.where(valid, value, 0.0)


def _normalized_probabilities(log_weights: Array, /) -> Array:
    normalized, _, valid = normalize_log_weights(log_weights)
    if not bool(valid):
        raise ValueError("Cannot resample a degenerate weight vector.")
    return jnp.exp(normalized)


def resample_indices(
    key: Key[Array, ""],
    log_weights: Array,
    /,
    *,
    method: ResamplingMethod = "systematic",
) -> Array:
    """Draw one fixed-size ancestry vector from a one-dimensional weight set."""
    values = jnp.asarray(log_weights, dtype=float)
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError("log_weights must be a non-empty one-dimensional vector.")
    if method not in ("systematic", "stratified", "multinomial", "residual"):
        raise ValueError(f"Unknown resampling method {method!r}.")
    count = int(values.shape[0])
    probabilities = _normalized_probabilities(values)
    if method == "multinomial":
        return jr.categorical(key, jnp.log(probabilities), shape=(count,)).astype(
            jnp.int32
        )
    cumulative = jnp.cumsum(probabilities).at[-1].set(1.0)
    if method == "systematic":
        offset = jr.uniform(key, (), minval=0.0, maxval=1.0 / count)
        positions = offset + jnp.arange(count, dtype=float) / count
        return jnp.searchsorted(cumulative, positions, side="right").astype(jnp.int32)
    if method == "stratified":
        positions = (jnp.arange(count, dtype=float) + jr.uniform(key, (count,))) / count
        return jnp.searchsorted(cumulative, positions, side="right").astype(jnp.int32)
    expected = count * probabilities
    deterministic_counts = jnp.floor(expected).astype(jnp.int32)
    deterministic_total = jnp.sum(deterministic_counts)
    deterministic = jnp.repeat(
        jnp.arange(count, dtype=jnp.int32),
        deterministic_counts,
        total_repeat_length=count,
    )
    remainder_count = count - deterministic_total
    remainder_probabilities = jnp.where(
        remainder_count > 0,
        (expected - deterministic_counts) / jnp.maximum(remainder_count, 1),
        jnp.full_like(probabilities, 1.0 / count),
    )
    random_indices = jr.categorical(
        key,
        jnp.log(jnp.maximum(remainder_probabilities, jnp.finfo(float).tiny)),
        shape=(count,),
    ).astype(jnp.int32)
    positions = jnp.arange(count, dtype=jnp.int32)
    return jnp.where(
        positions < deterministic_total,
        deterministic,
        random_indices[jnp.maximum(positions - deterministic_total, 0)],
    )


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
        raise ValueError(f"Unknown resampling method {method!r}.")
    if policy not in ("ess", "always", "never"):
        raise ValueError(f"Unknown resampling policy {policy!r}.")
    level = float(threshold)
    if not np.isfinite(level) or not 0.0 < level <= 1.0:
        raise ValueError("resampling_threshold must lie in (0, 1].")
    return count, method, policy, level


def _case_count(problem: StateSpaceProblem) -> int:
    shape = problem.observations.case_shape
    return prod(shape) if shape else 1


def _case_value(value: Array, case_index: int, case_shape: tuple[int, ...], /) -> Array:
    array = jnp.asarray(value)
    if not case_shape:
        return array
    return array.reshape((prod(case_shape),) + array.shape[len(case_shape) :])[case_index]


class ParticleFilterState(StrictModule):
    """Replayable streaming particle-filter state with semantic root-key lineage."""

    particles: Array
    log_weights: Array
    time: Array
    log_likelihood: Array
    valid: Array
    status: Array
    root_key: Array
    step_index: int = eqx.field(static=True)
    num_particles: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    resampling_method: ResamplingMethod = eqx.field(static=True)
    resampling_policy: ResamplingPolicy = eqx.field(static=True)
    resampling_threshold: float = eqx.field(static=True)


class ParticleFilterStep(StrictModule):
    """One bootstrap propagation, weighting, and optional resampling record."""

    predicted_particles: Array
    posterior_log_weights: Array
    particles: Array
    log_weights: Array
    ancestor_indices: Array
    transition_valid: Array
    effective_sample_size: Array
    resampled: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    active: Array
    valid: Array
    status: Array


class ParticleFilterResult(StrictModule):
    """Fixed-shape particle history with genealogy and complete run provenance."""

    predicted_particles: Array
    posterior_log_weights: Array
    particles: Array
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
    final_state: ParticleFilterState
    problem: StateSpaceProblem
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    num_particles: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    resampling_method: ResamplingMethod = eqx.field(static=True)
    resampling_policy: ResamplingPolicy = eqx.field(static=True)
    resampling_threshold: float = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


def initialize_particle_filter(
    key: Key[Array, ""],
    problem: StateSpaceProblem,
    /,
    *,
    num_particles: int,
    resampling_method: ResamplingMethod = "systematic",
    resampling_policy: ResamplingPolicy = "ess",
    resampling_threshold: float = 0.5,
) -> ParticleFilterState:
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    count, method, policy, threshold = _configuration(
        num_particles, resampling_method, resampling_policy, resampling_threshold
    )
    case_shape = problem.observations.case_shape
    case_count = _case_count(problem)
    draws = []
    for case_index, case_id in enumerate(problem.observations.case_ids):
        case_draws = []
        for particle_index in range(count):
            draw_key = state_space_key(
                key, "particle-filter-prior", case_id, 0, member=particle_index
            )
            complete_draw = problem.model.prior.sample(draw_key)
            case_draws.append(_case_value(complete_draw, case_index, case_shape))
        draws.append(jnp.stack(case_draws, axis=0))
    particles = jnp.stack(draws, axis=0).reshape(
        case_shape + (count,) + problem.model.state_shape
    )
    finite_axes = tuple(range(len(case_shape) + 1, particles.ndim))
    particle_finite = jnp.all(jnp.isfinite(particles), axis=finite_axes)
    valid = jnp.all(particle_finite, axis=-1)
    status = jnp.where(valid, PARTICLE_FILTER_SUCCESS, PARTICLE_FILTER_NONFINITE).astype(
        jnp.int32
    )
    return ParticleFilterState(
        particles=particles,
        log_weights=jnp.full(
            case_shape + (count,), -jnp.log(float(count)), dtype=particles.dtype
        ),
        time=problem.initial_time,
        log_likelihood=jnp.zeros(case_shape, dtype=particles.dtype),
        valid=valid,
        status=status,
        root_key=jnp.asarray(key),
        step_index=0,
        num_particles=count,
        problem_id=problem.problem_id,
        resampling_method=method,
        resampling_policy=policy,
        resampling_threshold=threshold,
    )


def _propagate_particles(
    problem: StateSpaceProblem,
    state: ParticleFilterState,
    active: Array,
    target_time: Array,
) -> tuple[Array, Array]:
    case_shape = problem.observations.case_shape
    case_count = _case_count(problem)
    count = state.num_particles
    previous = state.particles.reshape((case_count, count) + problem.model.state_shape)
    starts = state.time.reshape((case_count,))
    ends = target_time.reshape((case_count,))
    active_flat = active.reshape((case_count,))
    propagated_cases = []
    valid_cases = []
    for case_index, case_id in enumerate(problem.observations.case_ids):
        propagated = []
        particle_valid = []
        for particle_index in range(count):
            if bool(active_flat[case_index]) and bool(
                state.valid.reshape((-1,))[case_index]
            ):
                transition_key = state_space_key(
                    state.root_key,
                    "particle-filter-transition",
                    case_id,
                    state.step_index,
                    member=particle_index,
                )
                sample = problem.model.transition.sample(
                    transition_key,
                    previous[case_index, particle_index],
                    starts[case_index],
                    ends[case_index],
                )
                sample_valid = jnp.all(sample.valid) & jnp.all(sample.status == 0)
                propagated.append(
                    jnp.where(
                        sample_valid,
                        sample.values,
                        previous[case_index, particle_index],
                    )
                )
                particle_valid.append(sample_valid)
            else:
                propagated.append(previous[case_index, particle_index])
                particle_valid.append(jnp.asarray(True))
        propagated_cases.append(jnp.stack(propagated, axis=0))
        valid_cases.append(jnp.stack(particle_valid, axis=0))
    return (
        jnp.stack(propagated_cases, axis=0).reshape(
            case_shape + (count,) + problem.model.state_shape
        ),
        jnp.stack(valid_cases, axis=0).reshape(case_shape + (count,)),
    )


def _observation_log_likelihoods(
    problem: StateSpaceProblem,
    state: ParticleFilterState,
    predicted_particles: Array,
    active: Array,
    target_time: Array,
    values: Array,
    mask: Array,
) -> Array:
    case_shape = problem.observations.case_shape
    case_count = _case_count(problem)
    count = state.num_particles
    particles = predicted_particles.reshape(
        (case_count, count) + problem.model.state_shape
    )
    flat_times = target_time.reshape((case_count,))
    flat_values = values.reshape((case_count,) + problem.model.observation_shape)
    flat_mask = mask.reshape((case_count,) + problem.model.observation_shape)
    active_flat = active.reshape((case_count,))
    likelihoods = []
    for case_index in range(case_count):
        case_likelihoods = []
        for particle_index in range(count):
            if bool(active_flat[case_index]):
                value = problem.model.observation.log_prob(
                    flat_values[case_index],
                    particles[case_index, particle_index],
                    flat_times[case_index],
                    flat_mask[case_index],
                )
                case_likelihoods.append(jnp.asarray(value).reshape(()))
            else:
                case_likelihoods.append(jnp.asarray(0.0))
        likelihoods.append(jnp.stack(case_likelihoods, axis=0))
    return jnp.stack(likelihoods, axis=0).reshape(case_shape + (count,))


def particle_filter_step(
    problem: StateSpaceProblem,
    state: ParticleFilterState,
    /,
) -> tuple[ParticleFilterState, ParticleFilterStep]:
    """Run one bootstrap-filter step with stable case/step/particle key derivation."""
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    if not isinstance(state, ParticleFilterState):
        raise TypeError("state must be a ParticleFilterState.")
    if state.problem_id != problem.problem_id:
        raise ValueError("Particle-filter state and problem IDs do not match.")
    index = state.step_index
    sequence = problem.observations
    if index >= sequence.num_steps:
        raise ValueError("The particle-filter state has consumed every observation step.")
    case_shape = sequence.case_shape
    case_count = _case_count(problem)
    count = state.num_particles
    active = sequence.step_valid[..., index]
    target_time = sequence.times[..., index]
    step_axis = len(case_shape)
    values = jnp.take(sequence.values, index, axis=step_axis)
    mask = jnp.take(sequence.observation_mask, index, axis=step_axis)
    predicted, transition_valid = _propagate_particles(
        problem, state, active, target_time
    )
    likelihoods = _observation_log_likelihoods(
        problem, state, predicted, active, target_time, values, mask
    )
    likelihoods = jnp.where(transition_valid, likelihoods, -jnp.inf)
    candidates = state.log_weights + likelihoods
    posterior_log_weights, log_normalizers, weights_valid = normalize_log_weights(
        candidates
    )
    ess = effective_sample_size(posterior_log_weights)
    transition_case_valid = jnp.any(transition_valid, axis=-1)
    finite_particles = jnp.all(
        jnp.isfinite(predicted),
        axis=tuple(range(len(case_shape), predicted.ndim)),
    )
    active_valid = transition_case_valid & weights_valid & finite_particles
    accepted = active & state.valid & active_valid
    identity = jnp.arange(count, dtype=jnp.int32)
    flat_predicted = predicted.reshape((case_count, count) + problem.model.state_shape)
    flat_posterior = posterior_log_weights.reshape((case_count, count))
    flat_previous_particles = state.particles.reshape(
        (case_count, count) + problem.model.state_shape
    )
    flat_previous_weights = state.log_weights.reshape((case_count, count))
    flat_ess = ess.reshape((case_count,))
    flat_active = active.reshape((case_count,))
    flat_accepted = accepted.reshape((case_count,))
    output_particles = []
    output_weights = []
    ancestor_indices = []
    resampled_values = []
    for case_index, case_id in enumerate(sequence.case_ids):
        should_resample = state.resampling_policy == "always" or (
            state.resampling_policy == "ess"
            and float(flat_ess[case_index]) < state.resampling_threshold * count
        )
        do_resample = bool(flat_accepted[case_index]) and should_resample
        if do_resample:
            resampling_key = state_space_key(
                state.root_key,
                "particle-filter-resampling",
                case_id,
                index,
            )
            ancestors = resample_indices(
                resampling_key,
                flat_posterior[case_index],
                method=state.resampling_method,
            )
            output_particles.append(flat_predicted[case_index, ancestors])
            output_weights.append(
                jnp.full((count,), -jnp.log(float(count)), dtype=predicted.dtype)
            )
        elif bool(flat_accepted[case_index]):
            ancestors = identity
            output_particles.append(flat_predicted[case_index])
            output_weights.append(flat_posterior[case_index])
        else:
            ancestors = identity
            output_particles.append(flat_previous_particles[case_index])
            output_weights.append(flat_previous_weights[case_index])
        ancestor_indices.append(ancestors)
        resampled_values.append(jnp.asarray(do_resample))
    next_particles = jnp.stack(output_particles, axis=0).reshape(
        case_shape + (count,) + problem.model.state_shape
    )
    next_log_weights = jnp.stack(output_weights, axis=0).reshape(case_shape + (count,))
    ancestors = jnp.stack(ancestor_indices, axis=0).reshape(case_shape + (count,))
    resampled = jnp.stack(resampled_values, axis=0).reshape(case_shape)
    status = jnp.where(
        ~active,
        PARTICLE_FILTER_SUCCESS,
        jnp.where(
            ~transition_case_valid,
            PARTICLE_FILTER_TRANSITION_FAILURE,
            jnp.where(
                ~weights_valid,
                PARTICLE_FILTER_WEIGHT_DEGENERACY,
                jnp.where(
                    ~finite_particles,
                    PARTICLE_FILTER_NONFINITE,
                    PARTICLE_FILTER_SUCCESS,
                ),
            ),
        ),
    ).astype(jnp.int32)
    next_valid = state.valid & jnp.where(active, active_valid, True)
    increment = jnp.where(active, log_normalizers, 0.0)
    next_log_likelihood = jnp.where(
        active,
        state.log_likelihood + increment,
        state.log_likelihood,
    )
    next_state = ParticleFilterState(
        particles=next_particles,
        log_weights=next_log_weights,
        time=jnp.where(active, target_time, state.time),
        log_likelihood=next_log_likelihood,
        valid=next_valid,
        status=status,
        root_key=state.root_key,
        step_index=index + 1,
        num_particles=count,
        problem_id=problem.problem_id,
        resampling_method=state.resampling_method,
        resampling_policy=state.resampling_policy,
        resampling_threshold=state.resampling_threshold,
    )
    record = ParticleFilterStep(
        predicted_particles=predicted,
        posterior_log_weights=posterior_log_weights,
        particles=next_particles,
        log_weights=next_log_weights,
        ancestor_indices=ancestors,
        transition_valid=transition_valid,
        effective_sample_size=ess,
        resampled=resampled,
        incremental_log_likelihood=increment,
        cumulative_log_likelihood=next_log_likelihood,
        active=active,
        valid=active_valid,
        status=status,
    )
    return next_state, record


def _stack(values: list[Array], case_rank: int, /) -> Array:
    return jnp.stack(values, axis=case_rank)


def bootstrap_particle_filter(
    key: Key[Array, ""],
    problem: StateSpaceProblem,
    /,
    *,
    num_particles: int,
    resampling_method: ResamplingMethod = "systematic",
    resampling_policy: ResamplingPolicy = "ess",
    resampling_threshold: float = 0.5,
    raise_on_failure: bool = False,
) -> ParticleFilterResult:
    """Run a bootstrap particle filter without assuming a transition density."""
    state = initialize_particle_filter(
        key,
        problem,
        num_particles=num_particles,
        resampling_method=resampling_method,
        resampling_policy=resampling_policy,
        resampling_threshold=resampling_threshold,
    )
    records: list[ParticleFilterStep] = []
    for _ in range(problem.observations.num_steps):
        state, record = particle_filter_step(problem, state)
        records.append(record)
    rank = len(problem.observations.case_shape)
    result = ParticleFilterResult(
        predicted_particles=_stack(
            [record.predicted_particles for record in records], rank
        ),
        posterior_log_weights=_stack(
            [record.posterior_log_weights for record in records], rank
        ),
        particles=_stack([record.particles for record in records], rank),
        log_weights=_stack([record.log_weights for record in records], rank),
        ancestor_indices=_stack([record.ancestor_indices for record in records], rank),
        transition_valid=_stack([record.transition_valid for record in records], rank),
        effective_sample_sizes=_stack(
            [record.effective_sample_size for record in records], rank
        ),
        resampled=_stack([record.resampled for record in records], rank),
        incremental_log_likelihood=_stack(
            [record.incremental_log_likelihood for record in records], rank
        ),
        cumulative_log_likelihood=_stack(
            [record.cumulative_log_likelihood for record in records], rank
        ),
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
        num_particles=state.num_particles,
        model_id=problem.model.model_id,
        problem_id=problem.problem_id,
        sequence_id=problem.observations.sequence_id,
        resampling_method=state.resampling_method,
        resampling_policy=state.resampling_policy,
        resampling_threshold=state.resampling_threshold,
    )
    if raise_on_failure and not bool(jnp.all(result.successful)):
        raise RuntimeError("Particle filtering failed for at least one physical case.")
    return result


def _sample_terminal_index(
    key: Array,
    log_weights: Array,
    /,
) -> int:
    return int(jr.categorical(key, log_weights))


def sample_particle_ancestry_paths(
    key: Key[Array, ""],
    result: ParticleFilterResult,
    /,
    *,
    sample_shape: tuple[int, ...] = (),
) -> Array:
    """Trace complete paths through the stored resampling genealogy."""
    if not isinstance(result, ParticleFilterResult):
        raise TypeError("result must be a ParticleFilterResult.")
    samples = tuple(int(size) for size in sample_shape)
    if any(size <= 0 for size in samples):
        raise ValueError("sample_shape dimensions must be positive.")
    sample_count = prod(samples) if samples else 1
    case_count = prod(result.case_shape) if result.case_shape else 1
    num_steps = result.step_valid.shape[-1]
    state_size = prod(result.state_shape) if result.state_shape else 1
    particles = result.particles.reshape(
        (case_count, num_steps, result.num_particles, state_size)
    )
    weights = result.log_weights.reshape((case_count, num_steps, result.num_particles))
    ancestors = result.ancestor_indices.reshape(
        (case_count, num_steps, result.num_particles)
    )
    active = result.step_valid.reshape((case_count, num_steps))
    paths = np.zeros((sample_count, case_count, num_steps, state_size))
    for sample_index in range(sample_count):
        for case_index, case_id in enumerate(result.case_ids):
            valid_count = int(np.sum(np.asarray(active[case_index])))
            if valid_count == 0:
                continue
            terminal = valid_count - 1
            terminal_key = state_space_key(
                key,
                "particle-ancestry-smoother",
                case_id,
                terminal,
                member=sample_index,
            )
            particle_index = _sample_terminal_index(
                terminal_key, weights[case_index, terminal]
            )
            path = np.zeros((num_steps, state_size))
            path[terminal] = np.asarray(particles[case_index, terminal, particle_index])
            for step in range(terminal, 0, -1):
                particle_index = int(ancestors[case_index, step, particle_index])
                path[step - 1] = np.asarray(
                    particles[case_index, step - 1, particle_index]
                )
            if valid_count < num_steps:
                path[valid_count:] = path[terminal]
            paths[sample_index, case_index] = path
    output = jnp.asarray(paths).reshape(
        samples + result.case_shape + (num_steps,) + result.state_shape
    )
    if samples:
        return output
    return output.reshape(result.case_shape + (num_steps,) + result.state_shape)


def sample_particle_backward_paths(
    key: Key[Array, ""],
    result: ParticleFilterResult,
    /,
    *,
    sample_shape: tuple[int, ...] = (),
) -> Array:
    """Run backward simulation using normalized transition densities."""
    if not isinstance(result, ParticleFilterResult):
        raise TypeError("result must be a ParticleFilterResult.")
    transition = result.problem.model.transition
    if not transition.has_log_density:
        raise ValueError("Backward simulation requires a normalized transition density.")
    samples = tuple(int(size) for size in sample_shape)
    if any(size <= 0 for size in samples):
        raise ValueError("sample_shape dimensions must be positive.")
    sample_count = prod(samples) if samples else 1
    case_count = prod(result.case_shape) if result.case_shape else 1
    num_steps = result.step_valid.shape[-1]
    state_size = prod(result.state_shape) if result.state_shape else 1
    particles = result.particles.reshape(
        (case_count, num_steps, result.num_particles, state_size)
    )
    weights = result.log_weights.reshape((case_count, num_steps, result.num_particles))
    times = result.times.reshape((case_count, num_steps))
    active = result.step_valid.reshape((case_count, num_steps))
    paths = np.zeros((sample_count, case_count, num_steps, state_size))
    for sample_index in range(sample_count):
        for case_index, case_id in enumerate(result.case_ids):
            valid_count = int(np.sum(np.asarray(active[case_index])))
            if valid_count == 0:
                continue
            terminal = valid_count - 1
            terminal_key = state_space_key(
                key,
                "particle-backward-smoother",
                case_id,
                terminal,
                member=sample_index,
            )
            particle_index = _sample_terminal_index(
                terminal_key, weights[case_index, terminal]
            )
            path = jnp.zeros((num_steps, state_size), dtype=particles.dtype)
            path = path.at[terminal].set(particles[case_index, terminal, particle_index])
            for step in range(terminal - 1, -1, -1):
                transition_terms = []
                for candidate in range(result.num_particles):
                    transition_terms.append(
                        jnp.asarray(
                            transition.log_prob(
                                path[step + 1].reshape(result.state_shape),
                                particles[case_index, step, candidate].reshape(
                                    result.state_shape
                                ),
                                times[case_index, step],
                                times[case_index, step + 1],
                            )
                        ).reshape(())
                    )
                backward_weights = weights[case_index, step] + jnp.stack(transition_terms)
                normalized, _, valid = normalize_log_weights(backward_weights)
                if not bool(valid):
                    raise RuntimeError(
                        "Backward particle weights degenerated for a physical case."
                    )
                draw_key = state_space_key(
                    key,
                    "particle-backward-smoother",
                    case_id,
                    step,
                    member=sample_index,
                )
                particle_index = _sample_terminal_index(draw_key, normalized)
                path = path.at[step].set(particles[case_index, step, particle_index])
            if valid_count < num_steps:
                path = path.at[valid_count:].set(path[terminal])
            paths[sample_index, case_index] = np.asarray(path)
    output = jnp.asarray(paths).reshape(
        samples + result.case_shape + (num_steps,) + result.state_shape
    )
    if samples:
        return output
    return output.reshape(result.case_shape + (num_steps,) + result.state_shape)


def particle_filter_predictive(
    key: Key[Array, ""],
    result: ParticleFilterResult,
    /,
    *,
    particle_dim: str = "particle",
    time_dim: str = "time",
) -> PredictiveField:
    """Convert weighted filtering marginals to an unweighted PredictiveField."""
    if not isinstance(result, ParticleFilterResult):
        raise TypeError("result must be a ParticleFilterResult.")
    if not particle_dim or not time_dim or particle_dim == time_dim:
        raise ValueError("particle_dim and time_dim must be distinct non-empty names.")
    if particle_dim in result.case_axes or time_dim in result.case_axes:
        raise ValueError(
            "Predictive particle/time dimensions must not collide with case axes."
        )
    case_count = prod(result.case_shape) if result.case_shape else 1
    num_steps = result.step_valid.shape[-1]
    state_size = prod(result.state_shape) if result.state_shape else 1
    source_particles = result.predicted_particles.reshape(
        (case_count, num_steps, result.num_particles, state_size)
    )
    weights = result.posterior_log_weights.reshape(
        (case_count, num_steps, result.num_particles)
    )
    active = result.step_valid.reshape((case_count, num_steps))
    converted = np.full((case_count, num_steps, result.num_particles, state_size), np.nan)
    for case_index, case_id in enumerate(result.case_ids):
        for step in range(num_steps):
            if bool(active[case_index, step]) and bool(
                result.valid.reshape((case_count, num_steps))[case_index, step]
            ):
                resampling_key = state_space_key(
                    key, "particle-predictive", case_id, step
                )
                indices = resample_indices(
                    resampling_key, weights[case_index, step], method="systematic"
                )
                converted[case_index, step] = np.asarray(
                    source_particles[case_index, step, indices]
                )
    values = jnp.asarray(converted).reshape(
        result.case_shape + (num_steps, result.num_particles) + result.state_shape
    )
    dims = result.case_axes + (time_dim, particle_dim) + (None,) * len(result.state_shape)
    return PredictiveField(
        cx.Field(values, dims=dims),
        (SampleAxis(particle_dim, "process"),),
    )


class ParticleFilterDiagnostics(StrictModule):
    """Weight, transition, resampling, and finite-value diagnostics by case."""

    minimum_effective_sample_size: Array
    mean_effective_sample_size: Array
    resampling_count: Array
    transition_rejection_fraction: Array
    final_log_likelihood: Array
    valid_steps: Array
    finite: Array

    @property
    def passed(self) -> bool:
        return bool(jnp.all(self.finite))


def particle_filter_diagnostics(
    result: ParticleFilterResult,
    /,
) -> ParticleFilterDiagnostics:
    if not isinstance(result, ParticleFilterResult):
        raise TypeError("result must be a ParticleFilterResult.")
    active = result.step_valid
    valid_count = jnp.sum(active, axis=-1)
    ess = result.effective_sample_sizes
    minimum = jnp.min(jnp.where(active, ess, jnp.inf), axis=-1)
    mean = jnp.sum(jnp.where(active, ess, 0.0), axis=-1) / jnp.maximum(valid_count, 1)
    transition_valid = result.transition_valid
    attempted = jnp.sum(active, axis=-1) * result.num_particles
    rejected = jnp.sum(
        jnp.where(active[..., None], ~transition_valid, False), axis=(-1, -2)
    )
    finite = (
        jnp.all(result.valid | ~active, axis=-1)
        & jnp.all(jnp.isfinite(result.effective_sample_sizes) | ~active, axis=-1)
        & jnp.isfinite(result.final_state.log_likelihood)
    )
    return ParticleFilterDiagnostics(
        minimum_effective_sample_size=minimum,
        mean_effective_sample_size=mean,
        resampling_count=jnp.sum(result.resampled & active, axis=-1),
        transition_rejection_fraction=rejected / jnp.maximum(attempted, 1),
        final_log_likelihood=result.final_state.log_likelihood,
        valid_steps=valid_count,
        finite=finite,
    )


def _particle_checkpoint_compatibility(
    problem: StateSpaceProblem,
    /,
    *,
    num_particles: int,
    resampling_method: ResamplingMethod,
    resampling_policy: ResamplingPolicy,
    resampling_threshold: float,
) -> dict[str, object]:
    count, method, policy, threshold = _configuration(
        num_particles,
        resampling_method,
        resampling_policy,
        resampling_threshold,
    )
    return {
        "problem_id": problem.problem_id,
        "model_id": problem.model.model_id,
        "sequence_id": problem.observations.sequence_id,
        "state_shape": list(problem.model.state_shape),
        "observation_shape": list(problem.model.observation_shape),
        "case_shape": list(problem.observations.case_shape),
        "case_ids": list(problem.observations.case_ids),
        "num_particles": count,
        "resampling_method": method,
        "resampling_policy": policy,
        "resampling_threshold": threshold,
        "problem_arrays": array_tree_fingerprint(problem),
    }


def write_particle_filter_checkpoint(
    path: str | PathLike[str],
    problem: StateSpaceProblem,
    state: ParticleFilterState,
    /,
) -> Path:
    """Atomically save a pickle-free streaming particle-filter checkpoint."""
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    if not isinstance(state, ParticleFilterState):
        raise TypeError("state must be a ParticleFilterState.")
    if state.problem_id != problem.problem_id:
        raise ValueError("Particle-filter state and problem IDs do not match.")
    compatibility = _particle_checkpoint_compatibility(
        problem,
        num_particles=state.num_particles,
        resampling_method=state.resampling_method,
        resampling_policy=state.resampling_policy,
        resampling_threshold=state.resampling_threshold,
    )
    return write_checkpoint_archive(
        path,
        kind="particle-filter-state-v1",
        compatibility=compatibility,
        state={"step_index": state.step_index},
        arrays={
            "particles": state.particles,
            "log_weights": state.log_weights,
            "time": state.time,
            "log_likelihood": state.log_likelihood,
            "valid": state.valid,
            "status": state.status,
            "root_key_data": jr.key_data(state.root_key),
        },
    )


def read_particle_filter_checkpoint(
    path: str | PathLike[str],
    problem: StateSpaceProblem,
    /,
    *,
    num_particles: int,
    resampling_method: ResamplingMethod = "systematic",
    resampling_policy: ResamplingPolicy = "ess",
    resampling_threshold: float = 0.5,
) -> ParticleFilterState:
    """Load a particle state only when its problem and algorithm contract match."""
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    count, method, policy, threshold = _configuration(
        num_particles,
        resampling_method,
        resampling_policy,
        resampling_threshold,
    )
    compatibility = _particle_checkpoint_compatibility(
        problem,
        num_particles=count,
        resampling_method=method,
        resampling_policy=policy,
        resampling_threshold=threshold,
    )
    state_data, arrays = read_checkpoint_archive(
        path,
        kind="particle-filter-state-v1",
        compatibility=compatibility,
    )
    if set(state_data) != {"step_index"}:
        raise ValueError("Particle-filter checkpoint state manifest is invalid.")
    step_index = int(state_data["step_index"])
    if not 0 <= step_index <= problem.observations.num_steps:
        raise ValueError("Particle-filter checkpoint step_index is outside the schedule.")
    expected_shapes = {
        "particles": problem.observations.case_shape
        + (count,)
        + problem.model.state_shape,
        "log_weights": problem.observations.case_shape + (count,),
        "time": problem.observations.case_shape,
        "log_likelihood": problem.observations.case_shape,
        "valid": problem.observations.case_shape,
        "status": problem.observations.case_shape,
        "root_key_data": jr.key_data(jr.key(0)).shape,
    }
    if set(arrays) != set(expected_shapes):
        raise ValueError("Particle-filter checkpoint array inventory is invalid.")
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"Particle-filter checkpoint array {name!r} has shape "
                f"{arrays[name].shape}; expected {shape}."
            )
    return ParticleFilterState(
        particles=arrays["particles"],
        log_weights=arrays["log_weights"],
        time=arrays["time"],
        log_likelihood=arrays["log_likelihood"],
        valid=arrays["valid"].astype(bool),
        status=arrays["status"].astype(jnp.int32),
        root_key=jr.wrap_key_data(arrays["root_key_data"].astype(jnp.uint32)),
        step_index=step_index,
        num_particles=count,
        problem_id=problem.problem_id,
        resampling_method=method,
        resampling_policy=policy,
        resampling_threshold=threshold,
    )


__all__ = [
    "bootstrap_particle_filter",
    "effective_sample_size",
    "initialize_particle_filter",
    "normalize_log_weights",
    "PARTICLE_FILTER_NONFINITE",
    "PARTICLE_FILTER_SUCCESS",
    "read_particle_filter_checkpoint",
    "PARTICLE_FILTER_TRANSITION_FAILURE",
    "PARTICLE_FILTER_WEIGHT_DEGENERACY",
    "particle_filter_diagnostics",
    "ParticleFilterDiagnostics",
    "particle_filter_predictive",
    "ParticleFilterResult",
    "ParticleFilterState",
    "particle_filter_status_name",
    "ParticleFilterStatus",
    "ParticleFilterStep",
    "particle_filter_step",
    "resample_indices",
    "ResamplingMethod",
    "write_particle_filter_checkpoint",
    "ResamplingPolicy",
    "sample_particle_ancestry_paths",
    "sample_particle_backward_paths",
]
