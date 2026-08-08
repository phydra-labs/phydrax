#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule
from ..stochastic._state_space import (
    LinearGaussianObservationModel,
    LinearGaussianTransitionKernel,
    state_space_key,
    StateSpaceProblem,
    StateSpaceStepContext,
    TransitionSample,
)
from ._particle import (
    effective_sample_size,
    normalize_log_weights,
    PARTICLE_FILTER_NONFINITE,
    PARTICLE_FILTER_SUCCESS,
    PARTICLE_FILTER_WEIGHT_DEGENERACY,
    resample_indices,
    ResamplingMethod,
    ResamplingPolicy,
)


AuxiliaryResamplingPolicy: TypeAlias = Literal["always", "ess", "never"]
GuidedParticleFilterStatus: TypeAlias = Literal[
    "success", "proposal_failure", "weight_degeneracy", "nonfinite"
]
GUIDED_PARTICLE_PROPOSAL_FAILURE = 4


class ParticleProposalSample(StrictModule):
    """One proposal draw and its exact target-to-proposal log-density correction."""

    values: Array
    log_importance_correction: Array
    valid: Array
    status: Array
    proposal_id: str = eqx.field(static=True)


class AbstractParticleProposal(StrictModule):
    """Observation-guided Markov proposal with normalized density correction."""

    state_shape: tuple[int, ...]
    proposal_id: str

    @abstractmethod
    def propose(
        self,
        key: Key[Array, ""],
        problem: StateSpaceProblem,
        previous_state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        observation: ArrayLike,
        mask: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> ParticleProposalSample:
        raise NotImplementedError

    @abstractmethod
    def lookahead_log_weight(
        self,
        problem: StateSpaceProblem,
        previous_state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        observation: ArrayLike,
        mask: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> Array:
        raise NotImplementedError


class BootstrapParticleProposal(AbstractParticleProposal):
    """Canonical transition-prior proposal with zero density correction."""

    state_shape: tuple[int, ...] = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(self, state_shape: tuple[int, ...], /):
        shape = tuple(int(size) for size in state_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("state_shape dimensions must be positive.")
        self.state_shape = shape
        self.proposal_id = "bootstrap"

    def propose(
        self,
        key,
        problem,
        previous_state,
        t0,
        t1,
        observation,
        mask,
        context,
        /,
    ) -> ParticleProposalSample:
        del observation, mask
        _validate_problem_shape(problem, self.state_shape)
        sample = problem.model.transition.sample(key, previous_state, t0, t1, context)
        valid = jnp.all(sample.valid) & jnp.all(sample.status == 0)
        return ParticleProposalSample(
            values=sample.values,
            log_importance_correction=jnp.zeros((), dtype=sample.values.dtype),
            valid=valid,
            status=jnp.where(valid, 0, GUIDED_PARTICLE_PROPOSAL_FAILURE).astype(
                jnp.int32
            ),
            proposal_id=self.proposal_id,
        )

    def lookahead_log_weight(
        self,
        problem,
        previous_state,
        t0,
        t1,
        observation,
        mask,
        context,
        /,
    ) -> Array:
        del previous_state, t0, t1, observation, mask, context
        _validate_problem_shape(problem, self.state_shape)
        return jnp.zeros(())


class CallableGuidedParticleProposal(AbstractParticleProposal):
    """User-defined normalized proposal with Phydrax-computed density correction.

    ``sample`` receives ``(key, problem, previous_state, t0, t1,
    observation, mask, context)``. ``log_prob`` receives the same arguments
    after the key, prefixed by ``next_state``. An optional ``lookahead`` receives
    the arguments of :meth:`lookahead_log_weight`. All log densities must be
    normalized.
    """

    sample_fn: Callable[..., ArrayLike | TransitionSample] = eqx.field(static=True)
    log_prob_fn: Callable[..., ArrayLike] = eqx.field(static=True)
    lookahead_fn: Callable[..., ArrayLike] | None = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample: Callable[..., ArrayLike | TransitionSample],
        log_prob: Callable[..., ArrayLike],
        /,
        *,
        state_shape: tuple[int, ...],
        lookahead: Callable[..., ArrayLike] | None = None,
        proposal_id: str = "guided",
    ):
        if not callable(sample) or not callable(log_prob):
            raise TypeError("sample and log_prob must be callable.")
        if lookahead is not None and not callable(lookahead):
            raise TypeError("lookahead must be callable or None.")
        shape = tuple(int(size) for size in state_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("state_shape dimensions must be positive.")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("proposal_id must be a non-empty string.")
        self.sample_fn = sample
        self.log_prob_fn = log_prob
        self.lookahead_fn = lookahead
        self.state_shape = shape
        self.proposal_id = proposal_id

    def propose(
        self,
        key,
        problem,
        previous_state,
        t0,
        t1,
        observation,
        mask,
        context,
        /,
    ) -> ParticleProposalSample:
        _validate_problem_shape(problem, self.state_shape)
        if not problem.model.transition.has_log_density:
            raise ValueError(
                "A guided proposal requires a transition with a normalized log density."
            )
        raw = self.sample_fn(
            key, problem, previous_state, t0, t1, observation, mask, context
        )
        if isinstance(raw, TransitionSample):
            values = raw.values
            sample_valid = jnp.all(raw.valid) & jnp.all(raw.status == 0)
        else:
            values = jnp.asarray(raw)
            sample_valid = jnp.asarray(True)
        previous = jnp.asarray(previous_state)
        if values.shape != previous.shape:
            raise ValueError("Guided proposal samples must preserve state shape.")
        proposal_log_prob = jnp.asarray(
            self.log_prob_fn(
                values,
                problem,
                previous,
                t0,
                t1,
                observation,
                mask,
                context,
            )
        ).reshape(())
        target_log_prob = jnp.asarray(
            problem.model.transition.log_prob(values, previous, t0, t1, context)
        ).reshape(())
        correction = target_log_prob - proposal_log_prob
        valid = (
            sample_valid
            & jnp.all(jnp.isfinite(values))
            & jnp.isfinite(proposal_log_prob)
            & jnp.isfinite(target_log_prob)
            & jnp.isfinite(correction)
        )
        return ParticleProposalSample(
            values=values,
            log_importance_correction=correction,
            valid=valid,
            status=jnp.where(valid, 0, GUIDED_PARTICLE_PROPOSAL_FAILURE).astype(
                jnp.int32
            ),
            proposal_id=self.proposal_id,
        )

    def lookahead_log_weight(
        self,
        problem,
        previous_state,
        t0,
        t1,
        observation,
        mask,
        context,
        /,
    ) -> Array:
        _validate_problem_shape(problem, self.state_shape)
        if self.lookahead_fn is None:
            return jnp.zeros(())
        return jnp.asarray(
            self.lookahead_fn(
                problem,
                previous_state,
                t0,
                t1,
                observation,
                mask,
                context,
            )
        ).reshape(())


class LinearGaussianGuidedParticleProposal(AbstractParticleProposal):
    """Fully adapted Gaussian proposal using the current masked observation."""

    state_shape: tuple[int, ...] = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(self, state_shape: tuple[int, ...], /):
        shape = tuple(int(size) for size in state_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("state_shape dimensions must be positive.")
        self.state_shape = shape
        self.proposal_id = "linear-gaussian-fully-adapted"

    def propose(
        self,
        key,
        problem,
        previous_state,
        t0,
        t1,
        observation,
        mask,
        context,
        /,
    ) -> ParticleProposalSample:
        mean, covariance, lookahead, valid = _linear_gaussian_condition(
            problem, previous_state, t0, t1, observation, mask, context
        )
        eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
        factor = eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, 0.0))[None, :]
        flat_draw = mean + factor @ jr.normal(key, mean.shape, dtype=mean.dtype)
        values = flat_draw.reshape(self.state_shape)
        proposal_log_prob = _gaussian_log_prob(flat_draw, mean, covariance)
        target_log_prob = jnp.asarray(
            problem.model.transition.log_prob(values, previous_state, t0, t1, context)
        ).reshape(())
        correction = target_log_prob - proposal_log_prob
        valid = (
            valid
            & jnp.all(eigenvalues > 0.0)
            & jnp.all(jnp.isfinite(values))
            & jnp.isfinite(correction)
            & jnp.isfinite(lookahead)
        )
        return ParticleProposalSample(
            values=values,
            log_importance_correction=correction,
            valid=valid,
            status=jnp.where(valid, 0, GUIDED_PARTICLE_PROPOSAL_FAILURE).astype(
                jnp.int32
            ),
            proposal_id=self.proposal_id,
        )

    def lookahead_log_weight(
        self,
        problem,
        previous_state,
        t0,
        t1,
        observation,
        mask,
        context,
        /,
    ) -> Array:
        _, _, lookahead, valid = _linear_gaussian_condition(
            problem, previous_state, t0, t1, observation, mask, context
        )
        return jnp.where(valid, lookahead, -jnp.inf)


class GuidedParticleFilterState(StrictModule):
    """Terminal state of a guided or auxiliary particle filter."""

    particles: Array
    log_weights: Array
    time: Array
    log_likelihood: Array
    valid: Array
    status: Array
    root_key: Array
    num_particles: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)


class GuidedParticleFilterResult(StrictModule):
    """Guided particle history with both proposal and final genealogies."""

    predicted_particles: Array
    proposal_log_corrections: Array
    auxiliary_log_weights: Array
    posterior_log_weights: Array
    particles: Array
    log_weights: Array
    proposal_ancestor_indices: Array
    ancestor_indices: Array
    proposal_valid: Array
    effective_sample_sizes: Array
    auxiliary_resampled: Array
    resampled: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    step_valid: Array
    valid: Array
    status: Array
    times: Array
    final_state: GuidedParticleFilterState
    problem: StateSpaceProblem
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    num_particles: int = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)
    auxiliary_resampling_policy: AuxiliaryResamplingPolicy = eqx.field(static=True)
    resampling_method: ResamplingMethod = eqx.field(static=True)
    resampling_policy: ResamplingPolicy = eqx.field(static=True)
    resampling_threshold: float = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


def guided_particle_filter_status_name(value: int, /) -> GuidedParticleFilterStatus:
    code = int(value)
    if code == PARTICLE_FILTER_SUCCESS:
        return "success"
    if code == GUIDED_PARTICLE_PROPOSAL_FAILURE:
        return "proposal_failure"
    if code == PARTICLE_FILTER_WEIGHT_DEGENERACY:
        return "weight_degeneracy"
    if code == PARTICLE_FILTER_NONFINITE:
        return "nonfinite"
    raise ValueError(f"Unknown guided particle-filter status code {code}.")


def _validate_problem_shape(
    problem: StateSpaceProblem, state_shape: tuple[int, ...], /
) -> None:
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    if problem.model.state_shape != state_shape:
        raise ValueError("Proposal and state-space model state shapes do not match.")


def _gaussian_log_prob(value: Array, mean: Array, covariance: Array, /) -> Array:
    scale = jnp.linalg.cholesky(covariance)
    residual = value - mean
    solved = jax.scipy.linalg.solve_triangular(scale, residual[..., None], lower=True)[
        ..., 0
    ]
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(scale, axis1=-2, axis2=-1)), axis=-1)
    return -0.5 * (
        jnp.sum(solved**2, axis=-1) + logdet + value.shape[-1] * jnp.log(2.0 * jnp.pi)
    )


def _linear_gaussian_condition(
    problem: StateSpaceProblem,
    previous_state: ArrayLike,
    t0: ArrayLike,
    t1: ArrayLike,
    observation_value: ArrayLike,
    mask_value: ArrayLike,
    context: StateSpaceStepContext,
    /,
) -> tuple[Array, Array, Array, Array]:
    transition = problem.model.transition
    observation = problem.model.observation
    if not isinstance(transition, LinearGaussianTransitionKernel) or not isinstance(
        observation, LinearGaussianObservationModel
    ):
        raise TypeError(
            "LinearGaussianGuidedParticleProposal requires linear-Gaussian "
            "transition and observation models."
        )
    state_size = prod(problem.model.state_shape)
    observation_size = prod(problem.model.observation_shape)
    previous = jnp.asarray(previous_state).reshape((state_size,))
    transition_parameters = transition.parameters(t0, t1, context)
    transition_matrix = transition_parameters.transition
    transition_offset = transition_parameters.offset
    process_covariance = transition_parameters.covariance
    observation_matrix, observation_offset, observation_covariance = (
        observation.parameters(t1, context)
    )
    transition_mean = transition_matrix @ previous + transition_offset
    values = jnp.asarray(observation_value).reshape((observation_size,))
    mask = jnp.asarray(mask_value, dtype=bool).reshape((observation_size,))
    active = mask.astype(transition_mean.dtype)
    effective_matrix = observation_matrix * active[:, None]
    effective_covariance = observation_covariance * active[:, None] * active[
        None, :
    ] + jnp.diag(1.0 - active)
    residual = jnp.where(
        mask, values - observation_matrix @ transition_mean - observation_offset, 0.0
    )
    innovation_covariance = (
        effective_matrix @ process_covariance @ effective_matrix.T + effective_covariance
    )
    scale = jnp.linalg.cholesky(innovation_covariance)
    gain = jnp.linalg.solve(
        innovation_covariance,
        effective_matrix @ process_covariance,
    ).T
    conditional_mean = transition_mean + gain @ residual
    identity = jnp.eye(state_size, dtype=transition_mean.dtype)
    update = identity - gain @ effective_matrix
    conditional_covariance = (
        update @ process_covariance @ update.T + gain @ effective_covariance @ gain.T
    )
    conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
    solved = jax.scipy.linalg.solve_triangular(scale, residual[:, None], lower=True)[:, 0]
    diagonal = jnp.diagonal(scale)
    count = jnp.sum(mask)
    lookahead = -0.5 * (
        jnp.sum(solved**2)
        + 2.0 * jnp.sum(jnp.log(diagonal))
        + count * jnp.log(2.0 * jnp.pi)
    )
    valid = (
        jnp.all(jnp.isfinite(conditional_mean))
        & jnp.all(jnp.isfinite(conditional_covariance))
        & jnp.all(jnp.isfinite(scale))
        & jnp.all(diagonal > 0.0)
        & jnp.isfinite(lookahead)
    )
    return conditional_mean, conditional_covariance, lookahead, valid


def _configuration(
    num_particles: int,
    auxiliary_policy: AuxiliaryResamplingPolicy,
    method: ResamplingMethod,
    policy: ResamplingPolicy,
    threshold: float,
) -> tuple[
    int,
    AuxiliaryResamplingPolicy,
    ResamplingMethod,
    ResamplingPolicy,
    float,
]:
    count = int(num_particles)
    if count < 1:
        raise ValueError("num_particles must be positive.")
    if auxiliary_policy not in ("always", "ess", "never"):
        raise ValueError("Unknown auxiliary_resampling_policy.")
    if method not in ("systematic", "stratified", "multinomial", "residual"):
        raise ValueError("Unknown resampling_method.")
    if policy not in ("ess", "always", "never"):
        raise ValueError("Unknown resampling_policy.")
    level = float(threshold)
    if not np.isfinite(level) or not 0.0 < level <= 1.0:
        raise ValueError("resampling_threshold must lie in (0, 1].")
    return count, auxiliary_policy, method, policy, level


def _resampling_decision(
    policy: str,
    ess: Array,
    count: int,
    threshold: float,
    /,
) -> bool:
    return policy == "always" or (policy == "ess" and float(ess) < threshold * count)


def guided_particle_filter(
    key: Key[Array, ""],
    problem: StateSpaceProblem,
    proposal: AbstractParticleProposal,
    /,
    *,
    num_particles: int,
    auxiliary_resampling_policy: AuxiliaryResamplingPolicy = "always",
    resampling_method: ResamplingMethod = "systematic",
    resampling_policy: ResamplingPolicy = "ess",
    resampling_threshold: float = 0.5,
    raise_on_failure: bool = False,
) -> GuidedParticleFilterResult:
    """Run a corrected guided/auxiliary particle filter on the canonical schedule."""
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    if not isinstance(proposal, AbstractParticleProposal):
        raise TypeError("proposal must implement AbstractParticleProposal.")
    _validate_problem_shape(problem, proposal.state_shape)
    count, auxiliary_policy, method, policy, threshold = _configuration(
        num_particles,
        auxiliary_resampling_policy,
        resampling_method,
        resampling_policy,
        resampling_threshold,
    )
    sequence = problem.observations
    case_shape = sequence.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = sequence.num_steps
    state_shape = problem.model.state_shape
    state_rank = len(state_shape)
    identity = jnp.arange(count, dtype=jnp.int32)

    initial_cases = []
    initial_valid = []
    for case_index, case_id in enumerate(sequence.case_ids):
        draws = []
        draw_valid = []
        for particle_index in range(count):
            draw_key = state_space_key(
                key, "guided-particle-prior", case_id, 0, member=particle_index
            )
            complete = problem.model.prior.sample(draw_key)
            draw = (
                complete
                if not case_shape
                else complete.reshape((case_count,) + state_shape)[case_index]
            )
            draws.append(draw)
            draw_valid.append(jnp.all(jnp.isfinite(draw)))
        initial_cases.append(jnp.stack(draws))
        initial_valid.append(jnp.all(jnp.stack(draw_valid)))
    particles = jnp.stack(initial_cases)
    log_weights = jnp.full(
        (case_count, count), -jnp.log(float(count)), dtype=particles.dtype
    )
    times = problem.initial_time.reshape((case_count,))
    cumulative = jnp.zeros((case_count,), dtype=particles.dtype)
    alive = jnp.stack(initial_valid)
    final_status = jnp.where(
        alive, PARTICLE_FILTER_SUCCESS, PARTICLE_FILTER_NONFINITE
    ).astype(jnp.int32)

    predicted_history: list[Array] = []
    correction_history: list[Array] = []
    lookahead_history: list[Array] = []
    posterior_history: list[Array] = []
    particle_history: list[Array] = []
    weight_history: list[Array] = []
    proposal_ancestor_history: list[Array] = []
    ancestor_history: list[Array] = []
    proposal_valid_history: list[Array] = []
    ess_history: list[Array] = []
    auxiliary_resampled_history: list[Array] = []
    resampled_history: list[Array] = []
    increment_history: list[Array] = []
    cumulative_history: list[Array] = []
    valid_history: list[Array] = []
    status_history: list[Array] = []

    flat_times = sequence.times.reshape((case_count, num_steps))
    flat_active = sequence.step_valid.reshape((case_count, num_steps))
    flat_values = sequence.values.reshape(
        (case_count, num_steps) + sequence.observation_shape
    )
    flat_masks = sequence.observation_mask.reshape(
        (case_count, num_steps) + sequence.observation_shape
    )

    for index in range(num_steps):
        step_predicted = []
        step_corrections = []
        step_lookahead = []
        step_posterior = []
        step_particles = []
        step_weights = []
        step_proposal_ancestors = []
        step_ancestors = []
        step_proposal_valid = []
        step_ess = []
        step_auxiliary_resampled = []
        step_resampled = []
        step_increment = []
        step_cumulative = []
        step_validity = []
        step_status = []

        for case_index, case_id in enumerate(sequence.case_ids):
            active = bool(flat_active[case_index, index])
            if not active or not bool(alive[case_index]):
                step_predicted.append(particles[case_index])
                step_corrections.append(jnp.zeros((count,), dtype=particles.dtype))
                step_lookahead.append(jnp.zeros((count,), dtype=particles.dtype))
                step_posterior.append(log_weights[case_index])
                step_particles.append(particles[case_index])
                step_weights.append(log_weights[case_index])
                step_proposal_ancestors.append(identity)
                step_ancestors.append(identity)
                step_proposal_valid.append(
                    jnp.full((count,), alive[case_index], dtype=bool)
                )
                step_ess.append(effective_sample_size(log_weights[case_index]))
                step_auxiliary_resampled.append(jnp.asarray(False))
                step_resampled.append(jnp.asarray(False))
                step_increment.append(jnp.asarray(0.0, dtype=particles.dtype))
                step_cumulative.append(cumulative[case_index])
                step_validity.append(alive[case_index])
                step_status.append(final_status[case_index])
                continue

            value = flat_values[case_index, index]
            mask = flat_masks[case_index, index]
            start = times[case_index]
            context = problem.step_context(case_index, index)
            end = flat_times[case_index, index]
            lookahead = jnp.stack(
                [
                    proposal.lookahead_log_weight(
                        problem,
                        particles[case_index, particle_index],
                        start,
                        end,
                        value,
                        mask,
                        context,
                    )
                    for particle_index in range(count)
                ]
            )
            auxiliary_candidates = log_weights[case_index] + lookahead
            auxiliary_weights, auxiliary_normalizer, auxiliary_valid = (
                normalize_log_weights(auxiliary_candidates)
            )
            auxiliary_ess = effective_sample_size(auxiliary_weights)
            use_auxiliary = bool(auxiliary_valid) and (
                _resampling_decision(auxiliary_policy, auxiliary_ess, count, threshold)
            )
            if use_auxiliary:
                auxiliary_key = state_space_key(
                    key, "guided-particle-auxiliary", case_id, index
                )
                proposal_ancestors = resample_indices(
                    auxiliary_key, auxiliary_weights, method=method
                )
            else:
                proposal_ancestors = identity
            parents = particles[case_index, proposal_ancestors]
            parent_lookahead = lookahead[proposal_ancestors]

            proposals = []
            for particle_index in range(count):
                proposal_key = state_space_key(
                    key,
                    "guided-particle-proposal",
                    case_id,
                    index,
                    member=particle_index,
                )
                proposals.append(
                    proposal.propose(
                        proposal_key,
                        problem,
                        parents[particle_index],
                        start,
                        end,
                        value,
                        mask,
                        context,
                    )
                )
            predicted = jnp.stack([sample.values for sample in proposals])
            corrections = jnp.stack(
                [sample.log_importance_correction for sample in proposals]
            )
            proposal_valid = jnp.stack([sample.valid for sample in proposals])
            observation_log_weights = jnp.stack(
                [
                    problem.model.observation.log_prob(
                        value,
                        predicted[particle_index],
                        end,
                        mask,
                        context,
                    )
                    for particle_index in range(count)
                ]
            )
            finite_observations = ~jnp.isnan(observation_log_weights)
            proposal_valid = proposal_valid & finite_observations
            corrected = jnp.where(
                proposal_valid,
                observation_log_weights + corrections,
                -jnp.inf,
            )
            if use_auxiliary:
                child_candidates = corrected - parent_lookahead
                log_increment = (
                    auxiliary_normalizer
                    + jax.scipy.special.logsumexp(child_candidates)
                    - jnp.log(float(count))
                )
            else:
                child_candidates = log_weights[case_index, proposal_ancestors] + corrected
                log_increment = jax.scipy.special.logsumexp(child_candidates)
            posterior, _, weights_valid = normalize_log_weights(child_candidates)
            finite_particles = jnp.all(
                jnp.isfinite(predicted),
                axis=tuple(range(1, 1 + state_rank)),
            )
            proposal_case_valid = jnp.any(proposal_valid & finite_particles)
            accepted = bool(
                proposal_case_valid & weights_valid & jnp.isfinite(log_increment)
            )
            posterior_ess = effective_sample_size(posterior)
            post_resample = accepted and _resampling_decision(
                policy, posterior_ess, count, threshold
            )
            if post_resample:
                resampling_key = state_space_key(
                    key, "guided-particle-resampling", case_id, index
                )
                child_indices = resample_indices(resampling_key, posterior, method=method)
                output_particles = predicted[child_indices]
                output_weights = jnp.full_like(posterior, -jnp.log(float(count)))
                ancestors = proposal_ancestors[child_indices]
            elif accepted:
                output_particles = predicted
                output_weights = posterior
                ancestors = proposal_ancestors
            else:
                output_particles = particles[case_index]
                output_weights = log_weights[case_index]
                ancestors = identity
            if accepted:
                status = PARTICLE_FILTER_SUCCESS
            elif not bool(proposal_case_valid):
                status = GUIDED_PARTICLE_PROPOSAL_FAILURE
            elif not bool(weights_valid) or not bool(jnp.isfinite(log_increment)):
                status = PARTICLE_FILTER_WEIGHT_DEGENERACY
            else:
                status = PARTICLE_FILTER_NONFINITE
            next_cumulative = cumulative[case_index] + jnp.where(
                accepted, log_increment, 0.0
            )

            step_predicted.append(predicted)
            step_corrections.append(corrections)
            step_lookahead.append(lookahead)
            step_posterior.append(posterior)
            step_particles.append(output_particles)
            step_weights.append(output_weights)
            step_proposal_ancestors.append(proposal_ancestors)
            step_ancestors.append(ancestors)
            step_proposal_valid.append(proposal_valid)
            step_ess.append(posterior_ess)
            step_auxiliary_resampled.append(jnp.asarray(use_auxiliary))
            step_resampled.append(jnp.asarray(post_resample))
            step_increment.append(jnp.where(accepted, log_increment, 0.0))
            step_cumulative.append(next_cumulative)
            step_validity.append(jnp.asarray(accepted))
            step_status.append(jnp.asarray(status, dtype=jnp.int32))

        particles = jnp.stack(step_particles)
        log_weights = jnp.stack(step_weights)
        cumulative = jnp.stack(step_cumulative)
        step_valid_array = jnp.stack(step_validity)
        alive = alive & step_valid_array
        final_status = jnp.stack(step_status)
        times = jnp.where(flat_active[:, index], flat_times[:, index], times)
        predicted_history.append(jnp.stack(step_predicted))
        correction_history.append(jnp.stack(step_corrections))
        lookahead_history.append(jnp.stack(step_lookahead))
        posterior_history.append(jnp.stack(step_posterior))
        particle_history.append(particles)
        weight_history.append(log_weights)
        proposal_ancestor_history.append(jnp.stack(step_proposal_ancestors))
        ancestor_history.append(jnp.stack(step_ancestors))
        proposal_valid_history.append(jnp.stack(step_proposal_valid))
        ess_history.append(jnp.stack(step_ess))
        auxiliary_resampled_history.append(jnp.stack(step_auxiliary_resampled))
        resampled_history.append(jnp.stack(step_resampled))
        increment_history.append(jnp.stack(step_increment))
        cumulative_history.append(cumulative)
        valid_history.append(step_valid_array)
        status_history.append(final_status)

    def restore(history: list[Array], trailing_shape: tuple[int, ...] = ()) -> Array:
        stacked = jnp.stack(history, axis=1)
        return stacked.reshape(case_shape + (num_steps,) + trailing_shape)

    final_state = GuidedParticleFilterState(
        particles=particles.reshape(case_shape + (count,) + state_shape),
        log_weights=log_weights.reshape(case_shape + (count,)),
        time=times.reshape(case_shape),
        log_likelihood=cumulative.reshape(case_shape),
        valid=alive.reshape(case_shape),
        status=final_status.reshape(case_shape),
        root_key=jnp.asarray(key),
        num_particles=count,
        problem_id=problem.problem_id,
        proposal_id=proposal.proposal_id,
    )
    result = GuidedParticleFilterResult(
        predicted_particles=restore(predicted_history, (count,) + state_shape),
        proposal_log_corrections=restore(correction_history, (count,)),
        auxiliary_log_weights=restore(lookahead_history, (count,)),
        posterior_log_weights=restore(posterior_history, (count,)),
        particles=restore(particle_history, (count,) + state_shape),
        log_weights=restore(weight_history, (count,)),
        proposal_ancestor_indices=restore(proposal_ancestor_history, (count,)),
        ancestor_indices=restore(ancestor_history, (count,)),
        proposal_valid=restore(proposal_valid_history, (count,)),
        effective_sample_sizes=restore(ess_history),
        auxiliary_resampled=restore(auxiliary_resampled_history),
        resampled=restore(resampled_history),
        incremental_log_likelihood=restore(increment_history),
        cumulative_log_likelihood=restore(cumulative_history),
        step_valid=sequence.step_valid,
        valid=restore(valid_history),
        status=restore(status_history),
        times=sequence.times,
        final_state=final_state,
        problem=problem,
        state_shape=state_shape,
        observation_shape=problem.model.observation_shape,
        case_shape=case_shape,
        case_axes=sequence.case_axes,
        case_ids=sequence.case_ids,
        num_particles=count,
        proposal_id=proposal.proposal_id,
        auxiliary_resampling_policy=auxiliary_policy,
        resampling_method=method,
        resampling_policy=policy,
        resampling_threshold=threshold,
    )
    if raise_on_failure and not bool(jnp.all(result.successful)):
        raise RuntimeError("Guided particle filtering failed for at least one case.")
    return result


__all__ = [
    "AbstractParticleProposal",
    "AuxiliaryResamplingPolicy",
    "BootstrapParticleProposal",
    "CallableGuidedParticleProposal",
    "GUIDED_PARTICLE_PROPOSAL_FAILURE",
    "guided_particle_filter",
    "GuidedParticleFilterResult",
    "GuidedParticleFilterState",
    "GuidedParticleFilterStatus",
    "guided_particle_filter_status_name",
    "LinearGaussianGuidedParticleProposal",
    "ParticleProposalSample",
]
