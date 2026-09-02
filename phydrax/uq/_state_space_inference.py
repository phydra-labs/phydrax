#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

import phydrax.ein as ein

from .._strict import StrictModule
from ..stochastic._solver_transition import FiniteStateTransitionKernel
from ..stochastic._state_space import (
    CategoricalStatePrior,
    GaussianStatePrior,
    LinearGaussianObservationModel,
    LinearGaussianTransitionKernel,
    StateSpaceProblem,
)
from ._kalman import kalman_filter, KalmanExecutionMethod
from ._posterior_terms import AbstractPosteriorTerm


ExactStateSpaceMethod: TypeAlias = Literal["auto", "kalman", "finite-state"]

EXACT_STATE_SPACE_SUCCESS = 0
EXACT_STATE_SPACE_TRANSITION_FAILURE = 1
EXACT_STATE_SPACE_DEGENERATE_LIKELIHOOD = 2
EXACT_STATE_SPACE_NONFINITE = 3
EXACT_STATE_SPACE_STATE_MISMATCH = 4


class FiniteStateFilterResult(StrictModule):
    """Exact finite-state forward probabilities and likelihood increments."""

    predicted_probabilities: Array
    filtered_probabilities: Array
    transition_matrices: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    step_valid: Array
    valid: Array
    status: Array
    final_probabilities: Array
    problem: StateSpaceProblem
    state_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    input_id: str | None = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    execution_method: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


class FiniteStateSmootherResult(StrictModule):
    """Exact fixed-interval state and adjacent-transition posterior probabilities."""

    smoothed_probabilities: Array
    initial_probabilities: Array
    transition_probabilities: Array
    step_valid: Array
    valid: Array
    status: Array
    filter_result: FiniteStateFilterResult
    input_id: str | None = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.filter_result.successful


class FiniteStateViterbiResult(StrictModule):
    """Exact maximum-joint-probability finite-state path."""

    state_indices: Array
    states: Array
    initial_state_indices: Array
    initial_states: Array
    joint_log_probability: Array
    step_valid: Array
    valid: Array
    status: Array
    problem: StateSpaceProblem
    input_id: str | None = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


class FiniteStateTransitionCountResult(StrictModule):
    """Posterior expected endpoint-transition counts over physical intervals."""

    per_case_counts: Array
    total_counts: Array
    transition_probabilities: Array
    step_valid: Array
    valid: Array
    status: Array
    smoother_result: FiniteStateSmootherResult
    input_id: str | None = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.smoother_result.successful


class FiniteStateSufficientStatisticsResult(StrictModule):
    """Posterior expectation of a transition sufficient-statistic PyTree."""

    per_step_statistics: PyTree[Any]
    per_case_statistics: PyTree[Any]
    total_statistics: PyTree[Any]
    step_valid: Array
    valid: Array
    status: Array
    smoother_result: FiniteStateSmootherResult
    input_id: str | None = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.smoother_result.successful


class ExactStateSpaceLikelihood(StrictModule):
    """Backend-independent exact marginal state-space likelihood."""

    per_case_log_likelihood: Array
    total_log_likelihood: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    step_valid: Array
    valid: Array
    status: Array
    backend: Any
    problem: StateSpaceProblem
    method: str = eqx.field(static=True)
    temporal_method: str = eqx.field(static=True)
    input_id: str | None = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


class StateSpaceMarginalLikelihood(AbstractPosteriorTerm):
    """Exact state-space likelihood term for MAP, MCMC, SMC, and Laplace inference.

    ``problem`` receives physical parameters and must return a prevalidated
    ``StateSpaceProblem`` with the same static structure on every call. Inside a
    differentiated posterior, update a template with ``equinox.tree_at`` rather
    than invoking constructors that perform host-side validation.
    """

    problem_fn: Callable[[PyTree[Any]], StateSpaceProblem] = eqx.field(static=True)
    method: ExactStateSpaceMethod = eqx.field(static=True)
    covariance_regularization: float = eqx.field(static=True)
    temporal_method: KalmanExecutionMethod = eqx.field(static=True)

    def __init__(
        self,
        problem: Callable[[PyTree[Any]], StateSpaceProblem],
        /,
        *,
        method: ExactStateSpaceMethod = "auto",
        covariance_regularization: float = 0.0,
        temporal_method: KalmanExecutionMethod = "auto",
        label: str = "state_space",
    ):
        if not callable(problem):
            raise TypeError("problem must be callable.")
        _validate_method(method)
        if temporal_method not in ("sequential", "parallel", "auto"):
            raise ValueError(
                "temporal_method must be 'sequential', 'parallel', or 'auto'."
            )
        regularization = float(covariance_regularization)
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("covariance_regularization must be finite and nonnegative.")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string.")
        self.problem_fn = problem
        self.method = method
        self.covariance_regularization = regularization
        self.temporal_method = temporal_method
        self.label = label

    def evaluate(self, parameters: PyTree[Any], /) -> ExactStateSpaceLikelihood:
        """Return the complete exact filtering and likelihood result."""
        problem = self.problem_fn(parameters)
        if not isinstance(problem, StateSpaceProblem):
            raise TypeError("problem(parameters) must return a StateSpaceProblem.")
        return exact_state_space_log_likelihood(
            problem,
            method=self.method,
            covariance_regularization=self.covariance_regularization,
            temporal_method=self.temporal_method,
        )

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        return self.evaluate(parameters).per_case_log_likelihood


class StateSpaceIdentifiabilityReport(StrictModule):
    """Observed-information rank and weak parameter directions at one point."""

    score: Array
    observed_information: Array
    eigenvalues: Array
    eigenvectors: Array
    identifiable: Array
    weak_directions: Array
    condition_number: Array
    score_norm: Array
    finite: bool = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    numerical_rank: int = eqx.field(static=True)
    parameter_paths: tuple[str, ...] = eqx.field(static=True)
    parameter_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)

    @property
    def full_rank(self) -> bool:
        return self.finite and self.numerical_rank == self.dimension


def _validate_method(method: str, /) -> None:
    if method not in ("auto", "kalman", "finite-state"):
        raise ValueError("method must be 'auto', 'kalman', or 'finite-state'.")


def _resolved_method(problem: StateSpaceProblem, method: ExactStateSpaceMethod, /) -> str:
    _validate_method(method)
    prior = problem.model.prior
    transition = problem.model.transition
    observation = problem.model.observation
    linear = (
        isinstance(prior, GaussianStatePrior)
        and isinstance(transition, LinearGaussianTransitionKernel)
        and isinstance(observation, LinearGaussianObservationModel)
    )
    finite = isinstance(prior, CategoricalStatePrior) and isinstance(
        transition, FiniteStateTransitionKernel
    )
    if method == "auto":
        if linear:
            return "kalman"
        if finite:
            return "finite-state"
        raise TypeError(
            "No exact likelihood backend matches this state-space model; use a "
            "particle likelihood for nonlinear or solver-defined transitions."
        )
    if method == "kalman" and not linear:
        raise TypeError(
            "The Kalman likelihood requires GaussianStatePrior, "
            "LinearGaussianTransitionKernel, and LinearGaussianObservationModel."
        )
    if method == "finite-state" and not finite:
        raise TypeError(
            "The finite-state likelihood requires CategoricalStatePrior and "
            "FiniteStateTransitionKernel."
        )
    return method


def _finite_state_filter(problem: StateSpaceProblem, /) -> FiniteStateFilterResult:
    prior = problem.model.prior
    transition = problem.model.transition
    observation = problem.model.observation
    if not isinstance(prior, CategoricalStatePrior) or not isinstance(
        transition, FiniteStateTransitionKernel
    ):
        raise TypeError(
            "Finite-state filtering requires CategoricalStatePrior and "
            "FiniteStateTransitionKernel."
        )

    sequence = problem.observations
    case_shape = sequence.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = sequence.num_steps
    num_states = int(prior.states.shape[0])
    probabilities = prior.probabilities.reshape((case_count, num_states))
    times = problem.initial_time.reshape((case_count,))
    alive = jnp.ones((case_count,), dtype=bool)
    cumulative = jnp.zeros((case_count,), dtype=probabilities.dtype)
    state_alignment = jnp.all(prior.states == transition.generator.states)

    predicted_history: list[Array] = []
    filtered_history: list[Array] = []
    transition_history: list[Array] = []
    increment_history: list[Array] = []
    cumulative_history: list[Array] = []
    valid_history: list[Array] = []
    status_history: list[Array] = []

    flat_times = sequence.times.reshape((case_count, num_steps))
    flat_values = sequence.values.reshape(
        (case_count, num_steps) + sequence.observation_shape
    )
    flat_masks = sequence.observation_mask.reshape(
        (case_count, num_steps) + sequence.observation_shape
    )
    flat_active = sequence.step_valid.reshape((case_count, num_steps))

    def transition_matrix(duration: Array) -> Array:
        return transition.generator.transition_matrix(duration)

    def state_log_likelihood(
        value: Array,
        mask: Array,
        time: Array,
        case_index: Array,
        step_index: int,
    ) -> Array:
        context = problem.step_context(case_index, step_index)
        return jax.vmap(
            lambda state: observation.log_prob(value, state, time, mask, context)
        )(prior.states)

    for index in range(num_steps):
        active = flat_active[:, index]
        target_time = flat_times[:, index]
        safe_time = jnp.where(active, target_time, times)
        durations = safe_time - times
        matrices = jax.vmap(transition_matrix)(durations)
        matrix_finite = jnp.all(jnp.isfinite(matrices), axis=(-1, -2))
        matrix_nonnegative = jnp.all(matrices >= 0.0, axis=(-1, -2))
        row_sums = jnp.sum(matrices, axis=-1)
        matrix_stochastic = jnp.all(
            jnp.abs(row_sums - 1.0) <= 1e-5,
            axis=-1,
        )
        transition_valid = (
            matrix_finite & matrix_nonnegative & matrix_stochastic & (durations >= 0.0)
        )
        predicted = ein.contract("ci,cij->cj", probabilities, matrices)

        log_likelihoods = jax.vmap(
            lambda value, mask, time, case_index: state_log_likelihood(
                value, mask, time, case_index, index
            )
        )(
            flat_values[:, index],
            flat_masks[:, index],
            safe_time,
            jnp.arange(case_count, dtype=jnp.int32),
        )
        likelihood_valid = ~jnp.any(jnp.isnan(log_likelihoods), axis=-1)
        log_joint = jnp.where(
            predicted > 0.0,
            jnp.log(predicted) + log_likelihoods,
            -jnp.inf,
        )
        increment = jax.scipy.special.logsumexp(log_joint, axis=-1)
        normalizer_valid = jnp.isfinite(increment)
        candidate_valid = (
            alive
            & state_alignment
            & transition_valid
            & likelihood_valid
            & normalizer_valid
        )
        accepted = active & candidate_valid
        posterior = jnp.exp(log_joint - increment[:, None])
        posterior = jnp.where(jnp.isfinite(posterior), posterior, 0.0)
        next_probabilities = jnp.where(accepted[:, None], posterior, probabilities)
        accepted_increment = jnp.where(accepted, increment, 0.0)
        cumulative = cumulative + accepted_increment
        step_valid = jnp.where(active, candidate_valid, alive)
        status = jnp.where(
            ~active,
            EXACT_STATE_SPACE_SUCCESS,
            jnp.where(
                ~state_alignment,
                EXACT_STATE_SPACE_STATE_MISMATCH,
                jnp.where(
                    ~transition_valid,
                    EXACT_STATE_SPACE_TRANSITION_FAILURE,
                    jnp.where(
                        ~likelihood_valid,
                        EXACT_STATE_SPACE_NONFINITE,
                        jnp.where(
                            ~normalizer_valid,
                            EXACT_STATE_SPACE_DEGENERATE_LIKELIHOOD,
                            EXACT_STATE_SPACE_SUCCESS,
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)

        predicted_history.append(jnp.where(active[:, None], predicted, probabilities))
        filtered_history.append(next_probabilities)
        transition_history.append(matrices)
        increment_history.append(accepted_increment)
        cumulative_history.append(cumulative)
        valid_history.append(step_valid)
        status_history.append(status)
        probabilities = next_probabilities
        times = safe_time
        alive = alive & jnp.where(active, candidate_valid, True)

    def restore_steps(values: list[Array], trailing_shape: tuple[int, ...] = ()) -> Array:
        stacked = jnp.stack(values, axis=1)
        return stacked.reshape(case_shape + (num_steps,) + trailing_shape)

    return FiniteStateFilterResult(
        predicted_probabilities=restore_steps(predicted_history, (num_states,)),
        filtered_probabilities=restore_steps(filtered_history, (num_states,)),
        transition_matrices=restore_steps(transition_history, (num_states, num_states)),
        incremental_log_likelihood=restore_steps(increment_history),
        cumulative_log_likelihood=restore_steps(cumulative_history),
        step_valid=sequence.step_valid,
        valid=restore_steps(valid_history),
        status=restore_steps(status_history),
        final_probabilities=probabilities.reshape(case_shape + (num_states,)),
        problem=problem,
        state_shape=prior.state_shape,
        case_shape=case_shape,
        case_ids=sequence.case_ids,
        model_id=problem.model.model_id,
        problem_id=problem.problem_id,
        sequence_id=sequence.sequence_id,
        input_id=(
            None if problem.input_signal is None else problem.input_signal.input_id
        ),
        process_id=transition.process_id,
        approximation_id=transition.approximation_id,
        execution_method="sequential",
        method_id="finite-state-forward",
    )


def finite_state_backward_smoother(
    result: FiniteStateFilterResult,
    /,
) -> FiniteStateSmootherResult:
    """Run the exact backward recursion over a finite-state filter history.

    ``transition_probabilities[..., step, i, j]`` is the posterior probability
    of state ``i`` at the start and state ``j`` at the end of that physical
    schedule interval. The first interval therefore starts at
    ``problem.initial_time``; padded intervals carry zero transition mass.
    """
    if not isinstance(result, FiniteStateFilterResult):
        raise TypeError("result must be a FiniteStateFilterResult.")
    prior = result.problem.model.prior
    if not isinstance(prior, CategoricalStatePrior):
        raise TypeError("Finite-state smoothing requires CategoricalStatePrior.")
    case_shape = result.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = result.problem.observations.num_steps
    num_states = int(prior.states.shape[0])
    filtered = result.filtered_probabilities.reshape((case_count, num_steps, num_states))
    predicted = result.predicted_probabilities.reshape(
        (case_count, num_steps, num_states)
    )
    matrices = result.transition_matrices.reshape(
        (case_count, num_steps, num_states, num_states)
    )
    active = result.step_valid.reshape((case_count, num_steps))
    valid = result.valid.reshape((case_count, num_steps))
    successful = result.successful.reshape((case_count,))
    smoothed = filtered

    for index in range(num_steps - 2, -1, -1):
        denominator = jnp.where(
            predicted[:, index + 1] > 0.0,
            predicted[:, index + 1],
            1.0,
        )
        ratio = jnp.where(
            predicted[:, index + 1] > 0.0,
            smoothed[:, index + 1] / denominator,
            0.0,
        )
        proposed = filtered[:, index] * ein.contract(
            "cij,cj->ci",
            matrices[:, index + 1],
            ratio,
        )
        pair_valid = (
            successful
            & active[:, index]
            & active[:, index + 1]
            & valid[:, index]
            & valid[:, index + 1]
        )
        smoothed = smoothed.at[:, index].set(
            jnp.where(pair_valid[:, None], proposed, smoothed[:, index])
        )

    prior = result.problem.model.prior
    if not isinstance(prior, CategoricalStatePrior):
        raise TypeError("Finite-state smoothing requires CategoricalStatePrior.")
    prior_probabilities = prior.probabilities.reshape((case_count, num_states))
    transition_probabilities: list[Array] = []
    for index in range(num_steps):
        previous = prior_probabilities if index == 0 else filtered[:, index - 1]
        denominator = jnp.where(
            predicted[:, index] > 0.0,
            predicted[:, index],
            1.0,
        )
        ratio = jnp.where(
            predicted[:, index] > 0.0,
            smoothed[:, index] / denominator,
            0.0,
        )
        pair = previous[:, :, None] * matrices[:, index] * ratio[:, None, :]
        pair_valid = successful & active[:, index] & valid[:, index]
        transition_probabilities.append(jnp.where(pair_valid[:, None, None], pair, 0.0))

    pairwise = jnp.stack(transition_probabilities, axis=1)
    initial = jnp.sum(pairwise[:, 0], axis=-1)
    return FiniteStateSmootherResult(
        smoothed_probabilities=smoothed.reshape(case_shape + (num_steps, num_states)),
        initial_probabilities=initial.reshape(case_shape + (num_states,)),
        transition_probabilities=pairwise.reshape(
            case_shape + (num_steps, num_states, num_states)
        ),
        step_valid=result.step_valid,
        valid=result.valid,
        status=result.status,
        filter_result=result,
        input_id=result.input_id,
        method_id="finite-state-backward",
    )


def finite_state_viterbi(
    result: FiniteStateFilterResult,
    /,
) -> FiniteStateViterbiResult:
    """Return the exact maximum-joint-probability path.

    Ties use JAX's first-maximum rule, giving the lowest state index at each
    deterministic backtracking choice. Zero prior or transition probabilities
    contribute exact negative infinity.
    """
    if not isinstance(result, FiniteStateFilterResult):
        raise TypeError("result must be a FiniteStateFilterResult.")
    problem = result.problem
    prior = problem.model.prior
    observation = problem.model.observation
    if not isinstance(prior, CategoricalStatePrior):
        raise TypeError("Finite-state Viterbi requires CategoricalStatePrior.")
    case_shape = result.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = problem.observations.num_steps
    num_states = int(prior.states.shape[0])
    probabilities = prior.probabilities.reshape((case_count, num_states))
    scores = jnp.where(
        probabilities > 0.0,
        jnp.log(probabilities),
        -jnp.inf,
    )
    matrices = result.transition_matrices.reshape(
        (case_count, num_steps, num_states, num_states)
    )
    sequence = problem.observations
    flat_values = sequence.values.reshape(
        (case_count, num_steps) + sequence.observation_shape
    )
    flat_masks = sequence.observation_mask.reshape(
        (case_count, num_steps) + sequence.observation_shape
    )
    flat_times = sequence.times.reshape((case_count, num_steps))
    active = sequence.step_valid.reshape((case_count, num_steps))
    times = problem.initial_time.reshape((case_count,))
    case_indices = jnp.arange(case_count, dtype=jnp.int32)
    identity = jnp.broadcast_to(
        jnp.arange(num_states, dtype=jnp.int32),
        (case_count, num_states),
    )
    backpointers: list[Array] = []

    for index in range(num_steps):
        target_time = flat_times[:, index]
        safe_time = jnp.where(active[:, index], target_time, times)

        def one_case_log_likelihood(
            value: Array,
            mask: Array,
            time: Array,
            case_index: Array,
        ) -> Array:
            context = problem.step_context(case_index, index)
            return jax.vmap(
                lambda state: observation.log_prob(
                    value,
                    state,
                    time,
                    mask,
                    context,
                )
            )(prior.states)

        log_likelihood = jax.vmap(one_case_log_likelihood)(
            flat_values[:, index],
            flat_masks[:, index],
            safe_time,
            case_indices,
        )
        matrix = matrices[:, index]
        log_transition = jnp.where(
            matrix > 0.0,
            jnp.log(matrix),
            -jnp.inf,
        )
        candidates = scores[:, :, None] + log_transition
        best_previous = jnp.argmax(candidates, axis=1).astype(jnp.int32)
        best_score = jnp.max(candidates, axis=1) + log_likelihood
        step_active = active[:, index]
        scores = jnp.where(step_active[:, None], best_score, scores)
        backpointers.append(jnp.where(step_active[:, None], best_previous, identity))
        times = safe_time

    pointers = jnp.stack(backpointers, axis=1)
    final_indices = jnp.argmax(scores, axis=-1).astype(jnp.int32)
    current = final_indices
    state_indices = jnp.zeros(
        (case_count, num_steps),
        dtype=jnp.int32,
    )
    for index in range(num_steps - 1, -1, -1):
        state_indices = state_indices.at[:, index].set(current)
        current = jnp.take_along_axis(
            pointers[:, index],
            current[:, None],
            axis=-1,
        )[:, 0]
    initial_indices = current
    successful = result.successful.reshape((case_count,))
    joint_log_probability = jnp.where(
        successful,
        jnp.max(scores, axis=-1),
        -jnp.inf,
    )
    return FiniteStateViterbiResult(
        state_indices=state_indices.reshape(case_shape + (num_steps,)),
        states=prior.states[state_indices].reshape(
            case_shape + (num_steps,) + prior.state_shape
        ),
        initial_state_indices=initial_indices.reshape(case_shape),
        initial_states=prior.states[initial_indices].reshape(
            case_shape + prior.state_shape
        ),
        joint_log_probability=joint_log_probability.reshape(case_shape),
        step_valid=result.step_valid,
        valid=result.valid,
        status=result.status,
        problem=problem,
        input_id=result.input_id,
        method_id="finite-state-viterbi",
    )


def finite_state_expected_transition_counts(
    result: FiniteStateSmootherResult,
    /,
) -> FiniteStateTransitionCountResult:
    """Aggregate exact posterior endpoint-transition probabilities by case."""
    if not isinstance(result, FiniteStateSmootherResult):
        raise TypeError("result must be a FiniteStateSmootherResult.")
    prior = result.filter_result.problem.model.prior
    if not isinstance(prior, CategoricalStatePrior):
        raise TypeError("Finite-state transition counts require CategoricalStatePrior.")
    case_shape = result.filter_result.case_shape
    num_states = int(prior.states.shape[0])
    case_count = prod(case_shape) if case_shape else 1
    probabilities = result.transition_probabilities.reshape(
        (case_count, -1, num_states, num_states)
    )
    per_case = jnp.sum(probabilities, axis=1)
    return FiniteStateTransitionCountResult(
        per_case_counts=per_case.reshape(case_shape + (num_states, num_states)),
        total_counts=jnp.sum(per_case, axis=0),
        transition_probabilities=result.transition_probabilities,
        step_valid=result.step_valid,
        valid=result.valid,
        status=result.status,
        smoother_result=result,
        input_id=result.input_id,
        method_id="finite-state-expected-transition-counts",
    )


def finite_state_expected_sufficient_statistics(
    result: FiniteStateSmootherResult,
    statistic: Callable[[Array, Array, Array, Array, Any], PyTree[Any]],
    /,
) -> FiniteStateSufficientStatisticsResult:
    """Evaluate a transition statistic under the exact path posterior.

    ``statistic(previous_state, state, t0, t1, context)`` receives the canonical
    :class:`StateSpaceStepContext` as its final positional argument. It may
    return any PyTree of arrays with a structure and leaf shapes independent of
    the case, interval, and state pair.
    """
    if not isinstance(result, FiniteStateSmootherResult):
        raise TypeError("result must be a FiniteStateSmootherResult.")
    if not callable(statistic):
        raise TypeError("statistic must be callable.")
    filter_result = result.filter_result
    problem = filter_result.problem
    prior = problem.model.prior
    if not isinstance(prior, CategoricalStatePrior):
        raise TypeError(
            "Finite-state sufficient statistics require CategoricalStatePrior."
        )
    case_shape = filter_result.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = problem.observations.num_steps
    num_states = int(prior.states.shape[0])
    weights = result.transition_probabilities.reshape(
        (case_count, num_steps, num_states, num_states)
    )
    times = problem.initial_time.reshape((case_count,))
    targets = problem.observations.times.reshape((case_count, num_steps))
    active = problem.observations.step_valid.reshape((case_count, num_steps))
    case_indices = jnp.arange(case_count, dtype=jnp.int32)
    expected_steps: list[PyTree[Any]] = []

    for index in range(num_steps):
        end = jnp.where(active[:, index], targets[:, index], times)

        def one_case_statistics(
            case_index: Array,
            start_time: Array,
            end_time: Array,
        ) -> PyTree[Any]:
            context = problem.step_context(case_index, index)

            def one_previous(previous_state: Array) -> PyTree[Any]:
                return jax.vmap(
                    lambda state: jax.tree_util.tree_map(
                        jnp.asarray,
                        statistic(
                            previous_state,
                            state,
                            start_time,
                            end_time,
                            context,
                        ),
                    )
                )(prior.states)

            return jax.vmap(one_previous)(prior.states)

        evaluated = jax.vmap(one_case_statistics)(case_indices, times, end)
        step_weights = weights[:, index]

        def weighted_sum(values: Array) -> Array:
            expanded_weights = step_weights.reshape(
                step_weights.shape + (1,) * (values.ndim - 3)
            )
            masked_values = jnp.where(
                expanded_weights != 0, values, jnp.zeros_like(values)
            )
            return jnp.sum(expanded_weights * masked_values, axis=(1, 2))

        expected = jax.tree_util.tree_map(weighted_sum, evaluated)
        expected_steps.append(expected)
        times = end

    flat_per_step = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values, axis=1),
        *expected_steps,
    )
    per_step = jax.tree_util.tree_map(
        lambda values: values.reshape(case_shape + (num_steps,) + values.shape[2:]),
        flat_per_step,
    )
    per_case = jax.tree_util.tree_map(
        lambda values: jnp.sum(values, axis=1).reshape(case_shape + values.shape[2:]),
        flat_per_step,
    )
    total = jax.tree_util.tree_map(
        lambda values: jnp.sum(values, axis=(0, 1)),
        flat_per_step,
    )
    return FiniteStateSufficientStatisticsResult(
        per_step_statistics=per_step,
        per_case_statistics=per_case,
        total_statistics=total,
        step_valid=result.step_valid,
        valid=result.valid,
        status=result.status,
        smoother_result=result,
        input_id=result.input_id,
        method_id="finite-state-expected-sufficient-statistics",
    )


def exact_state_space_log_likelihood(
    problem: StateSpaceProblem,
    /,
    *,
    method: ExactStateSpaceMethod = "auto",
    covariance_regularization: float = 0.0,
    temporal_method: KalmanExecutionMethod = "auto",
) -> ExactStateSpaceLikelihood:
    """Evaluate an exact Kalman or finite-state marginal likelihood."""
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    if temporal_method not in ("sequential", "parallel", "auto"):
        raise ValueError("temporal_method must be 'sequential', 'parallel', or 'auto'.")
    resolved = _resolved_method(problem, method)
    if resolved == "kalman":
        backend = kalman_filter(
            problem,
            covariance_regularization=covariance_regularization,
            raise_on_failure=False,
            method=temporal_method,
        )
        per_case = jnp.where(
            backend.successful,
            backend.final_state.log_likelihood,
            -jnp.inf,
        )
        resolved_temporal = backend.execution_method
    else:
        if float(covariance_regularization) != 0.0:
            raise ValueError(
                "covariance_regularization applies only to the Kalman backend."
            )
        if temporal_method == "parallel":
            raise ValueError(
                "temporal_method='parallel' applies only to the Kalman backend."
            )
        backend = _finite_state_filter(problem)
        per_case = jnp.where(
            backend.successful,
            backend.cumulative_log_likelihood[..., -1],
            -jnp.inf,
        )
        resolved_temporal = "sequential"
    return ExactStateSpaceLikelihood(
        per_case_log_likelihood=per_case,
        total_log_likelihood=jnp.sum(per_case).reshape(()),
        incremental_log_likelihood=backend.incremental_log_likelihood,
        cumulative_log_likelihood=backend.cumulative_log_likelihood,
        step_valid=backend.step_valid,
        valid=backend.valid,
        status=backend.status,
        backend=backend,
        problem=problem,
        method=resolved,
        temporal_method=resolved_temporal,
        input_id=(
            None if problem.input_signal is None else problem.input_signal.input_id
        ),
    )


def state_space_identifiability(
    likelihood: StateSpaceMarginalLikelihood,
    parameters: PyTree[Any],
    /,
    *,
    relative_tolerance: float = 1e-6,
) -> StateSpaceIdentifiabilityReport:
    """Differentiate the exact likelihood and diagnose observed-information rank."""
    if not isinstance(likelihood, StateSpaceMarginalLikelihood):
        raise TypeError("likelihood must be a StateSpaceMarginalLikelihood.")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(parameters)
    if not leaves_with_paths or any(
        not eqx.is_inexact_array(leaf) for _, leaf in leaves_with_paths
    ):
        raise TypeError("parameters must be a non-empty PyTree of inexact arrays.")
    flat, unravel = ravel_pytree(parameters)

    def negative_log_likelihood(vector: Array) -> Array:
        return -likelihood.log_prob(unravel(vector))

    score = -jax.grad(negative_log_likelihood)(flat)
    information = jax.hessian(negative_log_likelihood)(flat)
    information = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(information)
    scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
    threshold = tolerance * scale
    identifiable = eigenvalues > threshold
    host_eigenvalues = np.asarray(jax.device_get(eigenvalues))
    host_identifiable = np.asarray(jax.device_get(identifiable))
    finite = bool(
        np.all(np.isfinite(np.asarray(jax.device_get(score))))
        and np.all(np.isfinite(np.asarray(jax.device_get(information))))
    )
    rank = int(np.sum(host_identifiable)) if finite else 0
    positive = host_eigenvalues[host_identifiable]
    condition = (
        float(np.max(positive) / np.min(positive))
        if finite and positive.size > 0
        else np.inf
    )
    paths = tuple(jax.tree_util.keystr(path) for path, _ in leaves_with_paths)
    shapes = tuple(
        tuple(int(size) for size in leaf.shape) for _, leaf in leaves_with_paths
    )
    return StateSpaceIdentifiabilityReport(
        score=score,
        observed_information=information,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        identifiable=identifiable,
        weak_directions=eigenvectors[:, ~identifiable],
        condition_number=jnp.asarray(condition, dtype=flat.dtype),
        score_norm=jnp.linalg.norm(score),
        finite=finite,
        dimension=int(flat.size),
        numerical_rank=rank,
        parameter_paths=paths,
        parameter_shapes=shapes,
        relative_tolerance=tolerance,
    )


__all__ = [
    "EXACT_STATE_SPACE_DEGENERATE_LIKELIHOOD",
    "EXACT_STATE_SPACE_NONFINITE",
    "EXACT_STATE_SPACE_STATE_MISMATCH",
    "EXACT_STATE_SPACE_SUCCESS",
    "EXACT_STATE_SPACE_TRANSITION_FAILURE",
    "ExactStateSpaceLikelihood",
    "ExactStateSpaceMethod",
    "FiniteStateFilterResult",
    "FiniteStateSmootherResult",
    "FiniteStateSufficientStatisticsResult",
    "FiniteStateTransitionCountResult",
    "FiniteStateViterbiResult",
    "StateSpaceIdentifiabilityReport",
    "StateSpaceMarginalLikelihood",
    "exact_state_space_log_likelihood",
    "finite_state_backward_smoother",
    "finite_state_expected_sufficient_statistics",
    "finite_state_expected_transition_counts",
    "finite_state_viterbi",
    "state_space_identifiability",
]
