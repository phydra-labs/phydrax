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

from .._strict import StrictModule
from ..stochastic._solver_transition import FiniteStateTransitionKernel
from ..stochastic._state_space import (
    CategoricalStatePrior,
    GaussianStatePrior,
    LinearGaussianObservationModel,
    LinearGaussianTransitionKernel,
    StateSpaceProblem,
)
from ._kalman import kalman_filter
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
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    step_valid: Array
    valid: Array
    status: Array
    final_probabilities: Array
    problem: StateSpaceProblem

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


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

    def __init__(
        self,
        problem: Callable[[PyTree[Any]], StateSpaceProblem],
        /,
        *,
        method: ExactStateSpaceMethod = "auto",
        covariance_regularization: float = 0.0,
        label: str = "state_space",
    ):
        if not callable(problem):
            raise TypeError("problem must be callable.")
        _validate_method(method)
        regularization = float(covariance_regularization)
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("covariance_regularization must be finite and nonnegative.")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string.")
        self.problem_fn = problem
        self.method = method
        self.covariance_regularization = regularization
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

    def state_log_likelihood(value: Array, mask: Array, time: Array) -> Array:
        return jax.vmap(lambda state: observation.log_prob(value, state, time, mask))(
            prior.states
        )

    for index in range(num_steps):
        active = flat_active[:, index]
        target_time = flat_times[:, index]
        safe_time = jnp.where(active, target_time, times)
        durations = safe_time - times
        matrices = jax.vmap(transition_matrix)(durations)
        matrix_finite = jnp.all(jnp.isfinite(matrices), axis=(-1, -2))
        matrix_nonnegative = jnp.all(matrices >= -1e-10, axis=(-1, -2))
        matrices = jnp.maximum(matrices, 0.0)
        row_sums = jnp.sum(matrices, axis=-1, keepdims=True)
        matrix_normalized = jnp.where(row_sums > 0.0, matrices / row_sums, 0.0)
        transition_valid = (
            matrix_finite
            & matrix_nonnegative
            & jnp.all(row_sums[..., 0] > 0.0, axis=-1)
            & (durations >= 0.0)
        )
        predicted = jnp.einsum("ci,cij->cj", probabilities, matrix_normalized)

        log_likelihoods = jax.vmap(state_log_likelihood)(
            flat_values[:, index], flat_masks[:, index], safe_time
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
        incremental_log_likelihood=restore_steps(increment_history),
        cumulative_log_likelihood=restore_steps(cumulative_history),
        step_valid=sequence.step_valid,
        valid=restore_steps(valid_history),
        status=restore_steps(status_history),
        final_probabilities=probabilities.reshape(case_shape + (num_states,)),
        problem=problem,
    )


def exact_state_space_log_likelihood(
    problem: StateSpaceProblem,
    /,
    *,
    method: ExactStateSpaceMethod = "auto",
    covariance_regularization: float = 0.0,
) -> ExactStateSpaceLikelihood:
    """Evaluate an exact Kalman or finite-state marginal likelihood."""
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    resolved = _resolved_method(problem, method)
    if resolved == "kalman":
        backend = kalman_filter(
            problem,
            covariance_regularization=covariance_regularization,
            raise_on_failure=False,
        )
        per_case = jnp.where(
            backend.successful,
            backend.final_state.log_likelihood,
            -jnp.inf,
        )
    else:
        if float(covariance_regularization) != 0.0:
            raise ValueError(
                "covariance_regularization applies only to the Kalman backend."
            )
        backend = _finite_state_filter(problem)
        per_case = jnp.where(
            backend.successful,
            backend.cumulative_log_likelihood[..., -1],
            -jnp.inf,
        )
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
    "StateSpaceIdentifiabilityReport",
    "StateSpaceMarginalLikelihood",
    "exact_state_space_log_likelihood",
    "state_space_identifiability",
]
