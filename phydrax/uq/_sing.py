#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, prod
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg._gaussian_chain import (
    gaussian_markov_information_from_moments,
    gaussian_markov_moments,
    gaussian_markov_moments_from_marginals,
    GaussianMarkovExecutionMethod,
    GaussianMarkovInformation,
    GaussianMarkovMoments,
    sample_gaussian_markov,
)
from ..stochastic._euler_maruyama import EulerMaruyamaTransitionKernel
from ..stochastic._state_space import (
    GaussianStatePrior,
    state_space_key,
    StateSpaceProblem,
)
from ._gaussian_factor import GaussianFactor
from ._nonlinear_gaussian import (
    gaussian_expectation,
    GaussianExpectationMethod,
)


SINGExpectationMethod: TypeAlias = GaussianExpectationMethod
SINGExecutionMethod: TypeAlias = GaussianMarkovExecutionMethod
SINGStatus: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6]

SING_SUCCESS: SINGStatus = 0
SING_MAXIMUM_ITERATIONS: SINGStatus = 1
SING_INITIALIZATION_FAILURE: SINGStatus = 2
SING_INFORMATION_NOT_POSITIVE_DEFINITE: SINGStatus = 3
SING_NONFINITE: SINGStatus = 4
SING_LINE_SEARCH_FAILURE: SINGStatus = 5
SING_TRANSITION_COVARIANCE_FAILURE: SINGStatus = 6


def sing_status_name(value: int, /) -> str:
    """Return the stable name of one SING status code."""
    names = (
        "success",
        "maximum_iterations",
        "initialization_failure",
        "information_not_positive_definite",
        "nonfinite",
        "line_search_failure",
        "transition_covariance_failure",
    )
    code = int(value)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown SING status code {code}.")
    return names[code]


def _identifier(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _positive_int(value: int, /, *, owner: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{owner} must be an integer.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{owner} must be positive.")
    return resolved


def _nonnegative_float(value: float, /, *, owner: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{owner} must be finite and nonnegative.")
    return resolved


def _expectation_configuration(
    method: SINGExpectationMethod,
    key: Array | None,
    num_samples: int,
    order: int,
    max_dimension: int,
    max_points: int,
    alpha: float,
    beta: float,
    kappa: float,
    /,
) -> tuple[Array, int, int, int, int, float, float, float]:
    if method not in ("cubature", "unscented", "gauss-hermite", "monte-carlo"):
        raise ValueError(
            "expectation_method must be 'cubature', 'unscented', "
            "'gauss-hermite', or 'monte-carlo'."
        )
    samples = _positive_int(num_samples, owner="num_samples")
    order_ = _positive_int(order, owner="order")
    max_dimension_ = _positive_int(max_dimension, owner="max_dimension")
    max_points_ = _positive_int(max_points, owner="max_points")
    alpha_ = float(alpha)
    beta_ = float(beta)
    kappa_ = float(kappa)
    if not all(isfinite(value) for value in (alpha_, beta_, kappa_)):
        raise ValueError("alpha, beta, and kappa must be finite.")
    if alpha_ <= 0.0:
        raise ValueError("alpha must be positive.")
    if method == "monte-carlo" and key is None:
        raise ValueError("key is required for Monte Carlo SING expectations.")
    resolved_key = jr.key(0) if key is None else key
    return (
        resolved_key,
        samples,
        order_,
        max_dimension_,
        max_points_,
        alpha_,
        beta_,
        kappa_,
    )


def _validate_problem(problem: StateSpaceProblem, /):
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    from ._sing_transition import _ProjectedEulerTransition

    transition = problem.model.transition
    if not isinstance(
        transition, (EulerMaruyamaTransitionKernel, _ProjectedEulerTransition)
    ):
        raise TypeError(
            "SING requires an EulerMaruyamaTransitionKernel or the canonical "
            "fixed affine-Hausdorff projected Euler transition."
        )
    prior = problem.model.prior
    if not isinstance(prior, GaussianStatePrior) or not prior.has_log_density:
        raise TypeError(
            "Automatic SING initialization requires a full-rank GaussianStatePrior."
        )
    return transition


class SINGGrid(StrictModule):
    """Case-aligned latent grid retaining the original observation-node map."""

    times: Array
    node_valid: Array
    observation_node_indices: Array
    transition_step_indices: Array
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    @property
    def num_nodes(self) -> int:
        return int(self.times.shape[-1])

    @property
    def num_observations(self) -> int:
        return int(self.observation_node_indices.shape[-1])


class SINGState(StrictModule):
    """Restartable Gaussian Markov state for SDE natural-gradient inference."""

    information: GaussianMarkovInformation
    grid: SINGGrid
    expectation_key: Array
    iteration: Array
    valid: Array
    status: Array
    expectation_method: str = eqx.field(static=True)
    execution_method: str = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    max_dimension: int = eqx.field(static=True)
    max_points: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    kappa: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


class SINGELBOResult(StrictModule):
    """Decomposed fixed-posterior evidence lower bound."""

    per_case_elbo: Array
    total_elbo: Array
    expected_initial_log_density: Array
    expected_transition_log_density: Array
    expected_observation_log_density: Array
    entropy: Array
    valid: Array
    status: Array
    expectation_method: str = eqx.field(static=True)
    execution_method: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    information_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


class SINGStepResult(StrictModule):
    """One accepted natural-gradient step with per-case globalization evidence."""

    state: SINGState
    moments: GaussianMarkovMoments
    elbo: SINGELBOResult
    accepted_step_size: Array
    natural_residual: Array
    accepted: Array
    valid: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.valid


class SINGResult(StrictModule):
    """Final SING posterior with optimization history and model provenance."""

    state: SINGState
    moments: GaussianMarkovMoments
    elbo: SINGELBOResult
    elbo_history: Array
    step_size_history: Array
    natural_residual_history: Array
    accepted_history: Array
    converged: Array
    valid: Array
    status: Array
    max_iterations: int = eqx.field(static=True)
    expectation_method: str = eqx.field(static=True)
    execution_method: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid

    @property
    def means(self) -> Array:
        return self.moments.means.reshape(
            self.case_shape + (self.moments.num_nodes,) + self.state_shape
        )

    @property
    def covariances(self) -> Array:
        return self.moments.covariances

    @property
    def transition_cross_covariances(self) -> Array:
        return self.moments.transition_cross_covariances

    @property
    def observation_means(self) -> Array:
        case_count = prod(self.case_shape) if self.case_shape else 1
        state_size = self.moments.state_size
        means = self.moments.means.reshape(
            (case_count, self.moments.num_nodes, state_size)
        )
        indices = self.state.grid.observation_node_indices.reshape((case_count, -1))
        gathered = jax.vmap(lambda values, locations: values[locations])(means, indices)
        return gathered.reshape(self.case_shape + (indices.shape[-1],) + self.state_shape)

    @property
    def observation_covariances(self) -> Array:
        case_count = prod(self.case_shape) if self.case_shape else 1
        state_size = self.moments.state_size
        covariances = self.moments.covariances.reshape(
            (
                case_count,
                self.moments.num_nodes,
                state_size,
                state_size,
            )
        )
        indices = self.state.grid.observation_node_indices.reshape((case_count, -1))
        gathered = jax.vmap(lambda values, locations: values[locations])(
            covariances, indices
        )
        return gathered.reshape(
            self.case_shape + (indices.shape[-1], state_size, state_size)
        )


def _build_sing_grid(problem: StateSpaceProblem, /) -> SINGGrid:
    observations = problem.observations
    case_shape = observations.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = observations.num_steps
    initial = problem.initial_time.reshape((case_count,))
    times = observations.times.reshape((case_count, num_steps))
    step_valid = observations.step_valid.reshape((case_count, num_steps))

    def case_grid(initial_time, observation_times, valid_steps):
        previous = jnp.concatenate((initial_time[None], observation_times[:-1]), axis=0)
        new_node = valid_steps & (observation_times > previous)
        observation_indices = jnp.cumsum(new_node, dtype=jnp.int32)
        node_count = 1 + jnp.sum(new_node, dtype=jnp.int32)
        grid = jnp.zeros((num_steps + 1,), dtype=observation_times.dtype)
        grid = grid.at[0].set(initial_time)
        grid = grid.at[observation_indices].add(
            jnp.where(new_node, observation_times, 0.0)
        )
        node_valid = jnp.arange(num_steps + 1, dtype=jnp.int32) < node_count
        last_time = grid[node_count - 1]
        grid = jnp.where(node_valid, grid, last_time)
        edge_indices = jnp.maximum(observation_indices - 1, 0)
        encoded_steps = jnp.zeros((num_steps,), dtype=jnp.int32)
        encoded_steps = encoded_steps.at[edge_indices].add(
            jnp.where(
                new_node,
                jnp.arange(num_steps, dtype=jnp.int32) + 1,
                0,
            )
        )
        transition_steps = jnp.maximum(encoded_steps - 1, 0)
        return grid, node_valid, observation_indices, transition_steps

    grid_times, node_valid, observation_indices, transition_steps = jax.vmap(case_grid)(
        initial, times, step_valid
    )
    return SINGGrid(
        grid_times.reshape(case_shape + (num_steps + 1,)),
        node_valid.reshape(case_shape + (num_steps + 1,)),
        observation_indices.reshape(case_shape + (num_steps,)),
        transition_steps.reshape(case_shape + (num_steps,)),
        case_axes=observations.case_axes,
        case_shape=case_shape,
        case_ids=observations.case_ids,
        grid_id=f"sing-grid:{problem.problem_id}:{observations.sequence_id}",
    )


def _expectation(
    function,
    mean: Array,
    factor: GaussianFactor,
    key: Array,
    /,
    *,
    method: SINGExpectationMethod,
    num_samples: int,
    order: int,
    max_dimension: int,
    max_points: int,
    alpha: float,
    beta: float,
    kappa: float,
):
    return gaussian_expectation(
        function,
        mean,
        factor,
        method=method,
        key=key,
        num_samples=num_samples,
        order=order,
        max_dimension=max_dimension,
        max_points=max_points,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )


def initialize_sing(
    problem: StateSpaceProblem,
    /,
    *,
    key: Array | None = None,
    expectation_method: SINGExpectationMethod = "cubature",
    method: SINGExecutionMethod = "auto",
    num_samples: int = 64,
    order: int = 3,
    max_dimension: int = 5,
    max_points: int = 100_000,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    rank_tolerance: float = 0.0,
) -> SINGState:
    """Initialize SING from the Euler-discretized Gaussian SDE prior."""
    transition = _validate_problem(problem)
    (
        expectation_key,
        num_samples_,
        order_,
        max_dimension_,
        max_points_,
        alpha_,
        beta_,
        kappa_,
    ) = _expectation_configuration(
        expectation_method,
        key,
        num_samples,
        order,
        max_dimension,
        max_points,
        alpha,
        beta,
        kappa,
    )
    tolerance = _nonnegative_float(rank_tolerance, owner="rank_tolerance")
    grid = _build_sing_grid(problem)
    case_shape = problem.observations.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_nodes = grid.num_nodes
    state_shape = problem.model.state_shape
    state_size = prod(state_shape) if state_shape else 1
    prior = problem.model.prior
    prior_means = prior.mean.reshape((case_count, state_size))
    prior_covariances = prior.covariance.reshape((case_count, state_size, state_size))
    grid_times = grid.times.reshape((case_count, num_nodes))
    node_valid = grid.node_valid.reshape((case_count, num_nodes))
    transition_steps = grid.transition_step_indices.reshape((case_count, num_nodes - 1))
    case_means: list[Array] = []
    case_covariances: list[Array] = []
    case_cross: list[Array] = []
    case_validity: list[Array] = []

    for case_index, case_id in enumerate(grid.case_ids):
        initial_mean = prior_means[case_index]
        initial_covariance = prior_covariances[case_index]

        def rollout_step(carry, inputs):
            source_mean, source_covariance = carry
            edge_index, active, start, end, step_index = inputs
            context = problem.step_context(case_index, step_index)
            edge_key = state_space_key(
                expectation_key,
                "sing-initialization",
                case_id,
                edge_index,
            )

            def active_rollout(operands):
                mean, covariance = operands
                source_factor = GaussianFactor(
                    jnp.linalg.cholesky(covariance),
                    rank_tolerance=tolerance,
                    factor_id="sing-initialization-source",
                    resolved_method="dense-cholesky",
                )
                interval = end - start

                def deterministic_euler(value):
                    physical_state = value.reshape(state_shape)
                    drift = transition.drift(start, physical_state, context).reshape(
                        (state_size,)
                    )
                    target = value + interval * drift
                    return {
                        "mean": target,
                        "second": target[:, None] * target[None, :],
                        "source_target": value[:, None] * target[None, :],
                    }

                transformed = _expectation(
                    deterministic_euler,
                    mean,
                    source_factor,
                    edge_key,
                    method=expectation_method,
                    num_samples=num_samples_,
                    order=order_,
                    max_dimension=max_dimension_,
                    max_points=max_points_,
                    alpha=alpha_,
                    beta=beta_,
                    kappa=kappa_,
                )
                target_mean = transformed.value["mean"]
                target_second = transformed.value["second"]
                source_target_second = transformed.value["source_target"]
                dispersion = transition.dispersion(
                    start,
                    jax.lax.stop_gradient(mean).reshape(state_shape),
                    context,
                )
                process_covariance = interval * (dispersion @ dispersion.T)
                target_covariance = (
                    target_second
                    - target_mean[:, None] * target_mean[None, :]
                    + process_covariance
                )
                target_covariance = 0.5 * (target_covariance + target_covariance.T)
                source_target_covariance = (
                    source_target_second - mean[:, None] * target_mean[None, :]
                )
                valid = (
                    transformed.valid
                    & context.input_valid
                    & jnp.all(jnp.isfinite(target_mean))
                    & jnp.all(jnp.isfinite(target_covariance))
                    & jnp.all(jnp.isfinite(source_target_covariance))
                )
                return (
                    target_mean,
                    target_covariance,
                    source_target_covariance,
                    valid,
                )

            def inactive_rollout(operands):
                mean, covariance = operands
                return mean, covariance, covariance, jnp.asarray(True)

            target_mean, target_covariance, cross_covariance, valid = jax.lax.cond(
                active,
                active_rollout,
                inactive_rollout,
                (source_mean, source_covariance),
            )
            return (target_mean, target_covariance), (
                target_mean,
                target_covariance,
                cross_covariance,
                valid,
            )

        _, rollout = jax.lax.scan(
            rollout_step,
            (initial_mean, initial_covariance),
            (
                jnp.arange(num_nodes - 1, dtype=jnp.int32),
                node_valid[case_index, 1:],
                grid_times[case_index, :-1],
                grid_times[case_index, 1:],
                transition_steps[case_index],
            ),
        )
        rollout_means, rollout_covariances, cross_covariances, valid_edges = rollout
        case_means.append(jnp.concatenate((initial_mean[None, :], rollout_means), axis=0))
        case_covariances.append(
            jnp.concatenate(
                (initial_covariance[None, :, :], rollout_covariances),
                axis=0,
            )
        )
        case_cross.append(cross_covariances)
        case_validity.append(jnp.all(valid_edges))

    means = jnp.stack(case_means).reshape(case_shape + (num_nodes, state_size))
    covariances = jnp.stack(case_covariances).reshape(
        case_shape + (num_nodes, state_size, state_size)
    )
    cross_covariances = jnp.stack(case_cross).reshape(
        case_shape + (num_nodes - 1, state_size, state_size)
    )
    initialization_valid = jnp.stack(case_validity).reshape(case_shape)
    provided_moments = gaussian_markov_moments_from_marginals(
        means,
        covariances,
        cross_covariances,
        node_valid=grid.node_valid,
        moments_id="sing-initial-prior-moments",
        information_id=f"sing-information:{problem.problem_id}",
        rank_tolerance=tolerance,
    )
    information = gaussian_markov_information_from_moments(provided_moments)
    recovered = gaussian_markov_moments(information, method=method)
    valid = initialization_valid & recovered.valid
    status = jnp.where(
        ~initialization_valid,
        SING_INITIALIZATION_FAILURE,
        jnp.where(
            ~recovered.valid,
            SING_INFORMATION_NOT_POSITIVE_DEFINITE,
            SING_SUCCESS,
        ),
    ).astype(jnp.int32)
    return SINGState(
        information,
        grid,
        expectation_key,
        jnp.asarray(0, dtype=jnp.int32),
        valid,
        status,
        expectation_method=expectation_method,
        execution_method=recovered.execution_method,
        num_samples=num_samples_,
        order=order_,
        max_dimension=max_dimension_,
        max_points=max_points_,
        alpha=alpha_,
        beta=beta_,
        kappa=kappa_,
        problem_id=problem.problem_id,
        model_id=problem.model.model_id,
        process_id=transition.process_id,
        sequence_id=problem.observations.sequence_id,
        state_shape=state_shape,
    )


def _validate_state(problem: StateSpaceProblem, state: SINGState, /) -> None:
    _validate_problem(problem)
    if not isinstance(state, SINGState):
        raise TypeError("state must be a SINGState.")
    if state.problem_id != problem.problem_id:
        raise ValueError("SINGState and StateSpaceProblem problem IDs do not agree.")
    if state.model_id != problem.model.model_id:
        raise ValueError("SINGState and StateSpaceProblem model IDs do not agree.")
    if state.sequence_id != problem.observations.sequence_id:
        raise ValueError("SINGState and observation sequence IDs do not agree.")
    if state.state_shape != problem.model.state_shape:
        raise ValueError("SINGState and model state shapes do not agree.")


def _covariance_moments(
    means: Array,
    second_moments: Array,
    transition_second_moments: Array,
    /,
) -> tuple[Array, Array]:
    covariances = second_moments - means[..., :, :, None] * means[..., :, None, :]
    covariances = 0.5 * (covariances + jnp.swapaxes(covariances, -1, -2))
    cross_covariances = (
        transition_second_moments - means[..., :-1, :, None] * means[..., 1:, None, :]
    )
    return covariances, cross_covariances


def _spd_data(matrix: Array, /) -> tuple[Array, Array, Array]:
    symmetric = 0.5 * (matrix + matrix.T)
    factor = jnp.linalg.cholesky(symmetric)
    diagonal = jnp.diag(factor)
    valid = (
        jnp.all(jnp.isfinite(symmetric))
        & jnp.all(jnp.isfinite(factor))
        & jnp.all(diagonal > 0.0)
    )
    safe_matrix = jnp.where(valid, symmetric, jnp.eye(matrix.shape[-1]))
    safe_factor = jnp.where(valid, factor, jnp.eye(matrix.shape[-1]))
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(safe_factor)))
    return safe_matrix, log_determinant, valid


def _expected_gaussian_log_density(
    mean: Array,
    covariance: Array,
    location: Array,
    target_covariance: Array,
    /,
) -> tuple[Array, Array]:
    safe_covariance, log_determinant, valid = _spd_data(target_covariance)
    residual = mean - location
    residual_second = covariance + residual[:, None] * residual[None, :]
    quadratic = jnp.trace(jnp.linalg.solve(safe_covariance, residual_second))
    dimension = mean.shape[-1]
    value = -0.5 * (dimension * jnp.log(2.0 * jnp.pi) + log_determinant + quadratic)
    valid = valid & jnp.isfinite(value)
    return jnp.where(valid, value, -jnp.inf), valid


def _case_entropy(
    covariances: Array,
    cross_covariances: Array,
    node_valid: Array,
    /,
) -> tuple[Array, Array]:
    dimension = covariances.shape[-1]
    constant = dimension * (1.0 + jnp.log(2.0 * jnp.pi))
    _, initial_log_determinant, initial_valid = _spd_data(covariances[0])
    initial_entropy = 0.5 * (constant + initial_log_determinant)

    def edge_entropy(edge_index):
        active = node_valid[edge_index + 1]

        def active_entropy(_):
            source_covariance = covariances[edge_index]
            target_covariance = covariances[edge_index + 1]
            cross_covariance = cross_covariances[edge_index]
            safe_source, _, source_valid = _spd_data(source_covariance)
            conditional = target_covariance - (
                cross_covariance.T @ jnp.linalg.solve(safe_source, cross_covariance)
            )
            _, log_determinant, conditional_valid = _spd_data(conditional)
            value = 0.5 * (constant + log_determinant)
            valid = source_valid & conditional_valid & jnp.isfinite(value)
            return jnp.where(valid, value, 0.0), valid

        return jax.lax.cond(
            active,
            active_entropy,
            lambda _: (
                jnp.asarray(0.0, dtype=covariances.dtype),
                jnp.asarray(True),
            ),
            operand=None,
        )

    contributions, edge_valid = jax.vmap(edge_entropy)(
        jnp.arange(covariances.shape[0] - 1, dtype=jnp.int32)
    )
    entropy = initial_entropy + jnp.sum(contributions)
    valid = initial_valid & jnp.all(edge_valid) & jnp.isfinite(entropy)
    return entropy, valid


def _case_expected_log_joint(
    problem: StateSpaceProblem,
    state: SINGState,
    case_index: int,
    case_id: str,
    means: Array,
    second_moments: Array,
    transition_second_moments: Array,
    /,
):
    transition = problem.model.transition
    prior = problem.model.prior
    observations = problem.observations
    state_shape = state.state_shape
    state_size = prod(state_shape) if state_shape else 1
    num_nodes = state.grid.num_nodes
    num_observations = state.grid.num_observations
    covariances, cross_covariances = _covariance_moments(
        means, second_moments, transition_second_moments
    )
    grid_times = state.grid.times.reshape((-1, num_nodes))[case_index]
    node_valid = state.grid.node_valid.reshape((-1, num_nodes))[case_index]
    transition_steps = state.grid.transition_step_indices.reshape((-1, num_nodes - 1))[
        case_index
    ]
    observation_nodes = state.grid.observation_node_indices.reshape(
        (-1, num_observations)
    )[case_index]
    observation_times = observations.times.reshape((-1, num_observations))[case_index]
    observation_values = observations.values.reshape(
        (-1, num_observations) + observations.observation_shape
    )[case_index]
    observation_mask = observations.observation_mask.reshape(
        (-1, num_observations) + observations.observation_shape
    )[case_index]
    observation_valid = observations.step_valid.reshape((-1, num_observations))[
        case_index
    ]
    prior_mean = prior.mean.reshape((-1, state_size))[case_index]
    prior_covariance = prior.covariance.reshape((-1, state_size, state_size))[case_index]
    initial_term, initial_valid = _expected_gaussian_log_density(
        means[0],
        covariances[0],
        prior_mean,
        prior_covariance,
    )
    status = jnp.where(initial_valid, SING_SUCCESS, SING_INITIALIZATION_FAILURE).astype(
        jnp.int32
    )
    multiplicative = any(term.structure != "additive" for term in transition.wiener_terms)

    def transition_factor(edge_index):
        active = node_valid[edge_index + 1]
        start = grid_times[edge_index]
        end = grid_times[edge_index + 1]
        step_index = transition_steps[edge_index]
        context = problem.step_context(case_index, step_index)
        factor_key = state_space_key(
            state.expectation_key,
            "sing-transition-objective",
            case_id,
            edge_index,
        )

        def active_transition(_):
            source_mean = means[edge_index]
            target_mean = means[edge_index + 1]
            source_covariance = covariances[edge_index]
            target_covariance = covariances[edge_index + 1]
            cross_covariance = cross_covariances[edge_index]
            if multiplicative:
                joint_mean = jnp.concatenate((source_mean, target_mean))
                joint_covariance = jnp.concatenate(
                    (
                        jnp.concatenate((source_covariance, cross_covariance), axis=-1),
                        jnp.concatenate((cross_covariance.T, target_covariance), axis=-1),
                    ),
                    axis=-2,
                )
                safe_joint, _, joint_valid = _spd_data(joint_covariance)
                joint_factor = GaussianFactor(
                    jnp.linalg.cholesky(safe_joint),
                    rank_tolerance=state.information.rank_tolerance,
                    factor_id="sing-transition-joint",
                    resolved_method="dense-cholesky",
                )

                def log_factor(value):
                    source = value[:state_size].reshape(state_shape)
                    target = value[state_size:].reshape(state_shape)
                    return transition.log_prob(target, source, start, end, context)

                expected_log_factor = _expectation(
                    log_factor,
                    joint_mean,
                    joint_factor,
                    factor_key,
                    method=state.expectation_method,
                    num_samples=state.num_samples,
                    order=state.order,
                    max_dimension=state.max_dimension,
                    max_points=state.max_points,
                    alpha=state.alpha,
                    beta=state.beta,
                    kappa=state.kappa,
                )
                value = expected_log_factor.value
                valid = (
                    joint_valid
                    & joint_factor.valid
                    & expected_log_factor.valid
                    & context.input_valid
                    & jnp.isfinite(value)
                )
                failure = jnp.where(
                    joint_valid,
                    SING_NONFINITE,
                    SING_TRANSITION_COVARIANCE_FAILURE,
                ).astype(jnp.int32)
                return jnp.where(valid, value, 0.0), valid, failure

            safe_source, _, source_valid = _spd_data(source_covariance)
            source_factor = GaussianFactor(
                jnp.linalg.cholesky(safe_source),
                rank_tolerance=state.information.rank_tolerance,
                factor_id="sing-transition-source",
                resolved_method="dense-cholesky",
            )
            interval = end - start

            def drift(value):
                return transition.drift(
                    start, value.reshape(state_shape), context
                ).reshape((state_size,))

            def drift_statistics(value):
                evaluated = drift(value)
                return {
                    "first": evaluated,
                    "second": evaluated[:, None] * evaluated[None, :],
                }

            transformed = _expectation(
                drift_statistics,
                source_mean,
                source_factor,
                factor_key,
                method=state.expectation_method,
                num_samples=state.num_samples,
                order=state.order,
                max_dimension=state.max_dimension,
                max_points=state.max_points,
                alpha=state.alpha,
                beta=state.beta,
                kappa=state.kappa,
            )

            def expected_drift(shifted_mean):
                return _expectation(
                    drift,
                    shifted_mean,
                    source_factor,
                    factor_key,
                    method=state.expectation_method,
                    num_samples=state.num_samples,
                    order=state.order,
                    max_dimension=state.max_dimension,
                    max_points=state.max_points,
                    alpha=state.alpha,
                    beta=state.beta,
                    kappa=state.kappa,
                ).value

            drift_jacobian = jax.jacfwd(expected_drift)(source_mean)
            expected_drift_value = transformed.value["first"]
            expected_drift_outer = transformed.value["second"]
            dispersion = transition.dispersion(
                start,
                jax.lax.stop_gradient(source_mean).reshape(state_shape),
                context,
            )
            process_covariance = interval * (dispersion @ dispersion.T)
            safe_process, log_determinant, process_valid = _spd_data(process_covariance)
            increment_mean = target_mean - source_mean
            increment_second = (
                target_covariance
                + source_covariance
                - cross_covariance
                - cross_covariance.T
                + increment_mean[:, None] * increment_mean[None, :]
            )
            increment_drift = (
                increment_mean[:, None] * expected_drift_value[None, :]
                + (cross_covariance.T - source_covariance) @ drift_jacobian.T
            )
            residual_second = (
                increment_second
                - interval * (increment_drift + increment_drift.T)
                + interval**2 * expected_drift_outer
            )
            residual_second = 0.5 * (residual_second + residual_second.T)
            quadratic = jnp.trace(jnp.linalg.solve(safe_process, residual_second))
            value = -0.5 * (
                state_size * jnp.log(2.0 * jnp.pi) + log_determinant + quadratic
            )
            finite = (
                jnp.isfinite(value)
                & jnp.all(jnp.isfinite(drift_jacobian))
                & jnp.all(jnp.isfinite(expected_drift_value))
                & jnp.all(jnp.isfinite(expected_drift_outer))
            )
            valid = (
                source_valid
                & source_factor.valid
                & transformed.valid
                & process_valid
                & context.input_valid
                & finite
            )
            failure = jnp.where(
                ~process_valid,
                SING_TRANSITION_COVARIANCE_FAILURE,
                SING_NONFINITE,
            ).astype(jnp.int32)
            return jnp.where(valid, value, 0.0), valid, failure

        return jax.lax.cond(
            active,
            active_transition,
            lambda _: (
                jnp.asarray(0.0, dtype=means.dtype),
                jnp.asarray(True),
                jnp.asarray(SING_SUCCESS, dtype=jnp.int32),
            ),
            operand=None,
        )

    transition_array, transition_valid, transition_status = jax.vmap(transition_factor)(
        jnp.arange(num_nodes - 1, dtype=jnp.int32)
    )

    def observation_factor(observation_index):
        node_index = observation_nodes[observation_index]
        active = observation_valid[observation_index] & jnp.any(
            observation_mask[observation_index]
        )
        time = observation_times[observation_index]
        context = problem.step_context(case_index, observation_index)
        factor_key = state_space_key(
            state.expectation_key,
            "sing-observation-objective",
            case_id,
            observation_index,
        )

        def active_observation(_):
            mean = means[node_index]
            covariance = covariances[node_index]
            safe_covariance, _, covariance_valid = _spd_data(covariance)
            factor = GaussianFactor(
                jnp.linalg.cholesky(safe_covariance),
                rank_tolerance=state.information.rank_tolerance,
                factor_id="sing-observation-state",
                resolved_method="dense-cholesky",
            )

            def log_likelihood(value):
                return problem.model.observation.log_prob(
                    observation_values[observation_index],
                    value.reshape(state_shape),
                    time,
                    observation_mask[observation_index],
                    context,
                )

            expectation = _expectation(
                log_likelihood,
                mean,
                factor,
                factor_key,
                method=state.expectation_method,
                num_samples=state.num_samples,
                order=state.order,
                max_dimension=state.max_dimension,
                max_points=state.max_points,
                alpha=state.alpha,
                beta=state.beta,
                kappa=state.kappa,
            )
            value = jnp.asarray(expectation.value).reshape(())
            valid = (
                covariance_valid
                & factor.valid
                & expectation.valid
                & context.input_valid
                & jnp.isfinite(value)
            )
            return jnp.where(valid, value, 0.0), valid

        return jax.lax.cond(
            active,
            active_observation,
            lambda _: (
                jnp.asarray(0.0, dtype=means.dtype),
                jnp.asarray(True),
            ),
            operand=None,
        )

    observation_array, observation_factor_valid = jax.vmap(observation_factor)(
        jnp.arange(num_observations, dtype=jnp.int32)
    )
    entropy, entropy_valid = _case_entropy(covariances, cross_covariances, node_valid)
    transition_failed = ~transition_valid
    first_transition_failure = transition_status[jnp.argmax(transition_failed)]
    status = jnp.where(
        (status == SING_SUCCESS) & jnp.any(transition_failed),
        first_transition_failure,
        status,
    )
    status = jnp.where(
        (status == SING_SUCCESS) & ~jnp.all(observation_factor_valid),
        SING_NONFINITE,
        status,
    )
    status = jnp.where(
        (status == SING_SUCCESS) & ~entropy_valid,
        SING_INFORMATION_NOT_POSITIVE_DEFINITE,
        status,
    )
    expected_joint = initial_term + jnp.sum(transition_array) + jnp.sum(observation_array)
    valid = (
        initial_valid
        & jnp.all(transition_valid)
        & jnp.all(observation_factor_valid)
        & entropy_valid
        & jnp.isfinite(expected_joint)
    )
    status = jnp.where(
        (status == SING_SUCCESS) & ~valid,
        SING_NONFINITE,
        status,
    ).astype(jnp.int32)
    expected_joint = jnp.where(valid, expected_joint, -jnp.inf)
    return expected_joint, (
        initial_term,
        transition_array,
        observation_array,
        entropy,
        valid,
        status,
    )


def _sing_statistics(
    problem: StateSpaceProblem,
    state: SINGState,
    /,
    *,
    natural_target: bool,
) -> tuple[
    GaussianMarkovMoments,
    SINGELBOResult,
    GaussianMarkovInformation | None,
]:
    _validate_state(problem, state)
    moments = gaussian_markov_moments(
        state.information,
        method=state.execution_method,
        moments_id=f"sing-moments:{problem.problem_id}",
    )
    case_shape = state.grid.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_nodes = state.grid.num_nodes
    state_size = state.information.state_size
    means = moments.means.reshape((case_count, num_nodes, state_size))
    second = moments.second_moments.reshape(
        (case_count, num_nodes, state_size, state_size)
    )
    transition_second = moments.transition_second_moments.reshape(
        (case_count, num_nodes - 1, state_size, state_size)
    )
    expected_joint_values: list[Array] = []
    initial_values: list[Array] = []
    transition_values: list[Array] = []
    observation_values: list[Array] = []
    entropy_values: list[Array] = []
    validity_values: list[Array] = []
    status_values: list[Array] = []
    mean_gradients: list[Array] = []
    second_gradients: list[Array] = []
    transition_gradients: list[Array] = []

    for case_index, case_id in enumerate(state.grid.case_ids):

        def objective(case_means, case_second, case_transition):
            return _case_expected_log_joint(
                problem,
                state,
                case_index,
                case_id,
                case_means,
                case_second,
                case_transition,
            )

        if natural_target:
            (value_and_aux, gradients) = jax.value_and_grad(
                objective,
                argnums=(0, 1, 2),
                has_aux=True,
            )(means[case_index], second[case_index], transition_second[case_index])
            expected_joint, auxiliary = value_and_aux
            mean_gradient, second_gradient, transition_gradient = gradients
            mean_gradients.append(mean_gradient)
            second_gradients.append(second_gradient)
            transition_gradients.append(transition_gradient)
        else:
            expected_joint, auxiliary = objective(
                means[case_index],
                second[case_index],
                transition_second[case_index],
            )
        (
            initial_value,
            transition_value,
            observation_value,
            entropy,
            valid,
            status,
        ) = auxiliary
        expected_joint_values.append(expected_joint)
        initial_values.append(initial_value)
        transition_values.append(transition_value)
        observation_values.append(observation_value)
        entropy_values.append(entropy)
        validity_values.append(valid)
        status_values.append(status)

    expected_joint = jnp.stack(expected_joint_values).reshape(case_shape)
    entropy = jnp.stack(entropy_values).reshape(case_shape)
    per_case_elbo = expected_joint + entropy
    factor_valid = jnp.stack(validity_values).reshape(case_shape)
    valid = state.valid & moments.valid & factor_valid & jnp.isfinite(per_case_elbo)
    status = jnp.stack(status_values).reshape(case_shape)
    status = jnp.where(
        ~state.valid,
        state.status,
        jnp.where(
            ~moments.valid,
            SING_INFORMATION_NOT_POSITIVE_DEFINITE,
            status,
        ),
    ).astype(jnp.int32)
    total_elbo = jnp.where(jnp.all(valid), jnp.sum(per_case_elbo), -jnp.inf)
    result = SINGELBOResult(
        per_case_elbo,
        total_elbo,
        jnp.stack(initial_values).reshape(case_shape),
        jnp.stack(transition_values).reshape(case_shape + (num_nodes - 1,)),
        jnp.stack(observation_values).reshape(
            case_shape + (state.grid.num_observations,)
        ),
        entropy,
        valid,
        status,
        expectation_method=state.expectation_method,
        execution_method=moments.execution_method,
        problem_id=problem.problem_id,
        information_id=state.information.information_id,
    )
    if not natural_target:
        return moments, result, None

    mean_gradient = jnp.stack(mean_gradients).reshape(
        case_shape + (num_nodes, state_size)
    )
    second_gradient = jnp.stack(second_gradients).reshape(
        case_shape + (num_nodes, state_size, state_size)
    )
    transition_gradient = jnp.stack(transition_gradients).reshape(
        case_shape + (num_nodes - 1, state_size, state_size)
    )
    node_valid = state.grid.node_valid
    edge_valid = node_valid[..., :-1] & node_valid[..., 1:]
    diagonal_precision = -(second_gradient + jnp.swapaxes(second_gradient, -1, -2))
    diagonal_precision = jnp.where(
        node_valid[..., None, None],
        diagonal_precision,
        jnp.eye(state_size, dtype=diagonal_precision.dtype),
    )
    transition_precision = jnp.where(
        edge_valid[..., None, None], -transition_gradient, 0.0
    )
    information_vector = jnp.where(node_valid[..., None], mean_gradient, 0.0)
    target = GaussianMarkovInformation(
        diagonal_precision,
        transition_precision,
        information_vector,
        node_valid=node_valid,
        information_id=f"sing-natural-target:{problem.problem_id}",
        rank_tolerance=state.information.rank_tolerance,
    )
    return moments, result, target


def sing_elbo(
    problem: StateSpaceProblem,
    state: SINGState,
    /,
) -> SINGELBOResult:
    """Evaluate the decomposed SING ELBO while holding the posterior fixed."""
    _, result, _ = _sing_statistics(problem, state, natural_target=False)
    return result


def _state_with_information(
    state: SINGState,
    information: GaussianMarkovInformation,
    iteration: Array,
    valid: Array,
    status: Array,
    /,
) -> SINGState:
    return SINGState(
        information,
        state.grid,
        state.expectation_key,
        iteration,
        valid,
        status,
        expectation_method=state.expectation_method,
        execution_method=state.execution_method,
        num_samples=state.num_samples,
        order=state.order,
        max_dimension=state.max_dimension,
        max_points=state.max_points,
        alpha=state.alpha,
        beta=state.beta,
        kappa=state.kappa,
        problem_id=state.problem_id,
        model_id=state.model_id,
        process_id=state.process_id,
        sequence_id=state.sequence_id,
        state_shape=state.state_shape,
    )


def _interpolate_information(
    current: GaussianMarkovInformation,
    target: GaussianMarkovInformation,
    step_size: Array,
    /,
) -> GaussianMarkovInformation:
    matrix_weight = step_size[..., None, None, None]
    vector_weight = step_size[..., None, None]
    diagonal = current.diagonal_precision + matrix_weight * (
        target.diagonal_precision - current.diagonal_precision
    )
    transition = current.transition_precision + matrix_weight * (
        target.transition_precision - current.transition_precision
    )
    vector = current.information_vector + vector_weight * (
        target.information_vector - current.information_vector
    )
    return GaussianMarkovInformation(
        diagonal,
        transition,
        vector,
        node_valid=current.node_valid,
        information_id=current.information_id,
        rank_tolerance=current.rank_tolerance,
    )


def _information_residual(
    current: GaussianMarkovInformation,
    target: GaussianMarkovInformation,
    /,
) -> Array:
    diagonal_difference = target.diagonal_precision - current.diagonal_precision
    transition_difference = target.transition_precision - current.transition_precision
    vector_difference = target.information_vector - current.information_vector
    difference_norm = jnp.sqrt(
        jnp.sum(diagonal_difference**2, axis=(-3, -2, -1))
        + jnp.sum(transition_difference**2, axis=(-3, -2, -1))
        + jnp.sum(vector_difference**2, axis=(-2, -1))
    )
    target_norm = jnp.sqrt(
        jnp.sum(target.diagonal_precision**2, axis=(-3, -2, -1))
        + jnp.sum(target.transition_precision**2, axis=(-3, -2, -1))
        + jnp.sum(target.information_vector**2, axis=(-2, -1))
    )
    return difference_norm / (1.0 + target_norm)


def _select_candidate(values: Array, indices: Array, /) -> Array:
    candidate_count = values.shape[0]
    selector = jnp.moveaxis(
        jax.nn.one_hot(indices, candidate_count, dtype=values.dtype), -1, 0
    )
    trailing = values.ndim - selector.ndim
    return jnp.sum(values * selector.reshape(selector.shape + (1,) * trailing), axis=0)


def sing_step(
    problem: StateSpaceProblem,
    state: SINGState,
    /,
    *,
    step_size: ArrayLike = 1.0,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 12,
    acceptance_tolerance: float = 1e-10,
) -> SINGStepResult:
    """Perform one independently globalized natural-gradient update per case."""
    _validate_state(problem, state)
    if not isinstance(max_backtracks, Integral) or isinstance(max_backtracks, bool):
        raise TypeError("max_backtracks must be an integer.")
    backtracks = int(max_backtracks)
    if backtracks < 0:
        raise ValueError("max_backtracks must be nonnegative.")
    factor = float(backtrack_factor)
    if not isfinite(factor) or factor <= 0.0 or factor >= 1.0:
        raise ValueError(
            "backtrack_factor must be finite and strictly between zero and one."
        )
    tolerance = _nonnegative_float(acceptance_tolerance, owner="acceptance_tolerance")
    case_shape = state.grid.case_shape
    base_step = jnp.broadcast_to(jnp.asarray(step_size), case_shape)
    _, current_elbo, target = _sing_statistics(problem, state, natural_target=True)
    if target is None:
        raise RuntimeError("SING natural-target construction failed.")
    candidate_information: list[GaussianMarkovInformation] = []
    candidate_acceptance: list[Array] = []
    for backtrack_index in range(backtracks + 1):
        candidate_step = base_step * factor**backtrack_index
        information = _interpolate_information(state.information, target, candidate_step)
        candidate_state = _state_with_information(
            state,
            information,
            state.iteration + 1,
            state.valid,
            state.status,
        )
        _, candidate_elbo, _ = _sing_statistics(
            problem, candidate_state, natural_target=False
        )
        scale = 1.0 + jnp.abs(current_elbo.per_case_elbo)
        acceptable = (
            state.valid
            & current_elbo.valid
            & candidate_elbo.valid
            & (candidate_step > 0.0)
            & (
                candidate_elbo.per_case_elbo
                >= current_elbo.per_case_elbo - tolerance * scale
            )
        )
        candidate_information.append(information)
        candidate_acceptance.append(acceptable)

    acceptance = jnp.stack(candidate_acceptance)
    accepted = jnp.any(acceptance, axis=0)
    selected_index = jnp.argmax(acceptance, axis=0)
    candidate_steps = jnp.stack(
        [base_step * factor**index for index in range(backtracks + 1)]
    )
    accepted_step_size = jnp.where(
        accepted,
        _select_candidate(candidate_steps, selected_index),
        0.0,
    )
    diagonal = _select_candidate(
        jnp.stack(
            [information.diagonal_precision for information in candidate_information]
        ),
        selected_index,
    )
    transition = _select_candidate(
        jnp.stack(
            [information.transition_precision for information in candidate_information]
        ),
        selected_index,
    )
    vector = _select_candidate(
        jnp.stack(
            [information.information_vector for information in candidate_information]
        ),
        selected_index,
    )
    diagonal = jnp.where(
        accepted[..., None, None, None],
        diagonal,
        state.information.diagonal_precision,
    )
    transition = jnp.where(
        accepted[..., None, None, None],
        transition,
        state.information.transition_precision,
    )
    vector = jnp.where(
        accepted[..., None, None],
        vector,
        state.information.information_vector,
    )
    selected_information = GaussianMarkovInformation(
        diagonal,
        transition,
        vector,
        node_valid=state.information.node_valid,
        information_id=state.information.information_id,
        rank_tolerance=state.information.rank_tolerance,
    )
    selected_status = jnp.where(accepted, SING_SUCCESS, SING_LINE_SEARCH_FAILURE).astype(
        jnp.int32
    )
    selected_state = _state_with_information(
        state,
        selected_information,
        state.iteration + 1,
        state.valid & accepted,
        selected_status,
    )
    moments, elbo, next_target = _sing_statistics(
        problem, selected_state, natural_target=True
    )
    if next_target is None:
        raise RuntimeError("SING natural-target construction failed.")
    residual = _information_residual(selected_information, next_target)
    valid = selected_state.valid & moments.valid & elbo.valid
    status = jnp.where(
        ~accepted,
        SING_LINE_SEARCH_FAILURE,
        jnp.where(valid, SING_SUCCESS, elbo.status),
    ).astype(jnp.int32)
    final_state = _state_with_information(
        selected_state,
        selected_information,
        selected_state.iteration,
        valid,
        status,
    )
    return SINGStepResult(
        final_state,
        moments,
        elbo,
        accepted_step_size,
        residual,
        accepted,
        valid,
        status,
    )


def _merge_frozen_state(
    previous: SINGState,
    proposed: SINGState,
    frozen: Array,
    /,
) -> SINGState:
    diagonal = jnp.where(
        frozen[..., None, None, None],
        previous.information.diagonal_precision,
        proposed.information.diagonal_precision,
    )
    transition = jnp.where(
        frozen[..., None, None, None],
        previous.information.transition_precision,
        proposed.information.transition_precision,
    )
    vector = jnp.where(
        frozen[..., None, None],
        previous.information.information_vector,
        proposed.information.information_vector,
    )
    information = GaussianMarkovInformation(
        diagonal,
        transition,
        vector,
        node_valid=previous.information.node_valid,
        information_id=previous.information.information_id,
        rank_tolerance=previous.information.rank_tolerance,
    )
    return _state_with_information(
        previous,
        information,
        proposed.iteration,
        jnp.where(frozen, previous.valid, proposed.valid),
        jnp.where(frozen, previous.status, proposed.status),
    )


def sing_smoother(
    problem: StateSpaceProblem,
    /,
    *,
    state: SINGState | None = None,
    key: Array | None = None,
    expectation_method: SINGExpectationMethod = "cubature",
    method: SINGExecutionMethod = "auto",
    num_samples: int = 64,
    order: int = 3,
    max_dimension: int = 5,
    max_points: int = 100_000,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    rank_tolerance: float = 0.0,
    max_iterations: int = 100,
    step_size: ArrayLike = 1.0,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 12,
    acceptance_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-6,
) -> SINGResult:
    """Infer a Gaussian Markov path posterior for a represented Euler SDE."""
    iterations = _positive_int(max_iterations, owner="max_iterations")
    absolute = _nonnegative_float(absolute_tolerance, owner="absolute_tolerance")
    relative = _nonnegative_float(relative_tolerance, owner="relative_tolerance")
    if state is None:
        initial_state = initialize_sing(
            problem,
            key=key,
            expectation_method=expectation_method,
            method=method,
            num_samples=num_samples,
            order=order,
            max_dimension=max_dimension,
            max_points=max_points,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            rank_tolerance=rank_tolerance,
        )
    else:
        _validate_state(problem, state)
        if key is not None:
            raise ValueError(
                "key cannot be replaced when restarting from an existing SINGState."
            )
        initial_state = state
    steps = jnp.asarray(step_size)
    if steps.ndim == 0:
        schedule = jnp.broadcast_to(steps, (iterations,))
    elif steps.shape == (iterations,):
        schedule = steps
    else:
        raise ValueError(
            f"step_size must be scalar or have shape ({iterations},); got {steps.shape}."
        )
    initial_converged = jnp.zeros(initial_state.grid.case_shape, dtype=bool)

    def iteration_step(carry, scheduled_step):
        current_state, converged = carry
        update = sing_step(
            problem,
            current_state,
            step_size=scheduled_step,
            backtrack_factor=backtrack_factor,
            max_backtracks=max_backtracks,
            acceptance_tolerance=acceptance_tolerance,
        )
        next_state = _merge_frozen_state(current_state, update.state, converged)
        residual = jnp.where(converged, 0.0, update.natural_residual)
        newly_converged = update.valid & (residual <= absolute + relative)
        next_converged = converged | newly_converged
        step_history = jnp.where(converged, 0.0, update.accepted_step_size)
        accepted_history = converged | update.accepted
        return (next_state, next_converged), (
            update.elbo.per_case_elbo,
            step_history,
            residual,
            accepted_history,
        )

    (final_state, converged), history = jax.lax.scan(
        iteration_step,
        (initial_state, initial_converged),
        schedule,
    )
    elbo_history, step_history, residual_history, accepted_history = history
    moments, elbo, _ = _sing_statistics(problem, final_state, natural_target=False)
    valid = final_state.valid & moments.valid & elbo.valid
    status = jnp.where(
        ~valid,
        final_state.status,
        jnp.where(converged, SING_SUCCESS, SING_MAXIMUM_ITERATIONS),
    ).astype(jnp.int32)
    final_state = _state_with_information(
        final_state,
        final_state.information,
        final_state.iteration,
        valid,
        status,
    )
    return SINGResult(
        final_state,
        moments,
        elbo,
        elbo_history,
        step_history,
        residual_history,
        accepted_history,
        converged,
        valid,
        status,
        max_iterations=iterations,
        expectation_method=final_state.expectation_method,
        execution_method=moments.execution_method,
        approximation_id="sing",
        model_id=problem.model.model_id,
        problem_id=problem.problem_id,
        process_id=final_state.process_id,
        sequence_id=problem.observations.sequence_id,
        case_axes=problem.observations.case_axes,
        case_shape=problem.observations.case_shape,
        case_ids=problem.observations.case_ids,
        state_shape=problem.model.state_shape,
    )


def sample_sing_paths(
    key: Array,
    result: SINGResult,
    /,
    *,
    sample_shape: Sequence[int] = (),
) -> Array:
    """Draw coherent posterior paths aligned to the observation schedule."""
    if not isinstance(result, SINGResult):
        raise TypeError("result must be a SINGResult.")
    samples = tuple(int(size) for size in sample_shape)
    paths = sample_gaussian_markov(key, result.moments, sample_shape=samples)
    case_count = prod(result.case_shape) if result.case_shape else 1
    state_size = result.moments.state_size
    num_nodes = result.moments.num_nodes
    num_observations = result.state.grid.num_observations
    flattened = paths.reshape(samples + (case_count, num_nodes, state_size))
    indices = result.state.grid.observation_node_indices.reshape(
        (case_count, num_observations)
    )
    broadcast_indices = jnp.broadcast_to(
        indices,
        samples + (case_count, num_observations),
    )
    gathered = jnp.take_along_axis(
        flattened,
        broadcast_indices[..., None],
        axis=len(samples) + 1,
    )
    return gathered.reshape(
        samples + result.case_shape + (num_observations,) + result.state_shape
    )


__all__ = [
    "initialize_sing",
    "sample_sing_paths",
    "sing_elbo",
    "sing_smoother",
    "sing_status_name",
    "sing_step",
    "SINGELBOResult",
    "SINGExecutionMethod",
    "SINGExpectationMethod",
    "SINGGrid",
    "SINGResult",
    "SINGState",
    "SINGStatus",
    "SINGStepResult",
    "SING_INFORMATION_NOT_POSITIVE_DEFINITE",
    "SING_INITIALIZATION_FAILURE",
    "SING_LINE_SEARCH_FAILURE",
    "SING_MAXIMUM_ITERATIONS",
    "SING_NONFINITE",
    "SING_SUCCESS",
    "SING_TRANSITION_COVARIANCE_FAILURE",
]
