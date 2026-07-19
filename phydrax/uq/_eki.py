#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree

from .._frozendict import frozendict
from .._strict import StrictModule
from ._posterior import PosteriorProblem
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField


class EnsembleKalmanDiagnostics(StrictModule):
    """Tempering, fit, spread, rank, update, and forward-solve evidence."""

    temperature_increments: Array
    residual_norms: Array
    ensemble_spreads: Array
    effective_ranks: Array
    parameter_update_norms: Array
    forward_solve_count: int = eqx.field(static=True)
    collapsed: bool = eqx.field(static=True)
    collapse_step: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        temperature_increments: Array,
        residual_norms: Array,
        ensemble_spreads: Array,
        effective_ranks: Array,
        parameter_update_norms: Array,
        forward_solve_count: int,
        collapse_step: int | None,
    ):
        self.temperature_increments = jnp.asarray(temperature_increments)
        self.residual_norms = jnp.asarray(residual_norms)
        self.ensemble_spreads = jnp.asarray(ensemble_spreads)
        self.effective_ranks = jnp.asarray(effective_ranks)
        self.parameter_update_norms = jnp.asarray(parameter_update_norms)
        self.forward_solve_count = int(forward_solve_count)
        self.collapsed = collapse_step is not None
        self.collapse_step = collapse_step


class EnsembleKalmanResult(StrictModule):
    """Approximate derivative-free inverse ensemble in parameter coordinates."""

    problem: PosteriorProblem
    initial_unconstrained_ensemble: PyTree[Array]
    unconstrained_ensemble: PyTree[Array]
    initial_ensemble: PyTree[Array]
    ensemble: PyTree[Array]
    residuals: Array
    temperatures: Array
    diagnostics: EnsembleKalmanDiagnostics
    root_key: Array
    num_steps: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)
    target_ess: float = eqx.field(static=True)
    inflation: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: PosteriorProblem,
        initial_unconstrained_ensemble: PyTree[Array],
        unconstrained_ensemble: PyTree[Array],
        residuals: Array,
        temperatures: Array,
        diagnostics: EnsembleKalmanDiagnostics,
        root_key: Array,
        num_steps: int,
        converged: bool,
        termination_reason: str,
        duration_seconds: float,
        target_ess: float,
        inflation: float,
    ):
        self.problem = problem
        self.initial_unconstrained_ensemble = initial_unconstrained_ensemble
        self.unconstrained_ensemble = unconstrained_ensemble
        self.initial_ensemble = problem.parameter_space.constrain(
            initial_unconstrained_ensemble
        )
        self.ensemble = problem.parameter_space.constrain(unconstrained_ensemble)
        self.residuals = jnp.asarray(residuals)
        self.temperatures = jnp.asarray(temperatures)
        self.diagnostics = diagnostics
        self.root_key = jnp.asarray(root_key)
        self.num_steps = int(num_steps)
        self.converged = bool(converged)
        self.termination_reason = str(termination_reason)
        self.duration_seconds = float(duration_seconds)
        self.sample_memory_bytes = _tree_nbytes(initial_unconstrained_ensemble) + (
            _tree_nbytes(unconstrained_ensemble)
        )
        self.target_ess = float(target_ess)
        self.inflation = float(inflation)

    @property
    def ensemble_size(self) -> int:
        return int(self.residuals.shape[0])

    @property
    def mean(self) -> PyTree[Array]:
        """Return the mean of the transformed physical ensemble."""
        return jax.tree_util.tree_map(
            lambda value: jnp.mean(value, axis=0), self.ensemble
        )

    @property
    def unconstrained_mean(self) -> PyTree[Array]:
        """Return the mean in the coordinates updated by EKI."""
        return jax.tree_util.tree_map(
            lambda value: jnp.mean(value, axis=0),
            self.unconstrained_ensemble,
        )

    def predict(
        self,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        ensemble_dim: str = "__phydra_uq_ensemble",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Evaluate latent predictions for every final EKI member."""
        return predict_from_position_samples(
            self.problem,
            self.unconstrained_ensemble,
            *args,
            sample_dims=(ensemble_dim,),
            sample_sources=("epistemic",),
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )

    def predict_observations(
        self,
        key: Array,
        /,
        *args: Any,
        num_observation_samples: int = 1,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        ensemble_dim: str = "__phydra_uq_ensemble",
        observation_dim: str = "__phydra_uq_observation",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Draw conditional measurements for every final EKI member."""
        return sample_observations_from_position_samples(
            self.problem,
            key,
            self.unconstrained_ensemble,
            *args,
            sample_dims=(ensemble_dim,),
            sample_sources=("epistemic",),
            num_observation_samples=num_observation_samples,
            batch_size=batch_size,
            valid_policy=valid_policy,
            observation_dim=observation_dim,
            **kwargs,
        )


class EnsembleKalmanConvergenceError(RuntimeError):
    """Raised when EKI cannot complete the requested likelihood tempering."""

    result: EnsembleKalmanResult

    def __init__(self, result: EnsembleKalmanResult):
        self.result = result
        super().__init__(
            f"Ensemble Kalman inversion did not converge: {result.termination_reason}."
        )


def fit_eki(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    ensemble_size: int = 100,
    initial_ensemble: PyTree[Array] | None = None,
    prior_position_sampler: Callable[[Array, int], PyTree[Array]] | None = None,
    target_ess: float = 0.8,
    max_steps: int = 20,
    inflation: float = 1.0,
    jitter: float = 1e-8,
    rank_tolerance: float = 1e-8,
    collapse_tolerance: float = 1e-10,
    raise_on_failure: bool = False,
) -> EnsembleKalmanResult:
    """Fit a tempered stochastic EKI ensemble using normalized residuals only.

    The update is derivative-free and remains in the affine span of the initial
    unconstrained ensemble when ``inflation=1``. It is an exact posterior update
    only in the ideal linear-Gaussian, infinite-ensemble limit.
    """
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    count = int(ensemble_size)
    steps = int(max_steps)
    if count < 3:
        raise ValueError("ensemble_size must be at least three.")
    if steps <= 0:
        raise ValueError("max_steps must be positive.")
    ess_target = float(target_ess)
    if not 0.0 < ess_target < 1.0:
        raise ValueError("target_ess must lie strictly between zero and one.")
    inflation_value = float(inflation)
    if not jnp.isfinite(inflation_value) or inflation_value <= 0.0:
        raise ValueError("inflation must be finite and positive.")
    jitter_value = float(jitter)
    if not jnp.isfinite(jitter_value) or jitter_value < 0.0:
        raise ValueError("jitter must be finite and non-negative.")
    rank_threshold = float(rank_tolerance)
    collapse_threshold = float(collapse_tolerance)
    if rank_threshold <= 0.0 or collapse_threshold < 0.0:
        raise ValueError(
            "rank_tolerance must be positive and collapse_tolerance non-negative."
        )
    if initial_ensemble is not None and prior_position_sampler is not None:
        raise ValueError(
            "initial_ensemble and prior_position_sampler are mutually exclusive."
        )
    if prior_position_sampler is not None and not callable(prior_position_sampler):
        raise TypeError("prior_position_sampler must be callable or None.")

    root_key = jnp.asarray(key)
    prior_key, update_key = jr.split(root_key)
    if initial_ensemble is not None:
        unconstrained_ensemble = initial_ensemble
    elif prior_position_sampler is not None:
        unconstrained_ensemble = prior_position_sampler(prior_key, count)
    else:
        unconstrained_ensemble = problem.parameter_space.sample_prior(
            prior_key,
            num_samples=count,
        )
    _validate_ensemble(problem, unconstrained_ensemble, count)
    initial_unconstrained_ensemble = unconstrained_ensemble
    parameter_matrix, tree_definition, leaf_shapes, leaf_sizes = _ensemble_to_matrix(
        unconstrained_ensemble,
        count,
    )

    started = time.perf_counter()
    residual_matrix = _evaluate_residuals(problem, unconstrained_ensemble, count)
    forward_solve_count = count
    temperatures = [jnp.zeros(())]
    increments: list[Array] = []
    residual_norms = [_residual_norm(residual_matrix)]
    ensemble_spreads = [_ensemble_spread(parameter_matrix)]
    effective_ranks = [_effective_rank(parameter_matrix, rank_threshold)]
    parameter_update_norms: list[Array] = []
    collapse_step = None
    termination_reason = "max_steps"

    for step in range(steps):
        remaining = 1.0 - float(temperatures[-1])
        if remaining <= 1e-10:
            termination_reason = "unit_temperature"
            break
        increment = _adaptive_tempering_increment(
            residual_matrix,
            remaining=remaining,
            target_ess=ess_target,
        )
        if increment <= 0.0 or not jnp.isfinite(increment):
            termination_reason = "tempering_stagnation"
            break
        alpha = 1.0 / increment
        noise_key = jr.fold_in(update_key, step)
        perturbations = jr.normal(
            noise_key,
            residual_matrix.shape,
            dtype=residual_matrix.dtype,
        )
        next_matrix = _eki_update(
            parameter_matrix,
            residual_matrix,
            perturbations,
            jnp.asarray(alpha, dtype=residual_matrix.dtype),
            jnp.asarray(jitter_value, dtype=residual_matrix.dtype),
            jnp.asarray(inflation_value, dtype=residual_matrix.dtype),
        )
        update_norm = jnp.sqrt(jnp.mean((next_matrix - parameter_matrix) ** 2))
        unconstrained_ensemble = _matrix_to_ensemble(
            next_matrix,
            tree_definition,
            leaf_shapes,
            leaf_sizes,
            count,
        )
        _validate_ensemble(problem, unconstrained_ensemble, count)
        residual_matrix = _evaluate_residuals(problem, unconstrained_ensemble, count)
        forward_solve_count += count
        parameter_matrix = next_matrix
        next_temperature = min(1.0, float(temperatures[-1]) + increment)
        temperatures.append(jnp.asarray(next_temperature))
        increments.append(jnp.asarray(increment))
        residual_norms.append(_residual_norm(residual_matrix))
        spread = _ensemble_spread(parameter_matrix)
        ensemble_spreads.append(spread)
        effective_ranks.append(_effective_rank(parameter_matrix, rank_threshold))
        parameter_update_norms.append(update_norm)
        if float(spread) <= collapse_threshold:
            collapse_step = step + 1
            if next_temperature < 1.0 - 1e-10:
                termination_reason = "ensemble_collapse"
                break
    else:
        if float(temperatures[-1]) >= 1.0 - 1e-10:
            termination_reason = "unit_temperature"

    converged = float(temperatures[-1]) >= 1.0 - 1e-10
    diagnostics = EnsembleKalmanDiagnostics(
        temperature_increments=jnp.asarray(increments),
        residual_norms=jnp.asarray(residual_norms),
        ensemble_spreads=jnp.asarray(ensemble_spreads),
        effective_ranks=jnp.asarray(effective_ranks, dtype=jnp.int32),
        parameter_update_norms=jnp.asarray(parameter_update_norms),
        forward_solve_count=forward_solve_count,
        collapse_step=collapse_step,
    )
    jax.block_until_ready(residual_matrix)
    result = EnsembleKalmanResult(
        problem=problem,
        initial_unconstrained_ensemble=initial_unconstrained_ensemble,
        unconstrained_ensemble=unconstrained_ensemble,
        residuals=residual_matrix,
        temperatures=jnp.asarray(temperatures),
        diagnostics=diagnostics,
        root_key=root_key,
        num_steps=len(increments),
        converged=converged,
        termination_reason=termination_reason,
        duration_seconds=time.perf_counter() - started,
        target_ess=ess_target,
        inflation=inflation_value,
    )
    if not result.converged and raise_on_failure:
        raise EnsembleKalmanConvergenceError(result)
    return result


@jax.jit
def _eki_update(
    parameters,
    residuals,
    perturbations,
    alpha,
    jitter,
    inflation,
):
    count = parameters.shape[0]
    parameter_anomalies = parameters - jnp.mean(parameters, axis=0, keepdims=True)
    residual_anomalies = residuals - jnp.mean(residuals, axis=0, keepdims=True)
    innovation = residuals - jnp.sqrt(alpha) * perturbations
    ensemble_system = residual_anomalies @ residual_anomalies.T
    ensemble_system = ensemble_system + ((count - 1) * alpha + jitter) * jnp.eye(
        count, dtype=residuals.dtype
    )
    coefficients = jnp.linalg.solve(
        ensemble_system,
        residual_anomalies @ innovation.T,
    )
    updated = parameters - (parameter_anomalies.T @ coefficients).T
    mean = jnp.mean(updated, axis=0, keepdims=True)
    updated = mean + inflation * (updated - mean)
    return updated


def _adaptive_tempering_increment(residuals, *, remaining, target_ess):
    misfits = 0.5 * jnp.sum(residuals**2, axis=1)

    def relative_ess(increment):
        log_weights = -increment * misfits
        weights = jax.nn.softmax(log_weights)
        return 1.0 / (residuals.shape[0] * jnp.sum(weights**2))

    if float(relative_ess(remaining)) >= target_ess:
        return float(remaining)
    lower = 0.0
    upper = float(remaining)
    for _ in range(50):
        midpoint = 0.5 * (lower + upper)
        if float(relative_ess(midpoint)) >= target_ess:
            lower = midpoint
        else:
            upper = midpoint
    return lower


def _evaluate_residuals(problem, ensemble, count):
    residual_tree = jax.vmap(problem.gauss_newton_residual)(ensemble)
    leaves = jax.tree_util.tree_leaves(residual_tree)
    if not leaves:
        raise ValueError("Gauss-Newton residual must contain at least one array leaf.")
    matrices = []
    for leaf in leaves:
        value = jnp.asarray(leaf)
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(
                "Gauss-Newton residual must preserve the ensemble leading axis."
            )
        matrices.append(value.reshape(count, -1))
    residuals = jnp.concatenate(matrices, axis=1)
    if residuals.shape[1] == 0:
        raise ValueError("Gauss-Newton residual cannot be empty.")
    if not bool(jnp.all(jnp.isfinite(residuals))):
        raise FloatingPointError("Ensemble Kalman residuals must be finite.")
    return residuals


def _validate_ensemble(problem, ensemble, expected_count):
    leaves = jax.tree_util.tree_leaves(ensemble)
    if not leaves or any(
        jnp.asarray(leaf).ndim == 0 or jnp.asarray(leaf).shape[0] != expected_count
        for leaf in leaves
    ):
        raise ValueError(
            "Initial ensemble leaves must share the ensemble_size leading axis."
        )
    one_member = jax.tree_util.tree_map(lambda value: value[0], ensemble)
    problem.parameter_space.constrain(one_member)
    if not all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves):
        raise FloatingPointError("Initial ensemble must be finite.")


def _ensemble_to_matrix(ensemble, count):
    leaves, tree_definition = jax.tree_util.tree_flatten(ensemble)
    shapes = tuple(tuple(int(size) for size in leaf.shape[1:]) for leaf in leaves)
    sizes = tuple(int(jnp.prod(jnp.asarray(shape))) if shape else 1 for shape in shapes)
    matrix = jnp.concatenate(
        [jnp.asarray(leaf).reshape(count, size) for leaf, size in zip(leaves, sizes)],
        axis=1,
    )
    return matrix, tree_definition, shapes, sizes


def _matrix_to_ensemble(matrix, tree_definition, shapes, sizes, count):
    leaves = []
    start = 0
    for shape, size in zip(shapes, sizes, strict=True):
        leaves.append(matrix[:, start : start + size].reshape((count, *shape)))
        start += size
    return jax.tree_util.tree_unflatten(tree_definition, leaves)


def _residual_norm(residuals):
    return jnp.sqrt(jnp.mean(jnp.sum(residuals**2, axis=1)))


def _ensemble_spread(parameters):
    return jnp.sqrt(jnp.sum(jnp.var(parameters, axis=0, ddof=1)))


def _effective_rank(parameters, tolerance):
    singular_values = jnp.linalg.svd(
        parameters - jnp.mean(parameters, axis=0, keepdims=True),
        compute_uv=False,
    )
    threshold = tolerance * jnp.maximum(singular_values[0], 1.0)
    return jnp.sum(singular_values > threshold)


def _tree_nbytes(tree: PyTree[Any], /) -> int:
    return sum(
        int(jnp.asarray(leaf).nbytes)
        for leaf in jax.tree_util.tree_leaves(tree)
        if eqx.is_array(leaf)
    )


__all__ = [
    "EnsembleKalmanConvergenceError",
    "EnsembleKalmanDiagnostics",
    "EnsembleKalmanResult",
    "fit_eki",
]
