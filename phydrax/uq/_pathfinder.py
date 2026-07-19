#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from typing import Any, Literal

import blackjax
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


class PathfinderResult(StrictModule):
    """L-BFGS-path Gaussian approximation and coherent posterior draws."""

    problem: PosteriorProblem
    state: Any
    path: Any
    samples: PyTree[Array]
    unconstrained_samples: PyTree[Array]
    log_density: Array
    log_approximation_density: Array
    root_key: Array
    approximation_duration_seconds: float = eqx.field(static=True)
    sampling_duration_seconds: float = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)
    optimization_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: PosteriorProblem,
        state: Any,
        path: Any,
        samples: PyTree[Array],
        unconstrained_samples: PyTree[Array],
        log_density: Array,
        log_approximation_density: Array,
        root_key: Array,
        approximation_duration_seconds: float,
        sampling_duration_seconds: float,
        optimization_steps: int,
    ):
        self.problem = problem
        self.state = state
        self.path = path
        self.samples = samples
        self.unconstrained_samples = unconstrained_samples
        self.log_density = jnp.asarray(log_density)
        self.log_approximation_density = jnp.asarray(log_approximation_density)
        self.root_key = jnp.asarray(root_key)
        self.approximation_duration_seconds = float(approximation_duration_seconds)
        self.sampling_duration_seconds = float(sampling_duration_seconds)
        self.sample_memory_bytes = _tree_nbytes(samples) + _tree_nbytes(
            unconstrained_samples
        )
        self.optimization_steps = int(optimization_steps)

    @property
    def elbo(self) -> Array:
        """Monte Carlo ELBO of the selected path location."""
        return jnp.asarray(self.state.elbo)

    @property
    def num_samples(self) -> int:
        return int(self.log_density.shape[0])

    @property
    def importance_log_weights(self) -> Array:
        """Unnormalized target-to-approximation log density ratios."""
        return self.log_density - self.log_approximation_density

    @property
    def duration_seconds(self) -> float:
        return self.approximation_duration_seconds + self.sampling_duration_seconds

    def sample_approximation(
        self,
        key: Array,
        /,
        *,
        num_samples: int,
        constrained: bool = True,
    ) -> PyTree[Array]:
        """Draw fresh samples from the selected Pathfinder approximation."""
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        positions, _ = blackjax.pathfinder.sample(key, self.state, count)
        if constrained:
            return self.problem.parameter_space.constrain(positions)
        return positions

    def predict(
        self,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Evaluate latent predictions while retaining the draw dimension."""
        return predict_from_position_samples(
            self.problem,
            self.unconstrained_samples,
            *args,
            sample_dims=("__phydra_uq_draw",),
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
        observation_dim: str = "__phydra_uq_observation",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Draw conditional measurements for every Pathfinder parameter draw."""
        return sample_observations_from_position_samples(
            self.problem,
            key,
            self.unconstrained_samples,
            *args,
            sample_dims=("__phydra_uq_draw",),
            sample_sources=("epistemic",),
            num_observation_samples=num_observation_samples,
            batch_size=batch_size,
            valid_policy=valid_policy,
            observation_dim=observation_dim,
            **kwargs,
        )


def fit_pathfinder(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    num_samples: int = 1_000,
    num_elbo_samples: int = 200,
    initial_position: PyTree[Array] | None = None,
    max_steps: int = 100,
    history_size: int = 10,
    max_line_search_steps: int = 1_000,
    gradient_tolerance: float = 1e-8,
    relative_objective_tolerance: float = 1e-5,
) -> PathfinderResult:
    """Fit BlackJAX Pathfinder and draw from its best local approximation."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    draws = int(num_samples)
    elbo_draws = int(num_elbo_samples)
    steps = int(max_steps)
    history = int(history_size)
    line_search_steps = int(max_line_search_steps)
    if draws <= 0:
        raise ValueError("num_samples must be positive.")
    if elbo_draws <= 0:
        raise ValueError("num_elbo_samples must be positive.")
    if steps <= 0 or history <= 0 or line_search_steps <= 0:
        raise ValueError(
            "max_steps, history_size, and max_line_search_steps must be positive."
        )
    if float(gradient_tolerance) < 0.0:
        raise ValueError("gradient_tolerance must be non-negative.")
    if float(relative_objective_tolerance) < 0.0:
        raise ValueError("relative_objective_tolerance must be non-negative.")

    position = problem.initial_position if initial_position is None else initial_position
    problem.parameter_space.constrain(position)
    initial_value, initial_gradient = jax.value_and_grad(problem.log_density)(position)
    if not bool(jnp.isfinite(initial_value)) or any(
        bool(jnp.any(~jnp.isfinite(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(initial_gradient)
    ):
        raise FloatingPointError(
            "Initial Pathfinder log density and gradient must be finite."
        )

    approximation_key, sample_key = jr.split(key)
    started = time.perf_counter()
    state, info = blackjax.pathfinder.approximate(
        approximation_key,
        problem.log_density,
        position,
        num_samples=elbo_draws,
        maxiter=steps,
        maxcor=history,
        maxls=line_search_steps,
        gtol=float(gradient_tolerance),
        ftol=float(relative_objective_tolerance),
    )
    jax.block_until_ready(state.elbo)
    approximation_duration = time.perf_counter() - started
    if not bool(jnp.isfinite(state.elbo)):
        raise FloatingPointError("Pathfinder did not produce a finite ELBO.")

    sampling_started = time.perf_counter()
    unconstrained_samples, log_q = blackjax.pathfinder.sample(
        sample_key,
        state,
        draws,
    )
    log_density = jax.vmap(problem.log_density)(unconstrained_samples)
    samples = problem.parameter_space.constrain(unconstrained_samples)
    jax.block_until_ready(log_density)
    sampling_duration = time.perf_counter() - sampling_started
    if not bool(jnp.all(jnp.isfinite(log_density))) or not bool(
        jnp.all(jnp.isfinite(log_q))
    ):
        raise FloatingPointError("Pathfinder produced non-finite posterior draws.")

    optimization_steps = int(jnp.sum(jnp.isfinite(info.path.elbo)))
    return PathfinderResult(
        problem=problem,
        state=state,
        path=info.path,
        samples=samples,
        unconstrained_samples=unconstrained_samples,
        log_density=log_density,
        log_approximation_density=log_q,
        root_key=key,
        approximation_duration_seconds=approximation_duration,
        sampling_duration_seconds=sampling_duration,
        optimization_steps=optimization_steps,
    )


def _tree_nbytes(tree: PyTree[Any], /) -> int:
    return sum(
        int(jnp.asarray(leaf).nbytes)
        for leaf in jax.tree_util.tree_leaves(tree)
        if eqx.is_array(leaf)
    )


__all__ = ["PathfinderResult", "fit_pathfinder"]
