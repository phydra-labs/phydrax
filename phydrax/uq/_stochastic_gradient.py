#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint
from .._strict import StrictModule
from ._minibatch_posterior import LikelihoodBatch, MinibatchPosteriorProblem
from ._parameterized_state_space import ParameterizedStateSpaceProblem
from ._particle import (
    bootstrap_particle_filter,
    ResamplingMethod,
    ResamplingPolicy,
)
from ._particle_parameter_score import parameterized_particle_genealogical_score


STOCHASTIC_GRADIENT_SUCCESS = 0
STOCHASTIC_GRADIENT_INVALID = 1


class StochasticGradientEstimate(StrictModule):
    """One stochastic log-density gradient with explicit numerical validity."""

    gradient: PyTree[Array]
    log_density: Array
    gradient_norm: Array
    valid: Array
    status: Array
    likelihood_estimate: Array
    estimator_id: str = eqx.field(static=True)


class AbstractStochasticGradientEstimator(StrictModule):
    """Replaceable gradient source for fixed-step stochastic-gradient MCMC."""

    @property
    @abstractmethod
    def estimator_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def supports_control_variate(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def configuration(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def estimate(
        self,
        problem: MinibatchPosteriorProblem,
        position: PyTree[Any],
        batch: LikelihoodBatch,
        key: Array,
        /,
    ) -> StochasticGradientEstimate:
        raise NotImplementedError


class AutodiffStochasticGradientEstimator(AbstractStochasticGradientEstimator):
    """Current unbiased factor-minibatch gradient implemented by autodiff."""

    @property
    def estimator_id(self) -> str:
        return "autodiff-factor-minibatch"

    @property
    def supports_control_variate(self) -> bool:
        return True

    def configuration(self) -> dict[str, Any]:
        return {"estimator_id": self.estimator_id}

    def estimate(
        self,
        problem: MinibatchPosteriorProblem,
        position: PyTree[Any],
        batch: LikelihoodBatch,
        key: Array,
        /,
    ) -> StochasticGradientEstimate:
        del key
        value, gradient = jax.value_and_grad(problem.log_density_estimate)(
            position,
            batch,
        )
        gradient_norm = optax.tree.norm(gradient)
        finite = (
            jnp.isfinite(value)
            & jnp.isfinite(gradient_norm)
            & jnp.all(
                jnp.stack(
                    [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient)]
                )
            )
        )
        physical = problem.parameter_space.constrain(position)
        likelihood = problem.log_likelihood_estimate(physical, batch)
        return StochasticGradientEstimate(
            gradient=gradient,
            log_density=value,
            gradient_norm=gradient_norm,
            valid=finite,
            status=jnp.where(
                finite,
                STOCHASTIC_GRADIENT_SUCCESS,
                STOCHASTIC_GRADIENT_INVALID,
            ).astype(jnp.int32),
            likelihood_estimate=likelihood,
            estimator_id=self.estimator_id,
        )


class ParticleGenealogicalGradientEstimator(AbstractStochasticGradientEstimator):
    """Complete-sequence particle likelihood score plus exact parameter prior."""

    parameterized: ParameterizedStateSpaceProblem
    num_particles: int = eqx.field(static=True)
    resampling_method: ResamplingMethod = eqx.field(static=True)
    resampling_policy: ResamplingPolicy = eqx.field(static=True)
    resampling_threshold: float = eqx.field(static=True)

    def __init__(
        self,
        parameterized: ParameterizedStateSpaceProblem,
        /,
        *,
        num_particles: int,
        resampling_method: ResamplingMethod = "systematic",
        resampling_policy: ResamplingPolicy = "ess",
        resampling_threshold: float = 0.5,
    ):
        if not isinstance(parameterized, ParameterizedStateSpaceProblem):
            raise TypeError("parameterized must be ParameterizedStateSpaceProblem.")
        count = int(num_particles)
        if count < 1:
            raise ValueError("num_particles must be positive.")
        if resampling_method not in (
            "systematic",
            "stratified",
            "multinomial",
            "residual",
        ):
            raise ValueError("Unknown particle resampling method.")
        if resampling_policy not in ("ess", "always", "never"):
            raise ValueError("Unknown particle resampling policy.")
        threshold = float(resampling_threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("resampling_threshold must lie in (0, 1].")
        self.parameterized = parameterized
        self.num_particles = count
        self.resampling_method = resampling_method
        self.resampling_policy = resampling_policy
        self.resampling_threshold = threshold

    @property
    def estimator_id(self) -> str:
        return "particle-complete-sequence-genealogical"

    @property
    def supports_control_variate(self) -> bool:
        return False

    def configuration(self) -> dict[str, Any]:
        return {
            "estimator_id": self.estimator_id,
            "parameterization_id": self.parameterized.parameterization_id,
            "parameterized_digest": array_tree_fingerprint(self.parameterized)["sha256"],
            "num_particles": self.num_particles,
            "resampling_method": self.resampling_method,
            "resampling_policy": self.resampling_policy,
            "resampling_threshold": self.resampling_threshold,
        }

    def estimate(
        self,
        problem: MinibatchPosteriorProblem,
        position: PyTree[Any],
        batch: LikelihoodBatch,
        key: Array,
        /,
    ) -> StochasticGradientEstimate:
        del batch
        if (
            problem.parameter_space.raw_shapes
            != self.parameterized.parameter_space.raw_shapes
        ):
            raise ValueError(
                "SG-MCMC and parameterized state-space coordinates do not match."
            )
        bound_problem = self.parameterized.bind(position)
        filtered = bootstrap_particle_filter(
            key,
            bound_problem,
            num_particles=self.num_particles,
            resampling_method=self.resampling_method,
            resampling_policy=self.resampling_policy,
            resampling_threshold=self.resampling_threshold,
        )
        likelihood_score = parameterized_particle_genealogical_score(
            self.parameterized,
            position,
            filtered,
        )
        prior_gradient = jax.grad(problem.parameter_space.unconstrained_log_prior)(
            position
        )
        gradient = jax.tree.map(
            jnp.add,
            prior_gradient,
            likelihood_score.gradient,
        )
        gradient_norm = optax.tree.norm(gradient)
        likelihood_estimate = jnp.sum(filtered.final_state.log_likelihood)
        log_density = (
            likelihood_estimate
            + problem.parameter_space.unconstrained_log_prior(position)
        )
        finite = (
            jnp.all(likelihood_score.valid)
            & jnp.isfinite(log_density)
            & jnp.isfinite(gradient_norm)
            & jnp.all(
                jnp.stack(
                    [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient)]
                )
            )
        )
        return StochasticGradientEstimate(
            gradient=gradient,
            log_density=log_density,
            gradient_norm=gradient_norm,
            valid=finite,
            status=jnp.where(
                finite,
                STOCHASTIC_GRADIENT_SUCCESS,
                STOCHASTIC_GRADIENT_INVALID,
            ).astype(jnp.int32),
            likelihood_estimate=likelihood_estimate,
            estimator_id=self.estimator_id,
        )


__all__ = [
    "AbstractStochasticGradientEstimator",
    "AutodiffStochasticGradientEstimator",
    "ParticleGenealogicalGradientEstimator",
    "STOCHASTIC_GRADIENT_INVALID",
    "STOCHASTIC_GRADIENT_SUCCESS",
    "StochasticGradientEstimate",
]
