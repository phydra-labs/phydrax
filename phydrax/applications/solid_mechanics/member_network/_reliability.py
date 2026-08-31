#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class StructuralRandomModel(StrictModule, NonTrainableState):
    """Correlated Gaussian latent variables mapped into structural parameters."""

    mean: Array
    covariance: Array
    labels: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        labels: tuple[str, ...],
        /,
        *,
        model_id: str = "structural-random-model",
    ):
        mean_ = jnp.asarray(mean)
        covariance_ = jnp.asarray(covariance, dtype=mean_.dtype)
        if mean_.ndim != 1 or covariance_.shape != (mean_.size, mean_.size):
            raise ValueError("Random-model mean/covariance shapes are incompatible.")
        if len(labels) != mean_.size or len(set(labels)) != mean_.size:
            raise ValueError("Random-variable labels must be unique and complete.")
        symmetry = jnp.max(jnp.abs(covariance_ - covariance_.T))
        eigenvalues = jnp.linalg.eigvalsh(covariance_)
        if bool(symmetry > 1.0e-10) or bool(jnp.any(eigenvalues <= 0.0)):
            raise ValueError(
                "Random-model covariance must be symmetric positive definite."
            )
        self.mean = mean_
        self.covariance = covariance_
        self.labels = tuple(str(value) for value in labels)
        self.model_id = str(model_id)

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    def transform(self, standard_normal: ArrayLike, /) -> Array:
        standard = jnp.asarray(standard_normal, dtype=self.mean.dtype)
        if standard.shape[-1] != self.dimension:
            raise ValueError("Standard-normal samples have the wrong trailing dimension.")
        factor = jnp.linalg.cholesky(self.covariance)
        return self.mean + standard @ factor.T

    def sample(self, key: PRNGKeyArray, sample_count: int, /) -> Array:
        standard = jax.random.normal(
            key, (int(sample_count), self.dimension), dtype=self.mean.dtype
        )
        return self.transform(standard)


class StructuralLimitState(StrictModule, NonTrainableState):
    function: Callable = eqx.field(static=True)
    limit_state_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Array], Array],
        /,
        *,
        limit_state_id: str,
    ):
        if not callable(function):
            raise TypeError("Limit-state function must be callable.")
        self.function = function
        self.limit_state_id = str(limit_state_id)

    def margin(self, parameters: ArrayLike, /) -> Array:
        value = jnp.asarray(self.function(jnp.asarray(parameters)))
        if value.shape != ():
            raise ValueError("Structural limit state must return one scalar margin.")
        return value


class MonteCarloReliabilityResult(StrictModule):
    failure_probability: Array
    standard_error: Array
    failure_count: Array
    sample_count: int = eqx.field(static=True)
    margins: Array


class FORMReliabilityResult(StrictModule):
    reliability_index: Array
    design_point_standard: Array
    design_point_physical: Array
    margin: Array
    gradient: Array
    iterations: Array
    converged: Array
    smooth: Array


def monte_carlo_reliability(
    model: StructuralRandomModel,
    limit_state: StructuralLimitState,
    key: PRNGKeyArray,
    sample_count: int,
    /,
) -> MonteCarloReliabilityResult:
    parameters = model.sample(key, sample_count)
    margins = jax.vmap(limit_state.margin)(parameters)
    failed = margins <= 0.0
    count = jnp.sum(failed, dtype=jnp.int32)
    probability = count / float(sample_count)
    error = jnp.sqrt(probability * (1.0 - probability) / float(sample_count))
    return MonteCarloReliabilityResult(
        probability,
        error,
        count,
        int(sample_count),
        margins,
    )


def form_reliability(
    model: StructuralRandomModel,
    limit_state: StructuralLimitState,
    /,
    *,
    maximum_steps: int = 100,
    tolerance: float = 1.0e-8,
) -> FORMReliabilityResult:
    """Hasofer--Lind--Rackwitz--Fiessler iteration in standard-normal space."""
    factor = jnp.linalg.cholesky(model.covariance)

    def margin_standard(value):
        return limit_state.margin(model.mean + factor @ value)

    point = jnp.zeros((model.dimension,), dtype=model.mean.dtype)
    converged = False
    gradient = jnp.zeros_like(point)
    margin = margin_standard(point)
    iterations = 0
    for index in range(int(maximum_steps)):
        margin, gradient = jax.value_and_grad(margin_standard)(point)
        squared = jnp.dot(gradient, gradient)
        next_point = ((jnp.dot(gradient, point) - margin) / squared) * gradient
        change = jnp.sqrt(jnp.sum((next_point - point) ** 2))
        point = next_point
        iterations = index + 1
        if float(change) <= tolerance and abs(float(margin)) <= tolerance:
            converged = True
            break
    margin, gradient = jax.value_and_grad(margin_standard)(point)
    beta = jnp.sqrt(jnp.sum(point * point))
    smooth = jnp.all(jnp.isfinite(gradient)) & (
        jnp.sqrt(jnp.dot(gradient, gradient)) > 0.0
    )
    return FORMReliabilityResult(
        beta,
        point,
        model.transform(point),
        margin,
        gradient,
        jnp.asarray(iterations, dtype=jnp.int32),
        jnp.asarray(converged),
        smooth,
    )


__all__ = [
    "FORMReliabilityResult",
    "MonteCarloReliabilityResult",
    "StructuralLimitState",
    "StructuralRandomModel",
    "form_reliability",
    "monte_carlo_reliability",
]
