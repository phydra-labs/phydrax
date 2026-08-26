#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..kernels import FiniteFeatureKernel
from ._covariance import _factor_and_solve_covariance_system
from ._gp_likelihood import GaussianProcessLikelihoodState
from ._gp_scalar import ExactGaussianProcessFactor, FiniteFeatureGaussianProcessFactor


class BernoulliGaussianProcessLikelihood(StrictModule):
    """Bernoulli observation likelihood for a scalar latent GP."""

    def probability(self, latent: ArrayLike, /) -> Array:
        return jax.nn.sigmoid(jnp.asarray(latent))

    def log_probability(self, observation: ArrayLike, latent: ArrayLike, /) -> Array:
        y = jnp.asarray(observation)
        f = jnp.asarray(latent)
        if y.shape != f.shape:
            raise ValueError("Bernoulli observations and latent values must share shape.")
        valid = jnp.isfinite(y) & jnp.isfinite(f) & ((y == 0) | (y == 1))
        safe_y = jnp.where(valid, y, 0)
        safe_f = jnp.where(valid, f, 0)
        value = safe_y * jax.nn.log_sigmoid(safe_f) + (1.0 - safe_y) * jax.nn.log_sigmoid(
            -safe_f
        )
        return jnp.where(valid, value, -jnp.inf)


class CategoricalGaussianProcessLikelihood(StrictModule):
    """Categorical softmax likelihood over independent latent GP logits."""

    class_count: int = eqx.field(static=True)

    def __init__(self, class_count: int):
        if isinstance(class_count, bool):
            raise TypeError("class_count must be an integer.")
        count = index(class_count)
        if count < 2:
            raise ValueError("class_count must be at least two.")
        self.class_count = count

    def probabilities(self, latent: ArrayLike, /) -> Array:
        logits = jnp.asarray(latent)
        if logits.shape[-1] != self.class_count:
            raise ValueError("Latent class axis does not match class_count.")
        return jax.nn.softmax(logits, axis=-1)

    def log_probability(self, observation: ArrayLike, latent: ArrayLike, /) -> Array:
        raw_labels = jnp.asarray(observation)
        logits = jnp.asarray(latent)
        if logits.ndim == 0 or raw_labels.shape != logits.shape[:-1]:
            raise ValueError("Categorical labels must match the latent batch shape.")
        labels_value = raw_labels.astype(jnp.result_type(raw_labels, 0.0))
        valid = (
            jnp.all(jnp.isfinite(logits), axis=-1)
            & jnp.isfinite(labels_value)
            & (labels_value >= 0)
            & (labels_value < self.class_count)
            & (labels_value == jnp.floor(labels_value))
        )
        labels = jnp.where(valid, labels_value, 0).astype(jnp.int32)
        safe_logits = jnp.where(valid[..., None], logits, 0)
        selected = jnp.take_along_axis(
            jax.nn.log_softmax(safe_logits, axis=-1),
            labels[..., None],
            axis=-1,
        )[..., 0]
        return jnp.where(valid, selected, -jnp.inf)


class BernoulliGaussianProcessPosterior(StrictModule):
    """Laplace posterior represented through the existing exact GP conditioner."""

    factor: ExactGaussianProcessFactor | FiniteFeatureGaussianProcessFactor
    pseudo_observations: Array
    likelihood: BernoulliGaussianProcessLikelihood
    iterations: int = eqx.field(static=True)

    def latent_moments(self, query_points: ArrayLike, /) -> tuple[Array, Array]:
        condition = self.factor.condition(
            self.pseudo_observations, query_points, output_dim="point"
        )
        return condition.mean, condition.variance

    def probabilities(self, query_points: ArrayLike, /) -> Array:
        mean, variance = self.latent_moments(query_points)
        # Logistic-Gaussian moment approximation; unlike thresholding this is smooth.
        return jax.nn.sigmoid(
            mean / jnp.sqrt(1.0 + jnp.pi * jnp.maximum(variance, 0.0) / 8.0)
        )

    def predict(self, query_points: ArrayLike, /) -> Array:
        return (self.probabilities(query_points) >= 0.5).astype(jnp.int32)


class CategoricalGaussianProcessPosterior(StrictModule):
    """Independent latent-GP Laplace factors normalized by one softmax path."""

    factors: tuple[BernoulliGaussianProcessPosterior, ...]
    likelihood: CategoricalGaussianProcessLikelihood

    def __init__(self, factors: tuple[BernoulliGaussianProcessPosterior, ...]):
        if len(factors) < 2:
            raise ValueError(
                "Categorical GP posterior requires at least two latent factors."
            )
        self.factors = tuple(factors)
        self.likelihood = CategoricalGaussianProcessLikelihood(len(factors))

    def latent_moments(self, query_points: ArrayLike, /) -> tuple[Array, Array]:
        moments = tuple(factor.latent_moments(query_points) for factor in self.factors)
        means = jnp.stack(tuple(value[0] for value in moments), axis=-1)
        variances = jnp.stack(tuple(value[1] for value in moments), axis=-1)
        return means, variances

    def probabilities(self, query_points: ArrayLike, /) -> Array:
        means, variances = self.latent_moments(query_points)
        corrected = means / jnp.sqrt(1.0 + jnp.pi * jnp.maximum(variances, 0.0) / 8.0)
        return self.likelihood.probabilities(corrected)

    def predict(self, query_points: ArrayLike, /) -> Array:
        return jnp.argmax(self.probabilities(query_points), axis=-1).astype(jnp.int32)


def condition_bernoulli_gaussian_process(
    observation_points: ArrayLike,
    observations: ArrayLike,
    /,
    *,
    state: GaussianProcessLikelihoodState,
    observation_weight: ArrayLike | None = None,
    iterations: int = 12,
    curvature_floor: ArrayLike = 1e-6,
) -> BernoulliGaussianProcessPosterior:
    """Fit a Bernoulli Laplace site and route it through exact GP factors."""
    points = jnp.asarray(observation_points, dtype=float)
    labels = jnp.asarray(observations, dtype=float)
    if points.ndim != 2 or labels.shape != (points.shape[0],):
        raise ValueError(
            "Bernoulli GP data must have shapes (sample, feature) and (sample,)."
        )
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive.")
    if not isinstance(state, GaussianProcessLikelihoodState):
        raise TypeError("state must be a GaussianProcessLikelihoodState.")
    floor = jnp.asarray(curvature_floor, dtype=float)
    if floor.ndim != 0 or bool(floor <= 0):
        raise ValueError("curvature_floor must be positive.")
    weight = (
        jnp.ones_like(labels)
        if observation_weight is None
        else jnp.broadcast_to(jnp.asarray(observation_weight, dtype=float), labels.shape)
    )
    weight = eqx.error_if(
        weight,
        jnp.any(~jnp.isfinite(weight)) | jnp.any(weight < 0.0),
        "observation_weight must be finite and nonnegative.",
    )
    labels = eqx.error_if(
        labels,
        jnp.any((weight > 0) & (labels != 0.0) & (labels != 1.0)),
        "Positive-weight Bernoulli observations must be zero or one.",
    )
    labels = jnp.where(weight > 0, labels, 0.0)
    point_finite = jnp.all(jnp.isfinite(points), axis=-1)
    points = eqx.error_if(
        points,
        jnp.any((weight > 0) & ~point_finite),
        "Positive-weight GP observation points must be finite.",
    )
    points = jnp.where(point_finite[:, None], points, 0.0)
    kernel_matrix = state.kernel.matrix(points, points)
    identity = jnp.eye(points.shape[0], dtype=kernel_matrix.dtype)
    latent0 = jnp.zeros_like(labels)

    def newton_step(_, latent):
        probabilities = jax.nn.sigmoid(latent)
        curvature = jnp.where(
            weight > 0,
            jnp.maximum(weight * probabilities * (1.0 - probabilities), floor),
            0.0,
        )
        root = jnp.sqrt(curvature)
        gradient = weight * (labels - probabilities)
        b = curvature * latent + gradient
        system = identity + root[:, None] * kernel_matrix * root[None, :]
        correction, _ = _factor_and_solve_covariance_system(
            system,
            root * (kernel_matrix @ b),
        )
        correction = correction.value
        alpha = b - root * correction
        return kernel_matrix @ alpha

    latent = jax.lax.fori_loop(0, int(iterations), newton_step, latent0)
    probability = jax.nn.sigmoid(latent)
    curvature = jnp.where(
        weight > 0,
        jnp.maximum(weight * probability * (1.0 - probability), floor),
        0.0,
    )
    pseudo = jnp.where(
        weight > 0,
        latent + weight * (labels - probability) / jnp.maximum(curvature, floor),
        0.0,
    )
    inactive_noise = jnp.sqrt(jnp.sqrt(jnp.finfo(curvature.dtype).max))
    noise_scale = jnp.where(weight > 0, jax.lax.rsqrt(curvature), inactive_noise)
    approximate_state = GaussianProcessLikelihoodState(
        kernel=state.kernel,
        noise_scale=noise_scale,
        jitter=state.jitter,
    )
    factor = (
        FiniteFeatureGaussianProcessFactor(points, state=approximate_state)
        if isinstance(state.kernel, FiniteFeatureKernel)
        else ExactGaussianProcessFactor(points, state=approximate_state)
    )
    return BernoulliGaussianProcessPosterior(
        factor=factor,
        pseudo_observations=pseudo,
        likelihood=BernoulliGaussianProcessLikelihood(),
        iterations=int(iterations),
    )


def condition_categorical_gaussian_process(
    observation_points: ArrayLike,
    observations: ArrayLike,
    /,
    *,
    state: GaussianProcessLikelihoodState,
    observation_weight: ArrayLike | None = None,
    class_count: int,
    iterations: int = 12,
    curvature_floor: ArrayLike = 1e-6,
) -> CategoricalGaussianProcessPosterior:
    """Fit fixed-capacity one-vs-rest Laplace sites sharing one GP kernel state."""
    raw_labels = jnp.asarray(observations)
    if isinstance(class_count, bool):
        raise TypeError("class_count must be an integer.")
    classes = index(class_count)
    if classes < 2:
        raise ValueError("class_count must be at least two.")
    labels_value = raw_labels.astype(jnp.result_type(raw_labels, 0.0))
    exact_labels = jnp.isfinite(labels_value) & (labels_value == jnp.floor(labels_value))
    labels = jnp.where(exact_labels, labels_value, 0).astype(jnp.int32)
    active = (
        jnp.ones_like(labels, dtype=bool)
        if observation_weight is None
        else jnp.broadcast_to(jnp.asarray(observation_weight) > 0, labels.shape)
    )
    labels = eqx.error_if(
        labels,
        jnp.any(active & (~exact_labels | (labels < 0) | (labels >= classes))),
        "Positive-weight categorical labels exceed the configured class capacity.",
    )
    labels = jnp.where(active, labels, 0)
    factors = tuple(
        condition_bernoulli_gaussian_process(
            observation_points,
            (labels == class_index).astype(float),
            state=state,
            observation_weight=observation_weight,
            iterations=iterations,
            curvature_floor=curvature_floor,
        )
        for class_index in range(classes)
    )
    return CategoricalGaussianProcessPosterior(factors)


__all__ = [
    "BernoulliGaussianProcessLikelihood",
    "BernoulliGaussianProcessPosterior",
    "CategoricalGaussianProcessLikelihood",
    "CategoricalGaussianProcessPosterior",
    "condition_bernoulli_gaussian_process",
    "condition_categorical_gaussian_process",
]
