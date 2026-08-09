#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._predictive import PredictiveField, SampleAxis


class GaussianProcessCondition(StrictModule):
    """Conditioned scalar latent discrepancy at one fixed query design."""

    query_points: Array
    mean: Array
    covariance: Array
    variance: Array
    output_dims: tuple[str | None, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        query_points: ArrayLike,
        mean: ArrayLike,
        covariance: ArrayLike,
        variance: ArrayLike,
        output_dims: tuple[str | None, ...],
    ):
        points = _as_design(query_points)
        mean_array = _as_vector(mean, name="conditioned GP mean")
        covariance_array = jnp.asarray(covariance, dtype=float)
        variance_array = _as_vector(variance, name="conditioned GP variance")
        count = int(points.shape[0])
        if mean_array.shape != (count,) or variance_array.shape != (count,):
            raise ValueError("Conditioned GP moments must align with query points.")
        if covariance_array.shape != (count, count):
            raise ValueError(
                "Conditioned GP covariance must be square over query points."
            )
        if len(output_dims) != 1:
            raise ValueError("Scalar GP output requires one output dimension.")
        self.query_points = points
        self.mean = mean_array
        self.covariance = covariance_array
        self.variance = variance_array
        self.output_dims = tuple(output_dims)

    def sample(self, key: Array, /, *, num_samples: int) -> Array:
        """Draw coherent latent-discrepancy functions at all query points."""
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        return _sample_gaussian_psd(
            self.mean,
            self.covariance,
            key,
            num_samples=count,
        )

    def predictive_field(
        self,
        base_mean: ArrayLike,
        key: Array,
        /,
        *,
        num_samples: int,
        observation_variance: ArrayLike | None = None,
        sample_dim: str = "__phydra_uq_discrepancy",
    ) -> PredictiveField:
        """Add discrepancy draws to a physical mean and preserve observation variance."""
        base = _as_vector(base_mean, name="physical predictive mean")
        if base.shape != self.mean.shape:
            raise ValueError("base_mean must align with conditioned GP query points.")
        data = base + self.sample(key, num_samples=num_samples)
        valid_data = jnp.all(jnp.isfinite(data), axis=1)
        conditional = None
        if observation_variance is not None:
            variance = jnp.asarray(observation_variance, dtype=float)
            if bool(jnp.any(variance < 0.0)):
                raise ValueError("observation_variance must be non-negative.")
            variance = jnp.broadcast_to(variance, self.mean.shape)
            conditional = cx.Field(variance, dims=self.output_dims)
        return PredictiveField(
            cx.Field(data, dims=(sample_dim, *self.output_dims)),
            (SampleAxis(sample_dim, "epistemic"),),
            conditional_variance=conditional,
            valid=cx.Field(valid_data, dims=(sample_dim,)),
        )


class GaussianProcessConditioner(StrictModule):
    """Reusable residual projection and covariance for one fixed query design."""

    query_points: Array
    residual_projection: Array
    covariance: Array
    variance: Array
    output_dims: tuple[str | None, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        query_points: ArrayLike,
        residual_projection: ArrayLike,
        covariance: ArrayLike,
        variance: ArrayLike,
        output_dims: tuple[str | None, ...],
    ):
        points = _as_design(query_points)
        projection = jnp.asarray(residual_projection, dtype=float)
        covariance_array = jnp.asarray(covariance, dtype=float)
        variance_array = _as_vector(variance, name="conditioned GP variance")
        count = int(points.shape[0])
        if projection.ndim != 2 or int(projection.shape[0]) != count:
            raise ValueError("GP residual projection must have one row per query point.")
        if covariance_array.shape != (count, count) or variance_array.shape != (count,):
            raise ValueError("Conditioner moments must align with query points.")
        if len(output_dims) != 1:
            raise ValueError("Scalar GP output requires one output dimension.")
        self.query_points = points
        self.residual_projection = projection
        self.covariance = covariance_array
        self.variance = variance_array
        self.output_dims = tuple(output_dims)

    def condition(self, residual: ArrayLike, /) -> GaussianProcessCondition:
        """Project a new residual vector without rebuilding any GP factors."""
        values = _as_vector(residual, name="GP residual")
        if int(values.shape[0]) != int(self.residual_projection.shape[1]):
            raise ValueError("GP residual must align with conditioner observations.")
        return GaussianProcessCondition(
            query_points=self.query_points,
            mean=self.residual_projection @ values,
            covariance=self.covariance,
            variance=self.variance,
            output_dims=self.output_dims,
        )


def _sample_gaussian_psd(
    mean: Array,
    covariance: Array,
    key: Array,
    /,
    *,
    num_samples: int,
) -> Array:
    """Sample a numerically semidefinite covariance without artificial noise."""
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    factor = eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, 0.0))[None, :]
    noise = jr.normal(key, (num_samples, mean.size), dtype=mean.dtype)
    return mean + noise @ factor.T


def _as_design(value: ArrayLike) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim < 2:
        raise ValueError(
            "GP inputs must have one design axis and at least one input axis."
        )
    if any(int(size) <= 0 for size in array.shape[1:]):
        raise ValueError("GP kernel input axes must be nonempty.")
    return array


def _as_vector(value: ArrayLike, /, *, name: str) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


__all__ = ["GaussianProcessCondition", "GaussianProcessConditioner"]
