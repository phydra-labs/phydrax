#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


def _nonnegative_scalar(value: Any, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    return eqx.error_if(
        scalar,
        ~jnp.isfinite(scalar) | (scalar < 0.0),
        f"{name} must be finite and nonnegative.",
    )


class StreamingGaussianMoments(StrictModule):
    """Immutable weighted Gaussian moments supporting stable update and merge."""

    mass: Array
    squared_mass: Array
    mean: Array
    scatter: Array
    updates: Array

    def __init__(
        self,
        mass: ArrayLike,
        squared_mass: ArrayLike,
        mean: ArrayLike,
        scatter: ArrayLike,
        updates: ArrayLike = 0,
    ):
        mean_ = jnp.asarray(mean)
        scatter_ = jnp.asarray(scatter)
        if mean_.ndim < 1 or scatter_.shape != mean_.shape[:-1] + (
            mean_.shape[-1],
            mean_.shape[-1],
        ):
            raise ValueError("scatter must have shape case + (feature, feature).")
        self.mass = jnp.asarray(mass, dtype=mean_.real.dtype)
        self.squared_mass = jnp.asarray(squared_mass, dtype=mean_.real.dtype)
        self.mean = mean_
        self.scatter = scatter_
        self.updates = jnp.asarray(updates, dtype=jnp.int32)

    @classmethod
    def initialize(
        cls,
        feature_count: int,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        dtype: Any = jnp.float32,
    ) -> "StreamingGaussianMoments":
        if feature_count <= 0:
            raise ValueError("feature_count must be positive.")
        value_dtype = jnp.dtype(dtype)
        real_dtype = jnp.empty((), dtype=value_dtype).real.dtype
        return cls(
            jnp.zeros(case_shape, dtype=real_dtype),
            jnp.zeros(case_shape, dtype=real_dtype),
            jnp.zeros(case_shape + (feature_count,), dtype=value_dtype),
            jnp.zeros(case_shape + (feature_count, feature_count), dtype=value_dtype),
            jnp.zeros(case_shape, dtype=jnp.int32),
        )

    def update(
        self,
        values: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        mask: ArrayLike | None = None,
    ) -> "StreamingGaussianMoments":
        x = jnp.asarray(values, dtype=self.mean.dtype)
        if x.shape[:-2] != self.mean.shape[:-1] or x.shape[-1] != self.mean.shape[-1]:
            raise ValueError("values must have shape case + (sample, feature).")
        real_dtype = self.mass.dtype
        w = (
            jnp.ones(x.shape[:-1], dtype=real_dtype)
            if weights is None
            else jnp.broadcast_to(jnp.asarray(weights, dtype=real_dtype), x.shape[:-1])
        )
        active = jnp.isfinite(w) & (w >= 0.0) & jnp.all(jnp.isfinite(x), axis=-1)
        if mask is not None:
            active &= jnp.broadcast_to(jnp.asarray(mask, dtype=bool), x.shape[:-1])
        w = jnp.where(active, w, 0.0)
        safe_x = jnp.where(active[..., None], x, 0)
        batch_mass = jnp.sum(w, axis=-1)
        batch_squared_mass = jnp.sum(jnp.square(w), axis=-1)
        tiny = jnp.finfo(real_dtype).tiny
        batch_mean = (
            jnp.einsum("...n,...nf->...f", w, safe_x)
            / jnp.maximum(batch_mass, tiny)[..., None]
        )
        centered = jnp.where(active[..., None], safe_x - batch_mean[..., None, :], 0)
        batch_scatter = jnp.einsum(
            "...ni,...n,...nj->...ij", jnp.conj(centered), w, centered
        )
        other = StreamingGaussianMoments(
            batch_mass,
            batch_squared_mass,
            batch_mean,
            batch_scatter,
            jnp.ones_like(self.updates),
        )
        return self.merge(other)

    def merge(self, other: "StreamingGaussianMoments", /) -> "StreamingGaussianMoments":
        if other.mean.shape != self.mean.shape:
            raise ValueError("streaming moment shapes must match.")
        total = self.mass + other.mass
        safe_total = jnp.maximum(total, jnp.finfo(self.mass.dtype).tiny)
        delta = other.mean - self.mean
        mean = self.mean + delta * (other.mass / safe_total)[..., None]
        correction = jnp.einsum("...i,...j->...ij", jnp.conj(delta), delta)
        correction = correction * (self.mass * other.mass / safe_total)[..., None, None]
        scatter = self.scatter + other.scatter + correction
        empty_self = self.mass <= 0.0
        empty_other = other.mass <= 0.0
        mean = jnp.where(empty_self[..., None], other.mean, mean)
        mean = jnp.where(empty_other[..., None], self.mean, mean)
        scatter = jnp.where(empty_self[..., None, None], other.scatter, scatter)
        scatter = jnp.where(empty_other[..., None, None], self.scatter, scatter)
        return StreamingGaussianMoments(
            total,
            self.squared_mass + other.squared_mass,
            mean,
            scatter,
            self.updates + other.updates,
        )

    def covariance(
        self, /, *, correction: float = 0.0, regularization: float = 0.0
    ) -> Array:
        correction_ = _nonnegative_scalar(correction, "correction")
        regularization_ = _nonnegative_scalar(regularization, "regularization")
        denominator = self.mass - correction_ * self.squared_mass / jnp.maximum(
            self.mass, jnp.finfo(self.mass.dtype).tiny
        )
        covariance = (
            self.scatter
            / jnp.maximum(denominator, jnp.finfo(self.mass.dtype).tiny)[..., None, None]
        )
        scale = jnp.maximum(
            jnp.real(jnp.trace(covariance, axis1=-2, axis2=-1)) / self.mean.shape[-1], 1.0
        )
        return covariance + (regularization_ * scale)[..., None, None] * jnp.eye(
            self.mean.shape[-1], dtype=covariance.dtype
        )

    def model(self, /, *, correction: float = 0.0, regularization: float = 1e-6):
        """Materialize the current moments as executable Gaussian covariance geometry."""
        regularization_ = _nonnegative_scalar(regularization, "regularization")
        from ._estimators import _regularize, CovarianceModel

        covariance, precision, log_determinant, _ = _regularize(
            self.covariance(correction=correction), regularization_
        )
        return CovarianceModel(
            self.mean,
            covariance,
            precision,
            log_determinant,
            method="streaming-gaussian-moments",
        )

    @property
    def effective_samples(self) -> Array:
        return jnp.where(
            self.squared_mass > 0.0, self.mass * self.mass / self.squared_mass, 0.0
        )

    @property
    def valid(self) -> Array:
        return (self.mass > 0.0) & jnp.all(jnp.isfinite(self.scatter), axis=(-2, -1))


__all__ = ["StreamingGaussianMoments"]
