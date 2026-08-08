#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ._strict import StrictModule


class AbstractLikelihood(StrictModule):
    """Elementwise observation likelihood protocol."""

    @abstractmethod
    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        raise NotImplementedError


class GaussianLikelihood(AbstractLikelihood):
    """Gaussian observation likelihood with a fixed positive scale."""

    scale: Array

    def __init__(self, scale: ArrayLike):
        scale_array = jnp.asarray(scale, dtype=float)
        if bool(jnp.any(~jnp.isfinite(scale_array))) or bool(jnp.any(scale_array <= 0.0)):
            raise ValueError("Gaussian scale must be finite and strictly positive.")
        self.scale = scale_array

    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        if parameters:
            raise TypeError(
                f"GaussianLikelihood received unknown parameters {tuple(parameters)!r}."
            )
        location_array = jnp.asarray(location, dtype=float)
        target_array = jnp.asarray(target, dtype=float)
        standardized = (target_array - location_array) / self.scale
        return -0.5 * standardized**2 - jnp.log(self.scale) - 0.5 * jnp.log(2.0 * jnp.pi)

    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        if parameters:
            raise TypeError(
                f"GaussianLikelihood received unknown parameters {tuple(parameters)!r}."
            )
        location_array = jnp.asarray(location, dtype=float)
        shape = jnp.broadcast_shapes(location_array.shape, self.scale.shape)
        noise = jr.normal(key, shape=shape, dtype=location_array.dtype)
        return location_array + self.scale * noise


class GaussianLocationScaleLikelihood(AbstractLikelihood):
    """Heteroscedastic Gaussian with a softplus-transformed raw scale."""

    min_scale: float

    def __init__(self, *, min_scale: float = 1e-6):
        minimum = float(min_scale)
        if not jnp.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("min_scale must be finite and strictly positive.")
        self.min_scale = minimum

    def scale_from_raw(self, raw_scale: ArrayLike, /) -> Array:
        return jax_softplus(jnp.asarray(raw_scale, dtype=float)) + self.min_scale

    def log_prob(
        self,
        location: ArrayLike,
        target: ArrayLike,
        /,
        *,
        raw_scale: ArrayLike | None = None,
        **parameters: Any,
    ) -> Array:
        if parameters:
            raise TypeError(
                "GaussianLocationScaleLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        if raw_scale is None:
            raise ValueError("raw_scale is required.")
        scale = self.scale_from_raw(raw_scale)
        location_array = jnp.asarray(location, dtype=float)
        target_array = jnp.asarray(target, dtype=float)
        standardized = (target_array - location_array) / scale
        return -0.5 * standardized**2 - jnp.log(scale) - 0.5 * jnp.log(2.0 * jnp.pi)

    def sample(
        self,
        key,
        location: ArrayLike,
        /,
        *,
        raw_scale: ArrayLike | None = None,
        **parameters: Any,
    ) -> Array:
        if parameters:
            raise TypeError(
                "GaussianLocationScaleLikelihood received unknown parameters "
                f"{tuple(parameters)!r}."
            )
        if raw_scale is None:
            raise ValueError("raw_scale is required.")
        location_array = jnp.asarray(location, dtype=float)
        scale = self.scale_from_raw(raw_scale)
        shape = jnp.broadcast_shapes(location_array.shape, scale.shape)
        return location_array + scale * jr.normal(
            key, shape=shape, dtype=location_array.dtype
        )


class StudentTLikelihood(AbstractLikelihood):
    """Student-t observation likelihood with fixed degrees of freedom and scale."""

    df: Array
    scale: Array

    def __init__(self, df: ArrayLike, scale: ArrayLike):
        df_array = jnp.asarray(df, dtype=float)
        scale_array = jnp.asarray(scale, dtype=float)
        if bool(jnp.any(~jnp.isfinite(df_array))) or bool(jnp.any(df_array <= 0.0)):
            raise ValueError("Student-t degrees of freedom must be finite and positive.")
        if bool(jnp.any(~jnp.isfinite(scale_array))) or bool(jnp.any(scale_array <= 0.0)):
            raise ValueError("Student-t scale must be finite and strictly positive.")
        self.df = df_array
        self.scale = scale_array

    def log_prob(
        self, location: ArrayLike, target: ArrayLike, /, **parameters: Any
    ) -> Array:
        if parameters:
            raise TypeError(
                f"StudentTLikelihood received unknown parameters {tuple(parameters)!r}."
            )
        location_array = jnp.asarray(location, dtype=float)
        target_array = jnp.asarray(target, dtype=float)
        standardized = (target_array - location_array) / self.scale
        normalizer = (
            jsp.special.gammaln((self.df + 1.0) / 2.0)
            - jsp.special.gammaln(self.df / 2.0)
            - 0.5 * jnp.log(self.df * jnp.pi)
            - jnp.log(self.scale)
        )
        return normalizer - 0.5 * (self.df + 1.0) * jnp.log1p(standardized**2 / self.df)

    def sample(self, key, location: ArrayLike, /, **parameters: Any) -> Array:
        if parameters:
            raise TypeError(
                f"StudentTLikelihood received unknown parameters {tuple(parameters)!r}."
            )
        location_array = jnp.asarray(location, dtype=float)
        shape = jnp.broadcast_shapes(
            location_array.shape, self.scale.shape, self.df.shape
        )
        return location_array + self.scale * jr.t(
            key, self.df, shape=shape, dtype=location_array.dtype
        )


def jax_softplus(value: Array, /) -> Array:
    """Stable softplus kept local to avoid a public activation dependency."""
    return jnp.logaddexp(value, jnp.zeros((), dtype=value.dtype))


__all__ = [
    "AbstractLikelihood",
    "GaussianLikelihood",
    "GaussianLocationScaleLikelihood",
    "StudentTLikelihood",
]
