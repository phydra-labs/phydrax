#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ._distributions import _open_unit_interval, AbstractDistribution


class PowerLaw(AbstractDistribution):
    """Normalized finite-interval density proportional to x**alpha."""

    alpha: Array
    low: Array
    high: Array

    def __init__(self, alpha: ArrayLike, low: ArrayLike, high: ArrayLike):
        exponent = jnp.asarray(alpha, dtype=float).reshape(())
        lower = jnp.asarray(low, dtype=float).reshape(())
        upper = jnp.asarray(high, dtype=float).reshape(())
        if not bool(jnp.all(jnp.isfinite(jnp.stack((exponent, lower, upper))))):
            raise ValueError("Power-law parameters must be finite.")
        if not bool((lower > 0.0) & (lower < upper)):
            raise ValueError("Power-law support must satisfy 0 < low < high.")
        self.alpha = exponent
        self.low = lower
        self.high = upper

    def _power_integral(self, order: int) -> Array:
        exponent = self.alpha + float(order) + 1.0
        logarithmic = jnp.log(self.high / self.low)
        is_logarithmic = exponent == 0.0
        safe_exponent = jnp.where(is_logarithmic, 1.0, exponent)
        ordinary = self.low**exponent * jnp.expm1(exponent * logarithmic) / safe_exponent
        return jnp.where(is_logarithmic, logarithmic, ordinary)

    @property
    def _normalizer(self) -> Array:
        return self._power_integral(0)

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.icdf(jr.uniform(key, tuple(sample_shape), dtype=self.low.dtype))

    def icdf(self, value: ArrayLike, /) -> Array:
        probability = _open_unit_interval(value)
        exponent = self.alpha + 1.0
        logarithmic = jnp.log(self.high / self.low)
        is_logarithmic = exponent == 0.0
        safe_exponent = jnp.where(is_logarithmic, 1.0, exponent)
        log_scale = (
            jnp.log1p(probability * jnp.expm1(exponent * logarithmic)) / safe_exponent
        )
        ordinary = self.low * jnp.exp(log_scale)
        logarithmic_value = self.low * jnp.exp(probability * logarithmic)
        return jnp.where(is_logarithmic, logarithmic_value, ordinary)

    def log_prob(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value, dtype=float)
        density = self.alpha * jnp.log(jnp.where(values > 0.0, values, 1.0)) - jnp.log(
            self._normalizer
        )
        return jnp.where(self.contains(values), density, -jnp.inf)

    @property
    def mean(self) -> Array:
        return self._power_integral(1) / self._normalizer

    @property
    def variance(self) -> Array:
        second = self._power_integral(2) / self._normalizer
        return second - self.mean**2

    @property
    def support(self) -> tuple[Array, Array]:
        return self.low, self.high

    def contains(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        return jnp.isfinite(values) & (values >= self.low) & (values <= self.high)


class TruncatedNormal(AbstractDistribution):
    """Normal law conditioned on one nonempty finite or infinite interval."""

    location: Array
    scale: Array
    low: Array
    high: Array
    lower_cdf: Array
    mass: Array

    def __init__(
        self,
        location: ArrayLike,
        scale: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
    ):
        location_ = jnp.asarray(location, dtype=float).reshape(())
        scale_ = jnp.asarray(scale, dtype=float).reshape(())
        low_ = jnp.asarray(low, dtype=float).reshape(())
        high_ = jnp.asarray(high, dtype=float).reshape(())
        if not bool(jnp.isfinite(location_) & jnp.isfinite(scale_) & (scale_ > 0.0)):
            raise ValueError(
                "Truncated-normal location and scale must be finite and positive."
            )
        if bool(jnp.isnan(low_) | jnp.isnan(high_) | (low_ >= high_)):
            raise ValueError("Truncated-normal support must satisfy low < high.")
        lower = jsp.special.ndtr((low_ - location_) / scale_)
        upper = jsp.special.ndtr((high_ - location_) / scale_)
        mass = upper - lower
        if not bool(jnp.isfinite(mass) & (mass > 0.0)):
            raise ValueError(
                "Truncated-normal support has no representable probability mass."
            )
        self.location = location_
        self.scale = scale_
        self.low = low_
        self.high = high_
        self.lower_cdf = lower
        self.mass = mass

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.icdf(jr.uniform(key, tuple(sample_shape), dtype=self.location.dtype))

    def icdf(self, value: ArrayLike, /) -> Array:
        probability = _open_unit_interval(value)
        normal_probability = self.lower_cdf + probability * self.mass
        return self.location + self.scale * jsp.special.ndtri(normal_probability)

    def log_prob(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value, dtype=float)
        standardized = (values - self.location) / self.scale
        density = (
            -0.5 * standardized**2
            - jnp.log(self.scale)
            - 0.5 * jnp.log(2.0 * jnp.pi)
            - jnp.log(self.mass)
        )
        return jnp.where(self.contains(values), density, -jnp.inf)

    @property
    def mean(self) -> Array:
        a = (self.low - self.location) / self.scale
        b = (self.high - self.location) / self.scale
        phi_a = jnp.where(
            jnp.isfinite(a), jnp.exp(-0.5 * a * a) / jnp.sqrt(2.0 * jnp.pi), 0.0
        )
        phi_b = jnp.where(
            jnp.isfinite(b), jnp.exp(-0.5 * b * b) / jnp.sqrt(2.0 * jnp.pi), 0.0
        )
        return self.location + self.scale * (phi_a - phi_b) / self.mass

    @property
    def variance(self) -> Array:
        a = (self.low - self.location) / self.scale
        b = (self.high - self.location) / self.scale
        phi_a = jnp.where(
            jnp.isfinite(a), jnp.exp(-0.5 * a * a) / jnp.sqrt(2.0 * jnp.pi), 0.0
        )
        phi_b = jnp.where(
            jnp.isfinite(b), jnp.exp(-0.5 * b * b) / jnp.sqrt(2.0 * jnp.pi), 0.0
        )
        mean_shift = (phi_a - phi_b) / self.mass
        safe_a = jnp.where(jnp.isfinite(a), a, 0.0)
        safe_b = jnp.where(jnp.isfinite(b), b, 0.0)
        boundary = (safe_a * phi_a - safe_b * phi_b) / self.mass
        return self.scale**2 * (1.0 + boundary - mean_shift**2)

    @property
    def support(self) -> tuple[Array, Array]:
        return self.low, self.high

    def contains(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        return jnp.isfinite(values) & (values >= self.low) & (values <= self.high)


class HalfNormal(AbstractDistribution):
    scale: Array

    def __init__(self, scale: ArrayLike):
        scale_ = jnp.asarray(scale, dtype=float).reshape(())
        if not bool(jnp.isfinite(scale_) & (scale_ > 0.0)):
            raise ValueError("Half-normal scale must be finite and positive.")
        self.scale = scale_

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return (
            jnp.abs(jr.normal(key, tuple(sample_shape), dtype=self.scale.dtype))
            * self.scale
        )

    def icdf(self, value: ArrayLike, /) -> Array:
        probability = _open_unit_interval(value)
        return self.scale * jnp.sqrt(2.0) * jsp.special.erfinv(probability)

    def log_prob(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value, dtype=float)
        density = (
            -0.5 * (values / self.scale) ** 2
            + 0.5 * jnp.log(2.0 / jnp.pi)
            - jnp.log(self.scale)
        )
        return jnp.where(self.contains(values), density, -jnp.inf)

    @property
    def mean(self) -> Array:
        return self.scale * jnp.sqrt(2.0 / jnp.pi)

    @property
    def variance(self) -> Array:
        return self.scale**2 * (1.0 - 2.0 / jnp.pi)

    @property
    def support(self) -> tuple[Array, Array]:
        return jnp.asarray(0.0), jnp.asarray(jnp.inf)

    def contains(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        return jnp.isfinite(values) & (values >= 0.0)


class Cauchy(AbstractDistribution):
    location: Array
    scale: Array

    def __init__(self, location: ArrayLike, scale: ArrayLike):
        location_ = jnp.asarray(location, dtype=float).reshape(())
        scale_ = jnp.asarray(scale, dtype=float).reshape(())
        if not bool(jnp.isfinite(location_) & jnp.isfinite(scale_) & (scale_ > 0.0)):
            raise ValueError("Cauchy location and scale must be finite and positive.")
        self.location = location_
        self.scale = scale_

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.icdf(jr.uniform(key, tuple(sample_shape), dtype=self.location.dtype))

    def icdf(self, value: ArrayLike, /) -> Array:
        probability = _open_unit_interval(value)
        return self.location + self.scale * jnp.tan(jnp.pi * (probability - 0.5))

    def log_prob(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value, dtype=float)
        standardized = (values - self.location) / self.scale
        density = -jnp.log(jnp.pi * self.scale) - jnp.log1p(standardized**2)
        return jnp.where(self.contains(values), density, -jnp.inf)

    @property
    def mean(self) -> Array:
        return jnp.asarray(jnp.nan, dtype=self.location.dtype)

    @property
    def variance(self) -> Array:
        return jnp.asarray(jnp.inf, dtype=self.location.dtype)

    @property
    def support(self) -> None:
        return None

    def contains(self, value: ArrayLike, /) -> Array:
        return jnp.isfinite(jnp.asarray(value))


class StudentT(AbstractDistribution):
    degrees_of_freedom: Array
    location: Array
    scale: Array

    def __init__(
        self,
        degrees_of_freedom: ArrayLike,
        location: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
    ):
        df = jnp.asarray(degrees_of_freedom, dtype=float).reshape(())
        location_ = jnp.asarray(location, dtype=float).reshape(())
        scale_ = jnp.asarray(scale, dtype=float).reshape(())
        if not bool(
            jnp.all(jnp.isfinite(jnp.stack((df, location_, scale_))))
            & (df > 0.0)
            & (scale_ > 0.0)
        ):
            raise ValueError(
                "Student-t degrees of freedom and scale must be positive and finite."
            )
        self.degrees_of_freedom = df
        self.location = location_
        self.scale = scale_

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.location + self.scale * jr.t(
            key,
            self.degrees_of_freedom,
            shape=tuple(sample_shape),
            dtype=self.location.dtype,
        )

    def _standard_cdf(self, value: Array) -> Array:
        ratio = self.degrees_of_freedom / (self.degrees_of_freedom + value * value)
        tail = 0.5 * jsp.special.betainc(0.5 * self.degrees_of_freedom, 0.5, ratio)
        return jnp.where(value >= 0.0, 1.0 - tail, tail)

    def icdf(self, value: ArrayLike, /) -> Array:
        probability = _open_unit_interval(value)

        def body(_, bounds):
            lower, upper = bounds
            middle = 0.5 * (lower + upper)
            candidate = jnp.tan(middle)
            left = self._standard_cdf(candidate) < probability
            return jnp.where(left, middle, lower), jnp.where(left, upper, middle)

        lower = jnp.full_like(
            probability, -0.5 * jnp.pi + jnp.finfo(probability.dtype).eps
        )
        upper = jnp.full_like(
            probability, 0.5 * jnp.pi - jnp.finfo(probability.dtype).eps
        )
        lower, upper = jax.lax.fori_loop(0, 80, body, (lower, upper))
        return self.location + self.scale * jnp.tan(0.5 * (lower + upper))

    def log_prob(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value, dtype=float)
        standardized = (values - self.location) / self.scale
        half = 0.5 * (self.degrees_of_freedom + 1.0)
        density = (
            jsp.special.gammaln(half)
            - jsp.special.gammaln(0.5 * self.degrees_of_freedom)
            - 0.5 * jnp.log(self.degrees_of_freedom * jnp.pi)
            - jnp.log(self.scale)
            - half * jnp.log1p(standardized**2 / self.degrees_of_freedom)
        )
        return jnp.where(self.contains(values), density, -jnp.inf)

    @property
    def mean(self) -> Array:
        return jnp.where(self.degrees_of_freedom > 1.0, self.location, jnp.nan)

    @property
    def variance(self) -> Array:
        df = self.degrees_of_freedom
        return jnp.where(
            df > 2.0,
            self.scale**2 * df / (df - 2.0),
            jnp.where(df > 1.0, jnp.inf, jnp.nan),
        )

    @property
    def support(self) -> None:
        return None

    def contains(self, value: ArrayLike, /) -> Array:
        return jnp.isfinite(jnp.asarray(value))


class SineAngle(AbstractDistribution):
    """Polar angle on [0, pi] induced by isotropic spherical area."""

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.icdf(jr.uniform(key, tuple(sample_shape)))

    def icdf(self, value: ArrayLike, /) -> Array:
        return jnp.arccos(1.0 - 2.0 * jnp.clip(jnp.asarray(value), 0.0, 1.0))

    def log_prob(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value, dtype=float)
        return jnp.where(
            self.contains(values), jnp.log(jnp.sin(values)) - jnp.log(2.0), -jnp.inf
        )

    @property
    def mean(self) -> Array:
        return jnp.asarray(0.5 * jnp.pi)

    @property
    def variance(self) -> Array:
        return jnp.asarray(0.25 * (jnp.pi**2 - 8.0))

    @property
    def support(self) -> tuple[Array, Array]:
        return jnp.asarray(0.0), jnp.asarray(jnp.pi)

    def contains(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        return jnp.isfinite(values) & (values >= 0.0) & (values <= jnp.pi)


class CosineAngle(AbstractDistribution):
    """Latitude-like angle on [-pi/2, pi/2] induced by spherical area."""

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.icdf(jr.uniform(key, tuple(sample_shape)))

    def icdf(self, value: ArrayLike, /) -> Array:
        return jnp.arcsin(2.0 * jnp.clip(jnp.asarray(value), 0.0, 1.0) - 1.0)

    def log_prob(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value, dtype=float)
        return jnp.where(
            self.contains(values), jnp.log(jnp.cos(values)) - jnp.log(2.0), -jnp.inf
        )

    @property
    def mean(self) -> Array:
        return jnp.asarray(0.0)

    @property
    def variance(self) -> Array:
        return jnp.asarray(0.25 * (jnp.pi**2 - 8.0))

    @property
    def support(self) -> tuple[Array, Array]:
        return jnp.asarray(-0.5 * jnp.pi), jnp.asarray(0.5 * jnp.pi)

    def contains(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        return jnp.isfinite(values) & (values >= -0.5 * jnp.pi) & (values <= 0.5 * jnp.pi)


__all__ = [
    "Cauchy",
    "CosineAngle",
    "HalfNormal",
    "PowerLaw",
    "SineAngle",
    "StudentT",
    "TruncatedNormal",
]
