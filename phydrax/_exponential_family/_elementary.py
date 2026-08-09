#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ._contracts import (
    _AbstractAnalyticExponentialFamily,
    _mean_domain_result,
    _natural_domain_result,
    ExponentialFamilyDomainResult,
    ExponentialFamilySignature,
    StatisticBatch,
)


_BERNOULLI_SIGNATURE = ExponentialFamilySignature(
    "bernoulli",
    1,
    (),
    "counting",
    "binary",
    "scalar-log-odds",
)
_POISSON_SIGNATURE = ExponentialFamilySignature(
    "poisson",
    1,
    (),
    "counting",
    "nonnegative-integers",
    "scalar-log-rate",
)
_EXPONENTIAL_RATE_SIGNATURE = ExponentialFamilySignature(
    "exponential-rate",
    1,
    (),
    "lebesgue",
    "nonnegative-real",
    "negative-rate",
)
_NORMAL_SIGNATURE = ExponentialFamilySignature(
    "normal",
    2,
    (),
    "lebesgue",
    "real-line",
    "linear-quadratic",
)


def _real_observation(value: ArrayLike, /) -> Array:
    raw = jnp.asarray(value)
    if jnp.issubdtype(raw.dtype, jnp.complexfloating):
        raise TypeError("Exponential-family observations must be real-valued.")
    return raw.astype(jnp.result_type(raw, 0.0))


class BernoulliFamily(_AbstractAnalyticExponentialFamily):
    """Bernoulli laws in scalar log-odds coordinates."""

    @property
    def signature(self) -> ExponentialFamilySignature:
        return _BERNOULLI_SIGNATURE

    def _natural_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        shape = values.shape[:-1]
        return _natural_domain_result(
            self.signature,
            values,
            interior=jnp.ones(shape, dtype=bool),
            boundary=jnp.zeros(shape, dtype=bool),
        )

    def _mean_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        probability = values[..., 0]
        return _mean_domain_result(
            self.signature,
            values,
            interior=(probability > 0.0) & (probability < 1.0),
            boundary=(probability == 0.0) | (probability == 1.0),
        )

    def _sufficient_statistics(self, value: ArrayLike, /) -> StatisticBatch:
        observation = _real_observation(value)
        valid = jnp.isfinite(observation) & ((observation == 0.0) | (observation == 1.0))
        safe = jnp.where(valid, observation, 0.0)
        return StatisticBatch(safe[..., None], valid, self.signature)

    def _log_base_density(self, value: ArrayLike, /) -> Array:
        return jnp.zeros_like(_real_observation(value))

    def _log_normalizer(self, natural_values: Array, /) -> Array:
        return jax.nn.softplus(natural_values[..., 0])

    def _mean_values(self, natural_values: Array, /) -> Array:
        return jax.nn.sigmoid(natural_values)

    def _natural_from_mean_values(self, mean_values: Array, /) -> Array:
        probability = mean_values[..., 0]
        safe = jnp.where((probability > 0.0) & (probability < 1.0), probability, 0.5)
        return (jnp.log(safe) - jnp.log1p(-safe))[..., None]

    def _sample(
        self,
        key,
        natural_values: Array,
        sample_shape: tuple[int, ...],
        /,
    ) -> Array:
        probability = jax.nn.sigmoid(natural_values[..., 0])
        return jr.bernoulli(
            key,
            probability,
            shape=sample_shape + probability.shape,
        )


class PoissonFamily(_AbstractAnalyticExponentialFamily):
    """Poisson laws in scalar log-rate coordinates."""

    @property
    def signature(self) -> ExponentialFamilySignature:
        return _POISSON_SIGNATURE

    def _natural_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        shape = values.shape[:-1]
        return _natural_domain_result(
            self.signature,
            values,
            interior=jnp.ones(shape, dtype=bool),
            boundary=jnp.zeros(shape, dtype=bool),
        )

    def _mean_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        rate = values[..., 0]
        return _mean_domain_result(
            self.signature,
            values,
            interior=rate > 0.0,
            boundary=rate == 0.0,
        )

    def _sufficient_statistics(self, value: ArrayLike, /) -> StatisticBatch:
        observation = _real_observation(value)
        valid = (
            jnp.isfinite(observation)
            & (observation >= 0.0)
            & (observation == jnp.floor(observation))
        )
        safe = jnp.where(valid, observation, 0.0)
        return StatisticBatch(safe[..., None], valid, self.signature)

    def _log_base_density(self, value: ArrayLike, /) -> Array:
        statistics = self._sufficient_statistics(value)
        return -jsp.special.gammaln(statistics.values[..., 0] + 1.0)

    def _log_normalizer(self, natural_values: Array, /) -> Array:
        return jnp.exp(natural_values[..., 0])

    def _mean_values(self, natural_values: Array, /) -> Array:
        return jnp.exp(natural_values)

    def _natural_from_mean_values(self, mean_values: Array, /) -> Array:
        rate = mean_values[..., 0]
        safe = jnp.where(rate > 0.0, rate, 1.0)
        return jnp.log(safe)[..., None]

    def _sample(
        self,
        key,
        natural_values: Array,
        sample_shape: tuple[int, ...],
        /,
    ) -> Array:
        rate = jnp.exp(natural_values[..., 0])
        return jr.poisson(key, rate, shape=sample_shape + rate.shape)


class ExponentialRateFamily(_AbstractAnalyticExponentialFamily):
    """Exponential laws in the negative-rate natural coordinate."""

    @property
    def signature(self) -> ExponentialFamilySignature:
        return _EXPONENTIAL_RATE_SIGNATURE

    def _natural_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        rate_coordinate = values[..., 0]
        return _natural_domain_result(
            self.signature,
            values,
            interior=rate_coordinate < 0.0,
            boundary=rate_coordinate == 0.0,
        )

    def _mean_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        mean = values[..., 0]
        return _mean_domain_result(
            self.signature,
            values,
            interior=mean > 0.0,
            boundary=mean == 0.0,
        )

    def _sufficient_statistics(self, value: ArrayLike, /) -> StatisticBatch:
        observation = _real_observation(value)
        valid = jnp.isfinite(observation) & (observation >= 0.0)
        safe = jnp.where(valid, observation, 0.0)
        return StatisticBatch(safe[..., None], valid, self.signature)

    def _log_base_density(self, value: ArrayLike, /) -> Array:
        return jnp.zeros_like(_real_observation(value))

    def _log_normalizer(self, natural_values: Array, /) -> Array:
        return -jnp.log(-natural_values[..., 0])

    def _mean_values(self, natural_values: Array, /) -> Array:
        return -1.0 / natural_values

    def _natural_from_mean_values(self, mean_values: Array, /) -> Array:
        mean = mean_values[..., 0]
        safe = jnp.where(mean > 0.0, mean, 1.0)
        return (-1.0 / safe)[..., None]

    def _sample(
        self,
        key,
        natural_values: Array,
        sample_shape: tuple[int, ...],
        /,
    ) -> Array:
        rate = -natural_values[..., 0]
        standard = jr.exponential(
            key,
            shape=sample_shape + rate.shape,
            dtype=natural_values.dtype,
        )
        return standard / rate


class NormalFamily(_AbstractAnalyticExponentialFamily):
    """Univariate Normal laws in linear-quadratic natural coordinates."""

    @property
    def signature(self) -> ExponentialFamilySignature:
        return _NORMAL_SIGNATURE

    def _natural_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        quadratic = values[..., 1]
        return _natural_domain_result(
            self.signature,
            values,
            interior=quadratic < 0.0,
            boundary=quadratic == 0.0,
        )

    def _mean_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        first = values[..., 0]
        second = values[..., 1]
        variance = second - first * first
        scale = jnp.maximum(jnp.maximum(jnp.abs(second), first * first), 1.0)
        tolerance = 64.0 * jnp.finfo(values.dtype).eps * scale
        return _mean_domain_result(
            self.signature,
            values,
            interior=variance > tolerance,
            boundary=jnp.abs(variance) <= tolerance,
        )

    def _sufficient_statistics(self, value: ArrayLike, /) -> StatisticBatch:
        observation = _real_observation(value)
        valid = jnp.isfinite(observation)
        safe = jnp.where(valid, observation, 0.0)
        return StatisticBatch(
            jnp.stack((safe, safe * safe), axis=-1),
            valid,
            self.signature,
        )

    def _log_base_density(self, value: ArrayLike, /) -> Array:
        return jnp.zeros_like(_real_observation(value))

    def _log_normalizer(self, natural_values: Array, /) -> Array:
        linear = natural_values[..., 0]
        quadratic = natural_values[..., 1]
        return -(linear * linear) / (4.0 * quadratic) + 0.5 * jnp.log(-jnp.pi / quadratic)

    def _mean_values(self, natural_values: Array, /) -> Array:
        linear = natural_values[..., 0]
        quadratic = natural_values[..., 1]
        location = -linear / (2.0 * quadratic)
        variance = -1.0 / (2.0 * quadratic)
        return jnp.stack((location, location * location + variance), axis=-1)

    def _natural_from_mean_values(self, mean_values: Array, /) -> Array:
        location = mean_values[..., 0]
        variance = mean_values[..., 1] - location * location
        safe_variance = jnp.where(variance > 0.0, variance, 1.0)
        return jnp.stack(
            (location / safe_variance, -0.5 / safe_variance),
            axis=-1,
        )

    def _sample(
        self,
        key,
        natural_values: Array,
        sample_shape: tuple[int, ...],
        /,
    ) -> Array:
        linear = natural_values[..., 0]
        quadratic = natural_values[..., 1]
        location = -linear / (2.0 * quadratic)
        scale = jnp.sqrt(-1.0 / (2.0 * quadratic))
        noise = jr.normal(
            key,
            shape=sample_shape + location.shape,
            dtype=natural_values.dtype,
        )
        return location + scale * noise


__all__ = [
    "BernoulliFamily",
    "ExponentialRateFamily",
    "NormalFamily",
    "PoissonFamily",
]
