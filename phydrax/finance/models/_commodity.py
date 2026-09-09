#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measure-neutral commodity factor and seasonality records."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._dependence import CorrelationMatrix


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(
        array,
        ~jnp.isfinite(array) | (array <= 0.0),
        f"{name} must be finite and positive.",
    )


class CommoditySeasonality(StrictModule):
    """Positive periodic multiplicative factors on an ordered phase grid."""

    phases: Array
    factors: Array
    period: Array
    node_count: int = eqx.field(static=True)

    def __init__(
        self, phases: ArrayLike, factors: ArrayLike, /, *, period: ArrayLike = 1.0
    ):
        phases_ = jnp.asarray(phases, dtype=float)
        factors_ = jnp.asarray(factors, dtype=float)
        if phases_.ndim != 1 or phases_.size < 2 or factors_.shape != phases_.shape:
            raise ValueError(
                "phases and factors must be aligned vectors with at least two nodes."
            )
        period_ = _positive_scalar(period, "period")
        phases_ = eqx.error_if(
            phases_,
            jnp.any(~jnp.isfinite(phases_))
            | jnp.any(jnp.diff(phases_) <= 0.0)
            | ~jnp.isclose(phases_[0], 0.0)
            | (phases_[-1] >= period_),
            "phases must be finite, begin at zero, increase, and remain below period.",
        )
        factors_ = eqx.error_if(
            factors_,
            jnp.any(~jnp.isfinite(factors_)) | jnp.any(factors_ <= 0.0),
            "seasonality factors must be finite and positive.",
        )
        self.phases = phases_
        self.factors = factors_
        self.period = period_
        self.node_count = int(phases_.size)

    def factor(self, time: ArrayLike, /) -> Array:
        phase = jnp.mod(jnp.asarray(time, dtype=float), self.period)
        wrapped_phases = jnp.concatenate((self.phases, self.phases[:1] + self.period))
        wrapped_factors = jnp.concatenate((self.factors, self.factors[:1]))
        return jnp.interp(phase, wrapped_phases, wrapped_factors)


class SchwartzOneFactorModel(StrictModule):
    """Mean-reverting log-spot factor without a measure-specific level."""

    mean_reversion: Array
    volatility: Array
    initial_factor: Array

    def __init__(
        self,
        mean_reversion: ArrayLike,
        volatility: ArrayLike,
        initial_factor: ArrayLike,
        /,
    ):
        initial = jnp.asarray(initial_factor, dtype=float)
        if initial.shape != ():
            raise ValueError("initial_factor must be scalar.")
        self.mean_reversion = _positive_scalar(mean_reversion, "mean_reversion")
        self.volatility = _positive_scalar(volatility, "volatility")
        self.initial_factor = eqx.error_if(
            initial, ~jnp.isfinite(initial), "initial_factor must be finite."
        )

    def factor_variance(self, horizon: ArrayLike, /) -> Array:
        horizon_ = jnp.asarray(horizon, dtype=float)
        horizon_ = eqx.error_if(
            horizon_,
            jnp.any(~jnp.isfinite(horizon_)) | jnp.any(horizon_ < 0.0),
            "horizon must be finite and non-negative.",
        )
        return (
            self.volatility**2
            * (-jnp.expm1(-2.0 * self.mean_reversion * horizon_))
            / (2.0 * self.mean_reversion)
        )


class SchwartzTwoFactorModel(StrictModule):
    """Mean-reverting short factor and Brownian long factor."""

    short_mean_reversion: Array
    short_volatility: Array
    long_volatility: Array
    correlation: Array
    initial_factors: Array

    def __init__(
        self,
        short_mean_reversion: ArrayLike,
        short_volatility: ArrayLike,
        long_volatility: ArrayLike,
        correlation: ArrayLike,
        initial_factors: ArrayLike,
        /,
    ):
        correlation_ = jnp.asarray(correlation, dtype=float)
        factors = jnp.asarray(initial_factors, dtype=float)
        if correlation_.shape != () or factors.shape != (2,):
            raise ValueError(
                "correlation must be scalar and initial_factors must have shape (2,)."
            )
        correlation_ = eqx.error_if(
            correlation_,
            ~jnp.isfinite(correlation_) | (jnp.abs(correlation_) >= 1.0),
            "correlation must lie strictly in (-1, 1).",
        )
        factors = eqx.error_if(
            factors, jnp.any(~jnp.isfinite(factors)), "initial_factors must be finite."
        )
        self.short_mean_reversion = _positive_scalar(
            short_mean_reversion, "short_mean_reversion"
        )
        self.short_volatility = _positive_scalar(short_volatility, "short_volatility")
        self.long_volatility = _positive_scalar(long_volatility, "long_volatility")
        self.correlation = correlation_
        self.initial_factors = factors

    def log_spot_variance(self, horizon: ArrayLike, /) -> Array:
        t = jnp.asarray(horizon, dtype=float)
        t = eqx.error_if(
            t,
            jnp.any(~jnp.isfinite(t)) | jnp.any(t < 0.0),
            "horizon must be finite and non-negative.",
        )
        kappa = self.short_mean_reversion
        short = self.short_volatility**2 * (-jnp.expm1(-2.0 * kappa * t)) / (2.0 * kappa)
        long = self.long_volatility**2 * t
        cross = (
            2.0
            * self.correlation
            * self.short_volatility
            * self.long_volatility
            * (-jnp.expm1(-kappa * t))
            / kappa
        )
        return short + long + cross


class CommodityFactorModel(StrictModule):
    """General correlated mean-reverting Gaussian commodity factors."""

    mean_reversions: Array
    volatilities: Array
    initial_factors: Array
    spot_loadings: Array
    dependence: CorrelationMatrix
    factor_count: int = eqx.field(static=True)

    def __init__(
        self,
        mean_reversions: ArrayLike,
        volatilities: ArrayLike,
        initial_factors: ArrayLike,
        spot_loadings: ArrayLike,
        dependence: CorrelationMatrix,
        /,
    ):
        if not isinstance(dependence, CorrelationMatrix):
            raise TypeError("dependence must be a CorrelationMatrix.")
        count = dependence.dimension
        arrays = tuple(
            jnp.asarray(value, dtype=float)
            for value in (mean_reversions, volatilities, initial_factors, spot_loadings)
        )
        if any(value.shape != (count,) for value in arrays):
            raise ValueError(
                "all commodity factor vectors must match dependence dimension."
            )
        mean, volatility, initial, loadings = arrays
        mean = eqx.error_if(
            mean,
            jnp.any(~jnp.isfinite(mean)) | jnp.any(mean < 0.0),
            "mean_reversions must be finite and non-negative.",
        )
        volatility = eqx.error_if(
            volatility,
            jnp.any(~jnp.isfinite(volatility)) | jnp.any(volatility <= 0.0),
            "volatilities must be finite and positive.",
        )
        initial = eqx.error_if(
            initial, jnp.any(~jnp.isfinite(initial)), "initial_factors must be finite."
        )
        loadings = eqx.error_if(
            loadings, jnp.any(~jnp.isfinite(loadings)), "spot_loadings must be finite."
        )
        self.mean_reversions = mean
        self.volatilities = volatility
        self.initial_factors = initial
        self.spot_loadings = loadings
        self.dependence = dependence
        self.factor_count = count

    def instantaneous_covariance(self) -> Array:
        return (
            self.volatilities[:, None]
            * self.dependence.matrix
            * self.volatilities[None, :]
        )


__all__ = [
    "CommodityFactorModel",
    "CommoditySeasonality",
    "SchwartzOneFactorModel",
    "SchwartzTwoFactorModel",
]
