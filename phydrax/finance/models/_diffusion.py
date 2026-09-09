#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measure-neutral diffusion-model parameter records.

These records describe local dynamics only.  Drifts, numeraires, collateral and
measure choices belong to a ``PricingLaw`` and are deliberately not embedded here.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


def _scalar(value: ArrayLike, name: str, /, *, lower: float | None = None) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    invalid = ~jnp.isfinite(array)
    if lower is not None:
        invalid = invalid | (array <= lower)
    return eqx.error_if(
        array, invalid, f"{name} must be finite and greater than {lower}."
    )


def _correlation(value: ArrayLike, name: str = "correlation") -> Array:
    array = _scalar(value, name)
    return eqx.error_if(
        array, jnp.abs(array) >= 1.0, f"{name} must lie strictly in (-1, 1)."
    )


def _ordered_grid(value: ArrayLike, name: str, /, *, positive: bool = False) -> Array:
    grid = jnp.asarray(value, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError(f"{name} must be a rank-one grid with at least two nodes.")
    invalid = jnp.any(~jnp.isfinite(grid)) | jnp.any(jnp.diff(grid) <= 0.0)
    if positive:
        invalid = invalid | jnp.any(grid <= 0.0)
    return eqx.error_if(grid, invalid, f"{name} must be finite and strictly increasing.")


class BlackScholesModel(StrictModule):
    """Constant lognormal diffusion scale, independent of pricing measure."""

    volatility: Array

    def __init__(self, volatility: ArrayLike, /):
        self.volatility = _scalar(volatility, "volatility", lower=0.0)

    def instantaneous_variance(self, spot: ArrayLike | None = None, /) -> Array:
        del spot
        return self.volatility**2


class Black76Model(StrictModule):
    """Constant lognormal forward diffusion scale."""

    volatility: Array

    def __init__(self, volatility: ArrayLike, /):
        self.volatility = _scalar(volatility, "volatility", lower=0.0)

    def instantaneous_variance(self, forward: ArrayLike | None = None, /) -> Array:
        del forward
        return self.volatility**2


class BachelierModel(StrictModule):
    """Constant normal (absolute) diffusion scale."""

    volatility: Array

    def __init__(self, volatility: ArrayLike, /):
        self.volatility = _scalar(volatility, "volatility", lower=0.0)

    def instantaneous_variance(self, forward: ArrayLike | None = None, /) -> Array:
        del forward
        return self.volatility**2


class HestonModel(StrictModule):
    """Square-root stochastic variance dynamics without a measure-specific drift."""

    mean_reversion: Array
    long_run_variance: Array
    volatility_of_variance: Array
    correlation: Array
    initial_variance: Array
    require_feller: bool = eqx.field(static=True)

    def __init__(
        self,
        mean_reversion: ArrayLike,
        long_run_variance: ArrayLike,
        volatility_of_variance: ArrayLike,
        correlation: ArrayLike,
        initial_variance: ArrayLike,
        /,
        *,
        require_feller: bool = True,
    ):
        if not isinstance(require_feller, bool):
            raise TypeError("require_feller must be boolean.")
        kappa = _scalar(mean_reversion, "mean_reversion", lower=0.0)
        theta = _scalar(long_run_variance, "long_run_variance", lower=0.0)
        xi = _scalar(volatility_of_variance, "volatility_of_variance", lower=0.0)
        variance = _scalar(initial_variance, "initial_variance", lower=0.0)
        if require_feller:
            kappa = eqx.error_if(
                kappa,
                2.0 * kappa * theta < xi**2,
                "Heston parameters violate the Feller non-attainment condition.",
            )
        self.mean_reversion = kappa
        self.long_run_variance = theta
        self.volatility_of_variance = xi
        self.correlation = _correlation(correlation)
        self.initial_variance = variance
        self.require_feller = require_feller

    @property
    def feller_margin(self) -> Array:
        return (
            2.0 * self.mean_reversion * self.long_run_variance
            - self.volatility_of_variance**2
        )

    def variance_drift(self, variance: ArrayLike, /) -> Array:
        variance_ = jnp.asarray(variance)
        return self.mean_reversion * (self.long_run_variance - variance_)

    def variance_diffusion(self, variance: ArrayLike, /) -> Array:
        variance_ = jnp.asarray(variance)
        return self.volatility_of_variance * jnp.sqrt(jnp.maximum(variance_, 0.0))


class SABRModel(StrictModule):
    """SABR local/stochastic volatility parameters with an explicit displacement."""

    alpha: Array
    beta: Array
    volatility_of_volatility: Array
    correlation: Array
    shift: Array

    def __init__(
        self,
        alpha: ArrayLike,
        beta: ArrayLike,
        volatility_of_volatility: ArrayLike,
        correlation: ArrayLike,
        /,
        *,
        shift: ArrayLike = 0.0,
    ):
        beta_ = _scalar(beta, "beta")
        beta_ = eqx.error_if(
            beta_, (beta_ < 0.0) | (beta_ > 1.0), "beta must lie in [0, 1]."
        )
        nu = _scalar(volatility_of_volatility, "volatility_of_volatility")
        nu = eqx.error_if(nu, nu < 0.0, "volatility_of_volatility must be non-negative.")
        self.alpha = _scalar(alpha, "alpha", lower=0.0)
        self.beta = beta_
        self.volatility_of_volatility = nu
        self.correlation = _correlation(correlation)
        self.shift = _scalar(shift, "shift")

    def implied_volatility(
        self, forward: ArrayLike, strike: ArrayLike, maturity: ArrayLike, /
    ) -> Array:
        """Hagan lognormal SABR approximation with a stable ATM branch."""

        forward_ = jnp.asarray(forward, dtype=float) + self.shift
        strike_ = jnp.asarray(strike, dtype=float) + self.shift
        maturity_ = jnp.asarray(maturity, dtype=float)
        checked = eqx.error_if(
            forward_,
            jnp.any(~jnp.isfinite(forward_))
            | jnp.any(~jnp.isfinite(strike_))
            | jnp.any(~jnp.isfinite(maturity_))
            | jnp.any(forward_ <= 0.0)
            | jnp.any(strike_ <= 0.0)
            | jnp.any(maturity_ < 0.0),
            "Shifted forward/strike must be positive and maturity non-negative.",
        )
        forward_ = checked
        one_minus_beta = 1.0 - self.beta
        log_fk = jnp.log(forward_ / strike_)
        fk_beta = (forward_ * strike_) ** (0.5 * one_minus_beta)
        z = (self.volatility_of_volatility / self.alpha) * fk_beta * log_fk
        root = jnp.sqrt(jnp.maximum(1.0 - 2.0 * self.correlation * z + z**2, 0.0))
        x_z = jnp.log((root + z - self.correlation) / (1.0 - self.correlation))
        near_atm = jnp.abs(z) < 1.0e-7
        safe_x_z = jnp.where(near_atm, 1.0, x_z)
        z_over_x = jnp.where(near_atm, 1.0 - 0.5 * self.correlation * z, z / safe_x_z)
        log_correction = (
            1.0
            + (one_minus_beta**2 / 24.0) * log_fk**2
            + (one_minus_beta**4 / 1920.0) * log_fk**4
        )
        time_correction = 1.0 + maturity_ * (
            (one_minus_beta**2 / 24.0) * self.alpha**2 / fk_beta**2
            + 0.25
            * self.correlation
            * self.beta
            * self.volatility_of_volatility
            * self.alpha
            / fk_beta
            + (2.0 - 3.0 * self.correlation**2) * self.volatility_of_volatility**2 / 24.0
        )
        return self.alpha * z_over_x * time_correction / (fk_beta * log_correction)


class LocalVolatilityModel(StrictModule):
    """Positive local-variance surface on expiry/log-moneyness nodes."""

    expiries: Array
    log_moneyness: Array
    local_variance: Array
    expiry_count: int = eqx.field(static=True)
    strike_count: int = eqx.field(static=True)

    def __init__(
        self,
        expiries: ArrayLike,
        log_moneyness: ArrayLike,
        local_variance: ArrayLike,
        /,
    ):
        expiries_ = _ordered_grid(expiries, "expiries", positive=True)
        strikes_ = _ordered_grid(log_moneyness, "log_moneyness")
        variance = jnp.asarray(local_variance, dtype=float)
        expected = (int(expiries_.size), int(strikes_.size))
        if variance.shape != expected:
            raise ValueError(f"local_variance must have shape {expected}.")
        variance = eqx.error_if(
            variance,
            jnp.any(~jnp.isfinite(variance)) | jnp.any(variance <= 0.0),
            "local_variance must be finite and strictly positive.",
        )
        self.expiries = expiries_
        self.log_moneyness = strikes_
        self.local_variance = variance
        self.expiry_count, self.strike_count = expected

    def variance(self, expiry: ArrayLike, log_moneyness: ArrayLike, /) -> Array:
        """Bilinear interpolation with flat boundary extrapolation."""

        t = jnp.asarray(expiry, dtype=float)
        k = jnp.asarray(log_moneyness, dtype=float)
        t, k = jnp.broadcast_arrays(t, k)
        t = eqx.error_if(
            t,
            jnp.any(~jnp.isfinite(t)) | jnp.any(t < 0.0),
            "expiry must be finite and non-negative.",
        )
        k = eqx.error_if(k, jnp.any(~jnp.isfinite(k)), "log_moneyness must be finite.")
        ti = jnp.clip(
            jnp.searchsorted(self.expiries, t, side="right") - 1, 0, self.expiry_count - 2
        )
        ki = jnp.clip(
            jnp.searchsorted(self.log_moneyness, k, side="right") - 1,
            0,
            self.strike_count - 2,
        )
        t0, t1 = self.expiries[ti], self.expiries[ti + 1]
        k0, k1 = self.log_moneyness[ki], self.log_moneyness[ki + 1]
        wt = jnp.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        wk = jnp.clip((k - k0) / (k1 - k0), 0.0, 1.0)
        v00 = self.local_variance[ti, ki]
        v01 = self.local_variance[ti, ki + 1]
        v10 = self.local_variance[ti + 1, ki]
        v11 = self.local_variance[ti + 1, ki + 1]
        return (1.0 - wt) * ((1.0 - wk) * v00 + wk * v01) + wt * (
            (1.0 - wk) * v10 + wk * v11
        )

    def volatility(self, expiry: ArrayLike, log_moneyness: ArrayLike, /) -> Array:
        return jnp.sqrt(self.variance(expiry, log_moneyness))


class LocalStochasticVolatilityModel(StrictModule):
    """Heston variance factor combined with a positive leverage surface."""

    variance_model: HestonModel
    leverage_surface: LocalVolatilityModel
    reference_variance: Array

    def __init__(
        self,
        variance_model: HestonModel,
        leverage_surface: LocalVolatilityModel,
        /,
        *,
        reference_variance: ArrayLike | None = None,
    ):
        if not isinstance(variance_model, HestonModel):
            raise TypeError("variance_model must be a HestonModel.")
        if not isinstance(leverage_surface, LocalVolatilityModel):
            raise TypeError("leverage_surface must be a LocalVolatilityModel.")
        reference = (
            variance_model.long_run_variance
            if reference_variance is None
            else _scalar(reference_variance, "reference_variance", lower=0.0)
        )
        self.variance_model = variance_model
        self.leverage_surface = leverage_surface
        self.reference_variance = reference

    def spot_variance(
        self, expiry: ArrayLike, log_moneyness: ArrayLike, variance: ArrayLike, /
    ) -> Array:
        variance_ = jnp.asarray(variance, dtype=float)
        variance_ = eqx.error_if(
            variance_,
            jnp.any(~jnp.isfinite(variance_)) | jnp.any(variance_ < 0.0),
            "stochastic variance must be finite and non-negative.",
        )
        leverage_variance = self.leverage_surface.variance(expiry, log_moneyness)
        return leverage_variance * variance_ / self.reference_variance


__all__ = [
    "BachelierModel",
    "Black76Model",
    "BlackScholesModel",
    "HestonModel",
    "LocalStochasticVolatilityModel",
    "LocalVolatilityModel",
    "SABRModel",
]
