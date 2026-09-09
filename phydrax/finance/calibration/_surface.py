#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""SVI/eSSVI volatility surfaces and static-arbitrage evidence."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


def _scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(array, ~jnp.isfinite(array), f"{name} must be finite.")


def _ordered(value: ArrayLike, name: str, /, *, positive: bool = False) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty vector.")
    invalid = jnp.any(~jnp.isfinite(array))
    if array.size > 1:
        invalid = invalid | jnp.any(jnp.diff(array) <= 0.0)
    if positive:
        invalid = invalid | jnp.any(array <= 0.0)
    return eqx.error_if(array, invalid, f"{name} must be finite and strictly increasing.")


class SVIParameters(StrictModule):
    """Raw-SVI total-variance parameters with non-negative global minimum."""

    level: Array
    slope: Array
    correlation: Array
    center: Array
    width: Array

    def __init__(
        self,
        level: ArrayLike,
        slope: ArrayLike,
        correlation: ArrayLike,
        center: ArrayLike,
        width: ArrayLike,
        /,
    ):
        level, slope, correlation, center, width = tuple(
            _scalar(value, name)
            for value, name in zip(
                (level, slope, correlation, center, width),
                ("level", "slope", "correlation", "center", "width"),
                strict=True,
            )
        )
        slope = eqx.error_if(slope, slope <= 0.0, "SVI slope must be strictly positive.")
        correlation = eqx.error_if(
            correlation,
            jnp.abs(correlation) >= 1.0,
            "SVI correlation must lie strictly in (-1, 1).",
        )
        width = eqx.error_if(width, width <= 0.0, "SVI width must be strictly positive.")
        minimum = level + slope * width * jnp.sqrt(1.0 - correlation**2)
        level = eqx.error_if(
            level,
            minimum < 0.0,
            "SVI global minimum total variance must be non-negative.",
        )
        self.level, self.slope, self.correlation, self.center, self.width = (
            level,
            slope,
            correlation,
            center,
            width,
        )

    def total_variance(self, log_moneyness: ArrayLike, /) -> Array:
        k = jnp.asarray(log_moneyness, dtype=float)
        shifted = k - self.center
        return self.level + self.slope * (
            self.correlation * shifted + jnp.sqrt(shifted**2 + self.width**2)
        )

    def first_derivative(self, log_moneyness: ArrayLike, /) -> Array:
        shifted = jnp.asarray(log_moneyness, dtype=float) - self.center
        return self.slope * (
            self.correlation + shifted / jnp.sqrt(shifted**2 + self.width**2)
        )

    def second_derivative(self, log_moneyness: ArrayLike, /) -> Array:
        shifted = jnp.asarray(log_moneyness, dtype=float) - self.center
        return self.slope * self.width**2 / (shifted**2 + self.width**2) ** 1.5


class SVISlice(StrictModule):
    expiry: Array
    parameters: SVIParameters

    def __init__(self, expiry: ArrayLike, parameters: SVIParameters, /):
        if not isinstance(parameters, SVIParameters):
            raise TypeError("parameters must be SVIParameters.")
        expiry_ = _scalar(expiry, "expiry")
        self.expiry = eqx.error_if(
            expiry_, expiry_ <= 0.0, "SVI expiry must be positive."
        )
        self.parameters = parameters

    def total_variance(self, log_moneyness: ArrayLike, /) -> Array:
        return self.parameters.total_variance(log_moneyness)

    def implied_volatility(self, log_moneyness: ArrayLike, /) -> Array:
        return jnp.sqrt(self.total_variance(log_moneyness) / self.expiry)


class SVISurface(StrictModule):
    slices: tuple[SVISlice, ...]
    expiries: Array
    slice_count: int = eqx.field(static=True)

    def __init__(self, slices: Sequence[SVISlice], /):
        slices_ = tuple(slices)
        if not slices_ or any(not isinstance(value, SVISlice) for value in slices_):
            raise TypeError("slices must contain at least one SVISlice.")
        ordered = tuple(sorted(slices_, key=lambda value: float(value.expiry)))
        expiries = jnp.stack(tuple(value.expiry for value in ordered))
        expiries = eqx.error_if(
            expiries,
            jnp.any(jnp.diff(expiries) <= 0.0),
            "SVI slice expiries must be unique.",
        )
        self.slices, self.expiries, self.slice_count = ordered, expiries, len(ordered)

    def total_variance(self, expiry: ArrayLike, log_moneyness: ArrayLike, /) -> Array:
        expiry_, k = jnp.broadcast_arrays(
            jnp.asarray(expiry, dtype=float), jnp.asarray(log_moneyness, dtype=float)
        )
        stacked = jnp.stack(
            tuple(value.total_variance(k) for value in self.slices), axis=0
        )
        upper = (
            jnp.clip(
                jnp.searchsorted(self.expiries, expiry_, side="right"),
                1,
                self.slice_count - 1,
            )
            if self.slice_count > 1
            else jnp.zeros_like(expiry_, dtype=jnp.int32)
        )
        lower = jnp.maximum(upper - 1, 0)
        if self.slice_count == 1:
            return stacked[0]
        weight = jnp.clip(
            (expiry_ - self.expiries[lower])
            / (self.expiries[upper] - self.expiries[lower]),
            0.0,
            1.0,
        )
        lower_value = jnp.take_along_axis(stacked, lower[None, ...], axis=0)[0]
        upper_value = jnp.take_along_axis(stacked, upper[None, ...], axis=0)[0]
        return (1.0 - weight) * lower_value + weight * upper_value


class ESSVISurface(StrictModule):
    """Extended SSVI surface with monotone ATM variance and sufficient bounds."""

    expiries: Array
    atm_total_variances: Array
    correlations: Array
    eta: Array
    gamma: Array
    expiry_count: int = eqx.field(static=True)

    def __init__(
        self,
        expiries: ArrayLike,
        atm_total_variances: ArrayLike,
        correlations: ArrayLike,
        eta: ArrayLike,
        gamma: ArrayLike,
        /,
    ):
        expiries_ = _ordered(expiries, "expiries", positive=True)
        theta = jnp.asarray(atm_total_variances, dtype=float)
        rho = jnp.asarray(correlations, dtype=float)
        if theta.shape != expiries_.shape or rho.shape != expiries_.shape:
            raise ValueError(
                "ATM total variance and correlations must align with expiries."
            )
        theta = eqx.error_if(
            theta,
            jnp.any(~jnp.isfinite(theta))
            | jnp.any(theta <= 0.0)
            | jnp.any(jnp.diff(theta) < 0.0),
            "ATM total variance must be positive, finite and nondecreasing.",
        )
        rho = eqx.error_if(
            rho,
            jnp.any(~jnp.isfinite(rho)) | jnp.any(jnp.abs(rho) >= 1.0),
            "eSSVI correlations must lie strictly in (-1, 1).",
        )
        eta_ = _scalar(eta, "eta")
        eta_ = eqx.error_if(eta_, eta_ <= 0.0, "eta must be positive.")
        gamma_ = _scalar(gamma, "gamma")
        gamma_ = eqx.error_if(
            gamma_, (gamma_ < 0.0) | (gamma_ > 1.0), "gamma must lie in [0, 1]."
        )
        phi = eta_ / (theta**gamma_ * (1.0 + theta) ** (1.0 - gamma_))
        eta_ = eqx.error_if(
            eta_,
            jnp.any(theta * phi * (1.0 + jnp.abs(rho)) >= 4.0)
            | jnp.any(theta * phi**2 * (1.0 + jnp.abs(rho)) > 4.0),
            "eSSVI parameters violate sufficient butterfly-arbitrage bounds.",
        )
        self.expiries, self.atm_total_variances, self.correlations = expiries_, theta, rho
        self.eta, self.gamma, self.expiry_count = eta_, gamma_, int(expiries_.size)

    def _theta_rho(self, expiry: ArrayLike) -> tuple[Array, Array]:
        expiry_ = jnp.asarray(expiry, dtype=float)
        expiry_ = eqx.error_if(
            expiry_,
            jnp.any(~jnp.isfinite(expiry_)) | jnp.any(expiry_ <= 0.0),
            "expiry must be finite and positive.",
        )
        return (
            jnp.interp(expiry_, self.expiries, self.atm_total_variances),
            jnp.interp(expiry_, self.expiries, self.correlations),
        )

    def total_variance(self, expiry: ArrayLike, log_moneyness: ArrayLike, /) -> Array:
        theta, rho = self._theta_rho(expiry)
        k = jnp.asarray(log_moneyness, dtype=float)
        theta, rho, k = jnp.broadcast_arrays(theta, rho, k)
        phi = self.eta / (theta**self.gamma * (1.0 + theta) ** (1.0 - self.gamma))
        return (
            0.5
            * theta
            * (1.0 + rho * phi * k + jnp.sqrt((phi * k + rho) ** 2 + 1.0 - rho**2))
        )

    def implied_volatility(self, expiry: ArrayLike, log_moneyness: ArrayLike, /) -> Array:
        return jnp.sqrt(self.total_variance(expiry, log_moneyness) / jnp.asarray(expiry))


class VolatilityObservationSet(StrictModule):
    expiries: Array
    log_moneyness: Array
    implied_volatilities: Array
    weights: Array
    valid: Array
    observation_count: int = eqx.field(static=True)

    def __init__(
        self,
        expiries: ArrayLike,
        log_moneyness: ArrayLike,
        implied_volatilities: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        valid: ArrayLike | None = None,
    ):
        expiries_, k, volatility = tuple(
            jnp.asarray(value, dtype=float)
            for value in (expiries, log_moneyness, implied_volatilities)
        )
        if (
            expiries_.ndim != 1
            or expiries_.size < 5
            or k.shape != expiries_.shape
            or volatility.shape != expiries_.shape
        ):
            raise ValueError(
                "volatility observations must be aligned vectors with at least five entries."
            )
        mask = (
            jnp.ones_like(expiries_, dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        weight = (
            jnp.ones_like(expiries_)
            if weights is None
            else jnp.asarray(weights, dtype=float)
        )
        if mask.shape != expiries_.shape or weight.shape != expiries_.shape:
            raise ValueError("weights and valid must align with observations.")
        expiries_ = eqx.error_if(
            expiries_,
            jnp.any(mask & (~jnp.isfinite(expiries_) | (expiries_ <= 0.0)))
            | jnp.any(~mask & (expiries_ != 0.0)),
            "active expiries must be finite/positive and inactive expiries zero.",
        )
        k = eqx.error_if(
            k,
            jnp.any(mask & ~jnp.isfinite(k)) | jnp.any(~mask & (k != 0.0)),
            "active log-moneyness must be finite and inactive values zero.",
        )
        volatility = eqx.error_if(
            volatility,
            jnp.any(mask & (~jnp.isfinite(volatility) | (volatility <= 0.0)))
            | jnp.any(~mask & (volatility != 0.0)),
            "active implied volatility must be finite/positive and inactive values zero.",
        )
        weight = eqx.error_if(
            weight,
            jnp.any(mask & (~jnp.isfinite(weight) | (weight <= 0.0)))
            | jnp.any(~mask & (weight != 0.0)),
            "active weights must be finite/positive and inactive values zero.",
        )
        self.expiries, self.log_moneyness, self.implied_volatilities = (
            expiries_,
            k,
            volatility,
        )
        self.weights, self.valid, self.observation_count = (
            weight,
            mask,
            int(expiries_.size),
        )

    @property
    def total_variances(self) -> Array:
        return self.implied_volatilities**2 * self.expiries


class SurfaceArbitrageEvidence(StrictModule):
    minimum_total_variance: Array
    minimum_density_factor: Array
    minimum_calendar_increment: Array
    maximum_wing_slope: Array
    butterfly_free: Array
    calendar_free: Array
    wing_admissible: Array
    finite: Array
    valid: Array


def _surface_values(surface, expiry, k):
    if isinstance(surface, SVISlice):
        return surface.total_variance(k)
    return surface.total_variance(expiry, k)


def evaluate_surface_arbitrage(
    surface: SVISlice | SVISurface | ESSVISurface,
    /,
    *,
    log_moneyness_grid: ArrayLike | None = None,
) -> SurfaceArbitrageEvidence:
    if not isinstance(surface, (SVISlice, SVISurface, ESSVISurface)):
        raise TypeError("surface must be an SVI/eSSVI surface record.")
    k = (
        jnp.linspace(-5.0, 5.0, 401)
        if log_moneyness_grid is None
        else _ordered(log_moneyness_grid, "log_moneyness_grid")
    )
    expiries = surface.expiry[None] if isinstance(surface, SVISlice) else surface.expiries

    def one_slice(expiry):
        function = lambda point: _surface_values(surface, expiry, point)
        w = jax.vmap(function)(k)
        first = jax.vmap(jax.grad(function))(k)
        second = jax.vmap(jax.grad(jax.grad(function)))(k)
        safe_w = jnp.maximum(w, jnp.finfo(w.dtype).eps)
        density = (
            (1.0 - k * first / (2.0 * safe_w)) ** 2
            - 0.25 * first**2 * (1.0 / safe_w + 0.25)
            + 0.5 * second
        )
        return w, density, first

    variances, densities, slopes = jax.vmap(one_slice)(expiries)
    calendar = jnp.diff(variances, axis=0)
    minimum_calendar = jnp.min(calendar) if expiries.size > 1 else jnp.asarray(jnp.inf)
    minimum_variance = jnp.min(variances)
    minimum_density = jnp.min(densities)
    maximum_wing = jnp.max(jnp.abs(jnp.stack((slopes[:, 0], slopes[:, -1]), axis=0)))
    finite = (
        jnp.all(jnp.isfinite(variances))
        & jnp.all(jnp.isfinite(densities))
        & jnp.all(jnp.isfinite(slopes))
    )
    butterfly = minimum_density >= -1.0e-8
    calendar_free = minimum_calendar >= -1.0e-8
    wing = maximum_wing <= 2.0 + 1.0e-8
    return SurfaceArbitrageEvidence(
        minimum_variance,
        minimum_density,
        minimum_calendar,
        maximum_wing,
        butterfly,
        calendar_free,
        wing,
        finite,
        finite & (minimum_variance >= 0.0) & butterfly & calendar_free & wing,
    )


__all__ = [
    "ESSVISurface",
    "SVIParameters",
    "SVISlice",
    "SVISurface",
    "SurfaceArbitrageEvidence",
    "VolatilityObservationSet",
    "evaluate_surface_arbitrage",
]
