# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Measure-aware finite-dimensional interest-rate models.

The records in this module are measure-neutral numerical structures.  A concrete
``PhysicalLaw``, ``PricingLaw``, or ``StressLaw`` is supplied to a calculation;
the three descriptors are deliberately not interchangeable.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...stochastic import WienerRealization
from ..core import PhysicalLaw, PricingLaw, StressLaw


RatesLaw: TypeAlias = PhysicalLaw | PricingLaw | StressLaw
LMMMeasure: TypeAlias = Literal["spot", "terminal"]


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string.")
    return value


def _scalar(value: ArrayLike, name: str, /, *, positive: bool = False) -> Array:
    result = jnp.asarray(value, dtype=float)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    host = float(np.asarray(jax.device_get(result)))
    if not isfinite(host) or (positive and host <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _nonnegative_scalar(value: ArrayLike, name: str, /) -> Array:
    result = _scalar(value, name)
    if float(np.asarray(jax.device_get(result))) < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _finite_vector(value: ArrayLike, name: str, /, *, minimum_size: int = 1) -> Array:
    result = jnp.asarray(value, dtype=float)
    host = np.asarray(jax.device_get(result))
    if result.ndim != 1 or result.shape[0] < minimum_size:
        raise ValueError(f"{name} must be a vector with at least {minimum_size} entries.")
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must be finite.")
    return result


def _strict_grid(value: ArrayLike, name: str, /, *, minimum_size: int = 2) -> Array:
    result = _finite_vector(value, name, minimum_size=minimum_size)
    host = np.asarray(jax.device_get(result))
    if host[0] < 0.0 or np.any(np.diff(host) <= 0.0):
        raise ValueError(f"{name} must be nonnegative and strictly increasing.")
    return result


def _correlation(value: ArrayLike, factor_count: int, /) -> Array:
    result = jnp.asarray(value, dtype=float)
    host = np.asarray(jax.device_get(result))
    expected = (factor_count, factor_count)
    if result.shape != expected:
        raise ValueError(f"factor_correlation must have shape {expected}.")
    if not np.all(np.isfinite(host)):
        raise ValueError("factor_correlation must be finite.")
    if not np.allclose(host, host.T, rtol=0.0, atol=1.0e-10):
        raise ValueError("factor_correlation must be symmetric.")
    if not np.allclose(np.diag(host), 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("factor_correlation must have a unit diagonal.")
    if float(np.min(np.linalg.eigvalsh(host))) < -1.0e-10:
        raise ValueError("factor_correlation must be positive semidefinite.")
    return result


def _law_id(law: RatesLaw, /) -> str:
    return law.law_id


def _require_law(
    law: RatesLaw,
    /,
    *,
    state_layout_id: str,
    pricing_measure_id: str | None = None,
    pricing_only: bool = False,
) -> None:
    if pricing_only and not isinstance(law, PricingLaw):
        raise TypeError(
            "This valuation requires a PricingLaw; physical and stress laws are invalid."
        )
    if not isinstance(law, (PhysicalLaw, PricingLaw, StressLaw)):
        raise TypeError("law must be a PhysicalLaw, PricingLaw, or StressLaw.")
    if law.factor_layout_id != state_layout_id:
        raise ValueError("Law factor layout is incompatible with the rates model.")
    if pricing_measure_id is not None:
        if not isinstance(law, PricingLaw):
            raise TypeError("A pricing measure can only be bound by a PricingLaw.")
        if law.measure_id != pricing_measure_id:
            raise ValueError("Pricing-law measure is incompatible with the rates model.")


class VasicekModel(StrictModule):
    """One-factor Gaussian mean-reverting short-rate structure."""

    mean_reversion: Array
    long_run_rate: Array
    volatility: Array
    currency_id: str = eqx.field(static=True)
    state_layout_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean_reversion: ArrayLike,
        long_run_rate: ArrayLike,
        volatility: ArrayLike,
        /,
        *,
        currency_id: str,
        state_layout_id: str,
        model_id: str,
    ):
        self.mean_reversion = _scalar(mean_reversion, "mean_reversion", positive=True)
        self.long_run_rate = _scalar(long_run_rate, "long_run_rate")
        self.volatility = _nonnegative_scalar(volatility, "volatility")
        self.currency_id = _identifier(currency_id, "currency_id")
        self.state_layout_id = _identifier(state_layout_id, "state_layout_id")
        self.model_id = _identifier(model_id, "model_id")


class HullWhiteModel(StrictModule):
    """One-factor Hull--White structure with an explicit deterministic mean curve."""

    mean_reversion: Array
    volatility: Array
    mean_times: Array
    mean_values: Array
    currency_id: str = eqx.field(static=True)
    state_layout_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean_reversion: ArrayLike,
        volatility: ArrayLike,
        mean_times: ArrayLike,
        mean_values: ArrayLike,
        /,
        *,
        currency_id: str,
        state_layout_id: str,
        model_id: str,
    ):
        times = _strict_grid(mean_times, "mean_times")
        values = _finite_vector(mean_values, "mean_values", minimum_size=2)
        if values.shape != times.shape:
            raise ValueError("mean_values must match mean_times.")
        self.mean_reversion = _scalar(mean_reversion, "mean_reversion", positive=True)
        self.volatility = _nonnegative_scalar(volatility, "volatility")
        self.mean_times = times
        self.mean_values = values
        self.currency_id = _identifier(currency_id, "currency_id")
        self.state_layout_id = _identifier(state_layout_id, "state_layout_id")
        self.model_id = _identifier(model_id, "model_id")


class CIRModel(StrictModule):
    """Cox--Ingersoll--Ross square-root short-rate structure."""

    mean_reversion: Array
    long_run_rate: Array
    volatility: Array
    currency_id: str = eqx.field(static=True)
    state_layout_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean_reversion: ArrayLike,
        long_run_rate: ArrayLike,
        volatility: ArrayLike,
        /,
        *,
        currency_id: str,
        state_layout_id: str,
        model_id: str,
    ):
        self.mean_reversion = _scalar(mean_reversion, "mean_reversion", positive=True)
        self.long_run_rate = _scalar(long_run_rate, "long_run_rate", positive=True)
        self.volatility = _scalar(volatility, "volatility", positive=True)
        self.currency_id = _identifier(currency_id, "currency_id")
        self.state_layout_id = _identifier(state_layout_id, "state_layout_id")
        self.model_id = _identifier(model_id, "model_id")

    @property
    def feller_margin(self) -> Array:
        return 2.0 * self.mean_reversion * self.long_run_rate - self.volatility**2

    @property
    def feller_condition(self) -> Array:
        return self.feller_margin >= 0.0


class CIRPlusPlusModel(StrictModule):
    """CIR factor plus an explicit piecewise-linear deterministic shift."""

    base: CIRModel
    shift_times: Array
    shift_values: Array
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: CIRModel,
        shift_times: ArrayLike,
        shift_values: ArrayLike,
        /,
        *,
        model_id: str,
    ):
        if not isinstance(base, CIRModel):
            raise TypeError("base must be a CIRModel.")
        times = _strict_grid(shift_times, "shift_times")
        values = _finite_vector(shift_values, "shift_values", minimum_size=2)
        if values.shape != times.shape:
            raise ValueError("shift_values must match shift_times.")
        self.base = base
        self.shift_times = times
        self.shift_values = values
        self.model_id = _identifier(model_id, "model_id")

    @property
    def currency_id(self) -> str:
        return self.base.currency_id

    @property
    def state_layout_id(self) -> str:
        return self.base.state_layout_id


class FiniteFactorHJMModel(StrictModule):
    """Finite-factor HJM model on one fixed maturity grid.

    ``volatility`` has shape ``(maturity, factor)``.  The drift returned by
    :func:`hjm_risk_neutral_drift` enforces the finite-grid HJM restriction under
    the descriptor's named risk-neutral measure.
    """

    tenor_times: Array
    volatility: Array
    factor_correlation: Array
    factor_ids: tuple[str, ...] = eqx.field(static=True)
    currency_id: str = eqx.field(static=True)
    pricing_measure_id: str = eqx.field(static=True)
    state_layout_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        tenor_times: ArrayLike,
        volatility: ArrayLike,
        factor_correlation: ArrayLike,
        /,
        *,
        factor_ids: tuple[str, ...],
        currency_id: str,
        pricing_measure_id: str,
        state_layout_id: str,
        model_id: str,
    ):
        times = _strict_grid(tenor_times, "tenor_times")
        ids = tuple(_identifier(value, "factor_id") for value in factor_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("factor_ids must be non-empty and unique.")
        sigma = jnp.asarray(volatility, dtype=float)
        sigma_host = np.asarray(jax.device_get(sigma))
        if sigma.shape != (times.shape[0], len(ids)):
            raise ValueError("volatility must have shape (tenor, factor).")
        if not np.all(np.isfinite(sigma_host)):
            raise ValueError("volatility must be finite.")
        self.tenor_times = times
        self.volatility = sigma
        self.factor_correlation = _correlation(factor_correlation, len(ids))
        self.factor_ids = ids
        self.currency_id = _identifier(currency_id, "currency_id")
        self.pricing_measure_id = _identifier(pricing_measure_id, "pricing_measure_id")
        self.state_layout_id = _identifier(state_layout_id, "state_layout_id")
        self.model_id = _identifier(model_id, "model_id")

    @property
    def factor_count(self) -> int:
        return len(self.factor_ids)


class LiborMarketModel(StrictModule):
    """Displaced finite-factor market model on a fixed forward-tenor grid."""

    tenor_times: Array
    displacements: Array
    volatility: Array
    factor_correlation: Array
    factor_ids: tuple[str, ...] = eqx.field(static=True)
    measure: LMMMeasure = eqx.field(static=True)
    currency_id: str = eqx.field(static=True)
    pricing_measure_id: str = eqx.field(static=True)
    state_layout_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        tenor_times: ArrayLike,
        displacements: ArrayLike,
        volatility: ArrayLike,
        factor_correlation: ArrayLike,
        /,
        *,
        factor_ids: tuple[str, ...],
        measure: LMMMeasure,
        currency_id: str,
        pricing_measure_id: str,
        state_layout_id: str,
        model_id: str,
    ):
        times = _strict_grid(tenor_times, "tenor_times")
        ids = tuple(_identifier(value, "factor_id") for value in factor_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("factor_ids must be non-empty and unique.")
        if measure not in ("spot", "terminal"):
            raise ValueError("measure must be 'spot' or 'terminal'.")
        forward_count = times.shape[0] - 1
        shifts = _finite_vector(displacements, "displacements", minimum_size=1)
        sigma = jnp.asarray(volatility, dtype=float)
        if shifts.shape != (forward_count,):
            raise ValueError("displacements must have one value per forward tenor.")
        if sigma.shape != (forward_count, len(ids)):
            raise ValueError("volatility must have shape (forward_tenor, factor).")
        if not np.all(np.isfinite(np.asarray(jax.device_get(sigma)))):
            raise ValueError("volatility must be finite.")
        self.tenor_times = times
        self.displacements = shifts
        self.volatility = sigma
        self.factor_correlation = _correlation(factor_correlation, len(ids))
        self.factor_ids = ids
        self.measure = measure
        self.currency_id = _identifier(currency_id, "currency_id")
        self.pricing_measure_id = _identifier(pricing_measure_id, "pricing_measure_id")
        self.state_layout_id = _identifier(state_layout_id, "state_layout_id")
        self.model_id = _identifier(model_id, "model_id")

    @property
    def factor_count(self) -> int:
        return len(self.factor_ids)

    @property
    def forward_count(self) -> int:
        return self.tenor_times.shape[0] - 1


class RatesPathBatch(StrictModule):
    """Fixed-grid rate paths with explicit law and realization identity."""

    times: Array
    values: Array
    valid: Array
    model_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    state_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        model_id: str,
        law_id: str,
        realization_id: str,
        state_layout_id: str,
    ):
        nodes = jnp.asarray(times, dtype=float)
        paths = jnp.asarray(values, dtype=float)
        path_valid = jnp.asarray(valid, dtype=bool)
        if nodes.ndim != 1 or nodes.shape[0] < 2:
            raise ValueError("times must be a vector with at least two nodes.")
        if paths.ndim != 3 or paths.shape[1] != nodes.shape[0]:
            raise ValueError("values must have shape (path, time, state).")
        if path_valid.shape != (paths.shape[0],):
            raise ValueError("valid must have shape (path,).")
        self.times = nodes
        self.values = paths
        self.valid = path_valid
        self.model_id = _identifier(model_id, "model_id")
        self.law_id = _identifier(law_id, "law_id")
        self.realization_id = _identifier(realization_id, "realization_id")
        self.state_layout_id = _identifier(state_layout_id, "state_layout_id")


def vasicek_zero_coupon_bond(
    model: VasicekModel,
    law: PricingLaw,
    short_rate: ArrayLike,
    time: ArrayLike,
    maturity: ArrayLike,
    /,
) -> Array:
    """Analytic Vasicek zero-coupon value under the supplied pricing law."""

    if not isinstance(model, VasicekModel):
        raise TypeError("model must be a VasicekModel.")
    _require_law(law, state_layout_id=model.state_layout_id, pricing_only=True)
    t = jnp.asarray(time, dtype=float)
    maturity_ = jnp.asarray(maturity, dtype=t.dtype)
    tau = maturity_ - t
    tau = eqx.error_if(tau, jnp.any(tau < 0.0), "maturity must not precede time.")
    a = model.mean_reversion
    sigma = model.volatility
    b = -jnp.expm1(-a * tau) / a
    log_a = (model.long_run_rate - sigma**2 / (2.0 * a**2)) * (
        b - tau
    ) - sigma**2 * b**2 / (4.0 * a)
    return jnp.exp(log_a - b * jnp.asarray(short_rate, dtype=t.dtype))


def hull_white_zero_coupon_bond(
    model: HullWhiteModel,
    law: PricingLaw,
    short_rate: ArrayLike,
    time: ArrayLike,
    maturity: ArrayLike,
    /,
    *,
    initial_discount_at_time: ArrayLike,
    initial_discount_at_maturity: ArrayLike,
    initial_forward_at_time: ArrayLike,
) -> Array:
    """Analytic one-factor Hull--White bond value from an explicit initial curve."""

    if not isinstance(model, HullWhiteModel):
        raise TypeError("model must be a HullWhiteModel.")
    _require_law(law, state_layout_id=model.state_layout_id, pricing_only=True)
    t = jnp.asarray(time, dtype=float)
    maturity_ = jnp.asarray(maturity, dtype=t.dtype)
    tau = maturity_ - t
    tau = eqx.error_if(tau, jnp.any(tau < 0.0), "maturity must not precede time.")
    p_t = jnp.asarray(initial_discount_at_time, dtype=t.dtype)
    p_maturity = jnp.asarray(initial_discount_at_maturity, dtype=t.dtype)
    p_t = eqx.error_if(
        p_t, jnp.any(p_t <= 0.0), "Initial discount factors must be positive."
    )
    p_maturity = eqx.error_if(
        p_maturity,
        jnp.any(p_maturity <= 0.0),
        "Initial discount factors must be positive.",
    )
    a = model.mean_reversion
    b = -jnp.expm1(-a * tau) / a
    variance_term = model.volatility**2 * (-jnp.expm1(-2.0 * a * t)) * b**2 / (4.0 * a)
    return (p_maturity / p_t) * jnp.exp(
        b * jnp.asarray(initial_forward_at_time, dtype=t.dtype)
        - b * jnp.asarray(short_rate, dtype=t.dtype)
        - variance_term
    )


def cir_zero_coupon_bond(
    model: CIRModel,
    law: PricingLaw,
    short_rate: ArrayLike,
    time: ArrayLike,
    maturity: ArrayLike,
    /,
) -> Array:
    """Analytic CIR zero-coupon value under the supplied pricing law."""

    if not isinstance(model, CIRModel):
        raise TypeError("model must be a CIRModel.")
    _require_law(law, state_layout_id=model.state_layout_id, pricing_only=True)
    rate = jnp.asarray(short_rate, dtype=float)
    rate = eqx.error_if(rate, jnp.any(rate < 0.0), "CIR short rates must be nonnegative.")
    t = jnp.asarray(time, dtype=rate.dtype)
    tau = jnp.asarray(maturity, dtype=rate.dtype) - t
    tau = eqx.error_if(tau, jnp.any(tau < 0.0), "maturity must not precede time.")
    kappa = model.mean_reversion
    theta = model.long_run_rate
    sigma = model.volatility
    gamma = jnp.sqrt(kappa**2 + 2.0 * sigma**2)
    expm1 = jnp.expm1(gamma * tau)
    denominator = (gamma + kappa) * expm1 + 2.0 * gamma
    b = 2.0 * expm1 / denominator
    log_a = (2.0 * kappa * theta / sigma**2) * (
        jnp.log(2.0 * gamma) + 0.5 * (kappa + gamma) * tau - jnp.log(denominator)
    )
    return jnp.exp(log_a - b * rate)


def cir_plus_plus_shift_integral(
    model: CIRPlusPlusModel,
    start: ArrayLike,
    end: ArrayLike,
    /,
) -> Array:
    """Integrate the piecewise-linear deterministic shift exactly."""

    if not isinstance(model, CIRPlusPlusModel):
        raise TypeError("model must be a CIRPlusPlusModel.")
    left = jnp.asarray(start, dtype=float)
    right = jnp.asarray(end, dtype=left.dtype)
    right = eqx.error_if(right, jnp.any(right < left), "end must not precede start.")
    left = eqx.error_if(
        left,
        jnp.any((left < model.shift_times[0]) | (right > model.shift_times[-1])),
        "Shift integration interval lies outside the shift grid.",
    )
    segment_left = model.shift_times[:-1]
    segment_right = model.shift_times[1:]
    width = segment_right - segment_left
    lo = jnp.maximum(left[..., None], segment_left)
    hi = jnp.minimum(right[..., None], segment_right)
    active = hi > lo
    slope = (model.shift_values[1:] - model.shift_values[:-1]) / width
    value_lo = model.shift_values[:-1] + slope * (lo - segment_left)
    value_hi = model.shift_values[:-1] + slope * (hi - segment_left)
    contribution = 0.5 * (value_lo + value_hi) * (hi - lo)
    return jnp.sum(jnp.where(active, contribution, 0.0), axis=-1)


def cir_plus_plus_zero_coupon_bond(
    model: CIRPlusPlusModel,
    law: PricingLaw,
    factor: ArrayLike,
    time: ArrayLike,
    maturity: ArrayLike,
    /,
) -> Array:
    _require_law(law, state_layout_id=model.state_layout_id, pricing_only=True)
    base_value = cir_zero_coupon_bond(model.base, law, factor, time, maturity)
    return base_value * jnp.exp(-cir_plus_plus_shift_integral(model, time, maturity))


def hjm_risk_neutral_drift(
    model: FiniteFactorHJMModel,
    law: PricingLaw,
    observation_index: int,
    /,
) -> Array:
    """Finite-grid HJM drift ``sigma(T) C integral_t^T sigma(u) du``."""

    if not isinstance(model, FiniteFactorHJMModel):
        raise TypeError("model must be a FiniteFactorHJMModel.")
    _require_law(
        law,
        state_layout_id=model.state_layout_id,
        pricing_measure_id=model.pricing_measure_id,
        pricing_only=True,
    )
    index = int(observation_index)
    maturity_count = model.tenor_times.shape[0]
    if index < 0 or index >= maturity_count:
        raise ValueError("observation_index lies outside the tenor grid.")
    widths = jnp.diff(model.tenor_times)
    segment_volatility = 0.5 * (model.volatility[:-1] + model.volatility[1:])
    integrals = jnp.concatenate(
        (
            jnp.zeros((1, model.factor_count), dtype=model.volatility.dtype),
            jnp.cumsum(segment_volatility * widths[:, None], axis=0),
        ),
        axis=0,
    )
    from_observation = integrals - integrals[index]
    covariance_integral = from_observation @ model.factor_correlation.T
    drift = jnp.sum(model.volatility * covariance_integral, axis=-1)
    maturity_mask = jnp.arange(maturity_count) >= index
    return jnp.where(maturity_mask, drift, 0.0)


def lmm_drift(
    model: LiborMarketModel,
    law: PricingLaw,
    forwards: ArrayLike,
    /,
    *,
    alive_index: int = 0,
) -> Array:
    """Spot- or terminal-measure LMM drift for the live fixed-tenor forwards."""

    if not isinstance(model, LiborMarketModel):
        raise TypeError("model must be a LiborMarketModel.")
    _require_law(
        law,
        state_layout_id=model.state_layout_id,
        pricing_measure_id=model.pricing_measure_id,
        pricing_only=True,
    )
    if isinstance(alive_index, bool) or not isinstance(alive_index, int):
        raise TypeError("alive_index must be an integer.")
    if alive_index < 0 or alive_index >= model.forward_count:
        raise ValueError("alive_index lies outside the forward-tenor axis.")
    rates = jnp.asarray(forwards, dtype=float)
    if rates.shape[-1:] != (model.forward_count,):
        raise ValueError("forwards must end in the model forward-tenor axis.")
    shifted = rates + model.displacements
    shifted = eqx.error_if(
        shifted,
        jnp.any(shifted <= 0.0),
        "Displaced forwards must be positive.",
    )
    accrual = jnp.diff(model.tenor_times)
    denominator = 1.0 + accrual * rates
    denominator = eqx.error_if(
        denominator,
        jnp.any(denominator <= 0.0),
        "LMM compounding denominators must be positive.",
    )
    loading = accrual * shifted / denominator
    covariance = model.volatility @ model.factor_correlation @ model.volatility.T
    indices = jnp.arange(model.forward_count)
    live = indices >= alive_index
    if model.measure == "terminal":
        mask = (indices[None, :] > indices[:, None]) & live[None, :]
        sign = -1.0
    else:
        mask = (indices[None, :] <= indices[:, None]) & live[None, :]
        sign = 1.0
    weighted = jnp.where(mask, covariance, 0.0)
    drift = sign * ein.contract("...j,ij->...i", loading, weighted)
    return jnp.where(live, drift, 0.0)


def require_tenor_compatibility(
    model: FiniteFactorHJMModel | LiborMarketModel,
    cashflow_times: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> None:
    """Require every active cashflow time to coincide with a model tenor node."""

    if not isinstance(model, (FiniteFactorHJMModel, LiborMarketModel)):
        raise TypeError("model must be a FiniteFactorHJMModel or LiborMarketModel.")
    values = np.asarray(jax.device_get(jnp.asarray(cashflow_times, dtype=float)))
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("cashflow_times must be one-dimensional and finite.")
    tol = float(tolerance)
    if not isfinite(tol) or tol < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    tenor = np.asarray(jax.device_get(model.tenor_times))
    distance = np.min(np.abs(values[:, None] - tenor[None, :]), axis=1)
    if np.any(distance > tol):
        raise ValueError("Cashflow tenor is incompatible with the model tenor grid.")


def simulate_short_rate_paths(
    model: VasicekModel | HullWhiteModel | CIRModel | CIRPlusPlusModel,
    law: RatesLaw,
    realization: WienerRealization,
    times: ArrayLike,
    initial_rate: ArrayLike,
    /,
) -> RatesPathBatch:
    """Simulate a fixed-grid one-factor path using an explicit Wiener realization.

    CIR uses full-truncation Euler.  Positivity of model parameters and the Feller
    diagnostic do not falsely imply positivity of an arbitrary discretization.
    """

    if not isinstance(model, (VasicekModel, HullWhiteModel, CIRModel, CIRPlusPlusModel)):
        raise TypeError("model must be a supported one-factor short-rate model.")
    if not isinstance(realization, WienerRealization):
        raise TypeError("realization must be a WienerRealization.")
    state_layout_id = model.state_layout_id
    _require_law(law, state_layout_id=state_layout_id)
    nodes = _strict_grid(times, "times")
    host_nodes = np.asarray(jax.device_get(nodes))
    if host_nodes[0] < realization.support[0] or host_nodes[-1] > realization.support[1]:
        raise ValueError("Simulation time grid lies outside the Wiener support.")
    if realization.noise_shape != (1,):
        raise ValueError("One-factor short-rate simulation requires noise_shape=(1,).")
    initial = jnp.asarray(initial_rate, dtype=float)
    if initial.shape != ():
        raise ValueError("initial_rate must be scalar.")
    invalid_initial = ~jnp.isfinite(initial)
    if isinstance(model, (CIRModel, CIRPlusPlusModel)):
        invalid_initial = invalid_initial | (initial < 0.0)
    initial = eqx.error_if(
        initial,
        invalid_initial,
        "initial_rate must be finite and nonnegative for CIR structures.",
    )
    increments = realization.increments(nodes[:-1], nodes[1:])[..., 0]
    path_count = realization.num_paths
    increments = increments.reshape((path_count, nodes.shape[0] - 1)).T
    durations = jnp.diff(nodes)

    def step(previous: Array, data: tuple[Array, Array, Array]) -> tuple[Array, Array]:
        time, duration, noise = data
        if isinstance(model, VasicekModel):
            drift = model.mean_reversion * (model.long_run_rate - previous)
            next_value = previous + drift * duration + model.volatility * noise
        elif isinstance(model, HullWhiteModel):
            mean = jnp.interp(time, model.mean_times, model.mean_values)
            drift = mean - model.mean_reversion * previous
            next_value = previous + drift * duration + model.volatility * noise
        else:
            base = model.base if isinstance(model, CIRPlusPlusModel) else model
            positive = jnp.maximum(previous, 0.0)
            drift = base.mean_reversion * (base.long_run_rate - positive)
            next_value = (
                previous + drift * duration + base.volatility * jnp.sqrt(positive) * noise
            )
            next_value = jnp.maximum(next_value, 0.0)
        return next_value, next_value

    start = jnp.broadcast_to(initial, (path_count,))
    _, evolved = jax.lax.scan(step, start, (nodes[:-1], durations, increments))
    factor_paths = jnp.concatenate((start[None, :], evolved), axis=0).T
    if isinstance(model, CIRPlusPlusModel):
        shifts = jnp.interp(nodes, model.shift_times, model.shift_values)
        output = factor_paths + shifts[None, :]
    else:
        output = factor_paths
    output = output[..., None]
    valid = jnp.all(jnp.isfinite(output), axis=(1, 2))
    return RatesPathBatch(
        nodes,
        output,
        valid,
        model_id=model.model_id,
        law_id=_law_id(law),
        realization_id=realization.realization_id,
        state_layout_id=state_layout_id,
    )


def simulate_hjm_paths(
    model: FiniteFactorHJMModel,
    law: PricingLaw,
    realization: WienerRealization,
    observation_times: ArrayLike,
    initial_forward_curve: ArrayLike,
    /,
) -> RatesPathBatch:
    """Euler-simulate the finite-factor HJM curve on compatible tenor nodes."""

    if not isinstance(model, FiniteFactorHJMModel):
        raise TypeError("model must be a FiniteFactorHJMModel.")
    if not isinstance(realization, WienerRealization):
        raise TypeError("realization must be a WienerRealization.")
    _require_law(
        law,
        state_layout_id=model.state_layout_id,
        pricing_measure_id=model.pricing_measure_id,
        pricing_only=True,
    )
    nodes = _strict_grid(observation_times, "observation_times")
    require_tenor_compatibility(model, nodes)
    host_nodes = np.asarray(jax.device_get(nodes))
    tenor = np.asarray(jax.device_get(model.tenor_times))
    observation_indices = tuple(
        int(np.argmin(np.abs(tenor - value))) for value in host_nodes[:-1]
    )
    if host_nodes[0] < realization.support[0] or host_nodes[-1] > realization.support[1]:
        raise ValueError("HJM observation grid lies outside the Wiener support.")
    if realization.noise_shape != (model.factor_count,):
        raise ValueError("HJM Wiener noise shape must equal the model factor count.")
    initial = _finite_vector(
        initial_forward_curve, "initial_forward_curve", minimum_size=2
    )
    if initial.shape != model.tenor_times.shape:
        raise ValueError("initial_forward_curve must match the HJM tenor grid.")
    path_count = realization.num_paths
    increments = realization.increments(nodes[:-1], nodes[1:]).reshape(
        (path_count, nodes.shape[0] - 1, model.factor_count)
    )
    current = jnp.broadcast_to(initial, (path_count, initial.shape[0]))
    history = [current]
    for step, observation_index in enumerate(observation_indices):
        duration = nodes[step + 1] - nodes[step]
        drift = hjm_risk_neutral_drift(model, law, observation_index)
        shock = ein.contract("pf,mf->pm", increments[:, step, :], model.volatility)
        current = current + drift[None, :] * duration + shock
        history.append(current)
    values = jnp.stack(history, axis=1)
    valid = jnp.all(jnp.isfinite(values), axis=(1, 2))
    return RatesPathBatch(
        nodes,
        jnp.where(valid[:, None, None], values, 0.0),
        valid,
        model_id=model.model_id,
        law_id=law.law_id,
        realization_id=realization.realization_id,
        state_layout_id=model.state_layout_id,
    )


def simulate_lmm_paths(
    model: LiborMarketModel,
    law: PricingLaw,
    realization: WienerRealization,
    observation_times: ArrayLike,
    initial_forwards: ArrayLike,
    alive_indices: tuple[int, ...],
    /,
) -> RatesPathBatch:
    """Euler-simulate displaced LMM forwards with explicit reset-state indices."""

    if not isinstance(model, LiborMarketModel):
        raise TypeError("model must be a LiborMarketModel.")
    if not isinstance(realization, WienerRealization):
        raise TypeError("realization must be a WienerRealization.")
    _require_law(
        law,
        state_layout_id=model.state_layout_id,
        pricing_measure_id=model.pricing_measure_id,
        pricing_only=True,
    )
    nodes = _strict_grid(observation_times, "observation_times")
    host_nodes = np.asarray(jax.device_get(nodes))
    if host_nodes[0] < realization.support[0] or host_nodes[-1] > realization.support[1]:
        raise ValueError("LMM observation grid lies outside the Wiener support.")
    if realization.noise_shape != (model.factor_count,):
        raise ValueError("LMM Wiener noise shape must equal the model factor count.")
    resets = tuple(alive_indices)
    if (
        len(resets) != nodes.shape[0] - 1
        or any(isinstance(value, bool) or not isinstance(value, int) for value in resets)
        or any(value < 0 or value >= model.forward_count for value in resets)
        or any(right < left for left, right in zip(resets[:-1], resets[1:], strict=True))
    ):
        raise ValueError(
            "alive_indices must be a nondecreasing in-range integer per observation interval."
        )
    initial = _finite_vector(initial_forwards, "initial_forwards")
    if initial.shape != (model.forward_count,):
        raise ValueError("initial_forwards must match the LMM forward-tenor axis.")
    shifted_initial = initial + model.displacements
    shifted_initial = eqx.error_if(
        shifted_initial,
        jnp.any(shifted_initial <= 0.0),
        "Initial displaced forwards must be positive.",
    )
    path_count = realization.num_paths
    increments = realization.increments(nodes[:-1], nodes[1:]).reshape(
        (path_count, nodes.shape[0] - 1, model.factor_count)
    )
    current = jnp.broadcast_to(initial, (path_count, model.forward_count))
    history = [current]
    indices = jnp.arange(model.forward_count)
    for step, alive_index in enumerate(resets):
        duration = nodes[step + 1] - nodes[step]
        drift = lmm_drift(model, law, current, alive_index=alive_index)
        diffusion = ein.contract("pf,if->pi", increments[:, step, :], model.volatility)
        shifted = current + model.displacements
        candidate = current + shifted * (drift * duration + diffusion)
        current = jnp.where(indices[None, :] >= alive_index, candidate, current)
        history.append(current)
    values = jnp.stack(history, axis=1)
    valid = jnp.all(
        jnp.isfinite(values) & (values + model.displacements[None, None, :] > 0.0),
        axis=(1, 2),
    )
    return RatesPathBatch(
        nodes,
        jnp.where(valid[:, None, None], values, 0.0),
        valid,
        model_id=model.model_id,
        law_id=law.law_id,
        realization_id=realization.realization_id,
        state_layout_id=model.state_layout_id,
    )


def rates_model_identity(
    model: VasicekModel
    | HullWhiteModel
    | CIRModel
    | CIRPlusPlusModel
    | FiniteFactorHJMModel
    | LiborMarketModel,
    /,
) -> str:
    """Content address the model's declared identity and fixed public shapes."""

    if isinstance(model, CIRPlusPlusModel):
        shape = (int(model.shift_times.shape[0]),)
    elif isinstance(model, FiniteFactorHJMModel):
        shape = (int(model.tenor_times.shape[0]), model.factor_count)
    elif isinstance(model, LiborMarketModel):
        shape = (model.forward_count, model.factor_count)
    else:
        shape = (1,)
    return canonical_fingerprint(
        {
            "kind": "finance-rates-model",
            "model_id": model.model_id,
            "state_layout_id": model.state_layout_id,
            "currency_id": model.currency_id,
            "shape": shape,
        }
    )


__all__ = [
    "CIRModel",
    "CIRPlusPlusModel",
    "FiniteFactorHJMModel",
    "HullWhiteModel",
    "LiborMarketModel",
    "LMMMeasure",
    "RatesLaw",
    "RatesPathBatch",
    "VasicekModel",
    "cir_plus_plus_shift_integral",
    "cir_plus_plus_zero_coupon_bond",
    "cir_zero_coupon_bond",
    "hjm_risk_neutral_drift",
    "hull_white_zero_coupon_bond",
    "lmm_drift",
    "rates_model_identity",
    "require_tenor_compatibility",
    "simulate_hjm_paths",
    "simulate_lmm_paths",
    "simulate_short_rate_paths",
    "vasicek_zero_coupon_bond",
]
