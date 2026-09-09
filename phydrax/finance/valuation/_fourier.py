#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Characteristic-function Fourier and COS valuation routes."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..contracts._options import OptionType
from ..core._currency import Currency
from ..core._evidence import FinanceEvidenceBinding
from ..core._laws import PricingLaw
from ..models._diffusion import HestonModel
from ._types import ValuationEvidence, ValuationResult


def _kind(value: OptionType | str) -> OptionType:
    if isinstance(value, OptionType):
        return value
    if value == "call":
        return OptionType.CALL
    if value == "put":
        return OptionType.PUT
    raise ValueError("option_type must be call or put.")


def _positive_integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}.")
    return value


class HestonFourierPlan(StrictModule):
    """Carr-Madan quadrature with an explicit admissible damping moment."""

    damping: float = eqx.field(static=True)
    upper_frequency: float = eqx.field(static=True)
    num_nodes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        damping: float = 1.5,
        upper_frequency: float = 150.0,
        num_nodes: int = 4096,
    ):
        damping_ = float(damping)
        upper = float(upper_frequency)
        if not isfinite(damping_) or damping_ <= 0.0:
            raise ValueError("Fourier damping must be finite and strictly positive.")
        if not isfinite(upper) or upper <= 0.0:
            raise ValueError("upper_frequency must be finite and strictly positive.")
        self.damping = damping_
        self.upper_frequency = upper
        self.num_nodes = _positive_integer(num_nodes, "num_nodes", 32)


class HestonCOSPlan(StrictModule):
    """COS density expansion on a cumulant-scaled log-price interval."""

    num_terms: int = eqx.field(static=True)
    truncation_width: float = eqx.field(static=True)

    def __init__(self, *, num_terms: int = 256, truncation_width: float = 12.0):
        width = float(truncation_width)
        if not isfinite(width) or width <= 2.0:
            raise ValueError("truncation_width must be finite and greater than two.")
        self.num_terms = _positive_integer(num_terms, "num_terms", 16)
        self.truncation_width = width


def heston_log_price_characteristic_function(
    model: HestonModel,
    frequency: ArrayLike,
    spot: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
    /,
) -> Array:
    """Stable little-trap Heston characteristic function under supplied Q carry."""

    if not isinstance(model, HestonModel):
        raise TypeError("model must be a HestonModel.")
    u = jnp.asarray(frequency, dtype=complex)
    spot_ = jnp.asarray(spot, dtype=float)
    maturity_ = jnp.asarray(maturity, dtype=float)
    rate_ = jnp.asarray(rate, dtype=float)
    dividend = jnp.asarray(dividend_yield, dtype=float)
    spot_ = eqx.error_if(
        spot_,
        jnp.any(~jnp.isfinite(spot_))
        | jnp.any(spot_ <= 0.0)
        | jnp.any(~jnp.isfinite(maturity_))
        | jnp.any(maturity_ < 0.0)
        | jnp.any(~jnp.isfinite(rate_))
        | jnp.any(~jnp.isfinite(dividend)),
        "Heston characteristic-function market inputs are invalid.",
    )
    kappa = model.mean_reversion
    theta = model.long_run_variance
    xi = model.volatility_of_variance
    rho = model.correlation
    iu = 1j * u
    beta = kappa - rho * xi * iu
    discriminant = jnp.sqrt(beta**2 + xi**2 * (u**2 + iu))
    g = (beta - discriminant) / (beta + discriminant)
    decay = jnp.exp(-discriminant * maturity_)
    denominator = 1.0 - g * decay
    c = iu * (jnp.log(spot_) + (rate_ - dividend) * maturity_) + (
        kappa * theta / xi**2
    ) * ((beta - discriminant) * maturity_ - 2.0 * jnp.log(denominator / (1.0 - g)))
    d = ((beta - discriminant) / xi**2) * (1.0 - decay) / denominator
    value = jnp.exp(c + d * model.initial_variance)
    return eqx.error_if(
        value,
        jnp.any(~jnp.isfinite(value)),
        "Heston characteristic function is non-finite; requested complex moment is not admissible.",
    )


def _market_inputs(spot, strike, maturity, rate, dividend_yield):
    spot_, strike_, maturity_, rate_, dividend = tuple(
        jnp.asarray(value, dtype=float)
        for value in (spot, strike, maturity, rate, dividend_yield)
    )
    if any(value.shape != () for value in (spot_, strike_, maturity_, rate_, dividend)):
        raise ValueError("transform valuation currently requires scalar market inputs.")
    spot_ = eqx.error_if(
        spot_,
        ~jnp.isfinite(spot_)
        | ~jnp.isfinite(strike_)
        | ~jnp.isfinite(maturity_)
        | ~jnp.isfinite(rate_)
        | ~jnp.isfinite(dividend)
        | (spot_ <= 0.0)
        | (strike_ <= 0.0)
        | (maturity_ <= 0.0),
        "transform valuation requires positive spot/strike/maturity and finite carry.",
    )
    return spot_, strike_, maturity_, rate_, dividend


def _bounds(spot, strike, maturity, rate, dividend, kind):
    discount = jnp.exp(-rate * maturity)
    carry = jnp.exp(-dividend * maturity)
    if kind is OptionType.CALL:
        return jnp.maximum(spot * carry - strike * discount, 0.0), spot * carry
    return jnp.maximum(strike * discount - spot * carry, 0.0), strike * discount


def evaluate_heston_fourier(
    model: HestonModel,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    plan: HestonFourierPlan,
    /,
    *,
    dividend_yield: ArrayLike = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    notional: ArrayLike = 1.0,
    currency: Currency | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
    pricing_law: PricingLaw | None = None,
) -> ValuationResult:
    """Carr-Madan damped Fourier integral for a Heston European."""

    if not isinstance(plan, HestonFourierPlan):
        raise TypeError("plan must be a HestonFourierPlan.")
    kind = _kind(option_type)
    spot_, strike_, maturity_, rate_, dividend = _market_inputs(
        spot, strike, maturity, rate, dividend_yield
    )
    notional_ = jnp.asarray(notional, dtype=float)
    if notional_.shape != ():
        raise ValueError("notional must be scalar.")
    notional_ = eqx.error_if(
        notional_,
        ~jnp.isfinite(notional_) | (notional_ <= 0.0),
        "notional must be finite and positive.",
    )
    frequencies = jnp.linspace(0.0, plan.upper_frequency, plan.num_nodes)
    alpha = jnp.asarray(plan.damping, dtype=spot_.dtype)
    shifted = frequencies - 1j * (alpha + 1.0)
    characteristic = heston_log_price_characteristic_function(
        model, shifted, spot_, maturity_, rate_, dividend
    )
    denominator = (
        alpha**2 + alpha - frequencies**2 + 1j * (2.0 * alpha + 1.0) * frequencies
    )
    transform = jnp.exp(-rate_ * maturity_) * characteristic / denominator
    log_strike = jnp.log(strike_)
    integrand = jnp.real(jnp.exp(-1j * frequencies * log_strike) * transform)
    integrand = eqx.error_if(
        integrand,
        jnp.any(~jnp.isfinite(integrand)),
        "Fourier damping requests a non-finite Heston moment.",
    )
    spacing = frequencies[1] - frequencies[0]
    weights = jnp.ones_like(frequencies).at[0].set(0.5).at[-1].set(0.5)
    call = jnp.exp(-alpha * log_strike) * spacing * jnp.sum(weights * integrand) / jnp.pi
    parity = spot_ * jnp.exp(-dividend * maturity_) - strike_ * jnp.exp(
        -rate_ * maturity_
    )
    value = notional_ * (call if kind is OptionType.CALL else call - parity)
    lower, upper = _bounds(spot_, strike_, maturity_, rate_, dividend, kind)
    tail = (
        spacing * jnp.sum(jnp.abs(integrand[-8:])) * jnp.exp(-alpha * log_strike) / jnp.pi
    )
    evidence = ValuationEvidence(
        route="heston-carr-madan-fourier",
        finite=jnp.isfinite(value),
        error_estimate=notional_ * tail,
        binding=evidence_binding,
        pricing_law=pricing_law,
    )
    return ValuationResult(
        value,
        evidence,
        lower_bound=notional_ * lower,
        upper_bound=notional_ * upper,
        currency=currency,
        diagnostics={"tail_bound": tail, "frequency_spacing": spacing},
    )


def _cos_payoff_coefficients(frequencies, lower, upper, log_strike):
    omega = frequencies
    c = log_strike
    d = upper
    angle_d = omega * (d - lower)
    angle_c = omega * (c - lower)
    chi = (
        jnp.exp(d) * (jnp.cos(angle_d) + omega * jnp.sin(angle_d))
        - jnp.exp(c) * (jnp.cos(angle_c) + omega * jnp.sin(angle_c))
    ) / (1.0 + omega**2)
    safe_omega = jnp.where(omega == 0.0, 1.0, omega)
    psi_regular = (jnp.sin(angle_d) - jnp.sin(angle_c)) / safe_omega
    psi = jnp.where(omega == 0.0, d - c, psi_regular)
    return 2.0 / (upper - lower) * (chi - jnp.exp(log_strike) * psi)


def evaluate_heston_cos(
    model: HestonModel,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    plan: HestonCOSPlan,
    /,
    *,
    dividend_yield: ArrayLike = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    notional: ArrayLike = 1.0,
    currency: Currency | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
    pricing_law: PricingLaw | None = None,
) -> ValuationResult:
    """Fang-Oosterlee COS expansion for a Heston European payoff."""

    if not isinstance(plan, HestonCOSPlan):
        raise TypeError("plan must be a HestonCOSPlan.")
    kind = _kind(option_type)
    spot_, strike_, maturity_, rate_, dividend = _market_inputs(
        spot, strike, maturity, rate, dividend_yield
    )
    notional_ = jnp.asarray(notional, dtype=float)
    if notional_.shape != ():
        raise ValueError("notional must be scalar.")
    notional_ = eqx.error_if(
        notional_,
        ~jnp.isfinite(notional_) | (notional_ <= 0.0),
        "notional must be finite and positive.",
    )
    mean = jnp.log(spot_) + (rate_ - dividend - 0.5 * model.long_run_variance) * maturity_
    variance_scale = maturity_ * (model.initial_variance + model.long_run_variance) * 0.5
    standard_deviation = jnp.sqrt(jnp.maximum(variance_scale, jnp.finfo(spot_.dtype).eps))
    lower = mean - plan.truncation_width * standard_deviation
    upper = mean + plan.truncation_width * standard_deviation
    log_strike = jnp.log(strike_)
    spot_ = eqx.error_if(
        spot_,
        (log_strike <= lower) | (log_strike >= upper),
        "COS truncation interval does not contain log strike; increase truncation_width.",
    )
    indices = jnp.arange(plan.num_terms, dtype=spot_.dtype)
    frequencies = indices * jnp.pi / (upper - lower)
    characteristic = heston_log_price_characteristic_function(
        model, frequencies, spot_, maturity_, rate_, dividend
    )
    payoff_coefficients = _cos_payoff_coefficients(frequencies, lower, upper, log_strike)
    terms = (
        jnp.real(characteristic * jnp.exp(-1j * frequencies * lower))
        * payoff_coefficients
    )
    terms = terms.at[0].multiply(0.5)
    call = jnp.exp(-rate_ * maturity_) * jnp.sum(terms)
    parity = spot_ * jnp.exp(-dividend * maturity_) - strike_ * jnp.exp(
        -rate_ * maturity_
    )
    value = notional_ * (call if kind is OptionType.CALL else call - parity)
    no_arb_lower, no_arb_upper = _bounds(spot_, strike_, maturity_, rate_, dividend, kind)
    tail = jnp.exp(-rate_ * maturity_) * jnp.sum(jnp.abs(terms[-8:]))
    evidence = ValuationEvidence(
        route="heston-cos",
        finite=jnp.isfinite(value),
        error_estimate=notional_ * tail,
        binding=evidence_binding,
        pricing_law=pricing_law,
    )
    return ValuationResult(
        value,
        evidence,
        lower_bound=notional_ * no_arb_lower,
        upper_bound=notional_ * no_arb_upper,
        currency=currency,
        diagnostics={
            "truncation_lower": lower,
            "truncation_upper": upper,
            "tail_bound": tail,
        },
    )


__all__ = [
    "HestonCOSPlan",
    "HestonFourierPlan",
    "evaluate_heston_cos",
    "evaluate_heston_fourier",
    "heston_log_price_characteristic_function",
]
