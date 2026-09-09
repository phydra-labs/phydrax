#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Closed-form European and digital valuation routes."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ...nonlinear._scalar import Brent, ScalarRootProblem
from ...nonlinear._types import NonlinearTermination
from ..contracts._options import OptionType
from ..core._currency import Currency
from ..core._evidence import FinanceEvidenceBinding
from ..core._laws import PricingLaw
from ..models._diffusion import BachelierModel, Black76Model, BlackScholesModel
from ._types import ImpliedVolatilityResult, ValuationEvidence, ValuationResult


def _option_type(value: OptionType | str) -> OptionType:
    if isinstance(value, OptionType):
        return value
    if value == "call":
        return OptionType.CALL
    if value == "put":
        return OptionType.PUT
    raise ValueError("option_type must be OptionType.CALL or OptionType.PUT.")


def _inputs(
    underlying: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    /,
    *,
    positive_underlying: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    underlying_, strike_, maturity_, rate_, volatility_ = jnp.broadcast_arrays(
        jnp.asarray(underlying, dtype=float),
        jnp.asarray(strike, dtype=float),
        jnp.asarray(maturity, dtype=float),
        jnp.asarray(rate, dtype=float),
        jnp.asarray(volatility, dtype=float),
    )
    invalid = (
        ~jnp.isfinite(underlying_)
        | ~jnp.isfinite(strike_)
        | ~jnp.isfinite(maturity_)
        | ~jnp.isfinite(rate_)
        | ~jnp.isfinite(volatility_)
        | (strike_ <= 0.0)
        | (maturity_ < 0.0)
        | (volatility_ <= 0.0)
    )
    if positive_underlying:
        invalid = invalid | (underlying_ <= 0.0)
    underlying_ = eqx.error_if(
        underlying_,
        jnp.any(invalid),
        "analytic valuation inputs are outside their admissible domain.",
    )
    return underlying_, strike_, maturity_, rate_, volatility_


def _normal_pdf(value: Array) -> Array:
    return jnp.exp(-0.5 * value**2) / jnp.sqrt(2.0 * jnp.pi)


def _black_price(
    forward: Array,
    strike: Array,
    maturity: Array,
    volatility: Array,
    discount: Array,
    option_type: OptionType,
) -> Array:
    root_time = jnp.sqrt(jnp.maximum(maturity, jnp.finfo(forward.dtype).tiny))
    total = volatility * root_time
    d1 = (jnp.log(forward / strike) + 0.5 * total**2) / total
    d2 = d1 - total
    sign = 1.0 if option_type is OptionType.CALL else -1.0
    regular = (
        discount
        * sign
        * (forward * jsp.special.ndtr(sign * d1) - strike * jsp.special.ndtr(sign * d2))
    )
    intrinsic = discount * jnp.maximum(sign * (forward - strike), 0.0)
    return jnp.where(maturity == 0.0, intrinsic, regular)


def _bachelier_price(
    forward: Array,
    strike: Array,
    maturity: Array,
    volatility: Array,
    discount: Array,
    option_type: OptionType,
) -> Array:
    root_time = jnp.sqrt(jnp.maximum(maturity, jnp.finfo(forward.dtype).tiny))
    scale = volatility * root_time
    d = (forward - strike) / scale
    sign = 1.0 if option_type is OptionType.CALL else -1.0
    regular = discount * (
        sign * (forward - strike) * jsp.special.ndtr(sign * d) + scale * _normal_pdf(d)
    )
    intrinsic = discount * jnp.maximum(sign * (forward - strike), 0.0)
    return jnp.where(maturity == 0.0, intrinsic, regular)


def _result(
    value: Array,
    lower: Array,
    upper: Array,
    route: str,
    /,
    *,
    currency: Currency | None,
    evidence_binding: FinanceEvidenceBinding | None,
    pricing_law: PricingLaw | None,
) -> ValuationResult:
    evidence = ValuationEvidence(
        route=route,
        finite=jnp.all(jnp.isfinite(value)),
        binding=evidence_binding,
        pricing_law=pricing_law,
    )
    return ValuationResult(
        value, evidence, lower_bound=lower, upper_bound=upper, currency=currency
    )


def evaluate_black_scholes_european(
    model: BlackScholesModel,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    /,
    *,
    dividend_yield: ArrayLike = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    notional: ArrayLike = 1.0,
    currency: Currency | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
    pricing_law: PricingLaw | None = None,
) -> ValuationResult:
    """Value a cash-settled European under constant lognormal diffusion."""

    if not isinstance(model, BlackScholesModel):
        raise TypeError("model must be a BlackScholesModel.")
    kind = _option_type(option_type)
    spot_, strike_, maturity_, rate_, volatility = _inputs(
        spot, strike, maturity, rate, model.volatility, positive_underlying=True
    )
    dividend = jnp.broadcast_to(jnp.asarray(dividend_yield, dtype=float), spot_.shape)
    notional_ = jnp.broadcast_to(jnp.asarray(notional, dtype=float), spot_.shape)
    spot_ = eqx.error_if(
        spot_,
        jnp.any(~jnp.isfinite(dividend))
        | jnp.any(~jnp.isfinite(notional_))
        | jnp.any(notional_ <= 0.0),
        "dividend_yield must be finite and notional finite and positive.",
    )
    discount = jnp.exp(-rate_ * maturity_)
    carry_discount = jnp.exp(-dividend * maturity_)
    forward = spot_ * carry_discount / discount
    value = notional_ * _black_price(
        forward, strike_, maturity_, volatility, discount, kind
    )
    call_lower = jnp.maximum(spot_ * carry_discount - strike_ * discount, 0.0)
    put_lower = jnp.maximum(strike_ * discount - spot_ * carry_discount, 0.0)
    lower = notional_ * (call_lower if kind is OptionType.CALL else put_lower)
    upper = notional_ * (
        spot_ * carry_discount if kind is OptionType.CALL else strike_ * discount
    )
    return _result(
        value,
        lower,
        upper,
        "black-scholes-analytic",
        currency=currency,
        evidence_binding=evidence_binding,
        pricing_law=pricing_law,
    )


def evaluate_black76_european(
    model: Black76Model,
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount_factor: ArrayLike,
    /,
    *,
    option_type: OptionType | str = OptionType.CALL,
    notional: ArrayLike = 1.0,
    currency: Currency | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
    pricing_law: PricingLaw | None = None,
) -> ValuationResult:
    if not isinstance(model, Black76Model):
        raise TypeError("model must be a Black76Model.")
    kind = _option_type(option_type)
    forward_, strike_, maturity_, _, volatility = _inputs(
        forward, strike, maturity, 0.0, model.volatility, positive_underlying=True
    )
    discount = jnp.broadcast_to(jnp.asarray(discount_factor, dtype=float), forward_.shape)
    notional_ = jnp.broadcast_to(jnp.asarray(notional, dtype=float), forward_.shape)
    forward_ = eqx.error_if(
        forward_,
        jnp.any(~jnp.isfinite(discount))
        | jnp.any((discount <= 0.0) | (discount > 1.0))
        | jnp.any(~jnp.isfinite(notional_))
        | jnp.any(notional_ <= 0.0),
        "discount_factor must lie in (0, 1] and notional must be positive.",
    )
    value = notional_ * _black_price(
        forward_, strike_, maturity_, volatility, discount, kind
    )
    sign = 1.0 if kind is OptionType.CALL else -1.0
    lower = notional_ * discount * jnp.maximum(sign * (forward_ - strike_), 0.0)
    upper = notional_ * discount * (forward_ if kind is OptionType.CALL else strike_)
    return _result(
        value,
        lower,
        upper,
        "black76-analytic",
        currency=currency,
        evidence_binding=evidence_binding,
        pricing_law=pricing_law,
    )


def evaluate_bachelier_european(
    model: BachelierModel,
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount_factor: ArrayLike,
    /,
    *,
    option_type: OptionType | str = OptionType.CALL,
    notional: ArrayLike = 1.0,
    currency: Currency | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
    pricing_law: PricingLaw | None = None,
) -> ValuationResult:
    if not isinstance(model, BachelierModel):
        raise TypeError("model must be a BachelierModel.")
    kind = _option_type(option_type)
    forward_, strike_, maturity_, _, volatility = _inputs(
        forward, strike, maturity, 0.0, model.volatility, positive_underlying=False
    )
    discount = jnp.broadcast_to(jnp.asarray(discount_factor, dtype=float), forward_.shape)
    notional_ = jnp.broadcast_to(jnp.asarray(notional, dtype=float), forward_.shape)
    forward_ = eqx.error_if(
        forward_,
        jnp.any(~jnp.isfinite(discount))
        | jnp.any((discount <= 0.0) | (discount > 1.0))
        | jnp.any(~jnp.isfinite(notional_))
        | jnp.any(notional_ <= 0.0),
        "discount_factor must lie in (0, 1] and notional must be positive.",
    )
    value = notional_ * _bachelier_price(
        forward_, strike_, maturity_, volatility, discount, kind
    )
    sign = 1.0 if kind is OptionType.CALL else -1.0
    lower = notional_ * discount * jnp.maximum(sign * (forward_ - strike_), 0.0)
    upper = jnp.full_like(lower, jnp.inf)
    evidence = ValuationEvidence(
        route="bachelier-analytic",
        finite=jnp.all(jnp.isfinite(value)),
        binding=evidence_binding,
        pricing_law=pricing_law,
    )
    return ValuationResult(
        value, evidence, currency=currency, diagnostics={"intrinsic_lower_bound": lower}
    )


def evaluate_black_scholes_digital(
    model: BlackScholesModel,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    /,
    *,
    dividend_yield: ArrayLike = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    cash_amount: ArrayLike = 1.0,
    currency: Currency | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
    pricing_law: PricingLaw | None = None,
) -> ValuationResult:
    if not isinstance(model, BlackScholesModel):
        raise TypeError("model must be a BlackScholesModel.")
    kind = _option_type(option_type)
    spot_, strike_, maturity_, rate_, volatility = _inputs(
        spot, strike, maturity, rate, model.volatility, positive_underlying=True
    )
    dividend = jnp.broadcast_to(jnp.asarray(dividend_yield, dtype=float), spot_.shape)
    cash = jnp.broadcast_to(jnp.asarray(cash_amount, dtype=float), spot_.shape)
    spot_ = eqx.error_if(
        spot_,
        jnp.any(~jnp.isfinite(dividend))
        | jnp.any(~jnp.isfinite(cash))
        | jnp.any(cash < 0.0),
        "digital carry and cash amount must be finite; cash must be non-negative.",
    )
    root_time = jnp.sqrt(jnp.maximum(maturity_, jnp.finfo(spot_.dtype).tiny))
    d2 = (
        jnp.log(spot_ / strike_) + (rate_ - dividend - 0.5 * volatility**2) * maturity_
    ) / (volatility * root_time)
    sign = 1.0 if kind is OptionType.CALL else -1.0
    terminal = jnp.where(sign * (spot_ - strike_) > 0.0, cash, 0.0)
    value = jnp.where(
        maturity_ == 0.0,
        terminal,
        cash * jnp.exp(-rate_ * maturity_) * jsp.special.ndtr(sign * d2),
    )
    lower = jnp.zeros_like(value)
    upper = cash * jnp.exp(-rate_ * maturity_)
    return _result(
        value,
        lower,
        upper,
        "black-scholes-digital-analytic",
        currency=currency,
        evidence_binding=evidence_binding,
        pricing_law=pricing_law,
    )


def evaluate_black76_digital(
    model: Black76Model,
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount_factor: ArrayLike,
    /,
    *,
    option_type: OptionType | str = OptionType.CALL,
    cash_amount: ArrayLike = 1.0,
) -> ValuationResult:
    if not isinstance(model, Black76Model):
        raise TypeError("model must be a Black76Model.")
    kind = _option_type(option_type)
    forward_, strike_, maturity_, _, volatility = _inputs(
        forward, strike, maturity, 0.0, model.volatility, positive_underlying=True
    )
    discount = jnp.asarray(discount_factor, dtype=float)
    cash = jnp.asarray(cash_amount, dtype=float)
    forward_, discount, cash = jnp.broadcast_arrays(forward_, discount, cash)
    forward_ = eqx.error_if(
        forward_,
        jnp.any(~jnp.isfinite(discount))
        | jnp.any((discount <= 0.0) | (discount > 1.0))
        | jnp.any(~jnp.isfinite(cash))
        | jnp.any(cash < 0.0),
        "digital discount/cash inputs are invalid.",
    )
    root_time = jnp.sqrt(jnp.maximum(maturity_, jnp.finfo(forward_.dtype).tiny))
    d2 = (jnp.log(forward_ / strike_) - 0.5 * volatility**2 * maturity_) / (
        volatility * root_time
    )
    sign = 1.0 if kind is OptionType.CALL else -1.0
    value = jnp.where(
        maturity_ == 0.0,
        cash * (sign * (forward_ - strike_) > 0.0),
        cash * discount * jsp.special.ndtr(sign * d2),
    )
    return _result(
        value,
        jnp.zeros_like(value),
        cash * discount,
        "black76-digital-analytic",
        currency=None,
        evidence_binding=None,
        pricing_law=None,
    )


def evaluate_bachelier_digital(
    model: BachelierModel,
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount_factor: ArrayLike,
    /,
    *,
    option_type: OptionType | str = OptionType.CALL,
    cash_amount: ArrayLike = 1.0,
) -> ValuationResult:
    if not isinstance(model, BachelierModel):
        raise TypeError("model must be a BachelierModel.")
    kind = _option_type(option_type)
    forward_, strike_, maturity_, _, volatility = _inputs(
        forward, strike, maturity, 0.0, model.volatility, positive_underlying=False
    )
    discount = jnp.asarray(discount_factor, dtype=float)
    cash = jnp.asarray(cash_amount, dtype=float)
    forward_, discount, cash = jnp.broadcast_arrays(forward_, discount, cash)
    forward_ = eqx.error_if(
        forward_,
        jnp.any(~jnp.isfinite(discount))
        | jnp.any((discount <= 0.0) | (discount > 1.0))
        | jnp.any(~jnp.isfinite(cash))
        | jnp.any(cash < 0.0),
        "digital discount/cash inputs are invalid.",
    )
    root_time = jnp.sqrt(jnp.maximum(maturity_, jnp.finfo(forward_.dtype).tiny))
    d = (forward_ - strike_) / (volatility * root_time)
    sign = 1.0 if kind is OptionType.CALL else -1.0
    value = jnp.where(
        maturity_ == 0.0,
        cash * (sign * (forward_ - strike_) > 0.0),
        cash * discount * jsp.special.ndtr(sign * d),
    )
    return _result(
        value,
        jnp.zeros_like(value),
        cash * discount,
        "bachelier-digital-analytic",
        currency=None,
        evidence_binding=None,
        pricing_law=None,
    )


def _implied(
    price: ArrayLike,
    lower_price: Array,
    upper_price: Array | None,
    price_function,
    upper_volatility: Array,
    model_name: str,
    /,
    *,
    tolerance: float,
    maximum_steps: int,
) -> ImpliedVolatilityResult:
    target = jnp.asarray(price, dtype=float)
    if target.shape != ():
        raise ValueError("implied-volatility inversion requires a scalar price.")
    invalid = ~jnp.isfinite(target) | (target < lower_price)
    if upper_price is not None:
        invalid = invalid | (target > upper_price)
    target = eqx.error_if(
        target, invalid, "option price violates model-free no-arbitrage bounds."
    )
    lower_volatility = jnp.asarray(1.0e-10, dtype=target.dtype)
    problem = ScalarRootProblem(
        lambda volatility, _: price_function(volatility) - target,
        bracket=(lower_volatility, upper_volatility),
        problem_id=f"{model_name}-implied-volatility",
    )
    root = Brent().solve(
        problem,
        termination=NonlinearTermination(
            absolute_residual=tolerance,
            relative_residual=tolerance,
            absolute_step=tolerance,
            relative_step=tolerance,
            maximum_steps=maximum_steps,
        ),
    )
    volatility = eqx.error_if(
        root.root,
        ~root.successful,
        "implied-volatility root solve did not converge inside its admissible bracket.",
    )
    return ImpliedVolatilityResult(
        volatility,
        root.value,
        root.lower,
        root.upper,
        root.nonlinear_result.diagnostics.iterations,
        root.successful,
        model_name,
    )


def invert_black_scholes_implied_volatility(
    price: ArrayLike,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    /,
    *,
    dividend_yield: ArrayLike = 0.0,
    option_type: OptionType | str = OptionType.CALL,
    tolerance: float = 1.0e-10,
    maximum_steps: int = 100,
) -> ImpliedVolatilityResult:
    kind = _option_type(option_type)
    spot_, strike_, maturity_, rate_, _ = _inputs(
        spot, strike, maturity, rate, 1.0, positive_underlying=True
    )
    if spot_.shape != ():
        raise ValueError("implied-volatility inversion requires scalar inputs.")
    maturity_ = eqx.error_if(
        maturity_, maturity_ <= 0.0, "implied volatility is undefined at zero maturity."
    )
    dividend = jnp.asarray(dividend_yield, dtype=float)
    discount, carry = jnp.exp(-rate_ * maturity_), jnp.exp(-dividend * maturity_)
    forward = spot_ * carry / discount
    sign = 1.0 if kind is OptionType.CALL else -1.0
    lower = discount * jnp.maximum(sign * (forward - strike_), 0.0)
    upper = spot_ * carry if kind is OptionType.CALL else strike_ * discount
    return _implied(
        price,
        lower,
        upper,
        lambda volatility: _black_price(
            forward, strike_, maturity_, volatility, discount, kind
        ),
        jnp.asarray(10.0),
        "black-scholes",
        tolerance=tolerance,
        maximum_steps=maximum_steps,
    )


def invert_black76_implied_volatility(
    price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount_factor: ArrayLike,
    /,
    *,
    option_type: OptionType | str = OptionType.CALL,
    tolerance: float = 1.0e-10,
    maximum_steps: int = 100,
) -> ImpliedVolatilityResult:
    kind = _option_type(option_type)
    forward_, strike_, maturity_, _, _ = _inputs(
        forward, strike, maturity, 0.0, 1.0, positive_underlying=True
    )
    if forward_.shape != ():
        raise ValueError("implied-volatility inversion requires scalar inputs.")
    maturity_ = eqx.error_if(
        maturity_, maturity_ <= 0.0, "implied volatility is undefined at zero maturity."
    )
    discount = jnp.asarray(discount_factor, dtype=float)
    discount = eqx.error_if(
        discount,
        ~jnp.isfinite(discount) | (discount <= 0.0) | (discount > 1.0),
        "discount_factor must lie in (0, 1].",
    )
    sign = 1.0 if kind is OptionType.CALL else -1.0
    lower = discount * jnp.maximum(sign * (forward_ - strike_), 0.0)
    upper = discount * (forward_ if kind is OptionType.CALL else strike_)
    return _implied(
        price,
        lower,
        upper,
        lambda volatility: _black_price(
            forward_, strike_, maturity_, volatility, discount, kind
        ),
        jnp.asarray(10.0),
        "black76",
        tolerance=tolerance,
        maximum_steps=maximum_steps,
    )


def invert_bachelier_implied_volatility(
    price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount_factor: ArrayLike,
    /,
    *,
    option_type: OptionType | str = OptionType.CALL,
    tolerance: float = 1.0e-10,
    maximum_steps: int = 100,
) -> ImpliedVolatilityResult:
    kind = _option_type(option_type)
    forward_, strike_, maturity_, _, _ = _inputs(
        forward, strike, maturity, 0.0, 1.0, positive_underlying=False
    )
    if forward_.shape != ():
        raise ValueError("implied-volatility inversion requires scalar inputs.")
    maturity_ = eqx.error_if(
        maturity_, maturity_ <= 0.0, "implied volatility is undefined at zero maturity."
    )
    discount = jnp.asarray(discount_factor, dtype=float)
    discount = eqx.error_if(
        discount,
        ~jnp.isfinite(discount) | (discount <= 0.0) | (discount > 1.0),
        "discount_factor must lie in (0, 1].",
    )
    sign = 1.0 if kind is OptionType.CALL else -1.0
    lower = discount * jnp.maximum(sign * (forward_ - strike_), 0.0)
    target = jnp.asarray(price, dtype=float)
    upper_volatility = (
        16.0
        * (jnp.abs(forward_) + jnp.abs(strike_) + jnp.abs(target / discount) + 1.0)
        / jnp.sqrt(maturity_)
    )
    return _implied(
        price,
        lower,
        None,
        lambda volatility: _bachelier_price(
            forward_, strike_, maturity_, volatility, discount, kind
        ),
        upper_volatility,
        "bachelier",
        tolerance=tolerance,
        maximum_steps=maximum_steps,
    )


__all__ = [
    "evaluate_bachelier_digital",
    "evaluate_bachelier_european",
    "evaluate_black76_digital",
    "evaluate_black76_european",
    "evaluate_black_scholes_digital",
    "evaluate_black_scholes_european",
    "invert_bachelier_implied_volatility",
    "invert_black76_implied_volatility",
    "invert_black_scholes_implied_volatility",
]
