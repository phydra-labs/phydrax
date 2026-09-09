#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resolved equity, FX, and commodity forwards and commodity futures."""

from __future__ import annotations

from math import isfinite
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..core import Currency, FinanceDate, FXPair
from ._base import AbstractResolvedContract
from ._cashflows import CashflowBatch
from ._rates import DeterministicCashflowReplay, PayReceive


if TYPE_CHECKING:
    from ..curves._core import CurveSet


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be a non-empty string.")
    return identifier


def _finite(value: float, name: str, /) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _positive(value: float, name: str, /) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return number


def _direction(value: PayReceive | str, /) -> PayReceive:
    if isinstance(value, PayReceive):
        return value
    if value not in (PayReceive.PAY.value, PayReceive.RECEIVE.value):
        raise ValueError("pay_receive must be 'pay' or 'receive'.")
    return PayReceive(value)


def _settlement_fixing(
    settlement_price: ArrayLike,
    settlement_price_known: bool,
    /,
) -> Array:
    if type(settlement_price_known) is not bool:
        raise TypeError("settlement_price_known must be bool.")
    value = jnp.asarray(settlement_price)
    if value.shape != ():
        raise ValueError("settlement_price must be scalar.")
    concrete = float(np.asarray(value))
    if settlement_price_known and not isfinite(concrete):
        raise ValueError("A known settlement price must be finite.")
    if not settlement_price_known and concrete != 0.0:
        raise ValueError("An unknown settlement price must use neutral zero.")
    return value


def _dates(
    valuation_date: FinanceDate,
    maturity_date: FinanceDate,
    maturity_time: float,
    /,
) -> float:
    if not isinstance(valuation_date, FinanceDate) or not isinstance(
        maturity_date, FinanceDate
    ):
        raise TypeError("valuation_date and maturity_date must be FinanceDate values.")
    if maturity_date.ordinal <= valuation_date.ordinal:
        raise ValueError("maturity_date must be later than valuation_date.")
    return _positive(maturity_time, "maturity_time")


def _curve(curves: CurveSet, curve_id: str, valuation_date: FinanceDate, /):
    curve = curves.curve(curve_id)
    if curve.definition.valuation_date.ordinal != valuation_date.ordinal:
        raise ValueError("Product and curve valuation dates must match.")
    return curve


def _known_settlement(
    *,
    contract_id: str,
    maturity_date: FinanceDate,
    amount: Array,
    currency: Currency,
    known: bool,
) -> CashflowBatch:
    if not known:
        return CashflowBatch((), jnp.zeros((0,)), ())
    return CashflowBatch(
        (maturity_date,),
        jnp.asarray((amount,)),
        (currency,),
        obligation_ids=(f"{contract_id}:settlement",),
    )


def _single_replay(
    *,
    contract_id: str,
    maturity_date: FinanceDate,
    maturity_time: float,
    amount: Array,
    currency: Currency,
    known: bool,
    discount_factor: Array,
    curve_ids: tuple[str, ...],
) -> DeterministicCashflowReplay:
    valid = jnp.asarray((True,))
    return DeterministicCashflowReplay(
        payment_ordinals=jnp.asarray((maturity_date.ordinal,), dtype=jnp.int32),
        payment_times=jnp.asarray((maturity_time,)),
        amounts=jnp.asarray((amount,)),
        slot_currencies=(currency,),
        valid_mask=valid,
        known_mask=jnp.asarray((known,), dtype=bool),
        projected_mask=jnp.asarray((not known,), dtype=bool),
        notional_exchange_mask=jnp.zeros((1,), dtype=bool),
        discount_factors=jnp.asarray((discount_factor,)),
        obligation_ids=(f"{contract_id}:settlement",),
        curve_ids=curve_ids,
    )


def _resolved_id(kind: str, facts: dict[str, Any], /) -> str:
    return canonical_fingerprint({"kind": kind, **facts})


class ResolvedEquityForward(AbstractResolvedContract):
    """Equity forward with separate financing and dividend curves."""

    settlement_price: Array
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    valuation_date: FinanceDate = eqx.field(static=True)
    maturity_date: FinanceDate = eqx.field(static=True)
    maturity_time: float = eqx.field(static=True)
    units: float = eqx.field(static=True)
    delivery_price: float = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    dividend_curve_id: str = eqx.field(static=True)
    pay_receive: PayReceive = eqx.field(static=True)
    settlement_price_known: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        currency: Currency,
        valuation_date: FinanceDate,
        maturity_date: FinanceDate,
        maturity_time: float,
        units: float,
        delivery_price: float,
        discount_curve_id: str,
        dividend_curve_id: str,
        pay_receive: PayReceive | str,
        settlement_price: ArrayLike = 0.0,
        settlement_price_known: bool,
    ):
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        maturity = _dates(valuation_date, maturity_date, maturity_time)
        contract = _identifier(contract_id, "contract_id")
        self.settlement_price = _settlement_fixing(
            settlement_price, settlement_price_known
        )
        self.contract_id = contract
        self.currency = currency
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.maturity_time = maturity
        self.units = _positive(units, "units")
        self.delivery_price = _positive(delivery_price, "delivery_price")
        self.discount_curve_id = _identifier(discount_curve_id, "discount_curve_id")
        self.dividend_curve_id = _identifier(dividend_curve_id, "dividend_curve_id")
        self.pay_receive = _direction(pay_receive)
        self.settlement_price_known = settlement_price_known
        self.resolved_id = _resolved_id(
            "resolved-equity-forward",
            {
                "contract_id": contract,
                "currency": currency.currency_id,
                "maturity": maturity_date.ordinal,
                "units": self.units,
                "delivery_price": self.delivery_price,
                "discount_curve": self.discount_curve_id,
                "dividend_curve": self.dividend_curve_id,
                "direction": self.pay_receive.value,
                "settlement_known": settlement_price_known,
            },
        )

    def forward_price(self, curves: CurveSet, spot_price: ArrayLike, /) -> Array:
        spot = jnp.asarray(spot_price)
        if spot.shape != ():
            raise ValueError("spot_price must be scalar.")
        spot = eqx.error_if(
            spot,
            (~jnp.isfinite(spot)) | (spot <= 0.0),
            "spot_price must be finite and positive.",
        )
        discount = _curve(
            curves, self.discount_curve_id, self.valuation_date
        ).discount_factor(self.maturity_time)
        dividend = _curve(
            curves, self.dividend_curve_id, self.valuation_date
        ).discount_factor(self.maturity_time)
        return spot * dividend / discount

    def settlement_amount(self, curves: CurveSet, spot_price: ArrayLike, /) -> Array:
        price = (
            self.settlement_price
            if self.settlement_price_known
            else self.forward_price(curves, spot_price)
        )
        return self.pay_receive.sign * self.units * (price - self.delivery_price)

    @property
    def known_cashflows(self) -> CashflowBatch:
        amount = (
            self.pay_receive.sign
            * self.units
            * (self.settlement_price - self.delivery_price)
        )
        return _known_settlement(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            amount=amount,
            currency=self.currency,
            known=self.settlement_price_known,
        )

    def cashflow_replay(
        self, curves: CurveSet, spot_price: ArrayLike, /
    ) -> DeterministicCashflowReplay:
        discount = _curve(
            curves, self.discount_curve_id, self.valuation_date
        ).discount_factor(self.maturity_time)
        return _single_replay(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            maturity_time=self.maturity_time,
            amount=self.settlement_amount(curves, spot_price),
            currency=self.currency,
            known=self.settlement_price_known,
            discount_factor=discount,
            curve_ids=(self.discount_curve_id, self.dividend_curve_id),
        )

    def present_value(self, curves: CurveSet, spot_price: ArrayLike, /) -> Array:
        return self.cashflow_replay(curves, spot_price).present_value(self.currency)


class ResolvedFXForward(AbstractResolvedContract):
    """Deliverable FX forward quoted as quote currency per unit base currency."""

    settlement_rate: Array
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    fx_pair: FXPair = eqx.field(static=True)
    valuation_date: FinanceDate = eqx.field(static=True)
    maturity_date: FinanceDate = eqx.field(static=True)
    maturity_time: float = eqx.field(static=True)
    base_notional: float = eqx.field(static=True)
    delivery_rate: float = eqx.field(static=True)
    quote_discount_curve_id: str = eqx.field(static=True)
    base_discount_curve_id: str = eqx.field(static=True)
    pay_receive: PayReceive = eqx.field(static=True)
    settlement_rate_known: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        fx_pair: FXPair,
        valuation_date: FinanceDate,
        maturity_date: FinanceDate,
        maturity_time: float,
        base_notional: float,
        delivery_rate: float,
        quote_discount_curve_id: str,
        base_discount_curve_id: str,
        pay_receive: PayReceive | str,
        settlement_rate: ArrayLike = 0.0,
        settlement_rate_known: bool,
    ):
        if not isinstance(fx_pair, FXPair):
            raise TypeError("fx_pair must be an FXPair.")
        maturity = _dates(valuation_date, maturity_date, maturity_time)
        contract = _identifier(contract_id, "contract_id")
        self.settlement_rate = _settlement_fixing(settlement_rate, settlement_rate_known)
        self.contract_id = contract
        self.fx_pair = fx_pair
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.maturity_time = maturity
        self.base_notional = _positive(base_notional, "base_notional")
        self.delivery_rate = _positive(delivery_rate, "delivery_rate")
        self.quote_discount_curve_id = _identifier(
            quote_discount_curve_id, "quote_discount_curve_id"
        )
        self.base_discount_curve_id = _identifier(
            base_discount_curve_id, "base_discount_curve_id"
        )
        self.pay_receive = _direction(pay_receive)
        self.settlement_rate_known = settlement_rate_known
        self.resolved_id = _resolved_id(
            "resolved-fx-forward",
            {
                "contract_id": contract,
                "fx_pair": fx_pair.pair_id,
                "maturity": maturity_date.ordinal,
                "base_notional": self.base_notional,
                "delivery_rate": self.delivery_rate,
                "quote_discount_curve": self.quote_discount_curve_id,
                "base_discount_curve": self.base_discount_curve_id,
                "direction": self.pay_receive.value,
                "settlement_known": settlement_rate_known,
            },
        )

    def forward_rate(self, curves: CurveSet, spot_quote_per_base: ArrayLike, /) -> Array:
        spot = jnp.asarray(spot_quote_per_base)
        if spot.shape != ():
            raise ValueError("spot_quote_per_base must be scalar.")
        spot = eqx.error_if(
            spot,
            (~jnp.isfinite(spot)) | (spot <= 0.0),
            "spot_quote_per_base must be finite and positive.",
        )
        quote_discount = _curve(
            curves, self.quote_discount_curve_id, self.valuation_date
        ).discount_factor(self.maturity_time)
        base_discount = _curve(
            curves, self.base_discount_curve_id, self.valuation_date
        ).discount_factor(self.maturity_time)
        return spot * base_discount / quote_discount

    def settlement_amount(
        self, curves: CurveSet, spot_quote_per_base: ArrayLike, /
    ) -> Array:
        rate = (
            self.settlement_rate
            if self.settlement_rate_known
            else self.forward_rate(curves, spot_quote_per_base)
        )
        return self.pay_receive.sign * self.base_notional * (rate - self.delivery_rate)

    @property
    def known_cashflows(self) -> CashflowBatch:
        amount = (
            self.pay_receive.sign
            * self.base_notional
            * (self.settlement_rate - self.delivery_rate)
        )
        return _known_settlement(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            amount=amount,
            currency=self.fx_pair.quote,
            known=self.settlement_rate_known,
        )

    def cashflow_replay(
        self, curves: CurveSet, spot_quote_per_base: ArrayLike, /
    ) -> DeterministicCashflowReplay:
        discount = _curve(
            curves, self.quote_discount_curve_id, self.valuation_date
        ).discount_factor(self.maturity_time)
        return _single_replay(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            maturity_time=self.maturity_time,
            amount=self.settlement_amount(curves, spot_quote_per_base),
            currency=self.fx_pair.quote,
            known=self.settlement_rate_known,
            discount_factor=discount,
            curve_ids=(
                self.quote_discount_curve_id,
                self.base_discount_curve_id,
            ),
        )

    def present_value(self, curves: CurveSet, spot_quote_per_base: ArrayLike, /) -> Array:
        return self.cashflow_replay(curves, spot_quote_per_base).present_value(
            self.fx_pair.quote
        )


class ResolvedCommodityForward(AbstractResolvedContract):
    """Commodity forward with an explicit net-benefit carry curve."""

    settlement_price: Array
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    valuation_date: FinanceDate = eqx.field(static=True)
    maturity_date: FinanceDate = eqx.field(static=True)
    maturity_time: float = eqx.field(static=True)
    quantity: float = eqx.field(static=True)
    delivery_price: float = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    net_benefit_curve_id: str = eqx.field(static=True)
    pay_receive: PayReceive = eqx.field(static=True)
    settlement_price_known: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        currency: Currency,
        valuation_date: FinanceDate,
        maturity_date: FinanceDate,
        maturity_time: float,
        quantity: float,
        delivery_price: float,
        discount_curve_id: str,
        net_benefit_curve_id: str,
        pay_receive: PayReceive | str,
        settlement_price: ArrayLike = 0.0,
        settlement_price_known: bool,
    ):
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        maturity = _dates(valuation_date, maturity_date, maturity_time)
        contract = _identifier(contract_id, "contract_id")
        self.settlement_price = _settlement_fixing(
            settlement_price, settlement_price_known
        )
        self.contract_id = contract
        self.currency = currency
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.maturity_time = maturity
        self.quantity = _positive(quantity, "quantity")
        self.delivery_price = _positive(delivery_price, "delivery_price")
        self.discount_curve_id = _identifier(discount_curve_id, "discount_curve_id")
        self.net_benefit_curve_id = _identifier(
            net_benefit_curve_id, "net_benefit_curve_id"
        )
        self.pay_receive = _direction(pay_receive)
        self.settlement_price_known = settlement_price_known
        self.resolved_id = _resolved_id(
            "resolved-commodity-forward",
            {
                "contract_id": contract,
                "currency": currency.currency_id,
                "maturity": maturity_date.ordinal,
                "quantity": self.quantity,
                "delivery_price": self.delivery_price,
                "discount_curve": self.discount_curve_id,
                "net_benefit_curve": self.net_benefit_curve_id,
                "direction": self.pay_receive.value,
                "settlement_known": settlement_price_known,
            },
        )

    def forward_price(self, curves: CurveSet, spot_price: ArrayLike, /) -> Array:
        spot = jnp.asarray(spot_price)
        if spot.shape != ():
            raise ValueError("spot_price must be scalar.")
        spot = eqx.error_if(
            spot,
            (~jnp.isfinite(spot)) | (spot <= 0.0),
            "spot_price must be finite and positive.",
        )
        discount = _curve(
            curves, self.discount_curve_id, self.valuation_date
        ).discount_factor(self.maturity_time)
        net_benefit = _curve(
            curves, self.net_benefit_curve_id, self.valuation_date
        ).discount_factor(self.maturity_time)
        return spot * net_benefit / discount

    def settlement_amount(self, curves: CurveSet, spot_price: ArrayLike, /) -> Array:
        price = (
            self.settlement_price
            if self.settlement_price_known
            else self.forward_price(curves, spot_price)
        )
        return self.pay_receive.sign * self.quantity * (price - self.delivery_price)

    @property
    def known_cashflows(self) -> CashflowBatch:
        amount = (
            self.pay_receive.sign
            * self.quantity
            * (self.settlement_price - self.delivery_price)
        )
        return _known_settlement(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            amount=amount,
            currency=self.currency,
            known=self.settlement_price_known,
        )

    def cashflow_replay(
        self, curves: CurveSet, spot_price: ArrayLike, /
    ) -> DeterministicCashflowReplay:
        discount = _curve(
            curves, self.discount_curve_id, self.valuation_date
        ).discount_factor(self.maturity_time)
        return _single_replay(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            maturity_time=self.maturity_time,
            amount=self.settlement_amount(curves, spot_price),
            currency=self.currency,
            known=self.settlement_price_known,
            discount_factor=discount,
            curve_ids=(self.discount_curve_id, self.net_benefit_curve_id),
        )

    def present_value(self, curves: CurveSet, spot_price: ArrayLike, /) -> Array:
        return self.cashflow_replay(curves, spot_price).present_value(self.currency)


class ResolvedCommodityFuture(AbstractResolvedContract):
    """Daily-settled commodity future; no forward discounting is hidden."""

    settlement_price: Array
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    valuation_date: FinanceDate = eqx.field(static=True)
    maturity_date: FinanceDate = eqx.field(static=True)
    maturity_time: float = eqx.field(static=True)
    quantity: float = eqx.field(static=True)
    contract_price: float = eqx.field(static=True)
    pay_receive: PayReceive = eqx.field(static=True)
    settlement_price_known: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        currency: Currency,
        valuation_date: FinanceDate,
        maturity_date: FinanceDate,
        maturity_time: float,
        quantity: float,
        contract_price: float,
        pay_receive: PayReceive | str,
        settlement_price: ArrayLike = 0.0,
        settlement_price_known: bool,
    ):
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        maturity = _dates(valuation_date, maturity_date, maturity_time)
        contract = _identifier(contract_id, "contract_id")
        self.settlement_price = _settlement_fixing(
            settlement_price, settlement_price_known
        )
        self.contract_id = contract
        self.currency = currency
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.maturity_time = maturity
        self.quantity = _positive(quantity, "quantity")
        self.contract_price = _positive(contract_price, "contract_price")
        self.pay_receive = _direction(pay_receive)
        self.settlement_price_known = settlement_price_known
        self.resolved_id = _resolved_id(
            "resolved-commodity-future",
            {
                "contract_id": contract,
                "currency": currency.currency_id,
                "maturity": maturity_date.ordinal,
                "quantity": self.quantity,
                "contract_price": self.contract_price,
                "direction": self.pay_receive.value,
                "settlement_known": settlement_price_known,
            },
        )

    def settlement_amount(self, model_futures_price: ArrayLike, /) -> Array:
        model = jnp.asarray(model_futures_price)
        if model.shape != ():
            raise ValueError("model_futures_price must be scalar.")
        model = eqx.error_if(
            model,
            ~jnp.isfinite(model),
            "model_futures_price must be finite.",
        )
        price = self.settlement_price if self.settlement_price_known else model
        return self.pay_receive.sign * self.quantity * (price - self.contract_price)

    @property
    def known_cashflows(self) -> CashflowBatch:
        amount = (
            self.pay_receive.sign
            * self.quantity
            * (self.settlement_price - self.contract_price)
        )
        return _known_settlement(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            amount=amount,
            currency=self.currency,
            known=self.settlement_price_known,
        )

    def cashflow_replay(
        self, model_futures_price: ArrayLike, /
    ) -> DeterministicCashflowReplay:
        return _single_replay(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            maturity_time=self.maturity_time,
            amount=self.settlement_amount(model_futures_price),
            currency=self.currency,
            known=self.settlement_price_known,
            discount_factor=jnp.asarray(1.0),
            curve_ids=(),
        )

    def present_value(self, model_futures_price: ArrayLike, /) -> Array:
        return self.cashflow_replay(model_futures_price).present_value(self.currency)


class ResolvedEquityFuture(AbstractResolvedContract):
    """Daily-settled equity future with explicit contract and settlement prices."""

    settlement_price: Array
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    valuation_date: FinanceDate = eqx.field(static=True)
    maturity_date: FinanceDate = eqx.field(static=True)
    maturity_time: float = eqx.field(static=True)
    units: float = eqx.field(static=True)
    contract_price: float = eqx.field(static=True)
    pay_receive: PayReceive = eqx.field(static=True)
    settlement_price_known: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        currency: Currency,
        valuation_date: FinanceDate,
        maturity_date: FinanceDate,
        maturity_time: float,
        units: float,
        contract_price: float,
        pay_receive: PayReceive | str,
        settlement_price: ArrayLike = 0.0,
        settlement_price_known: bool,
    ):
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        maturity = _dates(valuation_date, maturity_date, maturity_time)
        contract = _identifier(contract_id, "contract_id")
        self.settlement_price = _settlement_fixing(
            settlement_price, settlement_price_known
        )
        self.contract_id = contract
        self.currency = currency
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.maturity_time = maturity
        self.units = _positive(units, "units")
        self.contract_price = _positive(contract_price, "contract_price")
        self.pay_receive = _direction(pay_receive)
        self.settlement_price_known = settlement_price_known
        self.resolved_id = _resolved_id(
            "resolved-equity-future",
            {
                "contract_id": contract,
                "currency": currency.currency_id,
                "maturity": maturity_date.ordinal,
                "units": self.units,
                "contract_price": self.contract_price,
                "direction": self.pay_receive.value,
                "settlement_known": settlement_price_known,
            },
        )

    def settlement_amount(self, model_futures_price: ArrayLike, /) -> Array:
        model = jnp.asarray(model_futures_price)
        if model.shape != ():
            raise ValueError("model_futures_price must be scalar.")
        model = eqx.error_if(
            model,
            ~jnp.isfinite(model),
            "model_futures_price must be finite.",
        )
        price = self.settlement_price if self.settlement_price_known else model
        return self.pay_receive.sign * self.units * (price - self.contract_price)

    @property
    def known_cashflows(self) -> CashflowBatch:
        amount = (
            self.pay_receive.sign
            * self.units
            * (self.settlement_price - self.contract_price)
        )
        return _known_settlement(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            amount=amount,
            currency=self.currency,
            known=self.settlement_price_known,
        )

    def cashflow_replay(
        self, model_futures_price: ArrayLike, /
    ) -> DeterministicCashflowReplay:
        return _single_replay(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            maturity_time=self.maturity_time,
            amount=self.settlement_amount(model_futures_price),
            currency=self.currency,
            known=self.settlement_price_known,
            discount_factor=jnp.asarray(1.0),
            curve_ids=(),
        )

    def present_value(self, model_futures_price: ArrayLike, /) -> Array:
        return self.cashflow_replay(model_futures_price).present_value(self.currency)


class ResolvedFXFuture(AbstractResolvedContract):
    """Daily-settled FX future quoted in quote currency per unit base."""

    settlement_rate: Array
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    fx_pair: FXPair = eqx.field(static=True)
    valuation_date: FinanceDate = eqx.field(static=True)
    maturity_date: FinanceDate = eqx.field(static=True)
    maturity_time: float = eqx.field(static=True)
    base_notional: float = eqx.field(static=True)
    contract_rate: float = eqx.field(static=True)
    pay_receive: PayReceive = eqx.field(static=True)
    settlement_rate_known: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        fx_pair: FXPair,
        valuation_date: FinanceDate,
        maturity_date: FinanceDate,
        maturity_time: float,
        base_notional: float,
        contract_rate: float,
        pay_receive: PayReceive | str,
        settlement_rate: ArrayLike = 0.0,
        settlement_rate_known: bool,
    ):
        if not isinstance(fx_pair, FXPair):
            raise TypeError("fx_pair must be an FXPair.")
        maturity = _dates(valuation_date, maturity_date, maturity_time)
        contract = _identifier(contract_id, "contract_id")
        self.settlement_rate = _settlement_fixing(settlement_rate, settlement_rate_known)
        self.contract_id = contract
        self.fx_pair = fx_pair
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.maturity_time = maturity
        self.base_notional = _positive(base_notional, "base_notional")
        self.contract_rate = _positive(contract_rate, "contract_rate")
        self.pay_receive = _direction(pay_receive)
        self.settlement_rate_known = settlement_rate_known
        self.resolved_id = _resolved_id(
            "resolved-fx-future",
            {
                "contract_id": contract,
                "fx_pair": fx_pair.pair_id,
                "maturity": maturity_date.ordinal,
                "base_notional": self.base_notional,
                "contract_rate": self.contract_rate,
                "direction": self.pay_receive.value,
                "settlement_known": settlement_rate_known,
            },
        )

    def settlement_amount(self, model_futures_rate: ArrayLike, /) -> Array:
        model = jnp.asarray(model_futures_rate)
        if model.shape != ():
            raise ValueError("model_futures_rate must be scalar.")
        model = eqx.error_if(
            model,
            (~jnp.isfinite(model)) | (model <= 0.0),
            "model_futures_rate must be finite and positive.",
        )
        rate = self.settlement_rate if self.settlement_rate_known else model
        return self.pay_receive.sign * self.base_notional * (rate - self.contract_rate)

    @property
    def known_cashflows(self) -> CashflowBatch:
        amount = (
            self.pay_receive.sign
            * self.base_notional
            * (self.settlement_rate - self.contract_rate)
        )
        return _known_settlement(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            amount=amount,
            currency=self.fx_pair.quote,
            known=self.settlement_rate_known,
        )

    def cashflow_replay(
        self, model_futures_rate: ArrayLike, /
    ) -> DeterministicCashflowReplay:
        return _single_replay(
            contract_id=self.contract_id,
            maturity_date=self.maturity_date,
            maturity_time=self.maturity_time,
            amount=self.settlement_amount(model_futures_rate),
            currency=self.fx_pair.quote,
            known=self.settlement_rate_known,
            discount_factor=jnp.asarray(1.0),
            curve_ids=(),
        )

    def present_value(self, model_futures_rate: ArrayLike, /) -> Array:
        return self.cashflow_replay(model_futures_rate).present_value(self.fx_pair.quote)


__all__ = [
    "ResolvedCommodityForward",
    "ResolvedCommodityFuture",
    "ResolvedEquityForward",
    "ResolvedEquityFuture",
    "ResolvedFXForward",
    "ResolvedFXFuture",
]
