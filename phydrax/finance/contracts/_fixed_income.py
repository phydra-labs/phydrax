#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resolved deterministic zero, fixed, and floating-rate bonds."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..core import Currency, FinanceDate
from ._base import AbstractResolvedContract
from ._cashflows import CashflowBatch
from ._rates import (
    DeterministicCashflowReplay,
    PayReceive,
    ResolvedFixedLeg,
    ResolvedFloatingLeg,
)


if TYPE_CHECKING:
    from ..curves._core import CurveSet


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be a non-empty string.")
    return identifier


def _positive(value: float, name: str, /) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _settlement_time(value: ArrayLike, /) -> Array:
    time = jnp.asarray(value)
    if time.shape != ():
        raise ValueError("settlement_time must be scalar.")
    return eqx.error_if(
        time,
        ~jnp.isfinite(time) | (time < 0.0),
        "settlement_time must be finite and nonnegative.",
    )


def _bond_resolved_id(contract_id: str, kind: str, leg_id: str, /) -> str:
    return canonical_fingerprint(
        {"kind": kind, "contract_id": contract_id, "leg_id": leg_id}
    )


class ResolvedZeroCouponBond(AbstractResolvedContract):
    """Single known redemption with an explicit valuation date and curve role."""

    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    valuation_date: FinanceDate = eqx.field(static=True)
    maturity_date: FinanceDate = eqx.field(static=True)
    maturity_time: float = eqx.field(static=True)
    face_value: float = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        currency: Currency,
        valuation_date: FinanceDate,
        maturity_date: FinanceDate,
        maturity_time: float,
        face_value: float,
        discount_curve_id: str,
    ):
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        if not isinstance(valuation_date, FinanceDate) or not isinstance(
            maturity_date, FinanceDate
        ):
            raise TypeError(
                "valuation_date and maturity_date must be FinanceDate values."
            )
        if maturity_date.ordinal <= valuation_date.ordinal:
            raise ValueError("maturity_date must be later than valuation_date.")
        maturity = _positive(maturity_time, "maturity_time")
        contract = _identifier(contract_id, "contract_id")
        discount_id = _identifier(discount_curve_id, "discount_curve_id")
        face = _positive(face_value, "face_value")
        self.contract_id = contract
        self.currency = currency
        self.valuation_date = valuation_date
        self.maturity_date = maturity_date
        self.maturity_time = maturity
        self.face_value = face
        self.discount_curve_id = discount_id
        self.resolved_id = canonical_fingerprint(
            {
                "kind": "resolved-zero-coupon-bond",
                "contract_id": contract,
                "currency": currency.currency_id,
                "valuation_date": valuation_date.ordinal,
                "maturity_date": maturity_date.ordinal,
                "maturity_time": maturity,
                "face_value": face,
                "discount_curve": discount_id,
            }
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        return CashflowBatch(
            (self.maturity_date,),
            jnp.asarray((self.face_value,)),
            (self.currency,),
            obligation_ids=(f"{self.contract_id}:redemption",),
        )

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        curve = curves.curve(self.discount_curve_id)
        if curve.definition.valuation_date.ordinal != self.valuation_date.ordinal:
            raise ValueError("Bond and discount-curve valuation dates must match.")
        valid = jnp.asarray((True,))
        times = jnp.asarray((self.maturity_time,))
        return DeterministicCashflowReplay(
            payment_ordinals=jnp.asarray((self.maturity_date.ordinal,), dtype=jnp.int32),
            payment_times=times,
            amounts=jnp.asarray((self.face_value,)),
            slot_currencies=(self.currency,),
            valid_mask=valid,
            known_mask=valid,
            projected_mask=jnp.asarray((False,)),
            notional_exchange_mask=valid,
            discount_factors=curve.discount_factor(times),
            obligation_ids=(f"{self.contract_id}:redemption",),
            curve_ids=(self.discount_curve_id,),
        )

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.cashflow_replay(curves).present_value(self.currency)

    def dirty_price(
        self, curves: CurveSet, /, *, settlement_time: ArrayLike = 0.0
    ) -> Array:
        settlement = _settlement_time(settlement_time)
        curve = curves.curve(self.discount_curve_id)
        unsettled = (
            100.0
            * curve.discount_factor(self.maturity_time)
            / curve.discount_factor(settlement)
        )
        return jnp.where(settlement < self.maturity_time, unsettled, 0.0)

    def accrued_interest(self, settlement_time: ArrayLike, /) -> Array:
        _settlement_time(settlement_time)
        return jnp.asarray(0.0)

    def clean_price(
        self, curves: CurveSet, /, *, settlement_time: ArrayLike = 0.0
    ) -> Array:
        return self.dirty_price(curves, settlement_time=settlement_time)


class ResolvedFixedRateBond(AbstractResolvedContract):
    """Coupon bond exposing clean, dirty, and accrued values separately."""

    coupon_leg: ResolvedFixedLeg
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)

    def __init__(self, contract_id: str, coupon_leg: ResolvedFixedLeg, /):
        if not isinstance(coupon_leg, ResolvedFixedLeg):
            raise TypeError("coupon_leg must be a ResolvedFixedLeg.")
        if coupon_leg.pay_receive is not PayReceive.RECEIVE:
            raise ValueError("A held bond coupon leg must be receive-directed.")
        if coupon_leg.exchange_initial or not coupon_leg.exchange_final:
            raise ValueError(
                "A bond coupon leg must omit initial exchange and include final redemption."
            )
        contract = _identifier(contract_id, "contract_id")
        self.contract_id = contract
        self.coupon_leg = coupon_leg
        self.resolved_id = _bond_resolved_id(
            contract, "resolved-fixed-rate-bond", coupon_leg.resolved_id
        )

    @property
    def currency(self) -> Currency:
        return self.coupon_leg.currency

    @property
    def face_value(self) -> float:
        return self.coupon_leg.notional

    @property
    def known_cashflows(self) -> CashflowBatch:
        return self.coupon_leg.known_cashflows

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        return self.coupon_leg.cashflow_replay(curves)

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.coupon_leg.present_value(curves)

    def accrued_interest(self, settlement_time: ArrayLike, /) -> Array:
        settlement = _settlement_time(settlement_time)
        schedule = self.coupon_leg.schedule
        active = schedule.schedule.valid
        inside = (
            active
            & (schedule.accrual_start_times <= settlement)
            & (settlement < schedule.accrual_end_times)
        )
        elapsed = jnp.where(
            inside,
            (settlement - schedule.accrual_start_times)
            / (schedule.accrual_end_times - schedule.accrual_start_times)
            * schedule.schedule.year_fractions,
            0.0,
        )
        return self.face_value * self.coupon_leg.fixed_rate * jnp.sum(elapsed)

    def accrued_price(self, settlement_time: ArrayLike, /) -> Array:
        return 100.0 * self.accrued_interest(settlement_time) / self.face_value

    def dirty_price(
        self, curves: CurveSet, /, *, settlement_time: ArrayLike = 0.0
    ) -> Array:
        settlement = _settlement_time(settlement_time)
        replay = self.cashflow_replay(curves)
        curve = curves.curve(self.coupon_leg.discount_curve_id)
        future = replay.valid_mask & (replay.payment_times > settlement)
        settlement_value = jnp.sum(
            jnp.where(
                future,
                replay.amounts
                * replay.discount_factors
                / curve.discount_factor(settlement),
                0.0,
            )
        )
        return 100.0 * settlement_value / self.face_value

    def clean_price(
        self, curves: CurveSet, /, *, settlement_time: ArrayLike = 0.0
    ) -> Array:
        return self.dirty_price(
            curves, settlement_time=settlement_time
        ) - self.accrued_price(settlement_time)


class ResolvedFloatingRateBond(AbstractResolvedContract):
    """Floating coupon bond retaining known/projected coupon provenance."""

    coupon_leg: ResolvedFloatingLeg
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)

    def __init__(self, contract_id: str, coupon_leg: ResolvedFloatingLeg, /):
        if not isinstance(coupon_leg, ResolvedFloatingLeg):
            raise TypeError("coupon_leg must be a ResolvedFloatingLeg.")
        if coupon_leg.pay_receive is not PayReceive.RECEIVE:
            raise ValueError("A held floating bond leg must be receive-directed.")
        if coupon_leg.exchange_initial or not coupon_leg.exchange_final:
            raise ValueError(
                "A floating bond leg must omit initial exchange and include final redemption."
            )
        contract = _identifier(contract_id, "contract_id")
        self.contract_id = contract
        self.coupon_leg = coupon_leg
        self.resolved_id = _bond_resolved_id(
            contract, "resolved-floating-rate-bond", coupon_leg.resolved_id
        )

    @property
    def currency(self) -> Currency:
        return self.coupon_leg.currency

    @property
    def face_value(self) -> float:
        return self.coupon_leg.notional

    @property
    def known_cashflows(self) -> CashflowBatch:
        return self.coupon_leg.known_cashflows

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        return self.coupon_leg.cashflow_replay(curves)

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.coupon_leg.present_value(curves)

    def accrued_interest(self, curves: CurveSet, settlement_time: ArrayLike, /) -> Array:
        settlement = _settlement_time(settlement_time)
        schedule = self.coupon_leg.schedule
        active = schedule.schedule.valid
        inside = (
            active
            & (schedule.accrual_start_times <= settlement)
            & (settlement < schedule.accrual_end_times)
        )
        elapsed = jnp.where(
            inside,
            (settlement - schedule.accrual_start_times)
            / (schedule.accrual_end_times - schedule.accrual_start_times)
            * schedule.schedule.year_fractions,
            0.0,
        )
        return self.face_value * jnp.sum(elapsed * self.coupon_leg.coupon_rates(curves))

    def accrued_price(self, curves: CurveSet, settlement_time: ArrayLike, /) -> Array:
        return 100.0 * self.accrued_interest(curves, settlement_time) / self.face_value

    def dirty_price(
        self, curves: CurveSet, /, *, settlement_time: ArrayLike = 0.0
    ) -> Array:
        settlement = _settlement_time(settlement_time)
        replay = self.cashflow_replay(curves)
        curve = curves.curve(self.coupon_leg.discount_curve_id)
        future = replay.valid_mask & (replay.payment_times > settlement)
        settlement_value = jnp.sum(
            jnp.where(
                future,
                replay.amounts
                * replay.discount_factors
                / curve.discount_factor(settlement),
                0.0,
            )
        )
        return 100.0 * settlement_value / self.face_value

    def clean_price(
        self, curves: CurveSet, /, *, settlement_time: ArrayLike = 0.0
    ) -> Array:
        return self.dirty_price(
            curves, settlement_time=settlement_time
        ) - self.accrued_price(curves, settlement_time)


__all__ = [
    "ResolvedFixedRateBond",
    "ResolvedFloatingRateBond",
    "ResolvedZeroCouponBond",
]
