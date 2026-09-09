#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resolved deterministic rate cashflows with explicit curve and fixing roles."""

from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..core import (
    Currency,
    DayCount,
    FinanceDate,
    FXPair,
    ResolvedSchedule,
    year_fraction,
)
from ._base import AbstractResolvedContract
from ._cashflows import CashflowBatch


if TYPE_CHECKING:
    from ..curves._core import CurveSet, PreparedCurve


class PayReceive(str, Enum):
    """Cashflow direction from the holder's perspective."""

    PAY = "pay"
    RECEIVE = "receive"

    @property
    def sign(self) -> float:
        return -1.0 if self is PayReceive.PAY else 1.0


class FRASettlement(str, Enum):
    """FRA settlement convention."""

    PERIOD_START = "period_start"
    PERIOD_END = "period_end"


class FuturesQuoteConvention(str, Enum):
    """Interest-rate futures quote representation."""

    RATE = "rate"
    PRICE_100_MINUS_RATE = "price_100_minus_rate"


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


def _enum(value: Any, enum_type: type[Enum], name: str, /):
    if isinstance(value, enum_type):
        return value
    allowed_values = tuple(member.value for member in enum_type)
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise ValueError(f"{name} must be one of: {allowed}.")
    return enum_type(value)


def _signed_year_fraction(
    valuation_date: FinanceDate,
    date: FinanceDate,
    day_count: DayCount,
    /,
) -> float:
    if date.ordinal >= valuation_date.ordinal:
        return year_fraction(valuation_date, date, day_count)
    return -year_fraction(date, valuation_date, day_count)


class ResolvedRateSchedule(StrictModule):
    """Calendar-resolved schedule with explicit curve-time coordinates."""

    schedule: ResolvedSchedule
    accrual_start_times: Array
    accrual_end_times: Array
    payment_times: Array
    valuation_date: FinanceDate = eqx.field(static=True)
    curve_time_day_count: DayCount = eqx.field(static=True)

    def __init__(
        self,
        schedule: ResolvedSchedule,
        accrual_start_times: ArrayLike,
        accrual_end_times: ArrayLike,
        payment_times: ArrayLike,
        /,
        *,
        valuation_date: FinanceDate,
        curve_time_day_count: DayCount,
    ):
        if not isinstance(schedule, ResolvedSchedule):
            raise TypeError("schedule must be a ResolvedSchedule.")
        if not isinstance(valuation_date, FinanceDate):
            raise TypeError("valuation_date must be a FinanceDate.")
        if not isinstance(curve_time_day_count, DayCount):
            raise TypeError("curve_time_day_count must be a DayCount.")
        start = jnp.asarray(accrual_start_times)
        end = jnp.asarray(accrual_end_times)
        payment = jnp.asarray(payment_times)
        expected = (schedule.capacity,)
        if start.shape != expected or end.shape != expected or payment.shape != expected:
            raise ValueError("Resolved curve-time arrays must match schedule capacity.")
        mask = np.asarray(schedule.valid, dtype=bool)
        for name, values in (
            ("accrual_start_times", start),
            ("accrual_end_times", end),
            ("payment_times", payment),
        ):
            concrete = np.asarray(values)
            if np.any(~np.isfinite(concrete[mask])):
                raise ValueError(f"Active {name} must be finite.")
            if np.any(concrete[~mask] != 0.0):
                raise ValueError(f"Inactive {name} must use neutral zero padding.")
        start_np, end_np, payment_np = map(np.asarray, (start, end, payment))
        if np.any(end_np[mask] <= start_np[mask]) or np.any(
            payment_np[mask] < end_np[mask]
        ):
            raise ValueError("Resolved curve times are inconsistent.")
        self.schedule = schedule
        self.accrual_start_times = start
        self.accrual_end_times = end
        self.payment_times = payment
        self.valuation_date = valuation_date
        self.curve_time_day_count = curve_time_day_count

    @classmethod
    def from_schedule(
        cls,
        schedule: ResolvedSchedule,
        valuation_date: FinanceDate,
        curve_time_day_count: DayCount,
        /,
    ) -> ResolvedRateSchedule:
        if not isinstance(schedule, ResolvedSchedule):
            raise TypeError("schedule must be a ResolvedSchedule.")
        mask = np.asarray(schedule.valid, dtype=bool)

        def converted(ordinals: Array) -> Array:
            values = np.zeros((schedule.capacity,), dtype=float)
            source = np.asarray(ordinals)
            for index in np.flatnonzero(mask):
                values[index] = _signed_year_fraction(
                    valuation_date,
                    FinanceDate(int(source[index])),
                    curve_time_day_count,
                )
            return jnp.asarray(values)

        return cls(
            schedule,
            converted(schedule.accrual_start_dates),
            converted(schedule.accrual_end_dates),
            converted(schedule.payment_dates),
            valuation_date=valuation_date,
            curve_time_day_count=curve_time_day_count,
        )

    @property
    def active_mask(self) -> Array:
        return self.schedule.valid

    @property
    def future_payment_mask(self) -> Array:
        return self.schedule.valid & (self.payment_times >= 0.0)


class DeterministicCashflowReplay(StrictModule):
    """Fixed-shape projected/known cashflows and their discounted values."""

    payment_ordinals: Array
    payment_times: Array
    amounts: Array
    currency_index: Array
    valid_mask: Array
    known_mask: Array
    projected_mask: Array
    notional_exchange_mask: Array
    discount_factors: Array
    present_values: Array
    currencies: tuple[Currency, ...] = eqx.field(static=True)
    slot_currency_codes: tuple[str, ...] = eqx.field(static=True)
    obligation_ids: tuple[str, ...] = eqx.field(static=True)
    curve_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        payment_ordinals: ArrayLike,
        payment_times: ArrayLike,
        amounts: ArrayLike,
        slot_currencies: tuple[Currency, ...],
        valid_mask: ArrayLike,
        known_mask: ArrayLike,
        projected_mask: ArrayLike,
        notional_exchange_mask: ArrayLike,
        discount_factors: ArrayLike,
        obligation_ids: tuple[str, ...],
        curve_ids: tuple[str, ...],
    ):
        payment = jnp.asarray(payment_ordinals, dtype=jnp.int32)
        times = jnp.asarray(payment_times)
        amounts_ = jnp.asarray(amounts)
        valid = jnp.asarray(valid_mask, dtype=bool)
        known = jnp.asarray(known_mask, dtype=bool)
        projected = jnp.asarray(projected_mask, dtype=bool)
        notional = jnp.asarray(notional_exchange_mask, dtype=bool)
        discounts = jnp.asarray(discount_factors)
        shape = amounts_.shape
        arrays = (payment, times, valid, known, projected, notional, discounts)
        if amounts_.ndim != 1 or any(value.shape != shape for value in arrays):
            raise ValueError(
                "Deterministic cashflow replay arrays must share rank-one shape."
            )
        if len(slot_currencies) != int(shape[0]) or any(
            not isinstance(value, Currency) for value in slot_currencies
        ):
            raise TypeError("slot_currencies must contain one Currency per replay slot.")
        if len(obligation_ids) != int(shape[0]):
            raise ValueError("obligation_ids must contain one entry per replay slot.")
        identifiers = tuple(str(value).strip() for value in obligation_ids)
        if any(not identifier for identifier in identifiers) or len(
            set(identifiers)
        ) != len(identifiers):
            raise ValueError("Replay obligation identifiers must be nonempty and unique.")
        amounts_ = eqx.error_if(
            amounts_,
            jnp.any(~jnp.isfinite(amounts_))
            | jnp.any(~jnp.isfinite(discounts))
            | jnp.any(valid & (discounts <= 0.0))
            | jnp.any(valid & (known == projected))
            | jnp.any(~valid & (amounts_ != 0.0))
            | jnp.any(~valid & (payment != 0))
            | jnp.any(~valid & (times != 0.0))
            | jnp.any(~valid & (known | projected | notional)),
            "Cashflow replay arrays are invalid or non-neutral when inactive.",
        )
        by_code: dict[str, Currency] = {}
        for currency in slot_currencies:
            by_code.setdefault(currency.code, currency)
        currencies = tuple(by_code[code] for code in sorted(by_code))
        index_by_code = {
            currency.code: index for index, currency in enumerate(currencies)
        }
        indices = jnp.asarray(
            tuple(index_by_code[value.code] for value in slot_currencies), dtype=jnp.int32
        )
        self.payment_ordinals = payment
        self.payment_times = times
        self.amounts = amounts_
        self.currency_index = indices
        self.valid_mask = valid
        self.known_mask = known
        self.projected_mask = projected
        self.notional_exchange_mask = notional
        self.discount_factors = discounts
        self.present_values = jnp.where(valid, amounts_ * discounts, 0.0)
        self.currencies = currencies
        self.slot_currency_codes = tuple(value.code for value in slot_currencies)
        self.obligation_ids = identifiers
        self.curve_ids = tuple(
            dict.fromkeys(_identifier(value, "curve_id") for value in curve_ids)
        )

    @property
    def active_count(self) -> Array:
        return jnp.sum(self.valid_mask, dtype=jnp.int32)

    def present_value(self, currency: Currency, /) -> Array:
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        codes = tuple(value.code for value in self.currencies)
        if currency.code not in codes:
            return jnp.asarray(0.0, dtype=self.present_values.dtype)
        index = codes.index(currency.code)
        return jnp.sum(
            jnp.where(
                self.valid_mask & (self.currency_index == index),
                self.present_values,
                0.0,
            )
        )


def _resolved_id(contract_id: str, kind: str, facts: dict[str, Any], /) -> str:
    return canonical_fingerprint(
        {"kind": kind, "contract_id": contract_id, "facts": facts}
    )


def _curve(
    curves: CurveSet, curve_id: str, schedule: ResolvedRateSchedule, /
) -> PreparedCurve:
    curve = curves.curve(curve_id)
    if curve.definition.valuation_date.ordinal != schedule.valuation_date.ordinal:
        raise ValueError("Curve and resolved schedule valuation dates must match.")
    return curve


def _known_batch(
    payment_ordinals: Array,
    amounts: Array,
    slot_currencies: tuple[Currency, ...],
    valid: Array,
    obligation_ids: tuple[str, ...],
    /,
) -> CashflowBatch:
    mask = np.asarray(valid, dtype=bool)
    ordinals = np.asarray(payment_ordinals)[mask]
    amount_values = np.asarray(amounts)[mask]
    dates = tuple(FinanceDate(int(value)) for value in ordinals)
    currencies = tuple(
        value for value, active in zip(slot_currencies, mask, strict=True) if active
    )
    identifiers = tuple(
        value for value, active in zip(obligation_ids, mask, strict=True) if active
    )
    return CashflowBatch(dates, amount_values, currencies, obligation_ids=identifiers)


def _empty_known_cashflows() -> CashflowBatch:
    return CashflowBatch((), jnp.zeros((0,)), ())


def _combine_known(*batches: CashflowBatch) -> CashflowBatch:
    dates: list[FinanceDate] = []
    amounts: list[float] = []
    currencies: list[Currency] = []
    identifiers: list[str] = []
    for batch in batches:
        mask = np.asarray(batch.valid_mask, dtype=bool)
        for index in np.flatnonzero(mask):
            dates.append(FinanceDate(int(np.asarray(batch.payment_ordinals)[index])))
            amounts.append(float(np.asarray(batch.amounts)[index]))
            currencies.append(batch.currency_for(int(index)))
            identifiers.append(batch.obligation_ids[int(index)])
    return CashflowBatch(
        tuple(dates),
        jnp.asarray(amounts),
        tuple(currencies),
        obligation_ids=tuple(identifiers),
    )


def _slot_currency(replay: DeterministicCashflowReplay, code: str, /) -> Currency:
    for currency in replay.currencies:
        if currency.code == code:
            return currency
    raise RuntimeError("Replay currency topology is inconsistent.")


def _combine_replays(
    *replays: DeterministicCashflowReplay,
) -> DeterministicCashflowReplay:
    if not replays:
        raise ValueError("At least one replay is required.")
    slot_currencies = tuple(
        _slot_currency(replay, code)
        for replay in replays
        for code in replay.slot_currency_codes
    )
    return DeterministicCashflowReplay(
        payment_ordinals=jnp.concatenate(
            tuple(value.payment_ordinals for value in replays)
        ),
        payment_times=jnp.concatenate(tuple(value.payment_times for value in replays)),
        amounts=jnp.concatenate(tuple(value.amounts for value in replays)),
        slot_currencies=slot_currencies,
        valid_mask=jnp.concatenate(tuple(value.valid_mask for value in replays)),
        known_mask=jnp.concatenate(tuple(value.known_mask for value in replays)),
        projected_mask=jnp.concatenate(tuple(value.projected_mask for value in replays)),
        notional_exchange_mask=jnp.concatenate(
            tuple(value.notional_exchange_mask for value in replays)
        ),
        discount_factors=jnp.concatenate(
            tuple(value.discount_factors for value in replays)
        ),
        obligation_ids=tuple(
            identifier for value in replays for identifier in value.obligation_ids
        ),
        curve_ids=tuple(curve_id for value in replays for curve_id in value.curve_ids),
    )


def _leg_layout(
    schedule: ResolvedRateSchedule,
    currency: Currency,
    *,
    exchange_initial: bool,
    exchange_final: bool,
    prefix: str,
) -> tuple[Array, Array, Array, tuple[Currency, ...], tuple[str, ...], Array]:
    capacity = schedule.schedule.capacity
    count = schedule.schedule.period_count
    shape = (capacity + 2,)
    payment = jnp.zeros(shape, dtype=jnp.int32)
    times = jnp.zeros(shape)
    valid = jnp.zeros(shape, dtype=bool)
    notional = jnp.zeros(shape, dtype=bool)
    payment = payment.at[1 : capacity + 1].set(schedule.schedule.payment_dates)
    times = times.at[1 : capacity + 1].set(schedule.payment_times)
    valid = valid.at[1 : capacity + 1].set(schedule.future_payment_mask)
    initial_active = jnp.asarray(exchange_initial) & (
        schedule.accrual_start_times[0] >= 0.0
    )
    payment = payment.at[0].set(
        jnp.where(initial_active, schedule.schedule.accrual_start_dates[0], 0)
    )
    times = times.at[0].set(
        jnp.where(initial_active, schedule.accrual_start_times[0], 0.0)
    )
    valid = valid.at[0].set(initial_active)
    notional = notional.at[0].set(initial_active)
    final_active = jnp.asarray(exchange_final) & (
        schedule.payment_times[count - 1] >= 0.0
    )
    payment = payment.at[-1].set(
        jnp.where(final_active, schedule.schedule.payment_dates[count - 1], 0)
    )
    times = times.at[-1].set(
        jnp.where(final_active, schedule.payment_times[count - 1], 0.0)
    )
    valid = valid.at[-1].set(final_active)
    notional = notional.at[-1].set(final_active)
    identifiers = (
        f"{prefix}:initial-notional",
        *(f"{prefix}:coupon:{index}" for index in range(capacity)),
        f"{prefix}:final-notional",
    )
    return (
        payment,
        times,
        valid,
        (currency,) * (capacity + 2),
        identifiers,
        notional,
    )


def _discount_layout(curve: PreparedCurve, times: Array, valid: Array, /) -> Array:
    safe_times = jnp.where(valid, times, 0.0)
    values = curve.discount_factor(safe_times)
    return jnp.where(valid, values, 1.0)


class ResolvedFixedLeg(AbstractResolvedContract):
    """Fixed coupons and explicit initial/final notional exchanges."""

    schedule: ResolvedRateSchedule
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    notional: float = eqx.field(static=True)
    fixed_rate: float = eqx.field(static=True)
    pay_receive: PayReceive = eqx.field(static=True)
    exchange_initial: bool = eqx.field(static=True)
    exchange_final: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        schedule: ResolvedRateSchedule,
        currency: Currency,
        discount_curve_id: str,
        notional: float,
        fixed_rate: float,
        pay_receive: PayReceive | str,
        exchange_initial: bool,
        exchange_final: bool,
    ):
        if not isinstance(schedule, ResolvedRateSchedule):
            raise TypeError("schedule must be a ResolvedRateSchedule.")
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        if type(exchange_initial) is not bool or type(exchange_final) is not bool:
            raise TypeError("Notional-exchange flags must be bool values.")
        contract = _identifier(contract_id, "contract_id")
        discount_id = _identifier(discount_curve_id, "discount_curve_id")
        notional_ = _positive(notional, "notional")
        rate = _finite(fixed_rate, "fixed_rate")
        direction = _enum(pay_receive, PayReceive, "pay_receive")
        self.schedule = schedule
        self.contract_id = contract
        self.currency = currency
        self.discount_curve_id = discount_id
        self.notional = notional_
        self.fixed_rate = rate
        self.pay_receive = direction
        self.exchange_initial = exchange_initial
        self.exchange_final = exchange_final
        self.resolved_id = _resolved_id(
            contract,
            "resolved-fixed-leg",
            {
                "schedule": schedule.schedule.schedule_rule_id,
                "calendar": schedule.schedule.calendar_snapshot_id,
                "currency": currency.currency_id,
                "discount_curve": discount_id,
                "notional": notional_,
                "fixed_rate": rate,
                "direction": direction.value,
                "exchange_initial": exchange_initial,
                "exchange_final": exchange_final,
            },
        )

    def _cashflow_arrays(
        self,
    ) -> tuple[
        Array,
        Array,
        Array,
        tuple[Currency, ...],
        tuple[str, ...],
        Array,
        Array,
        Array,
        Array,
    ]:
        payment, times, valid, currencies, identifiers, notional_mask = _leg_layout(
            self.schedule,
            self.currency,
            exchange_initial=self.exchange_initial,
            exchange_final=self.exchange_final,
            prefix=self.contract_id,
        )
        capacity = self.schedule.schedule.capacity
        amounts = jnp.zeros((capacity + 2,))
        coupon_amounts = (
            self.pay_receive.sign
            * self.notional
            * self.fixed_rate
            * self.schedule.schedule.year_fractions
        )
        amounts = amounts.at[1 : capacity + 1].set(
            jnp.where(self.schedule.future_payment_mask, coupon_amounts, 0.0)
        )
        amounts = amounts.at[0].set(
            jnp.where(valid[0], -self.pay_receive.sign * self.notional, 0.0)
        )
        amounts = amounts.at[-1].set(
            jnp.where(valid[-1], self.pay_receive.sign * self.notional, 0.0)
        )
        known = valid
        projected = jnp.zeros_like(valid)
        return (
            payment,
            times,
            amounts,
            currencies,
            identifiers,
            valid,
            known,
            projected,
            notional_mask,
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        (
            payment,
            _times,
            amounts,
            currencies,
            identifiers,
            valid,
            _known,
            _projected,
            _notional,
        ) = self._cashflow_arrays()
        return _known_batch(payment, amounts, currencies, valid, identifiers)

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        (
            payment,
            times,
            amounts,
            currencies,
            identifiers,
            valid,
            known,
            projected,
            notional,
        ) = self._cashflow_arrays()
        discount = _curve(curves, self.discount_curve_id, self.schedule)
        return DeterministicCashflowReplay(
            payment_ordinals=payment,
            payment_times=times,
            amounts=amounts,
            slot_currencies=currencies,
            valid_mask=valid,
            known_mask=known,
            projected_mask=projected,
            notional_exchange_mask=notional,
            discount_factors=_discount_layout(discount, times, valid),
            obligation_ids=identifiers,
            curve_ids=(self.discount_curve_id,),
        )

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.cashflow_replay(curves).present_value(self.currency)


class ResolvedFloatingLeg(AbstractResolvedContract):
    """Fixed-shape floating coupons with distinct known and projected rates."""

    schedule: ResolvedRateSchedule
    fixing_values: Array
    fixing_mask: Array
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    projection_curve_id: str = eqx.field(static=True)
    notional: float = eqx.field(static=True)
    spread: float = eqx.field(static=True)
    gearing: float = eqx.field(static=True)
    pay_receive: PayReceive = eqx.field(static=True)
    exchange_initial: bool = eqx.field(static=True)
    exchange_final: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        schedule: ResolvedRateSchedule,
        currency: Currency,
        discount_curve_id: str,
        projection_curve_id: str,
        notional: float,
        spread: float,
        gearing: float,
        pay_receive: PayReceive | str,
        fixing_values: ArrayLike,
        fixing_mask: ArrayLike,
        exchange_initial: bool,
        exchange_final: bool,
    ):
        if not isinstance(schedule, ResolvedRateSchedule):
            raise TypeError("schedule must be a ResolvedRateSchedule.")
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        if type(exchange_initial) is not bool or type(exchange_final) is not bool:
            raise TypeError("Notional-exchange flags must be bool values.")
        values = jnp.asarray(fixing_values)
        mask = jnp.asarray(fixing_mask, dtype=bool)
        expected = (schedule.schedule.capacity,)
        if values.shape != expected or mask.shape != expected:
            raise ValueError("Fixing arrays must match the resolved schedule capacity.")
        active = np.asarray(schedule.schedule.valid, dtype=bool)
        values_np = np.asarray(values)
        mask_np = np.asarray(mask)
        if np.any(mask_np & ~active):
            raise ValueError("Inactive schedule slots cannot contain known fixings.")
        if np.any(~mask_np & (values_np != 0.0)):
            raise ValueError("Unknown fixing slots must use neutral zero values.")
        if np.any(mask_np & ~np.isfinite(values_np)):
            raise ValueError("Known fixing values must be finite.")
        start_np = np.asarray(schedule.accrual_start_times)
        if np.any(active & (start_np < 0.0) & ~mask_np):
            raise ValueError("Already-started floating periods require known fixings.")
        contract = _identifier(contract_id, "contract_id")
        discount_id = _identifier(discount_curve_id, "discount_curve_id")
        projection_id = _identifier(projection_curve_id, "projection_curve_id")
        notional_ = _positive(notional, "notional")
        spread_ = _finite(spread, "spread")
        gearing_ = _finite(gearing, "gearing")
        if gearing_ == 0.0:
            raise ValueError("gearing must be nonzero.")
        direction = _enum(pay_receive, PayReceive, "pay_receive")
        self.schedule = schedule
        self.fixing_values = values
        self.fixing_mask = mask
        self.contract_id = contract
        self.currency = currency
        self.discount_curve_id = discount_id
        self.projection_curve_id = projection_id
        self.notional = notional_
        self.spread = spread_
        self.gearing = gearing_
        self.pay_receive = direction
        self.exchange_initial = exchange_initial
        self.exchange_final = exchange_final
        self.resolved_id = _resolved_id(
            contract,
            "resolved-floating-leg",
            {
                "schedule": schedule.schedule.schedule_rule_id,
                "calendar": schedule.schedule.calendar_snapshot_id,
                "currency": currency.currency_id,
                "discount_curve": discount_id,
                "projection_curve": projection_id,
                "notional": notional_,
                "spread": spread_,
                "gearing": gearing_,
                "direction": direction.value,
                "fixing_mask": mask_np.tolist(),
                "exchange_initial": exchange_initial,
                "exchange_final": exchange_final,
            },
        )

    def coupon_rates(self, curves: CurveSet, /) -> Array:
        projection = _curve(curves, self.projection_curve_id, self.schedule)
        projected_mask = self.schedule.schedule.valid & ~self.fixing_mask
        safe_start = jnp.where(projected_mask, self.schedule.accrual_start_times, 0.0)
        safe_end = jnp.where(projected_mask, self.schedule.accrual_end_times, 1.0)
        safe_accrual = jnp.where(
            projected_mask, self.schedule.schedule.year_fractions, 1.0
        )
        projected = projection.forward_rate(
            safe_start, safe_end, accrual_fractions=safe_accrual
        )
        base_rate = jnp.where(self.fixing_mask, self.fixing_values, projected)
        return jnp.where(
            self.schedule.schedule.valid,
            self.gearing * base_rate + self.spread,
            0.0,
        )

    def _cashflow_arrays(
        self, rates: Array, /
    ) -> tuple[
        Array,
        Array,
        Array,
        tuple[Currency, ...],
        tuple[str, ...],
        Array,
        Array,
        Array,
        Array,
    ]:
        payment, times, valid, currencies, identifiers, notional_mask = _leg_layout(
            self.schedule,
            self.currency,
            exchange_initial=self.exchange_initial,
            exchange_final=self.exchange_final,
            prefix=self.contract_id,
        )
        capacity = self.schedule.schedule.capacity
        coupon_valid = self.schedule.future_payment_mask
        coupon_amounts = (
            self.pay_receive.sign
            * self.notional
            * rates
            * self.schedule.schedule.year_fractions
        )
        amounts = jnp.zeros((capacity + 2,))
        amounts = amounts.at[1 : capacity + 1].set(
            jnp.where(coupon_valid, coupon_amounts, 0.0)
        )
        amounts = amounts.at[0].set(
            jnp.where(valid[0], -self.pay_receive.sign * self.notional, 0.0)
        )
        amounts = amounts.at[-1].set(
            jnp.where(valid[-1], self.pay_receive.sign * self.notional, 0.0)
        )
        coupon_known = coupon_valid & self.fixing_mask
        coupon_projected = coupon_valid & ~self.fixing_mask
        known = jnp.zeros_like(valid).at[1 : capacity + 1].set(coupon_known)
        projected = jnp.zeros_like(valid).at[1 : capacity + 1].set(coupon_projected)
        known = known.at[0].set(valid[0]).at[-1].set(valid[-1])
        return (
            payment,
            times,
            amounts,
            currencies,
            identifiers,
            valid,
            known,
            projected,
            notional_mask,
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        rates = jnp.where(
            self.fixing_mask,
            self.gearing * self.fixing_values + self.spread,
            0.0,
        )
        (
            payment,
            _times,
            amounts,
            currencies,
            identifiers,
            _valid,
            known,
            _projected,
            _notional,
        ) = self._cashflow_arrays(rates)
        return _known_batch(payment, amounts, currencies, known, identifiers)

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        rates = self.coupon_rates(curves)
        (
            payment,
            times,
            amounts,
            currencies,
            identifiers,
            valid,
            known,
            projected,
            notional,
        ) = self._cashflow_arrays(rates)
        discount = _curve(curves, self.discount_curve_id, self.schedule)
        return DeterministicCashflowReplay(
            payment_ordinals=payment,
            payment_times=times,
            amounts=amounts,
            slot_currencies=currencies,
            valid_mask=valid,
            known_mask=known,
            projected_mask=projected,
            notional_exchange_mask=notional,
            discount_factors=_discount_layout(discount, times, valid),
            obligation_ids=identifiers,
            curve_ids=(self.discount_curve_id, self.projection_curve_id),
        )

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.cashflow_replay(curves).present_value(self.currency)


def _require_single_period(schedule: ResolvedRateSchedule, /) -> None:
    if schedule.schedule.period_count != 1:
        raise ValueError("This resolved product requires exactly one active period.")


class ResolvedDeposit(AbstractResolvedContract):
    """Resolved simple-compounded deposit with both notional exchanges."""

    schedule: ResolvedRateSchedule
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    notional: float = eqx.field(static=True)
    rate: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        schedule: ResolvedRateSchedule,
        currency: Currency,
        discount_curve_id: str,
        notional: float,
        rate: float,
    ):
        if not isinstance(schedule, ResolvedRateSchedule):
            raise TypeError("schedule must be a ResolvedRateSchedule.")
        _require_single_period(schedule)
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        contract = _identifier(contract_id, "contract_id")
        discount_id = _identifier(discount_curve_id, "discount_curve_id")
        self.schedule = schedule
        self.contract_id = contract
        self.currency = currency
        self.discount_curve_id = discount_id
        self.notional = _positive(notional, "notional")
        self.rate = _finite(rate, "rate")
        self.resolved_id = _resolved_id(
            contract,
            "resolved-deposit",
            {
                "schedule": schedule.schedule.schedule_rule_id,
                "currency": currency.currency_id,
                "discount_curve": discount_id,
                "notional": self.notional,
                "rate": self.rate,
            },
        )

    def _arrays(
        self,
    ) -> tuple[Array, Array, Array, Array, tuple[Currency, ...], tuple[str, ...]]:
        start_valid = self.schedule.accrual_start_times[0] >= 0.0
        end_valid = self.schedule.payment_times[0] >= 0.0
        valid = jnp.asarray((start_valid, end_valid), dtype=bool)
        payment = jnp.asarray(
            (
                jnp.where(start_valid, self.schedule.schedule.accrual_start_dates[0], 0),
                jnp.where(end_valid, self.schedule.schedule.payment_dates[0], 0),
            ),
            dtype=jnp.int32,
        )
        times = jnp.asarray(
            (
                jnp.where(start_valid, self.schedule.accrual_start_times[0], 0.0),
                jnp.where(end_valid, self.schedule.payment_times[0], 0.0),
            )
        )
        repayment = self.notional * (
            1.0 + self.rate * self.schedule.schedule.year_fractions[0]
        )
        amounts = jnp.asarray(
            (
                jnp.where(start_valid, -self.notional, 0.0),
                jnp.where(end_valid, repayment, 0.0),
            )
        )
        return (
            payment,
            times,
            amounts,
            valid,
            (self.currency, self.currency),
            (f"{self.contract_id}:initial", f"{self.contract_id}:repayment"),
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        payment, _times, amounts, valid, currencies, identifiers = self._arrays()
        return _known_batch(payment, amounts, currencies, valid, identifiers)

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        payment, times, amounts, valid, currencies, identifiers = self._arrays()
        discount = _curve(curves, self.discount_curve_id, self.schedule)
        return DeterministicCashflowReplay(
            payment_ordinals=payment,
            payment_times=times,
            amounts=amounts,
            slot_currencies=currencies,
            valid_mask=valid,
            known_mask=valid,
            projected_mask=jnp.zeros_like(valid),
            notional_exchange_mask=valid,
            discount_factors=_discount_layout(discount, times, valid),
            obligation_ids=identifiers,
            curve_ids=(self.discount_curve_id,),
        )

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.cashflow_replay(curves).present_value(self.currency)


class ResolvedForwardRateAgreement(AbstractResolvedContract):
    """FRA with explicit fixing state and start/end settlement convention."""

    schedule: ResolvedRateSchedule
    fixing_value: Array
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    projection_curve_id: str = eqx.field(static=True)
    notional: float = eqx.field(static=True)
    fixed_rate: float = eqx.field(static=True)
    pay_receive: PayReceive = eqx.field(static=True)
    settlement: FRASettlement = eqx.field(static=True)
    fixing_known: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        schedule: ResolvedRateSchedule,
        currency: Currency,
        discount_curve_id: str,
        projection_curve_id: str,
        notional: float,
        fixed_rate: float,
        pay_receive: PayReceive | str,
        settlement: FRASettlement | str,
        fixing_value: ArrayLike = 0.0,
        fixing_known: bool,
    ):
        if not isinstance(schedule, ResolvedRateSchedule):
            raise TypeError("schedule must be a ResolvedRateSchedule.")
        _require_single_period(schedule)
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        if type(fixing_known) is not bool:
            raise TypeError("fixing_known must be bool.")
        fixing = jnp.asarray(fixing_value)
        if fixing.shape != ():
            raise ValueError("fixing_value must be scalar.")
        fixing_np = float(np.asarray(fixing))
        if fixing_known and not isfinite(fixing_np):
            raise ValueError("A known fixing must be finite.")
        if not fixing_known and fixing_np != 0.0:
            raise ValueError("An unknown fixing must use neutral zero.")
        if float(np.asarray(schedule.accrual_start_times[0])) < 0.0 and not fixing_known:
            raise ValueError("An already-started FRA requires its fixing.")
        contract = _identifier(contract_id, "contract_id")
        discount_id = _identifier(discount_curve_id, "discount_curve_id")
        projection_id = _identifier(projection_curve_id, "projection_curve_id")
        self.schedule = schedule
        self.fixing_value = fixing
        self.contract_id = contract
        self.currency = currency
        self.discount_curve_id = discount_id
        self.projection_curve_id = projection_id
        self.notional = _positive(notional, "notional")
        self.fixed_rate = _finite(fixed_rate, "fixed_rate")
        self.pay_receive = _enum(pay_receive, PayReceive, "pay_receive")
        self.settlement = _enum(settlement, FRASettlement, "settlement")
        self.fixing_known = fixing_known
        self.resolved_id = _resolved_id(
            contract,
            "resolved-fra",
            {
                "schedule": schedule.schedule.schedule_rule_id,
                "currency": currency.currency_id,
                "discount_curve": discount_id,
                "projection_curve": projection_id,
                "notional": self.notional,
                "fixed_rate": self.fixed_rate,
                "direction": self.pay_receive.value,
                "settlement": self.settlement.value,
                "fixing_known": fixing_known,
            },
        )

    def rate(self, curves: CurveSet, /) -> Array:
        if self.fixing_known:
            return self.fixing_value
        projection = _curve(curves, self.projection_curve_id, self.schedule)
        return projection.forward_rate(
            self.schedule.accrual_start_times[0],
            self.schedule.accrual_end_times[0],
            accrual_fractions=self.schedule.schedule.year_fractions[0],
        )

    def settlement_amount(self, rate: ArrayLike, /) -> Array:
        floating = jnp.asarray(rate)
        accrual = self.schedule.schedule.year_fractions[0]
        amount = (
            self.pay_receive.sign * self.notional * accrual * (floating - self.fixed_rate)
        )
        if self.settlement is FRASettlement.PERIOD_START:
            amount = amount / (1.0 + accrual * floating)
        return amount

    def _settlement_coordinates(self) -> tuple[Array, Array, Array]:
        if self.settlement is FRASettlement.PERIOD_START:
            ordinal = self.schedule.schedule.accrual_start_dates[0]
            time = self.schedule.accrual_start_times[0]
        else:
            ordinal = self.schedule.schedule.payment_dates[0]
            time = self.schedule.payment_times[0]
        valid = time >= 0.0
        return (
            jnp.asarray((jnp.where(valid, ordinal, 0),), dtype=jnp.int32),
            jnp.asarray((jnp.where(valid, time, 0.0),)),
            jnp.asarray((valid,), dtype=bool),
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        if not self.fixing_known:
            return _empty_known_cashflows()
        payment, _times, valid = self._settlement_coordinates()
        amounts = jnp.asarray(
            (jnp.where(valid[0], self.settlement_amount(self.fixing_value), 0.0),)
        )
        return _known_batch(
            payment,
            amounts,
            (self.currency,),
            valid,
            (f"{self.contract_id}:settlement",),
        )

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        rate = self.rate(curves)
        payment, times, valid = self._settlement_coordinates()
        amounts = jnp.asarray((jnp.where(valid[0], self.settlement_amount(rate), 0.0),))
        discount = _curve(curves, self.discount_curve_id, self.schedule)
        known = valid & self.fixing_known
        projected = valid & (not self.fixing_known)
        return DeterministicCashflowReplay(
            payment_ordinals=payment,
            payment_times=times,
            amounts=amounts,
            slot_currencies=(self.currency,),
            valid_mask=valid,
            known_mask=known,
            projected_mask=projected,
            notional_exchange_mask=jnp.zeros_like(valid),
            discount_factors=_discount_layout(discount, times, valid),
            obligation_ids=(f"{self.contract_id}:settlement",),
            curve_ids=(self.discount_curve_id, self.projection_curve_id),
        )

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.cashflow_replay(curves).present_value(self.currency)


class ResolvedInterestRateFuture(AbstractResolvedContract):
    """Daily-settled rate future with an explicit quote convention."""

    schedule: ResolvedRateSchedule
    fixing_value: Array
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    projection_curve_id: str = eqx.field(static=True)
    contract_quote: float = eqx.field(static=True)
    quote_value_multiplier: float = eqx.field(static=True)
    convexity_adjustment: float = eqx.field(static=True)
    pay_receive: PayReceive = eqx.field(static=True)
    quote_convention: FuturesQuoteConvention = eqx.field(static=True)
    fixing_known: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        schedule: ResolvedRateSchedule,
        currency: Currency,
        projection_curve_id: str,
        contract_quote: float,
        quote_value_multiplier: float,
        convexity_adjustment: float,
        pay_receive: PayReceive | str,
        quote_convention: FuturesQuoteConvention | str,
        fixing_value: ArrayLike = 0.0,
        fixing_known: bool,
    ):
        if not isinstance(schedule, ResolvedRateSchedule):
            raise TypeError("schedule must be a ResolvedRateSchedule.")
        _require_single_period(schedule)
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        if type(fixing_known) is not bool:
            raise TypeError("fixing_known must be bool.")
        fixing = jnp.asarray(fixing_value)
        if fixing.shape != ():
            raise ValueError("fixing_value must be scalar.")
        fixing_np = float(np.asarray(fixing))
        if fixing_known and not isfinite(fixing_np):
            raise ValueError("A known fixing must be finite.")
        if not fixing_known and fixing_np != 0.0:
            raise ValueError("An unknown fixing must use neutral zero.")
        contract = _identifier(contract_id, "contract_id")
        self.schedule = schedule
        self.fixing_value = fixing
        self.contract_id = contract
        self.currency = currency
        self.projection_curve_id = _identifier(projection_curve_id, "projection_curve_id")
        self.contract_quote = _finite(contract_quote, "contract_quote")
        self.quote_value_multiplier = _positive(
            quote_value_multiplier, "quote_value_multiplier"
        )
        self.convexity_adjustment = _finite(convexity_adjustment, "convexity_adjustment")
        self.pay_receive = _enum(pay_receive, PayReceive, "pay_receive")
        self.quote_convention = _enum(
            quote_convention, FuturesQuoteConvention, "quote_convention"
        )
        self.fixing_known = fixing_known
        self.resolved_id = _resolved_id(
            contract,
            "resolved-interest-rate-future",
            {
                "schedule": schedule.schedule.schedule_rule_id,
                "currency": currency.currency_id,
                "projection_curve": self.projection_curve_id,
                "contract_quote": self.contract_quote,
                "quote_value_multiplier": self.quote_value_multiplier,
                "convexity_adjustment": self.convexity_adjustment,
                "direction": self.pay_receive.value,
                "quote_convention": self.quote_convention.value,
            },
        )

    def model_rate(self, curves: CurveSet, /) -> Array:
        if self.fixing_known:
            return self.fixing_value
        projection = _curve(curves, self.projection_curve_id, self.schedule)
        forward = projection.forward_rate(
            self.schedule.accrual_start_times[0],
            self.schedule.accrual_end_times[0],
            accrual_fractions=self.schedule.schedule.year_fractions[0],
        )
        return forward + self.convexity_adjustment

    def quote_from_rate(self, rate: ArrayLike, /) -> Array:
        rate_ = jnp.asarray(rate)
        if self.quote_convention is FuturesQuoteConvention.RATE:
            return rate_
        return 100.0 * (1.0 - rate_)

    def model_quote(self, curves: CurveSet, /) -> Array:
        return self.quote_from_rate(self.model_rate(curves))

    def settlement_amount(self, curves: CurveSet, /) -> Array:
        return (
            self.pay_receive.sign
            * self.quote_value_multiplier
            * (self.model_quote(curves) - self.contract_quote)
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        if (
            not self.fixing_known
            or float(np.asarray(self.schedule.payment_times[0])) < 0.0
        ):
            return _empty_known_cashflows()
        amount = (
            self.pay_receive.sign
            * self.quote_value_multiplier
            * (self.quote_from_rate(self.fixing_value) - self.contract_quote)
        )
        return CashflowBatch(
            (FinanceDate(int(np.asarray(self.schedule.schedule.payment_dates[0]))),),
            jnp.asarray((amount,)),
            (self.currency,),
            obligation_ids=(f"{self.contract_id}:variation-margin",),
        )

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        valid = jnp.asarray((self.schedule.payment_times[0] >= 0.0,), dtype=bool)
        payment = jnp.asarray(
            (jnp.where(valid[0], self.schedule.schedule.payment_dates[0], 0),),
            dtype=jnp.int32,
        )
        times = jnp.asarray((jnp.where(valid[0], self.schedule.payment_times[0], 0.0),))
        amounts = jnp.asarray((jnp.where(valid[0], self.settlement_amount(curves), 0.0),))
        known = valid & self.fixing_known
        projected = valid & (not self.fixing_known)
        return DeterministicCashflowReplay(
            payment_ordinals=payment,
            payment_times=times,
            amounts=amounts,
            slot_currencies=(self.currency,),
            valid_mask=valid,
            known_mask=known,
            projected_mask=projected,
            notional_exchange_mask=jnp.zeros_like(valid),
            discount_factors=jnp.ones_like(amounts),
            obligation_ids=(f"{self.contract_id}:variation-margin",),
            curve_ids=(self.projection_curve_id,),
        )

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.cashflow_replay(curves).present_value(self.currency)


class InflationLegStyle(str, Enum):
    """Inflation coupon ratio convention."""

    ZERO_COUPON = "zero_coupon"
    YEAR_ON_YEAR = "year_on_year"


class ResolvedInflationLeg(AbstractResolvedContract):
    """Inflation coupons preserving which lagged indices were known or projected."""

    schedule: ResolvedRateSchedule
    observation_start_times: Array
    observation_end_times: Array
    start_index_values: Array
    end_index_values: Array
    start_fixing_mask: Array
    end_fixing_mask: Array
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    index_curve_id: str = eqx.field(static=True)
    notional: float = eqx.field(static=True)
    spread: float = eqx.field(static=True)
    pay_receive: PayReceive = eqx.field(static=True)
    style: InflationLegStyle = eqx.field(static=True)

    def __init__(
        self,
        *,
        contract_id: str,
        schedule: ResolvedRateSchedule,
        currency: Currency,
        discount_curve_id: str,
        index_curve_id: str,
        notional: float,
        spread: float,
        pay_receive: PayReceive | str,
        style: InflationLegStyle | str,
        observation_start_times: ArrayLike,
        observation_end_times: ArrayLike,
        start_index_values: ArrayLike,
        end_index_values: ArrayLike,
        start_fixing_mask: ArrayLike,
        end_fixing_mask: ArrayLike,
    ):
        if not isinstance(schedule, ResolvedRateSchedule):
            raise TypeError("schedule must be a ResolvedRateSchedule.")
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        expected = (schedule.schedule.capacity,)
        observation_start = jnp.asarray(observation_start_times)
        observation_end = jnp.asarray(observation_end_times)
        start_values = jnp.asarray(start_index_values)
        end_values = jnp.asarray(end_index_values)
        start_mask = jnp.asarray(start_fixing_mask, dtype=bool)
        end_mask = jnp.asarray(end_fixing_mask, dtype=bool)
        arrays = (
            observation_start,
            observation_end,
            start_values,
            end_values,
            start_mask,
            end_mask,
        )
        if any(value.shape != expected for value in arrays):
            raise ValueError(
                "Inflation observation/fixing arrays must match schedule capacity."
            )
        active = np.asarray(schedule.schedule.valid, dtype=bool)
        obs_start_np = np.asarray(observation_start)
        obs_end_np = np.asarray(observation_end)
        start_np = np.asarray(start_values)
        end_np = np.asarray(end_values)
        start_mask_np = np.asarray(start_mask)
        end_mask_np = np.asarray(end_mask)
        if np.any(~np.isfinite(obs_start_np[active])) or np.any(
            ~np.isfinite(obs_end_np[active])
        ):
            raise ValueError("Active inflation observation times must be finite.")
        if np.any(obs_end_np[active] <= obs_start_np[active]):
            raise ValueError("Inflation end observations must follow start observations.")
        if np.any((start_mask_np | end_mask_np) & ~active):
            raise ValueError("Inactive inflation periods cannot carry fixing masks.")
        if np.any(~start_mask_np & (start_np != 0.0)) or np.any(
            ~end_mask_np & (end_np != 0.0)
        ):
            raise ValueError("Unknown inflation fixings must use neutral zero values.")
        if np.any(start_mask_np & ((start_np <= 0.0) | ~np.isfinite(start_np))) or np.any(
            end_mask_np & ((end_np <= 0.0) | ~np.isfinite(end_np))
        ):
            raise ValueError("Known inflation index levels must be finite and positive.")
        if np.any(active & (obs_start_np < 0.0) & ~start_mask_np) or np.any(
            active & (obs_end_np < 0.0) & ~end_mask_np
        ):
            raise ValueError("Past inflation observations require known fixings.")
        style_ = _enum(style, InflationLegStyle, "style")
        if (
            style_ is InflationLegStyle.ZERO_COUPON
            and schedule.schedule.period_count != 1
        ):
            raise ValueError("A zero-coupon inflation leg requires exactly one period.")
        contract = _identifier(contract_id, "contract_id")
        self.schedule = schedule
        self.observation_start_times = observation_start
        self.observation_end_times = observation_end
        self.start_index_values = start_values
        self.end_index_values = end_values
        self.start_fixing_mask = start_mask
        self.end_fixing_mask = end_mask
        self.contract_id = contract
        self.currency = currency
        self.discount_curve_id = _identifier(discount_curve_id, "discount_curve_id")
        self.index_curve_id = _identifier(index_curve_id, "index_curve_id")
        self.notional = _positive(notional, "notional")
        self.spread = _finite(spread, "spread")
        self.pay_receive = _enum(pay_receive, PayReceive, "pay_receive")
        self.style = style_
        self.resolved_id = _resolved_id(
            contract,
            "resolved-inflation-leg",
            {
                "schedule": schedule.schedule.schedule_rule_id,
                "currency": currency.currency_id,
                "discount_curve": self.discount_curve_id,
                "index_curve": self.index_curve_id,
                "notional": self.notional,
                "spread": self.spread,
                "direction": self.pay_receive.value,
                "style": self.style.value,
                "start_fixing_mask": start_mask_np.tolist(),
                "end_fixing_mask": end_mask_np.tolist(),
            },
        )

    def index_levels(self, curves: CurveSet, /) -> tuple[Array, Array]:
        curve = _curve(curves, self.index_curve_id, self.schedule)
        active = self.schedule.schedule.valid
        project_start = active & ~self.start_fixing_mask
        project_end = active & ~self.end_fixing_mask
        safe_start = jnp.where(project_start, self.observation_start_times, 0.0)
        safe_end = jnp.where(project_end, self.observation_end_times, 0.0)
        projected_start = curve.parameter(safe_start)
        projected_end = curve.parameter(safe_end)
        start = jnp.where(
            self.start_fixing_mask, self.start_index_values, projected_start
        )
        end = jnp.where(self.end_fixing_mask, self.end_index_values, projected_end)
        start = eqx.error_if(
            start,
            jnp.any(active & ((start <= 0.0) | ~jnp.isfinite(start))),
            "Inflation start index levels must be finite and positive.",
        )
        end = eqx.error_if(
            end,
            jnp.any(active & ((end <= 0.0) | ~jnp.isfinite(end))),
            "Inflation end index levels must be finite and positive.",
        )
        return start, end

    def coupon_amounts(self, curves: CurveSet, /) -> Array:
        start, end = self.index_levels(curves)
        ratio_return = end / start - 1.0
        amounts = (
            self.pay_receive.sign
            * self.notional
            * (ratio_return + self.spread * self.schedule.schedule.year_fractions)
        )
        return jnp.where(self.schedule.future_payment_mask, amounts, 0.0)

    def _replay_from_amounts(
        self, amounts: Array, curves: CurveSet, /
    ) -> DeterministicCashflowReplay:
        valid = self.schedule.future_payment_mask
        known = valid & self.start_fixing_mask & self.end_fixing_mask
        projected = valid & ~known
        payment = jnp.where(
            valid,
            self.schedule.schedule.payment_dates,
            jnp.zeros_like(valid, dtype=jnp.int32),
        )
        times = jnp.where(valid, self.schedule.payment_times, 0.0)
        identifiers = tuple(
            f"{self.contract_id}:inflation:{index}"
            for index in range(self.schedule.schedule.capacity)
        )
        discount = _curve(curves, self.discount_curve_id, self.schedule)
        return DeterministicCashflowReplay(
            payment_ordinals=payment,
            payment_times=times,
            amounts=amounts,
            slot_currencies=(self.currency,) * self.schedule.schedule.capacity,
            valid_mask=valid,
            known_mask=known,
            projected_mask=projected,
            notional_exchange_mask=jnp.zeros_like(valid),
            discount_factors=_discount_layout(discount, times, valid),
            obligation_ids=identifiers,
            curve_ids=(self.discount_curve_id, self.index_curve_id),
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        known = (
            self.schedule.future_payment_mask
            & self.start_fixing_mask
            & self.end_fixing_mask
        )
        ratio = self.end_index_values / jnp.where(
            self.start_fixing_mask, self.start_index_values, 1.0
        )
        amounts = (
            self.pay_receive.sign
            * self.notional
            * (ratio - 1.0 + self.spread * self.schedule.schedule.year_fractions)
        )
        amounts = jnp.where(known, amounts, 0.0)
        payment = jnp.where(
            known,
            self.schedule.schedule.payment_dates,
            jnp.zeros_like(self.schedule.schedule.payment_dates),
        )
        identifiers = tuple(
            f"{self.contract_id}:inflation:{index}"
            for index in range(self.schedule.schedule.capacity)
        )
        return _known_batch(
            payment,
            amounts,
            (self.currency,) * self.schedule.schedule.capacity,
            known,
            identifiers,
        )

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        return self._replay_from_amounts(self.coupon_amounts(curves), curves)

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.cashflow_replay(curves).present_value(self.currency)


def _validate_fixed_float(
    fixed_leg: ResolvedFixedLeg, floating_leg: ResolvedFloatingLeg, /
) -> None:
    if not isinstance(fixed_leg, ResolvedFixedLeg):
        raise TypeError("fixed_leg must be a ResolvedFixedLeg.")
    if not isinstance(floating_leg, ResolvedFloatingLeg):
        raise TypeError("floating_leg must be a ResolvedFloatingLeg.")
    if fixed_leg.currency.currency_id != floating_leg.currency.currency_id:
        raise ValueError("Single-currency swap legs must use the same currency.")
    if (
        fixed_leg.schedule.valuation_date.ordinal
        != floating_leg.schedule.valuation_date.ordinal
    ):
        raise ValueError("Swap leg valuation dates must match.")


class ResolvedIborSwap(AbstractResolvedContract):
    """Fixed versus term-index floating swap."""

    fixed_leg: ResolvedFixedLeg
    floating_leg: ResolvedFloatingLeg
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)

    def __init__(
        self,
        contract_id: str,
        fixed_leg: ResolvedFixedLeg,
        floating_leg: ResolvedFloatingLeg,
        /,
    ):
        _validate_fixed_float(fixed_leg, floating_leg)
        contract = _identifier(contract_id, "contract_id")
        self.contract_id = contract
        self.fixed_leg = fixed_leg
        self.floating_leg = floating_leg
        self.resolved_id = _resolved_id(
            contract,
            "resolved-ibor-swap",
            {
                "fixed_leg": fixed_leg.resolved_id,
                "floating_leg": floating_leg.resolved_id,
            },
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        return _combine_known(
            self.fixed_leg.known_cashflows, self.floating_leg.known_cashflows
        )

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        return _combine_replays(
            self.fixed_leg.cashflow_replay(curves),
            self.floating_leg.cashflow_replay(curves),
        )

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.cashflow_replay(curves).present_value(self.fixed_leg.currency)


class ResolvedOvernightIndexedSwap(AbstractResolvedContract):
    """Fixed versus compounded-overnight leg with coupon-level fixing states."""

    fixed_leg: ResolvedFixedLeg
    overnight_leg: ResolvedFloatingLeg
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)

    def __init__(
        self,
        contract_id: str,
        fixed_leg: ResolvedFixedLeg,
        overnight_leg: ResolvedFloatingLeg,
        /,
    ):
        _validate_fixed_float(fixed_leg, overnight_leg)
        contract = _identifier(contract_id, "contract_id")
        self.contract_id = contract
        self.fixed_leg = fixed_leg
        self.overnight_leg = overnight_leg
        self.resolved_id = _resolved_id(
            contract,
            "resolved-ois",
            {
                "fixed_leg": fixed_leg.resolved_id,
                "overnight_leg": overnight_leg.resolved_id,
            },
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        return _combine_known(
            self.fixed_leg.known_cashflows, self.overnight_leg.known_cashflows
        )

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        return _combine_replays(
            self.fixed_leg.cashflow_replay(curves),
            self.overnight_leg.cashflow_replay(curves),
        )

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.cashflow_replay(curves).present_value(self.fixed_leg.currency)


class ResolvedBasisSwap(AbstractResolvedContract):
    """Two explicitly directed floating legs."""

    first_leg: ResolvedFloatingLeg
    second_leg: ResolvedFloatingLeg
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)

    def __init__(
        self,
        contract_id: str,
        first_leg: ResolvedFloatingLeg,
        second_leg: ResolvedFloatingLeg,
        /,
    ):
        if not isinstance(first_leg, ResolvedFloatingLeg) or not isinstance(
            second_leg, ResolvedFloatingLeg
        ):
            raise TypeError("Basis-swap legs must be ResolvedFloatingLeg values.")
        if first_leg.currency.currency_id != second_leg.currency.currency_id:
            raise ValueError("Basis-swap legs must use the same currency.")
        if (
            first_leg.schedule.valuation_date.ordinal
            != second_leg.schedule.valuation_date.ordinal
        ):
            raise ValueError("Basis-swap valuation dates must match.")
        contract = _identifier(contract_id, "contract_id")
        self.contract_id = contract
        self.first_leg = first_leg
        self.second_leg = second_leg
        self.resolved_id = _resolved_id(
            contract,
            "resolved-basis-swap",
            {"first_leg": first_leg.resolved_id, "second_leg": second_leg.resolved_id},
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        return _combine_known(
            self.first_leg.known_cashflows, self.second_leg.known_cashflows
        )

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        return _combine_replays(
            self.first_leg.cashflow_replay(curves),
            self.second_leg.cashflow_replay(curves),
        )

    def present_value(self, curves: CurveSet, /) -> Array:
        return self.cashflow_replay(curves).present_value(self.first_leg.currency)


class ResolvedCrossCurrencySwap(AbstractResolvedContract):
    """Two-currency floating swap retaining both currency cashflow streams."""

    base_leg: ResolvedFloatingLeg
    quote_leg: ResolvedFloatingLeg
    fx_pair: FXPair = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)

    def __init__(
        self,
        contract_id: str,
        base_leg: ResolvedFloatingLeg,
        quote_leg: ResolvedFloatingLeg,
        fx_pair: FXPair,
        /,
    ):
        if not isinstance(base_leg, ResolvedFloatingLeg) or not isinstance(
            quote_leg, ResolvedFloatingLeg
        ):
            raise TypeError("Cross-currency legs must be ResolvedFloatingLeg values.")
        if not isinstance(fx_pair, FXPair):
            raise TypeError("fx_pair must be an FXPair.")
        if base_leg.currency.currency_id != fx_pair.base.currency_id:
            raise ValueError("base_leg currency must match fx_pair.base.")
        if quote_leg.currency.currency_id != fx_pair.quote.currency_id:
            raise ValueError("quote_leg currency must match fx_pair.quote.")
        if (
            base_leg.schedule.valuation_date.ordinal
            != quote_leg.schedule.valuation_date.ordinal
        ):
            raise ValueError("Cross-currency leg valuation dates must match.")
        contract = _identifier(contract_id, "contract_id")
        self.contract_id = contract
        self.base_leg = base_leg
        self.quote_leg = quote_leg
        self.fx_pair = fx_pair
        self.resolved_id = _resolved_id(
            contract,
            "resolved-cross-currency-swap",
            {
                "base_leg": base_leg.resolved_id,
                "quote_leg": quote_leg.resolved_id,
                "fx_pair": fx_pair.pair_id,
            },
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        return _combine_known(
            self.base_leg.known_cashflows, self.quote_leg.known_cashflows
        )

    def cashflow_replay(self, curves: CurveSet, /) -> DeterministicCashflowReplay:
        return _combine_replays(
            self.base_leg.cashflow_replay(curves),
            self.quote_leg.cashflow_replay(curves),
        )

    def present_value(
        self,
        curves: CurveSet,
        /,
        *,
        reporting_currency: Currency,
        spot_quote_per_base: ArrayLike,
    ) -> Array:
        replay = self.cashflow_replay(curves)
        spot = jnp.asarray(spot_quote_per_base)
        if spot.shape != ():
            raise ValueError("spot_quote_per_base must be scalar.")
        spot = eqx.error_if(
            spot,
            (~jnp.isfinite(spot)) | (spot <= 0.0),
            "spot_quote_per_base must be finite and positive.",
        )
        base_pv = replay.present_value(self.fx_pair.base)
        quote_pv = replay.present_value(self.fx_pair.quote)
        if reporting_currency.currency_id == self.fx_pair.quote.currency_id:
            return quote_pv + spot * base_pv
        if reporting_currency.currency_id == self.fx_pair.base.currency_id:
            return base_pv + quote_pv / spot
        raise ValueError("reporting_currency must be one endpoint of fx_pair.")


__all__ = [
    "DeterministicCashflowReplay",
    "FRASettlement",
    "FuturesQuoteConvention",
    "InflationLegStyle",
    "PayReceive",
    "ResolvedBasisSwap",
    "ResolvedCrossCurrencySwap",
    "ResolvedDeposit",
    "ResolvedFixedLeg",
    "ResolvedFloatingLeg",
    "ResolvedForwardRateAgreement",
    "ResolvedIborSwap",
    "ResolvedInflationLeg",
    "ResolvedInterestRateFuture",
    "ResolvedOvernightIndexedSwap",
    "ResolvedRateSchedule",
]
