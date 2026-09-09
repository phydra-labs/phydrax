#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed option definitions and payoff markers.

Contract resolution remains a host operation.  The payoff records contain only
fixed-shape numerical terms consumed by explicit valuation routes; none defines a
universal pricing dispatcher.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..core._currency import CurrencyAmount
from ..core._identifiers import FinancialIdentifier
from ..core._time import FinanceDate
from ._base import AbstractContract, AbstractPayoff
from ._cashflows import CashflowBatch
from ._exercise import ExerciseSchedule, ExerciseStyle, SettlementTerms
from ._resolution import (
    ContractResolutionContext,
    ContractResolutionStatus,
    ResolvedContract,
)


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class BarrierDirection(str, Enum):
    UP = "up"
    DOWN = "down"


class BarrierActivation(str, Enum):
    KNOCK_IN = "knock_in"
    KNOCK_OUT = "knock_out"


class AverageType(str, Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(
        array,
        ~jnp.isfinite(array) | (array <= 0.0),
        f"{name} must be finite and positive.",
    )


def _option_type(value: OptionType) -> OptionType:
    if not isinstance(value, OptionType):
        raise TypeError("option_type must be an OptionType.")
    return value


def _amount(
    value: CurrencyAmount, name: str, /, *, positive: bool = True
) -> CurrencyAmount:
    if not isinstance(value, CurrencyAmount):
        raise TypeError(f"{name} must be a CurrencyAmount.")
    if positive and int(value.atoms) <= 0:
        raise ValueError(f"{name} must be strictly positive.")
    return value


def _underlyings(
    values: Sequence[FinancialIdentifier], /
) -> tuple[FinancialIdentifier, ...]:
    underlyings = tuple(values)
    if not underlyings or any(
        not isinstance(value, FinancialIdentifier) for value in underlyings
    ):
        raise TypeError("underlyings must contain at least one FinancialIdentifier.")
    ids = tuple(value.canonical for value in underlyings)
    if len(set(ids)) != len(ids):
        raise ValueError("underlyings must be unique.")
    return underlyings


def _dates(values: Sequence[FinanceDate], name: str, /) -> tuple[FinanceDate, ...]:
    dates = tuple(values)
    if not dates or any(not isinstance(value, FinanceDate) for value in dates):
        raise TypeError(f"{name} must contain at least one FinanceDate.")
    if any(
        right.ordinal <= left.ordinal
        for left, right in zip(dates[:-1], dates[1:], strict=True)
    ):
        raise ValueError(f"{name} must be strictly increasing.")
    return dates


def _settlement(value: SettlementTerms) -> SettlementTerms:
    if not isinstance(value, SettlementTerms):
        raise TypeError("settlement must be SettlementTerms.")
    return value


def _definition_id(kind: str, fields: dict[str, object], /) -> str:
    return canonical_fingerprint({"kind": kind, **fields})


def _resolve(
    definition: AbstractContract,
    underlyings: tuple[FinancialIdentifier, ...],
    settlement: SettlementTerms,
    exercise: ExerciseSchedule,
    context: ContractResolutionContext,
    /,
) -> ResolvedContract:
    if not isinstance(context, ContractResolutionContext):
        raise TypeError("context must be a ContractResolutionContext.")
    known_ids = {
        value.identifier.canonical for value in context.reference_data.assets
    } | {value.identifier.canonical for value in context.reference_data.instruments}
    status = ContractResolutionStatus(context.context_status)
    if any(value.canonical not in known_ids for value in underlyings):
        status |= ContractResolutionStatus.INVALID_REFERENCE
    if settlement.calendar_id not in {value.calendar_id for value in context.calendars}:
        status |= ContractResolutionStatus.CALENDAR_MISMATCH
    cashflows = CashflowBatch(
        (),
        jnp.zeros((0,), dtype=float),
        (),
        capacity=context.cashflow_capacity,
    )
    return ResolvedContract(
        definition,
        cashflows,
        settlement,
        exercise=exercise,
        resolution_status=status,
    )


def amount_in_major_units(value: CurrencyAmount, /) -> Array:
    if not isinstance(value, CurrencyAmount):
        raise TypeError("value must be a CurrencyAmount.")
    return value.atoms.astype(float) / value.currency.atoms_per_unit


class VanillaPayoff(AbstractPayoff):
    strike: Array
    notional: Array
    option_type: OptionType = eqx.field(static=True)
    payoff_id: str = eqx.field(static=True)

    def __init__(
        self, strike: ArrayLike, option_type: OptionType, /, *, notional: ArrayLike = 1.0
    ):
        self.strike = _positive_scalar(strike, "strike")
        self.notional = _positive_scalar(notional, "notional")
        self.option_type = _option_type(option_type)
        self.payoff_id = f"vanilla:{option_type.value}"


class CashDigitalPayoff(AbstractPayoff):
    strike: Array
    cash_amount: Array
    option_type: OptionType = eqx.field(static=True)
    payoff_id: str = eqx.field(static=True)

    def __init__(
        self, strike: ArrayLike, cash_amount: ArrayLike, option_type: OptionType, /
    ):
        self.strike = _positive_scalar(strike, "strike")
        self.cash_amount = _positive_scalar(cash_amount, "cash_amount")
        self.option_type = _option_type(option_type)
        self.payoff_id = f"cash-digital:{option_type.value}"


class PathBarrierPayoff(AbstractPayoff):
    strike: Array
    barrier: Array
    rebate: Array
    notional: Array
    option_type: OptionType = eqx.field(static=True)
    direction: BarrierDirection = eqx.field(static=True)
    activation: BarrierActivation = eqx.field(static=True)
    payoff_id: str = eqx.field(static=True)

    def __init__(
        self,
        strike: ArrayLike,
        barrier: ArrayLike,
        option_type: OptionType,
        direction: BarrierDirection,
        activation: BarrierActivation,
        /,
        *,
        rebate: ArrayLike = 0.0,
        notional: ArrayLike = 1.0,
    ):
        if not isinstance(direction, BarrierDirection) or not isinstance(
            activation, BarrierActivation
        ):
            raise TypeError("direction and activation must be barrier enums.")
        rebate_ = jnp.asarray(rebate, dtype=float)
        if rebate_.shape != ():
            raise ValueError("rebate must be scalar.")
        rebate_ = eqx.error_if(
            rebate_,
            ~jnp.isfinite(rebate_) | (rebate_ < 0.0),
            "rebate must be finite and non-negative.",
        )
        self.strike = _positive_scalar(strike, "strike")
        self.barrier = _positive_scalar(barrier, "barrier")
        self.rebate = rebate_
        self.notional = _positive_scalar(notional, "notional")
        self.option_type = _option_type(option_type)
        self.direction = direction
        self.activation = activation
        self.payoff_id = (
            f"barrier:{direction.value}:{activation.value}:{option_type.value}"
        )


class AsianPayoff(AbstractPayoff):
    strike: Array
    notional: Array
    option_type: OptionType = eqx.field(static=True)
    average_type: AverageType = eqx.field(static=True)
    payoff_id: str = eqx.field(static=True)

    def __init__(
        self,
        strike: ArrayLike,
        option_type: OptionType,
        /,
        *,
        average_type: AverageType = AverageType.ARITHMETIC,
        notional: ArrayLike = 1.0,
    ):
        if not isinstance(average_type, AverageType):
            raise TypeError("average_type must be an AverageType.")
        self.strike = _positive_scalar(strike, "strike")
        self.notional = _positive_scalar(notional, "notional")
        self.option_type = _option_type(option_type)
        self.average_type = average_type
        self.payoff_id = f"asian:{average_type.value}:{option_type.value}"


class LookbackPayoff(AbstractPayoff):
    strike: Array
    notional: Array
    option_type: OptionType = eqx.field(static=True)
    payoff_id: str = eqx.field(static=True)

    def __init__(
        self, strike: ArrayLike, option_type: OptionType, /, *, notional: ArrayLike = 1.0
    ):
        self.strike = _positive_scalar(strike, "strike")
        self.notional = _positive_scalar(notional, "notional")
        self.option_type = _option_type(option_type)
        self.payoff_id = f"fixed-strike-lookback:{option_type.value}"


class BasketPayoff(AbstractPayoff):
    strike: Array
    weights: Array
    notional: Array
    option_type: OptionType = eqx.field(static=True)
    payoff_id: str = eqx.field(static=True)
    asset_count: int = eqx.field(static=True)

    def __init__(
        self,
        strike: ArrayLike,
        weights: ArrayLike,
        option_type: OptionType,
        /,
        *,
        notional: ArrayLike = 1.0,
    ):
        weights_ = jnp.asarray(weights, dtype=float)
        if weights_.ndim != 1 or weights_.size < 1:
            raise ValueError("weights must be a non-empty vector.")
        weights_ = eqx.error_if(
            weights_, jnp.any(~jnp.isfinite(weights_)), "weights must be finite."
        )
        weights_ = eqx.error_if(
            weights_,
            jnp.isclose(jnp.sum(jnp.abs(weights_)), 0.0),
            "basket weights cannot all be zero.",
        )
        self.strike = _positive_scalar(strike, "strike")
        self.weights = weights_
        self.notional = _positive_scalar(notional, "notional")
        self.option_type = _option_type(option_type)
        self.payoff_id = f"basket:{option_type.value}"
        self.asset_count = int(weights_.size)


class VarianceSwapPayoff(AbstractPayoff):
    variance_strike: Array
    variance_notional: Array
    payoff_id: str = eqx.field(static=True)

    def __init__(self, variance_strike: ArrayLike, variance_notional: ArrayLike, /):
        strike = jnp.asarray(variance_strike, dtype=float)
        if strike.shape != ():
            raise ValueError("variance_strike must be scalar.")
        self.variance_strike = eqx.error_if(
            strike,
            ~jnp.isfinite(strike) | (strike < 0.0),
            "variance_strike must be finite and non-negative.",
        )
        self.variance_notional = _positive_scalar(variance_notional, "variance_notional")
        self.payoff_id = "variance-swap"


class BermudanPayoff(AbstractPayoff):
    strike: Array
    notional: Array
    option_type: OptionType = eqx.field(static=True)
    payoff_id: str = eqx.field(static=True)

    def __init__(
        self, strike: ArrayLike, option_type: OptionType, /, *, notional: ArrayLike = 1.0
    ):
        self.strike = _positive_scalar(strike, "strike")
        self.notional = _positive_scalar(notional, "notional")
        self.option_type = _option_type(option_type)
        self.payoff_id = f"bermudan:{option_type.value}"


class EuropeanOption(AbstractContract):
    underlying: FinancialIdentifier = eqx.field(static=True)
    strike: CurrencyAmount
    expiry: FinanceDate = eqx.field(static=True)
    option_type: OptionType = eqx.field(static=True)
    settlement: SettlementTerms
    quantity: Array
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        underlying: FinancialIdentifier,
        strike: CurrencyAmount,
        expiry: FinanceDate,
        option_type: OptionType,
        settlement: SettlementTerms,
        /,
        *,
        quantity: ArrayLike = 1.0,
    ):
        if not isinstance(underlying, FinancialIdentifier) or not isinstance(
            expiry, FinanceDate
        ):
            raise TypeError(
                "underlying and expiry must be financial identifier/date values."
            )
        strike_ = _amount(strike, "strike")
        settlement_ = _settlement(settlement)
        if strike_.currency.currency_id != settlement_.currency.currency_id:
            raise ValueError("option strike must use settlement currency.")
        self.underlying = underlying
        self.strike = strike_
        self.expiry = expiry
        self.option_type = _option_type(option_type)
        self.settlement = settlement_
        self.quantity = _positive_scalar(quantity, "quantity")
        self.contract_id = _definition_id(
            "european-option",
            {
                "underlying": underlying.canonical,
                "strike_atoms": int(strike.atoms),
                "currency": strike.currency.code,
                "expiry": expiry.ordinal,
                "option_type": option_type.value,
                "settlement": settlement.terms_id,
                "quantity": float(quantity),
            },
        )

    def payoff(self) -> VanillaPayoff:
        return VanillaPayoff(
            amount_in_major_units(self.strike), self.option_type, notional=self.quantity
        )

    def resolve(self, context: ContractResolutionContext, /) -> ResolvedContract:
        exercise = ExerciseSchedule((self.expiry,), style=ExerciseStyle.EUROPEAN)
        return _resolve(self, (self.underlying,), self.settlement, exercise, context)


class DigitalOption(AbstractContract):
    underlying: FinancialIdentifier = eqx.field(static=True)
    strike: CurrencyAmount
    cash_payout: CurrencyAmount
    expiry: FinanceDate = eqx.field(static=True)
    option_type: OptionType = eqx.field(static=True)
    settlement: SettlementTerms
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        underlying: FinancialIdentifier,
        strike: CurrencyAmount,
        cash_payout: CurrencyAmount,
        expiry: FinanceDate,
        option_type: OptionType,
        settlement: SettlementTerms,
        /,
    ):
        if not isinstance(underlying, FinancialIdentifier) or not isinstance(
            expiry, FinanceDate
        ):
            raise TypeError(
                "underlying and expiry must be financial identifier/date values."
            )
        strike_ = _amount(strike, "strike")
        payout = _amount(cash_payout, "cash_payout")
        if (
            strike_.currency.currency_id != payout.currency.currency_id
            or payout.currency.currency_id != settlement.currency.currency_id
        ):
            raise ValueError(
                "digital strike, payout, and settlement currencies must match."
            )
        self.underlying, self.strike, self.cash_payout, self.expiry = (
            underlying,
            strike_,
            payout,
            expiry,
        )
        self.option_type, self.settlement = (
            _option_type(option_type),
            _settlement(settlement),
        )
        self.contract_id = _definition_id(
            "digital-option",
            {
                "underlying": underlying.canonical,
                "strike_atoms": int(strike.atoms),
                "payout_atoms": int(cash_payout.atoms),
                "currency": strike.currency.code,
                "expiry": expiry.ordinal,
                "option_type": option_type.value,
                "settlement": settlement.terms_id,
            },
        )

    def payoff(self) -> CashDigitalPayoff:
        return CashDigitalPayoff(
            amount_in_major_units(self.strike),
            amount_in_major_units(self.cash_payout),
            self.option_type,
        )

    def resolve(self, context: ContractResolutionContext, /) -> ResolvedContract:
        return _resolve(
            self,
            (self.underlying,),
            self.settlement,
            ExerciseSchedule((self.expiry,), style=ExerciseStyle.EUROPEAN),
            context,
        )


class BarrierOption(AbstractContract):
    underlying: FinancialIdentifier = eqx.field(static=True)
    strike: CurrencyAmount
    barrier: CurrencyAmount
    expiry: FinanceDate = eqx.field(static=True)
    option_type: OptionType = eqx.field(static=True)
    direction: BarrierDirection = eqx.field(static=True)
    activation: BarrierActivation = eqx.field(static=True)
    settlement: SettlementTerms
    rebate: CurrencyAmount
    quantity: Array
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        underlying: FinancialIdentifier,
        strike: CurrencyAmount,
        barrier: CurrencyAmount,
        expiry: FinanceDate,
        option_type: OptionType,
        direction: BarrierDirection,
        activation: BarrierActivation,
        settlement: SettlementTerms,
        /,
        *,
        rebate: CurrencyAmount | None = None,
        quantity: ArrayLike = 1.0,
    ):
        if not isinstance(underlying, FinancialIdentifier) or not isinstance(
            expiry, FinanceDate
        ):
            raise TypeError(
                "underlying and expiry must be financial identifier/date values."
            )
        if not isinstance(direction, BarrierDirection) or not isinstance(
            activation, BarrierActivation
        ):
            raise TypeError("direction and activation must be barrier enums.")
        settlement_ = _settlement(settlement)
        strike_, barrier_ = _amount(strike, "strike"), _amount(barrier, "barrier")
        rebate_ = (
            CurrencyAmount(strike.currency, 0)
            if rebate is None
            else _amount(rebate, "rebate", positive=False)
        )
        if int(rebate_.atoms) < 0:
            raise ValueError("barrier rebate must be non-negative.")
        if any(
            value.currency.currency_id != settlement_.currency.currency_id
            for value in (strike_, barrier_, rebate_)
        ):
            raise ValueError("barrier monetary terms must use settlement currency.")
        self.underlying, self.strike, self.barrier, self.expiry = (
            underlying,
            strike_,
            barrier_,
            expiry,
        )
        self.option_type, self.direction, self.activation = (
            _option_type(option_type),
            direction,
            activation,
        )
        self.settlement, self.rebate = settlement_, rebate_
        self.quantity = _positive_scalar(quantity, "quantity")
        self.contract_id = _definition_id(
            "barrier-option",
            {
                "underlying": underlying.canonical,
                "strike_atoms": int(strike.atoms),
                "barrier_atoms": int(barrier.atoms),
                "rebate_atoms": int(rebate_.atoms),
                "currency": strike.currency.code,
                "expiry": expiry.ordinal,
                "option_type": option_type.value,
                "direction": direction.value,
                "activation": activation.value,
                "settlement": settlement.terms_id,
                "quantity": float(quantity),
            },
        )

    def payoff(self) -> PathBarrierPayoff:
        return PathBarrierPayoff(
            amount_in_major_units(self.strike),
            amount_in_major_units(self.barrier),
            self.option_type,
            self.direction,
            self.activation,
            rebate=amount_in_major_units(self.rebate),
            notional=self.quantity,
        )

    def resolve(self, context: ContractResolutionContext, /) -> ResolvedContract:
        return _resolve(
            self,
            (self.underlying,),
            self.settlement,
            ExerciseSchedule((self.expiry,), style=ExerciseStyle.EUROPEAN),
            context,
        )


class AsianOption(AbstractContract):
    underlying: FinancialIdentifier = eqx.field(static=True)
    strike: CurrencyAmount
    observation_dates: tuple[FinanceDate, ...] = eqx.field(static=True)
    expiry: FinanceDate = eqx.field(static=True)
    option_type: OptionType = eqx.field(static=True)
    average_type: AverageType = eqx.field(static=True)
    settlement: SettlementTerms
    quantity: Array
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        underlying: FinancialIdentifier,
        strike: CurrencyAmount,
        observation_dates: Sequence[FinanceDate],
        expiry: FinanceDate,
        option_type: OptionType,
        settlement: SettlementTerms,
        /,
        *,
        average_type: AverageType = AverageType.ARITHMETIC,
        quantity: ArrayLike = 1.0,
    ):
        if not isinstance(underlying, FinancialIdentifier) or not isinstance(
            expiry, FinanceDate
        ):
            raise TypeError(
                "underlying and expiry must be financial identifier/date values."
            )
        dates = _dates(observation_dates, "observation_dates")
        if dates[-1].ordinal > expiry.ordinal:
            raise ValueError("observation dates cannot follow expiry.")
        if not isinstance(average_type, AverageType):
            raise TypeError("average_type must be an AverageType.")
        strike_ = _amount(strike, "strike")
        settlement_ = _settlement(settlement)
        if strike_.currency.currency_id != settlement_.currency.currency_id:
            raise ValueError("option strike must use settlement currency.")
        self.underlying, self.strike, self.observation_dates, self.expiry = (
            underlying,
            strike_,
            dates,
            expiry,
        )
        self.option_type, self.average_type, self.settlement = (
            _option_type(option_type),
            average_type,
            settlement_,
        )
        self.quantity = _positive_scalar(quantity, "quantity")
        self.contract_id = _definition_id(
            "asian-option",
            {
                "underlying": underlying.canonical,
                "strike_atoms": int(strike.atoms),
                "currency": strike.currency.code,
                "observations": [value.ordinal for value in dates],
                "expiry": expiry.ordinal,
                "option_type": option_type.value,
                "average_type": average_type.value,
                "settlement": settlement.terms_id,
                "quantity": float(quantity),
            },
        )

    def payoff(self) -> AsianPayoff:
        return AsianPayoff(
            amount_in_major_units(self.strike),
            self.option_type,
            average_type=self.average_type,
            notional=self.quantity,
        )

    def resolve(self, context: ContractResolutionContext, /) -> ResolvedContract:
        return _resolve(
            self,
            (self.underlying,),
            self.settlement,
            ExerciseSchedule((self.expiry,), style=ExerciseStyle.EUROPEAN),
            context,
        )


class LookbackOption(AbstractContract):
    underlying: FinancialIdentifier = eqx.field(static=True)
    strike: CurrencyAmount
    observation_dates: tuple[FinanceDate, ...] = eqx.field(static=True)
    expiry: FinanceDate = eqx.field(static=True)
    option_type: OptionType = eqx.field(static=True)
    settlement: SettlementTerms
    quantity: Array
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        underlying: FinancialIdentifier,
        strike: CurrencyAmount,
        observation_dates: Sequence[FinanceDate],
        expiry: FinanceDate,
        option_type: OptionType,
        settlement: SettlementTerms,
        /,
        *,
        quantity: ArrayLike = 1.0,
    ):
        if not isinstance(underlying, FinancialIdentifier) or not isinstance(
            expiry, FinanceDate
        ):
            raise TypeError(
                "underlying and expiry must be financial identifier/date values."
            )
        dates = _dates(observation_dates, "observation_dates")
        if dates[-1].ordinal > expiry.ordinal:
            raise ValueError("observation dates cannot follow expiry.")
        strike_ = _amount(strike, "strike")
        settlement_ = _settlement(settlement)
        if strike_.currency.currency_id != settlement_.currency.currency_id:
            raise ValueError("option strike must use settlement currency.")
        self.underlying, self.strike, self.observation_dates, self.expiry = (
            underlying,
            strike_,
            dates,
            expiry,
        )
        self.option_type, self.settlement = _option_type(option_type), settlement_
        self.quantity = _positive_scalar(quantity, "quantity")
        self.contract_id = _definition_id(
            "lookback-option",
            {
                "underlying": underlying.canonical,
                "strike_atoms": int(strike.atoms),
                "currency": strike.currency.code,
                "observations": [value.ordinal for value in dates],
                "expiry": expiry.ordinal,
                "option_type": option_type.value,
                "settlement": settlement.terms_id,
                "quantity": float(quantity),
            },
        )

    def payoff(self) -> LookbackPayoff:
        return LookbackPayoff(
            amount_in_major_units(self.strike), self.option_type, notional=self.quantity
        )

    def resolve(self, context: ContractResolutionContext, /) -> ResolvedContract:
        return _resolve(
            self,
            (self.underlying,),
            self.settlement,
            ExerciseSchedule((self.expiry,), style=ExerciseStyle.EUROPEAN),
            context,
        )


class BasketOption(AbstractContract):
    underlyings: tuple[FinancialIdentifier, ...] = eqx.field(static=True)
    weights: Array
    strike: CurrencyAmount
    expiry: FinanceDate = eqx.field(static=True)
    option_type: OptionType = eqx.field(static=True)
    settlement: SettlementTerms
    quantity: Array
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        underlyings: Sequence[FinancialIdentifier],
        weights: ArrayLike,
        strike: CurrencyAmount,
        expiry: FinanceDate,
        option_type: OptionType,
        settlement: SettlementTerms,
        /,
        *,
        quantity: ArrayLike = 1.0,
    ):
        underlyings_ = _underlyings(underlyings)
        strike_ = _amount(strike, "strike")
        settlement_ = _settlement(settlement)
        if strike_.currency.currency_id != settlement_.currency.currency_id:
            raise ValueError("option strike must use settlement currency.")
        payoff = BasketPayoff(
            amount_in_major_units(strike_), weights, option_type, notional=quantity
        )
        if payoff.asset_count != len(underlyings_):
            raise ValueError("weights must contain one entry per underlying.")
        if not isinstance(expiry, FinanceDate):
            raise TypeError("expiry must be a FinanceDate.")
        self.underlyings, self.weights, self.strike, self.expiry = (
            underlyings_,
            payoff.weights,
            strike_,
            expiry,
        )
        self.option_type, self.settlement, self.quantity = (
            payoff.option_type,
            settlement_,
            payoff.notional,
        )
        self.contract_id = _definition_id(
            "basket-option",
            {
                "underlyings": [value.canonical for value in underlyings_],
                "weights": [float(value) for value in self.weights],
                "strike_atoms": int(strike.atoms),
                "currency": strike.currency.code,
                "expiry": expiry.ordinal,
                "option_type": option_type.value,
                "settlement": settlement.terms_id,
                "quantity": float(quantity),
            },
        )

    def payoff(self) -> BasketPayoff:
        return BasketPayoff(
            amount_in_major_units(self.strike),
            self.weights,
            self.option_type,
            notional=self.quantity,
        )

    def resolve(self, context: ContractResolutionContext, /) -> ResolvedContract:
        return _resolve(
            self,
            self.underlyings,
            self.settlement,
            ExerciseSchedule((self.expiry,), style=ExerciseStyle.EUROPEAN),
            context,
        )


class VarianceSwap(AbstractContract):
    underlying: FinancialIdentifier = eqx.field(static=True)
    variance_strike: Array
    variance_notional: CurrencyAmount
    observation_dates: tuple[FinanceDate, ...] = eqx.field(static=True)
    expiry: FinanceDate = eqx.field(static=True)
    settlement: SettlementTerms
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        underlying: FinancialIdentifier,
        variance_strike: ArrayLike,
        variance_notional: CurrencyAmount,
        observation_dates: Sequence[FinanceDate],
        expiry: FinanceDate,
        settlement: SettlementTerms,
        /,
    ):
        if not isinstance(underlying, FinancialIdentifier) or not isinstance(
            expiry, FinanceDate
        ):
            raise TypeError(
                "underlying and expiry must be financial identifier/date values."
            )
        dates = _dates(observation_dates, "observation_dates")
        if dates[-1].ordinal > expiry.ordinal:
            raise ValueError("observation dates cannot follow expiry.")
        notional = _amount(variance_notional, "variance_notional")
        if notional.currency.currency_id != settlement.currency.currency_id:
            raise ValueError("variance notional must use settlement currency.")
        payoff = VarianceSwapPayoff(variance_strike, amount_in_major_units(notional))
        self.underlying, self.variance_strike, self.variance_notional = (
            underlying,
            payoff.variance_strike,
            notional,
        )
        self.observation_dates, self.expiry, self.settlement = (
            dates,
            expiry,
            _settlement(settlement),
        )
        self.contract_id = _definition_id(
            "variance-swap",
            {
                "underlying": underlying.canonical,
                "variance_strike": float(variance_strike),
                "notional_atoms": int(notional.atoms),
                "currency": notional.currency.code,
                "observations": [value.ordinal for value in dates],
                "expiry": expiry.ordinal,
                "settlement": settlement.terms_id,
            },
        )

    def payoff(self) -> VarianceSwapPayoff:
        return VarianceSwapPayoff(
            self.variance_strike, amount_in_major_units(self.variance_notional)
        )

    def resolve(self, context: ContractResolutionContext, /) -> ResolvedContract:
        return _resolve(
            self,
            (self.underlying,),
            self.settlement,
            ExerciseSchedule((self.expiry,), style=ExerciseStyle.EUROPEAN),
            context,
        )


class BermudanOption(AbstractContract):
    underlying: FinancialIdentifier = eqx.field(static=True)
    strike: CurrencyAmount
    exercise_dates: tuple[FinanceDate, ...] = eqx.field(static=True)
    option_type: OptionType = eqx.field(static=True)
    settlement: SettlementTerms
    quantity: Array
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        underlying: FinancialIdentifier,
        strike: CurrencyAmount,
        exercise_dates: Sequence[FinanceDate],
        option_type: OptionType,
        settlement: SettlementTerms,
        /,
        *,
        quantity: ArrayLike = 1.0,
    ):
        if not isinstance(underlying, FinancialIdentifier):
            raise TypeError("underlying must be a FinancialIdentifier.")
        dates = _dates(exercise_dates, "exercise_dates")
        if len(dates) < 2:
            raise ValueError("Bermudan options require at least two exercise dates.")
        strike_ = _amount(strike, "strike")
        settlement_ = _settlement(settlement)
        if strike_.currency.currency_id != settlement_.currency.currency_id:
            raise ValueError("option strike must use settlement currency.")
        self.underlying, self.strike, self.exercise_dates = underlying, strike_, dates
        self.option_type, self.settlement = _option_type(option_type), settlement_
        self.quantity = _positive_scalar(quantity, "quantity")
        self.contract_id = _definition_id(
            "bermudan-option",
            {
                "underlying": underlying.canonical,
                "strike_atoms": int(strike.atoms),
                "currency": strike.currency.code,
                "exercise_dates": [value.ordinal for value in dates],
                "option_type": option_type.value,
                "settlement": settlement.terms_id,
                "quantity": float(quantity),
            },
        )

    @property
    def expiry(self) -> FinanceDate:
        return self.exercise_dates[-1]

    def payoff(self) -> BermudanPayoff:
        return BermudanPayoff(
            amount_in_major_units(self.strike), self.option_type, notional=self.quantity
        )

    def resolve(self, context: ContractResolutionContext, /) -> ResolvedContract:
        schedule = ExerciseSchedule(self.exercise_dates, style=ExerciseStyle.BERMUDAN)
        return _resolve(self, (self.underlying,), self.settlement, schedule, context)


__all__ = [
    "AsianOption",
    "AsianPayoff",
    "AverageType",
    "BarrierActivation",
    "BarrierDirection",
    "BarrierOption",
    "BasketOption",
    "BasketPayoff",
    "BermudanOption",
    "BermudanPayoff",
    "CashDigitalPayoff",
    "DigitalOption",
    "EuropeanOption",
    "LookbackOption",
    "LookbackPayoff",
    "OptionType",
    "PathBarrierPayoff",
    "VanillaPayoff",
    "VarianceSwap",
    "VarianceSwapPayoff",
    "amount_in_major_units",
]
