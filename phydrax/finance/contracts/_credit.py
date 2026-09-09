# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Resolved credit contracts, recovery terms, and event cashflow lowering."""

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..core import CurrencyAmount, FinanceDate, InstrumentReference, ResolvedSchedule
from ._base import AbstractPayoff, AbstractResolvedContract
from ._cashflows import CashflowBatch


ProtectionSide: TypeAlias = Literal["buy", "sell"]
DefaultBoundarySide: TypeAlias = Literal["before_payment", "after_payment"]
CreditPayoffKind: TypeAlias = Literal["defaultable_bond", "credit_default_swap"]
RecoveryConvention: TypeAlias = Literal["par", "market_value", "treasury"]
DefaultTimingConvention: TypeAlias = Literal["at_default", "period_end"]
_DEFAULT_EVENT_SUCCESS = 0


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string.")
    return value


def _scalar(value: ArrayLike, name: str, /, *, nonnegative: bool = False) -> Array:
    result = jnp.asarray(value, dtype=float)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    host = float(np.asarray(jax.device_get(result)))
    if not np.isfinite(host) or (nonnegative and host < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _face_units(face: CurrencyAmount, /) -> Array:
    return face.atoms.astype(float) / float(face.currency.atoms_per_unit)


def _validate_payment_times(
    schedule: ResolvedSchedule, payment_times: ArrayLike, /
) -> Array:
    if not isinstance(schedule, ResolvedSchedule):
        raise TypeError("schedule must be a ResolvedSchedule.")
    times = jnp.asarray(payment_times, dtype=float)
    host = np.asarray(jax.device_get(times))
    mask = np.asarray(jax.device_get(schedule.valid))
    if times.shape != (schedule.capacity,):
        raise ValueError("payment_times must match the resolved schedule capacity.")
    if np.any(~np.isfinite(host[mask])) or np.any(host[mask] <= 0.0):
        raise ValueError("Active payment times must be finite and positive.")
    if np.any(np.diff(host[mask]) <= 0.0):
        raise ValueError("Active payment times must be strictly increasing.")
    if np.any(host[~mask] != 0.0):
        raise ValueError("Inactive payment times must use neutral zero padding.")
    return times


def _resolved_identity(
    kind: str,
    contract_id: str,
    instrument: InstrumentReference,
    schedule: ResolvedSchedule,
    payment_times: Array,
    terms: dict[str, object],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": kind,
            "contract_id": contract_id,
            "instrument_id": instrument.instrument_id,
            "schedule_rule_id": schedule.schedule_rule_id,
            "calendar_snapshot_id": schedule.calendar_snapshot_id,
            "payment_times": np.asarray(jax.device_get(payment_times)).tolist(),
            "terms": terms,
        }
    )


class RecoveryTerms(StrictModule):
    """Recovery convention and settlement timing supplied by the contract owner."""

    rate: Array
    settlement_lag: Array
    convention: RecoveryConvention = eqx.field(static=True)
    timing: DefaultTimingConvention = eqx.field(static=True)
    terms_id: str = eqx.field(static=True)

    def __init__(
        self,
        rate: ArrayLike,
        /,
        *,
        convention: RecoveryConvention,
        timing: DefaultTimingConvention,
        settlement_lag: ArrayLike = 0.0,
        terms_id: str,
    ):
        rate_ = _scalar(rate, "recovery rate", nonnegative=True)
        if float(np.asarray(jax.device_get(rate_))) > 1.0:
            raise ValueError("recovery rate must not exceed one.")
        if convention not in ("par", "market_value", "treasury"):
            raise ValueError("Unsupported recovery convention.")
        if timing not in ("at_default", "period_end"):
            raise ValueError("Unsupported default timing convention.")
        self.rate = rate_
        self.settlement_lag = _scalar(
            settlement_lag, "recovery settlement_lag", nonnegative=True
        )
        self.convention = convention
        self.timing = timing
        self.terms_id = _identifier(terms_id, "terms_id")


class DefaultEventState(StrictModule):
    """At most one resolved default per path with explicit law and coupling IDs."""

    default_times: Array
    occurred: Array
    recoveries: Array
    valid: Array
    status: Array
    reference_entity_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    recovery_terms_id: str = eqx.field(static=True)

    def __init__(
        self,
        default_times: ArrayLike,
        occurred: ArrayLike,
        recoveries: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        /,
        *,
        reference_entity_id: str,
        law_id: str,
        realization_id: str,
        coupling_id: str,
        recovery_terms_id: str,
    ):
        times = jnp.asarray(default_times, dtype=float)
        event_mask = jnp.asarray(occurred)
        recovery = jnp.asarray(recoveries, dtype=float)
        path_valid = jnp.asarray(valid)
        status_ = jnp.asarray(status)
        if event_mask.dtype != jnp.dtype(bool) or path_valid.dtype != jnp.dtype(bool):
            raise TypeError("occurred and valid must be boolean arrays.")
        if not jnp.issubdtype(status_.dtype, jnp.integer):
            raise TypeError("Default-event status must have an integer dtype.")
        status_ = status_.astype(jnp.int32)
        if any(
            value.shape != times.shape
            for value in (event_mask, recovery, path_valid, status_)
        ):
            raise ValueError("Default-event path arrays must have equal shapes.")
        if times.ndim != 1 or times.shape[0] == 0:
            raise ValueError("Default-event arrays must be non-empty path vectors.")
        times = eqx.error_if(
            times,
            jnp.any(event_mask & (~jnp.isfinite(times) | (times < 0.0)))
            | jnp.any(~event_mask & (times != 0.0)),
            "Occurred defaults require finite nonnegative times; absent defaults use zero.",
        )
        recovery = eqx.error_if(
            recovery,
            jnp.any(
                event_mask
                & (~jnp.isfinite(recovery) | (recovery < 0.0) | (recovery > 1.0))
            )
            | jnp.any(~event_mask & (recovery != 0.0)),
            "Occurred recoveries must lie in [0, 1]; absent defaults use zero.",
        )
        path_valid = eqx.error_if(
            path_valid,
            jnp.any(event_mask & ~path_valid)
            | jnp.any(path_valid != (status_ == _DEFAULT_EVENT_SUCCESS)),
            "Default occurrence and status must agree with path validity.",
        )
        self.default_times = times
        self.occurred = event_mask
        self.recoveries = recovery
        self.valid = path_valid
        self.status = status_
        self.reference_entity_id = _identifier(reference_entity_id, "reference_entity_id")
        self.law_id = _identifier(law_id, "law_id")
        self.realization_id = _identifier(realization_id, "realization_id")
        self.coupling_id = _identifier(coupling_id, "coupling_id")
        self.recovery_terms_id = _identifier(recovery_terms_id, "recovery_terms_id")


class CreditPayoff(AbstractPayoff):
    """Typed marker for one resolved credit payoff family."""

    payoff_id: str = eqx.field(static=True)
    reference_entity_id: str = eqx.field(static=True)
    kind: CreditPayoffKind = eqx.field(static=True)

    def __init__(
        self,
        payoff_id: str,
        reference_entity_id: str,
        kind: CreditPayoffKind,
        /,
    ):
        if kind not in ("defaultable_bond", "credit_default_swap"):
            raise ValueError("Unsupported credit payoff kind.")
        self.payoff_id = _identifier(payoff_id, "payoff_id")
        self.reference_entity_id = _identifier(reference_entity_id, "reference_entity_id")
        self.kind = kind


class DefaultableBondContract(AbstractResolvedContract):
    """Resolved fixed-coupon bond with explicit recovery and boundary semantics."""

    instrument: InstrumentReference = eqx.field(static=True)
    schedule: ResolvedSchedule
    face: CurrencyAmount
    coupon_rate: Array
    payment_times: Array
    recovery: RecoveryTerms
    payoff: CreditPayoff = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    default_boundary_side: DefaultBoundarySide = eqx.field(static=True)

    def __init__(
        self,
        instrument: InstrumentReference,
        schedule: ResolvedSchedule,
        face: CurrencyAmount,
        coupon_rate: ArrayLike,
        payment_times: ArrayLike,
        recovery: RecoveryTerms,
        payoff: CreditPayoff,
        /,
        *,
        contract_id: str,
        default_boundary_side: DefaultBoundarySide = "before_payment",
    ):
        if not isinstance(instrument, InstrumentReference):
            raise TypeError("instrument must be an InstrumentReference.")
        if not isinstance(face, CurrencyAmount):
            raise TypeError("face must be a CurrencyAmount.")
        if face.currency.currency_id != instrument.settlement_currency.currency_id:
            raise ValueError(
                "Bond face currency must match instrument settlement currency."
            )
        if int(np.asarray(jax.device_get(face.atoms))) <= 0:
            raise ValueError("Bond face amount must be positive.")
        if not isinstance(recovery, RecoveryTerms):
            raise TypeError("recovery must be RecoveryTerms.")
        if not isinstance(payoff, CreditPayoff) or payoff.kind != "defaultable_bond":
            raise TypeError("payoff must be a defaultable-bond CreditPayoff.")
        if default_boundary_side not in ("before_payment", "after_payment"):
            raise ValueError("Unsupported default boundary side.")
        identifier = _identifier(contract_id, "contract_id")
        times = _validate_payment_times(schedule, payment_times)
        coupon = _scalar(coupon_rate, "coupon_rate", nonnegative=True)
        self.instrument = instrument
        self.schedule = schedule
        self.face = face
        self.coupon_rate = coupon
        self.payment_times = times
        self.recovery = recovery
        self.payoff = payoff
        self.contract_id = identifier
        self.default_boundary_side = default_boundary_side
        self.resolved_id = _resolved_identity(
            "resolved-defaultable-bond",
            identifier,
            instrument,
            schedule,
            times,
            {
                "face_atoms": int(np.asarray(jax.device_get(face.atoms))),
                "coupon_rate": float(np.asarray(jax.device_get(coupon))),
                "recovery_terms_id": recovery.terms_id,
                "payoff_id": payoff.payoff_id,
                "default_boundary_side": default_boundary_side,
            },
        )

    @property
    def promised_amounts(self) -> Array:
        mask = self.schedule.valid
        coupons = _face_units(self.face) * self.coupon_rate * self.schedule.year_fractions
        last_index = jnp.sum(mask, dtype=jnp.int32) - 1
        principal = jnp.where(
            jnp.arange(self.schedule.capacity) == last_index,
            _face_units(self.face),
            0.0,
        )
        return jnp.where(mask, coupons + principal, 0.0)

    @property
    def known_cashflows(self) -> CashflowBatch:
        active = np.asarray(jax.device_get(self.schedule.valid))
        dates = tuple(
            FinanceDate(int(value))
            for value in np.asarray(jax.device_get(self.schedule.payment_dates))[active]
        )
        amounts = np.asarray(jax.device_get(self.promised_amounts))[active]
        currency = self.instrument.settlement_currency
        obligations = tuple(
            f"{self.contract_id}:promised:{index}" for index in range(len(dates))
        )
        return CashflowBatch(
            dates,
            amounts,
            (currency,) * len(dates),
            obligation_ids=obligations,
            capacity=self.schedule.capacity,
        )


class CreditDefaultSwapContract(AbstractResolvedContract):
    """Resolved running-spread CDS with explicit accrued-on-default convention."""

    instrument: InstrumentReference = eqx.field(static=True)
    schedule: ResolvedSchedule
    notional: CurrencyAmount
    running_spread: Array
    payment_times: Array
    recovery: RecoveryTerms
    payoff: CreditPayoff = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)
    protection_side: ProtectionSide = eqx.field(static=True)
    accrued_on_default: bool = eqx.field(static=True)
    default_boundary_side: DefaultBoundarySide = eqx.field(static=True)

    def __init__(
        self,
        instrument: InstrumentReference,
        schedule: ResolvedSchedule,
        notional: CurrencyAmount,
        running_spread: ArrayLike,
        payment_times: ArrayLike,
        recovery: RecoveryTerms,
        payoff: CreditPayoff,
        /,
        *,
        contract_id: str,
        protection_side: ProtectionSide,
        accrued_on_default: bool = True,
        default_boundary_side: DefaultBoundarySide = "before_payment",
    ):
        if not isinstance(instrument, InstrumentReference):
            raise TypeError("instrument must be an InstrumentReference.")
        if not isinstance(notional, CurrencyAmount):
            raise TypeError("notional must be a CurrencyAmount.")
        if notional.currency.currency_id != instrument.settlement_currency.currency_id:
            raise ValueError("CDS notional currency must match settlement currency.")
        if int(np.asarray(jax.device_get(notional.atoms))) <= 0:
            raise ValueError("CDS notional must be positive.")
        if not isinstance(recovery, RecoveryTerms):
            raise TypeError("recovery must be RecoveryTerms.")
        if recovery.convention != "par":
            raise ValueError("CDS protection currently requires recovery-of-par terms.")
        if not isinstance(payoff, CreditPayoff) or payoff.kind != "credit_default_swap":
            raise TypeError("payoff must be a CDS CreditPayoff.")
        if protection_side not in ("buy", "sell"):
            raise ValueError("protection_side must be 'buy' or 'sell'.")
        if type(accrued_on_default) is not bool:
            raise TypeError("accrued_on_default must be bool.")
        if default_boundary_side not in ("before_payment", "after_payment"):
            raise ValueError("Unsupported default boundary side.")
        identifier = _identifier(contract_id, "contract_id")
        times = _validate_payment_times(schedule, payment_times)
        spread = _scalar(running_spread, "running_spread", nonnegative=True)
        self.instrument = instrument
        self.schedule = schedule
        self.notional = notional
        self.running_spread = spread
        self.payment_times = times
        self.recovery = recovery
        self.payoff = payoff
        self.contract_id = identifier
        self.protection_side = protection_side
        self.accrued_on_default = accrued_on_default
        self.default_boundary_side = default_boundary_side
        self.resolved_id = _resolved_identity(
            "resolved-credit-default-swap",
            identifier,
            instrument,
            schedule,
            times,
            {
                "notional_atoms": int(np.asarray(jax.device_get(notional.atoms))),
                "running_spread": float(np.asarray(jax.device_get(spread))),
                "recovery_terms_id": recovery.terms_id,
                "payoff_id": payoff.payoff_id,
                "protection_side": protection_side,
                "accrued_on_default": accrued_on_default,
                "default_boundary_side": default_boundary_side,
            },
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        # Running premiums remain contingent on survival and are not currently
        # known unconditional obligations.
        return CashflowBatch(
            (),
            np.zeros((0,), dtype=float),
            (),
            capacity=self.schedule.capacity,
        )


class CreditEventCashflows(StrictModule):
    """Pathwise scheduled and default-triggered contract cashflows."""

    scheduled: Array
    default_settlement: Array
    accrued_on_default: Array
    default_settlement_times: Array
    default_index: Array
    occurred: Array
    valid: Array
    contract_id: str = eqx.field(static=True)
    default_law_id: str = eqx.field(static=True)


def credit_event_cashflows(
    contract: DefaultableBondContract | CreditDefaultSwapContract,
    defaults: DefaultEventState,
    /,
) -> CreditEventCashflows:
    """Resolve pathwise default events, including exact payment-grid boundaries."""

    if not isinstance(contract, (DefaultableBondContract, CreditDefaultSwapContract)):
        raise TypeError("contract must be a supported resolved credit contract.")
    if not isinstance(defaults, DefaultEventState):
        raise TypeError("defaults must be a DefaultEventState.")
    if defaults.reference_entity_id != contract.payoff.reference_entity_id:
        raise ValueError("Default events and credit payoff reference entities differ.")
    if defaults.recovery_terms_id != contract.recovery.terms_id:
        raise ValueError("Default events and contract recovery terms differ.")
    times = contract.payment_times
    active = contract.schedule.valid[None, :]
    event_time = defaults.default_times[:, None]
    if contract.default_boundary_side == "before_payment":
        paid_before_default = times[None, :] < event_time
    else:
        paid_before_default = times[None, :] <= event_time
    survives_slot = (~defaults.occurred[:, None]) | paid_before_default
    searchable_times = jnp.where(contract.schedule.valid, times, jnp.inf)
    default_index = jnp.searchsorted(
        searchable_times, defaults.default_times, side="left"
    )
    default_index = jnp.minimum(default_index, times.shape[0] - 1)
    maturity = times[jnp.sum(contract.schedule.valid) - 1]
    if contract.default_boundary_side == "before_payment":
        occurred_in_term = defaults.occurred & (defaults.default_times <= maturity)
    else:
        occurred_in_term = defaults.occurred & (defaults.default_times < maturity)
    event_slot = jnp.arange(times.shape[0])[None, :] == default_index[:, None]

    if isinstance(contract, DefaultableBondContract):
        if contract.recovery.convention != "par":
            raise ValueError(
                "Pathwise bond default settlement currently supports recovery of par only."
            )
        scheduled = jnp.where(
            active & survives_slot, contract.promised_amounts[None, :], 0.0
        )
        settlement = jnp.where(
            event_slot & occurred_in_term[:, None],
            _face_units(contract.face) * defaults.recoveries[:, None],
            0.0,
        )
        accrued = jnp.zeros_like(settlement)
    else:
        notional = _face_units(contract.notional)
        premium_base = (
            notional * contract.running_spread * contract.schedule.year_fractions
        )
        premium_sign = -1.0 if contract.protection_side == "buy" else 1.0
        protection_sign = -premium_sign
        scheduled = jnp.where(
            active & survives_slot,
            premium_sign * premium_base[None, :],
            0.0,
        )
        settlement = jnp.where(
            event_slot & occurred_in_term[:, None],
            protection_sign * notional * (1.0 - defaults.recoveries[:, None]),
            0.0,
        )
        starts = jnp.concatenate((jnp.zeros((1,), dtype=times.dtype), times[:-1]))
        elapsed = jnp.maximum(defaults.default_times - starts[default_index], 0.0)
        accrued_amount = premium_sign * notional * contract.running_spread * elapsed
        accrued = jnp.where(
            event_slot & occurred_in_term[:, None] & contract.accrued_on_default,
            accrued_amount[:, None],
            0.0,
        )
    valid = defaults.valid[:, None] & active
    if contract.recovery.timing == "at_default":
        settlement_times = defaults.default_times + contract.recovery.settlement_lag
    else:
        settlement_times = (
            searchable_times[default_index] + contract.recovery.settlement_lag
        )
    settlement_times = jnp.where(occurred_in_term & defaults.valid, settlement_times, 0.0)
    return CreditEventCashflows(
        jnp.where(valid, scheduled, 0.0),
        jnp.where(valid, settlement, 0.0),
        jnp.where(valid, accrued, 0.0),
        settlement_times,
        default_index,
        occurred_in_term,
        defaults.valid,
        contract.contract_id,
        defaults.law_id,
    )


__all__ = [
    "CreditDefaultSwapContract",
    "CreditEventCashflows",
    "CreditPayoff",
    "CreditPayoffKind",
    "DefaultBoundarySide",
    "DefaultEventState",
    "DefaultTimingConvention",
    "DefaultableBondContract",
    "ProtectionSide",
    "RecoveryConvention",
    "RecoveryTerms",
    "credit_event_cashflows",
]
