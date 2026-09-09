#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..core import (
    adjust_business_day,
    BusinessDayRule,
    CalendarSnapshot,
    Currency,
    FinanceDate,
)


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    BERMUDAN = "bermudan"
    AMERICAN = "american"


class SettlementType(str, Enum):
    CASH = "cash"
    PHYSICAL = "physical"


class ExerciseSchedule(StrictModule, NonTrainableState):
    """Fixed-shape exercise dates, or two endpoints for an American window."""

    exercise_ordinals: Array
    valid_mask: Array
    style: ExerciseStyle = eqx.field(static=True)
    notice_days: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        dates: Sequence[FinanceDate],
        /,
        *,
        style: ExerciseStyle,
        notice_days: int = 0,
        capacity: int | None = None,
    ):
        values = tuple(dates)
        if not values or not all(isinstance(value, FinanceDate) for value in values):
            raise TypeError("dates must contain at least one FinanceDate.")
        if not isinstance(style, ExerciseStyle):
            raise TypeError("style must be an ExerciseStyle.")
        if (
            isinstance(notice_days, bool)
            or not isinstance(notice_days, int)
            or notice_days < 0
        ):
            raise ValueError("notice_days must be a nonnegative integer.")
        ordinals = tuple(value.ordinal for value in values)
        if any(
            right <= left for left, right in zip(ordinals[:-1], ordinals[1:], strict=True)
        ):
            raise ValueError("Exercise dates must be strictly increasing.")
        if style is ExerciseStyle.EUROPEAN and len(values) != 1:
            raise ValueError("European exercise requires exactly one date.")
        if style is ExerciseStyle.AMERICAN and len(values) != 2:
            raise ValueError("American exercise requires inclusive window endpoints.")
        if capacity is None:
            capacity_ = len(values)
        elif (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < len(values)
        ):
            raise ValueError("capacity must be at least the number of exercise dates.")
        else:
            capacity_ = capacity
        padded = np.zeros((capacity_,), dtype=np.int32)
        valid = np.zeros((capacity_,), dtype=bool)
        padded[: len(values)] = np.asarray(ordinals, dtype=np.int32)
        valid[: len(values)] = True
        self.exercise_ordinals = jnp.asarray(padded)
        self.valid_mask = jnp.asarray(valid)
        self.style = style
        self.notice_days = notice_days
        self.capacity = capacity_
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "financial-exercise-schedule",
                "exercise_ordinals": padded.tolist(),
                "valid_mask": valid.tolist(),
                "style": style.value,
                "notice_days": notice_days,
            }
        )

    @property
    def active_count(self) -> int:
        return int(np.sum(np.asarray(self.valid_mask)))

    def pad(self, capacity: int, /) -> ExerciseSchedule:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < self.active_count
        ):
            raise ValueError("capacity must not truncate exercise dates.")
        dates = tuple(
            FinanceDate(int(value))
            for value in np.asarray(self.exercise_ordinals)[np.asarray(self.valid_mask)]
        )
        return ExerciseSchedule(
            dates,
            style=self.style,
            notice_days=self.notice_days,
            capacity=capacity,
        )

    def to_record(self) -> Mapping[str, Any]:
        return {
            "exercise_ordinals": np.asarray(self.exercise_ordinals).tolist(),
            "valid_mask": np.asarray(self.valid_mask).tolist(),
            "style": self.style.value,
            "notice_days": self.notice_days,
            "schedule_id": self.schedule_id,
        }


class SettlementTerms(StrictModule, NonTrainableState):
    """Explicit contractual settlement convention, separate from market FX data."""

    currency: Currency = eqx.field(static=True)
    settlement_lag_days: int = eqx.field(static=True)
    calendar_id: str = eqx.field(static=True)
    business_day_rule: BusinessDayRule = eqx.field(static=True)
    settlement_type: SettlementType = eqx.field(static=True)
    terms_id: str = eqx.field(static=True)

    def __init__(
        self,
        currency: Currency,
        /,
        *,
        settlement_lag_days: int = 0,
        calendar_id: str,
        business_day_rule: BusinessDayRule,
        settlement_type: SettlementType = SettlementType.CASH,
    ):
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        if (
            isinstance(settlement_lag_days, bool)
            or not isinstance(settlement_lag_days, int)
            or settlement_lag_days < 0
        ):
            raise ValueError("settlement_lag_days must be a nonnegative integer.")
        if not isinstance(calendar_id, str) or not calendar_id.strip():
            raise ValueError("calendar_id must be a non-empty string.")
        if not isinstance(business_day_rule, BusinessDayRule):
            raise TypeError("business_day_rule must be a BusinessDayRule.")
        if not isinstance(settlement_type, SettlementType):
            raise TypeError("settlement_type must be a SettlementType.")
        self.currency = currency
        self.settlement_lag_days = settlement_lag_days
        self.calendar_id = calendar_id.strip()
        self.business_day_rule = business_day_rule
        self.settlement_type = settlement_type
        self.terms_id = canonical_fingerprint(
            {
                "kind": "financial-settlement-terms",
                "currency_id": currency.currency_id,
                "settlement_lag_days": settlement_lag_days,
                "calendar_id": self.calendar_id,
                "business_day_rule": business_day_rule.value,
                "settlement_type": settlement_type.value,
            }
        )

    def resolve_date(
        self,
        trade_date: FinanceDate,
        calendar: CalendarSnapshot,
        /,
    ) -> FinanceDate:
        """Resolve the settlement date on the host using the pinned calendar."""

        if not isinstance(trade_date, FinanceDate):
            raise TypeError("trade_date must be a FinanceDate.")
        if not isinstance(calendar, CalendarSnapshot):
            raise TypeError("calendar must be a CalendarSnapshot.")
        if calendar.calendar_id != self.calendar_id:
            raise ValueError("Settlement terms and calendar identifiers do not match.")
        candidate = trade_date
        remaining = self.settlement_lag_days
        while remaining:
            candidate = candidate.add_days(1)
            if calendar.is_business_day(candidate):
                remaining -= 1
        return adjust_business_day(candidate, calendar, self.business_day_rule)

    def to_record(self) -> Mapping[str, Any]:
        return {
            "currency": self.currency.code,
            "settlement_lag_days": self.settlement_lag_days,
            "calendar_id": self.calendar_id,
            "business_day_rule": self.business_day_rule.value,
            "settlement_type": self.settlement_type.value,
            "terms_id": self.terms_id,
        }


__all__ = [
    "ExerciseSchedule",
    "ExerciseStyle",
    "SettlementTerms",
    "SettlementType",
]
