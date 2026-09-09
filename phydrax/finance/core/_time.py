#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import calendar as _calendar
from bisect import bisect_left
from collections.abc import Sequence
from datetime import date as _date
from enum import Enum
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule

from ._identifiers import _canonical_text, _token


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_DATE_MIN = _date.min.toordinal()
_DATE_MAX = _date.max.toordinal()


class BusinessDayRule(str, Enum):
    FOLLOWING = "following"
    MODIFIED_FOLLOWING = "modified_following"
    PRECEDING = "preceding"
    MODIFIED_PRECEDING = "modified_preceding"
    UNADJUSTED = "unadjusted"


class DayCount(str, Enum):
    ACT_360 = "act_360"
    ACT_365F = "act_365f"
    ACT_ACT_ISDA = "act_act_isda"
    THIRTY_360_US = "30_360_us"
    THIRTY_E_360 = "30e_360"


StubRule: TypeAlias = Literal[
    "none", "short_front", "long_front", "short_back", "long_back"
]
_STUB_RULES = frozenset(("none", "short_front", "long_front", "short_back", "long_back"))


def _integral(value: object, name: str, /, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < lower or result > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return result


def _ordinal(value: FinanceDate | int, name: str, /) -> int:
    if isinstance(value, FinanceDate):
        return value.ordinal
    return _integral(value, name, lower=_DATE_MIN, upper=_DATE_MAX)


def _business_day_rule(value: BusinessDayRule | str, /) -> BusinessDayRule:
    if isinstance(value, BusinessDayRule):
        return value
    if value not in tuple(rule.value for rule in BusinessDayRule):
        raise ValueError("business_day_rule is not supported.")
    return BusinessDayRule(value)


def _day_count(value: DayCount | str, /) -> DayCount:
    if isinstance(value, DayCount):
        return value
    if value not in tuple(convention.value for convention in DayCount):
        raise ValueError("day_count is not supported.")
    return DayCount(value)


class FinanceDate(StrictModule):
    """A proleptic-Gregorian civil date stored as a Python civil ordinal."""

    ordinal: int = eqx.field(static=True)

    def __init__(self, ordinal: int, /):
        self.ordinal = _integral(
            ordinal, "finance date ordinal", lower=_DATE_MIN, upper=_DATE_MAX
        )

    @classmethod
    def from_ymd(cls, year: int, month: int, day: int, /) -> FinanceDate:
        """Construct from a validated proleptic-Gregorian year, month, and day."""
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (year, month, day)
        ):
            raise TypeError("year, month, and day must be integers.")
        return cls(_date(int(year), int(month), int(day)).toordinal())

    @classmethod
    def from_iso(cls, value: str, /) -> FinanceDate:
        """Construct from canonical ISO ``YYYY-MM-DD`` text."""
        if not isinstance(value, str):
            raise TypeError("finance date ISO value must be a string.")
        parsed = _date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("finance date must use canonical YYYY-MM-DD form.")
        return cls(parsed.toordinal())

    def to_date(self) -> _date:
        return _date.fromordinal(self.ordinal)

    def isoformat(self) -> str:
        return self.to_date().isoformat()

    @property
    def year(self) -> int:
        return self.to_date().year

    @property
    def month(self) -> int:
        return self.to_date().month

    @property
    def day(self) -> int:
        return self.to_date().day

    def add_days(self, days: int, /) -> FinanceDate:
        offset = _integral(days, "date offset", lower=-_DATE_MAX, upper=_DATE_MAX)
        return FinanceDate(self.ordinal + offset)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, FinanceDate):
            return NotImplemented
        return self.ordinal < other.ordinal

    def __le__(self, other: object) -> bool:
        if not isinstance(other, FinanceDate):
            return NotImplemented
        return self.ordinal <= other.ordinal


class TemporalAdmissibilityPolicy(StrictModule):
    """Static point-in-time clock-order policy.

    Event time is intentionally unrelated to publication time. Only when future
    effective events are forbidden is event time constrained, and then only against
    the availability clock.
    """

    require_published_before_received: bool = eqx.field(static=True)
    require_received_before_available: bool = eqx.field(static=True)
    allow_future_effective_event: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        require_published_before_received: bool,
        require_received_before_available: bool,
        allow_future_effective_event: bool,
        /,
    ):
        values = (
            require_published_before_received,
            require_received_before_available,
            allow_future_effective_event,
        )
        if any(type(value) is not bool for value in values):
            raise TypeError("temporal admissibility policy flags must be bool values.")
        self.require_published_before_received = require_published_before_received
        self.require_received_before_available = require_received_before_available
        self.allow_future_effective_event = allow_future_effective_event
        self.policy_id = canonical_fingerprint(
            {
                "kind": "temporal_admissibility_policy",
                "require_published_before_received": require_published_before_received,
                "require_received_before_available": require_received_before_available,
                "allow_future_effective_event": allow_future_effective_event,
            }
        )

    def admits(
        self,
        event_ns: int,
        published_ns: int,
        received_ns: int,
        available_ns: int,
        /,
    ) -> bool:
        clocks = tuple(
            _integral(value, "UTC nanosecond clock", lower=_INT64_MIN, upper=_INT64_MAX)
            for value in (event_ns, published_ns, received_ns, available_ns)
        )
        event, published, received, available = clocks
        return (
            (not self.require_published_before_received or published <= received)
            and (not self.require_received_before_available or received <= available)
            and (self.allow_future_effective_event or event <= available)
        )


class FinancialTimestamp(StrictModule):
    """Four preserved UTC signed-int64 clocks plus vintage and admissibility policy."""

    event_ns: int = eqx.field(static=True)
    published_ns: int = eqx.field(static=True)
    received_ns: int = eqx.field(static=True)
    available_ns: int = eqx.field(static=True)
    vintage_id: str = eqx.field(static=True)
    policy: TemporalAdmissibilityPolicy = eqx.field(static=True)

    def __init__(
        self,
        event_ns: int,
        published_ns: int,
        received_ns: int,
        available_ns: int,
        vintage_id: str,
        policy: TemporalAdmissibilityPolicy,
        /,
    ):
        if not isinstance(policy, TemporalAdmissibilityPolicy):
            raise TypeError("policy must be a TemporalAdmissibilityPolicy.")
        clocks = tuple(
            _integral(value, "UTC nanosecond clock", lower=_INT64_MIN, upper=_INT64_MAX)
            for value in (event_ns, published_ns, received_ns, available_ns)
        )
        if not policy.admits(*clocks):
            raise ValueError("financial timestamp violates its admissibility policy.")
        self.event_ns, self.published_ns, self.received_ns, self.available_ns = clocks
        self.vintage_id = _token(vintage_id, "vintage_id")
        self.policy = policy

    @property
    def epoch_nanoseconds(self) -> int:
        """Return the event clock; availability remains explicitly ``available_ns``."""
        return self.event_ns

    def admissible_at(self, as_of_ns: int, /) -> bool:
        """Return whether this vintage was available at the requested UTC clock."""
        as_of = _integral(
            as_of_ns, "as_of UTC nanosecond clock", lower=_INT64_MIN, upper=_INT64_MAX
        )
        return self.available_ns <= as_of


class CalendarSnapshot(StrictModule):
    """Caller-supplied Gregorian weekend/holiday semantics with fixed provenance."""

    calendar_id: str = eqx.field(static=True)
    holiday_ordinals: tuple[int, ...] = eqx.field(static=True)
    weekend_weekdays: tuple[int, ...] = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        calendar_id: str,
        holiday_ordinals: Sequence[FinanceDate | int],
        weekend_weekdays: Sequence[int],
        provenance: str,
        /,
    ):
        if isinstance(holiday_ordinals, (str, bytes)) or not isinstance(
            holiday_ordinals, Sequence
        ):
            raise TypeError("holiday_ordinals must be a sequence of civil ordinals.")
        if isinstance(weekend_weekdays, (str, bytes)) or not isinstance(
            weekend_weekdays, Sequence
        ):
            raise TypeError("weekend_weekdays must be a sequence of weekday indices.")
        holidays = tuple(
            sorted(_ordinal(value, "holiday ordinal") for value in holiday_ordinals)
        )
        if len(set(holidays)) != len(holidays):
            raise ValueError("holiday_ordinals must be unique.")
        weekends = tuple(
            sorted(
                _integral(value, "weekend weekday", lower=0, upper=6)
                for value in weekend_weekdays
            )
        )
        if len(set(weekends)) != len(weekends):
            raise ValueError("weekend_weekdays must be unique.")
        if len(weekends) == 7:
            raise ValueError(
                "calendar snapshot must contain at least one working weekday."
            )
        calendar_id_ = _token(calendar_id, "calendar_id")
        provenance_ = _canonical_text(provenance, "calendar provenance")
        self.calendar_id = calendar_id_
        self.holiday_ordinals = holidays
        self.weekend_weekdays = weekends
        self.provenance = provenance_
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "calendar_snapshot",
                "calendar_id": calendar_id_,
                "holiday_ordinals": list(holidays),
                "weekend_weekdays": list(weekends),
                "provenance": provenance_,
            }
        )

    def is_business_day(self, value: FinanceDate | int, /) -> bool:
        ordinal = _ordinal(value, "business-day query")
        holiday_index = bisect_left(self.holiday_ordinals, ordinal)
        holiday = (
            holiday_index < len(self.holiday_ordinals)
            and self.holiday_ordinals[holiday_index] == ordinal
        )
        return (
            _date.fromordinal(ordinal).weekday() not in self.weekend_weekdays
            and not holiday
        )


class ScheduleRule(StrictModule):
    """Host-resolved periodic schedule semantics."""

    start: FinanceDate = eqx.field(static=True)
    end: FinanceDate = eqx.field(static=True)
    frequency_months: int = eqx.field(static=True)
    calendar_id: str = eqx.field(static=True)
    business_day_rule: BusinessDayRule = eqx.field(static=True)
    stub_rule: StubRule = eqx.field(static=True)
    end_of_month: bool = eqx.field(static=True)
    payment_lag_days: int = eqx.field(static=True)
    day_count: DayCount = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        start: FinanceDate,
        end: FinanceDate,
        frequency_months: int,
        calendar_id: str,
        business_day_rule: BusinessDayRule | str,
        stub_rule: StubRule,
        end_of_month: bool,
        payment_lag_days: int,
        day_count: DayCount | str,
        /,
    ):
        if not isinstance(start, FinanceDate) or not isinstance(end, FinanceDate):
            raise TypeError("schedule start and end must be FinanceDate values.")
        if start.ordinal >= end.ordinal:
            raise ValueError("schedule start must precede schedule end.")
        frequency = _integral(frequency_months, "frequency_months", lower=1, upper=1200)
        if stub_rule not in _STUB_RULES:
            raise ValueError("stub_rule is not supported.")
        if type(end_of_month) is not bool:
            raise TypeError("end_of_month must be bool.")
        lag = _integral(payment_lag_days, "payment_lag_days", lower=0, upper=10000)
        calendar_id_ = _token(calendar_id, "calendar_id")
        business_rule = _business_day_rule(business_day_rule)
        day_count_ = _day_count(day_count)
        self.start = start
        self.end = end
        self.frequency_months = frequency
        self.calendar_id = calendar_id_
        self.business_day_rule = business_rule
        self.stub_rule = stub_rule
        self.end_of_month = end_of_month
        self.payment_lag_days = lag
        self.day_count = day_count_
        self.rule_id = canonical_fingerprint(
            {
                "kind": "schedule_rule",
                "start": start.ordinal,
                "end": end.ordinal,
                "frequency_months": frequency,
                "calendar_id": calendar_id_,
                "business_day_rule": business_rule.value,
                "stub_rule": stub_rule,
                "end_of_month": end_of_month,
                "payment_lag_days": lag,
                "day_count": day_count_.value,
            }
        )


class ResolvedSchedule(StrictModule):
    """Fixed-capacity aligned period arrays produced by host calendar resolution."""

    unadjusted_dates: Array
    adjusted_dates: Array
    accrual_start_dates: Array
    accrual_end_dates: Array
    payment_dates: Array
    year_fractions: Array
    valid: Array
    calendar_id: str = eqx.field(static=True)
    calendar_snapshot_id: str = eqx.field(static=True)
    schedule_rule_id: str = eqx.field(static=True)
    day_count: DayCount = eqx.field(static=True)
    capacity: int = eqx.field(static=True)

    def __init__(
        self,
        unadjusted_dates: ArrayLike,
        adjusted_dates: ArrayLike,
        accrual_start_dates: ArrayLike,
        accrual_end_dates: ArrayLike,
        payment_dates: ArrayLike,
        year_fractions: ArrayLike,
        valid: ArrayLike,
        calendar_id: str,
        calendar_snapshot_id: str,
        schedule_rule_id: str,
        day_count: DayCount | str,
        /,
    ):
        raw_dates = tuple(
            np.asarray(value)
            for value in (
                unadjusted_dates,
                adjusted_dates,
                accrual_start_dates,
                accrual_end_dates,
                payment_dates,
            )
        )
        if any(array.ndim != 1 for array in raw_dates):
            raise ValueError("resolved schedule date arrays must be one-dimensional.")
        shape = raw_dates[0].shape
        if shape == (0,) or any(array.shape != shape for array in raw_dates):
            raise ValueError(
                "resolved schedule date arrays must share a non-empty shape."
            )
        if any(not np.issubdtype(array.dtype, np.integer) for array in raw_dates):
            raise TypeError("resolved schedule dates must have integer dtype.")
        if any(np.any((array < 0) | (array > _DATE_MAX)) for array in raw_dates):
            raise ValueError("resolved schedule date ordinals are out of range.")
        fractions = np.asarray(year_fractions)
        mask = np.asarray(valid)
        if fractions.shape != shape or not np.issubdtype(fractions.dtype, np.floating):
            raise TypeError(
                "year_fractions must be a floating array matching date shape."
            )
        if mask.shape != shape or mask.dtype != np.dtype(bool):
            raise TypeError("valid must be a boolean array matching date shape.")
        count = int(np.sum(mask))
        if count < 1 or not np.array_equal(mask, np.arange(shape[0]) < count):
            raise ValueError(
                "resolved schedule valid entries must form a non-empty prefix."
            )
        active_dates = tuple(
            array[:count].astype(np.int64, copy=False) for array in raw_dates
        )
        inactive_dates = tuple(array[count:] for array in raw_dates)
        if any(np.any(array != 0) for array in inactive_dates) or np.any(
            fractions[count:] != 0
        ):
            raise ValueError(
                "inactive resolved schedule slots must use neutral zero padding."
            )
        unadjusted, adjusted, starts, ends, payments = active_dates
        if (
            np.any(unadjusted <= 0)
            or np.any(adjusted <= 0)
            or np.any(starts <= 0)
            or np.any(ends <= starts)
            or np.any(payments < ends)
            or np.any(np.diff(unadjusted) <= 0)
            or np.any(np.diff(adjusted) <= 0)
            or np.any(np.diff(starts) <= 0)
            or not np.array_equal(adjusted, ends)
            or (count > 1 and not np.array_equal(starts[1:], ends[:-1]))
        ):
            raise ValueError(
                "active resolved schedule dates are inconsistent or unordered."
            )
        if np.any(~np.isfinite(fractions[:count])) or np.any(fractions[:count] <= 0):
            raise ValueError(
                "active resolved schedule year fractions must be finite and positive."
            )
        self.unadjusted_dates = jnp.asarray(raw_dates[0], dtype=jnp.int32)
        self.adjusted_dates = jnp.asarray(raw_dates[1], dtype=jnp.int32)
        self.accrual_start_dates = jnp.asarray(raw_dates[2], dtype=jnp.int32)
        self.accrual_end_dates = jnp.asarray(raw_dates[3], dtype=jnp.int32)
        self.payment_dates = jnp.asarray(raw_dates[4], dtype=jnp.int32)
        self.year_fractions = jnp.asarray(fractions)
        self.valid = jnp.asarray(mask)
        self.calendar_id = _token(calendar_id, "calendar_id")
        self.calendar_snapshot_id = _token(calendar_snapshot_id, "calendar_snapshot_id")
        self.schedule_rule_id = _token(schedule_rule_id, "schedule_rule_id")
        self.day_count = _day_count(day_count)
        self.capacity = int(shape[0])

    @property
    def period_count(self) -> int:
        return int(np.sum(np.asarray(self.valid)))


def adjust_business_day(
    value: FinanceDate,
    calendar: CalendarSnapshot,
    rule: BusinessDayRule | str,
    /,
) -> FinanceDate:
    """Apply one explicit business-day rule using caller-supplied calendar semantics."""
    if not isinstance(value, FinanceDate):
        raise TypeError("value must be a FinanceDate.")
    if not isinstance(calendar, CalendarSnapshot):
        raise TypeError("calendar must be a CalendarSnapshot.")
    rule_ = _business_day_rule(rule)
    if rule_ is BusinessDayRule.UNADJUSTED or calendar.is_business_day(value):
        return value

    def seek(direction: int) -> FinanceDate:
        candidate = value
        while not calendar.is_business_day(candidate):
            candidate = candidate.add_days(direction)
        return candidate

    if rule_ in (BusinessDayRule.FOLLOWING, BusinessDayRule.MODIFIED_FOLLOWING):
        candidate = seek(1)
        if rule_ is BusinessDayRule.MODIFIED_FOLLOWING and candidate.month != value.month:
            return seek(-1)
        return candidate
    candidate = seek(-1)
    if rule_ is BusinessDayRule.MODIFIED_PRECEDING and candidate.month != value.month:
        return seek(1)
    return candidate


def _is_last_day_of_february(value: _date, /) -> bool:
    return value.month == 2 and value.day == _calendar.monthrange(value.year, 2)[1]


def year_fraction(
    start: FinanceDate,
    end: FinanceDate,
    convention: DayCount | str,
    /,
) -> float:
    """Return a host-resolved year fraction under a standard named convention."""
    if not isinstance(start, FinanceDate) or not isinstance(end, FinanceDate):
        raise TypeError("year-fraction endpoints must be FinanceDate values.")
    if end.ordinal < start.ordinal:
        raise ValueError("year-fraction end must not precede start.")
    convention_ = _day_count(convention)
    actual_days = end.ordinal - start.ordinal
    if convention_ is DayCount.ACT_360:
        return actual_days / 360.0
    if convention_ is DayCount.ACT_365F:
        return actual_days / 365.0
    if convention_ is DayCount.ACT_ACT_ISDA:
        current = start.to_date()
        terminal = end.to_date()
        fraction = 0.0
        while current < terminal:
            segment_end = (
                terminal
                if current.year == _date.max.year
                else min(_date(current.year + 1, 1, 1), terminal)
            )
            denominator = 366.0 if _calendar.isleap(current.year) else 365.0
            fraction += (segment_end.toordinal() - current.toordinal()) / denominator
            current = segment_end
        return fraction

    left = start.to_date()
    right = end.to_date()
    d1 = left.day
    d2 = right.day
    if convention_ is DayCount.THIRTY_E_360:
        d1 = min(d1, 30)
        d2 = min(d2, 30)
    else:
        left_feb_eom = _is_last_day_of_february(left)
        right_feb_eom = _is_last_day_of_february(right)
        if left_feb_eom:
            d1 = 30
        if right_feb_eom and left_feb_eom:
            d2 = 30
        if d1 == 31:
            d1 = 30
        if d2 == 31 and d1 >= 30:
            d2 = 30
    days_360 = (right.year - left.year) * 360 + (right.month - left.month) * 30 + d2 - d1
    return days_360 / 360.0


def _add_months(value: FinanceDate, months: int, end_of_month: bool, /) -> FinanceDate:
    source = value.to_date()
    absolute_month = source.year * 12 + (source.month - 1) + months
    year, month_zero = divmod(absolute_month, 12)
    month = month_zero + 1
    last_day = _calendar.monthrange(year, month)[1]
    day = last_day if end_of_month else min(source.day, last_day)
    return FinanceDate.from_ymd(year, month, day)


def _schedule_boundaries(rule: ScheduleRule, /) -> list[FinanceDate]:
    front = rule.stub_rule in ("short_front", "long_front")
    if front:
        reverse = [rule.end]
        step = 1
        candidate = _add_months(
            rule.end, -step * rule.frequency_months, rule.end_of_month
        )
        while candidate.ordinal > rule.start.ordinal:
            reverse.append(candidate)
            step += 1
            candidate = _add_months(
                rule.end, -step * rule.frequency_months, rule.end_of_month
            )
        has_stub = candidate.ordinal != rule.start.ordinal
        reverse.append(rule.start)
        boundaries = list(reversed(reverse))
        if has_stub and rule.stub_rule == "long_front" and len(boundaries) > 2:
            del boundaries[1]
        return boundaries

    boundaries = [rule.start]
    step = 1
    candidate = _add_months(rule.start, step * rule.frequency_months, rule.end_of_month)
    while candidate.ordinal < rule.end.ordinal:
        boundaries.append(candidate)
        step += 1
        candidate = _add_months(
            rule.start, step * rule.frequency_months, rule.end_of_month
        )
    has_stub = candidate.ordinal != rule.end.ordinal
    if has_stub and rule.stub_rule == "none":
        raise ValueError(
            "schedule endpoints do not fit the frequency and stubs are disabled."
        )
    boundaries.append(rule.end)
    if has_stub and rule.stub_rule == "long_back" and len(boundaries) > 2:
        del boundaries[-2]
    return boundaries


def _add_business_days(
    value: FinanceDate, days: int, calendar: CalendarSnapshot, /
) -> FinanceDate:
    candidate = value
    remaining = days
    while remaining:
        candidate = candidate.add_days(1)
        if calendar.is_business_day(candidate):
            remaining -= 1
    return candidate


def resolve_schedule(
    rule: ScheduleRule,
    calendar: CalendarSnapshot,
    /,
    *,
    capacity: int,
) -> ResolvedSchedule:
    """Resolve host calendar semantics into neutral-padded fixed-capacity arrays."""
    if not isinstance(rule, ScheduleRule):
        raise TypeError("rule must be a ScheduleRule.")
    if not isinstance(calendar, CalendarSnapshot):
        raise TypeError("calendar must be a CalendarSnapshot.")
    if rule.calendar_id != calendar.calendar_id:
        raise ValueError("schedule rule and calendar identifiers must match.")
    capacity_ = _integral(capacity, "schedule capacity", lower=1, upper=1_000_000)
    boundaries = _schedule_boundaries(rule)
    period_count = len(boundaries) - 1
    if period_count > capacity_:
        raise ValueError("resolved schedule exceeds its fixed capacity.")
    adjusted_boundaries = [
        adjust_business_day(value, calendar, rule.business_day_rule)
        for value in boundaries
    ]
    unadjusted = np.zeros((capacity_,), dtype=np.int32)
    adjusted = np.zeros((capacity_,), dtype=np.int32)
    starts = np.zeros((capacity_,), dtype=np.int32)
    ends = np.zeros((capacity_,), dtype=np.int32)
    payments = np.zeros((capacity_,), dtype=np.int32)
    fractions = np.zeros((capacity_,), dtype=float)
    valid = np.zeros((capacity_,), dtype=bool)
    for index in range(period_count):
        accrual_start = adjusted_boundaries[index]
        accrual_end = adjusted_boundaries[index + 1]
        payment = _add_business_days(accrual_end, rule.payment_lag_days, calendar)
        unadjusted[index] = boundaries[index + 1].ordinal
        adjusted[index] = accrual_end.ordinal
        starts[index] = accrual_start.ordinal
        ends[index] = accrual_end.ordinal
        payments[index] = payment.ordinal
        fractions[index] = year_fraction(accrual_start, accrual_end, rule.day_count)
        valid[index] = True
    return ResolvedSchedule(
        unadjusted,
        adjusted,
        starts,
        ends,
        payments,
        fractions,
        valid,
        rule.calendar_id,
        calendar.snapshot_id,
        rule.rule_id,
        rule.day_count,
    )


__all__ = [
    "adjust_business_day",
    "BusinessDayRule",
    "CalendarSnapshot",
    "DayCount",
    "FinanceDate",
    "FinancialTimestamp",
    "ResolvedSchedule",
    "resolve_schedule",
    "ScheduleRule",
    "StubRule",
    "TemporalAdmissibilityPolicy",
    "year_fraction",
]
