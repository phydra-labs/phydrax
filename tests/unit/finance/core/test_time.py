import jax


jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from phydrax.finance.core import (
    adjust_business_day,
    BusinessDayRule,
    CalendarSnapshot,
    DayCount,
    FinanceDate,
    FinancialTimestamp,
    resolve_schedule,
    ScheduleRule,
    TemporalAdmissibilityPolicy,
    year_fraction,
)


def _weekday_calendar(*holidays: FinanceDate) -> CalendarSnapshot:
    return CalendarSnapshot(
        "nyc",
        tuple(value.ordinal for value in holidays),
        (5, 6),
        "test calendar publication",
    )


def test_finance_date_and_day_counts_cover_leap_and_month_end_rules():
    leap_day = FinanceDate.from_iso("2024-02-29")
    assert leap_day.isoformat() == "2024-02-29"
    assert FinanceDate.from_ymd(2024, 3, 1).ordinal - leap_day.ordinal == 1

    start = FinanceDate.from_iso("2019-07-01")
    end = FinanceDate.from_iso("2020-07-01")
    expected_isda = 184.0 / 365.0 + 182.0 / 366.0
    assert year_fraction(start, end, DayCount.ACT_ACT_ISDA) == pytest.approx(
        expected_isda
    )
    assert year_fraction(start, end, DayCount.ACT_360) == pytest.approx(366.0 / 360.0)
    assert year_fraction(start, end, DayCount.ACT_365F) == pytest.approx(366.0 / 365.0)
    assert year_fraction(
        FinanceDate.from_iso("2024-02-29"),
        FinanceDate.from_iso("2024-03-31"),
        DayCount.THIRTY_360_US,
    ) == pytest.approx(30.0 / 360.0)
    assert year_fraction(
        FinanceDate.from_iso("2024-01-31"),
        FinanceDate.from_iso("2024-02-29"),
        DayCount.THIRTY_E_360,
    ) == pytest.approx(29.0 / 360.0)


def test_modified_following_does_not_cross_the_calendar_month():
    calendar = _weekday_calendar()
    saturday_month_end = FinanceDate.from_iso("2021-07-31")

    following = adjust_business_day(
        saturday_month_end, calendar, BusinessDayRule.FOLLOWING
    )
    modified = adjust_business_day(
        saturday_month_end, calendar, BusinessDayRule.MODIFIED_FOLLOWING
    )
    assert following.isoformat() == "2021-08-02"
    assert modified.isoformat() == "2021-07-30"


def test_schedule_resolution_preserves_eom_short_stub_and_neutral_padding():
    rule = ScheduleRule(
        FinanceDate.from_iso("2024-01-31"),
        FinanceDate.from_iso("2024-05-15"),
        1,
        "nyc",
        BusinessDayRule.UNADJUSTED,
        "short_back",
        True,
        0,
        DayCount.ACT_365F,
    )
    schedule = resolve_schedule(rule, _weekday_calendar(), capacity=6)

    assert schedule.period_count == 4
    assert [
        FinanceDate(int(value)).isoformat()
        for value in np.asarray(schedule.unadjusted_dates[:4])
    ] == ["2024-02-29", "2024-03-31", "2024-04-30", "2024-05-15"]
    np.testing.assert_array_equal(schedule.valid, (True, True, True, True, False, False))
    np.testing.assert_array_equal(schedule.unadjusted_dates[4:], (0, 0))
    np.testing.assert_array_equal(schedule.year_fractions[4:], (0.0, 0.0))
    assert schedule.year_fractions[3] == pytest.approx(15.0 / 365.0)


def test_long_front_stub_removes_the_first_regular_boundary():
    rule = ScheduleRule(
        FinanceDate.from_iso("2024-01-15"),
        FinanceDate.from_iso("2024-05-31"),
        1,
        "nyc",
        BusinessDayRule.UNADJUSTED,
        "long_front",
        True,
        0,
        DayCount.ACT_365F,
    )
    schedule = resolve_schedule(rule, _weekday_calendar(), capacity=5)

    assert FinanceDate(int(schedule.accrual_start_dates[0])).isoformat() == "2024-01-15"
    assert FinanceDate(int(schedule.accrual_end_dates[0])).isoformat() == "2024-02-29"
    assert schedule.period_count == 4


def test_financial_timestamp_keeps_event_clock_independent_of_publication_clock():
    point_in_time = TemporalAdmissibilityPolicy(True, True, True)
    timestamp = FinancialTimestamp(300, 100, 110, 120, "vendor-v1", point_in_time)

    assert timestamp.event_ns == 300
    assert timestamp.epoch_nanoseconds == 300
    assert timestamp.available_ns == 120
    assert timestamp.admissible_at(120)
    assert not timestamp.admissible_at(119)

    no_future_effect = TemporalAdmissibilityPolicy(True, True, False)
    with pytest.raises(ValueError, match="admissibility policy"):
        FinancialTimestamp(121, 100, 110, 120, "vendor-v1", no_future_effect)


def test_calendar_semantic_identity_is_separate_from_snapshot_content():
    semantic = _weekday_calendar()
    revised = _weekday_calendar(FinanceDate.from_iso("2026-01-01"))

    assert semantic.calendar_id == revised.calendar_id
    assert semantic.snapshot_id != revised.snapshot_id
