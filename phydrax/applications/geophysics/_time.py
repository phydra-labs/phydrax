# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Host-side CF calendar arithmetic; compiled models receive numerical durations only."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..._fingerprint import canonical_fingerprint


_CALENDARS = {
    "standard": "standard",
    "gregorian": "standard",
    "proleptic_gregorian": "proleptic_gregorian",
    "noleap": "noleap",
    "365_day": "noleap",
    "all_leap": "all_leap",
    "366_day": "all_leap",
    "360_day": "360_day",
}
_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
_DATE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?)?Z?\Z"
)
_DAY_US = 86_400_000_000


def _month_lengths(year: int, calendar: str) -> tuple[int, ...]:
    if calendar == "360_day":
        return (30,) * 12
    if calendar == "noleap":
        leap = False
    elif calendar == "all_leap":
        leap = True
    elif calendar == "standard" and year <= 1582:
        leap = year % 4 == 0
    else:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _day_number(year: int, month: int, day: int, calendar: str) -> int:
    """Integer astronomical day for civil calendars, ordinal for model calendars."""
    if calendar in ("noleap", "all_leap", "360_day"):
        length = {"noleap": 365, "all_leap": 366, "360_day": 360}[calendar]
        return (
            (year - 1) * length
            + sum(_month_lengths(year, calendar)[: month - 1])
            + day
            - 1
        )
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    base = day + (153 * m + 2) // 5 + 365 * y + y // 4
    if calendar == "standard" and (year, month, day) <= (1582, 10, 4):
        return base - 32083
    return base - y // 100 + y // 400 - 32045


def _parse(text: str, calendar: str) -> tuple[int, int, int, int, int, int, int]:
    if not isinstance(text, str):
        raise TypeError("Calendar dates must be ISO date or date-time strings.")
    match = _DATE.fullmatch(text)
    if match is None:
        raise ValueError(
            "Dates require YYYY-MM-DD[THH:MM:SS[.ffffff]][Z]; offsets are not implicit."
        )
    year, month, day = (int(match.group(i)) for i in (1, 2, 3))
    hour, minute, second = (int(match.group(i) or 0) for i in (4, 5, 6))
    microsecond = int((match.group(7) or "0").ljust(6, "0"))
    if not (1 <= year <= 9999 and 1 <= month <= 12):
        raise ValueError("Calendar dates require years 1..9999 and months 1..12.")
    if not (1 <= day <= _month_lengths(year, calendar)[month - 1]):
        raise ValueError("Day is invalid in the declared calendar.")
    if calendar == "standard" and (1582, 10, 5) <= (year, month, day) <= (1582, 10, 14):
        raise ValueError(
            "Dates in the Gregorian transition gap are undefined in the standard calendar."
        )
    if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
        raise ValueError(
            "Time is invalid; leap seconds require an explicit external time-scale conversion."
        )
    return year, month, day, hour, minute, second, microsecond


def _ticks(parts: tuple[int, int, int, int, int, int, int], calendar: str) -> int:
    year, month, day, hour, minute, second, microsecond = parts
    return (
        _day_number(year, month, day, calendar) * _DAY_US
        + ((hour * 60 + minute) * 60 + second) * 1_000_000
        + microsecond
    )


def _format(parts: tuple[int, int, int, int, int, int, int]) -> str:
    y, m, d, hh, mm, ss, us = parts
    suffix = f".{us:06d}".rstrip("0") if us else ""
    return f"{y:04d}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}:{ss:02d}{suffix}"


def _from_ticks(ticks: int, calendar: str) -> tuple[int, int, int, int, int, int, int]:
    day_number, remainder = divmod(ticks, _DAY_US)
    if not (
        _day_number(1, 1, 1, calendar) <= day_number < _day_number(10000, 1, 1, calendar)
    ):
        raise ValueError("Decoded calendar date is outside years 1..9999.")
    low, high = 1, 10000
    while high - low > 1:
        mid = (low + high) // 2
        if _day_number(mid, 1, 1, calendar) <= day_number:
            low = mid
        else:
            high = mid
    year = low
    month = 12
    while _day_number(year, month, 1, calendar) > day_number:
        month -= 1
    day = 1 + day_number - _day_number(year, month, 1, calendar)
    if calendar == "standard" and (year, month) == (1582, 10) and day >= 5:
        day += 10
    seconds, us = divmod(remainder, 1_000_000)
    hour, seconds = divmod(seconds, 3600)
    minute, second = divmod(seconds, 60)
    return year, month, day, hour, minute, second, us


@dataclass(frozen=True, slots=True, init=False)
class GeophysicalTimeSpec:
    """Calendar and epoch identity for a continuous model clock; not a leap-second clock."""

    calendar: str
    epoch: str
    unit: str
    time_id: str = field(init=False)
    _epoch_ticks: int = field(init=False, repr=False)

    def __init__(
        self,
        calendar: str = "proleptic_gregorian",
        epoch: str = "2000-01-01T00:00:00",
        unit: str = "s",
    ):
        if calendar not in _CALENDARS:
            raise ValueError("Unsupported model calendar; conversion must be explicit.")
        if unit not in _UNITS:
            raise ValueError(
                "Time units must be s, min, h or d; years/months are not fixed durations."
            )
        calendar = _CALENDARS[calendar]
        parts = _parse(epoch, calendar)
        canonical_epoch = _format(parts)
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(self, "epoch", canonical_epoch)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "_epoch_ticks", _ticks(parts, calendar))
        object.__setattr__(
            self,
            "time_id",
            canonical_fingerprint(
                {
                    "kind": "geophysical-time",
                    "calendar": calendar,
                    "epoch": canonical_epoch,
                    "unit": unit,
                    "time_scale": "continuous-model-clock-no-leap-seconds",
                }
            ),
        )

    @property
    def seconds_per_unit(self) -> float:
        return _UNITS[self.unit]

    def encode(self, dates: Sequence[str]) -> np.ndarray:
        if isinstance(dates, str):
            raise TypeError("encode expects a sequence of dates, not one string.")
        scale = 1_000_000 * self.seconds_per_unit
        offsets = [
            _ticks(_parse(text, self.calendar), self.calendar) - self._epoch_ticks
            for text in dates
        ]
        values = np.asarray([offset / scale for offset in offsets], dtype=np.float64)
        if any(
            round(float(value) * scale) != offset
            for value, offset in zip(values, offsets, strict=True)
        ):
            raise ValueError(
                "Dates are not representable at microsecond precision with this epoch and unit."
            )
        return values

    def decode(self, values: Any) -> tuple[str, ...]:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(
                "decode requires a finite one-dimensional numerical time array."
            )
        scale = self.seconds_per_unit * 1_000_000
        return tuple(
            _format(
                _from_ticks(
                    self._epoch_ticks + round(float(value) * scale), self.calendar
                )
            )
            for value in values
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "calendar": self.calendar,
            "epoch": self.epoch,
            "unit": self.unit,
            "time_id": self.time_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeophysicalTimeSpec:
        if set(payload) != {"calendar", "epoch", "unit", "time_id"}:
            raise ValueError("Time descriptor must use exactly the canonical fields.")
        spec = cls(payload["calendar"], payload["epoch"], payload["unit"])
        if spec.time_id != payload["time_id"]:
            raise ValueError("Time descriptor fingerprint does not match its content.")
        return spec


@dataclass(frozen=True, slots=True, init=False)
class TemporalSupport:
    """Physical support of samples, independent of their clock and storage axes."""

    kind: str
    bounds: tuple[tuple[float, float], ...] | None
    position: str
    support_id: str = field(init=False)

    def __init__(
        self, kind: str = "instantaneous", bounds: Any = None, position: str = "point"
    ):
        if kind not in ("instantaneous", "mean", "accumulation", "minimum", "maximum"):
            raise ValueError("Unknown temporal support kind.")
        if position not in ("point", "start", "midpoint", "end"):
            raise ValueError("Unknown temporal sample position.")
        if kind != "instantaneous" and bounds is None:
            raise ValueError("Interval quantities require explicit bounds.")
        if bounds is None:
            resolved = None
        else:
            values = np.asarray(bounds, dtype=float)
            if (
                values.ndim != 2
                or values.shape[1] != 2
                or values.shape[0] == 0
                or not np.all(np.isfinite(values))
                or np.any(values[:, 1] <= values[:, 0])
            ):
                raise ValueError(
                    "Temporal bounds must be nonempty finite increasing pairs."
                )
            resolved = tuple((float(a), float(b)) for a, b in values)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "bounds", resolved)
        object.__setattr__(self, "position", position)
        object.__setattr__(
            self,
            "support_id",
            canonical_fingerprint(
                {
                    "kind": kind,
                    "bounds": resolved,
                    "position": position,
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "bounds": self.bounds,
            "position": self.position,
            "support_id": self.support_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TemporalSupport:
        if set(payload) != {"kind", "bounds", "position", "support_id"}:
            raise ValueError("Temporal descriptor must use exactly the canonical fields.")
        support = cls(payload["kind"], payload["bounds"], payload["position"])
        if support.support_id != payload["support_id"]:
            raise ValueError(
                "Temporal descriptor fingerprint does not match its content."
            )
        return support


__all__ = ["GeophysicalTimeSpec", "TemporalSupport"]
