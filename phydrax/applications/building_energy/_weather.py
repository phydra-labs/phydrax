# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""EPW import without silently interpolating missing values or changing time bases."""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...series import SampledSeries, SeriesSupport
from ...units import derived_unit, JOULE, KELVIN, METER, ONE, PASCAL, SECOND
from .._energy_series import EnergySeries


class EPWLocation(StrictModule):
    city: str = eqx.field(static=True)
    region: str = eqx.field(static=True)
    country: str = eqx.field(static=True)
    source: str = eqx.field(static=True)
    station: str = eqx.field(static=True)
    latitude: float = eqx.field(static=True)
    longitude: float = eqx.field(static=True)
    standard_utc_offset: float = eqx.field(static=True)
    elevation: float = eqx.field(static=True)


class EPWWeather(StrictModule):
    location: EPWLocation
    series: tuple[EnergySeries, ...]
    record_calendar: Array
    missing: Array
    uncertainty_flags: tuple[str, ...] = eqx.field(static=True)
    headers: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    records_per_hour: int = eqx.field(static=True)
    typical_year: bool = eqx.field(static=True)
    leap_year_observed: bool = eqx.field(static=True)
    calendar_year: int | None = eqx.field(static=True)
    content_sha256: str = eqx.field(static=True)

    def quantity(self, name: str) -> EnergySeries:
        for item in self.series:
            if item.quantity == name:
                return item
        raise KeyError(name)


def parse_epw(
    text: str,
    *,
    typical_year: bool | None = None,
    calendar_year: int | None = None,
    asset_id: str = "weather",
) -> EPWWeather:
    """Read all eight EPW headers and standard records.

    EPW clock is local *standard* time. Hour 1/minute 60 ends at 01:00;
    hour 24/minute 60 ends at next midnight. Radiation is the preceding
    interval's Wh/m², converted to J/m². Dry-bulb/dew-point observations become
    Kelvin. TMY dates are placed on one explicit representative calendar while
    original source years and quality flags remain available. Discontinuous
    records are rejected rather than fabricated or relabelled.
    """
    rows = [
        tuple(cell.strip() for cell in row)
        for row in csv.reader(io.StringIO(text))
        if row
    ]
    expected = (
        "LOCATION",
        "DESIGN CONDITIONS",
        "TYPICAL/EXTREME PERIODS",
        "GROUND TEMPERATURES",
        "HOLIDAYS/DAYLIGHT SAVINGS",
        "COMMENTS 1",
        "COMMENTS 2",
        "DATA PERIODS",
    )
    if len(rows) < 9 or tuple(row[0].upper() for row in rows[:8]) != expected:
        raise ValueError(
            "EPW requires the eight standard headers followed by weather records."
        )
    loc = rows[0]
    if len(loc) < 10:
        raise ValueError("EPW LOCATION header is incomplete.")
    lat, lon, offset, elevation = map(float, loc[6:10])
    if not all(np.isfinite((lat, lon, offset, elevation))) or not (
        -90 <= lat <= 90 and -180 <= lon <= 180 and -14 <= offset <= 14
    ):
        raise ValueError("EPW location or standard time offset is invalid.")
    location = EPWLocation(*loc[1:6], lat, lon, offset, elevation)
    periods = rows[7]
    if len(periods) < 7 or int(periods[1]) != 1:
        raise ValueError(
            "EPW import currently requires one declared continuous DATA PERIOD."
        )
    rph = int(periods[2])
    if rph < 1 or 60 % rph:
        raise ValueError("EPW records per hour must divide 60.")
    leap = rows[4][1].lower() == "yes"
    records = rows[8:]
    if any(len(row) < 35 for row in records):
        raise ValueError("EPW weather records require all 35 standard fields.")
    dates = np.asarray(
        [[int(cell) for cell in row[:5]] for row in records], dtype=np.int64
    )
    stitched_years = any(
        left[0] != right[0]
        and not (
            right[0] == left[0] + 1
            and tuple(left[1:3]) == (12, 31)
            and tuple(right[1:3]) == (1, 1)
        )
        for left, right in zip(dates[:-1], dates[1:], strict=True)
    )
    inferred_tmy = "TMY" in location.source.upper() or stitched_years
    tmy = inferred_tmy if typical_year is None else bool(typical_year)
    has_feb29 = bool(np.any((dates[:, 1] == 2) & (dates[:, 2] == 29)))
    if has_feb29 and not leap:
        raise ValueError(
            "EPW contains February 29 but declares leap-year observation disabled."
        )
    if calendar_year is not None and not tmy:
        raise ValueError(
            "calendar_year remapping is only valid for an explicit typical year."
        )
    representative = (
        (2000 if has_feb29 else 2001) if calendar_year is None else int(calendar_year)
    )
    if tmy and has_feb29 and not calendar.isleap(representative):
        raise ValueError("Typical-year calendar cannot represent February 29.")
    tz = timezone(timedelta(hours=offset))
    duration = 3600.0 / rph
    ends = []
    rollover = 0
    previous_md = None
    for year, month, day, hour, minute in dates:
        if hour < 1 or hour > 24 or minute < 1 or minute > 60 or minute % (60 // rph):
            raise ValueError(
                "EPW record must specify an allowed interval-ending hour/minute."
            )
        md = (month, day)
        if tmy and previous_md is not None and md < previous_md:
            if previous_md != (12, 31) or md != (1, 1):
                raise ValueError("EPW typical-year dates are not ordered.")
            rollover += 1
        y = representative + rollover if tmy else int(year)
        end = datetime(y, int(month), int(day), tzinfo=tz) + timedelta(
            hours=int(hour) - 1, minutes=int(minute)
        )
        if ends and abs((end - ends[-1]).total_seconds() - duration) > 1e-6:
            raise ValueError(
                "EPW contains duplicate, missing, or nonuniform interval endings."
            )
        ends.append(end)
        previous_md = md
    origin = ends[0] - timedelta(seconds=duration)
    boundaries = np.arange(len(records) + 1, dtype=float) * duration
    edge_support = SeriesSupport(
        boundaries, coordinate_name="time", coordinate_id="epw-standard-time"
    )
    node_support = SeriesSupport(
        boundaries[1:], coordinate_name="time", coordinate_id="epw-standard-time"
    )
    solar_unit = derived_unit("J/m²", ((JOULE, 1), (METER, -2)))
    speed_unit = derived_unit("m/s", ((METER, 1), (SECOND, -1)))
    # Column, sentinel, quantity, unit, meaning, multiplicative scale, additive offset.
    fields = (
        (6, 99.9, "dry_bulb_temperature", KELVIN, "instantaneous", 1, 273.15),
        (7, 99.9, "dew_point_temperature", KELVIN, "instantaneous", 1, 273.15),
        (8, 999, "relative_humidity", ONE, "instantaneous", 0.01, 0),
        (9, 999999, "atmospheric_pressure", PASCAL, "instantaneous", 1, 0),
        (
            12,
            9999,
            "horizontal_infrared_energy",
            solar_unit,
            "interval_integral",
            3600,
            0,
        ),
        (13, 9999, "global_horizontal_energy", solar_unit, "interval_integral", 3600, 0),
        (14, 9999, "direct_normal_energy", solar_unit, "interval_integral", 3600, 0),
        (15, 9999, "diffuse_horizontal_energy", solar_unit, "interval_integral", 3600, 0),
        (21, 999, "wind_speed", speed_unit, "instantaneous", 1, 0),
    )
    digest = hashlib.sha256(text.encode()).hexdigest()
    quantities, masks = [], []
    for column, sentinel, name, unit, meaning, scale, shift in fields:
        raw = np.asarray([float(row[column]) for row in records])
        valid = np.isfinite(raw) & (raw != sentinel)
        if column in (8, 9, 12, 13, 14, 15, 21):
            valid &= raw >= 0
        if column == 8:
            valid &= raw <= 100
        masks.append(~valid)
        values = np.where(valid, raw * scale + shift, 0)
        edge = meaning == "interval_integral"
        sampled = SampledSeries(
            edge_support if edge else node_support,
            values,
            alignment="edge" if edge else "node",
            value_valid=valid,
            series_id=f"{asset_id}:{name}",
        )
        quantities.append(
            EnergySeries(
                sampled,
                quantity=name,
                unit=unit,
                meaning=meaning,
                time_basis="cyclic" if tmy else "absolute",
                origin=origin.isoformat(),
                timezone=tz.tzname(None),
                asset_id=asset_id,
                provenance=(
                    f"epw:sha256:{digest}",
                    location.source,
                    "typical-year" if tmy else "observed-calendar",
                ),
            )
        )
    return EPWWeather(
        location,
        tuple(quantities),
        jnp.asarray(dates),
        jnp.asarray(np.stack(masks, axis=1)),
        tuple(row[5] for row in records),
        tuple(rows[:8]),
        rph,
        tmy,
        leap,
        representative if tmy else None,
        digest,
    )


def read_epw(path: str | Path, **kwargs) -> EPWWeather:
    return parse_epw(Path(path).read_text(encoding="utf-8-sig"), **kwargs)
