# Copyright © 2026 PHYDRA, Inc. All rights reserved.
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications._energy_series import EnergySeries
from phydrax.applications.building_energy import (
    compare_energyplus_reference,
    energyplus_reference_weather,
    EnergyPlusVariable,
    parse_energyplus_csv,
    parse_epw,
)
from phydrax.series import SampledSeries
from phydrax.units import KELVIN


def test_epw_subhour_tmy_leap_and_interval_endings():
    rows = energyplus_reference_weather().decode().splitlines()
    rows[0] = rows[0].replace("Synthetic", "TMY3")
    rows[4] = "HOLIDAYS/DAYLIGHT SAVINGS,Yes,0,0,0"
    rows[7] = "DATA PERIODS,1,2,Data,Monday,2/28,2/29"
    first = rows[8].split(",")
    first[:5] = ["1988", "2", "28", "24", "60"]
    second = first.copy()
    second[:5] = ["1992", "2", "29", "1", "30"]
    weather = parse_epw("\n".join(rows[:8] + [",".join(first), ",".join(second)]))
    assert weather.typical_year and weather.calendar_year == 2000
    assert weather.records_per_hour == 2 and weather.leap_year_observed
    assert weather.record_calendar[:, 0].tolist() == [1988, 1992]
    radiation = weather.quantity("global_horizontal_energy")
    np.testing.assert_allclose(radiation.samples.support.coordinates, [0, 1800, 3600])
    assert radiation.origin == "2000-02-28T23:30:00+00:00"
    with pytest.raises(ValueError, match="February 29"):
        parse_epw(
            "\n".join(rows[:8] + [",".join(first), ",".join(second)]), calendar_year=2001
        )


def test_observed_year_rollover_is_not_relabelled_as_typical_year():
    rows = energyplus_reference_weather().decode().splitlines()
    rows[7] = "DATA PERIODS,1,1,Data,Monday,12/31,1/1"
    first = rows[8].split(",")
    first[:5] = ["2001", "12", "31", "24", "60"]
    second = first.copy()
    second[:5] = ["2002", "1", "1", "1", "60"]
    weather = parse_epw("\n".join(rows[:8] + [",".join(first), ",".join(second)]))
    assert not weather.typical_year
    temperature = weather.quantity("dry_bulb_temperature")
    assert temperature.time_basis == "absolute"
    assert temperature.origin == "2001-12-31T23:00:00+00:00"


def test_energyplus_reference_temperature_conversion_balance_and_missing_failure():
    column = "ZONE:Zone Mean Air Temperature [C](Hourly)"
    csv = "Date/Time," + column + "\n 01/01  01:00:00,20\n 01/01  02:00:00,20\n"
    variable = EnergyPlusVariable(column, "zone_temperature", KELVIN, offset=273.15)
    reference = parse_energyplus_csv(
        csv, (variable,), year=2001, standard_utc_offset=0, interval_seconds=3600
    )[0]
    np.testing.assert_allclose(reference.samples.values, [293.15, 293.15])
    prediction = EnergySeries(
        SampledSeries(
            reference.samples.support,
            jnp.array([293.15, 293.15]),
            alignment="edge",
            series_id="native",
        ),
        quantity="zone_temperature",
        unit=KELVIN,
        meaning="interval_average",
        time_basis="absolute",
        origin=reference.origin,
        timezone=reference.timezone,
    )
    comparison = compare_energyplus_reference(
        prediction, reference, absolute_tolerance=0.05
    )
    assert bool(comparison.passed) and float(comparison.maximum_absolute_error) == 0
    missing = parse_energyplus_csv(
        csv.replace("02:00:00,20", "02:00:00,"),
        (variable,),
        year=2001,
        standard_utc_offset=0,
        interval_seconds=3600,
    )[0]
    assert not bool(
        compare_energyplus_reference(prediction, missing, absolute_tolerance=1000).passed
    )
    wrong_clock = parse_energyplus_csv(
        csv, (variable,), year=2002, standard_utc_offset=0, interval_seconds=3600
    )[0]
    with pytest.raises(ValueError, match="clock"):
        compare_energyplus_reference(prediction, wrong_clock, absolute_tolerance=0.05)
