# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Real EnergyPlus input/result interchange with explicit matched-model scope."""

from __future__ import annotations

import csv
import hashlib
import io
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...series import SampledSeries, SeriesSupport
from ...units import UnitDefinition
from .._energy_series import EnergySeries


class EnergyPlusVariable(StrictModule):
    column: str = eqx.field(static=True)
    quantity: str = eqx.field(static=True)
    unit: UnitDefinition = eqx.field(static=True)
    meaning: str = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    offset: float = eqx.field(static=True)

    def __init__(
        self,
        column: str,
        quantity: str,
        unit: UnitDefinition,
        *,
        meaning: str = "interval_average",
        scale: float = 1.0,
        offset: float = 0.0,
    ):
        if (
            not column
            or not quantity
            or not np.isfinite(scale)
            or scale == 0
            or not np.isfinite(offset)
        ):
            raise ValueError(
                "EnergyPlus variable conversion must be explicit and finite."
            )
        self.column, self.quantity, self.unit, self.meaning = (
            column,
            quantity,
            unit,
            meaning,
        )
        self.scale, self.offset = float(scale), float(offset)


class EnergyPlusReference(StrictModule):
    """Caller-owned licensed IDF or epJSON; bytes are never differentiated."""

    model: bytes = eqx.field(static=True)
    model_format: str = eqx.field(static=True)
    provenance: tuple[str, ...] = eqx.field(static=True)
    content_sha256: str = eqx.field(static=True)

    def __init__(
        self, model: bytes, *, model_format: str = "idf", provenance: Sequence[str]
    ):
        if not model or model_format not in ("idf", "epjson") or not provenance:
            raise ValueError(
                "Reference requires model bytes, supported format, and provenance."
            )
        self.model, self.model_format, self.provenance = (
            model,
            model_format,
            tuple(provenance),
        )
        self.content_sha256 = hashlib.sha256(model).hexdigest()

    def run(self, executable, weather: bytes, *, timeout: float = 120):
        from ...interchange.energy_runtime import run_energyplus

        result = run_energyplus(
            executable,
            self.model,
            weather,
            model_format=self.model_format,
            timeout=timeout,
        )
        result.require_success()
        errors = result.output("eplusout.err").decode(errors="replace")
        if "**  Fatal  **" in errors or "EnergyPlus Terminated" in errors:
            raise ValueError("EnergyPlus reported a fatal model/run failure: " + errors)
        return result


def parse_energyplus_csv(
    text: str,
    variables: Sequence[EnergyPlusVariable],
    *,
    year: int,
    standard_utc_offset: float,
    interval_seconds: float,
    asset_id: str = "energyplus-reference",
) -> tuple[EnergySeries, ...]:
    """Import explicitly selected CSV columns at EnergyPlus interval-ending times.

    Reporting frequency must be homogeneous and supplied by the caller. Missing
    cells stay invalid. Celsius offsets are explicit variable metadata, never a
    generic multiplicative unit conversion. No header substring guessing occurs.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("EnergyPlus CSV has no header.")
    fieldnames = tuple(name.strip() for name in reader.fieldnames)
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError(
            "EnergyPlus CSV headers must be unique after boundary whitespace normalization."
        )
    reader.fieldnames = list(fieldnames)
    date_column = "Date/Time" if "Date/Time" in fieldnames else None
    if date_column is None or any(v.column not in fieldnames for v in variables):
        raise ValueError(
            "EnergyPlus CSV lacks Date/Time or an exactly requested variable."
        )
    rows = list(reader)
    if not rows or not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError(
            "EnergyPlus CSV must contain records with a positive reporting interval."
        )
    tz = timezone(timedelta(hours=float(standard_utc_offset)))
    ends, year_offset, previous_md = [], 0, None
    for row in rows:
        date, time = row[date_column].strip().split()
        month, day = map(int, date.split("/"))
        hour, minute, second = map(int, time.split(":"))
        md = (month, day)
        if previous_md is not None and md < previous_md:
            if previous_md != (12, 31) or md != (1, 1):
                raise ValueError("EnergyPlus CSV date sequence is not monotonic.")
            year_offset += 1
        if (
            not 0 <= hour <= 24
            or not 0 <= minute < 60
            or not 0 <= second < 60
            or (hour == 24 and (minute or second))
        ):
            raise ValueError("EnergyPlus timestamp is invalid.")
        end = datetime(year + year_offset, month, day, tzinfo=tz) + timedelta(
            hours=hour, minutes=minute, seconds=second
        )
        if ends and abs((end - ends[-1]).total_seconds() - interval_seconds) > 1e-6:
            raise ValueError(
                "EnergyPlus CSV has mixed reporting frequencies or missing intervals."
            )
        ends.append(end)
        previous_md = md
    origin = ends[0] - timedelta(seconds=interval_seconds)
    coordinates = np.arange(len(rows) + 1) * interval_seconds
    digest = hashlib.sha256(text.encode()).hexdigest()
    output = []
    for variable in variables:
        raw = np.asarray(
            [
                float(row[variable.column]) if row[variable.column].strip() else np.nan
                for row in rows
            ]
        )
        valid = np.isfinite(raw)
        values = np.where(valid, raw * variable.scale + variable.offset, 0)
        edge = variable.meaning in ("interval_average", "interval_integral")
        support = SeriesSupport(
            coordinates if edge else coordinates[1:],
            coordinate_name="time",
            coordinate_id="energyplus-standard-time",
        )
        samples = SampledSeries(
            support,
            values,
            alignment="edge" if edge else "node",
            value_valid=valid,
            series_id=asset_id + ":" + variable.quantity,
        )
        output.append(
            EnergySeries(
                samples,
                quantity=variable.quantity,
                unit=variable.unit,
                meaning=variable.meaning,
                time_basis="absolute",
                origin=origin.isoformat(),
                timezone=tz.tzname(None),
                asset_id=asset_id,
                provenance=(
                    "energyplus-csv:sha256:" + digest,
                    "foreign-execution:nondifferentiable",
                ),
            )
        )
    return tuple(output)


class EnergyPlusComparison(StrictModule):
    maximum_absolute_error: Array
    rms_error: Array
    complete: Array
    passed: Array
    absolute_tolerance: float = eqx.field(static=True)


def compare_energyplus_reference(
    prediction: EnergySeries, reference: EnergySeries, *, absolute_tolerance: float
) -> EnergyPlusComparison:
    """Compare matching physical quantities/clocks; do not silently resample or align."""
    from ...units import conversion_factor

    if (
        prediction.quantity != reference.quantity
        or prediction.meaning != reference.meaning
        or prediction.time_basis != reference.time_basis
        or prediction.origin != reference.origin
        or prediction.timezone != reference.timezone
    ):
        raise ValueError(
            "Reference comparison requires identical quantity/meaning/clock semantics."
        )
    if absolute_tolerance < 0 or not np.isfinite(absolute_tolerance):
        raise ValueError("Comparison tolerance must be finite and nonnegative.")
    p, r = prediction.samples, reference.samples
    pt = np.asarray(p.support.coordinates) * float(
        conversion_factor(prediction.time_unit, reference.time_unit)
    )
    if pt.shape != r.support.coordinates.shape or not np.allclose(
        pt, np.asarray(r.support.coordinates), rtol=0, atol=1e-6
    ):
        raise ValueError(
            "Reference comparison requires explicitly aligned time coordinates."
        )
    if p.values.shape != r.values.shape:
        raise ValueError("Reference/prediction sample shapes differ.")
    error = (
        p.values * float(conversion_factor(prediction.unit, reference.unit)) - r.values
    )
    complete = jnp.all(p.sample_valid & r.sample_valid)
    maximum = jnp.max(jnp.abs(error))
    rms = jnp.sqrt(jnp.mean(error**2))
    return EnergyPlusComparison(
        maximum,
        rms,
        complete,
        complete & (maximum <= absolute_tolerance),
        float(absolute_tolerance),
    )


def energyplus_adiabatic_reference(
    *,
    version: str = "26.1",
    internal_heat: float = 100.0,
    setpoint_kelvin: float = 293.15,
) -> EnergyPlusReference:
    """One 60m³ ideal-load zone with adiabatic massless walls and fixed convective heat.

    Steady matched native model: C dT/dt = Qinternal + Qhvac, G=0,
    T=setpoint, sensible cooling=Qinternal. This qualifies signs/units/balance,
    not general transient equivalence of EnergyPlus and reduced RC models.
    """
    if (
        internal_heat <= 0
        or not np.isfinite(internal_heat)
        or not 283.15 <= setpoint_kelvin <= 303.15
    ):
        raise ValueError(
            "Reference requires positive finite gains and a supported comfort setpoint."
        )
    temperature = setpoint_kelvin - 273.15
    # Vertex ordering is counterclockwise viewed from outside the closed box.
    faces = (
        ("floor", "Floor", ((0, 0, 0), (0, 4, 0), (5, 4, 0), (5, 0, 0))),
        ("roof", "Roof", ((0, 0, 3), (5, 0, 3), (5, 4, 3), (0, 4, 3))),
        ("south", "Wall", ((0, 0, 0), (5, 0, 0), (5, 0, 3), (0, 0, 3))),
        ("east", "Wall", ((5, 0, 0), (5, 4, 0), (5, 4, 3), (5, 0, 3))),
        ("north", "Wall", ((5, 4, 0), (0, 4, 0), (0, 4, 3), (5, 4, 3))),
        ("west", "Wall", ((0, 4, 0), (0, 0, 0), (0, 0, 3), (0, 4, 3))),
    )
    text = f"""Version,{version};
SimulationControl,No,No,No,No,Yes;
Building,AdiabaticReference,0,Suburbs,0.04,0.4,FullExterior,25,6;
Timestep,4;
RunPeriod,Reference,1,1,2001,1,1,2001,Monday,No,No,No,Yes,Yes;
GlobalGeometryRules,UpperLeftCorner,CounterClockWise,World;
Material:NoMass,Massless,Rough,2.0,0.9,0.7,0.7;
Construction,WallConstruction,Massless;
Zone,Zone,0,0,0,0,1,1,3,60,20;
ScheduleTypeLimits,AnyNumber;
Schedule:Constant,Always,AnyNumber,1;
Schedule:Constant,ControlType,AnyNumber,4;
Schedule:Constant,Setpoint,AnyNumber,{temperature};
ThermostatSetpoint:DualSetpoint,Thermostat,Setpoint,Setpoint;
ZoneControl:Thermostat,ZoneThermostat,Zone,ControlType,ThermostatSetpoint:DualSetpoint,Thermostat;
ElectricEquipment,Gains,Zone,Always,EquipmentLevel,{internal_heat},,,0,0,0;
ZoneHVAC:EquipmentConnections,Zone,Equipment,SupplyNode,,ZoneAirNode,ReturnNode;
ZoneHVAC:EquipmentList,Equipment,SequentialLoad,ZoneHVAC:IdealLoadsAirSystem,Ideal,1,1;
ZoneHVAC:IdealLoadsAirSystem,Ideal,Always,SupplyNode,,,50,13,0.0156,0.0077,NoLimit,,,NoLimit,,,,,None,,None;
Output:Variable,Zone,Zone Mean Air Temperature,Hourly;
Output:Variable,Ideal,Zone Ideal Loads Zone Sensible Cooling Rate,Hourly;
Output:Variable,Zone,Zone Electric Equipment Convective Heating Rate,Hourly;
Output:Variable,*,Zone Air Heat Balance Air Energy Storage Rate,Hourly;
OutputControl:Table:Style,Comma;
"""
    for name, kind, vertices in faces:
        coordinates = ",".join(str(value) for vertex in vertices for value in vertex)
        text += (
            f"BuildingSurface:Detailed,{name},{kind},WallConstruction,Zone,,"
            f"Adiabatic,,NoSun,NoWind,0.5,4,{coordinates};\n"
        )
    return EnergyPlusReference(
        text.encode(),
        provenance=(
            "phydrax-authored-analytic-adiabatic-reference",
            "six-massless-adiabatic-surfaces;no-ventilation;all-gains-convective;ideal-setpoint-control",
        ),
    )


def energyplus_reference_weather() -> bytes:
    """Synthetic complete January 1 weather for the weather-independent adiabatic case."""
    headers = (
        "LOCATION,Analytic,NA,NA,Synthetic,000000,0,0,0,0",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,Authored synthetic weather for adiabatic energy-balance qualification",
        "COMMENTS 2,Not measured data; all surfaces adiabatic and no ventilation",
        "DATA PERIODS,1,1,Data,Monday,1/1,1/1",
    )
    records = []
    for hour in range(1, 25):
        values = [
            2001,
            1,
            1,
            hour,
            60,
            "?" * 29,
            20,
            10,
            50,
            101325,
            0,
            0,
            300,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            20,
            77777,
            9,
            999999999,
            10,
            0.1,
            0,
            99,
            0.2,
            0,
            0,
        ]
        records.append(",".join(map(str, values)))
    return ("\n".join((*headers, *records)) + "\n").encode()
